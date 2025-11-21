from laser.generic.newutils import TimingStats as ts  # noqa: I001

import numba as nb
import numpy as np

from .newutils import validate
from .shared import State
from .shared import sample_dobs


class BirthsByCBR:
    def __init__(self, model, birthrates, pyramid, validating=False):
        self.model = model
        self.birthrates = birthrates
        self.pyramid = pyramid
        self.validating = validating

        self.model.people.add_property("dob", dtype=np.int16)
        self.model.nodes.add_vector_property("births", model.params.nticks + 1, dtype=np.int32)

        dobs = self.model.people.dob
        sample_dobs(dobs, self.pyramid, tick=0)

        return

    def prevalidate_step(self, tick: int) -> None:
        self.prv_count = self.model.people.count

        return

    def postvalidate_step(self, tick: int) -> None:
        nbirths = self.model.nodes.births[tick].sum()
        assert self.model.people.count == self.prv_count + nbirths, "Population count mismatch after births"

        istart = self.prv_count
        iend = self.model.people.count
        birth_counts = np.bincount(self.model.people.nodeid[istart:iend], minlength=self.model.nodes.count)
        assert np.all(birth_counts == self.model.nodes.births[tick]), "Birth counts by patch mismatch"
        assert np.all(self.model.people.state[istart:iend] == State.SUSCEPTIBLE.value), "Newborns should be susceptible"

        return

    @validate(prevalidate_step, postvalidate_step)
    def step(self, tick):
        # Get total population size at time t+1. Use "tomorrow's" population which accounts for any mortality.
        N = np.zeros(len(self.model.scenario), dtype=np.int32)
        for state in self.model.states:
            if (pop := getattr(self.model.nodes, state, None)) is not None:
                N += pop[tick + 1]
        rates = np.power(1.0 + self.birthrates[tick] / 1000, 1.0 / 365) - 1.0
        births = np.round(np.random.poisson(rates * N)).astype(np.int32)

        tbirths = births.sum()
        if tbirths > 0:
            istart, iend = self.model.people.add(tbirths)
            self.model.people.nodeid[istart:iend] = np.repeat(np.arange(self.model.nodes.count, dtype=np.uint16), births)
            # State.SUSCEPTIBLE.value is the default
            # self.model.people.state[istart:iend] = State.SUSCEPTIBLE.value

            dobs = self.model.people.dob[istart:iend]
            dobs[:] = tick

            # state(t+1) = state(t) + ∆state(t)
            self.model.nodes.S[tick + 1] += births
            # Record today's ∆
            self.model.nodes.births[tick] = births

        for component in self.model.components:
            if hasattr(component, "on_birth") and callable(component.on_birth):
                with ts.start(f"{component.__class__.__name__}.on_birth()"):
                    component.on_birth(istart, iend, tick)

        return


class MortalityByCDR:
    def __init__(self, model, mortalityrates, additional_mappings=None, validating=False):
        self.model = model
        self.mortalityrates = mortalityrates
        self.validating = validating

        self.mappings = [
            (State.SUSCEPTIBLE.value, "S"),
            (State.EXPOSED.value, "E"),
            (State.INFECTIOUS.value, "I"),
            (State.RECOVERED.value, "R"),
        ]

        if additional_mappings is not None:
            self.mappings.extend(additional_mappings)

        self.mapping = np.full(np.max([value for value, _name in self.mappings]) + 1, -1, dtype=np.int32)
        for index, (value, _name) in enumerate(self.mappings):
            self.mapping[value] = index

        model.nodes.add_vector_property("deaths", model.params.nticks + 1, dtype=np.int32)

        return

    def prevalidate_step(self, tick: int) -> None:
        self._deaths_prv = self.model.people.state == State.DECEASED.value

        return

    def postvalidate_step(self, tick: int) -> None:
        self._deaths_now = self.model.people.state == State.DECEASED.value

        # Check that diff between _deaths_now and _deaths_prv matches recorded deaths
        # Use np.bincount and compare with self.model.nodes.deaths[tick]
        prv = np.bincount(self.model.people.nodeid[self._deaths_prv], minlength=self.model.nodes.count)
        now = np.bincount(self.model.people.nodeid[self._deaths_now], minlength=self.model.nodes.count)
        diff = now - prv
        assert np.all(diff == self.model.nodes.deaths[tick]), "Death counts by patch mismatch after mortality"

        return

    @staticmethod
    @nb.njit(parallel=True, cache=True)
    def nb_process_mortality(
        states: np.ndarray,
        nodeids: np.ndarray,
        p_mortality: np.ndarray,
        newly_deceased: np.ndarray,
        mapping: np.ndarray,
        deceased_by_state: np.ndarray,
    ) -> None:
        for i in nb.prange(len(states)):
            if states[i] == State.DECEASED.value:
                continue
            draw = np.random.rand()
            nid = nodeids[i]
            if draw < p_mortality[nid]:
                index = mapping[states[i]]
                states[i] = State.DECEASED.value
                newly_deceased[nb.get_thread_id(), nid] += 1
                if index >= 0:
                    deceased_by_state[nb.get_thread_id(), index, nid] += 1

        return

    @validate(prevalidate_step, postvalidate_step)
    def step(self, tick):
        # Convert CDR per 1000 per year to daily mortality rate and probability
        annual_survival_rate = 1.0 - self.mortalityrates[tick] / 1000.0
        daily_survival_rate = np.power(annual_survival_rate, 1.0 / 365.0)
        daily_mortality_rate = 1.0 - daily_survival_rate
        daily_p_mortality = -np.expm1(-daily_mortality_rate)

        deceased_by_state = np.zeros((nb.get_num_threads(), len(self.mapping), self.model.nodes.count), dtype=np.int32)
        newly_deceased = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        self.nb_process_mortality(
            self.model.people.state,
            self.model.people.nodeid,
            daily_p_mortality,
            newly_deceased,
            self.mapping,
            deceased_by_state,
        )
        total_deceased = np.sum(newly_deceased, axis=0)
        deceased_by_state = np.sum(deceased_by_state, axis=0)

        self.model.nodes.deaths[tick] = total_deceased

        # Get State.NNN.value and state name from mappings
        for value, state_name in self.mappings:
            # Get index in deceased_by_state for State.NNN.value
            index = self.mapping[value]
            # If the state exists in nodes, decrement by deceased count
            if (prop := getattr(self.model.nodes, state_name, None)) is not None:
                prop[tick + 1] -= deceased_by_state[index]

        return


class MortalityBySurvival:
    def __init__(self, model, survival_probs, validating=False):
        self.model = model
        self.survival_probs = survival_probs
        self.validating = validating

        return

    def prevalidate_step(self, tick: int) -> None:
        self.prv_count = self.model.people.count

        return

    def postvalidate_step(self, tick: int) -> None:
        ndeaths = self.prv_count - self.model.people.count
        recorded_deaths = 0
        for state in self.model.states:
            if (pop := getattr(self.model.nodes, state, None)) is not None:
                recorded_deaths += pop[tick].sum() - pop[tick + 1].sum()
        assert ndeaths == recorded_deaths, "Death count mismatch after mortality"

        return

    @validate(prevalidate_step, postvalidate_step)
    def step(self, tick):
        random_values = np.random.random(size=self.model.people.count)
        survival_probs = self.survival_probs[tick]
        death_mask = random_values >= survival_probs
        ndeaths = np.sum(death_mask)

        if ndeaths > 0:
            deceased_nodeids, deceased_states = (
                self.model.people.nodeid[death_mask],
                self.model.people.state[death_mask],
            )

            self.model.people.remove(death_mask)

            for nodeid, state in zip(deceased_nodeids, deceased_states):
                state_name = State(state).name
                if (prop := getattr(self.model.nodes, state_name, None)) is not None:
                    prop[tick + 1, nodeid] -= 1

        return
