from laser.generic.newutils import TimingStats as ts  # noqa: I001

import numpy as np

from .newutils import validate
from .shared import State
from .shared import sample_dobs


class BirthsByCBR:
    def __init__(self, model, birthrates, pyramid, states=None, validating=False):
        self.model = model
        self.birthrates = birthrates
        self.pyramid = pyramid
        self.states = states if states is not None else ["S", "E", "I", "R"]
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
        for state in self.states:
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
