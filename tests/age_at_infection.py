"""
Prompt #1:
Let's create an SEIR model with a custom Transmission component, based on SEIR.Transmission, which add a "doi" property to the people in the model, implements a new Numba-compiled function to run transmission including storing the current tick in doi when an agent is infected, and overriding the step() function to call this new Numba function with all the previous properties plus the current tick and the doi property array.
Please implement in age-at-infection.py but you can reference code from test_seir.py and the implementation of TransmissionSE in components.py that the base SEIR model uses for Transmission.
Do not modify code in any other file at this time.

Prompt #2:
Please revisit the implementation of the Numba function and the step function to more closely mirror the implementation in the base class, TransmissionSE in components.py.
Note the force of infection setup, ft, in the base implementation outside the Numba function.
The modified Numba function should merely include setting the date-of-infection, doi, in addition to the base implementation.
Again, all changes should go into age-at-infection.py and no other files.

Prompt #3:
Please import the State enum from SEIR and update the Numba compiled function to use the values of the enums rather than hardcoded integers for the states.

Interlude

Prompt #4:
Our infections are dying out. We need an Importation component which will periodically infect some susceptible agents in each node.
This component should take a value representing that period, e.g., 30 days or ticks, and an array with the number of new infections per node.
On the given day(s), the component should look at the current number of susceptible agents in each node, the target number of infections, and probabilistically infect that number of susceptible agents in each node.
Like the Transmission component, the step() function of this Importation component will have some NumPy code to calculate the per node probability of infection for susceptible agents and a Numba compiled function to process all the agents in parallel.
"""

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from laser.core.demographics import AliasedDistribution
from laser.core.demographics import KaplanMeierEstimator

from laser.generic import SEIR
from laser.generic import Model
from laser.generic.newutils import ValuesMap
from laser.generic.newutils import validate
from utils import stdgrid

State = SEIR.State


class TransmissionWithDOI(SEIR.Transmission):
    def __init__(self, model, expdurdist, expdurmin=1):
        super().__init__(model, expdurdist, expdurmin)
        # Add 'doi' property to people (default 0, dtype=int32)
        self.model.people.add_scalar_property("doi", dtype=np.int16)
        # Optionally initialize to -1 to indicate never infected
        self.model.people.doi[:] = -1

    def prevalidate_step(self, tick: int) -> None:
        self.prv_state = self.model.people.state.copy()
        self.prv_snext = self.model.nodes.S[tick + 1].copy()
        self.prv_enext = self.model.nodes.E[tick + 1].copy()

        return

    def postvalidate_step(self, tick: int) -> None:
        # Everyone who has different state should a) have been SUSCEPTIBLE before and b) be EXPOSED now
        changed = np.nonzero(self.model.people.state != self.prv_state)[0]
        assert np.all(self.prv_state[changed] == State.SUSCEPTIBLE.value), (
            "Only susceptible individuals should change state in transmission."
        )
        assert np.all(self.model.people.state[changed] == State.EXPOSED.value), "Only newly infected individuals should now be exposed."
        # Everyone who is newly exposed should have etimer > 0
        assert np.all(self.model.people.etimer[changed] > 0), (
            f"Newly exposed individuals should have etimer > 0 ({self.model.people.etimer[changed].min()=})"
        )
        # everyone who is newly exposed should have doi == tick
        assert np.all(self.model.people.doi[changed] == tick), "Newly exposed individuals should have doi equal to the current tick."

        # S[tick+1] - S'[tick+1] should equal number of newly exposed
        exposed_by_node = np.bincount(self.model.people.nodeid[changed], minlength=self.model.nodes.count)
        assert np.all((self.prv_snext - self.model.nodes.S[tick + 1]) == exposed_by_node), (
            "Number of newly exposed individuals should match the change in susceptible individuals."
        )
        # E'[tick+1] - E[tick+1] should equal number of newly exposed
        assert np.all((self.model.nodes.E[tick + 1] - self.prv_enext) == exposed_by_node), (
            "Number of newly exposed individuals should match the change in exposed individuals."
        )
        # incidence[tick] should equal number of newly exposed
        assert np.all(self.model.nodes.newly_infected[tick] == exposed_by_node), (
            "Incidence should match the number of newly exposed individuals."
        )

        return

    @staticmethod
    @nb.njit(nogil=True, parallel=True, cache=True)
    def nb_transmission_step(states, nodeids, ft, exp_by_node, etimers, expdurdist, expdurmin, tick, doi):
        for i in nb.prange(len(states)):
            if states[i] == State.SUSCEPTIBLE.value:
                draw = np.random.rand()
                nid = nodeids[i]
                if draw < ft[nid]:
                    states[i] = State.EXPOSED.value
                    etimers[i] = np.maximum(np.round(expdurdist(tick, nid)), expdurmin)
                    exp_by_node[nb.get_thread_id(), nid] += 1
                    doi[i] = tick  # Record tick of infection
        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        ft = self.model.nodes.forces[tick]
        N = self.model.nodes.S[tick] + self.model.nodes.E[tick] + (I := self.model.nodes.I[tick])  # noqa: E741
        if hasattr(self.model.nodes, "R"):
            N += self.model.nodes.R[tick]
        ft[:] = self.model.params.beta * I / N
        transfer = ft[:, None] * self.model.network
        ft += transfer.sum(axis=0)
        ft -= transfer.sum(axis=1)
        ft = -np.expm1(-ft)
        exp_by_node = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.uint32)
        self.nb_transmission_step(
            self.model.people.state,
            self.model.people.nodeid,
            ft,
            exp_by_node,
            self.model.people.etimer,
            self.expdurdist,
            self.expdurmin,
            tick,
            self.model.people.doi,
        )
        exp_by_node = exp_by_node.sum(axis=0).astype(self.model.nodes.S.dtype)
        self.model.nodes.S[tick + 1] -= exp_by_node
        self.model.nodes.E[tick + 1] += exp_by_node
        self.model.nodes.newly_infected[tick] = exp_by_node
        return


class Importation:
    def __init__(self, model, period, new_infections, infdurdist, infdurmin=1):
        self.model = model
        self.period = period  # e.g., 30 (days/ticks)
        self.new_infections = np.array(new_infections, dtype=np.int32)
        self.infdurdist = infdurdist
        self.infdurmin = infdurmin

    def prevalidate_step(self, tick: int) -> None:
        self.prv_state = self.model.people.state.copy()
        self.prv_snext = self.model.nodes.S[tick + 1].copy()
        self.prv_inext = self.model.nodes.I[tick + 1].copy()

        return

    def postvalidate_step(self, tick: int) -> None:
        # Everyone who has different state should a) have been SUSCEPTIBLE before and b) be EXPOSED now
        changed = np.nonzero(self.model.people.state != self.prv_state)[0]

        if tick % self.period == 0:
            assert changed.sum() > 0, "There should be new infections on importation days."
        else:
            assert changed.sum() == 0, "There should be no new infections on non-importation days."

        assert np.all(self.prv_state[changed] == State.SUSCEPTIBLE.value), (
            "Only susceptible individuals should change state in transmission."
        )
        assert np.all(self.model.people.state[changed] == State.INFECTIOUS.value), "Newly infected individuals should now be infectious."
        # Everyone who is newly infectious should have itimer > 0
        assert np.all(self.model.people.itimer[changed] > 0), (
            f"Newly infectious individuals should have itimer > 0 ({self.model.people.itimer[changed].min()=})"
        )
        # Importations don't get a doi
        # # everyone who is newly infectious should have doi == tick
        # assert np.all(self.model.people.doi[changed] == tick), "Newly infectious individuals should have doi equal to the current tick."

        # Change in S[tick+1] should equal number of newly infectious
        infectious_by_node = np.bincount(self.model.people.nodeid[changed], minlength=self.model.nodes.count)
        assert np.all((self.prv_snext - self.model.nodes.S[tick + 1]) == infectious_by_node), (
            "Number of newly infectious individuals should match the change in susceptible individuals."
        )
        # Change in I[tick+1] should equal number of newly infectious
        assert np.all((self.model.nodes.I[tick + 1] - self.prv_inext) == infectious_by_node), (
            "Number of newly infectious individuals should match the change in infectious individuals."
        )
        # Don't count importation in incidence
        # # incidence[tick] should equal number of newly infectious
        # assert np.all(self.model.nodes.incidence[tick] == infectious_by_node), (
        #     "Incidence should match the number of newly infectious individuals."
        # )

        return

    @staticmethod
    @nb.njit(nogil=True, parallel=True, cache=True)
    def nb_importation_step(states, probabilities, nodeids, itimers, infdurdist, infdurmin, inf_by_node, tick):
        for i in nb.prange(len(states)):
            if states[i] == State.SUSCEPTIBLE.value:
                nid = nodeids[i]
                if np.random.rand() < probabilities[nid]:
                    states[i] = State.INFECTIOUS.value
                    itimers[i] = np.maximum(np.round(infdurdist(tick, nid)), infdurmin)  # Set the infection timer
                    inf_by_node[nb.get_thread_id(), nid] += 1

        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        # Only act on scheduled ticks
        if tick % self.period != 0:
            return

        # Calculate per-node probability to achieve target infections

        susceptible = self.model.nodes.S[tick]
        non_zero = np.nonzero(susceptible)[0]
        probabilities = np.zeros_like(susceptible, dtype=np.float32)
        probabilities[non_zero] = np.minimum(self.new_infections[non_zero] / susceptible[non_zero], 1.0)
        # TODO - did we actually calculate a rate? Should we map to a probability with -np.expm1(-rate)?

        self.nb_importation_step(
            self.model.people.state,
            probabilities,
            self.model.people.nodeid,
            self.model.people.itimer,
            self.infdurdist,
            self.infdurmin,
            inf_by_node := np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.uint32),
            tick,
        )
        inf_by_node = inf_by_node.sum(axis=0).astype(self.model.nodes.S.dtype)
        self.model.nodes.S[tick + 1] -= inf_by_node
        self.model.nodes.I[tick + 1] += inf_by_node

        return


# Example usage in a test or simulation setup:
if __name__ == "__main__":
    # Build a model as in test_seir.py, but use TransmissionWithDOI
    import laser.core.distributions as dists
    from laser.core import PropertySet

    NTICKS = 3650
    R0 = 10  # measles-ish 1.386
    EXPOSED_DURATION_MEAN = 4.5
    EXPOSED_DURATION_SCALE = 1.0
    INFECTIOUS_DURATION_MEAN = 7.0
    INFECTIOUS_DURATION_SCALE = 2.0

    scenario = stdgrid(5, 5)  # Build scenario as in test_seir.py
    init_susceptible = np.round(scenario.population / R0).astype(np.int32)  # 1/R0 already recovered
    equilibrium_prevalence = 9000 / 12_000_000
    init_infected = np.round(equilibrium_prevalence * scenario.population).astype(np.int32)
    scenario["S"] = init_susceptible
    scenario["E"] = 0
    scenario["I"] = init_infected
    scenario["R"] = scenario.population - init_susceptible - init_infected

    params = PropertySet({"nticks": NTICKS, "beta": R0 / INFECTIOUS_DURATION_MEAN})
    birthrates_map = ValuesMap.from_scalar(35, nsteps=NTICKS, nnodes=len(scenario))

    model = Model(scenario, params, birthrates=birthrates_map.values)
    # model.validating = True

    expdist = dists.normal(loc=EXPOSED_DURATION_MEAN, scale=EXPOSED_DURATION_SCALE)
    infdist = dists.normal(loc=INFECTIOUS_DURATION_MEAN, scale=INFECTIOUS_DURATION_SCALE)

    s = SEIR.Susceptible(model)
    e = SEIR.Exposed(model, expdist, infdist)
    i = SEIR.Infectious(model, infdist)
    r = SEIR.Recovered(model)
    tx = TransmissionWithDOI(model, expdist)
    importation = Importation(model, period=30, new_infections=[5] * model.nodes.count, infdurdist=infdist)

    pyramid = AliasedDistribution(np.full(89, 1_000))
    survival = KaplanMeierEstimator(np.full(89, 1_000).cumsum())
    vitals = SEIR.VitalDynamics(model, birthrates_map.values, pyramid, survival)

    model.components = [s, r, i, e, tx, vitals, importation]

    label = f"SEIR with DOI ({model.people.count:,} agents in {model.nodes.count:,} nodes)"
    model.run(label)
    # After run, model.people.doi contains tick of infection for each agent

    # model.plot()

    # Let's look at people infected in the last year of the simulation, doi >= NTICKS - 365
    recent_infections = (model.people.doi >= (NTICKS - 365)) & (model.people.doi != -1)
    aoi_recent = model.people.doi[recent_infections] - model.people.dob[recent_infections]

    plt.hist(aoi_recent, bins=range(aoi_recent.min(), aoi_recent.max() + 1), alpha=0.7)
    plt.xlabel("Age at Infection (Days)")
    plt.ylabel("Number of Infections")
    plt.title(label)
    plt.tight_layout()
    plt.show()

    print("done.")
