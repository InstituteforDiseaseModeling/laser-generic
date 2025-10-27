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

import numba as nb
import numpy as np
from laser_core.demographics import AliasedDistribution
from laser_core.demographics import KaplanMeierEstimator

import laser_generic.models.SEIR as SEIR
from laser_generic.newutils import RateMap
from utils import stdgrid

State = SEIR.State


class TransmissionWithDOI(SEIR.Transmission):
    def __init__(self, model, expdurdist, expdurmin=1):
        super().__init__(model, expdurdist, expdurmin)
        # Add 'doi' property to people (default 0, dtype=int32)
        self.model.people.add_scalar_property("doi", dtype=np.int16)
        # Optionally initialize to -1 to indicate never infected
        self.model.people.doi[:] = -1

    @staticmethod
    @nb.njit(nogil=True, parallel=True, cache=True)
    def nb_transmission_step(states, nodeids, ft, exp_by_node, etimers, expdurdist, expdurmin, tick, doi):
        for i in nb.prange(len(states)):
            if states[i] == State.SUSCEPTIBLE.value:
                draw = np.random.rand()
                nid = nodeids[i]
                if draw < ft[nid]:
                    states[i] = State.EXPOSED.value
                    etimers[i] = np.maximum(np.round(expdurdist()), expdurmin)
                    exp_by_node[nb.get_thread_id(), nid] += 1
                    doi[i] = tick  # Record tick of infection
        return

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
        self.model.nodes.incidence[tick] = exp_by_node
        return


class Importation:
    def __init__(self, model, period, new_infections, infdurdist):
        self.model = model
        self.period = period  # e.g., 30 (days/ticks)
        self.new_infections = np.array(new_infections, dtype=np.int32)
        self.infdurdist = infdurdist

    @staticmethod
    @nb.njit(nogil=True, parallel=True, cache=True)
    def nb_importation_step(states, probabilities, nodeids, itimers, infdurdist, inf_by_node):
        for i in nb.prange(len(states)):
            if states[i] == State.SUSCEPTIBLE.value:
                if np.random.rand() < probabilities[nodeids[i]]:
                    states[i] = State.INFECTIOUS.value
                    itimers[i] = infdurdist()
                    inf_by_node[nb.get_thread_id(), nodeids[i]] += 1

        return

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
            inf_by_node := np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.uint32),
        )
        inf_by_node = inf_by_node.sum(axis=0).astype(self.model.nodes.S.dtype)
        self.model.nodes.S[tick + 1] -= inf_by_node
        self.model.nodes.I[tick + 1] += inf_by_node

        return


# Example usage in a test or simulation setup:
if __name__ == "__main__":
    # Build a model as in test_seir.py, but use TransmissionWithDOI
    import laser_core.distributions as dists
    from laser_core import PropertySet

    NTICKS = 3650
    R0 = 10  # measles-ish 1.386
    INFECTIOUS_DURATION_MEAN = 7.0
    EXPOSED_DURATION_SHAPE = 4.5
    EXPOSED_DURATION_SCALE = 1.0

    scenario = stdgrid(5, 5)  # Build scenario as in test_seir.py
    init_susceptible = np.round(scenario.population / R0).astype(np.int32)  # 1/R0 already recovered
    equilibrium_prevalence = 9000 / 12_000_000
    init_infected = np.round(equilibrium_prevalence * scenario.population).astype(np.int32)
    scenario["S"] = init_susceptible
    scenario["E"] = 0
    scenario["I"] = init_infected
    scenario["R"] = scenario.population - init_susceptible - init_infected
    params = PropertySet({"nticks": NTICKS, "beta": R0 / INFECTIOUS_DURATION_MEAN})
    birthrates_map = RateMap.from_scalar(35, nsteps=NTICKS, nnodes=len(scenario))
    model = SEIR.Model(scenario, params, birthrates=birthrates_map.rates)
    # expdist = dists.gamma(shape=EXPOSED_DURATION_SHAPE, scale=EXPOSED_DURATION_SCALE)
    expdist = dists.normal(loc=EXPOSED_DURATION_SHAPE, scale=EXPOSED_DURATION_SCALE)
    infdist = dists.normal(loc=INFECTIOUS_DURATION_MEAN, scale=2)
    s = SEIR.Susceptible(model)
    e = SEIR.Exposed(model, expdist, infdist)
    i = SEIR.Infectious(model, infdist)
    r = SEIR.Recovered(model)
    tx = TransmissionWithDOI(model, expdist)
    importation = Importation(model, period=30, new_infections=[5] * model.nodes.count, infdurdist=infdist)
    pyramid = AliasedDistribution(np.full(89, 1_000))
    survival = KaplanMeierEstimator(np.full(89, 1_000).cumsum())
    vitals = SEIR.VitalDynamics(model, birthrates_map.rates, pyramid, survival)
    model.components = [s, r, i, e, tx, vitals, importation]
    label = f"SEIR with DOI (N={model.people.count:,}, Nodes={model.nodes.count:,})"
    model.run(label)
    # After run, model.people.doi contains tick of infection for each agent

    # model.plot()

    # Let's look at people infected in the last year of the simulation, doi >= NTICKS - 365
    recent_infections = (model.people.doi >= (NTICKS - 365)) & (model.people.doi != -1)
    aoi_recent = model.people.doi[recent_infections] - model.people.dob[recent_infections]
    import matplotlib.pyplot as plt

    plt.hist(aoi_recent, bins=range(aoi_recent.min(), aoi_recent.max() + 1), alpha=0.7)
    plt.xlabel("Age at Infection (Days)")
    plt.ylabel("Number of Infections")
    plt.title(label)
    plt.tight_layout()
    plt.show()

    print("done.")
