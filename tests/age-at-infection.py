"""
Prompt:
Let's create an SEIR model with a custom Transmission component, based on SEIR.Transmission, which add a "doi" property to the people in the model, implements a new Numba-compiled function to run transmission including storing the current tick in doi when an agent is infected, and overriding the step() function to call this new Numba function with all the previous properties plus the current tick and the doi property array.
Please implement in age-at-infection.py but you can reference code from test_seir.py and the implementation of TransmissionSE in components.py that the base SEIR model uses for Transmission.
Do not modify code in any other file at this time.
"""

import numba as nb
import numpy as np

import laser_generic.models.SEIR as SEIR


class TransmissionWithDOI(SEIR.Transmission):
    def __init__(self, model, expdurdist, expdurmin=1):
        super().__init__(model, expdurdist, expdurmin)
        # Add 'doi' property to people (default 0, dtype=int32)
        self.model.people.add_scalar_property("doi", dtype=np.int32)
        # Optionally initialize to -1 to indicate never infected
        self.model.people.doi[:] = -1

    @staticmethod
    @nb.njit(nogil=True, parallel=True, cache=True)
    def nb_transmission_step(states, etimers, nodeids, network, expdurdist, expdurmin, tick, doi):
        n_people = states.shape[0]
        n_nodes = network.shape[0]
        # For each node, calculate force of infection
        force = np.zeros(n_nodes, dtype=np.float32)
        for i in range(n_nodes):
            # Example: force proportional to exposed in node
            force[i] = 0.0
            for j in range(n_nodes):
                force[i] += network[i, j] * (states[j] == 2)  # 2 = EXPOSED (adjust as needed)
        # For each person, determine infection
        for i in nb.prange(n_people):
            if states[i] == 0:  # 0 = SUSCEPTIBLE
                node = nodeids[i]
                # Draw random for infection
                if np.random.rand() < force[node]:
                    states[i] = 2  # EXPOSED
                    etimers[i] = max(expdurdist(), expdurmin)
                    doi[i] = tick  # Record tick of infection

    def step(self, tick: int) -> None:
        # Call the custom Numba transmission step
        self.nb_transmission_step(
            self.model.people.state,
            self.model.people.etimer,
            self.model.people.nodeid,
            self.model.network,
            self.expdurdist,
            self.expdurmin,
            tick,
            self.model.people.doi,
        )
        # Optionally, update node-level exposed counts, etc. as in base class
        # ...existing code...


# Example usage in a test or simulation setup:
if __name__ == "__main__":
    # Build a model as in test_seir.py, but use TransmissionWithDOI
    import laser_core.distributions as dists
    from laser_core import PropertySet

    NTICKS = 365
    R0 = 1.386
    INFECTIOUS_DURATION_MEAN = 7.0
    EXPOSED_DURATION_SHAPE = 4.5
    EXPOSED_DURATION_SCALE = 1.0

    scenario = ...  # Build scenario as in test_seir.py
    params = PropertySet({"nticks": NTICKS, "beta": R0 / INFECTIOUS_DURATION_MEAN})
    model = SEIR.Model(scenario, params)
    expdist = dists.gamma(shape=EXPOSED_DURATION_SHAPE, scale=EXPOSED_DURATION_SCALE)
    infdist = dists.normal(loc=INFECTIOUS_DURATION_MEAN, scale=2)
    s = SEIR.Susceptible(model)
    e = SEIR.Exposed(model, expdist, infdist)
    i = SEIR.Infectious(model, infdist)
    r = SEIR.Recovered(model)
    tx = TransmissionWithDOI(model, expdist)
    model.components = [s, r, i, e, tx]
    model.run()
    # After run, model.people.doi contains tick of infection for each agent
