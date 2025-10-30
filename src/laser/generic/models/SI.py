"""
Components for an SI model.

Agents transition from Susceptible to Infectious upon infection.
Agents remain in the Infectious state indefinitely (no recovery).
"""

import warnings

import matplotlib.pyplot as plt
import numba as nb
import numpy as np

from laser.generic.newutils import ValuesMap
from laser.generic.newutils import validate

from .components import InfectiousSI as Infectious
from .components import Susceptible
from .components import TransmissionSIX as Transmission
from .components import VitalDynamicsSI as VitalDynamics
from .components import _check_flow_vs_census
from .shared import State

__all__ = ["ConstantPopVitalDynamics", "Infectious", "State", "Susceptible", "Transmission", "VitalDynamics"]


class ConstantPopVitalDynamics:
    def __init__(self, model, birthrates=None, mortalityrates=None):
        self.model = model
        self.model.nodes.add_vector_property("births", model.params.nticks + 1, dtype=np.uint32)
        self.model.nodes.add_vector_property("deaths", model.params.nticks + 1, dtype=np.uint32)

        if birthrates is not None:
            self.birthrates = birthrates
        else:
            self.birthrates = ValuesMap.from_scalar(0.0, model.nodes.count, model.params.nticks).values
            warnings.warn("No birthrates found in model; defaulting to zero birthrates.", stacklevel=2)

        if mortalityrates is not None:
            self.mortalityrates = mortalityrates
        else:
            self.mortalityrates = ValuesMap.from_scalar(0.0, model.nodes.count, model.params.nticks).values
            warnings.warn("No mortalityrates found in model; defaulting to zero mortalityrates.", stacklevel=2)

        # We will use the larger of birthrates or mortalityrates for recycling
        self.rates = np.maximum(self.birthrates, self.mortalityrates)

        assert self.rates.shape == (self.model.params.nticks, self.model.nodes.count), (
            f"Births ({self.birthrates.shape})/deaths ({self.mortalityrates.shape}) array shape mismatch, expected ({self.model.params.nticks}, {self.model.nodes.count})"
        )

        return

    def prevalidate_step(self, tick: int) -> None:
        # Look ahead to state at tick+1 since transmission may (should) have occurred meaning S[tick] and I[tick] are outdated.
        _check_flow_vs_census(self.model.nodes.S[tick + 1], self.model.people, State.SUSCEPTIBLE, "Susceptible")
        _check_flow_vs_census(self.model.nodes.I[tick + 1], self.model.people, State.INFECTIOUS, "Infectious")

        return

    def postvalidate_step(self, tick: int) -> None:
        _check_flow_vs_census(self.model.nodes.S[tick + 1], self.model.people, State.SUSCEPTIBLE, "Susceptible")
        _check_flow_vs_census(self.model.nodes.I[tick + 1], self.model.people, State.INFECTIOUS, "Infectious")

        return

    @staticmethod
    @nb.njit(
        # (nb.float32[:], nb.int8[:], nb.uint16[:], nb.int32[:, :], nb.int32[:, :]),
        nogil=True,
        parallel=True,
        cache=True,
    )
    def nb_process_recycling(rates, states, nodeids, recycled, infected):
        for i in nb.prange(len(states)):
            draw = np.random.rand()
            nid = nodeids[i]
            if draw < rates[nid]:
                tid = nb.get_thread_id()
                recycled[tid, nid] += 1
                if states[i] == State.INFECTIOUS.value:
                    states[i] = State.SUSCEPTIBLE.value
                    infected[tid, nid] += 1
                # else: # states[i] is already SUSCEPTIBLE, no change

        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        cxr = self.rates[tick]  # cxr because we've selected the larger of birth/death rates (CBR or CDR)
        annual_growth_rates = 1.0 + (cxr / 1_000)
        daily_growth_rates = np.power(annual_growth_rates, 1.0 / 365)
        probabilities = -np.expm1(1.0 - daily_growth_rates)

        recycled = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        infected = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        self.nb_process_recycling(probabilities, self.model.people.state, self.model.people.nodeid, recycled, infected)
        recycled = recycled.sum(axis=0).astype(self.model.nodes.S.dtype)  # Sum over threads
        infected = infected.sum(axis=0).astype(self.model.nodes.I.dtype)  # Sum over threads

        # state(t+1) = state(t) + ∆state(t)
        self.model.nodes.S[tick + 1] += infected
        self.model.nodes.I[tick + 1] -= infected

        # Record today's ∆
        self.model.nodes.births[tick] = recycled  # Record recycled as "births"
        self.model.nodes.deaths[tick] = recycled  # Record recycled as "deaths"

        return

    def plot(self):
        plt.figure(figsize=(16, 9), dpi=200)
        plt.plot(np.sum(self.rates, axis=1), label="Daily Recycling")
        plt.xlabel("Tick")
        plt.ylabel("Count")
        plt.title("Recycling Over Time")
        plt.legend()
        plt.show()

        return
