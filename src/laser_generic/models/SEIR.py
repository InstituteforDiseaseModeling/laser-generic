import contextily as ctx
import geopandas as gpd
import laser_core.distributions as dists
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from laser_core import LaserFrame
from laser_core.migration import distance
from laser_core.migration import gravity
from laser_core.migration import row_normalizer
from tqdm import tqdm

from laser_generic.newutils import RateMap
from laser_generic.newutils import TimingStats as ts
from laser_generic.newutils import estimate_capacity
from laser_generic.newutils import get_centroids
from laser_generic.newutils import validate

from .shared import State
from .shared import sample_dobs
from .shared import sample_dods

__all__ = ["Exposed", "Infectious", "Model", "Recovered", "State", "Susceptible", "Transmission", "VitalDynamics"]


class Susceptible:
    def __init__(self, model):
        self.model = model
        self.model.people.add_scalar_property("nodeid", dtype=np.uint16)
        self.model.people.add_scalar_property("state", dtype=np.int8, default=State.SUSCEPTIBLE.value)
        self.model.nodes.add_vector_property("S", model.params.nticks + 1, dtype=np.int32)

        self.model.people.nodeid[:] = np.repeat(np.arange(self.model.nodes.count, dtype=np.uint16), self.model.scenario.population)
        self.model.nodes.S[0] = self.model.scenario.S

        return

    def prevalidate_step(self, tick: int) -> None:
        # np.bincount where state == State.SUSCEPTIBLE.value and by nodeid should match self.model.nodes.S[tick]
        nodeids = self.model.people.nodeid
        states = self.model.people.state
        assert np.all(
            (expected := self.model.nodes.S[tick])
            == (actual := np.bincount(nodeids, states == State.SUSCEPTIBLE.value, minlength=self.model.nodes.count))
        ), f"Susceptible census does not match susceptible counts.\nExpected: {expected}\nActual: {actual}"

        return

    def postvalidate_step(self, tick: int) -> None:
        # Check that agents with state SUSCEPTIBLE by patch match self.model.nodes.S[tick]
        nodeids = self.model.people.nodeid
        states = self.model.people.state
        assert np.all(
            (expected := self.model.nodes.S[tick])
            == (actual := np.bincount(nodeids, states == State.SUSCEPTIBLE.value, minlength=self.model.nodes.count))
        ), f"Susceptible census does not match susceptible counts.\nExpected: {expected}\nActual: {actual}"
        assert np.all(self.model.nodes.S[tick + 1] == self.model.nodes.S[tick]), (
            "Susceptible counts should not change in Susceptible.step()."
        )

        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        # Propagate the number of susceptible individuals in each patch
        # state(t+1) = state(t) + ∆state(t), initialize state(t+1) with state(t)
        self.model.nodes.S[tick + 1] = self.model.nodes.S[tick]

        return

    def plot(self):
        _fig, ax1 = plt.subplots()
        for node in range(self.model.nodes.count):
            ax1.plot(self.model.nodes.S[:, node], label=f"Node {node}")
        ax1.set_xlabel("Tick")
        ax1.set_ylabel("Susceptible (by Node)")
        ax1.set_title("Susceptible over Time by Node")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(np.sum(self.model.nodes.S, axis=1), color="black", linestyle="--", label="Total Susceptible")
        ax2.set_ylabel("Total Susceptible")
        ax2.legend(loc="upper right")

        plt.show()

        return


class Exposed:
    def __init__(self, model, expdurdist, infdurdist, expdurmin=1, infdurmin=1):
        self.model = model
        self.model.people.add_scalar_property("etimer", dtype=np.uint8)
        self.model.nodes.add_vector_property("E", model.params.nticks + 1, dtype=np.int32)
        self.model.nodes.add_vector_property("symptomatic", model.params.nticks + 1, dtype=np.uint32)

        self.model.nodes.E[0] = self.model.scenario.E

        self.expdurdist = expdurdist
        self.infdurdist = infdurdist
        self.expdurmin = expdurmin
        self.infdurmin = infdurmin

        # convenience
        nodeids = self.model.people.nodeid
        states = self.model.people.state

        for node in range(self.model.nodes.count):
            nseeds = self.model.scenario.E[node]
            if nseeds > 0:
                i_susceptible = np.nonzero((nodeids == node) & (states == State.SUSCEPTIBLE.value))[0]
                assert nseeds <= len(i_susceptible), (
                    f"Node {node} has more initial exposed ({nseeds}) than available susceptible ({len(i_susceptible)})"
                )
                i_exposed = np.random.choice(i_susceptible, size=nseeds, replace=False)
                self.model.people.state[i_exposed] = State.EXPOSED.value
                samples = dists.sample_floats(self.expdurdist, np.zeros(nseeds, dtype=np.float32))
                samples = np.round(samples)
                samples = np.maximum(samples, self.expdurmin).astype(self.model.people.etimer.dtype)
                self.model.people.etimer[i_exposed] = samples
                assert np.all(self.model.people.etimer[i_exposed] > 0), "Exposed individuals should have etimer > 0"

        return

    def prevalidate_step(self, tick: int) -> None:
        # Note that we check exposed counts before the step (tick), so we check self.model.nodes.E[tick]

        # Make sure flow based accounting (faster) matches census based accounting (slower)
        nodeids = self.model.people.nodeid
        states = self.model.people.state
        assert np.all(self.model.nodes.E[tick] == np.bincount(nodeids, states == State.EXPOSED.value, minlength=self.model.nodes.count)), (
            "Exposed census does not match exposed counts (by state)."
        )

        # Don't need this since np.bincount can't (shouldn't) return negative counts
        # assert np.all(self.model.nodes.E[tick] >= 0), "Exposed counts must be non-negative"

        i_exposed = np.nonzero(self.model.people.state == State.EXPOSED.value)[0]
        assert np.all(self.model.people.etimer[i_exposed] > 0), "Exposed individuals should currently have etimer > 0."
        i_non_zero = np.nonzero(self.model.people.etimer > 0)[0]
        assert np.all(self.model.people.state[i_non_zero] == State.EXPOSED.value), "Only exposed individuals should have etimer > 0."

        return

    def postvalidate_step(self, tick: int) -> None:
        # Note that we check exposed counts after the step (tick), so we check self.model.nodes.E[tick+1]

        # Make sure flow based accounting (faster) matches census based accounting (slower)
        nodeids = self.model.people.nodeid
        states = self.model.people.state
        assert np.all(
            self.model.nodes.E[tick + 1] == np.bincount(nodeids, states == State.EXPOSED.value, minlength=self.model.nodes.count)
        ), "Exposed census does not match exposed counts (by state)."

        # Don't need this since np.bincount can't (shouldn't) return negative counts
        # assert np.all(self.model.nodes.E[tick+1] >= 0), "Exposed counts must be non-negative"

        i_exposed = np.nonzero(self.model.people.state == State.EXPOSED.value)[0]
        assert np.all(self.model.people.etimer[i_exposed] > 0), "Exposed individuals should currently have etimer > 0."
        i_non_zero = np.nonzero(self.model.people.etimer > 0)[0]
        assert np.all(self.model.people.state[i_non_zero] == State.EXPOSED.value), "Only exposed individuals should have etimer > 0."

        return

    @staticmethod
    @nb.njit(
        # (nb.int8[:], nb.uint8[:], nb.uint8[:], nb.uint32[:, :], nb.uint16[:], nb.types.FunctionType(nb.types.uint8()), nb.int32),
        nogil=True,
        parallel=True,
        cache=True,
    )
    def nb_exposed_step(states, etimers, itimers, symptomatic, nodeids, infdurdist, infdurmin):
        for i in nb.prange(len(states)):
            if states[i] == State.EXPOSED.value:
                etimers[i] -= 1
                if etimers[i] == 0:
                    states[i] = State.INFECTIOUS.value
                    itimers[i] = np.maximum(np.round(infdurdist()), infdurmin)  # Set the infection timer
                    symptomatic[nb.get_thread_id(), nodeids[i]] += 1

        return

    def step(self, tick: int) -> None:
        # Propagate the number of exposed individuals in each patch
        # state(t+1) = state(t) + ∆state(t), initialize state(t+1) with state(t)
        self.model.nodes.E[tick + 1] = self.model.nodes.E[tick]

        symptomatic_by_node = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.uint32)
        self.nb_exposed_step(
            self.model.people.state,
            self.model.people.etimer,
            self.model.people.itimer,
            symptomatic_by_node,
            self.model.people.nodeid,
            self.infdurdist,
            self.infdurmin,
        )
        symptomatic_by_node = symptomatic_by_node.sum(axis=0).astype(self.model.nodes.S.dtype)  # Sum over threads

        # state(t+1) = state(t) + ∆state(t)
        self.model.nodes.E[tick + 1] -= symptomatic_by_node
        self.model.nodes.I[tick + 1] += symptomatic_by_node
        # Record today's ∆
        self.model.nodes.symptomatic[tick] = symptomatic_by_node

        return

    def plot(self):
        _fig, ax1 = plt.subplots()
        for node in range(self.model.nodes.count):
            ax1.plot(self.model.nodes.E[:, node], label=f"Node {node}")
        ax1.set_xlabel("Tick")
        ax1.set_ylabel("Exposed (by Node)")
        ax1.set_title("Exposed over Time by Node")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(np.sum(self.model.nodes.E, axis=1), color="black", linestyle="--", label="Total Exposed")
        ax2.set_ylabel("Total Exposed")
        ax2.legend(loc="upper right")
        plt.show()

        return


class Infectious:
    def __init__(self, model, infdurdist, infdurmin=1):
        self.model = model
        self.model.people.add_scalar_property("itimer", dtype=np.uint8)
        self.model.nodes.add_vector_property("I", model.params.nticks + 1, dtype=np.int32)
        self.model.nodes.add_vector_property("recovered", model.params.nticks + 1, dtype=np.uint32)

        self.model.nodes.I[0] = self.model.scenario.I

        self.infdurdist = infdurdist
        self.infdurmin = infdurmin

        # convenience
        nodeids = self.model.people.nodeid
        states = self.model.people.state

        for node in range(self.model.nodes.count):
            nseeds = self.model.scenario.I[node]
            if nseeds > 0:
                i_susceptible = np.nonzero((nodeids == node) & (states == State.SUSCEPTIBLE.value))[0]
                assert nseeds <= len(i_susceptible), (
                    f"Node {node} has more initial infectious ({nseeds}) than available susceptible ({len(i_susceptible)})"
                )
                i_infectious = np.random.choice(i_susceptible, size=nseeds, replace=False)
                self.model.people.state[i_infectious] = State.INFECTIOUS.value
                samples = dists.sample_floats(self.infdurdist, np.zeros(nseeds, dtype=np.float32))
                samples = np.round(samples)
                samples = np.maximum(samples, self.infdurmin).astype(self.model.people.itimer.dtype)
                self.model.people.itimer[i_infectious] = samples
                assert np.all(self.model.people.itimer[i_infectious] > 0), "Infected individuals should have itimer > 0"

        return

    def prevalidate_step(self, tick: int) -> None:
        assert np.all(self.model.nodes.I[tick] >= 0), "Infected counts must be non-negative"
        i_infectious = np.nonzero(self.model.people.state == State.INFECTIOUS.value)[0]
        assert np.all(self.model.people.itimer[i_infectious] > 0), "Infected individuals should have itimer > 0"
        i_non_zero = np.nonzero(self.model.people.itimer > 0)[0]
        assert np.all(self.model.people.state[i_non_zero] == State.INFECTIOUS.value), "Only infectious individuals should have itimer > 0"

        return

    def postvalidate_step(self, tick: int) -> None:
        assert np.all(self.model.nodes.I[tick + 1] >= 0), "Infected counts must be non-negative"

        states = self.model.people.state
        itimers = self.model.people.itimer

        i_infectious = np.nonzero(states == State.INFECTIOUS.value)[0]
        assert np.all(itimers[i_infectious] > 0), "Infected individuals should have itimer > 0"
        i_non_zero = np.nonzero(itimers > 0)[0]
        assert np.all(states[i_non_zero] == State.INFECTIOUS.value), "Only infectious individuals should have itimer > 0"

        nodeids = self.model.people.nodeid
        # nodes.I should match count of infectious by node
        assert np.all(
            self.model.nodes.I[tick + 1] == np.bincount(nodeids, states == State.INFECTIOUS.value, minlength=self.model.nodes.count)
        ), "Infected census does not match infectious counts (by state)."
        assert np.all(self.model.nodes.I[tick + 1] == np.bincount(nodeids, itimers > 0, minlength=self.model.nodes.count)), (
            "Infected census does not match infectious counts (by itimer)."
        )

        return

    @staticmethod
    @nb.njit((nb.int8[:], nb.uint8[:], nb.uint32[:, :], nb.uint16[:]), nogil=True, parallel=True, cache=True)
    def nb_infectious_step(states, itimers, recovered, nodeids):
        for i in nb.prange(len(states)):
            if states[i] == State.INFECTIOUS.value:
                itimers[i] -= 1
                if itimers[i] == 0:
                    states[i] = State.RECOVERED.value
                    recovered[nb.get_thread_id(), nodeids[i]] += 1

        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        """Step function for the Infected component.

        Args:
            tick (int): The current tick of the simulation.
        """
        # Propagate the number of infectious individuals in each patch
        # state(t+1) = state(t) + ∆state(t), initialize state(t+1) with state(t)
        self.model.nodes.I[tick + 1] = self.model.nodes.I[tick]

        recovered_by_node = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.uint32)
        self.nb_infectious_step(self.model.people.state, self.model.people.itimer, recovered_by_node, self.model.people.nodeid)
        recovered_by_node = recovered_by_node.sum(axis=0).astype(self.model.nodes.S.dtype)  # Sum over threads

        # state(t+1) = state(t) + ∆state(t)
        self.model.nodes.I[tick + 1] -= recovered_by_node
        self.model.nodes.R[tick + 1] += recovered_by_node
        # Record today's ∆
        self.model.nodes.recovered[tick] = recovered_by_node

        return

    def plot(self):
        _fig, ax1 = plt.subplots()
        for node in range(self.model.nodes.count):
            ax1.plot(self.model.nodes.I[:, node], label=f"Node {node}")
        ax1.set_xlabel("Tick")
        ax1.set_ylabel("Infected (by Node)")
        ax1.set_title("Infected over Time by Node")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(np.sum(self.model.nodes.I, axis=1), color="black", linestyle="--", label="Total Infected")
        ax2.set_ylabel("Total Infected")
        ax2.legend(loc="upper right")
        plt.show()

        return


class Recovered:
    def __init__(self, model):
        self.model = model
        self.model.nodes.add_vector_property("R", model.params.nticks + 1, dtype=np.int32)

        self.model.nodes.R[0] = self.model.scenario.R

        # convenience
        nodeids = self.model.people.nodeid
        states = self.model.people.state

        for node in range(self.model.nodes.count):
            nseeds = self.model.scenario.I[node]
            if nseeds > 0:
                i_susceptible = np.nonzero((nodeids == node) & (states == State.SUSCEPTIBLE.value))[0]
                assert nseeds <= len(i_susceptible), (
                    f"Node {node} has more initial recovered ({nseeds}) than available susceptible ({len(i_susceptible)})"
                )
                i_recovered = np.random.choice(i_susceptible, size=nseeds, replace=False)
                self.model.people.state[i_recovered] = State.RECOVERED.value

        return

    def prevalidate_step(self, tick: int) -> None:
        assert np.all(self.model.nodes.R[tick] >= 0), "Recovered counts must be non-negative"

        return

    def postvalidate_step(self, tick: int) -> None:
        # Check that agents with state RECOVERED by patch match self.model.nodes.R[tick]
        nodeids = self.model.people.nodeid
        states = self.model.people.state
        assert np.all(
            (expected := self.model.nodes.R[tick])
            == (actual := np.bincount(nodeids, states == State.RECOVERED.value, minlength=self.model.nodes.count))
        ), f"Recovered census does not match recovered counts.\nExpected: {expected}\nActual: {actual}"
        assert np.all(self.model.nodes.R[tick + 1] == self.model.nodes.R[tick]), "Recovered counts should not change in Recovered.step()."

        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        # Propagate the number of recovered individuals in each patch
        # state(t+1) = state(t) + ∆state(t), initialize state(t+1) with state(t)
        self.model.nodes.R[tick + 1] = self.model.nodes.R[tick]

        return

    def plot(self):
        _fig, ax1 = plt.subplots()
        for node in range(self.model.nodes.count):
            ax1.plot(self.model.nodes.R[:, node], label=f"Node {node}")
        ax1.set_xlabel("Tick")
        ax1.set_ylabel("Recovered (by Node)")
        ax1.set_title("Recovered over Time by Node")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(np.sum(self.model.nodes.R, axis=1), color="black", linestyle="--", label="Total Recovered")
        ax2.set_ylabel("Total Recovered")
        ax2.legend(loc="upper right")
        plt.show()

        return


class Transmission:
    def __init__(self, model, expdurdist, expdurmin=1):
        self.model = model
        self.model.nodes.add_vector_property("forces", model.params.nticks + 1, dtype=np.float32)
        self.model.nodes.add_vector_property("incidence", model.params.nticks + 1, dtype=np.uint32)

        self.expdurdist = expdurdist
        self.expdurmin = expdurmin

        return

    def prevalidate_step(self, tick: int) -> None: ...

    def postvalidate_step(self, tick: int) -> None:
        assert np.all(self.model.nodes.incidence[tick] >= 0), "Incidence counts must be non-negative"
        i_infectious = np.nonzero(self.model.people.state == State.INFECTIOUS.value)[0]
        assert np.all(self.model.people.itimer[i_infectious] > 0), "Infectious individuals should have itimer > 0"

        return

    @staticmethod
    @nb.njit(
        # (
        #     nb.int8[:],
        #     nb.uint16[:],
        #     nb.float32[:],
        #     nb.uint32[:, :],
        #     nb.uint8[:],
        #     nb.types.FunctionType(nb.types.uint8()),
        #     nb.int32,
        # ),
        nogil=True,
        parallel=True,
        cache=True,
    )
    def nb_transmission_step(states, nodeids, ft, exp_by_node, etimers, expdurdist, expdurmin):
        for i in nb.prange(len(states)):
            if states[i] == State.SUSCEPTIBLE.value:
                # Check for infection
                draw = np.random.rand()
                nid = nodeids[i]
                if draw < ft[nid]:
                    states[i] = State.EXPOSED.value
                    etimers[i] = np.maximum(np.round(expdurdist()), expdurmin)  # Set the exposure timer
                    exp_by_node[nb.get_thread_id(), nid] += 1

        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        ft = self.model.nodes.forces[tick]
        N = self.model.nodes.S[tick] + self.model.nodes.I[tick] + self.model.nodes.R[tick]
        ft[:] = self.model.params.beta * self.model.nodes.I[tick] / N
        transfer = ft[:, None] * self.model.network
        ft += transfer.sum(axis=0)
        ft -= transfer.sum(axis=1)
        ft = -np.expm1(-ft)  # Convert to probability of infection

        exp_by_node = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.uint32)
        self.nb_transmission_step(
            self.model.people.state, self.model.people.nodeid, ft, exp_by_node, self.model.people.etimer, self.expdurdist, self.expdurmin
        )
        exp_by_node = exp_by_node.sum(axis=0).astype(self.model.nodes.S.dtype)  # Sum over threads

        # state(t+1) = state(t) + ∆state(t)
        self.model.nodes.S[tick + 1] -= exp_by_node
        self.model.nodes.E[tick + 1] += exp_by_node
        # Record today's ∆
        self.model.nodes.incidence[tick] = exp_by_node

        return

    def plot(self):
        for node in range(self.model.nodes.count):
            plt.plot(self.model.nodes.forces[:, node], label=f"Node {node}")
        plt.xlabel("Tick")
        plt.ylabel("Force of Infection")
        plt.title("Force of Infection over Time by Node")
        plt.legend()
        plt.show()

        return


class VitalDynamics:
    def __init__(self, model):
        self.model = model

        # This component requires these model properties
        assert hasattr(self.model, "birthrates")
        assert hasattr(self.model, "mortalityrates")
        assert hasattr(self.model, "pyramid")
        assert hasattr(self.model, "survival")

        # Date-Of-Birth and Date-Of-Death properties per agent
        self.model.people.add_property("dob", dtype=np.int16)
        self.model.people.add_property("dod", dtype=np.int16)
        # birth and death statistics per node
        self.model.nodes.add_vector_property("births", model.params.nticks + 1, dtype=np.uint32)
        self.model.nodes.add_vector_property("deaths", model.params.nticks + 1, dtype=np.uint32)

        # Initialize starting population
        self.pyramid = self.model.pyramid
        self.survival = self.model.survival
        dobs = self.model.people.dob[0 : self.model.people.count]
        dods = self.model.people.dod[0 : self.model.people.count]
        sample_dobs(dobs, self.pyramid, tick=0)
        sample_dods(dobs, dods, self.survival, tick=0)

        return

    def prevalidate_step(self, tick: int) -> None:
        self._cpeople = self.model.people.count
        self._deceased = self.model.people.state == State.DECEASED.value

        return

    def postvalidate_step(self, tick: int) -> None:
        assert np.all(self.model.nodes.I[tick + 1] >= 0), "Infected counts must be non-negative"

        nbirths = self.model.births[tick].sum()
        assert self.model.people.count == self._cpeople + nbirths, "Population count mismatch after births"
        istart = self._cpeople
        iend = self.model.people.count
        # Assert that number of births by patch matches self.model.births[tick]
        birth_counts = np.bincount(self.model.people.nodeid[istart:iend], minlength=self.model.nodes.count)
        assert np.all(birth_counts == self.model.births[tick]), "Birth counts by patch mismatch"
        assert np.all(self.model.people.state[istart:iend] == State.SUSCEPTIBLE.value), "Newborns should be susceptible"
        assert np.all(self.model.people.itimer[istart:iend] == 0), "Newborns should have itimer == 0"

        ndeaths = self.model.deaths[tick].sum()
        deceased = self.model.people.state == State.DECEASED.value
        assert ndeaths == deceased.sum() - self._deceased.sum(), "Death counts mismatch"
        # Assert that new deaths by patch matches self.model.deaths[tick]
        prv = np.bincount(self.model.people.nodeid[0 : self._cpeople][self._deceased], minlength=self.model.nodes.count)
        now = np.bincount(self.model.people.nodeid[deceased], minlength=self.model.nodes.count)
        death_counts = now - prv
        assert np.all(death_counts == self.model.deaths[tick]), "Death counts by patch mismatch"

        return

    @staticmethod
    @nb.njit(
        (nb.int16[:], nb.int8[:], nb.uint16[:], nb.int32[:, :], nb.int32[:, :], nb.int32[:, :], nb.int32),
        nogil=True,
        parallel=True,
        cache=True,
    )
    def nb_process_deaths(dods, states, nodeids, deceased_S, deceased_I, deceased_R, tick):
        for i in nb.prange(len(dods)):
            if dods[i] == tick:
                state = states[i]
                if state >= 0:  # Ignore already deceased
                    if state == State.SUSCEPTIBLE.value:
                        deceased_S[nb.get_thread_id(), nodeids[i]] += 1
                    elif state == State.INFECTIOUS.value:
                        deceased_I[nb.get_thread_id(), nodeids[i]] += 1
                    else:  # if state == State.RECOVERED.value:
                        deceased_R[nb.get_thread_id(), nodeids[i]] += 1
                    states[i] = State.DECEASED.value

        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        # Do mortality first, then births
        deceased_S = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        deceased_I = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        deceased_R = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        self.nb_process_deaths(
            self.model.people.dod, self.model.people.state, self.model.people.nodeid, deceased_S, deceased_I, deceased_R, tick
        )
        # Combine thread results
        deceased_S = deceased_S.sum(axis=0).astype(self.model.nodes.S.dtype)
        deceased_I = deceased_I.sum(axis=0).astype(self.model.nodes.I.dtype)
        deceased_R = deceased_R.sum(axis=0).astype(self.model.nodes.R.dtype)
        # state(t+1) = state(t) + ∆state(t)
        self.model.nodes.S[tick + 1] -= deceased_S
        self.model.nodes.I[tick + 1] -= deceased_I
        self.model.nodes.R[tick + 1] -= deceased_R
        # Record today's ∆
        self.model.nodes.deaths[tick] = deceased_S + deceased_I + deceased_R  # Record

        # Births in one fell swoop
        rates = np.power(1.0 + self.model.birthrates[tick] / 1000, 1.0 / 365) - 1.0
        # Use "tomorrow's" population which accounts for mortality above.
        N = self.model.nodes.S[tick + 1] + self.model.nodes.I[tick + 1] + self.model.nodes.R[tick + 1]
        births = np.round(np.random.poisson(rates * N)).astype(np.uint32)
        tbirths = births.sum()
        if tbirths > 0:
            istart, iend = self.model.people.add(tbirths)
            self.model.people.nodeid[istart:iend] = np.repeat(np.arange(self.model.nodes.count, dtype=np.uint16), births)
            # State.SUSCEPTIBLE.value is the default
            # self.model.people.state[istart:iend] = State.SUSCEPTIBLE.value
            # state(t+1) = state(t) + ∆state(t)
            self.model.nodes.S[tick + 1] += births
            # Record today's ∆
            self.model.nodes.births[tick] = births

        return

    def plot(self):
        _fig, ax1 = plt.subplots(figsize=(16, 12))
        births = np.sum(self.model.nodes.births, axis=1)
        deaths = np.sum(self.model.nodes.deaths, axis=1)
        ax1.plot(births, label="Daily Births", color="green")
        ax1.plot(deaths, label="Daily Deaths", color="red")
        ax1.set_xlabel("Tick")
        ax1.set_ylabel("Count")
        ax1.set_title("Births and Deaths Over Time")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(np.cumsum(births), color="tab:green", linestyle="--", label="Cumulative Births")
        ax2.plot(np.cumsum(deaths), color="tab:red", linestyle="--", label="Cumulative Deaths")
        ax2.set_ylabel("Cumulative Count")
        ax2.legend(loc="upper right")
        plt.show()

        return


class Model:
    def __init__(self, scenario, params, birthrates=None, mortalityrates=None, skip_capacity: bool = False):
        """
        Initialize the SI model.

        Args:
            scenario (GeoDataFrame): The scenario data containing per patch population, initial S and I counts, and geometry.
            params (PropertySet): The parameters for the model, including 'nticks' and 'beta'.
            birthrates (np.ndarray, optional): Birth rates in CBR per patch per tick. Defaults to None.
            mortalityrates (np.ndarray, optional): Mortality rates in count/1000 per patch per tick. Defaults to None.
            skip_capacity (bool, optional): If True, skips capacity checks. Defaults to False.
        """
        self.params = params

        num_nodes = max(np.unique(scenario.nodeid)) + 1
        self.birthrates = birthrates if birthrates is not None else RateMap.from_scalar(0, num_nodes, self.params.nticks).rates
        self.mortalityrates = mortalityrates if mortalityrates is not None else RateMap.from_scalar(0, num_nodes, self.params.nticks).rates
        num_active = scenario.population.sum()
        if not skip_capacity:
            num_agents = estimate_capacity(self.birthrates, scenario.population).sum()
        else:
            # Ignore births for capacity calculation
            num_agents = num_active

        # TODO - remove int() cast with newer version of laser-core
        self.people = LaserFrame(int(num_agents), int(num_active))
        self.nodes = LaserFrame(int(num_nodes))

        self.scenario = scenario
        self.validating = False

        centroids = get_centroids(scenario)
        self.scenario["x"] = centroids.x
        self.scenario["y"] = centroids.y

        # Calculate pairwise distances between nodes using centroids
        longs = self.scenario["x"].values
        lats = self.scenario["y"].values
        population = self.scenario["population"].values

        # Compute distance matrix
        if len(scenario) > 1:
            dist_matrix = distance(lats, longs, lats, longs)
        else:
            dist_matrix = np.array([[0.0]], dtype=np.float32)
        assert dist_matrix.shape == (self.nodes.count, self.nodes.count), "Distance matrix shape mismatch"

        # Compute gravity network matrix
        self.network = gravity(population, dist_matrix, k=500, a=1, b=1, c=2)
        self.network = row_normalizer(self.network, (1 / 16) + (1 / 32))

        self.basemap_provider = ctx.providers.Esri.WorldImagery

        self._components = []

        return

    def run(self, label="SIR Model") -> None:
        with ts.start(f"Running Simulation {label}"):
            for tick in tqdm(range(self.params.nticks), desc=f"Running Simulation {label}"):
                for c in self.components:
                    with ts.start(f"{c.__class__.__name__}.step()"):
                        c.step(tick)

        return

    @property
    def components(self):
        return self._components

    @components.setter
    def components(self, value):
        self._components = value

        return

    def _plot(self, basemap_provider=ctx.providers.Esri.WorldImagery):
        if "geometry" in self.scenario.columns:
            gdf = gpd.GeoDataFrame(self.scenario, geometry="geometry")

            if basemap_provider is None:
                pop = gdf["population"].values
                norm = mcolors.Normalize(vmin=pop.min(), vmax=pop.max())
                saturations = norm(pop)
                colors = [plt.cm.Blues(sat) for sat in saturations]
                ax = gdf.plot(facecolor=colors, edgecolor="black", linewidth=1)
                sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
                cbar.set_label("Population")
                plt.title("Node Boundaries and Populations")
            else:
                gdf_merc = gdf.to_crs(epsg=3857)
                pop = gdf_merc["population"].values
                # Plot the basemap and shape outlines
                _fig, ax = plt.subplots(figsize=(8, 8))
                bounds = gdf_merc.total_bounds  # [minx, miny, maxx, maxy]
                xmid = (bounds[0] + bounds[2]) / 2
                ymid = (bounds[1] + bounds[3]) / 2
                xhalf = (bounds[2] - bounds[0]) / 2
                yhalf = (bounds[3] - bounds[1]) / 2
                ax.set_xlim(xmid - 2 * xhalf, xmid + 2 * xhalf)
                ax.set_ylim(ymid - 2 * yhalf, ymid + 2 * yhalf)
                ctx.add_basemap(ax, source=basemap_provider)
                gdf_merc.boundary.plot(ax=ax, edgecolor="black", linewidth=1)

                # Draw circles at centroids sized by log(population)
                centroids = gdf_merc.geometry.centroid
                print(f"{pop=}")
                sizes = 20 + 2 * pop / 10_000
                ax.scatter(centroids.x, centroids.y, s=sizes, color="red", edgecolor="black", zorder=10, alpha=0.8)

                plt.title("Node Boundaries, Centroids, and Basemap")

            """
            # Add interactive hover to display population
            cursor = mplcursors.cursor(ax.collections[0], hover=True)

            @cursor.connect("add")
            def on_add(sel):
                # sel.index is a tuple; sel.index[0] is the nodeid (row index in gdf)
                nodeid = sel.index[0]
                pop_val = gdf.iloc[nodeid]["population"]
                sel.annotation.set_text(f"Population: {pop_val}")
            """

            plt.show()

        # Plot active population (S + E + I + R) and total deceased over time
        _fig, ax1 = plt.subplots(figsize=(10, 6))
        active_population = self.nodes.S + self.nodes.E + self.nodes.I + self.nodes.R
        total_active = np.sum(active_population, axis=1)
        ax1.plot(total_active, label="Active Population (S + E + I + R)", color="blue")
        ax1.set_xlabel("Tick")
        ax1.set_ylabel("Active Population", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.legend(loc="upper left")

        if hasattr(self.nodes, "deaths"):
            ax2 = ax1.twinx()
            total_deceased = np.sum(self.nodes.deaths, axis=1).cumsum()
            ax2.plot(total_deceased, label="Total Deceased", color="red")
            ax2.set_ylabel("Total Deceased", color="red")
            ax2.tick_params(axis="y", labelcolor="red")
            ax2.legend(loc="upper right")

            plt.title("Active Population and Total Deceased Over Time")
        else:
            plt.title("Active Population Over Time")

        plt.tight_layout()
        plt.show()

        # Plot total S, total E, total I, total R over time
        _fig, ax1 = plt.subplots(figsize=(10, 6))
        total_S = np.sum(self.nodes.S, axis=1)
        total_E = np.sum(self.nodes.E, axis=1)
        total_I = np.sum(self.nodes.I, axis=1)
        total_R = np.sum(self.nodes.R, axis=1)
        ax1.plot(total_S, label="Total Susceptible (S)", color="blue")
        ax1.plot(total_E, label="Total Exposed (E)", color="orange")
        ax1.plot(total_I, label="Total Infectious (I)", color="red")
        ax1.plot(total_R, label="Total Recovered (R)", color="green")
        ax1.set_xlabel("Tick")
        ax1.set_ylabel("Count")
        ax1.legend(loc="upper right")
        plt.title("Total Susceptible, Exposed, Infectious and Recovered Over Time")
        plt.tight_layout()
        plt.show()

        return

    def plot(self):
        self._plot(getattr(self, "basemap_provider", None))  # Pass basemap_provider argument to _plot if provided
        for c in self.components:
            if hasattr(c, "plot") and callable(c.plot):
                c.plot()

        return
