import laser_core.distributions as dists
import matplotlib.pyplot as plt
import numba as nb
import numpy as np

from laser_generic.newutils import validate

from .shared import State
from .shared import sample_dobs
from .shared import sample_dods


class Susceptible:
    def __init__(self, model):
        self.model = model
        self.model.people.add_scalar_property("nodeid", dtype=np.uint16)
        self.model.people.add_scalar_property("state", dtype=np.int8)
        self.model.nodes.add_vector_property("S", model.params.nticks + 1)

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
    def nb_exposed_step(states, etimers, itimers, symptomatic, nodeids, infdurdist, infdurmin, tick):
        for i in nb.prange(len(states)):
            if states[i] == State.EXPOSED.value:
                etimers[i] -= 1
                if etimers[i] == 0:
                    states[i] = State.INFECTIOUS.value
                    nid = nodeids[i]
                    itimers[i] = np.maximum(np.round(infdurdist(tick, nid)), infdurmin)  # Set the infection timer
                    symptomatic[nb.get_thread_id(), nid] += 1

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
            tick,
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


class InfectiousSI:
    """Infectious component for an SI model - no recovery."""

    def __init__(self, model):
        self.model = model
        self.model.nodes.add_vector_property("I", model.params.nticks + 1)

        # convenience
        nodeids = self.model.people.nodeid
        populations = self.model.scenario.population.values

        for node in range(self.model.nodes.count):
            citizens = np.nonzero(nodeids == node)[0]
            assert len(citizens) == populations[node], f"Found {len(citizens)} citizens in node {node} but expected {populations[node]}"
            nseeds = self.model.scenario.I[node]
            assert nseeds <= len(citizens), f"Node {node} has more initial infected ({nseeds}) than population ({len(citizens)})"
            if nseeds > 0:
                indices = np.random.choice(citizens, size=nseeds, replace=False)
                self.model.people.state[indices] = State.INFECTIOUS.value

        self.model.nodes.I[0] = self.model.scenario.I
        assert np.all(self.model.nodes.S[0] + self.model.nodes.I[0] == self.model.scenario.population), (
            "Initial S + I does not equal population"
        )

        return

    def prevalidate_step(self, tick: int) -> None: ...

    def postvalidate_step(self, tick: int) -> None:
        nodeids = self.model.people.nodeid
        state = self.model.people.state
        assert np.all(
            (expected := self.model.nodes.I[tick])
            == (actual := np.bincount(nodeids, state == State.INFECTIOUS.value, minlength=self.model.nodes.count))
        ), f"Infected census does not match infected counts.\nExpected: {expected}\nActual: {actual}"
        assert np.all(self.model.nodes.I[tick + 1] == self.model.nodes.I[tick]), (
            "Infected counts should not change outside of Transmission and VitalDynamics."
        )

        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        """Step function for the Infected component.

        Args:
            tick (int): The current tick of the simulation.
        """
        # Propagate the number of infected individuals in each patch
        # state(t+1) = state(t) + ∆state(t), initialize state(t+1) with state(t)
        self.model.nodes.I[tick + 1] = self.model.nodes.I[tick]

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


class InfectiousIS:
    def __init__(self, model, infdurdist, infdurmin=1):
        self.model = model
        self.model.people.add_scalar_property("itimer", dtype=np.uint8)
        self.model.nodes.add_vector_property("I", model.params.nticks + 1, dtype=np.int32)
        self.model.nodes.add_vector_property("recovered", model.params.nticks + 1, dtype=np.uint32)

        self.infdurdist = infdurdist
        self.infdurmin = infdurmin

        # convenience
        nodeids = self.model.people.nodeid
        populations = self.model.scenario.population.values

        for node in range(self.model.nodes.count):
            nseeds = self.model.scenario.I[node]
            if nseeds > 0:
                i_citizens = np.nonzero(nodeids == node)[0]
                assert len(i_citizens) == populations[node], (
                    f"Found {len(i_citizens)} citizens in node {node} but expected {populations[node]}"
                )
                assert nseeds <= len(i_citizens), f"Node {node} has more initial infectious ({nseeds}) than population ({len(i_citizens)})"
                i_infectious = np.random.choice(i_citizens, size=nseeds, replace=False)
                self.model.people.state[i_infectious] = State.INFECTIOUS.value
                samples = dists.sample_floats(self.infdurdist, np.zeros(nseeds, dtype=np.int32))
                samples = np.round(samples)
                samples = np.maximum(samples, self.infdurmin).astype(self.model.people.itimer.dtype)
                self.model.people.itimer[i_infectious] = samples
                assert np.all(self.model.people.itimer[i_infectious] > 0), "Infected individuals should have itimer > 0"

        self.model.nodes.I[0] = self.model.scenario.I
        assert np.all(self.model.nodes.S[0] + self.model.nodes.I[0] == self.model.scenario.population), (
            "Initial S + I does not equal population"
        )

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
    @nb.njit(
        # (nb.int8[:], nb.uint8[:], nb.uint32[:, :], nb.uint16[:]),
        nogil=True,
        parallel=True,
        cache=True,
    )
    def nb_infectious_step(states, itimers, recovered, nodeids):
        for i in nb.prange(len(states)):
            if states[i] == State.INFECTIOUS.value:
                itimers[i] -= 1
                if itimers[i] == 0:
                    states[i] = State.SUSCEPTIBLE.value
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
        self.model.nodes.S[tick + 1] += recovered_by_node
        self.model.nodes.I[tick + 1] -= recovered_by_node
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


class InfectiousIR:
    def __init__(self, model, infdurdist, infdurmin=1):
        self.model = model
        self.model.people.add_scalar_property("itimer", dtype=np.uint16)
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
    @nb.njit(
        # (nb.int8[:], nb.uint8[:], nb.uint32[:, :], nb.uint16[:]),
        nogil=True,
        parallel=True,
        cache=True,
    )
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


class InfectiousIRS:
    def __init__(self, model, infdurdist, wandurdist, infdurmin=1, wandurmin=1):
        self.model = model
        self.model.people.add_scalar_property("itimer", dtype=np.uint16)
        if not hasattr(self.model.people, "rtimer"):
            self.model.people.add_scalar_property("rtimer", dtype=np.uint16)
        self.model.nodes.add_vector_property("I", model.params.nticks + 1, dtype=np.int32)
        self.model.nodes.add_vector_property("recovered", model.params.nticks + 1, dtype=np.uint32)

        self.model.nodes.I[0] = self.model.scenario.I

        self.infdurdist = infdurdist
        self.wandurdist = wandurdist
        self.infdurmin = infdurmin
        self.wandurmin = wandurmin

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
                samples = np.maximum(self.infdurmin, samples).astype(self.model.people.itimer.dtype)
                self.model.people.itimer[i_infectious] = samples
                assert np.all(self.model.people.itimer[i_infectious] > 0), "Infected individuals should have itimer > 0"

        return

    def prevalidate_step(self, tick: int) -> None:
        # Check that agents with state INFECTIOUS by patch match self.model.nodes.I[tick]
        nodeids = self.model.people.nodeid
        states = self.model.people.state
        assert np.all(
            self.model.nodes.I[tick] == np.bincount(nodeids, states == State.INFECTIOUS.value, minlength=self.model.nodes.count)
        ), "Infected census does not match infectious counts (by state)."

        # Check that all infectious agents have itimer > 0
        i_infectious = np.nonzero(states == State.INFECTIOUS.value)[0]
        assert np.all(self.model.people.itimer[i_infectious] > 0), "Infected individuals should have itimer > 0"

        # Check that only infectious agents have itimer > 0
        i_non_zero = np.nonzero(self.model.people.itimer > 0)[0]
        assert np.all(states[i_non_zero] == State.INFECTIOUS.value), "Only infectious individuals should have itimer > 0"

        return

    def postvalidate_step(self, tick: int) -> None:
        # Check that agents with state INFECTIOUS by patch match self.model.nodes.I[tick+1]
        nodeids = self.model.people.nodeid
        states = self.model.people.state
        assert np.all(
            self.model.nodes.I[tick + 1] == np.bincount(nodeids, states == State.INFECTIOUS.value, minlength=self.model.nodes.count)
        ), "Infected census does not match infectious counts (by state)."

        # Check that all infectious agents have itimer > 0
        i_infectious = np.nonzero(states == State.INFECTIOUS.value)[0]
        assert np.all(self.model.people.itimer[i_infectious] > 0), "Infected individuals should have itimer > 0"

        # Check that only infectious agents have itimer > 0
        i_non_zero = np.nonzero(self.model.people.itimer > 0)[0]
        assert np.all(states[i_non_zero] == State.INFECTIOUS.value), "Only infectious individuals should have itimer > 0"

        return

    @staticmethod
    @nb.njit(
        # (nb.int8[:], nb.uint8[:], nb.uint8[:], nb.uint32[:, :], nb.uint16[:], nb.types.FunctionType(nb.types.uint8()), min),
        nogil=True,
        parallel=True,
        cache=True,
    )
    def nb_infectious_step(states, itimers, rtimers, recovered, nodeids, wandurdist, wandurmin, tick):
        for i in nb.prange(len(states)):
            if states[i] == State.INFECTIOUS.value:
                itimers[i] -= 1
                if itimers[i] == 0:
                    states[i] = State.RECOVERED.value
                    nid = nodeids[i]
                    rtimers[i] = np.maximum(np.round(wandurdist(tick, nid)), wandurmin)  # Set the recovery timer
                    recovered[nb.get_thread_id(), nid] += 1

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
        self.nb_infectious_step(
            self.model.people.state,
            self.model.people.itimer,
            self.model.people.rtimer,
            recovered_by_node,
            self.model.people.nodeid,
            self.wandurdist,
            self.wandurmin,
            tick,
        )
        recovered_by_node = recovered_by_node.sum(axis=0).astype(self.model.nodes.S.dtype)  # Sum over threads

        # state(t+1) = state(t) + ∆state(t)
        self.model.nodes.I[tick + 1] -= recovered_by_node
        self.model.nodes.R[tick + 1] += recovered_by_node
        # Record today's ∆
        self.model.nodes.recovered[tick] = recovered_by_node

        return

    def plot(self):
        # First plot: Infected over Time by Node
        _fig, ax1 = plt.subplots()
        for node in range(self.model.nodes.count):
            ax1.plot(self.model.nodes.I[:, node], label=f"Node {node}")
        ax1.set_xlabel("Tick")
        ax1.set_ylabel("Infected (by Node)")
        ax1.set_title("Infected over Time by Node")
        ax1.legend(loc="upper left")
        plt.show()

        # Second plot: Total Infected and Total Recovered over Time
        _fig, ax2 = plt.subplots()
        total_infected = np.sum(self.model.nodes.I, axis=1)
        total_recovered = np.sum(self.model.nodes.recovered, axis=1)
        ax2.plot(total_infected, color="black", linestyle="--", label="Total Infected")
        ax2.plot(total_recovered, color="green", linestyle="-.", label="Total Recovered")
        ax2.set_xlabel("Tick")
        ax2.set_ylabel("Count")
        ax2.set_title("Total Infected and Total Recovered Over Time")
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
            nseeds = self.model.scenario.R[node]
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


class RecoveredRS:
    def __init__(self, model, wandurdist, wandurmin=1):
        self.model = model
        if not hasattr(self.model.people, "rtimer"):
            self.model.people.add_scalar_property("rtimer", dtype=np.uint16)
        self.model.nodes.add_vector_property("R", model.params.nticks + 1, dtype=np.int32)
        self.model.nodes.add_vector_property("waned", model.params.nticks + 1, dtype=np.uint32)

        self.model.nodes.R[0] = self.model.scenario.R

        self.wandurdist = wandurdist
        self.wandurmin = wandurmin

        # convenience
        nodeids = self.model.people.nodeid
        states = self.model.people.state

        for node in range(self.model.nodes.count):
            nseeds = self.model.scenario.R[node]
            if nseeds > 0:
                i_susceptible = np.nonzero((nodeids == node) & (states == State.SUSCEPTIBLE.value))[0]
                assert nseeds <= len(i_susceptible), (
                    f"Node {node} has more initial recovered ({nseeds}) than available susceptible ({len(i_susceptible)})"
                )
                i_recovered = np.random.choice(i_susceptible, size=nseeds, replace=False)
                self.model.people.state[i_recovered] = State.RECOVERED.value
                samples = dists.sample_floats(self.wandurdist, np.zeros(nseeds, dtype=np.float32))
                samples = np.round(samples)
                samples = np.maximum(samples, self.wandurmin).astype(self.model.people.rtimer.dtype)
                self.model.people.rtimer[i_recovered] = samples
                assert np.all(self.model.people.rtimer[i_recovered] > 0), "Recovered individuals should have rtimer > 0"

        return

    def prevalidate_step(self, tick: int) -> None:
        # Check that agents with state RECOVERED by patch match self.model.nodes.R[tick]
        nodeids = self.model.people.nodeid
        states = self.model.people.state
        assert np.all(
            self.model.nodes.R[tick] == np.bincount(nodeids, states == State.RECOVERED.value, minlength=self.model.nodes.count)
        ), "Recovered census does not match recovered counts (by state)."
        # Don't need this since np.bincount can't (shouldn't) return negative counts
        # assert np.all(self.model.nodes.R[tick] >= 0), "Recovered counts must be non-negative"

        i_non_zero = np.nonzero(self.model.people.rtimer > 0)[0]
        assert np.all(states[i_non_zero] == State.RECOVERED.value), "Only recovered individuals should have rtimer > 0."
        i_recovered = np.nonzero(states == State.RECOVERED.value)[0]
        assert np.all(self.model.people.rtimer[i_recovered] > 0), "Recovered individuals should currently have rtimer > 0."

        return

    def postvalidate_step(self, tick: int) -> None:
        # Check that agents with state RECOVERED by patch match self.model.nodes.R[tick+1]
        nodeids = self.model.people.nodeid
        states = self.model.people.state
        assert np.all(
            self.model.nodes.R[tick + 1] == np.bincount(nodeids, states == State.RECOVERED.value, minlength=self.model.nodes.count)
        ), "Recovered census does not match recovered counts (by state)."
        # Don't need this since np.bincount can't (shouldn't) return negative counts
        # assert np.all(self.model.nodes.R[tick] >= 0), "Recovered counts must be non-negative"

        i_non_zero = np.nonzero(self.model.people.rtimer > 0)[0]
        assert np.all(states[i_non_zero] == State.RECOVERED.value), "Only recovered individuals should have rtimer > 0."
        i_recovered = np.nonzero(states == State.RECOVERED.value)[0]
        assert np.all(self.model.people.rtimer[i_recovered] > 0), "Recovered individuals should currently have rtimer > 0."

        return

    @staticmethod
    @nb.njit(
        # (nb.int8[:], nb.uint8[:], nb.uint32[:, :], nb.uint16[:]),
        nogil=True,
        parallel=True,
        cache=True,
    )
    def nb_recovered_step(states, rtimers, waned_by_node, nodeids):
        for i in nb.prange(len(states)):
            if states[i] == State.RECOVERED.value:
                rtimers[i] -= 1
                if rtimers[i] == 0:
                    states[i] = State.SUSCEPTIBLE.value
                    waned_by_node[nb.get_thread_id(), nodeids[i]] += 1

        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        # Propagate the number of recovered individuals in each patch
        # state(t+1) = state(t) + ∆state(t), initialize state(t+1) with state(t)
        self.model.nodes.R[tick + 1] = self.model.nodes.R[tick]

        waned_by_node = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.uint32)
        self.nb_recovered_step(
            self.model.people.state,
            self.model.people.rtimer,
            waned_by_node,
            self.model.people.nodeid,
        )
        waned_by_node = waned_by_node.sum(axis=0).astype(self.model.nodes.S.dtype)  # Sum over threads

        # state(t+1) = state(t) + ∆state(t)
        self.model.nodes.R[tick + 1] -= waned_by_node
        self.model.nodes.S[tick + 1] += waned_by_node
        # Record today's ∆
        self.model.nodes.waned[tick] = waned_by_node

        return

    def plot(self):
        # First plot: Recovered over Time by Node
        _fig, ax1 = plt.subplots()
        for node in range(self.model.nodes.count):
            ax1.plot(self.model.nodes.R[:, node], label=f"Node {node}")
        ax1.set_xlabel("Tick")
        ax1.set_ylabel("Recovered (by Node)")
        ax1.set_title("Recovered over Time by Node")
        ax1.legend(loc="upper left")
        plt.show()

        # Second plot: Total Recovered and Total Waned over Time
        _fig, ax2 = plt.subplots()
        total_recovered = np.sum(self.model.nodes.R, axis=1)
        total_waned = np.sum(self.model.nodes.waned, axis=1)
        ax2.plot(total_recovered, color="green", linestyle="--", label="Total Recovered")
        ax2.plot(total_waned, color="purple", linestyle="-.", label="Total Waned")
        ax2.set_xlabel("Tick")
        ax2.set_ylabel("Count")
        ax2.set_title("Total Recovered and Total Waned Over Time")
        ax2.legend(loc="upper right")
        plt.show()

        return


class TransmissionSIX:
    """Transmission component for a model S -> I and no recovery."""

    def __init__(self, model):
        self.model = model
        self.model.nodes.add_vector_property("forces", model.params.nticks + 1, dtype=np.float32)
        self.model.nodes.add_vector_property("incidence", model.params.nticks + 1, dtype=np.uint32)

        return

    @staticmethod
    @nb.njit(
        # (nb.int8[:], nb.uint16[:], nb.float32[:], nb.uint32[:, :]),
        nogil=True,
        parallel=True,
        cache=True,
    )
    def nb_transmission_step(states, nodeids, ft, inf_by_node):
        for i in nb.prange(len(states)):
            if states[i] == State.SUSCEPTIBLE.value:
                # Check for infection
                draw = np.random.rand()
                nid = nodeids[i]
                if draw < ft[nid]:
                    states[i] = State.INFECTIOUS.value
                    inf_by_node[nb.get_thread_id(), nid] += 1

        return

    def step(self, tick: int) -> None:
        ft = self.model.nodes.forces[tick]
        N = self.model.nodes.S[tick] + self.model.nodes.I[tick]
        ft[:] = self.model.params.beta * self.model.nodes.I[tick] / N
        transfer = ft[:, None] * self.model.network
        ft += transfer.sum(axis=0)
        ft -= transfer.sum(axis=1)
        ft = -np.expm1(-ft)  # Convert to probability of infection

        inf_by_node = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.uint32)
        self.nb_transmission_step(
            self.model.people.state,
            self.model.people.nodeid,
            ft,
            inf_by_node,
        )
        inf_by_node = inf_by_node.sum(axis=0)  # Sum over threads

        # state(t+1) = state(t) + ∆state(t)
        self.model.nodes.S[tick + 1] -= inf_by_node
        self.model.nodes.I[tick + 1] += inf_by_node
        # Record today's ∆
        self.model.nodes.incidence[tick] = inf_by_node

        return

    def plot(self):
        for node in range(self.model.nodes.count):
            plt.plot(self.model.nodes.forces[:-1, node], label=f"Node {node}")
        plt.xlabel("Tick")
        plt.ylabel("Force of Infection")
        plt.title("Force of Infection over Time by Node")
        plt.legend()
        plt.show()

        return


class TransmissionSI:
    def __init__(self, model, infdurdist, infdurmin=1):
        self.model = model
        self.model.nodes.add_vector_property("forces", model.params.nticks + 1, dtype=np.float32)
        self.model.nodes.add_vector_property("incidence", model.params.nticks + 1, dtype=np.uint32)

        self.infdurdist = infdurdist
        self.infdurmin = infdurmin

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
    def nb_transmission_step(states, nodeids, ft, inf_by_node, itimers, infdurdist, infdurmin, tick):
        for i in nb.prange(len(states)):
            if states[i] == State.SUSCEPTIBLE.value:
                # Check for infection
                draw = np.random.rand()
                nid = nodeids[i]
                if draw < ft[nid]:
                    states[i] = State.INFECTIOUS.value
                    itimers[i] = np.maximum(np.round(infdurdist(tick, nid)), infdurmin)  # Set the infection timer
                    inf_by_node[nb.get_thread_id(), nid] += 1

        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        ft = self.model.nodes.forces[tick]

        N = self.model.nodes.S[tick] + (I := self.model.nodes.I[tick])  # noqa: E741
        # Shouldn't be any exposed (E), because this is an S->I model
        # Might have R
        if hasattr(self.model.nodes, "R"):
            N += self.model.nodes.R[tick]

        ft[:] = self.model.params.beta * I / N
        transfer = ft[:, None] * self.model.network
        ft += transfer.sum(axis=0)
        ft -= transfer.sum(axis=1)
        ft = -np.expm1(-ft)  # Convert to probability of infection

        inf_by_node = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.uint32)
        self.nb_transmission_step(
            self.model.people.state,
            self.model.people.nodeid,
            ft,
            inf_by_node,
            self.model.people.itimer,
            self.infdurdist,
            self.infdurmin,
            tick,
        )
        inf_by_node = inf_by_node.sum(axis=0).astype(self.model.nodes.S.dtype)  # Sum over threads

        # state(t+1) = state(t) + ∆state(t)
        self.model.nodes.S[tick + 1] -= inf_by_node
        self.model.nodes.I[tick + 1] += inf_by_node
        # Record today's ∆
        self.model.nodes.incidence[tick] = inf_by_node

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


class TransmissionSE:
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
    def nb_transmission_step(states, nodeids, ft, exp_by_node, etimers, expdurdist, expdurmin, tick):
        for i in nb.prange(len(states)):
            if states[i] == State.SUSCEPTIBLE.value:
                # Check for infection
                draw = np.random.rand()
                nid = nodeids[i]
                if draw < ft[nid]:
                    states[i] = State.EXPOSED.value
                    etimers[i] = np.maximum(np.round(expdurdist(tick, nid)), expdurmin)  # Set the exposure timer
                    exp_by_node[nb.get_thread_id(), nid] += 1

        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        ft = self.model.nodes.forces[tick]

        N = self.model.nodes.S[tick] + self.model.nodes.E[tick] + (I := self.model.nodes.I[tick])  # noqa: E741
        # Might have R
        if hasattr(self.model.nodes, "R"):
            N += self.model.nodes.R[tick]

        ft[:] = self.model.params.beta * I / N
        transfer = ft[:, None] * self.model.network
        ft += transfer.sum(axis=0)
        ft -= transfer.sum(axis=1)
        ft = -np.expm1(-ft)  # Convert to probability of infection

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


class VitalDynamicsBase:
    def __init__(self, model, birthrates, pyramid, survival):
        self.model = model
        self.birthrates = birthrates
        self.pyramid = pyramid
        self.survival = survival

        # Date-Of-Birth and Date-Of-Death properties per agent
        self.model.people.add_property("dob", dtype=np.int16)
        self.model.people.add_property("dod", dtype=np.int16)
        # birth and death statistics per node
        self.model.nodes.add_vector_property("births", model.params.nticks + 1, dtype=np.uint32)
        self.model.nodes.add_vector_property("deaths", model.params.nticks + 1, dtype=np.uint32)

        # Initialize starting population
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

    def step(self, tick: int) -> None:
        raise NotImplementedError("VitalDynamicsBase is an abstract base class and cannot be stepped directly.")

    def _births(self, tick: int) -> None:
        rates = np.power(1.0 + self.model.birthrates[tick] / 1000, 1.0 / 365) - 1.0
        # Use "tomorrow's" population which accounts for mortality above.
        N = self.model.nodes.S[tick + 1] + self.model.nodes.I[tick + 1]
        births = np.round(np.random.poisson(rates * N)).astype(np.uint32)
        tbirths = births.sum()
        if tbirths > 0:
            istart, iend = self.model.people.add(tbirths)
            self.model.people.nodeid[istart:iend] = np.repeat(np.arange(self.model.nodes.count, dtype=np.uint16), births)
            # State.SUSCEPTIBLE.value is the default
            # self.model.people.state[istart:iend] = State.SUSCEPTIBLE.value

            dobs = self.model.people.dob[istart:iend]
            dods = self.model.people.dod[istart:iend]
            dobs[:] = tick
            sample_dods(dobs, dods, self.survival, tick=tick)

            # state(t+1) = state(t) + ∆state(t)
            self.model.nodes.S[tick + 1] += births
            # Record today's ∆
            self.model.nodes.births[tick] = births

        return

    def plot(self):
        _fig, ax1 = plt.subplots(figsize=(16, 9), dpi=200)
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


class VitalDynamicsSI(VitalDynamicsBase):
    def __init__(self, model, birthrates, pyramid, survival):
        super().__init__(model, birthrates, pyramid, survival)

        return

    @staticmethod
    @nb.njit(
        # (nb.int16[:], nb.int8[:], nb.uint16[:], nb.int32[:, :], nb.int32[:, :], nb.int32),
        nogil=True,
        parallel=True,
        cache=True,
    )
    def nb_process_deaths(dods, states, nodeids, delta_S, delta_I, tick):
        for i in nb.prange(len(dods)):
            if dods[i] == tick:
                if states[i] == State.SUSCEPTIBLE.value:
                    delta_S[nb.get_thread_id(), nodeids[i]] -= 1
                else:
                    delta_I[nb.get_thread_id(), nodeids[i]] -= 1
                states[i] = State.DECEASED.value

        return

    @validate(pre=VitalDynamicsBase.prevalidate_step, post=VitalDynamicsBase.postvalidate_step)
    def step(self, tick: int) -> None:
        # Do mortality first, then births
        delta_S = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        delta_I = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        self.nb_process_deaths(self.model.people.dod, self.model.people.state, self.model.people.nodeid, delta_S, delta_I, tick)
        # Combine thread results
        delta_S = delta_S.sum(axis=0).astype(self.model.nodes.S.dtype)
        delta_I = delta_I.sum(axis=0).astype(self.model.nodes.I.dtype)
        # state(t+1) = state(t) + ∆state(t)
        self.model.nodes.S[tick + 1] += delta_S  # delta_S is negative or zero
        self.model.nodes.I[tick + 1] += delta_I  # delta_I is negative or zero
        # Record today's ∆
        self.model.nodes.deaths[tick] = -(delta_S + delta_I)  # Record

        self._births(tick)

        return


class VitalDynamicsSIR(VitalDynamicsBase):
    def __init__(self, model, birthrates, pyramid, survival):
        super().__init__(model, birthrates, pyramid, survival)

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
        # (nb.int16[:], nb.int8[:], nb.uint16[:], nb.int32[:, :], nb.int32[:, :], nb.int32[:, :], nb.int32),
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

        self._births(tick)

        return


class VitalDynamicsSEIR(VitalDynamicsBase):
    def __init__(self, model, birthrates, pyramid, survival):
        super().__init__(model, birthrates, pyramid, survival)

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
        # (nb.int16[:], nb.int8[:], nb.uint16[:], nb.int32[:, :], nb.int32[:, :], nb.int32[:, :], nb.int32),
        nogil=True,
        parallel=True,
        cache=True,
    )
    def nb_process_deaths(dods, states, nodeids, deceased_S, deceased_E, deceased_I, deceased_R, tick):
        for i in nb.prange(len(dods)):
            if dods[i] == tick:
                state = states[i]
                if state >= 0:  # Ignore already deceased
                    if state == State.SUSCEPTIBLE.value:
                        deceased_S[nb.get_thread_id(), nodeids[i]] += 1
                    elif state == State.EXPOSED.value:
                        deceased_E[nb.get_thread_id(), nodeids[i]] += 1
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
        deceased_E = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        deceased_I = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        deceased_R = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        self.nb_process_deaths(
            self.model.people.dod, self.model.people.state, self.model.people.nodeid, deceased_S, deceased_E, deceased_I, deceased_R, tick
        )
        # Combine thread results
        deceased_S = deceased_S.sum(axis=0).astype(self.model.nodes.S.dtype)
        deceased_E = deceased_E.sum(axis=0).astype(self.model.nodes.E.dtype)
        deceased_I = deceased_I.sum(axis=0).astype(self.model.nodes.I.dtype)
        deceased_R = deceased_R.sum(axis=0).astype(self.model.nodes.R.dtype)
        # state(t+1) = state(t) + ∆state(t)
        self.model.nodes.S[tick + 1] -= deceased_S
        self.model.nodes.E[tick + 1] -= deceased_E
        self.model.nodes.I[tick + 1] -= deceased_I
        self.model.nodes.R[tick + 1] -= deceased_R
        # Record today's ∆
        self.model.nodes.deaths[tick] = deceased_S + deceased_E + deceased_I + deceased_R  # Record

        self._births(tick)

        return
