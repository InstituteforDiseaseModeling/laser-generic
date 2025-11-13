import laser.core.distributions as dists
import matplotlib.pyplot as plt
import numba as nb
import numpy as np

from laser.generic.newutils import validate

from .shared import State
from .shared import sample_dobs
from .shared import sample_dods


class Susceptible:
    """
    Simple Susceptible component suitable for all models (SI/SIS/SIR/SIRS/SEIR/SEIRS).
    """

    def __init__(self, model):
        self.model = model
        self.model.people.add_scalar_property("nodeid", dtype=np.uint16)
        self.model.people.add_scalar_property("state", dtype=np.int8, default=State.SUSCEPTIBLE.value)
        self.model.nodes.add_vector_property("S", model.params.nticks + 1, dtype=np.int32)

        self.model.people.nodeid[:] = np.repeat(np.arange(self.model.nodes.count, dtype=np.uint16), self.model.scenario.population)
        self.model.nodes.S[0] = self.model.scenario.S

        return

    def prevalidate_step(self, tick: int) -> None:
        _check_flow_vs_census(self.model.nodes.S[tick], self.model.people, State.SUSCEPTIBLE, "Susceptible")

        return

    def postvalidate_step(self, tick: int) -> None:
        _check_flow_vs_census(self.model.nodes.S[tick + 1], self.model.people, State.SUSCEPTIBLE, "Susceptible")
        _check_unchanged(self.model.nodes.S[tick], self.model.nodes.S[tick + 1], "Susceptible counts")

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
    """
    Simple Exposed component for an SEIR/SEIRS model - includes incubation period.

    Agents transition from Exposed to Infectious when their incubation timer (etimer) expires.
    Tracks number of agents becoming infectious each tick in `model.nodes.symptomatic`.
    """

    def __init__(self, model, expdurdist, infdurdist, expdurmin=1, infdurmin=1):
        self.model = model
        self.model.people.add_scalar_property("etimer", dtype=np.uint16)
        self.model.nodes.add_vector_property("E", model.params.nticks + 1, dtype=np.int32)
        self.model.nodes.add_vector_property("symptomatic", model.params.nticks + 1, dtype=np.int32)

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
                assert np.all(self.model.people.etimer[i_exposed] > 0), (
                    f"Exposed individuals should have etimer > 0 ({self.model.people.etimer[i_exposed].min()=})"
                )

        return

    def prevalidate_step(self, tick: int) -> None:
        _check_flow_vs_census(self.model.nodes.E[tick], self.model.people, State.EXPOSED, "Exposed")
        _check_timer_active(self.model.people.state, State.EXPOSED.value, self.model.people.etimer, "Exposed", "etimer")
        _check_state_timer_consistency(self.model.people.state, State.EXPOSED.value, self.model.people.etimer, "Exposed", "etimer")

        alive = self.model.people.state != State.DECEASED.value
        self.etimers_one = (alive) & (self.model.people.etimer == 1)

        return

    def postvalidate_step(self, tick: int) -> None:
        _check_flow_vs_census(self.model.nodes.E[tick + 1], self.model.people, State.EXPOSED, "Exposed")
        _check_timer_active(self.model.people.state, State.EXPOSED.value, self.model.people.etimer, "Exposed", "etimer")
        _check_state_timer_consistency(self.model.people.state, State.EXPOSED.value, self.model.people.etimer, "Exposed", "etimer")

        assert np.all(self.model.people.state[self.etimers_one] == State.INFECTIOUS.value), (
            "Individuals with etimer == 1 before should now be infectious."
        )
        assert np.all(self.model.people.etimer[self.etimers_one] == 0), "Individuals with etimer == 1 before should now have etimer == 0."
        assert np.all(self.model.people.itimer[self.etimers_one] > 0), "Individuals with etimer == 1 before should now have itimer > 0."

        return

    @staticmethod
    @nb.njit(
        # (nb.int8[:], nb.uint8[:], nb.uint8[:], nb.uint32[:, :], nb.uint16[:], nb.types.FunctionType(nb.types.uint8()), nb.int32),
        nogil=True,
        parallel=True,
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

        symptomatic_by_node = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
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
    """
    Infectious component for an SI model - no recovery.

    Agents remain in the Infectious state indefinitely (no recovery).
    """

    def __init__(self, model):
        self.model = model
        self.model.nodes.add_vector_property("I", model.params.nticks + 1, dtype=np.int32)

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

    def prevalidate_step(self, tick: int) -> None:
        _check_flow_vs_census(self.model.nodes.I[tick], self.model.people, State.INFECTIOUS, "Infectious")

        return

    def postvalidate_step(self, tick: int) -> None:
        _check_flow_vs_census(self.model.nodes.I[tick + 1], self.model.people, State.INFECTIOUS, "Infectious")
        _check_unchanged(self.model.nodes.I[tick], self.model.nodes.I[tick + 1], "Infected counts")

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
    """
    Infectious component for an SIS model - includes infectious duration.

    Agents transition from Infectious back to Susceptible after the infectious period (itimer).
    Tracks number of agents recovering each tick in `model.nodes.recovered`.
    """

    def __init__(self, model, infdurdist, infdurmin=1):
        self.model = model
        self.model.people.add_scalar_property("itimer", dtype=np.uint16)
        self.model.nodes.add_vector_property("I", model.params.nticks + 1, dtype=np.int32)
        self.model.nodes.add_vector_property("recovered", model.params.nticks + 1, dtype=np.int32)

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
                assert np.all(self.model.people.itimer[i_infectious] > 0), (
                    f"Infected individuals should have itimer > 0 ({self.model.people.itimer[i_infectious].min()=})"
                )

        self.model.nodes.I[0] = self.model.scenario.I
        assert np.all(self.model.nodes.S[0] + self.model.nodes.I[0] == self.model.scenario.population), (
            "Initial S + I does not equal population"
        )

        return

    def prevalidate_step(self, tick: int) -> None:
        _check_flow_vs_census(self.model.nodes.I[tick], self.model.people, State.INFECTIOUS, "Infectious")
        _check_timer_active(self.model.people.state, State.INFECTIOUS.value, self.model.people.itimer, "Infectious", "itimer")
        _check_state_timer_consistency(self.model.people.state, State.INFECTIOUS.value, self.model.people.itimer, "Infectious", "itimer")

        return

    def postvalidate_step(self, tick: int) -> None:
        _check_flow_vs_census(self.model.nodes.I[tick + 1], self.model.people, State.INFECTIOUS, "Infectious")
        _check_timer_active(self.model.people.state, State.INFECTIOUS.value, self.model.people.itimer, "Infectious", "itimer")
        _check_state_timer_consistency(self.model.people.state, State.INFECTIOUS.value, self.model.people.itimer, "Infectious", "itimer")

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

        recovered_by_node = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
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
    """
    Infectious component for an SIR/SEIR model - includes infectious duration, no waning immunity in recovered state.

    Agents transition from Infectious to Recovered after the infectious period (itimer).
    Tracks number of agents recovering each tick in `model.nodes.recovered`.
    """

    def __init__(self, model, infdurdist, infdurmin=1):
        self.model = model
        self.model.people.add_scalar_property("itimer", dtype=np.uint16)
        self.model.nodes.add_vector_property("I", model.params.nticks + 1, dtype=np.int32)
        self.model.nodes.add_vector_property("recovered", model.params.nticks + 1, dtype=np.int32)

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
                assert np.all(self.model.people.itimer[i_infectious] > 0), (
                    f"Infected individuals should have itimer > 0 ({self.model.people.itimer[i_infectious].min()=})"
                )

        return

    def prevalidate_step(self, tick: int) -> None:
        _check_flow_vs_census(self.model.nodes.I[tick], self.model.people, State.INFECTIOUS, "Infectious")
        _check_timer_active(self.model.people.state, State.INFECTIOUS.value, self.model.people.itimer, "Infectious", "itimer")
        _check_state_timer_consistency(self.model.people.state, State.INFECTIOUS.value, self.model.people.itimer, "Infectious", "itimer")

        alive = self.model.people.state != State.DECEASED.value
        self.itimers_one = (alive) & (self.model.people.itimer == 1)

        return

    def postvalidate_step(self, tick: int) -> None:
        _check_flow_vs_census(self.model.nodes.I[tick + 1], self.model.people, State.INFECTIOUS, "Infectious")
        _check_timer_active(self.model.people.state, State.INFECTIOUS.value, self.model.people.itimer, "Infectious", "itimer")
        _check_state_timer_consistency(self.model.people.state, State.INFECTIOUS.value, self.model.people.itimer, "Infectious", "itimer")

        assert np.all(self.model.people.state[self.itimers_one] == State.RECOVERED.value), (
            "Individuals with itimer == 1 before should now be recovered."
        )
        assert np.all(self.model.people.itimer[self.itimers_one] == 0), "Individuals with itimer == 1 before should now have itimer == 0."

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

        recovered_by_node = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
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
    """
    Infectious component for an SIRS/SEIRS model - includes infectious duration and waning immunity.

    Agents transition from Infectious to Recovered after the infectious period (itimer).
    Set the waning immunity timer (rtimer) upon recovery.
    Tracks number of agents recovering each tick in `model.nodes.recovered`.
    """

    def __init__(self, model, infdurdist, wandurdist, infdurmin=1, wandurmin=1):
        self.model = model
        self.model.people.add_scalar_property("itimer", dtype=np.uint16)
        if not hasattr(self.model.people, "rtimer"):
            self.model.people.add_scalar_property("rtimer", dtype=np.uint16)
        self.model.nodes.add_vector_property("I", model.params.nticks + 1, dtype=np.int32)
        self.model.nodes.add_vector_property("recovered", model.params.nticks + 1, dtype=np.int32)

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
                assert np.all(self.model.people.itimer[i_infectious] > 0), (
                    f"Infected individuals should have itimer > 0 ({self.model.people.itimer[i_infectious].min()=})"
                )

        return

    def prevalidate_step(self, tick: int) -> None:
        _check_flow_vs_census(self.model.nodes.I[tick], self.model.people, State.INFECTIOUS, "Infectious")
        _check_timer_active(self.model.people.state, State.INFECTIOUS.value, self.model.people.itimer, "Infectious", "itimer")
        _check_state_timer_consistency(self.model.people.state, State.INFECTIOUS.value, self.model.people.itimer, "Infectious", "itimer")

        return

    def postvalidate_step(self, tick: int) -> None:
        _check_flow_vs_census(self.model.nodes.I[tick + 1], self.model.people, State.INFECTIOUS, "Infectious")
        _check_timer_active(self.model.people.state, State.INFECTIOUS.value, self.model.people.itimer, "Infectious", "itimer")
        _check_state_timer_consistency(self.model.people.state, State.INFECTIOUS.value, self.model.people.itimer, "Infectious", "itimer")

        return

    @staticmethod
    @nb.njit(
        # (nb.int8[:], nb.uint8[:], nb.uint8[:], nb.uint32[:, :], nb.uint16[:], nb.types.FunctionType(nb.types.uint8()), min),
        nogil=True,
        parallel=True,
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

        recovered_by_node = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
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
    """
    Simple Recovered component for an SIR/SEIR model - no waning immunity.

    Agents remain in the Recovered state indefinitely (no waning immunity).
    """

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
        _check_flow_vs_census(self.model.nodes.R[tick], self.model.people, State.RECOVERED, "Recovered")

        return

    def postvalidate_step(self, tick: int) -> None:
        _check_flow_vs_census(self.model.nodes.R[tick + 1], self.model.people, State.RECOVERED, "Recovered")
        _check_unchanged(self.model.nodes.R[tick], self.model.nodes.R[tick + 1], "Recovered counts")

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
    """
    Recovered component for an SIRS/SEIRS model - includes waning immunity.

    Agents transition from Recovered back to Susceptible after the waning immunity period (rtimer).
    Tracks number of agents losing immunity each tick in `model.nodes.waned`.
    """

    def __init__(self, model, wandurdist, wandurmin=1):
        self.model = model
        if not hasattr(self.model.people, "rtimer"):
            self.model.people.add_scalar_property("rtimer", dtype=np.uint16)
        self.model.nodes.add_vector_property("R", model.params.nticks + 1, dtype=np.int32)
        self.model.nodes.add_vector_property("waned", model.params.nticks + 1, dtype=np.int32)

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
                assert np.all(self.model.people.rtimer[i_recovered] > 0), (
                    f"Recovered individuals should have rtimer > 0 ({self.model.people.rtimer[i_recovered].min()=})"
                )

        return

    def prevalidate_step(self, tick: int) -> None:
        _check_flow_vs_census(self.model.nodes.S[tick], self.model.people, State.SUSCEPTIBLE, "Susceptible")
        _check_flow_vs_census(self.model.nodes.R[tick], self.model.people, State.RECOVERED, "Recovered")
        _check_timer_active(self.model.people.state, State.RECOVERED.value, self.model.people.rtimer, "Recovered", "rtimer")
        _check_state_timer_consistency(self.model.people.state, State.RECOVERED.value, self.model.people.rtimer, "Recovered", "rtimer")

        alive = self.model.people.state != State.DECEASED.value
        self.rtimers_one = (alive) & (self.model.people.rtimer == 1)

        return

    def postvalidate_step(self, tick: int) -> None:
        _check_flow_vs_census(self.model.nodes.S[tick + 1], self.model.people, State.SUSCEPTIBLE, "Susceptible")
        _check_flow_vs_census(self.model.nodes.R[tick + 1], self.model.people, State.RECOVERED, "Recovered")
        _check_timer_active(self.model.people.state, State.RECOVERED.value, self.model.people.rtimer, "Recovered", "rtimer")
        _check_state_timer_consistency(self.model.people.state, State.RECOVERED.value, self.model.people.rtimer, "Recovered", "rtimer")

        assert np.all(self.model.people.state[self.rtimers_one] == State.SUSCEPTIBLE.value), (
            "Individuals with rtimer == 1 before should now be susceptible."
        )

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

        waned_by_node = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
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
    """
    Transmission component for a model S -> I and no recovery.

    Agents transition from Susceptible to Infectious based on force of infection.
    Tracks number of new infections each tick in `model.nodes.incidence`.
    """

    def __init__(self, model):
        self.model = model
        self.model.nodes.add_vector_property("forces", model.params.nticks + 1, dtype=np.float32)
        self.model.nodes.add_vector_property("incidence", model.params.nticks + 1, dtype=np.int32)

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

    def prevalidate_step(self, tick: int) -> None:
        _check_flow_vs_census(self.model.nodes.S[tick], self.model.people, State.SUSCEPTIBLE, "Susceptible")
        _check_flow_vs_census(self.model.nodes.I[tick], self.model.people, State.INFECTIOUS, "Infectious")

        return

    def postvalidate_step(self, tick: int) -> None:
        _check_flow_vs_census(self.model.nodes.S[tick + 1], self.model.people, State.SUSCEPTIBLE, "Susceptible")
        _check_flow_vs_census(self.model.nodes.I[tick + 1], self.model.people, State.INFECTIOUS, "Infectious")

        I = self.model.nodes.I  # noqa: E741
        assert np.all(self.model.nodes.incidence[tick] == (I[tick + 1] - I[tick])), "Incidence does not match change in Infectious counts"

        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        ft = self.model.nodes.forces[tick]
        N = self.model.nodes.S[tick] + self.model.nodes.I[tick]
        ft[:] = self.model.params.beta * self.model.nodes.I[tick] / N
        transfer = ft[:, None] * self.model.network
        ft += transfer.sum(axis=0)
        ft -= transfer.sum(axis=1)
        ft = -np.expm1(-ft)  # Convert to probability of infection

        inf_by_node = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
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
    """
    Transmission component for an SIS/SIR/SIRS model with S -> I transition and infectious duration.

    Agents transition from Susceptible to Infectious based on force of infection.
    Sets newly infectious agents' infection timers (itimer) based on `infdurdist` and `infdurmin`.
    Tracks number of new infections each tick in `model.nodes.incidence`.
    """

    def __init__(self, model, infdurdist, infdurmin=1):
        """
        Initializes the TransmissionSI component.

        Args:
            model: The epidemiological model instance.
            infdurdist: A function that returns the infectious duration for a given tick and node.
            infdurmin: Minimum infectious duration.
        """
        self.model = model
        self.model.nodes.add_vector_property("forces", model.params.nticks + 1, dtype=np.float32)
        self.model.nodes.add_vector_property("incidence", model.params.nticks + 1, dtype=np.int32)

        self.infdurdist = infdurdist
        self.infdurmin = infdurmin

        return

    def prevalidate_step(self, tick: int) -> None:
        # Check ahead because I->R and R->S transitions may have happened meaning S[tick] and I[tick] are "out of date"
        _check_flow_vs_census(self.model.nodes.S[tick + 1], self.model.people, State.SUSCEPTIBLE, "Susceptible")
        _check_flow_vs_census(self.model.nodes.I[tick + 1], self.model.people, State.INFECTIOUS, "Infectious")

        self.prv_inext = self.model.nodes.I[tick + 1].copy()

        return

    def postvalidate_step(self, tick: int) -> None:
        _check_flow_vs_census(self.model.nodes.S[tick + 1], self.model.people, State.SUSCEPTIBLE, "Susceptible")
        _check_flow_vs_census(self.model.nodes.I[tick + 1], self.model.people, State.INFECTIOUS, "Infectious")

        Inext = self.model.nodes.I[tick + 1]
        assert np.all(self.model.nodes.incidence[tick] == (Inext - self.prv_inext)), "Incidence does not match change in Infectious counts"

        return

    @staticmethod
    @nb.njit(nogil=True, parallel=True)
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

        inf_by_node = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
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
    """
    Transmission component for an SIER/SIERS model with S -> E transition and incubation duration.

    Agents transition from Susceptible to Exposed based on force of infection.
    Sets newly exposed agents' infection timers (etimer) based on `expdurdist` and `expdurmin`.
    Tracks number of new infections each tick in `model.nodes.incidence`.
    """

    def __init__(self, model, expdurdist, expdurmin=1):
        """
        Initializes the TransmissionSE component.

        Args:
            model: The epidemiological model instance.
            expdurdist: A function that returns the incubation duration for a given tick and node.
            expdurmin: Minimum incubation duration.
        """
        self.model = model
        self.model.nodes.add_vector_property("forces", model.params.nticks + 1, dtype=np.float32)
        self.model.nodes.add_vector_property("incidence", model.params.nticks + 1, dtype=np.int32)

        self.expdurdist = expdurdist
        self.expdurmin = expdurmin

        return

    def prevalidate_step(self, tick: int) -> None:
        # Check ahead because E->I and R->S transitions may have happened meaning S[tick] and E[tick] are "out of date"
        _check_flow_vs_census(self.model.nodes.S[tick + 1], self.model.people, State.SUSCEPTIBLE, "Susceptible")
        _check_flow_vs_census(self.model.nodes.E[tick + 1], self.model.people, State.EXPOSED, "Exposed")

        self.prv_enext = self.model.nodes.E[tick + 1].copy()

        return

    def postvalidate_step(self, tick: int) -> None:
        _check_flow_vs_census(self.model.nodes.S[tick + 1], self.model.people, State.SUSCEPTIBLE, "Susceptible")
        _check_flow_vs_census(self.model.nodes.E[tick + 1], self.model.people, State.EXPOSED, "Exposed")

        Enext = self.model.nodes.E[tick + 1]
        assert np.all(self.model.nodes.incidence[tick] == (Enext - self.prv_enext)), "Incidence does not match change in Exposed counts"

        return

    @staticmethod
    @nb.njit(nogil=True, parallel=True)
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

        exp_by_node = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
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
    """
    Base class for Vital Dynamics components (births and deaths).

    Args:
        model: The epidemiological model instance.
        birthrates: Array of birth rates, CBR, in effect for each tick and node.
        pyramid (AliasedDistribution): Age pyramid distribution for sampling date of birth.
        survival (KaplanMeierEstimator): Survival curve for sampling date of death.
        states: List of states to consider for population counts (default: ["S", "E", "I", "R"]).
    """

    def __init__(self, model, birthrates, pyramid, survival, states=None):
        """
        Initializes the VitalDynamicsBase component.

        Args:

            model: The epidemiological model instance.
            birthrates: Array of birth rates, CBR, in effect for each tick and node.
            pyramid (AliasedDistribution): Age pyramid distribution for sampling date of birth.
            survival (KaplanMeierEstimator): Survival curve for sampling date of death.
            states: List of states to consider for population counts (default: ["S", "E", "I", "R"]).
        """
        self.model = model
        self.birthrates = birthrates
        self.pyramid = pyramid
        self.survival = survival
        self.states = states or ["S", "E", "I", "R"]

        # Date-Of-Birth and Date-Of-Death properties per agent
        self.model.people.add_property("dob", dtype=np.int16)
        self.model.people.add_property("dod", dtype=np.int16)
        # birth and death statistics per node
        self.model.nodes.add_vector_property("births", model.params.nticks + 1, dtype=np.int32)
        self.model.nodes.add_vector_property("deaths", model.params.nticks + 1, dtype=np.int32)

        # Initialize starting population
        dobs = self.model.people.dob[0 : self.model.people.count]
        dods = self.model.people.dod[0 : self.model.people.count]
        sample_dobs(dobs, self.pyramid, tick=0)
        sample_dods(dobs, dods, self.survival, tick=0)

        return

    def prevalidate_step(self, tick: int) -> None:
        self.prv_count = self.model.people.count
        self.prv_dead = self.model.people.state == State.DECEASED.value

        return

    def postvalidate_step(self, tick: int) -> None:
        nbirths = self.model.nodes.births[tick].sum()
        assert self.model.people.count == self.prv_count + nbirths, "Population count mismatch after births"

        istart = self.prv_count
        iend = self.model.people.count
        birth_counts = np.bincount(self.model.people.nodeid[istart:iend], minlength=self.model.nodes.count)
        assert np.all(birth_counts == self.model.nodes.births[tick]), "Birth counts by patch mismatch"
        assert np.all(self.model.people.state[istart:iend] == State.SUSCEPTIBLE.value), "Newborns should be susceptible"

        if hasattr(self.model.people, "itimer"):
            assert np.all(self.model.people.itimer[istart:iend] == 0), "Newborns should have itimer == 0"

        ndeaths = self.model.nodes.deaths[tick].sum()
        deceased = self.model.people.state == State.DECEASED.value
        assert ndeaths == deceased.sum() - self.prv_dead.sum(), "Death counts mismatch"

        prv = np.bincount(self.model.people.nodeid[0 : self.prv_count][self.prv_dead], minlength=self.model.nodes.count)
        now = np.bincount(self.model.people.nodeid[deceased], minlength=self.model.nodes.count)
        death_counts = now - prv
        assert np.all(death_counts == self.model.nodes.deaths[tick]), "Death counts by patch mismatch"

        # TODO - check flow delta against nbirths and ndeaths
        previous_N = np.zeros(len(self.model.scenario), dtype=np.int32)
        for state in self.states:
            if (pop := getattr(self.model.nodes, state, None)) is not None:
                previous_N += pop[tick]
        expected_N = previous_N + self.model.nodes.births[tick] - self.model.nodes.deaths[tick]
        actual_N = 0
        for state in self.states:
            if (pop := getattr(self.model.nodes, state, None)) is not None:
                actual_N += pop[tick + 1]
        assert np.all(expected_N == actual_N), "Population counts by state mismatch after births and deaths"

        return

    def step(self, tick: int) -> None:
        raise NotImplementedError("VitalDynamicsBase is an abstract base class and cannot be stepped directly.")

    def _births(self, tick: int) -> None:
        rates = np.power(1.0 + self.birthrates[tick] / 1000, 1.0 / 365) - 1.0
        # Use "tomorrow's" population which accounts for mortality above.
        N = np.zeros(len(self.model.scenario), dtype=np.int32)
        for state in self.states:
            if (pop := getattr(self.model.nodes, state, None)) is not None:
                N += pop[tick + 1]
        births = np.round(np.random.poisson(rates * N)).astype(np.int32)
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

        for component in self.model.components:
            if hasattr(component, "on_birth") and callable(component.on_birth):
                # TODO - account for time here in TimingStatistics
                component.on_birth(istart, iend, tick)

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
    """
    Vital dynamics component for an SI/SIS model with just S and I states - includes births and deaths.
    """

    def __init__(self, model, birthrates, pyramid, survival):
        super().__init__(model, birthrates, pyramid, survival)

        return

    @staticmethod
    @nb.njit(nogil=True, parallel=True, cache=True)
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
    """
    Vital dynamics component for an SIR/SIRS model with S, I, and R states - includes births and deaths.
    """

    def __init__(self, model, birthrates, pyramid, survival):
        super().__init__(model, birthrates, pyramid, survival)

        return

    @staticmethod
    @nb.njit(nogil=True, parallel=True, cache=True)
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

    @validate(pre=VitalDynamicsBase.prevalidate_step, post=VitalDynamicsBase.postvalidate_step)
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
    """
    Vital dynamics component for an SEIR/SEIRS model with S, E, I, and R states - includes births and deaths.
    """

    def __init__(self, model, birthrates, pyramid, survival):
        super().__init__(model, birthrates, pyramid, survival)

        return

    @staticmethod
    @nb.njit(nogil=True, parallel=True, cache=True)
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

    @validate(pre=VitalDynamicsBase.prevalidate_step, post=VitalDynamicsBase.postvalidate_step)
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


## Validation helper functions


def _check_flow_vs_census(flow, people, state, name):
    """Compare a given flow vector against the census counts by state."""
    assert np.all(flow == (_actual := np.bincount(people.nodeid, people.state == state.value, len(flow)))), (
        f"{name} census does not match {name} counts (by state)."
    )
    return


def _check_unchanged(previous, current, name):
    """Check that a given array is unchanged after a step."""
    assert np.all(current == previous), f"{name} should be unchanged after step()."
    return


def _check_timer_active(states, value, timers, state_name, timer_name):
    """Check that individuals in a given state have active (greater than zero) timers."""
    assert np.all(_test := (timers[states == value] > 0)), (
        f"{state_name} individuals should have {timer_name} > 0 ({timers[states == value].min()=})"
    )
    return


def _check_state_timer_consistency(states, value, timers, state_name, timer_name):
    """Check that only live individuals in a given state have active (greater than zero) timers."""
    alive = states != State.DECEASED.value
    active = timers > 0
    assert np.all(_test := (states[alive & active] == value)), f"Only {state_name} individuals should have {timer_name} > 0."
    return
