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
    Susceptible Component for Patch-Based Agent-Based Models (S, SI, SIS, SIR, SEIR, etc.)

    This component initializes and tracks the count of susceptible individuals (`S`) in
    a spatially structured agent-based model. It is compatible with all standard LASER
    disease progression models that include a "susceptible" state.

    Responsibilities:
    - Initializes agent-level properties:
        • `nodeid`: Patch ID of each agent (uint16)
        • `state`: Infection state (int8), defaulting to `State.SUSCEPTIBLE`
    - Initializes node-level property:
        • `S[t, i]`: Susceptible count in node `i` at time `t`
    - At each timestep, propagates the susceptible count forward (`S[t+1] = S[t]`),
      unless modified by other components (e.g., exposure, births).
    - Validates consistency between patch-level susceptible counts and agent-level state.

    Usage:
    Add this component early in the component list for any model with SUSCEPTIBLE agents,
    typically before transmission or exposure components. Compatible with:
        - `SIR.Transmission`
        - `SIR.Exposure`
        - `SIR.Infectious`
        - `SIR.Recovered`
        - Custom SEIRS extensions

    Requires:
    - `model.people`: A LaserFrame for all agents
    - `model.nodes`: Patch-level state
    - `model.scenario`: Input DataFrame with `population` and optionally `S` columns
    - `model.params.nticks`: Number of simulation ticks

    Validation:
    - Ensures consistency of susceptible counts before and after each step
    - Prevents unintentional state drift by validating against agent `state` values

    Output:
    - `model.nodes.S`: A `(nticks+1, num_nodes)` array of susceptible counts
    - Optional plotting via `plot()` for visual inspection of per-node and total `S`

    Step Behavior:
        For tick t:
            S[t+1] = S[t]     # Unless explicitly modified by other components

    This component does not alter agent states directly but serves as a synchronized
    counter and validator of susceptible individuals.

    Example:
        model.components = [
            SIR.Susceptible(model),
            SIR.Transmission(model, ...),
            SIR.Exposure(model),
            SIR.Infectious(model, ...),
            SIR.Recovered(model),
        ]
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
    Exposed Component for SEIR/SEIRS Models with Explicit Incubation Period

    This component handles the incubation phase in models where agents must transition from
    an 'exposed' (E) state to 'infectious' (I) after a delay. It supports custom incubation
    and infectious duration distributions and handles both initialization and per-tick dynamics.

    Responsibilities:
    - Initializes exposed individuals at time 0 (if provided in the scenario)
    - Assigns and tracks per-agent incubation timers (`etimer`)
    - Transitions agents from `EXPOSED` to `INFECTIOUS` when `etimer == 0`
    - Assigns new infection timers (`itimer`) upon becoming infectious
    - Updates patch-level EXPOSED (`E`) and INFECTIOUS case counts
    - Provides validation hooks for state and timer consistency

    Required Inputs:
    - `model.scenario.E`: initial count of exposed individuals per node (optional)
    - `expdurdist`: callable returning sampled incubation durations
    - `infdurdist`: callable returning sampled infectious durations
    - `expdurmin`: minimum incubation period (default 1 day)
    - `infdurmin`: minimum infectious period (default 1 day)

    Outputs:
    - `model.people.etimer`: agent-level incubation timer
    - `model.nodes.E[t, i]`: number of exposed individuals at time `t` in node `i`
    - `model.nodes.newly_infectious[t, i]`: number of newly infectious cases per node per day

    Validation:
    - Ensures consistency between individual states and `etimer` values
    - Ensures that agents becoming infectious have valid `itimer` values assigned
    - Prevents agents with expired `etimer` from remaining in EXPOSED state

    Step Behavior:
        For each agent:
            - Decrease `etimer`
            - If `etimer == 0`, change state to `INFECTIOUS` and assign `itimer`
            - Update `model.nodes.E` and `model.nodes.I` counts accordingly

    Plotting:
    The `plot()` method provides a time series of exposed individuals per node and total across all nodes.

    Example:
        model.components = [
            SIR.Susceptible(model),
            Exposed(model, expdurdist, infdurdist),
            SIR.Infectious(model, infdurdist),
            ...
        ]
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
    Infectious Component for SI Models (No Recovery)

    This component manages the infectious state in SI-style epidemic models where agents
    remain infectious indefinitely. It is appropriate for use in models without a recovered
    or removed state (i.e., no `R` compartment).

    Responsibilities:
    - Initializes agents as infectious based on `model.scenario.I`
    - Tracks the number of infectious individuals (`I`) in each patch over time
    - Maintains per-tick, per-node counts in `model.nodes.I`
    - Validates consistency between agent states and patch-level totals

    Required Inputs:
    - `model.scenario.I`: array of initial infected counts per patch
    - `model.people.state`: infection state per agent
    - `model.people.nodeid`: patch assignment per agent
    - `model.params.nticks`: number of timesteps to simulate

    Outputs:
    - `model.nodes.I[t, i]`: number of infectious individuals in node `i` at time `t`

    Step Behavior:
        For each timestep `t`, this component copies:
            I[t+1] = I[t]
        (No recovery or removal; new infections may be added externally.)

    Validation:
    - Ensures that patch-level infectious counts (`model.nodes.I`) match the agent-level state
    - Asserts that the sum of `S` and `I` matches total population at initialization
    - Validates that infected counts do not change unexpectedly (unless altered by another component)

    Plotting:
    The `plot()` method shows the number of infectious agents per patch and in total across time.

    Example:
        model.components = [
            SIR.Susceptible(model),
            InfectiousSI(model),
            SIR.Transmission(model, ...),
        ]
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
    Infectious Component for SIS Models (Infection + Recovery to Susceptible)

    This component handles the infectious state in SIS-style models, where agents
    recover from infection and immediately return to the susceptible pool. It supports
    per-agent infection durations and manages patch-level infectious counts over time.

    Responsibilities:
    - Initializes infected agents and their infection timers (`itimer`)
    - Decrements `itimer` daily for infectious agents
    - Automatically transitions agents from `INFECTIOUS` to `SUSCEPTIBLE` when `itimer == 0`
    - Tracks per-day recoveries at the node level in `model.nodes.recovered`
    - Maintains node-level `I` and `S` counts with full timestep resolution

    Required Inputs:
    - `model.scenario.I`: initial number of infectious agents per patch
    - `infdurdist`: a callable function which samples the infectious duration distribution
    - `infdurmin`: the minimum infection period (default = 1 time step)

    Outputs:
    - `model.people.itimer`: per-agent infection countdown timer
    - `model.nodes.I[t, i]`: number of infectious individuals at tick `t` in node `i`
    - `model.nodes.recovered[t, i]`: number of recoveries at tick `t` in node `i`

    Step Behavior:
        At each tick:
        - Infectious agents decrement their `itimer`
        - Agents with `itimer == 0` are transitioned back to susceptible
        - `model.nodes.I` is updated accordingly
        - Recovered counts are recorded in `model.nodes.recovered`

    Validation:
    - Ensures consistency between agent `state` and infection timer (`itimer`)
    - Validates `I` census against agent-level state before and after each tick

    Plotting:
    The `plot()` method displays both per-node and total infectious counts over time.

    Example:
        model.components = [
            SIR.Susceptible(model),
            InfectiousIS(model, infdurdist),
            SIR.Transmission(model, ...),
        ]
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
    Infectious Component for SIR/SEIR Models (With Recovery to Immune)

    This component manages agents in the infectious state for models where infected individuals
    recover permanently (i.e., transition to a `RECOVERED` state without waning). It supports
    agent-level infection durations and patch-level tracking of recoveries over time.

    Responsibilities:
    - Initializes infected agents and their infection timers (`itimer`) based on scenario input
    - Decrements `itimer` daily for infectious agents
    - Transitions agents from `INFECTIOUS` to `RECOVERED` when `itimer == 0`
    - Updates patch-level state variables:
        • `I[t, i]`: infectious count at tick `t` in node `i`
        • `R[t, i]`: recovered count
        • `recovered[t, i]`: number of recoveries during tick `t`

    Required Inputs:
    - `model.scenario.I`: number of initially infected individuals per patch
    - `infdurdist`: function returning infection durations
    - `infdurmin`: minimum infectious period (default = 1 day)

    Outputs:
    - `model.people.itimer`: countdown timers per agent
    - `model.nodes.I[t]`, `.R[t]`: infectious and recovered counts per patch
    - `model.nodes.recovered[t]`: daily recoveries per patch

    Step Behavior:
    - Infectious agents decrement `itimer`
    - When `itimer == 0`, agent state is set to `RECOVERED`
    - Patch-level `I` and `R` are updated; `recovered` logs today's transitions

    Validation:
    - Ensures internal consistency between agent state and timer
    - Confirms agents with `itimer == 1` recover exactly one day later
    - Validates population conservation (`S + I + R = N`)

    Plotting:
    The `plot()` method shows per-node and total infectious counts across time.

    Example:
        model.components = [
            SIR.Susceptible(model),
            InfectiousIR(model, infdurdist),
            SIR.Recovered(model),
            ...
        ]
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
    Infectious Component for SIRS/SEIRS Models (Recovery with Waning Immunity)

    This component manages infectious individuals in models where recovery confers
    temporary immunity, after which agents become susceptible again (SIRS/SEIRS).

    Responsibilities:
    - Initializes infectious agents from `model.scenario.I`
    - Assigns and tracks infectious timers (`itimer`) per agent
    - Transitions agents from `INFECTIOUS` to `RECOVERED` when `itimer == 0`
    - Assigns a waning immunity timer (`rtimer`) upon recovery
    - Updates patch-level state:
        • `I[t, i]`: current infectious count
        • `R[t, i]`: current recovered count
        • `recovered[t, i]`: number of agents recovering on tick `t`

    Required Inputs:
    - `model.scenario.I`: number of initially infected agents per node
    - `infdurdist`: function that samples the infectious duration distribution
    - `wandurdist`: function that samples the waning immunity duration distribution
    - `infdurmin`: minimum infectious period (default = 1 day)
    - `wandurmin`: minimum duration of immunity (default = 1 day)

    Outputs:
    - `model.people.itimer`: days remaining in the infectious state
    - `model.people.rtimer`: days remaining in the recovered state
    - `model.nodes.I`, `model.nodes.R`: counts per node per tick
    - `model.nodes.recovered[t]`: number of recoveries recorded on tick `t`

    Step Behavior:
    - Infectious agents decrement their `itimer`
    - When `itimer == 0`, agents become recovered and receive an `rtimer`
    - Patch-level totals are updated
    - Downstream components (e.g., `Recovered`) handle `rtimer` countdown and eventual return to `SUSCEPTIBLE`

    Validation:
    - Ensures timer consistency and population accounting
    - Confirms correct infectious-to-recovered transitions
    - Can be chained with recovery and waning components for full SIRS/SEIRS loops

    Plotting:
    Two plots are provided:
    1. Infected counts per node
    2. Total infected and recovered counts across time

    Example:
        model.components = [
            SIR.Susceptible(model),
            InfectiousIRS(model, infdurdist, wandurdist),
            Exposed(model, ...),
            Recovered(model),
        ]
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
    Recovered Component for SIR/SEIR Models (Permanent Immunity)

    This component manages agents in the recovered state in models where immunity does
    not wane (i.e., once recovered, agents stay recovered permanently). It tracks the
    number of recovered individuals over time at the patch level, but performs no active
    transitions itself — recovery transitions must be handled by upstream components.

    Responsibilities:
    - Initializes agents as recovered if specified in `model.scenario.R`
    - Tracks per-patch recovered counts over time in `model.nodes.R`
    - Verifies consistency between agent state and aggregate recovered counts
    - Propagates recovered totals forward unchanged (unless modified by other components)

    Required Inputs:
    - `model.scenario.R`: number of initially recovered individuals per node

    Outputs:
    - `model.nodes.R[t, i]`: number of recovered individuals at tick `t` in node `i`

    Step Behavior:
    - At each tick, carries forward:
        R[t+1] = R[t]
    - This component does not change any agent's state or internal timers

    🧪 Validation:
    - Ensures per-agent state matches aggregate `R` counts before and after each step
    - Detects accidental changes to recovered counts not explained by upstream logic

    Plotting:
    The `plot()` method shows per-node and total recovered counts over time.

    Example:
        model.components = [
            SIR.Susceptible(model),
            InfectiousIR(model, infdurdist),
            Recovered(model),  # passive tracker, assumes recovery handled upstream
        ]
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
    Recovered Component for SIRS/SEIRS Models (Waning Immunity)

    This component manages agents in the recovered state in models where immunity is temporary.
    It supports per-agent recovery timers, enabling individuals to return to the susceptible
    state after a configurable waning period. This is essential for SEIRS/SIRS model dynamics.

    Responsibilities:
    - Initializes agents in the `RECOVERED` state using `model.scenario.R`
    - Assigns `rtimer` values to track the duration of immunity
    - Decrements `rtimer` each tick; transitions agents to `SUSCEPTIBLE` when `rtimer == 0`
    - Updates patch-level counts:
        • `R[t, i]`: number of recovered individuals in node `i` at time `t`
        • `waned[t, i]`: number of agents who re-entered susceptibility on time step `t`

    Required Inputs:
    - `model.scenario.R`: initial number of recovered individuals per node
    - `wandurdist`: a function sampling the waning immunity duration distribution
    - `wandurmin`: minimum duration of immunity (default = 1 time step)

    Outputs:
    - `model.people.rtimer`: per-agent countdown to immunity expiration
    - `model.nodes.R`: recovered count per patch per timestep
    - `model.nodes.waned`: number of immunity losses per patch per tick

    Step Behavior:
    - Agents with `state == RECOVERED` decrement `rtimer`
    - When `rtimer == 0`, they return to `SUSCEPTIBLE`
    - `R` and `S` counts are updated to reflect this transition
    - `waned[t]` logs the number of agents who lost immunity on time step `t`

    Validation:
    - Ensures population conservation and consistency between agent states and patch totals
    - Detects unexpected changes in `R` or invalid transitions

    Plotting:
    The `plot()` method provides two views:
    1. Per-node recovered trajectories
    2. Total recovered and waned agents over time

    Example:
        model.components = [
            SIR.Susceptible(model),
            SEIRS.Infectious(model, infdurdist, wandurdist),
            Exposed(model, ...),
            RecoveredRS(model, wandurdist),
        ]
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
    Transmission Component for SI-Style Models (S → I Only, No Recovery)

    This component simulates the transmission process in simple epidemic models where
    agents move from the `SUSCEPTIBLE` to `INFECTIOUS` state and remain infectious
    indefinitely. It computes the force of infection (FOI) for each patch and applies
    it stochastically to susceptible agents.

    Responsibilities:
    - Computes per-node force of infection (`λ`) at each tick:
        λ = β * (I / N), with spatial coupling via a migration matrix
    - Applies probabilistic infection to susceptible agents using `nb_transmission_step`
    - Updates per-node `S` and `I` counts accordingly
    - Tracks new infections (incidence) and FOI values per node and tick

    Required Inputs:
    - `model.nodes.I[t]`: number of infectious agents per node at tick `t`
    - `model.nodes.S[t]`: number of susceptible agents per node at tick `t`
    - `model.params.beta`: transmission rate (global)
    - `model.network`: matrix of spatial coupling between nodes

    Outputs:
    - `model.nodes.forces[t, i]`: force of infection in node `i` at tick `t`
    - `model.nodes.incidence[t, i]`: number of new infections in node `i` at tick `t`

    Step Behavior:
    - Computes FOI (`λ`) for each node
    - Applies inter-node infection pressure via `model.network`
    - Converts FOI into a Bernoulli probability using: `p = 1 - exp(-λ)`
    - Infects susceptible agents probabilistically
    - Updates state and records incidence

    Validation:
    - Ensures consistency between state transitions and incidence records
    - Checks conservation of population in `S` and `I` states
    - Validates `incidence[t] == I[t+1] - I[t]`

    Plotting:
    The `plot()` method displays the force of infection over time per node.

    Example:
        model.components = [
            SIR.Susceptible(model),
            TransmissionSIX(model),
            InfectiousSI(model),
        ]
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
    Transmission Component for SIS/SIR/SIRS Models (S → I with Duration)

    This component simulates the transition from `SUSCEPTIBLE` to `INFECTIOUS` in
    models where infectious individuals have a finite infection duration (`itimer`).
    It supports full spatial coupling and allows infection durations to vary by node
    and tick.

    Responsibilities:
    - Computes force of infection (FOI) `λ = β * (I / N)` per patch each tick
    - Applies optional spatial coupling via `model.network` (infection pressure transfer)
    - Converts FOI into Bernoulli probabilities using `p = 1 - exp(-λ)`
    - Infects susceptible agents stochastically, assigning per-agent `itimer`
    - Updates patch-level susceptible (`S`) and infectious (`I`) counts
    - Records number of new infections per tick in `model.nodes.incidence`

    Required Inputs:
    - `model.params.beta`: transmission rate (global)
    - `model.network`: [n x n] matrix of transmission coupling
    - `infdurdist(tick, node)`: callable sampling the infectious duration distribution
    - `model.people.itimer`: preallocated per-agent infection timer

    Outputs:
    - `model.nodes.forces[t, i]`: computed FOI in node `i` at time `t`
    - `model.nodes.incidence[t, i]`: new infections in node `i` on time step `t`

    Step Behavior:
    - Computes FOI (`λ`) for each node
    - Applies inter-node infection pressure via `model.network`
    - Converts FOI into a Bernoulli probability using: `p = 1 - exp(-λ)`
    - Infects susceptible agents probabilistically
    - Updates state and records incidence

    Validation:
    - Ensures consistency between incidence and change in `I`
    - Checks for correct state and population accounting before and after tick

    Plotting:
    The `plot()` method visualizes per-node FOI (`λ`) over simulation time.

    Example:
        model.components = [
            SIR.Susceptible(model),
            TransmissionSI(model, infdurdist),
            InfectiousIR(model, infdurdist),
        ]
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
    @nb.njit(nogil=True, parallel=True, cache=True)
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
    Transmission Component for SEIR/SEIRS Models (S → E with Incubation Duration)

    This component simulates the transition from `SUSCEPTIBLE` to `EXPOSED` in models
    where infection includes an incubation period before agents become infectious.
    It handles stochastic exposure based on per-node force of infection (FOI), and
    assigns individual incubation timers to newly exposed agents.

    Responsibilities:
    - Computes force of infection `λ = β * (I / N)` at each tick per node
    - Adjusts FOI using `model.network` for inter-node transmission coupling. Required but can be nullified by filling with all zeros.
    - Applies FOI to susceptible agents to determine exposure
    - Assigns incubation durations (`etimer`) to each newly exposed agent
    - Updates node-level counts for `S` and `E` and logs daily incidence

    Required Inputs:
    - `model.params.beta`: global transmission rate
    - `model.network`: [n x n] matrix for FOI migration
    - `expdurdist(tick, node)`: callable that samples the exposure/incubation duration distribution
    - `expdurmin`: minimum incubation period (default = 1)

    Outputs:
    - `model.nodes.forces[t, i]`: computed FOI in node `i` at tick `t`
    - `model.nodes.incidence[t, i]`: new exposures per node per day
    - `model.people.etimer`: per-agent incubation countdown

    Step Behavior:
    - Computes FOI (`λ`) for each node
    - Optionally applies inter-node infection pressure via `model.network`
    - Converts FOI into a Bernoulli probability using: `p = 1 - exp(-λ)`
    - Infects susceptible agents probabilistically
    - Updates state and records incidence

    Validation:
    - Validates consistency between agent states and patch-level counts before and after tick
    - Confirms that `incidence[t] == E[t+1] - E[t]`

    Plotting:
    The `plot()` method shows per-node FOI (`λ`) trajectories over time.

    Example:
        model.components = [
            SIR.Susceptible(model),
            TransmissionSE(model, expdurdist),
            Exposed(model, ...),
            InfectiousIR(model, ...),
            Recovered(model),
        ]
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
    @nb.njit(nogil=True, parallel=True, cache=True)
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
    Base Component for Vital Dynamics (Births and Deaths)

    This abstract base class implements shared functionality for components that model
    vital population dynamics in infectious disease simulations — including both births
    and non-disease deaths. Subclasses must define a `step()` method that controls the
    timing and triggering of these events (e.g., daily, seasonal, or condition-based).

    Responsibilities:
    - Adds `dob` (date of birth) and `dod` (date of death) agent-level properties
    - Samples date of birth using an age pyramid distribution
    - Samples date of death using a survival curve estimator
    - Tracks births and deaths at the node level per tick
    - Updates susceptible and deceased populations accordingly
    - Validates population accounting before and after demographic changes
    - Invokes `on_birth(...)` hooks in downstream components (if defined)

    Required Inputs:
    - `birthrates[t, i]`: annual crude birth rate for each node
    - `pyramid`: instance of `AliasedDistribution` to sample agent age at birth (only for initial population)
    - `survival`: instance of `KaplanMeierEstimator` to sample lifespan from age (for both initial population and future births)
    - `states`: list of state labels used in population accounting (default = `["S", "E", "I", "R"]`)

    Outputs:
    - `model.people.dob`, `model.people.dod`: per-agent date of birth and death
    - `model.nodes.births[t, i]`: number of births on tick `t` in node `i`
    - `model.nodes.deaths[t, i]`: number of deaths on tick `t` in node `i`
    - `model.nodes.S[t, i]`: susceptible counts incremented by births

    Behavior Overview:
    - `prevalidate_step()`: caches population and death status for change detection
    - `postvalidate_step()`: checks that births/deaths were applied consistently
    - `_births()`: allocates newborns across patches, assigns `dob`, `dod`, and default state
    - Death processing is expected to occur in a subclass implementation

    Validation:
    - Verifies changes in total agent count are consistent with births
    - Ensures patch-level birth and death counts match aggregated agent transitions
    - Asserts population conservation across susceptible and infectious compartments

    Extension Notes:
    This class does not implement `step()` and must be subclassed with a scheduling strategy.

    Plotting:
    The `plot()` method provides daily and cumulative trends for births and deaths.

    Example:
        class VitalDaily(VitalDynamicsBase):
            def step(self, tick: int) -> None:
                self._births(tick)
                self._deaths(tick)

        model.components.append(VitalDaily(...))
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
    Vital Dynamics for SI/SIS Models (Births and Deaths with S and I Only)

    This component implements demographic changes for simple SI or SIS models, where the only
    health states are `SUSCEPTIBLE` and `INFECTIOUS`. It supports both daily births and
    non-disease deaths using age-structured survival curves and crude birth rates (CBR).

    Responsibilities:
    - Applies daily mortality by checking agent `dod == tick`
    - Removes agents from `S` or `I` and marks them `DECEASED`
    - Adds new agents as `SUSCEPTIBLE` using `_births()`, sampling age and lifespan
    - Updates patch-level counts in `S`, `I`, `births`, and `deaths`

    Validation:
    - Ensures birth and death counts match population changes
    - Verifies newborn states and death state transitions

    Example:
        model.components.append(
            VitalDynamicsSI(model, birthrates, pyramid, survival)
        )
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
    Vital Dynamics for SIR/SIRS Models (Births and Deaths with S, I, R States)

    This component manages demographic events for models with susceptible, infectious,
    and recovered compartments. It handles per-agent births and age-dependent deaths
    and adjusts patch-level state counts accordingly.

    Responsibilities:
    - Applies mortality by removing agents in `S`, `I`, or `R` with `dod == tick`
    - Marks deceased agents with `State.DECEASED`
    - Adds newborns as `SUSCEPTIBLE`, updating `S`, `births`, and `deaths`
    - Maintains per-tick, per-patch counts for `S`, `I`, `R`, `births`, `deaths`

    Validation:
    - Verifies consistent population accounting across all compartments
    - Asserts correct transition timing for births and deaths

    Example:
        model.components.append(
            VitalDynamicsSIR(model, birthrates, pyramid, survival)
        )
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
    Vital Dynamics for SEIR/SEIRS Models (Births and Deaths with S, E, I, R States)

    This component supports full vital dynamics in compartmental models that include
    an incubation stage (`EXPOSED`) in addition to infection and recovery.

    Responsibilities:
    - Applies daily age-dependent mortality across `S`, `E`, `I`, and `R`
    - Marks deceased agents as `DECEASED`, removing them from the simulation
    - Adds new agents as `SUSCEPTIBLE`, assigning age and expected lifespan
    - Updates per-tick, per-node counts for `S`, `E`, `I`, `R`, `births`, and `deaths`

    Validation:
    - Ensures consistent agent transitions and patch-level flows
    - Confirms that changes in population reflect births minus deaths

    Example:
        model.components.append(
            VitalDynamicsSEIR(model, birthrates, pyramid, survival)
        )
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
