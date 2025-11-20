"""
Immunization components for LASER models.

These components introduce immunity into the agent population during a simulation.

Notes
-----
- Deployment is currently global (all patches). Future extensions may include:
  * targeting by patch or lists of patches,
  * patch-varying coverage,
  * time-varying routine immunization (RI) coverage.
- The routine immunization window centers on the target age with width ≈ period,
  i.e., [age - period/2, age + period/2).

"""

from typing import Optional

import numpy as np
from matplotlib.figure import Figure


class RoutineImmunization:
    """
    A LASER component that updates immunity via routine immunization (RI).

    At eligible ticks, agents whose age (in ticks) falls within an RI window
    centered at `age` with half-width `period // 2` are sampled with probability
    `coverage` and made immune (by setting `population.susceptibility[idx] = 0`).

    This component follows the general component style in `laser-generic` and can
    be added to `Model.components`. See package documentation for details on the component pattern.
    """

    def __init__(
        self,
        model,
        period: int,
        coverage: float,
        age: int,
        start: int = 0,
        end: int = -1,
        verbose: bool = False,
    ) -> None:
        """
        Initialize a RoutineImmunization instance.

        Args:
            model (object): LASER `Model` with `population`, `patches`, and `params`.
                `params.nticks` must be defined.
            period (int): Number of ticks between RI events. Must be >= 1.
            coverage (float): Per-event immunization probability in [0.0, 1.0].
            age (int): Target age (in ticks) around which to immunize.
            start (int, optional): First tick (inclusive) to run RI. Default 0.
            end (int, optional): Last tick (exclusive) to run RI. If -1, defaults
                to `model.params.nticks`. Default -1.
            verbose (bool, optional): Enable verbose logging. Default False.

        Attributes:
            model (object): The LASER model instance.
            period (int): Ticks between RI events.
            coverage (float): Immunization probability at each event.
            age (int): Target age in ticks.
            start (int): First RI tick (inclusive).
            end (int): Last RI tick (exclusive).
            verbose (bool): Verbosity flag.

        Raises:
            ValueError: If `period < 1`, `coverage` not in [0, 1], or `age < 0`.
        """
        if period < 1:
            raise ValueError("period must be >= 1")
        if not (0.0 <= coverage <= 1.0):
            raise ValueError("coverage must be within [0.0, 1.0]")
        if age < 0:
            raise ValueError("age must be >= 0")

        self.model = model
        self.period = int(period)
        self.coverage = float(coverage)
        self.age = int(age)
        self.start = int(start)
        self.end = int(model.params.nticks if end == -1 else end)
        self.verbose = bool(verbose)

    def __call__(self, model, tick: int) -> None:
        """
        Apply routine immunization at the given tick, if eligible.

        An event fires when:
            tick >= start
            and ((tick - start) % period == 0)
            and tick < end

        On each event:
            - Agents with age in [age - period//2, age + period//2) are considered.
            - A Binomial draw with probability `coverage` selects agents to immunize.
            - Selected agents have `susceptibility` set to 0 (immune).
            - If present, test arrays on `model.nodes` are updated for validation.

        Args:
            model (object): LASER model (unused; provided for signature parity).
            tick (int): Current simulation tick.

        Returns:
            None
        """
        if (tick >= self.start) and ((tick - self.start) % self.period == 0) and (tick < self.end):
            half_window = int(self.period // 2)
            lower = max(0, int(self.age - half_window))
            upper = int(self.age + half_window)
            immunize_nodeids = immunize_in_age_window(self.model, lower, upper, self.coverage, tick)

            # Update validation arrays if they exist and we immunized some agents
            if hasattr(self.model.nodes, "recovered_test") and immunize_nodeids is not None and len(immunize_nodeids) > 0:
                np.add.at(self.model.nodes.recovered_test, (tick + 1, immunize_nodeids), 1)
                np.add.at(self.model.nodes.susceptibility_test, (tick + 1, immunize_nodeids), -1)

    def plot(self, fig: Optional[Figure] = None):
        """
        Placeholder for RI visualization.

        Args:
            fig (Figure, optional): A Matplotlib Figure to draw into.

        Returns:
            None
        """
        yield
        return


def immunize_in_age_window(model, lower: int, upper: int, coverage: float, tick: int) -> Optional[np.ndarray]:
    """
    Immunize susceptible agents whose age is in [lower, upper).

    This function updates agent-level susceptibility and returns the corresponding
    node IDs for accounting or test-array updates.

    Args:
        model (object): LASER `Model` containing `population` with fields:
            - dob (array[int]): Agent date-of-birth ticks.
            - susceptibility (array[int|bool]): 1/True if susceptible, 0/False if immune.
            - nodeid (array[int]): Patch index per agent.
        lower (int): Inclusive lower bound on age (in ticks). Clamped to >= 0.
        upper (int): Exclusive upper bound on age (in ticks). Must be >= lower.
        coverage (float): Probability in [0, 1] to immunize each eligible susceptible.
        tick (int): Current simulation tick.

    Returns:
        np.ndarray | None: Array of `nodeid` for immunized agents, or None if none.

    Raises:
        ValueError: If `upper < lower` or `coverage` not in [0, 1].
    """
    if upper < lower:
        raise ValueError("upper must be >= lower")
    if not (0.0 <= coverage <= 1.0):
        raise ValueError("coverage must be within [0.0, 1.0]")

    pop = model.people

    # Ages in ticks
    ages = tick - pop.dob

    # Eligible: in window AND currently susceptible
    in_window = (ages >= max(0, lower)) & (ages < upper)
    # Cast susceptibility to boolean in case it's int array (0/1)
    susceptible = np.asarray(pop.susceptibility).astype(bool)
    eligible_idx = np.flatnonzero(susceptible & in_window)

    if eligible_idx.size == 0:
        return None

    # Binomial draw on the eligible set size
    n_immunize = np.random.binomial(eligible_idx.size, coverage)
    if n_immunize == 0:
        return None

    # Sample without replacement
    chosen = eligible_idx.copy()
    np.random.shuffle(chosen)
    chosen = chosen[:n_immunize]

    # Apply immunity
    pop.susceptibility[chosen] = 0  # mark as immune
    return pop.nodeid[chosen]


class ImmunizationCampaign:
    """
    A LASER component that applies an immunization campaign over an age band.

    On eligible ticks, all agents with age in [age_lower, age_upper) are considered
    and immunized with probability `coverage`. Susceptibles become immune
    (`population.susceptibility[idx] = 0`). This aligns with the campaign-style
    immunization component described in the `laser-generic` docs.
    """

    def __init__(
        self,
        model,
        period: int,
        coverage: float,
        age_lower: int,
        age_upper: int,
        start: int = 0,
        end: int = -1,
        verbose: bool = False,
    ) -> None:
        """
        Initialize an ImmunizationCampaign instance.

        Args:
            model (object): LASER `Model` with `population`, `patches`, and `params`.
                `params.nticks` must be defined.
            period (int): Number of ticks between campaign events. Must be >= 1.
            coverage (float): Per-event immunization probability in [0.0, 1.0].
            age_lower (int): Inclusive lower bound of target age band (ticks).
            age_upper (int): Exclusive upper bound of target age band (ticks). Must be > age_lower.
            start (int, optional): First tick (inclusive) to run campaigns. Default 0.
            end (int, optional): Last tick (exclusive) to run campaigns. If -1, defaults
                to `model.params.nticks`. Default -1.
            verbose (bool, optional): Enable verbose logging. Default False.

        Attributes:
            model (object): The LASER model instance.
            period (int): Ticks between campaign events.
            coverage (float): Immunization probability at each event.
            age_lower (int): Inclusive lower age (ticks).
            age_upper (int): Exclusive upper age (ticks).
            start (int): First campaign tick (inclusive).
            end (int): Last campaign tick (exclusive).
            verbose (bool): Verbosity flag.

        Raises:
            ValueError: If inputs are out of range (e.g., period < 1, coverage not in [0, 1],
                        age bounds invalid).
        """
        if period < 1:
            raise ValueError("period must be >= 1")
        if not (0.0 <= coverage <= 1.0):
            raise ValueError("coverage must be within [0.0, 1.0]")
        if age_lower < 0:
            raise ValueError("age_lower must be >= 0")
        if age_upper <= age_lower:
            raise ValueError("age_upper must be > age_lower")

        self.model = model
        self.period = int(period)
        self.coverage = float(coverage)
        self.age_lower = int(age_lower)
        self.age_upper = int(age_upper)
        self.start = int(start)
        self.end = int(model.params.nticks if end == -1 else end)
        self.verbose = bool(verbose)

    def __call__(self, model, tick: int) -> None:
        """
        Apply the immunization campaign at the given tick, if eligible.

        Triggers when:
            tick >= start
            and ((tick - start) % period == 0)
            and tick < end

        On each event:
            - Agents with age in [age_lower, age_upper) are considered.
            - A Binomial draw with probability `coverage` selects agents to immunize.
            - Selected agents have `susceptibility` set to 0 (immune).
            - If present, test arrays on `model.nodes` are updated for validation.

        Args:
            model (object): LASER model (unused; provided for signature parity).
            tick (int): Current simulation tick.

        Returns:
            None
        """
        if (tick >= self.start) and ((tick - self.start) % self.period == 0) and (tick < self.end):
            immunize_nodeids = immunize_in_age_window(self.model, self.age_lower, self.age_upper, self.coverage, tick)

            if hasattr(self.model.nodes, "recovered_test") and immunize_nodeids is not None and len(immunize_nodeids) > 0:
                np.add.at(self.model.nodes.recovered_test, (tick + 1, immunize_nodeids), 1)
                np.add.at(self.model.nodes.susceptibility_test, (tick + 1, immunize_nodeids), -1)

    def plot(self, fig: Optional[Figure] = None) -> None:
        """
        Placeholder for campaign visualization.

        Args:
            fig (Figure, optional): A Matplotlib Figure to draw into.

        Returns:
            None
        """
        yield
        return


##### ============================= #####

from typing import Callable  # noqa: E402

import numba as nb  # noqa: E402

from .shared import State  # noqa: E402


class RoutineImmunizationEx:
    def __init__(
        self,
        model,
        coverage_fn: Callable[[int, int], float],
        dose_timing_dist: Callable[[int, int], int],
        dose_timing_min: int = 1,
        initialize: bool = True,
        track: bool = False,
    ) -> None:
        self.model = model
        self.coverage_fn = coverage_fn
        self.dose_timing_dist = dose_timing_dist
        self.dose_timing_min = dose_timing_min

        self.track = track

        self.model.people.add_scalar_property("ritimer", dtype=np.int16, default=0)
        self.model.nodes.add_vector_property("ri_immunized", model.params.nticks + 1, dtype=np.int32, default=0)
        self.model.nodes.add_vector_property("ri_doses", model.params.nticks + 1, dtype=np.int32, default=0)
        if track:
            self.model.people.add_scalar_property("initial_ri", dtype=np.int16, default=0)

        if initialize:
            self.initialize_ritimers(
                self.model.people.dob,
                0,  # tick
                self.model.people.nodeid,
                self.dose_timing_dist,
                self.dose_timing_min,
                self.model.people.ritimer,
                self.coverage_fn,
                self.model.people.state,
            )
            if track:
                non_zero = self.model.people.ritimer > 0
                self.model.people.initial_ri[non_zero] = (self.model.people.ritimer - self.model.people.dob)[non_zero]

        return

    @staticmethod
    @nb.njit(nogil=True, parallel=True)
    def initialize_ritimers(
        dobs: np.ndarray,
        tick: int,
        nodeids: np.ndarray,
        dose_timing_dist: Callable[[int, int], int],
        dose_timing_min: int,
        ritimers: np.ndarray,
        coverage_fn: Callable[[int, int], float],
        states: np.ndarray,
    ) -> None:
        for i in nb.prange(len(dobs)):
            age = tick - dobs[i]
            if age < 365:
                nid = nodeids[i]
                time_to_immunization = np.maximum(dose_timing_dist(0, nid), dose_timing_min)
                if time_to_immunization >= age:
                    ritimers[i] = time_to_immunization - age
                else:
                    coverage = coverage_fn(tick, nid)
                    draw = np.random.rand()
                    if draw < coverage:
                        states[i] = State.RECOVERED.value

        return

    @staticmethod
    @nb.njit(nogil=True, parallel=True)
    def routine_immunization_ex_step(
        states: np.ndarray,
        ritimers: np.ndarray,
        nodeids: np.ndarray,
        coverage_fn: Callable[[int, int], float],
        newly_immunized: np.ndarray,
        new_doses: np.ndarray,
        tick: int,
    ) -> None:
        for i in nb.prange(len(states)):
            timer = ritimers[i]
            if timer > 0:
                timer -= 1
                ritimers[i] = timer
                if timer == 0:
                    nid = nodeids[i]
                    coverage = coverage_fn(tick, nid)
                    draw = np.random.rand()
                    if draw < coverage:
                        if states[i] == State.SUSCEPTIBLE.value:
                            states[i] = State.RECOVERED.value
                            newly_immunized[nb.get_thread_id(), nid] += 1  # How many actually move from S to R
                        new_doses[nb.get_thread_id(), nid] += 1  # Count the dose as delivered regardless of state

        return

    def step(self, tick: int) -> None:
        newly_immunized = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        new_doses = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        self.routine_immunization_ex_step(
            self.model.people.state,
            self.model.people.ritimer,
            self.model.people.nodeid,
            self.coverage_fn,
            newly_immunized,
            new_doses,
            tick,
        )
        newly_immunized = np.sum(newly_immunized, axis=0)

        self.model.nodes.S[tick] -= newly_immunized
        self.model.nodes.R[tick] += newly_immunized
        self.model.nodes.ri_immunized[tick] = newly_immunized

        new_doses = np.sum(new_doses, axis=0)
        self.model.nodes.ri_doses[tick] = new_doses

        return

    def on_birth(self, istart: int, iend: int, tick: int) -> None:
        # Initialize ritimers for newborn agents
        self.initialize_ritimers(
            self.model.people.dob[istart:iend],
            tick,
            self.model.people.nodeid[istart:iend],
            self.dose_timing_dist,
            self.dose_timing_min,
            self.model.people.ritimer[istart:iend],
            self.coverage_fn,
            self.model.people.state[istart:iend],
        )
        if self.track:
            self.model.people.initial_ri[istart:iend] = self.model.people.ritimer[istart:iend]

        return
