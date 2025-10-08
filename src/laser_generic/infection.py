"""
This module defines components for modeling infection dynamics in a LASER population.

Classes
-------
Infection
    Handles general infection-to-recovery transitions.
Infection_SIS
    Variant for SIS models, where recovered agents become susceptible again.
"""

import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


class Infection:
    """
    LASER component for managing infections in an SIR-style model.

    Each agent has an ``itimer`` (infectious timer). When ``itimer > 0``, the
    agent is infectious. On each tick, ``itimer`` is decremented. When it reaches
    zero, the agent transitions to the recovered state (``state=3``).
    """

    def __init__(self, model, verbose: bool = False) -> None:
        """
        Initialize the Infection component.

        Parameters
        ----------
        model : object
            LASER model containing:
              * ``population`` with attributes ``count`` and method
                ``add_scalar_property(name, dtype, default)``
              * ``patches`` with method
                ``add_vector_property(name, length, dtype)``
              * ``params.nticks`` (int): number of simulation ticks
        verbose : bool, optional
            If True, enable verbose output. Default is False.

        Side Effects
        ------------
        - Adds a scalar property ``itimer`` (dtype=uint16, default=0) to the population.
        - Adds a vector property ``recovered`` (dtype=uint32) to the patches.
        - Initializes all infection timers to zero.
        """
        self.model = model

        model.population.add_scalar_property("itimer", dtype=np.uint16, default=0)
        model.patches.add_vector_property("recovered", length=model.params.nticks, dtype=np.uint32)
        Infection._nb_set_itimers_slice(0, model.population.count, model.population.itimer, np.uint16(0))

    def census(self, model, tick: int) -> None:
        """
        Aggregate recovered counts into patches at the given tick.

        Parameters
        ----------
        model : object
            LASER model containing ``population`` and ``patches``.
        tick : int
            Current simulation tick.

        Notes
        -----
        - At tick 0, recovered counts are computed from the population and written
          to ``patches.recovered``.
        - Recovery condition:
          * With etimers: susceptibility=0, etimer=0, itimer=0
          * Without etimers: susceptibility=0, itimer=0
        - For a single patch, recovered agents are counted globally.
        - For multiple patches, counts are distributed by agent ``nodeid``.
        - In all cases, values are copied into ``patches.recovered_test`` and carried
          forward to tick+1.
        """
        patches = model.patches
        if tick == 0:
            population = model.population
            recovered_count = patches.recovered[tick, :]

            if hasattr(population, "etimer"):
                condition = (
                    (population.susceptibility[0 : population.count] == 0)
                    & (population.etimer[0 : population.count] == 0)
                    & (population.itimer[0 : population.count] == 0)
                )
            else:
                condition = (population.susceptibility[0 : population.count] == 0) & (population.itimer[0 : population.count] == 0)

            if len(patches) == 1:
                value = np.uint32(np.count_nonzero(condition))  # Explicit cast
                np.add(recovered_count, value, out=recovered_count)
            else:
                nodeids = population.nodeid[0 : population.count]
                np.add.at(recovered_count, nodeids[condition], np.uint32(1))

            patches.recovered_test[tick, :] = patches.recovered[tick, :].copy()

        patches.recovered_test[tick + 1, :] = patches.recovered_test[tick, :].copy()

    def __call__(self, model, tick: int) -> None:
        """
        Update infection timers and patch-level case/recovery counts.

        Parameters
        ----------
        model : object
            LASER model containing ``population`` and ``patches``.
        tick : int
            Current simulation tick.

        Notes
        -----
        - Decrements ``itimer`` for all infectious agents.
        - Agents whose ``itimer`` reaches 0 transition to recovered (``state=3``).
        - Patch-level accounting:
          * ``patches.cases_test[t+1]`` is decremented by number of transitions.
          * ``patches.recovered_test[t+1]`` is incremented by the same number.
        """
        flow = np.zeros(len(model.patches), dtype=np.uint32)
        Infection._nb_infection_update_test(
            model.population.count,
            model.population.itimer,
            model.population.state,
            flow,
            model.population.nodeid,
        )
        model.patches.cases_test[tick + 1, :] -= flow
        model.patches.recovered_test[tick + 1, :] += flow

    def on_birth(self, model, _tick: int, istart, iend) -> None:
        """
        Reset infection timers for newborn agents.

        Parameters
        ----------
        model : object
            LASER model containing the population.
        _tick : int
            Current simulation tick (unused).
        istart : int or ndarray of int64
            Start index of newborns (if slice mode), or array of agent indices
            (if random-access mode).
        iend : int or None
            End index of newborns (exclusive). If None, ``istart`` is treated as
            an array of indices.

        Notes
        -----
        - Newborns are initialized with ``itimer=0`` (not infectious).
        """
        if iend is not None:
            Infection._nb_set_itimers_slice(istart, iend, model.population.itimer, np.uint16(0))
        else:
            Infection._nb_set_itimers_randomaccess(istart, model.population.itimer, np.uint16(0))

    def plot(self, fig: Figure = None):
        """
        Plot the distribution of infections by age.

        Parameters
        ----------
        fig : Figure, optional
            A Matplotlib Figure. If None, a new figure is created.

        Yields
        ------
        None
            This is a generator that currently yields once (``None``).
        """
        fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        fig.suptitle("Infections By Age")

        ages_in_years = (self.model.params.nticks - self.model.population.dob[0 : self.model.population.count]) // 365
        age_counts = np.bincount(ages_in_years)
        plt.bar(range(len(age_counts)), age_counts)

        itimers = self.model.population.itimer[0 : self.model.population.count]
        infected = itimers > 0
        infection_counts = np.bincount(ages_in_years[infected])
        plt.bar(range(len(infection_counts)), infection_counts)

        yield

    # ---------------------
    # Private Numba helpers
    # ---------------------

    @staticmethod
    @nb.njit((nb.uint32, nb.uint16[:], nb.uint8[:]), parallel=True, cache=True)
    def _nb_infection_update(count, itimers, state):
        for i in nb.prange(count):
            itimer = itimers[i]
            if itimer > 0:
                itimer -= 1
                if itimer == 0:
                    state[i] = 3
                itimers[i] = itimer

    @staticmethod
    @nb.njit((nb.uint32, nb.uint16[:], nb.uint8[:], nb.uint32[:], nb.uint16[:]), parallel=True, cache=True)
    def _nb_infection_update_test(count, itimers, state, flow, nodeid):
        max_node_id = np.max(nodeid) + 1
        thread_flow = np.zeros((nb.config.NUMBA_DEFAULT_NUM_THREADS, max_node_id), dtype=np.uint32)

        for i in nb.prange(count):
            itimer = itimers[i]
            if itimer > 0:
                itimer -= 1
                if itimer == 0:
                    thread_flow[nb.get_thread_id(), nodeid[i]] += 1
                    state[i] = 3
                itimers[i] = itimer

        flow[:] += thread_flow.sum(axis=0)

    @staticmethod
    @nb.njit((nb.uint32[:], nb.bool_[:], nb.uint16[:], nb.int64), parallel=True, cache=True)
    def _accumulate_recovered(node_rec, agent_recovered, nodeids, count):
        max_node_id = np.max(nodeids)
        thread_recovereds = np.zeros((nb.config.NUMBA_DEFAULT_NUM_THREADS, max_node_id + 1), dtype=np.uint32)

        for i in nb.prange(count):
            nodeid = nodeids[i]
            recovered = agent_recovered[i]
            thread_recovereds[nb.get_thread_id(), nodeid] += recovered

        node_rec[:] = thread_recovereds.sum(axis=0)

    @staticmethod
    @nb.njit((nb.uint32, nb.uint32, nb.uint16[:], nb.uint16), parallel=True, cache=True)
    def _nb_set_itimers_slice(istart, iend, itimers, value):
        for i in nb.prange(istart, iend):
            itimers[i] = value

    @staticmethod
    @nb.njit((nb.int64[:], nb.uint16[:], nb.uint16), parallel=True, cache=True)
    def _nb_set_itimers_randomaccess(indices, itimers, value):
        for i in nb.prange(len(indices)):
            itimers[indices[i]] = value


class Infection_SIS:
    """
    LASER component for infection in an SIS model.

    Each agent has an ``itimer``. When it reaches 0, the agent becomes susceptible again
    (``susceptibility[i] = 1``).
    """

    def __init__(self, model, verbose: bool = False) -> None:
        """
        Initialize the Infection_SIS component.

        Parameters
        ----------
        model : object
            LASER model containing a population and parameters.
        verbose : bool, optional
            If True, enable verbose output. Default is False.

        Side Effects
        ------------
        - Adds a scalar property ``itimer`` (dtype=uint16, default=0) to the population.
        - Initializes all infection timers to zero.
        """
        self.model = model

        model.population.add_scalar_property("itimer", dtype=np.uint16, default=0)
        Infection_SIS._nb_set_itimers(0, model.population.count, model.population.itimer, np.uint16(0))

    def __call__(self, model, tick: int) -> None:
        """
        Update infection timers for the population (SIS model).

        Parameters
        ----------
        model : object
            LASER model containing ``population``.
        tick : int
            Current simulation tick.

        Notes
        -----
        - Decrements ``itimer`` for all infectious agents.
        - When ``itimer`` reaches 0, the agent becomes susceptible again
          (``susceptibility=1``).
        """
        Infection_SIS._nb_infection_update(model.population.count, model.population.itimer, model.population.susceptibility)

    def on_birth(self, model, _tick: int, istart, iend) -> None:
        """
        Reset infection timers for newborn agents.

        Parameters
        ----------
        model : object
            LASER model containing the population.
        _tick : int
            Current simulation tick (unused).
        istart : int
            Start index of newborns.
        iend : int
            End index of newborns (exclusive).

        Notes
        -----
        - Newborns are initialized with ``itimer=0``.
        """
        model.population.itimer[istart:iend] = 0

    def plot(self, fig: Figure = None):
        """
        Plot the distribution of infections by age.

        Parameters
        ----------
        fig : Figure, optional
            A Matplotlib Figure. If None, a new figure is created.

        Yields
        ------
        None
            This is a generator that currently yields once (``None``).
        """
        fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        fig.suptitle("Infections By Age")

        ages_in_years = (self.model.params.nticks - self.model.population.dob[0 : self.model.population.count]) // 365
        age_counts = np.bincount(ages_in_years)
        plt.bar(range(len(age_counts)), age_counts)

        itimers = self.model.population.itimer[0 : self.model.population.count]
        infected = itimers > 0
        infection_counts = np.bincount(ages_in_years[infected])
        plt.bar(range(len(infection_counts)), infection_counts)

        yield

    # ---------------------
    # Private Numba helpers
    # ---------------------

    @staticmethod
    @nb.njit((nb.uint32, nb.uint16[:], nb.uint8[:]), parallel=True, cache=True)
    def _nb_infection_update(count, itimers, susceptibility):
        for i in nb.prange(count):
            itimer = itimers[i]
            if itimer > 0:
                itimer -= 1
                itimers[i] = itimer
                if itimer == 0:
                    susceptibility[i] = 1

    @staticmethod
    @nb.njit((nb.uint32, nb.uint32, nb.uint16[:], nb.uint16), parallel=True, cache=True)
    def _nb_set_itimers(istart, iend, itimers, value):
        for i in nb.prange(istart, iend):
            itimers[i] = value
