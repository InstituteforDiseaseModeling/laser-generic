"""
This module defines the Exposure class, which models the transition of individuals
from being exposed (infected but not yet infectious) to becoming infectious.

The component manages the `etimer` (exposed timer) for each agent, updates
patch-level exposed/case counts, handles newborns, and provides simple
visualization of exposure by age.
"""

import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


class Exposure:
    """
    LASER component representing the exposed (latent) state of a population.

    Each agent has an ``etimer`` (exposed timer). When ``etimer > 0``, the agent
    is in the exposed state. On each tick, ``etimer`` is decremented. When it
    reaches zero, the agent transitions to the infectious state, and its
    ``itimer`` (infectious timer) is initialized.
    """

    def __init__(self, model, verbose: bool = False) -> None:
        """
        Initialize the Exposure component.

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
        - Adds a scalar property ``etimer`` (dtype=uint16, default=0) to the population.
        - Adds a vector property ``exposed`` (dtype=uint32) to the patches.
        - Initializes all etimers to zero.
        """
        self.model = model

        model.population.add_scalar_property("etimer", dtype=np.uint16, default=0)
        model.patches.add_vector_property("exposed", length=model.params.nticks, dtype=np.uint32)
        Exposure._nb_set_etimers_slice(0, model.population.count, model.population.etimer, np.uint16(0))

    def census(self, model, tick: int) -> None:
        """
        Aggregate exposed counts into patches at the given tick.

        Parameters
        ----------
        model : object
            LASER model with ``population`` and ``patches``.
        tick : int
            Current simulation tick.

        Notes
        -----
        - At tick 0, exposed counts are computed from the population and written
          to ``patches.exposed``.
        - For a single patch, exposed agents are counted globally.
        - For multiple patches, exposed agents are distributed according to their
          ``nodeid``.
        - In all cases, values are copied into ``patches.exposed_test`` for
          validation/debugging and carried forward to tick+1.
        """
        patches = model.patches
        if tick == 0:
            population = model.population
            exposed_count = patches.exposed[tick, :]
            condition = population.etimer[0 : population.count] > 0

            if len(patches) == 1:
                np.add(exposed_count, np.uint32(np.count_nonzero(condition)), out=exposed_count)
            else:
                nodeids = population.nodeid[0 : population.count]
                np.add.at(exposed_count, nodeids[condition], np.uint32(1))

            patches.exposed_test[tick, :] = patches.exposed[tick, :].copy()

        patches.exposed_test[tick + 1, :] = patches.exposed_test[tick, :].copy()

    def __call__(self, model, tick: int) -> None:
        """
        Update exposed timers for the population at the given tick.

        Parameters
        ----------
        model : object
            LASER model containing ``population`` and ``patches``.
        tick : int
            Current simulation tick.

        Notes
        -----
        - Decrements ``etimer`` for all exposed agents.
        - Agents whose ``etimer`` reaches 0 transition to infectious:
          * ``itimer`` is set to a draw from Normal(inf_mean, inf_sigma),
            with minimum value 1.
          * ``state`` is set to 2 (infectious).
        - Patch-level accounting:
          * ``patches.exposed_test[t+1]`` is decremented by the number of
            transitions.
          * ``patches.cases_test[t+1]`` is incremented by the same number.
        """
        flow = np.zeros(len(model.patches), dtype=np.uint32)
        Exposure._nb_exposure_update_test(
            model.population.count,
            model.population.etimer,
            model.population.itimer,
            model.population.state,
            model.params.inf_mean,
            model.params.inf_sigma,
            flow,
            model.population.nodeid,
        )

        model.patches.exposed_test[tick + 1, :] -= flow
        model.patches.cases_test[tick + 1, :] += flow

    def on_birth(self, model, _tick: int, istart, iend) -> None:
        """
        Reset exposure timers for newborn agents.

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
        - Newborns are set to not exposed: ``etimer = 0``.
        """
        if iend is not None:
            Exposure._nb_set_etimers_slice(istart, iend, model.population.etimer, np.uint16(0))
        else:
            Exposure._nb_set_etimers_randomaccess(istart, model.population.etimer, np.uint16(0))

    def plot(self, fig: Figure = None):
        """
        Plot the distribution of exposures by age.

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
        fig.suptitle("Exposed By Age")

        ages_in_years = (self.model.params.nticks - self.model.population.dob[0 : self.model.population.count]) // 365
        age_counts = np.bincount(ages_in_years)
        plt.bar(range(len(age_counts)), age_counts)

        etimers = self.model.population.etimer[0 : self.model.population.count]
        exposed = etimers > 0
        exposed_counts = np.bincount(ages_in_years[exposed])
        plt.bar(range(len(exposed_counts)), exposed_counts)

        yield

    # -------------------------------------------------------------------------
    # Private Numba helper functions (not part of public API)
    # -------------------------------------------------------------------------

    @staticmethod
    @nb.njit((nb.uint32, nb.uint16[:], nb.uint16[:], nb.uint8[:], nb.float32, nb.float32), parallel=True, cache=True)
    def _nb_exposure_update(count, etimers, itimers, state, inf_mean, inf_sigma):
        for i in nb.prange(count):
            etimer = etimers[i]
            if etimer > 0:
                etimer -= 1
                if etimer <= 0:
                    itimers[i] = np.maximum(np.uint16(1), np.uint16(np.ceil(np.random.normal(inf_mean, inf_sigma))))
                    state[i] = 2
                etimers[i] = etimer

    @staticmethod
    @nb.njit(
        (nb.uint32, nb.uint16[:], nb.uint16[:], nb.uint8[:], nb.float32, nb.float32, nb.uint32[:], nb.uint16[:]),
        parallel=True,
        cache=True,
    )
    def _nb_exposure_update_test(count, etimers, itimers, state, inf_mean, inf_sigma, flow, nodeid):
        max_node_id = np.max(nodeid) + 1
        thread_flow = np.zeros((nb.config.NUMBA_DEFAULT_NUM_THREADS, max_node_id), dtype=np.uint32)

        for i in nb.prange(count):
            etimer = etimers[i]
            if etimer > 0:
                etimer -= 1
                if etimer <= 0:
                    itimers[i] = np.maximum(np.uint16(1), np.uint16(np.ceil(np.random.normal(inf_mean, inf_sigma))))
                    thread_flow[nb.get_thread_id(), nodeid[i]] += 1
                    state[i] = 2
                etimers[i] = etimer

        flow[:] += thread_flow.sum(axis=0)

    @staticmethod
    @nb.njit((nb.uint32[:], nb.bool_[:], nb.uint16[:], nb.int64), parallel=True, cache=True)
    def _accumulate_exposed(node_exp, agent_exposed, nodeids, count):
        max_node_id = np.max(nodeids)
        thread_exposed = np.zeros((nb.config.NUMBA_DEFAULT_NUM_THREADS, max_node_id + 1), dtype=np.uint32)

        for i in nb.prange(count):
            nodeid = nodeids[i]
            exposed = agent_exposed[i]
            thread_exposed[nb.get_thread_id(), nodeid] += exposed
        node_exp[:] = thread_exposed.sum(axis=0)

    @staticmethod
    @nb.njit((nb.uint32, nb.uint32, nb.uint16[:], nb.uint16), parallel=True, cache=True)
    def _nb_set_etimers_slice(istart, iend, etimers, value):
        for i in nb.prange(istart, iend):
            etimers[i] = value

    @staticmethod
    @nb.njit((nb.int64[:], nb.uint16[:], nb.uint16), parallel=True, cache=True)
    def _nb_set_etimers_randomaccess(indices, etimers, value):
        for i in nb.prange(len(indices)):
            etimers[indices[i]] = value
