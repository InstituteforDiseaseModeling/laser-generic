"""
This module defines the Susceptibility class for managing and visualizing
the susceptibility of a population in LASER models.
"""

import numba as nb
import numpy as np
from matplotlib.figure import Figure


class Susceptibility:
    """
    A LASER model component representing the susceptibility of a population.

    This component adds a scalar property ``susceptibility`` to the population
    and a vector property to the patches. It tracks which agents are susceptible
    at any given tick and provides helper functions for updating these values.
    """

    def __init__(self, model, verbose: bool = False):
        """
        Initialize the susceptibility component of the model.

        Parameters
        ----------
        model : object
            A LASER model instance. Must have ``population``, ``patches``, and ``params``
            with the following:
              * population.add_scalar_property(name, dtype, default)
              * patches.add_vector_property(name, nticks, dtype)
              * params.nticks (int) – number of simulation ticks.
        verbose : bool, optional
            If True, enables verbose output. Default is False.

        Notes
        -----
        - This method adds a scalar property ``susceptibility`` to each agent in
          the population, initialized to ``1`` (susceptible).
        - It also adds a vector property ``susceptibility`` to each patch, which
          tracks aggregated susceptible counts over time.
        """
        self.model = model

        model.population.add_scalar_property("susceptibility", dtype=np.uint8, default=1)
        model.patches.add_vector_property("susceptibility", model.params.nticks, dtype=np.uint32)
        return

    def __call__(self, model, tick: int):
        """
        Make the component callable. Required for LASER component interface.

        Parameters
        ----------
        model : object
            The LASER model containing population and patches.
        tick : int
            Current simulation tick.

        Returns
        -------
        None

        Notes
        -----
        This component does not perform per-tick updates. The method exists
        only to satisfy the LASER component interface.
        """
        return

    def census(self, model, tick: int) -> None:
        """
        Aggregate susceptible counts into patches at the given tick.

        Parameters
        ----------
        model : object
            LASER model containing ``population`` and ``patches``.
        tick : int
            Current simulation tick.

        Returns
        -------
        None

        Notes
        -----
        - At tick 0, this computes susceptible counts from the population
          and writes them into ``patches.susceptibility``.
        - For a single patch, counts are summed globally.
        - For multiple patches, counts are distributed by agent ``nodeid``.
        - In all cases, values are copied into ``patches.susceptibility_test``
          for validation/debugging.
        """
        patches = model.patches
        if tick == 0:
            population = model.population
            susceptible_count = patches.susceptibility[tick, :]
            condition = population.susceptibility[0:population.count] > 0

            if len(model.patches) == 1:
                np.add(susceptible_count, np.count_nonzero(condition), out=susceptible_count)
            else:
                nodeids = population.nodeid[0:population.count]
                np.add.at(susceptible_count, nodeids[condition], np.uint32(1))

            patches.susceptibility_test[tick, :] = patches.susceptibility[tick, :].copy()

        # Carry forward susceptibility_test to next tick
        patches.susceptibility_test[tick + 1, :] = patches.susceptibility_test[tick, :].copy()
        return

    def on_birth(self, model, _tick: int, istart, iend):
        """
        Handle the birth event by setting susceptibility of newborns.

        Parameters
        ----------
        model : object
            LASER model containing the population.
        _tick : int
            Current simulation tick (unused).
        istart : int or ndarray of int64
            Start index of newborns (if slice mode), or array of agent indices (if random-access mode).
        iend : int or None
            End index of newborns (exclusive). If None, ``istart`` is treated as
            an index array for random-access updates.

        Returns
        -------
        None

        Notes
        -----
        - Newborns are set to susceptible (1).
        - Uses either slice-based or random-access update functions.
        """
        if iend is not None:
            Susceptibility._nb_set_susceptibility_slice(
                istart, iend, model.population.susceptibility, np.uint8(1)
            )
        else:
            Susceptibility._nb_set_susceptibility_randomaccess(
                istart, model.population.susceptibility, np.uint8(1)
            )
        return

    def plot(self, fig: Figure = None):
        """
        Placeholder for plotting susceptibility distribution by age.

        Parameters
        ----------
        fig : Figure, optional
            A Matplotlib Figure to draw into. If None, a new figure would be created
            in a full implementation.

        Yields
        ------
        None
            The generator currently yields once (``None``). This is a placeholder
            and should be replaced with actual plotting logic in the future.
        """
        yield
        return

    # -------------------------------------------------------------------------
    # Private Numba-compiled helper functions
    # -------------------------------------------------------------------------

    @staticmethod
    @nb.njit((nb.uint32, nb.int32[:], nb.uint8[:]), parallel=True, cache=True)
    def _nb_initialize_susceptibility(count, dob, susceptibility) -> None:  # pragma: no cover
        for i in nb.prange(count):
            susceptibility[i] = 1
        return

    @staticmethod
    @nb.njit((nb.uint32, nb.uint32, nb.uint8[:], nb.uint8), parallel=True, cache=True)
    def _nb_set_susceptibility_slice(istart, iend, susceptibility, value) -> None:  # pragma: no cover
        for i in nb.prange(istart, iend):
            susceptibility[i] = value
        return

    @staticmethod
    @nb.njit((nb.int64[:], nb.uint8[:], nb.uint8), parallel=True, cache=True)
    def _nb_set_susceptibility_randomaccess(indices, susceptibility, value) -> None:  # pragma: no cover
        for i in nb.prange(len(indices)):
            susceptibility[indices[i]] = value
        return

    @staticmethod
    @nb.njit((nb.uint32[:], nb.bool_[:], nb.uint16[:], nb.int64), parallel=True, cache=True)
    def _accumulate_susceptibility(node_susc, agent_susc, nodeids, count) -> None:  # pragma: no cover
        max_node_id = np.max(nodeids)
        thread_susceptibilities = np.zeros(
            (nb.config.NUMBA_DEFAULT_NUM_THREADS, max_node_id + 1),
            dtype=np.uint32,
        )

        for i in nb.prange(count):
            nodeid = nodeids[i]
            susceptibility = agent_susc[i]
            thread_susceptibilities[nb.get_thread_id(), nodeid] += susceptibility

        node_susc[:] = thread_susceptibilities.sum(axis=0)
        return
