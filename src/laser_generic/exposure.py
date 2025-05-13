"""
This module defines the Exposure class, which models the transition between being infected and become infectious.

Classes:
    Exposure: A class to handle exposed state updates, initialization, and plotting of exposed data.

Functions:
    Exposure.__init__(self, model, verbose: bool = False) -> None:
        Initializes the Exposure class with a given model and verbosity option.

    Exposure.__call__(self, model, tick) -> None:
        Updates the exposed status of the population at each tick.

    Exposure.nb_exposure_update(count, itimers):
        A static method that updates the exposed timers for the population using Numba for performance.

    Exposure.on_birth(self, model, _tick, istart, iend) -> None:
        Resets the exposed timer for newborns in the population.

    Exposure.nb_set_etimers(istart, iend, itimers, value) -> None:
        A static method that sets the exposure timers for a range of individuals in the population using Numba for performance.

    Exposure.plot(self, fig: Figure = None):
        Plots the exposed data by age using Matplotlib.
"""

import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


class Exposure:
    """
    A class to define the exposed state of individuals in a population.
    """

    def __init__(self, model, verbose: bool = False) -> None:
        """
        Initialize an Exposed instance.

        Args:

            model: The model object that contains the population.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Attributes:

            model: The model object that contains the population.

        Side Effects:

            Adds a scalar property "etimer" to the model's population with dtype np.uint16 and default value 0.
            Calls the nb_set_etimers method to initialize the itimer values for the population.
        """

        self.model = model

        model.population.add_scalar_property("etimer", dtype=np.uint16, default=0)
        model.patches.add_vector_property("exposed", length=model.params.nticks, dtype=np.uint32)
        Exposure.nb_set_etimers_slice(0, model.population.count, model.population.etimer, 0)

        return

    def census(self, model, tick) -> None:
        population = model.population
        patches = model.patches
        exposed_count = patches.exposed[tick, :]
        condition = population.etimer[0 : population.count] > 0

        if len(patches) == 1:
            np.add(
                exposed_count,
                np.count_nonzero(condition),  # if you are susceptible or infected, you're not recovered
                out=exposed_count,
            )
        else:
            nodeids = population.nodeid[0 : population.count]
            self.accumulate_exposed(exposed_count, condition, nodeids, population.count)
            #np.add.at(recovered_count, nodeids[condition], np.uint32(1))
        return

    def __call__(self, model, tick) -> None:
        """
        Updates the exposed timers for the population in the model.

        Args:

            model: The model containing the population data.
            tick: The current tick or time step in the simulation.

        Returns:

            None
        """

        Exposure.nb_exposure_update(model.population.count, model.population.itimer)

        return

    @staticmethod
    @nb.njit((nb.uint32, nb.uint16[:]), parallel=True, cache=True)
    def nb_exposure_update(count, etimers):  # pragma: no cover
        """Numba compiled function to check and update exposed timers for the population in parallel."""
        for i in nb.prange(count):
            etimer = etimers[i]
            if etimer > 0:
                etimer -= 1
                etimers[i] = etimer

        return

    @staticmethod
    @nb.njit((nb.uint32[:], nb.bool_[:], nb.uint16[:], nb.int64), parallel=True, cache=True)
    def accumulate_exposed(node_exp, agent_exposed, nodeids, count) -> None:  # pragma: no cover
        """Numba compiled function to accumulate recovered individuals."""
        max_node_id = np.max(nodeids)
        thread_exposed = np.zeros((nb.config.NUMBA_DEFAULT_NUM_THREADS, max_node_id + 1), dtype=np.uint32)

        for i in nb.prange(count):
            nodeid = nodeids[i]
            recovered = agent_exposed[i]
            thread_exposed[nb.get_thread_id(), nodeid] += recovered
        for t in range(nb.config.NUMBA_DEFAULT_NUM_THREADS):
            for j in range(max_node_id + 1):
                node_exp[j] += thread_exposed[t, j]

        return

    def on_birth(self, model, _tick, istart, iend) -> None:
        """
        This function sets the exposure timer for newborns to zero, indicating that they are not exposed.

        Args:

            model: The simulation model containing the population data.
            tick: The current tick or time step in the simulation (unused in this function).
            istart: The starting index of the newborns in the population array.
            iend: The ending index of the newborns in the population array.

        Returns:

            None
        """

        # newborns are not exposed
        # Exposure.nb_set_itimers(istart, iend, model.population.itimer, 0)
        if iend is not None:
            Exposure.nb_set_etimers_slice(istart, iend, model.population.etimer, np.uint16(0))
        else:
            Exposure.nb_set_etimers_randomaccess(istart, model.population.etimer, np.uint16(0))
        return

    @staticmethod
    @nb.njit((nb.uint32, nb.uint32, nb.uint16[:], nb.uint16), parallel=True, cache=True)
    def nb_set_etimers_slice(istart, iend, etimers, value) -> None:  # pragma: no cover
        """Numba compiled function to set exposure timers for a range of individuals in parallel."""
        for i in nb.prange(istart, iend):
            etimers[i] = value

        return

    @staticmethod
    @nb.njit((nb.int64[:], nb.uint16[:], nb.uint16), parallel=True, cache=True)
    def nb_set_etimers_randomaccess(indices, etimers, value) -> None:  # pragma: no cover
        """Numba compiled function to set exposure timers for a range of individuals in parallel."""
        for i in nb.prange(len(indices)):
            etimers[indices[i]] = value

        return

    def plot(self, fig: Figure = None):
        """
        Plots the distribution of exposures by age.

        This function creates a bar chart showing the number of individuals in each age group,
        and overlays a bar chart showing the number of exposed individuals in each age group.

        Parameters:

            fig (Figure, optional): A Matplotlib Figure object to plot on. If None, a new figure is created.

        Yields:

            None: This function uses a generator to yield control back to the caller.
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
        return
