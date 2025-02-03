"""
This module defines the Infection class, which models the infection dynamics within a population.

Classes:
    Infection: A class to handle infection updates, initialization, and plotting of infection data.
    Infection_SIS: Duplicate of infection class, but sets susceptibility to 1 when itimer hits zero.

Functions:
    Infection.__init__(self, model, verbose: bool = False) -> None:
        Initializes the Infection class with a given model and verbosity option.

    Infection.__call__(self, model, tick) -> None:
        Updates the infection status of the agents at each tick.

    Infection.nb_infection_update(count, itimers):
        A static method that updates the infection timers for the agents using Numba for performance.

    Infection.on_birth(self, model, _tick, istart, iend) -> None:
        Resets the infection timer for newborns in the population.

    Infection.nb_set_itimers(istart, iend, itimers, value) -> None:
        A static method that sets the infection timers for a range of individuals in the population using Numba for performance.

    Infection.plot(self, fig: Figure = None):
        Plots the infection data by age using Matplotlib.
"""

import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


class Infection:
    """
    A component to update the infection timers of agents in a model.
    """

    def __init__(self, model, verbose: bool = False) -> None:
        """
        Initialize an Infection instance.

        Args:

            model: The model object that contains the agents.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Attributes:

            model: The model object that contains the agents.

        Side Effects:

            Adds a scalar property "itimer" to the model's agents with dtype np.uint16 and default value 0.
            Calls the nb_set_itimers method to initialize the itimer values for the agents.
        """

        self.model = model

        model.agents.add_scalar_property("itimer", dtype=np.uint16, default=0)
        model.patches.add_vector_property("recovered", length=model.params.nticks, dtype=np.uint32)
        Infection.nb_set_itimers_slice(0, model.agents.count, model.agents.itimer, 0)

        return

    def census(self, model, tick) -> None:
        agents = model.agents
        patches = model.patches
        recovered_count = patches.recovered[tick, :]
        rec = np.logical_and(agents.susceptibility[0 : agents.count] == 0, agents.itimer[0 : agents.count] == 0)

        if len(model.patches) == 1:
            np.add(
                recovered_count,
                np.count_nonzero(rec),  # if you are susceptible or infected, you're not recovered
                out=recovered_count,
            )
        else:
            nodeids = agents.nodeid[0 : agents.count]
            self.accumulate_recovered(recovered_count, rec, nodeids, agents.count)
        return

    def __call__(self, model, tick) -> None:
        """
        Updates the infection timers for the agents in the model.

        Args:

            model: The model containing the agent data.
            tick: The current tick or time step in the simulation.

        Returns:

            None
        """

        Infection.nb_infection_update(model.agents.count, model.agents.itimer)

        return

    @staticmethod
    @nb.njit((nb.uint32, nb.uint16[:]), parallel=True, cache=True)
    def nb_infection_update(count, itimers):  # pragma: no cover
        """Numba compiled function to check and update infection timers for the agents in parallel."""
        for i in nb.prange(count):
            itimer = itimers[i]
            if itimer > 0:
                itimer -= 1
                itimers[i] = itimer

        return

    @staticmethod
    @nb.njit(
        [
            (nb.uint32[:], nb.uint32[:], nb.uint16[:], nb.uint32),
            (nb.uint32[:], nb.bool[:], nb.uint16[:], nb.int64),
        ],
        parallel=True,
        cache=True,
    )
    def accumulate_recovered(node_rec, agent_recovered, nodeids, count) -> None:  # pragma: no cover
        """Numba compiled function to accumulate recovered individuals."""
        max_node_id = np.max(nodeids)
        thread_recovereds = np.zeros((nb.config.NUMBA_DEFAULT_NUM_THREADS, max_node_id + 1), dtype=np.uint32)

        for i in nb.prange(count):
            nodeid = nodeids[i]
            recovered = agent_recovered[i]
            thread_recovereds[nb.get_thread_id(), nodeid] += recovered
        for t in range(nb.config.NUMBA_DEFAULT_NUM_THREADS):
            for j in range(max_node_id + 1):
                node_rec[j] += thread_recovereds[t, j]

        return

    def on_birth(self, model, _tick, istart, iend) -> None:
        """
        This function sets the infection timer for newborns to zero, indicating that they are not infectious.

        Args:

            model: The simulation model containing the agents data.
            tick: The current tick or time step in the simulation (unused in this function).
            istart: The starting index of the newborns in the agents array.
            iend: The ending index of the newborns in the agents array.

        Returns:

            None
        """

        # newborns are not infectious
        # Infection.nb_set_itimers(istart, iend, model.agents.itimer, 0)
        if iend is not None:
            Infection.nb_set_itimers_slice(istart, iend, model.agents.itimer, np.uint16(0))
        else:
            Infection.nb_set_itimers_randomaccess(istart, model.agents.itimer, np.uint16(0))
        return

    @staticmethod
    @nb.njit((nb.uint32, nb.uint32, nb.uint16[:], nb.uint16), parallel=True, cache=True)
    def nb_set_itimers_slice(istart, iend, itimers, value) -> None:  # pragma: no cover
        """Numba compiled function to set infection timers for a range of individuals in parallel."""
        for i in nb.prange(istart, iend):
            itimers[i] = value

        return

    @staticmethod
    @nb.njit((nb.int64[:], nb.uint16[:], nb.uint16), parallel=True, cache=True)
    def nb_set_itimers_randomaccess(indices, itimers, value) -> None:  # pragma: no cover
        """Numba compiled function to set infection timers for a range of individuals in parallel."""
        for i in nb.prange(len(indices)):
            itimers[indices[i]] = value

        return

    def plot(self, fig: Figure = None):
        """
        Plots the distribution of infections by age.

        This function creates a bar chart showing the number of individuals in each age group,
        and overlays a bar chart showing the number of infected individuals in each age group.

        Parameters:

            fig (Figure, optional): A Matplotlib Figure object to plot on. If None, a new figure is created.

        Yields:

            None: This function uses a generator to yield control back to the caller.
        """

        fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        fig.suptitle("Infections By Age")

        ages_in_years = (self.model.params.nticks - self.model.agents.dob[0 : self.model.agents.count]) // 365
        age_counts = np.bincount(ages_in_years)
        plt.bar(range(len(age_counts)), age_counts)
        itimers = self.model.agents.itimer[0 : self.model.agents.count]
        infected = itimers > 0
        infection_counts = np.bincount(ages_in_years[infected])
        plt.bar(range(len(infection_counts)), infection_counts)

        yield
        return


class Infection_SIS:
    """
    A component to update the infection timers of agents in a model.
    """

    def __init__(self, model, verbose: bool = False) -> None:
        """
        Initialize an Infection instance.

        Args:

            model: The model object that contains the agents.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Attributes:

            model: The model object that contains the agents.

        Side Effects:

            Adds a scalar property "itimer" to the model's agents with dtype np.uint16 and default value 0.
            Calls the nb_set_itimers method to initialize the itimer values for the agents.
        """

        self.model = model

        model.agents.add_scalar_property("itimer", dtype=np.uint16, default=0)
        Infection_SIS.nb_set_itimers(0, model.agents.count, model.agents.itimer, 0)

        return

    def __call__(self, model, tick) -> None:
        """
        Updates the infection timers for the agents in the model.

        Args:

            model: The model containing the agents data.
            tick: The current tick or time step in the simulation.

        Returns:

            None
        """

        Infection_SIS.nb_infection_update(model.agents.count, model.agents.itimer, model.agents.susceptibility)
        return

    @staticmethod
    @nb.njit((nb.uint32, nb.uint16[:], nb.uint8[:]), parallel=True, cache=True)
    def nb_infection_update(count, itimers, susceptibility):  # pragma: no cover
        """Numba compiled function to check and update infection timers for the agents in parallel."""
        for i in nb.prange(count):
            itimer = itimers[i]
            if itimer > 0:
                itimer -= 1
                itimers[i] = itimer
                if itimer == 0:
                    susceptibility[i] = 1

        return

    def on_birth(self, model, _tick, istart, iend) -> None:
        """
        This function sets the infection timer for newborns to zero, indicating that they are not infectious.

        Args:

            model: The simulation model containing the agents data.
            tick: The current tick or time step in the simulation (unused in this function).
            istart: The starting index of the newborns in the agents array.
            iend: The ending index of the newborns in the agents array.

        Returns:

            None
        """

        # newborns are not infectious
        # Infection.nb_set_itimers(istart, iend, model.agents.itimer, 0)
        model.agents.itimer[istart:iend] = 0
        return

    @staticmethod
    @nb.njit((nb.uint32, nb.uint32, nb.uint16[:], nb.uint16), parallel=True, cache=True)
    def nb_set_itimers(istart, iend, itimers, value) -> None:  # pragma: no cover
        """Numba compiled function to set infection timers for a range of individuals in parallel."""
        for i in nb.prange(istart, iend):
            itimers[i] = value

        return

    def plot(self, fig: Figure = None):
        """
        Plots the distribution of infections by age.

        This function creates a bar chart showing the number of individuals in each age group,
        and overlays a bar chart showing the number of infected individuals in each age group.

        Parameters:

            fig (Figure, optional): A Matplotlib Figure object to plot on. If None, a new figure is created.

        Yields:

            None: This function uses a generator to yield control back to the caller.
        """

        fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        fig.suptitle("Infections By Age")

        ages_in_years = (self.model.params.nticks - self.model.agents.dob[0 : self.model.agents.count]) // 365
        age_counts = np.bincount(ages_in_years)
        plt.bar(range(len(age_counts)), age_counts)
        itimers = self.model.agents.itimer[0 : self.model.agents.count]
        infected = itimers > 0
        infection_counts = np.bincount(ages_in_years[infected])
        plt.bar(range(len(infection_counts)), infection_counts)

        yield
        return
