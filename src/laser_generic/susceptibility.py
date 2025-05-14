"""
This module defines the Susceptibility class and associated functions for managing and visualizing
the susceptibility of a population.

Classes:

    Susceptibility: Manages the susceptibility property of a population and provides methods for
                    updating and plotting susceptibility data.

    Susceptibility_SI: Manages the susceptibility property of a population for the SI model.

Functions:

    nb_initialize_susceptibility(count, dob, susceptibility):
        Initializes the susceptibility of individuals in the population.

    nb_set_susceptibility(istart, iend, susceptibility, value):
        Sets the susceptibility of a range of individuals in the population to a specified value.
"""

import numba as nb
import numpy as np
from matplotlib.figure import Figure


class Susceptibility:
    """
    A component to represent the susceptibility of a population in a model.
    """

    def __init__(self, model, verbose: bool = False):
        """
        Initialize the susceptibility component of the model.

        Parameters:

            model : object

                The model object that contains the population data.

            verbose : bool, optional

                If True, enables verbose output (default is False).

        Attributes:

            model : object

                The model object passed to the initializer.

        The method also adds a scalar property "susceptibility" to the model's population
        with a default value of 1 and initializes the susceptibility values based on the
        population count, date of birth (dob), and susceptibility attributes.
        """

        self.model = model

        model.population.add_scalar_property("susceptibility", dtype=np.uint8, default=1)
        model.patches.add_vector_property("susceptibility", model.params.nticks, dtype=np.uint32)
        # self.nb_initialize_susceptibility(model.population.count, model.population.dob, model.population.susceptibility)

        return

    @staticmethod
    @nb.njit((nb.uint32, nb.int32[:], nb.uint8[:]), parallel=True, cache=True)
    def nb_initialize_susceptibility(count, dob, susceptibility) -> None:  # pragma: no cover
        """Numba compiled function to initialize susceptibility based on date of birth."""
        for i in nb.prange(count):
            # 5 y.o. and older are _not_ susceptible (dobs are negative)
            susceptibility[i] = 1

        return

    @staticmethod
    @nb.njit((nb.uint32, nb.uint32, nb.uint8[:], nb.uint8), parallel=True, cache=True)
    def nb_set_susceptibility_slice(istart, iend, susceptibility, value) -> None:  # pragma: no cover
        """Numba compiled function to set the susceptibility of a range of individuals."""
        for i in nb.prange(istart, iend):
            susceptibility[i] = value

        return

    @staticmethod
    @nb.njit((nb.int64[:], nb.uint8[:], nb.uint8), parallel=True, cache=True)
    def nb_set_susceptibility_randomaccess(indices, susceptibility, value) -> None:  # pragma: no cover
        """Numba compiled function to set the susceptibility of a range of individuals."""
        for i in nb.prange(len(indices)):
            susceptibility[indices[i]] = value

        return

    @staticmethod
    @nb.njit((nb.uint32[:], nb.bool_[:], nb.uint16[:], nb.int64), parallel=True, cache=True)
    def accumulate_susceptibility(node_susc, agent_susc, nodeids, count) -> None:  # pragma: no cover
        """Numba compiled function to accumulate susceptibility of individuals."""
        max_node_id = np.max(nodeids)
        thread_susceptibilities = np.zeros((nb.config.NUMBA_DEFAULT_NUM_THREADS, max_node_id + 1), dtype=np.uint32)

        for i in nb.prange(count):
            nodeid = nodeids[i]
            susceptibility = agent_susc[i]
            thread_susceptibilities[nb.get_thread_id(), nodeid] += susceptibility
        node_susc[:] = thread_susceptibilities.sum(axis=0)

        return

    def __call__(self, model, tick):
        #     """
        #     This method allows the instance to be called as a function.

        #     Parameters:

        #         model (object): The model object that contains the state and behavior of the simulation.
        #         tick (int): The current tick or time step in the simulation.

        #     Returns:

        #         None
        #     """

        # Don't actually need to do anything here for generic models, but this method is required for the class to be callable.

        return

    def census(self, model, tick) -> None:
        patches = model.patches
        if tick == 0:
            population = model.population
            susceptible_count = patches.susceptibility[tick, :]  # we will accumulate current susceptibles into this view into the susceptibility array
            condition = population.susceptibility[0 : population.count]>0

            if len(model.patches) == 1:
                np.add(susceptible_count, np.count_nonzero(condition), out=susceptible_count)
            else:
                nodeids = population.nodeid[0 : population.count]
                #self.accumulate_susceptibility(susceptible_count, condition, nodeids, population.count)
                np.add.at(susceptible_count, nodeids[condition], np.uint32(1))  # increment by the number of active agents with non-zero itimer

            #if tick == 0:
            patches.susceptibility_test[tick, :] = patches.susceptibility[tick, :].copy()

        patches.susceptibility_test[tick+1, :] = patches.susceptibility_test[tick, :].copy()
        return

    def on_birth(self, model, _tick, istart, iend):
        """
        Handle the birth event in the model by setting the susceptibility of newborns.

        This method is called when a birth event occurs in the model. It sets the
        susceptibility of the newborns to 0, indicating that they are not susceptible
        to the disease.

        Parameters:

            model (object): The model object containing the population data.
            tick (int): The current tick or time step in the simulation.
            istart (int): The starting index of the newborns in the population array.
            iend (int): The ending index of the newborns in the population array.

        Returns:

            None
        """

        # newborns are susceptible
        # nb_set_susceptibility(istart, iend, model.population.susceptibility, 0)
        if iend is not None:
            Susceptibility.nb_set_susceptibility_slice(istart, iend, model.population.susceptibility, np.uint8(1))
        else:
            Susceptibility.nb_set_susceptibility_randomaccess(istart, model.population.susceptibility, np.uint8(1))

        return

    def plot(self, fig: Figure = None):
        """
        Plots the susceptibility distribution by age.

        Parameters:

            fig (Figure, optional): A Matplotlib Figure object. If None, a new figure is created with a size of 12x9 inches and a DPI of 128.

        Yields:

            None: This function uses a generator to yield control back to the caller.
        """

        # fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        # fig.suptitle("Susceptibility Distribution By Age")
        # age_bins = (self.model.params.nticks - self.model.population.dob[0 : self.model.population.count]) // 365
        # sus_counts = np.bincount(age_bins, weights=self.model.population.susceptibility[0 : self.model.population.count].astype(np.uint32))
        # age_counts = np.bincount(age_bins)
        # # TODO - convert this to %age of susceptible individuals by age group
        # plt.bar(range(len(age_counts)), age_counts)
        # plt.bar(range(len(sus_counts)), sus_counts, alpha=0.5)

        yield
        return
