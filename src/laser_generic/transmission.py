"""
This module defines the Transmission class, which models the transmission of measles in a population.

Classes:

    Transmission: A class to model the transmission dynamics of measles within a population.

Functions:

    Transmission.__init__(self, model, verbose: bool = False) -> None:

        Initializes the Transmission object with the given model and verbosity.

    Transmission.__call__(self, model, tick) -> None:

        Executes the transmission dynamics for a given model and tick.

    Transmission.nb_transmission_update(susceptibilities, nodeids, forces, etimers, count, exp_shape, exp_scale, incidence):

        A Numba-compiled static method to update the transmission dynamics in parallel.

    Transmission.plot(self, fig: Figure = None):

        Plots the cases and incidence for the two largest patches in the model.
"""

import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


class Transmission:
    """
    A component to model the transmission of disease in a population.
    """

    def __init__(self, model, verbose: bool = False) -> None:
        """
        Initializes the transmission object.

        Args:

            model: The model object that contains the patches and parameters.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Attributes:

            model: The model object passed during initialization.

        The model's patches are extended with the following properties:

            - 'cases': A vector property with length equal to the number of ticks, dtype is uint32.
            - 'forces': A scalar property with dtype float32.
            - 'incidence': A vector property with length equal to the number of ticks, dtype is uint32.
        """

        self.model = model

        model.patches.add_vector_property("cases", length=model.params.nticks, dtype=np.uint32)
        model.patches.add_scalar_property("forces", dtype=np.float32)
        model.patches.add_vector_property("incidence", model.params.nticks, dtype=np.uint32)
        model.population.add_scalar_property("doi", dtype=np.uint32, default=0)

        return

    def census(self, model, tick) -> None:
        patches = model.patches
        if tick == 0:
            population = model.population

            contagion = patches.cases[tick, :]  # we will accumulate current infections into this view into the cases array
            #condition = population.state[0:population.count] == 2  # just look at the active agent indices
            if hasattr(population, "itimer"):
                condition = population.itimer[0 : population.count] > 0  # just look at the active agent indices
            else:
                condition = population.susceptibility[0 : population.count] == 0  # just look at the active agent indices

            if len(patches) == 1:
                np.add(contagion, np.count_nonzero(condition), out=contagion)  # add.at takes a lot of time when n_infections is large
            else:
                nodeids = population.nodeid[0 : population.count]  # just look at the active agent indices
                np.add.at(contagion, nodeids[condition], np.uint32(1))  # increment by the number of active agents with non-zero itimer
            patches.cases_test[tick, :] = patches.cases[tick, :].copy()

        patches.cases_test[tick+1, :] = patches.cases_test[tick, :].copy()
        return

    def __call__(self, model, tick) -> None:
        """
        Simulate the transmission of measles for a given model at a specific tick.

        This method updates the state of the model by simulating the spread of disease
        through the population and patches. It calculates the contagion, handles the
        migration of infections between patches, and updates the forces of infection
        based on the effective transmission rate and seasonality factors. Finally, it
        updates the infected state of the population.

        Parameters:

            model (object): The model object containing the population, patches, and parameters.
            tick (int): The current time step in the simulation.

        Returns:

            None

        """
        patches = model.patches
        population = model.population

        #contagion = patches.cases[tick, :]  # we will accumulate current infections into this view into the cases array
        contagion = patches.cases_test[tick, :].copy().astype(np.float32)
        if hasattr(patches, "network"):
            network = patches.network
            transfer = contagion * network.T
            contagion += transfer.sum(axis=1)  # increment by incoming "migration"
            contagion -= transfer.sum(axis=0)  # decrement by outgoing "migration"

        forces = patches.forces
        beta_effective = model.params.beta
        if 'seasonality_factor' in model.params:
            beta_effective *= (1+ model.params.seasonality_factor * np.sin(
            2 * np.pi * (tick - model.params.seasonality_phase) / 365
            ))

        np.multiply(contagion, beta_effective, out=forces)
        np.divide(forces, model.patches.populations[tick, :], out=forces)  # per agent force of infection
        np.negative(forces, out=forces)
        np.expm1(forces, out=forces)
        np.negative(forces, out=forces)

        # TODO: This is a hack to handle the different transmission dynamics for all of these SIS, SI, SIR, SEIR, ... models.
        #       We should refactor this to be more general and flexible.
        #       First, find a way to allow user to parametrize the timer distributions rather than hard-coding here.
        #       For example, the "_exposed" & "_noexposed" functions have the same signature but a different timer distribution.
        #       Second, maybe there's a way to overload the update function so we don't have to switch on the population attributes.

        if hasattr(population, "etimer"):
            Transmission.nb_transmission_update_exposed(
                population.susceptibility,
                population.nodeid,
                population.state,
                forces,
                population.etimer,
                population.count,
                model.params.exp_shape,
                model.params.exp_scale,
                model.patches.incidence[tick, :],
                population.doi,
                tick,
            )
        elif hasattr(population, "itimer"):
            Transmission.nb_transmission_update_noexposed(
                population.susceptibility,
                population.nodeid,
                population.state,
                forces,
                population.itimer,
                population.count,
                model.params.inf_mean,
                model.patches.incidence[tick, :],
                population.doi,
                tick,
            )
        else:
            Transmission.nb_transmission_update_SI(
                population.susceptibility,
                population.nodeid,
                population.state,
                forces,
                population.count,
                model.patches.incidence[tick, :],
                population.doi,
                tick,
            )
        if hasattr(population, "etimer"):
            model.patches.exposed_test[tick+1, :] += model.patches.incidence[tick, :]
            model.patches.susceptibility_test[tick+1, :] -= model.patches.incidence[tick, :]
        else:
            model.patches.cases_test[tick+1, :] += model.patches.incidence[tick, :]
            model.patches.susceptibility_test[tick+1, :] -= model.patches.incidence[tick, :]

        return

    @staticmethod
    @nb.njit(
        (nb.uint8[:], nb.uint16[:], nb.uint8[:], nb.float32[:], nb.uint16[:], nb.uint32, nb.float32, nb.float32, nb.uint32[:], nb.uint32[:], nb.int_),
        parallel=True,
        nogil=True,
        cache=True,
    )
    def nb_transmission_update_exposed(
        susceptibilities, nodeids, state, forces, etimers, count, exp_shape, exp_scale, incidence, doi, tick
    ):  # pragma: no cover
        """Numba compiled function to stochastically transmit infection to agents in parallel."""
        max_node_id = np.max(nodeids)
        thread_incidences = np.zeros((nb.config.NUMBA_DEFAULT_NUM_THREADS, max_node_id + 1), dtype=np.uint32)

        for i in nb.prange(count):
            susceptibility = susceptibilities[i]
            if susceptibility > 0:
                nodeid = nodeids[i]
                force = susceptibility * forces[nodeid]  # force of infection attenuated by personal susceptibility
                if (force > 0) and (np.random.random_sample() < force):  # draw random number < force means infection
                    susceptibilities[i] = 0  # no longer susceptible
                    # set exposure timer for newly infected individuals to a draw from a gamma distribution, must be at least 1 day
                    etimers[i] = np.maximum(np.uint16(1), np.uint16(np.round(np.random.gamma(exp_shape, exp_scale))))
                    state[i] = 1  # set state to exposed
                    doi[i] = tick
                    thread_incidences[nb.get_thread_id(), nodeid] += 1

        incidence[:] = thread_incidences.sum(axis=0)

        return

    @staticmethod
    @nb.njit(
        (nb.uint8[:], nb.uint16[:], nb.uint8[:], nb.float32[:], nb.uint16[:], nb.uint32, nb.float32, nb.uint32[:], nb.uint32[:], nb.int_),
        parallel=True,
        nogil=True,
        cache=True,
    )
    def nb_transmission_update_noexposed(
        susceptibilities, nodeids, state, forces, itimers, count, inf_mean, incidence, doi, tick
    ):  # pragma: no cover
        """Numba compiled function to stochastically transmit infection to agents in parallel."""
        max_node_id = np.max(nodeids)
        thread_incidences = np.zeros((nb.config.NUMBA_DEFAULT_NUM_THREADS, max_node_id + 1), dtype=np.uint32)

        for i in nb.prange(count):
            susceptibility = susceptibilities[i]
            if susceptibility > 0:
                nodeid = nodeids[i]
                force = susceptibility * forces[nodeid]  # force of infection attenuated by personal susceptibility
                if (force > 0) and (np.random.random_sample() < force):  # draw random number < force means infection
                    susceptibilities[i] = 0  # no longer susceptible
                    # set infectious timer for the individual to an exponential draw
                    itimers[i] = np.maximum(np.uint16(1), np.uint16(np.ceil(np.random.exponential(inf_mean))))
                    doi[i] = tick
                    state[i] = 2
                    thread_incidences[nb.get_thread_id(), nodeid] += 1

        # for t in range(nb.config.NUMBA_DEFAULT_NUM_THREADS):
        #    for j in range(max_node_id + 1):
        #        incidence[j] += thread_incidences[t, j]
        incidence[:] = thread_incidences.sum(axis=0)

        return

    @staticmethod
    @nb.njit(
        (nb.uint8[:], nb.uint16[:], nb.uint8[:], nb.float32[:], nb.uint32, nb.uint32[:], nb.uint32[:], nb.int_),
        parallel=True,
        nogil=True,
        cache=True,
    )
    def nb_transmission_update_SI(susceptibilities, nodeids, state, forces, count, incidence, doi, tick):  # pragma: no cover
        """Numba compiled function to stochastically transmit infection to agents in parallel."""
        max_node_id = np.max(nodeids)
        thread_incidences = np.zeros((nb.config.NUMBA_DEFAULT_NUM_THREADS, max_node_id + 1), dtype=np.uint32)
        for i in nb.prange(count):
            susceptibility = susceptibilities[i]
            if susceptibility > 0:
                nodeid = nodeids[i]
                force = susceptibility * forces[nodeid]  # force of infection attenuated by personal susceptibility
                if (force > 0) and (np.random.random_sample() < force):  # draw random number < force means infection
                    # All we do is become no longer susceptible, which means infected in an SI model.  No timers.
                    susceptibilities[i] = 0  # no longer susceptible
                    doi[i] = tick
                    state[i] = 2
                    thread_incidences[nb.get_thread_id(), nodeid] += 1
        for t in range(nb.config.NUMBA_DEFAULT_NUM_THREADS):
            for j in range(max_node_id + 1):
                incidence[j] += thread_incidences[t, j]

        return

    def on_birth(self, model, _tick, istart, iend) -> None:
        """
        This function sets the date of infection for newborns to zero.
        Appears here because transmission is where I have decided to add the "doi" property,
        and I think it thus makes sense to also have the on-birth initializer here.  Could
        just as easily choose to do this over in Infection class instead.

        Args:

            model: The simulation model containing the population data.
            tick: The current tick or time step in the simulation (unused in this function).
            istart: The starting index of the newborns in the population array.
            iend: The ending index of the newborns in the population array.

        Returns:

            None
        """

        if iend is not None:
            model.population.doi[istart:iend] = 0
        else:
            model.population.doi[istart] = 0
        return

    def plot(self, fig: Figure = None):
        """
        Plots the cases and incidence for the two largest patches in the model.

        This function creates a figure with four subplots:

            - Cases for the largest patch
            - Incidence for the largest patch
            - Cases for the second largest patch
            - Incidence for the second largest patch

        If no figure is provided, a new figure is created with a size of 12x9 inches and a DPI of 128.

        Parameters:

            fig (Figure, optional): A Matplotlib Figure object to plot on. If None, a new figure is created.

        Yields:

            None
        """

        fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        fig.suptitle("Cases and Incidence for Two Largest Patches")

        itwo, ione = np.argsort(self.model.patches.populations[-1, :])[-2:]

        fig.add_subplot(2, 2, 1)
        plt.title(f"Cases - Node {ione}")  # ({self.names[ione]})")
        plt.plot(self.model.patches.cases[:, ione])

        fig.add_subplot(2, 2, 2)
        plt.title(f"Incidence - Node {ione}")  # ({self.names[ione]})")
        plt.plot(self.model.patches.incidence[:, ione])

        fig.add_subplot(2, 2, 3)
        plt.title(f"Cases - Node {itwo}")  # ({self.names[itwo]})")
        plt.plot(self.model.patches.cases[:, itwo])

        fig.add_subplot(2, 2, 4)
        plt.title(f"Incidence - Node {itwo}")  # ({self.names[itwo]})")
        plt.plot(self.model.patches.incidence[:, itwo])

        yield
        return


class TransmissionSIR(Transmission):
    """
    A component to model the transmission of disease in a population using the SIR model.
    """

    def __call__(self, model, tick) -> None:
        """
        Simulate the transmission of measles for a given model at a specific tick using the SIR model.

        This method updates the state of the model by simulating the spread of disease
        through the population and patches. It calculates the contagion, handles the
        migration of infections between patches, and updates the forces of infection
        based on the effective transmission rate and seasonality factors. Finally, it
        updates the infected state of the population.

        Parameters:

            model (object): The model object containing the population, patches, and parameters.
            tick (int): The current time step in the simulation.

        Returns:

            None
        """
        patches = model.patches
        population = model.population

        contagion = patches.cases[tick, :]
        condition = population.susceptibility[0 : population.count] == 0

        if len(patches) == 1:
            np.add(contagion, np.sum(condition), out=contagion)
        else:
            nodeids = population.nodeid[0 : population.count]
            np.add.at(contagion, nodeids[condition], 1)

        if hasattr(patches, "network"):
            network = patches.network
            transfer = (contagion * network).round().astype(np.uint32)
            contagion += transfer.sum(axis=1)
            contagion -= transfer.sum(axis=0)

        forces = patches.forces
        beta_effective = model.params.beta

        np.multiply(contagion, beta_effective, out=forces)
        np.divide(forces, model.patches.populations[tick, :], out=forces)
        np.negative(forces, out=forces)
        np.expm1(forces, out=forces)
        np.negative(forces, out=forces)

        Transmission.nb_transmission_update_noexposed(
            population.susceptibility,
            population.nodeid,
            forces,
            population.itimer,
            population.count,
            model.params.inf_mean,
            model.patches.incidence[tick, :],
            population.doi,
            tick,
        )

        return
