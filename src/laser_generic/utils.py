"""
This module provides utility functions for the laser-measles project.

Functions:
    calc_distances(latitudes: np.ndarray, longitudes: np.ndarray, verbose: bool = False) -> np.ndarray:
        Calculate the pairwise distances between points given their latitudes and longitudes.

    calc_capacity(population: np.uint32, nticks: np.uint32, cbr: np.float32, verbose: bool = False) -> np.uint32:
        Calculate the population capacity after a given number of ticks based on a constant birth rate.

    seed_infections_randomly(model, ninfections: int = 100) -> None:
        Seed initial infections in random locations at the start of the simulation.

    seed_infections_in_patch(model, ipatch: int, ninfections: int = 100) -> None:
        Seed initial infections in a specific location at the start of the simulation.

    set_initial_susceptibility_randomly(model, susc_frac: float = 1.0) -> None:
        Set the population susceptibility level at the start of the simulation.

    set_initial_susceptibility_in_patch(model, ipatch: int, susc_frac: float = 1.0) -> None:
        Set the population susceptibility level at the start of the simulation in a specific patch.
"""

from math import ceil

import click
import numpy as np
from laser_core import PropertySet
from laser_core.migration import distance


def calc_distances(latitudes: np.ndarray, longitudes: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Calculate the pairwise distances between points given their latitudes and longitudes.

    Parameters:

        latitudes (np.ndarray): A 1-dimensional array of latitudes.
        longitudes (np.ndarray): A 1-dimensional array of longitudes with the same shape as latitudes.
        verbose (bool, optional): If True, prints the upper left corner of the distance matrix. Default is False.

    Returns:

        np.ndarray: A 2-dimensional array where the element at [i, j] represents the distance between the i-th and j-th points.

    Raises:

        AssertionError: If latitudes is not 1-dimensional or if latitudes and longitudes do not have the same shape.
    """

    assert latitudes.ndim == 1, "Latitude array must be one-dimensional"
    assert longitudes.shape == latitudes.shape, "Latitude and longitude arrays must have the same shape"
    npatches = len(latitudes)
    distances = np.zeros((npatches, npatches), dtype=np.float32)
    for i, (lat, long) in enumerate(zip(latitudes, longitudes)):
        distances[i, :] = distance(lat, long, latitudes, longitudes)

    if verbose:
        click.echo(f"Upper left corner of distance matrix:\n{distances[0:4, 0:4]}")

    return distances


def calc_capacity(population: np.uint32, nticks: np.uint32, cbr: np.float32, verbose: bool = False) -> np.uint32:
    """
    Calculate the population capacity after a given number of ticks based on a constant birth rate (CBR).

    Args:

        population (np.uint32): The initial population.
        nticks (np.uint32): The number of ticks (time steps) to simulate.
        cbr (np.float32): The constant birth rate per 1000 people per year.
        verbose (bool, optional): If True, prints detailed population growth information. Defaults to False.

    Returns:

        np.uint32: The estimated population capacity after the given number of ticks.
    """

    # We assume a constant birth rate (CBR) for the population growth
    # The formula is: P(t) = P(0) * (1 + CBR)^t
    # where P(t) is the population at time t, P(0) is the initial population, and t is the number of ticks
    # We need to allocate space for the population data for each tick
    # We will use the maximum population growth to estimate the capacity
    # We will use the maximum population growth to estimate the capacity

    # KM we should think about unitization of births, people might want to define it otherwise.  Should we consider using a
    # package that actually handles unitization, like pint?  Probably not, but just wanted to put that here.
    daily_rate = (cbr / 1000) / 365.0  # CBR is per 1000 people per year
    capacity = np.uint32(population * (1 + daily_rate) ** nticks)

    if verbose:
        click.echo(f"Population growth: {population:,} … {capacity:,}")
        alternate = np.uint32(population * (1 + cbr / 1000) ** (nticks / 365))  # KM What is the purpose of this line?
        click.echo(f"Alternate growth:  {population:,} … {alternate:,}")

    return capacity


# Want to think about the ways to seed infections.  Not all infections have a timer!
def seed_infections_randomly_SI(model, ninfections: int = 100) -> None:
    """
    Seed initial infections in random locations at the start of the simulation.
    This function randomly selects individuals from the population and seeds
    them with an infection, based on the specified number of initial infections.

    Args:

        model: The simulation model containing the population and parameters.
        ninfections (int, optional): The number of initial infections to seed.
                                     Defaults to 100.

    Returns:

        None
    """

    # Seed initial infections in random locations at the start of the simulation
    cinfections = 0
    while cinfections < ninfections:
        index = model.prng.integers(0, model.population.count)
        if model.population.susceptibility[index] > 0:
            model.population.susceptibility[index] = 0
            cinfections += 1

    return


def seed_infections_randomly(model, ninfections: int = 100) -> None:
    """
    Seed initial infections in random locations at the start of the simulation.
    This function randomly selects individuals from the population and seeds
    them with an infection, based on the specified number of initial infections.

    Args:

        model: The simulation model containing the population and parameters.
        ninfections (int, optional): The number of initial infections to seed.
                                     Defaults to 100.

    Returns:

        None
    """

    # Seed initial infections in random locations at the start of the simulation
    pop = model.population
    params = model.params

    myinds = np.flatnonzero(pop.susceptibility)
    if len(myinds) > ninfections:
        myinds = np.random.permutation(myinds)[:ninfections]

    # all_inds = np.random.permutation(len(pop.susceptibility))
    # myinds = []

    # for i in all_inds:
    #     if pop.susceptibility[i] > 0:
    #         myinds.append(i)
    #         if len(myinds) == ninfections:
    #             break
    # myinds = np.array(myinds)

    pop.itimer[myinds] = params.inf_mean
    pop.susceptibility[myinds] = 0
    inf_nodeids = pop.nodeid[myinds]
    # myinds = np.where(model.population.susceptibility > 0)[0]
    # if len(myinds) > ninfections:
    #     myinds = np.random.choice(myinds, ninfections, replace=False)
    # model.population.itimer[myinds] = model.params.inf_mean
    # model.population.susceptibility[myinds] = 0
    # inf_nodeids = model.population.nodeid[myinds]

    return inf_nodeids


def seed_infections_in_patch(model, ipatch: int, ninfections: int = 1) -> None:
    """
    Seed initial infections in a specific patch of the population at the start of the simulation.
    This function randomly selects individuals from the specified patch and sets their infection timer
    to the mean infection duration, effectively marking them as infected. The process continues until
    the desired number of initial infections is reached.

    Args:

        model: The simulation model containing the population and parameters.
        ipatch (int): The identifier of the patch where infections should be seeded.
        ninfections (int, optional): The number of initial infections to seed. Defaults to 100.

    Returns:

        None
    """

    # Seed initial infections in a specific location at the start of the simulation
    myinds = np.where((model.population.susceptibility > 0) & (model.population.nodeid == ipatch))[0]
    if len(myinds) > ninfections:
        myinds = np.random.choice(myinds, ninfections, replace=False)
    model.population.itimer[myinds] = model.params.inf_mean
    model.population.susceptibility[myinds] = 0

    return


def set_initial_susceptibility_in_patch(model, ipatch: int, susc_frac: float = 1.0) -> None:
    """
    Set the population susceptibility level at the start of the simulation, in a specific patch.
    This function randomly selects individuals from the patch and changes
    their susceptibility to zero, according to the parameter susc_frac.

    Args:

        model: The simulation model containing the population and parameters.
        ipatch: The patch to set susceptibility in
        susc_frac (float, optional): The fraction of individuals to keep susceptible.

    Returns:

        None
    """

    # Seed initial infections in random locations at the start of the simulation
    indices = np.squeeze(np.where(model.population.nodeid == ipatch))
    patch_indices = model.prng.choice(indices, int(len(indices) * (1 - susc_frac)), replace=False)
    model.population.susceptibility[patch_indices] = 0

    return


def set_initial_susceptibility_randomly(model, susc_frac: float = 1.0) -> None:
    """
    Set the population susceptibility level at the start of the simulation.
    This function randomly selects individuals from the population and changes
    their susceptibility to zero, according to the parameter susc_frac.

    Args:

        model: The simulation model containing the population and parameters.
        susc_frac (float, optional): The fraction of individuals to keep susceptible.

    Returns:

        None
    """

    # Seed initial infections in random locations at the start of the simulation
    indices = model.prng.choice(model.population.count, int(model.population.count * (1 - susc_frac)), replace=False)
    model.population.susceptibility[indices] = 0

    return


def add_at(A, indices, B):
    sorted_indices = np.argsort(indices)
    uniques, run_lengths = np.unique(indices[sorted_indices], return_counts=True)
    for i, length, end in zip(uniques, run_lengths, run_lengths.cumsum()):
        A[i] += B[sorted_indices[end - length : end]].sum(axis=0)



    Each parameter in the returned PropertySet is described below, along with its default value and rationale:

        nticks (int, default=730): Number of simulation ticks (days). Default is 2 years (365*2), which is a typical duration for seasonal epidemic simulations.
        beta (float, default=0.15): Transmission rate per contact. Chosen as a moderate value for SIR-type models to reflect realistic disease spread.
        biweekly_beta_scalar (list of float, default=[1.0]*biweekly_steps): Scalar for beta for each biweekly period. Default is 1.0 for all periods, meaning no seasonal variation unless specified.
        cbr (float, default=0.03): Constant birth rate. Set to 0.03 to represent a typical annual birth rate in population models.
        exp_shape (float, default=2.0): Shape parameter for the exposed period distribution. Default chosen for moderate dispersion.
        exp_scale (float, default=2.0): Scale parameter for the exposed period distribution. Default chosen for moderate mean duration.
        inf_mean (float, default=4.0): Mean infectious period (days). Set to 4.0 to reflect typical infectious durations for diseases like measles.
        inf_sigma (float, default=1.0): Standard deviation of infectious period. Default is 1.0 for moderate variability.
        seasonality_factor (float, default=0.2): Amplitude of seasonal forcing. Chosen to allow moderate seasonal variation in transmission.
        seasonality_phase (float, default=0.0): Phase offset for seasonality. Default is 0.0, meaning no phase shift.
        importation_count (int, default=1): Number of cases imported per importation event. Default is 1 for sporadic importation.
        importation_period (int, default=30): Days between importation events. Default is 30 to represent monthly importation.
        importation_start (int, default=0): Start day for importation events. Default is 0 (simulation start).
        importation_end (int, default=730): End day for importation events. Default is 2 years (365*2).
        seed (int, default=123): Random seed for reproducibility. Default is 123.
        verbose (bool, default=False): If True, enables verbose output. Default is False for minimal output.
    These values are chosen to be broadly reasonable for seasonal SIR-type models with importation.

    We need a function like this because even-though laser-core requires no particular param name,
    laser-generic code does presume certain parameters and there's no elegant way to just discover
    what those are. So we put them here.
    """
    nticks = 365 * 2
    biweekly_steps = ceil(nticks / 14)
    return PropertySet(
        {
            "nticks": nticks,
            "beta": 0.15,
            "biweekly_beta_scalar": [1.0] * biweekly_steps,
            "cbr": 0.03,
            "exp_shape": 2.0,
            "exp_scale": 2.0,
            "inf_mean": 4.0,
            "inf_sigma": 1.0,
            "seasonality_factor": 0.2,
            "seasonality_phase": 0.0,
            "importation_count": 1,
            "importation_period": 30,
            "importation_start": 0,
            "importation_end": 365 * 2,
            "seed": 123,
            "verbose": False,
        }
    )
