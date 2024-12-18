"""
This module defines Importation classes, which provide methods to import cases into a population during simulation.

Classes:
    Infect_Random_Agents: A class to periodically infect a random subset of agents in the population

Functions:
    Infect_Random_Agents.__init__(self, model, period, count, verbose: bool = False) -> None:
        Initializes the Infect_Random_Agents class with a given model, period, count, and verbosity option.

    Infect_Random_Agents.__call__(self, model, tick) -> None:
        Checks whether it is time to infect a random subset of agents and infects them if necessary.

    Infect_Random_Agents.plot(self, fig: Figure = None):
        Nothing yet.
"""

import numba as nb
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from laser_generic.utils import seed_infections_randomly


class Infect_Random_Agents:
    """
    A component to update the infection timers of a population in a model.
    """

    def __init__(self, model, verbose: bool = False) -> None:
        """
        Initialize an Infect_Random_Agents instance.

        Args:

            model: The model object that contains the population.
            period: The number of ticks between each infection event.
            count: The number of agents to infect at each event.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Attributes:

            model: The model object that contains the population.

        Side Effects:

        """

        self.model = model
        self.period = model.params.importation_period
        self.count = model.params.importation_count

        return

    def __call__(self, model, tick) -> None:
        """
        Updates the infection timers for the population in the model.

        Args:

            model: The model containing the population data.
            tick: The current tick or time step in the simulation.

        Returns:

            None
        """
        if tick % self.period == 0:
            seed_infections_randomly(model, self.count)

        return


    def plot(self, fig: Figure = None):
        """
        Nothing yet
        """
        return

