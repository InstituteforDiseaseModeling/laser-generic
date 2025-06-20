"""
This module defines Immunization classes, which provide methods to import cases into a population during simulation.

Classes:
    RoutineImmunization: A class to periodically immunize a random subset of agents in the population
    CampaignImmunization: A class to periodically immunize a random subset of agents in the population

To do:
    Right now, neither intervention deploys to nodes, only globally.  Would like to add targeting by patch, 
    coverage by patch, and so on.
    RI coverage is constant over time, should have some ways to vary this.
    RI coverage basically looks like a campaign with the age window = target_age +/- period/2.
"""
import numpy as np
from matplotlib.figure import Figure


class RoutineImmunization:
    """
    A component to update the immunity status of a population in a model via routine immunization.
    """

    def __init__(self, model, period, coverage, age, start=0, end=-1, verbose: bool = False) -> None:
        """
        Initialize a RoutineImmunization instance.

        Args:
            model: The model object that contains the population.
            period: How frequently to survey the population and immunize children around the target age.
            coverage: The proportion of the population to immunize at each event.
            age: The target age for immunization.
            intervention_start (int, optional): The tick at which to start the immunization events.
            intervention_end (int, optional): The tick at which to end the immunization events.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Attributes:
            model: The model object that contains the population.

        Side Effects:
            None
        """
        self.model = model
        self.period = period
        self.coverage = coverage
        self.age = age
        self.start = start
        self.end = end if end != -1 else model.params.nticks
        self.verbose = verbose

        return

    def __call__(self, model, tick) -> None:
        """
        Updates the immunity status for the population in the model.

        Args:
            model: The model containing the population data.
            tick: The current tick or time step in the simulation.

        Returns:
            None
        """
        if (tick >= self.start) and ((tick - self.start) % self.period == 0) and (tick < self.end):
            # Immunize random agents
            half_window = int(self.period // 2)
            lower = int(self.age - half_window)
            upper = int(self.age + half_window)
            immunize_nodeids = immunize_in_age_window(model, lower, upper, self.coverage, tick)
            if (hasattr(model.patches, 'recovered_test')) and (immunize_nodeids is not None) and (len(immunize_nodeids) > 0):
                np.add.at(model.patches.recovered_test, (tick+1, immunize_nodeids), 1)
                np.add.at(model.patches.susceptibility_test, (tick+1, immunize_nodeids), -1)
        return




    def plot(self, fig: Figure = None):
        """
        Nothing yet
        """
        return

def immunize_in_age_window(model, lower, upper, coverage, tick):
    pop = model.population

    # Find agents whose age is within a window centered on self.age and width = self.period/2
    ages = tick - pop.dob
    in_window = (ages >= lower) & (ages < upper)
    #For now, immunization doesn't impact exposed or infected agents
    myinds = np.flatnonzero(pop.susceptibility & in_window)
    if len(myinds) == 0:
        return None

    n_immunize = np.random.binomial(len(myinds), coverage)
    if n_immunize > 0:
        np.random.shuffle(myinds)
        myinds = myinds[:n_immunize]
        pop.susceptibility[myinds] = 0
        inf_nodeids = pop.nodeid[myinds]
        return inf_nodeids

    else:
        return None

class ImmunizationCampaign:
    """
    A component to update the immunity status of a population in a model via routine immunization.
    """
    def __init__(self, model, period, coverage, age_lower, age_upper, start=0, end=-1, verbose: bool = False) -> None:
        """
        Initialize an ImmunizationCampaign instance.

        Args:
            model: The model object that contains the population.
            period: How frequently to survey the population and immunize children around the target age.
            coverage: The proportion of the population to immunize at each event.
            age: The target age for immunization.
            intervention_start (int, optional): The tick at which to start the immunization events.
            intervention_end (int, optional): The tick at which to end the immunization events.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Attributes:
            model: The model object that contains the population.

        Side Effects:
            None
        """
        self.model = model
        self.period = period
        self.coverage = coverage
        self.age_lower = age_lower
        self.age_upper = age_upper
        self.start = start
        self.end = end if end != -1 else model.params.nticks
        self.verbose = verbose

        return

    def __call__(self, model, tick) -> None:
        """
        Updates the immunity status for the population in the model.

        Args:
            model: The model containing the population data.
            tick: The current tick or time step in the simulation.

        Returns:
            None
        """
        if (tick >= self.start) and ((tick - self.start) % self.period == 0) and (tick < self.end):
            # Immunize random agents
            immunize_nodeids = immunize_in_age_window(model, self.age_lower, self.age_upper, self.coverage, tick)
            if (hasattr(model.patches, 'recovered_test')) and (immunize_nodeids is not None) and (len(immunize_nodeids) > 0):
                np.add.at(model.patches.recovered_test, (tick+1, immunize_nodeids), 1)
                np.add.at(model.patches.susceptibility_test, (tick+1, immunize_nodeids), -1)
        return

    def plot(self, fig: Figure = None):
        """
        Nothing yet
        """
        return
