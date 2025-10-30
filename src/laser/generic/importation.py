"""
This module defines Importation classes, which provide methods to import cases into a population during simulation.

"""

from collections.abc import Generator
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from laser.generic.utils import seed_infections_in_patch
from laser.generic.utils import seed_infections_randomly


class Infect_Random_Agents:
    """
    A LASER model component that introduces random infections into the population
    at regular intervals. This is typically used to simulate importation events
    or background infection pressure.
    """

    def __init__(self, model, verbose: bool = False) -> None:
        """
        Initialize an Infect_Random_Agents instance.

        Args:
            model (object): The LASER model object that contains the population,
                patches, and parameters. The following attributes must exist
                in `model.params`:
                  - importation_period (int): Number of ticks between each infection event.
                  - importation_count (int): Number of agents to infect per event.
                  - nticks (int): Total number of ticks in the simulation.
                  - importation_start (int, optional): First tick to introduce infections.
                    Defaults to 0 if not provided.
                  - importation_end (int, optional): Last tick to introduce infections.
                    Defaults to nticks if not provided.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Attributes:
            model (object): The LASER model object used by the component.
            period (int): Number of ticks between infection events.
            count (int): Number of agents infected at each event.
            start (int): First tick to apply infections.
            end (int): Last tick to apply infections.
        """
        self.model = model
        self.period = model.params.importation_period
        self.count = model.params.importation_count
        self.start = 0
        self.end = model.params.nticks
        if hasattr(model.params, "importation_start"):
            self.start = model.params.importation_start
        if hasattr(model.params, "importation_end"):
            self.end = model.params.importation_end

        return

    def __call__(self, model, tick) -> None:
        """
        Introduce random infections into the population at the given tick.

        Infections are seeded if:
          - The current tick is greater than or equal to `start`.
          - The tick falls on a multiple of `period` (relative to `start`).
          - The tick is less than `end`.

        This updates both the agent-level infections and, if present,
        the test arrays in `model.patches` for validation.

        Args:
            model (object): The LASER model containing the population and patches.
            tick (int): The current tick (time step) of the simulation.

        Returns:
            None
        """
        # KM: something about this importation function is so slow I can see the timer slow down when running in a notebook.  Need to
        # figure this out.
        if (tick >= self.start) and ((tick - self.start) % self.period == 0) and (tick < self.end):
            inf_nodeids = seed_infections_randomly(model, self.count)
            if hasattr(model.patches, "cases_test"):
                # Use numpy for efficient batch updates
                np.add.at(model.patches.cases_test, (tick + 1, inf_nodeids), 1)
                np.add.at(model.patches.susceptibility_test, (tick + 1, inf_nodeids), -1)
                # unique, counts = np.unique(inf_nodeids, return_counts=True)
                # for nodeid, count in zip(unique, counts):
                #     model.patches.cases_test[tick+1, nodeid] += count
                #     model.patches.susceptibility_test[tick+1, nodeid] -= count

        return

    def plot(self, fig: Figure = None) -> Generator[Any, Any, Any]:
        """
        Placeholder for visualization of infection events.

        Args:
            fig (Figure, optional): A matplotlib Figure to plot into. If None,
                no plot is generated.

        Returns:
            None
        """
        yield
        return


class Infect_Agents_In_Patch:
    """
    A LASER model component that introduces infections into specific patches
    of the population at regular intervals. This is useful for modeling
    geographically targeted importations or outbreaks.
    """

    def __init__(self, model, verbose: bool = False) -> None:
        """
        Initialize an Infect_Agents_In_Patch instance.

        Args:
            model (object): The LASER model object that contains the population,
                patches, and parameters. The following attributes must exist
                or may optionally exist in `model.params`:
                  - importation_period (int): Number of ticks between infection events.
                  - importation_count (int, optional): Number of agents to infect
                    per patch per event. Defaults to 1 if not provided.
                  - importation_patchlist (array-like of int, optional): Indices
                    of patches where infections will be seeded. Defaults to all patches
                    if not provided.
                  - importation_start (int, optional): First tick to apply infections.
                    Defaults to 0 if not provided.
                  - importation_end (int, optional): Last tick to apply infections.
                    Defaults to `nticks` if not provided.
                  - nticks (int): Total number of ticks in the simulation.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Attributes:
            model (object): The LASER model used by this component.
            period (int): Number of ticks between infection events.
            count (int): Number of agents infected per patch at each event.
            patchlist (ndarray): List of patch indices to target with infections.
            start (int): First tick to apply infections.
            end (int): Last tick to apply infections.
        """
        self.model = model
        self.period = model.params.importation_period

        self.count = model.params.importation_count if hasattr(model.params, "importation_count") else 1
        self.patchlist = (
            model.params.importation_patchlist if hasattr(model.params, "importation_patchlist") else np.arange(model.patches.count)
        )
        self.start = model.params.importation_start if hasattr(model.params, "importation_start") else 0
        self.end = model.params.importation_end if hasattr(model.params, "importation_end") else model.params.nticks

        return

    def __call__(self, model, tick) -> None:
        """
        Introduce infections into the specified patches at the given tick.

        Infections are seeded if:
          - The current tick is greater than or equal to `start`.
          - The tick falls on a multiple of `period` (relative to `start`).
          - The tick is less than `end`.

        At each eligible tick, every patch in `patchlist` receives `count`
        infections via `seed_infections_in_patch`.

        Args:
            model (object): The LASER model containing the population and patches.
            tick (int): The current tick (time step) of the simulation.

        Returns:
            None
        """
        if (tick >= self.start) and ((tick - self.start) % self.period == 0) and (tick < self.end):
            for patch in self.patchlist:
                seed_infections_in_patch(model, patch, self.count)

        return

    def plot(self, fig: Figure = None) -> Generator[Any, Any, Any]:
        """
        Placeholder for visualization of targeted patch infections.

        Args:
            fig (Figure, optional): A matplotlib Figure to plot into. If None,
                no plot is generated.

        Returns:
            None
        """
        yield
        return
