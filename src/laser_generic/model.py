"""
This module defines a `Model` class for simulating classic "generic" disease models (SI, SIS, SIR, SEIR, ...),
with options for simple demographics (births, deaths, aging) and single or multiple patches with flexible connection structure.

Classes:
    Model: A general class from which to define specific types of simulation models.

Imports:
    - datetime: For handling date and time operations.
    - click: For command-line interface utilities.
    - numpy as np: For numerical operations.
    - pandas as pd: For data manipulation and analysis.
    - laser_core.demographics: For demographic data handling.
    - laser_core.laserframe: For handling laser frame data structures.
    - laser_core.migration: For migration modeling.
    - laser_core.propertyset: For handling property sets.
    - laser_core.random: For random number generation.
    - matplotlib.pyplot as plt: For plotting.
    - matplotlib.backends.backend_pdf: For PDF generation.
    - matplotlib.figure: For figure handling.
    - tqdm: For progress bar visualization.

Model Class:
    Methods:
        __init__(self, scenario: pd.DataFrame, parameters: PropertySet, name: str) -> None:
            Initializes the model with the given scenario and parameters.

        components(self) -> list:
            Gets the list of components in the model.

        components(self, components: list) -> None:
            Sets the list of components in the model and initializes instances and phases.

        __call__(self, model, tick: int) -> None:
            Updates the model for a given tick.

        run(self) -> None:
            Runs the model for the specified number of ticks.

        visualize(self, pdf: bool = True) -> None:
            Generates visualizations of the model's results, either displaying them or saving to a PDF.

        plot(self, fig: Figure = None):
            Generates plots for the scenario patches and populations, distribution of day of birth, and update phase times.
"""

from datetime import datetime

import click
import numpy as np
import pandas as pd
from laser_core.laserframe import LaserFrame
from laser_core.propertyset import PropertySet
from laser_core.random import seed as seed_prng
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from tqdm import tqdm

from laser_generic import Births
from laser_generic import Births_ConstantPop
from laser_generic.utils import calc_capacity


class Model:
    """
    A class to represent a simulation model.
    """

    def __init__(self, scenario: pd.DataFrame, parameters: PropertySet, name: str = "generic") -> None:
        """
        Initialize the disease model with the given scenario and parameters.

        Args:

            scenario (pd.DataFrame): A DataFrame containing the metapopulation patch data, including population, latitude, and longitude.
            parameters (PropertySet): A set of parameters for the model and simulations.
            name (str, optional): The name of the model. Defaults to "generic".

        Returns:

            None
        """

        self.tinit = datetime.now(tz=None)  # noqa: DTZ005
        click.echo(f"{self.tinit}: Creating the {name} model…")
        self.scenario = scenario
        self.params = parameters
        self.name = name

        self.prng = seed_prng(parameters.seed if parameters.seed is not None else self.tinit.microsecond)

        click.echo(f"Initializing the {name} model with {len(scenario)} patches…")

        self.initialize_patches(scenario, parameters)
        self.initialize_population(scenario, parameters)
        # self.initialize_network(scenario, parameters)

        return

    def initialize_patches(self, scenario: pd.DataFrame, parameters: PropertySet) -> None:
        # We need some patches with population data ...
        npatches = len(scenario)
        self.patches = LaserFrame(npatches)

        # "activate" all the patches (count == capacity)
        self.patches.add(npatches)
        self.patches.add_vector_property("populations", length=parameters.nticks+1)
        # set patch populations at t = 0 to initial populations
        self.patches.populations[0, :] = scenario.population

        return

    def initialize_population(self, scenario: pd.DataFrame, parameters: PropertySet) -> None:
        # Initialize the model population
        # Is there a better pattern than checking for cbr in parameters?  Many modelers might use "mu", for example.
        # Would rather check E.g., if there is a birth component, but that might come later.
        #if "cbr" in parameters:
        #    capacity = calc_capacity(self.patches.populations[0, :].sum(), parameters.nticks, parameters.cbr, parameters.verbose)
        #else:
        capacity = np.sum(self.patches.populations[0, :])
        self.population = LaserFrame(capacity)

        self.population.add_scalar_property("nodeid", dtype=np.uint16)
        for nodeid, count in enumerate(self.patches.populations[0, :]):
            first, last = self.population.add(count)
            self.population.nodeid[first:last] = nodeid

        # Initialize population ages
        # With the simple demographics I'm using, I won't always need ages, and when I do they will just be exponentially distributed.
        # Note - should we separate population initialization routines from initialization of the model class?

        # pyramid_file = parameters.pyramid_file
        # age_distribution = load_pyramid_csv(pyramid_file)
        # both = age_distribution[:, 2] + age_distribution[:, 3]  # males + females
        # sampler = AliasedDistribution(both)
        # bin_min_age_days = age_distribution[:, 0] * 365  # minimum age for bin, in days (include this value)
        # bin_max_age_days = (age_distribution[:, 1] + 1) * 365  # maximum age for bin, in days (exclude this value)
        # initial_pop = self.population.count
        # samples = sampler.sample(initial_pop)  # sample for bins from pyramid
        # self.population.add_scalar_property("dob", dtype=np.int32)
        # mask = np.zeros(initial_pop, dtype=bool)
        # dobs = self.population.dob[0:initial_pop]
        # click.echo("Assigning day of year of birth to agents…")
        # for i in tqdm(range(len(age_distribution))):  # for each possible bin value...
        #     mask[:] = samples == i  # ...find the agents that belong to this bin
        #     # ...and assign a random age, in days, within the bin
        #     dobs[mask] = self.prng.integers(bin_min_age_days[i], bin_max_age_days[i], mask.sum())

        # dobs *= -1  # convert ages to date of birth prior to _now_ (t = 0) ∴ negative

        return

    def initialize_network(self, scenario: pd.DataFrame, parameters: PropertySet) -> None:
        # Come back to network setup.  Shouldn't need for N=1 networks, and shouldn't default to gravity
        # distances = calc_distances(scenario.latitude.values, scenario.longitude.values, parameters.verbose)
        # network = gravity(
        #     scenario.population.values,
        #     distances,
        #     parameters.k,
        #     parameters.a,
        #     parameters.b,
        #     parameters.c,
        # )
        # network = row_normalizer(network, parameters.max_frac)
        # self.patches.add_vector_property("network", length=npatches, dtype=np.float32)
        # self.patches.network[:, :] = network
        return

    @property
    def components(self) -> list:
        """
        Retrieve the list of model components.

        Returns:

            list: A list containing the components.
        """

        return self._components

    @components.setter
    def components(self, components: list) -> None:
        """
        Sets up the components of the model and initializes instances and phases.

        This function takes a list of component types, creates an instance of each, and adds each callable component to the phase list.
        It also registers any components with an `on_birth` function with the `Births` component.

        Args:

            components (list): A list of component classes to be initialized and integrated into the model.

        Returns:

            None
        """

        self._components = components
        self.instances = [self]  # instantiated instances of components
        self.phases = [self]  # callable phases of the model
        for component in components:
            instance = component(self, self.params.verbose)
            self.instances.append(instance)
            if "__call__" in dir(instance):
                self.phases.append(instance)

        births = next(filter(lambda object: isinstance(object, (Births, Births_ConstantPop)), self.instances), None)
        # TODO: raise an exception if there are components with an on_birth function but no Births component
        for instance in self.instances:
            if births is not None and "on_birth" in dir(instance):
                births.initializers.append(instance)
        return

    def __call__(self, model, tick: int) -> None:
        """
        Updates the population of patches for the next tick. Copies the previous
        population data to the next tick to be updated, optionally, by a Birth and/or
        Mortality component.

        Args:

            model: The model containing the patches and their populations.
            tick (int): The current time step or tick.

        Returns:

            None
        """

        model.patches.populations[tick + 1, :] = model.patches.populations[tick, :]
        return

    def run(self) -> None:
        """
        Execute the model for a specified number of ticks, recording the time taken for each phase.

        This method initializes the start time, iterates over the number of ticks specified in the model parameters,
        and for each tick, it executes each phase of the model while recording the time taken for each phase.

        The metrics for each tick are stored in a list. After completing all ticks, it records the finish time and,
        if verbose mode is enabled, prints a summary of the timing metrics.

        Attributes:

            tstart (datetime): The start time of the model execution.
            tfinish (datetime): The finish time of the model execution.
            metrics (list): A list of timing metrics for each tick and phase.

        Returns:

            None
        """

        self.tstart = datetime.now(tz=None)  # noqa: DTZ005
        click.echo(f"{self.tstart}: Running the {self.name} model for {self.params.nticks} ticks…")

        self.metrics = []
        for tick in tqdm(range(self.params.nticks)):
            timing = [tick]
            for phase in self.phases:
                tstart = datetime.now(tz=None)  # noqa: DTZ005
                phase(self, tick)
                tfinish = datetime.now(tz=None)  # noqa: DTZ005
                delta = tfinish - tstart
                timing.append(delta.seconds * 1_000_000 + delta.microseconds)
            self.metrics.append(timing)

        self.tfinish = datetime.now(tz=None)  # noqa: DTZ005
        print(f"Completed the {self.name} model at {self.tfinish}…")

        if self.params.verbose:
            metrics = pd.DataFrame(self.metrics, columns=["tick"] + [type(phase).__name__ for phase in self.phases])
            plot_columns = metrics.columns[1:]
            sum_columns = metrics[plot_columns].sum()
            width = max(map(len, sum_columns.index))
            for key in sum_columns.index:
                print(f"{key:{width}}: {sum_columns[key]:13,} µs")
            print("=" * (width + 2 + 13 + 3))
            print(f"{'Total:':{width+1}} {sum_columns.sum():13,} microseconds")

        return

    def visualize(self, pdf: bool = True) -> None:
        """
        Visualize each compoonent instances either by displaying plots or saving them to a PDF file.

        Parameters:

            pdf (bool): If True, save the plots to a PDF file. If False, display the plots interactively. Default is True.

        Returns:

            None
        """

        if not pdf:
            for instance in self.instances:
                for _plot in instance.plot():
                    plt.show()

        else:
            click.echo("Generating PDF output…")
            pdf_filename = f"{self.name} {self.tstart:%Y-%m-%d %H%M%S}.pdf"
            with PdfPages(pdf_filename) as pdf:
                for instance in self.instances:
                    for _plot in instance.plot():
                        pdf.savefig()
                        plt.close()

            click.echo(f"PDF output saved to '{pdf_filename}'.")

        return

    def plot(self, fig: Figure = None):
        """
        Plots various visualizations related to the scenario and population data.

        Parameters:

            fig (Figure, optional): A matplotlib Figure object to use for plotting. If None, a new figure will be created.

        Yields:

            None: This function uses a generator to yield control back to the caller after each plot is created.

        The function generates three plots:

            1. A scatter plot of the scenario patches and populations.
            2. A histogram of the distribution of the day of birth for the initial population.
            3. A pie chart showing the distribution of update phase times.
        """

        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        _fig.suptitle("Scenario Patches and Populations")
        if "geometry" in self.scenario.columns:
            ax = plt.gca()
            self.scenario.plot(ax=ax)
        scatter = plt.scatter(
            self.scenario.longitude,
            self.scenario.latitude,
            s=self.scenario.population / 1000,
            c=self.scenario.population,
            cmap="inferno",
        )
        plt.colorbar(scatter, label="Population")

        yield

        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        _fig.suptitle("Distribution of Day of Birth for Initial Population")

        count = self.patches.populations[0, :].sum()  # just the initial population
        dobs = self.population.dob[0:count]
        plt.hist(dobs, bins=100)
        plt.xlabel("Day of Birth")

        yield

        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig

        metrics = pd.DataFrame(self.metrics, columns=["tick"] + [type(phase).__name__ for phase in self.phases])
        plot_columns = metrics.columns[1:]
        sum_columns = metrics[plot_columns].sum()

        plt.pie(
            sum_columns,
            labels=[name if not name.startswith("do_") else name[3:] for name in sum_columns.index],
            autopct="%1.1f%%",
            startangle=140,
        )
        plt.title("Update Phase Times")

        yield
        return
