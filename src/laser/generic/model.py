"""
This module defines a `Model` class for simulating classic "generic" compartmental
disease models (SI, SIS, SIR, SEIR, ...). The model supports simple demographics
(e.g., births, deaths, aging) and can simulate either a single population patch
or multiple patches with an arbitrary connection structure.

**Imports:**
- datetime: For handling date and time operations.
- click: For command-line interface utilities.
- numpy as np: For numerical operations.
- pandas as pd: For data handling and tabular reports.
- laser_core.laserframe: Provides the LaserFrame class for structured data arrays.
- laser_core.propertyset: Provides the PropertySet class for simulation parameters.
- laser_core.random: Provides random number generator seeding utilities.
- matplotlib: For plotting results.
- tqdm: For progress bar visualization during runs.
"""

from datetime import datetime

import click
import numpy as np
import pandas as pd
from laser.core.laserframe import LaserFrame
from laser.core.propertyset import PropertySet
from laser.core.random import seed as seed_prng
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from tqdm import tqdm

from laser.generic import Births
from laser.generic import Births_ConstantPop


class Model:
    """
    A LASER simulation model for generic disease dynamics.

    The `Model` manages:
      - Patch-level populations and their attributes.
      - Agent-level population initialization.
      - Integration of components (e.g., Births, Infection, Immunization).
      - Execution of simulation ticks via `run()`.
      - Recording of metrics and plotting/visualization utilities.

    Typical usage:
    ```python
    scenario = pd.DataFrame({"population": [1000, 500], "latitude": [...], "longitude": [...]})
    params = PropertySet({"nticks": 100, "seed": 123, "verbose": True})
    model = Model(scenario, params)
    model.components = [Births, Infection, ImmunizationCampaign]
    model.run()
    model.visualize(pdf=True)
    ```
    """

    def __init__(self, scenario: pd.DataFrame, parameters: PropertySet, name: str = "generic") -> None:
        """
        Initialize the model with a scenario and simulation parameters.

        Parameters
        ----------
        scenario : pd.DataFrame
            Patch-level data. Must include at least:
            - `population`: initial population per patch.
            - `latitude`: latitude coordinate.
            - `longitude`: longitude coordinate.
            May also include optional columns like `geometry`.
        parameters : PropertySet
            Simulation parameters. Must include:
            - `nticks` (int): number of simulation ticks.
            - `seed` (int, optional): RNG seed.
            - `verbose` (bool, optional): enable verbose logging.
        name : str, optional
            Name of the model. Default is "generic".

        Side Effects
        ------------
        - Seeds the random number generator.
        - Initializes patches and population.
        """
        self.tinit = datetime.now(tz=None)  # noqa: DTZ005
        click.echo(f"{self.tinit}: Creating the {name} model…")
        self.scenario = scenario
        self.params = parameters
        self.name = name

        self.prng = seed_prng(parameters.seed if parameters.seed is not None else self.tinit.microsecond)

        click.echo(f"Initializing the {name} model with {len(scenario)} patches…")

        self._initialize_patches(scenario, parameters)
        self._initialize_population(scenario, parameters)
        # self.initialize_network(scenario, parameters)

        return

    def _initialize_patches(self, scenario: pd.DataFrame, parameters: PropertySet) -> None:
        # We need some patches with population data ...
        npatches = len(scenario)
        self.patches = LaserFrame(npatches, initial_count=0)

        # "activate" all the patches (count == capacity)
        self.patches.add(npatches)
        self.patches.add_vector_property("populations", length=parameters.nticks + 1)
        self.patches.add_vector_property("cases_test", length=parameters.nticks + 1, dtype=np.uint32)
        self.patches.add_vector_property("exposed_test", length=parameters.nticks + 1, dtype=np.uint32)
        self.patches.add_vector_property("recovered_test", length=parameters.nticks + 1, dtype=np.uint32)
        self.patches.add_vector_property("susceptibility_test", length=parameters.nticks + 1, dtype=np.uint32)
        # set patch populations at t = 0 to initial populations
        self.patches.populations[0, :] = scenario.population

        return

    def _initialize_population(self, scenario: pd.DataFrame, parameters: PropertySet) -> None:
        # Initialize the model population
        # Is there a better pattern than checking for cbr in parameters?  Many modelers might use "mu", for example.
        # Would rather check E.g., if there is a birth component, but that might come later.
        # if "cbr" in parameters:
        #    capacity = calc_capacity(self.patches.populations[0, :].sum(), parameters.nticks, parameters.cbr, parameters.verbose)
        # else:
        capacity = np.sum(self.patches.populations[0, :])
        self.population = LaserFrame(capacity=int(capacity), initial_count=0)

        self.population.add_scalar_property("nodeid", dtype=np.uint16)
        self.population.add_scalar_property("state", dtype=np.uint8, default=0)
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

    def _initialize_network(self, scenario: pd.DataFrame, parameters: PropertySet) -> None:
        raise RuntimeError("_initialize_network not yet implemented.")
        return

    @property
    def components(self) -> list:
        """
        Retrieve the list of model components currently configured.

        Returns
        -------
        list
            List of component classes used in the model.
        """
        return self._components

    @components.setter
    def components(self, components: list) -> None:
        """
        Configure the model components.

        For each provided component class:
          - Instantiate it with `(self, self.params.verbose)`.
          - Register it in `self.instances`.
          - If the instance is callable, add it to `self.phases`.
          - If it has a `census` method, add it to `self.censuses`.

        Special handling:
          - If a `Births` or `Births_ConstantPop` instance is present,
            any other component with an `on_birth` method is registered
            as a births initializer.

        Parameters
        ----------
        components : list
            List of component classes (not instances).
        """

        self._components = components
        self.instances = [self]  # instantiated instances of components
        self.phases = [self]  # callable phases of the model
        self.censuses = []  # callable censuses of the model - to be called at the beginning of a tick to record state
        for component in components:
            instance = component(self, self.params.verbose)
            self.instances.append(instance)
            if "__call__" in dir(instance):
                self.phases.append(instance)
            if "census" in dir(instance):
                self.censuses.append(instance)

        births = next(filter(lambda object: isinstance(object, (Births, Births_ConstantPop)), self.instances), None)
        # TODO: raise an exception if there are components with an on_birth function but no Births component
        for instance in self.instances:
            if births is not None and "on_birth" in dir(instance):
                births.initializers.append(instance)
        return

    def __call__(self, model, tick: int) -> None:
        """
        Advance patch populations one tick forward.

        Copies population counts from tick `t` to tick `t+1`.
        Components such as births or mortality may then update these values.

        Parameters
        ----------
        model : Model
            The current model instance.
        tick : int
            Current tick index.
        """

        model.patches.populations[tick + 1, :] = model.patches.populations[tick, :]
        return

    def run(self) -> None:
        """
        Execute the model simulation.

        For each tick (0..nticks-1):
          - Run all censuses (recording metrics).
          - Run all phases (update components).
          - Record execution times per census/phase.

        After completion:
          - Records `self.metrics`, `self.tstart`, and `self.tfinish`.
          - Prints a timing summary if verbose mode is enabled.

        Attributes Set
        --------------
        tstart : datetime
            Start time of execution.
        tfinish : datetime
            End time of execution.
        metrics : list
            List of timing metrics (per tick, per phase).
        """
        self.tstart = datetime.now(tz=None)  # noqa: DTZ005
        click.echo(f"{self.tstart}: Running the {self.name} model for {self.params.nticks} ticks…")

        self.metrics = []
        for tick in tqdm(range(self.params.nticks)):
            timing = [tick]
            for census in self.censuses:
                tstart = datetime.now(tz=None)  # noqa: DTZ005
                census.census(self, tick)
                tfinish = datetime.now(tz=None)  # noqa: DTZ005
                delta = tfinish - tstart
                timing.append(delta.seconds * 1_000_000 + delta.microseconds)
            self.metrics.append(timing)

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
            names = [type(census).__name__ + "_census" for census in self.censuses] + [type(phase).__name__ for phase in self.phases]
            metrics = pd.DataFrame(self.metrics, columns=["tick", *list(names)])
            plot_columns = metrics.columns[1:]
            sum_columns = metrics[plot_columns].sum()
            width = max(map(len, sum_columns.index))
            for key in sum_columns.index:
                print(f"{key:{width}}: {sum_columns[key]:13,} µs")
            print("=" * (width + 2 + 13 + 3))
            print(f"{'Total:':{width + 1}} {sum_columns.sum():13,} microseconds")

        return

    def visualize(self, pdf: bool = True) -> None:
        """
        Generate visualizations for all components.

        Parameters
        ----------
        pdf : bool, optional
            If True (default), save plots to a PDF file named
            "<model name> <timestamp>.pdf".
            If False, display plots interactively with `plt.show()`.

        Side Effects
        ------------
        - Saves a PDF file when `pdf=True`.
        - Calls `plt.show()` when `pdf=False`.
        """

        if not pdf:
            for instance in self.instances:
                try:
                    for _plot in instance.plot():
                        plt.show()
                except Exception as ex:
                    print(f"Exception iterating on plot function of instance: {ex}")

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
        Yield three plots for model visualization:

        1. A scatter plot of scenario patches by location and population.
        2. A histogram of day-of-birth values for the initial population.
           (requires that `population.dob` exists, e.g. via a Births component).
        3. A pie chart of cumulative update phase timings.

        Parameters
        ----------
        fig : Figure, optional
            An existing matplotlib Figure. If None, a new figure is created.

        Yields
        ------
        None
            Each `yield` produces one plot, so iterating this generator
            produces three figures sequentially.
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

        if hasattr(self.population, "dob"):
            _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
            _fig.suptitle("Distribution of Day of Birth for Initial Population")

            count = self.patches.populations[0, :].sum()  # just the initial population
            dobs = self.population.dob[0:count]
            plt.hist(dobs, bins=100)
            plt.xlabel("Day of Birth")

            yield

        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig

        # metrics = pd.DataFrame(self.metrics, columns=["tick"] + [type(phase).__name__ for phase in self.phases])

        # Build proper column names for both census and phase timings
        names = []
        for census in self.censuses:
            names.append(type(census).__name__ + "_census")
        for phase in self.phases:
            names.append(type(phase).__name__)

        metrics = pd.DataFrame(self.metrics, columns=["tick", *names])
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
