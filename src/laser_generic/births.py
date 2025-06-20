"""
This module defines the Births component, which is responsible for simulating births in a population model.

Classes:

    Births:

        Manages the birth process within a population model, including initializing births, updating population data, and plotting birth statistics.

Usage:

    The Births component requires a model with a `population` attribute that has a `dob` attribute.
    It calculates the number of births based on the model's parameters and updates the population
    accordingly. It also provides methods to plot birth statistics.

Example:

    model = YourModelClass()
    births = Births(model)
    births(model, tick)
    births.plot()

Attributes:

    model (object): The population model.
    _initializers (list): List of initializers to be called on birth.
    _metrics (list): List to store timing metrics for initializers.
"""

from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


class Births:
    """
    A component to handle the birth events in a model.

    Attributes:

        model: The model instance containing population and parameters.
        verbose (bool): Flag to enable verbose output. Default is False.
        initializers (list): List of initializers to be called on birth events.
        metrics (DataFrame): DataFrame to holding timing metrics for initializers.
    """

    def __init__(self, model, verbose: bool = False):
        """
        Initialize the Births component.

        Parameters:

            model (object): The model object which must have a `population` attribute.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Raises:

            AssertionError: If the model does not have a `population` attribute.
            AssertionError: If the model's population does not have a `dob` attribute.
        """

        assert getattr(model, "population", None) is not None, "Births requires the model to have a `population` attribute"

        self.model = model

        nyears = (model.params.nticks + 364) // 365
        model.patches.add_vector_property("births", length=nyears, dtype=np.int32)
        model.population.add_scalar_property("dob", dtype=np.int32)

        self._initializers = []
        self._metrics = []

        return

    @property
    def initializers(self):
        """
        Returns the initializers to call on new agent births.

        This method retrieves the initializers that are used to set up the
        initial state or configuration for agents at birth.

        Returns:

            list: A list of initializers - instances of objects with an `on_birth` method.
        """

        return self._initializers

    @property
    def metrics(self):
        """
        Returns the timing metrics for the births initializers.

        This method retrieves the timing metrics for the births initializers.

        Returns:

            DataFrame: A Pandas DataFrame of timing metrics for the births initializers.
        """

        return pd.DataFrame(self._metrics, columns=["tick"] + [type(initializer).__name__ for initializer in self._initializers])

    def __call__(self, model, tick) -> None:
        """
        Adds new agents to each patch based on expected daily births calculated from CBR. Calls each of the registered initializers for the newborns.

        Args:

            model: The simulation model containing patches, population, and parameters.
            tick: The current time step in the simulation.

        Returns:

            None

        This method performs the following steps:

            1. Calculates the day of the year (doy) and the current year based on the tick.
            2. On the first day of the year, it generates annual births for each patch using a Poisson distribution.
            3. Calculates the number of births for the current day.
            4. Adds the newborns to the population and sets their date of birth.
            5. Assigns node IDs to the newborns.
            6. Calls any additional initializers for the newborns and records the timing of these initializations.
            7. Updates the population counts for the next tick with the new births.
        """
        # KM: I like this setup for now; I think there are ways we could improve it but not a priority for now.
        # Potential improvements - if population is growing/shrinking, there should be more/fewer births later in the year
        # If we are doing annually, could generate a 1-year random series of births all at once, rather than a number for the year and then interpolate every day
        # Could consider increments other than 1 year.
        doy = tick % 365 + 1  # day of year 1…365
        year = tick // 365

        if doy == 1:
            model.patches.births[year, :] = model.prng.poisson(model.patches.populations[tick, :] * model.params.cbr / 1000)

        annual_births = model.patches.births[year, :]
        todays_births = (annual_births * doy // 365) - (
            annual_births * (doy - 1) // 365
        )  # Is this not always basically annual_births / 365?
        count_births = todays_births.sum()
        istart, iend = model.population.add(count_births)

        if hasattr(model.population, "dob"):
            model.population.dob[istart:iend] = tick  # set to current tick

        # set the nodeids for the newborns in case subsequent initializers need them (properties varying by patch)
        index = istart
        nodeids = model.population.nodeid
        for nodeid, births in enumerate(todays_births):
            nodeids[index : index + births] = nodeid
            index += births

        timing = [tick]
        for initializer in self._initializers:
            tstart = datetime.now(tz=None)  # noqa: DTZ005
            initializer.on_birth(model, tick, istart, iend)
            tfinish = datetime.now(tz=None)  # noqa: DTZ005
            delta = tfinish - tstart
            timing.append(int(delta.total_seconds() * 1_000_000))
        self._metrics.append(timing)

        model.patches.populations[tick + 1, :] += todays_births

        return

    def plot(self, fig: Figure = None):
        """
        Plots the births in the top 5 most populous patches and a pie chart of birth initializer times.

        Parameters:

            fig (Figure, optional): A matplotlib Figure object. If None, a new figure will be created. Defaults to None.

        Yields:

            None: This function yields twice to allow for intermediate plotting steps.
        """

        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        _fig.suptitle("Births in Top 5 Most Populous Patches")

        indices = self.model.patches.populations[0, :].argsort()[-5:]
        ax1 = plt.gca()
        ticks = list(range(0, self.model.params.nticks, 365))
        for index in indices:
            ax1.plot(self.model.patches.populations[ticks, index], marker="x", markersize=4)

        ax2 = ax1.twinx()
        for index in indices:
            ax2.plot(self.model.patches.births[:, index], marker="+", markersize=4)

        yield

        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        _fig.suptitle("Births in Top 5 Most Populous Patches")

        metrics = pd.DataFrame(self._metrics, columns=["tick"] + [type(initializer).__name__ for initializer in self._initializers])
        plot_columns = metrics.columns[1:]
        sum_columns = metrics[plot_columns].sum()

        plt.pie(sum_columns, labels=sum_columns.index, autopct="%1.1f%%", startangle=140)
        plt.title("On Birth Initializer Times")

        yield

        return


class Births_ConstantPop:
    """
    A component to handle the birth events in a model with constant population - that is, births == deaths.

    Attributes:

        model: The model instance containing population and parameters.
        verbose (bool): Flag to enable verbose output. Default is False.
        initializers (list): List of initializers to be called on birth events.
        metrics (DataFrame): DataFrame to holding timing metrics for initializers.
    """

    def __init__(self, model, verbose: bool = False):
        """
        Initialize the Births component.

        Parameters:

            model (object): The model object which must have a `population` attribute.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Raises:

            AssertionError: If the model does not have a `population` attribute.
            AssertionError: If the model's population does not have a `dob` attribute.
        """

        assert getattr(model, "population", None) is not None, "Births requires the model to have a `population` attribute"

        self.model = model

        model.population.add_scalar_property("dob", dtype=np.int32)
        # Simple initializer for ages where birth rate = mortality rate:
        daily_mortality_rate = (1 + model.params.cbr / 1000) ** (1 / 365) - 1
        model.population.dob[0 : model.population.count] = -1 * model.prng.exponential(
            1 / daily_mortality_rate, model.population.count
        ).astype(np.int32)

        # nyears = (model.params.nticks + 364) // 365
        model.patches.add_vector_property("births", length=model.params.nticks, dtype=np.uint32)
        mu = (1 + model.params.cbr / 1000) ** (1 / 365) - 1
        model.patches.births = model.prng.poisson(lam=model.patches.populations[0, :] * mu, size=model.patches.births.shape)
        # model.patches.births[year, :] = model.prng.poisson(model.patches.populations[tick, :] * model.params.cbr / 1000)
        self._initializers = []
        self._metrics = []

        return

    @property
    def initializers(self):
        """
        Returns the initializers to call on new agent births.

        This method retrieves the initializers that are used to set up the
        initial state or configuration for agents at birth.

        Returns:

            list: A list of initializers - instances of objects with an `on_birth` method.
        """

        return self._initializers

    @property
    def metrics(self):
        """
        Returns the timing metrics for the births initializers.

        This method retrieves the timing metrics for the births initializers.

        Returns:

            DataFrame: A Pandas DataFrame of timing metrics for the births initializers.
        """

        return pd.DataFrame(self._metrics, columns=["tick"] + [type(initializer).__name__ for initializer in self._initializers])

    def __call__(self, model, tick) -> None:
        """
        Adds new agents to each patch based on expected daily births calculated from CBR. Calls each of the registered initializers for the newborns.

        Args:

            model: The simulation model containing patches, population, and parameters.
            tick: The current time step in the simulation.

        Returns:

            None

        This method performs the following steps:

            1. Draw a random set of indices, or size size "number of births"  from the population,
        """

        # When we get to having birth rate per node, will need to be more clever here, but with constant birth rate across nodes,
        # random selection will be population proportional.  If node id is not contiguous, could be tricky?
        indices = model.prng.choice(model.patches.populations[tick, :].sum(), size=model.patches.births[tick, :].sum(), replace=False)

        if hasattr(model.population, "dob"):
            model.population.dob[indices] = tick  # set to current tick
        model.population.state[indices] = 0


        model.patches.exposed_test[tick+1, :] -= np.bincount(
            model.population.nodeid[indices],
            weights=(model.population.etimer[indices] > 0),
            minlength=model.patches.populations.shape[1]
        ).astype(np.uint32)

        model.patches.cases_test[tick+1, :] -= np.bincount(
            model.population.nodeid[indices],
            weights=(model.population.itimer[indices] > 0),
            minlength=model.patches.populations.shape[1]
        ).astype(np.uint32)

        model.patches.recovered_test[tick+1, :] -= np.bincount(
            model.population.nodeid[indices],
            weights=((model.population.etimer[indices]==0 ) &
                     (model.population.itimer[indices]==0) &
                     (model.population.susceptibility[indices] == 0)),
            minlength=model.patches.populations.shape[1]
        ).astype(np.uint32)

        model.patches.susceptibility_test[tick+1, :] += np.bincount(
            model.population.nodeid[indices],
            weights=(model.population.susceptibility[indices] == 0),
            minlength=model.patches.populations.shape[1]
        ).astype(np.uint32)

        timing = [tick]
        for initializer in self._initializers:
            tstart = datetime.now(tz=None)  # noqa: DTZ005
            initializer.on_birth(model, tick, indices, None)
            tfinish = datetime.now(tz=None)  # noqa: DTZ005
            delta = tfinish - tstart
            timing.append(int(delta.total_seconds() * 1_000_000))
        self._metrics.append(timing)

        return

    def plot(self, fig: Figure = None):
        """
        Plots the births in the top 5 most populous patches and a pie chart of birth initializer times.

        Parameters:

            fig (Figure, optional): A matplotlib Figure object. If None, a new figure will be created. Defaults to None.

        Yields:

            None: This function yields twice to allow for intermediate plotting steps.
        """

        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        _fig.suptitle("Births in Top 5 Most Populous Patches")

        indices = self.model.patches.populations[0, :].argsort()[-5:]
        ax1 = plt.gca()
        ticks = list(range(0, self.model.params.nticks, 365))
        for index in indices:
            ax1.plot(self.model.patches.populations[ticks, index], marker="x", markersize=4)

        ax2 = ax1.twinx()
        for index in indices:
            ax2.plot(self.model.patches.births[:, index], marker="+", markersize=4)

        yield

        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig
        _fig.suptitle("Births in Top 5 Most Populous Patches")

        metrics = pd.DataFrame(self._metrics, columns=["tick"] + [type(initializer).__name__ for initializer in self._initializers])
        plot_columns = metrics.columns[1:]
        sum_columns = metrics[plot_columns].sum()

        plt.pie(sum_columns, labels=sum_columns.index, autopct="%1.1f%%", startangle=140)
        plt.title("On Birth Initializer Times")

        yield

        return


class Births_ConstantPop_VariableBirthRate(Births_ConstantPop):
    """
    A component to handle birth events in a model with constant population but variable birth rates over time.  .

    This class extends Births_ConstantPop to allow for a variable birth rate over time (cbr).
    """

    def __init__(self, model, verbose: bool = False):
        """
        Initialize the Births_ConstantPop_VariableBirthRate component.

        Parameters:
            model (object): The model object which must have a `population` attribute.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Raises:
            AssertionError: If the model does not have a `population` attribute.
        """
        assert getattr(model, "population", None) is not None, "Births requires the model to have a `population` attribute"

        self.model = model

        model.population.add_scalar_property("dob", dtype=np.int32)

        # Expect model.params.cbr to be a dict with keys "rates" and "timesteps"
        cbr_param = model.params.cbr
        if not (isinstance(cbr_param, dict) and "rates" in cbr_param and "timesteps" in cbr_param):
            raise ValueError("model.params.cbr must be a dict with keys 'rates' and 'timesteps'")

        rates = np.asarray(cbr_param["rates"], dtype=float)
        timesteps = np.asarray(cbr_param["timesteps"], dtype=int)

        if len(rates) != len(timesteps):
            raise ValueError("'rates' and 'timesteps' must have the same length")

        nticks = model.params.nticks
        cbr = np.empty(nticks, dtype=float)

        # Handle before first timestep
        if timesteps[0] > 0:
            cbr[:timesteps[0]] = rates[0]

        # Interpolate between timesteps
        for i in range(len(timesteps) - 1):
            start, end = timesteps[i], timesteps[i + 1]
            cbr[start:end] = np.linspace(rates[i], rates[i + 1], end - start, endpoint=False)

        # Handle after last timestep
        if timesteps[-1] < nticks:
            cbr[timesteps[-1]:] = rates[-1]


        # Calculate daily mortality rate for each tick
        daily_mortality_rate = (1 + cbr / 1000) ** (1 / 365) - 1

        # Assign random ages to initial population (using first day's rate)
        mu0 = daily_mortality_rate[0]
        model.population.dob[0 : model.population.count] = -1 * model.prng.exponential(
            1 / mu0, model.population.count
        ).astype(np.int32)

        model.patches.add_vector_property("births", length=model.params.nticks, dtype=np.uint32)
        # For each tick, calculate births for each patch using the time-varying cbr
        # Vectorized sampling for all ticks and patches at once
        mu_t = daily_mortality_rate[:, np.newaxis]  # shape (nticks, 1)
        lam = model.patches.populations[0, :][np.newaxis, :] * mu_t  # shape (nticks, n_patches)
        model.patches.births[:, :] = model.prng.poisson(lam=lam)

        self._initializers = []
        self._metrics = []

        return
