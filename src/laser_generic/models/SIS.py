"""Components for the SIS model."""

from enum import Enum

import contextily as ctx
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from laser_core import LaserFrame
from laser_core.migration import distance
from laser_core.migration import gravity
from laser_core.migration import row_normalizer
from tqdm import tqdm

from laser_generic.newutils import RateMap
from laser_generic.newutils import TimingStats as ts
from laser_generic.newutils import validate


class State(Enum):
    SUSCEPTIBLE = 0
    INFECTIOUS = 1
    DECEASED = -1

    def __new__(cls, value):
        obj = object.__new__(cls)
        obj._value_ = np.int8(value)
        return obj


class Susceptible:
    def __init__(self, model):
        self.model = model
        self.model.people.add_scalar_property("nodeid", dtype=np.uint16)
        self.model.people.add_scalar_property("state", dtype=np.int8, default=State.SUSCEPTIBLE.value)
        self.model.nodes.add_vector_property("S", model.params.nticks + 1, dtype=np.int32)

        self.model.people.nodeid[:] = np.repeat(np.arange(self.model.nodes.count, dtype=np.uint16), self.model.scenario.population)
        self.model.nodes.S[0] = self.model.scenario.S

        return

    def prevalidate_step(self, tick: int) -> None:
        assert np.all(self.model.nodes.S[tick] >= 0), "Susceptible counts must be non-negative"

        return

    def postvalidate_step(self, tick: int) -> None:
        # Check that agents with state SUSCEPTIBLE by patch match self.model.nodes.S[tick]
        nodeids = self.model.people.nodeid
        states = self.model.people.state
        assert np.all(
            (expected := self.model.nodes.S[tick])
            == (actual := np.bincount(nodeids, states == State.SUSCEPTIBLE.value, minlength=self.model.nodes.count))
        ), f"Susceptible census does not match susceptible counts.\nExpected: {expected}\nActual: {actual}"
        assert np.all(self.model.nodes.S[tick + 1] == self.model.nodes.S[tick]), (
            "Susceptible counts should not change in Susceptible.step()."
        )

        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        # Propagate the number of susceptible individuals in each patch
        # state(t+1) = state(t) + ∆state(t), initialize state(t+1) with state(t)
        self.model.nodes.S[tick + 1] = self.model.nodes.S[tick]

        return

    def plot(self):
        _fig, ax1 = plt.subplots()
        for node in range(self.model.nodes.count):
            ax1.plot(self.model.nodes.S[:, node], label=f"Node {node}")
        ax1.set_xlabel("Tick")
        ax1.set_ylabel("Susceptible (by Node)")
        ax1.set_title("Susceptible over Time by Node")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(np.sum(self.model.nodes.S, axis=1), color="black", linestyle="--", label="Total Susceptible")
        ax2.set_ylabel("Total Susceptible")
        ax2.legend(loc="upper right")

        plt.show()

        return


class Infectious:
    def __init__(self, model):
        self.model = model
        self.model.people.add_scalar_property("itimer", dtype=np.uint8)
        self.model.nodes.add_vector_property("I", model.params.nticks + 1, dtype=np.int32)
        self.model.nodes.add_vector_property("recovered", model.params.nticks + 1, dtype=np.uint32)

        self.infectious_duration_fn = model.infectious_duration_distribution

        # convenience
        nodeids = self.model.people.nodeid
        populations = self.model.scenario.population.values

        for node in range(self.model.nodes.count):
            i_citizens = np.nonzero(nodeids == node)[0]
            assert len(i_citizens) == populations[node], f"Found {len(i_citizens)} citizens in node {node} but expected {populations[node]}"
            nseeds = self.model.scenario.I[node]
            assert nseeds <= len(i_citizens), f"Node {node} has more initial infectious ({nseeds}) than population ({len(i_citizens)})"
            if nseeds > 0:
                i_infectious = np.random.choice(i_citizens, size=nseeds, replace=False)
                self.model.people.state[i_infectious] = State.INFECTIOUS.value
                self.model.people.itimer[i_infectious] = self.infectious_duration_fn(nseeds)
                assert np.all(self.model.people.itimer[i_infectious] > 0), "Infected individuals should have itimer > 0"

        self.model.nodes.I[0] = self.model.scenario.I
        assert np.all(self.model.nodes.S[0] + self.model.nodes.I[0] == self.model.scenario.population), (
            "Initial S + I does not equal population"
        )

        return

    def prevalidate_step(self, tick: int) -> None:
        assert np.all(self.model.nodes.I[tick] >= 0), "Infected counts must be non-negative"
        i_infectious = np.nonzero(self.model.people.state == State.INFECTIOUS.value)[0]
        assert np.all(self.model.people.itimer[i_infectious] > 0), "Infected individuals should have itimer > 0"
        i_non_zero = np.nonzero(self.model.people.itimer > 0)[0]
        assert np.all(self.model.people.state[i_non_zero] == State.INFECTIOUS.value), "Only infectious individuals should have itimer > 0"

        return

    def postvalidate_step(self, tick: int) -> None:
        assert np.all(self.model.nodes.I[tick + 1] >= 0), "Infected counts must be non-negative"

        states = self.model.people.state
        itimers = self.model.people.itimer

        i_infectious = np.nonzero(states == State.INFECTIOUS.value)[0]
        assert np.all(itimers[i_infectious] > 0), "Infected individuals should have itimer > 0"
        i_non_zero = np.nonzero(itimers > 0)[0]
        assert np.all(states[i_non_zero] == State.INFECTIOUS.value), "Only infectious individuals should have itimer > 0"

        nodeids = self.model.people.nodeid
        # nodes.I should match count of infectious by node
        assert np.all(
            self.model.nodes.I[tick + 1] == np.bincount(nodeids, states == State.INFECTIOUS.value, minlength=self.model.nodes.count)
        ), "Infected census does not match infectious counts (by state)."
        assert np.all(self.model.nodes.I[tick + 1] == np.bincount(nodeids, itimers > 0, minlength=self.model.nodes.count)), (
            "Infected census does not match infectious counts (by itimer)."
        )

        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        """Step function for the Infected component.

        Args:
            tick (int): The current tick of the simulation.
        """
        # Propagate the number of infectious individuals in each patch
        # state(t+1) = state(t) + ∆state(t), initialize state(t+1) with state(t)
        self.model.nodes.I[tick + 1] = self.model.nodes.I[tick]

        i_infectious = np.nonzero(self.model.people.state == State.INFECTIOUS.value)[0]
        self.model.people.itimer[i_infectious] -= 1
        i_recovered = i_infectious[np.nonzero(self.model.people.itimer[i_infectious] == 0)[0]]
        self.model.people.state[i_recovered] = State.SUSCEPTIBLE.value

        # Update patch counts
        recovered_by_node = np.bincount(self.model.people.nodeid[i_recovered], minlength=self.model.nodes.count).astype(np.uint32)
        # state(t+1) = state(t) + ∆state(t)
        self.model.nodes.S[tick + 1] += recovered_by_node
        self.model.nodes.I[tick + 1] -= recovered_by_node
        # Record today's ∆
        self.model.nodes.recovered[tick] = recovered_by_node

        return

    def plot(self):
        _fig, ax1 = plt.subplots()
        for node in range(self.model.nodes.count):
            ax1.plot(self.model.nodes.I[:, node], label=f"Node {node}")
        ax1.set_xlabel("Tick")
        ax1.set_ylabel("Infected (by Node)")
        ax1.set_title("Infected over Time by Node")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(np.sum(self.model.nodes.I, axis=1), color="black", linestyle="--", label="Total Infected")
        ax2.set_ylabel("Total Infected")
        ax2.legend(loc="upper right")
        plt.show()

        return


class Transmission:
    def __init__(self, model):
        self.model = model
        self.model.nodes.add_vector_property("forces", model.params.nticks + 1, dtype=np.float32)
        self.model.nodes.add_vector_property("incidence", model.params.nticks + 1, dtype=np.uint32)

        self.infectious_duration_fn = model.infectious_duration_distribution

        return

    def prevalidate_step(self, tick: int) -> None: ...

    def postvalidate_step(self, tick: int) -> None:
        assert np.all(self.model.nodes.incidence[tick] >= 0), "Incidence counts must be non-negative"
        i_infectious = np.nonzero(self.model.people.state == State.INFECTIOUS.value)[0]
        assert np.all(self.model.people.itimer[i_infectious] > 0), "Infectious individuals should have itimer > 0"

        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        ft = self.model.nodes.forces[tick]
        if np.all((self.model.nodes.S[tick] + self.model.nodes.I[tick]) == 0):
            ...
        ft[:] = self.model.params.beta * self.model.nodes.I[tick] / (self.model.nodes.S[tick] + self.model.nodes.I[tick])
        transfer = ft[:, None] * self.model.network
        ft += transfer.sum(axis=0)
        ft -= transfer.sum(axis=1)

        itimers = self.model.people.itimer
        states = self.model.people.state
        susceptible = states == State.SUSCEPTIBLE.value
        draws = np.random.rand(self.model.people.count).astype(np.float32)
        nodeids = self.model.people.nodeid
        i_infections = np.nonzero((draws < ft[nodeids]) & susceptible)[0]
        itimers[i_infections] = self.infectious_duration_fn(len(i_infections))
        states[i_infections] = State.INFECTIOUS.value

        inf_by_node = np.bincount(nodeids[i_infections], minlength=self.model.nodes.count).astype(np.uint32)
        # state(t+1) = state(t) + ∆state(t)
        self.model.nodes.S[tick + 1] -= inf_by_node
        self.model.nodes.I[tick + 1] += inf_by_node
        # Record today's ∆
        self.model.nodes.incidence[tick] = inf_by_node

        return

    def plot(self):
        for node in range(self.model.nodes.count):
            plt.plot(self.model.nodes.forces[:, node], label=f"Node {node}")
        plt.xlabel("Tick")
        plt.ylabel("Force of Infection")
        plt.title("Force of Infection over Time by Node")
        plt.legend()
        plt.show()

        return


class VitalDynamics:
    def __init__(self, model):
        self.model = model
        assert hasattr(self.model, "births")
        assert hasattr(self.model, "deaths")
        self.model.nodes.add_vector_property("births", model.params.nticks + 1, dtype=np.uint32)
        self.model.nodes.add_vector_property("deaths", model.params.nticks + 1, dtype=np.uint32)
        return

    def prevalidate_step(self, tick: int) -> None:
        self._cpeople = self.model.people.count
        self._deceased = self.model.people.state == State.DECEASED.value

        return

    def postvalidate_step(self, tick: int) -> None:
        assert np.all(self.model.nodes.I[tick + 1] >= 0), "Infected counts must be non-negative"

        nbirths = self.model.births[tick].sum()
        assert self.model.people.count == self._cpeople + nbirths, "Population count mismatch after births"
        istart = self._cpeople
        iend = self.model.people.count
        # Assert that number of births by patch matches self.model.births[tick]
        birth_counts = np.bincount(self.model.people.nodeid[istart:iend], minlength=self.model.nodes.count)
        assert np.all(birth_counts == self.model.births[tick]), "Birth counts by patch mismatch"
        assert np.all(self.model.people.state[istart:iend] == State.SUSCEPTIBLE.value), "Newborns should be susceptible"
        assert np.all(self.model.people.itimer[istart:iend] == 0), "Newborns should have itimer == 0"

        ndeaths = self.model.deaths[tick].sum()
        deceased = self.model.people.state == State.DECEASED.value
        assert ndeaths == deceased.sum() - self._deceased.sum(), "Death counts mismatch"
        # Assert that new deaths by patch matches self.model.deaths[tick]
        prv = np.bincount(self.model.people.nodeid[0 : self._cpeople][self._deceased], minlength=self.model.nodes.count)
        now = np.bincount(self.model.people.nodeid[deceased], minlength=self.model.nodes.count)
        death_counts = now - prv
        assert np.all(death_counts == self.model.deaths[tick]), "Death counts by patch mismatch"

        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        # For each node id...
        for node in range(self.model.nodes.count):
            # If there are deaths in this node at this tick...
            if (ndeaths := self.model.deaths[tick, node]) > 0:
                # Find indices of non-deceased people in this node
                i_citizens = np.nonzero((self.model.people.nodeid == node) & (self.model.people.state != State.DECEASED.value))[0]
                npop = len(i_citizens)
                if npop < ndeaths:
                    print(
                        f"Warning: Node {node} has {npop} citizens but {ndeaths} deaths requested at tick {tick}. Capping deaths to {npop}."
                    )
                    ndeaths = npop
                # Sample deaths from the population
                if ndeaths > 0:
                    i_to_remove = np.random.choice(i_citizens, size=ndeaths, replace=False)
                    nsus_deaths = np.sum(self.model.people.state[i_to_remove] == State.SUSCEPTIBLE.value)

                    i_infectious = i_to_remove[np.nonzero(self.model.people.state[i_to_remove] == State.INFECTIOUS.value)[0]]
                    self.model.people.itimer[i_infectious] = 0
                    ninf_deaths = len(i_infectious)

                    self.model.people.state[i_to_remove] = State.DECEASED.value

                    # Update patch counts
                    assert nsus_deaths + ninf_deaths == ndeaths, "Death state count mismatch"
                    # state(t+1) = state(t) + ∆state(t)
                    self.model.nodes.S[tick + 1, node] -= nsus_deaths
                    self.model.nodes.I[tick + 1, node] -= ninf_deaths
                    # Record today's ∆
                    self.model.nodes.deaths[tick, node] = ndeaths

        tbirths = self.model.births[tick].sum()
        if tbirths > 0:
            istart, iend = self.model.people.add(tbirths)
            nodeids = self.model.people.nodeid[istart:iend]
            nodeids[:] = np.repeat(np.arange(self.model.nodes.count, dtype=np.uint16), self.model.births[tick])
            # State.SUSCEPTIBLE.value is the default
            # self.model.people.state[istart:iend] = State.SUSCEPTIBLE.value
            # self.model.people.itimer[istart:iend] = 0
            # state(t+1) = state(t) + ∆state(t)
            self.model.nodes.S[tick + 1] += self.model.births[tick]
            # Record today's ∆
            self.model.nodes.births[tick] = self.model.births[tick]

        return

    def plot(self):
        _fig, ax1 = plt.subplots(figsize=(16, 12))
        births = np.sum(self.model.births, axis=1)
        deaths = np.sum(self.model.deaths, axis=1)
        ax1.plot(births, label="Daily Births", color="green")
        ax1.plot(deaths, label="Daily Deaths", color="red")
        ax1.set_xlabel("Tick")
        ax1.set_ylabel("Count")
        ax1.set_title("Births and Deaths Over Time")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(np.cumsum(births), color="tab:green", linestyle="--", label="Cumulative Births")
        ax2.plot(np.cumsum(deaths), color="tab:red", linestyle="--", label="Cumulative Deaths")
        ax2.set_ylabel("Cumulative Count")
        ax2.legend(loc="upper right")
        plt.show()

        return


class Model:
    def __init__(self, scenario, params, births=None, deaths=None, skip_capacity: bool = False):
        """
        Initialize the SI model.

        Args:
            scenario (GeoDataFrame): The scenario data containing per patch population, initial S and I counts, and geometry.
            params (PropertySet): The parameters for the model, including 'nticks' and 'beta'.
            births (np.ndarray, optional): Birth counts per patch per tick. Defaults to None.
            deaths (np.ndarray, optional): Death counts per patch per tick. Defaults to None.
            skip_capacity (bool, optional): If True, skips capacity checks. Defaults to False.
        """
        self.params = params

        num_nodes = max(np.unique(scenario.nodeid)) + 1
        self.births = births if births is not None else RateMap.from_scalar(0, num_nodes, self.params.nticks).rates
        self.deaths = deaths if deaths is not None else RateMap.from_scalar(0, num_nodes, self.params.nticks).rates
        num_active = scenario.population.sum()
        if not skip_capacity:
            num_agents = num_active + self.births.sum()
        else:
            # Ignore births for capacity calculation
            num_agents = num_active

        # TODO - remove int() cast with newer version of laser-core
        self.people = LaserFrame(int(num_agents), int(num_active))
        self.nodes = LaserFrame(int(num_nodes))

        self.scenario = scenario
        self.validating = False

        # Project to EPSG:3857, calculate centroids, then convert back to degrees
        gdf_proj = self.scenario.to_crs(epsg=3857)
        centroids_proj = gdf_proj.geometry.centroid
        centroids_deg = centroids_proj.to_crs(epsg=4326)

        self.scenario["x"] = centroids_deg.x
        self.scenario["y"] = centroids_deg.y

        # Calculate pairwise distances between nodes using centroids
        longs = self.scenario["x"].values
        lats = self.scenario["y"].values
        population = self.scenario["population"].values

        # Compute distance matrix
        if len(scenario) > 1:
            dist_matrix = distance(lats, longs, lats, longs)
        else:
            dist_matrix = np.array([[0.0]], dtype=np.float32)
        assert dist_matrix.shape == (self.nodes.count, self.nodes.count), "Distance matrix shape mismatch"

        # Compute gravity network matrix
        self.network = gravity(population, dist_matrix, k=500, a=1, b=1, c=2)
        self.network = row_normalizer(self.network, (1 / 16) + (1 / 32))

        self.basemap_provider = ctx.providers.Esri.WorldImagery

        self._components = []

        return

    def run(self, label="SIS Model") -> None:
        with ts.start(f"Running Simulation {label}"):
            for tick in tqdm(range(self.params.nticks), desc=f"Running Simulation {label}"):
                for c in self.components:
                    with ts.start(f"{c.__class__.__name__}.step()"):
                        c.step(tick)

        return

    @property
    def components(self):
        return self._components

    @components.setter
    def components(self, value):
        self._components = value

        return

    @property
    def infectious_duration_distribution(self):
        return self._infectious_duration_distribution

    @infectious_duration_distribution.setter
    def infectious_duration_distribution(self, value):
        if callable(value):
            self._infectious_duration_distribution = value
        elif isinstance(value, (list, np.ndarray)):
            values = np.array(value)
            assert np.all(values > 0), "All infectious duration values must be positive"
            assert np.all(values == values.astype(int)), "All infectious duration values must be integers"
            values = values.astype(np.uint8)

            def sampler(n):
                return np.random.choice(values, size=n, replace=True)

            self._infectious_duration_distribution = sampler
        else:
            raise ValueError("infectious_duration_distribution must be a callable or a list/array of positive integers")

        return

    def _plot(self, basemap_provider=ctx.providers.Esri.WorldImagery):
        if "geometry" in self.scenario.columns:
            gdf = gpd.GeoDataFrame(self.scenario, geometry="geometry")

            if basemap_provider is None:
                pop = gdf["population"].values
                norm = mcolors.Normalize(vmin=pop.min(), vmax=pop.max())
                saturations = norm(pop)
                colors = [plt.cm.Blues(sat) for sat in saturations]
                ax = gdf.plot(facecolor=colors, edgecolor="black", linewidth=1)
                sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
                cbar.set_label("Population")
                plt.title("Node Boundaries and Populations")
            else:
                gdf_merc = gdf.to_crs(epsg=3857)
                pop = gdf_merc["population"].values
                # Plot the basemap and shape outlines
                _fig, ax = plt.subplots(figsize=(8, 8))
                bounds = gdf_merc.total_bounds  # [minx, miny, maxx, maxy]
                xmid = (bounds[0] + bounds[2]) / 2
                ymid = (bounds[1] + bounds[3]) / 2
                xhalf = (bounds[2] - bounds[0]) / 2
                yhalf = (bounds[3] - bounds[1]) / 2
                ax.set_xlim(xmid - 2 * xhalf, xmid + 2 * xhalf)
                ax.set_ylim(ymid - 2 * yhalf, ymid + 2 * yhalf)
                ctx.add_basemap(ax, source=basemap_provider)
                gdf_merc.boundary.plot(ax=ax, edgecolor="black", linewidth=1)

                # Draw circles at centroids sized by log(population)
                centroids = gdf_merc.geometry.centroid
                print(f"{pop=}")
                sizes = 20 + 2 * pop / 10_000
                ax.scatter(centroids.x, centroids.y, s=sizes, color="red", edgecolor="black", zorder=10, alpha=0.8)

                plt.title("Node Boundaries, Centroids, and Basemap")

            """
            # Add interactive hover to display population
            cursor = mplcursors.cursor(ax.collections[0], hover=True)

            @cursor.connect("add")
            def on_add(sel):
                # sel.index is a tuple; sel.index[0] is the nodeid (row index in gdf)
                nodeid = sel.index[0]
                pop_val = gdf.iloc[nodeid]["population"]
                sel.annotation.set_text(f"Population: {pop_val}")
            """

            plt.show()
        else:
            return

    def plot(self):
        self._plot(getattr(self, "basemap_provider", None))  # Pass basemap_provider argument to _plot if provided
        for c in self.components:
            if hasattr(c, "plot") and callable(c.plot):
                c.plot()

        return
