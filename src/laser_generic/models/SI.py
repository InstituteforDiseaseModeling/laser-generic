"""Components for the SI model."""

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
from laser_generic.newutils import validate

__all__ = ["ConstantPopVitalDynamics", "Infected", "Model", "Susceptible", "Transmission", "VitalDynamics"]


class State(Enum):
    SUSCEPTIBLE = 0
    INFECTED = 1

    def __new__(cls, value):
        obj = object.__new__(cls)
        obj._value_ = np.uint8(value)
        return obj


class Susceptible:
    def __init__(self, model):
        self.model = model
        self.model.people.add_scalar_property("nodeid", dtype=np.uint16)
        self.model.people.add_scalar_property("state", dtype=np.uint8)
        self.model.patches.add_vector_property("S", model.params.nticks)

        self.model.people.nodeid[:] = np.repeat(np.arange(self.model.patches.count, dtype=np.uint16), self.model.scenario.population)
        self.model.patches.S[0] = self.model.scenario.S

        return

    def prevalidate_step(self, tick: int) -> None:
        if self.model.validating:
            ...

    def postvalidate_step(self, tick: int) -> None:
        if self.model.validating:
            # Check that agents with state SUSCEPTIBLE by patch match self.model.patches.S[tick]
            nodeids = self.model.people.nodeid
            states = self.model.people.state
            assert np.all(
                self.model.patches.S[tick] == np.bincount(nodeids, states == State.SUSCEPTIBLE.value, minlength=self.model.patches.count)
            ), "Susceptible census does not match susceptible counts."

        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        # Propagate the number of susceptible individuals in each patch
        if tick > 0:
            self.model.patches.S[tick] = self.model.patches.S[tick - 1]

        return

    def plot(self):
        for node in range(self.model.patches.count):
            plt.plot(self.model.patches.S[:, node], label=f"Node {node}")
        plt.xlabel("Tick")
        plt.ylabel("Susceptible")
        plt.title("Susceptible over Time by Node")
        plt.legend()
        plt.show()

        return


class Transmission:
    def __init__(self, model):
        self.model = model
        self.model.patches.add_vector_property("forces", model.params.nticks, dtype=np.float32)
        self.model.patches.add_vector_property("incidence", model.params.nticks, dtype=np.uint32)

        return

    def step(self, tick: int) -> None:
        ft = self.model.patches.forces[tick]
        ft[:] = self.model.params.beta * self.model.patches.I[tick] / (self.model.patches.S[tick] + self.model.patches.I[tick])
        transfer = ft[:, None] * self.model.network
        ft += transfer.sum(axis=0)
        ft -= transfer.sum(axis=1)

        states = self.model.people.state
        susceptible = states == State.SUSCEPTIBLE.value
        draws = np.random.rand(self.model.people.count).astype(np.float32)
        nodeids = self.model.people.nodeid
        infections = (draws < ft[nodeids]) & susceptible
        states[infections] = State.INFECTED.value
        inf_by_node = np.bincount(nodeids[infections], minlength=self.model.patches.count).astype(np.uint32)
        self.model.patches.S[tick] -= inf_by_node
        self.model.patches.I[tick] += inf_by_node
        self.model.patches.incidence[tick] = inf_by_node

        return

    def plot(self):
        for node in range(self.model.patches.count):
            plt.plot(self.model.patches.forces[:, node], label=f"Node {node}")
        plt.xlabel("Tick")
        plt.ylabel("Force of Infection")
        plt.title("Force of Infection over Time by Node")
        plt.legend()
        plt.show()

        return


class Infected:
    def __init__(self, model):
        self.model = model
        self.model.patches.add_vector_property("I", model.params.nticks)

        # convenience
        nodeids = self.model.people.nodeid
        populations = self.model.scenario.population.values

        for node in range(self.model.patches.count):
            citizens = np.nonzero(nodeids == node)[0]
            assert len(citizens) == populations[node], f"Found {len(citizens)} citizens in node {node} but expected {populations[node]}"
            nseeds = self.model.scenario.I[node]
            assert nseeds <= len(citizens), f"Node {node} has more initial infected ({nseeds}) than population ({len(citizens)})"
            if nseeds > 0:
                indices = np.random.choice(citizens, size=nseeds, replace=False)
                self.model.people.state[indices] = State.INFECTED.value

        self.model.patches.I[0] = self.model.scenario.I
        assert np.all(self.model.patches.S[0] + self.model.patches.I[0] == self.model.scenario.population), (
            "Initial S + I does not equal population"
        )

        return

    def prevalidate_step(self, tick: int) -> None:
        if self.model.validating:
            ...

    def postvalidate_step(self, tick: int) -> None:
        if self.model.validating:
            nodeids = self.model.people.nodeid
            state = self.model.people.state
            assert np.all(
                self.model.patches.I[tick] == np.bincount(nodeids, state == State.INFECTED.value, minlength=self.model.patches.count)
            ), "Infected census does not match infected counts."

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        """Step function for the Infected component.

        Args:
            tick (int): The current tick of the simulation.
        """
        # Propagate the number of infected individuals in each patch
        if tick > 0:
            self.model.patches.I[tick] = self.model.patches.I[tick - 1]

        return

    def plot(self):
        for node in range(self.model.patches.count):
            plt.plot(self.model.patches.I[:, node], label=f"Node {node}")
        plt.xlabel("Tick")
        plt.ylabel("Infected")
        plt.title("Infected over Time by Node")
        plt.legend()
        plt.show()

        return


class VitalDynamics:
    def __init__(self, model):
        self.model = model
        assert hasattr(self.model, "births")
        assert hasattr(self.model, "deaths")
        return

    def prevalidate_step(self, tick: int) -> None: ...

    def postvalidate_step(self, tick: int) -> None: ...

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None: ...

    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(np.sum(self.model.births, axis=1), label="Daily Births")
        plt.plot(np.sum(self.model.deaths, axis=1), label="Daily Deaths")
        plt.xlabel("Tick")
        plt.ylabel("Count")
        plt.title("Births and Deaths Over Time")
        plt.legend()
        plt.show()

        return


class ConstantPopVitalDynamics:
    def __init__(self, model):
        self.model = model
        self.model.patches.add_vector_property("births", model.params.nticks, dtype=np.uint32)
        self.model.patches.add_vector_property("deaths", model.params.nticks, dtype=np.uint32)

        births = (
            self.model.births if hasattr(self.model, "births") else RateMap.from_scalar(0.0, model.params.nticks, model.patches.count).rates
        )
        deaths = (
            self.model.deaths if hasattr(self.model, "deaths") else RateMap.from_scalar(0.0, model.params.nticks, model.patches.count).rates
        )

        self.rates = np.maximum(births, deaths)

        assert self.rates.shape == (self.model.params.nticks, self.model.patches.count), "Births/deaths array shape mismatch"

        return

    def prevalidate_step(self, tick: int) -> None: ...

    def postvalidate_step(self, tick: int) -> None: ...

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        for node in range(self.model.patches.count):
            # TODO - figure out tick and time step indexing
            if self.rates[tick, node] > 0:
                citizens = np.nonzero(self.model.people.nodeid == node)[0]
                recycled = np.random.choice(citizens, size=self.rates[tick, node], replace=False)
                cinfected = (self.model.people.state[recycled] == State.INFECTED.value).sum()
                self.model.people.state[recycled] = State.SUSCEPTIBLE.value
                if cinfected > self.model.patches.I[tick, node]:
                    ...
                self.model.patches.S[tick, node] += cinfected
                self.model.patches.I[tick, node] -= cinfected

        return

    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(np.sum(self.rates, axis=1), label="Daily Recycling")
        plt.xlabel("Tick")
        plt.ylabel("Count")
        plt.title("Recycling Over Time")
        plt.legend()
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

        num_patches = max(np.unique(scenario.nodeid)) + 1
        self.births = births if births is not None else RateMap.from_scalar(0, num_patches, self.params.nticks).rates
        self.deaths = deaths if deaths is not None else RateMap.from_scalar(0, num_patches, self.params.nticks).rates
        num_active = scenario.population.sum()
        if not skip_capacity:
            num_agents = num_active + self.births.sum()
        else:
            # Ignore births for capacity calculation
            num_agents = num_active

        # TODO - remove int() cast with newer version of laser-core
        self.people = LaserFrame(int(num_agents), int(num_active))
        self.patches = LaserFrame(int(num_patches))

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
        assert dist_matrix.shape == (self.patches.count, self.patches.count), "Distance matrix shape mismatch"

        # Compute gravity network matrix
        self.network = gravity(population, dist_matrix, k=500, a=1, b=1, c=2)
        self.network = row_normalizer(self.network, (1 / 16) + (1 / 32))

        self.basemap_provider = ctx.providers.Esri.WorldImagery

        self._components = []

        return

    def run(self, label="SI Model") -> None:
        for tick in tqdm(range(self.params.nticks), desc=f"Running Simulation: {label}"):
            for c in self.components:
                c.step(tick)

        return

    @property
    def components(self):
        return self._components

    @components.setter
    def components(self, value):
        self._components = value

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
