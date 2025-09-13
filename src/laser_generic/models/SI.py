"""Components for the SI model."""

import contextily as ctx
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from laser_core import LaserFrame
from laser_core import PropertySet
from laser_core.migration import distance
from laser_core.migration import gravity
from laser_core.migration import row_normalizer
from shapely.geometry import Polygon
from tqdm import tqdm


def validate(pre, post):
    def decorator(func):
        def wrapper(self, tick: int, *args, **kwargs):
            if pre:
                getattr(self, pre.__name__)(tick)
            result = func(self, tick, *args, **kwargs)
            if post:
                getattr(self, post.__name__)(tick)
            return result

        return wrapper

    return decorator


class Susceptible:
    def __init__(self, model):
        self.model = model
        self.model.people.add_scalar_property("nodeid")  # uint32 (default), initial value 0 (default)
        self.model.patches.add_vector_property("S", model.params.nticks + 1)  # uint32 (default), initial value 0 (default)

        self.model.people.nodeid[: self.model.people.count] = np.repeat(
            np.arange(self.model.patches.count, dtype=np.uint32), self.model.scenario.population
        )  # .values)
        self.model.patches.S[0] = self.model.scenario.S  # .values

        return

    def prevalidate_step(self, tick: int) -> None:
        if self.model.validating and tick < 10:
            ...

    def postvalidate_step(self, tick: int) -> None:
        if self.model.validating and tick < 10:
            ...

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        # Update the number of infected individuals in each patch
        nodeids = self.model.people.nodeid[: self.model.people.count]
        infflag = self.model.people.infected[: self.model.people.count]
        self.model.patches.S[tick] = np.bincount(nodeids, infflag == 0, minlength=self.model.patches.count)

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
        self.model.patches.add_vector_property("force", model.params.nticks + 1, dtype=np.float32)  # initial value 0.0 (default)

        return

    def step(self, tick: int) -> None:
        ft = self.model.patches.force[tick]
        ft[:] = self.model.params.beta * self.model.patches.I[tick] / (self.model.patches.S[tick] + self.model.patches.I[tick])
        transfer = ft[:, None] * self.model.network
        ft += transfer.sum(axis=0)
        ft -= transfer.sum(axis=1)

        infflags = self.model.people.infected[: self.model.people.count]
        susceptible = infflags == 0
        draws = np.random.rand(self.model.people.count).astype(np.float32)
        nodeids = self.model.people.nodeid[: self.model.people.count]  # convenience
        infections = (draws < ft[nodeids]) & susceptible
        infflags[infections] = 1
        inf_by_node = np.bincount(nodeids[infections], minlength=self.model.patches.count).astype(np.uint32)
        self.model.patches.S[tick] -= inf_by_node
        self.model.patches.I[tick] += inf_by_node

        return

    def plot(self):
        for node in range(self.model.patches.count):
            plt.plot(self.model.patches.force[:, node], label=f"Node {node}")
        plt.xlabel("Tick")
        plt.ylabel("Force of Infection")
        plt.title("Force of Infection over Time by Node")
        plt.legend()
        plt.show()

        return


class Infected:
    def __init__(self, model):
        self.model = model
        self.model.people.add_scalar_property("infected", dtype=np.uint32)  # initial value 0 (default)
        self.model.patches.add_vector_property("I", model.params.nticks + 1)  # uint32 (default), initial value 0 (default)

        # Naïve/brute force approach to seed initial infections
        # seeds = np.random.choice(model.people.count, size=32, replace=False)
        # model.people.infected[seeds] = 1

        # convenience
        nodeids = self.model.people.nodeid[: self.model.people.count]
        populations = self.model.scenario.population.values

        for node in range(self.model.patches.count):
            citizens = np.nonzero(nodeids == node)[0]
            assert len(citizens) == populations[node], f"Found {len(citizens)} citizens in node {node} but expected {populations[node]}"
            nseeds = self.model.scenario.I[node]
            assert nseeds <= len(citizens), f"Node {node} has more initial infected ({nseeds}) than population ({len(citizens)})"
            if nseeds > 0:
                indices = np.random.choice(citizens, size=nseeds, replace=False)
                self.model.people.infected[indices] = 1

        self.model.patches.I[0] = self.model.scenario.I  # .values
        assert np.all(
            self.model.patches.S[0] + self.model.patches.I[0] == self.model.scenario.population
        ), "Initial S + I does not equal population"

        return

    def prevalidate_step(self, tick: int) -> None:
        if self.model.validating and tick < 10:
            ...

    def postvalidate_step(self, tick: int) -> None:
        if self.model.validating and tick < 10:
            ...

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        """Step function for the Infected component.

        Args:
            tick (int): The current tick of the simulation.
        """
        # Update the number of infected individuals in each patch
        nodeids = self.model.people.nodeid[: self.model.people.count]
        infflag = self.model.people.infected[: self.model.people.count]
        self.model.patches.I[tick] = np.bincount(nodeids, infflag, minlength=self.model.patches.count)

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
        plt.plot(np.sum(self.model.births, axis=1), label="Total Births")
        plt.plot(np.sum(self.model.deaths, axis=1), label="Total Deaths")
        plt.xlabel("Tick")
        plt.ylabel("Count")
        plt.title("Births and Deaths Over Time")
        plt.legend()
        plt.show()

        return


def grid(M=5, N=5, grid_size=10_000, population=None, origin_x=0, origin_y=0):
    """
    Create an MxN grid of cells anchored at (0, 0) with populations and geometries.

    Args:
        M (int): Number of rows (north-south).
        N (int): Number of columns (east-west).
        grid_size (float): Size of each cell in meters.
        population (callable): Function returning population for a cell.
        origin_x (float): X-coordinate of the origin (bottom-left corner).
        origin_y (float): Y-coordinate of the origin (bottom-left corner).

    Returns:
        GeoDataFrame: Columns are nodeid, population, geometry.
    """
    if population is None:

        def population():
            return int(np.random.uniform(1000, 100000))

    # Convert grid_size from meters to degrees (approximate)
    meters_per_degree = 111_320
    grid_size_deg = grid_size / meters_per_degree

    cells = []
    nodeid = 0
    for i in range(M):
        for j in range(N):
            x0 = origin_x + j * grid_size_deg
            y0 = origin_y + i * grid_size_deg
            x1 = x0 + grid_size_deg
            y1 = y0 + grid_size_deg
            poly = Polygon(
                [
                    (x0, y0),  # NW
                    (x1, y0),  # NE
                    (x1, y1),  # SE
                    (x0, y1),  # SW
                    (x0, y0),  # Close polygon
                ]
            )
            cells.append({"nodeid": nodeid, "population": population(), "geometry": poly})
            nodeid += 1

    gdf = gpd.GeoDataFrame(cells, columns=["nodeid", "population", "geometry"], crs="EPSG:4326")
    return gdf


class RateMap:
    def __init__(self, npatches: int, nsteps: int):
        self._npatches = npatches
        self._nsteps = nsteps

        return

    @staticmethod
    def from_scalar(scalar: float, npatches: int, nsteps: int) -> "RateMap":
        assert scalar >= 0.0, "scalar must be non-negative"
        assert npatches > 0, "npatches must be greater than 0"
        assert nsteps > 0, "nsteps must be greater than 0"
        instance = RateMap(npatches=npatches, nsteps=nsteps)
        tmp = np.array([[scalar]], dtype=np.float32)
        instance._data = np.broadcast_to(tmp, (nsteps, npatches))

        return instance

    @staticmethod
    def from_timeseries(data: np.ndarray, npatches: int) -> "RateMap":
        assert all(data >= 0.0), "data must be non-negative"
        assert len(data.shape) == 1, "data must be a 1D array"
        assert data.shape[0] > 0, "data must have at least one element"
        assert npatches > 0, "npatches must be greater than 0"
        nsteps = data.shape[0]
        instance = RateMap(npatches=npatches, nsteps=nsteps)
        instance._data = np.broadcast_to(data[:, None], (nsteps, npatches))

        return instance

    @staticmethod
    def from_patches(data: np.ndarray, nsteps: int) -> "RateMap":
        assert all(data >= 0.0), "data must be non-negative"
        assert len(data.shape) == 1, "data must be a 1D array"
        assert data.shape[0] > 0, "data must have at least one element"
        assert nsteps > 0, "nsteps must be greater than 0"
        npatches = data.shape[0]
        instance = RateMap(npatches=npatches, nsteps=nsteps)
        instance._data = np.broadcast_to(data[None, :], (nsteps, npatches))

        return instance

    @staticmethod
    def from_array(data: np.ndarray, writeable: bool = False) -> "RateMap":
        assert all(data >= 0.0), "data must be non-negative"
        assert len(data.shape) == 2, "data must be a 2D array"
        assert data.shape[0] > 0, "data must have at least one row"
        assert data.shape[1] > 0, "data must have at least one column"
        nsteps, npatches = data.shape
        instance = RateMap(npatches=npatches, nsteps=nsteps)
        instance._data = data.astype(np.float32)
        instance._data.flags.writeable = writeable

        return instance

    @property
    def rates(self):
        return self._data

    @property
    def npatches(self):
        return self._npatches

    @property
    def nsteps(self):
        return self._nsteps


def draw_vital_dynamics(birthrates: RateMap, mortality: RateMap, initial_pop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert (
        birthrates.npatches == mortality.npatches == initial_pop.shape[0]
    ), "birthrates, mortality, and initial_pop must have the same number of patches"
    assert birthrates.nsteps == mortality.nsteps, "birthrates and mortality must have the same number of steps"

    current_pop = initial_pop.copy()
    births = np.zeros_like(birthrates.rates, dtype=np.uint32)
    deaths = np.zeros_like(mortality.rates, dtype=np.uint32)

    for t in range(birthrates.nsteps):
        # Poisson draw for births per patch
        births[t] = np.random.poisson(birthrates.rates[t] * current_pop / 1000)  # CBR is per 1,000 population
        # Binomial draw for deaths per patch
        # np.expm1(x) computes exp(x) - 1 accurately for small x
        # -np.expm1(x) computes 1 - exp(x) accurately for small x
        # -np.expm1(-mortality.rates[t]) gives the probability of death in a time step
        deaths[t] = np.random.binomial(current_pop, -np.expm1(-mortality.rates[t]))
        # Update population
        current_pop += births[t]
        current_pop -= deaths[t]

    return births, deaths


if __name__ == "__main__":

    class Model:
        def __init__(self, scenario, births=None, deaths=None):
            # self.params = PropertySet({"nticks": 365, "beta": 1.0 / 32})
            self.params = PropertySet({"nticks": 31, "beta": 1.0 / 32})

            num_patches = max(np.unique(scenario.nodeid)) + 1
            self.births = births if births is not None else RateMap.from_scalar(0, num_patches, self.params.nticks)
            self.deaths = deaths if deaths is not None else RateMap.from_scalar(0, num_patches, self.params.nticks)
            num_active = scenario.population.sum()
            num_agents = num_active + self.births.sum() + self.deaths.sum()

            # TODO - remove int() cast with newer version of laser-core
            self.people = LaserFrame(int(num_agents), int(num_active))
            self.patches = LaserFrame(int(num_patches))

            self.scenario = scenario
            self.validating = True

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
            dist_matrix = distance(lats, longs, lats, longs)

            # Compute gravity network matrix
            self.network = gravity(population, dist_matrix, k=100, a=1, b=1, c=2)
            self.network = row_normalizer(self.network, (1 / 16) + (1 / 32))

            self.basemap_provider = ctx.providers.Esri.WorldImagery

            self._components = []

            return

        def run(self):
            for tick in tqdm(range(1, self.params.nticks + 1), desc="Running Simulation"):
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
                    # Only plot the basemap and set bounds
                    fig, ax = plt.subplots(figsize=(12, 9))
                    bounds = gdf_merc.total_bounds  # [minx, miny, maxx, maxy]
                    xmid = (bounds[0] + bounds[2]) / 2
                    ymid = (bounds[1] + bounds[3]) / 2
                    xhalf = (bounds[2] - bounds[0]) / 2
                    yhalf = (bounds[3] - bounds[1]) / 2
                    ax.set_xlim(xmid - 2 * xhalf, xmid + 2 * xhalf)
                    ax.set_ylim(ymid - 2 * yhalf, ymid + 2 * yhalf)
                    ctx.add_basemap(ax, source=basemap_provider)
                    plt.title("Basemap Only")

                # pop = gdf["population"].values
                # norm = mcolors.Normalize(vmin=pop.min(), vmax=pop.max())
                # saturations = 0.1 + 0.9 * norm(pop)
                # colors = [mcolors.to_rgba("blue", alpha=sat) for sat in saturations]

                # # Project to Web Mercator for basemap compatibility
                # gdf_merc = gdf.to_crs(epsg=3857)
                # ax = gdf_merc.plot(facecolor=colors, edgecolor="black", linewidth=1, alpha=0.5)

                # # Add basemap if provider specified
                # if basemap_provider:
                #     ctx.add_basemap(ax, source=basemap_provider)
                # sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm)
                # sm.set_array([])
                # cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
                # cbar.set_label("Population")
                # plt.title("Node Boundaries and Populations")

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
            # Pass basemap_provider argument to _plot if provided
            self._plot(getattr(self, "basemap_provider", None))
            for c in self.components:
                if hasattr(c, "plot") and callable(c.plot):
                    c.plot()

            return

    # scenario = grid(M=10, N=10, grid_size=10_000)
    # Brothers, OR = 43°48'47"N 120°36'05"W (43.8130555556, -120.601388889)
    scenario = grid(
        M=4,
        N=4,
        grid_size=10_000,
        population=lambda: int(np.random.uniform(10_000, 1_000_000)),
        origin_x=-120.601388889,
        origin_y=43.8130555556,
    )
    scenario["S"] = scenario["population"] - 10
    scenario["I"] = 10

    crude_birthrate = np.random.uniform(5, 35, scenario.shape[0]) / 365
    birthrate_map = RateMap.from_patches(crude_birthrate, nsteps=365)
    crude_mortality_rate = (1 / 60) / 365  # daily mortality rate (assuming life expectancy of 60 years)
    mortality_map = RateMap.from_scalar(crude_mortality_rate, npatches=scenario.shape[0], nsteps=365)
    births, deaths = draw_vital_dynamics(birthrate_map, mortality_map, scenario["population"].values)

    model = Model(scenario, births, deaths)

    # model.basemap_provider = ctx.providers.OpenStreetMap.Mapnik

    s = Susceptible(model)
    i = Infected(model)
    tx = Transmission(model)
    vitals = VitalDynamics(model)
    model.components = [s, i, tx, vitals]

    model.run()
    model.plot()

    print("done")

"""
OpenStreetMap
 - ctx.providers.OpenStreetMap.Mapnik (default OSM map)
 - ctx.providers.OpenStreetMap.HOT
 - ctx.providers.OpenStreetMap.BlackAndWhite
Stamen
 - ctx.providers.Stamen.Terrain
 - ctx.providers.Stamen.Toner
 - ctx.providers.Stamen.Watercolor
CartoDB
 - ctx.providers.CartoDB.Positron (light)
 - ctx.providers.CartoDB.DarkMatter (dark)
Esri
 - ctx.providers.Esri.WorldStreetMap
 - ctx.providers.Esri.WorldImagery (satellite)
 - ctx.providers.Esri.WorldTopoMap
Other
 - ctx.providers.NASAGIBS.ModisTerraTrueColorCR (NASA satellite imagery)
 - ctx.providers.GeoportailFrance.orthos (France aerial imagery)
"""
