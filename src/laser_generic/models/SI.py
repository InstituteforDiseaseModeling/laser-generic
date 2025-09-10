"""Components for the SI model."""

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mplcursors
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

        self.model.people.nodeid[:] = np.repeat(
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
        self.model.patches.S[tick] = np.bincount(
            self.model.people.nodeid, self.model.people.infected == 0, minlength=self.model.patches.count
        )

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

        susceptible = self.model.people.infected == 0
        draws = np.random.rand(self.model.people.count).astype(np.float32)
        infections = (draws < ft[self.model.people.nodeid]) & susceptible
        self.model.people.infected[infections] = 1
        inf_by_node = np.bincount(self.model.people.nodeid[infections], minlength=self.model.patches.count).astype(np.uint32)
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

        for node in range(self.model.patches.count):
            citizens = np.nonzero(self.model.people.nodeid == node)[0]
            assert (
                len(citizens) == self.model.scenario.population[node]
            ), f"Found {len(citizens)} citizens in node {node} but expected {self.model.scenario.population[node]}"
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
        self.model.patches.I[tick] = np.bincount(self.model.people.nodeid, self.model.people.infected, minlength=self.model.patches.count)

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


def grid(M=5, N=5, grid_size=10_000, population=None):
    """
    Create an MxN grid of cells anchored at (0, 0) with populations and geometries.

    Args:
        M (int): Number of rows (north-south).
        N (int): Number of columns (east-west).
        grid_size (float): Size of each cell in meters.
        population (callable): Function returning population for a cell.

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
            x0 = j * grid_size_deg
            y0 = i * grid_size_deg
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


if __name__ == "__main__":

    class Model:
        def __init__(self, scenario):
            # self.params = PropertySet({"nticks": 365, "beta": 1.0 / 32})
            self.params = PropertySet({"nticks": 31, "beta": 1.0 / 32})

            num_agents = scenario.population.sum()
            num_patches = max(np.unique(scenario.nodeid)) + 1

            # TODO - remove int() cast with newer version of laser-core
            self.people = LaserFrame(int(num_agents))
            self.patches = LaserFrame(int(num_patches))

            self.scenario = scenario
            self.validating = True

            self.scenario["x"] = self.scenario.geometry.centroid.x
            self.scenario["y"] = self.scenario.geometry.centroid.y
            # Calculate pairwise distances between nodes using centroids
            longs = self.scenario["x"].values
            lats = self.scenario["y"].values
            population = self.scenario["population"].values

            # Compute distance matrix
            dist_matrix = distance(lats, longs, lats, longs)

            # Compute gravity network matrix
            self.network = gravity(population, dist_matrix, k=100, a=1, b=1, c=2)
            self.network = row_normalizer(self.network, (1 / 16) + (1 / 32))

            # self.network = np.array(
            #     [
            #         [0.0, 0.1, 0.0],
            #         [0.1, 0.0, 0.1],
            #         [0.0, 0.1, 0.0],
            #     ],
            #     dtype=np.float32,
            # )

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

        def _plot(self):
            if "geometry" in self.scenario.columns:
                gdf = gpd.GeoDataFrame(self.scenario, geometry="geometry")

                pop = gdf["population"].values
                norm = mcolors.Normalize(vmin=pop.min(), vmax=pop.max())
                saturations = 0.1 + 0.9 * norm(pop)
                colors = [mcolors.to_rgba("blue", alpha=sat) for sat in saturations]

                ax = gdf.plot(facecolor=colors, edgecolor="black", linewidth=1)
                sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
                cbar.set_label("Population")
                plt.title("Node Boundaries and Populations")

                # Add interactive hover to display population
                cursor = mplcursors.cursor(ax.collections[0], hover=True)

                @cursor.connect("add")
                def on_add(sel):
                    # sel.index is a tuple; sel.index[0] is the nodeid (row index in gdf)
                    nodeid = sel.index[0]
                    pop_val = gdf.iloc[nodeid]["population"]
                    sel.annotation.set_text(f"Population: {pop_val}")

                plt.show()
            else:
                return

        def plot(self):
            self._plot()
            for c in self.components:
                if hasattr(c, "plot") and callable(c.plot):
                    c.plot()

            return

    # # nodeid, population, initial S, initial I
    # data = [
    #     (0, 100, 90, 10),
    #     (1, 150, 130, 20),
    #     (2, 200, 175, 25),
    # ]
    # scenario = pd.DataFrame(data, columns=["nodeid", "population", "S", "I"])
    scenario = grid(M=10, N=10, grid_size=10_000)
    scenario["S"] = scenario["population"] - 10
    scenario["I"] = 10

    model = Model(scenario)
    s = Susceptible(model)
    i = Infected(model)
    tx = Transmission(model)
    model.components = [s, i, tx]

    model.run()

    model.plot()

    print("done")
