from laser.generic.newutils import TimingStats as ts  # noqa: I001

import datetime

import contextily as ctx
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from laser.core import LaserFrame
from laser.core.migration import distance
from laser.core.migration import gravity
from laser.core.migration import row_normalizer
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from tqdm import tqdm

from laser.generic.newutils import ValuesMap
from laser.generic.newutils import estimate_capacity
from laser.generic.newutils import get_centroids


class Model:
    def __init__(self, scenario, params, birthrates=None, name: str = "generic", skip_capacity: bool = False):
        """
        Initialize the SI model.

        Args:
            scenario (GeoDataFrame): The scenario data containing per patch population, initial S and I counts, and geometry.
            params (PropertySet): The parameters for the model, including 'nticks' and 'beta'.
            birthrates (np.ndarray, optional): Birth rates in CBR per patch per tick. Defaults to None.
            name (str, optional): Name of the model instance. Defaults to "generic".
            skip_capacity (bool, optional): If True, skips capacity checks. Defaults to False.
        """
        self.params = params
        self.name = name

        num_nodes = max(np.unique(scenario.nodeid)) + 1
        self.birthrates = birthrates if birthrates is not None else ValuesMap.from_scalar(0, num_nodes, self.params.nticks).values
        num_active = scenario.population.sum()
        if not skip_capacity:
            num_agents = estimate_capacity(self.birthrates, scenario.population).sum()
        else:
            # Ignore births for capacity calculation
            num_agents = num_active

        # TODO - remove int() cast with newer version of laser-core
        self.people = LaserFrame(int(num_agents), int(num_active))
        self.nodes = LaserFrame(int(num_nodes))

        self.scenario = scenario
        self.validating = False

        centroids = get_centroids(scenario)
        self.scenario["x"] = centroids.x
        self.scenario["y"] = centroids.y

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
        k = getattr(self.params, "gravity_k", 500)
        a = getattr(self.params, "gravity_a", 1)
        b = getattr(self.params, "gravity_b", 1)
        c = getattr(self.params, "gravity_c", 2)
        self.network = gravity(population, dist_matrix, k=k, a=a, b=b, c=c)
        self.network = row_normalizer(self.network, (1 / 16) + (1 / 32))

        self.basemap_provider = ctx.providers.Esri.WorldImagery

        self._components = []

        return

    def run(self, label=None) -> None:
        label = label or f"{self.people.count} agents in {len(self.scenario)} nodes"
        with ts.start(f"Running Simulation: {label}"):
            for tick in tqdm(range(self.params.nticks), desc=label):
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

    def plot(self, figure: Figure = None, basemap_provider=ctx.providers.Esri.WorldImagery, **rest):
        _fig = plt.figure(figsize=(12, 9), dpi=200) if figure is None else figure

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
                _fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
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

            yield  # plt.show()

        pops = {
            pop[0]: (pop[1], pop[2])
            for pop in [
                ("S", "Susceptible", "blue"),
                ("E", "Exposed", "purple"),
                ("I", "Infectious", "orange"),
                ("R", "Recovered", "green"),
            ]
            if hasattr(self.nodes, pop[0])
        }

        # _fig, ax1 = plt.subplots(figsize=(16, 9), dpi=200)
        _fig = plt.figure(figsize=(12, 9), dpi=200) if figure is None else figure
        ax1 = _fig.add_subplot(111)

        active_population = sum([getattr(self.nodes, p) for p in pops])
        total_active = np.sum(active_population, axis=1)
        sumstr = " + ".join(p for p in pops)
        ax1.plot(total_active, label=f"Active Population ({sumstr})", color="blue")
        ax1.set_xlabel("Tick")
        ax1.set_ylabel("Active Population", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.legend(loc="upper left")

        if hasattr(self.nodes, "deaths"):
            ax2 = ax1.twinx()
            total_deceased = np.sum(self.nodes.deaths, axis=1).cumsum()
            ax2.plot(total_deceased, label="Total Deceased", color="red")
            ax2.set_ylabel("Total Deceased", color="red")
            ax2.tick_params(axis="y", labelcolor="red")
            ax2.legend(loc="upper right")

            plt.title("Active Population and Total Deceased Over Time")
        else:
            plt.title("Active Population Over Time")

        plt.tight_layout()
        yield  # plt.show()

        # Plot total pops over time
        # _fig, ax1 = plt.subplots(figsize=(16, 9), dpi=200)
        _fig = plt.figure(figsize=(12, 9), dpi=200) if figure is None else figure
        ax1 = _fig.add_subplot(111)

        totals = [(p, np.sum(getattr(self.nodes, p), axis=1)) for p in pops]
        for pop, total in totals:
            ax1.plot(total, label=f"Total {pops[pop][0]} ({pop})", color=pops[pop][1])
        ax1.set_xlabel("Tick")
        ax1.set_ylabel("Count")
        ax1.legend(loc="upper right")
        plt.title("Total Populations Over Time")
        plt.tight_layout()
        yield  # plt.show()

        return

    def visualize(self, pdf: bool = True) -> None:
        if not pdf:
            bmp = getattr(self, "basemap_provider", None)
            for c in [self, *self.components]:
                if hasattr(c, "plot") and callable(c.plot):
                    c.plot(bmp)
        else:
            pdf_filename = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            with PdfPages(pdf_filename) as pdf:
                bmp = getattr(self, "basemap_provider", None)
                for c in [self, *self.components]:
                    if hasattr(c, "plot") and callable(c.plot):
                        for _plot in c.plot(bmp):
                            pdf.savefig(_plot)
                            plt.close(_plot)

        return
