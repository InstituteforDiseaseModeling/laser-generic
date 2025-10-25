import contextily as ctx
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from laser_core import LaserFrame
from laser_core.migration import distance
from laser_core.migration import gravity
from laser_core.migration import row_normalizer
from tqdm import tqdm

from laser_generic.newutils import RateMap
from laser_generic.newutils import TimingStats as ts
from laser_generic.newutils import estimate_capacity
from laser_generic.newutils import get_centroids
from laser_generic.newutils import validate

from .components import Exposed
from .components import InfectiousIRS as Infectious
from .components import RecoveredRS as Recovered
from .components import Susceptible
from .components import TransmissionSE as Transmission
from .shared import State
from .shared import sample_dobs
from .shared import sample_dods

__all__ = ["Exposed", "Infectious", "Model", "Recovered", "State", "Susceptible", "Transmission", "VitalDynamics"]


class VitalDynamics:
    def __init__(self, model, birthrates, pyramid, survival):
        self.model = model
        self.birthrates = birthrates
        self.pyramid = pyramid
        self.survival = survival

        # Date-Of-Birth and Date-Of-Death properties per agent
        self.model.people.add_property("dob", dtype=np.int16)
        self.model.people.add_property("dod", dtype=np.int16)
        # birth and death statistics per node
        self.model.nodes.add_vector_property("births", model.params.nticks + 1, dtype=np.uint32)
        self.model.nodes.add_vector_property("deaths", model.params.nticks + 1, dtype=np.uint32)

        # Initialize starting population
        dobs = self.model.people.dob[0 : self.model.people.count]
        dods = self.model.people.dod[0 : self.model.people.count]
        sample_dobs(dobs, self.pyramid, tick=0)
        sample_dods(dobs, dods, self.survival, tick=0)

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

    @staticmethod
    @nb.njit(
        # (nb.int16[:], nb.int8[:], nb.uint16[:], nb.int32[:, :], nb.int32[:, :], nb.int32[:, :], nb.int32),
        nogil=True,
        parallel=True,
        cache=True,
    )
    def nb_process_deaths(dods, states, nodeids, deceased_S, deceased_I, deceased_R, tick):
        for i in nb.prange(len(dods)):
            if dods[i] == tick:
                state = states[i]
                if state >= 0:  # Ignore already deceased
                    if state == State.SUSCEPTIBLE.value:
                        deceased_S[nb.get_thread_id(), nodeids[i]] += 1
                    elif state == State.INFECTIOUS.value:
                        deceased_I[nb.get_thread_id(), nodeids[i]] += 1
                    else:  # if state == State.RECOVERED.value:
                        deceased_R[nb.get_thread_id(), nodeids[i]] += 1
                    states[i] = State.DECEASED.value

        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        # Do mortality first, then births
        deceased_S = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        deceased_I = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        deceased_R = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        self.nb_process_deaths(
            self.model.people.dod, self.model.people.state, self.model.people.nodeid, deceased_S, deceased_I, deceased_R, tick
        )
        # Combine thread results
        deceased_S = deceased_S.sum(axis=0).astype(self.model.nodes.S.dtype)
        deceased_I = deceased_I.sum(axis=0).astype(self.model.nodes.I.dtype)
        deceased_R = deceased_R.sum(axis=0).astype(self.model.nodes.R.dtype)
        # state(t+1) = state(t) + ∆state(t)
        self.model.nodes.S[tick + 1] -= deceased_S
        self.model.nodes.I[tick + 1] -= deceased_I
        self.model.nodes.R[tick + 1] -= deceased_R
        # Record today's ∆
        self.model.nodes.deaths[tick] = deceased_S + deceased_I + deceased_R  # Record

        # Births in one fell swoop
        rates = np.power(1.0 + self.birthrates[tick] / 1000, 1.0 / 365) - 1.0
        # Use "tomorrow's" population which accounts for mortality above.
        N = self.model.nodes.S[tick + 1] + self.model.nodes.I[tick + 1] + self.model.nodes.R[tick + 1]
        births = np.round(np.random.poisson(rates * N)).astype(np.uint32)
        tbirths = births.sum()
        if tbirths > 0:
            istart, iend = self.model.people.add(tbirths)
            self.model.people.nodeid[istart:iend] = np.repeat(np.arange(self.model.nodes.count, dtype=np.uint16), births)
            # State.SUSCEPTIBLE.value is the default
            # self.model.people.state[istart:iend] = State.SUSCEPTIBLE.value
            # state(t+1) = state(t) + ∆state(t)
            self.model.nodes.S[tick + 1] += births
            # Record today's ∆
            self.model.nodes.births[tick] = births

        return

    def plot(self):
        _fig, ax1 = plt.subplots(figsize=(16, 12))
        births = np.sum(self.model.nodes.births, axis=1)
        deaths = np.sum(self.model.nodes.deaths, axis=1)
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
    def __init__(self, scenario, params, birthrates=None, skip_capacity: bool = False):
        """
        Initialize the SI model.

        Args:
            scenario (GeoDataFrame): The scenario data containing per patch population, initial S and I counts, and geometry.
            params (PropertySet): The parameters for the model, including 'nticks' and 'beta'.
            birthrates (np.ndarray, optional): Birth rates in CBR per patch per tick. Defaults to None.
            skip_capacity (bool, optional): If True, skips capacity checks. Defaults to False.
        """
        self.params = params

        num_nodes = max(np.unique(scenario.nodeid)) + 1
        self.birthrates = birthrates if birthrates is not None else RateMap.from_scalar(0, num_nodes, self.params.nticks).rates
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
        self.network = gravity(population, dist_matrix, k=500, a=1, b=1, c=2)
        self.network = row_normalizer(self.network, (1 / 16) + (1 / 32))

        self.basemap_provider = ctx.providers.Esri.WorldImagery

        self._components = []

        return

    def run(self, label="SIR Model") -> None:
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

        _fig, ax1 = plt.subplots(figsize=(10, 6))
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
        plt.show()

        # Plot total pops over time
        _fig, ax1 = plt.subplots(figsize=(10, 6))
        totals = [(p, np.sum(getattr(self.nodes, p), axis=1)) for p in pops]
        for pop, total in totals:
            ax1.plot(total, label=f"Total {pops[pop][0]} ({pop})", color=pops[pop][1])
        ax1.set_xlabel("Tick")
        ax1.set_ylabel("Count")
        ax1.legend(loc="upper right")
        plt.title("Total Populations Over Time")
        plt.tight_layout()
        plt.show()

        return

    def plot(self):
        self._plot(getattr(self, "basemap_provider", None))  # Pass basemap_provider argument to _plot if provided
        for c in self.components:
            if hasattr(c, "plot") and callable(c.plot):
                c.plot()

        return
