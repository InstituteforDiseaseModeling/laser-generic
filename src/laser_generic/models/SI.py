"""Components for the SI model."""

import warnings
from enum import Enum

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
from laser_generic.newutils import validate

__all__ = ["ConstantPopVitalDynamics", "Infected", "Model", "Susceptible", "Transmission", "VitalDynamics"]


class State(Enum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    DECEASED = -1

    def __new__(cls, value):
        obj = object.__new__(cls)
        obj._value_ = np.int8(value)
        return obj


class Susceptible:
    def __init__(self, model):
        self.model = model
        self.model.people.add_scalar_property("nodeid", dtype=np.uint16)
        self.model.people.add_scalar_property("state", dtype=np.int8)
        self.model.nodes.add_vector_property("S", model.params.nticks + 1)

        self.model.people.nodeid[:] = np.repeat(np.arange(self.model.nodes.count, dtype=np.uint16), self.model.scenario.population)
        self.model.nodes.S[0] = self.model.scenario.S

        return

    def prevalidate_step(self, tick: int) -> None: ...

    def postvalidate_step(self, tick: int) -> None:
        # Check that agents with state SUSCEPTIBLE by patch match self.model.nodes.S[tick]
        nodeids = self.model.people.nodeid
        states = self.model.people.state
        assert np.all(
            (expected := self.model.nodes.S[tick])
            == (actual := np.bincount(nodeids, states == State.SUSCEPTIBLE.value, minlength=self.model.nodes.count))
        ), f"Susceptible census does not match susceptible counts.\nExpected: {expected}\nActual: {actual}"
        assert np.all(self.model.nodes.S[tick + 1] == self.model.nodes.S[tick]), (
            "Susceptible counts should not change outside of Transmission and VitalDynamics."
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


class Transmission:
    def __init__(self, model):
        self.model = model
        self.model.nodes.add_vector_property("forces", model.params.nticks + 1, dtype=np.float32)
        self.model.nodes.add_vector_property("incidence", model.params.nticks + 1, dtype=np.uint32)

        return

    @staticmethod
    @nb.njit((nb.int8[:], nb.uint16[:], nb.float32[:], nb.uint32[:, :]), nogil=True, parallel=True, cache=True)
    def nb_transmission_step(states, nodeids, ft, inf_by_node):
        for i in nb.prange(len(states)):
            if states[i] == State.SUSCEPTIBLE.value:
                # Check for infection
                draw = np.random.rand()
                nid = nodeids[i]
                if draw < ft[nid]:
                    states[i] = State.INFECTED.value
                    inf_by_node[nb.get_thread_id(), nid] += 1

        return

    def step(self, tick: int) -> None:
        ft = self.model.nodes.forces[tick]
        N = self.model.nodes.S[tick] + self.model.nodes.I[tick]
        ft[:] = self.model.params.beta * self.model.nodes.I[tick] / N
        transfer = ft[:, None] * self.model.network
        ft += transfer.sum(axis=0)
        ft -= transfer.sum(axis=1)
        ft = -np.expm1(-ft)

        inf_by_node = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.uint32)
        self.nb_transmission_step(
            self.model.people.state,
            self.model.people.nodeid,
            ft,
            inf_by_node,
        )
        inf_by_node = inf_by_node.sum(axis=0)  # Sum over threads

        # state(t+1) = state(t) + ∆state(t)
        self.model.nodes.S[tick + 1] -= inf_by_node
        self.model.nodes.I[tick + 1] += inf_by_node
        # Record today's ∆
        self.model.nodes.incidence[tick] = inf_by_node

        return

    def plot(self):
        for node in range(self.model.nodes.count):
            plt.plot(self.model.nodes.forces[:-1, node], label=f"Node {node}")
        plt.xlabel("Tick")
        plt.ylabel("Force of Infection")
        plt.title("Force of Infection over Time by Node")
        plt.legend()
        plt.show()

        return


class Infected:
    def __init__(self, model):
        self.model = model
        self.model.nodes.add_vector_property("I", model.params.nticks + 1)

        # convenience
        nodeids = self.model.people.nodeid
        populations = self.model.scenario.population.values

        for node in range(self.model.nodes.count):
            citizens = np.nonzero(nodeids == node)[0]
            assert len(citizens) == populations[node], f"Found {len(citizens)} citizens in node {node} but expected {populations[node]}"
            nseeds = self.model.scenario.I[node]
            assert nseeds <= len(citizens), f"Node {node} has more initial infected ({nseeds}) than population ({len(citizens)})"
            if nseeds > 0:
                indices = np.random.choice(citizens, size=nseeds, replace=False)
                self.model.people.state[indices] = State.INFECTED.value

        self.model.nodes.I[0] = self.model.scenario.I
        assert np.all(self.model.nodes.S[0] + self.model.nodes.I[0] == self.model.scenario.population), (
            "Initial S + I does not equal population"
        )

        return

    def prevalidate_step(self, tick: int) -> None: ...

    def postvalidate_step(self, tick: int) -> None:
        nodeids = self.model.people.nodeid
        state = self.model.people.state
        assert np.all(
            (expected := self.model.nodes.I[tick])
            == (actual := np.bincount(nodeids, state == State.INFECTED.value, minlength=self.model.nodes.count))
        ), f"Infected census does not match infected counts.\nExpected: {expected}\nActual: {actual}"
        assert np.all(self.model.nodes.I[tick + 1] == self.model.nodes.I[tick]), (
            "Infected counts should not change outside of Transmission and VitalDynamics."
        )

        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        """Step function for the Infected component.

        Args:
            tick (int): The current tick of the simulation.
        """
        # Propagate the number of infected individuals in each patch
        # state(t+1) = state(t) + ∆state(t), initialize state(t+1) with state(t)
        self.model.nodes.I[tick + 1] = self.model.nodes.I[tick]

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


class VitalDynamics:
    def __init__(self, model):
        self.model = model

        # This component requires these model properties
        assert hasattr(self.model, "birthrates")
        assert hasattr(self.model, "mortalityrates")
        assert hasattr(self.model, "pyramid")
        assert hasattr(self.model, "survival")

        # Date-Of-Birth and Date-Of-Death properties per agent
        self.model.people.add_property("dob", dtype=np.int16)
        self.model.people.add_property("dod", dtype=np.int16)
        # birth and death statistics per node
        self.model.nodes.add_vector_property("births", model.params.nticks + 1, dtype=np.uint32)
        self.model.nodes.add_vector_property("deaths", model.params.nticks + 1, dtype=np.uint32)

        # Initialize starting population
        self.pyramid = self.model.pyramid
        dobs = self.model.people.dob
        # Get years of age sampled from the population pyramid
        dobs[:] = self.pyramid.sample(self.model.people.count).astype(np.int16)  # Fit in np.int16
        dobs *= 365  # Convert years to days
        dobs += np.random.randint(0, 365, size=len(dobs))  # add some noise within the year
        dods = self.model.people.dod
        dods[:] = self.model.survival.predict_age_at_death(dobs).astype(np.int16)  # Fit in np.int16
        dods -= dobs  # How many more days will each person live?
        # pyramid.sample actually returned ages. Turn them into dobs by treating them
        # as days before today (t = 0). dobs = 0 - dobs == dobs = -dobs
        dobs *= -1

        return

    def prevalidate_step(self, tick: int) -> None:
        self._cpeople = self.model.people.count
        self._deceased = self.model.people.state == State.DECEASED.value

        return

    def postvalidate_step(self, tick: int) -> None:
        nbirths = self.model.birthrates[tick].sum()
        assert self.model.people.count == self._cpeople + nbirths, "Population count mismatch after births"
        istart = self._cpeople
        iend = self.model.people.count
        # Assert that number of births by patch matches self.model.birthrates[tick]
        birth_counts = np.bincount(self.model.people.nodeid[istart:iend], minlength=self.model.nodes.count)
        assert np.all(birth_counts == self.model.birthrates[tick]), "Birth counts by patch mismatch"
        assert np.all(self.model.people.state[istart:iend] == State.SUSCEPTIBLE.value), "Newborns should be susceptible"
        assert np.all(self.model.people.itimer[istart:iend] == 0), "Newborns should have itimer == 0"

        ndeaths = self.model.mortalityrates[tick].sum()
        deceased = self.model.people.state == State.DECEASED.value
        assert ndeaths == deceased.sum() - self._deceased.sum(), "Death counts mismatch"
        # Assert that new deaths by patch matches self.model.mortalityrates[tick]
        prv = np.bincount(self.model.people.nodeid[0 : self._cpeople][self._deceased], minlength=self.model.nodes.count)
        now = np.bincount(self.model.people.nodeid[deceased], minlength=self.model.nodes.count)
        death_counts = now - prv
        assert np.all(death_counts == self.model.mortalityrates[tick]), "Death counts by patch mismatch"

        return

    @staticmethod
    @nb.njit((nb.int16[:], nb.int8[:], nb.uint16[:], nb.int32[:, :], nb.int32[:, :], nb.int32), nogil=True, parallel=True, cache=True)
    def nb_process_deaths(dods, states, nodeids, delta_S, delta_I, tick):
        for i in nb.prange(len(dods)):
            if dods[i] == tick:
                if states[i] == State.SUSCEPTIBLE.value:
                    delta_S[nb.get_thread_id(), nodeids[i]] -= 1
                else:
                    delta_I[nb.get_thread_id(), nodeids[i]] -= 1
                states[i] = State.DECEASED.value

        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        # Do mortality first, then births
        delta_S = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        delta_I = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        self.nb_process_deaths(self.model.people.dod, self.model.people.state, self.model.people.nodeid, delta_S, delta_I, tick)
        # Combine thread results
        delta_S = delta_S.sum(axis=0).astype(self.model.nodes.S.dtype)
        delta_I = delta_I.sum(axis=0).astype(self.model.nodes.I.dtype)
        # state(t+1) = state(t) + ∆state(t)
        self.model.nodes.S[tick + 1] += delta_S  # delta_S is negative or zero
        self.model.nodes.I[tick + 1] += delta_I  # delta_I is negative or zero
        # Record today's ∆
        self.model.nodes.deaths[tick] = -(delta_S + delta_I)  # Record

        # Births in one fell swoop
        tbirths = self.model.birthrates[tick].sum()
        if tbirths > 0:
            istart, iend = self.model.people.add(tbirths)
            self.model.people.nodeid[istart:iend] = np.repeat(
                np.arange(self.model.nodes.count, dtype=np.uint16), self.model.birthrates[tick]
            )
            # State.SUSCEPTIBLE.value is the default
            # self.model.people.state[istart:iend] = State.SUSCEPTIBLE.value
            # state(t+1) = state(t) + ∆state(t)
            self.model.nodes.S[tick + 1] += self.model.birthrates[tick]
            # Record today's ∆
            self.model.nodes.births[tick] = self.model.birthrates[tick]

        return

    def plot(self):
        _fig, ax1 = plt.subplots(figsize=(16, 12))
        birthrates = np.sum(self.model.birthrates, axis=1)
        mortalityrates = np.sum(self.model.mortalityrates, axis=1)
        ax1.plot(birthrates, label="Daily Birth Rates", color="green")
        ax1.plot(mortalityrates, label="Daily Mortality Rates", color="red")
        ax1.set_xlabel("Tick")
        ax1.set_ylabel("Count")
        ax1.set_title("Birth Rates and Mortality Rates Over Time")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(np.cumsum(birthrates), color="tab:green", linestyle="--", label="Cumulative Birth Rates")
        ax2.plot(np.cumsum(mortalityrates), color="tab:red", linestyle="--", label="Cumulative Mortality Rates")
        ax2.set_ylabel("Cumulative Count")
        ax2.legend(loc="upper right")
        plt.show()

        return


class ConstantPopVitalDynamics:
    def __init__(self, model):
        self.model = model
        self.model.nodes.add_vector_property("births", model.params.nticks + 1, dtype=np.uint32)
        self.model.nodes.add_vector_property("deaths", model.params.nticks + 1, dtype=np.uint32)

        if hasattr(self.model, "birthrates"):
            birthrates = self.model.birthrates
        else:
            birthrates = RateMap.from_scalar(0.0, model.params.nticks, model.nodes.count).rates
            warnings.warn("No birthrates found in model; defaulting to zero birthrates.", stacklevel=2)

        if hasattr(self.model, "mortalityrates"):
            mortalityrates = self.model.mortalityrates
        else:
            mortalityrates = RateMap.from_scalar(0.0, model.params.nticks, model.nodes.count).rates
            warnings.warn("No mortalityrates found in model; defaulting to zero mortalityrates.", stacklevel=2)

        self.rates = np.maximum(birthrates, mortalityrates)

        assert self.rates.shape == (self.model.params.nticks, self.model.nodes.count), "Births/deaths array shape mismatch"

        return

    def prevalidate_step(self, tick: int) -> None: ...

    def postvalidate_step(self, tick: int) -> None: ...

    @staticmethod
    @nb.njit((nb.float32[:], nb.int8[:], nb.uint16[:], nb.int32[:, :], nb.int32[:, :]), nogil=True, parallel=True, cache=True)
    def nb_process_recycling(rates, states, nodeids, recycled, infected):
        for i in nb.prange(len(states)):
            draw = np.random.rand()
            nid = nodeids[i]
            if draw < rates[nid]:
                tid = nb.get_thread_id()
                recycled[tid, nid] += 1
                if states[i] == State.INFECTED.value:
                    states[i] = State.SUSCEPTIBLE.value
                    infected[tid, nid] += 1
                # else: # states[i] is already SUSCEPTIBLE, no change

        return

    @validate(pre=prevalidate_step, post=postvalidate_step)
    def step(self, tick: int) -> None:
        # for node in range(self.model.nodes.count):
        #     # TODO - figure out tick and time step indexing
        #     if self.rates[tick, node] > 0:
        #         citizens = np.nonzero(self.model.people.nodeid == node)[0]
        #         recycled = np.random.choice(citizens, size=self.rates[tick, node], replace=False)
        #         cinfected = (self.model.people.state[recycled] == State.INFECTED.value).sum()
        #         self.model.people.state[recycled] = State.SUSCEPTIBLE.value
        #         if cinfected > self.model.nodes.I[tick, node]:
        #             ...
        #         # state(t+1) = state(t) + ∆state(t)
        #         self.model.nodes.S[tick + 1, node] += cinfected
        #         self.model.nodes.I[tick + 1, node] -= cinfected

        probabilities = (self.rates[tick] / (self.model.nodes.S[tick] + self.model.nodes.I[tick])).astype(np.float32)
        # probabilities = np.clip(probabilities, a_min=0.0, a_max=1.0)
        probabilities = -np.expm1(-probabilities)  # Convert to rates to probabilities
        recycled = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        infected = np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32)
        self.nb_process_recycling(probabilities, self.model.people.state, self.model.people.nodeid, recycled, infected)
        recycled = recycled.sum(axis=0).astype(self.model.nodes.S.dtype)  # Sum over threads
        infected = infected.sum(axis=0).astype(self.model.nodes.I.dtype)  # Sum over threads
        # state(t+1) = state(t) + ∆state(t)
        self.model.nodes.S[tick + 1] += infected
        self.model.nodes.I[tick + 1] -= infected
        # Record today's ∆
        self.model.nodes.deaths[tick] = recycled  # Record recycled as "deaths

        return

    def plot(self):
        plt.figure(figsize=(16, 12))
        plt.plot(np.sum(self.rates, axis=1), label="Daily Recycling")
        plt.xlabel("Tick")
        plt.ylabel("Count")
        plt.title("Recycling Over Time")
        plt.legend()
        plt.show()

        return


class Model:
    def __init__(self, scenario, params, birthrates=None, mortalityrates=None, skip_capacity: bool = False):
        """
        Initialize the SI model.

        Args:
            scenario (GeoDataFrame): The scenario data containing per patch population, initial S and I counts, and geometry.
            params (PropertySet): The parameters for the model, including 'nticks' and 'beta'.
            birthrates (np.ndarray, optional): Birth counts per patch per tick. Defaults to None.
            deaths (np.ndarray, optional): Death counts per patch per tick. Defaults to None.
            skip_capacity (bool, optional): If True, skips capacity checks. Defaults to False.
        """
        self.params = params

        num_nodes = max(np.unique(scenario.nodeid)) + 1
        self.birthrates = birthrates if birthrates is not None else RateMap.from_scalar(0, num_nodes, self.params.nticks).rates
        self.mortalityrates = mortalityrates if mortalityrates is not None else RateMap.from_scalar(0, num_nodes, self.params.nticks).rates
        num_active = scenario.population.sum()
        if not skip_capacity:
            num_agents = num_active + self.birthrates.sum()
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
        # TODO - figure out how to avoid the DeprecationWarning coming from NumPy via GeoPandas
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
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
        # self.on_birth = PubSub("on_birth")

        return

    def run(self, label="SI Model") -> None:
        with ts.start(f"Running Simulation: {label}"):
            for tick in tqdm(range(self.params.nticks), desc=f"Running Simulation: {label}"):
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
                _fig, ax = plt.subplots(figsize=(16, 12))
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
