"""Components for the SI model."""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from laser_core import LaserFrame
from laser_core import PropertySet


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
        if self.model.validating:
            print(f"Pre-validating Susceptible.step() at tick {tick}")
        # ...

    def postvalidate_step(self, tick: int) -> None:
        if self.model.validating:
            print(f"Post-validating Susceptible.step() at tick {tick}")
        # ...

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
        if self.model.validating:
            print(f"Pre-validating Infected.step() at tick {tick}")
        # ...

    def postvalidate_step(self, tick: int) -> None:
        if self.model.validating:
            print(f"Post-validating Infected.step() at tick {tick}")
        # ...

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


def grid():
    return


if __name__ == "__main__":

    class Model:
        def __init__(self, scenario):
            self.params = PropertySet({"nticks": 365, "beta": 1.0 / 32})

            num_agents = scenario.population.sum()
            num_patches = max(np.unique(scenario.nodeid)) + 1

            # TODO - remove int() cast with newer version of laser-core
            self.people = LaserFrame(int(num_agents))
            self.patches = LaserFrame(int(num_patches))

            self.scenario = scenario
            self.validating = True

            self.network = np.array(
                [
                    [0.0, 0.1, 0.0],
                    [0.1, 0.0, 0.1],
                    [0.0, 0.1, 0.0],
                ],
                dtype=np.float32,
            )

            self._components = []

            return

        def run(self):
            for tick in range(1, self.params.nticks + 1):
                print(f"Tick {tick}")
                for c in self.components:
                    print(f"  Component: {c.__class__.__name__}")
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
                gdf.boundary.plot(color="black", linewidth=1)
                plt.title("Node Boundaries")
                plt.show()
            else:
                return

        def plot(self):
            for c in self.components:
                if hasattr(c, "plot") and callable(c.plot):
                    c.plot()

            return

    import pandas as pd

    # nodeid, population, initial S, initial I
    data = [
        (0, 100, 90, 10),
        (1, 150, 130, 20),
        (2, 200, 175, 25),
    ]
    scenario = pd.DataFrame(data, columns=["nodeid", "population", "S", "I"])

    model = Model(scenario)
    s = Susceptible(model)
    i = Infected(model)
    tx = Transmission(model)
    model.components = [s, i, tx]

    model.run()

    model.plot()

    print("done")
