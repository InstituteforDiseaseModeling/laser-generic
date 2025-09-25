from laser_generic.newutils import TimingStats as ts  # noqa: I001

import sys
import unittest
from argparse import ArgumentParser

import contextily as ctx
import numpy as np
from laser_core import PropertySet

import laser_generic.models.SI as SI
from laser_generic.newutils import RateMap
from laser_generic.newutils import draw_vital_dynamics
from laser_generic.newutils import grid
from laser_generic.newutils import linear
from laser_generic.tstreemap import generate_d3_treemap_html

PLOTTING = False
VERBOSE = False
EM = 10
EN = 10
PEE = 10

base_maps = [
    ctx.providers.Esri.NatGeoWorldMap,
    ctx.providers.Esri.WorldGrayCanvas,
    ctx.providers.Esri.WorldImagery,
    ctx.providers.Esri.WorldPhysical,
    ctx.providers.Esri.WorldShadedRelief,
    ctx.providers.Esri.WorldStreetMap,
    ctx.providers.Esri.WorldTerrain,
    ctx.providers.Esri.WorldTopoMap,
    # ctx.providers.NASAGIBS.ModisTerraTrueColorCR,
]


class Default(unittest.TestCase):
    def test_grid(self):
        with ts.start("test_grid"):
            # Brothers, OR = 43°48'47"N 120°36'05"W (43.8130555556, -120.601388889)
            grd = grid(
                M=EM,
                N=EN,
                grid_size=10,
                population_fn=lambda x, y: int(np.random.uniform(10_000, 1_000_000)),
                # population_fn=lambda x, y: int(np.random.exponential(50_000)),
                origin_x=-120.601388889,
                origin_y=43.8130555556,
            )
            scenario = grd
            scenario["S"] = scenario["population"] - 10
            scenario["I"] = 10

            crude_birthrate = np.random.uniform(5, 35, scenario.shape[0]) / 365
            birthrate_map = RateMap.from_patches(crude_birthrate, nsteps=365)
            crude_mortality_rate = (1 / 60) / 365  # daily mortality rate (assuming life expectancy of 60 years)
            mortality_map = RateMap.from_scalar(crude_mortality_rate, npatches=scenario.shape[0], nsteps=365)
            births, deaths = draw_vital_dynamics(birthrate_map, mortality_map, scenario["population"].values)

            params = PropertySet({"nticks": 365, "beta": 1.0 / 32})

            with ts.start("Model Initialization"):
                model = SI.Model(scenario, params, births, deaths)

                s = SI.Susceptible(model)
                i = SI.Infected(model)
                tx = SI.Transmission(model)
                vitals = SI.VitalDynamics(model)
                model.components = [s, i, tx, vitals]

            model.run(f"SI Grid ({model.people.count:,}/{model.patches.count:,})")

        if VERBOSE:
            print(model.people.describe("People"))
            print(model.patches.describe("Patches"))

        if PLOTTING:
            ibm = np.random.choice(len(base_maps))
            model.basemap_provider = base_maps[ibm]
            print(f"Using basemap: {model.basemap_provider.name}")
            model.plot()

        return

    def test_linear(self):
        with ts.start("test_linear"):
            # Brothers, OR = 43°48'47"N 120°36'05"W (43.8130555556, -120.601388889)
            lin = linear(
                N=PEE,
                node_size_km=10,
                population_fn=lambda idx: int(np.random.uniform(10_000, 1_000_000)),
                origin_x=-120.601388889,
                origin_y=43.8130555556,
            )
            scenario = lin
            scenario["S"] = scenario["population"] - 10
            scenario["I"] = 10

            crude_birthrate = np.random.uniform(5, 35, scenario.shape[0]) / 365
            birthrate_map = RateMap.from_patches(crude_birthrate, nsteps=365)
            crude_mortality_rate = (1 / 60) / 365  # daily mortality rate (assuming life expectancy of 60 years)
            mortality_map = RateMap.from_scalar(crude_mortality_rate, npatches=scenario.shape[0], nsteps=365)
            births, deaths = draw_vital_dynamics(birthrate_map, mortality_map, scenario["population"].values)

            params = PropertySet({"nticks": 365, "beta": 1.0 / 32})

            with ts.start("Model Initialization"):
                model = SI.Model(scenario, params, births, deaths)

                s = SI.Susceptible(model)
                i = SI.Infected(model)
                tx = SI.Transmission(model)
                vitals = SI.VitalDynamics(model)
                model.components = [s, i, tx, vitals]

            model.run(f"SI Linear ({model.people.count:,}/{model.patches.count:,})")

        if VERBOSE:
            print(model.people.describe("People"))
            print(model.patches.describe("Patches"))

        if PLOTTING:
            ibm = np.random.choice(len(base_maps))
            model.basemap_provider = base_maps[ibm]
            print(f"Using basemap: {model.basemap_provider.name}")
            model.plot()

        return

    def test_constant_pop(self):
        with ts.start("test_constant_pop"):
            pop = 1e6
            init_inf = 1
            # Seattle, WA = 47°36'35"N 122°19'59"W (47.609722, -122.333056)
            latitude = 47 + (36 + (35 / 60)) / 60
            longitude = -(122 + (19 + (59 / 60)) / 60)
            scenario = grid(M=1, N=1, grid_size=10, population_fn=lambda x, y: pop, origin_x=latitude, origin_y=longitude)
            scenario["S"] = scenario.population - init_inf
            scenario["I"] = init_inf
            parameters = PropertySet({"seed": 2, "nticks": 730, "verbose": True, "beta": 0.04, "cbr": 400})
            birthrates = RateMap.from_scalar(parameters.cbr / 365, nsteps=parameters.nticks, npatches=1)
            mortality = RateMap.from_scalar(0.0, nsteps=parameters.nticks, npatches=1)
            births, deaths = draw_vital_dynamics(birthrates, mortality, scenario.population)
            model = SI.Model(scenario, parameters, births=births, deaths=deaths, skip_capacity=True)
            model.validating = True

            with ts.start("Model Initialization"):
                model.components = [
                    SI.Susceptible(model),
                    SI.Infected(model),
                    SI.ConstantPopVitalDynamics(model),
                    SI.Transmission(model),
                ]

            model.run(f"SI Constant Pop ({model.people.count:,}/{model.patches.count:,})")

        if VERBOSE:
            print(model.people.describe("People"))
            print(model.patches.describe("Patches"))

        if PLOTTING:
            ibm = np.random.choice(len(base_maps))
            model.basemap_provider = base_maps[ibm]
            print(f"Using basemap: {model.basemap_provider.name}")
            model.plot()

        return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Enable plotting")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-m", type=int, default=5, help="Number of grid rows (M)")
    parser.add_argument("-n", type=int, default=5, help="Number of grid columns (N)")
    parser.add_argument("-p", type=int, default=10, help="Number of linear nodes (N)")
    parser.add_argument("unittest", nargs="*")  # Catch all for unittest args

    args = parser.parse_args()
    PLOTTING = args.plot
    VERBOSE = args.verbose

    EM = args.m
    EN = args.n
    PEE = args.p

    sys.argv[1:] = args.unittest  # Pass remaining args to unittest
    unittest.main(exit=False)

    ts.freeze()

    print("\nTiming Summary:")
    print("-" * 30)
    print(ts.to_string(scale="ms"))

    generate_d3_treemap_html(ts, "timing_treemap_standard.html", title="Workflow Execution Treemap", scale="ms", width=1200, height=800)
    print("✓ Created: timing_treemap_standard.html")
