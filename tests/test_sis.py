import sys
import unittest
from argparse import ArgumentParser
from functools import partial

import contextily as ctx
import numpy as np
from laser_core import PropertySet

import laser_generic.models.SIS as SIS
from laser_generic.newutils import RateMap
from laser_generic.newutils import draw_vital_dynamics
from laser_generic.newutils import grid
from laser_generic.newutils import linear

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
        # Black Rock Desert, NV = 40°47'13"N 119°12'15"W (40.786944, -119.204167)
        grd = grid(
            M=EM,
            N=EN,
            grid_size=10,
            population_fn=lambda x, y: int(np.random.uniform(10_000, 1_000_000)),
            # population_fn=lambda x, y: int(np.random.exponential(50_000)),
            origin_x=-119.204167,
            origin_y=40.786944,
        )
        scenario = grd
        scenario["S"] = scenario["population"] - 10
        scenario["I"] = 10

        crude_birthrate = np.random.uniform(5, 35, scenario.shape[0]) / 365
        birthrate_map = RateMap.from_patches(crude_birthrate, nsteps=365)
        crude_mortality_rate = (1 / 60) / 365  # daily mortality rate (assuming life expectancy of 60 years)
        mortality_map = RateMap.from_scalar(crude_mortality_rate, npatches=scenario.shape[0], nsteps=365)
        births, deaths = draw_vital_dynamics(birthrate_map, mortality_map, scenario["population"].values)

        R0 = 8.0
        infectious_duration_mean = 7.0
        beta = R0 / infectious_duration_mean
        params = PropertySet({"nticks": 365, "beta": beta})

        model = SIS.Model(scenario, params, births, deaths)
        draw = partial(np.random.normal, loc=infectious_duration_mean, scale=2)
        model.infectious_duration_distribution = lambda size: np.clip(np.round(draw(size=size)).astype(np.uint8), a_min=1, a_max=None)

        s = SIS.Susceptible(model)
        i = SIS.Infected(model)
        tx = SIS.Transmission(model)
        vitals = SIS.VitalDynamics(model)
        model.components = [s, i, tx, vitals]

        model.run()

        if VERBOSE:
            print(model.people.describe("People"))
            print(model.patches.describe("Patches"))

        if PLOTTING:
            ibm = np.random.choice(len(base_maps))
            model.basemap_provider = base_maps[ibm]
            print(f"Using basemap: {model.basemap_provider.name}")
            model.plot()

        return

    @unittest.skip("demonstrating skipping")
    def test_linear(self):
        # Black Rock Desert, NV = 40°47'13"N 119°12'15"W (40.786944, -119.204167)
        lin = linear(
            N=PEE,
            node_size_km=10,
            population_fn=lambda idx: int(np.random.uniform(10_000, 1_000_000)),
            origin_x=-119.204167,
            origin_y=40.786944,
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

        model = SIS.Model(scenario, params, births, deaths)

        s = SIS.Susceptible(model)
        i = SIS.Infected(model)
        tx = SIS.Transmission(model)
        vitals = SIS.VitalDynamics(model)
        model.components = [s, i, tx, vitals]

        model.run()

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
    unittest.main()
