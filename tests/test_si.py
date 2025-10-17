from laser_generic.newutils import TimingStats as ts  # noqa: I001

import sys
import unittest
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from laser_core import PropertySet
from laser_core.demographics import AliasedDistribution
from laser_core.demographics import KaplanMeierEstimator

import laser_generic.models.SI as SI
from laser_generic.newutils import RateMap
from laser_generic.newutils import grid
from laser_generic.tstreemap import generate_d3_treemap_html
from utils import base_maps
from utils import stdgrid

PLOTTING = False
VERBOSE = False
EM = 10
EN = 10
PEE = 10
VALIDATING = False
NTICKS = 365


class Default(unittest.TestCase):
    def test_grid(self):
        with ts.start("test_grid"):
            scenario = stdgrid(M=EM, N=EN)
            scenario["S"] = scenario["population"] - 10
            scenario["I"] = 10

            cbr = np.random.uniform(5, 35, len(scenario))  # CBR = per 1,000 per year
            birthrate_map = RateMap.from_nodes(cbr, nsteps=NTICKS)
            cdr = 1_000 / 60  # CDR = per 1,000 per year (assuming life expectancy of 60 years)
            mortality_map = RateMap.from_scalar(cdr, nnodes=len(scenario), nsteps=NTICKS)

            params = PropertySet({"nticks": NTICKS, "beta": 1.0 / 32})

            with ts.start("Model Initialization"):
                model = SI.Model(scenario, params, birthrate_map.rates, mortality_map.rates)
                model.validating = VALIDATING

                # Sampling this pyramid will return indices in [0, 88] with equal probability.
                model.pyramid = AliasedDistribution(np.full(89, 1_000))
                # The survival function will return the probability of surviving past each age.
                model.survival = KaplanMeierEstimator(np.full(89, 1_000).cumsum())

                s = SI.Susceptible(model)
                i = SI.Infected(model)
                tx = SI.Transmission(model)
                vitals = SI.VitalDynamics(model)
                model.components = [s, i, tx, vitals]

            model.run(f"SI Grid ({model.people.count:,}/{model.nodes.count:,})")

        if VERBOSE:
            print(model.people.describe("People"))
            print(model.nodes.describe("Nodes"))

        if PLOTTING:
            if base_maps:
                ibm = np.random.choice(len(base_maps))
                model.basemap_provider = base_maps[ibm]
                print(f"Using basemap: {model.basemap_provider.name}")
            else:
                print("No base maps available.")
            model.plot()

        return

    def test_linear(self):
        with ts.start("test_linear"):
            scenario = stdgrid(M=1, N=PEE)
            scenario["S"] = scenario["population"] - 10
            scenario["I"] = 10

            cbr = np.random.uniform(5, 35, len(scenario))  # CBR = per 1,000 per year
            birthrate_map = RateMap.from_nodes(cbr, nsteps=NTICKS)
            cdr = 1_000 / 60  # CDR = per 1,000 per year (assuming life expectancy of 60 years)
            mortality_map = RateMap.from_scalar(cdr, nnodes=len(scenario), nsteps=NTICKS)

            params = PropertySet({"nticks": NTICKS, "beta": 1.0 / 32})

            with ts.start("Model Initialization"):
                model = SI.Model(scenario, params, birthrate_map.rates, mortality_map.rates)
                model.validating = VALIDATING

                # Sampling this pyramid will return indices in [0, 88] with equal probability.
                model.pyramid = AliasedDistribution(np.full(89, 1_000))
                # The survival function will return the probability of surviving past each age.
                model.survival = KaplanMeierEstimator(np.full(89, 1_000).cumsum())

                s = SI.Susceptible(model)
                i = SI.Infected(model)
                tx = SI.Transmission(model)
                vitals = SI.VitalDynamics(model)
                model.components = [s, i, tx, vitals]

            model.run(f"SI Linear ({model.people.count:,}/{model.nodes.count:,})")

        if VERBOSE:
            print(model.people.describe("People"))
            print(model.nodes.describe("Nodes"))

        if PLOTTING:
            if base_maps:
                ibm = np.random.choice(len(base_maps))
                model.basemap_provider = base_maps[ibm]
                print(f"Using basemap: {model.basemap_provider.name}")
            else:
                print("No base maps available.")
            model.plot()

        return

    def test_constant_pop(self):
        with ts.start("test_constant_pop"):
            pop = 1e6
            init_inf = 1
            # Seattle, WA = 47°36'35"N 122°19'59"W (47.609722, -122.333056)
            latitude = 47 + (36 + (35 / 60)) / 60
            longitude = -(122 + (19 + (59 / 60)) / 60)
            scenario = grid(M=1, N=1, node_size_km=10, population_fn=lambda x, y: pop, origin_x=longitude, origin_y=latitude)
            scenario["S"] = scenario.population - init_inf
            scenario["I"] = init_inf
            parameters = PropertySet({"seed": 2, "nticks": NTICKS, "verbose": True, "beta": 0.04, "cbr": 400})

            birthrate_map = RateMap.from_scalar(parameters.cbr, nsteps=parameters.nticks, nnodes=1)
            mortality_map = RateMap.from_scalar(0.0, nsteps=parameters.nticks, nnodes=1)

            with ts.start("Model Initialization"):
                model = SI.Model(scenario, parameters, birthrate_map.rates, mortality_map.rates, skip_capacity=True)
                model.validating = VALIDATING

                model.components = [
                    SI.Susceptible(model),
                    SI.Infected(model),
                    SI.ConstantPopVitalDynamics(model),
                    SI.Transmission(model),
                ]

            model.run(f"SI Constant Pop ({model.people.count:,}/{model.nodes.count:,})")

        if VERBOSE:
            print(model.people.describe("People"))
            print(model.nodes.describe("Nodes"))

        if PLOTTING:
            if base_maps:
                ibm = np.random.choice(len(base_maps))
                model.basemap_provider = base_maps[ibm]
                print(f"Using basemap: {model.basemap_provider.name}")
            else:
                print("No base maps available.")
            model.plot()

        return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Enable plotting")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-m", type=int, default=5, help="Number of grid rows (M)")
    parser.add_argument("-n", type=int, default=5, help="Number of grid columns (N)")
    parser.add_argument("-p", type=int, default=10, help="Number of linear nodes (N)")
    parser.add_argument("--validating", action="store_true", help="Enable validating mode")

    parser.add_argument("-t", "--ticks", type=int, default=365, help="Number of days to simulate (nticks)")

    parser.add_argument("-g", "--grid", action="store_true", help="Run grid test")
    parser.add_argument("-l", "--linear", action="store_true", help="Run linear test")
    parser.add_argument("-c", "--constant", action="store_true", help="Run constant population test")

    parser.add_argument("unittest", nargs="*")  # Catch all for unittest args

    args = parser.parse_args()
    PLOTTING = args.plot
    VERBOSE = args.verbose
    VALIDATING = args.validating

    NTICKS = args.ticks

    EM = args.m
    EN = args.n
    PEE = args.p

    # # debugging
    # args.grid = True

    if not (args.grid or args.linear or args.constant):  # Run everything
        sys.argv[1:] = args.unittest  # Pass remaining args to unittest
        unittest.main(exit=False)

    else:  # Run selected tests only
        tc = Default()

        if args.grid:
            tc.test_grid(args)

        if args.linear:
            tc.test_linear(args)

        if args.constant:
            tc.test_constant_pop(args)

    ts.freeze()

    print("\nTiming Summary:")
    print("-" * 30)
    print(ts.to_string(scale="ms"))

    treemap = Path.cwd() / "timing_treemap.html"
    generate_d3_treemap_html(ts, treemap, title="Workflow Execution Treemap", scale="ms", width=1200, height=800)
    print(f"✓ Created: '{treemap}'")
