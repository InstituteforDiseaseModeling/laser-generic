from laser.generic.newutils import TimingStats as ts  # noqa: I001

import json
import sys
import unittest
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from laser.core import PropertySet
from laser.core.demographics import AliasedDistribution
from laser.core.demographics import KaplanMeierEstimator
import laser.core.distributions as dists

from laser.generic import SIS
from laser.generic import Model
from laser.generic.newutils import ValuesMap
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
            # Black Rock Desert, NV = 40°47'13"N 119°12'15"W (40.786944, -119.204167)
            grd = stdgrid(
                M=EM,
                N=EN,
                node_size_km=10,
                population_fn=lambda x, y: int(np.random.uniform(10_000, 1_000_000)),
                # population_fn=lambda x, y: int(np.random.exponential(50_000)),
                origin_x=-119.204167,
                origin_y=40.786944,
            )
            scenario = grd
            scenario["S"] = scenario["population"] - 10
            scenario["I"] = 10

            cbr = np.random.uniform(5, 35, len(scenario))  # CBR = per 1,000 per year
            birthrate_map = ValuesMap.from_nodes(cbr, nsteps=NTICKS)

            R0 = 1.2
            infectious_duration_mean = 7.0
            beta = R0 / infectious_duration_mean
            params = PropertySet({"nticks": NTICKS, "beta": beta})

            with ts.start("Model Initialization"):
                model = Model(scenario, params, birthrates=birthrate_map.values)

                infdist = dists.normal(loc=infectious_duration_mean, scale=2)

                # Sampling this pyramid will return indices in [0, 88] with equal probability.
                pyramid = AliasedDistribution(np.full(89, 1_000))
                # The survival function will return the probability of surviving past each age.
                survival = KaplanMeierEstimator(np.full(89, 1_000).cumsum())

                s = SIS.Susceptible(model)
                i = SIS.Infectious(model, infdist)
                tx = SIS.Transmission(model, infdist)
                vitals = SIS.VitalDynamics(model, birthrate_map.values, pyramid, survival)
                model.components = [s, i, tx, vitals]

                model.validating = VALIDATING

            model.run(f"SIS Grid ({model.people.count:,}/{model.nodes.count:,})")

        if VERBOSE:
            print(model.people.describe("People"))
            print(model.nodes.describe("Nodes"))

        if PLOTTING:
            ibm = np.random.choice(len(base_maps))
            model.basemap_provider = base_maps[ibm]
            print(f"Using basemap: {model.basemap_provider.name}")
            model.plot()

        return

    # @unittest.skip("demonstrating skipping")
    def test_linear(self):
        with ts.start("test_linear"):
            # Black Rock Desert, NV = 40°47'13"N 119°12'15"W (40.786944, -119.204167)
            lin = stdgrid(
                M=1,
                N=PEE,
                node_size_km=10,
                population_fn=lambda x, y: int(np.random.uniform(10_000, 1_000_000)),
                origin_x=-119.204167,
                origin_y=40.786944,
            )
            scenario = lin
            scenario["S"] = scenario["population"] - 10
            scenario["I"] = 10

            cbr = np.random.uniform(5, 35, len(scenario))  # CBR = per 1,000 per year
            birthrate_map = ValuesMap.from_nodes(cbr, nsteps=NTICKS)

            R0 = 1.2
            infectious_duration_mean = 7.0
            beta = R0 / infectious_duration_mean
            params = PropertySet({"nticks": NTICKS, "beta": beta})

            with ts.start("Model Initialization"):
                model = Model(scenario, params, birthrate_map.values)

                infdist = dists.normal(loc=infectious_duration_mean, scale=2)

                # Sampling this pyramid will return indices in [0, 88] with equal probability.
                pyramid = AliasedDistribution(np.full(89, 1_000))
                # The survival function will return the probability of surviving past each age.
                survival = KaplanMeierEstimator(np.full(89, 1_000).cumsum())

                s = SIS.Susceptible(model)
                i = SIS.Infectious(model, infdist)
                tx = SIS.Transmission(model, infdist)
                vitals = SIS.VitalDynamics(model, birthrate_map.values, pyramid, survival)
                model.components = [s, i, tx, vitals]

                model.validating = VALIDATING

            model.run(f"SIS Linear ({model.people.count:,}/{model.nodes.count:,})")

        if VERBOSE:
            print(model.people.describe("People"))
            print(model.nodes.describe("Nodes"))

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
    parser.add_argument("--validating", action="store_true", help="Enable validating mode")

    parser.add_argument("-t", "--ticks", type=int, default=365, help="Number of days to simulate (nticks)")

    parser.add_argument("-g", "--grid", action="store_true", help="Run grid test")
    parser.add_argument("-l", "--linear", action="store_true", help="Run linear test")
    parser.add_argument("-c", "--constant", action="store_true", help="Run constant population test")

    parser.add_argument("unittest", nargs="*")  # Catch all for unittest args

    args = parser.parse_args()

    # # debugging
    # args.grid = True
    # args.validating = True

    PLOTTING = args.plot
    VERBOSE = args.verbose
    VALIDATING = args.validating

    NTICKS = args.ticks

    EM = args.m
    EN = args.n
    PEE = args.p

    print(f"Using arguments {args=}")

    if not (args.grid or args.linear or args.constant):  # Run everything
        sys.argv[1:] = args.unittest  # Pass remaining args to unittest
        unittest.main(exit=False)

    else:  # Run selected tests only
        tc = Default()

        if args.grid:
            tc.test_grid()

        if args.linear:
            tc.test_linear()

        # if args.constant:
        #     tc.test_constant() --- IGNORE ---

    ts.freeze()

    print("\nTiming Summary:")
    print("-" * 30)
    print(ts.to_string(scale="ms"))

    with Path("timing_data.json").open("w") as f:
        json.dump(ts.to_dict(scale="ms"), f, indent=4)
