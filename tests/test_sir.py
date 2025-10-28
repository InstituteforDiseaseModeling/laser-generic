from laser_generic.newutils import TimingStats as ts  # noqa: I001

import json
import sys
import unittest
from argparse import ArgumentParser
from pathlib import Path

import laser_core.distributions as dists
import numpy as np
from laser_core import PropertySet
from laser_core.demographics import AliasedDistribution
from laser_core.demographics import KaplanMeierEstimator

import laser_generic.models.SIR as SIR
from laser_generic.models.model import Model
from laser_generic.newutils import ValuesMap
from utils import base_maps
from utils import stdgrid

PLOTTING = False
VERBOSE = False
EM = 10
EN = 10
PEE = 10
VALIDATING = False
NTICKS = 365
R0 = 1.386  # final attack fraction of 50%


class Default(unittest.TestCase):
    def test_single(self):
        with ts.start("test_single_node"):
            scenario = stdgrid(M=1, N=1, population_fn=lambda x, y: 100_000)
            scenario["S"] = scenario["population"] - 10
            scenario["I"] = 10
            scenario["R"] = 0

            infectious_duration_mean = 7.0
            beta = R0 / infectious_duration_mean
            params = PropertySet({"nticks": NTICKS, "beta": beta})

            with ts.start("Model Initialization"):
                model = Model(scenario, params)

                infdist = dists.normal(loc=infectious_duration_mean, scale=2)

                s = SIR.Susceptible(model)
                i = SIR.Infectious(model, infdist)
                r = SIR.Recovered(model)
                tx = SIR.Transmission(model, infdist)
                # Recovered has to run _before_ Infectious to move people correctly (Infectious updates model.nodes.R)
                model.components = [s, r, i, tx]

                model.validating = VALIDATING

            model.run(f"SIR Single Node ({model.people.count:,}/{model.nodes.count:,})")

        if VERBOSE:
            print(model.people.describe("People"))
            print(model.nodes.describe("Nodes"))

        if PLOTTING:
            # ibm = np.random.choice(len(base_maps))
            model.basemap_provider = base_maps[0]  # [ibm]
            print(f"Using basemap: {model.basemap_provider.name}")
            model.plot()

        return

    def test_grid(self):
        with ts.start("test_grid"):
            scenario = stdgrid()
            scenario["S"] = scenario["population"] - 10
            scenario["I"] = 10
            scenario["R"] = 0

            cbr = np.random.uniform(5, 35, len(scenario))  # CBR = per 1,000 per year
            birthrate_map = ValuesMap.from_nodes(cbr, nsteps=NTICKS)
            # cdr = 1_000 / 60  # CDR = per 1,000 per year (assuming life expectancy of 60 years)
            # mortality_map = ValuesMap.from_scalar(cdr, nnodes=len(scenario), nsteps=NTICKS)

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

                s = SIR.Susceptible(model)
                i = SIR.Infectious(model, infdist)
                r = SIR.Recovered(model)
                tx = SIR.Transmission(model, infdist)
                vitals = SIR.VitalDynamics(model, birthrates=birthrate_map.values, pyramid=pyramid, survival=survival)
                # Recovered has to run _before_ Infectious to move people correctly (Infectious updates model.nodes.R)
                model.components = [s, r, i, tx, vitals]

                model.validating = VALIDATING

            model.run(f"SIR Grid ({model.people.count:,}/{model.nodes.count:,})")

        if VERBOSE:
            print(model.people.describe("People"))
            print(model.nodes.describe("Nodes"))

        if PLOTTING:
            # ibm = np.random.choice(len(base_maps))
            model.basemap_provider = base_maps[0]  # [ibm]
            print(f"Using basemap: {model.basemap_provider.name}")
            model.plot()

        return

    # @unittest.skip("demonstrating skipping")
    def test_linear(self):
        with ts.start("test_linear"):
            scenario = stdgrid(M=1, N=PEE)
            scenario["S"] = scenario["population"] - 10
            scenario["I"] = 10
            scenario["R"] = 0

            cbr = np.random.uniform(5, 35, len(scenario))  # CBR = per 1,000 per year
            birthrate_map = ValuesMap.from_nodes(cbr, nsteps=NTICKS)
            # cdr = 1_000 / 60  # CDR = per 1,000 per year (assuming life expectancy of 60 years)
            # mortality_map = ValuesMap.from_scalar(cdr, nnodes=len(scenario), nsteps=NTICKS)

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

                s = SIR.Susceptible(model)
                i = SIR.Infectious(model, infdist)
                r = SIR.Recovered(model)
                tx = SIR.Transmission(model, infdist)
                vitals = SIR.VitalDynamics(model, birthrates=birthrate_map.values, pyramid=pyramid, survival=survival)
                # Recovered has to run _before_ Infectious to move people correctly (Infectious updates model.nodes.R)
                model.components = [s, r, i, tx, vitals]

                model.validating = VALIDATING

            model.run(f"SIR Linear ({model.people.count:,}/{model.nodes.count:,})")

        if VERBOSE:
            print(model.people.describe("People"))
            print(model.nodes.describe("Nodes"))

        if PLOTTING:
            # ibm = np.random.choice(len(base_maps))
            model.basemap_provider = base_maps[0]  # [ibm]
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
    parser.add_argument(
        "-r",
        "--r0",
        type=float,
        default=1.386,
        help="Basic reproduction number (R0) [1.151 for 25%% attack fraction, 1.386=50%%, and 1.848=75%%]",
    )

    parser.add_argument("-g", "--grid", action="store_true", help="Run grid test")
    parser.add_argument("-l", "--linear", action="store_true", help="Run linear test")
    parser.add_argument("-s", "--single", action="store_true", help="Run single node test")
    # parser.add_argument("-c", "--constant", action="store_true", help="Run constant population test")

    parser.add_argument("unittest", nargs="*")  # Catch all for unittest args

    args = parser.parse_args()
    PLOTTING = args.plot
    VERBOSE = args.verbose
    VALIDATING = args.validating

    NTICKS = args.ticks
    R0 = args.r0

    EM = args.m
    EN = args.n
    PEE = args.p

    # # debugging
    # args.grid = True

    print(f"Using arguments {args=}")

    if not (args.grid or args.linear or args.single):  # Run everything
        sys.argv[1:] = args.unittest  # Pass remaining args to unittest
        unittest.main(exit=False)

    else:  # Run selected tests only
        tc = Default()

        if args.grid:
            tc.test_grid()

        if args.linear:
            tc.test_linear()

        if args.single:
            tc.test_single()

    ts.freeze()

    print("\nTiming Summary:")
    print("-" * 30)
    print(ts.to_string(scale="ms"))

    with Path("timing_data.json").open("w") as f:
        json.dump(ts.to_dict(scale="ms"), f, indent=4)
