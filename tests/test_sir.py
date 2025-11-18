from laser.generic.newutils import TimingStats as ts  # noqa: I001

import json
import sys
import unittest
from argparse import ArgumentParser
from pathlib import Path

import laser.core.distributions as dists
import numpy as np
from laser.core import PropertySet
from laser.core.demographics import AliasedDistribution
from laser.core.demographics import KaplanMeierEstimator
from scipy.special import lambertw

from laser.generic import SIR
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

    def test_kermack_mckendrick(self):
        def attack_fraction(beta, inf_mean, pop, init_inf):
            """Approximate final attack fraction from R0 using the Kermack-McKendrick equation."""
            # Solve the equation: AF = 1 - exp(-R0 * AF)
            R0 = beta * inf_mean
            S0 = (pop - init_inf) / pop
            S_inf = -1 / R0 * lambertw(-R0 * S0 * np.exp(-R0)).real
            A = 1 - S_inf  # Attack fraction

            return A

        INIT_INF = 1_000

        cases = [
            # (1.11525 / 7, 7.0, 0.2),
            (1.2160953 / 7, 7.0, 1.0 / 3.0),
            (1.27685 / 7, 7.0, 0.4),
            (1.527 / 7, 7.0, 0.6),
            (2.011675 / 7, 7.0, 0.8),
        ]
        for beta, inf_mean, expected_af in cases:
            failed = 0
            NITERS = 10  # MAGIC# #1
            for _ in range(NITERS):
                scenario = stdgrid(M=1, N=1, population_fn=lambda x, y: 1_000_000)
                scenario["S"] = scenario["population"] - INIT_INF
                scenario["I"] = INIT_INF
                scenario["R"] = 0

                params = PropertySet({"nticks": NTICKS, "beta": beta})

                model = Model(scenario, params)

                infdurdist = dists.normal(loc=inf_mean, scale=2)

                s = SIR.Susceptible(model)
                i = SIR.Infectious(model, infdurdist)
                r = SIR.Recovered(model)
                tx = SIR.Transmission(model, infdurdist)
                # Recovered has to run _before_ Infectious to move people correctly (Infectious updates model.nodes.R)
                model.components = [s, r, i, tx]

                model.run(f"SIR Kermack-McKendrick ({model.people.count:,}/{model.nodes.count:,})")

                actual_af = model.nodes.R[-1].sum() / scenario.population.sum()
                # Check that actual attack fraction is within 5% of expected
                diff = np.abs(actual_af - expected_af)
                frac = diff / expected_af
                # assert frac <= 0.05, f"Attack fraction {actual_af:.4f} differs from expected {expected_af:.4f} by more than 5% ({frac:.2%})"
                if frac > 0.05:
                    failed += 1

            THRESHOLD = 3  # MAGIC# #2
            assert failed < THRESHOLD, (
                f"Attack fraction test failed {failed} out of {NITERS} iterations for R0={beta * inf_mean:.4f}\n***** This is a statistical test; occasional failures are expected. *****"
            )

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
