from laser_generic.newutils import TimingStats as ts  # noqa: I001

import sys
import unittest
from argparse import ArgumentParser
from pathlib import Path

import numba as nb
import numpy as np
from laser_core import PropertySet
from laser_core.demographics import AliasedDistribution
from laser_core.demographics import KaplanMeierEstimator

import laser_generic.models.SIRS as SIRS
from laser_generic.newutils import RateMap
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
R0 = 1.386  # final attack fraction of 50%
INFECTIOUS_DURATION_MEAN = 7.0
WANING_DURATION_MEAN = 30.0


def build_model(m, n, pop_fn, init_infected=0, init_recovered=0, birthrates=None, mortalityrates=None, pyramid=None, survival=None):
    scenario = stdgrid(M=m, N=n, population_fn=pop_fn)
    scenario["S"] = scenario["population"]
    assert np.all(scenario["S"] >= init_infected), "Initial susceptible population must be >= initial infected"
    scenario["S"] -= init_infected
    scenario["I"] = init_infected
    assert np.all(scenario["S"] >= init_recovered), "Initial susceptible population, minus initial infected, must be >= initial recovered"
    scenario["S"] -= init_recovered
    scenario["R"] = init_recovered

    beta = R0 / INFECTIOUS_DURATION_MEAN
    params = PropertySet({"nticks": NTICKS, "beta": beta})

    with ts.start("Model Initialization"):
        model = SIRS.Model(scenario, params, birthrates=birthrates, mortalityrates=mortalityrates)

        @nb.njit(nogil=True, cache=True)
        def infectious_duration_distribution():
            draw = np.random.normal(loc=INFECTIOUS_DURATION_MEAN, scale=2)
            rounded = np.round(draw)
            asuint8 = np.uint8(rounded)
            clipped = np.maximum(1, asuint8)
            return clipped

        model.infectious_duration_fn = infectious_duration_distribution

        @nb.njit(nogil=True, cache=True)
        def waning_duration_distribution():
            draw = np.random.normal(loc=WANING_DURATION_MEAN, scale=5)
            rounded = np.round(draw)
            asuint8 = np.uint8(rounded)
            clipped = np.maximum(1, asuint8)
            return clipped

        model.waning_duration_fn = waning_duration_distribution

        s = SIRS.Susceptible(model)
        i = SIRS.Infectious(model)
        r = SIRS.Recovered(model)
        tx = SIRS.Transmission(model)
        if birthrates is not None or mortalityrates is not None:
            assert birthrates is not None, "Birthrates must be provided for vital dynamics."
            assert mortalityrates is not None, "Mortalityrates must be provided for vital dynamics."
            model.pyramid = pyramid
            model.survival = survival
            vitals = SIRS.VitalDynamics(model)
            # Recovered has to run _before_ Infectious to move people correctly (Infectious updates model.nodes.R)
            model.components = [s, r, i, tx, vitals]
        else:
            # Recovered has to run _before_ Infectious to move people correctly (Infectious updates model.nodes.R)
            model.components = [s, r, i, tx]

        model.validating = VALIDATING

    return model


class Default(unittest.TestCase):
    def test_single(self):
        with ts.start("test_single_node"):
            model = build_model(1, 1, lambda x, y: 100_000, init_infected=10, init_recovered=0)

            model.run(f"SIRS Single Node ({model.people.count:,}/{model.nodes.count:,})")

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
            with ts.start("setup"):
                cbr = np.random.uniform(5, 35, EM * EN)  # CBR = per 1,000 per year
                birthrate_map = RateMap.from_nodes(cbr, nsteps=NTICKS)
                cdr = 1_000 / 60  # CDR = per 1,000 per year (assuming life expectancy of 60 years)
                mortality_map = RateMap.from_scalar(cdr, nnodes=EM * EN, nsteps=NTICKS)

                pyramid = AliasedDistribution(np.full(89, 1_000))  # [0, 88] with equal probability
                survival = KaplanMeierEstimator(np.full(89, 1_000).cumsum())  # equal probability each year

                model = build_model(
                    EM,
                    EN,
                    lambda x, y: int(np.random.uniform(10_000, 1_000_000)),
                    init_infected=10,
                    init_recovered=0,
                    birthrates=birthrate_map.rates,
                    mortalityrates=mortality_map.rates,
                    pyramid=pyramid,
                    survival=survival,
                )

            model.run(f"SIRS Grid ({model.people.count:,}/{model.nodes.count:,})")

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
            with ts.start("setup"):
                cbr = np.random.uniform(5, 35, PEE)  # CBR = per 1,000 per year
                birthrate_map = RateMap.from_nodes(cbr, nsteps=NTICKS)
                cdr = 1_000 / 60  # CDR = per 1,000 per year (assuming life expectancy of 60 years)
                mortality_map = RateMap.from_scalar(cdr, nnodes=PEE, nsteps=NTICKS)

                pyramid = AliasedDistribution(np.full(89, 1_000))  # [0, 88] with equal probability
                survival = KaplanMeierEstimator(np.full(89, 1_000).cumsum())  # equal probability each year

                model = build_model(
                    1,
                    PEE,
                    lambda x, y: int(np.random.uniform(10_000, 1_000_000)),
                    init_infected=10,
                    init_recovered=0,
                    birthrates=birthrate_map.rates,
                    mortalityrates=mortality_map.rates,
                    pyramid=pyramid,
                    survival=survival,
                )

            model.run(f"SIRS Linear ({model.people.count:,}/{model.nodes.count:,})")

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
        help="Basic reproduction number (R0) [1.151 for 25% attack fraction, 1.386=50%, and 1.848=75%]",
    )
    parser.add_argument("-i", "--infdur", type=float, default=7.0, help="Mean infectious duration in days")
    parser.add_argument("-w", "--wandur", type=float, default=30.0, help="Mean waning duration in days")

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
    INFECTIOUS_DURATION_MEAN = args.infdur
    WANING_DURATION_MEAN = args.wandur

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

    treemap = Path.cwd() / "timing_treemap.html"
    generate_d3_treemap_html(ts, treemap, title="Workflow Execution Treemap", scale="ms", width=1200, height=800)
    print(f"✓ Created: '{treemap}'")
