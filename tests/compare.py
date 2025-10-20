from laser_generic.newutils import TimingStats as ts  # noqa: I001

from argparse import ArgumentParser

import geopandas as gpd
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from laser_core import PropertySet

import laser_generic.models.SIR as SIR
import laser_generic.models.SIRS as SIRS
import laser_generic.models.SEIR as SEIR
import laser_generic.models.SEIRS as SEIRS
from utils import stdgrid

EM = 10
EN = 10
PEE = 10
VALIDATING = False
NTICKS = 365
R0 = 1.386  # final attack fraction of 50%
EXPOSED_DURATION_SHAPE = 4.5
EXPOSED_DURATION_SCALE = 1.0
INFECTIOUS_DURATION_MEAN = 7.0
WANING_DURATION_MEAN = 30.0
OVERLAY = True


def build_models(m, n, pop_fn, init_infected=0, init_recovered=0, birthrates=None, mortalityrates=None, pyramid=None, survival=None):
    scenario = stdgrid(M=m, N=n, population_fn=pop_fn)
    scenario["S"] = scenario["population"]
    scenario["E"] = 0
    assert np.all(scenario["S"] >= init_infected), "Initial susceptible population must be >= initial infected"
    scenario["S"] -= init_infected
    scenario["I"] = init_infected
    assert np.all(scenario["S"] >= init_recovered), "Initial susceptible population, minus initial infected, must be >= initial recovered"
    scenario["S"] -= init_recovered
    scenario["R"] = init_recovered

    beta = R0 / INFECTIOUS_DURATION_MEAN
    params = PropertySet({"nticks": NTICKS, "beta": beta})

    with ts.start("Model Initialization"):
        with ts.start("Numba Distributions"):

            @nb.njit(nogil=True, cache=True)
            def exposure_duration_distribution():
                draw = np.random.gamma(shape=EXPOSED_DURATION_SHAPE, scale=EXPOSED_DURATION_SCALE)
                rounded = np.round(draw)
                asuint8 = np.uint8(rounded)
                clipped = np.maximum(1, asuint8)
                return clipped

            @nb.njit(nogil=True, cache=True)
            def infectious_duration_distribution():
                draw = np.random.normal(loc=INFECTIOUS_DURATION_MEAN, scale=2)
                rounded = np.round(draw)
                asuint8 = np.uint8(rounded)
                clipped = np.maximum(1, asuint8)
                return clipped

            @nb.njit(nogil=True, cache=True)
            def waning_duration_distribution():
                draw = np.random.normal(loc=WANING_DURATION_MEAN, scale=5)
                rounded = np.round(draw)
                asuint8 = np.uint8(rounded)
                clipped = np.maximum(1, asuint8)
                return clipped

        sir = SIR.Model(gpd.GeoDataFrame(scenario), params, birthrates=birthrates, mortalityrates=mortalityrates)
        sir.infectious_duration_fn = infectious_duration_distribution
        sir.components = [SIR.Susceptible(sir), SIR.Recovered(sir), SIR.Infectious(sir), SIR.Transmission(sir)]

        sirs = SIRS.Model(gpd.GeoDataFrame(scenario), params, birthrates=birthrates, mortalityrates=mortalityrates)
        sirs.infectious_duration_fn = infectious_duration_distribution
        sirs.waning_duration_fn = waning_duration_distribution
        sirs.components = [SIRS.Susceptible(sirs), SIRS.Recovered(sirs), SIRS.Infectious(sirs), SIRS.Transmission(sirs)]

        seir = SEIR.Model(gpd.GeoDataFrame(scenario), params, birthrates=birthrates, mortalityrates=mortalityrates)
        seir.exposure_duration_fn = exposure_duration_distribution
        seir.infectious_duration_fn = infectious_duration_distribution
        seir.components = [SEIR.Susceptible(seir), SEIR.Recovered(seir), SEIR.Infectious(seir), SEIR.Exposed(seir), SEIR.Transmission(seir)]

        seirs = SEIRS.Model(gpd.GeoDataFrame(scenario), params, birthrates=birthrates, mortalityrates=mortalityrates)
        seirs.exposure_duration_fn = exposure_duration_distribution
        seirs.infectious_duration_fn = infectious_duration_distribution
        seirs.waning_duration_fn = waning_duration_distribution
        seirs.components = [
            SEIRS.Susceptible(seirs),
            SEIRS.Recovered(seirs),
            SEIRS.Infectious(seirs),
            SEIRS.Exposed(seirs),
            SEIRS.Transmission(seirs),
        ]

        for model in (sir, sirs, seir, seirs):
            model.validating = VALIDATING

    return sir, sirs, seir, seirs


def main():
    POPULATION = int(1e6)
    sir, sirs, seir, seirs = build_models(m=1, n=1, pop_fn=lambda x, y: POPULATION, init_infected=10)

    sir.run(f"SIR Model of {POPULATION} people on {EM}x{EN} grid)")
    sirs.run(f"SIR Model of {POPULATION} people on {EM}x{EN} grid)")
    seir.run(f"SIR Model of {POPULATION} people on {EM}x{EN} grid)")
    seirs.run(f"SIR Model of {POPULATION} people on {EM}x{EN} grid)")

    do_plots(sir, sirs, seir, seirs, OVERLAY)

    return


def do_plots(sir, sirs, seir, seirs, overlay=True):
    # Plot S, E (optionally), I, and R for each model using different styles for each model
    # Use blue for S, orange for E, red for I, and green for R

    def plot_model(ax, model, label_prefix, linestyle):
        # times = np.arange(NTICKS+1)
        S = model.nodes.S.sum(axis=1)
        I = model.nodes.I.sum(axis=1)  # noqa: E741
        R = model.nodes.R.sum(axis=1)
        # ax.plot(times, S, color="blue", linestyle="-", label=f"{label_prefix} S")
        ax.plot(S, color="blue", linestyle=linestyle, label=f"{label_prefix} S")
        if hasattr(model.nodes, "E"):
            E = model.nodes.E.sum(axis=1)
            # ax.plot(times, E, color="orange", linestyle="--", label=f"{label_prefix} E")
            ax.plot(E, color="orange", linestyle=linestyle, label=f"{label_prefix} E")
        # ax.plot(times, I, color="red", linestyle="-.", label=f"{label_prefix} I")
        # ax.plot(times, R, color="green", linestyle=":", label=f"{label_prefix} R")
        ax.plot(I, color="red", linestyle=linestyle, label=f"{label_prefix} I")
        ax.plot(R, color="green", linestyle=linestyle, label=f"{label_prefix} R")

    active = [
        (sir, "SIR", "-"),
        (sirs, "SIRS", "--"),
        (seir, "SEIR", "-."),
        (seirs, "SEIRS", ":"),
    ]

    if overlay:
        plt.figure(figsize=(10, 6))
        for model, label, style in active:
            plot_model(plt.gca(), model, label, style)
        plt.title("Comparison of SIR, SIRS, SEIR, and SEIRS Models")
        plt.xlabel("Time (days)")
        plt.ylabel("Population")
        plt.legend()
        plt.grid(True)
    else:
        _fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        for i, (model, label, style) in enumerate(active):
            plot_model(axs[i // 2, i % 2], model, label, style)
            axs[i // 2, i % 2].set_title(f"{label} Model")

        for ax in axs.flat:
            ax.set_xlabel("Time (days)")
            ax.set_ylabel("Population")
            ax.legend()
            ax.grid(True)

    plt.tight_layout()
    plt.show()

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
        help=r"Basic reproduction number (R0) [1.151 for 25%% attack fraction, 1.386=50%%, and 1.848=75%%]",
    )
    parser.add_argument("-i", "--infdur", type=float, default=7.0, help="Mean infectious duration in days")
    parser.add_argument("-w", "--wandur", type=float, default=30.0, help="Mean waning duration in days")

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

    print(f"Using arguments {args=}")

    main()
