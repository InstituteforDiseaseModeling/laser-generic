"""
Waning Maternal Antibodies (MAB) Example

Builds on TransmssionWithDOI from age_at_infection.py to track age at infection,
adding a MaternalAntibodies component that gives newborns temporary immunity via maternal antibodies.
"""

from argparse import ArgumentParser

import laser.core.distributions as dists
import numba as nb
import numpy as np
from laser.core import PropertySet
from laser.core.demographics import AliasedDistribution
from laser.core.demographics import KaplanMeierEstimator
from laser.core.distributions import sample_ints
from matplotlib import pyplot as plt

from age_at_infection import TransmissionWithDOI
from laser.generic import SEIR
from laser.generic import Model
from laser.generic.newutils import ValuesMap
from utils import stdgrid

State = SEIR.State


# Implement a MaternalAntibodies component that adds a "mtimer" (maternal antibody timer) property to newborns,
# decrements it on each tick, and has an on_birth() method to set it for newborns in addition to setting newborns to Recovered.
# The component takes a mabdurdist distribution sampling function and mabdurmin minimum timer value which is used in
# the Numba compiled function called on step() to update the timer and revert to Susceptible when expired.
class MaternalAntibodies:
    def __init__(self, model, mabdurdist, mabdurmin=1, validating=False):
        self.model = model
        self.mabdurdist = mabdurdist
        self.mabdurmin = mabdurmin
        self.validating = validating
        self.model.people.add_scalar_property("mtimer", dtype=np.uint16)

        if self.validating or self.model.validating:
            self.model.people.add_scalar_property("mvalid", dtype=np.uint16)

    def on_birth(self, istart: int, iend: int, tick: int) -> None:
        # Set mtimer for newborns and set state to Recovered
        timers = self.model.people.mtimer[istart:iend]
        sample_ints(self.mabdurdist, dest=timers, tick=0, node=0)
        timers[:] = np.maximum(timers, self.mabdurmin).astype(timers.dtype)
        self.model.people.state[istart:iend] = State.RECOVERED.value

        births_by_node = np.bincount(self.model.people.nodeid[istart:iend], minlength=self.model.nodes.count)
        self.model.nodes.S[tick + 1] -= births_by_node
        self.model.nodes.R[tick + 1] += births_by_node

        if self.validating or self.model.validating:
            self.model.people.mvalid[istart:iend] = timers.astype(np.uint16)

        return

    @staticmethod
    @nb.njit(nogil=True, parallel=True, cache=True)
    def nb_mabs_step(states, mtimers, nodeids, sus_by_node):
        for i in nb.prange(len(states)):
            if states[i] == State.RECOVERED.value:
                timer = mtimers[i]
                if timer > 0:
                    timer -= 1
                    mtimers[i] = timer
                    if timer == 0:
                        states[i] = State.SUSCEPTIBLE.value
                        sus_by_node[nb.get_thread_id(), nodeids[i]] += 1
        return

    def step(self, tick: int) -> None:
        # Decrement mtimer and revert to Susceptible when expired
        self.nb_mabs_step(
            self.model.people.state,
            self.model.people.mtimer,
            self.model.people.nodeid,
            sus_by_node := np.zeros((nb.get_num_threads(), self.model.nodes.count), dtype=np.int32),
        )
        sus_by_node = sus_by_node.sum(axis=0).astype(self.model.nodes.S.dtype)
        self.model.nodes.S[tick + 1] += sus_by_node
        self.model.nodes.R[tick + 1] -= sus_by_node
        return


class LenientVitalDynamics(SEIR.VitalDynamics):
    """A VitalDynamics component that does not validate flows vs census."""

    def prevalidate_step(self, tick: int) -> None:
        # Override to do nothing
        return

    def postvalidate_step(self, tick: int) -> None:
        # Override to do nothing
        return


# Example usage in a test or simulation setup:
if __name__ == "__main__":
    # Build a model as in test_seir.py, but use TransmissionWithDOI
    parser = ArgumentParser()
    parser.add_argument("-t", "--nticks", type=int, default=3650)
    parser.add_argument("--r0", type=float, default=10.0)
    parser.add_argument("--expmean", type=float, default=4.5)
    parser.add_argument("--expscale", type=float, default=1.0)
    parser.add_argument("--infmean", type=float, default=7.0)
    parser.add_argument("--infscale", type=float, default=2.0)
    parser.add_argument("-v", "--validating", action="store_true", default=False)
    parser.add_argument("-m", type=int, default=5, help="Number of grid rows.")
    parser.add_argument("-n", type=int, default=5, help="Number of grid columns.")
    args = parser.parse_args()

    # debugging
    args.validating = True

    NTICKS = args.nticks
    R0 = args.r0
    EXPOSED_DURATION_MEAN = args.expmean
    EXPOSED_DURATION_SCALE = args.expscale
    INFECTIOUS_DURATION_MEAN = args.infmean
    INFECTIOUS_DURATION_SCALE = args.infscale

    scenario = stdgrid(args.m, args.n)  # Build scenario as in test_seir.py
    init_susceptible = np.round(scenario.population / R0).astype(np.int32)  # 1/R0 already recovered
    equilibrium_prevalence = 9000 / 12_000_000
    init_infected = np.round(equilibrium_prevalence * scenario.population).astype(np.int32)
    scenario["S"] = init_susceptible
    scenario["E"] = 0
    scenario["I"] = init_infected
    scenario["R"] = scenario.population - init_susceptible - init_infected

    params = PropertySet({"nticks": NTICKS, "beta": R0 / INFECTIOUS_DURATION_MEAN})
    birthrates_map = ValuesMap.from_scalar(35, nsteps=NTICKS, nnodes=len(scenario))

    model = Model(scenario, params, birthrates=birthrates_map.values)
    # model.validating = args.validating

    expdist = dists.normal(loc=EXPOSED_DURATION_MEAN, scale=EXPOSED_DURATION_SCALE)
    infdist = dists.normal(loc=INFECTIOUS_DURATION_MEAN, scale=INFECTIOUS_DURATION_SCALE)
    mabdist = dists.normal(loc=180, scale=5)  # Example maternal antibody duration distribution

    s = SEIR.Susceptible(model)
    e = SEIR.Exposed(model, expdist, infdist)
    i = SEIR.Infectious(model, infdist)
    r = SEIR.Recovered(model)
    tx = TransmissionWithDOI(model, expdist)
    mabs = MaternalAntibodies(model, mabdist, mabdurmin=150, validating=args.validating)
    # importation = Importation(model, period=30, new_infections=[5] * model.nodes.count, infdurdist=infdist)

    pyramid = AliasedDistribution(np.full(89, 1_000))
    survival = KaplanMeierEstimator(np.full(89, 1_000).cumsum())
    # vitals = SEIR.VitalDynamics(model, birthrates_map.values, pyramid, survival)
    vitals = LenientVitalDynamics(model, birthrates_map.values, pyramid, survival)

    # model.components = [s, r, i, e, tx, mabs, importation, vitals]
    model.components = [s, r, i, e, tx, mabs, vitals]

    label = f"SEIR with MABs and DOI ({model.people.count:,} agents in {model.nodes.count:,} nodes)"
    model.run(label)
    # After run, model.people.doi contains tick of infection for each agent

    # model.plot()

    # Let's look at people infected in the last year of the simulation, doi >= NTICKS - 365
    recent_infections = (model.people.doi >= (NTICKS - 365)) & (model.people.doi != -1)
    aoi_recent = model.people.doi[recent_infections] - model.people.dob[recent_infections]

    plt.hist(aoi_recent, bins=range(aoi_recent.min(), aoi_recent.max() + 1), alpha=0.7)
    plt.xlabel("Age at Infection (Days)")
    plt.ylabel("Number of Infections")
    plt.title(label)
    plt.tight_layout()
    plt.show()

    if args.validating:
        # Plot maternal antibody timer validation histogram
        mab_timers = model.people.mvalid[model.people.mvalid > 0]
        plt.hist(mab_timers, bins=range(mab_timers.min(), mab_timers.max() + 1), alpha=0.7)
        plt.xlabel("Maternal Antibody Timer (Days)")
        plt.ylabel("Number of Agents")
        plt.title("Maternal Antibody Timer Validation Histogram")
        plt.tight_layout()
        plt.show()

    print("done.")
