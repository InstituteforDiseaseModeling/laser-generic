import numpy as np
from laser_core.demographics import AliasedDistribution
from laser_core.demographics import KaplanMeierEstimator
from matplotlib import pyplot as plt

from age_at_infection import TransmissionWithDOI
from laser_generic.models import SEIR
from laser_generic.models.model import Model
from laser_generic.newutils import ValuesMap
from utils import stdgrid

# Implement a MaternalAntibodies component that adds a "mtimer" (maternal antibody timer) property to newborns,
# decrements it on each tick, and has an on_birth() method to set it for newborns in addition to setting newborns to Recovered.
# The component takes a mabdurdist distribution sampling function and mabdurmin minimum timer value which is used in
# the Numba compiled function called on step() to update the timer and revert to Susceptible when expired.

# Example usage in a test or simulation setup:
if __name__ == "__main__":
    # Build a model as in test_seir.py, but use TransmissionWithDOI
    import laser_core.distributions as dists
    from laser_core import PropertySet

    NTICKS = 3650
    R0 = 10  # measles-ish 1.386
    EXPOSED_DURATION_MEAN = 4.5
    EXPOSED_DURATION_SCALE = 1.0
    INFECTIOUS_DURATION_MEAN = 7.0
    INFECTIOUS_DURATION_SCALE = 2.0

    scenario = stdgrid(5, 5)  # Build scenario as in test_seir.py
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
    # model.validating = True

    expdist = dists.normal(loc=EXPOSED_DURATION_MEAN, scale=EXPOSED_DURATION_SCALE)
    infdist = dists.normal(loc=INFECTIOUS_DURATION_MEAN, scale=INFECTIOUS_DURATION_SCALE)

    s = SEIR.Susceptible(model)
    e = SEIR.Exposed(model, expdist, infdist)
    i = SEIR.Infectious(model, infdist)
    r = SEIR.Recovered(model)
    tx = TransmissionWithDOI(model, expdist)
    # importation = Importation(model, period=30, new_infections=[5] * model.nodes.count, infdurdist=infdist)

    pyramid = AliasedDistribution(np.full(89, 1_000))
    survival = KaplanMeierEstimator(np.full(89, 1_000).cumsum())
    vitals = SEIR.VitalDynamics(model, birthrates_map.values, pyramid, survival)

    model.components = [s, r, i, e, tx, vitals]  # , importation]

    label = f"SEIR with DOI ({model.people.count:,} agents in {model.nodes.count:,} nodes)"
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

    print("done.")
