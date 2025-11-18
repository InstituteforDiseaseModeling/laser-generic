import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from laser.core import PropertySet
from laser.generic.infection import Infection
from laser.generic.infection import Infection_SIS
from laser.generic.transmission import TransmissionSIR

from laser.generic import Births
from laser.generic import Births_ConstantPop
from laser.generic import Births_ConstantPop_VariableBirthRate
from laser.generic import Exposure
from laser.generic import ImmunizationCampaign
from laser.generic import Infect_Random_Agents
from laser.generic import Model
from laser.generic import RoutineImmunization
from laser.generic import Susceptibility
from laser.generic import Transmission
from laser.generic.importation import Infect_Agents_In_Patch
from laser.generic.utils import get_default_parameters
from laser.generic.utils import seed_infections_in_patch
from laser.generic.utils import seed_infections_randomly
from laser.generic.utils import seed_infections_randomly_SI


def assert_model_sanity(model):
    S_counts = model.patches.susceptibility_test[1:, 0]  # drop tick 0
    I_counts = model.patches.cases_test[1:, 0]  # drop tick 0
    N_counts = model.patches.populations[1:, 0]  # already len == nticks
    inc = model.patches.incidence[:, 0]  # already len == nticks

    assert np.sum(inc) > 0, "No transmission occurred"
    assert np.any(S_counts < S_counts[0]), "Susceptibles never decreased"
    assert np.all(S_counts >= 0), "Negative susceptible count"
    assert np.all(S_counts <= N_counts), "Susceptibles exceed population"

    I_derived = np.cumsum(inc) + model.patches.cases_test[0, 0]
    assert np.allclose(I_counts, I_derived, atol=1e-5), "Cases not consistent with incidence"


@pytest.fixture
def stable_transmission_model():
    return baseline_model()


def baseline_model():
    pop = int(1e5)
    nticks = 365
    params = get_default_parameters() | {
        "seed": 42,
        "nticks": nticks,
        "beta": 0.3,
        "inf_mean": 7,
        "verbose": False,
    }
    scenario = pd.DataFrame({"name": ["home"], "population": [pop]})
    model = Model(scenario, params)
    model.components = [
        Births_ConstantPop,
        Susceptibility,
        Exposure,
        Infection,
        Transmission,
    ]
    # seed_infections_randomly_SI(model, ninfections=50)
    seed_infections_in_patch(model, ninfections=50, ipatch=0)
    return model


@pytest.mark.modeltest
def test_si_model_nobirths_flow():
    nticks = 180
    pop = int(1e5)
    scenario = pd.DataFrame(data=[["homenode", pop, "0,0"]], columns=["name", "population", "location"])
    parameters = PropertySet({"seed": 42, "nticks": nticks, "verbose": False, "beta": 0.03})

    model = Model(scenario, parameters)
    model.components = [Susceptibility, Transmission]
    seed_infections_randomly_SI(model, ninfections=1)
    model.run()

    assert_model_sanity(model)
    assert model.patches.cases_test[-1, 0] > model.patches.cases_test[0, 0], "Infection count should increase"


@pytest.mark.modeltest
@pytest.mark.xfail(reason="Known issue not yet fixed", strict=True)
def test_sir_nobirths_short():
    pop = int(1e5)
    nticks = 365
    beta = 0.06
    gamma = 1 / 20
    scenario = pd.DataFrame(data=[["homenode", pop]], columns=["name", "population"])
    parameters = PropertySet({"seed": 1, "nticks": nticks, "verbose": False, "beta": beta, "inf_mean": 1 / gamma})

    model = Model(scenario, parameters)
    model.components = [Susceptibility, Infection, Transmission]
    seed_infections_randomly(model, ninfections=50)
    model.run()

    assert_model_sanity(model)
    peak_I = np.max(model.patches.cases_test[:, 0])
    final_I = model.patches.cases_test[-1, 0]
    assert final_I < peak_I, "SIR model should recover"


@pytest.mark.modeltest
@pytest.mark.xfail(
    reason="Known issue not yet fixed: AttributeError: 'LaserFrame' object has no attribute 'etimer'. Issue #24.", strict=True
)
def test_si_model_with_births_short():
    pop = int(1e5)
    nticks = 365 * 2
    beta = 0.02
    cbr = 0.03
    scenario = pd.DataFrame(data=[["homenode", pop]], columns=["name", "population"])
    parameters = PropertySet({"seed": 123, "nticks": nticks, "verbose": False, "beta": beta, "cbr": cbr})

    model = Model(scenario, parameters)
    model.components = [Births_ConstantPop, Susceptibility, Transmission]
    seed_infections_randomly_SI(model, ninfections=10)
    model.run()

    assert_model_sanity(model)
    assert model.patches.cases_test[-1, 0] > 0, "Infections should persist with demographic turnover"


@pytest.mark.modeltest
@pytest.mark.xfail(
    reason="Known issue not yet fixed: AttributeError: 'LaserFrame' object has no attribute 'etimer'. Issue #24.", strict=True
)
def test_sei_model_with_births_short():
    pop = int(1e5)
    nticks = 365 * 2
    beta = 0.05
    cbr = 0.03
    scenario = pd.DataFrame(data=[["homenode", pop]], columns=["name", "population"])
    parameters = get_default_parameters() | {
        "seed": 123,
        "nticks": nticks,
        "verbose": False,
        "beta": beta,
        "cbr": cbr,
        "inf_mean": 5,
    }

    model = Model(scenario, parameters)
    model.components = [Births_ConstantPop, Susceptibility, Infection, Transmission]
    seed_infections_randomly_SI(model, ninfections=10)
    model.run()

    assert_model_sanity(model)
    assert model.patches.cases_test[-1, 0] > 0, "Infections should persist with demographic turnover"


@pytest.mark.modeltest
@pytest.mark.xfail(reason="Known issue not yet fixed: Apparent unsigned integer underflow in model counters", strict=True)
def test_sis_model_short():
    pop = int(1e5)
    nticks = 500
    beta = 0.05
    inf_mean = 10
    scenario = pd.DataFrame(data=[["homenode", pop]], columns=["name", "population"])
    parameters = PropertySet({"seed": 99, "nticks": nticks, "verbose": False, "beta": beta, "inf_mean": inf_mean})

    model = Model(scenario, parameters)
    model.components = [Susceptibility, Infection_SIS, Transmission]
    seed_infections_randomly(model, ninfections=50)
    model.run()

    assert_model_sanity(model)
    I_counts = model.patches.cases_test[:, 0]
    assert np.any(I_counts[1:] > I_counts[0]), "Infections should initially rise"
    assert I_counts[-1] > 0, "SIS should maintain nonzero infections"


@pytest.mark.modeltest
@pytest.mark.xfail(reason="Known issue not yet fixed: AssertionError: No transmission occurred", strict=True)
def test_routine_immunization_blocks_spread():
    pop = int(1e5)
    nticks = 365 * 2
    parameters = get_default_parameters() | {
        "seed": 321,
        "nticks": nticks,
        "beta": 0.05,
        "cbr": 0.03,
        "inf_mean": 5,
    }
    scenario = pd.DataFrame(data=[["homenode", pop]], columns=["name", "population"])
    model = Model(scenario, parameters)
    model.components = [
        Births_ConstantPop,
        Susceptibility,
        Exposure,
        Infection,
        Transmission,
        lambda m, v: RoutineImmunization(m, period=365, coverage=0.9, age=365, verbose=v),
    ]
    seed_infections_randomly_SI(model, ninfections=10)
    model.run()

    assert_model_sanity(model)
    assert model.patches.cases_test[-1, 0] < 0.5 * pop, "Immunization should suppress large outbreaks"


@pytest.mark.modeltest
@pytest.mark.xfail(reason="Known issue not yet fixed: AssertionError: Cases not consistent with incidence", strict=True)
def test_mobility_spreads_infection_across_nodes():
    """
    Test that infections spread from one patch to another via mobility network.
    """
    pop = 50000
    nticks = 180
    beta = 0.05

    # Two patches, connected by a simple symmetric mobility matrix
    scenario = pd.DataFrame(
        {
            "name": ["node0", "node1"],
            "population": [pop, pop],
            "longitude": [0.0, 1.0],
            "latitude": [0.0, 0.0],
        }
    )

    parameters = get_default_parameters() | {"seed": 42, "nticks": nticks, "verbose": False, "beta": beta, "inf_mean": 5, "inf_sigma": 2}

    model = Model(scenario, parameters)

    # Define uniform bidirectional mobility between the two nodes
    mobility_matrix = np.array([[0.95, 0.05], [0.05, 0.95]])
    model.patches.network = mobility_matrix

    model.components = [
        Susceptibility,
        Exposure,
        Infection,
        Transmission,
    ]

    # Infect node 0 only
    seed_infections_in_patch(model, ninfections=10, ipatch=0)
    model.run()

    assert_model_sanity(model)

    # Confirm node 1 has some non-zero infections by end
    I_node1 = model.patches.cases_test[:, 1]
    assert I_node1[-1] > 0, "Infection should spread to node 1 via mobility network"


@pytest.mark.modeltest
@pytest.mark.xfail(
    reason="Known issue not yet fixed: AttributeError: 'LaserFrame' object has no attribute 'etimer'. Issue #24.", strict=True
)
def test_births_only_maintain_population_stability():
    """
    Confirm that Births_ConstantPop maintains stable population when transmission is disabled.
    """
    pop = int(1e5)
    nticks = 365 * 3  # 3 years
    cbr = 0.03  # Crude birth rate

    parameters = get_default_parameters() | {
        "seed": 888,
        "nticks": nticks,
        "beta": 0.0,  # No transmission
        "cbr": cbr,
        "verbose": False,
    }

    scenario = pd.DataFrame(
        {
            "name": ["home"],
            "population": [pop],
        }
    )

    model = Model(scenario, parameters)
    model.components = [
        Births_ConstantPop,
        Susceptibility,
    ]

    model.run()

    populations = model.patches.populations[:, 0]
    assert np.all(populations == pop), "Population changed under Births_ConstantPop when it shouldn't"


@pytest.mark.modeltest
@pytest.mark.xfail(reason="Known issue not yet fixed: AssertionError: No transmission occurred", strict=True)
def test_biweekly_scalar_modulates_transmission():
    """
    Ensure that biweekly_beta_scalar modulates transmission rate as expected.
    """

    pop = int(1e5)
    nticks = 364  # One year with 26 biweekly periods
    base_beta = 0.05

    # Start with all 1.0
    base_params = get_default_parameters() | {
        "seed": 999,
        "nticks": nticks,
        "beta": base_beta,
        "inf_mean": 5,
        "biweekly_beta_scalar": [1.0] * 26,
        "verbose": False,
    }

    # Perturbed version with lower transmission in first half of year
    low_transmission = PropertySet(base_params)
    low_transmission["biweekly_beta_scalar"][:13] = [0.5] * 13

    scenario = pd.DataFrame(
        {
            "name": ["home"],
            "population": [pop],
        }
    )

    # Baseline model
    model1 = Model(scenario, base_params)
    model1.components = [Susceptibility, Exposure, Infection, Transmission]
    seed_infections_randomly_SI(model1, ninfections=10)
    model1.run()
    assert_model_sanity(model1)

    # Perturbed model
    model2 = Model(scenario, low_transmission)
    model2.components = [Susceptibility, Exposure, Infection, Transmission]
    seed_infections_randomly_SI(model2, ninfections=10)
    model2.run()
    assert_model_sanity(model2)

    # Compare cumulative infections
    total1 = model1.patches.cases_test[-1, 0]
    total2 = model2.patches.cases_test[-1, 0]

    assert total2 < total1, "Reduced beta scalar should lower cumulative infections"


@pytest.mark.modeltest
def test_births_base_runs_minimally():
    """
    Ensure the base Births class integrates without crashing.
    Does not assert on population dynamics, only integration.
    """
    pop = 1000
    nticks = 10

    scenario = pd.DataFrame(
        {
            "name": ["home"],
            "population": [pop],
        }
    )

    params = get_default_parameters() | {
        "nticks": nticks,
        "seed": 42,
        "verbose": False,
    }

    model = Model(scenario, params)
    model.components = [
        Births,
        Susceptibility,
    ]

    try:
        model.run()
    except Exception as e:
        pytest.fail(f"Births base class failed during run: {e}")


@pytest.mark.modeltest
@pytest.mark.xfail(
    reason="Known issue not yet fixed: AttributeError: 'LaserFrame' object has no attribute 'etimer'. Issue #24.", strict=True
)
def test_births_variable_birthrate_maintains_population():
    """
    Ensure that Births_ConstantPop_VariableBirthRate maintains population size over time.
    """
    pop = 10000
    nticks = 100
    scenario = pd.DataFrame(
        {
            "name": ["home"],
            "population": [pop],
        }
    )

    params = get_default_parameters() | {
        "nticks": nticks,
        "seed": 123,
        "verbose": False,
        "cbr": {
            "rates": [0.02],  # constant birth rate
            "timesteps": [0],  # start at tick 0
        },
    }

    model = Model(scenario, params)
    model.components = [
        Births_ConstantPop_VariableBirthRate,
        Susceptibility,
    ]

    model.run()

    final_pop = model.patches.populations[-1, 0]
    pop_diff = abs(final_pop - pop)

    # Allow minor fluctuations
    assert pop_diff < pop * 0.01, f"Population drifted too far: {pop_diff}"


@pytest.mark.modeltest
@pytest.mark.xfail(reason="Known issue not yet fixed: AssertionError: Routine immunization should reduce infections", strict=True)
def test_routine_immunization_blocks_spread_compare(stable_transmission_model):
    """
    Routine immunization at high coverage should suppress outbreak spread.
    """
    pop = 100_000
    nticks = 365 * 5
    beta = 0.05
    cbr = 0.03

    scenario = pd.DataFrame({"name": ["home"], "population": [pop]})
    base_params = get_default_parameters() | {
        "nticks": nticks,
        "seed": 777,
        "beta": beta,
        "cbr": cbr,
        "inf_mean": 5,
    }

    model1 = stable_transmission_model
    model1.run()
    total_cases1 = model1.patches.cases_test[-1, 0]

    # With routine immunization
    model2 = Model(scenario, base_params)
    model2.components = [
        Births_ConstantPop,
        Susceptibility,
        Exposure,
        Infection,
        Transmission,
        lambda m, v: RoutineImmunization(m, period=365, coverage=0.9, age=365, verbose=v),
    ]
    seed_infections_randomly_SI(model2, ninfections=100)
    model2.run()
    total_cases2 = model2.patches.cases_test[-1, 0]

    assert total_cases2 < total_cases1 * 0.5, "Routine immunization should reduce infections"


@pytest.mark.skip(reason="Test needs fix")
@pytest.mark.modeltest
def test_immunization_campaign_temporarily_blocks_spread(stable_transmission_model):
    """
    ImmunizationCampaign should reduce infections during its active window.
    """
    model1 = stable_transmission_model
    model1.run()
    cases1 = model1.patches.cases_test[:, 0]

    # Add campaign to a copy of the model
    model2 = baseline_model()
    campaign = ImmunizationCampaign(
        model2,
        period=1,  # Apply daily
        coverage=0.9,  # High coverage
        age_lower=0,
        age_upper=5 * 365,  # Target ages 0-5 years
        start=100,
        end=120,
        verbose=model2.params.verbose,
    )
    # TODO - this doesn't work because the Model class expects _classes_ not _instances_
    model2.components = model2.components.append(campaign)
    model2.run()
    cases2 = model2.patches.cases_test[:, 0]

    mean_cases_no_campaign = np.mean(cases1[100:121])
    mean_cases_with_campaign = np.mean(cases2[100:121])
    print(f"Mean cases (no campaign): {mean_cases_no_campaign}")
    print(f"Mean cases (with campaign): {mean_cases_with_campaign}")

    assert mean_cases_with_campaign < mean_cases_no_campaign, "Campaign should reduce infections during its window"


@pytest.mark.modeltest
def test_importation_keeps_infection_alive():
    """
    Importation events should sustain transmission over long timeframes.

    This test builds from a stable transmission model, but delays seeding,
    relying on Infect_Random_Agents to initiate and maintain infections.
    """
    pop = int(1e5)
    nticks = 365 * 5
    params = get_default_parameters() | {
        "nticks": nticks,
        "seed": 123,
        "beta": 0.35,
        "inf_mean": 5,
        "cbr": 0.03,
        "importation_period": 15,  # More frequent than original test
        "importation_count": 10,
        "importation_start": 10,
        "importation_end": nticks,
        "verbose": False,
    }

    scenario = pd.DataFrame({"name": ["home"], "population": [pop]})
    model = Model(scenario, params)
    model.components = [
        Births_ConstantPop,
        Susceptibility,
        Exposure,
        Infection,
        Transmission,
        Infect_Random_Agents,
    ]

    # No initial infections — let importation trigger and sustain
    model.run()

    incidence = model.patches.incidence[:, 0]
    total_inc = np.sum(incidence)
    assert total_inc > 0, "Importation should trigger infections"
    # Obviously we could just do the > 1000 test but we want to see an explicit error if the >0 test fails.
    assert total_inc > 1000  # e.g., 1000 cases over 5 years


@pytest.mark.modeltest
@pytest.mark.xfail(reason="Known issue not yet fixed: Failed: Patch 0 should remain uninfected", strict=True)
def test_targeted_importation_hits_correct_patch():
    """
    Infect_Agents_In_Patch should import cases only into the specified patch.
    """
    nticks = 365
    pop = 100_000
    scenario = pd.DataFrame(
        {
            "name": ["patch0", "patch1"],
            "nodeid": [0, 1],
            "population": [pop, pop],
        }
    )

    params = get_default_parameters() | {
        "nticks": nticks,
        "seed": 222,
        "beta": 0.05,
        "cbr": 0.03,
        "inf_mean": 5,
        "importation_period": 30,
        "importation_count": 5,
        "importation_start": 0,
        "importation_end": nticks,
        "importation_target": 1,
        "verbose": False,
    }

    model = Model(scenario, params)
    importation = Infect_Agents_In_Patch(model, verbose=params.verbose)  # Inject into patch 1
    model.components = [Births_ConstantPop, Susceptibility, Exposure, Infection, Transmission, importation]
    # seed_infections_in_patch(model, ipatch=1, ninfections=1)
    model.run()

    cases_patch0 = model.patches.cases_test[:, 0]
    cases_patch1 = model.patches.cases_test[:, 1]

    # assert np.sum(cases_patch1) > 0, "Patch 1 should receive infections"
    # assert np.sum(cases_patch0) == 0, "Patch 0 should remain uninfected"
    if not (np.sum(cases_patch1) > 0):
        print("🚨 Patch 1 cases:\n", cases_patch1)
        print("🚨 Total infections in Patch 1:", np.sum(cases_patch1))
        pytest.fail("Patch 1 should receive infections")

    if not (np.sum(cases_patch0) == 0):
        print("🚨 Patch 0 cases:\n", cases_patch0)
        print("🚨 Total infections in Patch 0:", np.sum(cases_patch0))
        print("🚨 Full incidence matrix:\n", model.patches.incidence)
        print("🚨 Parameters:\n", params)
        print("🚨 Population state summary:")
        print("  nodeid counts:", np.bincount(model.population.nodeid[: model.population.count]))
        print("  infected states:", np.sum(model.population.state[: model.population.count] == 2))
        pytest.fail("Patch 0 should remain uninfected")


@pytest.mark.modeltest
@pytest.mark.xfail(reason="Known issue not yet fixed. Failed: Outputs of both transmissions differ significantly.", strict=True)
def test_transmission_sir_behaves_like_transmission():
    """
    TransmissionSIR should behave similarly to the standard Transmission class.

    This test currently fails in the last phase. Neither model1 nor model2 curves seem to make sense.
    """
    pop = 100_000
    nticks = 365
    scenario = pd.DataFrame({"name": ["home"], "population": [pop]})

    params = get_default_parameters() | {
        "nticks": nticks,
        "seed": 777,
        "beta": 0.04,
        "inf_mean": 5,
        "verbose": False,
    }

    # Standard Transmission
    model1 = Model(scenario, params)
    model1.components = [Susceptibility, Infection, Transmission]
    seed_infections_randomly(model1, ninfections=5)
    model1.run()
    cases1 = model1.patches.cases_test[:, 0]

    # TransmissionSIR
    model2 = Model(scenario, params)
    model2.components = [Susceptibility, Infection, TransmissionSIR]
    seed_infections_randomly(model2, ninfections=5)
    model2.run()
    cases2 = model2.patches.cases_test[:, 0]

    assert np.any(cases1 > cases1[0]), "Standard transmission should spread"
    assert np.any(cases2 > cases2[0]), "TransmissionSIR should spread"

    # Optional: ensure they're similar (within ~20%)
    diff = np.abs(cases1 - cases2)
    # assert np.max(diff) < 0.2 * pop, "Outputs of both transmissions should be similar"

    if np.max(diff) >= 0.2 * pop:
        print("Max difference:", np.max(diff))
        print("Final case count (standard):", cases1[-1])
        print("Final case count (SIR):", cases2[-1])
        print("Cumulative difference:", np.sum(diff))

        plt.figure(figsize=(10, 6))
        plt.plot(cases1, label="Standard Transmission")
        plt.plot(cases2, label="TransmissionSIR", linestyle="--")
        plt.title("Comparison of Transmission vs TransmissionSIR")
        plt.xlabel("Tick")
        plt.ylabel("Cases")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("test_transmission_comparison.png")  # Save for inspection

        pytest.fail("Outputs of both transmissions differ significantly. See 'test_transmission_comparison.png'")


def test_stable_transmission_model_runs(stable_transmission_model):
    model = stable_transmission_model
    # print("Initial itimer > 0:", np.sum(model.population.itimer[: model.population.count] > 0))
    # print("Initial etimer > 0:", np.sum(model.population.etimer[: model.population.count] > 0))
    # print("Initial susceptible count:", np.sum(model.population.susceptibility[: model.population.count] > 0))
    model.run()

    # Sanity checks
    inc = model.patches.incidence[:, 0]
    total_inc = inc.sum()

    print(f"Total incidence: {total_inc}")
    assert total_inc > 0, "Model should produce ongoing transmission"

    # final_cases = model.patches.cases_test[-1, 0]
    # assert final_cases > 0, "Final case count should be non-zero"

    max_cases = np.max(model.patches.cases_test[:, 0])
    assert max_cases > 0, "At least some infections should occur"
