import itertools

import numpy as np
import pandas as pd
import pytest
from laser_core import PropertySet

from laser_generic import Births_ConstantPop, Model, Susceptibility, Exposure, Transmission
from laser_generic.infection import Infection, Infection_SIS
from laser_generic.utils import (
    seed_infections_randomly,
    seed_infections_randomly_SI,
    set_initial_susceptibility_randomly,
    get_default_parameters,
)

def assert_model_sanity(model):
    I = model.patches.cases_test[-1:, 0]
    S = model.patches.susceptibility_test[-1:, 0]
    N = model.patches.populations[:-1, 0]
    inc = model.patches.incidence[:, 0]

    assert np.sum(inc) > 0, "No transmission occurred"
    assert np.any(S < S[0]), "Susceptibles never decreased"
    assert np.all(S >= 0), "Negative susceptible count"
    assert np.all(S <= N), "Susceptibles exceed population"

    #I_derived = np.cumsum(inc[:-1]) + I[0]
    #assert np.allclose(I[1:], I_derived, atol=1e-5), "Cases not consistent with incidence"
    """
    if len(I) == len(inc):
        I_derived = np.cumsum(inc[:-1]) + I[0]
        if len(I_derived) == len(I) - 1:
            assert np.allclose(I[1:], I_derived, atol=1e-5), "Cases not consistent with incidence"
    """

@pytest.mark.modeltest
def test_si_model_nobirths_flow():
    nticks = 180
    pop = 1e5
    scenario = pd.DataFrame(data=[["homenode", pop, "0,0"]], columns=["name", "population", "location"])
    parameters = PropertySet({"seed": 42, "nticks": nticks, "verbose": False, "beta": 0.01})

    model = Model(scenario, parameters)
    model.components = [Susceptibility, Transmission]
    seed_infections_randomly_SI(model, ninfections=1)
    model.run()

    assert_model_sanity(model)
    assert model.patches.cases_test[-1, 0] > model.patches.cases_test[0, 0], "Infection count should increase"

@pytest.mark.modeltest
def test_sir_nobirths_short():
    pop = 1e5
    nticks = 365
    beta = 0.05
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
def test_si_model_with_births_short():
    pop = 1e5
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
def test_sei_model_with_births_short():
    pop = 1e5
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
    model.components = [Births_ConstantPop, Susceptibility, Exposure, Infection, Transmission]
    seed_infections_randomly_SI(model, ninfections=10)
    model.run()

    assert_model_sanity(model)
    assert model.patches.cases_test[-1, 0] > 0, "Infections should persist with demographic turnover"

@pytest.mark.modeltest
def test_sis_model_short():
    pop = 1e5
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
    I = model.patches.cases_test[:, 0]
    assert np.any(I[1:] > I[0]), "Infections should initially rise"
    assert I[-1] > 0, "SIS should maintain nonzero infections"
