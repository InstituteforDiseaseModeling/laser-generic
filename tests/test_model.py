import itertools

import numpy as np
import pandas as pd
import pytest
from laser_core import PropertySet


from laser_generic import Births_ConstantPop
from laser_generic import Model
from laser_generic import Susceptibility
from laser_generic import Transmission
from laser_generic.infection import Infection
from laser_generic.infection import Infection_SIS
from laser_generic.utils import seed_infections_randomly
from laser_generic.utils import seed_infections_randomly_SI
from laser_generic.utils import set_initial_susceptibility_randomly

@pytest.mark.modeltest
def test_si_model_nobirths_census():
    """
    Quick test that SI model without births produces epidemic growth.
    """
    nticks = 180
    t = np.arange(nticks)
    pop = 1e5

    scenario = pd.DataFrame(data=[["homenode", pop, "0,0"]], columns=["name", "population", "location"])
    parameters = PropertySet({"seed": 42, "nticks": nticks, "verbose": False, "beta": 0.01})
    model = Model(scenario, parameters)
    model.components = [
        Susceptibility,
        Transmission,
    ]
    seed_infections_randomly_SI(model, ninfections=1)
    model.run()

    cases = model.patches.cases[:, 0]

    # Assert infections grow
    assert cases[-1] > cases[0], "Infection count should increase in SI model"

@pytest.mark.modeltest
def test_si_model_nobirths_flow():
    """
    Quick test that SI model without births produces epidemic growth.
    """
    nticks = 180
    t = np.arange(nticks)
    pop = 1e5

    scenario = pd.DataFrame(data=[["homenode", pop, "0,0"]], columns=["name", "population", "location"])
    parameters = PropertySet({"seed": 42, "nticks": nticks, "verbose": False, "beta": 0.01})
    model = Model(scenario, parameters)
    model.components = [
        Susceptibility,
        Transmission,
    ]
    seed_infections_randomly_SI(model, ninfections=1)
    model.run()

    cases = model.patches.cases_test[:, 0]

    # Assert infections grow
    assert cases[-1] > cases[0], "Infection count should increase in SI model"

@pytest.mark.modeltest
def test_sir_nobirths_short():
    """
    Quick test that SIR model shows recovery and drop in infections.
    """
    pop = 1e5
    nticks = 365
    beta = 0.05
    gamma = 1 / 20

    scenario = pd.DataFrame(data=[["homenode", pop]], columns=["name", "population"])
    parameters = PropertySet({"seed": 1, "nticks": nticks, "verbose": False, "beta": beta, "inf_mean": 1 / gamma})

    model = Model(scenario, parameters)
    model.components = [
        Susceptibility,
        Infection,
        Transmission,
    ]
    seed_infections_randomly(model, ninfections=50)
    model.run()

    peak_I = np.max(model.patches.cases_test[:, 0])
    final_I = model.patches.cases_test[-1, 0]

    assert final_I < peak_I, "SIR model should recover (cases should drop)"

@pytest.mark.modeltest
def test_si_model_with_births_short():
    """
    Quick test that SI model with births shows continued transmission.
    """
    pop = 1e5
    nticks = 365 * 2
    beta = 0.02
    cbr = 0.03  # 3% crude birth rate

    scenario = pd.DataFrame(data=[["homenode", pop]], columns=["name", "population"])
    parameters = PropertySet({
        "seed": 123,
        "nticks": nticks,
        "verbose": False,
        "beta": beta,
        "cbr": cbr,
    })

    model = Model(scenario, parameters)
    model.components = [
        Births_ConstantPop,
        Susceptibility,
        Transmission,
    ]
    seed_infections_randomly_SI(model, ninfections=10)
    model.run()

    final_I = model.patches.cases_test[-1, 0]
    assert final_I > 0, "Infections should persist with demographic turnover"

@pytest.mark.modeltest
def test_sis_model_short():
    """
    Quick test that SIS model supports reinfection and dynamic equilibrium.
    """
    pop = 1e5
    nticks = 500
    beta = 0.05
    inf_mean = 10

    scenario = pd.DataFrame(data=[["homenode", pop]], columns=["name", "population"])
    parameters = PropertySet({
        "seed": 99,
        "nticks": nticks,
        "verbose": False,
        "beta": beta,
        "inf_mean": inf_mean,
    })

    model = Model(scenario, parameters)
    model.components = [
        Susceptibility,
        Infection_SIS,
        Transmission,
    ]
    seed_infections_randomly(model, ninfections=50)
    model.run()

    I = model.patches.cases_test[:, 0]
    assert np.any(I[1:] > I[0]), "Infections should initially rise"
    assert I[-1] > 0, "SIS should maintain nonzero infections"
