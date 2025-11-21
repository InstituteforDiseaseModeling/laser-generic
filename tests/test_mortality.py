import unittest

import laser.core.distributions as dists
import numpy as np
from laser.core import PropertySet
from laser.core.random import seed as set_seed

from laser.generic import SEIR
from laser.generic import Model
from laser.generic.newutils import ValuesMap
from laser.generic.vitaldynamics import MortalityByCDR
from utils import stdgrid

# Shared test parameters
NTICKS = 3650  # 10 years
SEED = 271828


def create_seir_scenario_with_mortality(cdr=20.0):
    """Create a scenario with S/E/I/R populations and mortality."""
    scenario = stdgrid(M=1, N=1, population_fn=lambda r, c: 100_000)

    scenario["E"] = (scenario.population * 0.125).astype(np.int32)
    scenario["I"] = (scenario.population * 0.125).astype(np.int32)
    scenario["R"] = (scenario.population * 0.375).astype(np.int32)
    scenario["S"] = (scenario.population - (scenario.E + scenario.I + scenario.R)).astype(np.int32)

    parameters = PropertySet({"nticks": NTICKS})
    mortalityrates = ValuesMap.from_scalar(cdr, 1, NTICKS).values

    expdurdist = dists.normal(loc=30.0, scale=3.0)
    infdurdist = dists.normal(loc=30.0, scale=5.0)

    model = Model(scenario, parameters, birthrates=None)
    model.components = [
        SEIR.Susceptible(model),
        SEIR.Exposed(model, expdurdist, infdurdist),
        SEIR.Infectious(model, infdurdist),
        SEIR.Recovered(model),
        MortalityByCDR(model, mortalityrates),
    ]

    return model, cdr


def calculate_observed_cdr(model):
    """Calculate observed CDR from model results."""
    N = model.nodes.S + model.nodes.E + model.nodes.I + model.nodes.R
    starts = np.array(range(0, NTICKS, 365), dtype=np.int32)
    ends = starts + 364
    mortality = (1000 * (N[starts] - N[ends]) / N[starts]).mean(axis=0)[0]
    return mortality


class TestMortalityByCDR(unittest.TestCase):
    def setUp(self):
        """Set random seed for reproducibility."""
        set_seed(SEED)
        return

    def test_cdr_2(self):
        """Test mortality with CDR=2 per 1000 per year."""
        cdr = 2.0
        model, _ = create_seir_scenario_with_mortality(cdr)
        pop_start = model.people.count

        model.run()

        pop_finish = model.people.count
        observed_cdr = calculate_observed_cdr(model)

        # Population should have decreased
        assert pop_finish < pop_start, "Population should decrease due to mortality"

        # Observed CDR should be close to target CDR (within 10% tolerance)
        percent_diff = abs((observed_cdr - cdr) / cdr * 100)
        assert percent_diff < 10.0, f"Observed CDR {observed_cdr:.2f} deviated by {percent_diff:.2f}% from target CDR {cdr}"

        # Verify deaths were recorded
        total_deaths = model.nodes.deaths.sum()
        assert total_deaths > 0, "Deaths should have been recorded"
        assert total_deaths == pop_start - pop_finish, "Recorded deaths should match population decrease"

        return

    def test_cdr_10(self):
        """Test mortality with CDR=10 per 1000 per year."""
        cdr = 10.0
        model, _ = create_seir_scenario_with_mortality(cdr)
        pop_start = model.people.count

        model.run()

        pop_finish = model.people.count
        observed_cdr = calculate_observed_cdr(model)

        # Population should have decreased
        assert pop_finish < pop_start, "Population should decrease due to mortality"

        # Observed CDR should be close to target CDR (within 10% tolerance)
        percent_diff = abs((observed_cdr - cdr) / cdr * 100)
        assert percent_diff < 10.0, f"Observed CDR {observed_cdr:.2f} deviated by {percent_diff:.2f}% from target CDR {cdr}"

        # Verify deaths were recorded
        total_deaths = model.nodes.deaths.sum()
        assert total_deaths > 0, "Deaths should have been recorded"
        assert total_deaths == pop_start - pop_finish, "Recorded deaths should match population decrease"

        return

    def test_cdr_20(self):
        """Test mortality with CDR=20 per 1000 per year."""
        cdr = 20.0
        model, _ = create_seir_scenario_with_mortality(cdr)
        pop_start = model.people.count

        model.run()

        pop_finish = model.people.count
        observed_cdr = calculate_observed_cdr(model)

        # Population should have decreased
        assert pop_finish < pop_start, "Population should decrease due to mortality"

        # Observed CDR should be close to target CDR (within 10% tolerance)
        percent_diff = abs((observed_cdr - cdr) / cdr * 100)
        assert percent_diff < 10.0, f"Observed CDR {observed_cdr:.2f} deviated by {percent_diff:.2f}% from target CDR {cdr}"

        # Verify deaths were recorded
        total_deaths = model.nodes.deaths.sum()
        assert total_deaths > 0, "Deaths should have been recorded"
        assert total_deaths == pop_start - pop_finish, "Recorded deaths should match population decrease"

        return

    def test_cdr_40(self):
        """Test mortality with CDR=40 per 1000 per year."""
        cdr = 40.0
        model, _ = create_seir_scenario_with_mortality(cdr)
        pop_start = model.people.count

        model.run()

        pop_finish = model.people.count
        observed_cdr = calculate_observed_cdr(model)

        # Population should have decreased
        assert pop_finish < pop_start, "Population should decrease due to mortality"

        # Observed CDR should be close to target CDR (within 10% tolerance)
        percent_diff = abs((observed_cdr - cdr) / cdr * 100)
        assert percent_diff < 10.0, f"Observed CDR {observed_cdr:.2f} deviated by {percent_diff:.2f}% from target CDR {cdr}"

        # Verify deaths were recorded
        total_deaths = model.nodes.deaths.sum()
        assert total_deaths > 0, "Deaths should have been recorded"
        assert total_deaths == pop_start - pop_finish, "Recorded deaths should match population decrease"

        return


if __name__ == "__main__":
    unittest.main()
