import unittest
from pathlib import Path

import laser.core.distributions as dists
import numpy as np
from laser.core import PropertySet
from laser.core.demographics import AliasedDistribution
from laser.core.random import seed as set_seed

from laser.generic import SEIR
from laser.generic import Model
from laser.generic.newutils import ValuesMap
from laser.generic.vitaldynamics import BirthsByCBR
from utils import stdgrid

# Shared test parameters
NTICKS = 3650  # 10 years
SEED = 42


def load_age_distribution():
    """Load Nigeria age distribution for birth age assignment."""
    age_data_path = Path(__file__).parent.parent / "docs/tutorials/notebooks/Nigeria-Distribution-2020.csv"
    age_data = np.loadtxt(age_data_path, delimiter=",", usecols=0)[0:89]
    return AliasedDistribution(age_data)


def create_basic_scenario_susceptible_only(cbr=20.0):
    """Create a scenario with only Susceptible population."""
    scenario = stdgrid(M=1, N=1, population_fn=lambda r, c: 20_000)
    scenario["S"] = scenario.population
    scenario["E"] = scenario["I"] = scenario["R"] = 0

    parameters = PropertySet({"nticks": NTICKS})
    birthrates = ValuesMap.from_scalar(cbr, 1, NTICKS).values
    pyramid = load_age_distribution()

    model = Model(scenario, parameters, birthrates=birthrates)
    model.components = [
        SEIR.Susceptible(model),
        BirthsByCBR(model, birthrates, pyramid),
    ]

    return model, cbr


def create_equilibrium_seir_scenario(cbr=20.0):
    """Create a scenario with equilibrium S/E/I/R populations."""
    scenario = stdgrid(M=1, N=1, population_fn=lambda r, c: 20_000)

    # Set up equilibrium-like populations
    scenario["S"] = 15_000
    scenario["E"] = 2_000
    scenario["I"] = 1_500
    scenario["R"] = 1_500

    parameters = PropertySet({"nticks": NTICKS})
    birthrates = ValuesMap.from_scalar(cbr, 1, NTICKS).values
    pyramid = load_age_distribution()

    expdurdist = dists.normal(loc=5.0, scale=1.0)
    infdurdist = dists.normal(loc=7.0, scale=2.0)

    model = Model(scenario, parameters, birthrates=birthrates)
    model.components = [
        SEIR.Susceptible(model),
        SEIR.Exposed(model, expdurdist, infdurdist),
        SEIR.Infectious(model, infdurdist),
        SEIR.Recovered(model),
        BirthsByCBR(model, birthrates, pyramid),
    ]

    return model, cbr


def create_scenario_with_additional_states(cbr=20.0):
    """Create a scenario with custom state configuration (testing with fewer initial states)."""
    scenario = stdgrid(M=1, N=1, population_fn=lambda r, c: 20_000)

    # Initialize with only S and E states (no I or R initially)
    scenario["S"] = 18_000
    scenario["E"] = 2_000
    scenario["I"] = 0
    scenario["R"] = 0

    parameters = PropertySet({"nticks": NTICKS})
    birthrates = ValuesMap.from_scalar(cbr, 1, NTICKS).values
    pyramid = load_age_distribution()

    expdurdist = dists.normal(loc=5.0, scale=1.0)
    infdurdist = dists.normal(loc=7.0, scale=2.0)

    model = Model(scenario, parameters, birthrates=birthrates)

    # Include all SEIR components
    model.components = [
        SEIR.Susceptible(model),
        SEIR.Exposed(model, expdurdist, infdurdist),
        SEIR.Infectious(model, infdurdist),
        SEIR.Recovered(model),
        BirthsByCBR(model, birthrates, pyramid),
    ]

    return model, cbr


def create_multi_node_scenario(cbr=20.0):
    """Create a multi-node scenario with S/E/I/R populations."""
    scenario = stdgrid(M=2, N=2, population_fn=lambda r, c: 50_000)

    # Initialize all nodes with similar proportions
    total_pop = scenario.population
    scenario["S"] = (total_pop * 0.70).astype(np.int32)
    scenario["E"] = (total_pop * 0.10).astype(np.int32)
    scenario["I"] = (total_pop * 0.10).astype(np.int32)
    scenario["R"] = (total_pop * 0.10).astype(np.int32)

    parameters = PropertySet({"nticks": NTICKS})
    birthrates = ValuesMap.from_scalar(cbr, 4, NTICKS).values
    pyramid = load_age_distribution()

    expdurdist = dists.normal(loc=5.0, scale=1.0)
    infdurdist = dists.normal(loc=7.0, scale=2.0)

    model = Model(scenario, parameters, birthrates=birthrates)
    model.components = [
        SEIR.Susceptible(model),
        SEIR.Exposed(model, expdurdist, infdurdist),
        SEIR.Infectious(model, infdurdist),
        SEIR.Recovered(model),
        BirthsByCBR(model, birthrates, pyramid),
    ]

    return model, cbr


def create_spatially_varying_cbr_scenario():
    """Create an 8-node scenario with spatially-varying birth rates."""
    scenario = stdgrid(M=2, N=4, population_fn=lambda r, c: 100_000)

    scenario["S"] = scenario.population
    scenario["E"] = scenario["I"] = scenario["R"] = 0

    parameters = PropertySet({"nticks": NTICKS})

    # Different CBR for each node (ranging from rural to urban)
    cbrs = np.array([35.0, 30.0, 25.0, 20.0, 18.0, 16.0, 14.0, 12.0])
    birthrates = ValuesMap.from_nodes(cbrs, NTICKS).values
    pyramid = load_age_distribution()

    model = Model(scenario, parameters, birthrates=birthrates)
    model.components = [
        SEIR.Susceptible(model),
        BirthsByCBR(model, birthrates, pyramid),
    ]

    return model, cbrs


def create_time_varying_cbr_scenario():
    """Create a scenario with time-varying birth rates over 10 years."""
    scenario = stdgrid(M=1, N=1, population_fn=lambda r, c: 100_000)

    scenario["S"] = scenario.population
    scenario["E"] = scenario["I"] = scenario["R"] = 0

    parameters = PropertySet({"nticks": NTICKS})

    # CBR decreases from 30 to 10 over the simulation period
    time_varying_cbr = np.linspace(30.0, 10.0, NTICKS)
    birthrates = ValuesMap.from_timeseries(time_varying_cbr, 1).values
    pyramid = load_age_distribution()

    model = Model(scenario, parameters, birthrates=birthrates)
    model.components = [
        SEIR.Susceptible(model),
        BirthsByCBR(model, birthrates, pyramid),
    ]

    return model, time_varying_cbr


def calculate_expected_population(pop_start, cbr, nticks):
    """Calculate expected population after growth with constant CBR."""
    growth_factor = (1.0 + cbr / 1000.0) ** (nticks / 365.0)
    return int(pop_start * growth_factor)


def calculate_expected_population_time_varying(pop_start, cbr_timeseries):
    """Calculate expected population with time-varying CBR."""
    expected_pop = pop_start
    for cbr in cbr_timeseries:
        growth = (1.0 + cbr / 1000.0) ** (1.0 / 365.0)
        expected_pop *= growth
    return int(expected_pop)


class TestBirthsByCBR(unittest.TestCase):
    def setUp(self):
        """Set random seed for reproducibility."""
        set_seed(SEED)

        return

    def test_susceptible_only(self):
        """Test case with only Susceptible population (1 node)."""
        # Scenario
        model, cbr = create_basic_scenario_susceptible_only()
        pop_start = model.people.count

        # Run
        model.run()
        pop_finish = model.people.count

        # Check
        expected_pop = calculate_expected_population(pop_start, cbr, NTICKS)
        difference = pop_finish - expected_pop
        percent_diff = abs(difference / expected_pop * 100)

        # Population should grow according to CBR within 1% tolerance
        assert percent_diff < 1.0, f"Population growth deviated by {percent_diff:.2f}% (expected: {expected_pop:,}, actual: {pop_finish:,})"

        # All agents should remain susceptible (no disease introduced)
        total_s = model.nodes.S[-1].sum()
        assert total_s == pop_finish, "All agents should be susceptible when no disease is introduced"

        return

    def test_equilibrium_seir(self):
        """Test case with equilibrium S/E/I/R populations (1 node)."""
        # Scenario
        model, cbr = create_equilibrium_seir_scenario()
        pop_start = model.people.count

        # Run
        model.run()
        pop_finish = model.people.count

        # Check
        expected_pop = calculate_expected_population(pop_start, cbr, NTICKS)
        difference = pop_finish - expected_pop
        percent_diff = abs(difference / expected_pop * 100)

        # Population should grow according to CBR within 10% tolerance
        # (Higher tolerance due to disease dynamics affecting population)
        assert percent_diff < 10.0, (
            f"Population growth deviated by {percent_diff:.2f}% (expected: {expected_pop:,}, actual: {pop_finish:,})"
        )

        # Verify population actually grew
        assert pop_finish > pop_start, "Population should have grown due to births"

        return

    def test_additional_states_custom(self):
        """Test case with custom state list (only S/E/I, no R) (1 node)."""
        # Scenario
        model, cbr = create_scenario_with_additional_states()
        pop_start = model.people.count

        # Run
        model.run()
        pop_finish = model.people.count

        # Check
        expected_pop = calculate_expected_population(pop_start, cbr, NTICKS)
        difference = pop_finish - expected_pop
        percent_diff = abs(difference / expected_pop * 100)

        # Population should grow according to CBR within 10% tolerance
        # (Higher tolerance due to disease dynamics affecting population)
        assert percent_diff < 10.0, (
            f"Population growth deviated by {percent_diff:.2f}% (expected: {expected_pop:,}, actual: {pop_finish:,})"
        )

        # Verify population actually grew
        assert pop_finish > pop_start, "Population should have grown due to births"

        return

    def test_multi_node_seir(self):
        """Test case with multiple nodes with S/E/I/R populations."""
        # Scenario
        model, cbr = create_multi_node_scenario()
        pop_start = model.people.count

        # Run
        model.run()
        pop_finish = model.people.count

        # Check
        expected_pop = calculate_expected_population(pop_start, cbr, NTICKS)
        difference = pop_finish - expected_pop
        percent_diff = abs(difference / expected_pop * 100)

        # Population should grow according to CBR within 10% tolerance
        # (Higher tolerance due to disease dynamics affecting population)
        assert percent_diff < 10.0, (
            f"Population growth deviated by {percent_diff:.2f}% (expected: {expected_pop:,}, actual: {pop_finish:,})"
        )

        # Each node should have positive population
        total_pop_per_node = model.nodes.S[-1] + model.nodes.E[-1] + model.nodes.I[-1] + model.nodes.R[-1]
        for node_idx, pop in enumerate(total_pop_per_node):
            assert pop > 0, f"Node {node_idx} should have positive population"

        return

    def test_spatially_varying_cbr(self):
        """Test case with spatially-varying birth rates across nodes (8 nodes)."""
        # Scenario
        model, cbrs = create_spatially_varying_cbr_scenario()

        # Run
        model.run()

        # Check
        total_pop = model.nodes.S

        # Verify each node grew according to its specific CBR
        for node_idx, cbr in enumerate(cbrs):
            pop_start = total_pop[0, node_idx]
            pop_finish = total_pop[-1, node_idx]
            expected_pop = calculate_expected_population(pop_start, cbr, NTICKS)

            difference = pop_finish - expected_pop
            percent_diff = abs(difference / expected_pop * 100)

            assert percent_diff < 1.0, (
                f"Node {node_idx} (CBR={cbr}) growth deviated by {percent_diff:.2f}% (expected: {expected_pop:,}, actual: {pop_finish:,})"
            )

        # Nodes with higher CBR should have grown more
        growth_rates = total_pop[-1] / total_pop[0]
        # Since cbrs are in descending order, growth rates should also descend
        for i in range(len(growth_rates) - 1):
            assert growth_rates[i] > growth_rates[i + 1], (
                f"Node {i} (CBR={cbrs[i]}) should have higher growth than Node {i + 1} (CBR={cbrs[i + 1]})"
            )

        return

    def test_time_varying_cbr(self):
        """Test case with time-varying birth rates (1 node over 10 years)."""
        # Scenario
        model, cbr_timeseries = create_time_varying_cbr_scenario()
        pop_start = model.people.count

        # Run
        model.run()
        pop_finish = model.people.count

        # Check
        expected_pop = calculate_expected_population_time_varying(pop_start, cbr_timeseries)
        difference = pop_finish - expected_pop
        percent_diff = abs(difference / expected_pop * 100)

        # Population should grow according to time-varying CBR within 1% tolerance
        assert percent_diff < 1.0, f"Population growth deviated by {percent_diff:.2f}% (expected: {expected_pop:,}, actual: {pop_finish:,})"

        # Growth rate should decrease over time (as CBR decreases)
        total_pop = model.nodes.S
        early_growth = total_pop[365] / total_pop[0]  # First year growth
        late_growth = total_pop[-1] / total_pop[-366]  # Last year growth

        assert early_growth > late_growth, "Early growth rate should be higher than late growth rate as CBR decreases"

        return


if __name__ == "__main__":
    unittest.main()
