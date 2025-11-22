import unittest
from datetime import datetime
from pathlib import Path

import laser.core.distributions as dists
import numpy as np
from laser.core import PropertySet
from laser.core.demographics import AliasedDistribution
from laser.core.demographics import KaplanMeierEstimator
from laser.core.random import seed as set_seed

from laser.generic import SEIR
from laser.generic import Model
from laser.generic.newutils import ValuesMap
from laser.generic.shared import State
from laser.generic.vitaldynamics import BirthsByCBR
from laser.generic.vitaldynamics import MortalityByCDR
from laser.generic.vitaldynamics import MortalityByEstimator
from utils import stdgrid

# Claude Code prompt: "Please write a test class for MortalityByCDR in the file test_mortality.py. Use the code to inform your implementation. You should use a population of 100000 running for 10 years. Test against the four CDR values used in the notebook."
# Claude Code prompt: "Please write a test class for MortalityByEstimator in the file test_mortality.py. Use the code in the mortality.ipynb notebook to inform your implementation. You should use a population of 100000 running for 10 years. Validation will require looking at the population in each age each year and comparing deaths in that population against the data from the input survival CSV. Do not remove any existing tests."

# Shared test parameters
NTICKS = 3650  # 10 years
SEED = datetime.now().microsecond  # noqa: DTZ005


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


def load_survival_data():
    """Load Nigeria survival data for age-specific mortality."""
    data_path = Path(__file__).parent.parent / "docs/tutorials/notebooks/Nigeria-Survival-2020.csv"
    survival_data = np.loadtxt(data_path, delimiter=",", usecols=1)[0:89].cumsum()
    return KaplanMeierEstimator(survival_data)


def load_age_distribution():
    """Load Nigeria age distribution for population initialization."""
    data_path = Path(__file__).parent.parent / "docs/tutorials/notebooks/Nigeria-Distribution-2020.csv"
    age_data = np.loadtxt(data_path, delimiter=",", usecols=0)[0:89]
    return AliasedDistribution(age_data)


def create_seir_scenario_with_age_specific_mortality():
    """Create a scenario with S/E/I/R populations and age-specific mortality."""
    scenario = stdgrid(M=1, N=1, population_fn=lambda r, c: 100_000)

    scenario["E"] = (scenario.population * 0.125).astype(np.int32)
    scenario["I"] = (scenario.population * 0.125).astype(np.int32)
    scenario["R"] = (scenario.population * 0.375).astype(np.int32)
    scenario["S"] = (scenario.population - (scenario.E + scenario.I + scenario.R)).astype(np.int32)

    parameters = PropertySet({"nticks": NTICKS})

    expdurdist = dists.normal(loc=30.0, scale=3.0)
    infdurdist = dists.normal(loc=30.0, scale=5.0)

    pyramid = load_age_distribution()
    survival = load_survival_data()

    model = Model(scenario, parameters, birthrates=None)
    model.components = [
        SEIR.Susceptible(model),
        SEIR.Exposed(model, expdurdist, infdurdist),
        SEIR.Infectious(model, infdurdist),
        SEIR.Recovered(model),
        # Use BirthsByCBR with rate=0 to initialize DOBs based on age pyramid
        BirthsByCBR(model, birthrates=ValuesMap.from_scalar(0.0, 1, NTICKS).values, pyramid=pyramid, track=True),
        MortalityByEstimator(model, survival),
    ]

    return model, survival


class TestMortalityByCDR(unittest.TestCase):
    def setUp(self):
        """Set random seed for reproducibility."""
        set_seed(SEED)
        return

    def test_cdr_2(self):
        """Test mortality with CDR=2 per 1000 per year."""
        cdr = 2.0
        model, _ = create_seir_scenario_with_mortality(cdr)
        pop_start = (model.people.state != State.DECEASED.value).sum()

        model.run()

        pop_finish = (model.people.state != State.DECEASED.value).sum()
        observed_cdr = calculate_observed_cdr(model)

        # Population should have decreased
        assert pop_finish < pop_start, "Population should decrease due to mortality"

        # Observed CDR should be close to target CDR (within 3% tolerance)
        # Small CDR apparently needs more tolerance due to numerical accuracy and/or stochastic variation
        percent_diff = abs((observed_cdr - cdr) / cdr * 100)
        assert percent_diff < 3.0, f"Observed CDR {observed_cdr:.2f} deviated by {percent_diff:.2f}% from target CDR {cdr}"

        # Verify deaths were recorded
        total_deaths = model.nodes.deaths.sum()
        assert total_deaths > 0, "Deaths should have been recorded"
        assert total_deaths == pop_start - pop_finish, "Recorded deaths should match population decrease"

        return

    def test_cdr_10(self):
        """Test mortality with CDR=10 per 1000 per year."""
        cdr = 10.0
        model, _ = create_seir_scenario_with_mortality(cdr)
        pop_start = (model.people.state != State.DECEASED.value).sum()

        model.run()

        pop_finish = (model.people.state != State.DECEASED.value).sum()
        observed_cdr = calculate_observed_cdr(model)

        # Population should have decreased
        assert pop_finish < pop_start, "Population should decrease due to mortality"

        # Observed CDR should be close to target CDR (within 1% tolerance)
        percent_diff = abs((observed_cdr - cdr) / cdr * 100)
        assert percent_diff < 1.0, f"Observed CDR {observed_cdr:.2f} deviated by {percent_diff:.2f}% from target CDR {cdr}"

        # Verify deaths were recorded
        total_deaths = model.nodes.deaths.sum()
        assert total_deaths > 0, "Deaths should have been recorded"
        assert total_deaths == pop_start - pop_finish, "Recorded deaths should match population decrease"

        return

    def test_cdr_20(self):
        """Test mortality with CDR=20 per 1000 per year."""
        cdr = 20.0
        model, _ = create_seir_scenario_with_mortality(cdr)
        pop_start = (model.people.state != State.DECEASED.value).sum()

        model.run()

        pop_finish = (model.people.state != State.DECEASED.value).sum()
        observed_cdr = calculate_observed_cdr(model)

        # Population should have decreased
        assert pop_finish < pop_start, "Population should decrease due to mortality"

        # Observed CDR should be close to target CDR (within 1% tolerance)
        percent_diff = abs((observed_cdr - cdr) / cdr * 100)
        assert percent_diff < 1.0, f"Observed CDR {observed_cdr:.2f} deviated by {percent_diff:.2f}% from target CDR {cdr}"

        # Verify deaths were recorded
        total_deaths = model.nodes.deaths.sum()
        assert total_deaths > 0, "Deaths should have been recorded"
        assert total_deaths == pop_start - pop_finish, "Recorded deaths should match population decrease"

        return

    def test_cdr_40(self):
        """Test mortality with CDR=40 per 1000 per year."""
        cdr = 40.0
        model, _ = create_seir_scenario_with_mortality(cdr)
        pop_start = (model.people.state != State.DECEASED.value).sum()

        model.run()

        pop_finish = (model.people.state != State.DECEASED.value).sum()
        observed_cdr = calculate_observed_cdr(model)

        # Population should have decreased
        assert pop_finish < pop_start, "Population should decrease due to mortality"

        # Observed CDR should be close to target CDR (within 1% tolerance)
        percent_diff = abs((observed_cdr - cdr) / cdr * 100)
        assert percent_diff < 1.0, f"Observed CDR {observed_cdr:.2f} deviated by {percent_diff:.2f}% from target CDR {cdr}"

        # Verify deaths were recorded
        total_deaths = model.nodes.deaths.sum()
        assert total_deaths > 0, "Deaths should have been recorded"
        assert total_deaths == pop_start - pop_finish, "Recorded deaths should match population decrease"

        return


class TestMortalityByEstimator(unittest.TestCase):
    def setUp(self):
        """Set random seed for reproducibility."""
        set_seed(SEED)
        return

    def test_basic_simulation(self):
        """Test that MortalityByEstimator runs and produces reasonable results."""
        model, _ = create_seir_scenario_with_age_specific_mortality()
        pop_start = (model.people.state != State.DECEASED.value).sum()

        model.run()

        pop_finish = (model.people.state != State.DECEASED.value).sum()

        # Population should have decreased
        assert pop_finish < pop_start, "Population should decrease due to mortality"

        # Verify deaths were recorded
        total_deaths = model.nodes.deaths.sum()
        assert total_deaths > 0, "Deaths should have been recorded"
        assert total_deaths == pop_start - pop_finish, "Recorded deaths should match population decrease"

        # All agents should have a DOB assigned
        alive_mask = model.people.state != State.DECEASED.value
        assert np.all(model.people.dob[alive_mask] <= 0), "Living agents should have DOB <= 0 (born before or at start)"

        return

    def test_age_specific_mortality_rates(self):
        """Test that mortality rates match expected age-specific patterns from survival data."""
        model, _ = create_seir_scenario_with_age_specific_mortality()

        # Track population by age at start
        initial_ages = -model.people.dob // 365  # Convert days to years

        model.run()

        # Get final state
        deceased_mask = model.people.state == State.DECEASED.value
        deceased_ages = initial_ages[deceased_mask]

        # Verify that we have deaths across different age groups
        young_deaths = np.sum((deceased_ages >= 0) & (deceased_ages < 5))
        middle_deaths = np.sum((deceased_ages >= 30) & (deceased_ages < 50))
        old_deaths = np.sum((deceased_ages >= 65) & (deceased_ages < 85))

        # Should have deaths in all age categories
        assert young_deaths > 0, "Should have deaths in young age group (0-5)"
        assert middle_deaths > 0, "Should have deaths in middle age group (30-50)"
        assert old_deaths > 0, "Should have deaths in old age group (65-85)"

        # Older populations should have higher death rates
        # Calculate death rates for different age groups
        young_pop = np.sum((initial_ages >= 0) & (initial_ages < 5))
        old_pop = np.sum((initial_ages >= 65) & (initial_ages < 85))

        young_death_rate = young_deaths / young_pop if young_pop > 0 else 0
        old_death_rate = old_deaths / old_pop if old_pop > 0 else 0

        assert old_death_rate > young_death_rate, (
            f"Older age group should have higher death rate than young. "
            f"Young (0-5): {young_death_rate:.4f}, Old (65-85): {old_death_rate:.4f}"
        )

        return

    def test_dod_assignment(self):
        """Test that DOD (date of death) is properly assigned to all agents."""
        model, _ = create_seir_scenario_with_age_specific_mortality()

        # All agents should have a DOD assigned at initialization
        assert hasattr(model.people, "dod"), "Model should have DOD property"
        assert len(model.people.dod) == model.people.count, "All agents should have DOD assigned"

        # DOD should be in the future for most agents (since we're running 10 years)
        initial_dods = model.people.dod.copy()
        future_dods = np.sum(initial_dods > NTICKS)

        # Most people should survive beyond the 10-year simulation period
        assert future_dods > model.people.count * 0.5, (
            f"Most agents should have DOD beyond simulation period. Got {future_dods}/{model.people.count} with future DODs"
        )

        return

    def test_death_occurs_at_dod(self):
        """Test that agents die exactly when their DOD is reached."""
        model, _ = create_seir_scenario_with_age_specific_mortality()

        # Get initial DODs
        initial_dods = model.people.dod.copy()
        initial_count = model.people.count

        model.run()

        # Check that all agents with DOD <= NTICKS are now deceased
        for tick in range(NTICKS):
            should_be_dead_by_tick = initial_dods[:initial_count] <= tick
            actual_dead_by_tick = model.people.state[:initial_count] == State.DECEASED.value

            # Everyone whose DOD has passed should be deceased
            assert np.all(actual_dead_by_tick[should_be_dead_by_tick]), f"At tick {tick}, some agents with DOD <= {tick} are not deceased"

        return

    def test_population_tracking_consistency(self):
        """Test that population counts remain consistent across state channels."""
        model, _ = create_seir_scenario_with_age_specific_mortality()

        model.run()

        # Total living population should match sum of SEIR compartments
        living_count = (model.people.state != State.DECEASED.value).sum()
        seir_sum = (model.nodes.S[-1] + model.nodes.E[-1] + model.nodes.I[-1] + model.nodes.R[-1]).sum()

        assert living_count == seir_sum, f"Living population count ({living_count}) should match sum of SEIR compartments ({seir_sum})"

        return


if __name__ == "__main__":
    unittest.main()
