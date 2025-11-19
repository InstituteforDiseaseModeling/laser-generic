import unittest

import numpy as np

from laser.generic.newutils import ValuesMap
from laser.generic.newutils import estimate_capacity
from utils import stdgrid

NTICKS = 365 * 5


class TestCapacity(unittest.TestCase):
    def test_capacity_estimate(self):
        scenario = stdgrid()

        cbr = np.random.uniform(5, 35, len(scenario))  # CBR = per 1,000 per year
        birthrate_map = ValuesMap.from_nodes(cbr, nsteps=NTICKS)

        num_agents = estimate_capacity(birthrate_map.values, scenario.population)
        estimate = num_agents.sum()

        assert estimate > scenario.population.sum(), f"Estimate {estimate} not greater than population {scenario.population.sum()}"

        # Run scenarios across two dimensions, iterating over CBRs from [5, 10, 20, 25, 30, 40, 50]
        # Use a scenario with nodes with initial population in [10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
        cbrs = np.array([5, 10, 20, 25, 30, 40, 50])
        populations = np.array([10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000])
        previous = None
        for cbr in cbrs:
            birthrate_map = ValuesMap.from_scalar(cbr, nnodes=len(populations), nsteps=NTICKS)
            estimate = estimate_capacity(birthrate_map.values, populations)  # default safety_factor=1.0

            assert np.all(estimate > populations), f"Estimate \n{estimate}\n not greater than population \n{populations}"

            # Check that estimates increase with increasing CBR
            if previous is not None:
                assert np.all(estimate > previous), (
                    f"Estimate \n{estimate}\n with CBR {cbr} not greater than previous estimate at lower CBR \n{previous}"
                )
            previous = estimate

            # Now test with safety factor of 2.0
            safer = estimate_capacity(birthrate_map.values, populations, safety_factor=2.0)
            assert np.all(safer > estimate), (
                f"Estimate with safety factor 2.0 \n{safer}\n not greater than estimate with safety factor 1.0 \n{estimate}"
            )

        return


if __name__ == "__main__":
    unittest.main()
