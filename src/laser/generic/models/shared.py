from enum import Enum

import numpy as np
from laser.core.demographics import AliasedDistribution
from laser.core.demographics import KaplanMeierEstimator

__all__ = ["State", "sample_dobs", "sample_dods"]


def sample_dobs(dobs: np.ndarray, pyramid: AliasedDistribution, tick: int) -> None:
    # Get years of age sampled from the population pyramid
    dobs[:] = pyramid.sample(len(dobs)).astype(np.int16)  # Fit in np.int16
    dobs *= 365  # Convert years to days
    dobs += np.random.randint(0, 365, size=len(dobs))  # add some noise within the year
    # pyramid.sample actually returned ages. Turn them into dobs by treating them
    # as days before today.
    dobs[:] = tick - dobs

    return


def sample_dods(dobs: np.ndarray, dods: np.ndarray, survival: KaplanMeierEstimator, tick: int) -> None:
    # An agent's age is (tick - dob).
    dods[:] = survival.predict_age_at_death(ages := tick - dobs).astype(np.int16)  # Fit in np.int16
    dods -= ages  # How many more days will each person live?

    return


class State(Enum):
    DECEASED = -1
    SUSCEPTIBLE = 0
    EXPOSED = 1
    INFECTIOUS = 2
    RECOVERED = 3

    def __new__(cls, value):
        obj = object.__new__(cls)
        obj._value_ = np.int8(value)
        return obj
