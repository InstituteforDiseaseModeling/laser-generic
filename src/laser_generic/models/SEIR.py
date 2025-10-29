"""
Export required components for an SEIR model.

Agents transition from Susceptible to Exposed upon infection, with an incubation duration.
Agents transition from Exposed to Infectious after the incubation period and are infectious for a duration.
Agents transition from Infectious to Recovered after the infectious period.
Agents remain in the Recovered state indefinitely (no waning immunity).
"""

from .components import Exposed
from .components import InfectiousIR as Infectious
from .components import Recovered
from .components import Susceptible
from .components import TransmissionSE as Transmission
from .components import VitalDynamicsSEIR as VitalDynamics
from .shared import State

__all__ = ["Exposed", "Infectious", "Recovered", "State", "Susceptible", "Transmission", "VitalDynamics"]
