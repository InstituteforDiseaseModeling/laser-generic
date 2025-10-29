"""
Export required components for an SIR model.

Agents transition from Susceptible to Infectious upon infection and are infectious for a duration.
Agents transition from Infectious to Recovered upon recovery.
Agents remain in the Recovered state indefinitely (no waning immunity).
"""

from .components import InfectiousIR as Infectious
from .components import Recovered
from .components import Susceptible
from .components import TransmissionSI as Transmission
from .components import VitalDynamicsSIR as VitalDynamics
from .shared import State

__all__ = ["Infectious", "Recovered", "State", "Susceptible", "Transmission", "VitalDynamics"]
