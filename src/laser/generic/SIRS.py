"""
Export required components for an SIRS model.

Agents transition from Susceptible to Infectious upon infection.
Agents transition from Infectious to Recovered upon recovery after the infectious duration.
Agents transition from Recovered back to Susceptible upon waning immunity after the waning duration.
"""

from .components import InfectiousIRS as Infectious
from .components import RecoveredRS as Recovered
from .components import Susceptible
from .components import TransmissionSI as Transmission
from .components import VitalDynamicsSIR as VitalDynamics
from .shared import State

__all__ = ["Infectious", "Recovered", "State", "Susceptible", "Transmission", "VitalDynamics"]
