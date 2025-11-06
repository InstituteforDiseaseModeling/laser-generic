"""
Export required components for an SIS model.

Agents transition from Susceptible to Infectious upon infection and are infectious for a duration.
Agents transition from Infectious back to Susceptible upon recovery.
"""

from .components import InfectiousIS as Infectious
from .components import Susceptible
from .components import TransmissionSI as Transmission
from .components import VitalDynamicsSI as VitalDynamics
from .shared import State

__all__ = ["Infectious", "State", "Susceptible", "Transmission", "VitalDynamics"]
