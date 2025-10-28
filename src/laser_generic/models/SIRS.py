"""
Export required components for the SIRS model.
"""

from .components import InfectiousIRS as Infectious
from .components import RecoveredRS as Recovered
from .components import Susceptible
from .components import TransmissionSI as Transmission
from .components import VitalDynamicsSIR as VitalDynamics
from .shared import State

__all__ = ["Infectious", "Recovered", "State", "Susceptible", "Transmission", "VitalDynamics"]
