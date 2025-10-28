"""
Export required components for the SIS model.
"""

from .components import InfectiousIS as Infectious
from .components import Susceptible
from .components import TransmissionSI as Transmission
from .components import VitalDynamicsSI as VitalDynamics
from .shared import State

__all__ = ["Infectious", "State", "Susceptible", "Transmission", "VitalDynamics"]
