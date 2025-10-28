"""
Export required components for the SIR model.
"""

from .components import InfectiousIR as Infectious
from .components import Recovered
from .components import Susceptible
from .components import TransmissionSI as Transmission
from .components import VitalDynamicsSIR as VitalDynamics
from .shared import State

__all__ = ["Infectious", "Recovered", "State", "Susceptible", "Transmission", "VitalDynamics"]
