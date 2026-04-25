"""
@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

from abc import ABC, abstractmethod


class EventObject(ABC):
    @abstractmethod
    def process(self, rescheduling_us: int = -1) -> int:
        pass  # This is an abstract method, no implementation here.
