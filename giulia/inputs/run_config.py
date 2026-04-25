"""
@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

class RunConfig:
    example: str
    """
    Which example file to run.
    
    Options: EV, EV_batch, EV_EE_UEassociation, EV_EE_UEassociation_batch, EV_HapsSym_Reflector
    """

    def __init__(self, example: str):
        self.example = example


    @staticmethod
    def from_config(data: dict) -> 'RunConfig':
        example = data['example']
        return RunConfig(example)
