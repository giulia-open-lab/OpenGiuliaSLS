"""
@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

class Snapshot_control:
    _instance = None

    @staticmethod
    def get_instance():
        if Snapshot_control._instance is None:
            Snapshot_control._instance = Snapshot_control()
        return Snapshot_control._instance

    def __init__(self):
        if Snapshot_control._instance is not None:
            raise Exception("This class is a singleton and has already been instantiated.")
        self.num_snapshots = 0
