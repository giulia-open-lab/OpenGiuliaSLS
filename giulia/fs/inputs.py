"""
Created on Fri Nov 29 10:05:00 2024

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import os.path

from giulia import PROJ_DIR


def phy_file(package: str, file: str) -> str:
    """
    Resolves a file in the "data" dir of the project.
    Arguments resolve to: ``/data/phy/<package>/<file>``
    Args:
        package: The package of the file.
        file: The file name to resolve
    Returns:
        The resolved file.
    """
    data_dir = os.path.join(PROJ_DIR, 'data', 'phy')
    return os.path.join(data_dir, package, file)


def data_driven_extras(option: str) -> str:
    """
    Resolves a file in the "data" dir of the project.
    Arguments resolve to: ``/data/data_driven_extras/files_<option>_by_Giulia``
    Args:
        option: Either ``loaded`` or ``saved``
    """
    if option not in ['loaded', 'saved']:
        raise ValueError("Valid options: loaded, saved")
    return os.path.join(PROJ_DIR, 'data', 'data_driven_extras', f'files_{option}_by_Giulia')

def data_file(name: str) -> str:
    """
    Resolves a file in the "data" dir of the project.
    Arguments resolve to: ``/data/<name>``
    Args:
        name: The file name to resolve
    Returns:
        The resolved file.
    """
    return os.path.join(PROJ_DIR, 'data', name)