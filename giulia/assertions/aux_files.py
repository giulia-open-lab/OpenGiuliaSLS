"""
Created on Fri Nov 29 10:05:00 2024

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import os

from giulia.assertions.paths import assert_exists
from giulia.fs.shadowing import shadowing_dir


def validate_shadowing_files():
    """
    Validates the existence of the shadowing files. This is:
    - That the shadowing directory exists.
    - That there is at least one ``*.mat`` file in the shadowing directory.
    
    Raises:
        AssertionError: If the shadowing files are not found.
    
    See Also:
        - :func:`giulia.fs.shadowing.shadowing_dir`
    """

    shadowing_files = shadowing_dir()
    assert_exists(shadowing_files, "Shadowing files not found.")
    
    mat_files = [f for f in os.listdir(shadowing_files) if f.endswith('.mat')]
    assert len(mat_files) > 0, "No .mat files found in the shadowing directory."
