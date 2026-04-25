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
from typing import Optional

def assert_exists(path: str, message: Optional[str] = None):
    """
    Validates the existence of a file or directory.
    
    If the file or directory does not exist, a FileNotFoundError is raised.

    Args:
        path (str): The absolute path of the file or directory
        
    Raises:
        FileNotFoundError: If the file or directory does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found." if message is None else message)
