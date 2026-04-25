"""
Created on Fri Nov 29 10:05:00 2024

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

from os import environ

from giulia.fs.path import project_dir

def shadowing_dir() -> str:
    if 'SHADOWING_FILES' in environ:
        return environ['SHADOWING_FILES']
    else:
        return project_dir('shadowing')
