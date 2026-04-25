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


def project_file(name: str) -> str:
    """
    Provides a file in the root of the project, aka the repository root.
    Args:
        name: The name of the file to obtain.
    Returns:
        A full path with the name desired targeting the file requested.
    """
    return os.path.join(PROJ_DIR, name)


def project_dir(name: str) -> str:
    """
    Provides a directory in the root of the project, aka the repository root.
    Creates it automatically if it doesn't exist.
    Args:
        name: The name of the directory to obtain.

    Returns:
        A full path with the name desired targeting the folder requested.
    """
    path = project_file(name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def outputs_dir() -> str:
    """
    Provides the directory where all outputs should be stored.
    In most cases, the correct function to use is ``results_file``.
    Use this one only if you are completely sure that it's needed.
    Returns:
        str: The full absolute path to the outputs directory.
    """
    outputs = project_dir('outputs')
    return os.path.realpath(outputs)


def results_dir() -> str:
    """
    Provides the directory where all results should be stored.
    In most cases, the correct function to use is ``results_file``.
    Use this one only if you are completely sure that it's needed.
    Returns:
        str: The full absolute path to the results directory.
    """
    outputs = outputs_dir()
    results = os.path.join(outputs, 'results')
    return os.path.realpath(results)


def results_file(project_name: str, file_name: str) -> str:
    """
    Provides a file path to be used as an output in an operation of a project.
    The directory where the file is stored, will be created automatically.
    Args:
        project_name: The name of the project that generated the result
        file_name: The name of the file to store the result into.
    Returns:
        A full path with the name desired targeting the file requested.
        It will be inside of: ``<Giulia root>/outputs/results/<project_name>/<file_name>``
    """
    results = results_dir()
    directory = os.path.join(results, project_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return os.path.join(directory, file_name)
