"""
Created on Mon Oct 23 16:39:49 2023


@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import pandas as pd


class ProjectKeyResult:
    key: str
    expected: pd.Series
    actual: pd.Series
    
    def __init__(self, key: str, expected: pd.Series, actual: pd.Series):
        self.key = key
        self.expected = expected
        self.actual = actual


class ProjectResult:
    name: str
    failures: list[ProjectKeyResult] = []
    success: list[ProjectKeyResult] = []
    
    def __init__(self, name: str):
        self.name = name
        self.failures = []
        self.success = []

    def add_failure(self, key: str, expected: pd.Series, actual: pd.Series):
        """
        Adds a failure to the project result.
        
        Args:
            key (str): The key of the result.
            expected (pd.Series): The expected result.
            actual (pd.Series): The actual result.
        """
        self.failures.append(ProjectKeyResult(key, expected, actual))

    def add_success(self, key: str, expected: pd.Series, actual: pd.Series):
        """
        Adds a success to the project result.
        
        Args:
            key (str): The key of the result.
            expected (pd.Series): The expected result.
            actual (pd.Series): The actual result.
        """
        self.success.append(ProjectKeyResult(key, expected, actual))
    
    def has_failures(self) -> bool:
        """
        Returns True if the project has failures.
        
        Returns:
            bool: True if the project has failures.
        """
        return len(self.failures) > 0


class ResultsOutput:
    results: list[ProjectResult]
    
    def __init__(self, *results: ProjectResult) -> None:
        self.results = list(results)
    
    def write_yml(self, file: str, model: str, ue_distribution: str, run_time: float):
        with open(file, 'w') as f:
            f.write(f"model: {model}\n")
            f.write(f"ue_distribution: {ue_distribution}\n")
            f.write(f"run_time: {run_time}\n")
            for result in self.results:
                f.write(f"{result.name}:\n")
                f.write("  successes:\n")
                for key_result in result.success:
                    f.write(f"    {key_result.key}:\n")
                    f.write(f"      actual: {key_result.actual}\n")
                    f.write(f"      expected: {key_result.expected}\n")
                f.write("  failures:\n")
                for key_result in result.failures:
                    f.write(f"    {key_result.key}:\n")
                    f.write(f"      actual: {key_result.actual}\n")
                    f.write(f"      expected: {key_result.expected}\n")
