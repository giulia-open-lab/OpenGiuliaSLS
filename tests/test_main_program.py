"""
Created on Fri Apr 11 17:24:43 2025

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org 
matteo.bernabe@iteam.upv.es
"""
import multiprocessing as mp
from multiprocessing import set_start_method

if __name__ == "__main__":
    # Use forkserver for the subprocesses
    # set_start_method('forkserver')
    set_start_method('spawn')

import argparse

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU usage for tests

import time
import traceback
from typing import Optional

import numpy as np
import pandas as pd

# Configure Giulia into path
this_dir = os.path.dirname(__file__)
root_dir = os.path.join(this_dir, '..')
sys.path.insert(1, root_dir)

from giulia.fs.path import results_file

from tests.arguments import parse_arguments
from tests.art import tests_art
from tests.assertions import calculate_error
from tests.const import test_keys, atol
from tests.precomputed_files import load_precomputed_files, precomputed_files_dir
from tests.results_data import ProjectResult, ResultsOutput
from tests.results_output import results_files_to_markdown, results_files_to_html
from tests.shadowing_files import load_shadowing_files
from tests.system_monitor import SystemMonitor, compute_ranges

print(tests_art)

run_tests: list[str] | None = None 

monitor_interval: float = 0.1
"""How often in seconds to take measurements about system resources usage."""

arguments: Optional[argparse.Namespace] = None
"""Set by ``parse_arguments``, contains all the CLI arguments given by the user."""

max_memory_attempts = 2
"""The maximum attempts to perform when the GPU fills up."""

available_tests = [
    ("ITU_R_M2135_UMa", "uniform"),
    ("ITU_R_M2135_UMa", "uniform_with_hotspots"),
    ("ITU_R_M2135_UMa", "inhomogeneous_per_cell"),
    ("ITU_R_M2135_UMa", "inhomogeneous_per_cell_with_hotspots"),
    ("ITU_R_M2135_UMi", "uniform"),
    ("3GPPTR36_814_Case_1", "uniform"),
    ("3GPPTR38_901_UMa_lsc", "uniform"),
    ("3GPPTR38_901_UMi_lsc", "uniform"),
    ("3GPPTR36_777_UMa_AV", "uniform"),
    ("3GPPTR36_777_UMi_AV", "uniform"),
    ("3GPPTR38_901_4G", "inhomogeneous_per_cell"),
    ("3GPPTR38_901_5G", "inhomogeneous_per_cell"),
    ("3GPPTR38_901_4G5G_multilayer", "inhomogeneous_per_cell"),
    ("3GPPTR38_901_4G_5G_multilayer", "inhomogeneous_per_cell"),
    # ("3GPPTR38_901_4G_5G_6G_multilayer", "inhomogeneous_per_cell"),
    # ("3GPPTR38_901_UMa_C1", "uniform"),
    # ("3GPPTR38_901_UMa_C2", "uniform"),
    # ("3GPPTR38_901_UMi_C1", "uniform"),
    # ("3GPPTR38_901_UMi_C2", "uniform"),
]
"""A list of pairs with all the tests available."""

outputs_dir = os.path.join(root_dir, 'outputs')
test_outputs_dir = os.path.join(outputs_dir, 'tests')


def skip_test(name: str) -> bool:
    """
    Checks whether a test with name ``name`` should be skipped.
    The test will be skipped if ``run_tests`` is not null, not empty, and doesn't contain ``name``.
    Args:
        name: The name of the test to run.
    Returns:
        ``true`` if the test should be skipped, ``false`` otherwise.
    """
    return run_tests is not None and len(run_tests) > 0 and not name in run_tests


def load_test_preferences():
    """
    Load the tests to run from the environment variable TESTS.
    
    Updates the value of the global variable ``run_tests``.
    """
    global run_tests, arguments, monitor_interval

    monitor_interval = arguments.MONITOR

    if 'TESTS' in os.environ:
        tests = os.environ['TESTS']
        run_tests = tests.split(' ')
    elif arguments is not None and arguments.TESTS is not None:
        run_tests = arguments.TESTS
    else:
        print("Running all tests")
        return

    print("Running tests:")
    for test_name in run_tests:
        print(f"- {test_name}")
    print()


def validate_tests():
    """
    Validates that the tests requested are all valid.
    """
    global run_tests
    if run_tests is None:
        return
    print("Validating tests...")
    temp_run_tests = []
    for test_name in run_tests:
        if not '_' in test_name:
            is_valid = False
        else:
            is_valid = test_name in '_'.join(f'{model}_{ue_dist}' for (model, ue_dist) in available_tests)
        if not is_valid:
            print(f"Invalid test given: {test_name}")
            sys.exit(1)

        # Replace the '__' with '_' for the sake of being compliant with rest of the code            
        temp_run_tests.append(test_name.replace('__', '_'))
        
        
    run_tests = temp_run_tests
    del temp_run_tests
    print("Requested tests validated.")


def check_template(name: str, results_load: pd.DataFrame, gt_load: pd.DataFrame) -> ProjectResult:
    """
    Check the results of the test against the ground truth.

    Args:
        name (str): Name of the type of operation applied to the results.
        results_load (pd.DataFrame): Results obtained in the test.
        gt_load (pd.DataFrame): Expected results.
    """

    # Make all the checks for isclose
    result = ProjectResult(name)
    for key in test_keys:
        actual: pd.Series = results_load[key]
        expected: pd.Series = gt_load[key]
        # Calculate the error percentage
        err = calculate_error(actual, expected)
        
        if isinstance(err, pd.Series):
            if (err > atol).any():
                result.add_failure(key, expected, actual)
            else:
                result.add_success(key, expected, actual)
        else:
            if err > atol:
                result.add_failure(key, expected, actual)
            else:
                result.add_success(key, expected, actual)

    return result


def compare_test_files(name: str, path_results: str, path_gt: str) -> ProjectResult:
    """
    Compare the results of the test against the ground truth.

    Args:
        name (str): Name of the type of operation applied to the results.
        path_results (str): The path to the expected results file.
        path_gt (str): The path to the actual results file.
    """
    results_load: pd.DataFrame = np.load(path_results)
    gt_load: pd.DataFrame = np.load(path_gt)

    return check_template(name, results_load, gt_load)


def __run_giulia(model: str, ue_distribution: str):
    # We have to import Giulia from the sub-process, otherwise the GPU device cannot be used
    from examples.giulia_EV import main
    main(["--scenario_model", model, "--ue_distribution", ue_distribution, "--snapshots", str(2), "--regression", str(True)])


def run_test(model: str, ue_distribution: str):
    """
    Runs Giulia in a test environment, monitoring the system resources usage, and making all the required comparisons
    to make sure the operation is successful.
    """
    global monitor_interval

    # Check whether the test shall be run
    if skip_test(f'{model}_{ue_distribution}'):
        print(f'Skipping test for {model=} and {ue_distribution=}')
        return

    print(f'Running test for {model=} and {ue_distribution=}')

    # Test your main function
    test_monitor = SystemMonitor(
        filename=os.path.join(test_outputs_dir, f"usage_{model}_{ue_distribution}.csv"),
        interval=monitor_interval
    )
    test_monitor.start()  # Start monitoring CPU-GPU usage
    try:
        main_start = time.time()
        # Run Giulia in a subprocess so that when the execution is over, all resources are freed up
        p = mp.Process( target=__run_giulia, args=[model, ue_distribution])
        p.start()
        p.join()
        main_end = time.time()

    finally:
        test_monitor.stop()
        test_monitor.plot_usage(
            output_file=os.path.join(test_outputs_dir, f"usage_{model}_{ue_distribution}.png"),
            label=f'{model}_{ue_distribution}',
        )

    if arguments.GENERATE:
        print('Running with --generate option, won\'t compare results.')
        return

    project_name: str = model + '_' + ue_distribution

    # Check mean
    computed_file = results_file(project_name, 'results-mean.npz')
    precomputed_file = os.path.join(precomputed_files_dir, project_name, 'results-mean.npz')
    mean_results = compare_test_files("mean", computed_file, precomputed_file)

    # Check median
    computed_file = results_file(project_name, 'results-median.npz')
    precomputed_file = os.path.join(precomputed_files_dir, project_name, 'results-median.npz')
    median_results = compare_test_files("median", computed_file, precomputed_file)

    # Store results
    test_output_file = os.path.join(test_outputs_dir, f'{project_name}.yml')
    output = ResultsOutput(mean_results, median_results)
    output.write_yml(test_output_file, model, ue_distribution, main_end - main_start)

    # Fail if there are any failures
    if mean_results.has_failures():
        raise AssertionError(f'There are failures in mean results for {model=} and {ue_distribution=}')
    if median_results.has_failures():
        raise AssertionError(f'There are failures in median results for {model=} and {ue_distribution=}')


if __name__ == "__main__":
    arguments = parse_arguments(available_tests)

    load_test_preferences()
    validate_tests()
    load_precomputed_files(run_tests, arguments.DOWNLOAD_PRECOMPUTED_FILES)
    load_shadowing_files()

    if not os.path.exists(test_outputs_dir):
        os.makedirs(test_outputs_dir)

    monitor = SystemMonitor(filename=os.path.join(test_outputs_dir, f"usage.csv"), interval=monitor_interval)
    monitor.start() # Start monitoring CPU-GPU usage

    try:
        # Run all the tests available, run_test will skip the disabled ones
        for (m, d) in available_tests:
            try:
                run_test(m, d)
            except AssertionError as error:
                print(f'AssertionError for {m} and {d}. {error}')
            except Exception:
                print(f'Error for {m} and {d}.')
                print(traceback.format_exc())
    finally:
        if arguments.OUTPUT_MD:
            results_files_to_markdown(test_outputs_dir, os.path.join(outputs_dir, 'Summary.md'))
        if arguments.OUTPUT_HTML:
            results_files_to_html(test_outputs_dir, os.path.join(outputs_dir, 'Summary.html'))
        monitor.stop() # Stop monitoring usage
        monitor.plot_usage(
            output_file=os.path.join(test_outputs_dir, f"usage.png"),
            label=f'Global Usage. {len(run_tests) if run_tests is not None else len(available_tests)} tests',
            ranges=compute_ranges(test_outputs_dir,lambda file: file != 'usage.csv' and file.endswith('.csv'))
        )

    display_summary_in = input("Want to display resulting test summary? (yes/no) [default: no]:  ")
    
    #Print Summary Results In Terminal
    if display_summary_in.lower()=="yes":
        from tests.test_read_summary_tools import _print_summary_to_terminal

        summary_md_path = os.path.join(outputs_dir, "Summary.md")
        try:
            print("\n\n[Summary] Rendering to terminal...\n")
            if os.path.exists(summary_md_path):
                _print_summary_to_terminal(summary_md_path)
            else:
                print(f"[Summary] {summary_md_path} not found. Did you pass --output-md?")
        except Exception as e:
            import traceback
            print("[Summary] Error:", e)
            print(traceback.format_exc())
    else: 
        pass