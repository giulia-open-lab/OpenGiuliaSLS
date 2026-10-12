# -*- coding: utf-8 -*-
"""
giulia_EV_batch_test_main_program.py

Batch runner that reproduces the (scenario_model, ue_distribution) combinations
of the regression test suite (``available_tests`` in tests/test_main_program.py).

Unlike tests/test_main_program.py -- which runs each case in a subprocess with a
Parquet-only InputConfig and compares against precomputed ground truth -- this
batch simply runs every combination through giulia_EV.main and lets the output
formats from ``config.yml`` apply (so NPZ results-raw.npz are produced, like
giulia_EV_batch.py).

Run directly (python examples/giulia_EV_batch_test_main_program.py) or via
main.py with run.example pointing here.
"""

import gc
import os
import sys

# Ensure the repo root is importable so `giulia.inputs` resolves even when this
# file is run directly, not only via main.py.
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import giulia_EV
from giulia.inputs import InputConfig


def run_simulations(args=None, config=None):
    # Output formats (e.g. NPZ) are taken from config.yml. main.py passes `config`
    # in; when run directly we load it here. Without a config, giulia_EV.main
    # defaults to Parquet-only and produces NO .npz files.
    if config is None:
        config = InputConfig.load()

    # (scenario_model, ue_distribution) pairs -- mirrors `available_tests`
    # in tests/test_main_program.py. Each scenario carries its own UE distribution.
    scenarios = [
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
        ("3GPPTR38_901_UMa_C1", "uniform"),
        ("3GPPTR38_901_UMa_C2", "uniform"),
        ("3GPPTR38_901_UMi_C1", "uniform"),
        ("3GPPTR38_901_UMi_C2", "uniform"),
    ]

    # tests/test_main_program.py runs every scenario with GiuliaStd/GiuliaMfl.
    preset = "GiuliaStd"

    for scenario_model, ue_distribution in scenarios:
        print(f"Running simulation with scenario: {scenario_model} and UE distribution: {ue_distribution}")

        # Prepare the arguments as they would be provided in the command line
        args = [
            "--preset", preset,
            "--scenario_model", scenario_model,
            "--ue_distribution", ue_distribution,
        ]

        # Call the main function of the imported script (pass config so the output
        # formats from config.yml -- e.g. NPZ -- are honoured)
        giulia_EV.main(args, config)

        # Clear module-level variables after each iteration to free up memory
        for name in list(globals().keys()):
            if name not in ('giulia_EV', 'run_simulations', 'gc', 'InputConfig', 'os', 'sys'):
                del globals()[name]
        gc.collect()


if __name__ == "__main__":
    run_simulations()
