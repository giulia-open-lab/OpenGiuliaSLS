# -*- coding: utf-8 -*-

"""
Created on Tue Aug 27 14:39:31 2024

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@itea.upv.es
"""

import gc
import os
import sys

# Ensure the repo root is importable so `giulia.inputs` resolves even when this
# file is run directly (python examples/giulia_EV_batch.py), not only via main.py.
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import giulia_EV
from giulia.inputs import InputConfig

    
def run_simulations(args=None, config=None):
    # Output formats (e.g. NPZ) are taken from config.yml. main.py passes `config`
    # in; when this file is run directly we load it here. Without a config,
    # giulia_EV.main defaults to Parquet-only and produces NO .npz files.
    if config is None:
        config = InputConfig.load()

    # Define the scenarios and UE distributions you want to run

    # Scenarios that have benchmark curves defined in giulia/plots/plotting.py
    # (get_coupling_loss_benchmark / get_geometry_sinr_benchmark).
    # The benchmark keys all end in "_uniform", so these must be run with the
    # "uniform" UE distribution for the benchmark overlay to match.
    scenarios = [
        # --- coupling-loss AND geometry-SINR benchmark available ---
        # "ITU_R_M2135_UMa",          # antenna element
        # "ITU_R_M2135_UMi",          # antenna element
        # "3GPPTR36_814_Case_1",      # antenna element
        # "3GPPTR38_901_UMa_lsc",     # SSB (.mat)
        "3GPPTR38_901_UMa_lsc_sn",  # SSB (.mat) - Sionna variant (no coupling benchmark)
        # "3GPPTR38_901_UMi_lsc",     # SSB (.mat)
        # --- geometry-SINR benchmark only (no coupling benchmark) ---
        "3GPPTR38_901_UMa_C1",      # SSB (.txt)
        # "3GPPTR38_901_UMa_C2",      # SSB (.txt)
        "3GPPTR38_901_UMi_C1",      # SSB (.txt)
        # "3GPPTR38_901_UMi_C2",      # SSB (.txt)
    ]

    ue_distributions = [
        "uniform"
    ]

    # Simulation preset: "GiuliaStd" (single-layer) or "GiuliaMfl" (multi-frequency-layer).
    # Set explicitly here so we don't rely on giulia_EV.py's default.
    preset = "GiuliaStd"
    
    
    # Iterate over each scenario and UE distribution
    for scenario in scenarios:
        for ue_distribution in ue_distributions:
            print(f"Running simulation with scenario: {scenario} and UE distribution: {ue_distribution}")
            
            
            
            # Prepare the arguments as they would be provided in the command line
            args = [
                "--preset", preset,
                "--scenario_model", scenario,
                "--ue_distribution", ue_distribution
            ]
            
            # Call the main function of the imported script (pass config so the
            # output formats from config.yml -- e.g. NPZ -- are honoured)
            giulia_EV.main(args, config)
            
            # Clear memory after each iteration
            # Clear all variables in memory after each iteration
            for name in list(globals().keys()):
                if name not in ['giulia_EV', 'run_simulations', 'gc']:  # Don't delete necessary modules and functions
                    del globals()[name]
                    
            gc.collect()  # Force garbage collection to free up memory            

if __name__ == "__main__":
    run_simulations()

