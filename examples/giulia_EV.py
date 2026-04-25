#!/usr/bin/env python3+
# -*- coding: utf-8 -*-

"""
Created on Mon Mar 13 16:36:13 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org 
matteo.bernabe@iteam.upv.es
"""

### This is the MAIN of Guilia 

import argparse
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
from typing import Type, List, Tuple

# Configure Giulia into path
examples_dir = os.path.dirname(__file__)
root_dir = os.path.join(examples_dir, '..')
sys.path.insert(1, root_dir)

from giulia.logger import info, get_log_level, parse_log_level, set_log_level, warning#, LogLevel, debug, error
from giulia.outputs import save_episode_performance

from examples.art import giulia_art
from giulia.tools.tools import free_memory, log_saveable_available_results
    
def inputs(args: List[str]) -> Tuple[str, str, str, str, str, str, bool, int, int, bool, bool, str]:
    
    """
    Parses command-line arguments and returns simulation parameters.

    Args:
        args (List[str]): Command-line arguments passed to the script.

    Returns:
        Tuple containing:
        - preset (str): The simulation preset ('GiuliaStd' or 'GiuliaMfl').
        - scenario_model (str): The simulation scenario model.
        - ue_playground_model (str): User equipment playground model.
        - ue_distribution (str): Distribution model of UEs.
        - ue_mobility (str): Mobility model for UEs.
        - link_direction (str): Direction of the link ('downlink', 'uplink', etc.).
        - wraparound (bool): Whether wraparound is enabled.
        - number_of_episodes (int): Number of simulation snapshots.
        - save_results (int): Whether to save results (1 = Yes, 0 = No).
        - plots (bool): Whether to generate plots.
        - regression (bool): Whether this is a regression test.
        - project_name (str): Generated project name based on scenario and UE distribution.
    """
    
    
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Giulia Simulation Configuration")
    
    # Simulation preset
    parser.add_argument("--preset", type=str, default="GiuliaStd",
                        help="Simulation preset:\n"
                             "  - 'GiuliaStd': Standard simulation\n"
                             "  - 'GiuliaMfl': Multi-frequency layer simulation")
    
    # Scenario/Network model
    parser.add_argument("--scenario_model", type=str, default="ITU_R_M2135_UMa", #Dataset_HW
                        help="Specify the simulation model:\n"
                            "ITU_R_M2135_UMa,\n" 
                            "ITU_R_M2135_UMi,\n"
                            
                            "3GPPTR36_814_Case_1,\n"  
                            "3GPPTR36_814_Case_1_omni,\n" 
                            "3GPPTR36_814_Case_1_single_bs,\n"
                            "3GPPTR36_814_Case_1_single_bs_omni,\n"
                            "3GPPTR36_814_Case_1_omni_dana,\n"
                            
                            "3GPPTR38_901_UMa_C1,\n" 
                            "3GPPTR38_901_UMa_C2,\n" 
                            "3GPPTR38_901_UMa_lsc,\n" 
                            "3GPPTR38_901_UMa_lsc_sn,\n" 
                            "3GPPTR38_901_UMa_lsc_single_bs,\n"
                            "3GPPTR38_901_UMa_lsc_single_sector,\n"
                            "3GPPTR38_901_UMa_2GHz_lsc,\n"
                            "3GPPTR38_901_UMa_C_band_lsc,\n"
                            
                            "3GPPTR38_901_UMi_C1,\n"
                            "3GPPTR38_901_UMi_C2,\n"
                            "3GPPTR38_901_UMi_lsc, \n"
                            "3GPPTR38_901_UMi_C_band_lsc,\n"
                            "3GPPTR38_901_UMi_fr3_lsc,\n"  
                            
                            "3GPPTR38_901_UPi_fr3_lsc,\n"                          
                            
                            "3GPPTR36_777_UMa_AV,\n"
                            "3GPPTR36_777_UMi_AV,\n"
                            "3GPPTR38_811_Urban_NTN,\n"
                            "3GPPTR38_811_Dense_Urban_NTN,\n"
                            "3GPPTR38_811_Dense_Urban_HAPS_ULA,\n"
                            "3GPPTR38_811_Dense_Urban_HAPS_UPA,\n" 
                            "3GPPTR38_811_Dense_Urban_HAPS_Reflector,\n" 
                            
                            "ITU_R_M2135_UMa_multilayer,\n"
                            "ITU_R_M2135_UMa_Umi_colocated_multilayer,\n"
                            "ITU_R_M2135_UMa_Umi_noncolocated_multilayer,\n"
                            
                            "3GPPTR38_901_4G,\n"
                            "3GPPTR38_901_5G,\n"
                            "3GPPTR38_901_6G,\n"
                            "3GPPTR38_901_4G5G_multilayer,\n"
                            "3GPPTR38_901_4G_5G_multilayer,\n"
                            "3GPPTR38_901_4G_5G6G_multilayer,\n"
                            "3GPPTR38_901_4G_5G_6G_multilayer,\n"
                            
                            "3GPPTR38_901_4G5G_cell_reselection,\n"
                            
                            "dataset\n")
            
    # User Equipment (UE) configurations
    parser.add_argument("--ue_playground_model", type=str, default=None,
                        help="User playground model:\n"
                        "Every ITU and 3GPP scenario model above has its corresponding UE playground deployment model. If we want to use such corresponding model, no need to specify any input.\n"
                        "Alternatively, we have other UE playground deployment models:\n"                        
                             "  - 'None' (default scenario-based)\n"
                             "  - 'rectangular'\n"
                             "  - 'circular'")

    parser.add_argument("--ue_distribution", type=str, default="uniform",
                        help="UE distribution model:\n"
                             "  - 'grid'\n"
                             "  - 'uniform'\n"
                             "  - 'uniform_with_hotspots'\n"
                             "  - 'inhomogeneous_per_cell'\n"
                             "  - 'inhomogeneous_per_cell_with_hotspots'")

    parser.add_argument("--ue_mobility", type=str, default=None,
                        help="UE mobility model:\n"
                             "  - 'None' (static UEs)\n"
                             "  - 'straight_walk'\n"
                             "  - 'circular_walk'")
    
    # Network configurations
    parser.add_argument("--link_direction", type=str, default="downlink",
                        help="Link direction:\n"
                             "  - 'downlink'\n"
                             "  - 'uplink'\n"
                             "  - 'downlink_uplink' (both)")

    parser.add_argument("--wraparound", type=bool, default=None,
                        help="Enable wraparound for network topology.")    

    # Simulation parameters
    parser.add_argument("--snapshots", type=int, default=2,
                        help="Number of simulation snapshots (default: 2).")
    
    parser.add_argument("--enable_GPU", type=bool, default=True, 
                        help="Specify whether you want to enable GPU usage or not.")        

    parser.add_argument("--regression", type=bool, default=False,
                        help="Enable regression testing mode.")

    # Output settings
    parser.add_argument("--log_level", type=int, default=0,
                        choices=range(0, 4),
                        help="Set the logging level.")
    
    parser.add_argument("--save_results", type=int, default=1,
                        help="Save simulation results (1 = Yes, 0 = No).")

    parser.add_argument("--plots", type=bool, default=True,
                        help="Generate plots for simulation results.")
    
    # Parse arguments
    args = parser.parse_args(args)
    
    # Set project name
    project_name = args.scenario_model + "_" + args.ue_distribution  
    
    return args, project_name
    

def get_giulia_class(preset: str) -> Type:
    """
    Dynamically imports and returns the correct Giulia class based on the preset.

    Args:
        preset (str): The preset name, either 'GiuliaMfl' or 'GiuliaStd'.

    Returns:
        Type: The corresponding Giulia class.

    Raises:
        ValueError: If the preset name is invalid.
    """
    if preset == "GiuliaMfl":
        from giulia.presets.g_multi_frequency_layer_preset import Giulia
    elif preset == "GiuliaStd":
        from giulia.presets.g_standard_preset import Giulia
    else:
        raise ValueError(f"Invalid preset '{preset}'. Choose 'GiuliaMfl' or 'GiuliaStd'.")

    return Giulia

    
def main(args: List[str]):
    """
    Main function to run Giulia simulations.

    Args:
        args (List[str]): Command-line arguments.
    """
    #### INPUTS ####
    args, project_name = inputs(args)
    
    preset: str = args.preset
    scenario_model: str = args.scenario_model
    ue_playground_model: str = args.ue_playground_model
    ue_distribution: str = args.ue_distribution
    ue_mobility: str = args.ue_mobility
    link_direction: str = args.link_direction
    wraparound: bool = args.wraparound
    number_of_episodes: int = args.snapshots
    regression: bool = args.regression
    save_results: int = args.save_results
    plots: bool = args.plots
    log_level = args.log_level
    enable_GPU=args.enable_GPU

    if log_level is not None:
        level = parse_log_level(log_level)
        set_log_level(level)
    
    # Print Giulia Art
    info(giulia_art)

    # Show current log level as error so that it's always shown
    warning("Logging level is set to", get_log_level())
    
    #### OUTPUTS #### 
    ###########################          
    outputs_obj = save_episode_performance.Episode_Performance(number_of_episodes, regression)

    #### RUNNING GIULIA ####
    max_simulation_time_us = 1e6  # Max simulation time in microseconds
    decision_making_interval_us = 1e6  # Decision-making interval
    number_of_steps = int(max_simulation_time_us/decision_making_interval_us)  # Total simulation steps

    # Dynamically load the appropriate Giulia class
    Giulia = get_giulia_class(preset)

    # Initialize Giulia simulation
    g = Giulia(preset, scenario_model, ue_playground_model, ue_distribution, ue_mobility,
               link_direction, wraparound, save_results, plots, number_of_episodes, regression, enable_GPU)


    # Run simulation for each episode
    for episode_index in range(number_of_episodes):
        # Configure Giulia for the current episode
        g.configure(episode_index, max_simulation_time_us)

        # Run simulation steps
        for step_index in range(number_of_steps):
            info("#" * 40 + 
                 f"\nEpisode = {episode_index}, Step = {step_index}\n" + 
                 "#" * 40)
            
            # Run a step of the simulation
            g.run_simulation(step_index, decision_making_interval_us)


        # Save performance metrics for this episode
        outputs_obj.save_episode_performance(episode_index, g)

    # Save all episodes' performance data to file
    outputs_obj.save_episodes_performance_in_file(project_name)

    # Free The Memory
    free_memory()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Run with default values if no arguments provided
        main([])
    else:
        # Run with command-line arguments
        main(sys.argv[1:])