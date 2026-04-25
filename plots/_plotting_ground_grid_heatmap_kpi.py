# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 10:07:09 2025

@author: David López Pérez
dr.david.lopez@ieee.org
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

# =============================================================================
# Imports and path setup
# =============================================================================

# Set up paths
import os
import sys

# Configure Giulia into path
plots_dir = os.path.dirname(__file__)
root_dir = os.path.join(plots_dir, "..")
sys.path.insert(1, root_dir)

import numpy as np
import matplotlib.pyplot as plt

from giulia.logger import info, warning, set_log_level, parse_log_level, get_log_level
from examples.giulia_EV import get_giulia_class


# =============================================================================
# Internal setters and plotters
# =============================================================================
def _deactivate_log()->None:
    set_log_level(parse_log_level(3))
def _activate_log()->None:
    set_log_level(parse_log_level(0))
def _reset_log_level(initial_level)->None:
    set_log_level(parse_log_level(initial_level))

def _remove_shadowing(_g) -> None:
    """Force deterministic channel conditions by removing shadowing and fast fading."""
    assert not isinstance(_g.shadowing_map_cell_obj, list), "shadowing_map_cell_obj has not been processed yet"

    # Set shadow fading to 0dB
    for key in _g.shadowing_map_cell_obj.shadowing_maps_dB.keys():
        for m in range(len(_g.shadowing_map_cell_obj.shadowing_maps_dB[key])):
            _g.shadowing_map_cell_obj.shadowing_maps_dB[key][m] = np.zeros_like(_g.shadowing_map_cell_obj.shadowing_maps_dB[key][m])

    # Deactivate fast fading
    _g.simulation_config_obj.debug_no_randomness = True  # Disable randomness for reproducibility
    return


def _plot_heatmap(_g, kpi: str, scenario_model: str, save_fig: bool = False) -> None:
    """Scatter UE positions with KPI intensity (if available) + BS positions."""
    # Do heatmap using values as intensity
    x_pos_ue_m = _g.ue_deployment_obj.df_ep["position_x_m"].values
    y_pos_ue_m = _g.ue_deployment_obj.df_ep["position_y_m"].values

    x_pos_bs_m = _g.network_deployment_obj.df_ep["position_x_m"].values
    y_pos_bs_m = _g.network_deployment_obj.df_ep["position_y_m"].values

    try:
        values = getattr(_g.performance_obj, "r_e_" + f"{kpi}")[0]
        plt.scatter(x_pos_ue_m, y_pos_ue_m, c=values, cmap="viridis", s=10)
        plt.colorbar(label=f"{kpi} Value")
    except:
        warning(f"Warning: KPI {kpi} not found in performance object. Plotting UE positions only.")
        plt.scatter(x_pos_ue_m, y_pos_ue_m, s=1)

    x_pos_bs_m, y_pos_bs_m = (
        _g.network_deployment_obj.df_ep["position_x_m"].values,
        _g.network_deployment_obj.df_ep["position_y_m"].values,
    )
    plt.scatter(x_pos_bs_m, y_pos_bs_m, marker="^", color="red", label="BSs", s=50)

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title(f"{scenario_model}\n{kpi}")
    plt.legend()
    plt.grid(True, ls="-.", alpha=0.5)
    if save_fig:
        fig_filename = f"ground_grid_heatmap_{scenario_model}_{kpi}.png"
        plt.savefig(fig_filename, dpi=1000)
        info(f"Figure saved as {fig_filename}")
    plt.show()


# =============================================================================
# Public functions
# =============================================================================
@staticmethod
def plot_ground_grid_heatmap(preset,
                             scenario_model,
                             ue_playground_model,
                             ue_distribution,
                             ue_mobility,
                             link_direction,
                             wraparound,
                             save_results,
                             plots,
                             number_of_episodes,
                             regression,
                             enable_GPU,
                             kpi_list: list|str=[],
                             initial_level = None, 
                             save_fig: bool = False,
                             grid_resol_m:int =10
                            )-> None:
    """
    Run a Giulia simulation with UE distribution forced to 'grid' and plot KPI heatmaps.
    """
    _deactivate_log()
    # Impose UE distribution as grid
    ue_distribution = "grid"
    Giulia = get_giulia_class(preset)

    # Initialize Giulia for grid heatmap
    _g = Giulia(preset,
                scenario_model,
                ue_playground_model,
                ue_distribution,
                ue_mobility,
                link_direction,
                wraparound,
                save_results,
                plots,
                number_of_episodes,
                regression,
                enable_GPU)

    # Create a dictionary to map grid resolution to number of UE/grid points
    grid_points_dict_from_resolution = {
        10: 40873,
        15: 18060,
        20: 10132,
        25: 6476,
        50: 1594,
        100: 394,
    }
    
    # Set grid Resolution
    _g.ue_playground_deployment_obj.grid_resol_m = int(grid_resol_m) #  10:40873 , 25: 6476,  50:, 1594


    if isinstance(_g.simulation_config_obj.ue_deployDensity_info, dict):
        _g.simulation_config_obj.ue_deployDensity_info["construct_ue_deployment_3GPPTR38_901_UMa_large_scale_calibration"].number_of_ues = grid_points_dict_from_resolution[grid_resol_m]
    else:
        _g.simulation_config_obj.ue_deployDensity_info.number_of_ues = grid_points_dict_from_resolution[grid_resol_m]
    

    # Configure Simulation
    _g.configure(0, 1e6)

    # Remove Shadowing
    _remove_shadowing(_g)

    # Run Simulation
    _g.run_simulation(0, 1e6)

    _activate_log()
    # Plot Heatmap
    for kpi in kpi_list if isinstance(kpi_list, list) else [kpi_list]:
        _plot_heatmap(_g, kpi=kpi, scenario_model=scenario_model, save_fig=save_fig)
    
    _reset_log_level(initial_level)

    return


if __name__ == "__main__":
    _activate_log()
    
    preset = "GiuliaStd"

    # Define Example Parameters
    scenario_model = "3GPPTR38_901_5G"
    ue_playground_model = None
    ue_distribution = "uniform"  # grid, uniform
    ue_mobility = None
    link_direction = "downlink"
    wraparound = None
    save_results = 0
    plots = False
    number_of_episodes = 1
    regression = False
    enable_GPU = True

    kpi_list = ["SSB_sinr_ue_to_cell_dB"] # e.g. "SSB_sinr_ue_to_cell_dB", "SSB_usefulPower_ue_dBm", "SSB_interfPower_ue_dBm"

    info(f"Running plot_ground_grid_heatmap example:\n{preset} - {scenario_model} - {kpi_list}")

    g = plot_ground_grid_heatmap(preset,
                                 scenario_model,
                                 ue_playground_model,
                                 ue_distribution,
                                 ue_mobility,
                                 link_direction,
                                 wraparound,
                                 save_results,
                                 plots,
                                 number_of_episodes, 
                                 regression,
                                 enable_GPU,
                                 kpi_list=kpi_list,
                                 initial_level=get_log_level(), 
                                 save_fig=True, 
                                 grid_resol_m=20
                            )

    ###############################################
    # Available grid resolutions and corresponding number of UEs/grid points:
    # 10 m : 40873
    # 15 m : 18060
    # 20 m : 10132
    # 25 m : 6476
    # 50 m : 1594
    # 100 m: 394
    # You can change the 'grid_resol_m' parameter in the function call above to test different resolutions.
    # The number of UEs/grid points will adjust accordingly.
    ###############################################