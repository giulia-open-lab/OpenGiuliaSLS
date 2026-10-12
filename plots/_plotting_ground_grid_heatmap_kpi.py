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

def _remove_randomness_shadow_and_fading(_g) -> None:
    """Force deterministic channel conditions by removing shadowing and fast fading."""
    assert not isinstance(
        _g.shadowing_map_cell_obj, list
    ), "shadowing_map_cell_obj has not been processed yet"

    # Set shadow fading to 0dB
    for key in _g.shadowing_map_cell_obj.shadowing_maps_dB.keys():
        for m in range(len(_g.shadowing_map_cell_obj.shadowing_maps_dB[key])):
            _g.shadowing_map_cell_obj.shadowing_maps_dB[key][m] = np.zeros_like(
                _g.shadowing_map_cell_obj.shadowing_maps_dB[key][m]
            )

    # Deactivate fast fading
    _g.simulation_config_obj.debug_no_randomness = True  # Disable randomness for reproducibility
    return


def _set_prb_max(_g, prb_max_value: int = 1) -> None:
    # Set maximum number of PRBs to avoid memory issues
    _g.network_deployment_obj.df_ep[["dl_PRBs_available"]] = prb_max_value

    # Process time frequency resource object again to update PRB values in Fast Fading object
    _g.time_frequency_resource_obj.process()

    return


def _plot_heatmap(_g, kpi: str, scenario_model: str, save_fig: bool = False) -> None:
    """Scatter UE positions with KPI intensity (if available) + BS positions."""
    # Do heatmap using values as intensity
    x_pos_ue_m = _g.ue_deployment_obj.df_ep["position_x_m"].values
    y_pos_ue_m = _g.ue_deployment_obj.df_ep["position_y_m"].values

    x_pos_bs_m = _g.network_deployment_obj.df_ep["position_x_m"].values
    y_pos_bs_m = _g.network_deployment_obj.df_ep["position_y_m"].values

    plt.figure(figsize=(10, 8))

    try:
        values = getattr(_g.performance_obj, "r_e_" + f"{kpi}")[0]

        # if kpi == "best_serving_SSB_per_ue":
        if kpi == "best_serving_SSB_per_ue": 
            n_beam = 8
            values = np.mod(values, n_beam)

        plt.scatter(x_pos_ue_m, y_pos_ue_m, c=values, cmap="viridis", s=1)
        plt.colorbar(label=f"{kpi} Value")
    except:
        warning(f"Warning: KPI {kpi} not found in performance object. Plotting UE positions only.")
        plt.scatter(x_pos_ue_m, y_pos_ue_m, s=1)

    x_pos_bs_m, y_pos_bs_m = (
        _g.network_deployment_obj.df_ep["position_x_m"].values,
        _g.network_deployment_obj.df_ep["position_y_m"].values,
    )
    plt.scatter(x_pos_bs_m, y_pos_bs_m, marker="^", color="red", label="BSs", s=100)

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title(f"{scenario_model}\n{kpi}")
    # plt.xlim(-500, 500)
    # plt.ylim(-500, 500)
    plt.legend()
    plt.grid(True, ls="-.", alpha=0.5)
    if save_fig:
        fig_filename = f"ground_grid_heatmap_{scenario_model}_{kpi}.png"
        plt.savefig(fig_filename, dpi=1000)
        info(f"Figure saved as {fig_filename}")
    plt.show()


# =============================================================================
# Public API
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
                             grid_resol_m:int = 10, 
                            _customize_df_ep=None
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
    
    # Customize df_ep scenario
    if _customize_df_ep is not None:
        _customize_df_ep(_g)

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
    

    # Remove Shadowing
    _remove_randomness_shadow_and_fading(_g)

    # Set number of PRB to a max. Need for memory issues
    _set_prb_max(_g, prb_max_value = 1) 


    # Configure Simulation
    _g.configure(0, 1e6)

    # Run Simulation
    _g.run_simulation(0, 1e6)

    _activate_log()
    # Plot Heatmap
    for kpi in kpi_list if isinstance(kpi_list, list) else [kpi_list]:
        _plot_heatmap(_g, kpi=kpi, scenario_model=scenario_model, save_fig=save_fig)
    
    _reset_log_level(initial_level)

    return



@staticmethod
def _customize_df_ep(_g, dft_flag: bool = True) -> None:
    """
    Customize the df_ep (deployment parameters) dataframe.

    This function:
    - Disables BS transmit power for all but the last element (HAPS)
    - Adjusts position and beam configuration
    - Optionally exits early when DFT flag is enabled
    - Forces re-processing of dependent configuration objects

    Parameters
    ----------
    _g : object
        Global simulation object holding deployment and configuration data
    dft_flag : bool, optional
        If True, exits early after basic configuration (default: True)
    """

    # ------------------------------------------------------------------
    # Basic deployment customization
    # ------------------------------------------------------------------

    # Disable BS transmit power for all elements except the last one
    _g.network_deployment_obj.df_ep.loc[_g.network_deployment_obj.df_ep.index[0:-1], "BS_tx_power_max_dBm"] = -250#-np.inf

    # Set altitude (z-position) for the last element (HAPS)
    _g.network_deployment_obj.df_ep.loc[ _g.network_deployment_obj.df_ep.index[-1], "position_z_m"] = 20000

    # Default number of SSB beams (may be overridden later)
    n_ssb = 8

    # Early exit if DFT flag is enabled
    if dft_flag:
        return

    # ------------------------------------------------------------------
    # Beam customization
    # ------------------------------------------------------------------
    # Set SSB precoder type for the last element
    _g.network_deployment_obj.df_ep.loc[_g.network_deployment_obj.df_ep.index[-1], "SSB_precoder"] = "custom_vMB"

    # Optional CSI-RS precoder (currently disabled)
    # _g.network_deployment_obj.df_ep.loc[_g.network_deployment_obj.df_ep.index[-1], "CSI_RS_precoder"] = "DFT"

    # Beam angle definitions
    ssb_angle_theta_deg = str([0, -45, -45, -45,
                            45,  45,  45,  45 ])

    ssb_angle_phi_deg = str([0., -30.,  30.,  90.,
                            -90., -30.,  30.,  90.])

    # Active beam angle definitions (override previous ones)
    ssb_angle_theta_deg = [90., 90]
    ssb_angle_phi_deg   = [45, -45]

    # Update number of SSB beams based on angle definitions
    n_ssb = int(len(ssb_angle_theta_deg))

    # Sanity check on angle list lengths
    assert len(ssb_angle_phi_deg) == len(ssb_angle_theta_deg), \
        "SSB angle lists must have the same length"

    # SSB beam count configuration
    _g.network_deployment_obj.df_ep.at[_g.network_deployment_obj.df_ep.index[-1], "SSB_number_of_beams"] = n_ssb
    _g.network_deployment_obj.df_ep.at[_g.network_deployment_obj.df_ep.index[-1], "SSB_number_of_beams_H"] = -1
    _g.network_deployment_obj.df_ep.at[_g.network_deployment_obj.df_ep.index[-1], "SSB_number_of_beams_V"] = -1


    # SSB beam direction configuration
    _g.network_deployment_obj.df_ep.at[_g.network_deployment_obj.df_ep.index[-1], "SSB_BeamDirection_theta_deg"] = str(ssb_angle_theta_deg)

    _g.network_deployment_obj.df_ep.at[_g.network_deployment_obj.df_ep.index[-1], "SSB_BeamDirection_phi_deg"] = str(ssb_angle_phi_deg)

    # CSI-RS beam configuration placeholders (not used)
    # _g.network_deployment_obj.df_ep['CSI_RS_BeamDirection_phi_deg'] = ...
    # _g.network_deployment_obj.df_ep['CSI_RS_BeamDirection_theta_deg'] = ...


    # Re-process dependent configuration objects
    object_list_name = [
        'cell_SSB_conf_obj',
        'cell_CSI_RS_conf_obj',
        'power_consumption_obj',
        'cell_re_selection_conf_obj',
        'time_frequency_resource_obj',
        'mcs_resource_obj',
        'bler_vs_sinrs_lut_obj',
        'mi_obj' if _g.simulation_config_obj.sinr_mapping == "MIESM" else -1,
        'shadowing_map_cell_obj'
    ]

    for object_name in object_list_name:
        if object_name == -1:
            continue
        obj = getattr(_g, object_name)
        # Special handling for SSB configuration
        if object_name == 'cell_SSB_conf_obj':
            obj.TX_ver_active_number = np.full_like( obj.TX_ver_active_number, 2, dtype=int)
            obj.TX_hor_active_number = np.full_like(obj.TX_hor_active_number, 4, dtype=int)
            pass

        # Apply configuration changes
        obj.process()

    return


if __name__ == "__main__":
    _activate_log()
    
    preset = "GiuliaStd"

    # Define Example Parameters
    scenario_model = "3GPPTR38_811_Dense_Urban_singleHAPS_reflector_4G5G"  # "3GPPTR38_811_Dense_Urban_singleHAPS_UPA_4G5G" "3GPPTR38_901_5G"  # ITU_R_M2135_UMa
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

    kpi_list = ["SSB_sinr_ue_to_cell_dB", "SSB_usefulPower_ue_dBm"]
                # , "SSB_usefulPower_ue_dBm", "best_serving_SSB_per_ue"] # "SSB_sinr_ue_to_cell_dB", "SSB_usefulPower_ue_dBm", "SSB_interfPower_ue_dBm"

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
                                 grid_resol_m=10, 
                                 _customize_df_ep=_customize_df_ep
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
