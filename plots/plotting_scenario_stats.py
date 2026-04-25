# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:15:15 2024

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import os
import sys

import numpy as np

# Configure Giulia into path
plots_dir = os.path.dirname(__file__)
root_dir = os.path.join(plots_dir, '..')
sys.path.insert(1, root_dir)

from giulia.fs import results_file
from giulia.plots import plotting

project_name = "3GPPTR38_901_4G5G_multilayer" + "_uniform"
folder_name = project_name

#################################################
# Plot scenario 
#################################################

a_load = np.load(results_file(project_name, 'to_plot_scenario.npz'), allow_pickle=True)
cell_site_positions_m = a_load["cell_site_positions_m"]
site_names = a_load["site_names"]
isd_m = a_load["isd_m"]
plotting.plot_scenario(project_name, isd_m, site_names, cell_site_positions_m) 


#################################################
# Plot BS antennas
#################################################

a_load = np.load(results_file(project_name, 'to_plot_cell_antenna_arrays.npz'))
node_type = "cell"
antenna_to_node_mapping = a_load["antenna_to_node_mapping"]
antenna_element_GCS_position_m = a_load["antenna_element_GCS_position_m"]
plotting.plot_antenna_arrays(project_name, node_type, antenna_to_node_mapping, antenna_element_GCS_position_m)


#################################################
# Plot bler versus SINR tables
#################################################

a_load = np.load(results_file(project_name, 'to_plot_lut_bler_vs_sinr.npz'))
modulation_and_coding_schemes = a_load["modulation_and_coding_schemes"]
bler_per_sinr_mcs = a_load["bler_per_sinr_mcs"]
plotting.plot_luts(project_name, modulation_and_coding_schemes, bler_per_sinr_mcs) 


#################################################
# Plot UE deployment 
#################################################

a_load = np.load(results_file(project_name, 'to_plot_ue_deployment.npz'), allow_pickle=True)
site_names = a_load["site_names"]
cell_site_positions_m = a_load["cell_site_positions_m"]
isd_m = a_load["isd_m"]
ue_position_m =a_load["ue_position_m"]
hot_spot_position_m = a_load["hotspot_position_m"]
plotting.plot_ue_locations(project_name, isd_m, site_names, cell_site_positions_m, ue_position_m, hot_spot_position_m)   


#################################################
# Plot UE antennas
#################################################

a_load = np.load(results_file(project_name, 'to_plot_ue_antenna_arrays.npz'))
node_type = "ue"
antenna_to_node_mapping = a_load["antenna_to_node_mapping"]
antenna_element_GCS_position_m = a_load["antenna_element_GCS_position_m"]
plotting.plot_antenna_arrays(project_name, node_type, antenna_to_node_mapping, antenna_element_GCS_position_m)


#######################################################################
#######################################################################
# Plot RELATED INFORMATION TO ALL LINKS
#######################################################################
#######################################################################


#################################################
# Plot distances
#################################################

a_load = np.load(results_file(project_name, 'to_plot_distances_angles.npz'))

to_plot = [a_load["distances_b_to_a_2d_m"], a_load["distance_b_to_a_2d_wraparound_m"], a_load["distances_b_to_a_3d_m"], a_load["distance_b_to_a_3d_wraparound_m"]]
plotting.plot_CDF(project_name, to_plot, "Distance [m]",
                  ["distances_2d_m", "distances_2d_wraparound_m", "distances_3d_m", "distances_3d_wraparound_m"],
                  500)

to_plot = [a_load["azimuths_b_to_a_degrees"], a_load["azimuths_b_to_a_wraparound_degrees"]]
plotting.plot_CDF(project_name, to_plot, "Azimuth [degrees]", ["azimuths_degrees", "azimuths_wraparound_degrees"],500) 

to_plot = [a_load["zeniths_b_to_a_degrees"], a_load["zeniths_b_to_a_wraparound_degrees"]]
plotting.plot_CDF(project_name, to_plot, "Zenith [degrees]", ["zeniths_degrees", "zeniths_wraparound_degrees"],500) 


#################################################
# Plot antenna pattern 
#################################################

a_load = np.load(results_file(project_name, 'to_plot_antenna_pattern.npz'))
plotting.plot_antenna_gain_4_figures(np.radians(a_load["azimuth_deg"]), a_load["A_h_dB"], [-np.pi, np.pi], 
                                     np.radians(a_load["zenith_deg"]), a_load["A_v_dB"], [0, np.pi], 
                                      'Antenna pattern gain')  


#################################################
# Plot antenna pattern gain
#################################################

a_load = np.load(results_file(project_name, 'to_plot_antenna_pattern_gains.npz'))

to_plot = [a_load["antenna_pattern_gain_b_to_a_dB"]]
plotting.plot_CDF(project_name, to_plot, "Antenna pattern gain [dB]", ["antenna_pattern_gain_dB"], 500) 


if project_name != '3GPPTR38_901_UMa_C1_uniform' and project_name != '3GPPTR38_901_UMa_C2_uniform' and\
    project_name != '3GPPTR38_901_UMi_C1_uniform' and project_name != '3GPPTR38_901_UMi_C2_uniform' and\
    project_name != '3GPPTR38_901_UMa_lsc_sn_uniform': 

    #################################################
    # Plot array steering vector gain
    #################################################
    
    a_load = np.load(results_file(project_name, 'to_plot_array_steering_vector_gains.npz'))
    
    to_plot = [a_load["array_steering_vector_gain_dB"]]
    plotting.plot_CDF(project_name, to_plot, "Antenna array gain [dB]", ["antenna_array_gain_dB"], 500)    
    
    
    #################################################
    # Plot LoS
    #################################################
    
    a_load = np.load(results_file(project_name, 'to_plot_LoS.npz'))
    
    to_plot = [a_load["LoS_b_to_a"]]
    plotting.plot_PMF(project_name, to_plot, "LoS",["LoS", "NLoS"]) 
    
    
    #################################################
    # Plot K-factor
    #################################################
    
    a_load = np.load(results_file(project_name, 'to_plot_K_factor.npz'))
    
    to_plot = [a_load["K_factor_b_to_a_dB"]]
    plotting.plot_CDF(project_name, to_plot, "K factor [dB] (outdoor UE only)", ["K_factor_dB"], 500)
    
    
    #################################################
    # Plot path loss 
    #################################################
    
    a_load = np.load(results_file(project_name, 'to_plot_path_loss.npz'))
    
    to_plot = [a_load["path_loss_b_to_a_dB"]]
    plotting.plot_CDF(project_name, to_plot, "path loss [dB]", ["path_loss_dB"], 500)
    
    
    #################################################
    # Plot Outdoor to Indoor penetration loss
    #################################################
    
    a_load = np.load(results_file(project_name, 'to_plot_o2i_penetration_loss.npz'))
    
    to_plot = [a_load["o2i_penetration_losses_b_to_a_dB"]]
    plotting.plot_CDF(project_name, to_plot, "O2I penetration loss [dB] (indoor UEs only)", ["o2i_penetration_loss_dB"], 500)
    
    
    #################################################
    # Plot shadowing gain
    #################################################
    
    a_load = np.load(results_file(project_name, 'to_plot_shadowing_gain.npz'))
    
    to_plot = [a_load["shadowing_gain_b_to_a_dB"]]
    plotting.plot_CDF(project_name, to_plot, "Shadowing gain [dB]", ["shadowing_gain_dB"],500)
    
    
    #################################################
    # Plot slow channel gain
    #################################################
    
    a_load = np.load(results_file(project_name, 'to_plot_slow_channel_gain.npz'))
    
    to_plot = [a_load["slow_channel_gain_b_to_a_dB"]]
    plotting.plot_CDF(project_name, to_plot, "Slow channel gain [dB]", ["slow_channel_gain_dB"], 500)
    
    
    #################################################
    # Plot fast fading channel gain
    #################################################
    
    a_load = np.load(results_file(project_name, 'to_plot_fast_fading_gain.npz'))
    
    to_plot = [a_load["fast_fading_gain_bAnt_to_aAnt_dB"]]
    plotting.plot_CDF(project_name, to_plot, "Fast fading gain [dB]", ["fast_fading_gain_dB"], 500)