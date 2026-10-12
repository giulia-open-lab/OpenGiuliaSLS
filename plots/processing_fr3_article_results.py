# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 12:45:28 2024

@author: Usuario
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Configure Giulia into path
plots_dir = os.path.dirname(__file__)
root_dir = os.path.join(plots_dir, '..')
sys.path.insert(1, root_dir)

from giulia.fs import results_file
from giulia.plots import plotting


def load_results(project_name):
    
    file_path = results_file(project_name, 'results-raw.npz')
    
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        print(f"File {file_path} not found.")
        return None
        
def plot_CDF_helper(project_name, data_label, xlabel, data_legend=None):
    
    # Initialize a list to store data for CDF plotting
    result = []
    legends = []
    
    # Iterate through each dataset in a_load
    for i, data in enumerate(a_load):
        
        if data_label in data:
            
            # Extract and clean data (removing NaNs)
            cleaned_data = data[data_label].flatten()
            cleaned_data = cleaned_data[cleaned_data>0]
            result.append(cleaned_data)
           
            # Set the legend for the current data series
            if data_legend is None:
                legends.append(f"{data_label} - Dataset {i+1}")
            else:
                legends.append(data_legend[i] if isinstance(data_legend, list) else data_legend)
       
        else:
            print(f"Warning: {data_label} not found in dataset {i+1}")
    
    if not result:
        
        print(f"Error: No valid data found for label '{data_label}'")
        return

    # Generate color string
    #color_string = ['r'] + ['b' for _ in range(len(a_load) - 1)]
    num_colors = 7
    cmap = plt.get_cmap('tab10')  # 'tab10' is a good colormap for qualitative data
    color_string = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    
    # Plotting with special x-axis limits for throughput data
    if "ue_throughput" in data_label.lower():
        xlim = (0, 1000)
    else:
        xlim = None  # No specific xlim if not throughput      

    # Call the plotting function
    plotting.plot_CDF(project_name,
                      result,
                      xlabel,
                      legends,
                      10000,
                      color_string,
                      xlim)   
    
    
def plot_bar_helper(project_name, data_label, xlabel, data_legend=None):
    
    # Initialize a list to store data for CDF plotting
    result = []
    legends = []
    
    # Iterate through each dataset in a_load
    for i, data in enumerate(a_load):
        
        if data_label in data:
            
            # Extract and clean data (removing NaNs)
            cleaned_data = data[data_label].flatten()
            result.append(cleaned_data)
           
            # Set the legend for the current data series
            if data_legend is None:
                legends.append(f"{data_label} - Dataset {i+1}")
            else:
                legends.append(data_legend[i] if isinstance(data_legend, list) else data_legend)
       
        else:
            print(f"Warning: {data_label} not found in dataset {i+1}")
    
    if not result:
        
        print(f"Error: No valid data found for label '{data_label}'")
        return

    # Generate color string
    #color_string = ['r'] + ['b' for _ in range(len(a_load) - 1)]
    num_colors = 7
    cmap = plt.get_cmap('tab10')  # 'tab10' is a good colormap for qualitative data
    color_string = [cmap(i) for i in np.linspace(0, 1, num_colors)]

    # Call the plotting function
    plotting.plot_avg_std_bars(project_name,
                               result,
                               xlabel,
                               legends,
                               color_string)
      
    
# Load the results
a_load = []

project_name_list = [
    "3GPPTR38_901_4G_inhomogeneous_per_cell",
    "3GPPTR38_901_5G_inhomogeneous_per_cell",
    "3GPPTR38_901_4G5G_multilayer_inhomogeneous_per_cell",
    "3GPPTR38_901_4G_5G2_multilayer_inhomogeneous_per_cell",
    "3GPPTR38_901_4G_5G_multilayer_inhomogeneous_per_cell",  
    "3GPPTR38_901_4G_5G6G_multilayer_inhomogeneous_per_cell",
    "3GPPTR38_901_4G_5G_6G_multilayer_inhomogeneous_per_cell"
]

label_name_list = [
    "4GUMa",
    "5GUMa",
    "[4GUMa+5GUMa]",
    "4GUMa+5GUMi(UMaBS)",  
    "4GUMa+5GUMi",  
    "4GUMa+[5GUMi+6GUmi]",
    "4GUMa+5GUMi+6GHS",
]

# project_name_list = [
#     "3GPPTR38_901_4G_5G6G_multilayer_inhomogeneous_per_cell_noRel",
#     "3GPPTR38_901_4G_5G_6G_multilayer_inhomogeneous_per_cell_noRel",    
#     "3GPPTR38_901_4G_5G6G_multilayer_inhomogeneous_per_cell",
#     "3GPPTR38_901_4G_5G_6G_multilayer_inhomogeneous_per_cell"
# ]

# label_name_list = [
#     "4GUMa+[5GUMi+6GUmi] (noRel)",
#     "4GUMa+5GUMi+6GHS (noRel)",   
#     "4GUMa+[5GUMi+6GUmi]",
#     "4GUMa+5GUMi+6GHS"
# ]

for project_name in project_name_list:
    result = load_results(project_name)
    if result is not None:
        a_load.append(result)
        
        
    # CRS_RSRP_no_fast_fading_ue_to_cell_dBm = r_CRS_RSRP_no_fast_fading_ue_to_cell_dBm, 
    # best_serving_cell_per_ue_based_on_CRS = r_best_serving_cell_per_ue_based_on_CRS,  
    # best_serving_CRS_rsrp_per_ue_dBm = r_best_serving_CRS_rsrp_per_ue_dBm, 
    # best_serving_CRS_re_noise_per_ue_dBm = r_best_serving_CRS_re_noise_per_ue_dBm,
    # cell_activity_per_ue = r_cell_activity_per_ue, 
    # ues_per_cell = r_ues_per_cell, 
    # CRS_sinr_ue_to_cell_dB = r_CRS_sinr_ue_to_cell_dB, 

    # SSB_RSRP_no_fast_fading_ue_to_cell_dBm = r_SSB_RSRP_no_fast_fading_ue_to_cell_dBm,   
    # SSB_RSS_per_PRB_ue_to_cell_dBm = r_SSB_RSS_per_PRB_ue_to_cell_dBm, 
    # best_serving_cell_per_ue_based_on_SSB = r_best_serving_cell_per_ue_based_on_SSB,
    # best_serving_SSB_per_ue = r_best_serving_SSB_per_ue, 
    # best_serving_SSB_rsrp_per_ue_dBm = r_best_serving_SSB_rsrp_per_ue_dBm, 
    # best_serving_SSB_re_noise_per_ue_dBm = r_best_serving_SSB_re_noise_per_ue_dBm,  
    # SSB_beam_activity_per_ue = r_SSB_beam_activity_per_ue, 
    # ues_per_SSB_beam = r_ues_per_SSB_beam,
    # SSB_sinr_ue_to_cell_dB = r_SSB_sinr_ue_to_cell_dB,
    
    # CSI_RS_RSRP_no_fast_fading_ue_to_cell_dBm = r_CSI_RS_RSRP_no_fast_fading_ue_to_cell_dBm, 
    # CSI_RS_RSS_per_PRB_ue_to_cell_dBm = r_CSI_RS_RSS_per_PRB_ue_to_cell_dBm,            
    # best_serving_CSI_RS_per_ue = r_best_serving_CSI_RS_per_ue, 
    # best_serving_CSI_RS_rsrp_per_ue_dBm = r_best_serving_CSI_RS_rsrp_per_ue_dBm, 
    # best_serving_CSI_RS_re_noise_per_ue_dBm = r_best_serving_CSI_RS_re_noise_per_ue_dBm , 
    # CSI_RS_beam_activity_per_ue = r_CSI_RS_beam_activity_per_ue, 
    # ues_per_CSI_RS_beam = r_ues_per_CSI_RS_beam, 
    # CSI_RS_sinr_ue_to_cell_dB = r_CSI_RS_sinr_ue_to_cell_dB,
    # CSI_RS_sinr_per_PRB_ue_to_cell_dB = r_CSI_RS_sinr_per_PRB_ue_to_cell_dB,
    # effective_CSI_RS_sinr_ue_to_cell_dB = r_effective_CSI_RS_sinr_ue_to_cell_dB, 
    # ue_throughput_based_on_avg_CSI_RS_SINR_Mbps = r_ue_throughput_based_on_avg_CSI_RS_SINR_Mbps, 
    # ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps = r_ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps, 
    # ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps = r_ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps, 
    
    # network_power_consumption_kW = r_network_power_consumption_kW)
     
    
#######################################################################
#######################################################################
# Plot RELATED INFORMATION TO SERVER LINKS ONLY
#######################################################################
#######################################################################


# # #################################################
# # # Plot 3D distance to server 
# # #################################################

# # beam_type = "CRS"

# # plot_CDF_helper(project_name, "best_serving_CRS_distance_3d_m", "UE distance to serving BS [m] (based on " + beam_type + ")", label_name_list)

# beam_type = "SSB"

# plot_CDF_helper(project_name, "best_serving_SSB_distance_3d_m", "UE distance to serving BS [m] (based on " + beam_type + ")", label_name_list)
    
# # #################################################
# # # Plot coupling gain to server 
# # #################################################
    
# # beam_type = "CRS"
    
# # plot_CDF_helper(project_name, "best_serving_CRS_coupling_gain_dB", "Coupling [dB] (based on " + beam_type + ")", label_name_list)
    
# beam_type = "SSB"
    
# plot_CDF_helper(project_name, "best_serving_SSB_coupling_gain_dB", "Coupling [dB] (based on " + beam_type + ")", label_name_list)        
     
       
# # #################################################
# # # Plot RSRP  
# # #################################################

# # beam_type = "CRS"

# # plot_CDF_helper(project_name, "best_serving_CRS_rsrp_per_ue_dBm", "UE RSRP [dBm] (based on " + beam_type + ")", label_name_list)

# beam_type = "SSB"

# plot_CDF_helper(project_name, "best_serving_SSB_rsrp_per_ue_dBm", "UE RSRP [dB] (based on " + beam_type + ")", label_name_list)

# beam_type = "CSI_RS"

# plot_CDF_helper(project_name, "best_serving_CSI_RS_rsrp_per_ue_dBm", "UE RSRP [dB] (based on " + beam_type + ")", label_name_list)


# # #################################################
# # # Plot SINR  
# # #################################################

# # beam_type = "CRS"

# # plot_CDF_helper(project_name, "CRS_sinr_ue_to_cell_dB", "UE SINR [dB] (based on " + beam_type + ")", label_name_list)

# beam_type = "SSB"

# plot_CDF_helper(project_name, "SSB_sinr_ue_to_cell_dB", "UE SINR [dB] (based on " + beam_type + ")", label_name_list)

# beam_type = "CSI_RS"

# plot_CDF_helper(project_name, "CSI_RS_sinr_ue_to_cell_dB", "UE SINR [dB] (based on " + beam_type + ")", label_name_list)
# #plot_CDF_helper(project_name, "CSI_RS_sinr_per_PRB_ue_to_cell_dB", "UE PRB SINR [dB] (based on " + beam_type + ")", label_name_list)
# plot_CDF_helper(project_name, "effective_CSI_RS_sinr_ue_to_cell_dB", "UE effective SINR [dB] (based on " + beam_type + ")", label_name_list)


#################################################
# Plot UE rates
#################################################

#plot_CDF_helper(project_name, "ue_throughput_based_on_avg_CSI_RS_SINR_Mbps", "UE throughput [Mbps]", label_name_list)
plot_CDF_helper(project_name, "ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps", "UE throughput [Mbps]", label_name_list)
plot_CDF_helper(project_name, "ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps", "UE throughput [Mbps]", label_name_list)    


# #################################################
# # Plot cell rates
# #################################################

plot_CDF_helper(project_name, "cell_throughput_Mbps", "Cell throughput [Mbps]", label_name_list)


#################################################
# Energy
#################################################

plot_bar_helper(project_name, "network_power_consumption_kW", "Network power consumption [kW]", label_name_list)