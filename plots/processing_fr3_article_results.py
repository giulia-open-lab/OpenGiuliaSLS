# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 12:45:28 2024

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
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
    num_colors = 7
    cmap = plt.get_cmap('tab10')
    color_string = [cmap(i) for i in np.linspace(0, 1, num_colors)]
    
    # Plotting with special x-axis limits for throughput data
    if "ue_throughput" in data_label.lower():
        xlim = (0, 1000)
    else:
        xlim = None  # No specific xlim if not throughput      

    plotting.plot_CDF(project_name, result, xlabel, legends, 10000, color_string, xlim)   
    
    
def plot_bar_helper(project_name, data_label, xlabel, data_legend=None):
    
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
    num_colors = 7
    cmap = plt.get_cmap('tab10')
    color_string = [cmap(i) for i in np.linspace(0, 1, num_colors)]

    plotting.plot_avg_std_bars(project_name,result,xlabel,legends,color_string)
      
    
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

for project_name in project_name_list:
    result = load_results(project_name)
    if result is not None:
        a_load.append(result)
        
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