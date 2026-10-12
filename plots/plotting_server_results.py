# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:15:15 2024

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@itea.upv.es
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

project_name = "ITU_R_M2135_UMa" + "_uniform" 
file_name = results_file(project_name, 'results-raw.npz')
a_load = np.load(file_name, allow_pickle=True)


    
# def plot_hist_helper(project_name, data_label, xlabel, data_legend=None):

#     data = a_load.get(data_label, None)  # Retorna None si no existe
#     if data is None:
#         print(f"The key '{data_label}' does not exist in a_load")
#     else: 
#         data = a_load[data_label]
#         averaged_data = np.mean(data, axis=(0, 1))
#         result = [averaged_data[~np.isnan(averaged_data)]]
        
#         if data_legend == None:
#             data_legend = data_label
            
#         color_string = 'r' + ''.join(['b' for _ in range(len(result) - 1)]) 
#         plotting.plot_hist(project_name,
#                            result,
#                            xlabel,
#                            [data_legend],
#                            color_string)
        
def plot_hist_helper(project_name, data_label, xlabel, data_legend=None, group_size=1):
    data = a_load.get(data_label, None)  # Retorna None si no existe
    if data is None:
        print(f"The key '{data_label}' does not exist in a_load")
        return

    # default legend base
    base_legend = data_legend if data_legend is not None else data_label

    if group_size == 1:
        # original behavior
        data = a_load[data_label]
        averaged_data = np.mean(data, axis=(0, 1))
        result = [averaged_data[~np.isnan(averaged_data)]]
        legends = [base_legend]
        plot_hist_helper_data(project_name, result, xlabel, legends)
    else:
        # grouped: make a different histogram per group
        data = np.asarray(a_load[data_label])           # expects (episodes, 1, samples)
        n_episodes = data.shape[0]

        for start in range(0, n_episodes, group_size):
            end = min(start + group_size, n_episodes)
            group_slice = data[start:end, :, :]         # shape (g, 1, samples)
            averaged_data = np.mean(group_slice, axis=(0, 1))  # -> (samples,)
            result = [averaged_data[~np.isnan(averaged_data)]]
            legends = [f"{base_legend} (episodes {start}-{end-1})"]

            # separate figure/file per group via project_name suffix
            plot_hist_helper_data(f"{project_name}_ep_{start}-{end-1}",
                                  result, xlabel, legends)


def plot_hist_helper_data(project_name, data, xlabel, data_legend):
    result = data
    color_string = 'r' + ''.join(['b' for _ in range(len(result) - 1)])
    plotting.plot_hist(project_name,
                       result,
                       xlabel,
                       data_legend,
                       color_string)
        
    
# def plot_CDF_helper(project_name, data_label, xlabel, data_legend=None):

#     data = a_load.get(data_label, None)  # Retorna None si no existe
#     if data is None:
#         print(f"The key '{data_label}' does not exist in a_load")
#     else:  
#         data = a_load[data_label]
#         result = [data[~np.isnan(data)]]
        
#         if data_legend == None:
#             data_legend = data_label
            
#         color_string = 'r' + ''.join(['b' for _ in range(len(result) - 1)]) 
#         plotting.plot_CDF(project_name,
#                           result,
#                           xlabel,
#                           [data_legend],
#                           1000,
#                           color_string) 
        
def plot_CDF_helper(project_name, data_label, xlabel, data_legend=None, group_size=1):
    data = a_load.get(data_label, None)  # Retorna None si no existe
    if data is None:
        print(f"The key '{data_label}' does not exist in a_load")
        return

    data = a_load[data_label]
    base_legend = data_legend if data_legend is not None else data_label

    results = []
    legends = []

    if group_size == 1:
        # original behavior
        results = [data[~np.isnan(data)]]
        legends = [base_legend]
    else:
        # grouped by episodes (expects shape (episodes, 1, samples))
        data = np.asarray(data)
        n_episodes = data.shape[0]
        for start in range(0, n_episodes, group_size):
            end = min(start + group_size, n_episodes)
            group = data[start:end, 0, :].reshape(-1)
            group = group[~np.isnan(group)]
            results.append(group)
            legends.append(f"{base_legend} (episodes {start}-{end-1})")
            
    # delegate plotting & colors to the helper below
    plot_CDF_helper_data(project_name, results, xlabel, legends)                
        
def plot_CDF_helper_data(project_name, results, xlabel, legends):
       
    # Plotting with special x-axis limits for throughput data
    # if "UE throughput" in xlabel:
    #     xlim = (0, 100)
    # else:
    #     xlim = None  # No specific xlim if not throughput  
        
    color_string = 'r' + ''.join(['b' for _ in range(len(results) - 1)]) 
    plotting.plot_CDF(project_name,
                      results,
                      xlabel,
                      legends,
                      1000,
                      color_string)
                      #xlim)     
    
# def plot_CDF_helper_bench(project_name, beam_type, data_label, xlabel, data_legend=None):

#     if "coupling" in data_label:
#         result = plotting.get_coupling_loss_benchmark(project_name, beam_type) 
#     else:
#         result = plotting.get_geometry_sinr_benchmark(project_name, beam_type)         
        
#     data = a_load.get(data_label, None)  # Retorna None si no existe
#     if data is None:
#         print(f"The key '{data_label}' does not exist in a_load")
#     else: 
#         result.insert(0, data[~np.isnan(data)])
        
#         if data_legend == None:   
#            data_legend = [data_label] + [f"benchmark_{i}_{project_name}" for i in range(len(result)-1)]
        
#         color_string = 'r' + ''.join(['b' for _ in range(len(result) - 1)]) 
#         plotting.plot_CDF(project_name,
#                           result,
#                           xlabel,
#                           data_legend,
#                           1000,
#                           color_string)   

def plot_CDF_helper_bench(project_name, beam_type, data_label, xlabel, data_legend=None, group_size=1):

    # Get benchmark curves (same as before)
    if "coupling" in data_label:
        bench_curves = plotting.get_coupling_loss_benchmark(project_name, beam_type)
    else:
        bench_curves = plotting.get_geometry_sinr_benchmark(project_name, beam_type)

    # Load data (same check/message as before)
    data = a_load.get(data_label, None)  # Retorna None si no existe
    if data is None:
        print(f"The key '{data_label}' does not exist in a_load")
    else:
        # Build our curves (grouped or not)
        if group_size == 1:
            my_results = [data[~np.isnan(data)]]
            my_legends = [data_label]
        else:
            data = np.asarray(data)  # expects shape (episodes, 1, samples)
            n_episodes = data.shape[0]
            my_results, my_legends = [], []
            for start in range(0, n_episodes, group_size):
                end = min(start + group_size, n_episodes)
                group = data[start:end, 0, :].reshape(-1)
                group = group[~np.isnan(group)]
                my_results.append(group)
                my_legends.append(f"{data_label} (episodes {start}-{end-1})")

        # Combine our curves + benchmark curves
        result = my_results + bench_curves

        # Legends: keep original behavior
        if data_legend is None:
            bench_legends = [f"benchmark_{i}_{project_name}" for i in range(len(bench_curves))]
            legends = my_legends + bench_legends
        else:
            legends = data_legend  # caller-provided (same as before)

        # Delegate plotting & colors
        plot_CDF_helper_data(project_name, result, xlabel, legends)

    
#######################################################################
#######################################################################
# Plot RELATED INFORMATION TO SERVER LINKS ONLY
#######################################################################
#######################################################################

#################################################
# Plot UE per cell 
#################################################

plot_hist_helper(project_name, "ues_per_cell", "UEs per cell [.]",group_size=5)

plot_hist_helper(project_name, "ues_per_SSB_beam", "UEs per beam [.]")


#################################################
# Plot 3D distance to server 
#################################################

beam_type = "CRS"

plot_CDF_helper(project_name, "best_serving_CRS_distance_3d_m", "UE distance to serving BS [m] (based on " + beam_type + ")")

beam_type = "SSB"

plot_CDF_helper(project_name, "best_serving_SSB_distance_3d_m", "UE distance to serving BS [m] (based on " + beam_type + ")")
    
    #################################################
    # Plot coupling gain to server 
    #################################################
    
if project_name != '3GPPTR38_901_UMa_C1_uniform' and project_name != '3GPPTR38_901_UMa_C2_uniform' and\
    project_name != '3GPPTR38_901_UMi_C1_uniform' and project_name != '3GPPTR38_901_UMi_C2_uniform' and\
    project_name != '3GPPTR38_901_UMa_lsc_sn_uniform' and project_name != 'Dataset_uniform':

    beam_type = "CRS"
    
    plot_CDF_helper_bench(project_name, "antenna element", "best_serving_CRS_coupling_gain_dB", "Coupling [dB] (based on " + beam_type + ")")
    
    beam_type = "SSB"
    
    plot_CDF_helper_bench(project_name, beam_type, "best_serving_SSB_coupling_gain_dB", "Coupling [dB] (based on " + beam_type + ")")        
     
       
#################################################
# Plot RSRP  
#################################################

beam_type = "CRS"

plot_CDF_helper(project_name, "best_serving_CRS_rsrp_per_ue_dBm", "UE RSRP [dBm] (based on " + beam_type + ")")

beam_type = "SSB"

plot_CDF_helper(project_name, "best_serving_SSB_rsrp_per_ue_dBm", "UE RSRP [dB] (based on " + beam_type + ")")

beam_type = "CSI_RS"

plot_CDF_helper(project_name, "best_serving_CSI_RS_rsrp_per_ue_dBm", "UE RSRP [dB] (based on " + beam_type + ")")


#################################################
# Plot SINR  
#################################################

beam_type = "CRS"

plot_CDF_helper_bench(project_name, "antenna element", "CRS_sinr_ue_to_cell_dB", "UE SINR [dB] (based on " + beam_type + ")")

beam_type = "SSB"

plot_CDF_helper_bench(project_name, beam_type, "SSB_sinr_ue_to_cell_dB", "UE SINR [dB] (based on " + beam_type + ")")

beam_type = "CSI_RS"

#plot_CDF_helper_bench(project_name, beam_type, "CSI_RS_sinr_ue_to_cell_dB", "UE SINR [dB] (based on " + beam_type + ")")
#plot_CDF_helper(project_name, "CSI_RS_sinr_per_PRB_ue_to_cell_dB", "UE PRB SINR [dB] (based on " + beam_type + ")")
plot_CDF_helper(project_name, "effective_CSI_RS_sinr_ue_to_cell_dB", "UE effective SINR [dB] (based on " + beam_type + ")")


#################################################
# Plot UE rates
#################################################

#plot_CDF_helper(project_name, "ue_throughput_based_on_avg_CSI_RS_SINR_Mbps", "UE throughput [Mbps]")
plot_CDF_helper(project_name, "ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps", "UE throughput [Mbps]",group_size=5)
plot_CDF_helper(project_name, "ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps", "UE throughput [Mbps]")    


# #################################################
# # Plot cell rates
# #################################################

plot_CDF_helper(project_name, "cell_throughput_Mbps", "Cell throughput [Mbps]")


#################################################
# Plot carrier rates
#################################################

to_plot = a_load["carrier_throughput_Mbps"]
legends = ["carrier_" + str(row) + "_throughput_Mbps" for row in range(to_plot.shape[2])] #Size: Episod, steps, carriers
results = [to_plot[:,:, i].flatten() for i in range(to_plot.shape[2])]
plot_CDF_helper_data(project_name, results, "Carrier rate [Mbps]", legends) 

to_plot = a_load["ue_throughput_per_carrier_Mbps"]
legends = ["carrier_" + str(row) + "_ue_throughput_Mbps" for row in range(to_plot.shape[2])] #Size: Episod, steps, carriers, UEs
results = [to_plot[:, :, i, :].flatten() for i in range(to_plot.shape[2])]
plot_CDF_helper_data(project_name, results, "UE throughput per carrier [Mbps]", legends) 


#################################################
# Energy
#################################################

plot_CDF_helper(project_name, "total_network_power_consumption_kW", "Network power consumption [kW]")