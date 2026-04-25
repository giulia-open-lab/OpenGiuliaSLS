# -*- coding: utf-8 -*-
"""
Created on Mon May 12 19:06:17 2025

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import os
import sys
# Configure Giulia into path
examples_dir = os.path.dirname(__file__)
root_dir = os.path.join(examples_dir, '..')
sys.path.insert(1, root_dir)

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from scipy.stats import t
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon


color_dict_var = {
    10:"k", 
    15:"darkred",
    20:"darkblue"
    }

ls_dict_var = {
    10:":", 
    15:"--",
    20:"-"
    }


#### LOAD MODULE ####
#####################        
def load_results(UE_density_list, var_param_perc_list, num_episods) -> dict:
    dict_opt_results = {}
    
    dict_giuliaSls_results = {}

    for ue_density_value in UE_density_list:
        for var_perc_value in var_param_perc_list:
            # Upload Optimizer results
            folder_path = f"../outputs/results/Opt_EE_UserAssociation_4G/{scenario_model}_{ue_distribution}/UEdensity_{ue_density_value}_varParam_{var_perc_value}/"  
            for episode in range(num_episods):
                file_path = folder_path + f"opt/results_opt_EE_userAssociation_singleStep_{episode}.npz"
                # Upload
                dict_opt_results[f"ueDens_{ue_density_value}_varPerc_{var_perc_value}_ep_{episode}"] = np.load(file_path)
                del file_path

            # Upload Giulia SLS results
            file_path = folder_path + "giulia_sls/results-raw.npz"
            # Upload
            dict_giuliaSls_results[f"ueDens_{ue_density_value}_varPerc_{var_perc_value}"]  =  np.load(file_path)

    return dict_opt_results, dict_giuliaSls_results


def calculate_mean_and_ci(data, confidence=0.95):
    mean = np.mean(data, axis=0)
    std_error = np.std(data, ddof=1, axis=0) / np.sqrt(data.shape[0])
    ci = std_error * t.ppf((1 + confidence) / 2, df=data.shape[0] - 1)
    return mean, ci


def plot_netPower_consumption_overTime(name_fig, dict_giuliaSls_results, ue_density_value, var_param_perc_list, enable_save=True):
    for var_perc_value in var_param_perc_list:
        name_key = f"ueDens_{ue_density_value}_varPerc_{var_perc_value}"
        
        # Take data
        network_power_consumption_kW = dict_giuliaSls_results[name_key]["network_power_consumption_kW"]
        
        # Compute mean and confidence interval
        mean_value, ci_value = calculate_mean_and_ci(network_power_consumption_kW.squeeze())
        
        time_steps = np.arange(1, 1 + mean_value.shape[0])
        
        plt.plot(
            time_steps,
            mean_value,
            label=f"Variance {var_perc_value}%",
            linewidth=2, 
            ls=ls_dict_var[var_perc_value], 
            color=color_dict_var[var_perc_value]
        )
        plt.fill_between(
            time_steps,
            mean_value - ci_value,
            mean_value + ci_value,
            alpha=0.3, 
            color=color_dict_var[var_perc_value]
        )

    plt.title("Network Power Consumption Over Time", fontsize=14)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("Power Consumption (kW)", fontsize=12)
    plt.grid(True, which='both', linestyle='-.', linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./figures/{name_fig}.png", format='png')
    plt.show()
        
    
def plot_netPower_savingPercentage(name_fig, dict_giuliaSls_results, UE_density_list, var_param_perc_list, enable_save=True):
    for var_perc_value in var_param_perc_list:
        mean_values = []
        ci_values = []
        
        for ue_density_value in UE_density_list:
            name_key = f"ueDens_{ue_density_value}_varPerc_{var_perc_value}"
            
            network_power_consumption_kW = dict_giuliaSls_results[name_key]["network_power_consumption_kW"]
            diff_power_cons_perc = 100 * ((network_power_consumption_kW[:, -1] - network_power_consumption_kW[:, 0]) / network_power_consumption_kW[:, 0])
            mean_value, ci_value = calculate_mean_and_ci(diff_power_cons_perc)
            mean_values.append(mean_value)
            ci_values.append(ci_value)
        
        plt.errorbar(
            UE_density_list,
            mean_values,
            yerr=ci_values,
            capsize=5,
            label=f"Var. {var_perc_value}%",
            color=color_dict_var[var_perc_value],
            linewidth=1,
            ls=ls_dict_var[var_perc_value]
        )

    plt.xticks(UE_density_list, UE_density_list)
    plt.title("Network Power Consumption Percentage Saving", fontsize=14)
    plt.xlabel("Number of UEs", fontsize=12)
    plt.ylabel("%", fontsize=12)
    plt.grid(True, which='both', linestyle='-.', linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./figures/{name_fig}.png", format='png')
    plt.show()
    
    
def plot_netThp_diffPercentage(name_fig, dict_giuliaSls_results, UE_density_list, var_param_perc_list, enable_save=True):
    for var_perc_value in var_param_perc_list:
        mean_values = []
        ci_values = []
        
        for ue_density_value in UE_density_list:
            name_key = f"ueDens_{ue_density_value}_varPerc_{var_perc_value}"
            
            # Take data
            sum_cell_network_throughput_Mbps = dict_giuliaSls_results[name_key]["cell_throughput_Mbps"].sum(2)
            
            # Compute
            diff_power_cons_perc = 100 * ((sum_cell_network_throughput_Mbps[:, -1] - sum_cell_network_throughput_Mbps[:, 0])/ sum_cell_network_throughput_Mbps[:, 0])
            mean_value, ci_value = calculate_mean_and_ci(diff_power_cons_perc)
            mean_values.append(mean_value)
            ci_values.append(ci_value)
        
        plt.errorbar(
            UE_density_list,
            mean_values,
            yerr=ci_values,
            capsize=5,
            label=f"Var. {var_perc_value}%",
            color=color_dict_var[var_perc_value],
            linewidth=1,
            ls=ls_dict_var[var_perc_value]
        )

    plt.xticks(UE_density_list, UE_density_list)
    plt.title("Network Throughput Difference Percentage", fontsize=14)
    plt.xlabel("Number of UEs", fontsize=12)
    plt.ylabel("%", fontsize=12)
    plt.grid(True, which='both', linestyle='-.', linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.ylim(-0.1, 0.1)
    plt.savefig(f"./figures/{name_fig}.png", format='png')
    plt.show()


def plot_number_of_offloaded_users(name_fig, dict_opt_results, UE_density_list, var_param_perc_list, num_episods, enable_save=True):
    for var_perc_value in var_param_perc_list:
        mean_values = []
        ci_values = []
        
        for ue_density_value in UE_density_list:
            unique_id_len = []
            
            for episode in range(num_episods):
                data = dict_opt_results[f"ueDens_{ue_density_value}_varPerc_{var_perc_value}_ep_{episode}"]
                unique_id_len.append(len(np.unique(data["user_id"])))
            
            unique_id_len = np.array(unique_id_len)
            mean_value, ci_value = calculate_mean_and_ci(unique_id_len)
            
            mean_values.append(mean_value)
            ci_values.append(ci_value)
        
        plt.errorbar(
            UE_density_list,
            mean_values,
            yerr=ci_values,
            capsize=5,
            label=f"Var. {var_perc_value}%",
            color=color_dict_var[var_perc_value],
            linewidth=1,
            ls=ls_dict_var[var_perc_value]
        )

    plt.xticks(UE_density_list, UE_density_list)
    plt.title("Number of Offloaded Users", fontsize=14)
    plt.xlabel("Number of UEs", fontsize=12)
    plt.ylabel("Offloaded Users", fontsize=12)
    plt.grid(True, which='both', linestyle='-.', linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./figures/{name_fig}.png", format='png')
    plt.show()
               
    
def plot_layout_with_ue_locations_with_association_and_highlight_ues(name_fig, 
                                                                     scenario_model, 
                                                                     ue_distribution, 
                                                                     ue_density_value=8, 
                                                                     var_perc_value=10, 
                                                                     episode=1, 
                                                                     enable_save=True):
    # Construct the folder path where simulation results are stored
    folder_path = (f"../outputs/results/Opt_EE_UserAssociation_4G/",f"{scenario_model}_{ue_distribution}/",f"UEdensity_{ue_density_value}_varParam_{var_perc_value}/")
    
    # Load UE deployment data
    ue_deployment_path = folder_path + "to_plot_ue_deployment.npz"
    ue_deployment = np.load(ue_deployment_path)
    
    # Extract deployment data
    cell_site_positions_m = ue_deployment["cell_site_positions_m"]
    isd_m = ue_deployment["isd_m"]
    ue_position_m = ue_deployment["ue_position_m"]
    hot_spot_position_m = ue_deployment["hotspot_position_m"]
    
    # Retrieve IDs of highlighted UEs for the given episode
    highlighted_ue_ids = dict_opt_results[f"ueDens_{ue_density_value}_varPerc_{var_perc_value}_ep_{episode}"]["user_id"]
    
    # Load best-serving cell data (SSB-based)
    serving_cell_path = folder_path + "to_plot_best_serving_cell_heat_map_based_on_SSB.npz"
    best_serving_cell_ids = np.load(serving_cell_path)["data"]
    
    # Plot layout with highlighted UEs and serving cell associations
    _plot_layout_with_ue_locations_with_association_and_highlight_ues(name_fig, 
                                                                      cell_site_positions_m, 
                                                                      isd_m, 
                                                                      ue_position_m, 
                                                                      hot_spot_position_m, 
                                                                      highlighted_ue_ids, 
                                                                      best_serving_cell_ids, 
                                                                      enable_save=enable_save) 
    
    
def _plot_layout_with_ue_locations_with_association_and_highlight_ues(name_fig,
                                                                      site_positions_m, 
                                                                      isd_m, 
                                                                      ue_ids_positions_m, 
                                                                      hot_spot_position_m, 
                                                                      highlighted_ue_ids,
                                                                      best_serving_cell_ids, 
                                                                      enable_save=True):
    
    fig, ax = plt.subplots(figsize=(6.5,6.5))
    
    ax.set_aspect('equal')
    
    number_of_sites = np.size(site_positions_m,0)
    labels = [["site"]] * number_of_sites
    labels = [[''] for i in range(number_of_sites)]
    
    color_list = ['green', 'blue', 'red', 'orange', 'purple', 'yellow', 'pink', 'cyan', 'magenta', 'brown']
    # Assign colors to sites, cycling through the list every 19 sites
    color_vector = [color_list[i // 19 % len(color_list)] for i in range(number_of_sites)]
         
    # Add some coloured hexagons
    counter = 0
    for x, y, c, l in zip(site_positions_m[:,0], site_positions_m[:,1], color_vector[:number_of_sites], labels):

        if not np.isnan(isd_m) and counter < 19:
            
            hex = RegularPolygon((x, y), numVertices=6, radius=isd_m/np.sqrt(3),
                                 orientation=np.radians(30),
                                 facecolor="grey", alpha=0.2, edgecolor='k')
            
            ax.add_patch(hex)

        # Also add a text labels
        counter += 1
 
    # Also add scatter points in hexagon centres
    # Add scatter points for the base stations (BS) with different colors every 19 sites
    for i in range(number_of_sites):
        if i < 19:
            label = "BS Site" if i % 19 == 0 else None
        elif 19 <= i < 38:
            label = "5G BSs" if i % 19 == 0 else None
        elif 38 <= i < 57:
            label = "6G BSs" if i % 19 == 0 else None
        else:
            label = None  # No label for intermediate points
        ax.scatter(site_positions_m[i, 0], site_positions_m[i, 1], marker='^', c="black", alpha=1.0, label=label, s=120)
    # Also add UE positions
    ax.scatter(ue_ids_positions_m[:,0], ue_ids_positions_m[:,1], c="blue", marker='.', alpha=0.5, label="UE")
    # Highlight the UE that have moved
    ax.scatter(ue_ids_positions_m[highlighted_ue_ids,0], ue_ids_positions_m[highlighted_ue_ids,1], c="red", marker='o', alpha=0.8, label="Offloaded UE", s=100)
    
    # Also add hot spot positions
    if hot_spot_position_m is not None and hot_spot_position_m.size > 0:
        ax.scatter(hot_spot_position_m[:,0], hot_spot_position_m[:,1], c="red", marker='2', alpha=0.5, label="hotspot")       

    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.legend()
    ax.tick_params(axis='both', which='major')
    ax.grid()

    plt.tight_layout()    
    
    plt.savefig(f"./figures/{name_fig}.png", format='png')
    plt.show()  





if __name__ == "__main__":

    enable_save = True    
    

    """
    Set main parameters for uploading results and plots results
    """
    scenario_model = "ITU_R_M2135_UMa"
    ue_distribution = "uniform"
    num_episods = 8

    UE_density_list = list(range(1,11))
    var_param_perc_list = [10, 15, 20]

    """
    Loading the data results
    """
    # Load data
    dict_opt_results, dict_giuliaSls_results = load_results(UE_density_list, var_param_perc_list, num_episods)

    
    # Check for plot save folder
    os.makedirs("figures", exist_ok=True) 

    """
    2D network layout with UE offloaded by the proposed algorithm.
    """
    name_fig = "2D_net_Layout_highlightOffloadUE"
    plot_layout_with_ue_locations_with_association_and_highlight_ues(name_fig, 
                                                                     scenario_model, 
                                                                     ue_distribution, 
                                                                     ue_density_value=8, 
                                                                     var_perc_value=10, 
                                                                     episode=1, 
                                                                     enable_save=enable_save)
    del name_fig
    
    """
    Network power consumption (mean and 95% confidence interval) achieved 
    by the proposed re-association algorithm considering an average of five UE per cell
    """
    name_fig = "NetPower_consumption_overTime"
    plot_netPower_consumption_overTime(name_fig, dict_giuliaSls_results, ue_density_value=5,
                                       var_param_perc_list=var_param_perc_list, 
                                       enable_save=enable_save)
    del name_fig      
    
    """
    Network power saving percentage (mean and 95% confidence interval) achieved by 
    the proposed re-association algorithm for different average numbers of ue per cell.
    """
    name_fig = "NetPower_consumption_savingPercentage"
    plot_netPower_savingPercentage(name_fig, dict_giuliaSls_results, UE_density_list, 
                                   var_param_perc_list, enable_save=enable_save)
    del name_fig    
    
    """
    Network throughput percentage difference (mean and 95% confidence interval) achieved by 
    the proposed re-association algorithm for different average numbers of UE per cell.
    """
    name_fig = "NetPower_thp_diffPercentage"
    plot_netThp_diffPercentage(name_fig, dict_giuliaSls_results, UE_density_list, var_param_perc_list, 
                               enable_save=enable_save)
    
    del name_fig    
    
    """
    Number of offloaded UE (mean and 95% confidence interval) resulting from the proposed 
    re-association algorithm for different average numbers of UE per cell.
    """
    name_fig = "Number_offloaded_UE"
    plot_number_of_offloaded_users(name_fig, dict_opt_results, UE_density_list, var_param_perc_list, num_episods, 
                                   enable_save=enable_save)
    del name_fig    
    

    print("\nProgram Terminated")
    