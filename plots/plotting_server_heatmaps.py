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

from giulia.plots import plotting

project_name = "ITU_R_M2135_UMa" + "_uniform"
folder_name = "results_" + project_name


#################################################
# Plot best serving cell ID HEAT MAP
#################################################

beam_type = "antenna element"
a_load = np.load(os.path.join(os.getcwd(), folder_name, "to_plot_best_serving_cell_heat_map_based_on_" + beam_type + ".npz"))

plotting.plot_sparse_heat_map(project_name,
                              a_load["x_size"], a_load["y_size"],
                              a_load["ue_grid_position"][0], a_load["ue_grid_position"][1],
                              a_load["best_serving_cell_ID"], 
                              a_load["grid_resol_m"],
                              a_load["number_of_cells"], 
                              'Best server heatmap based on ' + beam_type,
                              'Cell ID')

beam_type = "SSB"
a_load = np.load(os.path.join(os.getcwd(), folder_name, "to_plot_best_serving_cell_heat_map_based_on_" + beam_type + ".npz"))

plotting.plot_sparse_heat_map(project_name,
                              a_load["x_size"], a_load["y_size"],
                              a_load["ue_grid_position"][0], a_load["ue_grid_position"][1],
                              a_load["best_serving_cell_ID"],
                              a_load["grid_resol_m"],
                              a_load["number_of_cells"], 
                              'Best server heatmap based on ' + beam_type,
                              'Cell ID')


#################################################
# Plot best serving beam ID HEAT MAP
#################################################

beam_type = "SSB"
a_load = np.load(os.path.join(os.getcwd(), folder_name, "to_plot_best_serving_beam_heat_map_based_on_" + beam_type + ".npz"))

plotting.plot_sparse_heat_map(project_name,
                              a_load["x_size"], a_load["y_size"],
                              a_load["ue_grid_position"][0], a_load["ue_grid_position"][1],
                              a_load["best_serving_beam_ID"],
                              a_load["grid_resol_m"],
                              a_load["number_of_beams"], 
                              'Best server heatmap based on ' + beam_type,
                              'Beam ID')

beam_type = "CSI_RS"
a_load = np.load(os.path.join(os.getcwd(), folder_name, "to_plot_best_serving_beam_heat_map_based_on_" + beam_type + ".npz"))

plotting.plot_sparse_heat_map(project_name,
                              a_load["x_size"], a_load["y_size"],
                              a_load["ue_grid_position"][0], a_load["ue_grid_position"][1],
                              a_load["best_serving_beam_ID"],
                              a_load["grid_resol_m"],
                              a_load["number_of_beams"], 
                              'Best server heatmap based on ' + beam_type,
                              'Beam ID')


#################################################
# Plot best serving cell RSRP HEAT MAP
#################################################

beam_type = "SSB"
a_load = np.load(os.path.join(os.getcwd(), folder_name, "to_plot_best_serving_cell_rsrp_map_based_on_" + beam_type + ".npz"))

plotting.plot_sparse_heat_map(project_name,
                              a_load["x_size"], a_load["y_size"],
                              a_load["ue_grid_position"][0], a_load["ue_grid_position"][1],
                              a_load["best_serving_cell_rsrp_dBm"],
                              a_load["grid_resol_m"],
                              500, 
                              'RSRP heatmap based on ' + beam_type,
                              'RSRP [dBm]')

beam_type = "CSI_RS"
a_load = np.load(os.path.join(os.getcwd(), folder_name, "to_plot_best_serving_cell_rsrp_map_based_on_" + beam_type + ".npz"))

plotting.plot_sparse_heat_map(project_name, 
                              a_load["x_size"], a_load["y_size"],
                              a_load["ue_grid_position"][0], a_load["ue_grid_position"][1],
                              a_load["best_serving_cell_rsrp_dBm"],
                              a_load["grid_resol_m"],
                              a_load["number_of_cells"], 
                              'RSRP heatmap based on ' + beam_type,
                              'RSRP [dBm]') 


#################################################
# Plot best serving geometry SINR HEAT MAP
#################################################

beam_type = "antenna element"
a_load = np.load(os.path.join(os.getcwd(), folder_name, "to_plot_geometry_sinr_heat_map_based_on_" + beam_type + ".npz"))
plotting.plot_sparse_heat_map(project_name,
                              a_load["x_size"], a_load["y_size"],
                              a_load["ue_grid_position"][0], a_load["ue_grid_position"][1],
                              a_load["rsrp_based_sinr_dB"],
                              a_load["grid_resol_m"],
                              500, 
                              'Geoemtry SINR heatmap based on ' + beam_type,
                              'SINR [dB]')

beam_type = "SSB"
a_load = np.load(os.path.join(os.getcwd(), folder_name, "to_plot_geometry_sinr_heat_map_based_on_" + beam_type + ".npz"))
plotting.plot_sparse_heat_map(project_name,
                              a_load["x_size"], a_load["y_size"],
                              a_load["ue_grid_position"][0], a_load["ue_grid_position"][1],
                              a_load["rsrp_based_sinr_dB"],
                              a_load["grid_resol_m"],
                              500, 
                              'Geoemtry SINR heatmap based on ' + beam_type,
                              'SINR [dB]')

a_load = np.load(os.path.join(os.getcwd(), folder_name, "to_plot_geometry_interference_heat_map_based_on_" + beam_type + ".npz"))
plotting.plot_sparse_heat_map(project_name,
                              a_load["x_size"], a_load["y_size"],
                              a_load["ue_grid_position"][0], a_load["ue_grid_position"][1],
                              a_load["interference_dB"],
                              a_load["grid_resol_m"],
                              500, 
                              'Interference heatmap based on ' + beam_type,
                              'Interference [dBm]')

a_load = np.load(os.path.join(os.getcwd(), folder_name, "to_plot_geometry_interference_heat_map_based_on_" + beam_type + ".npz"))
plotting.plot_sparse_heat_map(project_name,
                              a_load["x_size"], a_load["y_size"],
                              a_load["ue_grid_position"][0], a_load["ue_grid_position"][1],
                              a_load["interference_dB"],
                              a_load["grid_resol_m"],
                              500, 
                              'Interference heatmap based on ' + beam_type,
                              'Interference [dBm]')


#################################################
# Plot best serving geometry SINR HEAT MAP
#################################################

beam_type = "CSI_RS"
a_load = np.load(os.path.join(os.getcwd(), folder_name, "to_plot_per_prb_sinr_heat_map_based_on_" + beam_type + ".npz"))
plotting.plot_sparse_heat_map(project_name,
                              a_load["x_size"], a_load["y_size"],
                              a_load["ue_grid_position"][0], a_load["ue_grid_position"][1],
                              a_load["effiective_sinr_mean"],
                              a_load["grid_resol_m"],
                              a_load["number_of_cells"], 
                              'PRB SINR heatmap based on ' + beam_type,
                              'PRB SINR [dB]')
