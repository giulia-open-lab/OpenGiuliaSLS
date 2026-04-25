# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:48:13 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import time

import numpy as np

from giulia.event_driven.snapshot_control import Snapshot_control
from giulia.kpis import calculate_cell_beam_activity
from giulia.tools import tools


class Cell_Selection:
    
    def __init__(self,
                 compute_type, 
                 simulation_config_obj,
                 site_deployment_obj,
                 ue_playground_deployment_obj,
                 ue_deployment_obj,
                 beam_conf_obj,
                 cell_re_selection_conf_obj,
                 distance_angles_ue_to_cell_obj,
                 channel_gain_ue_to_cell_obj,
                 RSRP_ue_to_cell_obj,
                 dl_noise_ue_to_cell_obj):
        
        ##### Input storage 
        ######################## 
        
        self.compute_type = compute_type
        self.simulation_config_obj = simulation_config_obj
        self.site_deployment_obj = site_deployment_obj
        self.ue_playground_deployment_obj = ue_playground_deployment_obj
        self.ue_deployment_obj = ue_deployment_obj
        self.beam_conf_obj = beam_conf_obj
        self.cell_re_selection_conf_obj = cell_re_selection_conf_obj
        self.distance_angles_ue_to_cell_obj = distance_angles_ue_to_cell_obj
        self.channel_gain_ue_to_cell_obj = channel_gain_ue_to_cell_obj
        self.RSRP_ue_to_cell_obj = RSRP_ue_to_cell_obj
        self.dl_noise_ue_to_cell_obj = dl_noise_ue_to_cell_obj
  
                
        ##### Outputs 
        ########################   
        
        # Placeholder to store antenna pattern gain results
        
        self.best_serving_beam_ID_per_ue = []
        self.best_serving_cell_ID_per_ue = []
        
        self.best_server_distance_3d_per_ue_m = [] 
        self.best_server_coupling_gain_per_ue_dB = []
        self.best_server_rsrp_per_ue_dBm = []
        self.best_server_re_noise_per_ue_dBm = []
        self.beam_activity_per_ue = []
        self.ues_per_cell = []
        self.ues_per_beam = []        
        
        
    def process(self, rescheduling_us=-1):
        
        ##### Start timer
        ########################
        t_start = time.perf_counter() 
        
        ##### Switch
        ########################          

        # Calculate best server   
        if self.compute_type == 'strongest_rsrp':
            
            self.best_serving_beam_ID_per_ue, self.best_serving_cell_ID_per_ue = \
                self.strongest_rsrp(self.beam_conf_obj, self.RSRP_ue_to_cell_obj.RSRP_ue_to_cell_dBm)                 
                     
        ##### End 
        ########################
        print("Cell selection calculations, elapsed time", time.perf_counter() - t_start)     
        
        return rescheduling_us 
    
    
    def strongest_rsrp(
            self,
            beam_conf_obj, 
            rsrp_ue_to_cell_dBm):
        
        # Get beam to cell mapping
        if beam_conf_obj is None:  # One beam per cell, as in LTE
            beam_to_node_mapping = np.arange(0, np.size(rsrp_ue_to_cell_dBm, 1)).astype(int)
            
        else:  # Multiple beams per cell, as in NR
            beam_to_node_mapping = beam_conf_obj.beam_to_node_mapping    
                
        
        # Find for every UE the cell/beam that provides the strongest RSRP
        best_serving_beam_ID_per_ue = np.argmax(rsrp_ue_to_cell_dBm,axis=-1)
        best_serving_cell_ID_per_ue = beam_to_node_mapping[best_serving_beam_ID_per_ue]           
        
        return best_serving_beam_ID_per_ue, best_serving_cell_ID_per_ue
    
    
    def calculate_server_stats(self, rescheduling_us=-1):
        
        # Channel characterisitics  
        distance_3d_ue_to_cell_m = self.distance_angles_ue_to_cell_obj.distance_b_to_a_3d_wraparound_m       
        
        # Number of cells
        number_of_cells = np.size(distance_3d_ue_to_cell_m,1)
         
        # Get beam to cell mapping
        if self.beam_conf_obj == None : #In this case, there is one beam per cell, as in LTE
            beam_type = "antenna element"
            number_of_beams_per_cell = np.ones(number_of_cells, dtype=int)
            beam_to_node_mapping = np.arange(0,len(self.ue_deployment_obj.df_ep)).astype(int)   
        else: #In this case, there is one or more beams per cell, as in NR  
            beam_type = self.beam_conf_obj.beam_type
            number_of_beams_per_cell = self.beam_conf_obj.number_of_beams_per_node      
            beam_to_node_mapping = self.beam_conf_obj.beam_to_node_mapping        
    
        # Channel
        if self.beam_conf_obj is None:  # one beam per cell
            coupling_gain_ue_to_cell_dB: np.ndarray = \
                self.channel_gain_ue_to_cell_obj.slow_channel_results_per_frequency_layer["all_freq"]["slow_channel_gain_b_to_a_dB"]
            
        else:  # multiple beams per cell
            coupling_gain_ue_to_cell_dB: np.ndarray = \
                self.channel_gain_ue_to_cell_obj.precoded_channel_gain_results_per_frequency_layer["all_freq"]["precoded_channel_gain_b_to_a_dB"]
            
        # RSRP 
        rsrp_ue_to_cell_dBm = self.RSRP_ue_to_cell_obj.RSRP_ue_to_cell_dBm 
        
        # Noise
        dl_full_bandwidth_noise_dBm = self.dl_noise_ue_to_cell_obj.dl_noise_per_resource_element_ue_to_cell_dBm    
            
        # Set UE IDs
        ue_IDs = np.arange(0,np.size(distance_3d_ue_to_cell_m,0))
        
        # Find for every UE the distance of the cell/beam that provides the strongest RSRP
        self.best_server_distance_3d_per_ue_m = distance_3d_ue_to_cell_m[ue_IDs, self.best_serving_cell_ID_per_ue]  
        
        # Find for every UE the coupling gain of the cell/beam that provides the strongest RSRP
        if coupling_gain_ue_to_cell_dB.ndim == 3:
            self.best_server_coupling_gain_per_ue_dB = \
                tools.mW_to_dBm(np.mean(tools.dBm_to_mW(coupling_gain_ue_to_cell_dB[:, ue_IDs, self.best_serving_beam_ID_per_ue]), axis=0))
        else:
            self.best_server_coupling_gain_per_ue_dB = coupling_gain_ue_to_cell_dB[ue_IDs, self.best_serving_beam_ID_per_ue]        
        
        # Find for every UE the RSRP of the cell/beam that provides the strongest RSRP
        self.best_server_rsrp_per_ue_dBm = rsrp_ue_to_cell_dBm[ue_IDs, self.best_serving_beam_ID_per_ue]    
        
        # Find for every UE the noise power of the cell/beam that provides the strongest RSRP 
        self.best_server_re_noise_per_ue_dBm = dl_full_bandwidth_noise_dBm[ue_IDs, self.best_serving_cell_ID_per_ue]   


        # Calculate cell/beam activity
        if self.beam_conf_obj == None:
            self.beam_activity_per_ue, self.ues_per_beam =\
                calculate_cell_beam_activity.cell_activity(np.size(distance_3d_ue_to_cell_m,1), self.best_serving_cell_ID_per_ue) 

            self.ues_per_cell = self.ues_per_beam
                
        elif self.beam_conf_obj.beam_type == "SSB":
            self.beam_activity_per_ue, self.ues_per_beam =\
                calculate_cell_beam_activity.SSB_beam_activity(self.beam_conf_obj,self.best_serving_beam_ID_per_ue)  
                
            self.ues_per_cell = np.bincount(beam_to_node_mapping, weights=self.ues_per_beam)          
                
        # Save to plot  
        snapshot_control = Snapshot_control.get_instance()
        if self.simulation_config_obj.plot_heatmaps == 1 and snapshot_control.num_snapshots == 0: 
            
            # Get variables for heatmap plots        
            x_size = int(np.ceil(self.ue_playground_deployment_obj.scenario_x_side_length_m/self.ue_playground_deployment_obj.grid_resol_m))
            y_size = int(np.ceil(self.ue_playground_deployment_obj.scenario_y_side_length_m/self.ue_playground_deployment_obj.grid_resol_m))
            ue_grid_position = self.ue_deployment_obj.ue_grid_position        
            
            folder_name = "results_" + self.simulation_config_obj.project_name + "/"
            
            # Plot best serving cell ID heat map
            np.savez(folder_name + "to_plot_best_serving_cell_heat_map_based_on_" + beam_type,
                    x_size = x_size,
                    y_size = y_size,
                    ue_grid_position = ue_grid_position,
                    best_serving_cell_ID = self.best_serving_cell_ID_per_ue,
                    grid_resol_m = self.ue_playground_deployment_obj.grid_resol_m,
                    number_of_cells = number_of_cells)   
            
            # Plot best serving beam ID heatmap
            np.savez(folder_name + "to_plot_best_serving_beam_heat_map_based_on_" + beam_type,
                    x_size = x_size,
                    y_size = y_size,
                    ue_grid_position = ue_grid_position,
                    best_serving_beam_ID = self.best_serving_beam_ID_per_ue,
                    grid_resol_m = self.ue_playground_deployment_obj.grid_resol_m,
                    number_of_beams = np.sum(number_of_beams_per_cell))          
            
            # Plot best server RSRP heatmap
            np.savez(folder_name + "to_plot_best_serving_cell_rsrp_map_based_on_" + beam_type,
                    x_size = x_size,
                    y_size = y_size,
                    ue_grid_position = ue_grid_position,
                    best_serving_cell_rsrp_dBm = self.best_server_rsrp_per_ue_dBm,
                    grid_resol_m = self.ue_playground_deployment_obj.grid_resol_m,
                    number_of_cells = number_of_cells)          
        
        return rescheduling_us
    

    
class Cell_ReSelection:
    
    def __init__(self,
                 compute_type, 
                 ue_deployment_obj,
                 beam_conf_obj,
                 cell_re_selection_conf_obj,
                 best_serving_cell_ID_per_ue_obj,
                 RSRP_ue_to_cell_obj):
        
        ##### Input storage 
        ######################## 
        
        self.compute_type = compute_type
        self.ue_deployment_obj = ue_deployment_obj
        self.beam_conf_obj = beam_conf_obj
        self.cell_re_selection_conf_obj = cell_re_selection_conf_obj
        self.best_serving_cell_ID_per_ue_obj = best_serving_cell_ID_per_ue_obj
        self.RSRP_ue_to_cell_obj = RSRP_ue_to_cell_obj
        
                    
        ##### Outputs 
        ########################   
        
        # Placeholder to store antenna pattern gain results
        
        self.best_serving_beam_ID_per_ue = []
        self.best_serving_cell_ID_per_ue = []     
        
    
    def process(self, rescheduling_us=-1):
                
        ##### Start timer
        ########################
        t_start = time.perf_counter() 
        
        ##### Switch
        ########################          

        # Calculate cell reselection: the UE reselects to the cell with highest priority and RSRP   
        if self.compute_type == 'priority_plus_strongest_rsrp':

            self.best_serving_beam_ID_per_ue, self.best_serving_cell_ID_per_ue =\
                self.priority_plus_strongest_rsrp(
                    self.beam_conf_obj, 
                    self.cell_re_selection_conf_obj,
                    self.best_serving_cell_ID_per_ue_obj.best_serving_cell_ID_per_ue,
                    self.best_serving_cell_ID_per_ue_obj.best_serving_beam_ID_per_ue,
                    self.RSRP_ue_to_cell_obj.RSRP_ue_to_cell_dBm)
                
            self.best_serving_cell_ID_per_ue_obj.best_serving_beam_ID_per_ue =  self.best_serving_beam_ID_per_ue
            self.best_serving_cell_ID_per_ue_obj.best_serving_cell_ID_per_ue =  self.best_serving_cell_ID_per_ue
                                
        ##### End 
        ########################
        print("Cell selection calculations, elapsed time", time.perf_counter() - t_start)     
        
        return rescheduling_us
    
    
    def priority_plus_strongest_rsrp(
            self,
            beam_conf_obj, 
            cell_re_selection_conf_obj,
            best_serving_cell_ID_per_ue,
            best_serving_beam_ID_per_ue,
            rsrp_ue_to_cell_dBm):
        
        # Get beam to cell mapping
        if beam_conf_obj is None:  # One beam per cell, as in LTE
            beam_to_node_mapping = np.arange(0, np.size(rsrp_ue_to_cell_dBm, 1)).astype(int)
            number_of_beams = np.ones(np.size(rsrp_ue_to_cell_dBm, 1), dtype=int)
            
        else:  # Multiple beams per cell, as in NR
            beam_to_node_mapping = beam_conf_obj.beam_to_node_mapping
            number_of_beams = beam_conf_obj.number_of_beams_per_node
    
        # Get cell reselection priorities [cell x cell]
        cell_reselection_priority = cell_re_selection_conf_obj.cell_reselection_priority   
        q_Rx_Lev_Min_dBm = cell_re_selection_conf_obj.q_Rx_Lev_Min_dBm 
        s_search_P = cell_re_selection_conf_obj.s_search_P
    
        # Create a new 2D array to store user-cell priorities [ue x cell]
        num_ue = len(best_serving_cell_ID_per_ue)
        cell_reselection_priority_ue_to_cell = np.zeros((num_ue, cell_reselection_priority.shape[1])) 
        q_Rx_Lev_Min_ue_to_cell_dBm = np.zeros((num_ue, q_Rx_Lev_Min_dBm.shape[1]), dtype=np.single) 
        s_search_P_ue_to_cell = np.zeros((num_ue, s_search_P.shape[1]), dtype=np.single) 
    
        # Find the priorities per ue based on its serving cell broadcast information [ue x cell]
        for i, server_index in enumerate(best_serving_cell_ID_per_ue):
            cell_reselection_priority_ue_to_cell[i, :] = cell_reselection_priority[server_index, :]
            q_Rx_Lev_Min_ue_to_cell_dBm[i, :] = q_Rx_Lev_Min_dBm[server_index, :]   
            s_search_P_ue_to_cell[i, :] = s_search_P[server_index, :] 
    
        # Resize for the beam case
        cell_reselection_priority_ue_to_beam = np.repeat(cell_reselection_priority_ue_to_cell, number_of_beams, axis=1)   
        q_Rx_Lev_Min_ue_to_beam_dBm = np.repeat(q_Rx_Lev_Min_ue_to_cell_dBm, number_of_beams, axis=1) 
        
        # Initialize arrays to store the new best serving beam/cell IDs
        new_best_serving_beam_ID_per_ue = np.full(num_ue, -1, dtype=int)
        new_best_serving_cell_ID_per_ue = np.full(num_ue, -1, dtype=int)
    
        # Iterate over each UE and determine the best serving cell/beam based on priority and RSRP
        for user_ID in range(num_ue):
            
            found_serving_beam = False  # Flag to track if a valid serving beam is found
            
            # Iterate through the unique priorities in descending order
            for priority in sorted(np.unique(cell_reselection_priority_ue_to_beam[user_ID, :]), reverse=True):
                
                # If the UE is already associated to a cell with this priority we do not need to process more data
                if cell_reselection_priority[best_serving_cell_ID_per_ue[user_ID],best_serving_cell_ID_per_ue[user_ID]] == priority:
                    break
                
                # Find all beams/cells with the same max priority
                cells_with_same_max_priority = np.where(cell_reselection_priority_ue_to_beam[user_ID, :] == priority)[0]
                
                # Get the corresponding SrxLev values for these cells
                SrxLev = rsrp_ue_to_cell_dBm[user_ID, cells_with_same_max_priority]
                
                if len(cells_with_same_max_priority) > 1:
                    # Multiple cells with the same priority, choose the one with the strongest RSRP
                    strongest_SrxLev_index = np.argmax(SrxLev)
                    strongest_beam_ID = cells_with_same_max_priority[strongest_SrxLev_index]
                else:
                    # Only one cell/beam with this priority
                    strongest_beam_ID = cells_with_same_max_priority[0]
                
                # Check if the strongest RSRP meets the minimum threshold
                if rsrp_ue_to_cell_dBm[user_ID, strongest_beam_ID] > q_Rx_Lev_Min_ue_to_beam_dBm[user_ID, strongest_beam_ID]:
                    new_best_serving_beam_ID_per_ue[user_ID] = strongest_beam_ID
                    found_serving_beam = True
                    break  # Exit the loop once a valid beam is found
    
            if not found_serving_beam:
                # If no beam was found with valid RSRP, stick with the current serving cell
                new_best_serving_beam_ID_per_ue[user_ID] = best_serving_beam_ID_per_ue[user_ID]
                new_best_serving_cell_ID_per_ue[user_ID] = best_serving_cell_ID_per_ue[user_ID]
                
            else:
                # Map the selected beam back to the corresponding cell
                new_best_serving_cell_ID_per_ue[user_ID] = beam_to_node_mapping[new_best_serving_beam_ID_per_ue[user_ID]]
                
        return new_best_serving_beam_ID_per_ue, new_best_serving_cell_ID_per_ue