# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 15:39:58 2024

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import time
from typing import List

import numpy as np
import pandas as pd

from giulia.tools.tools import log_calculations_time, log_elapsed_time
from giulia.outputs.saveable import Saveable

class Cell_Re_Selection_Conf(Saveable):
    
    def __init__(self, simulation_config_obj, network_deployment_obj): 
       
       super().__init__()
        
       ##### Plots 
       ########################
       self.plot = 0 # Switch to control plots if any

       ##### Input storage 
       ######################## 
       self.simulation_config_obj = simulation_config_obj
       self.network_deployment_obj = network_deployment_obj

       ##### Output  
       ########################   
       self.df_q_Rx_Lev_Min_dBm = []
       self.df_s_search_P = []
       self.df_cell_reselection_priority = []
       
       #Srxlev = Qrxlevmeas - qRxLevMin where Qrxlevmeas= RSRP
       
       #s-IntraSearchP
       #s-NonIntraSearchP
       
       #SrxLev > s-IntraSearchP
       #SrxLev > s-NonIntraSearchP
       
    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["plot"]

       
    def process(self, rescheduling_us=-1):  
        
        # Process inputs
        self.dl_carrier_frequencies_GHz = self.network_deployment_obj.df_ep["dl_carrier_frequency_GHz"].to_numpy(dtype=np.single)  
        
        # Init outputs
        cell_reselection_priority = np.full((len(self.network_deployment_obj.df_ep), len(self.network_deployment_obj.df_ep)), -1, dtype=np.single) 
        q_Rx_Lev_Min_dBm = np.full((len(self.network_deployment_obj.df_ep), len(self.network_deployment_obj.df_ep)), -1, dtype=np.single) 
        s_search_P = np.full((len(self.network_deployment_obj.df_ep), len(self.network_deployment_obj.df_ep)), -1, dtype=np.single) 
        
        # Start timer       
        t_start = time.perf_counter() 
        
        # Calculate q_Rx_Lev_Min_dBm and s_search_P
            # For simplicity we assume that all cells adopt the same configuration
            # Any other configuration is possible and this code should be updated accordingly
            
            # the (i,j) element of q_Rx_Lev_Min_dBm matrix indicates the q_Rx_Lev_Min_dBm value transmitted at the MIB/SIB of the i-th cell for the j-th cell 
        q_Rx_Lev_Min_dBm = np.ones((len(self.network_deployment_obj.df_ep), len(self.network_deployment_obj.df_ep)), dtype=np.single) * -200
        np.fill_diagonal(q_Rx_Lev_Min_dBm, None) 
            
        # the (i,j) element of s_search_P matrix indicates the s_search_P value transmitted at the MIB/SIB of the i-th cell for the j-th cell         
        s_search_P = np.zeros((len(self.network_deployment_obj.df_ep), len(self.network_deployment_obj.df_ep)), dtype=np.single)
        np.fill_diagonal(s_search_P, None) 
        
        # Find the set of frequencies to process them independently
        dl_carrier_frequency_GHz_set = sorted(set(self.dl_carrier_frequencies_GHz))
        
        # Process each frequency independently
        for priority, dl_carrier_frequency_GHz in enumerate(dl_carrier_frequency_GHz_set): 
            
            # Identify cells with the selected frequency
            mask = dl_carrier_frequency_GHz ==  self.dl_carrier_frequencies_GHz
            
            # Provide parameters
            cell_reselection_priority[:,mask] = priority 
            
            if priority == 1 :
                q_Rx_Lev_Min_dBm[:,mask] = -108
            elif priority == 2 :
                q_Rx_Lev_Min_dBm[:,mask] = -104
            
        # Store in data frames the results as it may be useful to post process
        self.df_q_Rx_Lev_Min_dBm = pd.DataFrame(q_Rx_Lev_Min_dBm, 
                                                columns=self.network_deployment_obj.df_ep["name"], index=self.network_deployment_obj.df_ep["name"])  
        self.df_s_search_P = pd.DataFrame(s_search_P, 
                                          columns=self.network_deployment_obj.df_ep["name"], index=self.network_deployment_obj.df_ep["name"])    
        self.df_cell_reselection_priority = pd.DataFrame(cell_reselection_priority, 
                                                         columns=self.network_deployment_obj.df_ep["name"], index=self.network_deployment_obj.df_ep["name"])  

          
        ##### End
        log_elapsed_time('Cell (re)selection parameters', t_start)
        
        return rescheduling_us


class Cell_Re_Selection(Saveable):
    
    def __init__(self,
                 compute_type, 
                 simulation_config_obj,
                 ue_deployment_obj,
                 beam_conf_obj,
                 cell_re_selection_conf_obj,
                 best_serving_cell_ID_per_ue_obj,
                 RSRP_ue_to_cell_obj):
        
        super().__init__()

        ##### Input storage 
        ######################## 
        
        self.simulation_config_obj = simulation_config_obj
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


    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return [
            "best_serving_beam_ID_per_ue",
            "best_serving_cell_ID_per_ue"
        ]
    
        
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
                    self.RSRP_ue_to_cell_obj.RSRP_results_per_frequency_layer["all_freq"]["RSRP_ue_to_cell_dBm"])
                
            self.best_serving_cell_ID_per_ue_obj.best_serving_beam_ID_per_ue =  self.best_serving_beam_ID_per_ue
            self.best_serving_cell_ID_per_ue_obj.best_serving_cell_ID_per_ue =  self.best_serving_cell_ID_per_ue
                                
        ##### End
        log_calculations_time('Cell reselection', t_start)
        
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
            # It is important to use the pandas dataframes and not the numpy arrays as these values may be modified over the top by an optimizer
                    
        q_Rx_Lev_Min_dBm = cell_re_selection_conf_obj.df_q_Rx_Lev_Min_dBm.to_numpy(dtype=np.single) 
        s_search_P = cell_re_selection_conf_obj.df_s_search_P.to_numpy(dtype=np.single)  
        cell_reselection_priority = cell_re_selection_conf_obj.df_cell_reselection_priority.to_numpy(dtype=np.single)
    
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
                SrxLev = rsrp_ue_to_cell_dBm[user_ID, cells_with_same_max_priority] #- q_Rx_Lev_Min_ue_to_beam_dBm[user_ID, cells_with_same_max_priority]
                
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
    

class Cell_Re_Selection_CRS(Cell_Re_Selection):
    pass

class Cell_Re_Selection_SSB(Cell_Re_Selection):
    pass
