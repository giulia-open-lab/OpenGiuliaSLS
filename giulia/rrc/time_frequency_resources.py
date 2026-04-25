# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:37:25 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import copy
import time
from typing import Any, Dict, List

import numpy as np
import sionna

from giulia.channel import carriers
from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable

class Time_Frequency_Resources(Saveable):
        
    def __init__(self, simulation_config_obj, network_deployment_obj, cell_antenna_array_structure_obj, cell_SSB_conf_obj, cell_CSI_RS_conf_obj):

       super().__init__()

       ##### Plots 
       ########################
       self.plot = 0 # Switch to control plots if any
       
       ##### Input storage 
       ########################   
       
       # Models 
       self.simulation_config_obj = simulation_config_obj 
       self.network_deployment_obj = network_deployment_obj 
       self.cell_antenna_array_structure_obj = cell_antenna_array_structure_obj
       self.cell_SSB_conf_obj = cell_SSB_conf_obj
       self.cell_CSI_RS_conf_obj = cell_CSI_RS_conf_obj
              
       ##### Outputs 
       ########################         
       
       # DL carriers list
       self.dl_carriers = []
       # Dictionary to store results for each frequency layer
       self.dl_frequency_layer_info: Dict[float, Dict[str, Any]] = {}
       
       # UL carriers list
       self.ul_carriers = []   
       
       # Indeces indicating the DL and UL carrier of each cell
       self.dl_carrier_index = np.full(len(self.network_deployment_obj.df_ep), -1, dtype=int)  
       self.ul_carrier_index = np.full(len(self.network_deployment_obj.df_ep), -1, dtype=int)   
       
       self.cell_by_cell_dl_carrier_mask = np.zeros((len(self.network_deployment_obj.df_ep),len(self.network_deployment_obj.df_ep)), dtype=bool)
       self.cell_by_cell_ul_carrier_mask = np.zeros((len(self.network_deployment_obj.df_ep),len(self.network_deployment_obj.df_ep)), dtype=bool)
       
       self.SSB_beam_by_SSB_beam_dl_carrier_mask = np.zeros((len(self.network_deployment_obj.df_ep),len(self.network_deployment_obj.df_ep)), dtype=bool)
       
       self.CSI_RS_beam_by_CSI_RS_beam_dl_carrier_mask = np.zeros((len(self.network_deployment_obj.df_ep),len(self.network_deployment_obj.df_ep)), dtype=bool)
       
       # Sionna related  
       self.dl_resource_grid = [] # DL resource grid list 
       self.ul_resource_grid = [] # UL resource grid list      
       
       self.dl_resource_grid_index = np.full(len(self.network_deployment_obj.df_ep), -1, dtype=int)   # Indeces indicating the DL resource grid of each cell
       self.ul_resource_grid_index = np.full(len(self.network_deployment_obj.df_ep), -1, dtype=int)   # Indeces indicating the UL resource grid of each cell  


    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["presetdl_carriers"]

       
    def process(self, rescheduling_us=-1): 
       
       # Start timer       
       t_start = time.perf_counter()    
       
       # Store key information relating to all cells, independently of the frequency layer
       num_cells = len(self.network_deployment_obj.df_ep)
       self.dl_frequency_layer_info["all_freq"] = {
           "available_PRBs": self.network_deployment_obj.df_ep['dl_PRBs_available'].iloc[0],
           "cells_in_frequency_mask":  np.ones((num_cells,), dtype=bool),
           "cell_IDs": np.arange(0, num_cells),  # List of cell IDs
           "num_cells": num_cells,
           "cell_ants_in_frequency_mask": np.ones((len(self.cell_antenna_array_structure_obj.antenna_to_node_mapping),), dtype=bool),
           "num_cell_ants": len(self.cell_antenna_array_structure_obj.antenna_to_node_mapping),
           "ssb_beams_in_frequency_mask":  np.ones((len(self.cell_SSB_conf_obj.beam_to_node_mapping),), dtype=bool),
           "global_ssb_beams_IDs": np.arange(0, len(self.cell_SSB_conf_obj.beam_to_node_mapping)),
           "csi_rs_beams_in_frequency_mask":  np.ones((len(self.cell_CSI_RS_conf_obj.beam_to_node_mapping),), dtype=bool),
           "global_csi_rs_beams_IDs": np.arange(0, len(self.cell_CSI_RS_conf_obj.beam_to_node_mapping))
       } 
       
       # Store key information per frequency layer 
       
       # Creating DL carrier
              
       # Group by carrier characterstics to find the different DL carriers
       grouped_dl = self.network_deployment_obj.df_ep.groupby(['fdd_tdd_ind', #0
                                   'subframe_assignment', #1
                                   'FR', #2
                                   'subcarriers_per_PRB', #3
                                   'ofdm_symbols_in_slot', #4
                                   'dl_bandwidth_MHz', #5
                                   'dl_PRBs_available', #6
                                   'dl_subcarrier_spacing_kHz', #7
                                   'dl_ofdm_symbol_duration_us', #8
                                   'dl_control_channel_overhead', #9
                                   'dl_carrier_frequency_GHz']) #10                                   
       
       # Iterate over the DL carriers
       index = 0
       for model_params, group in grouped_dl: 
           self.dl_carrier_index[group['ID']] = index
           obj = carriers.Carrier(True,
                                  group['ID'],
                                  group['antenna_config_number_of_elements'],
                                  model_params[0], model_params[1], model_params[2], model_params[3], model_params[4], 
                                  model_params[5], model_params[6], model_params[7], model_params[8], model_params[9])
           
           self.dl_carriers.append(obj)
           
           if self.simulation_config_obj.sn_indicator == 1 :
               self.dl_resource_grid_index[group['ID']] = index
               dl_rg = sionna.phy.ofdm.ResourceGrid(num_ofdm_symbols = 1,#model_params[4]
                                                fft_size = model_params[6],#model_params[6]*model_params[3]
                                                subcarrier_spacing = model_params[7] * 1e3 * model_params[3],#model_params[7] 
                                                num_tx = 1,
                                                num_streams_per_tx = 1,
                                                num_guard_carriers = [0, 0],#[5,6]
                                                dc_null = True,
                                                pilot_pattern = "kronecker",
                                                pilot_ofdm_symbol_indices = [0])#[2, 11]
           
               self.dl_resource_grid.append(dl_rg)
               
               
           # Identify cells and antennas operating in the current carrier
           dl_carrier_frequency_GHz = group['dl_carrier_frequency_GHz'].iloc[0]
           available_PRBs = group['dl_PRBs_available'].iloc[0]
           antenna_to_node_a_mapping: np.ndarray = self.cell_antenna_array_structure_obj.antenna_to_node_mapping
           cells_in_frequency_mask: np.ndarray = self.network_deployment_obj.df_ep['dl_carrier_frequency_GHz'].to_numpy() ==  dl_carrier_frequency_GHz 
           cell_IDs_in_frequency: np.ndarray = np.flatnonzero(cells_in_frequency_mask)  # Indices of cells in frequency
           num_cells_in_frequency: int = cell_IDs_in_frequency.size
           cell_ants_in_frequency_mask: np.ndarray = np.isin(antenna_to_node_a_mapping, cell_IDs_in_frequency)
           num_cell_ants_in_frequency: int = cell_ants_in_frequency_mask.sum()    
           ssb_beams_in_frequency_mask: np.ndarray = np.isin(self.cell_SSB_conf_obj.beam_to_node_mapping, cell_IDs_in_frequency)
           global_ssb_beams_IDs: np.ndarray = np.where(ssb_beams_in_frequency_mask)[0]  # Global beam indices
           csi_rs_beams_in_frequency_mask: np.ndarray = np.isin(self.cell_CSI_RS_conf_obj.beam_to_node_mapping, cell_IDs_in_frequency)     
           global_csi_rs_beams_IDs: np.ndarray = np.where(csi_rs_beams_in_frequency_mask)[0]  # Global beam indices

           # Store key information in the dictionary, per frequency layer
           self.dl_frequency_layer_info[dl_carrier_frequency_GHz] = {
               "available_PRBs": available_PRBs,
               "cells_in_frequency_mask": cells_in_frequency_mask,
               "cell_IDs": cell_IDs_in_frequency,
               "num_cells": num_cells_in_frequency,
               "cell_ants_in_frequency_mask": cell_ants_in_frequency_mask,
               "num_cell_ants": num_cell_ants_in_frequency,
               "ssb_beams_in_frequency_mask": ssb_beams_in_frequency_mask,
               "global_ssb_beams_IDs": global_ssb_beams_IDs,
               "csi_rs_beams_in_frequency_mask": csi_rs_beams_in_frequency_mask,
               "global_csi_rs_beams_IDs": global_csi_rs_beams_IDs
           }   
                       
           index += 1
        
       # Creating UL carrier
       
       # Group by carrier characterstics to find the different UL carriers       
       grouped_ul = self.network_deployment_obj.df_ep.groupby(['fdd_tdd_ind', #0
                                   'subframe_assignment', #1
                                   'FR', #2
                                   'subcarriers_per_PRB', #3
                                   'ofdm_symbols_in_slot', #4
                                   'ul_bandwidth_MHz', #5
                                   'ul_available_PRBs', #6
                                   'ul_subcarrier_spacing_kHz', #7
                                   'ul_ofdm_symbol_duration_us', #8
                                   'ul_control_channel_overhead', #9
                                   'ul_carrier_frequency_GHz']) #10
       
       # Iterate over the DL carriers
       for model_params, group in grouped_ul:
           self.ul_carrier_index[group['ID']] = index
           obj = carriers.Carrier(False,
                                  group['ID'],
                                  group['antenna_config_number_of_elements'],
                                  model_params[0], model_params[1], model_params[2], model_params[3], model_params[4], 
                                  model_params[5], model_params[6], model_params[7], model_params[8], model_params[9])
           
           self.ul_carriers.append(obj)  
           
           if self.simulation_config_obj.sn_indicator == 1 :
               self.ul_resource_grid_index[group['ID']] = index
               ul_rg = sionna.phy.ofdm.ResourceGrid(num_ofdm_symbols = 1,#model_params[4]
                                                fft_size = model_params[6],#model_params[6]*model_params[3]
                                                subcarrier_spacing = model_params[7] * 1e3 * model_params[3], #model_params[7]
                                                num_tx = 1,
                                                num_streams_per_tx = 1,
                                                num_guard_carriers = [0, 0],#[5,6]
                                                dc_null = True,
                                                pilot_pattern = "kronecker",
                                                pilot_ofdm_symbol_indices = [0])#[2, 11] 
           
               self.dl_resource_grid.append(ul_rg)           
           
           index += 1
           
       # Create a 2D array indicating whether each element coincides with any other element in the original array
       self.cell_by_cell_dl_carrier_mask = self.dl_carrier_index[:, np.newaxis] == self.dl_carrier_index
       self.cell_by_cell_ul_carrier_mask = self.ul_carrier_index[:, np.newaxis] == self.ul_carrier_index
       
       extended_dl_carrier_index = np.repeat(self.dl_carrier_index, self.cell_SSB_conf_obj.number_of_beams_per_node)
       self.SSB_beam_by_SSB_beam_dl_carrier_mask = extended_dl_carrier_index[:, np.newaxis] == extended_dl_carrier_index
       
       extended_dl_carrier_index = np.repeat(self.dl_carrier_index, self.cell_CSI_RS_conf_obj.number_of_beams_per_node)
       self.CSI_RS_beam_by_CSI_RS_beam_dl_carrier_mask = extended_dl_carrier_index[:, np.newaxis] == extended_dl_carrier_index       

       # Adding the indeces of the resource grid to the self.network_deployment_obj.df_ep
       self.network_deployment_obj.df_ep['dl_carrier_index'] = self.dl_carrier_index
       self.network_deployment_obj.df_ep['ul_carrier_index'] = self.ul_carrier_index       
       self.network_deployment_obj.df_ep['dl_resource_grid_index'] = self.dl_resource_grid_index
       self.network_deployment_obj.df_ep['ul_resource_grid_index'] = self.ul_resource_grid_index
       
       ##### End
       log_calculations_time('Time frequency resource', t_start)
       
       return rescheduling_us

       
    def beams_in_same_carrier_than_serving_beam(self, beam_type, best_serving_beam_of_ue):
       # This method creates a cell by cell or a beam by beam 2D array in which 
       # each row indicates if the cell or beam with that index uses the same carrier frequency as any other cell or beam 
       if beam_type == "antenna element" :
           beams_in_same_carrier_than_server = self.cell_by_cell_dl_carrier_mask[best_serving_beam_of_ue]
           
       elif beam_type == "SSB" :
           beams_in_same_carrier_than_server = self.SSB_beam_by_SSB_beam_dl_carrier_mask[best_serving_beam_of_ue]
       
       elif beam_type == "CSI_RS":
           beams_in_same_carrier_than_server = self.CSI_RS_beam_by_CSI_RS_beam_dl_carrier_mask[best_serving_beam_of_ue]
      
       return beams_in_same_carrier_than_server

   
    def beams_in_same_carrier_than_servring_cell(self, beam_type, beam_to_node_mapping, best_serving_cell_of_ue):
       # This method creates a cell by cell or a beam by beam 2D array in which 
       # each row indicates if the cell or beam with that index uses the same carrier frequency as any other cell or beam 
       if beam_type == "antenna element" :
           beams_in_same_carrier_than_server = self.cell_by_cell_dl_carrier_mask[best_serving_cell_of_ue]
           
       elif beam_type == "SSB" :
           best_serving_beam_of_ue = np.where(beam_to_node_mapping==best_serving_cell_of_ue)[0][0]
           beams_in_same_carrier_than_server = self.SSB_beam_by_SSB_beam_dl_carrier_mask[best_serving_beam_of_ue]
       
       elif beam_type == "CSI_RS":
           best_serving_beam_of_ue = np.where(beam_to_node_mapping==best_serving_cell_of_ue)[0][0]
           beams_in_same_carrier_than_server = self.CSI_RS_beam_by_CSI_RS_beam_dl_carrier_mask[best_serving_beam_of_ue]
      
       return beams_in_same_carrier_than_server
   

    def update_ue_info(
        self, 
        ue_deployment_obj: Any,
        ue_antenna_array_structure_obj: Any, 
        rescheduling_us: int = -1
    ) -> int:
        """
        Computes and updates carrier-related information for each frequency layer.
    
        Args:
            ue_antenna_array_structure_obj (Any): Object containing UE antenna-to-node mappings.
            best_serving_cell_ID_per_ue_obj (Any): Object containing the best serving cell ID per UE.
            rescheduling_us (int, optional): Rescheduling time in microseconds. Default is -1.
    
        Returns:
            int: The updated rescheduling time in microseconds.
        """
    
        # Update dictionary --all_freq entry-- without overwriting existing keys
        
        num_ues = len(ue_deployment_obj.df_ep)
        self.dl_frequency_layer_info["all_freq"].update({
            "ues_in_frequency_mask": np.ones((num_ues,), dtype=bool),
            "ue_IDs": np.arange(0, num_ues),
            "num_ues": num_ues,
            "ue_ants_in_frequency_mask": np.ones((len(ue_antenna_array_structure_obj.antenna_to_node_mapping),), dtype=bool),
            "num_ue_ants": len(ue_antenna_array_structure_obj.antenna_to_node_mapping)
        })    
                
        return rescheduling_us
    
    
    def update_ue_carrier_info(
        self, 
        ue_antenna_array_structure_obj: Any, 
        best_serving_cell_ID_per_ue_obj: Any, 
        SSB_RSRP_no_fast_fading_ue_to_cell_obj: Any,
        rescheduling_us: int = -1
    ) -> int:
        """
        Computes and updates carrier-related information for each frequency layer.
    
        Args:
            ue_antenna_array_structure_obj (Any): Object containing UE antenna-to-node mappings.
            best_serving_cell_ID_per_ue_obj (Any): Object containing the best serving cell ID per UE.
            SSB_RSRP_no_fast_fading_ue_to_cell_obj (Any): Object storing RSRP results without fast fading.
            rescheduling_us (int, optional): Rescheduling time in microseconds. Default is -1.
    
        Returns:
            int: The updated rescheduling time in microseconds.
        """
            
        # Update dictionary --all per frequency layer entries-- without overwriting existing keys
        
        # Retrieve best serving cell IDs for all UEs
        best_serving_cell_ID_per_ue: np.ndarray = best_serving_cell_ID_per_ue_obj.best_serving_cell_ID_per_ue
    
        # Process each frequency layer
        for frequency_key, frequency_layer_info in self.dl_frequency_layer_info.items():
            
            # Identify UEs and antennas associated with cells in the current frequency
            antenna_to_node_b_mapping: np.ndarray = ue_antenna_array_structure_obj.antenna_to_node_mapping
            ues_in_frequency_mask: np.ndarray = np.isin(best_serving_cell_ID_per_ue, frequency_layer_info["cell_IDs"])
            ue_IDs_in_frequency: np.ndarray = np.flatnonzero(ues_in_frequency_mask)  # Indices of UEs in this frequency layer
            num_ues_in_frequency: int = ue_IDs_in_frequency.size
            ue_ants_in_frequency_mask: np.ndarray = np.isin(antenna_to_node_b_mapping, ue_IDs_in_frequency)
            num_ue_ants_in_frequency: int = ue_ants_in_frequency_mask.sum()
    
            # Update dictionary without overwriting existing keys
            frequency_layer_info.update({
                "ues_in_frequency_mask": ues_in_frequency_mask,
                "ue_IDs": ue_IDs_in_frequency,
                "num_ues": num_ues_in_frequency,
                "ue_ants_in_frequency_mask": ue_ants_in_frequency_mask,
                "num_ue_ants": num_ue_ants_in_frequency
            })
          
        if (SSB_RSRP_no_fast_fading_ue_to_cell_obj != None):
            
            # Retrieve RSRP matrix containing information for all cells and UEs
            RSRP_ue_to_cell_all_layers_dBm: np.ndarray = SSB_RSRP_no_fast_fading_ue_to_cell_obj.RSRP_results_per_frequency_layer["all_freq"]["RSRP_ue_to_cell_dBm"]
            
            # Store a reference to "all_freq" separately to preserve its values
            all_freq_data = copy.deepcopy(SSB_RSRP_no_fast_fading_ue_to_cell_obj.RSRP_results_per_frequency_layer["all_freq"])
            
            # Set up a fresh multi-layer structure without modifying "all_freq"
            SSB_RSRP_no_fast_fading_ue_to_cell_obj.RSRP_results_per_frequency_layer = copy.deepcopy(self.dl_frequency_layer_info)
            
            # Reinsert "all_freq" to ensure it remains unchanged
            SSB_RSRP_no_fast_fading_ue_to_cell_obj.RSRP_results_per_frequency_layer["all_freq"] = all_freq_data
            
            # Process each frequency layer (excluding "all_freq" to prevent modification)
            for frequency_key, frequency_layer_info in SSB_RSRP_no_fast_fading_ue_to_cell_obj.RSRP_results_per_frequency_layer.items():
                if frequency_key == "all_freq":
                    continue  # Ensure "all_freq" remains unchanged
        
                # Extract and store RSRP information for each layer
                frequency_layer_info["RSRP_ue_to_cell_dBm"] = RSRP_ue_to_cell_all_layers_dBm[
                    np.ix_(frequency_layer_info["ues_in_frequency_mask"], frequency_layer_info["ssb_beams_in_frequency_mask"])
                ]
                 
        return rescheduling_us

        
       
       