#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:41:06 2024

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import time
from typing import List, Any

import numpy as np
import pandas as pd

from giulia.logger import debug, info
from giulia.outputs.saveable import saveables
from giulia.tools.tools import log_elapsed_time
from giulia.outputs.saveable import Saveable

class Performance(Saveable):
    """
    A class to store and process network performance metrics, including 
    signal strength (RSRP), SINR, throughput, and power consumption.
    """

    def __init__(
        self, 
        simulation_config_obj: Any,
        best_serving_cell_ID_per_ue_based_on_CRS_obj: Any,
        best_serving_cell_ID_per_ue_based_on_SSB_obj: Any,
        CRS_RSRP_no_fast_fading_ue_to_cell_obj: Any,
        SSB_RSRP_no_fast_fading_ue_to_cell_obj: Any,
        best_serving_CSI_RS_per_ue_obj: Any,
        CRS_sinr_ue_to_cell_obj: Any,
        SSB_sinr_ue_to_cell_obj: Any,
        CSI_RS_sinr_per_PRB_ue_to_cell_obj: Any,
        ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj: Any,
        ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_obj: Any,
        power_consumption_obj: Any
    ):
        
        super().__init__()

        """Initializes the Performance class with required network metrics objects."""

        # ---- Input Storage ----
        self.simulation_config_obj = simulation_config_obj
        self.best_serving_cell_ID_per_ue_based_on_CRS_obj: Any = best_serving_cell_ID_per_ue_based_on_CRS_obj
        self.best_serving_cell_ID_per_ue_based_on_SSB_obj: Any = best_serving_cell_ID_per_ue_based_on_SSB_obj
        self.CRS_RSRP_no_fast_fading_ue_to_cell_obj: Any = CRS_RSRP_no_fast_fading_ue_to_cell_obj
        self.SSB_RSRP_no_fast_fading_ue_to_cell_obj: Any = SSB_RSRP_no_fast_fading_ue_to_cell_obj
        self.best_serving_CSI_RS_per_ue_obj: Any = best_serving_CSI_RS_per_ue_obj
        self.CRS_sinr_ue_to_cell_obj: Any = CRS_sinr_ue_to_cell_obj
        self.SSB_sinr_ue_to_cell_obj: Any = SSB_sinr_ue_to_cell_obj
        self.CSI_RS_sinr_per_PRB_ue_to_cell_obj: Any = CSI_RS_sinr_per_PRB_ue_to_cell_obj
        self.ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj: Any = ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj
        self.ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_obj: Any = ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_obj
        self.power_consumption_obj: Any = power_consumption_obj        
        
        # ---- Output Storage ---- 
        
        # ---- CRS (Channel State Reporting Reference Signal) ----
        self.r_e_best_serving_cell_ID_per_ue_based_on_CRS: List[int] = []  
        self.r_e_best_serving_CRS_distance_3d_m: List[float] = []  
        self.r_e_best_serving_CRS_coupling_gain_dB: List[float] = [] 
        self.r_e_best_serving_CRS_re_noise_per_ue_dBm: List[float] = [] 
        self.r_e_best_serving_CRS_rsrp_per_ue_dBm: List[float] = [] 
        self.r_e_CRS_RSRP_no_fast_fading_ue_to_cell_dBm: List[np.ndarray] = [] 

        self.r_e_CRS_sinr_ue_to_cell_dB: List[np.ndarray] = [] 
        self.r_e_CRS_usefulPower_ue_dBm: List[np.ndarray] = []
        self.r_e_CRS_interfPower_ue_dBm: List[np.ndarray] = []
        
        # ---- SSB (Synchronization Signal Block) ----
        self.r_e_best_serving_cell_ID_per_ue_based_on_SSB: List[int] = []  
        self.r_e_best_serving_SSB_per_ue: List[int] = []
        self.r_e_best_serving_SSB_distance_3d_m: List[float] = []
        self.r_e_best_serving_SSB_coupling_gain_dB: List[float] = []
        self.r_e_best_serving_SSB_re_noise_per_ue_dBm: List[float] = []
        self.r_e_best_serving_SSB_rsrp_per_ue_dBm: List[float] = []
        self.r_e_SSB_RSRP_no_fast_fading_ue_to_cell_dBm: List[float] = []

        self.r_e_SSB_sinr_ue_to_cell_dB: List[np.ndarray] = []
        self.r_e_SSB_usefulPower_ue_dBm: List[np.ndarray] = []
        self.r_e_SSB_interfPower_ue_dBm: List[np.ndarray] = []
        
        # ---- CSI-RS (Channel State Information Reference Signal) ----
        self.r_e_best_serving_CSI_RS_per_ue: List[int] = []
        self.r_e_best_serving_CSI_RS_rsrp_per_ue_dBm: List[float] = []

        self.r_e_effective_CSI_RS_sinr_ue_to_cell_dB: List[np.ndarray] = []
        self.r_e_CSI_RS_usefulPower_ue_dBm: List[np.ndarray] = []
        self.r_e_CSI_RS_interfPower_ue_dBm: List[np.ndarray] = []   
        
        # ---- UE counts per cell and beam ----
        self.r_e_ues_per_cell_based_on_CRS: List[int] = [] 
        self.r_e_ues_per_cell_based_on_SSB: List[int] = []
        self.r_e_ues_per_SSB_beam: List[int] = []
        self.r_e_ues_per_CSI_RS_beam: List[int] = []
        
        # ---- Cell and beam activity ----
        self.r_e_cell_activity_per_ue: List[int] = [] 
        self.r_e_SSB_beam_activity_per_ue: List[int] = []
        self.r_e_CSI_RS_beam_activity_per_ue: List[int] = []

        # ---- Throughput and Power Consumption ----
        self.r_e_ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps: List[float] = []
        self.r_e_ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps: List[float] = []
        self.r_e_cell_throughput_Mbps: List[float] = []
        self.r_e_carrier_throughput_Mbps: List[np.ndarray] = []
        self.r_e_ue_throughput_per_carrier_Mbps: List[np.ndarray] = []
        
        self.r_e_power_consumption_perRadio_kW: List[float] = []  
        self.r_e_network_power_consumption_per_radioType_kW: List[float] = []
        self.r_e_total_network_power_consumption_kW: List[float] = []
        
        
    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["r_e_ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps"]


    def reset(self):  
        """Resets all stored performance metrics to empty lists."""
    
        # ---- CRS Metrics ----
        self.r_e_best_serving_cell_ID_per_ue_based_on_CRS.clear()
        self.r_e_best_serving_CRS_distance_3d_m.clear()
        self.r_e_best_serving_CRS_coupling_gain_dB.clear()
        self.r_e_best_serving_CRS_re_noise_per_ue_dBm.clear()        
        self.r_e_best_serving_CRS_rsrp_per_ue_dBm.clear()
        self.r_e_CRS_RSRP_no_fast_fading_ue_to_cell_dBm.clear()

        self.r_e_CRS_sinr_ue_to_cell_dB.clear()
        self.r_e_CRS_usefulPower_ue_dBm.clear()
        self.r_e_CRS_interfPower_ue_dBm.clear()

        # ---- SSB Metrics ----
        self.r_e_best_serving_cell_ID_per_ue_based_on_SSB.clear()
        self.r_e_best_serving_SSB_per_ue.clear()
        self.r_e_best_serving_SSB_distance_3d_m.clear()
        self.r_e_best_serving_SSB_coupling_gain_dB.clear()
        self.r_e_best_serving_SSB_re_noise_per_ue_dBm.clear()
        self.r_e_best_serving_SSB_rsrp_per_ue_dBm.clear()
        self.r_e_SSB_RSRP_no_fast_fading_ue_to_cell_dBm.clear()

        self.r_e_SSB_sinr_ue_to_cell_dB.clear()
        self.r_e_SSB_usefulPower_ue_dBm.clear()
        self.r_e_SSB_interfPower_ue_dBm.clear()

        # ---- CSI-RS Metrics ----
        self.r_e_best_serving_CSI_RS_per_ue.clear()
        self.r_e_best_serving_CSI_RS_rsrp_per_ue_dBm.clear()

        self.r_e_effective_CSI_RS_sinr_ue_to_cell_dB.clear()
        self.r_e_CSI_RS_usefulPower_ue_dBm.clear()
        self.r_e_CSI_RS_interfPower_ue_dBm.clear()
        
        # ---- UE counts per cell and beam ----
        self.r_e_ues_per_cell_based_on_CRS.clear()
        self.r_e_ues_per_cell_based_on_SSB.clear()
        self.r_e_ues_per_SSB_beam.clear()
        self.r_e_ues_per_CSI_RS_beam.clear()
        
        # ---- Cell and beam activity ----
        self.r_e_cell_activity_per_ue.clear()
        self.r_e_SSB_beam_activity_per_ue.clear()
        self.r_e_CSI_RS_beam_activity_per_ue.clear()     

        # ---- Throughput and Power Consumption ----
        self.r_e_ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps.clear()
        self.r_e_ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps.clear()
        self.r_e_cell_throughput_Mbps.clear()
        self.r_e_carrier_throughput_Mbps.clear()
        self.r_e_ue_throughput_per_carrier_Mbps.clear()

        self.r_e_power_consumption_perRadio_kW.clear()
        self.r_e_network_power_consumption_per_radioType_kW.clear()
        self.r_e_total_network_power_consumption_kW.clear()


    def get_combined_rsrp_matrix(self, RSRP_no_fast_fading_ue_to_cell_obj) -> np.ndarray:
        """
        Constructs a unified RSRP array for all UEs and BSs across all frequency layers.
    
        Returns:
            np.ndarray: A 2D array where each row represents a UE and each column a BS,
                        with values corresponding to RSRP (dBm).
        """
        # Extract RSRP dictionary
        rsrp_layers = RSRP_no_fast_fading_ue_to_cell_obj.RSRP_results_per_frequency_layer
    
        # Determine the total size of the RSRP matrix based on the first layer
        first_layer = next(iter(rsrp_layers.values()))
        matrix_shape = np.shape(first_layer["RSRP_ue_to_cell_dBm"])
        
        # Preallocate the final RSRP matrix with NaN values
        rsrp_matrix = np.full(matrix_shape, np.nan)

    
        # Iterate through each frequency layer and populate the matrix
        for key, layer_data in rsrp_layers.items():
            if self.simulation_config_obj.preset == "GiuliaMfl" and key == "all_freq":
                continue
            
            # Extract boolean masks for active UEs and BSs
            active_ue_mask = layer_data["ues_in_frequency_mask"]
            
            # Check which mask matches the second dimension of the matrix
            if matrix_shape[1] == len(first_layer["cells_in_frequency_mask"]):
                active_bs_mask = layer_data["cells_in_frequency_mask"]
            elif matrix_shape[1] == len(first_layer["ssb_beams_in_frequency_mask"]):
                active_bs_mask = layer_data["ssb_beams_in_frequency_mask"]
            else:
                raise ValueError("No matching active_bs_mask found for the given matrix shape.")            
                
            # Extract RSRP values for active elements
            rsrp_values = layer_data["RSRP_ue_to_cell_dBm"]
    
            # Fill the final matrix at correct indices
            rsrp_matrix[np.ix_(active_ue_mask, active_bs_mask)] = rsrp_values
    
        return rsrp_matrix


    def get_combined_usefulInterf_matrix(self, sinr_ue_to_cell_obj, label) -> np.ndarray:
        return self.get_combined_sinr_matrix(sinr_ue_to_cell_obj, label)


    def get_combined_sinr_matrix(self, sinr_ue_to_cell_obj, label) -> np.ndarray:
        """
        Constructs a unified SINR array for all UEs across all frequency layers.
    
        Returns:
            np.ndarray: A 1D array where each entry represents a UE,
                        with values corresponding to SINR (dB).
        """
        # Extract SINR dictionary
        sinr_layers = sinr_ue_to_cell_obj.SINR_results_per_frequency_layer
    
        # Get total number of UEs (assuming all layers share the same dimensions)
        first_layer = next(iter(sinr_layers.values()))
        total_ues = len(first_layer["ues_in_frequency_mask"])
    
        # Preallocate the final SINR array (size: total_ues) with NaNs
        sinr_matrix = np.full(total_ues, np.nan)
    
        # Iterate through each frequency layer and populate the matrix
        for key, layer_data in sinr_layers.items():
            if self.simulation_config_obj.preset == "GiuliaMfl" and key == "all_freq":
                continue
            
            # Extract boolean mask for active UEs
            active_ue_mask = layer_data["ues_in_frequency_mask"]
    
            # Extract SINR values for active UEs
            sinr_values = layer_data[label]
    
            # Assign values to the final array using boolean indexing
            sinr_matrix[active_ue_mask] = sinr_values
    
        return sinr_matrix


    def safe_append(self, attr_name, value):
        # Check if the attributes exist before appending to prevent AttributeError
        if hasattr(self, attr_name):
            getattr(self, attr_name).append(value)


    def process(self, rescheduling_us=-1):
        
        # Start timer
        t_start = time.perf_counter()
        
        self.r_e_CSI_RS_usefulPower_ue_dBm = []
        self.r_e_CSI_RS_interfPower_ue_dBm = []
        
        # Convert the corresponding outputs to a single matrix format
        # CRS
        crs_rsrp_ue_to_cell_dBm = (
            self.get_combined_rsrp_matrix(getattr(self, "CRS_RSRP_no_fast_fading_ue_to_cell_obj", None))
            if getattr(self, "CRS_RSRP_no_fast_fading_ue_to_cell_obj", None) is not None else np.array([])
        )
        
        crs_sinr_ue_to_cell_dB = (
            self.get_combined_sinr_matrix(getattr(self, "CRS_sinr_ue_to_cell_obj", None), "sinr_ue_to_cell_dB")
            if getattr(self, "CRS_sinr_ue_to_cell_obj", None) is not None else np.array([])
        )
        
        crs_usefulPower_ue_dBm = (
            self.get_combined_usefulInterf_matrix(getattr(self, "CRS_sinr_ue_to_cell_obj", None), "usefulPower_ue_dBm")
            if getattr(self, "CRS_sinr_ue_to_cell_obj", None) is not None else np.array([])
        )
        
        
        crs_interfPower_ue_dBm = (
            self.get_combined_usefulInterf_matrix(getattr(self, "CRS_sinr_ue_to_cell_obj", None), "interfPower_ue_dBm")
            if getattr(self, "CRS_sinr_ue_to_cell_obj", None) is not None else np.array([])
        )
        
        
        # SSB
        ssb_rsrp_ue_to_cell_dBm = (
            self.get_combined_rsrp_matrix(getattr(self, "SSB_RSRP_no_fast_fading_ue_to_cell_obj", None))
            if getattr(self, "SSB_RSRP_no_fast_fading_ue_to_cell_obj", None) is not None else np.array([])
        )        
        
        ssb_sinr_ue_to_cell_dB = (
            self.get_combined_sinr_matrix(getattr(self, "SSB_sinr_ue_to_cell_obj", None), "sinr_ue_to_cell_dB")
            if getattr(self, "SSB_sinr_ue_to_cell_obj", None) is not None else np.array([])
        )
        
        
        SSB_usefulPower_ue_dBm = (
            self.get_combined_usefulInterf_matrix(getattr(self, "SSB_sinr_ue_to_cell_obj", None), "usefulPower_ue_dBm")
            if getattr(self, "SSB_sinr_ue_to_cell_obj", None) is not None else np.array([])
        )
        
        
        SSB_interfPower_ue_dBm = (
            self.get_combined_usefulInterf_matrix(getattr(self, "SSB_sinr_ue_to_cell_obj", None), "interfPower_ue_dBm")
            if getattr(self, "SSB_sinr_ue_to_cell_obj", None) is not None else np.array([])
        )
        

        # CSI-RS
        effective_csi_rs_sinr_ue_to_cell_dB = (
            self.get_combined_sinr_matrix(getattr(self, "CSI_RS_sinr_per_PRB_ue_to_cell_obj", None), "effective_sinr_ue_to_cell_dB")
            if getattr(self, "CSI_RS_sinr_per_PRB_ue_to_cell_obj", None) is not None else np.array([])
        )
        
        
        carrier_throughput_Mbps = np.array([
            layer for layer in getattr(self.ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj, "carrier_throughput_Mbps", {}).values()
        ]) if getattr(self, "ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj", None) is not None else np.array([])
        
        ue_throughput_per_carrier_Mbps = np.array([
            np.array(layer) for layer in getattr(self.ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj, "ue_throughput_per_carrier_Mbps", {}).values()
        ]) if getattr(self, "ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj", None) is not None else np.array([])

        # Reshape for uniformed output format
        if self.simulation_config_obj.preset == "GiuliaStd":
            ue_throughput_per_carrier_Mbps = self._reshape_ue_throughput_per_carrier_std_to_mfl_format(ue_throughput_per_carrier_Mbps)
         
        
        power_consumption_per_radioType_kW = np.array([
            np.array(layer) for layer in getattr(self.power_consumption_obj, "power_consumption_results_per_radioType_kW_dict", {}).values()
        ]) if getattr(self, "power_consumption_obj", None) is not None else np.array([])
        

        #### Save the corresponding outputs
        # ---- CRS (Channel State Reporting Reference Signal) ----
        self.safe_append("r_e_best_serving_cell_ID_per_ue_based_on_CRS", getattr(self.best_serving_cell_ID_per_ue_based_on_CRS_obj, "best_serving_cell_ID_per_ue", np.array([])))
        self.safe_append("r_e_best_serving_CRS_distance_3d_m", getattr(self.best_serving_cell_ID_per_ue_based_on_CRS_obj, "best_server_distance_3d_per_ue_m", np.array([])))
        self.safe_append("r_e_best_serving_CRS_coupling_gain_dB", getattr(self.best_serving_cell_ID_per_ue_based_on_CRS_obj, "best_server_coupling_gain_per_ue_dB", np.array([])))
        self.safe_append("r_e_best_serving_CRS_re_noise_per_ue_dBm", getattr(self.best_serving_cell_ID_per_ue_based_on_CRS_obj, "best_server_re_noise_per_ue_dBm", np.array([])))
        self.safe_append("r_e_best_serving_CRS_rsrp_per_ue_dBm", getattr(self.best_serving_cell_ID_per_ue_based_on_CRS_obj, "best_server_rsrp_per_ue_dBm", np.array([])))
        self.safe_append("r_e_CRS_RSRP_no_fast_fading_ue_to_cell_dBm", crs_rsrp_ue_to_cell_dBm)

        self.safe_append("r_e_CRS_sinr_ue_to_cell_dB", crs_sinr_ue_to_cell_dB)
        self.safe_append("r_e_CRS_usefulPower_ue_dBm", crs_usefulPower_ue_dBm) # LTE TYPE - Useful Power
        self.safe_append("r_e_CRS_interfPower_ue_dBm", crs_interfPower_ue_dBm) # LTE TYPE - Interf Power
        
        # ---- SSB (Synchronization Signal Block) ----
        self.safe_append("r_e_best_serving_cell_ID_per_ue_based_on_SSB", getattr(self.best_serving_cell_ID_per_ue_based_on_SSB_obj, "best_serving_cell_ID_per_ue", np.array([])))
        self.safe_append("r_e_best_serving_SSB_per_ue", getattr(self.best_serving_cell_ID_per_ue_based_on_SSB_obj, "best_serving_beam_ID_per_ue", np.array([])))
        self.safe_append("r_e_best_serving_SSB_distance_3d_m", getattr(self.best_serving_cell_ID_per_ue_based_on_SSB_obj, "best_server_distance_3d_per_ue_m", np.array([])))
        self.safe_append("r_e_best_serving_SSB_coupling_gain_dB", getattr(self.best_serving_cell_ID_per_ue_based_on_SSB_obj, "best_server_coupling_gain_per_ue_dB", np.array([])))
        self.safe_append("r_e_best_serving_SSB_re_noise_per_ue_dBm", getattr(self.best_serving_cell_ID_per_ue_based_on_SSB_obj, "best_server_re_noise_per_ue_dBm", np.array([])))
        self.safe_append("r_e_best_serving_SSB_rsrp_per_ue_dBm", getattr(self.best_serving_cell_ID_per_ue_based_on_SSB_obj, "best_server_rsrp_per_ue_dBm", np.array([])))
        self.safe_append("r_e_SSB_RSRP_no_fast_fading_ue_to_cell_dBm", ssb_rsrp_ue_to_cell_dBm)

        self.safe_append("r_e_SSB_sinr_ue_to_cell_dB", ssb_sinr_ue_to_cell_dB)
        self.safe_append("r_e_SSB_usefulPower_ue_dBm", SSB_usefulPower_ue_dBm) # SSB - Useful Power 
        self.safe_append("r_e_SSB_interfPower_ue_dBm", SSB_interfPower_ue_dBm) # SSB - Interf Power
        
        # ---- CSI-RS (Channel State Information Reference Signal) ----
        self.safe_append("r_e_best_serving_CSI_RS_per_ue", getattr(self.best_serving_CSI_RS_per_ue_obj, "best_serving_beam_ID_per_ue", np.array([])))
        self.safe_append("r_e_best_serving_CSI_RS_rsrp_per_ue_dBm", getattr(self.best_serving_CSI_RS_per_ue_obj, "best_serving_beam_rsrp_per_ue_dBm", np.array([])))
        self.safe_append("r_e_effective_CSI_RS_sinr_ue_to_cell_dB", effective_csi_rs_sinr_ue_to_cell_dB)
        
        # ---- UE counts per cell and beam ----
        self.safe_append("r_e_ues_per_cell_based_on_CRS", getattr(self.best_serving_cell_ID_per_ue_based_on_CRS_obj, "ues_per_beam", np.array([])))
        self.safe_append("r_e_ues_per_cell_based_on_SSB", getattr(self.best_serving_cell_ID_per_ue_based_on_SSB_obj, "ues_per_cell", np.array([])))
        self.safe_append("r_e_ues_per_SSB_beam", getattr(self.best_serving_cell_ID_per_ue_based_on_SSB_obj, "ues_per_beam", np.array([])))
        self.safe_append("r_e_ues_per_CSI_RS_beam", getattr(self.best_serving_CSI_RS_per_ue_obj, "ues_per_beam", np.array([])))
        
        # ---- Cell and beam activity ----
        self.safe_append("r_e_cell_activity_per_ue", getattr(self.best_serving_cell_ID_per_ue_based_on_CRS_obj, "beam_activity_per_ue", np.array([])))
        self.safe_append("r_e_SSB_beam_activity_per_ue", getattr(self.best_serving_cell_ID_per_ue_based_on_SSB_obj, "beam_activity_per_ue", np.array([])))
        self.safe_append("r_e_CSI_RS_beam_activity_per_ue", getattr(self.best_serving_CSI_RS_per_ue_obj, "beam_activity_per_ue", np.array([])))   
        
        
        # ---- Throughput and Power Consumption ----
        self.safe_append("r_e_ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps", getattr(self.ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj, "ue_throughput_Mbps", np.array([])))
        self.safe_append("r_e_ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps", getattr(self.ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_obj, "ue_throughput_Mbps", np.array([])))
        self.safe_append("r_e_cell_throughput_Mbps", getattr(self.ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj, "cell_throughput_Mbps", np.array([])))
        self.safe_append("r_e_carrier_throughput_Mbps", carrier_throughput_Mbps)
        self.safe_append("r_e_ue_throughput_per_carrier_Mbps", ue_throughput_per_carrier_Mbps)
        
        self.safe_append("r_e_power_consumption_perRadio_kW", getattr(self.power_consumption_obj, "power_consumption_perRadio_kW", np.array([])))
        self.safe_append("r_e_network_power_consumption_per_radioType_kW", power_consumption_per_radioType_kW)
        self.safe_append("r_e_total_network_power_consumption_kW", getattr(self.power_consumption_obj, "total_RAN_power_consumption_kW", np.array([])))
        
        # Log execution time
        log_elapsed_time('Saving selected outputs', t_start)

        # End 
        return rescheduling_us 



    def _reshape_ue_throughput_per_carrier_std_to_mfl_format(self, ue_throughput_per_carrier_Mbps) -> np.ndarray:
        # Get each UE serving cell
        # Note that this is the same serving cell used for thp computation
        serving_cell_array = \
            getattr(self.ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj.best_serving_cell_per_ue_obj, "best_serving_cell_ID_per_ue", None)
        assert isinstance(serving_cell_array, np.ndarray), "Error, serving_cell_array is not a numpy array"
            
        # Get network dataframe eng parameters
        df_ep = getattr(self.ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj.network_deployment_obj, "df_ep", None)
        assert isinstance(df_ep, pd.DataFrame), "Error, df_ep is not a DataFrame"
    
        # Get network deployed frequencies
        network_freq_deployed = np.unique(df_ep['dl_carrier_frequency_GHz'])
        
        # Create mapping cell_ID -> freq
        cell2freq = dict(zip(df_ep['ID'], df_ep['dl_carrier_frequency_GHz']))

        # Create mapping freq -> index of freq_available
        freq2idx = {f: i for i, f in enumerate(network_freq_deployed)}
    
        # Init auxiliary array for output
        ue_throughput_per_carrier_Mbps_aux = np.full_like(ue_throughput_per_carrier_Mbps, np.nan)
        ue_throughput_per_carrier_Mbps_aux = np.tile(ue_throughput_per_carrier_Mbps_aux, (len(network_freq_deployed), 1))
        
        # Populate the new auxiliary output
        throughputs_1d = np.atleast_1d(ue_throughput_per_carrier_Mbps).ravel()
        serving_cells_1d = np.atleast_1d(serving_cell_array).ravel()
        if throughputs_1d.size != serving_cells_1d.size:
            raise ValueError(f"Length mismatch: throughputs {throughputs_1d.size} vs serving_cells {serving_cells_1d.size}")
        
        # Populate the new auxiliary output
        for ue_idx, (throughput_mbps, serving_cell) in enumerate(zip(throughputs_1d, serving_cells_1d)):
            ue_throughput_per_carrier_Mbps_aux[freq2idx[cell2freq[serving_cell]], ue_idx] = throughput_mbps
       
        # Return reshaped array
        return ue_throughput_per_carrier_Mbps_aux.astype(np.float64)
    
    