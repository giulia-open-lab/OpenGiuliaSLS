#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 14:41:06 2024

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@itea.upv.es
"""

import time
from typing import Any, Optional, List, Dict

import numpy as np
import numpy.typing as npt
import pandas as pd

from giulia.config.sim_config import Simulation_Config
from giulia.logger import debug
from giulia.mac import calculate_best_serving_beam
from giulia.rrc.cell_selections import Cell_Selection
from giulia.tools.tools import log_elapsed_time
from giulia.outputs.saveable import _get_saveables
from giulia.outputs.utils import _reshape_ue_throughput_per_carrier_std_to_mfl_format


class Performance:

    def __init__(
        self, 
        simulation_config_obj: Simulation_Config,
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
        power_consumption_obj: Any,
        time_frequency_resource_obj: Any,
        base_stations_obj: Any
    ):

        """Initializes the Performance class with required network metrics objects."""

        # ---- Input Storage ----
        self.simulation_config_obj: Simulation_Config = simulation_config_obj
        self.best_serving_cell_ID_per_ue_based_on_CRS_obj = best_serving_cell_ID_per_ue_based_on_CRS_obj
        self.best_serving_cell_ID_per_ue_based_on_SSB_obj = best_serving_cell_ID_per_ue_based_on_SSB_obj
        self.CRS_RSRP_no_fast_fading_ue_to_cell_obj = CRS_RSRP_no_fast_fading_ue_to_cell_obj
        self.SSB_RSRP_no_fast_fading_ue_to_cell_obj = SSB_RSRP_no_fast_fading_ue_to_cell_obj
        self.best_serving_CSI_RS_per_ue_obj = best_serving_CSI_RS_per_ue_obj
        self.CRS_sinr_ue_to_cell_obj = CRS_sinr_ue_to_cell_obj
        self.SSB_sinr_ue_to_cell_obj = SSB_sinr_ue_to_cell_obj
        self.CSI_RS_sinr_per_PRB_ue_to_cell_obj = CSI_RS_sinr_per_PRB_ue_to_cell_obj
        self.ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj = ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj
        self.ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_obj = ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_obj
        self.power_consumption_obj = power_consumption_obj
        self.time_frequency_resource_obj: Any = time_frequency_resource_obj      
        self.base_stations_obj: Any = base_stations_obj
        
        # ---- Output Storage ---- 

        # Reset the instance in order to clear all the memory arrays
        self.reset()


    def reset(self):
        """Resets all stored performance metrics to empty lists."""

        # ---- CRS (Channel State Reporting Reference Signal) ----
        self.r_e_best_serving_cell_ID_per_ue_based_on_CRS: npt.NDArray[np.int_] = np.array([], dtype=int)
        self.r_e_best_serving_CRS_distance_3d_m: npt.NDArray[np.float64] = np.array([], dtype=float)
        self.r_e_best_serving_CRS_coupling_gain_dB: npt.NDArray[np.float64] = np.array([], dtype=float)
        self.r_e_best_serving_CRS_re_noise_per_ue_dBm: npt.NDArray[np.float64] = np.array([], dtype=float)
        self.r_e_best_serving_CRS_rsrp_per_ue_dBm: npt.NDArray[np.float64] = np.array([], dtype=float)
        self.r_e_CRS_RSRP_no_fast_fading_ue_to_cell_dBm: npt.NDArray[np.object_] = np.array([], dtype=np.ndarray)

        self.r_e_CRS_sinr_ue_to_cell_dB: npt.NDArray[np.object_] = np.array([], np.ndarray)
        self.r_e_CRS_usefulPower_ue_dBm: npt.NDArray[np.object_] = np.array([], np.ndarray)
        self.r_e_CRS_interfPower_ue_dBm: npt.NDArray[np.object_] = np.array([], np.ndarray)

        # ---- SSB (Synchronization Signal Block) ----
        self.r_e_best_serving_cell_ID_per_ue_based_on_SSB: npt.NDArray[np.int_] = np.array([], dtype=int)
        self.r_e_best_serving_SSB_per_ue: npt.NDArray[np.int_] = np.array([], dtype=int)
        self.r_e_best_serving_SSB_distance_3d_m: npt.NDArray[np.float64] = np.array([], dtype=float)
        self.r_e_best_serving_SSB_coupling_gain_dB: npt.NDArray[np.float64] = np.array([], dtype=float)
        self.r_e_best_serving_SSB_re_noise_per_ue_dBm: npt.NDArray[np.float64] = np.array([], dtype=float)
        self.r_e_best_serving_SSB_rsrp_per_ue_dBm: npt.NDArray[np.float64] = np.array([], dtype=float)
        self.r_e_SSB_RSRP_no_fast_fading_ue_to_cell_dBm: npt.NDArray[np.float64] = np.array([], dtype=float)

        self.r_e_SSB_sinr_ue_to_cell_dB: npt.NDArray[np.object_] = np.array([], np.ndarray)
        self.r_e_SSB_usefulPower_ue_dBm: npt.NDArray[np.object_] = np.array([], np.ndarray)
        self.r_e_SSB_interfPower_ue_dBm: npt.NDArray[np.object_] = np.array([], np.ndarray)

        # ---- CSI-RS (Channel State Information Reference Signal) ----
        self.r_e_best_serving_CSI_RS_per_ue: npt.NDArray[np.int_] = np.array([], dtype=int)
        self.r_e_best_serving_CSI_RS_rsrp_per_ue_dBm: npt.NDArray[np.float64] = np.array([], dtype=float)

        self.r_e_effective_CSI_RS_sinr_ue_to_cell_dB: npt.NDArray[np.object_] = np.array([], np.ndarray)
        self.r_e_CSI_RS_usefulPower_ue_dBm: npt.NDArray[np.object_] = np.array([], np.ndarray)
        self.r_e_CSI_RS_interfPower_ue_dBm: npt.NDArray[np.object_] = np.array([], np.ndarray)

        # ---- UE counts per cell and beam ----
        self.r_e_ues_per_cell_based_on_CRS: npt.NDArray[np.int_] = np.array([], dtype=int)
        self.r_e_ues_per_cell_based_on_SSB: npt.NDArray[np.int_] = np.array([], dtype=int)
        self.r_e_ues_per_SSB_beam: npt.NDArray[np.int_] = np.array([], dtype=int)
        self.r_e_ues_per_CSI_RS_beam: npt.NDArray[np.int_] = np.array([], dtype=int)

        # ---- Cell and beam activity ----
        self.r_e_cell_activity_per_ue: npt.NDArray[np.int_] = np.array([], dtype=int)
        self.r_e_SSB_beam_activity_per_ue: npt.NDArray[np.int_] = np.array([], dtype=int)
        self.r_e_CSI_RS_beam_activity_per_ue: npt.NDArray[np.int_] = np.array([], dtype=int)
        self.r_e_PRB_usage_per_cell: npt.NDArray[np.float64] = np.array([], dtype=float)

        # ---- Base station radio-unit state ----
        self.r_e_cell_radio_unit_statuses: npt.NDArray[np.object_] = np.array([], dtype=object)

        # ---- Base station radio-unit state ----
        self.r_e_cell_radio_unit_statuses: npt.NDArray[np.object_] = np.array([], dtype=object)

        # ---- Throughput and Power Consumption ----
        self.r_e_ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps: npt.NDArray[np.float64] = np.array([], dtype=float)
        self.r_e_ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps: npt.NDArray[np.float64] = np.array([], dtype=float)
        self.r_e_cell_throughput_Mbps: npt.NDArray[np.float64] = np.array([], dtype=float)
        self.r_e_carrier_throughput_Mbps: npt.NDArray[np.object_] = np.array([], np.ndarray)
        self.r_e_ue_throughput_per_carrier_Mbps: npt.NDArray[np.object_] = np.array([], np.ndarray)

        self.r_e_power_consumption_per_radio_unit_incl_bbu_kW: npt.NDArray[np.float64] = np.array([], dtype=float)
        self.r_e_power_consumption_per_radio_unit_kW: npt.NDArray[np.float64] = np.array([], dtype=float)
        self.r_e_network_power_consumption_per_radioType_kW: npt.NDArray[np.float64] = np.array([], dtype=float)
        self.r_e_total_network_power_consumption_kW: npt.NDArray[np.float64] = np.array([], dtype=float)

        self.r_e_saveables_data: Dict[str, Any] = {}


    def get_combined_rsrp_matrix(self, RSRP_no_fast_fading_ue_to_cell_obj) -> np.ndarray:
        return self._get_combined_rsrp_matrix(RSRP_no_fast_fading_ue_to_cell_obj,
                                              self.time_frequency_resource_obj,
                                              self.simulation_config_obj.preset)

    @staticmethod
    def _get_combined_rsrp_matrix(RSRP_no_fast_fading_ue_to_cell_obj, time_frequency_resource_obj,
                                  preset: str | None = None) -> np.ndarray:
        """
        Constructs a unified RSRP array for all UEs and BSs across all frequency layers.

        Returns:
            np.ndarray: A 2D array where each row represents a UE and each column a BS,
                        with values corresponding to RSRP (dBm).
        """
        # Extract RSRP dictionary
        rsrp_layers = RSRP_no_fast_fading_ue_to_cell_obj.RSRP_results_per_frequency_layer

        # Get the corresponding resource information for the current frequency layer
        time_frequency_resource_layers: Dict[str, Dict[str, Any]] = time_frequency_resource_obj.dl_frequency_layer_info

        # Determine the total size of the RSRP matrix based on the first layer
        rsrp_first_layer = next(iter(rsrp_layers.values()))
        time_freq_rsc_first_layer = next(iter(time_frequency_resource_layers.values()))

        matrix_shape = np.shape(rsrp_first_layer["RSRP_ue_to_cell_dBm"])

        # Preallocate the final RSRP matrix with NaN values
        rsrp_matrix = np.full(matrix_shape, np.nan)


        # Iterate through each frequency layer and populate the matrix
        for key, rsrp_layer_data in rsrp_layers.items():
            # Skip the aggregate "all_freq" layer only when real per-frequency
            # layers exist (e.g. SSB/CSI-RS). For CRS the only layer is "all_freq",
            # so it must be kept - otherwise the combined matrix stays all-NaN.
            if preset == "GiuliaMfl" and key == "all_freq" and len(rsrp_layers) > 1:
                continue

            # Get the corresponding resource information for the current frequency layer
            time_freq_rsc_layer = time_frequency_resource_layers[key]

            # Extract boolean masks for active UEs and BSs
            active_ue_mask = time_freq_rsc_layer["ues_in_frequency_mask"]
            #active_bs_mask = layer_data["cells_in_frequency_mask"]
            
            # Check which mask matches the second dimension of the matrix
            if matrix_shape[1] == len(time_freq_rsc_first_layer["cells_in_frequency_mask"]):
                active_bs_mask = time_freq_rsc_layer["cells_in_frequency_mask"]
            elif matrix_shape[1] == len(time_freq_rsc_first_layer["ssb_beams_in_frequency_mask"]):
                active_bs_mask = time_freq_rsc_layer["ssb_beams_in_frequency_mask"]
            else:
                raise ValueError("No matching active_bs_mask found for the given matrix shape.")            
                
            # Extract RSRP values for active elements
            rsrp_values = rsrp_layer_data["RSRP_ue_to_cell_dBm"]
    
            # Fill the final matrix at correct indices
            rsrp_matrix[np.ix_(active_ue_mask, active_bs_mask)] = rsrp_values
    
        return rsrp_matrix


    def get_combined_usefulInterf_matrix(self, sinr_ue_to_cell_obj, label) -> np.ndarray:
        return self.get_combined_sinr_matrix(sinr_ue_to_cell_obj, label)
    
    def get_combined_sinr_matrix(self, sinr_ue_to_cell_obj, label) -> np.ndarray:
        return self._get_combined_sinr_matrix(sinr_ue_to_cell_obj, self.time_frequency_resource_obj,
                                              label, self.simulation_config_obj.preset)

    @staticmethod
    def _get_combined_sinr_matrix(sinr_ue_to_cell_obj, time_frequency_resource_obj,
                                  label, preset: str | None = None) -> np.ndarray:
        """
        Constructs a unified SINR array for all UEs across all frequency layers.
    
        Returns:
            np.ndarray: A 1D array where each entry represents a UE,
                        with values corresponding to SINR (dB).
        """
        # Extract SINR dictionary
        sinr_layers = sinr_ue_to_cell_obj.SINR_results_per_frequency_layer

        # Get the corresponding resource information for the current frequency layer
        time_frequency_resource_layers: Dict[str, Dict[str, Any]] = time_frequency_resource_obj.dl_frequency_layer_info
    

        # Get total number of UEs (assuming all layers share the same dimensions)
        sinr_first_layer = next(iter(sinr_layers.values()))
        time_freq_rsc_first_layer = next(iter(time_frequency_resource_layers.values()))
        total_ues = len(time_freq_rsc_first_layer["ues_in_frequency_mask"])

        # Preallocate the final SINR array (size: total_ues) with NaNs
        sinr_matrix = np.full(total_ues, np.nan, dtype=float)

        # Iterate through each frequency layer and populate the matrix
        for key, sinr_layer_data in sinr_layers.items():
            # Skip the aggregate "all_freq" layer only when real per-frequency
            # layers exist (e.g. SSB/CSI-RS). For CRS the only layer is "all_freq",
            # so it must be kept - otherwise the combined matrix stays all-NaN.
            if preset == "GiuliaMfl" and key == "all_freq" and len(sinr_layers) > 1:
                continue

            # Extract boolean mask for active UEs from the layer-membership snapshot
            # captured when the SINR was computed, so the scatter below matches the length
            # of the stored SINR array even if UEs have since re-associated (which changes
            # the live dl_frequency_layer_info mask).
            active_ue_mask = np.asarray(sinr_layer_data["ues_in_frequency_mask"], dtype=bool)

            # Extract SINR values for active UEs
            sinr_values = np.asarray(sinr_layer_data[label], dtype=float)

            # Assign values to the final array using boolean indexing
            sinr_matrix[active_ue_mask] = sinr_values

        return sinr_matrix

    def safe_append(self, variable_name: str, value: Any):
        # Ensure the attribute exists on the object.
        # If it does not exist, do nothing to avoid AttributeError.
        if not hasattr(self, variable_name):
            return

        # Retrieve the current numpy array stored in the attribute
        current_arr = getattr(self, variable_name)

        # If the array is empty (first insertion case)
        if current_arr.size == 0:
            new_arr = np.array([value])

        # Otherwise, concatenate the new value
        else:
            new_arr = np.concatenate((current_arr, np.array([value])), axis=0)

        # Update the attribute with the newly created array
        setattr(self, variable_name, new_arr)

    def _safe_append_saveable(self, variable_name: str, value: Any):
        """
        Appends a value to the list of values for a specific saveable variable.
        This is used to store data for saveables that are not part of the core performance metrics.
        """
        if variable_name not in self.r_e_saveables_data:
            self.r_e_saveables_data[variable_name] = []
        
        self.r_e_saveables_data[variable_name].append(value)


    def process(self, rescheduling_us=-1):
        
        # Start timer
        t_start = time.perf_counter()

        self.r_e_CSI_RS_usefulPower_ue_dBm = np.array([], dtype=np.ndarray)
        self.r_e_CSI_RS_interfPower_ue_dBm = np.array([], dtype=np.ndarray)
        
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
            ue_throughput_per_carrier_Mbps = _reshape_ue_throughput_per_carrier_std_to_mfl_format(self.ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj, 
                                                                                                  ue_throughput_per_carrier_Mbps)
         
        
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
        # self.safe_append("r_e_CSI_RS_usefulPower_ue_dBm", CSI_RS_usefulPower_ue_dBm) # CSI_RS - Useful Power
        # self.safe_append("r_e_CSI_RS_interfPower_ue_dBm", CSI_RS_interfPower_ue_dBm) # CSI_RS - Interf Power
        
        # ---- UE counts per cell and beam ----
        self.safe_append("r_e_ues_per_cell_based_on_CRS", getattr(self.best_serving_cell_ID_per_ue_based_on_CRS_obj, "ues_per_beam", np.array([])))
        self.safe_append("r_e_ues_per_cell_based_on_SSB", getattr(self.best_serving_cell_ID_per_ue_based_on_SSB_obj, "ues_per_cell", np.array([])))
        self.safe_append("r_e_ues_per_SSB_beam", getattr(self.best_serving_cell_ID_per_ue_based_on_SSB_obj, "ues_per_beam", np.array([])))
        self.safe_append("r_e_ues_per_CSI_RS_beam", getattr(self.best_serving_CSI_RS_per_ue_obj, "ues_per_beam", np.array([])))
        
        # ---- Cell and beam activity ----
        self.safe_append("r_e_cell_activity_per_ue", getattr(self.best_serving_cell_ID_per_ue_based_on_CRS_obj, "beam_activity_per_ue", np.array([])))
        self.safe_append("r_e_SSB_beam_activity_per_ue", getattr(self.best_serving_cell_ID_per_ue_based_on_SSB_obj, "beam_activity_per_ue", np.array([])))
        self.safe_append("r_e_CSI_RS_beam_activity_per_ue", getattr(self.best_serving_CSI_RS_per_ue_obj, "beam_activity_per_ue", np.array([])))   
        self.safe_append("r_e_PRB_usage_per_cell",
                          np.array([bs_obj.mac_layer.scheduler_obj._prb_usage for bs_obj in self.base_stations_obj.base_stations], dtype=float))

        # ---- Base station radio-unit state ----
        self.safe_append("r_e_cell_radio_unit_statuses", self.base_stations_obj.cell_radio_unit_statuses)


        # ---- Throughput and Power Consumption ----
        self.safe_append("r_e_ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps", getattr(self.ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj, "ue_throughput_Mbps", np.array([])))
        self.safe_append("r_e_ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps", getattr(self.ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_obj, "ue_throughput_Mbps", np.array([])))
        self.safe_append("r_e_cell_throughput_Mbps", getattr(self.ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj, "cell_throughput_Mbps", np.array([])))
        self.safe_append("r_e_carrier_throughput_Mbps", carrier_throughput_Mbps)
        self.safe_append("r_e_ue_throughput_per_carrier_Mbps", ue_throughput_per_carrier_Mbps)
        
        self.safe_append("r_e_power_consumption_per_radio_unit_incl_bbu_kW", getattr(self.power_consumption_obj, "power_consumption_per_radio_unit_incl_bbu_kW", np.array([])))
        self.safe_append("r_e_power_consumption_per_radio_unit_kW", getattr(self.power_consumption_obj, "power_consumption_per_radio_unit_kW", np.array([])))

        self.safe_append("r_e_network_power_consumption_per_radioType_kW", power_consumption_per_radioType_kW)
        self.safe_append("r_e_total_network_power_consumption_kW", getattr(self.power_consumption_obj, "total_RAN_power_consumption_kW", np.array([])))
        
        # ---- Saveables ----
        for saveable in _get_saveables():
            expose_variables = saveable.expose_variables()
            for var_name in expose_variables:
                full_var_name = f"{saveable.name}.{var_name}"
                value = saveable.get_output(var_name)
                self._safe_append_saveable(full_var_name, value)

        # Log execution time
        log_elapsed_time('Saving selected outputs', t_start)
        
        # End 
        return rescheduling_us 
