#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:39:12 2024

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""
import time


import numpy as np
import numpy.typing as npt

from typing import Callable, List

from giulia.logger import warning
from giulia.fs import results_file
from giulia.presets import Giulia
from giulia.tools.tools import log_elapsed_time

class Episode_Performance:

    def __init__(self, number_of_episodes: int, regression: bool):
        """
        Initializes the class and allocates empty numpy arrays for storing episode performance metrics.
    
        Args:
            number_of_episodes (int): Number of episodes to store performance data for.
            regression (bool): Flag indicating if regression analysis is performed.
        """
        # Store the number of episodes and regression flag
        self.number_of_episodes: int = number_of_episodes
        self.regression: bool = regression
    
        # Function to initialize an empty numpy array for each episode
        def init_empty_list() -> List[np.ndarray]:
            return [np.empty(0) for _ in range(self.number_of_episodes)]
    
        # ---- CRS (Channel State Reporting Reference Signal) ----
        self.r_best_serving_cell_ID_per_ue_based_on_CRS: List[np.ndarray] = init_empty_list()
        self.r_best_serving_CRS_distance_3d_m: List[np.ndarray] = init_empty_list()
        self.r_best_serving_CRS_coupling_gain_dB: List[np.ndarray] = init_empty_list()
        self.r_best_serving_CRS_re_noise_per_ue_dBm: List[np.ndarray] = init_empty_list()
        self.r_best_serving_CRS_rsrp_per_ue_dBm: List[np.ndarray] = init_empty_list()
        self.r_CRS_RSRP_no_fast_fading_ue_to_cell_dBm: List[np.ndarray] = init_empty_list()

        self.r_CRS_sinr_ue_to_cell_dB: List[np.ndarray] = init_empty_list()
        self.r_CRS_usefulPower_ue_dBm: List[np.ndarray] = init_empty_list()
        self.r_CRS_interfPower_ue_dBm: List[np.ndarray] = init_empty_list()
    
        # ---- SSB (Synchronization Signal Block) ----
        self.r_best_serving_cell_ID_per_ue_based_on_SSB: List[np.ndarray] = init_empty_list()
        self.r_best_serving_SSB_per_ue: List[np.ndarray] = init_empty_list()
        self.r_best_serving_SSB_distance_3d_m: List[np.ndarray] = init_empty_list()
        self.r_best_serving_SSB_coupling_gain_dB: List[np.ndarray] = init_empty_list()
        self.r_best_serving_SSB_re_noise_per_ue_dBm: List[np.ndarray] = init_empty_list()
        self.r_best_serving_SSB_rsrp_per_ue_dBm: List[np.ndarray] = init_empty_list()
        self.r_SSB_RSRP_no_fast_fading_ue_to_cell_dBm: List[np.ndarray] = init_empty_list()

        self.r_SSB_sinr_ue_to_cell_dB: List[np.ndarray] = init_empty_list()
        self.r_SSB_usefulPower_ue_dBm: List[np.ndarray] = init_empty_list()
        self.r_SSB_interfPower_ue_dBm: List[np.ndarray] = init_empty_list()
    
        # ---- CSI-RS (Channel State Information Reference Signal) ----
        self.r_best_serving_CSI_RS_per_ue: List[np.ndarray] = init_empty_list()
        self.r_best_serving_CSI_RS_rsrp_per_ue_dBm: List[np.ndarray] = init_empty_list()
        self.r_effective_CSI_RS_sinr_ue_to_cell_dB: List[np.ndarray] = init_empty_list()
        
        # ---- UE counts per cell and beam ----
        self.r_ues_per_cell_based_on_CRS: List[np.ndarray] = init_empty_list()
        self.r_ues_per_cell_based_on_SSB: List[np.ndarray] = init_empty_list()
        self.r_ues_per_SSB_beam: List[np.ndarray] = init_empty_list()
        self.r_ues_per_CSI_RS_beam: List[np.ndarray] = init_empty_list()
        
        # ---- Cell and beam activity ----
        self.r_cell_activity_per_ue: List[np.ndarray] = init_empty_list()
        self.r_SSB_beam_activity_per_ue: List[np.ndarray] = init_empty_list()
        self.r_CSI_RS_beam_activity_per_ue: List[np.ndarray] = init_empty_list()  
    
        # ---- Throughput and Power Consumption ----
        self.r_ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps: List[np.ndarray] = init_empty_list()
        self.r_ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps: List[np.ndarray] = init_empty_list()
        self.r_cell_throughput_Mbps: List[np.ndarray] = init_empty_list()
        self.r_carrier_throughput_Mbps: List[np.ndarray] = init_empty_list()
        self.r_ue_throughput_per_carrier_Mbps: List[np.ndarray] = init_empty_list()

        self.r_power_consumption_perRadio_kW: List[np.ndarray] = init_empty_list()
        self.r_network_power_consumption_per_radioType_kW: List[np.ndarray] = init_empty_list()
        self.r_total_network_power_consumption_kW: List[np.ndarray] = init_empty_list()


    def save_episode_performance(self, episode_index: int, g: Giulia):
        """
        Saves the performance metrics for a specific episode.

        Args:
            episode_index (int): The index of the episode being saved.
            g (Giulia): The Giulia simulation object containing performance data.
        """
        
        # ---- CRS (Channel State Reporting Reference Signal) ----
        self.r_best_serving_cell_ID_per_ue_based_on_CRS[episode_index] = g.performance_obj.r_e_best_serving_cell_ID_per_ue_based_on_CRS
        self.r_best_serving_CRS_distance_3d_m[episode_index] = g.performance_obj.r_e_best_serving_CRS_distance_3d_m  
        self.r_best_serving_CRS_re_noise_per_ue_dBm[episode_index] = g.performance_obj.r_e_best_serving_CRS_re_noise_per_ue_dBm
        self.r_best_serving_CRS_coupling_gain_dB[episode_index] = g.performance_obj.r_e_best_serving_CRS_coupling_gain_dB
        self.r_best_serving_CRS_rsrp_per_ue_dBm[episode_index] = g.performance_obj.r_e_best_serving_CRS_rsrp_per_ue_dBm
        self.r_CRS_RSRP_no_fast_fading_ue_to_cell_dBm[episode_index] = g.performance_obj.r_e_CRS_RSRP_no_fast_fading_ue_to_cell_dBm
        
        self.r_CRS_sinr_ue_to_cell_dB[episode_index] = g.performance_obj.r_e_CRS_sinr_ue_to_cell_dB
        self.r_CRS_usefulPower_ue_dBm[episode_index] = g.performance_obj.r_e_CRS_usefulPower_ue_dBm
        self.r_CRS_interfPower_ue_dBm[episode_index] = g.performance_obj.r_e_CRS_interfPower_ue_dBm        
    
        # ---- SSB (Synchronization Signal Block) ----
        self.r_best_serving_cell_ID_per_ue_based_on_SSB[episode_index] = g.performance_obj.r_e_best_serving_cell_ID_per_ue_based_on_SSB
        self.r_best_serving_SSB_per_ue[episode_index] = g.performance_obj.r_e_best_serving_SSB_per_ue  
        self.r_best_serving_SSB_distance_3d_m[episode_index] = g.performance_obj.r_e_best_serving_SSB_distance_3d_m  
        self.r_best_serving_SSB_coupling_gain_dB[episode_index] = g.performance_obj.r_e_best_serving_SSB_coupling_gain_dB
        self.r_best_serving_SSB_re_noise_per_ue_dBm[episode_index] = g.performance_obj.r_e_best_serving_SSB_re_noise_per_ue_dBm
        self.r_best_serving_SSB_rsrp_per_ue_dBm[episode_index] = g.performance_obj.r_e_best_serving_SSB_rsrp_per_ue_dBm
        self.r_SSB_RSRP_no_fast_fading_ue_to_cell_dBm[episode_index] = g.performance_obj.r_e_SSB_RSRP_no_fast_fading_ue_to_cell_dBm

        self.r_SSB_sinr_ue_to_cell_dB[episode_index] = g.performance_obj.r_e_SSB_sinr_ue_to_cell_dB
        self.r_SSB_usefulPower_ue_dBm[episode_index] = g.performance_obj.r_e_SSB_usefulPower_ue_dBm
        self.r_SSB_interfPower_ue_dBm[episode_index] = g.performance_obj.r_e_SSB_interfPower_ue_dBm        
    
        # ---- CSI-RS (Channel State Information Reference Signal) ----
        self.r_best_serving_CSI_RS_per_ue[episode_index] = g.performance_obj.r_e_best_serving_CSI_RS_per_ue
        self.r_best_serving_CSI_RS_rsrp_per_ue_dBm[episode_index] = g.performance_obj.r_e_best_serving_CSI_RS_rsrp_per_ue_dBm 

        self.r_effective_CSI_RS_sinr_ue_to_cell_dB[episode_index] = g.performance_obj.r_e_effective_CSI_RS_sinr_ue_to_cell_dB
        
        # ---- UE counts per cell and beam ----
        self.r_ues_per_cell_based_on_CRS[episode_index] = g.performance_obj.r_e_ues_per_cell_based_on_CRS
        self.r_ues_per_cell_based_on_SSB[episode_index] = g.performance_obj.r_e_ues_per_cell_based_on_SSB
        self.r_ues_per_SSB_beam[episode_index] = g.performance_obj.r_e_ues_per_SSB_beam
        self.r_ues_per_CSI_RS_beam[episode_index] = g.performance_obj.r_e_ues_per_CSI_RS_beam
        
        # ---- Cell and beam activity ----
        self.r_cell_activity_per_ue[episode_index] = g.performance_obj.r_e_cell_activity_per_ue
        self.r_SSB_beam_activity_per_ue[episode_index] = g.performance_obj.r_e_SSB_beam_activity_per_ue
        self.r_CSI_RS_beam_activity_per_ue[episode_index] = g.performance_obj.r_e_CSI_RS_beam_activity_per_ue        
    
        # ---- Throughput and Power Consumption ----
        self.r_ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps[episode_index] = g.performance_obj.r_e_ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps
        self.r_ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps[episode_index] = g.performance_obj.r_e_ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps
        self.r_cell_throughput_Mbps[episode_index] = g.performance_obj.r_e_cell_throughput_Mbps  
        self.r_carrier_throughput_Mbps[episode_index] = g.performance_obj.r_e_carrier_throughput_Mbps
        self.r_ue_throughput_per_carrier_Mbps[episode_index] = g.performance_obj.r_e_ue_throughput_per_carrier_Mbps      

        self.r_power_consumption_perRadio_kW[episode_index] = g.performance_obj.r_e_power_consumption_perRadio_kW
        self.r_network_power_consumption_per_radioType_kW[episode_index] = g.performance_obj.r_e_network_power_consumption_per_radioType_kW
        self.r_total_network_power_consumption_kW[episode_index] = g.performance_obj.r_e_total_network_power_consumption_kW
        

    def save_episodes_performance_in_file(self, project_name: str):
        """
        Saves the performance of all episodes into separate files (mean, median, raw values).

        Args:
            project_name (str): The name of the project to store results.
        """
        # Start timer
        t_start = time.perf_counter()  

        # Save mean values
        self._save_episodes_performance(results_file(project_name, 'results-mean'), np.nanmean)
        # Save median values
        self._save_episodes_performance(results_file(project_name, 'results-median'), np.nanmedian)
        # Save raw data
        self._save_episodes_performance(results_file(project_name, 'results-raw'), lambda value: value)

        # Log execution time
        log_elapsed_time('Saving selected outputs', t_start)  


    def _save_episodes_performance(self, file: str, operation: Callable[[npt.NDArray[np.float64]], npt.ArrayLike]):
        """
        Stores the episodes performance on the given file, after applying ``operation`` to each field.
    
        Args:
            file: The full path of the file to store the data into.
            operation: The operation to apply to each data type, normally a numpy operation.
        """
        np.savez(
            file,
    
            # ---- CRS (Channel State Reporting Reference Signal) ----
            best_serving_cell_ID_per_ue_based_on_CRS=operation(self.r_best_serving_cell_ID_per_ue_based_on_CRS),
            best_serving_CRS_distance_3d_m=operation(self.r_best_serving_CRS_distance_3d_m),
            best_serving_CRS_coupling_gain_dB=operation(self.r_best_serving_CRS_coupling_gain_dB),
            best_serving_CRS_re_noise_per_ue_dBm=operation(self.r_best_serving_CRS_re_noise_per_ue_dBm),
            best_serving_CRS_rsrp_per_ue_dBm=operation(self.r_best_serving_CRS_rsrp_per_ue_dBm),
            CRS_RSRP_no_fast_fading_ue_to_cell_dBm=operation(self.r_CRS_RSRP_no_fast_fading_ue_to_cell_dBm),

            CRS_sinr_ue_to_cell_dB=operation(self.r_CRS_sinr_ue_to_cell_dB),
            CRS_usefulPower_ue_dBm = operation(self.r_CRS_usefulPower_ue_dBm), 
            CRS_interfPower_ue_dBm = operation(self.r_CRS_interfPower_ue_dBm), 
    
            # ---- SSB (Synchronization Signal Block) ----
            best_serving_cell_ID_per_ue_based_on_SSB=operation(self.r_best_serving_cell_ID_per_ue_based_on_SSB),
            best_serving_SSB_per_ue=operation(self.r_best_serving_SSB_per_ue),
            best_serving_SSB_distance_3d_m=operation(self.r_best_serving_SSB_distance_3d_m),
            best_serving_SSB_coupling_gain_dB=operation(self.r_best_serving_SSB_coupling_gain_dB),
            best_serving_SSB_re_noise_per_ue_dBm=operation(self.r_best_serving_SSB_re_noise_per_ue_dBm),
            best_serving_SSB_rsrp_per_ue_dBm=operation(self.r_best_serving_SSB_rsrp_per_ue_dBm),
            SSB_RSRP_no_fast_fading_ue_to_cell_dBm=operation(self.r_SSB_RSRP_no_fast_fading_ue_to_cell_dBm),

            SSB_sinr_ue_to_cell_dB=operation(self.r_SSB_sinr_ue_to_cell_dB),
            SSB_usefulPower_ue_dBm = operation(self.r_SSB_usefulPower_ue_dBm), 
            SSB_interfPower_ue_dBm = operation(self.r_SSB_interfPower_ue_dBm), 
    
            # ---- CSI-RS (Channel State Information Reference Signal) ----
            best_serving_CSI_RS_per_ue=operation(self.r_best_serving_CSI_RS_per_ue),
            best_serving_CSI_RS_rsrp_per_ue_dBm=operation(self.r_best_serving_CSI_RS_rsrp_per_ue_dBm),

            effective_CSI_RS_sinr_ue_to_cell_dB=operation(self.r_effective_CSI_RS_sinr_ue_to_cell_dB),
            
            # ---- UE counts per cell and beam ----
            ues_per_cell_based_on_CRS=operation(self.r_ues_per_cell_based_on_CRS),
            ues_per_cell=operation(self.r_ues_per_cell_based_on_SSB),
            ues_per_SSB_beam=operation(self.r_ues_per_SSB_beam),
            ues_per_CSI_RS_beam=operation(self.r_ues_per_CSI_RS_beam),
            
            # ---- Cell and beam activity ----
            cell_activity_per_ue=operation(self.r_cell_activity_per_ue),
            SSB_beam_activity_per_ue=operation(self.r_SSB_beam_activity_per_ue),
            CSI_RS_beam_activity_per_ue=operation(self.r_CSI_RS_beam_activity_per_ue),              
        
            # ---- Throughput and Power Consumption ----
            ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps=operation(self.r_ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps),
            ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps=operation(self.r_ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps),
            cell_throughput_Mbps=operation(self.r_cell_throughput_Mbps),
            carrier_throughput_Mbps=operation(self.r_carrier_throughput_Mbps),
            ue_throughput_per_carrier_Mbps=operation(self.r_ue_throughput_per_carrier_Mbps),

            power_consumption_perRadio_kW=operation(self.r_power_consumption_perRadio_kW),
            network_power_consumption_per_radioType_kW=operation(self.r_network_power_consumption_per_radioType_kW),
            total_network_power_consumption_kW=operation(self.r_total_network_power_consumption_kW),
            
            )