# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 11:42:19 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import time
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, List

from giulia.tools import tools
from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable

class Rate(Saveable):
    """
    A class to compute UE, cell, and carrier throughput based on different SINR computation models.
    Supports multiple frequency layers and provides methods for effective throughput estimation.
    """

    def __init__(
        self,
        compute_type: str,
        simulation_config_obj: Any,
        network_deployment_obj: Any,
        ue_deployment_obj: Any,
        beam_conf_obj: Any,
        best_serving_cell_per_ue_obj: Any,
        best_serving_beam_per_ue_obj: Any,
        base_stations_obj: Any,
        beam_sinr_ue_to_cell_obj: Any
    ):
        """
        Initializes the Rate class.

        Args:
            compute_type: Type of throughput computation (e.g., 'theoretical_long_term_equal_resource_share_UE_throughput_based_on_avgSINR').
            simulation_config_obj: Simulation configuration details.
            network_deployment_obj: Network deployment details.
            beam_conf_obj: Beam configuration details.
            best_serving_cell_per_ue_obj: Best serving cell information for each UE.
            best_serving_beam_per_ue_obj: Best serving beam information for each UE.
            base_stations_obj: Base station configuration details.
            beam_sinr_ue_to_cell_obj: SINR results per frequency layer.
        """

        super().__init__()

        # Inputs
        self.compute_type: str = compute_type
        self.simulation_config_obj: Any = simulation_config_obj
        self.network_deployment_obj: Any = network_deployment_obj
        self.beam_conf_obj: Any = beam_conf_obj
        self.best_serving_cell_per_ue_obj: Any = best_serving_cell_per_ue_obj
        self.best_serving_beam_per_ue_obj: Any = best_serving_beam_per_ue_obj
        self.base_stations_obj: Any = base_stations_obj
        self.beam_sinr_ue_to_cell_obj: Any = beam_sinr_ue_to_cell_obj
        self.ue_deployment_obj: Any = ue_deployment_obj

        # Outputs
        self.ue_throughput_Mbps: np.ndarray = None
        self.cell_throughput_Mbps: np.ndarray = None
        self.carrier_throughput_Mbps: np.ndarray = None
        self.ue_throughput_per_carrier_Mbps: np.ndarray = None
        
        self.ue_satisfiedFlag_demanded_throughput_Mbps = []
        self.ue_satisfied_demanded_throughput_ratio = []


    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["ue_throughput_per_carrier_Mbps"]
    

    def process(self, rescheduling_us: int = -1) -> int:
        """
        Processes throughput calculations for all frequency layers.
    
        Args:
            rescheduling_us: Rescheduling time in microseconds. Defaults to -1.
    
        Returns:
            Updated rescheduling time.
        """
        t_start: float = time.perf_counter()
    
        # Initialize UE throughput array
        num_ues: int = len(self.best_serving_cell_per_ue_obj.best_serving_cell_ID_per_ue)
        self.ue_throughput_Mbps: np.ndarray = np.zeros((num_ues,))
    
        # Initialize carrier throughput dictionaries
        self.carrier_throughput_Mbps: Dict[str, float] = {}
        self.ue_throughput_per_carrier_Mbps: Dict[str, np.ndarray] = {}
    
        # Compute UE throughput and carrier throughput in one pass
        for freq_key, sinr_data_dict in self.beam_sinr_ue_to_cell_obj.SINR_results_per_frequency_layer.items():
            ues_mask: np.ndarray = sinr_data_dict["ues_in_frequency_mask"]
            
            if self.compute_type == "theoretical_long_term_equal_resource_share_UE_throughput_based_on_avgSINR":
                self.ue_throughput_Mbps[ues_mask] = self.theoretical_long_term_equal_resource_share_UE_throughput_based_on_avgSINR(
                    self.network_deployment_obj.df_ep,
                    self.beam_conf_obj.number_of_beams_per_node,
                    self.best_serving_beam_per_ue_obj.best_serving_beam_ID_per_ue[ues_mask],
                    self.best_serving_beam_per_ue_obj.ues_per_beam,
                    sinr_data_dict["sinr_ue_to_cell_dB"],
                )
    
            elif self.compute_type == "theoretical_long_term_equal_resource_share_UE_throughput_based_on_prbSINR":
                self.ue_throughput_Mbps[ues_mask] = self.theoretical_long_term_equal_resource_share_UE_throughput_based_on_prbSINR(
                    self.simulation_config_obj,
                    self.network_deployment_obj.df_ep,
                    self.beam_conf_obj.number_of_beams_per_node,
                    self.best_serving_beam_per_ue_obj.best_serving_beam_ID_per_ue[ues_mask],
                    sinr_data_dict["sinr_ue_to_cell_dB"],
                )
    
            elif self.compute_type == "theoretical_long_term_equal_resource_share_UE_throughput_based_on_effSINR":                
                self.ue_throughput_Mbps[ues_mask] = self.theoretical_long_term_equal_resource_share_UE_throughput_based_on_effSINR(
                    self.simulation_config_obj,
                    self.network_deployment_obj.df_ep,
                    self.beam_conf_obj.number_of_beams_per_node,
                    self.best_serving_beam_per_ue_obj.best_serving_beam_ID_per_ue[ues_mask],
                    self.base_stations_obj.resource_allocation["PRB_ue_activity"][:,ues_mask],
                    sinr_data_dict["effective_sinr_ue_to_cell_dB"],
                )
                
            # Assume 2 cross-polarized antennas at the UE
            self.ue_throughput_Mbps[ues_mask] *= 2 
                
            # If target rate exist and the traffic model allows, limit the rate to the target rate            
            if any(self.ue_deployment_obj.df_ep["traffic_generation_model"] == 'rate_requirement'):
                mask_rate_target = self.ue_deployment_obj.df_ep["traffic_generation_model"].values == 'rate_requirement'
                
                self.ue_throughput_Mbps[mask_rate_target] = np.minimum(self.ue_throughput_Mbps[mask_rate_target], 
                                                                       self.ue_deployment_obj.df_ep["ue_target_rate_Mbps"].values[mask_rate_target])
                del mask_rate_target
            else: 
                pass
            
            # Store per-carrier UE throughput, setting values outside the mask to NaN
            self.ue_throughput_per_carrier_Mbps[freq_key] = np.full_like(self.ue_throughput_Mbps, np.nan)
            self.ue_throughput_per_carrier_Mbps[freq_key][ues_mask] = self.ue_throughput_Mbps[ues_mask]                   

        # Compute cell throughput by aggregating UE throughputs
        self.cell_throughput_Mbps: np.ndarray = np.bincount(
            self.best_serving_cell_per_ue_obj.best_serving_cell_ID_per_ue,
            weights=self.ue_throughput_Mbps,
            minlength=len(self.network_deployment_obj.df_ep)
        )
        
        # Compute carrier throughput by aggregating cell throughputs
        dl_carrier_frequency_GHz = self.network_deployment_obj.df_ep["dl_carrier_frequency_GHz"]        
        for freq_key in np.unique(dl_carrier_frequency_GHz):
            self.carrier_throughput_Mbps[freq_key] = np.sum(self.cell_throughput_Mbps[np.where(dl_carrier_frequency_GHz == freq_key)])  
            
            
        # Check if UEs met the requirements, if any.
        # If there are no requirement, are set to 1 (True)
        # Then, append in  self.ue_satisfiedFlag_demanded_throughput_Mbps and self.ue_satisfied_demanded_throughput_ratio attributes
        ue_flag_satisfied = self.check_ue_satisfied_demand_throughput(self.ue_throughput_Mbps, 
                                                                          self.ue_deployment_obj.df_ep['ue_target_rate_Mbps'].values )
        self.ue_satisfiedFlag_demanded_throughput_Mbps.append(ue_flag_satisfied)
        self.ue_satisfied_demanded_throughput_ratio.append( np.sum(ue_flag_satisfied) / len(ue_flag_satisfied) ) 
        
        # Log execution time
        log_calculations_time('Rate', t_start)
    
        return rescheduling_us
    
    
    def theoretical_long_term_equal_resource_share_UE_throughput_based_on_avgSINR(
        self,
        df_ep: pd.DataFrame,
        number_of_beams_per_cell: np.ndarray,
        best_serving_beam_per_ue: np.ndarray,
        ues_per_beam: np.ndarray,
        sinr_ue_to_cell_dB: np.ndarray,
    ) -> np.ndarray:
        """
        Computes theoretical long-term UE throughput assuming equal resource sharing 
        based on average SINR.
    
        Args:
            df_ep: DataFrame containing network deployment parameters.
            number_of_beams_per_cell: Array containing the number of beams per cell.
            best_serving_beam_per_ue: Array mapping UEs to their best serving beam.
            ues_per_beam: Array containing the number of UEs per beam.
            sinr_ue_to_cell_dB: SINR values (dB) for UEs.
    
        Returns:
            ue_throughput_Mbps: Computed throughput per UE (Mbps).
        """
        # Extract network parameters
        dl_PRBs_available = df_ep["dl_PRBs_available"].to_numpy(dtype=int)
        subcarrier_per_PRB = df_ep["subcarriers_per_PRB"].to_numpy(dtype=int)
        dl_subcarrier_spacing_kHz = df_ep["dl_subcarrier_spacing_kHz"].to_numpy(dtype=int)
        
        # Compute available bandwidth per PRB in MHz
        if isinstance(number_of_beams_per_cell, np.ndarray): 
            bandwidth_per_ue_MHz: np.ndarray = \
                np.repeat(dl_PRBs_available * subcarrier_per_PRB * dl_subcarrier_spacing_kHz, number_of_beams_per_cell) / 1e3 # Shape: (num_beams,)
        else:
            bandwidth_per_ue_MHz: np.ndarray = \
                dl_PRBs_available * subcarrier_per_PRB * dl_subcarrier_spacing_kHz / 1e3 # Shape: (num_cells,)
    
        # Select bandwidth for the best serving beam per UE
        bandwidth_per_ue_MHz = bandwidth_per_ue_MHz[best_serving_beam_per_ue]  # Shape: (num_UEs,)
    
        # Compute spectral efficiency and UE throughput
        ue_spectral_efficiency_bps_per_Hz = np.log2(1 + tools.dBm_to_mW(sinr_ue_to_cell_dB))
        ue_throughput_Mbps = bandwidth_per_ue_MHz * ue_spectral_efficiency_bps_per_Hz
    
        return ue_throughput_Mbps
    
    
    def theoretical_long_term_equal_resource_share_UE_throughput_based_on_prbSINR(
        self,
        simulation_config_obj: Any,
        df_ep: pd.DataFrame,
        number_of_beams_per_cell: np.ndarray,
        best_serving_beam_per_ue: np.ndarray,
        sinr_prb_ue_to_cell_dB: np.ndarray,
    ) -> np.ndarray:
        """
        Computes theoretical long-term UE throughput assuming equal resource sharing 
        based on per-PRB SINR.
    
        Args:
            simulation_config_obj: Simulation configuration object (unused in this method).
            df_ep: DataFrame containing network deployment parameters.
            number_of_beams_per_cell: Array indicating the number of beams per cell.
            best_serving_beam_per_ue: Array mapping UEs to their best serving beam.
            sinr_prb_ue_to_cell_dB: SINR values (dB) for UEs on a per-PRB basis.
    
        Returns:
            ue_throughput_Mbps: Computed throughput per UE (Mbps).
        """
        # Extract relevant network parameters
        subcarrier_per_PRB: np.ndarray = df_ep["subcarriers_per_PRB"].to_numpy(dtype=int)        
        dl_subcarrier_spacing_kHz: np.ndarray = df_ep["dl_subcarrier_spacing_kHz"].to_numpy(dtype=int)
    
        # Compute available bandwidth per PRB in MHz
        if isinstance(number_of_beams_per_cell, np.ndarray): 
            bandwidth_per_prb_MHz: np.ndarray = \
                np.repeat(subcarrier_per_PRB * dl_subcarrier_spacing_kHz, number_of_beams_per_cell) / 1e3  # Shape: (num_beams,)
        else:
            bandwidth_per_prb_MHz: np.ndarray = \
                subcarrier_per_PRB * dl_subcarrier_spacing_kHz / 1e3  # Shape: (num_cells,)
    
        # Select bandwidth for the best serving beam per UE
        bandwidth_per_prb_MHz = bandwidth_per_prb_MHz[best_serving_beam_per_ue]  # Shape: (num_UEs,)
    
        # Compute spectral efficiency using SINR values
        ue_spectral_efficiency_bps_per_Hz: np.ndarray = np.log2(1 + tools.dBm_to_mW(sinr_prb_ue_to_cell_dB))
    
        # Compute total UE throughput by summing across PRBs
        ue_throughput_Mbps: np.ndarray = np.nansum(bandwidth_per_prb_MHz * ue_spectral_efficiency_bps_per_Hz, axis=0)
    
        return ue_throughput_Mbps

    
    def theoretical_long_term_equal_resource_share_UE_throughput_based_on_effSINR(
        self,
        simulation_config_obj: Any,
        df_ep: pd.DataFrame,
        number_of_beams_per_cell: np.ndarray,
        best_serving_beam_per_ue: np.ndarray,
        PRB_ue_activity: np.ndarray,
        sinr_prb_ue_to_cell_dB: np.ndarray,
    ) -> np.ndarray:
        """
        Computes theoretical long-term UE throughput assuming equal resource sharing 
        based on effective SINR.
    
        Args:
            simulation_config_obj: Simulation configuration object (unused in this method).
            df_ep: DataFrame containing network deployment parameters.
            number_of_beams_per_cell: Array containing the number of beams per cell.
            best_serving_beam_per_ue: Array mapping UEs to their best serving beam.
            PRB_ue_activity: PRB activity mask indicating allocated PRBs per UE.
            sinr_prb_ue_to_cell_dB: Effective SINR values (dB) for UEs.
    
        Returns:
            ue_throughput_Mbps: Computed throughput per UE (Mbps).
        """
        # Extract network parameters
        subcarrier_per_PRB: np.ndarray = df_ep["subcarriers_per_PRB"].to_numpy(dtype=int)
        dl_subcarrier_spacing_kHz: np.ndarray = df_ep["dl_subcarrier_spacing_kHz"].to_numpy(dtype=int)
    
        # Compute available bandwidth per PRB in MHz
        if isinstance(number_of_beams_per_cell, np.ndarray):
            bandwidth_per_prb_MHz: np.ndarray = \
                np.repeat(subcarrier_per_PRB * dl_subcarrier_spacing_kHz, number_of_beams_per_cell) / 1e3  # Shape: (num_beams,)
        else:
            bandwidth_per_prb_MHz: np.ndarray = \
                subcarrier_per_PRB * dl_subcarrier_spacing_kHz / 1e3  # Shape: (num_cells,)
    
        # Select bandwidth for the best serving beam per UE
        bandwidth_per_prb_MHz = bandwidth_per_prb_MHz[best_serving_beam_per_ue]  # Shape: (num_UEs,)
    
        # Compute spectral efficiency using SINR values
        ue_spectral_efficiency_bps_per_Hz: np.ndarray = np.log2(1 + tools.dBm_to_mW(sinr_prb_ue_to_cell_dB))
    
        # Compute total UE throughput considering PRB activity
        ue_throughput_Mbps: np.ndarray = np.nansum(PRB_ue_activity * bandwidth_per_prb_MHz * ue_spectral_efficiency_bps_per_Hz, axis=0)
    
        return ue_throughput_Mbps


    def cell_rate(
        self,
        simulation_config_obj: Any,
        df_ep: pd.DataFrame,
        best_serving_cell_per_ue: np.ndarray,
        ue_rate_Mbps: np.ndarray,
    ) -> np.ndarray:
        """
        Computes the total throughput per cell by aggregating UE throughput.
    
        Args:
            simulation_config_obj: Simulation configuration object (unused in this method).
            df_ep: DataFrame containing network deployment parameters.
            best_serving_cell_per_ue: Array indicating the best serving cell for each UE.
            ue_rate_Mbps: Array of UE throughputs in Mbps.
    
        Returns:
            cell_throughput_Mbps: Total throughput per cell in Mbps.
                                  Shape: (num_cells,)
        """
        # Initialize cell throughput array
        num_cells: int = len(df_ep)
        cell_throughput_Mbps: np.ndarray = np.zeros(num_cells, dtype=np.float32)
    
        # Vectorized aggregation of UE throughput per serving cell
        np.add.at(cell_throughput_Mbps, best_serving_cell_per_ue, ue_rate_Mbps)
    
        return cell_throughput_Mbps    
    
    
    def check_ue_satisfied_demand_throughput(self, ue_throughput_Mbps, ue_demand_throughput_Mbps):
        # Compute flag 
        ue_flag_satisfied = ue_throughput_Mbps >= ue_demand_throughput_Mbps
        ue_flag_satisfied = ue_flag_satisfied.astype(int)
        # Do checks
        mask_rate_target = self.ue_deployment_obj.df_ep["traffic_generation_model"].values == 'rate_requirement'
        
        # If there are not requirement, it is set by default to True 
        ue_flag_satisfied[~mask_rate_target] = True
                
        return ue_flag_satisfied
    

class Rate_based_on_avg_CSI_RS_SINR(Rate):
    pass

class Rate_based_on_ins_CSI_RS_SINR(Rate):
    pass

class Rate_based_on_eff_CSI_RS_SINR(Rate):
    pass
