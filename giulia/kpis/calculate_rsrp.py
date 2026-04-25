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
import copy

import numpy as np
import torch
from typing import Any, Dict, List

from giulia.tools import tools
from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable

class RSRP(Saveable):
    """
    A class for calculating Reference Signal Received Power (RSRP) for all UEs and cells.
    Supports various computation types and multiple frequency layers.
    """

    def __init__(
        self,
        compute_type: str,
        network_deployment_obj: Any,
        beam_conf_obj: Any,
        channel_gain_ue_to_cell_obj: Any,
    ):
        """
        Initializes the RSRP class.

        Args:
            compute_type: Type of RSRP computation.
            network_deployment_obj: Network deployment details.
            beam_conf_obj: Beam configuration details.
            channel_gain_ue_to_cell_obj: Channel gain results for UE-to-cell links.
        """

        super().__init__()

        # Device setup
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Inputs
        self.compute_type: str = compute_type
        self.network_deployment_obj: Any = network_deployment_obj
        self.beam_conf_obj: Any = beam_conf_obj
        self.channel_gain_ue_to_cell_obj: Any = channel_gain_ue_to_cell_obj

        # Outputs
        self.RSRP_results_per_frequency_layer: Dict[str, np.ndarray] = {}


    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["RSRP_results_per_frequency_layer"]
    

    def process(self, rescheduling_us: int = -1) -> int:
        """
        Processes the RSRP calculations for all UEs and cells, iterating over frequency layers.

        Args:
            rescheduling_us: Rescheduling time in microseconds. Defaults to -1.

        Returns:
            Updated rescheduling time.
        """
        
        # Determine beam type
        self.beam_type: str = "CRS" if self.beam_conf_obj is None else self.beam_conf_obj.beam_type

        # Extract network parameters
        # Number of beams
        if self.beam_type != "CRS":
            self.number_of_beams: np.ndarray = self.network_deployment_obj.df_ep[
                f"{self.beam_type}_number_of_beams"
            ].to_numpy(dtype=int)
        
        # TX power
        self.BS_tx_power: np.ndarray = self.network_deployment_obj.df_ep[
            f"BS_tx_power_{self.beam_type}_RE_dBm" #f"BS_tx_power_{self.beam_type}_RE_dBm"  #"BS_tx_power_SSB_RE_dBm"
        ].to_numpy(dtype=np.single)

        # Start performance timer
        t_start: float = time.perf_counter()

        # Perform calculations based on compute type
        if self.compute_type == "CRS_RSRP_no_fast_fading":
            # Process each frequency layer
            for frequency_key, frequency_layer_data in self.channel_gain_ue_to_cell_obj.slow_channel_results_per_frequency_layer.items():
                RSRP_ue_to_cell_dBm = \
                    self.antenna_pattern_RSRP_no_fast_fading_all_ues_and_cells(self.BS_tx_power, frequency_layer_data["slow_channel_gain_b_to_a_dB"])
                                
                # Store results for the frequency layer
                self.RSRP_results_per_frequency_layer[frequency_key] = copy.deepcopy(frequency_layer_data)
                del self.RSRP_results_per_frequency_layer[frequency_key]["slow_channel_gain_b_to_a_dB"]
                self.RSRP_results_per_frequency_layer[frequency_key].update({
                    "RSRP_ue_to_cell_dBm": RSRP_ue_to_cell_dBm })     

        elif self.compute_type == "beam_RSRP_no_fast_fading":
            # Process each frequency layer
            for frequency_key, frequency_layer_data in self.channel_gain_ue_to_cell_obj.precoded_channel_gain_results_per_frequency_layer.items():
                precoded_gain = frequency_layer_data["precoded_channel_gain_b_to_a_dB"]
                RSRP_ue_to_cell_dBm = self.beam_RSRP_no_fast_fading_all_ues_and_cells(self.number_of_beams, self.BS_tx_power, precoded_gain )
                                
                # Store results for the frequency layer
                self.RSRP_results_per_frequency_layer[frequency_key] = copy.deepcopy(frequency_layer_data)
                del self.RSRP_results_per_frequency_layer[frequency_key]["precoded_channel_gain_b_to_a_dB"]
                self.RSRP_results_per_frequency_layer[frequency_key].update({
                    "RSRP_ue_to_cell_dBm": RSRP_ue_to_cell_dBm })                  

        elif self.compute_type == "beam_RSRP_based_on_RSS":
            # Process each frequency layer
            for frequency_key, frequency_layer_data in self.channel_gain_ue_to_cell_obj.RSS_results_per_frequency_layer.items():
                RSS_per_PRB_ue_to_cell_dBm: np.ndarray = frequency_layer_data["RSS_per_PRB_ue_to_cell_dBm"]
                RSRP_ue_to_cell_dBm = self.calculate_RSRP_based_on_RSS_per_PRB(RSS_per_PRB_ue_to_cell_dBm, self.device)
                
                # Store results for the frequency layer
                self.RSRP_results_per_frequency_layer[frequency_key] = copy.deepcopy(frequency_layer_data)
                del self.RSRP_results_per_frequency_layer[frequency_key]["RSS_per_PRB_ue_to_cell_dBm"]
                self.RSRP_results_per_frequency_layer[frequency_key].update({
                    "RSRP_ue_to_cell_dBm": RSRP_ue_to_cell_dBm })                
            
        # Log elapsed time
        log_calculations_time('RSRP', t_start)

        return rescheduling_us


    def set_RSRP_ue_to_cell_dBm(
        self,
        RSRP_ue_to_cell_obj: object,  
        rescheduling_us: int = -1    
    ) -> int:  
        """
        Sets the RSRP results per frequency layer for the object and returns the rescheduling time.
    
        Args:
            RSRP_ue_to_cell_obj: Object containing RSRP results per frequency layer.
            rescheduling_us: Rescheduling time in microseconds. Defaults to -1.
    
        Returns:
            Updated rescheduling time.
        """
        self.RSRP_results_per_frequency_layer = RSRP_ue_to_cell_obj.RSRP_results_per_frequency_layer
    
        return rescheduling_us


    def antenna_pattern_RSRP_no_fast_fading_all_ues_and_cells(
        self, BS_tx_power: np.ndarray, slow_channel_gain: np.ndarray
    ) -> np.ndarray:
        """
        Calculates RSRP using antenna pattern without fast fading.

        Args:
            BS_tx_power: Transmission power per SSB RE (dBm).
            slow_channel_gain: Slow channel gain (dB).

        Returns:
            RSRP results (dBm).
        """
        extended_tx_power = BS_tx_power[np.newaxis, :] * np.ones((slow_channel_gain.shape[0], 1))
        return extended_tx_power + slow_channel_gain


    def beam_RSRP_no_fast_fading_all_ues_and_cells(
        self, number_of_beams: np.ndarray, BS_tx_power: np.ndarray, precoded_gain: np.ndarray
    ) -> np.ndarray:
        """
        Calculates beam RSRP without fast fading.

        Args:
            BS_tx_power: Transmission power per SSB RE (dBm).
            number_of_beams: Number of beams per cell.
            precoded_gain: Precoded channel gain (dB).

        Returns:
            Beam RSRP results (dBm).
        """
        extended_tx_power = np.repeat(BS_tx_power[np.newaxis, :] * np.ones((precoded_gain.shape[1], 1)), number_of_beams, axis=1)
        return extended_tx_power + precoded_gain[0, :, :]  # All positions are the same
    
    
    def calculate_RSRP_based_on_RSS_per_PRB(
            self, rss_data: np.ndarray, device: torch.device
    ) -> np.ndarray:
        """
        Calculates RSRP based on RSS per PRB.

        Args:
            rss_data: RSS data per PRB (dBm).
            device: Torch device for computation.

        Returns:
            RSRP results (dBm).
        """
        return tools.mW_to_dBm_torch(torch.mean(tools.dBm_to_mW_torch(torch.tensor(rss_data, device=device)), axis=0)).cpu().numpy()
        


class RSRP_CRS_no_fast_fading(RSRP):
    pass

class RSRP_SSB_no_fast_fading(RSRP):
    pass

class RSRP_SSB(RSRP):
    pass

class RSRP_CSI_RS_no_fast_fading(RSRP):
    pass

class RSRP_CSI_RS(RSRP):
    pass
