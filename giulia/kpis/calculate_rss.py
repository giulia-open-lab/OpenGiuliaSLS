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
from typing import Any, Dict, List

from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable


class RSS(Saveable):
    """
    A class for calculating the Received Signal Strength (RSS) per PRB for all UEs and cells.
    Handles multiple beams and frequency layers based on configuration and deployment objects.
    """

    def __init__(
        self,
        compute_type: str,
        network_deployment_obj: Any,
        beam_conf_obj: Any,
        beam_precoded_channel_gain_ue_to_cell_obj: Any,
    ):
        """
        Initializes the RSS class.

        Args:
            compute_type: Type of RSS computation.
            network_deployment_obj: Network deployment details.
            beam_conf_obj: Beam configuration details.
            beam_precoded_channel_gain_ue_to_cell_obj: Precoded channel gain results.
        """

        super().__init__()

        # Inputs
        self.compute_type: str = compute_type
        self.network_deployment_obj: Any = network_deployment_obj
        self.beam_conf_obj: Any = beam_conf_obj
        self.beam_precoded_channel_gain_ue_to_cell_obj: Any = beam_precoded_channel_gain_ue_to_cell_obj

        # Outputs
        self.RSS_results_per_frequency_layer: Dict[str, np.ndarray] = {}


    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["RSS_results_per_frequency_layer"]
    

    def process(self, rescheduling_us: int = -1) -> int:
        """
        Processes the RSS calculations for all UEs and cells.

        Args:
            rescheduling_us: Rescheduling time in microseconds. Defaults to -1.

        Returns:
            Updated rescheduling time.
        """
        # Determine beam type
        self.beam_type: str = "CRS" if self.beam_conf_obj is None else self.beam_conf_obj.beam_type

        # Extract network parameters
        
        # Number of beams
        number_of_beams: np.ndarray = self.network_deployment_obj.df_ep[f"{self.beam_type}_number_of_beams"].to_numpy(dtype=int)
        
        # TX power
        BS_tx_power: np.ndarray = self.network_deployment_obj.df_ep[f"BS_tx_power_{self.beam_type}_RE_dBm"].to_numpy(dtype=np.single)

        # Start performance timer
        t_start: float = time.perf_counter()

        # Calculate RSS 
        if self.compute_type == "beam_RSS_per_PRB":
            # Process each frequency layer
            for frequency_key, frequency_layer_data in self.beam_precoded_channel_gain_ue_to_cell_obj.precoded_channel_gain_results_per_frequency_layer.items():
                precoded_gain: np.ndarray = frequency_layer_data["precoded_channel_gain_b_to_a_dB"]

                # Compute RSS and store results
                RSS_per_PRB_ue_to_cell_dBm = self._compute_rss(
                    BS_tx_power[frequency_layer_data["cells_in_frequency_mask"]], 
                    number_of_beams[frequency_layer_data["cells_in_frequency_mask"]], 
                    precoded_gain
                )
                
                # Store results for the frequency layer
                self.RSS_results_per_frequency_layer[frequency_key] = copy.deepcopy(frequency_layer_data)
                del self.RSS_results_per_frequency_layer[frequency_key]["precoded_channel_gain_b_to_a_dB"]
                self.RSS_results_per_frequency_layer[frequency_key].update({
                    "RSS_per_PRB_ue_to_cell_dBm": RSS_per_PRB_ue_to_cell_dBm,
                })                

        # Log elapsed time
        log_calculations_time('RSS', t_start)

        return rescheduling_us


    def _compute_rss(
        self,
        BS_tx_power: np.ndarray,
        number_of_beams: np.ndarray,
        precoded_gain: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the RSS per PRB for all UEs and cells.

        Args:
            BS_tx_power: Transmission power per SSB RE (dBm). Shape: (num_cells,).
            number_of_beams: Number of beams per cell. Shape: (num_cells,).
            precoded_gain: Precoded channel gains (dB). Shape: (num_UE_antennas, num_beams).

        Returns:
            np.ndarray: RSS per PRB (dBm). Shape: (num_UE_antennas, num_beams).
        """
        # Extend TX power matrix from cell to UE x cell and UE x beam levels
        extended_tx_power: np.ndarray = np.repeat(BS_tx_power[np.newaxis, :] * np.ones((precoded_gain.shape[1], 1)), number_of_beams, axis=1)

        # Calculate RSS
        return extended_tx_power + precoded_gain


class RSS_SSB(RSS):
    pass

class RSS_CSI_RS(RSS):
    pass
