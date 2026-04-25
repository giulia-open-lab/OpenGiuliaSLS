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
from typing import Any, List

from giulia.tools import tools
from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable

class Noise(Saveable):
    """
    A class for calculating noise power metrics for downlink communication in a network.
    Computes noise power over the full bandwidth and per resource element.
    """

    def __init__(
        self,
        network_deployment_obj: Any,
        ue_deployment_obj: Any,
    ):
        """
        Initializes the Noise class with deployment objects.

        Args:
            network_deployment_obj: Object containing network deployment details.
            ue_deployment_obj: Object containing user equipment (UE) deployment details.
        """

        super().__init__()

        # Input storage
        self.network_deployment_obj = network_deployment_obj
        self.ue_deployment_obj = ue_deployment_obj

        # Outputs
        self.dl_noise_full_bandwidth_ue_to_cell_dBm: np.ndarray = np.array([])  # Noise over full bandwidth
        self.dl_noise_per_resource_element_ue_to_cell_dBm: np.ndarray = np.array([])  # Noise per resource element


    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["dl_noise_full_bandwidth_ue_to_cell_dBm"]


    def process(self, rescheduling_us: int = -1) -> int:
        """
        Processes noise calculations and updates class outputs.

        Args:
            rescheduling_us: Rescheduling time in microseconds. Defaults to -1.

        Returns:
            The updated rescheduling time.
        """
        # Extract network parameters
        dl_bandwidth_MHz = self.network_deployment_obj.df_ep["dl_bandwidth_MHz"].to_numpy()
        dl_PRBs_available = self.network_deployment_obj.df_ep["dl_PRBs_available"].to_numpy()
        subcarriers_per_PRB = self.network_deployment_obj.df_ep["subcarriers_per_PRB"].to_numpy()

        # Extract UE deployment parameters
        noise_figure_dB = self.ue_deployment_obj.df_ep["noise_figure_dB"].to_numpy()

        # Start timer
        t_start: float = time.perf_counter()

        # Calculate noise power over the full bandwidth
        self.dl_noise_full_bandwidth_ue_to_cell_dBm = \
            self.dl_noise_full_bandwidth(dl_bandwidth_MHz, noise_figure_dB)

        # Calculate noise power per resource element
        self.dl_noise_per_resource_element_ue_to_cell_dBm = \
            self.dl_noise_per_resource_element(dl_bandwidth_MHz, dl_PRBs_available, subcarriers_per_PRB, noise_figure_dB)

        # Print elapsed time
        log_calculations_time('Noise', t_start)

        return rescheduling_us


    def dl_noise_full_bandwidth(
        self,
        dl_bandwidth_MHz: np.ndarray,
        noise_figure_dBm: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates noise power over the full bandwidth.

        Args:
            dl_bandwidth_MHz: Downlink bandwidth in MHz for each cell.
            noise_figure_dBm: Noise figure in dB for each UE.

        Returns:
            A numpy array of noise power in dBm for the full bandwidth.
        """
        noise_density_dBm_per_Hz: float = -174  # Thermal noise density in dBm/Hz

        # Number of UEs and cells
        num_ues = len(noise_figure_dBm)
        num_cells = len(dl_bandwidth_MHz)

        # Expand bandwidth and noise figure to match UE-cell pairs
        dl_bandwidth_ue_cell_MHz = np.tile(dl_bandwidth_MHz, (num_ues, 1))
        noise_figure_ue_cell_dBm = np.tile(noise_figure_dBm[:, np.newaxis], (1, num_cells))

        # Adjust bandwidth to account for 10% guard bands and convert to Hz
        bandwidth_Hz = 0.9 * dl_bandwidth_ue_cell_MHz * 1e6

        # Calculate noise power in mW
        noise_power_mW = tools.dBm_to_mW(noise_density_dBm_per_Hz) * bandwidth_Hz

        # Convert noise power to dBm and add noise figure
        return tools.mW_to_dBm(noise_power_mW) + noise_figure_ue_cell_dBm


    def dl_noise_per_resource_element(
        self,
        dl_bandwidth_MHz: np.ndarray,
        dl_PRBs_available: np.ndarray,
        subcarriers_per_PRB: np.ndarray,
        noise_figure_dB: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates noise power per resource element (RE).

        Args:
            dl_bandwidth_MHz: Downlink bandwidth in MHz for each cell.
            dl_PRBs_available: Number of physical resource blocks (PRBs) available per cell.
            subcarriers_per_PRB: Number of subcarriers per PRB.
            noise_figure_dB: Noise figure in dB for each UE.

        Returns:
            A numpy array of noise power in dBm per resource element.
        """
        noise_density_dBm_per_Hz: float = -174  # Thermal noise density in dBm/Hz

        # Number of UEs and cells
        num_ues = len(noise_figure_dB)
        num_cells = len(dl_bandwidth_MHz)

        # Expand parameters to match UE-cell pairs
        dl_bandwidth_ue_cell_MHz = np.tile(dl_bandwidth_MHz, (num_ues, 1))
        dl_PRBs_ue_cell = np.tile(dl_PRBs_available, (num_ues, 1))
        subcarriers_ue_cell = np.tile(subcarriers_per_PRB, (num_ues, 1))
        noise_figure_ue_cell_dB = np.tile(noise_figure_dB[:, np.newaxis], (1, num_cells))

        # Calculate bandwidth per resource element (RE) in Hz, considering guard bands
        bandwidth_per_RE_Hz = (0.9 * dl_bandwidth_ue_cell_MHz * 1e6) / dl_PRBs_ue_cell / subcarriers_ue_cell

        # Calculate noise power in mW
        noise_power_mW = tools.dBm_to_mW(noise_density_dBm_per_Hz) * bandwidth_per_RE_Hz

        # Convert noise power to dBm and add noise figure
        return tools.mW_to_dBm(noise_power_mW) + noise_figure_ue_cell_dB

