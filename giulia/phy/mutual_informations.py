# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:52:45 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import os
import time
from typing import List

import numpy as np
import pandas as pd

from giulia import ROOT_DIR
from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable


class Mutual_Information(Saveable):
    """
    Class to compute mutual information and associated calculations for SINR (Signal-to-Interference-plus-Noise Ratio).
    """

    def __init__(self, simulation_config_obj):

        super().__init__()

        """
        Initializes the Mutual_Information class with placeholders for data structures.
        """
        ##### Input storage 
        ######################## 
        self.simulation_config_obj = simulation_config_obj
        
        # DataFrame for Mutual Information vs. SINR
        self.df_mi_table: pd.DataFrame = pd.DataFrame()

        # Mutual Information table as a NumPy array (excluding the SINR column)
        self.mi_table: np.ndarray = np.array([])

        # Array of SINR values extracted from the data
        self.sinrs_array: np.ndarray = np.array([])


    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["mi_table"]


    def process(self, rescheduling_us: int = -1) -> None:
        """
        Processes the mutual information data from a file and populates class attributes.

        Args:
            rescheduling_us: (Unused) Intended for rescheduling timer (default: -1).
        """
        t_start = time.perf_counter()  # Start the timer

        # Define the path to the mutual information file
        file_path: str = os.path.join(ROOT_DIR, '..', 'data', 'phy', 'mi', 'MutualInformationVsSNR.txt')

        # Read the data file into a pandas DataFrame
        self.df_mi_table = pd.read_table(file_path, delimiter=r'\s+')

        # Convert the DataFrame to NumPy arrays
        self.mi_table = self.df_mi_table.to_numpy(dtype=np.single)[:, 2:]  # Exclude the first (SINR) column
        self.sinrs_array = self.df_mi_table.to_numpy(dtype=np.single)[:, 0]  # Extract only the SINR column

        ##### End
        log_calculations_time('Mutual information', t_start)


    def find_closest_value(self, mi_value: float, mi_pool: np.ndarray) -> float:
        """
        Finds the closest SINR value in the SINR array for a given target mutual information value.

        Args:
            mi_value: The target mutual information value.
            mi_pool: Array of mutual information values to search.

        Returns:
            Closest SINR value from `self.sinrs_array`.
        """
        # Calculate the index of the closest value in the mi_pool
        idx: int = np.abs(mi_pool - mi_value).argmin()

        # Return the corresponding SINR value
        return self.sinrs_array[idx]
    

    def calculate_effective_sinr_per_ue_mcs(
        self, PRB_ue_activity: np.ndarray, sinr_per_prb_ue_dB: np.ndarray
    ) -> np.ndarray:
        """
        Calculates effective SINR for each modulation and coding scheme (MCS) for UEs (User Equipments).

        Args:
            PRB_ue_activity: Boolean array indicating active PRBs  (Physical Resource Blocks) for each UE.
                             Shape: (Number of PRBs, Number of UEs)
            sinr_per_prb_ue_dB: Array of SINR values per PRB and UE in dB.
                                Shape: (Number of PRBs, Number of UEs)

        Returns:
            Effective SINR values per UE and MCS.
            Shape: (Number of UEs, Number of MCSs).
        """
        # Initialize an indices array with a default value of -100 to indicate invalid indices
        indices: np.ndarray = np.full(sinr_per_prb_ue_dB.shape, -100, dtype=int)

        # Create a mask for valid SINR values (i.e., those that are not NaN)
        valid_mask: np.ndarray = ~np.isnan(sinr_per_prb_ue_dB)

        # Combine PRB activity and valid SINR values into a single mask
        number_of_prbs = np.size(sinr_per_prb_ue_dB,0)
        active_valid_mask: np.ndarray = PRB_ue_activity[:number_of_prbs,:] & valid_mask

        # Assign indices based on SINR values within the valid range
        indices[active_valid_mask] = (np.round(sinr_per_prb_ue_dB[active_valid_mask] * 10) + 450).astype(int)

        # Assign special index values for extreme SINRs
        indices[valid_mask & (sinr_per_prb_ue_dB >= 35)] = 800  # High SINR
        indices[valid_mask & (sinr_per_prb_ue_dB <= -45)] = 0   # Low SINR

        # Initialize a 3D mutual information array with NaN values
        mi_per_prb_ue_mcs: np.ndarray = \
            np.full((sinr_per_prb_ue_dB.shape[0], sinr_per_prb_ue_dB.shape[1], self.mi_table.shape[1]), 
                    np.nan, dtype=np.single)

        # Populate the mutual information array using valid indices
        valid_indices_mask: np.ndarray = indices != -100  # Mask for valid indices
        valid_indices: np.ndarray = indices[valid_indices_mask]  # Extract valid indices
        mi_per_prb_ue_mcs[valid_indices_mask] = self.mi_table[valid_indices]

        # Compute the average mutual information across PRBs for each UE and MCS
        # Shape of `mi_per_ue_mcs`: (Number of UEs, Number of MCSs)
        mi_per_ue_mcs: np.ndarray = np.nanmean(mi_per_prb_ue_mcs, axis=0)

        # Calculate the distances between average mutual information and the LUT values
        distances: np.ndarray = np.abs(mi_per_ue_mcs[np.newaxis, :, :] - self.mi_table[:, np.newaxis, :])

        # Find the index of the closest SINR for each UE and MCS
        closest_indices: np.ndarray = np.argmin(distances, axis=0)  # Shape: (Number of UEs, Number of MCSs)

        # Map the closest indices to actual SINR values
        effective_sinr_per_ue_mcs: np.ndarray = self.sinrs_array[closest_indices]

        return effective_sinr_per_ue_mcs
