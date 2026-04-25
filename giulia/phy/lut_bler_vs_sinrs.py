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

import time
import warnings
from typing import List

import numpy as np
import pandas as pd

from giulia.event_driven.snapshot_control import Snapshot_control
from giulia.fs import phy_file, results_file
from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable


class Lut_Bler_Vs_Sinr(Saveable):
    """
    Class for LUT-based calculation of BLER (Block Error Rate) versus SINR (Signal-to-Interference-plus-Noise Ratio).
    """

    def __init__(self, simulation_config_obj: object, resource_mcs_obj: object):
        """
        Initialize the Lut_Bler_Vs_Sinr class with simulation and resource configuration objects.

        Args:
            simulation_config_obj: Configuration object for the simulation.
            resource_mcs_obj: Object containing MCSs (Modulation and Coding Schemes) and related parameters.
        """
        
        super().__init__()

        # Simulation and resource configuration objects
        self.simulation_config_obj: object = simulation_config_obj
        self.resource_mcs_obj: object = resource_mcs_obj

        # Outputs
        self.df_bler_table: pd.DataFrame = pd.DataFrame()  # BLER vs SINR DataFrame
        self.bler_table: np.ndarray = np.array([])        # BLER vs SINR NumPy array
        self.par1: np.ndarray = np.array([])             # Parameter 1 for BLER calculation
        self.par2: np.ndarray = np.array([])             # Parameter 2 for BLER calculation


    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["bler_table"]


    def process(self, rescheduling_us: int = -1) -> None:
        """
        Process the BLER vs SINR data and generate LUTs for the simulation.

        Args:
            rescheduling_us: (Unused) Intended for rescheduling timer (default: -1).
        """
        t_start = time.perf_counter()  # Start the timer

        # Extract MCS parameters from the resource object
        self.modulation_and_coding_schemes: list[str] = self.resource_mcs_obj.modulation_and_coding_schemes
        self.number_of_modulation_and_coding_schemes: int = self.resource_mcs_obj.number_of_modulation_and_coding_schemes
        self.modulation_index_per_modulation_and_coding_scheme: np.ndarray = self.resource_mcs_obj.modulation_index_per_modulation_and_coding_scheme
        self.transport_block_sizes_in_prbs: np.ndarray = self.resource_mcs_obj.transport_block_sizes_in_prbs
        self.number_of_transport_block_sizes: int = self.resource_mcs_obj.number_of_transport_block_sizes

        # Read BLER vs SINR table
        file = phy_file('bler', 'dirac_fit.txt')
        self.df_bler_table = pd.read_table(file, delimiter=r'\s+')
        self.bler_table = self.df_bler_table.to_numpy(dtype=np.single)

        # Extract fitting parameters (par1 and par2) for each MCS and TBS
        self.par1 = np.zeros((self.number_of_modulation_and_coding_schemes, self.number_of_transport_block_sizes))
        self.par2 = np.zeros((self.number_of_modulation_and_coding_schemes, self.number_of_transport_block_sizes))

        index = 0
        for mcs_index in range(self.number_of_modulation_and_coding_schemes):
            for tb_size_index in range(self.number_of_transport_block_sizes):
                self.par1[mcs_index, tb_size_index] = self.bler_table[index, 3]
                self.par2[mcs_index, tb_size_index] = self.bler_table[index, 4]
                index += 1

        # Generate BLER vs SINR LUTs
        bler_per_sinr_mcs: np.ndarray = \
            np.array([self.calculate_bler_per_ue(100, mcs_index, np.arange(-10, 30, 0.1)) 
                      for mcs_index in range(self.number_of_modulation_and_coding_schemes)], dtype=float).T

        # Save LUTs if snapshot control and plotting conditions are met
        snapshot_control = Snapshot_control.get_instance()
        if self.simulation_config_obj.plot == 1 and snapshot_control.num_snapshots == 0:
            file = results_file(self.simulation_config_obj.project_name, 'to_plot_lut_bler_vs_sinr')
            np.savez(file, modulation_and_coding_schemes=self.modulation_and_coding_schemes, bler_per_sinr_mcs=bler_per_sinr_mcs)

        ### End
        log_calculations_time('BLER vs SINR LUTs', t_start)


    def calculate_bler_per_ue_and_mcs(self, effective_sinr_per_ue_mcs: np.ndarray) -> np.ndarray:
        """
        Calculate the BLER for all UEs assuming all modulation and coding schemes (MCSs).

        Args:
            effective_sinr_per_ue_mcs: Effective SINR values for each UE and MCS.
                                       Shape: (Number of UEs, Number of Modulation Schemes).

        Returns:
            BLER values per UE and MCS. Shape: (Number of UEs, Number of MCSs).
        """
        bler_per_ue_mcs: np.ndarray = \
            np.array([self.calculate_bler_per_ue(100, mcs_index, effective_sinr_per_ue_mcs[:, modulation_index])
                        for mcs_index, modulation_index in enumerate(self.modulation_index_per_modulation_and_coding_scheme)], dtype=float).T
        
        return bler_per_ue_mcs


    def calculate_bler_per_ue(self, tb_size: int, mcs: int, sinr_dB: np.ndarray) -> np.ndarray:
        """
        Calculate BLER for a given TBS, a given MCS, and a set of SINR values using Dirac fitting.

        Args:
            tb_size: Transport block size.
            mcs: Modulation and coding scheme index.
            sinr_dB: SINR values in dB.

        Returns:
            BLER values corresponding to the input SINR values.
        """
        if tb_size < self.transport_block_sizes_in_prbs[0]:
            raise ValueError("Transport block size is smaller than the smallest defined TBS.")
        elif tb_size > self.transport_block_sizes_in_prbs[-1]:
            raise ValueError("Transport block size is larger than the largest defined TBS.")

        # Parameter extraction
        ix = np.argmax(self.transport_block_sizes_in_prbs >= tb_size)
        tbss0, tbss1 = self.transport_block_sizes_in_prbs[ix], self.transport_block_sizes_in_prbs[ix - 1]
        par1_0, par1_1 = self.par1[mcs, ix], self.par1[mcs, ix - 1]
        par2_0, par2_1 = self.par2[mcs, ix], self.par2[mcs, ix - 1]

        # Linear interpolation for fitting parameters
        par1 = (par1_1 - par1_0) * (tb_size - tbss0) / (tbss1 - tbss0) + par1_0
        par2 = (par2_1 - par2_0) * (tb_size - tbss0) / (tbss1 - tbss0) + par2_0

        # Calculate BLER
        warnings.filterwarnings('ignore')
        bler = 1 / (np.exp((sinr_dB - par1) * par2) + 1)
        return bler


    def calculate_selected_mcs_per_ue(self, bler_per_ue_and_mcs: np.ndarray, bler_target: float) -> np.ndarray:
        """
        Select the maximum MCS for each UE that satisfies the BLER target.

        Args:
            bler_per_ue_and_mcs: BLER values per UE and MCS.
                                 Shape: (Number of UEs, Number of MCSs).
            bler_target: Target BLER value.

        Returns:
            Array of selected MCS indices for each UE.
        """
        # Calculate BLER difference with target
        bler_diff: np.ndarray = bler_target - bler_per_ue_and_mcs
        bler_diff = np.where(bler_diff < 1e-12, np.inf, bler_diff)

        # Find the index of the smallest BLER that meets the target
        selected_mcs_per_ue: np.ndarray = np.argmin(bler_diff, axis=1)

        # Handle edge cases
        max_mcs = bler_per_ue_and_mcs.shape[1] - 1
        ues_with_max_mcs = bler_per_ue_and_mcs[:, -1] < bler_target
        selected_mcs_per_ue[ues_with_max_mcs] = max_mcs

        ues_with_no_mcs = np.all(bler_diff == np.inf, axis=1)
        selected_mcs_per_ue[ues_with_no_mcs] = -1

        return selected_mcs_per_ue

