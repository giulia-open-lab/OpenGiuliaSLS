# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 14:13:46 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import time
import copy

import numpy as np
import torch
from typing import Any, Dict, List

from giulia.event_driven.snapshot_control import Snapshot_control
from giulia.fs import results_file
from giulia.tools import tools
from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable


class Fast_Fading_Gain(Saveable):
    """
    A class for simulating and processing fast fading channel models in a communication network.
    Handles configurations, deployments, and fast fading coefficient calculations.
    """

    def __init__(
        self,
        simulation_config_obj: Any,
        network_deployment_obj: Any,
        cell_antenna_array_structure_obj: Any,
        ue_antenna_array_structure_obj: Any,
        time_frequency_resource_obj: Any,
        distance_angles_ueAnt_to_cellAnt_obj: Any,
        array_steering_vector_ue_to_cell_obj: Any,
        K_factor_ue_to_cell_obj: Any,
    ):
        """
        Initializes the Fast_Fading_Gain class with simulation, deployment, and channel-related parameters.

        Args:
            simulation_config_obj: Configuration for the simulation environment.
            network_deployment_obj: Network deployment details (e.g., base station configurations).
            cell_antenna_array_structure_obj: Details of cell antenna structures.
            ue_antenna_array_structure_obj: Details of user equipment (UE) antenna structures.
            distance_angles_ueAnt_to_cellAnt_obj: Object with distances and angles between UE and cell antennas.
            array_steering_vector_ue_to_cell_obj: Object with array steering vectors between UE and cell antennas.
            K_factor_ue_to_cell_obj: Object containing K-factor values between UE and cell antennas (LoS to NLoS ratio).
        """
        
        super().__init__()

        # Device setup
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Input storage
        self.simulation_config_obj = simulation_config_obj
        self.network_deployment_obj = network_deployment_obj
        self.cell_antenna_array_structure_obj = cell_antenna_array_structure_obj
        self.ue_antenna_array_structure_obj = ue_antenna_array_structure_obj
        self.time_frequency_resource_obj = time_frequency_resource_obj
        self.distance_angles_ueAnt_to_cellAnt_obj = distance_angles_ueAnt_to_cellAnt_obj
        self.array_steering_vector_ue_to_cell_obj = array_steering_vector_ue_to_cell_obj
        self.K_factor_ue_to_cell_obj = K_factor_ue_to_cell_obj

        # Placeholder for outputs
        self.fast_fading_channel_coeff_bAnt_to_aAnt_complex: torch.Tensor = torch.tensor([], dtype=torch.chalf)
        self.LoS_channel_coeff_bAnt_to_aAnt_complex: torch.Tensor = torch.tensor([], dtype=torch.chalf)

        # Dictionary to store results for each frequency layer
        self.fading_results_per_frequency_layer_result: Dict[float, Dict[str, Any]] = {}


    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["fading_results_per_frequency_layer_result"]

    def process(self, rescheduling_us: int = -1) -> int:
        """
        Processes input data to calculate fast fading coefficients for each frequency layer and channel model.

        Args:
            rescheduling_us: Rescheduling time in microseconds. Defaults to -1.

        Returns:
            The updated rescheduling time.
        """
        # Extract simulation parameters
        self.seed: int = self.simulation_config_obj.random_seed
        self.link_direction: str = self.simulation_config_obj.link_direction

        # Extract network parameters
        self.bs_fast_channel_models: np.ndarray = self.network_deployment_obj.df_ep["BS_fast_channel_model"].to_numpy()
        self.dl_carrier_frequency_GHz: np.ndarray = self.network_deployment_obj.df_ep["dl_carrier_frequency_GHz"].to_numpy(dtype=np.single)
        if self.link_direction == "downlink":
            self.available_PRBs: np.ndarray = self.network_deployment_obj.df_ep["dl_PRBs_available"].to_numpy(dtype=int)
        else:
            self.available_PRBs: np.ndarray = self.network_deployment_obj.df_ep["ul_available_PRBs"].to_numpy(dtype=int)

        # UE and cell antenna mappings
        self.antenna_to_node_b_mapping: np.ndarray = self.ue_antenna_array_structure_obj.antenna_to_node_mapping
        self.antenna_to_node_a_mapping: np.ndarray = self.cell_antenna_array_structure_obj.antenna_to_node_mapping

        # Channel characteristics
        self.distances_bAnt_to_aAnt_3d_wraparound_m: np.ndarray = self.distance_angles_ueAnt_to_cellAnt_obj.distance_b_to_a_3d_wraparound_m
        self.array_steering_vector_bAnt_to_aAnt_complex: np.ndarray = self.array_steering_vector_ue_to_cell_obj.array_steering_vector_bAnt_to_aAnt_complex
        self.K_factor_b_to_a_dB: np.ndarray = self.K_factor_ue_to_cell_obj.K_factor_b_to_a_dB

        # Start timer for performance monitoring
        t_start: float = time.perf_counter()

        # Expand UE and cell matrices to antenna level
        values_ue_ant, counts_ue_ant = np.unique(self.antenna_to_node_b_mapping, return_counts=True)
        k_factor_bAnt_to_a_dB: np.ndarray = np.repeat(self.K_factor_b_to_a_dB, counts_ue_ant, axis=0)

        values_cell_ant, counts_cell_ant = np.unique(self.antenna_to_node_a_mapping, return_counts=True)
        self.k_factor_bAnt_to_aAnt_dB: np.ndarray = np.repeat(k_factor_bAnt_to_a_dB, counts_cell_ant, axis=1)     
        
        # Determine what should be processed
        """
        Logic Explanation:
        This logic is designed to handle two different data structures based on the preset type, each with its own trade-offs.
        
        1. **Standard or Single-Matrix Preset** (e.g., Multi-Connectivity, Mobility, etc.):
           - In this case, there is only one entry in `dl_frequency_layer_info`, specifically "all_freq".
           - This single matrix contains channel information for all UEs across all cells.
           - **Pros:**
             - No need for prior knowledge of UE association.
             - Simplifies processing since all links are available in a single structure.
           - **Cons:**
             - Higher memory usage since it includes all UE-to-cell links, even those that may not be relevant after UE assocation.
           - **Matrix Size: (Num UEs) × (Num Cells Across All Frequencies)**
        
        2. **Multi-Frequency Layer Preset**:
           - Here, the data is structured as a dictionary where each entry corresponds to a specific frequency layer.
           - Each frequency layer is defined by its carrier frequency and contains a matrix with channel information relevant to that frequency.
           - The cells within a layer operate on the same carrier frequency.
           - UEs are associated with the best-serving cells in a given frequency layer based on a user association metric.
           - **Pros:**
             - More memory-efficient, as it avoids storing unnecessary UE-to-cell links across different frequency layers.
             - Better suited for cases where heavy frequency-layer-specific processing is needed.
           - **Cons:**
             - Requires prior knowledge of UE association, adding preprocessing complexity.
             - More fragmented structure, complicates multi-frequency operations, requiring additional management of multiple matrices.
           - **Matrix Size (Per Layer): (Num UEs in Layer) × (Num Cells in That Frequency Layer)**
        
        The following code ensures that:
        - In the single-matrix case, we process only the "all_freq" entry.
        - In the per-frequency-layer case, we efficiently process only the relevant frequency layers.
        """
         
        # Extract relevant entries to process based on a explicit indication of the preset being used. 
            # Extract relevant data
        valid_entries = self.time_frequency_resource_obj.dl_frequency_layer_info.copy()
        
            # Handle filtering based on the simulation preset
        if self.simulation_config_obj.preset == "GiuliaMfl":
            # Remove "all_freq" if other valid entries exist
            valid_entries.pop("all_freq", None)
        else:
            # Keep only "all_freq", discard everything else
            valid_entries = {"all_freq": valid_entries["all_freq"]} if "all_freq" in valid_entries else {}
        
        # Process the valid entries
        for dl_carrier_frequency_GHz, frequency_layer_info in valid_entries.items():
            
            # Identify cells and antennas associated with cells in the current frequency
            available_PRBs_for_frequency: int = frequency_layer_info["available_PRBs"] 
            cells_in_frequency_mask: np.ndarray = frequency_layer_info["cells_in_frequency_mask"]
            cell_ants_in_frequency_mask: np.ndarray = frequency_layer_info["cell_ants_in_frequency_mask"]
            num_cell_ants_in_frequency: int = frequency_layer_info["num_cell_ants"]
        
            # Identify UEs and antennas associated with cells in the current frequency
            ue_ants_in_frequency_mask: np.ndarray = frequency_layer_info["ue_ants_in_frequency_mask"]
            num_ue_ants_in_frequency: int = frequency_layer_info["num_ue_ants"]

            # Filter the array steering vector and K-factor
            array_steering_vector_bAnt_to_aAnt_complex: np.ndarray = self.array_steering_vector_bAnt_to_aAnt_complex[:, ue_ants_in_frequency_mask, :][:, :, cell_ants_in_frequency_mask]
            k_factor_bAnt_to_aAnt_dB: np.ndarray = self.k_factor_bAnt_to_aAnt_dB[ue_ants_in_frequency_mask, :][:, cell_ants_in_frequency_mask]
                
            # Initialize tensors for fast fading and LoS coefficients
            fast_fading_channel_coeff_bAnt_to_aAnt_complex = torch.zeros((available_PRBs_for_frequency, num_ue_ants_in_frequency, num_cell_ants_in_frequency),dtype=torch.chalf,device=self.device)
        
            # Process each channel model
            bs_fast_channel_model_set: set = set(self.bs_fast_channel_models)
            for bs_fast_channel_model in bs_fast_channel_model_set:
                # Mask for cells with the current channel model
                mask_cell: np.ndarray = self.bs_fast_channel_models[cells_in_frequency_mask] == bs_fast_channel_model
                mask_cell_antennas : np.ndarray = np.repeat(mask_cell, counts_cell_ant[cells_in_frequency_mask], axis=0)

                # Calculate fast fading
                if bs_fast_channel_model == "Rician":
                    fast_fading_channel_coeff_bAnt_to_aAnt_complex[:, :, mask_cell_antennas] = \
                        self.Rician_fading_far_field_approx_torch(available_PRBs_for_frequency, k_factor_bAnt_to_aAnt_dB, array_steering_vector_bAnt_to_aAnt_complex)
                elif bs_fast_channel_model == "Rayleigh":
                    fast_fading_channel_coeff_bAnt_to_aAnt_complex[:, :, mask_cell_antennas] = \
                        self.Rayleigh_fading(available_PRBs_for_frequency,array_steering_vector_bAnt_to_aAnt_complex.shape[1],array_steering_vector_bAnt_to_aAnt_complex.shape[2])
        

            # Store results
            self.fading_results_per_frequency_layer_result[dl_carrier_frequency_GHz] = copy.deepcopy(self.time_frequency_resource_obj.dl_frequency_layer_info[dl_carrier_frequency_GHz])
            if "RSRP_ue_to_cell_dBm" in self.fading_results_per_frequency_layer_result[dl_carrier_frequency_GHz]:
                del self.fading_results_per_frequency_layer_result[dl_carrier_frequency_GHz]["RSRP_ue_to_cell_dBm"]
            self.fading_results_per_frequency_layer_result[dl_carrier_frequency_GHz].update({    
                "fast_fading_channel_coeff_bAnt_to_aAnt_complex": fast_fading_channel_coeff_bAnt_to_aAnt_complex,
            })

            
        # Save results to plot
        snapshot_control = Snapshot_control.get_instance()
        if self.simulation_config_obj.plot == 1 and snapshot_control.num_snapshots == 0:
            aux: np.ndarray = 10 * torch.log10(torch.abs(self.fast_fading_channel_coeff_bAnt_to_aAnt_complex) ** 2).cpu().numpy()
            file_name: str = results_file(self.simulation_config_obj.project_name, 'to_plot_fast_fading_gain')
            np.savez(file_name, fast_fading_gain_bAnt_to_aAnt_dB=aux)            

        # Log elapsed time
        log_calculations_time('Fast fading', t_start)
        
        return rescheduling_us
    

    def Rician_fading(
        self,
        available_PRBs: int,
        k_factor_bAnt_to_aAnt_dB: np.ndarray,
        dl_carrier_wavelength_a_antennas_m: np.ndarray,
        distances_bAnt_to_aAnt_3d_wraparound_m: np.ndarray,
        array_factor_linear: np.ndarray,
    ) -> np.ndarray:
        """
        Computes Rician fading using both LoS and NLoS components.
    
        Args:
            available_PRBs: Number of physical resource blocks available.
            k_factor_bAnt_to_aAnt_dB: K-factor matrix in dB for Rician fading.
            dl_carrier_wavelength_a_antennas_m: Wavelengths at cell antennas.
            distances_bAnt_to_aAnt_3d_wraparound_m: Distances between UE and cell antennas.
            array_factor_linear: Array factor in linear scale.
    
        Returns:
            A NumPy array representing the fast fading coefficients.
        """
        # Convert K-factor from dB to linear scale
        K_factor: np.ndarray = tools.dBm_to_mW(k_factor_bAnt_to_aAnt_dB)
    
        # Calculate NLoS Rayleigh component
        H_Rayleigh_fading: np.ndarray = self.Rayleigh_fading_parallel_thread(available_PRBs, k_factor_bAnt_to_aAnt_dB.shape[0], k_factor_bAnt_to_aAnt_dB.shape[1])
    
        # Calculate LoS deterministic component
        phase_shifts_rad: np.ndarray = np.modf(distances_bAnt_to_aAnt_3d_wraparound_m / dl_carrier_wavelength_a_antennas_m[np.newaxis, :])[0]
        H_LoS_fading: np.ndarray = np.cos(2 * np.pi * phase_shifts_rad) + 1j * np.sin(2 * np.pi * phase_shifts_rad)
    
        # Compute fast fading using Rician principle
        return np.sqrt(1 / (K_factor + 1)) * H_Rayleigh_fading + np.sqrt(K_factor / (1 + K_factor)) * H_LoS_fading
    
    
    def Rician_fading_far_field_approx(
        self,
        available_PRBs: int,
        k_factor_bAnt_to_aAnt_dB: np.ndarray,
        array_factor_linear: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes Rician fading using a far-field approximation.
    
        Args:
            available_PRBs: Number of physical resource blocks available.
            k_factor_bAnt_to_aAnt_dB: K-factor matrix in dB for Rician fading.
            array_factor_linear: Array factor in linear scale.
    
        Returns:
            A tuple containing:
            - NLoS + LoS coefficients for fast fading.
            - LoS coefficients.
        """
        # Convert K-factor from dB to linear scale
        K_factor: np.ndarray = tools.dBm_to_mW(k_factor_bAnt_to_aAnt_dB)
    
        # Calculate NLoS Rayleigh component
        H_Rayleigh_fading: np.ndarray = self.Rayleigh_fading(available_PRBs, k_factor_bAnt_to_aAnt_dB.shape[0], k_factor_bAnt_to_aAnt_dB.shape[1])
    
        # Compute LoS and NLoS coefficients
        LoS_channel_coefficients: np.ndarray = np.sqrt(K_factor / (1 + K_factor)) * array_factor_linear
        NLoS_channel_coefficients: np.ndarray = np.sqrt(1 / (K_factor + 1)) * H_Rayleigh_fading
    
        # Combine to compute fast fading
        return NLoS_channel_coefficients + LoS_channel_coefficients #, LoS_channel_coefficients
    
    
    def Rician_fading_far_field_approx_torch(
        self,
        available_PRBs: int,
        k_factor_bAnt_to_aAnt_dB: np.ndarray,
        array_factor_linear: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Rician fading using a far-field approximation with PyTorch.
    
        Args:
            available_PRBs: Number of physical resource blocks available.
            k_factor_bAnt_to_aAnt_dB: K-factor matrix in dB for Rician fading.
            array_factor_linear: Array factor in linear scale.
    
        Returns:
            A tuple containing:
            - NLoS + LoS coefficients for fast fading.
            - LoS coefficients.
        """
        # Convert K-factor from dB to linear scale
        K_factor: np.ndarray = tools.dBm_to_mW(k_factor_bAnt_to_aAnt_dB)
    
        # Calculate NLoS Rayleigh component
        H_Rayleigh_fading: torch.Tensor = self.Rayleigh_fading_torch(available_PRBs, k_factor_bAnt_to_aAnt_dB.shape[0], k_factor_bAnt_to_aAnt_dB.shape[1])
    
        # Compute LoS and NLoS coefficients
        sqrt_K_factor: torch.Tensor = torch.sqrt(torch.tensor(K_factor / (1 + K_factor), device=self.device, dtype=torch.half))
        LoS_channel_coefficients: torch.Tensor = sqrt_K_factor * torch.tensor(array_factor_linear, device=self.device, dtype=torch.chalf)
    
        sqrt_K_factor_comp: torch.Tensor = torch.sqrt(torch.tensor(1 / (K_factor + 1), device=self.device, dtype=torch.half))
        NLoS_channel_coefficients: torch.Tensor = sqrt_K_factor_comp * H_Rayleigh_fading
    
        # Combine to compute fast fading
        return NLoS_channel_coefficients + LoS_channel_coefficients #, LoS_channel_coefficients
    
    
    def Rayleigh_fading(
        self,
        available_PRBs: int,
        number_of_RX_antennas: int,
        number_of_TX_antennas: int,
    ) -> np.ndarray:
        """
        Computes Rayleigh fading for the specified number of antennas and PRBs.
    
        Args:
            available_PRBs: Number of physical resource blocks available.
            number_of_RX_antennas: Number of receiving antennas.
            number_of_TX_antennas: Number of transmitting antennas.
    
        Returns:
            A NumPy array representing Rayleigh fading coefficients.
        """
        rng: np.random.RandomState = np.random.RandomState(self.seed + 0)
    
        # Generate real and imaginary parts for the fading coefficients
        real_part: np.ndarray = rng.standard_normal((available_PRBs, number_of_RX_antennas, number_of_TX_antennas)).astype(np.half)
        imag_part: np.ndarray = rng.standard_normal((available_PRBs, number_of_RX_antennas, number_of_TX_antennas)).astype(np.half)
    
        # Compute Rayleigh fading coefficients
        return (1 / np.sqrt(2)) * (real_part + 1j * imag_part)
    
    
    def Rayleigh_fading_torch(
        self,
        available_PRBs: int,
        number_of_RX_antennas: int,
        number_of_TX_antennas: int,
    ) -> torch.Tensor:
        """
        Computes Rayleigh fading for the specified number of antennas and PRBs using PyTorch.
    
        Args:
            available_PRBs: Number of physical resource blocks available.
            number_of_RX_antennas: Number of receiving antennas.
            number_of_TX_antennas: Number of transmitting antennas.
    
        Returns:
            A PyTorch tensor representing Rayleigh fading coefficients.
        """
        rng: torch.Generator = torch.Generator(device=self.device)
        rng.manual_seed(self.seed + 0)
    
        # Generate real and imaginary parts for the fading coefficients
        real_part: torch.Tensor = \
            torch.zeros((available_PRBs, number_of_RX_antennas, number_of_TX_antennas), device=self.device, dtype=torch.float16).normal_(generator=rng)
    
        imag_part: torch.Tensor = \
            torch.zeros((available_PRBs, number_of_RX_antennas, number_of_TX_antennas), device=self.device, dtype=torch.float16).normal_(generator=rng)
    
        # Compute Rayleigh fading coefficients
        return (1 / torch.sqrt(torch.tensor(2.0, device=self.device))) * (real_part + 1j * imag_part)
