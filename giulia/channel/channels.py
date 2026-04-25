
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:40:53 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import copy
import itertools
import os
import time
from typing import Any, Dict, List


import numpy as np
import sionna
import tensorflow as tf
import torch
from sionna.rt import Transmitter, Receiver  # Import Sionna RT components

from giulia.tools import tools
from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable


class LoS_Channel(Saveable):
    """
    A class for simulating and processing Line-of-Sight (LoS) channel models in a communication network.
    Handles configurations, deployments, and LoS channel gain calculations.
    """

    def __init__(
        self,
        simulation_config_obj: Any,
        network_deployment_obj: Any,
        resource_time_frequency_obj: Any,
        cell_antenna_array_structure_obj: Any,
        ue_antenna_array_structure_obj: Any,
        slow_channel_gain_ue_to_cell_obj: Any,
        array_steering_vector_ue_to_cell_obj: Any,
        K_factor_ue_to_cell_obj: Any,
    ):
        """
        Initializes the LoS_Channel class with configuration and deployment objects.

        Args:
            simulation_config_obj: Configuration for the simulation environment.
            network_deployment_obj: Network deployment details (e.g., base station configurations).
            resource_time_frequency_obj: Resource allocation for time-frequency grids.
            cell_antenna_array_structure_obj: Details of cell antenna structures.
            ue_antenna_array_structure_obj: Details of user equipment (UE) antenna structures.
            slow_channel_gain_ue_to_cell_obj: Object representing slow channel gain between UE and cell.
            array_steering_vector_ue_to_cell_obj: Object representing the array steering vector between UE and cell.
            K_factor_ue_to_cell_obj: Object containing K-factor values between UE and cell (LoS to NLoS ratio).
        """

        super().__init__()

        # Device setup
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Input parameters
        self.simulation_config_obj: Any = simulation_config_obj
        self.network_deployment_obj: Any = network_deployment_obj
        self.resource_time_frequency_obj: Any = resource_time_frequency_obj
        self.cell_antenna_array_structure_obj: Any = cell_antenna_array_structure_obj
        self.ue_antenna_array_structure_obj: Any = ue_antenna_array_structure_obj
        self.slow_channel_gain_ue_to_cell_obj: Any = slow_channel_gain_ue_to_cell_obj
        self.array_steering_vector_ue_to_cell_obj: Any = array_steering_vector_ue_to_cell_obj
        self.K_factor_ue_to_cell_obj: Any = K_factor_ue_to_cell_obj

        # Output storage for results
        self.channel_results_per_frequency_layer_result: Dict[float, Dict[str, Any]] = {}


    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["channel_results_per_frequency_layer_result"]


    def process(self, rescheduling_us: int = -1) -> int:
        """
        Processes the LoS channel calculations and updates the class output.

        Args:
            rescheduling_us: Rescheduling time in microseconds. Defaults to -1.

        Returns:
            The updated rescheduling time.
        """
        # Set random seed for reproducibility
        torch.manual_seed(self.simulation_config_obj.random_seed)

        # Extract relevant input data
        self.bs_fast_channel_models: np.ndarray = self.network_deployment_obj.df_ep["BS_fast_channel_model"].to_numpy()
        self.antenna_to_node_a_mapping: np.ndarray = self.cell_antenna_array_structure_obj.antenna_to_node_mapping
        self.antenna_to_node_b_mapping: np.ndarray = self.ue_antenna_array_structure_obj.antenna_to_node_mapping
        self.slow_channel_gain_b_to_a_dB: np.ndarray = self.slow_channel_gain_ue_to_cell_obj.slow_channel_results_per_frequency_layer["all_freq"]["slow_channel_gain_b_to_a_dB"]
        self.array_steering_vector_bAnt_to_aAnt_complex: np.ndarray = (self.array_steering_vector_ue_to_cell_obj.array_steering_vector_bAnt_to_aAnt_complex)
        self.K_factor_b_to_a_dB: np.ndarray = self.K_factor_ue_to_cell_obj.K_factor_b_to_a_dB

        # Initialize output matrix
        num_ues: int = self.slow_channel_gain_b_to_a_dB.shape[0]
        num_cells: int = self.slow_channel_gain_b_to_a_dB.shape[1]
        dl_LoS_channel_gain_ueAnt_to_cellAnt_complex: np.ndarray = np.zeros((num_ues, num_cells), dtype=np.csingle)

        # Start the timer for performance monitoring
        t_start: float = time.perf_counter()

        # Expand matrices for UE and cell antennas
        values_ue, counts_ue = np.unique(self.antenna_to_node_b_mapping, return_counts=True)
        slow_channel_gain_bAnt_to_a_dB: np.ndarray = np.repeat(self.slow_channel_gain_b_to_a_dB, counts_ue, axis=0)
        k_factor_bAnt_to_a_dB: np.ndarray = np.repeat(self.K_factor_b_to_a_dB, counts_ue, axis=0)

        values_cell, counts_cell = np.unique(self.antenna_to_node_a_mapping, return_counts=True)
        slow_channel_gain_bAnt_to_aAnt_dB: np.ndarray = np.repeat(slow_channel_gain_bAnt_to_a_dB, counts_cell, axis=1)
        self.k_factor_bAnt_to_aAnt_dB: np.ndarray = np.repeat(k_factor_bAnt_to_a_dB, counts_cell, axis=1)

        # Process unique propagation models
        for bs_fast_channel_model in np.unique(self.bs_fast_channel_models):
            mask_cell: np.ndarray = self.bs_fast_channel_models == bs_fast_channel_model
            mask_cell_antennas: np.ndarray = np.repeat(mask_cell, counts_cell, axis=0)

            # Filter matrices for the current model
            slow_gain_filtered: np.ndarray = slow_channel_gain_bAnt_to_aAnt_dB[:, mask_cell_antennas]
            steering_vector_filtered: np.ndarray = self.array_steering_vector_bAnt_to_aAnt_complex[:, :, mask_cell_antennas]
            k_factor_filtered: np.ndarray = self.k_factor_bAnt_to_aAnt_dB[:, mask_cell_antennas]

            # Compute LoS channel gains
            dl_LoS_channel_gain_ueAnt_to_cellAnt_complex = \
                self.los_channel_gain_b_to_a(slow_gain_filtered, steering_vector_filtered, k_factor_filtered)
            
        # Store results
        self.channel_results_per_frequency_layer_result["all_freq"] = copy.deepcopy(self.resource_time_frequency_obj.dl_frequency_layer_info["all_freq"])
        self.channel_results_per_frequency_layer_result["all_freq"].update({
            "available_PRBs": 1,  # LoS typically considers one PRB
            "dl_channel_gain_ueAnt_to_cellAnt_complex": dl_LoS_channel_gain_ueAnt_to_cellAnt_complex,
        })            

        # Log elapsed time
        log_calculations_time('LoS Channel', t_start)
        
        return rescheduling_us


    def los_channel_gain_b_to_a(
        self,
        slow_channel_gain_bAnt_to_aAnt_dB: np.ndarray,
        array_steering_vector_bAnt_to_aAnt_complex: np.ndarray,
        k_factor_bAnt_to_aAnt_dB: np.ndarray,
    ) -> torch.Tensor:
        """
        Calculates the LoS channel gain for the provided parameters.

        Args:
            slow_channel_gain_bAnt_to_aAnt_dB: Slow channel gains in dB.
            array_steering_vector_bAnt_to_aAnt_complex: Steering vectors.
            k_factor_bAnt_to_aAnt_dB: K-factor values in dB.

        Returns:
            A tensor representing the calculated LoS channel gain.
        """
        # Convert slow channel gain and K-factor from dB to linear scale
        slow_channel_gain_linear: np.ndarray = tools.dBm_to_mW(slow_channel_gain_bAnt_to_aAnt_dB)
        K_factor_linear: np.ndarray = tools.dBm_to_mW(k_factor_bAnt_to_aAnt_dB)
    
        # Compute the square root of the K-factor fraction
        sqrt_K_factor: torch.Tensor = \
            torch.sqrt(torch.tensor(K_factor_linear / (1 + K_factor_linear), device=self.device, dtype=torch.half))
    
        # Compute LoS coefficients using the steering vectors and K-factor
        LoS_coefficients: torch.Tensor = \
            sqrt_K_factor * torch.tensor(array_steering_vector_bAnt_to_aAnt_complex, device=self.device, dtype=torch.chalf)
    
        # Compute the final LoS channel gain
        LoS_channel_gain: torch.Tensor = \
            torch.sqrt(torch.tensor(slow_channel_gain_linear, device=self.device)) * LoS_coefficients
    
        return LoS_channel_gain


class Channel(Saveable):
    """
    A class for simulating and processing complete channel models, including Line-of-Sight (LoS)
    and fast fading components, for multiple frequency layers and antenna configurations.
    """

    def __init__(
        self,
        simulation_config_obj: Any,
        network_deployment_obj: Any,
        resource_time_frequency_obj: Any,
        cell_antenna_array_structure_obj: Any,
        ue_antenna_array_structure_obj: Any,
        slow_channel_gain_ue_to_cell_obj: Any,
        dl_fast_fading_gain_ueAnt_to_cellAnt_obj: Any,
        channel_sn_obj: Any,
    ):
        """
        Initializes the Channel class with configuration and deployment objects.

        Args:
            simulation_config_obj: Configuration for the simulation environment.
            network_deployment_obj: Network deployment details (e.g., base station configurations).
            resource_time_frequency_obj: Resource allocation for time-frequency grids.
            cell_antenna_array_structure_obj: Details of cell antenna structures.
            ue_antenna_array_structure_obj: Details of user equipment (UE) antenna structures.
            slow_channel_gain_ue_to_cell_obj: Object representing slow channel gain between UE and cell.
            dl_fast_fading_gain_ueAnt_to_cellAnt_obj: Object representing fast fading gain coefficients.
            channel_sn_obj: Object representing subcarrier noise (SN) channel model and assignments.
        """

        super().__init__()

        # Device setup
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Input parameters
        self.simulation_config_obj: Any = simulation_config_obj
        self.network_deployment_obj: Any = network_deployment_obj
        self.resource_time_frequency_obj: Any = resource_time_frequency_obj
        self.cell_antenna_array_structure_obj: Any = cell_antenna_array_structure_obj
        self.ue_antenna_array_structure_obj: Any = ue_antenna_array_structure_obj
        self.slow_channel_gain_ue_to_cell_obj: Any = slow_channel_gain_ue_to_cell_obj
        self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj: Any = dl_fast_fading_gain_ueAnt_to_cellAnt_obj
        self.channel_sn_obj: Any = channel_sn_obj

        # Output results for each frequency layer
        self.channel_results_per_frequency_layer_result: Dict[float, Dict[str, Any]] = {}


    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["channel_results_per_frequency_layer_result"]


    def process(self, rescheduling_us: int = -1) -> int:
        """
        Processes the channel calculations and updates class outputs.

        Args:
            rescheduling_us: Rescheduling time in microseconds. Defaults to -1.

        Returns:
            The updated rescheduling time.
        """
        # Set random seed for reproducibility
        torch.manual_seed(self.simulation_config_obj.random_seed)

        # Extract relevant input data
        self.fading_results_per_frequency_layer_result: Dict[float, Dict[str, Any]] = (
            self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj.fading_results_per_frequency_layer_result
        )
        
        self.bs_fast_channel_models: np.ndarray = self.network_deployment_obj.df_ep["BS_fast_channel_model"].to_numpy()
        self.antenna_to_node_a_mapping: np.ndarray = self.cell_antenna_array_structure_obj.antenna_to_node_mapping
        self.antenna_to_node_b_mapping: np.ndarray = self.ue_antenna_array_structure_obj.antenna_to_node_mapping
        self.slow_channel_gain_b_to_a_dB: np.ndarray = \
            self.slow_channel_gain_ue_to_cell_obj.slow_channel_results_per_frequency_layer["all_freq"]["slow_channel_gain_b_to_a_dB"]
        

        # Start timer for performance tracking
        t_start: float = time.perf_counter()

        # Expand matrices for UE and cell antennas
        values_ue_ant, counts_ue_ant = np.unique(self.antenna_to_node_b_mapping, return_counts=True)
        slow_channel_gain_bAnt_to_a_dB = np.repeat(self.slow_channel_gain_b_to_a_dB, counts_ue_ant, axis=0)

        values_cell_ant, counts_cell_ant = np.unique(self.antenna_to_node_a_mapping, return_counts=True)
        self.slow_channel_gain_bAnt_to_aAnt_dB: np.ndarray = np.repeat(slow_channel_gain_bAnt_to_a_dB, counts_cell_ant, axis=1)

        # Process each frequency layer
        for dl_carrier_frequency_GHz, fading_result in self.fading_results_per_frequency_layer_result.items():
            # Extract relevant data for the current frequency layer
            fast_fading_channel_coeff_bAnt_to_aAnt_complex: np.ndarray = fading_result["fast_fading_channel_coeff_bAnt_to_aAnt_complex"]
            
            cells_in_frequency_mask: np.ndarray = fading_result["cells_in_frequency_mask"]
            cell_ants_in_frequency_mask: np.ndarray = fading_result["cell_ants_in_frequency_mask"]

            ue_ants_in_frequency_mask: np.ndarray = fading_result["ue_ants_in_frequency_mask"]

            # Filter data for selected cells in frequency layer
            slow_channel_gain_bAnt_to_aAnt_dB_filtered: np.ndarray = \
                self.slow_channel_gain_bAnt_to_aAnt_dB[ue_ants_in_frequency_mask, :][:, cell_ants_in_frequency_mask]

            # Initialize output tensor for channel gains
            shape = fast_fading_channel_coeff_bAnt_to_aAnt_complex.shape
            dl_channel_gain_ueAnt_to_cellAnt_complex = torch.zeros(shape, dtype=torch.cfloat, device=self.device)

            # Process each propagation model independently
            for bs_fast_channel_model in np.unique(self.bs_fast_channel_models):
                mask_cell: np.ndarray = self.bs_fast_channel_models[cells_in_frequency_mask] == bs_fast_channel_model
                mask_cell_antennas: np.ndarray = np.repeat(mask_cell, counts_cell_ant[cells_in_frequency_mask], axis=0)

                # Filter data for selected cells with propagation model
                slow_gain_filtered: np.ndarray = slow_channel_gain_bAnt_to_aAnt_dB_filtered[:, mask_cell_antennas]
                fast_fading_filtered: np.ndarray|torch.Tensor = fast_fading_channel_coeff_bAnt_to_aAnt_complex[:, :, mask_cell_antennas]
                
                # Get the shape of the fast fading channel coefficients to determine output dimensions
                shape: tuple = fast_fading_filtered.shape

                # Calculate channel gains
                if bs_fast_channel_model in ["3GPPTR38_901_UMa", "3GPPTR38_901_UMi", "Ray_tracing"]:
                    dl_channel_gain_ueAnt_to_cellAnt_complex[:, :, mask_cell_antennas] = \
                        self.sn_channel_gain_b_to_a(shape, counts_ue_ant, counts_cell_ant, self.resource_time_frequency_obj, self.channel_sn_obj)
                else:
                    if not self.simulation_config_obj.debug_no_randomness:
                        dl_channel_gain_ueAnt_to_cellAnt_complex[:, :, mask_cell_antennas] = \
                            self.channel_gain_b_to_a(slow_gain_filtered, fast_fading_filtered)
                    else:
                        dl_channel_gain_ueAnt_to_cellAnt_complex[:, :, mask_cell_antennas] = \
                            self.channel_gain_b_to_a(slow_gain_filtered, torch.ones_like(fast_fading_filtered, device=self.device))

            # Store results
            self.channel_results_per_frequency_layer_result[dl_carrier_frequency_GHz] = copy.deepcopy(fading_result)
            del self.channel_results_per_frequency_layer_result[dl_carrier_frequency_GHz]["fast_fading_channel_coeff_bAnt_to_aAnt_complex"]
            self.channel_results_per_frequency_layer_result[dl_carrier_frequency_GHz].update({
                "dl_channel_gain_ueAnt_to_cellAnt_complex": dl_channel_gain_ueAnt_to_cellAnt_complex,
            })

        # Log elapsed time
        log_calculations_time('Channel', t_start)

        return rescheduling_us


    def channel_gain_b_to_a(
        self,
        slow_channel_gain_bAnt_to_aAnt_dB: np.ndarray,
        fast_fading_channel_coeff_bAnt_to_aAnt_complex: np.ndarray|torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates the channel gain for the provided parameters.

        Args:
            slow_channel_gain_bAnt_to_aAnt_dB: Slow channel gains in dB.
            fast_fading_channel_coeff_bAnt_to_aAnt_complex: Fast fading coefficients.

        Returns:
            A tensor representing the calculated channel gain.
        """
            
        # Convert slow channel gain from dB to linear scale
        slow_channel_gain_linear: np.ndarray = tools.dBm_to_mW(slow_channel_gain_bAnt_to_aAnt_dB)
    
        # Compute the channel gain by combining the slow channel gain and fast fading coefficients
        channel_gain: torch.Tensor = torch.sqrt(torch.tensor(slow_channel_gain_linear, device=self.device, dtype=torch.cfloat)) * \
            torch.tensor(fast_fading_channel_coeff_bAnt_to_aAnt_complex, device=self.device, dtype=torch.cfloat)

        return channel_gain
    

    def sn_channel_gain_b_to_a(
        self,
        shape: tuple[int, int, int],
        counts_ue: np.ndarray,
        counts_cell: np.ndarray,
        resource_grid_obj: object,
        channel_sn_obj: object,
    ) -> torch.Tensor:
        """
        Computes the SN channel gain from transmitter (b) to receiver (a) based on OFDM channel models.
        This function supports mapping channel gains to a specified tensor output.
    
        Args:
            shape: The shape of the output tensor [batch, num_rx_ant, num_tx_ant].
            counts_ue: An array indicating the number of UEs per receiver.
            counts_cell: An array indicating the number of antennas per cell.
            resource_grid_obj: The resource grid object containing configuration details.
            channel_sn_obj: An object containing OFDM channel models and their assignments.
    
        Returns:
            A torch tensor of complex type with the specified shape, containing channel gains.
        """
        # Initialize the output tensor on the specified device
        output: torch.Tensor = torch.zeros(shape, dtype=torch.cfloat, device=self.device)
    
        # Iterate through all OFDM channels in the SN model
        for ofdm_channel_index in range(len(channel_sn_obj.ofdm_channels)):
            # Retrieve the current OFDM channel model
            ofdm_channel = channel_sn_obj.ofdm_channels[ofdm_channel_index]
    
            # Retrieve the assignment of the current channel model
            ofdm_channel_assignment_b_to_a: np.ndarray = channel_sn_obj.ofdm_channel_assignment_b_to_a == ofdm_channel_index
    
            # Expand the assignment matrix from UE size to UE antenna size
            ofdm_channel_assignment_bAnt_to_a: np.ndarray = np.repeat(ofdm_channel_assignment_b_to_a, counts_ue, axis=0)
    
            # Further expand the assignment matrix to include cell antennas
            ofdm_channel_assignment_bAnt_to_aAnt: np.ndarray = np.repeat(ofdm_channel_assignment_bAnt_to_a, counts_cell, axis=1)
    
            # Generate a batch of frequency responses
            # Shape: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers]
            with tf.device("/GPU:0"):  # Force CPU usage due to GPU memory constraints
                if callable(ofdm_channel):
                    # If the channel model is callable, generate the frequency response
                    h_freq = ofdm_channel()
                else:
                    # If the channel model is a tensor, use it directly
                    h_freq = ofdm_channel
    
            # Process the frequency response tensor
            h_freq = tf.squeeze(h_freq, axis=(0, 5))  # Remove unnecessary dimensions
            h_freq = tf.reduce_sum(h_freq, axis=1)  # Sum across OFDM symbols
            h_freq = tf.reshape(
                h_freq,
                [
                    np.size(h_freq, 0),
                    np.size(h_freq, 1) * np.size(h_freq, 2),
                    np.size(h_freq, 3),
                ],
            )
            h_freq = tf.transpose(h_freq, perm=[2, 0, 1])  # Permute axes for alignment
            h_freq_np = h_freq.numpy()  # Convert TensorFlow tensor to numpy array
    
            # Create a mask to identify insertion positions
            mask: np.ndarray = np.zeros_like(output.cpu().numpy(), dtype=bool)
            indices = np.where(ofdm_channel_assignment_bAnt_to_aAnt)
            mask[:, indices[0], indices[1]] = True
    
            # Insert frequency response values into the output tensor
            output_np: np.ndarray = output.cpu().numpy()  # Convert output to numpy
            output_np[mask] = h_freq_np.reshape(-1)  # Map h_freq values to output
            output = torch.tensor(output_np, dtype=torch.cfloat, device=self.device)  # Convert back to PyTorch
    
        return output

             
class ChannelSn(Saveable):
    
    def __init__(self, 
                 plot,
                 simulation_config_obj,
                 site_deployment_obj,
                 network_deployment_obj,
                 ue_deployment_obj,
                 resource_time_frequency_obj,
                 cell_antenna_array_structure_obj,
                 ue_antenna_array_structure_obj):
        
        super().__init__()
        
        ##### Plots 
        ########################
        
        self.plot = plot # Switch to control plots if any


        ##### Input storage 
        ######################## 
        self.site_deployment_obj = site_deployment_obj
        self.simulation_config_obj = simulation_config_obj
        self.network_deployment_obj = network_deployment_obj
        self.ue_deployment_obj = ue_deployment_obj   
        self.resource_time_frequency_obj = resource_time_frequency_obj
        self.cell_antenna_array_structure_obj = cell_antenna_array_structure_obj
        self.ue_antenna_array_structure_obj = ue_antenna_array_structure_obj        


        ##### Outputs 
        ######################## 
        self.ofdm_channels = []
        self.ofdm_channel_assignment_b_to_a = []
        
        
    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["ofdm_channels"]

       
    def process(self, rescheduling_us=-1): 
        
        ##### Process inputs
        ######################## 
        
        # Random numbers
        tf.random.set_seed(self.simulation_config_obj.random_seed+0)      
        
        # Network
        self.bs_fast_channel_models = self.network_deployment_obj.df_ep["BS_fast_channel_model"].to_numpy()  
        
        
        ##### Process outputs
        ########################
        self.ofdm_channel_assignment_b_to_a = np.ones((len(self.ue_deployment_obj.df_ep), len(self.network_deployment_obj.df_ep)), dtype=int) * np.nan
       
        
        ##### Start timer
        ########################  
        t_start = time.perf_counter() 
                
        
        ##### Switch
        ########################        
        
        # Find the set of unique propagation models to process them independently
        bs_fast_channel_model_set = set(self.bs_fast_channel_models)

        self.ofdm_channel_assignment_b_to_a = np.ones(
            (len(self.ue_deployment_obj.df_ep), len(self.network_deployment_obj.df_ep)), dtype=int) * np.nan

        # Process each propagation model independently
        for bs_fast_channel_model in bs_fast_channel_model_set:
            
            # Identify cells with the selected propagation model
            mask = bs_fast_channel_model ==  self.bs_fast_channel_models 
            
            # Calculate channel
            if (bs_fast_channel_model == "3GPPTR38_901_UMa"): 
                
                self.ofdm_channels, self.ofdm_channel_assignment_b_to_a =\
                    self.channel_3GPPTR38_901_UMa(self.simulation_config_obj,
                                                  self.network_deployment_obj.df_ep,
                                                  self.ue_deployment_obj.df_ep,
                                                  self.resource_time_frequency_obj,
                                                  self.cell_antenna_array_structure_obj,
                                                  self.ue_antenna_array_structure_obj,
                                                  mask,
                                                  self.ofdm_channels,
                                                  self.ofdm_channel_assignment_b_to_a)  
                
            elif (bs_fast_channel_model == "3GPPTR38_901_UMi"):
                
                self.ofdm_channels, self.ofdm_channel_assignment_b_to_a =\
                    self.channel_3GPPTR38_901_UMi(self.simulation_config_obj,
                                                  self.network_deployment_obj.df_ep,
                                                  self.ue_deployment_obj.df_ep,
                                                  self.resource_time_frequency_obj,
                                                  self.cell_antenna_array_structure_obj,
                                                  self.ue_antenna_array_structure_obj,
                                                  mask,
                                                  self.ofdm_channels,
                                                  self.ofdm_channel_assignment_b_to_a)     

            elif (bs_fast_channel_model == "Ray_tracing"):

                self.ofdm_channels, self.ofdm_channel_assignment_b_to_a =\
                    self.channel_ray_tracing(self.site_deployment_obj,
                                             self.simulation_config_obj,
                                             self.network_deployment_obj.df_ep,
                                             self.ue_deployment_obj.df_ep,
                                             self.resource_time_frequency_obj,
                                             self.cell_antenna_array_structure_obj,
                                             self.ue_antenna_array_structure_obj,
                                             mask,
                                             self.ofdm_channels,
                                             self.ofdm_channel_assignment_b_to_a)


        ##### End
        ########################
        log_calculations_time('Sionna', t_start)

        return rescheduling_us
                                                
    
    def channel_3GPPTR38_901_UMa(self, 
                                 simulation_config_obj,
                                 df_ep,
                                 df_ue_ep,
                                 resrouce_grid_obj,
                                 cell_antenna_array_structure_obj,
                                 ue_antenna_array_structure_obj,
                                 mask,
                                 ofdm_channels,
                                 ofdm_channel_assignment_b_to_a):
                                                           
        # Get necessary inputs 
                    
            # Get cell information 
        cell_positions_m = self.network_deployment_obj.df_ep[["position_x_m", "position_y_m", "position_z_m"]].to_numpy(dtype=np.single)[mask]
        cell_orientations_rad = np.radians(df_ep[["antenna_config_hor_alpha_mec_bearing_deg", 
                                                  "antenna_config_ver_beta_mec_downtilt_deg", 
                                                  "antenna_config_gamma_mec_slant_deg"]].to_numpy(dtype=np.single))[mask]
        cell_orientations_rad[:,1] -= np.pi/2 # Note that Sionna uses 0 as the horizon. We use 0 as the sky
            
            # Get UE information
        ue_positions_m = df_ue_ep[["position_x_m", "position_y_m", "position_z_m"]].to_numpy(dtype=np.single)
        ue_orientations_rad = np.radians(df_ue_ep[["antenna_config_hor_alpha_mec_bearing_deg", 
                                                   "antenna_config_ver_beta_mec_downtilt_deg", 
                                                   "antenna_config_gamma_mec_slant_deg"]].to_numpy(dtype=np.single))
        ue_orientations_rad[:,1] -= np.pi/2 # Note that Sionna uses 0 as the horizon. We use 0 as the sky
        ue_velocity_ms =  tools.kmh_to_ms(df_ue_ep[["velocity_x_kmh", "velocity_y_kmh", "velocity_z_kmh"]].to_numpy(dtype=np.single))
        ue_indoor = df_ue_ep["indoor"].to_numpy(dtype=bool)     
        
            # Get frequencies, and antenna arrays
        dl_carrier_frequency_GHz = cell_antenna_array_structure_obj.antenna_array_carrier_frequency_GHz
        cell_antenna_array_type_index = cell_antenna_array_structure_obj.antenna_array_type_index[mask]
        ue_antenna_array_type_index = ue_antenna_array_structure_obj.antenna_array_type_index   
        dl_resource_grid_index = resrouce_grid_obj.dl_resource_grid_index
        
        # Get uniques 
        
        cell_antenna_array_type_index_set = set(cell_antenna_array_type_index)
        ue_antenna_array_type_index_set = set(ue_antenna_array_type_index)   
        dl_resource_grid_index_set = set(dl_resource_grid_index)              
        
        # Iterate over unique combinations
        
        for model_params in itertools.product(cell_antenna_array_type_index_set, ue_antenna_array_type_index_set, dl_resource_grid_index_set):  #carrier_frequency_GHz_set
            
            # Derive masks for UEs and cells 
            ue_antenna_array_type_mask = ue_antenna_array_type_index == model_params[1]  
            cell_antenna_array_type_mask = cell_antenna_array_type_index == model_params[0]
            ue_to_cell_mask = np.logical_and(ue_antenna_array_type_mask[:, np.newaxis], cell_antenna_array_type_mask)   

            # Create UMa channel model
            channel_model = sionna.phy.channel.tr38901.UMa(carrier_frequency = dl_carrier_frequency_GHz[model_params[0]] * 1e9,
                                                       o2i_model = 'low',
                                                       ut_array = ue_antenna_array_structure_obj.antenna_array_types[model_params[1]],
                                                       bs_array = cell_antenna_array_structure_obj.antenna_array_types[model_params[0]],
                                                       direction = simulation_config_obj.link_direction)  
        
            # Set topology in channel model
            channel_model.set_topology(ut_loc = tf.convert_to_tensor(ue_positions_m[np.newaxis,ue_antenna_array_type_mask,:]),
                                       bs_loc = tf.convert_to_tensor(cell_positions_m[np.newaxis,cell_antenna_array_type_mask,:]),
                                       ut_orientations = tf.convert_to_tensor(ue_orientations_rad[np.newaxis,ue_antenna_array_type_mask,:]),
                                       bs_orientations = tf.convert_to_tensor(cell_orientations_rad[np.newaxis,cell_antenna_array_type_mask,:]),
                                       ut_velocities = tf.convert_to_tensor(ue_velocity_ms[np.newaxis,ue_antenna_array_type_mask,:]),
                                       in_state = tf.convert_to_tensor(ue_indoor[np.newaxis,ue_antenna_array_type_mask]))
            
            # Get SN resource grid 
            
            rg = resrouce_grid_obj.dl_resource_grid[model_params[2]]
        
            # Instanting the SN frequency domain channel
            ofdm_channel = sionna.phy.channel.generate_ofdm_channel.GenerateOFDMChannel(channel_model, rg)
        
            # Store channel models and asignments
            ofdm_channels.append(ofdm_channel)
            ofdm_channel_assignment_b_to_a[ue_to_cell_mask] = len(ofdm_channels)-1            
        
        return ofdm_channels, ofdm_channel_assignment_b_to_a
    

    def channel_3GPPTR38_901_UMi(self, 
                                 simulation_config_obj,
                                 df_ep,
                                 df_ue_ep,
                                 resrouce_grid_obj,
                                 cell_antenna_array_structure_obj,
                                 ue_antenna_array_structure_obj,
                                 mask,
                                 ofdm_channels,
                                 ofdm_channel_assignment_b_to_a):
                                                           
        # Get necessary inputs 
                    
            # Get cell information 
        cell_positions_m = self.network_deployment_obj.df_ep[["position_x_m", "position_y_m", "position_z_m"]].to_numpy(dtype=np.single)[mask]
        cell_orientations_rad = np.radians(df_ep[["antenna_config_hor_alpha_mec_bearing_deg", 
                                                  "antenna_config_ver_beta_mec_downtilt_deg", 
                                                  "antenna_config_gamma_mec_slant_deg"]].to_numpy(dtype=np.single))[mask]
        cell_orientations_rad[:,1] -= np.pi/2 # Note that Sionna uses 0 as the horizon. We use 0 as the sky        
            
            # Get UE information
        ue_positions_m = df_ue_ep[["position_x_m", "position_y_m", "position_z_m"]].to_numpy(dtype=np.single)
        ue_orientations_rad = np.radians(df_ue_ep[["antenna_config_hor_alpha_mec_bearing_deg", 
                                                   "antenna_config_ver_beta_mec_downtilt_deg", 
                                                   "antenna_config_gamma_mec_slant_deg"]].to_numpy(dtype=np.single))
        ue_orientations_rad[:,1] -= np.pi/2 # Note that Sionna uses 0 as the horizon. We use 0 as the sky
        ue_velocity_ms =  tools.kmh_to_ms(df_ue_ep[["velocity_x_kmh","velocity_y_kmh", "velocity_z_kmh"]].to_numpy(dtype=np.single))
        ue_indoor = df_ue_ep["indoor"].to_numpy(dtype=bool)     
        
            # Get frequencies, and antenna arrays
        dl_carrier_frequency_GHz = cell_antenna_array_structure_obj.antenna_array_carrier_frequency_GHz
        cell_antenna_array_type_index = cell_antenna_array_structure_obj.antenna_array_type_index[mask]
        ue_antenna_array_type_index = ue_antenna_array_structure_obj.antenna_array_type_index   
        dl_resource_grid_index = resrouce_grid_obj.dl_resource_grid_index
        
        # Get uniques 
        
        cell_antenna_array_type_index_set = set(cell_antenna_array_type_index)
        ue_antenna_array_type_index_set = set(ue_antenna_array_type_index)   
        dl_resource_grid_index_set = set(dl_resource_grid_index)              
        
        # Iterate over unique combinations
        
        for model_params in itertools.product(cell_antenna_array_type_index_set, ue_antenna_array_type_index_set, dl_resource_grid_index_set):  #carrier_frequency_GHz_set
            
            # Derive masks for UEs and cells 
            ue_antenna_array_type_mask = ue_antenna_array_type_index == model_params[1]  
            cell_antenna_array_type_mask = cell_antenna_array_type_index == model_params[0]
            ue_to_cell_mask = np.logical_and(ue_antenna_array_type_mask[:, np.newaxis], cell_antenna_array_type_mask)   

            # Create UMi channel model
            channel_model = sionna.phy.channel.tr38901.UMi(carrier_frequency = dl_carrier_frequency_GHz[model_params[0]] * 1e9,
                                                       o2i_model = 'low',
                                                       ut_array = ue_antenna_array_structure_obj.antenna_array_types[model_params[1]],
                                                       bs_array = cell_antenna_array_structure_obj.antenna_array_types[model_params[0]],
                                                       direction = simulation_config_obj.link_direction)  
        
            # Set topology in channel model
            channel_model.set_topology(ut_loc = tf.convert_to_tensor(ue_positions_m[np.newaxis,ue_antenna_array_type_mask,:]),
                                       bs_loc = tf.convert_to_tensor(cell_positions_m[np.newaxis,cell_antenna_array_type_mask,:]),
                                       ut_orientations = tf.convert_to_tensor(ue_orientations_rad[np.newaxis,ue_antenna_array_type_mask,:]),
                                       bs_orientations = tf.convert_to_tensor(cell_orientations_rad[np.newaxis,cell_antenna_array_type_mask,:]),
                                       ut_velocities = tf.convert_to_tensor(ue_velocity_ms[np.newaxis,ue_antenna_array_type_mask,:]),
                                       in_state = tf.convert_to_tensor(ue_indoor[np.newaxis,ue_antenna_array_type_mask]))
            
            # Get SN resource grid 
            rg = resrouce_grid_obj.dl_resource_grid[model_params[2]]
        
            # Instanting the SN frequency domain channel
            ofdm_channel = sionna.phy.channel.generate_ofdm_channel.GenerateOFDMChannel(channel_model, rg)
        
            # Store channel models and asignments
            ofdm_channels.append(ofdm_channel)
            ofdm_channel_assignment_b_to_a[ue_to_cell_mask] = len(ofdm_channels)-1            
        
        return ofdm_channels, ofdm_channel_assignment_b_to_a    


    def channel_ray_tracing(self,
                            site_deployment_obj,
                            simulation_config_obj,
                            df_ep,
                            df_ue_ep,
                            resrouce_grid_obj,
                            cell_antenna_array_structure_obj,
                            ue_antenna_array_structure_obj,
                            mask,
                            ofdm_channels,
                            ofdm_channel_assignment_b_to_a):

        #############################################################
        # Load scene in Mitsuba format
        scene = site_deployment_obj.scene
        scene_save_folder = site_deployment_obj.dataset_export_folder
        scene_name = site_deployment_obj.scene_name

        #############################################################
        # Remove existing transmitters and receivers
        num_transmitters = len(scene.transmitters)
        num_receivers = len(scene.receivers)

        # Remove all transmitters
        for i in range(num_transmitters):
            scene.remove(f'tx_{i}')

        # Remove all receivers
        for i in range(num_receivers):
            scene.remove(f'rx_{i}')

        #############################################################
        # Set transmitters and receivers
        min_lat = site_deployment_obj.scene_min_lat
        max_lat = site_deployment_obj.scene_max_lat
        min_long = site_deployment_obj.scene_min_long
        max_long = site_deployment_obj.scene_max_long

        min_easting, min_northing = tools.latlon_to_utm(min_lat, min_long)
        max_easting, max_northing = tools.latlon_to_utm(max_lat, max_long)
        scene_center_easting = (min_easting + max_easting) / 2
        scene_center_northing = (min_northing + max_northing) / 2

        # Get cell information
        cell_positions_m = self.network_deployment_obj.df_ep[["position_x_m", "position_y_m", "position_z_m"]].to_numpy(dtype=np.single)[mask]
        cell_orientations_rad = np.radians(df_ep[["antenna_config_hor_alpha_mec_bearing_deg",
                                                  #"antenna_config_ver_beta_mec_downtilt_deg",
                                                  "antenna_config_ver_beta_elec_downtilt_deg",
                                                  "antenna_config_gamma_mec_slant_deg"]].to_numpy(dtype=np.single))[mask]
        cell_orientations_rad -= [0, np.pi/2, 0] # Note that Sionna uses 0 as the horizon. We use 0 as the sky.
        # Get UE information
        ue_positions_m = df_ue_ep[["position_x_m", "position_y_m", "position_z_m"]].to_numpy(dtype=np.single)
        ue_orientations_rad = np.radians(df_ue_ep[["antenna_config_hor_alpha_mec_bearing_deg",
                                                   "antenna_config_ver_beta_mec_downtilt_deg",
                                                   "antenna_config_gamma_mec_slant_deg"]].to_numpy(dtype=np.single))

        ue_velocity_ms = tools.kmh_to_ms(df_ue_ep[["velocity_x_kmh", "velocity_y_kmh", "velocity_z_kmh"]].to_numpy(dtype=np.single))
        ue_indoor = df_ue_ep["indoor"].to_numpy(dtype=bool)

        # Get frequencies, and antenna arrays
        dl_carrier_frequency_GHz = cell_antenna_array_structure_obj.antenna_array_carrier_frequency_GHz
        cell_antenna_array_type_index = cell_antenna_array_structure_obj.antenna_array_type_index[mask]
        ue_antenna_array_type_index = ue_antenna_array_structure_obj.antenna_array_type_index
        dl_resource_grid_index = resrouce_grid_obj.dl_resource_grid_index

        # Get uniques
        cell_antenna_array_type_index_set = set(cell_antenna_array_type_index)
        ue_antenna_array_type_index_set = set(ue_antenna_array_type_index)
        dl_resource_grid_index_set = set(dl_resource_grid_index)

        # Iterate over unique combinations
        for model_params in itertools.product(cell_antenna_array_type_index_set,   #0
                                            ue_antenna_array_type_index_set,   #1
                                            dl_resource_grid_index_set):       #2
                                            # carrier_frequency_GHz_set

            # Derive masks for UEs and cells
            ue_antenna_array_type_mask = ue_antenna_array_type_index == model_params[1]
            cell_antenna_array_type_mask = cell_antenna_array_type_index == model_params[0]
            ue_to_cell_mask = np.logical_and(ue_antenna_array_type_mask[:, np.newaxis], cell_antenna_array_type_mask)

            # Configure tx/rx antenna array (must be the same for all tx/rx)
            scene.tx_array = cell_antenna_array_structure_obj.antenna_array_types[model_params[0]]
            scene.rx_array = ue_antenna_array_structure_obj.antenna_array_types[model_params[1]]

            scene.synthetic_array = True  # If set to False, ray tracing will be done per antenna element (slower for large arrays)

            for index, (x, y, z) in enumerate(cell_positions_m):
                # Adjust positions relative to the scene center
                adjusted_x = x - scene_center_easting
                adjusted_y = y - scene_center_northing
                adjusted_z = z  # Z coordinate remains unchanged

                # Create transmitter
                tx = Transmitter(name=f"tx_{index}", position=[adjusted_x, adjusted_y, adjusted_z], orientation=cell_orientations_rad[index])
                # Note that with sionna.rt.PlanarArray, the antennas are regularly spaced, located in the y-z plane.
                # I.e., unlike 3GPP, in Sionna the 0 deg elevation points to the horizon.

                # Add transmitter to scene
                scene.add(tx)

            for index, (x, y, z) in enumerate(ue_positions_m):
                # Adjust positions relative to the scene center
                adjusted_x = x - scene_center_easting
                adjusted_y = y - scene_center_northing
                adjusted_z = z  # Z coordinate remains unchanged

                # Create receiver
                rx = Receiver(name=f"rx_{index}", position=[adjusted_x, adjusted_y, adjusted_z], orientation=ue_orientations_rad[index])

                # Add receiver to scene
                scene.add(rx)

            #############################################################
            # Save images of the scene with all tx and rx
            resolution_for_rendering = [720, 720]   #[1920, 1080]    [720, 720]   [480, 320]  # increase for higher quality of renderings
            render_to_file = True  # Set to True to render image to file
            num_samples_for_rendering = 64  # Enough: 256.    Set to 64 to speed this up

            # Use bird's eye view camera and save image
            if render_to_file:
                scene_image_file = "{}_birds_view.png".format(scene_name)
                scene_image_file_path = os.path.join(scene_save_folder, scene_image_file)
                scene.render_to_file(camera="birds_view", 
                                     filename=scene_image_file_path,
                                     resolution=resolution_for_rendering,
                                     num_samples=num_samples_for_rendering)

            # Use corner camera and save image
            if render_to_file:
                scene_image_file = "{}_corner_view.png".format(scene_name)
                scene_image_file_path = os.path.join(scene_save_folder, scene_image_file)
                scene.render_to_file(camera="corner_view",
                                     filename=scene_image_file_path,
                                     resolution=resolution_for_rendering,
                                     num_samples=num_samples_for_rendering)

            #############################################################
            # Compute paths     (scene frequency already set in scenarios.py)

            # Compute propagation paths
            paths = scene.compute_paths(max_depth=3,  # number of bounces
                                        num_samples=1e6,
                                        los=True,
                                        reflection=True,
                                        diffraction=True,
                                        scattering=False,
                                        check_scene=True)
            # "num_samples" is the #rays shot into directions defined by a Fibonacci sphere
            # note that too few rays can lead to missing paths

            #############################################################
            # Save the paths as an OBJ file for visualization in Blender

            # To see it in Blender: Go to File > Import > Wavefront (.obj).
            # Important: When importing int Blender, select Y-forward Z-up.
            if render_to_file:
                scene_image_file = "{}_paths.obj".format(scene_name)
                scene_image_file_path = os.path.join(scene_save_folder,scene_image_file)
                paths.export(scene_image_file_path)

            #############################################################
            # Save images of the scene with all paths
            if render_to_file:
                scene_image_file = "{}_paths_birds_view.png".format(scene_name)
                scene_image_file_path = os.path.join(scene_save_folder, scene_image_file)
                scene.render_to_file(camera="birds_view",
                                     paths=paths,
                                     show_devices=True,
                                     show_paths=True,
                                     filename=scene_image_file_path,
                                     resolution=resolution_for_rendering,
                                     num_samples=num_samples_for_rendering)

            if render_to_file:
                scene_image_file = "{}_paths_corner_view.png".format(scene_name)
                scene_image_file_path = os.path.join(scene_save_folder, scene_image_file)
                scene.render_to_file(camera="corner_view",
                                     paths=paths,
                                     show_devices=True,
                                     show_paths=True,
                                     filename=scene_image_file_path,
                                     resolution=resolution_for_rendering,
                                     num_samples=num_samples_for_rendering)

            #############################################################
            # Compute channel response

            # Methods of the class path here:
            # https: // nvlabs.github.io / sionna / api / rt.html  # sionna.rt.Paths

            # paths.normalize_delays = False

            # To compute Doppler shift: currently not used. Channel is approx as constant in each PRB.
            # sampling_frequency = ...
            # num_time_steps = ...
            # paths.apply_doppler(sampling_frequency, num_time_steps, tx_velocities=(0.0, 0.0, 0.0), rx_velocities=(0.0, 0.0, 0.0))

            a, tau = paths.cir(los=True, reflection=True, diffraction=True, scattering=True, num_paths=None)

            # Get SN resource grid and create the frequency-domain channels
            rg = resrouce_grid_obj.dl_resource_grid[model_params[2]]
            num_subcarriers = rg.fft_size
            subcarrier_spacing = rg.subcarrier_spacing
            frequencies = sionna.phy.ofdm.subcarrier_frequencies(num_subcarriers, subcarrier_spacing, dtype=tf.complex64)
            ofdm_channel = sionna.phy.ofdm.cir_to_ofdm_channel(frequencies, a, tau, normalize=False)

            # Store channels
            ofdm_channels.append(ofdm_channel)
            ofdm_channel_assignment_b_to_a[ue_to_cell_mask] = len(ofdm_channels) - 1

        return ofdm_channels, ofdm_channel_assignment_b_to_a