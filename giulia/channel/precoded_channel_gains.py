# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:40:53 2023

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
from typing import Any, Dict, Optional, List

from giulia.event_driven.snapshot_control import Snapshot_control
from giulia.fs import results_file
from giulia.tools import tools
from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable


class Precoded_Channel_Gain(Saveable):
    """
    A class to compute precoded channel gains for multiple frequency layers and antenna configurations.
    Handles both Line-of-Sight (LoS) and non-LoS channel types, with optional steering vector usage.
    """

    def __init__(
        self,
        compute_type: str,
        simulation_config_obj: Any,
        network_deployment_obj: Any,
        cell_antenna_array_structure_obj: Any,
        beam_conf_obj: Any,
        ue_deployment_obj: Any,
        channel_gain_ue_to_cell_obj: Any,
        array_steering_vector_ue_to_cell_obj: Optional[Any] = None,
    ):
        """
        Initialize the Precoded_Channel_Gain class with simulation and deployment objects.

        Args:
            compute_type: Type of channel gain computation ('LoS_channel' or other).
            simulation_config_obj: Simulation environment configuration object.
            network_deployment_obj: Network deployment details (e.g., base stations, cells).
            cell_antenna_array_structure_obj: Antenna array structures for the cells.
            beam_conf_obj: Beam configuration object (e.g., codebooks, mappings).
            ue_deployment_obj: User equipment (UE) deployment details.
            channel_gain_ue_to_cell_obj: Object containing channel gain data for UEs and cells.
            array_steering_vector_ue_to_cell_obj: Optional steering vector object for array computations.
        """

        super().__init__()

        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Input parameters
        self.compute_type: str = compute_type
        self.simulation_config_obj: Any = simulation_config_obj
        self.network_deployment_obj: Any = network_deployment_obj
        self.cell_antenna_array_structure_obj: Any = cell_antenna_array_structure_obj
        self.beam_conf_obj: Any = beam_conf_obj
        self.ue_deployment_obj: Any = ue_deployment_obj
        self.channel_gain_ue_to_cell_obj: Any = channel_gain_ue_to_cell_obj
        self.array_steering_vector_ue_to_cell_obj: Optional[Any] = array_steering_vector_ue_to_cell_obj

        # Output storage
        self.precoded_channel_gain_results_per_frequency_layer: Dict[str, Dict[str, Any]] = {}


    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["precoded_channel_gain_results_per_frequency_layer"]
    

    def process(self, rescheduling_us: int = -1) -> int:
        """
        Compute precoded channel gains for each frequency layer.

        Args:
            rescheduling_us: Time (in microseconds) to reschedule. Defaults to -1.

        Returns:
            Updated rescheduling time.
        """
        # Determine beam type from configuration
        self.beam_type: str = "antenna element" if self.beam_conf_obj is None else self.beam_conf_obj.beam_type

        # Extract input data and pre-calculate key mappings
        self.channel_gain_results: Dict[str, Dict[str, Any]] = self.channel_gain_ue_to_cell_obj.channel_results_per_frequency_layer_result
        
        self.antenna_pattern_models: np.ndarray = self.network_deployment_obj.df_ep["antenna_pattern_model"].to_numpy()
        self.A_m_dB: np.ndarray = self.network_deployment_obj.df_ep["antenna_config_hor_A_m_dB"].to_numpy()
        self.antenna_to_node_a_mapping: np.ndarray = self.cell_antenna_array_structure_obj.antenna_to_node_mapping
        self.beam_to_node_mapping: np.ndarray = self.beam_conf_obj.beam_to_node_mapping
        self.codebook_index_to_node_mapping = self.beam_conf_obj.codebook_index_to_node_mapping
        
        # Start the timer for performance monitoring
        t_start: float = time.perf_counter()        
        
        # Count the number of cell antennas
        values_cell_ant, counts_cell_ant = np.unique(self.antenna_to_node_a_mapping, return_counts=True)

        # Process each frequency layer
        for frequency_key, frequency_layer_data in self.channel_gain_results.items():
            # Extract channel gain data for the current frequency layer
            dl_channel_gain_ueAnt_to_cellAnt_complex: np.ndarray = frequency_layer_data["dl_channel_gain_ueAnt_to_cellAnt_complex"]
            
            cell_IDs_in_frequency: np.ndarray = frequency_layer_data["cell_IDs"]
            cells_in_frequency_mask: np.ndarray = frequency_layer_data["cells_in_frequency_mask"]
            available_PRBs_for_frequency: int = frequency_layer_data["available_PRBs"]
            number_of_ues_in_frequency: int = frequency_layer_data["num_ues"]
            number_of_beams_in_frequency: int = np.sum(np.isin(self.beam_to_node_mapping, cell_IDs_in_frequency))
            
            # Filter data for selected cells in frequency layer
            A_m_dB: np.ndarray = self.A_m_dB[cells_in_frequency_mask]
            antenna_to_node_a_mapping: np.ndarray = self.antenna_to_node_a_mapping[np.isin(self.antenna_to_node_a_mapping, cell_IDs_in_frequency)]
            beam_to_node_mapping: np.ndarray = self.beam_to_node_mapping[np.isin(self.beam_to_node_mapping, cell_IDs_in_frequency)]
            codebook_index_to_node_mapping = self.codebook_index_to_node_mapping[cells_in_frequency_mask]

            # Initialize tensor for precoded channel gains
            precoded_channel_gain_b_to_a_dB: np.ndarray = np.zeros((available_PRBs_for_frequency, number_of_ues_in_frequency, number_of_beams_in_frequency),dtype=np.single)

            # Process data for each antenna pattern model
            for antenna_pattern_model in np.unique(self.antenna_pattern_models):
                mask_cell: np.ndarray = self.antenna_pattern_models[cells_in_frequency_mask] == antenna_pattern_model
                mask_cell_antennas: np.ndarray = np.repeat(mask_cell, counts_cell_ant[cells_in_frequency_mask], axis=0)
                selected_cell_IDs: np.ndarray = cell_IDs_in_frequency[mask_cell]
                
                # Filter data for selected cells with antenna pattern model
                dl_channel_gain_filtered: np.ndarray = dl_channel_gain_ueAnt_to_cellAnt_complex[:, :, mask_cell_antennas]
                A_m_dB_filtered: np.ndarray = A_m_dB[mask_cell]
                antenna_to_node_a_mapping_filtered: np.ndarray = antenna_to_node_a_mapping[np.isin(antenna_to_node_a_mapping, selected_cell_IDs)]
                beam_to_node_mapping_filtered: np.ndarray = beam_to_node_mapping[np.isin(beam_to_node_mapping, selected_cell_IDs)]

                codebook_index_to_node_mapping_filtered = codebook_index_to_node_mapping[mask_cell]

                # Calculate precoded channel gains
                if antenna_pattern_model in ["3GPPTR38_901", "Ray_tracing"]:
                    precoded_channel_gain_b_to_a_dB[:, :, np.isin(beam_to_node_mapping, selected_cell_IDs)] = \
                        self.calculate_precoded_channel_gains(
                            selected_cell_IDs,
                            self.beam_conf_obj.codebook_dictionary,
                            A_m_dB_filtered,
                            codebook_index_to_node_mapping_filtered,
                            beam_to_node_mapping_filtered,
                            antenna_to_node_a_mapping_filtered,
                            dl_channel_gain_filtered,
                        )
                else:
                    precoded_channel_gain_b_to_a_dB[:, :, np.isin(beam_to_node_mapping, selected_cell_IDs)] = \
                        tools.mW_to_dBm_torch(torch.pow(torch.abs(torch.tensor(dl_channel_gain_filtered, device=self.device)), 2)).cpu().numpy()

            # Store results for the frequency layer
            self.precoded_channel_gain_results_per_frequency_layer[frequency_key] = copy.deepcopy(frequency_layer_data)
            del self.precoded_channel_gain_results_per_frequency_layer[frequency_key]["dl_channel_gain_ueAnt_to_cellAnt_complex"]
            self.precoded_channel_gain_results_per_frequency_layer[frequency_key].update({
                "precoded_channel_gain_b_to_a_dB": precoded_channel_gain_b_to_a_dB,
            })

        # Save results for plotting if needed
        snapshot_control = Snapshot_control.get_instance()
        if self.simulation_config_obj.plot == 1 and snapshot_control.num_snapshots == 0:
            file_name: str = results_file(self.simulation_config_obj.project_name, f"to_plot_{self.beam_type}_precoded_channel_gain_{self.compute_type}",)
            np.savez(file_name, precoded_channel_gain_results=self.precoded_channel_gain_results_per_frequency_layer)
        
        # Log elapsed time
        log_calculations_time(f'{self.beam_type} precoded channel', t_start)
        
        return rescheduling_us
    
    
    def calculate_precoded_channel_gains(
        self,
        selected_cell_IDs: np.ndarray,
        codebook_dictionary: Dict[Any, np.ndarray],
        A_m_dB: np.ndarray,
        codebook_indexing: np.ndarray,
        beam_to_node_mapping: np.ndarray,
        antenna_to_node_a_mapping: np.ndarray,
        channel_gain_ueAnt_to_cellAnt_complex: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the precoded channel gains for multiple cells and beams using torch for GPU acceleration.
    
        Args:
            selected_cell_IDs (np.ndarray): IDs of selected cells.
            codebook_dictionary (Dict[Any, np.ndarray]): Dictionary of precoding codebooks per cell.
            A_m_dB (np.ndarray): Per-cell array gain values in dB (currently unused).
            codebook_indexing (np.ndarray): Indices to match cells to codebooks.
            beam_to_node_mapping (np.ndarray): Maps beams to node IDs.
            antenna_to_node_a_mapping (np.ndarray): Maps antennas to node IDs.
            channel_gain_ueAnt_to_cellAnt_complex (np.ndarray): Complex channel gains, shape (UE_ant, cell_ant, freq).
    
        Returns:
            np.ndarray: Precoded channel gains in dB, shape (num_UE_antennas, num_cell_antennas, num_beams).
        """
        num_UE_antennas = np.size(channel_gain_ueAnt_to_cellAnt_complex, 0)
        num_cell_antennas = np.size(channel_gain_ueAnt_to_cellAnt_complex, 1)
        num_beams = np.size(beam_to_node_mapping)
    
        # Allocate output tensor on GPU
        precoded_channel_gain_b_to_a_dB = torch.zeros((num_UE_antennas, num_cell_antennas, num_beams),dtype=torch.float32,device=self.device)
    
        for cell_index in range(np.size(codebook_indexing, 0)):
            # Get the codebook for the current cell
            cell_codebook = codebook_dictionary[codebook_indexing[cell_index]]
            cell_codebook_torch = torch.tensor(cell_codebook[np.newaxis, np.newaxis, :, :], dtype=torch.complex32, device=self.device)
    
            # Get the complex channel gains for the current cell
            cell_channel_gain = channel_gain_ueAnt_to_cellAnt_complex[:, :, antenna_to_node_a_mapping == selected_cell_IDs[cell_index]]
            cell_channel_gain_torch = torch.tensor(cell_channel_gain[:, :, :, np.newaxis], dtype=torch.complex64, device=self.device)
    
            # Compute the precoded channel gains in dB
            gain_dB_torch = tools.mW_to_dBm_torch(torch.pow(torch.abs(torch.sum(cell_codebook_torch * cell_channel_gain_torch, dim=cell_channel_gain.ndim - 1)),2))
    
            # Assign the result to the corresponding beam indices
            beam_mask = beam_to_node_mapping == selected_cell_IDs[cell_index]
            precoded_channel_gain_b_to_a_dB[:, :, beam_mask] = gain_dB_torch.to(torch.float32)
    
            # Cleanup
            del cell_codebook_torch, cell_channel_gain_torch, gain_dB_torch
            if torch.cuda.is_available() and (cell_index % 10 == 0 or cell_index == np.size(codebook_indexing, 0) - 1):
                torch.cuda.empty_cache()
    
        # Transfer result to CPU and return as NumPy array
        out = precoded_channel_gain_b_to_a_dB.cpu().numpy()
        del precoded_channel_gain_b_to_a_dB
    
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
        return out  
    
        
    def example_precoding_gains(self, cell_index, beam_conf_obj, example_cell_channel_H, example_cell_channel_V):
        
        # Set parameters for the example 
        azimuth_deg = np.arange(-180, 180, 180/500)
        zenith_deg = np.arange(0, 180, 180/1000)
        
        A_m_dB = self.A_m_dB[cell_index]
        
        # Get precoders in cell
        cell_codebook = beam_conf_obj.codebook_dictionary[self.codebook_indexing[cell_index]] 
         
        # Get precoder plus array response gain in the H plane - we make a cut in the V-plane in the direction of the beta_downtilit
        precoded_channel_gain_b_to_a_dB_H =\
            self.calculate_precoded_channel_gain_per_cell(cell_codebook, example_cell_channel_H, A_m_dB)  
        
        # Get precoder plus array response gain in the V plane - we make a cut in the H-plane in the direction 0 degree aziimuth
        precoded_channel_gain_b_to_a_dB_V =\
            self.calculate_precoded_channel_gain_per_cell(cell_codebook, example_cell_channel_V, A_m_dB) 
            

    def calculate_precoded_channel_gain_per_cell(
        self,
        cell_codebook: np.ndarray,
        cell_channel_gain_ueAnt_to_cellAnt_complex: np.ndarray,
        A_m_dB: float,
    ) -> np.ndarray:
        """
        Calculates the precoded channel gain for a single cell using numpy.
    
        Args:
    
        Returns:
            np.ndarray:
                A numpy array representing the calculated precoded channel gains in dB.
                Shape: (num_UE_antennas, num_frequencies, num_beams).
        """
        # Add the precoder gain to the array factor gain
        cell_precoded_channel_gain_b_to_a_linear: np.ndarray = np.power(
            np.abs(
                np.sum(
                    cell_codebook[np.newaxis, np.newaxis, :, :] 
                    * cell_channel_gain_ueAnt_to_cellAnt_complex[:, :, :, np.newaxis],
                    axis=cell_channel_gain_ueAnt_to_cellAnt_complex.ndim - 1,
                )
            ),
            2,
        )

        cell_precoded_channel_gain_b_to_a_linear: np.ndarray = np.sum(
                    cell_codebook[np.newaxis, np.newaxis, :, :] 
                    * cell_channel_gain_ueAnt_to_cellAnt_complex[:, :, :, np.newaxis],
                    axis=cell_channel_gain_ueAnt_to_cellAnt_complex.ndim - 1,
                ).abs().pow(2)
    
        # Calculate the array response gains in dB
        cell_precoded_channel_gain_b_to_a_dB: np.ndarray = tools.mW_to_dBm(cell_precoded_channel_gain_b_to_a_linear)
    
        return cell_precoded_channel_gain_b_to_a_dB
    

class Precoded_Channel_Gain_SSB_no_fast_fading_ue_to_cell(Precoded_Channel_Gain):
    pass

class Precoded_Channel_Gain_SSB_ue_to_cell(Precoded_Channel_Gain):
    pass

class Precoded_Channel_Gain_CSI_RS_ue_to_cell(Precoded_Channel_Gain):
    pass

class Precoded_Channel_Gain_CSI_RS_no_fast_fading_ue_to_cell(Precoded_Channel_Gain):
    pass
