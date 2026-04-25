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

import sys
import time

import numpy as np
from typing import Any, Dict, List, Tuple

from giulia.event_driven.snapshot_control import Snapshot_control
from giulia.fs import results_file
from giulia.kpis import calculate_cell_beam_activity
from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable

class Beam_Selection(Saveable):
    """
    A class for selecting the best serving beam for each UE based on RSRP values across frequency layers.
    Computes beam statistics and outputs beam activity and UE-to-beam mappings.
    """

    def __init__(
        self,
        compute_type: str,
        simulation_config_obj: Any,
        network_deployment_obj: Any,
        ue_playground_deployment_obj: Any,
        ue_deployment_obj: Any,
        beam_conf_obj: Any,
        RSRP_ue_to_cell_obj: Any,
        best_serving_cell_per_ue_obj: Any,
    ):
        """
        Initializes the Beam_Selection class.

        Args:
            compute_type: Type of beam selection computation (e.g., 'strongest_rsrp').
            simulation_config_obj: Simulation configuration details.
            network_deployment_obj: Network deployment details.
            ue_playground_deployment_obj: UE playground deployment details.
            ue_deployment_obj: UE deployment details.
            beam_conf_obj: Beam configuration details.
            RSRP_ue_to_cell_obj: RSRP values for UEs and cells across frequency layers.
            best_serving_cell_per_ue_obj: Object containing the best serving cell per UE.
        """

        super().__init__()

        # Inputs
        self.compute_type: str = compute_type
        self.simulation_config_obj: Any = simulation_config_obj
        self.network_deployment_obj: Any = network_deployment_obj
        self.ue_playground_deployment_obj: Any = ue_playground_deployment_obj
        self.ue_deployment_obj: Any = ue_deployment_obj
        self.beam_conf_obj: Any = beam_conf_obj
        self.RSRP_ue_to_cell_obj: Any = RSRP_ue_to_cell_obj
        self.best_serving_cell_per_ue_obj: Any = best_serving_cell_per_ue_obj

        # Outputs
        self.best_serving_beam_ID_per_ue: np.ndarray = np.array([])
        self.best_serving_beam_rsrp_per_ue_dBm: np.ndarray = np.array([])


    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["best_serving_beam_ID_per_ue","best_serving_beam_rsrp_per_ue_dBm"]
    

    def process(self, rescheduling_us: int = -1) -> int:
        """
        Processes the beam selection calculations across frequency layers.
    
        Args:
            rescheduling_us: Rescheduling time in microseconds. Defaults to -1.
    
        Returns:
            Updated rescheduling time.
        """
        # Start performance timer
        t_start: float = time.perf_counter()
    
        # Get beam-to-cell mapping
        beam_to_node_mapping: np.ndarray = \
            np.arange(len(self.network_deployment_obj.df_ep)).astype(int) if self.beam_conf_obj is None \
            else self.beam_conf_obj.beam_to_node_mapping
    
        # Best serving cell per UE
        best_serving_cell_ID_per_ue: np.ndarray = self.best_serving_cell_per_ue_obj.best_serving_cell_ID_per_ue
    
        # Initialize outputs
        self.best_serving_beam_ID_per_ue: np.ndarray = np.full(len(self.ue_deployment_obj.df_ep), -1, dtype=int)
        self.best_serving_beam_rsrp_per_ue_dBm: np.ndarray = np.full(len(self.ue_deployment_obj.df_ep), np.nan, dtype=float)
    
        # Perform beam selection for each frequency layer
        if self.compute_type == "strongest_rsrp":
            for frequency_key, rsrp_data in self.RSRP_ue_to_cell_obj.RSRP_results_per_frequency_layer.items():
                
                # Compute best serving beams and corresponding RSRP values
                best_beam_global_idx, best_beam_rsrp_dBm = \
                    self.strongest_rsrp(beam_to_node_mapping,best_serving_cell_ID_per_ue,rsrp_data)                
    
                # Update results for the current frequency layer
                ues_in_frequency_mask = rsrp_data["ues_in_frequency_mask"]
                self.best_serving_beam_ID_per_ue[ues_in_frequency_mask] = best_beam_global_idx
                self.best_serving_beam_rsrp_per_ue_dBm[ues_in_frequency_mask] = best_beam_rsrp_dBm
    
        # Calculate beam activity if beam type is "CSI_RS"
        if self.beam_conf_obj and self.beam_conf_obj.beam_type == "CSI_RS":
            self.beam_activity_per_ue, self.ues_per_beam = \
                calculate_cell_beam_activity.CSI_RS_beam_activity(self.beam_conf_obj,self.best_serving_beam_ID_per_ue)
            
        # Plot some stats    
        self.calculate_beam_stats()
    
        # Log elapsed time
        log_calculations_time('Beam selection', t_start)
    
        return rescheduling_us
    

    def strongest_rsrp(
        self,
        beam_to_node_mapping: np.ndarray,
        best_serving_cell_ID_per_ue: np.ndarray,
        rsrp_data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Identifies the strongest RSRP beam for each UE and retrieves the corresponding RSRP values.
    
        Args:
            beam_to_node_mapping (np.ndarray): Mapping of beams to cell IDs. Shape: (Beams,)
            best_serving_cell_ID_per_ue (np.ndarray): Best serving cell ID per UE. Shape: (UEs,)
            rsrp_data (Dict[str, np.ndarray]): Dictionary containing:
                - "RSRP_ue_to_cell_dBm" (np.ndarray): RSRP values (dBm) for UEs and beams. Shape: (UEs, Beams)
                - "ues_in_frequency_mask" (np.ndarray): Boolean mask for UEs in frequency. Shape: (UEs,)
                - "cell_IDs" (np.ndarray): Selected cell IDs. Shape: (Cells,)
    
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - best_beam_local_idx (np.ndarray): Local index of the strongest beam per UE. Shape: (UEs,)
                - best_beam_global_idx (np.ndarray): Global index of the strongest beam per UE. Shape: (UEs,)
                - best_serving_beam_rsrp_per_ue (np.ndarray): RSRP values for the strongest beams per UE. Shape: (UEs,)
        """
    
        # Extract relevant RSRP data
        rsrp_ue_to_cell_dBm: np.ndarray = rsrp_data["RSRP_ue_to_cell_dBm"]
        ues_in_frequency_mask: np.ndarray = rsrp_data["ues_in_frequency_mask"]
        selected_cell_IDs: np.ndarray = rsrp_data["cell_IDs"]
    
        # Filter UEs that are active in the given frequency
        best_serving_cells_filtered: np.ndarray = best_serving_cell_ID_per_ue[ues_in_frequency_mask]
    
        # Identify beams belonging to the selected cells
        beam_mask: np.ndarray = np.isin(beam_to_node_mapping, selected_cell_IDs)
        beam_indices: np.ndarray = np.where(beam_mask)[0]  # Global beam indices
    
        # Identify beams corresponding to the best-serving cell of each UE
        beams_of_best_server_per_ue: np.ndarray = \
            beam_to_node_mapping[beam_mask][np.newaxis, :] == best_serving_cells_filtered[:, np.newaxis]
    
        # Mask RSRP values for beams not associated with the best-serving cell
        rsrp_aux: np.ndarray = np.where(beams_of_best_server_per_ue, rsrp_ue_to_cell_dBm, np.nan)
    
        # Find the strongest RSRP beam (local index within the filtered beams)
        best_beam_local_idx: np.ndarray = np.nanargmax(rsrp_aux, axis=-1)
    
        # Convert local beam index to global beam index
        best_beam_global_idx: np.ndarray = beam_indices[best_beam_local_idx]
    
        # Retrieve the corresponding strongest RSRP values
        best_serving_beam_rsrp_per_ue: np.ndarray = np.nanmax(rsrp_aux, axis=-1)
    
        return best_beam_global_idx, best_serving_beam_rsrp_per_ue


    def calculate_beam_stats(self):
        """
        Calculates beam statistics and generates heatmaps for best serving beam IDs and RSRP values.
        """
        # Check if heatmap plotting is enabled and there are no snapshots yet
        snapshot_control = Snapshot_control.get_instance()
        if self.simulation_config_obj.plot_heatmaps == 1 and snapshot_control.num_snapshots == 0:
            # Retrieve beam-to-cell mapping information
            number_of_cells: int = self.network_deployment_obj.df_ep.shape[0]
            beam_type: str = self.beam_conf_obj.beam_type
            number_of_beams_per_cell: np.ndarray = self.beam_conf_obj.number_of_beams_per_node
    
            # UE grid and deployment variables
            x_size: int = int(
                np.ceil(self.ue_playground_deployment_obj.scenario_x_side_length_m /
                        self.ue_playground_deployment_obj.grid_resol_m)
            )
            y_size: int = int(
                np.ceil(self.ue_playground_deployment_obj.scenario_y_side_length_m /
                        self.ue_playground_deployment_obj.grid_resol_m)
            )
            ue_grid_position: np.ndarray = self.ue_deployment_obj.ue_grid_position
    
            # Generate and save the best serving beam ID heatmap
            self._save_heatmap(
                file_suffix=f"best_serving_beam_heat_map_based_on_{beam_type}",
                x_size=x_size,
                y_size=y_size,
                ue_grid_position=ue_grid_position,
                data=self.best_serving_beam_ID_per_ue,
                grid_resol_m=self.ue_playground_deployment_obj.grid_resol_m,
                metadata={"number_of_beams": np.sum(number_of_beams_per_cell)},
            )
    
            # Generate and save the best serving RSRP heatmap
            self._save_heatmap(
                file_suffix=f"best_serving_cell_rsrp_map_based_on_{beam_type}",
                x_size=x_size,
                y_size=y_size,
                ue_grid_position=ue_grid_position,
                data=self.best_serving_beam_rsrp_per_ue_dBm,
                grid_resol_m=self.ue_playground_deployment_obj.grid_resol_m,
                metadata={"number_of_cells": number_of_cells},
            )
    

    def _save_heatmap(
        self,
        file_suffix: str,
        x_size: int,
        y_size: int,
        ue_grid_position: np.ndarray,
        data: np.ndarray,
        grid_resol_m: float,
        metadata: Dict[str, Any],
    ):
        """
        Saves heatmap data to a file for plotting.
    
        Args:
            file_suffix: Suffix for the file name.
            x_size: Size of the heatmap grid in the x-dimension.
            y_size: Size of the heatmap grid in the y-dimension.
            ue_grid_position: UE positions on the grid.
            data: Heatmap data to save (e.g., best serving beam IDs or RSRP).
            grid_resol_m: Resolution of the grid in meters.
            metadata: Additional metadata to include in the file.
        """
        # Generate file name
        file_name = results_file(self.simulation_config_obj.project_name, f"to_plot_{file_suffix}")
    
        # Save data to a .npz file
        np.savez(
            file_name,
            x_size=x_size,
            y_size=y_size,
            ue_grid_position=ue_grid_position,
            data=data,
            grid_resol_m=grid_resol_m,
            **metadata,
        )
