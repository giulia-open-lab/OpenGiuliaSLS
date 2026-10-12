# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:48:13 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@itea.upv.es
"""

import time

import numpy as np
from typing import Any, Dict, List

from giulia.event_driven.snapshot_control import Snapshot_control
from giulia.fs import results_file
from giulia.kpis import calculate_cell_beam_activity
from giulia.outputs.saveable import Saveable
from giulia.tools import tools
from giulia.tools.tools import log_calculations_time


class Cell_Selection(Saveable):
    """
    A class for selecting the best serving cell and beam for UEs based on RSRP values.
    Supports frequency-layered processing and UE-cell association.
    """

    def __init__(
        self,
        compute_type: str,
        simulation_config_obj: Any,
        ue_playground_deployment_obj: Any,
        ue_deployment_obj: Any,
        beam_conf_obj: Any,
        cell_re_selection_conf_obj: Any,
        distance_angles_ue_to_cell_obj: Any,
        channel_gain_ue_to_cell_obj: Any,
        RSRP_ue_to_cell_obj: Any,
        dl_noise_ue_to_cell_obj: Any,
    ):
        """
        Initializes the Cell_Selection class.

        Args:
            compute_type: Type of cell selection computation.
            simulation_config_obj: Simulation configuration details.
            ue_playground_deployment_obj: UE playground deployment details.
            ue_deployment_obj: UE deployment details.
            beam_conf_obj: Beam configuration details.
            cell_re_selection_conf_obj: Cell reselection configuration details.
            distance_angles_ue_to_cell_obj: Distance and angle details between UEs and cells.
            channel_gain_ue_to_cell_obj: Channel gain results between UEs and cells.
            RSRP_ue_to_cell_obj: RSRP values for UEs and cells.
            dl_noise_ue_to_cell_obj: Downlink noise details for UEs and cells.
        """
        super().__init__()
        # Inputs
        self.compute_type: str = compute_type
        self.simulation_config_obj: Any = simulation_config_obj
        self.ue_playground_deployment_obj: Any = ue_playground_deployment_obj
        self.ue_deployment_obj: Any = ue_deployment_obj
        self.beam_conf_obj: Any = beam_conf_obj
        self.cell_re_selection_conf_obj: Any = cell_re_selection_conf_obj
        self.distance_angles_ue_to_cell_obj: Any = distance_angles_ue_to_cell_obj
        self.channel_gain_ue_to_cell_obj: Any = channel_gain_ue_to_cell_obj
        self.RSRP_ue_to_cell_obj: Any = RSRP_ue_to_cell_obj
        self.dl_noise_ue_to_cell_obj: Any = dl_noise_ue_to_cell_obj

        # Outputs
        self.best_serving_beam_ID_per_ue: np.ndarray = np.array([])
        self.best_serving_cell_ID_per_ue: np.ndarray = np.array([])
        self.ue_association_results_per_frequency_layer: Dict[float, Dict[str, Any]] = {}


    def expose_variables(self) -> List[str]:
        """List of attributes name to be saved.

        Concrete (non-abstract) on the base class so the base ``Cell_Selection`` can be
        instantiated directly — e.g. by the pairing sub-run, which needs cell selection
        for its internal computation but saves nothing. Subclasses override this to export
        their KPIs.
        """
        return ['ues_per_cell']

    def process(self, rescheduling_us: int = -1) -> int:
        """
        Processes cell selection calculations.

        Args:
            rescheduling_us: Rescheduling time in microseconds. Defaults to -1.

        Returns:
            Updated rescheduling time.
        """
        # Start timer
        t_start: float = time.perf_counter()

        # Perform cell and beam selection
        if self.compute_type == "strongest_rsrp":
            self.best_serving_beam_ID_per_ue, self.best_serving_cell_ID_per_ue = self.strongest_rsrp(
                self.beam_conf_obj,
                self.RSRP_ue_to_cell_obj.RSRP_results_per_frequency_layer["all_freq"]["RSRP_ue_to_cell_dBm"],
            )

        # Log elapsed time
        log_calculations_time('Cell selection', t_start)

        return rescheduling_us


    def strongest_rsrp(
        self,
        beam_conf_obj: Any,
        rsrp_ue_to_cell_dBm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds the strongest RSRP for each UE.

        Args:
            beam_conf_obj: Beam configuration object.
            rsrp_ue_to_cell_dBm: RSRP values (dBm) for UEs and beams.

        Returns:
            - Best serving beam ID for each UE.
            - Best serving cell ID for each UE.
        """
        # Determine beam-to-cell mapping
        beam_to_node_mapping = np.arange(rsrp_ue_to_cell_dBm.shape[1]) \
            if beam_conf_obj is None else beam_conf_obj.beam_to_node_mapping
        
        # Find the strongest RSRP
        best_serving_beam_ID_per_ue = np.argmax(rsrp_ue_to_cell_dBm, axis=-1)
        best_serving_cell_ID_per_ue = beam_to_node_mapping[best_serving_beam_ID_per_ue]

        return best_serving_beam_ID_per_ue, best_serving_cell_ID_per_ue

    
    def calculate_server_stats(self, rescheduling_us: int = -1) -> int:
        """
        Calculates server statistics for the best-serving cell or beam for each UE.
        This includes metrics like distance, coupling gain, RSRP, and noise.
        Also calculates beam activity and optionally generates heatmaps for visualization.
    
        Args:
            rescheduling_us: Rescheduling time in microseconds. Defaults to -1.
    
        Returns:
            Updated rescheduling time.
        """
        # 1. Retrieve channel characteristics
        distance_3d_ue_to_cell_m: np.ndarray = self.distance_angles_ue_to_cell_obj.distance_b_to_a_3d_wraparound_m
        number_of_cells: int = distance_3d_ue_to_cell_m.shape[1]
    
        # 2. Configure beam-to-cell mapping
        if self.beam_conf_obj is None:  # LTE-like scenario: one beam per cell
            beam_type: str = "antenna element"
            number_of_beams_per_cell: np.ndarray = np.ones(number_of_cells, dtype=int)
            beam_to_node_mapping: np.ndarray = np.arange(len(self.ue_deployment_obj.df_ep))
        else:  # NR-like scenario: multiple beams per cell
            beam_type: str = self.beam_conf_obj.beam_type
            number_of_beams_per_cell: np.ndarray = self.beam_conf_obj.number_of_beams_per_node
            beam_to_node_mapping: np.ndarray = self.beam_conf_obj.beam_to_node_mapping
    
        # 3. Extract channel gains
        if self.beam_conf_obj is None:  # One beam per cell
            coupling_gain_ue_to_cell_dB: np.ndarray =\
                self.channel_gain_ue_to_cell_obj.slow_channel_results_per_frequency_layer["all_freq"]["slow_channel_gain_b_to_a_dB"].astype(np.float32)
            
            
        else:  # Multiple beams per cell
            coupling_gain_ue_to_cell_dB: np.ndarray =\
                self.channel_gain_ue_to_cell_obj.precoded_channel_gain_results_per_frequency_layer["all_freq"]["precoded_channel_gain_b_to_a_dB"]
            
    
        # 4. Extract RSRP and noise values
        rsrp_ue_to_cell_dBm: np.ndarray = self.RSRP_ue_to_cell_obj.RSRP_results_per_frequency_layer["all_freq"]["RSRP_ue_to_cell_dBm"]
        dl_full_bandwidth_noise_dBm: np.ndarray = self.dl_noise_ue_to_cell_obj.dl_noise_per_resource_element_ue_to_cell_dBm
    
        # 5. Define UE IDs
        ue_IDs: np.ndarray = np.arange(distance_3d_ue_to_cell_m.shape[0])
    
        # 6. Calculate statistics for the best server (cell/beam with strongest RSRP)
        # Get 3D distances
        self.best_server_distance_3d_per_ue_m: np.ndarray = distance_3d_ue_to_cell_m[
            ue_IDs, self.best_serving_cell_ID_per_ue
        ]
    
        # Calculate coupling gain
        if coupling_gain_ue_to_cell_dB.ndim == 3:  # Beam-level gain (3D tensor)
            self.best_server_coupling_gain_per_ue_dB: np.ndarray = tools.mW_to_dBm(
                np.mean(
                    tools.dBm_to_mW(coupling_gain_ue_to_cell_dB[:, ue_IDs, self.best_serving_beam_ID_per_ue]),
                    axis=0,
                )
            )
        else:  # Cell-level gain (2D tensor)
            self.best_server_coupling_gain_per_ue_dB: np.ndarray = coupling_gain_ue_to_cell_dB[
                ue_IDs, self.best_serving_beam_ID_per_ue
            ]
    
        # RSRP and noise values for the best server
        self.best_server_rsrp_per_ue_dBm: np.ndarray = rsrp_ue_to_cell_dBm[
            ue_IDs, self.best_serving_beam_ID_per_ue
        ]
        self.best_server_re_noise_per_ue_dBm: np.ndarray = dl_full_bandwidth_noise_dBm[
            ue_IDs, self.best_serving_cell_ID_per_ue
        ]
    
        # 7. Calculate cell/beam activity
        if self.beam_conf_obj is None:  # LTE: No beams, only cells
            self.beam_activity_per_ue, self.ues_per_beam = calculate_cell_beam_activity.cell_activity(
                number_of_cells, self.best_serving_cell_ID_per_ue
            )
            self.ues_per_cell = self.ues_per_beam
            
        elif self.beam_conf_obj.beam_type == "SSB":  # NR: SSB beams
            self.beam_activity_per_ue, self.ues_per_beam = calculate_cell_beam_activity.SSB_beam_activity(
                self.beam_conf_obj, self.best_serving_beam_ID_per_ue
            )
            # Map UE activity to cells
            self.ues_per_cell = np.bincount(beam_to_node_mapping, weights=self.ues_per_beam).astype(int)
    
        # 8. Generate heatmaps for visualization (if enabled)
        self._generate_heatmaps(beam_type, number_of_cells, number_of_beams_per_cell)
    
        # Return updated rescheduling time
        return rescheduling_us

    
    def _generate_heatmaps(
        self,
        beam_type: str,
        number_of_cells: int,
        number_of_beams_per_cell: np.ndarray,
    ):
        """
        Generates and saves heatmaps for best serving cell ID, beam ID, and RSRP.
    
        Args:
            beam_type: The type of beam used in the simulation.
            number_of_cells: Total number of cells.
            number_of_beams_per_cell: Array containing the number of beams per cell.
        """
        snapshot_control = Snapshot_control.get_instance()
        if self.simulation_config_obj.plot_heatmaps == 1 and snapshot_control.num_snapshots == 0:
            x_size: int = int(
                np.ceil(
                    self.ue_playground_deployment_obj.scenario_x_side_length_m
                    / self.ue_playground_deployment_obj.grid_resol_m
                )
            )
            y_size: int = int(
                np.ceil(
                    self.ue_playground_deployment_obj.scenario_y_side_length_m
                    / self.ue_playground_deployment_obj.grid_resol_m
                )
            )
            ue_grid_position: np.ndarray = self.ue_deployment_obj.ue_grid_position
    
            # Save heatmaps
            self._save_heatmap(
                f"best_serving_cell_heat_map_based_on_{beam_type}",
                x_size,
                y_size,
                ue_grid_position,
                self.best_serving_cell_ID_per_ue,
                self.ue_playground_deployment_obj.grid_resol_m,
                {"number_of_cells": number_of_cells},
            )
    
            self._save_heatmap(
                f"best_serving_beam_heat_map_based_on_{beam_type}",
                x_size,
                y_size,
                ue_grid_position,
                self.best_serving_beam_ID_per_ue,
                self.ue_playground_deployment_obj.grid_resol_m,
                {"number_of_beams": np.sum(number_of_beams_per_cell)},
            )
    
            self._save_heatmap(
                f"best_serving_cell_rsrp_map_based_on_{beam_type}",
                x_size,
                y_size,
                ue_grid_position,
                self.best_server_rsrp_per_ue_dBm,
                self.ue_playground_deployment_obj.grid_resol_m,
                {"number_of_cells": number_of_cells},
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
            data: Heatmap data to save (e.g., best serving cell ID or RSRP).
            grid_resol_m: Resolution of the grid in meters.
            metadata: Additional metadata to include in the file.
        """
        file_name = results_file(self.simulation_config_obj.project_name, f"to_plot_{file_suffix}")
        np.savez(
            file_name,
            x_size=x_size,
            y_size=y_size,
            ue_grid_position=ue_grid_position,
            data=data,
            grid_resol_m=grid_resol_m,
            **metadata,
        )


class Cell_Selection_CRS(Cell_Selection):
    def expose_variables(self) -> List[str]:
        """List of attributes name to be saved."""
        return [
            # ---- CRS (Cell-specific Reference Signal) ----
            "best_serving_cell_ID_per_ue",
            "best_server_distance_3d_per_ue_m",
            "best_server_coupling_gain_per_ue_dB",
            "best_server_re_noise_per_ue_dBm",
            "best_server_rsrp_per_ue_dBm",
            # ---- UE counts per cell and beam ----
            "ues_per_beam",
            # ---- Cell and beam activity ----
            "beam_activity_per_ue"
            ]

    def expose_variable_labels(self) -> Dict[str, str]:
        """Explicit per-variable routing — every variable is mapped, no label fallback."""
        return {
            # ---- CRS (Cell-specific Reference Signal) ----
            "best_serving_cell_ID_per_ue": "ue_kpi",
            "best_server_distance_3d_per_ue_m": "ue_kpi",
            "best_server_coupling_gain_per_ue_dB": "ue_kpi",
            "best_server_re_noise_per_ue_dBm": "ue_kpi",
            "best_server_rsrp_per_ue_dBm": "ue_kpi",
            # ---- UE counts per cell and beam ----
            "ues_per_beam": "crsBeam_kpi",
            # ---- Cell and beam activity ----
            # 2-D UE×cell activity mask: one row per UE, cells spread into columns.
            "beam_activity_per_ue": "ue_kpi"
        }

    def expose_variable_column_prefixes(self) -> Dict[str, str]:
        """Label the expanded sub-columns of the 2-D activity mask (cells spread into columns)."""
        return {
            # -> '<Class>.beam_activity_per_ue.cell0', '.cell1', ...
            "beam_activity_per_ue": "cell"
        }

    def expose_variable_descriptions(self) -> Dict[str, str]:
        """Human-readable description for each exposed variable (shown in the Saveable catalog)."""
        return {
            # ---- CRS (Cell-specific Reference Signal) ----
            "best_serving_cell_ID_per_ue": "ID of the serving cell for each UE, where the selection is based on measurements over the common reference signal (CRS). Used for LTE cells that do not employ massive MIMO arrays. In LTE or NR cells with massive MIMO arrays, this represents an association considering only the antenna gain from a single element, not from the full array.",
            "best_server_distance_3d_per_ue_m": "3D distance between each UE and its serving cell based on CRS-based selection.",
            "best_server_coupling_gain_per_ue_dB": "Coupling gain between the UE and its serving cell based on CRS-based selection. Represents the link budget gain, accounting for path loss, shadowing, outdoor-to-indoor penetration loss, and antenna/array gain.",
            "best_server_re_noise_per_ue_dBm": "Received noise power per resource element at the UE, typically used in SNR or SINR computations for CRS.",
            "best_server_rsrp_per_ue_dBm": "Reference Signal Received Power (RSRP) measured by each UE from its serving cell based on its serving CRS. Reflects the average received power of resource elements carrying such CRS.",
            # ---- UE counts per cell and beam ----
            "ues_per_beam": "Number of UEs associated with each cell based on CRS-based selection.",
            # ---- Cell and beam activity ----
            "beam_activity_per_ue": "Binary mask indicating the cells that were actively transmitting from the UE's perspective at the time it was transmitting or receiving data."
        }


class Cell_Selection_SSB(Cell_Selection):
    def expose_variables(self) -> List[str]:
        """List of attributes name to be saved."""
        return [
            # ---- SSB (Synchronization Signal Block) ----
            "best_serving_cell_ID_per_ue",
            "best_server_distance_3d_per_ue_m",
            "best_server_coupling_gain_per_ue_dB",
            "best_server_re_noise_per_ue_dBm",
            "best_server_rsrp_per_ue_dBm",
            "best_serving_beam_ID_per_ue",
            # ---- UE counts per cell and beam ----
            "ues_per_beam",
            "ues_per_cell",
            # ---- Cell and beam activity ----
            "beam_activity_per_ue"
            ]

    def expose_variable_labels(self) -> Dict[str, str]:
        """Explicit per-variable routing — every variable is mapped, no label fallback."""
        return {
            # ---- SSB (Synchronization Signal Block) ----
            "best_serving_cell_ID_per_ue": "ue_kpi",
            "best_server_distance_3d_per_ue_m": "ue_kpi",
            "best_server_coupling_gain_per_ue_dB": "ue_kpi",
            "best_server_re_noise_per_ue_dBm": "ue_kpi",
            "best_server_rsrp_per_ue_dBm": "ue_kpi",
            "best_serving_beam_ID_per_ue": "ue_kpi",
            # ---- UE counts per cell and beam ----
            "ues_per_beam": "ssbBeam_kpi",
            "ues_per_cell": "cell_kpi",
            # ---- Cell and beam activity ----
            # 2-D UE×SSB-beam activity mask: one row per UE, beams spread into columns.
            "beam_activity_per_ue": "ue_kpi"
        }

    def expose_variable_column_prefixes(self) -> Dict[str, str]:
        """Label the expanded sub-columns of the 2-D activity mask (beams spread into columns)."""
        return {
            # -> '<Class>.beam_activity_per_ue.beam0', '.beam1', ...
            "beam_activity_per_ue": "beam"
        }

    def expose_variable_descriptions(self) -> Dict[str, str]:
        """Human-readable description for each exposed variable (shown in the Saveable catalog)."""
        return {
            # ---- SSB (Synchronization Signal Block) ----
            "best_serving_cell_ID_per_ue": "ID of the serving cell for each UE, where the selection is based on measurements over the synchronization signal block (SSB). Used for LTE and NR cells that employ massive MIMO arrays. In LTE cells, there is only one SSB, which represents the CRS.",
            "best_server_distance_3d_per_ue_m": "3D distance between each UE and its serving cell based on SSB-based selection.",
            "best_server_coupling_gain_per_ue_dB": "Coupling gain between the UE and its serving cell based on SSB-based selection. Represents the link budget gain, accounting for path loss, shadowing, outdoor-to-indoor penetration loss, and antenna/array gain, considering SSB precoding.",
            "best_server_re_noise_per_ue_dBm": "Received noise power per resource element at the UE, typically used in SNR or SINR computations for SSB.",
            "best_server_rsrp_per_ue_dBm": "Reference Signal Received Power (RSRP) measured by each UE from its serving cell based on its serving SSB. Reflects the average received power of resource elements carrying such SSB.",
            "best_serving_beam_ID_per_ue": "ID of the serving SSB beam (coverage) for each UE, where the selection is based on SSB measurements.",
            # ---- UE counts per cell and beam ----
            "ues_per_beam": "Number of UEs associated with each SSB beam based on the corresponding SSB-based selection.",
            "ues_per_cell": "Number of UEs associated with each cell based on SSB-based selection.",
            # ---- Cell and beam activity ----
            "beam_activity_per_ue": "Binary mask indicating the SSB beams that were actively transmitting from the UE's perspective at the time it performed the SSB measurement. Recall that SSB beams are swept sequentially over time."
        }
