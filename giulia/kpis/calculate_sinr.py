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
from typing import Any, Dict, Tuple, Optional, List

from giulia.event_driven.snapshot_control import Snapshot_control
from giulia.fs import results_file
from giulia.tools import tools
from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable


class SINR(Saveable):
    """
    A class to calculate SINR (Signal-to-Interference-plus-Noise Ratio) for UEs.
    Supports calculations across multiple frequency layers and provides methods
    for effective SINR calculations based on RSRP or per-PRB RSS.
    """

    def __init__(
        self,
        compute_type: str,
        beam_type: str,
        simulation_config_obj: Any,
        site_deployment_obj: Any,
        ue_playground_deployment_obj: Any,
        ue_deployment_obj: Any,
        time_frequency_resource_obj: Any,
        base_stations_obj: Any,
        RSRP_ue_to_cell_obj: Any,
        best_serving_cell_per_ue_obj: Any,
        best_serving_beam_per_ue_obj: Any,
        mi_obj: Optional[Any] = None,
        lut_bler_vs_sinrs_obj: Optional[Any] = None,
    ):
        """
        Initializes the SINR class.

        Args:
            compute_type: Type of SINR computation (e.g., 'sinr_based_on_beam_rsrp').
            beam_type: Type of beam (e.g., 'antenna element', 'SSB', 'CSI_RS').
            simulation_config_obj: Simulation configuration details.
            site_deployment_obj: Network site deployment details.
            ue_playground_deployment_obj: UE playground deployment details.
            ue_deployment_obj: UE deployment details.
            time_frequency_resource_obj: Time-frequency resource configuration.
            base_stations_obj: Base station configuration.
            RSRP_ue_to_cell_obj: RSRP values for UEs and cells across frequency layers.
            best_serving_cell_per_ue_obj: Object containing best serving cell information for each UE.
            best_serving_beam_per_ue_obj: Object containing best serving beam information for each UE.
            mi_obj: Mutual Information object for effective SINR calculations.
            lut_bler_vs_sinrs_obj: LUT-based BLER-vs-SINR mapping for effective SINR calculations.
        """

        super().__init__()

        # Inputs
        self.compute_type: str = compute_type
        self.beam_type: str = beam_type
        self.simulation_config_obj: Any = simulation_config_obj
        self.site_deployment_obj: Any = site_deployment_obj
        self.ue_playground_deployment_obj: Any = ue_playground_deployment_obj
        self.ue_deployment_obj: Any = ue_deployment_obj
        self.time_frequency_resource_obj: Any = time_frequency_resource_obj
        self.base_stations_obj: Any = base_stations_obj
        self.RSRP_ue_to_cell_obj: Any = RSRP_ue_to_cell_obj
        self.best_serving_cell_per_ue_obj: Any = best_serving_cell_per_ue_obj
        self.best_serving_beam_per_ue_obj: Any = best_serving_beam_per_ue_obj
        self.mi_obj: Optional[Any] = mi_obj
        self.lut_bler_vs_sinrs_obj: Optional[Any] = lut_bler_vs_sinrs_obj

        # Outputs
        self.SINR_results_per_frequency_layer: Dict[str, Dict[str, Any]] = {}

    
    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["SINR_results_per_frequency_layer"]


    def process(self, rescheduling_us: int = -1) -> int:
        """
        Processes SINR calculations for all frequency layers.
    
        Args:
            rescheduling_us: Rescheduling time in microseconds. Defaults to -1.
    
        Returns:
            Updated rescheduling time.
        """
        t_start: float = time.perf_counter()
    
        # Handle RSRP-based SINR calculations
        if self.compute_type == "sinr_based_on_beam_rsrp":
            for frequency_key, rsrp_data in self.RSRP_ue_to_cell_obj.RSRP_results_per_frequency_layer.items():
                # Extract RSRP data for the current frequency layer
                noise_dBm: np.ndarray = self.best_serving_cell_per_ue_obj.best_server_re_noise_per_ue_dBm
    
                # Select beam activity and best serving elements based on beam type
                if self.beam_type == "antenna element":
                    best_serving_elements = self.best_serving_cell_per_ue_obj.best_serving_cell_ID_per_ue
                    beam_activity: Optional[np.ndarray] = None  # No beam activity for antenna elements
                elif self.beam_type == "SSB":
                    best_serving_elements = self.best_serving_cell_per_ue_obj.best_serving_beam_ID_per_ue
                    beam_activity: Optional[np.ndarray] = self.best_serving_cell_per_ue_obj.beam_activity_per_ue  # SSB-specific activity
                elif self.beam_type == "CSI_RS":
                    best_serving_elements = self.best_serving_beam_per_ue_obj.best_serving_beam_ID_per_ue
                    beam_activity: Optional[np.ndarray] = self.best_serving_beam_per_ue_obj.beam_activity_per_ue  # CSI_RS-specific activity
                else:
                    raise ValueError(f"Unsupported beam type: {self.beam_type}")
    
                # Compute SINR for the current frequency layer
                sinr_ue_dB,  usefulPower_ue_dBm, interfPower_ue_dBm = self._sinr_based_on_beam_rsrp(
                    self.beam_type, rsrp_data, best_serving_elements, noise_dBm, beam_activity
                )
                
                # Store results for the frequency layer
                self.SINR_results_per_frequency_layer[frequency_key] = copy.deepcopy(rsrp_data)
                del self.SINR_results_per_frequency_layer[frequency_key]["RSRP_ue_to_cell_dBm"]
                self.SINR_results_per_frequency_layer[frequency_key].update({
                    "sinr_ue_to_cell_dB": sinr_ue_dB,
                    "usefulPower_ue_dBm": usefulPower_ue_dBm,
                    "interfPower_ue_dBm": interfPower_ue_dBm,
                })  
    
        # Handle RSS-based SINR calculations
        elif self.compute_type == "sinr_per_PRB_based_on_beam_rrs":
            for frequency_key, rss_data in self.RSRP_ue_to_cell_obj.RSS_results_per_frequency_layer.items():
                # Extract RSS data for the current frequency layer
                noise_dBm: np.ndarray = self.best_serving_cell_per_ue_obj.best_server_re_noise_per_ue_dBm
                beam_activity: Optional[np.ndarray] = self.best_serving_beam_per_ue_obj.beam_activity_per_ue
                best_serving_beams = self.best_serving_beam_per_ue_obj.best_serving_beam_ID_per_ue
    
                # Compute SINR per PRB
                sinr_PRB_ue_dB,  usefulPower_PRB_ue_dBm, interfPower_PRB_ue_dBm  = \
                    self._sinr_per_PRB_based_on_beam_rrs(self.beam_type, rss_data, best_serving_beams, noise_dBm, beam_activity)
    
                # Compute effective SINR for the current frequency layer
                PRB_ue_activity = self.base_stations_obj.resource_allocation["PRB_ue_activity"][:,rss_data["ues_in_frequency_mask"]]
                
                effective_sinr_ue_dB = \
                    self._effective_sinr_miesm(self.base_stations_obj, self.mi_obj, self.lut_bler_vs_sinrs_obj, PRB_ue_activity, sinr_PRB_ue_dB)
                
                # Store results for the frequency layer
                self.SINR_results_per_frequency_layer[frequency_key] = copy.deepcopy(rss_data)
                del self.SINR_results_per_frequency_layer[frequency_key]["RSS_per_PRB_ue_to_cell_dBm"]
                self.SINR_results_per_frequency_layer[frequency_key].update({
                    "sinr_ue_to_cell_dB": sinr_PRB_ue_dB,
                    "usefulPower_PRB_ue_dBm": usefulPower_PRB_ue_dBm,
                    "interfPower_PRB_ue_dBm": interfPower_PRB_ue_dBm,
                    "effective_sinr_ue_to_cell_dB": effective_sinr_ue_dB
                })                 
    
        else:
            raise ValueError(f"Unsupported compute type: {self.compute_type}")
    
        # Log elapsed time
        log_calculations_time('SINR', t_start)
    
        return rescheduling_us

    
    def _prepare_data(self, beam_type: str, data: Dict[str, np.ndarray], power_key: str):
        """
        Prepares UE mask, beam mask, global beam IDs, and power matrix for SINR computation.
        """
        # Extract mask for UEs that are active in the current frequency layer
        ues_mask = data["ues_in_frequency_mask"]
        
        # Extract beam mask for the same carrier as the best serving beam
        if beam_type == "antenna element":
            beams_mask, global_beam_IDs = data["cells_in_frequency_mask"], data["cell_IDs"]
        elif beam_type == "SSB":
            beams_mask, global_beam_IDs = data["ssb_beams_in_frequency_mask"], data["global_ssb_beams_IDs"]
        elif beam_type == "CSI_RS":
            beams_mask, global_beam_IDs = data["csi_rs_beams_in_frequency_mask"], data["global_csi_rs_beams_IDs"]
        else:
            raise ValueError(f"Unsupported beam type: {beam_type}")
            
        # Extract RSRP values (dBm) or RSS values (dBm) per PRB from the provided data
        power_matrix_dBm =data[power_key]
        
        return ues_mask, beams_mask, global_beam_IDs, power_matrix_dBm
    

    def _get_best_serving_beam_info(
        self, beam_type: str, best_serving_beam_per_ue: np.ndarray, 
        best_serving_beam_re_noise_per_ue_dBm: np.ndarray, 
        ues_in_frequency_mask: np.ndarray, beams_in_frequency_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves best serving beam information and relevant filtering.
    
        Args:
            beam_type (str): Type of beam (antenna element, SSB, CSI_RS).
            best_serving_beam_per_ue (np.ndarray): Best serving beam IDs for each UE.
            best_serving_beam_re_noise_per_ue_dBm (np.ndarray): Noise values (dBm) for best serving beams.
            ues_in_frequency_mask (np.ndarray): Mask for UEs active in the current frequency layer.
            beams_in_frequency_mask (np.ndarray): Mask for beams active in the frequency layer.
    
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - best_serving_beams (Global IDs for best serving beams).
            - beams_in_same_carrier_than_server (Beams operating in the same frequency).
            - best_serving_beam_re_noise_per_ue_dBm (Filtered noise values for best serving beams).
        """
        # Get best serving beams for UEs (Global ID)
        best_serving_beams = best_serving_beam_per_ue[ues_in_frequency_mask]
    
        # Get beams operating in the same frequency as best serving beam
        beams_in_same_carrier_than_server = \
            self.time_frequency_resource_obj.beams_in_same_carrier_than_serving_beam(beam_type, best_serving_beam_per_ue)[ues_in_frequency_mask]
        beams_in_same_carrier_than_server = beams_in_same_carrier_than_server[:, beams_in_frequency_mask]
    
        # Filter noise information
        best_serving_beam_re_noise_per_ue_dBm = best_serving_beam_re_noise_per_ue_dBm[ues_in_frequency_mask]
    
        return best_serving_beams, beams_in_same_carrier_than_server, best_serving_beam_re_noise_per_ue_dBm
    
    
    def _compute_signal_and_interference(self, global_beam_IDs, best_serving_beams, power_matrix, sum_power_mW, is_prb_level=False, prb_activity=None):
        """
        Computes signal power and interference power for both UE-level and PRB-level calculations.
    
        Args:
            global_beam_IDs (np.ndarray): Global beam IDs.
            best_serving_beams (np.ndarray): Best serving beam IDs for UEs.
            power_matrix (np.ndarray): RSRP or RSS values in dBm.
            sum_power_mW (np.ndarray): Total received power (signal + interference).
            is_prb_level (bool): Whether computation is at PRB level (default: False).
            prb_activity (np.ndarray): PRB activity mask for UEs (only required for PRB-level).
    
        Returns:
            Tuple[np.ndarray, np.ndarray]: Signal power (mW), Interference power (mW).
        """
        # Find local IDs of best serving beams
        best_beam_local_idx = np.searchsorted(global_beam_IDs, best_serving_beams)
    
        # Compute signal power
        if is_prb_level:
            signal_mW = tools.dBm_to_mW(power_matrix[:, np.arange(power_matrix.shape[1]), best_beam_local_idx])
            signal_mW_masked = np.full(sum_power_mW.shape, np.nan, dtype=np.single)
            signal_mW_masked[prb_activity] = signal_mW[prb_activity]  # Apply PRB mask
            signal_mW = signal_mW_masked  # Use the masked version
        else:
            signal_mW = tools.dBm_to_mW(power_matrix[np.arange(power_matrix.shape[0]), best_beam_local_idx])
    
        # Compute interference power
        interference_mW = sum_power_mW - signal_mW
        interference_mW[interference_mW < 1e-16] = 0  # Avoid negative values
    
        return signal_mW, interference_mW


    def _compute_sinr(self, signal_mW, interference_mW, noise_dBm, is_prb_level=False, prb_activity=None):
        """
        Computes SINR (Signal-to-Interference-plus-Noise Ratio) in dB.
    
        Args:
            signal_mW (np.ndarray): Signal power in mW.
            interference_mW (np.ndarray): Interference power in mW.
            noise_dBm (np.ndarray): Noise power in dBm.
            is_prb_level (bool): Whether computation is at PRB level (default: False).
            prb_activity (np.ndarray): PRB activity mask (only required for PRB-level).
    
        Returns:
            np.ndarray: SINR values in dB.
        """
        # Convert noise from dBm to mW
        if is_prb_level:
            noise_per_PRB_dBm = np.broadcast_to(noise_dBm[np.newaxis, :], prb_activity.shape)
            noise_mW = tools.dBm_to_mW(noise_per_PRB_dBm)
        else:
            noise_mW = tools.dBm_to_mW(noise_dBm)
    
        # Compute SINR
        sinr_mW = signal_mW / (interference_mW + noise_mW)
        sinr_dB = tools.mW_to_dBm(sinr_mW)
    
        if is_prb_level:
            sinr_dB_masked = np.full(prb_activity.shape, np.nan, dtype=np.single)
            sinr_dB_masked[prb_activity] = sinr_dB[prb_activity]  # Apply PRB mask
            return sinr_dB_masked
        else:
            return sinr_dB
        
    
    def _sinr_based_on_beam_rsrp(
        self,
        beam_type: str,
        rsrp_data: Dict[str, np.ndarray],
        best_serving_beam_per_ue: np.ndarray,
        best_serving_beam_re_noise_per_ue_dBm: np.ndarray,
        beam_activity_per_ue: np.ndarray = None,
    ) -> np.ndarray:
        """
        Calculates SINR using RSRPs, assuming all beams operate in the same frequency carrier.
    
        Args:
            rsrp_ue_to_cell_dBm: RSRP values (dBm) for UEs and beams.
            best_serving_beam_per_ue: Array of the best serving beam ID for each UE.
            best_serving_beam_re_noise_per_ue_dBm: Noise levels (dBm) for the best serving beams.
            beam_activity_per_ue: Optional mask indicating active beams per UE. If None, assumes all beams are active.
    
        Returns:
            rsrp_based_sinr_dB: SINR values (dB) for each UE.
        """
        # Get masks, global beam IDs, and power matrix
        ues_in_frequency_mask, beams_in_frequency_mask, global_beam_IDs, rsrp_ue_to_cell_dBm = self._prepare_data(beam_type, rsrp_data, "RSRP_ue_to_cell_dBm")
        
        # Get per UE beam info
        best_serving_beams, beams_in_same_carrier_than_server, best_serving_beam_re_noise_per_ue_dBm = self._get_best_serving_beam_info(
            beam_type, best_serving_beam_per_ue, 
            best_serving_beam_re_noise_per_ue_dBm, 
            ues_in_frequency_mask, beams_in_frequency_mask)           

        # Calculate sum of received power (interference + signal)
        if beam_activity_per_ue is not None:
            # Filter relevant information
            beam_activity_per_ue = beam_activity_per_ue[ues_in_frequency_mask, :][:, beams_in_frequency_mask]
            # Filter beams based on activity
            active_rsrp_mW = np.where(np.logical_and(beams_in_same_carrier_than_server, beam_activity_per_ue), tools.dBm_to_mW(rsrp_ue_to_cell_dBm), 0)
        else:
            # Assume all beams are active (worst-case scenario)
            active_rsrp_mW = np.where(beams_in_same_carrier_than_server, tools.dBm_to_mW(rsrp_ue_to_cell_dBm), 0)
        sum_rsrp_mW = np.sum(active_rsrp_mW, axis=1)
    
        # Calculate signal and interference power
        signal_mW, interference_mW = self._compute_signal_and_interference(global_beam_IDs, best_serving_beams, rsrp_ue_to_cell_dBm, sum_rsrp_mW)

        # Calculate SINR
        rsrp_based_sinr_dB = self._compute_sinr(signal_mW, interference_mW, best_serving_beam_re_noise_per_ue_dBm)

        # Generate heatmap plots (optional)
        snapshot_control = Snapshot_control.get_instance()
        if self.simulation_config_obj.plot_heatmaps == 1 and snapshot_control.num_snapshots == 0:
            self._generate_heatmaps(rsrp_based_sinr_dB, interference_mW, np.size(rsrp_ue_to_cell_dBm,1))
    
        return rsrp_based_sinr_dB, tools.mW_to_dBm(signal_mW), tools.mW_to_dBm(interference_mW)


    def _sinr_per_PRB_based_on_beam_rrs(
        self,
        beam_type: str,
        rss_data: Dict[str, np.ndarray],
        best_serving_beam_per_ue: np.ndarray,
        best_serving_beam_re_noise_per_ue_dBm: np.ndarray,
        beam_activity_per_ue: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calculates SINR per PRB for UEs, embracing variability in PRB usage.
    
        Args:
            rss_data (Dict[str, np.ndarray]): Dictionary containing RSS-related data.
                                              Expected keys: "ues_in_frequency_mask", "RSS_per_PRB_ue_to_cell_dBm".
            best_serving_beam_per_ue (np.ndarray): Best serving beam IDs for each UE. Shape: (UEs,)
            best_beam_local_idx (np.ndarray): Local indices of the best beams per UE. Shape: (UEs,)
            best_serving_beam_re_noise_per_ue_dBm (np.ndarray): Noise values (dBm) for the best serving beams. Shape: (UEs,)
            beam_activity_per_ue (Optional[np.ndarray]): Mask indicating active beams per UE. Shape: (UEs, Beams). Default is None.
    
        Returns:
            np.ndarray: SINR values (dB) per PRB for each UE. Shape: (PRBs, UEs)
        """    
        # Select computation device (GPU if available, otherwise CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Get masks, global beam IDs, and power matrix
        ues_in_frequency_mask, beams_in_frequency_mask, global_beam_IDs, rss_per_PRB_ue_to_cell_dBm = self._prepare_data(beam_type, rss_data, "RSS_per_PRB_ue_to_cell_dBm")     
        number_of_prbs = np.size(rss_per_PRB_ue_to_cell_dBm,0) 
        
        # Get per UE beam info
        best_serving_beams, beams_in_same_carrier_than_server, best_serving_beam_re_noise_per_ue_dBm = self._get_best_serving_beam_info(
            beam_type, 
            best_serving_beam_per_ue, 
            best_serving_beam_re_noise_per_ue_dBm, 
            ues_in_frequency_mask, beams_in_frequency_mask)            
        
        # Filter relevant information
        beam_activity_per_ue = beam_activity_per_ue[ues_in_frequency_mask, :][:, beams_in_frequency_mask]        

        # Retrieve PRB activity and interference activity from base station resource allocation
        PRB_ue_activity: np.ndarray = self.base_stations_obj.resource_allocation["PRB_ue_activity"][:number_of_prbs, ues_in_frequency_mask]
        PRB_ue_beam_interference_activity: np.ndarray = \
            self.base_stations_obj.resource_allocation["PRB_ue_beam_interference_activity"][:number_of_prbs, ues_in_frequency_mask, :][:, :, beams_in_frequency_mask]
        
        # Sum received power (interference + signal) per PRB
        # Convert inputs to tensors
        beams_mask = torch.tensor(beams_in_same_carrier_than_server, device=device, dtype=torch.bool)
        rss_tensor = torch.tensor(rss_per_PRB_ue_to_cell_dBm, device=device, dtype=torch.float32)
    
        # Apply beam activity mask if available
        if beam_activity_per_ue is not None:
            activity_mask = torch.tensor(beam_activity_per_ue, device=device, dtype=torch.bool)
            beams_mask = beams_mask & activity_mask
    
        # Apply PRB interference activity mask
        prb_interference_mask = torch.tensor(PRB_ue_beam_interference_activity, device=device, dtype=torch.bool)
        beams_mask = beams_mask & prb_interference_mask
    
        # Select RSS values for active beams and convert to mW
        selected_rss_dBm = torch.where(beams_mask, rss_tensor, torch.tensor(np.NINF, device=device))
        rss_per_PRB_mW = tools.dBm_to_mW_torch(selected_rss_dBm)
        sum_rss_mW = torch.sum(rss_per_PRB_mW, axis=2).cpu().numpy()
    
        # Calculate signal and interference power per PRB
        signal_mW, interference_mW = self._compute_signal_and_interference(
            global_beam_IDs, best_serving_beams, rss_per_PRB_ue_to_cell_dBm, sum_rss_mW, 
            is_prb_level=True, prb_activity=PRB_ue_activity
        )
    
        # # Calculate SINR per PRB
        rss_based_sinr_dB = self._compute_sinr(
            signal_mW, interference_mW, best_serving_beam_re_noise_per_ue_dBm,
            is_prb_level=True, prb_activity=PRB_ue_activity
        )
    
        # Optionally generate heatmaps
        snapshot_control = Snapshot_control.get_instance()
        if self.simulation_config_obj.plot_heatmaps == 1 and snapshot_control.num_snapshots == 0:
            self._generate_prb_heatmaps(rss_based_sinr_dB,np.size(selected_rss_dBm,2))
    
        return rss_based_sinr_dB, tools.mW_to_dBm(signal_mW), tools.mW_to_dBm(interference_mW)
    

    def _generate_heatmaps(self, rsrp_based_sinr_dB: np.ndarray, interference_mW: np.ndarray, number_of_cells : int):
        """
        Generates and saves SINR and interference heatmaps for visualization.
    
        Args:
            rsrp_based_sinr_dB: SINR values (dB) for each UE.
            interference_mW: Interference values (mW) for each UE.
        """
        # Calculate grid dimensions and UE positions
        x_size = int(np.ceil(self.ue_playground_deployment_obj.scenario_x_side_length_m / self.ue_playground_deployment_obj.grid_resol_m))
        y_size = int(np.ceil(self.ue_playground_deployment_obj.scenario_y_side_length_m / self.ue_playground_deployment_obj.grid_resol_m))
        ue_grid_position = self.ue_deployment_obj.ue_grid_position
    
        # Plot SINR heatmap
        file_name = results_file(
            self.simulation_config_obj.project_name,
            f"to_plot_geometry_sinr_heat_map_based_on_{self.beam_type}"
        )
        np.savez(
            file_name,
            x_size=x_size,
            y_size=y_size,
            ue_grid_position=ue_grid_position,
            rsrp_based_sinr_dB=rsrp_based_sinr_dB,
            grid_resol_m=self.ue_playground_deployment_obj.grid_resol_m,
            number_of_cells=number_of_cells,
        )
    
        # Plot interference heatmap
        file_name = results_file(self.simulation_config_obj.project_name, f"to_plot_geometry_interference_heat_map_based_on_{self.beam_type}")
        np.savez(
            file_name,
            x_size=x_size,
            y_size=y_size,
            ue_grid_position=ue_grid_position,
            interference_dB=tools.mW_to_dBm(interference_mW),
            grid_resol_m=self.ue_playground_deployment_obj.grid_resol_m,
            number_of_cells=number_of_cells,
        )   


    def _generate_prb_heatmaps(self, rss_based_sinr_dB: np.ndarray, number_of_cells : int):
        """
        Generates SINR heatmaps per PRB for visualization.
    
        Args:
            rss_based_sinr_dB: SINR values (dB) per PRB for each UE.
        """
        # Calculate grid dimensions and UE positions
        x_size = int(np.ceil(self.ue_playground_deployment_obj.scenario_x_side_length_m / self.ue_playground_deployment_obj.grid_resol_m))
        y_size = int(np.ceil(self.ue_playground_deployment_obj.scenario_y_side_length_m / self.ue_playground_deployment_obj.grid_resol_m))
        ue_grid_position = self.ue_deployment_obj.ue_grid_position
    
        # Plot SINR heatmap
        file_name = results_file(self.simulation_config_obj.project_name, f"to_plot_per_prb_sinr_heat_map_based_on_{self.beam_type}",)
        np.savez(
            file_name,
            x_size=x_size,
            y_size=y_size,
            ue_grid_position=ue_grid_position,
            effective_sinr_mean=self._effective_sinr_mean(rss_based_sinr_dB),
            grid_resol_m=self.ue_playground_deployment_obj.grid_resol_m,
            number_of_cells=number_of_cells,
        )


    def _effective_sinr_mean(
        self,
        rss_based_sinr_dB: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates the effective SINR of each UE as the average SINR of its PRBs.
    
        Args:
            rss_based_sinr_dB: SINR values (dB) per PRB for each UE. Shape: (PRBs, UEs).
    
        Returns:
            Effective SINR values (dB) for each UE. Shape: (UEs,).
        """
        # Convert SINR from dB to mW, compute the mean, and convert back to dB
        return tools.mW_to_dBm(np.nanmean(tools.dBm_to_mW(rss_based_sinr_dB), axis=0))
    
    
    def _effective_sinr_eesm(
        self,
        rss_based_sinr_dB: np.ndarray,
        beta: float = 5.0,
    ) -> np.ndarray:
        """
        Calculates the effective SINR of each UE using Exponential Effective SNR Mapping (EESM).
    
        Reference: COMPUTATION FOR WIMAX PHYSICAL LAYER by Abdel Karim Al Tamimi.
    
        Args:
            rss_based_sinr_dB: SINR values (dB) per PRB for each UE. Shape: (PRBs, UEs).
            beta: EESM scaling factor. Default is 5.
    
        Returns:
            Effective SINR values (dB) for each UE. Shape: (UEs,).
        """
        # Convert SINR from dB to mW, compute EESM, and convert back to dB
        return tools.mW_to_dBm(-beta * np.log(np.nanmean(np.exp(-tools.dBm_to_mW(rss_based_sinr_dB) / beta), axis=0)))
    
    
    def _effective_sinr_miesm(
        self,
        base_stations_obj: Any,
        mi_obj: Any,
        lut_bler_vs_sinrs_obj: Any,
        PRB_ue_activity: np.ndarray,
        rss_based_sinr_dB: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the effective SINR for all UEs using MIESM (Mutual Information-based Effective SINR Mapping).
    
        Args:
            base_stations_obj: Object containing base station configuration, including PRB activity for UEs.
            mi_obj: Instance of Mutual Information class to compute effective SINR.
            lut_bler_vs_sinrs_obj: Instance of LUT_BLER_VS_SINR class to compute BLER.
            rss_based_sinr_dB: SINR values (dB) per PRB for each UE. Shape: (PRBs, UEs).
    
        Returns:
            Effective SINR values (dB) for each UE for the selected MCS. Shape: (UEs,).
        """
        # Step 1: Calculate effective SINR for all UEs and MCSs
        effective_sinr_per_ue_mcs: np.ndarray = mi_obj.calculate_effective_sinr_per_ue_mcs(PRB_ue_activity, rss_based_sinr_dB)
    
        # Step 2: Calculate BLER for all UEs and MCSs
        bler_per_ue_mcs: np.ndarray = lut_bler_vs_sinrs_obj.calculate_bler_per_ue_and_mcs(effective_sinr_per_ue_mcs)
    
        # Step 3: Select the optimal MCS for each UE based on the BLER target
        bler_target: float = 0.1  # Target BLER (e.g., 10%)
        mcs_per_ue: np.ndarray = lut_bler_vs_sinrs_obj.calculate_selected_mcs_per_ue(bler_per_ue_mcs, bler_target)
    
        # Step 4: Retrieve the effective SINR corresponding to the selected MCS for each UE
        selected_sinr_per_ue: np.ndarray = effective_sinr_per_ue_mcs[
            np.arange(len(mcs_per_ue)),
            lut_bler_vs_sinrs_obj.modulation_index_per_modulation_and_coding_scheme[mcs_per_ue],
        ]
    
        return selected_sinr_per_ue
    

class SINR_CRS(SINR):
    pass

class SINR_SSB(SINR):
    pass

class SINR_CSI_RS(SINR):
    pass
