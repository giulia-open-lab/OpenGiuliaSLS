# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:05:46 2024

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@itea.upv.es
"""

from typing import List, Any, Dict

import numpy as np
import numpy.typing as npt
from scipy.optimize import linprog
from giulia.logger import warning, error


class BaseScheduler:
    def __init__(self, 
                 cell_index: int,
                 cell_name: str, 
                 simulation_config_obj: Any,
                 network_deployment_obj: Any,
                 beam_conf_obj: Any,
                 best_serving_beam_per_ue_obj: Any,
                 dl_fast_fading_gain_ueAnt_to_cellAnt_obj: Any, 
                 baseStationsAll_Parent_obj:Any):
        """
        Base scheduler class for resource allocation.

        Args:
            cell_index (int): Index of the cell.
            cell_name (str): Name of the cell.
            simulation_config_obj (Any): Simulation configuration.
            network_deployment_obj (Any): Network deployment configuration.
            ue_deployment_obj (Any): UE deployment configuration.
            beam_conf_obj (Any): Beam configuration.
            best_serving_beam_per_ue_obj (Any): Best serving beam information.
            dl_fast_fading_gain_ueAnt_to_cellAnt_obj (Any): DL fast fading gains.
            resource_allocation (Dict[str, np.ndarray]): Resource allocation data structure.
        """
        # Input storage
        self.cell_index: int = cell_index
        self.cell_name: str = cell_name
        self.simulation_config_obj: Any = simulation_config_obj
        self.network_deployment_obj: Any = network_deployment_obj
        self.beam_conf_obj: Any = beam_conf_obj
        self.best_serving_beam_per_ue_obj: Any = best_serving_beam_per_ue_obj
        self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj: Any = dl_fast_fading_gain_ueAnt_to_cellAnt_obj
        self.baseStationsAll_Parent_obj: Any = baseStationsAll_Parent_obj

        # Random number generator
        self.rng: np.random.RandomState = np.random.RandomState(
            self.simulation_config_obj.random_seed + self.cell_index
        )

    def _allocate_resources(self, 
                            beam_ID: int, 
                            ue_IDs_in_beam: List[int], 
                            PRBs_required: npt.NDArray[np.int_], 
                            dl_PRBs_available: int) -> np.ndarray:
        """
        Abstract method for resource allocation. To be implemented by derived classes.
        """
        raise NotImplementedError("Subclasses must implement the _allocate_resources method.")

    def _update_resource_allocation(self, 
                                    allocation_matrix: np.ndarray, 
                                    ue_IDs_in_beam: List[int], 
                                    beam_ID: int) -> None:
        """
        Updates resource allocation based on the allocation matrix.

        Args:
            allocation_matrix (np.ndarray): Allocation matrix.
            ue_IDs_in_beam (List[int]): List of UE IDs in the beam.
            beam_ID (int): Beam ID.
        """
        for user_index, ue_ID in enumerate(ue_IDs_in_beam):
            allocated_PRBs = np.where(allocation_matrix[user_index])[0]

            # Update PRB activities
            self.baseStationsAll_Parent_obj.resource_allocation["PRB_cell_activity"][allocated_PRBs, self.beam_conf_obj.beam_to_node_mapping[beam_ID]] = True
            self.baseStationsAll_Parent_obj.resource_allocation["PRB_ue_activity"][allocated_PRBs, ue_ID] = True
            self.baseStationsAll_Parent_obj.resource_allocation["PRB_beam_activity"][allocated_PRBs, beam_ID] = True
            self.baseStationsAll_Parent_obj.resource_allocation["PRB_ue_beam_activity"][allocated_PRBs, ue_ID, beam_ID] = True

            # Update interference activity
            self.baseStationsAll_Parent_obj.resource_allocation["PRB_ue_beam_interference_activity"][allocated_PRBs, :, beam_ID] = True

            # Update UE IDs per PRB and beam
            self.baseStationsAll_Parent_obj.resource_allocation["ue_IDs_per_PRB_and_beam"][allocated_PRBs, beam_ID] = ue_ID

        # Create local copy of resource allocation to be store within scheders
        self.resource_allocation = self.baseStationsAll_Parent_obj.resource_allocation
        
        return 
        

    def _run(self) -> None:
        """
        Executes the scheduling process.
        """
            
        # Network
        self.dl_PRBs_available: int = self.baseStationsAll_Parent_obj.dl_PRBs_available_per_cell_dict[self.cell_name]

        # Identify beams
        # Get the indices of beams associated with the current cell
        cell_global_beams_IDs: np.ndarray = np.where(self.beam_conf_obj.beam_to_node_mapping == self.cell_index)[0]    
        
        for beam_ID in cell_global_beams_IDs:  # For each beam 
            # UEs in beam
            ue_IDs_in_beam: List[int] = np.where(
                self.best_serving_beam_per_ue_obj.best_serving_beam_ID_per_ue == beam_ID
            )[0].tolist()
            
            # Required PRBs 
            PRBs_required: npt.NDArray[np.int_] =\
                self.baseStationsAll_Parent_obj.resource_allocation["dl_PRBs_required_per_ue"][ue_IDs_in_beam]
            
            # Perform resource allocation
            allocation_matrix: np.ndarray = self._allocate_resources(
                beam_ID, ue_IDs_in_beam, PRBs_required, self.dl_PRBs_available
            )

            # Update resource allocation
            self._update_resource_allocation(allocation_matrix, ue_IDs_in_beam, beam_ID)  
            
        return
    
    @property
    def _prb_usage(self) -> float:
        """
        Returns the PRB usage for the cell.

        Returns:
            float: The PRB usage for the cell.
        """
        # Get the number of available PRBs for the specified cell
        dl_PRBs_available = self.baseStationsAll_Parent_obj.dl_PRBs_available_per_cell_dict[self.cell_name]

        # Get the index of the specified cell
        cell_idx = self.network_deployment_obj.map_cellName_to_cellIndex[self.cell_name]

        # Calculate the PRB usage as the mean activity over available PRBs
        prb_usage = self.resource_allocation['PRB_cell_activity'][:dl_PRBs_available][:, cell_idx].mean()

        return prb_usage

class SchedulerRandomPerBeam(BaseScheduler):
    def _allocate_resources(
        self,
        beam_ID: int,
        ue_IDs_in_beam: List[int],
        PRBs_required: npt.NDArray[np.int_],
        dl_PRBs_available: int
    ) -> np.ndarray:
        """
        Allocates PRBs to UEs in the beam using a random approach.

        Args:
            beam_ID (int): ID of the beam.
            ue_IDs_in_beam (List[int]): List of UEs associated with the beam.
            PRBs_required (np.ndarray): Array of PRBs required by each UE.
            dl_PRBs_available (int): Number of PRBs available in the beam.

        Returns:
            np.ndarray: Allocation matrix indicating PRB allocations per UE.
        """
        
        # Reset RNG for each beam to match the original behavior
        self.rng = np.random.RandomState(self.simulation_config_obj.random_seed + beam_ID)

        # Initialize allocation matrix
        allocation_matrix: np.ndarray = np.zeros((len(ue_IDs_in_beam), dl_PRBs_available), dtype=bool)

        # Ensure that the total required PRBs, per beam, do not exceed available PRBs
        assert all(PRBs_required <= dl_PRBs_available)

        # Perform allocation
        for user_index, PRBs_needed in enumerate(PRBs_required):

            # Skip UEs that do not require PRBs
            if PRBs_needed == 0:
                continue

            PRBs_free: npt.NDArray[np.int_] = np.where(~allocation_matrix.any(axis=0))[0]

            if len(PRBs_free) >= PRBs_needed:
                selected_resources: npt.NDArray[np.int_] = self.rng.choice(
                    PRBs_free, size=PRBs_needed, replace=False
                )
                allocation_matrix[user_index, selected_resources] = True
            else:
                warning(f"Warning: Not enough PRBs for UE {ue_IDs_in_beam[user_index]} in {self.cell_name}.")
        
        return allocation_matrix

class SchedulerSimplex(BaseScheduler):
    def _allocate_resources(
        self,
        beam_ID: int,
        ue_IDs_in_beam: List[int],
        PRBs_required: npt.NDArray[np.int_],
        dl_PRBs_available: int
    ) -> np.ndarray:
        """
        Allocates PRBs using linear programming with the simplex method.
        """
        channel_gain_matrix: np.ndarray = (
            self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj.fast_fading_channel_coeff_bAnt_to_aAnt_complex[
                :dl_PRBs_available, ue_IDs_in_beam, beam_ID
            ]
            .abs()
            .cpu()
            .numpy()
        )

        c: np.ndarray = -channel_gain_matrix.flatten()

        num_UEs: int = len(ue_IDs_in_beam)
        allocation_matrix: np.ndarray = np.zeros((num_UEs, dl_PRBs_available), dtype=bool)

        A_eq: np.ndarray = np.zeros((num_UEs, num_UEs * dl_PRBs_available))
        b_eq: npt.NDArray[np.int_] = PRBs_required
        for i in range(num_UEs):
            A_eq[i, i * dl_PRBs_available:(i + 1) * dl_PRBs_available] = 1

        A_ub: np.ndarray = np.zeros((dl_PRBs_available, num_UEs * dl_PRBs_available))
        b_ub: npt.NDArray[np.float64] = np.ones(dl_PRBs_available)
        for j in range(dl_PRBs_available):
            A_ub[j, j::dl_PRBs_available] = 1

        bounds = [(0, 1) for _ in range(num_UEs * dl_PRBs_available)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

        if result.success:
            allocation_matrix = result.x.reshape(num_UEs, dl_PRBs_available) > 0.5
        else:
            print(f"Error: Optimization failed for beam {beam_ID}.")

        return allocation_matrix


                        
        
                