# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:05:46 2024

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

from typing import List, Any, Dict

import sys
import numpy as np
import numpy.typing as npt
from scipy.optimize import linprog
from giulia.logger import warning, error


class Scheduler:
    
    def __init__(self, 
                 simulation_config_obj,
                 network_deployment_obj, 
                 ue_deployment_obj,
                 beam_conf_obj,
                 traffic_generator_obj,
                 best_serving_beam_per_ue_obj,
                 dl_fast_fading_gain_ueAnt_to_cellAnt_obj):
        
        
        ##### Input storage 
        ######################## 
        self.simulation_config_obj = simulation_config_obj
        self.network_deployment_obj = network_deployment_obj
        self.ue_deployment_obj = ue_deployment_obj  
        self.beam_conf_obj = beam_conf_obj
        self.traffic_generator_obj = traffic_generator_obj
        self.best_serving_beam_per_ue_obj = best_serving_beam_per_ue_obj
        self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj = dl_fast_fading_gain_ueAnt_to_cellAnt_obj
        
                
        ##### Outputs 
        ########################   
        # Placeholder to store the number of DL PRBs allocated to each user
        self.ue_IDs_per_PRB_and_beam = []
        
        self.PRB_cell_activity = []
        self.PRB_ue_activity = []
        self.PRB_beam_activity =[]
        self.PRB_ue_beam_activity = [] 
        self.PRB_ue_beam_interference_activity =[]

       
    def process(self, rescheduling_us=-1):
        
        ##### Process inputs
        ########################

        # Random numbers 
        self.rng = np.random.RandomState(self.simulation_config_obj.random_seed+0)    
    
        # Network 
        self.scheduling_models = self.network_deployment_obj.df_ep["BS_scheduling_model"].to_numpy()  
        self.dl_PRBs_available = self.network_deployment_obj.df_ep["dl_PRBs_available"].to_numpy(dtype=int) # Size number of cells    
        
        ##### Process outputs
        ########################
        self.ue_IDs_per_PRB_and_beam = np.full((self.dl_PRBs_available[0], np.sum(self.beam_conf_obj.number_of_beams_per_node)), -1, dtype=int)
        
        self.PRB_cell_activity = np.zeros((self.dl_PRBs_available[0], len(self.network_deployment_obj.df_ep)), dtype=bool)
        self.PRB_ue_activity = np.zeros((self.dl_PRBs_available[0], len(self.ue_deployment_obj.df_ep)), dtype=bool)
        self.PRB_beam_activity = np.zeros((self.dl_PRBs_available[0], np.sum(self.beam_conf_obj.number_of_beams_per_node)), dtype=bool)
        self.PRB_ue_beam_activity = \
            np.zeros((self.dl_PRBs_available[0], len(self.ue_deployment_obj.df_ep), np.sum(self.beam_conf_obj.number_of_beams_per_node)), dtype=bool)
        self.PRB_ue_beam_interference_activity = \
            np.zeros((self.dl_PRBs_available[0], len(self.ue_deployment_obj.df_ep), np.sum(self.beam_conf_obj.number_of_beams_per_node)), dtype=bool)        

        ##### Switch
        ######################## 
  
        # Find the set of scheduling models to process them independently
        scheduling_models_set = set(self.scheduling_models) 
        
        # Calculate the number of DL PRBs available per beam 
        self.dl_PRBs_available_per_beam = np.repeat(self.dl_PRBs_available, self.beam_conf_obj.number_of_beams_per_node) # Size number of beams          
        
        # Process each scheduling model independently
        for scheduling_model in scheduling_models_set:
            
            # Identify cells with the selected scheduling model
            selected_cell_mask = scheduling_model ==  self.scheduling_models # Size of cells 
            selected_beam_mask = np.repeat(selected_cell_mask, self.beam_conf_obj.number_of_beams_per_node) # Size of beams
            
            # Get necessary information of the identified beams
            dl_PRBs_available_per_beam = self.dl_PRBs_available_per_beam[selected_beam_mask]
            beam_to_node_mapping = self.beam_conf_obj.beam_to_node_mapping [selected_beam_mask]       
            
            if scheduling_model == "random_per_cell_beam":
                for beam_index in range(np.size(dl_PRBs_available_per_beam, 0)):  # For all beams
                
                    self.rng = np.random.RandomState(self.simulation_config_obj.random_seed+beam_index)    
                
                    # Get UEs associated with this beam and the PRBs available
                    ue_IDs_in_beam = np.where(self.best_serving_beam_per_ue_obj.best_serving_beam_ID_per_ue == beam_index)[0]
                    PRBs_available = dl_PRBs_available_per_beam[beam_index]
                    PRBs_required = self.traffic_generator_obj.dl_PRBs_required_per_ue[ue_IDs_in_beam]
            
                    # Initialize allocation variables
                    num_UEs = len(ue_IDs_in_beam)
                    allocation_matrix = np.zeros((num_UEs, PRBs_available), dtype=bool)  # Tracks PRB allocations per UE
            
                    for user_index, PRBs_needed in enumerate(PRBs_required):
                        # Find unallocated PRBs
                        PRBs_free = np.where(~allocation_matrix.any(axis=0))[0]
            
                        # Check if enough PRBs are available for allocation
                        if len(PRBs_free) >= PRBs_needed:
                            # Randomly allocate the required PRBs
                            selected_resources = self.rng.choice(PRBs_free, size=PRBs_needed, replace=False)
                            allocation_matrix[user_index, selected_resources] = True  # Mark allocated PRBs
                            
                        else:
                            # Warning if PRBs needed exceed available resources
                            warning(f"Warning: Not enough PRBs for UE {ue_IDs_in_beam[user_index]} in beam {beam_index}.")
                                        
                    # Update activities and outputs based on allocation_matrix
                    for user_index, ue_ID in enumerate(ue_IDs_in_beam):
                        allocated_PRBs = np.where(allocation_matrix[user_index])[0]
                        self.PRB_cell_activity[allocated_PRBs, beam_to_node_mapping[beam_index]] = True
                        self.PRB_ue_activity[allocated_PRBs, ue_ID] = True
                        self.PRB_beam_activity[allocated_PRBs, beam_index] = True
                        self.PRB_ue_beam_activity[allocated_PRBs, ue_ID, beam_index] = True
                        self.PRB_ue_beam_interference_activity[allocated_PRBs, :, beam_index] = True
                        self.ue_IDs_per_PRB_and_beam[allocated_PRBs, beam_index] = ue_ID
                        

            elif scheduling_model == "simplex_based":
                for beam_index in range(np.size(dl_PRBs_available_per_beam, 0)):  # For all beams
                
                    self.rng = np.random.RandomState(self.simulation_config_obj.random_seed+beam_index)    
                
                    # Get UEs associated with this beam and the PRBs available
                    ue_IDs_in_beam = np.where(self.best_serving_beam_per_ue_obj.best_serving_beam_ID_per_ue == beam_index)[0]
                    PRBs_available = dl_PRBs_available_per_beam[beam_index]
                    PRBs_required = self.traffic_generator_obj.dl_PRBs_required_per_ue[ue_IDs_in_beam]
            
                    # Check if the total PRB demand exceeds the available PRBs
                    if np.sum(PRBs_required) > PRBs_available:
                        warning(f"Warning: Total PRB demand exceeds availability in beam {beam_index}.")
                    
                    # Constructing the optimization problem
                    # Convert PyTorch tensor to NumPy array for optimization
                    channel_gain_matrix = (
                        self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj.fast_fading_channel_coeff_bAnt_to_aAnt_complex[:PRBs_available,ue_IDs_in_beam, beam_index ]
                        .abs()  # Compute the magnitude
                        .cpu()  # Ensure it's on CPU if necessary
                        .numpy()  # Convert to NumPy array
                    )
            
                    # Objective: Maximize channel gain
                    c = -channel_gain_matrix.flatten()  # Minimize negative gains for maximization
            
                    # Equality constraints: Ensure PRBs allocated match the requirements of each UE
                    num_UEs = len(ue_IDs_in_beam)
                    A_eq = np.zeros((num_UEs, num_UEs * PRBs_available))
                    b_eq = PRBs_required
                    for i in range(num_UEs):
                        A_eq[i, i * PRBs_available:(i + 1) * PRBs_available] = 1
            
                    # Inequality constraints: Ensure no PRB is assigned to more than one UE
                    A_ub = np.zeros((PRBs_available, num_UEs * PRBs_available))
                    b_ub = np.ones(PRBs_available)
                    for j in range(PRBs_available):
                        A_ub[j, j::PRBs_available] = 1
            
                    # Bounds for variables: Each allocation variable is between 0 and 1
                    bounds = [(0, 1) for _ in range(num_UEs * PRBs_available)]
            
                    # Solve the linear program
                    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
            
                    # Check and update allocation results if the solver succeeded
                    if result.success:
                        allocation = result.x.reshape(num_UEs, PRBs_available)
                        for user_index, ue_ID in enumerate(ue_IDs_in_beam):
                            allocated_PRBs = np.where(allocation[user_index] > 0.5)[0]
                            self.PRB_cell_activity[allocated_PRBs, beam_to_node_mapping[beam_index]] = True
                            self.PRB_ue_activity[allocated_PRBs, ue_ID] = True
                            self.PRB_beam_activity[allocated_PRBs, beam_index] = True
                            self.PRB_ue_beam_activity[allocated_PRBs, ue_ID, beam_index] = True
                            self.PRB_ue_beam_interference_activity[allocated_PRBs, :, beam_index] = True
                            self.ue_IDs_per_PRB_and_beam[allocated_PRBs, beam_index] = ue_ID
                    else:
                        error(f"Error: Optimization failed for beam {beam_index}.")
            
            else:
                # Error handling for unsupported scheduling models
                np.disp("Error: The scheduling model specified does not exist.")
                sys.exit(0)          

        return rescheduling_us    


class BaseScheduler:
    def __init__(self, 
                 cell_index: int,
                 cell_name: str, 
                 simulation_config_obj: Any,
                 network_deployment_obj: Any,
                 beam_conf_obj: Any,
                 best_serving_beam_per_ue_obj: Any,
                 dl_fast_fading_gain_ueAnt_to_cellAnt_obj: Any,
                 resource_allocation: Dict[str, np.ndarray],
                 prb_requirement_generator_ue_obj):
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
        self.resource_allocation: Dict[str, np.ndarray] = resource_allocation
        self.prb_requirement_generator_ue_obj = prb_requirement_generator_ue_obj

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
            self.resource_allocation["PRB_cell_activity"][allocated_PRBs, self.beam_conf_obj.beam_to_node_mapping[beam_ID]] = True
            self.resource_allocation["PRB_ue_activity"][allocated_PRBs, ue_ID] = True
            self.resource_allocation["PRB_beam_activity"][allocated_PRBs, beam_ID] = True
            self.resource_allocation["PRB_ue_beam_activity"][allocated_PRBs, ue_ID, beam_ID] = True

            # Update interference activity
            self.resource_allocation["PRB_ue_beam_interference_activity"][allocated_PRBs, :, beam_ID] = True

            # Update UE IDs per PRB and beam
            self.resource_allocation["ue_IDs_per_PRB_and_beam"][allocated_PRBs, beam_ID] = ue_ID

    def _run(self) -> None:
        """
        Executes the scheduling process.
        """
        # Network
        dl_PRBs_available: int = self.network_deployment_obj.df_ep["dl_PRBs_available"].iloc[self.cell_index]

        # Identify beams
        # Get the indices of beams associated with the current cell
        cell_global_beams_IDs: np.ndarray = np.where(self.beam_conf_obj.beam_to_node_mapping == self.cell_index)[0]    
        
        for beam_ID in cell_global_beams_IDs:  # For each beam 
            # UEs in beam
            ue_IDs_in_beam: List[int] = np.where(
                self.best_serving_beam_per_ue_obj.best_serving_beam_ID_per_ue == beam_ID
            )[0].tolist()
            
            # Required PRBs 
            PRBs_required: npt.NDArray[np.int_] = self.resource_allocation["dl_PRBs_required_per_ue"][ue_IDs_in_beam]
            
            # Perform resource allocation
            allocation_matrix: np.ndarray = \
                self._allocate_resources(beam_ID, ue_IDs_in_beam, PRBs_required, dl_PRBs_available)

            # Update resource allocation
            self._update_resource_allocation(allocation_matrix, ue_IDs_in_beam, beam_ID)


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

        # Perform allocation
        for user_index, PRBs_needed in enumerate(PRBs_required):
            PRBs_free: npt.NDArray[np.int_] = np.where(~allocation_matrix.any(axis=0))[0]

            if len(PRBs_free) >= PRBs_needed:
                selected_resources: npt.NDArray[np.int_] = \
                    self.rng.choice(PRBs_free, size=PRBs_needed, replace=False)
                allocation_matrix[user_index, selected_resources] = True
            else:
                print(f"Warning: Not enough PRBs for UE {ue_IDs_in_beam[user_index]} in beam {beam_ID}.")
        
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
        b_ub: npt.NDArray[np.float_] = np.ones(dl_PRBs_available)
        for j in range(dl_PRBs_available):
            A_ub[j, j::dl_PRBs_available] = 1

        bounds = [(0, 1) for _ in range(num_UEs * dl_PRBs_available)]

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

        if result.success:
            allocation_matrix = result.x.reshape(num_UEs, dl_PRBs_available) > 0.5
        else:
            print(f"Error: Optimization failed for beam {beam_ID}.")

        return allocation_matrix          