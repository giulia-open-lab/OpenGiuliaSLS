# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 14:13:46 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import itertools
import time
from typing import List

import numpy as np
import pandas as pd

from giulia.event_driven import Snapshot_control
from giulia.fs import results_file
from giulia.tools import tools
from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable


class Shadowing_Gain(Saveable):
    
    def __init__(self, 
                 simulation_config_obj, 
                 network_deployment_obj, 
                 ue_deployment_obj,
                 ue_playground_deployment_obj,
                 shadowing_map_cell_obj,
                 LoS_probability_ue_to_cell_obj):
        
        super().__init__()
        
        ##### Input storage 
        ######################## 
        self.simulation_config_obj = simulation_config_obj
        self.network_deployment_obj = network_deployment_obj
        self.ue_deployment_obj = ue_deployment_obj  
        self.ue_playground_deployment_obj = ue_playground_deployment_obj  
        self.shadowing_map_cell_obj = shadowing_map_cell_obj  
        self.LoS_probability_ue_to_cell_obj = LoS_probability_ue_to_cell_obj
        
        
        ##### Outputs 
        ########################   
        # Placeholder to store shadowing gain results        
        self.shadowing_gain_b_to_a_dB = []
        self.df_shadowing_gain_b_to_a_dB = []
        
        
    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["shadowing_gain_b_to_a_dB"]

       
    def process(self, rescheduling_us=-1):
        
        ##### Process inputs
        ######################## 
        
        # Random numbers
        self.rng = np.random.RandomState(self.simulation_config_obj.random_seed+0)   
        
        # Network
        self.bs_propagation_models = self.network_deployment_obj.df_ep["BS_propagation_model"].to_numpy()  
        self.bs_fast_channel_models = self.network_deployment_obj.df_ep["BS_fast_channel_model"].to_numpy()          
        self.cell_IDs = self.network_deployment_obj.df_ep["ID"].to_numpy(dtype=int)
        self.cell_names_IDs = self.network_deployment_obj.df_ep["name"].to_numpy()  
        self.cell_site_IDs =  self.network_deployment_obj.df_ep["site_ID"].to_numpy(dtype=int)  
        
        # Users deployment 
        self.ue_positions_m = self.ue_deployment_obj.df_ep[["position_x_m", "position_y_m", "position_z_m"]].to_numpy(dtype=np.single) 
        self.ue_indoor = self.ue_deployment_obj.df_ep[["indoor"]].to_numpy(dtype=bool)   
        self.ue_type = self.ue_deployment_obj.df_ep[["type"]].to_numpy()           
        
        # Channel characterisitics  
        self.LoS_b_to_a = self.LoS_probability_ue_to_cell_obj.los_b_to_a


        ##### Process outputs
        ########################
        self.shadowing_gain_b_to_a_dB = np.zeros((np.size(self.LoS_b_to_a,0), np.size(self.LoS_b_to_a,1)), dtype=np.single)         
       

        ##### Start timer
        ########################  
        t_start = time.perf_counter()
        
        
        ##### Switch
        ########################         
        
        # Find the set of unique propagation models to process them independently
        bs_propagation_models_set = set(self.bs_propagation_models)
        bs_fast_channel_models_set = set(self.bs_fast_channel_models)        
        
        # Process each propagation model independently 
            # Note that the code for all models is very slmilar. However, the different models have diffent charaterstics. Some have LoS/NLoS. Others have LoS/NLoS/OtoI. Others have LoS/NLoS/aerial. Thus, we have a block of code for each one of them.
            
        for models in itertools.product(bs_propagation_models_set, bs_fast_channel_models_set):  
            
            # Identify cells with the selected propagation model
            bs_propagation_model = models[0]
            bs_fast_channel_model = models[1]
            mask = np.bitwise_and(bs_propagation_model == self.bs_propagation_models, bs_fast_channel_model == self.bs_fast_channel_models)
            
            # Update shadowing_map_to_cell_assignment for the sake of shadow fading stochasticity 
            self.update_shadowFading_map_assignment(mask, bs_propagation_model)

            # Load shadowing information according to propagation model
            # Compute center of UE playground to re-center UE coordinates and read shadowing map
            ue_playground_center_x_m = self.ue_playground_deployment_obj.scenario_lower_left_conner_m[0] + self.ue_playground_deployment_obj.scenario_x_side_length_m / 2
            ue_playground_center_y_m = self.ue_playground_deployment_obj.scenario_lower_left_conner_m[1] + self.ue_playground_deployment_obj.scenario_y_side_length_m / 2
            ue_playground_center_m = (ue_playground_center_x_m, ue_playground_center_y_m)

            if (((bs_propagation_model == "3GPPTR38_901_UMa" and bs_fast_channel_model != "3GPPTR38_901_UMa"))\
                or bs_propagation_model == "ITU_R_M2135_UMa"\
                    or bs_propagation_model == "3GPPTR38_811_Urban_NTN"\
                        or bs_propagation_model == "3GPPTR38_811_Dense_Urban_NTN"):
            
                
                # Identify cells with the selected propagation model
                mask = bs_propagation_model ==  self.bs_propagation_models                
            
                # Store some useful information
                ue_types = ["LoS", "NLoS"]
                ue_masks = [self.LoS_b_to_a[:, mask] == True, self.LoS_b_to_a[:, mask] == False]
                
                # Assign shadowing values for the selected cells to UEs  
                self.shadowing_gain_b_to_a_dB[:, mask] = self.read_shadowing_from_map(np.size(self.ue_positions_m,0), 
                                                                                      np.sum(mask), 
                                                                                      self.ue_positions_m, 
                                                                                      ue_playground_center_m,
                                                                                      ue_types, 
                                                                                      ue_masks,
                                                                                      self.shadowing_map_cell_obj.shadowing_maps_dB[bs_propagation_model],                                                                                  
                                                                                      self.shadowing_map_cell_obj.shadowing_map_scenario_resolution_m_dict[bs_propagation_model], 
                                                                                      self.shadowing_map_cell_obj.shadowing_map_scenario_offset_m_dict[bs_propagation_model],                                                                                  
                                                                                      self.shadowing_map_cell_obj.shadowing_map_to_cell_assignment[mask])   
            
                
            elif (bs_propagation_model == "3GPPTR38_901_UMi" and bs_fast_channel_model != "3GPPTR38_901_UMi"):
                
                # Identify cells with the selected propagation model
                mask = bs_propagation_model ==  self.bs_propagation_models
                                
                # Store some useful information
                ue_types = ["LoS", "NLoS"]
                ue_masks = [self.LoS_b_to_a[:, mask] == True, self.LoS_b_to_a[:, mask] == False]
                               
                # Assign shadowing values for the selected cells to UEs 
                self.shadowing_gain_b_to_a_dB[:, mask] = self.read_shadowing_from_map(np.size(self.ue_positions_m,0), 
                                                                                      np.sum(mask), 
                                                                                      self.ue_positions_m,
                                                                                      ue_playground_center_m,
                                                                                      ue_types, 
                                                                                      ue_masks, 
                                                                                      self.shadowing_map_cell_obj.shadowing_maps_dB[bs_propagation_model],                                                                                  
                                                                                      self.shadowing_map_cell_obj.shadowing_map_scenario_resolution_m_dict[bs_propagation_model], 
                                                                                      self.shadowing_map_cell_obj.shadowing_map_scenario_offset_m_dict[bs_propagation_model],                                                                                  
                                                                                      self.shadowing_map_cell_obj.shadowing_map_to_cell_assignment[mask])                               
            
            elif (bs_propagation_model == "ITU_R_M2135_UMi"):
                                           
                # Identify cells with the selected propagation model
                mask = bs_propagation_model ==  self.bs_propagation_models                                         
                
                # Store some useful information
                
                ue_types = ["LoS", "NLoS", "OtoI"]
                ue_masks = [self.LoS_b_to_a[:, mask] == True, self.LoS_b_to_a[:, mask] == False, self.ue_indoor * np.ones(np.sum(mask)) == True]
                               
                # Assign shadowing values for the selected cells to UEs 
                self.shadowing_gain_b_to_a_dB[:, mask] = self.read_shadowing_from_map(np.size(self.ue_positions_m,0), 
                                                                                      np.sum(mask), 
                                                                                      self.ue_positions_m,
                                                                                      ue_playground_center_m,
                                                                                      ue_types, 
                                                                                      ue_masks, 
                                                                                      self.shadowing_map_cell_obj.shadowing_maps_dB[bs_propagation_model],                                                                                  
                                                                                      self.shadowing_map_cell_obj.shadowing_map_scenario_resolution_m_dict[bs_propagation_model], 
                                                                                      self.shadowing_map_cell_obj.shadowing_map_scenario_offset_m_dict[bs_propagation_model],                                                                                  
                                                                                      self.shadowing_map_cell_obj.shadowing_map_to_cell_assignment[mask]) 
                         
            elif (bs_propagation_model == "3GPPTR36_814_Case_1"):  
                
                # Identify cells with the selected propagation model
                mask = bs_propagation_model ==  self.bs_propagation_models
                    
                # Store some useful information
                ue_types = ["NLoS"]
                ue_masks = [self.LoS_b_to_a[:, mask] == False]
                
                # Assign shadowing values for the selected cells to UEs 
                
                self.shadowing_gain_b_to_a_dB[:, mask] = self.read_shadowing_from_map(np.size(self.ue_positions_m,0), 
                                                                                      np.sum(mask), 
                                                                                      self.ue_positions_m, 
                                                                                      ue_playground_center_m,
                                                                                      ue_types, 
                                                                                      ue_masks,
                                                                                      self.shadowing_map_cell_obj.shadowing_maps_dB[bs_propagation_model],                                                                                  
                                                                                      self.shadowing_map_cell_obj.shadowing_map_scenario_resolution_m_dict[bs_propagation_model], 
                                                                                      self.shadowing_map_cell_obj.shadowing_map_scenario_offset_m_dict[bs_propagation_model],                                                                                  
                                                                                      self.shadowing_map_cell_obj.shadowing_map_to_cell_assignment[mask]) 
                
            elif (bs_propagation_model == "3GPPTR36_777_UMa_AV"):
                
                # Identify cells with the selected propagation model
                mask = bs_propagation_model ==  self.bs_propagation_models
                
                # Store some useful information
                ue_types = ["LoS"]
                ue_masks = [np.logical_and(self.LoS_b_to_a[:, mask] == True, np.transpose(np.asmatrix(np.logical_and(self.ue_positions_m[:,2] >= 1.5, self.ue_positions_m[:,2] <= 22.5))))]

                ue_types.append("NLoS") 
                ue_masks.append(self.LoS_b_to_a[:, mask] == False)
                
                if np.sum(self.ue_type == "aerial") > 0:
                    ue_types.append("aerial_LoS") 
                    ue_masks.append(np.logical_and(self.LoS_b_to_a[:, mask] == True, np.transpose(np.asmatrix(np.logical_and(self.ue_positions_m[:,2] > 22.5, self.ue_positions_m[:,2] <= 300)))))                 
                  
                # Assign shadowing values for the selected cells to UEs 
                self.shadowing_gain_b_to_a_dB[:, mask] = self.read_shadowing_from_map(np.size(self.ue_positions_m,0), 
                                                                                      np.sum(mask), 
                                                                                      self.ue_positions_m, 
                                                                                      ue_playground_center_m,
                                                                                      ue_types, 
                                                                                      ue_masks,
                                                                                      self.shadowing_map_cell_obj.shadowing_maps_dB[bs_propagation_model],                                                                                  
                                                                                      self.shadowing_map_cell_obj.shadowing_map_scenario_resolution_m_dict[bs_propagation_model], 
                                                                                      self.shadowing_map_cell_obj.shadowing_map_scenario_offset_m_dict[bs_propagation_model],                                                                                  
                                                                                      self.shadowing_map_cell_obj.shadowing_map_to_cell_assignment[mask])     

            elif (bs_propagation_model == "3GPPTR36_777_UMi_AV"):
                
                # Identify cells with the selected propagation model
                mask = bs_propagation_model ==  self.bs_propagation_models  
                
                # Store some useful information                
                ue_types = ["LoS"]
                ue_masks = [np.logical_and(self.LoS_b_to_a[:, mask] == True, np.transpose(np.asmatrix(np.logical_and(self.ue_positions_m[:,2] >= 1.5, self.ue_positions_m[:,2] <= 22.5))))]              
                
                ue_types.append("NLoS")
                ue_masks.append(np.logical_and(self.LoS_b_to_a[:, mask] == False, np.transpose(np.asmatrix(np.logical_and(self.ue_positions_m[:,2] >= 1.5, self.ue_positions_m[:,2] <= 22.5)))))      
                
                if np.sum(self.ue_type == "aerial") > 0:                

                    ue_types.append("aerial_LoS")
                    ue_masks.append(np.logical_and(self.LoS_b_to_a[:, mask] == True, np.transpose(np.asmatrix(np.logical_and(self.ue_positions_m[:,2] > 22.5, self.ue_positions_m[:,2] <= 300)))))                  
                    
                    ue_types.append("aerial_NLoS")
                    ue_masks.append(np.logical_and(self.LoS_b_to_a[:, mask] == False, np.transpose(np.asmatrix(np.logical_and(self.ue_positions_m[:,2] > 22.5, self.ue_positions_m[:,2] <= 300)))))                     
                        
                # Assign shadowing values for the selected cells to UEs 
                self.shadowing_gain_b_to_a_dB[:, mask] = self.read_shadowing_from_map(np.size(self.ue_positions_m,0), 
                                                                                      np.sum(mask), 
                                                                                      self.ue_positions_m, 
                                                                                      ue_playground_center_m,
                                                                                      ue_types, 
                                                                                      ue_masks,
                                                                                      self.shadowing_map_cell_obj.shadowing_maps_dB[bs_propagation_model],                                                                                  
                                                                                      self.shadowing_map_cell_obj.shadowing_map_scenario_resolution_m_dict[bs_propagation_model], 
                                                                                      self.shadowing_map_cell_obj.shadowing_map_scenario_offset_m_dict[bs_propagation_model],                                                                                  
                                                                                      self.shadowing_map_cell_obj.shadowing_map_to_cell_assignment[mask])                                     
                    
        # Store in data frames the results as it may be useful to post process
        self.df_shadowing_gain_b_to_a_dB = pd.DataFrame(self.shadowing_gain_b_to_a_dB, columns=self.network_deployment_obj.df_ep["name"], index=self.ue_deployment_obj.df_ep["name"])
        
        
        ##### Save to plot
        ########################  
        
        snapshot_control = Snapshot_control.get_instance()
        if self.simulation_config_obj.plot == 1 and snapshot_control.num_snapshots == 0:
            file_name = results_file(self.simulation_config_obj.project_name, 'to_plot_shadowing_gain')
            np.savez(file_name, shadowing_gain_b_to_a_dB = self.shadowing_gain_b_to_a_dB)
            
            
        ##### End 
        log_calculations_time('Shadowing', t_start)
        
        return rescheduling_us
        

    # Function to get valid indices
    def get_valid_indices(self, indices, max_index):
        valid_indices = np.clip(indices, 0, max_index - 1)
        return valid_indices
   
    
    def read_shadowing_from_map(self, 
                                number_of_UEs, 
                                number_of_cells, 
                                ue_positions_m, 
                                ue_playground_center_m, 
                                ue_types, ue_masks, 
                                shadowing_maps_dB, 
                                shadowing_map_scenario_resolution_m, 
                                shadowing_map_lower_left_corner_m, 
                                shadowing_map_to_cell_assignment):
        
        # Find grid point indices according to UE position
        grid_point_index_x, grid_point_index_y = tools.translate_position_to_grid_point(ue_positions_m,
                                                                                        ue_playground_center_m,
                                                                                        shadowing_map_scenario_resolution_m,
                                                                                        shadowing_map_lower_left_corner_m)

        # Dimension indeces for queries such that we cover all cells for all UEs
        grid_point_index_x_aux = np.repeat(grid_point_index_x, repeats=number_of_cells, axis=0) 
        grid_point_index_y_aux = np.repeat(grid_point_index_y, repeats=number_of_cells, axis=0)   
        shadowing_map_to_cell_assignment_aux = np.tile(shadowing_map_to_cell_assignment, number_of_UEs)  
        
        # Initialize shadowing gain
        shadowing_gain_b_to_a_dB = np.ones((number_of_UEs,number_of_cells)) * np.nan
        
        for shadowing_maps_index in range(len(shadowing_maps_dB)):
            
            # Read shawdowing map
            shadowing_map_dB = shadowing_maps_dB[shadowing_maps_index]
            
            # Obtain the shadowing gain for every UE
            # Get valid indices
            valid_index_x = self.get_valid_indices(grid_point_index_x_aux, shadowing_map_dB.shape[0])
            valid_index_y = self.get_valid_indices(grid_point_index_y_aux, shadowing_map_dB.shape[1])
            
            # Access the array with the validated indices
            shadowing_gain_dB = shadowing_map_dB[valid_index_x, valid_index_y, shadowing_map_to_cell_assignment_aux]
            
            # Reshape to the orginal size
            shadowing_gain_dB = np.reshape(shadowing_gain_dB, (-1, number_of_cells))

            # Read mask
            mask = ue_masks[shadowing_maps_index]
            
            # Store shadwoing gain in global memory
            shadowing_gain_b_to_a_dB[mask] = shadowing_gain_dB[mask]

        return shadowing_gain_b_to_a_dB      
    
    
    def update_shadowFading_map_assignment(self, mask, bs_propagation_model):
        # If Sionna is not used, update shadow map assignment
        if not self.simulation_config_obj.sn_indicator:
            # Update shadowing_map_to_cell_assignment
            self.shadowing_map_cell_obj.shadowing_map_to_cell_assignment[mask] =\
                self.shadowing_map_cell_obj.assign_maps_to_cells(self.shadowing_map_cell_obj.shadowing_map_number_of_maps_per_sigma_dict[bs_propagation_model],
                                                                self.shadowing_map_cell_obj.cell_site_IDs[mask])
        # Else, do nothing. If Sionna is used ShadowFading is not considered
        else: 
            pass
        
        return True
      
                

             