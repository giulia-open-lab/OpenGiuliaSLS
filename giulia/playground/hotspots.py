#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 17:05:52 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import copy
import math
import os
import random
import time
from typing import List

import numpy as np

from giulia.playground import sites
from giulia.tools import tools
from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable

# Force CPU usage, prevent error with Mitsuba trying to use OptiX in load_scene()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

   
class Hotspot(Saveable):

    def __init__(self, 
                 simulation_config_obj, 
                 ue_playground_deployment_obj):
        
        super().__init__()

        ##### Plots
        ########################
        self.plot = 0 # Switch to control plots if any


        ##### Input storage
        ########################
        self.simulation_config_obj = simulation_config_obj
        self.ue_playground_deployment_obj = ue_playground_deployment_obj


        ##### Outputs
        ########################
        self.hotspot_position_m = []
        
        self.number_of_hotspot_per_cell = (0 * np.ones(self.ue_playground_deployment_obj.ref_number_of_cells)).astype(int) # 0 hotspots per cell   
        self.number_of_hotspots = 0 # 0 hotspots per cell  
        
        self.hotspot_radius_m = 0
        self.min_BS_to_hotspot_distance_m = 0
        self.min_hotspot_to_hotspot_distance_m = 0 
        
        self.faction_of_ues_in_hotspots_per_cell = (0 * np.ones(self.ue_playground_deployment_obj.ref_number_of_cells)) # 0 of UEs in a cell are in hotspots  
                   
        self.faction_of_ues_in_hotspots_per_cell = (0 * np.ones(self.ue_playground_deployment_obj.ref_number_of_cells)).astype(int) # 0 of UEs in a cell are in hotspots  
        self.faction_of_ues_in_hotspots = 0 # 0 of UEs in a cell are in hotspots  
        self.min_UE_to_hotspot_distance_m = 0


    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["faction_of_ues_in_hotspots_per_cell"]


    def process(self, rescheduling_us=-1):
        
        ##### Process inputs
        ######################## 
        
        # Random numbers
        self.myrandom = random.Random(self.simulation_config_obj.random_seed)
        self.rng = np.random.RandomState(self.simulation_config_obj.random_seed+0)        


        ##### Start timer
        ########################  
        t_start = time.perf_counter()


        ##### Switch 
        ########################
        if self.ue_playground_deployment_obj.playground_model == "ITU_R_M2135_UMa" or\
            self.ue_playground_deployment_obj.playground_model == "3GPPTR36_814_Case_1" or\
            self.ue_playground_deployment_obj.playground_model == "3GPPTR36_814_Case_1_omni" or \
            self.ue_playground_deployment_obj.playground_model == "3GPPTR36_814_Case_1_single_bs" or\
            self.ue_playground_deployment_obj.playground_model == "3GPPTR36_814_Case_1_single_bs_omni" or\
            self.ue_playground_deployment_obj.playground_model == "3GPPTR38_901_UMa_C1" or\
            self.ue_playground_deployment_obj.playground_model == "3GPPTR38_901_UMa_C2" or\
            self.ue_playground_deployment_obj.playground_model == "3GPPTR38_901_UMa_lsc" or\
            self.ue_playground_deployment_obj.playground_model == "3GPPTR38_901_UMa_lsc_single_bs" or\
            self.ue_playground_deployment_obj.playground_model == "3GPPTR38_901_UMa_lsc_single_sector" or\
            self.ue_playground_deployment_obj.playground_model == "3GPPTR38_901_UMa_lsc_sn" or \
            self.ue_playground_deployment_obj.playground_model == "3GPPTR36_777_UMa_AV" or \
            self.ue_playground_deployment_obj.playground_model == "3GPPTR38_811_Dense_Urban_HAPS" or \
            self.ue_playground_deployment_obj.playground_model == "3GPPTR38_811_Dense_Urban_NTN" or\
            self.ue_playground_deployment_obj.playground_model == "3GPPTR38_811_Urban_NTN" or \
            self.ue_playground_deployment_obj.playground_model == "ITU_R_M2135_UMa_multilayer" or\
            self.ue_playground_deployment_obj.playground_model == "ITU_R_M2135_UMa_Umi_noncolocated_multilayer" or\
            self.ue_playground_deployment_obj.playground_model == "ITU_R_M2135_UMa_Umi_colocated_multilayer":

            if self.ue_playground_deployment_obj.distribution == "uniform_with_hotspots" or\
                self.ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell_with_hotspots":
    
                self.number_of_hotspot_per_cell = (4 * np.ones(self.ue_playground_deployment_obj.ref_number_of_cells)).astype(int) # 4 hotspots per cell      
                self.number_of_hotspots = np.sum(self.number_of_hotspot_per_cell) 
                self.hotspot_position_m = np.full((self.number_of_hotspots,2), np.nan, dtype=np.single) 
                
                self.hotspot_radius_m = 40
                self.min_BS_to_hotspot_distance_m = self.ue_playground_deployment_obj.min_BS_to_UE_2D_distance_m + self.hotspot_radius_m   
                self.min_hotspot_to_hotspot_distance_m = 40 
                
                self.faction_of_ues_in_hotspots_per_cell = (2/3 * np.ones(self.ue_playground_deployment_obj.ref_number_of_cells)) # 2/3 of UEs in a cell are in hotspots  
                self.min_UE_to_hotspot_distance_m = 10
                
                # Calculate hotspot positions
                self.calculate_hotspot_locations_at_random_per_site_using_hexagon(self.ue_playground_deployment_obj)


        elif self.ue_playground_deployment_obj.playground_model == "ITU_R_M2135_UMi" or\
            self.ue_playground_deployment_obj.playground_model == "3GPPTR38_901_UMi_C1" or\
            self.ue_playground_deployment_obj.playground_model == "3GPPTR38_901_UMi_C2" or\
            self.ue_playground_deployment_obj.playground_model == "3GPPTR38_901_UMi_lsc" or\
            self.ue_playground_deployment_obj.playground_model == "3GPPTR36_777_UMi_AV":
                
            if self.ue_playground_deployment_obj.distribution == "uniform_with_hotspots" or\
                self.ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell_with_hotspots":                
                
                self.number_of_hotspot_per_cell = (4 * np.ones(self.ue_playground_deployment_obj.ref_number_of_cells)).astype(int) # 4 hotspots per cell  
                self.number_of_hotspots = np.sum(self.number_of_hotspot_per_cell) 
                self.hotspot_position_m = np.full((self.number_of_hotspots,2), np.nan, dtype=np.single) 
                
                self.hotspot_radius_m = 20  
                self.min_BS_to_hotspot_distance_m = self.ue_playground_deployment_obj.min_BS_to_UE_2D_distance_m + self.hotspot_radius_m   
                self.min_hotspot_to_hotspot_distance_m = 20       
                
                self.faction_of_ues_in_hotspots_per_cell = (2/3 * np.ones(self.ue_playground_deployment_obj.ref_number_of_cells)) # 2/3 of UEs in a cell are in hotspots  
                self.min_UE_to_hotspot_distance_m = 10               
                
                # Calculate hotspot positions
                self.calculate_hotspot_locations_at_random_per_site_using_hexagon(self.ue_playground_deployment_obj)


        elif self.ue_playground_deployment_obj.playground_model == "3GPPTR38_901_4G" or \
            self.ue_playground_deployment_obj.playground_model == "3GPPTR38_901_5G" or \
            self.ue_playground_deployment_obj.playground_model == "3GPPTR38_901_6G" or \
            self.ue_playground_deployment_obj.playground_model == "3GPPTR38_901_4G5G_multilayer" or \
            self.ue_playground_deployment_obj.playground_model == "3GPPTR38_901_4G_5G_multilayer" or \
            self.ue_playground_deployment_obj.playground_model == "3GPPTR38_901_4G_5G2_multilayer" or \
            self.ue_playground_deployment_obj.playground_model == "3GPPTR38_901_4G_5G6G_multilayer" or \
            self.ue_playground_deployment_obj.playground_model == "3GPPTR38_901_4G_5G_6G_multilayer" or \
            self.ue_playground_deployment_obj.playground_model == "3GPPTR38_901_4G5G_cell_reselection" : 
                
            if self.ue_playground_deployment_obj.distribution == "uniform_with_hotspots" or\
                self.ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell_with_hotspots":                  
            
                # Creates the reference circular playground for UE distribution
                self.circular_ue_playground_deployment_obj =\
                    sites.Site(False, "circular", "uniform_with_hotspots")  
                self.circular_ue_playground_deployment_obj.process() 
    
                self.number_of_hotspots = 19 # 0 hotspots per cell    
                self.hotspot_position_m = np.full((self.number_of_hotspots,2), np.nan, dtype=np.single) 
                
                self.hotspot_radius_m = 40  
                self.min_BS_to_hotspot_distance_m = self.circular_ue_playground_deployment_obj.min_BS_to_UE_2D_distance_m + self.hotspot_radius_m   
                self.min_hotspot_to_hotspot_distance_m = 100    
                
                self.faction_of_ues_in_hotspots = 1 # All UEs in a cell are in hotspots  
                self.min_UE_to_hotspot_distance_m = 10
                
                # Calculate hotspot positions
                self.calculate_hotspot_locations_at_random_within_circular_playground(self.circular_ue_playground_deployment_obj)                  


        elif self.ue_playground_deployment_obj.playground_model == "rectangular" or\
            self.ue_playground_deployment_obj.playground_model == "dataset_rectangular":
            
            if self.ue_playground_deployment_obj.distribution == "uniform_with_hotspots"\
                or self.ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell_with_hotspots":
    
                self.number_of_hotspots = 10 # 0 hotspots per cell    
                self.hotspot_position_m = np.full((self.number_of_hotspots,2), np.nan, dtype=np.single) 
                
                self.hotspot_radius_m = 40  
                self.min_BS_to_hotspot_distance_m = self.ue_playground_deployment_obj.min_BS_to_UE_2D_distance_m + self.hotspot_radius_m   
                self.min_hotspot_to_hotspot_distance_m = 100 
                
                self.faction_of_ues_in_hotspots = 1 # All UEs in a cell are in hotspots  
                self.min_UE_to_hotspot_distance_m = 10 
                
                # Calculate hotspot positions
                self.calculate_hotspot_locations_at_random_within_rectangular_playground(self.ue_playground_deployment_obj)


        elif self.ue_playground_deployment_obj.playground_model == "circular" or\
            self.ue_playground_deployment_obj.playground_model == "dataset_circular":
            
            if self.ue_playground_deployment_obj.distribution == "uniform_with_hotspots" or\
                self.ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell_with_hotspots":
    
                self.number_of_hotspots = 10 # 0 hotspots per cell    
                self.hotspot_position_m = np.full((self.number_of_hotspots,2), np.nan, dtype=np.single) 
                
                self.hotspot_radius_m = 40  
                self.min_BS_to_hotspot_distance_m = self.ue_playground_deployment_obj.min_BS_to_UE_2D_distance_m + self.hotspot_radius_m   
                self.min_hotspot_to_hotspot_distance_m = 100   
                
                self.faction_of_ues_in_hotspots = 1 # All UEs in a cell are in hotspots    
                self.min_UE_to_hotspot_distance_m = 10 
                
                # Calculate hotspot positions
                self.calculate_hotspot_locations_at_random_within_circular_playground(self.ue_playground_deployment_obj)


        elif self.ue_playground_deployment_obj.playground_model == "3GPPTR36_814_Case_1_omni_dana":
            
            self.number_of_hotspots = 3    
            self.hotspot_position_m = np.full((self.number_of_hotspots,2), np.nan, dtype=np.single) 
            self.hotspot_position_m[0,:] = np.array([-50,-100])
            self.hotspot_position_m[1,:] = np.array([50,0])
            self.hotspot_position_m[2,:] = np.array([-50,100])
            
            self.hotspot_radius_m = 5  
            self.min_BS_to_hotspot_distance_m = self.ue_playground_deployment_obj.min_BS_to_UE_2D_distance_m + self.hotspot_radius_m   
            self.min_hotspot_to_hotspot_distance_m = 150 
            
            self.faction_of_ues_in_hotspots = 1 # All UEs in a cell are in hotspots  
            self.min_UE_to_hotspot_distance_m = 20             
            
        ##### End
        ########################
        log_calculations_time('Hotspot deployment', t_start)
        
        return rescheduling_us


    def calculate_hotspot_locations_at_random_per_site_using_hexagon(self, ue_playground_deployment_obj):
        
       ### Compute hotspot locations - uniform deployment within the cell
       cell_id = int(0)
       hotspot_id = int(0)
       
       for site_index in range(ue_playground_deployment_obj.ref_number_of_cell_sites): #for all sites 
           angle_range_deg = 360/ue_playground_deployment_obj.ref_number_of_sectors_per_site
           
           for bearing_index in range(ue_playground_deployment_obj.ref_number_of_sectors_per_site): #for all sectors in the site 
               bearing = ue_playground_deployment_obj.ref_antenna_config_hor_alpha_mec_bearing_deg[bearing_index]
               ang_a = tools.angle_range_0_180(bearing-angle_range_deg/2)
               ang_b = tools.angle_range_0_180(bearing+angle_range_deg/2)
               
               ## HOTSPOT DEPLOYMENT WITHIN THE AREA
               hotspots_in_cell_count = self.number_of_hotspot_per_cell[cell_id]
               
               while hotspots_in_cell_count > 0 : #deploy as many hotspots as needed
                                                            
                   # Get location within hexagon of flat-top orientation with size = 1 (seee htps://www.redblobgames.com/grids/hexagons/)
                   hotspot_position = tools.randinunithex(self.myrandom, ue_playground_deployment_obj.isd_m)
               
                   # Derive angle and check whether it is within this the sector range
                   angle_deg = math.degrees(np.arctan2(hotspot_position[1],hotspot_position[0]))
                   
                   # Calculate distance to other hotspot centers
                   hotspot_to_hotspots_distance_m = np.linalg.norm(ue_playground_deployment_obj.ref_cell_site_positions_m[site_index] + hotspot_position - self.hotspot_position_m[:,None], axis=-1)
                   
                   # Check conditions 
                   if (tools.isBetween(ang_a, ang_b, angle_deg) 
                       and np.linalg.norm(hotspot_position,ord=2) >= self.min_BS_to_hotspot_distance_m # Min distance to cell site
                       and np.all(hotspot_to_hotspots_distance_m[~np.isnan(hotspot_to_hotspots_distance_m)] > self.min_hotspot_to_hotspot_distance_m)): # Min distance to other hotspots
                                          
                       # Store hotspot position
                       self.hotspot_position_m[hotspot_id] = ue_playground_deployment_obj.ref_cell_site_positions_m[site_index] + hotspot_position  
                       
                       # Update counters
                       hotspot_id += 1 
                       hotspots_in_cell_count -= 1                     
               
               # Update counters
               cell_id += 1               
       
       return


    def calculate_hotspot_locations_at_random_within_circular_playground(self, ue_playground_deployment_obj):  
        
       ### Compute hotspot locations - uniform deployment within the circular area 

       # Scenario features 
       scenario_center_m = ue_playground_deployment_obj.scenario_centre_m  
       scenario_radius_m = ue_playground_deployment_obj.scenario_radius_m
      
        ## HOTSPOT DEPLOYMENT WITHIN THE AREA
       if self.number_of_hotspots > 0:
           
           hotspot_id = int(0)
           hotspots_count = copy.copy(self.number_of_hotspots)

           while hotspots_count > 0 : #deploy as many hotspots as needed
           
                # Generate random angles
                angles_rad = self.rng.uniform(0, 2 * np.pi, 1)
               
                # Generate random radii (within the given radius)
                radii_m = np.sqrt(self.rng.uniform(0, 1, 1)) * (scenario_radius_m)   
               
                # Convert polar coordinates to Cartesian coordinates
                hotspot_position = scenario_center_m + np.column_stack(( (radii_m - self.hotspot_radius_m) * np.cos(angles_rad), radii_m * np.sin(angles_rad)))  
                
                # Calculate distance to other hotspot centers
                hotspot_to_hotspots_distance_m = np.linalg.norm(hotspot_position - self.hotspot_position_m[:,None], axis=-1) 
                
                # Check conditions 
                if (np.all(hotspot_to_hotspots_distance_m[~np.isnan(hotspot_to_hotspots_distance_m)] > self.min_hotspot_to_hotspot_distance_m)): # Min distance to other hotspots                
                
                    # Store hotspot position
                    self.hotspot_position_m[hotspot_id,:] = hotspot_position  
                
                    # Update counters
                    hotspot_id += 1 
                    hotspots_count -= 1
            
       return      
   

    def calculate_hotspot_locations_at_random_within_rectangular_playground(self, ue_playground_deployment_obj):
        
       ### Compute hotspot locations - uniform deployment within the rectangular area 
        
       # Scenario features 
       scenario_lower_left_conner_m = ue_playground_deployment_obj.scenario_lower_left_conner_m   
       scenario_x_side_length_m = ue_playground_deployment_obj.scenario_x_side_length_m 
       scenario_y_side_length_m = ue_playground_deployment_obj.scenario_y_side_length_m  
      
       ## HOTSPOT DEPLOYMENT WITHIN THE AREA
       if self.number_of_hotspots > 0:
           
           hotspot_id = int(0)
           hotspots_count = copy.copy(self.number_of_hotspots)

           while hotspots_count > 0 : #deploy as many hotspots as needed
           
                # Get location within area
                hotspot_position = \
                    np.column_stack(
                        (   self.rng.uniform(scenario_lower_left_conner_m[0] + self.hotspot_radius_m, 
                                             scenario_lower_left_conner_m[0] + scenario_x_side_length_m - self.hotspot_radius_m , 1),

                            self.rng.uniform( scenario_lower_left_conner_m[1] + self.hotspot_radius_m, 
                                             scenario_lower_left_conner_m[1] + scenario_y_side_length_m - self.hotspot_radius_m, 1)
                        ))
                
                # Calculate distance to other hotspot centers
                hotspot_to_hotspots_distance_m = np.linalg.norm(hotspot_position - self.hotspot_position_m[:,None], axis=-1) 
                
                # Check conditions 
                if (np.all(hotspot_to_hotspots_distance_m[~np.isnan(hotspot_to_hotspots_distance_m)] > self.min_hotspot_to_hotspot_distance_m)): # Min distance to other hotspots                
                
                    # Store hotspot position
                    self.hotspot_position_m[hotspot_id,:] = hotspot_position  
                       
                    # Update counters
                    hotspot_id += 1 
                    hotspots_count -= 1
          
       return