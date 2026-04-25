#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:52:58 2024

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import time

import numpy as np

from giulia.plots import plotting
from giulia.tools import tools
from giulia.tools.tools import log_calculations_time


class Mobility:

    def __init__(self, 
                 simulation_config_obj, 
                 playground_deployment_obj,   
                 deployment_obj):

        ##### Plots
        ########################
        self.plot = 0 # Switch to control plots if any

        ##### Input storage
        ########################
        self.simulation_config_obj = simulation_config_obj
        self.playground_deployment_obj = playground_deployment_obj 
        self.deployment_obj = deployment_obj 
    
        ##### Input storage
        ########################
        self.updated_positions_b_m = []
        self.updated_velocity_b_kmh = []

    
    def process(self,rescheduling_us=-1): 
        
        ##### Process inputs
        ########################  
        
        # Random numbers
        self.rng = np.random.RandomState(self.simulation_config_obj.random_seed+0)     
        
        # Scenario 
        self.scenario_x_side_length_m = self.playground_deployment_obj.scenario_x_side_length_m
        self.scenario_y_side_length_m = self.playground_deployment_obj.scenario_y_side_length_m 
        self.scenario_lower_left_conner_m = self.playground_deployment_obj.scenario_lower_left_conner_m 
            
        # UEs
        self.position_z_a_m = self.deployment_obj.df_ep["position_z_m"].to_numpy(dtype=np.single)   
        
        # Users deployment 
        self.mobility_models = self.deployment_obj.df_ep["mobility_model"].to_numpy(dtype=str)  
        self.positions_b_m = self.deployment_obj.df_ep[["position_x_m", "position_y_m", "position_z_m"]].to_numpy(dtype=np.single) 
        self.velocity_b_kmh = self.deployment_obj.df_ep[["velocity_x_kmh", "velocity_y_kmh", "velocity_z_kmh"]].to_numpy(dtype=np.single) 
        self.indoor = self.deployment_obj.df_ep[["indoor"]].to_numpy(dtype=bool)   
        
        ##### Process outputs
        ########################  
        self.updated_positions_b_m = np.zeros((len(self.deployment_obj.df_ep),3),dtype = np.single)
        self.updated_velocity_b_kmh = np.zeros((len(self.deployment_obj.df_ep),3),dtype = np.single)


        ##### Start timer
        ########################        
        t_start = time.perf_counter() 
        
        
        ##### Switch
        ########################         
        
        # Find the set of unique mobility models to process them independently
        mobility_models_set = set(self.mobility_models)
        
        # Process each mobility model independently
        for mobility_model in mobility_models_set:  
            # Identify cells with the selected mobility model
            mask = self.mobility_models == mobility_model 
            
            # Get necessary information of the identified cells
            positions_b_m = self.positions_b_m[mask] 
            velocity_b_kmh = self.velocity_b_kmh[mask] 
            indoor = self.indoor[mask] 
            
            # Calculate path loss
            if (self.simulation_config_obj.ue_playground_model == "circular" and mobility_model == "straight_walk"):
                self.updated_positions_b_m[mask, :], self.updated_velocity_b_kmh[mask, :]  = \
                     self.straight_walk_circular(positions_b_m, velocity_b_kmh, indoor)
                
            elif ((self.simulation_config_obj.ue_playground_model == "rectangular" 
                        or self.simulation_config_obj.ue_playground_model == "3GPPTR36_814_Case_1_omni_dana")\
                    and mobility_model == "straight_walk"):
                
                self.updated_positions_b_m[mask, :], self.updated_velocity_b_kmh[mask, :]  = \
                     self.straight_walk_rectangular(positions_b_m, velocity_b_kmh, indoor)
                
            elif (self.simulation_config_obj.ue_playground_model == "rectangular" and mobility_model == "circular_walk"):
                self.updated_positions_b_m[mask, :], self.updated_velocity_b_kmh[mask, :]  \
                    = self.circular_walk_rectangular(positions_b_m, velocity_b_kmh, indoor)                
                
            else: 
                self.updated_positions_b_m[mask, :] = positions_b_m[mask]
                self.updated_velocity_b_kmh[mask, :] = velocity_b_kmh[mask] 
                
        # Store in data frames the results as it may be useful to post process
        self.deployment_obj.df_ep[["position_x_m", "position_y_m", "position_z_m"]] =  self.updated_positions_b_m
        self.deployment_obj.df_ep[["velocity_x_kmh", "velocity_y_kmh", "velocity_z_kmh"]] =  self.updated_velocity_b_kmh

        ##### End 
        log_calculations_time('UE position update', t_start)

        return 1e6  


    def straight_walk_circular(self, positions_b_m, velocity_b_kmh, indoor):
                
        # Define the circle center and radius
        scenario_center_m = self.playground_deployment_obj.scenario_centre_m  
        scenario_radius_m = self.playground_deployment_obj.scenario_radius_m
    
        # Update positions according to the velocity vector
        updated_positions_b_m = positions_b_m + tools.kmh_to_ms(velocity_b_kmh)
        
        # Create a new velocity array to possibly modify it
        new_velocity_b_kmh = velocity_b_kmh.copy()        
    
        # If UE is outside the bounderies of the scenario, it bounces back 
        # Check and handle boundaries
        # Calculate the distance of each position from the center (consider only x and y coordinates)
        distances_from_center = np.linalg.norm(updated_positions_b_m[:, 0:2] - scenario_center_m[0:2], axis=1)
    
        # Check if any position is outside the circle boundary
        mask = distances_from_center > scenario_radius_m
        
        if np.any(mask):
            # Get the positions that are outside the circle
            outside_positions = updated_positions_b_m[mask, 0:2]
    
            # Calculate the normal vectors at the boundary (from center to the outside positions)
            normal_vectors = outside_positions - scenario_center_m[0:2]
            normal_vectors /= np.linalg.norm(normal_vectors, axis=1, keepdims=True)
            
            # Note that this bouncing back mechanism will deform the clusters. 
            # Think of the case when UEs are moving (1,0) 
            # A user in the upper position of cluster will hit the upper part of the circualr boundary much earlier than one at the lower position of the cluster.
            # We would need a UE cluster logic to prevent this
    
            # Move the positions to the boundary
            updated_positions_b_m[mask, 0:2] = scenario_center_m[0:2] + normal_vectors * scenario_radius_m
            
            # Reverse the velocity vectors to point towards the inside of the circle
            new_velocity_b_kmh[mask,:] *= -1
    
        return updated_positions_b_m, new_velocity_b_kmh
    

    def straight_walk_rectangular(self, positions_b_m, velocity_b_kmh, indoor):
        
        # Define rectangle corners
        rect_min = self.playground_deployment_obj.scenario_lower_left_conner_m   # Bottom-left corner
        rect_max =  self.playground_deployment_obj.scenario_lower_left_conner_m + \
                        np.array([self.playground_deployment_obj.scenario_x_side_length_m,self.playground_deployment_obj.scenario_y_side_length_m]) # Top-right corner
        
        # Update positions according to velocity vector
        updated_positions_b_m = positions_b_m + tools.kmh_to_ms(velocity_b_kmh)
        
        # Create a new velocity array to possibly modify it
        new_velocity_b_kmh = velocity_b_kmh.copy()        
    
        # If UE is outside the bounderies of the scenario, it bounces back 
        # Check and handle boundaries
        for i in range(2):  # Iterate over x and y coordinates
            mask = updated_positions_b_m[:,i] < rect_min[i] # Check x coordinate 
            updated_positions_b_m[mask, i] = rect_min[i] # If it is smaller than the lower edge, move the positions to the boundary
            new_velocity_b_kmh[mask,i] *= -1  # Reverse the velocity component
              
            mask = updated_positions_b_m[:,i] > rect_max[i] # Check y coordinate
            updated_positions_b_m[mask,i] = rect_max[i] # If it is larger than the larger edge, move the positions to the boundary
            new_velocity_b_kmh[mask,i] *= -1  # Reverse the velocity component

        return updated_positions_b_m, new_velocity_b_kmh    


    def circular_walk_rectangular(self, positions_b_m, velocity_b_kmh, indoor):
        
        # Define rectangle corners
        rect_min = self.playground_deployment_obj.scenario_lower_left_conner_m # Bottom-left corner
        rect_max = self.playground_deployment_obj.scenario_lower_left_conner_m + \
                   np.array([self.playground_deployment_obj.scenario_x_side_length_m,self.playground_deployment_obj.scenario_y_side_length_m]) # Top-right corner
    
        # Update positions according to velocity vector
        updated_positions_b_m = positions_b_m + tools.kmh_to_ms(velocity_b_kmh)
    
        # Create a new velocity array to possibly modify it
        new_velocity_b_kmh = velocity_b_kmh.copy()
    
        # If UE is outside the boundaries of the scenario, it bounces back
        # Check and handle boundaries
        for i in range(2):  # Iterate over x and y coordinates
            mask = updated_positions_b_m[:, i] < rect_min[i]  # Check x coordinate
            updated_positions_b_m[mask, i] = rect_min[i]  # If it is smaller than the lower edge, move the positions to the boundary
            new_velocity_b_kmh[mask, i] *= -1  # Reverse the velocity component
    
            mask = updated_positions_b_m[:, i] > rect_max[i]  # Check y coordinate
            updated_positions_b_m[mask, i] = rect_max[i]  # If it is larger than the larger edge, move the positions to the boundary
            new_velocity_b_kmh[mask, i] *= -1  # Reverse the velocity component
    
        # Adjust velocity to follow a circular path
        for j in range(len(new_velocity_b_kmh)):
            angle = np.arctan2(new_velocity_b_kmh[j, 1], new_velocity_b_kmh[j, 0])  # Get current direction angle
            angle += np.deg2rad(5)  # Rotate by 5 degrees, you can change this value to adjust the circularity
            speed = np.linalg.norm(new_velocity_b_kmh[j])  # Get current speed
            new_velocity_b_kmh[j,:2] = np.array([np.cos(angle), np.sin(angle)]) * speed  # Update velocity to new direction with same speed
    
        return updated_positions_b_m, new_velocity_b_kmh