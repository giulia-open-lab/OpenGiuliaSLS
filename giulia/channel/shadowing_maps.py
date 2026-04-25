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
from os.path import join as pjoin
from typing import List

import numpy as np
import scipy.io as sio

from giulia.fs import shadowing_dir
from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable


class Shadowing_Map(Saveable):
    
    def __init__(self, 
                 plot, 
                 simulation_config_obj, 
                 network_deployment_obj):
        
        super().__init__()

        ##### Plots 
        ########################
        self.plot = plot # Switch to control plots if any
        
        ##### Input storage 
        ######################## 
        self.simulation_config_obj = simulation_config_obj 
        self.network_deployment_obj = network_deployment_obj
                
        ##### Outputs 
        ########################   
        # Placeholder to store shadowing maps
        
        self.shadowing_maps_dB = dict() 
        
        self.shadowing_map_scenario_resolution_m_dict = dict() 
        self.shadowing_map_scenario_offset_m_dict = dict()
        self.shadowing_map_number_of_maps_per_sigma_dict = dict()
        
        # Placeholder to store shadowing map to cell assignment
        self.shadowing_map_to_cell_assignment = np.ones((len(self.network_deployment_obj.df_ep)), dtype=int) 


    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["shadowing_map_to_cell_assignment"]
    
       
    def process(self, rescheduling_us=-1):
        
        # Process inputs
        self.rng = np.random.RandomState(self.simulation_config_obj.random_seed + 0)   
        
        self.bs_propagation_models = self.network_deployment_obj.df_ep["BS_propagation_model"].to_numpy()  
        self.bs_fast_channel_models = self.network_deployment_obj.df_ep["BS_fast_channel_model"].to_numpy()          
        self.cell_IDs = self.network_deployment_obj.df_ep["ID"].to_numpy(dtype=int)
        self.cell_names_IDs = self.network_deployment_obj.df_ep["name"].to_numpy()  
        self.cell_site_IDs =  self.network_deployment_obj.df_ep["site_ID"].to_numpy(dtype=int)               
       
        # Start timer       
        t_start = time.perf_counter()  
        
        # Find the set of unique propagation models to process them independently
        bs_propagation_models_set = set(self.bs_propagation_models)
        bs_fast_channel_models_set = set(self.bs_fast_channel_models)        
        
        # Process each propagation model independently 
            # Note that the code for all models is very slmilar. However, the different models have diffent charaterstics. Some have LoS/NLoS. Others have LoS/NLoS/OtoI. Others have LoS/NLoS/aerial. Thus, we have a block of code for each one of them.
            
        for models in itertools.product(bs_propagation_models_set, bs_fast_channel_models_set):  
            
            # Identify cells with the selected propagation model
            bs_propagation_model = models[0]
            bs_fast_channel_model = models[1]
            mask = np.bitwise_and(bs_propagation_model ==  self.bs_propagation_models, bs_fast_channel_model ==  self.bs_fast_channel_models)
            
            # Load shadowing information according to propagation model
            
            # Get .mat address
            data_dir = shadowing_dir()
            
            if (((bs_propagation_model == "3GPPTR38_901_UMa" and bs_fast_channel_model != "3GPPTR38_901_UMa"))\
                or bs_propagation_model == "ITU_R_M2135_UMa"\
                    or bs_propagation_model == "3GPPTR38_811_Urban_NTN"\
                        or bs_propagation_model == "3GPPTR38_811_Dense_Urban_NTN"):
            
                # Load .mat
                sigma_shadowing_LoS_dB = 4
                autocorrelation_LoS_m = 37
                mat_LoS_fname = pjoin(data_dir, 'G_shadowing_map_sigma_' + (str(sigma_shadowing_LoS_dB)).replace(".","_") + '_std_' + (str(autocorrelation_LoS_m)).replace(".","_") + '.mat')   
                mat_LoS = sio.loadmat(mat_LoS_fname) 
                shadowing_map_gain_LoS_dB = mat_LoS['G_shadow_map_dB']                
                
                sigma_shadowing_NLoS_dB = 6
                autocorrelation_NLoS_m = 50
                mat_NLoS_fname = pjoin(data_dir, 'G_shadowing_map_sigma_' + (str(sigma_shadowing_NLoS_dB)).replace(".","_") + '_std_' + (str(autocorrelation_NLoS_m)).replace(".","_") + '.mat')  
                mat_NLoS = sio.loadmat(mat_NLoS_fname) 
                shadowing_map_gain_NLoS_dB = mat_NLoS['G_shadow_map_dB']   
                
                # Identify cells with the selected propagation model
                mask = bs_propagation_model == self.bs_propagation_models                
            
                # Store some useful information
                self.shadowing_maps_dB[bs_propagation_model] = [shadowing_map_gain_LoS_dB, shadowing_map_gain_NLoS_dB]
                
                self.shadowing_map_scenario_resolution_m_dict[bs_propagation_model] = mat_LoS['scenario_resolution_m'][0,0] # Resolution of the shadowing map
                self.shadowing_map_scenario_offset_m_dict[bs_propagation_model] = mat_LoS['scenario_offset_m'][0] # Offset of the centre point
                self.shadowing_map_number_of_maps_per_sigma_dict[bs_propagation_model] = mat_LoS['number_of_maps_per_sigma'][0,0] # Number of shadowing maps per shadowing map standard deviation            
            

                # Assign shadowing maps to selected cells - here we are assigning the same map to the cells of the same cell site 
                self.shadowing_map_to_cell_assignment[mask] = self.assign_maps_to_cells(self.shadowing_map_number_of_maps_per_sigma_dict[bs_propagation_model], self.cell_site_IDs[mask])

            elif (bs_propagation_model == "3GPPTR38_901_UMi" and bs_fast_channel_model != "3GPPTR38_901_UMi"):
                # Load .mat
                sigma_shadowing_LoS_dB = 4
                autocorrelation_LoS_m = 10
                mat_LoS_fname = pjoin(data_dir, 'G_shadowing_map_sigma_' + (str(sigma_shadowing_LoS_dB)).replace(".","_") + '_std_' + (str(autocorrelation_LoS_m)).replace(".","_") + '.mat')   
                mat_LoS = sio.loadmat(mat_LoS_fname) 
                shadowing_map_gain_LoS_dB = mat_LoS['G_shadow_map_dB']                
                
                sigma_shadowing_NLoS_dB = 7.82
                autocorrelation_NLoS_m = 13
                mat_NLoS_fname = pjoin(data_dir, 'G_shadowing_map_sigma_' + (str(sigma_shadowing_NLoS_dB)).replace(".","_") + '_std_' + (str(autocorrelation_NLoS_m)).replace(".","_") + '.mat')  
                mat_NLoS = sio.loadmat(mat_NLoS_fname) 
                shadowing_map_gain_NLoS_dB =  mat_NLoS['G_shadow_map_dB']              
            
                # Identify cells with the selected propagation model
                mask = bs_propagation_model ==  self.bs_propagation_models
                
                # Store some useful information                
                self.shadowing_maps_dB[bs_propagation_model] = [shadowing_map_gain_LoS_dB, shadowing_map_gain_NLoS_dB]
                
                self.shadowing_map_scenario_resolution_m_dict[bs_propagation_model] = mat_LoS['scenario_resolution_m'][0,0] # Resolution of the shadowing map
                self.shadowing_map_scenario_offset_m_dict[bs_propagation_model] = mat_LoS['scenario_offset_m'][0] # Offset of the centre point
                self.shadowing_map_number_of_maps_per_sigma_dict[bs_propagation_model] = mat_LoS['number_of_maps_per_sigma'][0,0] # Number of shadowing maps per shadowing map standard deviation           
            
            
                # Assign shadowing maps to selected cells - here we are assigning the same map to the cells of the same cell site 
                self.shadowing_map_to_cell_assignment[mask] = self.assign_maps_to_cells(self.shadowing_map_number_of_maps_per_sigma_dict[bs_propagation_model], self.cell_site_IDs[mask])
            
            elif (bs_propagation_model == "ITU_R_M2135_UMi"):
                # Load .mat
                sigma_shadowing_LoS_dB = 3
                autocorrelation_LoS_m = 10
                mat_LoS_fname = pjoin(data_dir, 'G_shadowing_map_sigma_' + (str(sigma_shadowing_LoS_dB)).replace(".","_") + '_std_' + (str(autocorrelation_LoS_m)).replace(".","_") + '.mat')    
                mat_LoS = sio.loadmat(mat_LoS_fname) 
                shadowing_map_gain_LoS_dB = mat_LoS['G_shadow_map_dB']               
                
                sigma_shadowing_NLoS_dB = 4
                autocorrelation_NLoS_m = 13
                mat_NLoS_fname = pjoin(data_dir, 'G_shadowing_map_sigma_' + (str(sigma_shadowing_NLoS_dB)).replace(".","_") + '_std_' + (str(autocorrelation_NLoS_m)).replace(".","_") + '.mat') 
                mat_NLoS = sio.loadmat(mat_NLoS_fname) 
                shadowing_map_gain_NLoS_dB = mat_NLoS['G_shadow_map_dB']               
                
                sigma_shadowing_OtoI_dB = 7
                autocorrelation_OtoI_m =  7
                mat_OtoI_fname = pjoin(data_dir, 'G_shadowing_map_sigma_' + (str(sigma_shadowing_OtoI_dB)).replace(".","_") + '_std_' + (str(autocorrelation_OtoI_m)).replace(".","_") + '.mat')  
                mat_OtoI = sio.loadmat(mat_OtoI_fname) 
                shadowing_map_gain_OtoI_dB = mat_OtoI['G_shadow_map_dB'] 
                      
                # Identify cells with the selected propagation model
                mask = bs_propagation_model ==  self.bs_propagation_models  

                # Store some useful information
                self.shadowing_maps_dB[bs_propagation_model] = [shadowing_map_gain_LoS_dB, shadowing_map_gain_NLoS_dB, shadowing_map_gain_OtoI_dB]
                
                self.shadowing_map_scenario_resolution_m_dict[bs_propagation_model] = mat_LoS['scenario_resolution_m'][0,0] # Resolution of the shadowing map
                self.shadowing_map_scenario_offset_m_dict[bs_propagation_model] = mat_LoS['scenario_offset_m'][0] # Offset of the centre point
                self.shadowing_map_number_of_maps_per_sigma_dict[bs_propagation_model] = mat_LoS['number_of_maps_per_sigma'][0,0] # Number of shadowing maps per shadowing map standard deviation         
            
                # Assign shadowing maps to selected cells - here we are assigning the same map to the cells of the same cell site
                self.shadowing_map_to_cell_assignment[mask] = self.assign_maps_to_cells(self.shadowing_map_number_of_maps_per_sigma_dict[bs_propagation_model], self.cell_site_IDs[mask])         
                         
            elif (bs_propagation_model == "3GPPTR36_814_Case_1"):
                sigma_shadowing_NLoS_dB = 8
                autocorrelation_NLoS_m = 50
                mat_NLoS_fname = pjoin(data_dir, 'G_shadowing_map_sigma_' + (str(sigma_shadowing_NLoS_dB)).replace(".","_") + '_std_' + (str(autocorrelation_NLoS_m)).replace(".","_") + '.mat')  
                mat_NLoS = sio.loadmat(mat_NLoS_fname) 
                shadowing_map_gain_NLoS_dB = mat_NLoS['G_shadow_map_dB']   
                
                # Identify cells with the selected propagation model
                mask = bs_propagation_model ==  self.bs_propagation_models
                
                # Store some useful information
                self.shadowing_maps_dB[bs_propagation_model] = [shadowing_map_gain_NLoS_dB]
                
                self.shadowing_map_scenario_resolution_m_dict[bs_propagation_model] = mat_NLoS['scenario_resolution_m'][0,0] # Resolution of the shadowing map
                self.shadowing_map_scenario_offset_m_dict[bs_propagation_model] = mat_NLoS['scenario_offset_m'][0] # Offset of the centre point
                self.shadowing_map_number_of_maps_per_sigma_dict[bs_propagation_model] = mat_NLoS['number_of_maps_per_sigma'][0,0] # Number of shadowing maps per shadowing map standard deviation           
            
                # Assign shadowing maps to selected cells - here we are assigning the same map to the cells of the same cell site 
                self.shadowing_map_to_cell_assignment[mask] = self.assign_maps_to_cells(self.shadowing_map_number_of_maps_per_sigma_dict[bs_propagation_model], self.cell_site_IDs[mask])
                
            elif (bs_propagation_model == "3GPPTR36_777_UMa_AV"):
                # Load .mat
                sigma_shadowing_LoS_dB = 4
                autocorrelation_LoS_m = 37
                mat_LoS_fname = pjoin(data_dir, 'G_shadowing_map_sigma_' + (str(sigma_shadowing_LoS_dB)).replace(".","_") + '_std_' + (str(autocorrelation_LoS_m)).replace(".","_") + '.mat')    
                mat_LoS = sio.loadmat(mat_LoS_fname) 
                shadowing_map_gain_LoS_dB = mat_LoS['G_shadow_map_dB']   
                #
                self.shadowing_maps_dB[bs_propagation_model] = [shadowing_map_gain_LoS_dB]
                
                sigma_shadowing_NLoS_dB = 6
                autocorrelation_NLoS_m = 50
                mat_NLoS_fname = pjoin(data_dir, 'G_shadowing_map_sigma_' + (str(sigma_shadowing_NLoS_dB)).replace(".","_") + '_std_' + (str(autocorrelation_NLoS_m)).replace(".","_") + '.mat') 
                mat_NLoS = sio.loadmat(mat_NLoS_fname) 
                shadowing_map_gain_NLoS_dB =mat_NLoS['G_shadow_map_dB']   
                #
                self.shadowing_maps_dB[bs_propagation_model].append(shadowing_map_gain_NLoS_dB)
                
                av_height = self.simulation_config_obj.uav_height_m
                sigma_shadowing_aerial_LoS_dB = round(4.64*np.exp(-0.0066*av_height),2)
                autocorrelation_aerial_LoS_m =  37
                mat_aerial_LoS_fname = pjoin(data_dir, 'G_shadowing_map_sigma_' + (str(sigma_shadowing_aerial_LoS_dB)).replace(".","_") + '_std_' + (str(autocorrelation_aerial_LoS_m)).replace(".","_") + '.mat')  
                mat_aerial_LoS = sio.loadmat(mat_aerial_LoS_fname) 
                shadowing_map_gain_aerial_LoS_dB = mat_aerial_LoS['G_shadow_map_dB'] 
                #
                self.shadowing_maps_dB[bs_propagation_model].append(shadowing_map_gain_aerial_LoS_dB)  
                    
                # Identify cells with the selected propagation model
                mask = bs_propagation_model ==  self.bs_propagation_models                    

                # Store some useful information
                self.shadowing_map_scenario_resolution_m_dict[bs_propagation_model] = mat_LoS['scenario_resolution_m'][0,0] # Resolution of the shadowing map
                self.shadowing_map_scenario_offset_m_dict[bs_propagation_model] = mat_LoS['scenario_offset_m'][0] # Offset of the centre point
                self.shadowing_map_number_of_maps_per_sigma_dict[bs_propagation_model] = mat_LoS['number_of_maps_per_sigma'][0,0] # Number of shadowing maps per shadowing map standard deviation         
            
                # Assign shadowing maps to selected cells - here we are assigning the same map to the cells of the same cell site 
                self.shadowing_map_to_cell_assignment[mask] = self.assign_maps_to_cells(self.shadowing_map_number_of_maps_per_sigma_dict[bs_propagation_model], self.cell_site_IDs[mask])    

            elif (bs_propagation_model == "3GPPTR36_777_UMi_AV"):
                # Load .mat
                sigma_shadowing_LoS_dB = 4
                autocorrelation_LoS_m = 10
                mat_LoS_fname = pjoin(data_dir, 'G_shadowing_map_sigma_' + (str(sigma_shadowing_LoS_dB)).replace(".","_") + '_std_' + (str(autocorrelation_LoS_m)).replace(".","_") + '.mat')    
                mat_LoS = sio.loadmat(mat_LoS_fname) 
                shadowing_map_gain_LoS_dB = mat_LoS['G_shadow_map_dB']   
                #
                self.shadowing_maps_dB[bs_propagation_model] = [shadowing_map_gain_LoS_dB]                
                
                sigma_shadowing_NLoS_dB = 7.82
                autocorrelation_NLoS_m = 13
                mat_NLoS_fname = pjoin(data_dir, 'G_shadowing_map_sigma_' + (str(sigma_shadowing_NLoS_dB)).replace(".","_") + '_std_' + (str(autocorrelation_NLoS_m)).replace(".","_") + '.mat') 
                mat_NLoS = sio.loadmat(mat_NLoS_fname) 
                shadowing_map_gain_NLoS_dB = mat_NLoS['G_shadow_map_dB'] 
                #
                self.shadowing_maps_dB[bs_propagation_model].append(shadowing_map_gain_NLoS_dB)       
                
                av_height = self.simulation_config_obj.uav_height_m
                sigma_shadowing_aerial_LoS_dB = round(5*np.exp(-0.01*av_height),2)
                if sigma_shadowing_aerial_LoS_dB < 2:
                    sigma_shadowing_aerial_LoS_dB = 2    
                autocorrelation_aerial_LoS_m =  10
                mat_aerial_LoS_fname = pjoin(data_dir, 'G_shadowing_map_sigma_' + (str(sigma_shadowing_aerial_LoS_dB)).replace(".","_") + '_std_' + (str(autocorrelation_aerial_LoS_m)).replace(".","_") + '.mat')  
                mat_aerial_LoS = sio.loadmat(mat_aerial_LoS_fname) 
                shadowing_map_gain_aerial_LoS_dB = mat_aerial_LoS['G_shadow_map_dB']
                #
                self.shadowing_maps_dB[bs_propagation_model].append(shadowing_map_gain_aerial_LoS_dB)                     
                
                sigma_shadowing_aerial_NLoS_dB = 8
                autocorrelation_aerial_NLoS_m = 13
                mat_aerial_NLoS_fname = pjoin(data_dir, 'G_shadowing_map_sigma_' + (str(sigma_shadowing_aerial_NLoS_dB)).replace(".","_") + '_std_' + (str(autocorrelation_aerial_NLoS_m)).replace(".","_") + '.mat') 
                mat_aerial_NLoS = sio.loadmat(mat_aerial_NLoS_fname) 
                shadowing_map_gain_aerial_NLoS_dB = mat_aerial_NLoS['G_shadow_map_dB']   
                #
                self.shadowing_maps_dB[bs_propagation_model].append(shadowing_map_gain_aerial_NLoS_dB)   
                    
                # Identify cells with the selected propagation model
                mask = bs_propagation_model ==  self.bs_propagation_models                        
                
                # Store some useful information  
                self.shadowing_map_scenario_resolution_m_dict[bs_propagation_model] = mat_LoS['scenario_resolution_m'][0,0] # Resolution of the shadowing map
                self.shadowing_map_scenario_offset_m_dict[bs_propagation_model] = mat_LoS['scenario_offset_m'][0] # Offset of the centre point
                self.shadowing_map_number_of_maps_per_sigma_dict[bs_propagation_model] = mat_LoS['number_of_maps_per_sigma'][0,0] # Number of shadowing maps per shadowing map standard deviation     
            
            
                # Assign shadowing maps to selected cells - here we are assigning the same map to the cells of the same cell site 
                self.shadowing_map_to_cell_assignment[mask] = self.assign_maps_to_cells(self.shadowing_map_number_of_maps_per_sigma_dict[bs_propagation_model], self.cell_site_IDs[mask])                                       
                                
        ##### End
        log_calculations_time('Shadowing', t_start)
        
        return rescheduling_us
            
    
    def assign_maps_to_cells(self, shadowing_map_number_of_maps_per_sigma, cell_site_IDs):
        
        # Find the number of involved cell sites
        number_of_cell_sites = np.size(np.unique(cell_site_IDs))
        
        # Find the IDs of the involved cell sites and the indices of the unique array that reconstruct the input array 
        u, inverse = np.unique(cell_site_IDs,return_inverse=True)
        
        # Allocate a random shadowing map to each cell site from the pool of shadowing maps
        if shadowing_map_number_of_maps_per_sigma >= number_of_cell_sites:
            # If there are enough or more maps than cell sites, no repetitions needed
            rand_indices = self.rng.permutation(shadowing_map_number_of_maps_per_sigma)[:number_of_cell_sites]
        else:
            # If there are more cell sites than maps, allow repetitions
            rand_indices = self.rng.choice(shadowing_map_number_of_maps_per_sigma, size=number_of_cell_sites,
                                           replace=True)

        # Allocate the respective shadowing map to every cell - Cells of a cell site have the same map
        return np.take(rand_indices, inverse)