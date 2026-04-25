# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:40:53 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import itertools
import sys
import time
from typing import List

import astropy.units as u
import itur
import numpy as np
import pandas as pd
import scipy.constants

from giulia.event_driven import Snapshot_control
from giulia.fs import results_file
from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable


class Path_Loss(Saveable):
    
    def __init__(self, 
                 simulation_config_obj, 
                 network_deployment_obj, 
                 ue_deployment_obj,
                 distance_angles_ue_to_cell_obj,
                 LoS_probability_ue_to_cell_obj):
        
        super().__init__()

        ##### Input storage 
        ######################## 
        self.simulation_config_obj = simulation_config_obj
        self.network_deployment_obj = network_deployment_obj
        self.ue_deployment_obj = ue_deployment_obj    
        self.distance_angles_ue_to_cell_obj = distance_angles_ue_to_cell_obj
        self.LoS_probability_ue_to_cell_obj = LoS_probability_ue_to_cell_obj
        
        
        ##### Outputs 
        ########################   
        # Place holder to store path loss results
        
        self.path_loss_b_to_a_dB = []
        self.df_path_loss_b_to_a_dB = []
        

    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["path_loss_b_to_a_dB"]

       
    def process(self, rescheduling_us=-1): 
        
        ##### Process inputs
        ########################  
        
        # Random numbers
        self.rng = np.random.RandomState(self.simulation_config_obj.random_seed+0)        
            
        # Network
        self.bs_propagation_models = self.network_deployment_obj.df_ep["BS_propagation_model"].to_numpy()
        self.bs_fast_channel_models = self.network_deployment_obj.df_ep["BS_fast_channel_model"].to_numpy()        
        self.dl_carrier_frequency_GHz = self.network_deployment_obj.df_ep["dl_carrier_frequency_GHz"].to_numpy(dtype=np.single)   
        self.position_z_a_m = self.network_deployment_obj.df_ep["position_z_m"].to_numpy(dtype=np.single)   
        self.site_ID = self.network_deployment_obj.df_ep["site_ID"].to_numpy()
        
        # Users deployment 
        self.position_z_b_m = self.ue_deployment_obj.df_ep["position_z_m"].to_numpy(dtype=np.single)  
        self.indoor = self.ue_deployment_obj.df_ep[["indoor"]].to_numpy(dtype=bool)      
        
        # Channel characterisitics  
        self.zeniths_b_to_a_wraparound_degrees = self.distance_angles_ue_to_cell_obj.zeniths_b_to_a_wraparound_degrees
        self.distance_b_to_a_2d_m = self.distance_angles_ue_to_cell_obj.distance_b_to_a_2d_wraparound_m
        self.distance_b_to_a_3d_m = self.distance_angles_ue_to_cell_obj.distance_b_to_a_3d_wraparound_m  
        self.d_2D_in_m = self.distance_angles_ue_to_cell_obj.d_2D_in_m
        self.LoS_b_to_a = self.LoS_probability_ue_to_cell_obj.los_b_to_a
        
        
        ##### Process outputs
        ########################
        self.path_loss_b_to_a_dB = np.full((np.size(self.distance_b_to_a_3d_m,0), np.size(self.distance_b_to_a_3d_m,1)) , np.nan)
       
        
        ##### Start timer
        ########################        
        t_start = time.perf_counter() 
        
        
        ##### Switch
        ########################         
        
        # Find the set of unique propagation models to process them independently
        bs_propagation_models_set = set(self.bs_propagation_models)
        bs_fast_channel_models_set = set(self.bs_fast_channel_models)
        
        # Process each propagation model independently
            
        for models in itertools.product(bs_propagation_models_set, bs_fast_channel_models_set):  
            
            # Identify cells with the selected propagation model
            bs_propagation_model = models[0]
            bs_fast_channel_model = models[1]
            mask = np.bitwise_and(bs_propagation_model ==  self.bs_propagation_models, bs_fast_channel_model ==  self.bs_fast_channel_models)
            
            # Get necessary information of the identified cells
            dl_carrier_frequency_GHz = self.dl_carrier_frequency_GHz[mask] 
            
            position_z_a_m = self.position_z_a_m[mask] 
            site_ID = self.site_ID[mask] 
            
            zeniths_b_to_a_wraparound_degrees = self.zeniths_b_to_a_wraparound_degrees[:, mask]
            
            distance_b_to_a_2d_m = self.distance_b_to_a_2d_m[:, mask]
            distance_b_to_a_3d_m = self.distance_b_to_a_3d_m[:, mask]
            d_2D_in_m = self.d_2D_in_m[:, mask]
            
            LoS_b_to_a = self.LoS_b_to_a[:, mask]
            
            # Calculate path loss
            if (bs_propagation_model == "3GPPTR38_901_UMa" and bs_fast_channel_model != "3GPPTR38_901_UMa"):
                self.path_loss_b_to_a_dB[:, mask] = \
                    self.path_loss_3GPPTR38_901_UMa(dl_carrier_frequency_GHz, distance_b_to_a_2d_m, distance_b_to_a_3d_m, LoS_b_to_a, position_z_a_m, self.position_z_b_m, site_ID)
                
            elif (bs_propagation_model == "3GPPTR38_901_UMi" and bs_fast_channel_model != "3GPPTR38_901_UMi"):
                self.path_loss_b_to_a_dB[:, mask] = \
                    self.path_loss_3GPPTR38_901_UMi(dl_carrier_frequency_GHz, distance_b_to_a_2d_m, distance_b_to_a_3d_m, LoS_b_to_a, position_z_a_m, self.position_z_b_m)
            
            elif (bs_propagation_model == "ITU_R_M2135_UMa"):
                self.path_loss_b_to_a_dB[:, mask] = \
                    self.path_loss_ITU_R_M2135_UMa(dl_carrier_frequency_GHz, distance_b_to_a_3d_m, LoS_b_to_a, position_z_a_m, self.position_z_b_m, site_ID)
                
            elif (bs_propagation_model == "ITU_R_M2135_UMi"):
                self.path_loss_b_to_a_dB[:, mask] = \
                    self.path_loss_ITU_R_M2135_UMi(dl_carrier_frequency_GHz, distance_b_to_a_3d_m, d_2D_in_m, LoS_b_to_a, position_z_a_m, self.position_z_b_m, self.indoor)
                
            elif (bs_propagation_model == "3GPPTR36_814_Case_1"):
                self.path_loss_b_to_a_dB[:, mask] = \
                    self.path_loss_3GPPTR36_814_Case_1(dl_carrier_frequency_GHz, distance_b_to_a_3d_m)
                
            elif (bs_propagation_model == "3GPPTR36_777_UMa_AV"):
                self.path_loss_b_to_a_dB[:, mask] = \
                    self.path_loss_3GPPTR36_777_UMa_AV(dl_carrier_frequency_GHz, distance_b_to_a_2d_m, distance_b_to_a_3d_m, LoS_b_to_a, position_z_a_m, self.position_z_b_m, site_ID)      
                
            elif (bs_propagation_model == "3GPPTR36_777_UMi_AV"):
                self.path_loss_b_to_a_dB[:, mask] = \
                    self.path_loss_3GPPTR36_777_UMi_AV(dl_carrier_frequency_GHz, distance_b_to_a_2d_m, distance_b_to_a_3d_m, LoS_b_to_a, position_z_a_m, self.position_z_b_m, site_ID)     
                
            elif (bs_propagation_model == "3GPPTR38_811_Urban_NTN" or bs_propagation_model == "3GPPTR38_811_Dense_Urban_NTN"):
                self.path_loss_b_to_a_dB[:, mask] =\
                     self.path_loss_3GPPPTR38_811(dl_carrier_frequency_GHz, distance_b_to_a_3d_m, zeniths_b_to_a_wraparound_degrees, LoS_b_to_a, position_z_a_m)                   
                
        # Store in data frames the results as it may be useful to post process
        self.df_path_loss_b_to_a_dB = pd.DataFrame(self.path_loss_b_to_a_dB, columns=self.network_deployment_obj.df_ep["name"], index=self.ue_deployment_obj.df_ep["name"])  
        

        ##### Save to plot
        ########################        
        snapshot_control = Snapshot_control.get_instance()
        if self.simulation_config_obj.plot == 1 and snapshot_control.num_snapshots == 0: 
            
            file_name = results_file(self.simulation_config_obj.project_name, 'to_plot_path_loss')
            np.savez(file_name, path_loss_b_to_a_dB = self.path_loss_b_to_a_dB)
            

        ##### End
        log_calculations_time('Path loss', t_start)
        return rescheduling_us          
            
 
    def calcualte_g_2d(self, d_2D_m, h_ut_mat_m):
        
        g_2d = np.zeros((np.size(d_2D_m,0), np.size(d_2D_m,1)), dtype=np.single)  
        
        g_2d_aux = 5/4 * np.power(d_2D_m/100,3) * np.exp(-d_2D_m/150)
        
        mask = np.logical_and(d_2D_m > 18, h_ut_mat_m >= 13)
        g_2d[mask] = g_2d_aux[mask]
        
        return g_2d


    def calcualte_C(self, d_2D_m, h_ut_mat_m):
        
        C = np.zeros((np.size(d_2D_m,0), np.size(d_2D_m,1)), dtype=np.single)
        
        g_2d = self.calcualte_g_2d(d_2D_m, h_ut_mat_m)
        
        mask = h_ut_mat_m >= 13  
        C[mask] = np.power((h_ut_mat_m[mask] - 13)/ 10, 1.5) * g_2d[mask]
        
        return C      


    def calculate_h_e_UMa(self, d_2D_m, h_ut_mat_m, site_ID):
        
        # To maintain correlation across sites: Get unique elements, indices of the first occurrences, and inverse indices 
        unique_elements, first_indices, inverse_indices = np.unique(site_ID, return_index=True, return_inverse=True)   
    
        h_e_m = np.ones((np.size(d_2D_m,0), np.size(unique_elements)), dtype=np.single)  
        
        d_2D_m = d_2D_m[:,first_indices]
        h_ut_mat_m = h_ut_mat_m[:,first_indices]
        
        # Nested list comprehension with condition 
        h_e_rand_m = np.array([[ self.rng.choice(np.arange(12, h_ut_mat_m[i,j], 3)) if  h_ut_mat_m[i,j] > 13 else 0 for j in range(0, np.size(d_2D_m, 1))  ] for i in range(0, np.size(d_2D_m, 0)) ])

        C = self.calcualte_C(d_2D_m, h_ut_mat_m)
    
        mask = self.rng.rand(np.size(d_2D_m,0), np.size(d_2D_m,1)) < 1/(1+C) 
        
        h_e_m[mask] = h_e_rand_m[mask]
        
        # Map back to all cells of the site using `np.take` to replicate the values according to `inverse_indices`
        h_e_m = np.take(h_e_m, inverse_indices, axis=1)
        
        return h_e_m


    def calculate_d_BP_UMa(self, f_c_mat_GHz, d_2D_m, h_bs_mat_m, h_ut_mat_m, site_ID): 
        
        c =  scipy.constants.c
        h_e_m = self.calculate_h_e_UMa(d_2D_m, h_ut_mat_m, site_ID)
        hprime_bs_m = h_bs_mat_m - h_e_m
        hprime_ut_m = h_ut_mat_m - h_e_m
        
        return (4 * hprime_bs_m * hprime_ut_m * (f_c_mat_GHz * 1e9)  / c).astype(np.single)
    

    def calculate_d_BP_UMi(self,  f_c_mat_GHz, d_2D_m, h_bs_mat_m, h_ut_mat_m): 
        
        c =  scipy.constants.c
        h_e_m = 1.0
        hprime_bs_m = h_bs_mat_m - h_e_m
        hprime_ut_m = h_ut_mat_m - h_e_m
        
        return (4 * hprime_bs_m * hprime_ut_m * (f_c_mat_GHz * 1e9) / c).astype(np.single)
 

    def path_loss_3GPPTR38_901_UMa(self, f_c_GHz, d_2D_m, d_3D_m, LoS, h_bs_m, h_ut_m, site_ID):
        
        if np.sum(h_ut_m < 1.5) > 0:
            np.disp('Error: The path loss model, path_loss_3GPPTR38_901_UMa, cannot use heights smaller than 1.5m!')
            sys.exit(0)        
        
        if np.sum(h_ut_m > 22.5) > 0:
            np.disp('Error: The path loss model, path_loss_3GPPTR38_901_UMa, cannot use heights larger than 22.5m!')
            sys.exit(0)
            
        # Initialize path loss results 
        path_loss_for_model_dB = np.ones((np.size(d_3D_m,0), np.size(d_3D_m,1)), dtype=np.single)  
        
        # Replicate the vector of cell frequencies in a column manner to facilitate next operation
        f_c_mat_GHz = f_c_GHz[np.newaxis,:] * np.ones((np.size(d_2D_m,0),1), dtype=np.single) # This is the same as (np.tile(f_c_GHz, (np.size(d_2D_m,0),1)))    
        h_bs_mat_m = h_bs_m[np.newaxis,:] * np.ones((np.size(d_2D_m,0),1), dtype=np.single)  
        h_ut_mat_m = h_ut_m[:,np.newaxis] * np.ones((1,np.size(d_2D_m,1)), dtype=np.single)      
        
        # Calculate d_BP_m
        d_BP_m = self.calculate_d_BP_UMa(f_c_mat_GHz, d_2D_m, h_bs_mat_m, h_ut_mat_m, site_ID)
        
        # LoS 
        mask = d_2D_m <= d_BP_m
        path_loss_for_model_dB[mask] = 28.0 + 22*np.log10(d_3D_m[mask]) + 20*np.log10(f_c_mat_GHz[mask])
        
        mask = d_2D_m > d_BP_m
        path_loss_for_model_dB[mask] = 28.0 + 40*np.log10(d_3D_m[mask]) + 20*np.log10(f_c_mat_GHz[mask]) - \
                                            9*np.log10(np.power(d_BP_m[mask],2) + np.power(h_bs_mat_m[mask]-h_ut_mat_m[mask],2))
        
        # NLoS 
        mask = LoS == 0
        NLoS_path_loss_prime_for_model_dB = np.zeros((np.size(d_3D_m,0), np.size(d_3D_m,1)), dtype=np.single)  
        NLoS_path_loss_prime_for_model_dB[mask] = 13.54 + 39.08*np.log10(d_3D_m[mask]) + 20*np.log10(f_c_mat_GHz[mask]) - 0.6*(h_ut_mat_m[mask] - 1.5)
        
        mask = np.logical_and(LoS==0, NLoS_path_loss_prime_for_model_dB > path_loss_for_model_dB)
        path_loss_for_model_dB[mask] = NLoS_path_loss_prime_for_model_dB[mask]
            
        return path_loss_for_model_dB
        

    def path_loss_3GPPTR38_901_UMi(self, f_c_GHz, d_2D_m, d_3D_m, LoS, h_bs_m, h_ut_m):
        
        if np.sum(h_ut_m < 1.5) > 0:
            np.disp('Error: The path loss model, path_loss_3GPPTR38_901_UMi, cannot use heights smaller than 1.5m!')
            sys.exit(0)        
        
        if np.sum(h_ut_m > 22.5) > 0:
            np.disp('Error: The path loss model, path_loss_3GPPTR38_901_UMi, cannot use heights larger than 22.5m!')
            sys.exit(0)
                
        # Initialize path loss results 
        path_loss_for_model_dB = np.ones((np.size(d_3D_m,0), np.size(d_3D_m,1)))  
        
        # Replicate the vector of cell frequencies in a column manner to facilitate next operation
        f_c_mat_GHz = f_c_GHz[np.newaxis,:] * np.ones((np.size(d_2D_m,0),1)) # This is the same as (np.tile(f_c_GHz, (np.size(d_2D_m,0),1)))    
        h_bs_mat_m = h_bs_m[np.newaxis,:] * np.ones((np.size(d_2D_m,0),1))  
        h_ut_mat_m = h_ut_m[:,np.newaxis] * np.ones((1,np.size(d_2D_m,1)))    
        
        d_BP_m = self.calculate_d_BP_UMi(f_c_mat_GHz, d_2D_m, h_bs_mat_m, h_ut_mat_m)
             
        # LoS 
        mask = d_2D_m <= d_BP_m
        path_loss_for_model_dB[mask] = 32.4 + 21*np.log10(d_3D_m[mask]) + 20*np.log10(f_c_mat_GHz[mask])
        
        mask = d_2D_m > d_BP_m
        path_loss_for_model_dB[mask] = 32.4 + 40*np.log10(d_3D_m[mask]) + 20*np.log10(f_c_mat_GHz[mask]) \
                                                - 9.5*np.log10( np.power(d_BP_m[mask],2) + np.power(h_bs_mat_m[mask]-h_ut_mat_m[mask],2))
        
        # NLoS 
        mask = LoS == 0
        NLoS_path_loss_prime_for_model_dB = np.zeros((np.size(d_3D_m,0), np.size(d_3D_m,1)))  
        NLoS_path_loss_prime_for_model_dB[mask] = 22.4  + 35.3*np.log10(d_3D_m[mask]) + 21.3*np.log10(f_c_mat_GHz[mask]) - 0.3*(h_ut_mat_m[mask] - 1.5)
         
        mask = np.logical_and(LoS==0, NLoS_path_loss_prime_for_model_dB > path_loss_for_model_dB)
        path_loss_for_model_dB[mask] = NLoS_path_loss_prime_for_model_dB[mask]
            
        return path_loss_for_model_dB  
    

    def path_loss_ITU_R_M2135_UMa(self, f_c_GHz, d_3D_m, LoS, h_bs_m, h_ut_m, site_ID):
        
        if np.sum(h_bs_m != 25) > 0 and self.simulation_config_obj.scenario_model == 'ITU_R_M2135_UMa' :
            np.disp('Error: The path loss model, path_loss_ITU_R_M2135_UMa, cannot use BS heights different than 25m!')
            sys.exit(0) 
            
        if np.sum(h_ut_m != 1.5) > 0 and self.simulation_config_obj.scenario_model == 'ITU_R_M2135_UMa':
            np.disp('Error: The path loss model, path_loss_ITU_R_M2135_UMa, cannot use UE heights different than 1.5m!')
            sys.exit(0)   
            
        if np.sum(d_3D_m > 5000) > 0 and self.simulation_config_obj.scenario_model == 'ITU_R_M2135_UMa':
            np.disp('Error: The path loss model, path_loss_ITU_R_M2135_UMa, cannot use distances larger than 5000m!')
            sys.exit(0)             
            
        # Initialize path loss results 
        path_loss_for_model_dB = np.ones((np.size(d_3D_m,0), np.size(d_3D_m,1)))   
        
        # Replicate the vector of cell frequencies in a column manner to facilitate next operation
        f_c_mat_GHz = f_c_GHz[np.newaxis,:] * np.ones((np.size(d_3D_m,0),1)) # This is the same as (np.tile(f_c_GHz, (np.size(d_2D_m,0),1)))    
        h_bs_mat_m = h_bs_m[np.newaxis,:] * np.ones((np.size(d_3D_m,0),1))  
        h_ut_mat_m = h_ut_m[:,np.newaxis] * np.ones((1,np.size(d_3D_m,1)))          
        
        c =  scipy.constants.c
        hprime_bs_m = h_bs_mat_m - 1.0   
        hprime_ut_m = h_ut_mat_m - 1.0
        d_BP_m = 4 * hprime_bs_m * hprime_ut_m * (f_c_mat_GHz * 1e9)  / c
        
        # LoS 
        mask = np.logical_and(LoS==1, d_3D_m <= d_BP_m)
        path_loss_for_model_dB[mask] = 22*np.log10(d_3D_m[mask]) + 28.0 + 20*np.log10(f_c_mat_GHz[mask])
        
        mask = np.logical_and(LoS==1, d_3D_m > d_BP_m)
        path_loss_for_model_dB[mask] = 40*np.log10(d_3D_m[mask]) + 7.8 - 18*np.log10(h_bs_mat_m[mask]-1) - 18*np.log10(h_ut_mat_m[mask]-1) + 2*np.log10(f_c_mat_GHz[mask]) 
        
        # NLoS 
        mask = LoS == 0
        path_loss_for_model_dB[mask]  = 161.04 - 7.1*np.log10(20) + 7.5*np.log10(20) \
                                                    - (24.37-3.7*np.power(20/h_bs_mat_m[mask],2)) * np.log10(h_bs_mat_m[mask]) \
                                                    + (43.42-3.1*np.log10(h_bs_mat_m[mask])) * (np.log10(d_3D_m[mask]) - 3) \
                                                    + 20*np.log10(f_c_mat_GHz[mask]) - (3.2*np.power(np.log10(11.75*h_ut_mat_m[mask]),2) - 4.97 )
        
        # In cabin penetration LoSs
        in_cabin_penetration_LoSs_dB = 9 + 5 * self.rng.randn(np.size(d_3D_m,0), np.size(d_3D_m,1))  
        # Correlate LoSs across sites
        unique_elements, first_indices = np.unique(site_ID, return_index=True)  
        in_cabin_penetration_LoSs_dB = np.take(in_cabin_penetration_LoSs_dB[:, first_indices], site_ID, axis=1)
        
        # In the multi-layer scenarions, due to chance, a UE may fall nearby another cell, violating the min distance rule. We may the pathloss very large, such that the user does not associate to this cell
        path_loss_for_model_dB[d_3D_m < 10] = 500
              
        return path_loss_for_model_dB + in_cabin_penetration_LoSs_dB + 2
    

    def path_loss_ITU_R_M2135_UMi(self, f_c_GHz, d_3D_m, d_2D_in_m, LoS, h_bs_m, h_ut_m, indoor):
       
        if np.sum(h_bs_m != 10) > 0:
            np.disp('Error: The path loss model, path_loss_ITU_R_M2135_UMi, cannot use BS heights different than 25m!')
            sys.exit(0)
            
        if np.sum(h_ut_m != 1.5) > 0:
            np.disp('Error: The path loss model, path_loss_ITU_R_M2135_UMi, cannot use UE heights different than 1.5m!')
            sys.exit(0)   
            
        if np.sum(d_3D_m > 5000) > 0:
            np.disp('Error: The path loss model, path_loss_ITU_R_M2135_UMi, cannot use distances larger than 5000m!')
            sys.exit(0)       
           
        # Initialize path loss results 
        path_loss_for_model_dB = np.ones((np.size(d_3D_m,0), np.size(d_3D_m,1)))   
       
        # Replicate the vector of cell frequencies in a column manner to facilitate next operation          
        f_c_mat_GHz = f_c_GHz[np.newaxis,:] * np.ones((np.size(d_3D_m,0),1)) # This is the same as (np.tile(f_c_GHz, (np.size(d_2D_m,0),1)))    
        h_bs_mat_m = h_bs_m[np.newaxis,:] * np.ones((np.size(d_3D_m,0),1))  
        h_ut_mat_m = h_ut_m[:,np.newaxis] * np.ones((1,np.size(d_3D_m,1)))          
       
        c =  scipy.constants.c
        hprime_bs_m = h_bs_mat_m - 1.0   
        hprime_ut_m = h_ut_mat_m - 1.0
        d_BP_m = 4 * hprime_bs_m * hprime_ut_m * (f_c_mat_GHz * 1e9)  / c
       
        # LoS
        mask = np.logical_and(LoS==1, d_3D_m <= d_BP_m)
        path_loss_for_model_dB[mask] = 22*np.log10(d_3D_m[mask]) + 28.0 + 20*np.log10(f_c_mat_GHz[mask])
       
        mask = np.logical_and(LoS==1, d_3D_m > d_BP_m)
        path_loss_for_model_dB[mask] = 40*np.log10(d_3D_m[mask]) + 7.8 - 18*np.log10(h_bs_mat_m[mask]-1) - 18*np.log10(h_ut_mat_m[mask]-1) + 2*np.log10(f_c_mat_GHz[mask]) 
       
        # NLoS 
        mask = LoS == 0
        path_loss_for_model_dB[mask] = 36.7*np.log10(d_3D_m[mask]) + 22.7 + 26*np.log10(f_c_mat_GHz[mask])
       
        # Indoor UEs has PLoS = 0
        indoor_mat = indoor * np.ones((1,np.size(d_3D_m,1)), dtype=bool) 
        
        PLtw_dB = 20 # wall penetration LoSs
        path_loss_for_model_dB[indoor_mat] +=  PLtw_dB + 0.5 * d_2D_in_m[indoor_mat]    
        
        # In the multi-layer scenarions, due to chance, a UE may fall nearby another cell, violating the min distance rule. We may the pathloss very large, such that the user does not associate to this cell
        path_loss_for_model_dB[d_3D_m < 10] = 500        
              
        return path_loss_for_model_dB + 2    
    

    def path_loss_3GPPTR36_814_Case_1(self, f_c_GHz, d_3D_m):
        
        if not(np.all(f_c_GHz==2)):
            np.disp('Error: The path loss model, path_loss_3GPPTR36_814_UMa, cannot use frequencies different than 2!')
            sys.exit(0)        
            
        path_loss_for_model_dB = 128.1 + 37.6*np.log10(d_3D_m/1000)
            
        wall_penetration_LoSs_dB = 20
        
        return path_loss_for_model_dB + wall_penetration_LoSs_dB
    

    def path_loss_3GPPTR36_814_UMa(self, f_c_GHz, d_3D_m, LoS):
        
        if not(np.all(f_c_GHz==2)):
            np.disp('Error: The path loss model, path_loss_3GPPTR36_814_UMa, cannot use frequencies different than 2!')
            sys.exit(0)        
            
        # Initialize path loss results 
        path_loss_for_model_dB = np.ones((np.size(d_3D_m,0), np.size(d_3D_m,1)))  
        
        # LoS 
        mask = LoS == 1
        path_loss_for_model_dB[mask] = 103.4 + 24.0*np.log10(d_3D_m[mask]/1000) 
        
        # NLoS 
        mask = LoS == 0
        path_loss_for_model_dB[mask] = 131.1 + 42.8*np.log10(d_3D_m[mask]/1000)
            
        return path_loss_for_model_dB
    

    def path_loss_3GPPTR36_777_UMa_AV(self, f_c_GHz, d_2D_m, d_3D_m, LoS, h_bs_m, h_ut_m, site_ID):
        
        if np.sum(h_ut_m < 1.5) > 0:
            sys.exit('Error: The path loss model, path_loss_3GPPTR36_777_UMa_AV, cannot use heights smaller than 1.5m!')       
        
        if np.sum(h_ut_m > 300) > 0:
            sys.exit('Error: The path loss model, path_loss_3GPPTR36_777_UMa_AV, cannot use heights larger than 300m!')
            
        if np.sum(np.sum(h_ut_m[np.all(LoS, axis=1) == 0] > 100)) > 0:
            sys.exit('Error: The path loss model, path_loss_3GPPTR36_777_UMa_AV, cannot use heights larger than 100m for NLoS UEs!')           
            
        if np.sum(d_2D_m > 4000) > 0:
            sys.exit('Error: The path loss model, path_loss_3GPPTR36_777_UMa_AV, cannot use distances larger than 4000m!')     
                       
            
        # Initialize path loss results 
        path_loss_for_model_dB = np.ones((np.size(d_3D_m,0), np.size(d_3D_m,1)))  
        
        # Replicate the vector of cell frequencies in a column manner to facilitate next operation
        f_c_mat_GHz = f_c_GHz[np.newaxis,:] * np.ones((np.size(d_2D_m,0),1)) # This is the same as (np.tile(f_c_GHz, (np.size(d_2D_m,0),1)))    
        h_bs_mat_m = h_bs_m[np.newaxis,:] * np.ones((np.size(d_2D_m,0),1))  
        h_ut_mat_m = h_ut_m[:,np.newaxis] * np.ones((1,np.size(d_2D_m,1)))        
        
        d_BP_m = self.calculate_d_BP_UMa(f_c_mat_GHz, d_2D_m, h_bs_mat_m, h_ut_mat_m, site_ID)
        
        #### Below 22.5 m
        # LoS         
        mask = np.logical_and(h_ut_mat_m <= 22.5, d_2D_m <= d_BP_m)
        path_loss_for_model_dB[mask] = 28.0 + 22*np.log10(d_3D_m[mask]) + 20*np.log10(f_c_mat_GHz[mask])
        
        mask = np.logical_and(h_ut_mat_m <= 22.5, d_2D_m > d_BP_m)
        path_loss_for_model_dB[mask] = 28.0 + 40*np.log10(d_3D_m[mask]) + 20*np.log10(f_c_mat_GHz[mask]) - 9*np.log10(np.power(d_BP_m[mask],2) + np.power(h_bs_mat_m[mask]-h_ut_mat_m[mask],2))
        
        # NLoS 
        mask = np.logical_and(LoS == 0, h_ut_mat_m <= 22.5)
        NLoS_path_loss_prime_for_model_dB = np.zeros((np.size(d_3D_m,0), np.size(d_3D_m,1)))  
        NLoS_path_loss_prime_for_model_dB[mask] = 13.54 + 39.08*np.log10(d_3D_m[mask]) + 20*np.log10(f_c_mat_GHz[mask]) - 0.6*(h_ut_mat_m[mask] - 1.5)
        
        mask = np.logical_and(LoS == 0, np.logical_and(h_ut_mat_m <= 22.5, NLoS_path_loss_prime_for_model_dB > path_loss_for_model_dB))
        path_loss_for_model_dB[mask] = NLoS_path_loss_prime_for_model_dB[mask]        
        
        #### Above 22.5 m
        # LoS 
        mask = np.logical_and(LoS == 1, np.logical_and(h_ut_mat_m > 22.5, np.logical_and(h_ut_mat_m <= 300, d_2D_m < 4000)))
        path_loss_for_model_dB[mask] = 28.0 + 22*np.log10(d_3D_m[mask]) + 20*np.log10(f_c_mat_GHz[mask])        
        
        # NLoS 
        mask = np.logical_and(LoS == 0, np.logical_and(h_ut_mat_m > 22.5, np.logical_and(h_ut_mat_m <= 100, d_2D_m < 4000)))
        path_loss_for_model_dB[mask] = -17.5 + (46 - 7*np.log10(h_ut_mat_m[mask])) * np.log10(d_3D_m[mask]) + 20*np.log10( 40 * np.pi * f_c_mat_GHz[mask] / 3)
                  
        return path_loss_for_model_dB  
    

    def path_loss_3GPPTR36_777_UMi_AV(self, f_c_GHz, d_2D_m, d_3D_m, LoS, h_bs_m, h_ut_m, site_ID):
        
        if np.sum(h_ut_m < 1.5) > 0:
            sys.exit('Error: The path loss model, path_loss_3GPPTR36_777_UMi_AV, cannot use heights smaller than 1.5m!')       
        
        if np.sum(h_ut_m > 300) > 0:
            sys.exit('Error: The path loss model, path_loss_3GPPTR36_777_UMi_AV, cannot use heights larger than 300m!')
            
        if np.sum(d_2D_m > 4000) > 0:
            sys.exit('Error: The path loss model, path_loss_3GPPTR36_777_UMi_AV, cannot use distances larger than 4000m!')     
                       
            
        # Initialize path loss results 
        path_loss_for_model_dB = np.ones((np.size(d_3D_m,0), np.size(d_3D_m,1)))  
        
        # Replicate the vector of cell frequencies in a column manner to facilitate next operation
        f_c_mat_GHz = f_c_GHz[np.newaxis,:] * np.ones((np.size(d_2D_m,0),1)) # This is the same as (np.tile(f_c_GHz, (np.size(d_2D_m,0),1)))    
        h_bs_mat_m = h_bs_m[np.newaxis,:] * np.ones((np.size(d_2D_m,0),1))  
        h_ut_mat_m = h_ut_m[:,np.newaxis] * np.ones((1,np.size(d_2D_m,1)))        
        
        d_BP_m = self.calculate_d_BP_UMa(f_c_mat_GHz, d_2D_m, h_bs_mat_m, h_ut_mat_m, site_ID)
        
        #### Below 22.5 m
        # LoS 
        mask = np.logical_and(h_ut_mat_m <= 22.5, d_2D_m <= d_BP_m)
        path_loss_for_model_dB[mask] = 28.0 + 22*np.log10(d_3D_m[mask]) + 20*np.log10(f_c_mat_GHz[mask])
        
        mask = np.logical_and(h_ut_mat_m <= 22.5, d_2D_m > d_BP_m)
        path_loss_for_model_dB[mask] = 28.0 + 40*np.log10(d_3D_m[mask]) + 20*np.log10(f_c_mat_GHz[mask]) - 9*np.log10(np.power(d_BP_m[mask],2) + np.power(h_bs_mat_m[mask]-h_ut_mat_m[mask],2))
        
        # NLoS 
        mask = np.logical_and(LoS == 0, h_ut_mat_m <= 22.5)
        NLoS_path_loss_prime_for_model_dB = np.zeros((np.size(d_3D_m,0), np.size(d_3D_m,1)))  
        NLoS_path_loss_prime_for_model_dB[mask] = 13.54 + 39.08*np.log10(d_3D_m[mask]) + 20*np.log10(f_c_mat_GHz[mask]) - 0.6*(h_ut_mat_m[mask] - 1.5)
        
        mask = np.logical_and(LoS == 0, np.logical_and(h_ut_mat_m <= 22.5, NLoS_path_loss_prime_for_model_dB > path_loss_for_model_dB))
        path_loss_for_model_dB[mask] = NLoS_path_loss_prime_for_model_dB[mask]        
        
        #### Above 22.5 m
        # LoS 
        mask = np.logical_and(h_ut_mat_m > 22.5, np.logical_and(h_ut_mat_m <= 300, d_2D_m < 4000))
        
        free_space_path_loss_b_to_a_dB = np.zeros((np.size(d_3D_m,0), np.size(d_3D_m,1)))  
        free_space_path_loss_b_to_a_dB[mask] = 92.45 +  20*np.log10(d_3D_m[mask]/1000) + 20*np.log10(f_c_mat_GHz[mask]) 
        
        LoS_AV_path_loss_prime_for_model_dB = np.zeros((np.size(d_3D_m,0), np.size(d_3D_m,1)))   
        LoS_AV_path_loss_prime_for_model_dB[mask] = 30.9 + (22.25-0.5*np.log10(h_ut_mat_m[mask]))*np.log10(d_3D_m[mask]) + 20*np.log10(f_c_mat_GHz[mask])  
        
        max_operatrion = np.maximum(free_space_path_loss_b_to_a_dB, LoS_AV_path_loss_prime_for_model_dB)
        
        path_loss_for_model_dB[mask] = max_operatrion[mask]
        
        # NLoS 
        mask = np.logical_and(LoS == 0, np.logical_and(h_ut_mat_m > 22.5, np.logical_and(h_ut_mat_m <= 300, d_2D_m < 4000)))
        
        NLoS_AV_path_loss_prime_for_model_dB = np.zeros((np.size(d_3D_m,0), np.size(d_3D_m,1)))   
        NLoS_AV_path_loss_prime_for_model_dB[mask] = 32.4 + (43.2 - 7.6*np.log10(h_ut_mat_m[mask]))*np.log10(d_3D_m[mask]) + 20*np.log10(f_c_mat_GHz[mask])   
        
        mask = np.logical_and(LoS == 0, np.logical_and(h_ut_mat_m > 22.5, np.logical_and(h_ut_mat_m <= 300, np.logical_and(d_2D_m < 4000, NLoS_AV_path_loss_prime_for_model_dB > path_loss_for_model_dB ))))
        path_loss_for_model_dB[mask] = NLoS_AV_path_loss_prime_for_model_dB[mask]  
        
        return path_loss_for_model_dB    
    

    def path_loss_3GPPPTR38_811(self, f_c_GHz, d_3D_m, zeniths_b_to_a_wraparound_degrees, LoS, h_bs_m):
            
        # Find set of unique frequencies 
        set_of_unique_f_c_GHz = np.unique(f_c_GHz)
        
        # Replicate the vector of cell frequencies in a column manner to facilitate next operation   
        f_c_mat_GHz = f_c_GHz[np.newaxis,:] * np.ones((np.size(d_3D_m,0),1)) # This is the same as (np.tile(f_c_GHz, (np.size(d_2D_m,0),1))) 
        
        # Elevation angle 
        elevation_b_to_a_wraparound_degrees = zeniths_b_to_a_wraparound_degrees - 90   
        
        # Slant range 
        R_e_m = 6371000.0
        d_m = np.sqrt(np.square(R_e_m)*np.square(np.sin(np.radians(elevation_b_to_a_wraparound_degrees))) + np.square(h_bs_m) + 2*h_bs_m*R_e_m) - R_e_m * np.sin(np.radians(elevation_b_to_a_wraparound_degrees))
       
        # Initialize path loss results with free space path loss
        path_loss_for_model_dB = 20*np.log10(d_m) + 32.45 + 20*np.log10(f_c_mat_GHz)
       
        # Calculate mask for NLoS and elevation range 
        # Add clutter loss to path loss
        mask = np.logical_and(LoS == False, np.logical_and(elevation_b_to_a_wraparound_degrees >= 0, elevation_b_to_a_wraparound_degrees < 15))
        path_loss_for_model_dB[mask] +=  34.3

        mask = np.logical_and(LoS == False, np.logical_and(elevation_b_to_a_wraparound_degrees >= 15, elevation_b_to_a_wraparound_degrees < 25))
        path_loss_for_model_dB[mask] += 30.9  

        mask = np.logical_and(LoS == False, np.logical_and(elevation_b_to_a_wraparound_degrees >= 25, elevation_b_to_a_wraparound_degrees < 35))
        path_loss_for_model_dB[mask] += 29.0  

        mask = np.logical_and(LoS == False, np.logical_and(elevation_b_to_a_wraparound_degrees >= 35, elevation_b_to_a_wraparound_degrees < 45))
        path_loss_for_model_dB[mask] += 27.7  

        mask = np.logical_and(LoS == False, np.logical_and(elevation_b_to_a_wraparound_degrees >= 45, elevation_b_to_a_wraparound_degrees < 55))
        path_loss_for_model_dB[mask] += 26.8

        mask = np.logical_and(LoS == False, np.logical_and(elevation_b_to_a_wraparound_degrees >= 55, elevation_b_to_a_wraparound_degrees < 65))
        path_loss_for_model_dB[mask] += 26.2   

        mask = np.logical_and(LoS == False, np.logical_and(elevation_b_to_a_wraparound_degrees >= 65, elevation_b_to_a_wraparound_degrees < 75))
        path_loss_for_model_dB[mask] += 25.8  

        mask = np.logical_and(LoS == False, np.logical_and(elevation_b_to_a_wraparound_degrees >= 75, elevation_b_to_a_wraparound_degrees < 90))
        path_loss_for_model_dB[mask] += 25.5 
        
        # Add atmospheric attenuation 
        # Parameters 
        A_g = np.zeros(f_c_mat_GHz.shape)
        A_r = np.zeros(f_c_mat_GHz.shape)
        A_c = np.zeros(f_c_mat_GHz.shape)
        A_s = np.zeros(f_c_mat_GHz.shape)  
        
        for unique_f_c_GHz in set_of_unique_f_c_GHz:
            
            # Mask
            mask = f_c_mat_GHz == unique_f_c_GHz
            
            # Variables
            var_D = 3e8 / (unique_f_c_GHz * 1e9) / 2 * itur.u.m
            var_T = unique_f_c_GHz * 15 * u.deg_C
            var_P = unique_f_c_GHz  * 1013.25 * u.hPa
            var_rho = unique_f_c_GHz * 7.5 * u.g / u.m**3
            
            var_lat = 42.3601  
            var_lon =  -71.0942 
            var_hs = itur.topographic_altitude(var_lat, var_lon)
            var_p = 0.1
        
            # Gaseous attenuation
            A_g[mask] = itur.gaseous_attenuation_slant_path(unique_f_c_GHz, elevation_b_to_a_wraparound_degrees[mask], var_rho, var_P, var_T)   
    
            # Rain and cloud attenuation
            A_r[mask] = itur.rain_attenuation(var_lat, var_lon, unique_f_c_GHz, elevation_b_to_a_wraparound_degrees[mask], var_hs, var_p)
            A_c[mask] = itur.cloud_attenuation(var_lat, var_lon, elevation_b_to_a_wraparound_degrees[mask], unique_f_c_GHz, var_p)
    
            # Scintillation attenuation
            A_s[mask] = itur.scintillation_attenuation(var_lat, var_lon, unique_f_c_GHz, elevation_b_to_a_wraparound_degrees[mask], var_p, var_D) 
        
        # Total
        A_t = A_g + A_r + A_c + A_s

        path_loss_for_model_dB += A_t 
    
        return path_loss_for_model_dB 