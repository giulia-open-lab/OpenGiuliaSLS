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
import time
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import norm

from giulia.event_driven.snapshot_control import Snapshot_control
from giulia.fs import results_file
from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable


class O2I_Penetration_Loss(Saveable):
    
    def __init__(self, 
                 simulation_config_obj, 
                 network_deployment_obj, 
                 ue_deployment_obj,
                 distance_angles_ue_to_cell_obj):
        
        super().__init__()

        ##### Input storage 
        ######################## 
        self.simulation_config_obj = simulation_config_obj
        self.network_deployment_obj = network_deployment_obj
        self.ue_deployment_obj = ue_deployment_obj    
        self.distance_angles_ue_to_cell_obj = distance_angles_ue_to_cell_obj
        
       
        ##### Outputs 
        ########################   
        # Place holder to store path loss results
        
        self.o2i_penetration_losses_b_to_a_dB = []
        self.df_o2i_penetration_losses_b_to_a_dB = []
        

    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["o2i_penetration_losses_b_to_a_dB"]

       
    def process(self, rescheduling_us=-1): 
        
        ##### Process inputs
        ########################  
        
        # Random numbers
        self.rng = np.random.RandomState(self.simulation_config_obj.random_seed+0)        
            
        # Network
        self.bs_propagation_models = self.network_deployment_obj.df_ep["BS_propagation_model"].to_numpy()
        self.bs_fast_channel_models = self.network_deployment_obj.df_ep["BS_fast_channel_model"].to_numpy()        
        self.dl_carrier_frequency_GHz = self.network_deployment_obj.df_ep["dl_carrier_frequency_GHz"].to_numpy(dtype=np.single) 
        self.site_ID = self.network_deployment_obj.df_ep["site_ID"].to_numpy()
        
        # Users deployment 
        self.indoor = self.ue_deployment_obj.df_ep[["indoor"]].to_numpy(dtype=bool)      

        # Channel characterisitics  
        self.zeniths_b_to_a_wraparound_degrees = self.distance_angles_ue_to_cell_obj.zeniths_b_to_a_wraparound_degrees
        self.d_2D_in_m = self.distance_angles_ue_to_cell_obj.d_2D_in_m


        ##### Process outputs
        ########################
        self.o2i_penetration_losses_b_to_a_dB = np.zeros((np.size(self.d_2D_in_m,0), np.size(self.dl_carrier_frequency_GHz,0)), dtype=np.single)
     
       
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
            site_ID = self.site_ID[mask]
            
            zeniths_b_to_a_wraparound_degrees = self.zeniths_b_to_a_wraparound_degrees[:, mask]
            
            d_2D_in_m = self.d_2D_in_m[:, mask]

            
            # Calculate path loss
            
            if (bs_propagation_model == "3GPPTR38_901_UMa" and bs_fast_channel_model != "3GPPTR38_901_UMa") \
                or (bs_propagation_model == "3GPPTR38_901_UMi" and bs_fast_channel_model != "3GPPTR38_901_UMi"):

                self.o2i_penetration_losses_b_to_a_dB[:, mask] = self.o2i_loss_3GPPTR38_901_UMa_Umi(dl_carrier_frequency_GHz, site_ID, self.indoor, d_2D_in_m)
                
            elif bs_propagation_model == "3GPPTR38_811_Urban_NTN" or bs_propagation_model == "3GPPTR38_811_Dense_Urban_NTN":
                self.o2i_penetration_losses_b_to_a_dB[:, mask] = self.o2i_loss_3GPPTR38_811(dl_carrier_frequency_GHz, self.indoor, zeniths_b_to_a_wraparound_degrees, d_2D_in_m)
                
        # Store in data frames the results as it may be useful to post process
        self.df_o2i_penetration_losses_b_to_a_dB = pd.DataFrame(self.o2i_penetration_losses_b_to_a_dB, columns=self.network_deployment_obj.df_ep["name"], index=self.ue_deployment_obj.df_ep["name"])  
        

        ##### Save to plot
        ########################         
        snapshot_control = Snapshot_control.get_instance()
        if self.simulation_config_obj.plot == 1 and snapshot_control.num_snapshots == 0:
            file_name = results_file(self.simulation_config_obj.project_name, 'to_plot_o2i_penetration_loss')
            np.savez(file_name, o2i_penetration_losses_b_to_a_dB = self.o2i_penetration_losses_b_to_a_dB[self.o2i_penetration_losses_b_to_a_dB != 0])
            

        ##### End
        log_calculations_time('Outdoor to indoor penetration', t_start)

        return rescheduling_us
        
  
    def o2i_loss_3GPPTR38_901_UMa_Umi(self, f_c_GHz, site_ID, indoor, d_2D_in_m):
                    
        # Initialize path LoSs results 
        o2i_penetration_loss_for_model_dB = np.zeros((np.size(d_2D_in_m,0), np.size(f_c_GHz,0)), dtype=np.single)  
        
        if self.simulation_config_obj.debug_no_randomness == 0:
            
            unique_elements, first_indices, inverse_indices = np.unique(site_ID, return_index=True, return_inverse=True) 
            
            PL_in_dB = np.zeros(o2i_penetration_loss_for_model_dB.shape)
            PL_tw_dB = np.zeros(o2i_penetration_loss_for_model_dB.shape)
            P_building_penetration_loss_dB = np.zeros(o2i_penetration_loss_for_model_dB.shape)        
            
            # Replicate the vector of cell frequencies in a column manner to facilitate next operation
            f_c_mat_GHz = f_c_GHz[np.newaxis,:] * np.ones((np.size(d_2D_in_m,0),1), dtype=np.single) # This is the same as (np.tile(f_c_GHz, (np.size(d_2D_m,0),1)))    
            
            # Replicate the vector of indoor indicators in a column manner to facilitate next operation
            indoor_mat = indoor *  np.ones((1,np.size(f_c_GHz,0)),dtype=bool)      
            
            # O2I bulding penetration loss model
            
            # Parameters 
            PL_in_dB[indoor_mat] = 0.5 * d_2D_in_m[indoor_mat] #[:,np.newaxis] *  np.ones((1,np.size(f_c_GHz,0)),dtype=bool)    
            
            L_glass = 2 + 0.2 * f_c_mat_GHz
            L_IIRglass = 23 + 0.3 * f_c_mat_GHz
            L_concrete = 5 + 4 * f_c_mat_GHz
            
            # Mask for users in low or high loss 
            random_low_high_loss = self.rng.choice([True, False], size=np.size(d_2D_in_m,0), p=[0.5, 0.5])[:,np.newaxis] * np.ones((1,np.size(f_c_GHz,0)),dtype=bool)
            mask_high_loss = np.logical_and(indoor_mat, random_low_high_loss)
            
            
            # Low loss model 
            PL_tw_dB[indoor_mat] = 5 - 10 * np.log10(0.3 * np.power(10,-L_glass[indoor_mat]/10) + 0.7 * np.power(10,-L_concrete[indoor_mat]/10) )            
            rnd_generator = self.rng.normal(0, 4.4, (np.size(indoor_mat,0), np.size(unique_elements))) # This is to maintain correlation across sites

            # Map back to all cells of the site using `np.take` to replicate the values according to `inverse_indices`
            P_building_penetration_loss_dB[indoor_mat]  = np.take(rnd_generator, inverse_indices, axis=1)[indoor_mat]
            
            
            # High loss model 
            PL_tw_dB[mask_high_loss] = 5 - 10 * np.log10(0.7 * np.power(10,-L_IIRglass[mask_high_loss]/10) + 0.3 * np.power(10,-L_concrete[mask_high_loss]/10))            
            rnd_generator = self.rng.normal(0, 6.5, (np.size(indoor_mat,0), np.size(unique_elements))) # This is to maintain correlation across sites

            # Map back to all cells of the site using `np.take` to replicate the values according to `inverse_indices`
            P_building_penetration_loss_dB[mask_high_loss]  = np.take(rnd_generator, inverse_indices, axis=1)[mask_high_loss]            
            
            
            # Adding O2I building penetration to path loss
            o2i_penetration_loss_for_model_dB[indoor_mat] = PL_in_dB[indoor_mat] + PL_tw_dB[indoor_mat] + P_building_penetration_loss_dB[indoor_mat]
                        
            # O2I car penetration loss model  
            
            # P_car_penetration_loss_dB = np.zeros(PL_in_dB.shape)
            # P_car_penetration_loss_dB[~indoor_mat] = self.rng.normal(9, 5, indoor_mat.shape)[~indoor_mat] 
            
            # # Adding O2I building penetration to path loss
            
            # o2i_penetration_loss_for_model_dB[~indoor_mat] += P_car_penetration_loss_dB[~indoor_mat]
            
        return o2i_penetration_loss_for_model_dB
    
        
    def o2i_loss_3GPPTR38_811(self, f_c_GHz, indoor, zeniths_b_to_a_wraparound_degrees, d_2D_in_m):
        
        # Initialize O2I results 
        o2i_penetration_loss_for_model_dB = np.zeros((np.size(d_2D_in_m,0), np.size(f_c_GHz,0)), dtype=np.single)  
        
        # Replicate the vector of cell frequencies in a column manner to facilitate next operation 
        f_c_mat_GHz = f_c_GHz[np.newaxis,:] * np.ones((np.size(d_2D_in_m,0),1), dtype=np.single) # This is the same as (np.tile(f_c_GHz, (np.size(d_2D_m,0),1)))  
        
        # Replicate the vector of indoor indicators in a column manner to facilitate next operation
        indoor_mat = indoor *  np.ones((1,np.size(f_c_GHz,0)),dtype=bool)              
        
        # O2I bulding penetration loss model
        
        # Parameters 
        
        # Elevation angle 
        elevation_b_to_a_wraparound_degrees = zeniths_b_to_a_wraparound_degrees - 90  
        L_e = 0.212*np.abs(elevation_b_to_a_wraparound_degrees)
        C = -3.0  
        
        # Create random probabilities that loss is not exceeded 
        P = self.rng.rand(np.size(d_2D_in_m,0), np.size(f_c_GHz,0))
        
        # Mask for users in low or high loss 
        
        random_low_high_loss = self.rng.choice([True, False], size=np.size(d_2D_in_m,0), p=[0.5, 0.5])[:,np.newaxis] * np.ones((1,np.size(f_c_GHz,0)),dtype=bool)
        mask_high_loss = np.logical_and(indoor_mat, random_low_high_loss)  
        
        # Traditional 
        
        r = 12.64
        s = 3.72
        t = 0.96
        u = 9.6
        v = 2.0
        w = 9.1
        x = -3.0
        y = 4.50
        z = -2.0 
        
        L_h = r + s*np.log10(f_c_mat_GHz[indoor_mat]) + t*np.square(np.log10(f_c_mat_GHz[indoor_mat]))
        sigma_1 = u + v*np.log10(f_c_mat_GHz[indoor_mat])
        mu_1 = L_h + L_e[indoor_mat]
        sigma_2 = y + z*np.log10(f_c_mat_GHz[indoor_mat])
        mu_2 = w + x*np.log10(f_c_mat_GHz[indoor_mat])
        
        A_P = norm.ppf(P[indoor_mat]) * sigma_1 + mu_1
        B_P = norm.ppf(P[indoor_mat]) * sigma_2 + mu_2
        
        o2i_penetration_loss_for_model_dB[indoor_mat]  = 10 * np.log10(np.power(10, 0.1*A_P) + np.power(10, 0.1*B_P ) + np.power(10, 0.1*C ))
        
        # Thermally efficient  
        
        r = 28.19
        s = -3.0
        t = 8.48
        u = 13.5
        v = 3.8
        w = 27.8
        x = -2.9
        y = 9.4
        z = -2.1 
        
        L_h = r + s*np.log10(f_c_mat_GHz[mask_high_loss]) + t*np.square(np.log10(f_c_mat_GHz[mask_high_loss]))
        sigma_1 = u + v*np.log10(f_c_mat_GHz[mask_high_loss])
        mu_1 = L_h + L_e[mask_high_loss]
        sigma_2 = y + z*np.log10(f_c_mat_GHz[mask_high_loss])
        mu_2 = w + x*np.log10(f_c_mat_GHz[mask_high_loss])
        
        A_P = norm.ppf(P[mask_high_loss]) * sigma_1 + mu_1
        B_P = norm.ppf(P[mask_high_loss]) * sigma_2 + mu_2
      
        o2i_penetration_loss_for_model_dB[mask_high_loss]  = 10 * np.log10(np.power(10, 0.1*A_P ) + np.power(10, 0.1*B_P ) + np.power(10, 0.1*C))
            
        return o2i_penetration_loss_for_model_dB  