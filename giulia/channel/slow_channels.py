# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:40:53 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import time
import copy
from typing import List

import numpy as np
import pandas as pd
from typing import Any, Dict

from giulia.event_driven.snapshot_control import Snapshot_control
from giulia.fs import results_file
from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable

class Slow_Channel(Saveable):
    
    def __init__(self, 
                 simulation_config_obj,
                 network_deployment_obj, 
                 ue_deployment_obj,
                 time_frequency_resource_obj,
                 antenna_pattern_gain_ue_to_cell_obj,
                 path_loss_ue_to_cell_obj,
                 o2i_penetration_loss_ue_to_cell_obj,
                 shadowing_gain_ue_to_cell_obj):
        
        super().__init__()

        ##### Input storage 
        ######################## 
        self.simulation_config_obj = simulation_config_obj
        self.network_deployment_obj = network_deployment_obj
        self.ue_deployment_obj = ue_deployment_obj  
        self.time_frequency_resource_obj = time_frequency_resource_obj
        self.antenna_pattern_gain_ue_to_cell_obj = antenna_pattern_gain_ue_to_cell_obj
        self.path_loss_ue_to_cell_obj = path_loss_ue_to_cell_obj
        self.o2i_penetration_loss_ue_to_cell_obj = o2i_penetration_loss_ue_to_cell_obj
        self.shadowing_gain_ue_to_cell_obj = shadowing_gain_ue_to_cell_obj
        

        ##### Outputs 
        ######################## 
        # Placeholder to store fast fading results 
        
        slow_channel_gain_no_shadowing_b_to_a_dB = []
        self.df_slow_channel_gain_no_shadowing_b_to_a_dB = []
        
        # Output results for each frequency layer
        self.slow_channel_results_per_frequency_layer: Dict[float, Dict[str, Any]] = {}
        
        
    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["slow_channel_results_per_frequency_layer"]

       
    def process(self, rescheduling_us=-1): 
        
        ##### Process inputs
        ########################  
        
        # Network
        self.bs_fast_channel_models = self.network_deployment_obj.df_ep["BS_fast_channel_model"].to_numpy()  
        
        # Channel characterisitics  
        self.antenna_pattern_gain_b_to_a_dB = self.antenna_pattern_gain_ue_to_cell_obj.antenna_pattern_gain_b_to_a_dB              
        self.path_loss_b_to_a_dB = self.path_loss_ue_to_cell_obj.path_loss_b_to_a_dB
        self.o2i_penetration_losses_b_to_a_dB = self.o2i_penetration_loss_ue_to_cell_obj.o2i_penetration_losses_b_to_a_dB
        self.shadowing_gain_b_to_a_dB = self.shadowing_gain_ue_to_cell_obj.shadowing_gain_b_to_a_dB  
        
        ##### Process outputs
        ########################        
        slow_channel_gain_no_shadowing_b_to_a_dB = np.zeros((np.size(self.path_loss_b_to_a_dB,0), np.size(self.path_loss_b_to_a_dB,1)), dtype=np.single)
        slow_channel_gain_b_to_a_dB = np.zeros((np.size(self.path_loss_b_to_a_dB,0), np.size(self.path_loss_b_to_a_dB,1)), dtype=np.single)        
       

        ##### Start timer
        ########################         
        t_start = time.perf_counter() 
        
        
        ##### Switch
        ########################         
        
        # Find the set of unique propagation models to process them independently
        bs_fast_channel_model_set = set(self.bs_fast_channel_models)
        
        # Process each propagation model independently
        for bs_fast_channel_model in bs_fast_channel_model_set:
            
            # Identify cells with the selected propagation model
            mask = bs_fast_channel_model ==  self.bs_fast_channel_models  
            
            # Get necessary information of the identified cells
            antenna_pattern_gain_b_to_a_dB = self.antenna_pattern_gain_b_to_a_dB[:, mask]
            path_loss_b_to_a_dB = self.path_loss_b_to_a_dB[:, mask]
            o2i_penetration_losses_b_to_a_dB = self.o2i_penetration_losses_b_to_a_dB[:, mask]
            shadowing_gain_b_to_a_dB = self.shadowing_gain_b_to_a_dB[:, mask]
            
            # Calculate slow channel
            if (bs_fast_channel_model == "Rician" or bs_fast_channel_model == "Rayleigh"):  
                slow_channel_gain_no_shadowing_b_to_a_dB = \
                    self.slow_channel_gain_no_shadowing_b_to_a(path_loss_b_to_a_dB, o2i_penetration_losses_b_to_a_dB, antenna_pattern_gain_b_to_a_dB)
                
                slow_channel_gain_b_to_a_dB = \
                    self.slow_channel_gain_b_to_a(slow_channel_gain_no_shadowing_b_to_a_dB, shadowing_gain_b_to_a_dB)    
                
        # Store results for the frequency layer
        self.slow_channel_results_per_frequency_layer["all_freq"] = copy.deepcopy(self.time_frequency_resource_obj.dl_frequency_layer_info["all_freq"])
        self.slow_channel_results_per_frequency_layer["all_freq"].update({
            "slow_channel_gain_no_shadowing_b_to_a_dB": slow_channel_gain_no_shadowing_b_to_a_dB,
            "slow_channel_gain_b_to_a_dB": slow_channel_gain_b_to_a_dB,
        })                  
                
        # Store in data frames the results as it may be useful to post process
        self.df_slow_channel_gain_no_shadowing_b_to_a_dB = pd.DataFrame(slow_channel_gain_no_shadowing_b_to_a_dB, columns=self.network_deployment_obj.df_ep["name"], index=self.ue_deployment_obj.df_ep["name"])   
        self.df_slow_channel_gain_b_to_a_dB = pd.DataFrame(slow_channel_gain_b_to_a_dB, columns=self.network_deployment_obj.df_ep["name"], index=self.ue_deployment_obj.df_ep["name"])  

        ##### Save to plot
        ########################          
        snapshot_control = Snapshot_control.get_instance()
        if self.simulation_config_obj.plot == 1 and snapshot_control.num_snapshots == 0:
            file_name = results_file(self.simulation_config_obj.project_name, 'to_plot_slow_channel_gain')
            np.savez(file_name, slow_channel_gain_b_to_a_dB = slow_channel_gain_b_to_a_dB)
            

        ##### End
        log_calculations_time('Slow channel', t_start)

        return rescheduling_us                    
          
        
    def slow_channel_gain_no_shadowing_b_to_a(self, 
                                              path_loss_b_to_a_dB,
                                              o2i_penetration_losses_b_to_a_dB,
                                              antenna_pattern_gain_b_to_a_dB):
        #Note that slow channel gain calculations do not include noise power - ITU M 2135-1
       
        return - path_loss_b_to_a_dB - o2i_penetration_losses_b_to_a_dB + antenna_pattern_gain_b_to_a_dB 


    def slow_channel_gain_b_to_a(self, 
                                 slow_channel_gain_no_shadowing_b_to_a_dB,
                                 shadowing_gain_b_to_a_dB):
        #Note that slow channel gain calculations do not include noise power - ITU M 2135-1
        
        if not self.simulation_config_obj.debug_no_randomness:  
            return slow_channel_gain_no_shadowing_b_to_a_dB + shadowing_gain_b_to_a_dB      
        else: 
            return slow_channel_gain_no_shadowing_b_to_a_dB