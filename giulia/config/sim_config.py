# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:52:45 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@itea.upv.es
"""

import numpy as np
import pandas as pd 
from dataclasses import dataclass
from giulia.playground.sites import Site
from typing import Optional, Union, List, Dict, Any
from giulia.outputs.saveable import Saveable


class Simulation_Config (Saveable):
    
    def __init__(self,    
                 preset,
                 scenario_playground_model,
                 ue_playground_model,
                 ue_distribution, 
                 ue_mobility,
                 link_direction,
                 wraparound,
                 save_results,
                 plots,
                 number_of_episods,
                 regression,                    
                 sn_indicator,
                 uav_height_m,
                 enable_GPU,
                 project_name,
                 enable_saveable: bool,
                 **kwargs): 
       
       super().__init__()
    
       ##### Input storage
       ########################

       self.preset = preset
       self.scenario_playground_model = scenario_playground_model
       self.bs_mobility = None #"straight_walk"
       self.ue_playground_model = ue_playground_model
       self.ue_distribution = ue_distribution
       self.ue_mobility = ue_mobility
       self.link_direction = link_direction
       self.wraparound = wraparound
       self.save_results = save_results
       self.number_of_episods = number_of_episods
       self.regression = regression   
       self.sn_indicator = sn_indicator 
       self.uav_height_m = uav_height_m
       self.enable_GPU = enable_GPU
       self.enable_saveable = enable_saveable
       
       self.random_seed = 0     
       self.instantaneous_RSRP = 0 
       self.sinr_mapping = "MIESM"
       self.plot = plots
       self.plot_heatmaps = plots
       self.debug_no_randomness = False 

       self.enable_saveable: bool = Saveable.is_enabled()
       
       self.project_name = project_name
       
       
       # Set density of deployement UE information
       self.set_ue_deployement_info()
       self.number_of_ues_per_UEsite_cell: np.ndarray = np.array([])
       self.number_of_ues_per_UEsite_cell_mfl_dict: dict = {}
       self.number_of_aerial_ues_per_UEsite_cell: np.ndarray = np.array([])
       self.number_of_aerial_ues_per_UEsite_cell_mfl_dict: dict = {}
       
       
       # Place holder for output
       self.event_scheduling_info: dict = {}
       
       # Set Giulia Configure Event Scheduling us
       self.set_dflt_rescheduling_us(dflt_rescheduling_us=1e6, 
                                     dflt_rescheduling_UE_deployment_us=1e6)
       self.set_event_scheduling_info()


    def expose_variables(self) -> List[str]:
        """List of attributes name to be saved."""
        return []

    def set_sn_indicator(self, network_deployment_obj):
        
       bs_propagation_models = network_deployment_obj.df_ep["BS_propagation_model"].to_numpy()
       bs_fast_channel_models = network_deployment_obj.df_ep["BS_fast_channel_model"].to_numpy() 
       
       mask_Uma = np.bitwise_and("3GPPTR38_901_UMa" ==  bs_propagation_models, "3GPPTR38_901_UMa" ==  bs_fast_channel_models)
       mask_Umi = np.bitwise_and("3GPPTR38_901_UMi" ==  bs_propagation_models, "3GPPTR38_901_UMi" ==  bs_fast_channel_models)
       mask_RT = np.bitwise_and("Ray_tracing" ==  bs_propagation_models, "Ray_tracing" ==  bs_fast_channel_models)

       if self.sn_indicator == False and (sum(mask_Uma) > 1 or sum(mask_Umi) > 1 or sum(mask_RT) > 1):
           np.disp('Error Sionna is not installed and thus the 3GPPTR38_901_UMa/UMi channel model or ray tracing cannot be processed ')
           exit(0) 
           
       elif self.sn_indicator == True and (sum(mask_Uma) > 1 or sum(mask_Umi) > 1 or sum(mask_RT) > 1):
           self.sn_indicator = True
           
       else:
           self.sn_indicator = False
           
    def set_instantaneous_RSRP(self, network_deployment_obj):
        
       bs_propagation_models = network_deployment_obj.df_ep["BS_propagation_model"].to_numpy()
       bs_fast_channel_models = network_deployment_obj.df_ep["BS_fast_channel_model"].to_numpy() 
       
       mask_Uma = np.bitwise_and("3GPPTR38_901_UMa" ==  bs_propagation_models, "3GPPTR38_901_UMa" ==  bs_fast_channel_models)
       mask_Umi = np.bitwise_and("3GPPTR38_901_UMi" ==  bs_propagation_models, "3GPPTR38_901_UMi" ==  bs_fast_channel_models)
       mask_RT = np.bitwise_and("Ray_tracing" == bs_propagation_models, "Ray_tracing" == bs_fast_channel_models)
           
       if self.sn_indicator == True and (sum(mask_Uma) > 1 or sum(mask_Umi) > 1 or sum(mask_RT) > 1):
           self.instantaneous_RSRP = True
           
       else:
           self.instantaneous_RSRP = False


    def _set_ue_playground_deployment_obj(self, ue_playground_deployment_obj: Site):
        self.ue_playground_deployment_obj = ue_playground_deployment_obj

           
    def set_random_seed(self, random_seed):
        self.random_seed = random_seed
           
    def increase_random_seed(self):
        self.random_seed = self.random_seed + 1
                      

    def _set_attr_sim_config(self, **kwargs) -> None:
        """
        This function sets attributes in the Simulation_Config object.
        """
        for key, value in kwargs.items(): setattr(self, key, value)

        return 
           
            
    def _set_number_of_ues_per_UEsite_cell(self) -> None:
        """
        This function sets the number_of_ues_per_UEsite_cell attribute in the Simulation_Config object.
        This attribute is used internally by Giulia to configure the ue_deployments_obj.
        This permits to set the number of UEs per UE Site cell.

        Within the following, values are distinguished between two condition:
            1. Multi-layer UE deployment
            2. Single-layer UE deployment

        1. Multi-layer UE deployment:
            In this case, the ue_deployDensity_info attribute is a dictionary,
            where each key corresponds to a layer and the value is an object
            containing the ue_density_perCell attribute for that layer.
        2. Single-layer UE deployment:
            In this case, the ue_deployDensity_info attribute is a single object  
            containing the ue_density_perCell attribute for the single layer.

        The function supports different UE distribution types:
            - "uniform": All cells have the same number of UEs.
            - "uniform_with_hotspots": All cells have the same number of UEs, with additional UEs in hotspot areas.
            - "inhomogeneous_per_cell": Different cells have different numbers of UEs based on
               predefined density values.
            - "inhomogeneous_per_cell_with_hotspots": Different cells have different numbers of UEs,
               based on predefined density values, with additional UEs in hotspot areas.
        """
        # Get number of cell (no base station) of the UE Playground (hexagonal grid)
        N_UEsite_cell = self.ue_playground_deployment_obj.ref_number_of_cells
        assert N_UEsite_cell is not None, "Error, N_UEsite_cell is NoneType"

        # Manage Grid UE deployment case
        # If ue_distribution is "grid", the number of UEs is set directly in the ue_playground_deployment_obj
        # using the self.ue_deployDensity_info object attribute
        if self.ue_distribution == "grid": return

        # Manage Multi-Layer UE deployment case
        if isinstance(self.ue_deployDensity_info, dict):
            # Initialize
            number_of_ues_per_UEsite_cell = None
            number_of_aerial_ues_per_UEsite_cell = None
            number_of_ues_per_UEsite_cell_mfl_dict = {}
            number_of_aerial_ues_per_UEsite_cell_mfl_dict = {}

            for key, ue_density_obj in self.ue_deployDensity_info.items():
                # Set UEs numbers
                if self.ue_distribution in ["uniform", "uniform_with_hotspots"]:
                    number_of_ues_per_UEsite_cell_mfl_dict[key] = np.full(N_UEsite_cell, ue_density_obj.ue_density_perCell, dtype=int) 
                    
                elif self.ue_distribution in ["inhomogeneous_per_cell", "inhomogeneous_per_cell_with_hotspots"]:
                    number_of_ues_per_UEsite_cell_mfl_dict[key] = np.full(N_UEsite_cell, -1, dtype=int)
                    try:
                        if len(ue_density_obj.ue_density_perCell) == 3:
                            number_of_ues_per_UEsite_cell_mfl_dict[key][:3] = ue_density_obj.ue_density_perCell[0]  
                            number_of_ues_per_UEsite_cell_mfl_dict[key][3:21] = ue_density_obj.ue_density_perCell[1] 
                            number_of_ues_per_UEsite_cell_mfl_dict[key][21:] = ue_density_obj.ue_density_perCell[2] 
                            
                        elif len(ue_density_obj.ue_density_perCell) == 2:
                            number_of_ues_per_UEsite_cell_mfl_dict[key][:21] = ue_density_obj.ue_density_perCell[0]  
                            number_of_ues_per_UEsite_cell_mfl_dict[key][21:] = ue_density_obj.ue_density_perCell[1] 

                        else:
                            raise ValueError(f"Error, in {self.__class__.__name__} in inhomogeneous_per_cell distributions, ue_density_perCell should be a list of 2 or 3 elements")
                    
                    except TypeError:
                        # Manage Case for hotspot for which ue_density_obj.ue_density_perCell is not a list
                        # and len(...) cannot be applied
                        number_of_ues_per_UEsite_cell_mfl_dict[key] = np.full(N_UEsite_cell, ue_density_obj.ue_density_perCell, dtype=int)
                        

                number_of_aerial_ues_per_UEsite_cell_mfl_dict[key] = np.full(N_UEsite_cell, ue_density_obj.aerialsUE_density_perCell, dtype=int) 
            
            
        # Single-layer UE deployment case
        else:
            number_of_ues_per_UEsite_cell_mfl_dict = None
            number_of_aerial_ues_per_UEsite_cell_mfl_dict = None
            
            if self.ue_distribution in ["uniform", "uniform_with_hotspots"]:
                number_of_ues_per_UEsite_cell = np.full(N_UEsite_cell, self.ue_deployDensity_info.ue_density_perCell, dtype=int) 
                
            elif self.ue_distribution in ["inhomogeneous_per_cell", "inhomogeneous_per_cell_with_hotspots"]:
                # Basic checks
                if not isinstance(self.ue_deployDensity_info.ue_density_perCell, list):
                    raise TypeError("Error, ue_density_perCell should be a list for inhomogeneous_per_cell distributions")
                
                ### WARNING ###
                ### The following hardcoding is valid only 
                ### for the current Giulia UE playgrounds based on hexagonal grid
                ### with 19 trisectorial site, thus 57 cells
                number_of_ues_per_UEsite_cell = np.full(N_UEsite_cell, -1, dtype=int)
                if len(self.ue_deployDensity_info.ue_density_perCell) == 3:
                    number_of_ues_per_UEsite_cell[:3] = self.ue_deployDensity_info.ue_density_perCell[0]  
                    number_of_ues_per_UEsite_cell[3:21] = self.ue_deployDensity_info.ue_density_perCell[1] 
                    number_of_ues_per_UEsite_cell[21:] = self.ue_deployDensity_info.ue_density_perCell[2] 
                    
                elif len(self.ue_deployDensity_info.ue_density_perCell) == 2:
                    number_of_ues_per_UEsite_cell[:21] = self.ue_deployDensity_info.ue_density_perCell[0]  
                    number_of_ues_per_UEsite_cell[21:] = self.ue_deployDensity_info.ue_density_perCell[1] 
                else:
                    raise ValueError("Error, in inhomogeneous_per_cell distributions, ue_density_perCell should be a list of 2 or 3 elements")


            # Set UAVs numbers
            number_of_aerial_ues_per_UEsite_cell = np.full(N_UEsite_cell, self.ue_deployDensity_info.aerialsUE_density_perCell, dtype=int) 
            
        # Set finally
        self._set_attr_sim_config(
            number_of_ues_per_UEsite_cell=number_of_ues_per_UEsite_cell,
            number_of_aerial_ues_per_UEsite_cell=number_of_aerial_ues_per_UEsite_cell,
            number_of_ues_per_UEsite_cell_mfl_dict=number_of_ues_per_UEsite_cell_mfl_dict,
            number_of_aerial_ues_per_UEsite_cell_mfl_dict=number_of_aerial_ues_per_UEsite_cell_mfl_dict
            )
        
        return
    
    class _UE_info_deploy:
        def __init__(self, model_name, number_of_ues=-1,
                    ue_density_perCell=None,
                    aerialsUE_density_perCell=None,
                    frac_ue_hotspotPerCell_density=None,
                    min_UE_to_hotspot_distance_m=None):

            self.model_name = model_name
            self.number_of_ues = number_of_ues
            self.ue_density_perCell = ue_density_perCell
            self.aerialsUE_density_perCell = aerialsUE_density_perCell
            self.frac_ue_hotspotPerCell_density = frac_ue_hotspotPerCell_density
            self.min_UE_to_hotspot_distance_m = min_UE_to_hotspot_distance_m

    def set_ue_deployement_info(self):
        """
        This function sets up a dictionary/dataframe used internally by Giulia 
        to configure attributes of the ue_deployments_obj, specifically parameters such as number_of_ues.      
        
        Within the following, values are distinguished between two condition:
            1. UE Playground
            2. UE Distribution
        """
        
        distribution = self.ue_distribution
        ue_playground_model = self.ue_playground_model
        
        ##### Switch
        ########################
        # This switch mirrors the one in ue_deployments.py to ensure consistency 
 
        if self.ue_playground_model in [
                "ITU_R_M2135_UMa", 
                "ITU_R_M2135_UMa_multilayer", 
                "ITU_R_M2135_UMa_Umi_colocated_multilayer",
                "ITU_R_M2135_UMa_Umi_noncolocated_multilayer"]:

            if distribution == "grid":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                number_of_ues = 40873)
                
            elif distribution == "uniform":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=10,
                                                aerialsUE_density_perCell=0)
                
            elif distribution == "uniform_with_hotspots":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=30,
                                                aerialsUE_density_perCell=0
                                                )
                
            elif distribution == "inhomogeneous_per_cell":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=[20, 10],
                                                aerialsUE_density_perCell=0)
                
            elif distribution == "inhomogeneous_per_cell_with_hotspots":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=[60, 30],
                                                aerialsUE_density_perCell=0
                                                )

            else: raise ValueError(f"Error, distribution not correctly specified.\
                                   Value '{distribution}' not admitted in {ue_playground_model} ue_playground_model")
            

        elif self.ue_playground_model in ["ITU_R_M2135_UMi"]: 
            
            if distribution == "grid":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                number_of_ues = 26113)
            
            elif distribution == "uniform":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=10,
                                                aerialsUE_density_perCell=0)
                
            elif distribution == "uniform_with_hotspots":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=30,
                                                aerialsUE_density_perCell=0
                                                )
                
            elif distribution == "inhomogeneous_per_cell":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=[20, 10],
                                                aerialsUE_density_perCell=0)
                
            elif distribution == "inhomogeneous_per_cell_with_hotspots":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=[60, 30],
                                                aerialsUE_density_perCell=0
                                                )
            
            else: raise ValueError(f"Error, distribution not correctly specified.\
                                   Value '{distribution}' not admitted in {ue_playground_model} ue_playground_model")

            

        elif self.ue_playground_model in [
                "3GPPTR36_814_Case_1", 
                "3GPPTR36_814_Case_1_omni", 
                "3GPPTR36_814_Case_1_single_bs", 
                "3GPPTR36_814_Case_1_single_bs_omni"]  :
            
            if distribution == "grid":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                number_of_ues = 53842 \
                                                    if ue_playground_model == "3GPPTR36_814_Case_1_single_bs_omni" \
                                                        else 40873)
            
            elif distribution == "uniform":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=10,
                                                aerialsUE_density_perCell=0)
                
            elif distribution == "uniform_with_hotspots":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=30,
                                                aerialsUE_density_perCell=0
                                                )
                
            elif distribution == "inhomogeneous_per_cell":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=[20, 10],
                                                aerialsUE_density_perCell=0)
                
            elif distribution == "inhomogeneous_per_cell_with_hotspots":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=[60, 30],
                                                aerialsUE_density_perCell=0
                                                )

            else: raise ValueError(f"Error, distribution not correctly specified.\
                                   Value '{distribution}' not admitted in {ue_playground_model} ue_playground_model")

            

        elif self.ue_playground_model in [
                "3GPPTR38_901_UMa_C1", 
                "3GPPTR38_901_UMa_C2"]:
            
            if distribution == "grid":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                number_of_ues = 40873)
            
            elif distribution == "uniform":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=10,
                                                aerialsUE_density_perCell=0)
                
            elif distribution == "uniform_with_hotspots":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=30,
                                                aerialsUE_density_perCell=0
                                                )
                
            elif distribution == "inhomogeneous_per_cell":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=[80, 40, 20],
                                                aerialsUE_density_perCell=0)
                
            elif distribution == "inhomogeneous_per_cell_with_hotspots":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=[80, 40, 20],
                                                aerialsUE_density_perCell=0
                                                )

            else: raise ValueError(f"Error, distribution not correctly specified.\
                                   Value '{distribution}' not admitted in {ue_playground_model} ue_playground_model")



        elif self.ue_playground_model in [
                "3GPPTR38_901_UMa_lsc", 
                "3GPPTR38_901_UMa_lsc_sn", 
                "3GPPTR38_901_UMa_lsc_single_bs",
                "3GPPTR38_901_UMa_lsc_single_sector",
                "3GPPTR38_901_UMa_2GHz_lsc",
                "3GPPTR38_901_UMa_C_band_lsc",
                "3GPPTR38_811_Urban_NTN",
                "3GPPTR38_811_Dense_Urban_NTN", 
                "3GPPTR38_811_Dense_Urban_HAPS_ULA",
                "3GPPTR38_811_Dense_Urban_HAPS_UPA",
                "3GPPTR38_811_Dense_Urban_HAPS_Reflector",
                "3GPPTR38_811_Dense_Urban_HAPS_TriReflector",
                "3GPPTR38_811_Dense_Urban_SingleHAPS_Reflector",
                "3GPPTR38_811_Dense_Urban_SingleHAPS_TriReflector"]:
            

            if distribution == "grid":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                number_of_ues = 53842 \
                                                    if ue_playground_model == "3GPPTR38_901_UMa_lsc_single_bs" or ue_playground_model == "3GPPTR38_901_UMa_lsc_single_sector" \
                                                        else 40873)

            elif distribution == "uniform":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=10,
                                                aerialsUE_density_perCell=0)
                
            elif distribution == "uniform_with_hotspots":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=30,
                                                aerialsUE_density_perCell=0
                                                )
                
            elif distribution == "inhomogeneous_per_cell":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell= [80, 40, 20],
                                                aerialsUE_density_perCell=0)
                
            elif distribution == "inhomogeneous_per_cell_with_hotspots":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=[80, 40, 20],
                                                aerialsUE_density_perCell=0
                                                )
            
            else: raise ValueError(f"Error, distribution not correctly specified.\
                                   Value '{distribution}' not admitted in {ue_playground_model} ue_playground_model")
            
            
            
        elif self.ue_playground_model in [
                "3GPPTR38_901_4G",
                "3GPPTR38_901_5G",
                "3GPPTR38_901_6G",
                "3GPPTR38_901_4G5G_multilayer",
                "3GPPTR38_811_Dense_Urban_HAPS_901_4G5G",
                "3GPPTR38_901_4G_5G_multilayer",
                "3GPPTR38_901_4G_5G2_multilayer",
                "3GPPTR38_901_4G_5G6G_multilayer",
                "3GPPTR38_901_4G_5G_6G_multilayer",
                "3GPPTR38_901_4G5G_cell_reselection", 
                "3GPPTR38_811_Dense_Urban_singleHAPS_UPA_4G5G", 
                "3GPPTR38_811_Dense_Urban_singleHAPS_UPA_4G_5G",
                "3GPPTR38_811_Dense_Urban_singleHAPS_reflector_4G5G"]:
            
            ue_deployDensity_info = {}
            
            if distribution == "grid":
                ue_deployDensity_info["construct_ue_deployment_3GPPTR38_901_UMa_large_scale_calibration"] =\
                self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                               number_of_ues = 40873)
                ue_deployDensity_info["construct_ue_deployment_polygon"] =\
                self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                               number_of_ues = 40873)
                

            elif distribution == "uniform":
                ue_deployDensity_info["construct_ue_deployment_3GPPTR38_901_UMa_large_scale_calibration"] = \
                    self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=10,
                                                aerialsUE_density_perCell=0)
                ue_deployDensity_info["construct_ue_deployment_polygon"] = \
                    self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=10,
                                                aerialsUE_density_perCell=0)

            elif distribution == "uniform_with_hotspots":
                ue_deployDensity_info["construct_ue_deployment_3GPPTR38_901_UMa_large_scale_calibration"] = \
                self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=30,
                                                aerialsUE_density_perCell=0
                                                )
                ue_deployDensity_info["construct_ue_deployment_polygon"] = \
                    self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=10,
                                                aerialsUE_density_perCell=0
                                                )

            elif distribution == "inhomogeneous_per_cell":
                ue_deployDensity_info["construct_ue_deployment_3GPPTR38_901_UMa_large_scale_calibration"] =\
                    self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell= [80, 40, 20],
                                                aerialsUE_density_perCell=0)
                    
                ue_deployDensity_info["construct_ue_deployment_polygon"] = \
                    self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=10,
                                                aerialsUE_density_perCell=0
                                                )

            elif distribution == "inhomogeneous_per_cell_with_hotspots":
                ue_deployDensity_info["construct_ue_deployment_3GPPTR38_901_UMa_large_scale_calibration"] =\
                    self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=[80, 40, 20],
                                                aerialsUE_density_perCell=0
                                                )
                    
                    
                ue_deployDensity_info["construct_ue_deployment_polygon"] = \
                    self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=10,
                                                aerialsUE_density_perCell=0
                                                )

            else: raise ValueError(f"Error, distribution not correctly specified.\
                                   Value '{distribution}' not admitted in {ue_playground_model} ue_playground_model")



        elif self.ue_playground_model in [
                "3GPPTR38_901_UMi_C1",
                "3GPPTR38_901_UMi_C2"]:
                                                    
            if distribution == "grid":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                number_of_ues = 26113)
            
            elif distribution == "uniform":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=10,
                                                aerialsUE_density_perCell=0)
                
            elif distribution == "uniform_with_hotspots":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=30,
                                                aerialsUE_density_perCell=0
                                                )
                
            elif distribution == "inhomogeneous_per_cell":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=[80, 40, 20],
                                                aerialsUE_density_perCell=0)
                
            elif distribution == "inhomogeneous_per_cell_with_hotspots":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=[80, 40, 20],
                                                aerialsUE_density_perCell=0
                                                )
                
            else: raise ValueError(f"Error, distribution not correctly specified.\
                                   Value '{distribution}' not admitted in {ue_playground_model} ue_playground_model")

                

        elif self.ue_playground_model in [
                "3GPPTR38_901_UMi_lsc",
                "3GPPTR38_901_UMi_C_band_lsc",
                "3GPPTR38_901_UMi_fr3_lsc",
                "3GPPTR38_901_UPi_fr3_lsc"]:
            if distribution == "grid":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                number_of_ues = 26113)
            
            elif distribution == "uniform":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=10,
                                                aerialsUE_density_perCell=0)
                
            elif distribution == "uniform_with_hotspots":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=30,
                                                aerialsUE_density_perCell=0
                                                )
                
            elif distribution == "inhomogeneous_per_cell":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=[80, 40, 20],
                                                aerialsUE_density_perCell=0)
                
            elif distribution == "inhomogeneous_per_cell_with_hotspots":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=[80, 40, 20],
                                                aerialsUE_density_perCell=0
                                                )
                                                    
            else: raise ValueError(f"Error, distribution not correctly specified.\
                                   Value '{distribution}' not admitted in {ue_playground_model} ue_playground_model")



        elif self.ue_playground_model in ["3GPPTR36_777_UMa_AV"] : 

            if distribution == "uniform":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=15,
                                                aerialsUE_density_perCell=3
                                                )
            
            else: raise ValueError(f"Error, distribution not correctly specified.\
                                   Value '{distribution}' not admitted in {ue_playground_model} ue_playground_model")

                                   

        elif self.ue_playground_model in ["3GPPTR36_777_UMi_AV"] : 
            
            if distribution == "uniform":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=15,
                                                aerialsUE_density_perCell=3
                                                )
           
            else: raise ValueError(f"Error, distribution not correctly specified.\
                                   Value '{distribution}' not admitted in {ue_playground_model} ue_playground_model")



        elif self.ue_playground_model in [
                "rectangular",
                "circular"]:
            
            if distribution == "grid":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                number_of_ues = 40873)
            
            elif distribution == "uniform":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=10,
                                                aerialsUE_density_perCell=0)
                
            elif distribution == "uniform_with_hotspots":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=10,
                                                aerialsUE_density_perCell=0
                                                )
           
            else: raise ValueError(f"Error, distribution not correctly specified.\
                                   Value '{distribution}' not admitted in {ue_playground_model} ue_playground_model")

                

        elif self.ue_playground_model in [
                "dataset_rectangular", 
                "dataset_circular",
                "3GPPTR36_814_Case_1_omni_dana"]:

            if distribution == "grid":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                number_of_ues = 40873)
            
            elif distribution == "uniform":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=10,
                                                aerialsUE_density_perCell=0)
                
            elif distribution == "uniform_with_hotspots":
                ue_deployDensity_info = self._UE_info_deploy(f"{ue_playground_model}_{distribution}", 
                                                ue_density_perCell=10,
                                                aerialsUE_density_perCell=0
                                                )
            
        else: 
            raise ValueError(f"Error, ue_playground_model in sim_config not correctly specified. Value '{self.ue_playground_model}' not admitted")
        

        # Set finally
        self.ue_deployDensity_info = ue_deployDensity_info
        del ue_deployDensity_info



    def set_dflt_rescheduling_us(self, dflt_rescheduling_us:float=1e6, dflt_rescheduling_UE_deployment_us:float=1e9):
        """
        The following function set the default value for the rescheduling_us of giulia events
        """
        self.dflt_rescheduling_us = dflt_rescheduling_us
        self.dflt_rescheduling_UE_deployment_us = dflt_rescheduling_UE_deployment_us

    def set_event_scheduling_info(self, event_scheduling_info: Optional[Dict[str, Any]] = None):
        """
        This function sets the event_scheduling_info dictionary, 
        in which are contained the self.dflt_rescheduling_us information for each 
        object.process event set within the giulia.configure() method
        """
        if event_scheduling_info is None or len(event_scheduling_info) == 0:
            
            # Generate default Case for event_scheduling_info
            event_scheduling_info = {
                "snapshot_control": self.dflt_rescheduling_us,
                "eventDriven_obj": self.dflt_rescheduling_us,
                "ue_deployment_obj": self.dflt_rescheduling_UE_deployment_us,
                "ue_antenna_array_structure_obj": self.dflt_rescheduling_us,
                "channel_sn_obj": self.dflt_rescheduling_us,
                "distance_angles_ue_to_cell_obj": self.dflt_rescheduling_us,
                "distance_angles_ueAnt_to_cellAnt_obj": self.dflt_rescheduling_us,
                "antenna_pattern_gain_ue_to_cell_obj": self.dflt_rescheduling_us,
                "array_steering_vector_ue_to_cell_obj": self.dflt_rescheduling_us,
                "LOS_probability_ue_to_cell_obj": self.dflt_rescheduling_us,
                "K_factor_ue_to_cell_obj": self.dflt_rescheduling_us,
                "path_loss_ue_to_cell_obj": self.dflt_rescheduling_us,
                "o2i_penetration_loss_ue_to_cell_obj": self.dflt_rescheduling_us,
                "shadowing_gain_ue_to_cell_obj": self.dflt_rescheduling_us,
                "slow_channel_gain_ue_to_cell_obj": self.dflt_rescheduling_us,
                "los_channel_gain_ue_to_cell_obj": self.dflt_rescheduling_us,
                
                "time_frequency_resource_obj": 
                    {"update_ue_carrier_info":self.dflt_rescheduling_us},
                    
                "dl_fast_fading_gain_ueAnt_to_cellAnt_obj": self.dflt_rescheduling_us,
                "channel_gain_ue_to_cell_obj": self.dflt_rescheduling_us,
                "SSB_precoded_channel_gain_no_fast_fading_ue_to_cell_obj": self.dflt_rescheduling_us,
                "SSB_precoded_channel_gain_ue_to_cell_obj": self.dflt_rescheduling_us,
                "CSI_RS_precoded_channel_gain_ue_to_cell_obj": self.dflt_rescheduling_us,
                "dl_noise_ue_to_cell_obj": self.dflt_rescheduling_us,
                "SSB_RSS_per_PRB_ue_to_cell_obj": self.dflt_rescheduling_us,
                "CSI_RS_RSS_per_PRB_ue_to_cell_obj": self.dflt_rescheduling_us,
                "CRS_RSRP_no_fast_fading_ue_to_cell_obj": self.dflt_rescheduling_us,
        
                "SSB_RSRP_no_fast_fading_ue_to_cell_obj": {
                    "process": self.dflt_rescheduling_us,
                    "set_RSRP_ue_to_cell_dBm": self.dflt_rescheduling_us
                },
        
                "SSB_RSRP_ue_to_cell_obj": self.dflt_rescheduling_us,
                "CSI_RS_RSRP_ue_to_cell_obj": self.dflt_rescheduling_us,
        
                "best_serving_cell_ID_per_ue_based_on_CRS_obj": {
                    "process": self.dflt_rescheduling_us,
                    "calculate_server_stats": self.dflt_rescheduling_us
                },
        
                "best_serving_cell_ID_per_ue_based_on_SSB_obj": {
                    "process": self.dflt_rescheduling_us,
                    "calculate_server_stats": self.dflt_rescheduling_us
                },
        
                "cell_reselection_based_on_CRS_obj": self.dflt_rescheduling_us,
                "cell_reselection_based_on_SSB_obj": self.dflt_rescheduling_us,
                "best_serving_CSI_RS_per_ue_obj": self.dflt_rescheduling_us,
                "CRS_sinr_ue_to_cell_obj": self.dflt_rescheduling_us,
                "SSB_sinr_ue_to_cell_obj": self.dflt_rescheduling_us,

                "base_stations_obj": {
                    "process": self.dflt_rescheduling_us,
                    "mac": self.dflt_rescheduling_us,
                    "carrier_shutdown": self.dflt_rescheduling_us,
                },
                

                "CSI_RS_sinr_per_PRB_ue_to_cell_obj": self.dflt_rescheduling_us,
                "ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj": self.dflt_rescheduling_us,
                "ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_obj": self.dflt_rescheduling_us,
                "power_consumption_obj": self.dflt_rescheduling_us,
                "performance_obj": self.dflt_rescheduling_us,
                "ue_mobility_obj": self.dflt_rescheduling_us,
                "bs_mobility_obj": self.dflt_rescheduling_us
            }

        # Set object attribute
        self.event_scheduling_info: dict = event_scheduling_info
