#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 17:05:52 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@itea.upv.es
"""

import copy
import math
import os
import random
import sys
import time
from typing import List

import geopandas
import numpy as np
import pandas as pd
import shapely

from giulia.event_driven.snapshot_control import Snapshot_control
from giulia.fs import results_file, data_driven_extras
from giulia.playground import sites, hotspots
from giulia.tools import tools
from giulia.tools.tools import log_calculations_time, TrackedDataFrame
from giulia.logger import error

from giulia.config.sim_config import Simulation_Config
from giulia.playground.sites import Site
from giulia.playground.hotspots import Hotspot
from giulia.bs.bs_deployments import Network
from giulia.outputs.saveable import Saveable

class UE_Deployment(Saveable):
    
    def __init__(self, 
                 simulation_config_obj: Simulation_Config, 
                 network_deployment_obj: Network, 
                 ue_playground_deployment_obj:Site,
                 ue_hotspot_deployment_obj:Hotspot):
        
        super().__init__()
                
        ##### Plots 
        ########################
        
        self.plot = 0 # Switch to control plots if any
        
        ##### Input storage
        ########################   
        
        self.simulation_config_obj: Simulation_Config = simulation_config_obj
        self.network_deployment_obj: Network = network_deployment_obj
        self.ue_playground_deployment_obj: Site = ue_playground_deployment_obj
        self.ue_hotspot_deployment_obj: Hotspot = ue_hotspot_deployment_obj
     
        ##### Outputs 
        ########################         
          
        self.df_ep = []
       
        self.ue_position_m = []
        self.ue_geographical_area = []
        
        self.ue_category: None|str = None
        self.indoor_ue_ratio = None
        self.min_bulding_floors = None
        self.max_bulding_floors = None
        self.ue_indoor_vector = None 
        
        self.velocity_kmh = None        
        self.noise_figure_dB = None
        
        self.antenna_config_Mg = None
        self.antenna_config_Ng = None
        self.antenna_config_M = None
        self.antenna_config_N = None
        self.antenna_config_P = None
        self.antenna_config_orientation_omega_alpha_degrees = None
        self.antenna_config_orientation_omega_beta_degrees = None
        self.antenna_config_orientation_omega_gamma_degrees = None
        
        self.ue_antenna_element_type = None
        self.ue_antenna_element_config_max_gain_dBi = None
        
        self.ue_to_cell_association = None
        
        self.number_of_ues_indoor = None
        self.number_of_indoor_ues_per_cell = None
        self.number_of_ues_outdoor = None       
        self.number_of_outdoor_ues_per_cell = None
 
        self.grid_resol_m = None
          
        self.number_of_ues_per_UEsite_cell: list|np.ndarray = []
        self.number_of_ues = 0
        self.number_of_aerial_ues_per_UEsite_cell: list| np.ndarray = []
        self.number_of_aerial_ues = 0

        # For Sionna RT
        self.dataset_import_folder = data_driven_extras('loaded')
        
        # Get simulation_config_obj ue_deployDensity_info
        self.ue_info_obj: dict|Simulation_Config._UE_info_deploy = self.simulation_config_obj.ue_deployDensity_info

        # Get Avg UE height
        self._set_bs_position(self.network_deployment_obj)
        


    @property
    def label(self) -> str: return 'ue_kpi'
    
    def expose_variables(self) -> List[str]:
        """List of attributes name to be saved."""
        return []
       
        
    def process(self, rescheduling_us=-1): 
        
        ##### Process inputs
        ######################## 
        
        # Random numbers
        
        self.myrandom = random.Random(self.simulation_config_obj.random_seed)
        self.rng = np.random.RandomState(self.simulation_config_obj.random_seed+0)   
            
        # Simulation variables
        
        self.project_name = self.simulation_config_obj.project_name
        self.ue_playground_model = self.ue_playground_deployment_obj.playground_model
       
        
        ##### Start timer
        ########################     
       
        t_start = time.perf_counter()        
           
        
        ##### Switch
        ########################
        
        if self.ue_playground_model == "ITU_R_M2135_UMa" or\
            self.ue_playground_model == "ITU_R_M2135_UMa_multilayer" or\
            self.ue_playground_model == "ITU_R_M2135_UMa_Umi_colocated_multilayer" or\
            self.ue_playground_model == "ITU_R_M2135_UMa_Umi_noncolocated_multilayer" : 
                
            self.df_ep = self.construct_ue_deployment_ITU_R_M2135_UMa(self.ue_playground_deployment_obj, self.ue_hotspot_deployment_obj)
            
            
        elif self.ue_playground_model == "ITU_R_M2135_UMi": 

            self.df_ep = self.construct_ue_deployment_ITU_R_M2135_UMi(self.ue_playground_deployment_obj, self.ue_hotspot_deployment_obj) 
            
        elif self.ue_playground_model == "3GPPTR36_814_Case_1" or\
            self.ue_playground_model == "3GPPTR36_814_Case_1_omni" or\
            self.ue_playground_model == "3GPPTR36_814_Case_1_single_bs" or\
            self.ue_playground_model == "3GPPTR36_814_Case_1_single_bs_omni": 
                
            self.df_ep = self.construct_ue_deployment_3GPPTR36_814_Case_1(self.ue_playground_deployment_obj, self.ue_hotspot_deployment_obj)   
            
        elif self.ue_playground_model == "3GPPTR38_901_UMa_C1" or\
            self.ue_playground_model == "3GPPTR38_901_UMa_C2": 

            self.df_ep = self.construct_ue_deployment_3GPPTR38_901_UMa(self.ue_playground_deployment_obj, self.ue_hotspot_deployment_obj)
            
            
        elif self.ue_playground_model == "3GPPTR38_901_UMa_lsc" or\
            self.ue_playground_model == "3GPPTR38_901_UMa_lsc_sn" or\
            self.ue_playground_model == "3GPPTR38_901_UMa_lsc_single_bs"  or\
            self.ue_playground_model == "3GPPTR38_901_UMa_lsc_single_sector" or\
            self.ue_playground_model == "3GPPTR38_901_UMa_2GHz_lsc" or\
            self.ue_playground_model == "3GPPTR38_901_UMa_C_band_lsc" or\
            self.ue_playground_model == "3GPPTR38_811_Urban_NTN" or\
            self.ue_playground_model == "3GPPTR38_811_Dense_Urban_NTN" or\
            self.ue_playground_model == "3GPPTR38_811_Dense_Urban_HAPS_ULA" or\
            self.ue_playground_model == "3GPPTR38_811_Dense_Urban_HAPS_UPA" or\
            self.ue_playground_model == "3GPPTR38_811_Dense_Urban_HAPS_Reflector" or\
            self.ue_playground_model == "3GPPTR38_811_Dense_Urban_HAPS_TriReflector" or\
            self.ue_playground_model == "3GPPTR38_811_Dense_Urban_SingleHAPS_Reflector" or\
            self.ue_playground_model == "3GPPTR38_811_Dense_Urban_SingleHAPS_TriReflector":
            
            self.df_ep = self.construct_ue_deployment_3GPPTR38_901_UMa_large_scale_calibration(self.ue_playground_deployment_obj, self.ue_hotspot_deployment_obj) 
            
            
        elif self.ue_playground_model == "3GPPTR38_901_4G" or \
            self.ue_playground_model == "3GPPTR38_901_5G" or \
            self.ue_playground_model == "3GPPTR38_901_6G" or \
            self.ue_playground_model == "3GPPTR38_901_4G5G_multilayer" or \
            self.ue_playground_model == "3GPPTR38_901_4G_5G_multilayer" or \
            self.ue_playground_model == "3GPPTR38_901_4G_5G2_multilayer" or \
            self.ue_playground_model == "3GPPTR38_901_4G_5G6G_multilayer" or \
            self.ue_playground_model == "3GPPTR38_901_4G_5G_6G_multilayer" or \
            self.ue_playground_model == "3GPPTR38_901_4G5G_cell_reselection" or \
            self.ue_playground_model == "3GPPTR38_811_Dense_Urban_singleHAPS_UPA_4G5G" or \
            self.ue_playground_model == "3GPPTR38_811_Dense_Urban_singleHAPS_UPA_4G_5G" or \
            self.ue_playground_model == "3GPPTR38_811_Dense_Urban_singleHAPS_reflector_4G5G": 
            
            self.df_ep = self.construct_ue_deployment_3GPPTR38_901_multilayer_large_scale_calibration(self.ue_playground_deployment_obj, self.ue_hotspot_deployment_obj)              
            
        elif self.ue_playground_model == "3GPPTR38_901_UMi_C1" or\
            self.ue_playground_model == "3GPPTR38_901_UMi_C2" : 
                
            self.df_ep = self.construct_ue_deployment_3GPPTR38_901_UMi(self.ue_playground_deployment_obj, self.ue_hotspot_deployment_obj)
            
        elif self.ue_playground_model == "3GPPTR38_901_UMi_lsc" or\
            self.ue_playground_model == "3GPPTR38_901_UMi_C_band_lsc" or\
            self.ue_playground_model == "3GPPTR38_901_UMi_fr3_lsc" or\
            self.ue_playground_model == "3GPPTR38_901_UPi_fr3_lsc":
                
            self.df_ep = self.construct_ue_deployment_3GPPTR38_901_UMi_large_scale_calibration(self.ue_playground_deployment_obj, self.ue_hotspot_deployment_obj)   
            
        elif self.ue_playground_model == "3GPPTR36_777_UMa_AV" : 
            
            ue_aerial_height_m = self.simulation_config_obj.uav_height_m # None, 50, 100, 200, 300
            self.df_ep = self.construct_ue_deployment_3GPPTR36_777_UMa_AV(self.ue_playground_deployment_obj, self.ue_hotspot_deployment_obj, ue_aerial_height_m) 
            
        elif self.ue_playground_model == "3GPPTR36_777_UMi_AV" : 
            
            ue_aerial_height_m = self.simulation_config_obj.uav_height_m # None, 50, 100, 200, 300
            self.df_ep = self.construct_ue_deployment_3GPPTR36_777_UMi_AV(self.ue_playground_deployment_obj, self.ue_hotspot_deployment_obj, ue_aerial_height_m)  
            
        elif self.ue_playground_model == "rectangular"  or self.ue_playground_model == "circular" or self.ue_playground_model == "3GPPTR36_814_Case_1_omni_dana" : 
            
            self.df_ep = self.construct_ue_deployment_polygon(self.ue_playground_deployment_obj, self.ue_hotspot_deployment_obj)   
            
        elif self.ue_playground_model == "dataset_rectangular"  or self.ue_playground_model == "dataset_circular"  :
            
            self.df_ep = self.construct_ue_deployment_polygon_dataset(self.ue_playground_deployment_obj, self.ue_hotspot_deployment_obj)
                    
        else: 
            raise ValueError(
                f"Error! In {self.__class__.__name__}, {self.__class__.__name__}.ue_playground_model not correctly specified. Value '{self.ue_playground_model}' not admitted.")
        
        # Set UE Target Traffic
        self.set_ue_targetRate(model_targetRate='random_uniform')
        
        ##### Save to plot 
        ########################  
        
        snapshot_control = Snapshot_control.get_instance()
        if self.simulation_config_obj.plot == 1 and snapshot_control.num_snapshots == 0:
            
            file = results_file(self.project_name, 'to_plot_ue_deployment')

            # Checks
            if not isinstance(self.network_deployment_obj.df_ep, pd.DataFrame): raise TypeError(
                f"In {self.__class__.__name__}.process network_deployment_obj.df_ep must be a pandas DataFrame")
            if self.ue_playground_deployment_obj.isd_m is None: raise TypeError(
                f"In {self.__class__.__name__}.process {self.ue_playground_deployment_obj.__class__.__name__}.isd_m must be defined")

            # Create mask to get first occurrence of each site_name
            mask = ~self.network_deployment_obj.df_ep["site_name"].duplicated()
            
            # Use mask to extract unique site rows
            unique_sites_df = self.network_deployment_obj.df_ep.loc[mask]
            
            # Save data
            np.savez(
                file,
                isd_m=self.ue_playground_deployment_obj.isd_m,
                site_names=self.network_deployment_obj.df_ep["site_name"].drop_duplicates().astype(str).to_numpy(),
                cell_site_positions_m=unique_sites_df[["position_x_m", "position_y_m", "position_z_m"]].astype(np.single).to_numpy(),
                ue_position_m=self.df_ep[["position_x_m", "position_y_m", "position_z_m"]],
                hotspot_position_m=self.ue_hotspot_deployment_obj.hotspot_position_m
            )   
        
        ##### End 
        ########################
        log_calculations_time('UE deployment', t_start)

        return rescheduling_us             
    

    def _set_bs_position(self, network_deployment_obj):
        if not isinstance(network_deployment_obj.df_ep, pd.DataFrame):
            raise TypeError(
                f"In {self.__class__.__name__}._set_bs_position network_deployment_obj.df_ep must be a pandas DataFrame")
        
        # Get BS positions
        bs_position_x = network_deployment_obj.df_ep['position_x_m'].to_numpy()
        bs_position_y = network_deployment_obj.df_ep['position_y_m'].to_numpy()
        bs_position_z = network_deployment_obj.df_ep['position_z_m'].to_numpy()

        # Store BS positions
        self._bs_position_m = np.array([bs_position_x, bs_position_y, bs_position_z]).T

        return


    def _calculate_ue_locations_at_random_per_Cell(self, 
                                                   network_deployment_obj, 
                                                   ue_playground_deployment_obj, 
                                                   ue_hotspot_deployment_obj) -> None:
        """
        Calculate UE locations at random within each cell.

        This function compute the 2D positions of the UEs within each cell,
        ensuring that they are uniformly distributed and respect the sector.
        This method use the information of actually deplyed in Giulia cells.

        It updates self.ue_position_m and self.ue_geographical_area attributes.
        """
        raise DeprecationWarning("_calculate_ue_locations_at_random_per_Cell is deprecated. Use construct_ue_deployment_... methods instead.")
        raise NotImplementedError("This method is deprecated.")
        # Checks 
        if not isinstance(self.ue_position_m, np.ndarray): raise TypeError(
            f"in {self.__class__.__name__}.ue_position_m must be a numpy array within the {self.ue_playground_model} scenario")
             
        number_of_ues_per_cell = np.full(57, np.nan)
        
        # Get Scenario features 
        grid_resol_m = ue_playground_deployment_obj.grid_resol_m
        scenario_lower_left_conner_m = ue_playground_deployment_obj.scenario_lower_left_conner_m   
        scenario_x_side_length_m = ue_playground_deployment_obj.scenario_x_side_length_m 
        scenario_y_side_length_m = ue_playground_deployment_obj.scenario_y_side_length_m   

        # Get cell list
        df_ep = network_deployment_obj.df_ep
        cell_idx_list = df_ep["ID"].tolist()

        # Initialize counters for dpeloyment
        ues_in_scenario_count = 0
        hotspot_id = 0

        # Start per cell deployment
        for cell_idx in cell_idx_list:
            #### Get Info
            # Get cell with same site_ID in df_ep
            site_id = df_ep.loc[df_ep["ID"] == cell_idx, "site_ID"].iloc[0]
            # Get cell bearing
            cell_bearing_deg = df_ep.loc[df_ep["ID"] == cell_idx, "antenna_config_hor_alpha_mec_bearing_deg"].iloc[0]
            # Get all bearing of all sector in the site
            alpha_mec_bearing_deg_in_site = df_ep.loc[df_ep["site_ID"] == site_id, "antenna_config_hor_alpha_mec_bearing_deg"].values
            # Get site position
            site_position = df_ep.loc[df_ep["site_ID"] == site_id, ["position_x_m", "position_y_m"]].iloc[0].to_numpy()
            # Get sector info
            angle_range_deg = tools._get_sector_bearing_info_for_angle(alpha_mec_bearing_deg_in_site, cell_bearing_deg)['width_deg']

            ### Start Computation
            # Calculate ang_a and ang_b
            ang_a = tools.angle_range_0_180(cell_bearing_deg-angle_range_deg/2)
            ang_b = tools.angle_range_0_180(cell_bearing_deg+angle_range_deg/2)

            ## UNIFORM DEPLOYMENT WITHIN THE CELL
            
            ues_in_cell_count = 0
            if (ue_hotspot_deployment_obj.number_of_hotspot_per_cell[cell_idx] > 0):
                ues_in_cell_to_deploy = np.round(number_of_ues_per_cell[cell_idx] * (1 - ue_hotspot_deployment_obj.faction_of_ues_in_hotspots_per_cell[cell_idx])).astype(int)  
            else:
                ues_in_cell_to_deploy = number_of_ues_per_cell[cell_idx] 
                
                
            while ues_in_cell_to_deploy > 0 : #deploy as many UEs as needed
                # Get location within hexagon of flat-top orientation with size = 1 (seee htps://www.redblobgames.com/grids/hexagons/)
                ue_position = tools.randinunithex(self.myrandom, ue_playground_deployment_obj.isd_m)
                
                # Derive angle to check whether it is within this sector range
                angle_deg = math.degrees(np.arctan2(ue_position[1],ue_position[0]))

                # Get distance and minimum distance to cell site to check whether it is within this sector boundaries 
                if self.ue_type_in_cell[cell_idx][number_of_ues_per_cell[cell_idx] - ues_in_cell_to_deploy] == 2: #In case of UAVs, we need to look at the 3D distance
                    rel_ue_position_3d = np.append(ue_position, self.ue_position_m[ues_in_scenario_count, 2] - network_deployment_obj.bs_antenna_height_m)
                    distance_m = np.linalg.norm(rel_ue_position_3d,ord=2)
                    min_distance_m = ue_playground_deployment_obj.min_BS_to_AV_3D_distance_m
                else:
                    distance_m = np.linalg.norm(ue_position,ord=2)
                    min_distance_m = ue_playground_deployment_obj.min_BS_to_UE_2D_distance_m

                # Check conditions 
                if (tools.isBetween(ang_a, ang_b, angle_deg) 
                    and distance_m >= min_distance_m): # Min distance to cell site
                    
                    # Offset the position
                    # print("to check")
                    ue_position += site_position
                    
                    if (ue_position[0] > scenario_x_side_length_m/2 
                        or ue_position[0] < -scenario_x_side_length_m/2 
                        or ue_position[1] > scenario_y_side_length_m/2 
                        or ue_position[1] < -scenario_y_side_length_m/2):
                        
                        sys.exit('Error: Deployed UEs are outside scenario limits!')   
                    
                    # Store position
                    self.ue_position_m[ues_in_scenario_count, 0:2] = [ue_position[0], ue_position[1]]
                    # print("to ask david")
                    self.ue_geographical_area[ues_in_scenario_count] = site_id
                    
                    # Update counters
                    ues_in_cell_to_deploy -= 1
                    ues_in_cell_count += 1
                    ues_in_scenario_count += 1


            ## HOPSPOT DEPLOYMENT WITHIN THE CELL
            ues_per_hotspot = 0
            if ue_hotspot_deployment_obj.number_of_hotspot_per_cell[cell_idx] > 0:
                ues_per_hotspot = np.round(number_of_ues_per_cell[cell_idx] * ue_hotspot_deployment_obj.faction_of_ues_in_hotspots_per_cell[cell_idx] / ue_hotspot_deployment_obj.number_of_hotspot_per_cell[cell_idx]).astype(int)  
            hotspots_in_cell_count = ue_hotspot_deployment_obj.number_of_hotspot_per_cell[cell_idx]
            
            #Deploy as many hotspots as needed
            while hotspots_in_cell_count > 0 : 
                if hotspots_in_cell_count == 1:
                    ues_per_hotspot = number_of_ues_per_cell[cell_idx] - ues_in_cell_count

                # Generate random angles
                angles_rad = self.rng.uniform(0, 2 * np.pi, ues_per_hotspot)

                # Generate random radii (within the given radius)
                radii_m = ue_hotspot_deployment_obj.min_UE_to_hotspot_distance_m + \
                    np.sqrt(self.rng.uniform(0, 1, ues_per_hotspot)) * (ue_hotspot_deployment_obj.hotspot_radius_m - ue_hotspot_deployment_obj.min_UE_to_hotspot_distance_m)
                        
                # Convert polar coordinates to Cartesian coordinates
                ue_position = np.column_stack(( radii_m * np.cos(angles_rad), radii_m * np.sin(angles_rad))) + ue_hotspot_deployment_obj.hotspot_position_m[hotspot_id]   
                
                if (np.any(ue_position[:,0] > scenario_x_side_length_m/2) 
                    or np.any(ue_position[:,0] < -scenario_x_side_length_m/2) 
                    or np.any(ue_position[:,1] > scenario_y_side_length_m/2) 
                    or np.any(ue_position[:,1] < -scenario_y_side_length_m/2)):
                    
                    continue  

                # Store UE position
                self.ue_position_m[ues_in_scenario_count:ues_in_scenario_count+ues_per_hotspot,0:2] = ue_position
                self.ue_geographical_area[ues_in_scenario_count:ues_in_scenario_count+ues_per_hotspot] = site_id

                # Update counters
                hotspot_id += 1 
                hotspots_in_cell_count -= 1
                ues_in_cell_count += ues_per_hotspot  
                ues_in_scenario_count += ues_per_hotspot  


        # Finally, translate UE positions to grid points
        self.ue_grid_position = tools.translate_position_to_grid_point(self.ue_position_m[:,0:2], 
                                                                       scenario_lower_left_conner_m, 
                                                                       grid_resol_m)
        return


    def calculate_ue_locations_at_random_per_site_using_hexagon(self, 
                                                                ue_playground_deployment_obj: Site,
                                                                ue_hotspot_deployment_obj: Hotspot) -> None:
       
       
        
        # Checks 
        if not isinstance(self.ue_position_m, np.ndarray): raise TypeError(
            f"in {self.__class__.__name__}.ue_position_m must be a numpy array within the {self.ue_playground_model} scenario")
        if ue_playground_deployment_obj.ref_number_of_cell_sites is None: raise TypeError(
            f"in {self.__class__.__name__}.ref_number_of_cell_sites must be defined in {ue_playground_deployment_obj.__class__.__name__}")
        if ue_playground_deployment_obj.ref_number_of_sectors_per_site is None: raise TypeError(
               f"ref_number_of_sectors_per_site must be defined in {ue_playground_deployment_obj.__class__.__name__}")
        if ue_playground_deployment_obj.ref_antenna_config_hor_alpha_mec_bearing_deg is None: raise TypeError(
               f"ref_antenna_config_hor_alpha_mec_bearing_deg must be defined in {ue_playground_deployment_obj.__class__.__name__}")
        if ue_playground_deployment_obj.min_BS_to_UE_2D_distance_m is None: raise TypeError(
               f"min_BS_to_UE_2D_distance_m must be defined in {ue_playground_deployment_obj.__class__.__name__}")
        if not isinstance(ue_playground_deployment_obj.scenario_x_side_length_m, (int, float)): raise TypeError(
               f"scenario_x_side_length_m must be a int or float in {ue_playground_deployment_obj.__class__.__name__}")
        if not isinstance(ue_playground_deployment_obj.scenario_y_side_length_m, (int, float)): raise TypeError(
               f"scenario_y_side_length_m must be a int or float in {ue_playground_deployment_obj.__class__.__name__}")
       
       
        # Scenario features 
        grid_resol_m = ue_playground_deployment_obj.grid_resol_m
        scenario_lower_left_conner_m = ue_playground_deployment_obj.scenario_lower_left_conner_m   
        scenario_x_side_length_m = ue_playground_deployment_obj.scenario_x_side_length_m 
        scenario_y_side_length_m = ue_playground_deployment_obj.scenario_y_side_length_m     
        
        ### Compute UE locations - uniform deployment within the cell
        cell_id = int(0)
        hotspot_id = int(0)
        ues_in_scenario_count = int(0)

        if ue_playground_deployment_obj.ref_number_of_cells is None: raise TypeError(
            f"ref_number_of_cells must be defined in {ue_playground_deployment_obj.__class__.__name__}")
        
        for cell_ue in range(ue_playground_deployment_obj.ref_number_of_cells):

            bearing = ue_playground_deployment_obj.map_cell_to_bearing[cell_ue]

            angle_range_deg = 360/ue_playground_deployment_obj.ref_number_of_sectors_per_site
            ang_a = tools.angle_range_0_180(bearing-angle_range_deg/2)
            ang_b = tools.angle_range_0_180(bearing+angle_range_deg/2)

               
            ## UNIFORM DEPLOYMENT WITHIN THE CELL
            ues_in_cell_count = 0
            if (ue_hotspot_deployment_obj.number_of_hotspot_per_cell[cell_id] > 0):
                ues_in_cell_to_deploy = np.round(self.number_of_ues_per_UEsite_cell[cell_id] * (1 - ue_hotspot_deployment_obj.faction_of_ues_in_hotspots_per_cell[cell_id])).astype(int)  
            else:
                ues_in_cell_to_deploy = self.number_of_ues_per_UEsite_cell[cell_id] 


            while ues_in_cell_to_deploy > 0 : #deploy as many UEs as needed                  
                # Get location within hexagon of flat-top orientation with size = 1 (seee htps://www.redblobgames.com/grids/hexagons/)
                ue_position = tools.randinunithex(self.myrandom, ue_playground_deployment_obj.isd_m)
                
                # Derive angle to check whether it is within this sector range
                angle_deg = math.degrees(np.arctan2(ue_position[1],ue_position[0]))
                
                # Get distance and minimum distance to cell site to check whether it is within this sector boundaries 
                ue_position_m_3D_aux = np.append(ue_position, self.ue_position_m[ues_in_scenario_count, 2])
                # In case of UAVs (type in cell == 2), we need to look at the 3D distance 
                if self.ue_type_in_cell[cell_id][self.number_of_ues_per_UEsite_cell[cell_id] - ues_in_cell_to_deploy] == 2: 
                    # Compute 3D distance
                    distance_m = np.sqrt(np.square(ue_position_m_3D_aux[0] - self._bs_position_m[:, 0]) +
                                            np.square(ue_position_m_3D_aux[1] - self._bs_position_m[:, 1]) +
                                            np.square(ue_position_m_3D_aux[2] - self._bs_position_m[:, 2]) )
                    # Get min 3D distance
                    min_distance_m = ue_playground_deployment_obj.min_BS_to_AV_3D_distance_m
                else:
                    # Compute 2D distance
                    distance_m = np.sqrt(np.square(ue_position_m_3D_aux[0] - self._bs_position_m[:, 0]) +
                                            np.square(ue_position_m_3D_aux[1] - self._bs_position_m[:, 1]) )
                    # Get min 2D distance
                    min_distance_m = ue_playground_deployment_obj.min_BS_to_UE_2D_distance_m
                del ue_position_m_3D_aux

                # Check conditions 
                if (tools.isBetween(ang_a, ang_b, angle_deg) 
                    and all(distance_m >= min_distance_m)): # Min distance to cell site
                    
                    # Offset the position
                    site_index = ue_playground_deployment_obj.map_cell_to_site[cell_ue]
                    ue_position += ue_playground_deployment_obj.ref_cell_site_positions_m[site_index]
                    
                    if (ue_position[0] > scenario_x_side_length_m/2 + ue_playground_deployment_obj.layout_centre_m[0]
                        or ue_position[0] < -scenario_x_side_length_m/2 + ue_playground_deployment_obj.layout_centre_m[0]
                        or ue_position[1] > scenario_y_side_length_m/2 + ue_playground_deployment_obj.layout_centre_m[1]
                        or ue_position[1] < -scenario_y_side_length_m/2 + ue_playground_deployment_obj.layout_centre_m[1]):

                        error("Error: Deployed UEs are outside scenario limits! line 531 ue_deployment.py")
                        #sys.exit('Error: Deployed UEs are outside scenario limits!')
                        pass

                    # Store position
                    self.ue_position_m[ues_in_scenario_count,0:2] = [ue_position[0], ue_position[1]]
                    self.ue_geographical_area[ues_in_scenario_count] = site_index
                    
                    # Update counters
                    ues_in_cell_to_deploy -= 1
                    ues_in_cell_count += 1
                    ues_in_scenario_count += 1
            
            ## HOPSPOT DEPLOYMENT WITHIN THE CELL
            ues_per_hotspot = int(0)
            if ue_hotspot_deployment_obj.number_of_hotspot_per_cell[cell_id] > 0:
                ues_per_hotspot = np.round(self.number_of_ues_per_UEsite_cell[cell_id] * ue_hotspot_deployment_obj.faction_of_ues_in_hotspots_per_cell[cell_id] / ue_hotspot_deployment_obj.number_of_hotspot_per_cell[cell_id]).astype(int)  
            hotspots_in_cell_count = ue_hotspot_deployment_obj.number_of_hotspot_per_cell[cell_id]
            
            # Deploy as many hotspots as needed
            while hotspots_in_cell_count > 0 : 

                if hotspots_in_cell_count == 1:
                    ues_per_hotspot = self.number_of_ues_per_UEsite_cell[cell_id] - ues_in_cell_count
                    
                # Generate random angles
                angles_rad = self.rng.uniform(0, 2 * np.pi, ues_per_hotspot)
                
                # Generate random radii (within the given radius)
                radii_m = ue_hotspot_deployment_obj.min_UE_to_hotspot_distance_m + \
                    np.sqrt(self.rng.uniform(0, 1, ues_per_hotspot)) * (ue_hotspot_deployment_obj.hotspot_radius_m - ue_hotspot_deployment_obj.min_UE_to_hotspot_distance_m)
                        
                # Convert polar coordinates to Cartesian coordinates
                ue_position = np.column_stack(( radii_m * np.cos(angles_rad), radii_m * np.sin(angles_rad))) + ue_hotspot_deployment_obj.hotspot_position_m[hotspot_id]   
                
                if (np.any(ue_position[:,0] > scenario_x_side_length_m/2) 
                    or np.any(ue_position[:,0] < -scenario_x_side_length_m/2) 
                    or np.any(ue_position[:,1] > scenario_y_side_length_m/2) 
                    or np.any(ue_position[:,1] < -scenario_y_side_length_m/2)):
                    
                    continue  
                    
                # Store UE position
                self.ue_position_m[ues_in_scenario_count:ues_in_scenario_count+ues_per_hotspot, 0:2] = ue_position
                self.ue_geographical_area[ues_in_scenario_count:ues_in_scenario_count+ues_per_hotspot] = np.full(ues_per_hotspot, site_index)                           
                
                # Update counters
                hotspot_id += 1 
                hotspots_in_cell_count -= 1
                ues_in_cell_count += ues_per_hotspot  
                ues_in_scenario_count += ues_per_hotspot  
                    
            # Update counters
            cell_id += 1               
        
        # Store the grid indeces for each UE   
        self.ue_grid_position = tools.translate_position_to_grid_point(self.ue_position_m[:,0:2], 
                                                                       scenario_lower_left_conner_m, 
                                                                       grid_resol_m)
       
        return   
                   
    def calculate_ue_locations_in_grid_within_rectangular_playground(self, 
                                                                     ue_playground_deployment_obj): 
            
        # Scenario features 
        hex_layout_tiers = ue_playground_deployment_obj.hex_layout_tiers
        isd_m = ue_playground_deployment_obj.isd_m
        grid_resol_m = ue_playground_deployment_obj.grid_resol_m
        scenario_x_side_length_m = ue_playground_deployment_obj.scenario_x_side_length_m 
        scenario_y_side_length_m = ue_playground_deployment_obj.scenario_y_side_length_m 
        hexagons_gp = ue_playground_deployment_obj.hexagons_gp
        
        ### Compute UE locations 
        # Generate square grid locations
        steps_x = (int)(np.floor(scenario_x_side_length_m/ grid_resol_m))
        margin_x_m = (scenario_x_side_length_m - steps_x * grid_resol_m) /2
        range_x_m = np.linspace(-scenario_x_side_length_m/2 + margin_x_m, scenario_x_side_length_m/2 - margin_x_m, steps_x)
        
        steps_y = (int)(np.floor(scenario_y_side_length_m/ grid_resol_m))
        margin_y_m = (scenario_y_side_length_m - steps_y * grid_resol_m) /2
        range_y_m = np.linspace(-scenario_y_side_length_m/2 + margin_y_m, scenario_y_side_length_m/2 - margin_y_m, steps_y)        
        
        grid_points_m = np.array([(step_x,step_y) for  step_x in range_x_m for step_y in range_y_m])
        
        # Calculate distance to centre
        grid_distances_m =  np.linalg.norm(grid_points_m,axis=1) 
         
        # Initialize indicator, indicating whether a grid point of the square grid is within the hexagonal grid 
        isIn = np.ones((steps_x*steps_y),dtype=bool) * np.nan          
        
        # Check whether the grid locations are within the inner circle of the hexagonal grid 
        if hex_layout_tiers == 0:
            isIn[grid_distances_m <= (0.5) * isd_m] = True 
        else: 
            isIn[grid_distances_m <= (hex_layout_tiers) * isd_m] = True        
           
        # Check whether the rest of the points are within an hexagone
        # Identify those points
        in_between_positions_m = grid_points_m[np.isnan(isIn)]
        # Prepare points for the check
        points = geopandas.GeoSeries([shapely.geometry.Point(in_between_position_m) for in_between_position_m in in_between_positions_m])
        
        # Perform the check
        if hex_layout_tiers == 0:
            isIn_aux = np.any(hexagons_gp[0:].apply(lambda x: points.within(x)), axis=0)
        else:
            isIn_aux = np.any(hexagons_gp[1+6*(hex_layout_tiers-1):].apply(lambda x: points.within(x)), axis=0)
            
        # Update indicator
        isIn[np.isnan(isIn)] = isIn_aux  
                
        # Store UE locations 
               # Checks 
        if not isinstance(self.ue_position_m, np.ndarray): raise TypeError(
            f"in {self.__class__.__name__}.ue_position_m must be a numpy array within the {self.ue_playground_model} scenario")
        self.ue_position_m[:, :2] = grid_points_m[isIn == True]
        
        # Store the grid indeces for each UE  
        self.ue_grid_position = np.where(np.reshape(isIn, (steps_x, steps_y)))

        return     
    
    def calculate_ue_locations_at_random_within_rectangular_playground(self, 
                                                                       ue_playground_deployment_obj, 
                                                                       ue_hotspot_deployment_obj):
        
        if not isinstance(self.ue_position_m, np.ndarray): raise TypeError(
            f"in {self.__class__.__name__}.ue_position_m must be a numpy array within the {self.ue_playground_model} scenario")

        # Scenario features
        grid_resol_m = ue_playground_deployment_obj.grid_resol_m
        scenario_lower_left_conner_m = ue_playground_deployment_obj.scenario_lower_left_conner_m
        scenario_x_side_length_m = ue_playground_deployment_obj.scenario_x_side_length_m
        scenario_y_side_length_m = ue_playground_deployment_obj.scenario_y_side_length_m

        ### Compute UE locations

        ## UNIFORM DEPLOYMENT WITHIN THE AREA

        ues_at_random = np.round(self.number_of_ues * (1.0-ue_hotspot_deployment_obj.faction_of_ues_in_hotspots)).astype(int)

        # Case 1: Rectangular deployment without maps/scenes
        if ue_playground_deployment_obj.playground_model == "rectangular" or \
            self.ue_playground_model == "3GPPTR36_814_Case_1_omni_dana":
           self.ue_position_m[:ues_at_random, 0:2] = np.column_stack((
               self.rng.uniform(
                   scenario_lower_left_conner_m[0],
                   scenario_lower_left_conner_m[0] + scenario_x_side_length_m,
                   ues_at_random
               ),
               self.rng.uniform(
                   scenario_lower_left_conner_m[1],
                   scenario_lower_left_conner_m[1] + scenario_y_side_length_m,
                   ues_at_random
               )
           ))

        # Case 2: Dataset-based rectangular deployment (currently outdoor-only)
        elif ue_playground_deployment_obj.playground_model == "dataset_rectangular":

            # Note that meshes are centered around (0,0) but rectangular are is not
            # Compute the center of the rectangular area
            center_x = scenario_lower_left_conner_m[0] + scenario_x_side_length_m / 2
            center_y = scenario_lower_left_conner_m[1] + scenario_y_side_length_m / 2
            ply_folder = os.path.join(self.dataset_import_folder, 'meshes')

            # Get the list of files in the directory that start with 'indoorOutdoorData'
            user_deployment_files = [f for f in os.listdir(self.dataset_import_folder) if
                                    f.startswith('indoorOutdoorData')]

            # Check if there is more than one 'indoorOutdoorData' file
            if len(user_deployment_files) > 1:
                raise FileExistsError(
                    "Multiple 'indoorOutdoorData' files found in the specified folder. Please ensure there is at most one file.")

            # If there is exactly one 'indoorOutdoorData' file, proceed to load it
            elif len(user_deployment_files) == 1:
                indoor_outdoor_file = os.path.join(self.dataset_import_folder, user_deployment_files[0])
                print(f"Loading precomputed indoor/outdoor data from {indoor_outdoor_file}")

                # Load the data from the CSV
                data = pd.read_csv(indoor_outdoor_file)

                # Filter for outdoor points only (indoor == False)
                outdoor_points = data[data['indoor'] == False][['x', 'y']].values

                # Check if there are enough outdoor points to deploy users
                if len(outdoor_points) < ues_at_random:
                    raise ValueError(f"Not enough outdoor points in the precomputed data. "
                                    f"Needed: {ues_at_random}, Found: {len(outdoor_points)}")

                # Randomly sample from the available outdoor points
                sampled_indices = np.random.choice(len(outdoor_points), size=ues_at_random, replace=False)
                sampled_points = outdoor_points[sampled_indices]

                # Deploy the sampled outdoor users
                for idx, (candidate_x, candidate_y) in enumerate(sampled_points):
                    self.ue_position_m[idx, 0:2] = [candidate_x, candidate_y]
                    print(f"Deployed outdoor user at [{candidate_x}, {candidate_y}].")

            # If the precomputed file is not available, fall back to random generation
            else:
                print("Precomputed file not found, proceeding with random generation...")

                deployed_users = 0
                # TODO: Update this code with its version optimized for parallel computing
                while deployed_users < ues_at_random:
                    # Generate a random point
                    candidate_x = self.rng.uniform(scenario_lower_left_conner_m[0],
                                                  scenario_lower_left_conner_m[0] + scenario_x_side_length_m)
                    candidate_y = self.rng.uniform(scenario_lower_left_conner_m[1],
                                                  scenario_lower_left_conner_m[1] + scenario_y_side_length_m)
                    # Shift the candidate point by the center of the rectangular area to align with mesh coordinates
                    shifted_x = candidate_x - center_x
                    shifted_y = candidate_y - center_y
                    # Check if the point is outdoors
                    is_indoor = tools.check_point_in_buildings([shifted_x, shifted_y, 0], ply_folder, verbose=True)

                    if is_indoor:  # If the point is outdoors, deploy the user
                        print(f"User at [{candidate_x}, {candidate_y}] is indoor and was not deployed.")
                    else:
                        self.ue_position_m[deployed_users, 0:2] = [candidate_x, candidate_y]
                        deployed_users += 1  # Increment the count of deployed users
                        print(f"Deployed outdoor user at [{candidate_x}, {candidate_y}].")

        else:
            raise ValueError(f"Unsupported playground model: {ue_playground_deployment_obj.playground_model}")

        self.ue_position_m[:ues_at_random,0:2] = np.column_stack((self.rng.uniform(scenario_lower_left_conner_m[0], scenario_lower_left_conner_m[0] + scenario_x_side_length_m, ues_at_random),
                                                    self.rng.uniform(scenario_lower_left_conner_m[1], scenario_lower_left_conner_m[1] + scenario_y_side_length_m, ues_at_random)))

        #self.ue_position_m[:ues_at_random,0:2] = np.column_stack((self.rng.uniform(0, 10, ues_at_random),
        #                                             self.rng.uniform(0, 10, ues_at_random)))


        # test_number_of_ue = 10
        # test_distance_m = 100
        # test_square_cluster_length_m= 10
        # # Create the first cluster
        # cluster1_ue_position_m = np.column_stack((self.rng.uniform(0, test_square_cluster_length_m, test_number_of_ue),
        #                                            self.rng.uniform(0, test_square_cluster_length_m, test_number_of_ue)))

        # # Create the second cluster
        # cluster2_ue_position_m = np.column_stack((self.rng.uniform(test_distance_m, test_distance_m+test_square_cluster_length_m, test_number_of_ue),
        #                                            self.rng.uniform(test_distance_m, test_distance_m+test_square_cluster_length_m, test_number_of_ue)))

        # Combine both clusters
        #self.ue_position_m[:ues_at_random,0:2] = np.concatenate((cluster1_ue_position_m, cluster2_ue_position_m))


        ## CLUSTER DEPLOYMENT WITHIN THE AREA
       
        if ue_hotspot_deployment_obj.number_of_hotspots != 0:

            ues_per_hotspot = np.round(self.number_of_ues * ue_hotspot_deployment_obj.faction_of_ues_in_hotspots / ue_hotspot_deployment_obj.number_of_hotspots).astype(int)

            hotspot_id = int(0)
            hotspots_count = copy.copy(ue_hotspot_deployment_obj.number_of_hotspots)
            ues_in_scenario_count = ues_at_random
            while hotspots_count > 0 : #deploy as many hotspots as needed

                if hotspots_count == 1:
                    ues_per_hotspot = self.number_of_ues - ues_in_scenario_count

                # Generate random angles
                angles_rad = self.rng.uniform(0, 2 * np.pi, ues_per_hotspot)

                # Generate random radii (within the given radius)
                radii_m = ue_hotspot_deployment_obj.min_UE_to_hotspot_distance_m + \
                    np.sqrt(self.rng.uniform(0, 1, ues_per_hotspot)) * (ue_hotspot_deployment_obj.hotspot_radius_m - ue_hotspot_deployment_obj.min_UE_to_hotspot_distance_m)

                # Convert polar coordinates to Cartesian coordinates
                ue_position = np.column_stack(( radii_m * np.cos(angles_rad), radii_m * np.sin(angles_rad))) + ue_hotspot_deployment_obj.hotspot_position_m[hotspot_id]

                # Check boundary conditions
                if (np.any(ue_position[:,0] > scenario_x_side_length_m/2)
                    or np.any(ue_position[:,0] < -scenario_x_side_length_m/2)
                    or np.any(ue_position[:,1] > scenario_y_side_length_m/2)
                    or np.any(ue_position[:,1] < -scenario_y_side_length_m/2)):
                    continue
                    #sys.exit('Error: Deployed UEs are outside scenario limits!')

                # Store UE position
                self.ue_position_m[ues_in_scenario_count:ues_in_scenario_count+ues_per_hotspot,0:2] = ue_position

                # Plot
                #plt.plot([ue_position[0]],[ue_position[1]], 'o', color=color_vector[ue_playground_deployment_obj.ref_number_of_sectors_per_site*site_index + bearing_index])

                # Update counters
                hotspot_id += 1
                hotspots_count -= 1
                ues_in_scenario_count += ues_per_hotspot

        # Store the grid indices for each UE
        self.ue_grid_position = tools.translate_position_to_grid_point(self.ue_position_m[:,0:2],
                                                                          scenario_lower_left_conner_m,
                                                                          grid_resol_m)

        return
   
    def calculate_ue_locations_at_random_within_circular_playground(self, ue_playground_deployment_obj, 
                                                                    ue_hotspot_deployment_obj):
        
       # Checks
       if not isinstance(self.ue_position_m, np.ndarray): raise TypeError(
            f"in {self.__class__.__name__}.ue_position_m must be a numpy array within the {self.ue_playground_model} scenario")

       # Scenario features 
       grid_resol_m = ue_playground_deployment_obj.grid_resol_m     
       scenario_lower_left_conner_m = ue_playground_deployment_obj.scenario_lower_left_conner_m        
       scenario_center_m = ue_playground_deployment_obj.scenario_centre_m  
       scenario_radius_m = ue_playground_deployment_obj.scenario_radius_m
       
       ### Compute UE locations 
              
       ## UNIFORM DEPLOYMENT WITHIN THE AREA
       
       ues_at_random = np.round(self.number_of_ues * (1.0-ue_hotspot_deployment_obj.faction_of_ues_in_hotspots)).astype(int) 
       
       # Generate random angles
       angles_rad = self.rng.uniform(0, 2 * np.pi, ues_at_random)
       
       # Generate random radii (within the given radius)
       radii_m = np.sqrt(self.rng.uniform(0, 1, ues_at_random)) * (scenario_radius_m)   
       
       # Convert polar coordinates to Cartesian coordinates
       self.ue_position_m[:ues_at_random,0:2] = scenario_center_m + np.column_stack(( radii_m * np.cos(angles_rad), radii_m * np.sin(angles_rad))) 
       
        ## CLUSTER DEPLOYMENT WITHIN THE AREA
       
       if ue_hotspot_deployment_obj.number_of_hotspots != 0:
           
           ues_per_hotspot = np.round(self.number_of_ues * ue_hotspot_deployment_obj.faction_of_ues_in_hotspots / ue_hotspot_deployment_obj.number_of_hotspots).astype(int) 
           
           hotspot_id = int(0)
           hotspots_count = copy.copy(ue_hotspot_deployment_obj.number_of_hotspots)
           ues_in_scenario_count = ues_at_random
           while hotspots_count > 0 : #deploy as many hotspots as needed
           
                if hotspots_count == 1:
                    ues_per_hotspot = self.number_of_ues - ues_in_scenario_count                 
                   
                # Generate random angles
                angles_rad = self.rng.uniform(0, 2 * np.pi, ues_per_hotspot)
                
                # Generate random radii (within the given radius)
                radii_m = ue_hotspot_deployment_obj.min_UE_to_hotspot_distance_m + \
                    np.sqrt(self.rng.uniform(0, 1, ues_per_hotspot)) * (ue_hotspot_deployment_obj.hotspot_radius_m - ue_hotspot_deployment_obj.min_UE_to_hotspot_distance_m)   
                
                # Convert polar coordinates to Cartesian coordinates
                ue_position = np.column_stack(( radii_m * np.cos(angles_rad), radii_m * np.sin(angles_rad))) + ue_hotspot_deployment_obj.hotspot_position_m[hotspot_id]      
                                
                # Check boundary conditions
                if (np.any(np.linalg.norm(ue_position -scenario_center_m, axis=-1) > scenario_radius_m)):
                    continue
                    
                # Store UE position
                self.ue_position_m[ues_in_scenario_count:ues_in_scenario_count+ues_per_hotspot,0:2] = ue_position   
                
                # Update counters
                hotspot_id += 1 
                hotspots_count -= 1
                ues_in_scenario_count += ues_per_hotspot               
            
       # Store the grid indeces for each UE    
       
       self.ue_grid_position = tools.translate_position_to_grid_point(self.ue_position_m[:,0:2],
                                                                      scenario_lower_left_conner_m, 
                                                                      grid_resol_m)
       
       return        
     
    
    def calculate_sum_number_of_users_hexagonal_layouts(self):

        # Checks
        assert self.indoor_ue_ratio is not None, f"{self.__class__.__name__}.indoor_ue_ratio must be initialized."
        if not isinstance(self.number_of_ues_per_UEsite_cell, np.ndarray): raise TypeError(f"{self.__class__.__name__}.number_of_ues_per_UEsite_cell must be a numpy array.")
        if not isinstance(self.number_of_aerial_ues_per_UEsite_cell, np.ndarray): raise TypeError(f"{self.__class__.__name__}.number_of_aerial_ues_per_UEsite_cell must be a numpy array.")
        
        ### Compute number of UEs
        self.number_of_outdoor_ues_per_cell = np.round((1-self.indoor_ue_ratio) * (self.number_of_ues_per_UEsite_cell - self.number_of_aerial_ues_per_UEsite_cell)).astype(int)
        self.number_of_indoor_ues_per_cell = ((self.number_of_ues_per_UEsite_cell - self.number_of_outdoor_ues_per_cell - self.number_of_aerial_ues_per_UEsite_cell)).astype(int)
        
        self.number_of_outdoor_ues = np.sum(self.number_of_outdoor_ues_per_cell)        
        self.number_of_indoor_ues = np.sum(self.number_of_indoor_ues_per_cell)
        self.number_of_aerial_ues = np.sum(self.number_of_aerial_ues_per_UEsite_cell)
            
        self.number_of_ues = self.number_of_outdoor_ues + self.number_of_indoor_ues + self.number_of_aerial_ues  
        
    def calculate_sum_number_of_users_grid_layouts(self) -> None:
        
        ### Compute number of UEs
        self.number_of_outdoor_ues = self.number_of_ues        
        self.number_of_indoor_ues = 0
        self.number_of_aerial_ues = 0
            
        self.number_of_ues: int = self.number_of_outdoor_ues + self.number_of_indoor_ues + self.number_of_aerial_ues    
               
                   
    def initialize_user_positions(self) -> None:
        ### Initialize UE positions
        self.ue_position_m = np.full((self.number_of_ues, 3), np.nan, dtype=np.single)    
        self.ue_grid_position = np.full((self.number_of_ues,2), tools.int_nan(), dtype=int)   
        self.ue_geographical_area = np.full((self.number_of_ues), np.nan, dtype=np.single)         

    def set_user_types(self, ue_playground_deployment_obj):
        """
        Set the types of UEs (outdoor, indoor, aerial) based on the deployment configuration.
        0: outdoor, 1: indoor, 2: aerial
        """
        ### Setting UE types
        if ue_playground_deployment_obj.hexagonal_layout == True and ue_playground_deployment_obj.distribution != "grid":
            # Checks ensure required attributes are initialized
            assert self.number_of_ues_per_UEsite_cell is not None, "number_of_ues_per_UEsite_cell must be initialized before calling set_user_types"
            assert self.number_of_outdoor_ues_per_cell is not None, "number_of_outdoor_ues_per_cell must be initialized"
            assert self.number_of_indoor_ues_per_cell is not None, "number_of_indoor_ues_per_cell must be initialized"
            assert self.number_of_aerial_ues_per_UEsite_cell is not None, "number_of_aerial_ues_per_UEsite_cell must be initialized"
            # Create UE type array per cell
            self.ue_type_in_cell = [np.hstack((np.zeros((self.number_of_outdoor_ues_per_cell[i])).astype(int),\
                                               np.ones((self.number_of_indoor_ues_per_cell[i])).astype(int),\
                                               np.ones((self.number_of_aerial_ues_per_UEsite_cell[i])).astype(int) * 2)).ravel()\
                                    for i in range(0,np.size(self.number_of_ues_per_UEsite_cell,0))]
            self.ue_type = np.concatenate(self.ue_type_in_cell).ravel()              
           
        else:
            self.ue_type = self.rng.permutation(np.concatenate((np.zeros((self.number_of_outdoor_ues),dtype=int), np.ones((self.number_of_indoor_ues),dtype=int))))       
       
        # Set alphanumeric label for UE type
        # Create an easier to read label for the UE type
        ue_type_list = self.ue_type.tolist()
        replacements = {0:'outdoor', 1:'indoor', 2:'aerial'}
        replacer = replacements.get  # For faster gets
        self.ue_type_list = [replacer(n, n) for n in ue_type_list]   

    def calculate_ue_heights(self, ue_playground_deployment_obj):
        
        # Checks
        if not isinstance(self.ue_position_m, np.ndarray): raise TypeError(
            f"in {self.__class__.__name__}.ue_position_m must be a numpy array within the {self.ue_playground_model} scenario")
        if not isinstance(self.number_of_aerial_ues, (int, np.integer)): raise TypeError(
            f"in {self.__class__.__name__}.number_of_aerial_ues must be an integer within the {self.ue_playground_model} scenario")
        if not isinstance(self.min_bulding_floors, (int, np.integer)): raise TypeError(
            f"in {self.__class__.__name__}.min_bulding_floors must be an integer within the {self.ue_playground_model} scenario")
        if not isinstance(self.max_bulding_floors, (int, np.integer)): raise TypeError(
            f"in {self.__class__.__name__}.max_bulding_floors must be an integer within the {self.ue_playground_model} scenario")
        if not isinstance(self.number_of_indoor_ues, (int, np.integer)): raise TypeError(
            f"in {self.__class__.__name__}.number_of_indoor_ues must be an integer within the {self.ue_playground_model} scenario")

        ### Compute UE heights   
        # First, we populate the UE height with the outdoor UE height
        n_fl = 1
        h_ut = 3*(n_fl-1)+1.5
        ue_heights = np.ones(self.number_of_ues, dtype=np.single) * h_ut
            
        # Second, we identify the aerial UEs and populate their height, which is function of a stochastic process
        if self.number_of_aerial_ues > 0:
            if self.ue_aerial_height_m == None :
                ue_heights[self.ue_type == 2] = self.rng.uniform(low=1.5, high=300, size=(self.number_of_aerial_ues,))
            else:
                ue_heights[self.ue_type == 2] = self.ue_aerial_height_m       
            
        # Third, we identify the indoor UEs and populate their height, which is function of a stochastic process
        if any(substring in ue_playground_deployment_obj.playground_model for substring in ["3GPPTR38_901", "3GPPTR36_777", "3GPPTR38_811"]):
            N_fl_indoor = self.rng.uniform(low=self.min_bulding_floors, high=self.max_bulding_floors, size=(self.number_of_indoor_ues,))
            n_fl_indoor = self.rng.uniform(low=n_fl, high=N_fl_indoor, size=(self.number_of_indoor_ues,))
            ue_heights[self.ue_type == 1] = 3*(n_fl_indoor-1)+1.5 
            
        self.ue_position_m[:,2] = ue_heights      
         
    def calculate_ue_velocity_vectors(self):
        # Generate a random 3D vector with z component equal to zero
        # Generate random numbers between -1 and 1
        #random_velocities = 2 * self.rng.rand(self.number_of_ues, 2) - 1
        random_velocities = np.zeros((self.number_of_ues, 2))
        random_velocities[:,0] = 1
     
        # Combine with a zero column for a third dimension
        velocities = np.column_stack((random_velocities, np.zeros(self.number_of_ues)))        
         
        # Normalize each row
        self.velocity_vector_kmh = np.apply_along_axis(lambda row: row / np.linalg.norm(row), axis=1, arr=velocities)
         
        # First, we populate the UE velocity of outdoor UEs
        if self.number_of_outdoor_ues > 0:
            self.velocity_vector_kmh[self.ue_type == 0] = self.ue_outdoor_velocity_kmh * self.velocity_vector_kmh[self.ue_type == 0]
        
        # Second, we populate the UE velocity of indoor UEs
        if self.number_of_indoor_ues > 0:
            self.velocity_vector_kmh[self.ue_type == 1] = self.ue_indoor_velocity_kmh * self.velocity_vector_kmh[self.ue_type == 1]
             
        # Third, we populate the UE velocity of aerial UEs
        if self.number_of_aerial_ues > 0:             
            self.velocity_vector_kmh[self.ue_type == 2] = self.ue_aerial_velocity_kmh * self.velocity_vector_kmh[self.ue_type == 2]
         
    def calculate_ue_orientations(self):
        ### Compute UE antenna orientations (omega_alpha)
        self.antenna_config_orientation_omega_mec_alpha_degrees = self.rng.uniform(low=0, high=360, size=(self.number_of_ues,))   
    
    
    def set_ue_targetRate(self, model_targetRate: str = "random_uniform"):
        """
        Generate the target rate for each user based on the specified target rate model.
        Updates the 'ue_target_rate_Mbps' column in the UE dataframe (self.df_ep).
    
        Parameters:
            model_targetRate (str): Currently supported:
                "random_uniform" - Generates target rates uniformly between a fixed minimum (0.5 Mbps) and maximum (2 Mbps).
        """        
        # Checks
        if not isinstance(self.df_ep, pd.DataFrame): raise TypeError(f"in {self.__class__.__name__}.df_ep must be a DataFrame")

        # Check if any UE has a traffic generation model equal to 'rate_requirement'
        if any(self.df_ep["traffic_generation_model"] == 'rate_requirement'):
            # Create a boolean mask for UEs with a 'rate_requirement' traffic generation model
            mask_rate_target = np.array(self.df_ep["traffic_generation_model"].values == 'rate_requirement', dtype=bool)
            
            if model_targetRate == "random_uniform":
                # Define the minimum and maximum target rate requirements in Mbps
                min_rate_target_Mbps = 0.5
                max_rate_target_Mbps = 2.0
    
                assert mask_rate_target.sum() <= self.number_of_ues
    
                # Generate random target rates uniformly distributed between the defined bounds
                ue_rate_requirements_Mbps = self.rng.uniform(
                    low=min_rate_target_Mbps,
                    high=max_rate_target_Mbps,
                    size=(mask_rate_target.sum(),)
                )
    
                # Apply the computed target rates only to UEs that have a 'rate_requirement'
                self.df_ep.loc[mask_rate_target, 'ue_target_rate_Mbps'] = ue_rate_requirements_Mbps
                
                # Clean up the mask variable
                del mask_rate_target
                return True
    
        else: return True
         

    def build_ue_template(self, ue_playground_deployment_obj):
               
       ### Initialize UE positions
       self.initialize_user_positions()  

       ### Setting UE types 
       self.set_user_types(ue_playground_deployment_obj)

       ### Calculate UE heights 
       self.calculate_ue_heights(ue_playground_deployment_obj)    
       
       ### Calculate velocity vectors
       self.calculate_ue_velocity_vectors()
         
       ### Compute UE antenna orientations (omega_alpha)
       self.calculate_ue_orientations()


    def _safe_repeat(self, value, repeats) -> np.ndarray:
        if value is None:
            return np.full(repeats, None, dtype=object)
        else:
            return np.repeat(value, repeats=repeats, axis=0)

                   
    def construct_data_frame(self):    
        
        if not isinstance(self.ue_position_m, np.ndarray): raise TypeError(
            f"in {self.__class__.__name__}.ue_position_m must be a numpy array within the {self.ue_playground_model} scenario")

        if not isinstance(self.number_of_ues, (int, np.integer)): raise TypeError(
            f"in {self.__class__.__name__}.number_of_ues must be an integer")

        ### Construct UE engineering parameters
        ue_parameters_d = {
            'ID': np.arange(0, self.number_of_ues, dtype=int), 
            'name':  [ 'UE_%d'%(x) for x in range(0,self.number_of_ues)], 
            'geographical_area': self.ue_geographical_area,
            'category': self._safe_repeat( self.ue_category, repeats=self.number_of_ues), 
                
            'position_x_m': self.ue_position_m[:,0],
            'position_y_m': self.ue_position_m[:,1],
            'position_z_m': self.ue_position_m[:,2],
            'indoor': self.ue_type == 1,
            'type': self.ue_type_list,
            
            
            'TX_number': np.repeat( getattr(self, 'TX_number', 1), repeats=self.number_of_ues, axis=0), 
            'RX_number': np.repeat( getattr(self, 'RX_number', 1), repeats=self.number_of_ues, axis=0), 
            
            
            'velocity_kmh': self._safe_repeat(self.velocity_kmh, repeats=self.number_of_ues), 
            'velocity_x_kmh': self.velocity_vector_kmh[:,0],
            'velocity_y_kmh': self.velocity_vector_kmh[:,1],
            'velocity_z_kmh': self.velocity_vector_kmh[:,2],            
            'noise_figure_dB': self._safe_repeat(self.noise_figure_dB, repeats=self.number_of_ues),
            
            'UE_tx_power_max_dBm': np.repeat( getattr(self, 'UE_tx_power_max_dBm', 0), repeats=self.number_of_ues, axis=0), 
            
            'dl_carrier_frequency_GHz': self._safe_repeat(self.dl_carrier_frequency_GHz, repeats=self.number_of_ues), 
            'dl_carrier_wavelength_m': self._safe_repeat(self.dl_carrier_wavelength_m, repeats=self.number_of_ues), 
            'ul_carrier_frequency_GHz': self._safe_repeat(self.dl_carrier_frequency_GHz, repeats=self.number_of_ues), 
            'ul_carrier_wavelength_m': self._safe_repeat(self.dl_carrier_wavelength_m, repeats=self.number_of_ues),             
            
            'antenna_pattern_model': self._safe_repeat(self.antenna_pattern_model, repeats=self.number_of_ues), 
            'antenna_config_max_gain_dBi': self._safe_repeat(self.antenna_config_max_gain_dBi, repeats=self.number_of_ues),       
               
            'antenna_config_hor_phi_3dB_deg': self._safe_repeat(0, repeats=self.number_of_ues),  
            'antenna_config_hor_A_m_dB': self._safe_repeat(0, repeats=self.number_of_ues), 
            'antenna_config_ver_theta_3dB_deg': self._safe_repeat(0, repeats=self.number_of_ues),
            'antenna_config_ver_SLA_dB': self._safe_repeat(0, repeats=self.number_of_ues),

            #'antenna_config_hor_alpha_mec_bearing_deg': np.repeat(self.antenna_config_orientation_omega_mec_alpha_degrees, repeats=self.number_of_ues, axis=0),
            'antenna_config_hor_alpha_mec_bearing_deg': self.antenna_config_orientation_omega_mec_alpha_degrees,
            'antenna_config_hor_alpha_elec_bearing_deg': self._safe_repeat(self.antenna_config_orientation_omega_elec_alpha_degrees, repeats=self.number_of_ues), 
            'antenna_config_ver_beta_mec_downtilt_deg': self._safe_repeat(self.antenna_config_orientation_omega_mec_beta_degrees, repeats=self.number_of_ues),  
            'antenna_config_ver_beta_elec_downtilt_deg': self._safe_repeat(self.antenna_config_orientation_omega_elec_beta_degrees, repeats=self.number_of_ues),   
            'antenna_config_gamma_mec_slant_deg': self._safe_repeat(self.antenna_config_orientation_omega_mec_gamma_degrees, repeats=self.number_of_ues),  
            'antenna_config_gamma_elec_slant_deg': self._safe_repeat(self.antenna_config_orientation_omega_mec_gamma_degrees, repeats=self.number_of_ues),             
            
            'antenna_config_Mg': self._safe_repeat(self.antenna_config_Mg, repeats=self.number_of_ues), 
            'antenna_config_Ng': self._safe_repeat(self.antenna_config_Ng, repeats=self.number_of_ues), 
            'antenna_config_M': self._safe_repeat(self.antenna_config_M, repeats=self.number_of_ues), 
            'antenna_config_N': self._safe_repeat(self.antenna_config_N, repeats=self.number_of_ues), 
            'antenna_config_P': self._safe_repeat(self.antenna_config_P, repeats=self.number_of_ues), 
            'antenna_config_P_type': self._safe_repeat(self.antenna_config_P_type, repeats=self.number_of_ues),             
            'antenna_config_number_of_elements': self._safe_repeat(self.antenna_config_number_of_elements, repeats=self.number_of_ues),
            'antenna_config_dgh_m': self._safe_repeat(self.antenna_config_dgh_m, repeats=self.number_of_ues), 
            'antenna_config_dgv_m': self._safe_repeat(self.antenna_config_dgv_m, repeats=self.number_of_ues),                
            'antenna_config_dh_m': self._safe_repeat(self.antenna_config_dh_m, repeats=self.number_of_ues), 
            'antenna_config_dv_m': self._safe_repeat(self.antenna_config_dv_m, repeats=self.number_of_ues), 
            
            'ue_to_cell_association': self._safe_repeat(self.ue_to_cell_association, repeats=self.number_of_ues),
            
            'traffic_generation_model': self._safe_repeat(getattr(self, "traffic_generation_model", "strict_prb_fair_per_cell_beam"), 
                                                          repeats=self.number_of_ues),
            'ue_target_rate_Mbps': self._safe_repeat(getattr(self, "ue_target_rate_Mbps", np.nan), repeats=self.number_of_ues),             
                                            
            'mobility_model': np.repeat(self.simulation_config_obj.ue_mobility, repeats=self.number_of_ues, axis=0)
            }
        
        # df_ep = TrackedDataFrame(data=ue_parameters_d, 
        #                          key_to_track='ue_target_rate_Mbps') # key_to_track is the name of the coloumn to keep track
        df_ep = pd.DataFrame(data=ue_parameters_d)
          
        return df_ep     
    
        

    def construct_ue_deployment_ITU_R_M2135_UMa(self, ue_playground_deployment_obj: Site, ue_hotspot_deployment_obj: Hotspot) -> pd.DataFrame:
        
        ################################
        ##### Parameters
        ################################
        # UE category
        self.ue_category = "pedestrian"
        
        self.indoor_ue_ratio = 0
        self.min_bulding_floors = 0
        self.max_bulding_floors = 0
        
        self.noise_figure_dB = 7
        
        self.ue_outdoor_velocity_kmh = 30
        
        self.dl_carrier_frequency_GHz = 2
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.ul_carrier_frequency_GHz = 2
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        
        self.antenna_pattern_model = "omnidirectional"
        
        self.antenna_config_max_gain_dBi = 0        
        
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 1
        self.antenna_config_N = 1
        self.antenna_config_P = "single" # (“single” or “dual”)
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)      
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_orientation_omega_mec_alpha_degrees = 0 # Since the UE has only one antenna and this is omnidirectional, we use default values
        self.antenna_config_orientation_omega_elec_alpha_degrees = 0 # Since the UE has only one antenna and this is omnidirectional, we use default values        
        self.antenna_config_orientation_omega_mec_beta_degrees = 90
        self.antenna_config_orientation_omega_elec_beta_degrees = 90
        self.antenna_config_orientation_omega_mec_gamma_degrees = 0
        self.antenna_config_orientation_omega_elec_gamma_degrees = 0
        
        self.antenna_config_dgh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dgv_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m       
        
        self.ue_to_cell_association = "antenna_pattern-based" # "path_loss" "antenna_pattern-based" "SSB-RSRP"
        
        # Set number of UEs    
        if ue_playground_deployment_obj.distribution == "grid":
            # Set total number of UEs for the grid case
            if isinstance(self.ue_info_obj, dict): raise TypeError(f"{self.__class__.__name__}.ue_info_obj must be an instance of UeInfo class.")
            self.number_of_ues: int = self.ue_info_obj.number_of_ues
            # Calculate sum number of UEs for the grid case
            self.calculate_sum_number_of_users_grid_layouts()
            
        else: 
            # Set number of UEs per cell for the hexagonal case using sim_config 
            self.number_of_ues_per_UEsite_cell = self.simulation_config_obj.number_of_ues_per_UEsite_cell
            self.number_of_aerial_ues_per_UEsite_cell = self.simulation_config_obj.number_of_aerial_ues_per_UEsite_cell
            # Calculate sum number of UEs for the hexagonal case
            self.calculate_sum_number_of_users_hexagonal_layouts()

            
        ################################
        ##### Build UEs 
        ################################
            
        self.build_ue_template(ue_playground_deployment_obj)  
        
        ### Calculate UE locations
        if ue_playground_deployment_obj.distribution == "uniform" or ue_playground_deployment_obj.distribution == "uniform_with_hotspots" or \
            ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell" or ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell_with_hotspots":
            # Calculate UE locations at random per site using hexagonal layout
            self.calculate_ue_locations_at_random_per_site_using_hexagon(ue_playground_deployment_obj, ue_hotspot_deployment_obj) 

        elif ue_playground_deployment_obj.distribution == "grid":
            self.calculate_ue_locations_in_grid_within_rectangular_playground(ue_playground_deployment_obj)           
        
        ### Construct UE enginering parameters   
        return  self.construct_data_frame()     
    
    
    def construct_ue_deployment_ITU_R_M2135_UMi(self, ue_playground_deployment_obj: Site, ue_hotspot_deployment_obj: Hotspot) -> pd.DataFrame:
        
        # UE category
        self.ue_category = "pedestrian"

        self.indoor_ue_ratio = 0.5
        self.min_bulding_floors = 0
        self.max_bulding_floors = 0
        
        self.noise_figure_dB = 7
        
        self.ue_outdoor_velocity_kmh = 3
        self.ue_indoor_velocity_kmh = 3
        
        self.dl_carrier_frequency_GHz = 3.5
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.ul_carrier_frequency_GHz = 3.5
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)        
        
        self.antenna_pattern_model = "omnidirectional"
        
        self.antenna_config_max_gain_dBi = 0        
        
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 1
        self.antenna_config_N = 1
        self.antenna_config_P = "single" # (“single” or “dual”)
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)      
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
               
        self.antenna_config_orientation_omega_mec_alpha_degrees = 0 # Since the UE has only one antenna and this is omnidirectional, we use default values
        self.antenna_config_orientation_omega_elec_alpha_degrees = 0 # Since the UE has only one antenna and this is omnidirectional, we use default values        
        self.antenna_config_orientation_omega_mec_beta_degrees = 90
        self.antenna_config_orientation_omega_elec_beta_degrees = 90
        self.antenna_config_orientation_omega_mec_gamma_degrees = 0
        self.antenna_config_orientation_omega_elec_gamma_degrees = 0   
        
        self.antenna_config_dgh_m = 0 
        self.antenna_config_dgv_m = 0      
        self.antenna_config_dh_m = 0
        self.antenna_config_dv_m = 0            
        
        self.ue_to_cell_association = "antenna_pattern-based" # "path_loss" "antenna_pattern-based" "SSB-RSRP"


        # Set number of UEs    
        if ue_playground_deployment_obj.distribution == "grid":
            # Set total number of UEs for the grid case
            if isinstance(self.ue_info_obj, dict): raise TypeError(f"{self.__class__.__name__}.ue_info_obj must be an instance of UeInfo class.")
            self.number_of_ues:int = self.ue_info_obj.number_of_ues
            # Calculate sum number of UEs for the grid case
            self.calculate_sum_number_of_users_grid_layouts()
            
        else: 
            # Set number of UEs per cell for the hexagonal case using sim_config 
            self.number_of_ues_per_UEsite_cell = self.simulation_config_obj.number_of_ues_per_UEsite_cell
            self.number_of_aerial_ues_per_UEsite_cell = self.simulation_config_obj.number_of_aerial_ues_per_UEsite_cell
            # Calculate sum number of UEs for the hexagonal case
            self.calculate_sum_number_of_users_hexagonal_layouts()
            
                  
        ################################
        ##### Build UEs 
        ################################
            
        self.build_ue_template(ue_playground_deployment_obj)  
        
        ### Calculate UE locations
        
        if ue_playground_deployment_obj.distribution == "uniform" or ue_playground_deployment_obj.distribution == "uniform_with_hotspots" or ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell" or \
            ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell_with_hotspots":
            # Calculate UE locations at random per site using hexagonal layout
            self.calculate_ue_locations_at_random_per_site_using_hexagon(ue_playground_deployment_obj, ue_hotspot_deployment_obj) 
            
        elif ue_playground_deployment_obj.distribution == "grid":
            # Calculate UE locations considering the grid scenario
            self.calculate_ue_locations_in_grid_within_rectangular_playground(ue_playground_deployment_obj)           
        
        ### Construct UE enginering parameters   
        return  self.construct_data_frame()              
    
    
    def construct_ue_deployment_3GPPTR36_814_Case_1(self, ue_playground_deployment_obj: Site, ue_hotspot_deployment_obj: Hotspot) -> pd.DataFrame:
        
        # UE category
        self.ue_category = "pedestrian"
        
        self.indoor_ue_ratio = 0
        self.min_bulding_floors = 0
        self.max_bulding_floors = 0
        
        self.noise_figure_dB = 9
        
        self.ue_outdoor_velocity_kmh = 3
        self.ue_indoor_velocity_kmh = 3
        
        self.dl_carrier_frequency_GHz = 2
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.ul_carrier_frequency_GHz = 2
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)        
        
        self.antenna_pattern_model = "omnidirectional"
        
        self.antenna_config_max_gain_dBi = 0        
        
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 1
        self.antenna_config_N = 1
        self.antenna_config_P = "single" # (“single” or “dual”)
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)      
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_orientation_omega_mec_alpha_degrees = 0 # Since the UE has only one antenna and this is omnidirectional, we use default values
        self.antenna_config_orientation_omega_elec_alpha_degrees = 0 # Since the UE has only one antenna and this is omnidirectional, we use default values        
        self.antenna_config_orientation_omega_mec_beta_degrees = 90
        self.antenna_config_orientation_omega_elec_beta_degrees = 90
        self.antenna_config_orientation_omega_mec_gamma_degrees = 0
        self.antenna_config_orientation_omega_elec_gamma_degrees = 0
        
        self.antenna_config_dgh_m = 0 
        self.antenna_config_dgv_m = 0      
        self.antenna_config_dh_m = 0
        self.antenna_config_dv_m = 0            
        
        self.ue_to_cell_association = "antenna_pattern-based" # "path_loss" "antenna_pattern-based" "SSB-RSRP"
        
        # Set number of UEs    
        if ue_playground_deployment_obj.distribution == "grid":
            # Set total number of UEs for the grid case
            if isinstance(self.ue_info_obj, dict): raise TypeError(f"{self.__class__.__name__}.ue_info_obj must be an instance of UeInfo class.")
            self.number_of_ues: int = self.ue_info_obj.number_of_ues
            # Calculate sum number of UEs for the grid case
            self.calculate_sum_number_of_users_grid_layouts()
            
        else: 
            # Set number of UEs per cell for the hexagonal case using sim_config 
            self.number_of_ues_per_UEsite_cell = self.simulation_config_obj.number_of_ues_per_UEsite_cell
            self.number_of_aerial_ues_per_UEsite_cell = self.simulation_config_obj.number_of_aerial_ues_per_UEsite_cell
            # Calculate sum number of UEs for the hexagonal case
            self.calculate_sum_number_of_users_hexagonal_layouts()         
            
        ################################
        ##### Build UEs 
        ################################
            
        self.build_ue_template(ue_playground_deployment_obj)  
        
        ### Calculate UE locations
        if ue_playground_deployment_obj.distribution == "uniform" or ue_playground_deployment_obj.distribution == "uniform_with_hotspots" or \
            ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell" or ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell_with_hotspots":
            # Calculate UE locations at random per site using hexagonal layout
            self.calculate_ue_locations_at_random_per_site_using_hexagon(ue_playground_deployment_obj, ue_hotspot_deployment_obj) 
            
        elif ue_playground_deployment_obj.distribution == "grid":
            # Calculate UE locations considering the grid scenario
            self.calculate_ue_locations_in_grid_within_rectangular_playground(ue_playground_deployment_obj)           
        
        ### Construct UE enginering parameters   
        return  self.construct_data_frame()               
    
    
    def construct_ue_deployment_3GPPTR38_901_UMa(self, ue_playground_deployment_obj: Site, ue_hotspot_deployment_obj: Hotspot) -> pd.DataFrame:
        
        # UE category
        self.ue_category = "pedestrian"

        self.indoor_ue_ratio = 0.8
        self.min_bulding_floors = 4
        self.max_bulding_floors = 8
        
        self.noise_figure_dB = 9
        
        self.ue_outdoor_velocity_kmh = 3
        self.ue_indoor_velocity_kmh = 3
        
        self.dl_carrier_frequency_GHz = 6
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.ul_carrier_frequency_GHz = 6
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)        
        
        self.antenna_pattern_model = "3GPPTR38_901"
        
        self.antenna_config_max_gain_dBi = 0        
        
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 1
        self.antenna_config_N = 1
        self.antenna_config_P = "dual" # (“single” or “dual”)
        self.antenna_config_P_type = "cross" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)      
        
        # For the moment, dual polarization is only considered when using TR38.901. 
        # We generate the channel for the 2 polarizations using Sionna and sum the two channels per antenna element to get just one channel 
        # See line h_freq = torch.sum(h_freq, dim=1) in channels.py
        if self.antenna_config_P == 'single' or self.antenna_config_P == 'dual': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_orientation_omega_mec_alpha_degrees = 0 # Since the UE has only one antenna and this is omnidirectional, we use default values
        self.antenna_config_orientation_omega_elec_alpha_degrees = 0 # Since the UE has only one antenna and this is omnidirectional, we use default values        
        self.antenna_config_orientation_omega_mec_beta_degrees = 90
        self.antenna_config_orientation_omega_elec_beta_degrees = 90
        self.antenna_config_orientation_omega_mec_gamma_degrees = 0
        self.antenna_config_orientation_omega_elec_gamma_degrees = 0
        
        self.antenna_config_dgh_m = 0 
        self.antenna_config_dgv_m = 0      
        self.antenna_config_dh_m = 0
        self.antenna_config_dv_m = 0        
        
        self.ue_to_cell_association = "SSB-RSRP" # "path_loss" "antenna_pattern-based" "SSB-RSRP"
        
        # Set number of UEs    
        if ue_playground_deployment_obj.distribution == "grid":
            # Set total number of UEs for the grid case
            if isinstance(self.ue_info_obj, dict): raise TypeError(f"{self.__class__.__name__}.ue_info_obj must be an instance of UeInfo class.")
            self.number_of_ues: int = self.ue_info_obj.number_of_ues
            # Calculate sum number of UEs for the grid case
            self.calculate_sum_number_of_users_grid_layouts()
            
        else: 
            # Set number of UEs per cell for the hexagonal case using sim_config 
            self.number_of_ues_per_UEsite_cell = self.simulation_config_obj.number_of_ues_per_UEsite_cell
            self.number_of_aerial_ues_per_UEsite_cell = self.simulation_config_obj.number_of_aerial_ues_per_UEsite_cell
            # Calculate sum number of UEs for the hexagonal case
            self.calculate_sum_number_of_users_hexagonal_layouts()         
            
        ################################
        ##### Build UEs 
        ################################
            
        self.build_ue_template(ue_playground_deployment_obj)  
        
        ### Calculate UE locations
        
        if ue_playground_deployment_obj.distribution == "uniform" or ue_playground_deployment_obj.distribution == "uniform_with_hotspots" or \
            ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell" or ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell_with_hotspots":
            # Calculate UE locations at random per site using hexagonal layout
            self.calculate_ue_locations_at_random_per_site_using_hexagon(ue_playground_deployment_obj, ue_hotspot_deployment_obj) 
            
        elif ue_playground_deployment_obj.distribution == "grid":
            # Calculate UE locations considering the grid scenario
            self.calculate_ue_locations_in_grid_within_rectangular_playground(ue_playground_deployment_obj)
        
        ### Construct UE enginering parameters   
        return  self.construct_data_frame()             
    
    def construct_ue_deployment_3GPPTR38_901_UMa_large_scale_calibration(self, ue_playground_deployment_obj: Site, ue_hotspot_deployment_obj: Hotspot) -> pd.DataFrame:
        
        # UE category
        self.ue_category = "pedestrian"
        
        self.indoor_ue_ratio = 0.8
        self.min_bulding_floors = 4
        self.max_bulding_floors = 8
        
        self.noise_figure_dB = 9
        
        self.ue_outdoor_velocity_kmh = 3
        self.ue_indoor_velocity_kmh = 3
        
        self.dl_carrier_frequency_GHz = 6
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.ul_carrier_frequency_GHz = 6
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)        
        
        self.antenna_pattern_model = "omnidirectional" # "path_loss" "antenna_pattern-based" "SSB-RSRP"
        
        self.antenna_config_max_gain_dBi = 0        
        
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 1
        self.antenna_config_N = 1
        self.antenna_config_P = "single" # (“single” or “dual”)
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)      
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_orientation_omega_mec_alpha_degrees = [] # Since the UE has only one antenna and this is omnidirectional, we use default values
        self.antenna_config_orientation_omega_elec_alpha_degrees = 0 # Since the UE has only one antenna and this is omnidirectional, we use default values        
        self.antenna_config_orientation_omega_mec_beta_degrees = 90
        self.antenna_config_orientation_omega_elec_beta_degrees = 90
        self.antenna_config_orientation_omega_mec_gamma_degrees = 0
        self.antenna_config_orientation_omega_elec_gamma_degrees = 0
        
        self.antenna_config_dgh_m = 0 
        self.antenna_config_dgv_m = 0      
        self.antenna_config_dh_m = 0
        self.antenna_config_dv_m = 0        
        
        self.ue_to_cell_association = "RSRP-based"
        
        # Set number of UEs    
        if ue_playground_deployment_obj.distribution == "grid":
            # Set total number of UEs for the grid case
            if isinstance(self.ue_info_obj, dict): raise TypeError(f"{self.__class__.__name__}.ue_info_obj must be an instance of UeInfo class.")
            self.number_of_ues: int = self.ue_info_obj.number_of_ues
            # Calculate sum number of UEs for the grid case
            self.calculate_sum_number_of_users_grid_layouts()
            
        else: 
            # Set number of UEs per cell for the hexagonal case using sim_config 
            self.number_of_ues_per_UEsite_cell = self.simulation_config_obj.number_of_ues_per_UEsite_cell
            self.number_of_aerial_ues_per_UEsite_cell = self.simulation_config_obj.number_of_aerial_ues_per_UEsite_cell
            # Calculate sum number of UEs for the hexagonal case
            self.calculate_sum_number_of_users_hexagonal_layouts()
            
        ################################
        ##### Build UEs 
        ################################
            
        self.build_ue_template(ue_playground_deployment_obj)  
        
        ### Calculate UE locations
        
        if ue_playground_deployment_obj.distribution == "uniform" or ue_playground_deployment_obj.distribution == "uniform_with_hotspots" or \
            ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell" or ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell_with_hotspots":
            # Calculate UE locations at random per site using hexagonal layout
            self.calculate_ue_locations_at_random_per_site_using_hexagon(ue_playground_deployment_obj, ue_hotspot_deployment_obj)
            
        elif ue_playground_deployment_obj.distribution == "grid":
            # Calculate UE locations considering the grid scenario
            self.calculate_ue_locations_in_grid_within_rectangular_playground(ue_playground_deployment_obj)           
        
        ### Construct and return UE enginering parameters   
        return  self.construct_data_frame()   


    def construct_ue_deployment_3GPPTR38_901_UMi(self, ue_playground_deployment_obj: Site, ue_hotspot_deployment_obj: Hotspot) -> pd.DataFrame:
        
        self.indoor_ue_ratio = 0.8
        self.min_bulding_floors = 4
        self.max_bulding_floors = 8
        
        self.noise_figure_dB = 9
        
        self.ue_outdoor_velocity_kmh = 3
        self.ue_indoor_velocity_kmh = 3
        
        self.dl_carrier_frequency_GHz = 6
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.ul_carrier_frequency_GHz = 6
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)        
        
        self.antenna_pattern_model = "3GPPTR38_901"
        
        self.antenna_config_max_gain_dBi = 0        
        
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 1
        self.antenna_config_N = 1
        self.antenna_config_P = "single" # (“single” or “dual”)
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)      
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_orientation_omega_mec_alpha_degrees = 0 # Since the UE has only one antenna and this is omnidirectional, we use default values
        self.antenna_config_orientation_omega_elec_alpha_degrees = 0 # Since the UE has only one antenna and this is omnidirectional, we use default values        
        self.antenna_config_orientation_omega_mec_beta_degrees = 90
        self.antenna_config_orientation_omega_elec_beta_degrees = 90
        self.antenna_config_orientation_omega_mec_gamma_degrees = 0
        self.antenna_config_orientation_omega_elec_gamma_degrees = 0
        
        self.antenna_config_dgh_m = 0 
        self.antenna_config_dgv_m = 0      
        self.antenna_config_dh_m = 0
        self.antenna_config_dv_m = 0        
        
        self.ue_to_cell_association = "SSB-RSRP" # "path_loss" "antenna_pattern-based" "SSB-RSRP"
        
        # Set number of UEs    
        if ue_playground_deployment_obj.distribution == "grid":
            # Set total number of UEs for the grid case
            if isinstance(self.ue_info_obj, dict): raise TypeError(f"{self.__class__.__name__}.ue_info_obj must be an instance of UeInfo class.")
            self.number_of_ues: int = self.ue_info_obj.number_of_ues
            # Calculate sum number of UEs for the grid case
            self.calculate_sum_number_of_users_grid_layouts()
            
        else: 
            # Set number of UEs per cell for the hexagonal case using sim_config 
            self.number_of_ues_per_UEsite_cell = self.simulation_config_obj.number_of_ues_per_UEsite_cell
            self.number_of_aerial_ues_per_UEsite_cell = self.simulation_config_obj.number_of_aerial_ues_per_UEsite_cell
            # Calculate sum number of UEs for the hexagonal case
            self.calculate_sum_number_of_users_hexagonal_layouts()        
            
        ################################
        ##### Build UEs 
        ################################
            
        self.build_ue_template(ue_playground_deployment_obj)  
        
        ### Calculate UE locations
        
        if ue_playground_deployment_obj.distribution == "uniform" or ue_playground_deployment_obj.distribution == "uniform_with_hotspots" or \
            ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell" or ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell_with_hotspots":
            # Calculate UE locations at random per site using hexagonal layout
            self.calculate_ue_locations_at_random_per_site_using_hexagon(ue_playground_deployment_obj, ue_hotspot_deployment_obj) 
            
        elif ue_playground_deployment_obj.distribution == "grid":
            # Calculate UE locations considering the grid scenario
            self.calculate_ue_locations_in_grid_within_rectangular_playground(ue_playground_deployment_obj)           
        
        ### Construct UE enginering parameters   
        return  self.construct_data_frame()         
    
    def construct_ue_deployment_3GPPTR38_901_UMi_large_scale_calibration(self, ue_playground_deployment_obj: Site, ue_hotspot_deployment_obj: Hotspot) -> pd.DataFrame:
        
        # UE category
        self.ue_category = "pedestrian"
        
        
        self.indoor_ue_ratio = 0.8
        self.min_bulding_floors = 4
        self.max_bulding_floors = 8
        
        self.noise_figure_dB = 9
        
        self.ue_outdoor_velocity_kmh = 3
        self.ue_indoor_velocity_kmh = 3
        
        self.dl_carrier_frequency_GHz = 6
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.ul_carrier_frequency_GHz = 6
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)        
        
        self.antenna_pattern_model = "omnidirectionaL" # "path_loss" "antenna_pattern-based" "SSB-RSRP"
        
        self.antenna_config_max_gain_dBi = 0        
        
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 1
        self.antenna_config_N = 1
        self.antenna_config_P = "single" # (“single” or “dual”)
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)      
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_orientation_omega_mec_alpha_degrees = [] # Since the UE has only one antenna and this is omnidirectional, we use default values
        self.antenna_config_orientation_omega_elec_alpha_degrees = 0 # Since the UE has only one antenna and this is omnidirectional, we use default values        
        self.antenna_config_orientation_omega_mec_beta_degrees = 90
        self.antenna_config_orientation_omega_elec_beta_degrees = 90
        self.antenna_config_orientation_omega_mec_gamma_degrees = 0
        self.antenna_config_orientation_omega_elec_gamma_degrees = 0
        
        self.antenna_config_dgh_m = 0 
        self.antenna_config_dgv_m = 0      
        self.antenna_config_dh_m = 0
        self.antenna_config_dv_m = 0        
        
        self.ue_to_cell_association = "RSRP-based"
        
        # Set number of UEs    
        if ue_playground_deployment_obj.distribution == "grid":
            # Set total number of UEs for the grid case
            if isinstance(self.ue_info_obj, dict): raise TypeError(f"{self.__class__.__name__}.ue_info_obj must be an instance of UeInfo class.")
            self.number_of_ues: int = self.ue_info_obj.number_of_ues
            # Calculate sum number of UEs for the grid case
            self.calculate_sum_number_of_users_grid_layouts()
            
        else: 
            # Set number of UEs per cell for the hexagonal case using sim_config 
            self.number_of_ues_per_UEsite_cell = self.simulation_config_obj.number_of_ues_per_UEsite_cell
            self.number_of_aerial_ues_per_UEsite_cell = self.simulation_config_obj.number_of_aerial_ues_per_UEsite_cell
            # Calculate sum number of UEs for the hexagonal case
            self.calculate_sum_number_of_users_hexagonal_layouts()        
            
        ################################
        ##### Build UEs 
        ################################
            
        self.build_ue_template(ue_playground_deployment_obj)  
        
        ### Calculate UE locations
        
        if ue_playground_deployment_obj.distribution == "uniform" or ue_playground_deployment_obj.distribution == "uniform_with_hotspots" or \
            ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell" or ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell_with_hotspots":
            # Calculate UE locations at random per site using hexagonal layout
            self.calculate_ue_locations_at_random_per_site_using_hexagon(ue_playground_deployment_obj, ue_hotspot_deployment_obj) 
            
        elif ue_playground_deployment_obj.distribution == "grid":
            # Calculate UE locations in grid within rectangular playground
            self.calculate_ue_locations_in_grid_within_rectangular_playground(ue_playground_deployment_obj)           
        
        ### Construct UE enginering parameters   
        return  self.construct_data_frame()        
    
    
    def construct_ue_deployment_3GPPTR36_777_UMa_AV(self, ue_playground_deployment_obj: Site, ue_hotspot_deployment_obj: Hotspot, ue_aerial_height_m: None|int = None) -> pd.DataFrame:
        
        self.indoor_ue_ratio = 0.8
        self.min_bulding_floors = 4
        self.max_bulding_floors = 8
        
        self.noise_figure_dB = 9
        
        self.ue_aerial_velocity_kmh = 160
        self.ue_outdoor_velocity_kmh = 30
        self.ue_indoor_velocity_kmh = 3
        
        self.ue_aerial_height_m = ue_aerial_height_m
        
        self.dl_carrier_frequency_GHz = 2
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.ul_carrier_frequency_GHz = 2
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)        
        
        self.antenna_pattern_model = "omnidirectionaL"
        
        self.antenna_config_max_gain_dBi = 0        
        
        #TR 38.901
        #self.antenna_config_Mg = 1
        #self.antenna_config_Ng = 1
        #self.antenna_config_M = 1
        #self.antenna_config_N = 1
        #self.antenna_config_P = 2
        #TR 36.814
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 1
        self.antenna_config_N = 1
        self.antenna_config_P = "single" # (“single” or “dual”)
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)      
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num    
        
        self.antenna_config_orientation_omega_mec_alpha_degrees = 0 # Since the UE has only one antenna and this is omnidirectional, we use default values
        self.antenna_config_orientation_omega_elec_alpha_degrees = 0 # Since the UE has only one antenna and this is omnidirectional, we use default values        
        self.antenna_config_orientation_omega_mec_beta_degrees = 90
        self.antenna_config_orientation_omega_elec_beta_degrees = 90
        self.antenna_config_orientation_omega_mec_gamma_degrees = 0
        self.antenna_config_orientation_omega_elec_gamma_degrees = 0
        
        self.antenna_config_dgh_m = 0 
        self.antenna_config_dgv_m = 0      
        self.antenna_config_dh_m = 0
        self.antenna_config_dv_m = 0            
        
        self.ue_to_cell_association = "RSRP-based"
        
        # Set number of UEs    
        if ue_playground_deployment_obj.distribution == "grid":
            # Set total number of UEs for the grid case
            if isinstance(self.ue_info_obj, dict): raise TypeError(f"{self.__class__.__name__}.ue_info_obj must be an instance of UeInfo class.")
            self.number_of_ues: int = self.ue_info_obj.number_of_ues
            # Calculate sum number of UEs for the grid case
            self.calculate_sum_number_of_users_grid_layouts()
            
        else: 
            # Set number of UEs per cell for the hexagonal case using sim_config 
            self.number_of_ues_per_UEsite_cell = self.simulation_config_obj.number_of_ues_per_UEsite_cell
            self.number_of_aerial_ues_per_UEsite_cell = self.simulation_config_obj.number_of_aerial_ues_per_UEsite_cell
            # Calculate sum number of UEs for the hexagonal case
            self.calculate_sum_number_of_users_hexagonal_layouts()         
            
        ################################
        ##### Build UEs 
        ################################
            
        self.build_ue_template(ue_playground_deployment_obj)  
        
        ### Calculate UE locations
        
        if ue_playground_deployment_obj.distribution == "uniform" or ue_playground_deployment_obj.distribution == "uniform_with_hotspots" or ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell" or ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell_with_hotspots":
            # Calculate UE locations at random per site using hexagonal layout
            self.calculate_ue_locations_at_random_per_site_using_hexagon(ue_playground_deployment_obj, ue_hotspot_deployment_obj) 
            
        elif ue_playground_deployment_obj.distribution == "grid":
            # Calculate UE locations considering the grid scenario
            self.calculate_ue_locations_in_grid_within_rectangular_playground(ue_playground_deployment_obj)           
        
        ### Construct UE enginering parameters   
            
        return  self.construct_data_frame()    

    def construct_ue_deployment_3GPPTR36_777_UMi_AV(self, ue_playground_deployment_obj: Site, ue_hotspot_deployment_obj: Hotspot, ue_aerial_height_m: None|int = None) -> pd.DataFrame:
        
        
        self.indoor_ue_ratio = 0.8
        self.min_bulding_floors = 4
        self.max_bulding_floors = 8
        
        self.noise_figure_dB = 9
        
        self.ue_aerial_velocity_kmh = 160
        self.ue_outdoor_velocity_kmh = 30
        self.ue_indoor_velocity_kmh = 3
        
        self.ue_aerial_height_m = ue_aerial_height_m
        
        self.dl_carrier_frequency_GHz = 2
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.ul_carrier_frequency_GHz = 2
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)        
        
        self.antenna_pattern_model = "omnidirectionaL"
        
        self.antenna_config_max_gain_dBi = 0        
        
        #TR 38.901
        #self.antenna_config_Mg = 1
        #self.antenna_config_Ng = 1
        #self.antenna_config_M = 1
        #self.antenna_config_N = 1
        #self.antenna_config_P = 2
        #TR 36.814
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 1
        self.antenna_config_N = 1
        self.antenna_config_P = "single" # (“single” or “dual”)
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)      
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_orientation_omega_mec_alpha_degrees = 0 # Since the UE has only one antenna and this is omnidirectional, we use default values
        self.antenna_config_orientation_omega_elec_alpha_degrees = 0 # Since the UE has only one antenna and this is omnidirectional, we use default values        
        self.antenna_config_orientation_omega_mec_beta_degrees = 90
        self.antenna_config_orientation_omega_elec_beta_degrees = 90
        self.antenna_config_orientation_omega_mec_gamma_degrees = 0
        self.antenna_config_orientation_omega_elec_gamma_degrees = 0
        
        self.antenna_config_dgh_m = 0 
        self.antenna_config_dgv_m = 0      
        self.antenna_config_dh_m = 0
        self.antenna_config_dv_m = 0            
        
        self.ue_to_cell_association = "RSRP-based"
        
        # Set number of UEs    
        if ue_playground_deployment_obj.distribution == "grid":
            # Set total number of UEs for the grid case
            if isinstance(self.ue_info_obj, dict): raise TypeError(f"{self.__class__.__name__}.ue_info_obj must be an instance of UeInfo class.")
            self.number_of_ues: int = self.ue_info_obj.number_of_ues
            # Calculate sum number of UEs for the grid case
            self.calculate_sum_number_of_users_grid_layouts()
            
        else: 
            # Set number of UEs per cell for the hexagonal case using sim_config 
            self.number_of_ues_per_UEsite_cell = self.simulation_config_obj.number_of_ues_per_UEsite_cell
            self.number_of_aerial_ues_per_UEsite_cell = self.simulation_config_obj.number_of_aerial_ues_per_UEsite_cell
            # Calculate sum number of UEs for the hexagonal case
            self.calculate_sum_number_of_users_hexagonal_layouts()
            
        ################################
        ##### Build UEs 
        ################################
            
        self.build_ue_template(ue_playground_deployment_obj)  
        
        ### Calculate UE locations
        
        if ue_playground_deployment_obj.distribution == "uniform" or ue_playground_deployment_obj.distribution == "uniform_with_hotspots" or \
            ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell" or ue_playground_deployment_obj.distribution == "inhomogeneous_per_cell_with_hotspots":
            # Calculate UE locations at random per site using hexagonal layout
            self.calculate_ue_locations_at_random_per_site_using_hexagon(ue_playground_deployment_obj, ue_hotspot_deployment_obj) 
            
        elif ue_playground_deployment_obj.distribution == "grid":
            # Calculate UE locations considering the grid scenario
            self.calculate_ue_locations_in_grid_within_rectangular_playground(ue_playground_deployment_obj)           
        
        ### Construct UE enginering parameters   
        df_ep = self.construct_data_frame()   
        
        # Set Category
        df_ep['category'] = 'pedestrian'
        df_ep.loc[df_ep['type'] == 'aerial', 'category'] = 'uav'
        
        return df_ep
        

    def construct_ue_deployment_polygon(self, ue_playground_deployment_obj: Site, ue_hotspot_deployment_obj: Hotspot) -> pd.DataFrame:

        # UE category
        self.ue_category = "pedestrian"
        
        
        self.indoor_ue_ratio = 0
        self.min_bulding_floors = 4
        self.max_bulding_floors = 8
        
        self.noise_figure_dB = 9
        
        self.ue_outdoor_velocity_kmh = 30
        self.ue_indoor_velocity_kmh = 30        
        
        self.dl_carrier_frequency_GHz = 6
        self.dl_carrier_wavelength_m = 3e8 / (self.dl_carrier_frequency_GHz * 1e9)
        self.ul_carrier_frequency_GHz = 6
        self.dl_carrier_wavelength_m = 3e8 / (self.dl_carrier_frequency_GHz * 1e9)

        self.antenna_pattern_model = "omnidirectional"

        self.antenna_config_max_gain_dBi = 0        
        
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 1
        self.antenna_config_N = 1
        self.antenna_config_P = "single" # (“single” or “dual”)
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)      
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_orientation_omega_mec_alpha_degrees = [] # Since the UE has only one antenna and this is omnidirectional, we use default values
        self.antenna_config_orientation_omega_elec_alpha_degrees = 0 # Since the UE has only one antenna and this is omnidirectional, we use default values        
        self.antenna_config_orientation_omega_mec_beta_degrees = 90
        self.antenna_config_orientation_omega_elec_beta_degrees = 90
        self.antenna_config_orientation_omega_mec_gamma_degrees = 0
        self.antenna_config_orientation_omega_elec_gamma_degrees = 0
        
        self.antenna_config_dgh_m = 0
        self.antenna_config_dgv_m = 0
        self.antenna_config_dh_m = 0
        self.antenna_config_dv_m = 0  
        
        self.ue_to_cell_association = "RSRP-based" # "path_loss" "antenna_pattern-based" "SSB-RSRP"
        
        # Set number of UEs    
        if ue_playground_deployment_obj.distribution == "grid":
            # Set total number of UEs for the grid case
            if isinstance(self.ue_info_obj, dict): raise TypeError(f"{self.__class__.__name__}.ue_info_obj must be an instance of UeInfo class.")
            self.number_of_ues: int = self.ue_info_obj.number_of_ues
        else: 
            # Set number of UEs per cell for the hexagonal case using sim_config 
            self.number_of_ues = self.simulation_config_obj.number_of_ues_per_UEsite_cell.sum()
            self.number_of_aerial_ues = self.simulation_config_obj.number_of_aerial_ues_per_UEsite_cell.sum()

            

        self.number_of_indoor_ues = int(np.round(self.number_of_ues * self.indoor_ue_ratio))
        self.number_of_outdoor_ues = self.number_of_ues - self.number_of_indoor_ues
        self.number_of_ues = self.number_of_outdoor_ues + self.number_of_indoor_ues + self.number_of_aerial_ues                
            
        ################################
        ##### Build UEs 
        ################################
            
        self.build_ue_template(ue_playground_deployment_obj)  
        
        ### Calculate UE locations
        
        if ue_playground_deployment_obj.playground_model == "rectangular" or \
            ue_playground_deployment_obj.playground_model == "3GPPTR36_814_Case_1_omni_dana" :
            # Calculate UE locations at random within rectangular playground
            self.calculate_ue_locations_at_random_within_rectangular_playground(ue_playground_deployment_obj, ue_hotspot_deployment_obj)       
            
        elif ue_playground_deployment_obj.playground_model == "circular":
            # Calculate UE locations at random within circular playground
            self.calculate_ue_locations_at_random_within_circular_playground(ue_playground_deployment_obj, ue_hotspot_deployment_obj)          
        
        ### Construct UE engineering parameters
        return  self.construct_data_frame()
    
    
    def construct_ue_deployment_polygon_dataset(self, ue_playground_deployment_obj: Site, ue_hotspot_deployment_obj: Hotspot) -> pd.DataFrame:

        # Get the list of files in the directory that start with 'user_deployment'
        user_deployment_files = [f for f in os.listdir(self.dataset_import_folder) if f.startswith('user_deployment')]

        # Check if there is more than one 'user_deployment' file
        if len(user_deployment_files) > 1:
            raise FileExistsError(
                "Multiple 'user_deployment' files found in the specified folder. Please ensure there is at most one file.")

        # If there is exactly one 'user_deployment' file, proceed to load it
        elif len(user_deployment_files) == 1:
            user_deployment_dataset = user_deployment_files[0]
            user_deployment_dataset_full_file_path = os.path.join(self.dataset_import_folder, user_deployment_dataset)
            print(f"Loading 'user_deployment' dataset from file: {user_deployment_dataset_full_file_path}")
            user_parameters_df = pd.read_csv(user_deployment_dataset_full_file_path, low_memory=False, sep=',')

            self.dl_carrier_frequency_GHz = user_parameters_df['dl_carrier_frequency_GHz'].values.astype(float)[0]
            self.dl_carrier_wavelength_m = 3e8 / (self.dl_carrier_frequency_GHz * 1e9)
            self.ul_carrier_frequency_GHz = user_parameters_df['ul_carrier_frequency_GHz'].values.astype(float)[0]
            self.dl_carrier_wavelength_m = 3e8 / (self.dl_carrier_frequency_GHz * 1e9)
            # Note that the following must be set to Ray_tracing, else the wrong antenna panel function will be used
            # in antenna_arrays (sionna.channel.tr38901.PanelArray instead of sionna.rt.PlanarArray)
            # self.antenna_pattern_model = "omnidirectional"
            self.antenna_pattern_model = "Ray_tracing"

        # If there is no 'user_deployment' file, raise an error
        else:
            raise FileNotFoundError(
                "No 'user_deployment' file found in the specified folder. Please provide the required file.")

        # Other default values
        
        # UE category
        self.ue_category = "pedestrian"
        
        
        self.indoor_ue_ratio = 0
        self.min_bulding_floors = 0
        self.max_bulding_floors = 0
        
        self.noise_figure_dB = 7
        
        self.ue_outdoor_velocity_kmh = 30
        self.ue_indoor_velocity_kmh = 3
        
        ''' 
        These are now moved to the if/else condition above
        self.dl_carrier_frequency_GHz = 2
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.ul_carrier_frequency_GHz = 2
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.antenna_pattern_model = "omnidirectional"
        '''

        self.antenna_config_max_gain_dBi = 0        
        
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 1
        self.antenna_config_N = 1
        self.antenna_config_P = "single" # (“single” or “dual”)
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)      
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_orientation_omega_mec_alpha_degrees = 0 # Since the UE has only one antenna and this is omnidirectional, we use default values
        self.antenna_config_orientation_omega_elec_alpha_degrees = 0 # Since the UE has only one antenna and this is omnidirectional, we use default values        
        self.antenna_config_orientation_omega_mec_beta_degrees = 90
        self.antenna_config_orientation_omega_elec_beta_degrees = 90
        self.antenna_config_orientation_omega_mec_gamma_degrees = 0
        self.antenna_config_orientation_omega_elec_gamma_degrees = 0
        
        self.antenna_config_dgh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dgv_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m       
        
        self.ue_to_cell_association = "antenna_pattern-based" # "path_loss" "antenna_pattern-based" "SSB-RSRP"
        
        # Set number of UEs    
        if ue_playground_deployment_obj.distribution == "grid":
            # Set total number of UEs for the grid case
            if isinstance(self.ue_info_obj, dict): raise TypeError(f"{self.__class__.__name__}.ue_info_obj must be an instance of UeInfo class.")
            self.number_of_ues: int = self.ue_info_obj.number_of_ues
        else: 
            # Set number of UEs per cell for the hexagonal case using sim_config 
            self.number_of_ues = self.simulation_config_obj.number_of_ues_per_UEsite_cell.sum()
            self.number_of_aerial_ues = self.simulation_config_obj.number_of_aerial_ues_per_UEsite_cell.sum()
            
        self.number_of_indoor_ues = int(np.round(self.number_of_ues * self.indoor_ue_ratio))
        self.number_of_outdoor_ues = self.number_of_ues - self.number_of_indoor_ues
        self.number_of_ues = self.number_of_outdoor_ues + self.number_of_indoor_ues + self.number_of_aerial_ues                
            
        ################################
        ##### Build UEs 
        ################################
            
        self.build_ue_template(ue_playground_deployment_obj)  
        
        ### Calculate UE locations

        if ue_playground_deployment_obj.playground_model == "dataset_rectangular":

            self.calculate_ue_locations_at_random_within_rectangular_playground(ue_playground_deployment_obj,
                                                                                ue_hotspot_deployment_obj)

        elif ue_playground_deployment_obj.playground_model == "dataset_circular":

            self.calculate_ue_locations_at_random_within_circular_playground(ue_playground_deployment_obj,
                                                                             ue_hotspot_deployment_obj)

        ### Construct UE engineering parameters
        return  self.construct_data_frame()  


    def construct_ue_deployment_3GPPTR38_901_multilayer_large_scale_calibration(self, ue_playground_deployment_obj: Site, ue_hotspot_deployment_obj: Hotspot) -> pd.DataFrame:

        # Set the hotspot deployment to none using the uniform configuration in the default ue_playground_deployment_obj
        no_ue_hotspot_deployment_obj: Hotspot =\
        hotspots.Hotspot(
            self.simulation_config_obj,
            self.ue_playground_deployment_obj)  
        no_ue_hotspot_deployment_obj.process()


        ### Manage grid deployment
        ##########################################
        if self.simulation_config_obj.ue_distribution == "grid":
            # Store ue_info_obj to be able to reset it after deployments
            # ue_deployDensity_info_dict = copy.deepcopy(self.ue_info_obj)

            # Take dictionary ue info deployment
            assert isinstance(self.ue_info_obj, dict), "ue_info_obj must be a dictionary for the multilayer deployment."
            self.ue_info_obj =\
                     self.ue_info_obj["construct_ue_deployment_3GPPTR38_901_UMa_large_scale_calibration"]

            # Deploy UEs uniformly
            df_ep_UMa = self.construct_ue_deployment_3GPPTR38_901_UMa_large_scale_calibration(
                ue_playground_deployment_obj, 
                no_ue_hotspot_deployment_obj)
            
            # Reset ue_deployDensity_info_dict
            # self.ue_info_obj = copy.deepcopy(ue_deployDensity_info_dict)

            return df_ep_UMa


        ### Manage non-grid deployment
        ##########################################
        else:
            # Get number of UEs per cell for the uniform deployment
            number_of_ues_per_UEsite_cell =\
                    self.simulation_config_obj.number_of_ues_per_UEsite_cell_mfl_dict["construct_ue_deployment_3GPPTR38_901_UMa_large_scale_calibration"]
            number_of_aerial_ues_per_UEsite_cell =\
                    self.simulation_config_obj.number_of_aerial_ues_per_UEsite_cell_mfl_dict["construct_ue_deployment_3GPPTR38_901_UMa_large_scale_calibration"]
            
            # Temporarily set in sim config the number of UEs per cell for the uniform deployment
            self.simulation_config_obj._set_attr_sim_config(
                    number_of_ues_per_UEsite_cell=number_of_ues_per_UEsite_cell,
                    number_of_aerial_ues_per_UEsite_cell=number_of_aerial_ues_per_UEsite_cell)
            
            # Delete for safety
            del number_of_ues_per_UEsite_cell, number_of_aerial_ues_per_UEsite_cell

            # Deploy UEs uniformly
            df_ep_UMa = self.construct_ue_deployment_3GPPTR38_901_UMa_large_scale_calibration(
                ue_playground_deployment_obj, 
                no_ue_hotspot_deployment_obj)
            
            # Reset number of UEs per cell in sim config
            self.simulation_config_obj._set_attr_sim_config(
                    number_of_ues_per_UEsite_cell=None,
                    number_of_aerial_ues_per_UEsite_cell=None)


        # If there are hotspost, continue with the code for hotspot deployment
        # otherwise, return only the uniform deployment
        if ue_hotspot_deployment_obj.number_of_hotspots > 0: pass 
        else: return df_ep_UMa


        # Creates the reference circular playground for UE distribution
        circular_ue_playground_deployment_obj =\
            sites.Site(
                False,
                "circular",
                "uniform_with_hotspots")  # [Warning] This has to be fixed, if thios field changes, ue_deployDensity_info_dict does not change automatically, has it has been already set.
        circular_ue_playground_deployment_obj.process()

        # Get number of UEs per cell for the hotspot polygon deployment
        number_of_ues_per_UEsite_cell = self.simulation_config_obj.number_of_ues_per_UEsite_cell_mfl_dict["construct_ue_deployment_polygon"]
        number_of_aerial_ues_per_UEsite_cell = self.simulation_config_obj.number_of_aerial_ues_per_UEsite_cell_mfl_dict["construct_ue_deployment_polygon"]

        # Temporarily set in sim config the number of UEs per cell for the uniform deployment
        self.simulation_config_obj._set_attr_sim_config(
                number_of_ues_per_UEsite_cell=number_of_ues_per_UEsite_cell,
                number_of_aerial_ues_per_UEsite_cell=number_of_aerial_ues_per_UEsite_cell)
        
        # Delete for safety
        del number_of_ues_per_UEsite_cell, number_of_aerial_ues_per_UEsite_cell
                
        # Create hotspot deployment
        df_ep_circle = self.construct_ue_deployment_polygon(
            circular_ue_playground_deployment_obj, 
            ue_hotspot_deployment_obj)

        # Reset number of UEs per cell in sim config
        self.simulation_config_obj._set_attr_sim_config(
                number_of_ues_per_UEsite_cell=None,
                number_of_aerial_ues_per_UEsite_cell=None)
        
        # Concatenate the DataFrames of different deployment
        combined_df_ep = pd.concat([df_ep_UMa, df_ep_circle])  
        # Replace the values of the necessary columns to create a coherent scenario
        combined_df_ep['ID'] = range(len(combined_df_ep)) 
        combined_df_ep['name'] = [ 'UE_%d'%(x) for x in range(len(combined_df_ep))]  

        return combined_df_ep 
    