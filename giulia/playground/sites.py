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

import os
import time
from typing import List

import geopandas
import numpy as np
import pandas as pd
from sionna.rt import load_scene, Camera

from giulia.fs import data_driven_extras
from giulia.tools import tools
from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable

# Force CPU usage, prevent error with Mitsuba trying to use OptiX in load_scene()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = ''



class Site(Saveable):

    def __init__(self, 
                 wraparound,
                 playground_model, 
                 distribution = None):
        
        super().__init__()
        
        ##### Plots
        ########################
        self.plot = 0 # Switch to control plots if any


        ##### Input storage
        ########################
        self.wraparound = wraparound
        self.playground_model = playground_model
        self.distribution = distribution


        ##### Outputs
        ########################
        self.hexagonal_layout = None

        self.hex_layout_centre_m = None
        self.hex_layout_tiers = None
        self.isd_m = None
        self.grid_resol_m = None

        self.scenario_x_side_length_m = None
        self.scenario_y_side_length_m = None
        self.scenario_lower_left_conner_m = None
        self.hexagone_side_length_m = None
        self.hexagone_lower_left_conner_m = None

        self.ref_cell_site_positions_m = []
        self.ref_antenna_config_hor_alpha_mec_bearing_deg = None
        self.ref_number_of_sectors_per_site = None
        self.ref_number_of_cell_sites = None
        self.ref_number_of_cells = None
        self.hexagons = []
        self.hexagons_gp = []
        
        self.min_BS_to_UE_2D_distance_m = None
        self.min_BS_to_AV_3D_distance_m = None

        # For Sionna RT
        self.dataset_import_folder = data_driven_extras('loaded')
        self.dataset_export_folder = data_driven_extras('saved')
        self.scene = None
        self.scene_name = None
        self.scene_min_lat = None
        self.scene_max_lat = None
        self.scene_min_long = None
        self.scene_max_long = None
        

    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["playground_model"]
    

    def process(self, rescheduling_us=-1):

        ##### Start timer
        ########################  
        t_start = time.perf_counter()


        ##### Switch
        ########################
        self.hexagonal_layout = False

        if self.playground_model == "ITU_R_M2135_UMa" or\
            self.playground_model == "ITU_R_M2135_UMa_multilayer" or\
            self.playground_model == "ITU_R_M2135_UMa_Umi_colocated_multilayer" or \
            self.playground_model == "ITU_R_M2135_UMa_Umi_noncolocated_multilayer":

            self.hexagonal_layout = True
            self.hex_layout_centre_m = (0, 0, 0)
            self.hex_layout_tiers = 2
            self.isd_m = 500
            self.min_BS_to_UE_2D_distance_m = 25

            if self.wraparound == None :
                self.wraparound = True
            
            self.ref_antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
            self.ref_number_of_sectors_per_site = 3
            self.grid_resol_m = (int)(10)


        elif self.playground_model == "3GPPTR36_814_Case_1" or\
            self.playground_model == "3GPPTR36_814_Case_1_omni" or \
            self.playground_model == "3GPPTR38_901_UMa_C1" or\
            self.playground_model == "3GPPTR38_901_UMa_C2" or\
            self.playground_model == "3GPPTR38_901_UMa_lsc" or\
            self.playground_model == "3GPPTR38_901_UMa_lsc_sn" or \
            self.playground_model == "3GPPTR38_901_UMa_2GHz_lsc" or \
            self.playground_model == "3GPPTR38_901_UMa_C_band_lsc" or \
            self.playground_model == "3GPPTR36_777_UMa_AV" or \
            self.playground_model == "3GPPTR38_811_Dense_Urban_HAPS_ULA" or \
            self.playground_model == "3GPPTR38_811_Dense_Urban_HAPS_UPA" or \
            self.playground_model == "3GPPTR38_811_Dense_Urban_HAPS_Reflector" or \
            self.playground_model == "3GPPTR38_811_Dense_Urban_NTN" or\
            self.playground_model == "3GPPTR38_811_Urban_NTN" or \
            self.playground_model == "3GPPTR38_901_4G" or \
            self.playground_model == "3GPPTR38_901_5G" or \
            self.playground_model == "3GPPTR38_901_6G" or \
            self.playground_model == "3GPPTR38_901_4G5G_multilayer" or \
            self.playground_model == "3GPPTR38_901_4G_5G_multilayer" or \
            self.playground_model == "3GPPTR38_901_4G_5G2_multilayer" or \
            self.playground_model == "3GPPTR38_901_4G_5G6G_multilayer" or \
            self.playground_model == "3GPPTR38_901_4G_5G_6G_multilayer" or \
            self.playground_model == "3GPPTR38_901_4G5G_cell_reselection":

            self.hexagonal_layout = True
            self.hex_layout_centre_m = (0, 0, 0)
            self.hex_layout_tiers = 2
            self.isd_m = 500
            self.min_BS_to_UE_2D_distance_m = 35
            if self.playground_model == "3GPPTR36_777_UMa_AV":
                self.min_BS_to_AV_3D_distance_m = 10                
            
            if self.wraparound == None :
                self.wraparound = True
            
            self.ref_antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
            self.ref_number_of_sectors_per_site = 3
            self.grid_resol_m = (int)(10)


        elif self.playground_model == "3GPPTR36_814_Case_1_single_bs" or\
                self.playground_model == "3GPPTR36_814_Case_1_single_bs_omni" or\
                self.playground_model == "3GPPTR38_901_UMa_lsc_single_bs" or\
                self.playground_model == "3GPPTR38_901_UMa_lsc_single_sector":

            self.hexagonal_layout = True
            self.hex_layout_centre_m = (0, 0, 0)
            self.hex_layout_tiers = 0
            self.isd_m = 500
            self.min_BS_to_UE_2D_distance_m = 35
            
            if self.wraparound == None :
                self.wraparound = True
            
            self.ref_antenna_config_hor_alpha_mec_bearing_deg = np.array([0])
            self.ref_number_of_sectors_per_site = 1
            self.grid_resol_m = (int)(2)


        elif self.playground_model == "ITU_R_M2135_UMi" or\
                self.playground_model == "3GPPTR38_901_UMi_C1" or\
                self.playground_model == "3GPPTR38_901_UMi_C2" or\
                self.playground_model == "3GPPTR38_901_UMi_lsc" or\
                self.playground_model == "3GPPTR38_901_UMi_C_band_lsc" or \
                self.playground_model == "3GPPTR38_901_UMi_fr3_lsc" or \
                self.playground_model == "3GPPTR36_777_UMi_AV":

            self.hexagonal_layout = True
            self.hex_layout_centre_m = (0, 0, 0)
            self.hex_layout_tiers = 2
            self.isd_m = 200
            self.min_BS_to_UE_2D_distance_m = 10
            if self.playground_model == "3GPPTR36_777_UMi_AV":
                self.min_BS_to_AV_3D_distance_m = 10
                
            if self.wraparound == None :
                self.wraparound = True
            
            self.ref_antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
            self.ref_number_of_sectors_per_site = 3
            self.grid_resol_m = (int)(5)


        elif self.playground_model == "3GPPTR38_901_UPi_fr3_lsc":
            self.hexagonal_layout = True
            self.hex_layout_centre_m = (0, 0, 0)
            self.hex_layout_tiers = 2
            self.isd_m = 125
            self.min_BS_to_UE_2D_distance_m = 5
                
            if self.wraparound == None :
                self.wraparound = True
            
            self.ref_antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
            self.ref_number_of_sectors_per_site = 3
            self.grid_resol_m = (int)(5)            


        elif self.playground_model == "circular":
            self.grid_resol_m = (int)(2)
            self.isd_m = np.nan
            self.min_BS_to_UE_2D_distance_m = 10
            self.scenario_centre_m = (0, 0)
            self.scenario_radius_m = 500
            self.scenario_x_side_length_m = 2* self.scenario_radius_m 
            self.scenario_y_side_length_m = 2* self.scenario_radius_m 
            self.scenario_lower_left_conner_m = np.array(
                (-self.scenario_x_side_length_m / 2, -self.scenario_y_side_length_m / 2))
            
            if self.wraparound == None :
                self.wraparound = False   


        elif self.playground_model == "rectangular":
            self.grid_resol_m = (int)(10)
            self.isd_m = np.nan
            self.min_BS_to_UE_2D_distance_m = 10
            self.scenario_x_side_length_m = 3000
            self.scenario_y_side_length_m = 3000
            self.scenario_lower_left_conner_m = np.array(
                (-self.scenario_x_side_length_m / 2, -self.scenario_y_side_length_m / 2))

            if self.wraparound == None:
                self.wraparound = False


        elif self.playground_model == "3GPPTR36_814_Case_1_omni_dana":
            self.grid_resol_m = (int)(10)
            self.isd_m = np.nan
            self.min_BS_to_UE_2D_distance_m = 10
            self.scenario_x_side_length_m = 400 
            self.scenario_y_side_length_m = 1200
            self.scenario_lower_left_conner_m = np.array(
                (-self.scenario_x_side_length_m / 2, -self.scenario_y_side_length_m / 2))

            if self.wraparound == None:
                self.wraparound = False                


        elif self.playground_model == "dataset_rectangular":
            # Get the list of files in the directory that start with 'user_deployment'
            user_deployment_files = [f for f in os.listdir(self.dataset_import_folder) if f.startswith('user_deployment')]

            # Check if there is more than one 'user_deployment' file
            if len(user_deployment_files) > 1:
                raise FileExistsError(
                    "Multiple 'user_deployment' files found in the specified folder. Please ensure there is at most one file.")

            # If there is exactly one 'user_deployment' file, proceed to load it
            # In any case, for the ue_playground_deployment_obj, leave scene = None
            elif len(user_deployment_files) == 1:
                user_deployment_dataset = user_deployment_files[0]
                user_deployment_dataset_full_file_path = os.path.join(self.dataset_import_folder,
                                                                      user_deployment_dataset)
                print(f"Loading 'user_deployment' dataset from file: {user_deployment_dataset_full_file_path}")
                user_parameters_df = pd.read_csv(user_deployment_dataset_full_file_path, low_memory=False, sep=',')

                self.grid_resol_m = user_parameters_df['grid_resol_m'].values[0]
                self.isd_m = np.nan
                self.scenario_x_side_length_m = user_parameters_df['scenario_x_side_length_m'].values[0]
                self.scenario_y_side_length_m = user_parameters_df['scenario_y_side_length_m'].values[0]
                self.scenario_lower_left_conner_m = np.array(
                    (user_parameters_df['scenario_lower_left_conner_x_m'].values[0],
                     user_parameters_df['scenario_lower_left_conner_y_m'].values[0]))

            # If there is no 'user_deployment' file, raise an error
            else:
                raise FileNotFoundError(
                    "No 'user_deployment' file found in the specified folder. Please provide the required file.")

            if self.wraparound == None:
                self.wraparound = False


        elif self.playground_model == "dataset":

            # Load ray tracing scene in Mitsuba format for the site_deployment_obj

            # Get the list of files in the directory that start with 'scene'
            scene_files = [f for f in os.listdir(self.dataset_import_folder) if f.startswith('scene')]

            # Check if no file or more than one file matches the pattern
            if len(scene_files) == 0:
                raise FileNotFoundError("No 'scene' file found in the specified folder.")
            elif len(scene_files) > 1:
                raise FileExistsError(
                    "Multiple 'scene' files found in the specified folder. Please ensure there is only one file.")

            # There is exactly one file, proceed to load it
            scene_file = scene_files[0]
            scene_full_file_path = os.path.join(self.dataset_import_folder, scene_file)
            print(f"Loading scene from file: {scene_full_file_path}")
            self.scene = load_scene(scene_full_file_path)
            self.scene_name = os.path.splitext(scene_file)[0]   # Remove the .xml file extension

            # Extract scene coordinates from the file name
            file_name_parts = scene_file.split('_')

            # The coordinates are located at positions after 'scene_'
            scene_min_lat = float(file_name_parts[1].replace('p', '.'))
            scene_max_lat = float(file_name_parts[2].replace('p', '.'))
            scene_min_long = float(file_name_parts[3].replace('p', '.'))
            scene_max_long = float(file_name_parts[4].replace('p', '.'))

            # Assign the values to the appropriate variables
            self.scene_min_lat = scene_min_lat
            self.scene_max_lat = scene_max_lat
            self.scene_min_long = scene_min_long
            self.scene_max_long = scene_max_long

            print(
                f"Scene coordinates identified: min_lat={self.scene_min_lat}, max_lat={self.scene_max_lat}, min_long={self.scene_min_long}, max_long={self.scene_max_long}")

            for i, obj in enumerate(self.scene.objects.values()):
                print(f"{obj.name} : {obj.radio_material.name}")
                if i >= 10:
                    break

            print("Scene center: ", self.scene.center.numpy())
            print("Scene size: ", self.scene.size.numpy())

            # Create bird's eye view camera
            bird_pos = [0, 0, 2000]
            bird_pos[-2] -= 0.01  # Slightly move the camera for correct orientation
            bird_cam = Camera("birds_view", position=bird_pos, look_at=[0, 0, 0])
            self.scene.add(bird_cam)

            # Create corner view camera
            corner_view = Camera("corner_view", position=[800, 800, 200], look_at=[0, 0, 50])
            self.scene.add(corner_view)

            if self.wraparound == None:
                self.wraparound = False
                
                
        else: 
            raise ValueError(f"Error, site.playground_model not correctly specified. Value '{self.playground_model}' not admitted")


        ##### Create hexagonal grid
        ########################       
       
        if self.hexagonal_layout == True:

            ##### Create hexagonal grid locations
            ########################  
            self.define_hexagonal_layout(self.plot, self.hex_layout_centre_m, self.hex_layout_tiers, self.isd_m)

            ##### Store relevant variables
            ########################
            self.ref_number_of_cell_sites = len(self.ref_cell_site_positions_m)
            self.ref_number_of_cells = self.ref_number_of_cell_sites * self.ref_number_of_sectors_per_site


        ##### End
        ########################        
        log_calculations_time('Site deployment', t_start)
        
        return rescheduling_us

       
    def hexagonal_geographical_areas(self, cell_site_positions_m, hexagone_side_length_m):
        
       # Calculate cell site geographical area using Shapely
       hexagons = [tools.generate_hexagon(cell_site_position_m, hexagone_side_length_m)  for cell_site_position_m in cell_site_positions_m]
       
       # Store using Geopandas
       hexagons_gp = geopandas.GeoSeries(hexagons)  

       return hexagons, hexagons_gp        


    def define_hexagonal_layout(self, plot, hex_layout_centre_m, hex_layout_tiers, isd_m):
       
       # Set scenario dimensions 
       self.scenario_x_side_length_m = ( 2 + 3 * hex_layout_tiers) / np.sqrt(3) * isd_m + 2 
       self.scenario_y_side_length_m = (1+2*hex_layout_tiers) * isd_m 
       
       # Calculate scenario lower corner position 
       self.scenario_lower_left_conner_m = np.array((-1*self.scenario_x_side_length_m/2, -1*self.scenario_y_side_length_m/2),dtype=np.single)    
           
       # Calculate length of hexagone size
       self.hexagone_side_length_m = 1/np.sqrt(3) * isd_m 
       
       # Calculate bounded hexagone lower corner position 
       self.hexagone_lower_left_conner_m = (-1*self.hexagone_side_length_m, -1*np.sqrt(3)*self.hexagone_side_length_m/2)     

       # Calculate cell site positions
       self.ref_cell_site_positions_m = tools.create_hexagonal_layout(plot, hex_layout_centre_m, hex_layout_tiers, isd_m)   
       
       # Define cell site geographical area using Shapely and Geopandas
       self.hexagons, self.hexagons_gp = self.hexagonal_geographical_areas(self.ref_cell_site_positions_m, self.hexagone_side_length_m)       


# Create subclasses to differentiate the two deployments
class Site_siteDeployment(Site):
    pass
class Site_uePlaygroundDeployment(Site):
    pass
