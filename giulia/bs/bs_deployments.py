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
import sys
import time
from typing import List

import geopandas
import numpy as np
import pandas as pd

from giulia.channel import uaearfcn
from giulia.event_driven.snapshot_control import Snapshot_control
from giulia.fs import results_file, data_driven_extras
from giulia.playground import sites
from giulia.rrc import calculate_tx_power, tools_carrier
from giulia.tools import tools
from giulia.tools.tools import log_calculations_time, TrackedDataFrame
from giulia.outputs.saveable import Saveable

# Force CPU usage, prevent error with Mitsuba trying to use OptiX in load_scene()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = ''


class Network(Saveable):
    
    def __init__(self, 
                 simulation_config_obj, 
                 site_deployment_obj, 
                 ue_hotspot_deployment_obj):
       
       super().__init__()
      
       ##### Plots 
       ########################
       self.plot = 0 # Switch to control plots if any
       
       
       ##### Input storage
       ########################
       self.simulation_config_obj = simulation_config_obj
       self.site_deployment_obj = site_deployment_obj  
       self.ue_hotspot_deployment_obj = ue_hotspot_deployment_obj
       
       
       ##### Outputs
       ########################
       self.df_ep = []
               
       self.bs_antenna_height_m  = None
       self.sector_bearing_deg = None
       
       self.number_of_cell_sites = None
       self.number_of_sectors_per_site = None
       self.number_of_cells  = None
       
       self.cell_site_positions_m = []       
       
       self.RAT: None|str = None
       self.carrier_frequency_GHz = None
       self.carrier_wavelength_m = None
       self.bandwidth_MHz = None
       self.BS_tx_power_dBm = None
       self.BS_CRS_tx_power_per_RE_dBm = None
       
       self.antenna_config_Mg = None
       self.antenna_config_Ng = None
       self.antenna_config_M = None
       self.antenna_config_N = None
       self.antenna_config_P = None
       self.antenna_config_P_type = None
       self.antenna_config_dh_m = None
       self.antenna_config_dv_m = None
       
       self.BS_antenna_element_config_ver_theta_3dB_deg = None
       self.BS_antenna_element_config_ver_SLA_dB = None
       self.BS_antenna_element_config_hor_phi_3dB_deg = None
       self.BS_antenna_element_config_hor_Amax_dB = None
       self.BS_antenna_element_config_max_gain_dBi = None
       
       self.SSB_BeamDirection_phi_deg = [np.nan]
       self.SSB_BeamDirection_theta_deg = [np.nan]

       # For Sionna RT
       self.dataset_import_folder = data_driven_extras('loaded')


    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["carrier_frequency_GHz"]
       

    def process(self, rescheduling_us=-1):     
       
       ##### Start timer
       ########################        
       t_start = time.perf_counter()          
       
    
       ### Swtich
       ########################
       if self.site_deployment_obj.playground_model == "ITU_R_M2135_UMa" : 
           self.df_ep = self.construct_scenario_ITU_R_M2135_UMa(self.site_deployment_obj) 
           
       elif self.site_deployment_obj.playground_model == "ITU_R_M2135_UMi" : 
           self.df_ep = self.construct_scenario_ITU_R_M2135_UMi(self.site_deployment_obj) 
           
       elif self.site_deployment_obj.playground_model == "3GPPTR36_814_Case_1" : 
           self.df_ep = self.construct_scenario_3GPPTR36_814_Case_1(self.site_deployment_obj) 
           
       elif self.site_deployment_obj.playground_model == "3GPPTR36_814_Case_1_omni" : 
           self.df_ep = self.construct_scenario_3GPPTR36_814_Case_1_omni(self.site_deployment_obj)  
           
       elif self.site_deployment_obj.playground_model == "3GPPTR36_814_Case_1_single_bs" : 
           self.df_ep = self.construct_scenario_3GPPTR36_814_Case_1_single_bs(self.site_deployment_obj) 

       elif self.site_deployment_obj.playground_model == "3GPPTR36_814_Case_1_single_bs_omni" : 
           self.df_ep = self.construct_scenario_3GPPTR36_814_Case_1_single_bs_omni(self.site_deployment_obj) 
           
       elif self.site_deployment_obj.playground_model == "3GPPTR36_814_Case_1_omni_dana" : 
           self.df_ep = self.construct_scenario_3GPPTR36_814_Case_1_omni_dana(self.site_deployment_obj)            
           
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_UMa_C1" : 
           self.df_ep = self.construct_scenario_3GPPTR38_901_UMa_C1(self.site_deployment_obj) 
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_UMa_C2" : 
           self.df_ep = self.construct_scenario_3GPPTR38_901_UMa_C2(self.site_deployment_obj) 
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_UMa_lsc" : 
           self.df_ep = self.construct_scenario_3GPPTR38_901_UMa_large_scale_calibration(self.site_deployment_obj) 
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_UMa_lsc_sn" : 
           self.df_ep = self.construct_scenario_3GPPTR38_901_UMa_large_scale_calibration_sn(self.site_deployment_obj)            
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_UMa_lsc_single_bs" : 
           self.df_ep = self.construct_scenario_3GPPTR38_901_UMa_large_scale_calibration_single_bs(self.site_deployment_obj) 
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_UMa_lsc_single_sector" : 
           self.df_ep = self.construct_scenario_3GPPTR38_901_UMa_large_scale_calibration_single_sector(self.site_deployment_obj) 
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_UMa_2GHz_lsc" : 
           self.df_ep = self.construct_scenario_3GPPTR38_901_UMa_2GHz_large_scale_calibration(self.site_deployment_obj)  

       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_UMa_C_band_lsc" : 
           self.df_ep = self.construct_scenario_3GPPTR38_901_UMa_C_band_large_scale_calibration(self.site_deployment_obj)            
           
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_UMi_C1" : 
           self.df_ep = self.construct_scenario_3GPPTR38_901_UMi_C1(self.site_deployment_obj)  
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_UMi_C2" : 
           self.df_ep = self.construct_scenario_3GPPTR38_901_UMi_C2(self.site_deployment_obj)            

       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_UMi_lsc" : 
           self.df_ep = self.construct_scenario_3GPPTR38_901_UMi_large_scale_calibration(self.site_deployment_obj)  
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_UMi_C_band_lsc" : 
           self.df_ep = self.construct_scenario_3GPPTR38_901_UMi_C_band_large_scale_calibration(self.site_deployment_obj) 

       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_UMi_fr3_lsc" : 
           self.df_ep = self.construct_scenario_3GPPTR38_901_UMi_fr3_large_scale_calibration(self.site_deployment_obj) 
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_UPi_fr3_lsc" : 
           self.df_ep = self.construct_scenario_3GPPTR38_901_UPi_fr3_large_scale_calibration(self.site_deployment_obj)            
            
       elif self.site_deployment_obj.playground_model == "3GPPTR38_811_Dense_Urban_HAPS_ULA" : 
           self.df_ep = self.construct_scenario_3GPPTR38_811_Dense_Urban_HAPS_ULA(self.site_deployment_obj)     
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_811_Dense_Urban_HAPS_UPA" : 
           self.df_ep = self.construct_scenario_3GPPTR38_811_Dense_Urban_HAPS_UPA(self.site_deployment_obj)
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_811_Dense_Urban_HAPS_Reflector" : 
           self.df_ep = self.construct_scenario_3GPPTR38_811_Dense_Urban_HAPS_Reflector(self.site_deployment_obj)
           
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_811_Urban_NTN" : 
           self.df_ep = self.construct_scenario_3GPPTR38_811_Urban_NTN(self.site_deployment_obj)    
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_811_Dense_Urban_NTN" : 
           self.df_ep = self.construct_scenario_3GPPTR38_811_Desne_Urban_NTN(self.site_deployment_obj)             
           
       elif self.site_deployment_obj.playground_model == "3GPPTR36_777_UMa_AV" : 
           self.df_ep = self.construct_scenario_3GPPTR36_777_UMa_AV(self.site_deployment_obj) 
           
       elif self.site_deployment_obj.playground_model == "3GPPTR36_777_UMi_AV" : 
           self.df_ep = self.construct_scenario_3GPPTR36_777_UMi_AV(self.site_deployment_obj)   
           
       
       elif self.site_deployment_obj.playground_model == "ITU_R_M2135_UMa_multilayer":
           self.df_ep = self.construct_scenario_ITU_R_M2135_UMa_colocated_multilayer(self.site_deployment_obj, [2, 3.5], [20, 20])
           
       elif self.site_deployment_obj.playground_model == "ITU_R_M2135_UMa_Umi_colocated_multilayer":
           self.df_ep = self.construct_scenario_ITU_R_M2135_UMa_Umi_colocated_multilayer(self.site_deployment_obj)            
           
       elif self.site_deployment_obj.playground_model == "ITU_R_M2135_UMa_Umi_noncolocated_multilayer":
           self.df_ep = self.construct_scenario_ITU_R_M2135_UMa_Umi_noncolocated_multilayer(self.site_deployment_obj)  
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_4G":
           self.df_ep = self.construct_scenario_3GPPTR38_901_4G(self.site_deployment_obj)   
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_5G":
           self.df_ep = self.construct_scenario_3GPPTR38_901_5G(self.site_deployment_obj)  
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_6G":
           self.df_ep = self.construct_scenario_3GPPTR38_901_6G(self.site_deployment_obj)             
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_4G5G_multilayer":
           self.df_ep = self.construct_scenario_3GPPTR38_901_4G5G(self.site_deployment_obj)
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_4G_5G_multilayer":
           self.df_ep = self.construct_scenario_3GPPTR38_901_4G_5G(self.site_deployment_obj)
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_4G_5G2_multilayer":
           self.df_ep = self.construct_scenario_3GPPTR38_901_4G_5G2(self.site_deployment_obj)           

       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_4G_5G6G_multilayer":
           self.df_ep = self.construct_scenario_3GPPTR38_901_4G_5G6G(self.site_deployment_obj)
                      
       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_4G_5G_6G_multilayer":
           self.df_ep = self.construct_scenario_3GPPTR38_901_4G_5G_6G(self.site_deployment_obj, self.ue_hotspot_deployment_obj) 
           
       elif self.site_deployment_obj.playground_model == "3GPPTR38_901_4G5G_cell_reselection":
           self.df_ep = self.construct_scenario_3GPPTR38_901_4G5G_cell_reselection(self.site_deployment_obj)           
             
            
       elif self.site_deployment_obj.playground_model == "dataset":
           self.df_ep = self.construct_scenario_dataset(self.site_deployment_obj)


       else: 
           raise ValueError(f"Error, network.site_deployment_obj.playground_model not correctly specified. Value '{self.site_deployment_obj.playground_model}' not admitted")
       
       ### Checks
       ########################
       self.basic_checks(self.df_ep)
       
      
       ##### Save to plot
       ########################   
       snapshot_control = Snapshot_control.get_instance()
       if self.simulation_config_obj.plot == 1 and snapshot_control.num_snapshots == 0: 
           
            file = results_file(self.simulation_config_obj.project_name, 'to_plot_scenario')
            
            # Create mask to get first occurrence of each site_name
            mask = ~self.df_ep["site_name"].duplicated()
            
            # Use mask to extract unique site rows
            unique_sites_df = self.df_ep.loc[mask]
            
            # Save data
            np.savez(file,isd_m=self.site_deployment_obj.isd_m,
                     site_names=self.df_ep["site_name"].drop_duplicates().astype(str).to_numpy(),
                     cell_site_positions_m=unique_sites_df[["position_x_m", "position_y_m", "position_z_m"]].astype(np.single).to_numpy()
            )  
       
        
       ##### Plots 
       ########################    
    #    if self.plot == 1:
    #       plotting.plot_scenario(project_name, site_deployment_obj.isd_m, self.cell_site_positions_m)        
       
       
       ##### End 
       ########################
       log_calculations_time('BS deployment', t_start)
       
       return rescheduling_us
           
       
    def basic_checks(self, df_ep):
        
        # SSB checks 
        if np.any(df_ep["SSB_number_of_beams"].to_numpy() > df_ep["antenna_config_number_of_elements"].to_numpy() ):
            sys.exit("Error Number of SSB beams cannot be larger than number of antenna elements")
            
        if np.any(df_ep["SSB_number_of_beams_V"].to_numpy() * df_ep["SSB_number_of_beams_H"].to_numpy()  > df_ep["antenna_config_number_of_elements"].to_numpy() ):
            sys.exit("Error Number of SSB beams cannot be larger than number of antenna elements 2")          
            
        if np.any(df_ep["SSB_number_of_beams_V"].to_numpy() > df_ep["antenna_config_M"].to_numpy() ):
           sys.exit("Error Number of SSB beams in V domain cannot be larger than number of antenna elements in the V domain")
        
        if np.any(df_ep["SSB_number_of_beams_H"].to_numpy() > df_ep["antenna_config_N"].to_numpy() ):
            sys.exit("Error Number of SSB beams in H domain cannot be larger than number of antenna elements in the H domain")
            
        # CSI-RS checks
        if np.any(df_ep["CSI_RS_number_of_beams"].to_numpy() > df_ep["antenna_config_number_of_elements"].to_numpy() ):
            sys.exit("Error Number of CSI-RS beams cannot be larger than number of antenna elements 2") 
            
        if np.any(df_ep["CSI_RS_number_of_beams_V"].to_numpy() * df_ep["CSI_RS_number_of_beams_H"].to_numpy() > df_ep["antenna_config_number_of_elements"].to_numpy() ):
            sys.exit("Error Number of CSI-RS beams cannot be larger than number of antenna elements 2") 
              
        if np.any(df_ep["CSI_RS_number_of_beams_V"].to_numpy() > df_ep["antenna_config_M"].to_numpy() ):
           sys.exit("Error Number of CSI-RS beams in V domain cannot be larger than number of antenna elements in the V domain")
        
        if np.any(df_ep["CSI_RS_number_of_beams_H"].to_numpy() > df_ep["antenna_config_N"].to_numpy() ):
           sys.exit("Error Number of CSI-RS beams in H domain cannot be larger than number of antenna elements in the H domain")
 
    
    def sectorized_geographical_areas(self, hexagons, cell_site_positions_m, isd_m, bearing):
       # Calculate sector geographical area using Shapely - intersection between hexagon and sector shapes
       sectors = []
       number_of_sectors_per_site = np.size(bearing,0)
       for cell_site_index in range(0,np.size(cell_site_positions_m,0)):
           for sector_index in range(0,number_of_sectors_per_site): 
               sectors.append(hexagons[cell_site_index].intersection(tools.create_sector(self.cell_site_positions_m[cell_site_index], 
                                                                                              isd_m/np.sqrt(3), 
                                                                                              np.radians(bearing[sector_index] - 360/number_of_sectors_per_site/2), 
                                                                                              np.radians(bearing[sector_index] + 360/number_of_sectors_per_site/2))))
       # Store using Geopandas
       sectors_gp = geopandas.GeoSeries(sectors)  
       
       return sectors, sectors_gp        
    
    
    def construct_data_frame(self):
       ### Construct enginering parameters 
                      
       engineering_parameters_d = {
            'ID': np.arange(0, self.number_of_cells, dtype=int), 
            'name':  [ 'cell_%d'%(x) for x in range(0,self.number_of_cells)], 
            'site_ID':  np.repeat(np.arange(0,self.number_of_cell_sites), repeats=(int)(self.number_of_cells/self.number_of_cell_sites), axis=0),
            'site_name':  np.repeat([ 'site_%d'%(x) for x in range(0,self.number_of_cell_sites)], repeats=(int)(self.number_of_cells/self.number_of_cell_sites), axis=0),     
            'RAT': np.repeat(self.RAT, repeats=self.number_of_cells, axis=0), 
            'radio_unit_ID': np.repeat(np.arange(0,self.number_of_cell_sites), repeats=(int)(self.number_of_cells/self.number_of_cell_sites), axis=0), 
            'radio_unit_type': np.repeat(self.radio_unit_type, repeats=self.number_of_cells, axis=0), 
            'radio_unit_status': np.repeat('active', repeats=self.number_of_cells, axis=0),  #'active', 'symbol_shutdown', 'carrier_shutdown', 'dormancy'
            'TX_number': np.repeat(self.TX_number, repeats=self.number_of_cells, axis=0), 
            'RX_number': np.repeat(self.RX_number, repeats=self.number_of_cells, axis=0), 
            'bbu_ID': np.arange(0, self.number_of_cells, dtype=int), 
            'bbu_type': np.repeat(self.bbu_type, repeats=self.number_of_cells, axis=0),  
            'PCI': np.arange(0, self.number_of_cells, dtype=int), 
            
            'velocity_kmh': np.repeat(self.velocity_kmh, repeats=self.number_of_cells, axis=0), 
            'velocity_x_kmh': self.velocity_vector_kmh[:,0],
            'velocity_y_kmh': self.velocity_vector_kmh[:,1],
            'velocity_z_kmh': self.velocity_vector_kmh[:,2],              
            'mobility_model': np.repeat(self.simulation_config_obj.bs_mobility, repeats=self.number_of_cells, axis=0),
            
            'position_x_m': self.cell_positions_m[:,0],
            'position_y_m': self.cell_positions_m[:,1],
            'position_z_m': self.cell_positions_m[:,2], 
            'indoor': np.repeat(self.indoor, repeats=self.number_of_cells, axis=0),
            
            'fdd_tdd_ind': np.repeat(self.fdd_tdd_ind, repeats=self.number_of_cells, axis=0), 
            'subframe_assignment': np.repeat(self.subframe_assignment, repeats=self.number_of_cells, axis=0), 
            'FR': np.repeat(self.FR, repeats=self.number_of_cells, axis=0), 
            'subcarriers_per_PRB': np.repeat(self.subcarrier_per_PRB, repeats=self.number_of_cells, axis=0), 
            'ofdm_symbols_in_slot': np.repeat(self.ofdm_symbols_in_slot, repeats=self.number_of_cells, axis=0),
            
            'BS_tx_power_dBm': np.repeat(self.BS_tx_power_dBm, repeats=self.number_of_cells, axis=0), 
            
            'dl_freq_band': np.repeat(self.dl_freq_band, repeats=self.number_of_cells, axis=0), 
            'dl_earfcn': np.repeat(self.dl_earfcn, repeats=self.number_of_cells, axis=0),
            'dl_carrier_frequency_GHz': np.repeat(self.dl_carrier_frequency_GHz, repeats=self.number_of_cells, axis=0), 
            'dl_carrier_wavelength_m': np.repeat(self.dl_carrier_wavelength_m, repeats=self.number_of_cells, axis=0), 
            'dl_bandwidth_MHz': np.repeat(self.dl_bandwidth_MHz, repeats=self.number_of_cells, axis=0), 
            'dl_PRBs_available': np.repeat(self.dl_PRBs_available, repeats=self.number_of_cells, axis=0),
            'dl_subcarrier_spacing_kHz': np.repeat(self.dl_subcarrier_spacing_kHz, repeats=self.number_of_cells, axis=0),   
            'dl_ofdm_symbol_duration_us': np.repeat(self.dl_ofdm_symbol_duration_us, repeats=self.number_of_cells, axis=0),  
            'dl_control_channel_overhead': np.repeat(self.dl_control_channel_overhead, repeats=self.number_of_cells, axis=0),  
            
            'ul_freq_band': np.repeat(self.ul_freq_band, repeats=self.number_of_cells, axis=0), 
            'ul_earfcn': np.repeat(self.ul_earfcn, repeats=self.number_of_cells, axis=0),            
            'ul_carrier_frequency_GHz': np.repeat(self.ul_carrier_frequency_GHz, repeats=self.number_of_cells, axis=0), 
            'ul_carrier_wavelength_m': np.repeat(self.ul_carrier_wavelength_m, repeats=self.number_of_cells, axis=0),  
            'ul_bandwidth_MHz': np.repeat(self.ul_bandwidth_MHz, repeats=self.number_of_cells, axis=0),
            'ul_available_PRBs': np.repeat(self.ul_available_PRBs, repeats=self.number_of_cells, axis=0),
            'ul_subcarrier_spacing_kHz': np.repeat(self.ul_subcarrier_spacing_kHz, repeats=self.number_of_cells, axis=0),   
            'ul_ofdm_symbol_duration_us': np.repeat(self.ul_ofdm_symbol_duration_us, repeats=self.number_of_cells, axis=0),  
            'ul_control_channel_overhead': np.repeat(self.ul_control_channel_overhead, repeats=self.number_of_cells, axis=0),  
            
            'CRS_ports': np.repeat(self.CRS_ports, repeats=self.number_of_cells, axis=0), 
            'CRS_power_boosting_Pb': np.repeat(self.CRS_power_boosting_Pb, repeats=self.number_of_cells, axis=0), 
            'BS_tx_power_CRS_RE_dBm': np.repeat(self.CRS_RE_tx_power_dBm, repeats=self.number_of_cells, axis=0),  
            'BS_tx_power_PDSCH_dBm': np.repeat(self.CRS_RE_tx_power_dBm, repeats=self.number_of_cells, axis=0),  

            'antenna_pattern_model': np.repeat(self.antenna_pattern_model, repeats=self.number_of_cells, axis=0),
            
            'antenna_config_max_gain_dBi': np.repeat(self.antenna_config_max_gain_dBi, repeats=self.number_of_cells, axis=0), 
            
            'antenna_config_hor_phi_3dB_deg': np.repeat(self.antenna_config_hor_phi_3dB_deg, repeats=self.number_of_cells, axis=0),  
            'antenna_config_hor_A_m_dB': np.repeat(self.antenna_config_hor_A_m_dB, repeats=self.number_of_cells, axis=0), 
            'antenna_config_hor_alpha_elec_bearing_deg': np.tile(self.antenna_config_hor_alpha_elec_bearing_deg, len(self.cell_site_positions_m)), 
            'antenna_config_hor_alpha_mec_bearing_deg': np.tile(self.antenna_config_hor_alpha_mec_bearing_deg, len(self.cell_site_positions_m)),   
            
            'antenna_config_ver_theta_3dB_deg': np.repeat(self.antenna_config_ver_theta_3dB_deg, repeats=self.number_of_cells, axis=0),
            'antenna_config_ver_SLA_dB': np.repeat(self.antenna_config_ver_SLA_dB, repeats=self.number_of_cells, axis=0),
            'antenna_config_ver_beta_elec_downtilt_deg': np.repeat(self.antenna_config_ver_beta_elec_downtilt_deg, repeats=self.number_of_cells, axis=0), 
            'antenna_config_ver_beta_mec_downtilt_deg': np.repeat(self.antenna_config_ver_beta_mec_downtilt_deg, repeats=self.number_of_cells, axis=0),
            
            'antenna_config_gamma_elec_slant_deg': np.repeat(self.antenna_config_gamma_elec_slant_deg, repeats=self.number_of_cells, axis=0), 
            'antenna_config_gamma_mec_slant_deg': np.repeat(self.antenna_config_gamma_mec_slant_deg, repeats=self.number_of_cells, axis=0),  

            'antenna_config_Mg': np.repeat(self.antenna_config_Mg, repeats=self.number_of_cells, axis=0),  
            'antenna_config_Ng': np.repeat(self.antenna_config_Ng, repeats=self.number_of_cells, axis=0), 
            'antenna_config_M': np.repeat(self.antenna_config_M, repeats=self.number_of_cells, axis=0), 
            'antenna_config_N': np.repeat(self.antenna_config_N, repeats=self.number_of_cells, axis=0), 
            'antenna_config_P': np.repeat(self.antenna_config_P, repeats=self.number_of_cells, axis=0), 
            'antenna_config_P_type': np.repeat(self.antenna_config_P_type, repeats=self.number_of_cells, axis=0), 
            'antenna_config_number_of_elements': np.repeat(self.antenna_config_number_of_elements, repeats=self.number_of_cells, axis=0),
            'antenna_config_dgh_m': np.repeat(self.antenna_config_dgh_m, repeats=self.number_of_cells, axis=0), 
            'antenna_config_dgv_m': np.repeat(self.antenna_config_dgv_m, repeats=self.number_of_cells, axis=0),                
            'antenna_config_dh_m': np.repeat(self.antenna_config_dh_m, repeats=self.number_of_cells, axis=0), 
            'antenna_config_dv_m': np.repeat(self.antenna_config_dv_m, repeats=self.number_of_cells, axis=0),     

            'antenna_radius_ReflectorAperture_m' : np.repeat(getattr(self, 'antenna_radius_ReflectorAperture_m', None), repeats=self.number_of_cells, axis=0),
 
            'SSB_precoder': np.repeat(self.SSB_precoder, repeats=self.number_of_cells, axis=0), 
            'SSB_number_of_beams': np.repeat(self.SSB_number_of_beams, repeats=self.number_of_cells, axis=0),   
            'SSB_number_of_beams_H': np.repeat(self.SSB_number_of_beams_H, repeats=self.number_of_cells, axis=0), 
            'SSB_number_of_beams_V': np.repeat(self.SSB_number_of_beams_V, repeats=self.number_of_cells, axis=0),  
            'SSB_BeamDirection_phi_deg' : [self.SSB_BeamDirection_phi_deg]*self.number_of_cells, 
            'SSB_BeamDirection_theta_deg' : [self.SSB_BeamDirection_theta_deg]*self.number_of_cells,
      
            'CSI_RS_precoder': np.repeat(self.CSI_RS_precoder, repeats=self.number_of_cells, axis=0), 
            'CSI_RS_number_of_beams': np.repeat(self.CSI_RS_number_of_beams, repeats=self.number_of_cells, axis=0),   
            'CSI_RS_number_of_beams_H': np.repeat(self.CSI_RS_number_of_beams_H, repeats=self.number_of_cells, axis=0), 
            'CSI_RS_number_of_beams_V': np.repeat(self.CSI_RS_number_of_beams_V, repeats=self.number_of_cells, axis=0),    
            'CSI_RS_BeamDirection_phi_deg' : [self.SSB_BeamDirection_phi_deg]*self.number_of_cells, 
            'CSI_RS_BeamDirection_theta_deg' : [self.SSB_BeamDirection_theta_deg]*self.number_of_cells,
           
            'BS_propagation_model': np.repeat(self.BS_propagation_model, repeats=self.number_of_cells, axis=0), 
            'BS_fast_channel_model': np.repeat(self.BS_fast_channel_model, repeats=self.number_of_cells, axis=0),           
            'BS_scheduling_model': np.repeat("random_per_cell_beam", repeats=self.number_of_cells, axis=0)
            }
       
       
       # Create pandas dataframe with decorators for tracking
       df_ep = TrackedDataFrame(data=engineering_parameters_d, key_to_track='BS_prb_requirement_calculator_model') # key_to_track is the name of the coloumn to keep track
       
       return df_ep
        
    
    def calculate_bs_velocity_vectors(self):
         
        # Generate a random 3D vector with z component equal to zero
         
        # Generate random numbers between -1 and 1
        random_velocities = np.zeros((self.number_of_cells, 2))
        random_velocities[:,0] = 1
     
        # Combine with a zero column for a third dimension
        velocities = np.column_stack((random_velocities, np.zeros(self.number_of_cells)))        
         
        # Normalize each row
        self.velocity_vector_kmh = np.apply_along_axis(lambda row: row / np.linalg.norm(row), axis=1, arr=velocities)
        self.velocity_vector_kmh = self.velocity_kmh * self.velocity_vector_kmh
               
      
    def construct_scenario_ITU_R_M2135_UMa(self, site_deployment_obj):
            
       self.number_of_sectors_per_site = 3
       self.bs_antenna_height_m = 25
       self.indoor = 0  
        
       # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
           # Note that not in all scenarios reference site locations equals to cell site locations
       self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

       
       # Calculate number of cell sites and cells
       self.number_of_cell_sites = len(self.cell_site_positions_m)
       self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
       # Derive cell locations 
           # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
       self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
       self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
       
       # Set velocity 
       self.velocity_kmh = 0
       self.calculate_bs_velocity_vectors()       
       
       # Radio access technology
       self.RAT = "LTE"
       
       # Type of rado unit and bbu 
       self.radio_unit_type = "4G-Macro-RRU-v1"
       self.bbu_type = "4G-Macro-BBU-v1"
       
       # Set relevant parameters
       self.fdd_tdd_ind	= "TDD"
       self.subframe_assignment = "SA2"
       self.FR = 1
       self.subcarrier_per_PRB =  12
       self.ofdm_symbols_in_slot = 14
       
       self.dl_freq_band = -1 #41
       self.dl_earfcn = -1 #40936
       self.ul_freq_band = -1 #41
       self.ul_earfcn = -1 #40936
       if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
           self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
       else:
           self.dl_carrier_frequency_GHz = 2
           self.ul_carrier_frequency_GHz = 2

       self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
       self.dl_bandwidth_MHz = 20   
       self.dl_subcarrier_spacing_kHz = 15
       self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz)
       self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.dl_subcarrier_spacing_kHz)
       self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(self.dl_subcarrier_spacing_kHz)
       
       self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
       self.ul_bandwidth_MHz = 20
       if (self.fdd_tdd_ind	== "TDD"):
           if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
               np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
               exit(0) 
       self.ul_subcarrier_spacing_kHz =  15
       self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
       self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
       self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
       
       self.BS_propagation_model = "ITU_R_M2135_UMa"
       self.BS_fast_channel_model = "Rician"

       self.BS_tx_power_dBm = 49 #We substract here the feeder loss
       self.BS_noise_figure_dB = 5
       
       self.TX_number = 1 
       self.RX_number = 1 
       
       self.CRS_ports = 1,
       self.CRS_RE_tx_power_dBm = -1,
       self.CRS_power_boosting_Pb = -1,       
       
       self.antenna_pattern_model = "3GPPTR36_814_UMa" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
       
       #Array in TR 36.814
       self.antenna_config_max_gain_dBi = 17
       self.antenna_config_hor_phi_3dB_deg = 70
       self.antenna_config_hor_A_m_dB = 20
       self.antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
       self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0, 0, 0])

       self.antenna_config_ver_theta_3dB_deg = 15
       self.antenna_config_ver_SLA_dB = 20
       self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon      
       self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 12 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array

       self.antenna_config_gamma_mec_slant_deg = 0
       self.antenna_config_gamma_elec_slant_deg = 0

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
       
       self.antenna_config_dgh_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dgv_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
       
       # SSB precoding 
       self.SSB_precoder = "None" 
       self.SSB_number_of_beams_H =  1
       self.SSB_number_of_beams_V =  1
       self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
      
       # CSI-RS precoding
       self.CSI_RS_precoder = "None"
       self.CSI_RS_number_of_beams_H =  1
       self.CSI_RS_number_of_beams_V =  1
       self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V  
       
       # Create sectores 
       self.sectors, self.sectors_gp = self.sectorized_geographical_areas(site_deployment_obj.hexagons, self.cell_site_positions_m, site_deployment_obj.isd_m, self.antenna_config_hor_alpha_mec_bearing_deg)
       
       ### Construct UE enginering parameters 
       return self.construct_data_frame()           
      
       
    def construct_scenario_ITU_R_M2135_UMi(self, site_deployment_obj):
       
       self.number_of_sectors_per_site = 3
       self.bs_antenna_height_m = 10
       self.indoor = 0  
        
       # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
           # Note that not in all scenarios reference site locations equals to cell site locations
       self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

       # Calculate number of cell sites and cells
       self.number_of_cell_sites = len(self.cell_site_positions_m)
       self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
       # Derive cell locations 
           # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
       self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
       self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))

       # Set velocity
       self.velocity_kmh = 0
       self.calculate_bs_velocity_vectors()        

       # Radio access technology
       self.RAT = "LTE"
       
       # Type of rado unit and bbu 
       self.radio_unit_type = "4G-Micro-RRU-v1"
       self.bbu_type = "4G-Micro-BBU-v1"       
       
       # Set relevant parameters
       self.fdd_tdd_ind	= "TDD"
       self.subframe_assignment = "SA2"
       self.FR = 1
       self.subcarrier_per_PRB =  12
       self.ofdm_symbols_in_slot = 14
       
       self.dl_freq_band = -1 #41
       self.dl_earfcn = -1 #40936
       self.ul_freq_band = -1 #41
       self.ul_earfcn = -1 #40936
       if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
           self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
       else:
           self.dl_carrier_frequency_GHz = 2.5
           self.ul_carrier_frequency_GHz = 2.5

       self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
       self.dl_bandwidth_MHz = 20   
       self.dl_subcarrier_spacing_kHz = 15
       self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz)
       self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.dl_subcarrier_spacing_kHz)
       self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(self.dl_subcarrier_spacing_kHz)
       
       self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
       self.ul_bandwidth_MHz = 20
       if (self.fdd_tdd_ind	== "TDD"):
           if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
               np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
               exit(0) 
       self.ul_subcarrier_spacing_kHz =  15
       self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
       self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
       self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
       
       self.BS_propagation_model = "ITU_R_M2135_UMi"
       self.BS_fast_channel_model = "Rician"

       self.BS_tx_power_dBm = 44 #We substract here the feeder loss
       self.BS_noise_figure_dB = 5
       
       self.TX_number = 1 
       self.RX_number = 1        
       
       self.CRS_ports = 1,
       self.CRS_RE_tx_power_dBm = -1,
       self.CRS_power_boosting_Pb = -1,       
       
       self.antenna_pattern_model = "3GPPTR36_814_UMa" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
       
       #Array in TR 36.814
       self.antenna_config_max_gain_dBi = 17
       self.antenna_config_hor_phi_3dB_deg = 70
       self.antenna_config_hor_A_m_dB = 20
       self.antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
       self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0, 0, 0])       
      
       self.antenna_config_ver_theta_3dB_deg = 15
       self.antenna_config_ver_SLA_dB = 20
       self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon         
       self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 12 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array  # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array 
       
       self.antenna_config_gamma_mec_slant_deg = 0
       self.antenna_config_gamma_elec_slant_deg = 0

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
       
       self.antenna_config_dgh_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dgv_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
       
       # SSB precoding 
       self.SSB_precoder = "None" 
       self.SSB_number_of_beams_H =  1
       self.SSB_number_of_beams_V =  1
       self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
      
       # CSI-RS precoding
       self.CSI_RS_precoder = "None"
       self.CSI_RS_number_of_beams_H =  1
       self.CSI_RS_number_of_beams_V =  1
       self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V       
    
       ### Construct UE enginering parameters 
       return self.construct_data_frame()       
   

    def construct_scenario_3GPPTR36_814_Case_1(self, site_deployment_obj):
       
       self.number_of_sectors_per_site = 3
       self.bs_antenna_height_m = 32
       self.indoor = 0  
        
       # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
           # Note that not in all scenarios reference site locations equals to cell site locations
       self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

       # Calculate number of cell sites and cells
       self.number_of_cell_sites = len(self.cell_site_positions_m)
       self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
       # Derive cell locations 
           # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
       self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
       self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
       
       # Set velocity 
       self.velocity_kmh = 0
       self.calculate_bs_velocity_vectors()        
       
       # Radio access technology
       self.RAT = "LTE"
       
       # Type of rado unit and bbu 
       self.radio_unit_type = "4G-Macro-RRU-v1"
       self.bbu_type = "4G-Macro-BBU-v1"       
       
       # Set relevant parameters
       self.fdd_tdd_ind	= "TDD"
       self.subframe_assignment = "SA2"
       self.FR = 1
       self.subcarrier_per_PRB =  12
       self.ofdm_symbols_in_slot = 14
       
       self.dl_freq_band = -1 #41
       self.dl_earfcn = -1 #40936
       self.ul_freq_band = -1 #41
       self.ul_earfcn = -1 #40936
       if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
           self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
       else:
           self.dl_carrier_frequency_GHz = 2
           self.ul_carrier_frequency_GHz = 2

       self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
       self.dl_bandwidth_MHz = 5   
       self.dl_subcarrier_spacing_kHz = 15
       self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz)
       self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.dl_subcarrier_spacing_kHz)
       self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(self.dl_subcarrier_spacing_kHz)
       
       self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
       self.ul_bandwidth_MHz = 5
       if (self.fdd_tdd_ind	== "TDD"):
           if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
               np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
               exit(0) 
       self.ul_subcarrier_spacing_kHz =  15
       self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
       self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
       self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
       
       self.BS_propagation_model = "3GPPTR36_814_Case_1"
       self.BS_fast_channel_model = "Rician"

       self.BS_tx_power_dBm = 43-0 #We substract here the feeder loss
       self.BS_noise_figure_dB = 5
       
       self.TX_number = 1 
       self.RX_number = 1        
       
       self.CRS_ports = 1,
       self.CRS_RE_tx_power_dBm = -1,
       self.CRS_power_boosting_Pb = -1,       
       
       self.antenna_pattern_model = "3GPPTR36_814_UMa" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
       
       #Array in TR 36.814
       self.antenna_config_max_gain_dBi = 14
       self.antenna_config_hor_phi_3dB_deg = 70
       self.antenna_config_hor_A_m_dB = 25
       self.antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
       self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0, 0, 0])    
       
       self.antenna_config_ver_theta_3dB_deg = 10
       self.antenna_config_ver_SLA_dB = 20
       self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
       self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 15 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array   

       self.antenna_config_gamma_mec_slant_deg = 0
       self.antenna_config_gamma_elec_slant_deg = 0
                
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
       
       self.antenna_config_dgh_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dgv_m = 0.5 * self.dl_carrier_wavelength_m       
       self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
       
       # SSB precoding 
       self.SSB_precoder = "None" 
       self.SSB_number_of_beams_H =  1
       self.SSB_number_of_beams_V =  1
       self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
      
       # CSI-RS precoding
       self.CSI_RS_precoder = "None"
       self.CSI_RS_number_of_beams_H =  1
       self.CSI_RS_number_of_beams_V =  1
       self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V         
       
       ### Construct UE enginering parameters
       return self.construct_data_frame()     
   

    def construct_scenario_3GPPTR36_814_Case_1_single_bs(self, site_deployment_obj):
       
       self.number_of_sectors_per_site = 3
       self.bs_antenna_height_m = 32
       self.indoor = 0  
        
       # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
           # Note that not in all scenarios reference site locations equals to cell site locations
       self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 
       
       # Calculate number of cell sites and cells
       self.number_of_cell_sites = len(self.cell_site_positions_m)
       self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
       # Derive cell locations 
           # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
       self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
       self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
       
       # Set velocity 
       self.velocity_kmh = 0
       self.calculate_bs_velocity_vectors()        
       
       # Radio access technology
       self.RAT = "LTE"
       
       # Type of rado unit and bbu 
       self.radio_unit_type = "4G-Macro-RRU-v1"
       self.bbu_type = "4G-Macro-BBU-v1"       
       
       # Set relevant parameters
       self.fdd_tdd_ind	= "TDD"
       self.subframe_assignment = "SA2"
       self.FR = 1
       self.subcarrier_per_PRB =  12
       self.ofdm_symbols_in_slot = 14
       
       self.dl_freq_band = -1 #41
       self.dl_earfcn = -1 #40936
       self.ul_freq_band = -1 #41
       self.ul_earfcn = -1 #40936
       if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
           self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
       else:
           self.dl_carrier_frequency_GHz = 2
           self.ul_carrier_frequency_GHz = 2

       self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
       self.dl_bandwidth_MHz = 5   
       self.dl_subcarrier_spacing_kHz = 15
       self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz)
       self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.dl_subcarrier_spacing_kHz)
       self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(self.dl_subcarrier_spacing_kHz)
       
       self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
       self.ul_bandwidth_MHz = 5
       if (self.fdd_tdd_ind	== "TDD"):
           if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
               np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
               exit(0) 
       self.ul_subcarrier_spacing_kHz =  15
       self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
       self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
       self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
       
       self.BS_propagation_model = "3GPPTR36_814_Case_1"
       self.BS_fast_channel_model = "Rician"

       self.BS_tx_power_dBm = 43-0 #We substract here the feeder loss
       self.BS_noise_figure_dB = 5
       
       self.TX_number = 1 
       self.RX_number = 1        
       
       self.CRS_ports = 1,
       self.CRS_RE_tx_power_dBm = -1,
       self.CRS_power_boosting_Pb = -1,       
       
       self.antenna_pattern_model = "3GPPTR36_814_UMa" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
       
       #Array in TR 36.814
       self.antenna_config_max_gain_dBi = 14
       self.antenna_config_hor_phi_3dB_deg = 70
       self.antenna_config_hor_A_m_dB = 25
       self.antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
       self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0, 0, 0])    
       
       self.antenna_config_ver_theta_3dB_deg = 10
       self.antenna_config_ver_SLA_dB = 20
       self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
       self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 15 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array   

       self.antenna_config_gamma_mec_slant_deg = 0
       self.antenna_config_gamma_elec_slant_deg = 0
                
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
       
       self.antenna_config_dgh_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dgv_m = 0.5 * self.dl_carrier_wavelength_m       
       self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
       
       # SSB precoding 
       self.SSB_precoder = "None" 
       self.SSB_number_of_beams_H =  1
       self.SSB_number_of_beams_V =  1
       self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
      
       # CSI-RS precoding
       self.CSI_RS_precoder = "None"
       self.CSI_RS_number_of_beams_H =  1
       self.CSI_RS_number_of_beams_V =  1
       self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V         
       
       ### Construct UE enginering parameters
       return self.construct_data_frame()    
       

    def construct_scenario_3GPPTR36_814_Case_1_omni(self, site_deployment_obj):
       
       self.number_of_sectors_per_site = 1
       self.bs_antenna_height_m = 32
       self.indoor = 0  
        
       # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
           # Note that not in all scenarios reference site locations equals to cell site locations
       self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

       # Calculate number of cell sites and cells
       self.number_of_cell_sites = len(self.cell_site_positions_m)
       self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
       # Derive cell locations 
           # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
       self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
       self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
       
       # Set velocity 
       self.velocity_kmh = 0
       self.calculate_bs_velocity_vectors()        
       
       # Radio access technology
       self.RAT = "LTE"
       
       # Type of rado unit and bbu 
       self.radio_unit_type = "4G-Macro-RRU-v1"
       self.bbu_type = "4G-Macro-BBU-v1"       
       
       # Set relevant parameters
       self.fdd_tdd_ind	= "TDD"
       self.subframe_assignment = "SA2"
       self.FR = 1
       self.subcarrier_per_PRB =  12
       self.ofdm_symbols_in_slot = 14
       
       self.dl_freq_band = -1 #41
       self.dl_earfcn = -1 #40936
       self.ul_freq_band = -1 #41
       self.ul_earfcn = -1 #40936
       if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
           self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
       else:
           self.dl_carrier_frequency_GHz = 2
           self.ul_carrier_frequency_GHz = 2

       self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
       self.dl_bandwidth_MHz = 5   
       self.dl_subcarrier_spacing_kHz = 15
       self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz)
       self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.dl_subcarrier_spacing_kHz)
       self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(self.dl_subcarrier_spacing_kHz)
       
       self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
       self.ul_bandwidth_MHz = 5
       if (self.fdd_tdd_ind	== "TDD"):
           if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
               np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
               exit(0) 
       self.ul_subcarrier_spacing_kHz =  15
       self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
       self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
       self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
       
       self.BS_propagation_model = "3GPPTR36_814_Case_1"
       self.BS_fast_channel_model = "Rician"       

       self.BS_tx_power_dBm = 43-0 #We substract here the feeder loss
       self.BS_noise_figure_dB = 5
       
       self.TX_number = 1 
       self.RX_number = 1        
       
       self.CRS_ports = 1,
       self.CRS_RE_tx_power_dBm = -1,
       self.CRS_power_boosting_Pb = -1,       
       
       self.antenna_pattern_model = "omnidirectional" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
       
       #Array in TR 36.814
       self.antenna_config_max_gain_dBi = 5
       self.antenna_config_hor_phi_3dB_deg = np.nan
       self.antenna_config_hor_A_m_dB = np.nan
       self.antenna_config_hor_alpha_mec_bearing_deg = np.array([0])
       self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0])    
       
       self.antenna_config_ver_theta_3dB_deg = np.nan
       self.antenna_config_ver_SLA_dB = np.nan
       self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
       self.antenna_config_ver_beta_elec_downtilt_deg = 90 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array   

       self.antenna_config_gamma_mec_slant_deg = 0
       self.antenna_config_gamma_elec_slant_deg = 0
                
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
       
       self.antenna_config_dgh_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dgv_m = 0.5 * self.dl_carrier_wavelength_m       
       self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
       
       # SSB precoding 
       self.SSB_precoder = "None" 
       self.SSB_number_of_beams_H =  1
       self.SSB_number_of_beams_V =  1
       self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
      
       # CSI-RS precoding
       self.CSI_RS_precoder = "None"
       self.CSI_RS_number_of_beams_H =  1
       self.CSI_RS_number_of_beams_V =  1
       self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V         
       
       ### Construct UE enginering parameters
       return self.construct_data_frame()    


    def construct_scenario_3GPPTR36_814_Case_1_single_bs_omni(self, site_deployment_obj):
       
       self.number_of_sectors_per_site = 1
       self.bs_antenna_height_m = 32
       self.indoor = 0  
        
       # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
           # Note that not in all scenarios reference site locations equals to cell site locations
       self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

       
       # Calculate number of cell sites and cells
       self.number_of_cell_sites = len(self.cell_site_positions_m)
       self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
       # Derive cell locations 
           # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
       self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
       self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
       
       # Set velocity 
       self.velocity_kmh = 0
       self.calculate_bs_velocity_vectors()        
       
       # Radio access technology
       self.RAT = "LTE"
       
       # Type of rado unit and bbu 
       self.radio_unit_type = "4G-Macro-RRU-v1"
       self.bbu_type = "4G-Macro-BBU-v1"       
       
       # Set relevant parameters
       self.fdd_tdd_ind	= "TDD"
       self.subframe_assignment = "SA2"
       self.FR = 1
       self.subcarrier_per_PRB =  12
       self.ofdm_symbols_in_slot = 14
       
       self.dl_freq_band = -1 #41
       self.dl_earfcn = -1 #40936
       self.ul_freq_band = -1 #41
       self.ul_earfcn = -1 #40936
       if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
           self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
       else:
           self.dl_carrier_frequency_GHz = 2
           self.ul_carrier_frequency_GHz = 2

       self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
       self.dl_bandwidth_MHz = 5   
       self.dl_subcarrier_spacing_kHz = 15
       self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz)
       self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.dl_subcarrier_spacing_kHz)
       self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(self.dl_subcarrier_spacing_kHz)
       
       self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
       self.ul_bandwidth_MHz = 5
       if (self.fdd_tdd_ind	== "TDD"):
           if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
               np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
               exit(0) 
       self.ul_subcarrier_spacing_kHz =  15
       self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
       self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
       self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
       
       self.BS_propagation_model = "3GPPTR36_814_Case_1"
       self.BS_fast_channel_model = "Rician"

       self.BS_tx_power_dBm = 43-0 #We substract here the feeder loss
       self.BS_noise_figure_dB = 5
       
       self.TX_number = 1 
       self.RX_number = 1        
       
       self.CRS_ports = 1,
       self.CRS_RE_tx_power_dBm = -1,
       self.CRS_power_boosting_Pb = -1,       
       
       self.antenna_pattern_model = "omnidirectional" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
       
       #Array in TR 36.814
       self.antenna_config_max_gain_dBi = 5
       self.antenna_config_hor_phi_3dB_deg = np.nan
       self.antenna_config_hor_A_m_dB = np.nan
       self.antenna_config_hor_alpha_mec_bearing_deg = np.array([0])
       self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0])    
       
       self.antenna_config_ver_theta_3dB_deg = np.nan
       self.antenna_config_ver_SLA_dB = np.nan
       self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
       self.antenna_config_ver_beta_elec_downtilt_deg = 90 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array   

       self.antenna_config_gamma_mec_slant_deg = 0
       self.antenna_config_gamma_elec_slant_deg = 0
                
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
       
       self.antenna_config_dgh_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dgv_m = 0.5 * self.dl_carrier_wavelength_m       
       self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
       
       # SSB precoding 
       self.SSB_precoder = "None" 
       self.SSB_number_of_beams_H =  1
       self.SSB_number_of_beams_V =  1
       self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
      
       # CSI-RS precoding
       self.CSI_RS_precoder = "None"
       self.CSI_RS_number_of_beams_H =  1
       self.CSI_RS_number_of_beams_V =  1
       self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V         
       
       ### Construct UE enginering parameters
       return self.construct_data_frame() 


    def construct_scenario_3GPPTR36_814_Case_1_omni_dana(self, site_deployment_obj):
       
       self.number_of_sectors_per_site = 1
       self.bs_antenna_height_m = 100
       self.indoor = 0  
        
       # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
           # Note that not in all scenarios reference site locations equals to cell site locations       
       self.cell_site_positions_m = np.full((3,2), np.nan, dtype=np.single) 
       self.cell_site_positions_m[0,:] = np.array([-50,-100])
       self.cell_site_positions_m[1,:] = np.array([50, 0])
       self.cell_site_positions_m[2,:] = np.array([-50,100])       

       # Calculate number of cell sites and cells
       self.number_of_cell_sites = len(self.cell_site_positions_m)
       self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
       # Derive cell locations 
           # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
       self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
       self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
       
       # Set velocity 
       self.velocity_kmh = 30.0
       self.calculate_bs_velocity_vectors()
       
       # Radio access technology
       self.RAT = "LTE"
       
       # Type of rado unit and bbu 
       self.radio_unit_type = "4G-Macro-RRU-v1"
       self.bbu_type = "4G-Macro-BBU-v1"       
       
       # Set relevant parameters
       self.fdd_tdd_ind	= "TDD"
       self.subframe_assignment = "SA2"
       self.FR = 1
       self.subcarrier_per_PRB =  12
       self.ofdm_symbols_in_slot = 14
       
       self.dl_freq_band = -1 #41
       self.dl_earfcn = -1 #40936
       self.ul_freq_band = -1 #41
       self.ul_earfcn = -1 #40936
       if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
           self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
       else:
           self.dl_carrier_frequency_GHz = 2
           self.ul_carrier_frequency_GHz = 2

       self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
       self.dl_bandwidth_MHz = 5   
       self.dl_subcarrier_spacing_kHz = 15
       self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz)
       self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.dl_subcarrier_spacing_kHz)
       self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(self.dl_subcarrier_spacing_kHz)
       
       self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
       self.ul_bandwidth_MHz = 5
       if (self.fdd_tdd_ind	== "TDD"):
           if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
               np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
               exit(0) 
       self.ul_subcarrier_spacing_kHz =  15
       self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
       self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
       self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
       
       self.BS_propagation_model = "3GPPTR36_814_Case_1"
       self.BS_fast_channel_model = "Rician"

       self.BS_tx_power_dBm = 43-0 #We substract here the feeder loss
       self.BS_noise_figure_dB = 5
       
       self.TX_number = 1 
       self.RX_number = 1        
       
       self.CRS_ports = 1,
       self.CRS_RE_tx_power_dBm = -1,
       self.CRS_power_boosting_Pb = -1,       
       
       self.antenna_pattern_model = "omnidirectional" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
       
       #Array in TR 36.814
       self.antenna_config_max_gain_dBi = 5
       self.antenna_config_hor_phi_3dB_deg = np.nan
       self.antenna_config_hor_A_m_dB = np.nan
       self.antenna_config_hor_alpha_mec_bearing_deg = np.array([0])
       self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0])    
       
       self.antenna_config_ver_theta_3dB_deg = np.nan
       self.antenna_config_ver_SLA_dB = np.nan
       self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
       self.antenna_config_ver_beta_elec_downtilt_deg = 90 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array   

       self.antenna_config_gamma_mec_slant_deg = 0
       self.antenna_config_gamma_elec_slant_deg = 0
                
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
       
       self.antenna_config_dgh_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dgv_m = 0.5 * self.dl_carrier_wavelength_m       
       self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
       
       # SSB precoding 
       self.SSB_precoder = "None" 
       self.SSB_number_of_beams_H =  1
       self.SSB_number_of_beams_V =  1
       self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
      
       # CSI-RS precoding
       self.CSI_RS_precoder = "None"
       self.CSI_RS_number_of_beams_H =  1
       self.CSI_RS_number_of_beams_V =  1
       self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V         
       
       ### Construct UE enginering parameters
       return self.construct_data_frame()     
     
        
    def construct_scenario_3GPPTR38_901_UMa_C1(self, site_deployment_obj):
              
       self.number_of_sectors_per_site = 3
       self.bs_antenna_height_m = 25
       self.indoor = 0  
       
       # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
           # Note that not in all scenarios reference site locations equals to cell site locations
       self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

       # Calculate number of cell sites and cells
       self.number_of_cell_sites = len(self.cell_site_positions_m)
       self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
       # Derive cell locations 
           # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
       self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
       self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
       
       # Set velocity 
       self.velocity_kmh = 0
       self.calculate_bs_velocity_vectors()        
        
       # Radio access technology
       self.RAT = "NR"
       
       # Type of rado unit and bbu 
       self.radio_unit_type = "5G-Macro-AAU-v1"
       self.bbu_type = "5G-Macro-BBU-v1"   
        
       # Set relevant parameters
       self.fdd_tdd_ind	= "TDD"
       self.subframe_assignment = "SA2"
       self.FR = 1
       self.subcarrier_per_PRB =  12
       self.ofdm_symbols_in_slot = 14
       
       self.dl_freq_band = -1 #41
       self.dl_earfcn = -1 #40936
       self.ul_freq_band = -1 #41
       self.ul_earfcn = -1 #40936
       if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
           self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
       else:
           self.dl_carrier_frequency_GHz = 6
           self.ul_carrier_frequency_GHz = 6

       self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
       self.dl_bandwidth_MHz = 20   
       self.dl_subcarrier_spacing_kHz = 15
       self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz)
       self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.dl_subcarrier_spacing_kHz)
       self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(self.dl_subcarrier_spacing_kHz)
       
       self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
       self.ul_bandwidth_MHz = 20
       if (self.fdd_tdd_ind	== "TDD"):
           if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
               np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
               exit(0) 
       self.ul_subcarrier_spacing_kHz =  15
       self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
       self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
       self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
       
       self.BS_propagation_model = "3GPPTR38_901_UMa"
       self.BS_fast_channel_model = "3GPPTR38_901_UMa" #"3GPPTR38_901_UMa" #"Rician"

       self.BS_tx_power_dBm = 49
       self.BS_noise_figure_dB = 5 #Taken from ITU UMa 
       
       self.TX_number = 32 
       self.RX_number = 32        
       
       self.CRS_ports = 2
       self.CRS_RE_tx_power_dBm = -1
       self.CRS_power_boosting_Pb = -1       
       
       self.antenna_pattern_model = "3GPPTR38_901" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
       self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)
       
       #Element in TR 38.901
       self.antenna_config_max_gain_dBi = 8
       self.antenna_config_hor_phi_3dB_deg = 65
       self.antenna_config_hor_A_m_dB = 30
       self.antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
       self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0, 0, 0]) 
       
       self.antenna_config_ver_theta_3dB_deg = 65
       self.antenna_config_ver_SLA_dB = 30
       self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
       self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 20 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array  
       
       self.antenna_config_gamma_mec_slant_deg = 0
       self.antenna_config_gamma_elec_slant_deg = 0
         
       #TR 38.901
       self.antenna_config_Mg = 1
       self.antenna_config_Ng = 2
       self.antenna_config_M = 4 # number of rows (elements in vertical)
       self.antenna_config_N = 4 # number of columns (elements in horizontal)
       self.antenna_config_P = "single" # (“single” or “dual”)      
       
       if self.antenna_config_P == 'single': 
           P_num = 1
       else :
           P_num = 2
       self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
       
       self.antenna_config_dgh_m = 2.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dgv_m = 2.5 * self.dl_carrier_wavelength_m       
       self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
       
       # SSB precoding 
       self.SSB_precoder = "3GPPTR38_901_P2_C1_one_port_per_panel" 
       self.SSB_number_of_beams_H =  1
       self.SSB_number_of_beams_V =  2
       self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
      
       # CSI-RS precoding
       self.CSI_RS_precoder = "3GPPTR38_901_P2_C1_one_port_per_panel"
       self.CSI_RS_number_of_beams_H =  1
       self.CSI_RS_number_of_beams_V =  2
       self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V
    
       ### Construct UE enginering parameters
       return self.construct_data_frame()  


    def construct_scenario_3GPPTR38_901_UMa_C2(self, site_deployment_obj):
              
       self.number_of_sectors_per_site = 3
       self.bs_antenna_height_m = 25
       self.indoor = 0  
       
       # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
           # Note that not in all scenarios reference site locations equals to cell site locations
       self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

       # Calculate number of cell sites and cells
       self.number_of_cell_sites = len(self.cell_site_positions_m)
       self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
       # Derive cell locations 
           # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
       self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
       self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
       
       # Set velocity 
       self.velocity_kmh = 0
       self.calculate_bs_velocity_vectors()        
        
       # Radio access technology
       self.RAT = "NR"
       
       # Type of rado unit and bbu 
       self.radio_unit_type = "5G-Macro-AAU-v1"
       self.bbu_type = "5G-Macro-BBU-v1"   
        
       # Set relevant parameters
       self.fdd_tdd_ind	= "TDD"
       self.subframe_assignment = "SA2"
       self.FR = 1
       self.subcarrier_per_PRB =  12
       self.ofdm_symbols_in_slot = 14
       
       self.dl_freq_band = -1 #41
       self.dl_earfcn = -1 #40936
       self.ul_freq_band = -1 #41
       self.ul_earfcn = -1 #40936
       if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
           self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
       else:
           self.dl_carrier_frequency_GHz = 6
           self.ul_carrier_frequency_GHz = 6

       self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
       self.dl_bandwidth_MHz = 20   
       self.dl_subcarrier_spacing_kHz = 15
       self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz)
       self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.dl_subcarrier_spacing_kHz)
       self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(self.dl_subcarrier_spacing_kHz)
       
       self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
       self.ul_bandwidth_MHz = 20
       if (self.fdd_tdd_ind	== "TDD"):
           if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
               np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
               exit(0) 
       self.ul_subcarrier_spacing_kHz =  15
       self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
       self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
       self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
       
       self.BS_propagation_model = "3GPPTR38_901_UMa"
       self.BS_fast_channel_model = "3GPPTR38_901_UMa" #"3GPPTR38_901_UMa" #"Rician"

       self.BS_tx_power_dBm = 49
       self.BS_noise_figure_dB = 5 #Taken from ITU UMa 
       
       self.TX_number = 4 
       self.RX_number = 4       
       
       self.CRS_ports = 4
       self.CRS_RE_tx_power_dBm = -1
       self.CRS_power_boosting_Pb = -1      
       
       self.antenna_pattern_model = "3GPPTR38_901" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
       self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)
       
       #Element in TR 38.901
       self.antenna_config_max_gain_dBi = 8
       self.antenna_config_hor_phi_3dB_deg = 65
       self.antenna_config_hor_A_m_dB = 30
       self.antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
       self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0, 0, 0]) 
       
       self.antenna_config_ver_theta_3dB_deg = 65
       self.antenna_config_ver_SLA_dB = 30
       self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
       self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 20 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array  
       
       self.antenna_config_gamma_mec_slant_deg = 0
       self.antenna_config_gamma_elec_slant_deg = 0
         
       #TR 38.901
       self.antenna_config_Mg = 1
       self.antenna_config_Ng = 1
       self.antenna_config_M = 2 # number of rows (elements in vertical)
       self.antenna_config_N = 2 # number of columns (elements in horizontal)
       self.antenna_config_P = "single" # (“single” or “dual”)      
       
       if self.antenna_config_P == 'single': 
           P_num = 1
       else :
           P_num = 2
       self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
       
       self.antenna_config_dgh_m = 0 #0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dgv_m = 0 #0.5 * self.dl_carrier_wavelength_m       
       self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
       
       # SSB precoding 
       self.SSB_precoder = "None" 
       self.SSB_number_of_beams_H =  2
       self.SSB_number_of_beams_V =  2
       self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
      
       # CSI-RS precoding
       self.CSI_RS_precoder = "None"
       self.CSI_RS_number_of_beams_H =  2
       self.CSI_RS_number_of_beams_V =  2
       self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V
    
       ### Construct UE enginering parameters
       return self.construct_data_frame()  


    def construct_scenario_3GPPTR38_901_UMa_large_scale_calibration(self, site_deployment_obj):
      
        self.number_of_sectors_per_site = 3
        self.bs_antenna_height_m = 25
        self.indoor = 0  
        
        # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
            # Note that not in all scenarios reference site locations equals to cell site locations
        self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

        # Calculate number of cell sites and cells
        self.number_of_cell_sites = len(self.cell_site_positions_m)
        self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
        # Derive cell locations 
            # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
        self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
        self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
        
        # Set velocity 
        self.velocity_kmh = 0
        self.calculate_bs_velocity_vectors()         
        
        # Radio access technology
        self.RAT = "NR"
       
        # Type of rado unit and bbu 
        self.radio_unit_type = "5G-Macro-AAU-v1"
        self.bbu_type = "5G-Macro-BBU-v1"          
       
        # Set relevant parameters
        self.fdd_tdd_ind	= "TDD"
        self.subframe_assignment = "SA2"
        self.FR = 1
        self.subcarrier_per_PRB =  12
        self.ofdm_symbols_in_slot = 14
        
        self.dl_freq_band = -1 #41
        self.dl_earfcn = -1 #40936
        self.ul_freq_band = -1 #41
        self.ul_earfcn = -1 #40936
        if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
            self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
        else:
            self.dl_carrier_frequency_GHz = 6
            self.ul_carrier_frequency_GHz = 6
      
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.dl_bandwidth_MHz = 20   
        self.dl_subcarrier_spacing_kHz = 15
        self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz)
        self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.dl_subcarrier_spacing_kHz)
        self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(self.dl_subcarrier_spacing_kHz)
        
        self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
        self.ul_bandwidth_MHz = 20
        if (self.fdd_tdd_ind	== "TDD"):
            if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
                np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
                exit(0) 
        self.ul_subcarrier_spacing_kHz =  15
        self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
        self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
        self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
        
        self.BS_propagation_model = "3GPPTR38_901_UMa"
        self.BS_fast_channel_model = "Rician"        
      
        self.BS_tx_power_dBm = 49
        self.BS_noise_figure_dB = 5 #Taken from ITU UMa 

        self.TX_number = 10 
        self.RX_number = 10        
        
        self.CRS_ports = 1
        self.CRS_RE_tx_power_dBm = -1
        self.CRS_power_boosting_Pb = -1       
        
        self.antenna_pattern_model = "3GPPTR38_901" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
        
        #Element in TR 38.901   
        self.antenna_config_max_gain_dBi = 8
        self.antenna_config_hor_phi_3dB_deg = 65
        self.antenna_config_hor_A_m_dB = 30
        self.antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
        self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0, 0, 0])
      
        self.antenna_config_ver_theta_3dB_deg = 65
        self.antenna_config_ver_SLA_dB = 30
        self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
        self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 12 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array 
      
        self.antenna_config_gamma_mec_slant_deg = 0
        self.antenna_config_gamma_elec_slant_deg = 0
          
        #TR 38.901
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 10 # number of rows (elements in vertical)
        self.antenna_config_N = 1 # number of columns (elements in horizontal)
        self.antenna_config_P = "single" # (“single” or “dual”)    
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_dgh_m = 0 # 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dgv_m = 0 # 0.5 * self.dl_carrier_wavelength_m       
        self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
        
        # SSB precoding 
        self.SSB_precoder = "3GPPTR38_901_P1_single_column"
        self.SSB_number_of_beams_V =  1
        self.SSB_number_of_beams_H =  1        
        self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
        
        # CSI-RS precoding
        self.CSI_RS_precoder = "3GPPTR38_901_P1_single_column" 
        self.CSI_RS_number_of_beams_V =  1
        self.CSI_RS_number_of_beams_H =  1        
        self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V
     
        ### Construct UE enginering parameters       
        return self.construct_data_frame()  


    def construct_scenario_3GPPTR38_901_UMa_large_scale_calibration_single_bs(self, site_deployment_obj):
      
        self.number_of_sectors_per_site = 3
        self.bs_antenna_height_m = 25
        self.indoor = 0  
        
        # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
            # Note that not in all scenarios reference site locations equals to cell site locations
        self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

       
        # Calculate number of cell sites and cells
        self.number_of_cell_sites = len(self.cell_site_positions_m)
        self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
        # Derive cell locations 
            # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
        self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
        self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
        
        # Set velocity 
        self.velocity_kmh = 0
        self.calculate_bs_velocity_vectors()         
        
        # Radio access technology
        self.RAT = "NR"
       
        # Type of rado unit and bbu 
        self.radio_unit_type = "5G-Macro-AAU-v1"
        self.bbu_type = "5G-Macro-BBU-v1"            
       
        # Set relevant parameters        
        self.fdd_tdd_ind	= "TDD"
        self.subframe_assignment = "SA2"
        self.FR = 1
        self.subcarrier_per_PRB =  12
        self.ofdm_symbols_in_slot = 14
        
        self.dl_freq_band = -1 #41
        self.dl_earfcn = -1 #40936
        self.ul_freq_band = -1 #41
        self.ul_earfcn = -1 #40936
        if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
            self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
        else:
            self.dl_carrier_frequency_GHz = 6
            self.ul_carrier_frequency_GHz = 6
      
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.dl_bandwidth_MHz = 10   
        self.dl_subcarrier_spacing_kHz = 15
        self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz)
        self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.dl_subcarrier_spacing_kHz)
        self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(self.dl_subcarrier_spacing_kHz)
        
        self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
        self.ul_bandwidth_MHz = 10
        if (self.fdd_tdd_ind	== "TDD"):
            if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
                np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
                exit(0) 
        self.ul_subcarrier_spacing_kHz =  15
        self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
        self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
        self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
        
        self.BS_propagation_model = "3GPPTR38_901_UMa"
        self.BS_fast_channel_model = "Rician"        
      
        self.BS_tx_power_dBm = 49
        self.BS_noise_figure_dB = 5 #Taken from ITU UMa
        
        self.TX_number = 10 
        self.RX_number = 10           
        
        self.CRS_ports = 1
        self.CRS_RE_tx_power_dBm = -1
        self.CRS_power_boosting_Pb = -1       
        
        self.antenna_pattern_model = "3GPPTR38_901" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
        
        #Element in TR 38.901
        self.antenna_config_max_gain_dBi = 8
        self.antenna_config_hor_phi_3dB_deg = 65
        self.antenna_config_hor_A_m_dB = 30
        self.antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
        self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0, 0, 0])
      
        self.antenna_config_ver_theta_3dB_deg = 65
        self.antenna_config_ver_SLA_dB = 30
        self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
        self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 12 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array  
      
        self.antenna_config_gamma_mec_slant_deg = 0
        self.antenna_config_gamma_elec_slant_deg = 0
          
        #TR 38.901
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 10 # number of rows (elements in vertical)
        self.antenna_config_N = 1 # number of columns (elements in horizontal)       
        self.antenna_config_P = "single" # (“single” or “dual”)    
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_dgh_m = 0 # 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dgv_m = 0 # 0.5 * self.dl_carrier_wavelength_m       
        self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
        
        # SSB precoding 
        self.SSB_precoder = "3GPPTR38_901_P1_single_column"
        self.SSB_number_of_beams_V =  1
        self.SSB_number_of_beams_H =  1        
        self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
        
        # CSI-RS precoding
        self.CSI_RS_precoder = "3GPPTR38_901_P1_single_column" 
        self.CSI_RS_number_of_beams_V =  1
        self.CSI_RS_number_of_beams_H =  1    
        self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V
     
        ### Construct UE enginering parameters       
        return self.construct_data_frame()      


    def construct_scenario_3GPPTR38_901_UMa_large_scale_calibration_single_sector(self, site_deployment_obj):
      
        self.number_of_sectors_per_site = 1
        self.bs_antenna_height_m = 25
        self.indoor = 0  
        
        # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
            # Note that not in all scenarios reference site locations equals to cell site locations
        self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

       
        # Calculate number of cell sites and cells
        self.number_of_cell_sites = len(self.cell_site_positions_m)
        self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
        # Derive cell locations 
            # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
        self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
        self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
        
        # Set velocity 
        self.velocity_kmh = 0
        self.calculate_bs_velocity_vectors()        
        
        # Radio access technology
        self.RAT = "NR"
       
        # Type of rado unit and bbu 
        self.radio_unit_type = "5G-Macro-AAU-v1"
        self.bbu_type = "5G-Macro-BBU-v1"            
       
        # Set relevant parameters        
        self.fdd_tdd_ind	= "TDD"
        self.subframe_assignment = "SA2"
        self.FR = 1
        self.subcarrier_per_PRB =  12
        self.ofdm_symbols_in_slot = 14
        
        self.dl_freq_band = -1 #41
        self.dl_earfcn = -1 #40936
        self.ul_freq_band = -1 #41
        self.ul_earfcn = -1 #40936
        if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
            self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
        else:
            self.dl_carrier_frequency_GHz = 6
            self.ul_carrier_frequency_GHz = 6
      
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.dl_bandwidth_MHz = 10   
        self.dl_subcarrier_spacing_kHz = 15
        self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz)
        self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.dl_subcarrier_spacing_kHz)
        self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(self.dl_subcarrier_spacing_kHz)
        
        self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
        self.ul_bandwidth_MHz = 10
        if (self.fdd_tdd_ind	== "TDD"):
            if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
                np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
                exit(0) 
        self.ul_subcarrier_spacing_kHz =  15
        self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
        self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
        self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
        
        self.BS_propagation_model = "3GPPTR38_901_UMa"
        self.BS_fast_channel_model = "Rician"        
      
        self.BS_tx_power_dBm = 49
        self.BS_noise_figure_dB = 5 #Taken from ITU UMa 
        
        self.TX_number = 8 
        self.RX_number = 8           
        
        self.CRS_ports = 8
        self.CRS_RE_tx_power_dBm = -1
        self.CRS_power_boosting_Pb = -1      
        
        self.antenna_pattern_model = "3GPPTR38_901" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
        
        #Element in TR 38.901
        self.antenna_config_max_gain_dBi = 8
        self.antenna_config_hor_phi_3dB_deg = 65
        self.antenna_config_hor_A_m_dB = 30
        self.antenna_config_hor_alpha_mec_bearing_deg = np.array([90])
        self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0])
      
        self.antenna_config_ver_theta_3dB_deg = 65
        self.antenna_config_ver_SLA_dB = 30
        self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
        self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 12 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array  
      
        self.antenna_config_gamma_mec_slant_deg = 0
        self.antenna_config_gamma_elec_slant_deg = 0
          
        #TR 38.901
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 1 # number of rows (elements in vertical)
        self.antenna_config_N = 8 # number of columns (elements in horizontal)       
        self.antenna_config_P = "single" # (“single” or “dual”)    
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_dgh_m = 0 # 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dgv_m = 0 # 0.5 * self.dl_carrier_wavelength_m       
        self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
        
        # SSB precoding 
        self.SSB_precoder = "custom" #"3GPPTR38_901_P1_single_column"
        self.SSB_number_of_beams_V =  1
        self.SSB_number_of_beams_H =  8        
        self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
        
        # CSI-RS precoding
        self.CSI_RS_precoder = "custom" #"3GPPTR38_901_P1_single_column"
        self.CSI_RS_number_of_beams_V =  1
        self.CSI_RS_number_of_beams_H =  8    
        self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V
     
        ### Construct UE enginering parameters       
        return self.construct_data_frame()       


    def construct_scenario_3GPPTR38_901_UMa_large_scale_calibration_sn(self, site_deployment_obj):
      
        self.number_of_sectors_per_site = 3
        self.bs_antenna_height_m = 25
        self.indoor = 0  
        
        # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
            # Note that not in all scenarios reference site locations equals to cell site locations
        self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

        # Calculate number of cell sites and cells
        self.number_of_cell_sites = len(self.cell_site_positions_m)
        self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
        # Derive cell locations 
            # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
        self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
        self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
        
        # Set velocity 
        self.velocity_kmh = 0
        self.calculate_bs_velocity_vectors()        
        
        # Radio access technology
        self.RAT = "NR"
       
        # Type of rado unit and bbu 
        self.radio_unit_type = "5G-Macro-AAU-v1"
        self.bbu_type = "5G-Macro-BBU-v1"          
       
        # Set relevant parameters
        self.fdd_tdd_ind	= "TDD"
        self.subframe_assignment = "SA2"
        self.FR = 1
        self.subcarrier_per_PRB =  12
        self.ofdm_symbols_in_slot = 14
        
        self.dl_freq_band = -1 #41
        self.dl_earfcn = -1 #40936
        self.ul_freq_band = -1 #41
        self.ul_earfcn = -1 #40936
        if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
            self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
        else:
            self.dl_carrier_frequency_GHz = 6
            self.ul_carrier_frequency_GHz = 6
      
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.dl_bandwidth_MHz = 20   
        self.dl_subcarrier_spacing_kHz = 15
        self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz)
        self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.dl_subcarrier_spacing_kHz)
        self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(self.dl_subcarrier_spacing_kHz)
        
        self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
        self.ul_bandwidth_MHz = 20
        if (self.fdd_tdd_ind	== "TDD"):
            if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
                np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
                exit(0) 
        self.ul_subcarrier_spacing_kHz =  15
        self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
        self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
        self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
        
        self.BS_propagation_model = "3GPPTR38_901_UMa"
        self.BS_fast_channel_model = "3GPPTR38_901_UMa"        
      
        self.BS_tx_power_dBm = 49
        self.BS_noise_figure_dB = 5 #Taken from ITU UMa 
        
        self.TX_number = 10 
        self.RX_number = 10           
        
        self.CRS_ports = 1
        self.CRS_RE_tx_power_dBm = -1
        self.CRS_power_boosting_Pb = -1       
        
        self.antenna_pattern_model = "3GPPTR38_901" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
        
        #Element in TR 38.901   
        self.antenna_config_max_gain_dBi = 8
        self.antenna_config_hor_phi_3dB_deg = 65
        self.antenna_config_hor_A_m_dB = 30
        self.antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
        self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0, 0, 0])
      
        self.antenna_config_ver_theta_3dB_deg = 65
        self.antenna_config_ver_SLA_dB = 30
        self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
        self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 12 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array 
      
        self.antenna_config_gamma_mec_slant_deg = 0
        self.antenna_config_gamma_elec_slant_deg = 0
          
        #TR 38.901
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 10 # number of rows (elements in vertical)
        self.antenna_config_N = 1 # number of columns (elements in horizontal)
        self.antenna_config_P = "single" # (“single” or “dual”)    
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_dgh_m = 0 # 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dgv_m = 0 # 0.5 * self.dl_carrier_wavelength_m       
        self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
        
        # SSB precoding 
        self.SSB_precoder = "3GPPTR38_901_P1_single_column" #"3GPP_single_downtilted_beam"
        self.SSB_number_of_beams_V =  1
        self.SSB_number_of_beams_H =  1        
        self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
        
        # CSI-RS precoding
        self.CSI_RS_precoder = "3GPPTR38_901_P1_single_column" #"3GPP_single_downtilted_beam"
        self.CSI_RS_number_of_beams_V =  1
        self.CSI_RS_number_of_beams_H =  1        
        self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V
     
        ### Construct UE enginering parameters       
        return self.construct_data_frame()  


    def construct_scenario_3GPPTR38_901_UMa_2GHz_large_scale_calibration(self, site_deployment_obj):
      
        self.number_of_sectors_per_site = 3
        self.bs_antenna_height_m = 25
        self.indoor = 0  
        
        # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
            # Note that not in all scenarios reference site locations equals to cell site locations
        self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 
       
        # Calculate number of cell sites and cells
        self.number_of_cell_sites = len(self.cell_site_positions_m)
        self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
        # Derive cell locations 
            # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
        self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
        self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
        
        # Set velocity 
        self.velocity_kmh = 0
        self.calculate_bs_velocity_vectors()        
        
        # Radio access technology
        self.RAT = "LTE"
       
        # Type of rado unit and bbu 
        self.radio_unit_type = "4G-Macro-RRU-v1"
        self.bbu_type = "4G-Macro-BBU-v1"            
       
        # Set relevant parameters
        self.fdd_tdd_ind	= "TDD"
        self.subframe_assignment = "SA2"
        self.FR = 1
        self.subcarrier_per_PRB =  12
        self.ofdm_symbols_in_slot = 14
        
        self.dl_freq_band = -1 #41
        self.dl_earfcn = -1 #40936
        self.ul_freq_band = -1 #41
        self.ul_earfcn = -1 #40936
        if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
            self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
        else:
            self.dl_carrier_frequency_GHz = 2
            self.ul_carrier_frequency_GHz = 2
      
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.dl_bandwidth_MHz = 20   
        self.dl_subcarrier_spacing_kHz = 15 * 20/10
        self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(10, 15) #self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz
        self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(15) # self.dl_subcarrier_spacing_kHz
        self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(15) # self.dl_subcarrier_spacing_kHz
        
        self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
        self.ul_bandwidth_MHz = 20
        if (self.fdd_tdd_ind	== "TDD"):
            if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
                np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
                exit(0) 
        self.ul_subcarrier_spacing_kHz =  15
        self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
        self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
        self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
        
        self.BS_propagation_model = "3GPPTR38_901_UMa"
        self.BS_fast_channel_model = "Rician"        
      
        self.BS_tx_power_dBm = 49
        self.BS_noise_figure_dB = 5 #Taken from ITU UMa 
        
        self.TX_number = 8 
        self.RX_number = 8           
        
        self.CRS_ports = 1
        self.CRS_RE_tx_power_dBm = -1
        self.CRS_power_boosting_Pb = -1       
        
        self.antenna_pattern_model = "3GPPTR38_901" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
        
        #Element in TR 38.901   
        self.antenna_config_max_gain_dBi = 8
        self.antenna_config_hor_phi_3dB_deg = 65
        self.antenna_config_hor_A_m_dB = 30
        self.antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
        self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0, 0, 0])
      
        self.antenna_config_ver_theta_3dB_deg = 65
        self.antenna_config_ver_SLA_dB = 30
        self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
        self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 12 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array 
      
        self.antenna_config_gamma_mec_slant_deg = 0
        self.antenna_config_gamma_elec_slant_deg = 0
          
        #TR 38.901
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 8 # number of rows (elements in vertical)
        self.antenna_config_N = 1 # number of columns (elements in horizontal)
        self.antenna_config_P = "single" # (“single” or “dual”)    
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_dgh_m = 0 # 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dgv_m = 0 # 0.5 * self.dl_carrier_wavelength_m       
        self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
        
        # SSB precoding 
        self.SSB_precoder = "3GPPTR38_901_P1_single_column" 
        self.SSB_number_of_beams_V =  1
        self.SSB_number_of_beams_H =  1        
        self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
        
        # CSI-RS precoding
        self.CSI_RS_precoder = "3GPPTR38_901_P1_single_column" 
        self.CSI_RS_number_of_beams_V =  1
        self.CSI_RS_number_of_beams_H =  1        
        self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V
     
        ### Construct UE enginering parameters       
        return self.construct_data_frame()  

    def construct_scenario_3GPPTR38_901_UMa_C_band_large_scale_calibration(self, site_deployment_obj):
      
        self.number_of_sectors_per_site = 3
        self.bs_antenna_height_m = 25
        self.indoor = 0 
        
        # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
            # Note that not in all scenarios reference site locations equals to cell site locations
        self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

        # Calculate number of cell sites and cells
        self.number_of_cell_sites = len(self.cell_site_positions_m)
        self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
        # Derive cell locations 
            # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
        self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
        self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
        
        # Set velocity 
        self.velocity_kmh = 0
        self.calculate_bs_velocity_vectors()        
        
        # Radio access technology
        self.RAT = "NR"
       
        # Type of rado unit and bbu 
        self.radio_unit_type = "5G-Macro-AAU-v1"
        self.bbu_type = "5G-Macro-BBU-v1"           
       
        # Set relevant parameters
        self.fdd_tdd_ind	= "TDD"
        self.subframe_assignment = "SA2"
        self.FR = 1
        self.subcarrier_per_PRB =  12
        self.ofdm_symbols_in_slot = 14
        
        self.dl_freq_band = -1 #41
        self.dl_earfcn = -1 #40936
        self.ul_freq_band = -1 #41
        self.ul_earfcn = -1 #40936
        if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
            self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
        else:
            self.dl_carrier_frequency_GHz = 3.5
            self.ul_carrier_frequency_GHz = 3.5
      
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.dl_bandwidth_MHz = 100   
        self.dl_subcarrier_spacing_kHz = 15 * 100/10
        self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(10, 15)
        self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(15)
        self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(15)
        
        self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
        self.ul_bandwidth_MHz = 100
        if (self.fdd_tdd_ind	== "TDD"):
            if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
                np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
                exit(0) 
        self.ul_subcarrier_spacing_kHz = 15 * 100/10
        self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(10, 15)
        self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(15)
        self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(15)
        
        self.BS_propagation_model = "3GPPTR38_901_UMa"
        self.BS_fast_channel_model = "Rician"        
      
        self.BS_tx_power_dBm = 55
        self.BS_noise_figure_dB = 5 #Taken from ITU UMa 
        
        self.TX_number = 32 
        self.RX_number = 32           
        
        self.CRS_ports = 32
        self.CRS_RE_tx_power_dBm = -1
        self.CRS_power_boosting_Pb = -1      
    
        self.antenna_pattern_model = "3GPPTR38_901" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
        
        #Element in TR 38.901
        self.antenna_config_max_gain_dBi = 8
        self.antenna_config_hor_phi_3dB_deg = 65
        self.antenna_config_hor_A_m_dB = 30
        self.antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
        self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0, 0, 0])
      
        self.antenna_config_ver_theta_3dB_deg = 65
        self.antenna_config_ver_SLA_dB = 30
        self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
        self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 12 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array 
      
        self.antenna_config_gamma_mec_slant_deg = 0
        self.antenna_config_gamma_elec_slant_deg = 0
          
        #TR 38.901
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 8 # number of rows (elements in vertical)
        self.antenna_config_N = 1 # number of columns (elements in horizontal)
        self.antenna_config_P = "single" # (“single” or “dual”)    
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_dgh_m = 0 # 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dgv_m = 0 # 0.5 * self.dl_carrier_wavelength_m       
        self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
        
        # SSB precoding 
        self.SSB_precoder = "3GPPTR38_901_P1_single_column" 
        self.SSB_number_of_beams_V =  1
        self.SSB_number_of_beams_H =  1        
        self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
        
        # CSI-RS precoding
        self.CSI_RS_precoder = "3GPPTR38_901_P1_single_column" 
        self.CSI_RS_number_of_beams_V =  1
        self.CSI_RS_number_of_beams_H =  1        
        self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V
     
        ### Construct UE enginering parameters       
        return self.construct_data_frame()         


    def construct_scenario_3GPPTR38_901_UMa_C_band_large_scale(self, site_deployment_obj):
      
        self.number_of_sectors_per_site = 3
        self.bs_antenna_height_m = 25
        self.indoor = 0 
        
        # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
            # Note that not in all scenarios reference site locations equals to cell site locations
        self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

       
        # Calculate number of cell sites and cells
        self.number_of_cell_sites = len(self.cell_site_positions_m)
        self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
        # Derive cell locations 
            # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
        self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
        self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
        
        # Set velocity 
        self.velocity_kmh = 0
        self.calculate_bs_velocity_vectors()        
        
        # Radio access technology
        self.RAT = "NR"
       
        # Type of rado unit and bbu 
        self.radio_unit_type = "5G-Macro-AAU-v1"
        self.bbu_type = "5G-Macro-BBU-v1"           
       
        # Set relevant parameters
        self.fdd_tdd_ind	= "TDD"
        self.subframe_assignment = "SA2"
        self.FR = 1
        self.subcarrier_per_PRB =  12
        self.ofdm_symbols_in_slot = 14
        
        self.dl_freq_band = -1 #41
        self.dl_earfcn = -1 #40936
        self.ul_freq_band = -1 #41
        self.ul_earfcn = -1 #40936
        if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
            self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
        else:
            self.dl_carrier_frequency_GHz = 3.5
            self.ul_carrier_frequency_GHz = 3.5
      
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.dl_bandwidth_MHz = 100   
        self.dl_subcarrier_spacing_kHz = 15 * 100/10
        self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(10, 15)
        self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(15)
        self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(15)
        
        self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
        self.ul_bandwidth_MHz = 100
        if (self.fdd_tdd_ind	== "TDD"):
            if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
                np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
                exit(0) 
        self.ul_subcarrier_spacing_kHz = 15 * 100/10
        self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(10, 15)
        self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(15)
        self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(15)
        
        self.BS_propagation_model = "3GPPTR38_901_UMa"
        self.BS_fast_channel_model = "Rician"        
      
        self.BS_tx_power_dBm = 55
        self.BS_noise_figure_dB = 5 #Taken from ITU UMa 
        
        self.TX_number = 32 
        self.RX_number = 32           
        
        self.CRS_ports = 32
        self.CRS_RE_tx_power_dBm = -1
        self.CRS_power_boosting_Pb = -1      
    
        self.antenna_pattern_model = "3GPPTR38_901" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
        
        #Element in TR 38.901
        self.antenna_config_max_gain_dBi = 8
        self.antenna_config_hor_phi_3dB_deg = 65
        self.antenna_config_hor_A_m_dB = 30
        self.antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
        self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0, 0, 0])
      
        self.antenna_config_ver_theta_3dB_deg = 65
        self.antenna_config_ver_SLA_dB = 30
        self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
        self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 12 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array 
      
        self.antenna_config_gamma_mec_slant_deg = 0
        self.antenna_config_gamma_elec_slant_deg = 0
          
        #TR 38.901
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 4 # number of rows (elements in vertical)
        self.antenna_config_N = 8 # number of columns (elements in horizontal)
        self.antenna_config_P = "single" # (“single” or “dual”)    
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_dgh_m = 0 # 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dgv_m = 0 # 0.5 * self.dl_carrier_wavelength_m       
        self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
        
        # SSB precoding 
        self.SSB_precoder = "DFT" #"3GPP_single_downtilted_beam"
        self.SSB_number_of_beams_V =  2
        self.SSB_number_of_beams_H =  4        
        self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
        
        # CSI-RS precoding
        self.CSI_RS_precoder = "DFT" #"3GPP_single_downtilted_beam"
        self.CSI_RS_number_of_beams_V =  4
        self.CSI_RS_number_of_beams_H =  8        
        self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V
     
        ### Construct UE enginering parameters       
        return self.construct_data_frame() 
    
 
    def construct_scenario_3GPPTR38_901_UMi_C1(self, site_deployment_obj):
       
       self.number_of_sectors_per_site = 3
       self.bs_antenna_height_m = 10
       self.indoor = 0  
        
       # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
           # Note that not in all scenarios reference site locations equals to cell site locations
       self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

       
       # Calculate number of cell sites and cells
       self.number_of_cell_sites = len(self.cell_site_positions_m)
       self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
       # Derive cell locations 
           # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
       self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
       self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
       
       # Set velocity 
       self.velocity_kmh = 0
       self.calculate_bs_velocity_vectors()       
        
       # Radio access technology
       self.RAT = "NR"
       
       # Type of rado unit and bbu
       self.radio_unit_type = "5G-Micro-AAU-v1"
       self.bbu_type = "5G-Micro-BBU-v1"
       
       # Set relevant parameters
       self.fdd_tdd_ind	= "TDD"
       self.subframe_assignment = "SA2"
       self.FR = 1
       self.subcarrier_per_PRB =  12
       self.ofdm_symbols_in_slot = 14
       
       self.dl_freq_band = -1 #41
       self.dl_earfcn = -1 #40936
       self.ul_freq_band = -1 #41
       self.ul_earfcn = -1 #40936
       if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
           self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
       else:
           self.dl_carrier_frequency_GHz = 6
           self.ul_carrier_frequency_GHz = 6

       self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
       self.dl_bandwidth_MHz = 20   
       self.dl_subcarrier_spacing_kHz = 15
       self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz)
       self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.dl_subcarrier_spacing_kHz)
       self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(self.dl_subcarrier_spacing_kHz)
       
       self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
       self.ul_bandwidth_MHz = 20
       if (self.fdd_tdd_ind	== "TDD"):
           if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
               np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
               exit(0) 
       self.ul_subcarrier_spacing_kHz =  15
       self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
       self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
       self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
       
       self.BS_propagation_model = "3GPPTR38_901_UMi"
       self.BS_fast_channel_model = "3GPPTR38_901_UMi" #"3GPPTR38_901_UMa" #"Rician"

       self.BS_tx_power_dBm = 44
       self.BS_noise_figure_dB = 5 #Taken from ITU UMa 
       
       self.TX_number = 32 
       self.RX_number = 32          
       
       self.CRS_ports = 2
       self.CRS_RE_tx_power_dBm = -1
       self.CRS_power_boosting_Pb = -1      
       
       self.antenna_pattern_model = "3GPPTR38_901" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
       self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)
       
       #Element in TR 38.901
       self.antenna_config_max_gain_dBi = 8
       self.antenna_config_hor_phi_3dB_deg = 65
       self.antenna_config_hor_A_m_dB = 30
       self.antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
       self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0, 0, 0]) 
       
       self.antenna_config_ver_theta_3dB_deg = 65
       self.antenna_config_ver_SLA_dB = 30
       self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
       self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 20 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array  
       
       self.antenna_config_gamma_mec_slant_deg = 0
       self.antenna_config_gamma_elec_slant_deg = 0
         
       #TR 38.901
       self.antenna_config_Mg = 1
       self.antenna_config_Ng = 2
       self.antenna_config_M = 4 # number of rows (elements in vertical)
       self.antenna_config_N = 4 # number of columns (elements in horizontal)
       self.antenna_config_P = "single" # (“single” or “dual”)      
       
       if self.antenna_config_P == 'single': 
           P_num = 1
       else :
           P_num = 2
       self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
       
       self.antenna_config_dgh_m = 2.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dgv_m = 2.5 * self.dl_carrier_wavelength_m       
       self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
       
       # SSB precoding 
       self.SSB_precoder = "None" 
       self.SSB_number_of_beams_H =  1
       self.SSB_number_of_beams_V =  2
       self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
      
       # CSI-RS precoding
       self.CSI_RS_precoder = "None"
       self.CSI_RS_number_of_beams_H =  1
       self.CSI_RS_number_of_beams_V =  2
       self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V
    
       ### Construct UE enginering parameters
       return self.construct_data_frame() 
   

    def construct_scenario_3GPPTR38_901_UMi_C2(self, site_deployment_obj):
       
       self.number_of_sectors_per_site = 3
       self.bs_antenna_height_m = 10
       self.indoor = 0  
        
       # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
           # Note that not in all scenarios reference site locations equals to cell site locations
       self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

       # Calculate number of cell sites and cells
       self.number_of_cell_sites = len(self.cell_site_positions_m)
       self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
       # Derive cell locations 
           # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
       self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
       self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
       
       # Set velocity 
       self.velocity_kmh = 0
       self.calculate_bs_velocity_vectors()          
        
       # Radio access technology
       self.RAT = "NR"
       
       # Type of rado unit and bbu
       self.radio_unit_type = "5G-Micro-AAU-v1"
       self.bbu_type = "5G-Micro-BBU-v1"   

       # Set relevant parameters
       self.fdd_tdd_ind	= "TDD"
       self.subframe_assignment = "SA2"
       self.FR = 1
       self.subcarrier_per_PRB =  12
       self.ofdm_symbols_in_slot = 14
       
       self.dl_freq_band = -1 #41
       self.dl_earfcn = -1 #40936
       self.ul_freq_band = -1 #41
       self.ul_earfcn = -1 #40936
       if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
           self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
       else:
           self.dl_carrier_frequency_GHz = 6
           self.ul_carrier_frequency_GHz = 6

       self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
       self.dl_bandwidth_MHz = 20   
       self.dl_subcarrier_spacing_kHz = 15
       self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz)
       self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.dl_subcarrier_spacing_kHz)
       self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(self.dl_subcarrier_spacing_kHz)
       
       self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
       self.ul_bandwidth_MHz = 20
       if (self.fdd_tdd_ind	== "TDD"):
           if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
               np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
               exit(0) 
       self.ul_subcarrier_spacing_kHz =  15
       self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
       self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
       self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
       
       self.BS_propagation_model = "3GPPTR38_901_UMi"
       self.BS_fast_channel_model = "3GPPTR38_901_UMi" #"3GPPTR38_901_UMa" #"Rician"

       self.BS_tx_power_dBm = 44
       self.BS_noise_figure_dB = 5 #Taken from ITU UMa 
       
       self.TX_number = 4 
       self.RX_number = 4          
       
       self.CRS_ports = 4
       self.CRS_RE_tx_power_dBm = -1
       self.CRS_power_boosting_Pb = -1       
       
       self.antenna_pattern_model = "3GPPTR38_901" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
       self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)
       
       #Element in TR 38.901
       self.antenna_config_max_gain_dBi = 8
       self.antenna_config_hor_phi_3dB_deg = 65
       self.antenna_config_hor_A_m_dB = 30
       self.antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
       self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0, 0, 0]) 
       
       self.antenna_config_ver_theta_3dB_deg = 65
       self.antenna_config_ver_SLA_dB = 30
       self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
       self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 20 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array  
       
       self.antenna_config_gamma_mec_slant_deg = 0
       self.antenna_config_gamma_elec_slant_deg = 0
         
       #TR 38.901
       self.antenna_config_Mg = 1
       self.antenna_config_Ng = 1
       self.antenna_config_M = 2 # number of rows (elements in vertical)
       self.antenna_config_N = 2 # number of columns (elements in horizontal)
       self.antenna_config_P = "single" # (“single” or “dual”)      
       
       if self.antenna_config_P == 'single': 
           P_num = 1
       else :
           P_num = 2
       self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
       
       self.antenna_config_dgh_m = 0 #0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dgv_m = 0 #0.5 * self.dl_carrier_wavelength_m       
       self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
       
       # SSB precoding 
       self.SSB_precoder = "None" 
       self.SSB_number_of_beams_H =  2
       self.SSB_number_of_beams_V =  2
       self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
      
       # CSI-RS precoding
       self.CSI_RS_precoder = "None"
       self.CSI_RS_number_of_beams_H =  2
       self.CSI_RS_number_of_beams_V =  2
       self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V
    
       ### Construct UE enginering parameters
       return self.construct_data_frame() 
    

    def construct_scenario_3GPPTR38_901_UMi_large_scale_calibration(self, site_deployment_obj):
      
        self.number_of_sectors_per_site = 3
        self.bs_antenna_height_m = 10
        self.indoor = 0  
        
        # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
            # Note that not in all scenarios reference site locations equals to cell site locations
        self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

        # Calculate number of cell sites and cells
        self.number_of_cell_sites = len(self.cell_site_positions_m)
        self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
        # Derive cell locations 
            # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
        self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
        self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
        
        # Set velocity 
        self.velocity_kmh = 0
        self.calculate_bs_velocity_vectors()           
        
        # Radio access technology
        self.RAT = "NR"
       
        # Type of rado unit and bbu
        self.radio_unit_type = "5G-Micro-AAU-v1"
        self.bbu_type = "5G-Micro-BBU-v1"        
       
        # Set relevant parameters
        self.fdd_tdd_ind	= "TDD"
        self.subframe_assignment = "SA2"
        self.FR = 1
        self.subcarrier_per_PRB =  12
        self.ofdm_symbols_in_slot = 14
        
        self.dl_freq_band = -1 #41
        self.dl_earfcn = -1 #40936
        self.ul_freq_band = -1 #41
        self.ul_earfcn = -1 #40936
        if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
            self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
        else:
            self.dl_carrier_frequency_GHz = 6
            self.ul_carrier_frequency_GHz = 6
      
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.dl_bandwidth_MHz = 20   
        self.dl_subcarrier_spacing_kHz = 15
        self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz)
        self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.dl_subcarrier_spacing_kHz)
        self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(self.dl_subcarrier_spacing_kHz)
        
        self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
        self.ul_bandwidth_MHz = 20
        if (self.fdd_tdd_ind	== "TDD"):
            if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
                np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
                exit(0) 
        self.ul_subcarrier_spacing_kHz =  15
        self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
        self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
        self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
        
        self.BS_propagation_model = "3GPPTR38_901_UMi"
        self.BS_fast_channel_model = "Rician"        
      
        self.BS_tx_power_dBm = 44
        self.BS_noise_figure_dB = 5 #Taken from ITU UMa 
        
        self.TX_number = 10 
        self.RX_number = 10           
        
        self.CRS_ports = 1
        self.CRS_RE_tx_power_dBm = -1
        self.CRS_power_boosting_Pb = -1       
        
        self.antenna_pattern_model = "3GPPTR38_901" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
        
        #Element in TR 38.901
        self.antenna_config_max_gain_dBi = 8
        self.antenna_config_hor_phi_3dB_deg = 65
        self.antenna_config_hor_A_m_dB = 30
        self.antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
        self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0, 0, 0])
      
        self.antenna_config_ver_theta_3dB_deg = 65
        self.antenna_config_ver_SLA_dB = 30
        self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
        self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 12 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array 
      
        self.antenna_config_gamma_mec_slant_deg = 0
        self.antenna_config_gamma_elec_slant_deg = 0
          
        #TR 38.901
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 10 # number of rows (elements in vertical)
        self.antenna_config_N = 1 # number of columns (elements in horizontal)
        self.antenna_config_P = "single" # (“single” or “dual”)    
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_dgh_m = 0 # 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dgv_m = 0 # 0.5 * self.dl_carrier_wavelength_m       
        self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
        
        # SSB precoding 
        self.SSB_precoder = "3GPPTR38_901_P1_single_column" #"3GPP_single_downtilted_beam"
        self.SSB_number_of_beams_V =  1
        self.SSB_number_of_beams_H =  1        
        self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
        
        # CSI-RS precoding
        self.CSI_RS_precoder = "3GPPTR38_901_P1_single_column" #"3GPP_single_downtilted_beam"
        self.CSI_RS_number_of_beams_V =  1
        self.CSI_RS_number_of_beams_H =  1        
        self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V
     
        ### Construct UE enginering parameters       
        return self.construct_data_frame() 


    def construct_scenario_3GPPTR38_901_UMi_C_band_large_scale(self, site_deployment_obj):
      
        self.number_of_sectors_per_site = 3
        self.bs_antenna_height_m = 10
        self.indoor = 0  
        
        # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
            # Note that not in all scenarios reference site locations equals to cell site locations
        self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

        # Calculate number of cell sites and cells
        self.number_of_cell_sites = len(self.cell_site_positions_m)
        self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
        # Derive cell locations 
            # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
        self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
        self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
        
        # Set velocity 
        self.velocity_kmh = 0
        self.calculate_bs_velocity_vectors()           
        
        # Radio access technology
        self.RAT = "NR"
       
        # Type of rado unit and bbu
        self.radio_unit_type = "5G-Micro-AAU-v1"
        self.bbu_type = "5G-Micro-BBU-v1"           
       
        # Set relevant parameters
        self.fdd_tdd_ind	= "TDD"
        self.subframe_assignment = "SA2"
        self.FR = 1
        self.subcarrier_per_PRB =  12
        self.ofdm_symbols_in_slot = 14
        
        self.dl_freq_band = -1 #41
        self.dl_earfcn = -1 #40936
        self.ul_freq_band = -1 #41
        self.ul_earfcn = -1 #40936
        if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
            self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
        else:
            self.dl_carrier_frequency_GHz = 3.5
            self.ul_carrier_frequency_GHz = 3.5
      
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.dl_bandwidth_MHz = 100   
        self.dl_subcarrier_spacing_kHz = 15 * 100/10
        self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(10, 15)
        self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(15)
        self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(15)
        
        self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
        self.ul_bandwidth_MHz = 100
        if (self.fdd_tdd_ind	== "TDD"):
            if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
                np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
                exit(0) 
        self.ul_subcarrier_spacing_kHz = 15 * 100/10
        self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(10, 15)
        self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(15)
        self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(15)
        
        self.BS_propagation_model = "3GPPTR38_901_UMi"
        self.BS_fast_channel_model = "Rician"        
      
        self.BS_tx_power_dBm = 50
        self.BS_noise_figure_dB = 5 #Taken from ITU UMa 
        
        self.TX_number = 32 
        self.RX_number = 32           
        
        self.CRS_ports = 32
        self.CRS_RE_tx_power_dBm = -1
        self.CRS_power_boosting_Pb = -1      
    
        self.antenna_pattern_model = "3GPPTR38_901" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
        
        #Element in TR 38.901
        self.antenna_config_max_gain_dBi = 8
        self.antenna_config_hor_phi_3dB_deg = 65
        self.antenna_config_hor_A_m_dB = 30
        self.antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
        self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0, 0, 0])
      
        self.antenna_config_ver_theta_3dB_deg = 65
        self.antenna_config_ver_SLA_dB = 30
        self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
        self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 12 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array 
      
        self.antenna_config_gamma_mec_slant_deg = 0
        self.antenna_config_gamma_elec_slant_deg = 0
          
        #TR 38.901
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 4 # number of rows (elements in vertical)
        self.antenna_config_N = 8 # number of columns (elements in horizontal)
        self.antenna_config_P = "single" # (“single” or “dual”)    
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_dgh_m = 0 # 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dgv_m = 0 # 0.5 * self.dl_carrier_wavelength_m       
        self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
        
        # SSB precoding 
        self.SSB_precoder = "DFT" #"3GPP_single_downtilted_beam"
        self.SSB_number_of_beams_V =  2
        self.SSB_number_of_beams_H =  4        
        self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
        
        # CSI-RS precoding
        self.CSI_RS_precoder = "DFT" #"3GPP_single_downtilted_beam"
        self.CSI_RS_number_of_beams_V =  4
        self.CSI_RS_number_of_beams_H =  8        
        self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V
     
        ### Construct UE enginering parameters       
        return self.construct_data_frame() 


    def construct_scenario_3GPPTR38_901_UMi_fr3_large_scale(self, site_deployment_obj):
      
        self.number_of_sectors_per_site = 3
        self.bs_antenna_height_m = 10
        self.indoor = 0  
        
        # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
            # Note that not in all scenarios reference site locations equals to cell site locations
        self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 
       
        # Calculate number of cell sites and cells
        self.number_of_cell_sites = len(self.cell_site_positions_m)
        self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
        # Derive cell locations 
            # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
        self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
        self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
        
        # Set velocity 
        self.velocity_kmh = 0
        self.calculate_bs_velocity_vectors()           
        
        # Radio access technology
        self.RAT = "6G"
       
        # Type of rado unit and bbu
        self.radio_unit_type = "6G-Micro-AAU-v1"
        self.bbu_type = "6G-Micro-BBU-v1"           
       
        # Set relevant parameters
        self.fdd_tdd_ind	= "TDD"
        self.subframe_assignment = "SA2"
        self.FR = 3
        self.subcarrier_per_PRB =  12
        self.ofdm_symbols_in_slot = 14
        
        self.dl_freq_band = -1 #41
        self.dl_earfcn = -1 #40936
        self.ul_freq_band = -1 #41
        self.ul_earfcn = -1 #40936
        if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
            self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
        else:
            self.dl_carrier_frequency_GHz = 10
            self.ul_carrier_frequency_GHz = 10
      
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.dl_bandwidth_MHz = 200   
        self.dl_subcarrier_spacing_kHz = 15 * 200/10
        self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(10, 15)
        self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(15)
        self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(15)
        
        self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
        self.ul_bandwidth_MHz = 200
        if (self.fdd_tdd_ind	== "TDD"):
            if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
                np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
                exit(0) 
        self.ul_subcarrier_spacing_kHz = 15 * 200/10
        self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(10, 15)
        self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(15)
        self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(15)
        
        self.BS_propagation_model = "3GPPTR38_901_UMi"
        self.BS_fast_channel_model = "Rician"        
      
        self.BS_tx_power_dBm = 50
        self.BS_noise_figure_dB = 5 #Taken from ITU UMa 
        
        self.TX_number = 64 
        self.RX_number = 64           
        
        self.CRS_ports = 64
        self.CRS_RE_tx_power_dBm = -1
        self.CRS_power_boosting_Pb = -1       
        
        self.antenna_pattern_model = "3GPPTR38_901" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
        
        #Element in TR 38.901
        self.antenna_config_max_gain_dBi = 8
        self.antenna_config_hor_phi_3dB_deg = 65
        self.antenna_config_hor_A_m_dB = 30
        self.antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
        self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0, 0, 0])
      
        self.antenna_config_ver_theta_3dB_deg = 65
        self.antenna_config_ver_SLA_dB = 30
        self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
        self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 12 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array 
      
        self.antenna_config_gamma_mec_slant_deg = 0
        self.antenna_config_gamma_elec_slant_deg = 0
          
        #TR 38.901
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 4 # number of rows (elements in vertical)
        self.antenna_config_N = 16 # number of columns (elements in horizontal)
        self.antenna_config_P = "single" # (“single” or “dual”)    
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_dgh_m = 0 # 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dgv_m = 0 # 0.5 * self.dl_carrier_wavelength_m       
        self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
        
        # SSB precoding 
        self.SSB_precoder = "DFT" #"3GPP_single_downtilted_beam"
        self.SSB_number_of_beams_V =  2
        self.SSB_number_of_beams_H =  8        
        self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
        
        # CSI-RS precoding
        self.CSI_RS_precoder = "DFT" #"3GPP_single_downtilted_beam"
        self.CSI_RS_number_of_beams_V =  4
        self.CSI_RS_number_of_beams_H =  16        
        self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V
     
        ### Construct UE enginering parameters       
        return self.construct_data_frame()     
     

    def construct_scenario_3GPPTR38_901_UPi_fr3_large_scale(self, site_deployment_obj):
      
        self.number_of_sectors_per_site = 3
        self.bs_antenna_height_m = 10
        self.indoor = 0  
        
        # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
            # Note that not in all scenarios reference site locations equals to cell site locations
        if hasattr(site_deployment_obj, 'hotspot_position_m'):
            self.cell_site_positions_m = site_deployment_obj.hotspot_position_m
        else: 
            self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

       
        # Calculate number of cell sites and cells
        self.number_of_cell_sites = len(self.cell_site_positions_m)
        self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
        # Derive cell locations 
            # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
        self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
        self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
        
        # Set velocity 
        self.velocity_kmh = 0
        self.calculate_bs_velocity_vectors()          
        
        # Radio access technology
        self.RAT = "6G"
       
        # Type of rado unit and bbu
        self.radio_unit_type = "6G-Picro-AAU-v1"
        self.bbu_type = "6G-Picro-BBU-v1"           
       
        # Set relevant parameters
        self.fdd_tdd_ind	= "TDD"
        self.subframe_assignment = "SA2"
        self.FR = 3
        self.subcarrier_per_PRB =  12
        self.ofdm_symbols_in_slot = 14
        
        self.dl_freq_band = -1 #41
        self.dl_earfcn = -1 #40936
        self.ul_freq_band = -1 #41
        self.ul_earfcn = -1 #40936
        if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
            self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
        else:
            self.dl_carrier_frequency_GHz = 10
            self.ul_carrier_frequency_GHz = 10
      
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.dl_bandwidth_MHz = 200   
        self.dl_subcarrier_spacing_kHz = 15 * 200/10
        self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(10, 15)
        self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(15)
        self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(15)
        
        self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
        self.ul_bandwidth_MHz = 200
        if (self.fdd_tdd_ind	== "TDD"):
            if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
                np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
                exit(0) 
        self.ul_subcarrier_spacing_kHz = 15 * 200/10
        self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(10, 15)
        self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(15)
        self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(15)
        
        self.BS_propagation_model = "3GPPTR38_901_UMi"
        self.BS_fast_channel_model = "Rician"        
      
        self.BS_tx_power_dBm = 50
        self.BS_noise_figure_dB = 5 #Taken from ITU UMa 
        
        self.TX_number = 64 
        self.RX_number = 64           
        
        self.CRS_ports = 64
        self.CRS_RE_tx_power_dBm = -1
        self.CRS_power_boosting_Pb = -1       
        
        self.antenna_pattern_model = "3GPPTR38_901" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
        
        #Element in TR 38.901
        self.antenna_config_max_gain_dBi = 8
        self.antenna_config_hor_phi_3dB_deg = 65
        self.antenna_config_hor_A_m_dB = 30
        self.antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
        self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0, 0, 0])
      
        self.antenna_config_ver_theta_3dB_deg = 65
        self.antenna_config_ver_SLA_dB = 30
        self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
        self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 12 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array 
      
        self.antenna_config_gamma_mec_slant_deg = 0
        self.antenna_config_gamma_elec_slant_deg = 0
          
        #TR 38.901
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 4 # number of rows (elements in vertical)
        self.antenna_config_N = 32 # number of columns (elements in horizontal)
        self.antenna_config_P = "single" # (“single” or “dual”)    
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_dgh_m = 0 # 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dgv_m = 0 # 0.5 * self.dl_carrier_wavelength_m       
        self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
        
        # SSB precoding 
        self.SSB_precoder = "DFT" #"3GPP_single_downtilted_beam"
        self.SSB_number_of_beams_V =  4
        self.SSB_number_of_beams_H =  8        
        self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
        
        # CSI-RS precoding
        self.CSI_RS_precoder = "DFT" #"3GPP_single_downtilted_beam"
        self.CSI_RS_number_of_beams_V =  4
        self.CSI_RS_number_of_beams_H =  32        
        self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V
     
        ### Construct UE enginering parameters       
        return self.construct_data_frame()  
    

    def construct_scenario_3GPPTR38_811_Dense_Urban_HAPS_ULA(self, site_deployment_obj):
      
        self.number_of_sectors_per_site = 1
        self.bs_antenna_height_m = 20000
        self.indoor = 0  
        
        # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
            # Note that not in all scenarios reference site locations equals to cell site locations
        self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

        # Calculate number of cell sites and cells
        self.number_of_cell_sites = len(self.cell_site_positions_m)
        self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
        # Derive cell locations 
            # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
        self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
        self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
        
        # Set velocity 
        self.velocity_kmh = 0
        self.calculate_bs_velocity_vectors()          
        
        # Radio access technology
        self.RAT = "NR"
       
        # Type of rado unit and bbu
        self.radio_unit_type = "5G-Macro-AAU-v1"
        self.bbu_type = "5G-Macro-BBU-v1"          
       
        # Set relevant parameters
        self.fdd_tdd_ind	= "TDD"
        self.subframe_assignment = "SA2"
        self.FR = 1
        self.subcarrier_per_PRB =  12
        self.ofdm_symbols_in_slot = 14
        
        self.dl_freq_band = -1 #41
        self.dl_earfcn = -1 #40936
        self.ul_freq_band = -1 #41
        self.ul_earfcn = -1 #40936
        if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
            self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
        else:
            self.dl_carrier_frequency_GHz = 3.5
            self.ul_carrier_frequency_GHz = 3.5
      
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.dl_bandwidth_MHz = 100   
        self.dl_subcarrier_spacing_kHz = 15 * 100/20
        self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(20, 15)
        self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(15)
        self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(15)
        
        self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
        self.ul_bandwidth_MHz = 100
        if (self.fdd_tdd_ind	== "TDD"):
            if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
                np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
                exit(0) 
        self.ul_subcarrier_spacing_kHz = 15 * 100/20
        self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(20, 15)
        self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(15)
        self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(15)
        
        self.BS_propagation_model = "3GPPTR38_811_Dense_Urban_NTN"
        self.BS_fast_channel_model = "Rician"        
      
        self.BS_tx_power_dBm = 55
        self.BS_noise_figure_dB = 5 #Taken from ITU UMa 

        self.TX_number = 10 
        self.RX_number = 10        
        
        self.CRS_ports = 1
        self.CRS_RE_tx_power_dBm = -1
        self.CRS_power_boosting_Pb = -1       
        
        self.antenna_pattern_model = "3GPPTR36_814_UMa" #"omnidirectional" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
       
        #Array in TR 36.814
        self.antenna_config_max_gain_dBi = 17
        self.antenna_config_hor_phi_3dB_deg = 70
        self.antenna_config_hor_A_m_dB = 20
        self.antenna_config_hor_alpha_mec_bearing_deg = np.array([0])
        self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0])

        self.antenna_config_ver_theta_3dB_deg = 15
        self.antenna_config_ver_SLA_dB = 20
        self.antenna_config_ver_beta_mec_downtilt_deg = 180 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon      
        self.antenna_config_ver_beta_elec_downtilt_deg = 90 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array 

        self.antenna_config_gamma_mec_slant_deg = 0
        self.antenna_config_gamma_elec_slant_deg = 0

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
       
        self.antenna_config_dgh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dgv_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
       
        # SSB precoding 
        self.SSB_precoder = "None" 
        self.SSB_number_of_beams_H =  1
        self.SSB_number_of_beams_V =  1
        self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
      
        # CSI-RS precoding
        self.CSI_RS_precoder = "None"
        self.CSI_RS_number_of_beams_H =  1
        self.CSI_RS_number_of_beams_V =  1
        self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V  
     
        ### Construct UE enginering parameters       
        return self.construct_data_frame()    


    def construct_scenario_3GPPTR38_811_Dense_Urban_HAPS_UPA(self, site_deployment_obj):
        
        self.number_of_sectors_per_site = 1
        self.bs_antenna_height_m = 1000
        self.indoor = 0  
        
        # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
            # Note that not in all scenarios reference site locations equals to cell site locations
        self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

        # Calculate number of cell sites and cells
        self.number_of_cell_sites = len(self.cell_site_positions_m)
        self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
        # Derive cell locations 
            # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
        self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
        self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
        
        # Set velocity 
        self.velocity_kmh = 0
        self.calculate_bs_velocity_vectors()          
        
        # Radio access technology
        self.RAT = "NR"
       
        # Type of rado unit and bbu
        self.radio_unit_type = "5G-Macro-AAU-v1"
        self.bbu_type = "5G-Macro-BBU-v1"          
       
        # Set relevant parameters
        self.fdd_tdd_ind	= "TDD"
        self.subframe_assignment = "SA2"
        self.FR = 1
        self.subcarrier_per_PRB =  12
        self.ofdm_symbols_in_slot = 14
        
        self.dl_freq_band = -1 #41
        self.dl_earfcn = -1 #40936
        self.ul_freq_band = -1 #41
        self.ul_earfcn = -1 #40936
        if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
            self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
        else:
            self.dl_carrier_frequency_GHz = 3.5
            self.ul_carrier_frequency_GHz = 3.5
      
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.dl_bandwidth_MHz = 100   
        self.dl_subcarrier_spacing_kHz = 15 * 100/20
        self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(20, 15)
        self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(15)
        self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(15)
        
        self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
        self.ul_bandwidth_MHz = 100
        if (self.fdd_tdd_ind	== "TDD"):
            if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
                np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
                exit(0) 
        self.ul_subcarrier_spacing_kHz = 15 * 100/20
        self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(20, 15)
        self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(15)
        self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(15)
        
        self.BS_propagation_model = "3GPPTR38_811_Dense_Urban_NTN"
        self.BS_fast_channel_model = "Rician"        
      
        self.BS_tx_power_dBm = 55
        self.BS_noise_figure_dB = 5 #Taken from ITU UMa 

        self.TX_number = 32 
        self.RX_number = 32        
        
        self.CRS_ports = 1
        self.CRS_RE_tx_power_dBm = -1
        self.CRS_power_boosting_Pb = -1       
        
        
        self.antenna_pattern_model = "3GPPTR38_901" #"omnidirectional" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
       
        # Element in TR 38.901
        self.antenna_config_max_gain_dBi = 8
        self.antenna_config_hor_phi_3dB_deg = 65
        self.antenna_config_hor_A_m_dB = 30
        self.antenna_config_hor_alpha_mec_bearing_deg = np.array([0])
        self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0])

        self.antenna_config_ver_theta_3dB_deg = 65
        self.antenna_config_ver_SLA_dB = 30
        self.antenna_config_ver_beta_mec_downtilt_deg = 180 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon      
        self.antenna_config_ver_beta_elec_downtilt_deg = 90 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array 

        self.antenna_config_gamma_mec_slant_deg = 0
        self.antenna_config_gamma_elec_slant_deg = 0

        #TR 38.901
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 4 # number of rows (elements in vertical)
        self.antenna_config_N = 8 # number of columns (elements in horizontal)
        self.antenna_config_P = "single" # (“single” or “dual”)    
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
       
        self.antenna_config_dgh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dgv_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
       
        # SSB precoding 
        self.SSB_precoder = "custom_vMB" 
        self.SSB_number_of_beams_H =  1
        self.SSB_number_of_beams_V =  1
        self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
        
        # SSB Direction
        # Note that the direction of the beam are computed according to the LCS of the base panel
        self.SSB_BeamDirection_phi_deg = str(list(np.array([0.00])))
        self.SSB_BeamDirection_theta_deg = str(list(np.full(self.SSB_number_of_beams, self.antenna_config_ver_beta_elec_downtilt_deg-90, dtype=float)))
        

        # CSI-RS precoding
        self.CSI_RS_precoder = "custom_vMB"
        self.CSI_RS_number_of_beams_H =  1
        self.CSI_RS_number_of_beams_V =  1
        self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V  
        
        # CSI-RS Direction
        # Note that the direction of the beam are computed according to the LCS of the base panel
        self.CSI_RS_BeamDirection_phi_deg = str(list(np.array([0.00])))
        self.CSI_RS_BeamDirection_theta_deg = str(list(np.full(self.CSI_RS_number_of_beams, self.antenna_config_ver_beta_elec_downtilt_deg-90, dtype=float)))
        
        ### Construct UE enginering parameters       
        return self.construct_data_frame()

  
    def construct_scenario_3GPPTR38_811_Dense_Urban_HAPS_Reflector(self, site_deployment_obj):
        self.number_of_sectors_per_site = 1
        self.bs_antenna_height_m = 1000
        self.indoor = 0 
        
        # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
            # Note that not in all scenarios reference site locations equals to cell site locations
        self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 
        
        # Calculate number of cell sites and cells
        self.number_of_cell_sites = len(self.cell_site_positions_m)
        self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
        
        # Derive cell locations 
            # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
        self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
        self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
        
        # Set velocity 
        self.velocity_kmh = 0
        self.calculate_bs_velocity_vectors()          
        
        # Radio access technology
        self.RAT = "NR"
       
        # Type of rado unit and bbu
        self.radio_unit_type = "5G-Macro-AAU-v1"
        self.bbu_type = "5G-Macro-BBU-v1"          
       
        # Set relevant parameters
        self.fdd_tdd_ind	= "TDD"
        self.subframe_assignment = "SA2"
        self.FR = 1
        self.subcarrier_per_PRB =  12
        self.ofdm_symbols_in_slot = 14
        
        self.dl_freq_band = -1 #41
        self.dl_earfcn = -1 #40936
        self.ul_freq_band = -1 #41
        self.ul_earfcn = -1 #40936
        if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
            self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
        else:
            self.dl_carrier_frequency_GHz = 3.5
            self.ul_carrier_frequency_GHz = 3.5
      
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.dl_bandwidth_MHz = 100   
        self.dl_subcarrier_spacing_kHz = 15 * 100/20
        self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(20, 15)
        self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(15)
        self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(15)
        
        self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
        self.ul_bandwidth_MHz = 100
        if (self.fdd_tdd_ind	== "TDD"):
            if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
                np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
                exit(0) 
        self.ul_subcarrier_spacing_kHz = 15 * 100/20
        self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(20, 15)
        self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(15)
        self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(15)
        
        self.BS_propagation_model = "3GPPTR38_811_Dense_Urban_NTN"
        self.BS_fast_channel_model = "Rician"        
      
        self.BS_tx_power_dBm = 55
        self.BS_noise_figure_dB = 5 #Taken from ITU UMa 

        self.TX_number = 1 
        self.RX_number = 1        
        
        self.CRS_ports = 1
        self.CRS_RE_tx_power_dBm = -1
        self.CRS_power_boosting_Pb = -1     
        
        
        # Antenna patter model
        self.antenna_pattern_model = "3GPPTR38_811" # The following model defines a reflector antenna
    

        # Reflector Parameters
        ## Within 3GPPTR38_811, the driven parameter is 'a', defined as the radius of the aperture of the reflector antenna
        ## 'a' is defined as multiple of the reflector design Wavelenght
        ## Note that now the radius has been set using the carrier freq wavelenght, 
        ## but potentially can be any related to the design of the reflector.
        self.reflector_aperture_radius_multipleWavelenght = 10
        self.antenna_radius_ReflectorAperture_m = self.reflector_aperture_radius_multipleWavelenght * self.dl_carrier_wavelength_m
        # Note that the total aperture is defined as the diameter of the reflect. 
        # Often, in literature, this parameter is reffered as 'Reflector Diameter'
        self.antenna_TotalAperture_geom_m = 2*self.antenna_radius_ReflectorAperture_m

        self.reflector_aperture_efficiency = 1
        
        # Total Reflect Gain
        antenna_config_max_gain_lin = self.reflector_aperture_efficiency *\
                                      np.square((np.pi*self.antenna_TotalAperture_geom_m)/self.dl_carrier_wavelength_m).astype(float).item()  
        self.antenna_config_max_gain_dBi = 10*np.log10(antenna_config_max_gain_lin)
        del antenna_config_max_gain_lin
        
        # Antenna mechanical and electrical directions configuration
        self.antenna_config_hor_alpha_mec_bearing_deg = np.array([0])
        self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0])
        
        
        self.antenna_config_ver_beta_mec_downtilt_deg = 180 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon      
        self.antenna_config_ver_beta_elec_downtilt_deg = 90 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array 
        
        self.antenna_config_gamma_mec_slant_deg = 0
        self.antenna_config_gamma_elec_slant_deg = 0
        
        
        # Number of antennas and polarization
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 1 # number of rows (elements in vertical)
        self.antenna_config_N = 1 # number of columns (elements in horizontal)
        self.antenna_config_P = "single" # (“single” or “dual”)    
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)
        
        
        if self.antenna_config_P == 'single': P_num = 1
        else: P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
       
        
        # The following parameters have to be defined for the sake of compliance with df_ep inputs
        # And other part of the code (e.g. distance computation and LCS antenna computation)
        # Value initialized with None will break the code
        self.antenna_config_dgh_m = 0
        self.antenna_config_dgv_m = 0
        self.antenna_config_dh_m = 0
        self.antenna_config_dv_m = 0
        
        self.antenna_config_hor_phi_3dB_deg = None
        self.antenna_config_ver_theta_3dB_deg = None
        self.antenna_config_hor_A_m_dB = None
        self.antenna_config_ver_SLA_dB = None
        
        # SSB precoding 
        self.SSB_precoder = "None" 
        self.SSB_number_of_beams_H =  1
        self.SSB_number_of_beams_V =  1
        self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
      
        # CSI-RS precoding
        self.CSI_RS_precoder = "None"
        self.CSI_RS_number_of_beams_H =  1
        self.CSI_RS_number_of_beams_V =  1
        self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V  
        
        
        ### Construct UE enginering parameters       
        return self.construct_data_frame()


    def construct_scenario_3GPPTR38_811_Urban_NTN(self, site_deployment_obj):
        
        self.number_of_sectors_per_site = 1
        self.bs_antenna_height_m = 6371
        self.indoor = 0  
        
        # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
            # Note that not in all scenarios reference site locations equals to cell site locations
        self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m[:1,:]# [np.newaxis]

       
        # Calculate number of cell sites and cells
        self.number_of_cell_sites = len(self.cell_site_positions_m)
        self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
        # Derive cell locations 
            # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
        self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
        self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
        
        # Set velocity 
        self.velocity_kmh = 0
        self.calculate_bs_velocity_vectors()          
        
        # Radio access technology
        self.RAT = "NR"
       
        # Type of rado unit and bbu
        self.radio_unit_type = "5G-Macro-AAU-v1"
        self.bbu_type = "5G-Macro-BBU-v1"           
       
        # Set relevant parameters        
        self.fdd_tdd_ind	= "TDD"
        self.subframe_assignment = "SA2"
        self.FR = 1
        self.subcarrier_per_PRB =  12
        self.ofdm_symbols_in_slot = 14
        
        self.dl_freq_band = -1 #41
        self.dl_earfcn = -1 #40936
        self.ul_freq_band = -1 #41
        self.ul_earfcn = -1 #40936
        if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
            self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
        else:
            self.dl_carrier_frequency_GHz = 2
            self.ul_carrier_frequency_GHz = 2
      
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.dl_bandwidth_MHz = 5   
        self.dl_subcarrier_spacing_kHz = 15
        self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz)
        self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.dl_subcarrier_spacing_kHz)
        self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(self.dl_subcarrier_spacing_kHz)
        
        self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
        self.ul_bandwidth_MHz = 5
        if (self.fdd_tdd_ind	== "TDD"):
            if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
                np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
                exit(0) 
        self.ul_subcarrier_spacing_kHz =  15
        self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
        self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
        self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
        
        self.BS_propagation_model = "3GPPTR38_811_Urban_NTN"
        self.BS_fast_channel_model = "Rician"        
      
        self.BS_tx_power_dBm = 36 * self.dl_bandwidth_MHz + 30 # 36 dbW/MHz 
        self.BS_noise_figure_dB = 5 #Taken from ITU UMa 
        self.min_BS_to_UE_2D_distance_m = 0 
        
        self.TX_number = 10 
        self.RX_number = 10           
        
        self.CRS_ports = 1
        self.CRS_RE_tx_power_dBm = -1
        self.CRS_power_boosting_Pb = -1     
        
        self.antenna_pattern_model = "3GPPTR38_901" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
        
        #Element in TR 38.901    
        self.antenna_config_max_gain_dBi = 8
        self.antenna_config_hor_phi_3dB_deg = 65
        self.antenna_config_hor_A_m_dB = 30
        self.antenna_config_hor_alpha_mec_bearing_deg = np.array([0])
        self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0])
      
        self.antenna_config_ver_theta_3dB_deg = 65
        self.antenna_config_ver_SLA_dB = 30
        self.antenna_config_ver_beta_mec_downtilt_deg = 180 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
        self.antenna_config_ver_beta_elec_downtilt_deg = 90 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array 
      
        self.antenna_config_gamma_mec_slant_deg = 0
        self.antenna_config_gamma_elec_slant_deg = 0
          
        #TR 38.901
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 10 # number of rows (elements in vertical)
        self.antenna_config_N = 1 # number of columns (elements in horizontal)
        self.antenna_config_P = "single" # (“single” or “dual”)    
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_dgh_m = 0 # 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dgv_m = 0 # 0.5 * self.dl_carrier_wavelength_m       
        self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
        
        # SSB precoding 
        self.SSB_precoder = "3GPPTR38_901_P1_single_column" #"3GPP_single_downtilted_beam" 
        self.SSB_number_of_beams_V =  1
        self.SSB_number_of_beams_H =  1        
        self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
        
        # CSI-RS precoding
        self.CSI_RS_precoder = "3GPPTR38_901_P1_single_column" #"3GPP_single_downtilted_beam"
        self.CSI_RS_number_of_beams_V =  1
        self.CSI_RS_number_of_beams_H =  1        
        self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V
     
        ### Construct UE enginering parameters       
        return self.construct_data_frame()  
    

    def construct_scenario_3GPPTR38_811_Desne_Urban_NTN(self, site_deployment_obj):
        
        self.number_of_sectors_per_site = 1
        self.bs_antenna_height_m = 6371
        self.indoor = 0  
        
        # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
            # Note that not in all scenarios reference site locations equals to cell site locations
        self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m[:1,:]# [np.newaxis]

       
        # Calculate number of cell sites and cells
        self.number_of_cell_sites = len(self.cell_site_positions_m)
        self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
        # Derive cell locations 
            # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
        self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
        self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
        
        # Set velocity 
        self.velocity_kmh = 0
        self.calculate_bs_velocity_vectors()          
        
        # Radio access technology
        self.RAT = "NR"
       
        # Type of rado unit and bbu
        self.radio_unit_type = "5G-Micro-AAU-v1"
        self.bbu_type = "5G-Micro-BBU-v1"           
       
       # Set relevant parameters        
        self.fdd_tdd_ind	= "TDD"
        self.subframe_assignment = "SA2"
        self.FR = 1
        self.subcarrier_per_PRB =  12
        self.ofdm_symbols_in_slot = 14
        
        self.dl_freq_band = -1 #41
        self.dl_earfcn = -1 #40936
        self.ul_freq_band = -1 #41
        self.ul_earfcn = -1 #40936
        if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
            self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
        else:
            self.dl_carrier_frequency_GHz = 2
            self.ul_carrier_frequency_GHz = 2
      
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.dl_bandwidth_MHz = 5   
        self.dl_subcarrier_spacing_kHz = 15
        self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz)
        self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.dl_subcarrier_spacing_kHz)
        self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(self.dl_subcarrier_spacing_kHz)
        
        self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
        self.ul_bandwidth_MHz = 5
        if (self.fdd_tdd_ind	== "TDD"):
            if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
                np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
                exit(0) 
        self.ul_subcarrier_spacing_kHz =  15
        self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
        self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
        self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
        
        self.BS_propagation_model = "3GPPTR38_811_Dense_Urban_NTN"
        self.BS_fast_channel_model = "Rician"        
      
        self.BS_tx_power_dBm = tools.mW_to_dBm(tools.dBm_to_mW(36) * self.dl_bandwidth_MHz * 1000) # 36 dbW/MHz 
        self.BS_noise_figure_dB = 5 #Taken from ITU UMa 
        self.min_BS_to_UE_2D_distance_m = 0 
        
        self.TX_number = 10
        self.RX_number = 10     
        
        self.CRS_ports = 1
        self.CRS_RE_tx_power_dBm = -1
        self.CRS_power_boosting_Pb = -1       
        
        self.antenna_pattern_model = "3GPPTR38_901" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
        
        #Element in TR 38.901     
        self.antenna_config_max_gain_dBi = 8
        self.antenna_config_hor_phi_3dB_deg = 65
        self.antenna_config_hor_A_m_dB = 30
        self.antenna_config_hor_alpha_mec_bearing_deg = np.array([0])
        self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0])
      
        self.antenna_config_ver_theta_3dB_deg = 65
        self.antenna_config_ver_SLA_dB = 30
        self.antenna_config_ver_beta_mec_downtilt_deg = 180 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
        self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 0 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array 
      
        self.antenna_config_gamma_mec_slant_deg = 0
        self.antenna_config_gamma_elec_slant_deg = 0
          
        #TR 38.901
        self.antenna_config_Mg = 1
        self.antenna_config_Ng = 1
        self.antenna_config_M = 10 # number of rows (elements in vertical)
        self.antenna_config_N = 1 # number of columns (elements in horizontal)
        self.antenna_config_P = "single" # (“single” or “dual”)    
        self.antenna_config_P_type = "V" # (“V” or “H” for single polarization. “VH” or “cross” for dual polarization.)
        
        if self.antenna_config_P == 'single': 
            P_num = 1
        else :
            P_num = 2
        self.antenna_config_number_of_elements = self.antenna_config_Mg * self.antenna_config_Ng * self.antenna_config_M * self.antenna_config_N * P_num
        
        self.antenna_config_dgh_m = 0 # 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dgv_m = 0 # 0.5 * self.dl_carrier_wavelength_m       
        self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
        
        # SSB precoding 
        self.SSB_precoder = "3GPPTR38_901_P1_single_column" 
        self.SSB_number_of_beams_V =  1
        self.SSB_number_of_beams_H =  1        
        self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
        
        # CSI-RS precoding
        self.CSI_RS_precoder = "3GPPTR38_901_P1_single_column" 
        self.CSI_RS_number_of_beams_V =  1
        self.CSI_RS_number_of_beams_H =  1        
        self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V
     
        ### Construct UE enginering parameters       
        return self.construct_data_frame()          


    def construct_scenario_3GPPTR36_777_UMa_AV(self, site_deployment_obj):
       
        self.number_of_sectors_per_site = 3
        self.bs_antenna_height_m = 25
        self.indoor = 0  
        
        # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
            # Note that not in all scenarios reference site locations equals to cell site locations
        self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

       
        # Calculate number of cell sites and cells
        self.number_of_cell_sites = len(self.cell_site_positions_m)
        self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
        # Derive cell locations 
            # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
        self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
        self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
        
        # Set velocity 
        self.velocity_kmh = 0
        self.calculate_bs_velocity_vectors()          
        
        # Radio access technology
        self.RAT = "NR"
       
        # Type of rado unit and bbu
        self.radio_unit_type = "5G-Macro-AAU-v1"
        self.bbu_type = "5G-Macro-BBU-v1"           
       
        # Set relevant parameters
        self.fdd_tdd_ind	= "TDD"
        self.subframe_assignment = "SA2"
        self.FR = 1
        self.subcarrier_per_PRB =  12
        self.ofdm_symbols_in_slot = 14
        
        self.dl_freq_band = -1 #41
        self.dl_earfcn = -1 #40936
        self.ul_freq_band = -1 #41
        self.ul_earfcn = -1 #40936
        if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
            self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
        else:
            self.dl_carrier_frequency_GHz = 2
            self.ul_carrier_frequency_GHz = 2
    
        self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
        self.dl_bandwidth_MHz = 10   
        self.dl_subcarrier_spacing_kHz = 15
        self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz)
        self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.dl_subcarrier_spacing_kHz)
        self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(self.dl_subcarrier_spacing_kHz)
        
        self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
        self.ul_bandwidth_MHz = 10
        if (self.fdd_tdd_ind	== "TDD"):
            if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
                np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
                exit(0) 
        self.ul_subcarrier_spacing_kHz =  15
        self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
        self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
        self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
        
        self.BS_propagation_model = "3GPPTR36_777_UMa_AV"
        self.BS_fast_channel_model = "Rician"         
    
        self.BS_tx_power_dBm = 46 
        self.BS_noise_figure_dB = 5
        
        self.TX_number = 1 
        self.RX_number = 1           
        
        self.CRS_ports = 1
        self.CRS_RE_tx_power_dBm = -1
        self.CRS_power_boosting_Pb = -1       
        
        self.antenna_pattern_model = "3GPPTR36_814_UMa" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
        
        #Array in TR 36.814
        self.antenna_config_max_gain_dBi = 17
        self.antenna_config_hor_phi_3dB_deg = 70
        self.antenna_config_hor_A_m_dB = 20
        self.antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
        self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0, 0, 0]) 
        
        self.antenna_config_ver_theta_3dB_deg = 15
        self.antenna_config_ver_SLA_dB = 20
        self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
        self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 12 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array   
        
        self.antenna_config_gamma_mec_slant_deg = 0
        self.antenna_config_gamma_elec_slant_deg = 0
          
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
        
        self.antenna_config_dgh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dgv_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
        self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
        
        # SSB precoding 
        self.SSB_precoder = "None"
        self.SSB_number_of_beams_H =  1
        self.SSB_number_of_beams_V =  1
        self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
       
        # CSI-RS precoding
        self.CSI_RS_precoder = "None"
        self.CSI_RS_number_of_beams_H =  1
        self.CSI_RS_number_of_beams_V =  1
        self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V      
     
        ### Construct UE enginering parameters
        return self.construct_data_frame()  


    def construct_scenario_3GPPTR36_777_UMi_AV(self, site_deployment_obj):
       
       self.number_of_sectors_per_site = 3
       self.bs_antenna_height_m = 10
       self.indoor = 0  
        
       # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
           # Note that not in all scenarios reference site locations equals to cell site locations
       self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 

       
       # Calculate number of cell sites and cells
       self.number_of_cell_sites = len(self.cell_site_positions_m)
       self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
       
       # Derive cell locations 
          # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
       self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
       self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
       
       # Set velocity 
       self.velocity_kmh = 0
       self.calculate_bs_velocity_vectors()         
       
       # Radio access technology
       self.RAT = "NR"
       
       # Type of rado unit and bbu
       self.radio_unit_type = "5G-Micro-AAU-v1"
       self.bbu_type = "5G-Micro-BBU-v1"          
       
       # Set relevant parameters
       self.fdd_tdd_ind	= "TDD"
       self.subframe_assignment = "SA2"
       self.FR = 1
       self.subcarrier_per_PRB =  12
       self.ofdm_symbols_in_slot = 14
       
       self.dl_freq_band = -1 #41
       self.dl_earfcn = -1 #40936
       self.ul_freq_band = -1 #41
       self.ul_earfcn = -1 #40936
       if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
           self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
       else:
           self.dl_carrier_frequency_GHz = 2
           self.ul_carrier_frequency_GHz = 2

       self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
       self.dl_bandwidth_MHz = 10   
       self.dl_subcarrier_spacing_kHz = 15
       self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz)
       self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.dl_subcarrier_spacing_kHz)
       self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(self.dl_subcarrier_spacing_kHz)
       
       self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
       self.ul_bandwidth_MHz = 10
       if (self.fdd_tdd_ind	== "TDD"):
           if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
               np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
               exit(0) 
       self.ul_subcarrier_spacing_kHz =  15
       self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
       self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
       self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
       
       self.BS_propagation_model = "3GPPTR36_777_UMi_AV"
       self.BS_fast_channel_model = "Rician"       

       self.BS_tx_power_dBm = 41 
       self.BS_noise_figure_dB = 5
       
       self.TX_number = 1 
       self.RX_number = 1          
       
       self.CRS_ports = 1
       self.CRS_RE_tx_power_dBm = -1
       self.CRS_power_boosting_Pb = -1      
       
       self.antenna_pattern_model = "3GPPTR36_814_UMa" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
       
       #Array in TR 36.814
       self.antenna_config_max_gain_dBi = 17
       self.antenna_config_hor_phi_3dB_deg = 70
       self.antenna_config_hor_A_m_dB = 20
       self.antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
       self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0, 0, 0])  
       
       self.antenna_config_ver_theta_3dB_deg = 15
       self.antenna_config_ver_SLA_dB = 20
       self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon
       self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 12 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array   
       
       self.antenna_config_gamma_mec_slant_deg = 0
       self.antenna_config_gamma_elec_slant_deg = 0
         
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
       
       self.antenna_config_dgh_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dgv_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
       self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
       
       # SSB precoding 
       self.SSB_precoder = "None"
       self.SSB_number_of_beams_H =  1
       self.SSB_number_of_beams_V =  1
       self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
      
       # CSI-RS precoding
       self.CSI_RS_precoder = "None"
       self.CSI_RS_number_of_beams_H =  1
       self.CSI_RS_number_of_beams_V =  1
       self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V       
    
       ### Construct UE enginering parameters
       return self.construct_data_frame()     
       
    
    def construct_scenario_ITU_R_M2135_UMa_colocated_multilayer(self, site_deployment_obj, carrier_frequencies_GHz, bandwidths_MHz):
        
        # Initialize an empty DataFrame to store the concatenated result
        combined_df_ep = pd.DataFrame()
        
        offset = 0
        for index in range(0,len(carrier_frequencies_GHz)) : # Loop over the multiple network layers to create
        
            self.number_of_sectors_per_site = 3
            self.bs_antenna_height_m = 25
            self.indoor = 0  
            
            # Read reference cell site locations, used to drive BS and/or UE deployments, and set them as cell site locations 
                # Note that not in all scenarios reference site locations equals to cell site locations
            self.cell_site_positions_m = site_deployment_obj.ref_cell_site_positions_m 
    
           
            # Calculate number of cell sites and cells
            self.number_of_cell_sites = len(self.cell_site_positions_m)
            self.number_of_cells = self.number_of_cell_sites * self.number_of_sectors_per_site
           
            # Derive cell locations 
                # Note that striclity speaking this represents the location of the cell site where the BS that generetes the cell is located
            self.cell_positions_m = np.repeat(self.cell_site_positions_m, repeats=self.number_of_sectors_per_site, axis=0)
            self.cell_positions_m = np.hstack((self.cell_positions_m , np.ones((self.number_of_cells,1)) * self.bs_antenna_height_m))
            
            # Set velocity 
            self.velocity_kmh = 0
            self.calculate_bs_velocity_vectors()              
            
            # Radio access technology
            self.RAT = "LTE"
       
            # Type of rado unit and bbu
            self.radio_unit_type = "4G-Macro-RRU-v1"
            self.bbu_type = "4G-Macro-BBU-v1"            
           
            # Set relevant parameters
            self.fdd_tdd_ind	= "TDD"
            self.subframe_assignment = "SA2"
            self.FR = 1
            self.subcarrier_per_PRB =  12
            self.ofdm_symbols_in_slot = 14
            
            self.dl_freq_band = -1 #41
            self.dl_earfcn = -1 #40936
            self.ul_freq_band = -1 #41
            self.ul_earfcn = -1 #40936
            if (self.dl_freq_band != -1 and self.dl_earfcn != -1):
                self.dl_carrier_frequency_GHz, self.ul_carrier_frequency_GHz, =  uaearfcn.earfcn2freq(self.freq_band,self.dl_earfcn)
            else:
                self.dl_carrier_frequency_GHz = carrier_frequencies_GHz[index]
                self.ul_carrier_frequency_GHz = carrier_frequencies_GHz[index]
     
            self.dl_carrier_wavelength_m = 3e8/(self.dl_carrier_frequency_GHz*1e9)
            self.dl_bandwidth_MHz = bandwidths_MHz[index]   
            self.dl_subcarrier_spacing_kHz = 15
            self.dl_PRBs_available = tools_carrier.bandwidth_to_PRBs(self.dl_bandwidth_MHz, self.dl_subcarrier_spacing_kHz)
            self.dl_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.dl_subcarrier_spacing_kHz)
            self.dl_control_channel_overhead = tools_carrier.control_channel_overhead(self.dl_subcarrier_spacing_kHz)
            
            self.ul_carrier_wavelength_m = 3e8/(self.ul_carrier_frequency_GHz*1e9)
            self.ul_bandwidth_MHz = bandwidths_MHz[index]
            if (self.fdd_tdd_ind	== "TDD"):
                if (self.dl_bandwidth_MHz != self.ul_bandwidth_MHz):
                    np.disp('Error: DL and UL bandwidths have to be the same if we are in TDD mode')
                    exit(0) 
            self.ul_subcarrier_spacing_kHz =  15
            self.ul_available_PRBs = tools_carrier.bandwidth_to_PRBs(self.ul_bandwidth_MHz, self.ul_subcarrier_spacing_kHz)
            self.ul_ofdm_symbol_duration_us = tools_carrier.ofdm_symbol_duration(self.ul_subcarrier_spacing_kHz)
            self.ul_control_channel_overhead = tools_carrier.control_channel_overhead(self.ul_subcarrier_spacing_kHz)
            
            self.BS_propagation_model = "ITU_R_M2135_UMa"
            self.BS_fast_channel_model = "Rician"            
     
            self.BS_tx_power_dBm = 49 #We substract here the feeder loss
            self.BS_noise_figure_dB = 5

            self.TX_number = 1 
            self.RX_number = 1   
            
            self.CRS_ports = 1
            self.CRS_RE_tx_power_dBm = -1
            self.CRS_power_boosting_Pb = -1      
            
            self.antenna_pattern_model = "3GPPTR36_814_UMa" # Note that 3GPP TR36_814 and 3GPP TR38_901 follow the same model but with different parameters
            
            #Array in TR 36.814
            self.antenna_config_max_gain_dBi = 17
            self.antenna_config_hor_phi_3dB_deg = 70
            self.antenna_config_hor_A_m_dB = 20
            self.antenna_config_hor_alpha_mec_bearing_deg = np.array([30, 150, -90])
            self.antenna_config_hor_alpha_elec_bearing_deg = np.array([0, 0, 0])
     
            self.antenna_config_ver_theta_3dB_deg = 15
            self.antenna_config_ver_SLA_dB = 20
            self.antenna_config_ver_beta_mec_downtilt_deg = 90 # 0 degrees is pointing to the sky; 90 degrees is pointing to the horizon      
            self.antenna_config_ver_beta_elec_downtilt_deg = 90 + 12 # The electricaltilt is relative, 0 degress is pointing along the panel upwards; 90 means pointing perpedicular to the array   
     
            self.antenna_config_gamma_mec_slant_deg = 0
            self.antenna_config_gamma_elec_slant_deg = 0
     
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
            
            self.antenna_config_dgh_m = 0.5 * self.dl_carrier_wavelength_m
            self.antenna_config_dgv_m = 0.5 * self.dl_carrier_wavelength_m
            self.antenna_config_dh_m = 0.5 * self.dl_carrier_wavelength_m
            self.antenna_config_dv_m = 0.5 * self.dl_carrier_wavelength_m
            
            # SSB precoding 
            self.SSB_precoder = "None" 
            self.SSB_number_of_beams_H =  1
            self.SSB_number_of_beams_V =  1
            self.SSB_number_of_beams = self.SSB_number_of_beams_H *self.SSB_number_of_beams_V
           
            # CSI-RS precoding
            self.CSI_RS_precoder = "None"
            self.CSI_RS_number_of_beams_H =  1
            self.CSI_RS_number_of_beams_V =  1
            self.CSI_RS_number_of_beams = self.CSI_RS_number_of_beams_H *self.CSI_RS_number_of_beams_V  
            
            # Create sectors 
            self.sectors, self.sectors_gp = self.sectorized_geographical_areas(site_deployment_obj.hexagons, self.cell_site_positions_m, site_deployment_obj.isd_m, self.antenna_config_hor_alpha_mec_bearing_deg)        
         
            # Construct UE enginering parameters
            df_ep = self.construct_data_frame()       
            
            # Concatenate the DataFrames of different network layers
            combined_df_ep = pd.concat([combined_df_ep, df_ep])
            
            # Update offsetto make sure that cell ID and stuff are coherent
            offset += self.number_of_cells
               
        # Reset the index of the combined DataFrame
        combined_df_ep = combined_df_ep.reset_index(drop=True)
        
        return combined_df_ep


    def construct_scenario_ITU_R_M2135_UMa_Umi_colocated_multilayer(self, site_deployment_obj):
       
       ### NOTE that this is considered as a UMa layout to next modules
    
       # Set necessary values to drive the construction of the scenario
       number_of_cells_per_layer = 57
              
       # Initialize DataFrame with ITU_R_M2135_UMa deployment 
       df_ep_UMa = self.construct_scenario_ITU_R_M2135_UMa(site_deployment_obj)
       
       # Initialize DataFrame with ITU_R_M2135_UMi deployment
           # We change the site_deployment_obj parameters to accomodate for UMi and change them back
           # and change them back to make sure that the UMa variables are the ones stored in the object. This allow to drive the UE deployment over the UMa, and not the UMi grid  
       site_deployment_obj.isd_m = 200
       site_deployment_obj.grid_resol_m = (int)(5)       
       df_ep_UMi = self.construct_scenario_ITU_R_M2135_UMi(site_deployment_obj)  
       site_deployment_obj.isd_m = 500
       site_deployment_obj.grid_resol_m = (int)(10)          
       
       # Concatenate the DataFrames of different network layers
       combined_df_ep = pd.concat([df_ep_UMa, df_ep_UMi])    
       
       # Replace the values of the necessary columns to create a coherent scenario
       combined_df_ep['ID'] = range(len(combined_df_ep)) 
       combined_df_ep['name'] = [ 'cell_%d'%(x) for x in range(len(combined_df_ep))]     
       combined_df_ep['PCI'] = range(len(combined_df_ep))      
     
       combined_df_ep.iloc[-number_of_cells_per_layer:, combined_df_ep.columns.isin(['position_x_m', 'position_y_m'])] =\
           combined_df_ep.iloc[:number_of_cells_per_layer, combined_df_ep.columns.isin(['position_x_m', 'position_y_m'])]
       
       combined_df_ep['radio_unit_type'] = [ "4G-Macro-RRU-v1" for x in range(len(combined_df_ep))] 
       combined_df_ep['bbu_type'] = [ "4G-Macro-BBU-v1" for x in range(len(combined_df_ep))] 
       
       # Reset the index of the combined DataFrame
       combined_df_ep = combined_df_ep.reset_index(drop=True)      
       
       return combined_df_ep     


    def construct_scenario_ITU_R_M2135_UMa_Umi_noncolocated_multilayer(self, site_deployment_obj):
       
       ### NOTE that this is considered as a UMa layout to next modules
       
       # Set necessary values to drive the construction of the scenario
       number_of_cell_sites = 2*19
       number_of_sectors_per_site = 3
       
       # Initialize DataFrame with ITU_R_M2135_UMa deployment 
       df_ep_UMa = self.construct_scenario_ITU_R_M2135_UMa(site_deployment_obj)
       
       # Initialize DataFrame with ITU_R_M2135_UMi deployment
           # We change the site_deployment_obj parameters to accomodate for UMi and change them back
           # and change them back to make sure that the UMa variables are the ones stored in the object. This allow to drive the UE deployment over the UMa, and not the UMi grid  
       site_deployment_UMi_obj = sites.Site(True,"ITU_R_M2135_UMi")
       site_deployment_UMi_obj.process()  
                
       df_ep_UMi = self.construct_scenario_ITU_R_M2135_UMi(site_deployment_UMi_obj)  
       
       # Concatenate the DataFrames of different network layers
       combined_df_ep = pd.concat([df_ep_UMa, df_ep_UMi])  
       
       # Replace the values of the necessary columns to create a coherent scenario
       combined_df_ep['ID'] = range(len(combined_df_ep)) 
       combined_df_ep['name'] = [ 'cell_%d'%(x) for x in range(len(combined_df_ep))] 
       combined_df_ep['PCI'] = range(len(combined_df_ep))   
       
       combined_df_ep['site_ID'] = np.repeat(np.arange(0,number_of_cell_sites), repeats=number_of_sectors_per_site, axis=0) # UMa and UMi sites cells are sitting in different sites
       combined_df_ep['site_name'] = np.repeat(['site_%d'%(x) for x in range(number_of_cell_sites)], repeats=number_of_sectors_per_site, axis=0) # UMa and UMi sites cells are sitting in different sites
       
       combined_df_ep['radio_unit_ID'] = range(len(combined_df_ep))      
       combined_df_ep['bbu_ID'] = range(len(combined_df_ep))      
                  
       # Reset the index of the combined DataFrame
       combined_df_ep = combined_df_ep.reset_index(drop=True)
       
       return combined_df_ep   
          
    
    def construct_scenario_3GPPTR38_901_4G(self, site_deployment_obj):
       
       ### This scenario has 4G (2GHz) in UMa grid 
       df_ep_UMa = self.construct_scenario_3GPPTR38_901_UMa_2GHz_large_scale_calibration(site_deployment_obj)
       df_ep_UMa['BS_tx_power_dBm'] = 46
       df_ep_UMa['BS_propagation_model'] = "3GPPTR38_901_UMa"
       
       return df_ep_UMa   
    

    def construct_scenario_3GPPTR38_901_5G(self, site_deployment_obj):
       
       ### This scenario has 5G (3.5GHz) in UMa grid 
       df_ep_UMa = self.construct_scenario_3GPPTR38_901_UMa_C_band_large_scale(site_deployment_obj)
       df_ep_UMa['BS_tx_power_dBm'] = 49  
       df_ep_UMa['BS_propagation_model'] = "3GPPTR38_901_UMa"
       
       return df_ep_UMa  


    def construct_scenario_3GPPTR38_901_6G(self, site_deployment_obj):
       
       ### This scenario has 6G (10GHz) in UMa grid       
       df_ep_UPi = self.construct_scenario_3GPPTR38_901_UPi_fr3_large_scale(site_deployment_obj)
       df_ep_UPi['BS_tx_power_dBm'] = 41   
       df_ep_UPi['BS_propagation_model'] = "3GPPTR38_901_UMa"
       
       return df_ep_UPi         
    

    def construct_scenario_3GPPTR38_901_4G5G(self, site_deployment_obj):
       
       ### This scenario has colocated 4G (2GHz) and 5G (3.5GHz) sites in UMa grid 
       
       ### NOTE that this is considered as a UMa layout to next modules
       
       # Set necessary values to drive the construction of the scenario
       layers = 2
       number_of_cell_sites_per_layer = 19
       sectors_per_site = 3
       
       # Initialize DataFrame with 3GPPTR38_901_UMa deployment for 4G cells 
       df_ep_UMa = self.construct_scenario_3GPPTR38_901_UMa_2GHz_large_scale_calibration(site_deployment_obj)
       df_ep_UMa['BS_tx_power_dBm'] = 46  
       
       # Initialize DataFrame with 3GPPTR38_901_UMi deployment for 5G cells
           # We change the site_deployment_obj parameters to accomodate for UMi and change them back
           # and change them back to make sure that the UMa variables are the ones stored in the object. This allow to drive the UE deployment over the UMa, and not the UMi grid  

       site_deployment_UMa_obj = sites.Site(True, "3GPPTR38_901_UMa_C_band_lsc")
       site_deployment_UMa_obj.process()  
        
       df_ep_UMa_C_band = self.construct_scenario_3GPPTR38_901_UMa_C_band_large_scale(site_deployment_UMa_obj)  
       df_ep_UMa_C_band['BS_tx_power_dBm'] = 49  
       
       # Concatenate the DataFrames of different network layers
       combined_df_ep = pd.concat([df_ep_UMa, df_ep_UMa_C_band])  
       
       # Replace the values of the necessary columns to create a coherent scenario
       combined_df_ep['ID'] = range(len(combined_df_ep)) 
       combined_df_ep['site_ID'] = np.tile(np.repeat(np.arange(0,number_of_cell_sites_per_layer), repeats=sectors_per_site, axis=0), layers) # 4G and 5G cells are sitting in the same sites
       combined_df_ep['name'] = [ 'cell_%d'%(x) for x in range(len(combined_df_ep))] 
       combined_df_ep['radio_unit_ID'] = combined_df_ep['site_ID']      
       combined_df_ep['bbu_ID'] = range(len(combined_df_ep))      
       combined_df_ep['PCI'] = range(len(combined_df_ep))   
       
       combined_df_ep['radio_unit_type'] = np.repeat("4G5G-Macro-AAU-v1", repeats=len(combined_df_ep), axis=0)
       combined_df_ep['bbu_type'] = np.repeat("5G-Macro-BBU-v1", repeats=len(combined_df_ep), axis=0)
       
       combined_df_ep['BS_propagation_model'] = "3GPPTR38_901_UMa"
       
                     
       # Reset the index of the combined DataFrame
       combined_df_ep = combined_df_ep.reset_index(drop=True)
       
       return combined_df_ep 
      

    def construct_scenario_3GPPTR38_901_4G_5G(self, site_deployment_obj):
       
       ### This scenario has noncolocated 4G (2GHz) and 5G (3.5GHz) sites in UMa and UMi grids, respectively 
        
       ### NOTE that this is considered as a UMa layout to next modules
       
       # Set necessary values to drive the construction of the scenario
       layers = 2
       number_of_cell_sites_per_layer = 19
       sectors_per_site = 3
       
       # Initialize DataFrame with 3GPPTR38_901_UMa deployment for 4G cells
       df_ep_UMa = self.construct_scenario_3GPPTR38_901_UMa_2GHz_large_scale_calibration(site_deployment_obj)
       df_ep_UMa['BS_tx_power_dBm'] = 46 
       
       # Initialize DataFrame with 3GPPTR38_901_UMi deployment for 5G cells
           # We change the site_deployment_obj parameters to accomodate for UMi and change them back
           # and change them back to make sure that the UMa variables are the ones stored in the object. This allow to drive the UE deployment over the UMa, and not the UMi grid  

       site_deployment_UMi_obj = sites.Site(True, "3GPPTR38_901_UMi_C_band_lsc")    
       site_deployment_UMi_obj.process()  
        
       df_ep_UMi_C_band = self.construct_scenario_3GPPTR38_901_UMi_C_band_large_scale(site_deployment_UMi_obj)  
       df_ep_UMi_C_band['BS_tx_power_dBm'] = 44        
       
       # Concatenate the DataFrames of different network layers
       combined_df_ep = pd.concat([df_ep_UMa, df_ep_UMi_C_band])  
       
       # Replace the values of the necessary columns to create a coherent scenario
       combined_df_ep['ID'] = range(len(combined_df_ep)) 
       combined_df_ep['site_ID'] = np.repeat(np.arange(0,layers*number_of_cell_sites_per_layer), repeats=sectors_per_site, axis=0) # 4G and 5G cells are sitting in different sites
       combined_df_ep['name'] = [ 'cell_%d'%(x) for x in range(len(combined_df_ep))] 
       combined_df_ep['site_name'] = ['site_%d' % x for x in combined_df_ep['site_ID']]
       combined_df_ep['radio_unit_ID'] = combined_df_ep['site_ID']      
       combined_df_ep['bbu_ID'] = range(len(combined_df_ep))      
       combined_df_ep['PCI'] = range(len(combined_df_ep))   
       
       combined_df_ep['BS_propagation_model'] = "3GPPTR38_901_UMa"
                     
       # Reset the index of the combined DataFrame
       combined_df_ep = combined_df_ep.reset_index(drop=True)
       
       return combined_df_ep
     
   
    def construct_scenario_3GPPTR38_901_4G_5G2(self, site_deployment_obj):
       
       ### This scenario has noncolocated 4G (2GHz) and 5G (3.5GHz) sites in UMa and UMi grids, respectively 
        
       ### NOTE that this is considered as a UMa layout to next modules
       
       # Set necessary values to drive the construction of the scenario
       layers = 2
       number_of_cell_sites_per_layer = 19
       sectors_per_site = 3
       
       # Initialize DataFrame with 3GPPTR38_901_UMa deployment 
       df_ep_UMa = self.construct_scenario_3GPPTR38_901_UMa_2GHz_large_scale_calibration(site_deployment_obj)
       df_ep_UMa['BS_tx_power_dBm'] = 46        
       
       # Initialize DataFrame with 3GPPTR38_901_UMi deployment
           # We change the site_deployment_obj parameters to accomodate for UMi and change them back
           # and change them back to make sure that the UMa variables are the ones stored in the object. This allow to drive the UE deployment over the UMa, and not the UMi grid  

       site_deployment_UMi_obj = sites.Site(True, "3GPPTR38_901_UMi_C_band_lsc")
       site_deployment_UMi_obj.process()  
        
       df_ep_UMa_C_band = self.construct_scenario_3GPPTR38_901_UMa_C_band_large_scale(site_deployment_UMi_obj) 
       df_ep_UMa_C_band['BS_tx_power_dBm'] = 49  
       
       # Concatenate the DataFrames of different network layers
       combined_df_ep = pd.concat([df_ep_UMa, df_ep_UMa_C_band])  
       
       # Replace the values of the necessary columns to create a coherent scenario
       combined_df_ep['ID'] = range(len(combined_df_ep)) 
       combined_df_ep['site_ID'] = np.repeat(np.arange(0,layers*number_of_cell_sites_per_layer), repeats=sectors_per_site, axis=0) # 4G and 5G cells are sitting in different sites
       combined_df_ep['name'] = [ 'cell_%d'%(x) for x in range(len(combined_df_ep))] 
       combined_df_ep['site_name'] = ['site_%d' % x for x in combined_df_ep['site_ID']]
       combined_df_ep['radio_unit_ID'] = combined_df_ep['site_ID']      
       combined_df_ep['bbu_ID'] = range(len(combined_df_ep))      
       combined_df_ep['PCI'] = range(len(combined_df_ep))   
       
       combined_df_ep['BS_propagation_model'] = "3GPPTR38_901_UMa"
                     
       # Reset the index of the combined DataFrame
       combined_df_ep = combined_df_ep.reset_index(drop=True)
       
       return combined_df_ep      
   

    def construct_scenario_3GPPTR38_901_4G_5G6G(self, site_deployment_obj):
       
       ### This scenario has noncolocated 4G (2GHz) and 5G (3.5GHz) sites in UMa and UMi grids, respectively, and a colocated 6G (10GHz) with the 5G 
        
       ### NOTE that this is considered as a UMa layout to next modules
       
       # Set necessary values to drive the construction of the scenario
       layers = 3
       number_of_cell_sites_per_layer = 19
       sectors_per_site = 3
       
       # Initialize DataFrame with 3GPPTR38_901_UMa deployment for 4G cells
       df_ep_UMa = self.construct_scenario_3GPPTR38_901_UMa_2GHz_large_scale_calibration(site_deployment_obj)
       df_ep_UMa['BS_tx_power_dBm'] = 46         
       
       # Initialize DataFrame with 3GPPTR38_901_UMi deployment for 5G and 6G cells
           # We change the site_deployment_obj parameters to accomodate for UMi and change them back
           # and change them back to make sure that the UMa variables are the ones stored in the object. This allow to drive the UE deployment over the UMa, and not the UMi grid  

       site_deployment_UMi_obj = sites.Site(True, "3GPPTR38_901_UMi_C_band_lsc")
       site_deployment_UMi_obj.process()  
        
       # 5G cells
       df_ep_UMi_C_band = self.construct_scenario_3GPPTR38_901_UMi_C_band_large_scale(site_deployment_UMi_obj)  
       df_ep_UMi_C_band['BS_tx_power_dBm'] = 44  
       
       # 6G cells
       df_ep_UMi_F3 = self.construct_scenario_3GPPTR38_901_UMi_fr3_large_scale(site_deployment_obj)
       df_ep_UMi_F3['BS_tx_power_dBm'] = 44  
       #df_ep_UMi_F3 = df_ep_UMi_F3[(df_ep_UMi_F3['site_ID'] >= 1) & (df_ep_UMi_F3['site_ID'] <= 7)]
       
       # Concatenate the DataFrames of different network layers
       combined_df_ep = pd.concat([df_ep_UMa, df_ep_UMi_C_band, df_ep_UMi_F3])  
       
       # Replace the values of the necessary columns to create a coherent scenario
       combined_df_ep['ID'] = range(len(combined_df_ep)) 
       array1 =  np.repeat(np.arange(0,(layers-1)*number_of_cell_sites_per_layer), repeats=sectors_per_site, axis=0)
       combined_df_ep['site_ID'] = np.concatenate((array1, array1[sectors_per_site*number_of_cell_sites_per_layer : ])) # 4G and 5G cells are sitting in different sites, while 6G sites are sitting in the same sites as 5G cells
       combined_df_ep['name'] = [ 'cell_%d'%(x) for x in range(len(combined_df_ep))] 
       combined_df_ep['site_name'] = ['site_%d' % x for x in combined_df_ep['site_ID']]
       combined_df_ep['radio_unit_ID'] = combined_df_ep['site_ID']      
       combined_df_ep['bbu_ID'] = range(len(combined_df_ep))      
       combined_df_ep['PCI'] = range(len(combined_df_ep))   
       
       combined_df_ep.loc[np.logical_or(combined_df_ep['radio_unit_type'] == "5G-Micro-AAU-v1",combined_df_ep['radio_unit_type'] == "6G-Micro-AAU-v1"), 'radio_unit_type'] = "5G6G-Micro-AAU-v1"
       
       combined_df_ep.loc[combined_df_ep['bbu_type'] == "5G-Micro-BBU-v1", 'bbu_type'] = "6G-Micro-BBU-v1"
       
       combined_df_ep['BS_propagation_model'] = "3GPPTR38_901_UMa"
   
       # Reset the index of the combined DataFrame
       combined_df_ep = combined_df_ep.reset_index(drop=True)
       
       return combined_df_ep   


    def construct_scenario_3GPPTR38_901_4G_5G_6G(self, site_deployment_obj, ue_hotspot_deployment_obj):
       
       ### This scenario has noncolocated 4G (2GHz) and 5G (3.5GHz) and  6G (10GHz) sites in UMa, UMi and UPi grids, respectively 
        
       ### NOTE that this is considered as a UMa layout to next modules
       
       # Set necessary values to drive the construction of the scenario
       layers = 3
       number_of_cell_sites_per_layer = 19
       sectors_per_site = 3
       
       # Initialize DataFrame with 3GPPTR38_901_UMa deployment for 4G cells
       df_ep_UMa = self.construct_scenario_3GPPTR38_901_UMa_2GHz_large_scale_calibration(site_deployment_obj)
       df_ep_UMa['BS_tx_power_dBm'] = 46
       
       # Initialize DataFrame with ITU_R_M2135_UMi deployment for 5G cells
           # We change the site_deployment_obj parameters to accomodate for UMi and change them back
           # and change them back to make sure that the UMa variables are the ones stored in the object. This allow to drive the UE deployment over the UMa, and not the UMi grid  

       site_deployment_UMi_obj = sites.Site(True, "3GPPTR38_901_UMi_C_band_lsc")
       site_deployment_UMi_obj.process()  
        
       # 5G cells
       df_ep_UMi_C_band = self.construct_scenario_3GPPTR38_901_UMi_C_band_large_scale(site_deployment_UMi_obj) 
       df_ep_UMi_C_band['BS_tx_power_dBm'] = 44
       
       # 6G cells   
           #If the playground has hotspots, the noncolocated 6G cells are deployed on top of the hotspots. Otherwise, they are deployed in UPi grid
       if ue_hotspot_deployment_obj.number_of_hotspots > 0 :
           df_ep_UPi_F3 = self.construct_scenario_3GPPTR38_901_UPi_fr3_large_scale(ue_hotspot_deployment_obj)  
       else: 
           #Initialize DataFrame with 3GPPTR38_901_UPi_fr3_lsc deployment for 6G cells
           site_deployment_UPi_obj = sites.Site(True, "3GPPTR38_901_UPi_fr3_lsc")
           site_deployment_UPi_obj.process()     
           
           df_ep_UPi_F3 = self.construct_scenario_3GPPTR38_901_UPi_fr3_large_scale(site_deployment_UPi_obj)  
       
       df_ep_UPi_F3['BS_tx_power_dBm'] = 41       
       
       # Concatenate the DataFrames of different network layers
       combined_df_ep = pd.concat([df_ep_UMa, df_ep_UMi_C_band, df_ep_UPi_F3])  
       
       # Replace the values of the necessary columns to create a coherent scenario
       combined_df_ep['ID'] = range(len(combined_df_ep)) 
       combined_df_ep['site_ID'] = np.repeat(np.arange(0,layers*number_of_cell_sites_per_layer), repeats=sectors_per_site, axis=0) # 4G, 5G and 6G cells are sitting in different sites
       combined_df_ep['name'] = [ 'cell_%d'%(x) for x in range(len(combined_df_ep))] 
       combined_df_ep['site_name'] = ['site_%d' % x for x in combined_df_ep['site_ID']]
       combined_df_ep['radio_unit_ID'] = combined_df_ep['site_ID']      
       combined_df_ep['bbu_ID'] = range(len(combined_df_ep))      
       combined_df_ep['PCI'] = range(len(combined_df_ep))   
       
       combined_df_ep['BS_propagation_model'] = "3GPPTR38_901_UMa"
                     
       # Reset the index of the combined DataFrame
       combined_df_ep = combined_df_ep.reset_index(drop=True)
       
       return combined_df_ep     


    def construct_scenario_3GPPTR38_901_4G5G_cell_reselection(self, site_deployment_obj):
       
       ### This scenario has colocated 4G (2GHz) nd 5G (3.5GHz) sites in UMa grid 
       
       ### NOTE that this is considered as a UMa layout to next modules
       
       # Set necessary values to drive the construction of the scenario
       layers = 2
       number_of_cell_sites_per_layer = 19
       sectors_per_site = 3
       
       # Initialize DataFrame with 3GPPTR38_901_UMa deployment 
       df_ep_UMa = self.construct_scenario_3GPPTR38_901_UMa_2GHz_large_scale_calibration(site_deployment_obj)
       df_ep_UMa['BS_tx_power_dBm'] = 46  
       
       # Initialize DataFrame with 3GPPTR38_901_UMi deployment
           # We change the site_deployment_obj parameters to accomodate for UMi and change them back
           # and change them back to make sure that the UMa variables are the ones stored in the object. This allow to drive the UE deployment over the UMa, and not the UMi grid  

       site_deployment_UMa_obj = sites.Site(True, "3GPPTR38_901_UMa_C_band_lsc")
       site_deployment_UMa_obj.process()  
        
       #df_ep_UMa_C_band = self.construct_scenario_3GPPTR38_901_UMa_C_band_large_scale_calibration(site_deployment_UMa_obj)  
       df_ep_UMa_C_band = self.construct_scenario_3GPPTR38_901_UMa_C_band_large_scale(site_deployment_UMa_obj)         
       df_ep_UMa_C_band['BS_tx_power_dBm'] = 49
       
       # Concatenate the DataFrames of different network layers
       combined_df_ep = pd.concat([df_ep_UMa, df_ep_UMa_C_band])  
       
       # Replace the values of the necessary columns to create a coherent scenario
       combined_df_ep['ID'] = range(len(combined_df_ep)) 
       combined_df_ep['site_ID'] = np.tile(np.repeat(np.arange(0,number_of_cell_sites_per_layer), repeats=sectors_per_site, axis=0), layers) # 4G and 5G cells are sitting in the same sites
       combined_df_ep['name'] = [ 'cell_%d'%(x) for x in range(len(combined_df_ep))] 
       combined_df_ep['radio_unit_ID'] = combined_df_ep['site_ID']      
       combined_df_ep['bbu_ID'] = range(len(combined_df_ep))      
       combined_df_ep['PCI'] = range(len(combined_df_ep))   
       
       combined_df_ep['radio_unit_type'] = np.repeat("4G5G-Macro-AAU-v1", repeats=len(combined_df_ep), axis=0)
       combined_df_ep['bbu_type'] = np.repeat("5G-Macro-BBU-v1", repeats=len(combined_df_ep), axis=0)
       
       combined_df_ep['BS_propagation_model'] = "3GPPTR38_901_UMa"
       
       # Reset the index of the combined DataFrame
       combined_df_ep = combined_df_ep.reset_index(drop=True)
       
       return combined_df_ep 
    

    def construct_scenario_dataset(self, site_deployment_obj):

        # Since there is a dataset available, no need to construct the scenario
        # Note: to import it, the dataset must be formatted according to the right template

        # Note: some of the parameters in self are needed in calculate_server_stats
        #       this is why we initialize them to the value of cell #0
        #       but they should not be used in the simulation!

        # Get the list of files in the directory that start with 'cell_deployment'
        cell_deployment_files = [f for f in os.listdir(self.dataset_import_folder) if f.startswith('cell_deployment')]

        # Check if no file or more than one file matches the pattern
        if len(cell_deployment_files) == 0:
            raise FileNotFoundError("No 'cell_deployment' file found in the specified folder.")
        elif len(cell_deployment_files) > 1:
            raise FileExistsError(
                "Multiple 'cell_deployment' files found in the specified folder. Please ensure there is only one file.")

        # There is exactly one file, proceed to load it
        cell_deployment_dataset = cell_deployment_files[0]
        cell_deployment_dataset_full_file_path = os.path.join(self.dataset_import_folder, cell_deployment_dataset)

        # Read the csv, ensuring "None" values are not intepreted as Nan (e.g., for precoder type "None")
        na_values = ['NA', 'nan', 'null', 'NULL', '', 'NaN', '#N/A', '#N/A N/A', '#NA', '-1.#IND',
                     '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', 'N/A', 'n/a', 'NA ', 'null.',
                     'NULL.', 'nan ', 'Nan', 'NaN ', 'nan']
        print(f"Loading 'cell_deployment' dataset from file: {cell_deployment_dataset_full_file_path}")
        deployment_dataset_df = pd.read_csv(cell_deployment_dataset_full_file_path, low_memory=False, sep=',', keep_default_na=False, na_values=na_values)

        # Unique site positions
        unique_sites = deployment_dataset_df.drop_duplicates(subset='site_ID')
        self.cell_site_positions_m = unique_sites[['position_x_m', 'position_y_m']].to_numpy()

        # All cell positions
        self.cell_positions_m = deployment_dataset_df[['position_x_m', 'position_y_m']].to_numpy()

        # Calculate number of cell sites and cells
        self.number_of_cell_sites = len(self.cell_site_positions_m)
        self.number_of_cells = deployment_dataset_df.shape[0]

        # SSB precoding
        self.SSB_precoder = deployment_dataset_df.loc[0, 'SSB_precoder']
        self.SSB_number_of_beams_H = deployment_dataset_df.loc[0, 'SSB_number_of_beams_H']
        self.SSB_number_of_beams_V = deployment_dataset_df.loc[0, 'SSB_number_of_beams_V']
        self.SSB_number_of_beams = deployment_dataset_df.loc[0, 'SSB_number_of_beams']

        # CSI-RS precoding
        self.CSI_RS_precoder = deployment_dataset_df.loc[0, 'CSI_RS_precoder']
        self.CSI_RS_number_of_beams_H = deployment_dataset_df.loc[0, 'CSI_RS_number_of_beams_H']
        self.CSI_RS_number_of_beams_V = deployment_dataset_df.loc[0, 'CSI_RS_number_of_beams_V']
        self.CSI_RS_number_of_beams = deployment_dataset_df.loc[0, 'CSI_RS_number_of_beams']

        return deployment_dataset_df