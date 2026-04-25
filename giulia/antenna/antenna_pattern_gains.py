# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:40:53 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import time

import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import List

from giulia.event_driven.snapshot_control import Snapshot_control
from giulia.fs import results_file
from giulia.tools import tools
from giulia.tools.tools import log_calculations_time
from scipy.special import j1
from giulia.outputs.saveable import Saveable


class Antenna_Pattern_Gain(Saveable):
    
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
        self.antenna_pattern_gain_b_to_a_dB = []
        self.df_antenna_pattern_gain_b_to_a_dB = []

    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["antenna_pattern_gain_b_to_a_dB"]

       
    def process(self, rescheduling_us=-1):
            """
            Processes antenna pattern models and sector rotation values to compute antenna gain 
            for each User Equipment (UE) in the network.
            """
            
            # Extract antenna pattern models from network deployment data
            self.antenna_pattern_models = self.network_deployment_obj.df_ep["antenna_pattern_model"].to_numpy()
            
            # Extract antenna pattern offset parameters
            self.A_max_dBi = self.network_deployment_obj.df_ep["antenna_config_max_gain_dBi"].to_numpy(dtype=np.single)
            self.phi_3dB_deg = self.network_deployment_obj.df_ep["antenna_config_hor_phi_3dB_deg"].to_numpy(dtype=np.single)
            self.A_m_dB = self.network_deployment_obj.df_ep["antenna_config_hor_A_m_dB"].to_numpy(dtype=np.single)
            self.theta_3dB_deg = self.network_deployment_obj.df_ep["antenna_config_ver_theta_3dB_deg"].to_numpy(dtype=np.single)
            self.SLA_v_dB = self.network_deployment_obj.df_ep["antenna_config_ver_SLA_dB"].to_numpy(dtype=np.single)
            
            # Extract antenna sector rotation values
            ## Horizontal mechanical and electrical bearing
            self.alpha_mec_bearing_deg = self.network_deployment_obj.df_ep["antenna_config_hor_alpha_mec_bearing_deg"].to_numpy(dtype=np.single)
            self.alpha_elec_bearing_deg = self.network_deployment_obj.df_ep["antenna_config_hor_alpha_elec_bearing_deg"].to_numpy(dtype=np.single)
            
            ## Vertical mechanical and electrical downtilt
            self.beta_mec_downtilt_deg = self.network_deployment_obj.df_ep["antenna_config_ver_beta_mec_downtilt_deg"].to_numpy(dtype=np.single)
            self.beta_elec_downtilt_deg = self.network_deployment_obj.df_ep["antenna_config_ver_beta_elec_downtilt_deg"].to_numpy(dtype=np.single)
            
            ## Mechanical slant
            self.gamma_mec_slan_deg = self.network_deployment_obj.df_ep["antenna_config_gamma_mec_slant_deg"].to_numpy(dtype=np.single)
            
            # Extract UE angles in Global Coordinate System (GCS)
            self.azimuths_b_to_a_deg = self.distance_angles_ue_to_cell_obj.azimuths_b_to_a_wraparound_degrees
            self.zeniths_b_to_a_deg = self.distance_angles_ue_to_cell_obj.zeniths_b_to_a_wraparound_degrees
            
            # Initialize output array for antenna gain
            self.antenna_pattern_gain_b_to_a_dB = np.zeros(
                (np.size(self.azimuths_b_to_a_deg, 0), np.size(self.azimuths_b_to_a_deg, 1)), dtype=np.single)
            
            
            self.dl_carrier_wavelength_m = self.network_deployment_obj.df_ep["dl_carrier_wavelength_m"].to_numpy(dtype=np.single)
            self.antenna_radius_ReflectorAperture_m = self.network_deployment_obj.df_ep["antenna_radius_ReflectorAperture_m"].to_numpy(dtype=np.single)
            
            
            # Start processing time measurement
            t_start = time.perf_counter()
            
            # Identify unique antenna models for independent processing
            antenna_pattern_models_set = set(self.antenna_pattern_models)
            
            # Iterate through each unique antenna pattern model
            for antenna_pattern_model in antenna_pattern_models_set:
                
                # Mask to select cells using the current antenna model
                mask = antenna_pattern_model == self.antenna_pattern_models
                
                # Compute antenna gain based on the selected model
                if antenna_pattern_model == "omnidirectional":
                    # Extract relevant parameters for selected cells
                    A_max_dBi = self.A_max_dBi[mask]
                    
                    # Set Antenna Gains equally to A_max_dBi
                    self.antenna_pattern_gain_b_to_a_dB[:, mask] = np.ones(np.size(mask, 0), dtype=np.single) * A_max_dBi
                    
                    
                elif antenna_pattern_model in ["3GPPTR36_814_UMa", "3GPPTR38_901"]:
                    # Extract relevant parameters for selected cells
                    A_max_dBi = self.A_max_dBi[mask]
                    phi_3dB_deg = self.phi_3dB_deg[mask]
                    A_m_dB = self.A_m_dB[mask]
                    theta_3dB_deg = self.theta_3dB_deg[mask]
                    SLA_v_dB = self.SLA_v_dB[mask]
                    
                    # Mechanical rotation angles for GCS to Local Coordinate System (LCS) conversion
                    α_mec_deg_arr = self.alpha_mec_bearing_deg[mask]
                    β_mec_deg_arr = self.beta_mec_downtilt_deg[mask]
                    γ_mec_deg_arr = self.gamma_mec_slan_deg[mask]
                    
                    # Electrical rotation angles
                    α_elec_deg_arr = self.alpha_elec_bearing_deg[mask]
                    β_elec_deg_arr = self.beta_elec_downtilt_deg[mask]
                    
                    # Select UE angles in GCS for the identified cells
                    azimuths_b_to_a_deg = self.azimuths_b_to_a_deg[:, mask]
                    zeniths_b_to_a_deg = self.zeniths_b_to_a_deg[:, mask]
                    
                    # Initialize placeholders for angles in LCS
                    azimuths_b_to_a_deg_lcs = np.full_like(zeniths_b_to_a_deg, np.nan)
                    zeniths_b_to_a_deg_lcs = np.full_like(zeniths_b_to_a_deg, np.nan)
                    
                    # Retrieve rotation function
                    func_rotation = self.distance_angles_ue_to_cell_obj.calculate_azimuth_zenith_deg_from_GCS_to_LCS_3GPPComp_perCell
                    
                    # Compute angles in LCS per cell
                    for i, (α_mec_deg, β_mec_deg, γ_mec_deg, azimuths_b_deg, zeniths_b_deg) in \
                        enumerate(zip(α_mec_deg_arr, β_mec_deg_arr, γ_mec_deg_arr, azimuths_b_to_a_deg.T, zeniths_b_to_a_deg.T)):
                            
                            azimuths_b_to_a_deg_lcs[:, i], zeniths_b_to_a_deg_lcs[:, i] = func_rotation(α_mec_deg, β_mec_deg, γ_mec_deg, 
                                                                                                        azimuths_b_deg, zeniths_b_deg)
                    
                    # Compute Gain from Antenna Pattern following 3GPP TR 36.814 [UMa] or TR 38.901 models                 
                    _, _, self.antenna_pattern_gain_b_to_a_dB[:, mask] = self.antenna_pattern_gain_3GPP_model(
                        azimuths_b_to_a_deg_lcs, zeniths_b_to_a_deg_lcs, A_max_dBi,
                        α_elec_deg_arr, phi_3dB_deg, A_m_dB,
                        β_elec_deg_arr, theta_3dB_deg, SLA_v_dB)
                    
                    
                elif antenna_pattern_model in ["3GPPTR38_811"]:
                    # Extract relevant parameters for selected cells
                    A_max_dBi = self.A_max_dBi[mask]
                    
                    # Mechanical rotation angles for GCS to Local Coordinate System (LCS) conversion
                    α_mec_deg_arr = self.alpha_mec_bearing_deg[mask]
                    β_mec_deg_arr = self.beta_mec_downtilt_deg[mask]
                    
                    # Select UE angles in GCS for the identified cells
                    azimuths_b_to_a_deg = self.azimuths_b_to_a_deg[:, mask]
                    zeniths_b_to_a_deg = self.zeniths_b_to_a_deg[:, mask]
                    
                    # Carrier information
                    dl_carrier_wavelength_m = self.dl_carrier_wavelength_m[mask]
                    
                    # Aperture Information 
                    antenna_radius_ReflectorAperture_m = self.antenna_radius_ReflectorAperture_m[mask]
                    
                    # Compute Antenna Pattern from a Reflector Antenna following 3GPP TR38_811 Model
                    self.antenna_pattern_gain_b_to_a_dB[:, mask] =\
                        self.antenna_pattern_gain_3GPP_model_reflector(zeniths_b_to_a_deg, azimuths_b_to_a_deg,
                                                                      β_mec_deg_arr, α_mec_deg_arr,
                                                                      A_max_dBi, 
                                                                      antenna_radius_ReflectorAperture_m, dl_carrier_wavelength_m)
                        
                else: raise ValueError(f"Antenna Pattern Offset {antenna_pattern_model} not valid")
                  
                
            
            # Store results in a DataFrame for post-processing
            self.df_antenna_pattern_gain_b_to_a_dB = pd.DataFrame(
                self.antenna_pattern_gain_b_to_a_dB,
                columns=self.network_deployment_obj.df_ep["name"],
                index=self.ue_deployment_obj.df_ep["name"]
            )
            
            # Save results for plotting if required
            snapshot_control = Snapshot_control.get_instance()
            if self.simulation_config_obj.plot == 1 and snapshot_control.num_snapshots == 0:
                file_name = results_file(self.simulation_config_obj.project_name, 'to_plot_antenna_pattern_gains')
                np.savez(file_name, antenna_pattern_gain_b_to_a_dB=self.antenna_pattern_gain_b_to_a_dB)
                self.example_antenna_pattern_gain()
            
            # Log calculation time
            log_calculations_time('Antenna pattern', t_start)
            
            return rescheduling_us


    def horizontal_offset_3GPP_model(self, 
                                     phi_deg: npt.NDArray[np.float32],
                                     alpha_bearing_elec_deg: npt.NDArray[np.float32],
                                     phi_3dB_deg: npt.NDArray[np.float32],
                                     A_m_dB: npt.NDArray[np.float32]):

        phi_rel_deg = tools.angle_range_0_180(phi_deg - alpha_bearing_elec_deg)
             
        return -np.minimum(12*np.square(phi_rel_deg/phi_3dB_deg), A_m_dB) 
        

    def vertical_offset_3GPP_model(self,
                                   theta_deg: npt.NDArray[np.float32],
                                   beta_elec_downtilt_deg: npt.NDArray[np.float32],
                                   theta_3dB_deg: npt.NDArray[np.float32],
                                   SLA_v_dB: npt.NDArray[np.float32]):
        
        return -np.minimum(12*np.square((theta_deg - beta_elec_downtilt_deg) / theta_3dB_deg), SLA_v_dB)
        

    def combined_offset_3GPP_model(self,
                                   A_h_dB: npt.NDArray[np.float32], 
                                   A_v_dB: npt.NDArray[np.float32], 
                                   A_m_dB: npt.NDArray[np.float32]):
        
        return -1 * np.minimum(-1*(A_h_dB + A_v_dB), A_m_dB)


    def antenna_pattern_gain_3GPP_model(self, 
                                        azimuths_ue_to_cell_deg_lcs: npt.NDArray[np.float32], 
                                        zeniths_ue_to_cell_deg_lcs: npt.NDArray[np.float32], 
                                        A_max_dBi: npt.NDArray[np.float32], 
                                        alpha_bearing_elec_deg: npt.NDArray[np.float32], 
                                        phi_3dB_deg: npt.NDArray[np.float32], 
                                        A_m_dB: npt.NDArray[np.float32], 
                                        beta_elec_downtilt_deg: npt.NDArray[np.float32],
                                        theta_3dB_deg: npt.NDArray[np.float32], 
                                        SLA_v_dB: npt.NDArray[np.float32]):
            
        # Get horizontal offset
        A_h_dB = self.horizontal_offset_3GPP_model(azimuths_ue_to_cell_deg_lcs, alpha_bearing_elec_deg, phi_3dB_deg, A_m_dB)
        
        # Get vertical offset
        A_v_dB = self.vertical_offset_3GPP_model(zeniths_ue_to_cell_deg_lcs, beta_elec_downtilt_deg, theta_3dB_deg, SLA_v_dB)
        
        # Combining offsets in 3D antenna pattern and adding the maximum element gain              
        return A_h_dB, A_v_dB, A_max_dBi + self.combined_offset_3GPP_model(A_h_dB, A_v_dB, A_m_dB)

   
    def antenna_pattern_gain_3GPP_model_reflector(self, 
                                                  zeniths_b_to_a_deg_gcs: npt.NDArray[np.float32], 
                                                  azimuth_b_to_a_deg_gcs: npt.NDArray[np.float32],
                                                  β_mec_deg_arr: npt.NDArray[np.float32], 
                                                  α_mec_deg_arr: npt.NDArray[np.float32],
                                                  A_max_dBi: npt.NDArray[np.float32], 
                                                  antenna_radius_ReflectorAperture_m: npt.NDArray[np.float32],
                                                  dl_carrier_wavelength_m: npt.NDArray[np.float32]):
        """
        Computes the antenna pattern gain for a reflector using the 3GPP model.
        This code assumes that the reflector's aperture is perfectly circular.
        """
        # Convert mechanical elevation angles from degrees to radians
        β_mec_rad_arr = np.deg2rad(β_mec_deg_arr)
        # Convert mechanical azimuth angles from degrees to radians
        α_mec_rad_arr = np.deg2rad(α_mec_deg_arr)
        
        # Convert zenith angles from degrees to radians in the global coordinate system (GCS)
        zeniths_b_to_a_rad_gcs = np.deg2rad(zeniths_b_to_a_deg_gcs)
        # Convert azimuth angles from degrees to radians in the global coordinate system (GCS)
        azimuth_b_to_a_rad_gcs = np.deg2rad(azimuth_b_to_a_deg_gcs)
        
        # Delete the original degree arrays as they are no longer needed and may cause errors
        del β_mec_deg_arr, α_mec_deg_arr, zeniths_b_to_a_deg_gcs, azimuth_b_to_a_deg_gcs
        
        # Compute the normal vector for the mechanical directions of the reflector in GCS
        n_gcs_arr = np.array([
            np.sin(β_mec_rad_arr) * np.cos(α_mec_rad_arr),
            np.sin(β_mec_rad_arr) * np.sin(α_mec_rad_arr),
            np.cos(β_mec_rad_arr)
        ])
        
        # Compute the normalized position vector in GCS
        ρ_gcs_arr = np.array([
            np.sin(zeniths_b_to_a_rad_gcs) * np.cos(azimuth_b_to_a_rad_gcs),
            np.sin(zeniths_b_to_a_rad_gcs) * np.sin(azimuth_b_to_a_rad_gcs),
            np.cos(zeniths_b_to_a_rad_gcs)
        ])
        
        # Calculate the zenith angles in the local coordinate system (LCS) using the dot product
        ## This computation is performed for all masked cells using einsum
        zeniths_b_to_a_rad_lcs = np.arccos(np.einsum('ijk,ik->jk', ρ_gcs_arr, n_gcs_arr))
        
        # Compute the argument for the antenna pattern model:
        # (pi / wavelength) * total aperture * sin(zenith angle offset)        
        arg = (np.pi / dl_carrier_wavelength_m) *2* antenna_radius_ReflectorAperture_m * np.sin(zeniths_b_to_a_rad_lcs)
        # Avoid division by zero: replace any zeros in 'arg' with a small number (1e-20)
        arg = np.where(arg == 0, 1e-20, arg)
        
        # Compute the antenna gain offset in linear scale using the Bessel function (j1)
        ## according to 3GPP TR38.811
        offset_b_to_a_lin = 4 * np.square(np.abs(j1(arg) / arg))
        # For zenith angles equal to zero in LCS, set the offset to 1
        offset_b_to_a_lin[zeniths_b_to_a_rad_lcs == 0] = 1
      
        # Convert the linear scale gain to dBm and return the result
        return tools.mW_to_dBm(offset_b_to_a_lin) + A_max_dBi


    def example_antenna_pattern_gain(self):
        A_max_dBi = np.array([8], dtype=np.float32)
        alpha_elec_bearing_deg = np.array([0], dtype=np.float32)
        phi_3dB_deg = np.array([65], dtype=np.float32)
        A_m_dB = np.array([30], dtype=np.float32)
        beta_elec_downtilt_deg = np.array([90], dtype=np.float32)
        theta_3dB_deg = np.array([65], dtype=np.float32)
        SLA_v_dB = np.array([30], dtype=np.float32)
         
        azimuth_deg = np.expand_dims(np.arange(-180, 180, 180/500), axis=1)
        zenith_deg = np.expand_dims(np.arange(0, 180, 180/1000), axis=1)
    
        A_h_dB, A_v_dB, A_combined_dB = self.antenna_pattern_gain_3GPP_model(azimuth_deg, zenith_deg, A_max_dBi, alpha_elec_bearing_deg, phi_3dB_deg, A_m_dB, beta_elec_downtilt_deg, theta_3dB_deg, SLA_v_dB )
        
        snapshot_control = Snapshot_control.get_instance()
        if self.simulation_config_obj.plot == 1 and snapshot_control.num_snapshots == 0:
            file_name = results_file(self.simulation_config_obj.project_name, 'to_plot_antenna_pattern')
            np.savez(file_name,azimuth_deg=azimuth_deg,zenith_deg=zenith_deg,A_h_dB=A_h_dB,A_v_dB=A_v_dB)