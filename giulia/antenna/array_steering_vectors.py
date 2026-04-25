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
from typing import List

import numpy as np

from giulia.event_driven.snapshot_control import Snapshot_control
from giulia.fs import results_file
from giulia.tools import tools
from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable

class Array_Steering_Vector(Saveable):
    
    def __init__(self, 
                 simulation_config_obj,
                 network_deployment_obj, 
                 ue_deployment_obj,
                 cell_antenna_array_structure_obj, 
                 distance_angles_ue_to_cell_obj):
        
        super().__init__()

        ##### Input storage 
        ######################## 
        self.simulation_config_obj = simulation_config_obj
        self.network_deployment_obj = network_deployment_obj
        self.ue_deployment_obj = ue_deployment_obj 
        self.cell_antenna_array_structure_obj = cell_antenna_array_structure_obj
        self.distance_angles_ue_to_cell_obj = distance_angles_ue_to_cell_obj
        
        
        ##### Outputs 
        ########################           
        self.array_steering_vector_bAnt_to_aAnt_complex = [] 
        self.example_array_steering_vector_bAnt_to_aAnt_complex_H = None
        self.example_array_steering_vector_bAnt_to_aAnt_complex_V = None
  
    
    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["array_steering_vector_bAnt_to_aAnt_complex"]

       
    def process(self, rescheduling_us=-1): 
        
        ##### Process inputs
        ######################## 
        
        # Network
        self.antenna_pattern_models = self.network_deployment_obj.df_ep["antenna_pattern_model"].to_numpy() 
        
        self.cell_ID = self.network_deployment_obj.df_ep["ID"].to_numpy()   
        self.dl_carrier_wavelength_m = self.network_deployment_obj.df_ep["dl_carrier_wavelength_m"].to_numpy(dtype=np.single)   
        self.alpha_mec_bearing_deg = self.network_deployment_obj.df_ep["antenna_config_hor_alpha_mec_bearing_deg"].to_numpy(dtype=np.single) 
        self.beta_mec_downtilt_deg = self.network_deployment_obj.df_ep["antenna_config_ver_beta_mec_downtilt_deg"].to_numpy(dtype=np.single) 
        self.beta_elec_downtilt_deg = self.network_deployment_obj.df_ep["antenna_config_ver_beta_elec_downtilt_deg"].to_numpy(dtype=np.single) 
        self.antenna_config_dv_m = self.network_deployment_obj.df_ep["antenna_config_dv_m"].to_numpy(dtype=np.single)
        self.antenna_config_dh_m = self.network_deployment_obj.df_ep["antenna_config_dh_m"].to_numpy(dtype=np.single)       

        # Cell antenna array
        self.antenna_to_node_a_mapping = self.cell_antenna_array_structure_obj.antenna_to_node_mapping 
        self.antenna_LCS_index = self.cell_antenna_array_structure_obj.df_ep[["LCS_index_x", "LCS_index_y", "LCS_index_z"]].to_numpy(dtype=int)      
  
        # Channel characterisitics  
        self.azimuths_b_to_a_degrees = self.distance_angles_ue_to_cell_obj.azimuths_b_to_a_wraparound_degrees
        self.zeniths_b_to_a_degrees = self.distance_angles_ue_to_cell_obj.zeniths_b_to_a_wraparound_degrees   
        
         
        ##### Process outputs
        ########################          
        self.array_steering_vector_bAnt_to_aAnt_complex = \
            np.ones((1, np.size(self.ue_deployment_obj.df_ep,0), np.size(self.antenna_to_node_a_mapping) ), dtype=np.csingle) # The first dimension is a formality to cover for the PRB domain later        
       
        
        ##### Start timer
        ########################     
        t_start = time.perf_counter()      
        
        
        ##### Switch
        ########################         
        
        # Find the set of unique antenna array models to process them independently
        antenna_pattern_models_set = set(self.antenna_pattern_models) 
        
        # Process each antenna array model independently
        for  antenna_pattern_model in antenna_pattern_models_set:
            # Identify cells with the selected antenna array model
            mask = antenna_pattern_model ==  self.antenna_pattern_models
            
            # Get necessary information of the identified cells
            selected_cell_IDs = self.cell_ID[mask]
            
            antenna_to_node_a_mapping = self.antenna_to_node_a_mapping[np.isin(self.antenna_to_node_a_mapping, selected_cell_IDs)]
            
            azimuths_b_to_a_degrees = self.azimuths_b_to_a_degrees[:, mask]  
            zeniths_b_to_a_degrees = self.zeniths_b_to_a_degrees[:, mask]  
 
            # Calculate antenna array gains taking into account the antenna array geometry and the respective precoders. 
            if (antenna_pattern_model == "3GPPTR38_901"):    
                # In this model, we are presented with the antenna pattern of the antenna element and the array structure
                # The far-zone field of a uniform array of identical elements is equal to the product of the field of a single element and the array factor of that array.
                
                mask_antennas_of_selected_cells = np.isin(self.antenna_to_node_a_mapping, selected_cell_IDs)
                self.array_steering_vector_bAnt_to_aAnt_complex[:,:,mask_antennas_of_selected_cells] =\
                    self.calculate_array_steering_vectors(selected_cell_IDs, antenna_to_node_a_mapping, azimuths_b_to_a_degrees,zeniths_b_to_a_degrees)
                
        # Store in data frames the results as it may be useful to post process  
        ##### Save to plot
        ########################
        snapshot_control = Snapshot_control.get_instance()
        if self.simulation_config_obj.plot == 1 and snapshot_control.num_snapshots == 0:
            file_name = results_file(self.simulation_config_obj.project_name, 'to_plot_array_steering_vector_gains')
            np.savez(file_name, array_steering_vector_gain_dB=tools.mW_to_dBm(np.abs(self.array_steering_vector_bAnt_to_aAnt_complex) ** 2))

            self.example_array_steering_vector(cell_index=0)        
  
    
        ##### End
        ########################
        log_calculations_time('Array steering vector', t_start)

        return rescheduling_us    


    def calculate_array_steering_vector_per_cell(self, 
                                             cell_idx,
                                             phi_b_to_a_radians,
                                             theta_b_to_a_radians,
                                             is_example_flag=False):
        """
        Computes the array steering vector for a given cell in the network.
        
        This implementation follows the model described in 3GPP 38.901, section 7.5 (Fast fading model), 
        specifically equations 7.5-23, 7.5-24, and 7.5-29. In 3GPP notation, 'r' and 'd' are used, 
        but here we adopt the nomenclature from Emil Björnson's book
        "Massive MIMO Networks: Spectral, Energy, and Hardware Efficiency" (section 7.3.1 - 3D LoS Model with Arbitrary Array Geometry), 
        using 'K_wave' for the wave vector and 'Um_3xM' for the matrix containing the antenna position in GCS.
        
        Note that, if wraparound is not enabled, distances are pure distances. This has been previously handled by distance_angles_ue_to_cell_obj
        
        Parameters:
        - cell_idx (int): Index of the target cell.
        - phi_b_to_a_radians (np.ndarray, dtype=float32): Azimuth angle (radians).
        - theta_b_to_a_radians (np.ndarray, dtype=float32): Elevation angle (radians).
        - is_example_flag (bool, optional): Flag to enable example computation (default: False).
        
        Returns:
        - np.ndarray: Array steering vector for the given cell.
        """
        # --- Compute Wave Vector ---
        lambda_carrier = self.dl_carrier_wavelength_m[cell_idx]  # Carrier wavelength
    
        # Define the wave vector based on the azimuth and elevation angles
        K_WaveVector_3xUE = (2 * np.pi / lambda_carrier) * np.array([
            np.cos(phi_b_to_a_radians) * np.sin(theta_b_to_a_radians),
            np.sin(phi_b_to_a_radians) * np.sin(theta_b_to_a_radians),
            np.cos(theta_b_to_a_radians)
        ])
        
        # --- Retrieve Antenna Panel Global Positions ---
        df_ep = self.cell_antenna_array_structure_obj.df_ep.copy()
        df_ep_cellArray = df_ep[df_ep['host_node_ID'] == cell_idx]
        Um_3xM = np.array([
            df_ep_cellArray["position_x_m"], 
            df_ep_cellArray["position_y_m"], 
            df_ep_cellArray["position_z_m"]
        ])
        
        # --- Compute Phase Components ---
        # Compute Macro Phase Rotation
        if not is_example_flag: # If it is for example test purpose, the value for macroPhase_a is set to zero
            macroPhase_a = (2 * np.pi / lambda_carrier) * \
                           self.distance_angles_ue_to_cell_obj.distance_b_to_a_3d_wraparound_m[:, cell_idx]
        else:
            macroPhase_a = np.array([0.00])  
        
        # Compute relative phase based on Wave vector and antenna positions
        relPhase_axM = (K_WaveVector_3xUE.T @ Um_3xM)
        
        # Compute total phase and apply exponential operator
        total_phase_a = -macroPhase_a[:, None] + relPhase_axM
        
        return np.exp(1j * total_phase_a)

    
                 
    def calculate_array_steering_vectors(self, 
                                         selected_cell_IDs,  
                                         antenna_to_node_a_mapping,                                           
                                         azimuths_b_to_a_degrees, 
                                         zeniths_b_to_a_degrees):
                            
        # Placeholder to store the array gain results
        number_of_antennas_in_selection = np.sum(np.size(antenna_to_node_a_mapping))       
        array_steering_vector_bAnt_to_aAnt_complex = np.zeros((1,  # The first dimension is a formality to cover for the PRB domain later
                                                               np.size(azimuths_b_to_a_degrees,0), 
                                                               number_of_antennas_in_selection)).astype(complex)
        
        # Process cell by cell 
        for cell_index in range (0, np.size(azimuths_b_to_a_degrees,1)): 
                        
            cell_ID = selected_cell_IDs[cell_index]
            
            mask_antennas_of_selected_cells = antenna_to_node_a_mapping == cell_ID
            
            # Calculate per cell
            array_steering_vector_bAnt_to_aAnt_complex[:,:,mask_antennas_of_selected_cells] =\
                self.calculate_array_steering_vector_per_cell(cell_index, 
                                                              np.radians(azimuths_b_to_a_degrees[:,cell_index]),
                                                              np.radians(zeniths_b_to_a_degrees[:,cell_index]) )
            
        return array_steering_vector_bAnt_to_aAnt_complex       


    def example_array_steering_vector(self, cell_index=0):
        
        # Set parameters for the example 
        azimuth_deg = np.linspace(-180, 180, 1000)
        zenith_deg = np.linspace(0, 180, 1000)
        
        alpha_mec_bearing_deg = self.alpha_mec_bearing_deg[cell_index]
        beta_mec_downtilt_deg = self.beta_mec_downtilt_deg[cell_index]
        beta_elec_downtilt_deg = self.beta_elec_downtilt_deg[cell_index]
        
        # Take number of antennas
        number_of_antennas = (self.antenna_LCS_index[self.antenna_to_node_a_mapping == cell_index,:].T).shape[1] 
        
        # Horizontal
        self.example_array_steering_vector_bAnt_to_aAnt_complex_H = np.zeros((1, np.size(azimuth_deg), number_of_antennas), dtype=np.csingle)
        self.example_array_steering_vector_bAnt_to_aAnt_complex_H[0,:,:] =\
            self.calculate_array_steering_vector_per_cell(cell_index, 
                                                          np.radians(azimuth_deg),
                                                          np.full_like(azimuth_deg, beta_elec_downtilt_deg),
                                                          is_example_flag=True) # Set zenith all equals to beta_elec_downtilt_deg
            
        # Vertical 
        # Calculate array factor for vertical antenna and strore it to use it later in precoding_gains.py
        self.example_array_steering_vector_bAnt_to_aAnt_complex_V = np.zeros((1, np.size(azimuth_deg), number_of_antennas), dtype=np.csingle)
        self.example_array_steering_vector_bAnt_to_aAnt_complex_V[0,:,:] =\
            self.calculate_array_steering_vector_per_cell(cell_index, 
                                                          np.full_like(zenith_deg, alpha_mec_bearing_deg),
                                                          np.radians(zenith_deg),
                                                          is_example_flag=True) # Set zenith all equals to alpha_mec_bearing_deg
            
