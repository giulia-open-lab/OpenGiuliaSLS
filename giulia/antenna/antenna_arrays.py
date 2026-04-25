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

import sys
import time
from typing import List

import numpy as np
import pandas as pd
import sionna

from giulia.event_driven.snapshot_control import Snapshot_control
from giulia.fs import results_file
from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable


class Antenna_Array(Saveable):

    def __init__(self,
                 simulation_config_obj,
                 site_deployment_obj,
                 deployment_obj,
                 node_type):
       
        super().__init__()
        
        ##### Plots
        ########################
        self.plot = 0 # Switch to control plots if any


        ##### Input storage
        ########################
        self.simulation_config_obj = simulation_config_obj
        self.site_deployment_obj = site_deployment_obj
        self.deployment_obj = deployment_obj
        self.node_type = node_type


        ##### Outputs
        ########################
        self.df_ep = []
        
        self.antenna_to_node_mapping = []

        # Sionna related
        self.antenna_array_types = [] # Antenna array type list
        self.antenna_array_type_index = [] # Indeces indicating the antenna array type of each node
        self.antenna_array_carrier_frequency_GHz = [] # Carrier frequency of each antenna array type


    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["antenna_to_node_mapping"]


    def process(self, rescheduling_us=-1):
        
        ##### Process inputs
        ######################## 
        
        # Simulation
        self.project_name = self.simulation_config_obj.project_name
        self.sn_indicator = self.simulation_config_obj.sn_indicator        
        
        # Nodes
        self.node_positions_m = self.deployment_obj.df_ep[["position_x_m", "position_y_m", "position_z_m"]].to_numpy(dtype=np.single)
        self.indoor = self.deployment_obj.df_ep[["indoor"]].to_numpy(dtype=bool)
        self.dl_carrier_frequency_GHz = self.deployment_obj.df_ep["dl_carrier_frequency_GHz"].to_numpy()
        self.dl_carrier_wavelength_m = self.deployment_obj.df_ep["dl_carrier_wavelength_m"].to_numpy()

        # Antenna model

        self.bs_antenna_models = self.deployment_obj.df_ep["antenna_pattern_model"].to_numpy()

        # Antenna element information
        self.antenna_config_max_gain_dBi = self.deployment_obj.df_ep["antenna_config_max_gain_dBi"].to_numpy(dtype=np.single)

        self.antenna_config_hor_phi_3dB_deg = self.deployment_obj.df_ep["antenna_config_hor_phi_3dB_deg"].to_numpy(dtype=np.single)
        self.antenna_config_hor_A_m_dB = self.deployment_obj.df_ep["antenna_config_hor_A_m_dB"].to_numpy(dtype=np.single)

        self.antenna_config_ver_theta_3dB_deg = self.deployment_obj.df_ep["antenna_config_ver_theta_3dB_deg"].to_numpy(dtype=np.single)
        self.antenna_config_ver_SLA_dB = self.deployment_obj.df_ep["antenna_config_ver_SLA_dB"].to_numpy(dtype=np.single)

        # Antenna array information
        self.antenna_config_Mg = self.deployment_obj.df_ep["antenna_config_Mg"].to_numpy(dtype=int)
        self.antenna_config_Ng = self.deployment_obj.df_ep["antenna_config_Ng"].to_numpy(dtype=int)
        self.antenna_config_M = self.deployment_obj.df_ep["antenna_config_M"].to_numpy(dtype=int) # vertical
        self.antenna_config_N = self.deployment_obj.df_ep["antenna_config_N"].to_numpy(dtype=int) # horizontal
        self.antenna_config_P = self.deployment_obj.df_ep["antenna_config_P"].to_numpy(dtype=str)
        self.antenna_config_P_type = self.deployment_obj.df_ep["antenna_config_P_type"].to_numpy(dtype=str)
        self.antenna_config_number_of_elements = self.deployment_obj.df_ep["antenna_config_number_of_elements"].to_numpy(dtype=int)

        self.antenna_config_dgv_m = self.deployment_obj.df_ep["antenna_config_dgv_m"].to_numpy(dtype=np.single)
        self.antenna_config_dgh_m = self.deployment_obj.df_ep["antenna_config_dgh_m"].to_numpy(dtype=np.single)
        self.antenna_config_dv_m = self.deployment_obj.df_ep["antenna_config_dv_m"].to_numpy(dtype=np.single)
        self.antenna_config_dh_m = self.deployment_obj.df_ep["antenna_config_dh_m"].to_numpy(dtype=np.single)

        self.antenna_config_hor_alpha_mec_bearing_deg = self.deployment_obj.df_ep["antenna_config_hor_alpha_mec_bearing_deg"].to_numpy(dtype=np.single)
        self.antenna_config_ver_beta_mec_downtilt_deg = self.deployment_obj.df_ep["antenna_config_ver_beta_mec_downtilt_deg"].to_numpy(dtype=np.single)
        self.antenna_config_gamma_mec_slant_deg = self.deployment_obj.df_ep["antenna_config_gamma_mec_slant_deg"].to_numpy(dtype=np.single)        
        
        
        ##### Start timer
        ########################        
        t_start = time.perf_counter()


        ##### Calculate the position of each antenna element in the local coordinate system (LCS) as well as antenna element to node mapping vector and data frame
        ########################
        self.df_ep,\
        self.antenna_to_node_mapping =\
           self.calculate_antenna_element_LCS_positions(self.node_type,
                                                        np.size(self.antenna_config_number_of_elements ,0),
                                                        sum(self.antenna_config_number_of_elements),
                                                        self.dl_carrier_frequency_GHz,
                                                        self.dl_carrier_wavelength_m,
                                                        self.antenna_config_Mg,
                                                        self.antenna_config_Ng,
                                                        self.antenna_config_M,
                                                        self.antenna_config_N,
                                                        self.antenna_config_P,
                                                        self.antenna_config_P_type,
                                                        self.antenna_config_dgv_m,
                                                        self.antenna_config_dgh_m,
                                                        self.antenna_config_dv_m,
                                                        self.antenna_config_dh_m)


        ##### Calculate the position of each antenna element in the global coordinate system (GCS)
        ########################
        antenna_element_GCS_position_m = self.calculate_antenna_element_GCS_positions(self.node_type,
                                                                                     self.node_positions_m,
                                                                                     self.antenna_config_hor_alpha_mec_bearing_deg,
                                                                                     self.antenna_config_ver_beta_mec_downtilt_deg,
                                                                                     self.antenna_config_gamma_mec_slant_deg,
                                                                                     self.antenna_to_node_mapping,
                                                                                     self.df_ep.loc[:,"LCS_position_x_m":"LCS_position_z_m"].to_numpy())


        ##### Add the GCS locations to the df_antenna_arrays dataframe
        ########################
        self.df_ep['position_x_m'] = antenna_element_GCS_position_m[:,0]
        self.df_ep['position_y_m'] = antenna_element_GCS_position_m[:,1]
        self.df_ep['position_z_m'] = antenna_element_GCS_position_m[:,2]

        ### If Sionna is to be used, derive unique antenna array types and the indeces indicating the antenna array type of each node
        if self.sn_indicator == 1:

            self.antenna_array_types, self.antenna_array_type_index, self.antenna_array_carrier_frequency_GHz = \
                self.derive_unique_antenna_array_types(self.site_deployment_obj, self.deployment_obj)

            # Adding the indeces of the antenna array to the df_ep
            self.deployment_obj.df_ep['antenna_array_type_index'] = self.antenna_array_type_index


        ##### Save to plot
        ########################
        snapshot_control = Snapshot_control.get_instance()
        if self.simulation_config_obj.plot == 1 and snapshot_control.num_snapshots == 0:
            file_name = results_file(self.project_name, 'to_plot_' + self.node_type + '_antenna_arrays')
            np.savez(file_name, antenna_to_node_mapping=self.antenna_to_node_mapping, antenna_element_GCS_position_m=antenna_element_GCS_position_m)


        ##### End
        ########################
        log_calculations_time(f'{self.node_type} antenna array model', t_start)

        return rescheduling_us


    def basic_checks(self):

        if np.any(self.site_deployment_obj.df_ep["antenna_config_Mg"].to_numpy() > 1 ):
            sys.exit("Error We are not ready to process multipanel yet!")

        if np.any(self.site_deployment_obj.df_ep["antenna_config_Ng"].to_numpy() > 1 ):
            sys.exit("Error We are not ready to process multipanel yet!")

              
    def calculate_antenna_element_LCS_positions(self,
                                                node_type,
                                                number_of_nodes,
                                                number_of_elements,
                                                carrier_frequency_GHz,
                                                carrier_wavelength_m,
                                                Mg,
                                                Ng,
                                                M,
                                                N,
                                                P,
                                                P_type,
                                                dgv_m,
                                                dgh_m,
                                                dv_m,
                                                dh_m):

        # Initialise antenna element LCS indeces and positions
            # Note that the indeces matrix contains the numbering of the antenna elements in the uniform array.
            # The numbering is as per 3GPP TR 38.901 with (y,z) = (0,0) for the bottom left element.
            # Then, we grow in the y-axis and then in the z-axis
            # x is always equal to zero, as the reference array lays on the y-z plane
            # Same logic applies for the positions

        antenna_element_LCS_index  = np.zeros((number_of_elements,3), dtype=int)
        antenna_element_LCS_position_m = np.zeros((number_of_elements,3), dtype=np.single)

        # Other relevant variables
        ant_ID = np.zeros((number_of_elements,)).astype(int)
        ant_name = []
        antenna_to_node_mapping = np.zeros((number_of_elements,), dtype=int)
        ant_ID_in_node = np.zeros((number_of_elements,), dtype=int)
        Mg_vector = np.zeros((number_of_elements,), dtype=int)
        Ng_vector = np.zeros((number_of_elements,), dtype=int)
        M_vector = np.zeros((number_of_elements,), dtype=int)
        N_vector = np.zeros((number_of_elements,), dtype=int)
        P_type_vector = np.zeros((number_of_elements,), dtype=str)

        # Calculate LCS position of the antenna element in the left upper corner
        lower_left_corner_element_y_poistion =  (-((N-1)*dh_m*Ng + (Ng-1)*dgh_m) / 2).astype(np.single)
        lower_left_corner_element_z_poistion =  (-((M-1)*dv_m*Mg + (Mg-1)*dgv_m) / 2).astype(np.single)

        # Calculate LCS position of all the other antenna elements
        index = 0
        for node_index in range (0, number_of_nodes):

            ant_index_in_node = 0

            # For the moment, dual polarization is only considered when using TR38.901.
            # We generate the channel for the 2 polarizations using Sionna and sum the two channels per antenna element to get just one channel
            # See line h_freq = tf.reduce_sum(h_freq, axis=1) in channels.py
            if P[node_index] == 'single' or P[node_index] == 'dual':
                P_num = 1
            else:
                P_num = 2

            for Mg_index in range(0,Mg[node_index]):
                for Ng_index in range(0,Ng[node_index]):
                    for P_index in range(0,P_num):
                        for M_index in range(0,M[node_index]):
                            for N_index in range(0,N[node_index]):

                                ant_ID[index] = index
                                ant_name.append([node_type + "_" + str(node_index) + "_ant_" + str(ant_index_in_node)])
                                antenna_to_node_mapping[index] = node_index
                                ant_ID_in_node[index] = ant_index_in_node
                                Mg_vector[index] = Mg_index
                                Ng_vector[index] = Ng_index
                                M_vector[index] = M_index
                                N_vector[index] = N_index
                                P_type_vector[index] = P_index

                                antenna_element_LCS_index[index] = [0, N_index, M_index]
                                antenna_element_LCS_position_m[index,1] = \
                                    lower_left_corner_element_y_poistion[node_index] + (Ng_index*(N[node_index]-1) + N_index) * dh_m[node_index] + Ng_index*dgh_m[node_index]
                                antenna_element_LCS_position_m[index,2] = \
                                    lower_left_corner_element_z_poistion[node_index] + (Mg_index*(M[node_index]-1) + M_index) * dv_m[node_index] + Mg_index*dgv_m[node_index]

                                index += 1
                                ant_index_in_node += 1

        #Store calculations in data frame
        d = {"ID": ant_ID, "name": ant_name, "host_node_ID": antenna_to_node_mapping, "ID_in_host_node": ant_ID_in_node,  
             "Mg": Mg_vector, "Ng": Ng_vector, "M": M_vector, "N": N_vector, 
             "P_type": P_type_vector, "LCS_index_x": antenna_element_LCS_index[:,0], 
             "LCS_index_y": antenna_element_LCS_index[:,1], "LCS_index_z": antenna_element_LCS_index[:,2], 
             "LCS_position_x_m": antenna_element_LCS_position_m[:,0], "LCS_position_y_m": antenna_element_LCS_position_m[:,1], "LCS_position_z_m": antenna_element_LCS_position_m[:,2]}
        df_antenna_arrays_LCS = pd.DataFrame(d)

        return df_antenna_arrays_LCS, antenna_to_node_mapping


    def calculate_antenna_element_GCS_positions(self,
                                                node_type: str,
                                                node_positions_m: np.ndarray, 
                                                alpha_bearing_deg: np.ndarray, 
                                                beta_downtilt_deg: np.ndarray, 
                                                gamma_slant_deg: np.ndarray,
                                                antenna_to_node_mapping: np.ndarray, 
                                                antenna_element_LCS_position_m: np.ndarray) ->  np.ndarray:  
        

        # Get LCS coordinates xyz
        x_antenna_pos_m_lcs = antenna_element_LCS_position_m[:, 0]
        y_antenna_pos_m_lcs = antenna_element_LCS_position_m[:, 1]
        z_antenna_pos_m_lcs = antenna_element_LCS_position_m[:, 2]
        

        # Convert angle from degree to radian
        # degrees -> radians
        α_arr = np.deg2rad(alpha_bearing_deg)
        β_arr = np.deg2rad(beta_downtilt_deg)
        γ_arr = np.deg2rad(gamma_slant_deg)


        # Checks of the shape of the angles vs antenna to node mapping. This ensure that each node has one set of angles
        assert np.array_equal(np.arange(α_arr.shape[0]), np.unique(antenna_to_node_mapping)),  "Error in bearing angles shape vs antenna to node mapping"
        assert np.array_equal(np.arange(β_arr.shape[0]), np.unique(antenna_to_node_mapping)),  "Error in downtilt angles shape vs antenna to node mapping"
        assert np.array_equal(np.arange(γ_arr.shape[0]), np.unique(antenna_to_node_mapping)),  "Error in slant angles shape vs antenna to node mapping"
        assert α_arr.shape[0] == node_positions_m.shape[0],  "Error in downtilt angles shape vs node positions shape"
        assert antenna_to_node_mapping.shape[0] == antenna_element_LCS_position_m.shape[0], "Error: antenna_to_node_mapping length vs number of antenna elements"

        # Define Rotation Matrix 
        def Rz(a):
            return np.array([[np.cos(a), -np.sin(a), 0],
                             [np.sin(a),  np.cos(a), 0],
                             [0,          0,         1]], dtype=np.float64)

        def Ry(b):
            return np.array([[ np.cos(b),  0, np.sin(b)],
                             [ 0,          1, 0        ],
                             [-np.sin(b),  0, np.cos(b)]], dtype=np.float64)

        def Rx(c):
            return np.array([[1,             0,           0     ],
                             [0,         np.cos(c),   -np.sin(c)],
                             [0,         np.sin(c),    np.cos(c)]], dtype=np.float64)

        

        # Initialize GCS coordinates
        x_antenna_pos_m_gcs = np.zeros_like(x_antenna_pos_m_lcs)
        y_antenna_pos_m_gcs = np.zeros_like(y_antenna_pos_m_lcs)
        z_antenna_pos_m_gcs = np.zeros_like(z_antenna_pos_m_lcs)

        # Apply rotation for each node
        for node_idx, (α, β, γ) in enumerate(zip(α_arr, β_arr, γ_arr)):
            # Get antennas of the node
            mask_antennas_of_selected_cells = antenna_to_node_mapping == node_idx

            # Get LCS coordinates of the antennas of the node
            x_pos_lcs_m = x_antenna_pos_m_lcs[mask_antennas_of_selected_cells]
            y_pos_lcs_m = y_antenna_pos_m_lcs[mask_antennas_of_selected_cells]
            z_pos_lcs_m = z_antenna_pos_m_lcs[mask_antennas_of_selected_cells]

            # Forward per 38.901 (7.1-2): R = Rz(α) Ry(β_eff) Rx(γ)
            ## Apply downtilt correction for in Giulia values definition
            β_eff = β - (np.pi / 2)
            R_fwd = Rz(α) @ Ry(β_eff) @ Rx(γ)
            
            # Create unit vector in LCS
            ρ_lcs = np.vstack((x_pos_lcs_m, y_pos_lcs_m, z_pos_lcs_m))  # shape (3, num_antennas)
            
            # Apply rotation to get GCS coordinates
            ρ_gcs = R_fwd @ ρ_lcs  # shape (3, num_antennas)

            # Store the rotated coordinates back into the GCS arrays
            x_antenna_pos_m_gcs[mask_antennas_of_selected_cells] = ρ_gcs[0, :] + node_positions_m[node_idx, 0]
            y_antenna_pos_m_gcs[mask_antennas_of_selected_cells] = ρ_gcs[1, :] + node_positions_m[node_idx, 1]
            z_antenna_pos_m_gcs[mask_antennas_of_selected_cells] = ρ_gcs[2, :] + node_positions_m[node_idx, 2]

        # Create final GCS position array
        antenna_element_GCS_position_m = np.vstack((x_antenna_pos_m_gcs, y_antenna_pos_m_gcs, z_antenna_pos_m_gcs)).T  # shape (num_antennas, 3)


        # Quantize GCS coordinates to ensure numerical stability.
        # This removes small floating-point discrepancies (~1e-9) caused by
        # trigonometric functions and matrix operations, so nearly identical
        # positions collapse to the same value.
        tol = 1e-6
        antenna_element_GCS_position_m = np.round(antenna_element_GCS_position_m / tol) * tol

        return antenna_element_GCS_position_m

    
    def derive_unique_antenna_array_types(self, site_deployment_obj, deployment_obj):
       
        # Antenna array type list
        antenna_array_types = []
        
        # Indeces indicating the antenna array type of each node 
        antenna_array_type_index = np.ones(len(deployment_obj.df_ep)).astype(int)     
        
        # Carrier frequency of each array type
        antenna_array_dl_carrier_frequency_GHz = []
        
        # Group by antenna array characterstics to find the different pannels
        grouped = deployment_obj.df_ep.groupby(['dl_carrier_frequency_GHz', #0
                                 'dl_carrier_wavelength_m', #1
                                 # 'antenna_pattern_model', #2
                                 # 'antenna_config_max_gain_dBi', #3
                                 # 'antenna_config_hor_phi_3dB_deg', #4
                                 # 'antenna_config_hor_A_m_dB', #5
                                 # 'antenna_config_ver_theta_3dB_deg', #6
                                 # 'antenna_config_ver_SLA_dB', #7
                                 'antenna_config_Mg', #8 2
                                 'antenna_config_Ng', #9 3
                                 'antenna_config_M', #10 4
                                 'antenna_config_N', #11 5
                                 'antenna_config_P', #12 6
                                 'antenna_config_P_type', #13 7
                                 'antenna_config_number_of_elements', #14 8
                                 'antenna_config_dgv_m', #15 9
                                 'antenna_config_dgh_m', #16 10
                                 'antenna_config_dv_m', #17 11
                                 'antenna_config_dh_m', #18 12
                                 'antenna_pattern_model']) #19 13
        
        # Create and store objects
        index = 0
        for model_params, group in grouped:
            antenna_array_type_index[group['ID']] = index
            
            if model_params[2] == 1 : 
                panel_vertical_spacing_aux = None 
            else :
                panel_vertical_spacing_aux =  model_params[9]/model_params[1]
            
            if model_params[3] == 1 : 
                panel_horizontal_spacing_aux = None 
            else :
                panel_horizontal_spacing_aux =  model_params[10]/model_params[1]   
                
            if model_params[11] == 0:
                element_vertical_spacing_aux = None
            else:
                element_vertical_spacing_aux = model_params[11]/model_params[1]
                
            if model_params[12] == 0:
                element_horizontal_spacing_aux = None
            else:
                element_horizontal_spacing_aux = model_params[12]/model_params[1]

            ####################################################################
            if model_params[13] == '3GPPTR38_901' or model_params[13] == '3GPPTR36_814_UMa':

                unique_antenna_array = sionna.phy.channel.tr38901.PanelArray(num_rows_per_panel = model_params[4],
                                                                         num_cols_per_panel = model_params[5],
                                                                         polarization = model_params[6],
                                                                         polarization_type = model_params[7],
                                                                         antenna_pattern = '38.901',
                                                                         carrier_frequency = model_params[0] * 1e9,
                                                                         num_rows = model_params[2],
                                                                         num_cols = model_params[3],
                                                                         panel_vertical_spacing = panel_vertical_spacing_aux,
                                                                         panel_horizontal_spacing = panel_horizontal_spacing_aux,
                                                                         element_vertical_spacing = element_vertical_spacing_aux,
                                                                         element_horizontal_spacing = element_horizontal_spacing_aux)

            ####################################################################
            elif model_params[13] == 'omnidirectional':
                unique_antenna_array = sionna.phy.channel.tr38901.PanelArray(num_rows_per_panel=model_params[4],
                                                                         num_cols_per_panel=model_params[5],
                                                                         polarization=model_params[6],
                                                                         polarization_type=model_params[7],
                                                                         antenna_pattern='omni',
                                                                         carrier_frequency=model_params[0] * 1e9,
                                                                         num_rows=model_params[2],
                                                                         num_cols=model_params[3],
                                                                         panel_vertical_spacing=panel_vertical_spacing_aux,
                                                                         panel_horizontal_spacing=panel_horizontal_spacing_aux,
                                                                         element_vertical_spacing=element_vertical_spacing_aux,
                                                                         element_horizontal_spacing=element_horizontal_spacing_aux)

            ####################################################################
            elif model_params[13] == 'Ray_tracing':

                scene = site_deployment_obj.scene
                scene.frequency = model_params[0] * 1e9 # It must be in Hz
                if self.node_type == 'cell':
                    antenna_pattern_model = 'tr38901'
                elif self.node_type == 'ue':
                    antenna_pattern_model = 'iso'
                else:
                    raise ValueError("self.node_type must be either 'cell' or 'ue.")
                unique_antenna_array = sionna.phy.antennas.PlanarArray(num_rows=model_params[4],
                                                             num_cols=model_params[5],
                                                             vertical_spacing=element_vertical_spacing_aux,
                                                             horizontal_spacing=element_horizontal_spacing_aux,
                                                             pattern=antenna_pattern_model,
                                                             polarization=model_params[7])

            else:
                raise ValueError("Unsupported entry for 'antenna_pattern_model'. It must be '3GPPTR38_901', '3GPPTR36_814' or 'Ray_tracing'.")

            ####################################################################
            antenna_array_types.append(unique_antenna_array)
            antenna_array_dl_carrier_frequency_GHz.append(model_params[0])
            
            index += 1  
            
        return antenna_array_types, antenna_array_type_index, antenna_array_dl_carrier_frequency_GHz    
        
       
class Antenna_Array_BS(Antenna_Array):
    pass

class Antenna_Array_UE(Antenna_Array):
    pass    