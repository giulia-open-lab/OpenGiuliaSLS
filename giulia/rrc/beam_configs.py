# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 09:46:46 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import ast
import sys
import time
from typing import List

import numpy as np
import pandas as pd
from scipy.linalg import dft

from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable

class Beam_Conf(Saveable):
    
    def __init__(self, simulation_config_obj, network_deployment_obj, node_type, beam_type):
       
       super().__init__()   
       
       ##### Plots 
       ########################
       self.plot = 0 # Switch to control plots if any
       
       ##### Input storage 
       ########################       
       self.simulation_config_obj = simulation_config_obj
       self.network_deployment_obj = network_deployment_obj
       self.node_type = node_type
       self.beam_type = beam_type
              
       ##### Outputs 
       ########################   
        
       # Placeholder to store array steering vector results. Size UE-by-antennas
       self.codebook_dictionary = {}
       self.codebook_index_to_node_mapping =  np.zeros(np.size(self.network_deployment_obj.df_ep,0), dtype=np.single)   
       
       self.df_beam_info = []
       self.beam_to_node_mapping = []
       self.beam_ID_within_host = []
       self.number_of_beams_per_node = None

       
    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["beam_to_node_mapping"]

       
    def process(self, rescheduling_us=-1): 
        
       # Process inputs
       self.antenna_pattern_models = self.network_deployment_obj.df_ep["antenna_pattern_model"].to_numpy() 
       self.precoder_type = self.beam_type + "_precoder"
       self.precoder_models = self.network_deployment_obj.df_ep[self.precoder_type].to_numpy()   
       
       # Start timer       
       t_start = time.perf_counter() 
        
       # Find the set of unique precoder models to process them independently
       precoder_set = set(self.precoder_models) 
        
       # Process each precoder model independently
       for precoder_model in precoder_set:
            
            # Identify nodes with the precoder model
            mask = precoder_model ==  self.precoder_models
            
            # Construct precoding codebooks according to types
            if precoder_model == "None" :   
                self.codebook_dictionary[len(self.codebook_dictionary)] = np.ones((1, 1)).astype(complex) 
                self.codebook_index_to_node_mapping[mask] = len(self.codebook_dictionary)-1
            
            elif precoder_model == "3GPPTR38_901_P1_single_column" :   
                # Creating beams, SSB or CSI-RS beams
                self.codebook_dictionary, self.codebook_index_to_node_mapping[mask] =\
                    self.create_codebook_dictionary_3GPPTR38_901_P1_single_column(
                        self.codebook_dictionary,
                        pd.concat([self.network_deployment_obj.df_ep.loc[mask, "dl_carrier_wavelength_m"], 
                                    self.network_deployment_obj.df_ep.loc[mask, "antenna_config_M"], 
                                    self.network_deployment_obj.df_ep.loc[mask, "antenna_config_ver_beta_elec_downtilt_deg"],
                                    self.network_deployment_obj.df_ep.loc[mask, "antenna_config_dv_m"]], 
                                axis=1)) 
                
            elif precoder_model == "3GPPTR38_901_P2_C1_one_port_per_panel" :  
                # Creating beams, SSB or CSI-RS beams
                self.codebook_dictionary, self.codebook_index_to_node_mapping[mask] =\
                    self.create_codebook_dictionary_3GPPTR38_901_P2_C1_one_port_per_panel(
                        self.codebook_dictionary,
                        pd.concat([self.network_deployment_obj.df_ep.loc[mask, "dl_carrier_wavelength_m"], 
                                   self.network_deployment_obj.df_ep.loc[mask, "antenna_config_M"], 
                                   self.network_deployment_obj.df_ep.loc[mask, "antenna_config_N"], 
                                   self.network_deployment_obj.df_ep.loc[mask, "antenna_config_Ng"], 
                                   self.network_deployment_obj.df_ep.loc[mask, "antenna_config_ver_beta_elec_downtilt_deg"],
                                   self.network_deployment_obj.df_ep.loc[mask, "antenna_config_dv_m"]], 
                                axis=1)) 
                    
            elif precoder_model == "DFT" :  
                # Creating beams, SSB or CSI-RS beams
                self.codebook_dictionary, self.codebook_index_to_node_mapping[mask] =\
                    self.create_codebook_dictionary_dft(
                        self.codebook_dictionary,
                        pd.concat([self.network_deployment_obj.df_ep.loc[mask, "antenna_config_Mg":"antenna_config_P"], 
                                   self.network_deployment_obj.df_ep.loc[mask, self.precoder_type : self.beam_type + "_number_of_beams_V"]], axis=1))  
                    
            elif precoder_model == "custom" :  
                # Creating beams, SSB or CSI-RS beams
                self.codebook_dictionary,\
                self.codebook_index_to_node_mapping[mask] =\
                    self.create_codebook_dictionary_customized(
                        self.codebook_dictionary,
                        pd.concat([self.network_deployment_obj.df_ep.loc[mask, "dl_carrier_wavelength_m"], 
                                   self.network_deployment_obj.df_ep.loc[mask, "antenna_config_M"], 
                                   self.network_deployment_obj.df_ep.loc[mask, "antenna_config_N"], 
                                   self.network_deployment_obj.df_ep.loc[mask, "antenna_config_dv_m"],
                                   self.network_deployment_obj.df_ep.loc[mask, "antenna_config_dh_m"],
                                   self.network_deployment_obj.df_ep.loc[mask, self.precoder_type : self.beam_type + "_number_of_beams_V"]], axis=1)) 
                    
            elif precoder_model == "custom_vMB" :  
                
                # Creating beams, SSB or CSI-RS beams
                 
                self.codebook_dictionary, self.codebook_index_to_node_mapping[mask] =\
                    self.create_codebook_dictionary_customized_vMB(
                        self.codebook_dictionary,
                        pd.concat([self.network_deployment_obj.df_ep.loc[mask, "dl_carrier_wavelength_m"], 
                                   self.network_deployment_obj.df_ep.loc[mask, "antenna_config_M"], 
                                   self.network_deployment_obj.df_ep.loc[mask, "antenna_config_N"], 
                                   self.network_deployment_obj.df_ep.loc[mask, "antenna_config_dv_m"],
                                   self.network_deployment_obj.df_ep.loc[mask, "antenna_config_dh_m"],
                                   self.network_deployment_obj.df_ep.loc[mask, self.precoder_type : self.beam_type + "_BeamDirection_theta_deg"]], axis=1)) 
                                
            else:
                np.disp("Error! This beam precoder format is not defined")                    
                
       # Storing useful information  
       self.df_beam_info, self.beam_to_node_mapping, self.beam_ID_within_host, self.number_of_beams_per_node =\
           self.store_beam_configuration(self.network_deployment_obj.df_ep, self.codebook_index_to_node_mapping, self.beam_type)  
           
       ##### End
       log_calculations_time(f'{self.beam_type} codebook', t_start)
       
       return rescheduling_us         
      
        
    def precoder_dft(self, M, N, number_of_beams_V, number_of_beams_H):
        
        # Calcualte DFT matrices
            # Note that the current strategy uses as many elements in H and V domain as H and V beams we want to create
            # Other strategies may apply, e.g. generate as many H and V beams as H and V elements and down select the wanted beams                
        dft_matrix_M = np.zeros((M,M)).astype(complex)
        dft_matrix_M[:number_of_beams_V,:number_of_beams_V] = 1/np.sqrt(number_of_beams_V)*dft(number_of_beams_V) #The predocer is normalized such that ||w||^2 = 1
        dft_matrix_N = np.zeros((N,N)).astype(complex)
        dft_matrix_N[:number_of_beams_H,:number_of_beams_H] = 1/np.sqrt(number_of_beams_H)*dft(number_of_beams_H) #The predocer is normalized such that ||w||^2 = 1    
                
        # Calcualte DFT precoder code book and store as many precoders as beams are indicated
        dft_precoders = np.kron(dft_matrix_M, dft_matrix_N) 
            
        return dft_precoders 
                  
    
    def create_codebook_dictionary_dft(self, codebook_dictionary, df_beam_related_info):
        
        # Find the set of unique DFT codebooks to be created and process them independently
            # Different codebooks are required according to different array capabilities
        
        unique_rows = df_beam_related_info.drop_duplicates()
        
        # Group by unique rows and retrieve indices
        index_groups = df_beam_related_info.groupby(list(unique_rows.columns)).groups
        
        # Process each DFT codebook independently
        codebook_index_to_node_mapping_aux = np.full(np.size(df_beam_related_info,0), -1, dtype=int)
        for unique_row, indices in index_groups.items():

            # Get necessary information of the identified DFT codebook
            M = unique_row[2] # Number of elements in vertical
            N = unique_row[3] # Number of elements in horizontal
            
            number_of_beams = unique_row[6]
            number_of_beams_H = unique_row[7]
            number_of_beams_V = unique_row[8]
            
            # Placeholder to store codebook. The codebook size is antenna elements x beams   
            beam_codebook = np.zeros((M*N, number_of_beams)).astype(complex)  
            
            # Calculate DFT matrices
            precoders = self.precoder_dft(M,N,number_of_beams_V,number_of_beams_H)
            beam_codebook = precoders[:,~np.all(precoders == 0, axis=0)] # Removing the null precoders   
                
            # Store results 
            if self.check_column_norms(beam_codebook):
                codebook_dictionary[len(codebook_dictionary)] = beam_codebook
                mask = (np.all(df_beam_related_info == unique_row,axis= 1)).values
                codebook_index_to_node_mapping_aux[mask] = len(codebook_dictionary)-1
                
            else:
                np.disp('Error: There is an issue with the codeword normalization')
                sys.exit(0)         
                
        return codebook_dictionary, codebook_index_to_node_mapping_aux
    
    
    def precoder_single_horizontal_beam(self, N, dl_carrier_wavelength_m, antenna_config_dh_m, alpha_elec_bearing_deg):
        
        alpha_elec_bearing_deg = np.atleast_1d(alpha_elec_bearing_deg)
        element_number = np.arange(0,N,1)
        
        return np.sqrt(1/N) * np.exp(-1j * (2*np.pi/dl_carrier_wavelength_m) * element_number[:, None] * antenna_config_dh_m * np.sin(np.radians(alpha_elec_bearing_deg[None, :]))) 
    

    def precoder_single_vertical_beam(self, M, dl_carrier_wavelength_m, antenna_config_dv_m, beta_elec_downtilt_deg):
        
        beta_elec_downtilt_deg = np.atleast_1d(beta_elec_downtilt_deg)
        element_number = np.arange(0,M,1)
        
        return np.sqrt(1/M) * np.exp(-1j * (2*np.pi/dl_carrier_wavelength_m) * element_number[:, None] * antenna_config_dv_m * np.cos(np.radians(beta_elec_downtilt_deg[None, :])))    
    

    def create_codebook_dictionary_3GPPTR38_901_P1_single_column(self, codebook_dictionary, df_beam_related_info):
        
        # Find the set of unique codebook models to process them independently
        unique_rows = df_beam_related_info.drop_duplicates()
        
        # Group by unique rows and retrieve indices
        index_groups = df_beam_related_info.groupby(list(unique_rows.columns)).groups
        
        # Process each codebook independently
        codebook_index_to_node_mapping_aux = np.zeros((np.size(df_beam_related_info,0))).astype(int) 
        for unique_row, indices in index_groups.items():
            
            # Get necessary information of the identified codebook
            dl_carrier_wavelength_m = unique_row[0]
            M = unique_row[1] 
            beta_elec_downtilt_deg = unique_row[2]
            antenna_config_dv_m = unique_row[3]
            
            # In this case, we have one beam 
            number_of_beams = 1
            
            # Placeholder to store the codebook. The codebook size is antenna elements x beams   
            beam_codebook = np.zeros((M, number_of_beams)).astype(complex)  
            
            # Calculate beamformer for column as per 7.3.1 of TR 38.901
            precoder = self.precoder_single_vertical_beam(M,dl_carrier_wavelength_m,antenna_config_dv_m,beta_elec_downtilt_deg)
            beam_codebook = precoder[:,~np.all(precoder == 0, axis=0)] # Removing the null precoders  
                
            # Store results   
            if self.check_column_norms(precoder):
                codebook_dictionary[len(codebook_dictionary)] = beam_codebook
                mask = (np.all(df_beam_related_info == unique_row,axis= 1)).values                  
                codebook_index_to_node_mapping_aux[mask] = len(codebook_dictionary)-1
                
            else:
                np.disp('Error: There is an issue with the codeword normalization')
                sys.exit(0)  
                
        return codebook_dictionary, codebook_index_to_node_mapping_aux              


    def create_codebook_dictionary_3GPPTR38_901_P2_C1_one_port_per_panel(self, codebook_dictionary, df_beam_related_info):
        
        # Find the set of unique codebook models to process them independently
        unique_rows = df_beam_related_info.drop_duplicates()
        
        # Group by unique rows and retrieve indices
        index_groups = df_beam_related_info.groupby(list(unique_rows.columns)).groups
        
        # Process each codebook independently
        codebook_index_to_node_mapping_aux = np.zeros((np.size(df_beam_related_info,0))).astype(int) 
        for unique_row, indices in index_groups.items():
            
            # Get necessary information of the identified codebook
            dl_carrier_wavelength_m = unique_row[0]
            M = unique_row[1] 
            N = unique_row[2] 
            Ng = unique_row[3] 
            beta_elec_downtilt_deg = unique_row[4]
            antenna_config_dv_m = unique_row[5]
            
            # In this case, we have a beam per pannel
            number_of_beams = Ng
            
            # Placeholder to store the codebook. The codebook size is antenna elements x beams   
            beam_codebook = np.zeros((M*N*Ng, number_of_beams)).astype(complex)  
            
            # Calculate beamformer for a column as per 7.3.1 of TR 38.901
            precoder = self.precoder_single_vertical_beam(M,dl_carrier_wavelength_m,antenna_config_dv_m,beta_elec_downtilt_deg)
            
            # Repeat the precoder for all columns of a pannel
            precoder = np.sqrt(1/N) * np.repeat(precoder, N)
            
            # Extend the precoder to each pannel
            for i in range(0,number_of_beams):
                beam_codebook[i*M*N : i*M*N+M*N, i] = precoder     
            beam_codebook = beam_codebook[:,~np.all(beam_codebook == 0, axis=0)] # Removing the null precoders  
                
            # Store results   
            if self.check_column_norms(beam_codebook):
                codebook_dictionary[len(codebook_dictionary)] = beam_codebook
                mask = (np.all(df_beam_related_info == unique_row,axis= 1)).values                  
                codebook_index_to_node_mapping_aux[mask] = len(codebook_dictionary)-1
                
            else:
                np.disp('Error: There is an issue with the codeword normalization')
                sys.exit(0)            
                
        return codebook_dictionary, codebook_index_to_node_mapping_aux    


    def create_codebook_dictionary_customized(self, codebook_dictionary, df_beam_related_info):
        
        # Find the set of unique codebook models to process them independently
        unique_rows = df_beam_related_info.drop_duplicates()
        
        # Group by unique rows and retrieve indices
        index_groups = df_beam_related_info.groupby(list(unique_rows.columns)).groups
        
        # Process each codebook independently
        codebook_index_to_node_mapping_aux = np.zeros((np.size(df_beam_related_info,0))).astype(int) 
        for unique_row, indices in index_groups.items():

            # Get necessary information of the identified codebook
            dl_carrier_wavelength_m = unique_row[0]
            M = unique_row[1] 
            N = unique_row[2]  
            antenna_config_dv_m = unique_row[3]
            antenna_config_dh_m = unique_row[4]
            
            number_of_beams = unique_row[6]
            number_of_beams_H = unique_row[7]
            number_of_beams_V = unique_row[8]
            
            # Preset directions of beams
            # Horizontal coverage is 120 degrees, we want to center the beams within this range
            # Compute the horizontal beam angles centered around 0 degrees
            horizontal_coverage = 120  # degrees
            alpha_elec_bearing_deg = np.linspace(-horizontal_coverage/2+horizontal_coverage/number_of_beams_H/2, 
                                                 horizontal_coverage/2-horizontal_coverage/number_of_beams_H/2, number_of_beams_H)  
            
            # Vertical coverage is 90 degrees from horizon down (0 to -90 degrees)
            # Compute the vertical beam angles starting from the horizon (0 degrees) down to -90 degrees
            vertical_coverage = 90  # degrees
            beta_elec_downtilt_deg = (90+90+vertical_coverage)/2 + np.linspace(-vertical_coverage/2+vertical_coverage/number_of_beams_V/2, 
                                                                                vertical_coverage/2-vertical_coverage/number_of_beams_V/2, number_of_beams_V)  
             
            # Placeholder to store the codebook. The codebook size is antenna elements x beams   
            beam_codebook = np.zeros((M*N, number_of_beams)).astype(complex)  
            
            # Calculate code book 
            for codeword_H_index in range(0,number_of_beams_H): 
                
                # Get the precoder for the row
                precoder_N = self.precoder_single_horizontal_beam(N,dl_carrier_wavelength_m,antenna_config_dh_m,alpha_elec_bearing_deg[codeword_H_index])
                
                for codeword_V_index in range(0,number_of_beams_V):
                    
                    # Get the precoder for the column
                    precoder_M = self.precoder_single_vertical_beam(M,dl_carrier_wavelength_m,antenna_config_dv_m,beta_elec_downtilt_deg[codeword_V_index]) 
                    
                    # Perform multiplication 
                    precoder = np.matmul(np.expand_dims(precoder_M, axis=1), np.expand_dims(precoder_N, axis=0)) # Calculate precoding matrix    
                    
                    # Store code word in code book
                    beam_codebook[:,codeword_H_index * number_of_beams_V + codeword_V_index] = precoder.flatten()
              
            beam_codebook = beam_codebook[:,~np.all(beam_codebook == 0, axis=0)] # Removing the null precoders 
            
            # Store results   
            if self.check_column_norms(beam_codebook):
                codebook_dictionary[len(codebook_dictionary)] = beam_codebook
                mask = (np.all(df_beam_related_info == unique_row,axis= 1)).values                  
                codebook_index_to_node_mapping_aux[mask] = len(codebook_dictionary)-1
                
            else:
                np.disp('Error: There is an issue with the codeword normalization')
                sys.exit(0)
                    
        return codebook_dictionary, codebook_index_to_node_mapping_aux     
    
    
    def create_codebook_dictionary_customized_vMB(self, codebook_dictionary, df_beam_related_info):

        # Convert list columns to tuples for groupby operation
        # This is necessary because pandas groupby cannot handle list objects directly
        if self.beam_type == "SSB":
            cols_with_lists = ['SSB_BeamDirection_phi_deg', 'SSB_BeamDirection_theta_deg']  # aggiungi altre se serve
        elif self.beam_type == "CSI_RS":
            cols_with_lists = ['CSI_RS_BeamDirection_phi_deg', 'CSI_RS_BeamDirection_theta_deg']  # aggiungi altre se serve
        for c in cols_with_lists:
            df_beam_related_info[c] = df_beam_related_info[c].apply(lambda x: tuple(x) if isinstance(x, list) else x)


        # Find the set of unique codebook models to process them independently
        unique_rows = df_beam_related_info.drop_duplicates()
        
        # Group by unique rows and retrieve indices
        index_groups = df_beam_related_info.groupby(list(unique_rows.columns)).groups
        
        # Process each codebook independently
        codebook_index_to_node_mapping_aux = np.zeros((np.size(df_beam_related_info,0))).astype(int) 
        for unique_row, _ in index_groups.items():
            
            # Get necessary information of the identified codebook
            dl_carrier_wavelength_m = unique_row[0]
            antenna_Mv = M = unique_row[1] 
            antenna_Mh = N = unique_row[2]  
            antenna_config_dv_m = unique_row[3]
            antenna_config_dh_m = unique_row[4]
            
            # Note that the beamDirection_phi_deg, beamDirection_theta_deg
            # Contains the direction of each beams, considering the local system of the panel
            number_of_beams = unique_row[6]
            beamDirection_phi_deg = np.nan if unique_row[9]=='[nan]' else np.array(ast.literal_eval(unique_row[9]), dtype=float)
            beamDirection_theta_deg = np.nan if unique_row[10]=='[nan]' else np.array(ast.literal_eval(unique_row[10]), dtype=float)
            
            assert beamDirection_phi_deg is not np.nan and beamDirection_theta_deg is not np.nan, \
                "Error! (beamDirection_phi_deg, beamDirection_theta_deg) has to be defined."
            
            assert len(beamDirection_phi_deg)==number_of_beams, \
                f"Error: BeamDirection_phi_deg has to have {number_of_beams} entries, as specified by 'number_of_beams' variable.\
                The vector has to contain the direction of all to create beams"
            
            assert len(beamDirection_theta_deg)==number_of_beams, \
                f"Error:  BeamDirection_theta_deg has to have {number_of_beams} entries, as specified by 'number_of_beams' variable.\
                The vector has to contain the direction of all to create beams"

            #--- Horizontal
            AFh_MhxNword = self.precoder_single_horizontal_beam(N, dl_carrier_wavelength_m, antenna_config_dh_m, alpha_elec_bearing_deg=beamDirection_phi_deg)
            AFh_MxNword = np.tile(AFh_MhxNword, (antenna_Mv, 1))
            
            #--- Vertical
            AFv_MvxNword = self.precoder_single_vertical_beam(M, dl_carrier_wavelength_m, antenna_config_dv_m, beta_elec_downtilt_deg=beamDirection_theta_deg)
            AFv_MxNword = np.repeat(AFv_MvxNword, antenna_Mh, axis=0)
            
            #--- Compute Total
            codebook_MxNword =  AFv_MxNword * AFh_MxNword
            
            assert self.check_column_norms(codebook_MxNword), \
                'Error: There is an issue with the codeword normalization'
            
            codebook_dictionary[len(codebook_dictionary)] = codebook_MxNword
            mask = (np.all(df_beam_related_info == unique_row,axis= 1)).values                  
            codebook_index_to_node_mapping_aux[mask] = len(codebook_dictionary)-1
            
        return codebook_dictionary, codebook_index_to_node_mapping_aux    
    
    
    def store_beam_configuration(self, df_ep, codebook_to_node_mapping, beam_type):
        
        # Get input data
        number_of_beams_per_node = df_ep[beam_type + "_number_of_beams"].to_numpy(dtype=int)
        
        # Calculate total number of beams in the network
        number_of_beams_in_network = np.sum(number_of_beams_per_node)
        
        # Set ID per beam (unique in the network)
        beam_ID = np.arange(0, number_of_beams_in_network)
        name = ["beam_" + str(beam_ID[i]) for i in beam_ID]

        # Stablish mapping between general ID of the beam in the network and its node ID
        beam_to_node_mapping = np.fromiter([ node_index  for node_index in range(0,np.size(number_of_beams_per_node)) for beam_index in range(0,number_of_beams_per_node[node_index]) ], int)
       
        # Stablish mapping between the general ID of the beam in the network and the local ID of the beam in its node
        beam_ID_in_host_node = np.fromiter([ beam_index  for node_index in range(0,np.size(number_of_beams_per_node)) for beam_index in range(0,number_of_beams_per_node[node_index]) ], int) 
        
        # Indicate to which codebook it belong
        beam_codebook = np.fromiter([ codebook_to_node_mapping[node_index]  for node_index in range(0,np.size(number_of_beams_per_node)) for beam_index in range(0,number_of_beams_per_node[node_index]) ], int)
        
        # Indicate the entry of the codebook
        # Note that beam 1 is mapped to code book entry 1, beam 2 is mapped to code book entry 2. 
        # This mapping should not be broken. The simulator will not respond to a change in this mapping. There is no need to brake it.
        beam_codebook_entry = beam_ID_in_host_node

        # Create dataframe
        beams_info = {"ID": beam_ID, 
                      "name": name,  
                      "host_node_ID": beam_to_node_mapping, 
                      "ID_in_host_node": beam_ID_in_host_node, 
                      "codebook": beam_codebook,  
                      "codebook_entry": beam_codebook_entry}
        df_beam_info = pd.DataFrame(beams_info)
        
        return df_beam_info, beam_to_node_mapping, beam_ID_in_host_node, number_of_beams_per_node  


    def check_column_norms(self, array):
        
        # Calculate the norm of each column
        column_norms = np.linalg.norm(array, axis=0)
        
        # Check if all column norms are 1
        if np.allclose(column_norms, 1):
            return True
        else:
            return False


class Beam_Conf_SSB(Beam_Conf):
    pass

class Beam_Conf_CSI_RS(Beam_Conf):
    pass