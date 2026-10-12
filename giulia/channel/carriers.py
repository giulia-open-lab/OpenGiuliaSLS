# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:19:28 2023

@author: dalopezperez
"""

import numpy as np

class Carrier:
    
    def __init__(self, 
                 is_dl, 
                 cell_IDs, 
                 transmit_antennas, #receive_antennas,
                 fdd_tdd_ind,
                 subframe_assignment,
                 FR,
                 subcarriers_per_PRB,
                 ofdm_symbols_in_slot,
                 #carrier_frequency_GHz, 
                 #carrier_wavelength_m, 
                 bandwidth_MHz, 
                 available_PRBs, 
                 dl_subcarrier_spacing_kHz,
                 ofdm_symbol_duration_us, 
                 control_chgannel_overhead):
                
        ##### Plots 
        ########################
         
        self.plot = 0 # Switch to control plots if any
         
        #### Input storage 
        ########################      
        
        self.is_dl = is_dl
        
        self.cell_IDs_df = cell_IDs

        self.transmit_antennas_df = transmit_antennas
        #self.receive_antennas_df = receive_antennas
        
        self.fdd_tdd_ind = fdd_tdd_ind
        self.subframe_assignment = subframe_assignment
        self.FR = FR
        self.subcarriers_per_PRB = subcarriers_per_PRB
        self.ofdm_symbols_in_slot = ofdm_symbols_in_slot         
        #self.carrier_frequency_GHz = carrier_frequency_GHz 
        #self.carrier_wavelength_m = carrier_wavelength_m
        self.bandwidth_MHz = bandwidth_MHz
        self.available_PRBs = available_PRBs
        self.ofdm_symbol_duration_us = ofdm_symbol_duration_us
        self.control_chgannel_overhead = control_chgannel_overhead

        ##### Processing
        ########################   
        
        # Create mapping indicating the receive and transmit antennas of each UE and cell from the vectors comprising all UE antennas and all cell antennas, resepectively
        
        #self.mapping_receive_antennas_to_ue_IDs = np.repeat(np.arange(0,len(receive_antennas)), repeats=receive_antennas.to_numpy()) 
        self.mapping_transmit_antennas_to_cell_IDs = np.repeat(np.arange(0,len(transmit_antennas)), repeats=transmit_antennas.to_numpy())

        # Create structure where channel gains are going to be stored PRBs x tx_Antennas x rx_antennas
        
        #self.channel_gain = np.zeros((self.available_PRBs,self.receive_antennas_df.sum(), self.transmit_antennas_df.sum()))