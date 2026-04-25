# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:19:28 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import numpy as np

class Carrier:
    def __init__(self, 
                 is_dl, 
                 cell_IDs, 
                 transmit_antennas,
                 fdd_tdd_ind,
                 subframe_assignment,
                 FR,
                 subcarriers_per_PRB,
                 ofdm_symbols_in_slot,
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
        
        self.fdd_tdd_ind = fdd_tdd_ind
        self.subframe_assignment = subframe_assignment
        self.FR = FR
        self.subcarriers_per_PRB = subcarriers_per_PRB
        self.ofdm_symbols_in_slot = ofdm_symbols_in_slot         
        self.bandwidth_MHz = bandwidth_MHz
        self.available_PRBs = available_PRBs
        self.ofdm_symbol_duration_us = ofdm_symbol_duration_us
        self.control_chgannel_overhead = control_chgannel_overhead

        ##### Processing
        ########################  
         
        # Create mapping indicating the transmit antennas of each UE and cell from the vectors comprising all UE antennas and all cell antennas, resepectively
        self.mapping_transmit_antennas_to_cell_IDs = np.repeat(np.arange(0,len(transmit_antennas)), repeats=transmit_antennas.to_numpy())