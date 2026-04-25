# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 10:37:25 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import time
from typing import List

import numpy as np

from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable

class MCS_Resource(Saveable):

    def __init__(self, simulation_config_obj):

        super().__init__()
        
        ##### Input storage
        ########################
        self.simulation_config_obj = simulation_config_obj
        
        ##### Outputs 
        ########################   
        
        # Place holder to store path loss results        
        self.modulation_schemes = [] # Modulation schemes
        self.number_of_modulation_schemes = None # Number of modulation schemes
        self.modulation_and_coding_schemes = [] # Modulation and coding schemes
        self.number_of_modulation_and_coding_schemes = None # Number of modulation and coding schemes
        self.modulation_order_per_modulation_and_coding_scheme = [] # Modulation order
        self.modulation_index_per_modulation_and_coding_scheme = [] # Modulation index in self.modulation_schemes for every modulation and coding scheme
        self.transport_block_sizes_in_prbs = [] # Transport block sizes
        self.number_of_transport_block_sizes = None # Maximum number of transport block sizes
        

    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["modulation_schemes"]

    
    def process(self,rescheduling_us=-1): 
        
        # Start timer       
        t_start = time.perf_counter()          
        
        # QPSK_ 0.11 0.14 0.18 0.24 0.30 0.37 0.43 0.50 0.58 0.65
        # 16QAM_ 0.33 0.37 0.41 0.47 0.54 0.59 0.64
        # 64QAM_ 0.42 0.46 0.50 0.55 0.60 0.65 0.69 0.74 0.79 0.85 0.88 0.92
        # 256QAM_ 0.69 0.78 0.86 0.93
        
        # Modulation schemes
        self.modulation_schemes = ["QPSK", "16QAM", "64QAM", "256QAM"]
        
        # Maximum number of modulation schemes
        self.number_of_modulation_schemes = len(self.modulation_schemes)  
        
        # Modulation and coding schemes
        self.modulation_and_coding_schemes = ["QPSK_0.11", "QPSK_0.14", "QPSK_0.18", "QPSK_0.24", "QPSK_0.30", "QPSK_0.37", "QPSK_0.43", "QPSK_0.50", "QPSK_0.58", "QPSK_0.65",
                                              "16QAM_0.33", "16QAM_0.37", "16QAM_0.41", "16QAM_0.47", "16QAM_0.54", "16QAM_0.59", "16QAM_0.64", 
                                              "64QAM_0.42", "64QAM_0.46", "64QAM_0.50", "64QAM_0.55", "64QAM_0.60", "64QAM_0.65", "64QAM_0.69", "64QAM_0.74", "64QAM_0.79", "64QAM_0.85", "64QAM_0.88", "64QAM_0.92",  
                                              "256QAM_0.69", "256QAM_0.78", "256QAM_0.86", "256QAM_0.93"]
        
        # Maximum number of modulation and coding schemes
        self.number_of_modulation_and_coding_schemes = len(self.modulation_and_coding_schemes)
        
        # Modulation order
        self.modulation_order_per_modulation_and_coding_scheme = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8, 8], dtype=int)
        
        # Modulation index in self.modulation_schemes for every modulation and coding scheme
        self.modulation_index_per_modulation_and_coding_scheme = (self.modulation_order_per_modulation_and_coding_scheme / 2.0 - 1).astype(int)
        
        # Transport block sizes
        self.transport_block_sizes_in_prbs = (np.array([144, 288, 432, 576, 720, 1440, 2880, 4320, 5760, 7200, 10800, 14400]) / 144).astype(int)
        
        # Maximum number of transport block sizes
        self.number_of_transport_block_sizes = len(self.transport_block_sizes_in_prbs)   

        ##### End
        log_calculations_time('MCS resource', t_start)