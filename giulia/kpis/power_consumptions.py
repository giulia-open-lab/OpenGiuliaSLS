# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 17:26:07 2024


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

from giulia.tools import tools
from giulia.tools.tools import log_calculations_time
from typing import Dict
from giulia.outputs.saveable import Saveable


class Power_Consumption(Saveable):
    
    def __init__(self, simulation_config_obj, network_deployment_obj):
       
        super().__init__()
       
        ##### Plots 
        ########################
        self.plot = 0 # Switch to control plots if any

        ##### Input storage 
        ########################   
        self.simulation_config_obj = simulation_config_obj
        self.network_deployment_obj = network_deployment_obj
       
        ##### Output  
        ########################   
        # Place holder to store path loss results
        self.P_base_band_W = []   
       
        self.P_0: np.ndarray = np.array([])
        self.number_of_TX_transceivers: np.ndarray = np.array([])
        self.number_of_RF_chains_per_TX_transceiver: np.ndarray = np.array([])
        self.number_of_active_RF_chains_per_transceiver: np.ndarray = np.array([])
        self.consumption_per_RF_chain_W: np.ndarray = np.array([])
        self.consumption_per_MCPA_W: np.ndarray = np.array([])
        self.efficiency_of_MCPA: np.ndarray = np.array([])

        self.transmit_power_per_radio_unit_W: np.ndarray = np.array([])
        self.power_consumption_per_radio_unit_kW: np.ndarray = np.array([])
        self.total_radio_unit_power_consumption_kW = None
        self.total_bbu_power_consumption_kW = None
        self.total_RAN_power_consumption_kW = None


    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["power_consumption_per_radio_unit_kW"]


    def process(self, rescheduling_us=-1):
        
        self.rng = np.random.RandomState(self.simulation_config_obj.random_seed+0)
            
        # Porcess inputs
        self.cell_ID = self.network_deployment_obj.df_ep["ID"].to_numpy(dtype=int) 
        self.project_name = self.simulation_config_obj.project_name
        
        # Nodes 
        # Radio Unit
        self.radio_unit_IDs = self.network_deployment_obj.df_ep["radio_unit_ID"].to_numpy(dtype=int) 
        self.radio_unit_types = self.network_deployment_obj.df_ep["radio_unit_type"].to_numpy(dtype=str)  
        
        self.unique_radio_unit_IDs, self.indices_unique_radio_units_IDs = np.unique(self.radio_unit_IDs, return_index=True)
        self.number_of_radio_units_deployed = len(self.unique_radio_unit_IDs)    
        
        self.map_radioUnitIDs_to_radioUnitType_dict =\
            {ID: self.radio_unit_types[idx] for ID, idx in zip(self.unique_radio_unit_IDs, self.indices_unique_radio_units_IDs)}
        
        # BBU
        self.bbu_IDs = self.network_deployment_obj.df_ep["bbu_ID"].to_numpy(dtype=int) 
        self.bbu_types = self.network_deployment_obj.df_ep["bbu_type"].to_numpy(dtype=str) 
            
        self.unique_bbu_IDs, self.indices_unique_bbu_IDs = np.unique(self.bbu_IDs, return_index=True)
        self.number_of_bbus_deployed = len(self.unique_bbu_IDs) 
        
        self.map_radioUnitIDs_to_bbuUnitIDs_dict =\
            {radio_ID: self.bbu_IDs[np.where(self.radio_unit_IDs==radio_ID)[0]] for radio_ID in self.unique_radio_unit_IDs}
        
        self.TX_number = self.network_deployment_obj.df_ep["TX_number"].to_numpy(dtype=int) 
       
       
        # Process outputs
        self.P_0 = np.empty(self.number_of_radio_units_deployed, dtype=object) #np.full(self.number_of_radio_units_deployed, -1, dtype=np.single) 
        self.number_of_TX_transceivers = np.full(self.number_of_radio_units_deployed, -1, dtype=int)
        self.number_of_RF_chains_per_TX_transceiver = np.full(self.number_of_radio_units_deployed, -1, dtype=int)
        self.consumption_per_base_band_W = np.full(self.number_of_radio_units_deployed, -1, dtype=np.single) 
        self.consumption_per_RF_chain_W = np.full(self.number_of_radio_units_deployed, -1, dtype=np.single) 
        self.consumption_per_MCPA_W = np.full(self.number_of_radio_units_deployed, -1, dtype=np.single)     
        self.efficiency_of_MCPA = np.full(self.number_of_radio_units_deployed, -1, dtype=np.single) 
        
        self.P_base_band_W = np.full(self.number_of_bbus_deployed, -1, dtype=np.single)   
        
        self.number_of_active_RF_chains_per_transceiver = np.full(self.number_of_radio_units_deployed, -1, dtype=int) 
        
        self.transmit_power_per_radio_unit_W = np.full(self.number_of_radio_units_deployed, -1, np.single)
        self.power_consumption_per_radio_unit_kW = np.full(self.number_of_radio_units_deployed, -1, np.single)
        self.total_radio_unit_power_consumption_kW = -1
        self.total_bbu_power_consumption_kW = -1
        self.total_RAN_power_consumption_kW = -1        
        
        self.power_consumption_results_per_radioType_kW_dict: Dict[str, np.ndarray] = {}
        
        # Start timer       
        t_start = time.perf_counter()         
            
        ##### Processing for RADIO UNITS
        ########################  
                
        # Find the set of unique radio unit models to process them independently
        radio_unit_types_set = np.unique(self.radio_unit_types) 
            
        # Process each radio unit model indepnedently
        for radio_unit_type in radio_unit_types_set:
                
            # Identify cells with the selected radio unit type
            mask = radio_unit_type ==  self.radio_unit_types[self.indices_unique_radio_units_IDs]
                
            # Derive propagation model parameters
            # This is done to also manage the LNR type of cells where different technologies use the same radio and their cells are using a different number of RF chains. We take the max number
            self.number_of_RF_chains_per_TX_transceiver[mask] = np.tile(np.max(self.TX_number[radio_unit_type ==  self.radio_unit_types]), len(self.number_of_RF_chains_per_TX_transceiver[mask]))
            
            if (radio_unit_type == "4G-Macro-RRU-v1"):
                self.P_0[mask], \
                    self.number_of_TX_transceivers[mask], \
                    self.consumption_per_base_band_W[mask], \
                    self.consumption_per_RF_chain_W[mask], \
                    self.consumption_per_MCPA_W[mask], \
                    self.efficiency_of_MCPA[mask] = self.set_up_4G_Macro_RRU_v1_power_consumption(sum(mask))
                    
            elif (radio_unit_type == "4G-Micro-RRU-v1"):
                
                self.P_0[mask], \
                    self.number_of_TX_transceivers[mask], \
                    self.consumption_per_base_band_W[mask], \
                    self.consumption_per_RF_chain_W[mask], \
                    self.consumption_per_MCPA_W[mask], \
                    self.efficiency_of_MCPA[mask] = self.set_up_4G_Micro_RRU_v1_power_consumption(sum(mask))                   
                    
            elif (radio_unit_type == "5G-Macro-AAU-v1"):
                
                self.P_0[mask], \
                    self.number_of_TX_transceivers[mask], \
                    self.consumption_per_base_band_W[mask], \
                    self.consumption_per_RF_chain_W[mask], \
                    self.consumption_per_MCPA_W[mask], \
                    self.efficiency_of_MCPA[mask] = self.set_up_5G_Macro_AAU_v1_power_consumption(sum(mask)) 
                    
            elif (radio_unit_type == "5G-Micro-AAU-v1"):
                
                self.P_0[mask], \
                    self.number_of_TX_transceivers[mask], \
                    self.consumption_per_base_band_W[mask], \
                    self.consumption_per_RF_chain_W[mask], \
                    self.consumption_per_MCPA_W[mask], \
                    self.efficiency_of_MCPA[mask] = self.set_up_5G_Micro_AAU_v1_power_consumption(sum(mask)) 
                    
            elif (radio_unit_type == "6G-Micro-AAU-v1"):
                
                self.P_0[mask], \
                    self.number_of_TX_transceivers[mask], \
                    self.consumption_per_base_band_W[mask], \
                    self.consumption_per_RF_chain_W[mask], \
                    self.consumption_per_MCPA_W[mask], \
                    self.efficiency_of_MCPA[mask] = self.set_up_6G_Micro_AAU_v1_power_consumption(sum(mask))  
                    
            elif (radio_unit_type == "6G-Picro-AAU-v1"):
                
                self.P_0[mask], \
                    self.number_of_TX_transceivers[mask], \
                    self.consumption_per_base_band_W[mask], \
                    self.consumption_per_RF_chain_W[mask], \
                    self.consumption_per_MCPA_W[mask], \
                    self.efficiency_of_MCPA[mask] = self.set_up_6G_Micro_AAU_v1_power_consumption(sum(mask))   
                    
            elif (radio_unit_type == "4G5G-Macro-AAU-v1" ):
                
                self.P_0[mask], \
                    self.number_of_TX_transceivers[mask], \
                    self.consumption_per_base_band_W[mask], \
                    self.consumption_per_RF_chain_W[mask], \
                    self.consumption_per_MCPA_W[mask], \
                    self.efficiency_of_MCPA[mask] = self.set_up_4G5G_Macro_AAU_v1_power_consumption(sum(mask)) 

            elif (radio_unit_type == "5G6G-Micro-AAU-v1"):
                
                self.P_0[mask], \
                    self.number_of_TX_transceivers[mask], \
                    self.consumption_per_base_band_W[mask], \
                    self.consumption_per_RF_chain_W[mask], \
                    self.consumption_per_MCPA_W[mask], \
                    self.efficiency_of_MCPA[mask] = self.set_up_5G6G_Micro_AAU_v1_power_consumption(sum(mask))                    
                    
        # Create radio unit data frame  
        radio_unit_parameters_d = {
                'ID': np.arange(0, self.number_of_radio_units_deployed, dtype=int), 
                'type': self.radio_unit_types[self.indices_unique_radio_units_IDs], 
                'P_0': self.P_0, 
                'number_of_TX_transceivers': self.number_of_TX_transceivers,  
                'number_of_RF_chains_per_TX_transceiver': self.number_of_RF_chains_per_TX_transceiver,  
                'consumption_per_RF_chain_W': self.consumption_per_RF_chain_W,  
                'consumption_per_MCPA_W': self.consumption_per_MCPA_W,  
                'efficiency_of_MCPA': self.efficiency_of_MCPA
                }
        self.df_ru = pd.DataFrame(data=radio_unit_parameters_d)


        ##### Processing for BASE BAND UNITS
        ########################  
                   
        # Find the set of unique base band unit models to process them independently
        bbu_types_set = np.unique(self.bbu_types) 
        
        # Process each base band unit model indepnedently
                        
        for bbu_type in bbu_types_set:
                
            # Identify cells with the selected base band unit type
            mask = bbu_type ==  self.bbu_types[self.indices_unique_bbu_IDs]
                
            # Derive propagation model parameters
            if (bbu_type == "4G-Macro-BBU-v1"):
                self.P_base_band_W[mask] = self.set_up_4G_Macro_BBU_v1_power_consumption(sum(mask))
                    
            elif (bbu_type == "4G-Micro-BBU-v1"):
                self.P_base_band_W[mask] = self.set_up_4G_Micro_BBU_v1_power_consumption(sum(mask))                   
                    
            elif (bbu_type == "5G-Macro-BBU-v1"):
                self.P_base_band_W[mask] = self.set_up_5G_Macro_BBU_v1_power_consumption(sum(mask))     
                    
            elif (bbu_type == "5G-Micro-BBU-v1"):
                self.P_base_band_W[mask] = self.set_up_5G_Micro_BBU_v1_power_consumption(sum(mask)) 
                    
            elif (bbu_type == "6G-Micro-BBU-v1"):
                self.P_base_band_W[mask] = self.set_up_6G_Micro_BBU_v1_power_consumption(sum(mask))  
                    
            elif (bbu_type == "6G-Picro-BBU-v1"):
                self.P_base_band_W[mask] = self.set_up_6G_Picro_BBU_v1_power_consumption(sum(mask))                      
                    
        # Create base band unit data frame  
        base_band_unit_parameters_d = {
                'ID': np.arange(0, self.number_of_bbus_deployed, dtype=int), 
                'type': self.bbu_types[self.indices_unique_bbu_IDs], 
                'P_base_band_W': self.P_base_band_W  
                }
        self.df_bbu = pd.DataFrame(data=base_band_unit_parameters_d)      


        unique_bbu_IDs, indices_unique_bbu_IDs = np.unique(self.bbu_IDs, return_index=True)
        self.number_of_bbus_deployed = len(unique_bbu_IDs)       
        
        ### Checks
        ########################
        self.basic_checks(self.network_deployment_obj.df_ep)
    
        ### Initialize number of active RF chains
        ########################
        
        # This variable will be managed by the RF shutdwon algorithm, if any.
        # By default, all RF chains are active
        self.set_number_of_active_RF_chains_per_transceiver(self.number_of_RF_chains_per_TX_transceiver.copy())
        
        ##### End 
        ########################
        log_calculations_time('Power consumption model', t_start)

        return rescheduling_us                   
 
    
    def basic_checks(self, df_ep):

        # Preparing arrays for SSB and CSI-RS beams checks
        ssb_beams = df_ep["SSB_number_of_beams"].to_numpy()

        # Compute maximum RF chains per radio unit
        max_rf_chains = self.number_of_TX_transceivers * self.number_of_RF_chains_per_TX_transceiver
        
        # Check radio unit by radio unit 
        for radio_unit_ID in range(self.number_of_radio_units_deployed):
            # Get cell IDs of a given radio unit
            cells_IDs_in_radio_unit = np.where(self.radio_unit_IDs == radio_unit_ID)[0]
                     
            # Make sure that all cells powered by a radio unit belong to the same radio unit type
            if ~np.all(self.radio_unit_types[cells_IDs_in_radio_unit] == self.radio_unit_types[cells_IDs_in_radio_unit[0]]):
                sys.exit("Error: All cells powered by a radio unit should have the same radio unit type")
            
            # Make sure that all cells powered by a radio unit do not configure more SSB beams that RF chains available
            if ~np.all(ssb_beams[cells_IDs_in_radio_unit] <= max_rf_chains[radio_unit_ID]) : 
                sys.exit("Error: Number of SSB beams cannot be larger than the number of RF chains")             
             
                      
    def set_up_4G_Macro_RRU_v1_power_consumption(self, number_of_selected_cells):
        
        # [Active, Symbol shutdwon, carrier shutdown, dormancy]
        P_0 = np.full(number_of_selected_cells, {'active': 90.0, 'symbol_shutdown': 90.0*(1-0.34), 'carrier_shutdown': 90.0*(1-0.47), 'dormancy': 90.0*(1-0.623)}) 
        
        number_of_TX_transceivers = np.full(number_of_selected_cells, 1, dtype=np.single) 
        consumption_base_band_W = np.full(number_of_selected_cells, 0, dtype=np.single) 
        consumption_per_RF_chain_W = np.full(number_of_selected_cells, 2.8, dtype=np.single) 
        consumption_per_MCPA_W = np.full(number_of_selected_cells, 10, dtype=np.single) 
        efficiency_of_MCPA = np.full(number_of_selected_cells, 0.4, dtype=np.single) 
        
        return P_0, number_of_TX_transceivers, consumption_base_band_W, consumption_per_RF_chain_W, consumption_per_MCPA_W, efficiency_of_MCPA  
    

    def set_up_4G_Micro_RRU_v1_power_consumption(self, number_of_selected_cells):
        
        P_0 = np.full(number_of_selected_cells, {'active': 90.0*0.8, 'symbol_shutdown': 90.0*(1-0.34)*0.8, 'carrier_shutdown': 90.0*(1-0.47)*0.8, 'dormancy': 90.0*(1-0.623)*0.8}) 
        
        number_of_TX_transceivers = np.full(number_of_selected_cells, 1, dtype=np.single) 
        consumption_base_band_W = np.full(number_of_selected_cells, 0, dtype=np.single)        
        consumption_per_RF_chain_W = np.full(number_of_selected_cells, 2.8*0.8, dtype=np.single) 
        consumption_per_MCPA_W = np.full(number_of_selected_cells, 10*0.8, dtype=np.single) 
        efficiency_of_MCPA = np.full(number_of_selected_cells, 0.4, dtype=np.single) 
        
        return P_0, number_of_TX_transceivers, consumption_base_band_W, consumption_per_RF_chain_W, consumption_per_MCPA_W, efficiency_of_MCPA      

   
    def set_up_5G_Macro_AAU_v1_power_consumption(self, number_of_selected_cells):
        
        P_0 = np.full(number_of_selected_cells, {'active': 171.0, 'symbol_shutdown': 171.0*(1-0.34), 'carrier_shutdown': 171.0*(1-0.47), 'dormancy': 171.0*(1-0.623)}) 
        
        number_of_TX_transceivers = np.full(number_of_selected_cells, 1, dtype=np.single) 
        consumption_base_band_W = np.full(number_of_selected_cells, 127.0, dtype=np.single)        
        consumption_per_RF_chain_W = np.full(number_of_selected_cells, 1.28, dtype=np.single) 
        consumption_per_MCPA_W = np.full(number_of_selected_cells, 3.00, dtype=np.single) 
        efficiency_of_MCPA = np.full(number_of_selected_cells, 0.5, dtype=np.single) 
        
        return P_0, number_of_TX_transceivers, consumption_base_band_W, consumption_per_RF_chain_W, consumption_per_MCPA_W, efficiency_of_MCPA    


    def set_up_5G_Micro_AAU_v1_power_consumption(self, number_of_selected_cells):
        
        P_0 = np.full(number_of_selected_cells, {'active': 152.0, 'symbol_shutdown': 152.0*(1-0.34), 'carrier_shutdown':152.0*(1-0.47), 'dormancy': 152.0*(1-0.623)}) 
        
        number_of_TX_transceivers = np.full(number_of_selected_cells, 1, dtype=np.single) 
        consumption_base_band_W = np.full(number_of_selected_cells, 113.0, dtype=np.single)          
        consumption_per_RF_chain_W = np.full(number_of_selected_cells, 1.28, dtype=np.single) 
        consumption_per_MCPA_W = np.full(number_of_selected_cells, 2.64, dtype=np.single) 
        efficiency_of_MCPA = np.full(number_of_selected_cells, 0.5, dtype=np.single) 
        
        return P_0, number_of_TX_transceivers, consumption_base_band_W, consumption_per_RF_chain_W, consumption_per_MCPA_W, efficiency_of_MCPA 


    def set_up_6G_Micro_AAU_v1_power_consumption(self, number_of_selected_cells):
        
        P_0 = np.full(number_of_selected_cells, {'active': 152.0, 'symbol_shutdown': 152.0*(1-0.34), 'carrier_shutdown':152.0*(1-0.47), 'dormancy': 152.0*(1-0.623)}) 
        
        number_of_TX_transceivers = np.full(number_of_selected_cells, 1, dtype=np.single) 
        consumption_base_band_W = np.full(number_of_selected_cells, 167.0, dtype=np.single)          
        consumption_per_RF_chain_W = np.full(number_of_selected_cells, 1.28, dtype=np.single) 
        consumption_per_MCPA_W = np.full(number_of_selected_cells, 2.64, dtype=np.single) 
        efficiency_of_MCPA = np.full(number_of_selected_cells, 0.6, dtype=np.single) 
        
        return P_0, number_of_TX_transceivers, consumption_base_band_W, consumption_per_RF_chain_W, consumption_per_MCPA_W, efficiency_of_MCPA      


    def set_up_6G_Picro_AAU_v1_power_consumption(self, number_of_selected_cells):
        
        P_0 = np.full(number_of_selected_cells, {'active': 121.0, 'symbol_shutdown': 121.0*(1-0.34), 'carrier_shutdown':121.0*(1-0.47), 'dormancy': 121.0*(1-0.623)}) 
        
        number_of_TX_transceivers = np.full(number_of_selected_cells, 1, dtype=np.single) 
        consumption_base_band_W = np.full(number_of_selected_cells, 153.0, dtype=np.single)          
        consumption_per_RF_chain_W = np.full(number_of_selected_cells, 1.28, dtype=np.single) 
        consumption_per_MCPA_W = np.full(number_of_selected_cells, 2.64, dtype=np.single) 
        efficiency_of_MCPA = np.full(number_of_selected_cells, 0.6, dtype=np.single) 
        
        return P_0, number_of_TX_transceivers, consumption_base_band_W, consumption_per_RF_chain_W, consumption_per_MCPA_W, efficiency_of_MCPA   


    def set_up_4G5G_Macro_AAU_v1_power_consumption(self, number_of_selected_cells):
        
        P_0 = np.full(number_of_selected_cells, {'active': 190.0, 'symbol_shutdown': 190.0*(1-0.34), 'carrier_shutdown':190.0*(1-0.47), 'dormancy': 190.0*(1-0.623)}) 
        
        number_of_TX_transceivers = np.full(number_of_selected_cells, 1, dtype=np.single) 
        consumption_base_band_W = np.full(number_of_selected_cells, 142.0, dtype=np.single)          
        consumption_per_RF_chain_W = np.full(number_of_selected_cells, 1.28, dtype=np.single) 
        consumption_per_MCPA_W = np.full(number_of_selected_cells, 3.3, dtype=np.single) 
        efficiency_of_MCPA = np.full(number_of_selected_cells, 0.55, dtype=np.single) 
        
        return P_0, number_of_TX_transceivers, consumption_base_band_W, consumption_per_RF_chain_W, consumption_per_MCPA_W, efficiency_of_MCPA   


    def set_up_5G6G_Micro_AAU_v1_power_consumption(self, number_of_selected_cells):
        
        P_0 = np.full(number_of_selected_cells, {'active': 167.0, 'symbol_shutdown': 167.0*(1-0.34), 'carrier_shutdown':167.0*(1-0.47), 'dormancy': 167.0*(1-0.623)}) 
        
        number_of_TX_transceivers = np.full(number_of_selected_cells, 1, dtype=np.single) 
        consumption_base_band_W = np.full(number_of_selected_cells, 182.0, dtype=np.single)          
        consumption_per_RF_chain_W = np.full(number_of_selected_cells, 1.28, dtype=np.single) 
        consumption_per_MCPA_W = np.full(number_of_selected_cells, 2.97, dtype=np.single) 
        efficiency_of_MCPA = np.full(number_of_selected_cells, 0.65, dtype=np.single) 
        
        return P_0, number_of_TX_transceivers, consumption_base_band_W, consumption_per_RF_chain_W, consumption_per_MCPA_W, efficiency_of_MCPA  

    
    def set_up_4G_Macro_BBU_v1_power_consumption(self, number_of_selected_cells):
        P_base_band_W = np.full(number_of_selected_cells, 160, dtype=np.single) 
        return P_base_band_W  


    def set_up_4G_Micro_BBU_v1_power_consumption(self, number_of_selected_cells):
        P_base_band_W = np.full(number_of_selected_cells, 160*0.8, dtype=np.single) 
        return P_base_band_W     


    def set_up_5G_Macro_BBU_v1_power_consumption(self, number_of_selected_cells):
        P_base_band_W = np.full(number_of_selected_cells, 220, dtype=np.single) 
        return P_base_band_W    


    def set_up_5G_Micro_BBU_v1_power_consumption(self, number_of_selected_cells):
        P_base_band_W = np.full(number_of_selected_cells, 220, dtype=np.single) 
        return P_base_band_W  


    def set_up_6G_Micro_BBU_v1_power_consumption(self, number_of_selected_cells):
        P_base_band_W = np.full(number_of_selected_cells, 280, dtype=np.single) 
        return P_base_band_W  


    def set_up_6G_Picro_BBU_v1_power_consumption(self, number_of_selected_cells):
        P_base_band_W = np.full(number_of_selected_cells, 280, dtype=np.single) 
        return P_base_band_W 


    def calculate_power_consumption(self, network_deployment_obj, rescheduling_us, cell_status, base_stations_obj):
        
        # Start performance timer
        t_start: float = time.perf_counter()
        
        # Calculate the transmit power per cell 
        # Inputs
        self.BS_tx_power_dBm = BS_tx_power_dBm = network_deployment_obj.df_ep["BS_tx_power_dBm"].to_numpy(dtype=np.single) 
        
        # Here we assume that the TX power of the cell is equally split among all the resource elements in the bandwidth
        # This is in line with the policy used to calculate TX power per resource element in scenarios.py 
        self.PRB_load_per_cell = np.mean(base_stations_obj.resource_allocation["PRB_cell_activity"],axis=0)
        transmit_power_per_cell_W = self.PRB_load_per_cell * tools.dBm_to_mW(BS_tx_power_dBm) / 1000
        
        # Calculate the P_0 per radio unit according to the sleep statuses of its cells and the transmit power per radio unit 
        P_0_W = np.full(self.number_of_radio_units_deployed, -1, dtype=np.single) 
        for radio_unit_ID in range(self.number_of_radio_units_deployed):
            positions = np.where(self.radio_unit_IDs == radio_unit_ID)[0]
            
            # Calculate P_0 per radio unit according to the sleep statuses of its cells
            if np.all(cell_status[positions] == 'symbol_shutdown'):
                P_0_W[radio_unit_ID] = self.P_0[radio_unit_ID]['symbol_shutdown']
            elif np.all(cell_status[positions] == 'carrier_shutdown'):  
                P_0_W[radio_unit_ID] = self.P_0[radio_unit_ID]['carrier_shutdown']
            elif np.all(cell_status[positions] == 'dormancy'):  
                P_0_W[radio_unit_ID] = self.P_0[radio_unit_ID]['dormancy']    
            else:
                P_0_W[radio_unit_ID] = self.P_0[radio_unit_ID]['active'] 
                
            # Calculate the transmit power per radio unit 
            self.transmit_power_per_radio_unit_W[radio_unit_ID] = np.sum(transmit_power_per_cell_W[positions])       
        
        del radio_unit_ID
        
        # Calculate the power consumption of the radio 
        self.power_consumption_per_radio_unit_kW = (P_0_W + \
                                                        self.number_of_TX_transceivers * self.number_of_RF_chains_per_TX_transceiver * self.consumption_per_RF_chain_W +\
                                                        self.number_of_active_RF_chains_per_transceiver * self.consumption_per_MCPA_W +\
                                                        1/self.efficiency_of_MCPA * self.transmit_power_per_radio_unit_W) / 1000
                                                                                                                                
        
        # Calculate the power consumption of all bbus                                      
        self.total_bbu_power_consumption_kW = np.sum(self.P_base_band_W) / 1000 

        #### Compute Power Consumption per RAN
        # Init Dictionary for store
        self.power_consumption_perRadio_kW = np.full(self.number_of_radio_units_deployed, np.nan)
        self.power_consumption_results_per_radioType_kW_dict = {}
        for ru_id in self.unique_radio_unit_IDs:
            # RU power
            radio_power_kW = self.power_consumption_per_radio_unit_kW[ru_id]
            
            # BBUs attached to cells of this RU (deduplicate within the RU)
            bbu_ids_for_ru = np.unique(self.map_radioUnitIDs_to_bbuUnitIDs_dict[ru_id])
            bbu_power_for_ru_kW = float(self.P_base_band_W[bbu_ids_for_ru].sum()) / 1000.0

            # Get radio type for the considered deployed RU
            radio_type = self.map_radioUnitIDs_to_radioUnitType_dict[ru_id]
            
            # Store results
            self.power_consumption_perRadio_kW[ru_id] = radio_power_kW + bbu_power_for_ru_kW

            # Accumulate for total RAN computation
            self.power_consumption_results_per_radioType_kW_dict[radio_type] = \
                self.power_consumption_results_per_radioType_kW_dict.get(radio_type, 0) + (radio_power_kW + bbu_power_for_ru_kW)
        
        # Sanity Check
        assert np.allclose(sum(self.power_consumption_results_per_radioType_kW_dict.values()), 
                           np.sum(self.power_consumption_per_radio_unit_kW) + self.total_bbu_power_consumption_kW  ,
                           rtol=1e-6,   # relative tolerance
                           atol=1e-9    # absolute tolerance
                           ), f"Error: in {self.__module__}.{self.__class__.__name__} Mismatch between per-RAN sum and total RAN power"

        self.total_RAN_power_consumption_kW = sum(self.power_consumption_results_per_radioType_kW_dict.values())
        
        # Log elapsed time
        log_calculations_time('Calculate Network Power Consumption', t_start)

        return rescheduling_us
    

    def set_number_of_active_RF_chains_per_transceiver(self, number_of_active_RF_chains_per_transceiver):
        self.number_of_active_RF_chains_per_transceiver = number_of_active_RF_chains_per_transceiver
