# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 08:46:03 2024

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org 
matteo.bernabe@iteam.upv.es
"""
from typing import List

import numpy as np

from giulia.mac import macs, prb_requirement_calculators

class BaseStations:
    """
    Represents the collection of base stations in the network, 
    each with layers for PDCP, RLC, MAC, and PHY.
    """

    def __init__(self,
                 simulation_config_obj,
                 network_deployment_obj,
                 ue_deployment_obj,
                 beam_conf_obj,
                 best_serving_cell_ID_per_ue_based_on_SSB_obj,
                 best_serving_CSI_RS_per_ue_obj,
                 dl_fast_fading_gain_ueAnt_to_cellAnt_obj, 
                 SSB_sinr_ue_to_cell_obj, 
                 ):
        """
        Initializes the BaseStations collection and creates individual base station instances.
        
        Args:
            simulation_config_obj: Configuration object for the simulation.
            network_deployment_obj: Object containing network deployment details.
            ue_deployment_obj: Object with UE deployment information.
            cell_CSI_RS_conf_obj: Configuration for CSI-RS per cell.
            best_serving_CSI_RS_per_ue_obj: Object mapping best-serving CSI-RS per UE.
            dl_fast_fading_gain_ueAnt_to_cellAnt_obj: Downlink fast fading gain details.
            
            
            TO ADD 
            best_serving_cell_ID_per_ue_based_on_SSB_obj: Object containing information on the best UE serving cell from SSB
            SSB_sinr_ue_to_cell_obj: object containing information on the measured SSB SINRs
        """
        
        super().__init__()
        
        # Store input objects
        self.simulation_config_obj = simulation_config_obj
        self.network_deployment_obj = network_deployment_obj
        self.ue_deployment_obj = ue_deployment_obj
        self.beam_conf_obj = beam_conf_obj
        
        self.best_serving_CSI_RS_per_ue_obj = best_serving_CSI_RS_per_ue_obj
        self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj = dl_fast_fading_gain_ueAnt_to_cellAnt_obj
        
        self.best_serving_cell_ID_per_ue_based_on_SSB_obj = best_serving_cell_ID_per_ue_based_on_SSB_obj
        self.SSB_sinr_ue_to_cell_obj = SSB_sinr_ue_to_cell_obj
        
        # Outputs
        self.resource_allocation = {}


        # Initialize traffic generator for PRB
        # Necessary for MAC operations
        self.prb_requirement_generator_ue_obj = \
            prb_requirement_calculators.PRB_Requirement(
                self.network_deployment_obj,
                self.ue_deployment_obj,
                beam_conf_obj,
                self.best_serving_CSI_RS_per_ue_obj, 
                self.SSB_sinr_ue_to_cell_obj,
                self.best_serving_cell_ID_per_ue_based_on_SSB_obj)
            
        # Initialize base stations
        self.base_stations = []
        self._initialize_base_stations() 


    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return [""]
    

    def _initialize_base_stations(self):
        """
        Initializes individual base station objects for each cell in the deployment.
        """
        
        for cell_index in range(len(self.network_deployment_obj.df_ep)):
            cell_name = self.network_deployment_obj.df_ep.get("name", "").iloc[cell_index]
            base_station = BaseStation(
                cell_index,
                cell_name,
                self.simulation_config_obj,
                self.network_deployment_obj,
                self.ue_deployment_obj,
                self.beam_conf_obj,
                self.prb_requirement_generator_ue_obj,
                self.best_serving_CSI_RS_per_ue_obj,
                self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj,
                self.resource_allocation
            )
            self.base_stations.append(base_station)
            
            
    def process(self, rescheduling_us=-1):
        """
        Initializes the functioning of the base station.
        """
        
        # Outputs
        dl_PRBs_available = self.network_deployment_obj.df_ep["dl_PRBs_available"].max()  # Maximum PRBs available
        self.resource_allocation.update({
            "dl_PRBs_required_per_ue": np.zeros((dl_PRBs_available, len(self.ue_deployment_obj.df_ep)),dtype=int),            
            "ue_IDs_per_PRB_and_beam": np.full((dl_PRBs_available, np.sum(self.beam_conf_obj.number_of_beams_per_node)),-1,dtype=int),
            "PRB_cell_activity": np.zeros((dl_PRBs_available, len(self.network_deployment_obj.df_ep)),dtype=bool),
            "PRB_ue_activity": np.zeros((dl_PRBs_available, len(self.ue_deployment_obj.df_ep)),dtype=bool), 
            "PRB_beam_activity": np.zeros((dl_PRBs_available, np.sum(self.beam_conf_obj.number_of_beams_per_node)),dtype=bool),
            "PRB_ue_beam_activity": 
                np.zeros((dl_PRBs_available, len(self.ue_deployment_obj.df_ep), np.sum(self.beam_conf_obj.number_of_beams_per_node)),dtype=bool),
            "PRB_ue_beam_interference_activity": 
                np.zeros((dl_PRBs_available, len(self.ue_deployment_obj.df_ep), np.sum(self.beam_conf_obj.number_of_beams_per_node)),dtype=bool)
        })
        
        # #### MAC  ####
        # #####################
        # Compute the dl_PRBs_required_per_ue 
        self.prb_requirement_generator_ue_obj._process()
        self.resource_allocation ['dl_PRBs_required_per_ue'] = self.prb_requirement_generator_ue_obj.get_dl_PRBs_required_per_ue()
        
        for cell_index in range(len(self.network_deployment_obj.df_ep)):
            self.base_stations[cell_index].mac_layer._run()  
            
        return rescheduling_us               

                
class BaseStation:
    """
    Represents a single base station in the network with layers for PDCP, RLC, MAC, and PHY.
    """

    def __init__(self,
                 cell_index : int,
                 cell_name : str,
                 simulation_config_obj,
                 network_deployment_obj,
                 ue_deployment_obj,
                 beam_conf_obj,
                 prb_requirement_generator_ue_obj,
                 best_serving_CSI_RS_per_ue_obj,
                 dl_fast_fading_gain_ueAnt_to_cellAnt_obj,
                 resource_allocation):
        """
        Initializes the BaseStation with its basic properties and layer structures.
        
        Args:
            simulation_config_obj: Configuration object for the simulation.
            network_deployment_obj: Object containing network deployment details.
            ue_deployment_obj: Object with UE deployment information.
            cell_CSI_RS_conf_obj: Configuration for CSI-RS per cell.
            prb_requirement_generator_ue_obj: Object for generating UE traffic.
            best_serving_CSI_RS_per_ue_obj: Object mapping best-serving CSI-RS per UE.
            dl_fast_fading_gain_ueAnt_to_cellAnt_obj: Downlink fast fading gain details.
        """
        
        # Store input objects
        self.cell_index = cell_index
        self.cell_name = cell_name        
        self.simulation_config_obj = simulation_config_obj
        self.network_deployment_obj = network_deployment_obj
        self.ue_deployment_obj = ue_deployment_obj
        self.beam_conf_obj = beam_conf_obj
        self.prb_requirement_generator_ue_obj = prb_requirement_generator_ue_obj
        self.best_serving_CSI_RS_per_ue_obj = best_serving_CSI_RS_per_ue_obj
        self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj = dl_fast_fading_gain_ueAnt_to_cellAnt_obj
        self.resource_allocation = resource_allocation

        # Initialize layers
        self.pdcp_layer = self._initialize_pdcp_layer()
        self.rlc_layer = self._initialize_rlc_layer()
        self.mac_layer = self._initialize_mac_layer()
        self.phy_layer = self._initialize_phy_layer()


    def _initialize_pdcp_layer(self):
        """
        Initializes the PDCP layer for the base station.
        
        Returns:
            An instance of the PDCP layer.
        """
        #print("Initializing PDCP layer...")
        return None  # Replace with actual PDCP initialization


    def _initialize_rlc_layer(self):
        """
        Initializes the RLC layer for the base station.
        
        Returns:
            An instance of the RLC layer.
        """
        #print("Initializing RLC layer...")
        return None  # Replace with actual RLC initialization


    def _initialize_mac_layer(self):
        """
        Initializes the MAC layer for the base station.
        
        Returns:
            An instance of the MAC layer.
        """
        return macs.MAC(
            self.cell_index,
            self.cell_name,
            self.simulation_config_obj,
            self.network_deployment_obj,
            self.ue_deployment_obj,
            self.beam_conf_obj,
            self.best_serving_CSI_RS_per_ue_obj,
            self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj,
            self.resource_allocation,
            self.prb_requirement_generator_ue_obj
        )


    def _initialize_phy_layer(self):
        """
        Initializes the PHY layer for the base station.
        
        Returns:
            An instance of the PHY layer.
        """
        #print("Initializing PHY layer...")
        return None  # Replace with actual PHY initialization
