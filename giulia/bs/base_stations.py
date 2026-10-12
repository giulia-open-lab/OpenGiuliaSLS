# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 12:09:05 2025

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org 
matteo.bernabe@iteam.upv.es
"""    
import time
from typing import Dict, List

import numpy as np

from giulia.event_driven.eventDriven import EventDrivenSimulation
from giulia.bs import bs_tx_powers
from giulia.mac import macs, prb_requirement_calculators
from giulia.rrc import rrcs 
from giulia.tools.tools import log_calculations_time
from giulia.kpis.calculate_sinr import SINR
from giulia.logger import warning

from giulia.outputs.saveable import Saveable

class BaseStations(Saveable):
    """
    Represents the collection of base stations in the network, 
    each with layers for PDCP, RLC, MAC, and PHY.
    """
    
    def __init__(self,
                 eventDriven_obj,
                 simulation_config_obj,
                 network_deployment_obj,
                 ue_deployment_obj,
                 beam_conf_obj,
                 time_frequency_resource_obj,
                 SSB_RSRP_ue_to_cell_obj, 
                 best_serving_cell_ID_per_ue_based_on_SSB_obj,
                 best_serving_CSI_RS_per_ue_obj,
                 dl_fast_fading_gain_ueAnt_to_cellAnt_obj, 
                 SSB_sinr_ue_to_cell_obj, 
                 power_consumption_obj
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

        # Register this collection as a Saveable so its KPIs (e.g. the per-cell
        # radio-unit status) are collected and exported to the cell_kpi parquet.
        super().__init__()

        # Store input objects
        self.eventDriven_obj = eventDriven_obj
        self.simulation_config_obj = simulation_config_obj
        self.network_deployment_obj = network_deployment_obj
        self.ue_deployment_obj = ue_deployment_obj
        self.beam_conf_obj = beam_conf_obj
        self.time_frequency_resource_obj = time_frequency_resource_obj
        
        self.SSB_RSRP_ue_to_cell_obj = SSB_RSRP_ue_to_cell_obj
        
        self.best_serving_CSI_RS_per_ue_obj = best_serving_CSI_RS_per_ue_obj
        self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj = dl_fast_fading_gain_ueAnt_to_cellAnt_obj
        
        self.best_serving_cell_ID_per_ue_based_on_SSB_obj = best_serving_cell_ID_per_ue_based_on_SSB_obj
        self.SSB_sinr_ue_to_cell_obj: SINR = SSB_sinr_ue_to_cell_obj
        self.CSI_RS_sinr_per_PRB_ue_to_cell_obj: None|SINR = None
        
        self.power_consumption_obj = power_consumption_obj
        
        
        #### BS TX Power ####
        #####################
            
        self.bs_tx_power_obj = \
            bs_tx_powers.BS_TX_Power(self.network_deployment_obj)        
        self.bs_tx_power_obj.process()
        
        # Dictionary with the number of available PRBs per cell
        self.dl_PRBs_available_per_cell_dict: dict = dict(zip(
                                               self.network_deployment_obj.df_ep["name"],
                                               self.network_deployment_obj.df_ep["dl_PRBs_available"].astype(int)
                                               ))
        
        self.mapping_cellName_to_index_dict: dict = dict(zip(
                                                        self.network_deployment_obj.df_ep["name"],
                                                        range(len(self.network_deployment_obj.df_ep))
                                                        ))
        
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
                self.best_serving_cell_ID_per_ue_based_on_SSB_obj, 
                self.time_frequency_resource_obj)
            
            
        # Initialize base stations
        self.base_stations = self._initialize_base_stations() 

    @property
    def label(self) -> str: return 'cell_kpi'

    @property
    def prb_usage_per_cell(self) -> np.ndarray:
        """
        Fraction of the available downlink PRBs each cell is using (0-1), one value per cell.

        Read live from every base station's MAC scheduler, in ``df_ep`` cell order, so it is
        aligned with the ``cell_id`` index of the ``cell_kpi`` parquet. This is the same
        quantity the legacy NPZ stores as ``PRB_usage_per_cell``; exposing it here is what
        puts it in the Parquet output as well.
        """
        return np.array(
            [bs_obj.mac_layer.scheduler_obj._prb_usage for bs_obj in self.base_stations],
            dtype=float,
        )

    def expose_variables(self) -> List[str]:
        """List of attributes name to be saved."""
        # Both go to 'cell_kpi' via the class label above, so no expose_variable_labels()
        # override is needed (see docs/output-saveable/saveable-output.md section 3).
        return ['cell_radio_unit_statuses', 'prb_usage_per_cell']

    def expose_variable_descriptions(self) -> Dict[str, str]:
        """Human-readable description for each exposed variable (shown in the Saveable catalog)."""
        return {
            # NOTE: not present in data/output-info.csv — description inferred, please verify.
            'cell_radio_unit_statuses': "Operational status of each cell's radio unit (e.g. active / carrier_shutdown) at each step.",
            'prb_usage_per_cell': "Fraction of the available downlink PRBs used by each cell (0-1), averaged over the cell's available PRBs at each step.",
        }

    #### Process
    ############
    
    def process(self, rescheduling_us=-1):
        """
        Initializes the functioning of the base station.
        """
        
        # Start timer       
        t_start = time.perf_counter()  

        # Maximum PRBs available in the network
        # for over dimensioning the resource allocation matrices
        dl_PRBs_available_max_network = max(self.dl_PRBs_available_per_cell_dict.values())

        # Reset Resource Allocation
        self.resource_allocation.update({
            "dl_PRBs_required_per_ue": np.zeros(
                (dl_PRBs_available_max_network, len(self.ue_deployment_obj.df_ep)), 
                dtype=int
            ),            
            "ue_IDs_per_PRB_and_beam": np.full(
                (dl_PRBs_available_max_network, np.sum(self.beam_conf_obj.number_of_beams_per_node)), 
                -1, 
                dtype=int
            ),
            "PRB_cell_activity": np.zeros(
                (dl_PRBs_available_max_network, len(self.network_deployment_obj.df_ep)), 
                dtype=bool
            ),
            "PRB_ue_activity": np.zeros(
                (dl_PRBs_available_max_network, len(self.ue_deployment_obj.df_ep)), 
                dtype=bool
            ),
            "PRB_beam_activity": np.zeros(
                (dl_PRBs_available_max_network, np.sum(self.beam_conf_obj.number_of_beams_per_node)), 
                dtype=bool
            ),
            "PRB_ue_beam_activity": np.zeros(
                (dl_PRBs_available_max_network, len(self.ue_deployment_obj.df_ep), np.sum(self.beam_conf_obj.number_of_beams_per_node)),
                dtype=bool
            ),
            "PRB_ue_beam_interference_activity": np.zeros(
                (dl_PRBs_available_max_network, len(self.ue_deployment_obj.df_ep), np.sum(self.beam_conf_obj.number_of_beams_per_node)),
                dtype=bool
            )
        })
        
        #### PRB Requirement Calculation ####
        #####################################
        # Compute the dl_PRBs_required_per_ue 
        self.prb_requirement_generator_ue_obj._process()
        self.resource_allocation ['dl_PRBs_required_per_ue'] = self.prb_requirement_generator_ue_obj.get_dl_PRBs_required_per_ue()

        #### Process Pairing Information ####
        #####################################
        # Configure each BaseStation with its paired cells and RRC layers.
        for bs_obj in self.base_stations:
            
            # Get the paired cells list in the BaseStation object
            paired_cell_list = self.network_deployment_obj.df_ep.loc[
                                    self.network_deployment_obj.df_ep["name"] == bs_obj.cell_name, "paired_cell"
                                    ].iloc[0]
            
            if paired_cell_list is None: continue
            # Set the paired cells in the BaseStation object
            else: bs_obj.set_paired_cell(paired_cell_list)

        log_calculations_time('PRB Requirement and Power Consumption', t_start)
        return rescheduling_us     
    


    @property
    def cell_radio_unit_statuses(self) -> np.ndarray:
        """
        Current radio-unit status of every base station in the network.

        Returns:
            np.ndarray: A string array with the ``radio_unit_status`` of each
            base station, ordered as in ``self.base_stations``.
        """
        return np.array(
            [bs_obj.rrc_layer.radio_unit_status for bs_obj in self.base_stations],
            dtype=str,
        )


    
    #### Initialize Event Driven Base Station Methods #####
    #######################################################
        
    def initialize_eventDriven_layers_MAC_layers(self, eventDriven_obj:EventDrivenSimulation, timing_from_now_us:int) -> None:
        #### Initiate MAC Events
        ############################
        for _, bs_obj in enumerate(self.base_stations):
            eventDriven_obj.add_event(0, bs_obj.mac_layer,
                                      '_run', 
                                      bs_obj.mac_layer.rescheduling_us)
        return

    def initialize_eventDriven_RRC_layers(self, eventDriven_obj:EventDrivenSimulation, timing_from_now_us:int) -> None:
        #### Initiate RRC Events
        ############################
        df_ep = self.network_deployment_obj.df_ep

        #### #### Carrier Shudown
        for _, bs_obj in enumerate(self.base_stations):
            # If the cell is configured to enable carrier shutdown
            # schedule the carrier shutdown event
            row = df_ep.loc[df_ep['name'] == bs_obj.cell_name]
            condition_activate_carrier_shutdown = row['radio_enable_carrier_shutdown'].values[0] and row['radio_unit_initial_status'].values[0]
            if condition_activate_carrier_shutdown:
                eventDriven_obj.add_event(0, bs_obj.rrc_layer.carrier_shutdown_obj,
                                          'run',
                                          bs_obj.rrc_layer.carrier_shutdown_obj.rescheduling_us)
        return 


    def initialize_eventDriven_PowerConsumption(self, eventDriven_obj:EventDrivenSimulation, timing_from_now_us:int) -> None:

        """
        The computation for the power for all base station has to be scheduled as a single event itself,
        as it requires information processed by the MAC layers of all base stations.
        Thus, it can not be instantiated within the self.process() method, as the MAC layers are not yet running.
        """
        eventDriven_obj.add_event(timing_from_now_us, self.power_consumption_obj,
                                  'calculate_power_consumption',
                                  self, # base_stations_obj
                                  1e6) # rescheduling_us

        return

        
    #### Private Single Base Station Initialization Methods
    ###################################
    
    def _initialize_base_stations(self)->list:
        """
        Initializes BaseStation objects for each cell and configures their
        RRC layers based on coverage and capacity pairing.

        Returns:
            list: A list of initialized BaseStation objects.
        """
        # Create BaseStation objects for each cell
        return [
            BaseStation(
                cell_index,
                self.network_deployment_obj.df_ep.get("name", "").iloc[cell_index],
                self.simulation_config_obj,
                self.network_deployment_obj,
                self.ue_deployment_obj,
                self.beam_conf_obj,
                self.prb_requirement_generator_ue_obj,
                self.best_serving_CSI_RS_per_ue_obj,
                self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj,
                self,  # Pass the current object 
            )
            for cell_index in range(len(self.network_deployment_obj.df_ep))
        ]
    
    

                
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
                 baseStationsAll_Parent_obj
                 ):
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
        self.baseStationsAll_Parent_obj = baseStationsAll_Parent_obj
        
        # Init paired cell placeholder
        self.paired_cells_idx: np.ndarray|list = []

        # Get information on carrier shutdown and initial status from network deployment
        self.enable_carrier_shutdown: bool =\
            self.network_deployment_obj.df_ep.loc[self.network_deployment_obj.df_ep["name"] == self.cell_name, "radio_enable_carrier_shutdown"].values[0]
        
        # Initial Radio Unit Status
        self.initial_radio_unit_status: str =\
            self.network_deployment_obj.df_ep.loc[self.network_deployment_obj.df_ep["name"] == self.cell_name, "radio_unit_initial_status"].values[0]
        

        # Initialize layers
        self.phy_layer = self._initialize_phy_layer()
        self.mac_layer = self._initialize_mac_layer()
        self.rlc_layer = self._initialize_rlc_layer()
        self.pdcp_layer = self._initialize_pdcp_layer()
        
        self.rrc_layer = self._initialize_rrc_layer()
        
        
    #### Public Setters and Getters
    ########################
    def set_paired_cell(self, pairing_cell_list:list|int):
        """
        Sets the indices of cells that are paired with the current cell
        based on the provided coverage-capacity pairing information within network engineering parameters.
        """
        # Store paired coverage and capacity cell reference
        self.paired_cells_idx = np.array(pairing_cell_list, dtype=int)
        
        # Initialize the RRC layer of the BaseStation with its paired cells
        # and their corresponding RRC layer objects.
        self.rrc_layer.set_paired_cellIdx_and_rrcLayers(
            self.paired_cells_idx,
            [self.baseStationsAll_Parent_obj.base_stations[paired_idx].rrc_layer for paired_idx in self.paired_cells_idx],
        )
        
        
        
        
        
    #### Private Layer Initializers
    ########################

    def _initialize_phy_layer(self):
        """
        Initializes the PHY layer for the base station.
        
        Returns:
            An instance of the PHY layer.
        """
        #print("Initializing PHY layer...")
        return None  # Replace with actual PHY initialization
    
    def _initialize_mac_layer(self):
        """
        Initializes the MAC layer for the base station.
        
        Returns:
            An instance of the MAC layer.
        """
        #print("Initializing MAC layer...")
        return macs.MAC(
            self.cell_index,
            self.cell_name,
            self.simulation_config_obj,
            self.network_deployment_obj,
            self.ue_deployment_obj,
            self.beam_conf_obj,
            self.best_serving_CSI_RS_per_ue_obj,
            self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj,
            self.baseStationsAll_Parent_obj
        )
    
    def _initialize_rlc_layer(self):
        """
        Initializes the RLC layer for the base station.
        
        Returns:
            An instance of the RLC layer.
        """
        #print("Initializing RLC layer...")
        return None  # Replace with actual RLC initialization
    
    def _initialize_pdcp_layer(self):
        """
        Initializes the PDCP layer for the base station.
        
        Returns:
            An instance of the PDCP layer.
        """
        #print("Initializing PDCP layer...")
        return None  # Replace with actual PDCP initialization
    
    
    def _initialize_rrc_layer(self):
        """
        Initializes the RRC layer for the base station.
        
        Returns:
            An instance of the RRC layer.
        """
        return rrcs.RRC(
            self.cell_index,
            self.cell_name,
            self.simulation_config_obj,
            self.network_deployment_obj,
            self.network_deployment_obj.df_ep.loc[
                self.network_deployment_obj.df_ep["name"] == self.cell_name, 
                "cell_role"].iloc[0],
            self.initial_radio_unit_status,
            self.enable_carrier_shutdown,
            self.baseStationsAll_Parent_obj.bs_tx_power_obj,
            self.mac_layer, 
            self.baseStationsAll_Parent_obj
            )
        