# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 11:19:36 2025

@author: Matteo Bernabe
matteo.bernabe@ieee.org 
matteo.bernabe@iteam.upv.es
"""
import time 
import collections
import re
from typing import Any
import numpy as np
import pandas as pd
from giulia.config.sim_config import Simulation_Config
from giulia.bs.bs_deployments import Network
from giulia.bs.bs_tx_powers import BS_TX_Power
from giulia.mac.macs import MAC

from giulia.tools.tools import info, log_calculations_time
from giulia.logger import warning, error, info

class RRC:
    """
    Represents the Radio Resource Control (RRC) layer for a cellular base station.

    The RRC layer is responsible for managing radio resources, handling connection
    establishment, maintenance, and release for User Equipment (UEs), and
    coordinating with other base stations (e.g., via Xn interface) for resource
    sharing and inter-cell interference management.
    """

    def __init__(
        self,
        cell_index: int,
        cell_name: str,
        simulation_config_obj: Simulation_Config,
        network_deployment_obj: Network,
        cell_role: str,
        initial_radio_unit_status: str,
        enable_carrier_shutdown: bool,
        bs_tx_power_obj: BS_TX_Power,
        mac_layer: MAC, 
        baseStationsAll_Parent_obj
    ):
        # --- Basic Cell Identification Attributes ---
        self.cell_index: int = cell_index
        self.cell_name: str = cell_name

        # --- Simulation Config Information ---
        self.simulation_config_obj: Simulation_Config = simulation_config_obj

        # --- Object References for Inter-Layer/Module Communication ---
        self.network_deployment_obj: Network = network_deployment_obj
        self.bs_tx_power_obj: BS_TX_Power = bs_tx_power_obj
        self.mac_layer: MAC = mac_layer
        self.baseStationsAll_Parent_obj = baseStationsAll_Parent_obj

        # --- RRC Layer Specific Attributes (Initialized to None or Default) ---
        self.cell_role:str = cell_role
        self.enable_carrier_shutdown: bool = enable_carrier_shutdown

        #--- Init Paired Cells Information Attributes
        self.pairedCell_PRB_activity_dict: dict = {}
        self.pairedCell_PRB_usage_dict: dict = {}
        

        # These will be set by an external call after RRC objects are created for all cells
        self.paired_cells_idx: list|np.ndarray = []
        self.paired_rrc_layer: list = []

        #--- Cell Status
        self.radio_unit_status: str = initial_radio_unit_status
        # Tracks the last known number of connected UEs
        self.last_known_number_connected_UE: int = -1  
        
        #### Init Carrier Shutdown Object
        self.carrier_shutdown_obj = Carrier_Shutdown(
            self.cell_index,
            self.cell_name,
            self.cell_role,
            self.bs_tx_power_obj,
            self, # Passing self (the RRC instance) to Carrier_Shutdown
            rescheduling_us = self.simulation_config_obj.event_scheduling_info['base_stations_obj']['carrier_shutdown']
        )

        #--- Init InBox Message Collections
        self.inBox_event_queue = collections.deque()
        

    #### Public Setters and Getters
    ###################################
    def set_radio_unit_status(self, status: str) -> None:
        self.radio_unit_status = status

    def set_paired_cellIdx_and_rrcLayers(self, paired_cells_idx: list|np.ndarray, paired_rrc_layer: list) -> None:
        """
        Sets the indices and RRC layer instances for paired cells.

        Args:
            paired_cells_idx (list|np.ndarray): A list or array of unique numerical identifiers for paired cells.
            paired_rrc_layer (list): A list of RRC instances for the paired cells.
        """
        self.paired_cells_idx = paired_cells_idx
        self.paired_rrc_layer = paired_rrc_layer

    def set_last_known_number_connected_UE(self, number_connected_UE: int) -> None:
        self.last_known_number_connected_UE: int = number_connected_UE
        return
    def get_last_known_number_connected_UE(self) -> int:
        return self.last_known_number_connected_UE


    #### Public Methods
    #######################
    def send_event(self, source_cell_id: int, destination_rrc: 'RRC', event_content: str) -> None:
        """
        Sends an event from this RRC_Layer to a specified destination (another RRC_Layer).
        This is a "push" operation: the sender actively places the event
        into the receiver's queue.

        It should be noted that the message should adhere to a specific format:
        Format: #Event_<name_event>_ID_<event_id>

        Args:
            source_cell_id (int): The index of the cell sending the event.
            destination_rrc (RRC): The RRC instance of the destination cell.
            event_content (str): The content of the event, following a specific format.
        """
        # event_format_pattern = r"^#Event_[a-zA-Z0-9_]+_ID_[a-zA-Z0-9_]+$"
        event_format_pattern = r"^#Event_[a-zA-Z0-9_]+$"

        if not re.fullmatch(event_format_pattern, event_content):
            raise ValueError(
                f"Event content '{event_content}' does not adhere to the required format: "
                "#Event_<name_event>_ID_<event_id>"
            )

        destination_rrc.inBox_event_queue.append((source_cell_id, event_content))

    
    
    #### Private Methods
    ########################
    def _reset_inbox_queue(self) -> None:
        """
        Clears all events from the RRC layer's inbox queue.
        """
        self.inBox_event_queue.clear()

    def _pop_specific_arrived_event(self, specific_event_content: str, filder_by_paired_cell: bool = True):
        """
        Scans inbox for specific event. Removes all occurrences of the found (sender, event_content) tuple.
        Other events maintain relative order.
        Returns: sender, event_content of found event, or (None, None); plus boolean if found.

        It should be noted that the message should adhere to this format:
        #Event_<name_event>_ID_<event_id>

        Args:
            specific_event_content (str): The specific content string to look for (e.g., "#CarrierShutdown_wakeUp").
            filder_by_paired_cell (bool): If True, filters events only from paired cells.

        Returns:
            tuple: A tuple containing (sender, event_content, found_flag).
                   `found_flag` is True if the event was found, False otherwise.
        """
        if not self.inBox_event_queue:
            return None, None, False

        found_specific_event_tuple = (None, None)
        temp_queue = collections.deque()
        target_event_tuple_to_remove = None

        while self.inBox_event_queue:
            current_sender_id, current_event_content = self.inBox_event_queue.popleft()

            if target_event_tuple_to_remove is not None and \
               (current_sender_id, current_event_content) == target_event_tuple_to_remove:
                continue

            if found_specific_event_tuple == (None, None):
                event_name_part = current_event_content.split("_")[1] if len(current_event_content.split("_")) > 1 else ""

                match_found = False
                if filder_by_paired_cell:
                    if (current_sender_id in self.paired_cells_idx) and \
                       (event_name_part == specific_event_content.lstrip('#Event_')):
                        match_found = True
                else:
                    if event_name_part == specific_event_content.lstrip('#Event_'):
                        match_found = True

                if match_found:
                    found_specific_event_tuple = (current_sender_id, current_event_content)
                    target_event_tuple_to_remove = (current_sender_id, current_event_content)
                    continue
                else:
                    temp_queue.append((current_sender_id, current_event_content))
            else:
                temp_queue.append((current_sender_id, current_event_content))

        self.inBox_event_queue = temp_queue
        return found_specific_event_tuple[0], found_specific_event_tuple[1], found_specific_event_tuple != (None, None)

     

    def _exists_specific_event(self, specific_event_content: str, filter_by_paired_cell: bool = True):
        """
        Checks if a specific event exists in the inBox_event_queue.
        Does NOT remove anything.
    
        Args:
            specific_event_content (str): The specific event string to look for 
                                          (e.g., "#Event_Shutdown_ID_42").
            filter_by_paired_cell (bool): If True, only considers events from paired_cells_idx.
    
        Returns:
            tuple: (sender, event_content, found_flag)
                   - found_flag = True if the event exists, False otherwise.
        """
        if not self.inBox_event_queue:
            return None, None, False
    
        for sender_id, event_content in self.inBox_event_queue:
            # Extract the event name part
            parts = event_content.split("_")
            event_name_part = parts[1] if len(parts) > 1 else ""
    
            if filter_by_paired_cell:
                if (sender_id in self.paired_cells_idx) and \
                   (event_name_part == specific_event_content.lstrip('#Event_')):
                    return sender_id, event_content, True
            else:
                if event_name_part == specific_event_content.lstrip('#Event_'):
                    return sender_id, event_content, True
    
        return None, None, False

        
      
#### Carrier Shutdown Object
############################
        
class Carrier_Shutdown:
    """
    Manages the carrier shutdown logic, distinguishing between
    'coverage' and 'capacity' cell roles to apply appropriate shutdown procedures.

    The class initializes with cell-specific information and RRC layer access,
    and dynamically sets a private run method based on the cell's role.
    """

    def __init__(self, cell_index: int, cell_name: str, cell_role: str, bs_tx_power_obj: BS_TX_Power,
                 rrc_layer: RRC, rescheduling_us:int):
        """
        Initializes the Carrier_Shutdown instance with cell details and RRC layer access.

        Args:
            cell_index (int): A unique numerical identifier for the cell.
            cell_name (str): A human-readable name for the cell.
            cell_role (str): Defines the role of the cell, either "coverage" or "capacity".
                             This determines the shutdown logic to be applied.
            rrc_layer (RRC): An instance of the RRC (Radio Resource Control) layer
                             associated with this cell, providing access to lower layers
                             like MAC for resource allocation information.
        """
        # --- Basic Cell Identification Attributes ---
        self.cell_index: int = cell_index   # Stores the unique index of the cell.
        self.cell_name: str = cell_name     # Stores the descriptive name of the cell.
        self.cell_role: str = cell_role     # Stores the role of the cell (e.g., "coverage", "capacity").

        # Object managing the transmit power setting of base station within the network
        self.bs_tx_power_obj: BS_TX_Power = bs_tx_power_obj

        # --- RRC Layer and Paired Cell Attributes ---
        self.rrc_layer: RRC = rrc_layer  # RRC layer object for the current cell.

        # --- Dynamic Method Assignment Based on Cell Role ---
        # Assigns the appropriate private run method based on the cell's role.
        # This allows for polymorphic behavior based on the cell type.
        self._set_role_specific_method_and_parameters()
            

        # Init Status for Carrier Shutdown
        self.radio_unit_status_carrierShutdown : str | Any = None
        self.radio_unit_status_carrierShutdown_memory: list[str] = []
        
        # To set the cell status the setter method must be used, 
        # as it is updating also RRC attributes
        self.set_radio_unit_status_carrierShutdown(self.rrc_layer.radio_unit_status)

        # Set cell specific Carrier Shutdown layer rescheduling
        self.rescheduling_us = rescheduling_us
        
        
    def _set_role_specific_method_and_parameters(self) -> None:
        """
        --- Dynamic Method Assignment Based on Cell Role ---
        Assigns the appropriate private run method based on the cell's role.
        This allows for polymorphic behavior based on the cell type.
        Carrier Shutdown thresholds parameters:
          Parameters are got from df_ep according if they are coverage or capacity cells.
          These parameters are set based on the network deployment configuration
          and are used to determine when to shut down or wake up the carrier.
        """
        assert isinstance(self.rrc_layer.network_deployment_obj.df_ep, pd.DataFrame), "df_ep must be a pandas DataFrame"
        df_ep = self.rrc_layer.network_deployment_obj.df_ep
        
        if self.cell_role == "coverage":
            # Assign the method for coverage cell behavior
            self._run_method = self._run_coverageCell
            # Set coverage cell wakeup for capacity cell parameter
            self.wakeUp_pairedCell__currentCell_usage_threshold: float|None = df_ep.loc[
                        df_ep['name'] == self.cell_name, 'carrier_wakeUp_pairedCell__currentCell_usage_threshold'].iloc[0]
            
        elif self.cell_role == "capacity":
            # Assign the method for capacity cell behavior
            self._run_method = self._run_capacityCell
            # Set capacity cell Carrier Shutdown parameter
            self.shutdown_currentCell_usage_threshold: float|None = df_ep.loc[
                        df_ep['name'] == self.cell_name, 'carrier_shutdown_currentCell_usage_threshold'].iloc[0]
            
        elif self.cell_role == "coverage_capacity":
            # Assign the method for mixed coverage and capacity cell behavior
            self._run_method = self._run_mixedCoverageCapacityCell
            # Set both coverage and capacity cell parameters
            self.wakeUp_pairedCell__currentCell_usage_threshold: float|None = df_ep.loc[
                        df_ep['name'] == self.cell_name, 'carrier_wakeUp_pairedCell__currentCell_usage_threshold'].iloc[0]
            self.shutdown_currentCell_usage_threshold: float|None = df_ep.loc[
                        df_ep['name'] == self.cell_name, 'carrier_shutdown_currentCell_usage_threshold'].iloc[0]
             
        else:
            # Raises an error if an unadmitted cell role is provided, ensuring valid configuration.
            raise ValueError(f"Error! In {self.__class__.__name__}, cell_role '{self.cell_role}' not admitted. "
                             "Accepted roles are 'coverage' or 'capacity'.")
        


    #### Run
    ############
    # These are the methods users of the class will primarily interact with.
    def run(self, rescheduling_us: int = -1) -> int:
        """
        Executes the shutdown logic based on the cell's role. This is a wrapper
        method that calls the dynamically assigned _run_method.
        """
        #### Get current state
        # Determine the new cell state based on current conditions.
        current_state = self.get_current_radio_unit_status_carrierShutdown()
        
        #### Get paired cells and paired rrc layers
        self.paired_cells_idx = self.rrc_layer.paired_cells_idx
        self.paired_rrc_layer = self.rrc_layer.paired_rrc_layer
        

        #### Get Physical Resource Block (PRB) info 
        # Get Physical Resource Block (PRB) usage data for the current cell
        PRB_cell_usage: float = self._get_mine_PRB_usage()

        # Get PRB activity and usage for all paired cells
        PRB_pairedCell_usage_dict: dict[str, float] = self._get_pairedCell_PRB_activity_and_usage()

        # Get paired cell roles
        paired_cell_role: dict[str, str] = self._get_paired_cell_role()

        #### Get number of connected UE info
        number_connected_UE: int = self._get_number_connected_UE()
        number_connected_UE_pairedCell_dict: dict = self._get_number_connected_UE_pairedCell()

        # Tracks the last known number of connected UEs.  
        # When the cell is active, this value reflects the current number of connected UEs.  
        # When the cell is shut down, it serves as a reference to the number of UEs  
        # that were connected prior to the shutdown.
        self.rrc_layer.set_last_known_number_connected_UE(number_connected_UE)


        # if  current_state=="carrier_shutdown":
        #     # When the cell is in carrier shutdown state, log a warning if there are connected UEs
        #     if number_connected_UE > 0:
        #         raise ValueError(f"Warning! In {self.cell_name} is shutdown but there are {number_connected_UE} connected UEs.")
        
        
        # Run method
        return self._run_method(PRB_cell_usage=PRB_cell_usage,
                                # PRB_cell_activity=PRB_cell_activity,
                                number_connected_UE=number_connected_UE,
                                PRB_pairedCell_usage_dict=PRB_pairedCell_usage_dict,
                                paired_cell_role=paired_cell_role,
                                # PRB_pairedCell_activity=PRB_pairedCell_activity,
                                number_connected_UE_pairedCell_dict=number_connected_UE_pairedCell_dict,
                                current_state=current_state,
                                rescheduling_us=rescheduling_us)
    
        
    #### Public Setters and Getters
    ###################################
    # These provide controlled access to the class's attributes.
    def set_radio_unit_status_carrierShutdown(self, new_status:str):
        # Set attribute new status
        self.radio_unit_status_carrierShutdown = new_status
        # Append to memory
        self.radio_unit_status_carrierShutdown_memory.append(new_status)

        # Set also in father RRC layer attribute
        self.rrc_layer.set_radio_unit_status(new_status)

    def get_current_radio_unit_status_carrierShutdown(self) -> str: # Changed return type to str as per your usage ("active", carrierShutDown)
        return self.radio_unit_status_carrierShutdown


    
    
    #### Private Methods
    ########################

    def _get_mine_PRB_usage(self) -> float:
        # Get the PRB usage for the current cell from its MAC layer
        return self.rrc_layer.mac_layer.scheduler_obj._prb_usage

    def _get_pairedCell_PRB_activity_and_usage(self) -> dict:
        # Get the PRB usage for all paired cells from their MAC layers
        return {rrc.cell_name : rrc.mac_layer.scheduler_obj._prb_usage for rrc in self.paired_rrc_layer}

    
    def _get_number_connected_UE(self) -> int:
        return self.rrc_layer.baseStationsAll_Parent_obj.best_serving_cell_ID_per_ue_based_on_SSB_obj.ues_per_cell[self.cell_index].astype(int)
    
    def _get_number_connected_UE_pairedCell(self) -> dict:
        number_connected_UE_pairedCell_dict = {}
        for rrc in self.paired_rrc_layer:
            number_connected_UE_pairedCell_dict["PRB_cell_activity"] =\
                rrc.baseStationsAll_Parent_obj.best_serving_cell_ID_per_ue_based_on_SSB_obj.ues_per_cell[rrc.cell_index].astype(int)
        
        return number_connected_UE_pairedCell_dict
    
    def _get_paired_cell_role(self) -> dict[str, str]:
        # Create a dictionary mapping paired cell names to their roles 
        return {rrc.cell_name: rrc.cell_role for rrc in self.paired_rrc_layer}
        
    
    #### Shutdown/Wakeup Condition Methods
    ######################################
    def _is_wakeup_condition_met(self, PRB_usage) -> bool:
        # Determine if the wakeup condition is met based on cell PRB usage
        if PRB_usage>self.wakeUp_pairedCell__currentCell_usage_threshold: 
            return True
        else: return False
    

    def _is_shutdown_condition_met(self, prb_usage: 
                                   float, prb_usage_paired_dict: dict[str, float], 
                                   paired_cell_role_dict: dict[str, str]) -> bool:
        """
        Determine if the shutdown condition is met based on PRB (Physical Resource Block) usage.
        
        Logic:
        - A cell is considered eligible for shutdown if:
            1. Its own PRB usage is less than or equal to the defined threshold.
            2. At least one paired cell has PRB usage below a secondary threshold (defined per cell).
        
        Args:
            prb_usage (float): The PRB usage of the current cell (0 to 1).
            prb_usage_paired (dict[str, float]): A dictionary where keys are paired cell identifiers 
                and values are their PRB usage [0, 1].
        
        Returns:
            bool: True if shutdown conditions are met, otherwise False.
        """

        # Basic Sanity checks
        assert self.shutdown_currentCell_usage_threshold is not None,\
              f"{self.cell_name} {self.__class__.__name__} shutdown_currentCell_usage_threshold must not be None"
        
        
        # Check Condition
        # Check if the current cell's PRB usage plus the PRB usage of at least one paired
        # coverage cell is below the shutdown threshold.
        # This is done by evaluating the role and PRB usage of each paired cell.
        if any( [("coverage" in paired_cell_role) and (prb_usage + prb_usage_paired <= self.shutdown_currentCell_usage_threshold)
                    for prb_usage_paired, paired_cell_role in zip(prb_usage_paired_dict.values(), paired_cell_role_dict.values())] ):
            return True
        else:
            return False
    
    #### Private _run Logics
    ########################
    # These are the core operational methods, called internally by 'run'.
    
    def _run_coverageCell(self, *args, **kwargs) -> int:
        """
        Controls cell activation/shutdown based on usage and state metrics.

        Behavior:
        - Keeps the coverage cell active at all times.
        - When wake-up conditions are met, iterates paired capacity cells and wakes up
        only the first eligible shutdown cell (one cell per run).
        - Before sending a wake-up event, checks whether an equivalent wake-up event
        for the paired cell already exists to avoid duplicates.
        - Schedules the paired cell's immediate reaction and re-enables its rescheduling.

        Returns:
            int: The event rescheduling time provided in kwargs (or -1 by default).
        """

        #### Retrieve values from kwargs with defaults
        ##############################################
        PRB_cell_usage = kwargs.get('PRB_cell_usage', None)
        current_state = kwargs.get('current_state', None)
        rescheduling_us = kwargs.get('rescheduling_us', -1)
        
        # Assert that all required parameters are provided
        assert current_state is not None, f"Error! {self.cell_name} current_state is None."
        assert PRB_cell_usage is not None, f"Error! {self.cell_name} PRB_cell_usage is None."

        ####################################
        #### Check if wake up condition is met
        
        if self._is_wakeup_condition_met(PRB_cell_usage):  # Condition for wake up is met
            # Loop through all paired RRC layers associated with this cell.
            # If multiple paired cells are shutdown, only the first one found will be woken up.
            # This is done by sorting the paired RRC layers based on the last known number of connected UEs.
            for paired_rrc in sorted(self.paired_rrc_layer,
                                     key=lambda rrc: rrc.get_last_known_number_connected_UE(),
                                     reverse=True):
        
                # If the paired cell does not support carrier shutdown, skip it
                if not paired_rrc.enable_carrier_shutdown:
                    continue
        
                # Get the current status (either "active" or "carrier_shutdown") of the paired cell
                paired_cell_current_status = paired_rrc.carrier_shutdown_obj.get_current_radio_unit_status_carrierShutdown()
        
                # If the paired cell is already active, no further action is needed for it
                if paired_cell_current_status == "active":
                    continue  # Skip to the next paired cell
        
                # If the paired cell is shutdown, wake it up (and no specific wake-up event exists yet)
                # This check prevents sending duplicate wake-up events, needed to avoid ping pong effect
                # in the case of future multiple coverage cells paired with the same capacity cell
                elif paired_cell_current_status == "carrier_shutdown" and not paired_rrc._exists_specific_event(
                    "#Event_CarrierShutdownWakeUp", filter_by_paired_cell=True)[2]:
                    
                    # Send a wakeup event to the paired cell
                    self.rrc_layer.send_event(
                        self.cell_index,                 # Index of the sender coverage cell
                        paired_rrc,                      # Target paired capacity cell
                        "#Event_CarrierShutdownWakeUp",  # Wakeup event identifier
                    )
        
                    # Add cell event to immediate react and wakeup, re-enable its rescheduling
                    self.rrc_layer.baseStationsAll_Parent_obj.eventDriven_obj.add_event(
                        0,                                # Time from now
                        paired_rrc.carrier_shutdown_obj,  # The object
                        'run',                            # The method
                        paired_rrc.carrier_shutdown_obj.rescheduling_us,  # The rescheduling time
                    )
        
                    # Break the loop after waking up the first found shutdown paired cell
                    break
        
                elif paired_cell_current_status == "carrier_shutdown" and paired_rrc._exists_specific_event(
                    "#Event_CarrierShutdownWakeUp", filter_by_paired_cell=True)[2]:
                    # If the paired cell is shutdown, but a wake-up event already exists the work is already done. break the loop

                    # -------------------------------------------------------------------------------------------
                    # Case 1: CAPACITY CELL PAIRED WITH MULTIPLE COVERAGE CELLS
                    # -------------------------------------------------------------------------------------------

                    # Retrieve information about the existing wake-up event
                    # wakeUp_sender_id, _, _ = paired_rrc._exists_specific_event(
                    #     "#Event_CarrierShutdownWakeUp", filter_by_paired_cell=True)

                    # # If this cell is the one that sent the wake-up event,
                    # # there is nothing more to do, so we can exit the loop.
                    # if wakeUp_sender_id == self.cell_index:
                    #     break  # Break the loop after finding an existing wake-up event
                    # # else: continue  # Continue, and potentially wake up another paired cell

                    # -------------------------------------------------------------------------------------------
                    # Case 2: CAPACITY CELL PAIRED WITH A SINGLE COVERAGE CELL
                    # -------------------------------------------------------------------------------------------

                    # If the paired cell is shutdown, but a wake-up event already exists the work is already done
                    # and a capacity cell is already waking up.                    
                    break
                

                else: # If the current status is not a valid expected state, raise an error
                    raise ValueError(
                        f"In _run_CoverageCell: invalid state for carrier shutdown method in {self.cell_name}. "
                        f"Value paired_cell_current_status '{paired_cell_current_status}' not valid."
                    )
        
        else:  # Condition for wake up is not met
            pass


        #### Final safety check
        # Ensure that the coverage cell itself is still active
        assert self.radio_unit_status_carrierShutdown == "active", (
            f"Error! Coverage cell should always be active. {self.cell_name} "
            f"radio_unit_status_carrierShutdown '{self.radio_unit_status_carrierShutdown}' not valid."
        )

        # Return the event rescheduling time
        return rescheduling_us
                

        


    def _run_capacityCell(self, *args, **kwargs) -> int:
        """
        Controls capacity-cell activation/shutdown based on usage and state metrics.
    
        Behavior:
        - When ACTIVE: checks shutdown conditions using current and paired cells' PRB usage.
          If conditions are met, powers down TX, transitions to "carrier_shutdown",
          and disables rescheduling (returns -1).
        - When CARRIER_SHUTDOWN: checks for a specific wake-up event; if received,
          restores initial TX power and transitions back to "active".
        - Returns the rescheduling time in microseconds (from kwargs or -1 by default).
    
        Keyword Args:
            PRB_cell_usage (float): Current cell PRB usage.
            PRB_pairedCell_usage_dict (dict): Paired-cell PRB usage levels.
            current_state (str): Current state of the cell ("active" or "carrier_shutdown").
            rescheduling_us (int, optional): Time to next evaluation in microseconds. Defaults to -1.
        """
        
        # Start timer       
        t_start = time.perf_counter()  
        
        # Retrieve values from kwargs with defaults
        PRB_cell_usage = kwargs.get('PRB_cell_usage', None)
        PRB_pairedCell_usage_dict = kwargs.get('PRB_pairedCell_usage_dict', None)
        paired_cell_role = kwargs.get('paired_cell_role', None)
        current_state = kwargs.get('current_state', None)
        rescheduling_us = kwargs.get('rescheduling_us', -1)
    
        # Assert that all required parameters are provided
        assert current_state is not None, f"Error! {self.cell_name} current_state is None."
        assert PRB_cell_usage is not None, f"Error! {self.cell_name} PRB_cell_usage is None."
        assert PRB_pairedCell_usage_dict is not None, f"Error! {self.cell_name} PRB_pairedCell_usage_dict is None."
        assert paired_cell_role is not None, f"Error! {self.cell_name} paired_cell_role is None."

        #### New State Computation and Transition
        #########################################
        new_state = current_state
        
        if current_state == "active":
            # If active, check if shutdown conditions are met.
            if self._is_shutdown_condition_met(PRB_cell_usage, PRB_pairedCell_usage_dict, paired_cell_role):
                
                # Transition from "active" to 'shutdown'
                # Power down the cell's transmission
                self.bs_tx_power_obj.shutdown_singleBS_tx(target_cells_name=self.cell_name)
                
                # Set new state
                new_state = "carrier_shutdown"
                
                # Set new rescheduling value
                # If the cell is shutting down, no reschedule is needed. 
                # The paired cell will re-enable this processes and resetting the reschedule in the pile of events
                rescheduling_us = -1
                
                #Log info
                log_calculations_time(f"RRC Layer {self.cell_name} carrier shutdown", t_start)
                # print(f"RRC Layer {self.cell_name} carrier shutdown")
        
                
    
        elif current_state == "carrier_shutdown":
            # If shut down, check for a specific wake-up event.
            _, _, wakeup_event_received_flag = self.rrc_layer._pop_specific_arrived_event(
                specific_event_content="#Event_CarrierShutdownWakeUp",
                filder_by_paired_cell=True)
    
            if wakeup_event_received_flag:
                # If the waking message arrived, then:
                # Set the power to initial condition
                self.bs_tx_power_obj.set_initial_singleBS_tx_power_dBm(target_cells_name=self.cell_name)
                # Set new state
                new_state = "active"
                
                # Log info
                log_calculations_time(f"RRC Layer {self.cell_name} carrier wake up", t_start)
                # print(f"RRC Layer {self.cell_name} carrier wake up")
                
    
        else: # Raise an error for any unexpected cell states.
            raise ValueError(f"Unexpected {self.cell_name} state encountered: {current_state}")
    
    
        #### Updates and Send Messages
        ##############################
        # Update the cell's state
        self.set_radio_unit_status_carrierShutdown(new_state)

        # Return the event rescheduling time
        return rescheduling_us
    
    
    def _run_mixedCoverageCapacityCell(self, *args, **kwargs) -> int:
        # Start timer       
        t_start = time.perf_counter()  
        
        # Retrieve values from kwargs with defaults
        PRB_cell_usage = kwargs.get('PRB_cell_usage', None)
        PRB_pairedCell_usage_dict = kwargs.get('PRB_pairedCell_usage_dict', None)
        paired_cell_role = kwargs.get('paired_cell_role', None)
        current_state = kwargs.get('current_state', None)
        rescheduling_us = kwargs.get('rescheduling_us', -1)
    
        # Assert that all required parameters are provided
        assert current_state is not None, f"Error! {self.cell_name} current_state is None."
        assert PRB_cell_usage is not None, f"Error! {self.cell_name} PRB_cell_usage is None."
        assert PRB_pairedCell_usage_dict is not None, f"Error! {self.cell_name} PRB_pairedCell_usage_dict is None."
        assert paired_cell_role is not None, f"Error! {self.cell_name} paired_cell_role is None."
        
        
        #### New State Computation and Transition
        #########################################
        new_state = current_state
        
        if current_state == "active":
            
            # Compute Conditions
            ## Condition for shutting down, as per capacity cell
            condition_shutdown_flag = self._is_shutdown_condition_met(PRB_cell_usage, PRB_pairedCell_usage_dict, paired_cell_role)
            ## Condition for wake up paired, as per coverage cell
            condition_wakeUp_paired = self._is_wakeup_condition_met(PRB_cell_usage)
            # Final check, both conditions cannot be True at the same time
            assert not (condition_shutdown_flag and condition_wakeUp_paired), \
                f"Error! RRC: In mixed (both coverage and capacity) {self.cell_name}, both 'condition_shutdown_flag' and 'condition_wakeUp_paired' cannot be True at the same time."
            
            
            #### Coverage behaviour
            # Check if the wakeup condition is met, and if so, wake up one paired cell
            if condition_wakeUp_paired:
                # Loop through all paired RRC layers associated with this cell.
                # If multiple paired cells are shutdown, only the first one found will be woken up.
                # This is done by sorting the paired RRC layers based on the last known number of connected UEs.
                for paired_rrc in sorted(self.paired_rrc_layer,
                                         key=lambda rrc: rrc.get_last_known_number_connected_UE(),
                                         reverse=True):
                    
                    # If the paired cell does not support carrier shutdown, skip it
                    if not paired_rrc.enable_carrier_shutdown:
                        continue
                    
                    # Get the current status (either "active" or "carrier_shutdown") of the paired cell
                    paired_cell_current_status = paired_rrc.carrier_shutdown_obj.get_current_radio_unit_status_carrierShutdown()
            
                    # If the paired cell is already active, no further action is needed for it
                    if paired_cell_current_status == "active":
                        continue  # Skip to the next paired cell
            
                    # If the paired cell is shutdown, wake it up (and no specific wake-up event exists yet)
                    # This check prevents sending duplicate wake-up events, needed to avoid ping pong effect
                    # in the case of future multiple coverage cells paired with the same capacity cell
                    elif paired_cell_current_status == "carrier_shutdown" and not paired_rrc._exists_specific_event(
                        "#Event_CarrierShutdownWakeUp", filter_by_paired_cell=True)[2]:
                        
                        # Send a wakeup event to the paired cell
                        self.rrc_layer.send_event(
                            self.cell_index,                 # Index of the sender coverage cell
                            paired_rrc,                      # Target paired capacity cell
                            "#Event_CarrierShutdownWakeUp",  # Wakeup event identifier
                        )
            
                        # Add cell event to immediate react and wakeup, re-enable its rescheduling
                        self.rrc_layer.baseStationsAll_Parent_obj.eventDriven_obj.add_event(
                            0,                                # Time from now
                            paired_rrc.carrier_shutdown_obj,  # The object
                            'run',                            # The method
                            paired_rrc.carrier_shutdown_obj.rescheduling_us,  # The rescheduling time
                        )
            
                        # Break the loop after waking up the first found shutdown paired cell
                        break

                    elif paired_cell_current_status == "carrier_shutdown" and paired_rrc._exists_specific_event(
                        "#Event_CarrierShutdownWakeUp", filter_by_paired_cell=True)[2]:
                        # If the paired cell is shutdown, but a wake-up event already exists the work is already done. break the loop

                        # -------------------------------------------------------------------------------------------
                        # Case 1: CAPACITY CELL PAIRED WITH MULTIPLE COVERAGE CELLS
                        # -------------------------------------------------------------------------------------------

                        # Retrieve information about the existing wake-up event
                        # wakeUp_sender_id, _, _ = paired_rrc._exists_specific_event(
                        #     "#Event_CarrierShutdownWakeUp", filter_by_paired_cell=True)

                        # # If this cell is the one that sent the wake-up event,
                        # # there is nothing more to do, so we can exit the loop.
                        # if wakeUp_sender_id == self.cell_index:
                        #     break  # Break the loop after finding an existing wake-up event
                        # # else: continue  # Continue, and potentially wake up another paired cell

                        # -------------------------------------------------------------------------------------------
                        # Case 2: CAPACITY CELL PAIRED WITH A SINGLE COVERAGE CELL
                        # -------------------------------------------------------------------------------------------

                        # If the paired cell is shutdown, but a wake-up event already exists the work is already done
                        # and a capacity cell is already waking up.                    
                        break
            
                    # If the current status is not a valid expected state, raise an error
                    else:
                        raise ValueError(
                            f"In _run_mixedCoverageCapacityCell: invalid state for carrier shutdown method in {self.cell_name}. "
                            f"Value paired_cell_current_status '{paired_cell_current_status}' not valid."
                        )
    

            #### Capacity behaviour
            # Check if shutdown condition is met, and if so, shutdown self
            if condition_shutdown_flag:
                # Transition from "active" to 'shutdown'
                # Power down the cell's transmission
                self.bs_tx_power_obj.shutdown_singleBS_tx(target_cells_name=self.cell_name)
                
                # Set new state
                new_state = "carrier_shutdown"
                
                # Set new rescheduling value
                # If the cell is shutting down, no reschedule is needed. 
                # The paired cell will re-enable this processes and resetting the reschedule in the pile of events
                rescheduling_us = -1
                
                #Log info
                log_calculations_time(f"RRC Layer {self.cell_name} carrier shutdown", t_start)
                # print(f"RRC Layer {self.cell_name} carrier shutdown")

        
        elif current_state == "carrier_shutdown":
            # If shut down, check for a specific wake-up event.
            _, _, wakeup_event_received_flag = self.rrc_layer._pop_specific_arrived_event(
                specific_event_content="#Event_CarrierShutdownWakeUp",
                filder_by_paired_cell=True)
    
            if wakeup_event_received_flag:
                # If the waking message arrived, then:
                # Set the power to initial condition
                self.bs_tx_power_obj.set_initial_singleBS_tx_power_dBm(target_cells_name=self.cell_name)
                # Set new state
                new_state = "active"
                
                # Log info
                log_calculations_time(f"RRC Layer {self.cell_name} carrier wake up", t_start)
                # print(f"RRC Layer {self.cell_name} carrier wake up")
        
        
        else: # Raise an error for any unexpected cell states.
            raise ValueError(f"Unexpected {self.cell_name} state encountered: {current_state}")
        
        
        #### Updates and Send Messages
        ##############################
        # Update the cell's state
        self.set_radio_unit_status_carrierShutdown(new_state)

        # Return the event rescheduling time
        return rescheduling_us
    

    def _run_coverageCell_HAPS_multiple_wakeup(self, *args, **kwargs) -> int:
        """
        Controls cell activation/shutdown based on usage and state metrics.

        Behavior:
        - Keeps the coverage cell active at all times.
        - When wake-up conditions are met, iterates paired capacity cells and wakes up
        only the first eligible shutdown cell (one cell per run).
        - Before sending a wake-up event, checks whether an equivalent wake-up event
        for the paired cell already exists to avoid duplicates.
        - Schedules the paired cell's immediate reaction and re-enables its rescheduling.

        Returns:
            int: The event rescheduling time provided in kwargs (or -1 by default).
        """

        #### Retrieve values from kwargs with defaults
        ##############################################
        PRB_cell_usage = kwargs.get('PRB_cell_usage', None)
        current_state = kwargs.get('current_state', None)
        rescheduling_us = kwargs.get('rescheduling_us', -1)
        
        # Assert that all required parameters are provided
        assert current_state is not None, f"Error! {self.cell_name} current_state is None."
        assert PRB_cell_usage is not None, f"Error! {self.cell_name} PRB_cell_usage is None."

        ####################################
        #### Check if wake up condition is met
        
        if self._is_wakeup_condition_met(PRB_cell_usage):  # Condition for wake up is met
            # Loop through all paired RRC layers associated with this cell.
            # If multiple paired cells are shutdown, only the first one found will be woken up.
            # This is done by sorting the paired RRC layers based on the last known number of connected UEs.
            for paired_rrc in sorted(self.paired_rrc_layer,
                                     key=lambda rrc: rrc.get_last_known_number_connected_UE(),
                                     reverse=True):
        
                # If the paired cell does not support carrier shutdown, skip it
                if not paired_rrc.enable_carrier_shutdown:
                    continue

                # Get the current status (either "active" or "carrier_shutdown") of the paired cell
                paired_cell_current_status = paired_rrc.carrier_shutdown_obj.get_current_radio_unit_status_carrierShutdown()
    
                # If the paired cell is already active, no further action is needed for it
                if paired_cell_current_status == "active":
                    continue  # Skip to the next paired cell


                #### NEW AD-HOC code ####
                #########################
                paired_multiple_idx: list = self.rrc_layer.cell_multiple_wakeup_id_dict[paired_rrc.cell_index]   # type: ignore
                multiple_paired_rrc_list: list = [self.rrc_layer.baseStationsAll_Parent_obj.base_stations[i].rrc_layer for i in paired_multiple_idx]
                multiple_paired_rrc_list.append(paired_rrc)
                del paired_rrc
                for multiple_paired_rrc in multiple_paired_rrc_list:

                    if not multiple_paired_rrc.enable_carrier_shutdown:
                        continue
        
                    # Get the current status (either "active" or "carrier_shutdown") of the paired cell
                    paired_cell_current_status = multiple_paired_rrc.carrier_shutdown_obj.get_current_radio_unit_status_carrierShutdown()
        
                    # If the paired cell is already active, no further action is needed for it
                    if paired_cell_current_status == "active":
                        continue  # Skip to the next paired cell
        
                    # If the paired cell is shutdown, wake it up (and no specific wake-up event exists yet)
                    # This check prevents sending duplicate wake-up events, needed to avoid ping pong effect
                    # in the case of future multiple coverage cells paired with the same capacity cell
                    elif paired_cell_current_status == "carrier_shutdown" and not multiple_paired_rrc._exists_specific_event(
                        "#Event_CarrierShutdownWakeUp", filter_by_paired_cell=True)[2]:
                        
                        # Send a wakeup event to the paired cell
                        self.rrc_layer.send_event(
                            self.cell_index,                 # Index of the sender coverage cell
                            multiple_paired_rrc,                      # Target paired capacity cell
                            "#Event_CarrierShutdownWakeUp",  # Wakeup event identifier
                        )
            
                        # Add cell event to immediate react and wakeup, re-enable its rescheduling
                        self.rrc_layer.baseStationsAll_Parent_obj.eventDriven_obj.add_event(
                            0,                                # Time from now
                            multiple_paired_rrc.carrier_shutdown_obj,  # The object
                            'run',                            # The method
                            multiple_paired_rrc.carrier_shutdown_obj.rescheduling_us,  # The rescheduling time
                        )
            
                        # Break the loop after waking up the first found shutdown paired cell
                        continue
        
                    elif paired_cell_current_status == "carrier_shutdown" and multiple_paired_rrc._exists_specific_event(
                        "#Event_CarrierShutdownWakeUp", filter_by_paired_cell=True)[2]:
                        # If the paired cell is shutdown, but a wake-up event already exists the work is already done. break the loop

                        # -------------------------------------------------------------------------------------------
                        # Case 1: CAPACITY CELL PAIRED WITH MULTIPLE COVERAGE CELLS
                        # -------------------------------------------------------------------------------------------

                        # Retrieve information about the existing wake-up event
                        # wakeUp_sender_id, _, _ = multiple_paired_rrc._exists_specific_event(
                        #     "#Event_CarrierShutdownWakeUp", filter_by_paired_cell=True)

                        # # If this cell is the one that sent the wake-up event,
                        # # there is nothing more to do, so we can exit the loop.
                        # if wakeUp_sender_id == self.cell_index:
                        #     break  # Break the loop after finding an existing wake-up event
                        # # else: continue  # Continue, and potentially wake up another paired cell

                        # -------------------------------------------------------------------------------------------
                        # Case 2: CAPACITY CELL PAIRED WITH A SINGLE COVERAGE CELL
                        # -------------------------------------------------------------------------------------------

                        # If the paired cell is shutdown, but a wake-up event already exists the work is already done
                        # and a capacity cell is already waking up.                    
                        continue
                

                    else: # If the current status is not a valid expected state, raise an error
                        raise ValueError(
                            f"In _run_CoverageCell: invalid state for carrier shutdown method in {self.cell_name}. "
                            f"Value paired_cell_current_status '{paired_cell_current_status}' not valid."
                        )
                break  # Break the outer loop after processing multiple paired cells

        else:  # Condition for wake up is not met
            pass


        #### Final safety check
        # Ensure that the coverage cell itself is still active
        assert self.radio_unit_status_carrierShutdown == "active", (
            f"Error! Coverage cell should always be active. {self.cell_name} "
            f"radio_unit_status_carrierShutdown '{self.radio_unit_status_carrierShutdown}' not valid."
        )

        # Return the event rescheduling time
        return rescheduling_us