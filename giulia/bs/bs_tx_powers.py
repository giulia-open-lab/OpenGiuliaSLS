# -*- coding: utf-8 -*-
"""
Created on Thu May  8 08:24:22 2025

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org 
matteo.bernabe@iteam.upv.es
"""
from typing import List

import numpy as np
import pandas as pd
from giulia.tools import tools
from giulia.bs.bs_deployments import Network
from giulia.outputs.saveable import Saveable

class BS_TX_Power(Saveable):
    """
    Class to compute Base Station (BS) transmission power per Resource Element (RE)
    for different reference signals: CRS, SSB, and CSI-RS.

    This class assumes uniform power allocation across all available time-frequency
    resources and transmission ports/beams.
    """

    def __init__(self, network_deployment_obj: Network):
        """
        Initialize the TX power calculator with the network deployment object.

        Args:
            network_deployment_obj: Object containing the DataFrame `df_ep` 
                                    with cell deployment parameters.
        """
        super().__init__()
        
        self.network_deployment_obj: Network = network_deployment_obj
        # Basic checks
        if not isinstance(self.network_deployment_obj, Network):
            raise TypeError("network_deployment_obj must be an instance of Network class.")
        if not isinstance(self.network_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("The network_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")
        
        # Get Dataframe Engineering Parameters
        self.df_ep: pd.DataFrame = self.network_deployment_obj.df_ep

        
        # Extract relevant input parameters from the deployment DataFrame
        self.BS_tx_power_dBm: np.ndarray = self.df_ep["BS_tx_power_max_dBm"].to_numpy(dtype=np.single)  # Total BS TX power per cell (dBm)
        self.dl_PRBs: np.ndarray = self.df_ep["dl_PRBs_available"].to_numpy(dtype=int)           # Available DL PRBs per cell
        self.subcarriers_per_PRB: np.ndarray = self.df_ep["subcarriers_per_PRB"].to_numpy(dtype=int)  # Subcarriers per PRB
        self.num_cells: int = self.df_ep.shape[0]  # Number of cells
        self.radio_unit_initial_status: np.ndarray = self.df_ep["radio_unit_initial_status"].to_numpy(dtype=str)
    
        # Added counter to track process calls. Need for RRC cell activation
        self.process_Firstcall_done : bool = False

        # Store network dataframe engineering parameters for later use in RRC cell activation
        self.df_ep: pd.DataFrame = self.network_deployment_obj.df_ep

    @property
    def label(self) -> str: return 'cell_kpi'

    def expose_variables(self) -> List[str]:
        """List of attributes name to be saved."""
        return []

    #### Process ####
    ########################
    def process(self) -> None:
        
        # If this is the first process call, store the initial BS transmission power
        # This is useful for resetting powers levels later for RRC cell activation
        # Note that this function is here, as the initial power is dependednt on the 
        # power allocation strategy performed above
        if not self.process_Firstcall_done:
            self.initial_BS_tx_power_dBm = self.BS_tx_power_dBm.copy()
                    
            # Check valid values in radio_unit_initial_status
            if not np.isin(self.radio_unit_initial_status, list({"active", "carrier_shutdown"})).all():
                raise ValueError("Invalid values found")
            
            # Set TX power to 0 for inactive cells
            self.BS_tx_power_dBm = np.where(self.radio_unit_initial_status == "active", self.BS_tx_power_dBm, -np.inf)

            # Mark that the first call has been processed
            self.process_Firstcall_done = True
            
            
        # ---------------------------------------------------------------------------- #
        # NOTE: More sophisticated TX power allocation strategies could be implemented.
        # However, frequent changes in power distribution can lead to unstable 
        # interference patterns and impair link adaptation in practical deployments.
        # ---------------------------------------------------------------------------- #

        # Set Transmission power in netwrok base station
        self.set_uniform_split()

        return 
    
    
    #### Setter Methods ####
    ########################
    
    def set_BS_tx_power_dBm(self, BS_tx_power_dBm: np.ndarray) -> None:
        """
        Set transmission power per cell.
        """
        self.BS_tx_power_dBm = BS_tx_power_dBm
        
        # Process/Update power transmission levels
        self.process()
        
    def set_singleBS_tx_power_dBm(self, target_cells_name: str, value: float) -> None:
        """
        Set transmission power of a specific cell.
        """
        cell_index = self.network_deployment_obj.map_cellName_to_cellIndex[target_cells_name]
        self.BS_tx_power_dBm[cell_index] = value
    
    
    def set_dl_PRBs(self, dl_PRBs: np.ndarray) -> None:
        """
        Set the number of downlink PRBs per cell.
        """
        self.dl_PRBs = dl_PRBs
    
    
    def set_subcarriers_per_PRB(self, subcarriers_per_PRB: np.ndarray) -> None:
        """
        Set the number of subcarriers per PRB.
        """
        self.subcarriers_per_PRB = subcarriers_per_PRB


    def set_num_cells(self, num_cells: int) -> None:
        """
        Set the total number of cells.
        """
        self.num_cells = num_cells
        
        
    #### Setter for RRC Usage ####
    ##############################
    
    def shutdown_singleBS_tx(self, target_cells_name: str) -> None:
        """
        Shutdown specific cel.
        Shutdown is implemented by setting the transmission power of the cell to zero.
        """
        self.set_singleBS_tx_power_dBm(target_cells_name,-np.inf)
        
        # Process/Update power transmission levels
        self.process()
    
    
    def set_initial_singleBS_tx_power_dBm(self, target_cells_name: str) -> None:
        """
        Set the transmission power of a specific cell to its initial value.
        """
        cell_index = self.network_deployment_obj.map_cellName_to_cellIndex[target_cells_name]
        self.BS_tx_power_dBm[cell_index] = self.initial_BS_tx_power_dBm[cell_index]
        
        # Process/Update power transmission levels
        self.process()


    #### Power Setting Logics ####
    ##############################
    
    def set_uniform_split(self):
        """
        Calculates the TX power per Resource Element (RE) for CRS, SSB, and CSI-RS signals
        based on the assumption of uniform power distribution over frequency-time resources 
        and antennas/beams.

        This method adds new columns to `df_ep` with RE-level TX powers (in dBm):
        - "BS_tx_power_CRS_RE_dBm"
        - "BS_tx_power_SSB_RE_dBm"
        - "BS_tx_power_CSI_RS_RE_dBm"

        Args:
            rescheduling_us: Optional scheduling return value for synchronization, default -1.

        Returns:
            The unchanged rescheduling_us value.
        """

        # Get relevant input parameters
        BS_tx_power_dBm = self.BS_tx_power_dBm  # Total BS TX power per cell (dBm)
        dl_PRBs = self.dl_PRBs          # Available DL PRBs per cell
        subcarriers_per_PRB = self.subcarriers_per_PRB  # Subcarriers per PRB
        num_cells = self.num_cells  # Number of cells

        # ---------------------------------------------------------------------------- #
        # ASSUMPTION: The TX power of each BS is uniformly split across all
        # time-frequency REs and, where applicable, transmission ports/beams.
        # ---------------------------------------------------------------------------- #

        # -------- CRS (Common Reference Signal) --------
        # CRS uses fixed ports and is transmitted continuously.
        # We assume 1 port per cell for simplicity.
        self.BS_tx_power_CRS_RE_dBm = self._uniform_split(
            BS_tx_power_dBm,
            dl_PRBs,
            subcarriers_per_PRB,
            np.ones(num_cells)  # 1 port per cell
        )

        # -------- SSB (Synchronization Signal Block) --------
        # SSBs are transmitted using beam sweeping (one beam at a time).
        # As only one beam is active at any moment, the TX power is not split across beams.
        # Therefore, power per RE is the same as CRS.
        self.BS_tx_power_SSB_RE_dBm = self._uniform_split(
            BS_tx_power_dBm,
            dl_PRBs,
            subcarriers_per_PRB,
            np.ones(num_cells)  # 1 beam active at a time
        )

        # -------- CSI-RS (Channel State Information Reference Signal) --------
        # Multiple CSI-RS ports/beams can be active simultaneously per cell.
        # Hence, total TX power is equally divided among these beams in addition to REs.
        self.BS_tx_power_CSI_RS_RE_dBm = self._uniform_split(
            BS_tx_power_dBm,
            dl_PRBs,
            subcarriers_per_PRB,
            self.df_ep["CSI_RS_number_of_beams"].to_numpy(dtype=int)
        )


    def _uniform_split(
        self,
        BS_tx_power_max_dBm: np.ndarray,
        dl_PRBs_available: np.ndarray,
        subcarriers_per_PRB: np.ndarray,
        dl_ports: np.ndarray,
    ) -> np.ndarray:
        """
        Uniformly splits the BS transmission power over all Resource Elements and beams.

        TX power per RE is computed by:
        P_RE_dBm = P_total_dBm - 10*log10(N_PRBs) - 10*log10(N_subcarriers_per_PRB) - 10*log10(N_ports)

        Args:
            BS_tx_power_max_dBm: Total TX power per cell (dBm).
            dl_PRBs_available: Number of available PRBs per cell.
            subcarriers_per_PRB: Number of subcarriers per PRB (typically 12 in LTE/NR).
            dl_ports: Number of active ports/beams per cell.

        Returns:
            Per-RE TX power in dBm as a NumPy array.
        """
        return (
            BS_tx_power_max_dBm
            - tools.mW_to_dBm(dl_PRBs_available)
            - tools.mW_to_dBm(subcarriers_per_PRB)
            - tools.mW_to_dBm(dl_ports)
        )
