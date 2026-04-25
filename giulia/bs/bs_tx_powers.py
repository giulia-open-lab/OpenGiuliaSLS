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

import numpy as np
import pandas as pd
from typing import Optional
from giulia.tools import tools

class BS_TX_Power:
    """
    Class to compute Base Station (BS) transmission power per Resource Element (RE)
    for different reference signals: CRS, SSB, and CSI-RS.

    This class assumes uniform power allocation across all available time-frequency
    resources and transmission ports/beams.
    """

    def __init__(self, network_deployment_obj):
        """
        Initialize the TX power calculator with the network deployment object.

        Args:
            network_deployment_obj: Object containing the DataFrame `df_ep` 
                                    with cell deployment parameters.
        """
        self.df_ep: pd.DataFrame = network_deployment_obj.df_ep


    def process(self, rescheduling_us: int = -1) -> int:
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

        # Extract relevant input parameters from the deployment DataFrame
        tx_power_dBm = self.df_ep["BS_tx_power_dBm"].to_numpy(dtype=np.single)  # Total BS TX power per cell (dBm)
        dl_PRBs = self.df_ep["dl_PRBs_available"].to_numpy(dtype=int)           # Available DL PRBs per cell
        subcarriers_per_PRB = self.df_ep["subcarriers_per_PRB"].to_numpy(dtype=int)  # Subcarriers per PRB
        num_cells = self.df_ep.shape[0]  # Number of cells

        # ---------------------------------------------------------------------------- #
        # ASSUMPTION: The TX power of each BS is uniformly split across all
        # time-frequency REs and, where applicable, transmission ports/beams.
        # ---------------------------------------------------------------------------- #

        # -------- CRS (Common Reference Signal) --------
        # CRS uses fixed ports and is transmitted continuously.
        # We assume 1 port per cell for simplicity.
        self.df_ep["BS_tx_power_CRS_RE_dBm"] = self._uniform_split(tx_power_dBm, dl_PRBs, subcarriers_per_PRB, np.ones(num_cells))  # 1 port per cell

        # -------- SSB (Synchronization Signal Block) --------
        # SSBs are transmitted using beam sweeping (one beam at a time).
        # As only one beam is active at any moment, the TX power is not split across beams.
        # Therefore, power per RE is the same as CRS.
        self.df_ep["BS_tx_power_SSB_RE_dBm"] = self._uniform_split(tx_power_dBm, dl_PRBs, subcarriers_per_PRB, np.ones(num_cells))  # 1 beam active at a time

        # -------- CSI-RS (Channel State Information Reference Signal) --------
        # Multiple CSI-RS ports/beams can be active simultaneously per cell.
        # Hence, total TX power is equally divided among these beams in addition to REs.
        self.df_ep["BS_tx_power_CSI_RS_RE_dBm"] = self._uniform_split(tx_power_dBm, dl_PRBs, subcarriers_per_PRB, self.df_ep["CSI_RS_number_of_beams"].to_numpy(dtype=int))

        # ---------------------------------------------------------------------------- #
        # NOTE: More sophisticated TX power allocation strategies could be implemented.
        # However, frequent changes in power distribution can lead to unstable 
        # interference patterns and impair link adaptation in practical deployments.
        # ---------------------------------------------------------------------------- #

        return rescheduling_us


    def _uniform_split(
        self,
        BS_tx_power_dBm: np.ndarray,
        dl_PRBs_available: np.ndarray,
        subcarriers_per_PRB: np.ndarray,
        dl_ports: np.ndarray,
    ) -> np.ndarray:
        """
        Uniformly splits the BS transmission power over all Resource Elements and beams.

        TX power per RE is computed by:
        P_RE_dBm = P_total_dBm - 10*log10(N_PRBs) - 10*log10(N_subcarriers_per_PRB) - 10*log10(N_ports)

        Args:
            BS_tx_power_dBm: Total TX power per cell (dBm).
            dl_PRBs_available: Number of available PRBs per cell.
            subcarriers_per_PRB: Number of subcarriers per PRB (typically 12 in LTE/NR).
            dl_ports: Number of active ports/beams per cell.

        Returns:
            Per-RE TX power in dBm as a NumPy array.
        """
        return (BS_tx_power_dBm - tools.mW_to_dBm(dl_PRBs_available) - tools.mW_to_dBm(subcarriers_per_PRB) - tools.mW_to_dBm(dl_ports))
