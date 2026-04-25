# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:48:13 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es"""

from giulia.tools import tools


def dl_re_tx_power_unifrom_split_dBm(BS_tx_power_dBm,
                                     dl_PRBs_available,
                                     dl_subcarrier_per_PRB,
                                     dl_ports):
    
    dl_re_tx_power_dBm = BS_tx_power_dBm - tools.mW_to_dBm(dl_PRBs_available) - tools.mW_to_dBm(dl_subcarrier_per_PRB) - tools.mW_to_dBm(dl_ports)
    
    return dl_re_tx_power_dBm

