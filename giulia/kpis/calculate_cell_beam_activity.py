# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 10:39:07 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import numpy as np 

def cell_activity(number_of_cells, best_serving_cell_ID):
    # This module only applies for a non-beam system 

    # Get the cells in use and how many times they have been used 
    unique, counts = np.unique(best_serving_cell_ID, return_counts=True)
    
    # Create a mask with True being the cells that are in use
    cell_activity_mask = np.zeros(number_of_cells, dtype=bool)
    cell_activity_mask[unique] = True
    cell_activity_mask = cell_activity_mask[np.newaxis,:] * np.ones((np.size(best_serving_cell_ID),1))
    
    # Calculate the number of UEs per cell
    ues_per_cell = np.zeros(number_of_cells, dtype=int)
    ues_per_cell[unique] = counts
    
    return cell_activity_mask.astype(bool), ues_per_cell


def SSB_beam_activity(ssb_conf_obj, best_serving_SBB_beam_ID): 
    # In this case, we add processing, as not all SSB beams are transmitted simultaneously. A cell transmits one SSB at a time.
    # We assume that all cells using the same carrier have the same number of SSBs and that they are in sync, i.e. all cells transmit SSB0 at the same time, later SSB1, ...
    
    # Get mapping between the general ID of the beam in the network and the local ID of the beam in its cell
    beam_ID_in_host_cell = ssb_conf_obj.df_beam_info["ID_in_host_node"].to_numpy(dtype=int)
    
    # Get the local ID of the best serving beam of each UE in its serving cell
    best_serving_beam_ID_in_host_cell = beam_ID_in_host_cell[best_serving_SBB_beam_ID]
    
    # Mask indicating which are the active SSB beams at the time that the UE took the measurement over its best serving SSB beam. 
    beam_activity_mask_per_ue = beam_ID_in_host_cell[np.newaxis,:] * np.ones((np.size(best_serving_beam_ID_in_host_cell),1)) == best_serving_beam_ID_in_host_cell[:,np.newaxis] * np.ones(np.size(beam_ID_in_host_cell))
    
    # Calculate how many UEs fall within each beam 
    # Get the SSB beams in use and how many times they have been used 
    unique, counts = np.unique(best_serving_SBB_beam_ID, return_counts=True)
    # Calculate the number of UEs per SSB beam
    ues_per_ssb_beam = np.zeros(np.size(ssb_conf_obj.df_beam_info,0), dtype=int)
    ues_per_ssb_beam[unique] = counts    
    
    return beam_activity_mask_per_ue.astype(bool), ues_per_ssb_beam


def CSI_RS_beam_activity(csi_rs_conf_obj, best_serving_CSI_RS_beam_ID): 
    # In this case, we add processing, as not all CSI_RS beams are transmitted simultaneously. A cell transmits only the CSI-RS beams for which there is at least a UE for which that beam was the best serving CSI-RS beam.
    
    # Find the active beams
    active_beams = np.isin(np.arange(0,np.size(csi_rs_conf_obj.df_beam_info,0)), best_serving_CSI_RS_beam_ID)
    
    # Mask indicating which are the active CSI-RS beams. Note that the same row is replicated as many times as UE there are 
    beam_activity_mask_per_ue = active_beams[np.newaxis,:] * np.ones((np.size(best_serving_CSI_RS_beam_ID),1)) 
    
    # Calculate how many UEs fall within each beam 
        # Get the CSI-RS beams in use and how many times they have been used 
    unique, counts = np.unique(best_serving_CSI_RS_beam_ID, return_counts=True)
        # Calculate the number of UEs per SSB beam
    ues_per_csi_rs_beam = np.zeros(np.size(csi_rs_conf_obj.df_beam_info,0), dtype=int)
    ues_per_csi_rs_beam[unique] = counts
    
    return beam_activity_mask_per_ue.astype(bool), ues_per_csi_rs_beam