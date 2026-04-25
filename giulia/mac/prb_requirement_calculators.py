# -*- coding: utf-8 -*-
"""
Created on Mon May  5 15:23:24 2025

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author:  Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es

"""

import time

import numpy as np
import numpy.typing as npt
from giulia.tools.tools import log_calculations_time
from giulia.tools import tools
from giulia.kpis.calculate_sinr import SINR
from giulia.outputs.save_performance import Performance as output_module

class PRB_Requirement:
    
    def __init__(self, 
                 network_deployment_obj, 
                 ue_deployment_obj,
                 beam_conf_obj,
                 best_serving_beam_per_ue_obj, 
                 SSB_sinr_ue_to_cell_obj, 
                 best_serving_cell_ID_per_ue_based_on_SSB_obj):
        
        ##### Input storage 
        ########################
        self.network_deployment_obj = network_deployment_obj
        self.ue_deployment_obj = ue_deployment_obj  
        self.beam_conf_obj = beam_conf_obj
        self.best_serving_beam_per_ue_obj = best_serving_beam_per_ue_obj
        self.SSB_sinr_ue_to_cell_obj: SINR = SSB_sinr_ue_to_cell_obj
        self.CSI_RS_sinr_per_PRB_ue_to_cell_obj: None|SINR = None
        self.best_serving_cell_ID_per_ue_based_on_SSB_obj = best_serving_cell_ID_per_ue_based_on_SSB_obj
        
        ##### Outputs 
        ########################   
        # Placeholder to store the number of DL PRBs allocated to each user
        self.dl_PRBs_required_per_ue = []
        self.dl_PRBs_active_per_cell = 0
        
        
    def get_dl_PRBs_required_per_ue(self):
        return self.dl_PRBs_required_per_ue


    def _reset(self):
        self.dl_PRBs_active_per_cell = 0


    def _process(self, rescheduling_us=-1): 
        ##### Reset Inner Variables
        ########################
        self._reset()

        ##### Process inputs
        ########################
        
        # Network 
        # UE deployment
        self.traffic_generation_models = self.ue_deployment_obj.df_ep["traffic_generation_model"].to_numpy()  
        
        # Get the serving cell ID per UE based
        serving_cell_ue_to_cell = self.serving_cell_ue_to_cell = self.best_serving_cell_ID_per_ue_based_on_SSB_obj.best_serving_cell_ID_per_ue
       
        ##### Process outputs
        ########################
        self.dl_PRBs_required_per_ue = np.zeros(len(self.ue_deployment_obj.df_ep), dtype=int)
       
        
        ##### Start timer
        ########################  
        t_start = time.perf_counter()   


        ##### Switch
        ######################## 
        # Find the set of traffic generation models to process them independently
        # Define the required processing order for traffic generation models.
        # It is crucial that "rate_requirement" is processed before "strict_prb_fair_per_cell_beam":
        #   - "rate_requirement" assigns the minimum required PRBs
        #   - "strict_prb_fair_per_cell_beam" then distributes the assigned PRBs to beams
        # This specific sequence ensures correct PRB availability and allocation
        order_reference = ["rate_requirement", "strict_prb_fair_per_cell_beam"]
        traffic_generation_models_set = sorted(set(self.traffic_generation_models),
                                               key=lambda x: order_reference.index(x) if x in order_reference else len(order_reference))
        #############################################
        
                
        # Process each traffic generation model independently
        for traffic_generation_model in traffic_generation_models_set:
            
            # Identify users with the selected traffic generation model
            mask = traffic_generation_model ==  self.traffic_generation_models

            if (traffic_generation_model == "strict_prb_fair_per_cell_beam"): 
                # Fairly split the available PRBs per beam among UEs served by that beam
                out = self._fair_split_dl_PRBs_required_per_ue(mask)
                # Write back
                self.dl_PRBs_required_per_ue[mask] = out     
                del out           
                
            elif (traffic_generation_model == "rate_requirement"):
                """
                Note: This implementation assumes that the SINR used for data transmission 
                is the same as that measured from the SSB (Synchronization Signal Block) beams.
                """

                # Extract useful information per UE from dataframes and objects
                # Retrieve UE target data rates in Mbps for UEs defined
                ue_target_rate_Mbps = self.ue_deployment_obj.df_ep["ue_target_rate_Mbps"].to_numpy()[mask]
                
                # Get the SINR values for UEs
                sinr_ue_to_cell_dB = self.SSB_sinr_ue_to_cell_obj.SINR_results_per_frequency_layer["all_freq"]["sinr_ue_to_cell_dB"][mask]

                # Extract cell-specific parameters from the network dataframe (df_ep)
                subcarriers_per_PRB_servingCell = self.network_deployment_obj.df_ep["subcarriers_per_PRB"].to_numpy()[serving_cell_ue_to_cell][mask]
                dl_subcarrier_spacing_kHz_servingCell = self.network_deployment_obj.df_ep["dl_subcarrier_spacing_kHz"].to_numpy()[serving_cell_ue_to_cell][mask]
                
                # Check that no SINR values are None
                assert all(x is not None for x in sinr_ue_to_cell_dB), f"Error! in {self} sinr_ue_to_cell_dB values are None"
                
                # Compute the spectral efficiency for each UE-to-cell link using the Shannon capacity formula
                spectral_eff_ue_to_cell = np.log2(1 + tools.dBm_to_mW(sinr_ue_to_cell_dB))
                
                # Calculate the number of Physical Resource Blocks (PRBs) required per UE.
                # UE target rate are converted from Mbps to bps and subcarrier spacing from KHz to Hz
                dl_PRBs_required_per_ue_masked = np.ceil( (ue_target_rate_Mbps * 1e6) / 
                                                          (subcarriers_per_PRB_servingCell * (dl_subcarrier_spacing_kHz_servingCell * 1e3) * spectral_eff_ue_to_cell)  
                                                        ).astype(dtype=int)
                
                # Compute upper bound of DL PRBs required per UE based on fair split
                upper_bound_dl_PRBs_required_per_ue = self._fair_split_dl_PRBs_required_per_ue(mask)

                # Cap the required PRBs per UE to the upper bound
                self.dl_PRBs_required_per_ue[mask] = np.clip(dl_PRBs_required_per_ue_masked, 0, upper_bound_dl_PRBs_required_per_ue).astype(int)

                del mask
                
                
                # Calculate the total number of active PRBs per cell
                dl_PRBs_demanded_per_cell = np.bincount(serving_cell_ue_to_cell, 
                                                        weights=self.dl_PRBs_required_per_ue, 
                                                        minlength=self.network_deployment_obj.df_ep.values.shape[0]
                                                       ).astype(dtype=int)
                    
                # Check if the number of active PRBs per cell does not exceed the available PRBs
                cell_prb_availability_flag = dl_PRBs_demanded_per_cell <= self.network_deployment_obj.df_ep["dl_PRBs_available"].to_numpy()
                
                
                # If any cell has active PRBs exceeding the available PRBs, re-assign resources
                if not all(cell_prb_availability_flag):
                    # Do reassignment for each cell that meet the condition
                    for bs_idx in np.where(cell_prb_availability_flag==False)[0]:
                        self.dl_PRBs_required_per_ue [serving_cell_ue_to_cell == bs_idx] \
                            = self.re_allocate_resources_fullfillRequest(self.dl_PRBs_required_per_ue [serving_cell_ue_to_cell==bs_idx], 
                                                               self.network_deployment_obj.df_ep.iloc[bs_idx]["dl_PRBs_available"])
                            
                    # Update and total number of active PRBs per cell
                    dl_PRBs_demanded_per_cell = np.bincount(serving_cell_ue_to_cell, 
                                                            weights=self.dl_PRBs_required_per_ue, 
                                                            minlength=self.network_deployment_obj.df_ep.values.shape[0]
                                                           ).astype(dtype=int)
                    # Updated cell_prb_availability_flag
                    cell_prb_availability_flag = dl_PRBs_demanded_per_cell <= self.network_deployment_obj.df_ep["dl_PRBs_available"].to_numpy()
                    
                assert all(cell_prb_availability_flag), \
                    f"Error! Re-allocation of resources in {self.__class__} failed. Requested resources exceed available."
                    
                # Store the computed active PRBs per cell in the traffic generator object
                self.dl_PRBs_active_per_cell = dl_PRBs_demanded_per_cell
                del dl_PRBs_demanded_per_cell
                
            else: raise ValueError(f"Error, traffic_generators.traffic_generation_model not correctly specified. \
                                   Value '{traffic_generation_model}' not admitted")


        # Update final dl_PRBs_active_per_cell
        self.dl_PRBs_active_per_cell = np.bincount(serving_cell_ue_to_cell, 
                                                   weights=self.dl_PRBs_required_per_ue, 
                                                   minlength=self.network_deployment_obj.df_ep.values.shape[0]
                                                   ).astype(dtype=int)
    
        # Store the computed active PRBs per cell in the network deployment object
        self.network_deployment_obj.dl_PRBs_active_per_cell = self.dl_PRBs_active_per_cell.copy()
            

        ##### End 
        ########################
        log_calculations_time('Traffic generation', t_start)

        return rescheduling_us          


    # =============================================================================
    # Fair Split Functions
    # =============================================================================
    def _fair_split_dl_PRBs_required_per_ue(self, mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.int_]:
        """
        Fairly splits the available DL PRBs per beam among UEs served by that beam.
        Parameters:
            mask (np.array([bool])): Boolean mask to select UEs to consider

        Returns:
            np.array([int]): Fairly split DL PRBs required per UE
        """
        # PRBs per cell (in-place: subtract & clip)
        prbs_av_cell = np.asarray(self.network_deployment_obj.df_ep["dl_PRBs_available"].values,dtype=np.int64)
        np.subtract(prbs_av_cell, self.dl_PRBs_active_per_cell, out=prbs_av_cell, casting="unsafe")
        np.clip(prbs_av_cell, 0, None, out=prbs_av_cell)
    
        # Expand to PRBs per-beam
        prbs_av_beam = np.repeat(prbs_av_cell, self.beam_conf_obj.number_of_beams_per_node)
        total_beams = prbs_av_beam.size
    
        # Beam IDs (global) for masked UEs
        beam_ids = self.best_serving_beam_per_ue_obj.best_serving_beam_ID_per_ue[mask]
        ues_per_beam = np.bincount(beam_ids, minlength=total_beams)
    
        # Base per-UE PRBs via floor_divide (no divide-by-zero thanks to 'where')
        prbs_per_ue_base_by_beam = np.floor_divide(
            prbs_av_beam, ues_per_beam,
            out=np.zeros_like(prbs_av_beam),
            where=ues_per_beam > 0
        )
        out = prbs_per_ue_base_by_beam[beam_ids].copy()
    
        # Remainders per beam
        remainder = np.mod(
            prbs_av_beam, ues_per_beam,
            out=np.zeros_like(prbs_av_beam),
            where=ues_per_beam > 0
        )
    
        # ---- Vectorized remainder distribution ----
        n = beam_ids.size
        if n:
            # Stable sort by beam => original UE order preserved within each beam
            sort_idx = np.argsort(beam_ids, kind="mergesort")
            sorted_beams = beam_ids[sort_idx]
    
            # Group sizes for beams that actually have UEs, in ascending beam order
            nz_beams = np.flatnonzero(ues_per_beam)
            sizes = ues_per_beam[nz_beams]                      # lengths of each group
            starts = np.cumsum(np.r_[0, sizes[:-1]])            # start indices of groups (in sorted space)
    
            # Rank within each beam group: 0..(size-1)
            ranks_sorted = np.arange(n, dtype=np.int64) - np.repeat(starts, sizes)
    
            # Winners = first 'remainder[beam]' ranks
            winners_sorted = ranks_sorted < remainder[sorted_beams]
    
            # Scatter winners back to original order and add +1 PRB
            out[sort_idx] += winners_sorted.astype(out.dtype)

        return out


    # =============================================================================
    # Resource Re-Allocation Functions
    # =============================================================================
    def re_allocate_resources_fullfillRequest(self, requests: npt.NDArray[np.int_], total_resources: int):
        """
        Maximizes the number of fully satisfied requests under a total resource constraint
        using a Python-based approach
        
        Parameters:
            requests (np.array([int])): List of request amounts
            total_resources (int): The total available resources
        
        Returns:
            np.array([int]): Allocated resources corresponding to each request
            
            
        The function first sorts the requests (O(n log n)), then fulfills as many small ones 
        as possible in a single pass (O(n)). If resources remain, it distributes them 
        proportionally among the remaining requests, which involves computing fractional 
        allocations, sorting by decimal parts, and final assignment (another O(n log n))
    
        Time complexity:  O(n log n)  – dominated by sorting steps
        Space complexity: O(n)        – due to auxiliary lists and dictionaries
        
        Importantly, after conducting multiple analyses and tests, it has been proven that
        this version outperforms a NumPy-based implementation for n ≤ 65, i.e., 65 UEs per cell
        """
        # Keep track of original indices and sort requests in ascending order
        indexed_requests = list(enumerate(requests))
        indexed_requests.sort(key=lambda x: x[1])
        
        allocations = [0] * len(requests)
        remaining_resources = total_resources
        unfulfilled = []
        
        # Fully satisfy requests where possible
        for idx, req in indexed_requests:
            if req <= remaining_resources:
                allocations[idx] = req
                remaining_resources -= req
            else:
                unfulfilled.append((idx, req))
        
        # Proportionally distribute any remaining resources among unfulfilled requests
        if remaining_resources > 0 and unfulfilled:
            remaining_sum = sum(r for _, r in unfulfilled)
            # Calculate fractional allocations
            fractions = [(idx, r / remaining_sum * remaining_resources) for idx, r in unfulfilled]
            # Separate integer and fractional parts
            integers = {idx: int(f) for idx, f in fractions}
            decimals = sorted([(f - int(f), idx) for idx, f in fractions],reverse=True)
        
            # Assign integer parts
            for idx in integers:
                allocations[idx] = integers[idx]
        
            # Distribute leftover units based on highest fractional remainder
            leftover = remaining_resources - sum(integers.values())
            for _, idx in decimals[:leftover]:
                allocations[idx] += 1
                
        return allocations    
    