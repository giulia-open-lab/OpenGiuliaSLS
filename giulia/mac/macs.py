# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 10:10:48 2024

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org 
matteo.bernabe@iteam.upv.es
"""

from giulia.mac import schedulers

class MAC:
    """
    Represents the MAC (Medium Access Control) layer of a base station in 4G/5G networks.
    """

    def __init__(self,
                 cell_index: int,
                 cell_name: str,  
                 simulation_config_obj,
                 network_deployment_obj,
                 ue_deployment_obj,
                 cell_CSI_RS_conf_obj,
                 best_serving_CSI_RS_per_ue_obj,
                 dl_fast_fading_gain_ueAnt_to_cellAnt_obj,
                 resource_allocation,
                 prb_requirement_generator_ue_obj):
        """
        Initializes the MAC layer and its functions, starting with the scheduler.
        
        Args:
            simulation_config_obj: Configuration object for the simulation.
            network_deployment_obj: Network deployment configuration.
            ue_deployment_obj: User equipment deployment configuration.
            cell_CSI_RS_conf_obj: CSI-RS configuration for cells.
            traffic_generator_ue_obj: Traffic generator for user equipment.
            best_serving_CSI_RS_per_ue_obj: Best serving CSI-RS per user equipment.
            dl_fast_fading_gain_ueAnt_to_cellAnt_obj: Downlink fast fading gain object.
        """
        # Store inputs
        self.cell_index = cell_index
        self.cell_name = cell_name          
        self.simulation_config_obj = simulation_config_obj
        self.network_deployment_obj = network_deployment_obj
        self.ue_deployment_obj = ue_deployment_obj
        self.cell_CSI_RS_conf_obj = cell_CSI_RS_conf_obj
        self.best_serving_CSI_RS_per_ue_obj = best_serving_CSI_RS_per_ue_obj
        self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj = dl_fast_fading_gain_ueAnt_to_cellAnt_obj
        self.resource_allocation = resource_allocation
        self.prb_requirement_generator_ue_obj = prb_requirement_generator_ue_obj

        # Extract scheduling model from the network deployment object
        scheduling_model = network_deployment_obj.df_ep.get("BS_scheduling_model", "").iloc[self.cell_index]

        # Initialize scheduler based on the scheduling model
        self._initialize_scheduler(scheduling_model)


    def _initialize_scheduler(self, scheduling_model: str):
        """
        Initializes the scheduler for the MAC layer based on the scheduling model.
    
        Args:
            scheduling_model: The scheduling model defined for the cell.
        """
        
        # #### SCHEDULING MODEL  ####
        # ###########################
        
        # Select the frequency domain scheduler based on the model
        if scheduling_model == "random_per_cell_beam":
            self.scheduler_obj = schedulers.SchedulerRandomPerBeam(
                self.cell_index,
                self.cell_name,  
                self.simulation_config_obj,
                self.network_deployment_obj,
                self.cell_CSI_RS_conf_obj,
                self.best_serving_CSI_RS_per_ue_obj,
                self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj,
                self.resource_allocation,
                self.prb_requirement_generator_ue_obj
            )
        elif scheduling_model == "simplex_based":
            self.scheduler_obj = schedulers.SchedulerSimplex(
                self.cell_index,
                self.cell_name,                 
                self.simulation_config_obj,
                self.network_deployment_obj,
                self.cell_CSI_RS_conf_obj,
                self.best_serving_CSI_RS_per_ue_obj,
                self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj,
                self.resource_allocation,
                self.prb_requirement_generator_ue_obj
            )
        else:
            raise ValueError(f"Unsupported scheduling model: {scheduling_model}")


    def _run(self):
        """
        Initializes the functioning of the scheduler
        """    
        self.scheduler_obj._run()
