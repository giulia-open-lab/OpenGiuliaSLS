# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 09:15:15 2024

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import importlib
import time

import numpy as np
import torch

from giulia.antenna import antenna_pattern_gains, antenna_arrays, array_steering_vectors
from giulia.assertions.aux_files import validate_shadowing_files
from giulia.bs import bs_deployments, base_stations, bs_tx_powers
from giulia.channel import channels, los_probabilities, path_losses, shadowing_gains, shadowing_maps
from giulia.channel import o2i_penetration_losses, fast_fading_gains, precoded_channel_gains, slow_channels, k_factor
from giulia.config import sim_config
from giulia.event_driven import eventDriven
from giulia.event_driven.snapshot_control import Snapshot_control
from giulia.kpis import calculate_noise, calculate_rss, calculate_rsrp, calculate_sinr, calculate_rate, \
    power_consumptions
from giulia.logger import debug
from giulia.mac import calculate_best_serving_beam
from giulia.mobility import mobilities
from giulia.outputs import save_performance
from giulia.phy import lut_bler_vs_sinrs, mutual_informations
from giulia.playground import distances_angles, sites, hotspots
from giulia.rrc import time_frequency_resources, mcs_resources, beam_configs, cell_selections, cell_re_selections
from giulia.tools.tools import log_calculations_time, log_elapsed_time
from giulia.ue import ue_deployments
from giulia.outputs.saveable import Saveable


class Giulia:

    def help(self):
        print("Help for Giulia Simulator:")
        print("\nScenario Model Options:")
        print("  - 'ITU_R_M2135_UMa', 'ITU_R_M2135_UMi', '3GPPTR36_814_Case_1',")
        print("    '3GPPTR36_814_Case_1_omni', '3GPPTR36_814_Case_1_single_bs', '3GPPTR36_814_Case_1_single_bs_omni',")
        print("    '3GPPTR38_901_UMa_C2', '3GPPTR38_901_UMa_lsc',")
        print("    '3GPPTR38_901_UMa_lsc_sn', '3GPPTR38_901_UMa_lsc_single_bs',")
        print("    '3GPPTR38_901_UMi_C2', '3GPPTR38_901_UMi_lsc',")
        print("    '3GPPTR36_777_UMa_AV', '3GPPTR36_777_UMi_AV',")
        print("    '3GPPTR38_811_Urban_NTN', '3GPPTR38_811_Dense_Urban_NTN',")
        print("    'ITU_R_M2135_UMa_multilayer', 'ITU_R_M2135_UMa_Umi_colocated_multilayer',")
        print("    'ITU_R_M2135_UMa_Umi_noncolocated_multilayer',")
        print("    '3GPPTR38_901_lsc_UMa_fr1_Umi_C_band_plus_fr3_noncolocated_multilayer',")
        print("    'Dataset'")

        print("\nUE Playground Model Options:")
        print("  - If a corresponding ITU or 3GPP scenario model is selected, no input is needed.")
        print("  - Alternatively: 'rectangular',")

        print("\nUser Distribution Options:")
        print("  - 'grid', 'uniform', 'uniform_with_hotspots',")
        print("    'inhomogeneous_per_cell', 'inhomogeneous_per_cell_with_hot_spots'")

        print("\nLink Direction Options:")
        print("  - 'downlink', 'uplink', 'downlink_uplink'")

        print("\nSave Results:")
        print("  - 0: Do not save results of the simulation")
        print("  - 1: Save results of the simulation")

        
    def __init__(self,
                 preset,
                 scenario_playground_model='ITU_R_M2135_UMa',
                 ue_playground_model= None,
                 ue_distribution='uniform',
                 ue_mobility = None,
                 link_direction='downlink',
                 wraparound=None,
                 save_results = 0,
                 plots = 0,
                 number_of_episods = 0,
                 regression=0, 
                 enable_GPU=True,
                 project_name = None,
                 additional_input={},
                 enable_saveable=True):
        

        # Enable / Disable Saveable Features
        if enable_saveable: Saveable.enable()
        else: Saveable.disable()
        
        # Validate that the shadowing files exist
        validate_shadowing_files()


        #### CUDA DEVICE ####
        #####################

        # Select device cuda if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        debug(f"Giulia running device: {self.device}")


        #### INPUTS ####
        ################

        # Managing UE playground model
        if ue_playground_model is None:
            ue_playground_model = scenario_playground_model

        # Managing Sionna
        spam_loader = importlib.util.find_spec('sionna')
        sn_indicator = spam_loader is not None

        # Managing UAV height
        if ue_playground_model == "3GPPTR36_777_UMa_AV" or ue_playground_model == "3GPPTR36_777_UMi_AV":
            uav_height_m = 300
        else:
            uav_height_m = None


        #### SHOW CONFIGURATION ####
        ############################
        debug(
            "#" * 81 + f"\nSimulation: Scenario_model = {scenario_playground_model}" + \
            f", UE_distribution = {ue_distribution}\n" + "#" * 81
        )
        
        
        #### PRE-DECLARE ATTRIBUTES ####
        ################################
        
        # Declare all attributes to ensure addresses exist
        self.simulation_config_obj = []
        self.site_deployment_obj = []
        self.ue_playground_deployment_obj = []
        self.ue_hotspot_deployment_obj = []
        self.network_deployment_obj = []
        self.cell_antenna_array_structure_obj = []
        self.cell_SSB_conf_obj = []
        self.cell_CSI_RS_conf_obj = []
        self.power_consumption_obj = []
        self.time_frequency_resource_obj = []
        self.mcs_resource_obj = []
        self.bler_vs_sinrs_lut_obj = []
        self.shadowing_map_cell_obj = []
        self.ue_deployment_obj = []
        self.ue_antenna_array_structure_obj = []
        self.distance_angles_ue_to_cell_obj = []
        self.antenna_pattern_gain_ue_to_cell_obj = []
        self.array_steering_vector_ue_to_cell_obj = []
        self.path_loss_ue_to_cell_obj = []
        self.shadowing_gain_ue_to_cell_obj = []
        self.slow_channel_gain_ue_to_cell_obj = []
        self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj = []
        self.best_serving_CSI_RS_per_ue_obj = []
        self.traffic_generator_ue_obj = []
        self.scheduler_obj = []
        self.eventDriven_obj = []        


        #### SIMULATION CONFIG ####
        ###########################
        project_name = scenario_playground_model + "_" + ue_distribution if project_name==None else project_name
        self.simulation_config_obj = \
            sim_config.Simulation_Config(
                preset,
                scenario_playground_model,
                ue_playground_model,
                ue_distribution,
                ue_mobility,
                link_direction,
                wraparound,
                save_results,
                plots,
                number_of_episods,
                regression,
                sn_indicator,
                uav_height_m,
                project_name,
                additional_input)     


        #### PLAYGROUND ####
        ####################

        # Creates the reference site locations
        self.site_deployment_obj = \
            sites.Site_siteDeployment(
                self.simulation_config_obj.wraparound,
                self.simulation_config_obj.scenario_playground_model)
        self.site_deployment_obj.process()

        # Creates the reference playground for UE distribution
        self.ue_playground_deployment_obj = \
            sites.Site_uePlaygroundDeployment(
                self.simulation_config_obj.wraparound,
                self.simulation_config_obj.ue_playground_model,
                self.simulation_config_obj.ue_distribution)
        self.ue_playground_deployment_obj.process()

        # Creates the reference hotspot for UE distribution
        self.ue_hotspot_deployment_obj = \
            hotspots.Hotspot(
                self.simulation_config_obj,
                self.ue_playground_deployment_obj)
        self.ue_hotspot_deployment_obj.process()


        #### BS DEPLOYMENT ####
        #######################

        # Creates network deployment
        self.network_deployment_obj = \
            bs_deployments.Network(
                self.simulation_config_obj,
                self.site_deployment_obj,
                self.ue_hotspot_deployment_obj)
        self.network_deployment_obj.process()

        self.simulation_config_obj.set_sn_indicator(self.network_deployment_obj)
        self.simulation_config_obj.set_instantaneous_RSRP(self.network_deployment_obj)


        #### BS ANTENNA ARRAYS ####
        ###########################

        # Creates site antenna arrays - one array per cell
        self.cell_antenna_array_structure_obj = \
            antenna_arrays.Antenna_Array_BS(
                self.simulation_config_obj,
                self.site_deployment_obj,
                self.network_deployment_obj,
                "cell")
        self.cell_antenna_array_structure_obj.process()


        #### BS SSB and CSI-RS PRECODER CODEBOOKS ####
        ##############################################

        # Creates SSB precoders
        self.cell_SSB_conf_obj = \
            beam_configs.Beam_Conf_SSB(
                self.simulation_config_obj,
                self.network_deployment_obj,
                "cell",
                "SSB")
        self.cell_SSB_conf_obj.process()

        # Creates CSI-RS precoders
        self.cell_CSI_RS_conf_obj = \
            beam_configs.Beam_Conf_CSI_RS(
                self.simulation_config_obj,
                self.network_deployment_obj,
                "cell",
                "CSI_RS")
        self.cell_CSI_RS_conf_obj.process()


        #### BS POWER CONSUMPTION #########
        ###################################

        # Creates power consumption model
        self.power_consumption_obj = \
            power_consumptions.Power_Consumption(
                self.simulation_config_obj,
                self.network_deployment_obj)
        self.power_consumption_obj.process()


        #### BS CELL (RE)SELECTION PARAMTERS  ####
        ##########################################

        #Configures cell (re)selection parameters
        self.cell_re_selection_conf_obj = \
            cell_re_selections.Cell_Re_Selection_Conf(
                self.simulation_config_obj,
                self.network_deployment_obj)
        self.cell_re_selection_conf_obj.process()


        #### TIME/FREQUENCY RESOURCES ####
        ##################################

        # Creates DL and UL carriers
        self.time_frequency_resource_obj = \
            time_frequency_resources.Time_Frequency_Resources(
                self.simulation_config_obj,
                self.network_deployment_obj,
                self.cell_antenna_array_structure_obj,            
                self.cell_SSB_conf_obj,
                self.cell_CSI_RS_conf_obj)
        self.time_frequency_resource_obj.process()


        #### MCS RESOURCES #########
        #############################

        # Creates modulation and coding scheme resources
        self.mcs_resource_obj = mcs_resources.MCS_Resource(self.simulation_config_obj)
        self.mcs_resource_obj.process()


        #### LINK LEVEL ABSTRACTIONS #########
        ######################################

        # Configures bler versus SINR look up tables
        self.bler_vs_sinrs_lut_obj = \
            lut_bler_vs_sinrs.Lut_Bler_Vs_Sinr(
                self.simulation_config_obj,
                self.mcs_resource_obj)
        self.bler_vs_sinrs_lut_obj.process()


        #### MUTUAL INFORMATION ####
        ############################

        # Configures mutual information tables
        if self.simulation_config_obj.sinr_mapping == "MIESM" :
            self.mi_obj = mutual_informations.Mutual_Information(self.simulation_config_obj)
            self.mi_obj.process()


        #### LOAD SHADOWING MAPS ####
        ###################

        # Calculates shadowing
        self.shadowing_map_cell_obj = \
            shadowing_maps.Shadowing_Map(
                0,
                self.simulation_config_obj,
                self.network_deployment_obj)
        self.shadowing_map_cell_obj.process()


    def configure(self, snapshot_index, max_time_ms):

        # rescheduling_us = 1e6

        ### Rescheduling information  ###
        ###############################
        scheduling_info = self.simulation_config_obj.event_event_scheduling_info

        ### Start time count  ###
        ###############################
        t_start = time.perf_counter()


        ### UPDATE SNAPSHOT CONTROL ###
        ###############################
        snapshot_control = Snapshot_control.get_instance()
        snapshot_control.num_snapshots = snapshot_index


        #### SET RANDOM SEED ####
        ####################

        # IMPORTANT: The random seed is increased by one unit per snapshot. This is done at the end of the foor loop
        self.simulation_config_obj.set_random_seed(snapshot_control.num_snapshots)


        #### EVENT-DRIVEN ####
        ####################

        self.eventDriven_obj = eventDriven.EventDrivenSimulation(1, max_time_ms)


        #### UEs ####
        #############

        # Creates UE deployment
        self.ue_deployment_obj = \
            ue_deployments.UE_Deployment(
                self.simulation_config_obj,
                self.network_deployment_obj,
                None,   #self.traffic_stats_cell_obj,
                self.ue_playground_deployment_obj,
                self.ue_hotspot_deployment_obj)
        self.ue_deployment_obj.process()


        #### UE ANTENNA ARRAYS ####
        ###########################

        # Creates UE antenna arrays - one array per UE
        self.ue_antenna_array_structure_obj = \
            antenna_arrays.Antenna_Array_UE(
                self.simulation_config_obj,
                self.site_deployment_obj,
                self.ue_deployment_obj,
                "ue")
        self.ue_antenna_array_structure_obj.process()
        
        
        #### UPDATE CARRIER STATS ####
        ##############################
             
        self.eventDriven_obj.add_event(0, 
                                       self.time_frequency_resource_obj, 
                                       'update_ue_info', 
                                       self.ue_deployment_obj,
                                       self.ue_antenna_array_structure_obj, 
                                       -1)         


        #### SIONNA MODULE ####
        ###########################

        # Configures Sionna channel model
        if self.simulation_config_obj.sn_indicator == True:

            self.channel_sn_obj = \
                channels.ChannelSn(
                    0,
                    self.simulation_config_obj,
                    self.site_deployment_obj,
                    self.network_deployment_obj,
                    self.ue_deployment_obj,
                    self.time_frequency_resource_obj,
                    self.cell_antenna_array_structure_obj,
                    self.ue_antenna_array_structure_obj)
            self.eventDriven_obj.add_event(0, self.channel_sn_obj, 'process', scheduling_info["channel_sn_obj"])

        else:
            self.channel_sn_obj = []
            
            
        #### BS TX Power ####
        #####################
            
        self.bs_tx_power_obj = bs_tx_powers.BS_TX_Power(self.network_deployment_obj)     
        self.eventDriven_obj.add_event(0, self.bs_tx_power_obj, 'process', scheduling_info["bs_tx_power_obj"])                


        #### DISTANCES ####
        ###################

        # Calculates distances UE to Cell
        self.distance_angles_ue_to_cell_obj = \
            distances_angles.Distance_Angles_ue_to_cell(self.simulation_config_obj,
                                             self.site_deployment_obj,
                                             self.network_deployment_obj,
                                             self.ue_deployment_obj)
        self.eventDriven_obj.add_event(0, self.distance_angles_ue_to_cell_obj, 'process', scheduling_info["distance_angles_ue_to_cell_obj"])

        # Calculates distance UE antennas to cell antennas
        self.distance_angles_ueAnt_to_cellAnt_obj = \
            distances_angles.Distance_Angles_ueAnt_to_cellAnt(self.simulation_config_obj,
                                             self.site_deployment_obj,
                                             self.cell_antenna_array_structure_obj,
                                             self.ue_antenna_array_structure_obj)
        self.eventDriven_obj.add_event(0, self.distance_angles_ueAnt_to_cellAnt_obj, 'process', scheduling_info["distance_angles_ueAnt_to_cellAnt_obj"])


        #### ANTENNA PATTERN GAIN ####
        ##############################

        # Calculates antenna pattern gains
        # Provided the azimuths and zeniths of the UEs,
        # we calculate the antenna gains of the respective antenna elements (or antenna arrays) from their antenna patterns
        self.antenna_pattern_gain_ue_to_cell_obj = \
            antenna_pattern_gains.Antenna_Pattern_Gain(self.simulation_config_obj,
                                                       self.network_deployment_obj,
                                                       self.ue_deployment_obj,
                                                       self.distance_angles_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.antenna_pattern_gain_ue_to_cell_obj, 'process', scheduling_info["antenna_pattern_gain_ue_to_cell_obj"])


        #### ARRAY STEERING VECTOR ####
        ######################

        # Calculates array steering vectors
        # Provided the geometries of the site arrays and the distance, azimuths and zeniths to the UEs,
        # we calculate the respective array steering vectors,
        # evaluated at the UE antennas
        # Note that an array steering vector represents the set of phase delays a plane wave experiences at a set of antenna elements
        self.array_steering_vector_ue_to_cell_obj = \
            array_steering_vectors.Array_Steering_Vector(self.simulation_config_obj,
                                                         self.network_deployment_obj,
                                                         self.ue_deployment_obj,
                                                         self.cell_antenna_array_structure_obj,
                                                         self.distance_angles_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.array_steering_vector_ue_to_cell_obj, 'process', scheduling_info["array_steering_vector_ue_to_cell_obj"])


        #### LINE OF SIGHT ####
        #######################

        # Calculates LoS probabilities UE to Cell
        self.LOS_probability_ue_to_cell_obj = \
            los_probabilities.LOSProbability(self.simulation_config_obj,
                                             self.network_deployment_obj,
                                             self.ue_deployment_obj,
                                             self.distance_angles_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.LOS_probability_ue_to_cell_obj, 'process', scheduling_info["LOS_probability_ue_to_cell_obj"])


        #### K-FACTOR ####
        ##################

        # Calculates K-factor
        self.K_factor_ue_to_cell_obj = \
            k_factor.KFactor(self.simulation_config_obj,
                              self.network_deployment_obj,
                              self.ue_deployment_obj,
                              self.distance_angles_ue_to_cell_obj,
                              self.LOS_probability_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.K_factor_ue_to_cell_obj, 'process', scheduling_info["K_factor_ue_to_cell_obj"])


        #### PATH LOSS ####
        ###################

        # Calculates path loss
        self.path_loss_ue_to_cell_obj = \
            path_losses.Path_Loss(self.simulation_config_obj,
                                  self.network_deployment_obj,
                                  self.ue_deployment_obj,
                                  self.distance_angles_ue_to_cell_obj,
                                  self.LOS_probability_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.path_loss_ue_to_cell_obj, 'process', scheduling_info["path_loss_ue_to_cell_obj"])


        #### O2I PENETRATION LOSS ####
        ##############################

        # Calculates O2I penetration losses
        self.o2i_penetration_loss_ue_to_cell_obj = \
            o2i_penetration_losses.O2I_Penetration_Loss(self.simulation_config_obj,
                                                        self.network_deployment_obj,
                                                        self.ue_deployment_obj,
                                                        self.distance_angles_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.o2i_penetration_loss_ue_to_cell_obj, 'process', scheduling_info["o2i_penetration_loss_ue_to_cell_obj"])


        #### SHADOWING ####
        ###################

        # Calculates shadowing
        self.shadowing_gain_ue_to_cell_obj = \
            shadowing_gains.Shadowing_Gain(self.simulation_config_obj,
                                           self.network_deployment_obj,
                                           self.ue_deployment_obj,
                                           self.ue_playground_deployment_obj,
                                           self.shadowing_map_cell_obj,
                                           self.LOS_probability_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.shadowing_gain_ue_to_cell_obj, 'process', scheduling_info["shadowing_gain_ue_to_cell_obj"])


        #### SLOW CHANNEL GAIN ####
        ###########################

        # Calculates slow channel gain
        # Slow-fading channels have variations that occur gradually and are generally associated with changes over longer periods
        # The slow channel gain takes into account several factors, including path loss, shadowing, and antenna gain
        self.slow_channel_gain_ue_to_cell_obj = \
            slow_channels.Slow_Channel(self.simulation_config_obj,
                                       self.network_deployment_obj,
                                       self.ue_deployment_obj,
                                       self.time_frequency_resource_obj,
                                       self.antenna_pattern_gain_ue_to_cell_obj,
                                       self.path_loss_ue_to_cell_obj,
                                       self.o2i_penetration_loss_ue_to_cell_obj,
                                       self.shadowing_gain_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.slow_channel_gain_ue_to_cell_obj, 'process', scheduling_info["slow_channel_gain_ue_to_cell_obj"])
        
    
        #### LoS CHANNEL ####
        #####################

        # Calculates the LoS channel, 
        # which takes into accont the LoS component of the fast fading model, i.e. the steering vectors and the K-factor weights
        # This is to compute average behaviours, as the NLoS component of the fast fading model has zero mean
        self.los_channel_gain_ue_to_cell_obj = \
            channels.LoS_Channel(self.simulation_config_obj,
                                 self.network_deployment_obj,
                                 self.time_frequency_resource_obj,
                                 self.cell_antenna_array_structure_obj,
                                 self.ue_antenna_array_structure_obj,
                                 self.slow_channel_gain_ue_to_cell_obj,
                                 self.array_steering_vector_ue_to_cell_obj,
                                 self.K_factor_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.los_channel_gain_ue_to_cell_obj, 'process', scheduling_info["los_channel_gain_ue_to_cell_obj"])    
        

        ##### SBB ####
        # Calculates long-term SSB precoded channel with LoS coefficients/channel
        self.SSB_precoded_channel_gain_no_fast_fading_ue_to_cell_obj = \
            precoded_channel_gains.Precoded_Channel_Gain_SSB_no_fast_fading_ue_to_cell(
                'LoS_channel',
                self.simulation_config_obj,
                self.network_deployment_obj,
                self.cell_antenna_array_structure_obj,
                self.cell_SSB_conf_obj,
                self.ue_deployment_obj,
                self.los_channel_gain_ue_to_cell_obj,
                self.array_steering_vector_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.SSB_precoded_channel_gain_no_fast_fading_ue_to_cell_obj, 'process', 
                                       scheduling_info["SSB_precoded_channel_gain_no_fast_fading_ue_to_cell_obj"])

        
        #### NOISE POWER ####
        #####################

        # Calculates noise power per resource element
        self.dl_noise_ue_to_cell_obj = \
            calculate_noise.Noise(
                self.network_deployment_obj,
                self.ue_deployment_obj)
        self.eventDriven_obj.add_event(0, self.dl_noise_ue_to_cell_obj, 'process', scheduling_info["dl_noise_ue_to_cell_obj"])       


        ##### SBB ####
        # Calculates long-term SSB RSRP - This includes SSB precoded channel with long-term LoS coefficients/channel
        # This is handy for cell association
        self.SSB_RSRP_no_fast_fading_ue_to_cell_obj = \
            calculate_rsrp.RSRP_SSB_no_fast_fading(
                'beam_RSRP_no_fast_fading',
                self.network_deployment_obj,
                self.cell_SSB_conf_obj,
                self.SSB_precoded_channel_gain_no_fast_fading_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.SSB_RSRP_no_fast_fading_ue_to_cell_obj, 'process', scheduling_info["SSB_RSRP_no_fast_fading_ue_to_cell_obj"]["process"])
        
        
        #### CELL SELECTION ####
        ########################

        ##### SBB ####
        # In this case, the UE connects to the cell that provides the best SSB RSRP in average over time
        self.best_serving_cell_ID_per_ue_based_on_SSB_obj = \
            cell_selections.Cell_Selection_SSB(
                'strongest_rsrp',
                self.simulation_config_obj,
                self.ue_playground_deployment_obj,
                self.ue_deployment_obj,
                self.cell_SSB_conf_obj,
                self.cell_re_selection_conf_obj,
                self.distance_angles_ue_to_cell_obj,
                self.SSB_precoded_channel_gain_no_fast_fading_ue_to_cell_obj,
                self.SSB_RSRP_no_fast_fading_ue_to_cell_obj,
                self.dl_noise_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.best_serving_cell_ID_per_ue_based_on_SSB_obj, 'process', 
                                       scheduling_info["best_serving_cell_ID_per_ue_based_on_SSB_obj"]["process"])    
        
        
        #### CELL RE-SELECTION ####
        ###########################

        ##### SBB ####
        self.cell_reselection_based_on_SSB_obj = \
            cell_re_selections.Cell_Re_Selection_SSB(
                'priority_plus_strongest_rsrp',
                self.simulation_config_obj,
                self.ue_deployment_obj,
                self.cell_SSB_conf_obj,
                self.cell_re_selection_conf_obj,
                self.best_serving_cell_ID_per_ue_based_on_SSB_obj,
                self.SSB_RSRP_no_fast_fading_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.cell_reselection_based_on_SSB_obj, 'process',    
                                       scheduling_info["cell_reselection_based_on_SSB_obj"])


        #### BEST SERVER STATS ####
        ###########################

        ##### SBB ####
        self.eventDriven_obj.add_event(
            0, 
            self.best_serving_cell_ID_per_ue_based_on_SSB_obj, 
            'calculate_server_stats',
            scheduling_info["best_serving_cell_ID_per_ue_based_on_SSB_obj"]["calculate_server_stats"])        
        
        #### UPDATE CARRIER STATS ####
        ##############################
        
        ##### SBB ####
        self.eventDriven_obj.add_event(
            0, 
            self.time_frequency_resource_obj, 
            'update_ue_carrier_info', 
            self.ue_antenna_array_structure_obj, 
            self.best_serving_cell_ID_per_ue_based_on_SSB_obj, 
            self.SSB_RSRP_no_fast_fading_ue_to_cell_obj,
            scheduling_info["time_frequency_resource_obj"]["update_ue_carrier_info"])           


        #### FAST FADING ####
        #####################

        # Calculates fast fading
        # Fast-fading channels experience rapid and unpredictable variations in signal strength due to factors like multipath interference, movement of objects, or other obstructions.
        # These variations can occur on a timescale of milliseconds or even faster
        self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj = \
            fast_fading_gains.Fast_Fading_Gain(self.simulation_config_obj,
                                               self.network_deployment_obj,
                                               self.cell_antenna_array_structure_obj,
                                               self.ue_antenna_array_structure_obj,
                                               self.time_frequency_resource_obj,
                                               self.distance_angles_ueAnt_to_cellAnt_obj,
                                               self.array_steering_vector_ue_to_cell_obj,
                                               self.K_factor_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj, 'process',
                                        scheduling_info["dl_fast_fading_gain_ueAnt_to_cellAnt_obj"])

        #### CHANNEL ####
        #################

        # Calculates the channel, which is the composite of the slow and fast channels
        # For convenience, we also calculate the LoS channel,
        # which takes into accont the LoS component of the fast fading model, i.e. the steering vectors and the K-factor weights
        # This is to compute average behaviours, as the NLoS component of the fast fading model has zero mean
        self.channel_gain_ue_to_cell_obj = \
            channels.Channel(self.simulation_config_obj,
                              self.network_deployment_obj,
                              self.time_frequency_resource_obj,
                              self.cell_antenna_array_structure_obj,
                              self.ue_antenna_array_structure_obj,
                              self.slow_channel_gain_ue_to_cell_obj,
                              self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj,
                              self.channel_sn_obj)
        self.eventDriven_obj.add_event(0, self.channel_gain_ue_to_cell_obj, 'process', 
                                       scheduling_info["channel_gain_ue_to_cell_obj"])


        #### PRECODED CHANNEL ####
        ##########################

        ##### SBB ####
        # Calculates SSB precoded channel with fast fading coefficients/channel
        # In this case, there is a dependency on the PRB
        self.SSB_precoded_channel_gain_ue_to_cell_obj = \
            precoded_channel_gains.Precoded_Channel_Gain_SSB_ue_to_cell(
                'complete_channel',
                self.simulation_config_obj,
                self.network_deployment_obj,
                self.cell_antenna_array_structure_obj,
                self.cell_SSB_conf_obj,
                self.ue_deployment_obj,
                self.channel_gain_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.SSB_precoded_channel_gain_ue_to_cell_obj, 'process', 
                                       scheduling_info["SSB_precoded_channel_gain_ue_to_cell_obj"])
        

        ##### CSI_RS ####
        # Calculates CSI-RS precoded channel with fast fading coefficients/channel
        # In this case, there is a dependency on the PRB
        self.CSI_RS_precoded_channel_gain_ue_to_cell_obj = \
            precoded_channel_gains.Precoded_Channel_Gain_CSI_RS_ue_to_cell(
                'complete_channel',
                self.simulation_config_obj,
                self.network_deployment_obj,
                self.cell_antenna_array_structure_obj,
                self.cell_CSI_RS_conf_obj,
                self.ue_deployment_obj,
                self.channel_gain_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.CSI_RS_precoded_channel_gain_ue_to_cell_obj, 'process',  
                                       scheduling_info["CSI_RS_precoded_channel_gain_ue_to_cell_obj"])


        #### RSS ####
        ##############

        ##### SBB ####
        # Calculates instantaneous SSB RSS - This includes SSB precoded channel with fast fading coefficients/channel
        # In this case, the RSS is NOT averaged across the PRBs
        self.SSB_RSS_per_PRB_ue_to_cell_obj = \
            calculate_rss.RSS_SSB(
                'beam_RSS_per_PRB',
                self.network_deployment_obj,
                self.cell_SSB_conf_obj,
                self.SSB_precoded_channel_gain_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.SSB_RSS_per_PRB_ue_to_cell_obj, 'process', 
                                       scheduling_info["SSB_RSS_per_PRB_ue_to_cell_obj"])

        ##### CSI_RS ####
        # Calculates instantaneous CSI-RS RSRP - This includes CSI-RS precoded channel with fast fading coefficients/channel
        # In this case, the RSS is NOT averaged across the PRBs
        self.CSI_RS_RSS_per_PRB_ue_to_cell_obj = \
            calculate_rss.RSS_CSI_RS(
                'beam_RSS_per_PRB',
                self.network_deployment_obj,
                self.cell_CSI_RS_conf_obj,
                self.CSI_RS_precoded_channel_gain_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.CSI_RS_RSS_per_PRB_ue_to_cell_obj, 'process', 
                                       scheduling_info["CSI_RS_RSS_per_PRB_ue_to_cell_obj"])


        #### RSRP ####
        ##############

        ##### SBB ####
        # Calculates instantaneous SSB RSRP - This includes SSB precoded channel with fast fading coefficients/channel
        # In this case, the RSS is averaged across the PRBs

        self.SSB_RSRP_ue_to_cell_obj = \
            calculate_rsrp.RSRP_SSB(
                'beam_RSRP_based_on_RSS',
                self.network_deployment_obj,
                self.cell_SSB_conf_obj,
                self.SSB_RSS_per_PRB_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.SSB_RSRP_ue_to_cell_obj, 'process', 
                                       scheduling_info["SSB_RSRP_ue_to_cell_obj"])


        ##### CSI_RS ####
        # Calculates instantaneous CSI_RS RSRP - This includes CSI_RS precoded channel with fast fading coefficients/channel
        # In this case, the RSS is averaged across the PRBs
        self.CSI_RS_RSRP_ue_to_cell_obj = \
            calculate_rsrp.RSRP_CSI_RS(
                'beam_RSRP_based_on_RSS',
                self.network_deployment_obj,
                self.cell_CSI_RS_conf_obj,
                self.CSI_RS_RSS_per_PRB_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.CSI_RS_RSRP_ue_to_cell_obj, 'process', 
                                       scheduling_info["CSI_RS_RSRP_ue_to_cell_obj"])


        if self.simulation_config_obj.instantaneous_RSRP == True:
            self.eventDriven_obj.add_event(0, self.SSB_RSRP_no_fast_fading_ue_to_cell_obj, 'set_RSRP_ue_to_cell_dBm',
                                            self.SSB_RSRP_ue_to_cell_obj, scheduling_info["SSB_RSRP_no_fast_fading_ue_to_cell_obj"]["set_RSRP_ue_to_cell_dBm"])


        #### BEST SERVING BEAM ####
        ###########################

        ##### CSI_RS ####
        # In this case, the UE uses the strongest CSI-RS of its best server in average over time. The full fast fading is considered
        self.best_serving_CSI_RS_per_ue_obj = \
            calculate_best_serving_beam.Beam_Selection(
                'strongest_rsrp',
                self.simulation_config_obj,
                self.network_deployment_obj,
                self.ue_playground_deployment_obj,
                self.ue_deployment_obj,
                self.cell_CSI_RS_conf_obj,
                self.CSI_RS_RSRP_ue_to_cell_obj, #CSI_RS_RSRP_no_fast_fading_ue_to_cell_obj,
                self.best_serving_cell_ID_per_ue_based_on_SSB_obj)
        self.eventDriven_obj.add_event(0, self.best_serving_CSI_RS_per_ue_obj, 'process', 
                                       scheduling_info["best_serving_CSI_RS_per_ue_obj"])


        #### GEOMETRY SINR ####
        #######################

        # Calculate geometry SINR based on antenna element/array, without considering  array gains
        # In other words, the antenna gain is only that of the element/array derived from the antenna patterns
        # This is a benchmark

        ##### SBB ####
        # Calculate geometry SINR based on SSB beams RSRPs - This uses CSI-RS precoded channel with long-term LoS coefficients/channel
        self.SSB_sinr_ue_to_cell_obj = \
            calculate_sinr.SINR_SSB(
                'sinr_based_on_beam_rsrp',
                "SSB",
                self.simulation_config_obj,
                self.site_deployment_obj,
                self.ue_playground_deployment_obj,
                self.ue_deployment_obj,
                self.time_frequency_resource_obj,
                None,
                self.SSB_RSRP_no_fast_fading_ue_to_cell_obj,
                self.best_serving_cell_ID_per_ue_based_on_SSB_obj,
                None)
        self.eventDriven_obj.add_event(0, self.SSB_sinr_ue_to_cell_obj, 'process', 
                                       scheduling_info["SSB_sinr_ue_to_cell_obj"])       
        
        
        #### INITIALIZE BS PROTOCOL STACK ####
        ######################################
        self.base_stations_obj = \
            base_stations.BaseStations(                
                self.simulation_config_obj,
                self.network_deployment_obj,
                self.ue_deployment_obj,
                self.cell_CSI_RS_conf_obj,
                self.best_serving_cell_ID_per_ue_based_on_SSB_obj,
                self.best_serving_CSI_RS_per_ue_obj,
                self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj, 
                self.SSB_sinr_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.base_stations_obj, 'process', 
                                       scheduling_info["base_stations_obj"])        
        

        #### DATA CHANNEL SINR  ####
        ############################

        ##### CSI_RS ####
        # Calculate SINR per PRB based on CSI-RS beams RSSs per PRB. This uses CSI-RS precoded channel with fast fading coefficients/channel per PRB
        # The PRB domain is a dimension of the output here
        # This is handy to calculate PRB performance
        self.CSI_RS_sinr_per_PRB_ue_to_cell_obj = \
            calculate_sinr.SINR_CSI_RS(
                'sinr_per_PRB_based_on_beam_rrs',
                "CSI_RS",
                self.simulation_config_obj,
                self.site_deployment_obj,
                self.ue_playground_deployment_obj,
                self.ue_deployment_obj,
                self.time_frequency_resource_obj,
                self.base_stations_obj,
                self.CSI_RS_RSS_per_PRB_ue_to_cell_obj,
                self.best_serving_cell_ID_per_ue_based_on_SSB_obj,
                self.best_serving_CSI_RS_per_ue_obj,
                self.mi_obj,
                self.bler_vs_sinrs_lut_obj)
        self.eventDriven_obj.add_event(0, self.CSI_RS_sinr_per_PRB_ue_to_cell_obj, 'process', 
                                       scheduling_info["CSI_RS_sinr_per_PRB_ue_to_cell_obj"])


        #### DATA CHANNEL RATES  ####
        #############################

        ##### CSI_RS ####
        # Calculate UE rate using UE SINR per PRB
        self.ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj = \
            calculate_rate.Rate_based_on_ins_CSI_RS_SINR(
                'theoretical_long_term_equal_resource_share_UE_throughput_based_on_prbSINR',
                self.simulation_config_obj,
                self.network_deployment_obj,
                self.ue_deployment_obj,
                self.cell_CSI_RS_conf_obj,
                self.best_serving_cell_ID_per_ue_based_on_SSB_obj,
                self.best_serving_CSI_RS_per_ue_obj,
                None,
                self.CSI_RS_sinr_per_PRB_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj, 'process', 
                                       scheduling_info["ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj"])

        ##### CSI_RS ####
        # Calculate UE rate using effective UE SINR
        self.ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_obj = \
            calculate_rate.Rate_based_on_eff_CSI_RS_SINR(
                'theoretical_long_term_equal_resource_share_UE_throughput_based_on_effSINR',
                self.simulation_config_obj,
                self.network_deployment_obj,
                self.ue_deployment_obj,
                self.cell_CSI_RS_conf_obj,
                self.best_serving_cell_ID_per_ue_based_on_SSB_obj,
                self.best_serving_CSI_RS_per_ue_obj,
                self.base_stations_obj,
                self.CSI_RS_sinr_per_PRB_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_obj, 'process', 
                                       scheduling_info["ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_obj"])


        #### NETWORK ENERGY CONSUMPTION  ####
        #####################################

        self.eventDriven_obj.add_event(
            0,
            self.power_consumption_obj,
            'calculate_power_consumption',
            self.network_deployment_obj,
            scheduling_info["power_consumption_obj"],
            np.full(len(self.network_deployment_obj.df_ep), 'active'), # cell_status =
            self.base_stations_obj )# PRB_load


        #### SAVE DATA PER SNAPSHOT  ####
        #################################

        self.performance_obj = \
            save_performance.Performance(
                self.simulation_config_obj,
                None,
                self.best_serving_cell_ID_per_ue_based_on_SSB_obj,
                None, 
                self.SSB_RSRP_no_fast_fading_ue_to_cell_obj,
                self.best_serving_CSI_RS_per_ue_obj,
                None, 
                self.SSB_sinr_ue_to_cell_obj,
                self.CSI_RS_sinr_per_PRB_ue_to_cell_obj,
                self.ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj,
                self.ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_obj,
                self.power_consumption_obj)
        self.eventDriven_obj.add_event(0, self.performance_obj, 'process', 
                                       scheduling_info["performance_obj"])
        
    
        #### Mobility  ####
        #####################################

        # Calculate UE rate using effective UE SINR
        if self.simulation_config_obj.ue_mobility != None:
            ue_mobility_obj = \
                mobilities.Mobility(
                    self.simulation_config_obj,
                    self.site_deployment_obj,
                    self.ue_playground_deployment_obj,
                    self.ue_deployment_obj)
            self.eventDriven_obj.add_event(0, ue_mobility_obj, 'process', 
                                           scheduling_info["ue_mobility_obj"])

        log_calculations_time('Giulia configuration', t_start)


    def run_simulation(self, step_index, decision_making_interval_ms):


        ### Start time count  ###
        ###############################
        t_start = time.perf_counter()


        #### EVENT-DRIVEN ####
        ####################
        self.eventDriven_obj.run(step_index, decision_making_interval_ms)
        
        log_elapsed_time('Giulia run', t_start, debug_prints=True)
        
        return self.performance_obj.r_e_ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps
