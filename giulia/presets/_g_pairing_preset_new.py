# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:26:50 2026

@author: Matteo Bernabe
matteo.bernabe@ieee.org 
matteo.bernabe@iteam.upv.es
"""

import importlib
import importlib.util
import time

import torch

from giulia.event_driven import eventDriven
from giulia.event_driven.snapshot_control import Snapshot_control
from giulia.logger import debug
from giulia.tools.tools import log_calculations_time, log_elapsed_time
from giulia.assertions.aux_files import validate_shadowing_files

from giulia.config import sim_config
from giulia.playground import distances_angles, sites, hotspots
from giulia.bs import bs_deployments, base_stations
from giulia.ue import ue_deployments
from giulia.rrc import time_frequency_resources, mcs_resources, beam_configs, cell_selections, cell_re_selections
from giulia.phy import lut_bler_vs_sinrs, mutual_informations
from giulia.antenna import antenna_pattern_gains, antenna_arrays, array_steering_vectors
from giulia.channel import channels, los_probabilities, path_losses, shadowing_gains, shadowing_maps
from giulia.channel import o2i_penetration_losses, precoded_channel_gains, slow_channels, k_factor
from giulia.kpis import calculate_noise, calculate_rsrp
from giulia.outputs.saveable import Saveable


class Giulia_Pairing:
    def __init__(self,
                 preset,
                 ue_playground_model,
                 ue_distribution,
                 ue_mobility = None,
                 link_direction='downlink',
                 wraparound=None,
                 save_results = 0,
                 plots = 0,
                 number_of_episods = 0,
                 regression=0, 
                 enable_GPU=True,
                 **kwargs):
        
        # Disable Saveable Features
        Saveable.disable()

        # Validate that the shadowing files exist
        validate_shadowing_files()

        #### CUDA DEVICE ####
        #####################
        # Select device cuda if available
        self.enable_GPU = enable_GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() and enable_GPU else "cpu")
        
        #### INPUTS ####
        ################
        # Managing UE playground model
        # if ue_playground_model is None:
        #     ue_playground_model = scenario_playground_model

            # Managing Sionna
        spam_loader = importlib.util.find_spec('sionna')
        sn_indicator = spam_loader is not None

        # Managing UAV height
        if ue_playground_model == "3GPPTR36_777_UMa_AV" or ue_playground_model == "3GPPTR36_777_UMi_AV":
            uav_height_m = 300
        else: uav_height_m = None


        #### SIMULATION CONFIG ####
        ###########################

        # project_name = scenario_playground_model + "_" + ue_distribution if project_name==None else project_name
        self.simulation_config_obj = \
            sim_config.Simulation_Config(
                preset,
                None,
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
                enable_GPU,
                "Pairing",
                enable_saveable=False,
                **kwargs)


        #### PLAYGROUND ####
        ####################

        # # Creates the reference site locations
        # self.site_deployment_obj = \
        #     sites.Site_siteDeployment(
        #         self.simulation_config_obj.wraparound,
        #         self.simulation_config_obj.scenario_playground_model)
        # self.site_deployment_obj.process()

        # # Creates the reference playground for UE distribution
        # self.ue_playground_deployment_obj = \
        #     sites.Site_uePlaygroundDeployment(
        #         self.simulation_config_obj,
        #         self.simulation_config_obj.wraparound,
        #         self.simulation_config_obj.ue_playground_model,
        #         self.simulation_config_obj.ue_distribution)
        # self.ue_playground_deployment_obj.process()    

        # # Creates the reference hotspot for UE distribution
        # self.ue_hotspot_deployment_obj = \
        #     hotspots.Hotspot(
        #         self.simulation_config_obj,
        #         self.ue_playground_deployment_obj)
        # self.ue_hotspot_deployment_obj.process()


        # #### BS DEPLOYMENT ####
        # #######################

        # # Creates network deployment
        # self.network_deployment_obj = \
        #     bs_deployments.Network(
        #         self.simulation_config_obj,
        #         self.site_deployment_obj,
        #         self.ue_hotspot_deployment_obj)
        # self.network_deployment_obj.process()


        #### PLAYGROUND ####
        ####################
        # Get Legacy objects
        self.site_deployment_obj: sites.Site_siteDeployment|None = kwargs.get('site_deployment_obj', None)        
        self.ue_playground_deployment_obj: sites.Site_uePlaygroundDeployment|None = kwargs.get('ue_playground_deployment_obj', None)
        self.ue_hotspot_deployment_obj: hotspots.Hotspot|None = kwargs.get('ue_hotspot_deployment_obj', None)
        self.network_deployment_obj: bs_deployments.Network|None = kwargs.get('network_deployment_obj', None)
        # Basic Check
        if not isinstance(self.site_deployment_obj, sites.Site_siteDeployment):
            raise TypeError(f"In {self.__class__.__name__} __init__, site_deployment_obj must be an instance of Site_siteDeployment, got {type(self.site_deployment_obj)}")
        if not isinstance(self.ue_playground_deployment_obj, sites.Site_uePlaygroundDeployment):
            raise TypeError(f"In {self.__class__.__name__} __init__, ue_playground_deployment_obj must be an instance of Site_uePlaygroundDeployment, got {type(self.ue_playground_deployment_obj)}")
        if not isinstance(self.ue_hotspot_deployment_obj, hotspots.Hotspot):
            raise TypeError(f"In {self.__class__.__name__} __init__, ue_hotspot_deployment_obj must be an instance of Hotspot, got {type(self.ue_hotspot_deployment_obj)}")
        if not isinstance(self.network_deployment_obj, bs_deployments.Network):
            raise TypeError(f"In {self.__class__.__name__} __init__, network_deployment_obj must be an instance of Network, got {type(self.network_deployment_obj)}")
        
        # Update For Legacy Objects
        self.simulation_config_obj._set_ue_playground_deployment_obj(self.ue_playground_deployment_obj)
        self.simulation_config_obj._set_number_of_ues_per_UEsite_cell()

        # Set Sionna
        self.simulation_config_obj.set_sn_indicator(
            self.network_deployment_obj)
        self.simulation_config_obj.set_instantaneous_RSRP(
            self.network_deployment_obj)


        #### BS ANTENNA ARRAYS ####
        ###########################

        # Creates site antenna arrays - one array per cell
        self.cell_antenna_array_structure_obj = \
            antenna_arrays.Antenna_Array(
                self.simulation_config_obj,
                self.site_deployment_obj,
                self.network_deployment_obj,
                "cell")
        self.cell_antenna_array_structure_obj.process()


        #### BS SSB and CSI-RS PRECODER CODEBOOKS ####
        ##############################################

        # Creates SSB precoders
        self.cell_SSB_conf_obj = \
            beam_configs.Beam_Conf(
                self.simulation_config_obj,
                self.network_deployment_obj,
                "cell",
                "SSB")
        self.cell_SSB_conf_obj.process()

        # Creates CSI-RS precoders
        self.cell_CSI_RS_conf_obj = \
            beam_configs.Beam_Conf(
                self.simulation_config_obj,
                self.network_deployment_obj,
                "cell",
                "CSI_RS")
        self.cell_CSI_RS_conf_obj.process()

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
        self.mcs_resource_obj = \
            mcs_resources.MCS_Resource(self.simulation_config_obj)
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
            self.mi_obj = mutual_informations.Mutual_Information(self.simulation_config_obj,)
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



    def configure(self, snapshot_index, max_time_us):
        self.snapshot_index: int = snapshot_index
        
        # Basic type checks
        if not isinstance(self.simulation_config_obj, sim_config.Simulation_Config):
            raise TypeError(f"in {self.__class__.__name__}.configure simulation_config_obj must be an instance of Simulation_Config")

        if not isinstance(self.network_deployment_obj, bs_deployments.Network):
            raise TypeError(f"in {self.__class__.__name__}.configure network_deployment_obj must be an instance of Network")
        
        if not isinstance(self.ue_playground_deployment_obj, sites.Site):
            raise TypeError(f"in {self.__class__.__name__}.configure ue_playground_deployment_obj must be an instance of Site")
        
        if not isinstance(self.ue_hotspot_deployment_obj, hotspots.Hotspot):
            raise TypeError(f"in {self.__class__.__name__}.configure ue_hotspot_deployment_obj must be an instance of Hotspot")
        
        ### Rescheduling information  ###
        ###############################
        scheduling_info = self.simulation_config_obj.event_scheduling_info


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
        self.simulation_config_obj.set_random_seed(
            snapshot_control.num_snapshots)


        #### EVENT-DRIVEN ####
        ####################

        self.eventDriven_obj = \
            eventDriven.EventDrivenSimulation(
                1,
                max_time_us)
            
            
        #### UEs ####
        #############

        # Creates UE deployment
        self.ue_deployment_obj = \
            ue_deployments.UE_Deployment(
                self.simulation_config_obj,
                self.network_deployment_obj,
                self.ue_playground_deployment_obj,
                self.ue_hotspot_deployment_obj)
        self.ue_deployment_obj.process()



        #### UE ANTENNA ARRAYS ####
        ###########################

        # Creates UE antenna arrays - one array per UE
        self.ue_antenna_array_structure_obj = \
            antenna_arrays.Antenna_Array(
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

        else:  self.channel_sn_obj = []
            
            
        #### BS TX Power ####
        #####################
            
        # self.bs_tx_power_obj = \
        #     bs_tx_powers.BS_TX_Power(self.network_deployment_obj)     
        # self.eventDriven_obj.add_event(0, self.bs_tx_power_obj, 'process', scheduling_info["bs_tx_power_obj"])            


        #### DISTANCES ####
        ###################

        # Calculates distances UE to Cell
        self.distance_angles_ue_to_cell_obj = \
            distances_angles.Distance_Angles(self.simulation_config_obj,
                                             self.site_deployment_obj,
                                             self.network_deployment_obj,
                                             self.ue_deployment_obj)
        self.eventDriven_obj.add_event(0, self.distance_angles_ue_to_cell_obj, 'process', scheduling_info["distance_angles_ue_to_cell_obj"])

      

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

        #TO DO for cell to cell and ue to ue


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

        #TO DO for cell to cell and ue to ue


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
            precoded_channel_gains.Precoded_Channel_Gain(
                'LoS_channel',
                self.simulation_config_obj,
                self.network_deployment_obj,
                self.cell_antenna_array_structure_obj,
                self.cell_SSB_conf_obj,
                self.ue_deployment_obj,
                self.time_frequency_resource_obj,
                self.los_channel_gain_ue_to_cell_obj,
                self.array_steering_vector_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.SSB_precoded_channel_gain_no_fast_fading_ue_to_cell_obj, 'process', scheduling_info["SSB_precoded_channel_gain_no_fast_fading_ue_to_cell_obj"])

        
        #### NOISE POWER ####
        #####################

        # Calculates noise power per resource element
        self.dl_noise_ue_to_cell_obj = \
            calculate_noise.Noise(
                self.network_deployment_obj,
                self.ue_deployment_obj)
        self.eventDriven_obj.add_event(0, self.dl_noise_ue_to_cell_obj, 'process', scheduling_info["dl_noise_ue_to_cell_obj"])



        #### RSRP No fast Fading ####
        ##############

        # Calculates long-term SSB RSRP - This includes SSB precoded channel with long-term LoS coefficients/channel
        # This is handy for cell association
        self.SSB_RSRP_no_fast_fading_ue_to_cell_obj = \
            calculate_rsrp.RSRP(
                'beam_RSRP_no_fast_fading',
                self.simulation_config_obj,
                self.network_deployment_obj,
                self.cell_SSB_conf_obj,
                self.SSB_precoded_channel_gain_no_fast_fading_ue_to_cell_obj)
        self.eventDriven_obj.add_event(0, self.SSB_RSRP_no_fast_fading_ue_to_cell_obj, 'process', scheduling_info["SSB_RSRP_no_fast_fading_ue_to_cell_obj"]["process"])


        #### CELL SELECTION ####
        ########################

        ##### SBB ####
        # In this case, the UE connects to the cell that provides the best SSB RSRP in average over time
        self.best_serving_cell_ID_per_ue_based_on_SSB_obj = \
            cell_selections.Cell_Selection(
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
        self.eventDriven_obj.add_event(0, self.best_serving_cell_ID_per_ue_based_on_SSB_obj, 'process', scheduling_info["best_serving_cell_ID_per_ue_based_on_SSB_obj"]["process"])



        #### INITIALIZE BS PROTOCOL STACK ####
        ######################################
        self.base_stations_obj = \
            base_stations.BaseStations(
                self.eventDriven_obj,
                self.simulation_config_obj,
                self.network_deployment_obj,
                self.ue_deployment_obj,
                self.cell_CSI_RS_conf_obj,
                self.time_frequency_resource_obj,
                None, #self.SSB_RSRP_ue_to_cell_obj, 
                self.best_serving_cell_ID_per_ue_based_on_SSB_obj,
                None, # self.best_serving_CSI_RS_per_ue_obj,
                None, # self.dl_fast_fading_gain_ueAnt_to_cellAnt_obj, 
                None, # self.SSB_sinr_ue_to_cell_obj, 
                None  # self.power_consumption_obj)
            )
        

        #### ADDITIONAL ATTRIBUTE SETTING ####
        ######################################
        
        # These attributes cannot be set earlier because, at the time the
        # base_stations_obj is created, some of the required attributes
        # are not yet initialized or available.

        # List of attribute names (as strings) representing the objects
        rsrp_obj_names = [
            "SSB_RSRP_no_fast_fading_ue_to_cell_obj"
        ]
        
        # Iterate through all the names in the list.
        for name in rsrp_obj_names:
            # Dynamically set the `base_stations_obj` attribute of this object
            # to the common `base_stations_obj` value.
            setattr(getattr(self, name),
                    "base_stations_obj",
                    self.base_stations_obj)

        log_calculations_time('Giulia configuration Pairing', t_start)

        

    def run_simulation(self, step_index, decision_making_interval_us):
        self.step_index: int = step_index
    
        # Basic check object
        if not isinstance(self.eventDriven_obj, eventDriven.EventDrivenSimulation):
            raise TypeError("eventDriven_obj must be an instance of EventDriven")
        # Basic check object
        if not isinstance(self.simulation_config_obj, sim_config.Simulation_Config):
            raise TypeError("simulation_config_obj must be an instance of Simulation_Config")

        ### Start time count  ###
        ###############################
        t_start = time.perf_counter()


        #### EVENT-DRIVEN ####
        ####################
        self.eventDriven_obj.run(step_index, decision_making_interval_us)

        log_elapsed_time('Giulia run', t_start, debug_prints=True)

        # Saveable collection stays off for the whole pairing run (disabled in __init__), so that
        # these throw-away objects never replace the caller's ones. The registry is per execution
        # context, so "the caller's" means the one belonging to this thread: the pairing sub-run
        # shares it, which is exactly why the flag has to be turned off.
        # Restoring the caller's setting is deliberately NOT done here: this class cannot know it,
        # since its own Simulation_Config is always built with enable_saveable=False and therefore
        # only ever reports "off". PairingCell.process() snapshots the flag before building us and
        # puts it back once pairing is over.

        return True
    
    
    def reset(self, snapshot_index, max_time_us):
        return self.configure(snapshot_index, max_time_us)
    