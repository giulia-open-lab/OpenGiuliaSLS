# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:40:53 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import itertools
import time
from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd

from giulia.bs.bs_deployments import Network
from giulia.channel.los_probabilities import LOSProbability
from giulia.config.sim_config import Simulation_Config
from giulia.event_driven.EventObject import EventObject
from giulia.event_driven.snapshot_control import Snapshot_control
from giulia.fs import results_file
from giulia.playground import Distance_Angles
from giulia.tools.tools import log_calculations_time
from giulia.ue.ue_deployments import UE_Deployment
from giulia.outputs.saveable import Saveable    


class KFactor(Saveable):
    K_factor_b_to_a_dB: npt.NDArray
    df_K_factor_b_to_a_dB: npt.NDArray

    rng: np.random.RandomState

    bs_propagation_models: npt.NDArray
    bs_fast_channel_models: npt.NDArray
    site_id: npt.NDArray

    h_ut_m: np.single

    zeniths_b_to_a_wraparound_degrees: npt.NDArray
    los_b_to_a: npt.NDArray

    def __init__(
            self,
            simulation_config_obj: Simulation_Config,
            network_deployment_obj: Network,
            ue_deployment_obj: UE_Deployment,
            distance_angles_ue_to_cell_obj: Distance_Angles,
            los_probability_ue_to_cell_obj: LOSProbability,
    ):

        super().__init__()

        ##### Input storage 
        ########################  
        self.simulation_config_obj = simulation_config_obj
        self.network_deployment_obj = network_deployment_obj
        self.ue_deployment_obj = ue_deployment_obj
        self.distance_angles_ue_to_cell_obj = distance_angles_ue_to_cell_obj
        self.LoS_probability_ue_to_cell_obj = los_probability_ue_to_cell_obj


    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["K_factor_b_to_a_dB"]


    def process(self, rescheduling_us: int = -1) -> int:

        ##### Process inputs
        ########################  

        # Random numbers
        self.rng = np.random.RandomState(self.simulation_config_obj.random_seed + 0)

        # Network
        self.bs_propagation_models = self.network_deployment_obj.df_ep["BS_propagation_model"].to_numpy()
        self.bs_fast_channel_models = self.network_deployment_obj.df_ep["BS_fast_channel_model"].to_numpy()
        self.site_id = self.network_deployment_obj.df_ep["site_ID"].to_numpy()

        # Users deployment 
        self.h_ut_m = self.ue_deployment_obj.df_ep[["position_z_m"]].to_numpy(dtype=np.single)

        # Channel characterisitics  
        self.zeniths_b_to_a_wraparound_degrees = self.distance_angles_ue_to_cell_obj.zeniths_b_to_a_wraparound_degrees
        self.los_b_to_a = self.LoS_probability_ue_to_cell_obj.los_b_to_a

        ##### Process outputs
        ########################

        self.K_factor_b_to_a_dB = np.zeros((np.size(self.los_b_to_a, 0), np.size(self.los_b_to_a, 1)), dtype=np.single)

        ##### Start timer
        ########################        

        t_start = time.perf_counter()

        ##### Switch
        ########################          

        # Find the set of unique propagation models to process them independently
        bs_propagation_models_set = set(self.bs_propagation_models)
        bs_fast_channel_models_set = set(self.bs_fast_channel_models)

        # Process each propagation model indepnedently
        for models in itertools.product(bs_propagation_models_set, bs_fast_channel_models_set):

            # Identify cells with the selected propagation model

            bs_propagation_model = models[0]
            bs_fast_channel_model = models[1]
            mask = np.bitwise_and(bs_propagation_model == self.bs_propagation_models, bs_fast_channel_model == self.bs_fast_channel_models)

            # To maintain correlation acorss sites: Get unique elements, indices of the first occurrences, and inverse indices 

            unique_elements, first_indices, inverse_indices = np.unique(self.site_id[mask], return_index=True, return_inverse=True)

            # Get necessary information of the identified cells
            zeniths_b_to_a_wraparound_degrees = self.zeniths_b_to_a_wraparound_degrees[:, first_indices]
            los_b_to_a = self.los_b_to_a[:, first_indices]

            # Calculate K-factor loss
            k_factor = None

            if bs_propagation_model == "3GPPTR38_901_UMa" and bs_fast_channel_model != "3GPPTR38_901_UMa":
                k_factor = self.k_factor_3gpptr38_901_uma(los_b_to_a)

            elif bs_propagation_model == "3GPPTR38_901_UMi" and bs_fast_channel_model != "3GPPTR38_901_UMi":
                k_factor = self.k_factor_3gpptr38_901_umi(los_b_to_a)

            elif bs_propagation_model == "ITU_R_M2135_UMa":
                k_factor = self.k_factor_itu_r_m2135_uma(los_b_to_a)

            elif bs_propagation_model == "ITU_R_M2135_UMi":
                k_factor = self.k_factor_itu_r_m2135_umi(los_b_to_a)

            elif bs_propagation_model == "3GPPTR36_814_Case_1":
                k_factor = self.k_factor_3gpptr36_814_case_1(los_b_to_a)

            elif bs_propagation_model == "3GPPTR36_777_UMa_AV":
                k_factor = self.k_factor_3gpptr36_777_uma_av(self.h_ut_m, los_b_to_a)

            elif bs_propagation_model == "3GPPTR36_777_UMi_AV":
                k_factor = self.k_factor_3gpptr36_777_umi_av(self.h_ut_m, los_b_to_a)

            elif bs_propagation_model == "3GPPTR38_811_Urban_NTN":
                k_factor = self.k_factor_3gpptr38_811_urban_ntn(zeniths_b_to_a_wraparound_degrees, los_b_to_a)

            elif bs_propagation_model == "3GPPTR38_811_Dense_Urban_NTN":
                k_factor = self.k_factor_3gpptr38_811_dense_urban_ntn(zeniths_b_to_a_wraparound_degrees, los_b_to_a)

            if k_factor is not None:
                self.K_factor_b_to_a_dB[:, mask] = np.take(k_factor, inverse_indices, axis=1)

        self.K_factor_b_to_a_dB = np.clip(self.K_factor_b_to_a_dB, 0, 10000)

        # Store in data frames the results as it may be useful to post process
        self.df_K_factor_b_to_a_dB = pd.DataFrame(self.K_factor_b_to_a_dB, columns=self.network_deployment_obj.df_ep["name"], index=self.ue_deployment_obj.df_ep["name"])

        ##### Save to plot
        ########################

        snapshot_control = Snapshot_control.get_instance()
        if self.simulation_config_obj.plot == 1 and snapshot_control.num_snapshots == 0:
            file_name = results_file(self.simulation_config_obj.project_name, 'to_plot_K_factor')
            np.savez(file_name, K_factor_b_to_a_dB=self.K_factor_b_to_a_dB[self.K_factor_b_to_a_dB != 0])

        ##### End
        log_calculations_time('K-factor', t_start)

        return rescheduling_us


    def k_factor_3gpptr38_901_uma(self, los_b_to_a: npt.NDArray) -> npt.NDArray:

        # Initialize path loss results 
        K_factor_b_to_a_dB = np.zeros((np.size(los_b_to_a, 0), np.size(los_b_to_a, 1)))

        # K-factor for LoS 
        aux_rnd = self.rng.randn(np.size(los_b_to_a, 0), np.size(los_b_to_a, 1))
        K_factor_b_to_a_dB[los_b_to_a == True] = 9 + 3.5 * aux_rnd[los_b_to_a == True]

        return K_factor_b_to_a_dB


    def k_factor_3gpptr38_901_umi(self, los_b_to_a: npt.NDArray) -> npt.NDArray:

        # Initialize path loss results
        K_factor_b_to_a_dB = np.zeros((np.size(los_b_to_a, 0), np.size(los_b_to_a, 1)))

        aux_rnd = self.rng.randn(np.size(los_b_to_a, 0), np.size(los_b_to_a, 1))
        K_factor_b_to_a_dB[los_b_to_a == True] = 9 + 5.0 * aux_rnd[los_b_to_a == True]

        return K_factor_b_to_a_dB


    def k_factor_itu_r_m2135_uma(self, los_b_to_a: npt.NDArray) -> npt.NDArray:

        # Initialize path loss results
        K_factor_b_to_a_dB = np.zeros((np.size(los_b_to_a, 0), np.size(los_b_to_a, 1)))

        aux_rnd = self.rng.randn(np.size(los_b_to_a, 0), np.size(los_b_to_a, 1))
        K_factor_b_to_a_dB[los_b_to_a == True] = 9 + 3.5 * aux_rnd[los_b_to_a == True]

        return K_factor_b_to_a_dB


    def k_factor_itu_r_m2135_umi(self, los_b_to_a: npt.NDArray) -> npt.NDArray:

        # Initialize path loss results
        K_factor_b_to_a_dB = np.zeros((np.size(los_b_to_a, 0), np.size(los_b_to_a, 1)))

        aux_rnd = self.rng.randn(np.size(los_b_to_a, 0), np.size(los_b_to_a, 1))
        K_factor_b_to_a_dB[los_b_to_a == True] = 9 + 5 * aux_rnd[los_b_to_a == True]

        return K_factor_b_to_a_dB


    def k_factor_3gpptr36_814_case_1(self, los_b_to_a: npt.NDArray) -> npt.NDArray:

        # Initialize path loss results
        K_factor_b_to_a_dB = np.zeros((np.size(los_b_to_a, 0), np.size(los_b_to_a, 1)))

        aux_rnd = self.rng.randn(np.size(los_b_to_a, 0), np.size(los_b_to_a, 1))
        K_factor_b_to_a_dB[los_b_to_a == True] = 9 + 3.5 * aux_rnd[los_b_to_a == True]

        return K_factor_b_to_a_dB


    def k_factor_3gpptr36_814_uma(self, h_ut_m: np.single, los_b_to_a: npt.NDArray) -> npt.NDArray:

        # Initialize path loss results
        K_factor_b_to_a_dB = np.zeros((np.size(los_b_to_a, 0), np.size(los_b_to_a, 1)))

        aux_rnd = self.rng.randn(np.size(los_b_to_a, 0), np.size(los_b_to_a, 1))
        K_factor_b_to_a_dB[los_b_to_a == True] = 9 + 3.5 * aux_rnd[los_b_to_a == True]

        return K_factor_b_to_a_dB


    def k_factor_3gpptr36_777_uma_av(self, h_ut_m: np.single, los_b_to_a: npt.NDArray) -> npt.NDArray:

        # Initialize path loss results
        K_factor_b_to_a_dB = np.zeros((np.size(los_b_to_a, 0), np.size(los_b_to_a, 1)))

        # Replicate the vector of ue heights in a column manner to facilitate next operation    
        h_ut_mat_m = h_ut_m * np.ones(np.size(los_b_to_a, 1))

        ## #--- Rician Factor - K_factor
        #    Aerial
        #    3GPP TR 36.777 V15.0.0 (2017-12),
        #    Annex B:	Channel modelling details, Alternative 2
        #    To deal with the altitude dependence, the mean value is used
        mask = np.logical_and(los_b_to_a == True, h_ut_mat_m >= 22.5)
        aux_rnd = self.rng.randn(np.size(h_ut_mat_m, 0), np.size(h_ut_mat_m, 1))
        K_factor_b_to_a_dB[mask] = (4.217 * np.log10(h_ut_mat_m[mask]) + 5.787) + (
                8.158 * np.exp(0.0046 * h_ut_mat_m[mask]) * aux_rnd[mask])

        #   Ground
        #   Table 7.5-6
        #   As it is for Aerial, to remove the randomness, the mean value is used
        mask = np.logical_and(los_b_to_a == True, h_ut_mat_m < 22.5)
        K_factor_b_to_a_dB[mask] = 9 + 3.5 * aux_rnd[mask]

        return K_factor_b_to_a_dB


    def k_factor_3gpptr36_777_umi_av(self, h_ut_m, los_b_to_a: npt.NDArray) -> npt.NDArray:

        # Initialize path loss results
        K_factor_b_to_a_dB = np.zeros((np.size(los_b_to_a, 0), np.size(los_b_to_a, 1)))

        # Replicate the vector of ue heights in a column manner to facilitate next operation
        h_ut_mat_m = h_ut_m * np.ones(np.size(los_b_to_a, 1))

        ## #--- Rician Factor - K_factor
        #    Aerial
        #    3GPP TR 36.777 V15.0.0 (2017-12),
        #    Annex B:	Channel modelling details, Alternative 2
        #    To deal with the altitude dependence, the mean value is used
        mask = np.logical_and(los_b_to_a == True, h_ut_mat_m >= 22.5)
        aux_rnd = self.rng.randn(np.size(h_ut_mat_m, 0), np.size(h_ut_mat_m, 1))
        K_factor_b_to_a_dB[mask] = (4.217 * np.log10(h_ut_mat_m[mask]) + 5.787) + (
                8.158 * np.exp(0.0046 * h_ut_mat_m[mask]) * aux_rnd[mask])

        #   Ground
        #   Table 7.5-6
        #   As it is for Aerial, to remove the randomness, the mean value is used
        mask = np.logical_and(los_b_to_a == True, h_ut_mat_m < 22.5)
        K_factor_b_to_a_dB[mask] = 9 + 5 * aux_rnd[mask]

        return K_factor_b_to_a_dB


    def k_factor_3gpptr38_811_urban_ntn(
            self,
            zeniths_b_to_a_wraparound_degrees: npt.NDArray,
            los_b_to_a: npt.NDArray
    ) -> npt.NDArray:

        # Initialize path loss results 
        K_factor_b_to_a_dB = np.zeros((np.size(los_b_to_a, 0), np.size(los_b_to_a, 1)))

        # Elevation angle 
        elevation_b_to_a_wraparound_degrees = zeniths_b_to_a_wraparound_degrees - 90

        # Calculate normal random variable to aid lognormal distribution
        aux_rnd = self.rng.randn(np.size(los_b_to_a, 0), np.size(los_b_to_a, 1))

        # Calculate mask for LoS and elevation range 
        # Calculate K-factor for LoS

        mask = np.logical_and(los_b_to_a == True, np.logical_and(elevation_b_to_a_wraparound_degrees >= 0, elevation_b_to_a_wraparound_degrees < 15))
        K_factor_b_to_a_dB[mask] = 31.83 + 13.84 * aux_rnd[mask]

        mask = np.logical_and(los_b_to_a == True, np.logical_and(elevation_b_to_a_wraparound_degrees >= 15, elevation_b_to_a_wraparound_degrees < 25))
        K_factor_b_to_a_dB[mask] = 18.78 + 13.78 * aux_rnd[mask]

        mask = np.logical_and(los_b_to_a == True, np.logical_and(elevation_b_to_a_wraparound_degrees >= 25, elevation_b_to_a_wraparound_degrees < 35))
        K_factor_b_to_a_dB[mask] = 10.49 + 10.42 * aux_rnd[mask]

        mask = np.logical_and(los_b_to_a == True, np.logical_and(elevation_b_to_a_wraparound_degrees >= 35, elevation_b_to_a_wraparound_degrees < 45))
        K_factor_b_to_a_dB[mask] = 7.46 + 8.01 * aux_rnd[mask]

        mask = np.logical_and(los_b_to_a == True, np.logical_and(elevation_b_to_a_wraparound_degrees >= 45, elevation_b_to_a_wraparound_degrees < 55))
        K_factor_b_to_a_dB[mask] = 6.52 + 8.27 * aux_rnd[mask]

        mask = np.logical_and(los_b_to_a == True, np.logical_and(elevation_b_to_a_wraparound_degrees >= 55, elevation_b_to_a_wraparound_degrees < 65))
        K_factor_b_to_a_dB[mask] = 5.47 + 7.26 * aux_rnd[mask]

        mask = np.logical_and(los_b_to_a == True, np.logical_and(elevation_b_to_a_wraparound_degrees >= 65, elevation_b_to_a_wraparound_degrees < 75))
        K_factor_b_to_a_dB[mask] = 4.54 + 5.53 * aux_rnd[mask]

        mask = np.logical_and(los_b_to_a == True, np.logical_and(elevation_b_to_a_wraparound_degrees >= 75, elevation_b_to_a_wraparound_degrees < 85))
        K_factor_b_to_a_dB[mask] = 4.03 + 4.49 * aux_rnd[mask]

        mask = np.logical_and(los_b_to_a == True, np.logical_and(elevation_b_to_a_wraparound_degrees >= 85, elevation_b_to_a_wraparound_degrees < 90))
        K_factor_b_to_a_dB[mask] = 3.68 + 3.14 * aux_rnd[mask]

        return K_factor_b_to_a_dB

    def k_factor_3gpptr38_811_dense_urban_ntn(
            self,
            zeniths_b_to_a_wraparound_degrees: npt.NDArray,
            los_b_to_a: npt.NDArray
    ) -> npt.NDArray:

        # Initialize path loss results 
        K_factor_b_to_a_dB = np.zeros((np.size(los_b_to_a, 0), np.size(los_b_to_a, 1)))

        # Elevation angle 
        elevation_b_to_a_wraparound_degrees = zeniths_b_to_a_wraparound_degrees - 90

        # Calculate normal random variable to aid lognormal distribution
        aux_rnd = self.rng.randn(np.size(los_b_to_a, 0), np.size(los_b_to_a, 1))

        # Calculate mask for LoS and elevation range 
        # Calcualte K-factor for LoS   

        mask = np.logical_and(los_b_to_a == True, np.logical_and(elevation_b_to_a_wraparound_degrees >= 0, elevation_b_to_a_wraparound_degrees < 15))
        K_factor_b_to_a_dB[mask] = 4.4 + 3.3 * aux_rnd[mask]

        mask = np.logical_and(los_b_to_a == True, np.logical_and(elevation_b_to_a_wraparound_degrees >= 15, elevation_b_to_a_wraparound_degrees < 25))
        K_factor_b_to_a_dB[mask] = 9.4 + 6.6 * aux_rnd[mask]

        mask = np.logical_and(los_b_to_a == True, np.logical_and(elevation_b_to_a_wraparound_degrees >= 25, elevation_b_to_a_wraparound_degrees < 35))
        K_factor_b_to_a_dB[mask] = 9.3 + 6.1 * aux_rnd[mask]

        mask = np.logical_and(los_b_to_a == True, np.logical_and(elevation_b_to_a_wraparound_degrees >= 35, elevation_b_to_a_wraparound_degrees < 45))
        K_factor_b_to_a_dB[mask] = 7.9 + 4.0 * aux_rnd[mask]

        mask = np.logical_and(los_b_to_a == True, np.logical_and(elevation_b_to_a_wraparound_degrees >= 45, elevation_b_to_a_wraparound_degrees < 55))
        K_factor_b_to_a_dB[mask] = 7.4 + 3.0 * aux_rnd[mask]

        mask = np.logical_and(los_b_to_a == True, np.logical_and(elevation_b_to_a_wraparound_degrees >= 55, elevation_b_to_a_wraparound_degrees < 65))
        K_factor_b_to_a_dB[mask] = 7.0 + 2.6 * aux_rnd[mask]

        mask = np.logical_and(los_b_to_a == True, np.logical_and(elevation_b_to_a_wraparound_degrees >= 65, elevation_b_to_a_wraparound_degrees < 75))
        K_factor_b_to_a_dB[mask] = 6.9 + 2.2 * aux_rnd[mask]

        mask = np.logical_and(los_b_to_a == True, np.logical_and(elevation_b_to_a_wraparound_degrees >= 75, elevation_b_to_a_wraparound_degrees < 85))
        K_factor_b_to_a_dB[mask] = 6.5 + 2.1 * aux_rnd[mask]

        mask = np.logical_and(los_b_to_a == True, np.logical_and(elevation_b_to_a_wraparound_degrees >= 85, elevation_b_to_a_wraparound_degrees < 90))
        K_factor_b_to_a_dB[mask] = 6.8 + 1.9 * aux_rnd[mask]

        return K_factor_b_to_a_dB
