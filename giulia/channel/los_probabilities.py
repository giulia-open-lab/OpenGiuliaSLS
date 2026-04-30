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
from typing import Union, Optional, List

import numpy as np
import numpy.typing as npt
import pandas as pd

from giulia.bs.bs_deployments import Network
from giulia.config.sim_config import Simulation_Config
from giulia.event_driven.snapshot_control import Snapshot_control
from giulia.fs import results_file
from giulia.playground import Distance_Angles
from giulia.tools.tools import log_calculations_time
from giulia.ue.ue_deployments import UE_Deployment
from giulia.outputs.saveable import Saveable

class LOSProbability(Saveable):
    # Placeholder to store probability of line of sight results
    p_los_b_to_a: npt.NDArray
    los_b_to_a: npt.NDArray
    df_los_b_to_a: pd.DataFrame

    bs_propagation_models: Union[npt.NDArray, pd.Series]
    bs_fast_channel_models: npt.NDArray
    site_id: npt.NDArray

    positions_b_m: npt.NDArray[np.single]
    indoor: bool

    zeniths_b_to_a_wraparound_degrees: npt.NDArray
    distance_b_to_a_2d_m: npt.NDArray
    d_2d_in_m: np.float64

    rng: np.random.RandomState

    def __init__(
            self,
            simulation_config_obj: Simulation_Config,
            network_deployment_obj: Network,
            ue_deployment_obj: UE_Deployment,
            distance_angles_ue_to_cell_obj: Distance_Angles
    ):
        
        super().__init__()

        ##### Input storage 
        ########################  
        self.simulation_config_obj = simulation_config_obj
        self.network_deployment_obj = network_deployment_obj
        self.ue_deployment_obj = ue_deployment_obj
        self.distance_angles_ue_to_cell_obj = distance_angles_ue_to_cell_obj


    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["los_b_to_a"]


    def process(self, rescheduling_us=-1):

        ##### Process inputs
        ######################## 

        # Random numbers
        self.rng = np.random.RandomState(self.simulation_config_obj.random_seed + 0)

        # Network
        if "BS_propagation_model" in self.network_deployment_obj.df_ep:
            self.bs_propagation_models = self.network_deployment_obj.df_ep["BS_propagation_model"].to_numpy()
            self.bs_fast_channel_models = self.network_deployment_obj.df_ep["BS_fast_channel_model"].to_numpy()
            self.site_id = self.network_deployment_obj.df_ep["site_ID"].to_numpy()
        else:
            # If we are dealing with the UE to UE case, then there is no BS_propagation_model field and we drive the LoS prob by the model 3GPPTR38_901_UMi
            self.bs_propagation_models = pd.Series(np.repeat("3GPPTR38_901_UMi",repeats=np.size(self.network_deployment_obj.df_ep, 0),axis=0))

        # Users deployment
        self.positions_b_m = (self.ue_deployment_obj.df_ep[["position_x_m", "position_y_m", "position_z_m"]].to_numpy(dtype=np.single))
        self.indoor = self.ue_deployment_obj.df_ep["indoor"].to_numpy(dtype=bool)

        # Channel characteristics
        self.zeniths_b_to_a_wraparound_degrees = self.distance_angles_ue_to_cell_obj.zeniths_b_to_a_wraparound_degrees
        self.distance_b_to_a_2d_m = self.distance_angles_ue_to_cell_obj.distance_b_to_a_2d_wraparound_m
        self.d_2d_in_m = self.distance_angles_ue_to_cell_obj.d_2D_in_m

        ##### Process outputs
        ########################   
        self.p_los_b_to_a = np.ones((np.size(self.distance_b_to_a_2d_m, 0), np.size(self.distance_b_to_a_2d_m, 1)),dtype=np.single)

        ##### Start timer
        ######################## 
        t_start = time.perf_counter()

        ##### Switch
        ########################            

        # Find the set of unique LoS probability models to process them independently
        bs_propagation_models_set = set(self.bs_propagation_models)
        bs_fast_channel_models_set = set(self.bs_fast_channel_models)

        # Process each LoS probability model independently
        for models in itertools.product(bs_propagation_models_set, bs_fast_channel_models_set):

            # Identify cells with the selected LoS probability model
            bs_propagation_model = models[0]
            bs_fast_channel_model = models[1]
            mask = np.bitwise_and(
                bs_propagation_model == self.bs_propagation_models,
                bs_fast_channel_model == self.bs_fast_channel_models
            )

            # To maintain correlation acorss sites: Get unique elements, indices of the first occurrences, and inverse indices 
            unique_elements, first_indices, inverse_indices = np.unique(self.site_id[mask], return_index=True, return_inverse=True)

            # Get 2D distances of the identified cells
            zeniths_b_to_a_wraparound_degrees: npt.NDArray = self.zeniths_b_to_a_wraparound_degrees[:, first_indices]

            distance_b_to_a_2d_m: npt.NDArray = self.distance_b_to_a_2d_m[:, first_indices]
            d_2d_in_m: npt.NDArray = self.d_2d_in_m[:, first_indices]

            # Calculate LoS probability
            los_probabilities_b_to_a: Optional[npt.NDArray] = None

            if bs_propagation_model == "3GPPTR38_901_UMa" and bs_fast_channel_model != "3GPPTR38_901_UMa":
                los_probabilities_b_to_a = self.__los_prob_3gpptr38_901_uma(
                    self.simulation_config_obj.debug_no_randomness,
                    distance_b_to_a_2d_m,
                    d_2d_in_m,
                    self.positions_b_m[:, 2]
                )

            elif bs_propagation_model == "3GPPTR38_901_UMi" and bs_fast_channel_model != "3GPPTR38_901_UMi":
                los_probabilities_b_to_a = self.__los_prob_3gpptr38_901_umi(
                    distance_b_to_a_2d_m,
                    d_2d_in_m
                )

            elif bs_propagation_model == "ITU_R_M2135_UMa":
                los_probabilities_b_to_a = self.__los_prob_itu_r_m2135_uma(distance_b_to_a_2d_m)

            elif bs_propagation_model == "ITU_R_M2135_UMi":
                los_probabilities_b_to_a = self.__los_prob_itu_r_m2135_umi(distance_b_to_a_2d_m, self.indoor)

            elif bs_propagation_model == "3GPPTR36_814_Case_1":
                los_probabilities_b_to_a = self.__los_prob_3gpptr36_814_case_1(distance_b_to_a_2d_m)

            elif bs_propagation_model == "3GPPTR36_777_UMa_AV":
                los_probabilities_b_to_a = self.__los_prob_3gpptr36_777_uma_av(distance_b_to_a_2d_m, self.positions_b_m[:, 2])

            elif bs_propagation_model == "3GPPTR36_777_UMi_AV":
                los_probabilities_b_to_a = self.__los_prob_3gpptr36_777_umi_av(distance_b_to_a_2d_m, self.positions_b_m[:, 2])

            elif bs_propagation_model == "3GPPTR38_811_Urban_NTN":
                los_probabilities_b_to_a = self.__los_prob_3gpptr38_811_urban_ntn(zeniths_b_to_a_wraparound_degrees)

            elif bs_propagation_model == "3GPPTR38_811_Dense_Urban_NTN":
                los_probabilities_b_to_a = self.__los_prob_3gpptr38_811_dense_urban_ntn(zeniths_b_to_a_wraparound_degrees)

            # Map back to all cells of the site using `np.take` to replicate the values according to `inverse_indices`
            if los_probabilities_b_to_a is not None:
                self.p_los_b_to_a[:, mask] = np.take(los_probabilities_b_to_a, inverse_indices, axis=1)

        # Binary selecting whether a link is LoS or not
        self.los_b_to_a = self.__assign_los_based_on_plos(self.rng, self.p_los_b_to_a)

        # Store in data frames the results as it may be useful to post process
        self.df_los_b_to_a = pd.DataFrame(self.los_b_to_a, columns=self.network_deployment_obj.df_ep["name"], index=self.ue_deployment_obj.df_ep["name"])

        ##### Save to plot
        ########################
        snapshot_control = Snapshot_control.get_instance()
        if self.simulation_config_obj.plot == 1 and snapshot_control.num_snapshots == 0:
            file_name = results_file(self.simulation_config_obj.project_name, 'to_plot_LoS')
            np.savez(file_name, LoS_b_to_a=self.los_b_to_a * 1)

        ##### End
        log_calculations_time('LoS probability', t_start)

        return rescheduling_us


    @staticmethod
    def __assign_los_based_on_plos(rng: np.random.RandomState, p_los_b_to_a) -> npt.NDArray:

        # Find the number of UEs
        size_of_b = np.size(p_los_b_to_a, 0)

        # Find 
        #   - sorted unique columns
        #   - the indices of the first occurrences of the unique values in the original array
        #   - the indices of the unique array that reconstruct the input array 
        unique, indices, inverse = np.unique(p_los_b_to_a, axis=1, return_index=True, return_inverse=True)

        # Find the number of involved cell sites
        number_of_unique_elements_in_a = np.size(unique, 1)

        # Create number of ues x number of involved cell sites LoS matrix
        rand = rng.rand(size_of_b, number_of_unique_elements_in_a)
        los_b_to_a = rand < p_los_b_to_a[:, indices]

        # Allocate the respective LoS to every cell - Cells of a cell site have the same LoS
        return los_b_to_a[:, inverse]


    @staticmethod
    def __los_prob_3gpptr38_901_uma(debug_no_randomness: bool, d_2d: npt.NDArray, d_2d_in: npt.NDArray, h_ut: npt.NDArray) -> npt.NDArray:

        # Calculate d_2d_out
        d_2d_out = d_2d - d_2d_in

        # Initialize P_LoS results 
        p_los_b_to_a = np.ones((np.size(d_2d_out, 0), np.size(d_2d_out, 1)))

        if not debug_no_randomness:
            # Replicate the vector of UE heights in a column manner to facilitate next operation
            h_ut_mat = h_ut[:, np.newaxis] * np.ones(np.size(d_2d_out, 1))

            # Calculate values for distances larger than 18m
            mask = d_2d_out > 18
            p_los_b_to_a[mask] = 18 / d_2d_out[mask] + np.exp(-1 * d_2d_out[mask] / 63) * (1 - 18 / d_2d_out[mask])

            # Correct values for distances larger than 18m and UE heights larger than 13m
            mask = np.logical_and(d_2d_out > 18, h_ut_mat > 13)
            p_los_b_to_a[mask] *= \
                1 + np.power((h_ut_mat[mask] - 13) / 10, 1.5) * 5 / 4 * np.power(d_2d_out[mask] / 100, 3) * np.exp(-1 * d_2d_out[mask] / 150)

        return np.round(p_los_b_to_a, 5)


    @staticmethod
    def __los_prob_3gpptr38_901_umi(d_2d: npt.NDArray, d_2d_in: npt.NDArray) -> npt.NDArray:

        # Calculate d_2d_out
        d_2d_out = d_2d - d_2d_in

        # Initialize P_LoS results 
        p_los_b_to_a = np.ones((np.size(d_2d_out, 0), np.size(d_2d_out, 1)))

        # Calculate values for distances larger than 18m
        mask = d_2d_out > 18
        p_los_b_to_a[mask] = 18 / d_2d_out[mask] + np.exp(-1 * d_2d_out[mask] / 36) * (1 - 18 / d_2d_out[mask])

        return np.round(p_los_b_to_a, 5)


    @staticmethod
    def __los_prob_itu_r_m2135_uma(d_out: npt.NDArray) -> npt.NDArray:

        # Initialize P_LoS results 
        p_los_b_to_a = np.ones((np.size(d_out, 0), np.size(d_out, 1)))

        # Calculate values for distances larger than 18m
        np.seterr(divide='ignore')
        mask = 18 / d_out < 1
        p_los_b_to_a[mask] = 18 / d_out[mask]

        p_los_b_to_a = p_los_b_to_a * (1 - np.exp(-1 * d_out / 63)) + np.exp(-1 * d_out / 63)

        return np.round(p_los_b_to_a, 5)


    @staticmethod
    def __los_prob_itu_r_m2135_umi(d_out: npt.NDArray, indoor) -> npt.NDArray:

        # Initialize P_LoS results 
        p_los_b_to_a = np.ones((np.size(d_out, 0), np.size(d_out, 1)))

        # Calculate values for distances larger than 18m

        np.seterr(divide='ignore')
        mask = 18 / d_out < 1
        p_los_b_to_a[mask] = 18 / d_out[mask]

        p_los_b_to_a = p_los_b_to_a * (1 - np.exp(-1 * d_out / 36)) + np.exp(-1 * d_out / 36)

        return np.round(p_los_b_to_a, 5)


    @staticmethod
    def __los_prob_3gpptr36_814_case_1(d_2d_out: npt.NDArray) -> npt.NDArray:

        # Initialize P_LoS results 
        p_los_b_to_a = np.zeros((np.size(d_2d_out, 0), np.size(d_2d_out, 1)))

        return np.round(p_los_b_to_a, 5)


    @staticmethod
    def __los_prob_3gpptr36_777_uma_av(d_2d_out: npt.NDArray, h_ut: npt.NDArray) -> npt.NDArray:

        # Initialize P_LoS results 
        p_los_b_to_a = np.ones((np.size(d_2d_out, 0), np.size(d_2d_out, 1)))

        # Replicate the vector of UE heights in a column manner to facilitate next operation
        h_ut_mat = h_ut[:, np.newaxis] * np.ones(np.size(d_2d_out, 1))

        ## GROUND UEs
        # Calculate values for distances larger than 18m
        mask = np.logical_and(h_ut_mat <= 22.5, d_2d_out > 18)
        p_los_b_to_a[mask] = 18 / d_2d_out[mask] + np.exp(-1 * d_2d_out[mask] / 63) * (1 - 18 / d_2d_out[mask])

        # Correct values for distances larger than 18m and UE heights larger than 13m
        mask = np.logical_and(h_ut_mat <= 22.5, np.logical_and(d_2d_out > 18, h_ut_mat > 13))
        p_los_b_to_a[mask] *= \
            1 + np.power((h_ut_mat[mask] - 13) / 10, 1.5) * 5 / 4 * np.power(d_2d_out[mask] / 100, 3) * np.exp(-1 * d_2d_out[mask] / 150)

        ## AERIAL UEs from 22.5 to 100m
        d1 = np.ones((np.size(d_2d_out, 0), np.size(d_2d_out, 1))) * (460 * np.log10(h_ut_mat) - 700)
        d1[d1 < 18] = 18

        # Calculate values for distances smaller than d1 m 
        # Note that for distances smaller than d1, p_los = 1 from initialization, so no dedicated mask is needed

        # Calculate values for distances larger than d1 m
        mask = np.logical_and(h_ut_mat > 22.5, np.logical_and(h_ut_mat <= 100, d_2d_out > d1))
        p1 = 4300 * np.log10(h_ut_mat[mask]) - 3800
        p_los_b_to_a[mask] = d1[mask] / d_2d_out[mask] + np.exp(-1 * d_2d_out[mask] / p1) * (1 - d1[mask] / d_2d_out[mask])

        ## AERIAL UEs from 100 to 300m 
        # For h_ut > 100 m, p_los is defined as 1 in this model and is already preserved by initialization.

        return np.round(p_los_b_to_a, 5)


    @staticmethod
    def __los_prob_3gpptr36_777_umi_av(d_2d_out: npt.NDArray, h_ut: npt.NDArray) -> npt.NDArray:

        # Initialize P_LoS results 
        p_los_b_to_a = np.ones((np.size(d_2d_out, 0), np.size(d_2d_out, 1)))

        # Replicate the vector of UE heights in a column manner to facilitate next operation
        h_ut_mat = h_ut[:, np.newaxis] * np.ones(np.size(d_2d_out, 1))

        ## GROUND UEs
        # Calculate values for distances larger than 18m
        mask = np.logical_and(h_ut_mat <= 22.5, d_2d_out > 18)
        p_los_b_to_a[mask] = 18 / d_2d_out[mask] + np.exp(-1 * d_2d_out[mask] / 36) * (1 - 18 / d_2d_out[mask])

        ## AERIAL UEs from 22.5 to 100m
        d1 = np.ones((np.size(d_2d_out, 0), np.size(d_2d_out, 1))) * (294.05 * np.log10(h_ut_mat) - 432.94)
        d1[d1 < 18] = 18

        # Calculate values for distances larger than d1 m
        # Note that for distances smaller than d1, p_los = 1 from initialization, so no dedicated mask is needed
        mask = np.logical_and(h_ut_mat > 22.5, np.logical_and(h_ut_mat <= 300, d_2d_out > d1))
        p1 = 233.98 * np.log10(h_ut_mat[mask]) - 0.95
        p_los_b_to_a[mask] = d1[mask] / d_2d_out[mask] + np.exp(-1 * d_2d_out[mask] / p1) * (1 - d1[mask] / d_2d_out[mask])

        return np.round(p_los_b_to_a, 5)


    @staticmethod
    def __los_prob_3gpptr38_811_urban_ntn(zeniths_b_to_a_wraparound_degrees: npt.NDArray) -> npt.NDArray:

        # Initialize P_LoS results 
        p_los_b_to_a = np.ones((np.size(zeniths_b_to_a_wraparound_degrees, 0), np.size(zeniths_b_to_a_wraparound_degrees, 1)))

        # Elevation angle 
        elevation_b_to_a_wraparound_degrees = zeniths_b_to_a_wraparound_degrees - 90

        # Calculate mask for elevation range 
        # Calculate LoS probability
        mask = np.logical_and(elevation_b_to_a_wraparound_degrees >= 0, elevation_b_to_a_wraparound_degrees < 15)
        p_los_b_to_a[mask] = 0.246

        mask = np.logical_and(elevation_b_to_a_wraparound_degrees >= 15, elevation_b_to_a_wraparound_degrees < 25)
        p_los_b_to_a[mask] = 0.386

        mask = np.logical_and(elevation_b_to_a_wraparound_degrees >= 25, elevation_b_to_a_wraparound_degrees < 35)
        p_los_b_to_a[mask] = 0.493

        mask = np.logical_and(elevation_b_to_a_wraparound_degrees >= 35, elevation_b_to_a_wraparound_degrees < 45)
        p_los_b_to_a[mask] = 0.613

        mask = np.logical_and(elevation_b_to_a_wraparound_degrees >= 45, elevation_b_to_a_wraparound_degrees < 55)
        p_los_b_to_a[mask] = 0.726

        mask = np.logical_and(elevation_b_to_a_wraparound_degrees >= 55, elevation_b_to_a_wraparound_degrees < 65)
        p_los_b_to_a[mask] = 0.805

        mask = np.logical_and(elevation_b_to_a_wraparound_degrees >= 65, elevation_b_to_a_wraparound_degrees < 75)
        p_los_b_to_a[mask] = 0.919

        mask = np.logical_and(elevation_b_to_a_wraparound_degrees >= 75, elevation_b_to_a_wraparound_degrees < 85)
        p_los_b_to_a[mask] = 0.968

        mask = np.logical_and(elevation_b_to_a_wraparound_degrees >= 85, elevation_b_to_a_wraparound_degrees < 90)
        p_los_b_to_a[mask] = 0.992

        return np.round(p_los_b_to_a, 5)


    @staticmethod
    def __los_prob_3gpptr38_811_dense_urban_ntn(zeniths_b_to_a_wraparound_degrees: npt.NDArray) -> npt.NDArray:

        # Initialize P_LoS results 
        p_los_b_to_a = np.ones((np.size(zeniths_b_to_a_wraparound_degrees, 0), np.size(zeniths_b_to_a_wraparound_degrees, 1)))

        # Elevation angle 
        elevation_b_to_a_wraparound_degrees = zeniths_b_to_a_wraparound_degrees - 90

        # Calculate mask for elevation range 
        # Calculate LoS probability

        mask = np.logical_and(elevation_b_to_a_wraparound_degrees >= 0, elevation_b_to_a_wraparound_degrees < 15)
        p_los_b_to_a[mask] = 0.282

        mask = np.logical_and(elevation_b_to_a_wraparound_degrees >= 15, elevation_b_to_a_wraparound_degrees < 25)
        p_los_b_to_a[mask] = 0.331

        mask = np.logical_and(elevation_b_to_a_wraparound_degrees >= 25, elevation_b_to_a_wraparound_degrees < 35)
        p_los_b_to_a[mask] = 0.398

        mask = np.logical_and(elevation_b_to_a_wraparound_degrees >= 35, elevation_b_to_a_wraparound_degrees < 45)
        p_los_b_to_a[mask] = 0.468

        mask = np.logical_and(elevation_b_to_a_wraparound_degrees >= 45, elevation_b_to_a_wraparound_degrees < 55)
        p_los_b_to_a[mask] = 0.537

        mask = np.logical_and(elevation_b_to_a_wraparound_degrees >= 55, elevation_b_to_a_wraparound_degrees < 65)
        p_los_b_to_a[mask] = 0.612

        mask = np.logical_and(elevation_b_to_a_wraparound_degrees >= 65, elevation_b_to_a_wraparound_degrees < 75)
        p_los_b_to_a[mask] = 0.738

        mask = np.logical_and(elevation_b_to_a_wraparound_degrees >= 75, elevation_b_to_a_wraparound_degrees < 85)
        p_los_b_to_a[mask] = 0.820

        mask = np.logical_and(elevation_b_to_a_wraparound_degrees >= 85, elevation_b_to_a_wraparound_degrees < 90)
        p_los_b_to_a[mask] = 0.981

        return np.round(p_los_b_to_a, 5)