# -*- coding: utf-8 -*-
"""
@author: Matteo Bernabe
matteo.bernabe@ieee.org 
matteo.bernabe@iteam.upv.es

DEDALO_EE: energy-efficiency realization of Dedalo, built on the base class in dedalo/dedalo.py.
"""
# Configure Dedalo into path
import importlib.util
import os
import sys
examples_dir = os.path.dirname(__file__)
root_dir = os.path.join(examples_dir, '..')
sys.path.insert(1, root_dir)

import numpy as np
import pandas as pd
from scipy.stats import norm, wasserstein_distance

import matplotlib.pyplot as plt

from dedalo.dedalo import Dedalo, GiuliaLike
from giulia.presets._g_pairing_preset_new import Giulia_Pairing
from giulia.config.sim_config import Simulation_Config
from giulia.playground.sites import Site, Site_siteDeployment, Site_uePlaygroundDeployment
from giulia.playground.hotspots import Hotspot
from giulia.bs.bs_deployments import Network
from giulia.ue.ue_deployments import UE_Deployment
from giulia.bs.base_stations import BaseStations
from giulia.kpis.power_consumptions import Power_Consumption
from typing import Any, cast

from giulia.logger import info, get_log_level, parse_log_level, set_log_level
from giulia.tools.tools import free_memory


# Configure Giulia with an energy-efficiency focus.
# NOTE: this realization was written for the earlier energy-efficiency studies (HAPS Hypercell
# carrier shutdown) and is kept as it was used there. It may need updating before it is reused:
#   - _compute_pairing_coverage_cell_4G_only() runs Giulia_Pairing, which disables Saveable, and
#     never enables it again: KPI objects built afterwards (e.g. by the next configure()) are not
#     registered, so no KPI files are written. PairingCell restores the flag
#     (giulia/rrc/cell_pairing.py); this class does not.
#   - set_pairing_df_ep_as_per_terrestrial_haps_hierarchical() extends the paired_cell lists of
#     df_ep_input in place, so the input table is modified too (Dedalo_UnRL copies them first).
class Dedalo_EE(Dedalo):
    """Energy-efficiency oriented Dedalo configurator for Giulia runs."""

    # Prepare the Dedalo EE instance and prime defaults.
    def __init__(self, giulia: GiuliaLike, **kwargs) -> None:
        """Prepare Dedalo for a Giulia simulation.

        Parameters
        ----------
        giulia: Giulia
            Simulation instance to be customised.
        **kwargs:
            Optional overrides passed to the UE initialisation helper.
        """
        super().__init__(giulia)

        # Start hour offset for the daily profile. Defaults to 0.
        self.start_hour = 0

        # Build the UE population and density profiles.
        self._init_number_of_ue(total_number_of_ue=6412) #  6412=(9*114)/0.16  #10000 # 12000

        # Set the Giulia radio unit ID different for each sector.
        self._set_giulia_radio_unit_id()

        # Placeholder updated after querying Giulia runtime state.
        self.giulia_radio_status_dict: dict = {}

        # Initialize episode and step tracking for plotting
        self.episode_index: int | None = None
        self.step_index: int | None = None

        # Compute 4G only pairing coverage dict
        # For custom multi-cell wakeup logic
        self.pairing_4G_only_dict: dict = self._compute_pairing_coverage_cell_4G_only()

        return


    def _set_giulia_radio_unit_id(self) -> None:
        """Assign unique radio unit IDs to each base station in Giulia."""
        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise TypeError("The network_deployment_obj should be a Network object, not a list.")
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("The network_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")
        if not isinstance(self.giulia_obj.power_consumption_obj, Power_Consumption):
            raise TypeError("The power_consumption_obj should be a PowerConsumption object, not a list.")
        
        # Get Giulia DataFrame Engineering Parameters
        df_ep = self.giulia_obj.network_deployment_obj.df_ep

        # Assign unique radio unit IDs based on the ID.
        self.set_giulia_network_df_ep(name_cell_list=df_ep['name'].tolist(), 
                                      name_attribute="radio_unit_ID", 
                                      value_list=df_ep['ID'].tolist())

        df_ep = self.giulia_obj.network_deployment_obj.df_ep

        ### RE-Run Power Consumption Process in Giulia for Updating.
        self.giulia_obj.power_consumption_obj.process()
        
        return
    

    def get_giulia_N_UEsite_cell(self) -> int:
        """
        Retrieve the reference number of UE Site cells from the Giulia UE deployment object.
        """
        # Basic Checks
        if not isinstance(self.giulia_obj.ue_playground_deployment_obj, Site):
            raise TypeError("The ue_playground_deployment_obj should be a Site object, not a list.")
        if isinstance(self.giulia_obj.ue_playground_deployment_obj, list):
            raise TypeError("The ref_number_of_cells should be an integer, not a list.")
        if self.giulia_obj.ue_playground_deployment_obj.ref_number_of_cells is None:
            raise ValueError("ref_number_of_cells is None. Ensure the UE deployment is properly configured.")

        return self.giulia_obj.ue_playground_deployment_obj.ref_number_of_cells


    # Build UE population information and cache initial Giulia densities.
    def _init_number_of_ue(self, **kwargs) -> None:
        """Initialise the UE population and per-hour density profile.

        Parameters
        ----------
        **kwargs:
            ``total_number_of_ue`` (int): desired daily UE population. Defaults to 2280.

        Raises
        ------
        TypeError
            If the Giulia deployment object is not a Network instance.
        ValueError
            If the deployment object does not expose a pandas DataFrame.
        """
        # Retrieve the total number of UEs to deploy. Defaults to 2280.
        self.total_number_of_ue = int(kwargs.get("total_number_of_ue", 2280))

        # Normalised UE proportion for each day hour.
        proportion_ue = np.asarray([0.13,  # 0h
                                    0.10,  # 1h
                                    0.07,  # 2h
                                    0.04,  # 3h
                                    0.03,  # 4h
                                    0.02,  # 5h
                                    0.02,  # 6h
                                    0.03,  # 7h
                                    0.04,  # 8h
                                    0.06,  # 9h
                                    0.08,  # 10h
                                    0.09,  # 11h
                                    0.10,  # 12h
                                    0.10,  # 13h
                                    0.11,  # 14h
                                    0.12,  # 15h
                                    0.12,  # 16h
                                    0.12,  # 17h
                                    0.13,  # 18h
                                    0.14,  # 19h
                                    0.15,  # 20h
                                    0.16,  # 21h
                                    0.16,  # 22h
                                    0.15,  # 23h
                                    0.13   # 24h
                                ])
        # Build a dictionary for easy hour-based access.
        self.proportion_ue_dict: dict = {f"hour_{h}": value for h, value in enumerate(proportion_ue)}

        # Get number of UE site cells from Giulia
        self.N_UEsite_cell = self.get_giulia_N_UEsite_cell()

        ue_per_UEsite_cell_func = lambda value, total_number_of_ue, N_UEsite_cell: int(round(value*(total_number_of_ue / N_UEsite_cell)))

        # Compute the uniform UE density per BS for each hour.
        self.ue_per_UEsite_cell_per_hour_dict = {hour: np.full(self.N_UEsite_cell, 
                                                                    ue_per_UEsite_cell_func(value, self.total_number_of_ue, self.N_UEsite_cell), 
                                                                    dtype=int) for hour, value in self.proportion_ue_dict.items()}

        return

    # Expose hourly UE density values for other components.
    def get_density_uniform_ue_per_specific_hour(self, hour: int) -> int:
        #### Deprecated ####
        raise DeprecationWarning("This method is deprecated. Use get_ue_per_UEsite_cell_per_specific_hour instead.")
        raise ImportError("This method is deprecated. Use get_ue_per_UEsite_cell_per_specific_hour instead.")
    
        """Return the uniform UE density for a given hour of the day.

        Parameters
        ----------
        hour: int
            Hour index to sample. Values wrap around the 24-hour profile.

        Returns
        -------
        int
            Number of UEs per base station for the requested hour.
        """
        # Wrap around the 24-hour profile.
        hour = hour % 24

        return self.density_uniform_ue_per_hour[f"hour_{hour}"]
    

    def get_ue_per_UEsite_cell_per_specific_hour(self, hour: int) -> np.ndarray:
        """Return the uniform UE density per UE site cell for a given hour of the day.

        Parameters
        ----------
        hour: int
            Hour index to sample. Values wrap around the 24-hour profile.

        Returns
        -------
        np.ndarray
            Array of UEs per UE site cell for the requested hour.
        """
        # Wrap around the 24-hour profile.
        hour = hour % 24

        return self.ue_per_UEsite_cell_per_hour_dict[f"hour_{hour}"]

    # Push updated UE densities into the Giulia configuration.
    def set_in_Giulia_ue_density_per_bs(self, value: int) -> None:
        #### Deprecated ####
        raise DeprecationWarning("This method is deprecated. Use get_ue_per_UEsite_cell_per_specific_hour instead.")
        raise ImportError("This method is deprecated. Use get_ue_per_UEsite_cell_per_specific_hour instead.")
        """Persist the requested UE density into the Giulia configuration.

        Parameters
        ----------
        value: int
            UE density per base station to inject into Giulia.

        Raises
        ------
        TypeError
            If Giulia was initialised with an unexpected simulation object type.
        """
        if isinstance(self.giulia_obj.simulation_config_obj, list):
                raise TypeError("The simulation_config_obj should be a Network object, not a list. giulia has to be intiialized")
        
        
        dist =  self.giulia_obj.simulation_config_obj.ue_distribution
        if dist.split("_")[0] == "uniform":
            if isinstance(self.giulia_obj.simulation_config_obj.ue_deployDensity_info, dict):  # Case for 38901 and colocated/nocolocated"
                self.giulia_obj.simulation_config_obj.ue_deployDensity_info["construct_ue_deployment_3GPPTR38_901_UMa_large_scale_calibration"].ue_density_perCell = value
                self.giulia_obj.simulation_config_obj.ue_deployDensity_info["construct_ue_deployment_polygon"].ue_density_perCell = value

            else:
                self.giulia_obj.simulation_config_obj.ue_deployDensity_info.ue_density_perCell = value


        elif dist.split("_")[0] == "inhomogeneous":
            if isinstance(self.giulia_obj.simulation_config_obj.ue_deployDensity_info, dict):  # Case for 38901 and colocated/nocolocated"
                self.giulia_obj.simulation_config_obj.ue_deployDensity_info["construct_ue_deployment_3GPPTR38_901_UMa_large_scale_calibration"].ue_density_perCell = [int(value*4), int(value*1.5), int(value*0.5)]
                self.giulia_obj.simulation_config_obj.ue_deployDensity_info["construct_ue_deployment_polygon"].ue_density_perCell = value

            else:
                self.giulia_obj.simulation_config_obj.ue_deployDensity_info.ue_density_perCell = [int(value*4), int(value*1.5), int(value*0.5)]

        else:
            raise ValueError("UE Distribution type not recognized.")

        return
    

    def set_in_Giulia_ue_per_UEsite_cell(self, value_per_UEsite_cell: np.ndarray) -> None:
        if isinstance(self.giulia_obj.simulation_config_obj, list):
                raise TypeError("The simulation_config_obj should be a Network object, not a list. giulia has to be intiialized")
        
        if isinstance(self.giulia_obj.simulation_config_obj.number_of_ues_per_UEsite_cell_mfl_dict, dict):  # Case for 38901 and colocated/nocolocated"
            self.giulia_obj.simulation_config_obj.number_of_ues_per_UEsite_cell_mfl_dict["construct_ue_deployment_3GPPTR38_901_UMa_large_scale_calibration"] = value_per_UEsite_cell // 2 
            self.giulia_obj.simulation_config_obj.number_of_ues_per_UEsite_cell_mfl_dict["construct_ue_deployment_polygon"] = value_per_UEsite_cell // 2
        else:
            self.giulia_obj.simulation_config_obj.number_of_ues_per_UEsite_cell = value_per_UEsite_cell
        



    # Sync Dedalo's radio status view with Giulia runtime data.
    def set_dedalo_radio_status_from_giulia_runTime(self) -> None:
        """Capture the most recent radio unit status from Giulia.

        The method supports two scenarios: before Giulia runtime (read from
        deployment data) or after runtime (read from BaseStation objects).

        Raises
        ------
        TypeError
            If Giulia exposes unexpected container types for deployment objects.
        ValueError
            If the deployment DataFrame is missing or invalid.
        """
        # Very first ever iteration before Giulia confif and runtime
        if isinstance(self.giulia_obj.base_stations_obj, list):

            if not isinstance(self.giulia_obj.network_deployment_obj, Network):
                raise TypeError("The network_deployment_obj should be a Network object, not a list.")
            if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
                raise ValueError("The network_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")
            df_ep = self.giulia_obj.network_deployment_obj.df_ep
            self.giulia_radio_status_dict = dict(zip(df_ep["name"], df_ep["radio_unit_initial_status"]))


        # After Giulia first configuration and runtime
        elif isinstance(self.giulia_obj.base_stations_obj, BaseStations):
            # if first episode, reset all to active
            if self.episode_index == 0:
                self.giulia_radio_status_dict =\
                    {bs_obj.cell_name: "active" for bs_obj in self.giulia_obj.base_stations_obj.base_stations}
                
            # else, get from runtime
            else:
                self.giulia_radio_status_dict =\
                    {bs_obj.cell_name: bs_obj.rrc_layer.radio_unit_status for bs_obj in self.giulia_obj.base_stations_obj.base_stations}

        else:
            raise TypeError("Wrong type for giulia_obj.base_stations_obj. It should be a BaseStations object.")
        
        return

    # Write Dedalo's radio status snapshot back into Giulia's deployment frame.
    def set_giulia_df_ep_radio_status(self) -> None:
        """Write cached radio unit status back into Giulia's deployment frame.

        Raises
        ------
        TypeError
            If the Giulia deployment object is not a Network instance.
        ValueError
            If the deployment DataFrame is missing or invalid.
        """
        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise TypeError("The network_deployment_obj should be a Network object, not a list.")
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("The network_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")

        for cell_name, status in self.giulia_radio_status_dict.items():
            self.giulia_obj.network_deployment_obj.df_ep.loc[self.giulia_obj.network_deployment_obj.df_ep['name'] == cell_name, 'radio_unit_initial_status'] = status

        return
    


    def setGiulia_ue_target_rate(self, ue_rate_requirements_Mbps: list) -> None:
        """
        The following method set in giulia UE dataframe engeneer parmaeter df_ep the target rate for each UE.
        """
        self.set_giulia_ue_df_ep(name_ue_list=self.get_ue_names(attribute=None),
                                 name_attribute='ue_target_rate_Mbps',
                                 value_list=ue_rate_requirements_Mbps)
        
        return 
    

    def setGiulia_ue_traffic_generation_model(self, model: str='rate_requirement') -> None:
        """
        The following method set in giulia UE dataframe engeneer parmaeter df_ep the traffic generation model.
        """
        self.set_giulia_ue_df_ep(name_ue_list=self.get_ue_names(attribute=None),
                                 name_attribute='traffic_generation_model',
                                 value_list=model)
        return
    


    def set_pairing_df_ep_as_per_terrestrial_haps_hierarchical(self, df_ep_input: pd.DataFrame) -> None:
        """
        Update Giulia's df_ep pairing information based on df_ep_input.
    
        This function:
        - Adds the HAPS index to paired_cell lists (except for capacity cells).
        - Performs basic type checks.
        - Updates df_ep.paired_cell in the Giulia-associated Network.
        - Cleans up HAPS paired cells so that only coverage cells remain.
        - Sets the "cell_role" attribute to "coverage_capacity" for coverage cells.
        """

        # Basic Checks
        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise TypeError("The network_deployment_obj should be a Network object, not a list.")
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise TypeError(
                "The network_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")

        # take the last index of self.giulia_obj.network_deployment_obj.df_ep,
        # as it correspond to the HAPS
        index_haps_in_df_ep: int = self.giulia_obj.network_deployment_obj.df_ep.shape[0] - 1
    
        # Get cell names to modify in dedalo associated Giulia. Get cell names from df_ep_input
        name_cell_list: list = df_ep_input["name"].tolist()
        # Get paired cell values to set in dedalo associated Giulia. Get paired_cell from df_ep_input
        value_list_paired_cell_list: list = df_ep_input["paired_cell"].tolist()
    
        # Get 4G Cells which would have been coverage cells
        coverage_mask = df_ep_input["cell_role"] == "coverage"
        coverage_cell_name_list: list = df_ep_input.loc[coverage_mask, "name"].tolist()
        coverage_cell_indices: list = df_ep_input.index[coverage_mask].tolist()
    
        # Get 5G Cells which would have been capacity cells
        capacity_mask = df_ep_input["cell_role"] == "capacity"
        capacity_cell_name_list: list = df_ep_input.loc[capacity_mask, "name"].tolist()
        capacity_cell_indices: list = df_ep_input.index[capacity_mask].tolist()
    
        # Add HAPS ID to each paired cell list
        for name, value in zip(name_cell_list, value_list_paired_cell_list):
            if name in capacity_cell_name_list:
                # Capacity cells should be paired only with 4G coverage cells, not the HAPS
                continue
            if value is not None:
                value.extend([index_haps_in_df_ep])
    
        # Update paired cell values in dedalo associated Giulia from df_ep_input
        for name_cell, value in zip(name_cell_list, value_list_paired_cell_list):
            aux = df_ep_input.loc[df_ep_input["name"] == name_cell, "cell_role"]
            assert isinstance(aux, pd.Series)
            cell_role = aux.iloc[0]
            del aux
            # Get index of the cell to update in dedalo associated Giulia
            idx = self.giulia_obj.network_deployment_obj.df_ep.index[self.giulia_obj.network_deployment_obj.df_ep["name"] == name_cell][0]
            # Update
            self.giulia_obj.network_deployment_obj.df_ep.at[idx, "paired_cell"] = value

        del name_cell, value, idx, cell_role
    
        # Eliminate from paired cell of HAPS
        haps_paired_cells: list = self.giulia_obj.network_deployment_obj.df_ep["paired_cell"][index_haps_in_df_ep]
        haps_paired_cells_aux = haps_paired_cells.copy()
    
        for cell in haps_paired_cells_aux:
            if cell in coverage_cell_indices:
                continue
            if cell in capacity_cell_indices:
                haps_paired_cells.remove(cell)
        del haps_paired_cells_aux
    
        # Check that haps_paired_cells contains the same entries as coverage_cell_indices
        for cell in coverage_cell_indices:
            if cell not in haps_paired_cells:
                raise KeyError(
                    "Error in setting paired cells in dedalo associated Giulia from df_ep_input. "
                    "HAPS paired cells do not contain all coverage cells."
                )
    
        # Set new label in dedalo associated Giulia df_ep for 4G cells
        self.set_giulia_network_df_ep(
            name_cell_list=coverage_cell_name_list,
            name_attribute="cell_role",
            value_list="coverage_capacity",
        )
    
        return
    

    def set_thresholds_df_ep_as_per_terrestrial_haps_hierarchical(self, df_ep_input: pd.DataFrame) -> None:
        """
        Sync carrier shutdown and wake-up thresholds in Giulia's df_ep
        based on the provided df_ep_input configuration.
        """

        # Basic Checks
        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise TypeError("The network_deployment_obj should be a Network object, not a list.")
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise TypeError(
                "The network_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")
        
        # Get shutdown and wakeup thresholds
        shutdown_th = df_ep_input.loc[df_ep_input["cell_role"] == "capacity", "carrier_shutdown_currentCell_usage_threshold"].tolist()
        wakeup_th = df_ep_input.loc[df_ep_input["cell_role"] == "coverage", "carrier_wakeUp_pairedCell__currentCell_usage_threshold"].tolist()
        
        assert np.unique(wakeup_th).shape[0] == 1, "All coverage cells must have the same wakeup threshold."
        assert np.unique(shutdown_th).shape[0] == 1, "All capacity cells must have the same shutdown threshold."
        shutdown_th = shutdown_th[0]
        wakeup_th = wakeup_th[0]
        
        # Get all names to modify in final dedalo associated Giulia df_ep
        mask_coverage_df_giulia = (self.giulia_obj.network_deployment_obj.df_ep["cell_role"] == "coverage") | (self.giulia_obj.network_deployment_obj.df_ep["cell_role"] == "coverage_capacity")
        coverage_cell_name_list_df_giulia: list = self.giulia_obj.network_deployment_obj.df_ep.loc[mask_coverage_df_giulia, "name"].tolist()

        mask_capacity_df_giulia = (self.giulia_obj.network_deployment_obj.df_ep["cell_role"] == "capacity") | (self.giulia_obj.network_deployment_obj.df_ep["cell_role"] == "coverage_capacity")
        capacity_cell_name_list_df_giulia: list = self.giulia_obj.network_deployment_obj.df_ep.loc[mask_capacity_df_giulia, "name"].tolist()

        # Set shutdown threshold for capacity cells in dedalo associated Giulia
        self.set_giulia_network_df_ep(
            name_cell_list=coverage_cell_name_list_df_giulia,
            name_attribute="carrier_wakeUp_pairedCell__currentCell_usage_threshold",
            value_list=wakeup_th,
        )
        self.set_giulia_network_df_ep(
            name_cell_list=capacity_cell_name_list_df_giulia,
            name_attribute="carrier_shutdown_currentCell_usage_threshold",
            value_list=shutdown_th,
        )

        return 


    def set_pairing_df_ep_AllConnectedHierarchical(self) -> None:
        """
        Update Giulia's df_ep pairing information by letting be the haps paired with all cells.
        """
        # Basic Checks
        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise TypeError("The network_deployment_obj should be a Network object, not a list.")
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise TypeError(
                "The network_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")
        
        # take the last index of self.giulia_obj.network_deployment_obj.df_ep
        # as it correspond to the HAPS
        index_haps_in_df_ep: int = self.giulia_obj.network_deployment_obj.df_ep.shape[0] - 1

        # Get all ground cell ID 
        id_cell_list_ground: list = self.giulia_obj.network_deployment_obj.df_ep["ID"].tolist()[:-1]

        # Get all paired cell associated to the HAPS
        haps_paired_cells: list = self.giulia_obj.network_deployment_obj.df_ep["paired_cell"][index_haps_in_df_ep]

        # Add all ground cells to the HAPS paired cell list if not already present
        new_paired_cells: list = haps_paired_cells.copy()
        for cell in id_cell_list_ground:
            if cell not in haps_paired_cells:
                new_paired_cells.append(cell)
        # Sort new_paired_cells
        new_paired_cells.sort()

        # Set the new paired cell list for the HAPS in the Giulia df_ep
        self.giulia_obj.network_deployment_obj.df_ep.at[index_haps_in_df_ep, "paired_cell"] = cast(Any, new_paired_cells)

        return


    def get_target_rate_per_ue_Mbps(self, mu_lin: float = 1.0,
                                    sigma_lin: float = 1.0,
                                    n_sample: int = 1,
                                    seed: int|None = None) -> np.ndarray:
        
        # Set Seed
        seed = 42 if seed is None else seed
        rng: np.random.RandomState = np.random.RandomState(seed)
        """Generate a target rate for a UE based on a log-normal distribution."""

        x = rng.lognormal(mean=np.log(mu_lin), sigma=np.log(sigma_lin), size=n_sample).astype(np.float32)
        
        # Max value for the target rate in Mbps
        p = 0.95
        x = np.clip(x, 0, 
                    np.exp( np.log(mu_lin) + np.log(sigma_lin) * norm.ppf(p) ) )

        return x

        

    # Execute one simulation synchronisation step for Giulia.
    def process_static(self, episode_index: int = 0, **kwargs) -> None:
        """Synchronise Giulia with Dedalo choices for a specific simulation step.

        Parameters
        ----------
        episode_index: int, optional
            Episode offset used to select the hour of the day. Defaults to 0.
        """
        self.episode_index = episode_index
        self.step_index = 0
        del episode_index

        # # Get the UE density for the current hour.
        value_ue_per_UEsite_cell_per_specific_hour = self.get_ue_per_UEsite_cell_per_specific_hour(self.episode_index + self.start_hour)
        # Update number of UE per UE site cell in Giulia for the current hour.
        self.set_in_Giulia_ue_per_UEsite_cell(value_ue_per_UEsite_cell_per_specific_hour)

        # Sync the radio unit status between Dedalo and Giulia.
        self.set_dedalo_radio_status_from_giulia_runTime()
        # Set in giulia previous radio status to mantain consistency among episodes
        self.set_giulia_df_ep_radio_status()

        threshold_PRBusage_shutdown_value = kwargs.get("threshold_PRBusage_shutdown_value", None)
        threshold_PRBusage_wakeup_value = kwargs.get("threshold_PRBusage_wakeup_value", None)

        # Update the shutdown and wakeup thresholds.
        # Using parent method to set values in df_ep.
        coverage_cell_name_list: list = self.get_cell_names("cell_role", "coverage")
        capacity_cell_name_list: list = self.get_cell_names("cell_role", "capacity")
        coverage_capacity_cell_name_list: list = self.get_cell_names("cell_role", "coverage_capacity")

        # Add coverage_capacity cells to both lists
        coverage_cell_name_list.extend(coverage_capacity_cell_name_list)
        capacity_cell_name_list.extend(coverage_capacity_cell_name_list)
        coverage_cell_name_list = sorted(list(set(coverage_cell_name_list)))
        capacity_cell_name_list = sorted(list(set(capacity_cell_name_list)))

        assert len(coverage_cell_name_list) > 0 and len(capacity_cell_name_list) > 0, "No coverage or capacity cells found in the deployment."
        

        self.set_giulia_network_df_ep(name_cell_list=capacity_cell_name_list, 
                                      name_attribute="carrier_shutdown_currentCell_usage_threshold", 
                                      value_list=threshold_PRBusage_shutdown_value)
            
        self.set_giulia_network_df_ep(name_cell_list=coverage_cell_name_list,  
                                      name_attribute="carrier_wakeUp_pairedCell__currentCell_usage_threshold", 
                                      value_list=threshold_PRBusage_wakeup_value)
        
        # Set Traffic Rate Gen Stats
        self.traffic_rate_gen_stats: dict|None = kwargs.get("traffic_rate_gen_stats", None)


        # Set Rescheduling info
        self._set_rescheduling_info()

        return
    

    
    def process_realTime(self, step_index: int = 0, **kwargs) -> None:
        # Get the current step index.
        self.step_index = step_index
        del step_index

        #### Fixed PRB per UE
        # Retrieve the rate requirement for UEs. Defaults to None.
        kost_prb_usage_per_ue = kwargs.get("kost_prb_usage_per_ue", None)

        if kost_prb_usage_per_ue is not None:
            assert isinstance(self.giulia_obj.base_stations_obj, BaseStations), "giulia_obj.base_stations_obj must be set"
            
            # Use setattr to dynamically set the attribute
            if hasattr(self.giulia_obj.base_stations_obj, 'fix_prb_req_per_ue'):
                setattr(self.giulia_obj.base_stations_obj, 'fix_prb_req_per_ue', kost_prb_usage_per_ue)
            else:
                raise AttributeError("giulia_obj.base_stations_obj must have the attribute 'fix_prb_req_per_ue'")


        #### Generate Traffic for UEs
        if self.traffic_rate_gen_stats is not None:
            # Set traffic generation model
            self.setGiulia_ue_traffic_generation_model(
                model='rate_requirement'
            )

            # Get the target rate for each UE
            self.ue_rate_requirements_Mbps = self.get_target_rate_per_ue_Mbps(mu_lin=self.traffic_rate_gen_stats['mu_lin'],
                                                                              sigma_lin=self.traffic_rate_gen_stats['sigma_lin'],
                                                                              n_sample=self.giulia_number_of_ue,
                                                                              seed=self.episode_index)
            
            # Set the target rate for each UE in Giulia
            self.setGiulia_ue_target_rate(
                ue_rate_requirements_Mbps=self.ue_rate_requirements_Mbps.tolist()
            )

    
        return
    

    def activate_haps_multiple_wakeup_logic(self, step_index: int) -> None:
        ###### Do only once
        #### Set HAPS specific carrier shutdown method
        if step_index == 0: self._set_giulia_carrier_shutdown_method_HAPS()
        #### Set Variable for multiple wakeup logic
        if step_index == 0: self._set_HAPS_RRC_multiple_wakeup_logic_paired_cells()
    
    def _set_giulia_carrier_shutdown_method_HAPS(self) -> None:
        # Set ad hoc coverage method for multiple wakeup logic
        if not isinstance(self.giulia_obj.simulation_config_obj, Simulation_Config):
            raise TypeError("simulation_config_obj must be a Simulation_Config instance.")
        
        if "HAPS" in self.giulia_obj.simulation_config_obj.scenario_playground_model:
            if "singleHAPS" not in self.giulia_obj.simulation_config_obj.scenario_playground_model:
                raise NotImplementedError(
                    "Only single-HAPS scenarios are supported for multiple wakeup logic. "
                    f"Scenario '{self.giulia_obj.simulation_config_obj.scenario_playground_model}' is not valid."
                )
            if not isinstance(self.giulia_obj.base_stations_obj, BaseStations):
                raise TypeError("giulia_obj.base_stations_obj must be set")
            self.giulia_obj.base_stations_obj.base_stations[-1].rrc_layer.carrier_shutdown_obj._run_method =\
                self.giulia_obj.base_stations_obj.base_stations[-1].rrc_layer.carrier_shutdown_obj._run_coverageCell_HAPS_multiple_wakeup

        return 

    def _set_HAPS_RRC_multiple_wakeup_logic_paired_cells(self) -> None:
        """
        Set the multiple wakeup logic flag in Giulia's RRC layer for each base station.

        This enables that, when a base station sends a wakeup signal to its paired cells,
        all the paired cells (4G only) wake up.
        """
        # --- Basic checks ---
        if not isinstance(self.giulia_obj.simulation_config_obj, Simulation_Config):
            raise TypeError("simulation_config_obj must be a Simulation_Config instance.")

        if not isinstance(self.giulia_obj.base_stations_obj, BaseStations):
            raise TypeError("base_stations_obj must be a BaseStations instance.")

        # --- Scenario check ---
        scenario_model = self.giulia_obj.simulation_config_obj.scenario_playground_model

        coverage_capacity_cell_name_list: list = self.get_cell_names("cell_role", "coverage_capacity")

        if "HAPS" in scenario_model:

            if "singleHAPS" not in scenario_model:
                raise NotImplementedError(
                    "Only single-HAPS scenarios are supported for multiple wakeup logic. "
                    f"Scenario '{scenario_model}' is not valid."
                )

            value_dict: dict = {}
            for cell, paired in self.pairing_4G_only_dict.items():
                if f"cell_{cell}" in coverage_capacity_cell_name_list:
                    value_dict[cell] = paired



            # Apply multiple wakeup logic to the HAPS base station
            setattr(self.giulia_obj.base_stations_obj.base_stations[-1].rrc_layer, # Dest Object
                    'cell_multiple_wakeup_id_dict', # Attribute Name
                    value_dict # Attribute Value
                    )
        else:
            pass
        
        return
    

    def _set_in_ground_multiple_pairing_4G_only(self) -> None:

        pairing_dict = self.pairing_4G_only_dict
        # Basic Checks
        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise TypeError("The network_deployment_obj should be a Network object, not a list.")
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise TypeError(
                "The network_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")

        # Get ID from giulia df_ep
        df_ep = self.giulia_obj.network_deployment_obj.df_ep

        # Get coverage cells names
        coverage_capacity_cell_name_list: list = self.get_cell_names("cell_role", "coverage_capacity")

        if len(coverage_capacity_cell_name_list) == 0:
            raise ValueError("No coverage_capacity cells found in the deployment.")

        for cell_name in coverage_capacity_cell_name_list:

            # Get ID of the cell from df_ep
            cell_id: int = df_ep[df_ep["name"] == cell_name]["ID"].tolist()[0]
            idx = self.giulia_obj.network_deployment_obj.df_ep.index[self.giulia_obj.network_deployment_obj.df_ep["name"] == cell_name][0]   

            # Get paired cells
            current_paired_cells: list = df_ep[df_ep["name"] == cell_name]["paired_cell"].tolist()[0]

            # Case where no paired cells are present,
            # thus, init it as an empty list.
            if current_paired_cells is None: 
                current_paired_cells = []

            # Create new paired cells
            new_paired_cells: list = current_paired_cells.copy()
            new_paired_cells.extend(pairing_dict[cell_id])

            # Set new paired cells in giulia df_ep
            self.giulia_obj.network_deployment_obj.df_ep.at[idx, "paired_cell"] = cast(Any, new_paired_cells)

        return



    def _compute_pairing_coverage_cell_4G_only(self) -> dict[int, list[int]]:

        initial_log = get_log_level()
        set_log_level(parse_log_level(3))
        
        if not isinstance(self.giulia_obj.simulation_config_obj, Simulation_Config):
            raise TypeError("The simulation_config_obj should be a Simulation_Config object, not a list.")
    
        # Get Results from previous Giulia
        preset = "GiuliaStd"
        scenario_model = self.giulia_obj.simulation_config_obj.scenario_playground_model
        scenario_model = "3GPPTR38_901_4G" 
        ue_playground_model = self.giulia_obj.simulation_config_obj.ue_playground_model
        ue_mobility: str|None = None
        link_direction: str = "downlink"
        wraparound: bool = self.giulia_obj.simulation_config_obj.wraparound
        save_results: int = 0
        plots: bool = False
        number_of_episodes: int = 1
        regression: bool = False
        enable_GPU: bool = self.giulia_obj.simulation_config_obj.enable_GPU
        ue_distribution: str = "grid"

        # Giulia_Pairing does not build its own deployment: it takes the site, UE playground,
        # hotspot and network objects from the caller. Build them here for the 4G-only scenario,
        # with a throw-away configuration, as the standard preset does for the main scenario.
        sn_indicator: bool = importlib.util.find_spec('sionna') is not None
        _sim_config_4G = Simulation_Config(preset, scenario_model, ue_playground_model, ue_distribution, ue_mobility,
                                           link_direction, wraparound, save_results, plots, number_of_episodes, regression,
                                           sn_indicator, None, enable_GPU, "Pairing", False)

        site_deployment_obj = Site_siteDeployment(_sim_config_4G.wraparound, _sim_config_4G.scenario_playground_model)
        site_deployment_obj.process()

        ue_playground_deployment_obj = Site_uePlaygroundDeployment(_sim_config_4G, _sim_config_4G.wraparound,
                                                                   _sim_config_4G.ue_playground_model, _sim_config_4G.ue_distribution)
        ue_playground_deployment_obj.process()

        ue_hotspot_deployment_obj = Hotspot(_sim_config_4G, ue_playground_deployment_obj)
        ue_hotspot_deployment_obj.process()

        network_deployment_obj = Network(_sim_config_4G, site_deployment_obj, ue_hotspot_deployment_obj)
        network_deployment_obj.process()

        # Create Giulia instance iposing a grid of UEs
        # NOTE: Giulia_Pairing disables Saveable and it is not enabled again here (see the note above the class).
        _giulia_pairing_cov = Giulia_Pairing(preset, ue_playground_model, ue_distribution, ue_mobility,
                                        link_direction, wraparound, save_results, plots, number_of_episodes, regression, enable_GPU,
                                        site_deployment_obj=site_deployment_obj,
                                        ue_playground_deployment_obj=ue_playground_deployment_obj,
                                        ue_hotspot_deployment_obj=ue_hotspot_deployment_obj,
                                        network_deployment_obj=network_deployment_obj)

        # Configure Giulia
        _giulia_pairing_cov.configure(0, 1)

        assert not isinstance(_giulia_pairing_cov.shadowing_map_cell_obj, list), "shadowing_map_cell_obj has not been processed yet"
        # Set shadow fading to 0dB
        for key in _giulia_pairing_cov.shadowing_map_cell_obj.shadowing_maps_dB.keys():
            for m in range(len(_giulia_pairing_cov.shadowing_map_cell_obj.shadowing_maps_dB[key])):
                _giulia_pairing_cov.shadowing_map_cell_obj.shadowing_maps_dB[key][m] =\
                    np.zeros_like(_giulia_pairing_cov.shadowing_map_cell_obj.shadowing_maps_dB[key][m])
                
        # Run a single episode/step to compute pairing needed data
        _giulia_pairing_cov.run_simulation(0, 1)

        # Free memory
        free_memory()


        ###############################
        # Process Pairing Coverage Cells
        ###############################

        # Basic checks
        network = _giulia_pairing_cov.network_deployment_obj

        # Ensure `network_deployment_obj` is a Network instance (not a list or other type)
        if not isinstance(network, Network):
            raise TypeError("The network_deployment_obj should be a Network object, not a list.")

        # Ensure the Network object exposes a pandas DataFrame `df_ep`
        if not isinstance(network.df_ep, pd.DataFrame):
            raise ValueError("The network_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")

        # Pull RSRP matrix (UE x Cell) in dBm
        RSRP_ue_to_cell_dBm = _giulia_pairing_cov.best_serving_cell_ID_per_ue_based_on_SSB_obj.RSRP_ue_to_cell_obj.RSRP_results_per_frequency_layer["all_freq"]["RSRP_ue_to_cell_dBm"]

        # For each UE, select the serving cell = argmax across cells (axis=1)
        serving_cell = RSRP_ue_to_cell_dBm.argmax(axis=1)

        # Get Site ID per cell
        site_id_arr = network.df_ep["site_ID"].to_numpy()

        # Output: dict[cell_idx] -> list of top similar cells after filtering
        out_paired_cell_dict: dict[int, list[int]] = {}

        assert RSRP_ue_to_cell_dBm.shape[1] == 57, "Expected 57 cells in the RSRP matrix. This code is designed for the 4G-only scenario."
        # Compute pairwise Wasserstein distances and similarities for each cell
        for cell_idx in np.arange(RSRP_ue_to_cell_dBm.shape[1]):
            # Indices of UEs whose serving cell equals the current cell
            idx_connected = np.where(serving_cell == cell_idx)[0]

            # Subset of the RSRP matrix for UEs connected to this cell
            # Shape: (n_connected_ues, n_cells)
            RSRP_ueConnected_to_cell_dBm = RSRP_ue_to_cell_dBm[idx_connected]

            # Alias used for the pairwise-distance computation below (kept for clarity)
            X = RSRP_ueConnected_to_cell_dBm

            # Number of cells = number of columns in X
            n_cells = X.shape[1]

            # Initialize Wasserstein distance matrix (n_cells x n_cells)
            W = np.zeros((n_cells, n_cells), dtype=float)

            # Compute symmetric pairwise distances for (i, j), i < j
            for i in range(n_cells):
                for j in range(i + 1, n_cells):
                    # Wasserstein distance between the distributions of RSRP values
                    # for cell i vs cell j across the connected UEs
                    d = wasserstein_distance(X[:, i], X[:, j])

                    # Fill both upper and lower triangle to keep the matrix symmetric
                    W[i, j] = d
                    W[j, i] = d

            # Convert distances to similarities:
            # similarity = 1 / (1 + distance)
            W = 1.0 / (1.0 + W)

            # Remove self-similarity (force diagonal to 0)
            np.fill_diagonal(W, 0.0)

            # Take the 5 highest similarity scores in row `cell_idx` (descending order)
            top_cell = np.argsort(W[cell_idx])[::-1][:5]

            # Filter out:
            # 1) cells belonging to the same site as cell_idx
            # 2) the cell itself (redundant due to diagonal=0, but for safety)
            top_cell = [
                tc for tc in top_cell
                if site_id_arr[tc] != site_id_arr[cell_idx] and tc != cell_idx
            ]

            # Store results for this cell
            out_paired_cell_dict[cell_idx] = top_cell
            

        # Reset Log Level
        set_log_level(initial_log)
        return out_paired_cell_dict

        def _plot_debug_pairing_coverage_cell(_giulia_pairing_cov, cell_idx: int, top_cell: list) -> None:

            # Basic checks
            
            
            # Plots
            # Get positions of BSs
            bs_positions = _giulia_pairing_cov.network_deployment_obj.df_ep[['position_x_m', 'position_y_m']].to_numpy()
            # Get boresight directions of BSs
            bs_boresight_deg = _giulia_pairing_cov.network_deployment_obj.df_ep['antenna_config_hor_alpha_mec_bearing_deg'].to_numpy()

            plt.scatter(bs_positions[:, 0], bs_positions[:, 1], c='red', marker="^", s=150, label='Base Stations', edgecolors='k')

            plt.scatter(bs_positions[cell_idx, 0] + 100 * np.cos(np.deg2rad(bs_boresight_deg[cell_idx])), 
                        bs_positions[cell_idx, 1] + 100 * np.sin(np.deg2rad(bs_boresight_deg[cell_idx])), 
                        c='blue', marker="^", s=100, label='Selected BS', edgecolors='k')

            plt.scatter(bs_positions[top_cell, 0] + 100 * np.cos(np.deg2rad(bs_boresight_deg[top_cell])), 
                        bs_positions[top_cell, 1] + 100 * np.sin(np.deg2rad(bs_boresight_deg[top_cell])), 
                        c='green', marker="^", s=100, label='Similar BS', edgecolors='k')

            plt.title(f"Cell {cell_idx}\nTop Similar Cells {top_cell}")
            plt.grid(True, ls='-.', alpha=0.75)
            plt.axis('equal')
            plt.legend()
            plt.show()
            return

        for cell_idx, top_cell in out_paired_cell_dict.items():
            _plot_debug_pairing_coverage_cell(_giulia_pairing_cov, cell_idx, top_cell)


    def _set_rescheduling_info(self) -> None:

        return

        update_event_scheduling_info = {
            "snapshot_control": -1,
            "eventDriven_obj": -1,
            "ue_deployment_obj": -1,
            "ue_antenna_array_structure_obj": -1,
            "channel_sn_obj": -1,
            "distance_angles_ue_to_cell_obj": -1,
            "distance_angles_ueAnt_to_cellAnt_obj": -1,
            "antenna_pattern_gain_ue_to_cell_obj": -1,
            "array_steering_vector_ue_to_cell_obj": -1,
            "LOS_probability_ue_to_cell_obj": -1,
            "K_factor_ue_to_cell_obj": -1,
            "path_loss_ue_to_cell_obj": -1,
            "o2i_penetration_loss_ue_to_cell_obj": -1,
            "shadowing_gain_ue_to_cell_obj": -1,
            "slow_channel_gain_ue_to_cell_obj": -1,
            "los_channel_gain_ue_to_cell_obj": -1,
            
            # "time_frequency_resource_obj": 
            #     {"update_ue_carrier_info":-1},
                
            "dl_fast_fading_gain_ueAnt_to_cellAnt_obj": -1,
            "channel_gain_ue_to_cell_obj": -1,
            "SSB_precoded_channel_gain_no_fast_fading_ue_to_cell_obj": -1,
            "SSB_precoded_channel_gain_ue_to_cell_obj": -1,
            "CSI_RS_precoded_channel_gain_ue_to_cell_obj": -1,
            "dl_noise_ue_to_cell_obj": -1,
            # "SSB_RSS_per_PRB_ue_to_cell_obj": self.dflt_rescheduling_us,
            # "CSI_RS_RSS_per_PRB_ue_to_cell_obj": self.dflt_rescheduling_us,
            # "CRS_RSRP_no_fast_fading_ue_to_cell_obj": self.dflt_rescheduling_us,
    
            # "SSB_RSRP_no_fast_fading_ue_to_cell_obj": {
            #     "process": self.dflt_rescheduling_us,
            #     "set_RSRP_ue_to_cell_dBm": self.dflt_rescheduling_us
            # },
    
            # "SSB_RSRP_ue_to_cell_obj": self.dflt_rescheduling_us,
            # "CSI_RS_RSRP_ue_to_cell_obj": self.dflt_rescheduling_us,
    
            # "best_serving_cell_ID_per_ue_based_on_CRS_obj": {
            #     "process": self.dflt_rescheduling_us,
            #     "calculate_server_stats": self.dflt_rescheduling_us
            # },
    
            # "best_serving_cell_ID_per_ue_based_on_SSB_obj": {
            #     "process": self.dflt_rescheduling_us,
            #     "calculate_server_stats": self.dflt_rescheduling_us,
            # },
    
            # "cell_reselection_based_on_CRS_obj": self.dflt_rescheduling_us,
            # "cell_reselection_based_on_SSB_obj": self.dflt_rescheduling_us,
            # "best_serving_CSI_RS_per_ue_obj": self.dflt_rescheduling_us,
            # "CRS_sinr_ue_to_cell_obj": self.dflt_rescheduling_us,
            # "SSB_sinr_ue_to_cell_obj": self.dflt_rescheduling_us,

            # "base_stations_obj": {
            #     "process": self.dflt_rescheduling_us,
            #     "mac": self.dflt_rescheduling_us,
            #     "carrier_shutdown": self.dflt_rescheduling_us,
            # },
            

            # "CSI_RS_sinr_per_PRB_ue_to_cell_obj": self.dflt_rescheduling_us,
            # "ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_obj": self.dflt_rescheduling_us,
            # "ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_obj": self.dflt_rescheduling_us,
            # "performance_obj": self.dflt_rescheduling_us,
            # "ue_mobility_obj": self.dflt_rescheduling_us,
            # "bs_mobility_obj": self.dflt_rescheduling_us
        }

        assert isinstance(self.giulia_obj.simulation_config_obj, Simulation_Config), \
            "simulation_config_obj must be a Simulation_Config instanciated."
        
        # Update the event scheduling info in Giulia's simulation configuration.
        for key, value in update_event_scheduling_info.items():
            self.giulia_obj.simulation_config_obj.event_scheduling_info[key] = value

        return


    #### PLOTTING UTILITIES ####
    ############################

    def _plot_prb_required_per_ue(self) -> None:

        """Visualise the DL PRBs required per UE."""
        assert isinstance(self.giulia_obj.base_stations_obj, BaseStations), "giulia_obj.base_stations_obj must be set"
        assert isinstance(self.giulia_obj.ue_deployment_obj, UE_Deployment), "giulia_obj.ue_deployment_obj must be set"

        # Data
        values = self.giulia_obj.base_stations_obj.resource_allocation['dl_PRBs_required_per_ue']
        ue_indices = np.arange(len(self.giulia_obj.ue_deployment_obj.df_ep))

        # Plot
        # plt.figure(figsize=(10, 6))
        bars = plt.bar(ue_indices, values, color='green')

        # Labels & Title
        plt.xlabel('UE Index', fontsize=12)
        plt.ylabel('DL PRBs Required per UE', fontsize=12)
        plt.title('Downlink PRBs Required per User Equipment (UE)', fontsize=14, fontweight='bold')

        # Grid & Style
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.xticks(ue_indices)  # show all UE indices

        # Annotate values on top of bars
        # for bar in bars:
        #     height = bar.get_height()
        #     plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, 
        #             f'{int(height)}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.ylim(0, 10)
        plt.show()


    
    def _info_number_of_active_ues(self, hour) -> None:
        """
        Log the number of active UEs in Giulia for the specified hour.
        """
        # Basic Checks
        if not isinstance(self.giulia_obj.ue_deployment_obj, UE_Deployment):
            raise TypeError("The ue_deployment_obj should be a UE_Deployment object, not a list.")
        if not isinstance(self.giulia_obj.ue_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("The ue_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")
        # Get UE df_ep
        N_active_UE = self.giulia_obj.ue_deployment_obj.df_ep

        info(f"Total active UEs in Giulia at hour {hour + self.start_hour}: {N_active_UE.shape[0]}")

        return


    # Plot the UE proportion curve for the 24-hour profile.
    def _plot_proportion_ue(self) -> None:
        """Visualise the normalised UE proportion across the day."""
        proportion_ue = list(self.proportion_ue_dict.values())
        plt.plot(proportion_ue, marker='o')
        plt.title("Proportion of UEs Over 24 Hours")
        plt.xlabel("Hour of Day")
        plt.ylabel("Proportion of UEs")
        plt.xticks(ticks=range(24+1), labels=[f"{h}:00" for h in range(24+1)], rotation=45)
        plt.grid(True, ls='-.', alpha=0.75)
        plt.show()

    # Plot the UE density per BS against the proportion curve.
    def _plot_ue_deployed_per_hour(self) -> None:
        """Plot the UE density per base station alongside the proportion curve."""
        fig, ax1 = plt.subplots()  # larger figure for readability

        hours = list(self.ue_per_UEsite_cell_per_hour_dict.keys())
        x_ticks = range(len(hours))  # integer ticks for the hours

        proportion_ue = list(self.proportion_ue_dict.values())
        ax1.plot(x_ticks, proportion_ue, marker='o', label='Proportion of UEs', color='tab:blue', ls='-.', alpha=0.25)
        ax1.set_xlabel("Hour of Day", fontsize=14)
        ax1.set_ylabel("Proportion of UEs", color='tab:blue', fontsize=14)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.tick_params(axis='x', rotation=45, labelsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2 = ax1.twinx()
        
        ax2.plot(x_ticks, [number_of_ue for number_of_ue in self.ue_per_UEsite_cell_per_hour_dict.values()],
                 marker='s', color='tab:orange', label='Density of UEs per BS')
        ax2.set_ylabel("Density of UEs per BS", color='tab:orange', fontsize=14)
        ax2.tick_params(axis='y', labelcolor='tab:orange')


        plt.title(f"UE Distribution Over 24 Hours", fontsize=16, weight='bold')
        plt.xticks(ticks=x_ticks, labels=[f"{h}:00" for h in range(24)], fontsize=12)  # readable labels
        fig.tight_layout()

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=12)

        plt.show()

        return
    
    def _plot_network_layout(self, enable: bool = True) -> None:

        # Skip plotting if not enabled.
        if not enable: return

        """Visualise the network layout from Giulia's deployment data."""
        if not isinstance(self.giulia_obj.network_deployment_obj, Network):
            raise TypeError("The network_deployment_obj should be a Network object, not a list.")
        if not isinstance(self.giulia_obj.ue_deployment_obj, UE_Deployment):
            raise TypeError("The ue_deployment_obj should be a Network object, not a list.")
        
        if not isinstance(self.giulia_obj.network_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("The network_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")
        if not isinstance(self.giulia_obj.ue_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("The ue_deployment_obj must have a 'df_ep' attribute of type pandas DataFrame.")

        # Extract positions from Giulia deployment DataFrames.
        df_ep_BS = self.giulia_obj.network_deployment_obj.df_ep
        df_ep_ue = self.giulia_obj.ue_deployment_obj.df_ep

    
        # plt.figure(figsize=(8, 8))
        plt.scatter(df_ep_ue['position_x_m'], df_ep_ue['position_y_m'], c='blue', marker="o", s=20, label='UE', edgecolors='k', alpha=0.25) # type: ignore
        plt.scatter(df_ep_BS['position_x_m'], df_ep_BS['position_y_m'], c='red', marker="^", s=100, label='Base Stations', edgecolors='k') # type: ignore

        plt.title(f'Network Layout\n N_BS: {len(df_ep_BS)}, N_UE: {len(df_ep_ue)}')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True, ls='-.', alpha=0.75)
        plt.axis('equal')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    print('This the module dedalo/dedalo_EE.py. It is not meant to be run directly.')
