#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:39:12 2024

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@itea.upv.es
"""
import os.path
import shutil
import time
from collections import Counter


import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow.lib import ArrowNotImplementedError

from giulia.fs import outputs_dir
from giulia.fs import results_file
from giulia.inputs.config.OutputFormat import OutputFormat
from giulia.inputs.config.npz import NPZFormat
from giulia.inputs.config.parquet import ParquetFormat
from giulia.logger import debug, verbose, error, warning
from giulia.outputs.saveable import _get_saveables
from giulia.outputs.processing import pack_output
from giulia.presets import Giulia
from giulia.tools.tools import log_elapsed_time

from typing import List, Dict, Any, TypeVar, Optional
T = TypeVar('T', bound=OutputFormat)

# The only non-exploded label: a wide table with one row per (episode, step) holding the core
# hardcoded KPIs. EVERY other label is treated as "exploded" — one row per entity, with an
# entity-id index column derived from the label name (see _entity_id_column). A new per-entity
# parquet therefore appears automatically as soon as a Saveable declares a new label (e.g.
# 'bbu_kpi' -> a '<project>_bbu_kpi' file with a 'bbu_id' column); nothing here needs to change.
GENERIC_LABEL = 'generic'


def _entity_id_column(label: str) -> str:
    """
    Entity-id (index) column for an exploded label, derived from its name:
    ``ue_kpi`` -> ``ue_id``, ``cell_kpi`` -> ``cell_id``, ``ru_kpi`` -> ``ru_id``,
    ``bbu_kpi`` -> ``bbu_id``. A trailing ``_kpi`` / ``_kpis`` is stripped before adding ``_id``.
    """
    base = label
    for suffix in ('_kpis', '_kpi'):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return f'{base}_id'

class Episode_Performance:
    """
    Stores and manages performance metrics for multiple simulation episodes. 
    It tracks network signal quality, throughput, and power consumption.
    """

    formats: List[OutputFormat]
    """All the formats to export the data into."""

    def __init__(
            self,
            number_of_episodes: int,
            number_of_steps: int,
            regression: bool,
            output_dir: Optional[str] = None,
            output_formats: Optional[List[OutputFormat]] = None,
            fields: Optional[List[str]] = None,
    ):
        """
        Initializes the class and allocates empty numpy arrays for storing episode performance metrics.
    
        Args:
            number_of_episodes (int): Number of episodes to store performance data for.
            regression (bool): Flag indicating if regression analysis is performed.
        """
        # Store the number of episodes and regression flag
        self.number_of_episodes: int = number_of_episodes
        self.number_of_steps: int = number_of_steps
        self.regression: bool = regression

        # Store the formats to be exported into
        self.output_dir: Optional[str] = output_dir
        self.formats: Optional[List[OutputFormat]] = output_formats if output_formats is not None else []
        self.fields: Optional[List[str]] = fields

        # Function to initialize an empty numpy array for each episode
        def init_empty_list() -> npt.NDArray:
            return np.array([None for _ in range(self.number_of_episodes)], dtype=object)
    
        # ---- CRS (Channel State Reporting Reference Signal) ----
        self.r_best_serving_cell_ID_per_ue_based_on_CRS: npt.NDArray[np.int_] = init_empty_list()
        self.r_best_serving_CRS_distance_3d_m: npt.NDArray[np.float64] = init_empty_list()
        self.r_best_serving_CRS_coupling_gain_dB: npt.NDArray[np.float64] = init_empty_list()
        self.r_best_serving_CRS_re_noise_per_ue_dBm: npt.NDArray[np.float64] = init_empty_list()
        self.r_best_serving_CRS_rsrp_per_ue_dBm: npt.NDArray[np.float64] = init_empty_list()
        self.r_CRS_RSRP_no_fast_fading_ue_to_cell_dBm: npt.NDArray[np.object_] = init_empty_list()

        self.r_CRS_sinr_ue_to_cell_dB: npt.NDArray[np.float64] = init_empty_list()
        self.r_CRS_usefulPower_ue_dBm: npt.NDArray[np.float64] = init_empty_list()
        self.r_CRS_interfPower_ue_dBm: npt.NDArray[np.float64] = init_empty_list()
    
        # ---- SSB (Synchronization Signal Block) ----
        self.r_best_serving_cell_ID_per_ue_based_on_SSB: npt.NDArray[np.int_] = init_empty_list()
        self.r_best_serving_SSB_per_ue: npt.NDArray[np.int_] = init_empty_list()
        self.r_best_serving_SSB_distance_3d_m: npt.NDArray[np.float64] = init_empty_list()
        self.r_best_serving_SSB_coupling_gain_dB: npt.NDArray[np.float64] = init_empty_list()
        self.r_best_serving_SSB_re_noise_per_ue_dBm: npt.NDArray[np.float64] = init_empty_list()
        self.r_best_serving_SSB_rsrp_per_ue_dBm: npt.NDArray[np.float64] = init_empty_list()
        self.r_SSB_RSRP_no_fast_fading_ue_to_cell_dBm: npt.NDArray[np.object_] = init_empty_list()

        self.r_SSB_sinr_ue_to_cell_dB: npt.NDArray[np.float64] = init_empty_list()
        self.r_SSB_usefulPower_ue_dBm: npt.NDArray[np.float64] = init_empty_list()
        self.r_SSB_interfPower_ue_dBm: npt.NDArray[np.float64] = init_empty_list()
    
        # ---- CSI-RS (Channel State Information Reference Signal) ----
        self.r_best_serving_CSI_RS_per_ue: npt.NDArray[np.int_] = init_empty_list()
        self.r_best_serving_CSI_RS_rsrp_per_ue_dBm: npt.NDArray[np.float64] = init_empty_list()
        self.r_effective_CSI_RS_sinr_ue_to_cell_dB: npt.NDArray[np.float64] = init_empty_list()
        # self.r_CSI_RS_usefulPower_ue_dBm = init_empty_list()
        # self.r_CSI_RS_interfPower_ue_dBm = init_empty_list()
        
        # ---- UE counts per cell and beam ----
        self.r_ues_per_cell_based_on_CRS: npt.NDArray[np.int_] = init_empty_list()
        self.r_ues_per_cell_based_on_SSB: npt.NDArray[np.int_] = init_empty_list()
        self.r_ues_per_SSB_beam: npt.NDArray[np.int_] = init_empty_list()
        self.r_ues_per_CSI_RS_beam: npt.NDArray[np.int_] = init_empty_list()
        
        # ---- Cell and beam activity ----
        self.r_cell_activity_per_ue: npt.NDArray[np.object_] = init_empty_list()
        self.r_SSB_beam_activity_per_ue: npt.NDArray[np.object_] = init_empty_list()
        self.r_CSI_RS_beam_activity_per_ue: npt.NDArray[np.object_] = init_empty_list()
        self.r_PRB_usage_per_cell: npt.NDArray[np.float64] = init_empty_list()

        # ---- Base station radio-unit state ----
        self.r_cell_radio_unit_statuses: npt.NDArray[np.object_] = init_empty_list()

        # ---- Throughput and Power Consumption ----
        self.r_ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps: npt.NDArray[np.float64] = init_empty_list()
        self.r_ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps: npt.NDArray[np.float64] = init_empty_list()
        self.r_cell_throughput_Mbps: npt.NDArray[np.float64] = init_empty_list()
        self.r_carrier_throughput_Mbps: npt.NDArray[np.float64] = init_empty_list()
        self.r_ue_throughput_per_carrier_Mbps: npt.NDArray[np.float64] = init_empty_list()

        self.r_power_consumption_per_radio_unit_incl_bbu_kW : npt.NDArray[np.float64]= init_empty_list()
        self.r_power_consumption_per_radio_unit_kW: npt.NDArray[np.float64] = init_empty_list()
        self.r_network_power_consumption_per_radioType_kW: npt.NDArray[np.float64] = init_empty_list()
        self.r_total_network_power_consumption_kW: npt.NDArray[np.float64] = init_empty_list()
        
        self.saveables_data: npt.NDArray[np.object_] = init_empty_list() # npt.NDArray[Dict[str, Any]]


    def save_episode_performance(self, episode_index: int, g: 'Giulia'):
        """
        Saves the performance metrics for a specific episode.

        Args:
            episode_index (int): The index of the episode being saved.
            g (Giulia): The Giulia simulation object containing performance data.
        """

        # ---- CRS (Channel State Reporting Reference Signal) ----
        self.r_best_serving_cell_ID_per_ue_based_on_CRS[episode_index] = g.performance_obj.r_e_best_serving_cell_ID_per_ue_based_on_CRS
        self.r_best_serving_CRS_distance_3d_m[episode_index] = g.performance_obj.r_e_best_serving_CRS_distance_3d_m  
        self.r_best_serving_CRS_re_noise_per_ue_dBm[episode_index] = g.performance_obj.r_e_best_serving_CRS_re_noise_per_ue_dBm
        self.r_best_serving_CRS_coupling_gain_dB[episode_index] = g.performance_obj.r_e_best_serving_CRS_coupling_gain_dB
        self.r_best_serving_CRS_rsrp_per_ue_dBm[episode_index] = g.performance_obj.r_e_best_serving_CRS_rsrp_per_ue_dBm
        self.r_CRS_RSRP_no_fast_fading_ue_to_cell_dBm[episode_index] = g.performance_obj.r_e_CRS_RSRP_no_fast_fading_ue_to_cell_dBm
        
        self.r_CRS_sinr_ue_to_cell_dB[episode_index] = g.performance_obj.r_e_CRS_sinr_ue_to_cell_dB
        self.r_CRS_usefulPower_ue_dBm[episode_index] = g.performance_obj.r_e_CRS_usefulPower_ue_dBm
        self.r_CRS_interfPower_ue_dBm[episode_index] = g.performance_obj.r_e_CRS_interfPower_ue_dBm        
    
        # ---- SSB (Synchronization Signal Block) ----
        self.r_best_serving_cell_ID_per_ue_based_on_SSB[episode_index] = g.performance_obj.r_e_best_serving_cell_ID_per_ue_based_on_SSB
        self.r_best_serving_SSB_per_ue[episode_index] = g.performance_obj.r_e_best_serving_SSB_per_ue  
        self.r_best_serving_SSB_distance_3d_m[episode_index] = g.performance_obj.r_e_best_serving_SSB_distance_3d_m  
        self.r_best_serving_SSB_coupling_gain_dB[episode_index] = g.performance_obj.r_e_best_serving_SSB_coupling_gain_dB
        self.r_best_serving_SSB_re_noise_per_ue_dBm[episode_index] = g.performance_obj.r_e_best_serving_SSB_re_noise_per_ue_dBm
        self.r_best_serving_SSB_rsrp_per_ue_dBm[episode_index] = g.performance_obj.r_e_best_serving_SSB_rsrp_per_ue_dBm
        self.r_SSB_RSRP_no_fast_fading_ue_to_cell_dBm[episode_index] = g.performance_obj.r_e_SSB_RSRP_no_fast_fading_ue_to_cell_dBm

        self.r_SSB_sinr_ue_to_cell_dB[episode_index] = g.performance_obj.r_e_SSB_sinr_ue_to_cell_dB
        self.r_SSB_usefulPower_ue_dBm[episode_index] = g.performance_obj.r_e_SSB_usefulPower_ue_dBm
        self.r_SSB_interfPower_ue_dBm[episode_index] = g.performance_obj.r_e_SSB_interfPower_ue_dBm        
    
        # ---- CSI-RS (Channel State Information Reference Signal) ----
        self.r_best_serving_CSI_RS_per_ue[episode_index] = g.performance_obj.r_e_best_serving_CSI_RS_per_ue
        self.r_best_serving_CSI_RS_rsrp_per_ue_dBm[episode_index] = g.performance_obj.r_e_best_serving_CSI_RS_rsrp_per_ue_dBm 

        self.r_effective_CSI_RS_sinr_ue_to_cell_dB[episode_index] = g.performance_obj.r_e_effective_CSI_RS_sinr_ue_to_cell_dB
        # self.r_CSI_RS_usefulPower_ue_dBm[episode_index] = g.performance_obj.r_e_CSI_RS_usefulPower_ue_dBm
        # self.r_CSI_RS_interfPower_ue_dBm[episode_index] = g.performance_obj.r_e_CSI_RS_interfPower_ue_dBm    
        
        # ---- UE counts per cell and beam ----
        self.r_ues_per_cell_based_on_CRS[episode_index] = g.performance_obj.r_e_ues_per_cell_based_on_CRS
        self.r_ues_per_cell_based_on_SSB[episode_index] = g.performance_obj.r_e_ues_per_cell_based_on_SSB
        self.r_ues_per_SSB_beam[episode_index] = g.performance_obj.r_e_ues_per_SSB_beam
        self.r_ues_per_CSI_RS_beam[episode_index] = g.performance_obj.r_e_ues_per_CSI_RS_beam
        
        # ---- Cell and beam activity ----
        self.r_cell_activity_per_ue[episode_index] = g.performance_obj.r_e_cell_activity_per_ue
        self.r_SSB_beam_activity_per_ue[episode_index] = g.performance_obj.r_e_SSB_beam_activity_per_ue
        self.r_CSI_RS_beam_activity_per_ue[episode_index] = g.performance_obj.r_e_CSI_RS_beam_activity_per_ue
        self.r_PRB_usage_per_cell[episode_index] = g.performance_obj.r_e_PRB_usage_per_cell

        # ---- Base station radio-unit state ----
        self.r_cell_radio_unit_statuses[episode_index] = g.performance_obj.r_e_cell_radio_unit_statuses

        # ---- Throughput and Power Consumption ----
        self.r_ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps[episode_index] = g.performance_obj.r_e_ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps
        self.r_ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps[episode_index] = g.performance_obj.r_e_ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps
        self.r_cell_throughput_Mbps[episode_index] = g.performance_obj.r_e_cell_throughput_Mbps  
        self.r_carrier_throughput_Mbps[episode_index] = g.performance_obj.r_e_carrier_throughput_Mbps
        self.r_ue_throughput_per_carrier_Mbps[episode_index] = g.performance_obj.r_e_ue_throughput_per_carrier_Mbps      

        self.r_power_consumption_per_radio_unit_incl_bbu_kW[episode_index] = g.performance_obj.r_e_power_consumption_per_radio_unit_incl_bbu_kW
        self.r_power_consumption_per_radio_unit_kW[episode_index] = g.performance_obj.r_e_power_consumption_per_radio_unit_kW
        self.r_network_power_consumption_per_radioType_kW[episode_index] = g.performance_obj.r_e_network_power_consumption_per_radioType_kW
        self.r_total_network_power_consumption_kW[episode_index] = g.performance_obj.r_e_total_network_power_consumption_kW
        
        self.saveables_data[episode_index] = g.performance_obj.r_e_saveables_data


    def _is_field_selected(self, column_name: str) -> bool:
        """
        Whether a column survives the ``fields`` filter.

        A column is kept if it is listed verbatim in ``self.fields``, or — for an expanded 2-D
        column ``<base>.<suffix>`` (e.g. ``RSRP_SSB_no_fast_fading.rsrp_ue_to_cell_dBm.cell0``) — if
        its base name (``...rsrp_ue_to_cell_dBm``) is listed. This lets the user select all the
        expanded columns of a matrix variable by its base name, regardless of the suffix naming.
        With no filter, everything passes.
        """
        if self.fields is None:
            return True
        if column_name in self.fields:
            return True
        return any(column_name.startswith(f + '.') for f in self.fields)

    def _find_format_config(self, format: type[T]) -> Optional[T]:
        """
        Tries to find the configuration of the given format in the list of formats to export into. If not found, returns None.
        :arg format: The format class to find the configuration for (e.g., NPZFormat, ParquetFormat).
        :returns: The configuration of the given format if found in the list of formats to export into, or None otherwise.
        """
        entries = list(filter(lambda x: isinstance(x, format), self.formats))
        if len(entries) <= 0:
            return None
        return entries[0]  # type: ignore


    def save_episodes_performance_in_file(self, project_name: str):
        """
        Writes the collected performance of every episode to disk, in each format enabled
        under ``output.formats``: NPZ (raw core KPIs) and/or Parquet (core KPIs + Saveables).

        Only raw values are exported — no mean/median aggregation is performed here.

        Args:
            project_name (str): The name of the project to store results.
        """
        # Start timer
        t_start = time.perf_counter()

        dataset_root_dir = os.path.join(outputs_dir(), 'outputs') if self.output_dir is None else self.output_dir
        dataset_dir_path = os.path.join(dataset_root_dir, project_name)

        npz_format: Optional[NPZFormat] = self._find_format_config(NPZFormat)
        npz_format = npz_format if npz_format is not None else NPZFormat(False)
        parquet_format: Optional[ParquetFormat] = self._find_format_config(ParquetFormat)
        parquet_format = parquet_format if parquet_format is not None else ParquetFormat(False)

        debug(f'Export in NPZ? {"yes" if npz_format.enable else "no"}')
        debug(f'Export in Parquet? {"yes" if parquet_format.enable else "no"}')
        if parquet_format.enable:
            verbose(f'Parquet config: compression={parquet_format.compression}, rm_uncompressed={parquet_format.rm_uncompressed}')

        # Save NPZ data
        if npz_format.enable:
            verbose('Exporting raw results to NPZ...')
            self._save_episodes_performance_npz(results_file(project_name, 'results-raw'))
            
        # Save Parquet data
        if parquet_format.enable:
            verbose('Exporting raw results to Parquet...')
            self._save_episodes_performance_parquet(
                dataset_dir_path,
                parquet_format.compression,
                parquet_format.use_partition_cols,
                parquet_format.rm_uncompressed
            )

        # Log execution time
        log_elapsed_time('Saving selected outputs in files', t_start)  
        
    def _save_episodes_performance_npz(self, file: str):
        """
        Stores the episodes performance in NPZ (Numpy) format, at the ``file`` path given.

        Each field is stacked along the episode axis into a single array. Only the hardcoded
        core KPIs are written: Saveable variables are exported through Parquet only.

        Args:
            file: The full path of the file to store the data into.
        """
        # Convert a sequence of per-episode results into a single ndarray by stacking
        # them along the first dimension (episodes axis). This preserves the legacy
        # NPZ file format.
        _legacy_format_npz = lambda array: np.stack(array, axis=0)  # type: ignore

        np.savez(
            file,

            # ---- CRS (Channel State Reporting Reference Signal) ----
            best_serving_cell_ID_per_ue_based_on_CRS = _legacy_format_npz(self.r_best_serving_cell_ID_per_ue_based_on_CRS),
            best_serving_CRS_distance_3d_m = _legacy_format_npz(self.r_best_serving_CRS_distance_3d_m),
            best_serving_CRS_coupling_gain_dB = _legacy_format_npz(self.r_best_serving_CRS_coupling_gain_dB),
            best_serving_CRS_re_noise_per_ue_dBm = _legacy_format_npz(self.r_best_serving_CRS_re_noise_per_ue_dBm),
            best_serving_CRS_rsrp_per_ue_dBm = _legacy_format_npz(self.r_best_serving_CRS_rsrp_per_ue_dBm),
            CRS_RSRP_no_fast_fading_ue_to_cell_dBm = _legacy_format_npz(self.r_CRS_RSRP_no_fast_fading_ue_to_cell_dBm),

            CRS_sinr_ue_to_cell_dB = _legacy_format_npz(self.r_CRS_sinr_ue_to_cell_dB),
            CRS_usefulPower_ue_dBm = _legacy_format_npz(self.r_CRS_usefulPower_ue_dBm),
            CRS_interfPower_ue_dBm = _legacy_format_npz(self.r_CRS_interfPower_ue_dBm),

            # ---- SSB (Synchronization Signal Block) ----
            best_serving_cell_ID_per_ue_based_on_SSB = _legacy_format_npz(self.r_best_serving_cell_ID_per_ue_based_on_SSB),
            best_serving_SSB_per_ue = _legacy_format_npz(self.r_best_serving_SSB_per_ue),
            best_serving_SSB_distance_3d_m = _legacy_format_npz(self.r_best_serving_SSB_distance_3d_m),
            best_serving_SSB_coupling_gain_dB = _legacy_format_npz(self.r_best_serving_SSB_coupling_gain_dB),
            best_serving_SSB_re_noise_per_ue_dBm = _legacy_format_npz(self.r_best_serving_SSB_re_noise_per_ue_dBm),
            best_serving_SSB_rsrp_per_ue_dBm = _legacy_format_npz(self.r_best_serving_SSB_rsrp_per_ue_dBm),
            SSB_RSRP_no_fast_fading_ue_to_cell_dBm = _legacy_format_npz(self.r_SSB_RSRP_no_fast_fading_ue_to_cell_dBm),
            SSB_sinr_ue_to_cell_dB = _legacy_format_npz(self.r_SSB_sinr_ue_to_cell_dB),
            SSB_usefulPower_ue_dBm = _legacy_format_npz(self.r_SSB_usefulPower_ue_dBm),
            SSB_interfPower_ue_dBm = _legacy_format_npz(self.r_SSB_interfPower_ue_dBm),

            # ---- CSI-RS (Channel State Information Reference Signal) ----
            best_serving_CSI_RS_per_ue = _legacy_format_npz(self.r_best_serving_CSI_RS_per_ue),
            best_serving_CSI_RS_rsrp_per_ue_dBm = _legacy_format_npz(self.r_best_serving_CSI_RS_rsrp_per_ue_dBm),

            effective_CSI_RS_sinr_ue_to_cell_dB = _legacy_format_npz(self.r_effective_CSI_RS_sinr_ue_to_cell_dB),
            # CSI_RS_usefulPower_ue_dBm = _legacy_format_npz(self.r_CSI_RS_usefulPower_ue_dBm),
            # CSI_RS_interfPower_ue_dBm = _legacy_format_npz(self.r_CSI_RS_interfPower_ue_dBm),

            # ---- UE counts per cell and beam ----
            ues_per_cell_based_on_CRS = _legacy_format_npz(self.r_ues_per_cell_based_on_CRS),
            ues_per_cell = _legacy_format_npz(self.r_ues_per_cell_based_on_SSB),
            ues_per_SSB_beam = _legacy_format_npz(self.r_ues_per_SSB_beam),
            ues_per_CSI_RS_beam = _legacy_format_npz(self.r_ues_per_CSI_RS_beam),

            # ---- Cell and beam activity ----
            cell_activity_per_ue = _legacy_format_npz(self.r_cell_activity_per_ue),
            SSB_beam_activity_per_ue = _legacy_format_npz(self.r_SSB_beam_activity_per_ue),
            CSI_RS_beam_activity_per_ue = _legacy_format_npz(self.r_CSI_RS_beam_activity_per_ue),
            PRB_usage_per_cell = _legacy_format_npz(self.r_PRB_usage_per_cell),

            # ---- Base station radio-unit state ----
            cell_radio_unit_statuses = _legacy_format_npz(self.r_cell_radio_unit_statuses),

            # ---- Throughput and Power Consumption ----
            ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps = _legacy_format_npz(self.r_ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps),
            ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps = _legacy_format_npz(self.r_ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps),
            cell_throughput_Mbps = _legacy_format_npz(self.r_cell_throughput_Mbps),
            carrier_throughput_Mbps = _legacy_format_npz(self.r_carrier_throughput_Mbps),
            ue_throughput_per_carrier_Mbps = _legacy_format_npz(self.r_ue_throughput_per_carrier_Mbps),
            power_consumption_per_radio_unit_incl_bbu_kW = _legacy_format_npz(self.r_power_consumption_per_radio_unit_incl_bbu_kW),
            power_consumption_per_radio_unit_kW = _legacy_format_npz(self.r_power_consumption_per_radio_unit_kW),
            network_power_consumption_per_radioType_kW = _legacy_format_npz(self.r_network_power_consumption_per_radioType_kW),
            total_network_power_consumption_kW = _legacy_format_npz(self.r_total_network_power_consumption_kW),
        )

    @staticmethod
    def _infer_parquet_type(value: Optional[Any], name: str) -> Optional[pa.DataType]:
        """
        Tries to obtain the Parquet type of a given value.

        Args:
            value (Optional[Any]): The value to calculate the type for.
            name (str): Just for logging, message will display this column name.

        Returns:
            Optional[Union[pa.DataType, Tuple[pa.DataType, int]]]: ``None`` if the type could not be inferred, the inferred Parquet data type, or a tuple containing the inferred Parquet data type and the number of additional columns to allocate otherwise.
        """
        if value is None:
            # If value is None, skip this variable: we cannot infer its type
            error(f"Saveable variable '{name}' returned None value: not found. Variable won't be saved.")
            return None
        elif isinstance(value, bool):
            # Assuming type for '{name}' as bool.
            return pa.bool_()
        elif isinstance(value, float):
            # Assuming type for '{name}' as float64.
            return pa.float64()
        elif isinstance(value, int):
            # Assuming type for '{name}' as int64.
            return pa.int64()
        elif isinstance(value, str):
            # Assuming type for '{name}' as string.
            return pa.string()
        elif isinstance(value, bytes):
            # Assuming type for '{name}' as binary.
            return pa.binary()
        elif len(np.shape(value)) > 0:
            if len(value) <= 0:
                error(f"Saveable variable '{name}' is an empty list: cannot infer type. Variable won't be saved.")
                return None
            first_elem = value[0]
            shape = np.shape(first_elem)
            dimensions = len(shape)

            if dimensions == 0:
                # debug(f"Saveable variable '{name}' is a list of scalar ({type(first_elem)}) values. Trying to infer type for list elements...")
                # Saveable variable '{name}' is a list of scalar ({type(first_elem)}) values. Trying to infer type for list elements...
                value_type = Episode_Performance._infer_parquet_type(first_elem, f"{name} (list element)")
                if value_type is not None:
                    return pa.list_(value_type)
                else:
                    error(f"Could not infer Parquet type for saveable variable '{name}' list elements. Variable won't be saved.")
                    return None
            if dimensions == 1:
                # debug(f"Saveable variable '{name}' is a nested list. Additional columns will be allocated. Trying to infer type for list elements...")
                # Saveable variable '{name}' is a nested list. Additional columns will be allocated. Trying to infer type for list elements...
                sub_type = Episode_Performance._infer_parquet_type(first_elem[0], f"{name} (list element)")
                if sub_type is not None:
                    return (pa.list_(sub_type), shape[0])
                else:
                    error(f"Could not infer Parquet type for saveable variable '{name}' list elements. Variable won't be saved.")
                    return None
            else:
                error(f"Saveable variable '{name}' has more than 2 dimensions. Variable won't be saved.")
                return None
        elif isinstance(value, dict):
            # TODO: Dicts are curently not supported. eventually, we should create additional columns for each key-value pair.
            error(f"Saveable variable '{name}' is a dict: not supported yet. Variable won't be saved.")
            return None
        else:
            # As a fallback: try to infer from numpy dtype
            try:
                inferred_type = pa.from_numpy_dtype(np.dtype(type(value)))
                # debug(f"Inferred type for '{name}' as {inferred_type} from value.")
                return inferred_type
            except ArrowNotImplementedError as e:
                error(f"Could not infer Parquet type for saveable variable '{name}'. Variable won't be saved.\nValue type: {str(type(value))}\nNumpy type: {str(np.dtype(type(value)))}\nMessage: {str(e)}")
                return None

    @staticmethod
    def _infer_exploded_parquet_type(var_type: Any, full_var_name: str) -> Optional[pa.DataType]:
        """
        Resolves the scalar type to be used when exporting exploded entity rows.
        """
        if isinstance(var_type, tuple):
            error(f"Saveable variable '{full_var_name}' has nested data. Exploded export supports only 1D arrays. Variable won't be saved.")
            return None
        if isinstance(var_type, pa.DataType) and pa.types.is_list(var_type):
            return var_type.value_type
        if isinstance(var_type, pa.DataType):
            return var_type
        error(f"Could not infer exploded Parquet type for saveable variable '{full_var_name}'. Variable won't be saved.")
        return None

    def _save_episodes_performance_parquet(
            self,
            dataset_dir_path: str,
            compression: Optional[str] = 'zip',
            use_partition_cols: bool = True,
            rm_uncompressed: bool = True
    ):
        """
        Stores the episodes performance as Parquet, one dataset per Saveable label.

        Each label produces ``<dataset_dir_path>_<label>``: a single ``.parquet`` file, or — when
        ``use_partition_cols`` is set — a directory partitioned by ``episode_idx`` / ``step_idx``.

        Args:
            dataset_dir_path: Base path for the datasets; the label is appended to it.
            compression: ``zip`` packs the dataset into a ``.gof`` archive, ``none`` leaves it as is.
                Only applies to a *partitioned* dataset — an unpartitioned one is already a single
                snappy-compressed file, so nothing is packed.
            use_partition_cols: Whether to split the dataset into per-episode/step Parquet files.
            rm_uncompressed: If True, removes the partition directory once it has been packed.
        """
        verbose(f'Exporting into Parquet. {compression=}, {use_partition_cols=}, {rm_uncompressed=}')

        parquet_schema: Dict[str, List[pa.Field]] = {
            'generic': [
                pa.field('episode_idx', pa.uint64()),
                pa.field('step_idx', pa.uint64()),
                # ---- CRS (Channel State Reporting Reference Signal) ----
                pa.field('best_serving_cell_ID_per_ue_based_on_CRS', pa.list_(pa.int64())),
                pa.field('best_serving_CRS_distance_3d_m', pa.list_(pa.float64())),
                pa.field('best_serving_CRS_coupling_gain_dB', pa.list_(pa.float64())),
                pa.field('best_serving_CRS_re_noise_per_ue_dBm', pa.list_(pa.float64())),
                pa.field('best_serving_CRS_rsrp_per_ue_dBm', pa.list_(pa.float64())),
                pa.field('CRS_RSRP_no_fast_fading_ue_to_cell_dBm', pa.list_(pa.float64())),

                pa.field('CRS_sinr_ue_to_cell_dB', pa.list_(pa.float64())),
                pa.field('CRS_usefulPower_ue_dBm', pa.list_(pa.float64())),
                pa.field('CRS_interfPower_ue_dBm', pa.list_(pa.float64())),

                # ---- SSB (Synchronization Signal Block) ----
                pa.field('best_serving_cell_ID_per_ue_based_on_SSB', pa.list_(pa.int64())),
                pa.field('best_serving_SSB_per_ue', pa.list_(pa.int64())),
                pa.field('best_serving_SSB_distance_3d_m', pa.list_(pa.float64())),
                pa.field('best_serving_SSB_coupling_gain_dB', pa.list_(pa.float64())),
                pa.field('best_serving_SSB_re_noise_per_ue_dBm', pa.list_(pa.float64())),
                pa.field('best_serving_SSB_rsrp_per_ue_dBm', pa.list_(pa.float64())),
                pa.field('SSB_RSRP_no_fast_fading_ue_to_cell_dBm', pa.list_(pa.float64())),

                pa.field('SSB_sinr_ue_to_cell_dB', pa.list_(pa.float64())),
                pa.field('SSB_usefulPower_ue_dBm', pa.list_(pa.float64())),
                pa.field('SSB_interfPower_ue_dBm', pa.list_(pa.float64())),

                # ---- CSI-RS (Channel State Information Reference Signal) ----
                pa.field('best_serving_CSI_RS_per_ue', pa.list_(pa.int64())),
                pa.field('best_serving_CSI_RS_rsrp_per_ue_dBm', pa.list_(pa.float64())),
                pa.field('effective_CSI_RS_sinr_ue_to_cell_dB', pa.list_(pa.float64())),
                # pa.field('CSI_RS_usefulPower_ue_dBm', pa.list_(pa.list_(pa.float64()))),
                # pa.field('CSI_RS_interfPower_ue_dBm', pa.list_(pa.list_(pa.float64()))),

                # ---- UE counts per cell and beam ----
                pa.field('ues_per_cell_based_on_CRS', pa.list_(pa.int64())),
                pa.field('ues_per_cell', pa.list_(pa.int64())),
                pa.field('ues_per_SSB_beam', pa.list_(pa.int64())),
                pa.field('ues_per_CSI_RS_beam', pa.list_(pa.int64())),

                # ---- Cell and beam activity ----
                pa.field('cell_activity_per_ue', pa.list_(pa.int64())),
                pa.field('SSB_beam_activity_per_ue', pa.list_(pa.int64())),
                pa.field('CSI_RS_beam_activity_per_ue', pa.list_(pa.int64())),

                # ---- Throughput and Power Consumption ----
                pa.field('ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps', pa.list_(pa.float64())),
                pa.field('ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps', pa.list_(pa.float64())),
                pa.field('cell_throughput_Mbps', pa.list_(pa.float64())),
                pa.field('carrier_throughput_Mbps', pa.list_(pa.float64())),
                pa.field('ue_throughput_per_carrier_Mbps', pa.list_(pa.float64())),

                pa.field('power_consumption_per_radio_unit_incl_bbu_kW', pa.list_(pa.float64())),
                pa.field('network_power_consumption_per_radioType_kW', pa.list_(pa.float64())),
                pa.field('total_network_power_consumption_kW', pa.list_(pa.float64())),
            ]
        }

        # Append the saveables to the schema
        saveable_fields: Dict[str, List[str]] = {}
        # Per-variable prefix for the expanded sub-columns of a 2-D exploded variable (keyed by the
        # full '<Class>.<var>' name). Empty string -> numeric suffix ('.0', '.1', ...).
        column_prefixes: Dict[str, str] = {}

        def ensure_label_initialized(label: str) -> None:
            """Lazily create the field list and hardcoded schema columns for a parquet label."""
            if label not in saveable_fields:
                saveable_fields[label] = []
            if label not in parquet_schema:
                parquet_schema[label] = [
                    pa.field('episode_idx', pa.uint64()),
                    pa.field('step_idx', pa.uint64()),
                ]
                if label != GENERIC_LABEL:
                    parquet_schema[label].append(pa.field(_entity_id_column(label), pa.uint64()))

        for saveable in _get_saveables():
            expose_variables = saveable.expose_variables()
            # debug(f'Processing saveable for Parquet schema: {saveable.name} ({len(expose_variables)} variables)')
            # Processing saveable for Parquet schema: {saveable.name} ({len(expose_variables)} variables)
            var_types = saveable.variable_types()
            # Per-variable label mapping: a single class may route different variables to
            # different parquets (e.g. ue_kpi and cell_kpi). Falls back to the class label.
            var_labels = saveable.expose_variable_labels()
            for var_name in expose_variables:
                if var_name in var_labels:
                    label = var_labels[var_name]
                else:
                    # The variable is exported but its target parquet was not declared in
                    # expose_variable_labels(): warn instead of silently dumping it into the class
                    # label (usually 'generic'), which is an easy mistake when adding a KPI.
                    label = saveable.label
                    warning(
                        f"Saveable variable '{saveable.name}.{var_name}' is listed in expose_variables() "
                        f"but has no entry in expose_variable_labels(); falling back to label '{label}'. "
                        f"Add '{var_name}' to {saveable.name}.expose_variable_labels() to route it explicitly."
                    )
                is_exploded_label = label != GENERIC_LABEL
                ensure_label_initialized(label)

                full_var_name = f"{saveable.name}.{var_name}"
                # Skip if already present
                if full_var_name in saveable_fields[label]: continue

                var_type = var_types.get(var_name)
                if var_type is None:
                    # If type is not specified, try to infer it from the value
                    var_value = saveable.get_output(var_name)
                    var_type = self._infer_parquet_type(var_value, full_var_name)
                    if var_type is None: continue # error already logged in _infer_parquet_type

                if is_exploded_label:
                    if isinstance(var_type, tuple):
                        # 2-D per-entity array (e.g. a UE×cell matrix): explode along the entity
                        # axis (one row per entity) and spread the second axis into scalar columns
                        # '<name>.<prefix>0 .. .<prefix>N' (prefix defaults to empty -> '.0', '.1').
                        # The schema gets the split columns, but saveable_fields keeps the *base*
                        # name so the writer can find the per-step data (keyed by base name) and
                        # split it per entity at write time.
                        list_type, columns_count = var_type
                        scalar_type = list_type.value_type if pa.types.is_list(list_type) else list_type
                        col_prefix = saveable.expose_variable_column_prefixes().get(var_name, '')
                        for index in range(columns_count):
                            parquet_schema[label].append(pa.field(f"{full_var_name}.{col_prefix}{index}", scalar_type))
                        saveable_fields[label].append(full_var_name)
                        column_prefixes[full_var_name] = col_prefix
                        continue
                    exploded_type = self._infer_exploded_parquet_type(var_type, full_var_name)
                    if exploded_type is None:
                        continue
                    parquet_schema[label].append(pa.field(full_var_name, exploded_type))
                    saveable_fields[label].append(full_var_name)
                    continue

                if isinstance(var_type, tuple):
                    # debug(f"Variable '{full_var_name}' is a nested list. Allocating {var_type[1]} additional columns.")
                    # Variable '{full_var_name}' is a nested list. Allocating {var_type[1]} additional columns.
                    for index in range(var_type[1]):
                        indexed_var_name = f"{full_var_name}.{index}"
                        parquet_schema[label].append(pa.field(indexed_var_name, var_type[0]))
                        saveable_fields[label].append(indexed_var_name)
                else:
                    # debug('Adding to Parquet schema field:', full_var_name, 'of type', var_type)
                    # Adding to Parquet schema field: {full_var_name} of type {var_type}
                    parquet_schema[label].append(pa.field(full_var_name, var_type))
                    saveable_fields[label].append(full_var_name)

        for label in saveable_fields:
            verbose(f'Processing saveable data for label {label} ({len(saveable_fields[label])} fields)')
            self._save_saveables_data_parquet(
                dataset_dir_path + '_' + label,
                label,
                parquet_schema[label],
                saveable_fields[label],
                compression,
                use_partition_cols,
                rm_uncompressed,
                column_prefixes,
            )

    @staticmethod
    def _normalize_exploded_array(data: Any, field_name: str, episode_index: int, step_index: int) -> Optional[npt.NDArray[np.object_]]:
        """
        Validates and normalizes entity-aligned saveable data for exploded export.
        """
        if data is None:
            error(f"Saveable field '{field_name}' returned None at episode={episode_index}, step={step_index}. Field will be skipped for this step.")
            return None

        shape = np.shape(data)
        if len(shape) not in (1, 2):
            error(f"Saveable field '{field_name}' has invalid shape {shape} at episode={episode_index}, step={step_index}. Expected a 1D (one value per entity) or 2D (one row per entity) array.")
            return None

        # 1-D -> one scalar per entity; 2-D -> one row per entity (spread into '.0 .. .N' columns
        # at write time). In both cases the first axis is the entity axis, so len() is the entity count.
        return np.asarray(data, dtype=object)

    def _save_saveables_data_parquet(
            self,
            dataset_dir_path: str,
            label: str,
            parquet_schema: pa.Schema,
            saveable_fields: List[str],
            compression: Optional[str] = 'zip',
            use_partition_cols: bool = True,
            rm_uncompressed: bool = True,
            column_prefixes: Optional[Dict[str, str]] = None
    ):
        # Prefix inserted before the index of an expanded 2-D column ('.<prefix>0', ...), keyed by
        # the field's base name. Empty/absent -> numeric suffix.
        column_prefixes = column_prefixes or {}
        dataset_name = dataset_dir_path.split(os.sep)[-1]

        # Exploded files (every label except 'generic') are indexed by one entity per row;
        # the entity-id column name is derived from the label (e.g. 'bbu_kpi' -> 'bbu_id').
        explode_entity_key = _entity_id_column(label) if label != GENERIC_LABEL else None

        # Don't build an exploded file that would have no data columns — either no saveable field
        # is routed to this label, or all of them are removed by the config 'fields' filter. Such a
        # file would contain only index columns, so skip it entirely. This runs BEFORE any directory
        # is created, so a skipped label leaves nothing on disk (no folder, no schema.md).
        if explode_entity_key is not None:
            # A field survives the filter if it is selected by name, or (for a 2-D variable) if any
            # of its expanded '.<int>' columns is selected — see _is_field_selected.
            effective_fields = saveable_fields if self.fields is None else [
                f for f in saveable_fields
                if f in self.fields or any(s.startswith(f + '.') for s in self.fields)
            ]
            if not effective_fields:
                verbose(f"Skipping '{dataset_name}' parquet: no data columns are routed to this label.")
                return

        # create the upper directory
        os.makedirs(os.path.join(dataset_dir_path, '..'), exist_ok=True)

        saveables_schema_file = os.path.realpath(os.path.join(dataset_dir_path, '..', f'{dataset_name}.schema.md'))

        # Apply the 'fields' filter BEFORE writing the schema file, so that the .schema.md
        # describes exactly the columns written to the Parquet file.
        if self.fields is not None:
            # Filter schema to keep only selected fields. Always keep the index columns:
            # episode/step plus this file's entity-id column (if it is an exploded file).
            kept_index_columns = {'episode_idx', 'step_idx'}
            if explode_entity_key is not None:
                kept_index_columns.add(explode_entity_key)
            parquet_schema = [
                field for field in parquet_schema
                if field.name in kept_index_columns
                or self._is_field_selected(field.name)
            ]
            # debug(f'Filtered Parquet schema to keep only selected fields: {self.fields}')


        # Index columns are episode/step (+ the entity-id for exploded files). The remaining schema
        # fields are the data columns; a single 2-D saveable expands into several of them, so count
        # them from the schema rather than from len(saveable_fields) (which holds base names).
        index_columns_count = 2 + (1 if explode_entity_key is not None else 0)
        hardcoded_fields_count = index_columns_count
        saveable_columns_count = len(parquet_schema) - index_columns_count

        with open(saveables_schema_file, 'w') as f:
            from datetime import datetime
            f.write(f"**Saveables schema file generated on {datetime.now().isoformat()}**\n\n")
            f.write(f"**Total schema fields: {len(parquet_schema)}**\n\n")
            f.write(f"**Hardcoded fields: {hardcoded_fields_count}**\n\n")
            f.write(f"**Saveable fields: {saveable_columns_count}**\n\n")
            f.write(f"|Field name|Field Type|\n")
            f.write(f"|---|---|\n")
            for field in parquet_schema:
                 f.write(f"|`{field.name}`|`{str(field.type)}`|\n")

        # debug(f'Total Saveable schema fields: {len(saveable_fields)}: {", ".join(saveable_fields)}')
        

        episodes_count = self.number_of_episodes
        steps_count = self.number_of_steps
        records = []

        # Tracks exploded fields already reported as length-mismatched, so the warning is
        # emitted once per field instead of once per (episode, step).
        length_mismatch_warned: set = set()

        for episode_index in range(episodes_count):
            verbose(f'Preparing episode {episode_index + 1}/{episodes_count} for Parquet export...')
            
            for step_index in range(steps_count):
                if label == GENERIC_LABEL:
                    record = {
                        'episode_idx': episode_index,
                        'step_idx': step_index,

                        # ---- CRS (Channel State Reporting Reference Signal) ----
                        'best_serving_cell_ID_per_ue_based_on_CRS': self.r_best_serving_cell_ID_per_ue_based_on_CRS[episode_index][step_index].flatten(),
                        'best_serving_CRS_distance_3d_m': self.r_best_serving_CRS_distance_3d_m[episode_index][step_index].flatten(),
                        'best_serving_CRS_coupling_gain_dB': self.r_best_serving_CRS_coupling_gain_dB[episode_index][step_index].flatten(),
                        'best_serving_CRS_re_noise_per_ue_dBm': self.r_best_serving_CRS_re_noise_per_ue_dBm[episode_index][step_index].flatten(),
                        'best_serving_CRS_rsrp_per_ue_dBm': self.r_best_serving_CRS_rsrp_per_ue_dBm[episode_index][step_index].flatten(),
                        'CRS_RSRP_no_fast_fading_ue_to_cell_dBm': self.r_CRS_RSRP_no_fast_fading_ue_to_cell_dBm[episode_index][step_index].flatten(),

                        'CRS_sinr_ue_to_cell_dB': self.r_CRS_sinr_ue_to_cell_dB[episode_index][step_index].flatten(),
                        'CRS_usefulPower_ue_dBm': self.r_CRS_usefulPower_ue_dBm[episode_index][step_index].flatten(),
                        'CRS_interfPower_ue_dBm': self.r_CRS_interfPower_ue_dBm[episode_index][step_index].flatten(),

                        # ---- SSB (Synchronization Signal Block) ----
                        'best_serving_cell_ID_per_ue_based_on_SSB': self.r_best_serving_cell_ID_per_ue_based_on_SSB[episode_index][step_index].flatten(),
                        'best_serving_SSB_per_ue': self.r_best_serving_SSB_per_ue[episode_index][step_index].flatten(),
                        'best_serving_SSB_distance_3d_m': self.r_best_serving_SSB_distance_3d_m[episode_index][step_index].flatten(),
                        'best_serving_SSB_coupling_gain_dB': self.r_best_serving_SSB_coupling_gain_dB[episode_index][step_index].flatten(),
                        'best_serving_SSB_re_noise_per_ue_dBm': self.r_best_serving_SSB_re_noise_per_ue_dBm[episode_index][step_index].flatten(),
                        'best_serving_SSB_rsrp_per_ue_dBm': self.r_best_serving_SSB_rsrp_per_ue_dBm[episode_index][step_index].flatten(),
                        'SSB_RSRP_no_fast_fading_ue_to_cell_dBm': self.r_SSB_RSRP_no_fast_fading_ue_to_cell_dBm[episode_index][step_index].flatten(),

                        'SSB_sinr_ue_to_cell_dB': self.r_SSB_sinr_ue_to_cell_dB[episode_index][step_index].flatten(),
                        'SSB_usefulPower_ue_dBm': self.r_SSB_usefulPower_ue_dBm[episode_index][step_index].flatten(),
                        'SSB_interfPower_ue_dBm': self.r_SSB_interfPower_ue_dBm[episode_index][step_index].flatten(),

                        # ---- CSI-RS (Channel State Information Reference Signal) ----
                        'best_serving_CSI_RS_per_ue': self.r_best_serving_CSI_RS_per_ue[episode_index][step_index].flatten(),
                        'best_serving_CSI_RS_rsrp_per_ue_dBm': self.r_best_serving_CSI_RS_rsrp_per_ue_dBm[episode_index][step_index].flatten(),
                        'effective_CSI_RS_sinr_ue_to_cell_dB': self.r_effective_CSI_RS_sinr_ue_to_cell_dB[episode_index][step_index].flatten(),
                        # 'CSI_RS_usefulPower_ue_dBm': self.r_CSI_RS_usefulPower_ue_dBm[episode_index][step_index].flatten(),
                        # 'CSI_RS_interfPower_ue_dBm': self.r_CSI_RS_interfPower_ue_dBm[episode_index][step_index].flatten(),

                        # ---- UE counts per cell and beam ----
                        'ues_per_cell_based_on_CRS': self.r_ues_per_cell_based_on_CRS[episode_index][step_index].flatten(),
                        'ues_per_cell': self.r_ues_per_cell_based_on_SSB[episode_index][step_index].flatten(),
                        'ues_per_SSB_beam': self.r_ues_per_SSB_beam[episode_index][step_index].flatten(),
                        'ues_per_CSI_RS_beam': self.r_ues_per_CSI_RS_beam[episode_index][step_index].flatten(),

                        # ---- Cell and beam activity ----
                        'cell_activity_per_ue': self.r_cell_activity_per_ue[episode_index][step_index].flatten(),
                        'SSB_beam_activity_per_ue': self.r_SSB_beam_activity_per_ue[episode_index][step_index].flatten(),
                        'CSI_RS_beam_activity_per_ue': self.r_CSI_RS_beam_activity_per_ue[episode_index][step_index].flatten(),
                        'PRB_usage_per_cell': self.r_PRB_usage_per_cell[episode_index][step_index].flatten(),

                        # ---- Throughput and Power Consumption ----
                        'ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps': self.r_ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps[episode_index][step_index].flatten(),
                        'ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps': self.r_ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps[episode_index][step_index].flatten(),
                        'cell_throughput_Mbps': self.r_cell_throughput_Mbps[episode_index][step_index].flatten(),
                        'carrier_throughput_Mbps': self.r_carrier_throughput_Mbps[episode_index][step_index].flatten(),
                        'ue_throughput_per_carrier_Mbps': self.r_ue_throughput_per_carrier_Mbps[episode_index][step_index].flatten(),

                        'power_consumption_per_radio_unit_incl_bbu_kW': self.r_power_consumption_per_radio_unit_incl_bbu_kW[episode_index][step_index].flatten(),
                        'power_consumption_per_radio_unit_kW': self.r_power_consumption_per_radio_unit_kW[episode_index][step_index].flatten(),
                        'network_power_consumption_per_radioType_kW': self.r_network_power_consumption_per_radioType_kW[episode_index][step_index].flatten(),
                        'total_network_power_consumption_kW': self.r_total_network_power_consumption_kW[episode_index][step_index].flatten(),
                    }
                else:
                    record = {
                        'episode_idx': episode_index,
                        'step_idx': step_index,
                    }
                saveable_data_for_episode = self.saveables_data[episode_index]

                if explode_entity_key is not None:
                    # First pass: normalize every field to a per-entity array along its first axis
                    # (1-D -> one scalar per entity; 2-D -> one row per entity, spread into columns).
                    normalized_fields: Dict[str, npt.NDArray[np.object_]] = {}
                    for saveable_field in saveable_fields:
                        if saveable_field not in saveable_data_for_episode:
                            error(f"Saveable field '{saveable_field}' not found for exploded export at episode={episode_index}, step={step_index}.")
                            continue

                        try:
                            data = saveable_data_for_episode[saveable_field][step_index]
                        except (IndexError, TypeError, KeyError) as ex:
                            error(
                                f"Saveable field '{saveable_field}' cannot be accessed at episode={episode_index}, step={step_index}. "
                                f"Reason: {str(ex)}"
                            )
                            continue

                        normalized = self._normalize_exploded_array(data, saveable_field, episode_index, step_index)
                        if normalized is None:
                            continue

                        normalized_fields[saveable_field] = normalized

                    if not normalized_fields:
                        continue

                    # An exploded file is indexed by a single entity (one row per UE or per cell),
                    # so every column must share that entity count. Pick the length the majority of
                    # fields agree on (the true entity count) and drop the rest: this stops a stray
                    # mis-labeled field (e.g. a per-cell or per-beam array routed to ue_kpi) from
                    # hijacking the row count and silently NaN-ing every correct per-entity column.
                    expected_length = Counter(len(arr) for arr in normalized_fields.values()).most_common(1)[0][0]

                    valid_field_data: Dict[str, npt.NDArray[np.object_]] = {}
                    for saveable_field, normalized in normalized_fields.items():
                        if len(normalized) == expected_length:
                            valid_field_data[saveable_field] = normalized
                        elif saveable_field not in length_mismatch_warned:
                            length_mismatch_warned.add(saveable_field)
                            warning(
                                f"Saveable field '{saveable_field}' has first-axis length {len(normalized)} but "
                                f"'{dataset_name}' is indexed per {explode_entity_key} and expects {expected_length}. "
                                f"Its first axis is not the per-{explode_entity_key} axis, so it is dropped from this "
                                f"file. Fix its label / expose_variable_labels() to route it to the right parquet."
                            )

                    for entity_idx in range(expected_length):
                        exploded_record = dict(record)
                        exploded_record[explode_entity_key] = entity_idx

                        for saveable_field in saveable_fields:
                            if saveable_field not in valid_field_data:
                                # Absent/dropped field: leave its column(s) unset; pa.Table.from_pylist
                                # fills the schema columns it doesn't find with nulls.
                                continue
                            entity_value = valid_field_data[saveable_field][entity_idx]
                            if len(np.shape(entity_value)) > 0:
                                # 2-D field: spread this entity's row into '<field>.<prefix>0 .. .N' columns.
                                prefix = column_prefixes.get(saveable_field, '')
                                for col_idx, col_value in enumerate(entity_value):
                                    exploded_record[f"{saveable_field}.{prefix}{col_idx}"] = col_value
                            else:
                                # 1-D field: one scalar per entity.
                                exploded_record[saveable_field] = entity_value

                        if self.fields is not None:
                            exploded_record = {
                                key: val for key, val in exploded_record.items()
                                if key in ('episode_idx', 'step_idx') or key == explode_entity_key or self._is_field_selected(key)
                            }

                        records.append(exploded_record)
                    continue

                for saveable_field in saveable_fields:
                    if saveable_field in saveable_data_for_episode:
                        data = saveable_data_for_episode[saveable_field][step_index]
                        if len(np.shape(data)) > 1:
                            # if the data has more than 1 dimension, allocate additional columns with a `.{col}` suffix
                            for col_idx, col_data in enumerate(data):
                                record[f"{saveable_field}.{col_idx}"] = col_data
                        else:
                            # otherwise, just save the data
                            record[saveable_field] = data
                    else:
                        record[saveable_field] = None

                if self.fields is not None:
                    # Filter record to keep only selected fields
                    record = {key: val for key, val in record.items() if key in ('episode_idx', 'step_idx') or self._is_field_selected(key)}

                records.append(record)
        table = pa.Table.from_pylist(records, schema=pa.schema(parquet_schema))

        if os.path.exists(dataset_dir_path):
            debug("Overriding existing dataset...")
            shutil.rmtree(dataset_dir_path)

        # Write the Arrow Table to a Parquet file
        if use_partition_cols:
            pq.write_to_dataset(
                table,
                root_path=dataset_dir_path,
                # Partitioning by episode_idx and step_idx
                partition_cols=['episode_idx', 'step_idx'],
                compression='snappy',  # A good balance of speed and compression
            )
        else:
            pq.write_table(
                table,
                f'{dataset_dir_path}.parquet',
                compression='snappy',  # A good balance of speed and compression
            )

        # GOF compression only makes sense if the data is partitioned, otherwise we already have a single compressed Parquet file
        if compression == 'zip' and use_partition_cols:
            # Create a gof file with the Parquet dataset
            pack_output(dataset_dir_path, rm_uncompressed)
