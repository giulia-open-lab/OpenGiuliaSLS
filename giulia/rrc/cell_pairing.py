#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep  4 17:29:51 2025

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

### This is the MAIN of Guilia

import time
import hashlib
import os

from typing import Tuple

import numpy as np
import pandas as pd
from giulia.config.sim_config import Simulation_Config

from giulia.playground.sites import Site_siteDeployment, Site_uePlaygroundDeployment
from giulia.playground.hotspots import Hotspot

from giulia.bs.bs_deployments import Network
from giulia.tools.tools import free_memory, log_calculations_time, dataframe_hash_signature
from giulia.outputs.saveable import Saveable

from giulia.logger import parse_log_level, set_log_level, get_log_level
    

###############################################################################
from giulia.presets._g_pairing_preset_new import Giulia_Pairing


class PairingCell():

    def __init__(self,
                 simulation_config_obj: Simulation_Config,
                 site_deployment_obj: Site_siteDeployment,
                 ue_playground_deployment_obj: Site_uePlaygroundDeployment,
                 ue_hotspot_deployment_obj: Hotspot,
                 network_deployment_obj: Network,
                 pairing_model_type: str = "per_capacity_cell__best_coverage_cell_most_reported"):

        # Initialize attributes
        self.simulation_config_obj: Simulation_Config = simulation_config_obj
        self.network_deployment_obj: Network = network_deployment_obj

        # Initialize attributes from giulia Simulation_Config object
        self.pairing_model_type = pairing_model_type
        self.preset: str = "GiuliaStd"
        # self.scenario_model: str = simulation_config_obj.scenario_playground_model
        self.ue_playground_model: str = simulation_config_obj.ue_playground_model
        self.ue_distribution: str = simulation_config_obj.ue_distribution
        self.ue_mobility: str|None = None
        self.link_direction: str = "downlink"
        self.wraparound: bool = simulation_config_obj.wraparound
        self.save_results: int = 0
        self.plots: bool = False
        self.number_of_episodes: int = 1
        self.regression: bool = False
        self.enable_GPU: bool = simulation_config_obj.enable_GPU

        # Validate pairing model type
        available_models_type = ["per_capacity_cell__best_coverage_cell_most_reported", "adjacency_neighbouring_matrix_cellLevel_FallBack_PerLayer"]
        assert self.pairing_model_type in available_models_type, f"Error! pairing_model_type {self.pairing_model_type} not admitted. Must be one of {available_models_type}."
        # Set pairing method
        self._set_pairing_method(self.pairing_model_type)

        #### Outputs
        self._hash_df_ep: str|None = None
        self.adjacency_neighbouring_matrix: np.ndarray|None = None
        self.pairing_CovCap_cell_matrix: np.ndarray|None = None
        self.capCell_paired_covCell: np.ndarray|None = None

        # Store deployment objects in a dictionary for later argument passing
        self.legacy_deployment_objects: dict = {
            "site_deployment_obj": site_deployment_obj,
            "ue_playground_deployment_obj": ue_playground_deployment_obj,
            "ue_hotspot_deployment_obj": ue_hotspot_deployment_obj,
            "network_deployment_obj": self.network_deployment_obj}

        return

    def process(self) -> None:

        # Start timing
        t_start = time.perf_counter()

        ############### Run Giulia for pairing computation ###############
        # Store current log level
        current_log_level = get_log_level()
        # Set log level to 0 to avoid unnecessary prints during Giulia execution
        set_log_level(parse_log_level(4))

        # Basic Sanity Check
        if not isinstance(self.network_deployment_obj.df_ep, pd.DataFrame): raise TypeError("df_ep must be a pandas DataFrame")

        # Compute hash signature univocally identifying this pairing configuration, used to
        # store/retrieve pre-computed pairing results for the same configuration.
        # The fingerprint combines the df_ep content with every other input that affects the
        # pairing result (see _compute_pairing_cache_key), so configurations that share df_ep
        # but differ in e.g. pairing_model_type do NOT collide on the same cache entry.
        self._hash_df_ep = self._compute_pairing_cache_key()

        # Case 1: All cells are coverage type → Pairing not required
        if self.network_deployment_obj.df_ep["cell_role"].eq("coverage").all():
            pairing_CovCap_cell_matrix = np.full((self.network_deployment_obj.df_ep.shape[0], self.network_deployment_obj.df_ep.shape[0]),
                                                      False, dtype=bool)
            capCell_paired_covCell = np.array([])
            adjacency_neighbouring_matrix = None

        # Case 2: Pairing required
        ## Try to load pre-computed pairing results from disk, if not found, compute them
        ## This allows to compute only once the pairing results for a given deployment configuration, and re-use them in future runs
        else:
            # The recompute path below builds a Giulia_Pairing, which switches the *global*
            # Saveable collection flag off (so its own throw-away RSRP/path-loss objects don't
            # replace the caller's ones in the registry) and then "restores" it from its own
            # Simulation_Config — always built with enable_saveable=False, so it can only ever
            # restore "off". Snapshot the caller's setting here and put it back in the finally:
            # otherwise every Saveable the main simulation creates afterwards (all the KPI
            # objects, built later in Giulia.configure()) silently skips registration and the
            # run exports no KPI parquet at all.
            saveable_was_enabled = Saveable.is_enabled()
            try:
                # Try to load pre-computed pairing results from disk
                pairing_CovCap_cell_matrix, capCell_paired_covCell, adjacency_neighbouring_matrix = self._load_pairing_results()
                # Restore the adjacency matrix side-effect output (None for the per_capacity_cell model),
                # mirroring the state _pairing_method leaves behind on the compute path.
                self.adjacency_neighbouring_matrix = adjacency_neighbouring_matrix

            except (FileNotFoundError, ValueError, EOFError, OSError):
                # FileNotFoundError -> no cache yet; (ValueError, EOFError, OSError) -> partially
                # written/corrupted .npy files (e.g. process killed mid-save). In all cases recompute.
                # Create Giulia_Pairing instance
                giulia_pairing, RSRP_dBm_all_realization = self._init_and_run_giulia_pairing(self.preset, self.ue_playground_model, self.ue_distribution,
                                                                self.ue_mobility, self.link_direction, self.wraparound, self.save_results,
                                                                self.plots, self.number_of_episodes, self.regression, self.enable_GPU,
                                                                **self.legacy_deployment_objects)

                # Process pairing per capacity cell
                pairing_CovCap_cell_matrix, capCell_paired_covCell = self._pairing_method(giulia_pairing=giulia_pairing,
                                                                                        RSRP_dBm_all_realization=RSRP_dBm_all_realization)

                # Save pairing results to disk
                self._save_pairing_results(pairing_CovCap_cell_matrix, capCell_paired_covCell)

                # Free memory
                del giulia_pairing


            finally:
                # Undo Giulia_Pairing's global side effect (see the snapshot above). Done here
                # rather than inside Giulia_Pairing so it also covers the cache-hit path and any
                # failure in between: pairing must leave the flag exactly as it found it.
                if saveable_was_enabled:
                    Saveable.enable()
                else:
                    Saveable.disable()
                free_memory()
        ###################################################################

        # Restore log level
        set_log_level(current_log_level)

        # Store results
        self.set_pairing_CovCap_cell_matrix(pairing_CovCap_cell_matrix)
        self.set_capCell_paired_covCell(capCell_paired_covCell)

        # Set pairing infomration in dataframe engneering parameters df_ep
        self._write_pairing_to_df_ep()

        # Log calculations time
        log_calculations_time('Pairing Cells', t_start)

        # Return pairing results
        return


    ############### Setters ###############
    def set_pairing_CovCap_cell_matrix(self, pairing_CovCap_cell_matrix: np.ndarray):
        self.pairing_CovCap_cell_matrix = pairing_CovCap_cell_matrix

    def set_capCell_paired_covCell(self, capCell_paired_covCell: np.ndarray):
        self.capCell_paired_covCell = capCell_paired_covCell


    ############### Getters ###############
    def get_pairing_CovCap_cell_matrix(self) -> np.ndarray|None:
        return self.pairing_CovCap_cell_matrix

    def get_capCell_paired_covCell(self) -> np.ndarray|None:
        return self.capCell_paired_covCell

    ############### Internal: save/load pairing results ###############
    def _compute_pairing_cache_key(self) -> str:
        """Deterministic fingerprint identifying a pairing configuration for caching.

        Combines the df_ep content fingerprint with every other input that influences the
        pairing result. Two runs map to the same cache entry only if BOTH the df_ep content
        and all these parameters match, preventing stale/wrong cache hits when e.g. the
        pairing model or the UE distribution change while df_ep stays the same.
        """
        if not isinstance(self.network_deployment_obj.df_ep, pd.DataFrame):
            raise TypeError("df_ep must be a pandas DataFrame")
        df_signature = dataframe_hash_signature(self.network_deployment_obj.df_ep)

        # Result-affecting parameters that are NOT part of df_ep. Keep this in sync with the
        # inputs actually consumed by _init_and_run_giulia_pairing / _pairing_method.
        config_signature = repr((
            self.pairing_model_type,
            self.ue_playground_model,
            self.ue_distribution,
            self.wraparound,
        ))
        config_hash = hashlib.sha256(config_signature.encode("utf-8")).hexdigest()#[:16]

        return f"{df_signature}_{config_hash}"

    def _save_pairing_results(self, pairing_CovCap_cell_matrix: np.ndarray, capCell_paired_covCell: np.ndarray) -> None:
        # Save pairing results to disk
        cache_dir = f"./data/pairing_sym_results/{self._hash_df_ep}"
        os.makedirs(cache_dir, exist_ok=True)
        # Save the pairing results as .npy files
        np.save(f"{cache_dir}/pairing_CovCap_cell_matrix.npy", pairing_CovCap_cell_matrix)
        np.save(f"{cache_dir}/capCell_paired_covCell.npy", capCell_paired_covCell)
        # The adjacency-matrix model sets self.adjacency_neighbouring_matrix as a side effect of
        # _pairing_method and it is read back externally (e.g. by the UnRL workflow). Persist it so
        # a cache hit restores the same object state as a fresh computation. It stays None for the
        # per_capacity_cell model, in which case there is nothing to save.
        if self.adjacency_neighbouring_matrix is not None:
            np.save(f"{cache_dir}/adjacency_neighbouring_matrix.npy", self.adjacency_neighbouring_matrix)
        return

    def _load_pairing_results(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray|None]:
        # Load pairing results from disk
        cache_dir = f"./data/pairing_sym_results/{self._hash_df_ep}"
        pairing_CovCap_cell_matrix = np.load(f"{cache_dir}/pairing_CovCap_cell_matrix.npy")
        capCell_paired_covCell = np.load(f"{cache_dir}/capCell_paired_covCell.npy")
        # The adjacency matrix side-effect output is present only for the adjacency-matrix model;
        # return None otherwise. The caller is responsible for storing it on the instance.
        adjacency_path = f"{cache_dir}/adjacency_neighbouring_matrix.npy"
        adjacency_neighbouring_matrix = np.load(adjacency_path) if os.path.exists(adjacency_path) else None
        return pairing_CovCap_cell_matrix, capCell_paired_covCell, adjacency_neighbouring_matrix
    
    ############### Internal: setup / orchestration ###############
    def _set_pairing_method(self, pairing_model_type: str):
        if pairing_model_type == "per_capacity_cell__best_coverage_cell_most_reported":
            self._pairing_method = self._pair_by_reported_most_frequent

        elif pairing_model_type == "adjacency_neighbouring_matrix_cellLevel_FallBack_PerLayer":
             self._pairing_method = self._pair_by_reported_RSRP__adjacencyMatrixPerLayer

        else:
            raise NotImplementedError(f"Pairing model type {pairing_model_type} not implemented.")


    def _init_and_run_giulia_pairing(self, preset, ue_playground_model, ue_distribution, ue_mobility,
                link_direction, wraparound, save_results, plots, number_of_episodes, regression, enable_GPU, **kwargs) -> Tuple[Giulia_Pairing, np.ndarray]:

        ############### Set new Giulia for computing long-terms pairing caracteristic
        giulia_pairing = Giulia_Pairing(preset, ue_playground_model, ue_distribution, ue_mobility, link_direction, wraparound,
                                        save_results, plots, number_of_episodes, regression, enable_GPU,
                                        **kwargs)

        results_list_RSRP: list = []
        N_episode: int = 1 if ue_distribution== "grid" else 100
        for ep in range(N_episode):
            # Configure Giulia
            giulia_pairing.configure(ep, 1e6)

            assert not isinstance(giulia_pairing.shadowing_map_cell_obj, list), "shadowing_map_cell_obj has not been processed yet"
            # Set shadow fading to 0dB
            for key in giulia_pairing.shadowing_map_cell_obj.shadowing_maps_dB.keys():
                for m in range(len(giulia_pairing.shadowing_map_cell_obj.shadowing_maps_dB[key])):
                    giulia_pairing.shadowing_map_cell_obj.shadowing_maps_dB[key][m] =\
                        np.zeros_like(giulia_pairing.shadowing_map_cell_obj.shadowing_maps_dB[key][m])

            # Run
            giulia_pairing.run_simulation(0, 1)

            # Per-beam RSRP matrix in dBm, shape (N_ue, N_beams_total).
            # Columns are concatenated across all cells in cell-id order.
            RSRP_dBm = giulia_pairing.best_serving_cell_ID_per_ue_based_on_SSB_obj \
                        .RSRP_ue_to_cell_obj \
                        .RSRP_results_per_frequency_layer['all_freq']['RSRP_ue_to_cell_dBm']

            # Copy before mutating, otherwise we'd overwrite the array still held by giulia_pairing
            RSRP_dBm = RSRP_dBm.copy()

            # Store the RSRP data for later adjacency neighbouring matrix computation.
            results_list_RSRP.append(RSRP_dBm)


        # Eliminate the reference to the last RSRP_dBm from the loop to avoid mistakes
        del RSRP_dBm

        # Concatenate the per-episode RSRP arrays along the UE axis to get a single (N_ue_total, N_beams_total) array, where N_ue_total = N_ue_per_episode * N_episode.
        RSRP_dBm_all_realization = np.concatenate(results_list_RSRP, axis=0)

        # Floor invisible (UE, beam) pairs from -inf to the sentinel so max/comparisons stay finite
        RSRP_dBm_all_realization[RSRP_dBm_all_realization == -np.inf] = -200

        # Free memory
        free_memory()
        return giulia_pairing, RSRP_dBm_all_realization


    ############### Internal: pairing strategies ###############
    def _pair_by_reported_most_frequent(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each capacity cell, identify the UEs connected to it (by best reported beam/cell) and
        determine which coverage cell is most frequently their best-reported cell. Pair the capacity
        cell with that coverage cell.

        Parameters
        ----------
        **kwargs
            giulia_pairing : Giulia_Pairing
                The Giulia_Pairing instance holding the deployment and reported-RSRP data used for pairing.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            pairing_CovCap_cell_matrix : np.ndarray
                Symmetric (N_cell, N_cell) boolean matrix, True at paired coverage/capacity cell positions.
            capCell_paired_covCell : np.ndarray
                Integer array with the paired coverage-cell index for each capacity cell.
        """

        g: Giulia_Pairing|None = kwargs.get('giulia_pairing', None)
        assert g is not None, "giulia_pairing must be provided in kwargs for per-capacity-cell pairing method"

        # Basic assertion before processing
        assert not isinstance(g.network_deployment_obj, list), "network_deployment_obj has not been processed yet"
        assert hasattr(g.network_deployment_obj, "df_ep"), "network_deployment_obj must have a 'df_ep' attribute"
        assert isinstance(g.network_deployment_obj.df_ep, pd.DataFrame), "'df_ep' must be a pandas DataFrame"

        # Get engineering parameters DataFrame
        df_ep = g.network_deployment_obj.df_ep

        # Get indices of coverage and capacity cells
        idx_Cov_cell = df_ep[df_ep["cell_role"] == "coverage"].index.values.astype(np.int32)
        idx_Cap_cell = df_ep[df_ep["cell_role"] == "capacity"].index.values.astype(np.int32)

        # Get RSRP values used for first cell selection
        reported_RSRP_ue_to_cell_dBm = g.best_serving_cell_ID_per_ue_based_on_SSB_obj.RSRP_ue_to_cell_obj.RSRP_results_per_frequency_layer['all_freq']['RSRP_ue_to_cell_dBm']
        beam_to_cell_mapping = g.best_serving_cell_ID_per_ue_based_on_SSB_obj.beam_conf_obj.beam_to_node_mapping

        # Compute best serving beam and cell ID per UE
        # Note, this value maybe different to the actual serving cell
        # due to the priority cell re-selection process
        best_beam_ID_per_ue = np.argmax(reported_RSRP_ue_to_cell_dBm, axis=1)
        best_cell_ID_per_ue = beam_to_cell_mapping[best_beam_ID_per_ue]

        # Initialize pairing array
        capCell_paired_covCell = np.full_like(idx_Cap_cell, np.nan)

        for i, cap_idx in enumerate(idx_Cap_cell):
            # Find connected UEs to the capacity cell
            connected_ue = np.where(best_cell_ID_per_ue == cap_idx)[0]

            # Get their reported RSRP values for the coverage cells
            rsrp_covCell_per_ue_reported_dB = reported_RSRP_ue_to_cell_dBm[connected_ue][:, idx_Cov_cell]

            # Get the best coverage cell auxiliary index per UE
            best_cov_cell_per_ue_idx_aux = rsrp_covCell_per_ue_reported_dB.argmax(1)

            # Get the most frequently reported coverage cell auxiliary index
            most_frequent_covCell_idx_aux = np.bincount(best_cov_cell_per_ue_idx_aux).argmax()

            # Map auxiliary index to actual coverage cell index
            # store it for the considered capacity cell
            capCell_paired_covCell[i] =  idx_Cov_cell[most_frequent_covCell_idx_aux]

        # Make sure the capCell_paired_covCell is of integer type
        capCell_paired_covCell = capCell_paired_covCell.astype(int)
        # Get the number of cells
        number_of_cells = df_ep.shape[0]
        # Initialize symmetric pairing matrix
        pairing_CovCap_cell_matrix = np.zeros((number_of_cells, number_of_cells), dtype=bool)

        # Fill in pairing matrix symmetrically based on valid pairs
        for i, j in zip(idx_Cap_cell, capCell_paired_covCell):
            pairing_CovCap_cell_matrix[i,j] = True
            pairing_CovCap_cell_matrix[j,i] = True

        return pairing_CovCap_cell_matrix, capCell_paired_covCell


    def _pair_by_reported_RSRP__adjacencyMatrixPerLayer(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        # Compute the adjacency neighbouring matrix based on per-layer fallback logic
        adjacency_neighbouring_matrix: np.ndarray = self._compute_adjacency_per_layer(**kwargs)

        # Set adjacency neighbouring matrix
        self.adjacency_neighbouring_matrix = adjacency_neighbouring_matrix

        # Take only the most adjacent cell per row (i.e. the one with the highest percentage of UEs that would switch to it if they re-associated by max RSRP), and apply a threshold to avoid pairing cells that are not sufficiently adjacent. This results in a boolean adjacency matrix.
        most_neighbouring_matrix_perLayer: np.ndarray = self._compute_most_neighbour_per_layer(adjacency_neighbouring_matrix, **kwargs)

        # Get the Giulia object
        giulia_pairing = kwargs.get('giulia_pairing', None)
        assert giulia_pairing is not None, "giulia_pairing must be provided in kwargs for adjacency-based pairing method"

        # Engineering-parameters table (one row per cell)
        df_ep = giulia_pairing.network_deployment_obj.df_ep

        # Global indices of coverage and capacity cells
        idx_Cov_cell = df_ep[df_ep["cell_role"] == "coverage"].index.values.astype(np.int32)
        idx_Cap_cell = df_ep[df_ep["cell_role"] == "capacity"].index.values.astype(np.int32)

        N_cap_cell = len(idx_Cap_cell)
        # Paired coverage cell per capacity cell; float64 so np.nan is a valid "unassigned" sentinel
        capCell_paired_covCell = np.full(N_cap_cell, np.nan)

        # For each capacity cell, find its paired coverage cell
        for i, cap_idx in enumerate(idx_Cap_cell):

            # This capacity cell's most-neighbour flags restricted to coverage cells
            most_neighbouring_vec_covCell: np.ndarray = most_neighbouring_matrix_perLayer[cap_idx][idx_Cov_cell]

            # All-True mask, placeholder to later restrict pairing to a target layer
            mask_target_layer = np.ones_like(most_neighbouring_vec_covCell, dtype=bool) # For Future Extension

            # Apply the (currently no-op) layer mask
            most_neighbouring_vec_covCell_masked = most_neighbouring_vec_covCell[mask_target_layer]

            # Local index of the chosen coverage cell (first True flag)
            idx_paired_covCell_local = most_neighbouring_vec_covCell_masked.argmax()

            # Map local index back to the global coverage-cell index
            idx_paired_covCell_global = idx_Cov_cell[mask_target_layer][idx_paired_covCell_local]

            # Store the pairing for this capacity cell
            capCell_paired_covCell[i] =  idx_paired_covCell_global


        # Fail loudly if any capacity cell was left unpaired (NaN would silently become garbage on cast)
        assert not np.isnan(capCell_paired_covCell).any(), "Some capacity cell was left unpaired"
        # Cast pairings to integer cell indices
        capCell_paired_covCell = capCell_paired_covCell.astype(int)
        # Total number of cells in the deployment
        number_of_cells = df_ep.shape[0]
        # Symmetric cell-to-cell pairing matrix (all False initially)
        pairing_CovCap_cell_matrix = np.zeros((number_of_cells, number_of_cells), dtype=bool)

        # Mark each cap-cov pair True in both directions
        for i, j in zip(idx_Cap_cell, capCell_paired_covCell):
            pairing_CovCap_cell_matrix[i,j] = True
            pairing_CovCap_cell_matrix[j,i] = True

        # Return the symmetric pairing matrix and the per-capacity-cell pairing vector
        return pairing_CovCap_cell_matrix, capCell_paired_covCell


    ############### Internal: adjacency computation helpers ###############
    def _compute_adjacency_per_layer(self, **kwargs) -> np.ndarray:

        # Get per-beam RSRP and the Giulia object
        RSRP_dBm: np.ndarray|None = kwargs.get('RSRP_dBm_all_realization', None)
        assert RSRP_dBm is not None, "RSRP_dBm_all_realization must be provided in kwargs for adjacency-based pairing method"

        giulia_pairing: Giulia_Pairing|None = kwargs.get('giulia_pairing', None)
        assert giulia_pairing is not None, "giulia_pairing must be provided in kwargs for adjacency-based pairing method"

        # Collapse the beam dimension to get per-cell RSRP: shape (N_ue, N_BS)
        RSRP_cell_dBm: np.ndarray = self._compute_rsrp_cell_matrix(giulia_pairing, RSRP_dBm)
        N_BS: int = RSRP_cell_dBm.shape[1]

        if not isinstance(giulia_pairing.network_deployment_obj, Network):
            raise ValueError("Expected giulia_pairing.network_deployment_obj to be an instance of Network")
        if not isinstance(giulia_pairing.network_deployment_obj.df_ep, pd.DataFrame):
            raise ValueError("Expected giulia_pairing.network_deployment_obj.df_ep to be a pandas DataFrame")

        # Get the frequency of each cell from the Giulia object. This is needed to group cells by frequency for the per-layer fallback logic.
        freq_per_cell = giulia_pairing.network_deployment_obj.df_ep["dl_carrier_frequency_GHz"].to_numpy()
        # Compute a dictionary mapping each frequency to the indices of the cells operating on that frequency.
        cell_in_freq_dict = {freq: np.where(freq_per_cell == freq)[0] for freq in np.unique(freq_per_cell)}

        # Output: row i = adjacency neighbouring matrix of cell i to every other cell (and 0 on the diagonal)
        adjacency_neighbouring_matrix: np.ndarray = np.zeros((N_BS, N_BS), dtype=float)

        # For each UE, pick the cell with the highest RSRP — i.e. its serving cell
        serving_cell_ue = np.argmax(RSRP_cell_dBm, axis=1)
        for cell_idx in range(N_BS):
            # UEs currently served by this cell
            idx_ue_connected = np.where(serving_cell_ue == cell_idx)[0]
            # No UEs on this cell -> leave its adjacency neighbouring matrix row at zero
            if idx_ue_connected.size == 0: continue

            # RSRP of those UEs across every cell — shape (n_connected_ues, N_BS)
            X = RSRP_cell_dBm[idx_ue_connected]

            # Mask out the serving cell so the "next best" search ignores it.
            # X is a copy (fancy indexing above), so this doesn't touch RSRP_cell_dBm.
            X[:, cell_idx] = -np.inf

            for _, cell_indices in cell_in_freq_dict.items():

                # Boolean mask selecting only cells in the current frequency layer.
                # The serving cell was already neutralized with -inf above, so it won't win the argmax even if it's in this layer.
                mask = np.isin(np.arange(N_BS), cell_indices)
                X_masked = X[:, mask]

                # Second-choice cell for each UE within the selected frequency layer: argmax over the remaining columns
                new_serving_cell_ue_masked = np.argmax(X_masked, axis=1)

                # Percentage of UEs that would switch to each other cell in the same frequency layer
                # if they re-associated by max RSRP.
                perc = np.bincount(new_serving_cell_ue_masked, minlength=mask.sum()) / new_serving_cell_ue_masked.shape[0]

                # Store
                adjacency_neighbouring_matrix[cell_idx, mask] = perc

        return adjacency_neighbouring_matrix


    def _compute_most_neighbour_per_layer(self, adjacency_neighbouring_matrix: np.ndarray, **kwargs) -> np.ndarray:
        # Get the Giulia object
        giulia_pairing = kwargs.get('giulia_pairing', None)
        assert giulia_pairing is not None, "giulia_pairing must be provided in kwargs for adjacency-based pairing method"

        # Init
        most_neighbouring_matrix_perLayer: np.ndarray = np.full_like(adjacency_neighbouring_matrix, False, dtype=bool)

        # Get the frequency of each cell from the Giulia object. This is needed to group cells by frequency for the per-layer fallback logic.
        freq_per_cell = giulia_pairing.network_deployment_obj.df_ep["dl_carrier_frequency_GHz"].to_numpy()
        # Compute a dictionary mapping each frequency to the indices of the cells operating on that frequency.
        cell_in_freq_dict = {freq: np.where(freq_per_cell == freq)[0] for freq in np.unique(freq_per_cell)}

        N_cell: int = len(freq_per_cell)
        for cell_idx in range(N_cell):
            for freq_layer in cell_in_freq_dict.keys():
                # Get the indices of the cells in this frequency layer
                cell_indices_in_layer: np.ndarray = cell_in_freq_dict[freq_layer]
                # Get the adjacency neighbouring values of the current cell to the cells in this layer
                adjacent_neighbouring_in_layer: np.ndarray = adjacency_neighbouring_matrix[cell_idx][cell_indices_in_layer]
                # Get the index of the most adjacent cell in this layer
                most_adjacent_cell_in_layer_idx_local = np.argmax(adjacent_neighbouring_in_layer)
                # Get the actual cell index of the most adjacent cell in this layer
                most_adjacent_cell_in_layer = cell_indices_in_layer[most_adjacent_cell_in_layer_idx_local]
                # Store in the output boolean matrix that the most adjacent cell in this layer is a neighbour of the current cell
                most_neighbouring_matrix_perLayer[cell_idx, most_adjacent_cell_in_layer] = True

        return most_neighbouring_matrix_perLayer


    def _compute_rsrp_cell_matrix(self, _g: Giulia_Pairing, RSRP_dBm: np.ndarray) -> np.ndarray:
        # 1-D array of length N_beams_total mapping each beam column -> owning cell id
        beam_to_cell = _g.best_serving_cell_ID_per_ue_based_on_SSB_obj \
                        .beam_conf_obj.beam_to_node_mapping

        # Number of beams owned by each cell, ordered by sorted unique cell id.
        # This relies on RSRP_dBm's columns being laid out contiguously per cell in that same order.
        number_beam_per_cell = np.array([(beam_to_cell == cell).sum() for cell in np.unique(beam_to_cell)]).astype(int)

        N_ue = RSRP_dBm.shape[0]              # total number of UEs on the grid
        N_BS = number_beam_per_cell.shape[0]  # total number of cells (base stations)

        # Collapse the beam dimension via best-beam selection: per (UE, cell) take the max RSRP
        # across that cell's beams. Output shape: (N_ue, N_BS).
        RSRP_cell_dBm = np.zeros((N_ue, N_BS))

        # Walk the beam axis cell by cell, slicing each cell's contiguous block of columns
        n_beam_processed = 0
        for cell_idx in range(N_BS):
            # Width of this cell's beam block
            n_beam = number_beam_per_cell[cell_idx]

            # Best beam (max over the cell's beam columns) per UE
            RSRP_cell_dBm[:, cell_idx] = RSRP_dBm[:, n_beam_processed:n_beam_processed + n_beam].max(1)

            # Advance the cursor to the start of the next cell's block
            n_beam_processed += n_beam

        return RSRP_cell_dBm


    ############### Internal: write results back to df_ep ###############
    def _write_pairing_to_df_ep(self) -> None:
        # Get computed capCell_paired_covCell
        capCell_paired_covCell = self.get_capCell_paired_covCell()
        assert capCell_paired_covCell is not None, "capCell_paired_covCell is None."

        # Get df_ep from network deployment object
        df_ep = self.network_deployment_obj.df_ep

        # Basic Sanity Check
        if not isinstance(df_ep, pd.DataFrame): raise TypeError("df_ep must be a pandas DataFrame")

        # Case 1: All cells are coverage type → carrier shutdown disabled
        if df_ep["cell_role"].eq("coverage").all():
            return  # No further processing needed

        # Validate roles: must contain only "coverage" and "capacity"
        if set(df_ep["cell_role"].unique()) != {"capacity", "coverage"}:
            raise ValueError("df_ep['cell_role'] must contain only 'coverage' and 'capacity' roles")

        # Get indexes of capacity cells
        idx_Cap_cell = df_ep[df_ep["cell_role"] == "capacity"].index.values.astype(np.int32)


        ## Set for capacity cell in df_ep and coverage cell dict
        # Create dictionary to hold paired capacity cells for each coverage cell
        dict_cov_paired = {f"cov_cell_{idx}": [] for idx in df_ep[df_ep["cell_role"] == "coverage"].index.values.astype(np.int32)}
        for idx_Cap_cell, paired_cov_cell in zip(idx_Cap_cell, capCell_paired_covCell):
            # Populate capacity cell in data frame
            # In for loop is needed to ensure same list type assignment
            df_ep.at[ idx_Cap_cell, 'paired_cell'] = paired_cov_cell if isinstance(paired_cov_cell, list) else [paired_cov_cell] # type: ignore

            # Populate coverage cells dictionary with paired capacity cells
            dict_cov_paired[f"cov_cell_{paired_cov_cell}"].append(idx_Cap_cell)
        del idx_Cap_cell, paired_cov_cell

        # Set paired capacity cells for each coverage cell in df_ep
        for key in dict_cov_paired.keys():
            # Get coverage cell index
            cov_idx = int(key.split("_")[2])
            # If no capacity cell is paired, skip and leave the None value
            if len(dict_cov_paired[key]) == 0: continue
            # Set paired capacity cells as a list
            df_ep.at[ cov_idx, 'paired_cell'] = dict_cov_paired[key] # type: ignore
        del cov_idx, key

        # Update df_ep in network deployment object
        self.network_deployment_obj.df_ep = df_ep
        del df_ep
        free_memory()
        return
