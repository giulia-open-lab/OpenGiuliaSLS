# Models

This page lists the models exposed through the `args` block of `config.yml` and the link-level / numerical components Giulia uses internally.

## Scenario / propagation models

The active scenario is selected with `args.scenario_model`. Each value maps to a per-scenario builder in `giulia/bs/bs_deployments.py` (`construct_scenario_<scenario_model>`) and to a propagation/channel branch under `giulia/channel/`.

### ITU-R M.2135 (urban macro / micro)

- `ITU_R_M2135_UMa`
- `ITU_R_M2135_UMi`

### 3GPP TR 36.814 (LTE Case 1 family)

- `3GPPTR36_814_Case_1`
- `3GPPTR36_814_Case_1_omni`
- `3GPPTR36_814_Case_1_single_bs`
- `3GPPTR36_814_Case_1_single_bs_omni`
- `3GPPTR36_814_Case_1_omni_dana`

### 3GPP TR 38.901 (5G UMa / UMi)

UMa configurations (Configuration 1, Configuration 2, large-scale calibration variants, fixed-frequency variants):

- `3GPPTR38_901_UMa_C1`
- `3GPPTR38_901_UMa_C2`
- `3GPPTR38_901_UMa_lsc`
- `3GPPTR38_901_UMa_lsc_sn`
- `3GPPTR38_901_UMa_lsc_single_bs`
- `3GPPTR38_901_UMa_lsc_single_sector`
- `3GPPTR38_901_UMa_2GHz_lsc`
- `3GPPTR38_901_UMa_C_band_lsc`

UMi configurations:

- `3GPPTR38_901_UMi_C1`
- `3GPPTR38_901_UMi_C2`
- `3GPPTR38_901_UMi_lsc`
- `3GPPTR38_901_UMi_C_band_lsc`
- `3GPPTR38_901_UMi_fr3_lsc`
- `3GPPTR38_901_UPi_fr3_lsc`

### 3GPP TR 36.777 (UAV / aerial vehicles)

- `3GPPTR36_777_UMa_AV`
- `3GPPTR36_777_UMi_AV`

### 3GPP TR 38.811 (NTN / HAPS)

- `3GPPTR38_811_Urban_NTN`
- `3GPPTR38_811_Dense_Urban_NTN`
- `3GPPTR38_811_Dense_Urban_HAPS_ULA`
- `3GPPTR38_811_Dense_Urban_HAPS_UPA`
- `3GPPTR38_811_Dense_Urban_HAPS_Reflector`

### Multi-RAT and layered (4G / 5G / 6G)

ITU-R-based multilayer scenarios:

- `ITU_R_M2135_UMa_multilayer`
- `ITU_R_M2135_UMa_Umi_colocated_multilayer`
- `ITU_R_M2135_UMa_Umi_noncolocated_multilayer`

3GPP TR 38.901-based RAT layers:

- `3GPPTR38_901_4G`
- `3GPPTR38_901_5G`
- `3GPPTR38_901_6G`
- `3GPPTR38_901_4G5G_multilayer`
- `3GPPTR38_901_4G_5G_multilayer`
- `3GPPTR38_901_4G_5G6G_multilayer`
- `3GPPTR38_901_4G_5G_6G_multilayer`
- `3GPPTR38_901_4G5G_cell_reselection`

### Dataset (data-driven)

- `dataset` — uses an external dataset for the scenario layout instead of one of the standardised templates. When `sionna` is installed, this scenario can also drive the Sionna ray-traced channel from 3D meshes under `data/data_driven_extras/`.

### UE distribution options

`args.ue_distribution` selects how UEs are scattered across the playground:

- `grid`
- `uniform`
- `uniform_with_hotspots`
- `inhomogeneous_per_cell`
- `inhomogeneous_per_cell_with_hotspots`

`args.ue_playground_model` overrides the per-scenario default UE playground shape. Leave it empty (or `null`) to use the scenario-specific shape; otherwise pick `rectangular` or `circular`.

### UE mobility options

`args.ue_mobility` is one of:

- `null` / empty — static UEs.
- `straight_walk`
- `circular_walk`

### Link direction options

`args.link_direction` is one of:

- `downlink`
- `uplink`
- `downlink_uplink` (both)

## Physical-layer / link-level models

The propagation and link-level chain Giulia runs for each UE-cell pair includes:

- **Path loss** — per-scenario closed-form models for distance-dependent attenuation, dispatched in `giulia/channel/path_losses.py`. Each scenario family has its own formula (UMa, UMi, Case 1, UAV, NTN); see [Channel generation](channel-generation.md) for the per-scenario standards mapping.
- **LoS probability** — per-scenario models in `giulia/channel/los_probabilities.py`. Determines whether a given UE-cell link is treated as LoS for the rest of the chain.
- **Rician K-factor** — log-normal random K-factor on LoS links (`giulia/channel/k_factor.py`). NTN scenarios use elevation-binned tables.
- **Shadowing** — large-scale slow fading sampled from precomputed correlated maps. The maps live as `.mat` files under `shadowing/`, parameterised by sigma (dB) and decorrelation length (m), with the filename convention `G_shadowing_map_sigma_<sigma>_std_<std>.mat`. They are loaded at runtime in `giulia/channel/shadowing_maps.py` and queried per UE position in `giulia/channel/shadowing_gains.py`. Default maps can be downloaded with `python scripts/download_shadowing_files.py`; they can be regenerated from the MATLAB scripts in `shadowing/` (notably `shadow_fading_map_generator.m`).
- **Outdoor-to-Indoor (O2I) penetration loss** — `giulia/channel/o2i_penetration_losses.py`. Applied to UEs flagged as indoor; for NTN scenarios, an elevation-dependent variant is used.
- **Fast fading** — `giulia/channel/fast_fading_gains.py`. Combines a deterministic LoS component (steering-vector-based, weighted by the K-factor) with a Rayleigh NLoS component.
- **Antenna patterns and array steering vectors** — `giulia/antenna/`. Patterns follow the 3GPP TR 38.901 family or omni for the omnidirectional variants. Steering vectors are computed per UE antenna element in `giulia/antenna/array_steering_vectors.py`.
- **SSB and CSI-RS beam codebooks** — `giulia/rrc/beam_configs.py`. Each cell exposes a set of SSB beams (used for cell discovery) and CSI-RS beams (used for refined beam management).
- **BLER vs SINR look-up tables** — `giulia/phy/lut_bler_vs_sinrs.py`. Used to map per-PRB SINR to block-error probabilities for the configured MCS table.
- **MIESM** — Mutual-Information Effective SINR Mapping, available in `giulia/phy/mutual_informations.py`. Selected when `simulation_config_obj.sinr_mapping == "MIESM"`.
- **Power-consumption model** — per-cell and network-level energy accounting (`giulia/kpis/power_consumptions.py`).

For the full step-by-step procedure used to assemble the channel between a UE antenna and a cell antenna, see [Channel generation](channel-generation.md).

## Underlying numerical / scientific stack

Giulia builds on a fixed set of numerical and scientific libraries pinned by the installer (see [Environment setup](environment-setup.md)):

- **NVIDIA Sionna 1.1.0** — used for the optional ray-traced and TR 38.901 OFDM channels.
- **TensorFlow 2.17.0** — backend for Sionna and for some link-level utilities.
- **PyTorch 2.7.1** — used in the channel pipeline (slow channel × fast fading composition, GPU tensor operations).
- **Simulus 1.2.1** — the discrete-event simulation kernel that backs `giulia/event_driven/`.
- **NumPy / SciPy / Pandas / scikit-learn / matplotlib / seaborn** — general numerical, dataframe, and plotting stack.
- **Shapely 2.1.1 / GeoPandas 1.1.1** — geometric primitives for the playground layout.
- **Trimesh 4.6.13 / Mitsuba 3.6.2** — 3D mesh handling and rendering, used by the data-driven and ray-traced scenarios.
- **Astropy 7.1.0** — units and astronomical helpers used by the NTN channel branches.
- **itur 0.4.0** — ITU-R P-series propagation (gaseous, rain, cloud, scintillation) used by the NTN path-loss model.
- **PyArrow 20.0.0** — Parquet I/O for the output module.
- **tqdm 4.67.1** — progress bars.
- **nvidia-ml-py 12.575.51** — GPU monitoring used by `scripts/monitor_GPU.py`.
