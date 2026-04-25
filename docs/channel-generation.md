# Channel generation

This page documents how Giulia builds a channel between a UE antenna element and a cell antenna element, end-to-end. Each step lists the responsible source file and, where the source code itself cites a standard, the corresponding 3GPP / ITU document. The exact formulas live in the source — this page describes the procedure and the inputs/outputs of each step rather than reproducing the equations.

## Standards referenced by the code

The propagation and channel modelling under `giulia/channel/` is built on top of the following standards. Each one is cited explicitly in the code (in function names, comments, or both):

- **3GPP TR 38.901** — primary 5G channel model. Drives the UMa / UMi LoS probability, K-factor, path loss, O2I penetration, and the Sionna TR 38.901 OFDM channels. The array steering vector (`giulia/antenna/array_steering_vectors.py`) explicitly follows TR 38.901 section 7.5 (Fast fading model), equations 7.5-23, 7.5-24, and 7.5-29.
- **3GPP TR 36.814** — LTE Case 1 path loss / LoS probability / shadowing for the `3GPPTR36_814_*` scenarios.
- **3GPP TR 36.777** — UAV / aerial vehicles. The aerial K-factor branches in `giulia/channel/k_factor.py` cite "3GPP TR 36.777 V15.0.0 (2017-12), Annex B: Channel modelling details, Alternative 2".
- **3GPP TR 38.811** — non-terrestrial networks (NTN, HAPS). Drives the LoS probability, K-factor (elevation-binned tables), path loss (with atmospheric attenuation via `itur`), and O2I penetration for the `3GPPTR38_811_*` scenarios.
- **ITU-R M.2135** — UMa / UMi reference scenarios and their associated path loss / LoS probability formulas (`ITU_R_M2135_UMa`, `ITU_R_M2135_UMi`).
- **ITU-R P-series** — accessed indirectly through the `itur` Python library (gaseous, rain, cloud, scintillation attenuation), used by the NTN path-loss branch in `giulia/channel/path_losses.py` (function `path_loss_3GPPPTR38_811`).

The procedure that follows is the order in which the standard preset (`giulia/presets/g_standard_preset.py`) wires the channel pipeline.

## Step-by-step procedure

### (a) Geometry — distances and angles

**File:** `giulia/playground/distances_angles.py`.

Two passes are run:

- `Distance_Angles_ue_to_cell` — for every UE and every cell, computes the 2D and 3D distances and the azimuth/zenith angles, accounting for wraparound when enabled.
- `Distance_Angles_ueAnt_to_cellAnt` — refines the geometry to per-antenna-element resolution (each UE antenna element to each cell antenna element), which is needed for steering-vector and fast-fading computations.

Outputs feed essentially every following step.

### (b) LoS probability

**File:** `giulia/channel/los_probabilities.py`.

For each UE-cell link, the simulator draws a binary LoS flag from a per-scenario LoS probability formula. The formula varies by scenario family:

- `3GPPTR38_901_UMa` and `3GPPTR38_901_UMi` — implements the LoS probability model from 3GPP TR 38.901 (see source for the exact formula).
- `ITU_R_M2135_UMa` / `ITU_R_M2135_UMi` — uses the ITU-R M.2135 LoS probability formulas.
- `3GPPTR36_814_Case_1` — Case 1 of 3GPP TR 36.814 (treats links as NLoS, see source).
- `3GPPTR36_777_UMa_AV` / `3GPPTR36_777_UMi_AV` — UAV branches from 3GPP TR 36.777 (height-aware extensions of the TR 38.901 / TR 36.814 ground-UE formulas).
- `3GPPTR38_811_Urban_NTN` / `3GPPTR38_811_Dense_Urban_NTN` — elevation-dependent NTN tables from 3GPP TR 38.811.

The output is the per-link LoS flag and the underlying LoS probability matrix; both are consumed by the K-factor, path loss, and shadowing steps.

### (c) Rician K-factor

**File:** `giulia/channel/k_factor.py`.

For every LoS link, a log-normal Rician K-factor is sampled. The branches per scenario:

- `3GPPTR38_901_UMa` and `3GPPTR38_901_UMi` — uses the K-factor model from 3GPP TR 38.901 (LoS-only, log-normal; see source for the exact mean and standard deviation per scenario).
- `ITU_R_M2135_UMa` / `ITU_R_M2135_UMi` — ITU-R M.2135 K-factor.
- `3GPPTR36_814_Case_1` — TR 36.814 LoS K-factor.
- `3GPPTR36_777_UMa_AV` / `3GPPTR36_777_UMi_AV` — the aerial branch explicitly cites "3GPP TR 36.777 V15.0.0 (2017-12), Annex B: Channel modelling details, Alternative 2"; the ground branch references "Table 7.5-6" inline in the source.
- `3GPPTR38_811_Urban_NTN` / `3GPPTR38_811_Dense_Urban_NTN` — elevation-binned tables (0°-15°, 15°-25°, ..., 85°-90°) from 3GPP TR 38.811.

The output K-factor (in dB, clipped to a non-negative range) feeds the LoS-channel and fast-fading steps.

### (d) Path loss

**File:** `giulia/channel/path_losses.py`.

Each scenario has its own dedicated method, named `path_loss_<scenario>`, dispatched from `Path_Loss.process()`. The formulas are consistent with the corresponding 3GPP / ITU specification:

- `path_loss_3GPPTR38_901_UMa`, `path_loss_3GPPTR38_901_UMi` — implement the UMa / UMi path-loss formulas from 3GPP TR 38.901 (with breakpoint distance `d_BP`, LoS / NLoS branches and the standard validity ranges for UE height [1.5 m, 22.5 m]).
- `path_loss_ITU_R_M2135_UMa`, `path_loss_ITU_R_M2135_UMi` — implement the ITU-R M.2135 UMa / UMi path-loss formulas (with the 25 m / 10 m fixed BS heights and the 1.5 m UE height assumed by the spec).
- `path_loss_3GPPTR36_814_Case_1`, `path_loss_3GPPTR36_814_UMa` — implement the LTE Case 1 path-loss models from 3GPP TR 36.814.
- `path_loss_3GPPTR36_777_UMa_AV`, `path_loss_3GPPTR36_777_UMi_AV` — implement the UAV path-loss models from 3GPP TR 36.777 (with the height-extended validity ranges up to 300 m).
- `path_loss_3GPPPTR38_811` — implements the NTN path-loss model from 3GPP TR 38.811. Free-space path loss plus elevation-binned clutter loss plus atmospheric attenuation. The atmospheric component is computed via the `itur` library (ITU-R P-series), using `gaseous_attenuation_slant_path`, `rain_attenuation`, `cloud_attenuation`, and `scintillation_attenuation`.

Output: per-link path loss in dB, used by the slow-channel step.

### (e) Outdoor-to-Indoor (O2I) penetration loss

**File:** `giulia/channel/o2i_penetration_losses.py`.

For UEs flagged as indoor, an additional penetration loss is added on top of the path loss. Two branches:

- `o2i_loss_3GPPTR38_901_UMa_Umi` — the O2I penetration model from 3GPP TR 38.901 (low-loss / high-loss building mix, frequency-dependent).
- `o2i_loss_3GPPTR38_811` — the elevation-dependent NTN O2I model from 3GPP TR 38.811.

For other scenarios the indoor penetration is folded into the path-loss step itself (e.g. `path_loss_ITU_R_M2135_UMi` adds a fixed wall-penetration loss directly in the path-loss formula, which is consistent with the model used in scenarios from ITU-R M.2135 — see source for the exact formula).

Output: per-link O2I penetration loss in dB, used by the slow-channel step.

### (f) Shadowing

**Files:** `giulia/channel/shadowing_maps.py` and `giulia/channel/shadowing_gains.py`.

Shadowing is a slow large-scale random component on top of the path loss. Giulia models it with **precomputed correlated maps** rather than re-sampling at runtime:

- `Shadowing_Map.process()` (in `shadowing_maps.py`) loads the relevant `.mat` files at the start of the simulation. The maps are stored as `shadowing/G_shadowing_map_sigma_<sigma>_std_<std>.mat`, where `<sigma>` is the standard deviation of the shadowing in dB and `<std>` is the spatial decorrelation length in metres (with dots in numbers replaced by underscores — e.g. `G_shadowing_map_sigma_7_82_std_13.mat` for σ = 7.82 dB and decorrelation length 13 m). Each scenario family selects its own (σ, decorrelation length) pair for LoS, NLoS, and (for some scenarios) O2I components, consistent with the values in the corresponding 3GPP / ITU specification.
- `Shadowing_Gain.process()` (in `shadowing_gains.py`) queries the loaded map at each UE position and returns the per-link shadowing gain in dB.

The default maps can be downloaded with:

```shell
python scripts/download_shadowing_files.py
```

If you need maps with custom statistics, regenerate them with the MATLAB scripts that ship under `shadowing/` (notably `shadow_fading_map_generator.m`, plus `shadow_fading_map_generator_ind.m`, `shadow_map.m`, and `pdf.m`).

Output: per-link shadowing gain in dB, used by the slow-channel step.

### (g) Slow channel gain

**File:** `giulia/channel/slow_channels.py`.

The slow channel gain combines the deterministic large-scale components into a single per-link gain:

```
slow_channel_gain_dB = antenna_pattern_gain - path_loss - O2I_loss + shadowing_gain
```

(see the source for sign conventions and additional terms). Inputs come from the antenna-pattern step (`giulia/antenna/antenna_pattern_gains.py`), the path-loss step, the O2I step, and the shadowing-gain step. Output: per-(UE, cell) slow channel gain in dB. This is used as the magnitude that scales the fast-fading complex coefficients.

### (h) LoS channel and array steering vectors

**Files:** `giulia/channel/channels.py` (`LoS_Channel`) and `giulia/antenna/array_steering_vectors.py`.

`Array_Steering_Vector` computes the array steering vector at every cell — the set of phase delays a plane wave experiences across the cell's antenna elements at the UE direction. The implementation explicitly follows 3GPP TR 38.901 section 7.5 (equations 7.5-23, 7.5-24, and 7.5-29), with the wave-vector / antenna-position formulation taken from Björnson's "Massive MIMO Networks" (section 7.3.1 — 3D LoS Model with Arbitrary Array Geometry), as documented in the file's docstring.

`LoS_Channel` then assembles the deterministic LoS component of the fast-fading channel by combining the slow channel gain, the steering vectors, and the K-factor weights. This is used to compute average / no-fast-fading metrics (e.g. RSRP without fast fading), since the NLoS component of the fast-fading model has zero mean.

### (i) Fast fading

**File:** `giulia/channel/fast_fading_gains.py`.

The fast-fading step generates per-PRB complex coefficients per UE-antenna / cell-antenna pair. Two components are combined:

- A **LoS component**, weighted by `sqrt(K / (K + 1))` (with K being the linear K-factor from step (c)), built from the array steering vectors of step (h).
- A **NLoS component**, weighted by `sqrt(1 / (K + 1))`, drawn from a complex Gaussian distribution (Rayleigh fading).

For LoS-only links (`K = inf`) the LoS component dominates; for NLoS-only links (`K = 0`) only the Rayleigh component is present. Output: complex fast-fading coefficients, stored on the GPU when available.

### (j) Optional Sionna ray-traced channel

**File:** `giulia/channel/channels.py` (`ChannelSn`).

If `sionna` is importable in the active environment, the standard preset wires up a Sionna-based channel model in addition to the analytical chain above:

- For the `3GPPTR38_901_UMa` and `3GPPTR38_901_UMi` `BS_fast_channel_model` values, `ChannelSn` instantiates `sionna.phy.channel.tr38901.UMa` / `UMi` channel models, sets the topology (UE / BS positions, orientations, velocities, indoor flags), and generates frequency-domain OFDM channels (`GenerateOFDMChannel`).
- For the `Ray_tracing` model, the `dataset` scenario uses the 3D meshes shipped under `data/data_driven_extras/` to drive a Sionna ray tracer.

When the Sionna branch is active, the corresponding fast-fading coefficients are taken from the Sionna OFDM channel rather than from the analytical fast-fading step.

### (k) Composite channel

**File:** `giulia/channel/channels.py` (`Channel`).

The composite channel coefficient between a UE antenna and a cell antenna is:

```
H_complex = sqrt(linear(slow_channel_gain_dB)) * fast_fading_coeff_complex
```

`Channel.channel_gain_b_to_a` performs this multiplication on the GPU (PyTorch). When the Sionna branch is active, the composite channel comes directly from the Sionna OFDM channel via `Channel.sn_channel_gain_b_to_a`. Output: per-PRB complex channel coefficient between every UE antenna element and every cell antenna element.

### (l) Precoded channels (SSB and CSI-RS)

**File:** `giulia/channel/precoded_channel_gains.py`.

The precoded channel applies the SSB / CSI-RS beam codebook on top of the composite channel:

- `Precoded_Channel_Gain_SSB_no_fast_fading_ue_to_cell` — long-term SSB precoded channel using the LoS-only channel from step (h). No fast-fading averaging needed; used for RSRP-without-fast-fading metrics and as a stable input for cell selection.
- `Precoded_Channel_Gain_SSB_ue_to_cell` — instantaneous SSB precoded channel, per PRB, including fast fading.
- `Precoded_Channel_Gain_CSI_RS_ue_to_cell` — instantaneous CSI-RS precoded channel, per PRB, including fast fading.

These outputs feed the RSS / RSRP / SINR / throughput steps in `giulia/kpis/`.

## Customising the channel

The propagation and channel chain is intentionally split across small modules so that you can tweak individual steps without touching the rest of the pipeline:

- **Per-scenario base-station parameters** (carrier frequency, bandwidth, BS antenna height, sector bearings, transmit power, number of sectors, etc.) live in `giulia/bs/bs_deployments.py`. See the "Customizing base station parameters per scenario" section in [Running Giulia](running-giulia.md) for the recommended workflow.
- **Shadowing statistics** — to use shadowing maps with custom σ or decorrelation length, regenerate them with the MATLAB scripts under `shadowing/` (`shadow_fading_map_generator.m`, `shadow_fading_map_generator_ind.m`, `shadow_map.m`). Save the new `.mat` file with the `G_shadowing_map_sigma_<sigma>_std_<std>.mat` filename convention so that the loader in `giulia/channel/shadowing_maps.py` picks it up.
- **Path-loss / LoS-probability formulas** — to tweak the propagation formula for an existing scenario, edit the matching per-scenario branch inside `giulia/channel/path_losses.py` and `giulia/channel/los_probabilities.py`. As with BS parameters, modifying these branches means the scenario no longer strictly matches the standardised 3GPP / ITU model whose name it carries.
