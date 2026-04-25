# Running Giulia

This page covers how to launch a simulation once your environment is set up. For installation see [Environment setup](environment-setup.md). For the full list of supported scenarios, UE distributions and link directions see [Models](models.md).

## Preparing the config file

Giulia expects a YAML config file. The repository ships a fully commented template at [`config.example.yml`](../config.example.yml). The recommended workflow is to copy it to `config.yml` and edit in place:

```shell
cp config.example.yml config.yml
```

By default `main.py` looks for `config.yml` next to itself; you can also pass an explicit path on the command line (see "Launching" below).

The config file has three top-level sections: `args`, `run`, and `output`.

### `args` section

Controls the simulator's behaviour. The meaningful keys:

- `preset` — the pipeline preset. One of `GiuliaStd` (standard, single frequency layer) or `GiuliaMfl` (multi-frequency layer).
- `scenario_model` — the scenario / propagation model. See [Models](models.md) for the full list.
- `ue_playground_model` — overrides the default UE playground for the scenario. Leave empty / `null` to use the scenario default; otherwise `rectangular` or `circular`.
- `ue_distribution` — how UEs are scattered: `grid`, `uniform`, `uniform_with_hotspots`, `inhomogeneous_per_cell`, or `inhomogeneous_per_cell_with_hotspots`.
- `ue_mobility` — UE mobility. Empty / `null` for static UEs; otherwise `straight_walk` or `circular_walk`.
- `link_direction` — `downlink`, `uplink`, or `downlink_uplink`.
- `wraparound` — true/false. Enables network-topology wraparound for interference at the edges of the playground.
- `snapshots` — number of simulation snapshots (a positive integer). The random seed increments by one per snapshot.
- `enable_GPU` — true/false. Set to `false` to force CPU execution.
- `regression` — true/false. Enables the regression-test mode used by the test harness.
- `log_level` — integer in `[0, 4]` (higher is more verbose).
- `save_results` — `0` (do not save) or `1` (save).
- `plots` — true/false. Generate plots in addition to data files.

### `run` section

Selects which example script `main.py` dispatches to via the `example` key. Supported values, dispatched in `giulia/inputs/input_config.py`:

- `EV` — runs `examples/giulia_EV.py`. The standard end-to-end run, parameterised by the `args` block above.
- `EV_batch` — runs `examples/giulia_EV_batch.py`. Internally batches multiple runs; the `args` block from the YAML is **not** propagated (the script defines its own sweep).
- `EV_EE_UEassociation` — runs `examples/giulia_EV_EE_UEassociation.py`. Energy-efficiency-oriented run focused on UE-cell association.
- `EV_EE_UEassociation_batch` — runs `examples/giulia_EV_EE_UEassociation_batch.py`. Batched variant of the previous.
- `EV_HapsSym_Reflector` — runs `examples/giulia_EV_HapsSym_Reflector.py`. Specialised run for the HAPS reflector NTN scenario.

### `output` section

Controls where the results land:

- `directory` — output directory, absolute or relative to the repo root. Default: `outputs/outputs`.
- `compression` — `none` (write raw Parquet folder structure) or `zip` (compress the result into a single zip).
- `rm_uncompressed` — true/false. Only used when `compression` is `zip`. If true, the uncompressed Parquet folder is deleted after compression.
- `fields` — a YAML list of output fields to include. If empty or omitted, all available fields are exported. The example `config.example.yml` enumerates all the standard KPI fields (best serving cell IDs, RSRP/RSS variants, SINR, throughput, cell activity, network power consumption, and so on).

## Launching

From an activated environment, with `config.yml` next to `main.py`:

```shell
python main.py
```

To use a config at a different path:

```shell
python main.py path/to/config.yml
```

`main.py` reads the YAML, dispatches to the example script chosen in `run.example`, and then runs the output module to convert `.npz` files to Parquet.

## Where outputs land

After a successful run, results live under the directory configured in `output.directory` (default `outputs/outputs`). Internally, each saveable component of the pipeline writes one or more `.npz` files; the output module then converts them to Parquet, keyed by the field name. The Parquet output is a folder (one file per field) by default.

If you set `output.compression: zip`, the output module zips the Parquet folder. If `rm_uncompressed: true`, the uncompressed copy is deleted after the zip is created.

The list of standard fields exposed in the Parquet output is the one in `config.example.yml`, which includes:

- Cell-association: `best_serving_cell_ID_per_ue_based_on_CRS`, `best_serving_cell_ID_per_ue_based_on_SSB`, `best_serving_SSB_per_ue`, `best_serving_CSI_RS_per_ue`.
- RSRP / RSS / coupling gain: `best_serving_CRS_rsrp_per_ue_dBm`, `best_serving_SSB_rsrp_per_ue_dBm`, `best_serving_CSI_RS_rsrp_per_ue_dBm`, `best_serving_CRS_coupling_gain_dB`, `best_serving_SSB_coupling_gain_dB`, `CRS_RSRP_no_fast_fading_ue_to_cell_dBm`, `SSB_RSRP_no_fast_fading_ue_to_cell_dBm`.
- Distances: `best_serving_CRS_distance_3d_m`, `best_serving_SSB_distance_3d_m`.
- SINR / interference: `CRS_sinr_ue_to_cell_dB`, `SSB_sinr_ue_to_cell_dB`, `effective_CSI_RS_sinr_ue_to_cell_dB`, `CRS_interfPower_ue_dBm`, `SSB_interfPower_ue_dBm`, `CSI_RS_interfPower_ue_dBm`, `CRS_usefulPower_ue_dBm`, `SSB_usefulPower_ue_dBm`, `CSI_RS_usefulPower_ue_dBm`.
- Throughput: `carrier_throughput_Mbps`, `cell_throughput_Mbps`, `ue_throughput_per_carrier_Mbps`, `ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps`, `ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps`.
- Activity / counts: `cell_activity_per_ue`, `SSB_beam_activity_per_ue`, `CSI_RS_beam_activity_per_ue`, `ues_per_cell`, `ues_per_cell_based_on_CRS`, `ues_per_SSB_beam`, `ues_per_CSI_RS_beam`.
- Energy: `network_power_consumption_kW`.
- Noise: `best_serving_CRS_re_noise_per_ue_dBm`, `best_serving_SSB_re_noise_per_ue_dBm`.

To restrict the export, list only the fields you care about under `output.fields`.

## Testing

Regression and unit tests have their own guide. See [Testing](testing.md) for how to run the test harness, how to (re)generate the precomputed expected outputs, and how to produce HTML/Markdown reports.

## Customizing base station parameters per scenario

The YAML config lets you pick **which** scenario to run (e.g. `ITU_R_M2135_UMa`, `3GPPTR38_901_UMa_C2`). It does **not** expose the per-scenario base-station parameters: things like

- BS antenna height (`bs_antenna_height_m`),
- sector bearings (`sector_bearing_deg`),
- number of sectors per site (`number_of_sectors_per_site`),
- carrier frequency in GHz (`carrier_frequency_GHz`),
- channel bandwidth in MHz (`bandwidth_MHz`),
- BS transmit power in dBm (`BS_tx_power_dBm`),
- inter-site distance,

are hard-coded inside `giulia/bs/bs_deployments.py`. The `Network.process()` method dispatches on the chosen `scenario_model` and calls one dedicated builder method per scenario, named with the convention:

```
construct_scenario_<scenario_model>
```

For example, `construct_scenario_ITU_R_M2135_UMa`, `construct_scenario_3GPPTR38_901_UMa_C2`, `construct_scenario_3GPPTR38_901_UMi_lsc`, `construct_scenario_3GPPTR36_777_UMa_AV`, and so on for every `scenario_model` value listed in [Models](models.md).

To change the base-station configuration for a given scenario:

1. Open `giulia/bs/bs_deployments.py`.
2. Grep for `construct_scenario_` followed by your scenario name (this is more robust than maintaining a list of method names here, because new scenarios are added over time).
3. Edit the relevant attributes inside that method: `bs_antenna_height_m`, `sector_bearing_deg`, `number_of_sectors_per_site`, `carrier_frequency_GHz`, `bandwidth_MHz`, `BS_tx_power_dBm`, etc.

We recommend keeping such changes local — work on a branch, or copy the builder method under a new name and add a corresponding `scenario_model` value — so that you can always compare against the reference 3GPP / ITU values that ship with the repository.

> **Warning.** Once you modify these per-scenario values, the run no longer strictly matches the standardised 3GPP / ITU scenario whose name it carries. Results from a customised run should not be reported as compliant with the original scenario.
