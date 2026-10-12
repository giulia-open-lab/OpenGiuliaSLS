# Running Giulia

This page covers how to launch a simulation once your environment is set up. For installation see [Environment setup](environment-setup.md). For the full list of supported scenarios, UE distributions and link directions see [Models](models.md).

## Preparing the config file

Giulia expects a YAML config file. The repository ships a fully commented template at [`config.example.yml`](../config.example.yml), and a `config.yml` that runs a heavy 4G/5G multilayer scenario. The recommended workflow is to copy the template to `config.yml` (this replaces the `config.yml` of the repository) and edit in place:

```shell
cp config.example.yml config.yml
```

By default `main.py` looks for `config.yml` next to itself, and creates it from `config.example.yml` if it does not exist; you can also pass an explicit path on the command line (see "Launching" below).

The config file has three top-level sections: `args`, `run`, and `output`.

### `args` section

Controls the simulator's behaviour. The meaningful keys:

- `preset` — the pipeline preset. One of `GiuliaStd` (standard, single frequency layer) or `GiuliaMfl` (multi-frequency layer).
- `scenario_model` — the scenario / propagation model. See [Models](models.md) for the full list.
- `ue_playground_model` — overrides the default UE playground for the scenario. Leave empty / `null` to use the scenario default; otherwise `rectangular` or `circular`.
- `ue_distribution` — how UEs are scattered: `grid`, `uniform`, `uniform_with_hotspots`, `inhomogeneous_per_cell`, or `inhomogeneous_per_cell_with_hotspots`.
- `ue_mobility` — UE mobility. Empty / `null` for static UEs; otherwise `straight_walk` or `circular_walk`.
- `link_direction` — `downlink`, `uplink`, or `downlink_uplink`.
- `wraparound` — enables network-topology wraparound for interference at the edges of the playground. Leave it empty to use the scenario default (on for the hexagonal ITU and 3GPP layouts); see [Boolean entries](#boolean-entries).
- `snapshots` — number of simulation snapshots (a positive integer). The random seed increments by one per snapshot.
- `enable_GPU` — see [Boolean entries](#boolean-entries): `false` is currently read as true. To force CPU execution, start Giulia with `CUDA_VISIBLE_DEVICES=""`.
- `regression` — stored in the simulation configuration, but not read by the simulator modules.
- `log_level` — integer in `[0, 4]`. **Lower is more verbose**: `0` = VERBOSE, `1` = DEBUG,
  `2` = INFO, `3` = WARNING, `4` = ERROR (see `giulia/logger.py`). Set `0` to see everything and
  `4` to see only errors.
- `save_results` — stored in the simulation configuration, but not read by the simulator modules: the export is
  controlled by `output.formats` only (with `save_results: 0` the files are still written).
- `plots` — when on, writes the data of the diagnostic plots (`to_plot_*.npz` files) for the first episode into
  `outputs/results/<project>/` inside the repository. Leave it empty to keep it off; see [Boolean entries](#boolean-entries).

#### Boolean entries

The `args` entries are passed to the entry point as command-line text (`--plots=False`), and the entry point parses
the boolean options with Python's `bool()`, for which any non-empty text is true. The value `false` is therefore read
as **true** for `enable_GPU`, `wraparound`, `regression` and `plots`. Leave an entry empty to use its default: `plots`
and `regression` are off by default, and an empty `wraparound` follows the scenario.

### `run` section

Selects which example script `main.py` dispatches to via the `example` key. The names below are accepted by `giulia/inputs/input_config.py`, but **only `EV` currently runs through `main.py`**: `EV_batch` stops with `ModuleNotFoundError`.

- `EV` — runs `examples/giulia_EV.py`. The standard end-to-end run, parameterised by the `args` block above.
- `EV_batch` — runs `examples/giulia_EV_batch.py`. Internally batches multiple runs; the `args` block from the YAML is **not** propagated (the script defines its own sweep).

### `output` section

Controls where the results land and in which formats. Parsed in
[`giulia/inputs/input_config.py`](../giulia/inputs/input_config.py):

- `directory` — output directory of the Parquet datasets and their schemas, absolute or relative to the folder
  from which Giulia is started. In `config.example.yml` it is `outputs/gof`. The NPZ archive does not use it (see
  [Where outputs land](#where-outputs-land)).
- `formats` — a mapping of the output formats to produce. The two supported keys are `npz` and
  `parquet`; both can be enabled at the same time, and they are written **independently** (Parquet
  is not a post-hoc conversion of the `.npz` files). If the whole `formats` block is omitted,
  Giulia defaults to Parquet only, partitioned and packed into `.gof` archives, with the uncompressed folders removed.
  - `formats.npz.enable` — true/false. Writes the legacy `.npz` result archives.
  - `formats.parquet.enable` — true/false. Writes the Parquet datasets described in
    [Reading Parquet Output](output-saveable/reading-parquet.md).
  - `formats.parquet.compression` — `none` (leave the raw Parquet folder structure) or `zip`
    (compress it into a single archive, which Giulia names with a `.gof` extension — "Giulia
    Output File").
  - `formats.parquet.rm_uncompressed` — true/false. Only meaningful when `compression` is `zip`:
    if true, the uncompressed Parquet folder is deleted once the archive exists.
  - `formats.parquet.use_partition_cols` — true/false. If true, the data is written as a
    partitioned Parquet dataset (`episode_idx=…/step_idx=…/` folders) instead of a single file
    per label. If omitted, it defaults to `true`; the config files of the repository set it to `false`.
- `fields` — a YAML list of output columns to keep. If empty or omitted, **all** available columns
  are exported. Names are Saveable column names in `<ClassName>.<variable_name>` form (for example
  `SINR_SSB.sinr_ue_to_cell_dB`); see [Where outputs land](#where-outputs-land) below.

## Launching

From an activated environment, with `config.yml` next to `main.py`:

```shell
python main.py
```

To use a config at a different path:

```shell
python main.py path/to/config.yml
```

`main.py` reads the YAML, resolves the example entry point chosen in `run.example`, and calls it. The
example drives the snapshot loop and writes the results in the formats enabled under
`output.formats`.

## Where outputs land

After a successful run, the Parquet datasets and their schemas live under the directory configured in
`output.directory` (`outputs/gof` in `config.example.yml`). The legacy NPZ archive (`results-raw.npz`) and, when
`plots` is on, the `to_plot_*.npz` files are always written to `outputs/results/<project>/` inside the repository.

The Parquet output is **not** one file per field. It is one dataset **per label**, named
`<project_name>_<label>`, where the label says what a row represents:

| Dataset | One row per… |
|---|---|
| `<project>_generic` | `(episode, step)` — each KPI cell holds the whole vector for that step; written only when a Saveable is routed to `generic`, which is not the case in a standard run |
| `<project>_ue_kpi` | `(episode, step, ue_id)` — one scalar per UE |
| `<project>_cell_kpi` | `(episode, step, cell_id)` — one scalar per cell |
| `<project>_ru_kpi`, `<project>_bbu_kpi`, … | `(episode, step, <entity>_id)` — one scalar per entity |

Each dataset is written next to a `<dataset>.schema.md` sidecar listing every column and its type —
the quickest way to see what a run actually produced. Which file a variable lands in is decided in
code by its Saveable `label`, not by the config; see
[Saveable Output & Export](output-saveable/saveable-output.md) for the full model, and
[Reading Parquet Output](output-saveable/reading-parquet.md) for how to read the files back.

If you set `output.formats.parquet.compression: zip`, the Parquet folder is compressed into a single
`.gof` archive. If `rm_uncompressed: true`, the uncompressed copy is deleted afterwards.

### Restricting the export with `output.fields`

`output.fields` is a single, global filter applied to **all** the datasets above. Leave it empty (or
omit it) to export everything. The names it matches are Saveable column names in
`<ClassName>.<variable_name>` form, for example:

```yaml
output:
  fields:
    - Cell_Selection_SSB.best_serving_cell_ID_per_ue
    - Cell_Selection_SSB.best_server_rsrp_per_ue_dBm
    - SINR_SSB.sinr_ue_to_cell_dB
    - Rate_based_on_eff_CSI_RS_SINR.ue_throughput_Mbps
    - Power_Consumption.power_consumption_per_radio_unit_kW
```

Three things to know before enabling it:

- **The authoritative list of valid names is the [Saveable catalog](output-saveable/saveable-catalog.md)**,
  which is generated from the code per preset. `config.example.yml` ships an active selection
  drawn from it. The available keys depend on the preset and on the scenario: a key of a module that the
  run does not instantiate matches nothing, and keys that match nothing are ignored without a warning.
- **Index columns are always kept** — `episode_idx`, `step_idx`, and every entity id (`ue_id`,
  `cell_id`, `ru_id`, …).
- **Some core KPIs exist only in the NPZ archive.** The hardcoded core KPIs that are not Saveable
  variables (for example `total_network_power_consumption_kW` or `carrier_throughput_Mbps`) are not
  catalog entries, and in a standard run no Parquet `generic` file is written. Enable the NPZ format to
  get them. The `fields` filter does not apply to the NPZ archive.

A dataset left with nothing but index columns after filtering is not created at all.

To restrict the export, list only the fields you care about under `output.fields`.

## Testing

Regression and unit tests have their own guide. See [Testing](testing.md) for how to run the test harness, how to (re)generate the precomputed expected outputs, and how to produce HTML/Markdown reports.

## Customizing base station parameters per scenario

The YAML config lets you pick **which** scenario to run (e.g. `ITU_R_M2135_UMa`, `3GPPTR38_901_5G`). It does **not** expose the per-scenario base-station parameters: things like

- BS antenna height (`bs_antenna_height_m`),
- sector bearings (`sector_bearing_deg`),
- number of sectors per site (`number_of_sectors_per_site`),
- carrier frequency in GHz (`carrier_frequency_GHz`),
- channel bandwidth in MHz (`bandwidth_MHz`),
- BS transmit power in dBm (`BS_tx_power_dBm`),
- inter-site distance,

are hard-coded inside `giulia/bs/bs_deployments.py`. The `Network.process()` method is one long
`if / elif` chain on the chosen `scenario_model`, and each branch calls one dedicated builder
method named `construct_scenario_...`.

> **The builder name is usually, but not always, `construct_scenario_<scenario_model>`.** Roughly a
> third of the 48 branches deviate, so do **not** guess the method name from the config value. The
> recurring deviations:
>
> | `scenario_model` | Builder actually called |
> |---|---|
> | `..._lsc`, `..._lsc_sn`, `..._lsc_single_bs`, `..._lsc_single_sector` | `..._large_scale_calibration`, `..._large_scale_calibration_sn`, ... |
> | `3GPPTR38_901_UMi_C_band_lsc`, `3GPPTR38_901_UMi_fr3_lsc`, `3GPPTR38_901_UPi_fr3_lsc` | `..._C_band_large_scale`, `..._fr3_large_scale` (no `_calibration`) |
> | `3GPPTR38_901_4G5G_multilayer`, `3GPPTR38_901_4G_5G_multilayer`, ... | `construct_scenario_3GPPTR38_901_4G5G`, `..._4G_5G`, ... (no `_multilayer`) |
> | `ITU_R_M2135_UMa_multilayer` | `construct_scenario_ITU_R_M2135_UMa_colocated_multilayer` |
> | `3GPPTR38_811_Dense_Urban_NTN` | `construct_scenario_3GPPTR38_811_Desne_Urban_NTN` (typo in the source) |

To change the base-station configuration for a given scenario:

1. Open `giulia/bs/bs_deployments.py`.
2. Find the `elif ... playground_model == "<your scenario_model>"` branch in `Network.process()` and
   read the builder name off it. This is the reliable way -- searching for the method name directly
   can fail because of the deviations above.
3. Edit the relevant attributes inside that method: `bs_antenna_height_m`, `sector_bearing_deg`, `number_of_sectors_per_site`, `carrier_frequency_GHz`, `bandwidth_MHz`, `BS_tx_power_dBm`, etc.

We recommend keeping such changes local — work on a branch, or copy the builder method under a new name and add a corresponding `scenario_model` value — so that you can always compare against the reference 3GPP / ITU values that ship with the repository.

> **Warning.** Once you modify these per-scenario values, the run no longer strictly matches the standardised 3GPP / ITU scenario whose name it carries. Results from a customised run should not be reported as compliant with the original scenario.
