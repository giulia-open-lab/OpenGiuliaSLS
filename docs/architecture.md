# Architecture

This page describes how Giulia is organised and how a simulation flows from the entry point through the main pipeline stages.

## Entry point and configuration flow

The top-level entry point is [`main.py`](../main.py) at the repository root. It does three things:

1. Adds the repo root to `sys.path` so that `import giulia` and `import examples` resolve.
2. Reads the YAML config file (defaults to `config.yml` next to `main.py`, or a path supplied as the first command-line argument).
3. Builds an `InputConfig` (`giulia/inputs/input_config.py`) and calls `InputConfig.run()`.

`InputConfig.run()` translates the `args` block of the YAML file into command-line-style flags (e.g. `--scenario_model=ITU_R_M2135_UMa`) and dispatches to one of the example scripts under `examples/` based on the `run.example` key. The example script is the one that actually instantiates the simulator and drives the snapshot loop. Once the example finishes, an `OutputModule` collects the results from the configured output directory and converts them to Parquet, optionally compressing the result.

The YAML file has three sections: `args` (simulator behaviour, scenario, UEs, snapshots, GPU, logging), `run` (which example to dispatch to), and `output` (target directory, compression, list of fields to export). See [Running Giulia](running-giulia.md) for the full schema.

## Presets

Giulia exposes two presets that wire up the components of the pipeline differently:

- **GiuliaStd** — the standard preset, built in `giulia/presets/g_standard_preset.py`. Used for single-frequency-layer scenarios and for most 3GPP/ITU calibration cases.
- **GiuliaMfl** — the multi-frequency-layer preset, built in `giulia/presets/g_multi_frequency_layer_preset.py`. Used for scenarios that combine 4G/5G/6G layers or mix UMa/UMi sites.

The preset is selected via `args.preset` in the YAML config and resolved in `examples/giulia_EV.py`.

## Simulation pipeline

A simulation runs as a sequence of pipeline stages. The standard preset wires them up in `giulia/presets/g_standard_preset.py`; the multi-frequency-layer preset adds extra layering on top. The high-level flow is:

1. **Playground** (`giulia/playground/`) — builds the geometric playground: the reference site grid, the UE deployment area, and any hotspot regions.
2. **Network deployment** (`giulia/bs/`) — instantiates the network, places base stations on the playground, and assigns per-cell parameters (carrier frequency, bandwidth, antenna height, sector bearings, transmit power, and so on). The exact parameters are picked from the per-scenario builders in `giulia/bs/bs_deployments.py` (see also the "Customizing base station parameters per scenario" section in [Running Giulia](running-giulia.md)).
3. **Antennas and beams** (`giulia/antenna/`, `giulia/rrc/beam_configs.py`) — builds antenna arrays for cells and UEs, computes antenna pattern gains, derives array steering vectors, and creates the SSB and CSI-RS beam codebooks.
4. **Time-frequency resources and link-level setup** (`giulia/rrc/`, `giulia/phy/`, `giulia/mac/`) — defines the DL/UL carriers, PRB grid, MCS tables, BLER-vs-SINR look-up tables, and the SINR mapping (e.g. MIESM).
5. **UEs** (`giulia/ue/`, `giulia/mobility/`) — places UEs according to the chosen distribution and mobility model.
6. **Geometry** — distances and angles between every UE and every cell, as well as between each UE antenna element and each cell antenna element (`giulia/playground/distances_angles.py`).
7. **Channel modelling** — see [Channel generation](channel-generation.md) for the full procedure (LoS probability, K-factor, path loss, O2I penetration, shadowing, slow channel, LoS channel, fast fading, optional Sionna ray-traced channel, composite channel).
8. **Precoded channels** — SSB-precoded and CSI-RS-precoded channels per UE per beam (`giulia/channel/precoded_channel_gains.py`).
9. **RSS / RSRP** — received signal strength and reference-signal received power per UE per cell, both for SSB and CSI-RS (`giulia/kpis/calculate_rss.py`, `giulia/kpis/calculate_rsrp.py`).
10. **Cell association** — selects the best serving cell and beam per UE based on SSB and CSI-RS RSRP (`giulia/rrc/cell_selections.py`, `giulia/mac/calculate_best_serving_beam.py`).
11. **SINR and throughput** — interference accumulation, SINR per PRB, BLER lookup, and throughput per UE / per cell / per carrier (`giulia/kpis/calculate_sinr.py`, `giulia/kpis/calculate_rate.py`).
12. **Energy** — per-cell and network-level power consumption (`giulia/kpis/power_consumptions.py`).
13. **Output** — saves the requested fields as `.npz` files inside the configured output directory; the output module then converts them to Parquet and optionally zips the result (`giulia/outputs/`).

The pipeline runs once per snapshot. The number of snapshots is controlled by `args.snapshots` in the YAML config, and the random seed is incremented by one at every snapshot.

## Top-level packages under `giulia/`

- `giulia/playground/` — reference site placement, UE playground geometry, hotspot definitions, distances and angles between UEs and cells.
- `giulia/bs/` — network deployment, per-scenario base-station builders (in `bs_deployments.py`), and BS transmit-power computation. See the "Customizing base station parameters per scenario" section in [Running Giulia](running-giulia.md) to override per-scenario BS parameters.
- `giulia/ue/` — UE deployment (positions, indoor/outdoor flag, velocities) under different distributions.
- `giulia/antenna/` — antenna patterns, antenna arrays, array steering vectors.
- `giulia/channel/` — propagation and channel modelling: LoS probabilities, K-factor, path loss, O2I penetration loss, shadowing maps, shadowing gains, slow channel, fast fading, LoS channel, composite channel, optional Sionna channel, precoded channels, carriers (E-UTRA / NR ARFCN helpers).
- `giulia/rrc/` — radio resource control: time/frequency resources (PRB grid, OFDM numerology), MCS resources, beam configurations (SSB and CSI-RS codebooks), cell selection and reselection.
- `giulia/mac/` — MAC-layer scheduling and best-serving-beam selection.
- `giulia/phy/` — physical-layer abstractions: BLER-vs-SINR LUTs, mutual-information / MIESM mapping.
- `giulia/kpis/` — key performance indicators: noise, RSS, RSRP, SINR, throughput, and the network power-consumption model.
- `giulia/event_driven/` — the event-driven core (snapshot control, event scheduler) used by the presets to register pipeline stages as events.
- `giulia/mobility/` — UE mobility models (straight walk, circular walk).
- `giulia/presets/` — pipeline presets: `g_standard_preset.py` (GiuliaStd) and `g_multi_frequency_layer_preset.py` (GiuliaMfl).
- `giulia/inputs/` — YAML config loader and run-config dispatcher used by `main.py`.
- `giulia/outputs/` — output module: collects `.npz` files, filters fields, converts to Parquet, optionally compresses to a zip.
- `giulia/plots/` — plotting utilities used when `args.plots` is true.
- `giulia/tools/` — generic helpers (timers, logging utilities, memory utilities, unit conversions).
- `giulia/logger.py` — the simulator-wide logger.
- `giulia/config/` — the in-memory simulation configuration object (`Simulation_Config`) that carries all flags through the pipeline.

## Other top-level folders

- `examples/` — runnable example scripts dispatched from `InputConfig.run()`. The default is `examples/giulia_EV.py`.
- `data/` — input data assets used by data-driven scenarios (e.g. 3D meshes for ray tracing under `data/data_driven_extras/`).
- `shadowing/` — precomputed shadowing maps (`.mat` files) and the MATLAB scripts that generate them.
- `scripts/` — installer (`installer.sh`, `installer_dependencies.sh`), helper utilities (`download_shadowing_files.py`, `monitor_GPU.py`), and standalone GPU sanity-check scripts.
- `docs/` — this user documentation plus the Sphinx site under `docs/source/`.
- `tests/` — regression and unit tests.
- `outputs/` — default destination for simulation results.
- `plots/` — default destination for generated plots.
