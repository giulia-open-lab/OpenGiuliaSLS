# Architecture

This page describes how Giulia is organised and how a simulation flows from the entry point through the main pipeline stages.

## Entry point and configuration flow

The top-level entry point is [`main.py`](../main.py) at the repository root. It does four things:

1. Adds the repo root to `sys.path` so that `import giulia` and `import examples` resolve.
2. Reads the YAML config file into an `InputConfig` (`giulia/inputs/input_config.py`) — defaulting to `config.yml` next to `main.py`, or a path supplied as the first command-line argument. If `config.yml` does not exist, it is pre-filled from `config.example.yml`.
3. Starts a background `Monitor` thread that polls a chosen list of KPIs while the run proceeds (see [Saveable Monitor](output-saveable/SaveAble_monitor.md)).
4. Calls `InputConfig.run()`, then invokes what it returns, on the OS main thread.

`InputConfig.run()` does **not** execute the simulation. It translates the `args` block of the YAML into command-line-style flags (e.g. `--scenario_model=ITU_R_M2135_UMa`), resolves the `run.example` key to the matching example's entry point, and returns the `(callable, args)` pair. `main.py` then calls `giulia_main(args, config)`. The simulation must stay on the OS main thread because its C++ backends (PyTorch / Mitsuba) need to be initialised and torn down there.

The example script is the one that instantiates the simulator and drives the snapshot loop. Writing the results is part of that run, not a separate post-processing pass: `Performance` collects every Saveable's exposed variables each step, and `Episode_Performance` writes the enabled formats at the end (`giulia/outputs/`). NPZ and Parquet are produced **independently** according to `output.formats` — Parquet is not a conversion of the `.npz` files.

The YAML file has three sections: `args` (simulator behaviour, scenario, UEs, snapshots, GPU, logging), `run` (which example to dispatch to), and `output` (target directory, output formats, list of fields to export). See [Running Giulia](running-giulia.md) for the full schema.

## Presets

Giulia exposes two presets that wire up the components of the pipeline differently:

- **GiuliaStd** — the standard preset, built in `giulia/presets/g_standard_preset.py`. Used for single-frequency-layer scenarios and for most 3GPP/ITU calibration cases.
- **GiuliaMfl** — the multi-frequency-layer preset, built in `giulia/presets/g_multi_frequency_layer_preset.py`. Used for scenarios that combine 4G/5G/6G layers or mix UMa/UMi sites.

The preset is selected via `args.preset` in the YAML config and resolved in `examples/giulia_EV.py`.

## Simulation pipeline

A simulation runs as a sequence of pipeline stages. The standard preset wires them up in `giulia/presets/g_standard_preset.py`; the multi-frequency-layer preset adds extra layering on top.

Two things happen in the preset's `__init__`, before any event exists: the network deployment is processed (which assigns each cell a `coverage` / `capacity` role from its carrier frequency), and `PairingCell.process()` runs a dedicated throw-away simulation to decide which coverage cell backs each capacity cell. Both write into `df_ep`, and both are done by the time the constructor returns — see [Pairing](energy-efficiency/pairing.md).

The per-snapshot flow is then:

1. **Playground** (`giulia/playground/`) — builds the geometric playground: the reference site grid, the UE deployment area, and any hotspot regions.
2. **Network deployment** (`giulia/bs/`) — instantiates the network, places base stations on the playground, and assigns per-cell parameters (carrier frequency, bandwidth, antenna height, sector bearings, transmit power, and so on). The exact parameters are picked from the per-scenario builders in `giulia/bs/bs_deployments.py` (see also the "Customizing base station parameters per scenario" section in [Running Giulia](running-giulia.md)).
3. **Antennas and beams** (`giulia/antenna/`, `giulia/rrc/beam_configs.py`) — builds antenna arrays for cells and UEs, computes antenna pattern gains, derives array steering vectors, and creates the SSB and CSI-RS beam codebooks.
4. **Time-frequency resources and link-level setup** (`giulia/rrc/`, `giulia/phy/`, `giulia/mac/`) — defines the DL/UL carriers, PRB grid, MCS tables, BLER-vs-SINR look-up tables, and the SINR mapping (e.g. MIESM).
5. **UEs** (`giulia/ue/`, `giulia/mobility/`) — places UEs according to the chosen distribution and mobility model.
6. **Geometry** — distances and angles between every UE and every cell, as well as between each UE antenna element and each cell antenna element (`giulia/playground/distances_angles.py`).
7. **Channel modelling** — see [Channel generation](channel-generation.md) for the full procedure (LoS probability, K-factor, path loss, O2I penetration, shadowing, slow channel, LoS channel, fast fading, optional Sionna ray-traced channel, composite channel).
8. **Precoded channels** — SSB-precoded and CSI-RS-precoded channels per UE per beam (`giulia/channel/precoded_channel_gains.py`).
9. **RSS / RSRP** — received signal strength and reference-signal received power per UE per cell, both for SSB and CSI-RS (`giulia/kpis/calculate_rss.py`, `giulia/kpis/calculate_rsrp.py`).
10. **Cell association** — selects the best serving cell and beam per UE from SSB and CSI-RS RSRP (`giulia/rrc/cell_selections.py`, `giulia/mac/calculate_best_serving_beam.py`), then overrides it with the frequency-priority re-selection ladder (`giulia/rrc/cell_re_selections.py`). See [RRM](energy-efficiency/rrm.md).
11. **Scheduling** — PRB requirement per UE and the per-cell, per-beam allocation that produces the `_prb_usage` load figure every later stage reads (`giulia/mac/prb_requirement_calculators.py`, `giulia/mac/schedulers.py`).
12. **SINR and throughput** — interference accumulation, SINR per PRB, BLER lookup, and throughput per UE / per cell / per carrier (`giulia/kpis/calculate_sinr.py`, `giulia/kpis/calculate_rate.py`).
13. **Energy** — per-cell and network-level power consumption (`giulia/kpis/power_consumptions.py`).
14. **Radio resource control / energy saving** — per-cell `Carrier_Shutdown` decisions: a lightly loaded capacity carrier is switched off, or a dormant one woken, by setting transmit power (`giulia/rrc/rrcs.py`, `giulia/bs/bs_tx_powers.py`). Scheduled last in the step, so its effect lands on the next one. Disabled by default — see [Energy efficiency](energy-efficiency/README.md).
15. **Output** — every registered Saveable's exposed variables are collected each step and written, at the end of the episode, in each enabled format: `.npz` archives and/or Parquet datasets grouped by label, optionally zipped into a `.gof` archive (`giulia/outputs/`). See [Saveable Output & Export](output-saveable/saveable-output.md).

The pipeline runs once per snapshot. The number of snapshots is controlled by `args.snapshots` in the YAML config, and the random seed is incremented by one at every snapshot.

## Top-level packages under `giulia/`

- `giulia/playground/` — reference site placement, UE playground geometry, hotspot definitions, distances and angles between UEs and cells.
- `giulia/bs/` — network deployment, per-scenario base-station builders (in `bs_deployments.py`), and BS transmit-power computation. See the "Customizing base station parameters per scenario" section in [Running Giulia](running-giulia.md) to override per-scenario BS parameters.
- `giulia/ue/` — UE deployment (positions, indoor/outdoor flag, velocities) under different distributions.
- `giulia/antenna/` — antenna patterns, antenna arrays, array steering vectors.
- `giulia/channel/` — propagation and channel modelling: LoS probabilities, K-factor, path loss, O2I penetration loss, shadowing maps, shadowing gains, slow channel, fast fading, LoS channel, composite channel, optional Sionna channel, precoded channels, carriers (E-UTRA / NR ARFCN helpers).
- `giulia/rrc/` — radio resource control and management: time/frequency resources (PRB grid, OFDM numerology), MCS resources, beam configurations (SSB and CSI-RS codebooks), cell selection and reselection, plus the two energy-efficiency modules — `rrcs.py` (the per-cell `RRC` control object and the `Carrier_Shutdown` state machine) and `cell_pairing.py` (coverage↔capacity pairing). See [RRM](energy-efficiency/rrm.md) and [The RRC layer](energy-efficiency/rrc-layer.md).
- `giulia/mac/` — MAC-layer scheduling and best-serving-beam selection. `schedulers.py` also defines `_prb_usage`, the per-cell load figure the energy-saving decisions and the power model both run on.
- `giulia/phy/` — physical-layer abstractions: BLER-vs-SINR LUTs, mutual-information / MIESM mapping.
- `giulia/kpis/` — key performance indicators: noise, RSS, RSRP, SINR, throughput, and the network power-consumption model (`power_consumptions.py`, which turns each radio unit's on/off state and load into watts — see [Carrier shutdown § 6](energy-efficiency/carrier-shutdown.md#6-what-it-actually-saves)).
- `giulia/event_driven/` — the event-driven core (snapshot control, event scheduler) used by the presets to register pipeline stages as events. Every stage defaults to the same 1e6 µs cadence, so events sharing a timestamp run in **insertion order**; a stage that returns `-1` is removed from the pile.
- `giulia/mobility/` — UE mobility models (straight walk, circular walk).
- `giulia/presets/` — pipeline presets: `g_standard_preset.py` (GiuliaStd) and `g_multi_frequency_layer_preset.py` (GiuliaMfl).
- `giulia/inputs/` — YAML config loader and run-config resolver used by `main.py`, including the per-format output configuration objects under `giulia/inputs/config/`.
- `giulia/outputs/` — the Saveable registry and the output pipeline: per-step collection (`save_performance.py`), end-of-episode writing to NPZ / Parquet with the `fields` filter applied (`save_episode_performance.py`), the live KPI `Monitor`, and the `OutputReader` helper for reading results back.
- `giulia/plots/` — plotting utilities used when `args.plots` is true.
- `giulia/tools/` — generic helpers (timers, logging utilities, memory utilities, unit conversions).
- `giulia/fs/` — filesystem layout helpers: resolving input, output and shadowing paths.
- `giulia/assertions/` — shared sanity checks for paths and auxiliary input files.
- `giulia/logger.py` — the simulator-wide logger. Note that its levels run **from verbose to quiet**: `0` = VERBOSE, `4` = ERROR.
- `giulia/config/` — the in-memory simulation configuration object (`Simulation_Config`) that carries all flags through the pipeline, including the event-scheduling cadences.

## Other top-level folders

- `examples/` — runnable example scripts resolved by `InputConfig.run()`. The default is `examples/giulia_EV.py`.
- `data/` — input data assets used by data-driven scenarios (e.g. 3D meshes for ray tracing under `data/data_driven_extras/`), plus cached pairing results.
- `shadowing/` — precomputed shadowing maps (`.mat` files, downloaded separately and not tracked by Git or Git LFS) and the MATLAB scripts that generate them.
- `scripts/` — installer (`installer.sh`, `installer_dependencies.sh`), helper utilities (`download_shadowing_files.py`, `monitor_GPU.py`, `generate_saveable_catalog.py`), and standalone GPU sanity-check scripts.
- `docs/` — this user documentation plus the Sphinx site under `docs/source/`.
- `tests/` — the regression harness (`tests/run_tests.py`), its scenario definitions and version-controlled reference files, and the unit tests. See [Testing](testing.md).
- `outputs/` — default destination for simulation results and test reports (gitignored).
- `plots/` — plotting and result-post-processing scripts.
- `dedalo/` — **the configuration interface to Giulia**: the layer through which an external script, optimizer or agent modifies a live simulation, rather than reaching into the simulator's internals. It is not on the `main.py` / YAML path described above — a config-driven run never constructs it — but it is the documented write surface for external optimization and agent workflows (Bayesian optimization, reinforcement learning, parameter sweeps). See [Dedalo](dedalo/README.md).
- `optimizer/` — a standalone research sub-project built on top of Giulia; not part of the simulation pipeline described above.
