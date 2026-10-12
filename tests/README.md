# Regression Tests

Giulia's test suite is **regression-based**: run the simulator with a fixed configuration, then compare every numeric KPI column against pre-computed golden reference files.
If every column stays within an absolute tolerance of `0.1`, the test passes.

## Quick start

```shell
# List available scenarios
python tests/run_tests.py --list

# Run one scenario
python tests/run_tests.py ITU_R_M2135_UMa_uniform

# Run all scenarios
python tests/run_tests.py

# Interactive mode: choose scenarios from a menu, watch live progress
python tests/run_tests.py --tui
```

## Workflows

Reference files live in `tests/references/` and are version-controlled, so a fresh checkout is ready to run out of the box.

### CI / headless mode

```shell
# Restrict to specific scenarios via environment variable
GIULIA_SCENARIOS="ITU_R_M2135_UMa_uniform ITU_R_M2135_UMi_uniform" python tests/run_tests.py

# Machine-readable output
python tests/run_tests.py --format json
```

The process exits with code `0` on success and `1` if any scenario fails or errors.

### HTML report

Every run (including `--generate` and `--tui`) writes a self-contained HTML report to `outputs/test_reports/report_<timestamp>.html` and prints its path (as a clickable `file://` link) once the run finishes:

```
HTML report: file:///home/you/GiuliaSLS/outputs/test_reports/report_20260706_153000.html
```

The report includes the run date/time and options used, a collapsible section per scenario with its full captured log, and — for comparison runs — a KPI table showing expected/actual means, max error, and tolerance for each column. Crashes show the extracted stack trace up front. It's written under `outputs/`, which is gitignored, so it's never committed.

### Updating a reference

If you intentionally change simulator behaviour, update the affected reference file(s):

```shell
python tests/run_tests.py ITU_R_M2135_UMa_uniform --generate
```

Then re-run without `--generate` to confirm the new baseline passes before committing.

## Options reference

| Flag | Default | Description |
|------|---------|-------------|
| `SCENARIO ...` | all | One or more scenario names to run |
| `--generate` | off | Write simulation outputs as new reference files instead of comparing |
| `--tolerance FLOAT` | `0.1` | Override the absolute error tolerance |
| `--ref-dir PATH` | `tests/references/` | Reference files directory |
| `--output-dir PATH` | `outputs/test_runs/` | Where simulation outputs land during comparison runs |
| `--download` | prompt | Download missing shadowing files without asking |
| `--no-download` | prompt | Raise an error if shadowing files are missing instead of prompting |
| `--tui` | off | Interactive scenario selector + live progress table (requires a TTY) |
| `--list` | off | Print scenario names and exit |
| `--format text\|json` | `text` | Output format for results |

Environment variables:

| Variable | Description |
|----------|-------------|
| `GIULIA_SCENARIOS` | Space-separated list of scenarios (overridden by positional args) |
| `PRECOMPUTED_FILES` | Path to reference files directory (overridden by `--ref-dir`) |
| `SHADOWING_FILES` | Path to shadowing `.mat` files directory |

## Adding a new scenario

Create one YAML file in `tests/scenarios/`. No Python changes are needed.

```yaml
# tests/scenarios/my_new_scenario.yaml
name: my_new_scenario           # must equal scenario_model + "_" + ue_distribution

args:
  scenario_model: ITU_R_M2135_UMa
  ue_distribution: uniform
  snapshots: 2
  regression: true
  save_results: 1
  plots: false

run:
  example: EV
```

Don't set `args.preset` — it's ignored. Selecting `my_new_scenario` (by `--list`, a
positional arg, `GIULIA_SCENARIOS`, or the `--tui` picker) implicitly runs it once
per preset in `tests.scenario.PRESETS` (currently `GiuliaStd` and `GiuliaMfl`) —
internally as `my_new_scenario` (GiuliaStd) and `my_new_scenario_mfl` (GiuliaMfl),
each compared and reported separately, but there's only ever one scenario to pick.
The only exception is single-carrier scenarios — any scenario whose name contains
`C1` or `C2` is only run under GiuliaStd, since MFL doesn't apply to them (see
`tests.scenario.supports_preset`).

Then generate reference files for it (both preset variants at once):

```shell
python tests/run_tests.py my_new_scenario --generate
```

## Directory layout

```
tests/
├── scenarios/          # one YAML file per scenario (version-controlled)
├── references/         # golden Parquet files (version-controlled)
│
├── run_tests.py        # entry point
├── scenario.py         # ScenarioConfig dataclass + YAML loader; expands each file into one run per preset
├── runner.py           # runs one scenario in a subprocess (GPU isolation)
├── compare.py          # discovers and compares numeric KPI columns common to both datasets
├── references.py       # reference lookup + shadowing file download management
├── report.py           # Rich-based terminal output
├── tui.py              # interactive scenario selector + live progress table
│
├── system_info.py      # host/GPU details recorded in the report
├── html_report.py      # HTML rendering of a test run
│
├── unit_tests.py                      # unittest suite for the logger module
├── test_saveable_registry.py          # unittest suite for the Saveable registry
├── test_saveable_pipeline.py          # end-to-end Saveable output pipeline
├── test_saveable_pipeline_concurrency.py  # per-context registry isolation
├── saveable_pipeline_utils.py         # helpers shared by the two pipeline suites
│
├── network_df_ep_check_list.py   # consistency checker for df_ep; also backs Dedalo.check_df_ep
└── network_df_ep_update.py       # recomputes derived fields in df_ep; also backs Dedalo.repair_df_ep
```

## How it works internally

1. **Scenario discovery** — `load_all_scenarios(tests/scenarios/)` reads every `*.yaml` file and expands each into one `ScenarioConfig` per applicable preset (see "Adding a new scenario" above). `--list`, `GIULIA_SCENARIOS`, positional args, and the `--tui` picker all operate on `base_names()` instead — the per-preset expansion stays internal to running and reporting.
2. **Reference check** — before any simulation runs, `ensure_reference()` confirms the reference Parquet directory exists and contains at least one file. If not, it raises `FileNotFoundError` pointing you at `--generate`.
3. **Simulation** — `run_scenario()` spawns an `mp.Process` that constructs an `InputConfig` directly from the scenario's `args` dict and calls the selected example entry point (`giulia_EV.main`). The subprocess pattern isolates GPU memory between runs.
4. **Comparison** — `compare_scenario()` loads both the new output and the reference as `pyarrow.dataset` objects, then discovers every numeric column common to both schemas (excluding `episode_idx`/`step_idx` and entity id columns like `ue_id`, `cell_id`) and computes element-wise absolute error for each. A column passes if no element exceeds `atol`.
5. **Reporting** — `report.py` renders a Rich table; `tui.py` drives a live-updating table during the run.

## Unit tests

The logger unit tests are independent and can be run at any time:

```shell
python -m unittest tests/unit_tests.py
```
