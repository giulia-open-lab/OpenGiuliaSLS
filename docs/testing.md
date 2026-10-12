# Running Tests

Giulia's test suite is **regression-based**: it runs the simulator with a fixed configuration, then
compares every numeric KPI column against pre-computed golden reference files. If every column stays
within an absolute tolerance of `0.1`, the test passes.

The entry point is [`tests/run_tests.py`](../tests/run_tests.py). Reference files live in
`tests/references/` and are **version-controlled**, so a fresh checkout is ready to run out of the
box — nothing has to be generated or downloaded first.

> The full developer guide for the test suite, including how to add a scenario and how the harness
> works internally, is [`tests/README.md`](../tests/README.md). This page is the short version.

## Quick start

```shell
# List the available scenarios
python tests/run_tests.py --list

# Run one scenario
python tests/run_tests.py ITU_R_M2135_UMa_uniform

# Run all scenarios
python tests/run_tests.py

# Interactive mode: pick scenarios from a menu, watch live progress
python tests/run_tests.py --tui
```

The process exits with code `0` on success and `1` if any scenario fails or errors.

## Selecting scenarios

Pass scenario names as positional arguments, or use the `GIULIA_SCENARIOS` environment variable for
CI / headless runs (positional arguments take precedence):

```shell
GIULIA_SCENARIOS="ITU_R_M2135_UMa_uniform ITU_R_M2135_UMi_uniform" python tests/run_tests.py
```

Selecting a scenario runs it **once per preset** (`GiuliaStd` and `GiuliaMfl`), reported separately
as `<name>` and `<name>_mfl`. Single-carrier scenarios — any whose name contains `C1` or `C2` — run
under `GiuliaStd` only.

## Updating a reference

If you intentionally change simulator behaviour, regenerate the affected reference files, then re-run
without `--generate` to confirm the new baseline passes before committing:

```shell
python tests/run_tests.py ITU_R_M2135_UMa_uniform --generate
```

## Reports

Every run — including `--generate` and `--tui` — writes a self-contained HTML report to
`outputs/test_reports/report_<timestamp>.html` and prints its path as a clickable `file://` link when
the run finishes. The report contains the run date/time and options used, a collapsible section per
scenario with its full captured log, and, for comparison runs, a KPI table with expected/actual
means, max error and tolerance per column. Crashes show the extracted stack trace up front.

`outputs/` is gitignored, so reports are never committed.

For machine-readable output, use `--format json`.

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
| `PRECOMPUTED_FILES` | Path to the reference files directory (overridden by `--ref-dir`) |
| `SHADOWING_FILES` | Path to the shadowing `.mat` files directory |

## Adding a scenario

Create one YAML file in `tests/scenarios/` — no Python changes are needed — then generate its
reference files with `--generate`. See
[`tests/README.md`](../tests/README.md) for the file format and the details.

## Unit tests

The unit tests are independent of the regression suite, need no reference files, and run in under
a second. They are the tests the CI runs on every push
([`.github/workflows/unit-tests.yml`](../.github/workflows/unit-tests.yml)):

```shell
python -m unittest tests.unit_tests tests.test_saveable_registry
```

- [`tests/unit_tests.py`](../tests/unit_tests.py) — the logger and the Parquet comparison helpers.
- [`tests/test_saveable_registry.py`](../tests/test_saveable_registry.py) — the Saveable registry's
  behaviour under threading. For why it needs its own suite and what each test pins, see
  [Testing the Saveable registry](output-saveable/saveable-registry-tests.md).

## Saveable pipeline tests

Separate from both suites above: these run **real minimal simulations** and check what the Saveable
system writes to disk — the export files and their columns, the `output.fields` filter, parallel and
multi-process runs, the live `Monitor`, and the catalog generator. They need the data files and take
seconds to minutes, so they are **not** part of the per-push unit tests:

```shell
# fast half (~15 s)
python -m unittest tests.test_saveable_pipeline

# concurrency, monitor, subprocesses, catalog, enable/disable (~3 min)
python -m unittest tests.test_saveable_pipeline_concurrency
```

Two results are reported as `expected failure`: known, pre-existing bugs pinned on purpose. Full
guide, fixture API and CI recommendation:
[Testing the Saveable pipeline](output-saveable/saveable-pipeline-tests.md).
