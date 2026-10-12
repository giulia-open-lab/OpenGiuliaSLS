# Testing the Saveable pipeline

How to run — and how to extend —
[`tests/test_saveable_pipeline.py`](../../tests/test_saveable_pipeline.py) and
[`tests/test_saveable_pipeline_concurrency.py`](../../tests/test_saveable_pipeline_concurrency.py):
the tests that run **real Giulia simulations** and check what the Saveable system actually
writes to disk.

Companion page: [Testing the Saveable registry](saveable-registry-tests.md), which covers
[`tests/test_saveable_registry.py`](../../tests/test_saveable_registry.py) — the fast, simulation-free
tests of the registry primitives.

---

## 1. Why there are two suites

They answer different questions, and neither can answer the other's.

| | `test_saveable_registry.py` | `test_saveable_pipeline*.py` |
|---|---|---|
| What it uses | synthetic `Saveable` subclasses | real minimal simulations |
| What it proves | the registry **primitives** behave: isolation, `share_registry`, the enable flag, "latest wins" | the **pipeline around them** works: collection, routing, export, the files on disk |
| Runtime | well under a second | seconds to minutes |
| Needs data files | no | yes (shadowing maps, `data/`) |

The registry suite can pass completely while the export writes nothing at all — it never calls
the exporters. That is not hypothetical: the two places that read the registry
(`Performance.process` and `Episode_Performance._save_episodes_performance_parquet`) had never
been *executed* against the per-context registry until this suite existed. And an empty export
is silent — an absent parquet looks exactly like a label nobody routed anything to.

So: the registry suite pins the mechanism, this one pins the outcome.

---

## 2. Running them

From the repository root, in the project conda environment:

```bash
# The fast half: one-run export, exploded files, the fields filter, latest-wins  (~15 s)
python -m unittest tests.test_saveable_pipeline -v

# Concurrency, monitor, subprocesses, catalog, enable/disable                    (~3 min)
python -m unittest tests.test_saveable_pipeline_concurrency -v

# Everything
python -m unittest tests.test_saveable_pipeline tests.test_saveable_pipeline_concurrency
```

Expect **two `expectedFailure` results** — they are known bugs, deliberately pinned; see
[§6](#6-the-two-pinned-known-bugs).

The slow part is `EnableDisableInAngerTests` (about two minutes): it needs the cell-pairing
sub-run to genuinely *recompute*, and a cached pairing result would skip the code under test.

### The stress knob

`ThreadStressTests` and `SeparateProcessTests` each run **8** simulations at once by default.
Raise it to lean on a particular machine:

```bash
GIULIA_SAVEABLE_STRESS_RUNS=32 python -m unittest \
    tests.test_saveable_pipeline_concurrency.ThreadStressTests \
    tests.test_saveable_pipeline_concurrency.SeparateProcessTests
```

Verified on a 32-core / 125 GB box: 8 runs ≈ 15 s, 16 ≈ 36 s, 32 ≈ 128 s — all passing, in
threads and in processes alike.

---

## 3. What each class pins

### `tests/test_saveable_pipeline.py`

| Class | The promise it protects | What breaks if it regresses |
|---|---|---|
| `SingleRunExportTests` | a real run's Saveables are readable **at export time**: every label in use has a `.parquet` and a `.schema.md`, columns are `<Class>.<variable>`, values are populated and finite, NPZ still carries the core KPIs, and `BaseStations.prb_usage_per_cell` matches the hardcoded core KPI exactly | the export reads an empty registry and writes nothing — silently |
| `ExplodedFileTests` | per-entity files: row count = entities × steps × episodes, dense entity ids, 2-D variables split one column per element, prefixes come from `expose_variable_column_prefixes()` | wrong row counts, or a matrix collapsing into one unreadable column |
| `FieldsFilterTests` | `output.fields`: index columns always survive, a base name keeps every expanded column, a single `.<n>` keeps only that one, a label left with no data columns produces no file at all | the filter drops index columns, or leaves behind empty files |
| `LatestWinsAcrossEpisodesTests` | each episode's rows come from **that episode's** Saveable instances | exported data freezes at episode 0 |
| `GenericLabelExportTests` | the `generic` file and both fallbacks that fill it ([§2 of Saveable Output](saveable-output.md#when-the-generic-file-is-written)) | the core KPIs quietly stop reaching Parquet |
| `NewLabelExportTests` | a brand-new label auto-creates its parquet with an entity-id derived from the label name; a 2-D variable with no declared prefix falls back to `.0/.1/…` | adding a per-entity output stops being zero-code |

### `tests/test_saveable_pipeline_concurrency.py`

| Class | The promise it protects | What breaks if it regresses |
|---|---|---|
| `ParallelSimulationTests` | two runs with **different deployments** on two threads keep their own registries, files and numbers | the original bug: one run exports another's columns and row counts |
| `ThreadStressTests` | the same under load — 8 contexts alive at once, all registered simultaneously, all exporting simultaneously | eviction under contention that a two-run test would miss |
| `MonitorDuringRealRunTests` | `Monitor` reads live values through `share_registry`, follows the object across episodes, and does **not** report a valid KPI as "never available" | the Stage-1 trap: every KPI reads `None` and is blamed on a typo |
| `SeparateProcessTests` | 8 spawned processes each export their own deployment, and the runs really overlap | the isolation model the catalog generator depends on |
| `CatalogGeneratorTests` | `scripts/generate_saveable_catalog.py` produces a **non-empty** catalog for both presets, neither contaminated by the other | an empty catalog regenerated over the real one, without an error |
| `EnableDisableInAngerTests` | the real `Saveable.disable()` call sites: the cell-pairing sub-run restores the flag on its own thread, and cannot silence a run in parallel | a run that never touched pairing exports empty KPI parquets |

---

## 4. The fixture API

Everything shared lives in
[`tests/saveable_pipeline_utils.py`](../../tests/saveable_pipeline_utils.py).

### `run_simulation(...)`

A parameterised copy of `examples/giulia_EV.py::main`: same objects, same order, but with the
episode and step counts turned down (the example hardcodes three steps). It builds the same
`Episode_Performance`, constructs the preset, and calls `configure` / `run_simulation` /
`save_episode_performance` / `save_episodes_performance_in_file`.

```python
result = run_simulation(
    scenario='ITU_R_M2135_UMa',     # deployment
    preset='GiuliaStd',             # or 'GiuliaMfl'
    episodes=1, steps=2,
    output_dir=tmp_dir,             # must be a temp directory
    fields=None,                    # the output.fields filter
    npz=False,
)
result.read('ue_kpi')               # -> DataFrame
result.ue_count, result.cell_count, result.ru_count, result.bbu_count
```

Three hooks let a test observe the run from the inside, which is the only way to compare a
file against what the registry actually held:

| Hook | Fires | Typical use |
|---|---|---|
| `after_configure(episode_index, g)` | right after `g.configure()` — this episode's Saveables have just registered | register a probe Saveable; snapshot instance identity; park on a barrier |
| `after_episode(episode_index, g)` | after the episode's buffers are copied into `Episode_Performance` | snapshot `Saveable.get(...)` to compare against the exported rows |
| `before_export()` | immediately before `save_episodes_performance_in_file` — the moment the exporter reads the registry | make several runs export *simultaneously* |

### `run_in_isolated_context(fn)`

Runs `fn` on its own thread, so it gets its own registry.

**Every test class that runs a simulation uses this**, because the registry is never cleared
within a context ([Saveable Concurrency §9](saveable-concurrency.md#9-caveats)): without it,
one class's Saveables — especially synthetic probes — turn up in a later class's
`Saveable.registry()` and results start depending on test order.

The consequence to remember when writing a test: **anything read from the registry must be
captured inside `fn`.** From the test methods, that registry is invisible.

```python
def simulate():
    result = run_simulation(...)
    return result, registered_labels(), np.asarray(Saveable.get(KPI))   # captured in-context

cls.result, cls.labels, cls.expected = run_in_isolated_context(simulate)
```

### The rest

| Helper | What it does |
|---|---|
| `registered_labels()` | `{label: ['<Class>.<var>', …]}` resolved exactly as the exporter resolves it — assert on the files a run *must* have produced without hardcoding a list that drifts |
| `written_labels(dir, project)` | the labels that actually have a `.parquet` on disk |
| `column_family(df, base)` | a column and its `.<suffix>` split columns |
| `index_columns(label)` | the index columns the exporter always keeps |
| `npz_output_redirected_to(dir)` | NPZ ignores `output.directory` by design ([output-module.md §3](output-module.md#npz)); this redirects the one path helper so tests never write into the repo's `outputs/` |
| `captured_warnings(module)` | collects what a module sends to `giulia.logger.warning` (patched on the *importing* module, since it does `from giulia.logger import warning`) |
| `requires_giulia` | skips with the import error when torch / pyarrow / sionna are unavailable |
| `DEFAULT_SCENARIO`, `SMALL_SCENARIO` | 570 UEs / 57 cells, and 10 UEs / 3 cells |

---

## 5. Rules these tests follow

Keep to them when adding a case.

1. **Never modify production code to make a test pass.** If a test cannot be written without a
   change under `giulia/`, that is a finding — report it. The two known bugs in
   [§6](#6-the-two-pinned-known-bugs) are recorded that way rather than papered over.
2. **All output goes to a temp directory.** Parquet through `output_dir`; NPZ through
   `npz_output_redirected_to`, because it resolves its own path and would otherwise land in
   the repository's `outputs/results/`. Runs use `save_results=0`, `plots=0`, `enable_GPU=False`.
3. **Every simulation runs in an isolated context** (§4), so classes cannot leak into one
   another and can be run in any order or on their own.
4. **Restore anything global.** `CatalogGeneratorTests` rewrites
   `docs/output-saveable/saveable-catalog.md` — that is what the script does — and puts the
   original back byte-for-byte in `tearDownClass`.
5. **Keep runs minimal**: 1 episode, 1–2 steps, CPU. Reuse one run for several assertions via a
   class-level fixture instead of simulating per test.
6. **Make deployments differ.** The concurrency tests pair scenarios with different
   `(ues, cells, radio units, BBUs)` fingerprints, so a leak shows up as a wrong row count
   rather than as plausible-looking numbers.
7. **Force the path you mean to test.** `EnableDisableInAngerTests` points the working
   directory at an empty temp dir because `PairingCell` caches under the relative path
   `./data/pairing_sym_results` — a cache hit would skip the sub-run entirely and the test
   would pass without testing anything.

---

## 6. The two pinned known bugs

Both are marked `@unittest.expectedFailure`, so the suite is green today and reports an
**unexpected success** — which `unittest` counts as a failing run — the moment either is fixed.
That is the signal to delete the marker and turn the test into a normal one.

| Test | Bug |
|---|---|
| `GenericLabelExportTests.test_a_two_d_variable_in_generic_is_populated` | a 2-D variable routed to `generic` exports **all-null** columns: the schema registers the indexed names `<Class>.<var>.0/.1/…` while the per-step buffer is keyed by the base name, so the writer finds nothing |
| `GenericLabelExportTests.test_every_collected_core_kpi_reaches_the_generic_file` | `PRB_usage_per_cell` and `power_consumption_per_radio_unit_kW` are collected every step and written into the `generic` record, but have no `pa.field(...)` in the hardcoded `generic` schema, so `pa.Table.from_pylist` drops them. No data is lost — both reach Parquet as Saveables (`Power_Consumption.power_consumption_per_radio_unit_kW` → `ru_kpi`, `BaseStations.prb_usage_per_cell` → `cell_kpi`) — so this is a gap in the wide `generic` table only |

Both pre-date the per-context registry and are only reachable on the `generic` path, which no
shipped Saveable currently takes — see
[Saveable Output §2](saveable-output.md#when-the-generic-file-is-written).

---

## 7. Choosing what to run in CI

The unit tests CI runs today are simulation-free and finish in milliseconds
([`.github/workflows/unit-tests.yml`](../../.github/workflows/unit-tests.yml)):

```bash
python -m unittest tests.unit_tests tests.test_saveable_registry
```

These pipeline tests need the data files and real runtime, so they are a separate decision:

| Tier | What | Cost |
|---|---|---|
| **Per push** | `tests.test_saveable_pipeline` + `ParallelSimulationTests` + `MonitorDuringRealRunTests` + `ThreadStressTests` + `SeparateProcessTests` | ~40 s |
| **Nightly** | `CatalogGeneratorTests` (rewrites and restores `docs/`) and `EnableDisableInAngerTests` (needs a cold pairing cache) | ~2.5 min |

---

## 8. Troubleshooting

| Symptom | Cause |
|---|---|
| every test skipped | Giulia could not be imported — `requires_giulia` prints the reason; use the project conda environment |
| `AssertionError: variables … disagree on their entity count` | two variables routed to one exploded label have different first-axis lengths; the exporter's majority filter would drop one. Fix the label, not the test |
| `Saveable.get(...)` returns `None` in a test method | it is running in the runner's context, not the run's. Capture it inside `run_in_isolated_context` (§4) |
| `EnableDisableInAngerTests` finishes suspiciously fast and its cache assertion fails | the pairing sub-run was skipped by a cache hit; the temp working directory is not being applied |
| a concurrency test hangs | a worker died before reaching a barrier. Barriers carry timeouts and are aborted on error, so this points at a real crash — read the reported exception rather than re-running |
