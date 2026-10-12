# Saveable Output & Export

This document explains **how the Saveable system turns simulation results into output
files** — what files are produced, how each row and column is built, and how you control
where a variable ends up.

It is the reference to read when you want a KPI to appear in the output (or to understand
an existing one). For the surrounding topics, see:

- [Saveable Objects](../source/saveable-objects.rst) — how a class *becomes* a Saveable.
- [Saveable Catalog](saveable-catalog.md) — the list of every variable available per preset.
- [Saveable Monitor](SaveAble_monitor.md) — reading a Saveable value *live* during a run.
- [Saveable Concurrency](saveable-concurrency.md) — which registry a call sees, and how parallel runs stay separate.
- [Reading Parquet Output](reading-parquet.md) — how to *read* the files described here back into pandas.

---

## 1. The big picture

A **Saveable** is any class that registers itself so Giulia can read and export its
results. The output pipeline has three stages:

| Stage | When | What happens | Code |
|-------|------|--------------|------|
| **Register** | when the class is instantiated | `super().__init__()` appends the object to the registry of the running simulation's execution context (de-duplicated by class name). | [`saveable.py`](../../giulia/outputs/saveable.py) |
| **Collect** | every step | For each Saveable, every variable in `expose_variables()` is read and appended to a per-step buffer (`Class.var → [value_step0, value_step1, …]`). | [`save_performance.py`](../../giulia/outputs/save_performance.py) |
| **Export** | end of run | The buffered data is written to Parquet files, grouped by **label**. | [`save_episode_performance.py`](../../giulia/outputs/save_episode_performance.py) |

The collect stage is label-agnostic: it reads *everything* in `expose_variables()`. The
**label** only decides *which file* each variable lands in during export.

---

## 2. The output files

Every Saveable variable is routed to one Parquet dataset, named `<project_name>_<label>`.
There is one file per **label in use** — `generic` plus one per **exploded** label:

| File | Label | One row per… | A column holds… |
|------|-------|--------------|-----------------|
| `..._generic` | `generic` (default) | `(episode, step)` | the **whole vector/array** for that step (a list) |
| `..._ue_kpi` | `ue_kpi` | `(episode, step, ue_id)` | **one scalar for that UE** |
| `..._cell_kpi` | `cell_kpi` | `(episode, step, cell_id)` | **one scalar for that cell** |
| `..._ru_kpi` | `ru_kpi` | `(episode, step, ru_id)` | **one scalar for that radio unit** |

The exploded labels (`ue_kpi`, `cell_kpi`, `ru_kpi`, …) are **"exploded"** files: a
length-*N* array is unpacked into *N* rows, one per entity, and an extra entity-id column
(`ue_id` / `cell_id` / `ru_id`) is added so each row is traceable to a specific entity.
`generic` keeps the array whole on a single row.

Exploded labels are **dynamic**: any label other than `generic` is exported exploded and gets
its own file automatically — no registry to edit. The entity-id column is derived from the
label name (`ue_kpi → ue_id`, `bbu_kpi → bbu_id`); see
[§5](#5-adding-a-new-per-entity-parquet).

> Each dataset is written together with a `<dataset>.schema.md` sidecar that lists every
> column and its type — the quickest way to discover what a file contains.

### Column naming

A Saveable column is always named `<ClassName>.<variable_name>`, e.g.
`SINR_CRS.sinr_ue_to_cell_dB`. The index columns (`episode_idx`, `step_idx`, and the
entity-id for exploded files) are added automatically.

### Empty files are not created

If an exploded label ends up with **no data columns** — because no variable is routed to
it, or because the [`fields` filter](#6-selecting-which-columns-to-export-the-fields-filter)
removed them all — the file is **skipped entirely**: no `.parquet`, no `.schema.md`, not even
an empty folder.

### When the `generic` file is written

`generic` follows the same rule as every other label: **it is written if — and only if — at
least one Saveable variable resolves to it.** It is not produced unconditionally.

A variable resolves to `generic` through either of the two fallbacks in
[§3](#the-resolution-order-and-the-fallback):

| Fallback | When | Logged? |
|---|---|---|
| **class level** | the class overrides neither `label` nor `expose_variable_labels()`, so it inherits the base `label` of `generic` | **no** — the base `expose_variable_labels()` maps every variable explicitly, so nothing looks unusual |
| **per variable** | the class *does* override `expose_variable_labels()` but leaves a variable out of the mapping, so it falls back to the class `label` | **yes** — a warning naming the variable and the file it fell into |

This matters more than it looks, because `generic` is the **only** route the hardcoded core
KPIs have into Parquet. `save_performance.py` collects them on a separate code path
(`cell_throughput_Mbps`, `best_serving_CRS_rsrp_per_ue_dBm`,
`total_network_power_consumption_kW`, …) and `_save_episodes_performance_parquet` holds a
hardcoded `generic` schema for them — but that schema is only *written* when the `generic`
label is in play. So:

> **In a stock run, no `<project>_generic.parquet` is produced.** Every variable both shipped
> presets expose resolves to an exploded label (`ue_kpi`, `cell_kpi`, `ru_kpi`, `bbu_kpi`,
> `ssbBeam_kpi`, `csiRsBeam_kpi`, `crsBeam_kpi`) — some through an `expose_variable_labels()`
> entry, others through a class-level `label` such as `BaseStations`' `cell_kpi`. Either way
> nothing falls back, so the core KPIs stay in the NPZ only. Add one Saveable variable that
> resolves to `generic` and the file appears — carrying the whole core-KPI schema alongside
> that variable.

Both directions are pinned in
[`tests/test_saveable_pipeline.py`](../../tests/test_saveable_pipeline.py) (`GenericLabelExportTests`
and `SingleRunExportTests.test_no_generic_dataset_when_every_variable_is_explicitly_routed`);
see [Testing the Saveable pipeline](saveable-pipeline-tests.md).

> **Two known bugs live on this path**, both pre-dating the per-context registry and both
> pinned as `expectedFailure` tests so that fixing either is reported instead of going
> unnoticed:
>
> 1. A **2-D variable in `generic` exports as all-null columns.** The schema allocates one
>    column per element of the *second* axis and registers the indexed names
>    (`<Class>.<var>.0`, `.1`, …), but the per-step buffer is keyed by the *base* name, so the
>    writer looks up names that are not there and writes `None`.
> 2. **The `generic` record and the `generic` schema disagree.** `PRB_usage_per_cell` and
>    `power_consumption_per_radio_unit_kW` are collected every step and written into the
>    `generic` record, but neither has a `pa.field(...)` entry in the hardcoded `generic`
>    schema, and `pa.Table.from_pylist(records, schema=…)` silently keeps only what the schema
>    names — so the wide `generic` table is missing two columns it builds.
>    **No data is lost**, because both quantities reach Parquet through the Saveable path
>    instead: `Power_Consumption.power_consumption_per_radio_unit_kW` in `ru_kpi`, and
>    `BaseStations.prb_usage_per_cell` in `cell_kpi`.

---

## 3. How a variable is routed: `label` and `expose_variable_labels`

Two methods on `Saveable` decide the destination file.

### `label` — the per-class default

A property returning `'generic'` (default) or an exploded label (`'ue_kpi'`, `'cell_kpi'`,
`'ru_kpi'`, …). It applies to **every** variable of the class. Defined once in the base
class ([`saveable.py`](../../giulia/outputs/saveable.py)), so a class that does nothing is
`generic` **for free** — you never write `return 'generic'` yourself; you only declare
*exceptions*.

```python
class Network(Saveable):
    @property
    def label(self) -> str:
        return 'cell_kpi'          # all of Network's variables go to cell_kpi
```

### `expose_variable_labels` — the per-variable override

Returns a dict `{variable_name: label}`. Use it when a **single class holds variables that
belong to different files** — the case `label` alone cannot express, because it is
per-class.

```python
class Rate_based_on_ins_CSI_RS_SINR(Rate):
    @property
    def label(self) -> str:
        return 'ue_kpi'                       # default bucket for this class

    def expose_variables(self) -> List[str]:
        return ["ue_throughput_Mbps", "cell_throughput_Mbps"]

    def expose_variable_labels(self) -> Dict[str, str]:
        return {"cell_throughput_Mbps": "cell_kpi"}   # only the exception
```

Here `ue_throughput_Mbps` is **not** in the dict, so it falls back to `label` (`ue_kpi`);
`cell_throughput_Mbps` is redirected to `cell_kpi`. One class, two files.

### The resolution order (and the fallback)

For each variable, the export computes its label like this:

```
1. expose_variable_labels()[var]   →  if the variable has an explicit entry, use it
2. else  saveable.label     →  the class default
3. else  'generic'          →  inherited from the base class
```

The default `expose_variable_labels()` is itself built from `label`
(`{var: self.label for var in expose_variables()}`), so everything stays consistent: a class
that overrides only `label` still gets all its variables mapped correctly.

> **Warning on missing entries.** If a variable is in `expose_variables()` but **absent** from
> an overridden `expose_variable_labels()`, it silently falls back to the class label. To avoid the
> easy mistake of a KPI quietly landing in `generic`, the export logs a warning naming the
> variable and the file it fell back to. Either add the variable to `expose_variable_labels()`, or
> rely on a sensible `label` default.

---

## 4. Constraints for exploded files (`ue_kpi` / `cell_kpi` / `ru_kpi`)

Because exploded files turn an array into one row per entity, the data must fit that shape:

1. **1-D or 2-D, exploded along the first (entity) axis.**
   - A **1-D** array becomes one **scalar column** (`<Class>.<var>`), one value per entity.
   - A **2-D** array (e.g. a UE×cell matrix) is **split along its second axis** into scalar
     columns `<Class>.<var>.0`, `<Class>.<var>.1`, … — so the row stays one-per-entity (e.g. one
     row per UE) and each sub-column holds that entity's value for one element of the second axis
     (e.g. one cell). Arrays with 3+ dimensions are still rejected at schema time. The numeric
     suffix can be given a label via
     [`expose_variable_column_prefixes()`](../../giulia/outputs/saveable.py) — e.g. returning
     `{"rsrp_ue_to_cell_dBm": "cell"}` produces `….rsrp_ue_to_cell_dBm.cell0`, `.cell1`, ….
2. **One row per entity (first axis).** The first-axis length must equal the number of entities —
   UEs for `ue_kpi`, cells for `cell_kpi`, radio units for `ru_kpi`.
3. **Consistent first-axis length across columns.** Every column in the same file must share that
   entity count.

> A UE×cell matrix routed to `ue_kpi` therefore produces one row per UE with one column per cell
> (`…dBm.cell0`, `…dBm.cell1`, … when `expose_variable_column_prefixes()` returns `"cell"`, else `…dBm.0`,
> `…dBm.1`). The number of sub-columns is fixed from the matrix width at schema time, so it must be
> stable across steps.

### How length mismatches are handled (the majority filter)

A file may accidentally collect columns whose first (entity) axis has different lengths (e.g. a
per-cell array mistakenly routed to `ue_kpi`). To keep this from corrupting the file, the writer:

- normalizes every field to a per-entity array along its first axis (1-D or 2-D) for the step,
- takes the **first-axis length the majority of fields agree on** as the true entity count,
- keeps the fields of that length and **drops the others**, logging a one-time warning per
  dropped field (`… has first-axis length X but '…_ue_kpi' is indexed per ue_id and expects Y …`).

This is a *safety net*, not a substitute for correct labels: a dropped field still does not
appear in that file. If the **mis-labeled** fields were ever the majority, they would win
instead — so fix labels rather than relying on this.

`generic` has none of these constraints: it accepts scalars, 1-D and 2-D arrays. A 2-D array
in `generic` is split into indexed columns `Class.var.0`, `Class.var.1`, … one per sub-array.

> Rule of thumb: **one value per UE → `ue_kpi`; per cell → `cell_kpi`; per radio unit →
> `ru_kpi`** (a 2-D array whose *first* axis is the entity also fits, splitting into `.0/.1/…`).
> Anything not aligned to one entity per row (per-PRB, per-beam, free-form config) → `generic`.

Types are inferred automatically; override
[`variable_types()`](../../giulia/outputs/saveable.py) only when inference fails.

---

## 5. Adding a new per-entity parquet

Exploded files are **fully dynamic** — there is no registry to edit. Just route a class (or
variable) to a **new label** via `label` / `expose_variable_labels`, and a matching file appears:

```python
class Bbu_Power(Saveable):
    @property
    def label(self) -> str:
        return 'bbu_kpi'                  # any new label works

    def expose_variables(self) -> List[str]:
        return ["power_per_bbu_kW"]       # 1-D, one value per BBU
```

On the next run this produces `..._bbu_kpi.parquet` with a `bbu_id` column and the
`Bbu_Power.power_per_bbu_kW` column — automatically.

How it works:

- **Any label except `generic` is exploded** — no allow-list.
- The **entity-id column is derived from the label name**: `_entity_id_column()` in
  [`save_episode_performance.py`](../../giulia/outputs/save_episode_performance.py) strips a
  trailing `_kpi` / `_kpis` and appends `_id` (`ue_kpi → ue_id`, `bbu_kpi → bbu_id`).
- The schema builder, exploded writer, majority filter and empty-file skip are all entity-
  agnostic; the per-step collection in
  [`save_performance.py`](../../giulia/outputs/save_performance.py) needs no changes at all.

**Real example — `Power_Consumption` (one class → two new files).** Power consumption is
computed both per radio unit and per baseband unit — two different entity counts, neither equal
to UEs or cells. The class maps each variable explicitly with `expose_variable_labels()`:

```python
class Power_Consumption(Saveable):
    def expose_variables(self) -> List[str]:
        return ["power_consumption_per_radio_unit_kW", "P_base_band_W"]

    def expose_variable_labels(self) -> Dict[str, str]:
        return {
            "power_consumption_per_radio_unit_kW": "ru_kpi",   # 1-D, one value per radio unit
            "P_base_band_W": "bbu_kpi",                        # 1-D, one value per BBU
        }
```

This produces **two** files automatically: `..._ru_kpi.parquet` (column
`Power_Consumption.power_consumption_per_radio_unit_kW`, indexed by `ru_id`) and
`..._bbu_kpi.parquet` (column `Power_Consumption.P_base_band_W`, indexed by `bbu_id`) — neither
the export code nor any registry was touched. Note there is **no** `label` here: every variable
is mapped explicitly, so nothing relies on the fallback.

---

## 6. Selecting which columns to export (the `fields` filter)

By default **every** column is exported. To export only a subset, list the columns under
`output.fields` in [`config.yml`](../../config.yml). It is read in
[`input_config.py`](../../giulia/inputs/input_config.py) and applied in
[`save_episode_performance.py`](../../giulia/outputs/save_episode_performance.py).

Key facts (and common surprises):

- **Empty or absent → export everything.** An empty list is treated as "no filter".
- **It is a single, global filter** applied to *all* files (`generic`, `ue_kpi`, …). There is
  no per-file list. Routing to a file is decided by `label` (code); `fields` only trims the
  columns afterwards.
- **Index columns are always kept** — `episode_idx`, `step_idx` and every entity-id column
  (`ue_id`, `cell_id`, `ru_id`, …).
- **Names are matched as column names**, i.e. `<Class>.<variable>` (or a hardcoded `generic`
  core name). For an expanded 2-D variable (suffixed `….0`, `….1`, …) listing the **base name**
  keeps *all* of its expanded columns; you may also list a single `….0` to keep just that one.
- A file left with only index columns after filtering is **not created** (see
  [§2](#empty-files-are-not-created)).

> The hardcoded core KPIs in `generic` (e.g. `cell_throughput_Mbps`,
> `best_serving_CRS_rsrp_per_ue_dBm`) are **not** in the Saveable catalog. If you turn on
> `fields` with only catalog entries, those core columns are dropped from `generic`.

---

## 7. Worked examples

### a) A per-UE KPI

`SINR_CRS` exposes three values per UE (1-D). Route them to `ue_kpi`:

```python
class SINR_CRS(SINR):
    def expose_variables(self) -> List[str]:
        return ["sinr_ue_to_cell_dB", "usefulPower_ue_dBm", "interfPower_ue_dBm"]

    def expose_variable_labels(self) -> Dict[str, str]:
        return {
            "sinr_ue_to_cell_dB": "ue_kpi",
            "usefulPower_ue_dBm": "ue_kpi",
            "interfPower_ue_dBm": "ue_kpi",
        }
```

Result in `..._ue_kpi`: columns `episode_idx, step_idx, ue_id, SINR_CRS.sinr_ue_to_cell_dB,
SINR_CRS.usefulPower_ue_dBm, SINR_CRS.interfPower_ue_dBm`, one row per UE per step.

(Since all three go to the same file, `@property label = 'ue_kpi'` would be equivalent and
shorter — use `expose_variable_labels` when you need to *split*.)

### b) A per-cell KPI

```python
class Network(Saveable):
    @property
    def label(self) -> str:
        return 'cell_kpi'

    def expose_variables(self) -> List[str]:
        return ["_position_x_m", "_position_y_m", "_position_z_m"]
```

Result in `..._cell_kpi`: one row per cell per step, columns `Network._position_x_m`, etc.

### c) A class with both per-UE and per-cell data (the split)

See the `Rate` example in [§3](#expose_variable_labels--the-per-variable-override):
`ue_throughput_Mbps` → `ue_kpi`, `cell_throughput_Mbps` → `cell_kpi`, from one object.

### d) A generic variable (default)

No override needed. The variable lands in `..._generic` as a list column, one row per step:

```python
class Distance_Angles_ue_to_cell(Saveable):
    def expose_variables(self) -> List[str]:
        return ["distances_b_to_a_3d_m"]   # 2-D (UE×cell) → generic, split into .0 .. .N
```

---

## 8. Common pitfalls

| Pitfall | What happens | Fix |
|---------|--------------|-----|
| Variable in `expose_variables()` but not in `expose_variable_labels()` | Falls back to class `label` (often `generic`); a **warning** is logged. | Add it to `expose_variable_labels()`, or set a meaningful `label`. |
| 2-D array routed to an exploded label | Split along the second axis into `.0/.1/…` columns, one row per entity. (3+ dimensions are still rejected.) | Make sure the **first** axis is the entity, or keep it in `generic`. |
| A non-per-entity array routed to an exploded file (wrong length) | Dropped by the **majority filter** with a one-time warning; the correct columns are unaffected. | Re-label it to the right file (or `generic`). |
| Exploded file looks "empty" (only index columns, or not created) | All its data columns were filtered out by `fields`, or none were routed. | Add the column to `fields`, or empty `fields`; check the `label`. |
| Same class instantiated twice | Only **one** instance is registered per class name — the **most recently created** one replaces the previous (so re-creating a Saveable each episode tracks the current episode, not episode 0). | For two instances **at the same time**, define thin subclasses, one per instance (see [Saveable Objects](../source/saveable-objects.rst)). |
| Sending some variables to one file and others to another via `label` | Impossible — `label` is per-class. | Use `expose_variable_labels()`. |

---

## 9. Quick reference

```python
from typing import Dict, List
from giulia.outputs.saveable import Saveable

class MyKpi(Saveable):
    # 1. Register
    def __init__(self):
        super().__init__()
        self.my_per_ue_value = ...     # 1-D, length = number of UEs
        self.my_per_cell_value = ...   # 1-D, length = number of cells

    # 2. Declare what to export
    def expose_variables(self) -> List[str]:
        return ["my_per_ue_value", "my_per_cell_value"]

    # 3. (default bucket) optional
    @property
    def label(self) -> str:
        return 'ue_kpi'

    # 4. (per-variable routing) only for exceptions / splits
    def expose_variable_labels(self) -> Dict[str, str]:
        return {"my_per_cell_value": "cell_kpi"}

    # 5. (optional) label the expanded sub-columns of a 2-D variable in an exploded file
    def expose_variable_column_prefixes(self) -> Dict[str, str]:
        return {"my_per_ue_matrix": "cell"}   # -> '<Class>.my_per_ue_matrix.cell0', '.cell1', ...

    # 6. (optional) document each variable — shown in the generated Saveable catalog
    def expose_variable_descriptions(self) -> Dict[str, str]:
        return {"my_per_ue_value": "signal quality per UE (dB)"}
```

- Read a value live: `Saveable.get('MyKpi.my_per_ue_value')`.
- Discover columns of a produced file: open its `<dataset>.schema.md` sidecar.
- Export only some columns: set `output.fields` in [`config.yml`](../../config.yml) ([§6](#6-selecting-which-columns-to-export-the-fields-filter)).
- List every available key for your config (with any `expose_variable_descriptions()` text
  shown beneath each key): see [Saveable Catalog](saveable-catalog.md).

---

## 10. Integrating a new Saveable into the run

§1–§9 cover how a Saveable *declares and routes* its variables. But two things still have to be
true for those variables to actually appear in the output, and **both live outside the Saveable
class** — this is the most common reason a freshly-written KPI produces nothing.

**1. The object must be constructed during the run.** A Saveable only registers when it is
**instantiated**, and the simulation re-creates its objects **every episode** inside the preset's
`configure()` method. A brand-new KPI must be constructed there, alongside the existing ones — for
example `SINR_CRS` is built in
[`g_multi_frequency_layer_preset.py`](../../giulia/presets/g_multi_frequency_layer_preset.py)
inside `configure()`. If nothing constructs your class, it never registers and silently produces
**no column** (you get no error — see [§2 "empty files are not created"](#empty-files-are-not-created)).

**2. Its exposed attributes must hold a fresh, valid value at every collection point.** Collection
runs **every step** and simply *reads* the current attribute value
([`save_performance.py`](../../giulia/outputs/save_performance.py) → `saveable.get_output(var)`); it
does **no computation of its own**. So something must update those attributes each step *before*
collection. In practice the KPI object is scheduled on the event loop so its per-step method
(`process()`) runs each step and refreshes the attributes:

```python
# inside the preset's configure()
self.my_kpi_obj = MyKpi(...)          # construct -> registers as 'MyKpi'
self.eventDriven_obj.add_event(0, self.my_kpi_obj, 'process', scheduling_info["my_kpi_obj"])
```

If an exposed attribute is **missing** at collection time, `get_output()` returns `None` for that
step (it does not raise); if it is **stale**, the file silently stores the stale value. So every
name in `expose_variables()` must be an attribute that exists *and* is refreshed on the schedule
above.

> **Checklist for a new exposed KPI.** (a) Subclass `Saveable`, call `super().__init__()`, list the
> attributes in `expose_variables()`, and set a `label` ([§3](#3-how-a-variable-is-routed-label-and-expose_variable_labels)).
> (b) Construct it in the preset's `configure()`. (c) Make sure it is updated every step (schedule
> its `process()` / compute). Then confirm it appeared by opening the run's `<dataset>.schema.md`
> sidecar ([§2](#2-the-output-files)).

---

## 11. What happens automatically

The whole point of the system is that, once a class declares `expose_variables()` (and optionally
`label` / `expose_variable_labels()`), everything else is handled for you. You never edit the export
code to add a KPI or a new per-entity file. In particular:

- **Registration** — a Saveable registers itself on construction (`super().__init__()`); there
  is no list to maintain. The registry keeps one entry per class name, the **most recently created**
  instance (so re-creating a Saveable each episode tracks the current episode, not episode 0). It is
  scoped to the simulation's own execution context, so parallel runs export their own data
  ([Saveable Concurrency](saveable-concurrency.md)).
- **Collection** — every variable in `expose_variables()` is read once per step, with no per-class
  wiring in the exporter.
- **Routing & file creation** — `label` / `expose_variable_labels()` decide the destination file. Any
  label other than `generic` **auto-creates its own per-entity parquet**, with an `*_id` index
  column derived from the label name (`bbu_kpi → bbu_id`). New entity types need **zero** changes
  to the export code.
- **Type inference** — column types are inferred from the data (override only with
  `variable_types()` when inference fails).
- **Robustness** — exploded files drop length-mismatched columns (the majority entity count wins)
  with a one-time warning, and a label that ends up with no data columns is skipped entirely (no
  empty file or folder).
- **Schema sidecar** — every written file gets a `<dataset>.schema.md` listing its columns.
- **Catalog** — `scripts/generate_saveable_catalog.py` runs a minimal simulation per preset and
  regenerates the variable list in [`saveable-catalog.md`](saveable-catalog.md) — the one piece
  that is produced by a tool rather than at run time.

So adding a brand-new per-entity output is, end to end, just: write the 1-D per-entity array,
list it in `expose_variables()`, and point it at a label. The file appears on the next run.
