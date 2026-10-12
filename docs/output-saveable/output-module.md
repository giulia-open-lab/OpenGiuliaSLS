# The Output Module (`giulia/outputs/`)

A map of the package that turns a running simulation into files on disk, and the file
formats it produces. Where the other pages in this folder each cover one concern in depth,
this one answers "**which file does what, and what actually runs during a normal run?**"

Companion pages:

- [Saveable Objects](../source/saveable-objects.rst) — how a class exposes its variables.
- [Saveable Output & Export](saveable-output.md) — the routing rules that decide a variable's column and file.
- [Saveable Concurrency](saveable-concurrency.md) — which registry the exporters read.
- [Reading Parquet Output](reading-parquet.md) — reading the results back.
- [Saveable Monitor](SaveAble_monitor.md) — reading values live instead of from a file.

---

## 1. What is in the package

| File | Role | In the live run? |
|---|---|---|
| [`saveable.py`](../../giulia/outputs/saveable.py) | The `Saveable` base class and the per-context registry. | **Yes** |
| [`save_performance.py`](../../giulia/outputs/save_performance.py) | `Performance` — collects values **every step** into per-episode buffers. | **Yes** |
| [`save_episode_performance.py`](../../giulia/outputs/save_episode_performance.py) | `Episode_Performance` — accumulates episodes and **writes** the NPZ / Parquet output. | **Yes** |
| [`processing.py`](../../giulia/outputs/processing.py) | `.gof` archive pack / unpack. | **Yes** (packing only, conditionally — §3) |
| [`monitor.py`](../../giulia/outputs/monitor.py) | `Monitor` — background polling of KPIs during a run. | Opt-in, from [`main.py`](../../main.py) |
| [`read_outputs.py`](../../giulia/outputs/read_outputs.py) | `OutputReader` — the read-back API. | No — post-processing |
| [`utils.py`](../../giulia/outputs/utils.py) | `load_gof` / `load_dataset` helpers, plus a reshaping utility used by `save_performance`. | Partly |

`giulia/outputs/__init__.py` re-exports the read-side helpers, so analysis scripts can do:

```python
from giulia.outputs import OutputReader, load_gof, load_dataset
```

> **Note.** Older revisions also carried `output_module.py` — an offline `OutputModule` that
> converted already-written NPZ results to Parquet. It has been removed: the simulator writes
> Parquet directly (§2), including the Saveable columns that the NPZ path cannot carry (§3).

---

## 2. The pipeline that actually runs

```
every step        Performance.process()
                    reads each Saveable's expose_variables()  ──► r_e_* buffers
                                                                   (save_performance.py)
end of episode    Episode_Performance.save_episode_performance(episode_index, g)
                    copies the episode's buffers into r_* arrays
end of run        Episode_Performance.save_episodes_performance_in_file(project_name)
                    ├─ NPZ enabled?     ──► _save_episodes_performance_npz()
                    └─ Parquet enabled? ──► _save_episodes_performance_parquet()
                                              ├─ build one schema per label
                                              ├─ apply the `fields` filter
                                              ├─ write .parquet (file or partitioned folder)
                                              ├─ write the <dataset>.schema.md sidecar
                                              └─ pack_output() -> .gof   (conditional, §3)
```

The two collection stages are described in [Saveable Output & Export §1](saveable-output.md#1-the-big-picture);
the routing and shape rules for the export stage are that page's §2–§5.

---

## 3. The output formats

Both formats are configured under `output.formats` in [`config.yml`](../../config.yml) and are
independent — you can enable either, both, or neither.

### NPZ

```yaml
output:
  formats:
    npz:
      enable: true
```

Written by `_save_episodes_performance_npz`. It is the **legacy** format: one flat
`np.savez` archive whose keys are the hardcoded core KPI names (`CRS_sinr_ue_to_cell_dB`,
`cell_throughput_Mbps`, …), each stacked along the episode axis. Saveable variables are
**not** in it — NPZ carries only the core KPIs.

> **It ignores `output.directory`.** The NPZ always lands in
> `outputs/results/<project_name>/results-raw.npz` (via `results_file()` in
> [`giulia/fs/path.py`](../../giulia/fs/path.py)), *not* in the directory the Parquet output
> uses. This surprises people who set `output.directory` and then cannot find the `.npz`.

### Parquet

```yaml
output:
  formats:
    parquet:
      enable: true
      compression: zip          # none | zip  (zip -> .gof)
      rm_uncompressed: false
      use_partition_cols: false
```

Written by `_save_episodes_performance_parquet` into `output.directory`. This is the
format the rest of the documentation is about: one dataset per label
(`<project>_ue_kpi`, `<project>_cell_kpi`, …), each with a `<dataset>.schema.md` sidecar
listing its columns and types. A `<project>_generic` dataset is written only when a Saveable is
routed to the `generic` label, which is not the case in a standard run.

`use_partition_cols` changes the on-disk layout:

| `use_partition_cols` | What is written | `compression: zip` effect |
|---|---|---|
| `false` (set in the repository's config files) | a **single file** `<project>_<label>.parquet` | **none** — no `.gof` is produced |
| `true` (default when the key, or the whole `formats` block, is omitted) | a **folder** `<project>_<label>/episode_idx=…/step_idx=…/` | the folder is packed into `<project>_<label>.gof` |

> **Gotcha.** `compression: zip` only does something when `use_partition_cols: true`. The
> code deliberately skips packing an unpartitioned dataset, because a single `.parquet`
> file is already compressed (snappy) and needs no archive — so with
> `use_partition_cols: false`, as in the repository's config files, you will never see a `.gof`,
> whatever `compression` says.
> `rm_uncompressed` likewise only applies on that path: it deletes the partition folder
> once packed.

Both layouts read back identically — `pd.read_parquet` and
[`OutputReader`](reading-parquet.md) accept the file and the folder alike.

---

## 4. `.gof` files

A `.gof` is **a ZIP archive with a different extension** — nothing more. It exists so a
partitioned Parquet dataset (potentially hundreds of small files in nested folders) can be
moved around as one file.

[`processing.py`](../../giulia/outputs/processing.py):

| Function | What it does |
|---|---|
| `pack_output(output_dir, delete_uncompressed=True)` | Zips `output_dir` and renames it to `<output_dir>.gof`. Called at the end of the Parquet export. |
| `use_output(gof_file, delete_gof=True)` | Unpacks one `.gof` back to a folder and returns its path. |
| `use_outputs(output_dir, delete_gof=True)` | Unpacks every `.gof` in a directory. |

Note the defaults: `use_output` / `use_outputs` **delete the archive** after extracting
unless you pass `delete_gof=False`.

To read one without unpacking it by hand, use `load_gof` from
[`utils.py`](../../giulia/outputs/utils.py), which extracts (keeping the archive) and
returns a PyArrow dataset:

```python
from giulia.outputs import load_gof

dataset = load_gof("outputs/gof/3GPPTR38_901_5G_uniform_ue_kpi.gof")
df = dataset.to_table().to_pandas()
```

`load_gof` also accepts the case where the archive is already unpacked: if the `.gof` is
gone but the folder next to it exists, it loads that instead. `load_dataset(path)` skips
the archive handling entirely and opens a Parquet dataset directory.

---

## 5. Where the files end up

| Artefact | Path | Set by |
|---|---|---|
| Parquet datasets + `.schema.md` sidecars | `<output.directory>/<project_name>_<label>[.parquet]` | `output.directory` in `config.yml` |
| `.gof` archives | next to the dataset they pack | as above (partitioned runs only, §3) |
| NPZ raw results | `outputs/results/<project_name>/results-raw.npz` | fixed — `results_file()` |
| Default output directory | `outputs/gof` | `config.example.yml` |

Paths are resolved by [`giulia/fs/path.py`](../../giulia/fs/path.py); `output.directory` may
be absolute or relative to the folder from which Giulia is started.

---

## 6. Choosing a format

- **Analysing results, now or later → Parquet.** It is the only format carrying Saveable
  variables, it keeps types, and it supports column/row pruning
  ([Reading Parquet Output §5](reading-parquet.md#5-the-features-parquet-offers)).
- **Feeding legacy scripts that expect `results-raw.npz` → NPZ.** Core KPIs only.
- **Sharing a partitioned run → Parquet + `compression: zip`** for a single `.gof`.
- **Watching a KPI during the run → neither**; use the [Monitor](SaveAble_monitor.md).
