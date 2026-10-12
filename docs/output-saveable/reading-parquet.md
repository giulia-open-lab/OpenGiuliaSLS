# Reading Parquet Output

This page explains **how to read the Parquet files that Giulia produces**, and gives
a plain-language tour of **what Parquet is and what it can do**.

It is the companion to [Saveable Output & Export](./saveable-output.md): that page
explains how the files are *written*; this page explains how to *read* them back.

A small, ready-to-use reader API lives at
[`giulia/outputs/read_outputs.py`](../../giulia/outputs/read_outputs.py) — see §3.

---

## 1. What is Parquet, in one minute

Parquet is a **file format for tables** — rows and columns, like a spreadsheet or a
CSV, but designed for data analysis. The differences that matter to you:

- **It is binary, not text.** You cannot open it in a text editor; you read it with a
  library (pandas, PyArrow, Polars, DuckDB, …). In return it is much smaller and much
  faster than CSV.
- **It is columnar.** The values of one column are stored together. So reading "just
  the SINR column" does not require reading the whole file — the reader can skip the
  columns you did not ask for. This is the single biggest reason Parquet is fast.
- **It keeps the data types.** A CSV forgets whether `3` was an integer, a float, or
  text; you have to re-guess on every read. Parquet stores the type (int, float,
  string, list, …) *inside* the file, so a number comes back as a number and a list
  comes back as a list — no parsing, no surprises.
- **It is compressed.** Columns of similar values compress very well, so Parquet files
  are typically a fraction of the size of the equivalent CSV.

> **Mental model:** think of a Parquet file as "a saved pandas DataFrame that remembers
> its column types and is cheap to read one column at a time."

---

## 2. The files Giulia writes

When you run `python main.py config.yml` with Parquet enabled, the results land in the
output directory set by `output.directory` in [`config.yml`](../../config.yml)
(default `outputs/gof`). You get **one file per label**, named `<project>_<label>.parquet`:

| File | One row per… | Each KPI cell holds… |
|------|--------------|----------------------|
| `<project>_generic.parquet` (only when a Saveable is routed to `generic`; not in a standard run) | `(episode, step)` | the **whole vector** for that step (a list) |
| `<project>_ue_kpi.parquet` | `(episode, step, ue_id)` | **one number for that UE** |
| `<project>_cell_kpi.parquet` | `(episode, step, cell_id)` | **one number for that cell** |
| `<project>_ru_kpi.parquet` / `<project>_bbu_kpi.parquet` / … | `(episode, step, <entity>_id)` | **one number for that entity** |

This split is the **one thing to understand before reading**, because it changes how a
value comes back:

- In the **generic** file a KPI cell is a **list** (one entry per UE or per cell). One
  row already contains the whole step.
- In an **exploded** file (`ue_kpi`, `cell_kpi`, …) a KPI cell is a **single number**,
  and the entities are spread across many rows, one per `*_id`.

Every file is written next to a `<file>.schema.md` sidecar that lists every column and
its type — the fastest way to see what is inside without opening Python. See
[Saveable Output & Export §2](./saveable-output.md#2-the-output-files) for the full rules.

> **Note on the `fields` filter.** If `output.fields` is set in `config.yml`, only the
> listed columns are exported (plus the index columns). So a file may contain fewer KPIs
> than the full catalog — always check what is actually present (see §4).

---

## 3. Reading a file

The simplest way is the **`OutputReader`** API that ships with Giulia
([`giulia/outputs/read_outputs.py`](../../giulia/outputs/read_outputs.py)). Point it at the
output directory and ask it questions. It depends only on pandas + PyArrow and imports
nothing else from Giulia, so it is safe to use from a small analysis script.

### The OutputReader API

```python
from giulia.outputs import OutputReader

reader = OutputReader("outputs/gof")   # the folder holding the <project>_<label> datasets
```

It answers exactly four questions.

**1. How many datasets are there, and which?**

```python
reader.count()        # e.g. 6
reader.datasets()     # ['3GPPTR38_901_5G_uniform_cell_kpi', '..._ue_kpi', ...]
```

**2. What is inside one — columns and their types?** Read straight from the file metadata,
without loading any rows:

```python
reader.schema("ue_kpi")   # {'episode_idx': 'uint64', ..., 'SINR_CRS.sinr_ue_to_cell_dB': 'double'}
reader.kpis("ue_kpi")     # just the KPI columns (the index columns are dropped)
```

A dataset is addressed by its **full name** or any **unique suffix**, so `"ue_kpi"` resolves
to `"3GPPTR38_901_5G_uniform_ue_kpi"`.

**3. Give me a whole KPI column** — every episode, step and entity — as a pandas Series:

```python
sinr = reader.column("ue_kpi", "SINR_CRS.sinr_ue_to_cell_dB")
```

**4. Give me a KPI for one episode** (and optionally one step). Episode/step are applied as
Parquet row filters, so non-matching rows are never read from disk:

```python
sinr_e0_s0 = reader.values("ue_kpi", "SINR_CRS.sinr_ue_to_cell_dB", episode=0, step=0)
print(sinr_e0_s0.tolist())   # one value per UE
```

> **The result shape follows the file (see §2).** In an **exploded** file (`ue_kpi`,
> `cell_kpi`, …) you get one value per entity — one row per `*_id`. In the **generic** file
> one row holds the whole per-entity vector, so `values(..., episode=0, step=0).iloc[0]` is
> that list.

For the whole table (to slice it yourself) use `reader.read("ue_kpi")` — optionally
`columns=[...]`. To print a quick tour of an output folder, run it as a script:

```
python -m giulia.outputs.read_outputs outputs/gof
```

### Or by hand with pandas

`OutputReader` is a thin wrapper over [pandas](https://pandas.pydata.org/); these are the
same steps if you would rather not import Giulia at all.

#### Step 1 — Open the file

```python
import pandas as pd

df = pd.read_parquet("outputs/gof/3GPPTR38_901_5G_uniform_ue_kpi.parquet")
```

`pd.read_parquet` reads both a **single `.parquet` file** and a **partitioned dataset
folder** (a directory of `episode_idx=…/step_idx=…/*.parquet` parts — produced when
`use_partition_cols: true`). You pass the file in the first case, the folder in the second;
the call is the same.

#### Step 2 — See which KPIs are inside

The KPIs are the columns. The only non-KPI columns are the index columns that locate a
row (`episode_idx`, `step_idx`, and the entity id like `ue_id`):

```python
index_columns = {"episode_idx", "step_idx", "ue_id", "cell_id", "ru_id", "bbu_id"}
kpis = [c for c in df.columns if c not in index_columns]
print(kpis)
```

#### Step 3 — Pull a KPI value back out

Reading one KPI means selecting its column, usually for a chosen episode and step.

**From a `ue_kpi` (exploded) file** — one number per UE, spread over rows:

```python
# Effective CSI-RS SINR for every UE in episode 0, step 0:
mask = (df["episode_idx"] == 0) & (df["step_idx"] == 0)
sinr_per_ue = df.loc[mask, "SINR_CSI_RS.effective_sinr_ue_to_cell_dB"]
print(sinr_per_ue.tolist())   # e.g. [12.3, 8.7, 21.0, ...] — one value per UE
```

**From the `generic` file**, when one is written (in a standard run it is not) — the same KPI, but one row already holds the whole vector:

```python
mask = (df["episode_idx"] == 0) & (df["step_idx"] == 0)
sinr_vector = df.loc[mask, "effective_CSI_RS_sinr_ue_to_cell_dB"].iloc[0]
print(list(sinr_vector))      # the per-UE list, straight from that one row
```

> **The same KPI has two names**, depending on the file. In the exploded file it keeps
> its Saveable name `SINR_CSI_RS.effective_sinr_ue_to_cell_dB`; in the generic file it
> uses the hardcoded core name `effective_CSI_RS_sinr_ue_to_cell_dB`. When in doubt, list
> the columns (Step 2) and read the name off the list.

That is the whole workflow — and exactly what the `OutputReader` API in §3 does for you,
including resolving a KPI's name and filtering by episode/step.

---

## 4. Quick inspection without writing a loop

A few one-liners that answer "what is in here?":

```python
df.shape                 # (number_of_rows, number_of_columns)
df.columns.tolist()      # every column name
df.dtypes                # the type of each column
df.head()                # the first few rows
df[["episode_idx", "step_idx"]].drop_duplicates()   # which episodes/steps exist
```

To read the columns **without loading any data**, use PyArrow's schema — it reads only
the file's metadata:

```python
import pyarrow.parquet as pq

schema = pq.read_schema("outputs/gof/..._ue_kpi.parquet")
print(schema)            # names and types of every column, instantly
```

Or just open the `<file>.schema.md` sidecar Giulia wrote next to the data.

---

## 5. The features Parquet offers

Beyond "open it and get a DataFrame", these are the capabilities worth knowing. You will
not need most of them for small outputs, but they are what makes Parquet worth using.

### Read only the columns you need (column pruning)

Because storage is columnar, you can ask for a subset of columns and the reader skips the
rest on disk:

```python
df = pd.read_parquet(path, columns=["episode_idx", "step_idx",
                                    "SINR_CSI_RS.effective_sinr_ue_to_cell_dB"])
```

This is faster and uses less memory than reading everything and then dropping columns.

### Read only the rows you need (predicate pushdown / filters)

You can filter rows *while reading*, so non-matching rows are never loaded:

```python
df = pd.read_parquet(path, filters=[("episode_idx", "==", 0),
                                    ("step_idx", "==", 0)])
```

For a **partitioned dataset** (`episode_idx=…/step_idx=…/` folders), filtering on the
partition columns is essentially free — the reader skips whole folders it does not need.

### Partitioning

Giulia can write the data split into folders by `episode_idx` and `step_idx` (set
`use_partition_cols: true` in `config.yml`). This makes "give me episode 3, step 2" read
only that one folder instead of scanning the whole table. The trade-off is many small
files instead of one — which is why `config.yml` can `zip` the folder into a single
`.gof` archive for sharing.

### Types, including nested lists

Parquet stores real types, so the generic file's list columns (`list<double>`) come back
as Python lists / NumPy arrays directly — no string parsing. Integers stay integers,
floats stay floats, strings stay strings.

### Compression

Parquet compresses each column (Giulia uses `snappy`, a fast codec). You get small files
for free; the reader decompresses transparently. Nothing to configure on the read side.

### Schema / metadata you can read without the data

As shown in §4, the column names and types live in the file's metadata and can be read on
their own. This is how the `.schema.md` sidecar and tools like `parquet-tools` can show a
file's structure instantly.

### It is a standard, portable format

The same file reads in pandas, PyArrow, Polars, DuckDB, Spark, R, etc. You are not locked
into Python. For example, DuckDB can run SQL straight over a Parquet file:

```sql
SELECT step_idx, "SINR_CSI_RS.effective_sinr_ue_to_cell_dB"
FROM 'outputs/gof/..._ue_kpi.parquet'
WHERE episode_idx = 0;
```

### What Parquet is *not* good at

- **Editing in place.** Parquet files are written once; to "change a value" you rewrite
  the file. It is a format for results, not a live database.
- **Reading by hand.** It is binary — always go through a library.
- **Tiny single values.** For one scalar a CSV or a print is simpler; Parquet pays off
  once you have tables.

---

## 6. Common gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError` on a column name | Looked for the exploded name in the generic file (or vice-versa). | List the columns first (§3 Step 2); the same KPI has two names across files. |
| A KPI is missing entirely | `output.fields` filtered it out, or its label routed it to a different file. | Check the `.schema.md` sidecar; widen `fields` or open the right `<project>_<label>` file. |
| A cell prints as a whole list | You read the **generic** file, where each row holds the full per-UE vector. | Use `.iloc[0]` to get that one row's list, or read the exploded `ue_kpi` file for one-number-per-row. |
| `pd.read_parquet` raises about a missing engine | PyArrow not installed in the active environment. | Install/activate the project environment (PyArrow is a project dependency). |
| Reading is slow / uses too much memory | Reading every column and row of a large file. | Use `columns=[...]` and `filters=[...]` to read only what you need (§5). |

---

## 7. See also

- [Saveable Output & Export](./saveable-output.md) — how these files are produced, the
  `label` / `expose_variable_labels` routing, and the `fields` filter.
- [Saveable Catalog](./saveable-catalog.md) — every KPI available per preset.
- [The Output Module](./output-module.md) — the formats these files come in (Parquet vs. NPZ),
  `.gof` archives, and where each artefact is written.
- [Saveable Monitor](./SaveAble_monitor.md) — reading a KPI **live** during a run instead
  of from a file.
- [`giulia/outputs/read_outputs.py`](../../giulia/outputs/read_outputs.py) — the
  `OutputReader` API that implements §3.
- pandas: [`read_parquet`](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html)
  · PyArrow: [Parquet docs](https://arrow.apache.org/docs/python/parquet.html)
