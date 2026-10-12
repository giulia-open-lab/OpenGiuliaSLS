"""
Read Giulia's Parquet output.

After a run, Giulia writes its results as Parquet **datasets** in the output directory
(``output.directory`` in ``config.yml``, e.g. ``outputs/gof``). Each Saveable *label*
becomes one dataset named ``<project>_<label>`` — stored either as a single
``<project>_<label>.parquet`` file or, when partitioning is on, as a
``<project>_<label>/episode_idx=…/step_idx=…/`` folder. pandas reads both the same way.

This module is a tiny, readable API over those datasets. It answers four questions:

    1. How many datasets are there?           -> count() / datasets()
    2. What columns and types are inside?      -> schema() / kpis()
    3. Give me a whole KPI column.             -> column(dataset, kpi)
    4. Give me a KPI for one episode/step.     -> values(dataset, kpi, episode, step)

Example::

    from giulia.outputs.read_outputs import OutputReader

    reader = OutputReader("outputs/gof")
    reader.count()                                  # -> e.g. 6
    reader.datasets()                               # -> ['..._cell_kpi', '..._ue_kpi', ...]
    reader.schema("ue_kpi")                         # -> {'episode_idx': 'uint64', ...}
    reader.column("ue_kpi", "SINR_CRS.sinr_ue_to_cell_dB")            # whole column
    reader.values("ue_kpi", "SINR_CRS.sinr_ue_to_cell_dB", episode=0, step=0)

A dataset is addressed by its full name (``3GPPTR38_901_5G_uniform_ue_kpi``) or by a
unique suffix (``ue_kpi``). See ``docs/output-saveable/reading-parquet.md`` for the full
guide, including the generic-vs-exploded distinction.

This module depends only on ``pandas`` and ``pyarrow`` (both project dependencies); it
imports nothing from the rest of Giulia, so it is safe to use from a lightweight analysis
script.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

PARQUET_SUFFIX = ".parquet"


def is_index_column(name: str) -> bool:
    """
    True for columns that *locate* a row rather than hold a KPI value:
    ``episode_idx``, ``step_idx`` and the per-entity id columns (``ue_id``, ``cell_id``,
    ``ru_id``, ``bbu_id``, ``ssbBeam_id``, …).

    Entity-id columns are dynamic (their name is derived from the label), so they are
    matched by *shape* — a bare ``*_id`` name — instead of a fixed list. KPI columns are
    always ``<Class>.<variable>`` (they contain a dot) or a hardcoded generic core name,
    so a name that ends in ``_id`` and has no dot is an index column.
    """
    if name in ("episode_idx", "step_idx"):
        return True
    return name.endswith("_id") and "." not in name


class OutputReader:
    """
    A small reader over one Giulia output directory.

    Point it at the folder that holds the ``<project>_<label>`` datasets
    (e.g. ``outputs/gof``). It discovers every dataset once on construction, then answers
    questions about them. Each dataset may be a single ``.parquet`` file or a partitioned
    folder; both are handled transparently.
    """

    def __init__(self, output_dir: str):
        if not os.path.isdir(output_dir):
            raise FileNotFoundError(f"Output directory not found: {output_dir!r}")
        self.output_dir = output_dir
        # dataset name -> a path pandas/pyarrow can read (a .parquet file, or a
        # partitioned dataset folder).
        self._datasets: Dict[str, str] = self._discover()

    # -- discovery -------------------------------------------------------- #

    def _discover(self) -> Dict[str, str]:
        """Find every dataset in ``output_dir`` (single file or partitioned folder)."""
        datasets: Dict[str, str] = {}
        for entry in sorted(os.listdir(self.output_dir)):
            path = os.path.join(self.output_dir, entry)
            if os.path.isfile(path) and entry.endswith(PARQUET_SUFFIX):
                datasets[entry[: -len(PARQUET_SUFFIX)]] = path
            elif os.path.isdir(path) and glob.glob(os.path.join(path, "episode_idx=*")):
                datasets[entry] = path
        return datasets

    def _path(self, dataset: str) -> str:
        """
        Resolve a dataset name to its path. Accepts the full name
        (``3GPPTR38_901_5G_uniform_ue_kpi``) or a unique suffix (``ue_kpi``).
        """
        if dataset in self._datasets:
            return self._datasets[dataset]
        matches = [name for name in self._datasets if name.endswith(dataset)]
        if len(matches) == 1:
            return self._datasets[matches[0]]
        if not matches:
            raise KeyError(
                f"No dataset matching {dataset!r} in {self.output_dir!r}. "
                f"Available: {self.datasets()}"
            )
        raise KeyError(
            f"Ambiguous dataset {dataset!r}: matches {matches}. Use the full name."
        )

    # -- 1. how many datasets / which ones -------------------------------- #

    def datasets(self) -> List[str]:
        """The names of every dataset found, e.g. ``['..._cell_kpi', '..._ue_kpi']``."""
        return list(self._datasets)

    def count(self) -> int:
        """How many Parquet datasets are in the output directory."""
        return len(self._datasets)

    def __len__(self) -> int:
        return self.count()

    # -- 2. schema -------------------------------------------------------- #

    def schema(self, dataset: str) -> Dict[str, str]:
        """
        Map every column to its type string, read from the file metadata only — no row
        data is loaded. Includes the index columns (``episode_idx``, ``step_idx``, ``*_id``).
        """
        path = self._path(dataset)
        arrow_schema = (
            ds.dataset(path, partitioning="hive").schema
            if os.path.isdir(path)
            else pq.read_schema(path)
        )
        return {field.name: str(field.type) for field in arrow_schema}

    def kpis(self, dataset: str) -> List[str]:
        """The KPI columns of a dataset: every column that is not an index column."""
        return [name for name in self.schema(dataset) if not is_index_column(name)]

    # -- 3. a whole KPI column -------------------------------------------- #

    def column(self, dataset: str, kpi: str) -> pd.Series:
        """
        Return one KPI's entire column as a pandas Series (across every episode, step and
        entity). Thanks to Parquet's columnar layout, only that column is read from disk.
        """
        self._require_kpi(dataset, kpi)
        return pd.read_parquet(self._path(dataset), columns=[kpi])[kpi]

    # -- 4. a KPI for one episode (and optionally one step) --------------- #

    def values(self, dataset: str, kpi: str, episode: int,
               step: Optional[int] = None) -> pd.Series:
        """
        Return a KPI's values for one ``episode`` (and one ``step`` if given). The episode
        and step are applied as Parquet row filters, so non-matching rows are never read.

        The shape of the result depends on the file (see ``reading-parquet.md``):

        - **exploded** file (``ue_kpi`` / ``cell_kpi`` / …): one value per entity — a
          Series with one row per ``*_id``.
        - **generic** file: one row per ``(episode, step)`` whose cell holds the whole
          per-entity vector (a list). For a single step, ``.iloc[0]`` is that list.
        """
        self._require_kpi(dataset, kpi)
        filters = [("episode_idx", "==", episode)]
        if step is not None:
            filters.append(("step_idx", "==", step))
        frame = pd.read_parquet(self._path(dataset), columns=[kpi], filters=filters)
        return frame[kpi]

    # -- convenience ------------------------------------------------------ #

    def read(self, dataset: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Read a whole dataset (or just ``columns``) into a pandas DataFrame."""
        return pd.read_parquet(self._path(dataset), columns=columns)

    def _require_kpi(self, dataset: str, kpi: str) -> None:
        """Raise a helpful ``KeyError`` if ``kpi`` is not a column of ``dataset``."""
        columns = self.schema(dataset)
        if kpi not in columns:
            available = self.kpis(dataset)
            preview = available[:10] + (["…"] if len(available) > 10 else [])
            raise KeyError(
                f"KPI {kpi!r} is not in dataset {dataset!r}. "
                f"It has {len(available)} KPIs; e.g. {preview}"
            )


def _default_output_dir() -> Optional[str]:
    """Pick a directory to demo with: the folder holding the first output found under ``outputs/``."""
    files = sorted(glob.glob("outputs/**/*.parquet", recursive=True))
    if files:
        return os.path.dirname(files[0])
    partitions = sorted(glob.glob("outputs/**/episode_idx=*", recursive=True))
    if partitions:
        # the dataset folder is the parent of the 'episode_idx=…' partition folder
        return os.path.dirname(os.path.dirname(partitions[0]))
    return None


def _demo(output_dir: str) -> None:
    """Print a short tour of one output directory (used by ``__main__``)."""
    reader = OutputReader(output_dir)
    print(f"Output directory: {output_dir}")
    print(f"Datasets ({reader.count()}):")
    for name in reader.datasets():
        print(f"  - {name}")

    if reader.count() == 0:
        return

    first = reader.datasets()[0]
    schema = reader.schema(first)
    kpis = reader.kpis(first)
    print(f"\nDataset '{first}': {len(schema)} columns, {len(kpis)} of them KPIs. First few KPIs:")
    for name in kpis[:5]:
        print(f"  - {name}  ({schema[name]})")

    if kpis:
        example = kpis[0]
        values = reader.values(first, example, episode=0, step=0)
        print(f"\nExample - '{example}' at episode 0, step 0:")
        print(f"  {values.tolist()}")


if __name__ == "__main__":
    import sys

    chosen_dir = sys.argv[1] if len(sys.argv) > 1 else _default_output_dir()
    if not chosen_dir:
        print("No Parquet output found under outputs/. Run a simulation first, or pass a "
              "directory: python -m giulia.outputs.read_outputs <output_dir>")
        raise SystemExit(1)
    _demo(chosen_dir)
