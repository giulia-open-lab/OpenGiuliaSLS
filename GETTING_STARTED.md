# Getting Started

## Shadowing maps

The shadowing maps (14 `*.mat` files, about 2.7 GB) are not stored in the repository, neither in Git nor in Git LFS.
Download them with `python scripts/download_shadowing_files.py` (from the UPV network or its VPN; otherwise ask a UPV
member to share them), then check that the `*.mat` files are inside
`shadowing/`, or set the `SHADOWING_FILES` environment variable to the folder that contains them. See the
[README](README.md#shadowing-maps-and-git-lfs) for details. Git LFS is needed only for the test references.

## Initial configuration

Once all the dependencies are installed, and the Conda environment is ready, you can start working with Giulia.

The experiment is selected in a YAML configuration file. The repository contains `config.yml`, which runs a heavy
4G/5G multilayer scenario, and the fully commented template `config.example.yml` (ITU-R M.2135 UMa). To start from
the template, either run it directly with `python main.py config.example.yml`, or copy it over `config.yml` and edit
it (this replaces the `config.yml` of the repository):

```shell
cp config.example.yml config.yml
```

Two points to know before editing it:

- The `args` entries are passed to the entry point as command-line text, so for the boolean entries (`enable_GPU`,
  `wraparound`, `regression`, `plots`) the value `false` is read as **true**. Leave an entry empty to use its default:
  `plots` and `regression` are off by default, while an empty `wraparound` follows the scenario (on for the hexagonal
  ITU and 3GPP layouts).
- Only `run.example: EV` can be run through `main.py` at the moment.

## Running Giulia

Once your configuration is ready, you have two options, either calling it directly from the cli:
```shell
python main.py
```

Or from a Python script. Note that `config.run()` only *prepares* the run: it resolves which entry
point to use and parses the arguments, then hands both back to you. You call the entry point
yourself:
```python
from giulia.inputs import InputConfig

if __name__ == '__main__':
    # Load the configuration file
    config = InputConfig.load('config.yml')
    # Resolve the entry point and parse the args
    giulia_main, args = config.run()
    # Run the pipeline
    giulia_main(args, config)
```

> [!IMPORTANT]
> Run this on the process's main thread. Giulia's C++ backends (PyTorch / Mitsuba) must be
> initialized and torn down there, or the interpreter can segfault during shutdown. See
> [`main.py`](main.py) for the reference wiring, which also shows how to attach a live KPI monitor.

> [!TIP]
> You may not have Giulia available in your Python environment.
> To do so, with the script in Giulia's root directory (adapt the code accordingly otherwise), add this code snippet 
> **to the top of your script**:
> ```python
> import os
> import sys
> 
> # Configure Giulia into path
> root_dir = os.path.dirname(__file__)
> sys.path.insert(1, root_dir)
> ```

## Loading the results

Once Giulia has finished the simulation, the Parquet results are inside the output directory (`output.directory` in
the config). There is one dataset per label, named `<project>_<label>` (for example
`3GPPTR38_901_4G_uniform_ue_kpi`), each with a `<project>_<label>.schema.md` file that lists its columns; the project
name is `<scenario_model>_<ue_distribution>`. A relative `output.directory` is resolved from the folder in which
Giulia is started.

The legacy NPZ archive (`results-raw.npz`) and, when `plots` is on, the `to_plot_*.npz` files are always written to
`outputs/results/<project>/` inside the repository, whatever `output.directory` says.

Depending on your configured formats, the output will vary. We will focus now on the Parquet output.
If you have set `output.formats.parquet.enable` to `true`, the output will be exported into Parquet.

If you have enabled `output.formats.parquet.use_partition_cols` the data will be stored into a Parquet [DataSet](https://arrow.apache.org/docs/python/generated/pyarrow.dataset.Dataset.html).
Otherwise, all the data of each label will be placed into a single Parquet file. When `use_partition_cols` is not
specified, or the whole `formats` block is missing, Giulia uses the partitioned layout; the `config.yml` and
`config.example.yml` of the repository set it to `false`.

To load this data with Pyarrow, you have two options:
1. Loading a DataSet:
    ```python
    import pyarrow.dataset as ds
    # Update the path to your output accordingly
    dataset = ds.dataset('outputs/gof/3GPPTR38_901_4G_uniform_ue_kpi', format='parquet')
    table = dataset.to_table()
    ```
2. Loading a single Parquet file:
    ```python
    import pyarrow.parquet as pq
    # Update the path to your output accordingly
    table = pq.read_table('outputs/gof/3GPPTR38_901_4G_uniform_ue_kpi.parquet')
    ```

**GOF files:**

If you have set `output.formats.parquet.compression` to `zip`, and enabled partition cols, the data will be also
compressed into a so called GOF file (Giulia Output File). This is a zip file containing all the Parquet files.

Giulia provides an utility function to load those GOF files very easily, though you can of course simply decompress the
zip file, and use the data inside without any issues.

To use this function:
```python
from giulia.outputs.utils import load_gof

dataset = load_gof('outputs/gof/3GPPTR38_901_4G_uniform_ue_kpi.gof')
table = dataset.to_table()
```
