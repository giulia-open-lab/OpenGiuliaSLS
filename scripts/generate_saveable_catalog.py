#!/usr/bin/env python3
"""
Generate ``docs/output-saveable/saveable-catalog.md``: the list of every Saveable variable
available under each Giulia preset.

The Saveable registry is only populated at runtime, so this script runs a
minimal Giulia simulation (one short episode, no saving, no plots, CPU) and then
dumps the registered Saveables. That registry is scoped to an execution context and
is *never* cleared within one, so running two presets sequentially on the same
thread would let the first preset's Saveables leak into the second catalog. To keep
the catalogs clean, each preset is run in its own **subprocess** (which also gives
each preset a clean PyTorch/GPU state).
See ``docs/output-saveable/saveable-concurrency.md``.

Usage:
    python scripts/generate_saveable_catalog.py

Internally the script re-invokes itself once per preset in "worker" mode
(``--emit``); you normally never pass that flag by hand.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

# The repository root is the parent of this script's directory.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, ROOT_DIR)

# Presets to document: (preset name, human-readable label, scenario_model).
# Edit this list to change the scenario used for a preset.
PRESETS = [
    ("GiuliaStd", "Standard preset", "ITU_R_M2135_UMa"),
    ("GiuliaMfl", "Multi-frequency layer preset", "ITU_R_M2135_UMa"),
]

OUTPUT_MD = os.path.join(ROOT_DIR, "docs", "output-saveable", "saveable-catalog.md")

# Everything below this marker in OUTPUT_MD is generated; everything above it
# (the hand-written tutorial) is preserved across runs.
GENERATED_MARKER = (
    "<!-- BEGIN GENERATED CATALOG — managed by "
    "scripts/generate_saveable_catalog.py; edits below this line are overwritten -->"
)

# Used only to create OUTPUT_MD from scratch when it (or the marker) is missing.
# Keep this in sync with the tutorial at the top of docs/output-saveable/saveable-catalog.md.
_DEFAULT_HEADER = """# Saveable Catalog

This document is a **reference of every Saveable variable that Giulia can expose
at runtime**, grouped by simulation preset. Use it to find the exact key to pass
to the [`Monitor`](SaveAble_monitor.md) or to `Saveable.get()`.

> The catalog lists at the **end** of this file are **auto-generated**.
> Everything above the generated marker — this tutorial — is hand-written and is
> preserved when the script is re-run.

## What is in the catalog

Each entry is a key in the `<ClassName>.<variable_name>` format, for example
`Distance_Angles_ue_to_cell.distances_b_to_a_3d_m`. This is exactly the string
accepted by:

- `Saveable.get('<ClassName>.<variable_name>')` — read the current value, and
- the [`Monitor`](SaveAble_monitor.md) — watch one or more of them live during a run.

When a class documents a variable through `expose_variable_descriptions()`, that
description is rendered as an indented sub-bullet right below the key:

```
### `BaseStations`
- `BaseStations.cell_radio_unit_statuses`
  - cell_radio_unit_statuses this is its desc
```

Variables whose class declares no description are listed without a sub-bullet.

See [Saveable Objects](../source/saveable-objects.rst) for how a class becomes a
Saveable.

## Why it is split by preset

The set of available variables is **not fixed**: it depends on which modules the
simulation instantiates, which in turn depends on the **preset** and the
**scenario**. Giulia ships two presets:

- `GiuliaStd` — standard simulation.
- `GiuliaMfl` — multi-frequency-layer simulation.

Each is listed separately below, because a class present in one may be absent in
the other.

## How the catalog is generated

The Saveable registry only fills up while a simulation runs, so the lists are
produced by
[`scripts/generate_saveable_catalog.py`](../scripts/generate_saveable_catalog.py).
For each preset it:

1. Runs a **minimal** Giulia simulation (one short step, no saving, no plots, CPU).
2. Reads the registered Saveables and the variables they expose.
3. Writes the result into the generated section at the end of this file.

Each preset is run in its **own subprocess**. The registry is scoped to an
execution context and is never cleared within one, so running both presets
sequentially on the same thread would let the first preset's variables leak into
the second list. A subprocess gives each preset a context of its own — plus a
clean PyTorch/GPU state. See
[Saveable Concurrency](saveable-concurrency.md).

## Regenerating this file

From the repository root, run:

```
python scripts/generate_saveable_catalog.py
```

This **overwrites only the generated section** at the end (below the marker);
this tutorial is left untouched. If the file — or the generated section — does
not exist yet, it is created.

> **Note.** Because the catalog depends on the preset and scenario, treat it as a
> snapshot of those specific runs. To document a different scenario, edit the
> `PRESETS` table at the top of the script and regenerate.

## Configuration & the link to `config.yml`

The catalog is **independent of `config.yml`**: it is not configured by it and does **not** honour
its `output.fields` filter. Every regeneration lists **all** variables each preset exposes. The
only "configuration" the catalog has is the `PRESETS` table at the top of
[`scripts/generate_saveable_catalog.py`](../../scripts/generate_saveable_catalog.py) — a list of
`(preset, label, scenario_model)` tuples:

```python
PRESETS = [
    ("GiuliaStd", "Standard preset", "ITU_R_M2135_UMa"),
    ("GiuliaMfl", "Multi-frequency layer preset", "ITU_R_M2135_UMa"),
]
```

To document a different scenario (or add a preset), edit that table and re-run
`python scripts/generate_saveable_catalog.py`. There is **no `config.yml` step**.

The flow is **one-directional — the catalog feeds `config.yml`, never the reverse:**

```
edit PRESETS  ->  python scripts/generate_saveable_catalog.py  ->  this catalog (every key, per preset)
                                                                        |
                                            copy the keys you want      v
                              config.yml  output.fields: [ <Class>.<var>, ... ]
                                                                        |
                                                  run the simulator     v
                                                        filtered Parquet output
```

Use this catalog as the **menu** of valid `<Class>.<variable>` keys, then paste the subset you want
into `output.fields` in [`config.yml`](../../config.yml) to trim a real run's output (see
[Saveable Output & Export §6](saveable-output.md#6-selecting-which-columns-to-export-the-fields-filter)).
Leaving `output.fields` empty exports everything; the catalog itself is unaffected either way.
"""


# --------------------------------------------------------------------------- #
# Worker mode: run one preset and dump its Saveable catalog as JSON.
# --------------------------------------------------------------------------- #

def _run_preset(preset: str, scenario: str) -> dict:
    """
    Run a minimal Giulia simulation with the given preset and scenario, then
    return its Saveable catalog as ``{class_name: [variable, ...]}``.
    """
    from giulia.outputs.saveable import Saveable
    from giulia.tools.tools import free_memory

    # This is a throwaway introspection run (no saving, no plots): only real errors
    # matter, so silence Giulia's own WARNING-level logs (e.g. the Beam_Conf
    # antenna-count notes) by raising the log level to ERROR.
    from giulia.logger import set_log_level, LogLevel
    set_log_level(LogLevel.ERROR)

    Giulia = _get_giulia_class(preset)

    # Same positional arguments the EV example uses, but with everything turned
    # down to the minimum: no result saving, no plots, a single episode, CPU.
    #   (preset, scenario_model, ue_playground_model, ue_distribution,
    #    ue_mobility, link_direction, wraparound, save_results, plots,
    #    number_of_episods, regression, enable_GPU)
    g = Giulia(preset, scenario, None, "uniform", None, "downlink", None,
               0, 0, 1, 0, False)

    # One short episode is enough to instantiate (and therefore register) every
    # Saveable a normal run would. Saving the episode only *reads* the registry,
    # so it is skipped here.
    max_simulation_time_us = 1e6
    decision_making_interval_us = 1e6
    number_of_steps = int(max_simulation_time_us / decision_making_interval_us)

    g.configure(0, max_simulation_time_us)
    for step_index in range(number_of_steps):
        g.run_simulation(step_index, decision_making_interval_us)

    # Build the catalog *before* freeing memory, so the result holds plain
    # strings and no longer depends on the simulation objects. Names are unique
    # (the registry de-duplicates by class name), so a dict is safe. Each entry
    # keeps the exposed variables and their human-readable descriptions (from
    # ``expose_variable_descriptions``, empty when a class declares none).
    catalog = {
        obj.name: {
            "variables": list(obj.expose_variables()),
            "descriptions": dict(obj.expose_variable_descriptions()),
        }
        for obj in Saveable.registry()
    }

    # Free CPU/GPU memory (PyTorch) and run the garbage collector,
    # like the EV example does at the end of a run.
    free_memory()

    return catalog


def _get_giulia_class(preset: str):
    """Import and return the Giulia class for the given preset."""
    if preset == "GiuliaMfl":
        from giulia.presets.g_multi_frequency_layer_preset import Giulia
    elif preset == "GiuliaStd":
        from giulia.presets.g_standard_preset import Giulia
    else:
        raise ValueError(f"Invalid preset '{preset}'. Use 'GiuliaStd' or 'GiuliaMfl'.")
    return Giulia


# --------------------------------------------------------------------------- #
# Driver mode: run each preset in a subprocess and write the markdown.
# --------------------------------------------------------------------------- #

def _collect_catalog(preset: str, scenario: str) -> dict:
    """Run one preset in an isolated subprocess and read back its catalog."""
    fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix=f"catalog_{preset}_")
    os.close(fd)
    try:
        print(f"    - Running {preset}...", flush=True)
        result = subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--emit", "--preset", preset, "--scenario", scenario,
             "--out", tmp_path],
            cwd=ROOT_DIR,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"The worker for preset '{preset}' exited with code "
                f"{result.returncode}. See its output above."
            )
        with open(tmp_path, "r", encoding="utf-8") as f:
            return json.load(f)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _render_catalog(catalogs: list) -> str:
    """
    Render the generated section, i.e. everything that goes *below* the marker.

    Each variable is listed as ``- `<Class>.<variable>``` and, when the class
    provides one via ``expose_variable_descriptions``, its description is rendered
    as an indented sub-bullet right below it.

    Args:
        catalogs: a list of ``(preset, label, scenario, catalog)`` tuples, where each
            ``catalog`` maps ``class_name`` to ``{"variables": [...], "descriptions": {var: desc}}``.
    """
    lines = [
        "_These lists are generated by the script and reflect the presets in its"
        " `PRESETS` table._",
    ]

    for preset, label, scenario, catalog in catalogs:
        n_vars = sum(len(entry["variables"]) for entry in catalog.values())
        lines += [
            "",
            "---",
            "",
            f"## {label} — `{preset}`",
            "",
            f"{len(catalog)} Saveables · {n_vars} variables.",
            "",
        ]
        for class_name in sorted(catalog):
            entry = catalog[class_name]
            descriptions = entry["descriptions"]
            lines.append(f"### `{class_name}`")
            for var in entry["variables"]:
                lines.append(f"- `{class_name}.{var}`")
                description = descriptions.get(var)
                if description:
                    lines.append(f"  - {description}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _write_output(generated: str) -> None:
    """
    Write the generated section into OUTPUT_MD, *below* ``GENERATED_MARKER``.

    The hand-written tutorial above the marker is preserved. When the file (or
    the marker) does not exist yet, it is created from ``_DEFAULT_HEADER``.
    """
    existing = None
    if os.path.exists(OUTPUT_MD):
        with open(OUTPUT_MD, "r", encoding="utf-8") as f:
            existing = f.read()

    if existing is not None and GENERATED_MARKER in existing:
        # Keep whatever the user has above the marker; replace what is below it.
        head = existing.split(GENERATED_MARKER)[0].rstrip()
    else:
        head = _DEFAULT_HEADER.rstrip()

    content = f"{head}\n\n{GENERATED_MARKER}\n\n{generated.strip()}\n"
    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", action="store_true",
                        help="Worker mode: run one preset and write its catalog "
                             "as JSON to --out. Used internally.")
    parser.add_argument("--preset", help="Preset to run (worker mode).")
    parser.add_argument("--scenario", help="scenario_model to use (worker mode).")
    parser.add_argument("--out", help="Where to write the catalog JSON (worker mode).")
    args = parser.parse_args()

    if args.emit:
        catalog = _run_preset(args.preset, args.scenario)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(catalog, f)
        return

    # Driver mode: one isolated subprocess per preset, then write the markdown.
    preset_names = ", ".join(preset for preset, _, _ in PRESETS)
    print(f"Generating Saveable catalog for presets: {preset_names}")
    catalogs = [
        (preset, label, scenario, _collect_catalog(preset, scenario))
        for preset, label, scenario in PRESETS
    ]

    _write_output(_render_catalog(catalogs))

    print(f"Updated {OUTPUT_MD}")


if __name__ == "__main__":
    main()
