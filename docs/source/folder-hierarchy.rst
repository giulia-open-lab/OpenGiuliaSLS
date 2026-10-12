Folder Hierarchy
================

The repository is divided into multiple folders.
This structure is very clear, and aims to keep the code organized and easy to navigate.
The main folders are:

- ``giulia``: The source code of the simulator. See :doc:`architecture` for what each sub-package does.
- ``examples``: Runnable entry points; ``main.py`` resolves ``run.example`` to one of these.
- ``config.example.yml``: The commented template configuration. Copy it to ``config.yml`` to get started.
- ``data``: Auxiliary data files required by Giulia (3D meshes for ray tracing, cached pairing results).
- ``shadowing``: The pre-computed shadowing maps (``.mat``, downloaded separately and not tracked by Git or Git LFS) and the MATLAB scripts that generate them.
- ``scripts``: The installer and auxiliary tooling (shadowing download, GPU monitoring, catalog generation).
- ``tests``: The regression harness, its scenario definitions and version-controlled reference files, plus the unit tests. See :doc:`testing`.
- ``docs``: The source files of this documentation.
- ``env``: Auxiliary Conda environment files. The runtime ``environment.yml`` lives at the repository root instead.
- ``plots``: Plotting and result post-processing scripts.
- ``dedalo``: The configuration interface to Giulia — the layer through which an external script, optimizer or agent modifies a live simulation. Not on the ``main.py`` / YAML path, but the documented write surface for external optimization and agent workflows. See the ``docs/dedalo/`` notes.
- ``optimizer``: A standalone research sub-project built on top of Giulia; not part of the simulation pipeline.
- ``outputs``: Default destination for simulation results and test reports. Gitignored.
- ``.github``: The GitHub Actions workflows.
- ``.vscode``: Shared editor configuration.
