# Giulia

[![Documentation](https://a11ybadges.com/badge?text=Documentation&logo=book-open)](http://158.42.160.122/)
[![License](https://a11ybadges.com/badge?text=License&logo=shield)](https://github.com/david-lopez-perez/Giulia/blob/main/LICENSE)

[![Matrix](https://a11ybadges.com/badge?logo=matrix)](https://matrix.to/#/!hbiCQOdYflhWRcHwyT:matrix.org?via=matrix.org)
[![Docker Hub](https://a11ybadges.com/badge?text=DockerHub&logo=docker&badgeColor=#4287f5)](https://hub.docker.com/repository/docker/iteamupv/giulia/general)

**Data-driven system level simulator**

This repository hosts a one-of-a-kind system-level simulator developed in Python. It is aimed at encompassing both
cellular and Wi-Fi capabilities and is based on an event-driven philosophy.

To run the simulator, use the 'main.py' function and select the desired use case or scenario for simulation.

---

![Attention](https://a11ybadges.com/badge?text=Attention&logo=alert-triangle&badgeColor=rgb(235,180,28))

**Please note that the simulator is currently under construction. We appreciate your patience.**

---

### Dependencies
This simulator is using Sionna 2 by NVIDIA (PyTorch-based), which requires PyTorch 2.9+ and Python 3.11+.

The validated reference environment is Python 3.11 with Sionna 2.0.1, which brings in PyTorch 2.13.0 and
its CUDA 13.0 runtime automatically. The curated direct dependencies are pinned in the installer scripts
([`scripts/installer.sh`](scripts/installer.sh),
[`scripts/installer_dependencies.sh`](scripts/installer_dependencies.sh)); the full, bit-for-bit Conda
snapshot (including the exact PyTorch build) is in [`environment.yml`](environment.yml).

Other related packages: NumPy; Pandas; SciPy; Shapely; GeoPandas; PyProj; MatplotLib; Seaborn; scikit-learn; itur; Astropy; Trimesh; Mitsuba;

See [Environment setup](docs/environment-setup.md) to know more about the installation process.

### Shadowing maps and Git LFS

The spatially correlated shadowing maps (14 `*.mat` files, about 2.7 GB) are **not** stored in the repository,
neither in Git nor in Git LFS. Giulia looks for them in `shadowing/` at the root of the repository, or in the folder
given by the `SHADOWING_FILES` environment variable, and stops with
`AssertionError: No .mat files found in the shadowing directory.` when it finds none.

To get them, run:

```shell
python scripts/download_shadowing_files.py
```

The download server is reachable only from the UPV network (or its VPN); otherwise, ask a UPV member to share the
files. The script unpacks the archive into the root of the repository: check that the `*.mat` files end up inside
`shadowing/` (move them there if needed), or set `SHADOWING_FILES` to the folder that contains them.

[Git LFS](https://git-lfs.com/) is used only for the precomputed test references under `tests/references/`
(about 665 MB), which are needed only to run the regression tests. To fetch them:

```shell
git lfs install
git lfs pull
```

### Running Giulia

The easiest way to run Giulia is to use configuration files.
Those allow you to easily select how to run Giulia, and which outputs you want to get.

`main.py` reads `config.yml` at the root of the repository; if that file does not exist, it is created from the
fully commented template [`config.example.yml`](config.example.yml). Note that the repository already contains a
`config.yml`, which runs a 4G/5G multilayer scenario: this scenario is heavy and needs a GPU with a large memory.
For a first, light run, use the template directly (ITU-R M.2135 UMa scenario):

```shell
python main.py config.example.yml
```

Running `python main.py` without an argument uses `config.yml`. Both commands expect a valid environment with all
the required dependencies installed.

Get more detailed information in the [Getting Started guide](GETTING_STARTED.md).

### Running tests

Giulia is tested by running different models and distributions, and comparing them to some precomputed ones.

Know more in the [tests guide](tests/README.md).
