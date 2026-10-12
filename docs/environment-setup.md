# Environment setup

Giulia targets Linux machines with an NVIDIA GPU. The installer scripts assume a Conda installation and use a mix of `pip` and `conda` to install the toolchain.

## Prerequisites

- Linux. The installer is a Bash script and uses `conda activate` from a bash-compatible shell.
- Python 3.11 or newer. This is enforced by `pyproject.toml` (`requires-python = ">=3.11"`), required by Sionna 2, and is the version the installer creates new Conda environments with.
- Conda (Miniconda or Anaconda) available on the `PATH`.
- An NVIDIA GPU with a driver compatible with CUDA 13.0 is strongly recommended. Giulia can run on CPU, but the channel-modelling and link-level stages are written assuming GPU acceleration.
- Sufficient disk space — the GPU stack alone (PyTorch plus the CUDA runtime Sionna pulls in) is several gigabytes.
- An internet connection during installation: the installer downloads packages from PyPI and (by default) downloads precomputed shadowing maps.

## Installer script

The supported installation flow is `scripts/installer.sh`. It accepts the following flags:

```shell
bash scripts/installer.sh \
  --install-requirements \
  --install-dev-requirements \
  --env-name giulia
```

The flags are:

- `--install-requirements` — install runtime requirements (the scientific stack pinned in `scripts/installer_dependencies.sh`).
- `--install-dev-requirements` — install development extras (`pylint`).
- `--current-environment` — install everything into the currently active Conda environment instead of creating a new one.
- `--env-name <name>` — create a fresh Conda environment with the given name (Python 3.11) and install into it.
- `--no-shadowing-download` — skip the automatic download of the precomputed shadowing maps that ship under `shadowing/`. Use this if you intend to regenerate them yourself with the MATLAB scripts in `shadowing/`.

If you omit `--install-requirements`, `--install-dev-requirements`, `--env-name` or `--current-environment`, the script prompts interactively.

> **Note.** `scripts/installer.sh` also accepts an `--install-llm-requirements` flag. It is currently
> a no-op: the script parses it but installs nothing extra. Ignore it until it is implemented.

### Reusing an existing environment

If you already have a Conda environment with Python 3.11 and just want to install Giulia's dependencies into it, activate the environment and run `scripts/installer_dependencies.sh` directly:

```shell
conda activate my-env
bash scripts/installer_dependencies.sh
```

This skips the environment-creation step and installs the same scientific stack.

## What the installer installs

Both `scripts/installer.sh` and `scripts/installer_dependencies.sh` install the same fixed set of
pinned packages, reproducing the validated reference environment (Python 3.11):

- **Sionna 2.0.1** (`sionna==2.0.1`), which is PyTorch-based. It pulls in **PyTorch** and its bundled
  **CUDA 13.0 runtime** automatically — PyTorch is deliberately *not* installed separately, and there
  is no PyTorch wheel index or CUDA/cuDNN Conda channel involved. Do not install torch yourself: it
  would overwrite the build Sionna resolves.
- The scientific stack: `numpy==2.4.6`, `scipy==1.17.1`, `pandas==3.0.3`, `scikit-learn==1.9.0`,
  `matplotlib==3.11.0`, `seaborn==0.13.2`, `shapely==2.1.2`, `geopandas==1.1.4`, `pyproj==3.7.2`,
  `trimesh[easy]==4.12.2`, `mitsuba==3.8.0`, `astropy==8.0.1`, `itur==0.4.0`, `pyarrow==25.0.0`,
  `PyYAML==6.0.3`, `psutil==7.2.2`, `nvidia-ml-py==13.610.43`, `tqdm==4.68.4`, plus `requests`,
  `tomlkit` and `rich`.
- Development extras (when requested): `pylint`.

After installation the installer purges the pip cache and runs `conda clean --all -y`.

> **Giulia no longer depends on TensorFlow.** Sionna 2 moved to a PyTorch backend, and the TensorFlow
> dependency was dropped along with it. If you find a TensorFlow reference anywhere, it is stale.

> The pins above are duplicated in `scripts/installer.sh`, `scripts/installer_dependencies.sh` and
> `environment.yml`, and must be kept in sync. `environment.yml` (at the repository root, not under
> `env/`) is the full bit-for-bit Conda snapshot, including the exact PyTorch build; the installer
> scripts hold the curated list of direct dependencies.

## Activating the environment

If you let the installer create a new Conda environment, activate it before running the simulator:

```shell
conda activate giulia
```

(Replace `giulia` with whatever you passed to `--env-name`.)

## Verifying the GPU

A standalone script exercises a small neural-network training step on the GPU and reports which device was used. Run it from the activated environment:

```shell
python scripts/test_GPU_SimpleTrainNN_Torch.py
```

It is intended as a quick sanity check: it prints the visible CUDA devices, runs a few training iterations, and exits. If it falls back to CPU silently or fails to import the GPU build, your CUDA setup is not picked up correctly.

For a heavier workload that compares GPU and CPU on the same model, the repository also ships `scripts/Comparison_GPUandCPU_BigModel_Torch.py`.

## Building the documentation

The user docs (this site) and the developer docs share a Sphinx setup under `docs/`. The Sphinx config lives in `docs/source/conf.py`, with build infrastructure in `docs/Makefile` and `docs/make.bat`.

The docs build has its **own** dependency set, separate from the simulator's — you do not need the
GPU stack to build it. Install it either with pip:

```shell
pip install -r docs/requirements.txt
make -C docs html
```

or as a dedicated Conda environment (this is what CI uses):

```shell
conda env create -n giulia-docs --file env/environment-docs.yml
conda activate giulia-docs
make -C docs html
```

Keep the two in sync — they pin the same packages. The output is written to `docs/build/html`.

A Docker-based workflow is also provided under `docs/`:

- `docs/docker-compose.yml` — builds the image from `docs/Dockerfile` (which runs the Sphinx build
  inside the container) and serves it on port **8080**.
- `docs/docker-compose.dev.yml` — serves the **already built** `docs/build/html` through a volume on
  port **1080**, so you only need to re-run `make -C docs html` and refresh the page.
- `docs/docker-compose.prod.yml` — pulls the published `iteamupv/giulia:latest` image and exposes no
  port, expecting a reverse proxy in front.

See [`docs/README.md`](README.md) for the full Docker workflow.
