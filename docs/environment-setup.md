# Environment setup

Giulia targets Linux machines with an NVIDIA GPU. The installer scripts assume a Conda installation and use a mix of `pip` and `conda` to install the toolchain.

## Prerequisites

- Linux. The installer is a Bash script and uses `conda activate` from a bash-compatible shell.
- Python 3.11. This is enforced by `pyproject.toml` (`requires-python = ">=3.11"`) and is the version the installer creates new Conda environments with.
- Conda (Miniconda or Anaconda) available on the `PATH`.
- An NVIDIA GPU with a driver compatible with CUDA 12.8 is strongly recommended. Giulia can run on CPU, but the channel-modelling and link-level stages are written assuming GPU acceleration.
- Sufficient disk space — the GPU stack alone (TensorFlow + PyTorch + CUDA tooling) is several gigabytes.
- An internet connection during installation: the installer downloads packages from PyPI, the PyTorch CUDA index, the NVIDIA Conda channel, and conda-forge, and (by default) downloads precomputed shadowing maps.

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
- `--install-dev-requirements` — install development extras (`pylint`, `pyproj`).
- `--install-llm-requirements` — install LLM-related extras (only relevant if you are running the LLM-based components).
- `--current-environment` — install everything into the currently active Conda environment instead of creating a new one.
- `--env-name <name>` — create a fresh Conda environment with the given name (Python 3.11) and install into it.
- `--no-shadowing-download` — skip the automatic download of the precomputed shadowing maps that ship under `shadowing/`. Use this if you intend to regenerate them yourself with the MATLAB scripts in `shadowing/`.

If you omit `--install-requirements`, `--install-dev-requirements`, `--env-name` or `--current-environment`, the script prompts interactively.

### Reusing an existing environment

If you already have a Conda environment with Python 3.11 and just want to install Giulia's dependencies into it, activate the environment and run `scripts/installer_dependencies.sh` directly:

```shell
conda activate my-env
bash scripts/installer_dependencies.sh
```

This skips the environment-creation step and installs the same scientific stack.

## What the installer installs

Both `scripts/installer.sh` and `scripts/installer_dependencies.sh` install a fixed set of pinned packages:

- TensorFlow 2.17.0 with CUDA support: `tensorflow[and-cuda]==2.17.0`.
- PyTorch 2.7.1 (with `torchvision==0.22.1` and `torchaudio==2.7.1`) built for CUDA 12.8, from the PyTorch wheel index `https://download.pytorch.org/whl/cu128`.
- CUDA NVCC 12.8.1 from the `nvidia/label/cuda-12.8.1` Conda channel.
- cuDNN 9.7.1 from `conda-forge`.
- The scientific stack pinned in `scripts/installer_dependencies.sh`: `seaborn==0.13.2`, `sionna==1.1.0`, `scikit-learn==1.7.0`, `numpy==1.26.4`, `pandas==2.3.0`, `scipy==1.16.0`, `matplotlib==3.10.3`, `simulus==1.2.1`, `itur==0.4.0`, `shapely==2.1.1`, `geopandas==1.1.1`, `tqdm==4.67.1`, `astropy==7.1.0`, `trimesh[easy]==4.6.13`, `mitsuba==3.6.2`, `nvidia-ml-py==12.575.51`, `pyarrow==20.0.0`.
- Development extras (when requested): `pylint`, `pyproj==3.7.0`.

After installation the installer purges the pip cache and runs `conda clean --all -y`.

> Note: if you read the README at the repo root, it shows older Python (3.8–3.11) and TensorFlow (2.10–2.15) ranges. Those numbers are out of date. The authoritative versions are the ones in `pyproject.toml` and the installer scripts.

## Activating the environment

If you let the installer create a new Conda environment, activate it before running the simulator:

```shell
conda activate giulia
```

(Replace `giulia` with whatever you passed to `--env-name`.)

## Verifying the GPU

Two standalone scripts exercise a small neural-network training step on the GPU and report which device was used. Run them from the activated environment:

```shell
python scripts/test_GPU_SimpleTrainNN_Tensorflow.py
python scripts/test_GPU_SimpleTrainNN_Torch.py
```

Both scripts are intended as quick sanity checks: they print the visible CUDA devices, run a few training iterations, and exit. If either one falls back to CPU silently or fails to import the GPU build, your CUDA / cuDNN setup is not picked up correctly.

For a heavier workload that compares GPU and CPU on the same model, the repository also ships `scripts/Comparison_GPUandCPU_BigModel_Tensorflow.py` and `scripts/Comparison_GPUandCPU_BigModel_Torch.py`.

## Building the documentation

The user docs (this site) and the developer docs share a Sphinx setup under `docs/`. The Sphinx config lives in `docs/source/conf.py`, with build infrastructure in `docs/Makefile` and `docs/make.bat`.

To build the HTML site locally, install the docs dependencies and run `make html` from `docs/`:

```shell
pip install -r docs/requirements.txt
make -C docs html
```

The output is written to `docs/build/html`.

A Docker-based workflow is also provided under `docs/`:

```shell
docker compose -f docs/docker-compose.yml up
```

`docs/docker-compose.dev.yml` and `docs/docker-compose.prod.yml` are variants for local development and production previews respectively.
