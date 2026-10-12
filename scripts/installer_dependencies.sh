#!/bin/bash

# Script Name: installer_dependencies.sh
#
# Description:
#   This script sets up an existing Conda-based Python environment with all required
#   dependencies for Giulia, reproducing the validated reference environment
#   ("giulia-no-tf", Python 3.11) with exact package versions.
#
# Key components installed:
#   - Sionna 2.0.1 (PyTorch-based); it pulls in PyTorch and its bundled CUDA
#     runtime automatically (no dedicated PyTorch install / index URL needed)
#   - Core scientific stack (NumPy 2.4.6, Pandas 3.0.3, SciPy 1.17.1,
#     scikit-learn 1.9.0, matplotlib 3.11.0, seaborn 0.13.2)
#   - Specialized libraries (Itur, Astropy, Shapely, GeoPandas, PyProj, Trimesh, Mitsuba)
#   - Developer tools (pylint)
#
# Notes:
#   - Requires Conda to be installed and accessible in PATH.
#   - Requires Python 3.11 or newer (Sionna 2 requirement).
#   - Keep these pins in sync with scripts/installer.sh and environment.yml.
#   - Cleans up cached files after installation to save space.

    
# Create conda environment (uncomment if needed)
# conda create -n giulia python=3.11 -y; conda activate giulia || exit 1


# Initialize requirements variables.
# Notes:
#   - torch is NOT listed here: it is installed automatically as a dependency of
#     Sionna 2, together with its bundled CUDA runtime. Do not install it
#     separately, to avoid overwriting the build Sionna resolves.
#   - pyproj is a runtime dependency (used by giulia/tools/tools.py).
#   - requests and tomlkit are only needed by scripts/download_shadowing_files.py
#     and examples/giulia_EV.py respectively.
REQ="sionna==2.0.1 numpy==2.4.6 scipy==1.17.1 pandas==3.0.3 scikit-learn==1.9.0 matplotlib==3.11.0 seaborn==0.13.2 shapely==2.1.2 geopandas==1.1.4 pyproj==3.7.2 trimesh[easy]==4.12.2 mitsuba==3.8.0 astropy==8.0.1 itur==0.4.0 pyarrow==25.0.0 PyYAML==6.0.3 psutil==7.2.2 nvidia-ml-py==13.610.43 tqdm==4.68.4 requests tomlkit rich"
DEV_REQUIREMENTS="pylint"


# Install dependencies (torch and its CUDA runtime are pulled in automatically
# by Sionna 2)
pip install $DEV_REQUIREMENTS
pip install $REQ


# Eliminate cached files during download
pip cache purge
conda clean --all -y
