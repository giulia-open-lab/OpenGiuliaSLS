#!/bin/bash

# Script Name: installer_dependencies.sh
#
# Description:
#   This script sets up an existing Conda-based Python environment with all required
#   dependencies for Giulia. It installs libraries, along with CUDA and cuDNN support for TensorFlow
#   and PyTorch. Development tools and additional utilities are also included.
#
# Key components installed:
#   - PyTorch 2.7.1 (with torchvision & torchaudio) built for CUDA 12.8, GPU backend enabled
#   - TensorFlow 2.17.0 without GPU support (to avoid compatibility issues with Mitsuba)
#   - Core scientific stack (NumPy, Pandas, SciPy, scikit-learn, matplotlib, seaborn, etc.)
#   - Specialized libraries (Sionna, Simulus, Itur, Astropy, Shapely, GeoPandas, Trimesh, Mitsuba, etc.)
#   - Developer tools (pylint, pyproj)
#
# Notes:
#   - Requires Conda to be installed and accessible in PATH.
#   - Cleans up cached files after installation to save space.



# Create conda environment 
# conda create -n giulia python=3.11 -y
# conda activate giulia || exit 1


# Initialize requirements variables
REQ="seaborn==0.13.2 sionna==1.1.0 scikit-learn==1.7.0 numpy==1.26.4 pandas==2.3.0 scipy==1.16.0 matplotlib==3.10.3 simulus==1.2.1 itur==0.4.0 shapely==2.1.1 geopandas==1.1.1 tqdm==4.67.1 astropy==7.1.0 trimesh[easy]==4.6.13 mitsuba==3.6.2 nvidia-ml-py==12.575.51 pyarrow==20.0.0"
DEV_REQUIREMENTS="pylint pyproj==3.7.0"


# Install Torch 2.7.1 with CUDA 12.8
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128 


# Install other dependencies
pip install $DEV_REQUIREMENTS 
pip install $REQ 


# post install tensorflow to ensure version 2.17 is available
# tensorflow 2.21 (installed by sionna) has compatibility issues with mitsuba, breaking the program at import time. 
pip install "tensorflow==2.17.0" 


# Eliminate cached files during download
pip cache purge
conda clean --all -y