#!/bin/bash

# Script Name: installer_dependencies.sh
#
# Description:
#   This script sets up an existing Conda-based Python environment with all required
#   dependencies for Giulia. It installs libraries, along with CUDA and cuDNN support for TensorFlow
#   and PyTorch. Development tools and additional utilities are also included.
#
# Key components installed:
#   - TensorFlow 2.17.0 with CUDA 12.8 and cuDNN 9.7.1
#   - PyTorch 2.7.1 (with torchvision & torchaudio) built for CUDA 12.8
#   - CUDA NVCC compiler (12.8.1)
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


# See DEPENDENCIES.md for more information about the specifics of the following installations.
# TLDR; Torch 2.7.1, TensorFlow 2.17.0, CUDA 12.8, cuDNN 9.3

# Install TensorFlow with CUDA support
# See available versions: https://pypi.org/project/tensorflow/#history
pip install "tensorflow[and-cuda]==2.17.0" 

# Install Torch 2.7.1 with CUDA 12.8
# Note that it has to be installed before 
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128 

# Install CUDA NVCC
# Use Nvcc for CUDA 12.8 as per TensorFlow requirement.
# See available versions: https://anaconda.org/nvidia/cuda-nvcc
conda install nvidia/label/cuda-12.8.1::cuda-nvcc -y 

# Install cuDNN
# Use cuDNN 9.7.1 as per TensorFlow requirement.
# See available versions: https://developer.nvidia.com/rdp/cudnn-archive
# FIXME: This downgrades sqlite 3.45.3-h5eee18b_0 --> 3.31.1-h7b6447c_0
conda install conda-forge::cudnn=9.7.1 -y 
# conda install conda-forge::sqlite=3.45.3 -y || exit 1


# Install other dependencies
pip install $DEV_REQUIREMENTS 
pip install $REQ 


# Eliminate cached files during download
pip cache purge
conda clean --all -y