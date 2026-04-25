# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 2024

This script convert all saved npz files with the prefix 'to_plot_'
to mat equivalents and saves them in a different folder

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import os

import numpy as np
import scipy.io


# Function to convert npz to mat and save in a different folder
def convert_npz_to_mat(source_folder, target_folder, file_prefix):
    # Ensure the target folder exists, if not, create it
    os.makedirs(target_folder, exist_ok=True)

    # Get a list of all npz files in the directory
    npz_files = [f for f in os.listdir(source_folder) if f.startswith(file_prefix) and f.endswith('.npz')]

    # For each npz file, convert it to a mat file
    for npz_file in npz_files:
        data = np.load(os.path.join(source_folder, npz_file))
        # npz_file is the full filename, strip it to just the base name without extension
        mat_filename = os.path.splitext(npz_file)[0] + '.mat'
        scipy.io.savemat(os.path.join(target_folder, mat_filename), dict(data))


# Set the folder where your .npz files are located
project_name = 'ITU_R_M2135_UMa_uniform_20240214'
source_folder = "results_to_process/results_" + project_name + "/"

# Set the target folder where you want to save the .mat files
target_folder = "results_to_process/results_" + project_name + "_mat/"

# Convert all npz files with the prefix 'to_plot_' and save them in the target folder
convert_npz_to_mat(source_folder, target_folder, 'to_plot_')
