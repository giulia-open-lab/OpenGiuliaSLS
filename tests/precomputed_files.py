"""
Created on Mon Oct 23 16:39:49 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import os
import shutil
import sys

from typing import Optional

# Configure Giulia into path
this_dir = os.path.dirname(__file__)
root_dir = os.path.join(this_dir, '..')
sys.path.insert(1, root_dir)

from tests.const import precomputed_files_url, precomputed_files_list
from tests.fs import download, unzip

precomputed_files_dir: str = os.path.realpath(os.path.join(this_dir, 'regression_test_dlp'))
precomputed_files_dir: str = os.path.realpath(os.path.join(this_dir, 'regression_test_dlp'))
precomputed_files_zip: str = os.path.realpath(os.path.join(this_dir, 'precomputed_files.zip'))


def download_precomputed_files():
    download(precomputed_files_url, precomputed_files_zip, "Precomputed files download")
    print("Precomputed files downloaded.")

def unzip_precomputed_files():
    assert os.path.exists(precomputed_files_zip), f"File {precomputed_files_zip} does not exist."

    print("Unzipping precomputed files...")
    unzip(precomputed_files_zip, precomputed_files_dir)
    print("Precomputed files unzipped.")


def load_precomputed_files(run_tests: Optional[list[str]] = None, download_automatically: bool = False):
    """
    Load precomputed files from the environment variable ``PRECOMPUTED_FILES``.
    
    Makes the necessary checks in order to download the precomputed files if they are not available.
    Also validates them.
    
    Updates the value of the global variable ``precomputed_files_dir``.
    """
    global precomputed_files_dir # use the global variable
    
    try:
        precomputed_files_dir = os.environ['PRECOMPUTED_FILES']
    except KeyError:
        pass
    precomputed_files_dir = os.path.realpath(precomputed_files_dir)
    print(f"Precomputed files location: {precomputed_files_dir}")

    # If only the precomputed_files.zip file is in the directory, remove it, and ask to download again
    precomputed_files_listdir = os.listdir(precomputed_files_dir) if os.path.exists(precomputed_files_dir) else None
    if precomputed_files_listdir is not None and \
        (len(precomputed_files_listdir) <= 0 or (len(precomputed_files_listdir) == 1 and precomputed_files_listdir[0].endswith('precomputed_files.zip'))):
        print("Got an old process cut in the middle, removing history...")
        shutil.rmtree(precomputed_files_dir)

    should_download_precomputed_files = not os.path.exists(precomputed_files_dir)
    
    # Make sure that all the precomputed files are available
    for file_name in (run_tests if run_tests is not None else precomputed_files_list):
        test_dir = os.path.join(precomputed_files_dir, file_name)
        if not os.path.exists(test_dir):
            print("Precomputed test does not exist:", test_dir)
            should_download_precomputed_files = True
            break
        mean_file = os.path.join(test_dir, 'results-mean.npz')
        if not os.path.exists(mean_file):
            print("Mean file does not exist:", mean_file)
            should_download_precomputed_files = True
            break
        median_file = os.path.join(test_dir, 'results-median.npz')
        if not os.path.exists(median_file):
            print("Median file does not exist:", median_file)
            should_download_precomputed_files = True
            break

    if should_download_precomputed_files and os.path.exists(precomputed_files_dir):
        shutil.rmtree(precomputed_files_dir)

    if not os.path.exists(precomputed_files_dir):
        print()
        print("One or more precomputed files are missing (the precomputed files dir doesn't exist).")
        print("Do you want to download the precomputed files now?")
        print("Note: You must be in the UPV or be connected to the VPN to download the files.")
        print("yes/no: ", end="")
        answer = 'yes' if download_automatically else input()
        if answer.lower() == 'yes':
            # Create the directory if it does not exist
            if not os.path.exists(precomputed_files_dir):
                os.makedirs(precomputed_files_dir)
            
            # Remove the zip if it exists
            if os.path.exists(precomputed_files_zip):
                os.remove(precomputed_files_zip)

            download_precomputed_files()
            unzip_precomputed_files()
            os.remove(precomputed_files_zip)
        else:
            print("Exiting")
            sys.exit(1)
