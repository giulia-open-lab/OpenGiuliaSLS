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
import sys

# Configure Giulia into path
this_dir = os.path.dirname(__file__)
root_dir = os.path.join(this_dir, '..')
sys.path.insert(1, root_dir)

from giulia.fs import shadowing_dir

from tests.const import shadowing_files_url, shadowing_files_list
from tests.fs import download, unzip

shadowing_files_dir: str = shadowing_dir()
shadowing_files_zip: str = os.path.join(shadowing_files_dir, 'shadowing.zip')


def download_shadowing_files():
    download(shadowing_files_url, shadowing_files_zip, "Downloading files download")
    print(f"Shadowing files downloaded.")


def unzip_shadowing_files():
    assert os.path.exists(shadowing_files_zip), f"File {shadowing_files_zip} does not exist."

    print(f"Unzipping shadowing files...")
    unzip(shadowing_files_zip, shadowing_files_dir)
    print(f"Shadowing files unzipped.")


def load_shadowing_files():
    """
    Makes the necessary checks in order to download the shadowing files if they are not available.
    Also validates them.

    Updates the value of the global variable ``shadowing_files_dir``.
    """
    global shadowing_files_dir # use the global variable
    
    print(f"Shadowing files location: {shadowing_files_dir}")

    # If only the shadowing.zip file is in the directory, remove it, and ask to download again
    if os.path.exists(shadowing_files_dir) and \
        len(os.listdir(shadowing_files_dir)) == 1 and os.listdir(shadowing_files_dir)[0].endswith('shadowing.zip'):
        os.rmdir(shadowing_files_dir)

    should_download_shadowing_files = not os.path.exists(shadowing_files_dir)
    
    if not should_download_shadowing_files:
        for file_name in shadowing_files_list:
            file = os.path.join(shadowing_files_dir, file_name)
            if not os.path.exists(file):
                print(f"File {file} not found in the shadowing directory.")
                should_download_shadowing_files = True
                break

    if should_download_shadowing_files:
        print()
        print("Some or all shadowing files are missing.")
        print("Do you want to download the shadowing files now?")
        print("Any existing shadowing files in the directory will be overwritten.")
        print("Note: You must be in the UPV or be connected to the VPN to download the files.")
        print("yes/no: ", end="")
        answer = input()
        if answer.lower() == 'yes':
            # Create the directory if it does not exist
            if not os.path.exists(shadowing_files_dir):
                os.makedirs(shadowing_files_dir)
            
            # Remove the existing shadowing files if they exist
            for file_name in shadowing_files_list:
                file_path = os.path.join(shadowing_files_dir, file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Remove the zip if it exists
            if os.path.exists(shadowing_files_zip):
                os.remove(shadowing_files_zip)

            download_shadowing_files()
            unzip_shadowing_files()
            os.remove(shadowing_files_zip)
        else:
            print("Exiting")
            sys.exit(1)
