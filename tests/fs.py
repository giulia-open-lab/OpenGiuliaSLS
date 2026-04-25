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
import urllib.request
import zipfile

from tqdm import tqdm

class _ProgressBar(tqdm):
    def update_to(self, current, total):
        self.total = total
        self.update(current - self.n)

def download(url: str, file: str, desc: str):
    """
    Downloads a file from a URL to a local file.
    
    Args:
        url (str): The URL of the file to download.
        file (str): The local file where to save the downloaded file.
        desc (str): The description to show in the progress
    """
    with _ProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(
            url,file,lambda count, blockSize, totalSize: t.update_to(count * blockSize, totalSize)
        )

def unzip(file: str, dir: str):
    """
    Unzips a file into a directory.

    Args:
        file (str): The file to unzip.
        dir (str): The directory where to unzip the file.
    """
    parent_dir = os.path.dirname(dir)  # <-- cartella padre
    with zipfile.ZipFile(file, "r") as zip_ref:
        for member in tqdm(zip_ref.infolist(), desc='Extracting'):
            try:
                zip_ref.extract(member, parent_dir)
            except zipfile.error:
                pass
            