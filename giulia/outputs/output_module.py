"""
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
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.lib.npyio import NpzFile
from tqdm import tqdm

# Configure Giulia into path
this_dir = os.path.dirname(__file__)
root_dir = os.path.join(this_dir, "..", "..")
sys.path.insert(1, root_dir)

from giulia.fs import results_dir, data_file
from giulia.logger import warning, debug


def _convert_npz_to_parquet(npz: NpzFile, dataset_dir_path: str, npy_filter: Callable[[str], bool] = lambda x: True) -> List[str]:
    """
    Converts a NPZ file into a Parquet file.

    Args:
        npz: NPZ file to convert
        dataset_dir_path: dataset directory path
        npy_filter: filter function to apply to NPZ file. Will pass the npy file name as an argument. Defaults to accept all.

    Returns:
        A list of the NPY files used.
    """
    # We need to convert the data from numpy to parquet. We will hold the converted data here.
    # We will have multiple entries, grouped by episodes
    data_for_parquet: dict[int, dict[str, Any]] = {}
    # Initialize the schema array with just the episode index for now
    parquet_schema = [pa.field('episode_idx', pa.uint64())]

    # A list with all the names of the NPY files processed.
    npy_list: List[str] = []

    # Iterate each of the NPY files stored in the NPZ
    progress = tqdm(npz.files)
    for npy_file_name in progress:
        if not npy_filter(npy_file_name): continue

        npy_array = npz[npy_file_name]

        progress.set_description(npy_file_name)

        # Convert the data into a list of lists. The outermost list will hold all the different episodes
        npy_as_list = npy_array.tolist()
        for episode_index in range(len(npy_as_list)):
            npy_episode_data = npy_as_list[episode_index]
            # All data is exported with an extra column, just get rid of it by selecting the first element
            npy_episode_data = npy_episode_data[0]
            # Append the observation to the data_for_parquet array
            existing_data: dict[str, Any] = data_for_parquet[episode_index] if episode_index in data_for_parquet else {}
            existing_data[npy_file_name] = npy_episode_data
            data_for_parquet[episode_index] = existing_data

        # Fetch the dimensions of the data
        npy_array_shape: tuple[int, ...] = npy_array.shape
        npy_array_slen: int = len(npy_array_shape)

        # Let's determine the data type of the observation based on the dimensions of the array
        data_type: pa.DataType
        if npy_array_slen == 2:
            data_type = pa.float64()  # Single-value
        elif npy_array_slen == 3:
            data_type = pa.list_(pa.float64())  # Single-dimensional
        else:
            # Multidimensional array
            # We will use the length of the shape to know how many lists inside of lists are used.
            # First we subtract 2, the first dimension (number of episodes), and the second one (columns)
            depth = npy_array_slen - 2

            # Initialize the first iteration of the data type, and then loop until the desired range.
            # Note that be take 1 from depth since we have already added a list when initializing data_type
            data_type = pa.list_(pa.float64())
            for i in range(depth - 1):
                data_type = pa.list_(data_type)

        # Append the data type into the schema
        parquet_schema.append(pa.field(npy_file_name, data_type))

        # Append the npy file name into the result array.
        npy_list.append(npy_file_name)

    # Now we need to flatten the data_for_parquet in order to export it
    df_data = []
    for episode_index in data_for_parquet:
        episode_data: dict[str, Any] = {'episode_idx': episode_index}
        episode_data.update(data_for_parquet[episode_index])
        df_data.append(episode_data)

    # Convert the flattened data into a Pandas DataFrame
    df = pd.DataFrame(df_data)

    # Initialize the Parquet table from Pandas, and the schema defined
    table = pa.Table.from_pandas(df, schema=pa.schema(parquet_schema), preserve_index=False)

    # Write the Arrow Table to a Parquet file
    pq.write_to_dataset(
        table,
        root_path=dataset_dir_path,
        partition_cols=['episode_idx'],  # Partitioning by episode_idx for efficient retrieval per episode
        compression='snappy',  # A good balance of speed and compression
    )

    return npy_list


class OutputInfo:
    name: str
    description: Optional[str]
    unit: Optional[str]
    shape: Optional[str]

    def __init__(self, name: str, description: Optional[str] = None, unit: Optional[str] = None, shape: Optional[str] = None):
        self.name = name
        self.description = description
        self.unit = unit
        self.shape = shape


    def to_csv(self, separator: str = ';') -> str:
        return self.name + separator + (self.description or '') + separator + (self.unit or '')


class OutputInfoList:
    memory: List[OutputInfo]

    def __init__(self, memory: Optional[List[OutputInfo]] = None):
        self.memory = memory if memory is not None else []


    def append(self, info: OutputInfo):
        self.memory.append(info)


    def filter(self, names: List[str]) -> 'OutputInfoList':
        """
        Returns a list of OutputInfo objects that have a name contained in ``names``.
        Args:
            names: A list of names to search for.
        Returns:
            A list of ``OutputInfo`` whose ``name`` is in ``names``.
        """
        return OutputInfoList(list(filter(lambda info: info.name in names, self.memory)))


    def save(self, path: str, separator: str = ';'):
        """
        Saves the data in this list in the desired path.
        Args:
            path: The path to save the data to. Its parent directory must exist.
            separator: The separator between all the fields. For CSV usually `,` or `;`. Default: `;`.
        """
        # If the file already exists, delete it
        if os.path.exists(path):
            os.remove(path)
        # Open the file in append mode, and add the headers and data
        with open(path, 'a') as f:
            f.write('name' + separator + 'description' + separator + 'unit' + separator + 'shape' + separator + '\n')
            for info in self.memory:
                f.write(info.to_csv(separator=separator) + '\n')


class OutputModule:
    def __init__(self, output_dir: str, compression: str, rm_uncompressed: bool):
        self.output_dir = output_dir
        self.compression = compression
        self.rm_uncompressed = rm_uncompressed


    @staticmethod
    def __load_output_info() -> OutputInfoList:
        with open(data_file('output-info.csv'), "r") as f:
            raw_output_info = f.readlines()
        # Remove first line, it's the columns header
        raw_output_info.pop(0)
        result = OutputInfoList()
        for line in raw_output_info:
            pieces = line.split(",")
            result.append(OutputInfo(pieces[0], pieces[1], pieces[2], pieces[3]))
        return result


    def __compress(self, directory: str):
        if self.compression == 'none':
            warning('Compression method was set to "none". Output will be placed in a directory.')
        elif self.compression == 'zip':
            name = os.path.basename(directory)
            outputs_dir = os.path.abspath(os.path.join(directory, os.pardir))
            zip_file = os.path.join(outputs_dir, name)
            debug(f"Compressing {name} into {zip_file}.zip ...")
            shutil.make_archive(zip_file, 'zip', directory)
            debug("Compression complete.")

            if self.rm_uncompressed:
                debug(f"Removing {directory}...")
                shutil.rmtree(directory)
        else:
            warning(f'The compression method given is unknown ({self.compression}). Output will be placed in a directory.')


    def convert(self, filter_predicate: Callable[[str], bool], npy_filter: Callable[[str], bool] = lambda x: True):
        results_directory = results_dir()
        results = next(os.walk(results_directory))[1]

        base_output_info = OutputModule.__load_output_info()

        for dir_name in results:
            print(f"\n{dir_name}")
            directory = os.path.join(results_directory, dir_name)
            npz_files = os.listdir(directory)
            for file_name in npz_files:
                if not filter_predicate(file_name): continue
                print(f"-> {file_name}")
                file_path = os.path.join(directory, file_name)

                # Declare where the output parquet file will be stored
                dataset_dir_name = os.path.join(self.output_dir, dir_name)

                # Load the NPZ file. It's consisted of multiple npy files with all the outputs
                # See Episode_Performance$_save_episodes_performance for all the files of results-raw.npz
                # Each NPY file is a numpy array with an unknown shape.
                with np.load(file_path) as npz:
                    npy_files = _convert_npz_to_parquet(npz, dataset_dir_name, npy_filter)

                    # Now we take which NPY files have been processed, and export a custom output-info.csv file for
                    # the specific output set.
                    output_info = base_output_info.filter(npy_files)
                    output_info_file = os.path.join(dataset_dir_name, 'output-info.csv')
                    output_info.save(output_info_file)

                    self.__compress(dataset_dir_name)

# Only run automatically when running from a terminal
if __name__ == "__main__":
    output = os.path.join(root_dir, 'outputs', 'outputs')
    module = OutputModule(os.path.realpath(output), 'zip', True)
    module.convert(
        # Include npz files that are not intended for plotting, and are not already computed (mean, median)
        filter_predicate=lambda file_name : file_name.endswith('.npz') and not file_name.startswith('to_plot') and not file_name.endswith('-mean.npz') and not file_name.endswith('-median.npz')
    )
