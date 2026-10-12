import os

import numpy as np
import pandas as pd
import numpy.typing as npt
import pyarrow.dataset as ds
from pyarrow import Table

from giulia.logger import debug
# from giulia.kpis.calculate_rate import Rate
from giulia.outputs.processing import use_output

def load_gof(gof_file: str) -> ds.Dataset:
    """
    Loads a GOF file as a PyArrow Dataset.

    Args:
        gof_file (str): The path where the GOF file is located at.

    Raises:
        FileNotFoundError: If the GOF file given doesn't exist.

    Returns:
        ds.Dataset: The loaded PyArrow dataset.
    """
    output_directory = gof_file.replace(".gof", "")
    
    if not os.path.exists(gof_file):
        if os.path.exists(output_directory):
            # If the directory already exists, load from there
            debug('Loading dataset from existing directory:', output_directory)
            return load_dataset(output_directory)
        else:
            raise FileNotFoundError(f"The specified gof file does not exist: {gof_file}")
    
    dataset_dir = use_output(gof_file, False)
    
    return load_dataset(dataset_dir)

def load_gof_as_dict(gof_file: str) -> dict:
    dataset = load_gof(gof_file)
    table: Table = dataset.to_table()
    df = table.to_pandas()
    return df.to_dict(orient='list')  # Convert DataFrame to dict with lists as values

def load_dataset(dataset_path: str) -> ds.Dataset:
    """
    Loads a Parquet dataset from the specified path.

    Args:
        dataset_path (str): The path where the Parquet dataset is located at.

    Raises:
        FileNotFoundError: If the dataset path given doesn't exist.

    Returns:
        ds.Dataset: The loaded PyArrow dataset.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The specified dataset path does not exist: {dataset_path}")
    
    return ds.dataset(dataset_path, format="parquet", exclude_invalid_files=True)



def _reshape_ue_throughput_per_carrier_std_to_mfl_format(ue_throughput_obj, ue_throughput_per_carrier_Mbps: np.ndarray) -> np.ndarray:

    # Get each UE serving cell
    # Note that this is the same serving cell used for thp computation
    serving_cell_array = \
        getattr(ue_throughput_obj.best_serving_cell_per_ue_obj, "best_serving_cell_ID_per_ue", None)
    assert isinstance(serving_cell_array, np.ndarray), "Error, serving_cell_array is not a numpy array"
        
    # Get network dataframe eng parameters
    df_ep = getattr(ue_throughput_obj.network_deployment_obj, "df_ep", None)
    assert isinstance(df_ep, pd.DataFrame), "Error, df_ep is not a DataFrame"

    # Get network deployed frequencies
    network_freq_deployed = np.unique(df_ep['dl_carrier_frequency_GHz'])
    
    # Create mapping cell_ID -> freq
    cell2freq = dict(zip(df_ep['ID'], df_ep['dl_carrier_frequency_GHz']))

    # Create mapping freq -> index of freq_available
    freq2idx = {f: i for i, f in enumerate(network_freq_deployed)}

    # Init auxiliary array for output
    ue_throughput_per_carrier_Mbps_aux = np.full_like(ue_throughput_per_carrier_Mbps, np.nan)
    ue_throughput_per_carrier_Mbps_aux = np.tile(ue_throughput_per_carrier_Mbps_aux,
                                                (len(network_freq_deployed), 1))
    
    # Populate the new auxiliary output
    
    throughputs_1d = np.atleast_1d(ue_throughput_per_carrier_Mbps).ravel()
    serving_cells_1d = np.atleast_1d(serving_cell_array).ravel()
    if throughputs_1d.size != serving_cells_1d.size:
        raise ValueError(
            f"Length mismatch: throughputs {throughputs_1d.size} vs serving_cells {serving_cells_1d.size}"
        )
    
    # Populate the new auxiliary output
    for ue_idx, (throughput_mbps, serving_cell) in enumerate(zip(throughputs_1d, serving_cells_1d)):
        ue_throughput_per_carrier_Mbps_aux[freq2idx[cell2freq[serving_cell]], ue_idx] = throughput_mbps
    
    # Return reshaped array
    return ue_throughput_per_carrier_Mbps_aux.astype(np.float64)
