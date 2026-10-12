"""Consistency checks for Giulia's engineering-parameter table (``df_ep``).

Several ``df_ep`` columns are *derived* from others: the scenario builder computes them
once, consistently, and nothing recomputes them afterwards. A write that changes a source
column therefore leaves its dependants stale, and Giulia does not notice — see
``docs/dedalo/df-ep-consistency.md`` for a measured case where that silently doubled a
reported KPI.

Each ``_check_*`` recomputes one derived column from its sources and returns
``(passed, message)``. The matching repairs are in ``tests/network_df_ep_update.py``.
``Dedalo.check_df_ep`` / ``Dedalo.assert_df_ep_consistent`` call into here.
"""

import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd

# Configure Giulia into path
this_dir = os.path.dirname(__file__)
root_dir = os.path.join(this_dir, '..')
sys.path.insert(1, root_dir)

from giulia.rrc import tools_carrier


### dl_carrier_wavelength_m
###########################
def _check_dl_carrier_wavelength_m(df_ep: pd.DataFrame) -> tuple[bool, str]:
    attr_name: str = "dl_carrier_wavelength_m"
    dependent_attribute: str = "dl_carrier_frequency_GHz"

    # Compute expected wavelength from carrier frequency
    dl_carrier_frequency_GHz = np.asarray(df_ep[dependent_attribute].values, dtype=float)
    expected_dl_carrier_wavelength_m = 3e8 / (dl_carrier_frequency_GHz * 1e9)

    # Get current value
    dl_carrier_wavelength_m = np.asarray(df_ep[attr_name].values, dtype=float)

    if np.allclose(dl_carrier_wavelength_m, expected_dl_carrier_wavelength_m, rtol=1e-5):
        return True, "check passed"

    return False, (
        f"Invalid value for {attr_name}. "
        f"\nGot \n{dl_carrier_wavelength_m}, \nexpected \n{expected_dl_carrier_wavelength_m} "
        f"based on {dependent_attribute} from DataFrame Engineer Parameters df_ep."
    )

### ul_carrier_wavelength_m
###########################
def _check_ul_carrier_wavelength_m(df_ep: pd.DataFrame) -> tuple[bool, str]:
    attr_name: str = "ul_carrier_wavelength_m"
    dependent_attribute: str = "ul_carrier_frequency_GHz"

    # Compute expected wavelength from carrier frequency
    ul_carrier_frequency_GHz = np.asarray(df_ep[dependent_attribute].values, dtype=float)
    expected_ul_carrier_wavelength_m = 3e8 / (ul_carrier_frequency_GHz * 1e9)

    # Get current value
    ul_carrier_wavelength_m = np.asarray(df_ep[attr_name].values, dtype=float)

    if np.allclose(ul_carrier_wavelength_m, expected_ul_carrier_wavelength_m, rtol=1e-5):
        return True, "check passed"

    return False, (
        f"Invalid value for {attr_name}. "
        f"\nGot \n{ul_carrier_wavelength_m}, \nexpected \n{expected_ul_carrier_wavelength_m} "
        f"based on {dependent_attribute} from DataFrame Engineer Parameters df_ep."
    )

### dl_PRBs_available
#####################
def _check_dl_PRBs_available(df_ep: pd.DataFrame) -> tuple[bool, str]:
    attr_name: str = "dl_PRBs_available"
    dependent_attributes: dict = {"dl_bandwidth_MHz": "dl_bandwidth_MHz", "dl_subcarrier_spacing_kHz": "dl_subcarrier_spacing_kHz"}

    # Compute expected PRBs from bandwidth and subcarrier spacing
    #   Get dependent attributes
    dl_bandwidth_MHz = np.asarray(df_ep[dependent_attributes["dl_bandwidth_MHz"]].values, dtype=float)
    dl_subcarrier_spacing_kHz = np.asarray(df_ep[dependent_attributes["dl_subcarrier_spacing_kHz"]].values, dtype=float)

    # Get current value
    dl_PRBs_available = np.asarray(df_ep[attr_name].values, dtype=float)

    #   Compute expected PRBs
    try:
        expected_dl_PRBs_available = [tools_carrier.bandwidth_to_PRBs(bw, scs) for bw, scs in zip(dl_bandwidth_MHz, dl_subcarrier_spacing_kHz)]
        expected_dl_PRBs_available = np.array(expected_dl_PRBs_available, dtype=int)
    except Exception as e:
        return False, (
            f"Invalid value for {attr_name}. "
            f"\nGot \n{dl_PRBs_available}, \nexpected could not be computed: "
            f"tools_carrier.bandwidth_to_PRBs raised {type(e).__name__}: {e}. "
            f"\nBased on {dependent_attributes} from DataFrame Engineer Parameters df_ep."
        )

    if np.allclose(dl_PRBs_available, expected_dl_PRBs_available, rtol=1e-5):
        return True, "check passed"

    return False, (
        f"Invalid value for {attr_name}. "
        f"\nGot \n{dl_PRBs_available}, \nexpected \n{expected_dl_PRBs_available} "
        f"based on {dependent_attributes} from DataFrame Engineer Parameters df_ep."
    )

### dl_ofdm_symbol_duration_us
##############################
def _check_dl_ofdm_symbol_duration_us(df_ep: pd.DataFrame) -> tuple[bool, str]:
    attr_name: str = "dl_ofdm_symbol_duration_us"
    dependent_attribute: str = "dl_subcarrier_spacing_kHz"
    # Compute expected OFDM symbol duration from subcarrier spacing
    dl_subcarrier_spacing_kHz = np.asarray(df_ep[dependent_attribute].values, dtype=float)

    # Get current value
    dl_ofdm_symbol_duration_us = np.asarray(df_ep[attr_name].values, dtype=float)

    try:
        expected_dl_ofdm_symbol_duration_us = np.array([tools_carrier.ofdm_symbol_duration(scs) for scs in dl_subcarrier_spacing_kHz], dtype=float)
    except Exception as e:
        return False, (
            f"Invalid value for {attr_name}. "
            f"\nGot \n{dl_ofdm_symbol_duration_us}, \nexpected could not be computed: "
            f"tools_carrier.ofdm_symbol_duration raised {type(e).__name__}: {e}. "
            f"Based on {dependent_attribute} from DataFrame Engineer Parameters df_ep."
        )

    if np.allclose(dl_ofdm_symbol_duration_us, expected_dl_ofdm_symbol_duration_us, rtol=1e-5):
        return True, "check passed"

    return False, (
        f"Invalid value for {attr_name}. "
        f"\nGot \n{dl_ofdm_symbol_duration_us}, \nexpected \n{expected_dl_ofdm_symbol_duration_us} "
        f"based on {dependent_attribute} from DataFrame Engineer Parameters df_ep."
    )

### dl_control_channel_overhead
###############################
def _check_dl_control_channel_overhead(df_ep: pd.DataFrame) -> tuple[bool, str]:
    attr_name: str = "dl_control_channel_overhead"
    dependent_attribute: str = "dl_subcarrier_spacing_kHz"
    # Compute expected control channel overhead from subcarrier spacing
    dl_subcarrier_spacing_kHz = np.asarray(df_ep[dependent_attribute].values, dtype=float)

    # Get current value
    dl_control_channel_overhead = np.asarray(df_ep[attr_name].values, dtype=float)

    try:
        expected_dl_control_channel_overhead = np.array([tools_carrier.control_channel_overhead(scs) for scs in dl_subcarrier_spacing_kHz], dtype=float)
    except Exception as e:
        return False, (
            f"Invalid value for {attr_name}. "
            f"\nGot \n{dl_control_channel_overhead}, \nexpected could not be computed: "
            f"tools_carrier.control_channel_overhead raised {type(e).__name__}: {e}. "
            f"Based on {dependent_attribute} from DataFrame Engineer Parameters df_ep."
        )

    if np.allclose(dl_control_channel_overhead, expected_dl_control_channel_overhead, rtol=1e-5):
        return True, "check passed"

    return False, (
        f"Invalid value for {attr_name}. "
        f"\nGot \n{dl_control_channel_overhead}, \nexpected \n{expected_dl_control_channel_overhead} "
        f"based on {dependent_attribute} from DataFrame Engineer Parameters df_ep."
    )

### ul_available_PRBs
#####################
def _check_ul_available_PRBs(df_ep: pd.DataFrame) -> tuple[bool, str]:
    attr_name: str = "ul_available_PRBs"
    dependent_attributes: dict = {"ul_bandwidth_MHz": "ul_bandwidth_MHz", "ul_subcarrier_spacing_kHz": "ul_subcarrier_spacing_kHz"}

    # Compute expected PRBs from bandwidth and subcarrier spacing
    #   Get dependent attributes
    ul_bandwidth_MHz = np.asarray(df_ep[dependent_attributes["ul_bandwidth_MHz"]].values, dtype=float)
    ul_subcarrier_spacing_kHz = np.asarray(df_ep[dependent_attributes["ul_subcarrier_spacing_kHz"]].values, dtype=float)

    # Get current value
    ul_available_PRBs = np.asarray(df_ep[attr_name].values, dtype=float)

    #   Compute expected PRBs
    try:
        expected_ul_available_PRBs = [tools_carrier.bandwidth_to_PRBs(bw, scs) for bw, scs in zip(ul_bandwidth_MHz, ul_subcarrier_spacing_kHz)]
        expected_ul_available_PRBs = np.array(expected_ul_available_PRBs, dtype=int)
    except Exception as e:
        return False, (
            f"Invalid value for {attr_name}. "
            f"\nGot \n{ul_available_PRBs}, \nexpected could not be computed: "
            f"tools_carrier.bandwidth_to_PRBs raised {type(e).__name__}: {e}. "
            f"\nBased on {dependent_attributes} from DataFrame Engineer Parameters df_ep."
        )

    if np.allclose(ul_available_PRBs, expected_ul_available_PRBs, rtol=1e-5):
        return True, "check passed"

    return False, (
        f"Invalid value for {attr_name}. "
        f"\nGot \n{ul_available_PRBs}, \nexpected \n{expected_ul_available_PRBs} "
        f"based on {dependent_attributes} from DataFrame Engineer Parameters df_ep."
    )

### ul_ofdm_symbol_duration_us
##############################
def _check_ul_ofdm_symbol_duration_us(df_ep: pd.DataFrame) -> tuple[bool, str]:
    attr_name: str = "ul_ofdm_symbol_duration_us"
    dependent_attribute: str = "ul_subcarrier_spacing_kHz"
    # Compute expected OFDM symbol duration from subcarrier spacing
    ul_subcarrier_spacing_kHz = np.asarray(df_ep[dependent_attribute].values, dtype=float)

    # Get current value
    ul_ofdm_symbol_duration_us = np.asarray(df_ep[attr_name].values, dtype=float)

    try:
        expected_ul_ofdm_symbol_duration_us = np.array([tools_carrier.ofdm_symbol_duration(scs) for scs in ul_subcarrier_spacing_kHz], dtype=float)
    except Exception as e:
        return False, (
            f"Invalid value for {attr_name}. "
            f"\nGot \n{ul_ofdm_symbol_duration_us}, \nexpected could not be computed: "
            f"tools_carrier.ofdm_symbol_duration raised {type(e).__name__}: {e}. "
            f"Based on {dependent_attribute} from DataFrame Engineer Parameters df_ep."
        )

    if np.allclose(ul_ofdm_symbol_duration_us, expected_ul_ofdm_symbol_duration_us, rtol=1e-5):
        return True, "check passed"

    return False, (
        f"Invalid value for {attr_name}. "
        f"\nGot \n{ul_ofdm_symbol_duration_us}, \nexpected \n{expected_ul_ofdm_symbol_duration_us} "
        f"based on {dependent_attribute} from DataFrame Engineer Parameters df_ep."
    )

### ul_control_channel_overhead
###############################
def _check_ul_control_channel_overhead(df_ep: pd.DataFrame) -> tuple[bool, str]:
    attr_name: str = "ul_control_channel_overhead"
    dependent_attribute: str = "ul_subcarrier_spacing_kHz"
    # Compute expected control channel overhead from subcarrier spacing
    ul_subcarrier_spacing_kHz = np.asarray(df_ep[dependent_attribute].values, dtype=float)

    # Get current value
    ul_control_channel_overhead = np.asarray(df_ep[attr_name].values, dtype=float)

    try:
        expected_ul_control_channel_overhead = np.array([tools_carrier.control_channel_overhead(scs) for scs in ul_subcarrier_spacing_kHz], dtype=float)
    except Exception as e:
        return False, (
            f"Invalid value for {attr_name}. "
            f"\nGot \n{ul_control_channel_overhead}, \nexpected could not be computed: "
            f"tools_carrier.control_channel_overhead raised {type(e).__name__}: {e}. "
            f"Based on {dependent_attribute} from DataFrame Engineer Parameters df_ep."
        )

    if np.allclose(ul_control_channel_overhead, expected_ul_control_channel_overhead, rtol=1e-5):
        return True, "check passed"

    return False, (
        f"Invalid value for {attr_name}. "
        f"\nGot \n{ul_control_channel_overhead}, \nexpected \n{expected_ul_control_channel_overhead} "
        f"based on {dependent_attribute} from DataFrame Engineer Parameters df_ep."
    )

#### antenna_config_number_of_elements
######################################
def _check_antenna_config_number_of_elements(df_ep: pd.DataFrame) -> tuple[bool, str]:
    attr_name: str = "antenna_config_number_of_elements"
    dependent_attributes: dict = {
        "antenna_config_Mg": "antenna_config_Mg",
        "antenna_config_Ng": "antenna_config_Ng",
        "antenna_config_M":  "antenna_config_M",
        "antenna_config_N":  "antenna_config_N",
        "antenna_config_P":  "antenna_config_P",
    }
    # Compute expected number of elements from antenna configuration parameters
    Mg = np.asarray(df_ep[dependent_attributes["antenna_config_Mg"]].values, dtype=int)
    Ng = np.asarray(df_ep[dependent_attributes["antenna_config_Ng"]].values, dtype=int)
    M = np.asarray(df_ep[dependent_attributes["antenna_config_M"]].values, dtype=int)
    N = np.asarray(df_ep[dependent_attributes["antenna_config_N"]].values, dtype=int)
    P = df_ep[dependent_attributes["antenna_config_P"]].map({"single": 1, "dual": 2}).astype(int).values

    expected_antenna_config_number_of_elements = Mg * Ng * M * N * P

    # Get current value
    antenna_config_number_of_elements = np.asarray(df_ep[attr_name].values, dtype=int)

    if np.all(antenna_config_number_of_elements == expected_antenna_config_number_of_elements):
        return True, "check passed"

    return False, (
        f"Invalid value for {attr_name}. "
        f"\nGot \n{antenna_config_number_of_elements}, \nexpected \n{expected_antenna_config_number_of_elements} "
        f"based on {dependent_attributes} from DataFrame Engineer Parameters df_ep."
    )


#### SSB_number_of_beams
##########################
def _check_SSB_number_of_beams(df_ep: pd.DataFrame) -> tuple[bool, str]:
    attr_name: str = "SSB_number_of_beams"
    dependent_attributes: dict = {"SSB_number_of_beams_H": "SSB_number_of_beams_H", "SSB_number_of_beams_V": "SSB_number_of_beams_V"}
    # Load precoder type and beam number inputs from df_ep
    SSB_precoder = np.asarray(df_ep["SSB_precoder"].values, dtype=str)
    SSB_number_of_beams_H = np.asarray(df_ep[dependent_attributes["SSB_number_of_beams_H"]].values, dtype=int)
    SSB_number_of_beams_V = np.asarray(df_ep[dependent_attributes["SSB_number_of_beams_V"]].values, dtype=int)
    SSB_number_of_beams = np.asarray(df_ep[attr_name].values, dtype=int)

    # Create valid precoder list
    _VALID_SSB_PRECODERS = ["DFT", "custom", "3GPPTR38_901_P1_single_column", "3GPPTR38_901_P2_C1_one_port_per_panel", "custom_vMB"]
    # If any rows have an unrecognised precoder, return error with details
    unknown_mask = ~np.isin(SSB_precoder, _VALID_SSB_PRECODERS)
    if np.any(unknown_mask):
        return False, (
            f"Invalid value for SSB_precoder. Found {np.unique(SSB_precoder[unknown_mask])}. "
            f"Expected one of {_VALID_SSB_PRECODERS} "
            f"based on SSB_precoder from DataFrame Engineer Parameters df_ep."
        )

    # Initialize expected beam number array; each precoder group fills its slice below
    expected_SSB_number_of_beams = np.empty(len(df_ep), dtype=int)

    # DFT / custom: expected = H * V
    mask_dft = (SSB_precoder == "DFT") | (SSB_precoder == "custom")
    expected_SSB_number_of_beams[mask_dft] = SSB_number_of_beams_H[mask_dft] * SSB_number_of_beams_V[mask_dft]

    # 3GPPTR38_901_P1_single_column: single beam, H=1 and V=1 required
    mask_p1 = SSB_precoder == "3GPPTR38_901_P1_single_column"
    if np.any(mask_p1):
        if not np.all(SSB_number_of_beams_H[mask_p1] == 1):
            return False, "For 3GPPTR38_901_P1_single_column precoder, SSB_number_of_beams_H must be 1."
        if not np.all(SSB_number_of_beams_V[mask_p1] == 1):
            return False, "For 3GPPTR38_901_P1_single_column precoder, SSB_number_of_beams_V must be 1."
        expected_SSB_number_of_beams[mask_p1] = 1

    # 3GPPTR38_901_P2_C1_one_port_per_panel: two beams, H=1 and V=2 required
    mask_p2 = SSB_precoder == "3GPPTR38_901_P2_C1_one_port_per_panel"
    if np.any(mask_p2):
        if not np.all(SSB_number_of_beams_H[mask_p2] == 1):
            return False, "For 3GPPTR38_901_P2_C1_one_port_per_panel precoder, SSB_number_of_beams_H must be 1."
        if not np.all(SSB_number_of_beams_V[mask_p2] == 2):
            return False, "For 3GPPTR38_901_P2_C1_one_port_per_panel precoder, SSB_number_of_beams_V must be 2."
        expected_SSB_number_of_beams[mask_p2] = 2

    # custom_vMB: beam number does not depends on H and V, but should match the directions, as checked below
    mask_vmb = SSB_precoder == "custom_vMB"
    expected_SSB_number_of_beams[mask_vmb] = SSB_number_of_beams[mask_vmb]

    # Verify beam number matches expected for all rows: create first flag
    flag_out_SSB_number_of_beams = np.all(SSB_number_of_beams == expected_SSB_number_of_beams)

    #### Control beam direction for custom_vMB rows only
    flag_out_SSB_beam_direction = True
    if np.any(mask_vmb):
        # Filter to custom_vMB rows and normalise direction lists to tuples
        df_ep_vmb = df_ep[mask_vmb].copy()
        cols_direction = ['SSB_BeamDirection_phi_deg', 'SSB_BeamDirection_theta_deg']
        for c in cols_direction:
            df_ep_vmb[c] = df_ep_vmb[c].apply(lambda x: tuple(x) if isinstance(x, list) else x)

        # Count beam directions in each cell's list, per row
        number_directions_H = df_ep_vmb['SSB_BeamDirection_phi_deg'].apply(len).values
        number_directions_V = df_ep_vmb['SSB_BeamDirection_theta_deg'].apply(len).values

        # Beam count must equal the number of directions in both H and V, for every custom_vMB row
        flag_out_SSB_beam_direction = np.all(SSB_number_of_beams[mask_vmb] == number_directions_H) and np.all(number_directions_H == number_directions_V)

    if flag_out_SSB_number_of_beams and flag_out_SSB_beam_direction:
        return True, "check passed"

    return False, (
        f"Invalid value for {attr_name}. "
        f"\nGot \n{SSB_number_of_beams}, \nexpected \n{expected_SSB_number_of_beams} "
        f"based on {dependent_attributes} from DataFrame Engineer Parameters df_ep. "
        f"Additionally, if SSB_precoder is 'custom_vMB', the number of beams must match the number of directions in both horizontal and vertical dimensions."
    )


#### CSI_RS_number_of_beams
##########################
def _check_CSI_RS_number_of_beams(df_ep: pd.DataFrame) -> tuple[bool, str]:
    attr_name: str = "CSI_RS_number_of_beams"
    dependent_attributes: dict = {"CSI_RS_number_of_beams_H": "CSI_RS_number_of_beams_H", "CSI_RS_number_of_beams_V": "CSI_RS_number_of_beams_V"}
    # Load precoder type and beam number inputs from df_ep
    CSI_RS_precoder = np.asarray(df_ep["CSI_RS_precoder"].values, dtype=str)
    CSI_RS_number_of_beams_H = np.asarray(df_ep[dependent_attributes["CSI_RS_number_of_beams_H"]].values, dtype=int)
    CSI_RS_number_of_beams_V = np.asarray(df_ep[dependent_attributes["CSI_RS_number_of_beams_V"]].values, dtype=int)
    CSI_RS_number_of_beams = np.asarray(df_ep[attr_name].values, dtype=int)

    # Create valid precoder list
    _VALID_CSI_RS_PRECODERS = ["DFT", "custom", "3GPPTR38_901_P1_single_column", "3GPPTR38_901_P2_C1_one_port_per_panel", "custom_vMB"]
    # If any rows have an unrecognised precoder, return error with details
    unknown_mask = ~np.isin(CSI_RS_precoder, _VALID_CSI_RS_PRECODERS)
    if np.any(unknown_mask):
        return False, (
            f"Invalid value for CSI_RS_precoder. Found {np.unique(CSI_RS_precoder[unknown_mask])}. "
            f"Expected one of {_VALID_CSI_RS_PRECODERS} "
            f"based on CSI_RS_precoder from DataFrame Engineer Parameters df_ep."
        )

    # Initialize expected beam number array; each precoder group fills its slice below
    expected_CSI_RS_number_of_beams = np.empty(len(df_ep), dtype=int)

    # DFT / custom: expected = H * V
    mask_dft = (CSI_RS_precoder == "DFT") | (CSI_RS_precoder == "custom")
    expected_CSI_RS_number_of_beams[mask_dft] = CSI_RS_number_of_beams_H[mask_dft] * CSI_RS_number_of_beams_V[mask_dft]

    # 3GPPTR38_901_P1_single_column: single beam, H=1 and V=1 required
    mask_p1 = CSI_RS_precoder == "3GPPTR38_901_P1_single_column"
    if np.any(mask_p1):
        if not np.all(CSI_RS_number_of_beams_H[mask_p1] == 1):
            return False, "For 3GPPTR38_901_P1_single_column precoder, CSI_RS_number_of_beams_H must be 1."
        if not np.all(CSI_RS_number_of_beams_V[mask_p1] == 1):
            return False, "For 3GPPTR38_901_P1_single_column precoder, CSI_RS_number_of_beams_V must be 1."
        expected_CSI_RS_number_of_beams[mask_p1] = 1

    # 3GPPTR38_901_P2_C1_one_port_per_panel: two beams, H=1 and V=2 required
    mask_p2 = CSI_RS_precoder == "3GPPTR38_901_P2_C1_one_port_per_panel"
    if np.any(mask_p2):
        if not np.all(CSI_RS_number_of_beams_H[mask_p2] == 1):
            return False, "For 3GPPTR38_901_P2_C1_one_port_per_panel precoder, CSI_RS_number_of_beams_H must be 1."
        if not np.all(CSI_RS_number_of_beams_V[mask_p2] == 2):
            return False, "For 3GPPTR38_901_P2_C1_one_port_per_panel precoder, CSI_RS_number_of_beams_V must be 2."
        expected_CSI_RS_number_of_beams[mask_p2] = 2

    # custom_vMB: beam number does not depends on H and V, but should match the directions, as checked below
    mask_vmb = CSI_RS_precoder == "custom_vMB"
    expected_CSI_RS_number_of_beams[mask_vmb] = CSI_RS_number_of_beams[mask_vmb]

    # Verify beam number matches expected for all rows: create first flag
    flag_out_CSI_RS_number_of_beams = np.all(CSI_RS_number_of_beams == expected_CSI_RS_number_of_beams)

    #### Control beam direction for custom_vMB rows only
    flag_out_CSI_RS_beam_direction = True
    if np.any(mask_vmb):
        # Filter to custom_vMB rows and normalise direction lists to tuples
        df_ep_vmb = df_ep[mask_vmb].copy()
        cols_direction = ['CSI_RS_BeamDirection_phi_deg', 'CSI_RS_BeamDirection_theta_deg']
        for c in cols_direction:
            df_ep_vmb[c] = df_ep_vmb[c].apply(lambda x: tuple(x) if isinstance(x, list) else x)

        # Count beam directions in each cell's list, per row
        number_directions_H = df_ep_vmb['CSI_RS_BeamDirection_phi_deg'].apply(len).values
        number_directions_V = df_ep_vmb['CSI_RS_BeamDirection_theta_deg'].apply(len).values

        # Beam count must equal the number of directions in both H and V, for every custom_vMB row
        flag_out_CSI_RS_beam_direction = np.all(CSI_RS_number_of_beams[mask_vmb] == number_directions_H) and np.all(number_directions_H == number_directions_V)

    if flag_out_CSI_RS_number_of_beams and flag_out_CSI_RS_beam_direction:
        return True, "check passed"

    return False, (
        f"Invalid value for {attr_name}. "
        f"\nGot \n{CSI_RS_number_of_beams}, \nexpected \n{expected_CSI_RS_number_of_beams} "
        f"based on {dependent_attributes} from DataFrame Engineer Parameters df_ep. "
        f"Additionally, if CSI_RS_precoder is 'custom_vMB', the number of beams must match the number of directions in both horizontal and vertical dimensions."
    )


def run_all_checklist(df_ep: pd.DataFrame, verbose: bool = True) -> dict[str, tuple[bool, str]]:
    """Run every derived-field check and return ``{check_name: (passed, message)}``.

    ``verbose=True`` (the default, preserving the standalone script's output) prints one
    line per check. Library callers running this inside a loop should pass ``verbose=False``
    — :meth:`dedalo.dedalo.Dedalo.check_df_ep` does.
    """
    # Define a list of all check functions to run
    check_functions = [
        _check_dl_carrier_wavelength_m,
        _check_ul_carrier_wavelength_m,

        _check_dl_PRBs_available,
        _check_dl_ofdm_symbol_duration_us,
        _check_dl_control_channel_overhead,

        _check_ul_available_PRBs,
        _check_ul_ofdm_symbol_duration_us,
        _check_ul_control_channel_overhead,

        _check_antenna_config_number_of_elements,
        _check_SSB_number_of_beams,
        _check_CSI_RS_number_of_beams
    ]

    # Run every check and keep the raw (passed, msg) tuple keyed by check name
    results: dict[str, tuple[bool, str]] = {}
    for check_func in check_functions:
        passed, msg = check_func(df_ep)
        results[check_func.__name__] = (passed, msg)
        if verbose:
            print(f"{check_func.__name__}: {'PASS' if passed else 'FAIL - ' + msg}")

    return results


def print_checklist_results(
    results: dict[str, tuple[bool, str]],
    context: dict | None = None,
) -> None:
    GREEN = "\033[92m"
    RED   = "\033[91m"
    BOLD  = "\033[1m"
    DIM   = "\033[2m"
    RESET = "\033[0m"

    total    = len(results)
    n_passed = sum(1 for passed, _ in results.values() if passed)
    n_failed = total - n_passed
    overall  = n_failed == 0

    sep = "─" * 60
    print(f"\n{BOLD}{sep}{RESET}")
    print(f"{BOLD}  DataFrame Engineer Parameters Checklist Report{RESET}")
    print(f"{DIM}  {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print(f"{BOLD}{sep}{RESET}")

    overall_str = f"{GREEN}PASS{RESET}" if overall else f"{RED}FAIL{RESET}"
    print(f"  Overall : {BOLD}{overall_str}{RESET}   ({n_passed}/{total} passed)\n")

    if context:
        print(f"{BOLD}  Context{RESET}")
        for k, v in context.items():
            print(f"    {DIM}{k}{RESET}: {v}")
        print()

    print(f"{BOLD}  Results{RESET}")
    for name, (passed, _) in results.items():
        badge = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"    {badge}  {name}")

    failures = {name: msg for name, (passed, msg) in results.items() if not passed}
    if failures:
        print(f"\n{BOLD}  Failure details{RESET}")
        for name, msg in failures.items():
            print(f"  {RED}▶ {name}{RESET}")
            print(f"    {msg}")

    print(f"{BOLD}{sep}{RESET}\n")


def checklist_results_to_markdown(
    results: dict[str, tuple[bool, str]],
    output_file: str,
    context: dict | None = None,
) -> None:
    """
    Render the checklist outcome as a Markdown report, styled like tests/Summary.md.

    Args:
        results: mapping ``check_name -> (passed, message)`` as returned by ``run_all_checklist``.
        output_file: destination path for the ``.md`` file.
        context: optional dict with extra metadata (e.g. ``preset``, ``scenario_model``) rendered in the header.
    """
    total = len(results)
    n_passed = sum(1 for passed, _ in results.values() if passed)
    n_failed = total - n_passed
    overall_pass = n_failed == 0

    pass_badge = '<span style="color:green">PASS</span>'
    fail_badge = '<span style="color:red">FAIL</span>'

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# DataFrame Engineer Parameters Checklist Report\n")
        f.write(f"**Date**: {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Overall result**: {pass_badge if overall_pass else fail_badge}\n\n")
        f.write(f"**Total**: {total}  |  **Passed**: {n_passed}  |  **Failed**: {n_failed}\n\n")

        if context:
            f.write("## Context\n")
            f.write("| Parameter | Value |\n")
            f.write("|-----------|-------|\n")
            for k, v in context.items():
                f.write(f"| {k} | {v} |\n")
            f.write("\n")

        f.write("## Summary\n")
        f.write("| Check | Status |\n")
        f.write("|-------|--------|\n")
        for name, (passed, _) in results.items():
            f.write(f"| {name} | {pass_badge if passed else fail_badge} |\n")
        f.write("\n")

        failures = {name: msg for name, (passed, msg) in results.items() if not passed}
        if failures:
            f.write("## Failure details\n")
            for name, msg in failures.items():
                f.write(f"### {name}\n")
                f.write(f"{msg}\n\n")

    print(f"Generated Markdown report in: {os.path.realpath(output_file)}")


def __main__() ->None:
    # Imported here, not at module scope: building a Giulia pulls in the whole simulator
    # stack, and this module is imported by dedalo/ on the library path.
    from examples.giulia_EV import get_giulia_class

    # Setup a Giulia for testing the checklist
    preset = "GiuliaStd"

    # Define Example Parameters
    scenario_model = "3GPPTR38_901_5G"  # "3GPPTR38_901_5G"  # ITU_R_M2135_UMa
    ue_playground_model = None
    ue_distribution = "uniform"  # grid, uniform
    ue_mobility = None
    link_direction = "downlink"
    wraparound = None
    save_results = 0
    plots = False
    number_of_episodes = 1
    regression = False
    enable_GPU = True

    # Get Giulia class according to the preset
    Giulia = get_giulia_class(preset)

    _g = Giulia(preset,
                scenario_model,
                ue_playground_model,
                ue_distribution,
                ue_mobility,
                link_direction,
                wraparound,
                save_results,
                plots,
                number_of_episodes,
                regression,
                enable_GPU)


    # Get the DataFrame Engineer Parameters df_ep for the first episode
    # network df_ep is populated before the event set is configured via _g.configure(...)
    df_ep: pd.DataFrame = _g.network_deployment_obj.df_ep


    _context = {
        "preset": preset,
        "scenario_model": scenario_model,
        "ue_distribution": ue_distribution,
        "link_direction": link_direction
    }

    # Run the checklist on the df_ep
    checks_results = run_all_checklist(df_ep)

    # Print colored summary to terminal
    print_checklist_results(checks_results, context=_context)

    # Save a Markdown report next to the other Giulia outputs
    outputs_dir = os.path.join(os.path.abspath(root_dir), "outputs")
    report_path = os.path.join(outputs_dir, "df_ep_checklist.md")
    # checklist_results_to_markdown(checks_results, report_path, context=_context)

    overall_pass = all(passed for passed, _ in checks_results.values())
    print(f"Overall checklist result: {'PASS' if overall_pass else 'FAIL'}")



if __name__ == "__main__": __main__()
