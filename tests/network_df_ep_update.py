"""Repairs for the derived columns of Giulia's engineering-parameter table (``df_ep``).

Each ``update_*`` recomputes one derived column from its sources and returns the frame.
They are the counterpart of the checks in ``tests/network_df_ep_check_list.py``: every
failing check should have an updater that makes it pass.

:data:`DERIVED_FIELD_UPDATERS` maps a *source* column to the updaters that must run after
it is written, so a caller that writes one column can repair just its dependants.
``Dedalo.repair_df_ep`` / ``Dedalo.set_giulia_network_df_ep`` call into here.
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

from tests.network_df_ep_check_list import run_all_checklist, print_checklist_results


### update dl_carrier_wavelength_m from dl_carrier_frequency_GHz
###############################################################
def update_dl_carrier_wavelength_from_frequency(df_ep: pd.DataFrame) -> pd.DataFrame:
    attr_name: str = "dl_carrier_wavelength_m"
    dependent_attribute: str = "dl_carrier_frequency_GHz"

    dl_carrier_frequency_GHz = np.asarray(df_ep[dependent_attribute].values, dtype=float)
    df_ep[attr_name] = 3e8 / (dl_carrier_frequency_GHz * 1e9)
    return df_ep


### update ul_carrier_wavelength_m from ul_carrier_frequency_GHz
###############################################################
def update_ul_carrier_wavelength_from_frequency(df_ep: pd.DataFrame) -> pd.DataFrame:
    attr_name: str = "ul_carrier_wavelength_m"
    dependent_attribute: str = "ul_carrier_frequency_GHz"

    ul_carrier_frequency_GHz = np.asarray(df_ep[dependent_attribute].values, dtype=float)
    df_ep[attr_name] = 3e8 / (ul_carrier_frequency_GHz * 1e9)
    return df_ep


### update dl_PRBs_available from dl_bandwidth_MHz and dl_subcarrier_spacing_kHz
################################################################################
def update_dl_PRBs_from_bandwidth_and_scs(df_ep: pd.DataFrame) -> pd.DataFrame:
    attr_name: str = "dl_PRBs_available"
    dependent_attributes: dict = {"dl_bandwidth_MHz": "dl_bandwidth_MHz", "dl_subcarrier_spacing_kHz": "dl_subcarrier_spacing_kHz"}

    dl_bandwidth_MHz = np.asarray(df_ep[dependent_attributes["dl_bandwidth_MHz"]].values, dtype=float)
    dl_subcarrier_spacing_kHz = np.asarray(df_ep[dependent_attributes["dl_subcarrier_spacing_kHz"]].values, dtype=float)

    new_dl_PRBs_available = [tools_carrier.bandwidth_to_PRBs(bw, scs) for bw, scs in zip(dl_bandwidth_MHz, dl_subcarrier_spacing_kHz)]
    df_ep[attr_name] = np.array(new_dl_PRBs_available, dtype=int)
    return df_ep


### update dl_ofdm_symbol_duration_us from dl_subcarrier_spacing_kHz
###################################################################
def update_dl_ofdm_symbol_duration_from_scs(df_ep: pd.DataFrame) -> pd.DataFrame:
    attr_name: str = "dl_ofdm_symbol_duration_us"
    dependent_attribute: str = "dl_subcarrier_spacing_kHz"

    dl_subcarrier_spacing_kHz = np.asarray(df_ep[dependent_attribute].values, dtype=float)
    df_ep[attr_name] = np.array([tools_carrier.ofdm_symbol_duration(scs) for scs in dl_subcarrier_spacing_kHz], dtype=float)
    return df_ep


### update dl_control_channel_overhead from dl_subcarrier_spacing_kHz
#####################################################################
def update_dl_control_channel_overhead_from_scs(df_ep: pd.DataFrame) -> pd.DataFrame:
    attr_name: str = "dl_control_channel_overhead"
    dependent_attribute: str = "dl_subcarrier_spacing_kHz"

    dl_subcarrier_spacing_kHz = np.asarray(df_ep[dependent_attribute].values, dtype=float)
    df_ep[attr_name] = np.array([tools_carrier.control_channel_overhead(scs) for scs in dl_subcarrier_spacing_kHz], dtype=float)
    return df_ep


### update ul_available_PRBs from ul_bandwidth_MHz and ul_subcarrier_spacing_kHz
################################################################################
def update_ul_PRBs_from_bandwidth_and_scs(df_ep: pd.DataFrame) -> pd.DataFrame:
    attr_name: str = "ul_available_PRBs"
    dependent_attributes: dict = {"ul_bandwidth_MHz": "ul_bandwidth_MHz", "ul_subcarrier_spacing_kHz": "ul_subcarrier_spacing_kHz"}

    ul_bandwidth_MHz = np.asarray(df_ep[dependent_attributes["ul_bandwidth_MHz"]].values, dtype=float)
    ul_subcarrier_spacing_kHz = np.asarray(df_ep[dependent_attributes["ul_subcarrier_spacing_kHz"]].values, dtype=float)

    new_ul_available_PRBs = [tools_carrier.bandwidth_to_PRBs(bw, scs) for bw, scs in zip(ul_bandwidth_MHz, ul_subcarrier_spacing_kHz)]
    df_ep[attr_name] = np.array(new_ul_available_PRBs, dtype=int)
    return df_ep


### update ul_ofdm_symbol_duration_us from ul_subcarrier_spacing_kHz
###################################################################
def update_ul_ofdm_symbol_duration_from_scs(df_ep: pd.DataFrame) -> pd.DataFrame:
    attr_name: str = "ul_ofdm_symbol_duration_us"
    dependent_attribute: str = "ul_subcarrier_spacing_kHz"

    ul_subcarrier_spacing_kHz = np.asarray(df_ep[dependent_attribute].values, dtype=float)
    df_ep[attr_name] = np.array([tools_carrier.ofdm_symbol_duration(scs) for scs in ul_subcarrier_spacing_kHz], dtype=float)
    return df_ep


### update ul_control_channel_overhead from ul_subcarrier_spacing_kHz
#####################################################################
def update_ul_control_channel_overhead_from_scs(df_ep: pd.DataFrame) -> pd.DataFrame:
    attr_name: str = "ul_control_channel_overhead"
    dependent_attribute: str = "ul_subcarrier_spacing_kHz"

    ul_subcarrier_spacing_kHz = np.asarray(df_ep[dependent_attribute].values, dtype=float)
    df_ep[attr_name] = np.array([tools_carrier.control_channel_overhead(scs) for scs in ul_subcarrier_spacing_kHz], dtype=float)
    return df_ep


#### update antenna_config_number_of_elements from antenna_config_{Mg, Ng, M, N, P}
####################################################################################
def update_antenna_config_number_of_elements_from_MgNgMNP(df_ep: pd.DataFrame) -> pd.DataFrame:
    attr_name: str = "antenna_config_number_of_elements"
    dependent_attributes: dict = {
        "antenna_config_Mg": "antenna_config_Mg",
        "antenna_config_Ng": "antenna_config_Ng",
        "antenna_config_M":  "antenna_config_M",
        "antenna_config_N":  "antenna_config_N",
        "antenna_config_P":  "antenna_config_P",
    }

    Mg = np.asarray(df_ep[dependent_attributes["antenna_config_Mg"]].values, dtype=int)
    Ng = np.asarray(df_ep[dependent_attributes["antenna_config_Ng"]].values, dtype=int)
    M = np.asarray(df_ep[dependent_attributes["antenna_config_M"]].values, dtype=int)
    N = np.asarray(df_ep[dependent_attributes["antenna_config_N"]].values, dtype=int)
    P = df_ep[dependent_attributes["antenna_config_P"]].map({"single": 1, "dual": 2}).astype(int).values

    df_ep[attr_name] = (Mg * Ng * M * N * P).astype(int)
    return df_ep




# Registry of every available update function, keyed by name for selective runs
ALL_UPDATE_FUNCTIONS = [
    update_dl_carrier_wavelength_from_frequency,
    update_ul_carrier_wavelength_from_frequency,

    update_dl_PRBs_from_bandwidth_and_scs,
    update_dl_ofdm_symbol_duration_from_scs,
    update_dl_control_channel_overhead_from_scs,

    update_ul_PRBs_from_bandwidth_and_scs,
    update_ul_ofdm_symbol_duration_from_scs,
    update_ul_control_channel_overhead_from_scs,

    update_antenna_config_number_of_elements_from_MgNgMNP,
]
UPDATE_FUNCTIONS_BY_NAME = {f.__name__: f for f in ALL_UPDATE_FUNCTIONS}


# Source column -> the updaters that must run after it is written.
#
# Keyed by the column a caller actually writes, NOT by the column that goes stale, so a
# setter can look up its own argument. A column absent from this map has no derived
# dependants and needs no repair — which is why targeted repair is a no-op for writes such
# as ``radio_unit_initial_status`` or ``BS_tx_power_max_dBm``.
DERIVED_FIELD_UPDATERS: dict[str, list] = {
    "dl_carrier_frequency_GHz": [update_dl_carrier_wavelength_from_frequency],
    "ul_carrier_frequency_GHz": [update_ul_carrier_wavelength_from_frequency],

    "dl_bandwidth_MHz":         [update_dl_PRBs_from_bandwidth_and_scs],
    "ul_bandwidth_MHz":         [update_ul_PRBs_from_bandwidth_and_scs],

    # Subcarrier spacing drives the PRB count, the symbol duration and the control overhead.
    "dl_subcarrier_spacing_kHz": [update_dl_PRBs_from_bandwidth_and_scs,
                                  update_dl_ofdm_symbol_duration_from_scs,
                                  update_dl_control_channel_overhead_from_scs],
    "ul_subcarrier_spacing_kHz": [update_ul_PRBs_from_bandwidth_and_scs,
                                  update_ul_ofdm_symbol_duration_from_scs,
                                  update_ul_control_channel_overhead_from_scs],

    # Array geometry drives the total element count.
    "antenna_config_Mg": [update_antenna_config_number_of_elements_from_MgNgMNP],
    "antenna_config_Ng": [update_antenna_config_number_of_elements_from_MgNgMNP],
    "antenna_config_M":  [update_antenna_config_number_of_elements_from_MgNgMNP],
    "antenna_config_N":  [update_antenna_config_number_of_elements_from_MgNgMNP],
    "antenna_config_P":  [update_antenna_config_number_of_elements_from_MgNgMNP],
}


# Columns that are CHECKED but have no updater, so they cannot be repaired automatically.
# A caller that writes one of these must recompute the dependant itself; the checklist is
# the only thing that will report the drift. See docs/dedalo/df-ep-consistency.md.
UNREPAIRABLE_DEPENDANTS: dict[str, str] = {
    "SSB_precoder":             "SSB_number_of_beams",
    "SSB_number_of_beams_H":    "SSB_number_of_beams",
    "SSB_number_of_beams_V":    "SSB_number_of_beams",
    "CSI_RS_precoder":          "CSI_RS_number_of_beams",
    "CSI_RS_number_of_beams_H": "CSI_RS_number_of_beams",
    "CSI_RS_number_of_beams_V": "CSI_RS_number_of_beams",
}


def updaters_for_field(name_attribute: str) -> list:
    """Return the updaters invalidated by writing ``name_attribute`` (possibly empty)."""
    return list(DERIVED_FIELD_UPDATERS.get(name_attribute, []))


def repair_dependants_of(df_ep: pd.DataFrame, name_attribute: str) -> pd.DataFrame:
    """Recompute only the columns that writing ``name_attribute`` invalidated.

    A no-op for columns with no derived dependants, which is most of them. Unlike
    :func:`run_all_updates` this touches nothing unrelated to the write, so it cannot
    mask a pre-existing inconsistency elsewhere in the table.
    """
    for update_func in updaters_for_field(name_attribute):
        df_ep = update_func(df_ep)
    return df_ep


def run_all_updates(df_ep: pd.DataFrame) -> pd.DataFrame:
    """Apply every dependent-field update in ``ALL_UPDATE_FUNCTIONS`` to df_ep."""
    for update_func in ALL_UPDATE_FUNCTIONS:
        df_ep = update_func(df_ep)
        print(f"{update_func.__name__}: applied")
    return df_ep


def run_selected_updates(df_ep: pd.DataFrame, selected: list) -> pd.DataFrame:
    """
    Apply a chosen subset of dependent-field updates to df_ep.

    Args:
        df_ep: DataFrame Engineer Parameters to update in place.
        selected: list of update functions to run. Each item can be either the
            function itself (e.g. ``update_dl_PRBs_from_bandwidth_and_scs``) or its name
            as a string.
    """
    update_functions = []
    for item in selected:
        if callable(item):
            update_functions.append(item)
        elif isinstance(item, str):
            if item not in UPDATE_FUNCTIONS_BY_NAME:
                raise ValueError(
                    f"Unknown update function name: {item!r}. "
                    f"Available: {list(UPDATE_FUNCTIONS_BY_NAME)}"
                )
            update_functions.append(UPDATE_FUNCTIONS_BY_NAME[item])
        else:
            raise TypeError(f"selected entries must be callables or strings, got {type(item).__name__}")

    for update_func in update_functions:
        df_ep = update_func(df_ep)
        print(f"{update_func.__name__}: applied")

    return df_ep

def __main__():
    # Imported here, not at module scope: building a Giulia pulls in the whole simulator
    # stack, and this module is imported by dedalo/ on the library path.
    from examples.giulia_EV import get_giulia_class

    # Setup a Giulia for testing the updates
    preset = "GiuliaStd"

    scenario_model = "3GPPTR38_901_5G"
    ue_playground_model = None
    ue_distribution = "uniform"
    ue_mobility = None
    link_direction = "downlink"
    wraparound = None
    save_results = 0
    plots = False
    number_of_episodes = 1
    regression = False
    enable_GPU = True

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

    df_ep: pd.DataFrame = _g.network_deployment_obj.df_ep

    _context = {
        "preset": preset,
        "scenario_model": scenario_model,
        "ue_distribution": ue_distribution,
        "link_direction": link_direction,
    }

    # Run the checklist on the df_ep BEFORE applying any updates
    # print("\n=== Checklist BEFORE updates ===")
    # checks_results_before = run_all_checklist(df_ep)
    # print_checklist_results(checks_results_before, context=_context)

    # Apply every update. To run only a subset, use run_selected_updates instead, e.g.:
    #   df_ep = run_selected_updates(df_ep, [update_dl_PRBs_from_bandwidth_and_scs, update_ul_PRBs_from_bandwidth_and_scs])
    df_ep = run_selected_updates(df_ep, ["update_dl_PRBs_from_bandwidth_and_scs"])
    # df_ep = run_all_updates(df_ep)

    print("\nAll dependent fields in df_ep have been recomputed from their source attributes.")

if __name__ == "__main__": __main__()
    
