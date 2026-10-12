#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:27:04 2023

@author: dalopezperez
"""

import numpy as np 

def bandwidth_to_PRBs(bandwidth_MHz, subcarrier_spacing_Hz):
    # BW(j)- band Bandwidth, MHz (3GPP 38.104)
    # Section 5.3.2 Transmission bandwidth configuration
    # Key: (bandwidth_MHz, subcarrier_spacing_Hz) -> PRBs
    PRBs_table = {
        (1.4, 15): 6,
        (5,   15): 25,
        (10,  15): 50,   # 52
        (15,  15): 75,   # 79
        (20,  15): 100,  # 106
        (25,  15): 133,
        (30,  15): 160,
        (35,  15): 188,
        (40,  15): 216,
        (45,  15): 242,
        (50,  15): 270,
        (5,   30): 11,
        (10,  30): 24,
        (15,  30): 38,
        (20,  30): 51,
        (25,  30): 65,
        (30,  30): 78,
        (35,  30): 92,
        (40,  30): 106,
        (45,  30): 119,
        (50,  30): 133,
        (60,  30): 162,
        (70,  30): 189,
        (80,  30): 217,
        (90,  30): 245,
        (100, 30): 273,
        (10,  60): 11,
        (15,  60): 18,
        (20,  60): 24,
        (25,  60): 31,
        (30,  60): 38,
        (35,  60): 44,
        (40,  60): 51,
        (45,  60): 58,
        (50,  60): 65,
        (60,  60): 79,
        (70,  60): 93,
        (80,  60): 107,
        (90,  60): 121,
        (100, 60): 135,
        (100, 120): 90,   # Made up config to speed up procesing
        (100, 240): 45,   # Made up config to speed up procesing
        (200, 120): 132,
        (200, 240): 90,   # Made up config to speed up procesing
        (200, 360): 45,   # Made up config to speed up procesing
        (400, 240): 135,
        (400, 480): 90,   # Made up config to speed up procesing
        (400, 720): 45,   # Made up config to speed up procesing
    }

    key = (bandwidth_MHz, subcarrier_spacing_Hz)
    if key not in PRBs_table:
        # Group valid bandwidths by subcarrier spacing for a readable error message
        bw_by_scs: dict = {}
        for bw, scs in PRBs_table.keys():
            bw_by_scs.setdefault(scs, []).append(bw)

        header_scs = "Sub Carrier Spacing (SCS) [Hz]"
        header_bw  = "Valid Bandwidth [MHz]"
        rows = [(str(scs), ", ".join(str(bw) for bw in sorted(bws)))
                for scs, bws in sorted(bw_by_scs.items())]

        scs_width = max(len(header_scs), max(len(r[0]) for r in rows))
        bw_width  = max(len(header_bw),  max(len(r[1]) for r in rows))

        sep = f"  {'-' * scs_width}-+-{'-' * bw_width}"
        table_lines = [
            f"  {header_scs:<{scs_width}} | {header_bw}",
            sep,
            *[f"  {scs:<{scs_width}} | {bws}" for scs, bws in rows],
        ]
        valid_combinations = "\n".join(table_lines)

        raise ValueError(
            f"Invalid combination (bandwidth_MHz={bandwidth_MHz}, "
            f"subcarrier_spacing_Hz={subcarrier_spacing_Hz}).\n"
            f"Valid combinations are:\n{valid_combinations}"
        )

    return PRBs_table[key]


def ofdm_symbol_duration(subcarrier_spacing_kHz):

    # https://www.techplayon.com/5g-nr-physical-layer-timing-unit/
    # https://www.techplayon.com/5g-nr-cyclic-prefix-cp-design/
    # Key: subcarrier_spacing_kHz -> nu (numerology index)
    # Any value not in the table falls back to nu = 4 (matches original behavior).
    nu_table = {
        15:  0,
        30:  1,
        60:  2,
        120: 3,
    }

    # Use dict.get() to return the nu for the given subcarrier spacing, or 4 if it's not in the table.
    nu = nu_table.get(subcarrier_spacing_kHz, 4)

    kappa = 64
    Ts = 1 / (15 * 1e3 * 2048)
    Tc = 1 / (480 * 1e3 * 4096)
    cp_us = 144 * kappa * np.power(2.0, -nu) * Ts / kappa * 1e6

    ofdm_symbol_duration_us = 1e3 / (14*np.power(2, nu)) - cp_us

    return ofdm_symbol_duration_us


def control_channel_overhead(subcarrier_spacing_kHz):

    # OH(j) - overhead for control channels (3GPP 38.306)
    # The value represents a percentage
    # Key: subcarrier_spacing_kHz -> control channel overhead
    # Any value not in the table falls back to 0.18 (matches original behavior).
    overhead_table = {
        15: 0.14,
        30: 0.14,
        60: 0.14,
    }
    # Use dict.get() to return the overhead for the given subcarrier spacing, or 0.18 if it's not in the table.
    return overhead_table.get(subcarrier_spacing_kHz, 0.18)
