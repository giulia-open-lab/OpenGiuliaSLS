"""
Created on Mon Oct 23 16:39:49 2023


@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""



test_keys: list[str] = [
    "CRS_RSRP_no_fast_fading_ue_to_cell_dBm",
    "best_serving_cell_ID_per_ue_based_on_CRS",
    "best_serving_CRS_rsrp_per_ue_dBm",
    "best_serving_CRS_re_noise_per_ue_dBm",
    #"ues_per_cell",
    "CRS_sinr_ue_to_cell_dB",
    
    "SSB_RSRP_no_fast_fading_ue_to_cell_dBm",
    "best_serving_SSB_per_ue",
    "best_serving_SSB_rsrp_per_ue_dBm",
    "best_serving_SSB_re_noise_per_ue_dBm",
    "SSB_beam_activity_per_ue",
    "ues_per_SSB_beam",
    "SSB_sinr_ue_to_cell_dB",
    
    "best_serving_CSI_RS_per_ue",
    "best_serving_CSI_RS_rsrp_per_ue_dBm",
    "CSI_RS_beam_activity_per_ue",
    "ues_per_CSI_RS_beam",
    "effective_CSI_RS_sinr_ue_to_cell_dB",
    

    "ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps",
    "ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps",
    
    "cell_throughput_Mbps",
    "carrier_throughput_Mbps",
    "ue_throughput_per_carrier_Mbps",

    "total_network_power_consumption_kW"
]
"""
Contains the keys of the test results to be compared with the reference results.
"""


atol = 0.1
"""
The absolute tolerance for the comparison of the test results with the reference results.
"""


shadowing_files_url: str = "http://158.42.160.122:1234/shadowing.zip"
"""
The URL where the shadowing files can be downloaded from.
"""


shadowing_files_list: list[str] = [
    "G_shadowing_map_sigma_0_64_std_37.mat",
    "G_shadowing_map_sigma_3_34_std_37.mat",
    "G_shadowing_map_sigma_4_std_37.mat",
    "G_shadowing_map_sigma_8_std_13.mat",
    "G_shadowing_map_sigma_1_24_std_37.mat",
    "G_shadowing_map_sigma_3_std_10.mat",
    "G_shadowing_map_sigma_6_std_50.mat",
    "G_shadowing_map_sigma_8_std_50.mat",
    "G_shadowing_map_sigma_2_4_std_37.mat",
    "G_shadowing_map_sigma_4_std_10.mat",
    "G_shadowing_map_sigma_7_82_std_13.mat",
    "G_shadowing_map_sigma_2_std_10.mat",
    "G_shadowing_map_sigma_4_std_13.mat",
    "G_shadowing_map_sigma_7_std_7.mat"
]
"""
A list of the shadowing files to be expected in the shadowing directory.
"""


precomputed_files_url: str = "http://158.42.160.122:1234/regression_test.zip"
"""
The URL where the precomputed files can be downloaded from.
"""


precomputed_files_list: list[str] = [
    "3GPPTR36_777_UMa_AV_uniform",
    "3GPPTR36_777_UMi_AV_uniform",
    "3GPPTR36_814_Case_1_uniform",
    "3GPPTR38_901_4G_5G_6G_multilayer_inhomogeneous_per_cell",
    "3GPPTR38_901_4G_5G_multilayer_inhomogeneous_per_cell",
    "3GPPTR38_901_4G5G_multilayer_inhomogeneous_per_cell",
    "3GPPTR38_901_4G_inhomogeneous_per_cell",
    "3GPPTR38_901_5G_inhomogeneous_per_cell",
    "3GPPTR38_901_UMa_C1_uniform",
    "3GPPTR38_901_UMa_C2_uniform",
    "3GPPTR38_901_UMa_lsc_uniform",
    "3GPPTR38_901_UMi_C1_uniform",
    "3GPPTR38_901_UMi_C2_uniform",
    "3GPPTR38_901_UMi_lsc_uniform",
    "ITU_R_M2135_UMa_inhomogeneous_per_cell",
    "ITU_R_M2135_UMa_inhomogeneous_per_cell_with_hotspots",
    "ITU_R_M2135_UMa_uniform",
    "ITU_R_M2135_UMa_uniform_with_hotspots",
    "ITU_R_M2135_UMi_uniform",
]
"""
A list of the precomputed test directories to be expected in the precomputed files directory.
"""