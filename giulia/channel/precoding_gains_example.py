# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:53:23 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import numpy as np
from scipy.linalg import dft

from giulia.plots import plotting
from giulia.tools import tools


def calculate_array_factor(steering_vector_rad,
                           number_of_ues, 
                           phi_rad,
                           alpha_mec_bearing_rad,
                           theta_rad,
                           antenna_config_dh_lambda, 
                           antenna_config_dv_lambda, 
                           I):
    
    # Note that the I matrix contains the numbering of the antenna elements in the uniform array. 
    # The numbering is as per 3GPP TR 38.901 with (y,z) = (0,0) for the bottom left element. X is always equal to zero, as the reference array lays on the y-z plane
    stv_mat_rad = steering_vector_rad[np.newaxis,:] * np.ones((number_of_ues,1)) # This is the same as np.tile(steering_vector_rad,(number_of_ues,1)) 
    phi_rel_rad = tools.angle_range_0_pi(phi_rad - alpha_mec_bearing_rad) # We need an LCS persepctive. Thus, we substruct the bearing of the sector to get the relative phi angle of the UE to query the antenna pattern
    
    k = 2 * np.pi * np.array([np.cos(phi_rel_rad.ravel()) * np.sin(theta_rad.ravel()), 
                              antenna_config_dh_lambda * np.sin(phi_rel_rad.ravel()) * np.sin(theta_rad.ravel()),
                              antenna_config_dv_lambda * np.cos(theta_rad.ravel())])
    
    return np.exp(1j * np.matmul(k.T, I)) * np.exp(1j * np.matmul(stv_mat_rad, I))


def precoding_gain_per_cell(number_of_ues, 
                            number_of_beams_in_cell,
                            cell_codebook,
                            cell_complex_channel_gain,
                            A_m_dB):   
                                       
    # Placeholder to store the precoding gain results for all beams of a cell
    precoding_gain_per_beam_linear = np.zeros((number_of_ues, number_of_beams_in_cell)) #Solution for DFT precoder   
    
    # Iterate over all precoder codewords and compute the gain
    
    for cell_codebook_book_index in range(0, number_of_beams_in_cell):
        
        # Get precoder codeword
        precoder = cell_codebook[:,cell_codebook_book_index] 
        
        # Add the precoder gain to the array response gain
        precoding_gain_per_beam_linear[:, cell_codebook_book_index] = np.abs(np.sum(precoder * cell_complex_channel_gain, axis=1)) 
        
    # Calculate the array response gains in dB
    precoding_gain_per_beam_dB = 10 * np.log10(precoding_gain_per_beam_linear)
    
    # Cap minimum values
    precoding_gain_per_beam_dB[precoding_gain_per_beam_dB < -A_m_dB] = -A_m_dB   
        
    return precoding_gain_per_beam_dB    
    

def example_SSB():
  
    # Set parameters for the example   
    azimuth_deg = np.arange(-180, 180, 180/500)
    zenith_deg = np.arange(0, 180, 180/1000)
    
    alpha_mec_bearing_deg = 0 
    beta_elec_downtilt_deg = 90 
    
    antenna_config_dh_lambda = 1/2
    antenna_config_dv_lambda = 1/2 
    
    M = 10 # Number of elements in vertical
    N = 2 # Number of elements in horizontal
    
    beam_type = "SSB"
    beams_H = 2 # Number of H beams
    beams_V = 5 # Number of V beams      
    
    A_m_dB = 30     
    
    # Calculate electric steering vector 
    phase_difference_alpha = 0 # - 2 * np.pi * antenna_config_dh_lambda  * np.sin(alpha_bearing_deg * np.pi / 180) # Horizontal plane
    phase_difference_beta = - 2 * np.pi * antenna_config_dv_lambda * np.cos(beta_elec_downtilt_deg * np.pi / 180) # Vertical plane
    steering_vector_rad = np.array([0, phase_difference_alpha, phase_difference_beta]) 
    
    # Calcualte antenna indeces 
    I = np.array([[0, tools.order_i(m, N), tools.order_j(m, N)] for m in range(1, N * M + 1)]).T
    
    # Calculate array factor for horizontal antenna and strore it to use it later in precoding_gains.py
    number_of_ues_H = np.size(azimuth_deg)
    example_cell_channel_H = calculate_array_factor(steering_vector_rad, 
                                                     number_of_ues_H,
                                                     np.radians(azimuth_deg), 
                                                     np.radians(alpha_mec_bearing_deg), 
                                                     np.radians(np.ones(np.size(azimuth_deg))*beta_elec_downtilt_deg), 
                                                     antenna_config_dh_lambda, 
                                                     antenna_config_dv_lambda, 
                                                     I)
    
    # Calculate array factor for vertical antenna and strore it to use it later in precoding_gains.py
    number_of_ues_V = np.size(azimuth_deg)
    example_cell_channel_V = calculate_array_factor(steering_vector_rad, 
                                                     number_of_ues_V,
                                                     np.radians(np.ones(np.size(zenith_deg))*alpha_mec_bearing_deg),  
                                                     np.radians(alpha_mec_bearing_deg), 
                                                     np.radians(zenith_deg),
                                                     antenna_config_dh_lambda, 
                                                     antenna_config_dv_lambda, 
                                                     I)     
    
    # Calculate precoder
    # Placeholder to store codebook. The codebook size is antenna elements x beams     
    cell_codebook = np.zeros((M*N, beams_H*beams_V)).astype(complex) 
    
    # Calculate DFT matrices
    # Note that the current strategy uses as many elements in H and V domain as H and V beams we want to create
    # Other strategies may apply, e.g. generate as many H and V beams as H and V elements and down select the wanted beams  
    dft_V = np.zeros((M,M)).astype(complex)
    dft_V[:beams_V,:beams_V] = 1/np.sqrt(beams_V)*dft(beams_V)
    dft_H = np.zeros((N,N)).astype(complex)
    dft_H[:beams_H,:beams_H] = 1/np.sqrt(beams_H)*dft(beams_H)
    dft_precoders = np.kron(dft_V, dft_H)
    
    #Store precoder
    cell_codebook = dft_precoders[:,~np.all(dft_precoders == 0, axis=0)] # Removing the null precoders 
    
    #Get precoder plus array response gain in the H plane - we make a cut in the V-plane in the direction of the beta_downtilt
    precoding_gain_per_beam_dB_H = precoding_gain_per_cell(np.size(azimuth_deg), 
                                                                 np.size(cell_codebook,1),
                                                                 cell_codebook,
                                                                 example_cell_channel_H,
                                                                 A_m_dB)  
    
    # Get precoder plus array response gain in the V plane - we make a cut in the H-plane in the direction 0 degree azimuth
    precoding_gain_per_beam_dB_V = precoding_gain_per_cell(np.size(zenith_deg), 
                                                                 np.size(cell_codebook,1),
                                                                 cell_codebook,
                                                                 example_cell_channel_V,
                                                                 A_m_dB) 
    
    # Plot H plane 
    plotting.plot_antenna_gain_4_figures(np.radians(azimuth_deg), tools.dBm_to_mW(precoding_gain_per_beam_dB_H), [-np.pi / 2, np.pi / 2],
                                         np.radians(azimuth_deg), precoding_gain_per_beam_dB_H, [-np.pi/2, np.pi/2],
                                         beam_type + ' beam gain - Horizontal')
    

    # Plot V plane 
    plotting.plot_antenna_gain_4_figures(np.radians(zenith_deg), tools.dBm_to_mW(precoding_gain_per_beam_dB_V), [0, np.pi],
                                         np.radians(zenith_deg), precoding_gain_per_beam_dB_V, [0, np.pi],
                                         beam_type + ' beam gain - Vertical')       
