#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 10:27:04 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import numpy as np 

def bandwidth_to_PRBs(bandwidth_MHz, subcarrier_spacing_Hz):
    
    # BW(j)- band Bandwidth, MHz (3GPP 38.104)
    # Section 5.3.2 Transmission bandwidth configuration
    if bandwidth_MHz == 1.4 and subcarrier_spacing_Hz == 15 : PRBs = 6  
    elif bandwidth_MHz == 5 and subcarrier_spacing_Hz == 15 : PRBs = 25  
    elif bandwidth_MHz == 10 and subcarrier_spacing_Hz == 15 : PRBs = 50 #52       
    elif bandwidth_MHz == 15 and subcarrier_spacing_Hz == 15 : PRBs = 75 #79         
    elif bandwidth_MHz == 20 and subcarrier_spacing_Hz == 15 : PRBs = 100 #106  
    elif bandwidth_MHz == 25 and subcarrier_spacing_Hz == 15 : PRBs = 133  
    elif bandwidth_MHz == 30 and subcarrier_spacing_Hz == 15 : PRBs = 160   
    elif bandwidth_MHz == 35 and subcarrier_spacing_Hz == 15 : PRBs = 188  
    elif bandwidth_MHz == 40 and subcarrier_spacing_Hz == 15 : PRBs = 216  
    elif bandwidth_MHz == 45 and subcarrier_spacing_Hz == 15 : PRBs = 242   
    elif bandwidth_MHz == 50 and subcarrier_spacing_Hz == 15 : PRBs = 270  
    elif bandwidth_MHz == 5 and subcarrier_spacing_Hz == 30 : PRBs = 11  
    elif bandwidth_MHz == 10 and subcarrier_spacing_Hz == 30 : PRBs = 24  
    elif bandwidth_MHz == 15 and subcarrier_spacing_Hz == 30 : PRBs = 38         
    elif bandwidth_MHz == 20 and subcarrier_spacing_Hz == 30 : PRBs = 51  
    elif bandwidth_MHz == 25 and subcarrier_spacing_Hz == 30 : PRBs = 65  
    elif bandwidth_MHz == 30 and subcarrier_spacing_Hz == 30 : PRBs = 78   
    elif bandwidth_MHz == 35 and subcarrier_spacing_Hz == 30 : PRBs = 92  
    elif bandwidth_MHz == 40 and subcarrier_spacing_Hz == 30 : PRBs = 106  
    elif bandwidth_MHz == 45 and subcarrier_spacing_Hz == 30 : PRBs = 119  
    elif bandwidth_MHz == 50 and subcarrier_spacing_Hz == 30 : PRBs = 133  
    elif bandwidth_MHz == 60 and subcarrier_spacing_Hz == 30 : PRBs = 162  
    elif bandwidth_MHz == 70 and subcarrier_spacing_Hz == 30 : PRBs = 189  
    elif bandwidth_MHz == 80 and subcarrier_spacing_Hz == 30 : PRBs = 217  
    elif bandwidth_MHz == 90 and subcarrier_spacing_Hz == 30 : PRBs = 245  
    elif bandwidth_MHz == 100 and subcarrier_spacing_Hz == 30 : PRBs = 273  
    elif bandwidth_MHz == 10 and subcarrier_spacing_Hz == 60 : PRBs = 11    
    elif bandwidth_MHz == 15 and subcarrier_spacing_Hz == 60 : PRBs = 18  
    elif bandwidth_MHz == 20 and subcarrier_spacing_Hz == 60 : PRBs = 24  
    elif bandwidth_MHz == 25 and subcarrier_spacing_Hz == 60 : PRBs = 31  
    elif bandwidth_MHz == 30 and subcarrier_spacing_Hz == 60 : PRBs = 38  
    elif bandwidth_MHz == 35 and subcarrier_spacing_Hz == 60 : PRBs = 44  
    elif bandwidth_MHz == 40 and subcarrier_spacing_Hz == 60 : PRBs = 51  
    elif bandwidth_MHz == 45 and subcarrier_spacing_Hz == 60 : PRBs = 58  
    elif bandwidth_MHz == 50 and subcarrier_spacing_Hz == 60 : PRBs = 65   
    elif bandwidth_MHz == 60 and subcarrier_spacing_Hz == 60 : PRBs = 79 
    elif bandwidth_MHz == 70 and subcarrier_spacing_Hz == 60 : PRBs = 93  
    elif bandwidth_MHz == 80 and subcarrier_spacing_Hz == 60 : PRBs = 107  
    elif bandwidth_MHz == 90 and subcarrier_spacing_Hz == 60 : PRBs = 121 
    
    elif bandwidth_MHz == 100 and subcarrier_spacing_Hz == 60 : PRBs = 135 
    elif bandwidth_MHz == 100 and subcarrier_spacing_Hz == 120 : PRBs = 90 # Made up config to speed up procesing      
    elif bandwidth_MHz == 100 and subcarrier_spacing_Hz == 240 : PRBs = 45 # Made up config to speed up procesing  
    
    elif bandwidth_MHz == 200 and subcarrier_spacing_Hz == 120 : PRBs = 135
    elif bandwidth_MHz == 200 and subcarrier_spacing_Hz == 240 : PRBs = 90 # Made up config to speed up procesing      
    elif bandwidth_MHz == 200 and subcarrier_spacing_Hz == 360 : PRBs = 45 # Made up config to speed up procesing         
    
    elif bandwidth_MHz == 400 and subcarrier_spacing_Hz == 240 : PRBs = 135
    elif bandwidth_MHz == 400 and subcarrier_spacing_Hz == 480 : PRBs = 90 # Made up config to speed up procesing    
    elif bandwidth_MHz == 400 and subcarrier_spacing_Hz == 720 : PRBs = 45 # Made up config to speed up procesing    

    else:
        raise ValueError(f"Unsupported bandwidth and subcarrier spacing combination (bandwidth_MHz: {bandwidth_MHz}, subcarrier_spacing_Hz: {subcarrier_spacing_Hz})")
                
    return PRBs 


def ofdm_symbol_duration(subcarrier_spacing_kHz): 
    
    if subcarrier_spacing_kHz == 15 :
        nu = 0
    elif subcarrier_spacing_kHz == 30 :
        nu = 1 
    elif subcarrier_spacing_kHz == 60 : 
        nu = 2  
    elif subcarrier_spacing_kHz == 120 : 
        nu = 3  
    else: 
        nu = 4          
        
        #https://www.techplayon.com/5g-nr-physical-layer-timing-unit/
        #https://www.techplayon.com/5g-nr-cyclic-prefix-cp-design/
        
    kappa = 64
    Ts = 1 / (15 * 1e3 * 2048)
    Tc = 1 / (480 * 1e3 * 4096)
    cp_us = 144 * kappa * np.power(2.0,-nu) * Ts / kappa * 1e6
        
    ofdm_symbol_duration_us = 1e3 / (14*np.power(2,nu)) - cp_us
    
    return ofdm_symbol_duration_us 


def control_channel_overhead(subcarrier_spacing_kHz):
    
    #OH(j) -overhead for control channels (3GPP 38.306)
    #The value represents a percentage
    
    if subcarrier_spacing_kHz == 15 or subcarrier_spacing_kHz == 30 or subcarrier_spacing_kHz == 60 :
        control_channel_overhead = 0.14   
    else: 
        control_channel_overhead = 0.18   
    
    return control_channel_overhead     