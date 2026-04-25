# -*- coding: utf-8 -*-

"""
Created on Tue Aug 27 14:39:31 2024

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import gc

import giulia_EV

    
def run_simulations():
    # Define the scenarios and UE distributions you want to run

    scenarios = [
        "3GPPTR38_901_4G",
        "3GPPTR38_901_5G",
        "3GPPTR38_901_4G5G_multilayer",
        "3GPPTR38_901_4G_5G_multilayer",
        "3GPPTR38_901_4G_5G2_multilayer",
        "3GPPTR38_901_4G_5G6G_multilayer",
        "3GPPTR38_901_4G_5G_6G_multilayer",
    ]
    
    ue_distributions = [
        "inhomogeneous_per_cell"
    ]
    
    # Iterate over each scenario and UE distribution
    for scenario in scenarios:
        for ue_distribution in ue_distributions:
            print(f"Running simulation with scenario: {scenario} and UE distribution: {ue_distribution}")
            
            # Prepare the arguments as they would be provided in the command line
            args = [
                "--scenario_model", scenario,
                "--ue_distribution", ue_distribution
            ]
            
            # Call the main function of the imported script
            giulia_EV.main(args)
            
            # Clear memory after each iteration
            # Clear all variables in memory after each iteration
            for name in list(globals().keys()):
                if name not in ['giulia_EV', 'run_simulations', 'gc']:  # Don't delete necessary modules and functions
                    del globals()[name]
                    
            gc.collect()  # Force garbage collection to free up memory            

if __name__ == "__main__":
    run_simulations()

