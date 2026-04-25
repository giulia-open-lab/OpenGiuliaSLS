# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:08:30 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import time
from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.metrics
from scipy.spatial.distance import cdist
import torch

from giulia.event_driven.snapshot_control import Snapshot_control
from giulia.fs import results_file
from giulia.tools.tools import log_calculations_time
from giulia.outputs.saveable import Saveable

class Distance_Angles(Saveable):
    
    def __init__(self,  
                 simulation_config_obj,
                 site_deployment_obj,
                 network_deployment_obj,
                 ue_deployment_obj):

        super().__init__()

        #### Torch 
        ########################
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        ##### Input storage 
        ######################## 
        self.simulation_config_obj = simulation_config_obj
        self.site_deployment_obj = site_deployment_obj
        self.network_deployment_obj = network_deployment_obj
        self.ue_deployment_obj = ue_deployment_obj      
        
        
        ##### Outputs 
        ########################   
        
        # Place holder to store path loss results
        self.distances_b_to_a_2d_m = []
        self.distances_b_to_a_3d_m = []
        self.azimuths_b_to_a_degrees = []
        self.zeniths_b_to_a_degrees = []
        self.distance_b_to_a_2d_wraparound_m = []
        self.distance_b_to_a_3d_wraparound_m = []
        self.azimuths_b_to_a_wraparound_degrees = []
        self.zeniths_b_to_a_wraparound_degrees = []
                
        
    def variables_list(self) -> List[str]:
        """List of attributes name to be saved."""
        return ["distances_b_to_a_3d_m"]

       
    def process(self, rescheduling_us=-1): 
       
        ##### Process inputs
        ######################## 
        
        # Random numbers
        self.rng = np.random.RandomState(self.simulation_config_obj.random_seed+0)
        
        #Site
        self.hex_layout_tiers = self.site_deployment_obj.hex_layout_tiers
        self.isd_m = self.site_deployment_obj.isd_m
        
        # Network
            ### Please note the abuse of notation - A cell does not really have a position. 
            ### It is the cell site the one that has a position. 
            ### However, we adopt a UE x CELLS structure for convenience in latter operations.
        
        self.positions_a_m = self.network_deployment_obj.df_ep[["position_x_m", "position_y_m", "position_z_m"]].to_numpy(dtype=np.single)
       
        # UE deployment
        self.positions_b_m = self.ue_deployment_obj.df_ep[["position_x_m", "position_y_m", "position_z_m"]].to_numpy(dtype=np.single) 
       
        
        ##### Start timer
        ########################  
        t_start = time.perf_counter()
        
        
        ##### Calcualte distances
        ########################          

        # Calcualate 2D distances without wraparound. Size b-by-a (UE-by-cell)
        self.distances_b_to_a_2d_m = self.calculate_distances_b_to_a(self.positions_b_m[:,:2], self.positions_a_m[:,:2])      
        
        # Calcualate 3D distances without wraparound. Size b-by-a (UE-by-cell)
        self.distances_b_to_a_3d_m = self.calculate_distances_b_to_a(self.positions_b_m, self.positions_a_m)

        # Calcualate azimuths and zeniths without wraparound. Size b-by-a (UE-by-cell)
        self.azimuths_b_to_a_degrees, \
            self.zeniths_b_to_a_degrees \
                = self.calculate_azimuth_zenith_b_to_a(self.positions_b_m, self.positions_a_m, self.distances_b_to_a_2d_m)
                       
        # Store in data frames as it may be useful to post process
        self.df_distances_b_to_a_2d_m = pd.DataFrame(self.distances_b_to_a_2d_m, columns=self.network_deployment_obj.df_ep["name"], index=self.ue_deployment_obj.df_ep["name"])   
        self.df_distances_b_to_a_3d_m = pd.DataFrame(self.distances_b_to_a_3d_m, columns=self.network_deployment_obj.df_ep["name"], index=self.ue_deployment_obj.df_ep["name"])  
        self.df_azimuths_b_to_a_degrees = pd.DataFrame(self.azimuths_b_to_a_degrees, columns=self.network_deployment_obj.df_ep["name"], index=self.ue_deployment_obj.df_ep["name"])  
        self.df_zeniths_b_to_a_degrees = pd.DataFrame(self.zeniths_b_to_a_degrees, columns=self.network_deployment_obj.df_ep["name"], index=self.ue_deployment_obj.df_ep["name"])                  

        # Calcualate 2D and 3D distances as well as azimuths and zeniths with wraparound. Size b-by-a (UE-by-cell)
            # Wraparound only makes sense if we specified tiers and isd_m
        if self.site_deployment_obj.wraparound == True:
            self.distance_b_to_a_2d_wraparound_m, \
            self.distance_b_to_a_3d_wraparound_m, \
            self.azimuths_b_to_a_wraparound_degrees, \
            self.zeniths_b_to_a_wraparound_degrees \
                = self.calculate_distances_azimuth_zenith_b_to_a_wraparound(self.positions_b_m, self.positions_a_m, self.hex_layout_tiers, self.isd_m)
            
            # Otherwise use distances/angles without wraparound
        else:
            print("Number of tiers or isd_m are not specified. Cannot use wraparound.")
            self.distance_b_to_a_2d_wraparound_m = self.df_distances_b_to_a_2d_m.values
            self.distance_b_to_a_3d_wraparound_m = self.df_distances_b_to_a_3d_m.values
            self.azimuths_b_to_a_wraparound_degrees = self.df_azimuths_b_to_a_degrees.values
            self.zeniths_b_to_a_wraparound_degrees = self.df_zeniths_b_to_a_degrees.values

        # Store in data frames as it may be useful to post process
        self.df_distances_b_to_a_2d_wraparound_m = pd.DataFrame(self.distance_b_to_a_2d_wraparound_m, columns=self.network_deployment_obj.df_ep["name"], index=self.ue_deployment_obj.df_ep["name"])   
        self.df_distances_b_to_a_3d_wraparound_m = pd.DataFrame(self.distance_b_to_a_3d_wraparound_m, columns=self.network_deployment_obj.df_ep["name"], index=self.ue_deployment_obj.df_ep["name"])  
        self.df_azimuths_b_to_a_wraparound_degrees = pd.DataFrame(self.azimuths_b_to_a_wraparound_degrees, columns=self.network_deployment_obj.df_ep["name"], index=self.ue_deployment_obj.df_ep["name"])  
        self.df_zeniths_b_to_a_wraparound_degrees = pd.DataFrame(self.zeniths_b_to_a_wraparound_degrees, columns=self.network_deployment_obj.df_ep["name"], index=self.ue_deployment_obj.df_ep["name"])  
        
        # Calculate 2d_in distance
        if 'indoor' in self.ue_deployment_obj.df_ep.columns:  
            self.d_2D_in_m = np.minimum(25 * self.rng.rand(np.size(self.positions_b_m,0)), 25 * self.rng.rand(np.size(self.positions_b_m,0)))   
            self.indoor = self.ue_deployment_obj.df_ep[["indoor"]].to_numpy(dtype=bool)  
            self.d_2D_in_m[self.indoor[:,0] == False] = 0
            self.d_2D_in_m = self.d_2D_in_m[:, np.newaxis] * np.ones((1,np.size(self.positions_a_m,0))) 
          
            
        ##### Save to plot
        ########################
        snapshot_control = Snapshot_control.get_instance()
        if self.simulation_config_obj.plot == 1 and snapshot_control.num_snapshots == 0:
            file_name = results_file(self.simulation_config_obj.project_name, 'to_plot_distances_angles')
            np.savez(file_name,
                distances_b_to_a_2d_m=self.distances_b_to_a_2d_m,
                distance_b_to_a_2d_wraparound_m=self.distance_b_to_a_2d_wraparound_m,
                distances_b_to_a_3d_m=self.distances_b_to_a_3d_m,
                distance_b_to_a_3d_wraparound_m=self.distance_b_to_a_3d_wraparound_m,
                azimuths_b_to_a_degrees=self.azimuths_b_to_a_degrees,
                azimuths_b_to_a_wraparound_degrees=self.azimuths_b_to_a_wraparound_degrees,
                zeniths_b_to_a_degrees=self.zeniths_b_to_a_degrees,
                zeniths_b_to_a_wraparound_degrees=self.zeniths_b_to_a_wraparound_degrees
            )
            
            
        ##### End
        ########################   
        log_calculations_time('Distance', t_start)

        return rescheduling_us       
               
            
    def calculate_distances_b_to_a(self, b, a):
        return (cdist(b, a)).astype(np.single)


    def calculate_azimuth_zenith_b_to_a(self, positions_b_m, positions_a_m, distance_2d_m): 
        
        # Calculate deltas
        delta_x_m =  np.subtract.outer(positions_b_m[:,0], positions_a_m[:,0])
        delta_y_m =  np.subtract.outer(positions_b_m[:,1], positions_a_m[:,1])
        delta_z_m =  - np.subtract.outer(positions_b_m[:,2], positions_a_m[:,2]) #Note that the site position height is higher than the UE one
        
        # Calculate azimuth 
        azimuth_degrees = np.degrees(np.arctan2(delta_y_m, delta_x_m))
        
        # Calcualte zenith
        zenith_degrees = np.degrees(np.pi/2 + np.arctan2(delta_z_m, distance_2d_m))
        
        result = azimuth_degrees.astype(np.single), zenith_degrees.astype(np.single)

        return result


    def calculate_azimuth_zenith_b_to_a_torch(self, positions_b_m, positions_a_m, distance_2d_m): 
        
        # Calculate deltas
        delta_x_m = positions_b_m[:,0].unsqueeze(1) - positions_a_m[:,0].unsqueeze(0)
        delta_y_m =  positions_b_m[:,1].unsqueeze(1) - positions_a_m[:,1].unsqueeze(0)
        delta_z_m =  - (positions_b_m[:,2].unsqueeze(1) - positions_a_m[:,2].unsqueeze(0)) #Note that the site position height is higher than the UE one
        
        # Calculate azimuth 
        azimuth_degrees = torch.rad2deg(torch.arctan2(delta_y_m, delta_x_m))
        
        # Calcualte zenith
        zenith_degrees = torch.rad2deg(torch.tensor(np.pi/2, device=self.device)+ torch.arctan2(delta_z_m, distance_2d_m))
        
        result = azimuth_degrees.to(torch.float), zenith_degrees.to(torch.float)

        return result
    

    def calculate_distances_azimuth_zenith_b_to_a_wraparound(self, positions_b_m, positions_a_m, tiers, isd_m): 
        
        positions_b_m = torch.tensor(positions_b_m, device=self.device)
        positions_a_m = torch.tensor(positions_a_m, device=self.device)
        tiers = torch.tensor(tiers, device=self.device)
        isd_m = torch.tensor(isd_m, device=self.device)

        # Refer to ECC report 252 (SEAMCAT Handbook Edition 2 Approved 29 April 2016) - page 139

        # (x,y) is the UE position = positions_b_m
        # (a,b) is the BS position = positions_a_m - The first BS has to be centred at (0,0)m
        
        # The wrap around distance from a UE at (x,y) to a BS at (a,b) is the minimum of the following:
        # - Distance between (x,y) and (a,b);
        # - Distance between (x,y) and (a + 3D/sqrt3,b + 4D);
        # - Distance between (x,y) and (a - 3D/sqrt3,b - 4D);
        # - Distance between (x,y) and (a + 4.5D/sqrt3,b - 7D/2);
        # - Distance between (x,y) and (a - 4.5D/sqrt3,b + 7D/2);
        # - Distance between (x,y) and (a + 7.5D/sqrt3,b + D/ 2);
        # - Distance between (x,y) and (a - 7.5D/sqrt3,b - D/ 2),   
        distance_3d_partial_results = torch.ones((7, positions_b_m.size(0), positions_a_m.size(0)), dtype=torch.float, device=self.device)
        distance_2d_partial_results = torch.ones((7, positions_b_m.size(0), positions_a_m.size(0)), dtype=torch.float, device=self.device)
        azimuth_partial_results = torch.ones((7, positions_b_m.size(0), positions_a_m.size(0)), dtype=torch.float, device=self.device)
        zenith_partial_results = torch.ones((7, positions_b_m.size(0), positions_a_m.size(0)), dtype=torch.float, device=self.device)
        

        # Calculate the displacement of the cells in positions_a_m with respect to the reference site poistions

        # Distance between (x,y) and (a,b);
        distance_3d_partial_results[0,:,:] = torch.cdist(positions_b_m, positions_a_m)
        distance_2d_partial_results[0,:,:] = torch.cdist(positions_b_m[:,:2], positions_a_m[:,:2])
        azimuth_partial_results[0,:,:], \
            zenith_partial_results[0,:,:]  = self.calculate_azimuth_zenith_b_to_a_torch(positions_b_m,positions_a_m,distance_2d_partial_results[0,:,:])            


        # Distance between (x,y) and (a + 3D/sqrt3,b + 4D); 
        positions_a_aux_m = positions_a_m + torch.tensor([1.5*tiers*isd_m/np.sqrt(3), (1+1.5*tiers)*isd_m, 0], device = self.device) 
        
        distance_3d_partial_results[1,:,:] = torch.cdist(positions_b_m, positions_a_aux_m)
        distance_2d_partial_results[1,:,:] = torch.cdist(positions_b_m[:,:2], positions_a_aux_m[:,:2])
        azimuth_partial_results[1,:,:], zenith_partial_results[1,:,:] = \
            self.calculate_azimuth_zenith_b_to_a_torch(positions_b_m,positions_a_aux_m,distance_2d_partial_results[1,:,:])            


        # Distance between (x,y) and (a - 3D/sqrt3,b - 4D);
        positions_a_aux_m = positions_a_m - torch.tensor([1.5*tiers*isd_m/np.sqrt(3), (1+1.5*tiers)*isd_m, 0], device=self.device) 

        distance_3d_partial_results[2,:,:] = torch.cdist(positions_b_m, positions_a_aux_m)
        distance_2d_partial_results[2,:,:] = torch.cdist(positions_b_m[:,:2], positions_a_aux_m[:,:2])
        azimuth_partial_results[2,:,:], zenith_partial_results[2,:,:] = \
            self.calculate_azimuth_zenith_b_to_a_torch(positions_b_m,positions_a_aux_m,distance_2d_partial_results[2,:,:])


        # Distance between (x,y) and (a + 4.5D/sqrt3,b - 7D/ 2);
        positions_a_aux_m = positions_a_m + torch.tensor([(1.5+1.5*tiers)*isd_m/np.sqrt(3), -(0.5+1.5*tiers)*isd_m, 0], device=self.device) 
        distance_3d_partial_results[3,:,:] = torch.cdist(positions_b_m, positions_a_aux_m)
        distance_2d_partial_results[3,:,:] = torch.cdist(positions_b_m[:,:2], positions_a_aux_m[:,:2])
        azimuth_partial_results[3,:,:], zenith_partial_results[3,:,:] = \
            self.calculate_azimuth_zenith_b_to_a_torch(positions_b_m,positions_a_aux_m,distance_2d_partial_results[3,:,:]) 
             
            
        # Distance between (x,y) and (a - 4.5D/sqrt3,b + 7D/2);
        positions_a_aux_m = positions_a_m + torch.tensor([-(1.5+1.5*tiers)*isd_m/np.sqrt(3), (0.5+1.5*tiers)*isd_m, 0], device=self.device)
          
        distance_3d_partial_results[4,:,:] = torch.cdist(positions_b_m, positions_a_aux_m)
        distance_2d_partial_results[4,:,:] = torch.cdist(positions_b_m[:,:2], positions_a_aux_m[:,:2])
        azimuth_partial_results[4,:,:], zenith_partial_results[4,:,:] = \
            self.calculate_azimuth_zenith_b_to_a_torch(positions_b_m,positions_a_aux_m,distance_2d_partial_results[4,:,:])    
            
            
        # Distance between (x,y) and (a + 7.5D/sqrt3,b + D/ 2);
        positions_a_aux_m = positions_a_m + torch.tensor([(1.5+3*tiers)*isd_m/np.sqrt(3), isd_m/2, 0], device=self.device)

        distance_3d_partial_results[5,:,:] = torch.cdist(positions_b_m, positions_a_aux_m)   
        distance_2d_partial_results[5,:,:] = torch.cdist(positions_b_m[:,:2], positions_a_aux_m[:,:2])
        azimuth_partial_results[5,:,:], zenith_partial_results[5,:,:] = \
            self.calculate_azimuth_zenith_b_to_a_torch(positions_b_m,positions_a_aux_m,distance_2d_partial_results[5,:,:])    
            
    
        # Distance between (x,y) and (a - 7.5D/sqrt3,b - D/ 2),
        positions_a_aux_m = positions_a_m - torch.tensor([(1.5+3*tiers)*isd_m/np.sqrt(3), isd_m/2, 0], device = self.device)

        distance_3d_partial_results[6,:,:] = torch.cdist(positions_b_m, positions_a_aux_m)
        distance_2d_partial_results[6,:,:] = torch.cdist(positions_b_m[:,:2], positions_a_aux_m[:,:2])
        azimuth_partial_results[6,:,:], zenith_partial_results[6,:,:] = \
            self.calculate_azimuth_zenith_b_to_a_torch(positions_b_m,positions_a_aux_m,distance_2d_partial_results[6,:,:])                
            
            
        # Select the min in the 3 dimension
        index_3d_min = torch.argmin(distance_3d_partial_results, dim=0)
        index_2d_min = torch.argmin(distance_2d_partial_results, dim=0) 

        # Creating arrays of indices for gathering the results efficiently
        i_indices, j_indices = np.indices(index_3d_min.shape)

        # Efficiently gather results using the index_3d_min and index_2d_min
        distance_3d_results = distance_3d_partial_results[index_3d_min, i_indices, j_indices]
        distance_2d_results = distance_2d_partial_results[index_2d_min, i_indices, j_indices]
        azimuth_results = azimuth_partial_results[index_3d_min, i_indices, j_indices]
        zenith_results = zenith_partial_results[index_3d_min, i_indices, j_indices]

        # Converting to numpy 
        distance_3d_results = distance_3d_results.cpu().numpy()
        distance_2d_results = distance_2d_results.cpu().numpy()
        azimuth_results = azimuth_results.cpu().numpy()
        zenith_results = zenith_results.cpu().numpy()
        return distance_2d_results, distance_3d_results, azimuth_results, zenith_results 
        
    
    def calculate_azimuth_zenith_deg_from_GCS_to_LCS_3GPPComp_perCell(self, 
                                                                      α_mec_degree: float, 
                                                                      β_mec_degree: float,
                                                                      γ_mec_degree: float,
                                                                      azimuth_degree_arr_gcs: npt.NDArray[np.float32], 
                                                                      zenith_degree_arr_gcs: npt.NDArray[np.float32]
                                                                      ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Computes the azimuth and zenith angles in the Local Coordinate System (LCS) relative to a base station,
        given the angles in the Global Coordinate System (GCS) and mechanical tilts.
        
        Parameters:
            α_mec_degree (float): Mechanical bearing rotation angle (in degrees)
            β_mec_degree (float): Mechanical tilt rotation angle (in degrees)
            γ_mec_degree (float): Mechanical slant rotation angle (in degrees)
            azimuth_degree_arr_gcs (numpy.ndarray, dtype=float32): Array of azimuth angles in degrees (GCS)
            zenith_degree_arr_gcs (numpy.ndarray, dtype=float32): Array of zenith angles in degrees (GCS)
        
        Returns:
            tuple: (numpy.ndarray, numpy.ndarray)
                - Azimuth in degrees (LCS), dtype=float32
                - Zenith in degrees (LCS), dtype=float32
        """
        
        # degrees -> radians
        azimuth_rad_arr = np.deg2rad(azimuth_degree_arr_gcs.astype(np.float64))
        zenith_rad_arr = np.deg2rad(zenith_degree_arr_gcs.astype(np.float64))

        α = np.deg2rad(α_mec_degree)
        β = np.deg2rad(β_mec_degree)
        γ = np.deg2rad(γ_mec_degree)

        def Rz(a):
            return np.array([[np.cos(a), -np.sin(a), 0],
                             [np.sin(a),  np.cos(a), 0],
                             [0,          0,         1]], dtype=np.float64)

        def Ry(b):
            return np.array([[ np.cos(b), 0, np.sin(b)],
                             [ 0,         1, 0        ],
                             [-np.sin(b), 0, np.cos(b)]], dtype=np.float64)

        def Rx(c):
            return np.array([[1, 0,          0        ],
                             [0, np.cos(c), -np.sin(c)],
                             [0, np.sin(c),  np.cos(c)]], dtype=np.float64)

        # Convert to different reference system, 
        # to be aligned for 3GPP 38.901 computations
        β_eff = β - (np.pi / 2)

        # Forward per 38.901: R = Rz(α) Ry(β_eff) Rx(γ)
        # R_fwd = Rz(α) @ Ry(β_eff) @ Rx(γ)

        # Inverse per 38.901 Eq (7.1-3): R^{-1} = Rx(-γ) Ry(-β_eff) Rz(-α)
        # (equivalently: R_fwd.T)
        R_inv = Rx(-γ) @ Ry(-β_eff) @ Rz(-α)

        # unit vector in GCS (Eq 7.1-6)
        # Compute the normalized Cartesian position vector in GCS
        ρ_gcs_arr = np.array([np.sin(zenith_rad_arr) * np.cos(azimuth_rad_arr),
                              np.sin(zenith_rad_arr) * np.sin(azimuth_rad_arr),
                              np.cos(zenith_rad_arr)])

      
        # Compute the LCS angles with respect to the analyzed base station
        # To convert from GCS to LCS, we use the inverse of the rotation matrix R^{-1}(x) = R(-x)
        # From 3GPP 38.901 eqs. 7.1-7 and 7.1-8
        lcs_rad_zenith = np.arccos(np.array([0, 0, 1]).T  @ R_inv @ ρ_gcs_arr)
        lcs_rad_azimuth = np.angle(np.array([1, 1j, 0]).T @ R_inv @ ρ_gcs_arr)

        return np.rad2deg(lcs_rad_azimuth).astype(np.float32), np.rad2deg(lcs_rad_zenith).astype(np.float32)
    

    def calculate_azimuth_zenith_deg_from_LCS_to_GCS_3GPPComp_perCell(self, 
                                                                      α_mec_degree: float,
                                                                      β_mec_degree: float,
                                                                      γ_mec_degree: float,
                                                                      azimuth_degree_arr_lcs: npt.NDArray[np.float32],   # φ'  (LCS)
                                                                      zenith_degree_arr_lcs: npt.NDArray[np.float32],    # θ'  (LCS)
                                                                      ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """
        Computes the azimuth and zenith angles in the Global Coordinate System (GCS),
        given the angles in the Local Coordinate System (LCS) and the mechanical
        rotation angles (bearing, tilt, slant) defined by 3GPP-compliant conventions.

        Parameters
        ----------
        α_mec_degree : float
            Mechanical bearing rotation angle (degrees).
        β_mec_degree : float
            Mechanical tilt rotation angle (degrees).
        γ_mec_degree : float
            Mechanical slant rotation angle (degrees).
        azimuth_degree_arr_lcs : numpy.ndarray (dtype=float32)
            Array of azimuth angles in degrees in the LCS (φ').
        zenith_degree_arr_lcs : numpy.ndarray (dtype=float32)
            Array of zenith angles in degrees in the LCS (θ').

        Returns:
            tuple: (numpy.ndarray, numpy.ndarray)
                - Azimuth in degrees (GCS), dtype=float32
                - Zenith in degrees (GCS), dtype=float32
        """

        azimuth_rad_arr = np.deg2rad(azimuth_degree_arr_lcs.astype(np.float64)) # φ'
        zenith_rad_arr = np.deg2rad(zenith_degree_arr_lcs.astype(np.float64))   # θ'


        α = np.deg2rad(α_mec_degree)
        β = np.deg2rad(β_mec_degree)
        γ = np.deg2rad(γ_mec_degree)

        def Rz(a):
            return np.array([[np.cos(a), -np.sin(a), 0],
                             [np.sin(a),  np.cos(a), 0],
                             [0,          0,         1]], dtype=np.float64)

        def Ry(b):
            return np.array([[ np.cos(b), 0, np.sin(b)],
                             [ 0,         1, 0        ],
                             [-np.sin(b), 0, np.cos(b)]], dtype=np.float64)

        def Rx(c):
            return np.array([[1, 0,          0        ],
                             [0, np.cos(c), -np.sin(c)],
                             [0, np.sin(c),  np.cos(c)]], dtype=np.float64)

        # same convention you used
        β_eff = β - (np.pi / 2)

        # Forward per 38.901 (7.1-2): R = Rz(α) Ry(β_eff) Rx(γ)
        R_fwd = Rz(α) @ Ry(β_eff) @ Rx(γ)

        # unit vector in LCS (same as 7.1-6 but primed angles) :contentReference[oaicite:2]{index=2}
        ρ_lcs_arr = np.array([
            np.sin(zenith_rad_arr) * np.cos(azimuth_rad_arr),
            np.sin(zenith_rad_arr) * np.sin(azimuth_rad_arr),
            np.cos(zenith_rad_arr)
        ])  # shape (3, N)

        # rotate LCS -> GCS
        ρ_gcs_arr = R_fwd @ ρ_lcs_arr  # shape (3, N)

        # recover GCS angles (same method you used, but without R_inv)
        gcs_rad_zenith  = np.arccos(np.array([0, 0, 1], dtype=np.float64).T @ ρ_gcs_arr)
        gcs_rad_azimuth = np.angle(np.array([1, 1j, 0], dtype=np.complex128).T @ ρ_gcs_arr)

        return np.rad2deg(gcs_rad_azimuth).astype(np.float32), np.rad2deg(gcs_rad_zenith).astype(np.float32)

    
class Distance_Angles_ue_to_cell(Distance_Angles):
    pass

class Distance_Angles_ueAnt_to_cellAnt(Distance_Angles):
    pass