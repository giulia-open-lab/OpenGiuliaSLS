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

import time
import gc
import logging
import os
from math import sqrt, sin, cos, atan2
from typing import Union

import numpy as np
import tensorflow as tf
import torch
import trimesh
from pyproj import Proj, transform, Transformer
from shapely.geometry import Point, Polygon

from giulia.logger import verbose, debug, info
import pandas as pd
import inspect

from giulia.outputs.saveable import saveables


def int_nan():
    return -9999


def calculate_area_square(length):  
    """  
    Function to calculate the area of a square  
    :param length: length of the square  
    :return: area of the square  
    """  
    return length * length


def directory_exists(path):

   # Check whether the specified path exists or not
   isExist = os.path.exists(path)
   
   if not isExist:
      # Create a new directory because it does not exist
      os.makedirs(path)
      print("The new directory, " + path + ", is created!")


def round_to_zero(num):
   if np.abs(num) < 1e-10:
       num = 0.0
            
   return num 


def magnitude(x, y, z):
    return sqrt(x * x + y * y + z * z)


def cartesian_to_spherical(x, y, z):
    radius = magnitude(x, y, z)
    theta = atan2(sqrt(x * x + y * y), z)
    phi = atan2(y, x)
    
    return (radius, theta, phi) 


def spherical_to_cartesian(radius, theta, phi):
    x = radius * cos(phi) * sin(theta)
    y = radius * sin(phi) * sin(theta)
    z = radius * cos(theta)
    return (x, y, z)


def randinunithex(myrandom, isd):
    #See https://www.redblobgames.com/grids/hexagons/, flat top orientation, for definition of width and size
    #In our case, the width is the ISD
    #Note that in this deployment method the size = 1, and that in a hexagonal grid, the size is equal to ISD / sqrt(3)
    scaling = isd/np.sqrt(3) 
    vectors = [(-1.0*scaling,0),(.5*scaling,sqrt(3.)/2.*scaling),(.5*scaling,-sqrt(3.)/2.*scaling)]
    x = myrandom.randrange(3);
    (v1,v2) = (vectors[x], vectors[(x+1)%3])
    (x,y) = (myrandom.random(),myrandom.random())
    return (x*v1[0]+y*v2[0],x*v1[1]+y*v2[1])


def isBetween(start, end, mid) :   
    end = (end - start) 
    if end < 0:
       end += 360 

    mid = (mid - start) 
    if mid < 0:
       mid += 360 

    return mid < end


def angle_range_0_180(angle_deg):
    if isinstance(angle_deg,(int,float)):
        if angle_deg > 180:
            angle_deg = angle_deg - 360
            
        if angle_deg < -180:
            angle_deg = angle_deg + 360   
        
    else:
        mask =  angle_deg > 180
        angle_deg[mask] = angle_deg[mask] - 360
        
        mask =  angle_deg < -180
        angle_deg[mask] = angle_deg[mask] + 360    
            
    return angle_deg


def angle_range_0_pi(angle_rad):
    if isinstance(angle_rad,(int,float)):
        if angle_rad > np.pi:
            angle_rad = angle_rad - 2*np.pi
            
        if angle_rad < -np.pi:
            angle_rad = angle_rad + 2*np.pi   
        
    else:
        mask =  angle_rad > np.pi
        angle_rad[mask] = angle_rad[mask] - 2*np.pi
        
        mask =  angle_rad < -np.pi
        angle_rad[mask] = angle_rad[mask] + 2*np.pi    
            
    return angle_rad


def ms_to_kmh(velocity_ms):
    # Conversion factor from m/s to km/h
    conversion_factor = 3.6
    
    # Convert velocity
    velocity_kmh = velocity_ms * conversion_factor
    
    return velocity_kmh


def kmh_to_ms(velocity_kmh):
    # Conversion factor from km/h to m/s
    conversion_factor = 1 / 3.6
    
    # Convert velocity
    velocity_ms = velocity_kmh * conversion_factor
    
    return velocity_ms


def mW_to_dBm(value_mW):
    return 10 * np.log10(value_mW)


def mW_to_dBm_torch(value_mW):
    result = 10 * torch.log10(value_mW)
    return result


def dBm_to_mW(value_dBm):
    return np.power(10,value_dBm/10)


def dBm_to_mW_torch(value_dBm):
    result = torch.pow(10, value_dBm/10)
    return result


def isInsideHexagon(pos, center, h):
    
    # pos = position of point to test
    # center = position of centre of hexagon
    # h = 1/2 of the side of the hexagone  
    
    v = 2*h*np.cos(30*np.pi/180)

    q2x = np.abs(pos[0] - center[0])         # transform the test point locally and to quadrant 2
    q2y = np.abs(pos[1] - center[1])         # transform the test point locally and to quadrant 2
    if (q2x > 2*h or q2y > v): 
        return False           # bounding test (since q2 is in quadrant 2 only 2 tests are needed)
    
    return 2 * v * h - v * q2x - h * q2y >= 0   # finally the dot product can be reduced to this due to the hexagone symmetry


def isInsideHexagon_pointy_top(pos, center, v):
    
    # pos = position of point to test
    # center = position of centre of hexagon
    # v = 1/2 of the side of the hexagone  
    
    h = 2*v*np.cos(30*np.pi/180)

    q2x = np.abs(pos[0] - center[0])         # transform the test point locally and to quadrant 2
    q2y = np.abs(pos[1] - center[1])         # transform the test point locally and to quadrant 2
    if (q2x > h or q2y > v*2): 
        return False           # bounding test (since q2 is in quadrant 2 only 2 tests are needed)
    
    return 2 * v * h - v * q2x - h * q2y >= 0   # finally the dot product can be reduced to this due to the hexagone symmetry


def axial_to_oddq(q, r):
    col = q
    row = r + (q - (np.bitwise_and(q,1))) / 2
    return col, int(row)


def oddq_to_axial(col, row):
    q = col
    r = row - (col - (np.bitwise_and(col,1))) / 2
    s = -q-r
    return q, int(r), int(s)
    

def cube_direction_vectors(i):
    
    cube_direction_vectors = [(+1, 0, -1), 
                              (+1, -1, 0), 
                              (0, -1, +1), 
                              (-1, 0, +1), 
                              (-1, +1, 0), 
                              (0, +1, -1)] 
    
    return cube_direction_vectors[i]


def cube_neighbor(q, r, s, i):
    
    cube_direction = cube_direction_vectors(i)
    
    return q + cube_direction[0], r + cube_direction[1], s + cube_direction[2]


def create_hexagonal_layout(plot, hex_layout_centre_m, hex_layout_tiers, isd_m):
    
   # Number of sites
   number_of_sites = 1 + 3 * hex_layout_tiers * (hex_layout_tiers+1)
   
   # Initialise site positions
   site_positions_m = np.full((number_of_sites,2), np.nan, dtype=np.single)
    
   # First site is centered in the provided position
   site_positions_m[0] = hex_layout_centre_m[:2]

   #Create a hexagonal grid to find the location of the rest of the neighbours       
   ##Construct grid. Using “even-q” vertical layout shoves even columns down in https://www.redblobgames.com/grids/hexagons/
   
   # Number of samples in grid
   samples = int(20)
   
   # Initialise grid
   gridx = np. zeros((samples, samples))
   gridy = np. zeros((samples, samples))
    
   # Finding the indeces of the center grid point in the grid
   zero_x_index = int(samples/2)
   if zero_x_index%2 != 0:
       zero_x_index -= int(1)
   zero_y_index = int(samples/2)
   
   # Calculate the distance from the hex in the center to those in the borders of the grid
   offset_x_m = zero_x_index * np.sqrt(3)/2 * isd_m 
   offset_y_m = zero_y_index * isd_m

   # Calculate the position of each hexagone in the grid
   for j in range(0,samples) : # Loop over the y-axis
       for i in range(0,samples) : # Loop over the x-axis
           # Calculate x-position of the hexagon. In the x-axis hexagones are separated np.sqrt(3)/2 * isd_m meters
           gridx[i,j] = - offset_x_m + np.sqrt(3)/2 * isd_m * i # Calculate x-position
           
           # Calculate y-position of the hexagon. In the y-axis hexagones are separated isd_m meters. Note that there is an offset for the hexagones in the odd columns of the x-axis
           if (i%2==0):
               gridy[i,j] = + offset_y_m - j * isd_m # Calculate y-position
           else :
               gridy[i,j] = + offset_y_m - isd_m/2  - j * isd_m # Calculate y-position (Note that for the odd positions in the x-axis, there is an offset for the y-locations)
               

   #Traversing all tiers one by one in a spiral pattern, we can find the positions of the sites of interest
   #Look at https://www.redblobgames.com/grids/hexagons/ - Spiral rings
   index = 1
   #for each 1 ≤ tier ≤ hex_layout_tiers: 
   for tier in range(1,hex_layout_tiers+1): # Loop over the tiers
       
       # Find the cube coordiantes of the center grid point 
       zero_q, zero_r, zero_s = oddq_to_axial(zero_x_index, zero_y_index) 
       
       # Set the movement to a hexagone of the next tier 
       cube_direction = cube_direction_vectors(4)
       
       # Find the cube coordiantes of the new hexagone in next tier according to the previous movement
       q, r, s = (zero_q + cube_direction[0] * tier, zero_r + cube_direction[1] * tier, zero_s + cube_direction[2] * tier)
       
       # Find the offset coordinates of the new hexagon
       col, row = axial_to_oddq(q, r)
       
       for i in range(0,6): # Loop over all sites in a tier using this and the next for-loop
           for j in range(0,tier):  
               
               # Find the location of the site in this hexagone by using the grid and the offset coordinates
               site_positions_m[index] = (gridx[col,row], gridy[col,row])
               index += 1
               
               #Find the cube coordinates of the next neighbouring site in the same tier using
               q, r, s = cube_neighbor(q, r, s, i)
               
               # Find the offset coordinates of the new hexagon
               col, row = axial_to_oddq(q, r)
 
   return site_positions_m


def create_sector(center, radius, start_angle, end_angle, num_points=100):

    # Generate the vertices of a sector using the center and size
    
    #Calculate angles
    angles = np.linspace(start_angle, end_angle, num_points)

    # Calculate the coordinates of the two points where the arc meets the circle
    arc_points = [(center[0] + radius * np.cos(angle),
                   center[1] + radius * np.sin(angle)) for angle in angles]

    # Add the center point to close the sector
    vertices = [center] + arc_points + [center]
    
    # Create sector
    sector = Polygon(vertices)

    return sector


def generate_hexagon(center, size):
    
    # Generate the vertices of a hexagone using the center and size
    
    #Calculate angles
    angles = np.linspace(0, 2 * np.pi, 7)
    
    # Calculate the coordinates of the six points where the sides of the hexagon meet
    x_hexagon = center[0] + size * np.cos(angles)
    y_hexagon = center[1] + size * np.sin(angles)
    
    # Create hexagon
    hexagon = Polygon(zip(x_hexagon, y_hexagon))
    
    return hexagon


def generate_random_points_within_polygon(hexagon, num_points):
    
    #Get polygon bounds
    min_x, min_y, max_x, max_y = hexagon.bounds
    points = []

    while len(points) < num_points:
        # Generate random point within bounding box
        random_point = Point(np.random.uniform(min_x, max_x), np.random.uniform(min_y, max_y))

        # Check if the point is within the hexagon
        if hexagon.contains(random_point):
            points.append(random_point)

    return points


def order_i(m, M_H):
    return (m - 1) % M_H  # Horizontal index


def order_j(m, M_H):
    return (m - 1) // M_H  # Vertical index 


def translate_position_to_grid_point(ue_positions_m, ue_scenario_reference_m, scenario_resolution_m,
                                               shadowing_map_reference_m=None):

    # When this function is called in ue_deployments.py to compute ue_deployment.ue_grid_position for plotting,
    # ue_scenario_reference_m is ue_playground_lower_left_corner_m
    # shadowing_map_reference_m is None
    if shadowing_map_reference_m is None:
        shadowing_map_reference_m = [0, 0]

    # When this function is called in shadowing_gains.py to compute grid point indices to read the shadowing map,
    # ue_scenario_reference_m is ue_scenario_center_m
    # shadowing_map_reference_m is shadowing_map_lower_left_corner_m

    # Adjust UE 2D positions to be relative to the reference coordinates
    adjusted_ue_positions_m = ue_positions_m[:, :2] - ue_scenario_reference_m - shadowing_map_reference_m

    grid_point_index_x = (np.floor((adjusted_ue_positions_m[:, 0]) / scenario_resolution_m)).astype(int)
    grid_point_index_y = (np.floor((adjusted_ue_positions_m[:, 1]) / scenario_resolution_m)).astype(int)

    return grid_point_index_x, grid_point_index_y


def latlon_to_utm(latitudes, longitudes):
    """
    Convert arrays of latitude and longitude to UTM easting and northing in a batch.

    Args:
        latitudes: Array of latitudes in decimal degrees.
        longitudes: Array of longitudes in decimal degrees.

    Returns:
    - eastings, northings: Arrays of UTM coordinates.
    """
    # Create a transformer that converts from geographic (WGS84) to UTM coordinates
    # Use always the same UTM zone for simplicity, for UK, zone can be 30N
    #transformer = Transformer.from_crs("epsg:4326", "epsg:32630", always_xy=True)
    transformer = Transformer.from_crs("epsg:4326", "epsg:27700", always_xy=True)
    # Transform the latitudes and longitudes to easting and northing
    eastings, northings = transformer.transform(longitudes, latitudes)

    return eastings, northings


def geodetic_to_cartesian(lat, lon) -> tuple:
    """
    Convert geodetic coordinates (latitude and longitude) to Cartesian coordinates (x, y, z).

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.

    Returns:
        x, y: Cartesian coordinates.
    """

    # Define the geographic coordinate system (WGS84) and the target Cartesian system
    geographic = Proj(proj='latlong', datum='WGS84')
    cartesian = Proj(proj='geocent', datum='WGS84')

    x, y, _, _ = transform(geographic, cartesian, lon, lat)
    return x, y


def cartesian_to_geodetic(x, y) -> tuple:
    """
    Convert Cartesian coordinates (x, y) to geodetic coordinates (latitude, longitude).

    Returns:
        lat, lon: Latitude and longitude in decimal degrees.
    """
    cartesian = Proj(proj='geocent', datum='WGS84')
    geographic = Proj(proj='latlong', datum='WGS84')

    lon, lat, _, _ = transform(cartesian, geographic, x, y)
    return lat, lon


# Function to compute the signed distance of a point to the mesh
def compute_signed_distance(point: Union[list, np.ndarray], mesh: trimesh.Trimesh) -> float:
    """
    Compute the signed distance from a point to the nearest surface of a mesh.

    Args:
        point: The coordinates of the point [x, y, z].
        mesh: The mesh object of the building.

    Returns:
        The signed distance (positive if inside, negative if outside).
    """
    distance = mesh.nearest.signed_distance(np.array([point]))
    return distance[0]


# Function to check if a point is inside or outside any building
def check_point_in_buildings(point: Union[list, np.ndarray], ply_folder: str, threshold: float = 0.0, verbose: bool = True) -> bool:
    """
    Check if a point is inside or outside any building based on PLY meshes.

    The function loops over all PLY files in the folder, computes the signed
    distance between the point and the mesh surface, and determines whether
    the point is inside or outside the building.

    Args:
        point: The coordinates of the point [x, y, z].
        ply_folder: The folder containing PLY mesh files for buildings.
        threshold: A tolerance value for determining "inside" status.
                         (default is 0.0, can be adjusted for precision)
        verbose: Whether to print details of the process (default is True).

    Returns:
        True if the point is inside any building, False if outside all.
    """
    # Iterate over all PLY files in the folder
    for ply_file in os.listdir(ply_folder):
        if ply_file.endswith('.ply'):
            # Exclude 'Plane.ply' (the ground plane)
            if ply_file == 'Plane.ply':
                continue  # Skip checking this mesh
            # Load the mesh
            mesh: trimesh.Trimesh = trimesh.load_mesh(os.path.join(ply_folder, ply_file))

            # Compute signed distance for this mesh
            signed_distance = compute_signed_distance(point, mesh)

            # If the signed distance is positive, the point is inside the building
            if signed_distance > threshold:
                if verbose:
                    print(f"Point {point} is inside the building {ply_file} (Distance: {signed_distance})")
                return True  # The point is inside the building

    # If no building contains the point, it is outside
    if verbose:
        print(f"Point {point} is outside all buildings.")
    return False  # The point is outside all buildings


def free_memory(free_cpu=True, free_GPU=True):
    """Frees CPU and GPU memory (PyTorch, TensorFlow)."""
    
    if free_cpu:
        gc.collect()
        
    # Clean GPU memory
    if free_GPU:
        
        # Clean PyTorch GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            
        # Clean TensorFlow memory
        try:
            # Save the current TensorFlow logger level
            old_level = tf.get_logger().level
            # Set the logger level to ERROR to suppress warnings
            tf.get_logger().setLevel(logging.ERROR)
            
            tf.keras.backend.clear_session()
            
            # Restore the original logger level
            tf.get_logger().setLevel(old_level)
        except Exception as e:
            print(f"Error clearing TensorFlow memory: {e}")

    return True


def log_elapsed_time(name: str, t_start: float, debug_prints: bool = False):
    """
    Log elapsed time of execution if log level is verbose.
    If log level is debug, and ``verbose_prints`` is ``False`` (by default), only ``name`` will be printed.
    If ``verbose_prints`` is ``True``, the elapsed time will be logged, but cut to seconds.
    Args:
        name: The name of the function.
        t_start: The start time of the execution.
        debug_prints: Whether to print the elapsed time in debug mode.
    """
    verbose(f"{name}, elapsed time = {time.perf_counter() - t_start:0.4f}")
    debug(f"{name} run, elapsed time = {time.perf_counter() - t_start:0.2f} sec" if debug_prints else name, exclusive=True)


def log_calculations_time(name: str, t_start: float):
    """
    Log elapsed time of execution of a calculation if log level is verbose, otherwise just prints ``name``.
    Args:
        name: The name of the function.
        t_start: The start time of the execution.
    """
    log_elapsed_time(f"{name} calculations", t_start)


def log_saveable_available_results() -> None:
    """
    Log the available results that can be dynamically selected to be saved.
    """
    info(f"There are {len(saveables)} saveable objects registered.")
    info("Structure:")
    for saveable in saveables:
        info(f"- {saveable.name}:")
        for variable in saveable.variables_list():
            info("  - %s" % variable)
        
        
class TrackedDataFrame(pd.DataFrame):
    """
    TrackedDataFrame: A subclass of pandas.DataFrame that tracks access to a specified column (set via key_to_track).
    It logs the number of accesses and records the full name (class and method) of each caller, 
    aiding in debugging and usage analysis.
    
    Finally information is with self.attribute_access_count and self.attribute_access_points
    """
    _metadata = ['key_to_track', 'attribute_access_count', 'attribute_access_points']
    
    def __init__(self, *args, key_to_track='antenna_pattern_model', **kwargs):
        super().__init__(*args, **kwargs)
        self.key_to_track = key_to_track
        self.attribute_access_count = 0
        self.attribute_access_points = []

    def __getitem__(self, key):
        # If the accessed key matches the one we want to track, log the access
        if key == self.key_to_track:
            stack = inspect.stack()
            frame = stack[1]
            func_name = frame.function
            # If 'self' is in the local variables, assume it's a method call and capture class name too
            if 'self' in frame.frame.f_locals:
                instance = frame.frame.f_locals['self']
                full_name = f"{instance.__class__.__name__}.{func_name}"
            else:
                full_name = func_name
            self.attribute_access_count += 1
            self.attribute_access_points.append(full_name)
        return super().__getitem__(key)
    
    def reset_attribute_counter(self):
        self.attribute_access_count = 0
        self.attribute_access_points = []