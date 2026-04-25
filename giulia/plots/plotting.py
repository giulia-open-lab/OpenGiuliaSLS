# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:01:27 2023

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import RegularPolygon
from matplotlib.ticker import FuncFormatter

from giulia.tools import tools

# Configure Giulia into path
this_dir = os.path.dirname(__file__)

def radians_to_pi(x, pos):
    # Define a custom tick formatter to display radians using π
    return f'{x/np.pi:.1f}π'  


def degrees_formatter(x, pos):
    # Define a custom tick formatter to display degrees
    return f'{x:.0f}°'  


def db_formatter(x, pos):
    # Define a custom tick formatter to display dB values    
    return f'{x:.1f} dB' 


def custom_color_map_sudden(number_of_colors):
    # Generate random colors
    colors = np.random.rand(number_of_colors, 3)

    # Sort colors by luminance (perceptual lightness)
    luminance = np.sqrt(0.299 * colors[:, 0]**2 + 0.587 * colors[:, 1]**2 + 0.114 * colors[:, 2]**2)
    sorted_indices = np.argsort(luminance)
    colors = colors[sorted_indices]

    # Create a list of positions for the color changes
    positions = np.linspace(0, 1, len(colors))

    # Create a colormap with sudden changes among neighboring values
    colormap = LinearSegmentedColormap.from_list('custom_colormap', list(zip(positions, colors)), N=len(colors))
    
    return colormap


def plot_scenario(project_name, isd_m, site_names, site_positions_m):
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    number_of_sites = site_positions_m.shape[0]
    labels = [f"site{i}" for i in range(number_of_sites)]

    # --- Assign colors and tech-based labels ---
    color_vector = []
    label_vector = []
    used_labels = set()

    for name in site_names:
        if "B5_" in name:
            label = "5G BSs"
            color = "red"
        elif "B6_" in name:
            label = "6G BSs"
            color = "blue"
        elif "B_" in name:
            label = "4G BSs"
            color = "green"
        else:
            label = "BSs"
            color = "black"

        # Only show label once in legend
        if label and label not in used_labels:
            used_labels.add(label)
            label_vector.append(label)
        else:
            label_vector.append(None)

        color_vector.append(color)


    # --- Draw background hexagons ---
    for i, (x, y) in enumerate(site_positions_m[:, :2]):
        color = color_vector[i].lower()
        if not np.isnan(isd_m) and i < 19:
            hex = RegularPolygon(
                (x, y), numVertices=6, radius=isd_m / np.sqrt(3),
                orientation=np.radians(30),
                facecolor=color, alpha=0.2
            )
            ax.add_patch(hex)

        # Label text
        ax.text(x, y - 50.5, labels[i], ha='center', va='center', size=8)

    # --- Plot BSs with scatter and legend ---
    for i in range(number_of_sites):
        ax.scatter(
            site_positions_m[i, 0], site_positions_m[i, 1],
            marker='^', color=color_vector[i].lower(),
            label=label_vector[i]
        )

    # Set labels, legend, and grid
    ax.set_xlabel("x position [m]", fontsize=14)
    ax.set_ylabel("y position [m]", fontsize=14)
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid()
    plt.tight_layout()

    # Save
    folder_name = "results_" + project_name + "/"
    tools.directory_exists(folder_name)
    plt.savefig(folder_name + 'layout.pdf')
    plt.savefig(folder_name + 'layouts.png', dpi=600)

    plt.show()
    plt.close()
    

def plot_ue_locations(project_name, isd_m, site_names, site_positions_m, ue_position_m, hot_spot_position_m):
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    number_of_sites = len(site_positions_m)
    labels = [f"s{i}" for i in range(number_of_sites)]

    # --- Assign colors and BS type labels based on site_name ---
    color_vector = []
    label_vector = []
    used_labels = set()

    for name in site_names:
        if "B5_" in name:
            label = "5G BSs"
            color = "red"
        elif "B6_" in name:
            label = "6G BSs"
            color = "blue"
        elif "B_" in name:
            label = "4G BSs"
            color = "green"
        else:
            label = None
            color = "gray"

        # Add label only once to avoid repeated legend entries
        if label and label not in used_labels:
            used_labels.add(label)
            label_vector.append(label)
        else:
            label_vector.append(None)

        color_vector.append(color)

    # --- Add colored hexagons around each site ---
    for i, (x, y) in enumerate(site_positions_m[:, :2]):
        if not np.isnan(isd_m) and i < 19:
            hex = RegularPolygon(
                (x, y), numVertices=6, radius=isd_m / np.sqrt(3),
                orientation=np.radians(30),
                facecolor=color_vector[i].lower(), alpha=0.2, edgecolor=color_vector[i].lower()
            )
            ax.add_patch(hex)
        
        # Label the site with its ID
        ax.text(x, y - 50.5, labels[i], ha='center', va='center', size=8)

    # --- Scatter BSs with distinct color and label ---
    for i in range(number_of_sites):
        ax.scatter(
            site_positions_m[i, 0], site_positions_m[i, 1],
            marker='^', color=color_vector[i].lower(), alpha=0.5, label=label_vector[i]
        )

    # --- Add UE locations ---
    if ue_position_m is not None and len(ue_position_m) > 0:
        ax.scatter(ue_position_m[:, 0], ue_position_m[:, 1],
                   c="brown", marker='.', alpha=0.2, label="UE")

    # --- Add hotspot locations ---
    if hot_spot_position_m is not None and len(hot_spot_position_m) > 0:
        ax.scatter(hot_spot_position_m[:, 0], hot_spot_position_m[:, 1],
                   c="black", marker='2', alpha=0.5, label="hotspot")

    # --- Axis formatting ---
    ax.set_xlabel("x position [m]", fontsize=14)
    ax.set_ylabel("y position [m]", fontsize=14)
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid()

    plt.tight_layout()

    # --- Save the figure ---
    folder_name = "results_" + project_name + "/"
    tools.directory_exists(folder_name)
    plt.savefig(folder_name + 'layout_plus_ues.pdf')
    plt.savefig(folder_name + 'layout_plus_ues.png', dpi=600)

    plt.show()
    plt.close()


def plot_bs_ue_locations_plain(
    project_name,
    bs_position_m,
    ue_position_m,
    *,
    xlim=None,                 # e.g. (-500, 500)
    ylim=None,                 # e.g. (-500, 500)
    pad_frac=0.05,             # padding as fraction of data span if limits not given
    square=True,               # keep x/y spans equal
    figsize=(8, 8),
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')

    # --- Add BS locations ---
    if bs_position_m is not None and len(bs_position_m) > 0:
        bs = np.asarray(bs_position_m)
        ax.scatter(bs[:, 0], bs[:, 1], c="blue", marker='x', alpha=0.2, label="BS")
    else:
        bs = np.empty((0, 2))

    # --- Add UE locations ---
    if ue_position_m is not None and len(ue_position_m) > 0:
        ue = np.asarray(ue_position_m)
        ax.scatter(ue[:, 0], ue[:, 1], c="red", marker='.', alpha=0.2, label="UE")
    else:
        ue = np.empty((0, 2))

    # --- Axis limits ---
    if xlim is None or ylim is None:
        pts = np.vstack([bs, ue]) if (bs.size or ue.size) else np.array([[0, 0]])
        xmin, xmax = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
        ymin, ymax = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))

        # Handle the degenerate case where all points align
        if xmin == xmax:
            xmin -= 1.0
            xmax += 1.0
        if ymin == ymax:
            ymin -= 1.0
            ymax += 1.0

        # padding as a fraction of span
        dx, dy = xmax - xmin, ymax - ymin
        xmin -= dx * pad_frac
        xmax += dx * pad_frac
        ymin -= dy * pad_frac
        ymax += dy * pad_frac

        if square:
            # Make x/y spans equal around the center
            cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
            half = max(xmax - xmin, ymax - ymin) / 2.0
            xlim = (cx - half, cx + half)
            ylim = (cy - half, cy + half)
        else:
            xlim = (xmin, xmax)
            ylim = (ymin, ymax)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # --- Axis formatting ---
    ax.set_xlabel("x position [m]", fontsize=14)
    ax.set_ylabel("y position [m]", fontsize=14)
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid()
    plt.tight_layout()

    # --- Save the figure ---
    folder_name = f"results_{project_name}/"
    tools.directory_exists(folder_name)
    plt.savefig(folder_name + 'bs_ue_positions.pdf')
    plt.savefig(folder_name + 'bs_ue_positions.png', dpi=600)

    plt.show()
    plt.close()


def plot_antenna_arrays(project_name, node_type, antenna_to_node_mapping, antenna_element_GCS_position_m):
    
    # Create the figure and axis
    fig = plt.figure()
    
    color_vec = ['r', 'g', 'k']    
    
    # Plot
    ax = fig.add_subplot(projection='3d')
    ax.scatter(antenna_element_GCS_position_m[:,0], antenna_element_GCS_position_m[:,1], antenna_element_GCS_position_m[:,2], marker='o', color='r')
    
    # Set labels, legend, and grid
    plt.title(node_type + " antenna array")    
    
    ax.set_xlabel('x-location [m]')
    ax.set_ylabel('y-location [m]')
    ax.set_zlabel('z-location [m]')  

    # Improve layout 
    plt.tight_layout()
     
    # Save
    folder_name = "results_" + project_name + "/"
    tools.directory_exists(folder_name)
    plt.savefig(folder_name + node_type + " antenna array" + '.pdf')
    plt.savefig(folder_name + node_type + " antenna array" + '.png', dpi=600)          
    
    # Show 
    plt.show()
    
    # Close the current plot to clear the figure
    plt.close()  


def plot_luts(project_name, modulation_and_coding_schemes, bler_per_sinr_mcs):
    
    # Create the figure and axis
    
    fig, ax = plt.subplots(figsize=(12,8))
    
    # QPSK_ 0.11 0.14 0.18 0.24 0.30 0.37 0.43 0.50 0.58 0.65
    # 16QAM_ 0.33 0.37 0.41 0.47 0.54 0.59 0.64
    # 64QAM_ 0.42 0.46 0.50 0.55 0.60 0.65 0.69 0.74 0.79 0.85 0.88 0.92
    # 256QAM_ 0.69 0.78 0.86 0.93
        
    line_style = ['-k.', '-k,', '-ko', '-kv', '-k^', '-k<', '-k>', '-k1', '-k2', '-k3',
                  '--b.', '--b,', '--bo', '--bv', '--b^', '--b<', '--b>',
                  '-.r.', '-.r,', '-.ro', '-.rv', '-.r^', '-.r<', '-.r>', '-.r1', '-.r2', '-.r3', '-.r4', '-.rs',
                  ':g.', ':g,', ':go', ':gv']
        
    # Plot data 
    for i in range(0,np.size(bler_per_sinr_mcs,1)):
         
        # Plot the CDF
        plt.plot(np.arange(-10, 30, 0.1), bler_per_sinr_mcs[:,i], line_style[i], markevery=0.001, label=modulation_and_coding_schemes[i])
        
    # Set labels, legend, and grid
    ax.set_xlabel("SINR")
    ax.set_ylabel("BLER")
    ax.set_yticks(np.arange(0, 1.01, step=0.1)) 
    ax.legend(ncol=7, bbox_to_anchor=(0.5,-0.5), loc='lower center', edgecolor='w')
    ax.grid()
    
    # Improve layout 
    plt.tight_layout()
    
    # Save
    folder_name = "results_" + project_name + "/"
    tools.directory_exists(folder_name)
    file_name = "bler_versus_sinr_lut"
    plt.savefig(folder_name + file_name + '.pdf')
    plt.savefig(folder_name + file_name + '.png', dpi=600)
    
    # Show  
    plt.show() 
    
    # Close the current plot to clear the figure
    plt.close()  
    

# --- shared helpers ----------------------------------------------------------
def _clean_series(x):
    if isinstance(x, np.ndarray):
        return x[~np.isnan(x) & (x != -np.inf)]
    # list-like
    return np.asarray([v for v in x if not (math.isnan(v) or (math.isinf(v) and v < 0))], dtype=float)


def _stats_df(cleaned, legends, metric_label):
    return pd.DataFrame({
        "Metric": metric_label,
        "Series": legends,
        "N":      [len(s) for s in cleaned],
        "Mean":   [np.nanmean(s) if len(s) else np.nan for s in cleaned],
        "Std":    [np.nanstd(s, ddof=1) if len(s) > 1 else np.nan for s in cleaned],
        "Median": [np.nanmedian(s) if len(s) else np.nan for s in cleaned],
        "P5":     [np.nanpercentile(s, 5) if len(s) else np.nan for s in cleaned],
        "P95":    [np.nanpercentile(s, 95) if len(s) else np.nan for s in cleaned],
    })


def _norm_filename(s):
    return (str(s).replace(" ", "_")
                 .replace("[","").replace("]","")
                 .replace("(","").replace(")",""))    


# --- CDF with extra dataframe for stats --------------------------------------
def plot_CDF(project_name, data, xlabel, legend_used, bins_used, color_string=None, xlim=None):
    # 1) clean once
    cleaned = [_clean_series(d) for d in data]

    # if first is empty, match your earlier behavior
    if len(cleaned) == 0 or len(cleaned[0]) == 0:
        print(f"This CDF, {xlabel} cannot be plotted as there is no data")
        return None

    # 2) stats frame
    df = _stats_df(cleaned, legend_used, xlabel)

    # 3) plot from cleaned
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("CDF", fontsize=14)
    ax.set_yticks(np.arange(0, 1.01, step=0.1))
    ax.grid()
    plt.tight_layout()

    markers = ['', 'o', 's', 'D', '^', 'v', '>', '<', 'p', '*', 'h']
    line_styles = ['-', '--', '-', '--', ':', '-', '--', ':', '-', '--', ':', '-', '--', ':']
    use_colors = color_string is not None

    for i, s in enumerate(cleaned):
        if len(s) == 0:
            continue
        count, bins_count = np.histogram(s, bins=bins_used)
        pdf = count / np.sum(count) if np.sum(count) > 0 else np.zeros_like(count, dtype=float)
        cdf = np.cumsum(pdf)

        line_style = line_styles[i % len(line_styles)]
        marker = markers[i // len(line_styles) % len(markers)]

        if use_colors:
            plt.plot(bins_count[1:], cdf, label=legend_used[i],
                     color=color_string[i], marker=marker, linestyle=line_style,
                     markersize=4, markevery=100)
        else:
            plt.plot(bins_count[1:], cdf, label=legend_used[i],
                     marker=marker, linestyle=line_style, markersize=2, markevery=100)

    if xlim is not None:
        plt.xlim(xlim)
    ax.legend(prop={'size': 8})

    folder = f"results_{project_name}/"
    os.makedirs(folder, exist_ok=True)
    file_name = _norm_filename(xlabel)
    plt.savefig(folder + f'CDF_{file_name}.pdf')
    plt.savefig(folder + f'CDF_{file_name}.png', dpi=600)
    plt.show()
    plt.close()

    return df


# --- Bar chart CDF with extra dataframe for stats ------------------------------------
def plot_avg_std_bars(project_name, data, ylabel, legend_used, color_string=None, xlim=None):
    # 1) clean once
    cleaned = [_clean_series(d) for d in data]

    if len(cleaned) == 0 or len(cleaned[0]) == 0:
        print(f"Average and standard deviation for {ylabel} cannot be plotted as there is no data")
        return None

    # 2) stats frame
    df = _stats_df(cleaned, legend_used, ylabel)

    # 3) plot using stats in df
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid()
    plt.tight_layout()

    positions = np.arange(len(df))
    ax.bar(
        positions,
        df["Mean"].values,
        yerr=df["Std"].values,
        capsize=5,
        color=(color_string if color_string is not None else None),
        tick_label=df["Series"].values
    )

    if xlim is not None:
        plt.xlim(xlim)
    plt.xticks(rotation=20, ha='right', fontsize=14)

    folder = f"results_{project_name}/"
    os.makedirs(folder, exist_ok=True)
    file_name = _norm_filename(ylabel)
    plt.savefig(folder + f'Avg_Std_{file_name}.pdf')
    plt.savefig(folder + f'Avg_Std_{file_name}.png', dpi=600)
    plt.show()
    plt.close()

    return df       

    
def plot_avg_std_bubbles(project_name, data, ylabel, legend_used, color_string=None, xlim=None):
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8,6))
    
    # Set labels, legend, and grid
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(legend_used)))
    ax.set_xticklabels(legend_used, rotation=45, ha='right')
    ax.grid(True)
    
    # Improve layout
    plt.tight_layout()    
    
    # Calculate averages and standard deviations
    means = []
    std_devs = []
    
    for i in range(len(data)):
        data_temp = data[i]

        # Remove -inf and nan values
        if isinstance(data_temp, np.ndarray):
            data_temp = data_temp[~np.isnan(data_temp) & (data_temp != -np.inf)]
        else:
            data_temp = [x for x in data_temp if not (math.isnan(x) or (math.isinf(x) and x < 0))]

        if len(data_temp) == 0:
            if i == 0:
                print(f"Average and standard deviation for {ylabel} cannot be plotted as there is no data")
                return
            means.append(np.nan)
            std_devs.append(np.nan)
            continue
        
        # Calculate mean and standard deviation
        mean = np.mean(data_temp)
        std_dev = np.std(data_temp)
        
        means.append(mean)
        std_devs.append(std_dev)
    
    # Define positions for the bubbles, centered between ticks
    positions = np.arange(len(data))
    
    # Plot bubbles with sizes based on standard deviation
    if color_string is None:
        ax.scatter(positions, means, s=[std*100 for std in std_devs], alpha=0.5)
    else:
        ax.scatter(positions, means, s=[std*100 for std in std_devs], c=color_string, alpha=0.5)
    
    # Adjust x-axis limits for better centering of bubbles
    ax.set_xlim(-0.5, len(data) - 0.5)

    # Set x-axis limits if xlim is provided (overrides the previous adjustment)
    if xlim is not None:
        plt.xlim(xlim)
    
    # Save the plot
    folder_name = "results_" + project_name + "/"
    os.makedirs(folder_name, exist_ok=True)
    file_name = ylabel.replace(" ", "_").replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    plt.savefig(folder_name + 'Avg_Std_Bubbles_' + file_name + '.pdf')
    plt.savefig(folder_name + 'Avg_Std_Bubbles_' + file_name + '.png', dpi=600)
    
    # Show the plot
    plt.show()
    
    # Close the current plot to clear the figure
    plt.close()  


def plot_avg_std_violin(project_name, data, ylabel, legend_used, color_string=None):
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8,6))
    
    # Set labels
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(legend_used)))
    ax.set_xticklabels(legend_used, rotation=45, ha='right')
    
    # Create a list for plotting
    data_for_plotting = [d for d in data if len(d) > 0]
    legend_for_plotting = [legend_used[i] for i in range(len(data)) if len(data[i]) > 0]

    # Plot the violin plot
    if color_string is None:
        sns.violinplot(data=data_for_plotting, ax=ax)
    else:
        sns.violinplot(data=data_for_plotting, palette=color_string, ax=ax)
    
    # Improve layout
    plt.tight_layout()    
    
    # Set x-tick labels
    ax.set_xticklabels(legend_for_plotting, rotation=45, ha='right')

    # Save the plot
    folder_name = "results_" + project_name + "/"
    os.makedirs(folder_name, exist_ok=True)
    file_name = ylabel.replace(" ", "_").replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    plt.savefig(folder_name + 'Violin_' + file_name + '.pdf')
    plt.savefig(folder_name + 'Violin_' + file_name + '.png', dpi=600)
    
    # Show the plot
    plt.show()
    
    # Close the current plot to clear the figure
    plt.close()    


def plot_hist(project_name, data, xlabel, legend_used, color_string=None) :
    
    # Create the figure and axis    
    fig, ax = plt.subplots(figsize=(8,6))
    
    # Plot data 
    ax.bar(range(len(data[0])), data[0])
    
    # Set labels, legend, and grid
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PMF")
    ax.grid()
    
    # Improve layout     
    plt.tight_layout()
    
    # Save
    folder_name = "results_" + project_name + "/"
    tools.directory_exists(folder_name)
    file_name = xlabel.replace(" ", "_").replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    plt.savefig(folder_name + 'PMF_' + file_name + '.pdf')
    plt.savefig(folder_name + 'PMF_' + file_name + '.png', dpi=600)
    
    # Show 
    plt.show() 
    
    # Close the current plot to clear the figure
    plt.close()   


def plot_PMF(project_name, data, xlabel, legend_used, color_string=None) :
    
    # Create the figure and axis    
    fig, ax = plt.subplots(figsize=(8,6))
    
    # Plot data 
    # Calculate histogram heights and bins
    heights, bins = np.histogram(data, bins=2)
    unique = np.unique(data)
    heights = heights / np.sum(heights)
    
    # Calculate bar width
    bar_width = 1.8 * (max(unique) - min(unique)) / len(unique)
    
    # Plot the bar chart
    ax.bar(unique, heights, width=bar_width)
    

    # Set x-axis ticks
    ax.set_xticks(unique)

    # Set labels, legend, and grid
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PMF")
    #ax.legend(prop={'size': 8})
    ax.grid()
    
    # Improve layout 
    plt.tight_layout()
    
    # Save
    folder_name = "results_" + project_name + "/"
    tools.directory_exists(folder_name)
    file_name = xlabel.replace(" ", "_").replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    plt.savefig(folder_name + 'PMF_' + file_name + '.pdf')
    plt.savefig(folder_name + 'PMF_' + file_name + '.png', dpi=600)
    
    # Show 
    plt.show() 
    
    # Close the current plot to clear the figure
    plt.close()    


def plot_sparse_heat_map(project_name, x_size, y_size, x_indices, y_indices, values, grid_resol_m, number_of_colors, figure_title, label_title):
        
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8,8))
    
    # Create color map
    if number_of_colors < 2:
        number_of_colors = 2 
        
    if label_title == 'Cell ID':
        custom_cmap = custom_color_map_sudden(number_of_colors)
        vmin_value = None
        vmax_value = None
    elif label_title == 'Beam ID':
        custom_cmap = custom_color_map_sudden(number_of_colors)
        vmin_value = None
        vmax_value = None        
    elif label_title == 'RSRP [dBm]':
        custom_cmap = 'jet'
        vmin_value = -110
        vmax_value = -45
    elif label_title == 'Interference [dBm]':
        custom_cmap = 'jet'
        vmin_value = -110
        vmax_value = -45        
    elif label_title == 'SINR [dB]' or  label_title == 'PRB SINR [dB]':
        custom_cmap = 'jet'
        vmin_value = 60
        vmax_value = -10        

    # Create the sparse matrix
    sparse_matrix = np.ones((x_size, y_size)) * np.nan   # Assuming a x_size-by-y_size matrix
    sparse_matrix[x_indices, y_indices] = values

    # Plot the sparse heatmap
    plt.imshow(np.flip(np.transpose(sparse_matrix),axis=0), 
               cmap=custom_cmap, interpolation='nearest',vmin=vmin_value, vmax=vmax_value)

    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label(label_title)

    # Set title, labels, and ticks
    plt.title(figure_title)
    plt.xlabel('x-location [m]')
    plt.ylabel('y-location [m]')
    plt.xticks(np.linspace(0, x_size, 5), np.linspace(-x_size*grid_resol_m/2, x_size*grid_resol_m/2, 5))
    plt.yticks(np.linspace(0, y_size, 5), np.linspace(y_size*grid_resol_m/2, -y_size*grid_resol_m/2, 5))
    
    # Improve layout 
    plt.tight_layout()      
    
    # Save
    folder_name = "results_" + project_name + "/"
    tools.directory_exists(folder_name)
    plt.savefig(folder_name + 'heatmap_' + figure_title.replace(" ", "_") + '.pdf')
    plt.savefig(folder_name + 'heatmap_' + figure_title.replace(" ", "_") + '.png', dpi=600)  
    
    # Show
    plt.show()    
    
    # Close the current plot to clear the figure
    plt.close()


def plot_antenna_gain_4_figures(angles_a, gains_a, x_lim_a, angles_b, gains_b, x_lim_b, title):

    # Initialize figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.4)  # Adjust spacing between subplots
    axes = axes.flatten()
    fig.suptitle(title)
    
    # Process each code word
    for codeword_index in range(np.size(gains_a, 1)):
        ax1, ax2, ax3, ax4 = axes
        
        ax1.plot(np.degrees(angles_a), gains_a[:, codeword_index])
        
        ax2 = plt.subplot(222, projection='polar')
        ax2.plot(angles_a, gains_a[:, codeword_index])
        
        ax3.plot(np.degrees(angles_b), gains_b[:, codeword_index])
        
        ax4 = plt.subplot(224, projection='polar')
        ax4.plot(angles_b, gains_b[:, codeword_index]) 
    
    # Set axes
    ax1.set_xlim(np.degrees(x_lim_a))
    ax3.set_xlim(np.degrees(x_lim_b))
    
    ax1.xaxis.set_major_formatter(FuncFormatter(degrees_formatter))
    ax3.xaxis.set_major_formatter(FuncFormatter(degrees_formatter))
    ax3.yaxis.set_major_formatter(FuncFormatter(db_formatter))
    
    ax2.yaxis.set_major_formatter(FuncFormatter(db_formatter))
    ax4.yaxis.set_major_formatter(FuncFormatter(db_formatter))
    
    # Set grid
    ax1.grid()
    ax3.grid()
    
    # Plot
    plt.tight_layout()
    plt.show()    


def plot_antenna_gain_2_figures(angles_a, gains_a, x_lim_a, angles_b, gains_b, x_lim_b, title):

    # Initialize figure with two parallel subplots
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(10, 4))
    ax1, ax2 = axes  # Unpack the axes

    # Process each codeword and plot on both axes
    for codeword_index in range(np.size(gains_a, 1)):
        ax1.plot(angles_a, gains_a[:, codeword_index]) #label=f'Codeword {codeword_index + 1}'
        ax2.plot(angles_b, gains_b[:, codeword_index]) # label=f'Codeword {codeword_index + 1}'
        
    # Set y-axis labels in dB for both subplots
    ax1.yaxis.set_major_formatter(FuncFormatter(db_formatter))
    ax2.yaxis.set_major_formatter(FuncFormatter(db_formatter))
    
    # Add legends
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    
    # Plot
    plt.tight_layout()
    plt.show()   


def plot_antenna_gain_3D(horizontal_angles, horizontal_pattern, vertical_angles, vertical_pattern, title):
        
    # Antenna patterns
    theta = horizontal_angles # Azimuth angle
    phi = vertical_angles # Elevation angle

    # Create a meshgrid for the angles
    Theta, Phi = np.meshgrid(theta,phi)

    # Calculate the radiation pattern
    for codeword_index in range(0, np.size(horizontal_pattern,1)):
        
        pattern = horizontal_pattern[:,codeword_index] * vertical_pattern[:,codeword_index]
    
        # Convert spherical coordinates to Cartesian coordinates
        X = pattern * np.sin(Phi) * np.cos(Theta)
        Y = pattern * np.sin(Phi) * np.sin(Theta)
        Z = pattern * np.cos(Phi)
        
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')        

        # Plot the antenna pattern
        ax.plot_surface(X, Y, Z, cmap='viridis')

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
    
        # Show the plot
        plt.show()  
        

def get_coupling_loss_benchmark(model, beam_type):

    if model == "ITU_R_M2135_UMa_uniform" and beam_type == "antenna element":
        benchmark_best_serving_cell_coupling_gain_dB = [np.array([-129.70, -120.75, -118.62, -117.32, -116.43, -115.64, -114.96, -114.36, -113.80, -113.28, -112.78, -112.31, -111.86, -111.40, -110.99, 
                                                                 -110.63, -110.23, -109.91, -109.55, -109.21, -108.84, -108.49, -108.17, -107.86, -107.55, -107.19, -106.89, -106.58, -106.28, -105.99, -105.68, -105.36, -105.04, -104.71, -104.38, -104.07, -103.78, -103.47, 
                                                                 -103.18, -102.89, -102.56, -102.24, -101.96, -101.61, -101.32, -100.97, -100.66, -100.34, -99.98, -99.66, -99.30, -98.93, -98.57, -98.19, -97.84, -97.45, -97.07, -96.69, 
                                                                 -96.33, -95.92, -95.51, -95.09, -94.72, -94.30, -93.88, -93.41, -92.97, -92.52, -92.03, -91.55, -91.04, -90.52, -90.01, -89.47, -88.96, -88.42, -87.81, -87.25, -86.66, -86.09, 
                                                                 -85.42, -84.83, -84.22, -83.60, -82.91, -82.24, -81.53, -80.86, -80.09, -79.27, -78.48, -77.68, -76.88, -76.00, -75.05, -73.97, -72.68, -71.24, -69.33, -66.73, -55.60])]
    
    elif model == "ITU_R_M2135_UMi_uniform" and beam_type == "antenna element":
        benchmark_best_serving_cell_coupling_gain_dB = [np.array([-133.27,-124.91,-122.14,-120.55,-119.25,-118.04,-117.00,-115.89,-114.82,-113.91,-113.00,-112.12,-111.30,-110.42,-109.58,-108.75,
                                                                 -107.87,-107.04,-106.14,-105.35,-104.44,-103.65,-102.82,-102.10,-101.39,-100.66,-99.89,-99.26,-98.61,-98.02,-97.40,-96.86,-96.33,-95.88,-95.42,-94.99,-94.54,-94.10,-93.68,-93.23,-92.80,-92.39,-91.99,-91.57,-91.15,-90.73,
                                                                 -90.32,-89.89,-89.42,-88.99,-88.56,-88.11,-87.69,-87.24,-86.79,-86.31,-85.82,-85.31,-84.84,-84.31,-83.78,-83.25,-82.69,-82.14,-81.53,-80.94,-80.33,-79.65,-78.99,-78.32,-77.59,-76.88,-76.14,-75.46,-74.76,-74.06,
                                                                 -73.38,-72.68,-71.98,-71.32,-70.67,-70.06,-69.41,-68.72,-68.09,-67.39,-66.69,-65.97,-65.27,-64.57,-63.81,-63.06,-62.32,-61.50,-60.64,-59.75,-58.77,-57.58,-56.30,-54.26,-47.57])]          
        
    elif model == "3GPPTR36_814_Case_1_uniform" and beam_type == "antenna element":
        benchmark_best_serving_cell_coupling_gain_dB = [np.array([-144.41,-133.53,-131.37,-129.96,-128.84,-127.97,-127.24,-126.59,-125.99,-125.45,-124.94,-124.44,-124.01,-123.58,-123.17,-122.78,
                                                                 -122.39,-122.01,-121.65,-121.32,-120.99,-120.66,-120.33,-120.01,-119.69,-119.38,-119.07,-118.79,-118.50,-118.21,-117.93,-117.63,
                                                                 -117.36,-117.08,-116.81,-116.55,-116.26,-115.96,-115.70,-115.42,-115.14,-114.86,-114.59,-114.32,-114.06,-113.78,-113.49,-113.21,
                                                                 -112.94,-112.67,-112.39,-112.12,-111.84,-111.58,-111.30,-111.03,-110.75,-110.47,-110.19,-109.92,-109.63,-109.33,-109.03,-108.74,
                                                                 -108.41,-108.09,-107.78,-107.47,-107.17,-106.85,-106.53,-106.18,-105.84,-105.50,-105.10,-104.74,-104.38,-104.00,-103.62,-103.20,
                                                                 -102.80,-102.39,-101.97,-101.55,-101.05,-100.56,-100.09,-99.60,-99.03,-98.50,-97.83,-97.12,-96.37,-95.60,-94.83,-93.78,
                                                                 -92.59,-91.19,-89.38,-86.83,-71.43])]            
        
    elif (model == "3GPPTR38_901_UMa_lsc_uniform" or model == "3GPPTR38_901_UMa_lsc_sn_uniform") and beam_type == "SSB":
        file_path = os.path.join(os.getcwd(), '..', 'tests' , 'benchmark_results_3GPP', 'TR38901', 'couplingLoss38901_P1_UMa_6GHz.mat')
        mat = scipy.io.loadmat(file_path)  
        benchmark_best_serving_cell_coupling_gain_dB = mat["couplingLoss38901_P1_UMa_6GHz"][:,1:]   
        benchmark_best_serving_cell_coupling_gain_dB = [column for column in benchmark_best_serving_cell_coupling_gain_dB.T]   # Convert to list of arrays
        
    elif model == "3GPPTR38_901_UMi_lsc_uniform"  and beam_type == "SSB":
        file_path = os.path.join(os.getcwd(), '..', 'tests' , 'benchmark_results_3GPP', 'TR38901', 'couplingLoss38901_P1_UMi_6GHz.mat')
        mat = scipy.io.loadmat(file_path)  
        benchmark_best_serving_cell_coupling_gain_dB = mat["couplingLoss38901_P1_UMi_6GHz"][:,1:] 
        benchmark_best_serving_cell_coupling_gain_dB = [column for column in benchmark_best_serving_cell_coupling_gain_dB.T]   # Convert to list of arrays
                  
    else:
        benchmark_best_serving_cell_coupling_gain_dB = []

    return benchmark_best_serving_cell_coupling_gain_dB  


def get_geometry_sinr_benchmark(model, beam_type):
    
    if model == "ITU_R_M2135_UMa_uniform" and beam_type == "antenna element":
        benchmark_rsrp_based_sinr_dB = [np.array([-7.27, -4.34, -3.72, -3.34, -3.11, -2.97, -2.81, -2.54, -2.30, -2.05, -1.83, -1.63, -1.43, -1.25, -1.07, -0.91,
                                                 -0.74, -0.57, -0.43, -0.28, -0.14, 0.00, 0.16, 0.31, 0.47, 0.63, 0.77, 0.92, 1.07, 1.21, 1.36, 1.51, 1.66, 1.82, 1.98, 2.14, 2.31, 2.47, 2.63,
                                                 2.81, 2.97, 3.13, 3.30, 3.47, 3.64, 3.81, 3.99, 4.15, 4.33, 4.51, 4.69, 4.89, 5.08, 5.28, 5.46, 5.65, 5.85, 6.05, 6.25, 6.46, 6.67, 6.87, 7.07, 7.29, 7.51, 7.71, 7.91, 8.12, 8.35, 8.57, 8.80,
                                                 9.01, 9.26, 9.49, 9.74, 9.97, 10.21, 10.45, 10.69, 10.95, 11.21, 11.47, 11.73, 12.00, 12.26, 12.53, 12.80,
                                                 13.06, 13.31, 13.56, 13.82, 14.05, 14.32, 14.60, 14.87, 15.12, 15.40, 15.69, 16.04, 16.41, 16.92])] 
        
    elif model == "ITU_R_M2135_UMi_uniform" and beam_type == "antenna element":    
        benchmark_rsrp_based_sinr_dB = [np.array([-7.33,-4.17,-3.52,-3.15,-3.01,-2.74,-2.42,-2.13,-1.87,-1.64,-1.43,-1.19,-1.00,-0.82,-0.64,-0.47,-0.31,-0.14,0.01,0.17,0.33,0.48,0.63,0.78,0.94,1.11,1.27,1.43,1.59,1.76,1.92,2.09,
                                                 2.26,2.44,2.60,2.77,2.94,3.12,3.31,3.48,3.66,3.85,4.03,4.21,4.38,4.57,4.76,4.94,5.13,5.31,5.49,5.71,5.90,6.09,6.28,6.47,6.67,6.88,7.07,7.26,7.46,7.66,7.87,8.10,
                                                 8.31,8.53,8.74,8.96,9.39,9.59,9.79,10.00,10.22,10.45,10.67,10.90,11.11,11.34,11.59,11.84,12.07,12.30,12.53,12.78,13.03,13.24,13.46,13.68,13.92,14.14,14.37,14.61,14.83,15.06,15.32,15.56,
                                                 15.81,16.10,16.45,16.88])]
             
    elif model == "3GPPTR36_814_Case_1_uniform" and beam_type == "antenna element":   
        benchmark_rsrp_based_sinr_dB = [np.array([-6.89,-3.65,-3.13,-2.75,-2.38,-2.04,-1.78,-1.50,-1.29,-1.07,-0.87,-0.69,-0.51,-0.34,-0.16,0.01,0.19,0.35,0.53,0.69,0.84,0.99,1.18,1.33,1.48,1.64,1.80,1.95,2.10,2.29,2.47,2.64,
                                                 2.83,3.03,3.22,3.42,3.60,3.79,3.99,4.18,4.38,4.57,4.77,4.96,5.14,5.35,5.56,5.79,6.00,6.20,6.41,6.64,6.87,7.10,7.33,7.56,7.76,8.02,8.23,8.47,8.71,8.94,9.18,9.43,
                                                 9.68,9.93,10.19,10.45,10.68,10.92,11.18,11.45,11.72,11.98,12.25,12.56,12.85,13.13,13.47,13.79,14.11,14.40,14.70,15.00,15.33,15.64,15.96,16.30,16.64,16.99,17.38,17.74,18.11,18.48,18.87,19.31,
                                                 19.75,20.16,20.58,21.02,21.73])]
        
    elif (model == "3GPPTR38_901_UMa_lsc_uniform" or model == "3GPPTR38_901_UMa_lsc_sn_uniform") and beam_type == "SSB":        
        file_path = os.path.join(os.getcwd(), '..', 'tests' , 'benchmark_results_3GPP', 'TR38901', 'geometry38901_P1_UMa_6GHz.mat')
        mat = scipy.io.loadmat(file_path)         
        benchmark_rsrp_based_sinr_dB = mat["geometry38901_P1_UMa_6GHz"][:,1:] 
        benchmark_rsrp_based_sinr_dB = [column for column in benchmark_rsrp_based_sinr_dB.T]   # Convert to list of arrays
        
    elif model == "3GPPTR38_901_UMi_lsc_uniform"  and beam_type == "SSB":        
        file_path = os.path.join(os.getcwd(), '..', 'tests' , 'benchmark_results_3GPP', 'TR38901', 'geometry38901_P1_UMi_6GHz.mat')
        mat = scipy.io.loadmat(file_path)           
        benchmark_rsrp_based_sinr_dB = mat["geometry38901_P1_UMi_6GHz"][:,1:]
        benchmark_rsrp_based_sinr_dB = [column for column in benchmark_rsrp_based_sinr_dB.T]   
        
    elif model == "3GPPTR38_901_UMa_C1_uniform"  and beam_type == "SSB":        
        benchmark_rsrp_based_sinr_dB = np.loadtxt(os.path.join(os.getcwd(), '..', 'tests' , 'benchmark_results_3GPP', 'TR38901', 'noiselessGeometry38901_P2_C1_UMa_6GHz.txt'))  
        benchmark_rsrp_based_sinr_dB = [column for column in benchmark_rsrp_based_sinr_dB.T]  
        
    elif model == "3GPPTR38_901_UMa_C2_uniform"  and beam_type == "SSB":        
        benchmark_rsrp_based_sinr_dB = np.loadtxt(os.path.join(os.getcwd(), '..', 'tests' , 'benchmark_results_3GPP', 'TR38901', 'noiselessGeometry38901_P2_C2_UMa_6GHz.txt')) 
        benchmark_rsrp_based_sinr_dB = [column for column in benchmark_rsrp_based_sinr_dB.T]   
        
    elif model == "3GPPTR38_901_UMi_C1_uniform"  and beam_type == "SSB":        
        benchmark_rsrp_based_sinr_dB = np.loadtxt(os.path.join(os.getcwd(), '..', 'tests' , 'benchmark_results_3GPP', 'TR38901', 'noiselessGeometry38901_P2_C1_UMi_6GHz.txt')) 
        benchmark_rsrp_based_sinr_dB = [column for column in benchmark_rsrp_based_sinr_dB.T]          
        
    elif model == "3GPPTR38_901_UMi_C2_uniform"  and beam_type == "SSB":        
        benchmark_rsrp_based_sinr_dB = np.loadtxt(os.path.join(os.getcwd(), '..', 'tests' , 'benchmark_results_3GPP', 'TR38901', 'noiselessGeometry38901_P2_C2_UMi_6GHz.txt')) 
        benchmark_rsrp_based_sinr_dB = [column for column in benchmark_rsrp_based_sinr_dB.T]    
        
    else: 
        benchmark_rsrp_based_sinr_dB = []

    return benchmark_rsrp_based_sinr_dB         
