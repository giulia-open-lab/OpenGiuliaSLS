"""
Created on Fri Apr 11 17:24:43 2025

@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org 
matteo.bernabe@iteam.upv.es
"""

import threading
import time
from datetime import datetime
from typing import Optional, Callable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import psutil
import os

from x_11_colors import x11_colors

# Optional GPU monitoring (for NVIDIA GPUs)
try:
    from pynvml import nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit
    has_gpu = True
except ImportError:
    has_gpu = False


def __map_range(item: tuple[str, str, str]) -> dict[str, any]:
    name = item[0].lstrip('usage_').rstrip('.csv')
    return {
        'start': item[1],
        'end': item[2],
        'label': name
    }


def compute_ranges(directory: str, predicate: Callable[[str], bool],) -> list[dict[str, any]]:
    files: list[str] = os.listdir(directory)
    files = list(filter(predicate, files))
    result: list[tuple[str, str, str]] = list()
    for file in files:
        lines: list[str]
        with open(os.path.join(directory, file), 'r') as f:
            lines = f.readlines()
        start = lines[1].split(',')[0]
        end = lines[-1].split(',')[0]
        result.append((file, start, end))
    return list(
        map(__map_range, result)
    )


class SystemMonitor:
    def __init__(self, filename='usage_log.csv', interval: float = 1):
        self.filename = filename
        self.interval = interval
        self.stop_event = threading.Event()
        self.monitor_thread = None

        # Initialize GPU if available
        if has_gpu:
            nvmlInit()


    def get_gpu_usage(self):
        if not has_gpu:
            return {}

        gpu_info = {}
        device_count = nvmlDeviceGetCount()
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            gpu_info[f'gpu_{i}_mem_used'] = mem_info.used // (1024 ** 2)  # MB
            gpu_info[f'gpu_{i}_mem_total'] = mem_info.total // (1024 ** 2)  # MB
        return gpu_info


    def get_system_usage(self) -> dict[str, int]:
        # Get RAM usage
        ram = psutil.virtual_memory()
        ram_used = ram.used // (1024 ** 2)  # MB
        ram_total = ram.total // (1024 ** 2)  # MB

        # Get CPU usage (percentage)
        cpu_percent = psutil.cpu_percent()

        return {
            'timestamp': datetime.now().isoformat(timespec='microseconds'),
            'ram_used': ram_used,
            'ram_total': ram_total,
            'cpu_percent': cpu_percent,
            **self.get_gpu_usage()
        }


    def monitor_loop(self):
        # Write header to file
        with open(self.filename, 'w') as f:
            f.write('timestamp,ram_used,ram_total,cpu_percent')
            if has_gpu:
                device_count = nvmlDeviceGetCount() if has_gpu else 0
                for i in range(device_count):
                    f.write(f',gpu_{i}_mem_used,gpu_{i}_mem_total')
            f.write('\n')

        while not self.stop_event.is_set():
            usage = self.get_system_usage()
            with open(self.filename, 'a') as f:
                values = list(usage.values())
                f.write(','.join(str(x) for x in values))
                f.write('\n')
            time.sleep(self.interval)


    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.monitor_loop)
        thread.start()
        return thread


    def stop(self):
        self.stop_event.set()


    def plot_usage(self, output_file="usage_plot.png", label: Optional[str] = None, ranges = None):
        # Read the CSV file
        df = pd.read_csv(self.filename)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Create subplots
        fig, axes = plt.subplots(nrows=3 if 'gpu_0_mem_used' in df.columns else 2,
            figsize=(12, 8),sharex='all'
        )

        # Set up main title
        fig.suptitle('System Resource Usage', y=1.02, fontsize=14)
        if label is not None:
            fig.text(0.5, 0.97,label,ha='center', va='center',fontsize=10, style='italic', color='#666666')

            # Add highlighted regions if specified
            if ranges:
                legend_patches = []
                for i, range_spec in enumerate(ranges):
                    # Convert to datetime
                    start = pd.to_datetime(range_spec['start'])
                    end = pd.to_datetime(range_spec['end'])
                    label = range_spec.get('label', f'Phase {i + 1}')
                    color = x11_colors[i]
                    alpha = 0.8

                    # Add spans to all axes
                    for ax in axes:
                        ax.axvspan(start, end, facecolor=color, alpha=alpha)

                    # Create legend patch
                    legend_patches.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=alpha, label=label))

                # Add combined legend
                fig.legend(handles=legend_patches,bbox_to_anchor=(0.5, -0.27),loc='lower center',framealpha=0.9)

        # Plot CPU Usage
        axes[0].plot(df['timestamp'], df['cpu_percent'], color='tab:red')
        axes[0].set_ylabel('CPU Usage (%)')
        axes[0].set_title('CPU Utilization')
        axes[0].grid(True, alpha=0.3)

        # Plot RAM Usage
        axes[1].plot(df['timestamp'], df['ram_used'], label='Used', color='tab:blue')
        axes[1].axhline(y=df['ram_total'].iloc[0], color='tab:blue', linestyle='--', label='Total')
        axes[1].set_ylabel('RAM (MB)')
        axes[1].set_title('Memory Usage')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot GPU Memory if available
        if 'gpu_0_mem_used' in df.columns:
            gpu_axes = axes[2]
            gpu_count = sum(1 for col in df.columns if col.startswith('gpu_0')) // 2

            for i in range(gpu_count):
                gpu_axes.plot(df['timestamp'],
                              df[f'gpu_{i}_mem_used'],
                              label=f'GPU {i} Used')
                gpu_axes.axhline(y=df[f'gpu_{i}_mem_total'].iloc[0],
                                 linestyle='--',
                                 label=f'GPU {i} Total')

            gpu_axes.set_ylabel('GPU Memory (MB)')
            gpu_axes.set_title('GPU Memory Usage')
            gpu_axes.legend()
            gpu_axes.grid(True, alpha=0.3)

        # Format x-axis
        date_fmt = mdates.DateFormatter('%H:%M:%S')
        axes[-1].xaxis.set_major_formatter(date_fmt)
        fig.autofmt_xdate(rotation=45)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {output_file}")
