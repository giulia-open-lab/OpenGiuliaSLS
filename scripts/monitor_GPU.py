import os
import time
import platform
import pynvml
import psutil
import matplotlib.pyplot as plt
import numpy as np  # Import numpy


def clear_console():
    """
    Clears the terminal based on the operating system.
    """
    if platform.system().lower() == "windows":
        os.system("cls")
    else:
        os.system("clear")


def monitor_gpu_python_processes(refresh_interval=1.0, duration=None):
    """
    Continuously monitors the GPU memory used by Python processes,
    updating the output every `refresh_interval` seconds.

    If `duration` (in seconds) is provided, the monitor will run for that duration
    and then exit automatically.

    All memory usage data is stored in a list, which is returned at the end.
    """
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as err:
        print("Error initializing NVML:", err)
        return []

    num_gpu = pynvml.nvmlDeviceGetCount()
    start_time = time.time()
    
    # List to store memory consumption data for each iteration.
    memory_usage_history = []
    
    try:
        while True:
            clear_console()
            current_time = time.time()
            print("=== GPU Monitor - Real-time Update ===")
            if duration is not None:
                elapsed = current_time - start_time
                remaining = duration - elapsed
                print(f"Time remaining: {remaining:.1f} seconds")
            else:
                print("Press Ctrl+C to exit.")
            print()

            # Record for the current iteration.
            iteration_record = {
                "timestamp": current_time,
                "gpus": []
            }
            
            for i in range(num_gpu):
                gpu_record = {"gpu_index": i, "python_processes": []}
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                except pynvml.NVMLError as err:
                    print(f"Cannot get GPU #{i}: {err}")
                    continue

                print(f"--- GPU #{i} Status ---")
                try:
                    # Retrieve processes using the GPU in compute mode.
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                except pynvml.NVMLError as err:
                    print(f"Error retrieving processes on GPU #{i}: {err}")
                    processes = []

                if not processes:
                    print("No processes running on this GPU.")
                else:
                    for proc in processes:
                        pid = proc.pid
                        used_memory_bytes = proc.usedGpuMemory  # in bytes
                        
                        # Try to get the process name using psutil.
                        try:
                            process_obj = psutil.Process(pid)
                            process_name = process_obj.name()
                        except Exception:
                            process_name = "Unavailable"
                        
                        used_memory_mb = used_memory_bytes / (1024 * 1024)
                        
                        # Display only processes that have "python" in the name (case insensitive).
                        if "python" in process_name.lower():
                            print(f"PID: {pid:<8} Name: {process_name:<25} Memory used: {used_memory_mb:.2f} MB")
                            
                            # Save process information into the GPU record.
                            process_data = {
                                "pid": pid,
                                "process_name": process_name,
                                "memory_used_mb": used_memory_mb
                            }
                            gpu_record["python_processes"].append(process_data)
                iteration_record["gpus"].append(gpu_record)
            
            # Append the current iteration's record to the history list.
            memory_usage_history.append(iteration_record)
            
            # Check if the specified duration has elapsed (if duration is set).
            if duration is not None and (time.time() - start_time >= duration):
                print("\nMonitoring duration elapsed. Exiting.")
                break
            
            # Wait for the specified refresh interval before updating.
            time.sleep(refresh_interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped by user.")
    finally:
        pynvml.nvmlShutdown()
    
    # Return the history list containing all memory consumption records.
    return memory_usage_history


def plot_memory_usage(history):
    """
    Plots the GPU memory usage history.

    X-axis: Time (seconds since monitoring started)
    Y-axis: Memory usage (MB)
    Each GPU will have its own line.
    """
    if not history:
        print("No history data to plot.")
        return

    # Use the first timestamp as the reference point.
    start_time = history[0]["timestamp"]
    gpu_data = {}
    
    # Process the history data to extract memory usage for each GPU over time.
    for record in history:
        elapsed_time = record["timestamp"] - start_time
        for gpu in record["gpus"]:
            gpu_index = gpu["gpu_index"]
            # Sum memory usage for all Python processes on this GPU.
            total_memory = sum(proc["memory_used_mb"] for proc in gpu["python_processes"])
            if gpu_index not in gpu_data:
                gpu_data[gpu_index] = {"times": [], "memory": []}
            gpu_data[gpu_index]["times"].append(elapsed_time)
            gpu_data[gpu_index]["memory"].append(total_memory)

    # Plot the data.
    plt.figure(figsize=(10, 6))
    for gpu_index, data in gpu_data.items():
        plt.plot(data["times"], data["memory"], label=f"GPU #{gpu_index}")

    plt.xlabel("Time (s)")
    plt.ylabel("Memory Usage (MB)")
    plt.title("GPU Memory Usage Over Time for Python Processes")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.show()



if __name__ == "__main__":
    # Set the duration (in seconds) for monitoring; for example, 10 seconds.
    history = monitor_gpu_python_processes(refresh_interval=0.1, duration=900)
    
    # Convert the history list to a NumPy array (of type object)
    history_array = np.array(history, dtype=object)
    
    # Build the complete path for the output file.
    output_dir = os.path.join("..", "outputs")
    os.makedirs(output_dir, exist_ok=True)  # Create the folder if it does not exist
    output_file = os.path.join(output_dir, "gpu_memory_monitor.npy")
    
    # Save the NumPy array to the specified file.
    np.save(output_file, history_array)
    print(f"Monitoring data saved to '{output_file}'")
    
    # Plot the collected memory usage history.
    plot_memory_usage(history)