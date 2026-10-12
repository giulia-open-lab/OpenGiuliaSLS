import os
import sys
import threading

# Configure Giulia into path
root_dir = os.path.dirname(__file__)
sys.path.insert(1, str(root_dir))

from giulia.inputs import InputConfig
from giulia.outputs.monitor import Monitor
from giulia.outputs.saveable import Saveable


config_file = sys.argv[1] if len(sys.argv) > 1 else None
config = InputConfig.load(config_file)
stop_event = threading.Event()


def run_giulia() -> None:
    try:
        # Configure and prepare the run
        giulia_main, args = config.run()
        # Launch Giulia
        giulia_main(args, config)
    finally:
        stop_event.set()


if __name__ == '__main__':
    # Why the split between threads:
    # Giulia must run on the OS main thread, because its C++ backends
    # (PyTorch / Mitsuba) need to be initialized and torn down
    # there to avoid segfaults during interpreter shutdown. The monitor only
    # reads published values, so it is safe to push onto a background thread.
    #
    # The Saveable registry is per execution context — in practice one per thread — so the
    # Monitor must be constructed on the thread that runs the simulation; here, both are the
    # main thread. It captures that registry at construction time and shares it with its
    # polling thread. A monitor in another *process* could not see this registry at all.
    # See docs/output-saveable/saveable-concurrency.md

    # Start the background monitor; start() returns the running thread.
    # The monitor validates these keys against the Saveable registry: a
    # misspelled KPI is reported as a warning when the run ends (or raised
    # immediately if the registry is already populated). Each key must hold
    # exactly one '.', though — a key that does not is rejected by Saveable.get
    # with a ValueError that ends the polling thread. Use Saveable.catalog()
    # to discover the valid keys (printed at the end of this script).
    # poll_interval controls how often (in seconds) the monitor reads and prints
    # the KPIs; it defaults to 1.0 if omitted.
    kpis = ['Cell_Selection_SSB.best_serving_cell_ID_per_ue']
    monitor = Monitor(kpis, stop_event, poll_interval=1.0)
    monitor_thread = monitor.start()

    # Run Giulia on the main thread. This blocks until the simulation finishes
    # (or raises), and on exit signals stop_event so the monitor loop ends.
    run_giulia()

    # Wait for the monitor to observe stop_event and exit cleanly.
    monitor_thread.join()

    # Example: read the latest cached KPI value(s) from the monitor.
    # A specific KPI:
    best_serving_cell_ID_per_ue = monitor.get('Cell_Selection_SSB.best_serving_cell_ID_per_ue')
    print('Last best serving cell ID per UE:', best_serving_cell_ID_per_ue)
    # All monitored KPIs (single value if only one, otherwise a dict):
    print('Final KPI snapshot:', monitor.get())

    # Discover every Saveable variable available this run. The Saveable registry
    # is fully populated once the simulation has finished, so this lists all the
    # <ClassName>.<variable_name> keys that could have been passed to Monitor.
    # print('Saveable variables:', Saveable.catalog())
