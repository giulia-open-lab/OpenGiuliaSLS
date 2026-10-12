import threading
from typing import Any, Optional

from giulia.logger import warning
from giulia.outputs.saveable import Saveable


class Monitor:
    """Polls a set of KPIs from Giulia's outputs on a background thread.

    The latest value of each KPI is cached and exposed through ``get``.

    Each KPI is a ``<ClassName>.<variable_name>`` key, as accepted by
    ``Saveable.get`` and listed by ``Saveable.catalog``. A key that is merely
    *unknown* reads as ``None`` and is reported at shutdown; a key that is
    *malformed* — not exactly one ``'.'`` — makes ``Saveable.get`` raise, which
    ends the polling thread on its first cycle (see ``_monitor``).
    """

    def __init__(self, kpis: list[str], stop_event: threading.Event, poll_interval: float = 1.0):
        """
        Args:
            kpis: KPI keys to watch, each a ``<ClassName>.<variable_name>``.
            stop_event: shared event used to stop the polling loop.
            poll_interval: seconds to wait between polls (and prints). Defaults
                to 1.0. The wait is interruptible, so shutdown stays responsive
                regardless of this value.
        """
        # Construct this on the thread that runs the simulation: __init__ is where the
        # registry is captured (see the share_registry call below) and where the eager
        # validation reads the catalog.
        self._kpis = kpis
        self._stop_event = stop_event
        self._poll_interval = poll_interval
        self._values: dict[str, Any] = {}
        # Daemon thread: it keeps polling the simulation's outputs in the
        # background and won't keep the process alive on its own.
        #
        # share_registry() is what makes the polling actually see anything. The Saveable
        # registry is per execution context, and a new thread starts with an empty one, so
        # an unwrapped loop would read None for every KPI and then report them all as
        # "never available". Wrapping here — on the thread that creates the Monitor, which
        # is also the thread that runs the simulation — binds the monitor thread to that
        # thread's registry. See docs/output-saveable/saveable-concurrency.md.
        self._thread = threading.Thread(
            target=Saveable.share_registry(self._monitor), name='monitor', daemon=True
        )

        # If the Saveable registry is already populated (e.g. the monitor is
        # created after the simulation has instantiated its outputs), validate
        # eagerly so a misspelled key fails fast. When the monitor is created
        # before the run (the usual case) the registry is still empty, so there
        # is nothing to check against yet; unknown keys are then reported once
        # the run finishes (see _monitor).
        if Saveable.catalog():
            self._raise_if_unknown()

    def start(self) -> threading.Thread:
        self._thread.start()
        return self._thread

    def _monitor(self) -> None:
        # Nothing here catches: a malformed key (not exactly one '.') makes Saveable.get
        # raise ValueError, which kills this thread after threading prints the traceback.
        # The loop then never reaches the shutdown warning below, and join() still returns
        # normally — so a bad key costs the *whole* monitor, not just that KPI.
        while not self._stop_event.is_set():
            for kpi in self._kpis:
                value = Saveable.get(kpi)
                self._values[kpi] = value
                print(f'{kpi}: {value}')

            if self._stop_event.wait(timeout=self._poll_interval): break

        # The run is over, so this context's registry holds one entry per Saveable
        # class the run ever instantiated (re-registering a class replaces its entry;
        # nothing is ever dropped outright). Any requested KPI still missing from it was
        # never available and is almost certainly a typo.
        unknown = self._unknown_kpis()
        if unknown:
            warning(
                f'These KPIs were never available and may be misspelled: '
                f'{unknown}. Use Saveable.catalog() to list valid keys.'
            )

    def get(self, kpi: Optional[str] = None):
        """Return the latest cached KPI value(s).

        - With ``kpi``: the latest value of that single KPI.
        - Without arguments: the single value if only one KPI is monitored,
          otherwise a dict mapping each KPI to its latest value.

        The no-argument form branches on how many values are *cached*, not on how
        many KPIs were requested: before the first poll — or if ``start`` was never
        called — the cache is empty and it returns ``{}``.
        """
        if kpi is not None:
            return self._values.get(kpi)
        if len(self._values) == 1:
            return next(iter(self._values.values()))
        return dict(self._values)

    def _unknown_kpis(self) -> list[str]:
        """Requested KPIs that are not (yet) in the Saveable catalog."""
        catalog = Saveable.catalog()
        return [kpi for kpi in self._kpis if kpi not in catalog]

    def _raise_if_unknown(self) -> None:
        unknown = self._unknown_kpis()
        if unknown:
            raise ValueError(
                f'Unknown KPI(s): {unknown}. '
                f'Available keys: {sorted(Saveable.catalog())}'
            )
