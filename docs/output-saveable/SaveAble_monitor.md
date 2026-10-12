# Saveable Monitor

This document describes the **live monitoring** mechanism built on top of
[Saveable objects](../source/saveable-objects.rst): how the threading model is
structured, how the `Monitor` class works, and how to use it.

The goal is to read the value of one or more KPIs **while a simulation is
running**, without modifying the simulator core.

> The monitor reads Giulia's Saveable registry from a *different thread* than the one
> running the simulation. That is only possible because the monitor explicitly opts into
> the simulation's registry — see [§1.1](#11-how-the-monitor-thread-sees-the-registry) and,
> for the full model, [Saveable Concurrency](saveable-concurrency.md).

---

## 1. The threading model

Giulia is launched from [`main.py`](../../main.py). Two pieces of work run at the
same time:

| Work | Where it runs | Why |
|------|---------------|-----|
| The **simulation** (`run_giulia`) | OS **main thread** | Its C++ backends (PyTorch / Mitsuba) must be initialized and torn down on the OS main thread, otherwise the interpreter segfaults during shutdown. |
| The **monitor** (`Monitor`) | A **background daemon thread** | It only *reads* values that the simulation publishes, so it is safe to move off the main thread. |

Because those are two different threads, they would by default have two different Saveable
registries; [§1.1](#11-how-the-monitor-thread-sees-the-registry) explains how the monitor
opts into the simulation's.

The two are coordinated by a single `threading.Event`, named `stop_event`:

- The simulation runs on the main thread and blocks until it finishes.
- When it finishes (or raises), it **sets** `stop_event` in a `finally` block.
- The monitor loop checks `stop_event` on every iteration and exits as soon as
  it is set.

This guarantees the monitor always stops cleanly when the simulation ends,
even if the simulation crashes.

### 1.1 How the monitor thread sees the registry

Splitting the work across two threads is not free, because **the Saveable registry is per
execution context** — in practice one registry per thread, for the life of that thread
([Saveable Concurrency](saveable-concurrency.md)). A brand-new thread starts with an *empty*
registry and inherits nothing from the thread that started it, so a naive polling loop would
read `None` for every KPI, forever, and then report all of them as "never available" at
shutdown. Nothing would raise; the monitor would simply be useless.

`Monitor.__init__` therefore wraps the loop before starting the thread:

```python
self._thread = threading.Thread(
    target=Saveable.share_registry(self._monitor), name='monitor', daemon=True
)
```

`Saveable.share_registry` captures the registry of the thread that constructs the `Monitor`
and binds the monitor thread to that same object. Two things follow:

- **Construct the `Monitor` on the thread that runs the simulation.** In
  [`main.py`](../../main.py) both are the main thread, so this is automatic. If you move the
  simulation onto a worker thread, the `Monitor` must be created there too.
- **The shared registry is live, not a snapshot.** Saveables registered *after* the monitor
  starts — which is every episode, since `configure()` re-creates them — are visible
  immediately. This is what lets the monitor track a KPI across a whole run.

### Sequence

```
main thread                       monitor thread (daemon)
-----------                       -----------------------
monitor = Monitor(kpis, ev)
monitor.start() ----------------> _monitor() loop starts
run_giulia()  (blocks)            ... poll KPIs every poll_interval s ...
  config.run()                    ... poll KPIs every poll_interval s ...
  giulia_main(args, config)       ... poll KPIs every poll_interval s ...
  finally: stop_event.set() -----> loop sees stop_event, breaks
monitor_thread.join()  <--------- thread terminates
monitor.get(...)                  (read cached values)
```

---

## 2. The `Monitor` class

The monitor lives in [`giulia/outputs/monitor.py`](../../giulia/outputs/monitor.py).
It is placed under `giulia/outputs` because it reads simulation *outputs*
(the `Saveable` values).

```python
class Monitor:
    def __init__(self, kpis: list[str], stop_event: threading.Event, poll_interval: float = 1.0):
        self._kpis = kpis
        self._stop_event = stop_event
        self._poll_interval = poll_interval
        self._values: dict[str, Any] = {}
        # share_registry: bind the monitor thread to this thread's Saveable registry (§1.1)
        self._thread = threading.Thread(
            target=Saveable.share_registry(self._monitor), name='monitor', daemon=True
        )

    def start(self) -> threading.Thread:
        self._thread.start()
        return self._thread

    def _monitor(self) -> None:
        while not self._stop_event.is_set():
            for kpi in self._kpis:
                value = Saveable.get(kpi)
                self._values[kpi] = value
                print(f'{kpi}: {value}')
            if self._stop_event.wait(timeout=self._poll_interval):
                break

    def get(self, kpi: Optional[str] = None):
        if kpi is not None:
            return self._values.get(kpi)
        if len(self._values) == 1:
            return next(iter(self._values.values()))
        return dict(self._values)
```

> The listing above is trimmed to the threading core. The real class also
> **validates** the requested KPIs (in `__init__` and when the run ends), which
> is covered in [§5](#5-validating-kpis-and-discovering-keys).

### Constructor — `Monitor(kpis, stop_event, poll_interval=1.0)`

- `kpis`: the list of KPI keys to watch, each in the `Saveable.get` format
  `<ClassName>.<variable_name>` (for example
  `Cell_Selection_SSB.best_serving_cell_ID_per_ue`).
- `stop_event`: the shared `threading.Event` used to stop the loop. It is passed
  **in** rather than created internally so that `main.py` keeps ownership of the
  event and can set it from the simulation's `finally` block. This keeps
  `Monitor` decoupled and reusable.
- `poll_interval`: how often, in seconds, the loop reads and prints the KPIs.
  Defaults to `1.0`. It is stored on the instance as `self._poll_interval` and
  passed to `stop_event.wait(timeout=self._poll_interval)` in `_monitor()`.
  Because that wait is interruptible, raising this value only changes the print
  cadence — it does **not** delay shutdown.

The constructor only **builds** the thread — it does not start it. The thread is
created as a `daemon`, so it never keeps the process alive on its own.

### `start()`

Starts the background thread and returns it. Returning the `Thread` lets the
caller `join()` on it later:

```python
monitor_thread = monitor.start()
...
monitor_thread.join()
```

### `_monitor()` — the polling loop

This is the function that actually runs on the background thread. Every cycle it:

1. Iterates over every KPI in `self._kpis`.
2. Reads the latest value with `Saveable.get(kpi)`.
3. Caches it in `self._values[kpi]` (so `get()` can return it later).
4. Prints it.

Then `self._stop_event.wait(timeout=self._poll_interval)` pauses for up to
`poll_interval` seconds (1 second by default). It returns `True` immediately if
the event is set during the wait, in which case the loop breaks at once instead
of waiting out the full interval — this makes shutdown responsive regardless of
the configured interval.

### Why a single thread for all KPIs

All KPIs are polled in one loop, not one thread per KPI. This is intentional:

- `Saveable.get()` is a cheap in-memory lookup, not a slow/blocking operation,
  so parallelism brings no benefit.
- Because of Python's GIL, multiple threads doing `get` + `print` would not run
  in parallel anyway.
- A single loop gives a **consistent snapshot**: all KPIs are read in the same
  cycle, at the same interval.
- Multiple threads would interleave their `print` output and complicate
  shutdown (you would have to `join` each one).

A thread per KPI would only make sense if each KPI's retrieval were genuinely
independent *and* blocking/slow (for example, a network call per KPI) — which is
not the case here.

---

## 3. The `get()` method

`get()` reads from the `self._values` cache that `_monitor()` keeps up to date.
It has three behaviours:

```python
def get(self, kpi: Optional[str] = None):
    if kpi is not None:
        return self._values.get(kpi)              # case 1
    if len(self._values) == 1:
        return next(iter(self._values.values()))  # case 2
    return dict(self._values)                     # case 3
```

| Call | Condition | Returns |
|------|-----------|---------|
| `get('SomeClass.var')` | a KPI name is passed | the latest cached value of **that** KPI (or `None` if not read yet) |
| `get()` | exactly **one** value is cached | that single value directly |
| `get()` | any other number of values is cached | a `dict` mapping `{kpi: value}` |

Notes:

- The inner `self._values.get(kpi)` is the standard **dict** `.get` — distinct
  from the `Monitor.get` method that wraps it. Using the dict's `.get` returns
  `None` for a not-yet-read KPI instead of raising `KeyError`.
- Case 3 returns `dict(self._values)`, a **copy**, so callers cannot mutate the
  monitor's internal state while the loop keeps writing to it.
- The condition is `len(self._values) == 1` — how many values are **cached**, not
  how many KPIs were requested. The two agree once the loop has completed a cycle,
  because a cycle reads every requested key. They do not agree **before the first
  poll**, or if `start()` was never called: the cache is empty, so even a
  single-KPI monitor returns `{}` rather than `None`.

> **Design note.** Returning either a single value (case 2) or a dict (case 3)
> depending on how many KPIs are monitored is convenient but makes the return
> type variable. If a more predictable contract is preferred, `get()` can be
> changed to *always* return a dict, leaving `get('name')` as the way to fetch a
> single value.

---

## 4. Full usage example

This is the wiring in [`main.py`](../../main.py):

```python
import threading
from giulia.inputs import InputConfig
from giulia.outputs.monitor import Monitor
from giulia.outputs.saveable import Saveable

config = InputConfig.load(config_file)
stop_event = threading.Event()


def run_giulia() -> None:
    try:
        giulia_main, args = config.run()  # prepare the run
        giulia_main(args, config)         # launch Giulia
    finally:
        stop_event.set()                  # tell the monitor to stop


if __name__ == '__main__':
    # 1. Start the background monitor.
    #    poll_interval sets how often (seconds) the KPIs are read and printed.
    kpis = ['Cell_Selection_SSB.best_serving_cell_ID_per_ue']
    monitor = Monitor(kpis, stop_event, poll_interval=1.0)
    monitor_thread = monitor.start()

    # 2. Run Giulia on the main thread (blocks until done, then sets stop_event).
    run_giulia()

    # 3. Wait for the monitor to observe stop_event and exit.
    monitor_thread.join()

    # 4. Read the latest cached value(s).
    best_serving_cell_ID_per_ue = monitor.get('Cell_Selection_SSB.best_serving_cell_ID_per_ue')
    print('Last best serving cell ID per UE:', best_serving_cell_ID_per_ue)
    print('Final KPI snapshot:', monitor.get())

    # 5. List every Saveable variable available this run (registry now complete).
    print('Saveable variables:', Saveable.catalog())
```

### Monitoring several KPIs

Add more keys to the list. `get()` with no argument then returns a dict:

```python
kpis = [
    'Cell_Selection_SSB.best_serving_cell_ID_per_ue',
    'MyClass.result',
]
monitor = Monitor(kpis, stop_event)
monitor.start()
...
monitor.get()  # -> {'Cell_Selection_SSB.best_serving_cell_ID_per_ue': ..., 'MyClass.result': ...}
monitor.get('MyClass.result')  # -> just that value
```

---

## 5. Validating KPIs and discovering keys

A KPI key is just a string, so a typo (for example `best_serving_cell_ID_per_eu`
instead of `best_serving_cell_ID_per_ue`) would otherwise fail **silently**:
`Saveable.get()` returns `None` for an unknown key, so the monitor would keep
printing `None` forever without telling you why. To prevent this, a helper lets
you **discover** the valid keys, and the monitor **validates** the keys it is
given.

### `Saveable.catalog()`

`Saveable.catalog()` returns every retrievable key, in the
`<ClassName>.<variable_name>` format accepted by `Saveable.get()`:

```python
from giulia.outputs.saveable import Saveable

print(Saveable.catalog())
# ['Cell_Selection_SSB.best_serving_cell_ID_per_ue', 'SINR_CSI_RS.effective_sinr_ue_to_cell_dB', ...]
```

The catalog is built from the **runtime registry**: a Saveable only appears
once it has been instantiated. This matters for *when* you call it — and, because the
registry is per thread, for *where* you call it from ([§1.1](#11-how-the-monitor-thread-sees-the-registry)):

- **Before / during setup**, the registry is still being filled, so the catalog
  is partial (empty at the very start).
- **After the run**, the registry is complete: re-registering a class replaces its entry
  rather than adding one, and nothing is ever dropped outright, so the catalog lists every
  key that was available during the run.
- **From an unrelated thread**, the catalog is empty regardless of the run's state.

That is why, in the example above, `Saveable.catalog()` is called **after**
`monitor_thread.join()` (step 5): at that point the list is complete.

### How the monitor validates

Because the registry is populated only at runtime, the monitor cannot always
validate at construction time. It therefore checks in two places:

| When | Registry state | Behaviour |
|------|----------------|-----------|
| In `__init__` | already populated (monitor created *after* setup, e.g. from a script or a test) | **raises** `ValueError`, listing the available keys — fail fast |
| When the run ends | complete | **warns** for any KPI that was never available (almost certainly a typo) |

In the `main.py` flow the monitor is created *before* the run, so the registry
is empty at that point and the constructor check is skipped; the post-run
warning is what catches a mistyped key there. This two-step design gives fast
feedback when it is possible, while avoiding false positives — it never flags a
key that simply had not been registered *yet*.

#### The one typo neither check reports

Both checks assume the key is at least **shaped** like a KPI. `Saveable.get()` splits
it on `.` and requires exactly two parts, so a key with no dot (`'best_serving_cell'`)
or with two (`'A.b.c'`) raises `ValueError` instead of reading `None`. Raised inside
the polling loop, on the monitor's own thread, that:

- kills the monitor on its **first** cycle — the valid KPIs stop being polled too;
- prints a traceback from `Thread-monitor`, but raises nothing on the main thread;
- never reaches the shutdown warning, and leaves `monitor_thread.join()` returning
  perfectly normally.

So a malformed key looks like "the monitor printed a traceback and then went quiet",
not like the named warning a misspelled *variable* gets. Copy keys out of
`Saveable.catalog()` and this cannot arise.

### Discussion of the `main.py` example

Two lines in [`main.py`](../../main.py) exercise this:

```python
monitor = Monitor(kpis, stop_event, poll_interval=1.0)   # validates the keys (see table above)
...
print('Saveable variables:', Saveable.catalog())         # after join: discover valid keys
```

- The `Monitor(...)` call is where validation is wired in. With the correct key
  `'Cell_Selection_SSB.best_serving_cell_ID_per_ue'`, nothing happens. If the
  class or variable half were mistyped, the run would still complete normally,
  but on shutdown the monitor would log a warning naming the offending key. If
  the `.` itself were lost or doubled, the monitor would instead die on its first
  poll — see [the subsection above](#the-one-typo-neither-check-reports).
- The final `Saveable.catalog()` print is the recommended workflow to **find**
  the right key in the first place: run once, read the printed list, then copy
  the exact key you need into `kpis`. Because the catalog depends on the active
  configuration (only instantiated modules appear), always discover keys from a
  run using the same config you intend to monitor.

---

## 6. Requirements and caveats

- **The KPI must be a registered Saveable.** Each key must match a
  `<ClassName>.<variable_name>` exposed by a `Saveable` subclass. See
  [Saveable Objects](../source/saveable-objects.rst) for how to create one.
  `Saveable.get()` returns `None` if the class has not been instantiated yet or
  the variable does not exist; the monitor surfaces such mistakes through the
  validation described in [§5](#5-validating-kpis-and-discovering-keys).
- **The catalog reflects `expose_variables()`, not every attribute.**
  `Saveable.catalog()` — and therefore the monitor's discovery and validation
  ([§5](#5-validating-kpis-and-discovering-keys)) — lists only the variables each
  class declares in its `expose_variables()`. `Saveable.get()`, by contrast, reads
  the attribute directly with `getattr`, so it can still return a value for an
  attribute that exists but is **not** declared for export — that key simply won't
  appear in the catalog, and the monitor will flag it as "never available". To
  monitor such a value, add it to that class's `expose_variables()`.
- **`get()` from the main thread reads cached values, not live ones, after the
  run.** In the example above, `monitor.get()` is called *after* `run_giulia()`
  returns, so it yields the **last** values polled before shutdown. While Giulia
  is running, the main thread is blocked inside `run_giulia()` and cannot call
  `get()`; live reads would have to come from yet another thread or from a
  simulation callback.
- **The monitor thread is a daemon.** It will not, by itself, keep the process
  alive; always `join()` it after setting `stop_event` to let it finish its
  current cycle cleanly.
- **Build the `Monitor` on the simulation's thread.** It captures that thread's Saveable
  registry when it is constructed ([§1.1](#11-how-the-monitor-thread-sees-the-registry)).
  Constructing it elsewhere shares the wrong registry, and every KPI reads `None`.
- **Rolling your own reader thread?** Wrap its entry point in `Saveable.share_registry(...)`,
  exactly as `Monitor` does — a bare `threading.Thread(target=my_loop)` sees an empty
  registry. See [Saveable Concurrency §5](saveable-concurrency.md#5-reading-a-registry-from-another-thread-saveableshare_registry).
- **A polled value can briefly be `None` mid-episode.** Re-registering a Saveable is two
  steps (unregister, then append), so a poll landing in between finds the class missing.
  The next cycle recovers; for exact data use the exported Parquet.
- **Every key must contain exactly one `.`.** A malformed key raises inside the polling
  loop and takes the whole monitor down with it, silently from the caller's point of view
  ([§5](#the-one-typo-neither-check-reports)).
- **The monitor cannot watch a simulation in another process.** It shares the registry
  *object* with the simulation thread, which only works inside one interpreter. Monitoring
  a run in a separate process would need an explicit IPC channel; read its exported Parquet
  instead. See [Saveable Concurrency §5](saveable-concurrency.md#reading-from-a-process-not-a-thread).
