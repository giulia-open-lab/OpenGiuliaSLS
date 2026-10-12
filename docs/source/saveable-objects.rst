Saveable Objects
================

What is a Saveable?
-------------------

When Giulia runs a simulation, many internal classes perform calculations — computing distances,
channel gains, signal quality metrics, and so on. By default, those results stay inside each class
and are not directly accessible from outside the simulator.

A **Saveable** is a way to "mark" a class so that Giulia knows about it and can:

- **Save its outputs to a file** (Parquet or NPZ) after the simulation finishes.
- **Read its current value at any time** while the simulation is still running, from another thread
  or script (live monitoring).

Each simulation gets its **own** registry of Saveables, so several simulations can run in parallel
without interfering. Reading from another thread therefore takes one extra step — see
`Concurrency: which registry am I reading?`_.

If you are adding a new calculation to Giulia and you want its results to appear in the output files,
you make that class a Saveable.


When would I want to create one?
---------------------------------

You would create a Saveable class when:

- You add a new module to the simulator and want its computed variables included in the output data files.
- You want to monitor a specific variable in real time while the simulation runs (for example, to
  print it to the console or feed it into an external dashboard).
- You need to reference a class's output from a custom analysis script by name, without modifying
  the core simulator code.


Basic example
-------------

Suppose you have written a class that computes some custom metric:

.. code-block:: python

    class MyMetric:
        result: float

        def __init__(self):
            self.result = 0.0

        def compute(self):
            self.result = 42.0  # your actual calculation here

To make this class a Saveable, you need to make three small changes:

.. code-block:: python

    from giulia.outputs.saveable import Saveable
    from typing import List

    class MyMetric(Saveable):       # 1. Inherit from Saveable
        result: float

        def __init__(self):
            super().__init__()      # 2. Call the parent constructor so Giulia registers this class
            self.result = 0.0

        def compute(self):
            self.result = 42.0

        def expose_variables(self) -> List[str]:
            return ["result"]       # 3. Tell Giulia which attributes to expose

That is all. Once ``MyMetric`` is instantiated during the simulation, Giulia will:

- Register it automatically under the name ``MyMetric``.
- Include ``MyMetric.result`` in the output files at the end of each episode.
- Allow you to read ``MyMetric.result`` live during the run (see below).


Reading a Saveable value at runtime
------------------------------------

You can read the current value of any registered Saveable variable from anywhere in your code
using ``Saveable.get()``. The key has the format ``<ClassName>.<variable_name>``:

.. code-block:: python

    from giulia.outputs.saveable import Saveable

    value = Saveable.get('MyMetric.result')
    print(value)  # prints 42.0 (or whatever the latest computed value is)

This is particularly useful for live monitoring. For example, you can poll a variable in a
background thread while the simulator runs on the main thread:

.. code-block:: python

    import threading
    from giulia.outputs.saveable import Saveable

    stop_event = threading.Event()

    def monitor():
        while not stop_event.is_set():
            value = Saveable.get('MyMetric.result')
            print('Current result:', value)
            if stop_event.wait(timeout=1):
                break

    # share_registry() lets the background thread read the registry of the thread that
    # runs the simulation. Without it the thread starts with an empty registry of its
    # own and every get() returns None — see "Concurrency" below.
    monitor_thread = threading.Thread(target=Saveable.share_registry(monitor), daemon=True)
    monitor_thread.start()

    # ... run simulation on the main thread ...

    stop_event.set()
    monitor_thread.join()

``Saveable.get()`` returns ``None`` if the class has not been instantiated yet, or if the
variable name does not exist. The key itself must be well formed, though: it is split on
``.`` and must yield exactly two parts, so ``'MyMetric'`` or ``'a.b.c'`` raises a
``ValueError`` rather than returning ``None``.

.. important::

   ``Saveable.get()`` reads the registry of the **calling** thread. A background thread that
   is not wrapped with ``Saveable.share_registry`` has an empty registry, so every lookup
   returns ``None`` — with no error to tell you why. Giulia's built-in
   `Monitor <../output-saveable/SaveAble_monitor.md>`_ already does this wrapping for you.


Multiple instances of the same class
--------------------------------------

Giulia identifies Saveables by **class name**, so only one instance of a given class is registered
at a time. If you instantiate the same class again, the **most recently created** instance replaces
the previous one in the registry. This is deliberate: the simulation re-creates its Saveables every
episode (in ``configure()``), and the registry must point at the *current* episode's objects — not
the first episode's — otherwise collected data (and the per-entity row counts of exploded files)
would be frozen at episode 0.

If you need to save outputs from two separate instances **at the same time**, define thin subclasses
for each so they have distinct class names:

.. code-block:: python

    class MyMetric(Saveable):
        ...

    class MyMetricA(MyMetric):
        pass

    class MyMetricB(MyMetric):
        pass

    metric_a = MyMetricA()   # registered as 'MyMetricA'
    metric_b = MyMetricB()   # registered as 'MyMetricB'

You can then reference them as ``MyMetricA.result`` and ``MyMetricB.result`` independently.


Technical details
-----------------

This section describes how the Saveable system works internally.

The registry
~~~~~~~~~~~~

The module ``giulia.outputs.saveable`` keeps the registry in a
`ContextVar <https://docs.python.org/3/library/contextvars.html>`_, so there is **one registry per
execution context** — in practice, one per thread that runs a simulation, created on first access
and kept for the life of that thread. Read it with:

.. code-block:: python

    Saveable.registry()    # the Saveables registered in the calling context

Every time a ``Saveable`` subclass is instantiated and ``register=True`` (the default), the
``__init__`` method registers ``self`` under its class name, **replacing** any earlier instance of
the same class (``Saveable._unregister`` drops the old one first). This keeps exactly one entry per
class name, and ensures the registry always tracks the **most recently created** instance — the one
the current episode is actually using, since the simulation re-creates its Saveables each episode.

.. note::

   Older code imported a module-level list, ``from giulia.outputs.saveable import saveables``.
   That global was **removed** when the registry became per-context; importing it now raises an
   ``AttributeError`` pointing at ``Saveable.registry()``. See
   `Concurrency: which registry am I reading?`_.

Abstract interface
~~~~~~~~~~~~~~~~~~

``Saveable`` is an abstract base class (using ``ABCMeta``). Subclasses **must** implement:

``expose_variables() -> List[str]``
    Returns the list of attribute names that should be read from this object. This list must be
    constant — it is called multiple times during the run.

Subclasses **may** optionally override:

``variable_types() -> Dict[str, pa.DataType]``
    Maps variable names to explicit `PyArrow <https://arrow.apache.org/docs/python/>`_ data types.
    If not overridden, types are inferred automatically. Use this when automatic inference fails
    (for example, for arrays with a specific element type).

``label -> str`` (property)
    Returns a string that classifies the Saveable for output grouping, deciding which Parquet
    file its variables go to:

    - ``'generic'`` — the wide catch-all table (default), one row per ``(episode, step)``.
    - **any other value** — an *exploded* per-entity file: one row per entity, with an entity-id
      column added automatically. The column name is derived from the label (``'ue_kpi'`` →
      ``ue_id``, ``'cell_kpi'`` → ``cell_id``, ``'ru_kpi'`` → ``ru_id``, ``'bbu_kpi'`` →
      ``bbu_id``). New per-entity files appear automatically — no registry to edit.

    It is strongly recommended to set this to a proper per-entity label when appropriate.
    Variables routed to an exploded label must be entity-aligned on their first axis: a 1-D
    array (one scalar per entity), or a 2-D array (one row per entity, whose second axis is
    split into per-element columns ``<Class>.<var>.0``, ``.1``, …).

``expose_variable_labels() -> Dict[str, str]`` (method)
    Optional per-variable routing: maps each variable from ``expose_variables()`` to its own label.
    Use it when a single class must feed *different* files (e.g. a per-UE metric to ``'ue_kpi'``
    and a per-cell metric to ``'cell_kpi'``), which the per-class ``label`` alone cannot express.
    Variables not present in the mapping fall back to ``label`` (and a warning is logged, to catch
    a KPI silently landing in ``generic``). By default it maps every variable to ``label``.

    See `Saveable Output & Export <../output-saveable/saveable-output.md>`_ for the full output model (files,
    constraints, the ``fields`` filter, and how mismatched-length columns are handled).

``expose_variable_column_prefixes() -> Dict[str, str]`` (method)
    Optional naming for the expanded sub-columns of a **2-D** variable exported to an exploded
    label. By default the split columns get a numeric suffix (``<Class>.<var>.0``, ``.1``, …);
    return ``{variable_name: prefix}`` to insert a label before the index — e.g.
    ``{"rsrp_ue_to_cell_dBm": "cell"}`` produces ``<Class>.rsrp_ue_to_cell_dBm.cell0``,
    ``.cell1``, …. Has no effect on 1-D variables or on ``generic`` columns.

``expose_variable_descriptions() -> Dict[str, str]`` (method)
    Optional human-readable description for each variable from ``expose_variables()``. Return
    ``{variable_name: description}`` — e.g. ``{"rsrp_ue_to_cell_dBm": "downlink RSRP per UE and
    cell (dBm)"}``. These descriptions are surfaced in the generated
    `Saveable Catalog <../output-saveable/saveable-catalog.md>`_ as an indented sub-bullet below
    each ``<Class>.<variable>`` key, so a reader can tell what a KPI means without opening the
    code. By default it returns an empty mapping; variables left out simply have no description
    rendered.

Value retrieval
~~~~~~~~~~~~~~~

``get_output(variable_name)`` calls ``getattr(self, variable_name)`` and returns ``None`` on
``AttributeError``. The static ``Saveable.get(key)`` parses the ``<class>.<variable>`` key,
finds the matching object in the **calling context's** registry, and delegates to ``get_output()``.
It returns ``None`` both for an unknown key and for an empty registry, so see
`Concurrency: which registry am I reading?`_ before calling it from another thread.

Integration with episode saving
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the end of each simulation episode, ``Episode_Performance`` iterates over the registry of the
thread it runs on (the same thread that ran the simulation, so it sees exactly that run's
Saveables). For each Saveable it:

1. Calls ``expose_variables()`` to know which attributes to collect.
2. Calls ``variable_types()`` to build the PyArrow schema.
3. Uses ``expose_variable_labels()`` (falling back to ``label``) to decide which Parquet file each
   variable goes to: the wide ``generic`` table, or a per-entity exploded file.
4. Reads the current attribute values and writes them to the output file(s).

For any exploded label (anything other than ``'generic'``), each variable in ``expose_variables()`` must be
entity-aligned on its first axis, whose length equals the number of entities for that label. A 1-D array gives one
scalar per row; a 2-D array gives one row per entity with its second axis split into per-element columns
(``<Class>.<var>.0``, ``.1``, … — optionally prefixed via ``expose_variable_column_prefixes()``). Each row is automatically
indexed with the entity-id column (``ue_id`` / ``cell_id`` / ``ru_id`` / …) so results trace back to the individual
entity. ``'generic'`` variables have no such constraint and are written as a single value (or array) per
``(episode, step)`` without an entity index column.

Enabling and disabling registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can pause registration (for example, during a dry-run or unit test) using:

.. code-block:: python

    Saveable.disable()   # new instances are not registered
    Saveable.enable()    # resume registration

The flag lives alongside the registry, so it is **per execution context** too: disabling
collection in one simulation does not silence another one running in parallel.

.. warning::

   Within one context, ``enable()`` / ``disable()`` toggle **a single boolean**; there is no
   per-call stack. If you run a *nested* or *throwaway* simulation inside another one **on the
   same thread** (for example the cell-pairing sub-run,
   ``cell_pairing.PairingCell._init_and_run_giulia_pairing``, which calls ``Saveable.disable()``
   so its internal objects are not collected), the inner run must **save and restore** the outer
   run's state — otherwise it leaks its disabled state back to the parent and the parent's KPI
   Saveables silently fail to register, producing empty KPI Parquets. The correct pattern is:

   .. code-block:: python

       parent_enabled = Saveable.is_enabled()
       # ... run the nested/throwaway simulation (which may disable registration) ...
       Saveable.enable() if parent_enabled else Saveable.disable()

   Note also that a ``Simulation_Config``'s ``enable_saveable`` attribute is set from
   ``Saveable.is_enabled()`` at construction time, so it reflects the flag of the context that
   built it, not necessarily the value passed to its constructor — do not rely on it to restore
   another context's state.


Concurrency: which registry am I reading?
------------------------------------------

Because the registry is per execution context, *where* you call ``Saveable.get()`` /
``Saveable.catalog()`` / ``Saveable.registry()`` from decides what you see.

- **On the thread running the simulation** — you see that simulation's Saveables. This covers the
  normal cases: the exporters, and code called from inside the run.
- **On a thread you started yourself** — you see an **empty** registry. ``Saveable.get()`` returns
  ``None`` for every key, exactly as it would for a typo, with no error raised.
- **On a thread wrapped with** ``Saveable.share_registry`` — you see the registry of the thread
  that did the wrapping, live:

  .. code-block:: python

      # called on the simulation thread, before the worker starts
      worker = threading.Thread(target=Saveable.share_registry(my_loop), daemon=True)
      worker.start()

- **In another process** — you see nothing, and no wrapper can change that. ``share_registry``
  passes a *reference* to one in-memory object, which only means anything inside a single
  interpreter; a second process builds its own registry from scratch. Reading a running
  simulation from another process would take an explicit IPC channel (a ``multiprocessing``
  queue, a socket, a file). Read the exported Parquet files instead.

Two simulations running on two threads therefore have completely independent registries and
enable flags, and cannot evict each other's Saveables. Threads are where sharing is *possible
but opt-in*; processes are where it is *impossible without IPC*. Note also that a context
lasts as long as its thread, not as long as a run — a reused worker thread starts its second
simulation on the first one's registry, so give each run a thread of its own.

The full model — why it works this way, the failure it replaced, the rules for
``share_registry``, running simulations in parallel, and the caveats of sharing a live registry
— is documented in
`Saveable Concurrency <../output-saveable/saveable-concurrency.md>`_.


Listing all available Saveables
---------------------------------

The registry is populated at **runtime**, when each class is instantiated during the simulation.
There is no static registry you can inspect without running the simulator.

To see which Saveables are available for a given configuration, you need to **run the simulator**
at least until all the modules have been initialized. Then, from the same thread, inspect the
registry:

.. code-block:: python

    from giulia.outputs.saveable import Saveable

    for obj in Saveable.registry():
        print(f"{obj.name}: {obj.expose_variables()}")

``Saveable.catalog()`` gives the same information already flattened into the
``<class>.<variable>`` keys accepted by ``Saveable.get()``:

.. code-block:: python

    from giulia.outputs.saveable import Saveable

    for key in Saveable.catalog():
        print(key)

.. note::

    Because classes are only registered when they are instantiated, the list depends on your
    simulation configuration. A module that is disabled or unused in your current config will
    not appear. Always run with the same configuration you intend to analyze.

.. note::

    Both calls read the registry of the **calling** thread, so run them on the thread that ran the
    simulation (or one wrapped with ``Saveable.share_registry``) — otherwise they return an empty
    list. See `Concurrency: which registry am I reading?`_.

For a ready-made listing of every variable per preset, without writing any code, see the
generated `Saveable Catalog <../output-saveable/saveable-catalog.md>`_.
