"""
Saveable objects: the registry that lets any class expose its computed variables
so Giulia can export them (Parquet / NPZ) and read them live during a run.

Concurrency
-----------
The registry is **per execution context**, not process-global. It is stored in a
:class:`contextvars.ContextVar`, so every *thread* that runs a simulation gets its own
independent registry and its own enable flag. Concurrent simulations therefore cannot
evict each other's Saveables — the failure this design replaced, where ``__init__``'s
"latest wins" ``_unregister`` deleted entries belonging to a *different* simulation
running in parallel.

The flip side is that a thread which does **not** run the simulation — a live monitor,
for example — starts with an *empty* registry and would see nothing. Such a thread must
be wrapped with :meth:`Saveable.share_registry` to observe the registry of the thread
that created it. See ``docs/output-saveable/saveable-concurrency.md``.

Two boundaries this does *not* draw, both easy to assume:

* **``asyncio`` tasks are not isolated from their creator.** A new thread starts from an
  empty context, but a task *copies* the context that created it — so a task started once
  the state is already bound shares its creator's ``_RegistryState`` and can evict its
  entries. Only a task created before anything touched the registry gets one of its own.
* **A context outlives a run, not just the run.** The state stays bound for the life of
  the thread, so a pooled or reused worker thread starts its second simulation on the
  first one's registry.

The process boundary, by contrast, is absolute: separate processes share no memory, so no
wrapper can reach another process's registry (see ``Saveable.share_registry``).
"""

from abc import ABCMeta, abstractmethod

import functools
from contextvars import ContextVar
from typing import Any, Callable, Dict, List, Optional, TypeVar

import pyarrow as pa

_F = TypeVar('_F', bound=Callable[..., Any])


class _RegistryState:
    """
    The Saveable state owned by a single execution context.

    Holds both the registry and the enable flag, so that :meth:`Saveable.share_registry`
    can hand *one* object to another thread and keep the two consistent.
    """

    __slots__ = ('saveables', 'enabled')

    def __init__(self) -> None:
        self.saveables: List['Saveable'] = []
        self.enabled: bool = True


_state: ContextVar[_RegistryState] = ContextVar('giulia_saveable_state')
"""
Per-context Saveable state. A context that has never touched it gets a fresh, empty
state on first access (see ``_get_state``), which is what isolates concurrent runs.
"""


def _get_state() -> _RegistryState:
    """
    Return the current context's Saveable state, creating it on first use.

    The value is *bound* (not merely defaulted) on first access, so that
    :meth:`Saveable.share_registry` always has a concrete object to hand to another thread.
    """
    try:
        return _state.get()
    except LookupError:
        state = _RegistryState()
        _state.set(state)
        return state


def _get_saveables() -> List['Saveable']:
    """
    The current context's registry list.

    Internal accessor used by the exporters (``save_performance``,
    ``save_episode_performance``). Prefer the public :meth:`Saveable.registry`.
    """
    return _get_state().saveables


class Saveable(metaclass=ABCMeta):
    name: str
    """
    The name of the saveable object, typically the class name.
    """

    def __init__(self, register: bool = True):
        """
        Register this instance as a saveable object.

        Registration targets the **current execution context's** registry (see the module
        docstring), so a simulation running on its own thread never touches another one's.

        Args:
            register: Whether this instance should be registered.
        """
        self.name = type(self).__name__

        if register and Saveable.is_enabled():
            # "Latest wins": drop any previously-registered instance with the same class name
            # and register this one. The simulation re-creates its Saveables every episode
            # (in configure()), so keeping the *first* instance would freeze all collected
            # Saveable data — and the exploded files' per-entity counts — at episode 0.
            Saveable._unregister(self.name)
            _get_saveables().append(self)

    @staticmethod
    def enable():
        """
        Enables the saveable instance collection **for the current execution context**.
        """
        _get_state().enabled = True

    @staticmethod
    def disable():
        """
        Disables the saveable instance collection **for the current execution context**.

        Please note that this won't override any already registered saveables, it will just disable collecting new
        instances.

        Saveables work asynchronously, all the data is fetched when requested, not when the class is instantiated.
        """
        _get_state().enabled = False

    @staticmethod
    def is_enabled() -> bool:
        """
        Check if the saveable instance collection is enabled in the current execution context.

        Returns:
            True if the saveable instance collection is enabled.
        """
        return _get_state().enabled

    @staticmethod
    def registry() -> List['Saveable']:
        """
        The list of Saveables registered in the current execution context.

        This replaces the former module-level ``saveables`` list, which was shared by every
        thread. The returned list is the **live** registry, not a copy: treat it as read-only
        and let ``__init__`` / ``_unregister`` maintain it.

        Returns:
            The registered ``Saveable`` instances, one per class name.
        """
        return _get_saveables()

    @staticmethod
    def share_registry(target: _F) -> _F:
        """
        Wrap ``target`` so that, when it later runs on **another thread**, it sees the registry
        of the thread that called this method.

        A new thread starts with a fresh, empty context, so a background worker that reads
        Saveables — a live monitor, a dashboard feed — would otherwise find nothing and every
        ``Saveable.get`` would return ``None``. Wrapping its entry point binds the child context
        to the *same* state object, so registrations made by the simulation thread are visible
        immediately (the registry is mutated in place, never rebound)::

            worker = threading.Thread(target=Saveable.share_registry(my_loop), daemon=True)
            worker.start()

        Call it on the thread that owns the registry — typically the one that runs (or is about
        to run) the simulation — and *before* starting the thread. The enable flag is shared
        along with the registry, since both live in the same per-context state object.

        It shares memory, so it works between **threads only**. A separate process has its own
        interpreter and its own ``_RegistryState``; reaching it would take an explicit IPC
        mechanism (a queue, a socket, a file). Export to Parquet and read that back instead.

        Args:
            target: The callable that will run on the other thread.

        Returns:
            A wrapper around ``target`` that binds the shared state before delegating to it.
        """
        state = _get_state()

        @functools.wraps(target)
        def _runner(*args: Any, **kwargs: Any) -> Any:
            _state.set(state)
            return target(*args, **kwargs)

        return _runner  # type: ignore[return-value]

    @staticmethod
    def _unregister(name: str) -> None:
        """
        Remove every registered Saveable with the given class name, in the current context.

        Used by ``__init__`` to implement "latest wins" registration: a freshly created
        instance replaces any earlier one of the same class, so the registry tracks the
        object the current episode is actually using (see ``__init__``).
        """
        s = _get_saveables()
        s[:] = [obj for obj in s if obj.name != name]

    @staticmethod
    def exists(name: str) -> bool:
        """
        Check if a saveable object exists.
        Args:
            name: The name of the saveable object.
        Returns:
            True if the saveable object exists.
        """
        for obj in _get_saveables():
            if obj.name == name:
                return True
        return False

    @staticmethod
    def get(key: str) -> Optional[Any]:
        """
        Tries to retrieve a saveable object by its name.

        Reads the **current context's** registry, so a call from a thread that neither ran the
        simulation nor was wrapped with :meth:`share_registry` always returns ``None``.

        An unknown class or variable is *not* an error — it reads as ``None``, exactly like a
        registry that is simply empty. A **malformed** key is: ``key`` is split on ``'.'`` and
        must yield exactly two parts, so ``'NoDot'`` or ``'A.b.c'`` raises ``ValueError``.

        Args:
            key: The name of the saveable object (``<class>.<variable_name>``).
        Returns:
            The actual value of the saveable, or ``None`` if it doesn't exist or doesn't have one.
        Raises:
            ValueError: if ``key`` does not contain exactly one ``'.'``.
        """
        (cl_name, var_name) = key.split('.')
        for obj in _get_saveables():
            if obj.name == cl_name:
                return obj.get_output(var_name)
        return None

    @staticmethod
    def catalog() -> List[str]:
        """
        Lists every retrievable key, in the ``<class>.<variable_name>`` format
        accepted by ``get``, for all Saveables registered in the current execution context.

        The list is built from the runtime registry, so it only reflects the
        Saveables that have already been instantiated. Before the simulation
        starts — or for modules unused in the current configuration — the
        corresponding keys will not appear yet.

        Returns:
            A list of ``<class>.<variable_name>`` keys.
        """
        return [
            f"{obj.name}.{var}"
            for obj in _get_saveables()
            for var in obj.expose_variables()
        ]

    @abstractmethod
    def expose_variables(self) -> List[str]:
        """
        Returns a list of variable names that should be saved from this object.

        This list must be constant. This method may be called multiple times during runtime, which may cause the output
        to be inconsistent.
        """
        raise ValueError("expose_variables() must be implemented by the subclass.")

    # noinspection PyMethodMayBeStatic
    def variable_types(self) -> Dict[str, pa.DataType]:
        """
        Optionally overrides the automatic type detection of the variables.

        Returns:
            A dictionary mapping variable names from ``expose_variables`` to their corresponding PyArrow data types.
        """
        return {}

    # noinspection PyMethodMayBeStatic
    @property
    def label(self) -> str:
        """
        Returns a label for this object. This is used to identify the object type and decide
        which parquet file its variables are exported to.

        Common values: ``generic`` (default, a wide table), or an *exploded* per-entity label
        such as ``ue_kpi`` (one row per UE), ``cell_kpi`` (one row per cell) or ``ru_kpi`` (one
        row per radio unit). **Any** label other than ``generic`` is exported exploded and gets
        its own parquet automatically — there is no registry to edit. The entity-id column is
        derived from the label name (e.g. ``bbu_kpi`` -> a ``bbu_kpi`` file with a ``bbu_id``
        column); see ``_entity_id_column`` in ``giulia.outputs.save_episode_performance``.
        Variables routed to an exploded label must be entity-aligned on their first axis: a 1-D
        array (one scalar per entity), or a 2-D array (one row per entity, whose second axis is
        split into per-element columns ``<Class>.<var>.0``, ``.1``, …). See
        ``docs/output-saveable/saveable-output.md`` §4 for the full shape rules.

        Prefer a per-entity label whenever the data is one-value-per-entity; keep ``generic`` for
        data that is **not** entity-aligned (per-PRB, per-beam, free-form config).
        """
        return 'generic'

    # noinspection PyMethodMayBeStatic
    def expose_variable_labels(self) -> Dict[str, str]:
        """
        Maps each variable returned by ``expose_variables`` to the parquet label it should be
        exported under (``generic``, ``ue_kpi`` or ``cell_kpi``).

        By default every variable inherits this object's class-level ``label``. Override this
        when a single class holds variables that belong to *different* parquets — e.g. a
        per-UE metric that must go to ``ue_kpi`` and a per-cell metric that must go to
        ``cell_kpi`` (the ``label`` property alone cannot express this, as it is per-class).

        Any variable from ``expose_variables`` not present in the returned mapping falls back to
        ``label``. Remember the per-entity constraints: a variable routed to ``ue_kpi`` /
        ``cell_kpi`` must be entity-aligned on its first axis — a 1-D array (one value per UE /
        cell), or a 2-D array (one row per entity, second axis split into ``.0``, ``.1``, … columns).
        """
        return {var: self.label for var in self.expose_variables()}

    # noinspection PyMethodMayBeStatic
    def expose_variable_column_prefixes(self) -> Dict[str, str]:
        """
        Optional naming for the expanded sub-columns of a **2-D** variable exported to an *exploded*
        label (``ue_kpi``, ``cell_kpi``, …).

        When a 2-D variable lands in an exploded file, its first axis becomes the entity (one row
        per UE / cell / …) and its second axis is spread into one column per element. By default
        those columns get a numeric suffix (``<Class>.<var>.0``, ``.1``, …). Return
        ``{variable_name: prefix}`` to insert a label before the index, e.g.::

            def expose_variable_column_prefixes(self) -> Dict[str, str]:
                return {"rsrp_ue_to_cell_dBm": "cell"}   # -> '<Class>.rsrp_ue_to_cell_dBm.cell0', '.cell1', ...

        Has no effect on 1-D variables, on ``generic`` columns, or on variables not listed here.
        """
        return {}

    # noinspection PyMethodMayBeStatic
    def expose_variable_descriptions(self) -> Dict[str, str]:
        """
        Optional human-readable description for each variable returned by ``expose_variables``.

        Returns a dict ``{variable_name: description}`` documenting what each exposed variable
        represents. These descriptions are surfaced in the generated Saveable catalog
        (``docs/output-saveable/saveable-catalog.md``) beneath each ``<Class>.<variable>`` entry,
        so a reader can tell what a KPI means without opening the code.

        By default returns an empty mapping (no descriptions). Override it in a subclass to
        document that class's variables; any variable left out simply has no description rendered.
        """
        return {}

    def get_output(self, variable_name: str) -> Optional[Any]:
        """
        Fetches the actual value of the variable with the given name.
        Args:
            variable_name: The name of the variable.
        Returns:
            The value of the variable with the given name. Or ``None`` if it doesn't exist.
        """
        try:
            return getattr(self, variable_name)
        except AttributeError:
            return None


_REMOVED_MODULE_ATTRIBUTES = {
    'saveables': 'Saveable.registry()',
    'saveable_enabled': 'Saveable.is_enabled()',
}


def __getattr__(name: str) -> Any:
    """
    Fail loudly on the module-level names the per-context registry replaced.

    ``saveables`` and ``saveable_enabled`` used to be process-global. Leaving them in place as
    empty stand-ins would be worse than removing them: readers would silently observe an empty
    registry (or a stale flag) instead of an error, which is exactly how the monitor, the
    ``log_saveable_available_results`` report and the catalog generator broke unnoticed.
    """
    replacement = _REMOVED_MODULE_ATTRIBUTES.get(name)
    if replacement is not None:
        raise AttributeError(
            f"'{__name__}.{name}' was removed: the Saveable registry is now per execution "
            f"context, not process-global. Use {replacement} instead. "
            f"See docs/output-saveable/saveable-concurrency.md."
        )
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
