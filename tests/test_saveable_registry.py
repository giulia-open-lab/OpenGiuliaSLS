"""
Regression tests for the per-execution-context Saveable registry.

Two properties must hold at the same time, and they pull in opposite directions:

1. **Isolation** — two simulations running on two threads must not see, or evict,
   each other's Saveables. This is what the registry was moved off a process-global
   list for.
2. **Opt-in sharing** — a *helper* thread that only reads outputs (the ``Monitor``)
   must still see the registry of the thread that created it. A naive per-thread
   registry silently breaks this: every ``Saveable.get`` returns ``None``.

See docs/output-saveable/saveable-concurrency.md.
"""

import os
import sys
import threading
import unittest
from typing import List

# Configure Giulia into path
this_dir = os.path.dirname(__file__)
root_dir = os.path.join(this_dir, '..')
sys.path.insert(1, root_dir)

from giulia.outputs.saveable import Saveable


class _Kpi(Saveable):
    """A minimal Saveable exposing one attribute."""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def expose_variables(self) -> List[str]:
        return ['value']


class _OtherKpi(_Kpi):
    """A second class name, so both can be registered at once."""


def _run_in_thread(target):
    """Run ``target`` on its own thread (i.e. its own context) and return its result."""
    box = {}

    def _wrapper():
        box['result'] = target()

    thread = threading.Thread(target=_wrapper)
    thread.start()
    thread.join()
    return box['result']


class SaveableRegistryTests(unittest.TestCase):

    def test_registry_is_isolated_between_threads(self):
        """A Saveable registered on one thread is invisible to another."""
        _Kpi('main-thread')

        def other_thread():
            # Fresh context: nothing registered here yet.
            before = Saveable.catalog()
            _Kpi('other-thread')
            return before, Saveable.get('_Kpi.value')

        before, other_value = _run_in_thread(other_thread)

        self.assertEqual(before, [], 'a new thread must start with an empty registry')
        self.assertEqual(other_value, 'other-thread')
        # The other thread's instance must not have replaced ours.
        self.assertEqual(Saveable.get('_Kpi.value'), 'main-thread')

    def test_concurrent_registration_does_not_evict(self):
        """
        Simultaneous runs keep their own entries.

        This is the original failure: ``__init__`` calls ``_unregister(name)``, which on a
        shared list dropped the *other* simulation's instance of the same class.
        """
        started = threading.Barrier(3)
        registered = threading.Barrier(3)
        results = {}

        def simulation(tag):
            def run():
                started.wait()
                _Kpi(tag)
                registered.wait()          # both instances now exist concurrently
                results[tag] = Saveable.get('_Kpi.value')
            return run

        threads = [threading.Thread(target=simulation(tag)) for tag in ('sim-a', 'sim-b')]
        for thread in threads:
            thread.start()
        started.wait()
        registered.wait()
        for thread in threads:
            thread.join()

        self.assertEqual(results, {'sim-a': 'sim-a', 'sim-b': 'sim-b'})

    def test_share_registry_exposes_the_creating_threads_registry(self):
        """A wrapped worker sees the creator's registry, including later registrations."""
        _Kpi('before-worker')
        seen = {}
        ready = threading.Event()
        done = threading.Event()

        def worker():
            seen['at_start'] = Saveable.get('_Kpi.value')
            ready.set()
            done.wait(timeout=5)
            # Registered by the main thread *after* the worker started.
            seen['later'] = Saveable.get('_OtherKpi.value')
            seen['catalog'] = sorted(Saveable.catalog())

        thread = threading.Thread(target=Saveable.share_registry(worker))
        thread.start()
        ready.wait(timeout=5)
        _OtherKpi('added-later')
        done.set()
        thread.join(timeout=5)

        self.assertEqual(seen['at_start'], 'before-worker')
        self.assertEqual(seen['later'], 'added-later',
                         'the shared registry must stay live, not be a snapshot')
        self.assertIn('_OtherKpi.value', seen['catalog'])

    def test_unwrapped_worker_sees_nothing(self):
        """The trap ``share_registry`` exists to avoid, pinned so it stays documented."""
        _Kpi('main-thread')
        self.assertIsNone(_run_in_thread(lambda: Saveable.get('_Kpi.value')))

    def test_enable_flag_is_per_context(self):
        """Disabling collection in one run must not silence another's."""
        Saveable.disable()
        try:
            self.assertFalse(Saveable.is_enabled())
            self.assertTrue(_run_in_thread(Saveable.is_enabled),
                            'the enable flag must not leak across simulations')
        finally:
            Saveable.enable()

    def test_share_registry_also_shares_the_enable_flag(self):
        """Registry and flag live in one state object, so both propagate together."""
        Saveable.disable()
        try:
            wrapped = Saveable.share_registry(Saveable.is_enabled)
            box = {}
            thread = threading.Thread(target=lambda: box.update(enabled=wrapped()))
            thread.start()
            thread.join()
            self.assertFalse(box['enabled'])
        finally:
            Saveable.enable()

    def test_latest_instance_wins_within_a_context(self):
        """Re-creating a Saveable each episode must track the newest instance."""
        _Kpi('episode-0')
        _Kpi('episode-1')
        self.assertEqual(Saveable.get('_Kpi.value'), 'episode-1')
        self.assertEqual([obj.name for obj in Saveable.registry()].count('_Kpi'), 1)

    def test_removed_module_globals_raise_a_helpful_error(self):
        """``saveables`` used to be importable and would now be silently empty."""
        import giulia.outputs.saveable as saveable_module

        for name in ('saveables', 'saveable_enabled'):
            with self.assertRaises(AttributeError) as ctx:
                getattr(saveable_module, name)
            self.assertIn('per execution context', str(ctx.exception))

    def setUp(self):
        # Each test starts from a clean registry in this (the runner's) context.
        Saveable.enable()
        Saveable.registry().clear()


if __name__ == '__main__':
    unittest.main()
