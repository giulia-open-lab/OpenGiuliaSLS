"""
End-to-end tests for the Saveable pipeline **under concurrency** — the reason the registry
was moved into a ``ContextVar`` in the first place.

``tests/test_saveable_registry.py`` proves the registry primitives are isolated, using
synthetic Saveables and no simulation. This file runs **real Giulia simulations** through
the same situations:

1. ``ParallelSimulationTests``  — two differently-configured simulations on two threads.
2. ``MonitorDuringRealRunTests``— a live ``Monitor`` reading a real KPI during a real run.
3. ``SeparateProcessTests``     — two simulations in two spawned processes.
4. ``CatalogGeneratorTests``    — ``scripts/generate_saveable_catalog.py``, end to end.
5. ``EnableDisableInAngerTests``— the real preset paths that call ``Saveable.disable()``,
   including the cell-pairing sub-run, same-thread and in parallel.

See ``docs/output-saveable/saveable-concurrency.md`` §§5-7.

Run:
    python -m unittest tests.test_saveable_pipeline_concurrency -v

This file is slower than ``tests/test_saveable_pipeline.py``: ``EnableDisableInAngerTests``
needs the cell-pairing sub-run to actually *recompute* (a cache hit would skip the code under
test), which costs about a minute per case.
"""

import contextlib
import io
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from typing import Any, Dict, List, Optional, Tuple

from tests.saveable_pipeline_utils import (
    DEFAULT_SCENARIO,
    ROOT_DIR,
    SMALL_SCENARIO,
    GIULIA_IMPORT_ERROR,
    captured_warnings,
    npz_output_redirected_to,
    quiet_logs,
    registered_labels,
    requires_giulia,
    run_simulation,
    written_labels,
)

if GIULIA_IMPORT_ERROR is None:
    import numpy as np

    from giulia.outputs import monitor as monitor_module
    from giulia.outputs.monitor import Monitor
    from giulia.outputs.saveable import Saveable
    from giulia.outputs.save_episode_performance import GENERIC_LABEL

#: A multi-frequency deployment whose df_ep holds both 'coverage' and 'capacity' cells, which
#: is what makes PairingCell take its recompute branch and run the nested Giulia_Pairing
#: simulation that calls Saveable.disable().
PAIRING_SCENARIO = '3GPPTR38_901_4G_5G_multilayer'

CATALOG_SCRIPT = os.path.join(ROOT_DIR, 'scripts', 'generate_saveable_catalog.py')
CATALOG_MD = os.path.join(ROOT_DIR, 'docs', 'output-saveable', 'saveable-catalog.md')

#: How long a worker thread running a minimal simulation may take before we call it hung.
THREAD_TIMEOUT_S = 900


@contextlib.contextmanager
def temporary_cwd(path: str):
    """
    Run with ``path`` as the working directory.

    ``PairingCell`` caches its results under the *relative* path
    ``./data/pairing_sym_results/<hash>``. Pointing the process at an empty directory
    guarantees a cache miss, which is the only way to reach the recompute branch that runs
    the nested pairing simulation. It also keeps the cache out of the repository.
    """
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield path
    finally:
        os.chdir(previous)


def _thread_result(target):
    """Run ``target`` on its own thread (its own registry) and return its value or re-raise."""
    box: Dict[str, Any] = {}

    def _wrapper():
        try:
            box['value'] = target()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread
            box['error'] = exc

    thread = threading.Thread(target=_wrapper)
    thread.start()
    thread.join(timeout=THREAD_TIMEOUT_S)
    if thread.is_alive():
        raise AssertionError(f'worker thread did not finish within {THREAD_TIMEOUT_S}s')
    if 'error' in box:
        raise box['error']
    return box['value']


@requires_giulia
class ParallelSimulationTests(unittest.TestCase):
    """
    Two simulations, two threads, two *different* deployments, two output directories.

    This is the case the per-context registry exists for. Under the old process-global list
    each run's ``Saveable.__init__`` evicted the other's instance of the same class, so the
    export read whichever object won the last race: columns from the wrong simulation, and
    per-entity row counts from the wrong network.

    The two scenarios are chosen so that contamination cannot hide: 570 UEs / 57 cells versus
    10 UEs / 3 cells. A leak shows up immediately as the wrong row count.

    Note on the C++ backends: ``main.py`` runs Giulia on the OS main thread because
    PyTorch / Mitsuba want to be initialised there. Running on worker threads is fine from the
    Saveable side but inherits whatever those backends impose — if this test ever hangs or
    segfaults, that is the finding, not a flake.
    """

    KPI = 'Cell_Selection_SSB.best_server_rsrp_per_ue_dBm'

    @classmethod
    def setUpClass(cls):
        if GIULIA_IMPORT_ERROR is not None:
            raise unittest.SkipTest(f'Giulia unavailable: {GIULIA_IMPORT_ERROR}')
        quiet_logs()
        cls.tmp_dir = tempfile.mkdtemp(prefix='giulia_saveable_parallel_')

        # Both runs stop here once they have registered their Saveables, so the two registries
        # are provably populated at the same time — the exact overlap that used to corrupt the
        # shared list. A timeout keeps a crashed sibling from hanging the suite.
        registered = threading.Barrier(2, timeout=THREAD_TIMEOUT_S)
        cls.results: Dict[str, Dict[str, Any]] = {}
        cls.errors: Dict[str, BaseException] = {}

        def worker(tag: str, scenario: str):
            def _run():
                output_dir = os.path.join(cls.tmp_dir, tag)
                os.makedirs(output_dir, exist_ok=True)
                snapshot: Dict[str, Any] = {}

                def wait_for_sibling(_episode, _g):
                    registered.wait()

                def capture(_episode, _g):
                    snapshot['kpi'] = np.asarray(Saveable.get(cls.KPI), dtype=float).ravel().copy()
                    snapshot['catalog'] = sorted(Saveable.catalog())
                    snapshot['labels'] = registered_labels()

                try:
                    result = run_simulation(
                        scenario=scenario,
                        output_dir=output_dir,
                        project_name=tag,
                        after_configure=wait_for_sibling,
                        after_episode=capture,
                    )
                    cls.results[tag] = {
                        'output_dir': output_dir,
                        'ue_count': result.ue_count,
                        'cell_count': result.cell_count,
                        'entity_counts': result.entity_counts,
                        'labels': snapshot['labels'],
                        'catalog': snapshot['catalog'],
                        'kpi': snapshot['kpi'],
                        'result': result,
                    }
                except BaseException as exc:  # noqa: BLE001
                    cls.errors[tag] = exc
                    registered.abort()
            return _run

        threads = [
            threading.Thread(target=worker('big', DEFAULT_SCENARIO), name='sim-big'),
            threading.Thread(target=worker('small', SMALL_SCENARIO), name='sim-small'),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=THREAD_TIMEOUT_S)
        cls.still_running = [t.name for t in threads if t.is_alive()]

    @classmethod
    def tearDownClass(cls):
        tmp = getattr(cls, 'tmp_dir', None)
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)

    def test_both_simulations_completed(self):
        self.assertEqual(self.still_running, [],
                         f'threads still running after {THREAD_TIMEOUT_S}s: {self.still_running}')
        if self.errors:
            failures = '\n'.join(f'{tag}: {type(e).__name__}: {e}' for tag, e in self.errors.items())
            self.fail(f'a concurrent simulation raised:\n{failures}')
        self.assertEqual(sorted(self.results), ['big', 'small'])

    def test_each_run_has_its_own_entity_counts(self):
        """The deployments differ, so a leaked registry would show up as a wrong row count."""
        self.assertEqual(self.results['big']['ue_count'], 570)
        self.assertEqual(self.results['big']['cell_count'], 57)
        self.assertEqual(self.results['small']['ue_count'], 10)
        self.assertEqual(self.results['small']['cell_count'], 3)

    def test_per_entity_row_counts_match_each_runs_own_network(self):
        for tag, data in self.results.items():
            for label, entity_count in data['entity_counts'].items():
                if label not in data['labels']:
                    continue
                with self.subTest(run=tag, label=label):
                    df = data['result'].read(label)
                    self.assertEqual(len(df), entity_count)
                    self.assertEqual(sorted(df[df.columns[2]].tolist()), list(range(entity_count)))

    def test_neither_output_directory_contains_the_others_data(self):
        for tag in ('big', 'small'):
            other = 'small' if tag == 'big' else 'big'
            written = os.listdir(self.results[tag]['output_dir'])
            with self.subTest(run=tag):
                self.assertTrue(written)
                self.assertTrue(all(name.startswith(tag + '_') for name in written), written)
                self.assertFalse(any(name.startswith(other + '_') for name in written), written)

    def test_each_export_matches_the_values_its_own_thread_held(self):
        """
        The decisive check: the numbers in each file are the ones that run's *own* Saveable
        instance held at the end of the episode, captured inside that thread.
        """
        for tag, data in self.results.items():
            with self.subTest(run=tag):
                df = data['result'].read('ue_kpi').sort_values('ue_id')
                exported = df[self.KPI].to_numpy(dtype=float)
                np.testing.assert_allclose(exported, data['kpi'], rtol=1e-9)
                self.assertEqual(len(exported), data['ue_count'])

    def test_the_two_runs_produced_genuinely_different_data(self):
        """Guards against both files accidentally holding the same (wrong) numbers."""
        big = self.results['big']['kpi']
        small = self.results['small']['kpi']
        self.assertNotEqual(len(big), len(small))

    def test_both_registries_were_populated_at_the_same_time(self):
        """The barrier in ``after_configure`` means both catalogs coexisted, un-evicted."""
        for tag in ('big', 'small'):
            with self.subTest(run=tag):
                self.assertIn(self.KPI, self.results[tag]['catalog'])


@requires_giulia
class MonitorDuringRealRunTests(unittest.TestCase):
    """
    ``Monitor`` reading a real KPI live, from its background thread, during a real run.

    ``Monitor`` wraps its polling loop in ``Saveable.share_registry`` — without that, the
    monitor thread would start with an empty registry and read ``None`` forever while
    reporting every KPI as "misspelled" (``saveable-concurrency.md`` §11, the Stage-1 trap).

    The whole scenario runs on a worker thread so the registry starts empty, exactly as it is
    when ``main.py`` builds the Monitor before the simulation starts.
    """

    KPI = 'Cell_Selection_SSB.best_serving_cell_ID_per_ue'
    MISSPELLED_KPI = 'Cell_Selection_SSB.best_serving_cell_ID_per_uee'
    EPISODES = 2

    @classmethod
    def setUpClass(cls):
        if GIULIA_IMPORT_ERROR is not None:
            raise unittest.SkipTest(f'Giulia unavailable: {GIULIA_IMPORT_ERROR}')
        quiet_logs()
        cls.tmp_dir = tempfile.mkdtemp(prefix='giulia_saveable_monitor_')
        cls.observations: List[Optional[Any]] = []
        cls.warnings: List[str] = []

        def scenario():
            stop_event = threading.Event()
            # The registry of this (fresh) thread is empty, so Monitor cannot validate eagerly;
            # the unknown key is reported at shutdown instead. That is the path under test.
            assert Saveable.catalog() == [], 'this thread should start with an empty registry'
            monitor = Monitor([cls.KPI, cls.MISSPELLED_KPI], stop_event, poll_interval=0.05)

            def wait_until_monitor_sees(expected, timeout=30.0):
                """Block until the monitor's cache holds ``expected``; returns what it saw."""
                deadline = time.perf_counter() + timeout
                while time.perf_counter() < deadline:
                    observed = monitor.get(cls.KPI)
                    if observed is not None and np.array_equal(np.asarray(observed), expected):
                        return observed
                    time.sleep(0.02)
                return monitor.get(cls.KPI)

            def after_episode(_episode, _g):
                expected = np.asarray(Saveable.get(cls.KPI))
                cls.observations.append(wait_until_monitor_sees(expected))

            with captured_warnings(monitor_module) as messages:
                monitor_thread = monitor.start()
                try:
                    run_simulation(
                        scenario=SMALL_SCENARIO,
                        episodes=cls.EPISODES,
                        steps=1,
                        output_dir=cls.tmp_dir,
                        project_name='monitored',
                        after_episode=after_episode,
                    )
                finally:
                    stop_event.set()
                    monitor_thread.join(timeout=30)
                cls.monitor_alive = monitor_thread.is_alive()
                cls.warnings = list(messages)
            cls.final_snapshot = monitor.get()
            cls.catalog = sorted(Saveable.catalog())

        # The Monitor prints every poll; keep that out of the test report.
        with contextlib.redirect_stdout(io.StringIO()):
            _thread_result(scenario)

    @classmethod
    def tearDownClass(cls):
        tmp = getattr(cls, 'tmp_dir', None)
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_monitor_thread_stopped_cleanly(self):
        self.assertFalse(self.monitor_alive, 'the monitor thread ignored stop_event')

    def test_the_monitor_observed_real_values(self):
        """Non-``None`` for every episode: the shared registry really is visible."""
        self.assertEqual(len(self.observations), self.EPISODES)
        for episode, observed in enumerate(self.observations):
            with self.subTest(episode=episode):
                self.assertIsNotNone(observed, 'the monitor never saw a value for a valid KPI')
                self.assertEqual(len(np.asarray(observed)), 10)

    def test_the_observed_values_changed_across_episodes(self):
        """
        The registry is live, not a snapshot: each episode re-creates the Saveable and the
        monitor follows it (``share_registry`` binds the state object, never rebinds it).
        """
        first, second = (np.asarray(o) for o in self.observations)
        self.assertFalse(np.array_equal(first, second),
                         'the monitor reported identical values for both episodes')

    def test_no_never_available_warning_for_a_valid_kpi(self):
        never_available = [m for m in self.warnings if 'never available' in m]
        self.assertEqual(len(never_available), 1,
                         f'expected exactly one shutdown warning, got: {self.warnings}')
        self.assertNotIn(self.KPI + "'", never_available[0])
        self.assertNotIn(f"'{self.KPI}'", never_available[0])
        self.assertIn(self.MISSPELLED_KPI, never_available[0],
                      'the misspelled KPI should still be reported')

    def test_the_valid_kpi_is_in_the_catalog_and_the_misspelled_one_is_not(self):
        self.assertIn(self.KPI, self.catalog)
        self.assertNotIn(self.MISSPELLED_KPI, self.catalog)

    def test_get_without_arguments_returns_every_monitored_kpi(self):
        self.assertEqual(sorted(self.final_snapshot), sorted([self.KPI, self.MISSPELLED_KPI]))
        self.assertIsNone(self.final_snapshot[self.MISSPELLED_KPI])
        self.assertIsNotNone(self.final_snapshot[self.KPI])


#: Deployments with deliberately different entity fingerprints ``(cells, radio units, BBUs)``.
#: A run that picked up another run's Saveables exports the *other* deployment's row counts,
#: so a leak cannot hide behind a plausible-looking file.
STRESS_DEPLOYMENTS: Tuple[Tuple[str, int, int, int, int], ...] = (
    # scenario,                              ues, cells, radio units, bbus
    (DEFAULT_SCENARIO,                       570,    57,          19,   57),
    ('3GPPTR36_814_Case_1_omni',             570,    19,          19,   19),
    (SMALL_SCENARIO,                          10,     3,           1,    3),
    ('3GPPTR36_814_Case_1_single_bs_omni',    10,     1,           1,    1),
)

#: How many simulations the stress tests run at once, in threads and in processes alike.
#: Eight by default: enough to make eviction pressure real without assuming a big machine.
#: Raise it to lean on a specific box, e.g. ``GIULIA_SAVEABLE_STRESS_RUNS=16``.
STRESS_WIDTH = max(2, int(os.environ.get('GIULIA_SAVEABLE_STRESS_RUNS', '8')))


def _stress_jobs(width: int) -> Tuple[Tuple[str, str, int], ...]:
    """
    ``(tag, scenario, episodes)`` for ``width`` concurrent runs.

    Deployments are cycled, and each full cycle adds one episode. That keeps every job's
    ``(scenario, episodes)`` pair unique, so no two runs may legitimately export the same
    numbers and ``test_no_two_runs_exported_the_same_data`` stays meaningful at any width.
    """
    jobs = []
    for index in range(width):
        scenario = STRESS_DEPLOYMENTS[index % len(STRESS_DEPLOYMENTS)][0]
        episodes = 1 + index // len(STRESS_DEPLOYMENTS)
        jobs.append((f'run{index:02d}', scenario, episodes))
    return tuple(jobs)


STRESS_JOBS: Tuple[Tuple[str, str, int], ...] = _stress_jobs(STRESS_WIDTH)

STRESS_EXPECTED: Dict[str, Tuple[int, int, int, int]] = {
    scenario: (ues, cells, radio_units, bbus)
    for scenario, ues, cells, radio_units, bbus in STRESS_DEPLOYMENTS
}


@requires_giulia
class ThreadStressTests(unittest.TestCase):
    """
    ``STRESS_WIDTH`` simulations at once, on that many threads — the "many runs under
    stress" case. Eight by default; set ``GIULIA_SAVEABLE_STRESS_RUNS`` to push it further.

    ``ParallelSimulationTests`` proves two runs stay apart. This one raises the contention:

    * **eight** contexts alive simultaneously, over **four** different deployments;
    * every run parked on a barrier right after ``configure()``, so all eight registries hold
      an instance of every Saveable class *at the same time* — the maximum amount of
      "latest wins" eviction pressure a shared list could ever be put under;
    * a second barrier immediately before ``save_episodes_performance_in_file``, so all eight
      exporters read their registries **concurrently** too.

    Under the old process-global list this is the worst case: every run × ~50 classes each
    evicting one another until only the last writer survives.
    """

    KPI = 'Cell_Selection_SSB.best_server_rsrp_per_ue_dBm'
    TIMEOUT_S = THREAD_TIMEOUT_S

    @classmethod
    def setUpClass(cls):
        if GIULIA_IMPORT_ERROR is not None:
            raise unittest.SkipTest(f'Giulia unavailable: {GIULIA_IMPORT_ERROR}')
        quiet_logs()
        cls.tmp_dir = tempfile.mkdtemp(prefix='giulia_saveable_stress_')
        cls.results: Dict[str, Dict[str, Any]] = {}
        cls.errors: Dict[str, BaseException] = {}

        parties = len(STRESS_JOBS)
        registered = threading.Barrier(parties, timeout=cls.TIMEOUT_S)
        exporting = threading.Barrier(parties, timeout=cls.TIMEOUT_S)

        def worker(tag: str, scenario: str, episodes: int):
            def _run():
                output_dir = os.path.join(cls.tmp_dir, tag)
                os.makedirs(output_dir, exist_ok=True)
                captured: Dict[str, Any] = {}

                def hold_after_first_configure(episode_index, _g):
                    # Only episode 0: runs have different episode counts, so barriering on
                    # every episode would deadlock the shorter ones.
                    if episode_index == 0:
                        captured['catalog_size'] = len(Saveable.catalog())
                        registered.wait()

                def capture(_episode, _g):
                    captured['kpi'] = np.asarray(Saveable.get(cls.KPI), dtype=float).ravel().copy()

                try:
                    result = run_simulation(
                        scenario=scenario,
                        episodes=episodes,
                        output_dir=output_dir,
                        project_name=tag,
                        after_configure=hold_after_first_configure,
                        after_episode=capture,
                        before_export=exporting.wait,
                    )
                    cls.results[tag] = {
                        'scenario': scenario,
                        'episodes': episodes,
                        'output_dir': output_dir,
                        'fingerprint': (result.ue_count, result.cell_count,
                                        result.ru_count, result.bbu_count),
                        'entity_counts': result.entity_counts,
                        'labels': sorted(l for l in registered_labels() if l != GENERIC_LABEL),
                        'catalog_size': captured['catalog_size'],
                        'kpi': captured['kpi'],
                        'result': result,
                    }
                except BaseException as exc:  # noqa: BLE001
                    cls.errors[tag] = exc
                    # Free every sibling parked on either barrier instead of hanging the suite.
                    registered.abort()
                    exporting.abort()
            return _run

        threads = [threading.Thread(target=worker(tag, scenario, episodes), name=f'stress-{tag}')
                   for tag, scenario, episodes in STRESS_JOBS]
        started = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=cls.TIMEOUT_S)
        cls.elapsed = time.perf_counter() - started
        cls.still_running = [t.name for t in threads if t.is_alive()]

    @classmethod
    def tearDownClass(cls):
        tmp = getattr(cls, 'tmp_dir', None)
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)

    def test_every_simulation_completed(self):
        self.assertEqual(self.still_running, [],
                         f'threads still running after {self.TIMEOUT_S}s: {self.still_running}')
        if self.errors:
            failures = '\n'.join(f'{tag}: {type(e).__name__}: {e}' for tag, e in self.errors.items())
            self.fail(f'{len(self.errors)}/{len(STRESS_JOBS)} concurrent simulations raised:\n{failures}')
        self.assertEqual(len(self.results), len(STRESS_JOBS))

    def test_all_registries_were_populated_simultaneously(self):
        """The barrier guarantees it; this records that each registry was actually full."""
        for tag, data in self.results.items():
            with self.subTest(run=tag):
                self.assertGreater(data['catalog_size'], 20,
                                   f'{tag} held only {data["catalog_size"]} keys at the barrier')

    def test_each_run_kept_its_own_deployment(self):
        """Four different fingerprints across eight runs — a leak swaps one for another."""
        for tag, data in self.results.items():
            with self.subTest(run=tag, scenario=data['scenario']):
                self.assertEqual(data['fingerprint'], STRESS_EXPECTED[data['scenario']])

    def test_every_per_entity_file_has_its_own_row_count(self):
        for tag, data in self.results.items():
            steps, episodes = 1, data['episodes']
            for label, entity_count in data['entity_counts'].items():
                if label not in data['labels']:
                    continue
                with self.subTest(run=tag, label=label):
                    df = data['result'].read(label)
                    self.assertEqual(len(df), entity_count * steps * episodes)

    def test_every_export_matches_the_values_its_own_thread_held(self):
        for tag, data in self.results.items():
            with self.subTest(run=tag):
                df = data['result'].read('ue_kpi')
                last = df[df['episode_idx'] == data['episodes'] - 1].sort_values('ue_id')
                np.testing.assert_allclose(last[self.KPI].to_numpy(dtype=float), data['kpi'],
                                           rtol=1e-9)

    def test_no_two_runs_exported_the_same_data(self):
        """
        The cross-check that does not rely on any single run being self-consistent: all eight
        datasets must be pairwise different. Same-deployment pairs differ because the second
        run of each pair advances one more episode (and therefore one more seed).
        """
        signatures = {}
        for tag, data in self.results.items():
            values = data['kpi']
            signatures[tag] = (len(values), float(np.sum(values)), float(np.sum(values ** 2)))
        self.assertEqual(len(set(signatures.values())), len(signatures),
                         f'two runs exported identical data: {signatures}')

    def test_no_output_directory_holds_another_runs_files(self):
        for tag, data in self.results.items():
            with self.subTest(run=tag):
                written = os.listdir(data['output_dir'])
                self.assertTrue(written)
                self.assertTrue(all(name.startswith(tag + '_') for name in written), written)


def _process_worker(scenario: str, output_dir: str, project_name: str,
                    episodes: int, queue) -> None:
    """
    Entry point for the spawned processes in :class:`SeparateProcessTests`.

    Must be importable at module level: ``spawn`` re-imports this module in the child.
    Reports its own wall-clock window so the test can prove the runs really overlapped
    rather than quietly serialising.
    """
    import time as _time
    started = _time.time()
    try:
        from tests.saveable_pipeline_utils import quiet_logs, registered_labels, run_simulation
        quiet_logs()
        result = run_simulation(scenario=scenario, episodes=episodes, output_dir=output_dir,
                                project_name=project_name)
        df = result.read('ue_kpi')
        last = df[df['episode_idx'] == episodes - 1].sort_values('ue_id')
        kpi = last['Cell_Selection_SSB.best_server_rsrp_per_ue_dBm'].to_numpy(dtype=float)
        queue.put({
            'project_name': project_name,
            'scenario': scenario,
            'episodes': episodes,
            'fingerprint': (result.ue_count, result.cell_count, result.ru_count, result.bbu_count),
            'entity_counts': result.entity_counts,
            'rows': len(df),
            'labels': sorted(l for l in registered_labels() if l != 'generic'),
            'row_counts': {label: len(result.read(label))
                           for label in registered_labels() if label != 'generic'},
            'files': sorted(os.listdir(output_dir)),
            'kpi_signature': (len(kpi), float(kpi.sum()), float((kpi ** 2).sum())),
            'pid': os.getpid(),
            'started': started,
            'finished': _time.time(),
        })
    except BaseException as exc:  # noqa: BLE001
        import traceback
        queue.put({'project_name': project_name, 'error': f'{type(exc).__name__}: {exc}',
                   'traceback': traceback.format_exc(),
                   'started': started, 'finished': _time.time()})


@requires_giulia
class SeparateProcessTests(unittest.TestCase):
    """
    ``STRESS_WIDTH`` simulations in that many **processes** (``spawn``), each with its own
    output directory.

    Separate processes share no memory at all, so the registry cannot be the failure mode
    here — but this is the isolation model the catalog generator relies on
    (``saveable-concurrency.md`` §6), and running the same load through processes that
    :class:`ThreadStressTests` runs through threads answers a different question: does the
    whole pipeline — PyTorch, the exporters, the file writers — survive being run many times
    at once on one machine?

    Same jobs as the thread stress test, so the same "wrong row count / duplicate dataset"
    leaks would be caught if processes shared anything.
    """

    @classmethod
    def setUpClass(cls):
        if GIULIA_IMPORT_ERROR is not None:
            raise unittest.SkipTest(f'Giulia unavailable: {GIULIA_IMPORT_ERROR}')
        cls.tmp_dir = tempfile.mkdtemp(prefix='giulia_saveable_processes_')
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()

        cls.jobs = [(f'proc_{tag}', scenario, episodes)
                    for tag, scenario, episodes in STRESS_JOBS]
        processes = []
        started = time.perf_counter()
        for project_name, scenario, episodes in cls.jobs:
            output_dir = os.path.join(cls.tmp_dir, project_name)
            os.makedirs(output_dir, exist_ok=True)
            process = ctx.Process(target=_process_worker,
                                  args=(scenario, output_dir, project_name, episodes, queue),
                                  name=project_name)
            process.start()
            processes.append(process)

        cls.payloads: Dict[str, Dict[str, Any]] = {}
        for _ in cls.jobs:
            payload = queue.get(timeout=THREAD_TIMEOUT_S)
            cls.payloads[payload['project_name']] = payload
        for process in processes:
            process.join(timeout=THREAD_TIMEOUT_S)
        cls.elapsed = time.perf_counter() - started
        cls.exit_codes = {p.name: p.exitcode for p in processes}
        cls.still_running = [p.name for p in processes if p.is_alive()]

    @classmethod
    def tearDownClass(cls):
        tmp = getattr(cls, 'tmp_dir', None)
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)

    def test_every_process_exited_cleanly(self):
        self.assertEqual(self.still_running, [], f'processes still alive: {self.still_running}')
        failures = {name: p.get('traceback', p.get('error'))
                    for name, p in self.payloads.items() if 'error' in p}
        self.assertEqual(failures, {}, f'{len(failures)}/{len(self.jobs)} processes failed')
        for name, code in self.exit_codes.items():
            with self.subTest(process=name):
                self.assertEqual(code, 0)
        self.assertEqual(len(self.payloads), len(self.jobs))

    def test_the_processes_are_really_separate(self):
        pids = {p['pid'] for p in self.payloads.values()}
        self.assertEqual(len(pids), len(self.jobs))
        self.assertNotIn(os.getpid(), pids)

    def test_the_runs_actually_overlapped(self):
        """
        Guards the test itself: if the processes had queued up one after another this would
        not be a concurrency test at all.
        """
        windows = [(p['started'], p['finished']) for p in self.payloads.values()]
        latest_start = max(start for start, _ in windows)
        earliest_finish = min(finish for _, finish in windows)
        self.assertGreater(earliest_finish, latest_start,
                           f'the runs did not overlap: {sorted(windows)}')

    def test_each_process_exported_its_own_deployment(self):
        for name, payload in self.payloads.items():
            with self.subTest(process=name, scenario=payload['scenario']):
                self.assertEqual(tuple(payload['fingerprint']),
                                 STRESS_EXPECTED[payload['scenario']])
                self.assertEqual(payload['rows'],
                                 payload['fingerprint'][0] * payload['episodes'])

    def test_each_process_wrote_a_complete_set_of_files(self):
        for name, payload in self.payloads.items():
            with self.subTest(process=name):
                produced = {f[len(name) + 1:-len('.parquet')]
                            for f in payload['files'] if f.endswith('.parquet')}
                self.assertEqual(produced, set(payload['labels']))
                for label in payload['labels']:
                    self.assertIn(f'{name}_{label}.schema.md', payload['files'])

    def test_each_per_entity_file_has_its_own_row_count(self):
        for name, payload in self.payloads.items():
            for label, entity_count in payload['entity_counts'].items():
                if label not in payload['labels']:
                    continue
                with self.subTest(process=name, label=label):
                    self.assertEqual(payload['row_counts'][label],
                                     entity_count * payload['episodes'])

    def test_no_two_processes_exported_the_same_data(self):
        signatures = {name: tuple(p['kpi_signature']) for name, p in self.payloads.items()}
        self.assertEqual(len(set(signatures.values())), len(signatures),
                         f'two processes exported identical data: {signatures}')


def _parse_catalog_markdown(text: str, marker: str) -> Dict[str, set]:
    """
    ``{preset: {'<Class>.<variable>', ...}}`` parsed from the **generated** section.

    Only the part below ``marker`` is parsed: the hand-written tutorial above it has its own
    ``##`` headings with backticks (``## Configuration & the link to `config.yml```) that
    would otherwise look like preset sections.

    Sections look like ``## Standard preset — `GiuliaStd``` followed by ``### `Class``` and
    ``- `Class.variable``` bullets (descriptions are indented, so they do not match).
    """
    generated = text.split(marker, 1)[-1]
    catalogs: Dict[str, set] = {}
    current: Optional[str] = None
    for line in generated.splitlines():
        heading = re.match(r'^## .*`([^`]+)`\s*$', line)
        if heading:
            current = heading.group(1)
            catalogs[current] = set()
            continue
        if current is None:
            continue
        entry = re.match(r'^- `([^`]+)`\s*$', line)
        if entry:
            catalogs[current].add(entry.group(1))
    return catalogs


def _emit_isolated_catalog(preset: str, scenario: str, out_path: str) -> set:
    """Run the catalog script's own worker for one preset and read back its keys."""
    import json
    completed = subprocess.run(
        [sys.executable, CATALOG_SCRIPT, '--emit', '--preset', preset,
         '--scenario', scenario, '--out', out_path],
        cwd=ROOT_DIR, capture_output=True, text=True, timeout=THREAD_TIMEOUT_S,
    )
    if completed.returncode != 0:
        raise AssertionError(f'catalog worker for {preset} failed:\n{completed.stdout}\n{completed.stderr}')
    with open(out_path, encoding='utf-8') as handle:
        payload = json.load(handle)
    return {f'{class_name}.{var}'
            for class_name, entry in payload.items()
            for var in entry['variables']}


@requires_giulia
class CatalogGeneratorTests(unittest.TestCase):
    """
    ``scripts/generate_saveable_catalog.py``, run end to end.

    Under a thread-local registry (``saveable-concurrency.md`` §11, Stage 1) this script
    regenerated an **empty** catalog without raising — a failure indistinguishable from
    "the simulation has not started yet". So the test that matters is simply: is the
    generated catalog non-empty, for both presets?

    The generated markdown is written back into ``docs/`` by design, so the file is restored
    byte-for-byte afterwards: a test must not leave the repository modified.
    """

    EXPECTED_CLASSES = ('Cell_Selection_SSB', 'SINR_SSB', 'Power_Consumption',
                        'Beam_Selection', 'BaseStations')

    @classmethod
    def setUpClass(cls):
        if GIULIA_IMPORT_ERROR is not None:
            raise unittest.SkipTest(f'Giulia unavailable: {GIULIA_IMPORT_ERROR}')
        cls.tmp_dir = tempfile.mkdtemp(prefix='giulia_saveable_catalog_')
        cls.original_markdown = None
        if os.path.exists(CATALOG_MD):
            with open(CATALOG_MD, 'rb') as handle:
                cls.original_markdown = handle.read()

        cls.completed = subprocess.run(
            [sys.executable, CATALOG_SCRIPT],
            cwd=ROOT_DIR, capture_output=True, text=True, timeout=THREAD_TIMEOUT_S,
        )
        from scripts import generate_saveable_catalog

        with open(CATALOG_MD, encoding='utf-8') as handle:
            cls.generated_markdown = handle.read()
        cls.catalogs = _parse_catalog_markdown(cls.generated_markdown,
                                               generate_saveable_catalog.GENERATED_MARKER)

        # An independent, single-preset run of the same worker. If the driver had let one
        # preset's Saveables leak into the other's list, the two would not match.
        cls.isolated = {
            preset: _emit_isolated_catalog(preset, scenario,
                                           os.path.join(cls.tmp_dir, f'{preset}.json'))
            for preset, _label, scenario in generate_saveable_catalog.PRESETS
        }

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, 'original_markdown', None) is not None:
            with open(CATALOG_MD, 'wb') as handle:
                handle.write(cls.original_markdown)
        tmp = getattr(cls, 'tmp_dir', None)
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_script_succeeded(self):
        self.assertEqual(self.completed.returncode, 0,
                         f'stdout:\n{self.completed.stdout}\nstderr:\n{self.completed.stderr}')

    def test_the_catalog_is_not_empty_for_either_preset(self):
        self.assertEqual(sorted(self.catalogs), ['GiuliaMfl', 'GiuliaStd'])
        for preset, keys in self.catalogs.items():
            with self.subTest(preset=preset):
                self.assertGreater(len(keys), 20, f'{preset} catalog looks empty: {sorted(keys)[:5]}')
                for class_name in self.EXPECTED_CLASSES:
                    self.assertTrue(any(k.startswith(class_name + '.') for k in keys),
                                    f'{class_name} missing from the {preset} catalog')

    def test_neither_presets_list_is_contaminated_by_the_other(self):
        """
        Each rendered section equals what that preset produces **on its own**.

        With the shipped ``PRESETS`` table both presets use the same scenario and happen to
        expose the same keys, so "the lists differ" would prove nothing. Comparing each
        section against an independent single-preset run does: a leak would make a section a
        strict superset of its isolated run.
        """
        for preset, isolated_keys in self.isolated.items():
            with self.subTest(preset=preset):
                self.assertTrue(isolated_keys, f'the isolated {preset} run produced no keys')
                self.assertEqual(self.catalogs[preset], isolated_keys)

    def test_the_handwritten_tutorial_above_the_marker_is_preserved(self):
        from scripts import generate_saveable_catalog
        marker = generate_saveable_catalog.GENERATED_MARKER
        self.assertIn(marker, self.generated_markdown)
        head = self.generated_markdown.split(marker)[0]
        self.assertIn('# Saveable Catalog', head)
        self.assertIn('## How the catalog is generated', head)

    def test_the_repository_copy_is_restored(self):
        """Sanity: this test file must not leave docs/ modified (checked again in tearDown)."""
        self.assertIsNotNone(self.original_markdown, 'the catalog markdown should exist in the repo')


@requires_giulia
class EnableDisableInAngerTests(unittest.TestCase):
    """
    The real code paths that call ``Saveable.disable()``.

    Two of them ship in ``giulia/``:

    * a preset constructed with ``enable_saveable=False``
      (``g_standard_preset.py`` / ``g_multi_frequency_layer_preset.py``), and
    * ``PairingCell`` recomputing its pairing, which builds a throwaway ``Giulia_Pairing``
      that disables collection so its own RSRP objects do not replace the caller's
      (``giulia/rrc/cell_pairing.py``).

    The pairing case is the interesting one. Within a single context the enable flag is a
    plain boolean with no stack, so the caller has to snapshot and restore it — which
    ``cell_pairing.py`` does in a ``finally``. Without that, every Saveable the outer run
    creates *after* pairing skips registration and the run exports empty KPI parquets
    (``saveable-concurrency.md`` §7).

    A pairing **cache hit** skips the sub-run entirely, so these tests run with the working
    directory pointed at an empty temp dir: ``PairingCell`` caches under the relative path
    ``./data/pairing_sym_results``, so an empty cwd forces the recompute branch.
    """

    @classmethod
    def setUpClass(cls):
        if GIULIA_IMPORT_ERROR is not None:
            raise unittest.SkipTest(f'Giulia unavailable: {GIULIA_IMPORT_ERROR}')
        quiet_logs()

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix='giulia_saveable_disable_')
        self.output_dir = os.path.join(self.tmp_dir, 'out')
        self.cache_root = os.path.join(self.tmp_dir, 'cwd')
        os.makedirs(self.output_dir)
        os.makedirs(self.cache_root)
        Saveable.enable()

    def tearDown(self):
        Saveable.enable()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _assert_pairing_recomputed(self):
        """The sub-run only writes a cache when it actually recomputed."""
        cache_dir = os.path.join(self.cache_root, 'data', 'pairing_sym_results')
        self.assertTrue(os.path.isdir(cache_dir),
                        'no pairing cache was written: the sub-run under test never ran')
        self.assertTrue(os.listdir(cache_dir))

    def test_a_preset_built_with_enable_saveable_false_disables_only_its_own_context(self):
        """
        The cheap path: ``Giulia(..., enable_saveable=False)`` calls ``Saveable.disable()``.

        Constructed on a worker thread, it must leave this thread's flag alone — this is the
        parallel half of §7, using a real preset rather than a bare ``Saveable.disable()``.
        """
        from tests.saveable_pipeline_utils import giulia_class
        Giulia = giulia_class('GiuliaStd')

        def build_disabled():
            Giulia('GiuliaStd', SMALL_SCENARIO, None, 'uniform', None, 'downlink', None,
                   0, 0, 1, 0, False, None, False)
            return Saveable.is_enabled()

        self.assertFalse(_thread_result(build_disabled),
                         'the preset should have disabled collection in its own context')
        self.assertTrue(Saveable.is_enabled(),
                        'a preset built on another thread must not silence this one')

    def test_the_pairing_sub_run_leaves_the_outer_runs_saveables_registered(self):
        """
        Same thread, cold pairing cache: the nested sub-run disables collection and the outer
        run must still export full KPI parquets.

        If ``cell_pairing.py`` stopped restoring the flag, every KPI object built later in
        ``Giulia.configure()`` would skip registration and this run would produce no
        ``ue_kpi`` file at all.
        """
        with temporary_cwd(self.cache_root):
            result = run_simulation(
                scenario=PAIRING_SCENARIO,
                preset='GiuliaMfl',
                output_dir=self.output_dir,
                project_name='pairing',
            )
            labels = sorted(l for l in registered_labels() if l != GENERIC_LABEL)

        self._assert_pairing_recomputed()
        self.assertTrue(Saveable.is_enabled(),
                        'pairing must leave the enable flag exactly as it found it')
        self.assertTrue(labels, 'the outer run registered nothing after the pairing sub-run')

        produced = written_labels(self.output_dir, 'pairing')
        self.assertEqual(produced, set(labels))
        self.assertIn('ue_kpi', produced)

        ue_kpi = result.read('ue_kpi')
        self.assertEqual(len(ue_kpi), result.ue_count)
        data_columns = [c for c in ue_kpi.columns if c not in ('episode_idx', 'step_idx', 'ue_id')]
        self.assertTrue(data_columns, 'the ue_kpi parquet holds only index columns')

        cell_kpi = result.read('cell_kpi')
        self.assertEqual(len(cell_kpi), result.cell_count)

    def test_a_parallel_run_is_unaffected_by_a_pairing_sub_run(self):
        """
        The parallel half: while one thread's pairing sub-run has collection disabled, a
        simulation on another thread must register and export everything.

        Under the process-global flag this is precisely the reported failure — a run that
        never touched pairing exporting empty KPI parquets.
        """
        pairing_started = threading.Event()
        outcomes: Dict[str, Any] = {}

        def pairing_run():
            try:
                pairing_started.set()
                run_simulation(scenario=PAIRING_SCENARIO, preset='GiuliaMfl',
                               output_dir=os.path.join(self.tmp_dir, 'pairing'),
                               project_name='pairing')
                outcomes['pairing'] = 'ok'
            except BaseException as exc:  # noqa: BLE001
                outcomes['pairing'] = exc

        def plain_run():
            try:
                pairing_started.wait(timeout=60)
                flags: List[bool] = []
                result = run_simulation(
                    scenario=SMALL_SCENARIO,
                    output_dir=os.path.join(self.tmp_dir, 'plain'),
                    project_name='plain',
                    after_configure=lambda _e, _g: flags.append(Saveable.is_enabled()),
                    after_episode=lambda _e, _g: flags.append(Saveable.is_enabled()),
                )
                outcomes['plain'] = {
                    'flags': flags,
                    'labels': sorted(l for l in registered_labels() if l != GENERIC_LABEL),
                    'ue_rows': len(result.read('ue_kpi')),
                    'ue_count': result.ue_count,
                    'written': written_labels(os.path.join(self.tmp_dir, 'plain'), 'plain'),
                }
            except BaseException as exc:  # noqa: BLE001
                outcomes['plain'] = exc

        os.makedirs(os.path.join(self.tmp_dir, 'pairing'))
        os.makedirs(os.path.join(self.tmp_dir, 'plain'))

        with temporary_cwd(self.cache_root):
            threads = [threading.Thread(target=pairing_run, name='pairing'),
                       threading.Thread(target=plain_run, name='plain')]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=THREAD_TIMEOUT_S)
            still_running = [t.name for t in threads if t.is_alive()]

        self.assertEqual(still_running, [], f'threads hung: {still_running}')
        for tag in ('pairing', 'plain'):
            if isinstance(outcomes.get(tag), BaseException):
                raise outcomes[tag]
        self._assert_pairing_recomputed()

        plain = outcomes['plain']
        self.assertTrue(all(plain['flags']),
                        'the parallel run saw collection disabled by the pairing sub-run')
        self.assertTrue(plain['labels'], 'the parallel run registered nothing')
        self.assertEqual(plain['ue_rows'], plain['ue_count'])
        self.assertEqual(plain['written'], set(plain['labels']))


if __name__ == '__main__':
    unittest.main()
