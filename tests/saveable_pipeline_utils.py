"""
Shared fixtures for the **end-to-end** Saveable pipeline tests.

``tests/test_saveable_registry.py`` pins the registry itself with synthetic ``Saveable``
subclasses. This module is the other half: it runs a **real, minimal Giulia simulation**
so that the two production call sites which read the registry —
``Performance.process`` (per-step collection, ``save_performance.py``) and
``Episode_Performance._save_episodes_performance_parquet`` (export,
``save_episode_performance.py``) — are actually *executed*, and the files they write can
be inspected.

See ``docs/output-saveable/saveable-concurrency.md`` and
``docs/output-saveable/saveable-output.md``.

Design notes
------------
* :func:`run_simulation` mirrors ``examples/giulia_EV.py::main`` — the same objects, in the
  same order — but parameterised so a test can ask for 1 episode / 1 step instead of the
  example's hardcoded 3 steps. **No production code is patched to make this work.**
* Runs are CPU-only (``enable_GPU=False``), with ``save_results=0`` and ``plots=0``, so the
  simulator writes nothing into the repository's ``outputs/``.
* Parquet output goes to a caller-supplied temp directory. NPZ output *ignores*
  ``output.directory`` by design (it always goes to ``outputs/results/<project>/``, see
  ``docs/output-saveable/output-module.md`` §3), so :func:`npz_output_redirected_to`
  redirects the one path helper it uses — that is a test-side redirect of a filesystem
  location, not a change to the export logic under test.
"""

import contextlib
import os
import sys
import threading
import unittest
from typing import Any, Callable, Dict, List, Optional

# Configure Giulia into path
this_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(this_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(1, root_dir)

ROOT_DIR = root_dir

#: Scenario used by most tests: 570 UEs / 57 cells / 19 radio units, ~3 s per episode-step.
DEFAULT_SCENARIO = 'ITU_R_M2135_UMa'
#: A deliberately *different* deployment (10 UEs / 3 cells / 1 site) used by the concurrency
#: tests, so a run that silently picked up the other run's Saveables is impossible to miss.
SMALL_SCENARIO = '3GPPTR36_814_Case_1_single_bs'

# Importing giulia pulls in torch / pyarrow / sionna. If the environment cannot provide them
# there is nothing to test here, so record why and let `requires_giulia` skip with the reason
# instead of erroring out at collection time.
try:  # pragma: no cover - import guard
    import numpy as np
    import pandas as pd

    from giulia.inputs.config.npz import NPZFormat
    from giulia.inputs.config.parquet import ParquetFormat
    from giulia.logger import LogLevel, set_log_level
    from giulia.outputs import save_episode_performance
    from giulia.outputs.saveable import Saveable
    from giulia.tools.tools import free_memory

    GIULIA_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - import guard
    GIULIA_IMPORT_ERROR = f'{type(exc).__name__}: {exc}'


#: Decorator: skip a test (or a whole TestCase) when Giulia cannot be imported.
requires_giulia = unittest.skipIf(
    GIULIA_IMPORT_ERROR is not None,
    f'Giulia could not be imported in this environment ({GIULIA_IMPORT_ERROR}). '
    f'Run these tests in the project conda environment.',
)


def run_in_isolated_context(target: Callable[[], Any]) -> Any:
    """
    Run ``target`` on its own thread — and therefore in its own Saveable registry — and
    return its result (re-raising anything it raised on the calling thread).

    Test classes each run their own simulation, and the registry is **never cleared within a
    context** (``saveable-concurrency.md`` §9). Sharing the runner's context would let one
    class's Saveables — especially the synthetic ones some fixtures inject — show up in a
    later class's ``Saveable.registry()``, making results depend on test order.

    Anything a fixture needs from the registry (``Saveable.get``, ``Saveable.registry``,
    ``registered_labels``) must therefore be captured *inside* ``target``: from the runner's
    thread that registry is invisible, which is exactly the isolation being relied on here.
    """
    box: Dict[str, Any] = {}

    def _wrapper() -> None:
        try:
            box['value'] = target()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread
            box['error'] = exc

    thread = threading.Thread(target=_wrapper, name='giulia-test-run')
    thread.start()
    thread.join()
    if 'error' in box:
        raise box['error']
    return box.get('value')


def quiet_logs() -> None:
    """Silence Giulia's own logging down to ERROR, so test output stays readable."""
    set_log_level(LogLevel.ERROR)


def giulia_class(preset: str):
    """The preset class, resolved exactly like ``examples/giulia_EV.py::get_giulia_class``."""
    if preset == 'GiuliaMfl':
        from giulia.presets.g_multi_frequency_layer_preset import Giulia
    elif preset == 'GiuliaStd':
        from giulia.presets.g_standard_preset import Giulia
    else:
        raise ValueError(f"Invalid preset '{preset}'.")
    return Giulia


class RunResult:
    """Everything a test needs to assert about one finished simulation."""

    def __init__(self, giulia: Any, outputs_obj: Any, project_name: str,
                 output_dir: Optional[str], episodes: int, steps: int,
                 scenario: str, preset: str):
        self.giulia = giulia
        self.outputs_obj = outputs_obj
        self.project_name = project_name
        self.output_dir = output_dir
        self.episodes = episodes
        self.steps = steps
        self.scenario = scenario
        self.preset = preset

    # ---- entity counts, read from the simulation objects (never from the output) ----

    @property
    def ue_count(self) -> int:
        return int(self.giulia.ue_deployment_obj.number_of_ues)

    @property
    def cell_count(self) -> int:
        return int(self.giulia.network_deployment_obj.number_of_cells)

    @property
    def ru_count(self) -> int:
        return int(self.giulia.network_deployment_obj.df_ep['radio_unit_ID'].nunique())

    @property
    def bbu_count(self) -> int:
        return int(self.giulia.network_deployment_obj.df_ep['bbu_ID'].nunique())

    #: Entity count per exploded label, for the labels whose entity has a first-class
    #: accessor on the simulation objects.
    @property
    def entity_counts(self) -> Dict[str, int]:
        return {
            'ue_kpi': self.ue_count,
            'cell_kpi': self.cell_count,
            'ru_kpi': self.ru_count,
            'bbu_kpi': self.bbu_count,
        }

    def dataset(self, label: str) -> str:
        """Path of the single-file Parquet dataset written for ``label``."""
        return dataset_path(self.output_dir, self.project_name, label)

    def read(self, label: str) -> 'pd.DataFrame':
        return read_dataset(self.output_dir, self.project_name, label)

    def schema_sidecar(self, label: str) -> str:
        return schema_sidecar_path(self.output_dir, self.project_name, label)


def run_simulation(
        scenario: str = DEFAULT_SCENARIO,
        preset: str = 'GiuliaStd',
        episodes: int = 1,
        steps: int = 1,
        output_dir: Optional[str] = None,
        fields: Optional[List[str]] = None,
        project_name: Optional[str] = None,
        parquet: bool = True,
        npz: bool = False,
        after_configure: Optional[Callable[[int, Any], None]] = None,
        after_episode: Optional[Callable[[int, Any], None]] = None,
        before_export: Optional[Callable[[], None]] = None,
) -> RunResult:
    """
    Run a **minimal but real** Giulia simulation and export it.

    This is ``examples/giulia_EV.py::main`` with the episode/step counts turned down: it
    builds the same ``Episode_Performance``, constructs the same preset, and calls
    ``configure`` / ``run_simulation`` / ``save_episode_performance`` /
    ``save_episodes_performance_in_file`` in the same order. Every Saveable the run creates
    registers in the **calling thread's** registry, which is what makes this usable from a
    worker thread in the concurrency tests.

    Args:
        scenario: ``scenario_model`` to deploy.
        preset: ``GiuliaStd`` or ``GiuliaMfl``.
        episodes: number of episodes (snapshots).
        steps: simulation steps per episode.
        output_dir: where the Parquet datasets go. Must be a temp directory.
        fields: the ``output.fields`` filter, or ``None`` for "export everything".
        project_name: dataset name prefix; defaults to ``<scenario>_uniform`` as
            ``giulia_EV`` builds it.
        parquet / npz: which output formats to enable.
        after_configure: called as ``(episode_index, g)`` right after ``g.configure()`` —
            the point at which this episode's Saveables have just been (re-)registered.
        after_episode: called as ``(episode_index, g)`` after the episode's buffers have
            been copied into ``Episode_Performance``.
        before_export: called with no arguments immediately before
            ``save_episodes_performance_in_file`` — the moment the exporter reads the
            registry. The stress tests use it to make several runs export *simultaneously*.
    """
    Giulia = giulia_class(preset)

    step_us = 1e6
    max_simulation_time_us = step_us * steps
    project_name = project_name if project_name is not None else f'{scenario}_uniform'

    formats: List[Any] = []
    if parquet:
        # compression='none' + use_partition_cols=False -> one <project>_<label>.parquet file
        # per label, which is the default layout in config.example.yml.
        formats.append(ParquetFormat(enable=True, compression='none',
                                     use_partition_cols=False, rm_uncompressed=False))
    if npz:
        formats.append(NPZFormat(True))

    outputs_obj = save_episode_performance.Episode_Performance(
        episodes,
        steps,
        False,          # regression
        output_dir,
        formats,
        fields,
    )

    # Positional arguments as examples/giulia_EV.py passes them:
    #   (preset, scenario_model, ue_playground_model, ue_distribution, ue_mobility,
    #    link_direction, wraparound, save_results, plots, number_of_episods,
    #    regression, enable_GPU)
    # save_results=0 and plots=0 keep the run from writing to the repository's outputs/.
    g = Giulia(preset, scenario, None, 'uniform', None, 'downlink', None,
               0, 0, episodes, 0, False)

    for episode_index in range(episodes):
        g.configure(episode_index, max_simulation_time_us)
        if after_configure is not None:
            after_configure(episode_index, g)

        for step_index in range(steps):
            g.run_simulation(step_index, step_us)

        outputs_obj.save_episode_performance(episode_index, g)
        if after_episode is not None:
            after_episode(episode_index, g)

    if before_export is not None:
        before_export()

    outputs_obj.save_episodes_performance_in_file(project_name)
    free_memory()

    return RunResult(g, outputs_obj, project_name, output_dir, episodes, steps, scenario, preset)


# --------------------------------------------------------------------------- #
# Reading back what a run produced
# --------------------------------------------------------------------------- #

def dataset_path(output_dir: str, project_name: str, label: str) -> str:
    return os.path.join(output_dir, f'{project_name}_{label}.parquet')


def schema_sidecar_path(output_dir: str, project_name: str, label: str) -> str:
    return os.path.join(output_dir, f'{project_name}_{label}.schema.md')


def read_dataset(output_dir: str, project_name: str, label: str) -> 'pd.DataFrame':
    return pd.read_parquet(dataset_path(output_dir, project_name, label))


def written_labels(output_dir: str, project_name: str) -> set:
    """The labels for which a ``.parquet`` file exists in ``output_dir``."""
    prefix, suffix = f'{project_name}_', '.parquet'
    return {
        name[len(prefix):-len(suffix)]
        for name in os.listdir(output_dir)
        if name.startswith(prefix) and name.endswith(suffix)
    }


def registered_labels() -> Dict[str, List[str]]:
    """
    ``{label: ['<Class>.<var>', ...]}`` for the **current context's** registry.

    This is the same resolution the exporter performs
    (``expose_variable_labels()`` with a fall back to ``label``), so a test can assert on
    the files a run *must* have produced without hardcoding a list that drifts when a KPI
    is re-routed.
    """
    labels: Dict[str, List[str]] = {}
    for saveable in Saveable.registry():
        var_labels = saveable.expose_variable_labels()
        for var in saveable.expose_variables():
            label = var_labels.get(var, saveable.label)
            labels.setdefault(label, []).append(f'{saveable.name}.{var}')
    return labels


def column_family(df: 'pd.DataFrame', base_name: str) -> List[str]:
    """Every column of ``df`` belonging to ``base_name`` (itself, or its ``.<suffix>`` split)."""
    return [c for c in df.columns if c == base_name or c.startswith(base_name + '.')]


def index_columns(label: str) -> List[str]:
    """The index columns the exporter always keeps for a label."""
    from giulia.outputs.save_episode_performance import GENERIC_LABEL, _entity_id_column
    if label == GENERIC_LABEL:
        return ['episode_idx', 'step_idx']
    return ['episode_idx', 'step_idx', _entity_id_column(label)]


@contextlib.contextmanager
def npz_output_redirected_to(directory: str):
    """
    Send the NPZ export into ``directory`` instead of the repository's ``outputs/results/``.

    The NPZ writer deliberately ignores ``output.directory`` and resolves its path through
    ``giulia.fs.results_file`` (documented in ``docs/output-saveable/output-module.md`` §3).
    Tests must not write into the repo, so this swaps that one path helper — the export
    logic itself is untouched and fully exercised.
    """
    module = save_episode_performance
    original = module.results_file

    def _results_file(project_name: str, file_name: str) -> str:
        target = os.path.join(directory, 'results', project_name)
        os.makedirs(target, exist_ok=True)
        return os.path.join(target, file_name)

    module.results_file = _results_file
    try:
        yield
    finally:
        module.results_file = original


@contextlib.contextmanager
def captured_warnings(module) -> Any:
    """
    Collect the messages ``module`` sends to ``giulia.logger.warning``.

    ``monitor.py`` does ``from giulia.logger import warning``, so the name has to be
    replaced on the *importing* module, not on ``giulia.logger``.
    """
    messages: List[str] = []
    original = module.warning

    def _warning(*args: Any, **kwargs: Any) -> None:
        messages.append(' '.join(str(a) for a in args))

    module.warning = _warning
    try:
        yield messages
    finally:
        module.warning = original
