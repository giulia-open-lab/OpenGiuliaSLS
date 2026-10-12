"""
End-to-end tests for the Saveable **output pipeline**, run against a real simulation.

``tests/test_saveable_registry.py`` proves the registry behaves, using synthetic Saveables.
This file proves the rest of the chain actually works when a *real* Giulia run fills that
registry: that ``Performance.process`` collects from it every step, that
``Episode_Performance`` reads it at export time, and that the Parquet / NPZ files which come
out are complete and correct.

Covered here (single-process, single-thread):

1. ``SingleRunExportTests``   — one minimal run, Parquet + NPZ, end to end.
2. ``ExplodedFileTests``      — per-entity files: row counts, entity ids, 2-D column splits.
3. ``FieldsFilterTests``      — the ``output.fields`` filter.
4. ``LatestWinsAcrossEpisodesTests`` — the exported data comes from each episode's own objects.
5. ``NewLabelExportTests``    — a brand-new label auto-creates its parquet, and a 2-D variable
   with no ``expose_variable_column_prefixes()`` entry falls back to ``.0/.1/…``.

Concurrency, the monitor, subprocesses and the enable/disable paths live in
``tests/test_saveable_pipeline_concurrency.py``.

Run:
    python -m unittest tests.test_saveable_pipeline -v
"""

import os
import shutil
import tempfile
import unittest
from typing import Dict, List

from tests.saveable_pipeline_utils import (
    DEFAULT_SCENARIO,
    SMALL_SCENARIO,
    GIULIA_IMPORT_ERROR,
    captured_warnings,
    column_family,
    index_columns,
    npz_output_redirected_to,
    quiet_logs,
    registered_labels,
    requires_giulia,
    run_in_isolated_context,
    run_simulation,
    written_labels,
)

if GIULIA_IMPORT_ERROR is None:
    import numpy as np
    import pandas as pd

    from giulia.outputs import save_episode_performance
    from giulia.outputs.saveable import Saveable
    from giulia.outputs.save_episode_performance import GENERIC_LABEL, _entity_id_column


class _MinimalRunFixture(unittest.TestCase):
    """
    A single real run, shared by every test in the subclass.

    Deliberately tiny — one episode, two steps, CPU, no plots — so that a handful of
    assertions can be backed by one ~5 s simulation instead of one simulation each.
    Two steps (not one) so that per-step row counts are actually multiplied by something.
    """

    SCENARIO = DEFAULT_SCENARIO
    PRESET = 'GiuliaStd'
    EPISODES = 1
    STEPS = 2
    FIELDS = None
    NPZ = False
    #: Captured live from the run's own registry, for the provenance check.
    LIVE_KPI = 'Cell_Selection_SSB.best_server_rsrp_per_ue_dBm'

    @classmethod
    def setUpClass(cls):
        if GIULIA_IMPORT_ERROR is not None:
            raise unittest.SkipTest(f'Giulia unavailable: {GIULIA_IMPORT_ERROR}')
        quiet_logs()
        cls.tmp_dir = tempfile.mkdtemp(prefix='giulia_saveable_pipeline_')
        cls.output_dir = os.path.join(cls.tmp_dir, 'parquet')
        os.makedirs(cls.output_dir, exist_ok=True)

        def simulate():
            with npz_output_redirected_to(cls.tmp_dir):
                result = run_simulation(
                    scenario=cls.SCENARIO,
                    preset=cls.PRESET,
                    episodes=cls.EPISODES,
                    steps=cls.STEPS,
                    output_dir=cls.output_dir,
                    fields=cls.FIELDS,
                    npz=cls.NPZ,
                )
            # Everything derived from the registry has to be captured here, inside the run's
            # own context: it is not reachable from the test methods (see
            # run_in_isolated_context).
            labels_to_variables = registered_labels()
            exploded = sorted(l for l in labels_to_variables if l != GENERIC_LABEL)
            return {
                'result': result,
                'labels_to_variables': labels_to_variables,
                'exploded_labels': exploded,
                'entity_count_per_label': {label: cls._entity_count(labels_to_variables[label])
                                           for label in exploded},
                'column_prefixes': {
                    f'{saveable.name}.{variable}': prefix
                    for saveable in Saveable.registry()
                    for variable, prefix in saveable.expose_variable_column_prefixes().items()
                },
                'live_ue_kpi': {
                    name: np.asarray(Saveable.get(name), dtype=float).ravel().copy()
                    for name in (cls.LIVE_KPI,)
                },
            }

        captured = run_in_isolated_context(simulate)
        cls.run_result = captured['result']
        cls.labels_to_variables: Dict[str, List[str]] = captured['labels_to_variables']
        cls.exploded_labels: List[str] = captured['exploded_labels']
        cls.entity_count_per_label = captured['entity_count_per_label']
        cls.column_prefixes: Dict[str, str] = captured['column_prefixes']
        cls.live_ue_kpi = captured['live_ue_kpi']

    @staticmethod
    def _entity_count(variables: List[str]) -> int:
        """
        The first-axis length every variable routed to one exploded label agrees on.

        The exporter takes the majority length; a real preset should have no disagreement at
        all, so this asserts unanimity and returns it.
        """
        lengths = {len(np.asarray(Saveable.get(var), dtype=object)) for var in variables}
        if len(lengths) != 1:
            raise AssertionError(
                f'variables {variables} disagree on their entity count: {sorted(lengths)}'
            )
        return lengths.pop()

    @classmethod
    def tearDownClass(cls):
        tmp = getattr(cls, 'tmp_dir', None)
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)


@requires_giulia
class SingleRunExportTests(_MinimalRunFixture):
    """
    The single most important test in this file: a real run's Saveables are readable at
    export time and land in real files.

    If the registry were empty when ``_save_episodes_performance_parquet`` reads it — the
    Stage-1 failure described in ``saveable-concurrency.md`` §11 — every one of these
    assertions fails, whereas the run itself would still "succeed" and write nothing.
    """

    NPZ = True

    def test_a_dataset_exists_for_every_label_in_use(self):
        """Every label a registered Saveable routes to produced its own parquet."""
        self.assertTrue(self.exploded_labels, 'the run registered no exploded Saveable at all')
        produced = written_labels(self.output_dir, self.run_result.project_name)
        for label in self.exploded_labels:
            with self.subTest(label=label):
                self.assertIn(label, produced)
                self.assertTrue(os.path.isfile(self.run_result.dataset(label)))

    def test_saveable_columns_are_named_class_dot_variable(self):
        """Each routed ``<Class>.<variable>`` is present, as itself or as its ``.N`` split."""
        for label in self.exploded_labels:
            df = self.run_result.read(label)
            for variable in self.labels_to_variables[label]:
                with self.subTest(label=label, variable=variable):
                    self.assertTrue(
                        column_family(df, variable),
                        f"'{variable}' produced no column in '{label}'; columns: {list(df.columns)}",
                    )

    def test_a_schema_sidecar_is_written_per_dataset(self):
        """Every dataset gets a ``<dataset>.schema.md`` listing its columns."""
        for label in self.exploded_labels:
            with self.subTest(label=label):
                sidecar = self.run_result.schema_sidecar(label)
                self.assertTrue(os.path.isfile(sidecar), f'missing sidecar for {label}')
                content = open(sidecar, encoding='utf-8').read()
                self.assertIn('Saveables schema file generated on', content)
                df = self.run_result.read(label)
                for column in df.columns:
                    self.assertIn(f'|`{column}`|', content,
                                  f"column '{column}' missing from {os.path.basename(sidecar)}")

    def test_exported_values_are_populated_and_finite(self):
        """No dataset is a shell of index columns, and no numeric column is all-null or non-finite."""
        for label in self.exploded_labels:
            with self.subTest(label=label):
                df = self.run_result.read(label)
                data_columns = [c for c in df.columns if c not in index_columns(label)]
                self.assertTrue(data_columns, f"'{label}' has only index columns")
                self.assertGreater(len(df), 0, f"'{label}' has no rows")

                for column in data_columns:
                    self.assertFalse(df[column].isna().all(),
                                     f"column '{column}' of '{label}' is entirely null")

                numeric = df[data_columns].select_dtypes(include=[np.number])
                for column in numeric.columns:
                    values = numeric[column].to_numpy()
                    self.assertTrue(np.isfinite(values).all(),
                                    f"column '{column}' of '{label}' holds non-finite values")

    def test_a_known_kpi_has_a_physically_plausible_range(self):
        """Sanity, not just non-nullness: RSRP in dBm and SINR in dB must be in range."""
        ue_kpi = self.run_result.read('ue_kpi')
        rsrp = ue_kpi['Cell_Selection_SSB.best_server_rsrp_per_ue_dBm'].to_numpy()
        self.assertTrue(((rsrp > -200) & (rsrp < 50)).all(), f'RSRP out of range: {rsrp.min()}..{rsrp.max()}')
        sinr = ue_kpi['SINR_SSB.sinr_ue_to_cell_dB'].to_numpy()
        self.assertTrue(((sinr > -100) & (sinr < 100)).all(), f'SINR out of range: {sinr.min()}..{sinr.max()}')

    def test_prb_usage_per_cell_matches_the_hardcoded_core_kpi(self):
        """
        ``BaseStations.prb_usage_per_cell`` is the Saveable that carries the per-cell PRB
        usage into Parquet.

        The same quantity is also collected on the hardcoded core-KPI path as
        ``r_PRB_usage_per_cell`` (which only ever reaches the NPZ, because it has no
        ``pa.field`` in the generic schema — see
        ``GenericLabelExportTests.test_every_collected_core_kpi_reaches_the_generic_file``).
        Both read the same scheduler, so they must agree exactly, step by step; if they ever
        diverge, one of the two is reading a stale or differently-ordered array.
        """
        df = self.run_result.read('cell_kpi')
        column = 'BaseStations.prb_usage_per_cell'
        self.assertIn(column, df.columns)

        for episode in range(self.EPISODES):
            for step in range(self.STEPS):
                with self.subTest(episode=episode, step=step):
                    core = np.asarray(
                        self.run_result.outputs_obj.r_PRB_usage_per_cell[episode][step],
                        dtype=float,
                    ).ravel()
                    exported = df[(df['episode_idx'] == episode) &
                                  (df['step_idx'] == step)].sort_values('cell_id')[column]
                    self.assertEqual(len(core), self.run_result.cell_count)
                    np.testing.assert_allclose(exported.to_numpy(dtype=float), core, rtol=1e-12)
                    self.assertTrue(((core >= 0.0) & (core <= 1.0)).all(),
                                    f'PRB usage outside [0, 1]: {core.min()}..{core.max()}')

    def test_npz_is_written_with_the_core_kpis(self):
        """The NPZ path runs too — core KPIs only, stacked along the episode axis."""
        npz_file = os.path.join(self.tmp_dir, 'results', self.run_result.project_name, 'results-raw.npz')
        self.assertTrue(os.path.isfile(npz_file), f'no NPZ at {npz_file}')
        with np.load(npz_file, allow_pickle=True) as data:
            self.assertIn('cell_throughput_Mbps', data.files)
            self.assertIn('best_serving_cell_ID_per_ue_based_on_SSB', data.files)
            stacked = data['best_serving_cell_ID_per_ue_based_on_SSB']
            self.assertEqual(len(stacked), self.EPISODES)
            # Saveable variables are deliberately NOT in the NPZ (output-module.md §3).
            self.assertNotIn('Cell_Selection_SSB.best_serving_cell_ID_per_ue', data.files)

    def test_no_generic_dataset_when_every_variable_is_explicitly_routed(self):
        """
        ``generic`` is written **iff** at least one Saveable variable resolves to it.

        It is not written unconditionally: ``_save_episodes_performance_parquet`` builds the
        hardcoded ``generic`` schema, but only writes the labels that appear in
        ``saveable_fields`` — which is keyed off what the registered Saveables route to. In
        both shipped presets every exposed variable is explicitly routed to an exploded label,
        so nothing falls back and no ``<project>_generic.parquet`` appears.

        :class:`GenericLabelExportTests` pins the other direction: leave one variable
        unrouted and the file (core KPIs and all) does appear.
        """
        produced = written_labels(self.output_dir, self.run_result.project_name)
        self.assertEqual(GENERIC_LABEL in produced, GENERIC_LABEL in self.labels_to_variables)
        self.assertNotIn(GENERIC_LABEL, self.labels_to_variables,
                         'a shipped Saveable now falls back to generic; update this expectation')


@requires_giulia
class ExplodedFileTests(_MinimalRunFixture):
    """Per-entity ("exploded") files: one row per entity, per step, per episode."""

    def test_row_count_is_entity_count_times_steps_times_episodes(self):
        for label in self.exploded_labels:
            with self.subTest(label=label):
                df = self.run_result.read(label)
                expected = self.entity_count_per_label[label] * self.STEPS * self.EPISODES
                self.assertEqual(len(df), expected)

    def test_entity_counts_match_the_simulation_objects(self):
        """
        The row count is not merely self-consistent: for the labels whose entity has a
        first-class accessor, it equals what the deployment actually contains.
        """
        for label, expected in self.run_result.entity_counts.items():
            if label not in self.exploded_labels:
                continue
            with self.subTest(label=label):
                self.assertEqual(self.entity_count_per_label[label], expected)
                df = self.run_result.read(label)
                self.assertEqual(len(df), expected * self.STEPS * self.EPISODES)

    def test_entity_id_column_is_present_and_dense_per_step(self):
        for label in self.exploded_labels:
            with self.subTest(label=label):
                df = self.run_result.read(label)
                id_column = _entity_id_column(label)
                self.assertIn(id_column, df.columns)
                expected_ids = list(range(self.entity_count_per_label[label]))
                for (episode, step), group in df.groupby(['episode_idx', 'step_idx']):
                    self.assertEqual(sorted(group[id_column].tolist()), expected_ids,
                                     f'{label} episode={episode} step={step}')

    def test_two_dimensional_variable_splits_into_one_column_per_element(self):
        """
        A UE×cell matrix routed to ``ue_kpi`` becomes one row per UE and one column per cell.
        ``RSRP_CRS_no_fast_fading.rsrp_ue_to_cell_dBm`` declares the prefix ``cell``.
        """
        df = self.run_result.read('ue_kpi')
        base = 'RSRP_CRS_no_fast_fading.rsrp_ue_to_cell_dBm'
        columns = column_family(df, base)
        self.assertNotIn(base, columns, 'a 2-D variable must not keep an unsplit column')
        self.assertEqual(len(columns), self.run_result.cell_count)
        expected = [f'{base}.cell{i}' for i in range(self.run_result.cell_count)]
        self.assertEqual(sorted(columns), sorted(expected))

    def test_column_prefixes_come_from_expose_variable_column_prefixes(self):
        """Each 2-D variable's suffix is the prefix its class declares, not a shared default."""
        df = self.run_result.read('ue_kpi')
        self.assertTrue(self.column_prefixes, 'no class declared a column prefix')
        for base, prefix in self.column_prefixes.items():
                columns = column_family(df, base)
                if not columns:
                    continue  # variable is routed elsewhere or absent from this preset
                with self.subTest(variable=base, prefix=prefix):
                    self.assertTrue(
                        all(c.startswith(f'{base}.{prefix}') for c in columns),
                        f'{base} columns do not use the declared prefix {prefix!r}: {columns[:3]}',
                    )
                    self.assertIn(f'{base}.{prefix}0', columns)

    def test_one_dimensional_variable_stays_a_single_scalar_column(self):
        df = self.run_result.read('ue_kpi')
        base = 'Cell_Selection_SSB.best_serving_cell_ID_per_ue'
        self.assertEqual(column_family(df, base), [base])
        self.assertEqual(len(df[base]), self.run_result.ue_count * self.STEPS * self.EPISODES)

    def test_exploded_values_match_the_live_saveable_for_the_last_step(self):
        """
        Provenance: row *i* of the last step really holds entity *i*'s value, read from the
        very object still registered at the end of the run.
        """
        df = self.run_result.read('ue_kpi')
        base = self.LIVE_KPI
        live = self.live_ue_kpi[base]
        last_step = df[(df['episode_idx'] == self.EPISODES - 1) &
                       (df['step_idx'] == self.STEPS - 1)].sort_values('ue_id')
        np.testing.assert_allclose(last_step[base].to_numpy(dtype=float), live, rtol=1e-9)


@requires_giulia
class FieldsFilterTests(unittest.TestCase):
    """
    ``output.fields``: one global filter over every parquet
    (``docs/output-saveable/saveable-output.md`` §6).

    One filtered run backs all four claims: index columns always survive, a base name keeps
    every expanded column, a single ``.<n>`` keeps only that one, and a label left with no
    data columns produces no file at all.
    """

    #: Everything selected here routes to ue_kpi, so every other label must end up empty.
    KEPT_SCALAR = 'Cell_Selection_SSB.best_serving_cell_ID_per_ue'
    KEPT_MATRIX_BASE = 'RSRP_SSB_no_fast_fading.rsrp_ue_to_cell_dBm'
    KEPT_SINGLE_COLUMN = 'Cell_Selection_CRS.beam_activity_per_ue.cell0'
    DROPPED_SCALAR = 'SINR_SSB.sinr_ue_to_cell_dB'

    @classmethod
    def setUpClass(cls):
        if GIULIA_IMPORT_ERROR is not None:
            raise unittest.SkipTest(f'Giulia unavailable: {GIULIA_IMPORT_ERROR}')
        quiet_logs()
        cls.tmp_dir = tempfile.mkdtemp(prefix='giulia_saveable_fields_')
        cls.unfiltered_dir = os.path.join(cls.tmp_dir, 'all')
        cls.filtered_dir = os.path.join(cls.tmp_dir, 'subset')
        os.makedirs(cls.unfiltered_dir)
        os.makedirs(cls.filtered_dir)

        # Same scenario twice: the only difference is the filter, so any difference in the
        # output is attributable to it. SMALL_SCENARIO keeps this pair cheap (10 UEs, 3 cells).
        # Both runs share one isolated context, so the filtered run sees exactly the Saveables
        # the unfiltered one did.
        def simulate():
            unfiltered = run_simulation(scenario=SMALL_SCENARIO, output_dir=cls.unfiltered_dir,
                                        project_name='unfiltered')
            all_labels = sorted(l for l in registered_labels() if l != GENERIC_LABEL)
            filtered = run_simulation(
                scenario=SMALL_SCENARIO,
                output_dir=cls.filtered_dir,
                project_name='filtered',
                fields=[cls.KEPT_SCALAR, cls.KEPT_MATRIX_BASE, cls.KEPT_SINGLE_COLUMN],
            )
            return unfiltered, all_labels, filtered

        cls.unfiltered, cls.all_labels, cls.filtered = run_in_isolated_context(simulate)

    @classmethod
    def tearDownClass(cls):
        tmp = getattr(cls, 'tmp_dir', None)
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)

    def test_unfiltered_run_exports_every_label(self):
        """The baseline: without a filter, every label in use has a file."""
        produced = written_labels(self.unfiltered_dir, 'unfiltered')
        self.assertEqual(produced, set(self.all_labels))

    def test_index_columns_always_survive_the_filter(self):
        df = self.filtered.read('ue_kpi')
        for column in index_columns('ue_kpi'):
            self.assertIn(column, df.columns)

    def test_listing_a_base_name_keeps_all_of_its_expanded_columns(self):
        df = self.filtered.read('ue_kpi')
        kept = column_family(df, self.KEPT_MATRIX_BASE)
        reference = column_family(self.unfiltered.read('ue_kpi'), self.KEPT_MATRIX_BASE)
        self.assertTrue(reference, 'the reference run produced no columns for the matrix variable')
        self.assertEqual(sorted(kept), sorted(reference))

    def test_listing_one_expanded_column_keeps_only_that_column(self):
        df = self.filtered.read('ue_kpi')
        base = self.KEPT_SINGLE_COLUMN.rsplit('.', 1)[0]
        self.assertEqual(column_family(df, base), [self.KEPT_SINGLE_COLUMN])
        self.assertGreater(len(column_family(self.unfiltered.read('ue_kpi'), base)), 1,
                           'the unfiltered run should have several columns for this variable')

    def test_unselected_columns_are_dropped(self):
        df = self.filtered.read('ue_kpi')
        self.assertNotIn(self.DROPPED_SCALAR, df.columns)
        self.assertIn(self.KEPT_SCALAR, df.columns)
        expected = set(index_columns('ue_kpi')) | {self.KEPT_SCALAR, self.KEPT_SINGLE_COLUMN} \
            | set(column_family(df, self.KEPT_MATRIX_BASE))
        self.assertEqual(set(df.columns), expected)

    def test_a_label_left_with_no_data_columns_produces_no_file(self):
        """No parquet, no schema.md, not even an empty folder (saveable-output.md §2)."""
        produced = written_labels(self.filtered_dir, 'filtered')
        self.assertEqual(produced, {'ue_kpi'})
        for label in self.all_labels:
            if label == 'ue_kpi':
                continue
            with self.subTest(label=label):
                base = os.path.join(self.filtered_dir, f'filtered_{label}')
                self.assertFalse(os.path.exists(base + '.parquet'))
                self.assertFalse(os.path.exists(base + '.schema.md'))
                self.assertFalse(os.path.exists(base))

    def test_filtering_does_not_change_the_rows_that_remain(self):
        """The filter trims columns; it must not alter values or row counts."""
        filtered = self.filtered.read('ue_kpi').sort_values(['episode_idx', 'step_idx', 'ue_id'])
        unfiltered = self.unfiltered.read('ue_kpi').sort_values(['episode_idx', 'step_idx', 'ue_id'])
        self.assertEqual(len(filtered), len(unfiltered))
        # Two runs of the same scenario use the same seed per episode, so the KPI itself matches.
        np.testing.assert_array_equal(filtered[self.KEPT_SCALAR].to_numpy(),
                                      unfiltered[self.KEPT_SCALAR].to_numpy())


@requires_giulia
class LatestWinsAcrossEpisodesTests(unittest.TestCase):
    """
    "Latest wins" across **episodes**: the simulation re-creates its Saveables in every
    ``configure()``, and the export must carry each episode's own objects — not episode 0's,
    frozen (``saveable.py::Saveable.__init__``, ``saveable-output.md`` §11).
    """

    KPI = 'Cell_Selection_SSB.best_server_rsrp_per_ue_dBm'
    EPISODES = 2
    STEPS = 1

    @classmethod
    def setUpClass(cls):
        if GIULIA_IMPORT_ERROR is not None:
            raise unittest.SkipTest(f'Giulia unavailable: {GIULIA_IMPORT_ERROR}')
        quiet_logs()
        cls.tmp_dir = tempfile.mkdtemp(prefix='giulia_saveable_episodes_')
        cls.instance_ids: Dict[int, int] = {}
        cls.snapshots: Dict[int, 'np.ndarray'] = {}

        def record_instance(episode_index, _g):
            registered = [o for o in Saveable.registry() if o.name == 'Cell_Selection_SSB']
            assert len(registered) == 1, f'expected exactly one Cell_Selection_SSB, got {len(registered)}'
            cls.instance_ids[episode_index] = id(registered[0])

        def record_values(episode_index, _g):
            cls.snapshots[episode_index] = np.asarray(Saveable.get(cls.KPI), dtype=float).ravel().copy()

        cls.run_result = run_in_isolated_context(lambda: run_simulation(
            scenario=SMALL_SCENARIO,
            episodes=cls.EPISODES,
            steps=cls.STEPS,
            output_dir=cls.tmp_dir,
            project_name='episodes',
            after_configure=record_instance,
            after_episode=record_values,
        ))

    @classmethod
    def tearDownClass(cls):
        tmp = getattr(cls, 'tmp_dir', None)
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)

    def test_each_episode_registers_a_fresh_instance(self):
        """A new object per episode — and only one entry for the class, never two."""
        self.assertEqual(len(self.instance_ids), self.EPISODES)
        self.assertNotEqual(self.instance_ids[0], self.instance_ids[1],
                            'configure() must re-create the Saveable each episode')

    def test_episodes_do_not_export_identical_data(self):
        """The symptom of a frozen registry: every episode exporting episode 0's numbers."""
        df = self.run_result.read('ue_kpi')
        per_episode = [
            df[df['episode_idx'] == e].sort_values('ue_id')[self.KPI].to_numpy(dtype=float)
            for e in range(self.EPISODES)
        ]
        self.assertFalse(np.array_equal(per_episode[0], per_episode[1]),
                         'both episodes exported the same values — the registry looks frozen')

    def test_each_episodes_rows_come_from_that_episodes_instance(self):
        """
        The positive form: episode *e*'s rows equal what *e*'s own Saveable held at the end
        of the episode.
        """
        df = self.run_result.read('ue_kpi')
        for episode in range(self.EPISODES):
            with self.subTest(episode=episode):
                rows = df[(df['episode_idx'] == episode) &
                          (df['step_idx'] == self.STEPS - 1)].sort_values('ue_id')
                np.testing.assert_allclose(rows[self.KPI].to_numpy(dtype=float),
                                           self.snapshots[episode], rtol=1e-9)


@requires_giulia
class GenericLabelExportTests(unittest.TestCase):
    """
    The ``generic`` parquet and the two fallbacks that fill it.

    ``generic`` is the label a variable lands in when nothing routes it anywhere else, and it
    is written as soon as *any* variable resolves to it. There are two ways that happens, and
    they behave differently:

    * **Class-level fallback (silent).** A class that overrides neither ``label`` nor
      ``expose_variable_labels()`` inherits the base ``label`` of ``'generic'``; the base
      ``expose_variable_labels()`` then maps every variable to it, so the export sees an
      explicit entry and says nothing.
    * **Per-variable fallback (warned).** A class that *does* override
      ``expose_variable_labels()`` but leaves a variable out of the mapping falls back to the
      class ``label``. This is the easy mistake, so the export logs a warning naming the
      variable and the file it fell into.

    Either way, once ``generic`` is in play the file carries the whole hardcoded core-KPI
    schema (``cell_throughput_Mbps``, ``PRB_usage_per_cell``,
    ``total_network_power_consumption_kW``, …) — one row per ``(episode, step)`` — which is
    the only path those columns have into Parquet.

    No shipped Saveable falls back today, so the classes below are defined by the test; they
    are registered by, collected from and exported through the real pipeline inside a real run.
    """

    EPISODES = 1
    STEPS = 2
    VECTOR_LENGTH = 5
    MATRIX_SHAPE = (2, 3)
    FORGOTTEN_LENGTH = 4

    @classmethod
    def setUpClass(cls):
        if GIULIA_IMPORT_ERROR is not None:
            raise unittest.SkipTest(f'Giulia unavailable: {GIULIA_IMPORT_ERROR}')
        quiet_logs()
        cls.tmp_dir = tempfile.mkdtemp(prefix='giulia_saveable_generic_')

        cls.expected_vector = np.arange(cls.VECTOR_LENGTH, dtype=float)
        cls.expected_matrix = np.arange(int(np.prod(cls.MATRIX_SHAPE)),
                                        dtype=float).reshape(cls.MATRIX_SHAPE)
        cls.expected_forgotten = np.arange(cls.FORGOTTEN_LENGTH, dtype=float)

        class _UnroutedKpi(Saveable):
            """Overrides nothing: every variable resolves to the base label, silently."""

            def __init__(self):
                super().__init__()
                self.vector = cls.expected_vector
                self.matrix = cls.expected_matrix

            def expose_variables(self) -> List[str]:
                return ['vector', 'matrix']

        class _PartiallyRoutedKpi(Saveable):
            """Routes one variable and forgets the other — the warned fallback."""

            def __init__(self, ue_count: int):
                super().__init__()
                self.routed = np.full(ue_count, 7.0)
                self.forgotten = cls.expected_forgotten

            def expose_variables(self) -> List[str]:
                return ['routed', 'forgotten']

            def expose_variable_labels(self) -> Dict[str, str]:
                return {'routed': 'ue_kpi'}          # 'forgotten' deliberately omitted

        cls.unrouted_name = _UnroutedKpi.__name__
        cls.partial_name = _PartiallyRoutedKpi.__name__

        def register_probes(_episode_index, g):
            _UnroutedKpi()
            _PartiallyRoutedKpi(int(g.ue_deployment_obj.number_of_ues))

        def simulate():
            # The probes below are synthetic; an isolated context keeps them out of every
            # other test class's registry.
            with captured_warnings(save_episode_performance) as messages:
                result = run_simulation(
                    scenario=SMALL_SCENARIO,
                    episodes=cls.EPISODES,
                    steps=cls.STEPS,
                    output_dir=cls.tmp_dir,
                    project_name='generic',
                    after_configure=register_probes,
                )
            return result, list(messages), registered_labels()

        cls.run_result, cls.warnings, cls.labels_to_variables = run_in_isolated_context(simulate)

    @classmethod
    def tearDownClass(cls):
        tmp = getattr(cls, 'tmp_dir', None)
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)

    def test_a_variable_falling_back_to_generic_creates_the_generic_dataset(self):
        self.assertIn(GENERIC_LABEL, self.labels_to_variables)
        self.assertIn(GENERIC_LABEL, written_labels(self.tmp_dir, 'generic'))
        self.assertTrue(os.path.isfile(self.run_result.schema_sidecar(GENERIC_LABEL)))

    def test_generic_holds_one_row_per_episode_and_step(self):
        """Unlike an exploded file, ``generic`` keeps each array whole on a single row."""
        df = self.run_result.read(GENERIC_LABEL)
        self.assertEqual(len(df), self.EPISODES * self.STEPS)
        self.assertEqual(sorted(df.columns[:2]), ['episode_idx', 'step_idx'])
        self.assertEqual(df.groupby(['episode_idx', 'step_idx']).size().unique().tolist(), [1])

    def test_generic_carries_the_hardcoded_core_kpis(self):
        """
        The core KPIs written by ``save_performance.py``'s separate code path reach Parquet
        only through this file — this is what makes its absence matter.
        """
        df = self.run_result.read(GENERIC_LABEL)
        for column in ('cell_throughput_Mbps', 'carrier_throughput_Mbps',
                       'total_network_power_consumption_kW',
                       'best_serving_cell_ID_per_ue_based_on_SSB'):
            with self.subTest(column=column):
                self.assertIn(column, df.columns)
                values = np.asarray(df[column].iloc[0], dtype=float)
                self.assertGreater(len(values), 0, f"'{column}' is an empty list")
                self.assertTrue(np.isfinite(values).all())

        throughput = np.asarray(df['cell_throughput_Mbps'].iloc[0], dtype=float)
        self.assertEqual(len(throughput), self.run_result.cell_count)

    @unittest.expectedFailure
    def test_every_collected_core_kpi_reaches_the_generic_file(self):
        """
        KNOWN BUG, pinned deliberately — pre-dates the per-context registry.

        Two core KPIs are collected every step and put into the generic record, but are
        missing from the hardcoded ``parquet_schema['generic']`` field list:
        ``PRB_usage_per_cell`` and ``power_consumption_per_radio_unit_kW``.
        ``pa.Table.from_pylist(records, schema=...)`` keeps only what the schema names, so both
        are dropped from the generic file without a word.

        **The data is not lost**: both quantities reach Parquet through the Saveable path
        instead — ``Power_Consumption.power_consumption_per_radio_unit_kW`` in ``ru_kpi`` and
        ``BaseStations.prb_usage_per_cell`` in ``cell_kpi`` (pinned by
        ``SingleRunExportTests.test_prb_usage_per_cell_matches_the_hardcoded_core_kpi``). What
        remains is an inconsistency between the generic record and the generic schema, whose
        only effect is that the wide ``generic`` table is missing two columns it builds.

        Marked ``expectedFailure`` so the suite stays green today and reports an *unexpected
        success* once the two ``pa.field(...)`` entries are added.
        """
        df = self.run_result.read(GENERIC_LABEL)
        for column in ('PRB_usage_per_cell', 'power_consumption_per_radio_unit_kW'):
            with self.subTest(column=column):
                self.assertIn(column, df.columns)

    def test_the_class_level_fallback_is_silent_and_keeps_the_array_whole(self):
        df = self.run_result.read(GENERIC_LABEL)
        column = f'{self.unrouted_name}.vector'
        self.assertEqual(column_family(df, column), [column])
        for row in range(len(df)):
            np.testing.assert_allclose(np.asarray(df[column].iloc[row], dtype=float),
                                       self.expected_vector)
        self.assertFalse([m for m in self.warnings if self.unrouted_name in m],
                         'the class-level fallback should not warn')

    def test_the_per_variable_fallback_warns_and_names_the_variable(self):
        forgotten = f'{self.partial_name}.forgotten'
        matching = [m for m in self.warnings
                    if forgotten in m and 'expose_variable_labels' in m]
        self.assertEqual(len(matching), 1, f'expected one warning for {forgotten}: {self.warnings}')
        self.assertIn(f"falling back to label '{GENERIC_LABEL}'", matching[0])

        df = self.run_result.read(GENERIC_LABEL)
        self.assertIn(forgotten, df.columns)
        np.testing.assert_allclose(np.asarray(df[forgotten].iloc[0], dtype=float),
                                   self.expected_forgotten)

    def test_the_explicitly_routed_variable_does_not_land_in_generic(self):
        """One class, two files: the mapped variable goes where it was told."""
        routed = f'{self.partial_name}.routed'
        self.assertIn(routed, self.run_result.read('ue_kpi').columns)
        self.assertNotIn(routed, self.run_result.read(GENERIC_LABEL).columns)

    @unittest.expectedFailure
    def test_a_two_d_variable_in_generic_is_populated(self):
        """
        KNOWN BUG, pinned deliberately — pre-dates the per-context registry.

        A 2-D variable in ``generic`` produces columns that are **entirely null**. The schema
        builder allocates one column per element of the *second* axis and appends the indexed
        names (``<Class>.matrix.0``, ``.1``, …) to ``saveable_fields``, but the per-step buffer
        from ``save_performance.py`` is keyed by the **base** name (``<Class>.matrix``). The
        writer looks each indexed name up in that buffer, never finds it, and writes ``None``
        — so its own 2-D branch (which would split along the *first* axis, disagreeing with the
        schema anyway) is unreachable for ``generic``.

        Only reachable when a 2-D variable falls back to ``generic``; no shipped Saveable does,
        which is why it has gone unnoticed. Marked ``expectedFailure`` so the suite stays green
        today and reports an *unexpected success* once it is fixed.
        """
        df = self.run_result.read(GENERIC_LABEL)
        base = f'{self.unrouted_name}.matrix'
        columns = column_family(df, base)
        self.assertTrue(columns)
        for column in columns:
            self.assertFalse(df[column].isna().all(), f"'{column}' is entirely null")


@requires_giulia
class NewLabelExportTests(unittest.TestCase):
    """
    A label nothing in ``giulia/`` uses auto-creates its own parquet, with an entity-id column
    derived from the label name — and a 2-D variable whose class declares **no**
    ``expose_variable_column_prefixes()`` entry falls back to the numeric ``.0/.1/…`` suffix.

    No shipped Saveable exercises that fallback (every 2-D exploded variable in both presets
    declares a prefix), so the Saveable below is defined by the test — but it is registered by,
    collected from and exported through the *real* pipeline, inside a real run.
    """

    ENTITIES = 4
    COLUMNS = 3
    LABEL = 'probe_kpi'

    @classmethod
    def setUpClass(cls):
        if GIULIA_IMPORT_ERROR is not None:
            raise unittest.SkipTest(f'Giulia unavailable: {GIULIA_IMPORT_ERROR}')
        quiet_logs()
        cls.tmp_dir = tempfile.mkdtemp(prefix='giulia_saveable_newlabel_')

        matrix = np.arange(cls.ENTITIES * cls.COLUMNS, dtype=float).reshape(cls.ENTITIES, cls.COLUMNS)
        vector = np.arange(cls.ENTITIES, dtype=float) * 10.0
        cls.expected_matrix = matrix
        cls.expected_vector = vector

        class _Probe(Saveable):
            """A per-entity Saveable on a brand-new label, with a 2-D variable and no prefix."""

            def __init__(self):
                super().__init__()
                self.matrix = matrix
                self.vector = vector

            @property
            def label(self) -> str:
                return NewLabelExportTests.LABEL

            def expose_variables(self) -> List[str]:
                return ['matrix', 'vector']

            def expose_variable_labels(self) -> Dict[str, str]:
                return {'matrix': NewLabelExportTests.LABEL, 'vector': NewLabelExportTests.LABEL}

        cls.probe_class_name = _Probe.__name__

        # Isolated context: _Probe is synthetic and must not leak into other test classes.
        cls.run_result = run_in_isolated_context(lambda: run_simulation(
            scenario=SMALL_SCENARIO,
            output_dir=cls.tmp_dir,
            project_name='newlabel',
            # Registered inside the run, after configure(), so it is collected every step.
            after_configure=lambda _e, _g: _Probe(),
        ))

    @classmethod
    def tearDownClass(cls):
        tmp = getattr(cls, 'tmp_dir', None)
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)

    def test_a_new_label_creates_its_own_parquet(self):
        self.assertIn(self.LABEL, written_labels(self.tmp_dir, 'newlabel'))
        self.assertTrue(os.path.isfile(self.run_result.schema_sidecar(self.LABEL)))

    def test_the_entity_id_column_is_derived_from_the_label_name(self):
        df = self.run_result.read(self.LABEL)
        self.assertIn('probe_id', df.columns)
        self.assertEqual(sorted(df['probe_id'].tolist()), list(range(self.ENTITIES)))
        self.assertEqual(len(df), self.ENTITIES)

    def test_a_two_d_variable_without_a_prefix_uses_numeric_suffixes(self):
        df = self.run_result.read(self.LABEL)
        base = f'{self.probe_class_name}.matrix'
        self.assertEqual(sorted(column_family(df, base)),
                         sorted(f'{base}.{i}' for i in range(self.COLUMNS)))
        rows = df.sort_values('probe_id')
        for index in range(self.COLUMNS):
            np.testing.assert_allclose(rows[f'{base}.{index}'].to_numpy(dtype=float),
                                       self.expected_matrix[:, index])

    def test_a_one_d_variable_stays_scalar(self):
        df = self.run_result.read(self.LABEL).sort_values('probe_id')
        base = f'{self.probe_class_name}.vector'
        self.assertEqual(column_family(df, base), [base])
        np.testing.assert_allclose(df[base].to_numpy(dtype=float), self.expected_vector)


if __name__ == '__main__':
    unittest.main()
