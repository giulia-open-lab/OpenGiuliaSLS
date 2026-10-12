import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Configure Giulia into path
this_dir = os.path.dirname(__file__)
root_dir = os.path.join(this_dir, '..')
sys.path.insert(1, root_dir)

from giulia.logger import LogLevel, set_log_level, get_log_level, __should_print as should_print
from tests.compare import compare_scenario


class UnitTests(unittest.TestCase):
    def test_logger_set_log_level(self):
        set_log_level(LogLevel.ERROR)
        assert get_log_level() == LogLevel.ERROR

    def test_logger___should_print(self):
        set_log_level(LogLevel.WARNING)

        assert should_print(LogLevel.ERROR, False)
        assert should_print(LogLevel.WARNING, False)
        assert not should_print(LogLevel.INFO, False)

        assert not should_print(LogLevel.ERROR, True)
        assert should_print(LogLevel.WARNING, True)
        assert not should_print(LogLevel.INFO, True)


class CompareNonFiniteTests(unittest.TestCase):
    """KPI columns legitimately hold NaN (a cell the simulation never fills, e.g. the
    beams of a frequency layer a UE is not on) and -inf (a power of exactly 0 mW in dB).
    A pair that is identical on both sides must compare equal even when non-finite,
    while a non-finite on only one side must still fail.
    """

    def _compare_single_column(self, actual, expected, rtol=0.1):
        tmp = Path(tempfile.mkdtemp())
        actual_dir, ref_dir = tmp / 'actual', tmp / 'reference'
        actual_dir.mkdir()
        ref_dir.mkdir()
        for directory, values in ((actual_dir, actual), (ref_dir, expected)):
            rows = len(values)
            pq.write_table(
                pa.table({
                    'episode_idx': pa.array([0] * rows, pa.uint64()),
                    'step_idx': pa.array(range(rows), pa.uint64()),
                    'ue_id': pa.array(range(rows), pa.uint64()),
                    'Kpi.cell_throughput_Mbps': values,
                }),
                directory / 'scenario_ue_kpi.parquet',
            )
        return compare_scenario(actual_dir, ref_dir, rtol=rtol)

    def test_matched_non_finite_values_pass(self):
        for actual, expected, label in (
            ([np.nan, 1.0], [np.nan, 1.0], 'both NaN'),
            ([-np.inf, 1.0], [-np.inf, 1.0], 'both -inf'),
            ([np.inf, 1.0], [np.inf, 1.0], 'both +inf'),
            ([1.0, 1.05], [1.0, 1.0], 'drift within rtol'),
        ):
            with self.subTest(label):
                assert self._compare_single_column(actual, expected).passed

    def test_unmatched_non_finite_values_fail(self):
        for actual, expected, label in (
            ([np.nan, 1.0], [3.0, 1.0], 'NaN on one side only'),
            ([-np.inf, 1.0], [3.0, 1.0], '-inf on one side only'),
            ([1.0, 5.0], [1.0, 1.0], 'drift beyond rtol'),
        ):
            with self.subTest(label):
                assert not self._compare_single_column(actual, expected).passed

    def test_opposite_infinities_pass_when_means_of_finite_rows_match(self):
        """Pass/fail is decided on the two means, not per-row: with the infinities on
        differing rows excluded from both means as non-finite, the remaining finite
        rows are what's compared, and here they agree.
        """
        assert self._compare_single_column([-np.inf, 1.0], [np.inf, 1.0]).passed

    def test_mean_ignores_non_finite_values(self):
        """A single NaN must not collapse the reported mean of the whole column."""
        result = self._compare_single_column([np.nan, 2.0, 4.0], [np.nan, 2.0, 4.0])
        kpi = result.kpi_results[0]
        assert kpi.passed
        assert kpi.actual_mean == 3.0
        assert kpi.expected_mean == 3.0


class CompareMissingOutputTests(unittest.TestCase):
    """A run that exported nothing is a run error, not a set of failed KPIs: reporting one
    fabricated failing KPI per expected file hides the real problem and counts the scenario
    as FAIL instead of ERROR.
    """

    def _dirs(self, reference_names, actual_names):
        tmp = Path(tempfile.mkdtemp())
        actual_dir, ref_dir = tmp / 'actual', tmp / 'reference'
        actual_dir.mkdir()
        ref_dir.mkdir()
        table = pa.table({
            'episode_idx': pa.array([0], pa.uint64()),
            'step_idx': pa.array([0], pa.uint64()),
            'ue_id': pa.array([0], pa.uint64()),
            'Kpi.value': [1.0],
        })
        for name in reference_names:
            pq.write_table(table, ref_dir / name)
        for name in actual_names:
            pq.write_table(table, actual_dir / name)
        return actual_dir, ref_dir

    def test_no_output_at_all_is_a_run_error(self):
        actual_dir, ref_dir = self._dirs(['s_ue_kpi.parquet', 's_cell_kpi.parquet'], [])
        result = compare_scenario(actual_dir, ref_dir)
        assert result.run_error is not None
        assert not result.passed
        # No fabricated per-file KPI rows: the error carries the whole story.
        assert result.kpi_results == []

    def test_partial_output_still_reports_the_missing_file_as_a_kpi(self):
        actual_dir, ref_dir = self._dirs(
            ['s_ue_kpi.parquet', 's_cell_kpi.parquet'], ['s_ue_kpi.parquet']
        )
        result = compare_scenario(actual_dir, ref_dir)
        assert result.run_error is None
        assert not result.passed
        assert any('missing output file' in k.column for k in result.kpi_results)

    def test_complete_matching_output_passes(self):
        actual_dir, ref_dir = self._dirs(['s_ue_kpi.parquet'], ['s_ue_kpi.parquet'])
        result = compare_scenario(actual_dir, ref_dir)
        assert result.run_error is None
        assert result.passed
