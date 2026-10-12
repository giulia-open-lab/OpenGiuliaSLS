from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.types as pat

# Whitelist of KPI variable names to compare. A Saveable column is named
# `<ClassName>.<variable_name>` (or `<ClassName>.<variable_name>.<index>` for an
# expanded 2-D variable, see docs/output-saveable/saveable-output.md), so only the
# variable-name component is matched, via `_kpi_base_name`.
_TEST_KEYS = {
    "CRS_RSRP_no_fast_fading_ue_to_cell_dBm",
    "best_serving_cell_ID_per_ue_based_on_CRS",
    "best_serving_CRS_rsrp_per_ue_dBm",
    "best_serving_CRS_re_noise_per_ue_dBm",
    "ues_per_cell",
    "CRS_sinr_ue_to_cell_dB",
    "SSB_RSRP_no_fast_fading_ue_to_cell_dBm",
    # "SSB_RSS_per_PRB_ue_to_cell_dBm",
    "best_serving_SSB_per_ue",
    "best_serving_SSB_rsrp_per_ue_dBm",
    "best_serving_SSB_re_noise_per_ue_dBm",
    "SSB_beam_activity_per_ue",
    "ues_per_SSB_beam",
    "SSB_sinr_ue_to_cell_dB",
    # "CSI_RS_RSRP_no_fast_fading_ue_to_cell_dBm",
    # "CSI_RS_RSS_per_PRB_ue_to_cell_dBm",
    "best_serving_CSI_RS_per_ue",
    "best_serving_CSI_RS_rsrp_per_ue_dBm",
    "CSI_RS_beam_activity_per_ue",
    "ues_per_CSI_RS_beam",
    # "CSI_RS_sinr_ue_to_cell_dB",
    "effective_CSI_RS_sinr_ue_to_cell_dB",
    # "ue_throughput_based_on_avg_CSI_RS_SINR_Mbps",
    "ue_throughput_based_on_ins_CSI_RS_SINR_per_PRB_Mbps",
    "ue_throughput_based_on_eff_CSI_RS_SINR_per_PRB_Mbps",
    "cell_throughput_Mbps",
    "carrier_throughput_Mbps",
    "ue_throughput_per_carrier_Mbps",
    "total_network_power_consumption_kW",
    # "network_power_consumption_kW",
}


def _kpi_base_name(name: str) -> str:
    """The variable name a Saveable column is built from.

    Strips the `<ClassName>.` prefix and, for an expanded 2-D variable, the
    trailing per-index suffix (`.0`, `.cell3`, ...) too, since both
    `<ClassName>.<variable>` and `<ClassName>.<variable>.<index>` carry the
    variable name in the same second component. A column with no dot (a
    hardcoded, un-prefixed name) is its own base name.
    """
    parts = name.split(".")
    return parts[1] if len(parts) > 1 else name


def _is_kpi_column(name: str, arrow_type) -> bool:
    if _kpi_base_name(name) not in _TEST_KEYS:
        return False
    return pat.is_floating(arrow_type) or pat.is_integer(arrow_type) or pat.is_boolean(arrow_type)


def _finite_mean(values: np.ndarray) -> float:
    """Mean over the finite entries only, or NaN when the column has none.

    KPI columns legitimately hold NaN (a cell the simulation never fills) and +/-inf
    (a power of exactly 0 mW in dB); np.mean would collapse the whole column to NaN
    because of them, so the reported mean would say nothing about the real values.
    """
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


# How many of the worst-offending rows to keep per KPI column, so a report can point at
# the specific cells/UEs/etc. behind a "Max error" figure instead of just the aggregate.
_MAX_WORST_ROWS = 10

# Columns that identify *which row* a value belongs to (episode/step counters and the
# entity id every KPI parquet file carries, e.g. `cell_id`, `ue_id`, `bbu_id`), rather
# than being a KPI value themselves. Never matched by `_TEST_KEYS` since they have no
# dot-qualified `<ClassName>.` prefix, so they need their own detection.
def _is_id_column(name: str) -> bool:
    return name in ("episode_idx", "step_idx") or name.endswith("_id")


@dataclass
class WorstRow:
    """One offending row behind a KPI column's max_error, identified by its id columns
    (episode_idx, step_idx, and the entity id such as cell_id/ue_id) so it can be traced
    back to a concrete cell in the parquet output.
    """
    ids: dict[str, object]
    actual: float
    expected: float
    error: float


@dataclass
class KpiResult:
    column: str
    passed: bool
    actual_mean: float
    expected_mean: float
    max_error: float
    error: float
    relative_error: float
    worst_rows: list[WorstRow] = field(default_factory=list)


@dataclass
class ScenarioResult:
    scenario_name: str
    kpi_results: list[KpiResult] = field(default_factory=list)
    run_error: Exception | None = None

    @property
    def passed(self) -> bool:
        return self.run_error is None and all(r.passed for r in self.kpi_results)

    @property
    def failed_kpis(self) -> list[KpiResult]:
        return [r for r in self.kpi_results if not r.passed]


def compare_scenario(
    output_dir: Path,
    reference_dir: Path,
    rtol: float = 0.1,
) -> ScenarioResult:
    """Compare every numeric KPI column common to output_dir and reference_dir.

    Each KPI "label" (bbu_kpi, cell_kpi, ue_kpi, ...) is written to its own parquet
    file, with its own schema, so files are compared one at a time rather than as
    a single unified directory dataset (pyarrow.dataset would otherwise infer the
    schema from just the first fragment and silently ignore the rest).

    KPI columns are discovered from each file's schema rather than hardcoded,
    since the Saveable-based parquet output names them ``<ClassName>.<variable>``
    (and ``<ClassName>.<variable>.<index>`` for per-beam data), and that naming
    shifts whenever KPI classes change.
    """
    result = ScenarioResult(scenario_name=output_dir.name)
    reference_files = sorted(reference_dir.glob("*.parquet"))

    # A run that wrote no parquet at all did not "fail its KPIs": it exported nothing,
    # typically while still exiting 0 (the exporter discovers what to write by walking the
    # Saveable registry, so an empty registry silently produces no file). Reporting that as
    # one failed KPI per expected file buries the real problem under a wall of fabricated
    # KPI rows and counts the scenario as FAIL rather than ERROR. Surface it as a run error.
    # A partially written output still goes through the per-file path below, where a missing
    # file really is one specific KPI group that went absent.
    if reference_files and not any(output_dir.glob("*.parquet")):
        result.run_error = FileNotFoundError(
            f"The run exported no KPI parquet at all into '{output_dir}' "
            f"({len(reference_files)} expected, e.g. '{reference_files[0].name}'). "
            f"The simulation finished without writing any output."
        )
        return result

    for ref_file in reference_files:
        actual_file = output_dir / ref_file.name
        if not actual_file.exists():
            result.kpi_results.append(KpiResult(
                column=f"{ref_file.stem} (missing output file)",
                passed=False,
                actual_mean=float("nan"),
                expected_mean=float("nan"),
                max_error=float("nan"),
                error=float("nan"),
                relative_error=float("nan"),
            ))
            continue
        result.kpi_results.extend(_compare_file(actual_file, ref_file, rtol))
    return result


def _worst_rows(
    err: np.ndarray,
    matched: np.ndarray,
    actual_np: np.ndarray,
    expected_np: np.ndarray,
    actual_table: "pa.Table",
    id_columns: list[str],
) -> list[WorstRow]:
    """The `_MAX_WORST_ROWS` rows with the largest error, so a report can point at the
    specific cell/UE/etc. behind a column's "Max error" rather than just the aggregate.

    A mismatch where only one side is non-finite (e.g. actual is NaN but expected is a
    real number) leaves `err` as NaN too, via NaN/inf propagation through the subtraction
    - sorting would otherwise push these to the bottom even though they are among the
    worst possible mismatches, so they're ranked as if their error were infinite.
    """
    # Excludes exact ties (err == 0) too, not just NaN/inf `matched` pairs - otherwise a
    # column with only a couple of real mismatches would still pad the list out to
    # `_MAX_WORST_ROWS` with rows that are actually identical on both sides.
    candidate_idx = np.flatnonzero(~matched & ((err != 0) | np.isnan(err)))
    if candidate_idx.size == 0:
        return []
    candidate_err = err[candidate_idx]
    sort_key = np.where(np.isfinite(candidate_err), candidate_err, np.inf)
    order = np.argsort(-sort_key, kind="stable")[:_MAX_WORST_ROWS]
    top_idx = candidate_idx[order]

    worst = []
    for i in top_idx:
        row = int(i)
        ids = {col: actual_table.column(col)[row].as_py() for col in id_columns}
        worst.append(WorstRow(
            ids=ids,
            actual=float(actual_np[row]),
            expected=float(expected_np[row]),
            error=float(err[row]),
        ))
    # The rows are picked by error magnitude, but read better ordered by identity
    # (episode_idx, then step_idx, then the entity id) so progression through the
    # run is easy to follow instead of a jumbled worst-first list.
    worst.sort(key=lambda w: tuple(w.ids[col] for col in id_columns))
    return worst


def _compare_file(actual_file: Path, ref_file: Path, rtol: float) -> list[KpiResult]:
    actual_table = pq.read_table(actual_file)
    ref_table = pq.read_table(ref_file)

    actual_types = {f.name: f.type for f in actual_table.schema}
    ref_types = {f.name: f.type for f in ref_table.schema}
    common_columns = sorted(
        name for name in (actual_types.keys() & ref_types.keys())
        if _is_kpi_column(name, ref_types[name])
    )
    # Schema order (episode_idx, step_idx, then the entity id) reads more naturally than
    # the alphabetical order `common_columns` uses for KPI names.
    id_columns = [
        name for name in actual_table.schema.names
        if _is_id_column(name) and name in ref_types
    ]

    results = []
    for column in common_columns:
        actual_np = actual_table.column(column).to_numpy(zero_copy_only=False).astype(float)
        expected_np = ref_table.column(column).to_numpy(zero_copy_only=False).astype(float)

        if actual_np.shape != expected_np.shape:
            results.append(KpiResult(
                column=column,
                passed=False,
                actual_mean=float("nan"),
                expected_mean=float("nan"),
                max_error=float("nan"),
                error=float("nan"),
                relative_error=float("nan"),
            ))
            continue

        with np.errstate(invalid="ignore"):
            err = np.abs(actual_np - expected_np)
        # NaN marks a cell the simulation never fills (e.g. a UE has no RSRP for the
        # beams of a frequency layer it is not on), and -inf is a power of exactly 0 mW
        # in dB - both legitimate values. Treat a matched pair as equal so `_worst_rows`
        # doesn't surface it as an offender; a non-finite on only ONE side still shows up
        # there, via `err`'s own NaN/inf propagation.
        both_nan = np.isnan(actual_np) & np.isnan(expected_np)
        both_inf = (
            np.isinf(actual_np) & np.isinf(expected_np)
            & (np.sign(actual_np) == np.sign(expected_np))
        )
        matched = both_nan | both_inf
        non_matching_err = err[~matched]
        # Means ignore the matched non-finite cells too: a plain np.mean would report
        # `nan` for the whole column as soon as it holds a single NaN, hiding the actual
        # magnitude of every other value in it.
        actual_mean = _finite_mean(actual_np)
        expected_mean = _finite_mean(expected_np)
        # Pass/fail is decided on the two aggregate means, not on any individual row's
        # error: a scenario matches if its overall KPI level matches, even when specific
        # rows drift. `max_error`/`worst_rows` remain purely diagnostic, to help trace a
        # failing mean back to the rows behind it.
        #
        # The tolerance is relative, not absolute: KPIs span wildly different scales (a
        # dB value in the tens vs. a throughput in the thousands), so a single fixed
        # absolute tolerance would be far too loose for small-magnitude KPIs and far too
        # strict for large ones. When the expected mean is exactly 0, relative error is
        # undefined, so a pair only matches there if the actual mean is 0 too.
        means_finite = np.isfinite(actual_mean) and np.isfinite(expected_mean)
        diff = actual_mean - expected_mean
        if not means_finite:
            relative_error = float("nan")
        elif expected_mean != 0:
            relative_error = abs(diff) / abs(expected_mean)
        else:
            relative_error = 0.0 if diff == 0 else float("inf")
        results.append(KpiResult(
            column=column,
            passed=bool(means_finite and relative_error <= rtol),
            actual_mean=actual_mean,
            expected_mean=expected_mean,
            max_error=float(np.max(non_matching_err)) if non_matching_err.size else 0.0,
            error=diff,
            relative_error=relative_error,
            worst_rows=_worst_rows(
                err, matched, actual_np, expected_np, actual_table, id_columns
            ),
        ))

    return results
