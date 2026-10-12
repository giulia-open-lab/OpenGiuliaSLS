#!/usr/bin/env python3
"""
GiuliaSLS regression test runner.

Usage:
    python tests/run_tests.py [SCENARIO ...] [OPTIONS]

Examples:
    python tests/run_tests.py                             # run all scenarios
    python tests/run_tests.py ITU_R_M2135_UMa_uniform    # run one scenario
    python tests/run_tests.py --list                      # list available scenarios
    python tests/run_tests.py --generate                  # regenerate reference files
    python tests/run_tests.py --tui                       # interactive mode

Environment variables:
    GIULIA_SCENARIOS     Space-separated list of scenario names (overridden by positional args)
    PRECOMPUTED_FILES    Path to the reference files directory
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.scenario import ScenarioConfig, base_names, load_all_scenarios, select_by_base_name
from tests.runner import RunResult, run_scenario
from tests.compare import ScenarioResult, compare_scenario
from tests.references import ensure_reference, ensure_shadowing, resolve_references_dir
from tests.report import print_progress, print_summary, print_scenario_detail
from tests.html_report import generate_report

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
DEFAULT_OUTPUT_BASE = Path(__file__).parent.parent / "outputs" / "test_runs"
DEFAULT_RTOL = 1/100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GiuliaSLS regression test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "scenarios",
        nargs="*",
        metavar="SCENARIO",
        help="Names of scenarios to run (default: all)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Run simulations and write outputs as new reference files",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_RTOL,
        metavar="FLOAT",
        help=(
            "Relative tolerance for KPI comparison, as a fraction of the expected mean "
            f"(default: {DEFAULT_RTOL}, i.e. {DEFAULT_RTOL:.1%}".replace("%", "%%") + ")"
        ),
    )
    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Override the reference files directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        metavar="PATH",
        help="Where simulation outputs are written during comparison runs",
    )
    download_group = parser.add_mutually_exclusive_group()
    download_group.add_argument(
        "--download",
        dest="download",
        action="store_true",
        help="Download missing shadowing files without prompting",
    )
    download_group.add_argument(
        "--no-download",
        dest="download",
        action="store_false",
        help="Fail immediately if shadowing files are missing instead of prompting",
    )
    parser.set_defaults(download=None)
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Interactive mode: scenario selector + live progress table (requires TTY)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print available scenario names and exit",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format for test results (default: text)",
    )
    return parser.parse_args()


def _resolve_selected(
    all_scenarios: list[ScenarioConfig], names: list[str]
) -> list[ScenarioConfig]:
    if not names:
        env_names = [n for n in os.environ.get("GIULIA_SCENARIOS", "").split() if n]
        names = env_names
    if not names:
        return all_scenarios

    available = base_names(all_scenarios)
    unknown = [n for n in names if n not in available]
    if unknown:
        print(f"Unknown scenario(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"Available: {', '.join(available)}", file=sys.stderr)
        sys.exit(1)
    return select_by_base_name(all_scenarios, names)


def _filter_missing_references(
    scenarios: list[ScenarioConfig], references_dir: Path
) -> list[ScenarioConfig]:
    """Keep only scenarios with usable reference data, skipping (with a warning)
    any that are missing instead of aborting the whole run — e.g. a freshly
    added preset variant (like "_mfl") that hasn't been baselined with
    --generate yet shouldn't block every other scenario from running.
    """
    runnable = []
    for scenario in scenarios:
        try:
            ensure_reference(scenario.name, references_dir)
            runnable.append(scenario)
        except FileNotFoundError as exc:
            print(f"Skipping '{scenario.name}': {exc}", file=sys.stderr)
    return runnable


def _results_to_json(results: list[ScenarioResult]) -> str:
    data = [
        {
            "scenario": r.scenario_name,
            "passed": r.passed,
            "error": str(r.run_error) if r.run_error else None,
            "kpis": [
                {
                    "column": k.column,
                    "passed": k.passed,
                    "actual_mean": k.actual_mean,
                    "expected_mean": k.expected_mean,
                    "max_error": k.max_error,
                    "error": k.error,
                    "relative_error": k.relative_error,
                }
                for k in r.kpi_results
            ],
        }
        for r in results
    ]
    return json.dumps(data, indent=2)


def main() -> None:
    args = parse_args()
    started_at = datetime.now()
    all_scenarios = load_all_scenarios(SCENARIOS_DIR)

    if args.list:
        for name in base_names(all_scenarios):
            print(name)
        return

    references_dir = args.ref_dir or resolve_references_dir()
    selected = _resolve_selected(all_scenarios, args.scenarios)
    auto_download: bool | None = args.download

    ensure_shadowing(auto=auto_download)

    output_base = references_dir if args.generate else args.output_dir
    output_base.mkdir(parents=True, exist_ok=True)

    logs_dir = Path(tempfile.mkdtemp(prefix="giulia_test_logs_"))
    run_results: dict[str, RunResult] = {}

    def run_one(scenario: ScenarioConfig) -> RunResult:
        rr = run_scenario(scenario, output_base, log_path=logs_dir / f"{scenario.name}.log")
        run_results[scenario.name] = rr
        return rr

    def compare_one(run_result: RunResult) -> ScenarioResult | None:
        if args.generate:
            return None
        return compare_scenario(
            run_result.output_dir,
            references_dir / run_result.scenario.name,
            rtol=args.tolerance,
        )

    use_tui = args.tui and sys.stdout.isatty()
    if use_tui:
        from tests.tui import select_scenarios
        selected = select_scenarios(selected)

    if not args.generate:
        selected = _filter_missing_references(selected, references_dir)
        if not selected:
            print(
                "No scenarios with available reference data to run. "
                "Use --generate to create baselines for them.",
                file=sys.stderr,
            )
            sys.exit(1)

    if use_tui:
        from tests.tui import run_with_live_display
        results = run_with_live_display(selected, run_one, compare_one, args.tolerance)
    else:
        results = _run_headless(selected, run_one, compare_one)

    finished_at = datetime.now()

    report_path = generate_report(
        scenarios=selected,
        run_results=run_results,
        scenario_results=results,
        options=vars(args),
        started_at=started_at,
        finished_at=finished_at,
        generate_mode=args.generate,
        tolerance=args.tolerance,
    )
    shutil.rmtree(logs_dir, ignore_errors=True)
    print(f"\nHTML report: {report_path.resolve().as_uri()}")

    if args.generate:
        print(f"Reference files written to {references_dir}")
        return

    if args.format == "json":
        print(_results_to_json(results))
    else:
        print_summary(results, rtol=args.tolerance)
        for r in results:
            if not r.passed:
                print_scenario_detail(r)

    sys.exit(0 if all(r.passed for r in results) else 1)


def _run_headless(
    selected: list[ScenarioConfig],
    run_one,
    compare_one,
) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    for scenario in selected:
        print_progress(scenario.name, "RUNNING")
        run_result = run_one(scenario)

        if run_result.error:
            print_progress(scenario.name, "ERROR", run_result.elapsed_s)
            print(f"  {run_result.error}", file=sys.stderr)
            results.append(ScenarioResult(scenario_name=scenario.name, run_error=run_result.error))
            continue

        sr = compare_one(run_result)

        if sr is None:
            print_progress(scenario.name, "DONE", run_result.elapsed_s)
            continue

        if sr.run_error:
            print_progress(scenario.name, "ERROR", run_result.elapsed_s)
            print(f"  {sr.run_error}", file=sys.stderr)
        else:
            print_progress(scenario.name, "PASS" if sr.passed else "FAIL", run_result.elapsed_s)
        results.append(sr)

    return results


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn")
    main()
