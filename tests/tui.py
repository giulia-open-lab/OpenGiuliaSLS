from __future__ import annotations
import time
from typing import Callable

from rich.console import Console
from rich.live import Live
from rich.prompt import Prompt
from rich.table import Table
from rich import box

from tests.scenario import ScenarioConfig, base_names, select_by_base_name
from tests.compare import ScenarioResult
from tests.runner import RunResult

_console = Console()


def select_scenarios(all_scenarios: list[ScenarioConfig]) -> list[ScenarioConfig]:
    """Print a numbered list of base scenario names and let the user pick by
    index ranges. Preset expansion (e.g. the "_mfl" run) stays implicit here:
    picking a scenario runs every applicable preset for it automatically.
    """
    names = base_names(all_scenarios)
    _console.print("\n[bold]Available scenarios:[/bold]")
    for i, name in enumerate(names, 1):
        _console.print(f"  [{i:2d}]  {name}")
    _console.print()
    choice = Prompt.ask("Scenarios to run (e.g. 1,3,5-7 or 'all')", default="all")
    if choice.strip().lower() == "all":
        return all_scenarios
    return select_by_base_name(all_scenarios, _parse_selection(choice, names))


def _parse_selection(choice: str, names: list[str]) -> list[str]:
    indices: set[int] = set()
    for part in choice.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            indices.update(range(int(start), int(end) + 1))
        else:
            indices.add(int(part))
    return [names[i - 1] for i in sorted(indices) if 1 <= i <= len(names)]


def run_with_live_display(
    scenarios: list[ScenarioConfig],
    run_one: Callable[[ScenarioConfig], RunResult],
    compare_one: Callable[[RunResult], ScenarioResult | None],
    rtol: float,
) -> list[ScenarioResult]:
    """Run scenarios sequentially with a live-updating Rich table."""
    statuses: dict[str, str] = {s.name: "PENDING" for s in scenarios}
    start_times: dict[str, float] = {}
    end_times: dict[str, float] = {}
    failed_counts: dict[str, str] = {s.name: "" for s in scenarios}
    results: list[ScenarioResult] = []
    run_start = time.monotonic()

    def _elapsed(name: str) -> str:
        if name in end_times:
            return f"{end_times[name] - start_times[name]:.1f}"
        if name in start_times:
            return f"{time.monotonic() - start_times[name]:.1f}"
        return ""

    def _build_table() -> Table:
        total_elapsed = time.monotonic() - run_start
        table = Table(
            title=f"Running tests  (rtol={rtol:.3%})  •  total elapsed {total_elapsed:.0f}s",
            box=box.ROUNDED,
        )
        table.add_column("Scenario", style="white")
        # Fixed widths keep the table's rendered size constant across refreshes
        # (e.g. "RUNNING"/"PENDING" vs "PASS" differ in width) so the live
        # redraw doesn't visibly resize/jump on terminals with limited support
        # for in-place updates.
        table.add_column("Status", justify="center", width=9)
        table.add_column("Failed KPIs", justify="right", width=11)
        table.add_column("Time (s)", justify="right", width=10)
        for s in scenarios:
            st = statuses[s.name]
            if st == "PASS":
                status_cell = "[green]PASS[/green]"
            elif st in ("FAIL", "ERROR"):
                status_cell = f"[red]{st}[/red]"
            elif st == "RUNNING":
                status_cell = "[cyan]RUNNING[/cyan]"
            elif st == "DONE":
                status_cell = "[blue]DONE[/blue]"
            else:
                status_cell = "[dim]PENDING[/dim]"
            table.add_row(s.name, status_cell, failed_counts[s.name], _elapsed(s.name))
        return table

    with Live(
        get_renderable=_build_table,
        refresh_per_second=4,
        screen=True,
        redirect_stdout=True,
        redirect_stderr=True,
    ) as live:
        for scenario in scenarios:
            statuses[scenario.name] = "RUNNING"
            start_times[scenario.name] = time.monotonic()
            live.refresh()

            run_result = run_one(scenario)
            end_times[scenario.name] = time.monotonic()

            if run_result.error:
                statuses[scenario.name] = "ERROR"
                results.append(ScenarioResult(scenario_name=scenario.name, run_error=run_result.error))
            else:
                sr = compare_one(run_result)
                if sr is None:
                    statuses[scenario.name] = "DONE"
                elif sr.run_error:
                    # The simulation exited cleanly but exported nothing: an error, not a mismatch.
                    statuses[scenario.name] = "ERROR"
                elif sr.passed:
                    statuses[scenario.name] = "PASS"
                else:
                    statuses[scenario.name] = "FAIL"
                    failed_counts[scenario.name] = str(len(sr.failed_kpis))
                if sr is not None:
                    results.append(sr)

            live.refresh()

    return results
