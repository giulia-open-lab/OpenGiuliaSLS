from __future__ import annotations
from rich.console import Console
from rich.table import Table
from rich import box

from tests.compare import ScenarioResult

_console = Console()


def print_progress(scenario_name: str, status: str, elapsed_s: float | None = None) -> None:
    color = {
        "RUNNING": "cyan",
        "PASS": "green",
        "FAIL": "red",
        "ERROR": "red bold",
        "DONE": "blue",
    }.get(status, "white")
    suffix = f" ({elapsed_s:.1f}s)" if elapsed_s is not None else ""
    _console.print(f"[{color}][{status}][/{color}] {scenario_name}{suffix}")


def print_summary(results: list[ScenarioResult], rtol: float) -> None:
    table = Table(title=f"Test Results  (rtol={rtol:.3%})", box=box.ROUNDED)
    table.add_column("Scenario", style="white")
    table.add_column("Result", justify="center")
    table.add_column("Failed KPIs", justify="right")

    for r in results:
        if r.run_error:
            status = "[red bold]ERROR[/red bold]"
            failed = "-"
        elif r.passed:
            status = "[green]PASS[/green]"
            failed = "-"
        else:
            status = "[red]FAIL[/red]"
            failed = str(len(r.failed_kpis))
        table.add_row(r.scenario_name, status, failed)

    _console.print(table)


def print_scenario_detail(result: ScenarioResult) -> None:
    if result.run_error:
        _console.print(f"[red bold]{result.scenario_name}[/red bold]: {result.run_error}")
        return

    table = Table(title=result.scenario_name, box=box.SIMPLE)
    table.add_column("KPI", style="white")
    table.add_column("Actual Mean", justify="right")
    table.add_column("Expected Mean", justify="right")
    table.add_column("Error", justify="right")
    table.add_column("Relative Error", justify="right")
    table.add_column("Status", justify="center")

    for r in result.kpi_results:
        status = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
        table.add_row(
            r.column,
            f"{r.actual_mean:.4f}",
            f"{r.expected_mean:.4f}",
            f"{r.error:.4f}",
            _fmt_relative(r.relative_error),
            status,
        )
    _console.print(table)


def _fmt_relative(relative_error: float) -> str:
    """`relative_error` is a fraction (0.001 == 0.1%); render as a percentage,
    with explicit inf/nan labels since `{:.3%}` on them prints "infnan%" instead."""
    if relative_error != relative_error:  # NaN
        return "n/a"
    if relative_error == float("inf"):
        return "inf"
    return f"{relative_error:.3%}"
