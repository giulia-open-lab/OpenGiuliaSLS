from __future__ import annotations
import html
import io
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.terminal_theme import MONOKAI
from rich.text import Text

from tests.compare import KpiResult, ScenarioResult
from tests.runner import RunResult
from tests.scenario import ScenarioConfig
from tests.system_info import SystemInfo, collect_system_info

REPORTS_DIR = Path(__file__).parent.parent / "outputs" / "test_reports"

_TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\):.*", re.DOTALL)


@dataclass
class ScenarioEntry:
    name: str
    status: str  # PASS | FAIL | ERROR | GENERATED
    elapsed_s: float
    log_text: str
    error: Exception | None
    kpi_results: list[KpiResult] = field(default_factory=list)

    @property
    def traceback(self) -> str | None:
        if self.error is None:
            return None
        m = _TRACEBACK_RE.search(self.log_text)
        return m.group(0) if m else None

    @property
    def failed_kpis(self) -> list[KpiResult]:
        return [r for r in self.kpi_results if not r.passed]


def build_entries(
    scenarios: list[ScenarioConfig],
    run_results: dict[str, RunResult],
    scenario_results: list[ScenarioResult],
    generate_mode: bool,
) -> list[ScenarioEntry]:
    by_name = {r.scenario_name: r for r in scenario_results}
    entries: list[ScenarioEntry] = []
    for scenario in scenarios:
        rr = run_results.get(scenario.name)
        if rr is None:
            continue
        try:
            log_text = rr.log_path.read_text(errors="replace")
        except OSError:
            log_text = ""

        sr = by_name.get(scenario.name)
        # The run itself can fail (non-zero exit), but so can the comparison at the
        # scenario level - e.g. the simulation exited cleanly yet exported nothing.
        # Both are errors rather than KPI mismatches.
        error = rr.error or (sr.run_error if sr is not None else None)
        if error is not None:
            status = "ERROR"
        elif generate_mode:
            status = "GENERATED"
        elif sr is not None:
            status = "PASS" if sr.passed else "FAIL"
        else:
            status = "DONE"

        entries.append(ScenarioEntry(
            name=scenario.name,
            status=status,
            elapsed_s=rr.elapsed_s,
            log_text=log_text,
            error=error,
            kpi_results=sr.kpi_results if sr is not None else [],
        ))
    return entries


def generate_report(
    *,
    scenarios: list[ScenarioConfig],
    run_results: dict[str, RunResult],
    scenario_results: list[ScenarioResult],
    options: dict[str, Any],
    started_at: datetime,
    finished_at: datetime,
    generate_mode: bool,
    tolerance: float,
) -> Path:
    entries = build_entries(scenarios, run_results, scenario_results, generate_mode)
    system_info = collect_system_info()
    html_text = _render(entries, options, started_at, finished_at, tolerance, system_info)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"report_{started_at:%Y%m%d_%H%M%S}.html"
    report_path.write_text(html_text, encoding="utf-8")
    return report_path


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _mono(value: Any) -> str:
    return f"<code>{_esc(value)}</code>"


def _ansi_to_html(text: str) -> str:
    """Render ANSI SGR color codes (giulia.logger prints them unconditionally,
    even when stdout is redirected to a log file) as inline-styled HTML spans.
    Falls back to plain escaped text when there's nothing to convert.
    """
    if "\x1b[" not in text:
        return _esc(text)
    buf = io.StringIO()
    console = Console(record=True, file=buf, width=1_000_000, highlight=False)
    console.print(Text.from_ansi(text), soft_wrap=True)
    # MONOKAI's brighter palette stays legible against both the light and dark
    # page backgrounds (unlike the default theme's navy/maroon/olive, chosen
    # for light-on-white terminals).
    return console.export_html(inline_styles=True, code_format="{code}", theme=MONOKAI).rstrip("\n")


_STATUS_COLORS = {
    "PASS": "#2e7d32",
    "GENERATED": "#2e7d32",
    "FAIL": "#c62828",
    "ERROR": "#c62828",
    "DONE": "#616161",
}


def _status_badge(status: str) -> str:
    color = _STATUS_COLORS.get(status, "#616161")
    return f'<span class="badge" style="background:{color}">{_esc(status)}</span>'


def _options_table(options: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{_mono(key)}</td><td>{_mono(value)}</td></tr>"
        for key, value in sorted(options.items())
    )
    return f"<table class='options'><thead><tr><th>Option</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>"


def _worst_rows_table(k: KpiResult) -> str:
    """The individual cells with the largest per-row error behind a failed KPI, so it
    can be traced to a concrete row instead of just the aggregate means. Columns are
    whichever id fields (episode_idx, step_idx, cell_id/ue_id/...) that KPI's parquet
    file carries.
    """
    if not k.worst_rows:
        return ""
    id_cols = list(k.worst_rows[0].ids.keys())
    header = "".join(f"<th>{_esc(c)}</th>" for c in id_cols) + "<th>Expected</th><th>Actual</th><th>Error</th>"
    body_rows = []
    for w in k.worst_rows:
        id_cells = "".join(f"<td>{_esc(w.ids[c])}</td>" for c in id_cols)
        body_rows.append(
            f"<tr>{id_cells}"
            f"<td>{w.expected:.6g}</td><td>{w.actual:.6g}</td><td>{w.error:.6g}</td></tr>"
        )
    return (
        "<div class='worst-rows'>"
        f"<div class='worst-rows-label'>Worst {len(k.worst_rows)} cell(s) by absolute error:</div>"
        "<table class='worst'><thead><tr>" + header + "</tr></thead><tbody>"
        + "".join(body_rows) + "</tbody></table></div>"
    )


def _gpu_summary(info: SystemInfo) -> str:
    if not info.gpus:
        return "<em>No GPU detected</em>"
    parts = []
    for g in info.gpus:
        bits = [_esc(g.name)]
        if g.vram_total_gb is not None:
            bits.append(f"{g.vram_total_gb:g} GB VRAM")
        if g.compute_capability:
            bits.append(f"compute capability {_esc(g.compute_capability)}")
        if g.driver_version:
            bits.append(f"driver {_esc(g.driver_version)}")
        parts.append(", ".join(bits))
    return "<br>".join(parts)


def _system_info_box(info: SystemInfo) -> str:
    cpu_cores = (
        f"{info.cpu_physical_cores} physical / {info.cpu_logical_cores} logical cores"
        if info.cpu_physical_cores
        else f"{info.cpu_logical_cores} logical cores"
    )
    numpy_blas = _esc(info.numpy_version or "unknown")
    if info.blas_name:
        numpy_blas += f" &mdash; {_esc(info.blas_name)} {_esc(info.blas_version or '')}".rstrip()
    simd = ", ".join(info.cpu_simd_features) if info.cpu_simd_features else "unknown"
    thread_env = (
        ", ".join(f"{k}={v}" for k, v in info.thread_env.items())
        if info.thread_env
        else "(none set)"
    )
    rows = "".join(
        f"<tr><td>{label}</td><td>{value}</td></tr>"
        for label, value in (
            ("Hostname", _mono(info.hostname)),
            ("OS", f"{_esc(info.os_name)} {_esc(info.os_release)} ({_esc(info.architecture)})"),
            ("Python", _esc(info.python_version)),
            ("CPU", f"{_esc(info.cpu_model)} ({cpu_cores})"),
            ("RAM", f"{info.ram_total_gb:g} GB" if info.ram_total_gb else "unknown"),
            ("GPU", _gpu_summary(info)),
            ("numpy / BLAS", numpy_blas),
            ("CPU SIMD features", f"<code>{_esc(simd)}</code>"),
            ("Thread-count env vars", _mono(thread_env)),
        )
    )
    return f"""
    <details class="sysinfo" open>
      <summary>Report generation environment</summary>
      <table class='sysinfo-table'>{rows}</table>
      <div class="hash-box">
        Hardware hash: {_mono(info.hardware_hash)}
        <span class="muted">&mdash; derived only from CPU/GPU/RAM/SIMD; matches across machines with identical
        hardware regardless of OS, drivers, or installed software versions. Since RNG output and floating-point
        reductions can differ across hardware even with the same seed, a differing hash between two reports
        is a hint to check before treating a KPI drift as a real regression.</span>
      </div>
    </details>
    """


def _fmt_relative(relative_error: float) -> str:
    """`relative_error` is a fraction (0.001 == 0.1%); render as a percentage,
    with explicit inf/NaN labels since a plain format spells them "infnan%"."""
    if relative_error != relative_error:  # NaN
        return "n/a"
    if relative_error == float("inf"):
        return "inf"
    return f"{relative_error:.3%}"


def _kpi_table(entry: ScenarioEntry) -> str:
    if not entry.kpi_results:
        return "<p class='muted'>No KPI comparison was performed for this scenario.</p>"

    rows = []
    for k in entry.kpi_results:
        row_class = "" if k.passed else "row-fail"
        status = "PASS" if k.passed else "FAIL"
        rows.append(
            f"<tr class='{row_class}'>"
            f"<td>{_mono(k.column)}</td>"
            f"<td>{k.expected_mean:.6g}</td>"
            f"<td>{k.actual_mean:.6g}</td>"
            f"<td>{k.error:.6g}</td>"
            f"<td>{_fmt_relative(k.relative_error)}</td>"
            f"<td>{_status_badge(status)}</td>"
            f"</tr>"
        )
        if not k.passed and k.worst_rows:
            rows.append(
                f"<tr class='row-fail-detail'><td colspan='6'>{_worst_rows_table(k)}</td></tr>"
            )
    return (
        "<table class='kpi'><thead><tr>"
        "<th>KPI</th><th>Expected mean</th><th>Actual mean</th><th>Error</th><th>Relative error</th>"
        "<th>Status</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _scenario_section(entry: ScenarioEntry) -> str:
    open_attr = " open" if entry.status in ("FAIL", "ERROR") else ""
    failed_count = len(entry.failed_kpis)
    failed_note = f" &mdash; {failed_count} KPI(s) failed" if failed_count else ""

    error_block = ""
    if entry.error is not None:
        tb = entry.traceback
        error_block = (
            "<div class='error-box'>"
            f"<div class='error-message'>{_esc(entry.error)}</div>"
        )
        if tb:
            error_block += f"<pre class='traceback'>{_ansi_to_html(tb)}</pre>"
        error_block += "</div>"

    tab_id = re.sub(r"[^a-zA-Z0-9_-]", "_", entry.name)
    return f"""
    <details class="scenario status-{entry.status.lower()}"{open_attr}>
      <summary>
        {_status_badge(entry.status)}
        <span class="scenario-name">{_mono(entry.name)}</span>
        <span class="elapsed">{entry.elapsed_s:.1f}s</span>
        <span class="failed-note">{failed_note}</span>
      </summary>
      <div class="scenario-body">
        {error_block}
        <div class="tabs" data-group="{tab_id}">
          <button class="tab-btn active" data-target="kpi-{tab_id}">KPIs</button>
          <button class="tab-btn" data-target="log-{tab_id}">Log</button>
        </div>
        <div id="kpi-{tab_id}" class="tab-panel active">
          {_kpi_table(entry)}
        </div>
        <div id="log-{tab_id}" class="tab-panel">
          <pre class="log">{_ansi_to_html(entry.log_text) or '<em>(no log output)</em>'}</pre>
        </div>
      </div>
    </details>
    """


def _render(
    entries: list[ScenarioEntry],
    options: dict[str, Any],
    started_at: datetime,
    finished_at: datetime,
    tolerance: float,
    system_info: SystemInfo,
) -> str:
    counts = {"PASS": 0, "FAIL": 0, "ERROR": 0, "GENERATED": 0, "DONE": 0}
    for e in entries:
        counts[e.status] = counts.get(e.status, 0) + 1

    summary_rows = "".join(
        f"<tr onclick=\"document.getElementById('scn-{re.sub(r'[^a-zA-Z0-9_-]', '_', e.name)}').scrollIntoView({{behavior:'smooth'}})\">"
        f"<td>{_mono(e.name)}</td><td>{_status_badge(e.status)}</td>"
        f"<td>{len(e.failed_kpis) if e.kpi_results else '-'}</td>"
        f"<td>{e.elapsed_s:.1f}</td></tr>"
        for e in entries
    )

    duration = (finished_at - started_at).total_seconds()

    sections = "\n".join(
        f'<a id="scn-{re.sub(r"[^a-zA-Z0-9_-]", "_", e.name)}"></a>' + _scenario_section(e)
        for e in entries
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>GiuliaSLS Test Report &mdash; {started_at:%Y-%m-%d %H:%M:%S}</title>
<style>
{_CSS}
</style>
</head>
<body>
<header>
  <h1>GiuliaSLS Regression Test Report</h1>
  <div class="meta">
    <div><strong>Started:</strong> {started_at:%Y-%m-%d %H:%M:%S}</div>
    <div><strong>Finished:</strong> {finished_at:%Y-%m-%d %H:%M:%S}</div>
    <div><strong>Duration:</strong> {duration:.1f}s</div>
    <div><strong>Tolerance (relative):</strong> {tolerance:.3%}</div>
  </div>
</header>

<section class="summary-cards">
  <div class="card total"><div class="num">{len(entries)}</div><div class="label">Total</div></div>
  <div class="card pass"><div class="num">{counts["PASS"] + counts["GENERATED"]}</div><div class="label">Passed</div></div>
  <div class="card fail"><div class="num">{counts["FAIL"]}</div><div class="label">Failed</div></div>
  <div class="card error"><div class="num">{counts["ERROR"]}</div><div class="label">Errors</div></div>
</section>

{_system_info_box(system_info)}

<details class="options" open>
  <summary>Run options</summary>
  {_options_table({k: v for k, v in options.items() if k != "tolerance"})}
</details>

<details class="summary-table" open>
  <summary>Scenario summary</summary>
  <table class="summary">
    <thead><tr><th>Scenario</th><th>Status</th><th>Failed KPIs</th><th>Time (s)</th></tr></thead>
    <tbody>{summary_rows}</tbody>
  </table>
</details>

<h2>Scenarios</h2>
{sections}

<script>
{_JS}
</script>
</body>
</html>
"""


_CSS = """
:root {
  color-scheme: light dark;
  --bg: #ffffff;
  --fg: #1a1a1a;
  --muted: #666;
  --border: #ddd;
  --card-bg: #f5f5f5;
  --row-fail-bg: #fdecea;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #1e1e1e;
    --fg: #e0e0e0;
    --muted: #999;
    --border: #3a3a3a;
    --card-bg: #2a2a2a;
    --row-fail-bg: #3a1f1f;
  }
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--fg);
  margin: 0;
  padding: 24px;
  max-width: 1100px;
  margin: 0 auto;
}
header h1 { margin-bottom: 4px; }
.meta { display: flex; gap: 24px; color: var(--muted); font-size: 0.9em; flex-wrap: wrap; }
.summary-cards { display: flex; gap: 12px; margin: 20px 0; flex-wrap: wrap; }
.card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px 20px; text-align: center; min-width: 90px; }
.card .num { font-size: 1.8em; font-weight: 700; }
.card .label { color: var(--muted); font-size: 0.85em; text-transform: uppercase; }
.card.pass .num { color: #2e7d32; }
.card.fail .num, .card.error .num { color: #c62828; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; }
table th, table td { border: 1px solid var(--border); padding: 6px 10px; text-align: left; font-size: 0.9em; }
table th { background: var(--card-bg); }
table.summary tbody tr { cursor: pointer; }
table.summary tbody tr:hover { background: var(--card-bg); }
tr.row-fail { background: var(--row-fail-bg); }
tr.row-fail-detail td { padding: 0; border-top: none; }
.worst-rows { background: var(--row-fail-bg); border-top: 1px dashed var(--border); padding: 8px 10px 10px; }
.worst-rows-label { font-size: 0.8em; color: var(--muted); margin-bottom: 4px; }
table.worst { margin: 0; font-size: 0.9em; }
table.worst th, table.worst td { background: var(--bg); }
details { border: 1px solid var(--border); border-radius: 8px; margin: 10px 0; padding: 8px 12px; background: var(--card-bg); }
details.scenario { background: var(--bg); }
summary { cursor: pointer; font-weight: 600; display: flex; align-items: center; gap: 10px; }
.scenario-name { flex: 1; }
.elapsed, .failed-note { color: var(--muted); font-weight: 400; font-size: 0.85em; }
.badge { color: #fff; border-radius: 4px; padding: 2px 8px; font-size: 0.75em; font-weight: 700; }
.scenario-body { padding-top: 10px; }
.error-box { background: var(--row-fail-bg); border: 1px solid #c62828; border-radius: 6px; padding: 10px; margin-bottom: 10px; }
.error-message { color: #c62828; font-weight: 600; }
pre.traceback { white-space: pre-wrap; overflow-x: auto; margin-top: 8px; }
pre.log { white-space: pre-wrap; word-break: break-word; max-height: 500px; overflow-y: auto; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px; font-size: 0.82em; }
.tabs { display: flex; gap: 4px; margin-bottom: 8px; }
.tab-btn { border: 1px solid var(--border); background: var(--card-bg); color: var(--fg); border-radius: 6px 6px 0 0; padding: 6px 14px; cursor: pointer; font-size: 0.85em; }
.tab-btn.active { background: var(--bg); font-weight: 700; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }
.muted { color: var(--muted); }
table.sysinfo-table td:first-child { color: var(--muted); white-space: nowrap; width: 1%; }
.hash-box { margin-top: 10px; font-size: 0.9em; }
.hash-box .muted { display: block; margin-top: 4px; font-size: 0.92em; line-height: 1.4; }
code { font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace; font-size: 0.92em; background: var(--card-bg); border-radius: 4px; padding: 1px 5px; }
"""

_JS = """
document.querySelectorAll('.tabs').forEach(function (tabs) {
  tabs.querySelectorAll('.tab-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var body = tabs.parentElement;
      tabs.querySelectorAll('.tab-btn').forEach(function (b) { b.classList.remove('active'); });
      body.querySelectorAll('.tab-panel').forEach(function (p) { p.classList.remove('active'); });
      btn.classList.add('active');
      document.getElementById(btn.dataset.target).classList.add('active');
    });
  });
});
"""
