from __future__ import annotations
import multiprocessing as mp
import time
from dataclasses import dataclass
from pathlib import Path

from tests.scenario import ScenarioConfig


@dataclass
class RunResult:
    scenario: ScenarioConfig
    output_dir: Path
    elapsed_s: float
    error: Exception | None
    log_path: Path


def run_scenario(
    scenario: ScenarioConfig,
    output_base: Path,
    log_path: Path,
) -> RunResult:
    """Spawn a subprocess to run the simulator, isolating GPU memory between runs.

    stdout/stderr of the subprocess are redirected to log_path, so an unhandled
    exception's traceback ends up there for later inspection (e.g. in the HTML report).
    """
    output_dir = output_base / scenario.name
    start = time.monotonic()
    error: Exception | None = None
    try:
        p = mp.Process(
            target=_worker,
            args=(scenario.args, scenario.example, str(output_dir), str(log_path)),
        )
        p.start()
        p.join()
        if p.exitcode != 0:
            error = RuntimeError(f"Simulation process exited with code {p.exitcode}")
    except Exception as exc:
        error = exc
    return RunResult(
        scenario=scenario,
        output_dir=output_dir,
        elapsed_s=time.monotonic() - start,
        error=error,
        log_path=Path(log_path),
    )


def _worker(args: dict, example: str, output_dir: str, log_path: str) -> None:
    import sys
    from pathlib import Path
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    _log = open(log_path, "w")
    sys.stdout = _log
    sys.stderr = _log

    from giulia.inputs import InputConfig
    from giulia.inputs.config.parquet import ParquetFormat
    from giulia.inputs.run_config import RunConfig
    config = InputConfig(
        args=args,
        run_config=RunConfig(example),
        output_dir=output_dir,
        output_formats=[ParquetFormat(enable=True, compression=None, use_partition_cols=False, rm_uncompressed=False)],
    )
    main_fn, args_list = config.run()
    main_fn(args_list, config)
