from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import yaml

# --------------------------------------------------------------------------
# Preset expansion rule
#
# The `preset` value in a scenario YAML's `args` is ignored: every scenario
# is expanded into one run per preset below, regardless of what (if anything)
# its YAML file sets `preset` to.
#
# The only exception: single-carrier scenarios (name contains "C1" or "C2")
# are never run under GiuliaMfl, since multi-frequency-layer doesn't apply
# to them.
# --------------------------------------------------------------------------
PRESETS: tuple[str, ...] = ("GiuliaStd", "GiuliaMfl")
MFL_EXCLUDED_NAME_MARKERS: tuple[str, ...] = ("C1", "C2")


def supports_preset(scenario_name: str, preset: str) -> bool:
    """Whether `scenario_name` should be run under `preset`. See module docstring."""
    if preset == "GiuliaMfl" and any(marker in scenario_name for marker in MFL_EXCLUDED_NAME_MARKERS):
        return False
    return True
# --------------------------------------------------------------------------


@dataclass
class ScenarioConfig:
    name: str
    """Unique run identifier (e.g. "X" or "X_mfl"); used for output/reference dirs and reporting."""
    base_name: str
    """The scenario as authored in YAML and as picked by the user; preset expansion is invisible here."""
    args: dict
    example: str


def load_scenario(path: Path) -> list[ScenarioConfig]:
    """Load one scenario YAML file, expanded into one ScenarioConfig per
    applicable preset (see `supports_preset`). The GiuliaStd variant keeps
    the YAML's own name; other presets get a `_<preset>` suffix so they get
    their own output/reference directory.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    base_name = data["name"]
    example = data.get("run", {}).get("example", "EV")

    configs = []
    for preset in PRESETS:
        if not supports_preset(base_name, preset):
            continue
        args = {**data["args"], "preset": preset}
        name = base_name if preset == "GiuliaStd" else f"{base_name}_{preset[len('Giulia'):].lower()}"
        configs.append(ScenarioConfig(name=name, base_name=base_name, args=args, example=example))
    return configs


def load_all_scenarios(scenarios_dir: Path) -> list[ScenarioConfig]:
    scenarios = [
        scenario
        for path in scenarios_dir.glob("*.yaml")
        for scenario in load_scenario(path)
    ]
    return sorted(scenarios, key=lambda s: s.name)


def base_names(scenarios: list[ScenarioConfig]) -> list[str]:
    """Unique scenario names as authored/selected by the user (e.g. for --list and
    interactive selection). Preset expansion — and its "_mfl" naming — is an
    internal execution/reporting detail: picking one of these names runs every
    applicable preset for it automatically.
    """
    return sorted(dict.fromkeys(s.base_name for s in scenarios))


def select_by_base_name(scenarios: list[ScenarioConfig], names: list[str]) -> list[ScenarioConfig]:
    """Expand a list of base scenario names into their run-configs (every
    applicable preset for each name, per `supports_preset`)."""
    wanted = set(names)
    return [s for s in scenarios if s.base_name in wanted]
