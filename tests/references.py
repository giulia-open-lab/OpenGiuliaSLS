from __future__ import annotations
import os
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

SHADOWING_URL = "http://158.42.160.122:1234/shadowing.zip"
DEFAULT_REFERENCES_DIR = Path(__file__).parent / "references"


def resolve_references_dir() -> Path:
    env = os.environ.get("PRECOMPUTED_FILES")
    return Path(env) if env else DEFAULT_REFERENCES_DIR


def resolve_shadowing_dir() -> Path:
    from giulia.fs.shadowing import shadowing_dir
    return Path(shadowing_dir())


def ensure_reference(scenario_name: str, references_dir: Path) -> Path:
    """
    Return the path to the reference Parquet directory for scenario_name.
    """
    ref_path = references_dir / scenario_name
    if not _is_valid_parquet_dir(ref_path):
        raise FileNotFoundError(
            f"Reference files for '{scenario_name}' not found at {ref_path}.\n"
            "Run with --generate to create them."
        )
    return ref_path


def ensure_shadowing(auto: bool | None = None) -> None:
    """
    Ensure shadowing .mat files are present.
    auto=True  → download silently
    auto=None  → prompt the user (y/N)
    auto=False → raise FileNotFoundError
    """
    shadowing_dir = resolve_shadowing_dir()
    if _has_mat_files(shadowing_dir):
        return

    if auto is False:
        raise FileNotFoundError(
            f"Shadowing files not found at {shadowing_dir}.\n"
            "Run with --download to fetch them."
        )
    if auto is None:
        answer = input(
            f"Shadowing files are missing at {shadowing_dir}. Download now? [y/N] "
        ).strip().lower()
        if answer != "y":
            raise FileNotFoundError("Shadowing files missing. Aborting.")

    shadowing_dir.mkdir(parents=True, exist_ok=True)
    _download_and_unzip(SHADOWING_URL, shadowing_dir.parent, desc="Downloading shadowing files")


def _is_valid_parquet_dir(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.parquet"))


def _has_mat_files(path: Path) -> bool:
    return path.exists() and any(path.glob("*.mat"))


def _download_and_unzip(url: str, target_dir: Path | str, desc: str) -> None:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "_download.zip"

    class _Progress(tqdm):
        def update_to(self, current, total):
            self.total = total
            self.update(current - self.n)

    with _Progress(unit="B", unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, zip_path, lambda c, b, s: t.update_to(c * b, s))

    with zipfile.ZipFile(zip_path) as zf:
        for member in tqdm(zf.infolist(), desc="Extracting"):
            try:
                zf.extract(member, target_dir)
            except zipfile.error:
                pass

    zip_path.unlink()
