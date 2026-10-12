"""Hardware/software fingerprint of the machine a test report was generated on.

Simulation output can drift across machines even with an identical seed: RNG
algorithms and floating-point reductions are sensitive to the CPU's SIMD
instruction set, the GPU model/driver, and the BLAS build linked into numpy.
Recording that context next to a report's KPI comparison makes "it doesn't
reproduce" bugs into "it doesn't reproduce *on different hardware*" ones.

Every collector here degrades gracefully (missing psutil/pynvml/torch, a
non-Linux OS, a numpy build without CPU dispatch info, ...) since this module
runs in both the full simulation environment and the lightweight unit-test one.
"""
from __future__ import annotations
import hashlib
import json
import os
import platform
import socket
from dataclasses import dataclass

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHONHASHSEED",
)


@dataclass
class GpuInfo:
    name: str
    vram_total_gb: float | None
    driver_version: str | None
    compute_capability: str | None


@dataclass
class SystemInfo:
    hostname: str
    os_name: str
    os_release: str
    architecture: str
    python_version: str
    cpu_model: str
    cpu_physical_cores: int | None
    cpu_logical_cores: int | None
    ram_total_gb: float | None
    gpus: list[GpuInfo]
    numpy_version: str | None
    blas_name: str | None
    blas_version: str | None
    cpu_simd_features: list[str]
    thread_env: dict[str, str]
    hardware_hash: str = ""


def _cpu_model() -> str:
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _cpu_counts() -> tuple[int | None, int | None]:
    logical = os.cpu_count()
    physical = None
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True) or logical
    except Exception:
        pass
    return physical, logical


def _ram_total_gb() -> float | None:
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        pass
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return round(int(line.split()[1]) / (1024 ** 2), 1)
        except OSError:
            pass
    return None


def _gpus_via_pynvml() -> list[GpuInfo] | None:
    """NVML (via the `pynvml` / `nvidia-ml-py` package) is preferred over torch:
    it reports VRAM and driver version without needing a CUDA context, and works
    even in a torch-less environment.
    """
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute_capability = f"{major}.{minor}"
            except Exception:
                compute_capability = None
            gpus.append(GpuInfo(
                name=name,
                vram_total_gb=round(mem.total / (1024 ** 3), 1),
                driver_version=driver,
                compute_capability=compute_capability,
            ))
        return gpus
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _gpus_via_torch() -> list[GpuInfo]:
    gpus = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append(GpuInfo(
                    name=props.name,
                    vram_total_gb=round(props.total_memory / (1024 ** 3), 1),
                    driver_version=None,
                    compute_capability=f"{props.major}.{props.minor}",
                ))
    except Exception:
        pass
    return gpus


def _gpus() -> list[GpuInfo]:
    return _gpus_via_pynvml() or _gpus_via_torch()


def _numpy_info() -> tuple[str | None, str | None, str | None, list[str]]:
    try:
        import numpy as np
    except ImportError:
        return None, None, None, []

    blas_name = blas_version = None
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        blas_name = blas.get("name")
        blas_version = blas.get("version")
    except Exception:
        pass

    # Which SIMD extensions the CPU actually has, as detected by numpy's runtime
    # dispatch - unlike the build-time "SIMD Extensions" info from show_config(),
    # this reflects *this* physical CPU rather than the wheel's compiled variants.
    features: list[str] = []
    try:
        raw = np._core._multiarray_umath.__cpu_features__
        features = sorted(name for name, present in raw.items() if present)
    except Exception:
        pass

    return np.__version__, blas_name, blas_version, features


def _hardware_fingerprint(info: SystemInfo) -> dict:
    """The subset of `info` that reflects physical hardware only - software
    versions (numpy/BLAS, GPU driver) and the OS/kernel are deliberately left
    out, so two machines with identical hardware hash the same regardless of
    which environment or driver happens to be installed on each.
    """
    return {
        "architecture": info.architecture,
        "cpu_model": info.cpu_model,
        "cpu_physical_cores": info.cpu_physical_cores,
        "cpu_logical_cores": info.cpu_logical_cores,
        "ram_total_gb": info.ram_total_gb,
        "cpu_simd_features": info.cpu_simd_features,
        "gpus": sorted(
            (g.name, g.vram_total_gb, g.compute_capability) for g in info.gpus
        ),
    }


def _hardware_hash(info: SystemInfo) -> str:
    payload = json.dumps(_hardware_fingerprint(info), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def collect_system_info() -> SystemInfo:
    cpu_physical, cpu_logical = _cpu_counts()
    numpy_version, blas_name, blas_version, simd_features = _numpy_info()

    info = SystemInfo(
        hostname=socket.gethostname(),
        os_name=platform.system(),
        os_release=platform.release(),
        architecture=platform.machine(),
        python_version=platform.python_version(),
        cpu_model=_cpu_model(),
        cpu_physical_cores=cpu_physical,
        cpu_logical_cores=cpu_logical,
        ram_total_gb=_ram_total_gb(),
        gpus=_gpus(),
        numpy_version=numpy_version,
        blas_name=blas_name,
        blas_version=blas_version,
        cpu_simd_features=simd_features,
        thread_env={k: os.environ[k] for k in _THREAD_ENV_VARS if k in os.environ},
    )
    info.hardware_hash = _hardware_hash(info)
    return info
