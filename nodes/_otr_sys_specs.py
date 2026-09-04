"""Host system-spec collector for the episode treatment / credits sheet.

The treatment (``_write_story_treatment`` in ``video_engine.py``) embeds a
forensic SYSTEM block so anyone watching an episode can see exactly what
machine + software stack produced it -- CPU, RAM, GPU/VRAM, OS, and the
Python / torch / CUDA versions.

Hard contract: ``collect_system_specs`` is BEST-EFFORT and NEVER raises.
Every probe is individually guarded; a missing dependency (psutil, torch)
or an absent GPU degrades to a ``"(unknown)"`` string rather than failing
the render. The function returns a flat ``dict[str, str]`` ready to print.
"""

from __future__ import annotations

import os
import platform
import sys

try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore
try:
    from ._otr_shared import proc as otr_proc
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import proc as otr_proc  # type: ignore

_UNKNOWN = "(unknown)"


def _fmt_gib(num_bytes: float) -> str:
    try:
        return "%.1f GiB" % (float(num_bytes) / (1024 ** 3))
    except (TypeError, ValueError):
        return _UNKNOWN


def _cpu_model() -> str:
    """Human CPU model string. platform.processor() is often blank on Linux;
    fall back to /proc/cpuinfo, then the registry-free platform name."""
    name = (platform.processor() or "").strip()
    if name and name.lower() not in ("", "x86_64", "amd64", "i386"):
        return name
    # Linux: /proc/cpuinfo "model name"
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    # Windows: PROCESSOR_IDENTIFIER env carries a usable description
    env_id = (otr_env.get("PROCESSOR_IDENTIFIER") or "").strip()
    if env_id:
        return env_id
    return name or _UNKNOWN


def _cpu_cores() -> str:
    try:
        import psutil  # type: ignore

        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
        if physical and logical:
            return "%d physical / %d logical" % (physical, logical)
        if logical:
            return "%d logical" % logical
    except Exception:  # noqa: BLE001 -- psutil optional
        pass
    logical = os.cpu_count()
    return ("%d logical" % logical) if logical else _UNKNOWN


def _ram_total() -> str:
    try:
        import psutil  # type: ignore

        return _fmt_gib(psutil.virtual_memory().total)
    except Exception:  # noqa: BLE001 -- psutil optional
        pass
    # Linux fallback: MemTotal in /proc/meminfo (kB)
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    kb = float(line.split()[1])
                    return _fmt_gib(kb * 1024)
    except (OSError, IndexError, ValueError):
        pass
    return _UNKNOWN


def _gpu_via_torch() -> tuple[str, str, str]:
    """Return (gpu_name, vram_total, cuda_version) via torch, or unknowns."""
    try:
        import torch  # type: ignore

        cuda_ver = getattr(torch.version, "cuda", None) or _UNKNOWN
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            vram = _fmt_gib(getattr(props, "total_memory", 0))
            return (name or _UNKNOWN, vram, str(cuda_ver))
        return (_UNKNOWN, _UNKNOWN, str(cuda_ver))
    except Exception:  # noqa: BLE001 -- torch optional / no GPU
        return (_UNKNOWN, _UNKNOWN, _UNKNOWN)


def _gpu_via_nvidia_smi() -> tuple[str, str]:
    """Return (gpu_name, vram_total) by shelling nvidia-smi, or unknowns.

    Used only when torch could not report a GPU (e.g. the treatment runs in a
    torch-less context). Bounded 5s timeout; any failure -> unknowns."""
    try:
        out = otr_proc.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        first = (out.stdout or "").strip().splitlines()
        if first:
            parts = [p.strip() for p in first[0].split(",")]
            name = parts[0] if parts else _UNKNOWN
            vram = parts[1] if len(parts) > 1 else _UNKNOWN
            return (name or _UNKNOWN, vram or _UNKNOWN)
    except Exception:  # noqa: BLE001 -- nvidia-smi absent / timeout
        pass
    return (_UNKNOWN, _UNKNOWN)


def _ram_peak() -> str:
    """Peak RAM used by THIS process (the render server) so far. Best-effort.

    Windows: psutil ``peak_wset``. Linux: ``resource.ru_maxrss`` (kB).
    macOS: ``ru_maxrss`` (bytes). Reports the render process high-water mark,
    not a system-wide peak -- the most truthful per-render number available."""
    try:
        import psutil  # type: ignore

        mi = psutil.Process().memory_info()
        peak = getattr(mi, "peak_wset", None)  # Windows-only field
        if peak:
            return _fmt_gib(peak)
    except Exception:  # noqa: BLE001 -- psutil optional
        pass
    try:
        import resource  # type: ignore  # POSIX-only

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return _fmt_gib(ru)            # macOS: bytes
        return _fmt_gib(ru * 1024)         # Linux: kB
    except Exception:  # noqa: BLE001 -- resource absent on Windows
        pass
    return _UNKNOWN


def _torch_version() -> str:
    try:
        import torch  # type: ignore

        return str(getattr(torch, "__version__", _UNKNOWN))
    except Exception:  # noqa: BLE001
        return _UNKNOWN


def collect_system_specs() -> dict:
    """Gather host specs for the treatment SYSTEM block. Never raises."""
    specs: dict[str, str] = {}
    try:
        specs["os"] = "%s %s (%s)" % (
            platform.system() or _UNKNOWN,
            platform.release() or "",
            platform.machine() or "",
        )
    except Exception:  # noqa: BLE001
        specs["os"] = _UNKNOWN

    specs["cpu"] = _cpu_model()
    specs["cpu_cores"] = _cpu_cores()
    specs["ram"] = _ram_total()
    specs["ram_peak"] = _ram_peak()

    gpu_name, vram, cuda_ver = _gpu_via_torch()
    if gpu_name == _UNKNOWN:
        smi_name, smi_vram = _gpu_via_nvidia_smi()
        if smi_name != _UNKNOWN:
            gpu_name, vram = smi_name, smi_vram
    specs["gpu"] = gpu_name
    specs["vram"] = vram
    specs["cuda"] = cuda_ver

    specs["python"] = platform.python_version() or _UNKNOWN
    specs["torch"] = _torch_version()
    specs["hostname"] = (platform.node() or _UNKNOWN)
    return specs


__all__ = ["collect_system_specs"]
