"""
otr_v2.hyworld.lhm_monitor  --  Day 13 LibreHardwareMonitor poller
===================================================================

Background telemetry sampler for the 20-min overnight dry run.  Polls
``http://localhost:8085/data.json`` at a fixed interval (default 60 s)
and appends one JSON object per sample to an NDJSON log.  Summarises
the log into peak / mean / min for each tracked sensor so the Day 13
ROADMAP bar ("no OOM, no pagefile thrash, no shared-memory fallback")
has a machine-readable artefact instead of a screenshot.

Pure stdlib -- no torch, no numpy.  Safe to import from unit tests.
Shares the LHM endpoint + parser with ``backends._base._read_lhm_gpu_temp``
but adds VRAM-used, VRAM-total, RAM, and CPU temperature extraction
so the summariser can flag pagefile thrash (RAM > 14 GB sustained on
this 16 GB machine) and shared-memory fallback (GPU dedicated memory
pinned at 100 %).

No audio imports.  C7 audio byte-identical gate is unaffected.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable


# ---- Endpoint --------------------------------------------------------------

_LHM_URL = os.environ.get("OTR_LHM_URL", "http://localhost:8085/data.json")
_DEFAULT_INTERVAL_S = 60.0
_DEFAULT_TIMEOUT_S = 3.0

# Day 13 kill criteria (ROADMAP):
#   - No OOM (RUNNING -> OOM transitions in STATUS.json)
#   - No pagefile thrash (system RAM must stay under the thrash ceiling)
#   - No shared-memory fallback (GPU dedicated VRAM stays under ceiling)
#
# For a 16 GB-VRAM Blackwell laptop with 32 GB system RAM, reasonable
# ceilings are 14.5 GB VRAM and 28 GB RAM (leaving OS + ComfyUI headroom).
VRAM_CEILING_GB: float = 14.5
RAM_CEILING_GB: float = 28.0
GPU_TEMP_CEILING_C: float = 85.0


# ---- Single poll -----------------------------------------------------------


@dataclass
class LhmSample:
    """One snapshot from LibreHardwareMonitor.  Missing values are None
    so the summariser can skip rather than break on partial telemetry."""

    t_monotonic: float
    t_unix: float
    gpu_temp_c: float | None = None
    gpu_vram_used_gb: float | None = None
    gpu_vram_total_gb: float | None = None
    ram_used_gb: float | None = None
    ram_total_gb: float | None = None
    cpu_temp_c: float | None = None
    unreachable: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_monotonic": round(self.t_monotonic, 3),
            "t_unix": round(self.t_unix, 3),
            "gpu_temp_c": self.gpu_temp_c,
            "gpu_vram_used_gb": self.gpu_vram_used_gb,
            "gpu_vram_total_gb": self.gpu_vram_total_gb,
            "ram_used_gb": self.ram_used_gb,
            "ram_total_gb": self.ram_total_gb,
            "cpu_temp_c": self.cpu_temp_c,
            "unreachable": self.unreachable,
            "reason": self.reason,
        }


def _parse_value_number(value_text: str) -> float | None:
    """Extract a numeric value from an LHM ``Value`` string like
    ``"72.5 C"`` or ``"6.4 GB"``.  Returns None on parse failure."""
    if not isinstance(value_text, str):
        return None
    try:
        head = value_text.strip().split(" ")[0]
        return float(head.replace(",", "."))
    except (ValueError, IndexError):
        return None


def _walk_lhm_tree(
    node: Any,
    path: str,
    collector: Callable[[str, str, str], None],
) -> None:
    """DFS across LHM's nested dict tree; calls collector(path, name, value)
    on every leaf that carries a Value string."""
    if not isinstance(node, dict):
        return
    name = node.get("Text", "") or ""
    full_path = f"{path}/{name}"
    value_text = node.get("Value", "")
    if isinstance(value_text, str) and value_text.strip():
        collector(full_path, name, value_text)
    for child in node.get("Children", []) or []:
        _walk_lhm_tree(child, full_path, collector)


def _extract_sample_from_tree(data: Any, now_mono: float, now_unix: float) -> LhmSample:
    """Build an LhmSample from a parsed LHM JSON tree."""
    gpu_temp: float | None = None
    cpu_temp: float | None = None
    vram_used: float | None = None
    vram_total: float | None = None
    ram_used: float | None = None
    ram_total: float | None = None

    def _collect(path: str, _name: str, value_text: str) -> None:
        nonlocal gpu_temp, cpu_temp, vram_used, vram_total, ram_used, ram_total
        is_temp = value_text.endswith(" C") or value_text.endswith(" \u00b0C")
        is_gb = value_text.endswith(" GB")
        is_mb = value_text.endswith(" MB")
        num = _parse_value_number(value_text)
        if num is None:
            return

        # GPU temperature (hottest sensor wins).
        if is_temp and "GPU" in path:
            if gpu_temp is None or num > gpu_temp:
                gpu_temp = num
        # CPU package temperature (hottest sensor wins).
        if is_temp and "CPU" in path and "GPU" not in path:
            if cpu_temp is None or num > cpu_temp:
                cpu_temp = num

        # GPU memory (dedicated VRAM).
        if (is_gb or is_mb) and "GPU" in path:
            gb = num if is_gb else (num / 1024.0)
            low = path.lower()
            if "memory used" in low or "memory used" in _name.lower() or "used" in low:
                if "dedicated" in low or "vram" in low or "memory used" in low:
                    if vram_used is None or gb > vram_used:
                        vram_used = gb
            if ("total" in low or "available" in low) and (
                "dedicated" in low or "vram" in low or "memory" in low
            ):
                if vram_total is None or gb > vram_total:
                    vram_total = gb

        # System RAM.
        if (is_gb or is_mb) and "GPU" not in path:
            gb = num if is_gb else (num / 1024.0)
            low = path.lower()
            if ("memory" in low) and ("used" in low or "memory" == _name.lower()):
                if ram_used is None or gb > ram_used:
                    ram_used = gb
            if ("memory" in low) and ("available" in low or "total" in low):
                if ram_total is None or gb > ram_total:
                    ram_total = gb

    _walk_lhm_tree(data, "", _collect)
    return LhmSample(
        t_monotonic=now_mono,
        t_unix=now_unix,
        gpu_temp_c=gpu_temp,
        gpu_vram_used_gb=vram_used,
        gpu_vram_total_gb=vram_total,
        ram_used_gb=ram_used,
        ram_total_gb=ram_total,
        cpu_temp_c=cpu_temp,
    )


def poll_once(
    *,
    url: str = _LHM_URL,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    fetcher: Callable[[str, float], bytes] | None = None,
    now_mono: float | None = None,
    now_unix: float | None = None,
) -> LhmSample:
    """Perform one LHM poll and return an :class:`LhmSample`.

    ``fetcher`` is injectable for tests -- when given, it's called with
    ``(url, timeout_s)`` and must return the raw response bytes.  When
    absent, uses urllib.  Network / parse failures produce a sample
    with ``unreachable=True`` and a reason string so the summariser can
    count them without raising.
    """
    tm = now_mono if now_mono is not None else time.monotonic()
    tu = now_unix if now_unix is not None else time.time()
    try:
        if fetcher is not None:
            raw = fetcher(url, timeout_s)
        else:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                raw = resp.read()
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return LhmSample(
            t_monotonic=tm, t_unix=tu,
            unreachable=True, reason=f"network:{type(exc).__name__}",
        )
    except ValueError as exc:
        return LhmSample(
            t_monotonic=tm, t_unix=tu,
            unreachable=True, reason=f"parse:{type(exc).__name__}",
        )
    except Exception as exc:  # noqa: BLE001 -- never let telemetry kill a run
        return LhmSample(
            t_monotonic=tm, t_unix=tu,
            unreachable=True, reason=f"error:{type(exc).__name__}",
        )

    return _extract_sample_from_tree(data, tm, tu)


# ---- Poll loop + NDJSON writer --------------------------------------------


def append_ndjson(path: Path, sample: LhmSample) -> None:
    """Append one sample as a JSON line.  Creates the parent dir."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(sample.to_dict()) + "\n")


def poll_loop(
    out_path: Path,
    *,
    interval_s: float = _DEFAULT_INTERVAL_S,
    duration_s: float | None = None,
    max_samples: int | None = None,
    stop_when: Callable[[], bool] | None = None,
    fetcher: Callable[[str, float], bytes] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
    unix_fn: Callable[[], float] = time.time,
) -> list[LhmSample]:
    """Poll LHM every ``interval_s`` seconds, appending each sample to
    ``out_path`` (NDJSON).  Returns the full sample list.

    Exit conditions -- the loop stops when ANY of:
      - ``duration_s`` elapsed (default: unbounded -> runs until killed)
      - ``max_samples`` reached
      - ``stop_when()`` returns True (checked before each poll)

    ``sleep_fn`` / ``monotonic_fn`` / ``unix_fn`` are injectable for
    deterministic tests.  ``fetcher`` injects into :func:`poll_once`.
    """
    if interval_s <= 0:
        raise ValueError(f"interval_s must be > 0, got {interval_s}")
    out_path = Path(out_path)
    samples: list[LhmSample] = []
    t0 = monotonic_fn()

    def _time_up() -> bool:
        return duration_s is not None and (monotonic_fn() - t0) >= duration_s

    while True:
        if stop_when is not None and stop_when():
            break
        if _time_up():
            break
        if max_samples is not None and len(samples) >= max_samples:
            break

        s = poll_once(
            fetcher=fetcher,
            now_mono=monotonic_fn(),
            now_unix=unix_fn(),
        )
        samples.append(s)
        append_ndjson(out_path, s)

        # Sleep only if we'll poll again -- avoids a trailing sleep.
        # stop_when is intentionally NOT re-checked here; we keep a single
        # stop_when invocation per iteration at the top of the loop so
        # tests can count checks deterministically (N polls completed
        # means stop_when returned True on the (N+1)th call).
        if max_samples is not None and len(samples) >= max_samples:
            break
        if _time_up():
            break
        sleep_fn(interval_s)

    return samples


# ---- Summariser ------------------------------------------------------------


@dataclass
class LhmSummary:
    n_samples: int = 0
    n_unreachable: int = 0
    duration_s: float = 0.0
    gpu_temp_c: dict[str, float] = field(default_factory=dict)  # peak/mean/min
    gpu_vram_used_gb: dict[str, float] = field(default_factory=dict)
    ram_used_gb: dict[str, float] = field(default_factory=dict)
    cpu_temp_c: dict[str, float] = field(default_factory=dict)
    vram_ceiling_breached: bool = False
    ram_ceiling_breached: bool = False
    gpu_temp_ceiling_breached: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_unreachable": self.n_unreachable,
            "duration_s": round(self.duration_s, 3),
            "gpu_temp_c": dict(self.gpu_temp_c),
            "gpu_vram_used_gb": dict(self.gpu_vram_used_gb),
            "ram_used_gb": dict(self.ram_used_gb),
            "cpu_temp_c": dict(self.cpu_temp_c),
            "vram_ceiling_breached": self.vram_ceiling_breached,
            "ram_ceiling_breached": self.ram_ceiling_breached,
            "gpu_temp_ceiling_breached": self.gpu_temp_ceiling_breached,
            "vram_ceiling_gb": VRAM_CEILING_GB,
            "ram_ceiling_gb": RAM_CEILING_GB,
            "gpu_temp_ceiling_c": GPU_TEMP_CEILING_C,
            "notes": list(self.notes),
        }


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "peak": max(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "last": values[-1],
    }


def summarize(samples: Iterable[LhmSample]) -> LhmSummary:
    """Roll up an iterable of :class:`LhmSample` into a :class:`LhmSummary`
    with peak / mean / min / last per metric and the three Day 13
    ceiling-breach flags."""
    samples_list = list(samples)
    summary = LhmSummary(n_samples=len(samples_list))
    if not samples_list:
        summary.notes.append("no samples collected")
        return summary

    t_first = samples_list[0].t_monotonic
    t_last = samples_list[-1].t_monotonic
    summary.duration_s = max(0.0, t_last - t_first)
    summary.n_unreachable = sum(1 for s in samples_list if s.unreachable)

    gpu_temps = [s.gpu_temp_c for s in samples_list if s.gpu_temp_c is not None]
    vrams = [s.gpu_vram_used_gb for s in samples_list if s.gpu_vram_used_gb is not None]
    rams = [s.ram_used_gb for s in samples_list if s.ram_used_gb is not None]
    cpus = [s.cpu_temp_c for s in samples_list if s.cpu_temp_c is not None]

    summary.gpu_temp_c = _stats(gpu_temps)
    summary.gpu_vram_used_gb = _stats(vrams)
    summary.ram_used_gb = _stats(rams)
    summary.cpu_temp_c = _stats(cpus)

    if vrams and max(vrams) > VRAM_CEILING_GB:
        summary.vram_ceiling_breached = True
        summary.notes.append(
            f"VRAM ceiling {VRAM_CEILING_GB} GB breached "
            f"(peak {max(vrams):.2f} GB)"
        )
    if rams and max(rams) > RAM_CEILING_GB:
        summary.ram_ceiling_breached = True
        summary.notes.append(
            f"RAM ceiling {RAM_CEILING_GB} GB breached "
            f"(peak {max(rams):.2f} GB)"
        )
    if gpu_temps and max(gpu_temps) > GPU_TEMP_CEILING_C:
        summary.gpu_temp_ceiling_breached = True
        summary.notes.append(
            f"GPU temp ceiling {GPU_TEMP_CEILING_C}C breached "
            f"(peak {max(gpu_temps):.2f}C)"
        )

    return summary


def summarize_ndjson(path: Path) -> LhmSummary:
    """Load an NDJSON log produced by :func:`poll_loop` and summarise it."""
    path = Path(path)
    samples: list[LhmSample] = []
    if not path.exists():
        summary = LhmSummary()
        summary.notes.append(f"log file missing: {path}")
        return summary
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        samples.append(LhmSample(
            t_monotonic=float(d.get("t_monotonic", 0.0)),
            t_unix=float(d.get("t_unix", 0.0)),
            gpu_temp_c=d.get("gpu_temp_c"),
            gpu_vram_used_gb=d.get("gpu_vram_used_gb"),
            gpu_vram_total_gb=d.get("gpu_vram_total_gb"),
            ram_used_gb=d.get("ram_used_gb"),
            ram_total_gb=d.get("ram_total_gb"),
            cpu_temp_c=d.get("cpu_temp_c"),
            unreachable=bool(d.get("unreachable", False)),
            reason=str(d.get("reason") or ""),
        ))
    return summarize(samples)


__all__ = [
    "VRAM_CEILING_GB",
    "RAM_CEILING_GB",
    "GPU_TEMP_CEILING_C",
    "LhmSample",
    "LhmSummary",
    "poll_once",
    "poll_loop",
    "append_ndjson",
    "summarize",
    "summarize_ndjson",
]
