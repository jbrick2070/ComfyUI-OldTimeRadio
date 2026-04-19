"""
otr_v2.visual.backends._base  --  shared contract for all backends
===================================================================

Every video stack backend (FLUX, PuLID, LTX, Wan2.1, Florence-2, and
the Day 1 placeholder canary) inherits from ``Backend`` so the worker
can dispatch by name without branching on backend identity.

Also hosts two small helpers the worker used to hand-roll:

* ``write_status`` -- atomic STATUS.json write with the canonical
  status values ``RUNNING | READY | ERROR | OOM``.
* ``cooldown_gate`` -- pre-spawn LHM poll.  Returns quickly when the
  GPU is cool; logs and proceeds (never blocks) when LHM is down or
  the threshold holds after the max wait.  Protects the audio rails
  from a sidecar that wakes up while the card is still hot from Bark.

No torch / diffusers / GPU imports here.  Keeps unit tests fast.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Package-local atomic writer, with the same stripped-env fallback the
# worker uses when launched without full package context.
_THIS_DIR = Path(__file__).resolve().parent
_VISUAL_DIR = _THIS_DIR.parent
if str(_VISUAL_DIR) not in sys.path:
    sys.path.insert(0, str(_VISUAL_DIR))
try:
    from _atomic import atomic_write_json  # type: ignore
except ImportError:  # pragma: no cover -- only hit in bare-env runs
    def atomic_write_json(path: Path, data, indent: int = 2) -> None:  # type: ignore
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(data, indent=indent, ensure_ascii=False),
            encoding="utf-8",
        )


# Canonical status values.  Any backend that writes a status outside
# this set breaks the poll node's state machine.
STATUS_RUNNING = "RUNNING"
STATUS_READY = "READY"
STATUS_ERROR = "ERROR"
STATUS_OOM = "OOM"
VALID_STATUSES = frozenset({STATUS_RUNNING, STATUS_READY, STATUS_ERROR, STATUS_OOM})


# LibreHardwareMonitor default endpoint.  Jeffrey keeps it always-on;
# if it's not reachable we warn and proceed rather than block.
_LHM_URL = os.environ.get("OTR_LHM_URL", "http://localhost:8085/data.json")


def write_status(out_dir: Path, status: str, detail: str = "", **extra: Any) -> None:
    """Atomic STATUS.json write.  Worker and backends share this writer."""
    if status not in VALID_STATUSES:
        raise ValueError(
            f"status {status!r} not in {sorted(VALID_STATUSES)}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "detail": detail,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    payload.update(extra)
    atomic_write_json(out_dir / "STATUS.json", payload)


def _read_lhm_gpu_temp() -> float | None:
    """Poll LibreHardwareMonitor for the highest GPU temperature sensor.

    Returns ``None`` when LHM is unreachable or the schema doesn't
    match (ship-it-and-warn rather than ship-it-and-die).
    """
    try:
        with urllib.request.urlopen(_LHM_URL, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None

    # LHM JSON is a deeply nested tree.  Walk it looking for any node
    # whose Text ends in " C" and whose path includes "GPU" -- safe
    # enough across driver revisions without building a full schema.
    hottest: float | None = None

    def _walk(node, path):
        nonlocal hottest
        if not isinstance(node, dict):
            return
        name = node.get("Text", "")
        full_path = f"{path}/{name}"
        value_text = node.get("Value", "")
        if isinstance(value_text, str) and value_text.endswith(" C") and "GPU" in full_path:
            try:
                temp = float(value_text[:-2].strip().replace(",", "."))
                if hottest is None or temp > hottest:
                    hottest = temp
            except ValueError:
                pass
        for child in node.get("Children", []) or []:
            _walk(child, full_path)

    _walk(data, "")
    return hottest


def cooldown_gate(
    max_wait_s: float = 20.0,
    temp_threshold_c: float = 82.0,
    poll_interval_s: float = 2.0,
) -> tuple[bool, str]:
    """Block briefly if the GPU is hot, then proceed regardless.

    Returns ``(True, reason)`` when the card is cool enough or LHM
    was unreachable (we refuse to block on a missing telemetry source).
    Returns ``(False, reason)`` only when the threshold still holds
    after ``max_wait_s`` -- the caller can log this but should still
    proceed; the sidecar is the thing with the actual OOM guard.
    """
    deadline = time.monotonic() + max(0.0, max_wait_s)
    last_temp: float | None = None
    while True:
        temp = _read_lhm_gpu_temp()
        last_temp = temp
        if temp is None:
            return (True, "lhm_unreachable")
        if temp < temp_threshold_c:
            return (True, f"cool:{temp:.1f}C<{temp_threshold_c:.1f}C")
        if time.monotonic() >= deadline:
            return (False, f"hot_after_wait:{temp:.1f}C>={temp_threshold_c:.1f}C")
        time.sleep(poll_interval_s)
    # unreachable  # pragma: no cover


class Backend:
    """Abstract base for all video stack backends.

    Subclasses implement ``run(job_dir)``.  The job dir is
    ``io/visual_in/<job_id>/``; output goes to
    ``io/visual_out/<job_id>/``.  The path convention is fixed by
    the VisualBridge contract and must not change.
    """

    name: str = "base"

    def out_dir_for(self, job_dir: Path) -> Path:
        """Resolve the matching ``io/visual_out/<job_id>/`` directory."""
        job_id = job_dir.name
        otr_root = job_dir.parent.parent.parent
        return otr_root / "io" / "visual_out" / job_id

    def load_shotlist(self, job_dir: Path) -> list[dict]:
        """Read shotlist.json from the job dir.  Returns the shots list."""
        shotlist_path = job_dir / "shotlist.json"
        if not shotlist_path.exists():
            raise FileNotFoundError(f"shotlist.json not found in {job_dir}")
        payload = json.loads(shotlist_path.read_text(encoding="utf-8"))
        shots = payload.get("shots", [])
        if not isinstance(shots, list):
            raise ValueError("shotlist.json 'shots' field is not a list")
        return shots

    def run(self, job_dir: Path) -> None:  # pragma: no cover -- abstract
        raise NotImplementedError
