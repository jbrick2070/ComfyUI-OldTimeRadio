"""
poll.py  --  OTR_VisualPoll ComfyUI node
==========================================
Polls io/visual_out/<job_id>/STATUS.json for sidecar completion.
Blocks (with ComfyUI spinner) until ready, error, or timeout.

Design doc: docs/2026-04-15-visual-poc-design.md  Section 6

Phase A: also tracks the sidecar PID via io/visual_in/<job_id>/sidecar_pid.txt
(written by bridge._spawn_sidecar).  If the PID is no longer alive AND no
terminal status has been written, we surface ``WORKER_DEAD`` instead of
hanging until poll-timeout.  This handles the case where the worker crashes
or ComfyUI is hard-killed mid-run.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

log = logging.getLogger("OTR.visual.poll")

_OTR_ROOT = Path(__file__).resolve().parent.parent.parent
_IO_OUT = _OTR_ROOT / "io" / "visual_out"
_IO_IN = _OTR_ROOT / "io" / "visual_in"

# Terminal statuses that stop polling
_TERMINAL_STATUSES = {
    "READY",           # sidecar finished successfully
    "ERROR",           # sidecar hit a fatal error
    "OOM",             # out of memory
    "TIMEOUT",         # worker exceeded its own wall-clock limit
    "ENV_NOT_FOUND",   # bridge couldn't find conda env
    "WORKER_MISSING",  # worker.py not present
    "SPAWN_FAILED",    # sidecar process launch failed
    "DRY_RUN",         # sidecar_enabled was False
    "SIDECAR_UNAVAILABLE",
    "WORKER_DEAD",     # Phase A: sidecar PID no longer alive, no terminal status
}

# Statuses that mean "we have usable assets"
_SUCCESS_STATUSES = {"READY"}

# Statuses that should gracefully fall back (not crash the workflow)
_FALLBACK_STATUSES = _TERMINAL_STATUSES - _SUCCESS_STATUSES

# How often to check the sidecar PID (every Nth poll cycle).  Cheap on
# POSIX but a Windows OpenProcess call costs ~1ms, so we don't do it
# every iteration of a 0.5s poll loop.
_PID_CHECK_EVERY_N_CYCLES = 5

# Tolerance window: even after the PID dies, give it this many seconds
# for any final STATUS.json write to land before declaring WORKER_DEAD.
_PID_DEATH_GRACE_SEC = 3.0


def _read_sidecar_pid(job_id: str) -> int:
    """Read the sidecar's PID file written by bridge._spawn_sidecar.

    Returns 0 if the file is missing or unreadable -- callers treat
    that as "PID-tracking unavailable, fall back to status-only polling".
    """
    pid_file = _IO_IN / job_id / "sidecar_pid.txt"
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return 0


def _pid_alive(pid: int) -> bool:
    """Cross-platform PID liveness check.  Mirrors vram_coordinator._pid_alive
    but kept independent so poll.py has zero deps on the coordinator module
    (poll runs in the main ComfyUI process and we want it bulletproof)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if not handle:
                return False
            exit_code = ctypes.c_ulong(0)
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            return bool(ok) and exit_code.value == STILL_ACTIVE
        except Exception:
            return True  # err on the side of "alive" so we don't kill a real worker
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


class VisualPoll:
    """
    ComfyUI node: OTR_VisualPoll

    Polls STATUS.json until the sidecar finishes.  Returns the path
    to the output assets so OTR_VisualRenderer can consume them.

    If the sidecar is unavailable or errored, returns a fallback
    path string that downstream nodes can check to route to
    OTR_SignalLostVideo instead.
    """

    CATEGORY = "OTR/v2/Visual"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("visual_assets_path", "status", "status_detail")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "visual_job_id": ("STRING", {
                    "tooltip": "Job ID from OTR_VisualBridge.",
                }),
            },
            "optional": {},
        }

    def execute(self, visual_job_id: str):
        """Poll until sidecar finishes or timeout.  Return assets path + status."""
        # Bridge returns "PARSE_ERROR_<job_id>" when script_json is malformed.
        # Short-circuit so we don't poll a job dir that will never exist.
        if visual_job_id.startswith("PARSE_ERROR_"):
            return ("FALLBACK", "PARSE_ERROR", "Bridge received unparseable script_json")

        job_dir_out = _IO_OUT / visual_job_id
        job_dir_in = _IO_IN / visual_job_id
        status_file = job_dir_out / "STATUS.json"
        
        sidecar_pid = _read_sidecar_pid(visual_job_id)
        poll_count = 0
        grace_started = None
        
        while True:
            poll_count += 1
            
            # Check PID liveness every N cycles
            if sidecar_pid > 0 and poll_count % _PID_CHECK_EVERY_N_CYCLES == 0:
                if not _pid_alive(sidecar_pid):
                    grace_started = time.time()
            
            # Read STATUS if it exists
            status_data = None
            if status_file.exists():
                try:
                    status_data = json.loads(status_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass
            
            if status_data and status_data.get("status") in _TERMINAL_STATUSES:
                return (
                    str(job_dir_out),
                    status_data.get("status", "UNKNOWN"),
                    status_data.get("detail", ""),
                )
            
            # Check grace period after PID death
            if grace_started is not None:
                if time.time() - grace_started > _PID_DEATH_GRACE_SEC:
                    return (
                        str(job_dir_out),
                        "WORKER_DEAD",
                        f"Sidecar PID {sidecar_pid} died without terminal status",
                    )
            
            # Timeout fallback (10 minute hard limit)
            if poll_count > 1200:  # 0.5s * 1200 = 600s = 10 min
                return (
                    str(job_dir_out),
                    "TIMEOUT",
                    "Poll timeout exceeded",
                )
            
            time.sleep(0.5)
