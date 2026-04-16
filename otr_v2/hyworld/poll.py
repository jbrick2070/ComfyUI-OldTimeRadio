"""
poll.py  --  OTR_HyworldPoll ComfyUI node
==========================================
Polls io/hyworld_out/<job_id>/STATUS.json for sidecar completion.
Blocks (with ComfyUI spinner) until ready, error, or timeout.

Design doc: docs/superpowers/specs/2026-04-15-hyworld-poc-design.md  Section 6
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

log = logging.getLogger("OTR.hyworld.poll")

_OTR_ROOT = Path(__file__).resolve().parent.parent.parent
_IO_OUT = _OTR_ROOT / "io" / "hyworld_out"

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
}

# Statuses that mean "we have usable assets"
_SUCCESS_STATUSES = {"READY"}

# Statuses that should gracefully fall back (not crash the workflow)
_FALLBACK_STATUSES = _TERMINAL_STATUSES - _SUCCESS_STATUSES


class HyworldPoll:
    """
    ComfyUI node: OTR_HyworldPoll

    Polls STATUS.json until the sidecar finishes.  Returns the path
    to the output assets so OTR_HyworldRenderer can consume them.

    If the sidecar is unavailable or errored, returns a fallback
    path string that downstream nodes can check to route to
    OTR_SignalLostVideo instead.
    """

    CATEGORY = "OTR/v2/HyWorld"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("hyworld_assets_path", "status", "status_detail")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hyworld_job_id": ("STRING", {
                    "tooltip": "Job ID from OTR_HyworldBridge.",
                }),
            },
            "optional": {
                "timeout_sec": ("INT", {
                    "default": 600,
                    "min": 10,
                    "max": 3600,
                    "tooltip": "Maximum seconds to wait for sidecar completion.",
                }),
                "poll_interval_sec": ("FLOAT", {
                    "default": 2.0,
                    "min": 0.5,
                    "max": 30.0,
                    "tooltip": "Seconds between STATUS.json checks.",
                }),
            },
        }

    def execute(
        self,
        hyworld_job_id: str,
        timeout_sec: int = 600,
        poll_interval_sec: float = 2.0,
    ) -> tuple[str, str, str]:
        """
        Poll io/hyworld_out/<job_id>/STATUS.json until terminal.
        Returns (assets_path, status, status_detail).
        """
        # Handle error-prefixed job IDs from bridge parse failures
        if hyworld_job_id.startswith("PARSE_ERROR_"):
            return ("FALLBACK", "PARSE_ERROR", "Bridge received unparseable script_json")

        out_dir = _IO_OUT / hyworld_job_id
        status_file = out_dir / "STATUS.json"

        log.info("[HyworldPoll] Polling job %s (timeout=%ds)", hyworld_job_id, timeout_sec)

        start = time.monotonic()
        last_status = "WAITING"
        detail = ""

        while (time.monotonic() - start) < timeout_sec:
            if status_file.exists():
                try:
                    data = json.loads(status_file.read_text(encoding="utf-8"))
                    last_status = data.get("status", "UNKNOWN")
                    detail = data.get("detail", "")

                    if last_status in _TERMINAL_STATUSES:
                        log.info("[HyworldPoll] Job %s terminal: %s", hyworld_job_id, last_status)
                        break
                except (json.JSONDecodeError, OSError) as e:
                    # File may be mid-write; retry next cycle
                    log.debug("[HyworldPoll] STATUS.json read error (retrying): %s", e)

            time.sleep(poll_interval_sec)

        else:
            # Timeout reached
            last_status = "POLL_TIMEOUT"
            detail = f"No terminal status after {timeout_sec}s"
            log.warning("[HyworldPoll] Job %s timed out after %ds", hyworld_job_id, timeout_sec)

        # Determine assets path
        if last_status in _SUCCESS_STATUSES:
            assets_path = str(out_dir)
        else:
            assets_path = "FALLBACK"

        return (assets_path, last_status, detail)
