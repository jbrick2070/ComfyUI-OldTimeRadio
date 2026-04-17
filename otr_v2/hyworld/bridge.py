"""
bridge.py  --  OTR_HyworldBridge ComfyUI node
===============================================
Writes Director plan + scene manifest to io/hyworld_in/,
generates shotlist via deterministic rules, and spawns the
sidecar worker process.

Design doc: docs/2026-04-15-hyworld-poc-design.md  Section 6
Mapping doc: docs/2026-04-15-otr-to-hyworld-narrative-mapping.md

Audio path is NEVER touched.  If this node fails, downstream falls back
to OTR_SignalLostVideo (procedural).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

log = logging.getLogger("OTR.hyworld.bridge")

# Phase A: atomic writes for the contract files so the worker (which
# reads them at startup) never lands on a half-written JSON.
try:
    from ._atomic import atomic_write_json, atomic_write_text
except ImportError:
    # Fallback when bridge.py is imported outside its package context
    # (e.g. during certain test setups).
    from _atomic import atomic_write_json, atomic_write_text  # type: ignore


# ---------------------------------------------------------------------------
# Paths relative to OTR repo root
# ---------------------------------------------------------------------------
_OTR_ROOT = Path(__file__).resolve().parent.parent.parent
_IO_IN = _OTR_ROOT / "io" / "hyworld_in"
_IO_OUT = _OTR_ROOT / "io" / "hyworld_out"

# Canonical Audio Token types accepted in script_json.  Extra types are
# allowed (forward-compat) but at least one of these MUST appear or we
# refuse the run -- a script with zero recognized tokens is almost
# always a malformed upstream payload.
_CANONICAL_TOKEN_TYPES = {
    "title", "scene_break", "environment", "sfx",
    "pause", "dialogue", "direction", "music",
}

# Sidecar worker script (lives alongside this file)
_WORKER_SCRIPT = Path(__file__).resolve().parent / "worker.py"

# Sidecar conda env name (from PoC design doc Section 8B)
_SIDECAR_ENV = "hyworld2"

# Timeout before declaring sidecar spawn failure
_SPAWN_TIMEOUT_S = 30


def _ensure_io_dirs() -> None:
    """Create io/ exchange directories if they don't exist."""
    _IO_IN.mkdir(parents=True, exist_ok=True)
    _IO_OUT.mkdir(parents=True, exist_ok=True)


def _validate_script_lines(script_lines: list) -> tuple[bool, str]:
    """Light schema check on the parsed script_json payload.

    The full Canonical Audio Token spec is enforced upstream by the
    audio pipeline; here we only catch obvious malformations that
    would crash the worker after spawn:

    - top level must be a non-empty list
    - every entry must be a dict
    - every entry must have a string ``type`` field
    - at least one entry's ``type`` must be in the canonical set

    Returns (ok, reason).  reason is empty when ok=True.
    """
    if not isinstance(script_lines, list):
        return False, "top-level must be a JSON array"
    if not script_lines:
        return False, "script_lines is empty"
    canonical_seen = False
    for i, item in enumerate(script_lines):
        if not isinstance(item, dict):
            return False, f"line {i} is not an object"
        ttype = item.get("type")
        if not isinstance(ttype, str):
            return False, f"line {i} missing string 'type' field"
        if ttype in _CANONICAL_TOKEN_TYPES:
            canonical_seen = True
    if not canonical_seen:
        return False, (
            f"no canonical token types found "
            f"(expected at least one of {sorted(_CANONICAL_TOKEN_TYPES)})"
        )
    return True, ""


class HyworldBridge:
    """
    ComfyUI node: OTR_HyworldBridge

    Takes Director output + script_json from the v1.7 audio pipeline,
    generates a deterministic shotlist, writes the sidecar contract
    files, and spawns the HyWorld worker subprocess.

    If the sidecar env is not installed or the spawn fails, returns
    a fallback job_id with status "SIDECAR_UNAVAILABLE" so downstream
    nodes can gracefully fall back to OTR_SignalLostVideo.
    """

    CATEGORY = "OTR/v2/HyWorld"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("hyworld_job_id", "shotlist_json")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_json": ("STRING", {
                    "multiline": True,
                    "tooltip": "Canonical Audio Token array from OTR_Gemma4ScriptWriter (JSON string).",
                }),
                "episode_title": ("STRING", {
                    "default": "Untitled Episode",
                    "tooltip": "Episode title for style anchor hash.",
                }),
            },
            "optional": {
                "production_plan_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": "Director production plan (optional, enriches shotlist metadata).",
                }),
                "scene_manifest_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": "Scene manifest from OTR_SceneSequencer (optional, provides audio offsets).",
                }),
                "lane": (["faithful", "translated", "chaotic"], {
                    "default": "faithful",
                    "tooltip": "Visual lane: faithful (deterministic), translated (LLM-shaped), chaotic (avant-garde).",
                }),
                "chaos_ops": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated chaos operators for Lane 3 (e.g. 'swap,shuffle'). Ignored unless lane=chaotic.",
                }),
                "chaos_seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2**31 - 1,
                    "tooltip": "Seed for Lane 3 chaos operators. Same seed = same visual output.",
                }),
                "sidecar_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If False, generate shotlist only (no sidecar spawn). Useful for dry runs.",
                }),
            },
        }

    def execute(
        self,
        script_json: str,
        episode_title: str,
        production_plan_json: str = "{}",
        scene_manifest_json: str = "{}",
        lane: str = "faithful",
        chaos_ops: str = "",
        chaos_seed: int = 42,
        sidecar_enabled: bool = True,
    ) -> tuple[str, str]:
        """
        1. Parse script_json into script_lines.
        2. Generate deterministic shotlist via shotlist.py.
        3. Write contract files to io/hyworld_in/<job_id>/.
        4. Optionally spawn sidecar worker.
        5. Return (job_id, shotlist_json).
        """
        # Late import to avoid circular deps at ComfyUI scan time
        from .shotlist import generate_shotlist

        job_id = f"hw_{uuid.uuid4().hex[:12]}"
        log.info("[HyworldBridge] Job %s starting (lane=%s)", job_id, lane)

        # ---- 1. Parse + validate script_lines ----
        try:
            script_lines = json.loads(script_json)
        except json.JSONDecodeError as e:
            log.error("[HyworldBridge] script_json parse failed: %s", e)
            return (f"PARSE_ERROR_{job_id}", "[]")

        ok, reason = _validate_script_lines(script_lines)
        if not ok:
            log.error("[HyworldBridge] script_json schema check failed: %s", reason)
            return (f"PARSE_ERROR_{job_id}", "[]")

        # ---- 2. Generate shotlist (Lane 1 floor, always runs) ----
        shotlist_result = generate_shotlist(script_lines, episode_title)
        shotlist_json = json.dumps(shotlist_result, indent=2)
        log.info(
            "[HyworldBridge] Shotlist: %d scenes, %d shots, anchor=%s",
            shotlist_result["scene_count"],
            shotlist_result["total_shots"],
            shotlist_result["style_anchor_hash"],
        )

        # ---- 3. Write contract files ----
        _ensure_io_dirs()
        job_dir = _IO_IN / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        # Atomic writes prevent the worker (which reads these immediately
        # after spawn) from landing on a half-written contract file.
        atomic_write_text(job_dir / "script_lines.json", script_json)
        atomic_write_text(job_dir / "shotlist.json", shotlist_json)
        atomic_write_text(job_dir / "production_plan.json", production_plan_json)
        atomic_write_text(job_dir / "scene_manifest.json", scene_manifest_json)

        # Metadata for the sidecar
        meta = {
            "job_id": job_id,
            "episode_title": episode_title,
            "lane": lane,
            "chaos_ops": [op.strip() for op in chaos_ops.split(",") if op.strip()] if lane == "chaotic" else [],
            "chaos_seed": chaos_seed,
            "style_anchor_hash": shotlist_result["style_anchor_hash"],
            "scene_count": shotlist_result["scene_count"],
            "total_shots": shotlist_result["total_shots"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        atomic_write_json(job_dir / "meta.json", meta)

        log.info("[HyworldBridge] Contract written to %s", job_dir)

        # ---- 4. Spawn sidecar (optional) ----
        if sidecar_enabled:
            status = self._spawn_sidecar(job_id, job_dir)
            if status != "SPAWNED":
                log.warning("[HyworldBridge] Sidecar not available: %s. Shotlist still usable for fallback.", status)
                # Write status so poll node knows
                self._write_status(job_id, status)
        else:
            log.info("[HyworldBridge] Sidecar disabled (dry run). Shotlist written.")
            self._write_status(job_id, "DRY_RUN")

        return (job_id, shotlist_json)

    def _spawn_sidecar(self, job_id: str, job_dir: Path) -> str:
        """
        Attempt to spawn the HyWorld worker in the hyworld2 conda env.
        Falls back to the main ComfyUI Python for stub mode if conda
        env is unavailable.
        Returns: "SPAWNED", "SPAWN_FAILED".
        """
        # Priority 1: hyworld2 conda env (for real WorldMirror inference)
        home = Path.home()
        candidates = [
            home / "miniconda3" / "envs" / _SIDECAR_ENV / "python.exe",
            home / "anaconda3" / "envs" / _SIDECAR_ENV / "python.exe",
            home / "miniconda3" / "envs" / _SIDECAR_ENV / "bin" / "python",
            home / "anaconda3" / "envs" / _SIDECAR_ENV / "bin" / "python",
        ]
        # Priority 2: main ComfyUI venv Python (runs worker in stub mode)
        comfyui_venv = _OTR_ROOT.parent.parent / ".venv" / "Scripts" / "python.exe"
        if comfyui_venv.exists():
            candidates.append(comfyui_venv)
        # Priority 3: system python
        candidates.append(Path(sys.executable))

        sidecar_python = None
        for p in candidates:
            if p.exists():
                sidecar_python = p
                break

        if sidecar_python is None:
            log.warning("[HyworldBridge] No usable Python found for sidecar. Tried: %s", [str(c) for c in candidates])
            return "SPAWN_FAILED"

        log.info("[HyworldBridge] Sidecar Python: %s", sidecar_python)

        if not _WORKER_SCRIPT.exists():
            log.warning("[HyworldBridge] Worker script not found at %s", _WORKER_SCRIPT)
            return "WORKER_MISSING"

        proc = None
        try:
            proc = subprocess.Popen(
                [str(sidecar_python), str(_WORKER_SCRIPT), str(job_dir)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(_OTR_ROOT),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            log.info("[HyworldBridge] Sidecar spawned PID=%d for job %s", proc.pid, job_id)
            # Write PID for poll node to monitor / cleanup.
            # Atomic so poll never reads a partial PID string.
            atomic_write_text(job_dir / "sidecar_pid.txt", str(proc.pid))
            return "SPAWNED"
        except Exception as e:
            log.error("[HyworldBridge] Sidecar spawn failed: %s", e)
            return "SPAWN_FAILED"
        finally:
            # Sidecar is fire-and-forget; cleanup only on spawn failure.
            # On success the poll node monitors the PID via sidecar_pid.txt.
            if proc is not None and proc.poll() is None and proc.returncode is not None:
                proc.terminate()

    def _write_status(self, job_id: str, status: str) -> None:
        """Write a STATUS.json to io/hyworld_out/<job_id>/ for the poll node.

        Atomic so the poll node never reads a half-written status when
        the bridge is the one writing it (the worker also uses atomic
        writes via _atomic.py).
        """
        out_dir = _IO_OUT / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_dir / "STATUS.json", {
            "job_id": job_id,
            "status": status,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
