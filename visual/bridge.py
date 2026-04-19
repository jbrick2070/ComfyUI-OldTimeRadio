"""
bridge.py  --  OTR_VisualBridge ComfyUI node
===============================================
Writes Director plan + scene manifest to io/visual_in/,
generates shotlist via deterministic rules, and spawns the
sidecar worker process.

Design doc: docs/2026-04-15-visual-poc-design.md  Section 6
Mapping doc: docs/2026-04-15-otr-to-visual-narrative-mapping.md

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

log = logging.getLogger("OTR.visual.bridge")

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
_IO_IN = _OTR_ROOT / "io" / "visual_in"
_IO_OUT = _OTR_ROOT / "io" / "visual_out"

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
_SIDECAR_ENV = "visual2"

# Timeout before declaring sidecar spawn failure
_SPAWN_TIMEOUT_S = 30

# Explicit backend choices surfaced in the node UI.  "auto" preserves
# the legacy worldmirror-or-stub path; named entries dispatch into
# ``visual/backends/`` via the OTR_VISUAL_BACKEND env var on
# the spawned sidecar.  Day 1 sprint exposes only placeholder_test; the
# rest register as Days 2-7 of the 14-day video stack sprint land.
_BACKEND_CHOICES = [
    "auto",
    "video_stack",
    "placeholder_test",
    "flux_anchor",
    "flux_controlnet_keyframe",
    "pulid_portrait",
    "ltx_motion",
    "wan_i2v",
    "florence2_mask",
    "sdxl_inpaint",
]

# Pre-spawn GPU cooldown parameters.  Overridable via env var so the
# audio-focused test harness can force a tight gate.  Defaults chosen
# to avoid blocking normal generation: 82C threshold matches the RTX
# 5080 Laptop's documented thermal headroom, 20s ceiling guarantees we
# never stall a queue if the card is genuinely saturated.
_COOLDOWN_TEMP_C = float(os.environ.get("OTR_VISUAL_COOLDOWN_C", "82.0"))
_COOLDOWN_MAX_WAIT_S = float(os.environ.get("OTR_VISUAL_COOLDOWN_MAX_S", "20.0"))


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


class VisualBridge:
    """
    ComfyUI node: OTR_VisualBridge

    Takes Director output + script_json from the v1.7 audio pipeline,
    generates a deterministic shotlist, writes the sidecar contract
    files, and spawns the Visual worker subprocess.

    If the sidecar env is not installed or the spawn fails, returns
    a fallback job_id with status "SIDECAR_UNAVAILABLE" so downstream
    nodes can gracefully fall back to OTR_SignalLostVideo.
    """

    CATEGORY = "OTR/v2/Visual"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("visual_job_id", "shotlist_json")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_json": ("STRING", {
                    "multiline": True,
                    "tooltip": "Canonical Audio Token array from OTR_LLMScriptWriter (JSON string).",
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
                "backend": (_BACKEND_CHOICES, {
                    "default": "auto",
                    "tooltip": (
                        "Explicit video-stack backend to run in the sidecar. "
                        "'auto' preserves legacy worldmirror-or-stub behaviour. "
                        "Other choices dispatch into visual/backends/ "
                        "via the OTR_VISUAL_BACKEND env var."
                    ),
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
        backend: str = "auto",
    ) -> tuple[str, str]:
        """
        1. Parse script_json into script_lines.
        2. Generate deterministic shotlist via shotlist.py.
        3. Write contract files to io/visual_in/<job_id>/.
        4. Optionally spawn sidecar worker.
        5. Return (job_id, shotlist_json).
        """
        # Late import to avoid circular deps at ComfyUI scan time
        from .shotlist import generate_shotlist

        job_id = f"vs_{uuid.uuid4().hex[:12]}"
        log.info("[VisualBridge] Job %s starting (lane=%s)", job_id, lane)

        # ---- 1. Parse + validate script_lines ----
        try:
            script_lines = json.loads(script_json)
        except json.JSONDecodeError as e:
            log.error("[VisualBridge] script_json parse failed: %s", e)
            return (f"PARSE_ERROR_{job_id}", "[]")

        ok, reason = _validate_script_lines(script_lines)
        if not ok:
            log.error("[VisualBridge] script_json schema check failed: %s", reason)
            return (f"PARSE_ERROR_{job_id}", "[]")

        # ---- 2. Generate shotlist (Lane 1 floor, always runs) ----
        shotlist_result = generate_shotlist(script_lines, episode_title)
        shotlist_json = json.dumps(shotlist_result, indent=2)
        log.info(
            "[VisualBridge] Shotlist: %d scenes, %d shots, anchor=%s",
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

        log.info("[VisualBridge] Contract written to %s", job_dir)

        # ---- 4. Spawn sidecar (optional) ----
        if sidecar_enabled:
            status = self._spawn_sidecar(job_id, job_dir, backend=backend)
            if status != "SPAWNED":
                log.warning(
                    "[VisualBridge] Sidecar not available: %s. "
                    "Shotlist still usable for fallback.",
                    status,
                )
                self._write_status(job_id, status)
        else:
            log.info("[VisualBridge] Sidecar disabled (dry run). Shotlist written.")
            self._write_status(job_id, "DRY_RUN")

        return (job_id, shotlist_json)

    def _cooldown_gate(self, job_id: str) -> None:
        """Pre-spawn LHM poll; never blocks the queue indefinitely.

        If LibreHardwareMonitor is unreachable or the card is already
        cool, returns immediately.  If the GPU is hot, waits up to
        ``_COOLDOWN_MAX_WAIT_S`` then proceeds regardless -- the
        sidecar's own VRAM coordinator and OOM guard is the real
        safety net; this gate just smooths the Bark-to-video handoff.
        """
        try:
            from .backends._base import cooldown_gate
        except ImportError:
            try:
                from backends._base import cooldown_gate  # type: ignore
            except ImportError:
                log.debug("[VisualBridge] cooldown_gate unavailable; skipping")
                return
        try:
            ok, reason = cooldown_gate(
                max_wait_s=_COOLDOWN_MAX_WAIT_S,
                temp_threshold_c=_COOLDOWN_TEMP_C,
            )
        except Exception as exc:  # noqa: BLE001 -- never let telemetry kill a run
            log.debug("[VisualBridge] cooldown_gate errored (%s); proceeding", exc)
            return
        if ok:
            log.info("[VisualBridge] cooldown gate passed for %s (%s)", job_id, reason)
        else:
            log.warning(
                "[VisualBridge] cooldown gate timed out for %s (%s); proceeding anyway",
                job_id, reason,
            )

    def _pre_spawn_vram_flush(self, job_id: str) -> None:
        """Evict VRAM residues from the parent ComfyUI process before spawn.

        Addresses BUG-LOCAL-048: the sidecar subprocess inherits none of the
        parent's CUDA context (multiprocessing spawn start method), but the
        physical 16 GB VRAM is shared -- so LLM / Bark / Kokoro / anchor_gen
        residues held by ComfyUI leave the sidecar with almost no room to
        land real FLUX/LTX/Wan2.1 weights.  Observed 2026-04-19 on live
        run vs_3412f49920ef: 16,129 MB used / 173 MB free at spawn time,
        GPU pinned at 100% / 55W (PCIe thrashing, no compute progress).

        Called from _spawn_sidecar BEFORE the cooldown gate so the LHM
        readings the gate sees reflect the clean state, not the leftover.

        Safe no-op when CUDA is unavailable or when no callbacks are
        registered.  Errors are logged and swallowed -- never let a
        pre-spawn hygiene routine abort a real generation run.
        """
        # Snapshot current VRAM so the ceiling log has a before/after pair.
        try:
            from nodes._vram_log import (
                force_vram_offload,
                vram_snapshot,
                vram_reset_peak,
            )
        except ImportError:
            try:
                from _vram_log import (  # type: ignore
                    force_vram_offload,
                    vram_snapshot,
                    vram_reset_peak,
                )
            except ImportError:
                log.debug(
                    "[VisualBridge] _vram_log unavailable; skipping pre-spawn flush"
                )
                return

        try:
            before = vram_snapshot(f"pre_spawn_{job_id}_before")
            before_gb = before.get("current_gb", 0.0)
            log.info(
                "[VisualBridge] pre-spawn VRAM flush starting for %s (before=%.2f GB)",
                job_id, before_gb,
            )
            force_vram_offload()
            after = vram_snapshot(f"pre_spawn_{job_id}_after")
            after_gb = after.get("current_gb", 0.0)
            freed_gb = max(0.0, before_gb - after_gb)
            log.info(
                "[VisualBridge] pre-spawn VRAM flush complete for %s "
                "(after=%.2f GB, freed=%.2f GB)",
                job_id, after_gb, freed_gb,
            )
            # Fresh peak counter for the sidecar window so the episode
            # report attributes VRAM to the sidecar, not the main graph.
            vram_reset_peak(f"sidecar_{job_id}")
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[VisualBridge] pre-spawn VRAM flush errored for %s (%s); "
                "proceeding anyway -- subprocess still has its own OOM guard",
                job_id, exc,
            )

    def _spawn_sidecar(self, job_id: str, job_dir: Path, backend: str = "auto") -> str:
        """
        Attempt to spawn the Visual worker in the visual2 conda env.
        Falls back to the main ComfyUI Python for stub mode if conda
        env is unavailable.

        ``backend`` is plumbed through as the ``OTR_VISUAL_BACKEND``
        env var on the sidecar process.  ``"auto"`` leaves it unset so
        the worker falls through to the legacy worldmirror-or-stub
        branch; any other value dispatches via backends.resolve() in
        worker.main().

        Returns: ``"SPAWNED"``, ``"SPAWN_FAILED"``, or ``"WORKER_MISSING"``.
        """
        # Pre-spawn VRAM flush (BUG-LOCAL-048).  The subprocess inherits
        # none of our CUDA context but still shares physical 16 GB VRAM,
        # so LLM / Bark / Kokoro residues in the parent process starve
        # FLUX/LTX/Wan2.1 of load headroom.  Flush first, then cooldown.
        self._pre_spawn_vram_flush(job_id)

        # Pre-spawn cooldown gate -- LHM poll.  Never blocks forever; the
        # audio rails are protected by VRAMCoordinator in the worker, not
        # by this gate.
        self._cooldown_gate(job_id)

        # Priority 1: visual2 conda env (for real WorldMirror inference)
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
        for p_candidate in candidates:
            if p_candidate.exists():
                sidecar_python = p_candidate
                break

        if sidecar_python is None:
            log.warning(
                "[VisualBridge] No usable Python found for sidecar. Tried: %s",
                [str(c) for c in candidates],
            )
            return "SPAWN_FAILED"

        log.info("[VisualBridge] Sidecar Python: %s", sidecar_python)

        if not _WORKER_SCRIPT.exists():
            log.warning("[VisualBridge] Worker script not found at %s", _WORKER_SCRIPT)
            return "WORKER_MISSING"

        # Build sidecar env.  Only set OTR_VISUAL_BACKEND when a named
        # backend was chosen; "auto" must leave the env var unset so the
        # worker's legacy path runs unchanged.
        sidecar_env = os.environ.copy()
        if backend and backend.strip().lower() != "auto":
            sidecar_env["OTR_VISUAL_BACKEND"] = backend.strip().lower()
        else:
            sidecar_env.pop("OTR_VISUAL_BACKEND", None)

        # Redirect stdout/stderr to per-job log files.  Critical fix for
        # Windows: ``subprocess.PIPE`` without a drainer deadlocks once
        # the OS pipe buffer (~64 KB) fills, because the bridge is
        # fire-and-forget and never reads from the pipe.  Log files
        # preserve the output for post-mortem without any deadlock risk.
        log_dir = _IO_OUT / job_id
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_log = log_dir / "sidecar_stdout.log"
        stderr_log = log_dir / "sidecar_stderr.log"

        proc = None
        stdout_fp = None
        stderr_fp = None
        try:
            stdout_fp = open(stdout_log, "wb")
            stderr_fp = open(stderr_log, "wb")
            proc = subprocess.Popen(
                [str(sidecar_python), str(_WORKER_SCRIPT), str(job_dir)],
                stdout=stdout_fp,
                stderr=stderr_fp,
                cwd=str(_OTR_ROOT),
                env=sidecar_env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            log.info(
                "[VisualBridge] Sidecar spawned PID=%d for job %s (backend=%s)",
                proc.pid, job_id, backend,
            )
            atomic_write_text(job_dir / "sidecar_pid.txt", str(proc.pid))
            return "SPAWNED"
        except Exception as e:
            log.error("[VisualBridge] Sidecar spawn failed: %s", e)
            for fp in (stdout_fp, stderr_fp):
                if fp is not None:
                    try:
                        fp.close()
                    except Exception:
                        pass
            return "SPAWN_FAILED"
        finally:
            for fp in (stdout_fp, stderr_fp):
                if fp is not None:
                    try:
                        fp.close()
                    except Exception:
                        pass
            if proc is not None and proc.poll() is None and proc.returncode is not None:
                proc.terminate()

    def _write_status(self, job_id: str, status: str) -> None:
        """Write a STATUS.json to io/visual_out/<job_id>/ for the poll node."""
        out_dir = _IO_OUT / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out_dir / "STATUS.json", {
            "job_id": job_id,
            "status": status,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
