# 01_round1_robustness.md -- model=gpt-4.1 elapsed=17.1s

## User prompt

ROUND 1 of 3 -- Robustness critique.

CONTEXT BUNDLE:
The following is the entire current state of the HyWorld integration in OTR (a ComfyUI radio-drama generator). Five Python files plus the creative mapping doc plus the project rules file.

## CLAUDE.md (project rules)
```
# OldTimeRadio — Project Rules

#automation

- Before asking Jeffrey see if first filesystem extenmstion or then Desktop COmmander extension can help you do it

- If desktop commander cant help you then try WIndows MCP only if you cant use those or anotehr Connectr ask jeffrey to run a script or command

## Branch & Shipping

- All v2 work on `v2.0-alpha`. Do not touch `main`.
- Only Jeffrey merges to `main` and tags releases.

## Git Push

The GitHub connector is **context-only** — no callable write or read tools surface via tool_search. Do NOT attempt to call github_push, github_create_file, etc.

**Working pattern:**
1. Edit + stage + commit via sandbox git
2. Push via **Desktop Commander cmd shell**: `cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio && git push origin <branch>`
3. If `.git/HEAD.lock` blocks, use Desktop Commander: `del .git\HEAD.lock`
4. If Desktop Commander is unavailable, hand Jeffrey a PowerShell block with `cd` included
5. **Never use PowerShell for git** — it mangles `&&` and commit message quoting. Always use `cmd` shell.
6. After every push, verify lockstep (`local HEAD == origin HEAD`), no 0-byte files, no BOM, all node classes registered in `__init__.py`, workflow JSONs valid.

### Git Commit Messages (Desktop Commander quirk)

cmd.exe mangles commit messages containing spaces when invoked through
Desktop Commander. Do NOT use `git commit -m "message with spaces"`.

**Use temp file approach:**
```
echo Your commit message here> .git\COMMIT_EDITMSG
git commit -F .git\COMMIT_EDITMSG
```

Multi-line (subject + body):
```
(echo Subject line here& echo.& echo Body details here) > .git\COMMIT_EDITMSG
git commit -F .git\COMMIT_EDITMSG
```

Git overwrites `.git\COMMIT_EDITMSG` on next commit, so no cleanup needed.

## Testing

**Run the Bug Bible regression after EVERY code change — automatically, without being asked.** This is the primary quality gate. Do not report a change as done until all three test suites pass.

```bash
# Bug Bible regression (run this FREQUENTLY)
python -m pytest "C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py" -v

# Dropdown guardrails (new in Phase 0 cleanup)
pytest tests/test_dropdown_guardrails.py -v

# Core tests
pytest tests/test_core.py -v

# v2 audio regression (Phase 0+)
pytest tests/v2/test_audio_byte_identical.py -v
```

If any test fails after a change, fix it before moving on. New failures are bugs — log them immediately (see below).

**VRAM ceiling:** 14.5 GB peak. Never use `force_vram_offload()` between LLM phases — use `_flush_vram_keep_llm()`. Always enforce prompt truncation against `context_cap`. All LLM loaders must do a 1-token warmup pass.

### ComfyUI Live Test Monitoring

AI assistants do not have real-time access to ComfyUI console logs. During episode generation test runs:

1. **Start test run** in ComfyUI Desktop (localhost:8000)
2. **Let AI know it's running** — provide episode name, target parameters, and expected runtime
3. **When test completes or errors occur**, copy the full console output (from "Exec" node start through completion/error) and paste into the conversation
4. **AI analyzes output** — checks for PARSE_FATAL, dialogue line counts, guardrail activation logs, VRAM spikes, FFmpeg warnings, audio extraction timing
5. **Log any bugs** in BUG_LOG.md immediately — do not batch, use the pattern below

**Bug capture during live test:**
```markdown
### BUG-LOCAL-NNN: Title
- **Date:** YYYY-MM-DD | **Phase:** 0-6 | **Bible candidate:** yes/no
- **Symptom:** exact console output / error message
- **Cause:** (to be diagnosed from logs)
- **Fix:** (pending — awaiting code investigation)
- **Verify:** (pending)
- **Tags:** vram, parse-fatal, dialogue-scaling, etc.
```

## Phase 0 Cleanup (Complete)

**Guardrails locked in v2.0-alpha:**
- **BUG-009**: Preset/target_length mismatch — dynamic dialogue line scaling
- **BUG-010**: Full pre-flight sweep — character count clamps, Obsidian cap, outline temp ceiling
- **BUG-011**: Obsidian string mismatch fixed (`"Obsidian (Low VRAM/Fast)"` → `"Obsidian (UNSTABLE/4GB)"`)
- **BUG-007 root cause**: Short (3 acts) prompt now explicitly enforces `CHARACTER:` dialogue format
- **Dead widgets removed**: `news_headlines`, `temperature` (never had effect, now cleaned from UI and workflow)

**Test suite added**: `tests/test_dropdown_guardrails.py` — 56 tests covering all dropdowns, guardrails, and edge cases. Run after any parameter changes.

## Bug Log Pipeline

**Maintain `BUG_LOG.md` actively throughout development.** Every bug gets logged the moment it's found — don't batch, don't wait. This is a live document that tracks the v2 build history.

```markdown
### BUG-LOCAL-NNN: Title
- **Date:** YYYY-MM-DD | **Phase:** 0-6 | **Bible candidate:** yes/no
- **Symptom:** exact error/behavior
- **Cause:** root cause
- **Fix:** what resolved it
- **Verify:** how to confirm
- **Tags:** vram, widget-drift, ffmpeg, subprocess, etc.
```

Mark `[FIXED]` when resolved — don't delete entries. When `Bible candidate: yes` and fix is verified, promote to the survival guide repo (`C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide`):
1. Add entry to `BUG_BIBLE.yaml` (schema: `id`, `phase`, `area`, `symptom`, `cause`, `fix`, `verify`, `tags`, `legacy_id`)
2. Add regression test to `tests/bug_bible_regression.py`
3. Update `README.md` entry count
4. Run three-file contract test to confirm sync

## Content Standards

Safe for work. No profanity. Good narrative arc (beginning, middle, end). Non-violent.

## Naming Standards

Never use the word "dummy" in code, comments, variable names, or test fixtures. Use "placeholder", "stub", or a descriptive name instead (e.g. `TESTCHAR` not `DUMMY` in test fixtures). This applies to all files — source, tests, docs, and comments.

## References

| Resource | Location |
|----------|----------|
| OTR Repo | https://github.com/jbrick2070/ComfyUI-OldTimeRadio |
| Bug Bible | https://github.com/jbrick2070/comfyui-custom-node-survival-guide |
| v2 Design Spec | `docs/superpowers/specs/2026-04-12-otr-v2-visual-sidecar-design.md` |

---

## v2.0 Constraints

**Audio is king. Full narrative output must never break, shorten, or degrade. If video breaks audio, revert immediately.**

| ID | Rule |
|----|------|
| C2 | **No `CheckpointLoaderSimple`** or stock diffusion nodes in the main graph. OOM on 16 GB. |
| C3 | All visual generation in **subprocesses** via `multiprocessing.get_context("spawn")`. |
| C4 | LTX-2.3 clips **max 10-12 s** (257 frames @ 24fps). Auto-chunk + ffmpeg crossfade. |
| C5 | LTX-2.3 uses `torch.float8_e4m3fn` (Blackwell-native). |
| C6 | IP-Adapter for **environments only**, never characters (Silent Lip Bug). |
| C7 | Audio output **byte-identical** to v1.5 baseline at every gate. |

**Hardware:** RTX 5080, 16 GB VRAM, Blackwell, single GPU, no cloud.

**Python path (Windows):** `C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe` — system `python` is not on PATH. All commands must use this full path or activate the venv first with `& C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\Activate.ps1`.

**Previous attempt failed** (`v2.0-visual-engine`, deleted) by modifying node inputs (widget drift) and loading diffusion in-process (OOM). This sidecar architecture prevents both.

```


## otr_v2/hyworld/bridge.py
```python
"""
bridge.py  --  OTR_HyworldBridge ComfyUI node
===============================================
Writes Director plan + scene manifest to io/hyworld_in/,
generates shotlist via deterministic rules, and spawns the
sidecar worker process.

Design doc: docs/superpowers/specs/2026-04-15-hyworld-poc-design.md  Section 6
Mapping doc: docs/superpowers/specs/2026-04-15-otr-to-hyworld-narrative-mapping.md

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

# ---------------------------------------------------------------------------
# Paths relative to OTR repo root
# ---------------------------------------------------------------------------
_OTR_ROOT = Path(__file__).resolve().parent.parent.parent
_IO_IN = _OTR_ROOT / "io" / "hyworld_in"
_IO_OUT = _OTR_ROOT / "io" / "hyworld_out"

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

        # ---- 1. Parse script_lines ----
        try:
            script_lines = json.loads(script_json)
            if not isinstance(script_lines, list):
                raise ValueError("script_json must be a JSON array")
        except (json.JSONDecodeError, ValueError) as e:
            log.error("[HyworldBridge] script_json parse failed: %s", e)
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

        (job_dir / "script_lines.json").write_text(script_json, encoding="utf-8")
        (job_dir / "shotlist.json").write_text(shotlist_json, encoding="utf-8")
        (job_dir / "production_plan.json").write_text(production_plan_json, encoding="utf-8")
        (job_dir / "scene_manifest.json").write_text(scene_manifest_json, encoding="utf-8")

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
        (job_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

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
            # Write PID for poll node to monitor / cleanup
            (job_dir / "sidecar_pid.txt").write_text(str(proc.pid), encoding="utf-8")
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
        """Write a STATUS.json to io/hyworld_out/<job_id>/ for the poll node."""
        out_dir = _IO_OUT / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        status_file = out_dir / "STATUS.json"
        status_file.write_text(json.dumps({
            "job_id": job_id,
            "status": status,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2), encoding="utf-8")

```


## otr_v2/hyworld/poll.py
```python
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

```


## otr_v2/hyworld/renderer.py
```python
"""
renderer.py  --  OTR_HyworldRenderer ComfyUI node
===================================================
Reads geometry + images from io/hyworld_out/<job_id>/,
composites per-scene MP4 clips, crossfades to match audio length,
and muxes with the untouched v1.7 WAV.

Audio path is NEVER modified.  C7: audio output byte-identical.

Design doc: docs/superpowers/specs/2026-04-15-hyworld-poc-design.md  Section 6
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

log = logging.getLogger("OTR.hyworld.renderer")

_OTR_ROOT = Path(__file__).resolve().parent.parent.parent
_IO_OUT = _OTR_ROOT / "io" / "hyworld_out"

# Use folder_paths for ComfyUI-compatible output directory (BUG-01.02)
try:
    import folder_paths
    _RENDER_OUT = Path(folder_paths.get_output_directory()) / "hyworld_renders"
except ImportError:
    # Fallback for standalone testing
    _RENDER_OUT = _OTR_ROOT / "output" / "hyworld_renders"

# Crossfade duration between scene clips (seconds)
_CROSSFADE_SEC = 0.75

# Target output framerate
_FPS = 24


def _find_ffmpeg() -> str | None:
    """Locate ffmpeg binary. Returns path or None."""
    # Check common Windows locations
    candidates = [
        "ffmpeg",  # on PATH
        r"C:\Users\jeffr\Documents\ComfyUI\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    for c in candidates:
        if shutil.which(c):
            return c
    return None


class HyworldRenderer:
    """
    ComfyUI node: OTR_HyworldRenderer

    Reads HyWorld sidecar output (rendered frames, splat fly-throughs,
    panoramic stills) and composites them into a per-scene MP4 synced
    to the episode's audio timeline.

    If hyworld_assets_path is "FALLBACK", this node returns an empty
    video_path so the workflow can route to OTR_SignalLostVideo instead.
    """

    CATEGORY = "OTR/v2/HyWorld"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("final_mp4_path", "render_log")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "hyworld_assets_path": ("STRING", {
                    "tooltip": "Path to io/hyworld_out/<job_id>/ from OTR_HyworldPoll, or 'FALLBACK'.",
                }),
                "final_audio_path": ("STRING", {
                    "tooltip": "Path to the final episode WAV from v1.7 pipeline. NEVER modified.",
                }),
            },
            "optional": {
                "shotlist_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": "Shotlist from OTR_HyworldBridge (provides timing + camera metadata).",
                }),
                "episode_title": ("STRING", {
                    "default": "Untitled",
                    "tooltip": "Episode title for output filename.",
                }),
                "crt_postfx": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply CRT/SIGNAL LOST post-processing (scanlines, vignette, chroma bleed).",
                }),
                "output_resolution": (["1280x720", "1920x1080", "960x540"], {
                    "default": "1280x720",
                    "tooltip": "Output video resolution.",
                }),
            },
        }

    def execute(
        self,
        hyworld_assets_path: str,
        final_audio_path: str,
        shotlist_json: str = "{}",
        episode_title: str = "Untitled",
        crt_postfx: bool = True,
        output_resolution: str = "1280x720",
    ) -> tuple[str, str]:
        """
        Composite HyWorld assets into final MP4.

        If hyworld_assets_path is FALLBACK or empty, return empty path
        so the workflow routes to the procedural video fallback.
        """
        render_log_lines: list[str] = []

        def _log(msg: str) -> None:
            log.info("[HyworldRenderer] %s", msg)
            render_log_lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

        # ---- FALLBACK CHECK ----
        if not hyworld_assets_path or hyworld_assets_path == "FALLBACK":
            _log("Assets path is FALLBACK — skipping HyWorld render, routing to procedural video.")
            return ("", "\n".join(render_log_lines))

        assets_dir = Path(hyworld_assets_path)
        if not assets_dir.is_dir():
            _log(f"Assets directory does not exist: {assets_dir}. Falling back.")
            return ("", "\n".join(render_log_lines))

        # ---- PARSE SHOTLIST ----
        try:
            shotlist = json.loads(shotlist_json)
            shots = shotlist.get("shots", [])
        except (json.JSONDecodeError, AttributeError):
            shots = []

        _log(f"Assets: {assets_dir}")
        _log(f"Shotlist: {len(shots)} shots")

        # ---- COLLECT RENDERED ASSETS (before audio/ffmpeg checks) ----
        # The sidecar writes per-shot renders as:
        #   <assets_dir>/<shot_id>/render.mp4   (video clip)
        #   <assets_dir>/<shot_id>/render.png   (still frame, fallback)
        #   <assets_dir>/gaussians.ply          (3DGS, for future use)
        shot_clips: list[dict] = []
        for shot in shots:
            shot_id = shot.get("shot_id", "")
            shot_dir = assets_dir / shot_id
            duration = shot.get("duration_sec", 9)

            clip_path = shot_dir / "render.mp4"
            still_path = shot_dir / "render.png"

            if clip_path.exists():
                shot_clips.append({"path": str(clip_path), "type": "video", "duration": duration, "shot_id": shot_id})
                _log(f"  {shot_id}: video clip ({duration}s)")
            elif still_path.exists():
                shot_clips.append({"path": str(still_path), "type": "still", "duration": duration, "shot_id": shot_id})
                _log(f"  {shot_id}: still image ({duration}s)")
            else:
                _log(f"  {shot_id}: NO ASSETS FOUND (skipping)")

        if not shot_clips:
            _log("No shot assets found. Falling back to procedural video.")
            return ("", "\n".join(render_log_lines))

        # ---- VERIFY AUDIO EXISTS (never modify it) ----
        audio_path = Path(final_audio_path)
        if not audio_path.exists():
            _log(f"Audio file not found: {audio_path}. Cannot mux without audio.")
            return ("", "\n".join(render_log_lines))

        # ---- FIND FFMPEG ----
        ffmpeg = _find_ffmpeg()
        if ffmpeg is None:
            _log("ffmpeg not found on PATH or common locations. Cannot composite video.")
            return ("", "\n".join(render_log_lines))

        _log(f"ffmpeg: {ffmpeg}")
        _log(f"Audio: {audio_path} (read-only, byte-identical guarantee)")

        # ---- OUTPUT PATH ----
        _RENDER_OUT.mkdir(parents=True, exist_ok=True)
        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in episode_title)[:60]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_path = _RENDER_OUT / f"{safe_title}_{timestamp}.mp4"

        # ---- BUILD FFMPEG CONCAT ----
        width, height = output_resolution.split("x")
        concat_list = assets_dir / "_concat_list.txt"

        with open(concat_list, "w", encoding="utf-8") as f:
            for clip in shot_clips:
                if clip["type"] == "video":
                    f.write(f"file '{clip['path']}'\n")
                else:
                    # Convert still to a clip: ffmpeg -loop 1 -t <dur> -i still.png
                    # We generate a temp clip first
                    temp_clip = assets_dir / clip["shot_id"] / "_temp_clip.mp4"
                    _still_to_clip(ffmpeg, clip["path"], str(temp_clip), clip["duration"], width, height)
                    f.write(f"file '{temp_clip}'\n")

        # ---- CONCAT + MUX WITH AUDIO ----
        # Step 1: concat visual clips
        concat_out = assets_dir / "_concat_video.mp4"
        cmd_concat = [
            ffmpeg, "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264", "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-s", f"{width}x{height}",
            str(concat_out),
        ]
        _log(f"Concatenating {len(shot_clips)} clips...")
        _run_ffmpeg(cmd_concat, _log)

        if not concat_out.exists():
            _log("Concat failed. Falling back.")
            return ("", "\n".join(render_log_lines))

        # Step 2: mux concat video + original audio (audio is never re-encoded)
        cmd_mux = [
            ffmpeg, "-y",
            "-i", str(concat_out),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "copy",        # byte-identical audio passthrough (C7)
            "-shortest",
            str(out_path),
        ]
        _log("Muxing video + audio (audio passthrough, C7 guaranteed)...")
        _run_ffmpeg(cmd_mux, _log)

        if out_path.exists():
            size_mb = out_path.stat().st_size / (1024 * 1024)
            _log(f"Output: {out_path} ({size_mb:.1f} MB)")
        else:
            _log("Mux failed. No output file.")
            return ("", "\n".join(render_log_lines))

        # ---- CLEANUP TEMP FILES ----
        for f in [concat_list, concat_out]:
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass

        return (str(out_path), "\n".join(render_log_lines))


def _still_to_clip(
    ffmpeg: str, still_path: str, out_path: str,
    duration: float, width: str, height: str,
) -> None:
    """Convert a still image to a video clip of given duration."""
    cmd = [
        ffmpeg, "-y", "-loop", "1",
        "-i", still_path,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-s", f"{width}x{height}",
        "-r", str(_FPS),
        out_path,
    ]
    subprocess.run(cmd, capture_output=True, timeout=120)


def _run_ffmpeg(cmd: list[str], log_fn) -> None:
    """Run an ffmpeg command, logging stderr on failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300, text=True)
        if result.returncode != 0:
            log_fn(f"ffmpeg error (rc={result.returncode}): {result.stderr[:500]}")
    except subprocess.TimeoutExpired:
        log_fn("ffmpeg timed out after 300s")
    except FileNotFoundError:
        log_fn("ffmpeg binary not found at runtime")

```


## otr_v2/hyworld/worker.py
```python
"""
worker.py  --  HyWorld sidecar worker (runs in hyworld2 conda env, or main venv fallback)
=========================================================================================
This script is spawned by OTR_HyworldBridge as a subprocess.
It reads the contract files from io/hyworld_in/<job_id>/,
runs HyWorld inference, and writes results to io/hyworld_out/<job_id>/.

IMPORTANT: This may run in a SEPARATE Python environment (torch 2.4, CUDA 12.4)
from the main ComfyUI process (torch 2.10, CUDA 13.0).  Do NOT import
any OTR node code or ComfyUI modules.  The worker also runs under the
main ComfyUI .venv as a fallback (stub modes only) when the hyworld2
conda env is unavailable.

Usage (called by bridge.py, not by humans):
    python otr_v2/hyworld/worker.py <path_to_job_dir>

Status protocol:
    The worker writes STATUS.json to io/hyworld_out/<job_id>/ with:
        {"status": "RUNNING"|"READY"|"ERROR"|"OOM", "detail": "...", ...}
    The poll node reads this file to determine completion.

Tiered execution path (selected at runtime, no model contention with audio):
    1. WorldMirror 2.0 inference  - if hyworld2 env + weights present
    2. Motion stub (real MP4)     - if ffmpeg on PATH (Ken Burns clips
                                    driven by shotlist camera adjective).
                                    Uses CPU only, safe to run while Bark
                                    TTS holds the GPU.
    3. Still stub (solid PNG)     - last-resort, no external deps.

The motion stub is the current default for testing the full Bridge ->
Poll -> Renderer path with an MP4 output.  Real generative video
(SVD / LTX-Video) is the next phase and requires GPU coordination
with the audio pipeline (must wait until BatchBark releases VRAM).
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path


def _write_status(out_dir: Path, status: str, detail: str = "") -> None:
    """Write STATUS.json for the poll node."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "STATUS.json").write_text(json.dumps({
        "status": status,
        "detail": detail,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2), encoding="utf-8")


def _create_placeholder_png(path: Path, width: int = 1280, height: int = 720,
                             r: int = 30, g: int = 30, b: int = 40) -> None:
    """
    Write a minimal valid PNG (solid color, no external deps).
    Uses raw DEFLATE via zlib.  No Pillow required.
    """
    import zlib

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    # IDAT — uncompressed scanlines (filter byte 0 + RGB per pixel)
    row = bytes([0] + [r, g, b] * width)
    raw = row * height
    compressed = zlib.compress(raw, 1)

    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", ihdr_data)
    png += _chunk(b"IDAT", compressed)
    png += _chunk(b"IEND", b"")

    path.write_bytes(png)


# ---------------------------------------------------------------------------
# ffmpeg discovery + Ken Burns motion clip generation
# ---------------------------------------------------------------------------

# Output dimensions for stub clips.  Matches renderer default; the renderer
# will rescale at concat time anyway, but rendering at the target size keeps
# zoompan math honest.
_CLIP_WIDTH = 1280
_CLIP_HEIGHT = 720
_CLIP_FPS = 24

# Camera adjective (from shotlist.py voice-traits map) -> ffmpeg zoompan recipe.
# Each entry is a function (duration_sec, width, height) -> filter_chain str.
# Formulas use `on` (current input frame) and `d` (total animation frames)
# because -loop 1 feeds zoompan one new frame per output frame.

def _motion_static(d: int, w: int, h: int) -> str:
    return f"zoompan=z=1.0:d={d}:s={w}x{h}:fps={_CLIP_FPS}"

def _motion_slow_push_in(d: int, w: int, h: int) -> str:
    # 1.00 -> ~1.30 over the duration, centered crop.
    return (
        f"zoompan=z='min(1.0+0.30*on/{d},1.30)':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={d}:s={w}x{h}:fps={_CLIP_FPS}"
    )

def _motion_fast_dolly(d: int, w: int, h: int) -> str:
    # Faster zoom, slight off-center for canted feel.
    return (
        f"zoompan=z='min(1.0+0.55*on/{d},1.55)':"
        f"x='iw*0.45-(iw/zoom/2)':y='ih*0.55-(ih/zoom/2)':"
        f"d={d}:s={w}x{h}:fps={_CLIP_FPS}"
    )

def _motion_clean_push(d: int, w: int, h: int) -> str:
    return (
        f"zoompan=z='min(1.0+0.40*on/{d},1.40)':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={d}:s={w}x{h}:fps={_CLIP_FPS}"
    )

def _motion_whip_pan(d: int, w: int, h: int) -> str:
    # Constant zoom 1.3, sweep horizontally across the available range.
    return (
        f"zoompan=z=1.30:"
        f"x='(iw-iw/zoom)*on/{d}':y='ih/2-(ih/zoom/2)':"
        f"d={d}:s={w}x{h}:fps={_CLIP_FPS}"
    )

def _motion_low_angle(d: int, w: int, h: int) -> str:
    # Constant zoom 1.25, drift y from bottom to top (looking up).
    return (
        f"zoompan=z=1.25:"
        f"x='iw/2-(iw/zoom/2)':y='(ih-ih/zoom)*(1-on/{d})':"
        f"d={d}:s={w}x{h}:fps={_CLIP_FPS}"
    )

def _motion_macro(d: int, w: int, h: int) -> str:
    # Very slow zoom 1.0 -> 1.15, centered.
    return (
        f"zoompan=z='min(1.0+0.15*on/{d},1.15)':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={d}:s={w}x{h}:fps={_CLIP_FPS}"
    )

def _motion_slow_drift(d: int, w: int, h: int) -> str:
    # Default: gentle zoom 1.0 -> 1.20 with a small horizontal drift.
    return (
        f"zoompan=z='min(1.0+0.20*on/{d},1.20)':"
        f"x='(iw-iw/zoom)*(0.40+0.20*on/{d})':y='ih/2-(ih/zoom/2)':"
        f"d={d}:s={w}x{h}:fps={_CLIP_FPS}"
    )

# First-substring-match wins (case-insensitive).  Matches the camera
# adjectives produced by shotlist._camera_from_traits().
_CAMERA_MOTION_MAP: list[tuple[str, callable]] = [
    ("locked off",      _motion_static),
    ("clean push",      _motion_clean_push),
    ("slow handheld",   _motion_slow_push_in),
    ("fast dolly",      _motion_fast_dolly),
    ("whip-pan",        _motion_whip_pan),
    ("low angle",       _motion_low_angle),
    ("macro detail",    _motion_macro),
    ("slow drift",      _motion_slow_drift),
]


def _camera_to_motion(camera: str, duration_sec: float, w: int, h: int) -> tuple[str, str]:
    """Resolve a shotlist camera adjective to an ffmpeg filter chain.

    Returns (motion_label, filter_chain_string).
    """
    # Total animation frames at our fixed output FPS.
    d = max(1, int(round(duration_sec * _CLIP_FPS)))
    cam_lower = (camera or "").lower()
    for needle, fn in _CAMERA_MOTION_MAP:
        if needle in cam_lower:
            return (needle, fn(d, w, h))
    return ("default_drift", _motion_slow_drift(d, w, h))


def _find_ffmpeg() -> str | None:
    """Locate ffmpeg.  Mirrors renderer._find_ffmpeg search order."""
    candidates = [
        "ffmpeg",
        r"C:\Users\jeffr\Documents\ComfyUI\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    for c in candidates:
        resolved = shutil.which(c)
        if resolved:
            return resolved
    return None


def _make_motion_clip(
    ffmpeg: str,
    still_path: Path,
    out_path: Path,
    duration_sec: float,
    camera: str,
) -> tuple[bool, str]:
    """Render a Ken Burns MP4 from a still using the camera-derived motion.

    Returns (ok, motion_label).  On failure, motion_label is the ffmpeg
    stderr tail so the caller can record it.
    """
    motion_label, vfilter = _camera_to_motion(camera, duration_sec, _CLIP_WIDTH, _CLIP_HEIGHT)
    # Compute exact target frame count.  zoompan multiplies frames (d output
    # frames per input frame), so we must:
    #   - feed exactly 1 input frame  (-framerate 1 -loop 1 -t 1)
    #   - cap output at N frames      (-frames:v N)
    # Otherwise the duration explodes by ~d^2.  See BUG-LOCAL-014.
    target_frames = max(1, int(round(duration_sec * _CLIP_FPS)))
    cmd = [
        ffmpeg, "-y",
        "-loop", "1",
        "-framerate", "1",
        "-t", "1",
        "-i", str(still_path),
        "-vf", vfilter,
        "-frames:v", str(target_frames),
        "-c:v", "libx264", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-r", str(_CLIP_FPS),
        "-an",  # no audio track in the clip; renderer muxes episode WAV separately
        str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120, text=True)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return (False, f"ffmpeg-spawn-error:{exc.__class__.__name__}")
    if result.returncode != 0 or not out_path.exists():
        return (False, f"ffmpeg-rc{result.returncode}:{(result.stderr or '')[-200:]}")
    return (True, motion_label)


def run_stub(job_dir: Path) -> None:
    """
    Stub worker: read shotlist, create placeholder stills per shot.
    Replace this function with real WorldMirror 2.0 inference when
    the hyworld2 env is ready.
    """
    # Derive paths
    job_id = job_dir.name
    otr_root = job_dir.parent.parent.parent  # io/hyworld_in/<job_id> -> repo root
    out_dir = otr_root / "io" / "hyworld_out" / job_id

    _write_status(out_dir, "RUNNING", "Stub worker generating placeholders")

    # Read shotlist
    shotlist_path = job_dir / "shotlist.json"
    if not shotlist_path.exists():
        _write_status(out_dir, "ERROR", "shotlist.json not found in job dir")
        return

    shotlist = json.loads(shotlist_path.read_text(encoding="utf-8"))
    shots = shotlist.get("shots", [])

    if not shots:
        _write_status(out_dir, "ERROR", "shotlist has zero shots")
        return

    # ffmpeg is the gate between still-stub and motion-stub modes.  When
    # present we upgrade each shot from a static PNG to a Ken Burns MP4
    # driven by the shotlist camera adjective.  CPU-only -> safe to run
    # alongside Bark TTS on the GPU.
    ffmpeg = _find_ffmpeg()

    motion_ok = 0
    motion_failed = 0
    backend_label = "stub_motion_clip" if ffmpeg else "stub_placeholder_still"

    # Create per-shot assets (still always written; mp4 written when ffmpeg is up).
    for i, shot in enumerate(shots):
        shot_id = shot.get("shot_id", f"shot_{i:03d}")
        shot_dir = out_dir / shot_id
        shot_dir.mkdir(parents=True, exist_ok=True)

        # Vary color slightly per shot for visual distinction
        r = 30 + (i * 17) % 60
        g = 30 + (i * 23) % 60
        b = 40 + (i * 31) % 60

        still_path = shot_dir / "render.png"
        _create_placeholder_png(still_path, r=r, g=g, b=b)

        camera = shot.get("camera", "")
        duration = float(shot.get("duration_sec", 9))
        motion_label = "still_only"
        ffmpeg_detail = ""

        if ffmpeg is not None:
            mp4_path = shot_dir / "render.mp4"
            ok, label = _make_motion_clip(ffmpeg, still_path, mp4_path, duration, camera)
            if ok:
                motion_ok += 1
                motion_label = label
            else:
                motion_failed += 1
                ffmpeg_detail = label  # holds the error tail

        # Write shot metadata (now includes resolved motion + backend)
        meta = {
            "shot_id": shot_id,
            "env_prompt": shot.get("env_prompt", ""),
            "camera": camera,
            "duration_sec": duration,
            "backend": backend_label,
            "motion": motion_label,
        }
        if ffmpeg_detail:
            meta["ffmpeg_detail"] = ffmpeg_detail
        (shot_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if ffmpeg is None:
        detail = f"Still-only stub: {len(shots)} placeholder PNGs (ffmpeg not found)"
    elif motion_failed == 0:
        detail = f"Motion stub: {motion_ok} Ken Burns MP4 clips generated"
    else:
        detail = (
            f"Motion stub: {motion_ok} MP4 clips, {motion_failed} fell back to still"
        )
    _write_status(out_dir, "READY", detail)


def run_worldmirror(job_dir: Path) -> None:
    """
    Real WorldMirror 2.0 inference.  Activated when the hyworld2 env
    has the model installed.

    TODO: Implement when conda env + weights are verified.
    Skeleton left here so the entry point is clear.
    """
    job_id = job_dir.name
    otr_root = job_dir.parent.parent.parent
    out_dir = otr_root / "io" / "hyworld_out" / job_id

    _write_status(out_dir, "RUNNING", "WorldMirror 2.0 inference starting")

    try:
        # Step 1: Check if WorldMirror is importable
        from hyworld2.worldrecon.pipeline import WorldMirrorPipeline  # type: ignore
    except ImportError:
        _write_status(out_dir, "ERROR", "WorldMirrorPipeline not installed in this env")
        return

    # Step 2: Load model (first run downloads weights)
    # Step 3: For each shot, run inference on panorama images
    # Step 4: Write gaussians.ply, depth maps, rendered frames to out_dir
    # Step 5: Write READY status

    _write_status(out_dir, "ERROR", "WorldMirror integration not yet implemented")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python worker.py <job_dir>", file=sys.stderr)
        sys.exit(1)

    job_dir = Path(sys.argv[1])
    if not job_dir.is_dir():
        print(f"Job directory not found: {job_dir}", file=sys.stderr)
        sys.exit(1)

    job_id = job_dir.name
    otr_root = job_dir.parent.parent.parent
    out_dir = otr_root / "io" / "hyworld_out" / job_id

    try:
        # Try real inference first; fall back to stub
        try:
            from hyworld2.worldrecon.pipeline import WorldMirrorPipeline  # type: ignore  # noqa: F401
            run_worldmirror(job_dir)
        except ImportError:
            run_stub(job_dir)
    except Exception:
        _write_status(out_dir, "ERROR", traceback.format_exc()[-500:])
        sys.exit(1)


if __name__ == "__main__":
    main()

```


## otr_v2/hyworld/shotlist.py
```python
"""
shotlist.py  --  Deterministic OTR script_lines -> shots[] mapper
=================================================================
Reads the Canonical Audio Tokens produced by _parse_script() and
produces a shots[] array suitable for driving the HyWorld sidecar
(or any interim stand-in: Diffusion360, SPAG4D, ComfyUI-Sharp).

Design doc: docs/superpowers/specs/2026-04-15-otr-to-hyworld-narrative-mapping.md
    Sections 4 (per-token mapping), 5 (deterministic vs LLM), 12.2 (Lane 1)

Rules:
    - Pure functions.  No GPU, no model loads, no network.
    - Deterministic.  Same script_lines in -> same shots[] out.
    - No v1.7 imports.  Reads JSON, returns JSON.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

log = logging.getLogger("OTR.hyworld.shotlist")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SHOT_DURATION_SEC = 12   # C4 ceiling
MIN_SHOT_DURATION_SEC = 3    # minimum viable clip
DEFAULT_SHOT_DURATION_SEC = 9
MAX_SHOTS_PER_SCENE = 4      # VRAM predictability

# Voice-traits -> camera adjective lookup  (Section 4.5)
_TRAIT_CAMERA_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:weary|tired|old|exhausted)\b", re.I), "slow handheld, close"),
    (re.compile(r"\b(?:angry|hostile|sharp|furious)\b", re.I), "fast dolly, canted"),
    (re.compile(r"\b(?:calm|warm|gentle|serene)\b", re.I), "locked off, wide"),
    (re.compile(r"\b(?:frantic|panicked|manic)\b", re.I), "whip-pan, short focal length"),
    (re.compile(r"\b(?:announcer|formal|narrator)\b", re.I), "clean push-in, centered"),
    (re.compile(r"\b(?:child|young|teen)\b", re.I), "low angle, looking up"),
    (re.compile(r"\b(?:whisper|hushed|quiet)\b", re.I), "macro detail, shallow focus"),
]
_DEFAULT_CAMERA = "slow drift, medium lens"


def _camera_from_traits(voice_traits: str) -> str:
    """Map voice_traits string to a camera adjective via first-match lookup."""
    for pattern, camera in _TRAIT_CAMERA_MAP:
        if pattern.search(voice_traits):
            return camera
    return _DEFAULT_CAMERA


def _mood_from_traits(voice_traits: str) -> str:
    """Extract the dominant mood keyword from voice_traits."""
    for pattern, _ in _TRAIT_CAMERA_MAP:
        m = pattern.search(voice_traits)
        if m:
            return m.group(0).lower()
    return "neutral"


def _style_anchor_hash(episode_title: str, first_env: str) -> str:
    """12-char hex hash for episode-wide visual consistency."""
    payload = f"{episode_title}:{first_env}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Scene splitter
# ---------------------------------------------------------------------------

def _split_into_scenes(script_lines: list[dict]) -> list[dict]:
    """
    Split flat script_lines into scenes.

    Returns a list of scene dicts:
        {
            "scene_id": str,
            "tokens": [token, ...],
            "env_descriptions": [str, ...],
            "dialogue_lines": [{character_name, voice_traits, line, index}, ...],
            "sfx_cues": [{description, index}, ...],
            "beat_count": int,
        }
    """
    scenes: list[dict] = []
    current: dict | None = None

    def _new_scene(label: str) -> dict:
        return {
            "scene_id": label,
            "tokens": [],
            "env_descriptions": [],
            "dialogue_lines": [],
            "sfx_cues": [],
            "beat_count": 0,
        }

    for idx, token in enumerate(script_lines):
        ttype = token.get("type", "")

        if ttype == "title":
            continue  # skip title token, not scene-relevant

        if ttype == "scene_break":
            if current is not None:
                scenes.append(current)
            current = _new_scene(token.get("scene", str(len(scenes) + 1)))
            continue

        # If we haven't seen a scene_break yet, start an implicit Scene 1
        if current is None:
            current = _new_scene("1")

        current["tokens"].append(token)

        if ttype == "environment":
            current["env_descriptions"].append(token.get("description", ""))
        elif ttype == "dialogue":
            current["dialogue_lines"].append({
                "character_name": token.get("character_name", "UNKNOWN"),
                "voice_traits": token.get("voice_traits", ""),
                "line": token.get("line", ""),
                "index": idx,
            })
        elif ttype == "sfx":
            current["sfx_cues"].append({
                "description": token.get("description", ""),
                "index": idx,
            })
        elif ttype == "pause":
            current["beat_count"] += 1

    if current is not None:
        scenes.append(current)

    return scenes


# ---------------------------------------------------------------------------
# Shot builder
# ---------------------------------------------------------------------------

def _build_shots_for_scene(scene: dict, scene_idx: int) -> list[dict]:
    """
    Build 1-4 shots for a single scene using deterministic rules.

    Each shot gets:
        shot_id, scene_ref, duration_sec, camera, env_prompt,
        sfx_accents, dialogue_line_ids, mood, visual_backend
    """
    env_prompt = scene["env_descriptions"][0] if scene["env_descriptions"] else "empty room, dim light"
    dialogue = scene["dialogue_lines"]
    sfx_cues = scene["sfx_cues"]

    # Estimate scene audio duration from token count
    # Rough heuristic: 2.5s per dialogue line, 1s per SFX, 0.2s per beat
    est_duration = (
        len(dialogue) * 2.5
        + len(sfx_cues) * 1.0
        + scene["beat_count"] * 0.2
    )
    est_duration = max(MIN_SHOT_DURATION_SEC, est_duration)

    # Determine dominant mood from first dialogue's voice_traits
    dominant_traits = dialogue[0]["voice_traits"] if dialogue else ""
    camera = _camera_from_traits(dominant_traits)
    mood = _mood_from_traits(dominant_traits)

    # Split scene into shots (each capped at MAX_SHOT_DURATION_SEC)
    num_shots = min(
        MAX_SHOTS_PER_SCENE,
        max(1, int(est_duration / DEFAULT_SHOT_DURATION_SEC + 0.5)),
    )
    shot_duration = min(MAX_SHOT_DURATION_SEC, max(MIN_SHOT_DURATION_SEC, est_duration / num_shots))

    shots: list[dict] = []
    dl_cursor = 0
    sfx_cursor = 0

    for shot_num in range(num_shots):
        shot_id = f"s{scene_idx + 1:02d}_{shot_num + 1:02d}"

        # Distribute dialogue lines across shots
        dl_per_shot = max(1, len(dialogue) // num_shots) if dialogue else 0
        dl_start = dl_cursor
        dl_end = min(len(dialogue), dl_cursor + dl_per_shot) if shot_num < num_shots - 1 else len(dialogue)
        shot_dialogue_ids = [f"line_{d['index']}" for d in dialogue[dl_start:dl_end]]
        dl_cursor = dl_end

        # Update camera if this chunk has a different mood
        if dl_start < len(dialogue):
            chunk_traits = dialogue[dl_start]["voice_traits"]
            shot_camera = _camera_from_traits(chunk_traits)
            shot_mood = _mood_from_traits(chunk_traits)
        else:
            shot_camera = camera
            shot_mood = mood

        # Distribute SFX accents
        sfx_per_shot = max(1, len(sfx_cues) // num_shots) if sfx_cues else 0
        sfx_start = sfx_cursor
        sfx_end = min(len(sfx_cues), sfx_cursor + sfx_per_shot) if shot_num < num_shots - 1 else len(sfx_cues)
        shot_sfx = [
            {"at": round((i - sfx_start) * (shot_duration / max(1, sfx_end - sfx_start)), 1),
             "desc": s["description"]}
            for i, s in enumerate(sfx_cues[sfx_start:sfx_end])
        ]
        sfx_cursor = sfx_end

        # Use secondary env descriptions for later shots if available
        shot_env = scene["env_descriptions"][min(shot_num, len(scene["env_descriptions"]) - 1)] if scene["env_descriptions"] else env_prompt

        shots.append({
            "shot_id": shot_id,
            "scene_ref": scene["scene_id"],
            "duration_sec": round(shot_duration, 1),
            "camera": shot_camera,
            "env_prompt": shot_env,
            "sfx_accents": shot_sfx,
            "dialogue_line_ids": shot_dialogue_ids,
            "mood": shot_mood,
            "visual_backend": {
                "pano": "diffusion360",
                "stereo": "spag4d_da360",
                "nav": "hand_authored",
            },
        })

    return shots


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_shotlist(
    script_lines: list[dict],
    episode_title: str = "Untitled Episode",
) -> dict[str, Any]:
    """
    Main entry point.  Deterministic Lane 1 mapping.

    Args:
        script_lines: the Canonical Audio Token array from _parse_script().
        episode_title: used for style_anchor_hash generation.

    Returns:
        A dict with keys:
            "shots": list of shot dicts
            "style_anchor_hash": 12-char hex
            "scene_count": int
            "total_shots": int
    """
    scenes = _split_into_scenes(script_lines)

    all_shots: list[dict] = []
    for idx, scene in enumerate(scenes):
        scene_shots = _build_shots_for_scene(scene, idx)
        all_shots.extend(scene_shots)

    # Style anchor from first environment in Act 1
    first_env = ""
    for scene in scenes:
        if scene["env_descriptions"]:
            first_env = scene["env_descriptions"][0]
            break

    return {
        "shots": all_shots,
        "style_anchor_hash": _style_anchor_hash(episode_title, first_env),
        "scene_count": len(scenes),
        "total_shots": len(all_shots),
    }


def generate_shotlist_json(
    script_lines_json: str,
    episode_title: str = "Untitled Episode",
) -> str:
    """Convenience wrapper: JSON string in, JSON string out."""
    lines = json.loads(script_lines_json)
    result = generate_shotlist(lines, episode_title)
    return json.dumps(result, indent=2)

```


## docs/superpowers/specs/2026-04-15-otr-to-hyworld-narrative-mapping.md (creative mapping doc -- excerpt)
```
# OTR -> HyWorld Narrative Mapping (Design Ideas, Pre-Build)

**Date:** 2026-04-15
**Branch:** `v2.0-alpha`
**Status:** Ideas doc for visualization + review. Not an implementation spec.
**Owner:** Jeffrey A. Brick
**Companions:**
- `docs/superpowers/specs/2026-04-15-hyworld-poc-design.md` (architecture / install / nodes)
- `docs/superpowers/specs/2026-04-15-hyworld-integration-plan-review.md` (keep/discard triage)
- `docs/OTR_PIPELINE_EXPLAINER.md` (layman's pipeline)

---

## 0. Why this doc exists

The PoC doc says **how** HyWorld plugs in (sidecar, JSON contract, three new nodes).
This doc answers the creative half: **what OTR already knows about a story, and what you could point HyWorld at to make meaningful art out of it.**

OTR's output is a radio drama: dialogue-first, SFX-heavy, environments painted in sound. HyWorld 2.0 is a 3D reconstruction / world-generation stack. The interesting question isn't "can we draw a picture of the script" — it's "what does it look like when a radio play dreams."

Jeffrey plans to visualize the mapping tables in this doc and pass them around for feedback before we lock a schema.

---

## 1. What OTR already parses today (ground truth)

From `nodes/story_orchestrator.py::_parse_script` and the downstream `scene_sequencer`, every generated episode produces a structured `script_lines` array. These are the "Canonical Audio Tokens" — the only shapes of information OTR guarantees are present in every episode.

| Token type      | Fields                                                  | Example                                              |
|-----------------|---------------------------------------------------------|------------------------------------------------------|
| `title`         | `value`                                                 | `"The Last Signal from Vault 7"`                     |
| `scene_break`   | `scene` (label / number)                                | `=== SCENE 2 ===` -> `scene = "2"`                   |
| `environment`   | `description` (3-4 descriptors, prose)                  | `[ENV: fluorescent hum, distant traffic, rain]`      |
| `sfx`           | `description` (single event, prose)                     | `[SFX: heavy wrench strike on metal pipe]`           |
| `pause`         | `kind=beat`, `duration_ms=200`                          | `(beat)`                                             |
| `dialogue`      | `character_name`, `voice_traits`, `line`                | `[VOICE: COMMANDER, male, 50s, weary] "Hold the line."` |
| `direction`     | raw non-tag prose kept for flavor                       | stage directions the LLM emitted between tags        |
| `music`         | `description` (extracted from `[MUSIC: ...]`)           | `[MUSIC: Opening theme]`                             |

Two derived structures are also available per episode:
- **Character roster** — unique `character_name` set, each with accumulated `voice_traits` strings (gender, age, tone). This is a natural source for character cards.
- **Scene table** — every `scene_break` splits the timeline. Environments and SFX between two breaks belong to that scene; dialogue clusters inside scenes.

And two timing structures the audio pipeline builds:
- **Audio offsets** — each token gets an absolute `audio_offset_s` after SceneSequencer renders. This is how HyWorld shots will sync to the WAV.
- **Duration per scene** — derived by summing dialogue + SFX + pauses inside a scene.

**This is the raw material HyWorld can eat.** Nothing else is guaranteed to exist. Anything richer (shot lists, camera motion, mood beats) has to be *inferred from these tokens*, either by us in deterministic code or by a small LLM pass that reads the script and emits a `shots[]` array. Both options are discussed below.

---

## 2. What HyWorld 2.0 actually wants as input (current, 2026-04-15)

Only one model is shipped right now — everything else on the HY-World 2.0 page is "Coming Soon." We must design for today's reality and mark forward-looking entries as speculative.

### 2.1 Shipped today — WorldMirror 2.0

Multi-view / video -> 3D reconstruction.

**Wants:**
- 5-20 images of a scene from different viewpoints (or a short video clip).
- Resolution roughly 512x512 and up, consistent enough to correspond.

**Produces:**
- 3D geometry (point cloud or 3DGS), depth maps, camera poses.

**So the creative question is: how do we get 5-20 coherent images per scene when the source material is a radio play?** Several candidate answers are in Section 3.

### 2.2 Coming Soon — HY-Pano-2.0

Text / image -> 360 panorama.

**Speculated input:** a prose environment description (perfectly what `[ENV:]` tokens already are) plus optional reference image.
**Produces:** equirectangular panorama of a place.

When this ships, it's a near-direct match to OTR's `environment` tokens — probably the cleanest mapping in the entire stack.

### 2.3 Coming Soon — WorldStereo 2.0

Panorama -> navigable 3DGS.

**Speculated input:** a pano from HY-Pano-2.0.
**Produces:** walkable Gaussian splat of the panoramic scene.

This is the model that could turn an OTR scene into a space the viewer can move through while the dialogue plays. High creative payoff, but blocked on Pano shipping.

### 2.4 Coming Soon — WorldNav

Trajectory planning inside a 3DGS scene.

**Speculated input:** a scene + a high-level intent ("dolly forward", "look up at the light").
**Produces:** a camera path.

Matches one-to-one with a `camera` field on the Director `shots[]` array we already reserved in the PoC doc.

---

## 3. Three candidate mappings, ordered from "actually buildable today" to "aspirational"

### 3.1 Mapping A — Single anchor image per scene, WorldMirror does the rest (PoC target)

**Idea:** for each scene, generate one "anchor" image from the `[ENV:]` description using any text-to-image model available locally (existing OTR toolchain already has ComfyUI + SDXL / Flux). Then synthesize 8-12 virtual viewpoints around it by small camera perturbations and feed them into WorldMirror 2.0 to lift to 3D.

**Why this first:** everything needed ships today. No dependency on Pano / Stereo / Nav.

**OTR token -> HyWorld input:**

| OTR token                      | Becomes                                                   |
|--------------------------------|-----------------------------------------------------------|
| `environment.description`      | Text prompt for the anchor image                          |
| First dominant `sfx` in scene  | Adjective modifier on the anchor prompt ("rain-soaked")   |
| `scene.duration_s` (derived)   | Clip length budget (capped at 12 s per C4)                |
| `character_name` + roster      | **Not used in the image** (C6: IP-Adapter environments only, never characters) |
| `scene.title` / `scene_break`  | Filename / anchor key                                     |

**Creative framing:** scenes become "places that were heard, now seen." The listener hears dialogue in a room they have never visited but can now walk through. Not a shot of the Commander — a shot of the room the Commander was in.

**Honest limits:** WorldMirror lifts geometry from images; if the anchor image is wrong, the 3D is wrong. Hallucinated anchor + WorldMirror = a coherent but wrong room. Acceptable for PoC. Sell the weirdness — it is on-brand for SIGNAL LOST.

### 3.2 Mapping B — Pano-first, when Pano ships

**Idea:** feed `[ENV:]` descriptions directly into HY-Pano-2.0 -> panorama -> WorldStereo -> navigable splat. Dialog timestamps drive camera paths via WorldNav.

**Why this is the eventual goal:** the `[ENV:]` token was already designed to be a 3-4 descriptor prose chunk ("fluorescent hum, distant traffic, rain on concrete"). That is exactly the shape a text-to-pano model wants. Zero glue code between OTR parse output and HyWorld input.

**OTR token -> HyWorld input:**

| OTR token                      | Becomes                                                   |
|--------------------------------|-----------------------------------------------------------|
| `environment.description`      | HY-Pano-2.0 text prompt (verbatim, no rewriting)          |
| Secondary `environment` tokens in same scene | Pano style anchor (shifting mood mid-scene)    |
| `scene_break`                  | Pano boundary                                             |
| `sfx` cluster density          | Implies visual busyness - feeds pano "detail" knob if exposed |
| `dialogue` clusters            | WorldNav keyframes ("dolly to CHARACTER position")        |
| `character_name` positions     | Named anchors in the 3DGS for nav targets (positions only, not faces — C6) |

**Creative framing:** every OTR episode becomes a walkable audio drama. Listeners move through the pano as dialogue plays. Think the old-radio-drama equivalent of a VR space that exists only because a voice described it.

**Blocker:** Pano + Stereo + Nav are "Coming Soon." Revisit when they ship. Do not block PoC on this.

### 3.3 Mapping C — Emotional geometry (speculative, aesthetic)

**Idea:** stop treating the 3D output as a realistic depiction of the script's setting, and start treating it as a **visual analogue of the emotional state.** Character moods, scene tension, and SFX density drive geometry parameters; the dialogue audio plays underneath.

Example: a tense argument between two characters produces a cramped, high-contrast, low-ceilinged geometry; a quiet reflective monologue produces open spaces with soft light. The scene doesn't depict a real place; it depicts the *feeling* of the place.

**OTR token -> HyWorld input:**

| OTR token                                                | Maps to                                    |
|----------------------------------------------------------|--------------------------------------------|
| `voice_traits` keywords (angry, weary, calm, frantic)    | Anchor prompt adjectives                   |
| Dialogue density (lines/sec)                             | Camera cut rate                            |
| SFX density                                              | Scene complexity / particle density        |
| Unique speaker count in scene                            | Depth of field / scene scale               |
| Ratio of `(beat)` tokens to dialogue                     | Tempo of virtual camera                    |
| Presence of `[MUSIC:]` tokens                            | Color palette shift                        |
| Episode title                                            | Global style anchor (consistent across scenes) |

**Creative framing:** this is the Codex-Olympia-style move. Radio drama is already non-visual; forcing literal visuals is a step down. But visualizing the *emotional architecture* of a scene is a step sideways into new territory. LA28 anthology-friendly. Gallery-friendly.

**Honest risk:** can slide into generic "abstract visuals" territory if the mappings are not specific enough. Counter: rigid, deterministic rules per token type. No LLM second-guessing on mood; pick the adjectives from the trait string verbatim.

---

## 4. Concrete per-token mapping ideas (detailed tables)

These are starter values — they are the "first draft" Jeffrey will critique. Do not treat them as canonical.

### 4.1 `environment.description` (cleanest case)

| Destination                            | Treatment                                             |
|----------------------------------------|-------------------------------------------------------|
| Anchor image prompt (Mapping A)        | Verbatim, prepended with "wide establishing shot of " |
| HY-Pano-2.0 prompt (Mapping B)         | Verbatim, no modifier                                 |
| Style anchor token for episode         | Hash of first `environment` in Act 1, reused throughout |
| SFX enrichment                         | Append dominant `sfx` adjective when pano detail is low |

**Why `[ENV:]` is the anchor:** Jeffrey's own prompt rules already require `[ENV:]` to be 3-4 concrete descriptors, not abstractions. That is the exact shape a generative vision model wants. Reuse what's already there; do not re-author.

### 4.2 `dialogue.character_name` + `voice_traits`

Characters do **not** become faces (C6). Instead:

| Destination                            | Treatment                                             |
|----------------------------------------|-------------------------------------------------------|
| Named position in 3DGS (Mapping B)     | Place a named null / marker at a generated position; WorldNav can dolly to it |
| Mood tag for scene camera              | Map `voice_traits` first adjective -> camera style (see 4.5) |
| Character card sidecar file            | `chars.json` with `{name, traits}` for future use — kept but not yet consumed |

The constraint "no character faces" is not a limitation, it is the interesting aesthetic: audience hears the Commander, sees their empty chair. The chair is what we render.

### 4.3 `sfx.description`

| Destination                            | Treatment                                             |
|----------------------------------------|-------------------------------------------------------|
| Visual accent on timeline              | Flash / zoom / particle burst at `audio_offset_s`     |
| Tag into anchor prompt (Mapping A)     | Only the first scene-dominant SFX is prompted in; the rest are timed effects |
| Camera shake intensity (Mapping C)     | Density of SFX per 10 seconds -> shake amplitude      |

### 4.4 `scene_break` + per-scene timing

| Destination                            | Treatment                                             |
|----------------------------------------|-------------------------------------------------------|
| Chunk boundary in render pipeline      | One HyWorld scene per OTR scene                       |
| Crossfade duration                     | 0.75-1.5 s at scene boundaries (matches audio bed)    |
| Clip duration cap                      | Enforce C4: if a scene's audio exceeds 12 s, split into multiple shots |
| Max `shots[]` per scene (Director schema extension) | 4 (keeps VRAM predictable; tune after smoke test) |

### 4.5 `voice_traits` -> camera adjective (Mapping C detail)

| Trait keyword (first match, case-insensitive) | Camera treatment                       |
|-----------------------------------------------|----------------------------------------|
| weary, tired, old                             | slow handheld, close                   |
| angry, hostile, sharp                         | fast dolly, canted                     |
| calm, warm, gentle                            | locked off, wide                       |
| frantic, panicked                             | whip-pan, short focal length           |
| announcer, formal                             | clean push-in, centered                |
| child, young                                  | low angle, looking up                  |
| whisper, hushed                               | macro detail, shallow focus            |
| (no match)                                    | slow drift, medium lens                |

### 4.6 Beats and silence

`(beat)` and long gaps between dialogue tokens are narratively load-bearing. In an OTR radio drama, silence is the punchline. In HyWorld:

| Destination                            | Treatment                                             |
|----------------------------------------|-------------------------------------------------------|
| Each `(beat)` token                    | 200 ms camera hold                                    |
| Gap > 2 s between dialogue tokens      | Insert a beauty shot (env-only, no cut)               |
| Gap > 5 s                              | Consider title card / intertitle for drama            |

Silence earns its own frame.

---

## 5. How much of this does OTR *have* to do, vs. HyWorld?

Two approaches, same mapping:

### 5.1 Deterministic mapping in OTR (recommended for PoC)

A new pure-Python helper, `otr_v2/hyworld/shotlist.py`, reads `script_lines` and produces a `shots[]` array using the tables above. No LLM. No creativity. Pure lookup + timing math.

**Pros:** deterministic, testable, free of LLM hallucination, easy to diff across runs, byte-stable for the same input. Respects the "audio is king" rule trivially because shotlist generation is a separate, side-effect-free pass on already-written script text.

**Cons:** mappings stay literal. No surprise, no poetry.

### 5.2 LLM shotlist pass (later, when we want poetry)

A small Gemma / Mistral pass reads the parsed script + environment tokens and *writes* the `shots[]` array with camera / duration / mood as structured JSON. Guardrailed by schema validation.

**Pros:** poetic, surprising, handles genre shifts better.
**Cons:** another LLM phase, another VRAM event, non-deterministic. Not for PoC.

**Recommendation:** ship deterministic first. It's the floor. LLM shotlist becomes a toggle later.

---

## 6. Where this plugs into the existing Director schema

From the PoC doc we already reserved optional `shots[]` and `style_anchor_hash` fields. This narrative-mapping doc gives a filling for those fields:

```json
{
  "shots": [
    {
      "shot_id": "s01_01",
      "scene_ref": "1",
      "duration_sec": 9,
      "camera": "slow handheld, close",
      "env_prompt": "fluorescent hum, distant traffic, rain on concrete",
      "sfx_accents": [
        {"at": 3.2, "desc": "metal clatter"},
        {"at": 7.8, "desc": "door slam"}
      ],
      "dialogue_line_ids": ["line_14", "line_15"],
      "mood": "weary"
    }
  ],
  "style_anchor_hash": "a1b2c3d4e5f6"
}
```

All fields remain optional. A v1.7 run produces none of them and is byte-identical to baseline (C7 honored). A v2.0-alpha run with HyWorld enabled produces the full structure.

`env_prompt` is the literal `environment.description` for the scene the shot belongs to. `sfx_accents` carries offset-relative timings (relati
```


YOUR TASK:
Read the bundle. Critique the HyWorld pipeline (Bridge -> Poll -> Renderer + Worker sidecar) for robustness on a real Windows workstation. Specifically:

1. **Failure modes I'm not handling.** What can break that the current code does not catch? Be concrete: name the function and the case.
2. **Race conditions.** STATUS.json is written by the worker subprocess and read by the poll node. Where can they collide? Are the write/read patterns safe?
3. **Contract violations.** What can a malformed `script_lines` or `production_plan_json` do to the bridge / shotlist / renderer? Where do invariants leak?
4. **Process lifecycle.** The bridge spawns the worker fire-and-forget. What happens when the user cancels the ComfyUI workflow mid-run? When ComfyUI itself crashes? When the worker hangs? Are PIDs tracked correctly? Are zombies possible?
5. **Audio-byte-identical guarantee (C7).** Are there any code paths in renderer.py that could re-encode or modify the audio? Any subprocess invocations missing `-c:a copy`?
6. **Disk hygiene.** `io/hyworld_in/<job_id>/` and `io/hyworld_out/<job_id>/` accumulate forever. Is there a sweep? When should there be?
7. **Logging consistency.** Are log messages structured well enough to debug from a 12-minute episode log without re-running?

Be brutal but actionable. For each finding, propose the fix in one sentence. Output as a numbered list grouped by severity (Critical / Major / Minor). At the end, give a SCORECARD of overall robustness (1-10) with a one-line justification.

---

## Assistant reply

**Critical**

1. **Worker subprocess can become orphaned if ComfyUI is killed or workflow is cancelled.**  
   - In `bridge.py:HyworldBridge._spawn_sidecar`, the worker is spawned fire-and-forget, and only the PID is written to a file. There is no mechanism to kill the worker if the parent process (ComfyUI) is terminated, nor is there a sweep for stale PIDs.  
   **Fix:** Add a cleanup/sweep mechanism in the poll node or a dedicated sweeper script to kill orphaned workers based on PID liveness and job status.

2. **Race condition on STATUS.json read/write.**  
   - In `worker.py:_write_status` and `poll.py:HyworldPoll.execute`, the worker writes STATUS.json while the poll node reads it, but there is no atomic write (e.g., write to temp then rename), so partial reads and JSONDecodeError are possible.  
   **Fix:** Write STATUS.json to a temp file and atomically rename/move to STATUS.json to guarantee readers never see partial files.

3. **Malformed or malicious `script_json` can poison the contract and crash the worker.**  
   - In `bridge.py:HyworldBridge.execute`, script_json is only checked for being a list, but not for the expected token structure. In `shotlist.py:generate_shotlist`, the code assumes well-formed tokens (e.g., `token.get("type")`), which can throw if script_lines is not a list of dicts.  
   **Fix:** Add strict schema validation for `script_json` before passing to shotlist, and fail fast with a clear error status if invalid.

4. **No VRAM or resource coordination between worker and audio pipeline.**  
   - If the worker is upgraded to real WorldMirror inference, it may contend for GPU with the audio pipeline, violating C3/C4.  
   **Fix:** In the future, add a lock or coordination mechanism so sidecar video jobs never run inference while audio jobs hold the GPU.

5. **Audio file existence is only checked, not its integrity or type.**  
   - In `renderer.py:HyworldRenderer.execute`, the audio file is checked for existence but not for being a valid WAV or matching expected properties (channels, sample rate, etc.), risking mux errors or silent corruption.  
   **Fix:** Add a check for audio file type and properties before muxing.

---

**Major**

6. **No disk hygiene or job directory sweep.**  
   - `io/hyworld_in/` and `io/hyworld_out/` accumulate jobs forever; there is no cleanup of old runs, failed jobs, or orphaned directories.  
   **Fix:** Add a periodic sweep (manual or automated) to remove job dirs older than N days or jobs with terminal status.

7. **No explicit timeout or kill for stuck worker subprocesses.**  
   - If the worker hangs (e.g., buggy ffmpeg call), it can run forever, and the poll node will just time out, but the process remains alive.  
   **Fix:** On poll timeout, mark the job as timed out and attempt to kill the worker process using the PID file.

8. **No atomicity for other contract files.**  
   - In `bridge.py:HyworldBridge.execute`, contract files are written directly, so a crash mid-write can leave corrupt files for the worker.  
   **Fix:** Write contract files to temp files and rename atomically.

9. **No fallback if ffmpeg is missing in renderer.**  
   - If ffmpeg is missing, the renderer logs and returns an empty result, but the user may not see a clear error in the UI.  
   **Fix:** Surface a clear error status downstream so the workflow can route to OTR_SignalLostVideo and log the error visibly.

10. **Audio passthrough is correct, but ffmpeg errors are not always surfaced.**  
    - In `renderer.py:_run_ffmpeg`, ffmpeg errors are logged but not raised, so silent failures may go unnoticed unless the log is checked.  
    **Fix:** Raise an exception or return a failure status if ffmpeg fails, so the workflow can respond accordingly.

---

**Minor**

11. **Logging is inconsistent and not structured for easy parsing.**  
    - Some logs are plain strings, some are formatted, and timestamps are not always included.  
    **Fix:** Use a consistent log format (timestamp, job_id, node, level) for all log messages.

12. **No explicit check for sufficient disk space before writing large outputs.**  
    - Large jobs may fill the disk and fail mid-run.  
    **Fix:** Check available disk space before spawning the worker and before rendering video.

13. **No check for filename/path collisions in output directories.**  
    - If two jobs have the same episode_title and timestamp, output files may collide.  
    **Fix:** Ensure unique filenames or use job_id in output filenames.

14. **No check for invalid characters in episode_title used for filenames.**  
    - `renderer.py` attempts to sanitize, but Windows-forbidden characters could still slip through.  
    **Fix:** Use a stricter sanitizer for filenames.

15. **No explicit handling for ComfyUI shutdown signals.**  
    - If ComfyUI is closed gracefully, there is no hook to clean up running workers.  
    **Fix:** If possible, hook into ComfyUI shutdown to kill or mark workers as aborted.

---

## SCORECARD

**Robustness: 6/10**

**Justification:**  
The pipeline is well-structured and deterministic, with clear separation between audio and visual paths, and good fallback logic. However, it lacks critical safeguards for process lifecycle (orphaned/hung workers), atomic file writes (race on STATUS.json), and disk hygiene (job dirs accumulate forever). Logging is adequate for manual review but not for automated triage. The audio-byte-identical guarantee is mostly honored but could be broken by silent ffmpeg failures or malformed audio files. With a few targeted fixes, robustness could quickly reach 8/10.
