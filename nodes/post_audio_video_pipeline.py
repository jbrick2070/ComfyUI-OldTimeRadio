"""
post_audio_video_pipeline.py  --  OTR_PostAudioVideoPipeline ComfyUI node
========================================================================

Thin subprocess kickoff that wires the FULL workflow's audio output to
the post-audio video pipeline (FLUX composites + HuMo lip-sync batch +
final episode concat).

Why a subprocess and not in-graph nodes?
    Per CLAUDE.md C3: "All visual generation in subprocesses via
    multiprocessing.get_context('spawn')." HuMo + FLUX in-graph already
    OOM'd on 16 GB during the v2.0-visual-engine attempt; the sidecar
    pattern (`scripts/render_humo_batch.py` + `render_episode_concat.py`)
    keeps each phase in its own process so VRAM pressure resets between
    stages. This node is the trigger that lets the existing audio
    workflow JSON kick off the post-audio pipeline without a manual
    PowerShell paste.

Why fire-and-forget?
    The post-audio pipeline takes ~2h+ for a 31-line episode at 7 s
    clip-length (per BUG-LOCAL-074 / 075 stack). If this node blocked
    the queue worker until the wrapper finished, ComfyUI couldn't
    accept any other prompts during that time. Instead the node
    returns immediately with the launched PID + log path, and the
    wrapper runs in the background. The orchestrator inside the
    wrapper submits HuMo prompts to the same Comfy server one at a
    time -- ComfyUI happily processes them as the next queue items.

Inputs:
    audio_or_ledger_path  STRING  Path to either the audio episode mp4
                                  produced by SignalLostVideo /
                                  EpisodeAssembler, or directly to the
                                  ``*_ledger.json`` for that episode.
                                  ``.mp4`` -> ledger inferred by suffix
                                  swap. ``.json`` -> used as-is. A
                                  directory -> newest ``*_ledger.json``
                                  inside.
    clip_length           FLOAT   Base HuMo clip duration in seconds.
                                  Default 7.0 -- HuMo's empirical sweet
                                  spot at 175 frames @ 25 fps (clamped
                                  to 177 by Wan 2.1 VAE 4n+1 rule =
                                  7.08 s actual).
    max_clips             INT     0 = render every line; otherwise cap.
    portrait_map          STRING  Optional ``"NAME=NN,NAME=NN"`` mapping
                                  cast names to still indices. Empty ->
                                  use Pass1 portraits already on disk.
    skip_pass1            BOOL    Default True. The audio path doesn't
                                  trigger FLUX renders, so Pass1
                                  portraits must come from a prior
                                  smoke run or the portrait_map
                                  fallback.
    mode                  CHOICE  ``fire`` (default, async) /
                                  ``dry-run`` (validate args, no
                                  subprocess) / ``wait`` (block until
                                  the wrapper exits -- only use when
                                  you know you can afford the wait,
                                  e.g. testing).

Outputs:
    status     STRING  Human-readable single-line summary. Example:
                       ``"launched pid=58736 log=...\\post_audio.log"``
    log_path   STRING  Full path to the wrapper's combined stdout/stderr
                       log -- tail this from a separate shell to watch
                       progress without blocking the graph.

Notes:
    - Never raises. Validation failures return a status string
      describing the problem; downstream nodes can branch on the
      ``status.startswith("error:")`` convention.
    - Subprocess is launched with ``CREATE_NEW_PROCESS_GROUP`` on
      Windows so it survives ComfyUI restarts.
    - The wrapper script handles its own PowerShell/cmd invocation.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("OTR.post_audio_video_pipeline")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

import sys as _sys

# Path-helper bootstrap (see ``nodes/batch_humo_render.py`` for the
# full rationale). Sibling-module ``_otr_paths`` must be reachable
# both via the parent ``nodes`` package (production) and via direct
# file load (test fixtures using ``spec_from_file_location``).
_NODES_DIR = Path(__file__).resolve().parent
if str(_NODES_DIR) not in _sys.path:
    _sys.path.insert(0, str(_NODES_DIR))
from _otr_paths import otr_episodes_root  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WRAPPER_PATH = _REPO_ROOT / "scripts" / "test_humo_batch_concat.ps1"
_LOG_DIR = _REPO_ROOT / "scripts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_ledger_from_input(path_str: str) -> Path | None:
    """Map a flexible input path to a ledger JSON.

    Accepts:
      - ``.json``  -> used as-is
      - ``.mp4``   -> swap suffix to ``_ledger.json`` (matches the
                      EpisodeAssembler / SignalLostVideo convention)
      - directory  -> newest ``*_ledger.json`` (excluding ``pending_*``)
                      inside; mirrors the wrapper's auto-pick logic
      - empty str  -> auto-pick across the canonical audio dirs
    """
    if not path_str or not path_str.strip():
        # Fall back to scanning the canonical per-episode workspace.
        # S26-B6 cleanbreak: dropped the legacy flat-layout scan path
        # (this node is retired per __init__.py and the flat layout is
        # extinct in the v2 workspace).
        candidates: list[Path] = []
        d = otr_episodes_root()
        if d.exists():
            # Per-episode tree: <root>/<ep>/audio/*_ledger.json
            candidates.extend(
                p for p in d.glob("*/audio/*_ledger.json")
                if not p.name.startswith("pending_")
            )
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    p = Path(path_str.strip())
    if p.is_dir():
        candidates = [
            x for x in p.glob("*_ledger.json")
            if not x.name.startswith("pending_")
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x.stat().st_mtime)

    suffix = p.suffix.lower()
    if suffix == ".json":
        return p if p.exists() else None
    if suffix == ".mp4":
        # Swap .mp4 -> _ledger.json (the canonical pairing convention).
        ledger = p.with_suffix("").parent / f"{p.stem}_ledger.json"
        return ledger if ledger.exists() else None

    # Last-ditch: try the path as-is (might be a ledger without .json
    # suffix in some edge-case workflow).
    return p if p.exists() else None


def _build_wrapper_command(
    *,
    ledger: Path,
    clip_length: float,
    max_clips: int,
    portrait_map: str,
    skip_pass1: bool,
) -> list[str]:
    """Build the PowerShell command line for the wrapper script.

    PowerShell is required (not cmd) because the wrapper itself is a
    .ps1. The orchestrator INSIDE the wrapper still uses python, so
    git operations are safe -- nothing is invoked through PowerShell
    that touches git.
    """
    cmd: list[str] = [
        "powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", str(_WRAPPER_PATH),
        "-Ledger", str(ledger),
        "-AutoSlice", f"{float(clip_length):.4f}",
        "-MaxClips", str(int(max_clips)),
    ]
    if portrait_map and portrait_map.strip():
        cmd.extend(["-PortraitMap", portrait_map.strip()])
    if skip_pass1:
        cmd.append("-SkipPass1")
    return cmd


def _spawn_wrapper(
    cmd: list[str],
    log_path: Path,
) -> int:
    """Launch the wrapper in a detached child process.

    On Windows we use CREATE_NEW_PROCESS_GROUP so the wrapper survives
    ComfyUI shutdown / restart cycles. stdout + stderr are combined and
    written to log_path so the user can tail it from any shell without
    needing to attach to the orchestrator's console.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    creationflags = 0
    if sys.platform == "win32":
        # 0x00000200 = CREATE_NEW_PROCESS_GROUP
        creationflags = 0x00000200
    log_handle = open(log_path, "wb")
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(_REPO_ROOT),
            creationflags=creationflags,
            close_fds=True,
        )
    finally:
        # Don't close the log handle -- the child inherited it. Closing
        # it here is fine because Popen already dup'd the FD into the
        # child; our handle just ensures the file exists with the right
        # permissions.
        try:
            log_handle.close()
        except OSError:
            pass
    return int(proc.pid)


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class PostAudioVideoPipeline:
    """Trigger node that fires the post-audio video pipeline.

    Wire this into the FULL workflow downstream of
    OTR_SignalLostVideo / OTR_EpisodeAssembler so the HuMo batch +
    concat run automatically after audio is delivered.

    OUTPUT_NODE = True is LOAD-BEARING (BUG-LOCAL-077):
        ComfyUI's executor traces backward from OUTPUT_NODE-marked
        nodes to determine which nodes to actually run. A node whose
        outputs feed nothing downstream AND which is not flagged as
        OUTPUT_NODE gets pruned silently from the execution graph.
        This node is a side-effect-only trigger -- its outputs are
        diagnostic strings the user reads in the ComfyUI console, not
        wired to consumer nodes -- so it MUST self-declare as an
        output node or it never fires. Discovered when the FULL
        workflow ran 30 min (audio + FLUX) at 2026-04-27 09:54 and
        node 44 was missing from the prompt's outputs dict despite
        being in the graph.
    """

    CATEGORY = "OTR/v2/Pipeline"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("status", "log_path")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_or_ledger_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Path to the audio episode mp4 OR the ledger "
                        "JSON OR the directory containing both. Empty "
                        "string auto-picks the newest non-pending "
                        "ledger across the canonical audio dirs."
                    ),
                }),
            },
            "optional": {
                "clip_length": ("FLOAT", {
                    "default": 7.0,
                    "min": 1.32,
                    "max": 7.08,
                    "step": 0.04,
                    "tooltip": (
                        "HuMo clip duration in seconds. Default 7.0 "
                        "lands at 175 frames -> 177 (clamped by Wan "
                        "2.1 VAE 4n+1 rule = 7.08 s actual). Going "
                        "higher than 7.08 requires bumping "
                        "HUMO_MAX_FRAMES in render_humo_batch.py "
                        "after empirical GPU validation."
                    ),
                }),
                "max_clips": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "tooltip": (
                        "0 = render every line in the ledger (full "
                        "episode). Set 4-8 for smoke tests."
                    ),
                }),
                "portrait_map": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        'Optional cast-name -> still-index map: '
                        '"ANNOUNCER=17,LEV=29,TODD=28". Empty = use '
                        "Pass1 portraits already in "
                        "output/otr/portraits/<episode_id>/."
                    ),
                }),
                "skip_pass1": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Skip Pass1 portrait re-render. Default True "
                        "because portraits are usually already on "
                        "disk from a prior smoke run; disable only "
                        "when you want to regenerate them."
                    ),
                }),
                "mode": (["fire", "dry-run", "wait"], {
                    "default": "fire",
                    "tooltip": (
                        "fire = launch wrapper async, return PID "
                        "immediately (default; releases the queue). "
                        "dry-run = validate args without launching. "
                        "wait = block until wrapper exits (only use "
                        "for short test runs -- a 31-line episode "
                        "will hold the queue for ~2 hours)."
                    ),
                }),
            },
        }

    def execute(
        self,
        audio_or_ledger_path: str,
        clip_length: float = 7.0,
        max_clips: int = 0,
        portrait_map: str = "",
        skip_pass1: bool = True,
        mode: str = "fire",
    ):
        # 1. Resolve ledger
        ledger = _resolve_ledger_from_input(audio_or_ledger_path)
        if ledger is None or not ledger.exists():
            msg = f"error: cannot resolve ledger from input={audio_or_ledger_path!r}"
            log.warning("[PostAudioVideoPipeline] %s", msg)
            return (msg, "")

        # 2. Validate wrapper exists
        if not _WRAPPER_PATH.exists():
            msg = f"error: wrapper script not found at {_WRAPPER_PATH}"
            log.warning("[PostAudioVideoPipeline] %s", msg)
            return (msg, "")

        # 3. Build command
        cmd = _build_wrapper_command(
            ledger=ledger,
            clip_length=clip_length,
            max_clips=max_clips,
            portrait_map=portrait_map,
            skip_pass1=skip_pass1,
        )

        # 4. Compute log path
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = _LOG_DIR / f"post_audio_pipeline_{ts}.log"

        # 5. Mode dispatch
        if mode == "dry-run":
            status = (
                f"dry-run: ledger={ledger.name} "
                f"clip_length={clip_length} max_clips={max_clips} "
                f"portrait_map={portrait_map!r} skip_pass1={skip_pass1} "
                f"cmd={' '.join(cmd)}"
            )
            log.info("[PostAudioVideoPipeline] %s", status)
            return (status, str(log_path))

        if mode == "wait":
            # Synchronous run -- pipe output to the log file AND echo
            # to ComfyUI's console for live observability.
            log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(log_path, "wb") as lh:
                    proc = subprocess.Popen(
                        cmd,
                        stdin=subprocess.DEVNULL,
                        stdout=lh,
                        stderr=subprocess.STDOUT,
                        cwd=str(_REPO_ROOT),
                    )
                    rc = proc.wait()
                status = (
                    f"wait: rc={rc} ledger={ledger.name} "
                    f"log={log_path}"
                )
            except Exception as exc:  # noqa: BLE001
                status = f"error: wait-mode launch failed: {exc}"
            log.info("[PostAudioVideoPipeline] %s", status)
            return (status, str(log_path))

        # mode == "fire" (default): async launch, return immediately
        try:
            pid = _spawn_wrapper(cmd, log_path)
            status = (
                f"launched pid={pid} ledger={ledger.name} "
                f"clip_length={clip_length} max_clips={max_clips} "
                f"log={log_path}"
            )
            log.info("[PostAudioVideoPipeline] %s", status)
        except Exception as exc:  # noqa: BLE001
            status = f"error: fire-mode launch failed: {exc}"
            log.warning("[PostAudioVideoPipeline] %s", status)

        return (status, str(log_path))


__all__ = ["PostAudioVideoPipeline"]
