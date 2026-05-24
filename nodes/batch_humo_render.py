"""
batch_humo_render.py  --  OTR_BatchHumoRender ComfyUI node
==========================================================

Render N HuMo lip-sync clips in lockstep from a Production Ledger,
in a single graph run, sharing one HuMo + Lora + CLIP + VAE +
Whisper load.

This is the in-graph counterpart to ``scripts/render_humo_batch.py``
(the subprocess orchestrator). The CLI script remains for ad-hoc
smoke tests; production runs use this node so progress is visible
in ComfyUI's UI and the FULL workflow is a single self-contained
JSON with no hidden subprocess.

Pattern mirrors OTR_BatchFluxRender:
  - take pre-loaded MODEL/CLIP/VAE/AUDIO_ENCODER as INPUT_TYPES so
    Comfy's model-management owns the lifecycle
  - load core ComfyUI nodes lazily at execute() time
  - loop N ledger lines, calling node objects directly (NOT via
    /prompt HTTP API)
  - each line: slice audio tensor, load portrait from disk, encode
    text + audio + ref_image, run WanHuMoImageToVideo + KSampler +
    VAEDecode + CreateVideo + SaveVideo, write ``<line_id>.mp4``
  - return clips_dir + count + report

Per CLAUDE.md C5: HuMo runs at fp8_e4m3fn weights staged via
ComfyUI's dynamic VRAM loading. UnloadAll wires after this node to
free HuMo before VideoComposite reads the proc gen + clips.

Ported logic from render_humo_batch.py:
  - HUMO_FPS / HUMO_MIN_FRAMES / HUMO_MAX_FRAMES constants
  - humo_length_for_dur(): 4n+1 frame snap with floor + ceiling
  - find_portrait_for_speaker(): cast-position fallback
  - find_composite_for_shot_speaker(): per-(shot,speaker) pass3
    composite picker
  - cid_to_name speaker resolution (BUG-LOCAL-074)
  - Save filename = ``<line_id>.mp4`` so OTR_VideoComposite (or
    render_episode_concat.py) finds clips by ledger line_id
"""
from __future__ import annotations

import json
import logging
import math
import re
import sys as _sys
import time
from pathlib import Path
from typing import Any

# Path-helper bootstrap: ``_otr_paths`` is a sibling module. In
# production ComfyUI imports this file as ``nodes.batch_humo_render``
# so a relative import works. In tests + ad-hoc scripts the module is
# loaded directly via ``importlib.util.spec_from_file_location`` with
# no parent package, breaking relative imports. The sys.path prepend
# below makes ``_otr_paths`` reachable as a top-level module in both
# contexts; the import is idempotent across multiple node loads.
_NODES_DIR = Path(__file__).resolve().parent
if str(_NODES_DIR) not in _sys.path:
    _sys.path.insert(0, str(_NODES_DIR))
from _otr_paths import (  # noqa: E402
    comfy_output_dir,
    otr_audio_dir,
    otr_episodes_root,
    # S28 cleanbreak: dropped otr_legacy_audio_dir.
    otr_stills_dir,
    otr_videos_dir,
)
from _otr_speaker_role import (  # noqa: E402
    SPEAKER_ROLE_CHARACTER,
    is_never_humo_role,
    is_radio_role,
    resolve_speaker_role,
)

log = logging.getLogger("OTR.batch_humo_render")


_MIN_RADIO_STILL_BYTES = 256  # smaller than this is almost certainly truncated


# BUG-LOCAL-126 (2026-05-08): the overnight FULL acceptance soak crashed
# with `Fatal Python error: Aborted` after two prior CAUGHT HuMo OOMs.
# The PyTorch CUDA allocator pool drifted to ~24.5 GiB on a 15.92 GiB
# device after the recovery cycles -- pool fragmentation accumulating
# faster than `empty_cache()` could reclaim. The two helpers below are
# the alarm-plumbing fix (per `feedback_no_vram_dragons`'s carve-out --
# this is a soak-survival fault, not a chase-the-dragon optimization).


def _single_exception_is_oom(exc: BaseException) -> bool:
    """True if a SINGLE exception object is a CUDA OOM in any of the
    common Pythonic forms.

    PyTorch raises ``torch.OutOfMemoryError`` (subclass of
    ``RuntimeError``) on modern versions; older / forked builds still
    raise plain ``RuntimeError`` with ``"out of memory"`` in the
    message. Both forms are observed in OTR's overnight log.
    """
    try:
        import torch  # type: ignore
        oom_cls = getattr(torch, "OutOfMemoryError", None) or getattr(
            getattr(torch, "cuda", None), "OutOfMemoryError", None
        )
    except Exception:  # noqa: BLE001
        oom_cls = None
    if oom_cls is not None and isinstance(exc, oom_cls):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "out of memory" in msg or "outofmemoryerror" in msg:
            return True
    return False


def _is_oom_exception(exc: BaseException) -> bool:
    """True if ``exc`` -- OR any exception in its ``__cause__`` /
    ``__context__`` chain -- is a CUDA OOM.

    Round-robin Item D catch (2026-05-08): ComfyUI / wrapping code
    can re-raise a higher-level exception whose top-level message
    lacks "out of memory" while still chaining the original OOM
    via ``raise X from oom_err`` (sets ``__cause__``) or implicit
    ``__context__``. Walk the chain so the recovery path doesn't
    miss those wrapped cases. The ``seen`` set guards against
    pathological cycles in the chain.
    """
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        cur = stack.pop()
        cur_id = id(cur)
        if cur_id in seen:
            continue
        seen.add(cur_id)
        if _single_exception_is_oom(cur):
            return True
        cause = getattr(cur, "__cause__", None)
        if cause is not None and isinstance(cause, BaseException):
            stack.append(cause)
        ctx = getattr(cur, "__context__", None)
        if ctx is not None and isinstance(ctx, BaseException):
            stack.append(ctx)
    return False


# Module-level reference for ledger telemetry. Set by execute() so
# the helper can stamp `meta.cuda_hard_reset_count` from the same
# process state. Not threaded through args because the helper is
# called via try/except in code paths where threading kwargs would
# add noise to recovery code.
_CURRENT_LEDGER_REF: dict | None = None


def _hard_reset_cuda_context() -> None:
    """Soft CUDA cleanup chain after a CAUGHT OOM.

    NOTE: the name says "hard reset" but this is a *soft* cleanup
    chain, not a true CUDA context destroy/recreate (which would
    require `cudaDeviceReset` and a full re-init of every torch
    handle, including model weights -- not feasible mid-run).
    The 2026-05-08 round-robin code review correctly flagged the
    naming as overstated; kept the name to avoid churn across
    BUG_LOG / log strings, but the docstring is the source of
    truth.

    What this DOES do: chains
        mm.unload_all_models()
        gc.collect()
        mm.soft_empty_cache(force=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  -- no-op on Windows; harmless
    in that order, so the model patches release VRAM FIRST and
    the allocator's free-block coalesce runs against actually-free
    pages. Mirrors BUG-LOCAL-050's chained-backend teardown pattern.

    What this does NOT do: drop the PyTorch CUDA context, force
    the kernel to release VA mappings, or reset the GPU driver.
    If the underlying allocator drift is caused by retained
    references that live beyond the model patcher (custom node
    globals, tracebacks, weakly-rooted modules), this chain runs
    OK but VRAM stays high. The 2026-05-08 round-robin Element 2
    flagged this honestly.

    Best-effort: every step is wrapped so a missing API on an older
    torch/Comfy combo doesn't escalate the recovery into a second
    crash. Logs at INFO (success) / WARNING (any step skipped) so
    BUG-126 telemetry is greppable.

    MUST be called from OUTSIDE the active `except` block per the
    Element 3 traceback-retention catch -- otherwise the failed
    frame's f_locals are still pinned by `exc.__traceback__` and
    gc.collect can't clear them.
    """
    import gc

    steps_run: list[str] = []
    steps_failed: list[str] = []

    try:
        import comfy.model_management as mm  # type: ignore
    except Exception as exc:  # noqa: BLE001
        steps_failed.append(f"import comfy.model_management ({exc})")
        mm = None  # type: ignore[assignment]

    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        steps_failed.append(f"import torch ({exc})")
        log.warning("[BatchHumoRender] cuda hard-reset SKIPPED: %s", "; ".join(steps_failed))
        return

    # Round-robin Item H telemetry: capture before / after allocator
    # state so post-mortem can answer "did the cleanup chain actually
    # reclaim VRAM, or did it just run cleanly while the pool stayed
    # fragmented?" -- the single most load-bearing question per the
    # synthesis doc. Best-effort; failure here doesn't block cleanup.
    before_allocated_gib = float("nan")
    before_reserved_gib = float("nan")
    try:
        if torch.cuda.is_available():
            before_allocated_gib = torch.cuda.memory_allocated() / (1024 ** 3)
            before_reserved_gib = torch.cuda.memory_reserved() / (1024 ** 3)
    except Exception:  # noqa: BLE001
        pass

    if mm is not None:
        try:
            mm.unload_all_models()
            steps_run.append("unload_all_models")
        except Exception as exc:  # noqa: BLE001
            steps_failed.append(f"unload_all_models ({exc})")

    try:
        gc.collect()
        steps_run.append("gc.collect")
    except Exception as exc:  # noqa: BLE001
        steps_failed.append(f"gc.collect ({exc})")

    if mm is not None:
        try:
            mm.soft_empty_cache(force=True)
            steps_run.append("mm.soft_empty_cache")
        except Exception as exc:  # noqa: BLE001
            steps_failed.append(f"mm.soft_empty_cache ({exc})")

    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            steps_run.append("torch.cuda.synchronize")
    except Exception as exc:  # noqa: BLE001
        steps_failed.append(f"torch.cuda.synchronize ({exc})")

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            steps_run.append("torch.cuda.empty_cache")
    except Exception as exc:  # noqa: BLE001
        steps_failed.append(f"torch.cuda.empty_cache ({exc})")

    try:
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            steps_run.append("torch.cuda.ipc_collect")
    except Exception as exc:  # noqa: BLE001
        steps_failed.append(f"torch.cuda.ipc_collect ({exc})")

    # Round-robin Item H: capture after-state and log the reclaim
    # delta. This single line is what tells us whether the cleanup
    # chain is doing real work or just running in place.
    after_allocated_gib = float("nan")
    after_reserved_gib = float("nan")
    try:
        if torch.cuda.is_available():
            after_allocated_gib = torch.cuda.memory_allocated() / (1024 ** 3)
            after_reserved_gib = torch.cuda.memory_reserved() / (1024 ** 3)
    except Exception:  # noqa: BLE001
        pass

    if steps_failed:
        log.warning(
            "[BatchHumoRender] cuda hard-reset partial: ran=%s; failed=%s; "
            "allocated %.2f -> %.2f GiB, reserved %.2f -> %.2f GiB",
            steps_run, steps_failed,
            before_allocated_gib, after_allocated_gib,
            before_reserved_gib, after_reserved_gib,
        )
    else:
        log.info(
            "[BatchHumoRender] BUG-LOCAL-126 cuda hard-reset OK: %s; "
            "allocated %.2f -> %.2f GiB, reserved %.2f -> %.2f GiB",
            ", ".join(steps_run),
            before_allocated_gib, after_allocated_gib,
            before_reserved_gib, after_reserved_gib,
        )

    # l3-2026-05-08 telemetry: bump meta.cuda_hard_reset_count so the
    # ledger durably records that the cleanup chain fired in this
    # process, even if the ComfyUI log later rotates or vanishes.
    if _CURRENT_LEDGER_REF is not None:
        try:
            from . import _otr_ledger as _OTRL_HR  # type: ignore
        except ImportError:
            try:
                import _otr_ledger as _OTRL_HR  # type: ignore
            except Exception:  # noqa: BLE001
                _OTRL_HR = None  # type: ignore[assignment]
        if _OTRL_HR is not None:
            try:
                _OTRL_HR.bump_meta_cuda_hard_reset_count(_CURRENT_LEDGER_REF)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[BatchHumoRender] cuda_hard_reset_count bump failed: %s", exc,
                )


class HumoSoakCapReached(Exception):
    """Raised by the soak loop when ``humo_max_lines_per_process`` is
    set and that many FRESH HuMo lines (excluding resumed clips)
    have rendered successfully in the current ComfyUI process.

    Catching this in a workflow / orchestrator lets the caller
    cleanly stop, persist the ledger, and queue a follow-up run
    that picks up via ``resume_from_ledger=True``.

    Inherits from ``Exception`` (NOT ``RuntimeError``) per 2026-05-08
    external round-robin synthesis Item G: a downstream
    ``except RuntimeError`` clause must NOT silently swallow this
    structured soft-stop signal. Callers that genuinely want to
    handle the cap should catch ``HumoSoakCapReached`` by name.
    """

    def __init__(self, lines_completed: int, cap: int):
        self.lines_completed = lines_completed
        self.cap = cap
        super().__init__(
            f"HuMo soak cap reached: rendered {lines_completed} of cap {cap}; "
            "queue a follow-up run with resume_from_ledger=True to continue."
        )


def _is_valid_image_file(p: Path) -> bool:
    """``p`` exists, is a file, and has plausible byte size for a PNG.

    Defends against the 0-byte / truncated-file trap (BUG-LOCAL-121
    hardening 2026-05-01): ``Path.exists()`` and ``Path.is_file()``
    both return ``True`` for empty files left behind by an
    interrupted FLUX render. Downstream ``Image.open()`` then crashes
    the run instead of falling through to the portrait resolver.
    """
    try:
        if not p.is_file():
            return False
        size = p.stat().st_size
    except Exception:  # noqa: BLE001
        return False
    return size >= _MIN_RADIO_STILL_BYTES


def _safe_episode_id(eid) -> str | None:
    """Coerce + sanitize ``ledger.episode_id`` into a safe filename
    component, or return ``None`` if it can't be made safe.

    Defends against (BUG-LOCAL-121 hardening 2026-05-01):
      - Non-string types (None, int, dict, etc.)
      - Empty / whitespace-only strings
      - Path-traversal characters (``/`` ``\\`` ``..``)
      - Embedded NULs
    """
    if not isinstance(eid, str):
        return None
    s = eid.strip()
    if not s:
        return None
    # Reject anything that could escape the stills dir or cause
    # filesystem-API surprises.
    forbidden = ("/", "\\", "..", "\x00")
    for token in forbidden:
        if token in s:
            return None
    return s


def _resolve_radio_still_path(ledger):
    """Pull the radio still path from the ledger, in either schema
    location (top-level or under meta).  Returns ``Path`` if it
    points at an existing file, else ``None``.

    Resolution order (BUG-LOCAL-121 fix 2026-04-30, hardened 2026-05-01):
      1. ``ledger.radio_bookend_path`` (top-level)
      2. ``ledger.meta.radio_bookend_path``
      3. **Filesystem fallback:** ``output/otr/stills/radio_bookend_<episode_id>.png``
         derived from ``ledger.episode_id``.  This handles the case where
         BatchFluxRender successfully rendered the radio bookend image
         but the ledger stamp failed silently (e.g. ledger overwritten by
         a later phase, atomic write race, etc.).

    All three checks use ``_is_valid_image_file`` (existence + non-zero
    byte-size) so a stamped path pointing at a 0-byte / truncated PNG
    falls through to the next layer rather than crashing
    ``Image.open()`` downstream.
    """
    if not isinstance(ledger, dict):
        return None
    # Layer 1: top-level ledger path
    cand = ledger.get("radio_bookend_path")
    # Layer 2: meta ledger path
    if not cand:
        meta = ledger.get("meta") or {}
        if isinstance(meta, dict):
            cand = meta.get("radio_bookend_path")
    if cand:
        try:
            p = Path(cand)
        except Exception:  # noqa: BLE001
            p = None
        if p is not None and _is_valid_image_file(p):
            return p
        if p is not None:
            log.warning(
                "[BatchHumoRender] ledger radio_bookend_path=%s does "
                "not exist or is truncated; trying filesystem fallback",
                p,
            )
    # Layer 3: filesystem fallback by episode_id (BUG-LOCAL-121).
    # Even if ledger stamp is missing/stale, the FLUX phase saves
    # the radio bookend at a deterministic path we can reconstruct.
    safe_eid = _safe_episode_id(ledger.get("episode_id"))
    if safe_eid is None:
        return None
    try:
        stills_dir = otr_stills_dir(safe_eid)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[BatchHumoRender] otr_stills_dir(%r) failed; cannot run "
            "filesystem fallback: %s",
            safe_eid,
            exc,
        )
        return None
    try:
        fs_path = stills_dir / f"radio_bookend_{safe_eid}.png"
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[BatchHumoRender] radio still filesystem fallback "
            "path-build failed: %s",
            exc,
        )
        return None
    if _is_valid_image_file(fs_path):
        log.info(
            "[BatchHumoRender] radio still resolved via filesystem "
            "fallback (ledger stamp missing): %s",
            fs_path,
        )
        return fs_path
    return None


# ---------------------------------------------------------------------------
# HuMo constants (ported from render_humo_batch.py)
# ---------------------------------------------------------------------------

HUMO_FPS = 25
HUMO_MIN_FRAMES = 33   # smaller frame counts have hung this hardware
# BUG-LOCAL-086 (2026-05-04): bumped from 177 (~7.08s) to 353 (~14.12s) to
# cover normal character dialogue (Bark lines run 7-14s on the RTX 5080 stack).
# Lines that still exceed this cap (one-person plays, long monologues) fall
# into the chunking path below. If 353 OOMs at 1472x832 + Wan 2.1 14B on the
# 16 GB Blackwell, drop this back to 257 (10.28s) or 177 (7.08s) -- the
# chunking dispatch handles whatever the cap is.
HUMO_MAX_FRAMES = 353
# BUG-LOCAL-086: when an audio line exceeds HUMO_MAX_FRAMES, split it into
# consecutive chunks of this many frames each, render each chunk against the
# same portrait, then ffmpeg-concat into a single per-line mp4. 177 frames
# (7.08s) per chunk is the historically-stable HuMo render size; matches the
# pre-bump HUMO_MAX_FRAMES for safety.
HUMO_CHUNK_FRAMES = 177

# ByteDance Chinese negative prompt -- empirically the best HuMo neg
# on the official template.
_CHINESE_NEGATIVE = (
    "色调艳丽，过曝，静态，"
    "细节模糊不清，字幕，风格"
    "，作品，画作，画面，静止"
    "，整体发灰，最差质量，低"
    "质量，JPEG压缩残留，丑陋的"
    "，残缺的，多余的手指，画"
    "得不好的手部，画得不好的"
    "脸部，畸形的，毁容的，形"
    "态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背"
    "景，三条腿，背景人很多，"
    "倒着走"
)

_DEFAULT_POS_SUFFIX = (
    "dimly lit interior, ambient cinematic lighting, "
    "35mm film grain, shallow depth of field"
)

# Slug rule must stay in lockstep with render_flux_batch.slugify so
# pass3 composite filenames written by FLUX get found by HuMo.
_SLUG_RE_CONSUME = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def humo_length_for_dur(dur_s: float, *, fps: int = HUMO_FPS) -> int:
    """Pick the smallest valid HuMo `length` >= round(dur_s * fps).

    Wan 2.1 VAE temporal compression requires length = 4n + 1.
    Floored at HUMO_MIN_FRAMES. Capped at HUMO_MAX_FRAMES.
    """
    target = max(1, round(float(dur_s) * fps))
    n = (target - 1 + 3) // 4
    frames = 4 * n + 1
    if frames < HUMO_MIN_FRAMES:
        frames = HUMO_MIN_FRAMES
    if frames > HUMO_MAX_FRAMES:
        frames = HUMO_MAX_FRAMES
    return frames


def humo_length_for_dur_uncapped(dur_s: float, *, fps: int = HUMO_FPS) -> int:
    """Same as humo_length_for_dur but without the HUMO_MAX_FRAMES cap.

    BUG-LOCAL-086: used by the chunking dispatch to decide whether a
    line's audio duration exceeds the per-clip frame ceiling. If the
    uncapped value > HUMO_MAX_FRAMES, the line is split into multiple
    chunks before rendering.
    """
    target = max(1, round(float(dur_s) * fps))
    n = (target - 1 + 3) // 4
    frames = 4 * n + 1
    if frames < HUMO_MIN_FRAMES:
        frames = HUMO_MIN_FRAMES
    return frames


def _composite_slug(s: str, limit: int = 40) -> str:
    s = (s or "").strip().lower()
    s = _SLUG_RE_CONSUME.sub("_", s).strip("_")
    return (s[:limit] or "unnamed")


def _find_portrait(
    speaker: str,
    cast: list[dict],
    portraits_dir: Path,
) -> Path | None:
    """Find a portrait image for a speaker.

    BUG-LOCAL-093 (2026-05-04 EVENING): the FLUX env-still fallback
    tiers (formerly tiers 4-5) were REMOVED. They produced visibly
    wrong faces when triggered -- HuMo would lipsync against an
    environment scene that happened to contain a different person --
    and were a workaround from before BUG-LOCAL-078's per-cast
    portrait pass existed. With BUG-078 reliably writing
    ``portraits/c0X_portrait.png`` and stamping ``cast[i].portrait_path``,
    real character portraits are always available; if a portrait is
    missing, the correct behavior is to SKIP that line and let
    VideoComposite cover it with the BUG-129a static-radio fill --
    same as music/sfx -- rather than synthesize a wrong face.

    Strategy:

      1. cast[].portrait_path if populated and exists (BUG-078 canonical)
      2. otr_humo_pass1_portrait_*.png by index in cast list
         (legacy pass1 naming, kept for old episodes that pre-date BUG-078)
      3. any otr_humo_pass1_portrait_*.png
      4. None -- caller skips the line; VideoComposite fills with static radio

    BUG-LOCAL-087 ordering: every glob result is sorted by mtime
    descending (newest first). Without this, a freshly-rendered
    pass1_portrait_00059_.png would lose to a days-old pass1_portrait_00001_.png
    under default alphabetic sort.
    """
    speaker_norm = (speaker or "").upper().strip()

    # 1. Direct path from ledger cast entry (BUG-LOCAL-078 canonical)
    for c in cast:
        if (c.get("name") or "").upper().strip() == speaker_norm:
            p = c.get("portrait_path")
            if p and Path(p).exists():
                return Path(p)
            # BUG-LOCAL-235 fix (2026-05-19): Tier 1.5 deterministic
            # filename fallback. If the cast row's `portrait_path`
            # stamp is missing OR points to a stale path (e.g. the
            # BUG-LOCAL-234 rename-stale chain), fall back to looking
            # up `{char_id}_portrait.png` directly in portraits_dir.
            # PortraitRender's BUG-LOCAL-078 naming contract guarantees
            # this filename for every rendered cast portrait. The
            # legacy Tier 2 `pass1_portrait_*.png` glob below was
            # never updated when PortraitRender renamed its output
            # convention 2026-05-04, so this Tier 1.5 layer is what
            # actually closes the naming-contract drift.
            char_id = (c.get("char_id") or "").strip()
            if char_id:
                candidate = portraits_dir / f"{char_id}_portrait.png"
                if candidate.exists():
                    return candidate

    cast_names = [(c.get("name") or "").upper().strip() for c in cast]

    # 2. PASS1 humo portraits indexed by cast position (legacy pre-BUG-078 naming)
    if speaker_norm in cast_names:
        idx = cast_names.index(speaker_norm)
        candidates = sorted(
            list(portraits_dir.glob("otr/stills/pass1_portrait_*.png"))
            + list(portraits_dir.glob("otr_stills/pass1_portrait_*.png"))
            + list(portraits_dir.glob("otr_humo_pass1_portrait_*.png")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[idx % len(candidates)]

    # 3. Any HuMo portrait (no cast match) -- newest first
    candidates = sorted(
        portraits_dir.glob("otr_humo_pass1_portrait_*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    # 4. None. BUG-LOCAL-093 removed the env-still fallback that used
    #    to live here -- producing a wrong face is worse than skipping
    #    the line. Caller logs SKIP and VideoComposite fills with
    #    static radio per BUG-129a.

    return None


def _find_composite(
    shot_id: str | None,
    speaker: str | None,
    portraits_dir: Path,
) -> Path | None:
    """Per-(shot, speaker) FLUX pass3 composite picker."""
    if not shot_id or not speaker:
        return None
    shot_slug = _composite_slug(shot_id, limit=24)
    speaker_slug = _composite_slug(speaker, limit=40)
    new_pat = f"otr/stills/pass3_{shot_slug}_{speaker_slug}_*.png"
    mid_pat = f"otr_stills/pass3_{shot_slug}_{speaker_slug}_*.png"
    legacy_pat = f"otr_humo_pass3_{shot_slug}_{speaker_slug}_*.png"
    candidates = sorted(
        list(portraits_dir.glob(new_pat))
        + list(portraits_dir.glob(mid_pat))
        + list(portraits_dir.glob(legacy_pat)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _png_to_tensor(png_path: Path):
    """Load a PNG file as a ComfyUI IMAGE tensor [1, H, W, C] in
    [0, 1] float32. Bypasses ComfyUI's LoadImage which requires the
    file to be in input/ -- we read directly from the otr/ tree."""
    from PIL import Image  # type: ignore
    import numpy as np  # type: ignore
    import torch  # type: ignore
    img = Image.open(png_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr)[None, ...]  # [1, H, W, C]
    return tensor


def _rescue_orphan_line_char_ids(lines: list, cast: list) -> tuple[int, int, list]:
    """Detect lines whose ``char_id`` doesn't match any ``cast[]`` entry,
    and fuzzy-rescue them by speaker-name match.

    Mutates ``lines`` in place: orphan lines whose speaker name fuzzy-
    matches an existing cast member get their ``char_id`` rewritten to
    the matched cast member's char_id. Lines with no rescue path are
    left as-is and will fall through to the BUG-LOCAL-129a static-radio
    fallback in VideoComposite (no drift, just visual mismatch).

    Returns ``(rescued_count, unrescued_count, log_messages)``.

    Match strategy (in order):
      1. Exact uppercase name match (``MANFRED`` == ``MANFRED``)
      2. Substring match either direction (``MANFRED DIALLO`` contains
         ``MANFRED`` -> rescue) with length-ratio score >= 0.5
    """
    cast_by_id = {(c.get("char_id") or "").strip(): c for c in cast}
    cast_names = {
        (c.get("name") or "").strip().upper(): (c.get("char_id") or "").strip()
        for c in cast if c.get("name") and c.get("char_id")
    }

    log_msgs: list = []
    rescued = 0
    unrescued = 0
    for ln in lines:
        cid = (ln.get("char_id") or "").strip()
        if cid and cid in cast_by_id:
            continue  # OK, char_id resolves
        # ORPHAN
        speaker_name = (
            ln.get("speaker") or ln.get("name") or ln.get("character_name") or ""
        ).strip().upper()
        line_id = ln.get("line_id") or "?"
        if not speaker_name:
            log_msgs.append(
                f"ORPHAN_LINE_CHAR_ID line={line_id} char_id={cid!r} -- "
                f"no speaker name available, cannot fuzzy-rescue; "
                f"BUG-129a static-radio will fire"
            )
            unrescued += 1
            continue
        # Exact name match
        if speaker_name in cast_names:
            new_cid = cast_names[speaker_name]
            log_msgs.append(
                f"RESCUED_ORPHAN line={line_id} char_id={cid!r} "
                f"speaker={speaker_name!r} -> char_id={new_cid!r} "
                f"(exact name match)"
            )
            ln["char_id"] = new_cid
            rescued += 1
            continue
        # Fuzzy substring + length-ratio match
        best_cid = None
        best_score = 0.0
        best_name = ""
        for cast_name, cast_cid in cast_names.items():
            if speaker_name in cast_name or cast_name in speaker_name:
                ratio = (
                    min(len(speaker_name), len(cast_name))
                    / max(len(speaker_name), len(cast_name))
                )
                if ratio > best_score:
                    best_score = ratio
                    best_cid = cast_cid
                    best_name = cast_name
        if best_cid and best_score >= 0.5:
            log_msgs.append(
                f"RESCUED_ORPHAN line={line_id} char_id={cid!r} "
                f"speaker={speaker_name!r} -> char_id={best_cid!r} "
                f"(fuzzy match {best_score:.2f}: {best_name!r})"
            )
            ln["char_id"] = best_cid
            rescued += 1
        else:
            log_msgs.append(
                f"ORPHAN_LINE_CHAR_ID line={line_id} char_id={cid!r} "
                f"speaker={speaker_name!r} -- no fuzzy match in cast "
                f"{list(cast_names.keys())}; BUG-129a static-radio will fire"
            )
            unrescued += 1
    return rescued, unrescued, log_msgs


def _audio_slice_rms_db(audio_dict: dict, start_s: float, dur_s: float) -> float:
    """Compute the RMS amplitude (in dBFS) of an audio slice from a
    ComfyUI AUDIO dict. Returns ``float("-inf")`` if the slice is
    digitally silent (all zeros) or if the slice cannot be computed.

    Used by BUG-LOCAL-031 audio-silent-skip gate: BatchHumoRender peeks
    at the per-line audio slice BEFORE invoking HuMo. If the slice's
    RMS is below the configured speech threshold (default -28 dBFS),
    HuMo is skipped for that line and downstream
    ``OTR_VideoComposite`` falls back to the static radio bookend
    segment automatically (BUG-LOCAL-129a path).

    This catches the failure mode where Bark generated valid speech
    audio for a line but it got dropped / misaligned somewhere in
    the SceneSequencer / AudioEnhance / master-mix pipeline, leaving
    the master mix's slot for that line filled with ambient or
    silence. Without this gate HuMo wastes ~14 minutes lipsyncing a
    talking head to a silent track.
    """
    import math
    try:
        import torch  # type: ignore
    except Exception:
        return float("-inf")
    waveform = audio_dict.get("waveform")
    sr = int(audio_dict.get("sample_rate") or 0)
    if waveform is None or sr <= 0:
        return float("-inf")
    start_sample = max(0, int(float(start_s) * sr))
    end_sample = max(start_sample + 1, int((float(start_s) + float(dur_s)) * sr))
    try:
        sliced = waveform[..., start_sample:end_sample]
        if sliced.numel() == 0:
            return float("-inf")
        # mean-square -> RMS in linear; convert to dBFS (20*log10(rms))
        ms = float((sliced.float() ** 2).mean().item())
        if ms <= 0.0:
            return float("-inf")
        rms = math.sqrt(ms)
        if rms <= 1e-12:
            return float("-inf")
        return 20.0 * math.log10(rms)
    except Exception:
        return float("-inf")


def _slice_audio_tensor(audio_dict: dict, start_s: float, dur_s: float) -> dict:
    """Slice an AUDIO dict by start_s and dur_s. The AUDIO type in
    ComfyUI is ``{"waveform": Tensor[B, C, samples], "sample_rate":
    int}``. Returns a new dict with the sliced waveform; sample_rate
    unchanged."""
    waveform = audio_dict["waveform"]
    sr = int(audio_dict["sample_rate"])
    start_sample = max(0, int(start_s * sr))
    end_sample = max(start_sample + 1, int((start_s + dur_s) * sr))
    if waveform.dim() == 2:
        # Some callers pass [C, samples] instead of [B, C, samples]
        sliced = waveform[..., start_sample:end_sample]
    else:
        sliced = waveform[..., start_sample:end_sample]
    return {"waveform": sliced.contiguous(), "sample_rate": sr}


def _pad_audio_silence_lead(audio_dict: dict, pad_samples: int) -> dict:
    """BUG-LOCAL-102: prepend ``pad_samples`` of digital silence
    (zero samples) to a sliced AUDIO dict. Used to give HuMo's audio
    cross-attention some leading silence so the model burns its
    intrinsic ~3-6-frame motion-onset freeze on silence rather than
    on the first dialogue phoneme. ChatGPT + Gemini round-robin
    2026-04-28 PM both confirm zeros are sufficient (Whisper / HuMo
    audio encoders treat zeros, low-noise, and room tone identically
    for the purpose of "no speech here"). Returns a new audio dict;
    sample_rate is preserved.
    """
    if pad_samples <= 0:
        return audio_dict
    import torch  # type: ignore  # lazy: this module is imported headless by tests
    waveform = audio_dict["waveform"]
    sr = int(audio_dict["sample_rate"])
    pad_shape = list(waveform.shape)
    pad_shape[-1] = int(pad_samples)
    silence = torch.zeros(pad_shape, dtype=waveform.dtype, device=waveform.device)
    padded = torch.cat([silence, waveform], dim=-1).contiguous()
    return {"waveform": padded, "sample_rate": sr}


def _trim_audio_lead(audio_dict: dict, pad_samples: int) -> dict:
    """Inverse of ``_pad_audio_silence_lead``: drop the leading
    ``pad_samples`` from an AUDIO dict. Used at HuMo clip save time
    so the on-disk mp4 contains only the dialogue audio (no leading
    silence), keeping VideoComposite placement math unchanged.
    """
    if pad_samples <= 0:
        return audio_dict
    waveform = audio_dict["waveform"]
    sr = int(audio_dict["sample_rate"])
    pad_samples = int(pad_samples)
    if waveform.shape[-1] <= pad_samples:
        # Nothing left after trim -- caller should never hit this with
        # a positive pad_ms; degrade by returning a minimal 1-sample slice.
        trimmed = waveform[..., :1]
    else:
        trimmed = waveform[..., pad_samples:]
    return {"waveform": trimmed.contiguous(), "sample_rate": sr}


def _resolve_cast_stills_from_ledger(
    cast: list[dict],
    portraits_dir: Path,
    ledger_mtime: float | None,
    *,
    slack_seconds: float = 60.0,
) -> tuple[dict[str, Path], list[Path]]:
    """BUG-LOCAL-088: build a cast_char_id -> still_path map using
    the ledger's mtime as the freshness floor.

    Strategy:
      1. Glob `output/otr/stills/full_env_*.png` (both nested layouts).
      2. Filter to mtime >= (ledger_mtime - slack_seconds). If no
         ledger mtime is available (inline JSON ledger), fall back
         to "last 30 minutes" so we still favour fresh stills.
      3. Sort surviving candidates by mtime descending (newest first).
      4. Walk the cast list in order; assign candidates[i] to
         cast[i]['char_id']. If candidates run out, leave that
         char_id unmapped (caller falls back to the existing
         _find_portrait/_find_composite tiers).

    Returns ``(char_id_to_path, fresh_candidates_sorted_newest_first)``.
    The caller logs which assignment came from "fresh" vs "stale" so
    the runtime log shows the freshness state per line.
    """
    import time as _time

    fresh_floor: float
    if ledger_mtime is not None:
        fresh_floor = ledger_mtime - slack_seconds
    else:
        # No on-disk ledger source; conservatively accept anything
        # written in the last 30 minutes.
        fresh_floor = _time.time() - 30 * 60

    # BUG-LOCAL-028 (2026-05-03): added per-episode workspace pattern.
    # The first pattern walks the new canonical layout
    # ``output/otr/episodes/*/stills/full_env_*.png``; the second + third
    # are the legacy flat-dir patterns kept for back-compat. Mtime filter
    # below still enforces episode freshness so wrong-episode pollution
    # cannot survive even when multiple per-episode dirs exist on disk.
    candidates_all: list[Path] = []
    for pattern in (
        "otr/episodes/*/stills/full_env_*.png",
        "otr/stills/full_env_*.png",
        "otr_stills/full_env_*.png",
    ):
        for p in portraits_dir.glob(pattern):
            try:
                if p.stat().st_mtime >= fresh_floor:
                    candidates_all.append(p)
            except OSError:
                continue

    candidates_all.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    char_id_to_path: dict[str, Path] = {}
    for i, c in enumerate(cast):
        char_id = (c.get("char_id") or "").strip()
        if not char_id:
            continue
        if i < len(candidates_all):
            char_id_to_path[char_id] = candidates_all[i]

    return char_id_to_path, candidates_all


def _save_clip_via_ffmpeg(
    *,
    images,
    audio_dict: dict,
    out_path: Path,
    fps: int = 25,
) -> Path:
    """Write a HuMo clip .mp4 directly via ffmpeg, bypassing
    CreateVideo + SaveVideo.

    BUG-LOCAL-083 workaround: ComfyUI v0.20.1's CreateVideo +
    SaveVideo are V3-style nodes that read ``cls.hidden`` (set by
    the executor with extra_pnginfo + prompt metadata) at execute
    time. Direct-calling those nodes from inside another node's
    execute() leaves ``cls.hidden = None`` and they crash at::

        File "comfy_extras/nodes_video.py", line 100, in execute
            if cls.hidden.extra_pnginfo is not None:
        AttributeError: 'NoneType' object has no attribute
        'extra_pnginfo'

    HuMo renders the frames + audio segment fine, but the save step
    fails -- 12 clips x 9 minutes each = 108 minutes of GPU work
    discarded. Same family as BUG-LOCAL-074 (executor state assumed
    by node code).

    Fix: write the decoded image tensor as a PNG sequence + the
    audio waveform as raw f32 PCM into a temp dir, then mux into
    the final mp4 with ffmpeg. Same on-disk artifact (mp4 with
    embedded audio at canonical ``<line_id>.mp4`` filename), no
    executor coupling.

    Args:
        images: IMAGE tensor ``[N_frames, H, W, C]`` float in [0, 1]
            (or 3-D ``[H, W, C]`` for a single frame).
        audio_dict: ``{"waveform": Tensor[..., samples],
                       "sample_rate": int}`` (ComfyUI AUDIO type).
        out_path: final mp4 path, e.g.
            ``output/otr/videos/<ep_id>/<line_id>.mp4``.
        fps: frame rate (HuMo = 25).

    Returns:
        ``out_path`` on success.

    Raises:
        RuntimeError on ffmpeg failure.
    """
    import shutil
    import subprocess
    import tempfile
    import numpy as np  # type: ignore
    import torch  # type: ignore
    from PIL import Image  # type: ignore

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. Image tensor -> uint8 numpy [N, H, W, C] ----
    img = images
    if isinstance(img, torch.Tensor):
        if img.device.type != "cpu":
            img = img.detach().to("cpu", copy=False)
        arr = img.numpy()
    else:
        arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise RuntimeError(
            f"_save_clip_via_ffmpeg: unexpected image shape {arr.shape}"
        )
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    n_frames = int(arr.shape[0])
    if n_frames < 1:
        raise RuntimeError("_save_clip_via_ffmpeg: zero frames decoded")

    # ---- 2. Audio waveform -> raw f32 interleaved ----
    waveform = audio_dict.get("waveform") if audio_dict else None
    sr = int(audio_dict.get("sample_rate", 16000)) if audio_dict else 16000
    have_audio = waveform is not None
    if have_audio:
        if isinstance(waveform, torch.Tensor):
            if waveform.device.type != "cpu":
                waveform = waveform.detach().to("cpu", copy=False)
            wf = waveform.numpy()
        else:
            wf = np.asarray(waveform)
        # Strip leading batch dims down to [C, samples]
        while wf.ndim > 2:
            wf = wf[0]
        if wf.ndim == 1:
            wf = wf[None, :]
        # ffmpeg f32le wants samples-major interleaved for multi-ch
        wf_il = np.ascontiguousarray(wf.T.astype(np.float32, copy=False))
        n_channels = int(wf.shape[0])
        if wf_il.size == 0:
            have_audio = False

    tmp = Path(tempfile.mkdtemp(prefix="otr_humo_clip_"))
    try:
        # ---- 3. Write PNG frame sequence ----
        frames_dir = tmp / "frames"
        frames_dir.mkdir()
        for i in range(n_frames):
            Image.fromarray(arr[i]).save(frames_dir / f"f_{i:05d}.png")

        # ---- 4. Write audio as raw f32le ----
        audio_raw = tmp / "audio.f32"
        if have_audio:
            wf_il.tofile(audio_raw)

        # ---- 5. ffmpeg mux ----
        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "error",
            "-framerate", str(int(fps)),
            "-i", str(frames_dir / "f_%05d.png"),
        ]
        if have_audio:
            cmd += [
                "-f", "f32le",
                "-ar", str(sr),
                "-ac", str(n_channels),
                "-i", str(audio_raw),
            ]
        cmd += [
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "medium",
        ]
        if have_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += [str(out_path)]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg mux failed (rc={proc.returncode}): "
                f"{(proc.stderr or '').strip()[:500]}"
            )
        return out_path
    finally:
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass


def _concat_clips_via_ffmpeg(
    chunk_paths: list[Path],
    out_path: Path,
) -> Path:
    """BUG-LOCAL-086: concat per-chunk HuMo mp4 files into a single
    per-line mp4 via ffmpeg's concat demuxer.

    Used by the chunking dispatch when an audio line exceeds
    HUMO_MAX_FRAMES. Each chunk is rendered independently against the
    same portrait, written via _save_clip_via_ffmpeg (so all chunks
    share codec / pix_fmt / sample rate), then this helper stitches
    them into the final ``<line_id>.mp4`` that VideoComposite expects.

    Concat demuxer with ``-c copy`` requires identical encoding across
    inputs -- guaranteed because every chunk goes through the same
    _save_clip_via_ffmpeg call with the same fps + audio sample rate.

    Args:
        chunk_paths: ordered list of per-chunk mp4 paths (chunk 1 first).
        out_path: final per-line mp4 path.

    Returns:
        ``out_path`` on success.

    Raises:
        RuntimeError on ffmpeg failure or empty input list.
    """
    import os
    import subprocess
    import tempfile

    if not chunk_paths:
        raise RuntimeError("_concat_clips_via_ffmpeg: empty chunk list")
    if len(chunk_paths) == 1:
        # Single chunk: caller should have written directly to out_path.
        # Defensive copy if not already at the right location.
        if Path(chunk_paths[0]) == Path(out_path):
            return out_path
        import shutil
        shutil.copy2(str(chunk_paths[0]), str(out_path))
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Concat demuxer takes a list file: each line ``file '<absolute_path>'``.
    # Single quotes inside paths must be escaped as ``'\''`` per the
    # ffmpeg concat demuxer spec.
    list_fd, list_path = tempfile.mkstemp(prefix="humo_concat_", suffix=".txt")
    try:
        with os.fdopen(list_fd, "w", encoding="utf-8") as f:
            for p in chunk_paths:
                abs_p = str(Path(p).resolve()).replace("'", "'\\''")
                f.write(f"file '{abs_p}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            "-movflags", "+faststart",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg concat failed (rc={proc.returncode}): "
                f"{(proc.stderr or '').strip()[:500]}"
            )
        return out_path
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass


def _build_pos_prompt(
    speaker: str,
    ln: dict,
    cast: list[dict],
    meta: dict | None = None,
) -> str:
    """Build a HuMo positive prompt from the ledger line + cast desc.

    Sprint C C5f (2026-05-15): when meta carries a successful
    story_brief, the brief-derived lighting + atmosphere terms are
    appended BEFORE _DEFAULT_POS_SUFFIX so HuMo's lip-sync clip
    inherits the post-script reflection's mood without losing the
    composition guidance that _DEFAULT_POS_SUFFIX provides.
    """
    speaker_desc = ""
    for c in cast:
        if (c.get("name") or "").upper().strip() == (speaker or "").upper().strip():
            # S26-B3 cleanbreak: legacy `description` fallback removed;
            # callers MUST supply `character_description`.
            speaker_desc = (c.get("character_description") or "").strip()
            break
    if not speaker_desc:
        speaker_desc = (
            f"A {speaker.lower()} character speaks calmly with subtle facial "
            f"expressions"
        ) if speaker else "A character speaks calmly with subtle facial expressions"

    # Sprint C C5f: brief-derived lighting/atmosphere insertion.
    # Absolute import -- matches the module's documented sys.path-prepend
    # pattern (see top-of-file bootstrap + the _otr_paths import). A
    # relative import breaks when this file is loaded standalone (tests,
    # ad-hoc scripts) with no parent package.
    from _otr_story_brief_helpers import get_story_brief_lighting
    lighting = get_story_brief_lighting(meta or {})
    if lighting:
        return f"{speaker_desc}, {lighting}, {_DEFAULT_POS_SUFFIX}"
    return f"{speaker_desc}, {_DEFAULT_POS_SUFFIX}"


def _lazy_humo_nodes() -> dict[str, Any]:
    """Resolve the ComfyUI / extension node classes BatchHumoRender
    needs. Lazy import so the module loads even if Comfy isn't fully
    initialized (e.g., during static analysis or pytest)."""
    refs: dict[str, Any] = {}
    try:
        import nodes  # type: ignore
        refs["CLIPTextEncode"] = getattr(nodes, "CLIPTextEncode", None)
        refs["KSampler"] = getattr(nodes, "KSampler", None)
        refs["VAEDecode"] = getattr(nodes, "VAEDecode", None)
        node_mappings = getattr(nodes, "NODE_CLASS_MAPPINGS", {})
    except Exception:
        node_mappings = {}

    # Wan 2.1 / HuMo extension nodes (community / Kijai pack)
    refs["WanHuMoImageToVideo"] = node_mappings.get("WanHuMoImageToVideo")
    refs["AudioEncoderEncode"] = node_mappings.get("AudioEncoderEncode")
    refs["CreateVideo"] = node_mappings.get("CreateVideo")
    refs["SaveVideo"] = node_mappings.get("SaveVideo")
    return refs


def _call(node_instance, **kwargs):
    """Call a ComfyUI node by its declared FUNCTION method name with
    keyword arguments matching INPUT_TYPES.

    BUG-LOCAL-080 fix: ComfyUI node method names are NOT consistent.
    AudioEncoderEncode does not have ``.encode()`` -- the actual
    method is named by the class's ``FUNCTION`` attribute. Hardcoding
    method names (``audio_enc.encode(...)``) blows up at runtime.

    Using ``getattr(instance, instance.FUNCTION)`` resolves whatever
    name the node author declared. Passing keyword arguments avoids
    positional-order mismatches (CLIPTextEncode expects ``clip``
    first in some signatures, ``text`` first in others).

    Returns whatever the underlying method returns (typically a
    tuple of outputs).
    """
    fn_name = getattr(node_instance, "FUNCTION", "execute")
    fn = getattr(node_instance, fn_name, None)
    if fn is None:
        raise AttributeError(
            f"node {type(node_instance).__name__} has no method "
            f"{fn_name!r} (declared by FUNCTION attribute)"
        )
    return fn(**kwargs)


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class BatchHumoRender:
    """Render N HuMo lip-sync clips in one graph execution.

    OUTPUT_NODE = True (BUG-LOCAL-077 lesson): this node has
    side-effects (writes per-line .mp4 files to output/otr/videos/
    <ep_id>/) AND its output STRINGs are diagnostic. Without
    OUTPUT_NODE the executor would prune it when downstream consumers
    don't fully chain.
    """

    CATEGORY = "OTR/v2/Visual"
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("clips_dir", "clip_count", "report")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import comfy.samplers  # type: ignore
            samplers = comfy.samplers.KSampler.SAMPLERS
            schedulers = comfy.samplers.KSampler.SCHEDULERS
        except Exception:
            samplers = ["uni_pc"]
            schedulers = ["simple"]
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "HuMo model with lightx2v Lora applied (post ModelSamplingSD3).",
                }),
                "clip": ("CLIP", {
                    "tooltip": "umt5_xxl text encoder loaded via CLIPLoader.",
                }),
                "vae": ("VAE", {
                    "tooltip": "wan_2.1 VAE loaded via VAELoader.",
                }),
                "audio_encoder": ("AUDIO_ENCODER", {
                    "tooltip": "Whisper Large v3 fp16 loaded via AudioEncoderLoader.",
                }),
                "audio": ("AUDIO", {
                    "tooltip": "Full episode audio from EpisodeAssembler.",
                }),
                "ledger_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Ledger source. Four accepted forms "
                        "(duck-typed at _load_ledger_with_path in "
                        "this file, lines 1660-1762):\n"
                        "  1. Empty string -> auto-pick newest "
                        "non-pending ledger from disk.\n"
                        "  2. String starting with '{' -> parsed as "
                        "JSON manifest directly.\n"
                        "  3. Path to *_ledger.json -> loaded directly.\n"
                        "  4. Path to *.mp4 -> multi-tier stem fallback "
                        "derives the matching _ledger.json (exact "
                        "match -> underscore-collapse variant -> "
                        "directory scan with mtime + episode_id "
                        "substring match). This is intentional duck "
                        "typing; static analyzers regularly misflag "
                        "the .mp4 form as a wiring error (see "
                        "BUG-LOCAL-124 revert in BUG_LOG.md). Slot "
                        "name kept as 'ledger_json' for workflow "
                        "compat; rename would orphan existing wires "
                        "(ComfyUI matches inputs by name on load with "
                        "no native aliasing).\n"
                        "Original docstring: Production Ledger JSON "
                        "or path to *_ledger.json on disk."
                    ),
                }),
                "portraits_dir": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": (
                        "Directory holding PASS1 character portraits "
                        "(otr_humo_pass1_portrait_*.png) and optional "
                        "PASS3 (shot,speaker) composites. Empty -> "
                        "auto-resolve to output/otr/portraits/<ep_id>/."
                    ),
                }),
                "clip_length": ("FLOAT", {
                    "default": 7.0,
                    "min": 1.32,
                    "max": 14.12,
                    "step": 0.04,
                    "tooltip": (
                        "Max per-CHUNK duration in seconds. Lines whose "
                        "audio exceeds this are split into N consecutive "
                        "chunks (BUG-LOCAL-086) and rendered against the "
                        "same portrait, then ffmpeg-concat into the final "
                        "per-line mp4. Default 7.0 -> 175 frames -> 177 "
                        "(Wan 2.1 4n+1 = 7.08s, historically stable on "
                        "16 GiB Blackwell). Bump up to 14.12 (353 frames) "
                        "if VRAM holds, to single-pass typical Bark "
                        "dialogue lines."
                    ),
                }),
                "max_clips": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "tooltip": "0 = render every line; smoke-test caps with positive int.",
                }),
                "seed": ("INT", {"default": 7, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 50}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (samplers, {"default": "uni_pc"}),
                "scheduler": (schedulers, {"default": "simple"}),
                # 2026-05-03 EVENING (BUG-LOCAL-030 revised spec):
                # render at 480x832 PORTRAIT @ 25 FPS -- the canonical
                # Wan2.1-HuMo-14B trained dim per ROADMAP "Stable shape:
                # length=97 (3.88 s @ 25 fps), 480x832, batch=1". An
                # earlier revision attempted 1280x720 landscape for face
                # detail but Jeffrey reverted to native ("humo native
                # portrait render then native scaled to 1472x832 with
                # black pillaboxes") -- staying inside the trained range
                # avoids OOD lipsync drift + ~2.3x VRAM overhead.
                # VideoComposite pillarboxes to 1472x832 canvas with
                # ~496px black bars per side; the post-RTXUpscale procgen
                # blend (Phase B) fills those bars with audio-reactive
                # CRT scanlines so the visible black surround becomes
                # part of the SIGNAL LOST visual signature.
                "width": ("INT", {"default": 480, "min": 256, "max": 1280, "step": 8}),
                "height": ("INT", {"default": 832, "min": 256, "max": 1280, "step": 8}),
            },
            "optional": {
                # BUG-LOCAL-086: dependency-only gate input. Wire any
                # IMAGE output from the FLUX-render branch (typically
                # OTR_UnloadAll's IMAGE passthrough, which sits between
                # OTR_BatchFluxRender and SaveImage) to this socket so
                # ComfyUI's topological scheduler runs FLUX env stills
                # FIRST, then UnloadAll frees FLUX VRAM, THEN this
                # batch-HuMo loop starts with fresh per-episode
                # `full_env_*.png` stand-ins on disk. Without the wire,
                # HuMo would run before FLUX (both being independent
                # graph branches downstream of SignalLostVideo) and
                # `_find_portrait` would fall back to whatever stale
                # env stills happened to be on disk from prior runs.
                # The value is intentionally ignored at runtime; the
                # wire is purely a graph-edge for ordering.
                "flux_done_gate": ("IMAGE", {
                    "tooltip": (
                        "Optional FLUX->HuMo dependency gate. Wire "
                        "OTR_UnloadAll's IMAGE output here to force "
                        "FLUX env stills to render BEFORE this HuMo "
                        "loop. Value is ignored; only the dependency "
                        "edge matters."
                    ),
                }),
                "humo_warmup_pad_ms": ("INT", {
                    "default": 200,
                    "min": 0,
                    "max": 500,
                    "step": 10,
                    "tooltip": (
                        "BUG-LOCAL-102: leading silence (in ms) padded "
                        "onto each line's audio before HuMo's audio "
                        "cross-attention. Burns HuMo's intrinsic ~3-6 "
                        "frame motion-onset freeze on silence rather "
                        "than on the first dialogue word, eliminating "
                        "the constant audio-leads-lips lag the listener "
                        "hears in the rendered episode. The pad is "
                        "trimmed back off the on-disk clip so timeline "
                        "placement math is unchanged. Set to 0 to "
                        "disable (reverts to pre-BUG-102 behavior)."
                    ),
                }),
                "min_speech_rms_db": ("FLOAT", {
                    "default": -28.0,
                    "min": -90.0,
                    "max": 0.0,
                    "step": 0.5,
                    "tooltip": (
                        "BUG-LOCAL-031 audio-silent-skip gate. Per-line "
                        "audio slice RMS threshold in dBFS. If the slice's "
                        "RMS for a character line falls below this value, "
                        "BatchHumoRender SKIPS the HuMo render for that "
                        "line (saves ~14 min of GPU time). Downstream "
                        "OTR_VideoComposite then auto-falls-back to the "
                        "static radio-bookend segment for that line "
                        "(BUG-LOCAL-129a path). Catches the failure mode "
                        "where Bark generated valid speech but it got "
                        "dropped/misaligned in SceneSequencer / "
                        "AudioEnhance / master-mix routing, leaving the "
                        "slot silent. Default -28 dBFS is conservative: "
                        "normal Bark speech reads ~-20 to -22 dBFS, "
                        "background ambient ~-35 dBFS or quieter. Set to "
                        "-90 to disable the gate entirely (every "
                        "character line goes to HuMo regardless of audio)."
                    ),
                }),
                "resume_from_ledger": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "If ON, before rendering each clip, check "
                        "ledger.clips[] for an entry with matching "
                        "line_id whose mp4_path file still exists on "
                        "disk. If found, skip the HuMo render and "
                        "reuse the existing clip. Pairs with the "
                        "per-clip incremental ledger save: a crashed "
                        "or cancelled run can be re-queued and the "
                        "render picks up where it left off instead "
                        "of redoing finished clips. Set OFF to force "
                        "a full re-render (e.g. after changing the "
                        "warmup_pad_ms widget value -- existing clips "
                        "would have a stale pad)."
                    ),
                }),
                "humo_max_lines_per_process": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 200,
                    "tooltip": (
                        "BUG-LOCAL-126 soak cap. If > 0, halt the HuMo "
                        "render loop after this many lines complete in "
                        "the current ComfyUI process and raise "
                        "HumoSoakCapReached. Pairs with "
                        "resume_from_ledger=True so a follow-up run picks "
                        "up where this one stopped. Set to 0 to disable "
                        "the cap. Empirical floor on the RTX 5080 16 GB: "
                        "the overnight 2026-05-07 soak survived 9 lines "
                        "before the CUDA pool fragmented to a fatal "
                        "abort; 6-8 is a safe per-process budget for "
                        "now while the underlying allocator drift is "
                        "investigated."
                    ),
                }),
                "cuda_hard_reset_on_oom": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "BUG-LOCAL-126 alarm-plumbing. After a caught "
                        "HuMo OOM, run a hard CUDA reset "
                        "(unload_all_models + gc.collect + "
                        "soft_empty_cache + cuda.synchronize + "
                        "empty_cache + ipc_collect) before the next "
                        "line. Without this, two OOM cycles caused the "
                        "PyTorch pool to drift from ~14 GiB in-use to "
                        "~24 GiB allocated on a 15.92 GiB device, then "
                        "fatal-abort the third sample. Leave ON unless "
                        "you have a specific reason to keep allocator "
                        "state across an OOM."
                    ),
                }),
                "stop_workflow_on_soak_cap": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Round-robin Step 4 (post-Stage 1, 2026-05-08). "
                        "If ON (default, production overnight pattern): "
                        "raise HumoSoakCapReached when the per-process "
                        "cap fires; the workflow halts and the watcher "
                        "re-queues. If OFF (validation/smoke pattern): "
                        "save the ledger + soak_cap stamp, log a "
                        "SOAK_CAP_REACHED report line, and break out of "
                        "the loop so the rest of the DAG (LTX, "
                        "Composite, Upscale, PostProcgenBlend) still "
                        "runs on whatever fresh clips were rendered. "
                        "End-to-end smoke validation requires this OFF "
                        "with cap=3 to verify the downstream chain "
                        "without burning a full episode of HuMo time. "
                        "Unsafe in production resume mode until LTX "
                        "grows its own resume-from-ledger scan -- "
                        "currently re-running with this OFF on the same "
                        "ledger would re-render LTX on previously-done "
                        "clips."
                    ),
                }),
            },
        }

    def execute(
        self,
        model, clip, vae, audio_encoder, audio,
        ledger_json: str,
        portraits_dir: str,
        clip_length: float,
        max_clips: int,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        width: int,
        height: int,
        flux_done_gate=None,  # BUG-LOCAL-086: dependency gate, value ignored
        humo_warmup_pad_ms: int = 200,  # BUG-LOCAL-102: see INPUT_TYPES tooltip
        min_speech_rms_db: float = -28.0,  # BUG-LOCAL-031 audio-silent-skip gate (helper present but loop wiring deferred -- see investigation note)
        resume_from_ledger: bool = True,  # 2026-04-29: see INPUT_TYPES tooltip
        humo_max_lines_per_process: int = 0,  # BUG-LOCAL-126 soak cap; 0 = disabled
        cuda_hard_reset_on_oom: bool = True,  # BUG-LOCAL-126 alarm-plumbing
        stop_workflow_on_soak_cap: bool = True,  # 2026-05-08 Step 4: hard raise vs soft break
    ):
        t_start = time.time()
        # BUG-LOCAL-086: flux_done_gate is intentionally not consumed.
        # Its sole purpose is to insert a graph edge so ComfyUI's
        # topological scheduler runs the FLUX -> UnloadAll branch
        # BEFORE this node starts.
        del flux_done_gate

        # BUG-LOCAL-102: clamp the pad to sane bounds. 0 disables.
        warmup_pad_ms = max(0, min(500, int(humo_warmup_pad_ms)))

        # ---- 1. Parse ledger ----
        # BUG-LOCAL-234 fix (2026-05-19): rename-safe ledger refresh.
        # Mirrors visual/batch_flux_render.py:1015-1041 verbatim. The
        # in-flight Ledger singleton's `path` property advances through
        # `Ledger.rename_episode` (BUG-LOCAL-015 Phase B 2026-05-02),
        # so reading from the singleton always tracks the post-rename
        # location. The wired `ledger_json` socket may carry a
        # PRE-rename serialization from an upstream node whose output
        # was captured before SignalLostVideo ran the rename --
        # trusting that string would land path resolution in the
        # renamed-away dir (cf. BUG-LOCAL-234 surfaced 2026-05-18 23:42).
        # Fall back to the legacy wired-input loader only when the
        # singleton lookup AND mtime-walker both fail (standalone test
        # invocations).
        ledger = None
        ledger_path = None
        try:
            try:
                from . import production_ledger as _PROD_LEDGER  # type: ignore
                from . import _otr_ledger as _OTRL_LD  # type: ignore
            except ImportError:
                import production_ledger as _PROD_LEDGER  # type: ignore
                import _otr_ledger as _OTRL_LD  # type: ignore
            _led_singleton = _PROD_LEDGER.get_ledger()
            ledger_path = Path(_led_singleton.path)
            if not ledger_path.exists():
                log.warning(
                    "[BatchHumoRender] singleton path %s does not exist "
                    "on disk; falling back to mtime walker",
                    ledger_path,
                )
                ledger_path = _OTRL_LD.find_most_recent_ledger(
                    [otr_episodes_root()]
                )
        except Exception as _exc:  # noqa: BLE001
            log.warning(
                "[BatchHumoRender] singleton lookup failed (%s); "
                "falling back to mtime walker", _exc,
            )
            try:
                try:
                    from . import _otr_ledger as _OTRL_LD  # type: ignore
                except ImportError:
                    import _otr_ledger as _OTRL_LD  # type: ignore
                ledger_path = _OTRL_LD.find_most_recent_ledger(
                    [otr_episodes_root()]
                )
            except Exception:  # noqa: BLE001
                ledger_path = None
        if ledger_path is not None:
            try:
                try:
                    from . import _otr_ledger as _OTRL_LD  # type: ignore
                except ImportError:
                    import _otr_ledger as _OTRL_LD  # type: ignore
                ledger = _OTRL_LD.load_ledger_safe(ledger_path)
            except Exception as _exc:  # noqa: BLE001
                log.warning(
                    "[BatchHumoRender] load_ledger_safe(%s) raised %r; "
                    "falling back to wired ledger_json", ledger_path, _exc,
                )
                ledger = None
        if ledger is None:
            # Last-resort: parse the wired socket. This is the legacy
            # standalone-test path -- production runs always succeed
            # via the singleton-fallback above.
            log.warning(
                "[BatchHumoRender] singleton refresh produced no ledger; "
                "falling back to wired ledger_json socket (may be stale "
                "if upstream serialized before episode rename)"
            )
            ledger, ledger_path = self._load_ledger_with_path(ledger_json)

        # l3-2026-05-08 telemetry: expose this process's ledger to
        # `_hard_reset_cuda_context` so cuda_hard_reset_count bumps
        # without threading the ledger ref through every helper. Also
        # append a process_runs[] entry so multi-batch resume sequence
        # is provable from any single ledger.
        global _CURRENT_LEDGER_REF
        _CURRENT_LEDGER_REF = ledger
        try:
            try:
                from . import _otr_ledger as _OTRL_PR  # type: ignore
            except ImportError:
                import _otr_ledger as _OTRL_PR  # type: ignore
            _OTRL_PR.append_meta_process_run(ledger)
        except Exception as _pr_exc:  # noqa: BLE001
            log.warning(
                "[BatchHumoRender] process_runs append failed (non-fatal): %s",
                _pr_exc,
            )
        cast = ledger.get("cast") or []
        lines = ledger.get("lines") or []
        # Sprint C C5f (2026-05-15): surface story_brief_status once per
        # ledger load for E-07 observability. The brief flows into the
        # per-clip prompt via _build_pos_prompt below.
        try:
            # Absolute import -- see the note at the _build_pos_prompt
            # call site; relative imports break standalone module loads.
            from _otr_story_brief_helpers import get_story_brief_status
            _brief_status = get_story_brief_status(
                ledger.get("meta") if isinstance(ledger.get("meta"), dict) else {}
            )
            log.info(
                "[BatchHumoRender] ledger loaded; cast=%d lines=%d "
                "story_brief_status=%s",
                len(cast), len(lines), _brief_status,
            )
        except Exception as _exc:  # noqa: BLE001 -- diagnostic only
            log.info(
                "[BatchHumoRender] ledger loaded; cast=%d lines=%d "
                "(story_brief_status probe failed: %s)",
                len(cast), len(lines), _exc,
            )

        # BUG-LOCAL-079 (2026-05-03 EVENING): orphan char_id detection +
        # fuzzy rescue. Historically the legacy LLMDirector could emit
        # a cast member name like
        # "MANFRED" but a line is attributed to "MANFRED DIALLO" (or
        # the fuzzy consolidator BUG-098 misses a merge), the line's
        # char_id may NOT be in cast[]. Without this rescue:
        #   - HuMo can't find a portrait -> falls through to env-still
        #     tier 4 (now disabled by skip_env_stills) -> falls to
        #     tier 5 (any env still) -> may silently use the wrong face
        #   - WORST CASE: HuMo skips the line entirely -> no per-line
        #     mp4 -> VideoComposite BUG-129a fires static-radio
        #     fallback -> audio still aligned (no drift) but visual is
        #     a static radio image instead of the speaking character
        # The rescue here repoints orphan lines to the closest cast
        # member name match BEFORE HuMo dispatches, so the right
        # portrait is used. Fuzzy match strategy: exact uppercase,
        # then substring + length-ratio >= 0.5.
        try:
            _resc, _unresc, _resc_msgs = _rescue_orphan_line_char_ids(lines, cast)
            if _resc or _unresc:
                log.warning(
                    "[BatchHumoRender] BUG-079 orphan char_id pass: "
                    "rescued=%d unrescued=%d (out of %d lines, %d cast)",
                    _resc, _unresc, len(lines), len(cast),
                )
                for _m in _resc_msgs:
                    log.warning("[BatchHumoRender] %s", _m)
        except Exception as _resc_exc:
            log.warning(
                "[BatchHumoRender] BUG-079 orphan rescue helper failed: %s "
                "(non-fatal; lines pass through unchanged)", _resc_exc,
            )

        episode_id = ledger.get("episode_id", "episode")
        cid_to_name = {
            (c.get("char_id") or ""): (c.get("name") or "")
            for c in cast if c.get("char_id")
        }
        ledger_mtime: float | None = None
        if ledger_path is not None:
            try:
                ledger_mtime = ledger_path.stat().st_mtime
            except OSError:
                ledger_mtime = None
        log.info(
            "[BatchHumoRender] episode_id=%s cast=%d lines=%d ledger=%s",
            episode_id, len(cast), len(lines),
            (ledger_path.name if ledger_path is not None else "<inline>"),
        )

        # ---- 1b. Resolve radio still ref-image path ----
        # Step 5 (ROADMAP P0, 2026-04-30 lock): non-dialogue lines
        # (announcer, music_*, sfx) use this as the I2V reference
        # image instead of the cast portrait.  Stamped by
        # BatchFluxRender's _render_and_save_radio_bookend; checked
        # in both top-level ledger.radio_bookend_path AND
        # ledger.meta.radio_bookend_path.  None is non-fatal --
        # radio-role lines simply fall through to the legacy
        # portrait resolver in that case (with a warning), so an
        # episode without a rendered radio still still completes.
        radio_still_path = _resolve_radio_still_path(ledger)
        if radio_still_path is not None:
            log.info(
                "[BatchHumoRender] radio_still_path=%s (used for "
                "non-dialogue speaker_role lines)", radio_still_path,
            )
        else:
            log.info(
                "[BatchHumoRender] radio_still_path=None (radio-role "
                "lines will fall through to portrait resolver)"
            )

        # ---- 2. Resolve portraits dir ----
        # Default = ComfyUI output root so _find_portrait's globs hit
        # both `otr/portraits/<ep_id>/otr_humo_pass1_portrait_*.png`
        # AND `otr/stills/full_env_*.png` AND `otr/stills/pass1_*.png`.
        # User-supplied input wins if non-empty.
        if not portraits_dir or not portraits_dir.strip():
            portraits_dir_path = comfy_output_dir()
            # Sprint E E10 / M7: D0d wired BatchFluxPortraitRender's
            # new portraits_dir STRING output to HuMo's portraits_dir
            # input (L116 in workflow JSON). The empty-string branch
            # below is the auto-resolve FALLBACK that fires when the
            # wire is missing or upstream FluxPortrait did not run.
            # Pre-E10 this fallback was silent; post-E10 it logs WARN
            # so soak diagnostics surface the missing-wire case
            # explicitly.
            log.warning(
                "[BatchHumoRender] portraits_dir input is empty; "
                "auto-resolved to %s. This is the D0d fallback path; "
                "wire FluxPortrait.portraits_dir output to HuMo's "
                "portraits_dir input for the explicit D0d contract.",
                portraits_dir_path,
            )
        else:
            portraits_dir_path = Path(portraits_dir)
        log.info("[BatchHumoRender] portraits_dir=%s", portraits_dir_path)

        # Sprint E E10 / M6: pre-batch wall-time estimate. Operator
        # planning a soak needs to see the projected total before the
        # render starts. Reference: ~10-12 min per character line on
        # RTX 5080. We log the lower bound (10 min) since the upper
        # depends on per-line audio length and chunk count.
        _character_lines_count = sum(
            1 for _ln in lines
            if (_ln.get("speaker_role") or "").lower() == "character"
        )
        if max_clips > 0:
            _character_lines_count = min(_character_lines_count, max_clips)
        _est_min_low = _character_lines_count * 10
        _est_min_high = _character_lines_count * 12
        log.info(
            "[BatchHumoRender] PRE-BATCH ESTIMATE: %d character "
            "line(s) -> ~%d-%d minutes total HuMo wall time on RTX "
            "5080 (Sprint C-era reference; re-time on Sprint A "
            "before quoting). Non-character lines route to radio "
            "still and are excluded from this estimate.",
            _character_lines_count, _est_min_low, _est_min_high,
        )

        # ---- 3. Resolve clips output_dir (canonical OTR tree) ----
        # output_dir = ComfyUI/output/otr/videos/<episode_id>/
        clips_dir_path = otr_videos_dir(episode_id)
        clips_dir_path.mkdir(parents=True, exist_ok=True)

        # ---- 4. Lazy-load Comfy nodes ----
        # BUG-LOCAL-083: CreateVideo + SaveVideo dropped from the
        # required list. Their V3 ``cls.hidden`` coupling crashes
        # when called outside the executor; we mux clip mp4s with
        # ``_save_clip_via_ffmpeg`` instead.
        refs = _lazy_humo_nodes()
        missing = [k for k in (
            "CLIPTextEncode", "KSampler", "VAEDecode",
            "WanHuMoImageToVideo", "AudioEncoderEncode",
        ) if refs.get(k) is None]
        if missing:
            raise RuntimeError(
                f"BatchHumoRender: missing required ComfyUI node classes: {missing}"
            )

        text_enc = refs["CLIPTextEncode"]()
        sampler = refs["KSampler"]()
        vae_decoder = refs["VAEDecode"]()
        humo_node = refs["WanHuMoImageToVideo"]()
        audio_enc_node = refs["AudioEncoderEncode"]()

        # ---- 5. (REMOVED 2026-05-08) HuMo pre-pin was upside-down ----
        # Pre-2026-05-08: this block called
        #   mm.load_models_gpu([model], force_full_load=True)
        # to "pin" HuMo on GPU before Phase A/B encoders ran. The pin
        # was structurally broken on 16 GB devices: HuMo stages at
        # 16,531 MB; Phase A then needs umt5 (6.4 GB); Phase B needs
        # Whisper (1.2 GB); together they cannot coexist with the
        # pinned HuMo on a 16 GB total. Under encoder pressure
        # ComfyUI's dynamic VRAM offloader silently violated the pin,
        # but the weights remained referenced -- which fragmented the
        # cudaMallocAsync pool by Phase C entry. On large episodes
        # (60+ chunks) Phase C HuMo then ran at ~88x slowdown
        # (5,284 s/it vs healthy 60 s/it), because the offloader fell
        # into perpetual swap-to-CPU through PCIe instead of running
        # contiguously in GDDR.
        #
        # The "Model WAN21_HuMo prepared for dynamic VRAM loading.
        # 16531MB Staged" log line at Phase C entry confirmed the pin
        # did not survive: HuMo had to be re-staged anyway when
        # WanHuMoImageToVideo actually ran.
        #
        # Phase C's WanHuMoImageToVideo invocation goes through
        # load_models_gpu itself when it needs HuMo, so the pin was
        # also redundant. Removing it lets Phase A/B encoders own GPU
        # cleanly; the inter-phase reset (section 8.5) then drains
        # the pool before Phase C stages HuMo on a defragmented
        # allocator.
        #
        # Round-robin synthesis 2026-05-08 path-forward Fix 1.
        # ----------------------------------------------------------

        # ---- 5.5. BUG-LOCAL-088: ledger-driven cast->still binding ----
        # BUG-LOCAL-093 (2026-05-04 EVENING): cast_still_map binding
        # REMOVED. It globbed FLUX env stills (full_env_*.png) and used
        # them as character ref images, which produced visibly wrong
        # faces (HuMo lipsynced against a random environment scene that
        # happened to contain a different person). Pre-BUG-078 it was a
        # stopgap; post-BUG-078 the per-cast portrait pass writes
        # canonical c0X_portrait.png files and stamps cast[].portrait_path,
        # which _find_portrait tier 1 reads. The dispatch below relies
        # exclusively on _find_portrait + _find_composite; if neither
        # returns a portrait the line is SKIPPED and VideoComposite
        # covers it with static-radio fill (BUG-129a) -- same handling
        # as music/sfx. Wrong-face is worse than no-face.
        cast_still_map: dict = {}  # kept as empty defense-in-depth so the
                                   # downstream dispatch reads cleanly

        # ---- 6. Build per-line plan (no GPU work yet) ----
        if max_clips and max_clips > 0:
            lines = lines[:max_clips]

        report_lines: list[str] = [
            f"BatchHumoRender: episode={episode_id} target_lines={len(lines)}",
        ]
        if max_clips and max_clips > 0:
            report_lines.append(f"  capped to {max_clips} clips")

        # ---- 5.7. BUG-LOCAL-094: distribute clip timings across full episode ----
        # When upstream nodes (SceneSequencer / Bark / EpisodeAssembler)
        # don't write `lines[i].start_s` and `lines[i].dur_s` back to
        # the ledger -- which is the current state -- the legacy
        # fallback was `start_s = idx * clip_length` (idx * 7s) for
        # every line. That clumped all HuMo clips into the first
        # `N * 7s` of the episode, leaving the back half as
        # procgen-only. Worse: with episodes longer than 7s/line
        # average, the audio slice for each clip drifted progressively
        # out of sync with the actual Bark TTS playback in the
        # composite.
        #
        # Fix (heuristic, until proper Bark write-back lands):
        # estimate per-line timings from the line's `word_count`
        # share of the total episode duration. This:
        #   * spreads HuMo clips evenly across the whole timeline
        #   * scales the slice positions roughly with where each
        #     line actually plays in the assembled audio
        #   * still respects the `clip_length` ceiling for HuMo
        #     render budget (BUG-082 VRAM cap)
        # The estimate is overridden line-by-line whenever the
        # ledger DOES carry an explicit `start_s` / `dur_s` (so a
        # future SceneSequencer write-back upgrade replaces this
        # automatically with no further code changes here).
        estimated_timings: dict[int, tuple[float, float]] = {}
        try:
            total_episode_dur = ledger.get("total_episode_dur_s")
            if not total_episode_dur:
                # Fallback: estimate from the audio tensor itself.
                wf = audio.get("waveform") if isinstance(audio, dict) else None
                sr = int(audio.get("sample_rate", 0)) if isinstance(audio, dict) else 0
                if wf is not None and sr > 0:
                    try:
                        total_episode_dur = float(wf.shape[-1]) / float(sr)
                    except Exception:
                        total_episode_dur = None
            if total_episode_dur and total_episode_dur > 0:
                voiced = [ln for ln in lines if (ln.get("char_id") or "").strip()]
                # word-share denominator: prefer word_count, fall back
                # to char_count (so a 0-word ledger does not divide by 0)
                total_weight = sum(
                    max(int(ln.get("word_count") or 0), 1)
                    for ln in voiced
                ) or len(voiced) or 1
                cursor = 0.0
                # Map indices in `lines` (not just voiced) so the
                # per-line loop's `idx` lookup still works
                for idx, ln in enumerate(lines):
                    if not (ln.get("char_id") or "").strip():
                        # non-voiced line (announcer-only ledger
                        # entry, or filtered-out parse artefact);
                        # zero-dur slice that won't render.
                        estimated_timings[idx] = (cursor, 0.0)
                        continue
                    weight = max(int(ln.get("word_count") or 0), 1)
                    proportional = float(total_episode_dur) * (weight / total_weight)
                    estimated_timings[idx] = (cursor, proportional)
                    cursor += proportional
                log.info(
                    "[BatchHumoRender] BUG-094 estimated %d line timings "
                    "across %.1fs episode (avg %.2fs/line)",
                    len(estimated_timings), total_episode_dur,
                    total_episode_dur / max(len(voiced), 1),
                )
        except Exception as exc:
            log.warning("[BatchHumoRender] timing estimation failed: %s", exc)
            estimated_timings = {}

        # Per-line plan: (line_id, speaker, start_s, dur_s, humo_length,
        #                 pos_text, ref_image_tensor, audio_dict)
        plan: list[dict] = []
        for idx, ln in enumerate(lines):
            line_id = str(ln.get("line_id") or f"l{idx + 1:03d}")
            speaker = (
                ln.get("speaker")
                or cid_to_name.get(ln.get("char_id") or "", "")
                or ""
            ).strip()
            char_id = (ln.get("char_id") or "").strip()

            # Per-line timing priority (BUG-094):
            #   1. ledger.lines[i].start_s/dur_s (real values from a
            #      future SceneSequencer write-back -- best)
            #   2. estimated_timings[idx] (this run's word-share
            #      heuristic computed in section 5.7 above -- spreads
            #      clips across the full episode duration)
            #   3. legacy fallback: idx * clip_length (clumped at
            #      front of timeline; only triggers when neither the
            #      ledger nor the heuristic produced timings, e.g.
            #      total_episode_dur_s missing AND audio tensor
            #      unreadable)
            start_s = ln.get("start_s")
            dur_s = ln.get("dur_s")
            if (start_s is None or start_s == "") and idx in estimated_timings:
                start_s = estimated_timings[idx][0]
            if (dur_s is None or dur_s == "" or dur_s <= 0) and idx in estimated_timings:
                dur_s = estimated_timings[idx][1] or clip_length
            if start_s is None or start_s == "":
                start_s = idx * clip_length
            if dur_s is None or dur_s == "" or dur_s <= 0:
                dur_s = clip_length
            start_s = float(start_s)
            dur_s = float(dur_s)

            # BUG-LOCAL-102: extend humo_length to cover the leading
            # silence pad so HuMo has frames to render BOTH the
            # warm-up freeze AND the dialogue. The pad frames are
            # trimmed off the on-disk clip below so timeline math
            # stays in terms of dur_s.
            pad_s = warmup_pad_ms / 1000.0

            # BUG-LOCAL-086 (2026-05-04): chunking dispatch.
            # `clip_length` is the max single-pass duration (workflow
            # widget, default 7.0s). Lines whose audio exceeds it are
            # split into N consecutive chunks of <= clip_length each,
            # rendered independently against the same portrait, then
            # ffmpeg-concat into the final per-line mp4. This replaces
            # the old BUG-LOCAL-105 silent-clamp which truncated long
            # lines to the cap, leaving the back half of the audio
            # playing against a frozen last frame. The cap from
            # BUG-LOCAL-105 still applies to each individual chunk
            # (humo_length_for_dur clamps at HUMO_MAX_FRAMES) so VRAM
            # behaviour per chunk matches the historically-stable
            # single-pass path.
            chunk_max_dur_s = max(0.04, float(clip_length) - pad_s)
            if (float(dur_s) + pad_s) <= float(clip_length):
                # Single-chunk path (most lines, current behaviour)
                chunk_specs = [{"start_offset_s": 0.0, "dur_s": float(dur_s)}]
            else:
                # Multi-chunk path: split evenly so every chunk has the
                # same render budget. Even split avoids a tiny tail
                # chunk that would render with poor model context.
                n_chunks = max(2, math.ceil(float(dur_s) / chunk_max_dur_s))
                chunk_dur_s = float(dur_s) / n_chunks
                chunk_specs = [
                    {"start_offset_s": i * chunk_dur_s, "dur_s": chunk_dur_s}
                    for i in range(n_chunks)
                ]
                log.info(
                    "[BatchHumoRender] BUG-LOCAL-086: line %s dur_s=%.2fs "
                    "> clip_length=%.2fs -- splitting into %d chunks of "
                    "%.2fs each",
                    line_id, float(dur_s), float(clip_length),
                    n_chunks, chunk_dur_s,
                )

            # All chunks share the same dur (even split) so a single
            # humo_length covers each render call. Capped helper keeps
            # the BUG-LOCAL-105 safety net in place per chunk.
            humo_length = humo_length_for_dur(
                chunk_specs[0]["dur_s"] + pad_s
            )

            # I2V ref-image selection (post-BUG-LOCAL-129 routing,
            # 2026-05-01):
            #   character  -> existing BUG-088 portrait chain.
            #   announcer  -> portrait chain (renders if LLM emitted
            #                 ANNOUNCER as a cast member with a
            #                 portrait; falls through to no-clip
            #                 otherwise, which VideoComposite covers
            #                 via BUG-129a static-radio fill).
            #   music_*/sfx-> hard skip. Defense in depth via
            #                 is_never_humo_role(): even if a
            #                 portrait somehow resolves, do NOT
            #                 dispatch HuMo. VideoComposite's
            #                 BUG-129a static-fill path provides
            #                 visual coverage for these audio events.
            #   (no role)  -> default to character.
            #
            # The historical "radio role -> radio still as ref_image"
            # branch is retained as defense-in-depth: is_radio_role()
            # always returns False post-BUG-129, so this branch is
            # dead. If a future commit re-populates _RADIO_ROLES, the
            # regression resurfaces visibly via the
            # `radio-still (...)` ref_source stamp on rendered clips.
            speaker_role = resolve_speaker_role(ln)
            shot_id = ln.get("shot_id")
            ref_png: Path | None = None
            ref_source = "none"

            # BUG-LOCAL-129b (2026-05-01): hard skip for music/sfx
            # before any portrait lookup. These roles are visual
            # coverage only -- they get a deterministic static-radio
            # segment in VideoComposite, never a HuMo render.
            if is_never_humo_role(speaker_role):
                report_lines.append(
                    f"  {line_id}: SKIP HuMo (role={speaker_role}, "
                    f"covered by VideoComposite static-radio fill)"
                )
                log.info(
                    "[BatchHumoRender] line %s speaker_role=%s -- "
                    "skipping HuMo render (BUG-129b: visual covered "
                    "by VideoComposite static-fill)",
                    line_id, speaker_role,
                )
                continue

            if is_radio_role(speaker_role) and radio_still_path is not None:
                # Defense-in-depth dead branch (BUG-LOCAL-129 fix):
                # is_radio_role() always returns False post-BUG-129.
                # If this fires, _RADIO_ROLES was re-populated and
                # the regression is back. Logged loudly.
                ref_png = radio_still_path
                ref_source = f"radio-still ({speaker_role})"
                log.error(
                    "[BatchHumoRender] BUG-LOCAL-129 REGRESSION: line "
                    "%s speaker_role=%s routed to radio still as "
                    "HuMo ref_image. _RADIO_ROLES has been re-populated "
                    "in _otr_speaker_role.py; HuMo will produce an "
                    "unconstrained generic face. Revert that change.",
                    line_id, speaker_role,
                )
            else:
                # Character / announcer -> portrait chain.
                #
                # BUG-LOCAL-093 (2026-05-04 EVENING): stopgaps removed.
                # Dispatch is now strictly portrait-only. If neither
                # _find_portrait nor _find_composite returns a path,
                # the line is skipped and VideoComposite covers it with
                # static-radio fill (BUG-129a) -- same handling as
                # music/sfx. Pre-093 the dispatch fell through to the
                # cast_still_map (globbed full_env_*.png FLUX env
                # stills) which produced visibly wrong faces because
                # HuMo lipsynced against environment scenes that
                # happened to contain unrelated people. Wrong-face is
                # worse than no-face; better to surface "missing
                # portrait" loudly via the SKIP path so it gets fixed.
                #
                # Priority:
                #   1. _find_portrait (tier 1 = cast[i].portrait_path
                #      from BUG-LOCAL-078's per-cast portrait pass)
                #   2. _find_composite (per-shot pass3 composite override)
                ref_png = _find_portrait(speaker, cast, portraits_dir_path)
                if ref_png:
                    ref_source = "find_portrait"
                if not ref_png:
                    ref_png = _find_composite(shot_id, speaker, portraits_dir_path)
                    if ref_png:
                        ref_source = "find_composite"

            if not ref_png:
                report_lines.append(
                    f"  {line_id}: SKIP no portrait (role={speaker_role})"
                )
                log.warning(
                    "[BatchHumoRender] line %s speaker=%r role=%s: "
                    "no portrait AND no radio still",
                    line_id, speaker, speaker_role,
                )
                continue

            try:
                ref_image = _png_to_tensor(ref_png)
            except Exception as exc:
                report_lines.append(f"  {line_id}: SKIP portrait load failed: {exc}")
                continue

            # BUG-LOCAL-102: pad each chunk's audio with leading silence
            # so HuMo's first few frames (motion-onset freeze) happen
            # during silence, not during the first phoneme of the chunk.
            # BUG-LOCAL-086: when the line is multi-chunk every chunk
            # gets its own pad so the motion-onset freeze burns down at
            # each chunk boundary, NOT carried over -- otherwise the
            # second chunk's first 3-6 frames would be a stale lipsync
            # against the previous chunk's last phoneme, visible as a
            # micro-jolt at the join. The pad is trimmed off each chunk
            # before concat so the final line mp4 has zero artificial
            # silence inserted.
            _line_audio_full = _slice_audio_tensor(audio, start_s, float(dur_s))
            _line_sr = int(_line_audio_full.get("sample_rate", 0)) or 0
            pad_samples = (
                int(warmup_pad_ms * _line_sr / 1000)
                if warmup_pad_ms > 0 else 0
            )
            chunks_data: list[dict] = []
            for spec in chunk_specs:
                chunk_audio = _slice_audio_tensor(
                    audio,
                    start_s + float(spec["start_offset_s"]),
                    float(spec["dur_s"]),
                )
                chunk_audio = _pad_audio_silence_lead(chunk_audio, pad_samples)
                chunks_data.append({
                    "audio": chunk_audio,
                    "start_offset_s": float(spec["start_offset_s"]),
                    "dur_s": float(spec["dur_s"]),
                })
            # Sprint C C5f (2026-05-15): thread meta into the per-clip
            # prompt builder so brief-derived lighting + atmosphere
            # terms surface in the HuMo positive prompt.
            _meta_for_brief = ledger.get("meta") if isinstance(ledger.get("meta"), dict) else {}
            pos_text = _build_pos_prompt(speaker, ln, cast, meta=_meta_for_brief)

            # BUG-LOCAL-088: log which still each line consumed + source
            # tier + freshness so post-mortem can verify ledger
            # accuracy without re-globbing the disk.
            try:
                ref_mtime = ref_png.stat().st_mtime
                ref_age = (time.time() - ref_mtime) if ref_mtime else None
                age_str = (
                    f"{ref_age:.0f}s ago" if ref_age is not None and ref_age < 3600
                    else (f"{ref_age/60:.0f}min ago" if ref_age is not None and ref_age < 86400
                          else (f"{ref_age/3600:.0f}h ago" if ref_age is not None
                                else "unknown"))
                )
            except OSError:
                age_str = "unknown"
            log.info(
                "[BatchHumoRender] line %s speaker=%s role=%s "
                "char_id=%s ref=%s source=%s age=%s",
                line_id, speaker or "?", speaker_role,
                char_id or "?", ref_png.name, ref_source, age_str,
            )

            plan.append({
                "idx": idx, "line_id": line_id, "speaker": speaker,
                "speaker_role": speaker_role,
                "start_s": start_s, "dur_s": dur_s, "humo_length": humo_length,
                "pos_text": pos_text, "ref_image": ref_image,
                # BUG-LOCAL-086: per-chunk audio + offset metadata. List
                # of {audio, start_offset_s, dur_s}. Single-chunk lines
                # have len(chunks)==1 (the common path); long lines
                # have len(chunks) > 1 and the render loop concats.
                "chunks": chunks_data,
                "ref_png_name": ref_png.name,
                "ref_source": ref_source,
                # BUG-LOCAL-102: how many leading samples / frames the
                # save step must trim back off so the on-disk clip is
                # aligned (audio = dialogue from sample 0; video =
                # articulated motion from frame 0, no warm-up freeze).
                "warmup_pad_samples": pad_samples,
                "warmup_pad_frames": (
                    int(warmup_pad_ms * HUMO_FPS / 1000)
                    if warmup_pad_ms > 0 else 0
                ),
            })

        if not plan:
            report_lines.append("FATAL: empty plan -- no lines had portraits")
            return (str(clips_dir_path), 0, "\n".join(report_lines))

        # ---- 7. Phase A: encode all text prompts up front ----
        # Encode the negative once, then every positive in sequence.
        # This avoids reloading WanTEModel between every line during
        # the HuMo render loop (BUG-LOCAL-080 architectural complaint:
        # "audio was trying encode at the same time as humo was
        # running" -- the per-line encode + render mix forced ComfyUI
        # to swap models in and out of VRAM constantly).
        log.info("[BatchHumoRender] Phase A: encoding %d positive + 1 negative text prompts",
                 len(plan))
        try:
            negative = _call(text_enc, clip=clip, text=_CHINESE_NEGATIVE)[0]
        except Exception as exc:
            raise RuntimeError(f"BatchHumoRender: negative encode failed: {exc}")

        for entry in plan:
            try:
                entry["positive"] = _call(text_enc, clip=clip, text=entry["pos_text"])[0]
            except Exception as exc:
                log.warning("[BatchHumoRender] %s: text encode failed: %s",
                            entry["line_id"], exc)
                entry["positive"] = None

        # ---- 8. Phase B: encode all per-line audio up front ----
        # BUG-LOCAL-086: encode every chunk's audio separately so each
        # gets its own audio_emb keyed to its time window. Single-chunk
        # lines have one encode (matches pre-086 behaviour); multi-chunk
        # lines have N encodes, one per chunk.
        _total_audio_segments = sum(len(e["chunks"]) for e in plan)
        log.info("[BatchHumoRender] Phase B: encoding %d audio segments via Whisper",
                 _total_audio_segments)
        for entry in plan:
            for chunk in entry["chunks"]:
                try:
                    chunk["audio_emb"] = _call(
                        audio_enc_node,
                        audio_encoder=audio_encoder,
                        audio=chunk["audio"],
                    )[0]
                except Exception as exc:
                    log.warning("[BatchHumoRender] %s: audio encode failed: %s",
                                entry["line_id"], exc)
                    chunk["audio_emb"] = None

        # ---- 8.5. VRAM cleanup between Phase B and Phase C ----
        # BUG-LOCAL-081: at 30+ lines, Phase B reloaded Whisper for
        # every line (each AudioEncoderEncode call triggered a fresh
        # "WhisperLargeV3 prepared" log). At Phase B end, GPU still
        # holds: Whisper weights (1.2 GB), umt5_xxl text encoder
        # (6.4 GB from Phase A), 30 audio embedding tensors (~10 MB
        # each = 300 MB), 30 positive cond tensors (~50 MB each =
        # 1.5 GB). Total ~9.4 GB pinned before HuMo even starts to
        # load.  Phase C asks for HuMo (16.5 GB staged) on a 16 GB
        # card -- ComfyUI's dynamic VRAM loader thrashes pages
        # perpetually and never converges to forward progress.
        # Symptom: KSampler stuck at "0/6 [?it/s]" for 20+ minutes.
        #
        # Fix is two-step:
        #   1. Move every Phase A/B output tensor to CPU. That
        #      releases GPU pages backing positive/negative cond and
        #      audio embeddings -- Phase C's WanHuMoImageToVideo
        #      moves them back to GPU when it actually needs them.
        #   2. unload_all_models + soft_empty_cache to evict Whisper
        #      and umt5_xxl from GPU and return pages to the CUDA
        #      allocator pool.
        try:
            import torch  # type: ignore
            def _to_cpu(obj):
                """Best-effort move a Comfy CONDITIONING / AUDIO_EMB
                payload to CPU. Conditioning is a list of [tensor,
                meta_dict] pairs; meta_dict can carry pooled_output
                also as a tensor. AUDIO_EMB shape is unknown but
                likely tensor or dict."""
                if isinstance(obj, torch.Tensor):
                    return obj.detach().to("cpu", copy=False) if obj.device.type == "cuda" else obj
                if isinstance(obj, list):
                    return [_to_cpu(x) for x in obj]
                if isinstance(obj, tuple):
                    return tuple(_to_cpu(x) for x in obj)
                if isinstance(obj, dict):
                    return {k: _to_cpu(v) for k, v in obj.items()}
                return obj
            negative = _to_cpu(negative)
            for entry in plan:
                if entry.get("positive") is not None:
                    entry["positive"] = _to_cpu(entry["positive"])
                # Legacy fallback. ``entry["audio_emb"]`` was the
                # pre-BUG-086 location for audio embeddings (per-line,
                # not per-chunk). The current Phase B writes them to
                # ``chunk["audio_emb"]`` instead (line ~2071), so this
                # branch is a no-op on current code paths. Kept in case
                # a future refactor restores the per-line layout.
                if entry.get("audio_emb") is not None:
                    entry["audio_emb"] = _to_cpu(entry["audio_emb"])
                # Round-robin v3 (2026-05-08, Fix 1.5): the BUG-086
                # split moved audio embeddings onto chunks but the
                # cleanup walk above wasn't updated. Result: every
                # ``chunk["audio_emb"]`` tensor stayed GPU-resident
                # through the inter-phase boundary -- 90 chunks on a
                # 60-line script meant 0.9-4.5 GiB of stale GPU
                # references, which the subsequent
                # ``unload_all_models() + soft_empty_cache()`` could
                # not reclaim because Python still held references via
                # plan[*]["chunks"][*]["audio_emb"]. Now: walk every
                # chunk's audio_emb to CPU before the inter-phase
                # reset so the allocator pool can actually drain.
                for chunk in entry.get("chunks") or []:
                    if chunk.get("audio_emb") is not None:
                        chunk["audio_emb"] = _to_cpu(chunk["audio_emb"])
            log.info("[BatchHumoRender] Phase A/B tensors moved to CPU")
        except Exception as exc:
            log.warning("[BatchHumoRender] CPU offload failed: %s", exc)

        try:
            import comfy.model_management as mm  # type: ignore
            log.info("[BatchHumoRender] Inter-phase VRAM cleanup: unload_all_models + soft_empty_cache")
            mm.unload_all_models()
            mm.soft_empty_cache(force=True)
        except Exception as exc:
            log.warning("[BatchHumoRender] inter-phase VRAM cleanup failed: %s", exc)

        # HuMo VRAM-thrash instrumentation (2026-05-24). The bare-
        # workflow smoke proved HuMo-14B runs ~6x faster outside the
        # OTR pipeline (~47 s/it vs 140-279 s/it in-pipeline), so
        # something the pipeline leaves resident is starving HuMo's
        # VRAM budget at Phase C entry. This probe logs the real card
        # state right after the inter-phase cleanup so an operator run
        # shows exactly how much that cleanup reclaimed -- and whether
        # the occupant is tracked by ComfyUI's model manager (so
        # unload_all_models would catch it) or held out-of-band by
        # OTR's own loaders (writer LLM / MusicGen / FLUX), which it
        # would not. Diagnostic only -- no behavior change.
        try:
            import torch  # type: ignore
            import comfy.model_management as mm  # type: ignore
            _alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            _reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
            _free_b, _total_b = torch.cuda.mem_get_info()
            _tracked = [
                type(getattr(m, "model", m)).__name__
                for m in (getattr(mm, "current_loaded_models", None) or [])
            ]
            log.info(
                "[BatchHumoRender] PHASE-C-VRAM-PROBE: torch allocated=%.0f MB "
                "reserved=%.0f MB | cuda free=%.0f MB total=%.0f MB | "
                "comfy-tracked models=%d %s",
                _alloc_mb, _reserved_mb,
                _free_b / (1024 ** 2), _total_b / (1024 ** 2),
                len(_tracked), _tracked,
            )
        except Exception as probe_exc:  # noqa: BLE001
            log.warning(
                "[BatchHumoRender] Phase C VRAM probe failed: %s", probe_exc
            )

        # ---- 9. Phase C: HuMo render loop (model stays warm) ----
        # Round-robin v3 (2026-05-08, Fix 1.5): the pre-fix
        # ``e.get("audio_emb") is not None`` check always evaluated
        # False on current schema (audio embeddings live on chunks,
        # not entries -- BUG-086). That made this log line always
        # report "0 lines" regardless of the actual plan size. The
        # render loop itself at line ~2223 correctly checks
        # ``c.get("audio_emb") for c in _chunks``, which is why
        # rendering worked at all -- only the count log was wrong.
        # Now: a per-entry "ready" predicate that requires
        # ``positive`` plus at least one chunk plus ``audio_emb`` on
        # every chunk. Matches the render loop's gate semantics.
        def _entry_ready_for_humo(e):
            chunks = e.get("chunks") or []
            return (
                e.get("positive") is not None
                and bool(chunks)
                and all(c.get("audio_emb") is not None for c in chunks)
            )
        log.info(
            "[BatchHumoRender] Phase C: HuMo render loop, %d lines",
            sum(1 for e in plan if _entry_ready_for_humo(e)),
        )
        # Round-robin Item A counter split (2026-05-08 synthesis):
        # BEFORE this fix a single `rendered` counted both resumed and
        # fresh clips, and the cap fired off the combined value. A
        # resumed run that found 6 existing clips would hit the cap
        # after ONE fresh render -- the fix did the opposite of what
        # was intended. Now:
        #   total_clips_output    -- resumed + fresh; used for return
        #                            value + report (preserves the
        #                            existing /N/ in the report line)
        #   fresh_rendered_this_process -- fresh HuMo work only;
        #                                  used for the cap check and
        #                                  for HumoSoakCapReached's
        #                                  lines_completed field
        total_clips_output = 0
        fresh_rendered_this_process = 0
        # BUG-LOCAL-089: track per-line clip records for ledger write-back.
        # The `clips[]` ledger field lets downstream tools (concat,
        # post-mortem, OBS scheduler) find every rendered clip by
        # ledger lookup instead of filesystem glob.
        clip_records: list[dict] = []

        # 2026-04-29: build a lookup of already-rendered clips for resume.
        # If resume_from_ledger is ON and the on-disk ledger already has a
        # clips[] entry for a given line_id whose mp4_path file still
        # exists, skip the HuMo render and reuse the existing clip.
        # Pairs with the per-clip incremental save: a crashed run can be
        # re-queued and we pick up at the first un-rendered clip instead
        # of redoing all 33 clips at 9-10 min each.
        existing_clips_by_line_id: dict = {}
        if resume_from_ledger and ledger_path is not None:
            try:
                _existing = ledger.get("clips") or []
                for _ec in _existing:
                    _eid = (_ec or {}).get("line_id")
                    _emp4 = (_ec or {}).get("mp4_path")
                    if _eid and _emp4 and Path(_emp4).is_file():
                        existing_clips_by_line_id[_eid] = _ec
                if existing_clips_by_line_id:
                    log.info(
                        "[BatchHumoRender] resume_from_ledger=ON: %d "
                        "existing clip(s) found in ledger with mp4 on disk; "
                        "those line_ids will be skipped",
                        len(existing_clips_by_line_id),
                    )
            except Exception as _resume_exc:  # noqa: BLE001
                log.warning(
                    "[BatchHumoRender] resume scan failed (will re-render "
                    "everything): %s",
                    _resume_exc,
                )
                existing_clips_by_line_id = {}

        for entry in plan:
            line_id = entry["line_id"]

            # Resume: reuse existing clip if matching line_id + mp4 on disk.
            if line_id in existing_clips_by_line_id:
                _reused = existing_clips_by_line_id[line_id]
                clip_records.append(_reused)
                # Resume branch: bump total only, NOT the cap counter
                # (round-robin Item A).
                total_clips_output += 1
                report_lines.append(
                    f"  {line_id} ({entry.get('speaker')}): RESUMED "
                    f"(reused {_reused.get('mp4_path')})"
                )
                log.info(
                    "[BatchHumoRender] %s RESUMED: reused existing clip %s",
                    line_id, _reused.get("mp4_path"),
                )
                continue

            # BUG-LOCAL-086: skip lines with missing prompt OR any chunk
            # that failed audio encoding. Partial rendering of a line
            # would produce a clip with gaps at the chunk boundary.
            if entry.get("positive") is None:
                report_lines.append(f"  {line_id}: SKIP encode failed in earlier phase (positive)")
                continue
            _chunks = entry.get("chunks") or []
            if not _chunks:
                report_lines.append(f"  {line_id}: SKIP no chunks built")
                continue
            if any(c.get("audio_emb") is None for c in _chunks):
                report_lines.append(f"  {line_id}: SKIP encode failed in earlier phase (chunk audio)")
                continue

            # BUG-LOCAL-031 silent-skip gate (loop wiring landed
            # 2026-05-08, post-Stage-1 synthesis). The 2026-05-08 morning
            # Stage 1 run produced 0.1s of post_bark audio TOTAL across
            # 11 character lines and HuMo dutifully lipsynced 6 fresh
            # clips against silent slots before the cap fired. The gate
            # below catches that failure mode by checking the per-line
            # audio slice's RMS against ``min_speech_rms_db`` (default
            # -28 dBFS). A line below threshold gets:
            #   - render_method = "static_radio_fill" stamped to ledger
            #   - skipped from HuMo render
            #   - downstream OTR_VideoComposite falls back to the static
            #     radio bookend for that time slot via BUG-LOCAL-129a
            # This is graceful-degradation per the original BUG-031
            # design intent (helper_present_loop_wiring_deferred at
            # line 1473 pre-fix). Set min_speech_rms_db = -90 to
            # disable the gate entirely (every line goes to HuMo).
            try:
                _line_start_s = float(entry.get("start_s") or 0.0)
                _line_dur_s = float(entry.get("dur_s") or 0.0)
                _line_rms_db = _audio_slice_rms_db(
                    audio, _line_start_s, _line_dur_s,
                )
            except Exception as _rms_exc:  # noqa: BLE001
                log.warning(
                    "[BatchHumoRender] %s: RMS check failed (%s); "
                    "letting line through to HuMo",
                    line_id, _rms_exc,
                )
                _line_rms_db = 0.0  # neutral; pass through
            if _line_rms_db < float(min_speech_rms_db):
                log.warning(
                    "[BatchHumoRender] %s: BUG-LOCAL-031 silent-skip "
                    "(rms=%.2f dBFS < threshold=%.2f); marking "
                    "render_method=static_radio_fill, "
                    "OTR_VideoComposite will paint static radio bookend",
                    line_id, _line_rms_db, float(min_speech_rms_db),
                )
                report_lines.append(
                    f"  {line_id}: SILENT_SKIP (rms={_line_rms_db:.2f} "
                    f"< {float(min_speech_rms_db):.2f} dBFS)"
                )
                # Stamp render_method so the ledger reflects the
                # graceful-degradation decision; post-mortem can
                # then explain why HuMo was skipped without grepping
                # the log.
                try:
                    try:
                        from . import _otr_ledger as _OTRL_SS  # type: ignore
                    except ImportError:
                        import _otr_ledger as _OTRL_SS  # type: ignore
                    _OTRL_SS.stamp_line_render_method(
                        ledger, line_id, "static_radio_fill",
                    )
                except Exception as _ss_exc:  # noqa: BLE001
                    log.warning(
                        "[BatchHumoRender] %s: render_method stamp "
                        "failed (non-fatal): %s", line_id, _ss_exc,
                    )
                continue

            shot_t0 = time.time()
            try:
                n_chunks = len(_chunks)
                pad_frames = int(entry.get("warmup_pad_frames", 0) or 0)
                pad_samples_save = int(entry.get("warmup_pad_samples", 0) or 0)
                chunk_mp4_paths: list[Path] = []
                _total_mp4_frames = 0

                for chunk_idx, chunk in enumerate(_chunks):
                    # Per-chunk shot_seed: same line gets a stable base
                    # seed (so re-runs reproduce) but each chunk picks a
                    # different stride so chunks 1+2 don't render with an
                    # identical seed (would produce quasi-identical motion
                    # fragments and a visible "stutter back to start" at
                    # the chunk boundary).
                    shot_seed = (
                        seed
                        + entry["idx"] * 1009
                        + chunk_idx * 7919
                    ) & 0x7FFFFFFFFFFFFFFF

                    # WanHuMoImageToVideo returns (positive_with_humo_inputs,
                    # negative_with_humo_inputs, latent). Use kwargs matching
                    # the API-format prompt's input names (BUG-LOCAL-080 fix).
                    humo_out = _call(
                        humo_node,
                        width=width,
                        height=height,
                        length=entry["humo_length"],
                        batch_size=1,
                        positive=entry["positive"],
                        negative=negative,
                        vae=vae,
                        audio_encoder_output=chunk["audio_emb"],
                        ref_image=entry["ref_image"],
                    )
                    humo_pos, humo_neg, humo_latent = humo_out[:3]

                    samples = _call(
                        sampler,
                        model=model,
                        seed=shot_seed,
                        steps=steps,
                        cfg=cfg,
                        sampler_name=sampler_name,
                        scheduler=scheduler,
                        positive=humo_pos,
                        negative=humo_neg,
                        latent_image=humo_latent,
                        denoise=1.0,
                    )[0]

                    images_out = _call(vae_decoder, samples=samples, vae=vae)[0]

                    # BUG-LOCAL-102: trim the leading warmup window. HuMo
                    # rendered N extra frames against the silence we
                    # padded onto the audio so the model's intrinsic
                    # motion-onset freeze (~3-6 frames) burned down before
                    # the first dialogue phoneme. Drop those leading frames
                    # from the decoded image tensor and the matching
                    # leading silence from the audio so the on-disk chunk
                    # starts at the first articulated motion / first
                    # dialogue word. Each chunk gets its own pad+trim so
                    # the motion-onset freeze burns down at every chunk
                    # boundary, not just the line's first chunk.
                    if pad_frames > 0 and isinstance(images_out, torch.Tensor):
                        if images_out.shape[0] > pad_frames:
                            images_out = images_out[pad_frames:].contiguous()
                        else:
                            log.warning(
                                "[BatchHumoRender] BUG-102: line %s chunk %d has %d frames "
                                "but pad_frames=%d -- skipping trim (degraded)",
                                line_id, chunk_idx + 1, images_out.shape[0], pad_frames,
                            )
                    audio_for_save = (
                        _trim_audio_lead(chunk["audio"], pad_samples_save)
                        if pad_samples_save > 0 else chunk["audio"]
                    )

                    # BUG-LOCAL-083: ComfyUI v0.20.1's SaveVideo (and the
                    # paired CreateVideo) is a V3-style node that expects
                    # `cls.hidden` to be populated by the executor with
                    # `extra_pnginfo` + `prompt` metadata. Direct-calling
                    # those nodes from inside another node's execute()
                    # leaves `cls.hidden = None` and crashes at line 100
                    # of comfy_extras/nodes_video.py.
                    #
                    # Workaround: skip CreateVideo + SaveVideo entirely.
                    # Write each chunk mp4 directly via ffmpeg from the
                    # decoded image tensor + per-chunk audio tensor.
                    #
                    # BUG-LOCAL-086 destination policy:
                    #   single-chunk -> write directly to <line_id>.mp4
                    #                   (canonical, matches pre-086 path)
                    #   multi-chunk  -> write to <line_id>__chunk{NN}.mp4
                    #                   then ffmpeg-concat into
                    #                   <line_id>.mp4 after the loop.
                    if n_chunks == 1:
                        _chunk_dest = clips_dir_path / f"{line_id}.mp4"
                    else:
                        _chunk_dest = clips_dir_path / (
                            f"{line_id}__chunk{chunk_idx + 1:02d}.mp4"
                        )
                    _save_clip_via_ffmpeg(
                        images=images_out,
                        audio_dict=audio_for_save,
                        out_path=_chunk_dest,
                        fps=25,
                    )
                    chunk_mp4_paths.append(_chunk_dest)
                    _total_mp4_frames += (
                        int(images_out.shape[0])
                        if isinstance(images_out, torch.Tensor)
                            and images_out.dim() >= 1
                        else (int(entry["humo_length"]) - int(pad_frames))
                    )

                # BUG-LOCAL-086: concat per-chunk mp4s into the canonical
                # per-line mp4 if multi-chunk. Each chunk shares codec /
                # pix_fmt / sample rate (same _save_clip_via_ffmpeg call)
                # so the concat demuxer with -c copy is safe.
                clip_mp4_path = clips_dir_path / f"{line_id}.mp4"
                if n_chunks > 1:
                    _concat_clips_via_ffmpeg(chunk_mp4_paths, clip_mp4_path)
                    # Cleanup per-chunk part files now that concat
                    # succeeded. Concat failures raise above and we
                    # never reach here, so partial cleanup is impossible.
                    for _cp in chunk_mp4_paths:
                        try:
                            _cp.unlink()
                        except OSError:
                            pass

                # Capture timing + post-trim measurements BEFORE
                # building the ledger record so all diagnostic fields
                # are populated in one place.
                shot_ms = int((time.time() - shot_t0) * 1000)

                # Schema l3 (2026-04-28) diagnostic fields, BUG-086 sums:
                # mp4_frames is the SUM across chunks (= total frames in
                # the concatenated on-disk mp4). audio_fed_to_humo_dur_s
                # is the FULL line dur + N pads (one per chunk) so the
                # post-mortem can spot drift = (audio_fed - mp4_dur).
                _pad_s_for_meta = float(warmup_pad_ms) / 1000.0
                _mp4_dur_s = float(_total_mp4_frames) / float(HUMO_FPS)
                _audio_fed_dur_s = (
                    float(entry["dur_s"])
                    + (_pad_s_for_meta * n_chunks)
                )

                # BUG-LOCAL-089 + 086: single ledger record per line
                # regardless of chunk count. Downstream (VideoComposite)
                # sees one mp4 per line at <line_id>.mp4. n_chunks added
                # for traceability so ledger inspection shows whether
                # the line was chunked or single-pass.
                clip_records.append({
                    "line_id": line_id,
                    "char_id": (lines[entry["idx"]].get("char_id") or "").strip() or None,
                    "speaker": entry["speaker"] or None,
                    "mp4_path": str(clip_mp4_path),
                    "start_s": float(entry["start_s"]),
                    "dur_s": float(entry["dur_s"]),
                    "humo_length": int(entry["humo_length"]),
                    "ref_png_name": entry.get("ref_png_name"),
                    "ref_source": entry.get("ref_source"),
                    # BUG-LOCAL-102: how many ms of leading silence the
                    # model rendered against (and that the save trim
                    # removed). Recorded for post-mortem traceability.
                    "warmup_pad_ms": int(warmup_pad_ms),
                    # Schema l3 diagnostic fields (sync regression bait):
                    "humo_render_ms": int(shot_ms),
                    "mp4_frames": int(_total_mp4_frames),
                    "mp4_dur_s": float(_mp4_dur_s),
                    "audio_fed_to_humo_dur_s": float(_audio_fed_dur_s),
                    # BUG-LOCAL-086 traceability: how many chunks got
                    # rendered + concatenated for this line. 1 = single
                    # pass (pre-086 behaviour), >1 = chunked because the
                    # line's audio dur exceeded clip_length.
                    "n_chunks": int(n_chunks),
                    # BUG-LOCAL-106: stamp the timeline-space of
                    # start_s so EpisodeAssembler's shift stays
                    # idempotent. Whatever lines[].start_s_space
                    # was when we read it propagates here. If
                    # SceneSequencer + EpisodeAssembler ran before
                    # this node (FULL workflow), start_s is already
                    # master_mix and EpisodeAssembler will skip
                    # re-shifting on a re-run. If only Scene ran
                    # but Episode didn't (smoke), the value is
                    # scene_audio and a future Episode pass would
                    # apply the shift correctly.
                    "start_s_space": (
                        lines[entry["idx"]].get("start_s_space")
                        or "master_mix"
                    ),
                })
                if n_chunks > 1:
                    report_lines.append(
                        f"  {line_id} ({entry['speaker']}): {shot_ms} ms "
                        f"({n_chunks} chunks, length={entry['humo_length']} "
                        f"per chunk, ref={entry['ref_png_name']})"
                    )
                    log.info(
                        "[BatchHumoRender] %s done in %d ms (%d chunks, "
                        "BUG-LOCAL-086 chunked + concat)",
                        line_id, shot_ms, n_chunks,
                    )
                else:
                    report_lines.append(
                        f"  {line_id} ({entry['speaker']}): {shot_ms} ms "
                        f"(length={entry['humo_length']} ref={entry['ref_png_name']})"
                    )
                    log.info("[BatchHumoRender] %s done in %d ms", line_id, shot_ms)
                # Fresh-render branch: bump BOTH counters (round-robin Item A).
                total_clips_output += 1
                fresh_rendered_this_process += 1

                # l3-2026-05-08 telemetry: stamp lines[].render_method
                # so an audit can distinguish humo_native success from
                # static-radio fallback. Done HERE (post-success) so
                # only actually-rendered HuMo lines carry the tag.
                try:
                    try:
                        from . import _otr_ledger as _OTRL_RM  # type: ignore
                    except ImportError:
                        import _otr_ledger as _OTRL_RM  # type: ignore
                    _OTRL_RM.stamp_line_render_method(
                        ledger, line_id, "humo_native",
                    )
                except Exception as _rm_exc:  # noqa: BLE001
                    log.warning(
                        "[BatchHumoRender] render_method stamp "
                        "failed (non-fatal): %s", _rm_exc,
                    )

                # BUG-LOCAL-126 soak cap. Save the ledger first so the
                # follow-up resume picks up cleanly, then raise the
                # structured signal. Cap == 0 disables this branch.
                # Counter split (round-robin Item A): cap fires off
                # FRESH renders only -- a resume run that found 6
                # existing clips on disk doesn't immediately stop
                # after one fresh render.
                if humo_max_lines_per_process > 0 and fresh_rendered_this_process >= humo_max_lines_per_process:
                    try:
                        try:
                            from . import _otr_ledger as _OTRL_CAP  # type: ignore
                        except ImportError:
                            _NODES_DIR_CAP = Path(__file__).resolve().parent
                            if str(_NODES_DIR_CAP) not in _sys.path:
                                _sys.path.insert(0, str(_NODES_DIR_CAP))
                            import _otr_ledger as _OTRL_CAP  # type: ignore
                        # l3-2026-05-08 telemetry: stamp meta.soak_cap
                        # BEFORE the ledger save so the persisted state
                        # records the cap-hit fact, not just the line
                        # count. Lets the follow-up run distinguish
                        # "cap fired cleanly" from "process crashed
                        # mid-render and the next batch should ramp
                        # backoff before retrying".
                        _OTRL_CAP.stamp_meta_soak_cap(
                            ledger,
                            cap=humo_max_lines_per_process,
                            lines_completed_when_hit=fresh_rendered_this_process,
                            hit=True,
                        )
                        if ledger_path is not None:
                            ledger["clips"] = list(clip_records)
                            _OTRL_CAP.save_ledger_safe(ledger_path, ledger)
                    except Exception as _cap_save_exc:  # noqa: BLE001
                        log.warning(
                            "[BatchHumoRender] BUG-LOCAL-126 cap-reached "
                            "ledger save failed (non-fatal): %s",
                            _cap_save_exc,
                        )
                    if stop_workflow_on_soak_cap:
                        log.warning(
                            "[BatchHumoRender] BUG-LOCAL-126 soak cap reached "
                            "(%d fresh of cap %d; %d total clips on disk); "
                            "ledger persisted; raising HumoSoakCapReached "
                            "(stop_workflow_on_soak_cap=True). Queue follow-up "
                            "with resume_from_ledger=True.",
                            fresh_rendered_this_process, humo_max_lines_per_process,
                            total_clips_output,
                        )
                        raise HumoSoakCapReached(
                            lines_completed=fresh_rendered_this_process,
                            cap=humo_max_lines_per_process,
                        )
                    # Soft-cap path (Step 4, 2026-05-08): break out of
                    # the per-line loop without raising so downstream
                    # nodes (LTX, Composite, Upscale, PostProcgen)
                    # still execute on whatever was rendered. Used by
                    # end-to-end smoke validation; not safe for
                    # production resume mode until LTX grows its own
                    # resume-from-ledger scan.
                    log.warning(
                        "[BatchHumoRender] BUG-LOCAL-126 soak cap reached "
                        "(%d fresh of cap %d; %d total clips on disk); "
                        "ledger persisted; SOFT_CAP mode -- breaking "
                        "out of HuMo loop to let downstream DAG run.",
                        fresh_rendered_this_process, humo_max_lines_per_process,
                        total_clips_output,
                    )
                    report_lines.append(
                        f"  SOAK_CAP_REACHED: {fresh_rendered_this_process} "
                        f"fresh / cap {humo_max_lines_per_process}; "
                        f"downstream will run on partial output"
                    )
                    break  # exit per-line for-loop; fall through to phase 10 ledger writeback

                # 2026-04-29: per-clip incremental ledger save. Each HuMo
                # clip takes ~9-10 min on Blackwell; without per-clip
                # persistence a crash on clip 20 of 33 wipes ~3 hours of
                # state from the ledger (the post-loop write-back never
                # fires). With this save, every completed clip is on disk
                # immediately, the watcher / /otr/latest_ledger / tail
                # tools see real-time progress, and a crash leaves
                # recoverable state behind. Cost: ~5-15 ms per save via
                # BUG-108's atomic-rename + on-disk merge. ~0.3 sec
                # cumulative over a 33-clip / 5-hour render. Trivial.
                if ledger_path is not None:
                    try:
                        try:
                            from . import _otr_ledger as _OTRL_INC  # type: ignore
                        except ImportError:
                            _NODES_DIR_INC = Path(__file__).resolve().parent
                            if str(_NODES_DIR_INC) not in _sys.path:
                                _sys.path.insert(0, str(_NODES_DIR_INC))
                            import _otr_ledger as _OTRL_INC  # type: ignore
                        ledger["clips"] = list(clip_records)
                        _OTRL_INC.save_ledger_safe(ledger_path, ledger)
                    except Exception as _inc_exc:  # noqa: BLE001 -- per-clip save never blocks rendering
                        log.warning(
                            "[BatchHumoRender] per-clip ledger save failed "
                            "(non-fatal, render continues): %s",
                            _inc_exc,
                        )
            except HumoSoakCapReached:
                # Pass through: the cap raise is structured by design
                # so the orchestrator can distinguish soft-stop from
                # fault. Don't swallow it here.
                raise
            except Exception as exc:
                log.exception("[BatchHumoRender] line %s failed: %s", line_id, exc)
                report_lines.append(f"  {line_id}: FAILED ({exc})")
                # BUG-LOCAL-126 round-robin catch (2026-05-08): set a
                # flag here, but DEFER the actual cleanup until AFTER
                # the except block exits. Reason: inside the except
                # block, `exc.__traceback__` keeps the failed frame's
                # f_locals alive (humo_pos, humo_neg, humo_latent,
                # samples, etc.). gc.collect() sees them as live
                # roots, so empty_cache() can't reclaim. Once the
                # except block exits Python auto-deletes `exc` and
                # the traceback chain becomes collectable.
                _needs_oom_cleanup = (
                    cuda_hard_reset_on_oom and _is_oom_exception(exc)
                )
                _failed_line_id = line_id
                # l3-2026-05-08 telemetry: bump per-line OOM recovery
                # counter so post-mortem can answer "which lines
                # consumed the recovery cycles?" without grepping log.
                if _needs_oom_cleanup:
                    try:
                        try:
                            from . import _otr_ledger as _OTRL_OOM  # type: ignore
                        except ImportError:
                            import _otr_ledger as _OTRL_OOM  # type: ignore
                        _OTRL_OOM.bump_line_oom_recovery_count(ledger, line_id)
                    except Exception as _oom_te:  # noqa: BLE001
                        log.warning(
                            "[BatchHumoRender] oom_recovery_count bump "
                            "failed: %s", _oom_te,
                        )
            else:
                _needs_oom_cleanup = False
                _failed_line_id = None

            # Run the cleanup OUT OF the except block so the active
            # exception's traceback (and its frame locals) is released
            # first. Without this, the chain runs while the failed
            # line's tensors are still live -> no memory recovered.
            if _needs_oom_cleanup:
                log.warning(
                    "[BatchHumoRender] line %s hit OOM; running "
                    "BUG-LOCAL-126 cuda cleanup chain before next line",
                    _failed_line_id,
                )
                _hard_reset_cuda_context()
                # Round-robin Item C: persist the ledger IMMEDIATELY
                # after the cleanup chain so a subsequent C-level
                # abort (the BUG-126 failure mode) still leaves
                # cuda_hard_reset_count + oom_recovery_count durably
                # on disk. Without this, the post-mortem can't
                # answer "did the cleanup chain even fire?" if the
                # next sample fatal-aborts before the per-clip
                # ledger save runs.
                if ledger_path is not None:
                    try:
                        try:
                            from . import _otr_ledger as _OTRL_OOM_SAVE  # type: ignore
                        except ImportError:
                            import _otr_ledger as _OTRL_OOM_SAVE  # type: ignore
                        ledger["clips"] = list(clip_records)
                        _OTRL_OOM_SAVE.save_ledger_safe(ledger_path, ledger)
                    except Exception as _save_exc:  # noqa: BLE001
                        log.warning(
                            "[BatchHumoRender] post-OOM ledger save "
                            "failed (non-fatal): %s", _save_exc,
                        )

        total_ms = int((time.time() - t_start) * 1000)
        report_lines.append(
            f"Total: {total_clips_output}/{len(plan)} clip(s) in {total_ms} ms "
            f"({fresh_rendered_this_process} fresh, "
            f"{total_clips_output - fresh_rendered_this_process} resumed)"
        )
        log.info(
            "[BatchHumoRender] complete: %d/%d clips in %d ms "
            "(%d fresh, %d resumed)",
            total_clips_output, len(plan), total_ms,
            fresh_rendered_this_process,
            total_clips_output - fresh_rendered_this_process,
        )

        # ---- 10. BUG-LOCAL-089: write clips[] back to the ledger ----
        # Persist the per-clip records so VideoComposite + post-mortem
        # can resolve every clip via ledger lookup. Skipped silently
        # when the ledger came from an inline JSON blob (no source
        # path to write back to).
        # Schema l3 (2026-04-28) also stamps meta.git_commit +
        # meta.phase_ms.humo_batch + meta.schema_version so post-mortem
        # can correlate ledger contents with the code that produced
        # them.
        if ledger_path is not None and clip_records:
            try:
                from . import _otr_ledger as _OTRL  # type: ignore
            except ImportError:
                # Same dual-import pattern used elsewhere in this file
                # for sibling helpers. Tests load this module directly
                # via spec_from_file_location with no parent package.
                _NODES_DIR_2 = Path(__file__).resolve().parent
                if str(_NODES_DIR_2) not in _sys.path:
                    _sys.path.insert(0, str(_NODES_DIR_2))
                import _otr_ledger as _OTRL  # type: ignore

            try:
                ledger["clips"] = clip_records
                # Schema l3: meta block + phase timing + git commit
                _OTRL.record_phase_ms(ledger, "humo_batch", int(total_ms))
                _gc = _OTRL.lookup_git_commit(Path(__file__).resolve().parents[1])
                if _gc:
                    _OTRL.set_meta(ledger, "git_commit", _gc)
                _OTRL.save_ledger_safe(ledger_path, ledger)
                log.info(
                    "[BatchHumoRender] ledger updated: %d clip records -> %s "
                    "(schema=%s, git=%s)",
                    len(clip_records), ledger_path.name,
                    _OTRL.CURRENT_SCHEMA_VERSION, _gc or "n/a",
                )
                report_lines.append(
                    f"  ledger updated: {len(clip_records)} clip records "
                    f"(schema {_OTRL.CURRENT_SCHEMA_VERSION})"
                )
            except Exception as exc:
                log.warning(
                    "[BatchHumoRender] ledger clips write-back failed: %s", exc
                )
                report_lines.append(f"  ledger write-back FAILED: {exc}")
        elif ledger_path is None:
            log.info(
                "[BatchHumoRender] inline ledger (no path) -- "
                "skipping clips[] write-back"
            )

        return (str(clips_dir_path), total_clips_output, "\n".join(report_lines))

    @staticmethod
    def _load_ledger_with_path(ledger_arg: str) -> tuple[dict, Path | None]:
        """Accept either:
          - inline JSON string (starts with '{') -- returns (dict, None)
          - path to *_ledger.json -- returns (dict, Path)
          - path to *.mp4 (audio episode); ledger inferred via
            suffix swap (.mp4 -> _ledger.json), since that's the
            convention OTR_SignalLostVideo / EpisodeAssembler write.
            Lets us wire BatchHumoRender's ledger_json input directly
            from SignalLostVideo.video_path -- no separate ledger
            output node required.
          - empty -> auto-pick newest non-pending in the canonical
            audio dirs (BUG-LOCAL-076 fallback chain).

        Returns (ledger_dict, ledger_path_or_None). When the input is
        an inline JSON blob the path is None (no on-disk source
        exists; BUG-088 freshness check falls back to a wall-clock
        cutoff in that case).
        """
        s = (ledger_arg or "").strip()

        # Auto-pick fallback when input empty
        if not s:
            # S28 cleanbreak: dropped otr_legacy_audio_dir() from the
            # search list. Per-episode workspace is the only contract.
            audio_dirs = [
                otr_episodes_root(),
            ]
            cands = []
            for d in audio_dirs:
                if d.exists():
                    cands.extend(
                        p for p in d.glob("*_ledger.json")
                        if not p.name.startswith("pending_")
                    )
            if not cands:
                raise RuntimeError("BatchHumoRender: ledger_json empty and auto-pick found no ledger")
            p = max(cands, key=lambda x: x.stat().st_mtime)
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f), p

        if s.startswith("{"):
            return json.loads(s), None

        p = Path(s)
        # .mp4 path -> swap suffix to _ledger.json (SignalLostVideo
        # convention). BUG-LOCAL-118 hardening 2026-04-30: the
        # SignalLostVideo title sanitizer and the Ledger.rename_episode
        # joiner sometimes disagree on underscore-collapse rules
        # (e.g. .mp4 stem "..._a__20260430..." vs ledger
        # "..._a_20260430..."), so the strict stem-match lookup
        # raised a fatal error even though the right ledger was on
        # disk a few characters away. New logic tries a fallback chain:
        # (1) exact match, (2) collapsed-underscore variant,
        # (3) scan the directory for the newest ledger whose
        # episode_id appears in the .mp4 stem or vice versa.
        if p.suffix.lower() == ".mp4":
            audio_dir = p.parent
            stem = p.stem

            # 0. Layout-aware lookup (BUG-LOCAL-022, Phase G).
            # Per-episode workspace: mp4 is at
            # `output/otr/episodes/<ep>/audio/<file>.mp4`. The ledger
            # is canonically at `output/otr/episodes/<ep>/audio/<ep>_ledger.json`
            # where <ep> is the EPISODE DIR NAME, not the mp4 stem.
            # This decouples ledger discovery from the mp4 filename, which
            # may be truncated by `safe_title[:40]` in video_engine.py
            # (consult flagged stem swap is mathematically broken when
            # truncation drops characters from the title).
            #
            # Detection: audio_dir.parent.parent.name == "episodes". If
            # so, the parent dir IS the episode_id. Try exact ledger
            # filename first; if absent, glob for any *_ledger.json in
            # this audio_dir and pick the non-pending one.
            try:
                if (audio_dir.name == "audio"
                        and audio_dir.parent.parent.name == "episodes"):
                    ep_dir_name = audio_dir.parent.name
                    layout_ledger = audio_dir / f"{ep_dir_name}_ledger.json"
                    if layout_ledger.exists():
                        log.info(
                            "[BatchHumoRender] Phase G layout-aware ledger "
                            "lookup: %s (decoupled from mp4 stem %r)",
                            layout_ledger.name, stem,
                        )
                        with open(layout_ledger, "r", encoding="utf-8") as f:
                            return json.load(f), layout_ledger
                    # Glob fallback within the same audio dir: the
                    # canonical name didn't match the dir name (slug
                    # rule mismatch?), but any non-pending *_ledger.json
                    # in this dir IS this episode's ledger by
                    # construction.
                    for cand in audio_dir.glob("*_ledger.json"):
                        if cand.name.startswith("pending_"):
                            continue
                        log.info(
                            "[BatchHumoRender] Phase G layout-aware glob "
                            "lookup: %s (ep_dir=%s, mp4 stem %r)",
                            cand.name, ep_dir_name, stem,
                        )
                        with open(cand, "r", encoding="utf-8") as f:
                            return json.load(f), cand
            except Exception as exc_layout:  # noqa: BLE001
                log.debug(
                    "[BatchHumoRender] Phase G layout-aware lookup raised "
                    "(%s); falling through to legacy stem-swap tiers",
                    exc_layout,
                )

            # 1. Direct stem match (legacy, kept for back-compat with
            # mp4s that were written into the legacy flat layout AND
            # for the case where mp4 stem == ep_id exactly).
            ledger_p = audio_dir / f"{stem}_ledger.json"
            if ledger_p.exists():
                with open(ledger_p, "r", encoding="utf-8") as f:
                    return json.load(f), ledger_p

            # 2. Underscore-collapse variant (a__20260430 -> a_20260430).
            collapsed = stem
            while "__" in collapsed:
                collapsed = collapsed.replace("__", "_")
            if collapsed != stem:
                cand = audio_dir / f"{collapsed}_ledger.json"
                if cand.exists():
                    log.warning(
                        "[BatchHumoRender] BUG-LOCAL-118 underscore-mismatch "
                        "fallback: .mp4 stem %r had double underscores; "
                        "loaded matching ledger %r instead.",
                        stem, cand.name,
                    )
                    with open(cand, "r", encoding="utf-8") as f:
                        return json.load(f), cand

            # 3. Scan the directory for the newest ledger whose
            # episode_id is a substring of (or contains) the .mp4 stem.
            # Conservative: only accept a candidate that's < 1 hour old
            # so we don't bind to a stale ledger from yesterday.
            try:
                cands = list(audio_dir.glob("*_ledger.json"))
                cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                stem_norm = collapsed
                for cand in cands[:10]:
                    cand_eid = cand.stem
                    if cand_eid.endswith("_ledger"):
                        cand_eid = cand_eid[: -len("_ledger")]
                    cand_norm = cand_eid
                    while "__" in cand_norm:
                        cand_norm = cand_norm.replace("__", "_")
                    if cand_norm == stem_norm or cand_norm in stem_norm or stem_norm in cand_norm:
                        age_s = time.time() - cand.stat().st_mtime
                        if age_s > 3600:
                            continue
                        log.warning(
                            "[BatchHumoRender] BUG-LOCAL-118 fuzzy fallback: "
                            "binding to %r (age %.0fs) for .mp4 stem %r.",
                            cand.name, age_s, stem,
                        )
                        with open(cand, "r", encoding="utf-8") as f:
                            return json.load(f), cand
            except Exception as scan_exc:
                log.warning(
                    "[BatchHumoRender] BUG-LOCAL-118 directory-scan fallback "
                    "failed (%s) - falling through to error.", scan_exc,
                )

            raise RuntimeError(
                f"BatchHumoRender: derived ledger from .mp4 not found: "
                f"{ledger_p} (also tried collapsed-underscore variant + "
                f"directory scan in {audio_dir})"
            )

        # Plain ledger.json path
        if not p.exists():
            raise RuntimeError(f"BatchHumoRender: ledger path not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f), p

    @staticmethod
    def _rename_savevideo_output(
        clips_dir: Path,
        line_id: str,
        report_lines: list[str],
    ) -> None:
        """SaveVideo writes ``humo_<line_id>_NNNNN_.mp4``. Rename
        the newest match to ``<line_id>.mp4`` so concat finds it
        without globbing for raw SaveVideo prefixes."""
        import shutil as _shutil
        candidates = sorted(
            clips_dir.glob(f"humo_{line_id}_*.mp4"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            report_lines.append(f"  {line_id}: WARN SaveVideo output not found in {clips_dir}")
            return
        src = candidates[0]
        dst = clips_dir / f"{line_id}.mp4"
        try:
            _shutil.move(str(src), str(dst))
        except Exception as exc:
            report_lines.append(f"  {line_id}: WARN rename failed: {exc}")


__all__ = ["BatchHumoRender"]
