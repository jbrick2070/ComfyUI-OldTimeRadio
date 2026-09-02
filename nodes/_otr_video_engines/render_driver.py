"""In-process video render driver (A-S7.5) -- the model-agnostic render loop
that ``OTR_VideoRenderBatch`` walks to ship Subproject A.

For each shot the driver drives the registry engine lifecycle
(``assert_usable -> prepare -> render_clip -> canonicalize -> teardown``).
NO FALLBACKS (operator directive 2026-07-02: NO fallbacks / NO auto-defaults
anywhere): a HARD render failure is classified via the A-S7 retry taxonomy for
the error message and then RAISES :class:`RenderError` LOUD -- there is NO
engine swap, NO chain, NO still-image floor. The dropdown values saved in
``workflows/otr_canonical.json`` are the ONLY defaults. The frozen
``ledger['audio']`` section is read-only throughout (V-1 / the audio spine is
frozen).

In-process (V invariant: no HTTP server, no GraphBuilder): the heavy engines call
ComfyUI wrapper node classes directly via :mod:`wrapper_bridge`, so this driver
MUST run inside the ComfyUI process (``NODE_CLASS_MAPPINGS`` populated). The pure
pieces (``classify_failure`` / the fixture builder / ``assert_soak_ok``) are
CPU-tested; the live render + the A-S7.5 GPU soak are the operator gate.
UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time

from .._otr_shared import retry_taxonomy as _rt
from .._otr_shared import still_receipt as _receipt
from .._otr_shared.aspect import is_wide as _aspect_is_wide
from .._otr_speaker_role import (
    SPEAKER_ROLE_ANNOUNCER as _SPEAKER_ROLE_ANNOUNCER,
    SPEAKER_ROLE_MUSIC_OPEN as _SPEAKER_ROLE_MUSIC_OPEN,
    is_never_humo_role as _is_never_humo_role,
)
# The PURE Ghost Signal composer. Safe to import at module scope: it
# imports only hashlib/re/typing and nothing from this package, so it
# adds no cold-import weight and cannot create a cycle back to here.
from . import ghost_signal_prompt as _gsp
# The Ghost PROMPT v2 finalizer. Also safe at module scope: it imports
# only stdlib, the pure composer above and the stdlib-only banana
# route, and it never imports this module back.
from . import ghost_signal_author as _gsa
from . import motion_common as _mc
from . import registry as _vreg

_LOG = logging.getLogger("OTR.video.render_driver")

# NO FALLBACKS (operator directive 2026-07-02): FLOOR_NAMES / UNIVERSAL_FLOOR /
# SYNTH_FALLBACKS and the whole degrade-chain machinery were RIPPED (Sprint A,
# E1). Engine failure = LOUD RenderError; still_motion remains a registered
# SELECTABLE engine but has no floor role.

#: engine_id -> family (covers the soak stub + the A/cheap engines).
ENGINE_FAMILY = {
    "soak_oom_heavy": "audio_driven_face",  # synthetic soak stub (forced-OOM leg,
                                            # rebased off character_3d 2026-08-23
                                            # so the OOM contract survives the
                                            # order-4 family retirement)
    "humo": "audio_driven_face",
    "humo_1.7B": "audio_driven_face",
    "still_motion": "static_motion",
    # still_parallax UNREGISTERED 2026-06-30 (item 2 rip-out) -- removed here too.
    # 2026-08-28: image_to_video, matching the engine class. This map is
    # consulted BEFORE the live registry (see the wan_i2v note below), so a
    # stale row here would keep answering "text_to_video" forever and the
    # scene still would never attach.
    "ltx_video": "image_to_video",
    # wan_i2v RETIRED 2026-08-26 and removed HERE too: engine_family()
    # consults this static map BEFORE the live registry, so a stale row
    # keeps answering "image_to_video" forever with no error path at all.
    "mesh_stage": "image_to_video",
    # "abstract" + "station_card" entries REMOVED 2026-06-30 (C0, engines retired);
    # the "abstract" FAMILY name survives (viz_green) + is the engine_family() default.
    "viz_green": "abstract", "still_pan": "static_image_gen",   # renamed from "visualizer" 2026-06-30, item 2
    "viz_mxc_cpu": "abstract",       # OTR rainbow visualizer (2026-06-30)
    "viz_mxc_mandala": "abstract",   # Cosmic Radio Mandala, pycairo (2026-06-30)
    "viz_camera": "abstract",        # Golden Flicker camera visualizer (2026-07-05)
    "still_flat": "static_image_gen",
    # still_word (Sprint B, 2026-07-03): a still_flat sibling (word/title still,
    # held flat) -- same static_image_gen family, so it uses the explicit
    # :1044 still-init branch below (NOT _SCENE_INIT_FAMILIES).
    "still_word": "static_image_gen",
    # LTX-AV audio-input lane: the ONE ltx_audio_in engine (audio_conditioned_video;
    # the old talk/music split was removed 2026-06-26 -- routing is role-driven).
    "ltx_audio_in": "audio_conditioned_video",
}

#: The (role, engine, family) rotation covering the 3 roles + the non-3D
#: families (kept identical to scripts/otr_video_soak so the GPU soak walks the
#: same shape the shipped CPU harness proves). rip-sfx-broll (2026-07-01):
#: the retired_role_a/retired_role_b legs died with their roles; the still
#: families keep coverage via the extra announcer legs.
_PROFILES = (
    ("announcer_visual", "humo", "audio_driven_face"),
    # The text_to_video seat, RE-SEATED 2026-08-28: ltx_video declared its
    # true family (image_to_video) and left this chair, exactly as wan_i2v
    # left the image_to_video chair when the 14B retired. The remaining LOCAL
    # text_to_video engine takes it, so the rotation still walks one engine
    # per family and the family does not silently drop out of soak coverage.
    ("music_visual", "animatediff15_v3_haunted_video", "text_to_video"),
    # The image_to_video seat. It was `wan_i2v` until the 14B was retired on
    # 2026-08-26; `wan_ti2v` (the 5B) takes it rather than the row being
    # dropped, because dropping it would leave the soak walking NO
    # image_to_video lane at all -- a live family with several engines
    # (mesh_stage, ltx_8gb, fastwan_8gb, wan_ti2v, minimax_h3_video) would
    # have gone unexercised, and the fixture exists precisely to walk one row
    # per family. The 5B declares character_video and resolves to
    # image_to_video, so the shape is preserved exactly.
    ("character_video", "wan_ti2v", "image_to_video"),
    ("character_video", "still_motion", "static_motion"),
    ("music_visual", "still_flat", "static_image_gen"),
    ("announcer_visual", "still_pan", "static_image_gen"),
)
#: The forced-OOM group: the synthetic ``soak_oom_heavy`` stub, standing in for
#: a LIVE heavy family (audio_driven_face -- humo's). Rebased off character_3d
#: on 2026-08-23 so the contract does not die with that family's order-4
#: retirement: what it proves was never about 3D, it is that under NO FALLBACKS
#: a forced OOM RAISES a named RenderError -- the soak asserts the raise (no
#: trail, no swap).
_HEAVY_OOM = ("character_video", "soak_oom_heavy", "audio_driven_face")
#: The heavy engines the soak forces to OOM on the injected shot -- the
#: LOUD-failure contract leg asserts the resulting RenderError.
OOM_ENGINES = frozenset({"soak_oom_heavy", "humo", "humo_1.7B"})
#: The M1 frozen master-audio PCM marker the soak threads through + asserts is
#: byte-identical after the run (the decision layer must never touch audio).
FROZEN_AUDIO_SHA = "21aa71f6a4e5master_audio_pcm_marker"
_GOOGLE_SILENT_TEXT_PROVIDERS = frozenset({
    "google_veo_video",
    "google_omni_video",
})
#: rip-sfx (2026-08-06): the google_vid_sfx_* rows are retired, so every
#: surviving Google provider is a silent text provider and the prompt-provider
#: set narrows to the silent set.
_GOOGLE_PROVIDER_PROMPT_ENGINES = _GOOGLE_SILENT_TEXT_PROVIDERS


class OomSignal(RuntimeError):
    """Stand-in for a render-time CUDA OOM (a HARD failure) -- the soak forces it
    on the mid-episode injected stub shot to prove the LOUD-raise contract."""


class RenderFloorError(RuntimeError):
    """A radio-open beat rendered on the procgen/still floor instead of an LTX
    engine (BUG-LOCAL-413 strict mode -- see :func:`check_ltx_open_health`).
    NOTE: despite the historical name, this has nothing to do with the deleted
    fallback-chain floor machinery."""


# RenderError MOVED to the dependency-leaf `render_errors` (2026-07-29,
# WIRE-W2) so `otr_shot_lock` can import DeferredImageGapError without
# importing this module. Re-exported here, deliberately: every existing
# `from .render_driver import RenderError` keeps working, and because it is a
# re-export rather than a copy the class objects are IDENTICAL, so `except`
# and `isinstance` behave the same whichever path a caller imported by.
try:  # pragma: no cover -- package import
    from .render_errors import DeferredImageGapError, RenderError
except ImportError:  # pragma: no cover -- flat test imports
    from render_errors import DeferredImageGapError, RenderError  # type: ignore


class FamilyInputGap(RuntimeError):
    """An engine's FAMILY requires request inputs this request cannot satisfy:
    e.g. ``lipsync_overlay`` needs ``base_clip_ref`` that a ``character_3d``
    request lacks. Classified DEPENDENCY_MISSING; NO FALLBACKS -- the shot
    fails LOUD instead of feeding a mismatched request to the engine."""


class SoakError(AssertionError):
    """An A-S7.5 soak invariant was violated (the soak FAILED)."""


# --------------------------------------------------------------------------- #
# Pure helpers (CPU-tested)
# --------------------------------------------------------------------------- #
def classify_failure(exc):
    """Map a render exception to a HARD :class:`FailureKind` (all escalate)."""
    if isinstance(exc, OomSignal):
        return _rt.FailureKind.OOM
    name = type(exc).__name__
    if name in ("EngineUnusable", "WrapperNodeMissing", "LookupError",
                "KeyError", "FileNotFoundError", "FamilyInputGap"):
        return _rt.FailureKind.DEPENDENCY_MISSING
    if name == "GraphExecutionError":
        return _rt.FailureKind.INVALID_DAG
    return _rt.FailureKind.CRASH_BEFORE_LOAD


def engine_family(name, default=None):
    """The family for a (possibly unregistered) engine name."""
    if name in ENGINE_FAMILY:
        return ENGINE_FAMILY[name]
    if _vreg.is_registered(name):
        return getattr(_vreg.get_engine(name), "family", default) or default
    return default or "abstract"


def _required_inputs_for_engine(engine_name, fam=None):
    """Family requirements plus a registered engine's stricter contract.

    ``FAMILY_REQUIRED_INPUTS`` is the broad request shape. Some engines inside a
    family are stricter (Seedance needs an init still in addition to the
    audio-conditioned-video text/audio minimum), so preflight must validate the
    concrete engine contract before the adapter receives the request.
    """
    from .schemas import FAMILY_REQUIRED_INPUTS, REQUIRED_INPUT_TOKENS

    name = str(engine_name or "")
    family = fam if fam is not None else engine_family(name, "")
    required = set(FAMILY_REQUIRED_INPUTS.get(family, ()))
    if name and _vreg.is_registered(name):
        required.update(str(t) for t in (
            getattr(_vreg.get_engine(name), "required_inputs", ()) or ()))
    return tuple(t for t in REQUIRED_INPUT_TOKENS if t in required)


def build_soak_fixture(n_beats=40, oom_index=None):
    """Build a synthetic ``ledger['video']`` section + meta (pure; identical
    shape to scripts/otr_video_soak.build_soak_fixture).

    ``oom_index=None`` (default since the 2026-07-02 NO-FALLBACKS rip) builds a
    CLEAN all-profiles fixture; an integer injects the synthetic ``soak_oom_heavy``
    heavy-engine stub at that index for the LOUD-failure contract leg (the
    forced OOM must RAISE a named RenderError -- no trail, no swap)."""
    if oom_index is not None and not 0 <= oom_index < n_beats:
        raise ValueError("oom_index %d out of range for %d beats"
                         % (oom_index, n_beats))
    shots = []
    for i in range(n_beats):
        role, engine, family = _HEAVY_OOM if i == oom_index \
            else _PROFILES[i % len(_PROFILES)]
        shots.append({
            "shot_id": "shot_%04d" % i, "beat_id": "b%04d" % i, "role": role,
            "engine_id": engine, "family": family, "group_id": "grp_%04d" % i,
            "target_frame_count": 25, "degradation_trail": [],
        })
    section = {"video_revision": 1, "fps": 25, "shots": shots}
    meta = {"oom_shot_id": "shot_%04d" % oom_index, "oom_index": oom_index,
            "n_beats": n_beats}
    return section, meta


def build_full_ledger(section):
    """Wrap a video section in a full ledger with a FROZEN audio section."""
    return {"audio": {"master_audio_sha256": FROZEN_AUDIO_SHA,
                      "ledger_frozen": True},
            "video": section}


#: The latent-grid divisor every declared render canvas must respect. The
#: ``ltx_av`` comment records 1280x720 failing exactly this gate live.
_DECLARED_CANVAS_GRID = 32


def declared_render_canvas(engine_id):
    """The ``(w, h)`` an engine DECLARES it renders at, or ``None``.

    THE DEFECT THIS CLOSES (O1). ``build_request_from_shot`` overwrites the
    canvas to the shared landscape default for every non-face family, with
    deliberate per-engine branches after it for ``ltx_video`` and
    ``ltx_audio_in`` and NONE for ``ltx_8gb``. So the 8 GB tier rendered at
    1472x832 -- 8.3x the pixels, on the tier that exists because 8 GB cannot
    afford them -- while its own profile asked for 512x288.

    THE CANVAS IS READ FROM THE ADAPTER, NOT FROM THE LEDGER, and that is the
    whole point. The O1 canvas judgment enumerated five channels that name a
    render canvas and found this tier's -- ``render.canvas_w/h`` travelling to
    ``video.canonical_canvas`` -- to be the one DEAD channel, then ruled: the
    engine "declares its render canvas STATICALLY ... not an env var, not a
    ledger read, not a fourth inline branch."

    The first draft of this function DID read the ledger stamp, and the
    pre-push panel killed it with a case the design had not considered: the
    coverage-matrix harness routes ``ltx_8gb`` onto the CANONICAL workflow
    through profile ``role_overrides`` and never copies a canvas, so a
    ledger-read design inherits that workflow's 26:15 canvas and must either
    pillarbox or refuse an episode that renders today. A declaration cannot be
    displaced by where it is pointed. What the profile channel still owes is a
    DRIFT GUARD, not authority -- a test pinning the profile's canvas equal to
    the declaration, which is where that check now lives.

    Validates the DECLARATION itself: positive and /32 on both axes. This is a
    code-integrity check in the shape of ``FrameContract.__post_init__``, not an
    operator-facing gate -- an adapter that declares a canvas its own latent
    grid cannot accept is a bug in the adapter, and it must not reach a render.

    Pure: no ledger, no environment, no I/O. Called from
    ``build_request_from_shot`` so the request carries the declared canvas, and
    once in ``render_beat_coverage`` before ``BeatSession`` opens so a broken
    declaration cannot cost a 6.34 GiB load first.
    """
    name = str(engine_id or "")
    if not name or not _vreg.is_registered(name):
        return None
    try:
        declared = getattr(_vreg.get_engine(name), "render_canvas", None)
    except Exception:  # noqa: BLE001 -- an unbuildable engine declares nothing
        return None
    if declared is None:
        return None
    # A STRING IS INDEXABLE, which is why the shape is checked before the
    # values: ``"512x288"[0:2]`` is ``"5", "1"``, so a stringly-typed
    # declaration would sail through int() as 5x1 and be refused for the wrong
    # reason, naming the latent grid instead of the real mistake.
    if (isinstance(declared, (str, bytes))
            or not isinstance(declared, (tuple, list))
            or len(declared) != 2):
        raise RenderError(
            "CANVAS: engine %r declares render_canvas=%r, which is not a "
            "(width, height) pair. NO FALLBACK -- a malformed declaration must "
            "not fall through to the shared landscape default, which is the "
            "defect this seam exists to close." % (name, declared))
    try:
        width, height = (int(declared[0]), int(declared[1]))
    except (TypeError, ValueError) as exc:
        raise RenderError(
            "CANVAS: engine %r declares render_canvas=%r, which is not a "
            "(width, height) pair. NO FALLBACK -- a malformed declaration must "
            "not fall through to the shared landscape default, which is the "
            "defect this seam exists to close." % (name, declared)) from exc
    problems = []
    if width < 1 or height < 1:
        problems.append("both dimensions must be positive")
    if width % _DECLARED_CANVAS_GRID or height % _DECLARED_CANVAS_GRID:
        problems.append("both dimensions must be divisible by %d (the latent "
                        "grid)" % _DECLARED_CANVAS_GRID)
    if problems:
        raise RenderError(
            "CANVAS: engine %r declares an unrenderable render_canvas %dx%d -- "
            "%s. NO FALLBACK." % (name, width, height, "; ".join(problems)))
    return (width, height)


def build_request(shot, assets, frame_count, canvas=None):
    """A SCHEMA-VALID ``VideoRequest`` dict per shot (deterministic: the seed
    is keyed to the shot id so render-twice is identical -- V-7).

    W7-pre builder migration (3D plan 7.0, code-verified gap): the emitted
    dict passes ``VideoRequest.model_validate`` -- the old extras
    ``init_w``/``init_h`` are GONE (the adapters' aspect hint defaulted to the
    canvas dims anyway = an identity transform; hand-built requests may still
    carry the hint, the builders just never emit it), ``role`` /
    ``family_hint`` / ``profile_id`` are emitted, and observability stamps
    ride the REAL ``observability`` field -- never top-level underscore
    extras."""
    assets = assets or {}
    portrait = assets.get("init_image", "")
    audio = assets.get("audio_ref", "")
    sid = shot["shot_id"]
    try:
        idx = int(sid.rsplit("_", 1)[-1])
    except ValueError:
        idx = 0
    seed = (idx * 1009 + 7) & 0x7FFFFFFF
    cw, ch = (canvas or (480, 832))
    family = engine_family(str(shot.get("engine_id") or ""),
                           shot.get("family")) or "abstract"
    return {
        "shot_id": sid, "request_id": sid,
        "role": str(shot.get("role") or ""),
        "family_hint": family,
        "profile_id": str(shot.get("profile_id") or ""),
        "text_prompt": "a 1940s radio studio, on air sign illuminated, period broadcast set",
        "asset_refs": {"init_image": portrait} if portrait else {},
        "conditioning_refs": {},
        "audio_ref": {"path": audio} if audio else None,
        "base_clip_ref": None,
        "timing": {"target_frame_count": int(frame_count)},
        "canvas": {"w": int(cw), "h": int(ch), "fps": 25, "aspect_policy": "pad"},
        "seed_bundle": {"request_seed": seed},
        "observability": {},
    }


# --------------------------------------------------------------------------- #
# Per-beat audio slice from the frozen master mix (read-only).
#
# HuMo needs an ``audio_ref`` WAV per beat.  When the ledger carries no
# per-line ``*_wav_path`` (the common case when individual TTS clips are not
# re-exported as standalone files), we slice the FROZEN master mix by the
# beat's ``[start_s, start_s+dur_s]`` timing that OTR_EpisodeAssembler
# already stamped onto every ``lines[]`` entry.  The master file is opened
# read-only by ffmpeg (``-i``); output goes to a dedicated temp directory so
# the master is NEVER mutated (V-1 / audio spine frozen).
#
# CACHE KEYS -- the 7.3 slice/curve SPLIT (don't over-key the cheap WAV):
# the SLICE key binds the master CONTENT hash
# (``ledger['audio']['master_audio_sha256']``) + start_s + dur_s + sample
# rate + channels + slicer version -- the shipped path-only key
# under-invalidated when a NEW master landed at the SAME path.  The CURVE
# key (the W7 Rhubarb driver's artifact) DERIVES from the slice key and
# additionally binds line_id + fps + driver version + viseme-mapping hash +
# onset policy -- driver-side concerns that must never churn the cheap WAV.
# The HuMo 44.1 kHz mono slice semantics are UNCHANGED; the driver's 16 kHz
# input is a DOWNSAMPLE OF THE SLICE, never a re-slice of the master.
# --------------------------------------------------------------------------- #

#: Bumps when the SLICE ffmpeg recipe changes (codec/rate/channels/trim
#: semantics) -- part of the slice cache key, so old cached WAVs invalidate.
#: v3 (WIRE-W4c): the recipe grew a silence pad, so every v2 WAV on disk
#: describes a different contract and must not be served to a v3 caller.
SLICER_VERSION = "3"

#: The HuMo slice recipe constants (UNCHANGED semantics -- 7.3: HuMo's
#: slicer is NOT changed; they are named so the cache key can bind them).
_SLICE_SAMPLE_RATE = 44100
_SLICE_CHANNELS = 1


def slice_cache_key(master_hash, start_s, dur_s, *,
                    sample_rate=_SLICE_SAMPLE_RATE,
                    channels=_SLICE_CHANNELS,
                    slicer_version=SLICER_VERSION,
                    master_path="", pad_tail_s=0.0):
    """The SLICE cache key (3D plan 7.3): master CONTENT hash + timing +
    rate + channels + slicer version. ``master_path`` participates ONLY when
    ``master_hash`` is empty (the legacy hashless caller keeps the shipped
    path-keyed behavior instead of all colliding on one key). Pure; 16-hex.

    ``pad_tail_s`` (WIRE-W4c) is IN the key because it is part of the OUTPUT,
    not part of the request: two segments can copy the identical source
    interval and owe different amounts of trailing silence, and a key that
    ignored the pad would serve the first one's WAV to the second -- an
    under-length conditioning track wearing a cache hit."""
    ident = str(master_hash or "") or ("path:%s" % master_path)
    return hashlib.sha256(
        ("slice|v%s|%s|%.6f|%.6f|ar%d|ac%d|pad%.6f"
         % (slicer_version, ident, float(start_s), float(dur_s),
            int(sample_rate), int(channels), float(pad_tail_s or 0.0))
         ).encode("utf-8")
    ).hexdigest()[:16]


# `curve_cache_key` was removed 2026-08-28. It was the exported key for the
# W7 Rhubarb->ARKit CURVE cache -- a cache that was never built: no production
# caller, no curve artifact on disk, and the 3D lane it belonged to is retired.
# The audio SLICE half above (`slice_cache_key`, `SLICER_VERSION`) is live and
# untouched; only the curve half was a public contract for nothing.


def _slicer_ffmpeg_bin():
    """The CONFIGURED ffmpeg, then PATH (WIRE-W4c).

    This module hard-coded the bare literal ``"ffmpeg"`` while
    ``otr_credits_roll`` already honoured ``OTR_FFMPEG`` -- so on a box where
    ffmpeg is configured but not on PATH the credits rendered and the audio
    slice silently returned ``""``, which reads downstream as "this beat has
    no voice line" rather than as "this box cannot slice". Returns the bare
    name when nothing resolves, so the failure is still ffmpeg's own and still
    LOUD at the call site rather than an import-time raise."""
    cand = (os.environ.get("OTR_FFMPEG") or "").strip()
    if cand and (os.path.isfile(cand) or shutil.which(cand)):
        return cand
    return shutil.which("ffmpeg") or "ffmpeg"


def _slice_master_audio(master_path, start_s, dur_s, master_hash="",
                        pad_tail_s=0.0):
    """ffmpeg-slice ``[start_s, start_s+dur_s]`` from the FROZEN master into a
    temp WAV, optionally APPENDING ``pad_tail_s`` of silence.  Read-only
    (``-i`` only, never ``-o`` on the master).  Returns the temp path on
    success, ``""`` on failure (LOUD warning logged).

    ``pad_tail_s`` (WIRE-W4c, r4/A4) exists so a coverage-planned segment's
    conditioning WAV can be ``render_frames`` long while carrying only
    ``render_frames - trim_tail`` of real speech. The pad is genuine silence,
    never the master's continuation: the audio encoder sees the WHOLE waveform
    before a single frame is sampled, so speech from the next beat sitting in
    a tail that gets trimmed still conditions the frames that survive.

    ``apad`` also covers the other end of the same problem for free: a window
    that runs past the END of the master pads to length instead of emitting a
    short WAV.

    The output file is cached by :func:`slice_cache_key` so render-twice is
    deterministic without re-running ffmpeg AND a new master at the same
    path invalidates (the content hash is the identity). ``master_hash`` is
    fed from ``ledger['audio']['master_audio_sha256']`` by
    :func:`build_request_from_shot`; a hashless call keeps the legacy
    path-keyed behavior with a LOUD warning."""
    if not master_hash:
        _LOG.warning("[OTR.render_driver] _slice_master_audio called WITHOUT "
                     "the master content hash -- slice cache falls back to "
                     "the path-keyed identity (under-invalidates on a new "
                     "master at the same path); thread "
                     "ledger['audio']['master_audio_sha256'] in")
    key = slice_cache_key(master_hash, start_s, dur_s,
                          master_path=master_path,
                          pad_tail_s=pad_tail_s)
    from ._tmp import _in_tree_tmp_dir
    _base = _in_tree_tmp_dir() or tempfile.gettempdir()
    tmp_dir = os.path.join(_base, "audio_slices")
    try:
        os.makedirs(tmp_dir, exist_ok=True)
    except OSError as exc:
        _LOG.warning("[OTR.render_driver] _slice_master_audio: cannot create "
                     "tmp dir %s: %s", tmp_dir, exc)
        return ""
    out = os.path.join(tmp_dir, "slice_%s.wav" % key)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out                       # deterministic cache hit
    cmd = [
        _slicer_ffmpeg_bin(), "-y",
        # INPUT options: seek + read exactly the source interval.
        "-ss", "%.6f" % float(start_s),
        "-t",  "%.6f" % float(dur_s),
        "-i",  master_path,
        "-vn", "-c:a", "pcm_s16le",
        "-ar", str(_SLICE_SAMPLE_RATE), "-ac", str(_SLICE_CHANNELS),
    ]
    pad = float(pad_tail_s or 0.0)
    if pad > 0:
        # OUTPUT options: `apad` emits silence forever after the source ends
        # and the second `-t` truncates the result to the contracted length.
        # The pair is deliberate -- `apad` alone never terminates, and a `-t`
        # alone would just re-cut the source interval.
        cmd += ["-af", "apad", "-t", "%.6f" % (float(dur_s) + pad)]
    cmd.append(out)
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
    except Exception as exc:             # noqa: BLE001 - LOUD, never crash
        _LOG.warning("[OTR.render_driver] _slice_master_audio FAILED "
                     "(%s@%.3f+%.3fs): %s",
                     os.path.basename(master_path), start_s, dur_s, exc)
        return ""
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        _LOG.warning("[OTR.render_driver] _slice_master_audio: empty output "
                     "(%s@%.3f+%.3fs)", os.path.basename(master_path),
                     start_s, dur_s)
        return ""
    return out


# --------------------------------------------------------------------------- #
# Per-shot request builder (the REAL episode path). Resolves the character
# portrait + the per-beat voice audio + the shared visual prompt from the
# ShotLock-planned ledger, keyed to ONE shot. Additive: the soak/global-assets
# path (build_request) is untouched. Pure (reads the ledger, never writes; the
# frozen audio section is only ever read) and CPU-tested.
# --------------------------------------------------------------------------- #
def _seed_from_hash(request_hash, shot_id):
    """Deterministic 31-bit render seed from the shot's request_hash. Real
    shot_ids are not numeric (``shot_b001``), so build_request's index trick
    would collapse every per-shot seed to 0 -- V-7 keys the seed to the stable
    request hash so render-twice is identical AND each beat differs."""
    src = str(request_hash or shot_id or "")
    return int(hashlib.sha256(src.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def _portrait_index(ledger):
    """``{char_id: portrait_path}`` from ``ledger['images']['images']`` (the
    OTR_ImageGenDispatcher write-back; each entry is keyed by ``object_id``).

    KIND-FILTERED (3D image streams, 2026-06-21): ONLY ``kind=="portrait"`` rows
    (empty kind tolerated for legacy ledgers) feed the HuMo / portrait lookup.
    Without this, a ``mesh_fodder`` row (which carries a ``char_id``) or a
    ``scene_character`` row would leak into the portrait map and HuMo could pick
    up the clean 3D fodder instead of the cinematic portrait."""
    out = {}
    imgs = ((ledger or {}).get("images") or {}).get("images") or []
    for im in imgs:
        if not isinstance(im, dict):
            continue
        _kind = str(im.get("kind") or "")
        if _kind and _kind != "portrait":
            continue
        cid = str(im.get("object_id") or im.get("char_id") or "")
        path = str(im.get("path") or "")
        if cid and path:
            out.setdefault(cid, path)
    return out


def _portrait_style_index(ledger):
    """``{object_id: radio_host_style}`` over the portrait rows of
    ``ledger['images']['images']`` (radio-face logic 2026-07-04). The image node
    stamps ``radio_host_style`` on the announcer portrait ("console_face" when a
    HuMo/audio_driven_face engine will animate it, else the FACELESS
    "radio_object"); the dispatcher copies it onto the ledger row. The render
    guard reads it to fail CLOSED before feeding a faceless still to a HuMo
    announcer init. Old/frozen ledgers carry no field -> absent (missing), which
    the guard treats as NOT console_face. Pure, tolerant."""
    out = {}
    imgs = ((ledger or {}).get("images") or {}).get("images") or []
    for im in imgs:
        if not isinstance(im, dict):
            continue
        cid = str(im.get("object_id") or im.get("char_id") or "")
        style = str(im.get("radio_host_style") or "")
        if cid and style:
            out.setdefault(cid, style)
    return out


def _still_index(ledger):
    """``{beat_id: still_path}`` over ``ledger['images']['images']`` rows with
    ``kind=scene_*`` (the still-spine ST-3 dispatcher write-back). The NEWEST
    row for a beat wins (a cache-hit materialization appends a fresh row whose
    path is the current episode's copy). Pure, tolerant."""
    out = {}
    plate = {}
    imgs = ((ledger or {}).get("images") or {}).get("images") or []
    for im in imgs:
        if not isinstance(im, dict):
            continue
        _kind = str(im.get("kind") or "")
        if not _kind.startswith("scene_"):
            continue
        bid = str(im.get("beat_id") or "")
        path = str(im.get("path") or "")
        if not (bid and path):
            continue
        # 3D image streams (2026-06-21): the subject-free BACKGROUND PLATE wins
        # over any co-existing cinematic scene_* still for the same beat (a 3D
        # beat mints ONLY fodder+plate, but if a stale scene_* row co-exists the
        # plate is the correct background for the mesh composite). Verify-at-
        # build #3 -- priority, not last-write-wins.
        if _kind == "scene_background_plate":
            plate[bid] = path
        else:
            out[bid] = path
    out.update(plate)
    return out


def _mesh_fodder_index(ledger):
    """``{subject_id: fodder_still_path}`` over ``ledger['images']['images']``
    rows with ``kind=='mesh_fodder'`` (the 3D-image-streams clean-subject mint).
    Keyed by ``mesh_subject_id`` (``char_id`` | ``object_id``) PLUS ``beat_id``
    as a secondary key, so a beat-scoped object fodder is resolvable when no
    subject id is known. Any ``requires_mesh_fodder`` engine consumes THIS --
    NEVER the cinematic scene still (which would mesh the whole environment ->
    the clay blob). Pure, tolerant; the NEWEST row per key wins (mirrors
    :func:`_still_index` so a cache-hit materialization's fresh row is used)."""
    out = {}
    imgs = ((ledger or {}).get("images") or {}).get("images") or []
    for im in imgs:
        if not isinstance(im, dict):
            continue
        if str(im.get("kind") or "") != "mesh_fodder":
            continue
        path = str(im.get("path") or "")
        if not path:
            continue
        for key in (im.get("mesh_subject_id"), im.get("object_id"),
                    im.get("char_id"), im.get("beat_id")):
            k = str(key or "")
            if k:
                out[k] = path
    return out


def _mesh_fodder_row_index(ledger):
    """Index the newest fodder row by its subject and beat keys."""
    out = {}
    imgs = ((ledger or {}).get("images") or {}).get("images") or []
    for im in imgs:
        if not isinstance(im, dict):
            continue
        if str(im.get("kind") or "") != "mesh_fodder":
            continue
        if not str(im.get("path") or ""):
            continue
        for key in (im.get("mesh_subject_id"), im.get("object_id"),
                    im.get("char_id"), im.get("beat_id")):
            k = str(key or "")
            if k:
                out[k] = im
    return out


def _still_spine_row_key(value):
    """Return a filesystem-safe, stable component for a materialized still."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return value.strip("._") or "still"


def _still_spine_episode_id(ledger):
    """Resolve the durable episode id used by the active render handoff."""
    return str(
        (ledger or {}).get("episode_id")
        or ((ledger or {}).get("meta") or {}).get("episode_id")
        or ""
    ).strip()


def _still_spine_materialize_row(row, stills_root):
    """Repair one image row to an active-episode file, if its source exists.

    ``path`` is authoritative only when it is a real file under the active
    episode's stills directory. A content-addressed ``pool_path`` (or a stale
    path from a renamed pending episode) is copied into that directory and the
    row is updated. This is intentionally local and deterministic; it never
    selects a sibling episode or substitutes a different beat's still.
    """
    if not isinstance(row, dict):
        return ""
    root = os.path.abspath(str(stills_root))
    current = os.path.abspath(str(row.get("path") or "")) if row.get("path") else ""
    try:
        if current and os.path.isfile(current):
            if os.path.commonpath((root, current)) == root:
                return current
    except (OSError, ValueError):
        pass
    source = ""
    for candidate in (current, str(row.get("pool_path") or "")):
        if candidate and os.path.isfile(candidate):
            source = os.path.abspath(candidate)
            break
    if not source:
        return ""
    object_id = _still_spine_row_key(
        row.get("object_id") or row.get("beat_id") or row.get("char_id")
    )
    content_hash = str(
        row.get("content_hash") or row.get("sha256") or row.get("hash") or ""
    ).strip()
    if not content_hash:
        content_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    destination = os.path.join(
        root, "%s_%s.png" % (object_id, content_hash[:12])
    )
    os.makedirs(root, exist_ok=True)
    if not os.path.isfile(destination):
        shutil.copyfile(source, destination)
    if not os.path.isfile(destination):
        return ""
    row["path"] = destination
    row["materialized_path"] = destination
    return destination


def _still_spine_image_rows(ledger):
    rows = ((ledger or {}).get("images") or {}).get("images") or []
    return [row for row in rows if isinstance(row, dict)]


def _still_spine_row_for_beat(rows, beat_id):
    """Choose the same plate-over-scene precedence as ``_still_index``."""
    matches = [
        row for row in rows
        if str(row.get("beat_id") or "") == str(beat_id or "")
        and str(row.get("kind") or "").startswith("scene_")
    ]
    if not matches:
        return None
    plates = [row for row in matches
              if str(row.get("kind") or "") == "scene_background_plate"]
    return (plates or matches)[-1]


def _still_spine_row_for_portrait(rows, char_id):
    matches = [
        row for row in rows
        if str(row.get("kind") or "") in ("", "portrait")
        and str(row.get("object_id") or row.get("char_id") or "")
        == str(char_id or "")
    ]
    return matches[-1] if matches else None


def _still_spine_row_for_mesh(rows, beat_id, char_id):
    """The mesh row a beat owns, chosen by IDENTITY and never by ABSENCE.

    Four keys are legitimate because the recurring radio mesh deliberately
    shares ONE subject across beats -- that breadth is the contract, and it is
    why this lookup is looser than the beat-exact scene lookup above.

    What was NOT the contract: ``keys`` was built from four possibly-empty row
    fields and probed with ``str(char_id or "") in keys``. A music beat probes
    with ``char_id == ""`` and a mesh row stores ``char_id: ""``, so ``"" in
    keys`` was True for EVERY mesh row and the loop returned the first one it
    reached -- an answer to a question nobody asked. Measured on the failing
    2026-08-12 leg's own rows: asking for ``b000_music_open`` returned
    ``meshfodder_music_opening_001``.

    That is why the still spine's SCENE half failed loud on the beat-id split
    while the MESH half stayed quiet and served a plausible wrong image. Both
    sides are guarded here -- emptiness is dropped from ``keys`` AND an empty
    probe is refused -- because either one alone still lets absence match
    absence.
    """
    cid = str(char_id or "").strip()
    bid = str(beat_id or "").strip()
    if not cid and not bid:
        return None                     # no identity asked for, none answered
    for row in reversed(rows):
        if str(row.get("kind") or "") != "mesh_fodder":
            continue
        keys = {
            str(row.get("mesh_subject_id") or "").strip(),
            str(row.get("object_id") or "").strip(),
            str(row.get("char_id") or "").strip(),
            str(row.get("beat_id") or "").strip(),
        }
        keys.discard("")
        if (cid and cid in keys) or (bid and bid in keys):
            return row
    return None


def _still_spine_manifest_object_ids(images):
    receipt = images.get("required_scene_targets")
    if not isinstance(receipt, list):
        return set()
    return {
        str(row.get("object_id") or "")
        for row in receipt if isinstance(row, dict)
    }


def sanctioned_gap_beat_ids(ledger):
    """Beat ids whose required still was REFUSED by the image model.

    ONE PREDICATE, TWO CALL SITES (2026-08-28). The still-spine validator uses
    it to decline to raise, and ``run_episode`` uses it to decline to render.
    Those two decisions must never disagree: a beat the validator waves
    through but the loop still tries to render would die in
    ``cheap_families.render_clip`` on the same missing file, one layer later
    and with a worse message.

    Read from the RECEIPT, never from the absence of an image row. A beat with
    no still and no gap row is an unexplained absence and must keep failing --
    that distinction is the entire point of the sanctioned-gap work, and
    inferring a gap from absence is the defect the 2026-08-28 review panel
    caught in the first draft.

    WHOLE-SHOT RULE: one refused target gaps its whole beat. A beat can own a
    scene still plus several jump-segment stills; rendering it with some of
    them missing would produce a clip that silently drops coverage, so the
    beat is floored in full instead.
    """
    receipt = ((ledger.get("images") or {}).get("required_scene_targets"))
    if not isinstance(receipt, list):
        return set()
    return {
        str(row.get("beat_id") or "")
        for row in receipt
        if _receipt.is_sanctioned_gap(row) and row.get("beat_id")
    }


def sanctioned_gap_object_ids(ledger):
    """Object ids of refused targets -- used to IGNORE stale image rows.

    A refusal leaves no new file, but a PRIOR revision's file for the same
    object can still be sitting on disk and in ``ledger["images"]["images"]``.
    Without this, ``_still_spine_row_for_beat`` would hand back that stale row
    and the spine would validate a still this dispatch never produced.
    """
    receipt = ((ledger.get("images") or {}).get("required_scene_targets"))
    if not isinstance(receipt, list):
        return set()
    return {
        str(row.get("object_id") or "")
        for row in receipt
        if _receipt.is_sanctioned_gap(row) and row.get("object_id")
    }


def _still_spine_requires_scene(shot, engine_id, family):
    # CORRECTED 2026-08-12 (lane 20 QA). An earlier draft of this comment said
    # `minimax_h3_audio_in` was "deliberately NOT in this list" and therefore
    # kept a portrait spine. That was FALSE about the code below it: this
    # function also returns True for any non-face engine that declares
    # `init_image` (and for the whole `audio_conditioned_video` family), so the
    # H3 audio-in lane takes the scene SPINE exactly like `ltx_audio_in`.
    #
    # That is fine and intended -- the spine decides which stills get MINTED,
    # and a per-beat scene still is useful to have. The thing that had to be
    # fixed is one level up, where `_engine_scene_init_required` OVERWRITES
    # init_image with that scene still; both audio-in lanes are excluded there
    # so the reference the model lip-syncs stays a face.
    if str(engine_id or "") in (
            "still_pan", "still_flat", "still_word", "ltx_audio_in"):
        return True
    if family in _SCENE_INIT_FAMILIES:
        return True
    if family == "audio_driven_face":  # (character_3d token retired 2026-08-23)
        return False
    if "init_image" in _required_inputs_for_engine(engine_id, family):
        return True
    try:
        eng = _vreg.get_engine(engine_id) if _vreg.is_registered(engine_id) else None
        return bool(eng and getattr(eng, "provider_side", False)
                    and getattr(eng, "accepts_still", False))
    except Exception:  # noqa: BLE001 -- unknown engine fails elsewhere
        return False


def _jump_still_requests_for_shot(shot):
    """Every jump-segment still this shot owes, READ off the durable stamp.

    Never re-derived (2026-07-25, chunk 4). ShotLock minted these ids from the
    beat_id it was building, and this end READS them off the durable row -- so
    recomputing an id here would be a second derivation of one string, which is
    the mirror class chunk 1a collapsed. (This once said the id "passes through
    ``_canonical_visual_beat_id`` on the way here". That translation was deleted
    with PBUG-20260811-02: there is now exactly ONE id, the shot's own, which is
    what makes the never-re-derive rule cheap to keep.) A stamp that exists but is shaped wrong is
    TERMINAL: this pass is the last gate before a GPU render, and a request it
    cannot read is a still it cannot prove.
    """
    rows = shot.get("jump_still_requests")
    if rows is None:
        return ()
    if not isinstance(rows, list):
        raise RenderError(
            "still-spine handoff: shot %s carries a malformed "
            "jump_still_requests stamp (%s)"
            % (shot.get("shot_id"), type(rows).__name__))
    out = []
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("object_id") or ""):
            raise RenderError(
                "still-spine handoff: shot %s carries a jump-still request "
                "with no object_id, so its segment still can never be proven"
                % (shot.get("shot_id"),))
        out.append(row)
    return tuple(out)


def jump_segment_still_path(ledger, shot, segment_index):
    """The init image for jump segment ``segment_index``, BY OBJECT ID.

    THE HARD RULE OF CHUNK 6 (2026-07-26; found by the chunk-4 QA panel and
    carried forward deliberately). A per-segment render must NEVER resolve its
    init image through :func:`_still_index`. That index selects rows whose
    ``kind`` starts with ``scene_`` and keys them BY BEAT, and a jump-segment
    still deliberately wears ``jump_segment`` instead -- precisely so it stays
    invisible to the beat-keyed consumers. A loop that reused the existing
    lookup would therefore hand EVERY segment segment-0's still, silently
    re-creating the held-frame degradation chunks 3-6 exist to remove, with the
    correct stills sitting unused on disk and nothing in the log to say so.

    ONE AUTHORITY. The id comes off the durable stamp ShotLock minted, and the
    PATH comes off the still spine's own receipt -- the record the spine wrote
    when it proved that exact object id was materialized. Not a second lookup
    that happens to agree: the same fact, read once. If the spine did not prove
    it, this raises rather than substituting anything.

    ``segment_index`` 0 returns ``None``: segment 0 renders from the beat's own
    scene still, which every existing branch already resolves.
    """
    try:
        index = int(segment_index or 0)
    except (TypeError, ValueError) as exc:
        raise RenderError(
            "shot %s was asked for segment_index %r, which is not a segment "
            "number. Every other refusal in this module is NAMED; a bare "
            "ValueError from a coercion is not." % (shot.get("shot_id"),
                                                    segment_index)) from exc
    if index < 0:
        raise RenderError(
            "shot %s was asked for segment %d. A negative index used to read "
            "as 'segment 0' and silently hand back the BEAT's still, which is "
            "an off-by-one rendering the wrong image with no complaint. NO "
            "FALLBACK." % (shot.get("shot_id"), index))
    if index == 0:
        return None

    # ONLY A *JUMP* SEGMENT OWNS A STILL (2026-07-26, chunk 7a QA panel).
    #
    # This function is the DEMAND side of a mint that happens in
    # ``otr_shot_lock._stamp_coverage_plan``, and until multi-clip went live it
    # had never been asked about a real beat. The moment it was, it refused
    # every one: it treated "segment index >= 1" as "this is a jump segment",
    # which is only one of the two ways a beat gets a second clip.
    #
    # A CHAINED successor is the other, and it deliberately owns no still --
    # ``render_beat_coverage``'s own loop says so in as many words ("a chained
    # successor does not get a minted still at all -- it begins on the real
    # frame its predecessor ended on") and overwrites ``asset_refs["init_image"]``
    # with the extracted terminal frame right after this request is built.
    # ``coverage_plan.jump_still_requests`` agrees and returns nothing for a
    # CHAIN plan. So the mint correctly stamped no request, and the demand
    # correctly raised for its absence, and between them an ordinary 8-second
    # beat on wan_i2v / wan_ti2v / ltx_8gb / ltx_video died at segment 1.
    #
    # Two policies over one state, the third instance of this exact shape in
    # this feature. The fix is the same as the other two: ask the SAME
    # questions the mint asked, in the same order, off the same durable facts.
    plan_raw = shot.get("coverage_plan")
    if isinstance(plan_raw, dict) and plan_raw:
        try:
            from . import coverage_plan as _cp_local  # type: ignore
        except ImportError:  # pragma: no cover -- flat test imports
            import coverage_plan as _cp_local  # type: ignore
        join_mode = str(plan_raw.get("join_mode") or "")
        if join_mode and join_mode != _cp_local.JOIN_JUMP:
            return None
    # ...and a lane the still spine never asks a scene still of was never
    # minted one either (``_stamp_coverage_plan`` returns before minting when
    # ``_lane_consumes_a_still`` is false, and that predicate delegates to the
    # very function called here). An audio_driven_face beat renders from its
    # portrait, not from a per-segment scene still, so there is nothing to
    # demand -- and demanding it killed every HuMo beat over its cap, which for
    # the 49-frame humo_14B_169 tier is very nearly all of them.
    if not _still_spine_requires_scene(
            shot, str(shot.get("engine_id") or ""),
            engine_family(str(shot.get("engine_id") or ""),
                          str(shot.get("family") or ""))):
        return None

    requests = _jump_still_requests_for_shot(shot)
    request = next((row for row in requests
                    if int(row.get("segment_index") or 0) == index), None)
    if request is None:
        raise RenderError(
            "shot %s has no jump-still request for segment %d, so that "
            "segment has no still of its own to render from. NO FALLBACK to "
            "the beat's still: putting the identical image at both ends of a "
            "cut is the held-frame degradation this build removes."
            % (shot.get("shot_id"), index))
    object_id = str(request.get("object_id") or "")
    receipt = (((ledger or {}).get("images") or {}).get("still_spine_receipt")
               or {})
    for entry in receipt.get("validated") or ():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("object_id") or "") != object_id:
            continue
        path = str(entry.get("path") or "")
        if not path:
            # KEEP LOOKING, do not give up here (2026-07-26 QA panel). A
            # receipt can carry more than one entry for an object id -- an
            # unmaterialized placeholder written before the materialized row.
            # Breaking on the first pathless match refused a still that WAS
            # proven, two entries later.
            continue
        return path
    raise RenderError(
        "shot %s segment %d needs jump-segment still %s, which the still-spine "
        "receipt does not carry a materialized path for. The spine runs BEFORE "
        "any render and refuses a shot whose segment stills are missing, so "
        "reaching here means the receipt was rebuilt, replayed or bypassed. NO "
        "FALLBACK." % (shot.get("shot_id"), index, object_id or "<missing>"))


def segment_render_frames(shot, segment_index):
    """How many frames segment ``segment_index`` renders, off the STAMPED plan.

    The other half of "build one segment's request" (2026-07-26 QA panel). The
    first draft of the per-segment seam swapped the init IMAGE and left the
    LENGTH alone, so a request for segment 1 of a 120-frame beat carried the
    init image of segment 1 and the frame count of the whole beat -- correct
    picture, wrong duration, and the two disagreeing is precisely the shape
    this build keeps having to remove. A segment's length is not derivable from
    the shot row; it is in the plan, so it is read from the plan.

    Returns the beat's own ``target_frame_count`` only when the shot carries no
    plan at all -- which is every beat today. When a plan exists it answers
    from the plan for EVERY index, segment 0 included.

    ``render_frames``, deliberately, NOT the visible count: the engine renders
    that many, and ``drop_head`` / ``trim_tail`` are the assembler's business.
    """
    beat_frames = int(shot.get("target_frame_count") or 0)
    index = int(segment_index or 0)
    raw = shot.get("coverage_plan")
    if isinstance(raw, dict) and raw:
        # THE PLAN WINS FOR EVERY SEGMENT, INDEX 0 INCLUDED (corrected
        # 2026-07-26 by the chunk-6c loop's own end-to-end test). An earlier
        # draft short-circuited index 0 to the BEAT's length, which is right
        # for a single-clip beat and wrong for the first segment of a
        # multi-segment one: segment 0 of a two-segment 50-frame beat would
        # have rendered all 50 frames and then been concatenated with segment 1
        # on top of them. It read as a harmless special case because for a
        # single-clip plan the two numbers are equal.
        try:
            from . import coverage_plan as _cp  # type: ignore
        except ImportError:  # pragma: no cover -- flat test imports
            import coverage_plan as _cp  # type: ignore
        plan = _cp.CoveragePlan.from_dict(raw)
        for segment in plan.segments:
            if int(segment.index) == index:
                return int(segment.render_frames)
        raise RenderError(
            "shot %s was asked for segment %d but its coverage plan covers "
            "segment(s) %r. NO FALLBACK."
            % (shot.get("shot_id"), index,
               [int(s.index) for s in plan.segments]))
    if index <= 0:
        return beat_frames
    raise RenderError(
        "shot %s was asked for segment %d but carries no coverage_plan, so "
        "there is nothing that says how long that segment is. A segment "
        "request without a plan is a caller bug. NO FALLBACK to the beat's "
        "length -- that renders the WHOLE beat under a segment's name."
        % (shot.get("shot_id"), index))


def validate_and_repair_still_spine(ledger):
    """Prove every still-consuming beat has an active, materialized input.

    This is the final image-to-video handoff contract. It repairs only an
    already-authorized row whose ``path`` or ``pool_path`` is readable, writes
    the repaired path back to that row and the required-target receipt, and
    raises before the first video engine request when the target is absent.
    Mesh lanes are checked separately: fodder is never replaced by a scene
    still. Deferred request sentinels are not inspected here because this pass
    validates image assets, not cast-time handles.
    """
    if not isinstance(ledger, dict):
        raise RenderError("still-spine handoff requires a ledger object")
    images = ledger.get("images")
    if not isinstance(images, dict):
        raise RenderError("still-spine handoff has no images section")
    episode_id = _still_spine_episode_id(ledger)
    if not episode_id:
        raise RenderError("still-spine handoff requires a durable episode_id")
    try:
        from .._otr_paths import otr_stills_dir
        stills_root = str(otr_stills_dir(episode_id))
    except Exception as exc:  # noqa: BLE001
        raise RenderError(
            "still-spine handoff cannot resolve active episode stills: %s" % exc
        ) from exc
    rows = _still_spine_image_rows(ledger)
    shots = [shot for shot in ((ledger.get("video") or {}).get("shots") or [])
             if isinstance(shot, dict)]
    # Drop image rows for REFUSED objects before any lookup: a prior
    # revision's file for the same object can still be on disk, and
    # ``_still_spine_row_for_beat`` is last-row-wins with no notion of status,
    # so a stale row would otherwise validate a still this dispatch never
    # produced. Computed here because ``rows`` feeds every lookup below.
    _gap_objects_early = sanctioned_gap_object_ids(ledger)
    if _gap_objects_early:
        rows = [r for r in rows
                if str((r or {}).get("object_id") or "")
                not in _gap_objects_early]
    manifest_ids = _still_spine_manifest_object_ids(images)
    # SANCTIONED GAPS ARE NOT MISSING STILLS (2026-08-28). A model refusal is
    # recorded in the receipt as an explicit gap row, and the operator's
    # 2026-08-22 ruling says the episode continues without that card. Before
    # this, the refusal reached here as an ABSENCE and the per-shot raises
    # below killed the whole episode -- which is how a 30-minute render died
    # for one declined image. These beats are skipped here and skipped again
    # in ``run_episode``; nothing is substituted and nothing is silent.
    gap_beats = sanctioned_gap_beat_ids(ledger)
    # Audio-reactive visualizers and portrait-only face lanes deliberately do
    # not consume scene stills.  Their image phase can therefore have no
    # required_scene_targets receipt at all.  Require the producer-owned target
    # manifest only when a rendered shot actually needs scene or mesh input;
    # still-consuming lanes remain fail-closed below.
    requires_scene_manifest = False
    for shot in shots:
        engine_id = str(shot.get("engine_id") or "")
        family = engine_family(engine_id, str(shot.get("family") or ""))
        requires_mesh = bool(
            engine_id and _vreg.is_registered(engine_id)
            and getattr(_vreg.get_engine(engine_id), "requires_mesh_fodder", False)
        )
        if (requires_mesh or _still_spine_requires_scene(shot, engine_id, family)
                or _jump_still_requests_for_shot(shot)):
            requires_scene_manifest = True
            break
    if requires_scene_manifest and not manifest_ids:
        raise RenderError(
            "still-spine handoff has no required_scene_targets receipt; "
            "image generation did not publish an authoritative target manifest"
        )
    lines = _line_index(ledger)
    validated = []
    for shot in shots:
        raw_bid = _beat_id_for_shot(shot)
        line = lines.get(raw_bid, {})
        # The shot's own beat id IS the producer's key -- no translation.
        beat_id = raw_bid
        engine_id = str(shot.get("engine_id") or "")
        family = engine_family(engine_id, str(shot.get("family") or ""))
        char_id = str(shot.get("char_id") or line.get("char_id") or "")
        requires_mesh = bool(
            engine_id and _vreg.is_registered(engine_id)
            and getattr(_vreg.get_engine(engine_id), "requires_mesh_fodder", False)
        )
        if beat_id in gap_beats:
            # Skipped WHOLE, before any materialize attempt: the mesh, scene
            # and jump-segment checks below all raise on an absent file, and
            # this beat is absent by sanction rather than by fault. It is not
            # appended to ``validated`` either -- it delivered nothing, and a
            # validation record claiming otherwise would be the same lie the
            # gap row exists to prevent.
            _LOG.warning(
                "[OTR.render_driver] SANCTIONED GAP beat %s (shot %s): the "
                "image model "
                "refused its required still; the still spine accepts this "
                "and the beat will be floored rather than rendered.",
                beat_id, shot.get("shot_id"))
            continue
        scene_row = _still_spine_row_for_beat(rows, beat_id)
        if requires_mesh:
            fodder_row = _still_spine_row_for_mesh(rows, beat_id, char_id)
            fodder_path = _still_spine_materialize_row(fodder_row, stills_root)
            if not fodder_row or not fodder_path:
                raise RenderError(
                    "still-spine handoff missing materialized mesh_fodder for "
                    "shot %s beat %s; scene still substitution is forbidden"
                    % (shot.get("shot_id"), beat_id)
                )
            if str(fodder_row.get("object_id") or "") not in manifest_ids:
                raise RenderError(
                    "still-spine handoff mesh target %s is not in the "
                    "required_scene_targets receipt" % (
                        fodder_row.get("object_id") or beat_id)
                )
            validated.append({"shot_id": shot.get("shot_id"),
                              "beat_id": beat_id, "engine_id": engine_id,
                              "kind": "mesh_fodder", "path": fodder_path})
        if _still_spine_requires_scene(shot, engine_id, family):
            scene_path = _still_spine_materialize_row(scene_row, stills_root)
            if not scene_row or not scene_path:
                raise RenderError(
                    "still-spine handoff missing materialized scene still for "
                    "shot %s beat %s engine %s"
                    % (shot.get("shot_id"), beat_id, engine_id)
                )
            object_id = str(scene_row.get("object_id") or "")
            if object_id not in manifest_ids:
                raise RenderError(
                    "still-spine handoff scene target %s for beat %s is not "
                    "in the required_scene_targets receipt"
                    % (object_id or "<missing>", beat_id)
                )
            validated.append({"shot_id": shot.get("shot_id"),
                              "beat_id": beat_id, "engine_id": engine_id,
                              "kind": str(scene_row.get("kind") or "scene"),
                              "path": scene_path})
        # EVERY JUMP SEGMENT, not just the beat (2026-07-25, chunk 4). The
        # checks above prove the beat has ONE still, which is all a single-clip
        # beat needs. A jump-cut beat renders N independent clips and segment 0
        # is the only one that beat still covers -- so each remaining segment
        # is proven here, by object id, or the render does not start. There is
        # deliberately no repair-by-substitution: borrowing the beat's still
        # for a segment would put the identical image at both ends of the cut,
        # which is the held-frame degradation this build removes.
        for request in _jump_still_requests_for_shot(shot):
            segment_oid = str(request.get("object_id") or "")
            segment_row = next(
                (candidate for candidate in rows
                 if str(candidate.get("object_id") or "") == segment_oid), None)
            segment_path = _still_spine_materialize_row(segment_row, stills_root)
            if not segment_row or not segment_path:
                raise RenderError(
                    "still-spine handoff missing materialized jump-segment "
                    "still %s for shot %s beat %s segment %s; a jump cut whose "
                    "segment has no still would render from nothing"
                    % (segment_oid or "<missing>", shot.get("shot_id"),
                       beat_id, request.get("segment_index"))
                )
            if segment_oid not in manifest_ids:
                raise RenderError(
                    "still-spine handoff jump-segment target %s for beat %s is "
                    "not in the required_scene_targets receipt"
                    % (segment_oid, beat_id)
                )
            validated.append({"shot_id": shot.get("shot_id"),
                              "beat_id": beat_id, "engine_id": engine_id,
                              "kind": str(segment_row.get("kind") or "jump_segment"),
                              "segment_index": int(
                                  request.get("segment_index") or 0),
                              # The id the request minted and this pass just
                              # PROVED. The per-segment render reads its init
                              # image back by this id (chunk 6b) rather than
                              # deriving a path a second time.
                              "object_id": segment_oid,
                              "path": segment_path})
        if family == "audio_driven_face":
            portrait_row = _still_spine_row_for_portrait(rows, char_id)
            portrait_path = _still_spine_materialize_row(
                portrait_row, stills_root)
            if not portrait_row or not portrait_path:
                raise RenderError(
                    "still-spine handoff missing materialized portrait for "
                    "shot %s character %s" % (shot.get("shot_id"), char_id)
                )
            validated.append({"shot_id": shot.get("shot_id"),
                              "beat_id": beat_id, "engine_id": engine_id,
                              "kind": "portrait", "path": portrait_path})
    receipt = {
        "version": 1,
        "episode_id": episode_id,
        "validated": validated,
    }
    images["still_spine_receipt"] = receipt
    for target in images.get("required_scene_targets") or []:
        if not isinstance(target, dict):
            continue
        target_id = str(target.get("object_id") or "")
        row = next((candidate for candidate in rows
                    if str(candidate.get("object_id") or "") == target_id), None)
        if row and row.get("path"):
            target["path"] = str(row["path"])
            target["materialized_path"] = str(row["path"])
    return receipt


#: Engine FAMILIES whose render is conditioned on a SCENE still (still-spine
#: ST-4 / W6): image_to_video (wan_i2v) + static_motion (still_motion) take
#: asset_refs.init_image from the beat's scene still. Engines in other families
#: that explicitly declare ``init_image`` can opt into the same scene-still join
#: in ``build_request_from_shot``; face/talking lanes keep their own portrait or
#: radio-face branches.
_SCENE_INIT_FAMILIES = frozenset({"image_to_video", "static_motion"})


def _line_index(ledger):
    """``{line_id: line}`` from the frozen ledger lines. ``line_id`` equals a
    beat_id equals a shot's ``source_line_ids[0]``."""
    out = {}
    for ln in (ledger or {}).get("lines") or []:
        if isinstance(ln, dict):
            lid = str(ln.get("line_id") or "")
            if lid:
                out.setdefault(lid, ln)
    return out


def _voice_audio_for_line(line):
    """The per-beat VOICE audio path for a line, robust to the per-engine field
    name (``bark_wav_path`` / ``indextts2_wav_path`` / ...) plus the canonical
    ``audio_wav_path``; a music clip is the fallback for a music beat. SFX is
    never a face-driving voice. Returns "" when the ledger carries no per-line
    audio (then audio_ref is None and HuMo falls back LOUD, or the render node
    slices the frozen master mix by the beat timing)."""
    if not isinstance(line, dict):
        return ""
    for k in ("audio_wav_path", "wav_path"):
        if line.get(k):
            return str(line[k])
    for k, v in line.items():
        if v and str(k).endswith("wav_path") and not str(k).startswith("music"):
            return str(v)
    for k in ("music_wav_path", "clip_path", "video_clip_path"):
        if line.get(k):
            return str(line[k])
    return ""


def _beat_id_for_shot(shot):
    """The beat_id a shot renders: ``source_line_ids[0]`` (the ShotLock link),
    else the ``shot_`` prefix stripped off the shot_id."""
    sids = shot.get("source_line_ids")
    if isinstance(sids, list) and sids:
        return str(sids[0])
    sid = str(shot.get("shot_id") or "")
    return sid[len("shot_"):] if sid.startswith("shot_") else sid


#: Mirrors ``otr_shot_lock.OPENING_MUSIC_BEAT_ID`` -- duplicated as a local
#: constant (round 5): importing the ShotLock node module from the driver would
#: drag node-registration side effects into the engine package.
#:
#: STILL LIVE, and it is a CLASSIFIER, not a translation. Three callers ask
#: "is this shot ShotLock's synthetic head-gap opener?" by structure -- the
#: motion-register pick, the round-5-F2 synthetic-open subject detection, and
#: the LTX open-clip health check. That beat is real: ShotLock inserts it
#: whenever there is a genuine >= 2 s head gap, and the image producer mints a
#: target under that same id when it does.
_OPENING_MUSIC_SUFFIX = "b000_music_open"


# _canonical_visual_beat_id WAS HERE, AND ITS REMOVAL IS THE FIX FOR
# PBUG-20260811-02. Do not reintroduce it, under any name.
#
# It rewrote a shot's beat id to ``b000_music_open`` whenever the id was
# ``music_opening_001`` or the line looked like a mirrored music opener, so that
# visual-asset lookups would find rows the image producer had keyed under the
# SYNTHETIC id while the shot carried the POSITIONED one. That was a real fix
# for a real split -- while the producer read the PRE-AUDIO ledger.
#
# Commit 3446af3f retargeted canonical link 255 so the producer reads ShotLock's
# POST-AUDIO ledger, where EpisodeAssembler has already mirrored the cue into an
# ordinary beat named ``music_opening_001``. From that moment the producer keyed
# its rows under the positioned id and this rewrite pointed the consumer at an
# id nothing minted any more -- so it did not fail to fire, it fired and moved
# the mismatch from the closing beat to the opening one. The 2026-08-12
# ``mesh_stage`` leg died on exactly that, with the plate it wanted sitting on
# disk under the other name.
#
# THE AUTHORITATIVE ID IS THE EXACT BEAT ID IN SHOTLOCK'S FINAL EXECUTION PLAN,
# and both halves already agree on it once nothing translates between them: a
# positioned mirror carries its own ``music_<cue>_<NNN>`` line id, and a genuine
# synthetic opener carries ``b000_music_open`` with empty ``source_line_ids``,
# which ``_beat_id_for_shot`` recovers from the shot_id. Each resolves through
# its own id.
#
# This is also the only shape that survives a chunked cue.
# ``_MUSIC_MAX_CHUNK_DUR_S = 22.0`` means a long cue becomes ``_001``, ``_002``,
# ``_003`` against ONE synthetic beat, and no 1:1 rewrite can express
# one-to-many. Removing the translation is what scales; a smarter mapping is not.


#: 6/5 motion-centric LTX prompts (BUG-LOCAL-112, restored 2026-06-12 from the
#: legacy ``batch_ltx_render._PROMPT_BY_ROLE``). The i2v anchor carries the LOOK
#: from the FLUX still; the video prompt's ONLY job is MOTION. Design rules
#: (MAD-verified): every sentence is a motion verb, NO set-dressing nouns, NO
#: negation, total <= 240 chars, first motion verb within the first 140. The
#: refactor had diluted these into ~185-char scene-DESCRIPTIVE "brief+beat"
#: prompts ("a 1940s radio station studio, glowing warmly..."), which the model
#: reads as "render this set" -> flat pans on the conditioned still.
# Chunk A2 (visual-style TOTAL COVERAGE, 2026-07-05): the console/music
# motion VALUES are pack-owned (VisualStyle.motion_registers, exact keys
# below); the role->key SELECTOR + the OTR_LTX_OPEN_MOTION_KEY env retarget
# stay Python. This dict survives ONLY as the sci_fi_radio extraction
# fixture (pack byte-identity pinned in tests) -- production reads the pack
# in build_request_from_shot (AST-pinned).
_MOTION_REGISTER_KEYS = frozenset(
    {"announcer", "music_open", "music_close", "music_inter"})
#: COLLIDING WORDS ONLY (operator call 2026-08-17, option B). An earlier sweep
#: stripped the trailing camera clause from all four registers; a live LTX A/B
#: measured the cost at ~40% less motion (0.220 vs 0.373 mean frame delta), and
#: only TWO of the 35 camera clauses repo-wide ever actually collided. So the
#: authored choreography and camera moves are RESTORED, and only the words that
#: hit a frozen provider list changed:
#:   "Dial whip-pans"            -> "Dial races"        (Wan ban + Seedance)
#:   "white-hot"                 -> "fierce white"      (Seedance rewrote it to
#:                                                       "bright warm glow",
#:                                                       breaking the sentence)
#:   "vibrates aggressively"     -> "shudders with the music"
#:                                                      (Seedance rewrote it to
#:                                                       "subtly" -- inverted)
#:   "Slow handheld dolly forward" -> "Slow steady dolly forward"  ("handheld")
#:   "Dynamic dolly push forward"  -> "Steady dolly push forward"
#: `Slow dolly pull back` and `Slow orbit around the speaker` never collided and
#: are untouched. Kept: the continuity opener and "Tuning dial needle sweeps"
#: (pinned by tests/test_brief_prompt_finishing.py).
_LTX_MOTION_PROMPT_BY_ROLE = {
    "announcer": ("Continuous shot, same console throughout. Tuning dial needle "
                  "sweeps rhythmically. Vacuum tubes pulse. Brass speaker grille "
                  "trembles. Dust motes drift. Slow steady dolly forward."),
    "music_open": ("Continuous shot, same console throughout. Dial races across "
                   "frequencies. Tube filaments ignite from cold to fierce "
                   "white. Speaker grille shudders with the music. Steady dolly "
                   "push forward."),
    "music_close": ("Continuous shot, same console throughout. Dial settles. "
                    "Tube filaments cool from white through deep amber. Smoke "
                    "trails from cooling tubes. Slow dolly pull back."),
    "music_inter": ("Continuous shot, same console throughout. Dial steady, "
                    "glowing. Oscilloscope dances to the rhythm. VU meters "
                    "bounce. Tubes pulse with the bass. Slow orbit around the "
                    "speaker."),
}
#: BUG-LOCAL-112 char budget for the motion prompt (verb-only core + optional
#: short brief fragment appended AFTER, dropped if it breaks the budget).
_LTX_MOTION_PROMPT_MAX = 240

#: TALKING register for the ia2v_canonical recipe (lips-dont-talk kibitz,
#: 2026-07-02). Probe-proven: the ia2v two-stage recipe animates what the
#: prompt NARRATES; the console-motion register above steers motion into
#: dials/tubes/dolly and the lips freeze (P4: motion 4.15 -> 1.18). The
#: ANNOUNCER bookend swaps to this canonical-register talking prompt when the
#: routed engine's recipe wants it (LtxAudioInEngine.wants_talking_prompt()).
#: MUSIC bookends deliberately KEEP the console-motion register -- P2 proved
#: LTX cannot lip-sync to music (motion 0.59); a talking prompt there would
#: fight physics for nothing.
#: Canonical token skeleton (probe P8 2026-07-02: a paraphrased register
#: scored HALF the canonical's articulation at identical params -- 1.72 vs
#: 3.32). Keep the talking/lips/sync pattern intact; the mouth material can be
#: style-neutral so packs that forbid "cartoon" do not fight the prompt.
_IA2V_TALKING_PROMPT_ANNOUNCER = (
    "A face-forward vintage radio with a large expressive grille-mouth is "
    "talking to the viewer, its clear grille-cloth lips opening and closing "
    "naturally in sync with the speech, its round glass tuning-dial eyes "
    "glancing subtly as it speaks. The radio sits still; static camera, "
    "warm dramatic lighting.")
#: Character-beat talking clause (appended to a COMPACT identity fragment;
#: the 900-char shared prompt drowned the speech tokens). Mirrors the
#: canonical token pattern ("talking to the viewer ... opening and closing
#: naturally in sync with the speech") -- that half is P8-proven and is
#: reproduced VERBATIM; a paraphrase scored 1.72 against 3.32.
#:
#: THE TAIL WAS CORRECTED 2026-08-27. It read "subtle head and hand
#: gestures": "subtle" is the damping word PBUG-20260827-04 banned, and
#: hand acting is what LIPSYNC_MOTION_ENVELOPE retired for this family
#: because hands cross the face. It now names the JAW and BLINKS, per the
#: operator LTX 2.5 lip-sync reference -- a face that never blinks reads as
#: a still even while the mouth moves. Hands are not mentioned either way;
#: "no hands" would put hands back in the conditioning.
_IA2V_TALKING_CLAUSE_CHARACTER = (
    "is talking to the viewer, lips opening and closing naturally in sync "
    "with the speech, the jaw moving with the words, small head motion "
    "and natural blinks. Static camera.")


def _style_authority():
    """The shared style-token authority (`_otr_visual_styles`).

    ONE STYLE AUTHORITY (2026-08-17): the derivation and the prepend used to
    live here and served video prompts only. They now live in the style module
    so the STILL path applies the identical token; two definitions of "the
    style word" drifting apart is the defect this consolidation prevents.
    """
    try:
        from .._otr_visual_styles import (  # type: ignore
            compact_style_cue, prefix_style_cue)
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_visual_styles import (  # type: ignore
            compact_style_cue, prefix_style_cue)
    return compact_style_cue, prefix_style_cue


def _compact_style_talking_cue(vstyle):
    """Compact pack cue for IA2V character prompts.

    The talking register is deliberately tiny for lip-sync; long style tails
    drown the speech tokens. Non-default packs still need one blunt cue close
    to the front of the prompt so the video model does not drift back to a
    generic cinematic portrait.
    """
    compact_style_cue, _ = _style_authority()
    return compact_style_cue(vstyle)


def _prefix_video_style_cue(vstyle, prompt):
    """Put the selected non-default visual style at the front of video text.

    This is intentionally a short cue, not a full style tail. Long tails dilute
    LTX/IA2V talking prompts and can crowd out identity/motion tokens.
    """
    _, prefix_style_cue = _style_authority()
    return prefix_style_cue(vstyle, prompt)


def _style_cue_after_pinned_opener(vstyle, prompt):
    """Apply the style cue, but never in front of a lane's REQUIRED opener.

    Returns the ordinary prefixed prompt for every lane that has no pinned
    opener, so this is a no-op everywhere except the two H3 lanes. Pure.
    """
    text = str(prompt or "")
    try:
        from .eng_minimax_h3 import H3_REFERENCE_OPENER
    except ImportError:  # pragma: no cover -- flat test imports
        from eng_minimax_h3 import H3_REFERENCE_OPENER  # type: ignore
    if not text.startswith(H3_REFERENCE_OPENER):
        return _prefix_video_style_cue(vstyle, text)
    rest = text[len(H3_REFERENCE_OPENER):].lstrip()
    if not rest:
        return text
    cued = _prefix_video_style_cue(vstyle, rest)
    if cued == rest:                      # default pack -> empty cue
        return text
    return "%s %s" % (H3_REFERENCE_OPENER, cued)


def _ia2v_talking_register_active(engine_id):
    """True iff ``engine_id`` is the ltx_audio_in lane AND its active recipe
    is the two-stage ia2v_canonical lip-sync graph (engine-owned decision --
    the driver never parses recipe env strings). Any engine-side misconfig
    returns False here and still fails LOUD at the engine's own gate."""
    if str(engine_id or "") != "ltx_audio_in":
        return False
    try:
        from .eng_ltx_av import LtxAudioInEngine
        return bool(LtxAudioInEngine().wants_talking_prompt())
    except Exception as exc:  # noqa: BLE001
        # LOUD (kibitz r2): never silently fall back to the console register
        # on a misconfig -- the engine's own gate still hard-fails the render,
        # but the prompt-routing decision must be debuggable from the log.
        _LOG.warning(
            "[OTR.render_driver] IA2V talking-register probe failed (%s) -- "
            "console register kept; the engine gate will surface the real "
            "misconfig LOUD", exc)
        return False


def _ltx_motion_role_key(shot_role, shot_id, is_synthetic_open):
    """Map an OTR shot role + beat id to a motion-register key (the pack's
    ``motion_registers`` / :data:`_MOTION_REGISTER_KEYS`), or ``""`` if the
    beat is not a radio-console motion beat. Pure."""
    sid = str(shot_id or "")
    role = str(shot_role or "")
    # A SYNTHETIC opening-music beat can carry an announcer_visual role (the
    # b000_music_open structure is definitive, NOT the role) -- check it first.
    if is_synthetic_open or sid.endswith(_OPENING_MUSIC_SUFFIX):
        # Operator 2026-06-12 had retargeted this to the calm music_inter because
        # the aggressive music_open verbs (whip-pans / "vibrates aggressively" /
        # dynamic dolly push) SMEARED on the 2B LTX model. 2026-06-15: that smear
        # was the 1472x832 OVER-RESOLUTION mush (BUG-412), FIXED at native 832x480
        # -- a GPU A/B proved music_open now renders SHARP (Laplacian 934 vs 83)
        # AND moves ~9x more (paired with the ksampler default). So the dynamic
        # open is RESTORED as the default (operator "moving grooving"). music_inter
        # stays available via OTR_LTX_OPEN_MOTION_KEY=music_inter.
        _open_key = os.environ.get("OTR_LTX_OPEN_MOTION_KEY", "music_open")
        # A2: membership check against the STATIC key set (the values moved
        # to the style pack; the retired fixture dict is not consulted).
        return (_open_key if _open_key in _MOTION_REGISTER_KEYS
                else "music_inter")
    if role == "announcer_visual":
        return "announcer"
    if role == "music_visual":
        if any(t in sid for t in ("close", "outro", "_end", "tag", "sign_off")):
            return "music_close"
        return "music_inter"
    return ""


#: Deterministic fallback for a CHARACTER face beat whose shot carries no M4
#: creative prompt. It stays world-neutral; authored prompts are preserved.
_CHAR_FACE_FALLBACK_PROMPT = (
    "close-up cinematic portrait of a person speaking, face centered, subtle "
    "facial motion, attire from the story world, dramatic film lighting")

#: Round 5 F2 -- beat_intent -> scene clause. Unmapped intents fall back to a
#: loose "a beat of <intent>" clause + one INFO line (never a silent skip).
_INTENT_CLAUSES = {
    "revelation": "a moment of revelation",
    "reveal": "a moment of revelation",
    "discovery": "awe and discovery",
    "wonder": "awe and discovery",
    "dread": "gathering tension",
    "tension": "gathering tension",
    "warning": "gathering tension",
    "conflict": "voices in conflict",
    "confrontation": "voices in conflict",
    "calm": "a quiet steady moment",
    "comfort": "a quiet steady moment",
    "urgency": "urgent momentum",
    "resolution": "the tension easing",
}

#: Round 5 F2 -- arc_phase -> tone clause (exact-match; absent/unknown skips).
_ARC_CLAUSES = {
    "setup": "early scene-setting calm",
    "rising": "rising stakes",
    "climax": "the story's peak intensity",
    "falling": "the aftermath settling",
    "resolution": "aftermath hush",
}


def _beat_clauses(line, shot_id):
    """Per-beat scene clauses from the FROZEN line's own signals (round 5 F2:
    ``visual_plan.scenes`` is empty post-CW-1, so beat variety comes from
    ``beat_intent`` + ``arc_phase``). Absent fields skip silently; an unmapped
    intent gets the loose clause + one INFO line. Pure, read-only."""
    out = []
    intent = str((line or {}).get("beat_intent") or "").strip().lower()
    if intent:
        mapped = _INTENT_CLAUSES.get(intent)
        if mapped is None:
            # The writer's beat_intent vocabulary is FREE TEXT (live catch
            # 2026-06-10: "open the episode and orient the listener."), not
            # enum tokens -- bound the loose clause to its first 6 words and
            # drop trailing punctuation so the scene prompt stays prompt-like.
            short = " ".join(intent.split()[:6]).rstrip(".,;:!?")
            mapped = "a beat of %s" % short
            _LOG.info("[OTR.render_driver] unmapped beat_intent %r on %s -- "
                      "using the loose clause %r", intent, shot_id, mapped)
        out.append(mapped)
    arc = str((line or {}).get("arc_phase") or "").strip().lower()
    if arc in _ARC_CLAUSES:
        out.append(_ARC_CLAUSES[arc])
    return out


def _stamp_prompt_meta(req, source, prompt, *, subsource="", beat=""):
    """Stamp prompt observability onto the request's ``observability`` dict
    (round 5 F2): source enum (shared_video_prompting_engine|engine:<id>|env|brief+beat), sha8, char count --
    ``run_episode`` copies them onto the trace rows (durable in the node-92
    /history report) and one INFO line makes operator log review mechanical.
    The W7-pre builder migration moved these off the top level: VideoRequest
    is extra="forbid", so underscore extras made every request schema-invalid."""
    sha8 = hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()[:8]
    obs = req.setdefault("observability", {})
    obs["prompt_source"] = source
    if subsource:
        obs["prompt_subsource"] = subsource
    obs["prompt_sha8"] = sha8
    obs["prompt_chars"] = len(str(prompt))
    _LOG.info("[OTR.render_driver] prompt source=%s sha8=%s chars=%d beat=%s "
              "| %.100s", source, sha8, len(str(prompt)), beat, prompt)


def _apply_visual_safety_prompt(req, shot) -> None:
    engine_id = str((shot or {}).get("engine_id") or "").strip()
    if not _is_cloud_video_engine(engine_id):
        return
    prompt = str(req.get("text_prompt") or "").strip()
    if not prompt:
        return
    try:
        from .._otr_story_brief_helpers import append_visual_safety_clause
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_story_brief_helpers import append_visual_safety_clause  # type: ignore
    safe = append_visual_safety_clause(prompt)
    if safe == prompt:
        return
    req["text_prompt"] = safe
    obs = req.setdefault("observability", {})
    source = str(obs.get("prompt_source") or "safety")
    subsource = str(obs.get("prompt_subsource") or "")
    _stamp_prompt_meta(
        req, source, safe, subsource=subsource, beat=_beat_id_for_shot(shot))
    obs["visual_safety_prompt"] = "applied"


def ltx_prompt_diversity_status(trace):
    """Diversity status over the BRIEF-COMPOSED text-engine prompts in a
    :func:`run_episode` trace (round 5 acceptance: the per-beat composition
    must actually differ). ``ok`` is vacuously True for n < 2; operator
    ``env``-override prompts are exempt (an override may legitimately repeat).
    Pure; returns ``{n, distinct, ok, sha8s}``."""
    shas = [str(r.get("prompt_sha8") or "") for r in (trace or [])
            if isinstance(r, dict)
            and r.get("prompt_source") == "brief+beat"
            and r.get("prompt_sha8")]
    distinct = len(set(shas))
    return {"n": len(shas), "distinct": distinct,
            "ok": (len(shas) < 2) or distinct > 1, "sha8s": shas}


#: The AUDIO-IN engines that count as a talking head on a `character_video`
#: beat. A MEMBERSHIP test since lane 20 (2026-08-12); it was an equality test
#: on `ltx_audio_in` while that was the only such lane.
#:
#: THIS IS LOAD-BEARING AND IT FAILS AT PLAN TIME, not at render time. An
#: audio-in beat that is neither a character face nor a cabinet role raises
#: `MouthPolicyError` -- `mouth_owner_for_beat` refuses to guess whether the
#: still it will animate has lips. So registering an `audio_conditioned_video`
#: engine WITHOUT adding it here does not degrade anything: it makes every
#: character beat on that engine fail before a single weight loads.
#:
#: THE WRONG FIX, named so nobody reaches for it: minting a new family to dodge
#: the check. A family outside `content_oracle.MOTION_FAMILIES` makes frozen
#: clips motion-EXEMPT, so the beat would stop being checked for motion at all
#: -- trading a loud plan-time refusal for a silent quality hole.
_AUDIO_IN_CHARACTER_ENGINES = ("ltx_audio_in", "minimax_h3_audio_in")


def _is_character_face_beat(shot):
    """The ROLE-driven 'talking head' signal: a beat that needs the character's OWN
    clean voice + a character prompt, NOT the ambient master slice + scene prompt the
    announcer / music / scene BOOKEND beats get. Role is PRIMARY. announcer_visual /
    music_visual are NEVER character-face. The audio_driven_face family (HuMo) on a
    non-open beat counts; the unified ``ltx_audio_in`` audio-in lane counts when it
    drives a CHARACTER beat (it is one engine for bookends AND characters, so the
    talk-vs-scene split that used to live in ltx_av_talk/ltx_av_music now lives on the
    ROLE). Other engines (ltx_video / wan_i2v / still_pan) are UNCHANGED -- they are
    never character-face here, so their audio/prompt routing is untouched. Pure."""
    role = str((shot or {}).get("role") or "")
    if not role:
        _gid = str((shot or {}).get("group_id") or "")
        if _gid.startswith("grp_"):
            role = _gid[len("grp_"):]
    if role in ("announcer_visual", "music_visual"):
        return False
    if engine_family(str((shot or {}).get("engine_id") or ""), "") == "audio_driven_face":
        return True
    if (str((shot or {}).get("engine_id") or "") in _AUDIO_IN_CHARACTER_ENGINES
            and role == "character_video"):
        return True
    return False


def _uses_ambient_master_audio(engine_id, family, is_char_face=False, role=""):
    """Lanes that CONDITION on the ambient master MIX (not a specific voice): the
    ``audio_conditioned_video`` family (ltx_av_music / ltx_audio_in BOOKEND beats --
    which HARD-require audio_ref) + the viz_green scopes. These get a bounded master
    slice when a beat lacks per-line timing. Local ``audio_driven_face`` engines
    (HuMo / ltx_av_talk) are normally EXCLUDED: they need the character's OWN voice,
    so a master-mix slice would make them lip-sync to the wrong audio. A
    CHARACTER-FACE beat is excluded for the SAME reason regardless of family --
    ``ltx_audio_in`` on a character beat must use the character's clean own voice,
    never the ambient mix (2026-06-26 role-driven).

    OTR_ENABLE_HUMO_HOSTS (2026-07-01 brief-driven radio-host): when the toggle is ON,
    a LINELESS announcer/music BOOKEND routes to HuMo (audio_driven_face) as the
    radio-HOST face. That beat has NO per-line voice (it is the music/announcer bed),
    yet HuMo HARD-requires ``audio_ref`` -- so the host is driven by the ambient MASTER
    slice (it "hosts / sings" to the bed; deliberately not real lip-sync). This is the
    ONE local case an audio_driven_face beat uses the master mix, and only for the
    never-humo bookend roles (a real CHARACTER face beat is still excluded above).
    Partner/cloud avatar engines are also allowed to use the bounded bookend slice:
    they are the selected cloud audio-reactive lane, not the historical local HuMo
    default, and must not fall back to a local LTX rewrite just to satisfy
    ``audio_ref``."""
    if is_char_face:
        return False
    if (str(family) == "audio_driven_face"
            and _is_never_humo_video_role(str(role or ""))):
        return (os.environ.get("OTR_ENABLE_HUMO_HOSTS", "0") == "1"
                or _is_cloud_video_engine(engine_id))
    return (str(family) == "audio_conditioned_video"
            or str(engine_id) in ("viz_green", "viz_mxc_cpu", "viz_mxc_mandala",
                                  "viz_camera"))


def _role_of_shot(shot) -> str:
    """The beat's role: the explicit ShotRow ``role``, else derived from the
    ``grp_<role>`` group_id (mirrors :func:`_is_character_face_beat`). Pure."""
    role = str((shot or {}).get("role") or "")
    if not role:
        gid = str((shot or {}).get("group_id") or "")
        if gid.startswith("grp_"):
            role = gid[len("grp_"):]
    return role


#: RADIO IS THE HOST (2026-06-30 HuMo-improve plan, reversing Route-A
#: 2026-06-28). The Route-A workaround that animated a "radio-face" still as
#: HuMo's init_image for the instrumental MUSIC bookend (formerly here as
#: _RADIO_BOOKEND_IMAGE_DEFAULT / _radio_bookend_image()) is RETIRED: the
#: operator eyeballed the result and got a generic human host, not a radio --
#: confirming the original 2026-05-01 BUG-LOCAL-129 finding (HuMo's finetuned
#: weights only animate a face) still holds. The replacement is a structural
#: redirect (below), not another HuMo init-image trick.
#:
#: The engine every announcer_visual / music_visual beat is redirected to when
#: it would otherwise dispatch a HuMo-family (audio_driven_face) engine: the
#: EXISTING, general LTX-2.3 audio-in lane (family audio_conditioned_video,
#: already the per-role DEFAULT for these two roles -- see eng_ltx_av.py
#: default_roles) -- reused, not a new engine.
_NEVER_HUMO_REDIRECT_ENGINE = "ltx_audio_in"

#: The brief-driven radio-HOST FACE object minted once per episode by MetaBrief
#: under OTR_ENABLE_HUMO_HOSTS (must match
#: otr_meta_brief_image_prompt.RADIO_HOST_PORTRAIT_ID + the dispatcher's seed
#: pin). Resolved as the HuMo init_image for the lineless announcer/music
#: bookends when the toggle is ON.
_RADIO_HOST_PORTRAIT_ID = "radio_host_portrait"


def _ltx_radio_face_object_id(role: str) -> str:
    """The object_id of the WIDE radio-FACE still for an ltx_audio_in bookend
    (ltx talking radio-face). Per-role so announcer/music each carry
    their own wide face still; must match otr_meta_brief_image_prompt's mint.
    NOTE: this still is an LTX init asset (ambient motion), NOT a HuMo render --
    the name reserves 'radio_face', not 'humo'."""
    return "still_%s_radio_face_169" % str(role or "")

#: Bridges the VIDEO-ENGINE role vocabulary this module dispatches on
#: (announcer_visual / music_visual / character_video) to the LEDGER
#: speaker_role vocabulary nodes._otr_speaker_role.is_never_humo_role
#: actually checks (character / announcer / music_open / music_close /
#: music_inter). Only ONE representative speaker_role per video role is
#: needed -- all three MUSIC_* values are equally "never humo" in
#: _NEVER_HUMO_ROLES, so MUSIC_OPEN stands in for music_visual.
#: character_video has no entry -- it is never subject to this guard.
_VIDEO_ROLE_NEVER_HUMO_PROXY = {
    "announcer_visual": _SPEAKER_ROLE_ANNOUNCER,
    "music_visual": _SPEAKER_ROLE_MUSIC_OPEN,
}


def _is_never_humo_video_role(role: str) -> bool:
    """True iff ``role`` (a VIDEO-engine role) must never dispatch HuMo, per the
    canonical ledger-speaker_role policy table (_otr_speaker_role.py, 2026-05-01,
    "the radio IS the host") -- consulted through :data:`_VIDEO_ROLE_NEVER_HUMO_PROXY`
    since the two role vocabularies differ. Pure; unknown role -> False."""
    proxy = _VIDEO_ROLE_NEVER_HUMO_PROXY.get(role)
    return bool(proxy) and _is_never_humo_role(proxy)


def _is_cloud_video_engine(engine_id: str) -> bool:
    """True for provider-side video engines.

    Includes Partner/cloud engines plus direct BYO API engines such as Google
    Veo/Omni. These engines do not load local video weights, so local VRAM
    residue cleanup and local HuMo policy redirects must not treat them as
    in-process model routes.
    """
    eid = str(engine_id or "")
    if eid.startswith("cloud_"):
        return True
    try:
        if _vreg.is_registered(eid):
            eng = _vreg.get_engine(eid)
            if bool(getattr(eng, "provider_side", False)):
                return True
            node_key = str(getattr(eng, "node_key", "") or "")
            return node_key.startswith("cloud_")
    except Exception:  # noqa: BLE001 -- unknown/unbuilt engines are not cloud
        return False
    return False


def _prune_strict_text_only_request(req: dict, engine_id: str) -> dict:
    """Remove optional refs the selected provider adapter cannot accept.

    The full episode builder often has per-beat audio and stills available even
    when the selected video engine is text-to-video. Local engines may use some
    of those optional refs, so this is opt-in per adapter; provider adapters can
    declare ``accepts_*`` booleans to keep useful refs like Veo start frames
    while still pruning unsupported audio/video refs before invoke.
    """
    eid = str(engine_id or "")
    try:
        eng = _vreg.get_engine(eid) if eid and _vreg.is_registered(eid) else None
        strict = bool(eng and getattr(eng, "strict_text_only", False))
    except Exception:  # noqa: BLE001 -- unknown engine validates later
        eng = None
        strict = False
    if not strict and eng is None:
        return req

    def _accepts(attr: str, default: bool = True) -> bool:
        if strict:
            return False
        if eng is None or not hasattr(eng, attr):
            return default
        return bool(getattr(eng, attr))

    accepts_audio = _accepts("accepts_audio_ref")
    accepts_base_clip = _accepts("accepts_base_clip_ref")
    accepts_init = _accepts("accepts_init_image")
    accepts_refs = _accepts("accepts_reference_images")
    accepts_last_frame = _accepts("accepts_last_frame")

    removed = []
    if not accepts_audio and req.get("audio_ref") is not None:
        req["audio_ref"] = None
        removed.append("audio_ref")
    if not accepts_base_clip and req.get("base_clip_ref"):
        req["base_clip_ref"] = None
        removed.append("base_clip_ref")

    assets = dict(req.get("asset_refs") or {})
    image_keys = ("init_image", "image", "still")
    for key in image_keys:
        if not accepts_init and key in assets:
            assets.pop(key, None)
            removed.append("asset_refs.%s" % key)
    if not accepts_refs and "reference_images" in assets:
        assets.pop("reference_images", None)
        removed.append("asset_refs.reference_images")
    if not accepts_last_frame and "last_frame" in assets:
        assets.pop("last_frame", None)
        removed.append("asset_refs.last_frame")
    for key in ("reference_videos",):
        if key in assets:
            assets.pop(key, None)
            removed.append("asset_refs.%s" % key)
    req["asset_refs"] = assets

    if removed:
        _LOG.info(
            "[OTR.render_driver] engine %s: pruned unsupported "
            "request ref(s) %s before invoke",
            eid, ", ".join(removed))
    return req


def _is_strict_text_only_engine(engine_id: str) -> bool:
    """True for provider adapters that must receive text only."""
    eid = str(engine_id or "")
    try:
        return bool(
            eid and _vreg.is_registered(eid)
            and getattr(_vreg.get_engine(eid), "strict_text_only", False)
        )
    except Exception:  # noqa: BLE001 -- unknown engine validates later
        return False


def _radio_is_host_redirect_applies(engine_id: str) -> bool:
    """True iff the radio-host guard should rewrite this selected engine.

    The guard is a LOCAL HuMo-family policy escape hatch. Partner cloud engines
    can also be audio-driven-face engines (for example Kling Avatar), but those
    must remain cloud requests; rewriting them to local LTX creates the hybrid
    cloud/local behavior the cloud profiles exist to avoid.
    """
    eid = str(engine_id or "")
    return (
        bool(eid)
        and not _is_cloud_video_engine(eid)
        and engine_family(eid, "") == "audio_driven_face"
    )


def _section_has_local_video_engine(section: dict) -> bool:
    """True when a rendered section includes at least one non-cloud video engine.

    Pure-cloud sections should not run the local VRAM residue freer: Partner
    adapters do not load local video models, and importing cleanup machinery in
    an all-cloud route violates the cloud-only contract even when best-effort.
    """
    for shot in (section or {}).get("shots") or ():
        eid = str((shot or {}).get("engine_id") or "")
        if eid and not _is_cloud_video_engine(eid):
            return True
    return False


def _should_reclaim_between_engines(last_engine: str, this_engine: str) -> bool:
    """Inter-beat local residue reclaim only matters before a local engine load."""
    last = str(last_engine or "")
    this = str(this_engine or "")
    return bool(last and this and this != last and not _is_cloud_video_engine(this))


def _enforce_radio_is_host(shot):
    """Mutate ``shot`` IN PLACE (the ledger's own dict, by reference -- same
    pattern as the OTR_FORCE_ENGINE_MAP override below) so an announcer_visual /
    music_visual beat can NEVER dispatch a local HuMo-family (audio_driven_face)
    engine: "the radio is the host", never a talking human face for the
    bookends (operator 2026-05-01, re-affirmed 2026-06-30 after Route-A's HuMo
    bookend workaround produced a generic face on eyeball -- see the
    :data:`_NEVER_HUMO_REDIRECT_ENGINE` comment above).

    Wires the previously-dormant :func:`nodes._otr_speaker_role.is_never_humo_role`
    (defined 2026-05-01, zero callers until now) into REAL dispatch. A caught
    pick is LOUDLY redirected to :data:`_NEVER_HUMO_REDIRECT_ENGINE` -- never a
    hard-fail: this is a structural POLICY correction applied BEFORE any render
    is attempted, not a render-time failure, so it does not touch the separate
    NO-FALLBACKS rule in :func:`render_shot` (a HARD render failure still raises
    LOUD with no engine swap). Must run before any engine-specific request
    logic (canvas / init_image / audio resolution) so everything downstream
    sees the corrected engine -- call this FIRST in
    :func:`build_request_from_shot`. No-op for every other role/engine; pure
    except for the LOUD log + the in-place mutation.

    OTR_ENABLE_HUMO_HOSTS (2026-07-01 brief-driven radio-host): when the operator
    opts the toggle ON, the announcer/music bookends are ALLOWED to render a HuMo
    radio-host FACE (fed the brief-driven radio_host_portrait still downstream),
    so this redirect is a NO-OP. Default OFF = today's behavior byte-for-byte
    (HuMo on bookends still redirects to the ltx_audio_in animated console)."""
    if os.environ.get("OTR_ENABLE_HUMO_HOSTS", "0") == "1":
        return
    role = _role_of_shot(shot)
    if not _is_never_humo_video_role(role):
        return
    eng_id = str(shot.get("engine_id") or "")
    if not _radio_is_host_redirect_applies(eng_id):
        return
    _LOG.warning(
        "[OTR.render_driver] RADIO-IS-HOST (LOUD): role=%s picked local HuMo-family "
        "engine %r for shot %s -- HuMo's finetuned weights only animate a "
        "FACE (2026-05-01 BUG-LOCAL-129, re-confirmed 2026-06-30: an all-HuMo "
        "bookend eyeballed as a generic human host, not a radio); redirecting "
        "to %r. Pick a non-local-audio_driven_face engine for this role to "
        "silence this override.", role, eng_id, shot.get("shot_id"),
        _NEVER_HUMO_REDIRECT_ENGINE)
    shot["engine_id"] = _NEVER_HUMO_REDIRECT_ENGINE
    shot["family"] = engine_family(_NEVER_HUMO_REDIRECT_ENGINE,
                                   "audio_conditioned_video")


def _cumulative_beat_start(ledger, shot, fps):
    """The beat's start (seconds) in the MASTER mix, as BEAT-ACCURATE as possible
    without per-line timing on the beat itself (operator 2026-06-22: ltx_av should
    condition on the beat's OWN audio, like HuMo). Walk ``video.shots`` in order
    tracking a running master clock: a preceding beat whose LINE carries a real
    ``start_s``/``dur_s`` ANCHORS the clock to that TRUE master position (so the
    opening-music offset + any gaps are respected, not just a raw frame-count sum);
    a beat with no line timing advances the clock by its own duration
    (``target_frame_count``/fps -- ShotRow has no start_s/dur_s, extra=forbid).
    Returns the clock at THIS shot. Bounded + deterministic; clamped >= 0; 0.0 if
    the shot is not found (slice the head -- still bounded, never the whole
    master). NOTE: pooled/looped beats may not be exactly true to the beat
    count -- accepted (operator's call)."""
    shots = ((ledger or {}).get("video") or {}).get("shots") or []
    lines = _line_index(ledger)
    sid = str(shot.get("shot_id") or "")
    f = float(fps) if fps else 25.0
    clock = 0.0
    for s in shots:
        if not isinstance(s, dict):
            continue
        if str(s.get("shot_id") or "") == sid:
            return max(0.0, clock)
        ln = lines.get(_beat_id_for_shot(s), {}) if isinstance(lines, dict) else {}
        _ls = ln.get("start_s") if isinstance(ln, dict) else None
        _ld = ln.get("dur_s") if isinstance(ln, dict) else None
        if _ls is not None and _ld is not None:
            clock = float(_ls) + float(_ld)        # anchor to the TRUE master position
        else:
            n = int(s.get("target_frame_count") or 0)
            if n > 0 and f > 0:
                clock += n / f
    return 0.0


def build_request_from_shot(shot, ledger, *, canvas=None,
                            master_audio_path="", segment_index=0):
    """A per-shot VideoRequest from the ShotLock-planned ledger (the REAL
    episode path). Resolves the character portrait (``init_image``) + the
    per-beat voice audio + the M4 ``text_prompt`` + the audio-derived
    ``target_frame_count`` for THIS shot only. Reuses :func:`build_request` for
    the canonical request shape + canvas, then overrides the prompt, the
    request-hash seed, and the per-beat timing. Pure: it reads the ledger and
    never writes; the frozen audio section is only ever read.

    ``master_audio_path`` (optional): path to the FROZEN master mix (MP4 or
    WAV) from which per-beat audio is sliced when the ledger carries no
    per-line ``*_wav_path``.  Passed in via :func:`run_real_episode` so the
    file is never mutated (read-only ``ffmpeg -i``)."""
    # RADIO IS THE HOST (2026-06-30): must run FIRST, before any engine-keyed
    # branch below reads shot["engine_id"] -- a caught announcer/music + HuMo
    # pick is redirected here so every downstream resolution (init_image,
    # canvas, audio) already sees the corrected engine.
    #
    # FROZEN ROUTE WINS (2026-07-25, chunk 1c). ~30 reads of shot["engine_id"]
    # and shot["family"] follow in this function -- mesh-fodder routing, the
    # scene-still vs portrait choice, the radio-face raise, the still_pan /
    # ltx_audio_in branch, the portrait-vs-wide split, LTX-I2V init selection.
    # They must all see the SAME engine the stills were minted for. When the
    # ledger carries a frozen route, the row already holds it and re-deriving
    # here could only introduce a disagreement, so we verify instead of mutate.
    if not frozen_route_from_ledger(ledger):
        _enforce_radio_is_host(shot)
    # Chunk A2 (visual-style TOTAL COVERAGE): resolve the episode's visual
    # style ONCE per shot, OUTSIDE any swallow (fail-loud law) -- the
    # console/music motion registers below read pack VALUES; the style id is
    # stamped on the request observability for the trace/history rows.
    # ImportError shim only (3A pattern).
    try:
        from .._otr_visual_styles import get_visual_style  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_visual_styles import get_visual_style  # type: ignore
    _vstyle = get_visual_style((ledger or {}).get("meta") or {})
    _line_beat_id = _beat_id_for_shot(shot)
    line = _line_index(ledger).get(_line_beat_id, {})
    # Audio and visual assets share ONE key: the shot's own beat id. They used
    # to diverge here, and closing that split is PBUG-20260811-02's fix.
    _visual_beat_id = _line_beat_id
    # Round 5 F5: the SHOT row carries the ShotLock-normalized char_id (the
    # announcer 'announcer'->cast-row-id join); prefer it, fall back to the
    # raw line value for pre-round-5 planned ledgers.
    char_id = str(shot.get("char_id") or line.get("char_id") or "")
    _eng_id = str(shot.get("engine_id") or "")
    _family = engine_family(_eng_id, "")
    portrait = _portrait_index(ledger).get(char_id, "")
    init_image = portrait
    init_source = "portrait" if portrait else "none"
    # RADIO FACE LOGIC (2026-07-04) -- cross-node provenance guard, fail CLOSED.
    # The image node and this driver each read OTR_ENABLE_HUMO_HOSTS independently.
    # A ledger minted HUMO-OFF stamps the announcer portrait as the FACELESS
    # "radio_object"; if it is later RENDERED HUMO-ON (e.g. the visual-quicktest
    # harness REUSES a frozen ledger across runs), that faceless still would feed
    # HuMo -- the BUG-LOCAL-129 face-only failure. Guard on the POSITIVE: an
    # any portrait-backed announcer_visual + audio_driven_face init MUST carry
    # radio_host_style=="console_face"; anything else (radio_object OR a missing
    # field on an old/frozen ledger) RAISES rather than hand HuMo a faceless init.
    # ShotLock normalizes the role-tag id "announcer" onto the canonical cast-row
    # id (often c01), so role + engine family are the authority here -- never the
    # spelling of char_id.
    if (portrait
            and _family == "audio_driven_face"
            and _role_of_shot(shot) == "announcer_visual"):
        _rh_style = _portrait_style_index(ledger).get(char_id, "")
        if _rh_style != "console_face":
            raise RenderError(
                "RADIO FACE LOGIC: announcer_visual shot %s on a HuMo "
                "(audio_driven_face) engine %r would consume the announcer "
                "portrait as its init, but that still is radio_host_style=%r "
                "(need 'console_face'). A faceless radio_object minted HUMO-OFF "
                "must NOT drive HuMo (BUG-LOCAL-129 face-only failure). Re-mint "
                "the image phase with OTR_ENABLE_HUMO_HOSTS=1 so the announcer "
                "portrait is the animatable dial-face. NO FALLBACK." % (
                    shot.get("shot_id"), _eng_id, _rh_style or "<missing>"))
    # 3D-image-streams (2026-06-21): a requires_mesh_fodder engine (mesh_stage)
    # meshes a SINGLE isolated subject, so it must be fed the clean mesh_fodder
    # still -- NOT the cinematic scene still. The engine_id on the shot is the
    # FINAL routed engine (apply_engine_override / OTR_FORCE_ENGINE_MAP already
    # rewrote it before this builder runs), so reading the capability here is
    # post-override-correct. Resolve fodder by subject (char_id), then beat_id;
    # the scene still becomes the BACKGROUND plate only (consumed downstream),
    # so we MUST skip the _SCENE_INIT_FAMILIES override below for these engines.
    _requires_fodder = bool(
        _eng_id and _vreg.is_registered(_eng_id)
        and getattr(_vreg.get_engine(_eng_id), "requires_mesh_fodder", False))
    _mesh_fodder_missing = False
    _mesh_subject_id = ""
    if _requires_fodder:
        _bid = _visual_beat_id
        _fidx = _mesh_fodder_index(ledger)
        _frows = _mesh_fodder_row_index(ledger)
        _subj = char_id or str(shot.get("mesh_subject_id") or "")
        # SUBJECT POLICY (chunk 6): (a) a beat with a char_id -- INCLUDING the
        # announcer (char_id "announcer") -- meshes that character; (b) a beat
        # with no char (e.g. the b000 music-open) meshes a per-beat story OBJECT
        # under a STABLE "obj_<beat>" id that MATCHES the minted mesh_fodder
        # object's mesh_subject_id; (c) if no fodder was minted at all, the
        # missing-fodder branch below degrades LOUD (never meshes the
        # environment as "uncast"). Mirror the prompt-gen id convention exactly.
        _fodder_row = (_frows.get(_subj) if _subj else None) or _frows.get(_bid)
        _mesh_subject_id = (
            _subj
            or str((_fodder_row or {}).get("mesh_subject_id") or "")
            or str((_fodder_row or {}).get("object_id") or "")
            or ("obj_%s" % _bid)
        )
        _fodder = (_fidx.get(_subj, "") if _subj else "") or _fidx.get(_bid, "")
        if _fodder:
            init_image = _fodder
            init_source = "mesh_fodder"
            _LOG.info(
                "[OTR.render_driver] %s: beat %s meshing CLEAN fodder %s "
                "(subject=%s; scene still is background plate only)",
                _eng_id, _bid, os.path.basename(_fodder), _subj or "?")
        else:
            # LOUD: no clean fodder minted -> do NOT silently mesh the scene
            # still (the clay-blob bug). Keep the portrait/none init; NO
            # FALLBACKS -- a mesh_stage render without usable inputs fails LOUD.
            init_source = "missing_mesh_fodder"
            _mesh_fodder_missing = True
            _LOG.warning(
                "[OTR.render_driver] %s MISSING-FODDER (LOUD): beat %s subject "
                "%r has NO mesh_fodder still in the ledger -- NOT meshing the "
                "scene still (would mesh the environment); init stays %r. "
                "Investigate the image phase fork for this beat.",
                _eng_id, _bid, _subj or char_id, init_image and "portrait" or "none")
    # Family/contract-based init selection (still-spine ST-4 / W6):
    # static_motion + image_to_video shots drift/animate the beat's SCENE STILL
    # (the 6/5 look).  Cloud Seedance lives in audio_conditioned_video but its
    # adapter also hard-requires init_image, so any non-face/non-3D engine that
    # explicitly declares init_image joins this path too.
    # NO-FALLBACK (operator 2026-07-03): a missing scene still for these families
    # FAILS LOUD -- it never silently degrades to the pre-spine portrait init (a
    # scene beat would then show the wrong image). audio_driven_face keeps the
    # portrait by design (not in _SCENE_INIT_FAMILIES); fodder engines are excluded
    # by the guard; text engines are unchanged (LTX text-only by design).
    # `minimax_h3_audio_in` joins ltx_audio_in's exclusion (lane 20,
    # 2026-08-12), and it is a CORRECTNESS fix rather than a preference. This
    # branch OVERWRITES init_image with the beat's wide scene still. On that
    # lane init_image is the reference the model is asked to lip-sync -- it is
    # presented to the tokenizer as `<Picture 1>` and its own prompt says
    # "a medium close shot of <Picture 1> speaking directly to camera" -- so
    # handing it an establishing shot with no face in it does not fail, it
    # renders the wrong identity, silently, on every beat.
    #
    # Caught by the post-coding QA pass: the lane's comment on
    # `_still_spine_requires_scene` claimed it kept the portrait while this
    # generic rule took it anyway, and the test that was meant to prove it was
    # LEXICAL (it grepped the source instead of calling the function) so it
    # passed against the wrong behaviour.
    _engine_scene_init_required = (
        "init_image" in _required_inputs_for_engine(_eng_id, _family)
        and _family != "audio_driven_face"
        and _eng_id not in ("ltx_audio_in", "minimax_h3_audio_in")
    )
    _engine_scene_init_optional = bool(
        _eng_id and _vreg.is_registered(_eng_id)
        and getattr(_vreg.get_engine(_eng_id), "provider_side", False)
        and getattr(_vreg.get_engine(_eng_id), "accepts_still", False)
        and _family != "audio_driven_face"
        and _eng_id not in ("ltx_audio_in", "still_pan", "still_flat", "still_word")
    )
    if ((_family in _SCENE_INIT_FAMILIES
            or _engine_scene_init_required
            or _engine_scene_init_optional)
            and not _requires_fodder):
        # NB: mesh_stage is family=image_to_video (IN _SCENE_INIT_FAMILIES), so
        # without the not-_requires_fodder guard this override would clobber the
        # clean fodder with the scene still and re-introduce the clay blob.
        _bid = _visual_beat_id
        # (still_pool_key read removed 2026-07-01 with the pooling rip --
        # every shot resolves its own per-beat still.)
        _still = _still_index(ledger).get(str(_bid), "")
        if _still:
            init_image = _still
            init_source = "scene_still"
        else:
            raise DeferredImageGapError(
                "[OTR.render_driver] %s-family engine %r shot %s beat %s has NO scene still "
                "in the ledger. NO portrait-init fallback (no-fallback rip) -- a "
                "scene-init engine MUST have its per-beat scene still minted "
                "upstream; fix the image dispatcher / ledger." % (
                    _family, _eng_id, shot.get("shot_id"), _bid)
            )
    # FIX1 / BUG-LOCAL-403 (opener centre BLACK): still_pan is family
    # "static_image_gen" -- in NEITHER _SCENE_INIT_FAMILIES nor the ltx_video
    # branch -- so an opener that picks still_pan for the music slot kept
    # init_image="" (the b000 music-open beat has char_id="") and the cheap
    # family synthesized its dark floor (color=0x0A0E14) => a black centre.
    # Condition still_pan on the beat's SCENE still like the other init-driven
    # engines; LOUD if absent (never a silent black). station_card (the other
    # static_image_gen family) is intentionally NOT included here -- its
    # announcer-card behavior is unchanged.
    # still_pan (slow pan) + still_flat (static hold) are the static_image_gen
    # "show the selected still" engines -- both condition on the beat's scene still
    # so the operator's chosen image (e.g. flux2_klein) is what's displayed.
    # ltx_audio_in (2026-06-26) JOINS them: it is the unified WIDE audio-in LTX lane
    # (render_aspect="wide") that does I2V on the beat's scene still + the shot audio
    # -- the SAME per-beat wide still these engines use (scene_open for the
    # announcer/music BOOKENDS, scene_character for character beats), portrait
    # cleared so it can never leak into a wide frame. Unlike the cheap families it
    # has NO floor: a missing required still fails LOUD in render_clip (no fallbacks).
    # still_word (Sprint B, 2026-07-03) JOINS still_pan/still_flat here: it is a
    # static_image_gen flat-hold engine that conditions on the beat's minted scene
    # still (which carries the word/title prompt from compose_still_word_prompt).
    # Unlike still_flat it has NO floor -- a missing still fails LOUD in
    # its render_clip (_require_still), never a silent black card (no fallbacks).
    # UPDATED 2026-08-11: still_pan LEFT that comparison in lane 16 and
    # still_motion in lane 15 -- both now set _require_still too, so a missing
    # still is a refusal on three of the four -- and lane 17 made it FOUR:
    # still_flat took the refusal too, so no still family paints a dark floor.
    if str(shot.get("engine_id") or "") in ("still_pan", "still_flat", "still_word", "ltx_audio_in"):
        _eng = str(shot.get("engine_id") or "")
        _bid = _visual_beat_id
        _still = _still_index(ledger).get(str(_bid), "")
        # BUG 1 (2026-06-20 operator directive): still_pan / still_flat are
        # LANDSCAPE engines (render_aspect="wide") -- they NEVER condition on the
        # 832x1216 VERTICAL portrait. The 8bc5381 "_spk==character -> portrait"
        # branch pillarboxed the portrait in the 1472x832 frame and the procgen
        # radio-booth floor filled the sides ("radio booth images"). CHARACTER
        # beats now carry a per-beat 16:9 CHARACTER scene still
        # (kind=scene_character, minted in the image phase) so conditioning on the
        # scene still SHOWS the character full-frame. Only HuMo / audio_driven_face
        # (+ the 3D lane) -- the render_aspect="portrait" engines -- use the
        # vertical portrait. A missing still degrades to the cheap family's floor;
        # init_image is cleared so the portrait can NEVER leak into a wide frame.
        # S4 PORTRAIT INIT (2026-07-02, portrait-vs-wide A/B): under the ia2v
        # TALKING register a CHARACTER speech beat conditions on the cast
        # member's PORTRAIT, not the wide scene still -- the scene still
        # leaves the face too small for the audio coupling to articulate
        # (proof7 speech beats: scene-still chars 0.62/0.32/1.27 vs
        # face-forward announcers 4.62/5.51; isolation A/B same audio+prompt:
        # scene 0.57 vs portrait 2.86, lag 0). The 2026-06-20 "wide engines
        # NEVER condition on the vertical portrait" directive is SUPERSEDED
        # here ONLY under RECIPE_IA2V: its canvas-independent guide chain
        # (fixed 1920x1088 center-crop -> longer-edge 1536) COVERS the wide
        # canvas from a vertical portrait -- no pillarbox, no squash
        # (eyeballed: docs/2026-07-02-canonical-ia2v/pw_ab_portrait_frame.png).
        # Single-pass recipes keep the scene still; the directive holds there.
        _s4_portrait = (
            _eng == "ltx_audio_in"
            and _role_of_shot(shot) == "character_video"
            and _ia2v_talking_register_active(_eng))
        if _s4_portrait:
            if not portrait:
                raise DeferredImageGapError(
                    "IA2V TALKING register: character beat %s (char %r) has "
                    "NO portrait in the ledger -- NO FALLBACK to the wide "
                    "scene still (face too small to lip-sync; proof7 A/B "
                    "2026-07-02). Confirm the image phase minted the cast "
                    "portrait." % (_bid, char_id))
            # LOUD aspect guard: record the vertical still entering the wide
            # canvas so any future pillarbox/squash regression is traceable.
            _prow = next(
                (im for im in (((ledger or {}).get("images") or {}).get("images") or [])
                 if isinstance(im, dict) and str(im.get("path") or "") == portrait),
                None)
            _pw = int((_prow or {}).get("w") or 0)
            _ph = int((_prow or {}).get("h") or 0)
            init_image = portrait
            init_source = "character_portrait_ia2v"
            _LOG.warning(
                "[OTR.render_driver] IA2V PORTRAIT INIT (S4): character beat "
                "%s conditions on portrait %s (%sx%s; vertical enters the "
                "canvas-independent 1920x1088 center-crop guide chain -- no "
                "pillarbox under ia2v_canonical; scene still %s NOT used)",
                _bid, os.path.basename(portrait), _pw or "?", _ph or "?",
                os.path.basename(_still) if _still else "<absent>")
        elif _still:
            init_image = _still
            init_source = "scene_still"
            _LOG.info(
                "[OTR.render_driver] %s: beat %s conditioning on scene "
                "still %s (landscape; portrait never used)",
                _eng, _bid, os.path.basename(_still))
        else:
            init_image = ""
            init_source = "missing_scene_still"
            # The engine list in this message was FACTUALLY WRONG from lane 15
            # onward and is emitted at RUNTIME, so it misled exactly the person
            # debugging a missing still. All four still families now refuse.
            _LOG.warning(
                "[OTR.render_driver] %s MISSING-STILL (LOUD): beat %s has NO "
                "scene still in the ledger. Every still family (still_motion / "
                "still_pan / still_flat / still_word) and ltx_audio_in now "
                "FAIL LOUD in render_clip rather than painting a dark floor "
                "(no fallbacks). Investigate the image phase for beat %s.",
                _eng, _bid, _bid)
    # LTX audio-in bookends use a WIDE radio-FACE init under the ia2v talking
    # register. Music remains a radio in every visual mode; when its effective
    # engine is explicitly audio-driven, it gets the same radio-with-lips asset
    # as the announcer (operator 2026-07-12). Other music engines retain the
    # faceless radio scene still resolved above. HuMo is a per-role engine choice,
    # so its global opt-in must not suppress an explicitly selected LTX role.
    if (_ia2v_talking_register_active("ltx_audio_in")
            and str(shot.get("engine_id") or "") == "ltx_audio_in"
            and _role_of_shot(shot) in ("announcer_visual", "music_visual")):
        _abrole = _role_of_shot(shot)
        _fid = _ltx_radio_face_object_id(_abrole)
        _frow = next(
            (im for im in (((ledger or {}).get("images") or {}).get("images") or [])
             if isinstance(im, dict) and str(im.get("object_id") or "") == _fid),
            None)
        _fpath = str((_frow or {}).get("path") or "")
        if not _fpath:
            raise DeferredImageGapError(
                "a talking ltx_audio_in %s bookend REQUIRES the minted wide radio-face still "
                "(MetaBrief mints it when the engine lip-syncs). Missing still %r in the ledger "
                "for %s bookend shot %s -- NO FALLBACK (never black, never a portrait "
                "pillarbox). Confirm MetaBrief minted it + the dispatcher rendered it." % (_abrole, _fid,
                                                          _abrole, shot.get("shot_id")))
        _fw = int((_frow or {}).get("w") or 0)
        _fh = int((_frow or {}).get("h") or 0)
        if _fw and _fh and _fw <= _fh:
            raise RenderError(
                "a talking ltx_audio_in %s bookend REQUIRES the minted wide radio-face still. "
                "radio-face still %r is %dx%d (NOT wide) for "
                "the wide ltx_audio_in %s bookend -- feeding it would pillarbox. NO "
                "FALLBACK: mint a WIDE radio-face still (aspect-follows the bookend "
                "slot)." % (_abrole, _fid, _fw, _fh, _abrole))
        init_image = _fpath
        init_source = "ltx_radio_face"
        _LOG.warning(
            "[OTR.render_driver] LTX-RADIO-FACE: role=%s shot %s conditioning "
            "ltx_audio_in on the WIDE radio-face still %s (LIP-SYNC under the "
            "ia2v_canonical recipe since 2026-07-02)", _abrole, shot.get("shot_id"),
            os.path.basename(_fpath))
    # THE BESPOKE ltx_video STILL BLOCK IS GONE (2026-08-28). It existed only
    # because ltx_video declared `family = "text_to_video"` and so fell outside
    # `_SCENE_INIT_FAMILIES`, which is how every other scene-init engine gets
    # its per-beat still. The engine now declares `family = "image_to_video"`
    # and `required_inputs = ("text_prompt", "init_image")`, so it routes
    # through the shared path above like its siblings, and the
    # `OTR_ENABLE_LTX_I2V` switch it was gated on is retired -- operator
    # ruling: "no switches nor flags, all video models request and ingest
    # stills".
    # Route-A's local HuMo-radio-face music-bookend workaround (2026-06-28) is
    # RETIRED 2026-06-30 (see _enforce_radio_is_host above): a local HuMo
    # music_visual beat redirects to ltx_audio_in before _family is computed.
    # Partner/cloud avatar engines may still legitimately reach this point as
    # audio_driven_face; their own declared init_image requirement is honored by
    # the scene-still branch above.
    # OTR_ENABLE_HUMO_HOSTS (2026-07-01 brief-driven radio-host, chunk 4): a
    # LINELESS announcer/music bookend on a HuMo (audio_driven_face) engine has
    # char_id="" -> no cast portrait -> empty init_image. Under the toggle, feed
    # it the episode's ONE brief-driven radio_host_portrait FACE still (minted
    # once by MetaBrief, kind=portrait, resolved via _portrait_index). Fail LOUD
    # if the toggle is ON but the face still is absent -- NEVER a black-screen
    # fallback. (Toggle OFF: _enforce_radio_is_host already redirected these off
    # HuMo, so this block never triggers -> byte-identical.)
    if (os.environ.get("OTR_ENABLE_HUMO_HOSTS", "0") == "1"
            and not init_image
            and _family == "audio_driven_face"
            and _is_never_humo_video_role(_role_of_shot(shot))):
        _rh = _portrait_index(ledger).get(_RADIO_HOST_PORTRAIT_ID, "")
        if _rh:
            init_image = _rh
            init_source = "radio_host_portrait"
            _LOG.warning(
                "[OTR.render_driver] HUMO-HOSTS: role=%s shot %s conditioning on "
                "the brief-driven radio_host_portrait FACE still %s (the one "
                "animatable host face)", _role_of_shot(shot), shot.get("shot_id"),
                os.path.basename(_rh))
        else:
            raise DeferredImageGapError(
                "OTR_ENABLE_HUMO_HOSTS is ON but no radio_host_portrait FACE "
                "still is in the ledger for the lineless %s bookend shot %s -- "
                "the HuMo host has no init_image. NO FALLBACK (never a black "
                "screen): confirm MetaBrief minted radio_host_portrait (it mints "
                "only when the toggle was ON at the image phase) and the image "
                "dispatcher rendered it." % (_role_of_shot(shot),
                                             shot.get("shot_id")))
    if (not init_image
            and ENGINE_FAMILY.get(str(shot.get("engine_id") or ""))
            == "audio_driven_face"):
        # The b002-class silent miss: a talking-head shot whose char_id has no
        # portrait previously surfaced only as eng_humo's fail-closed error
        # mid-render. Warn at the JOIN so the gap is visible upstream.
        _LOG.warning("[OTR.render_driver] talking-head shot %s char_id=%r has "
                     "NO portrait-index entry -- HuMo will fail closed LOUD "
                     "(NO FALLBACKS)", shot.get("shot_id"), char_id)
    audio = _voice_audio_for_line(line)
    # A PER-LINE VOICE WAV IS SLICED PER SEGMENT TOO (WIRE-W4e, 2026-07-29).
    #
    # W4b/W4c gave the MASTER-slice path its per-segment window, and that left
    # exactly one hole: a beat whose line carries its own clean voice wav takes
    # the branch above and skips the slicer entirely, so every segment of a
    # multi-clip beat received the WHOLE line from its start. On
    # ``audio_driven_face`` -- the lane this matters for -- the assembled beat
    # then speaks the opening syllables once per segment.
    #
    # ``otr_shot_lock._stamp_coverage_plan`` REFUSED such a beat outright for
    # this reason, and said so: "per-segment audio is the prerequisite, not a
    # workaround". This is that prerequisite for the remaining source. The
    # window arithmetic is the same authority as the master path
    # (``coverage_plan.segment_render_window``); only the ORIGIN differs -- a
    # per-line wav starts at its own zero, so nothing is added to the offset,
    # where the master slice adds the beat's ``start_s``.
    if audio and isinstance(shot.get("coverage_plan"), dict):
        try:
            from . import coverage_plan as _cpv  # type: ignore
        except ImportError:  # pragma: no cover -- flat test imports
            import coverage_plan as _cpv  # type: ignore
        _vplan = _cpv.CoveragePlan.from_dict(shot.get("coverage_plan") or {})
        # ``requires_coverage_execution``, not ``is_multi_clip``: a ONE-segment
        # plan that rounded its length up to a legal rung still renders MORE
        # frames than the beat's audio covers, so it needs the same windowed,
        # silence-padded conditioning slice a successor does. Gated on segment
        # count, such a beat was handed the whole raw line against a longer
        # render -- the same defect the router fix closed for video, one layer
        # down. ``segment_render_window`` already handles a single segment.
        if _vplan.requires_coverage_execution:
            _vfps = int(((ledger or {}).get("video") or {}).get("fps")
                        or 25) or 25
            _vwin = _cpv.segment_render_window(
                _vplan, int(segment_index or 0), _vfps)
            _vhash = str(((ledger or {}).get("audio") or {})
                         .get("master_audio_sha256") or "")
            _vcut = _slice_master_audio(audio, _vwin.offset_s, _vwin.copy_s,
                                        master_hash=_vhash,
                                        pad_tail_s=_vwin.pad_s)
            if _vcut:
                _LOG.info("[OTR.render_driver] per-SEGMENT voice slice: %s "
                          "@%.3f+%.3fs (+%.3fs silence) -> %s (beat %s segment "
                          "%d/%d)", os.path.basename(audio), _vwin.offset_s,
                          _vwin.copy_s, _vwin.pad_s, os.path.basename(_vcut),
                          _beat_id_for_shot(shot), int(segment_index or 0),
                          _vplan.segment_count)
                audio = _vcut
            else:
                # NO FALLBACK to the whole line: that is the sync defect this
                # exists to remove, and it would ship as a finished episode.
                raise RenderError(
                    "beat %s segment %d needs its own slice of %s and the "
                    "slicer failed. NO FALLBACK -- handing this segment the "
                    "whole line makes the assembled beat repeat the opening "
                    "syllables once per segment."
                    % (_beat_id_for_shot(shot), int(segment_index or 0),
                       os.path.basename(audio)))
    # Per-beat audio fallback: slice the FROZEN master when no per-line wav.
    # The master is opened read-only by ffmpeg; the slice lands in a temp dir
    # so the master is NEVER mutated (V-1 / audio spine frozen).
    if not audio and master_audio_path and os.path.isfile(master_audio_path):
        bid = _beat_id_for_shot(shot)
        start_s = line.get("start_s")
        dur_s = line.get("dur_s")
        # M3 delta (c): the SYNTHETIC opening-music beat (b000) has no ledger
        # line, so the per-line start_s/dur_s are absent. The audio-reactive lanes
        # (ltx_av_music + viz_green, which paints FROM the audio analysis)
        # need the per-beat slice -- fall back to the SHOT row's start_s/dur_s for
        # THOSE engines only, so every other engine keeps the line-backed slice
        # path byte-identical. (2026-06-18 visualizer soak: b000_music_open reached
        # the engine -- renamed viz_green 2026-06-30 -- with an empty audio_ref and
        # failed LOUD without this.)
        # AMBIENT-AUDIO lanes (ltx_av_music's audio_conditioned_video family + the
        # viz_green scopes) condition on the master MIX, and ltx_av_music HARD-REQUIRES
        # audio_ref -- without it _assert_family_inputs_satisfiable raises
        # FamilyInputGap and the no-fallbacks rule CRASHES the episode (the
        # 2026-06-22 music-beat bug: b006/b013 inter-music beats have no per-line
        # timing). ShotRow is extra=forbid with NO start_s/dur_s, so the old
        # shot.get('start_s') fallback was ALWAYS None. Synthesize a BOUNDED window
        # from the beat's target_frame_count (its audio-derived length) at its
        # cumulative timeline position -- NEVER the whole master (an episode-length
        # WAV would blow up the audio encoder).
        # Route-A's dedicated local HuMo-music-bookend theme-slice carve-out
        # (2026-06-28) is RETIRED 2026-06-30: local HuMo bookends redirect to
        # ltx_audio_in upstream. _uses_ambient_master_audio now owns the valid
        # bookend audio lanes directly: ltx_audio_in/audio_conditioned_video,
        # visualizers, and Partner/cloud avatar engines.
        if ((start_s is None or dur_s is None)
                and _uses_ambient_master_audio(
                    shot.get("engine_id"), _family,
                    _is_character_face_beat(shot), role=_role_of_shot(shot))):
            _afps = int(((ledger or {}).get("video") or {}).get("fps") or 25) or 25
            _an = int(shot.get("target_frame_count") or 0)
            if dur_s is None:
                dur_s = (_an / float(_afps)) if (_an > 0 and _afps > 0) else 4.0
            if start_s is None:
                start_s = _cumulative_beat_start(ledger, shot, _afps)
            _LOG.info("[OTR.render_driver] ambient-audio %s beat %s: synthesized "
                      "BOUNDED master window @%.2f+%.2fs (no per-line timing; "
                      "ShotRow carries none) so audio_ref is satisfied",
                      _eng_id, bid, float(start_s), float(dur_s))
        if (start_s is not None and dur_s is not None
                and float(dur_s) > 0):
            # PER-SEGMENT, NOT PER-BEAT, ON A MULTI-CLIP BEAT (WIRE-W4b,
            # 2026-07-29). Every segment used to be handed the WHOLE beat's
            # slice, so a 3-segment HuMo beat rendered three clips all
            # lip-syncing to the same waveform from the top -- an assembled
            # beat that says the opening of the line three times. The window
            # arithmetic lives in ``coverage_plan.segment_render_window``
            # (pure, CPU-tested, and it is the RENDER window, so a chained
            # successor's dropped head frame gets its own audio too); the only
            # thing this site owns is adding the beat's own ``start_s``.
            #
            # It CANNOT raise a beat out of existence: a single-clip beat has
            # no multi-clip plan and takes the branch below untouched, and a
            # plan this shot could not describe is a stamp error that the
            # planner's own validator already refuses upstream.
            _slice_start, _slice_dur, _slice_pad = float(start_s), float(dur_s), 0.0
            _seg_label = ""
            _seg_plan_raw = shot.get("coverage_plan")
            if isinstance(_seg_plan_raw, dict) and _seg_plan_raw:
                try:
                    from . import coverage_plan as _cp  # type: ignore
                except ImportError:  # pragma: no cover -- flat test imports
                    import coverage_plan as _cp  # type: ignore
                _seg_plan = _cp.CoveragePlan.from_dict(_seg_plan_raw)
                # See the sibling gate above: a one-segment plan owing a tail
                # trim needs its windowed slice too, for the same reason.
                if _seg_plan.requires_coverage_execution:
                    _sfps = int(((ledger or {}).get("video") or {}).get("fps")
                                or 25) or 25
                    _win = _cp.segment_render_window(
                        _seg_plan, int(segment_index or 0), _sfps)
                    _slice_start = float(start_s) + _win.offset_s
                    _slice_dur = _win.copy_s
                    # r4/A4: the conditioning WAV is render_frames long, but
                    # only the untrimmed part is COPIED -- the rest is silence,
                    # never the next segment's speech.
                    _slice_pad = _win.pad_s
                    _seg_label = " segment %d/%d" % (
                        int(segment_index or 0), _seg_plan.segment_count)
                    if _slice_pad > 0:
                        _seg_label += " (+%.3fs silence for the trimmed tail)" \
                            % _slice_pad
            # 7.3 slice key: the master CONTENT hash is the cache identity
            # (a new master at the same path invalidates the slice).
            _mhash = str(((ledger or {}).get("audio") or {})
                         .get("master_audio_sha256") or "")
            sliced = _slice_master_audio(master_audio_path,
                                         _slice_start, _slice_dur,
                                         master_hash=_mhash,
                                         pad_tail_s=_slice_pad)
            if sliced:
                _LOG.info("[OTR.render_driver] per-beat audio: sliced "
                          "%s @%.3f+%.3fs -> %s (beat %s%s)",
                          os.path.basename(master_audio_path),
                          _slice_start, _slice_dur,
                          os.path.basename(sliced), bid, _seg_label)
                audio = sliced
            else:
                _LOG.warning("[OTR.render_driver] per-beat audio slice FAILED "
                             "for beat %s -- HuMo will degrade LOUD", bid)
        else:
            _LOG.warning("[OTR.render_driver] per-beat audio: beat %s has no "
                         "start_s/dur_s on line -- HuMo will degrade LOUD", bid)
    # JUMP SEGMENT N>0 BRINGS ITS OWN STILL (2026-07-26, chunk 6b). This runs
    # LAST, after every branch above has resolved the BEAT's init image,
    # because for a jump segment the beat's still is the wrong answer: it is
    # the image segment 0 already rendered, and reusing it puts the identical
    # frame at both ends of the cut. Resolved BY OBJECT ID off the stamp + the
    # spine's receipt -- never through _still_index, which cannot see a
    # jump_segment row at all. Segment 0 and every single-clip beat return None
    # here and keep exactly the init image they had.
    #
    # NOT FOR A FODDER LANE (2026-07-26 QA panel, both seats). mesh_stage is
    # family=image_to_video, so it IS in _SCENE_INIT_FAMILIES and DOES get
    # per-segment stills minted -- but its init image is the subject-isolated
    # FODDER, and the segment still is its background plate. Overwriting the
    # fodder here would mesh the whole environment: the clay blob the guard at
    # the scene-init branch above exists to prevent, arriving by a second door.
    _segment_still = jump_segment_still_path(ledger, shot, segment_index)
    if _segment_still and not _requires_fodder:
        # LOUD, because several branches above already logged the BEAT's still
        # as this shot's init image and that trail is now wrong. An operator
        # reading the log during the first multi-clip debugging session must
        # see the swap, not infer it.
        _LOG.warning(
            "[OTR.render_driver] JUMP SEGMENT %d of shot %s conditions on its "
            "OWN still %s (was %s) -- resolved by object id off the still-spine "
            "receipt", int(segment_index or 0), shot.get("shot_id"),
            os.path.basename(_segment_still),
            os.path.basename(init_image) if init_image else "<none>")
        init_image = _segment_still
        init_source = "jump_segment_still"
    elif _segment_still:
        _LOG.warning(
            "[OTR.render_driver] JUMP SEGMENT %d of shot %s KEEPS its mesh "
            "fodder as init_image; its segment still %s is the background "
            "plate, not the mesh subject", int(segment_index or 0),
            shot.get("shot_id"), os.path.basename(_segment_still))
    frame_count = segment_render_frames(shot, segment_index)
    req = build_request(shot, {"init_image": init_image, "audio_ref": audio},
                        frame_count, canvas)
    # 3D image streams (chunk 5): hand the mesher the STABLE subject id so its
    # GLB cache keys on the subject (char/object), not the per-beat still hash.
    if _requires_fodder and _mesh_subject_id:
        req["mesh_subject_id"] = _mesh_subject_id
    # ST-4 / pass-02 Gem-3: init observability stamped on the REQUEST's
    # observability dict (the round-5 pattern, schema-real since the W7-pre
    # builder migration); run_episode copies them to trace rows so the W7
    # acceptance check is mechanical.
    if _mesh_fodder_missing:
        # LOUD trace degrade: a 3D mesh beat with no clean fodder minted -- the
        # default "none" stamp (init_image is empty) would HIDE that we refused
        # to mesh the scene still; surface it explicitly (mirrors i2v above).
        req["observability"]["init_source"] = "missing_mesh_fodder"
        # (The separate `mesh_fodder_missing` boolean was removed 2026-08-28:
        # it restated the durable `init_source` stamp on the same line, and
        # only a direct-builder test ever read it.)
    else:
        req["observability"]["init_source"] = (init_source if init_image
                                               else "none")
    req["observability"]["init_image"] = (os.path.basename(init_image)
                                          if init_image else "")
    # Chunk A2 provenance (r2 codex M5): ADDITIVE observability -- which
    # visual style governed this request. prompt_field_source joins below
    # only when a pack field actually fed the prompt text.
    req["observability"]["visual_style"] = _vstyle.style_id
    # FULL-FRAME landscape for EVERY non-talking-head engine (operator look-QA
    # 2026-06-10 + 2026-06-14): build_request's default canvas is the HuMo
    # PORTRAIT (480x832, the accepted talking-head pillarbox). LTX/Wan were given
    # the landscape canvas in 2026-06-10, but the still/floor families
    # (still_pan, station_card, still_motion, viz_green) STILL inherited the
    # 480x832 portrait -> skinny-portrait b-roll pillarboxed in the 16:9 frame
    # (2026-06-14 operator catch on still_pan). Give the composite landscape
    # canvas to every engine EXCEPT the face families: audio_driven_face (HuMo)
    # keeps its accepted portrait pillarbox; lipsync_overlay / character_3d align
    # to a face. Both dims stay /32-friendly for the LTX latent grid; env-
    # overridable via OTR_VIDEO_LANDSCAPE_CANVAS.
    _canvas_fam = engine_family(str(shot.get("engine_id") or ""), "")
    # audio_driven_face (HuMo) renders PORTRAIT by default -> keep the accepted
    # pillarbox. The humo_1.7B_169 variant declares render_aspect='wide' (832x480
    # 16:9), so it JOINS the landscape composite like LTX (the clip scales to
    # fill, no pillarbox). The aspect is the SELECTED engine's identity, not a
    # global toggle. lipsync_overlay / character_3d always align to a face.
    _face_excl = {"audio_driven_face", "lipsync_overlay"}
    if _canvas_fam == "audio_driven_face":
        try:
            _sel_eng = _vreg.get_engine(str(shot.get("engine_id") or ""))
            _face_wide = _aspect_is_wide(
                getattr(_sel_eng, "render_aspect", "portrait"))
        except Exception:  # noqa: BLE001 -- unknown engine -> safe portrait
            _face_wide = False
        if _face_wide:
            _face_excl.discard("audio_driven_face")
    if _canvas_fam not in _face_excl:
        _lc = os.environ.get("OTR_VIDEO_LANDSCAPE_CANVAS", "1472x832")
        try:
            _lw, _lh = (int(x) for x in _lc.lower().split("x", 1))
        except (ValueError, AttributeError):
            _lw, _lh = 1472, 832
        req["canvas"]["w"], req["canvas"]["h"] = _lw, _lh
    # BUG-LOCAL-412 (operator 2026-06-15, "make it byte-identical to 6/5"): LTX-2B
    # renders MUSH above its native 480p. The 6/5 openers/bookends rendered at
    # 832x480 then upscaled to the composite, which is why they animated SHARP;
    # the 2026-06-10 landscape-canvas change pushed LTX to render NATIVELY at
    # 1472x832 -> "starts sharp then gets blurry" (the engine's own note: "0.75
    # re-noises into mush at 1472x832"). Render LTX at its 6/5 native canvas and
    # let OTR_SilentComposite scale the clip to the 1472x832 deliverable. The
    # still/floor families (still_pan etc.) KEEP the full landscape canvas above.
    # Env OTR_LTX_RENDER_CANVAS (default 832x480, the 6/5 value; /32-friendly).
    if str(shot.get("engine_id") or "") == "ltx_video":
        _lxc = os.environ.get("OTR_LTX_RENDER_CANVAS", "832x480")
        try:
            _lxw, _lxh = (int(x) for x in _lxc.lower().split("x", 1))
        except (ValueError, AttributeError):
            _lxw, _lxh = 832, 480
        req["canvas"]["w"], req["canvas"]["h"] = _lxw, _lxh
    # THE ltx_audio_in INLINE CANVAS BRANCH WAS DELETED IN LANE 7 (2026-08-11).
    #
    # It computed the canvas here, RECIPE-DEPENDENTLY -- 832x480 when
    # ia2v_canonical was live, 512x288 otherwise, either one overridable by
    # OTR_LTX_AV_RENDER_CANVAS. That was three channels deciding one number
    # while `declared_render_canvas` below is applied LAST and overrules all of
    # them, so at most one of the three could ever have been true. The comment
    # it carried also asserted the ia2v stage-A base "416x240 (all /32)", which
    # is false -- 240 % 32 == 16 -- and that was S8b-10.
    #
    # The lane now DECLARES (1024, 576) on the adapter (LtxAudioInEngine.
    # render_canvas), which is where the reasoning lives, and disagreement with
    # OTR_LTX_AV_RENDER_CANVAS is a NAMED refusal in the adapter rather than a
    # quiet re-plan here. The live receipts that walked the ladder -- 1280x720
    # failing the /32 gate (proof5b) and 1280x704 breaching the 14.5 GB ceiling
    # at 14,716 MB in the full production pipeline (proof6) -- are preserved at
    # the declaration, because that is now the only place a reader needs to go.
    # A DECLARED RENDER CANVAS WINS (B5, 2026-07-27). LAST in the chain on
    # purpose: every write above is guarded by a mutually exclusive engine id
    # or family, so nothing can clobber this and this clobbers nothing -- and
    # being last means a future branch inserted above cannot silently displace
    # it, which is precisely the defect (O1) this fixes. It reads the ADAPTER's
    # own declaration rather than the family/aspect logic above, so the two
    # mechanisms cannot disagree if a family mapping ever moves.
    #
    # Declared today: `ltx_8gb` (512x288); `ltx_video` (832x480) since
    # 2026-08-02, because its 169-frame decode floor is CANVAS-DEPENDENT, so
    # leaving the canvas to the env branch above left `OTR_LTX_RENDER_CANVAS`
    # able to invalidate a STATIC frame contract without touching engine code;
    # the four HuMo tiers and the three WAN lanes through the 2026-08-11 video
    # transplant; and `ltx_audio_in` (1024x576) in lane 7, whose inline branch
    # was DELETED rather than kept -- the sentence that used to sit here saying
    # it "keeps its env branch until the general resolver lands" is no longer
    # true, and a comment describing a mechanism that is gone is exactly the
    # defect lesson L6 is about.
    _declared = declared_render_canvas(str(shot.get("engine_id") or ""))
    if _declared is not None:
        req["canvas"]["w"], req["canvas"]["h"] = _declared
    _shot_role = str(shot.get("role") or "")
    if not _shot_role:
        # Resilience: older planned ledgers carry the role only inside
        # group_id ("grp_<role>") -- parse it so role-dependent prompt
        # handling never silently skips (2026-06-10 acceptance catch).
        _gid = str(shot.get("group_id") or "")
        if _gid.startswith("grp_"):
            _shot_role = _gid[len("grp_"):]
    creative = shot.get("creative") or {}
    text_prompt = str(creative.get("text_prompt") or "")
    # RESOLVED BEFORE CLASSIFIED (2026-08-27). `engine_family` is keyed on
    # INTERNAL ids only, so a shot row carrying a public or legacy spelling
    # -- `ltx23_low_audio_in`, `ltx23_16gb_audio_in`, both real registered
    # aliases for `ltx_audio_in` -- fell through to its `"abstract"` default.
    # That was survivable while `_fam` only picked fallback prompts; it is
    # NOT survivable now that the LTX character append keys the lip-sync
    # protection on it, because an audio-in lane misread as `abstract` would
    # lose the steadying clause that keeps its mouth in frame. Resolution is
    # for CLASSIFICATION only -- `shot["engine_id"]` is never rewritten and
    # dispatch still uses what ShotLock stamped.
    try:
        from .._otr_shared.public_engines import resolve_engine_id as _fam_rid
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_shared.public_engines import resolve_engine_id as _fam_rid  # type: ignore
    _fam = engine_family(
        str(_fam_rid(str(shot.get("engine_id") or "")) or ""), "")
    # ia2v TALKING register decision, computed ONCE per shot (kibitz r2: the
    # helper instantiates an engine to ask the recipe -- cheap today, but
    # three call sites x every shot invites a regression if the ctor ever
    # grows weight; memoize the bool here and reuse it below).
    _talking_register = _ia2v_talking_register_active(shot.get("engine_id"))
    # ia2v TALKING register, radio-bookend precedence: an explicitly audio-driven
    # announcer or music bookend that happens to carry an shared visual prompt would
    # otherwise take the M4 branch below and the talking swap in the motion
    # branch would never fire. Under the two-stage lip-sync recipe the
    # talking register OUTRANKS M4 on radio bookends -- clearing the text here
    # routes the shot into the matching radio-with-lips motion branch.
    if (text_prompt and _shot_role in ("announcer_visual", "music_visual")
            and _talking_register):
        _LOG.warning(
            "[OTR.render_driver] IA2V TALKING register: %s bookend %s "
            "M4 prompt OUTRANKED by the lip-sync register (M4 kept for "
            "non-ia2v engines)", _shot_role, shot.get("shot_id"))
        text_prompt = ""
    # ROLE-driven (2026-06-26): the shared classifier also catches ltx_audio_in
    # CHARACTER beats (audio_conditioned_video family), so the character
    # fallback prompt + scene-prompt exclusion apply exactly as they do to an
    # audio_driven_face talking head. Announcer/music bookends stay scenes.
    _is_char_face_beat = _is_character_face_beat(shot)
    # THE PROMPT-BUDGET INVARIANT (banana route, QA r2-r4). Every branch below
    # that CAPS its own prompt publishes two things: the number it capped to,
    # and the trailing clause it engineered and expects to survive. The banana
    # funnel further down re-caps to that published number, preserves that
    # published clause, and does nothing else. A branch that publishes NOTHING
    # is never capped -- which is the whole point: the funnel used to cap every
    # branch at `max(188, pre_transform_length)`, so branches that had promised
    # nothing (the operator's verbatim OTR_LTX_RADIO_PROMPT override, the M4
    # wall and its engineered composition tail, the 298-char announcer talking
    # prompt) were handed zero headroom and lost their tails to any growth.
    # Publish INSIDE the sub-path that composed the prompt, never at a shared
    # assignment -- a sibling sub-path must not inherit a budget or a clause it
    # never authored.
    _prompt_char_budget = None
    _prompt_protected_clause = None
    # THE GHOST SIGNAL BRANCH (2026-08-22), and it runs BEFORE the shared visual prompt.
    #
    # CAPABILITY, NEVER AN ENGINE NAME. The branch fires on the engine's declared
    # `prompt_profile`, resolved without assuming every id on a shot is
    # registered -- a frozen ledger can carry an engine this build no longer
    # ships, and `get_engine` on it would raise here rather than at the seam
    # that actually cares.
    #
    # WHY IT OUTRANKS M4. Ghost owns its whole subject: it declares
    # `accepts_still = False`, so nothing was minted for this beat and there is
    # no still carrying the look. The ~900-char M4 identity/scene wall is a
    # prompt written for a lane that HAS a still, and handing it to a 512x288
    # SD1.5 sampler is how you get murk. The composer reads the same authorities
    # M4 does -- the pack, the ledger motion clause, the beat's own mood and arc
    # -- and states them in the order this model can act on.
    #
    # It also has to work for the CAST-TIME TEMPORARY SHOT, which is why it sits
    # before every branch rather than inside one: ShotLock builds a request per
    # beat through this same function long before the durable rows are minted.
    _ghost_composed = False
    #: TRUE ONLY on the Prompt v2 path. It suppresses the common banana funnel
    #: below for this request and nothing else -- v2 has ALREADY applied the
    #: route inside its shared finalizer, and letting the funnel transform an
    #: already-transformed prompt would overwrite a real substitution receipt
    #: with a zero-substitution one (the transform is idempotent, so the second
    #: pass genuinely does nothing and would honestly report doing nothing).
    _ghost_v2_finalized = False
    _ghost_profile = (
        getattr(_vreg.get_engine(_eng_id), "prompt_profile", None)
        if _eng_id and _vreg.is_registered(_eng_id) else None)
    if _ghost_profile == _gsp.GHOST_PROMPT_PROFILE:
        # PROMPT v2 FIRST, and BEFORE the legacy character-sigil requirement.
        # A present object is the whole authority for this beat: it carries the
        # representation, the recurrence motif and the authored drawable leaf,
        # and it requires no sigil. Absence is explicit v1 compatibility for a
        # row frozen before this sprint. A present-but-MALFORMED object is
        # neither -- it fails closed, because a silent v1 downgrade would look
        # exactly like a healthy legacy replay.
        _g_obj = shot.get("ghost_prompt")
        if _g_obj is not None:
            try:
                _gsa.validate_ghost_prompt_object(_g_obj)
            except _gsa.GhostAuthorError as exc:
                raise FamilyInputGap(
                    "Ghost Signal beat %s carries a malformed ghost_prompt "
                    "(%s). A stamped object is the render authority for this "
                    "lane; refusing rather than downgrading to the v1 composer."
                    % (shot.get("shot_id"), exc)) from exc
            # THE CLOUD SAFETY HOOK IS PROVABLY INERT HERE, and it has to be:
            # `_apply_visual_safety_prompt` fires only for a cloud engine, and
            # it would append its clause AFTER this branch finalized, defeating
            # both the component survival check and the token measurement.
            if _is_cloud_video_engine(_eng_id):
                raise FamilyInputGap(
                    "Ghost Signal beat %s resolved to CLOUD engine %r. Prompt "
                    "v2 finalizes its own prompt -- style, motif, leaf, law, "
                    "banana receipt and measured window -- so a cloud peer "
                    "would have a safety clause appended after finalization "
                    "and past every check this branch just made."
                    % (shot.get("shot_id"), _eng_id))
            _g_final = _gsa.finalize_ghost_prompt_v2(
                role=_shot_role, style=_vstyle, mode=_g_obj["mode"],
                motif_cue=_g_obj["motif_cue"],
                drawable_beat=_g_obj["drawable_beat"],
                ledger_meta=(ledger or {}).get("meta") or {})
            _g_positive = str(_g_final["positive"]).strip()
            _g_negative = str(_g_final["negative"]).strip()
            req["text_prompt"] = _g_positive
            req["negative_prompt"] = _g_negative
            _stamp_prompt_meta(req, _gsp.GHOST_PROMPT_SOURCE, _g_positive,
                               beat=_beat_id_for_shot(shot))
            _g_obs = req.setdefault("observability", {})
            _g_obs["prompt_version"] = _gsp.GHOST_PROMPT_VERSION_V2
            _g_obs["negative_sha8"] = hashlib.sha256(
                _g_negative.encode("utf-8")).hexdigest()[:8]
            _g_obs["negative_chars"] = len(_g_negative)
            _g_obs["prompt_slots"] = list(_g_final.get("slots")
                                          or _gsp.GHOST_V2_SLOTS)
            _g_obs["author_version"] = _g_obj["author_version"]
            _g_obs["ghost_schema_version"] = _g_obj["schema_version"]
            _g_obs["ghost_source"] = _g_obj["source"]
            _g_obs["ghost_fallback_reason"] = _g_obj["fallback_reason"]
            _g_obs["ghost_model_id"] = _g_obj["model_id"]
            _g_obs["ghost_mode"] = _g_obj["mode"]
            _g_obs["ghost_request_sha8"] = _g_obj["request_sha256"][:8]
            _g_obs["ghost_output_sha8"] = _g_obj["output_sha256"][:8]
            _g_obs["ghost_drawable_beat"] = _g_obj["drawable_beat"]
            _g_obs["ghost_drawable_beat_sha8"] = _g_obj["output_sha256"][:8]
            _g_obs["ghost_motif_cue"] = _g_obj["motif_cue"]
            _g_obs["positive_clip_tokens"] = _g_final["positive_clip_tokens"]
            _g_obs["positive_clip_windows"] = _g_final["positive_clip_windows"]
            _g_obs["negative_clip_tokens"] = _g_final["negative_clip_tokens"]
            _g_obs["negative_clip_windows"] = _g_final["negative_clip_windows"]
            _g_obs["clip_window_max"] = _g_final["clip_window_max"]
            _g_obs["clip_counter"] = _g_final["clip_counter"]
            # The finalizer's OWN receipt, installed here so the funnel below
            # does not run a second, idempotent transform over the same text.
            _g_obs.update(_g_final["banana_receipt"])
            _ghost_v2_finalized = True
            _ghost_composed = True
            _LOG.warning(
                "[OTR.render_driver] GHOST PROMPT v2: %s beat %s mode=%s "
                "source=%s composed %d chars / %d SD1 tokens in %d window(s), "
                "negative %d chars / %d tokens",
                _shot_role, shot.get("shot_id"), _g_obj["mode"],
                _g_obj["source"], len(_g_positive),
                _g_final["positive_clip_tokens"],
                _g_final["positive_clip_windows"], len(_g_negative),
                _g_final["negative_clip_tokens"])
    if _ghost_profile == _gsp.GHOST_PROMPT_PROFILE and not _ghost_composed:
        # THE v1 COMPATIBILITY PATH, reached only when the row carries NO
        # ghost_prompt -- a ledger frozen before Prompt v2, or a direct unit
        # fixture. Every newly built Ghost preflight and row carries v2, so a
        # production episode never lands here.
        #
        # The DRIVER resolves the live authorities; the composer stays pure and
        # cycle-free. Deliberate: `_ltx_motion_role_key` and the exact-key
        # `motion_registers` lookup already live here, and a second role-to-
        # register table in the pure module is a table that drifts once.
        _g_sids = shot.get("source_line_ids")
        _g_synthetic = (
            str(shot.get("shot_id") or "").endswith(_OPENING_MUSIC_SUFFIX)
            or (isinstance(_g_sids, list) and not _g_sids
                and _shot_role in ("announcer_visual", "music_visual")))
        _g_motion_key = _ltx_motion_role_key(
            _shot_role, shot.get("shot_id"), _g_synthetic)
        _g_pack_register = (_vstyle.motion_registers[_g_motion_key]
                            if _g_motion_key else "")
        # The optional motion-clause pass was ripped 2026-08-27 (it was opt-in,
        # default OFF, set by no profile, and the Ghost lane was excluded from
        # it by capability anyway). Ghost resolves its action through the
        # pack register / beat intent / neutral-action chain, which is what
        # every production episode already did.
        _g_clause = None
        _g_sigil = str(shot.get("subject_sigil") or "")
        # A CHARACTER BEAT WITHOUT ITS DURABLE SIGIL IS A GAP, NOT A DEFAULT.
        # `build_request` seeds every request with a generic 1940s-studio prompt,
        # so required-input PRESENCE cannot prove this branch ran -- which is
        # exactly why the miss is raised by name instead of being papered over by
        # a seed that would render a lane-wrong picture and look fine doing it.
        if _shot_role == "character_video" and not _g_sigil:
            raise FamilyInputGap(
                "Ghost Signal beat %s is a character_video shot with no "
                "subject_sigil. The durable identity is stamped by "
                "otr_shot_lock.build_execution_plan and this lane cannot "
                "compose a character without it -- the generic request seed is "
                "NOT acceptable evidence that a Ghost prompt was built."
                % (shot.get("shot_id"),))
        _g_open_subject = ""
        if _shot_role == "music_visual":
            try:
                from .._otr_story_brief_helpers import (  # type: ignore
                    get_open_subject as _g_open)
            except ImportError:  # pragma: no cover -- flat test imports
                from _otr_story_brief_helpers import (  # type: ignore
                    get_open_subject as _g_open)
            _g_open_subject = _g_open(
                _shot_role, _g_synthetic, (ledger or {}).get("meta") or {},
                _vstyle)
        _g_out = _gsp.compose_ghost_prompt(
            role=_shot_role, style=_vstyle, subject_sigil=_g_sigil,
            motion_clause=_g_clause, pack_motion_fallback=_g_pack_register,
            beat_intent=str(line.get("beat_intent") or ""),
            emotion=str(line.get("traits") or ""),
            story_accent=str(line.get("arc_phase") or ""),
            open_subject=_g_open_subject,
            sigil_seed=_seed_from_hash(
                str(shot.get("render_request_hash") or ""),
                str(shot.get("shot_id") or "")))
        _g_positive = str(_g_out.get("positive") or "").strip()
        _g_negative = str(_g_out.get("negative") or "").strip()
        if not _g_positive or not _g_negative:
            raise FamilyInputGap(
                "Ghost Signal beat %s composed an empty %s prompt -- this lane "
                "owns its entire subject, so an empty composition is a blank "
                "picture, not a default to fall through from."
                % (shot.get("shot_id"),
                   "positive" if not _g_positive else "negative"))
        req["text_prompt"] = _g_positive
        req["negative_prompt"] = _g_negative
        # Published for the common banana funnel below: the number this branch
        # capped to, and the clause it expects to survive. THE PROTECTED CLAUSE
        # IS THE SUBJECT IDENTITY, not the trailing shot law -- the shot law is a
        # constant this composer can always re-emit, while the subject is the one
        # phrase that makes the figure the same figure beat to beat.
        _prompt_char_budget = _gsp.GHOST_PROMPT_MAX_CHARS
        _prompt_protected_clause = (
            _g_sigil or _g_positive.split(",")[0].strip())
        _stamp_prompt_meta(req, _gsp.GHOST_PROMPT_SOURCE, _g_positive,
                           beat=_beat_id_for_shot(shot))
        _g_obs = req.setdefault("observability", {})
        _g_obs["prompt_version"] = _gsp.GHOST_PROMPT_VERSION
        _g_obs["negative_sha8"] = hashlib.sha256(
            _g_negative.encode("utf-8")).hexdigest()[:8]
        _g_obs["negative_chars"] = len(_g_negative)
        _g_obs["prompt_slots"] = list(_g_out.get("slots") or ())
        if _g_sigil:
            _g_obs["subject_sigil_sha8"] = hashlib.sha256(
                _g_sigil.encode("utf-8")).hexdigest()[:8]
        # TRUE ONLY NOW -- after a non-empty positive, a non-empty negative and
        # the ghost_signal prompt_source are all really installed.
        _ghost_composed = (
            bool(req.get("text_prompt")) and bool(req.get("negative_prompt"))
            and _g_obs.get("prompt_source") == _gsp.GHOST_PROMPT_SOURCE)
        if not _ghost_composed:
            raise FamilyInputGap(
                "Ghost Signal beat %s did not install a complete composed "
                "request (positive/negative/prompt_source) -- refusing rather "
                "than letting the generic seed render as if it were Ghost."
                % (shot.get("shot_id"),))
        _LOG.warning(
            "[OTR.render_driver] GHOST SIGNAL: %s beat %s composed %d chars, "
            "negative %d chars, slots=%s", _shot_role, shot.get("shot_id"),
            len(_g_positive), len(_g_negative), ",".join(_g_obs["prompt_slots"]))
    # ===================================================================== #
    # PER-LANE PROMPT DISPATCH (Option B, operator ruling 2026-08-27).
    # ===================================================================== #
    #
    # THE DEFECT THIS FIXES. Every lane used to compose its prompt as
    # ``get("text_prompt") or <the lane's motion default>``, and ShotLock
    # populates ``text_prompt`` unconditionally for every character beat -- so
    # the per-lane motion prompts committed in 65538f41 lived on the dead side
    # of that ``or`` and never once rendered. The live H3 leg of 2026-08-27
    # proved it from production: every character beat logged ``source=m4``.
    #
    # THE RULE: one path per lane, always active. No flag, no shadow state, no
    # legacy route. A lane that declares its OWN ``compose_prompt`` composes
    # here, and the ENTIRE legacy chain below -- the shared visual prompt, the ia2v talking
    # rewrite, the env override, the motion-role branch, the brief+beat branch
    # and both fallbacks -- is skipped for it. Review flagged that omitting this
    # bypass would let a legacy branch silently overwrite the lane's output,
    # which would be the second prompt path the operator forbade.
    #
    # ``type(engine).__dict__`` AND NOT ``hasattr``, deliberately. The engine
    # classes are subclass chains (mime <- foley_plus <- video; h3 silent and
    # h3 audio-in share a base; fastwan subclasses wan). ``hasattr`` would find
    # an INHERITED formatter and hand one lane another lane's motion -- exactly
    # what "each lane independent" forbids. Only a formatter bound to the class
    # itself is visible to dispatch.
    #
    # SCOPE: ``character_video`` only. Announcer and music beats have no
    # ``creative`` row at all (CHARACTER_BEARING_ROLES is character_video
    # alone), so a formatter there would compose from nothing, return blank,
    # and ``finish_joint_av_positive`` raises on a blank positive -- a dead
    # episode at the foley/mime bookends. Those beats keep the motion-register
    # branch they already have.
    _lane_composed = False
    if (not _ghost_composed
            and _shot_role == "character_video"
            and _eng_id and _vreg.is_registered(_eng_id)):
        _lane_engine = _vreg.get_engine(_eng_id)
        _lane_formatter = type(_lane_engine).__dict__.get("compose_prompt")
        if _lane_formatter is not None:
            _lane_line = _line_index(ledger).get(str(_visual_beat_id)) or {}
            # DIALOGUE IS STRUCTURAL, NOT POLICY. A lane that does not preserve
            # dialogue is handed "" here, so it cannot leak a spoken line even
            # if its formatter wanted to. ``_lane_preserves_dialogue`` is the
            # existing authority (it keys on the engine FAMILY); a hand-written
            # membership list would drift from it.
            try:
                from ..otr_shot_lock import (  # type: ignore
                    _lane_preserves_dialogue as _lane_keeps_dialogue)
            except ImportError:  # pragma: no cover -- flat test imports
                from otr_shot_lock import (  # type: ignore
                    _lane_preserves_dialogue as _lane_keeps_dialogue)
            try:
                _keeps = bool(_lane_keeps_dialogue(_eng_id, _shot_role))
            except Exception:  # noqa: BLE001 -- a silent lane is the safe default
                _keeps = False
            # APPEARANCE AND SETTING ARE RESOLVED HERE, NOT STORED.
            # Adding them to the creative object would be a schema extension
            # Option B forbids, and `creative["text_prompt"]` already contains
            # the appearance -- storing a second raw copy invites a formatter
            # emitting it twice, and worse, re-bleeding the subject anchor's
            # "speaking to camera" into the silent lanes. Both resolvers are
            # pure ledger reads.
            try:
                from ..otr_shot_lock import (  # type: ignore
                    _appearance_for_char as _lane_appearance,
                    _read_setting as _lane_setting)
            except ImportError:  # pragma: no cover -- flat test imports
                from otr_shot_lock import (  # type: ignore
                    _appearance_for_char as _lane_appearance,
                    _read_setting as _lane_setting)
            try:
                _lane_look = str(_lane_appearance(
                    ledger, str(shot.get("char_id") or "")) or "")
            except Exception:  # noqa: BLE001 -- a missing look is not fatal
                _lane_look = ""
            try:
                _lane_where = str(_lane_setting(
                    (ledger or {}).get("meta") or {}) or "")
            except Exception:  # noqa: BLE001
                _lane_where = ""
            _lane_inputs = {
                "appearance": _lane_look,
                "setting": _lane_where,
                "expression": str(creative.get("expression") or ""),
                "motion": str(creative.get("motion") or ""),
                "camera": str(creative.get("camera") or ""),
                "text_prompt": str(creative.get("text_prompt") or ""),
                "dialogue": (str(_lane_line.get("text") or "")
                             if _keeps else ""),
                "role": _shot_role,
                "beat_id": str(_visual_beat_id or ""),
                "engine_id": _eng_id,
            }
            _lane_text = str(_lane_formatter(_lane_engine, _lane_inputs) or "")
            # A FORMATTER THAT HANDED BACK THE SHARED PROMPT DID NOT COMPOSE,
            # AND THE RECEIPT MUST NOT SAY IT DID. Every lane's documented rule
            # for a row with no authored leaves is "return text_prompt
            # unchanged" -- correct behaviour, but stamping `engine:<id>` over
            # it would claim the lane wrote something it did not. The stamp is
            # how a live leg proves reachability, so a lie here is expensive:
            # it would read as success on exactly the beats that still fall
            # through.
            if _lane_text.strip() and _lane_text != _lane_inputs["text_prompt"]:
                text_prompt = _lane_text
                _lane_composed = True
            else:
                # A formatter that returns nothing is a BUG in that lane, not a
                # reason to render a blank picture. Fall through to the legacy
                # chain and say so, loudly.
                _LOG.warning(
                    "[OTR.render_driver] lane formatter for %r returned an "
                    "empty prompt on beat %s -- falling back to the shared "
                    "composer; this is a defect in that lane's compose_prompt",
                    _eng_id, _visual_beat_id)

    if _lane_composed:
        # The lane owns its framing, so the generic LTX suffix and the family
        # split are deliberately NOT applied here. Style cue still runs (one
        # owner, unchanged), and the joint-AV / safety / banana finalizers
        # further down are untouched and each still run exactly once.
        #
        # A PINNED OPENER OUTRANKS THE CUE'S POSITION, NOT THE CUE ITSELF
        # (2026-08-28). `_prefix_video_style_cue` PREPENDS, and the two H3
        # lanes require `H3_REFERENCE_OPENER` verbatim as the FIRST thing in
        # the prompt -- the official one-image I2VA grammar. Prepending the cue
        # made "must begin exactly" false on every non-default style pack:
        # proven on the anime pack, where the prompt became "anime style. For
        # the target video, ...". `eng_minimax_h3.py` already documented these
        # lanes as exempt from the prefix; nothing implemented it.
        #
        # The cue is SEATED AFTER THE OPENER rather than dropped, so both
        # contracts hold at once -- an exemption would silently lose the pack's
        # look on two lanes. The default sci_fi_radio pack yields an empty cue,
        # which is why this never surfaced in production.
        text_prompt = _style_cue_after_pinned_opener(_vstyle, text_prompt)
        req["text_prompt"] = text_prompt
        _stamp_prompt_meta(req, "engine:%s" % _eng_id, text_prompt,
                           subsource=str(creative.get("source") or ""),
                           beat=_beat_id_for_shot(shot))
    elif text_prompt and not _ghost_composed:
        # Authored M4 visual vocabulary is not a publication gate. Preserve it
        # verbatim; framing/style composition below may add context but never
        # deletes story-world objects by means of a Python word list.
        # 2026-06-16 framing fix (roundtable pass03): LTX character/person beats
        # drift the subject's head out of frame over the clip. Append a POSITIVE
        # composition clause -- "stable centered subject" penalizes X/Y drift but
        # leaves the boomerang Z push-pull; NO "head"/"frame"/crop tokens (those
        # PLANT the artifact in this stack -- the c01 lesson). Gated to LTX
        # non-open person beats (announcer/music opens are radio-set objects, no
        # person). The dominant lever is OTR_LTX_I2V_STRENGTH (0.75 -> ~0.85);
        # this clause is the cheap, low-risk insurance alongside it.
        # ia2v TALKING register (lips-dont-talk fix, 2026-07-02): a CHARACTER
        # face beat on the two-stage recipe gets a COMPACT talking-forward
        # prompt -- the ~900-char M4 identity/scene wall drowned the speech
        # tokens (probe P4: scene-register prompts collapse articulation and
        # can hallucinate on-video title text). Keep a short identity
        # fragment (the M4's own opening clause), lead the motion with the
        # talking clause, hard-cap at the proven 240-char budget. Every
        # other engine (HuMo, wan, ltx_video) keeps the full M4 verbatim.
        if _is_char_face_beat and _talking_register:
            # fragment = the shared prompt's first sentence; else last full CLAUSE under
            # the remaining budget (kibitz r2: comma-heavy identity openings
            # have no period -- never hard-cut mid-word). Non-default visual
            # packs get a two-word cue first; the fragment shrinks so the
            # proven IA2V talking clause still stays intact.
            _style_cue = _compact_style_talking_cue(_vstyle)
            _cue_prefix = ("%s " % _style_cue) if _style_cue else ""
            _frag_budget = (
                _LTX_MOTION_PROMPT_MAX
                - len(_cue_prefix)
                - len(_IA2V_TALKING_CLAUSE_CHARACTER)
                - 1
            )
            _frag = text_prompt[:max(40, min(120, _frag_budget))]
            if "." in _frag:
                _frag = _frag.rsplit(".", 1)[0]
            elif "," in _frag and len(text_prompt) > 120:
                _frag = _frag.rsplit(",", 1)[0]
            # r3 guard: a degenerate fragment (e.g. a lone separator) must
            # never yield a leading-period prompt. The clause begins with
            # "is talking..." so the fragment flows into it grammatically
            # ("...Lead Astronomer is talking to the viewer, ...").
            _frag = _frag.strip(" ,.") or "a person"
            _talk = "%s%s %s" % (
                _cue_prefix, _frag, _IA2V_TALKING_CLAUSE_CHARACTER)
            text_prompt = _talk
            # This sub-path -- and only this one -- capped to 240 and engineered
            # the talking clause to survive (_frag shrinks above so it does).
            _prompt_char_budget = _LTX_MOTION_PROMPT_MAX
            _prompt_protected_clause = _IA2V_TALKING_CLAUSE_CHARACTER
            _LOG.warning(
                "[OTR.render_driver] IA2V TALKING register: character beat "
                "%s shared visual prompt -> compact talking prompt (%d chars, style_cue=%r)",
                _beat_id_for_shot(shot), len(text_prompt), _style_cue)
        elif (str(shot.get("engine_id") or "").startswith("ltx")
                and _shot_role not in ("announcer_visual", "music_visual")):
            # SPLIT BY WHAT THE LANE SELLS (2026-08-27, motion bake-in). The
            # audio-in LTX lane keeps the steadying clause: its product is
            # the lip-sync, and "stable centered" is protective framing for
            # a mouth the audio is driving. The SILENT ltx25 lanes drop it:
            # on a lane with no mouth to protect, "stable centered subject,
            # comfortably composed" was an instruction to hold still -- the
            # authored half of the silly pan. Face-visible and headroom stay
            # everywhere (the round-5 b002 no-person catch is about framing,
            # not stillness).
            if _fam in ("audio_driven_face", "audio_conditioned_video"):
                text_prompt = (text_prompt.rstrip().rstrip(",")
                               + ", stable centered subject, full face clearly "
                               "visible, generous headroom, comfortably composed")
            else:
                text_prompt = (text_prompt.rstrip().rstrip(",")
                               + ", full face clearly visible, generous "
                               "headroom, the subject in real motion")
        _pre_cue_chars = len(text_prompt)
        text_prompt = _prefix_video_style_cue(_vstyle, text_prompt)
        # A cue the branch legitimately PREFIXED is not charged against the
        # body it already capped. On the default sci_fi_radio pack the cue is
        # "" so this is 0; on the compact-talking path the cue is already
        # front-loaded inside the 240 budget, so the helper short-circuits and
        # this is 0 there too. It matters on a NON-DEFAULT pack, which is
        # exactly where an uncharged prefix would re-cap on arrival.
        if _prompt_char_budget is not None:
            _prompt_char_budget += len(text_prompt) - _pre_cue_chars
        req["text_prompt"] = text_prompt
        # "shared_video_prompting_engine", not the old undefined "m4": this is the prompt
        # ShotLock composes and EVERY lane shares unless it declares its own
        # compose_prompt. Reads against "engine:<id>" in the same log.
        _stamp_prompt_meta(req, "shared_video_prompting_engine", text_prompt,
                           subsource=str(creative.get("source") or ""),
                           beat=_beat_id_for_shot(shot))
    elif not _ghost_composed and (_is_char_face_beat
                                 or _fam == "audio_driven_face"):
        # HuMo-seam ticket Part C: a FACE beat with NO shared visual prompt is
        # the proven microphone re-introduction path -- the build_request
        # generic studio default is inappropriate for a character beat. Use a
        # world-neutral deterministic fallback; the announcer keeps the radio
        # default (radio-styled by design). Never silent. (2026-06-26: the
        # _is_char_face_beat arm also routes ltx_audio_in CHARACTER beats here
        # so they get the character fallback, not the generic radio default.)
        if _is_char_face_beat:
            _LOG.warning(
                "[OTR.render_driver] HuMo character beat %s carries NO M4 "
                "creative prompt (ShotLock seam gap) -- rendering on the "
                "gear-free character fallback prompt (LOUD)",
                _beat_id_for_shot(shot))
            _cf_fallback = _CHAR_FACE_FALLBACK_PROMPT
            if _talking_register:
                # kibitz r3 must-fix: under the ia2v lip-sync recipe the
                # scene-register fallback is exactly the prompt class probe
                # P4 proved collapses articulation -- the seam-gap fallback
                # must talk too.
                _cf_fallback = ("close-up cinematic portrait of a person "
                                "who %s" % _IA2V_TALKING_CLAUSE_CHARACTER)
            _cf_fallback = _prefix_video_style_cue(_vstyle, _cf_fallback)
            req["text_prompt"] = _cf_fallback
            _stamp_prompt_meta(req, "default_character", _cf_fallback,
                               beat=_beat_id_for_shot(shot))
        else:
            _default_prompt = _prefix_video_style_cue(
                _vstyle, req.get("text_prompt", ""))
            req["text_prompt"] = _default_prompt
            _stamp_prompt_meta(req, "default", _default_prompt,
                               beat=_beat_id_for_shot(shot))
    # SCENE PROMPTS for text-driven engines (gap-audit fix F2, roundtable-
    # hardened: docs/2026-06-10-brief-downstream-gaps/). Any ltx_video /
    # wan_i2v shot with NO writer creative prompt gets a prompt grounded in
    # THE EPISODE'S OWN BRIEF -- the source of the old episodes' scenic,
    # varied opens -- finished with the brief's era tail under the LTX char
    # budget. Covers ALL roles (announcer/music opens AND character),
    # killing the generic "a 1940s radio studio"
    # default for text engines. Precedence: shared visual prompt (finished at
    # ShotLock) > OTR_LTX_RADIO_PROMPT (operator override, VERBATIM, no
    # finishing) > brief-composed + finished. (_shot_role parsed above, once.)
    # RESOLVED ONCE, HERE, FOR EVERY MEMBERSHIP TEST BELOW (2026-08-27). All
    # four consumers -- strict-text-only, the two Google sets and the LTX scene
    # allowlist -- compare against INTERNAL ids, so a shot row carrying a public
    # menu string matched none of them: the beat skipped scene composition
    # entirely and shipped `build_request`'s hardcoded "a 1940s radio studio"
    # with `prompt_source` never stamped, which is the exact degrade this
    # change exists to close. Resolution is for the COMPARISONS only --
    # `shot["engine_id"]` is not rewritten and dispatch still uses what
    # ShotLock stamped. `foley_stems.is_foley_route` resolves before comparing
    # for the same reason and says so.
    try:
        from .._otr_shared.public_engines import resolve_engine_id as _rid
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_shared.public_engines import resolve_engine_id as _rid  # type: ignore
    _engine_id = str(_rid(str(shot.get("engine_id") or "")) or "")
    _strict_text_only = _is_strict_text_only_engine(_engine_id)
    _google_text_provider = _engine_id in _GOOGLE_SILENT_TEXT_PROVIDERS
    _google_prompt_provider = _engine_id in _GOOGLE_PROVIDER_PROMPT_ENGINES
    if ((_engine_id in ("ltx_video", "ltx25_video", "wan_i2v", "ltx_audio_in",
                        # THE TWO LANES THAT KEEP THE MODEL'S OWN AUDIO
                        # (2026-08-26). They ARE the LTX 2.5 picture graph, so
                        # they compose scene prompts exactly like ltx25_video.
                        # Omitted, they matched no branch at all: `roles =
                        # ROLES` makes them legal on the announcer and music
                        # bookends, those roles arrive with `text_prompt`
                        # cleared, and `_is_char_face_beat` is False -- so the
                        # beat shipped `build_request`'s hardcoded "a 1940s
                        # radio studio" default with `prompt_source` never
                        # stamped. That is the exact degrade
                        # `test_ltx25_HQ_open_keeps_the_role_motion_prompt`
                        # was written to stop for `ltx25_video`.
                        "ltx25_foley_plus", "ltx25_mime")
            or _google_prompt_provider
            or _strict_text_only)
            and not text_prompt and not _is_char_face_beat
            # Ghost already composed this shot's positive AND negative from
            # the same authorities; letting the scene branch run would
            # overwrite the positive and silently orphan the negative.
            and not _ghost_composed):
        # Round 5 F2: synthetic-open detection by STRUCTURE -- the ShotLock
        # beat-id suffix is definitive; empty source_line_ids counts only
        # for OPEN roles (a hypothetical provider/b-roll shot without source
        # lines must not inherit the radio-open subject).
        _sids = shot.get("source_line_ids")
        _suffix_hit = str(shot.get("shot_id") or "").endswith(
            _OPENING_MUSIC_SUFFIX)
        _no_sids = isinstance(_sids, list) and not _sids
        _is_synthetic_open = (_suffix_hit
                              or (_no_sids and _shot_role in
                                  ("announcer_visual", "music_visual")))
        _is_open = (_is_synthetic_open
                    or _shot_role in ("announcer_visual", "music_visual"))
        _override = (os.environ.get("OTR_LTX_RADIO_PROMPT", "").strip()
                     if _is_open else "")
        if _override:
            _LOG.warning("[OTR.render_driver] LTX SCENE: %s beat %s prompt "
                         "= OTR_LTX_RADIO_PROMPT operator override "
                         "(verbatim except the banana route, unfinished)",
                         _shot_role, shot.get("shot_id"))
            req["text_prompt"] = _override
            _stamp_prompt_meta(req, "env", _override,
                               beat=_beat_id_for_shot(shot))
        elif _is_open:
            # 6/5 MOTION-CENTRIC restoration (BUG-LOCAL-112): the announcer /
            # music radio-console beats render a MOTION-ONLY prompt. The i2v
            # anchor carries the LOOK from the FLUX still, so the prompt's only
            # job is to MOVE. The refactor had led with a scene-DESCRIPTIVE
            # subject ("a 1940s radio station studio, glowing warmly...") which
            # the model reads as "render this set" -> flat pans on the
            # conditioned still. An optional short atmosphere fragment appends
            # AFTER the motion verbs, dropped if it breaks the 240-char budget.
            _motion_key = _ltx_motion_role_key(
                _shot_role, shot.get("shot_id"), _is_synthetic_open)
            # Chunk A2: the motion VALUE comes from the style pack (exact-key
            # indexing -- the loader guarantees the 4 console keys, and the
            # old silent `or ...["announcer"]` fallback is RETIRED per r2
            # codex S3: a missing key raises LOUD, never remaps to announcer).
            # SUBSTITUTION ORDER (r3): pack motion value FIRST, THEN the
            # _talking_swap override -- the probe-locked IA2V talking prompt
            # is a VERBATIM Python constant (P8) and always outranks the pack.
            scene_prompt = _vstyle.motion_registers[_motion_key]
            _field_source = "motion_registers:%s" % _motion_key
            # ia2v TALKING register: both explicitly audio-driven radio bookends
            # use the lip-sync prompt and the matching radio-with-lips still.
            # Single-pass or non-audio-driven music stays on its faceless
            # console-motion register.
            _talking_swap = (_shot_role in ("announcer_visual", "music_visual")
                             and _talking_register)
            if _talking_swap:
                scene_prompt = _IA2V_TALKING_PROMPT_ANNOUNCER
                _field_source = "python:ia2v_talking"
                _LOG.warning(
                    "[OTR.render_driver] IA2V TALKING register: %s "
                    "bookend %s prompt swapped to the lip-sync register",
                    _shot_role, shot.get("shot_id"))
            _meta = (ledger or {}).get("meta") or {}
            _terms = (_meta.get("story_brief_terms")
                      if isinstance(_meta, dict) else None) or {}
            _atmo = ", ".join([str(t).strip() for t in
                               (_terms.get("atmosphere") or [])
                               if str(t).strip()][:1])
            # kibitz r2: the talking register stays PURE -- no atmosphere
            # fragment rides the lip-sync prompt (mood tokens dilute the
            # speech narration the recipe follows).
            if (_atmo and not _talking_swap
                    and len(scene_prompt) + len(_atmo) + 7
                    <= _LTX_MOTION_PROMPT_MAX):
                scene_prompt = f"{scene_prompt} {_atmo} mood."
            if (_google_text_provider
                    and _field_source.startswith("motion_registers:")):
                try:
                    from .._otr_story_brief_helpers import (  # type: ignore
                        finish_visual_prompt, get_open_subject)
                except ImportError:  # pragma: no cover -- flat test imports
                    from _otr_story_brief_helpers import (  # type: ignore
                        finish_visual_prompt, get_open_subject)
                _meta = (ledger or {}).get("meta") or {}
                _subject = get_open_subject(
                    _shot_role, _is_synthetic_open, _meta, style=_vstyle)
                _role_hint = (
                    "single luminous radio receiver as the only visible "
                    "subject")
                if _shot_role == "announcer_visual":
                    _role_hint = (
                        "anthropomorphic radio receiver as the only visible "
                        "host subject")
                scene_prompt = finish_visual_prompt(
                    _meta,
                    f"{_subject}, {_role_hint}, {scene_prompt}",
                    max_chars=620, style_tail=True, era_profile="still",
                    style=_vstyle)
                # Published INSIDE the google block on purpose: this is the only
                # one of the four sub-paths reaching the assignment below that
                # caps at all. (_field_source is REASSIGNED just below, so a
                # post-block test for "motion_registers:" would be false exactly
                # when the 620 cap did apply.) No engineered trailing clause.
                _prompt_char_budget = 620
                _field_source = (
                    "open_subjects:%s+motion_registers:%s" %
                    (("synthetic" if _is_synthetic_open
                      else ("announcer" if _shot_role == "announcer_visual"
                            else "default")), _motion_key))
            _pre_cue_chars = len(scene_prompt)
            scene_prompt = _prefix_video_style_cue(_vstyle, scene_prompt)
            if _prompt_char_budget is not None:
                _prompt_char_budget += len(scene_prompt) - _pre_cue_chars
            _LOG.warning("[OTR.render_driver] LTX MOTION: %s beat %s motion-"
                         "centric prompt (role=%s, %d chars): %.90s...",
                         _shot_role, shot.get("shot_id"), _motion_key,
                         len(scene_prompt), scene_prompt)
            req["text_prompt"] = scene_prompt
            req["observability"]["prompt_field_source"] = _field_source
            _stamp_prompt_meta(req, "motion_role", scene_prompt,
                               beat=_beat_id_for_shot(shot))
        else:
            try:
                from .._otr_story_brief_helpers import (  # type: ignore
                    finish_visual_prompt, get_story_brief_ltx)
            except ImportError:  # pragma: no cover -- flat test imports
                from _otr_story_brief_helpers import (  # type: ignore
                    finish_visual_prompt, get_story_brief_ltx)
            _meta = (ledger or {}).get("meta") or {}
            _terms = _meta.get("story_brief_terms") or {}

            def _term_join(key, n):
                raw = _terms.get(key) if isinstance(_terms, dict) else None
                return ", ".join([str(t).strip() for t in (raw or [])
                                  if str(t).strip()][:n])

            # Non-open text-engine roles keep the brief
            # logline core + beat clauses; the announcer/music OPEN roles use
            # the motion-centric branch above (the i2v still carries the look).
            # THE CORE TAKES ITS SHIPPED 90 CHARS. The variable budget that
            # used to live here existed solely to make room for the opt-in
            # motion clause, which was ripped 2026-08-27; with no clause to
            # accommodate, `get_story_brief_ltx`'s own default is the whole
            # answer and the arithmetic that shrank it has nothing to shrink
            # for. This is the value the non-override path always used, so
            # every production render is byte-identical across the rip.
            core = get_story_brief_ltx(_meta, max_chars=90)
            if not core:
                _setting = _term_join("setting", 2)
                core = ("cinematic establishing shot"
                        + (f", {_setting}" if _setting else ""))
            # ONE PATH. The override branch that used to sit here belonged to
            # the ripped motion-clause pass; with that gone this is the only
            # composition, and it is the one every production render already
            # took. "no on-screen text" stays last -- the P4 probe observed
            # on-video text hallucination on this lane, so the steer is not
            # optional decoration.
            clauses = list(_beat_clauses(line, shot.get("shot_id")))
            clauses.extend(["slow cinematic camera drift", "no on-screen text"])
            scene_prompt = finish_visual_prompt(
                _meta, f"{core}, {', '.join(clauses)}",
                max_chars=188, style_tail=False, era_profile="still")
            _prompt_char_budget = 188
            # The tail this branch engineered and expects to survive the banana
            # funnel's re-cap.
            _prompt_protected_clause = "slow cinematic camera drift"
            _pre_cue_chars = len(scene_prompt)
            scene_prompt = _prefix_video_style_cue(_vstyle, scene_prompt)
            # Budget is EXTENDED by exactly the cue's length, not re-truncated
            # to 188 -- and that is deliberate. `_prompt_char_budget` is not a
            # cap on this prompt; it is the budget published to the BANANA
            # re-cap below (`_banana_cap`), which only fires when the
            # substitution CROSSES it. The 188 governs the COMPOSED CONTENT;
            # the style cue is additive and sits outside it, so extending the
            # budget is what stops the banana cap from trimming the cue away.
            # (Confirmed 2026-08-08, follow-up chip closed: intended, not a
            # defect. Same idiom at the two sibling branches above.)
            _prompt_char_budget += len(scene_prompt) - _pre_cue_chars
            _LOG.warning("[OTR.render_driver] LTX SCENE: %s beat %s prompt "
                         "composed from the episode brief (%d chars): "
                         "%.90s...", _shot_role, shot.get("shot_id"),
                         len(scene_prompt), scene_prompt)
            req["text_prompt"] = scene_prompt
            _stamp_prompt_meta(req, "brief+beat", scene_prompt,
                               beat=_beat_id_for_shot(shot))
    # THE JOINT-AV AUDIO REQUIREMENT (2026-08-26). `ltx25_foley_plus` and
    # `ltx25_mime` KEEP the audio LTX 2.5 generates alongside the picture, and
    # BOTH halves are conditioned on this one positive string -- so a prompt
    # that never mentions sound leaves the model to score the scene from
    # whatever it guesses. Neither lane is audio-IN: no dialogue reaches here,
    # and the suffix forbids voices outright.
    #
    # WHY THIS SEAM. Every branch above writes its result under a different
    # local name (`_g_positive`, `text_prompt`, `_cf_fallback`,
    # `_default_prompt`, `_override`, `scene_prompt`) but all of them assign
    # into `req["text_prompt"]`. That key is the ONE place they converge, and
    # this is the last point before the banana route, so the finisher runs
    # exactly once over whatever actually won.
    # Reuses the id resolved once above -- one normalisation for the whole
    # request builder, not a second copy of the same map.
    _jav_engine = _engine_id
    if _jav_engine in ("ltx25_foley_plus", "ltx25_mime"):
        try:
            from .eng_ltx25 import (finish_joint_av_positive,
                                    identity_leaks_in,
                                    joint_av_prompt_is_finished,
                                    sounds_named_in)
        except ImportError:  # pragma: no cover -- flat test imports
            from eng_ltx25 import (  # type: ignore
                finish_joint_av_positive, identity_leaks_in,
                joint_av_prompt_is_finished, sounds_named_in)
        # NO MOOD READ ANY MORE (operator, 2026-08-28). Mime used to lead its
        # tail with the brief's music_mood_terms and ask for "instrumental
        # scene score" -- a CATEGORY, which is precisely the defect that made
        # the foley bed choose voices. The operator collapsed the two lanes to
        # one prompt: "foley / mime same thing, they use the new foley
        # prompting ... the only difference between foley and mime is the mux
        # layer". The brief's mood terms still drive the MUSIC bookends through
        # `_otr_music_prompt`, which was always their real owner.
        _jav_before = str(req.get("text_prompt") or "")
        _jav_after = finish_joint_av_positive(_jav_engine, _jav_before)
        # THE PROTECTION IS UNCONDITIONAL, THE MUTATION IS NOT (2026-08-27,
        # found independently by two panel lanes). These two lines used to sit
        # inside the `!=` branch below, so a prompt that ALREADY carried its
        # suffix -- an idempotent re-entry, or a pre-finished fixture -- took
        # the no-change path and left the scene branch's published 188-char
        # budget standing. The banana re-cap would then trim the prompt back to
        # 188 and throw away the mandatory non-speech tail, which is the one
        # clause the whole seam exists to protect. A lane that keeps its audio
        # owes that tail whether or not THIS call is what appended it.
        _prompt_char_budget = None
        _prompt_protected_clause = None
        _jav_obs = req.setdefault("observability", {})
        # THE RECEIPT HAS TO BE PROVABLE, NOT ASSERTED (r3 finding, 2026-08-28).
        # This field used to be stamped "finished" unconditionally, so a prompt
        # that came back WITHOUT a sound frame -- an operator override already
        # ending in the no-voice clause, say -- still reported finished. That is
        # a false claim in the one field used to prove the lane received its
        # audio requirement, and evidence in this repo is cited by hash.
        # `joint_av_prompt_is_finished` is the one owner of that judgement
        # since the 2026-08-29 golden-shape reordering (sound frame PRESENT
        # plus the no-voice clause LAST -- the frame-and-clause pair is no
        # longer contiguous at the tail on a normally composed prompt), and
        # the sounds receipt reads the FINISHED string by phrase membership,
        # because the composer now seats the sounds itself and a cue re-scan
        # would match the phrases' own words.
        _jav_ok = joint_av_prompt_is_finished(_jav_after)
        _jav_obs["joint_av_prompt"] = "finished" if _jav_ok else "UNFINISHED"
        _jav_obs["joint_av_sounds"] = sounds_named_in(_jav_after)
        if not _jav_ok:
            _LOG.warning(
                "[OTR.render_driver] JOINT-AV: %s beat %s did NOT end with the "
                "canonical audio terminator -- the lane keeps its audio and "
                "owes that clause", _jav_engine, _beat_id_for_shot(shot))

        # THE MODEL SPEAKS WHAT IT READS (2026-08-28, proven on a published
        # episode: a mime beat said "Queen of the Fairies" aloud because that
        # was the opening of her character_description). The two formatters no
        # longer pass appearance through, so this guard exists for every OTHER
        # path -- a setting naming a character's bower, a motion clause
        # carrying a name, an operator override, a future formatter. It
        # REPORTS: a wrong sound is a bad episode, a refused render is no
        # episode, and episodes in obs are the operator's success signal.
        try:
            from ..otr_shot_lock import (  # type: ignore
                _appearance_for_char as _jav_look_fn)
        except ImportError:  # pragma: no cover -- flat test imports
            try:
                from otr_shot_lock import (  # type: ignore
                    _appearance_for_char as _jav_look_fn)
            except ImportError:
                _jav_look_fn = None
        _jav_look = ""
        if _jav_look_fn is not None:
            try:
                _jav_look = str(_jav_look_fn(
                    ledger, str(shot.get("char_id") or "")) or "")
            except Exception:  # noqa: BLE001 -- a missing look is not fatal
                _jav_look = ""
        _jav_names = []
        for _row in ((ledger or {}).get("cast") or []):
            if isinstance(_row, dict):
                _nm = str(_row.get("name") or "").strip()
                if _nm:
                    _jav_names.append(_nm)
        _jav_leaks = identity_leaks_in(
            _jav_after, appearance=_jav_look, names=_jav_names)
        if _jav_leaks:
            _jav_obs["joint_av_identity_leak"] = list(_jav_leaks)
            _LOG.warning(
                "[OTR.render_driver] JOINT-AV IDENTITY LEAK: %s beat %s -- %s. "
                "This lane decodes audio from the same latent as the picture, "
                "so text it reads can be SPOKEN.",
                _jav_engine, _beat_id_for_shot(shot), "; ".join(_jav_leaks))
        if _jav_after != _jav_before:
            req["text_prompt"] = _jav_after
            # Restamp the digest and length ONLY, following the banana
            # precedent below. `_stamp_prompt_meta` would overwrite
            # prompt_source/prompt_subsource, and the composing branch's
            # provenance is the truth about where this prompt came from: the
            # audio tail finishes it, it does not author it. Without this the
            # receipt would describe the pre-tail text while a different
            # string rendered -- and evidence here is cited by hash.
            _jav_obs["prompt_sha8"] = hashlib.sha256(
                _jav_after.encode("utf-8")).hexdigest()[:8]
            _jav_obs["prompt_chars"] = len(_jav_after)
            _LOG.info(
                "[OTR.render_driver] JOINT-AV: %s beat %s positive finished "
                "with its audio requirement (%d -> %d chars)",
                _jav_engine, _beat_id_for_shot(shot),
                len(_jav_before), len(_jav_after))
    _apply_visual_safety_prompt(req, shot)
    # THE BANANA ROUTE (docs/2026-08-06-BUILD-SPEC-banana-route.md) -- the
    # PINNED ordering (QA ruling 4): safety hook above -> banana transform ->
    # phrase-safe cap -> assign final text_prompt -> restamp sha8/chars and the
    # receipt WITHOUT logging -> only then the seed derivation below. The gate
    # is ambient (env) + ledger (bank idiom), so every prebuilt segment request
    # through the run_real_episode partial sees the same state -- no flag
    # threading exists to forget.
    # UNSTRIPPED: the receipt digests must hash the literal prompt bytes the
    # request carries (Sonnet QA advisory) -- only the emptiness CHECK strips.
    _banana_prompt = str(req.get("text_prompt") or "")
    if _banana_prompt.strip() and not _ghost_v2_finalized:
        try:
            from .._otr_banana_route import (
                apply as _banana_apply, banana_gate as _banana_gate,
                cap_phrase_safe as _banana_cap,
                off_receipt as _banana_off_receipt,
                receipt_keys as _banana_receipt_keys)
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_banana_route import (  # type: ignore
                apply as _banana_apply, banana_gate as _banana_gate,
                cap_phrase_safe as _banana_cap,
                off_receipt as _banana_off_receipt,
                receipt_keys as _banana_receipt_keys)
        _bmeta = ((ledger or {}).get("meta") or {})
        _bkey = str(_bmeta.get("freeze_timestamp") or "")
        _bobs = req.setdefault("observability", {})
        if _banana_gate(_bmeta, lane="video"):
            # shield_quoted_card_text=False, and EXPLICITLY rather than by
            # default: no still_word card composes on the video lane (a card's
            # words travel in the minted still, never in a video text_prompt),
            # so a quoted span here is always decoration. Passing it explicitly
            # also keeps ShotLock's cast-time preflight -- which calls this
            # same builder per beat -- on the identical decision, so a future
            # change to the default cannot silently desynchronize the two.
            _bres = _banana_apply(_banana_prompt, variety_key=_bkey,
                                  shield_quoted_card_text=False)
            # Re-cap ONLY to the budget the composing branch published, and
            # ONLY when the transform CROSSED it (pre <= budget < post). An
            # unpublished budget means the branch promised nothing, so the text
            # passes through untouched; a prompt already over budget before the
            # route ran is a composer matter, not ours to repair; and a shrink
            # (revolver -> banana is -2, the common case) never re-caps.
            _btext = _bres.text
            if (_prompt_char_budget is not None
                    and len(_banana_prompt) <= _prompt_char_budget
                    < len(_bres.text)):
                _btext = _banana_cap(_bres.text, _prompt_char_budget,
                                     _prompt_protected_clause)
            # banana_substitutions counts what the TRANSFORM did, so a
            # substitution the cap later trimmed away is still counted -- the
            # receipt describes the transform step, and sha256_after below
            # describes the text that shipped.
            _bkeys = _banana_receipt_keys(_bres)
            if _btext != _bres.text:
                # digest recomputed AFTER capping (QA ruling 3)
                _bkeys["banana_sha256_after"] = hashlib.sha256(
                    _btext.encode("utf-8")).hexdigest()
            if _btext != _banana_prompt:
                req["text_prompt"] = _btext
                # restamp WITHOUT logging (_stamp_prompt_meta logs every call)
                _bobs["prompt_sha8"] = hashlib.sha256(
                    _btext.encode("utf-8")).hexdigest()[:8]
                _bobs["prompt_chars"] = len(_btext)
            _bobs.update(_bkeys)
        else:
            _bobs.update(_banana_off_receipt(_banana_prompt, variety_key=_bkey))
    req_hash = (shot.get("render_request_hash")
                or (shot.get("cache_keys") or {}).get("request_hash"))
    # PER SEGMENT, not per shot (2026-08-02). The seed was derived from the
    # request hash and the shot id alone, so every segment of a multi-clip beat
    # sampled from the SAME seed. HuMo starts each call from fresh noise and
    # carries no motion state, so an identical seed plus an identical reference
    # portrait regenerates a near-identical opening -- which is precisely the
    # snap-back-to-the-same-pose signature at every cut.
    #
    # External research (2026-08-02, HuMo/LTX report, grade A): a fixed seed
    # buys REPEATABILITY, not continuity, and re-using it "may make the reset
    # more visibly repetitive". Identity is carried by HuMo's native reference
    # conditioning -- the shared portrait on ``ref_image`` -- and not by the
    # seed, so varying the seed per segment costs no identity and breaks up the
    # repeated opening.
    #
    # Segment 0 keeps the historical value, so every single-clip beat in the
    # archive re-renders byte-identically; only successors move.
    _seg_index = int(segment_index or 0)
    _video_seed = _seed_from_hash(req_hash, shot.get("shot_id"))
    if _seg_index:
        _video_seed = _seed_from_hash(req_hash, "%s#seg%d"
                                      % (shot.get("shot_id"), _seg_index))
    req["seed_bundle"] = {"request_seed": _video_seed}
    # `video_seed_segment_index` was removed 2026-08-28. It was never projected
    # into the durable trace, and it would have MISLED if it were: multi-segment
    # requests are rebuilt per segment, so this original request's index 0 is
    # not the segment a reader would be looking at. The per-segment rows carry
    # their own `segment_index`, which is the honest one.
    # Operator 2026-06-12: surface the per-beat LTX/video sampler seed (the
    # deterministic request-hash seed the engines render with) in the trace so
    # future renders are apples-to-apples with the 6/5 baseline.
    req["observability"]["video_seed"] = _video_seed
    # Carry the per-beat timing so the render node can slice the frozen master
    # mix when the ledger has no per-line wav (audio_ref is None in that case).
    # SYNTHETIC shots (the opening-music scene, 2026-06-10) have no ledger
    # line; the shot row itself carries start_s/dur_s -- fall back to it so
    # the positioned timeline keeps every row placed. W7-pre migration: the
    # schema field is ``target_duration_s`` (never ``dur_s``); a missing value
    # is OMITTED (the Timing default 0.0 applies) -- an explicit None fails
    # ``model_validate``.
    _start = (line.get("start_s") if line.get("start_s") is not None
              else shot.get("start_s"))
    if _start is not None:
        req["timing"]["start_s"] = float(_start)
    _dur = (line.get("dur_s") if line.get("dur_s") is not None
            else shot.get("dur_s"))
    if _dur is not None:
        req["timing"]["target_duration_s"] = float(_dur)
    # Thread the beat identity for downstream cache keys (3D plan 7.3: the
    # CURVE key needs line_id) -- the schema-real Timing.source_line_ids.
    _sids = shot.get("source_line_ids")
    req["timing"]["source_line_ids"] = (
        [str(s) for s in _sids] if isinstance(_sids, list) and _sids
        else ([_beat_id_for_shot(shot)] if _beat_id_for_shot(shot) else []))
    # char_id rides in conditioning_refs (W7-pre migration: a top-level
    # char_id is a schema extra; VideoRequest is extra="forbid").
    if char_id:
        req["conditioning_refs"]["char_id"] = char_id
    _prune_strict_text_only_request(req, shot.get("engine_id"))
    return req


# --------------------------------------------------------------------------- #
# The in-process render (the GPU slice)
# --------------------------------------------------------------------------- #
def _present_request_tokens(request):
    """The role_compat input tokens a request dict actually carries (mirrors
    ``schemas.VideoRequest._present_input_tokens`` for plain dicts)."""
    get = request.get if isinstance(request, dict) else (
        lambda k, d=None: getattr(request, k, d))
    present = set()
    if get("text_prompt"):
        present.add("text_prompt")
    if "init_image" in (get("asset_refs") or {}):
        present.add("init_image")
    if get("audio_ref") is not None:
        present.add("audio_ref")
    if get("base_clip_ref"):
        present.add("base_clip_ref")
    return present


def _assert_family_inputs_satisfiable(engine_name, request):
    """p3 down-chain request shape (3D plan 7.0): before attempting a fallback
    CANDIDATE, re-validate the ONE request against the candidate FAMILY's
    required inputs. A family whose requirements the request cannot satisfy
    (e.g. ``lipsync_overlay`` needs ``base_clip_ref`` a ``character_3d``
    request lacks) raises :class:`FamilyInputGap` -- the chain SKIPS it LOUDLY
    (decision + restamp) instead of feeding the wrong-shaped request to the
    engine. The no-input floor families are always satisfiable, so termination
    holds."""
    fam = engine_family(engine_name, "")
    required = _required_inputs_for_engine(engine_name, fam)
    present = _present_request_tokens(request)
    if fam == "static_image_gen":
        if not ({"text_prompt", "init_image"} & present):
            raise FamilyInputGap(
                "candidate %r (family %s) needs text_prompt or init_image; "
                "the request carries neither -- LOUD skip down the chain"
                % (engine_name, fam))
        return
    missing = [t for t in required if t not in present]
    if missing:
        raise FamilyInputGap(
            "candidate %r (family %s) requires input(s) %s the request does "
            "not carry -- LOUD skip down the chain (never feed a wrong-shaped "
            "request to an engine)" % (engine_name, fam, missing))


def _render_one(engine_name, request, *, force_oom, host_caps=None,
                profile=None, segment=None):
    """Attempt ONE candidate engine: assert_usable -> prepare -> render_clip ->
    canonicalize, always teardown. ``force_oom`` raises BEFORE any work (the
    soak's deterministic mid-episode OOM -- it precedes the family-input check
    so the soak's expected OOM trail is exactly preserved). Raises on failure;
    returns the canonical clip dict on success.

    S4 platform-portability (2026-07-10): ``host_caps``/``profile`` carry
    REAL host facts (build_host_caps) + the ledger-stamped v2 device/dtype
    policy down to the adapter boundary -- the empty-dict boundary here was
    the campaign's most consequential wiring catch. ``None`` (direct debug
    callers) degrades to the old empty dicts.

    CHUNK 5 (2026-07-26): the prepare/teardown bracket now belongs to a
    :class:`beat_session.BeatSession`. ``segment`` is a
    :class:`beat_session.SegmentSlot` -- the beat's OPEN session plus WHICH
    segment this render is -- when a caller is walking a coverage plan. Every
    segment then renders from the handles the first one loaded, and the
    teardown happens once, in that caller's outer ``finally``. The slot is one
    argument rather than three so that "a session with no segment index" is a
    state nobody can construct; the session itself refuses anything that is
    not the next segment, in range, on the beat the slot claims.

    Passing no session keeps the historical bracket: one prepare, one render,
    one teardown, in that order, with teardown gated on handles having been
    taken. ONE deliberate delta -- ``prepare`` now receives a populated
    ``session_ctx`` (``beat_id``/``segment_count``/``multi_clip``) where it
    used to receive ``{}``, so an adapter can size for four clips before it
    loads. Every shipped adapter accepts and ignores that argument today."""
    if force_oom:
        raise OomSignal("forced soak OOM on %s" % engine_name)
    _assert_family_inputs_satisfiable(engine_name, request)
    # A RETIRED id must fail with the NAMED policy error BEFORE the generic
    # not-registered LookupError -- this is the boundary the shipped
    # /otr/video_render_single HTTP endpoint reaches (rip-sfx 2026-08-06).
    # Resolved FOR THE GUARD ONLY (idempotent on internal ids), so a future
    # retired id that also carries a public alias cannot slip past; the
    # registry checks below still see the caller's own name unchanged.
    from .._otr_shared.public_engines import (
        check_retired_engine, resolve_engine_id)
    check_retired_engine(resolve_engine_id(engine_name))
    if not _vreg.is_registered(engine_name):
        raise LookupError("engine %r is not registered" % engine_name)
    eng = _vreg.get_engine(engine_name)
    _caps = host_caps if host_caps is not None else {}
    _prof = profile if profile is not None else {}
    try:
        from . import beat_session as _bs  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        import beat_session as _bs  # type: ignore
    # M3 delta (d): pass the request as the assert_usable template so an
    # engine that validates request-shaped inputs (the LTX-AV av_dims gate
    # on request.canvas) sees the real canvas. Guarded -- a legacy adapter
    # whose assert_usable predates the request_template kwarg keeps working
    # (TypeError -> retry without it); real usability rejections raise
    # EngineUnusable, never TypeError, so this never masks one.
    #
    # It runs PER SEGMENT, before the handles are touched: it validates THIS
    # request against the engine, and a later segment's canvas is not the
    # first one's.
    try:
        eng.assert_usable(host_caps=_caps, profile=_prof,
                          request_template=request)
    except TypeError:
        eng.assert_usable(host_caps=_caps, profile=_prof)
    if segment is not None:
        if getattr(segment.session, "engine", None) is not eng:
            raise RenderError(
                "beat session holds engine %r but this segment renders on "
                "%r -- one session, one engine. Reusing handles across a "
                "route change is how a beat ends up half in one model and "
                "half in another. NO FALLBACK."
                % (getattr(getattr(segment.session, "engine", None),
                           "name", "?"), engine_name))
        prepared = segment.begin()
        raw = eng.render_clip(request, prepared)
        return eng.canonicalize(raw, request, _prof)
    # Name the beat even on the single-clip path: the adapter's session_ctx
    # and the teardown log both read better with a real id than with "?"
    # (2026-07-26 QA panel).
    _rid = (request.get("shot_id") if isinstance(request, dict)
            else getattr(request, "shot_id", ""))
    with _bs.BeatSession(eng, host_caps=_caps, profile=_prof,
                         beat_id=str(_rid or "")) as sess:
        raw = eng.render_clip(request, sess.prepared)
        return eng.canonicalize(raw, request, _prof)


def _assert_beat_affordable(shot, prebuilt):
    """ONE admission boundary for a coverage-planned beat, before any load.

    THE HOLE THIS CLOSES. ``_planned_length`` deliberately never consults the
    VRAM predictor -- a planned segment's length is arithmetic the rest of the
    beat was built around, not a preference it may re-negotiate. That was
    survivable while planned segments were rare. Once ``wan_ti2v``,
    ``fastwan_8gb`` and ``ltx_8gb`` became planning-capped it meant the only
    ENFORCING guard was bypassed on the path doing most of the rendering, and
    ``VramPeakProbe`` does not cover it ("sampled + logged, never enforced").
    An in-process CUDA OOM corrupts the allocator, so this is the direction that
    must never be left open.

    WHY HERE. Every segment's request now exists with its final frame count and
    canvas, and nothing has been loaded yet. So free VRAM is read ONCE, clean,
    with no hoist correction -- there is nothing hoisted to correct for. That is
    a different accounting from ``_floor_length``, which runs AFTER ``prepare()``
    and must add ``hoisted_vram_mb`` back precisely because the weights it is
    about to use are already resident and would otherwise be charged twice.
    Mixing those two conventions is what produced a refusal at "affordable 20
    frames (free=9675 MB)" on an engine that had shipped a published episode.

    WHY A MISSING COST ROW DOES NOT SILENTLY PASS. ``_DEFAULT_FRAME_COST`` makes
    an unmeasured engine look exactly like a calibrated one: the prediction runs,
    returns a limit, and the limit describes a different model. Enforcing a
    borrowed number is worse than not enforcing -- it refuses good renders and
    admits bad ones with equal confidence. So engines WITHOUT a measured row are
    reported as unenforced rather than judged against someone else's cost.

    Returns the admission record for the beat receipt. Raises
    :class:`motion_common.MotionBudgetError` when a measured row says a planned
    segment does not fit.
    """
    engine_id = str(shot.get("engine_id") or "")
    record = {"engine_id": engine_id, "checked": [], "enforced": False,
              "reason": "", "free_vram_mb": None}
    if not prebuilt:
        record["reason"] = "no segments"
        return record

    if not _mc.cost_row_may_refuse(engine_id):
        # Named, visible, and NOT a refusal.
        #
        # This asks whether a row may REFUSE, not whether one exists -- a
        # distinction that cost a live campaign leg. ``wan_ti2v`` HAS a row and
        # that row is disqualified: at every realistic free-VRAM level it
        # refuses every segment length the coverage planner produces. Gating on
        # mere existence re-armed, at this new call site, the exact refusal that
        # ``_planned_length`` had already stopped issuing.
        #
        # Most engines have no row at all, and refusing those would ground the
        # roster to guard none. What this must never do is look guarded.
        record["reason"] = (
            "no QUALIFIED cost row for %r -- admission NOT enforced for this "
            "beat (a row that is absent, or present but disqualified, refuses "
            "and admits with equal confidence)" % engine_id)
        return record

    free_mb = _mc.free_vram_mb()
    record["free_vram_mb"] = free_mb
    if not free_mb or float(free_mb) <= 0:
        # No NVML / no torch (the CPU box). Geometry only, and say so.
        record["reason"] = "free VRAM unreadable -- admission not enforced"
        return record

    for segment, request in prebuilt:
        canvas = (request or {}).get("canvas") or {}
        width = int(canvas.get("w") or 0)
        height = int(canvas.get("h") or 0)
        frames = int(getattr(segment, "render_frames", 0) or 0)
        if not (width and height and frames):
            continue
        _mc.assert_frame_affordable(free_mb, frames, width, height, engine_id)
        record["checked"].append(
            {"segment_index": int(getattr(segment, "index", 0) or 0),
             "render_frames": frames, "canvas": "%dx%d" % (width, height)})
    record["enforced"] = bool(record["checked"])
    return record


def render_shot(shot, request, *, oom_engines=frozenset(), oom_shot_id=None,
                host_caps=None, profile=None, segment=None):
    """Render ONE shot with its selected engine. NO FALLBACKS (operator
    2026-06-16 'this is art, not a space shuttle'; hardened 2026-07-02 NO
    fallbacks / NO auto-defaults directive): a HARD render failure RAISES
    :class:`RenderError` LOUD -- there is NO engine swap and NO still-image
    floor, so a proven model path must prove itself. A forced soak OOM simply
    raises loud like any other failure.

    Returns ``(clip, shot, [engine], vram_used_mb)``."""
    sid = shot["shot_id"]
    eng = shot["engine_id"]
    out_shot = dict(shot)
    force = (sid == oom_shot_id and eng in (oom_engines or frozenset()))
    try:
        clip = _render_one(eng, request, force_oom=force,
                           host_caps=host_caps, profile=profile,
                           segment=segment)
    except Exception as exc:              # noqa: BLE001 - no fallback: fail LOUD
        kind = _rt.FailureKind.OOM if force else classify_failure(exc)
        _LOG.error(
            "[OTR video] render FAILED (no fallback) shot %s engine %s: %s: %s",
            sid, eng, type(exc).__name__,
            str(exc).splitlines()[0] if str(exc) else "")
        raise RenderError(
            "shot %s engine %r failed to render; fallbacks are disabled (%s) -- "
            "fix the engine or its inputs: %s" % (sid, eng, kind, exc)) from exc
    # Prefer the engine's MEASURED render-window peak (LTX-AV threads it onto the
    # clip via VramPeakProbe) over the instantaneous post-render read, so the
    # episode report records the true render-phase peak; fall back when absent.
    clip_peak = clip.get("vram_peak_mb") if isinstance(clip, dict) else None
    return clip, out_shot, [eng], (clip_peak or _mc.vram_used_mb())


def render_beat_coverage(shot, ledger, *, request=None, request_builder=None,
                         canvas=None, oom_engines=frozenset(),
                         oom_shot_id=None, host_caps=None, profile=None):
    """Render ONE BEAT -- as one clip, or as its plan's segments assembled.

    Multi-clip coverage chunk 6c/6d (2026-07-26). This is the loop the whole
    block has been building toward, and the shape of it is the point:

    * ONE :class:`beat_session.BeatSession` wraps every segment, so the beat
      loads its weights once and releases them once, in one outer ``finally``;
    * each segment gets its OWN request (``segment_index=``), which carries its
      own still and its own length;
    * the TERMINAL TRANSACTION happens INSIDE the loop -- a CHAIN successor
      begins on the frame its predecessor just ended on, extracted immediately
      after that segment renders, because segment N+1 cannot wait for a
      post-episode pass to learn where it starts;
    * the segments are assembled and PROVEN before the beat is handed back.

    A beat with no plan, or a single-clip plan, takes the historical path
    exactly: one request, one ``render_shot``, no session threaded, nothing
    assembled. That is every beat today.

    Returns ``(clip, out_shot, attempts, vram_used_mb)`` -- the same tuple
    :func:`render_shot` returns, because the caller must not care how many
    renders it took to make one beat.
    """
    try:
        # ``acceptance`` owns the beat frame arithmetic, and it lives THERE
        # rather than here so the grader can reach it: a test pins that module's
        # import list to exactly ["__future__"], so it can never import this
        # one. Importing it here instead gives the mint side and the grading
        # side ONE implementation -- which is the point, because a receipt the
        # grader cannot re-derive is a number it has to take on faith.
        from . import acceptance as _acceptance  # type: ignore
        from . import beat_session as _bs  # type: ignore
        from . import coverage_plan as _cp  # type: ignore
        from . import foley_stems as _foley  # type: ignore
        from . import wan_shared as _ws  # type: ignore
        from . import wrapper_bridge as _wb2  # type: ignore
        from ._tmp import otr_engine_tmp_path as _tmp_path  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        import acceptance as _acceptance  # type: ignore
        import beat_session as _bs  # type: ignore
        import coverage_plan as _cp  # type: ignore
        import foley_stems as _foley  # type: ignore
        import wan_shared as _ws  # type: ignore
        import wrapper_bridge as _wb2  # type: ignore
        from _tmp import otr_engine_tmp_path as _tmp_path  # type: ignore

    raw_plan = shot.get("coverage_plan")
    plan = None
    if isinstance(raw_plan, dict) and raw_plan:
        plan = _cp.CoveragePlan.from_dict(raw_plan)
    if plan is None or not plan.requires_coverage_execution:
        # THE HISTORICAL PATH, byte for byte. The caller has usually already
        # built this request (it needs it for the trace and the audio-motion
        # profile), so it is used as-is rather than rebuilt -- rebuilding it
        # here would be a second derivation of one request.
        #
        # The predicate was ``is_multi_clip`` until 2026-08-02, which asked the
        # wrong question: a ONE-segment plan still owes a tail trim whenever its
        # length was rounded up to a legal rung, and the trim is the coverage
        # assembler's job. Such beats came down here and kept their surplus
        # frames, so the video outran its audio. See
        # ``CoveragePlan.requires_coverage_execution`` for why the question is
        # what the plan OWES rather than how many pieces it is in.
        single = request if request is not None else request_builder(
            shot, ledger, canvas=canvas)
        return render_shot(shot, single, oom_engines=oom_engines,
                           oom_shot_id=oom_shot_id, host_caps=host_caps,
                           profile=profile)

    if request_builder is None:
        raise RenderError(
            "shot %s carries a %d-segment coverage plan but no request builder "
            "was passed, so its segments cannot be given their own stills and "
            "lengths. A synthetic single-request caller cannot drive a "
            "multi-clip beat. NO FALLBACK."
            % (shot.get("shot_id"), plan.segment_count))
    engine_id = str(shot.get("engine_id") or "")
    # Multi-clip session creation is its own ingress (a frozen ledger may carry
    # a stale engine id): a RETIRED id gets the NAMED refusal, never the
    # generic not-registered error (rip-sfx 2026-08-06). Resolved for the
    # guard only -- idempotent on the internal ids real ledgers carry.
    from .._otr_shared.public_engines import (
        check_retired_engine, resolve_engine_id)
    check_retired_engine(resolve_engine_id(engine_id))
    if not _vreg.is_registered(engine_id):
        raise RenderError(
            "shot %s carries a %d-segment coverage plan but engine %r is not "
            "registered, so there is nothing to open a beat session on."
            % (shot.get("shot_id"), plan.segment_count, engine_id))
    engine = _vreg.get_engine(engine_id)
    # THE CANVAS IS JUDGED BEFORE THE WEIGHTS LOAD (B5, 2026-07-27).
    #
    # ``BeatSession`` below PREPARES -- it loads a 6.34 GiB checkpoint -- and
    # the per-segment ``request_builder`` call that would otherwise surface a
    # malformed declaration runs INSIDE that session. Same pure function the
    # request builder calls, so the two boundaries cannot disagree about what
    # a legal canvas is.
    #
    # Honestly stated: since B5 reads a STATIC declaration rather than the
    # ledger, this can only fire on an adapter whose own declaration is broken
    # -- it is not a per-episode condition. It stays because the ordering is
    # the contract the judgment asked for, and because a broken declaration
    # should cost nothing to discover.
    declared_render_canvas(engine_id)
    beat_id = str(shot.get("beat_id") or shot.get("shot_id") or "")
    chains = plan.join_mode == _cp.JOIN_CHAIN

    # EVERY SEGMENT REQUEST IS BUILT BEFORE THE LEASE IS TAKEN (WIRE-W4d, r3:
    # "Prebuild and validate all segment requests and audio slices BEFORE
    # entering BeatSession; only terminal-image chaining stays in the render
    # loop").
    #
    # The builder is not cheap and it is not pure: it resolves stills off the
    # ledger and it SHELLS OUT TO FFMPEG to cut each segment's conditioning
    # WAV out of the frozen master. Run inside the session, that filesystem
    # and subprocess work happened while the cross-process GPU lease was held
    # and a 14B UNET sat resident -- once per segment, between renders. The
    # lease is the scarcest thing this build owns; nothing that does not need
    # the GPU should be inside it.
    #
    # It is also where a bad request SHOULD surface. A builder that raises on
    # segment 2 used to do it after two renders and a 6 GiB load; now the beat
    # refuses before anything is taken.
    #
    # The CHAIN's terminal-frame substitution deliberately stays in the loop:
    # segment N's init image is segment N-1's last rendered frame, which does
    # not exist yet. That is the one thing here that genuinely cannot be
    # prebuilt, and r3 says so in as many words.
    prebuilt = []
    for segment in plan.segments:
        prebuilt.append(
            (segment, request_builder(shot, ledger, canvas=canvas,
                                      segment_index=int(segment.index))))

    admission = _assert_beat_affordable(shot, prebuilt)

    rendered = []            # (path, drop_head, keep_frames) in play order
    #: The same tuples for the FOLEY stems, in the same play order. Empty on
    #: every silent lane, which is every lane but one; an all-or-nothing check
    #: below refuses a beat that produced some stems and not others.
    foley_rendered = []
    out_shot = dict(shot)
    attempts = []
    # The beat's PEAK, not its last segment's (2026-07-26 QA panel). Taking
    # whatever the final segment happened to report under-reports a beat whose
    # heaviest render was segment 1, which is the number the episode report
    # exists to carry.
    peak_used = 0
    terminal = None
    # PER-SEGMENT IDENTITY (operator ask, 2026-08-01). A beat that renders as
    # one clip has always been traceable -- it IS the beat, and the ledger's
    # shot_id names it. A beat that splits did not: the ledger recorded only the
    # beat's aggregate (b001, rendered_frame_count=410) while the six real
    # renders behind it landed on disk as otr_beat_zx0zqd05.mp4 -- random hex
    # naming nothing. Nothing tied a file to its beat, its position in the
    # sequence, or its neighbours, which made four things unanswerable: which
    # clip a visible defect came from, whether segment N+1 actually began on
    # segment N's terminal frame, which segment wore the fear cape, and who
    # owned any given orphan in scratch.
    #
    # Every segment now gets a stable id -- <beat>_seg00, _seg01, ... -- that
    # names its temp file AND is stamped in the ledger. Purely ADDITIVE: no
    # existing field changes meaning, so every downstream consumer that reads
    # the beat row keeps reading exactly what it read before.
    beat_key = str(beat_id or shot.get("shot_id") or "beat")
    segment_rows = []
    # The cape is painted on the still that FEEDS the next segment, so the
    # segment that wears it is the one rendered on the following pass. Carry
    # the fact forward rather than stamping the segment that merely produced
    # the frame -- otherwise the receipt names the wrong clip.
    cape_pending = False
    # ``coverage_planned`` is passed EXPLICITLY rather than inferred from the
    # segment count. Reaching this function at all means the plan owes coverage
    # work, and a one-segment plan carrying a tail trim owes exactly that --
    # but ``segment_count > 1`` would call it single-clip and route it to the
    # ping-pong path.
    with _bs.BeatSession(engine, host_caps=host_caps, profile=profile,
                         beat_id=beat_id,
                         segment_count=plan.segment_count,
                         coverage_planned=True) as session:
        for position, (segment, request) in enumerate(prebuilt):
            index = int(segment.index)
            segment_id = "%s_seg%02d" % (beat_key, position)
            init_source = "chain_terminal_frame" if (chains and terminal) \
                else "still"
            if chains and terminal:
                # THE TERMINAL TRANSACTION. A chained successor does not get a
                # minted still at all -- it begins on the real frame its
                # predecessor ended on, which is what makes it a chain rather
                # than a jump with extra steps.
                #
                # INTO ``asset_refs``, WHICH IS WHERE AN ENGINE LOOKS (corrected
                # 2026-07-26 by the QA panel). ``build_request`` puts the init
                # image at ``asset_refs["init_image"]``; every adapter and
                # ``_present_request_tokens`` read it there and NOWHERE else. An
                # earlier draft set a top-level ``request["init_image"]``, which
                # no production code reads -- so a chained successor would have
                # rendered from its ORIGINAL init image, silently, and the beat
                # would have jumped at every cut it claimed to chain across.
                # The stub in the test agreed with the bug because the test's
                # own request builder used the same wrong key.
                if isinstance(request, dict):
                    refs = request.get("asset_refs")
                    if not isinstance(refs, dict):
                        refs = {}
                        request["asset_refs"] = refs
                    refs["init_image"] = terminal
                    obs = request.get("observability")
                    if isinstance(obs, dict):
                        obs["init_source"] = "chain_terminal_frame"
                        obs["init_image"] = os.path.basename(terminal)
            clip, out_shot, seg_attempts, seg_used = render_shot(
                shot, request, oom_engines=oom_engines, oom_shot_id=oom_shot_id,
                host_caps=host_caps, profile=profile,
                segment=_bs.SegmentSlot(session, index, beat_id))
            attempts = list(attempts) + list(seg_attempts or ())
            peak_used = max(int(peak_used), int(seg_used or 0))
            path = str((clip or {}).get("path") or "")
            if not path:
                raise RenderError(
                    "shot %s segment %d rendered no clip path, so the beat "
                    "cannot be assembled. NO FALLBACK."
                    % (shot.get("shot_id"), index))
            # CHECK THE LENGTH HERE, where the message can still name the
            # segment (2026-07-26 QA panel). A short render otherwise surfaces
            # much later as an assembly count mismatch, which reads as an
            # assembler bug rather than as the engine returning fewer frames
            # than it was asked for.
            # A SEGMENT THAT CANNOT SAY WHAT IT PRODUCED IS NOT ASSEMBLABLE
            # (2026-07-27, chunk 7b C2; found by the blockers kibitz arc).
            # This read `got = int(clip.get("frame_count") or 0)` followed by
            # `if got and got != ...`, so a clip reporting 0 -- or omitting the
            # field entirely -- SKIPPED the check and was assembled anyway.
            # `CanonicalClip.frame_count` defaults to 0 (``schemas.py:233``),
            # so "absent" and "zero" are the same value here and neither is a
            # length. That is a fail-OPEN guard inside a fail-closed function:
            # the exact shape of the four swallowed fail-closed sites chunk 1a's
            # QA panel found, arriving through a different door.
            #
            # The count is the whole basis of the plan-vs-output proof, so an
            # unreadable one is a FAILED verification, not a pass -- the same
            # rule `wan_shared.ffprobe_counted_frames` already states for the
            # assembly boundary ("NO GUESS -- an unreadable count is a failed
            # verification, not a zero").
            raw_count = (clip or {}).get("frame_count")
            try:
                got = int(raw_count)
            except (TypeError, ValueError):
                got = None
            if got is None or got < 1:
                raise RenderError(
                    "shot %s segment %d returned frame_count=%r, which is not a "
                    "length. The plan asked for %d frame(s) and the segment "
                    "cannot say what it produced, so the beat cannot be proven "
                    "against its own audio. NO FALLBACK -- an unreadable count "
                    "is a failed verification, not a zero."
                    % (shot.get("shot_id"), index, raw_count,
                       int(segment.render_frames)))
            # NAME THE DIRECTION, AND DO NOT EXPLAIN A SHORTFALL THAT DID NOT
            # HAPPEN. This message read "assembling a SHORT segment would make
            # the beat drift" for both directions, and the first live firing
            # (2026-07-29, ltx_video) was a SURPLUS -- 193 frames against a
            # planned 169, because the boomerang loop-fill ignored the ask.
            # A reader chasing that log was told to look for a shortfall.
            # Both directions are still terminal and for one reason: the
            # plan's count is what this segment's audio slice was cut against
            # (segment_render_window), so a segment of ANY other length drifts
            # the beat against its own audio. The leading clause is unchanged
            # so existing receipts and assertions still match.
            planned = int(segment.render_frames)
            if got != planned:
                delta = ("a surplus of %d" % (got - planned)) if got > planned \
                    else ("short by %d" % (planned - got))
                raise RenderError(
                    "shot %s segment %d rendered %d frame(s) but its plan asked "
                    "for %d (%s). NO FALLBACK -- the plan's count is what this "
                    "segment's audio slice was cut against, so assembling a "
                    "segment of any other length makes the beat drift against "
                    "its own audio."
                    % (shot.get("shot_id"), index, got, planned, delta))
            visible = (int(segment.render_frames)
                       - int(segment.drop_head)
                       - int(segment.trim_tail))
            rendered.append((path, int(segment.drop_head), visible))
            # THE FOLEY STEM RIDES THE SAME TUPLE AS THE VIDEO IT BELONGS TO
            # (the LTX 2.5 foley bed, 2026-08-26). Same path, same drop_head,
            # same visible count -- because it IS the same cut, applied in
            # sample space instead of frame space. Collected here rather than
            # derived later so the two lists cannot fall out of order: a stem
            # matched to the wrong segment is audio playing under a picture it
            # was not generated for, which is the one thing this whole feature
            # exists to avoid.
            #
            # Absent on every silent lane, which is the normal case.
            _stem = str((clip or {}).get("foley_path") or "")
            if _stem:
                foley_rendered.append(
                    (_stem, int(segment.drop_head), visible))
            segment_row = {
                "segment_id": segment_id,
                "beat_id": beat_key,
                "segment_index": position,
                "segment_count": int(plan.segment_count),
                "render_frames": int(segment.render_frames),
                "drop_head": int(segment.drop_head),
                "trim_tail": int(segment.trim_tail),
                "visible_frames": visible,
                # THE HONESTY RECEIPTS, PER SEGMENT (2026-08-06). Carried
                # UNCONVERTED -- whatever the adapter said, including nothing --
                # because this row is EVIDENCE and evidence that repairs itself
                # on the way past is not evidence. The aggregation below is what
                # judges them; a segment that cannot answer makes the beat
                # unknown, which is the honest outcome and not this loop's call.
                "native_frame_count": (clip or {}).get("native_frame_count"),
                "extension_mode": (clip or {}).get("extension_mode"),
                "init_source": init_source,
                "fear_cape": bool(cape_pending),
                "path": os.path.basename(path),
            }
            segment_rows.append(segment_row)
            cape_pending = False
            if chains and position + 1 < len(plan.segments):
                terminal = _wb2.extract_terminal_frame(
                    path, _tmp_path("otr_terminal_%s_" % segment_id, ".png"))
                # THE FEAR CAPE (operator's name, 2026-08-01). On a beat long enough
                # to need several clips, the still handed to the FINAL segment is
                # inverted, so that segment is GENERATED from an inverted frame
                # rather than having a filter painted over it. Defaulted ON;
                # OTR_FEAR_CAPE=0 turns it off.
                #
                # `position + 2 == len(plan.segments)` means "this extraction
                # feeds the last segment". It knowingly breaks the seamless cut
                # at that one seam -- that IS the effect.
                if (_wb2.fear_cape_enabled()
                        and len(plan.segments) >= _wb2.fear_cape_min_segments()
                        and position + 2 == len(plan.segments)):
                    terminal = _wb2.negate_image(
                        terminal, _tmp_path("otr_fearcape_%s_" % segment_id, ".png"))
                    cape_pending = True
                    _LOG.info(
                        "[OTR video] FEAR CAPE: beat %s has %d segments, so "
                        "the final segment begins on an INVERTED still",
                        shot.get("shot_id"), len(plan.segments))

    assembled = _ws.assemble_beat_segments(
        rendered, _tmp_path("otr_beat_%s_" % beat_key, ".mp4"),
        expect_frames=int(plan.target_visible_frames),
        expect_fps=int((clip or {}).get("fps") or 25))
    # THE FOLEY SIBLING OF THE ASSEMBLY ABOVE, and it runs on SINGLE-segment
    # beats too. A one-segment plan still owes a tail trim whenever its length
    # was rounded up to a legal rung, and routing only multi-segment beats
    # through the cutter is how that surplus survives on exactly the beats
    # nobody thinks to check. It is also why the engine emits an UNCUT stem:
    # `drop_head` / `keep_frames` belong to this stage, in sample space, once.
    #
    # ALL OR NOTHING. A beat whose segments disagree about whether they carry
    # foley is a producer fault, and "mix the rows that have one" would deliver
    # a bed that cuts in and out mid-beat while every log stayed green.
    foley_receipts = None
    if foley_rendered:
        if len(foley_rendered) != len(rendered):
            raise RenderError(
                "beat %s assembled %d video segment(s) but only %d foley "
                "stem(s). NO PARTIAL BED -- a bed that appears under some "
                "segments of one beat and not others is worse than none, and "
                "every log would still read green"
                % (beat_key, len(rendered), len(foley_rendered)))
        foley_receipts = _foley.assemble_beat_foley_segments(
            foley_rendered,
            _foley.durable_foley_dir() / ("beat_%s_foley.wav" % beat_key),
            expect_frames=int(plan.target_visible_frames),
            fps=int((clip or {}).get("fps") or 25))

    beat_clip = dict(clip or {})
    beat_clip["path"] = assembled
    beat_clip["frame_count"] = int(plan.target_visible_frames)
    # Say so on the clip itself: a beat made of four renders is not the same
    # artifact as a beat made of one, and the receipt should not have to be
    # reverse-engineered from a log.
    beat_clip["segment_count"] = int(plan.segment_count)
    beat_clip["join_mode"] = str(plan.join_mode)
    # The per-segment receipt. One row per REAL render, in render order, each
    # naming itself, where its first frame came from, and how many of its frames
    # survive the seam. This is what makes "a fresh clip for every second of
    # audio" checkable after the fact instead of merely intended: sum the
    # visible_frames and it must equal the beat's target.
    beat_clip["segments"] = segment_rows
    # THE FOLEY RECEIPTS ARE BEAT-SCOPE AND MUST BE OVERWRITTEN, never
    # inherited. `beat_clip` starts life as a copy of the LAST segment, so
    # leaving these alone would hand the manifest that segment's 97-frame stem
    # as if it were the whole beat's -- the identical trap the frame-accounting
    # block below this one was written to close, one field group over.
    if foley_receipts is not None:
        beat_clip.update(foley_receipts)
        # THE BEAT CLIP CARRIES ITS OWN AUDIO (operator ruling 2026-08-29: a
        # joint-AV clip he cannot HEAR is a failed render). The assembled beat
        # stem is muxed into the assembled beat mp4 as an AAC preview track --
        # the WAV stays the authoritative mix source, and OTRSilentComposite
        # re-encodes every row with -an, so this track can never reach the
        # episode master or be mixed twice. Fails LOUD: a foley beat whose
        # clip stayed silent is exactly the false green the lane must not
        # produce. has_audio flips to True HERE, at beat scope, because it is
        # true here and only here -- the per-segment rows stay silent and
        # their contract is unchanged.
        _foley.mux_native_audio_into_beat_clip(
            assembled, foley_receipts["foley_path"],
            fps=int((clip or {}).get("fps") or 25))
        beat_clip["has_audio"] = True
    else:
        for _fkey in _foley.FOLEY_RECEIPT_KEYS:
            beat_clip.pop(_fkey, None)
    # THE BEAT-SCOPE FRAME ACCOUNTING (2026-08-06).
    #
    # WRITTEN UNCONDITIONALLY, INCLUDING None. ``beat_clip`` starts life as
    # ``dict(clip or {})`` -- a copy of the LAST segment -- so any beat-scope
    # field this block does not assign silently keeps that segment's value.
    # That is not a hypothetical: it is the defect being repaired here. The
    # original code overwrote four keys and inherited two, so an assembled beat
    # of three real 81-frame renders claimed the last segment's 81 native frames
    # against its own 241, and the grader called every honest multi-segment beat
    # a pad. A conditional write here would rebuild that trap one field over.
    #
    # THE TWO COUNTS ARE DISTINCT ON PURPOSE (Bug Bible 12.69 /
    # PRODUCTION_SPRINT_LESSONS lesson 33). ``native_frame_count`` is the WORK:
    # every native frame this beat rendered, 243 for a 3x81 chain.
    # ``delivered_native_frame_count`` is the OUTPUT: the 241 that survived the
    # seams. Neither is wrong; they answer different questions, and collapsing
    # them into one field is what made an intentional seam drop look like an
    # engine underrun.
    #
    # FAIL-SOFT HERE, FAIL-CLOSED AT THE GRADER. The same arithmetic runs on
    # both sides, but a receipt problem must never destroy video: this path owns
    # a rendered beat and raising here would throw it away before a manifest
    # exists. So an unusable receipt becomes None -- and None is a FINDING when
    # the grader reads it. The render keeps its footage; the episode still
    # cannot claim coverage it did not prove.
    beat_clip.update(_acceptance.beat_frame_accounting(segment_rows))
    # Whether this beat's lengths were actually CHECKED against VRAM before any
    # weights loaded, and against a row measured for THIS engine. Stamped even
    # when unenforced, because "the guard did not run here" is the fact a reader
    # needs; a receipt that records only successful checks reads identically
    # whether the guard covered the beat or skipped it.
    # THE BEAT'S PEAK, ON THE CLIP THE MANIFEST READS (2026-08-06).
    #
    # ``peak_used`` has tracked the MAX across segments since the 2026-07-26 QA
    # panel, for the reason stated where it is initialised: taking whatever the
    # final segment reported under-reports a beat whose heaviest render was
    # segment 1. That fix reached the RETURN VALUE and stopped there --
    # ``beat_clip`` is ``dict(clip or {})``, a copy of the LAST segment, so the
    # durable receipt kept inheriting the last segment's number while
    # ``build_clip_manifest`` read it off the clip and published it.
    #
    # Same trap as the extension receipts this function was just repaired for,
    # one field over: a beat-scope value left unassigned silently becomes a
    # segment-scope one. Written UNCONDITIONALLY for that reason. ``None`` when
    # nothing was measured, which is the convention every consumer already uses
    # -- never 0, which would read as "measured, and it was free".
    beat_clip["vram_peak_mb"] = int(peak_used) if peak_used else None
    beat_clip["vram_admission"] = admission
    _LOG.warning(
        "[OTR video] BEAT %s assembled from %d %s segment(s) -> %d frame(s) "
        "at %s", beat_id or shot.get("shot_id"), plan.segment_count,
        plan.join_mode, plan.target_visible_frames,
        os.path.basename(assembled))
    return beat_clip, out_shot, attempts, peak_used


def build_episode_render_policy(section):
    """The v2 render policy handed to EVERY adapter's ``assert_usable`` /
    ``prepare`` for this episode, read from the ShotLock-stamped ledger
    ``video`` section. Pure; CPU-tested.

    S4 platform-portability (2026-07-10) put the device/dtype truth here so an
    adapter sees exactly what the directors emitted. WAN 8GB launch contract
    (2026-07-24) adds ``max_render_frames``, the tier's absolute render-length
    ceiling: a production episode leg is submitted to an already-booted server,
    so a profile's ``launch.env`` can never reach the engine -- the ceiling has
    to ride the ledger. 0 = unpinned, which is every shipped tier except the
    8GB Wan profile, so no other lane's behaviour moves."""
    section = section or {}
    return {
        "policy_version": 2,
        "device_policy": str(section.get("device_policy") or "cuda"),
        "dtype_policy": str(section.get("dtype_policy") or "fp8_ok"),
        "max_render_frames": max(0, int(section.get("max_render_frames") or 0)),
    }


def run_episode(ledger, *, oom_shot_id=None,
                oom_engines=frozenset(), assets=None, frame_count=25,
                canvas=None, request_builder=None):
    """Drive one episode end-to-end on REAL engines (deep-copies the ledger; the
    frozen ``audio`` section is never touched). Returns
    ``{ledger, clips, trace, vram_peak_mb}``.

    ``request_builder`` (default None) keeps the soak/global-assets path
    (``build_request`` with the shared ``assets`` + ``frame_count``). The REAL
    episode path passes ``build_request_from_shot``: ``request_builder(shot,
    ledger, canvas=canvas)`` is called per shot for a per-beat portrait + audio
    + prompt request (see :func:`run_real_episode`)."""
    ledger = copy.deepcopy(ledger)
    _log_if_legacy_render_plan_present(ledger)
    section = ledger["video"]
    # S4 platform-portability (2026-07-10): build REAL host facts ONCE and
    # carry the ledger-stamped v2 device/dtype policy to every adapter's
    # assert_usable/prepare (was hardcoded {} at every call).
    from .._otr_shared.host_caps import build_host_caps
    _episode_host_caps = build_host_caps()
    _episode_profile = build_episode_render_policy(section)
    clips, new_shots, trace = {}, [], []
    # S-C C1 (audio_motion_profile): per-shot resolved conditioning-WAV rows
    # (the per-line voice clip OR the read-only master slice) for the post-render
    # profiling pass. Additive + read-only -- collecting a path NEVER affects the
    # render; the frozen master is only ever read.
    amp_rows = []
    vram_peak = 0
    # Stills-first VRAM discipline (operator 2026-06-12, "evict flux when all
    # stills are done"): every portrait + scene still is already minted to disk
    # by the image phase -- build_request_from_shot only RESOLVES ledger paths,
    # it never regenerates Flux. So the Flux/LLM still-phase models are idle-but-
    # resident here and pin ~4.7GB+ that machine-wide NVML counts against the V-3
    # render-phase ceiling (the 15-16GB pin; CS-2). Evict them 100% BEFORE the
    # first video beat so HuMo/LTX load into a clean GPU and the render-phase peak
    # reflects ONE resident heavy engine. LOUD by contract; a no-op off the box.
    if _section_has_local_video_engine(section):
        try:
            # PHASE-BOUNDARY residue free (stills -> video). A ComfyUI-only
            # reclaim is NOT enough: the detach loop detached 0 AND free_memory
            # freed 0MB live (measured 2026-06-15, torch-free 6939MB -> 6939MB),
            # because the ~7GB resident residue is the OUT-OF-BAND transformers
            # caches (the writer LLM, Bark, ...) loaded through OTR's OWN
            # loaders -- ComfyUI's model_management cannot see them, so
            # free_memory/empty_cache cannot touch them. The canonical Lever-1
            # freer releases the writer LLM + Bark FIRST, THEN DETACHES any
            # ComfyUI FLUX patcher (surgical; NO unload_all_models per V-4/V-5),
            # then flushes the allocator, so the first local video engine loads
            # into a clean GPU instead of OOMing at the ksampler. Best-effort
            # (never raises); MEASURED telemetry.
            from .._otr_vram_levers import free_otr_pipeline_residue as _free_residue
            _rep = _free_residue(reason="pre-render: all stills minted")
            _LOG.warning(
                "[OTR video] pre-render residue free: ran=%s failed=%s free_gb_after=%s",
                _rep.get("steps_run"), _rep.get("steps_failed"),
                _rep.get("free_gb_after"))
        except Exception as _exc:  # noqa: BLE001 -- reclaim is best-effort, never fatal
            _LOG.warning("[OTR video] pre-render VRAM reclaim skipped: %s", _exc)
    else:
        _LOG.info("[OTR video] pre-render VRAM reclaim skipped: all selected "
                  "video engines are Partner/cloud")
    _last_engine = None
    # EPISODE-SCOPED ENGINE RESIDENCY (2026-08-20). Some adapters can reuse a
    # heavy handle across the beats of ONE episode -- LTX 2.5 re-read its
    # 8.86 GiB text encoder on every shot, 13 loads for 13 renders on a live
    # leg -- but they cannot own that decision themselves: the registry builds
    # each adapter ONCE at import and hands back the same instance forever
    # (``engine_registry_base.py:148-155``), so a cache kept on the adapter
    # would never end. Every engine-level hook runs too often to be the
    # release point (``teardown``/``unload`` are per BEAT), so the boundary is
    # HERE, where an episode actually begins and ends.
    #
    # DUCK-TYPED, like ``session_identity`` and the optional ``frame_contract``.
    # NOT like ``prepare``, which IS declared on the ``VideoEngine`` protocol
    # (``registry.py:109``) -- an earlier version of this comment cited it as
    # the precedent and was wrong. An adapter that declares nothing here is
    # untouched and behaves exactly as it always has.
    _scoped_engines = {}

    def _begin_engine_scope(engine_id):
        """Open an adapter's episode scope once. Never raises.

        BOTH HALVES ARE REQUIRED BEFORE EITHER IS CALLED. An adapter with a
        ``begin`` and no ``end`` would open a scope nothing can close, which is
        the exact leak this whole mechanism exists to prevent -- so the pair is
        checked up front rather than discovered at cleanup time.

        The engine is recorded BEFORE ``begin()`` runs, so a hook that raises
        part-way through still has an owner registered for the ``finally`` to
        close.
        """
        if not engine_id or engine_id in _scoped_engines:
            return
        try:
            if not _vreg.is_registered(engine_id):
                return
            eng = _vreg.get_engine(engine_id)
            begin = getattr(eng, "begin_encoder_scope", None)
            end = getattr(eng, "end_encoder_scope", None)
            if not (callable(begin) and callable(end)):
                return
            # THE TOKEN IS RETAINED so this run closes ITS OWN ownership and
            # not whatever scope happens to be open later. Adapters that
            # return nothing still work -- the token is simply None.
            _scoped_engines[engine_id] = (eng, begin())
        except Exception as _exc:  # noqa: BLE001 -- residency is best-effort
            _LOG.warning("[OTR video] episode scope not opened for %r: %s",
                         engine_id, _exc)

    def _end_engine_scope(engine_id):
        """Close ONE adapter's episode scope. Never raises.

        OWNERSHIP IS POPPED FIRST, so the `finally` sweep cannot double-close
        and cannot spin retrying a hook that keeps failing.

        A FAILED CLOSE IS LOGGED, NOT ESCALATED TO A GLOBAL RELEASE. An earlier
        version reached for ``release_encoder_cache()`` here as a belt-and-
        braces cure, and the r4 panel showed the cure was worse: that call
        drops the cache for EVERY owner, so one run's failing close would
        destroy a concurrent run's live cache. Losing this run's share of a
        cache is a slow render; taking someone else's is a broken one.
        """
        entry = _scoped_engines.pop(engine_id, None)
        if entry is None:
            return
        eng, token = entry
        try:
            try:
                eng.end_encoder_scope(token)
            except TypeError:      # adapter predates the token argument
                eng.end_encoder_scope()
        except Exception as _exc:  # noqa: BLE001 -- runs from cleanup paths
            _LOG.warning("[OTR video] episode scope not closed for %r: %s -- "
                         "this run's claim is leaked rather than forcing a "
                         "release that would evict other owners too",
                         engine_id, _exc)

    try:
        # Beats whose required still the image model REFUSED. Read once, not
        # per beat: the receipt is frozen by the time the render starts.
        _gap_beats = sanctioned_gap_beat_ids(ledger)
        for shot in section["shots"]:
            # A SANCTIONED GAP IS SKIPPED WHOLE, AND SKIPPED HERE (2026-08-28).
            #
            # HERE is load-bearing. The obvious place looks like the request
            # builder, but ``build_request_from_shot`` returns a REQUEST, not a
            # shot, and on the cheap families a missing still does not even
            # raise there -- it logs and sets ``init_image=""``, and the raise
            # lands later in ``cheap_families.render_clip``. Skipping at the
            # top of the loop is the only point that precedes BOTH the engine
            # scope and every raise site, so a beat nobody will render also
            # never loads an engine's weights.
            #
            # The ORIGINAL ShotLock shot is kept, not a synthesized stand-in:
            # the beat is a real fact about the episode and it keeps its id,
            # role, family, position and frame budget. What it does not get is
            # a clip -- ``build_clip_manifest`` derives existence from the clip
            # file, so the manifest row comes out absent on its own, and the
            # composite's existing floor-fill covers the hole. Nothing is
            # substituted and nothing is silent: the gap row in the receipt is
            # the record, and the warning below is the log.
            if str(shot.get("beat_id") or "") in _gap_beats:
                _LOG.warning(
                    "[OTR.render_driver] SANCTIONED GAP shot %s (beat %s): not "
                    "rendered -- the image model refused its required still. "
                    "The beat keeps its place in the timeline and is floored.",
                    shot.get("shot_id"), shot.get("beat_id"))
                new_shots.append(shot)
                continue
            # CS-3 inter-beat reclaim (2026-06-15): before a beat that loads a
            # DIFFERENT engine than the one the prior beat left resident, drain the
            # prior engine's residue and FLUSH the allocator, so two heavy engines
            # never co-reside on the 16GB card -- the cause of the 29s/it edge
            # page-thrash (ltx 12.5GB + humo 7GB -> 19.5GB). The per-beat teardown
            # already DETACHES + waits, but it does not return the freed-but-cached
            # blocks to the driver; the surgical Lever-1 freer's cuda flush does.
            # SELECTIVE: same-engine consecutive beats SKIP this (no reload churn --
            # the resident-stack reuse, e.g. humo x3, is preserved). Best-effort,
            # never fatal; no unload_all_models (V-4/V-5); MEASURED telemetry.
            _this_engine = str(shot.get("engine_id") or "")
            # CLOSE THE OUTGOING ENGINE'S SCOPE ON *ANY* ENGINE CHANGE, and on
            # its own condition -- NOT inside the reclaim gate below.
            #
            # THIS WAS A REAL BUG AND THE PANEL CAUGHT IT.
            # ``_should_reclaim_between_engines`` ends in
            # ``and not _is_cloud_video_engine(this)`` (:1885-1888), because a
            # VRAM reclaim only matters before a LOCAL engine loads. Nesting
            # the scope release inside it meant a local -> CLOUD hand-off never
            # released the local engine's 8.86 GiB: the predicate is about
            # whether to drain VRAM, and ownership is a different question that
            # happens to have shared a line.
            #
            # Ordering still matters: the release runs BEFORE the reclaim, so
            # the reclaim is not draining around the largest thing in the room.
            if _last_engine and _this_engine and _this_engine != _last_engine:
                _end_engine_scope(_last_engine)
            if _should_reclaim_between_engines(_last_engine, _this_engine):
                try:
                    from .._otr_vram_levers import (
                        free_otr_pipeline_residue as _free_residue)
                    _ir = _free_residue(
                        reason="inter-beat %s->%s" % (_last_engine, _this_engine))
                    _LOG.warning(
                        "[OTR video] inter-beat reclaim %s->%s: free_gb_after=%s",
                        _last_engine, _this_engine, _ir.get("free_gb_after"))
                except Exception as _exc:  # noqa: BLE001 -- best-effort, never fatal
                    _LOG.warning(
                        "[OTR video] inter-beat reclaim skipped: %s", _exc)
            # OPEN THIS ENGINE'S EPISODE SCOPE. After the reclaim above, so a
            # scope is never opened into memory that is about to be drained,
            # and idempotent, so the 2nd..Nth beat on the same engine is free.
            _begin_engine_scope(_this_engine)
            if request_builder is not None:
                request = request_builder(shot, ledger, canvas=canvas)
            else:
                request = build_request(shot, assets, frame_count, canvas)
            # S-C C1: record this shot's resolved conditioning WAV + id/timing so the
            # post-render pass can stamp a per-beat audio_motion_profile. Best-effort;
            # a collection hiccup must never perturb the render.
            try:
                _amp_ref = (request.get("audio_ref") or {}) if isinstance(request, dict) else {}
                _amp_wav = _amp_ref.get("path") if isinstance(_amp_ref, dict) else None
                if _amp_wav:
                    amp_rows.append({
                        "id": str(shot.get("shot_id") or ""),
                        "start_s": float(shot.get("start_s") or 0.0),
                        "dur_s": float(shot.get("dur_s") or 0.0),
                        "_wav": str(_amp_wav),
                    })
            except Exception:  # noqa: BLE001 -- profiling collection is never fatal
                pass
            # ONE BEAT, however many renders it takes (2026-07-26, chunks 6c/6d).
            # A shot with no coverage plan or a single-clip one goes straight to
            # render_shot with the request already built above -- unchanged. A
            # multi-segment plan opens ONE beat session, renders each segment from
            # its own request, chains terminal frames inside the loop, and assembles
            # the result before it comes back.
            clip, out_shot, attempts, used = render_beat_coverage(
                shot, ledger, request=request, request_builder=request_builder,
                canvas=canvas, oom_engines=oom_engines, oom_shot_id=oom_shot_id,
                host_caps=_episode_host_caps, profile=_episode_profile)
            # NO FALLBACKS (2026-07-02): render_shot either returns a clip or
            # raises RenderError; there are no runtime fallback decisions and no
            # AS-2 family-change group prune (the family can never change here).
            clips[out_shot["shot_id"]] = clip
            new_shots.append(out_shot)
            row = {"shot_id": out_shot["shot_id"], "attempts": attempts,
                   "final_engine": out_shot["engine_id"]}
            # Round 5 F2: prompt observability rides the trace (durable in the
            # node-92 /history report) -- the diversity gate + the operator's
            # "did the prompts actually differ" check read these. The stamps live
            # on the request's schema-real ``observability`` dict (W7-pre builder
            # migration; the legacy top-level ``_<key>`` spelling is still read
            # for hand-built requests).
            obs = (request.get("observability") or {}) if isinstance(request, dict) else {}
            for key in ("prompt_source", "prompt_subsource", "prompt_sha8",
                        "prompt_chars", "init_source", "init_image",
                        "i2v_still_missing", "video_seed",
                        "visual_style", "prompt_field_source",
                        # The banana receipt (docs/2026-08-06-BUILD-SPEC-banana-route.md)
                        # -- without these the keys never reach the node-92 report.
                        "banana_route", "banana_table_version",
                        "banana_substitutions", "banana_sha256_before",
                        "banana_sha256_after", "banana_varieties",
                        # GHOST SIGNAL PROMPT RECEIPTS (2026-08-22). These live
                        # on the REQUEST's observability, which is why they
                        # belong in this loop and the cadence keys below do not.
                        "prompt_version", "negative_sha8", "negative_chars",
                        "prompt_slots", "subject_sigil_sha8",
                        # GHOST PROMPT v2 RECEIPTS (2026-08-22). A key that is
                        # stamped but not listed here never reaches the node-92
                        # /history report, which is the only place the operator
                        # can read what a published episode actually asked for.
                        "author_version", "ghost_schema_version",
                        "ghost_source", "ghost_fallback_reason",
                        "ghost_model_id", "ghost_mode", "ghost_motif_cue",
                        "ghost_request_sha8", "ghost_output_sha8",
                        "ghost_drawable_beat", "ghost_drawable_beat_sha8",
                        "positive_clip_tokens", "positive_clip_windows",
                        "negative_clip_tokens", "negative_clip_windows",
                        "clip_window_max", "clip_counter",
                        # JOINT-AV RECEIPTS. Stamped on the request and declared
                        # in the schema, but omitted from this allowlist -- so
                        # the published /history evidence never carried what a
                        # joint-AV beat actually asked for or was heard to say.
                        # `joint_av_identity_leak` is only present when a leak
                        # was detected; absent stays absent, never invented.
                        "joint_av_prompt", "joint_av_sounds",
                        "joint_av_identity_leak"):
                if key in obs:
                    row[key] = obs[key]
                elif isinstance(request, dict) and ("_" + key) in request:
                    row[key] = request["_" + key]
            # THE CADENCE / DELIVERY KEYS COME FROM THE CLIP, NOT THE REQUEST.
            # Appending them to the loop above would have been the natural-
            # looking mistake and they would ALL have been silently absent: they
            # are stamped by the adapter on what it RETURNED, never on what was
            # asked for -- which is the point, since a receipt sourced from
            # request intent could describe a render that did not happen.
            # Without this, node 92's /history report drops them entirely.
            if isinstance(clip, dict):
                for key in _CADENCE_DELIVERY_RECEIPT_KEYS:
                    if key in clip:
                        row[key] = clip[key]
            trace.append(row)
            if used:
                vram_peak = max(vram_peak, int(used))
            # CS-3: remember what ACTUALLY rendered (post-fallback final_engine) so
            # the next beat reclaims only when it crosses to a different engine.
            _last_engine = str(out_shot.get("engine_id") or "")
    finally:
        # THE SCOPES CLOSE EVEN WHEN THE EPISODE DOES NOT FINISH. A raise
        # anywhere in the loop -- an OOM, a refused beat, a bad ledger row --
        # must still release the episode's heavy handles, or a failed render
        # leaves 8.86 GiB pinned for the life of the server and the NEXT
        # episode pays for this one's crash.
        for _scoped_id in list(_scoped_engines):
            _end_engine_scope(_scoped_id)
    section["shots"] = new_shots
    ledger["video"] = section
    return {"ledger": ledger, "clips": clips, "trace": trace,
            "vram_peak_mb": vram_peak, "audio_motion_rows": amp_rows}


def _log_if_legacy_render_plan_present(ledger):
    """Do not let story-QA render_plan metadata drop real episode shots.

    ``meta.render_plan`` was introduced for the retired/separate HuMo batch as
    a cost/attention filter. The generic episode renderer is now the visible
    delivery path, so every ShotLock row must render; otherwise a voiced line
    can fall through to the procgen floor while its audio/caption still plays.
    """
    meta = (ledger or {}).get("meta") or {}
    plan = meta.get("render_plan") or {}
    if not isinstance(plan, dict):
        return ledger
    if not plan:
        return ledger
    section = (ledger or {}).get("video") or {}
    shots = section.get("shots") or []
    if not isinstance(shots, list) or not shots:
        return ledger

    def _safe_count(value):
        if isinstance(value, (list, tuple, set)):
            return len(value)
        return 0

    line_ids_n = _safe_count(plan.get("line_ids"))
    excluded_n = _safe_count(plan.get("excluded_flat_lines"))
    try:
        applied_max_n = int(plan.get("applied_max_n") or 0)
    except (TypeError, ValueError):
        applied_max_n = 0
    blocked = bool(plan.get("blocked"))
    mode = str(plan.get("selection_mode") or "")
    active_legacy_filter = (
        blocked or excluded_n > 0 or applied_max_n > 0
        or mode not in ("", "all")
    )
    log_fn = _LOG.warning if active_legacy_filter else _LOG.info
    log_fn(
        "[OTR video] render_plan present but ignored by episode renderer "
        "(mode=%s line_ids=%d excluded=%d blocked=%s applied_max_n=%d); "
        "rendering all %d ShotLock shot(s)",
        mode, line_ids_n, excluded_n, blocked, applied_max_n, len(shots))
    return ledger


def run_real_episode(ledger, *, canvas=None,
                     master_audio_path=""):
    """Drive one REAL episode from a ShotLock-planned ledger: per-shot requests
    (character portrait + per-beat voice audio + the M4 prompt) via
    :func:`build_request_from_shot`, NO forced OOM, NO FALLBACKS (a failed
    engine raises LOUD), real per-shot assets. Returns the :func:`run_episode` result; the
    frozen audio section is untouched. The thin ``mode="episode"`` render node
    calls this inside the ComfyUI executor thread.

    ``master_audio_path``: path to the FROZEN master mix (MP4 or WAV).
    When set, beats whose ledger line has no ``*_wav_path`` get their
    ``audio_ref`` filled by slicing ``[start_s, start_s+dur_s]`` from the
    master (read-only ffmpeg; master NEVER mutated).  Passed via
    ``functools.partial`` into ``build_request_from_shot`` so the call
    signature of :func:`run_episode` stays unchanged.

    Rename-proofing: the episode dir is renamed ``pending_*`` -> final slug
    mid-run, so the ``master_audio_path`` captured upstream can be stale (its
    pending dir gone) by the time this renders -- which silently skipped the
    per-beat slice and starved HuMo (audio_ref=''). Re-resolve to the SAME
    master file under the renamed dir using the same contract the terminal mux
    uses. Read-only; the audio bytes are never touched."""
    # Brief disposition, ONCE per episode run (gap-audit G4 restore): the
    # canonical [story_brief:<id>] line proving the brief reached the scene
    # composer. Fail-soft -- never blocks the render.
    try:
        try:
            from .._otr_story_brief_helpers import (  # type: ignore
                log_story_brief_disposition)
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_story_brief_helpers import (  # type: ignore
                log_story_brief_disposition)
        log_story_brief_disposition((ledger or {}).get("meta") or {},
                                    "ltx_scene_open", _LOG)
    except Exception:  # noqa: BLE001
        pass
    if master_audio_path:
        try:
            from ..otr_master_audio_mux import _reresolve_master_audio
            master_audio_path = _reresolve_master_audio(str(master_audio_path))
        except Exception:  # noqa: BLE001 - never block the render on re-resolve
            pass
    # THE ROUTE LOCK (2026-07-25): resolve force-map + radio-host redirect in ONE
    # pass. Idempotent -- OTR_VideoRenderBatch already called this BEFORE the
    # still-spine check, which is the point; this call keeps a direct
    # run_real_episode caller (tests, harnesses) correct on its own.
    ledger = resolve_final_shot_engines(ledger)
    rb = functools.partial(build_request_from_shot,
                           master_audio_path=master_audio_path)
    return run_episode(ledger, request_builder=rb, canvas=canvas)


def parse_engine_override(spec: str) -> dict:
    """Parse ``OTR_FORCE_ENGINE_MAP`` (pure). Grammar: comma-separated
    ``role=engine`` pairs; the role ``*`` means EVERY shot regardless of role.
    Examples: ``*=ltx_video`` (the all-LTX episode);
    ``character_video=wan_i2v,announcer_visual=humo``.
    A PUBLIC menu id (``wan_8gb``) or a LEGACY id resolves to its internal engine id
    before the registry check, so the map is stored as internal ids (video-tiers
    2026-07-20, boundary 6). Unknown engines raise at parse time (fail-closed)."""
    from .._otr_shared.public_engines import (
        check_retired_engine, resolve_engine_id)
    out = {}
    for pair in (spec or "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(
                "OTR_FORCE_ENGINE_MAP entry %r is not role=engine" % pair)
        role, engine = (s.strip() for s in pair.split("=", 1))
        engine = resolve_engine_id(engine)   # public/legacy -> internal
        check_retired_engine(engine)         # NAMED refusal, after resolution
        if engine not in ENGINE_FAMILY and not _vreg.is_registered(engine):
            raise ValueError(
                "OTR_FORCE_ENGINE_MAP names unknown engine %r" % engine)
        out[role] = engine
    return out


def frozen_route_from_ledger(ledger):
    """The frozen role -> effective-engine map ShotLock stamped, or ``{}``.

    ``{}`` means this ledger never passed through the OTR_VideoDirector /
    OTR_ShotLock freeze: a legacy fixture, or one of the two shipped HTTP entry
    points (``/otr/video_render_single``, ``/otr/video_render_soak``) that build
    a ledger by hand with no OTR_VideoDirector upstream. Those keep the historic
    mutating behaviour -- see :func:`resolve_final_shot_engines`.
    """
    section = (ledger or {}).get("video") or {}
    frozen = section.get("roles_effective")
    return dict(frozen) if isinstance(frozen, dict) and frozen else {}


def assert_frozen_route(ledger):
    """VERIFY the frozen route instead of re-deriving it. Returns True if a
    frozen route was present (and therefore verified), False if there is none.

    Never mutates. Raises :class:`RenderError` on ANY disagreement:

    * the routing ENVIRONMENT moved since the freeze, so the frozen map no
      longer describes what will render;
    * a shot's engine/family is not what the freeze said it would be;
    * a shot disagrees with its own execution group.

    WHY AN ASSERTION AND NOT A REPAIR (2026-07-25, chunk 1c). Repairing here is
    what the old code did, and it is why the defect survived so long: the
    render phase silently corrected a route the image phase had already planned
    stills for, so the two phases disagreed and only the pixels showed it. If
    the plan and the environment disagree at render time, the stills are
    already wrong and no mutation of the shot rows can un-mint them. Fail loud,
    before GPU time.
    """
    frozen = frozen_route_from_ledger(ledger)
    if not frozen:
        return False

    section = (ledger or {}).get("video") or {}
    try:
        from .._otr_shared import route_freeze as _rf  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_shared import route_freeze as _rf  # type: ignore

    stamped_env = section.get("routing_env_snapshot")
    if isinstance(stamped_env, dict) and stamped_env:
        live_env = _rf.routing_env_snapshot()
        if not _rf.snapshots_agree(stamped_env, live_env):
            raise RenderError(
                "ROUTE EQUALITY: the routing environment changed after the "
                "plan was frozen (frozen %r, live %r). Every still and prompt "
                "in this episode was planned for the frozen route, so "
                "rendering the live one would render engines the image phase "
                "never minted assets for. Re-run from OTR_VideoDirector. "
                "NO FALLBACK." % (stamped_env, live_env))

    groups_by_id = {}
    for grp in section.get("execution_groups") or ():
        if isinstance(grp, dict):
            groups_by_id[str(grp.get("group_id") or "")] = str(
                grp.get("engine_id") or "")

    for shot in section.get("shots") or ():
        if not isinstance(shot, dict):
            continue
        role = str(shot.get("role") or _role_of_shot(shot) or "")
        expected = str(frozen.get(role) or "")
        actual = str(shot.get("engine_id") or "")
        if not expected:
            # A shot whose role the freeze never covered (2026-07-25 QA). The
            # docstring promises to catch ANY disagreement, and silently
            # skipping an unmapped role is the one hole that promise had: the
            # shot would render on whatever engine its row happens to carry,
            # unverified. Unreachable through OTR_VideoDirector, which fails
            # LOUD on an empty role before the freeze is taken -- so reaching
            # here means a hand-built or tampered ledger, and guessing is
            # exactly the wrong response to that.
            raise RenderError(
                "ROUTE EQUALITY: shot %s carries role %r, which the frozen "
                "route does not cover (frozen roles: %s). A shot whose route "
                "was never frozen cannot be verified, so it must not render. "
                "NO FALLBACK." % (shot.get("shot_id"), role,
                                  ", ".join(sorted(frozen)) or "<none>"))
        if actual != expected:
            raise RenderError(
                "ROUTE EQUALITY: shot %s (role=%s) carries engine %r but the "
                "frozen route says %r. The plan and the shot rows disagree, so "
                "the stills were minted for a different engine than the one "
                "about to render. NO FALLBACK."
                % (shot.get("shot_id"), role, actual, expected))
        grp_engine = groups_by_id.get(str(shot.get("group_id") or ""))
        if grp_engine and actual and grp_engine != actual:
            raise RenderError(
                "ROUTE EQUALITY: shot %s carries engine %r but its execution "
                "group %r carries %r. A shot must never diverge from its own "
                "group. NO FALLBACK."
                % (shot.get("shot_id"), actual, shot.get("group_id"), grp_engine))

    # DEBUG, not INFO: this fires on every healthy episode and is verified
    # twice per render (otr_video_render_batch calls the check, then
    # run_real_episode calls it again as defence in depth). Success is not
    # news; only a mismatch is, and that raises.
    _LOG.debug(
        "[OTR.render_driver] ROUTE EQUALITY OK: %d shot(s) match the frozen "
        "route %r", len(section.get("shots") or ()), frozen)
    return True


def assert_coverage_plans(ledger):
    """THE SECOND BOUNDARY: re-validate every stamped coverage plan (chunk 3b).

    ShotLock validated each plan when it built it; this validates the same
    plans again after they have crossed the wire, immediately before execution.
    Validating at both ends is the point -- a plan that survives serialization
    but not execution is precisely the class of defect a durable stamp is
    supposed to prevent, and only a second check at the far end can catch a
    ledger that was hand-edited, replayed from an older revision, or produced
    by a ShotLock built against a different contract.

    Re-validated AGAINST THE LIVE CONTRACT, not just for internal arithmetic:
    an adapter whose declared frame contract changed since the plan was made
    (a version bump, a re-registered engine) must not silently execute a plan
    its current contract would reject.

    Shots with no stamped plan are skipped -- see
    ``otr_shot_lock._stamp_coverage_plan`` for when that is legitimate.

    ALSO RE-DERIVES THE EFFECTIVE CONTRACT (B3, 2026-07-26). The tier's
    ``max_render_frames`` ceiling narrows what the PLANNER may emit for the
    engines in ``frame_contract.PLANNING_CAP_ENGINES``, so the plan must be
    re-validated against the NARROWED contract here, not the declared one -- a
    161-frame segment is perfectly legal for ``ltx_8gb``'s declaration and
    illegal under a tier that pinned 65. The narrowed contract is also stamped
    beside the plan as ``shot["coverage_contract"]``, re-derived here with the
    same two functions, and compared for EXACT equality. That comparison is
    what catches the cases arithmetic cannot: a ceiling that moved between
    plan time and render time, a receipt that appeared or vanished, and a shot
    whose engine was swapped after the stamp by the legacy force-map path
    (``resolve_final_shot_engines`` mutates and THEN calls this). A stamped
    receipt against a re-derived ``None`` is a refusal, and it should be: the
    plan was built under a ceiling the engine now rendering it does not honour.
    """
    try:
        from . import coverage_plan as _cp  # type: ignore
        from . import frame_contract as _fc  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        import coverage_plan as _cp  # type: ignore
        import frame_contract as _fc  # type: ignore

    # THE TIER CEILING, read ONCE off the ledger the plan rode in on, through
    # the SAME normalizer ``otr_shot_lock._planning_ceiling`` stamped it with
    # (B3). ``.get`` with no key present reads as unpinned, which is every
    # hand-built fixture and both HTTP entry points.
    ceiling = _fc.normalized_planning_ceiling(
        ((ledger or {}).get("video") or {}).get("max_render_frames"))

    checked = 0
    for shot in ((ledger or {}).get("video") or {}).get("shots") or ():
        if not isinstance(shot, dict):
            continue
        raw = shot.get("coverage_plan")
        if not isinstance(raw, dict) or not raw:
            continue
        plan = _cp.CoveragePlan.from_dict(raw)
        target = int(shot.get("target_frame_count") or 0)
        if target and int(plan.target_visible_frames) != target:
            raise RenderError(
                "COVERAGE PLAN: shot %s carries a plan for %d visible frame(s) "
                "but the beat's target is %d. The plan and the beat disagree, "
                "so the assembled clip would drift from the beat audio. "
                "NO FALLBACK." % (shot.get("shot_id"),
                                  plan.target_visible_frames, target))
        contract = None
        engine_id = str(shot.get("engine_id") or "")
        if engine_id and _vreg.is_registered(engine_id):
            try:
                contract = _fc.frame_contract_for(_vreg.get_engine(engine_id))
            except Exception:  # noqa: BLE001 -- arithmetic-only check instead
                contract = None
        # THE EFFECTIVE CONTRACT + ITS RECEIPT (B3), derived OUTSIDE the except
        # above: that catch means "this adapter is unbuildable, fall back to an
        # arithmetic-only check", and a misconfigured tier ceiling must not be
        # absorbed into it. ``PlanningCapError`` is a ValueError and is NOT a
        # ``CoveragePlanError``, so the narrow catch below would let it escape
        # this boundary unformatted -- wrap it here, named.
        effective = contract
        expected = None
        if contract is not None:
            try:
                effective = _fc.effective_frame_contract(
                    engine_id, contract, ceiling)
                expected = _fc.coverage_contract_receipt(
                    engine_id, contract, ceiling)
            except _fc.PlanningCapError as exc:
                raise RenderError(
                    "COVERAGE PLAN: shot %s cannot be checked against this "
                    "tier's render-length ceiling: %s. NO FALLBACK."
                    % (shot.get("shot_id"), exc)) from exc
        raw_receipt = shot.get("coverage_contract")
        if raw_receipt is not None and not (
                isinstance(raw_receipt, dict) and raw_receipt):
            raise RenderError(
                "COVERAGE CONTRACT: shot %s carries a coverage_contract of %r, "
                "which is not a receipt. An empty or malformed receipt is not "
                "the same as no receipt -- one means nothing narrowed, the "
                "other means the ledger is damaged. NO FALLBACK."
                % (shot.get("shot_id"), raw_receipt))
        stamped = raw_receipt or None
        if contract is not None:
            if stamped != expected:
                raise RenderError(
                    "COVERAGE CONTRACT: shot %s was planned under %r but this "
                    "ledger re-derives %r for engine %r at ceiling %d. The plan "
                    "was built against a contract that is no longer the one in "
                    "force -- a moved tier ceiling, a re-declared adapter, or "
                    "an engine swapped after the plan was stamped. NO FALLBACK."
                    % (shot.get("shot_id"), stamped, expected,
                       engine_id or "?", ceiling))
        elif isinstance(stamped, dict):
            # The adapter will not resolve here, so the full receipt cannot be
            # re-derived -- but BOTH facts the receipt carries about its own
            # provenance can still be held to the ledger: the ceiling it was
            # planned under, and the engine it was planned FOR.
            #
            # THE ENGINE HALF WAS MISSING (2026-07-27, post-code panel, two
            # seats independently, each with a live repro). Comparing only the
            # ceiling let a stale ltx_8gb receipt ride a shot whose engine had
            # been swapped to something unregistered: same ceiling, no
            # contract to re-derive, and the fall-through validated the plan
            # arithmetic-only -- a plan built for an 8n+1 ladder accepted for
            # an engine never checked against any length. The registered-engine
            # branch above compares the whole receipt, engine id included; this
            # branch has to say the same thing with less information.
            if str(stamped.get("engine_id") or "") != engine_id:
                raise RenderError(
                    "COVERAGE CONTRACT: shot %s was planned for engine %r but "
                    "now carries %r, whose adapter cannot be resolved here. A "
                    "plan built under one engine's ceiling may not be executed "
                    "by another. NO FALLBACK."
                    % (shot.get("shot_id"), stamped.get("engine_id"),
                       engine_id or "?"))
            if _fc.normalized_planning_ceiling(
                    stamped.get("max_render_frames")) != ceiling:
                raise RenderError(
                    "COVERAGE CONTRACT: shot %s was planned under a "
                    "render-length ceiling of %r but this ledger carries %d, "
                    "and its adapter (%s) cannot be resolved here to re-derive "
                    "the rest. NO FALLBACK."
                    % (shot.get("shot_id"), stamped.get("max_render_frames"),
                       ceiling, engine_id or "?"))
        try:
            _cp.validate_coverage_plan(plan, effective)
        except _cp.CoveragePlanError as exc:
            raise RenderError(
                "COVERAGE PLAN: shot %s carries a plan its adapter (%s) cannot "
                "execute: %s. NO FALLBACK."
                % (shot.get("shot_id"), engine_id or "?", exc)) from exc
        checked += 1
    if checked:
        _LOG.debug("[OTR.render_driver] COVERAGE PLANS OK: %d shot(s)", checked)
    return checked


def resolve_final_shot_engines(ledger):
    """THE ROUTE LOCK: resolve every shot's FINAL effective engine, once.

    Both engine mutations in this build run here and nowhere else that matters:
    the ``OTR_FORCE_ENGINE_MAP`` rewrite (:func:`apply_engine_override`) and
    the radio-is-host redirect (:func:`_enforce_radio_is_host`). After this
    returns, ``shot["engine_id"]`` / ``shot["family"]`` are the engine that
    will actually render the beat.

    WHY THIS EXISTS (2026-07-25, kibitz per-beat-stills r1; codex
    ``gpt-5.6-sol`` high and agy ``Gemini 3.6 Flash (High)`` found the same
    ordering defect independently, and the judge's anchor had it too):
    ``otr_video_render_batch`` validated the still spine BEFORE either
    mutation had run -- ``apply_engine_override`` fires inside
    :func:`run_real_episode`, and ``_enforce_radio_is_host`` fires per shot
    inside :func:`build_request_from_shot`, which is later still. So with a
    force map set, or on any announcer/music beat whose picked engine is a
    local HuMo, **the spine was validated against the PICKED engine and the
    beat was rendered by a DIFFERENT one** -- the aspect/geometry mismatch
    class this whole still-plans build exists to close.

    Calling this before ``validate_and_repair_still_spine`` makes the spine
    check see the real routing. It is IDEMPOTENT by construction -- the
    override skips a shot already on its target engine, and the redirect
    no-ops once the engine is no longer HuMo-family -- so the existing
    downstream calls stay in place as defence in depth and cost nothing.

    Mutates the ledger's own shot dicts in place (the established pattern for
    both passes) and returns the ledger for call-site symmetry.

    SUPERSEDED BY THE FREEZE FOR OTR_VideoDirector-BUILT LEDGERS (chunk 1c).
    When the ledger carries a frozen route (``video.roles_effective``, stamped
    by ShotLock from OTR_VideoDirector's capture), this VERIFIES that route and
    mutates NOTHING -- see :func:`assert_frozen_route`. The mutating body below
    now runs only for ledgers that never met the freeze: legacy CPU fixtures
    and the two hand-built HTTP entry points (``/otr/video_render_single``,
    ``/otr/video_render_soak``). That branch is named and logged rather than
    silent, because a fallback nobody can see is how the original defect hid.
    """
    # The coverage plans are validated on BOTH paths -- a hand-built ledger
    # that carries a plan is held to it just as a frozen one is (chunk 3b) --
    # but ALWAYS AFTER the route is final (2026-07-26 QA4). On the frozen path
    # the route arrived final, so the check runs immediately. On the legacy
    # path it must WAIT for the two mutations below: a plan stamped for the
    # PICKED engine and validated before the force map or the radio-host
    # redirect swaps it is a plan checked against an engine that does not
    # render the beat -- the same ordering defect this function's own
    # docstring exists to close for the still spine, one contract further
    # down. Checking early is not merely useless there, it is a false
    # receipt: it reports COVERAGE PLANS OK for routing that no longer holds.
    if assert_frozen_route(ledger):
        assert_coverage_plans(ledger)
        return ledger
    _LOG.debug(
        "[OTR.render_driver] no frozen route on this ledger (legacy / "
        "hand-built / soak path) -- resolving the effective engines here, as "
        "before the 2026-07-25 route freeze")
    ledger = apply_engine_override(ledger)
    for shot in ((ledger.get("video") or {}).get("shots") or []):
        if isinstance(shot, dict):
            _enforce_radio_is_host(shot)
    assert_coverage_plans(ledger)
    return ledger


def apply_engine_override(ledger):
    """Experiment knob (operator ask 2026-06-09: the all-LTX / forced-engine
    episodes): when ``OTR_FORCE_ENGINE_MAP`` is set, re-route each planned
    shot's ``engine_id``/``family`` by role BEFORE rendering, LOUDLY. NO
    FALLBACKS (2026-07-02): a forced engine that fails raises RenderError
    LOUD -- there is no degrade to a floor. Returns the (possibly
    rewritten) ledger.

    FAIL CLOSED ON A MALFORMED MAP (2026-07-25, kibitz per-beat-stills r1 --
    codex ``gpt-5.6-sol`` high and agy reached this independently, judge
    concurring). This used to log ``IGNORED (parse)`` and render the UNFORCED
    plan, which is a silent fallback: the operator asks for a forced-route
    episode, gets an ordinary one, and the only trace is a warning line in a
    long log. It is worse for the still spine specifically -- the spine is
    validated against whatever routing this function leaves behind, so a
    swallowed parse error means stills were minted for engines that are not
    the ones that render. A malformed ``OTR_FORCE_ENGINE_MAP`` is operator
    misconfiguration and it is now terminal, before any GPU time is spent."""
    spec = os.environ.get("OTR_FORCE_ENGINE_MAP", "").strip()
    if not spec:
        return ledger
    try:
        mapping = parse_engine_override(spec)
    except ValueError as exc:
        raise RenderError(
            "OTR_FORCE_ENGINE_MAP is malformed and the forced-route episode "
            "cannot be planned: %s -- fix the map or unset it. NO FALLBACK to "
            "the unforced plan (a silently unforced render is "
            "indistinguishable from a working one)." % exc) from exc
    section = (ledger.get("video") or {})
    n = 0
    try:
        from .._otr_shared import role_compat as _role_compat
    except Exception:  # noqa: BLE001 - never block the override on an import
        _role_compat = None
    for shot in section.get("shots") or []:
        role = str(shot.get("role") or "")
        engine = mapping.get(role) or mapping.get("*")
        if not engine or shot.get("engine_id") == engine:
            continue
        # M3 delta (f): OTR_FORCE_ENGINE_MAP is an UNCONDITIONAL operator
        # experiment knob (forcing any engine onto any role is intentional, e.g.
        # ltx_video onto character_video for b-roll) -- so the force is ALWAYS
        # applied. But annotate a force that does NOT fit the role per the shared
        # role_compat filter with a LOUD warning so a mis-route is visible; the
        # render-time assert_usable gate is still the final authority (forcing
        # never bypasses it -- an unfit pick degrades LOUD at render).
        if _role_compat is not None and role and _vreg.is_registered(engine):
            _eng = _vreg.get_engine(engine)
            _desc = {"engine_id": engine,
                     "roles": tuple(getattr(_eng, "roles", ()) or ()),
                     "required_inputs": tuple(
                         getattr(_eng, "required_inputs", ()) or ())}
            try:
                _fits = _role_compat.engine_fits_role(_desc, role)
            except Exception:  # noqa: BLE001 - unknown role etc. -> warn unfit
                _fits = False
            if not _fits:
                _LOG.warning(
                    "[OTR video] OTR_FORCE_ENGINE_MAP: role=%s does NOT fit "
                    "engine=%s per role_compat (roles=%r required_inputs=%r) -- "
                    "applying the force anyway (operator experiment knob); it "
                    "will degrade LOUD at render if assert_usable rejects it",
                    role, engine, _desc["roles"], _desc["required_inputs"])
        _LOG.warning(
            "[OTR video] LOUD ENGINE OVERRIDE shot=%s role=%s %s -> %s "
            "(OTR_FORCE_ENGINE_MAP)", shot.get("shot_id"), role or "?",
            shot.get("engine_id"), engine)
        shot["engine_id"] = engine
        shot["family"] = engine_family(engine, shot.get("family"))
        n += 1
    if n:
        _LOG.warning("[OTR video] engine override applied to %d shot(s): %r",
                     n, mapping)
    return ledger


#: Engines that count as a REAL LTX radio-open render (BUG-LOCAL-413 guard):
#: the prompt-only ltx_video, the LTX 2.5 HQ I2V lane, and the additive LTX-AV
#: audio lane.
_LTX_OPEN_ENGINES = frozenset(
    {"ltx_video", "ltx25_video", "ltx_audio_in",
     # The foley and mime lanes (2026-08-26) render the LTX 2.5 picture graph
     # unchanged -- only the audio latent's fate differs -- so an open that
     # renders on one is every bit as real an LTX open as `ltx25_video`.
     # Omitting them would report a healthy open as a procgen soft-open and,
     # under OTR_LTX_OPEN_STRICT=1, fail a build that did nothing wrong.
     "ltx25_foley_plus", "ltx25_mime"})
#: Roles whose beats are the radio-console OPENER -- expected to render on an
#: LTX engine, not the procgen/still floor (the 6/15 clips=0 soft-open).
_LTX_OPEN_ROLES = frozenset({"announcer_visual", "music_visual"})


def check_ltx_open_health(manifest, *, strict=None):
    """BUG-LOCAL-413 guard -- surface a radio-OPEN beat (announcer / music
    console opener, incl. the synthetic b000 music_open) that did NOT render on
    an LTX engine. The 6/15 ``eye_of_the_storm`` open was SOFT because the
    episode rendered ZERO LTX clips and the open fell to the by-design procgen /
    still_motion floor at raw 1472x832 (no upscale) -- a SILENT degrade. This
    makes it LOUD: every offending open beat logs a warning; with strict mode
    (env ``OTR_LTX_OPEN_STRICT=1`` or ``strict=True``) it RAISES so a build can
    never ship a procgen-fallback open unnoticed. Pure read of the manifest
    (never touches audio); the procgen floor STAYS the safety net -- this only
    surfaces the degrade, it does not remove the fallback. Returns the list of
    offending open rows (empty == healthy)."""
    if strict is None:
        strict = os.environ.get("OTR_LTX_OPEN_STRICT", "0") == "1"
    bad = []
    for row in (manifest or {}).get("clips") or []:
        role = str(row.get("role") or "")
        bid = str(row.get("beat_id") or "")
        is_open = (role in _LTX_OPEN_ROLES
                   or bid.endswith(_OPENING_MUSIC_SUFFIX))
        if not is_open:
            continue
        eid = str(row.get("engine_id") or "")
        if eid in _LTX_OPEN_ENGINES and row.get("exists"):
            continue                      # healthy: a real LTX open clip
        if _receipt.is_sanctioned_gap(row):
            # C6 (2026-08-28): a SANCTIONED open beat is not a health failure.
            # This check hunts an open that fell to the procgen/still floor
            # when an LTX engine was expected -- a wiring fault. An opener
            # whose required still the model refused never had an engine to
            # fall from, and the operator's ruling says that episode still
            # publishes; raising here in strict mode would contradict it.
            _LOG.warning(
                "[OTR.render_driver] LTX-OPEN: radio-open beat %s was "
                "SANCTIONED (the image model refused its still) -- floored by "
                "design, not a health failure.", bid or row.get("shot_id"))
            continue
        bad.append({"shot_id": row.get("shot_id"), "beat_id": bid,
                    "role": role, "engine_id": eid,
                    "exists": bool(row.get("exists"))})
        _LOG.warning(
            "[OTR.render_driver] LTX-OPEN HEALTH (BUG-LOCAL-413): radio-open "
            "beat %s role=%s rendered on %r (exists=%s) -- NOT an LTX engine; "
            "the open is the procgen/still floor (soft open). Expected one of "
            "%r. Set OTR_LTX_OPEN_STRICT=1 to fail the build.",
            bid or row.get("shot_id"), role, eid, bool(row.get("exists")),
            tuple(sorted(_LTX_OPEN_ENGINES)))
    if bad and strict:
        raise RenderFloorError(
            "LTX-OPEN HEALTH strict (BUG-LOCAL-413): %d radio-open beat(s) fell "
            "to the procgen/still floor instead of an LTX engine: %r"
            % (len(bad), [b["beat_id"] or b["shot_id"] for b in bad]))
    return bad


def _safe_clip_basename(text):
    """Filesystem-safe slug for a persisted clip filename (ASCII, no path chars)."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(text or ""))[:80]


def resolve_episode_id_for_clip_persistence(episode_id, freeze_timestamp=""):
    """Rename-proof the episode id used for durable clip persistence.

    VideoRenderBatch receives the ShotLock ledger while the episode is still
    named ``pending_<ts>``. SignalLostVideo may rename that episode directory to
    the title slug before clips are rendered, so writing to
    ``episodes/pending_<ts>/clips`` can re-create a stale folder and strand the
    provider clips away from the real episode. Mirror the stills/master-audio
    rename contract: if the pending dir is gone, re-key only to the active
    in-flight ledger and, when supplied, require its immutable freeze receipt
    to match. Never choose a newest-mtime sibling. Test mode leaves the id
    untouched unless tests call the helper directly with the env cleared.
    """
    eid = str(episode_id or "").strip()
    if not eid or not eid.startswith("pending_"):
        return eid
    if os.environ.get("OTR_TEST_MODE") == "1":
        return eid
    try:
        from pathlib import Path
        from .. import _otr_ledger as _OTRL
        from .._otr_paths import otr_episodes_root

        episodes_root = Path(otr_episodes_root())
        pending_dir = episodes_root / eid
        if pending_dir.is_dir():
            return eid
        ledger_path = _OTRL.in_flight_ledger_path()
        if not ledger_path:
            return eid
        ledger_path = Path(ledger_path)
        new_episode_dir = Path(ledger_path).parent.parent
        disk_ledger = _OTRL.load_ledger_safe(ledger_path)
        expected_freeze = str(freeze_timestamp or "").strip()
        disk_meta = (
            disk_ledger.get("meta") if isinstance(disk_ledger, dict)
            and isinstance(disk_ledger.get("meta"), dict) else {}
        )
        disk_freeze = str(disk_meta.get("freeze_timestamp") or "").strip()
        if expected_freeze and disk_freeze != expected_freeze:
            _LOG.warning(
                "[OTR video] clip episode-id re-resolve REJECTED: "
                "freeze_timestamp mismatch wire=%r disk=%r",
                expected_freeze, disk_freeze,
            )
            return eid
        if (
            new_episode_dir.is_dir()
            and new_episode_dir.parent == episodes_root
            and new_episode_dir.name != eid
            and isinstance(disk_ledger, dict)
            and str(disk_ledger.get("episode_id") or "").strip()
                == new_episode_dir.name
        ):
            _LOG.warning(
                "[OTR video] LOUD re-resolve: episode_id %r is a stale pending "
                "id (dir renamed before clip persistence); clips re-keyed "
                "to active ledger episode dir %r.",
                eid, new_episode_dir.name)
            return new_episode_dir.name
    except Exception as exc:  # noqa: BLE001 -- never block render on probe
        _LOG.warning("[OTR video] clip episode-id re-resolve skipped: %s", exc)
    return eid


def persist_episode_clips(result, episode_id):
    """Move each rendered per-beat clip from the janitor-swept ``_shared/tmp``
    scratch tier to the DURABLE ``episodes/<ep>/clips/`` workspace (clip-fill
    Piece 4; operator directive: every rendered asset lives under
    ``otr/episodes/<ep>/``). Rewrites ``clip['path']`` in place to the stable path
    so :func:`build_clip_manifest` (and the composite) reference the persisted clip
    instead of a path the janitor will delete.

    Best-effort + LOUD: a missing episode_id, an unresolvable output tree, or a
    per-clip move error is logged and skipped -- it NEVER aborts the render (the
    clip still plays from tmp until the next sweep). Directory clips (3D frame
    dirs) are left in place. Returns ``result`` (mutated in place)."""
    clips = (result or {}).get("clips") or {}
    if not episode_id or not clips:
        return result
    episode_id = resolve_episode_id_for_clip_persistence(episode_id)
    try:
        from .._otr_paths import otr_clips_dir
        dest_dir = str(otr_clips_dir(str(episode_id)))
        os.makedirs(dest_dir, exist_ok=True)
    except Exception as exc:              # noqa: BLE001 -- no output tree -> skip LOUD
        _LOG.warning("[OTR video] persist_episode_clips: cannot resolve "
                     "episodes/%s/clips/ (%s) -- clips stay in _shared/tmp "
                     "(janitor-swept)", episode_id, exc)
        return result
    # shot_id -> (role, engine) for nice <beat>_<role>_<engine> filenames.
    shot_meta = {}
    for shot in (((result or {}).get("ledger") or {}).get("video") or {}).get("shots") or []:
        if isinstance(shot, dict) and shot.get("shot_id"):
            shot_meta[str(shot["shot_id"])] = (
                str(shot.get("role") or ""), str(shot.get("engine_id") or ""))
    import shutil
    moved = 0
    for sid, clip in clips.items():
        if not isinstance(clip, dict):
            continue
        role, eng = shot_meta.get(str(sid), ("", str(clip.get("engine_id") or "")))
        stem = "_".join(p for p in (str(sid), role, eng) if p) or str(sid)
        safe_stem = _safe_clip_basename(stem)
        if str(clip.get("type") or "video") != "directory":
            src = str(clip.get("path") or "")
            if src and os.path.isfile(src):
                dst = os.path.join(dest_dir, safe_stem + ".mp4")
                if os.path.dirname(os.path.abspath(src)) != os.path.abspath(dest_dir):
                    try:
                        if os.path.abspath(src) != os.path.abspath(dst):
                            shutil.move(src, dst)
                            clip["path"] = dst
                            moved += 1
                    except Exception as exc:  # noqa: BLE001 -- one bad move never aborts
                        _LOG.warning(
                            "[OTR video] persist_episode_clips: move FAILED %s -> "
                            "%s (%s) -- clip stays in tmp", src, dst, exc)
    if moved:
        _LOG.info("[OTR video] persisted %d clip(s) to episodes/%s/clips/",
                  moved, episode_id)
    return result


def _segment_receipt_projection(segments):
    """The DURABLE per-segment receipt for one assembled beat, or ``None``.

    ``render_beat_coverage`` keeps a rich per-segment row for the operator's
    2026-08-01 traceability ask; this is the narrow slice of it that belongs in
    a manifest an audit will read months later.

    Why a projection and not the row itself:

    * ``path`` is DELIBERATELY dropped. ``persist_episode_clips`` moves only the
      ASSEMBLED beat, so every segment file stays in janitor-swept scratch. A
      segment filename in a durable manifest would name a file that was never
      persisted there -- a receipt pointing at nothing is worse than no receipt.
    * ``visible_frames`` is dropped too: it is exactly
      ``render_frames - drop_head - trim_tail``, already guaranteed by the
      coverage plan, so persisting it would only add a second number that can
      disagree with the first three.

    What survives is what makes the beat's frame accounting CHECKABLE rather
    than merely asserted -- the grader re-derives the beat totals from these
    rows and names the offending ``segment_id`` when they do not add up.
    """
    rows = segments if isinstance(segments, list) else None
    if not rows:
        return None
    return [{key: row.get(key) for key in _acceptance_module().SEGMENT_RECEIPT_KEYS}
            for row in rows if isinstance(row, dict)]


def _acceptance_module():
    """``acceptance``, imported lazily and locally.

    Same reason as ``render_beat_coverage``'s local import: the grader may
    import nothing (a test pins its import list to exactly ``["__future__"]``),
    so the shared receipt vocabulary lives there and is reached from here.
    """
    try:
        from . import acceptance as _mod  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        import acceptance as _mod  # type: ignore
    return _mod


#: The manifest's frame-receipt contract version (no-mirror enforcement, step 2).
#: Version 1 means: every delivered ``type == "video"`` row DECLARES how its
#: frames got there. Without a version, "reject an explicit non-none mode" can
#: never catch an adapter that REGRESSES by dropping its receipt entirely --
#: absence would stay permanently legal, and silence is exactly what a lane that
#: pads without saying so looks like.
FRAME_RECEIPT_VERSION = 1

#: How far two closing-theme chunks may disagree about their shared boundary and
#: still count as contiguous. The chunks are minted as ``start + i * (dur/count)``
#: (``scene_sequencer.py``), so consecutive rows meet exactly up to float error;
#: this absorbs the error and nothing more. A real gap must NOT be bridged --
#: that would extend the loop authorization over silence.
_CLOSING_WINDOW_GAP_TOLERANCE_S = 1e-6


def _finite_seconds(value):
    """A finite, non-bool number of seconds, or ``None``. NEVER raises.

    ``bool`` is rejected explicitly for the same reason ``acceptance.frame_count``
    rejects it: ``True`` is an ``int``, so it would silently become 1.0 second and
    read as a real measurement.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def closing_theme_frame_window(lines, fps):
    """The frame span the CLOSING THEME occupies, or ``None``. PURE.

    This is the ONLY thing that may authorize the composite to loop a clip, so
    it is written to REFUSE rather than to guess. Every failure below returns
    ``None``, and ``None`` means the tail gets floor/black. Never loop on doubt.

    THIS IS NOT ``video_engine``'S ``music_close_start_f``, AND THE TWO MUST NOT
    BE UNIFIED. That one places the closing TITLE CARD and, when the real window
    cannot be resolved, deliberately SYNTHESIZES one over the last few seconds
    (``video_engine.py:323-334``, stamped ``close_synthesized``) because a card
    has to be drawn somewhere. Sourcing this from that would authorize a loop
    over ANY episode's tail, which is precisely the "loops for any unexplained
    tail" defect this window exists to remove. A placement hint may be guessed;
    an authorization may not.

    THE EVIDENCE, as ``scene_sequencer`` actually mints it: the closing cue is
    mirrored into the ledger as one row per chunk, each carrying
    ``speaker_role == "music_close"``, ``start_s_space == "master_mix"``, a
    shared ``music_cue_id``, and a 1-based ``music_chunk_index`` out of
    ``music_chunk_count``. Both role and space are required: ``music_close``
    alone can also appear in the SCENE-audio space, whose seconds are measured
    against a different zero, and converting those to master-timeline frames
    would place the window somewhere the closing theme never was.

    ACCEPTED ONLY IF ALL OF:

    * every row has a finite ``start_s >= 0`` and a finite ``dur_s > 0``;
    * all rows name ONE ``music_cue_id`` -- two cues is an ambiguity, not a
      wider window, and merging them would authorize the span between them;
    * the chunk indices are exactly ``1..music_chunk_count``, each once, with
      every row agreeing on the count -- a missing chunk means the window has a
      hole and its true extent is unknown;
    * consecutive chunks are contiguous or overlapping. A real gap is refused
      rather than bridged.
    """
    # ``OverflowError`` belongs in this guard beside the other two. ``int()`` of
    # an infinite float raises it rather than ValueError, so an fps of inf would
    # have escaped a two-exception catch -- the same hole ``acceptance.frame_count``
    # documents for the bare token ``Infinity`` that ``json.load`` parses.
    try:
        rate = int(fps)
    except (TypeError, ValueError, OverflowError):
        return None
    if rate <= 0:
        return None
    # THE CONTAINER RULE, same as ``acceptance.manifest_clips``. ``lines or ()``
    # short-circuits on any TRUTHY value, so a scalar ``5`` passed straight
    # through and ``for ln in 5`` raised TypeError -- out of a function whose
    # contract is that it never raises, and through ``build_clip_manifest``,
    # which every render calls unconditionally. A corrupt ledger field must
    # refuse the window, not destroy the manifest.
    if not isinstance(lines, (list, tuple)):
        return None
    rows = [ln for ln in lines
            if isinstance(ln, dict)
            and str(ln.get("speaker_role") or "") == "music_close"
            and str(ln.get("start_s_space") or "") == "master_mix"]
    if not rows:
        return None

    # ``str()`` first, so an unhashable cue id cannot reach the set.
    cue_ids = {str(r.get("music_cue_id") or "") for r in rows}
    if len(cue_ids) != 1:
        return None

    # VALIDATE, THEN HASH -- never the other way round. This was a set
    # comprehension over the RAW field, so a ``music_chunk_count`` of ``[3]``
    # raised "unhashable type: 'list'" three lines BEFORE the isinstance check
    # that exists to reject it. A type guard placed after the operation it is
    # guarding is not a guard.
    declared_count = None
    for row in rows:
        count = row.get("music_chunk_count")
        if isinstance(count, bool) or not isinstance(count, int):
            return None
        if declared_count is None:
            declared_count = count
        elif count != declared_count:
            return None
    if declared_count < 1 or declared_count != len(rows):
        return None
    indices = set()
    for row in rows:
        index = row.get("music_chunk_index")
        if isinstance(index, bool) or not isinstance(index, int):
            return None
        indices.add(index)
    if indices != set(range(1, declared_count + 1)):
        return None

    spans = []
    for row in rows:
        start = _finite_seconds(row.get("start_s"))
        duration = _finite_seconds(row.get("dur_s"))
        if start is None or start < 0 or duration is None or duration <= 0:
            return None
        spans.append((start, start + duration))
    # A RUNNING MAX, not the previous row's end. Chunks are normally equal and
    # in order, but nothing here may ASSUME that: if one chunk fully contains a
    # shorter one, comparing the next start against the SHORT chunk's end would
    # report a gap that the long chunk already covers -- and taking the
    # last-sorted end as the window's end would stop the authorization short of
    # the theme it describes. Union the intervals properly and both go away.
    spans.sort()
    reach = spans[0][1]
    for next_start, next_end in spans[1:]:
        if next_start > reach + _CLOSING_WINDOW_GAP_TOLERANCE_S:
            return None
        reach = max(reach, next_end)

    # THE COMPOSITE'S OWN CONVENTION, so the window and the cursor it is compared
    # against are measured on one ruler. ``plan_timeline_segments`` places a beat
    # at ``int(round(start_s * fps))`` (``otr_silent_composite.py:385``), and the
    # manifest quantizes the accepted duration UP to a whole frame for the
    # terminal boundary (``timeline_total`` below). Deriving either differently
    # here would put the authorization a frame away from the thing it authorizes.
    # THE PRODUCT, NOT JUST THE OPERANDS. ``_finite_seconds`` proves each value
    # is finite on its own; multiplying a finite-but-enormous one by the frame
    # rate can still overflow to inf, and ``round(inf)`` raises OverflowError.
    # Checking the inputs and not the arithmetic is how a total function stops
    # being total.
    start_scaled = spans[0][0] * rate
    end_scaled = reach * rate
    if not (math.isfinite(start_scaled) and math.isfinite(end_scaled)):
        return None
    start_frame = max(0, int(round(start_scaled)))
    end_frame = int(math.ceil(end_scaled))
    if end_frame <= start_frame:
        return None
    return {"start": start_frame, "end": end_frame}


#: THE CADENCE / DELIVERY RECEIPT KEYS (Ghost Signal, 2026-08-22). ONE tuple,
#: because these travel through four INDEPENDENT hand-written projections --
#: the clip manifest, `_clip_summary`, the strict run trace, and the render
#: batch's lossless `per_clip` -- and a key list written four times is a key
#: list that drifts once. Every projection copies PRESENT-KEY-ONLY, so a legacy
#: row keeps its exact historical shape.
_CADENCE_DELIVERY_RECEIPT_KEYS = (
    "model_frame_count", "cadence_mode", "cadence_source_frame_count",
    "cadence_delivered_frame_count", "cadence_tail_trim",
    "delivery_scale_mode",
)

#: THE FOLEY SIDECAR RECEIPTS (the LTX 2.5 foley bed, 2026-08-26). Imported
#: from the module that owns the stem format rather than restated, so the
#: engine, this manifest and the mux cannot come to disagree about which keys
#: make a stem. A silent lane carries none of them and the copy below is
#: present-key-only, so every existing row stays byte-identical.
try:
    from .foley_stems import FOLEY_RECEIPT_KEYS as _FOLEY_RECEIPT_KEYS
except ImportError:  # pragma: no cover -- flat test imports
    from foley_stems import FOLEY_RECEIPT_KEYS as _FOLEY_RECEIPT_KEYS


def build_clip_manifest(result, *, episode_id=""):
    """Pure, beat-ordered per-beat clip manifest from a :func:`run_real_episode`
    result -- the STRING contract OTR_SilentComposite assembles. Shot order is
    the OUTPUT ledger's shots (already in beat order). Each row carries the clip
    path + its full render-request frame count; ``engine_histogram`` counts the
    on-disk clips per engine so the keystone can assert HuMo ran on the talking
    beats and the episode is NOT all-procgen.

    ``total_target_frames`` is the ONE positioned output-timeline boundary. A
    post-audio ledger may contain overlapping rows at opening/body and
    body/closing crossfades, so summing the full per-shot render requests is not
    the episode length. When every row is positioned and the durable ledger owns
    ``total_episode_dur_s``, quantize that accepted duration upward to a complete
    CFR frame. ``render_target_frames`` retains the old sum as truthful work
    telemetry. Sparse/legacy manifests keep the old sequential-sum behavior.
    The frozen audio bytes are never read or mutated here."""
    led = result.get("ledger") or {}
    section = led.get("video") or {}
    shots = section.get("shots") or []
    clips = result.get("clips") or {}
    canvas = section.get("canonical_canvas") or {}
    lines = {str(ln.get("line_id")): ln for ln in (led.get("lines") or [])
             if isinstance(ln, dict) and ln.get("line_id")}
    # ST-4: init observability joins the manifest rows via the trace (the
    # request stamps run_episode copied; keyed by shot_id).
    trace_by_shot = {str(r.get("shot_id") or ""): r
                     for r in (result.get("trace") or [])
                     if isinstance(r, dict)}
    rows = []
    total = 0
    hist = {}
    # Read once per manifest, from the receipt -- see the row's own note.
    _gap_beats_for_manifest = sanctioned_gap_beat_ids(led)
    for order, shot in enumerate(shots):
        sid = shot.get("shot_id")
        clip = clips.get(sid) or {}
        path = str(clip.get("path") or "")
        ctype = str(clip.get("type") or "video")
        if ctype == "directory":
            # 3D plan 7.2 p3: a directory clip is real when the dir holds
            # EXACTLY target_frame_count sorted nonzero frames (shared rule).
            from .directory_clip import frame_dir_summary
            exists, _n, _b = frame_dir_summary(
                path, expect_frames=shot.get("target_frame_count"))
        else:
            exists = bool(path) and os.path.isfile(path)
        tfc = int(shot.get("target_frame_count") or 0)
        total += tfc
        eid = clip.get("engine_id") or shot.get("engine_id")
        # Round 5 F5: beat ids via the shared rule (synthetic -> the bare
        # beat id, not "shot_..."), and start_s falls back to the SHOT row's
        # stamp -- one None row silently degraded the whole composite from
        # positioned to sequential mode (plan_timeline_segments requires ALL
        # rows positioned; the 2026-06-10 acceptance episode hit this).
        bid = _beat_id_for_shot(shot)
        sraw = lines.get(bid, {}).get("start_s")
        if sraw in (None, ""):
            sraw = shot.get("start_s")
        try:
            start_s = float(sraw) if sraw not in (None, "") else None
        except (TypeError, ValueError):
            start_s = None
        # Round 5 F5: the talking-head face check is mechanical -- each row
        # carries the resolved char_id + the portrait it staged.
        row_char = str(shot.get("char_id")
                       or lines.get(bid, {}).get("char_id") or "")
        trow = trace_by_shot.get(str(sid or ""), {})
        row = {
            "order": order, "shot_id": sid, "beat_id": bid,
            "engine_id": eid,
            # C3 (2026-08-28): the sanction travels to the CONSUMERS.
            # ``timeline_quality_report`` and node 92's success predicate both
            # read ``manifest["clips"]`` and never see ``video.shots``, so
            # without this projection a refused beat and a crashed one are the
            # same absent row to every reader downstream -- and counting them
            # alike is what would let a broken render publish as "degraded".
            # Stamped ONLY when the receipt says so; never inferred from the
            # missing file.
            "status": (_receipt.STATUS_SANCTIONED_GAP
                       if bid in _gap_beats_for_manifest
                       else _receipt.STATUS_OK),
            "role": str(shot.get("role") or ""),
            "family": clip.get("family") or shot.get("family") or "",
            "path": path,
            "type": ctype,
            "frame_count": int(clip.get("frame_count") or 0),
            "target_frame_count": tfc,
            "start_s": start_s,
            "char_id": row_char,
            "init_image": _portrait_index(led).get(row_char, ""),
            "init_source": str(trow.get("init_source") or ""),
            "init_image_used": str(trow.get("init_image") or ""),
            "exists": exists,
            # S-B/E5 recipe receipt: the render engine threads recipe / quant /
            # LoRA / canvas / measured peak into the canonical clip; carry it onto
            # the manifest row so the ledger recipe-stamp self-documents what made
            # each beat. None for engines that emit no receipt (OPTIONAL field).
            "recipe": clip.get("recipe"), "quant": clip.get("quant"),
            "use_lora": clip.get("use_lora"),
            "render_canvas": clip.get("render_canvas"),
            "vram_peak_mb": clip.get("vram_peak_mb"),
            # HOW MANY OF THOSE FRAMES ARE REAL (WIRE-W5, 2026-07-29). Same
            # shape and the same reason as ``recipe`` above: the engine stamps
            # them on the canonical clip and they die there unless the manifest
            # carries them. ``frame_count`` cannot answer the question -- a
            # ping-pong-extended clip carries exactly the number a real one
            # does, which is precisely the evidence a pad forges -- so the
            # acceptance grader has no way to reject a mirror-filled clip on a
            # lane claiming real multi-clip coverage without these. OPTIONAL:
            # None on every engine that emits no receipt.
            #
            # SCOPE (2026-08-06): on a single-render clip these are the engine's
            # own counts. On an ASSEMBLED beat they are beat-scope --
            # ``native_frame_count`` is every native frame the beat RENDERED
            # (243 for a 3x81 chain) and ``delivered_native_frame_count`` is the
            # 241 that survived the seams. The two are deliberately not the same
            # number, and only the second may be weighed against
            # ``frame_count``: comparing the work against the output is what
            # reported every honest multi-segment beat as padded.
            "native_frame_count": clip.get("native_frame_count"),
            "extension_mode": clip.get("extension_mode"),
            # THE CADENCE AND DELIVERY RECEIPTS (Ghost Signal, 2026-08-22).
            # PRESENT-KEY-ONLY, and that is the whole care in this block: a
            # legacy row that never carried these must not acquire six null
            # keys, because absence is what tells a reader "this lane made no
            # such declaration" and a null would read as "it declared nothing",
            # which is a different and false claim. Copied here beside
            # native_frame_count for the same reason those are: the engine
            # stamps them on the canonical clip and they die there unless the
            # manifest carries them.
            **{k: clip[k] for k in _CADENCE_DELIVERY_RECEIPT_KEYS
               if k in clip},
            # THE FOLEY SIDECAR, carried for the same reason and with the same
            # care: the engine stamps these on the canonical clip and they die
            # there unless the manifest carries them, and this is the channel
            # OTR_MasterAudioMux reads the bed off. PRESENT-KEY-ONLY -- a
            # silent row must not acquire six nulls, because absence is what
            # tells the mux "this beat has no bed" and a null would read as a
            # bed that failed to write.
            **{k: clip[k] for k in _FOLEY_RECEIPT_KEYS if k in clip},
            # WHICH CLOCK ``start_s`` ABOVE IS ON. Carried so the mux can
            # PROVE a foley stem's placement instead of assuming it: the
            # assembler rewrites the ledger's line rows from "scene_audio" to
            # "master_mix" space once the themes are folded in, and splatting
            # a bed at a scene-audio offset into a master-mix timeline puts
            # every beat's audio under the wrong picture by the length of the
            # opening theme. Present-key-only, so nothing that never declared
            # a space acquires one.
            **({"start_s_space": str(lines[bid]["start_s_space"])}
               if bid in lines and lines[bid].get("start_s_space") else {}),
            # PRESENT ONLY WHEN THE COVERAGE ASSEMBLER MINTED THIS BEAT, and
            # that absence is information rather than noise: a beat with no
            # plan, or a plan owing no head/tail work, takes the historical
            # single-render path and never acquires one. Do not "tidy up" the
            # null -- it says which path produced the clip.
            "delivered_native_frame_count":
                clip.get("delivered_native_frame_count"),
            "segments": _segment_receipt_projection(clip.get("segments")),
        }
        # C1 (textured-hero 3D PoC): a mesh_stage DIRECTORY clip is a textured
        # turntable mesh on a TRANSPARENT background -- it composites over a
        # GENERATED scene PLATE (the per-beat scene-still coverage already mints
        # one via the per-role image engine, so the plate is image-model
        # AGNOSTIC). Stamp it ONLY when the file exists (os.path.isfile -> never
        # a None/"null" path rides the manifest channel); a missing plate warns
        # LOUD and the composite falls back to the floor/black (never silent).
        # The field is OPTIONAL: every non-mesh row omits it -> byte-identical.
        if exists and ctype == "directory" and str(eid) == "mesh_stage":
            _plate = _still_index(led).get(bid, "")
            if _plate and os.path.isfile(_plate):
                row["bg_still_path"] = _plate
            else:
                _LOG.warning(
                    "[OTR.render_driver] mesh_stage beat %s has NO scene plate "
                    "(bg_still_path) -- the textured 3D hero composites over the "
                    "floor/black fallback (LOUD; expected a per-beat scene still)",
                    bid)
        rows.append(row)
        if exists:
            hist[eid] = hist.get(eid, 0) + 1
    render_rows = [r for r in rows if int(r.get("target_frame_count") or 0) > 0]
    positioned = (bool(render_rows)
                  and all(r.get("start_s") is not None for r in render_rows))
    timeline_total = total
    timeline_source = "render_target_sum"
    timeline_duration_s = None
    if positioned:
        try:
            timeline_duration_s = float(led.get("total_episode_dur_s"))
        except (TypeError, ValueError):
            timeline_duration_s = None
        if timeline_duration_s is not None and timeline_duration_s > 0:
            # A CFR video must cover the final fractional frame of the accepted
            # master timeline. The terminal mux retains its own small
            # quantization tolerance; this does not widen it.
            timeline_total = max(1, int(math.ceil(
                timeline_duration_s * int(section.get("fps") or 25))))
            timeline_source = "ledger.total_episode_dur_s"

    manifest = {
        "episode_id": str(episode_id or ""),
        "video_revision": int(section.get("video_revision") or 1),
        "frame_receipt_version": FRAME_RECEIPT_VERSION,
        "fps": int(section.get("fps") or 25),
        "canvas": {"w": int(canvas.get("w") or 0), "h": int(canvas.get("h") or 0)},
        "n_beats": len(rows),
        "clip_count": sum(1 for r in rows if r["exists"]),
        # Existing consumers read total_target_frames. Its intended contract is
        # the final output timeline, now corrected for positioned overlaps.
        "total_target_frames": timeline_total,
        "timeline_total_frames": timeline_total,
        "render_target_frames": total,
        "timeline_frame_source": timeline_source,
        "timeline_duration_s": timeline_duration_s,
        "engine_histogram": hist,
        "clips": rows,
    }
    # THE ONE SANCTIONED REUSE, GIVEN A FRAME-DOMAIN WINDOW (no-mirror step 2).
    # Operator ruling 2026-08-06: no mirror or ping-pong anywhere EXCEPT the
    # closing loop, which is confirmed OK -- and it is the CLOSING-THEME
    # BACKDROP, not the credits roll (``OTR_CreditsRoll`` freezes a frame and
    # never loops). The composite's tail currently loops the last clip for ANY
    # unexplained shortfall; this is the evidence that lets it tell the
    # sanctioned case from the rest.
    #
    # ONLY ON A POSITIONED MANIFEST, and that condition is not bookkeeping. On a
    # sequential manifest the rows carry no timeline position, so the composite's
    # cursor and these frame numbers are measured from different origins and
    # comparing them would authorize a loop at an arbitrary offset. Absent means
    # "not established", and the composite must then refuse to loop at all.
    if positioned:
        window = closing_theme_frame_window(led.get("lines"), manifest["fps"])
        if window is not None:
            manifest["closing_theme_frame_window"] = window
    # BUG-LOCAL-413 guard: LOUD-warn (opt-in strict raises) if a radio-open beat
    # fell to the procgen/still floor instead of an LTX engine -- so the 6/15
    # silent soft-open can never ship unnoticed. Read-only; fallback untouched.
    check_ltx_open_health(manifest)
    return manifest


# --------------------------------------------------------------------------- #
# A-S7.5 full-episode soak (two back-to-back episodes on REAL engines)
# --------------------------------------------------------------------------- #
def _clip_summary(clip):
    """Compact, JSON-able view of a rendered clip + its on-disk reality.

    Directory semantics (3D plan 7.2 p3): a ``type=="directory"`` clip is
    "real" when the dir exists with EXACTLY ``frame_count`` sorted nonzero
    frames (the shared :mod:`directory_clip` rule); ``size`` is the frames'
    total bytes so ``all_clips_real``'s ``size > 0`` keeps working."""
    path = (clip or {}).get("path", "")
    if (clip or {}).get("type") == "directory":
        from .directory_clip import frame_dir_summary
        exists, _n, size = frame_dir_summary(
            path, expect_frames=(clip or {}).get("frame_count"))
    else:
        exists = bool(path) and os.path.isfile(path)
        size = os.path.getsize(path) if exists else 0
    return {"engine_id": (clip or {}).get("engine_id"),
            "family": (clip or {}).get("family"),
            "frame_count": (clip or {}).get("frame_count"),
            "path": path, "exists": exists, "size": size,
            # Ghost Signal cadence/delivery, present-key-only. This projection
            # is independent of the manifest above -- it is what reaches the
            # episode manifest and the single-clip path -- so it must copy them
            # too or the receipts silently vanish on exactly one of the routes.
            **{k: (clip or {})[k] for k in _CADENCE_DELIVERY_RECEIPT_KEYS
               if k in (clip or {})},
            # THE TELEMETRY WAS BEING DROPPED HERE (relayed from the concurrent
            # coder window, 2026-08-11; folded into lane 7 because this lane's
            # solo smoke is what qualifies its new 1024x576 declaration, and a
            # peak that never reaches disk cannot qualify anything).
            #
            # These six keys are produced by VramPeakProbe and the adapters'
            # own receipts, and this summary -- the only shape that reaches the
            # episode manifest (:5016) and the single-clip path (:5190) --
            # returned six keys and none of them. So a lane could smoke green
            # and leave no usable cost-row seed on disk; wan_ti2v and fastwan
            # both did. Purely ADDITIVE: no existing key changes meaning.
            #
            # A cost row may be seeded ONLY from a true VramPeakProbe MAXIMUM.
            # A single nvidia-smi reading is a LOWER BOUND, and a row built on
            # a lower bound under-predicts -- which admits renders that then
            # OOM. A missing peak must stay NULL here rather than be filled in
            # from a watcher's sample.
            "vram_peak_mb": (clip or {}).get("vram_peak_mb"),
            "recipe": (clip or {}).get("recipe"),
            "quant": (clip or {}).get("quant"),
            "render_canvas": (clip or {}).get("render_canvas"),
            "native_frame_count": (clip or {}).get("native_frame_count"),
            "extension_mode": (clip or {}).get("extension_mode")}


def _episode_facts(ep):
    # `meta` was accepted and never read (removed 2026-08-28). The report this
    # builds reads its meta at the enclosing layer, where the value is actually
    # in scope. (scripts/otr_video_soak.py has its own same-named helper that
    # DOES use meta -- different function, untouched.)
    led = ep["ledger"]
    sec = led["video"]
    shots = {s["shot_id"]: s for s in sec["shots"]}
    clips = {sid: _clip_summary(c) for sid, c in ep["clips"].items()}
    # Route-A (2026-06-28 HuMo-14B promotion): count the promoted 14B tier and
    # assert it ONLY rendered on character_video shots (face + audio role). The
    # histogram gate previously saw only "humo" (the portrait base), so the 14B
    # promotion was invisible to acceptance. A non-empty misrouted list means the
    # per-role routing leaked humo_14B_169 onto a non-character role.
    humo_14b_169 = sorted(sid for sid, c in clips.items()
                          if c["engine_id"] == "humo_14B_169" and c["exists"])
    humo_14b_169_misrouted = sorted(
        sid for sid in humo_14b_169
        if (shots.get(sid) or {}).get("role") != "character_video")
    return {
        "n_clips": len(ep["clips"]),
        "all_clips_real": all(c["exists"] and c["size"] > 0
                              for c in clips.values()),
        "video_revision": sec["video_revision"],
        "audio_sha": led["audio"]["master_audio_sha256"],
        "humo_rendered": sum(1 for c in clips.values()
                             if c["engine_id"] == "humo" and c["exists"]),
        "humo_14B_169_rendered": len(humo_14b_169),
        "humo_14B_169_misrouted": humo_14b_169_misrouted,
        "vram_peak_mb": ep["vram_peak_mb"],
        "trace": ep["trace"],
        "clips": clips,
    }


def assemble_report(meta, input_ledger, e1, e2, *, elapsed_s,
                    oom_contract=None):
    return {
        "meta": meta,
        "elapsed_s": round(float(elapsed_s), 1),
        "episode_1": _episode_facts(e1),
        "episode_2": _episode_facts(e2),
        "input_shot_count": len(input_ledger["video"]["shots"]),
        # NO-TRAIL LOUD contract (2026-07-02): the forced-OOM leg's outcome --
        # {"raised": bool, "error_type": str, "detail": str} or None when the
        # leg was not run.
        "oom_contract": oom_contract,
    }


def assert_soak_ok(report):
    """Assert every A-S7.5 GPU-soak invariant; raise :class:`SoakError` on any
    violation. Returns the list of passed-check descriptions for the report.

    NO-TRAIL LOUD contract (2026-07-02 NO-FALLBACKS rip): the two CLEAN
    episodes must produce every clip deterministically with the frozen audio
    untouched, and the forced-OOM leg (when present) must have RAISED a named
    RenderError -- no trail matching, no decisions, no swap."""
    meta = report["meta"]
    n = meta["n_beats"]
    checks = []
    for tag in ("episode_1", "episode_2"):
        f = report[tag]
        if f["n_clips"] != n or not f["all_clips_real"]:
            raise SoakError("%s: not every beat produced a real on-disk clip "
                            "(%d/%d, all_real=%s)"
                            % (tag, f["n_clips"], n, f["all_clips_real"]))
        if f["video_revision"] != 1:
            raise SoakError("%s: video_revision bumped to %r (rendering never "
                            "re-locks the plan)" % (tag, f["video_revision"]))
        if f["audio_sha"] != FROZEN_AUDIO_SHA:
            raise SoakError("%s: frozen audio sha changed (%r) -- the render "
                            "driver must never touch audio" % (tag, f["audio_sha"]))
        if f["humo_rendered"] < 1:
            raise SoakError("%s: humo never rendered in-process (0 real humo "
                            "clips) -- the heavy in-process forward did not run"
                            % tag)
        # Route-A: the promoted 14B tier must NEVER land on a non-character role
        # (it needs face + audio; render_shot has no fallbacks). .get keeps the
        # synthetic-report path (which never routes the 14B) valid.
        if f.get("humo_14B_169_misrouted"):
            raise SoakError("%s: humo_14B_169 rendered on non-character_video "
                            "shot(s) %r -- per-role routing leak"
                            % (tag, f["humo_14B_169_misrouted"]))
        checks.append("%s: %d real clips; %d humo in-process renders; "
                      "VRAM peak %s MB (telemetry); frozen audio untouched"
                      % (tag, n, f["humo_rendered"], f["vram_peak_mb"]))
    if report["episode_1"]["trace"] != report["episode_2"]["trace"]:
        raise SoakError("non-deterministic: the two episodes' render traces "
                        "(per-shot attempts + final engine) differ")
    checks.append("determinism: two back-to-back episodes identical (traces)")
    oc = report.get("oom_contract")
    if oc is not None:
        if not oc.get("raised") or oc.get("error_type") != "RenderError":
            raise SoakError(
                "LOUD-failure contract violated: a forced OOM must RAISE "
                "RenderError (got raised=%s error_type=%r) -- NO FALLBACKS"
                % (oc.get("raised"), oc.get("error_type")))
        checks.append("LOUD-failure contract: forced OOM raised RenderError "
                      "(no swap, no trail)")
    return checks


def run_gpu_soak(*, n_beats=40, oom_index=20, frame_count=25, assets=None):
    """Run the A-S7.5 full-episode soak on REAL GPU engines TWICE back-to-back
    (CLEAN fixture -- no forced OOM), then prove the NO-TRAIL LOUD-failure
    contract on a separate forced-OOM leg (``oom_index`` names the beat; the
    forced OOM must RAISE RenderError). Asserts every invariant and returns
    the structured report. Raises nothing itself -- failures are embedded
    (never a fake pass). VRAM peak is recorded as telemetry -- no ceiling
    enforcement (the operator's tier JSON owns the OOM budget)."""
    section, meta = build_soak_fixture(n_beats=n_beats, oom_index=None)
    ledger = build_full_ledger(section)
    t0 = time.time()
    e1 = run_episode(ledger, assets=assets, frame_count=frame_count)
    e2 = run_episode(ledger, assets=assets, frame_count=frame_count)
    # LOUD-failure contract leg: a forced OOM on the synthetic heavy-engine
    # stub must RAISE RenderError -- no swap, no restamp, no trail.
    oom_contract = {"raised": False, "error_type": "", "detail": ""}
    if oom_index is not None:
        _n_oom = max(1, min(n_beats, oom_index + 1))
        oom_section, oom_meta = build_soak_fixture(
            n_beats=_n_oom, oom_index=min(oom_index, _n_oom - 1))
        oom_ledger = build_full_ledger(oom_section)
        try:
            run_episode(oom_ledger, oom_shot_id=oom_meta["oom_shot_id"],
                        oom_engines=OOM_ENGINES, assets=assets,
                        frame_count=frame_count)
        except RenderError as exc:
            oom_contract = {"raised": True, "error_type": "RenderError",
                            "detail": str(exc).splitlines()[0][:200]}
        except Exception as exc:  # noqa: BLE001 -- report the wrong type honestly
            oom_contract = {"raised": True,
                            "error_type": type(exc).__name__,
                            "detail": str(exc).splitlines()[0][:200]}
    report = assemble_report(meta, ledger, e1, e2,
                             elapsed_s=time.time() - t0,
                             oom_contract=(oom_contract
                                           if oom_index is not None else None))
    try:
        report["passed_checks"] = assert_soak_ok(report)
        report["ok"] = True
    except SoakError as exc:             # embed the failure -- never a fake pass
        report["ok"] = False
        report["error"] = str(exc)
    return report


def render_single(engine_name="humo", *, assets=None, frame_count=33,
                  canvas=None, profile=None):
    """Render ONE shot via a SINGLE engine with NO fallback -- the focused
    in-process validation (surfaces the real exception so the in-process forward
    can be debugged in isolation before the full soak). Returns a result dict.

    ``profile`` is the FOURTH thing this path has had to be taught to ask for,
    and it is the same gap as lane 7's canvas one commit later: ``render_single``
    builds its own inputs, so anything a production request carries and this
    function does not invent is simply absent from every solo lane smoke.

    Here that is the BOOT CONTRACT. ``_render_one`` passes ``profile or {}``
    onward, and an empty profile SELECTS the ``default`` contract -- so a lane
    that legitimately requires its own boot (lane 19's ``minimax_h3_video``, the
    first such lane) could never be smoked on the boot it declares. When the
    caller names no profile and the engine declares exactly ONE compatible
    contract, that contract is selected here. An engine with several named
    contracts is left unset here; its adapter matches the live server against
    only its own compatible contracts before asserting the selected state.

    THIS IS NOT A BYPASS, and the distinction is the whole reason it is safe:
    selecting a contract is a CLAIM, and the claim is still proved against
    ``comfy.cli_args`` on the running server by ``assert_running_server``. A
    smoke on a wrongly-booted server still refuses by name -- it just refuses
    for the true reason ("this server has no reserve clamp") instead of the
    false one ("you asked for the stock boot"). An engine declaring two or more
    contracts is left alone: there the selection is a real choice and inventing
    one would be guessing. MiniMax H3 now exercises that path because its 16 GB
    streaming and physical-8-GB lab launches intentionally differ.
    """
    if profile is None:
        try:
            from .._otr_shared import boot_contracts as _bc
            declared = tuple(
                getattr(_vreg.get_engine(engine_name),
                        "compatible_boot_contracts", ()) or ())
            if len(declared) == 1 and declared[0] != _bc.DEFAULT:
                profile = {"launch": {"boot_contract": declared[0]}}
        except Exception:  # noqa: BLE001 -- unknown engine -> stock behaviour
            pass
    shot = {"shot_id": "single_0000", "beat_id": "b0000",
            "engine_id": engine_name,
            "family": engine_family(engine_name, "audio_driven_face"),
            "target_frame_count": int(frame_count), "degradation_trail": []}
    # build_request defaults to the HuMo PORTRAIT canvas (480x832). For a WIDE
    # engine (render_aspect='wide': ltx_video, wan_*, the _169 HuMos, ...) that
    # letterboxes a 16:9 init still into a tall frame ("postage stamp" with black
    # bars). With no explicit canvas, derive it from the engine's render_aspect so
    # the single-engine validation renders in the engine's NATIVE aspect: wide ->
    # 832x480 (the VRAM-safe proven render canvas, env OTR_VIDEO_RENDER_CANVAS),
    # else the portrait default.
    # A DECLARED RENDER CANVAS WINS HERE TOO (lane 7, 2026-08-11). This was the
    # THIRD channel deciding a canvas, and the only one `declared_render_canvas`
    # did not reach: `build_request_from_shot` applies the declaration last, but
    # `render_single` builds its own request and never asked. Every solo lane
    # smoke runs through here, so every lane's smoke was validating the ASPECT
    # DEFAULT rather than the declaration -- invisible for lanes 1-6 only
    # because all six declared exactly what this path already defaulted to
    # (832x480 wide, 480x832 portrait). The first lane to declare something else
    # failed its own /32 guard on a live render, which is the gate working.
    #
    # An explicit `canvas=` argument still wins, so a caller can deliberately
    # probe an off-declaration size; the declaration only replaces the DERIVED
    # default.
    if canvas is None:
        try:
            _eng = _vreg.get_engine(engine_name)
            _declared = declared_render_canvas(engine_name)
            if _declared is not None:
                canvas = tuple(_declared)
            elif getattr(_eng, "render_aspect", "portrait") == "wide":
                _rc = os.environ.get("OTR_VIDEO_RENDER_CANVAS", "832x480")
                try:
                    _rw, _rh = (int(x) for x in _rc.lower().split("x", 1))
                except (ValueError, AttributeError):
                    _rw, _rh = 832, 480
                canvas = (_rw, _rh)
        except Exception:  # noqa: BLE001 -- unknown engine -> portrait default
            pass
    request = build_request(shot, assets, frame_count, canvas)
    t0 = time.time()
    try:
        clip = _render_one(engine_name, request, force_oom=False,
                           profile=profile)
        return {"ok": True, "engine": engine_name,
                "elapsed_s": round(time.time() - t0, 1),
                "clip": _clip_summary(clip),
                "vram_used_mb": _mc.vram_used_mb()}
    except Exception as exc:             # noqa: BLE001 - report honestly
        import traceback
        return {"ok": False, "engine": engine_name,
                "elapsed_s": round(time.time() - t0, 1),
                "error": "%s: %s" % (type(exc).__name__, exc),
                "traceback": traceback.format_exc()[-1800:]}


__all__ = [
    "ENGINE_FAMILY",
    "OOM_ENGINES", "FROZEN_AUDIO_SHA",
    "OomSignal", "RenderError", "RenderFloorError", "SoakError",
    "FamilyInputGap",
    "classify_failure", "engine_family",
    "build_soak_fixture", "build_full_ledger", "build_request",
    "build_request_from_shot", "_slice_master_audio",
    "SLICER_VERSION", "slice_cache_key",
    "run_real_episode", "build_clip_manifest", "persist_episode_clips",
    "parse_engine_override", "apply_engine_override",
    "render_shot", "run_episode", "assemble_report", "assert_soak_ok",
    "run_gpu_soak", "render_single",
]
