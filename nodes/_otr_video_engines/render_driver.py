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
import os
import re
import subprocess
import tempfile
import time

from .._otr_shared import retry_taxonomy as _rt
from .._otr_shared.aspect import is_wide as _aspect_is_wide
from .._otr_speaker_role import (
    SPEAKER_ROLE_ANNOUNCER as _SPEAKER_ROLE_ANNOUNCER,
    SPEAKER_ROLE_MUSIC_OPEN as _SPEAKER_ROLE_MUSIC_OPEN,
    is_never_humo_role as _is_never_humo_role,
)
from . import motion_common as _mc
from . import registry as _vreg

_LOG = logging.getLogger("OTR.video.render_driver")

# NO FALLBACKS (operator directive 2026-07-02): FLOOR_NAMES / UNIVERSAL_FLOOR /
# SYNTH_FALLBACKS and the whole degrade-chain machinery were RIPPED (Sprint A,
# E1). Engine failure = LOUD RenderError; still_motion remains a registered
# SELECTABLE engine but has no floor role.

#: engine_id -> family (covers the soak stub + the A/cheap engines).
ENGINE_FAMILY = {
    "soak_oom_3d": "character_3d",       # synthetic soak stub (forced-OOM leg)
    "humo": "audio_driven_face",
    "humo_1.7B": "audio_driven_face",
    "still_motion": "static_motion",
    # still_parallax UNREGISTERED 2026-06-30 (item 2 rip-out) -- removed here too.
    "ltx_video": "text_to_video",
    "wan_i2v": "image_to_video", "mesh_stage": "image_to_video",
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
    ("music_visual", "ltx_video", "text_to_video"),
    ("character_video", "wan_i2v", "image_to_video"),
    ("character_video", "still_motion", "static_motion"),
    ("music_visual", "still_flat", "static_image_gen"),
    ("announcer_visual", "still_pan", "static_image_gen"),
)
#: The forced-OOM character_3d group: the synthetic ``soak_oom_3d`` stub (a
#: stand-in heavy character_3d engine). Under NO FALLBACKS its forced OOM
#: RAISES a named RenderError -- the soak asserts the raise (no trail).
_CHAR3D = ("character_video", "soak_oom_3d", "character_3d")
#: The heavy engines the soak forces to OOM on the character_3d shot -- the
#: LOUD-failure contract leg asserts the resulting RenderError.
OOM_ENGINES = frozenset({"soak_oom_3d", "humo", "humo_1.7B"})
#: The M1 frozen master-audio PCM marker the soak threads through + asserts is
#: byte-identical after the run (the decision layer must never touch audio).
FROZEN_AUDIO_SHA = "21aa71f6a4e5master_audio_pcm_marker"
_GOOGLE_SILENT_TEXT_PROVIDERS = frozenset({
    "google_veo_video",
    "google_omni_video",
})
_GOOGLE_VIDEO_SFX_ENGINES = frozenset({
    "google_vid_sfx_omni",
    "google_vid_sfx_veo_lite",
    "google_vid_sfx_veo_fast",
    "google_vid_sfx_veo_pro",
})
_GOOGLE_PROVIDER_PROMPT_ENGINES = (
    _GOOGLE_SILENT_TEXT_PROVIDERS | _GOOGLE_VIDEO_SFX_ENGINES)


class OomSignal(RuntimeError):
    """Stand-in for a render-time CUDA OOM (a HARD failure) -- the soak forces it
    on the mid-episode character_3d shot to prove the LOUD-raise contract."""


class RenderFloorError(RuntimeError):
    """A radio-open beat rendered on the procgen/still floor instead of an LTX
    engine (BUG-LOCAL-413 strict mode -- see :func:`check_ltx_open_health`).
    NOTE: despite the historical name, this has nothing to do with the deleted
    fallback-chain floor machinery."""


class RenderError(RuntimeError):
    """A shot's selected engine failed to render. Fallbacks are DISABLED
    (operator 2026-06-16, 'this is art, not a space shuttle'): a proven model
    path must prove itself, so this is terminal -- the episode fails LOUD instead
    of swapping engines or degrading to a still floor."""


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
    CLEAN all-profiles fixture; an integer injects the synthetic ``soak_oom_3d``
    character_3d stub at that index for the LOUD-failure contract leg (the
    forced OOM must RAISE a named RenderError -- no trail, no swap)."""
    if oom_index is not None and not 0 <= oom_index < n_beats:
        raise ValueError("oom_index %d out of range for %d beats"
                         % (oom_index, n_beats))
    shots = []
    for i in range(n_beats):
        role, engine, family = _CHAR3D if i == oom_index \
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
SLICER_VERSION = "2"

#: The HuMo slice recipe constants (UNCHANGED semantics -- 7.3: HuMo's
#: slicer is NOT changed; they are named so the cache key can bind them).
_SLICE_SAMPLE_RATE = 44100
_SLICE_CHANNELS = 1


def slice_cache_key(master_hash, start_s, dur_s, *,
                    sample_rate=_SLICE_SAMPLE_RATE,
                    channels=_SLICE_CHANNELS,
                    slicer_version=SLICER_VERSION,
                    master_path=""):
    """The SLICE cache key (3D plan 7.3): master CONTENT hash + timing +
    rate + channels + slicer version. ``master_path`` participates ONLY when
    ``master_hash`` is empty (the legacy hashless caller keeps the shipped
    path-keyed behavior instead of all colliding on one key). Pure; 16-hex."""
    ident = str(master_hash or "") or ("path:%s" % master_path)
    return hashlib.sha256(
        ("slice|v%s|%s|%.6f|%.6f|ar%d|ac%d"
         % (slicer_version, ident, float(start_s), float(dur_s),
            int(sample_rate), int(channels))).encode("utf-8")
    ).hexdigest()[:16]


def curve_cache_key(slice_key, line_id, *, fps, driver_version,
                    mapping_hash, onset_policy="onset_in_clip_v1"):
    """The CURVE cache key (3D plan 7.3): the W7 Rhubarb->ARKit curve file
    binds the SLICE key (content-true audio identity) + line_id + fps +
    driver version + viseme-mapping-table hash + onset policy. Changing any
    driver-side input regenerates curves WITHOUT touching the cheap WAV
    (the split's whole point). Pure; 16-hex."""
    return hashlib.sha256(
        ("curve|%s|line=%s|fps=%d|drv=%s|map=%s|onset=%s"
         % (slice_key, line_id, int(fps), driver_version, mapping_hash,
            onset_policy)).encode("utf-8")
    ).hexdigest()[:16]


def _slice_master_audio(master_path, start_s, dur_s, master_hash=""):
    """ffmpeg-slice ``[start_s, start_s+dur_s]`` from the FROZEN master into a
    temp WAV.  Read-only (``-i`` only, never ``-o`` on the master).  Returns
    the temp path on success, ``""`` on failure (LOUD warning logged).

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
                          master_path=master_path)
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
        "ffmpeg", "-y",
        "-ss", "%.6f" % float(start_s),
        "-t",  "%.6f" % float(dur_s),
        "-i",  master_path,
        "-vn", "-c:a", "pcm_s16le",
        "-ar", str(_SLICE_SAMPLE_RATE), "-ac", str(_SLICE_CHANNELS),
        out,
    ]
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
# portrait + the per-beat voice audio + the M4 creative prompt from the
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
_OPENING_MUSIC_SUFFIX = "b000_music_open"

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
_LTX_MOTION_PROMPT_BY_ROLE = {
    "announcer": ("Continuous shot, same console throughout. Tuning dial needle "
                  "sweeps rhythmically. Vacuum tubes pulse. Brass speaker grille "
                  "trembles. Dust motes drift. Slow handheld dolly forward."),
    "music_open": ("Continuous shot, same console throughout. Dial whip-pans "
                   "across frequencies. Tube filaments ignite from cold to "
                   "white-hot. Speaker grille vibrates aggressively. Dynamic "
                   "dolly push forward."),
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
#: the 900-char M4 identity/scene wall drowned the speech tokens). Mirrors
#: the canonical token pattern ("talking to the viewer ... opening and
#: closing naturally in sync with the speech").
_IA2V_TALKING_CLAUSE_CHARACTER = (
    "is talking to the viewer, lips opening and closing naturally in sync "
    "with the speech, subtle head and hand gestures. Static camera.")


def _compact_style_talking_cue(vstyle):
    """Compact pack cue for IA2V character prompts.

    The talking register is deliberately tiny for lip-sync; long style tails
    drown the speech tokens. Non-default packs still need one blunt cue close
    to the front of the prompt so the video model does not drift back to a
    generic cinematic portrait.
    """
    if str(getattr(vstyle, "style_id", "") or "") == "sci_fi_radio":
        return ""
    raw = str(getattr(vstyle, "positive_tail", "") or "").strip()
    if not raw:
        raw = str(getattr(vstyle, "portrait_look_talking", "") or "").strip()
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9-]*", raw)
    lowered = [word.lower() for word in words]
    if lowered[:4] == ["recursive", "fractal", "light", "field"]:
        return " ".join(words[:4]).strip()
    limit = 2
    for i, word in enumerate(words[:4]):
        if word.lower() == "style":
            limit = i + 1
            break
    return " ".join(words[:limit]).strip()


def _prefix_video_style_cue(vstyle, prompt):
    """Put the selected non-default visual style at the front of video text.

    This is intentionally a short cue, not a full style tail. Long tails dilute
    LTX/IA2V talking prompts and can crowd out identity/motion tokens.
    """
    prompt = str(prompt or "").strip()
    cue = _compact_style_talking_cue(vstyle)
    if not prompt or not cue:
        return prompt
    low_prompt = prompt.lower()
    low_cue = cue.lower().rstrip(".")
    if low_prompt.startswith(low_cue):
        return prompt
    if low_cue.endswith(" style"):
        base_cue = low_cue[: -len(" style")].strip()
        if base_cue and low_prompt.startswith(base_cue):
            rest = prompt[len(base_cue):].lstrip()
            if rest.startswith("."):
                rest = rest[1:].lstrip()
            return ("%s. %s" % (cue.rstrip("."), rest)
                    if rest else "%s." % cue.rstrip("."))
    return "%s. %s" % (cue.rstrip("."), prompt)


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


def _motion_clause_override(shot):
    """Opt-in per-beat motion clause text, or ``None`` (default OFF -> byte-identical).
    See nodes/_otr_motion_clause + docs/2026-06-16-ltx-motion/MOTION_CLAUSE_SPEC.md."""
    try:
        from .._otr_motion_clause import (  # type: ignore
            enabled as _mc_enabled, resolve_motion_clause_text as _mc_text)
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_motion_clause import (  # type: ignore
            enabled as _mc_enabled, resolve_motion_clause_text as _mc_text)
    if not _mc_enabled():
        return None
    return _mc_text(shot)


#: HuMo-seam ticket Part C (2026-06-11). Broadcast-gear scrub for CHARACTER
#: face beats -- LOCAL mirror of nodes/otr_meta_brief_image_prompt._GEAR_WORDS
#: (the node module registers classes at import; mirroring follows the
#: _OPENING_MUSIC_SUFFIX local-constant pattern -- keep the two regexes in
#: LOCKSTEP; tests/test_brief_prompt_finishing.py pins the parity). NO
#: negations anywhere: negative phrasing PLANTS the tokens (the c01 giant-mic
#: lesson) -- gear is scrubbed from the OUTPUT, never "no microphone"-ed.
_GEAR_WORDS_RD = re.compile(
    r"\s*\b(?:radios?|microphones?|mics?|broadcasts?|broadcasters?|"
    r"broadcasting|recording\s+studios?|radio\s+(?:station|studio|set|"
    r"booth)s?|studios?|on[- ]air(?:\s+sign)?)\b[,;]?",
    re.IGNORECASE)

#: Gear-free fallback prompt for a CHARACTER face beat whose shot carries no
#: M4 creative prompt (the proven microphone re-introduction path). Keeps the
#: face anchored for the audio_driven_face family; zero broadcast tokens.
_CHAR_FACE_FALLBACK_PROMPT = (
    "close-up cinematic portrait of a person speaking, face centered, subtle "
    "facial motion, period 1940s costume, dramatic film lighting")


def _scrub_gear(prompt: str) -> str:
    """Remove broadcast-gear tokens from a character prompt, tidying the
    leftover separators. Pure; '' stays ''. Mirrors
    otr_meta_brief_image_prompt._scrub_gear_words (lockstep)."""
    out = _GEAR_WORDS_RD.sub("", prompt or "")
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"(,\s*)+,", ", ", out)
    out = re.sub(r"\s+,", ",", out)
    return out.strip(" ,;").strip()

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
    (round 5 F2): source enum (m4|env|brief+beat), sha8, char count --
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
    if engine_id in _GOOGLE_VIDEO_SFX_ENGINES:
        return
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
    if str((shot or {}).get("engine_id") or "") == "ltx_audio_in" and role == "character_video":
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
                            master_audio_path=""):
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
    line = _line_index(ledger).get(_beat_id_for_shot(shot), {})
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
        _bid = _beat_id_for_shot(shot)
        _fidx = _mesh_fodder_index(ledger)
        _subj = char_id or str(shot.get("mesh_subject_id") or "")
        # SUBJECT POLICY (chunk 6): (a) a beat with a char_id -- INCLUDING the
        # announcer (char_id "announcer") -- meshes that character; (b) a beat
        # with no char (e.g. the b000 music-open) meshes a per-beat story OBJECT
        # under a STABLE "obj_<beat>" id that MATCHES the minted mesh_fodder
        # object's mesh_subject_id; (c) if no fodder was minted at all, the
        # missing-fodder branch below degrades LOUD (never meshes the
        # environment as "uncast"). Mirror the prompt-gen id convention exactly.
        _mesh_subject_id = _subj or ("obj_%s" % _bid)
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
    _engine_scene_init_required = (
        "init_image" in _required_inputs_for_engine(_eng_id, _family)
        and _family not in ("audio_driven_face", "character_3d")
        and _eng_id != "ltx_audio_in"
    )
    _engine_scene_init_optional = bool(
        _eng_id and _vreg.is_registered(_eng_id)
        and getattr(_vreg.get_engine(_eng_id), "provider_side", False)
        and getattr(_vreg.get_engine(_eng_id), "accepts_still", False)
        and _family not in ("audio_driven_face", "character_3d")
        and _eng_id not in ("ltx_audio_in", "still_pan", "still_flat", "still_word")
    )
    if ((_family in _SCENE_INIT_FAMILIES
            or _engine_scene_init_required
            or _engine_scene_init_optional)
            and not _requires_fodder):
        # NB: mesh_stage is family=image_to_video (IN _SCENE_INIT_FAMILIES), so
        # without the not-_requires_fodder guard this override would clobber the
        # clean fodder with the scene still and re-introduce the clay blob.
        _bid = _beat_id_for_shot(shot)
        # (still_pool_key read removed 2026-07-01 with the pooling rip --
        # every shot resolves its own per-beat still.)
        _still = _still_index(ledger).get(str(_bid), "")
        if _still:
            init_image = _still
            init_source = "scene_still"
        else:
            raise ValueError(
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
    # Unlike still_pan/still_flat it has NO floor -- a missing still fails LOUD in
    # its render_clip (_require_still), never a silent black card (no fallbacks).
    if str(shot.get("engine_id") or "") in ("still_pan", "still_flat", "still_word", "ltx_audio_in"):
        _eng = str(shot.get("engine_id") or "")
        _bid = _beat_id_for_shot(shot)
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
                raise RenderError(
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
            _LOG.warning(
                "[OTR.render_driver] %s MISSING-STILL (LOUD): beat %s "
                "has NO scene still in the ledger -- a cheap family (still_pan/still_flat) "
                "synthesizes its dark floor; a still-REQUIRED engine (still_word/"
                "ltx_audio_in) fails LOUD in render_clip (no fallbacks). Investigate "
                "the image phase for beat %s.", _eng, _bid, _bid)
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
            raise RenderError(
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
    # LTX-I2V ticket Part B (2026-06-11) -- DEFAULT ON since LK-1a (the
    # look restoration): every ltx_video shot conditions on the beat's
    # ST-3-minted scene still (init_source=scene_still in the trace) --
    # the music open b000 included (its text-only render was the murk
    # cause). A missing still falls back LOUD to the round-5 text path --
    # never silent. Set OTR_ENABLE_LTX_I2V=0 to restore text-only LTX.
    _i2v_still_missing = False
    if (str(shot.get("engine_id") or "") == "ltx_video"
            and os.environ.get("OTR_ENABLE_LTX_I2V", "1") == "1"):
        _bid = _beat_id_for_shot(shot)
        _still = _still_index(ledger).get(str(_bid), "")
        if _still:
            init_image = _still
            init_source = "scene_still"
            _LOG.warning(
                "[OTR.render_driver] LTX-I2V: beat %s conditioning on scene "
                "still %s (default since LK-1a)", _bid,
                os.path.basename(_still))
        else:
            # Operator 2026-06-12: every i2v beat MUST have its still. The
            # derive_scene_still_targets coverage fix should make this
            # unreachable; if a still is STILL absent the degrade must be LOUD
            # IN THE TRACE (stamped below), never a silent text-only render.
            _i2v_still_missing = True
            _LOG.warning(
                "[OTR.render_driver] LTX-I2V MISSING-STILL (LOUD): i2v is "
                "enabled but beat %s has NO scene still in the ledger -- "
                "rendering text-only and STAMPING the trace as a degrade "
                "(init_source=missing_scene_still). This should not happen "
                "after the still-spine coverage fix; investigate the image "
                "phase for beat %s.", _bid, _bid)
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
            raise RenderError(
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
            # 7.3 slice key: the master CONTENT hash is the cache identity
            # (a new master at the same path invalidates the slice).
            _mhash = str(((ledger or {}).get("audio") or {})
                         .get("master_audio_sha256") or "")
            sliced = _slice_master_audio(master_audio_path,
                                         float(start_s), float(dur_s),
                                         master_hash=_mhash)
            if sliced:
                _LOG.info("[OTR.render_driver] per-beat audio: sliced "
                          "%s @%.3f+%.3fs -> %s (beat %s)",
                          os.path.basename(master_audio_path),
                          float(start_s), float(dur_s),
                          os.path.basename(sliced), bid)
                audio = sliced
            else:
                _LOG.warning("[OTR.render_driver] per-beat audio slice FAILED "
                             "for beat %s -- HuMo will degrade LOUD", bid)
        else:
            _LOG.warning("[OTR.render_driver] per-beat audio: beat %s has no "
                         "start_s/dur_s on line -- HuMo will degrade LOUD", bid)
    frame_count = int(shot.get("target_frame_count") or 0)
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
    if _i2v_still_missing:
        # LOUD trace degrade: an i2v beat that rendered with no scene still.
        req["observability"]["init_source"] = "missing_scene_still"
        req["observability"]["i2v_still_missing"] = True
    elif _mesh_fodder_missing:
        # LOUD trace degrade: a 3D mesh beat with no clean fodder minted -- the
        # default "none" stamp (init_image is empty) would HIDE that we refused
        # to mesh the scene still; surface it explicitly (mirrors i2v above).
        req["observability"]["init_source"] = "missing_mesh_fodder"
        req["observability"]["mesh_fodder_missing"] = True
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
    _face_excl = {"audio_driven_face", "lipsync_overlay", "character_3d"}
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
    # LTX-AV (audio-input) lane (M3 delta a): render at the M0-PROVEN-SAFE small
    # native canvas (the 22B A2V model would blow the budget at the 480x832
    # portrait (talk) / 1472x832 landscape (music) defaults set above), then let
    # OTR_SilentComposite scale the clip to the deliverable. Diverges from
    # ltx_video's 832x480 ON PURPOSE (heavier 22B). Env OTR_LTX_AV_RENDER_CANVAS
    # (default 512x288; /32-friendly). The MEASURED per-recipe VRAM peaks (the
    # fit lever is recipe/quant, NOT resolution -- see eng_ltx_av._quant_label +
    # the per-beat PLAN log line + the bakeoff manifest), NOT a hardcoded MB here.
    if str(shot.get("engine_id") or "") == "ltx_audio_in":
        # ia2v_canonical canvas (lips-dont-talk kibitz + operator quality
        # catch 2026-07-02): the 512x288 default was single-pass VRAM math
        # and upscaled ~2.9x to the deliverable = the "really low quality"
        # the operator flagged. Ladder walked LIVE: 1280x720 fails the /32
        # grid gate (proof5b); 1280x704 BREACHED the 14.5GB ceiling in the
        # full production pipeline (proof6: 14716 MB -- the isolation probes
        # carried less resident state). 832x480 = 2.6x the old pixels, a
        # 1.77x deliverable upscale, base 416x240 (all /32), and P1 proved
        # articulation with a sharp guide at even smaller bases. Single-pass
        # recipes keep the proven 512x288. Env-overridable either way.
        _av_default = ("832x480"
                       if _ia2v_talking_register_active("ltx_audio_in")
                       else "512x288")
        _avc = os.environ.get("OTR_LTX_AV_RENDER_CANVAS", _av_default)
        try:
            _avw, _avh = (int(x) for x in _avc.lower().split("x", 1))
        except (ValueError, AttributeError):
            _avw, _avh = 512, 288
        req["canvas"]["w"], req["canvas"]["h"] = _avw, _avh
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
    _fam = engine_family(str(shot.get("engine_id") or ""), "")
    # ia2v TALKING register decision, computed ONCE per shot (kibitz r2: the
    # helper instantiates an engine to ask the recipe -- cheap today, but
    # three call sites x every shot invites a regression if the ctor ever
    # grows weight; memoize the bool here and reuse it below).
    _talking_register = _ia2v_talking_register_active(shot.get("engine_id"))
    # ia2v TALKING register, radio-bookend precedence: an explicitly audio-driven
    # announcer or music bookend that happens to carry an M4 creative prompt would
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
    # CHARACTER beats (audio_conditioned_video family), so the gear scrub + the
    # char-fallback prompt + the scene-prompt EXCLUSION all apply to them exactly
    # like an audio_driven_face talking head. announcer/music bookends stay scene.
    _is_char_face_beat = _is_character_face_beat(shot)
    if text_prompt:
        # HuMo-seam ticket Part C (2026-06-11): CHARACTER face beats get the
        # gear scrub (the c01 lesson: scrub the OUTPUT, never add "no
        # microphone" -- negations PLANT the tokens). ANNOUNCER beats are
        # exempt (radio-styled BY DESIGN).
        if _is_char_face_beat:
            _scrubbed = _scrub_gear(text_prompt)
            if _scrubbed and _scrubbed != text_prompt:
                _LOG.warning(
                    "[OTR.render_driver] HuMo character beat %s: broadcast-"
                    "gear tokens scrubbed from the M4 prompt (announcer "
                    "stays radio-styled)", _beat_id_for_shot(shot))
                text_prompt = _scrubbed
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
            # fragment = the M4's first sentence; else last full CLAUSE under
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
            _LOG.warning(
                "[OTR.render_driver] IA2V TALKING register: character beat "
                "%s M4 wall -> compact talking prompt (%d chars, style_cue=%r)",
                _beat_id_for_shot(shot), len(text_prompt), _style_cue)
        elif (str(shot.get("engine_id") or "").startswith("ltx")
                and _shot_role not in ("announcer_visual", "music_visual")):
            text_prompt = (text_prompt.rstrip().rstrip(",")
                           + ", stable centered subject, full face clearly "
                           "visible, generous headroom, comfortably composed")
        text_prompt = _prefix_video_style_cue(_vstyle, text_prompt)
        req["text_prompt"] = text_prompt
        _stamp_prompt_meta(req, "m4", text_prompt,
                           subsource=str(creative.get("source") or ""),
                           beat=_beat_id_for_shot(shot))
    elif _is_char_face_beat or _fam == "audio_driven_face":
        # HuMo-seam ticket Part C: a FACE beat with NO M4 creative prompt is
        # the proven microphone re-introduction path -- the build_request
        # studio default ("a 1940s radio studio... on air") re-dresses the
        # gear the FLUX portraits were scrubbed of. LOUD + a gear-free
        # fallback for character beats; the announcer keeps the studio
        # default (radio-styled by design). Never silent. (2026-06-26: the
        # _is_char_face_beat arm also routes ltx_audio_in CHARACTER beats here
        # so they get the gear-free char fallback, not the generic radio default.)
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
            _stamp_prompt_meta(req, "default_scrubbed", _cf_fallback,
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
    # default for text engines. Precedence: M4 creative prompt (finished at
    # ShotLock) > OTR_LTX_RADIO_PROMPT (operator override, VERBATIM, no
    # finishing) > brief-composed + finished. (_shot_role parsed above, once.)
    _engine_id = str(shot.get("engine_id") or "")
    _strict_text_only = _is_strict_text_only_engine(_engine_id)
    _google_text_provider = _engine_id in _GOOGLE_SILENT_TEXT_PROVIDERS
    _google_prompt_provider = _engine_id in _GOOGLE_PROVIDER_PROMPT_ENGINES
    if ((_engine_id in ("ltx_video", "wan_i2v", "ltx_audio_in")
            or _google_prompt_provider
            or _strict_text_only)
            and not text_prompt and not _is_char_face_beat):
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
                         "(verbatim, unfinished)",
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
            # kibitz r3 must-fix: the opt-in motion-clause override must NOT
            # clobber the lip-sync register (same guard as the atmosphere
            # append) -- an _otr_motion_clause entry on an announcer bookend
            # would silently revert it to a non-talking prompt.
            _mc_override = _motion_clause_override(shot)
            if _mc_override and not _talking_swap:
                scene_prompt = _mc_override
                _field_source = "python:motion_clause_override"
            elif _mc_override and _talking_swap:
                _LOG.warning(
                    "[OTR.render_driver] IA2V TALKING register: motion-clause "
                    "override on %s bookend %s IGNORED (lip-sync register "
                    "outranks it)", _shot_role, shot.get("shot_id"))
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
                _field_source = (
                    "open_subjects:%s+motion_registers:%s" %
                    (("synthetic" if _is_synthetic_open
                      else ("announcer" if _shot_role == "announcer_visual"
                            else "default")), _motion_key))
            scene_prompt = _prefix_video_style_cue(_vstyle, scene_prompt)
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
            core = get_story_brief_ltx(_meta)
            if not core:
                _setting = _term_join("setting", 2)
                core = ("cinematic establishing shot"
                        + (f", {_setting}" if _setting else ""))
            _mc_override = _motion_clause_override(shot)
            if _mc_override:
                clauses = [_mc_override]
            else:
                clauses = list(_beat_clauses(line, shot.get("shot_id")))
                clauses.extend(["slow cinematic camera drift", "no on-screen text"])
            scene_prompt = finish_visual_prompt(
                _meta, f"{core}, {', '.join(clauses)}",
                max_chars=188, style_tail=False, era_profile="still")
            scene_prompt = _prefix_video_style_cue(_vstyle, scene_prompt)
            _LOG.warning("[OTR.render_driver] LTX SCENE: %s beat %s prompt "
                         "composed from the episode brief (%d chars): "
                         "%.90s...", _shot_role, shot.get("shot_id"),
                         len(scene_prompt), scene_prompt)
            req["text_prompt"] = scene_prompt
            _stamp_prompt_meta(req, "brief+beat", scene_prompt,
                               beat=_beat_id_for_shot(shot))
    _apply_visual_safety_prompt(req, shot)
    req_hash = (shot.get("render_request_hash")
                or (shot.get("cache_keys") or {}).get("request_hash"))
    _video_seed = _seed_from_hash(req_hash, shot.get("shot_id"))
    req["seed_bundle"] = {"request_seed": _video_seed}
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
                profile=None):
    """Attempt ONE candidate engine: assert_usable -> prepare -> render_clip ->
    canonicalize, always teardown. ``force_oom`` raises BEFORE any work (the
    soak's deterministic mid-episode OOM -- it precedes the family-input check
    so the soak's expected OOM trail is exactly preserved). Raises on failure;
    returns the canonical clip dict on success.

    S4 platform-portability (2026-07-10): ``host_caps``/``profile`` carry
    REAL host facts (build_host_caps) + the ledger-stamped v2 device/dtype
    policy down to the adapter boundary -- the empty-dict boundary here was
    the campaign's most consequential wiring catch. ``None`` (direct debug
    callers) degrades to the old empty dicts."""
    if force_oom:
        raise OomSignal("forced soak OOM on %s" % engine_name)
    _assert_family_inputs_satisfiable(engine_name, request)
    if not _vreg.is_registered(engine_name):
        raise LookupError("engine %r is not registered" % engine_name)
    eng = _vreg.get_engine(engine_name)
    _caps = host_caps if host_caps is not None else {}
    _prof = profile if profile is not None else {}
    prepared = None
    try:
        # M3 delta (d): pass the request as the assert_usable template so an
        # engine that validates request-shaped inputs (the LTX-AV av_dims gate
        # on request.canvas) sees the real canvas. Guarded -- a legacy adapter
        # whose assert_usable predates the request_template kwarg keeps working
        # (TypeError -> retry without it); real usability rejections raise
        # EngineUnusable, never TypeError, so this never masks one.
        try:
            eng.assert_usable(host_caps=_caps, profile=_prof,
                              request_template=request)
        except TypeError:
            eng.assert_usable(host_caps=_caps, profile=_prof)
        prepared = eng.prepare(host_caps=_caps, profile=_prof, session_ctx={})
        raw = eng.render_clip(request, prepared)
        return eng.canonicalize(raw, request, _prof)
    finally:
        if prepared is not None:
            try:
                eng.teardown(prepared)
            except Exception:            # noqa: BLE001 - teardown best-effort
                pass


def render_shot(shot, request, *, oom_engines=frozenset(), oom_shot_id=None,
                host_caps=None, profile=None):
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
                           host_caps=host_caps, profile=profile)
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
    _episode_profile = {
        "policy_version": 2,
        "device_policy": str(section.get("device_policy") or "cuda"),
        "dtype_policy": str(section.get("dtype_policy") or "fp8_ok"),
    }
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
    for shot in section["shots"]:
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
        clip, out_shot, attempts, used = render_shot(
            shot, request, oom_engines=oom_engines, oom_shot_id=oom_shot_id,
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
                    "visual_style", "prompt_field_source"):
            if key in obs:
                row[key] = obs[key]
            elif isinstance(request, dict) and ("_" + key) in request:
                row[key] = request["_" + key]
        trace.append(row)
        if used:
            vram_peak = max(vram_peak, int(used))
        # CS-3: remember what ACTUALLY rendered (post-fallback final_engine) so
        # the next beat reclaims only when it crosses to a different engine.
        _last_engine = str(out_shot.get("engine_id") or "")
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
    ledger = apply_engine_override(ledger)
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
    from .._otr_shared.public_engines import resolve_engine_id
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
        if engine not in ENGINE_FAMILY and not _vreg.is_registered(engine):
            raise ValueError(
                "OTR_FORCE_ENGINE_MAP names unknown engine %r" % engine)
        out[role] = engine
    return out


def apply_engine_override(ledger):
    """Experiment knob (operator ask 2026-06-09: the all-LTX / forced-engine
    episodes): when ``OTR_FORCE_ENGINE_MAP`` is set, re-route each planned
    shot's ``engine_id``/``family`` by role BEFORE rendering, LOUDLY. NO
    FALLBACKS (2026-07-02): a forced engine that fails raises RenderError
    LOUD -- there is no degrade to a floor. Returns the (possibly
    rewritten) ledger; a parse error logs LOUD and leaves the plan untouched
    (fail-safe: the production plan renders rather than aborting)."""
    spec = os.environ.get("OTR_FORCE_ENGINE_MAP", "").strip()
    if not spec:
        return ledger
    try:
        mapping = parse_engine_override(spec)
    except ValueError as exc:
        _LOG.warning("[OTR video] OTR_FORCE_ENGINE_MAP IGNORED (parse): %s", exc)
        return ledger
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
#: the prompt-only ltx_video + the additive LTX-AV audio lanes.
_LTX_OPEN_ENGINES = frozenset(
    {"ltx_video", "ltx_audio_in"})
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
    """Rename-proof the episode id used for durable clip/SFX persistence.

    VideoRenderBatch receives the ShotLock ledger while the episode is still
    named ``pending_<ts>``. SignalLostVideo may rename that episode directory to
    the title slug before clips are rendered, so writing to
    ``episodes/pending_<ts>/clips`` can re-create a stale folder and strand the
    provider clips/SFX away from the real episode. Mirror the stills/master-audio
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
                "id (dir renamed before clip persistence); clips/SFX re-keyed "
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
    moved_sfx = 0
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
        sfx_src = str(clip.get("sfx_stem_path") or "")
        if not sfx_src:
            continue
        if not os.path.isfile(sfx_src):
            _LOG.warning(
                "[OTR video] persist_episode_clips: SFX stem missing before "
                "persist %s -- clearing dangling path", sfx_src)
            clip["sfx_stem_path"] = ""
            continue
        sfx_dst = os.path.join(dest_dir, safe_stem + ".sfx.wav")
        if os.path.dirname(os.path.abspath(sfx_src)) == os.path.abspath(dest_dir):
            continue
        try:
            if os.path.abspath(sfx_src) != os.path.abspath(sfx_dst):
                shutil.move(sfx_src, sfx_dst)
            clip["sfx_stem_path"] = sfx_dst
            moved_sfx += 1
        except Exception as exc:         # noqa: BLE001 -- clear swept tmp paths
            _LOG.warning(
                "[OTR video] persist_episode_clips: SFX move FAILED %s -> "
                "%s (%s) -- clearing dangling path", sfx_src, sfx_dst, exc)
            clip["sfx_stem_path"] = ""
    if moved or moved_sfx:
        _LOG.info("[OTR video] persisted %d clip(s) and %d SFX stem(s) to "
                  "episodes/%s/clips/", moved, moved_sfx, episode_id)
    return result


def build_clip_manifest(result, *, episode_id=""):
    """Pure, beat-ordered per-beat clip manifest from a :func:`run_real_episode`
    result -- the STRING contract OTR_SilentComposite assembles. Shot order is
    the OUTPUT ledger's shots (already in beat order). Each row carries the clip
    path + the audio-derived frame counts; ``engine_histogram`` counts the
    on-disk clips per engine so the keystone can assert HuMo ran on the talking
    beats and the episode is NOT all-procgen. The frozen audio is never read."""
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
        sfx_stem = str(clip.get("sfx_stem_path") or "")
        if sfx_stem:
            row["sfx_stem_path"] = sfx_stem
            row["sfx_duration_s"] = clip.get("sfx_duration_s")
            row["sfx_sha256"] = clip.get("sfx_sha256")
        rows.append(row)
        if exists:
            hist[eid] = hist.get(eid, 0) + 1
    manifest = {
        "episode_id": str(episode_id or ""),
        "video_revision": int(section.get("video_revision") or 1),
        "fps": int(section.get("fps") or 25),
        "canvas": {"w": int(canvas.get("w") or 0), "h": int(canvas.get("h") or 0)},
        "n_beats": len(rows),
        "clip_count": sum(1 for r in rows if r["exists"]),
        "total_target_frames": total,
        "engine_histogram": hist,
        "clips": rows,
    }
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
            "path": path, "exists": exists, "size": size}


def _episode_facts(ep, meta):
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
        "episode_1": _episode_facts(e1, meta),
        "episode_2": _episode_facts(e2, meta),
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
    # LOUD-failure contract leg: a forced OOM on the synthetic character_3d
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
                  canvas=None):
    """Render ONE shot via a SINGLE engine with NO fallback -- the focused
    in-process validation (surfaces the real exception so the in-process forward
    can be debugged in isolation before the full soak). Returns a result dict."""
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
    if canvas is None:
        try:
            _eng = _vreg.get_engine(engine_name)
            if getattr(_eng, "render_aspect", "portrait") == "wide":
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
        clip = _render_one(engine_name, request, force_oom=False)
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
    "SLICER_VERSION", "slice_cache_key", "curve_cache_key",
    "run_real_episode", "build_clip_manifest", "persist_episode_clips",
    "parse_engine_override", "apply_engine_override",
    "render_shot", "run_episode", "assemble_report", "assert_soak_ok",
    "run_gpu_soak", "render_single",
]
