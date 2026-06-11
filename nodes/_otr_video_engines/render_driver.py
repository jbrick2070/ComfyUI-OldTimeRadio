"""In-process video render driver (A-S7.5) -- the model-agnostic render loop
that ``OTR_VideoRenderBatch`` walks to ship Subproject A.

For each shot the driver drives the registry engine lifecycle
(``assert_usable -> prepare -> render_clip -> canonicalize -> teardown``) and, on
a HARD failure, classifies it via the A-S7 retry taxonomy, walks the declared
``fallback_engine`` chain (resolved by :mod:`nodes._otr_shared.fallback`), and
restamps the ledger LOUDLY (log the swap + append a ``runtime_fallback_decisions``
record at the SAME ``video_revision``) until a clip renders. Every chain is
guaranteed to terminate at the registered radio floor (``still_kenburns``), so an
episode NEVER aborts and a beat is NEVER dropped. The frozen ``ledger['audio']``
section is read-only throughout (V-1 / the audio spine is frozen).

In-process (V invariant: no HTTP server, no GraphBuilder): the heavy engines call
ComfyUI wrapper node classes directly via :mod:`wrapper_bridge`, so this driver
MUST run inside the ComfyUI process (``NODE_CLASS_MAPPINGS`` populated). The pure
pieces (``make_fallback_of`` / ``classify_failure`` / the fixture builder /
``assert_soak_ok``) are CPU-tested; the live render + the A-S7.5 GPU soak are the
operator gate. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import logging
import os
import subprocess
import tempfile
import time

from .._otr_shared import retry_taxonomy as _rt
from .._otr_shared.fallback import resolve_fallback_chain
from . import motion_common as _mc
from . import registry as _vreg

_LOG = logging.getLogger("OTR.video.render_driver")

#: The cheap radio-floor engine names (terminal; a chain ends here).
FLOOR_NAMES = frozenset({"still_kenburns", "abstract", "station_card",
                         "visualizer", "flux_still"})
#: The universal floor terminus appended to any engine whose declared chain
#: would otherwise dangle (survival-guide BUG 12.23: no dangling fallback_engine
#: -- every chain terminates at a registered radio floor that always renders).
UNIVERSAL_FLOOR = "still_kenburns"

#: Subproject-B character_3d engine -> its A-side audio_driven_face fallback.
#: hunyuan3d_talk ships with B (not registered yet), so the hop is overlaid here
#: (mirrors scripts/otr_video_soak.make_fallback_of) to resolve the chain now.
SYNTH_FALLBACKS = {"hunyuan3d_talk": "humo"}

#: engine_id -> family, for restamping onto a fallback engine (covers the
#: not-yet-registered B engine + the A/cheap engines).
ENGINE_FAMILY = {
    "hunyuan3d_talk": "character_3d", "humo": "audio_driven_face",
    "humo_1.7B": "audio_driven_face",
    "latentsync": "lipsync_overlay", "still_kenburns": "static_motion",
    "ltx_video": "text_to_video", "wan_i2v": "image_to_video",
    "abstract": "abstract", "station_card": "static_image_gen",
    "visualizer": "abstract", "flux_still": "static_image_gen",
}

#: The (role, engine, family) rotation covering all 5 roles + the 7 non-3D
#: families (kept identical to scripts/otr_video_soak so the GPU soak walks the
#: same shape the shipped CPU harness proves).
_PROFILES = (
    ("announcer_visual", "humo", "audio_driven_face"),
    ("announcer_visual", "latentsync", "lipsync_overlay"),
    ("music_visual", "ltx_video", "text_to_video"),
    ("character_video", "wan_i2v", "image_to_video"),
    ("scene_broll", "still_kenburns", "static_motion"),
    ("background_abstract", "abstract", "abstract"),
    ("announcer_visual", "station_card", "static_image_gen"),
)
#: The forced-OOM character_3d group (Subproject B family; degrades to humo).
_CHAR3D = ("character_video", "hunyuan3d_talk", "character_3d")
#: The heavy engines the soak forces to OOM on the character_3d shot so the chain
#: walks all the way to the radio floor.
OOM_ENGINES = frozenset({"hunyuan3d_talk", "humo", "humo_1.7B", "latentsync"})
#: The M1 frozen master-audio PCM marker the soak threads through + asserts is
#: byte-identical after the run (the decision layer must never touch audio).
FROZEN_AUDIO_SHA = "21aa71f6a4e5master_audio_pcm_marker"
#: The expected character_3d degradation trail to the radio floor.
EXPECTED_OOM_TRAIL = ["hunyuan3d_talk->humo (oom)", "humo->humo_1.7B (oom)",
                      "humo_1.7B->latentsync (oom)",
                      "latentsync->still_kenburns (oom)"]


class OomSignal(RuntimeError):
    """Stand-in for a render-time CUDA OOM (a HARD failure) -- the soak forces it
    on the mid-episode character_3d shot to walk the chain to the floor."""


class RenderFloorError(RuntimeError):
    """The radio floor itself failed to render -- a chain genuinely exhausted
    (the soak's negative control; should never happen with a working ffmpeg)."""


class SoakError(AssertionError):
    """An A-S7.5 soak invariant was violated (the soak FAILED)."""


# --------------------------------------------------------------------------- #
# Pure helpers (CPU-tested)
# --------------------------------------------------------------------------- #
def make_fallback_of(synth=None):
    """``fallback_of(name) -> next | None`` over the REAL registry + the synthetic
    B overlay, guaranteeing termination at the radio floor (a dangling engine
    with no declared fallback degrades to ``still_kenburns``)."""
    overlay = dict(SYNTH_FALLBACKS)
    if synth:
        overlay.update(synth)

    def fallback_of(name):
        if name in overlay:
            return overlay[name]
        if _vreg.is_registered(name):
            nxt = getattr(_vreg.get_engine(name), "fallback_engine", None)
            if nxt:
                return nxt
        if name in FLOOR_NAMES:
            return None
        return UNIVERSAL_FLOOR

    return fallback_of


def classify_failure(exc):
    """Map a render exception to a HARD :class:`FailureKind` (all escalate)."""
    if isinstance(exc, OomSignal):
        return _rt.FailureKind.OOM
    name = type(exc).__name__
    if name in ("EngineUnusable", "WrapperNodeMissing", "LookupError",
                "KeyError", "FileNotFoundError"):
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


def build_soak_fixture(n_beats=40, oom_index=20):
    """Build a synthetic ``ledger['video']`` section + meta (pure; identical
    shape to scripts/otr_video_soak.build_soak_fixture)."""
    if not 0 <= oom_index < n_beats:
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
    """A VideoRequest-shaped dict per shot (deterministic: the seed is keyed to
    the shot id so render-twice is identical -- V-7)."""
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
    return {
        "shot_id": sid, "request_id": sid,
        "text_prompt": "a 1940s radio studio, warm tungsten light, on air",
        "asset_refs": {"init_image": portrait} if portrait else {},
        "audio_ref": {"path": audio} if audio else None,
        "base_clip_ref": None,
        "timing": {"target_frame_count": int(frame_count)},
        "canvas": {"w": int(cw), "h": int(ch), "fps": 25, "aspect_policy": "pad"},
        "seed_bundle": {"request_seed": seed},
        "init_w": 480, "init_h": 832,
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
# the master is NEVER mutated (V-1 / audio spine frozen).  The slice is
# deterministically keyed to the (path, start_s, dur_s) triple so a second
# render of the same beat reuses the cached file without re-running ffmpeg.
# --------------------------------------------------------------------------- #

def _slice_master_audio(master_path, start_s, dur_s):
    """ffmpeg-slice ``[start_s, start_s+dur_s]`` from the FROZEN master into a
    temp WAV.  Read-only (``-i`` only, never ``-o`` on the master).  Returns
    the temp path on success, ``""`` on failure (LOUD warning logged).

    The output file is cached by a hash of ``(start_s, dur_s, master_path)``
    so render-twice is deterministic without re-running ffmpeg."""
    key = hashlib.sha256(
        ("%.6f|%.6f|%s" % (float(start_s), float(dur_s), master_path)
         ).encode("utf-8")
    ).hexdigest()[:16]
    tmp_dir = os.path.join(tempfile.gettempdir(), "otr_audio_slices")
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
        "-vn", "-c:a", "pcm_s16le", "-ar", "44100", "-ac", "1",
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
    OTR_ImageGenDispatcher write-back; each entry is keyed by ``object_id``)."""
    out = {}
    imgs = ((ledger or {}).get("images") or {}).get("images") or []
    for im in imgs:
        if not isinstance(im, dict):
            continue
        cid = str(im.get("object_id") or im.get("char_id") or "")
        path = str(im.get("path") or "")
        if cid and path:
            out.setdefault(cid, path)
    return out


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
        if v and str(k).endswith("wav_path") and not str(k).startswith(("sfx", "music")):
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
            mapped = "a beat of %s" % intent
            _LOG.info("[OTR.render_driver] unmapped beat_intent %r on %s -- "
                      "using the loose clause", intent, shot_id)
        out.append(mapped)
    arc = str((line or {}).get("arc_phase") or "").strip().lower()
    if arc in _ARC_CLAUSES:
        out.append(_ARC_CLAUSES[arc])
    return out


def _stamp_prompt_meta(req, source, prompt, *, subsource="", beat=""):
    """Stamp prompt observability onto the request (round 5 F2): source enum
    (m4|env|brief+beat), sha8, char count -- ``run_episode`` copies them onto
    the trace rows (durable in the node-92 /history report) and one INFO line
    makes operator log review mechanical."""
    sha8 = hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()[:8]
    req["_prompt_source"] = source
    if subsource:
        req["_prompt_subsource"] = subsource
    req["_prompt_sha8"] = sha8
    req["_prompt_chars"] = len(str(prompt))
    _LOG.info("[OTR.render_driver] prompt source=%s sha8=%s chars=%d beat=%s "
              "| %.100s", source, sha8, len(str(prompt)), beat, prompt)


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
    line = _line_index(ledger).get(_beat_id_for_shot(shot), {})
    # Round 5 F5: the SHOT row carries the ShotLock-normalized char_id (the
    # announcer 'announcer'->cast-row-id join); prefer it, fall back to the
    # raw line value for pre-round-5 planned ledgers.
    char_id = str(shot.get("char_id") or line.get("char_id") or "")
    init_image = _portrait_index(ledger).get(char_id, "")
    if (not init_image
            and ENGINE_FAMILY.get(str(shot.get("engine_id") or ""))
            == "audio_driven_face"):
        # The b002-class silent miss: a talking-head shot whose char_id has no
        # portrait previously surfaced only as eng_humo's fail-closed error
        # mid-render. Warn at the JOIN so the gap is visible upstream.
        _LOG.warning("[OTR.render_driver] talking-head shot %s char_id=%r has "
                     "NO portrait-index entry -- HuMo will fail closed to its "
                     "fallback chain", shot.get("shot_id"), char_id)
    audio = _voice_audio_for_line(line)
    # Per-beat audio fallback: slice the FROZEN master when no per-line wav.
    # The master is opened read-only by ffmpeg; the slice lands in a temp dir
    # so the master is NEVER mutated (V-1 / audio spine frozen).
    if not audio and master_audio_path and os.path.isfile(master_audio_path):
        bid = _beat_id_for_shot(shot)
        start_s = line.get("start_s")
        dur_s = line.get("dur_s")
        if (start_s is not None and dur_s is not None
                and float(dur_s) > 0):
            sliced = _slice_master_audio(master_audio_path,
                                         float(start_s), float(dur_s))
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
    # FULL-FRAME landscape for the generative-motion engines (operator
    # look-QA 2026-06-10): build_request's default canvas is the HuMo
    # PORTRAIT (480x832, the accepted talking-head pillarbox), and LTX/Wan
    # inherited it -- skinny portrait b-roll in a 1472x832 frame. Those
    # engines render the composite canvas instead (both dims /32 for the
    # LTX latent grid; env-overridable).
    if str(shot.get("engine_id") or "") in ("ltx_video", "wan_i2v"):
        _lc = os.environ.get("OTR_VIDEO_LANDSCAPE_CANVAS", "1472x832")
        try:
            _lw, _lh = (int(x) for x in _lc.lower().split("x", 1))
        except (ValueError, AttributeError):
            _lw, _lh = 1472, 832
        req["canvas"]["w"], req["canvas"]["h"] = _lw, _lh
        req["init_w"], req["init_h"] = _lw, _lh
    creative = shot.get("creative") or {}
    text_prompt = str(creative.get("text_prompt") or "")
    if text_prompt:
        req["text_prompt"] = text_prompt
        _stamp_prompt_meta(req, "m4", text_prompt,
                           subsource=str(creative.get("source") or ""),
                           beat=_beat_id_for_shot(shot))
    # SCENE PROMPTS for text-driven engines (gap-audit fix F2, roundtable-
    # hardened: docs/2026-06-10-brief-downstream-gaps/). Any ltx_video /
    # wan_i2v shot with NO writer creative prompt gets a prompt grounded in
    # THE EPISODE'S OWN BRIEF -- the source of the old episodes' scenic,
    # varied opens -- finished with the brief's era tail under the LTX char
    # budget. Covers ALL roles (announcer/music opens AND scene_broll /
    # background_abstract), killing the generic "a 1940s radio studio"
    # default for text engines. Precedence: M4 creative prompt (finished at
    # ShotLock) > OTR_LTX_RADIO_PROMPT (operator override, VERBATIM, no
    # finishing) > brief-composed + finished.
    _shot_role = str(shot.get("role") or "")
    if not _shot_role:
        # Resilience: older planned ledgers carry the role only inside
        # group_id ("grp_<role>") -- parse it so scene composition never
        # silently skips (2026-06-10 acceptance catch).
        _gid = str(shot.get("group_id") or "")
        if _gid.startswith("grp_"):
            _shot_role = _gid[len("grp_"):]
    if (str(shot.get("engine_id") or "") in ("ltx_video", "wan_i2v")
            and not text_prompt):
        # Round 5 F2: synthetic-open detection by STRUCTURE (empty
        # source_line_ids / the ShotLock beat-id suffix), never role alone;
        # role still picks the clause WORDING below.
        _sids = shot.get("source_line_ids")
        _is_synthetic_open = ((isinstance(_sids, list) and not _sids)
                              or str(shot.get("shot_id") or "")
                              .endswith(_OPENING_MUSIC_SUFFIX))
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
        else:
            try:
                from .._otr_story_brief_helpers import (  # type: ignore
                    finish_visual_prompt, get_story_brief_ltx)
            except ImportError:  # pragma: no cover -- flat test imports
                from _otr_story_brief_helpers import (  # type: ignore
                    finish_visual_prompt, get_story_brief_ltx)
            _meta = (ledger or {}).get("meta") or {}
            core = get_story_brief_ltx(_meta)
            if not core:
                _terms = _meta.get("story_brief_terms") or {}
                _setting_raw = (_terms.get("setting")
                                if isinstance(_terms, dict) else None)
                _setting = ", ".join(
                    [str(t).strip() for t in (_setting_raw or [])
                     if str(t).strip()][:2])
                core = ("cinematic establishing shot"
                        + (f", {_setting}" if _setting else ""))
            # Per-beat composition (round 5 F2): role clause FIRST (the
            # bright-radio look survives the 240 trim), then the beat's OWN
            # clauses from the frozen line's beat_intent / arc_phase -- the
            # per-beat variety the empty visual_plan.scenes can't supply.
            clauses = []
            if _is_synthetic_open:
                clauses.append("opening establishing shot, the radio warming "
                               "up, warm glowing dial light")
            elif _shot_role == "announcer_visual":
                clauses.append("a vintage radio set glowing warmly, lit "
                               "dials and tubes")
            elif _shot_role == "music_visual":
                clauses.append("the radio warming up, warm glowing dial "
                               "light")
            clauses.extend(_beat_clauses(line, shot.get("shot_id")))
            clauses.extend(["slow cinematic camera drift",
                            "no on-screen text"])
            scene_prompt = finish_visual_prompt(
                _meta, f"{core}, {', '.join(clauses)}",
                max_chars=240, style_tail=False)
            _LOG.warning("[OTR.render_driver] LTX SCENE: %s beat %s prompt "
                         "composed from the episode brief (%d chars): "
                         "%.90s...", _shot_role, shot.get("shot_id"),
                         len(scene_prompt), scene_prompt)
            req["text_prompt"] = scene_prompt
            _stamp_prompt_meta(req, "brief+beat", scene_prompt,
                               beat=_beat_id_for_shot(shot))
    req_hash = (shot.get("render_request_hash")
                or (shot.get("cache_keys") or {}).get("request_hash"))
    req["seed_bundle"] = {"request_seed": _seed_from_hash(req_hash, shot.get("shot_id"))}
    # Carry the per-beat timing so the render node can slice the frozen master
    # mix when the ledger has no per-line wav (audio_ref is None in that case).
    # SYNTHETIC shots (the opening-music scene, 2026-06-10) have no ledger
    # line; the shot row itself carries start_s/dur_s -- fall back to it so
    # the positioned timeline keeps every row placed.
    req["timing"]["start_s"] = (line.get("start_s")
                                if line.get("start_s") is not None
                                else shot.get("start_s"))
    req["timing"]["dur_s"] = (line.get("dur_s")
                              if line.get("dur_s") is not None
                              else shot.get("dur_s"))
    req["char_id"] = char_id
    return req


# --------------------------------------------------------------------------- #
# The in-process render (the GPU slice)
# --------------------------------------------------------------------------- #
def _render_one(engine_name, request, *, force_oom):
    """Attempt ONE candidate engine: assert_usable -> prepare -> render_clip ->
    canonicalize, always teardown. ``force_oom`` raises BEFORE any work (the
    soak's deterministic mid-episode OOM). Raises on failure; returns the
    canonical clip dict on success."""
    if force_oom:
        raise OomSignal("forced soak OOM on %s" % engine_name)
    if not _vreg.is_registered(engine_name):
        raise LookupError("engine %r is not registered" % engine_name)
    eng = _vreg.get_engine(engine_name)
    prepared = None
    try:
        eng.assert_usable(host_caps={}, profile={})
        prepared = eng.prepare(host_caps={}, profile={}, session_ctx={})
        raw = eng.render_clip(request, prepared)
        return eng.canonicalize(raw, request, {})
    finally:
        if prepared is not None:
            try:
                eng.teardown(prepared)
            except Exception:            # noqa: BLE001 - teardown best-effort
                pass


#: Default text prompt for an on-the-fly lipsync BASE clip -- face-forward so
#: the overlay's landmarker has a mouth to drive (env-overridable).
_LSYNC_BASE_PROMPT = (
    "close-up portrait of a 1940s radio actor speaking into a studio "
    "microphone, face centered, warm tungsten light, period drama")


def _provide_lipsync_base(engine_name, request):
    """Provider seam (operator ask 2026-06-09, the LTX+latentsync combo): a
    ``lipsync_overlay`` engine needs a BASE clip; when the request has none and
    ``OTR_LSYNC_BASE_ENGINE`` names a provider (e.g. ``ltx_video``), render the
    base IN-LINE first and feed its path as ``base_clip_ref``. LOUD; additive;
    no env -> no behavior change. A base-render failure leaves base_clip_ref
    unset so the overlay fails its own usability check and the normal LOUD
    fallback chain runs."""
    if engine_family(engine_name, "") != "lipsync_overlay":
        return
    get = request.get if isinstance(request, dict) else (
        lambda k, d=None: getattr(request, k, d))
    if get("base_clip_ref"):
        return
    base_engine = os.environ.get("OTR_LSYNC_BASE_ENGINE", "").strip()
    if not base_engine:
        return
    base_req = copy.deepcopy(request) if isinstance(request, dict) else dict(request)
    base_req["base_clip_ref"] = None
    base_req["audio_ref"] = None          # the base is SILENT b-roll (V-1)
    # Gap-audit F2 decision (2026-06-10): the panel suggested preferring the
    # request's brief-grounded prompt here, but the FACE-FORWARD default is
    # functional, not aesthetic -- the overlay's landmarker needs a mouth,
    # and a scene-y prompt re-breaks the combo lane's face-detect lottery.
    # Env override verbatim; otherwise the face-forward default stands.
    # Revisit only if the lipsync combo lane is promoted past experiment.
    base_req["text_prompt"] = os.environ.get(
        "OTR_LSYNC_BASE_PROMPT", _LSYNC_BASE_PROMPT)
    _LOG.warning("[OTR video] LOUD lipsync base: rendering %s base for shot %s "
                 "via %s (OTR_LSYNC_BASE_ENGINE)", engine_name,
                 base_req.get("shot_id"), base_engine)
    try:
        base_clip = _render_one(base_engine, base_req, force_oom=False)
    except Exception as exc:              # noqa: BLE001 -- LOUD; chain handles it
        _LOG.warning("[OTR video] lipsync base render FAILED (%s: %s) -- the "
                     "overlay will fail closed and walk its fallback chain",
                     type(exc).__name__, str(exc).splitlines()[0][:160])
        return
    path = (base_clip or {}).get("path") or ""
    if path:
        request["base_clip_ref"] = {"path": path}
        _LOG.warning("[OTR video] lipsync base ready: %s", path)


def render_shot(shot, request, *, fallback_of, video_revision,
                oom_engines=frozenset(), oom_shot_id=None):
    """Render ONE shot through the fallback chain LOUDLY until a clip renders.

    Returns ``(clip, restamped_shot, decisions, attempts, vram_used_mb)``. Each
    HARD failure restamps the shot row + appends a runtime_fallback_decision at
    the SAME revision and logs the LOUD swap. The floor always succeeds, so the
    loop terminates with a clip (or raises RenderFloorError if the floor itself
    cannot render -- the negative control)."""
    sid = shot["shot_id"]
    bid = shot.get("beat_id", sid)
    chain = resolve_fallback_chain(shot["engine_id"], fallback_of)
    decisions = []
    attempts = []
    out_shot = dict(shot)
    used_mb = None
    for cand in chain:
        force = (sid == oom_shot_id and cand in oom_engines)
        attempts.append(cand)
        try:
            _provide_lipsync_base(cand, request)
            clip = _render_one(cand, request, force_oom=force)
            used_mb = _mc.vram_used_mb()
            return clip, out_shot, decisions, attempts, used_mb
        except Exception as exc:         # noqa: BLE001 - classified + LOUD swap
            nxt = fallback_of(cand)
            kind = _rt.FailureKind.OOM if force else classify_failure(exc)
            if nxt is None:
                raise RenderFloorError(
                    "radio floor %r failed for shot %s: %s: %s"
                    % (cand, sid, type(exc).__name__, exc))
            _rt.classify(kind)           # validate the decision invariants
            detail = ("forced soak OOM" if force
                      else "%s: %s" % (type(exc).__name__,
                                       str(exc).splitlines()[0]))[:200]
            rec = _rt.build_fallback_decision(
                shot_id=sid, beat_id=bid, from_engine=cand, to_engine=nxt,
                kind=kind, video_revision=video_revision, detail=detail)
            decisions.append(rec)
            out_shot = _rt.restamp_shot_row(
                out_shot, to_engine=nxt,
                to_family=engine_family(nxt, out_shot.get("family")),
                from_engine=cand, kind=kind)
            _LOG.warning(_rt.format_swap_log(rec))
    raise RenderFloorError("chain exhausted without a floor for shot %s" % sid)


def run_episode(ledger, *, fallback_of, oom_shot_id=None,
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
    section = ledger["video"]
    rev = int(section["video_revision"])
    clips, new_shots, trace = {}, [], []
    vram_peak = 0
    for shot in section["shots"]:
        if request_builder is not None:
            request = request_builder(shot, ledger, canvas=canvas)
        else:
            request = build_request(shot, assets, frame_count, canvas)
        clip, out_shot, decisions, attempts, used = render_shot(
            shot, request, fallback_of=fallback_of, video_revision=rev,
            oom_engines=oom_engines, oom_shot_id=oom_shot_id)
        for rec in decisions:
            section = _rt.append_runtime_fallback_decision(section, rec)
        clips[out_shot["shot_id"]] = clip
        new_shots.append(out_shot)
        row = {"shot_id": out_shot["shot_id"], "attempts": attempts,
               "final_engine": out_shot["engine_id"]}
        # Round 5 F2: prompt observability rides the trace (durable in the
        # node-92 /history report) -- the diversity gate + the operator's
        # "did the prompts actually differ" check read these.
        for key in ("_prompt_source", "_prompt_subsource", "_prompt_sha8",
                    "_prompt_chars"):
            if isinstance(request, dict) and key in request:
                row[key.lstrip("_")] = request[key]
        trace.append(row)
        if used:
            vram_peak = max(vram_peak, int(used))
    section["shots"] = new_shots
    ledger["video"] = section
    return {"ledger": ledger, "clips": clips, "trace": trace,
            "vram_peak_mb": vram_peak}


def run_real_episode(ledger, *, fallback_of=None, canvas=None,
                     master_audio_path=""):
    """Drive one REAL episode from a ShotLock-planned ledger: per-shot requests
    (character portrait + per-beat voice audio + the M4 prompt) via
    :func:`build_request_from_shot`, the full registry fallback chain (NO forced
    OOM), real per-shot assets. Returns the :func:`run_episode` result; the
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
    return run_episode(ledger, fallback_of=fallback_of or make_fallback_of(),
                       request_builder=rb, canvas=canvas)


def parse_engine_override(spec: str) -> dict:
    """Parse ``OTR_FORCE_ENGINE_MAP`` (pure). Grammar: comma-separated
    ``role=engine`` pairs; the role ``*`` means EVERY shot regardless of role.
    Examples: ``*=ltx_video`` (the all-LTX episode);
    ``character_video=latentsync,announcer_visual=latentsync,scene_broll=ltx_video``.
    Unknown engines raise at parse time (fail-closed, before any render)."""
    out = {}
    for pair in (spec or "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(
                "OTR_FORCE_ENGINE_MAP entry %r is not role=engine" % pair)
        role, engine = (s.strip() for s in pair.split("=", 1))
        if engine not in ENGINE_FAMILY and not _vreg.is_registered(engine):
            raise ValueError(
                "OTR_FORCE_ENGINE_MAP names unknown engine %r" % engine)
        out[role] = engine
    return out


def apply_engine_override(ledger):
    """Experiment knob (operator ask 2026-06-09: the all-LTX / LTX+latentsync
    episodes): when ``OTR_FORCE_ENGINE_MAP`` is set, re-route each planned
    shot's ``engine_id``/``family`` by role BEFORE rendering, LOUDLY. The
    fallback chains stay intact (a forced engine that fails still degrades to
    the radio floor with the usual LOUD restamp). Returns the (possibly
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
    for shot in section.get("shots") or []:
        role = str(shot.get("role") or "")
        engine = mapping.get(role) or mapping.get("*")
        if not engine or shot.get("engine_id") == engine:
            continue
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
    rows = []
    total = 0
    hist = {}
    for order, shot in enumerate(shots):
        sid = shot.get("shot_id")
        clip = clips.get(sid) or {}
        path = str(clip.get("path") or "")
        exists = bool(path) and os.path.exists(path)
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
        rows.append({
            "order": order, "shot_id": sid, "beat_id": bid,
            "engine_id": eid,
            "family": clip.get("family") or shot.get("family") or "",
            "path": path,
            "frame_count": int(clip.get("frame_count") or 0),
            "target_frame_count": tfc,
            "start_s": start_s,
            "char_id": row_char,
            "init_image": _portrait_index(led).get(row_char, ""),
            "exists": exists,
        })
        if exists:
            hist[eid] = hist.get(eid, 0) + 1
    return {
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


# --------------------------------------------------------------------------- #
# A-S7.5 full-episode soak (two back-to-back episodes on REAL engines)
# --------------------------------------------------------------------------- #
def _clip_summary(clip):
    """Compact, JSON-able view of a rendered clip + its on-disk reality."""
    path = (clip or {}).get("path", "")
    exists = bool(path) and os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    return {"engine_id": (clip or {}).get("engine_id"),
            "family": (clip or {}).get("family"),
            "frame_count": (clip or {}).get("frame_count"),
            "path": path, "exists": exists, "size": size}


def _norm_decisions(section):
    """Structural (path-free) view of the runtime_fallback_decisions for the
    determinism compare + the audit."""
    return [{k: d.get(k) for k in ("shot_id", "from_engine", "to_engine",
                                   "failure_kind", "block_class",
                                   "video_revision")}
            for d in section.get("runtime_fallback_decisions", [])]


def _episode_facts(ep, meta):
    led = ep["ledger"]
    sec = led["video"]
    shots = {s["shot_id"]: s for s in sec["shots"]}
    oom = shots[meta["oom_shot_id"]]
    clips = {sid: _clip_summary(c) for sid, c in ep["clips"].items()}
    return {
        "n_clips": len(ep["clips"]),
        "all_clips_real": all(c["exists"] and c["size"] > 0
                              for c in clips.values()),
        "oom_final_engine": oom["engine_id"],
        "oom_trail": oom["degradation_trail"],
        "decisions": _norm_decisions(sec),
        "video_revision": sec["video_revision"],
        "audio_sha": led["audio"]["master_audio_sha256"],
        "humo_rendered": sum(1 for c in clips.values()
                             if c["engine_id"] == "humo" and c["exists"]),
        "vram_peak_mb": ep["vram_peak_mb"],
        "trace": ep["trace"],
        "clips": clips,
    }


def assemble_report(meta, input_ledger, e1, e2, *, vram_ceiling_mb, elapsed_s):
    return {
        "meta": meta, "vram_ceiling_mb": int(vram_ceiling_mb),
        "elapsed_s": round(float(elapsed_s), 1),
        "episode_1": _episode_facts(e1, meta),
        "episode_2": _episode_facts(e2, meta),
        "input_oom_engine":
            {s["shot_id"]: s for s in
             input_ledger["video"]["shots"]}[meta["oom_shot_id"]]["engine_id"],
        "input_oom_trail":
            {s["shot_id"]: s for s in
             input_ledger["video"]["shots"]}[meta["oom_shot_id"]]
            ["degradation_trail"],
    }


def assert_soak_ok(report):
    """Assert every A-S7.5 GPU-soak invariant; raise :class:`SoakError` on any
    violation. Returns the list of passed-check descriptions for the report."""
    meta = report["meta"]
    n = meta["n_beats"]
    ceiling = report["vram_ceiling_mb"]
    checks = []
    for tag in ("episode_1", "episode_2"):
        f = report[tag]
        if f["n_clips"] != n or not f["all_clips_real"]:
            raise SoakError("%s: not every beat produced a real on-disk clip "
                            "(%d/%d, all_real=%s)"
                            % (tag, f["n_clips"], n, f["all_clips_real"]))
        if f["oom_final_engine"] != "still_kenburns":
            raise SoakError("%s: character_3d OOM did not converge to the radio "
                            "floor (got %r)" % (tag, f["oom_final_engine"]))
        if f["oom_trail"] != EXPECTED_OOM_TRAIL:
            raise SoakError("%s: OOM degradation trail %r != %r"
                            % (tag, f["oom_trail"], EXPECTED_OOM_TRAIL))
        oom_decisions = [d for d in f["decisions"]
                         if d["shot_id"] == meta["oom_shot_id"]]
        if len(oom_decisions) != 3:
            raise SoakError("%s: expected 3 LOUD OOM decisions on the "
                            "character_3d shot, got %d"
                            % (tag, len(oom_decisions)))
        for d in oom_decisions:
            if (d["failure_kind"] != "oom" or d["block_class"] != "hard"
                    or d["video_revision"] != 1):
                raise SoakError("%s: malformed OOM decision %r" % (tag, d))
        if f["video_revision"] != 1:
            raise SoakError("%s: video_revision bumped to %r (a restamp stays at "
                            "the same revision)" % (tag, f["video_revision"]))
        if f["audio_sha"] != FROZEN_AUDIO_SHA:
            raise SoakError("%s: frozen audio sha changed (%r) -- the render "
                            "driver must never touch audio" % (tag, f["audio_sha"]))
        if f["humo_rendered"] < 1:
            raise SoakError("%s: humo never rendered in-process (0 real humo "
                            "clips) -- the heavy in-process forward did not run"
                            % tag)
        if f["vram_peak_mb"] and f["vram_peak_mb"] > ceiling:
            raise SoakError("%s: VRAM peak %d MB > ceiling %d MB"
                            % (tag, f["vram_peak_mb"], ceiling))
        checks.append("%s: %d real clips; character_3d OOM->floor converged "
                      "(%d LOUD restamps @rev1); %d humo in-process renders; "
                      "VRAM peak %s MB <= %d; frozen audio untouched"
                      % (tag, n, len(oom_decisions), f["humo_rendered"],
                         f["vram_peak_mb"], ceiling))
    if report["episode_1"]["trace"] != report["episode_2"]["trace"]:
        raise SoakError("non-deterministic: the two episodes' render traces "
                        "(per-shot attempts + final engine) differ")
    if report["episode_1"]["decisions"] != report["episode_2"]["decisions"]:
        raise SoakError("non-deterministic: the two episodes' fallback "
                        "decisions differ")
    if (report["input_oom_engine"] != "hunyuan3d_talk"
            or report["input_oom_trail"]):
        raise SoakError("carryover: the shared input fixture was mutated")
    checks.append("determinism: two back-to-back episodes identical "
                  "(traces + decisions); input fixture unmutated (no carryover)")
    return checks


def run_gpu_soak(*, n_beats=40, oom_index=20, frame_count=25, assets=None,
                 vram_ceiling_mb=None):
    """Run the A-S7.5 full-episode soak on REAL GPU engines TWICE back-to-back,
    assert every invariant, and return the structured report. Raises
    :class:`SoakError` on a violation (never a fake pass)."""
    ceiling = int(vram_ceiling_mb or _mc.VRAM_CEILING_MB)
    section, meta = build_soak_fixture(n_beats=n_beats, oom_index=oom_index)
    ledger = build_full_ledger(section)
    fb = make_fallback_of()
    t0 = time.time()
    e1 = run_episode(ledger, fallback_of=fb, oom_shot_id=meta["oom_shot_id"],
                     oom_engines=OOM_ENGINES, assets=assets,
                     frame_count=frame_count)
    e2 = run_episode(ledger, fallback_of=fb, oom_shot_id=meta["oom_shot_id"],
                     oom_engines=OOM_ENGINES, assets=assets,
                     frame_count=frame_count)
    report = assemble_report(meta, ledger, e1, e2, vram_ceiling_mb=ceiling,
                             elapsed_s=time.time() - t0)
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
    request = build_request(shot, assets, frame_count, canvas)
    t0 = time.time()
    try:
        _provide_lipsync_base(engine_name, request)   # combo seam (env-gated)
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
    "FLOOR_NAMES", "UNIVERSAL_FLOOR", "SYNTH_FALLBACKS", "ENGINE_FAMILY",
    "OOM_ENGINES", "FROZEN_AUDIO_SHA", "EXPECTED_OOM_TRAIL",
    "OomSignal", "RenderFloorError", "SoakError",
    "make_fallback_of", "classify_failure", "engine_family",
    "build_soak_fixture", "build_full_ledger", "build_request",
    "build_request_from_shot", "_slice_master_audio",
    "run_real_episode", "build_clip_manifest",
    "parse_engine_override", "apply_engine_override",
    "render_shot", "run_episode", "assemble_report", "assert_soak_ok",
    "run_gpu_soak", "render_single",
]
