"""
batch_ltx_render.py  --  OTR_BatchLTXRender ComfyUI node
========================================================

Render N LTX-2 video clips in lockstep from a Production Ledger.
Sister to ``batch_humo_render.py``: walks ``ledger.lines[]`` filtering
for non-character roles ({announcer, music_open, music_close,
music_inter, sfx}) and renders each as an LTX I2V clip with the
radio_bookend.png as the conditioning image.

Architecture (locked 2026-05-01 with Jeffrey, after BUG-LOCAL-129
settled that HuMo cannot animate non-face references):

    character lines         -> HuMo (existing batch_humo_render.py)
    announcer / music / sfx -> LTX-2 (this node)
                                I2V ref = radio_bookend.png
                                25 fps native, 480p, 8n+1 frames

Pattern adapted from Jeffrey's ComfyUI-Goofer ``GooferBatchVideo``
which is already proven on this exact Blackwell sm_120 hardware:
distilled LTX-2 sigma schedule, SamplerCustomAdvanced, video-only
decode (we discard LTX's optional audio output -- master_mix audio
is muxed at VideoComposite step, byte-identical to v1.5 baseline).

VRAM strategy:
  - Frame count cap: 257 (LTX VAE max). Per Jeffrey 2026-05-01 EVENING,
    his proven ComfyUI-Goofer node renders 257-frame clips at 768x512
    on this exact Blackwell hardware -- the trick is ``VAEDecodeTiled``
    at decode time. Gemini's 97-frame round-robin estimate assumed
    non-tiled decode; with tiled decode, 257 fits.
  - VAE decode uses ``VAEDecodeTiled`` (tile_size=512, overlap=64,
    temporal_size=4096, temporal_overlap=8) -- exactly the parameters
    Goofer ships with. This is the only reason 257 frames doesn't OOM.
  - Strict teardown after the loop (unload_all_models + gc +
    empty_cache + cuda.synchronize) per
    ``reference_chained_backend_teardown.md`` so HuMo (16.5 GB
    staged) loads cleanly in the next phase.

Per CLAUDE.md C5: weights stay at the model's native precision; we
do NOT re-quantize.

Per CLAUDE.md C7: this node generates VIDEO-ONLY clips. No audio is
written. Audio path stays untouched all the way through to
VideoComposite's final mux which uses ``-c:a copy`` from procgen.
"""
from __future__ import annotations

import gc
import logging
import math
import os
import sys as _sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path-helper bootstrap (same pattern as batch_humo_render.py)
# ---------------------------------------------------------------------------
_NODES_DIR = Path(__file__).resolve().parent
if str(_NODES_DIR) not in _sys.path:
    _sys.path.insert(0, str(_NODES_DIR))

# folder_paths is the ComfyUI canonical path resolver. The OTR helpers
# (otr_videos_dir, otr_stills_dir, etc.) wrap folder_paths internally for
# the broadcast tree under output/otr/, but referencing folder_paths here
# explicitly satisfies the Bug Bible BUG-01.02 contract that every
# OUTPUT_NODE file references the canonical resolver.
#
# 2026-05-04 BUG-LOCAL-091 follow-up: wrapped in try/except so headless
# pytest collection doesn't blow up (folder_paths is provided by the
# ComfyUI runtime; pytest doesn't have it in scope). The runtime path
# inside execute() goes through _otr_paths helpers which already have
# their own folder_paths fallback chain.
try:
    import folder_paths  # noqa: F401,E402
except ImportError:
    folder_paths = None  # type: ignore[assignment]

from _otr_paths import (  # noqa: E402
    otr_audio_dir,
    # S28 cleanbreak: dropped otr_legacy_audio_dir.
    otr_stills_dir,
    otr_videos_dir,
)
from _otr_speaker_role import (  # noqa: E402
    SPEAKER_ROLE_CHARACTER,
    is_never_humo_role,
    resolve_speaker_role,
)

log = logging.getLogger("OTR.batch_ltx_render")


# ---------------------------------------------------------------------------
# LTX constants
# ---------------------------------------------------------------------------

# Match HuMo's frame rate so concat-demuxer at VideoComposite stage
# doesn't see frame rate seams. Gemini round-robin 2026-05-01 caught
# this as a stutter-prevention requirement.
LTX_FPS = 25

# 8n+1 frame count rule for LTX VAE temporal compression.
# BUG-LOCAL-091 (2026-05-04 EVENING): bumped from 177 (~7.08s) to 353
# (~14.12s) to match the post-BUG-086 HuMo cap. Lines exceeding the
# user-configurable per-chunk ceiling (the new ``clip_length`` widget,
# default 7.0s) fall into the chunking dispatch below. ComfyUI-Goofer
# proved 257 native is fine; 353 is untested-but-likely-fine because
# VAEDecodeTiled handles temporal chunking at the VAE level.
#
# 2026-05-07 PM: bumped to 705 (8*88+1 = ~28.20s @ 25fps) for the
# "how long can LTX 2.3 go" curiosity test. The distilled-1.1 LoRA
# may temporal-collapse or repeat at this length, OR may handle it
# fine since it's a 22B model with much more capacity than the 2B
# v0.9 BUG-091 era. VAEDecodeTiled's temporal chunking should still
# protect against decode-side OOM. Sample-side memory is the concern;
# watch VRAM during long chunk renders.
LTX_MAX_FRAMES = 705  # was 353; 705 = 8*88+1 = ~28.20s at 25fps
LTX_MIN_FRAMES = 9    # 8*1+1, smallest valid LTX render
# BUG-LOCAL-091 (2026-05-04 EVENING): when a non-character audio line
# exceeds the per-chunk ceiling, split it into consecutive LTX chunks of
# this many frames each, then ffmpeg-concat into a single per-line mp4.
# 177 frames (7.08s) per chunk matches the historically-stable LTX render
# size and the pre-bump LTX_MAX_FRAMES.
LTX_CHUNK_FRAMES = 177

# VAEDecodeTiled parameters (from ``ComfyUI-Goofer`` line 472-476).
# These have been empirically verified on RTX 5080 Blackwell --
# don't change without re-validating on hardware.
LTX_TILE_SIZE = 512
LTX_TILE_OVERLAP = 64
LTX_TEMPORAL_SIZE = 4096
LTX_TEMPORAL_OVERLAP = 8

# 480p native -- LTX 2B optimal resolution. Width chosen to match
# HuMo's 832 (so VideoComposite pillarbox math is identical for both
# clip sources).
# 2026-05-03 EVENING (BUG-LOCAL-030 revised spec): kept LTX at native
# 832x480 landscape per Jeffrey's revised wording ("ltx native landscape
# render downscaled to 1472x832"). An earlier revision bumped to
# 1216x704 for higher pre-upscale detail, but Jeffrey reverted to
# "native" twice -- safer trained-distribution default. VideoComposite
# pillarboxes to the 1472x832 canvas: 832x480 -> scale to height 832 =
# 1442x832 -> pad to 1472x832 with ~15px black per side. Final
# RTXUpscale 1472x832 -> 1920x1080 then post-upscale procgen blend
# (Phase B) for the SIGNAL LOST CRT signature.
LTX_WIDTH = 832
LTX_HEIGHT = 480

# Distilled sigma schedule from Jeffrey's ComfyUI-Goofer, proven on
# RTX 5080 Blackwell. 8 sampling steps, last sigma 0.0 = full denoise.
LTX_DISTILLED_SIGMAS = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975,
    0.909375, 0.725, 0.421875, 0.0,
]

# CFG = 1.0 for distilled LTX (no classifier-free guidance needed).
LTX_CFG = 1.0

# I2V conditioning strength: 0.75 = strong reference, but leaves room
# for the model to add motion. Goofer's empirically tuned default.
LTX_I2V_STRENGTH = 0.75

# 2026-05-01 Jeffrey ORIGINAL INTENT: feed radio still as BOTH start
# and end keyframes for seamless-loop ping-pong in VideoComposite.
# RETIRED 2026-05-03 EVENING per BUG-LOCAL-032: two strong anchors
# clamped LTX into near-static ping-pong.
# 2026-05-04 EVENING per BUG-LOCAL-095: even the start-frame guide
# (LTXVAddGuide) was acting as a hard keyframe pin and producing
# "ltx looks like a still" output. Replaced with the canonical
# LTXVImgToVideoConditionOnly i2v init which encodes the image into
# the first frames of the latent + adds a noise mask for free motion.
# Same proven path comfyui-data-media-machine uses.
LTX_END_FRAME_STRENGTH = 0.6  # DEPRECATED -- see batch_ltx_render.py BUG-032 fix block

# ---------------------------------------------------------------------------
# BUG-LOCAL-117: LTX 2.3 + RES4LYF engine config
# ---------------------------------------------------------------------------
# Engine selector via env var (NOT widget -- widget drift is recurring OTR
# bug source per BUG-LOCAL-097/113). Round-robin verdict 2026-05-06 (transcripts
# at docs/2026-05-06-bug-117-ltx23-res4lyf-migration__*.md): use env var,
# loud startup log, fail-fast dep check on engine=v2_3.
#
# v2_3:  LTX 2.3 22B-dev BF16 + distilled LoRA (in workflow JSON) + Gemma
#        encoder (LTXAVTextEncoderLoader in workflow JSON) + RES4LYF
#        ClownSampler_Beta + MultimodalGuider + GuiderParameters +
#        LTXVTiledVAEDecode. Produces "subtle zoom in" smooth motion on
#        radio_bookend smoke 2026-05-06.
# v0_9:  Legacy LTX 2B v0.9 + CLIPLoader/t5xxl + euler + CFGGuider +
#        VAEDecodeTiled. Kept as fallback for emergency rollback.
#        Rollback path: git checkout pre-bug-117-cutover workflows/otr_scifi_16gb_full.json
#        and `set OTR_LTX_ENGINE=v0_9` before launching ComfyUI.
# BUG-LOCAL-117a A/B verdict 2026-05-07 LATE MORNING (Jeffrey eyes-on):
# v0_9 + euler_cfg_pp on cargo_hold smoke = visually indistinguishable
# from v2_3 + ClownSampler res_2s on the same content. v0_9 ships in
# 8.3 min for 5 clips vs v2_3's 36.85 min (4.4x faster) and runs
# comfortably on 32 GB RAM where v2_3 thrashes 64 GB to 100%.
# DEFAULT = v0_9. v2_3 retained as opt-in "quality" engine for users
# who want crisper detail and have 64 GB+ RAM headroom.
OTR_LTX_ENGINE_DEFAULT = "v0_9"

# BUG-LOCAL-117a Phase 1 (2026-05-07): A/B/C/D sampler bake-off via env var.
# Applies only when engine=v0_9 (the lightweight single-stage path that
# doesn't need RES4LYF / MultimodalGuider / AV-stub). Lets us test each
# sampler in turn without rebuilds:
#   euler                       -- LTX 2.0 era plain euler (sanity reference)
#   euler_cfg_pp                -- Lightricks LTX 2.3 single-stage tuned default
#   res_multistep               -- DPM-Solver++ multistep (RES family cousin)
#   res_multistep_cfg_pp        -- Same with cfg_pp projection (likely best non-RES4LYF)
#   res_multistep_ancestral     -- Ancestral noise variant
#   res_multistep_ancestral_cfg_pp
# Per Jeffrey's research (2026-05-07): at 8 distilled steps, multistep's
# forward-pass advantage over euler vanishes (both = 8 evals); the only
# differentiator is higher-order accuracy from the history term, which has
# barely any room to express itself at 8 steps. So Phase 1 result may be
# near-neutral. That's still a valid finding -- it tells us to ship
# euler_cfg_pp and stop optimizing the sampler axis.
LTX_V0_9_SAMPLER_NAMES_VALID = (
    "euler",
    "euler_cfg_pp",
    "res_multistep",
    "res_multistep_cfg_pp",
    "res_multistep_ancestral",
    "res_multistep_ancestral_cfg_pp",
)
LTX_V0_9_SAMPLER_NAME_DEFAULT = "euler_cfg_pp"

# BUG-LOCAL-117c (Jeffrey 2026-05-07 12:50): seamless loop / chunk-boundary
# smoothing via end-frame anchor. Each LTX clip currently starts at the
# radio_bookend (LTXVImgToVideoConditionOnly start anchor at strength 0.75)
# but ends wherever the model takes it -- when chunks concat in
# VideoComposite, the end-of-clip-N -> start-of-clip-N+1 transition can show
# a visible snap because both ends land in different visual states.
#
# Adding LTXVAddGuide at frame_idx=-1 with a soft strength (e.g. 0.4) tells
# the model to land back near the radio_bookend at the END of each clip too.
# Result: end-of-clip-N == start-of-clip-N+1 == radio_bookend -> seamless.
#
# BUG-LOCAL-032 retired this for LTX 2B v0.9 because two strong anchors
# clamped that under-powered model into static ping-pong. LTX 2.3 has way
# more motion capacity (proven by 2026-05-07 smoke), so the trap may not
# apply. Default off (0.0) so existing users see no behavior change.
# Recommended starting value: 0.4 (lighter than start's 0.75).
LTX_END_ANCHOR_STRENGTH_DEFAULT = 0.0

# BUG-LOCAL-117d (Jeffrey 2026-05-07 PM): boomerang loop via ffmpeg
# reverse-and-concat. Empirical finding (BUG-LOCAL-117c InplaceKJ test
# at strength 0.7): forcing both endpoints with stacked anchors makes
# the model glitch in the middle (couldn't reconcile two strong
# constraints in 8 distilled steps). The post-process boomerang
# bypasses the architectural ceiling entirely.
#
# Strategy: render HALF the chunk's target duration so the post-process
# reverse-and-concat doubles back to the full audio-timeline duration.
# The boomeranged clip starts at radio_bookend, zooms in to peak motion
# at the midpoint, then plays the same frames in reverse to land back
# at radio_bookend. End-of-clip-N == start-of-clip-N+1 == radio_bookend
# -> seamless chunk concat in VideoComposite. Free perfect loops, no
# extra GPU time (ffmpeg is CPU-bound and dwarfed by sample wall time).
#
# Values:
#   "on"  (default 2026-05-07 PM) -- non-character chunks boomeranged
#                                    (music_*, announcer, sfx). Sample
#                                    wall time HALVES because we render
#                                    half-duration; output reads as the
#                                    full audio target after boomerang.
#   "off"                         -- pre-117d behavior (full-duration
#                                    render, no post-process, may show
#                                    visible chunk-boundary snap).
LTX_LOOP_VIA_REVERSE_DEFAULT = "on"
LTX_LOOP_VIA_REVERSE_TRUTHY = ("on", "1", "true", "yes")

# v2.3 GuiderParameters for VIDEO modality. Values mirror the stock
# Lightricks LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json widget
# (line 1043-1052 of that file). Round-robin (Gemini + NVIDIA) flagged
# MultimodalGuider as STRUCTURALLY required for LTX 2.3 DiT -- substituting
# CFGGuider would crash with tensor shape mismatch.
LTX_V2_3_VIDEO_CFG = 3.0
LTX_V2_3_VIDEO_STG = 1.0
LTX_V2_3_VIDEO_PERTURB_ATTN = True
LTX_V2_3_VIDEO_RESCALE = 0.9
LTX_V2_3_VIDEO_MODALITY_SCALE = 3.0
LTX_V2_3_VIDEO_SKIP_STEP = 0
LTX_V2_3_VIDEO_CROSS_ATTN = True

# MultimodalGuider widget: skip_blocks string parsed as comma-separated
# transformer block indices. Stock workflow uses "28" -- empirically tuned
# by Lightricks for LTX 2.3.
LTX_V2_3_SKIP_BLOCKS = "28"

# ClownSampler_Beta widgets. Stock workflow uses eta=0.25,
# sampler_name="exponential/res_2s", bongmath=true. The seed is
# overridden per-chunk in the render loop (shot_seed math from batch
# loop iteration, line index, chunk index).
LTX_V2_3_CLOWN_ETA = 0.25
LTX_V2_3_CLOWN_SAMPLER_NAME = "exponential/res_2s"
LTX_V2_3_CLOWN_BONGMATH = True

# RES4LYF + LTX 2.3 v2.3 required nodes -- fail-fast dep check fires here
# at execute() entry when engine=v2_3.
LTX_V2_3_REQUIRED_NODES = (
    "ClownSampler_Beta",       # RES4LYF beta/samplers.py
    "MultimodalGuider",        # ComfyUI-LTXVideo guiders/multimodal_guider.py
    "GuiderParameters",        # ComfyUI-LTXVideo guiders/parameters.py
    "LTXVTiledVAEDecode",      # ComfyUI-LTXVideo
)

# Speaker_role -> LTX prompt template.
#
# BUG-LOCAL-112 rewrite (2026-05-06, round-robin verified):
# round-robin (ChatGPT gpt-5.5 + Gemini gemini-3.1-pro-preview-customtools +
# NVIDIA llama-3.3-nemotron-super-49b-v1.5) all converged on the dominant
# cause of static LTX output being PROMPT DILUTION. The prior 700-char
# templates buried motion verbs in the middle of static set-dressing
# language ("obsidian console", "purple lighting", "35mm film grain")
# plus negation language ("no people in frame", "unattended equipment").
# T5-XXL doesn't truncate at 700 chars (its window is 512 tokens, plenty
# of room) but it DOES dilute attention -- the model interprets the
# prompt as "render a beautiful static set" rather than "make a
# continuous moving shot."
#
# Gemini's load-bearing insight: LTX-Video v0.9 was natively trained
# on 121 frames (4.84s @ 25fps). Our typical announcer beat is 5-7s =
# 125-169 frames, OOD on length. The classic OOD failure mode for
# diffusion video is to FREEZE INTO A STATIC IMAGE to prevent temporal
# collapse. So a diluted prompt + OOD length is the perfect recipe for
# the static output we measured.
#
# New design (each template < 160 chars):
#   - Lead with "Continuous shot." to establish the temporal frame.
#   - Front-load LOCAL motion verbs (dial sweeps, tubes pulse, grille
#     trembles, dust drifts) -- the i2v anchor already provides visual
#     identity, so the prompt doesn't need to redescribe the room.
#   - End with CAMERA motion (dolly forward / pull back / orbit) so
#     the model has both local and global motion cues.
#   - "Same console throughout" suppresses scene-cut spikes (we measured
#     32-35 MAD spikes which look like cuts, not motion).
#   - NO negation language ("no people", "unattended"). Text encoders
#     handle negation unreliably; the token "people" still activates
#     people concepts. The i2v anchor (radio_bookend FLUX still) is
#     already empty of people, so we don't need to re-state it here.
#   - NO static set-dressing nouns ("obsidian", "brass", "amber",
#     "vintage 1940s") -- those flow through the FLUX bookend init image,
#     not the LTX motion prompt.
#
# BUG-LOCAL-008 status (CFG=1.0 erases negative): unchanged. The
# negative branch is still mathematically inert and still discarded.
# The new templates simply don't try to compensate via positive negation.
_PROMPT_BY_ROLE = {
    "announcer": (
        "Continuous shot, same console throughout. Tuning dial needle "
        "sweeps rhythmically. Vacuum tubes pulse. Brass speaker grille "
        "trembles. Dust motes drift. Slow handheld dolly forward."
    ),
    "music_open": (
        "Continuous shot, same console throughout. Dial whip-pans across "
        "frequencies. Tube filaments ignite from cold to white-hot. "
        "Speaker grille vibrates aggressively. Dynamic dolly push forward."
    ),
    "music_close": (
        "Continuous shot, same console throughout. Dial settles. Tube "
        "filaments cool from white through deep amber. Smoke trails "
        "from cooling tubes. Slow dolly pull back."
    ),
    "music_inter": (
        "Continuous shot, same console throughout. Dial steady, glowing. "
        "Oscilloscope dances to the rhythm. VU meters bounce. Tubes "
        "pulse with the bass. Slow orbit around the speaker."
    ),
    "sfx": (
        "Continuous shot, same console throughout. Snap zoom on the "
        "dial as needle spikes hard. Tubes surge with electric arcs. "
        "Speaker grille rattles violently. Quick whip-pan to the dial."
    ),
}

# BUG-LOCAL-112 (2026-05-06, round-robin verified): the LTX prompt
# is now BRUTALLY MOTION-CENTRIC. The prior builder (BUG-LOCAL-025
# Phase H 2026-05-03) prepended the per-episode style brief, the role
# template, the scene_env, and the style tone in one comma-joined
# string -- producing a 600-800 char prompt that diluted the motion
# verbs into a tableau of static set-dressing. The model interpreted
# this as "render a beautiful static set" rather than "make a
# continuous moving shot," and quantitative MAD analysis on four
# 2026-05-06 runs confirmed the output was mostly-static (1.86-5.92
# MAD between consecutive frames -- effectively still images with
# occasional scene-cut spikes).
#
# The fix is to feed LTX ONLY the motion-centric role template. Visual
# identity is already provided by the FLUX radio_bookend still via
# LTXVImgToVideoConditionOnly (the i2v anchor) -- repeating the
# aesthetic in the LTX prompt is wasted bandwidth that throttles motion.
#
# What still happens to the deferred context:
#   - line scene_env / episode style: not consumed by LTX anymore;
#     they remain available in the ledger for future per-line FLUX
#     anchoring or post-process color grading work. Removing them
#     here is reversible -- if motion lands clean and we want to
#     re-introduce minimal context, we can append a <40-char tail.
# (The former second bullet describing the legacy ltx_style_brief stamp
# was deleted at Sprint C C3b -- the stamp itself was retired in S31 B3
# and the new per-episode style brief lands via meta.story_brief at
# Sprint C C5a2 / C5e.)
_LTX_TOTAL_CHAR_BUDGET: int = 240
_LTX_MOTION_VERB_POSITION_CEILING: int = 140


def _build_ltx_role_prompt(role: str, line: dict, ledger: dict) -> str:
    """Return the per-role LTX motion prompt verbatim, optionally with
    the meta.story_brief LTX fragment appended.

    Sprint C C5e (2026-05-15): per refinement section 6.1 the LTX
    prompt carries a 80-100 char brief fragment derived from the
    post-script reflection pass. The fragment is appended AFTER the
    motion-centric role template so the motion verbs stay in the
    first 140 chars (motion-first rule). The brief is dropped if:

      * total prompt exceeds 240 chars (LTX budget), OR
      * the brief would push the first motion verb past char 140
        (degenerate template-growth case; structural proxy only).

    Structural proxy only per R-05: char-counting proves the prompt
    is well-formed, NOT that LTX actually renders better motion.
    Empirical LTX motion fidelity verification is Sprint A scope per
    ROADMAP.

    The ``line`` argument is accepted for API stability with the prior
    signature and is currently unused -- the brief is per-episode
    (ledger-meta) not per-line. Re-introducing per-line context is a
    one-line append change.
    """
    template = _PROMPT_BY_ROLE.get(role, _PROMPT_BY_ROLE["sfx"])

    # Resolve brief fragment via the C5b helper (max 90 chars, trimmed
    # at sentence/clause boundary, never mid-word).
    from ._otr_story_brief_helpers import (
        get_story_brief_ltx,
        get_story_brief_status,
    )
    meta = ledger.get("meta") if isinstance(ledger, dict) else None
    if not isinstance(meta, dict):
        meta = {}
    brief_fragment = get_story_brief_ltx(meta, max_chars=90)
    brief_status = get_story_brief_status(meta)

    if not brief_fragment:
        log.info(
            "[BatchLTXRender] role=%s story_brief_status=%s "
            "(no brief fragment; legacy prompt)",
            role, brief_status,
        )
        return template

    candidate = f"{template} {brief_fragment}".strip()
    if len(candidate) > _LTX_TOTAL_CHAR_BUDGET:
        log.info(
            "[BatchLTXRender] role=%s story_brief_status=%s "
            "(brief dropped: combined %d > %d budget)",
            role, brief_status, len(candidate), _LTX_TOTAL_CHAR_BUDGET,
        )
        return template

    # Motion-first check: confirm at least one motion-class verb appears
    # in the first 140 chars. The role templates all start with motion
    # ("Continuous shot ...", "Dynamic dolly ..."); this guard catches
    # degenerate future template growth.
    first_motion_idx = _first_motion_verb_index(candidate)
    if first_motion_idx > _LTX_MOTION_VERB_POSITION_CEILING:
        log.info(
            "[BatchLTXRender] role=%s story_brief_status=%s "
            "(brief dropped: motion verb at char %d > %d ceiling)",
            role, brief_status, first_motion_idx,
            _LTX_MOTION_VERB_POSITION_CEILING,
        )
        return template

    log.info(
        "[BatchLTXRender] role=%s story_brief_status=%s "
        "(brief appended; %d total chars; motion@%d)",
        role, brief_status, len(candidate), first_motion_idx,
    )
    return candidate


# Sprint C C5e motion-verb list. Small fixed set covering the verbs
# used in _PROMPT_BY_ROLE so the position check is deterministic.
_LTX_MOTION_VERBS: tuple[str, ...] = (
    "shot", "sweeps", "pulse", "pulses", "trembles", "drift", "drifts",
    "dolly", "whip-pans", "settle", "settles", "cool", "cools",
    "ignite", "ignites", "bounce", "bounces", "orbit", "orbits",
    "zoom", "zooms", "vibrate", "vibrates", "rattle", "rattles",
    "spike", "spikes", "surge", "surges", "dance", "dances",
)


def _first_motion_verb_index(text: str) -> int:
    """Return the lowest character index at which any motion verb
    appears in ``text`` (case-insensitive), or len(text) if none."""
    lower = text.lower()
    best = len(text)
    for verb in _LTX_MOTION_VERBS:
        idx = lower.find(verb)
        if idx != -1 and idx < best:
            best = idx
    return best


# Negative prompt: aggressively suppress face hallucination. LTX is
# generic motion (not face-locked like HuMo), so without this it might
# wander off the radio still and try to add a person.
_LTX_NEGATIVE = (
    "person, human, face, woman, man, hands, fingers, body, "
    "people in frame, ugly, deformed, low quality, jpeg artifacts, "
    "blurry, motion blur, talking, mouth, lips, smile"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ltx_length_for_dur(dur_s: float, *, fps: int = LTX_FPS) -> int:
    """Pick the smallest valid LTX ``length`` >= round(dur_s * fps).

    LTX VAE temporal compression requires length = 8n + 1 (vs HuMo's
    4n + 1). Floored at LTX_MIN_FRAMES, capped at LTX_MAX_FRAMES.

    Post BUG-LOCAL-091 (2026-05-04): LTX_MAX_FRAMES = 353 = ~14.12 s
    @ 25 fps. Long announcer lines that exceed the per-chunk widget
    cap go through the chunking dispatch in execute() rather than
    relying on the legacy clamp.
    """
    target = max(1, round(float(dur_s) * fps))
    n = (target - 1 + 7) // 8
    frames = 8 * n + 1
    if frames < LTX_MIN_FRAMES:
        frames = LTX_MIN_FRAMES
    if frames > LTX_MAX_FRAMES:
        frames = LTX_MAX_FRAMES
    return frames


def ltx_length_for_dur_uncapped(dur_s: float, *, fps: int = LTX_FPS) -> int:
    """Same as ``ltx_length_for_dur`` but without the LTX_MAX_FRAMES cap.

    BUG-LOCAL-091: used by the chunking dispatch to decide whether a
    line's audio duration exceeds the per-chunk ceiling. If the
    uncapped value > the user-configured chunk cap, the line is split
    into multiple chunks before rendering.
    """
    target = max(1, round(float(dur_s) * fps))
    n = (target - 1 + 7) // 8
    frames = 8 * n + 1
    if frames < LTX_MIN_FRAMES:
        frames = LTX_MIN_FRAMES
    return frames


def _concat_clips_via_ffmpeg(
    chunk_paths: list,
    out_path: Path,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """BUG-LOCAL-091: concat per-chunk LTX mp4 files into a single
    per-line mp4 via ffmpeg's concat demuxer.

    Same pattern as the BUG-LOCAL-086 helper in batch_humo_render.py;
    duplicated here rather than imported to keep the LTX render path
    self-contained. Each chunk goes through ``_save_video_mp4`` with the
    same fps + codec, so concat with ``-c copy`` is safe.

    Single-chunk case is a defensive copy if not already at out_path.
    """
    import os
    import shutil
    import subprocess
    import tempfile

    if not chunk_paths:
        raise RuntimeError("_concat_clips_via_ffmpeg: empty chunk list")
    if len(chunk_paths) == 1:
        if Path(chunk_paths[0]) == Path(out_path):
            return out_path
        shutil.copy2(str(chunk_paths[0]), str(out_path))
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    list_fd, list_path = tempfile.mkstemp(prefix="ltx_concat_", suffix=".txt")
    try:
        with os.fdopen(list_fd, "w", encoding="utf-8") as f:
            for p in chunk_paths:
                abs_p = str(Path(p).resolve()).replace("'", "'\\''")
                f.write(f"file '{abs_p}'\n")
        cmd = [
            ffmpeg, "-y",
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


def _make_boomerang_via_ffmpeg(
    mp4_path: Path,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """BUG-LOCAL-117d: in-place forward+reverse boomerang of an LTX clip.

    Strategy: split the input video into two streams, reverse one,
    drop the first frame of the reversed stream (which is the duplicate
    midpoint frame), then concat back. The output reads as a smooth
    oscillation that starts at frame 0 (radio_bookend), reaches peak
    motion at the midpoint, and returns to frame 0 by the end -- so
    end-of-clip-N == start-of-clip-N+1 -> seamless concat in
    VideoComposite. Pairs with the half-duration render strategy in
    execute(): chunk_ltx_length is computed off chunk_dur_s/2 so the
    boomerang doubles back to the full audio-timeline duration.

    Output frame count = 2 * input_frames - 1 (midpoint dropped).
    Both 8n+1 (LTX VAE valid) and (16n+1) outputs are also 8m+1.

    Note: ffmpeg's reverse filter loads the entire stream into RAM.
    For 832x480 @ 25fps clips up to ~14s (350 frames) the RAM cost is
    well under 1 GB -- a non-issue on the OTR build.
    """
    import os
    import subprocess
    import tempfile

    src = Path(mp4_path)
    if not src.is_file():
        raise FileNotFoundError(
            f"_make_boomerang_via_ffmpeg: input mp4 not found: {src}"
        )

    fd, tmp_path = tempfile.mkstemp(
        prefix="ltx_boomerang_", suffix=".mp4", dir=str(src.parent),
    )
    os.close(fd)
    tmp = Path(tmp_path)

    cmd = [
        ffmpeg, "-y",
        "-i", str(src),
        "-filter_complex",
        # Split video into two streams [a] and [b].
        # Reverse [b], trim its first frame (the duplicate midpoint),
        # reset PTS so concat aligns. Then concat [a] (forward) +
        # [r] (reversed-minus-midpoint) -> [out].
        "[0:v]split[a][b];"
        "[b]reverse,trim=start_frame=1,setpts=PTS-STARTPTS[r];"
        "[a][r]concat=n=2:v=1:a=0[out]",
        "-map", "[out]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "18",
        # BUG-LOCAL-117d hardening 2026-05-07: match the timebase
        # _save_video_mp4 stamps (12800) so concat-demuxer with -c copy
        # in _concat_clips_via_ffmpeg and VideoComposite's per-clip
        # silent encodes share an identical timestamp grid. Without
        # this, ffmpeg picks per-encode (typically 16k or 90k) and
        # concat-demuxer emits "Non-monotonous DTS" at the seam.
        "-video_track_timescale", "12800",
        "-movflags", "+faststart",
        "-an",  # explicit: no audio (LTX clips are silent)
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"ffmpeg boomerang failed (rc={proc.returncode}): "
            f"{(proc.stderr or '').strip()[:500]}"
        )

    if not tmp.is_file() or tmp.stat().st_size < 256:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"ffmpeg boomerang produced empty/missing output for {src}"
        )

    # In-place swap: replace the half-duration source with the
    # full-duration boomerang. Path stays the same so chunk_dest /
    # cache_key invariants hold.
    try:
        src.unlink()
    except OSError:
        pass
    tmp.replace(src)
    return src


_NODE_CACHE: dict[str, Any] = {}


def _node(name: str):
    """Fetch a ComfyUI node class by its registered name (lazy)."""
    if name not in _NODE_CACHE:
        from nodes import NODE_CLASS_MAPPINGS  # noqa: late import
        cls = NODE_CLASS_MAPPINGS.get(name)
        if cls is None:
            raise RuntimeError(
                f"OTR_BatchLTXRender: node '{name}' not found in "
                f"NODE_CLASS_MAPPINGS. Required nodes: CLIPTextEncode, "
                f"LTXVImgToVideoConditionOnly (BUG-LOCAL-095), "
                f"LTXVConditioning, RandomNoise, CFGGuider, "
                f"KSamplerSelect, SamplerCustomAdvanced, VAEDecodeTiled, "
                f"EmptyLTXVLatentVideo. "
                f"Are stock comfy_extras + ComfyUI-LTXVideo installed?"
            )
        _NODE_CACHE[name] = cls
    return _NODE_CACHE[name]


def _call(name: str, **kwargs):
    """Instantiate a ComfyUI node and call its FUNCTION method."""
    cls = _node(name)
    fn_name = getattr(cls, "FUNCTION", "execute")
    obj = cls()
    result = getattr(obj, fn_name)(**kwargs)
    # io.NodeOutput unwrap (modern ComfyUI nodes return io.NodeOutput).
    if hasattr(result, "args"):
        return result.args
    return result


def _png_to_image_tensor(png_path: Path):
    """Load a PNG file as a ComfyUI IMAGE tensor [1, H, W, C] in [0, 1] float32.

    Mirrors ``batch_humo_render._png_to_tensor`` so the loading
    semantics are identical between the two render nodes -- VideoComposite
    sees clips that were both fed the same image-shape at I2V time.
    """
    from PIL import Image  # type: ignore
    import numpy as np  # type: ignore
    import torch  # type: ignore
    img = Image.open(png_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr)[None, ...]  # [1, H, W, C]
    return tensor


def _resolve_radio_still_path(ledger: dict | None) -> Path | None:
    """Locate the radio_bookend.png for this episode.

    Mirrors ``batch_humo_render._resolve_radio_still_path`` exactly so
    the two render nodes find the same file via the same fallback
    chain (top-level -> meta -> filesystem-by-episode_id).
    """
    if not isinstance(ledger, dict):
        return None

    def _is_valid(p: Path) -> bool:
        try:
            if not p.is_file():
                return False
            return p.stat().st_size >= 256
        except Exception:  # noqa: BLE001
            return False

    cand = ledger.get("radio_bookend_path")
    if not cand:
        meta = ledger.get("meta") or {}
        if isinstance(meta, dict):
            cand = meta.get("radio_bookend_path")
    if cand:
        try:
            p = Path(cand)
        except Exception:  # noqa: BLE001
            p = None
        if p is not None and _is_valid(p):
            return p

    # Filesystem fallback by episode_id.
    eid_raw = ledger.get("episode_id")
    if not isinstance(eid_raw, str):
        return None
    eid = eid_raw.strip()
    if not eid or any(t in eid for t in ("/", "\\", "..", "\x00")):
        return None
    try:
        # BUG-LOCAL-028 fix (2026-05-03): pass episode_id so the lookup
        # resolves to ``output/otr/episodes/<eid>/stills/radio_bookend_<eid>.png``
        # (the canonical Phase B per-episode workspace). Prior to this
        # fix, ``otr_stills_dir()`` with no arg fell back to
        # ``output/otr/_legacy_stills/`` per ``nodes/_otr_paths.py:208-218``;
        # after BUG-028's writer fix, the radio bookend now lands in the
        # per-episode dir and this READ site needs to point there too,
        # otherwise LTX can't find the radio still and falls back to a
        # generic motion clip with no scene continuity.
        fs_path = otr_stills_dir(eid) / f"radio_bookend_{eid}.png"
    except Exception as exc:  # noqa: BLE001
        log.warning("[BatchLTXRender] otr_stills_dir lookup failed: %s", exc)
        return None
    if _is_valid(fs_path):
        return fs_path
    return None


def _save_video_mp4(
    *,
    images,
    out_path: Path,
    fps: int,
    ffmpeg: str = "ffmpeg",
) -> Path:
    """Save an IMAGE tensor [N, H, W, C] in [0,1] float as a silent
    mp4 to ``out_path`` via ffmpeg. Uses h264 yuv420p libx264 to match
    the encoding profile HuMo clips use, so VideoComposite's
    concat-demuxer can ``-c copy`` cleanly across both sources.

    No audio -- LTX clips are video-only. VideoComposite muxes the
    master mix at the final stage with ``-c:a copy``.
    """
    import numpy as np  # type: ignore
    import subprocess
    import tempfile
    import torch  # type: ignore

    if hasattr(images, "detach"):
        arr = images.detach().cpu().numpy()
    else:
        arr = np.asarray(images)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    n_frames, h, w, _ = arr.shape

    tmp = Path(tempfile.mkdtemp(prefix="otr_ltx_clip_"))
    try:
        # Pipe raw frames to ffmpeg via stdin.
        cmd = [
            ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}",
            "-r", str(int(fps)),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-preset", "fast",
            # Match HuMo container timebase (Jeffrey review 2026-05-01,
            # video_composite.py BUG-129a static-fill comment).
            "-video_track_timescale", "12800",
            "-an",
            str(out_path),
        ]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        proc.stdin.write(arr.tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ffmpeg failed encoding LTX clip: {stderr[:500]}"
            )
    finally:
        # Clean up the tempdir even on success.
        try:
            for f in tmp.iterdir():
                f.unlink()
            tmp.rmdir()
        except Exception:  # noqa: BLE001
            pass
    return out_path


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class BatchLTXRender:
    """Render N LTX-2 video clips in one graph execution for non-character
    ledger lines (announcer / music_* / sfx).

    OUTPUT_NODE = True so ComfyUI doesn't prune this side-effect node
    when downstream consumers don't fully chain (lesson from
    BUG-LOCAL-077).
    """

    CATEGORY = "OTR/v2/Visual"
    OUTPUT_NODE = True
    FUNCTION = "execute"
    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("clips_dir", "clip_count", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "LTX-2 model (CheckpointLoaderSimple loading "
                               "ltx-video-2b-v0.9.x.safetensors)",
                }),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "ledger_json": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "Production ledger -- inline JSON or a path "
                               "to *_ledger.json. Walked for non-character "
                               "speaker_role lines.",
                }),
                "seed": ("INT", {
                    "default": 1,
                    "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                }),
            },
            "optional": {
                "ffmpeg": ("STRING", {
                    "default": "ffmpeg",
                    "tooltip": "ffmpeg binary path or PATH-resolvable name",
                }),
                "humo_clips_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "DAG SEQUENCING EDGE -- not data. Wire this to "
                        "BatchHumoRender.clips_dir (or .report) so ComfyUI "
                        "waits for HuMo to finish writing its per-line .mp4 "
                        "files before LTX starts loading. HuMo's strict "
                        "teardown then releases the 16.5 GB MODEL before "
                        "LTX claims VRAM. The value is intentionally NOT "
                        "used by execute() -- if you remove this input "
                        "thinking it's dead code, ComfyUI may schedule "
                        "LTX too early and OOM. Round-robin consult "
                        "2026-05-02 endorsed this pattern as acceptable "
                        "ComfyUI sequencing."
                    ),
                }),
                # BUG-LOCAL-097 (2026-05-04 EVENING): clip_length appended
                # at the END of optional. BUG-091 originally put it FIRST,
                # which inserted a new widget at position [3] of
                # widgets_values and shifted every subsequent saved value
                # by one slot -- existing workflow JSONs (otr_scifi_16gb_full.json
                # and any user-saved variants) read "ffmpeg" as FLOAT
                # clip_length and the workflow validation crashed at
                # `Failed to convert an input value to a FLOAT value:
                # clip_length, ffmpeg, could not convert string to float:
                # 'ffmpeg'` BEFORE the LTX node could even run. Moving
                # clip_length to the END means new widget appears at
                # position [5], past the existing 5 saved values, so old
                # workflows fall through to the FLOAT default (7.0)
                # cleanly. Backward-compat preserved.
                "clip_length": ("FLOAT", {
                    "default": 22.0,
                    "min": 1.32,
                    "max": 28.16,
                    "step": 0.04,
                    "tooltip": (
                        "Max per-CHUNK duration in seconds (BUG-LOCAL-091). "
                        "Lines whose audio exceeds this are split into N "
                        "consecutive chunks rendered against the radio "
                        "bookend, then ffmpeg-concat into the final per-line "
                        "mp4. Default 22.0s (BUG-LOCAL-117e): empirical "
                        "mega-test 2026-05-07 PM rendered 25s @ 832x480 "
                        "cleanly on 5080 16 GB + 64 GB RAM. 22s leaves a 3s "
                        "safety headroom. Combined with BUG-LOCAL-117d "
                        "boomerang (default ON) the actual sample window is "
                        "11s -- well within the 14.5 GB VRAM ceiling. Cap "
                        "28.16s (705 frames, 8*88+1) is the absolute hardware "
                        "ceiling. Coherence past 25s is unverified; watch "
                        "for temporal repetition if you push toward 28s."
                    ),
                }),
            },
        }

    def execute(self, model, clip, vae, ledger_json, seed=1,
                ffmpeg="ffmpeg", humo_clips_dir="",
                clip_length=22.0):
        # NOTE: ``humo_clips_dir`` is intentionally consumed but unused.
        # See INPUT_TYPES tooltip -- it is a pure DAG sequencing edge so
        # ComfyUI schedules this node after BatchHumoRender finishes its
        # render + teardown. Do NOT remove it; doing so will let the LTX
        # checkpoint load race HuMo's 16.5 GB MODEL and OOM on 16 GB.
        del humo_clips_dir  # explicit: value ignored, edge is the contract
        t_start = time.time()
        report_lines: list[str] = []

        # ----------------------------------------------------------------
        # BUG-LOCAL-117: engine selector + dep check + loud banner
        # ----------------------------------------------------------------
        # 2026-05-08 round-robin Item 3.2 catch: the production
        # workflow JSON loads `ltx-2.3-22b-dev.safetensors` + the
        # Gemma 3 12B encoder, both v2.3-path artifacts. But this
        # node defaults to v0_9 (the smoke-test path) when the env
        # var is unset. The two paths have different sampler chains,
        # different VAE wiring, and different latent shapes -- silent
        # mismatch produces garbage output OR a deep-stack tensor
        # shape error, not a clean failure.
        env_engine_raw = os.environ.get("OTR_LTX_ENGINE")
        engine = (env_engine_raw or OTR_LTX_ENGINE_DEFAULT).strip().lower()
        if engine not in ("v0_9", "v2_3"):
            log.warning(
                "[BatchLTXRender] BUG-LOCAL-117: unknown OTR_LTX_ENGINE=%r; "
                "falling back to default %r",
                engine, OTR_LTX_ENGINE_DEFAULT,
            )
            engine = OTR_LTX_ENGINE_DEFAULT
        if env_engine_raw is None:
            # Highly visible warning so the operational env-var
            # fix gets a reminder at every soak. Without this, a
            # user running production for the first time gets the
            # silent v0_9-vs-v2.3 mismatch with no signal.
            log.warning(
                "[BatchLTXRender] BUG-LOCAL-117 / 2026-05-08 round-robin: "
                "OTR_LTX_ENGINE env var is UNSET; defaulting to %r. "
                "The production workflow JSON is rigged for v2.3 (22B "
                "checkpoint + Gemma encoder). If the loaded checkpoint "
                "is the LTX 2.3 22B file, set OTR_LTX_ENGINE=v2_3 in "
                "your launch environment OR Windows User env vars "
                "(HKCU\\Environment) before queueing. Smoke-test runs "
                "on the LTX 2B v0.9 checkpoint can leave it unset.",
                OTR_LTX_ENGINE_DEFAULT,
            )

        # 2026-05-08 round-robin Item 3.2 defensive guard: if a future
        # checkpoint loader stamps `_otr_ckpt_name` on the model
        # object, fail FAST when the resolved engine doesn't match the
        # checkpoint family. Today this is forward-compat -- the stock
        # CheckpointLoaderSimple doesn't stamp this attribute, so the
        # guard is a no-op and degrades to the env-var warning above.
        # Wired so a later `LowVRAMCheckpointLoader` patch can add the
        # stamp and have this guard activate without further changes.
        try:
            ckpt_name = getattr(model, "_otr_ckpt_name", "") or ""
            ckpt_lower = str(ckpt_name).lower()
            if engine == "v0_9" and ckpt_lower:
                if "ltx-2.3" in ckpt_lower or "22b" in ckpt_lower:
                    raise RuntimeError(
                        f"BatchLTXRender engine/checkpoint mismatch: "
                        f"OTR_LTX_ENGINE=v0_9 (default if unset) but a "
                        f"v2.3-family checkpoint was loaded "
                        f"({ckpt_name!r}). Either set "
                        f"OTR_LTX_ENGINE=v2_3 in the launch environment "
                        f"OR swap Node 54's checkpoint widget back to "
                        f"the LTX 2B v0.9 file. The two engines have "
                        f"incompatible sampler / VAE / latent-shape "
                        f"contracts; running them mismatched produces "
                        f"garbage output or a tensor shape crash."
                    )
            if engine == "v2_3" and ckpt_lower:
                if (
                    "ltx-2-0.9.safetensors" in ckpt_lower
                    or ckpt_lower.endswith("/ltx-video-2b-v0.9.safetensors")
                    or "ltx-video-2b-v0.9" in ckpt_lower
                ):
                    raise RuntimeError(
                        f"BatchLTXRender engine/checkpoint mismatch: "
                        f"OTR_LTX_ENGINE=v2_3 but the LTX 2B v0.9 "
                        f"checkpoint was loaded ({ckpt_name!r}). "
                        f"Either unset OTR_LTX_ENGINE OR swap Node 54 "
                        f"to the v2.3 22B file."
                    )
        except RuntimeError:
            raise
        except Exception:  # noqa: BLE001 -- guard is defensive only
            # Any failure inside the guard (missing attribute, model
            # is a non-standard wrapper, etc.) means the guard can't
            # validate -- fall through silently rather than escalating
            # the recovery into a fault.
            pass

        # BUG-LOCAL-117c: end-frame anchor strength for seamless loops.
        # Default 0.0 = off (no behavior change). Set to 0.4 to test
        # chunk-boundary smoothing.
        try:
            end_anchor_strength = float(
                os.environ.get(
                    "OTR_LTX_END_ANCHOR_STRENGTH",
                    str(LTX_END_ANCHOR_STRENGTH_DEFAULT),
                )
            )
        except (TypeError, ValueError):
            end_anchor_strength = LTX_END_ANCHOR_STRENGTH_DEFAULT
        end_anchor_strength = max(0.0, min(1.0, end_anchor_strength))

        # BUG-LOCAL-117d: ffmpeg boomerang post-process. Default ON
        # 2026-05-07 PM so non-character chunks render half-duration
        # and the post-process doubles them back to full audio target,
        # producing a perfect radio-still -> motion -> radio-still loop
        # without stacked-anchor glitching. Halves sample wall time too.
        # Set OTR_LTX_LOOP_VIA_REVERSE=off for the legacy full-duration
        # render path (e.g. for A/B comparison or smoke debugging).
        loop_via_reverse_raw = os.environ.get(
            "OTR_LTX_LOOP_VIA_REVERSE", LTX_LOOP_VIA_REVERSE_DEFAULT
        ).strip().lower()
        loop_via_reverse = loop_via_reverse_raw in LTX_LOOP_VIA_REVERSE_TRUTHY

        # BUG-LOCAL-117a Phase 1: resolve sampler for v0_9 path from env var.
        # Validate against allowlist; fall back to default with a warning if
        # the user typo'd. Logged into the banner so the running config is
        # always visible at queue time.
        v0_9_sampler_name = os.environ.get(
            "OTR_LTX_V0_9_SAMPLER_NAME", LTX_V0_9_SAMPLER_NAME_DEFAULT
        ).strip().lower()
        if v0_9_sampler_name not in LTX_V0_9_SAMPLER_NAMES_VALID:
            log.warning(
                "[BatchLTXRender] BUG-LOCAL-117a: unknown OTR_LTX_V0_9_SAMPLER_NAME=%r; "
                "valid values: %s. Falling back to default %r",
                v0_9_sampler_name, LTX_V0_9_SAMPLER_NAMES_VALID,
                LTX_V0_9_SAMPLER_NAME_DEFAULT,
            )
            v0_9_sampler_name = LTX_V0_9_SAMPLER_NAME_DEFAULT

        log.info("=" * 64)
        log.info("[BatchLTXRender] BUG-LOCAL-117 engine=%s", engine)
        if engine == "v2_3":
            log.info("[BatchLTXRender]   role:     QUALITY OPT-IN (requires 64 GB system RAM)")
            log.info("[BatchLTXRender]   model:    LTX 2.3 22B-dev BF16 + distilled LoRA x2 (0.5, 0.2)")
            log.info("[BatchLTXRender]   encoder:  LTXAVTextEncoderLoader -> Gemma FP4 mixed")
            log.info("[BatchLTXRender]   sampler:  ClownSampler_Beta (%s, eta=%.2f, bongmath=%s)",
                     LTX_V2_3_CLOWN_SAMPLER_NAME, LTX_V2_3_CLOWN_ETA,
                     LTX_V2_3_CLOWN_BONGMATH)
            log.info("[BatchLTXRender]   guider:   MultimodalGuider + GuiderParameters (VIDEO cfg=%.1f stg=%.1f)",
                     LTX_V2_3_VIDEO_CFG, LTX_V2_3_VIDEO_STG)
            log.info("[BatchLTXRender]   decode:   LTXVTiledVAEDecode (LTX-specific tiled)")
            log.info("[BatchLTXRender]   sigmas:   LTX_DISTILLED_SIGMAS (9 vals, float32, CPU)")
            log.info("[BatchLTXRender]   note:     ~6-8 min/clip; expect swap thrashing on <64 GB RAM systems")
        else:
            log.info("[BatchLTXRender]   role:     SAFE DEFAULT (works on 16 GB VRAM + 32 GB RAM)")
            log.info("[BatchLTXRender]   model:    LTX checkpoint (model widget) -- 2.3 dev recommended; 2B v0.9 still loads")
            log.info("[BatchLTXRender]   encoder:  CLIP from input -- Gemma (LTXAVTextEncoderLoader) recommended; t5xxl tolerated")
            log.info("[BatchLTXRender]   sampler:  KSamplerSelect(%r)  <-- OTR_LTX_V0_9_SAMPLER_NAME",
                     v0_9_sampler_name)
            log.info("[BatchLTXRender]   guider:   CFGGuider (cfg=%.1f)", LTX_CFG)
            log.info("[BatchLTXRender]   decode:   VAEDecodeTiled")
            log.info("[BatchLTXRender]   sigmas:   LTX_DISTILLED_SIGMAS (9 vals, 8 sampling steps)")
            log.info("[BatchLTXRender]   note:     ~1.5-2.5 min/clip; visually indistinguishable from v2_3 on OTR content")
        if end_anchor_strength > 0.0:
            log.info("[BatchLTXRender]   loop:     END-ANCHOR ON at strength=%.2f "
                     "(LTXVImgToVideoInplaceKJ rigid latent-replace, frame -1)",
                     end_anchor_strength)
        else:
            log.info("[BatchLTXRender]   loop:     end-anchor OFF (set OTR_LTX_END_ANCHOR_STRENGTH=0.7 to test rigid pin)")
        if loop_via_reverse:
            log.info("[BatchLTXRender]   boomerang: ON (BUG-LOCAL-117d) -- "
                     "render HALF chunk_dur_s, ffmpeg-reverse-and-concat "
                     "doubles back to full audio target")
        else:
            log.info("[BatchLTXRender]   boomerang: OFF (set OTR_LTX_LOOP_VIA_REVERSE=on for seamless loops + half wall time)")
        log.info("=" * 64)
        report_lines.append(
            f"BatchLTXRender: engine={engine}"
        )

        # Fail-fast dep check on v2_3 -- if RES4LYF or LTXVideo nodes
        # missing, refuse to start so we never silently fall through.
        if engine == "v2_3":
            try:
                from nodes import NODE_CLASS_MAPPINGS  # noqa: late import
            except ImportError:
                NODE_CLASS_MAPPINGS = {}  # type: ignore[assignment]
            missing = [
                name for name in LTX_V2_3_REQUIRED_NODES
                if name not in NODE_CLASS_MAPPINGS
            ]
            if missing:
                raise RuntimeError(
                    "BatchLTXRender: OTR_LTX_ENGINE=v2_3 requires custom node "
                    f"pack(s) but these node classes are missing: {missing}. "
                    "Install RES4LYF (https://github.com/ClownsharkBatwing/RES4LYF) "
                    "into custom_nodes/ and ensure ComfyUI-LTXVideo is updated. "
                    "Or set OTR_LTX_ENGINE=v0_9 for legacy rollback."
                )

        # ----------------------------------------------------------------
        # 1. Load the ledger + resolve the radio bookend png
        # ----------------------------------------------------------------
        ledger, ledger_path = self._load_ledger(ledger_json)
        if ledger is None:
            raise RuntimeError(
                "BatchLTXRender: ledger could not be loaded from inline "
                "JSON or path"
            )

        episode_id = str(ledger.get("episode_id") or "episode")
        report_lines.append(
            f"BatchLTXRender: episode={episode_id}"
        )

        radio_png = _resolve_radio_still_path(ledger)
        if radio_png is None:
            report_lines.append(
                "  WARNING: no radio_bookend.png on disk -- "
                "non-character lines will fall through to "
                "VideoComposite static-radio fallback (BUG-129a) "
                "with no I2V animation. Proceed anyway."
            )
            log.warning(
                "[BatchLTXRender] no radio bookend resolved for episode "
                "%s -- skipping LTX render entirely",
                episode_id,
            )
            return (
                str(otr_videos_dir(episode_id)), 0, "\n".join(report_lines)
            )

        try:
            ref_image = _png_to_image_tensor(radio_png)
        except Exception as exc:
            raise RuntimeError(
                f"BatchLTXRender: failed to load radio bookend png "
                f"{radio_png}: {exc}"
            )

        report_lines.append(
            f"  radio_bookend: {radio_png.name} "
            f"({radio_png.stat().st_size} bytes)"
        )

        # ----------------------------------------------------------------
        # 2. Walk ledger.lines[] for non-character roles
        # ----------------------------------------------------------------
        lines = ledger.get("lines") or []
        plan: list[dict] = []
        for ln in lines:
            line_id = str(ln.get("line_id") or "")
            if not line_id:
                continue
            speaker_role = resolve_speaker_role(ln)
            if speaker_role == SPEAKER_ROLE_CHARACTER:
                continue  # HuMo will render this in the next phase
            if not is_never_humo_role(speaker_role):
                continue  # safety: only render roles we're sure about
            dur_s = ln.get("dur_s")
            if not isinstance(dur_s, (int, float)) or float(dur_s) <= 0.0:
                continue
            dur_s = float(dur_s)

            # BUG-LOCAL-091 (2026-05-04): chunking dispatch. ``clip_length``
            # is the max single-pass duration (workflow widget, default 7.0s).
            # Lines whose audio exceeds it are split into N consecutive
            # chunks of <= clip_length each, rendered independently against
            # the radio bookend, then ffmpeg-concat into the final per-line
            # mp4. Pre-091 the cap silently truncated the back half of any
            # >7s line, leaving the radio scene frozen while the announcer
            # audio kept playing.
            chunk_max_dur_s = max(0.04, float(clip_length))
            if dur_s <= chunk_max_dur_s:
                chunk_specs = [{"dur_s": dur_s}]
            else:
                n_chunks = max(2, math.ceil(dur_s / chunk_max_dur_s))
                chunk_dur_s = dur_s / n_chunks
                chunk_specs = [{"dur_s": chunk_dur_s} for _ in range(n_chunks)]
                log.info(
                    "[BatchLTXRender] BUG-LOCAL-091: line %s dur_s=%.2fs > "
                    "clip_length=%.2fs -- splitting into %d chunks of "
                    "%.2fs each",
                    line_id, dur_s, float(clip_length),
                    n_chunks, chunk_dur_s,
                )
            ltx_length = ltx_length_for_dur(chunk_specs[0]["dur_s"])
            # BUG-LOCAL-025 (Phase H): enrich per-role base with line's
            # scene context + episode style instead of using the bare
            # hardcoded role prompt. Same role across two episodes
            # now produces visibly different motion intent.
            prompt_text = _build_ltx_role_prompt(speaker_role, ln, ledger)
            plan.append({
                "line_id": line_id,
                "speaker_role": speaker_role,
                "dur_s": dur_s,
                "ltx_length": ltx_length,
                "chunks": chunk_specs,
                "prompt_text": prompt_text,
            })

        if not plan:
            report_lines.append(
                "  no non-character lines to render -- LTX phase skipped"
            )
            log.info(
                "[BatchLTXRender] no non-character lines in ledger; "
                "skipping render entirely"
            )
            return (
                str(otr_videos_dir(episode_id)), 0, "\n".join(report_lines)
            )

        report_lines.append(
            f"  plan: {len(plan)} clip(s) "
            f"({sum(1 for e in plan if e['speaker_role']=='announcer')} announcer + "
            f"{sum(1 for e in plan if e['speaker_role'].startswith('music'))} music + "
            f"{sum(1 for e in plan if e['speaker_role']=='sfx')} sfx)"
        )

        # ----------------------------------------------------------------
        # 3. Pre-bake shared LTX objects (sampler, sigmas, base negative)
        # ----------------------------------------------------------------
        import torch  # type: ignore

        # BUG-LOCAL-117a Phase 1: pre-bake the v0_9 sampler from the
        # env-resolved name (default euler_cfg_pp). If the requested sampler
        # isn't registered with KSamplerSelect (e.g. ComfyUI core too old
        # for cfg_pp variants, or RES4LYF disabled), fall back through:
        # res_multistep_cfg_pp -> euler_cfg_pp -> euler.
        sampler_obj = None
        v0_9_sampler_resolved = v0_9_sampler_name
        for _candidate in (
            v0_9_sampler_name,
            "euler_cfg_pp",
            "euler",
        ):
            try:
                sampler_obj = _call("KSamplerSelect", sampler_name=_candidate)[0]
                v0_9_sampler_resolved = _candidate
                if _candidate != v0_9_sampler_name:
                    log.warning(
                        "[BatchLTXRender] %r unavailable; using %r instead",
                        v0_9_sampler_name, _candidate,
                    )
                break
            except Exception as _exc:
                log.warning(
                    "[BatchLTXRender] KSamplerSelect(%r) failed: %s",
                    _candidate, _exc,
                )
        if sampler_obj is None:
            raise RuntimeError(
                "BatchLTXRender: KSamplerSelect could not resolve any sampler "
                "from the fallback chain. ComfyUI core may be broken."
            )
        log.info(
            "[BatchLTXRender] v0_9 sampler resolved: %r (requested %r)",
            v0_9_sampler_resolved, v0_9_sampler_name,
        )
        sigmas = torch.tensor(LTX_DISTILLED_SIGMAS, dtype=torch.float32)

        neg_tokens = clip.tokenize(_LTX_NEGATIVE)
        base_negative = clip.encode_from_tokens_scheduled(neg_tokens)

        # BUG-LOCAL-117a HOTFIX 2026-05-07: audio VAE for the v2_3 AV-joint
        # path. MultimodalGuider expects packed (video+audio) latents, so we
        # synthesize a zero-filled audio stub via LTXVEmptyLatentAudio (which
        # needs the audio_vae for shape inference) per chunk. The audio VAE
        # is loaded once here so the per-chunk loop just reads it. Memory
        # cost is small relative to the 22B transformer; the audio_vae model
        # adds < 0.5 GB.
        ltx_audio_vae = None
        if engine == "v2_3":
            try:
                ltx_audio_vae = _call(
                    "LTXVAudioVAELoader",
                    ckpt_name="ltx-2.3-22b-dev.safetensors",
                )[0]
                log.info(
                    "[BatchLTXRender] v2_3: loaded audio_vae from "
                    "ltx-2.3-22b-dev.safetensors for AV-joint stub"
                )
            except Exception as exc:
                raise RuntimeError(
                    "BatchLTXRender: OTR_LTX_ENGINE=v2_3 requires the audio "
                    "VAE from ltx-2.3-22b-dev.safetensors but loading via "
                    f"LTXVAudioVAELoader failed: {exc}. Verify the file is at "
                    "C:\\\\ComfyUI-Models\\\\checkpoints\\\\ltx-2.3-22b-dev.safetensors."
                )

        # ----------------------------------------------------------------
        # 4. Per-line render loop
        # ----------------------------------------------------------------
        clips_dir = otr_videos_dir(episode_id)
        clips_dir.mkdir(parents=True, exist_ok=True)
        rendered_clips: list[dict] = []

        # BUG-LOCAL-117b 2026-05-07: render-cache to dedupe identical-input
        # chunks within an episode. Per Jeffrey: "if the inputs are the same
        # we need to change that." Most common case: a music_open beat split
        # into 2 chunks (BUG-LOCAL-091) renders chunk 1 + chunk 2 with same
        # role (music_open) + same anchor (radio_bookend) + same prompt
        # (_PROMPT_BY_ROLE) + same ltx_length. Truly identical inputs ->
        # truly identical output. Cache hit copies the canonical mp4 to
        # the per-line/chunk filename instead of rendering again.
        # Saves ~40-50% wall time on a typical episode (cargo_hold smoke =
        # 5 chunks -> 3 unique renders).
        render_cache: dict[tuple, Path] = {}

        try:
            import comfy.model_management as mm  # type: ignore
            mm.load_models_gpu([model])
        except Exception as exc:
            log.debug("[BatchLTXRender] pre-pin model skipped: %s", exc)

        with torch.inference_mode():
            for idx, entry in enumerate(plan):
                line_id = entry["line_id"]
                prompt_text = entry["prompt_text"]
                chunks = entry.get("chunks") or [{"dur_s": entry["dur_s"]}]
                n_chunks = len(chunks)
                shot_t0 = time.time()

                # Anti-clobber (consult 2026-05-02 ChatGPT + Gemini):
                # HuMo writes <line_id>.mp4 for character lines; LTX writes
                # <line_id>.mp4 for non-character lines into the SAME dir.
                # Role filter (is_never_humo_role) should keep them disjoint,
                # but if a future ledger has a duplicate or drifted role and
                # LTX overwrites a HuMo character clip, the failure would be
                # invisible until final composite. Skip pre-existing files.
                pre_existing = clips_dir / f"{line_id}.mp4"
                if pre_existing.exists() and pre_existing.stat().st_size > 0:
                    # BUG-LOCAL-117f 2026-05-07: duration-aware anti-clobber.
                    # The pre-117d skip-if-exists check left half-duration
                    # boomerang-failed clips on disk through reruns. Now we
                    # ffprobe and compare to the audio-target duration; if
                    # the on-disk clip is materially short, delete it and
                    # fall through to a fresh render. Tolerance is 0.25s
                    # which exceeds VideoComposite's BUG-031 sync tolerance
                    # so we never re-render a clip the composite would have
                    # accepted as-is.
                    _expected_dur_s = float(entry["dur_s"])
                    _actual_dur_s = 0.0
                    try:
                        from . import _otr_probe as _PROBE  # type: ignore
                        _probed = _PROBE.probe_duration_s(pre_existing)
                        if _probed and _probed > 0.0:
                            _actual_dur_s = float(_probed)
                    except Exception as _probe_exc:  # noqa: BLE001
                        log.warning(
                            "[BatchLTXRender] %s: ffprobe on pre_existing "
                            "failed (%s); falling back to size-only skip",
                            line_id, _probe_exc,
                        )

                    _stale = (
                        _actual_dur_s > 0.0
                        and _actual_dur_s < _expected_dur_s - 0.25
                    )
                    if _stale:
                        log.warning(
                            "[BatchLTXRender] %s: pre_existing %s is stale "
                            "(actual=%.2fs, expected=%.2fs, gap=%.2fs). "
                            "Deleting and re-rendering.",
                            line_id, pre_existing.name,
                            _actual_dur_s, _expected_dur_s,
                            _expected_dur_s - _actual_dur_s,
                        )
                        try:
                            pre_existing.unlink()
                        except OSError as _del_exc:
                            log.error(
                                "[BatchLTXRender] %s: cannot unlink stale "
                                "%s (%s). Refusing to overwrite -- skipping.",
                                line_id, pre_existing.name, _del_exc,
                            )
                            report_lines.append(
                                f"  {line_id} ({entry['speaker_role']}): "
                                f"STALE-LOCKED {pre_existing.name}"
                            )
                            continue
                        report_lines.append(
                            f"  {line_id} ({entry['speaker_role']}): "
                            f"RE-RENDER (stale {_actual_dur_s:.2f}s "
                            f"-> expected {_expected_dur_s:.2f}s)"
                        )
                        # Fall through to render path below (no continue).
                    else:
                        log.warning(
                            "[BatchLTXRender] %s: skip -- existing clip on disk "
                            "(%s, %d bytes, dur=%.2fs). Refusing to overwrite.",
                            line_id, pre_existing.name,
                            pre_existing.stat().st_size, _actual_dur_s,
                        )
                        report_lines.append(
                            f"  {line_id} ({entry['speaker_role']}): "
                            f"SKIP existing {pre_existing.name}"
                        )
                        continue

                try:
                    pos_tokens = clip.tokenize(prompt_text)
                    positive = clip.encode_from_tokens_scheduled(pos_tokens)

                    # Set frame_rate via LTXVConditioning (must be 25 to
                    # match HuMo at concat time, Gemini round-robin
                    # 2026-05-01 catch).
                    cond_pos, cond_neg = _call(
                        "LTXVConditioning",
                        positive=positive, negative=base_negative,
                        frame_rate=LTX_FPS,
                    )

                    # BUG-LOCAL-091: per-chunk render loop. Single-chunk
                    # path (n_chunks==1) renders one pass and writes
                    # directly to <line_id>.mp4 -- matches pre-091 layout.
                    # Multi-chunk renders each chunk to a part file then
                    # ffmpeg-concats. All chunks share the same prompt +
                    # ref_image (radio bookend) so the visual snap at the
                    # chunk boundary is the radio scene resetting; the
                    # alternative (carry last frame to next chunk) is a
                    # future upgrade documented as BUG-LOCAL-091a.
                    chunk_mp4_paths: list = []
                    for chunk_idx, chunk in enumerate(chunks):
                        chunk_dur_s = float(chunk["dur_s"])
                        # BUG-LOCAL-117d: when boomerang is ON, render
                        # HALF the audio-timeline duration. The ffmpeg
                        # post-process below mirrors+concats the clip
                        # back to chunk_dur_s. This halves sample wall
                        # time AND makes loop endpoints exactly the
                        # radio_bookend (frame 0 == frame N) so the
                        # chunk-boundary snap disappears. ltx_length_for_dur
                        # rounds up to the nearest 8n+1, so the post-process
                        # output (2*half - 1) is also 8n+1 valid.
                        if loop_via_reverse:
                            chunk_render_dur_s = chunk_dur_s / 2.0
                        else:
                            chunk_render_dur_s = chunk_dur_s
                        chunk_ltx_length = ltx_length_for_dur(chunk_render_dur_s)

                        # BUG-LOCAL-117b 2026-05-07: cache key per identical
                        # input set. Same role -> same prompt (via
                        # _PROMPT_BY_ROLE). Same anchor (radio_bookend per
                        # episode). Same ltx_length -> same output.
                        # Engine + sampler are also part of the cache because
                        # if the user reruns with a different sampler we want
                        # FRESH renders, not stale cached ones.
                        if engine == "v2_3":
                            _cache_sampler_id = "v2_3_clownsampler_res_2s"
                        else:
                            _cache_sampler_id = f"v0_9_{v0_9_sampler_resolved}"
                        # Include end_anchor + boomerang in key so toggling
                        # either one doesn't serve stale cached renders.
                        # The boomerang flag matters because cache stores
                        # the POST-boomerang full-duration mp4 -- a cache
                        # built with boomerang=on is not safe to serve
                        # to a later run with boomerang=off (the file is
                        # already mirrored).
                        cache_key = (
                            entry["speaker_role"],
                            int(chunk_ltx_length),
                            _cache_sampler_id,
                            round(end_anchor_strength, 3),
                            "boomerang" if loop_via_reverse else "linear",
                        )

                        # Per-chunk destination resolved upfront (used by
                        # both the cache-hit copy path and the live-render
                        # path, plus the n_chunks==1 single-line path).
                        if n_chunks == 1:
                            chunk_dest = clips_dir / f"{line_id}.mp4"
                        else:
                            chunk_dest = clips_dir / (
                                f"{line_id}__chunk{chunk_idx + 1:02d}.mp4"
                            )

                        # Cache hit: identical inputs already rendered for
                        # an earlier line/chunk in this episode. Copy the
                        # canonical mp4 to chunk_dest. Eliminates a full
                        # 5-8 min sample+decode pass per duplicate.
                        if cache_key in render_cache:
                            import shutil as _shutil
                            _src = render_cache[cache_key]
                            try:
                                _shutil.copy2(str(_src), str(chunk_dest))
                                log.info(
                                    "[BatchLTXRender] cache HIT for %s: "
                                    "copied %s -> %s (key=%s)",
                                    line_id, _src.name, chunk_dest.name,
                                    cache_key,
                                )
                                chunk_mp4_paths.append(chunk_dest)
                                continue  # skip render -- chunk_dest is satisfied
                            except Exception as _copy_exc:
                                log.warning(
                                    "[BatchLTXRender] cache copy failed for %s "
                                    "(%s); falling through to fresh render: %s",
                                    line_id, cache_key, _copy_exc,
                                )

                        # Per-chunk shot_seed: stride-shifted so chunks
                        # 1+2 of the same line don't render with identical
                        # seed (would look quasi-identical at the join).
                        shot_seed = (
                            seed
                            + idx * 1009
                            + chunk_idx * 7919
                        ) & 0x7FFFFFFFFFFFFFFF

                        # Empty video latent at the chunk's requested length.
                        empty_latent = _call(
                            "EmptyLTXVLatentVideo",
                            width=LTX_WIDTH, height=LTX_HEIGHT,
                            length=chunk_ltx_length, batch_size=1,
                        )[0]

                        # BUG-LOCAL-095 (2026-05-04 EVENING): swapped from
                        # LTXVAddGuide to LTXVImgToVideoConditionOnly.
                        #
                        # LTXVAddGuide is for KEYFRAME PINNING inside an
                        # existing pipeline -- it attaches the image to
                        # positive/negative cond as a hard anchor at
                        # frame_idx and clamps motion away from that
                        # frame. Even at strength=0.75, frame 0 stays
                        # rigidly locked. This is what produced the
                        # "ltx looks like a still" artefact Jeffrey
                        # reported AFTER BUG-032 removed the end guide
                        # (the start guide was still pinning).
                        #
                        # LTXVImgToVideoConditionOnly is the canonical
                        # I2V INIT node -- it encodes the image into
                        # the FIRST FRAMES of the latent and creates a
                        # noise mask for strength control. The model
                        # sees "start with this image, then evolve
                        # freely" rather than "stay anchored to this
                        # frame". This matches the exact pattern in
                        # comfyui-data-media-machine's DMMBatchVideoGenerator
                        # (nodes/dmm_batch_video.py::_apply_i2v_conditioning,
                        # called via _call("LTXVImgToVideoConditionOnly",
                        # vae=vae, image=image, latent=latent,
                        # strength=strength)) which Jeffrey confirms
                        # produces visibly animated radio still output.
                        #
                        # Returns a single conditioned LATENT; the
                        # cond_pos / cond_neg from LTXVConditioning go
                        # straight to CFGGuider unchanged.
                        latent_chunk = _call(
                            "LTXVImgToVideoConditionOnly",
                            vae=vae,
                            image=ref_image,
                            latent=empty_latent,
                            strength=LTX_I2V_STRENGTH,
                        )[0]

                        # BUG-LOCAL-117c: optional end-frame anchor for
                        # seamless loops. When end_anchor_strength > 0,
                        # use LTXVImgToVideoInplaceKJ (Kijai) for RIGID
                        # latent replacement at frame_idx=-1.
                        #
                        # 2026-05-07 PM: switched FROM LTXVAddGuide TO
                        # LTXVImgToVideoInplaceKJ per research finding.
                        # AddGuide is the SOFT/conditioning variant (looser,
                        # driftier). InplaceKJ does direct latent replace
                        # (rigid). The community FLF workflows use Inplace
                        # for actually-strict endpoints. Even at strength
                        # 0.7, AddGuide didn't visibly close the loop on
                        # OTR's radio scenes.
                        #
                        # InplaceKJ returns ONLY a modified latent (no cond
                        # changes), so cur_cond_pos/cur_cond_neg stay tied
                        # to the original LTXVConditioning output.
                        cur_cond_pos = cond_pos
                        cur_cond_neg = cond_neg
                        cur_latent = latent_chunk
                        if end_anchor_strength > 0.0:
                            try:
                                _aug = _call(
                                    "LTXVImgToVideoInplaceKJ",
                                    vae=vae,
                                    latent=latent_chunk,
                                    num_images={
                                        "image_1": ref_image,
                                        "index_1": -1,
                                        "strength_1": end_anchor_strength,
                                    },
                                )
                                cur_latent = _aug[0]
                            except Exception as _aug_exc:
                                log.warning(
                                    "[BatchLTXRender] LTXVImgToVideoInplaceKJ "
                                    "(end-anchor) failed for %s chunk %d: %s "
                                    "-- proceeding without end anchor for "
                                    "this chunk",
                                    line_id, chunk_idx + 1, _aug_exc,
                                )

                        # BUG-LOCAL-117: engine-specific sample + decode chain.
                        # v0_9 path = legacy euler + CFGGuider + VAEDecodeTiled.
                        # v2_3 path = RES4LYF ClownSampler_Beta + MultimodalGuider
                        # + GuiderParameters + LTXVTiledVAEDecode. Round-robin
                        # 2026-05-06 (Gemini + NVIDIA both) flagged
                        # MultimodalGuider as STRUCTURALLY required for LTX 2.3
                        # DiT -- substituting CFGGuider would crash with tensor
                        # shape mismatch.
                        noise = _call("RandomNoise", noise_seed=shot_seed)[0]
                        if engine == "v2_3":
                            # BUG-LOCAL-117a HOTFIX 2026-05-07 MORNING:
                            # First production run hit
                            #   ValueError: not enough values to unpack
                            #   (expected 2, got 1)
                            # in multimodal_guider.unpack_latents:94. Cause:
                            # MultimodalGuider is designed for AV-joint
                            # pipelines that pack video+audio latents via
                            # LTXVConcatAVLatent BEFORE sampling, then
                            # LTXVSeparateAVLatent AFTER. The packed shape
                            # is what unpack_latents() expects.
                            #
                            # Per Jeffrey 2026-05-07 ~9:15 AM: keep
                            # MultimodalGuider's STG quality, feed it an
                            # AV-packed latent with a tiny zero-filled audio
                            # stub. Memory cost: empty audio latent tensor
                            # (~few KB) plus the audio_vae model patch
                            # cached at execute() top.
                            #
                            # Round-robin diagnosis (which path is right):
                            # - ChatGPT: "mirror stock workflow as closely as
                            #   possible" -- this branch implements that.
                            # - Gemini: said CFGGuider would crash with DiT
                            #   tensor shape mismatch -- WRONG diagnosis but
                            #   correct to discourage simplification; the
                            #   actual requirement is AV-packed latents, not
                            #   DiT conditioning shape.
                            audio_latent = _call(
                                "LTXVEmptyLatentAudio",
                                frames_number=int(chunk_ltx_length),
                                frame_rate=int(LTX_FPS),
                                batch_size=1,
                                audio_vae=ltx_audio_vae,
                            )[0]
                            av_latent = _call(
                                "LTXVConcatAVLatent",
                                video_latent=cur_latent,
                                audio_latent=audio_latent,
                            )[0]
                            video_params = _call(
                                "GuiderParameters",
                                modality="VIDEO",
                                cfg=LTX_V2_3_VIDEO_CFG,
                                stg=LTX_V2_3_VIDEO_STG,
                                perturb_attn=LTX_V2_3_VIDEO_PERTURB_ATTN,
                                rescale=LTX_V2_3_VIDEO_RESCALE,
                                modality_scale=LTX_V2_3_VIDEO_MODALITY_SCALE,
                                skip_step=LTX_V2_3_VIDEO_SKIP_STEP,
                                cross_attn=LTX_V2_3_VIDEO_CROSS_ATTN,
                            )[0]
                            guider = _call(
                                "MultimodalGuider",
                                model=model,
                                positive=cur_cond_pos,
                                negative=cur_cond_neg,
                                parameters=video_params,
                                skip_blocks=LTX_V2_3_SKIP_BLOCKS,
                            )[0]
                            v23_sampler = _call(
                                "ClownSampler_Beta",
                                eta=LTX_V2_3_CLOWN_ETA,
                                sampler_name=LTX_V2_3_CLOWN_SAMPLER_NAME,
                                seed=int(shot_seed) & 0x7FFFFFFFFFFFFFFF,
                                bongmath=LTX_V2_3_CLOWN_BONGMATH,
                            )[0]
                            # Sample with AV-packed latent so MultimodalGuider's
                            # unpack_latents() splits cleanly into (vx, ax).
                            samples_out, _denoised = _call(
                                "SamplerCustomAdvanced",
                                noise=noise, guider=guider,
                                sampler=v23_sampler, sigmas=sigmas,
                                latent_image=av_latent,
                            )
                            # Peel the packed sample apart and discard the
                            # audio side -- OTR is video-only, master mix
                            # comes from the audio path C7-byte-identical
                            # via VideoComposite at final mux time.
                            video_samples, _audio_samples_discard = _call(
                                "LTXVSeparateAVLatent",
                                av_latent=samples_out,
                            )
                            # BUG-LOCAL-117a HOTFIX 2026-05-07 LATE MORNING:
                            # LTXVTiledVAEDecode has TOTALLY different API
                            # than stock VAEDecodeTiled (which is what the
                            # v0_9 path uses). Stock 2.3 workflow line 1542
                            # widgets show:
                            #   latents (input, NOT 'samples')
                            #   horizontal_tiles=2, vertical_tiles=2
                            #   overlap=6
                            #   last_frame_fix=False
                            #   working_device="auto", working_dtype="auto"
                            # This kept failing with TypeError 'unexpected
                            # keyword argument samples' across all 5 clips
                            # AFTER they completed sampling cleanly.
                            frames = _call(
                                "LTXVTiledVAEDecode",
                                vae=vae,
                                latents=video_samples,
                                horizontal_tiles=2,
                                vertical_tiles=2,
                                overlap=6,
                                last_frame_fix=False,
                                working_device="auto",
                                working_dtype="auto",
                            )[0]
                        else:
                            # v0.9 legacy path -- proven on LTX 2B v0.9 since
                            # the original BatchLTXRender ship. Kept verbatim
                            # for emergency rollback (set OTR_LTX_ENGINE=v0_9).
                            guider = _call(
                                "CFGGuider",
                                model=model,
                                positive=cur_cond_pos,
                                negative=cur_cond_neg,
                                cfg=LTX_CFG,
                            )[0]
                            samples_out, _denoised = _call(
                                "SamplerCustomAdvanced",
                                noise=noise, guider=guider,
                                sampler=sampler_obj, sigmas=sigmas,
                                latent_image=cur_latent,
                            )
                            frames = _call(
                                "VAEDecodeTiled",
                                samples=samples_out, vae=vae,
                                tile_size=LTX_TILE_SIZE,
                                overlap=LTX_TILE_OVERLAP,
                                temporal_size=LTX_TEMPORAL_SIZE,
                                temporal_overlap=LTX_TEMPORAL_OVERLAP,
                            )[0]

                        # chunk_dest already resolved above (used by both
                        # cache-hit and live-render paths -- see BUG-LOCAL-117b
                        # cache key block ~80 lines up).
                        _save_video_mp4(
                            images=frames,
                            out_path=chunk_dest,
                            fps=LTX_FPS,
                            ffmpeg=ffmpeg,
                        )

                        # BUG-LOCAL-117d: in-place ffmpeg boomerang. The
                        # chunk was rendered at HALF duration (see the
                        # chunk_render_dur_s branch above) so the
                        # post-process doubles it back to chunk_dur_s with
                        # frame 0 == final frame == radio_bookend. Done
                        # BEFORE register-cache so the cache stores the
                        # full-duration mp4 (any later cache hit copies
                        # the already-boomeranged file straight to disk
                        # without re-running ffmpeg).
                        if loop_via_reverse:
                            try:
                                _make_boomerang_via_ffmpeg(
                                    chunk_dest, ffmpeg=ffmpeg,
                                )
                                log.info(
                                    "[BatchLTXRender] BUG-LOCAL-117d "
                                    "boomerang OK: %s (target dur=%.2fs, "
                                    "rendered half=%.2fs, "
                                    "ffmpeg-doubled to full)",
                                    chunk_dest.name,
                                    chunk_dur_s, chunk_render_dur_s,
                                )
                            except Exception as _bm_exc:
                                log.warning(
                                    "[BatchLTXRender] BUG-LOCAL-117d "
                                    "boomerang FAILED for %s: %s -- "
                                    "leaving half-duration clip on disk "
                                    "(VideoComposite will still mux it; "
                                    "audio will play over a shorter visual)",
                                    chunk_dest.name, _bm_exc,
                                )

                        chunk_mp4_paths.append(chunk_dest)

                        # BUG-LOCAL-117b register the freshly rendered mp4
                        # as the canonical for this cache key. Subsequent
                        # chunks/lines with identical (role, length, sampler,
                        # boomerang) will copy from this path instead of
                        # re-rendering.
                        render_cache[cache_key] = chunk_dest
                        log.info(
                            "[BatchLTXRender] cache STORE %s as canonical "
                            "for key=%s",
                            chunk_dest.name, cache_key,
                        )

                        # BUG-LOCAL-117 per-chunk GC. Round-robin verdict
                        # 2026-05-06 (Gemini): at 14.5 GB peak on a 16 GB
                        # card, Python's lazy GC will OOM on chunk 3+ if
                        # we let prior latent + decoded image refs linger.
                        # Cheap insurance.
                        try:
                            del frames
                            del samples_out
                            del _denoised
                            del latent_chunk
                            del empty_latent
                            del noise
                            del guider
                            if engine == "v2_3":
                                # AV-joint chain extras to release. Each in
                                # its own try because locals() mutation
                                # doesn't actually delete CPython locals.
                                try:
                                    del v23_sampler
                                except (NameError, UnboundLocalError):
                                    pass
                                try:
                                    del video_params
                                except (NameError, UnboundLocalError):
                                    pass
                                try:
                                    del audio_latent
                                except (NameError, UnboundLocalError):
                                    pass
                                try:
                                    del av_latent
                                except (NameError, UnboundLocalError):
                                    pass
                                try:
                                    del video_samples
                                except (NameError, UnboundLocalError):
                                    pass
                                try:
                                    del _audio_samples_discard
                                except (NameError, UnboundLocalError):
                                    pass
                        except (NameError, UnboundLocalError):
                            pass
                        gc.collect()
                        try:
                            import torch as _t  # type: ignore
                            if _t.cuda.is_available():
                                _t.cuda.empty_cache()
                        except Exception:  # noqa: BLE001
                            pass

                    # BUG-LOCAL-091: concat per-chunk part files into the
                    # canonical per-line mp4 if multi-chunk. Same codec /
                    # sample rate / fps across all chunks (same
                    # _save_video_mp4 invocation), so concat-demuxer with
                    # -c copy is safe.
                    out_path = clips_dir / f"{line_id}.mp4"
                    if n_chunks > 1:
                        _concat_clips_via_ffmpeg(
                            chunk_mp4_paths, out_path, ffmpeg=ffmpeg,
                        )
                        for _cp in chunk_mp4_paths:
                            try:
                                _cp.unlink()
                            except OSError:
                                pass
                    # Use the per-line ltx_length for log/report -- it's
                    # the per-chunk frame count which the user expects to
                    # see for sizing context.
                    ltx_length = entry["ltx_length"]

                    shot_ms = int((time.time() - shot_t0) * 1000)
                    if n_chunks > 1:
                        log.info(
                            "[BatchLTXRender] %s done: role=%s "
                            "dur_s=%.3f -> %s (%d ms, %d chunks, "
                            "BUG-LOCAL-091 chunked + concat)",
                            line_id, entry["speaker_role"],
                            entry["dur_s"], out_path.name, shot_ms, n_chunks,
                        )
                        report_lines.append(
                            f"  {line_id} ({entry['speaker_role']}, "
                            f"{ltx_length}f x {n_chunks} chunks, "
                            f"{entry['dur_s']:.2f}s): {shot_ms} ms"
                        )
                    else:
                        log.info(
                            "[BatchLTXRender] %s done: role=%s length=%d "
                            "dur_s=%.3f -> %s (%d ms)",
                            line_id, entry["speaker_role"], ltx_length,
                            entry["dur_s"], out_path.name, shot_ms,
                        )
                        report_lines.append(
                            f"  {line_id} ({entry['speaker_role']}, "
                            f"{ltx_length}f, {entry['dur_s']:.2f}s): "
                            f"{shot_ms} ms"
                        )

                    # BUG-LOCAL-084 fix: stamp start_s + REAL on-disk dur_s
                    # so downstream consumers (audit, debug, future composite
                    # paths) have ground truth instead of the audio-target
                    # placeholder that BUG-LOCAL-033 surfaced.
                    _start_s_truth = None
                    _dur_s_truth = float(entry["dur_s"])  # fallback to audio target
                    try:
                        # Pull start_s from the matching ledger.lines[] entry
                        # by line_id. Lines are stamped by SceneSequencer with
                        # the canonical audio timeline start_s.
                        _ll = ledger.get("lines") or []
                        for _line in _ll:
                            if str(_line.get("line_id") or "") == line_id:
                                _ss = _line.get("start_s")
                                if isinstance(_ss, (int, float)):
                                    _start_s_truth = float(_ss)
                                break
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        # ffprobe the actual rendered file for real dur_s
                        # (BUG-LOCAL-033: never trust ledger.clips[].dur_s
                        # for LTX -- it's the AUDIO TARGET, not video real)
                        from . import _otr_probe as _PROBE  # type: ignore
                        _real = _PROBE.probe_duration_s(out_path)
                        if _real and _real > 0.0:
                            _dur_s_truth = float(_real)
                    except Exception:  # noqa: BLE001
                        pass
                    # BUG-LOCAL-117a Phase 1: stamp engine + sampler so the
                    # ledger lets afternoon-Jeffrey match a clip to the exact
                    # sampler chain that rendered it. Crucial for the A/B/C
                    # bake-off across multiple env-var settings.
                    if engine == "v2_3":
                        _eng_label = "v2_3_clownsampler_res_2s"
                    else:
                        _eng_label = f"v0_9_{v0_9_sampler_resolved}"
                    rendered_clips.append({
                        "line_id": line_id,
                        "speaker_role": entry["speaker_role"],
                        "mp4_path": str(out_path),
                        "ltx_length": ltx_length,
                        "start_s": _start_s_truth,
                        "dur_s": _dur_s_truth,
                        "audio_target_dur_s": float(entry["dur_s"]),
                        "ltx_render_ms": shot_ms,
                        "ref_png_name": radio_png.name,
                        "ref_source": "ltx-radio-bookend",
                        "source_kind": "ltx",
                        "ltx_engine": engine,
                        "ltx_engine_label": _eng_label,
                        "ltx_sampler_name": (
                            "exponential/res_2s"
                            if engine == "v2_3"
                            else v0_9_sampler_resolved
                        ),
                        # BUG-LOCAL-091 traceability: how many chunks
                        # were rendered + concatenated for this line.
                        # 1 = single pass (pre-091 layout); >1 = chunked
                        # because audio dur exceeded clip_length.
                        "n_chunks": int(n_chunks),
                        # BUG-LOCAL-117d traceability: was this clip's
                        # frames produced via boomerang post-process?
                        # Tracking lets the audit + future debug
                        # tooling tell at a glance whether seamless
                        # loops were active.
                        "ltx_loop_via_reverse": bool(loop_via_reverse),
                    })

                except Exception as exc:
                    log.exception(
                        "[BatchLTXRender] %s failed: %s", line_id, exc
                    )
                    report_lines.append(
                        f"  {line_id}: FAILED ({exc})"
                    )

        # ----------------------------------------------------------------
        # 5. Strict teardown so HuMo can load cleanly in the next phase
        #    (per reference_chained_backend_teardown.md)
        # ----------------------------------------------------------------
        try:
            import comfy.model_management as mm  # type: ignore
            mm.unload_all_models()
            log.info("[BatchLTXRender] teardown: unload_all_models")
        except Exception as exc:
            log.warning("[BatchLTXRender] teardown unload failed: %s", exc)
        gc.collect()
        try:
            import torch as _t  # type: ignore
            if _t.cuda.is_available():
                _t.cuda.empty_cache()
                _t.cuda.synchronize()
            log.info("[BatchLTXRender] teardown: cuda empty_cache + sync")
        except Exception:  # noqa: BLE001
            pass

        # ----------------------------------------------------------------
        # 6. Stamp ledger.clips[] (additive -- HuMo will append its
        #    clips later in the next phase under the same clips_dir)
        # ----------------------------------------------------------------
        if ledger_path is not None and rendered_clips:
            try:
                from . import _otr_ledger as _OTRL  # type: ignore
                existing = ledger.get("clips") or []
                ledger["clips"] = list(existing) + rendered_clips
                _OTRL.save_ledger_safe(ledger_path, ledger)
                log.info(
                    "[BatchLTXRender] ledger updated: %d clip records -> %s",
                    len(rendered_clips), ledger_path.name,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[BatchLTXRender] ledger write-back failed: %s", exc
                )

        total_ms = int((time.time() - t_start) * 1000)
        report_lines.append(
            f"BatchLTXRender: {len(rendered_clips)}/{len(plan)} clips in "
            f"{total_ms} ms (avg "
            f"{total_ms // max(len(rendered_clips), 1)} ms/clip)"
        )
        log.info(
            "[BatchLTXRender] complete: %d/%d clips in %d ms",
            len(rendered_clips), len(plan), total_ms,
        )

        return (str(clips_dir), len(rendered_clips), "\n".join(report_lines))

    # ---------------------------------------------------------------------
    # Ledger loading helpers (mirror batch_humo_render.py shape)
    # ---------------------------------------------------------------------

    @staticmethod
    def _load_ledger(arg: str) -> tuple[dict | None, Path | None]:
        """Accept either:
          - inline JSON string (starts with '{' or '[') -- returns (dict, None)
          - path to *_ledger.json -- returns (dict, Path)
          - path to *.mp4 (audio episode); ledger inferred via
            suffix swap (.mp4 -> _ledger.json), since that's the
            convention OTR_SignalLostVideo / EpisodeAssembler write.
            Lets us wire BatchLTXRender's ledger_json input directly
            from SignalLostVideo.video_path -- no separate ledger
            output node required.
          - empty -> auto-pick newest non-pending in the canonical
            audio dirs (BUG-LOCAL-076 fallback chain).

        Returns (ledger_dict_or_None, ledger_path_or_None). Mirrors the
        contract of BatchHumoRender._load_ledger_with_path so the two
        sister nodes resolve the same ledger from the same SignalLostVideo
        STRING output.

        Resolution policy (post-2026-05-02 round-robin hardening on
        BUG-LOCAL-011):
          1. Use ``_otr_ledger.load_ledger_safe(path)`` for path loads
             so the read goes through the canonical loader (consistent
             ``[OTR_Ledger]`` log prefix; future-proof against any
             hardening added there).
          2. Tier 3 fuzzy directory scan from BatchHumoRender is
             intentionally NOT ported. Round-robin (gpt-5.5 + gemini-3.1
             + nemotron-49b) converged on rejecting it for LTX:
             non-deterministic, could plausibly bind to a wrong neighbour
             ledger, and silent fallback would burn ~1 hour rendering
             against bad metadata.
          3. If a Tier 1 or Tier 2 candidate file EXISTS but fails to
             parse / read, raise loud so the run halts immediately
             rather than fall through. Windows file-locking
             ``PermissionError`` falling silently to Tier 2/3 was the
             specific concern Gemini flagged; we honour it by failing
             fast.
        """
        import json as _json
        try:
            from . import _otr_ledger as _OTRL  # type: ignore
        except Exception:  # noqa: BLE001
            _OTRL = None  # type: ignore

        def _read(p: Path) -> dict:
            """Load a JSON file via _OTRL when available; raise on failure.

            ``_OTRL.load_ledger_safe`` returns None on any error; we
            convert that to a RuntimeError so an existing-but-unreadable
            ledger does NOT silently fall through to a fuzzy / wrong
            candidate. Direct ``json.load`` is the fallback when
            ``_otr_ledger`` is unimportable (test contexts).
            """
            if _OTRL is not None:
                led = _OTRL.load_ledger_safe(p)
                if led is None:
                    raise RuntimeError(
                        f"BatchLTXRender: _OTRL.load_ledger_safe returned "
                        f"None for {p} (file exists; check WARNING log "
                        f"line above for the underlying parse / OS error)"
                    )
                return led
            with open(p, "r", encoding="utf-8") as f:
                return _json.load(f)

        s = (arg or "").strip()

        # Layer 0: empty input -> auto-pick newest non-pending ledger.
        if not s:
            # S28 cleanbreak: dropped otr_legacy_audio_dir() from the
            # search list. Only the live audio dir is contract.
            audio_dirs = [otr_audio_dir()]
            cands = []
            for d in audio_dirs:
                if d.exists():
                    cands.extend(
                        p for p in d.glob("*_ledger.json")
                        if not p.name.startswith("pending_")
                    )
            if not cands:
                log.warning(
                    "[BatchLTXRender] ledger_json empty and auto-pick "
                    "found no ledger in audio dirs"
                )
                return None, None
            p = max(cands, key=lambda x: x.stat().st_mtime)
            return _read(p), p

        # Layer 1: inline JSON object.
        if s.startswith("{") or s.startswith("["):
            try:
                return _json.loads(s), None
            except _json.JSONDecodeError as exc:
                log.warning(
                    "[BatchLTXRender] inline JSON parse failed: %s", exc,
                )
                return None, None

        # Layer 2: filesystem path -- could be .mp4 or _ledger.json.
        try:
            p = Path(s)
        except Exception:  # noqa: BLE001
            return None, None

        # Layer 2a: .mp4 path -> swap suffix to _ledger.json
        # (SignalLostVideo / EpisodeAssembler convention). Two-tier
        # resolution per round-robin verdict:
        #   Tier 1 -- exact stem match.
        #   Tier 2 -- collapsed-underscore variant (BUG-LOCAL-118 carry-over).
        # If either candidate file exists but fails to load, _read
        # raises -- intentional fail-loud, no fuzzy fallback.
        if p.suffix.lower() == ".mp4":
            audio_dir = p.parent
            stem = p.stem

            ledger_p = audio_dir / f"{stem}_ledger.json"
            if ledger_p.exists():
                return _read(ledger_p), ledger_p

            collapsed = stem
            while "__" in collapsed:
                collapsed = collapsed.replace("__", "_")
            if collapsed != stem:
                cand = audio_dir / f"{collapsed}_ledger.json"
                if cand.exists():
                    log.warning(
                        "[BatchLTXRender] BUG-LOCAL-118 underscore-mismatch "
                        "fallback: .mp4 stem %r had double underscores; "
                        "loaded matching ledger %r instead.",
                        stem, cand.name,
                    )
                    return _read(cand), cand

            log.warning(
                "[BatchLTXRender] derived ledger from .mp4 not found: "
                "%s (tried Tier 1 exact + Tier 2 collapsed-underscore; "
                "Tier 3 fuzzy scan removed by 2026-05-02 round-robin)",
                ledger_p,
            )
            return None, None

        # Layer 2b: plain _ledger.json path.
        if not p.is_file():
            return None, None
        return _read(p), p


__all__ = ["BatchLTXRender"]
