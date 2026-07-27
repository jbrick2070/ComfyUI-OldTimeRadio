"""LTX-Video text->video motion adapter (A-S5 / CW-6) -- in-process, default-OFF.

LTX-Video is a fast text-driven video generator (it can also act as a base-clip
PROVIDER for a downstream consumer). Unlike a Path-B cu128 sidecar, LTX
runs IN-PROCESS in the main ComfyUI cu130 venv: ``render_clip`` drives the
installed LTX ComfyUI wrapper node classes directly. It is registered DEFAULT-OFF
/ dark (empty ``default_roles`` + gated behind ``OTR_ENABLE_LTX_VIDEO``) so it
shows in the static per-role dropdown (V-6) but is never a default and fails
CLOSED until the operator enables it AND the wrapper + checkpoints are installed
and verified on the GPU box (the CW-6 smoke).

BUG-070 gate: int8-PV SageAttention process-aborts LTX with no traceback, so
``assert_usable`` asserts SageAttention is NOT patched/resident BEFORE the first
forward (the S5 exit gate). The heavy LTX import + sampling is the GPU-smoke
slice; import-time here is cold-import clean (V-12) -- only stdlib + the dep-free
shared helpers + the registry. UTF-8, no BOM, ASCII-only source.

Config (env): ``OTR_ENABLE_LTX_VIDEO`` opt-OUT flag (DEFAULT ON -- the saved
workflow routes the radio open through ltx_video; set =0 to disable). The five
GGUF-recipe weights each resolve via ComfyUI folder_paths (extra_model_paths.yaml
-> C:\\ComfyUI-Models on this box), env-overridable: ``OTR_LTX_VIDEO_UNET`` (GGUF
unet in models/unet), ``OTR_LTX_VIDEO_TEXT_ENCODER`` (Gemma-3 in text_encoders),
``OTR_LTX_VIDEO_PROJECTION_CKPT`` (the LTX ckpt the encoder loader reads, in
checkpoints), ``OTR_LTX_VIDEO_VAE`` (video VAE in vae), ``OTR_LTX_DISTILLED_LORA``
(distilled LoRA in loras).

Recipe = the frozen mini ``workflows/ltx_bookend_mini_repro_gguf_mit.json`` (the
VERIFIED-WORKING path, reproduced EXACTLY): the 22B Q3_K_M GGUF unet + the
distilled LoRA @0.70 + the Gemma-3 text encoder + the distilled sampler chain
(KSamplerSelect ``euler_cfg_pp`` + 8-step LTX_DISTILLED_SIGMAS + RandomNoise +
CFGGuider cfg=1.0 + SamplerCustomAdvanced) + VAEDecodeTiled 512/64/4096/8 + i2v
strength 0.75 at 832x480. ``OTR_LTX_SAMPLER`` default ``distilled`` (these values
HARDCODED to the mini -- the #1 invariant; env knobs ignored in distilled mode);
``ksampler`` keeps the 30-step manual A/B path (env knobs apply there only).
``OTR_ENABLE_LTX_I2V`` defaults ON: LTX shots condition on their FULL-FRAME scene
still (never a portrait), ``OTR_LTX_I2V_STRENGTH`` (ksampler mode only; distilled
hardcodes 0.75).

S5 (2026-07-02, operator ratified): ``OTR_LTX_VIDEO_RECIPE``
(auto|hq_two_stage|single_pass, default auto = unet-family detection). On the
DEV-family unet the SILENT two-stage HQ recipe (the ia2v_canonical transplant
minus the whole audio lane) is the new default; ``single_pass`` is the frozen
GGUF-mini behavior, byte-identical. TWO LTX rows total in the dropdowns
(ltx_video + ltx_audio_in); NO ltx_lowvram (the 22B unet floor makes it
illusory -- 8GB boxes ride profiles). Measured silent-vs-audio-in VRAM/time
A/B on the first live clip is the S5 exit gate (GPU, operator-visible).
"""
from __future__ import annotations

import logging
import os

from . import motion_common as _MC
from .._otr_shared.still_plan_helpers import StillPlanRow
from .frame_contract import CONTINUITY_STRICT_FIRST_FRAME, FrameContract
from .registry import EngineUnusable, EngineUsabilityReason, register

_LOG = logging.getLogger("OTR.video.eng_ltx_video")

_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
_COMFY_ROOT = os.path.dirname(os.path.dirname(_REPO_ROOT))


def _resolve(folder, name):
    """Resolve a model filename to a full path via ComfyUI folder_paths (honors
    extra_model_paths.yaml -> C:\\ComfyUI-Models on this box), with a best-effort
    join fallback for the headless / CPU existence check (no folder_paths
    registered). Mirrors eng_ltx_av._resolve (the proven GGUF lane pattern)."""
    if not name:
        return ""
    try:
        import folder_paths  # type: ignore
        p = folder_paths.get_full_path(folder, name)
        if p:
            return p
    except Exception:  # noqa: BLE001 - headless/CPU has no folder_paths.get_full_path
        pass
    here = os.path.abspath(__file__)
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(here))))), "models", folder, name)


# Weight sanity floors (GiB) -- catch a truncated / wrong download, NOT exact
# byte checks. Grounded vs the on-disk GGUF set: Q3_K_M unet ~10.0 GiB (the
# 2026-06-16 default; Q4_K_S ~12.2); Gemma-3 fp4 ~8.8 GiB; video VAE ~1.35 GiB; distilled LoRA ~7.1 GiB;
# projection ckpt ~43 GiB. Each floor sits well below the real size.
# (eng_ltx_av.py:64-68 supplies the first three constants verbatim.)
_GiB = 1024 ** 3
_FLOOR_UNET = 8 * _GiB
_FLOOR_ENCODER = 6 * _GiB
_FLOOR_VIDEO_VAE = 1 * _GiB
_FLOOR_LORA = 5 * _GiB
_FLOOR_PROJECTION_CKPT = 30 * _GiB

# A-ship in-process forward. The shared mechanics (node resolution, the declarative
# graph executor, the silent bt709 encode, the VRAM guard) are proven in
# wrapper_bridge; the GRAPH below is the ASSUMED native LTXV text->video topology
# and is VERIFY-ON-GPU: the box runs the LTX 2.3 stack (gemma encoder + distilled
# LoRA + audio nodes), so the operator confirms / replaces the node candidates +
# widgets against the installed wrapper INPUT_TYPES (and the exact temporal length
# rule) before enabling. Filenames are env-overridable. (The legacy gate-bound
# deferred-loader shells were retired in the Chunk E cleanbreak -- V-5: loading
# is adapter-internal, lazy, inside execute.)
_LTX_MIN_FRAMES = 9
# Frame-ask CAP (look-QA round 5, D1): LTX coherence collapses far past its
# proven envelope (the 2026-06-10 mud open was a 238f ask -> 233f render for
# the 9.5s synthetic music gap). Asks above the cap render the cap; the
# composite's existing hold-last-frame (tpad clone) fills the rest of the beat
# window. DEFAULT 169 (8*21+1): live-DECODE-PROVEN on this box's LTX wrapper
# (Symphony b001, 2026-06-10). 121 is NOT usable as the default here -- the
# installed wrapper's VAEDecode raises a tensor-shape mismatch (256 vs 128,
# dim 1) on the smaller latent (caught live, ticking_lab run; all three LTX
# beats fell LOUD to the still floor). Env-overridable, clamped LOUD.
_LTX_MAX_FRAMES_DEFAULT = 169


def _env_int(name, default, floor):
    """``int(os.environ[name])`` with a LOUD warning + ``default`` on a missing/
    invalid value and a LOUD clamp up to ``floor``. Pure given the env."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError):
        _LOG.warning("[eng_ltx_video] invalid %s=%r -- using default %d",
                     name, raw, default)
        return int(default)
    if val < floor:
        _LOG.warning("[eng_ltx_video] %s=%d below floor %d -- clamping",
                     name, val, floor)
        return int(floor)
    return val


#: The DECODE floor (round 5, ticking_lab + r5b live catches): at the
#: 1472x832 landscape canvas the installed wrapper's VAEDecode survives ONLY
#: in its tiled band -- 169f and 233f decode clean, 121f and 137f raise the
#: tensor 256-vs-128 (dim 1) mismatch. Short asks are RAISED to the floor
#: (safe: the composite TRUNCATES long sources to the beat window), long asks
#: are CAPPED. With both defaults at 169, every landscape LTX clip renders
#: the one proven length.
_LTX_DECODE_FLOOR_DEFAULT = 169

#: BOOMERANG (loop_via_reverse, BUG-LOCAL-117d restore 2026-06-15): the proven-safe
#: SOURCE half-length the loop renders at the native 832x480 canvas before the
#: in-tensor mirror doubles it back. GPU-proven this session (97f decodes clean at
#: 832x480 in 12s). This is the LOOP path's OWN floor and deliberately does NOT
#: touch the 169 landscape floor above (which still guards the 1472x832 tiled band).
_LTX_LOOP_MIN_DECODE_FRAMES_DEFAULT = 97


def _ltx_frame_length(target_frame_count, fallback):
    """The FINAL LTX graph length for a shot's frame ask: floor to
    ``_LTX_MIN_FRAMES``, cap at ``OTR_LTX_MAX_FRAMES`` (LOUD -- the composite
    hold-fills the window), RAISE short asks to ``OTR_LTX_MIN_DECODE_FRAMES``
    (LOUD -- the composite truncates; see the decode-band note above), then
    snap to LTX's 8n+1 rule. Pure given the env; CPU-tested (round 5 F1)."""
    length = max(_LTX_MIN_FRAMES, int(target_frame_count or fallback))
    cap = _env_int("OTR_LTX_MAX_FRAMES", _LTX_MAX_FRAMES_DEFAULT,
                   _LTX_MIN_FRAMES)
    if length > cap:
        _LOG.warning("[eng_ltx_video] frame ask %d exceeds cap %d -- capping "
                     "(window fill = composite hold-last-frame)", length, cap)
        length = cap
    floor = _env_int("OTR_LTX_MIN_DECODE_FRAMES", _LTX_DECODE_FLOOR_DEFAULT,
                     _LTX_MIN_FRAMES)
    # An operator cap BELOW the decode floor wins (test rigs / tiny canvases
    # have their own decode behavior) -- the floor never exceeds the cap.
    floor = min(floor, cap)
    if length < floor:
        _LOG.warning("[eng_ltx_video] frame ask %d below the decode floor %d "
                     "-- raising (the wrapper VAEDecode fails outside its "
                     "tiled band at this canvas; the composite truncates to "
                     "the beat window)", length, floor)
        length = floor
    return ((length - 1) // 8) * 8 + 1


def _boomerang_frames(frames):
    """Forward + reverse-minus-turnaround mirror: ``[0,1,2,3] -> [0,1,2,3,2,1,0]``
    (length ``2N-1``; the duplicate LAST/turnaround frame is dropped). The
    in-tensor equivalent of the 6/3 ffmpeg ``[b]reverse,trim=start_frame=1;
    [a][r]concat`` boomerang (BUG-LOCAL-117d). ``frames`` is the uint8
    ``[N,H,W,C]`` array from ``images_to_uint8``; returned unchanged when N<2.
    Pure; CPU-testable (numpy imported lazily so module import stays cold V-12)."""
    if len(frames) < 2:
        return frames
    import numpy as _np
    return _np.concatenate([frames, frames[-2::-1]], axis=0)


def _ltx_loop_source_length(target_frame_count, fallback):
    """Boomerang SOURCE length: render ~half, the mirror (``2*src-1``) doubles it
    back. Rounds the half UP and bumps until ``2*src-1 >= target`` so the composite
    never freeze-fills the loop tail (the 169->161 shortfall the roundtable caught).
    Floors at the PROVEN-safe native-canvas loop min (env
    ``OTR_LTX_LOOP_MIN_DECODE_FRAMES`` default 97 -- NOT the global 169 decode
    floor), snaps to 8n+1, caps at ``OTR_LTX_MAX_FRAMES``. Pure given the env."""
    target = max(_LTX_MIN_FRAMES, int(target_frame_count or fallback))
    cap = _env_int("OTR_LTX_MAX_FRAMES", _LTX_MAX_FRAMES_DEFAULT, _LTX_MIN_FRAMES)
    loop_min = _env_int("OTR_LTX_LOOP_MIN_DECODE_FRAMES",
                        _LTX_LOOP_MIN_DECODE_FRAMES_DEFAULT, _LTX_MIN_FRAMES)
    src_ask = (target + 1) // 2                         # round the half UP
    src = ((max(src_ask, loop_min) - 1) // 8) * 8 + 1   # loop-floor, 8n+1 snap
    if src > cap:
        src = ((cap - 1) // 8) * 8 + 1
    # Guarantee the mirrored clip covers the beat window (no composite freeze).
    while (2 * src - 1) < target and src < cap:
        src += 8
    return src


_LTX_DEFAULT_W = 832
_LTX_DEFAULT_H = 480
_LTX_DEFAULT_NEGATIVE = (
    "low quality, worst quality, blurry, distorted, watermark, text, static")

# --------------------------------------------------------------------------- #
# The distilled GGUF sampler config -- the frozen mini recipe
# (workflows/ltx_bookend_mini_repro_gguf_mit.json, the VERIFIED-WORKING path).
# These values are HARDCODED in distilled mode (_build_graph) -- the #1 splice
# invariant. The 8-step schedule originates from ComfyUI-Goofer, proven on the
# RTX 5080 Blackwell box.
# --------------------------------------------------------------------------- #

#: Distilled sigma schedule from ComfyUI-Goofer, proven on RTX 5080
#: Blackwell. 8 sampling steps, last sigma 0.0 = full denoise.
LTX_DISTILLED_SIGMAS = (
    1.0, 0.99375, 0.9875, 0.98125, 0.975,
    0.909375, 0.725, 0.421875, 0.0,
)

#: CFG for the distilled schedule (legacy LTX_CFG: no classifier-free
#: guidance needed on the distilled path) vs the round-5 KSampler default.
_LTX_DISTILLED_CFG = 1.0
_LTX_KSAMPLER_CFG = 3.0

#: The mini's distilled LoRA (ltxv/ltx2/ltx-2.3-22b-distilled-lora-384-1.1). Its
#: keys target the LTX 2.3 22B DiT and it is ALWAYS wired on the 22B GGUF unet
#: (the GGUF recipe wires it in BOTH sampler modes @0.70).
_LTX_DISTILLED_LORA_DEFAULT = os.path.join(
    "ltxv", "ltx2", "ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
_LTX_DISTILLED_LORA_STRENGTH = 0.7

# --------------------------------------------------------------------------- #
# S5 (2026-07-02, operator ratified): the SILENT two-stage HQ recipe -- the
# ia2v_canonical transplant (eng_ltx_av RECIPE_IA2V, comfy.org "LTX-2.3:
# Image Audio to Video") with the ENTIRE audio lane removed (no audio VAE /
# encode / concat / separate / zero-mask -- clips here are silent by contract,
# V-1 mux-LAST). TWO-STAGE: motion at half canvas (Inplace i2v @0.7 soft
# anchor + euler_ancestral_cfg_pp over the front-loaded 8-step ladder) ->
# latent-upsample x2 -> re-anchor @1.0 -> 3-step euler_cfg_pp refine with
# LTXVCropGuides. Selection: env OTR_LTX_VIDEO_RECIPE
# (auto|hq_two_stage|single_pass), default auto -> derived from the
# OTR_LTX_VIDEO_UNET basename family, mirroring eng_ltx_av._detect_recipe
# (dev -> hq_two_stage; distilled -> single_pass; unknown RAISES). The
# single_pass recipe is the frozen GGUF-mini behavior, byte-identical.
# eng_ltx_av stays a SEPARATE module (V-12) -- constants duplicated on purpose.
# --------------------------------------------------------------------------- #
RECIPE_AUTO = "auto"
RECIPE_HQ_TWO_STAGE = "hq_two_stage"
RECIPE_SINGLE_PASS = "single_pass"
_VALID_RECIPES = (RECIPE_HQ_TWO_STAGE, RECIPE_SINGLE_PASS)
_RECIPE_MENU = "auto|hq_two_stage|single_pass"

#: Refine ladder (verbatim ia2v_canonical): 3 steps from 0.85 -- detail only,
#: never re-decides motion.
LTX_HQ_REFINE_SIGMAS = (0.85, 0.7250, 0.4219, 0.0)
#: The HQ recipe uses the ORIGINAL (non-1.1) distilled-lora-384 at HALF
#: strength on the DEV unet: half-distilled keeps dev's motion quality while
#: buying most of the distilled speed (canonical transplant value).
_LTX_HQ_LORA_DEFAULT = os.path.join(
    "ltxv", "ltx2", "ltx-2.3-22b-distilled-lora-384.safetensors")
_LTX_HQ_LORA_STRENGTH = 0.5
_LTX_HQ_I2V_BASE_STRENGTH = 0.7      # motion pass: the still is a soft anchor
_LTX_HQ_I2V_REFINE_STRENGTH = 1.0    # detail pass: hard re-anchor
_LTX_HQ_BASE_SAMPLER = "euler_ancestral_cfg_pp"  # ancestral keeps motion alive
_LTX_HQ_REFINE_SAMPLER = "euler_cfg_pp"
#: Guide-image conditioning chain (canonical texture prep) -- CANVAS-
#: INDEPENDENT fixed numbers (the lips-dont-talk kibitz catch): scale the
#: guide to 1920x1088 then cap the longer edge at 1536, at ANY render canvas.
_LTX_HQ_GUIDE_SCALE_W = 1920
_LTX_HQ_GUIDE_SCALE_H = 1088
_LTX_HQ_GUIDE_LONGER_EDGE = 1536
_LTX_HQ_GUIDE_COMPRESSION = 18
#: Spatial latent upsampler x2 (folder key latent_upscale_models); required
#: ONLY by hq_two_stage so single_pass stays usable on installs lacking it.
_FLOOR_UPSCALER = int(0.05 * _GiB)


def _ltx_hq_upscale_model_name():
    return os.environ.get("OTR_LTX_VIDEO_UPSCALE_MODEL",
                          "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")


class _SigmasFromValues:
    """In-adapter graph node: a float32 SIGMAS tensor from literal values.

    The stock workflow fed SamplerCustomAdvanced via a ManualSigmas node;
    this avoids guessing that node's widget API by mirroring the legacy
    ``torch.tensor(LTX_DISTILLED_SIGMAS, dtype=torch.float32)`` exactly.
    Injected into the resolved-classes mapping by ``render_clip`` (it is
    not a registered ComfyUI node). Lazy torch import (V-12)."""

    FUNCTION = "get"

    def get(self, values):
        import torch  # lazy -- only inside an actual GPU render
        return (torch.tensor([float(v) for v in values],
                             dtype=torch.float32),)


#: S1 (2026-07-25) per-model still plan for ltx_video (spec section 3,
#: Shape A -- scene spine with an LTX I2V twist). FILE-LOCAL, fully
#: declared. Scene stills carry ``required="when_ltx_i2v_enabled"``
#: because ltx_video renders text->video by default; only when the frozen
#: ``routing_state.enable_ltx_i2v`` flag is on does it consume an init
#: image per beat (S0b introduces the frozen flag; S2 wires this in).
_LTX_VIDEO_STILL_PLAN = (
    StillPlanRow(kind="scene_open", cardinality="per_beat",
                 target_class="scene", aspect="wide",
                 required="when_ltx_i2v_enabled",
                 framing_geometry=(
                     "full-frame macro, centered subject"),
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_beat", cardinality="per_beat",
                 target_class="scene", aspect="wide",
                 required="when_ltx_i2v_enabled",
                 framing_geometry=(
                     ("cinematic three-quarter framing, people shown with "
                      "full heads and clear headroom inside frame, faces "
                      "unobstructed, balanced composition")),
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_character", cardinality="per_beat",
                 target_class="scene", aspect="wide",
                 required="when_ltx_i2v_enabled",
                 framing_geometry=(
                     ("cinematic medium shot, the character framed within a "
                      "wide 16:9 environment, full head and shoulders with "
                      "clear headroom inside frame, face unobstructed, "
                      "balanced landscape composition")),
                 style_tail_policy="full"),
    StillPlanRow(kind="portrait", cardinality="per_subject",
                 target_class="portrait", aspect="inherit_engine",
                 required="never",
                 framing_geometry=("in-character cinematic medium shot, head "
                                   "and shoulders, face clearly visible, "
                                   "subject centred with natural headroom "
                                   "above the head (never crop the top of the "
                                   "head)"),
                 style_tail_policy="full"),
)


@register
class LtxVideoEngine(_MC.MotionEngineBase):
    """The ltx_video text->video adapter (in-process, default-OFF / dark)."""

    name = "ltx_video"
    family = "text_to_video"
    #: Still-aspect identity (2026-06-17): ltx_video renders 16:9 832x480, so the
    #: director must mint a WIDE init still (1472x832), NOT the portrait default --
    #: a portrait still fed to the wide render decapitates heads. A class attribute
    #: only (the sampler/sigma chain this file freezes is untouched).
    render_aspect = "wide"
    #: S1 per-model still plan (see ``_LTX_VIDEO_STILL_PLAN`` above -- scene
    #: stills required ONLY when frozen ``routing_state.enable_ltx_i2v``).
    still_plan = _LTX_VIDEO_STILL_PLAN
    # Generative motion music/announcer visuals -- the roles whose only
    # required input is a text prompt. ``announcer_visual`` GRANTED
    # 2026-06-10 (production restore): the operator's radio OPEN renders the
    # period radio-station b-roll on the announcer beats INSTEAD of a
    # lip-synced face -- the deliberate "like before" look (the role supplies
    # text_prompt; LTX ignores the beat's audio_ref by family). A talking-face
    # announcer remains available by selecting humo on the OTR_VideoDirector
    # announcer slot. (retired_role_a/retired_role_b removed 2026-07-01,
    # rip-sfx-broll.)
    roles = ("music_visual", "announcer_visual")
    # 2026-06-17: announcer + music DEFAULT moved to the audio-reactive LTX
    # audio-in lane (ltx_av_music). ltx_video stays SELECTABLE for these roles
    # (still listed in `roles`) but is no longer the in-stack default for any
    # role -- so default_engine_for_role(announcer/music) resolves to ltx_av_music.
    default_roles = ()
    fallback_engine = None               # NO FALLBACKS (2026-07-02): fail LOUD
    required_inputs = ("text_prompt",)
    #: THE FRAME LADDER (chunk 7a, 2026-07-26). 8n+1: 9 .. 169 (8*20+1 is
    #: 161; 169 = 8*21+1 is the live-DECODE-PROVEN ceiling on this box).
    #: max_frames is the LITERAL 169, deliberately NOT ``OTR_LTX_MAX_FRAMES``
    #: -- today an over-ask is silently capped with a warning and the shortfall
    #: becomes a composite hold-last-frame. That capping path is 7c's to delete;
    #: this declaration is what replaces it.
    #: CONTINUITY: the i2v graph uses LTXVImgToVideoConditionOnly at
    #: strength=1.0, which locks frame 0 to the supplied still (BUG-LOCAL-095).
    frame_contract = FrameContract(
        min_frames=9,
        max_frames=169,
        quantum=8,
        native_fps=25,
        allow_tail_trim=True,
        continuity=CONTINUITY_STRICT_FIRST_FRAME,
    )
    commercial_clean = True             # Apache GGUF + LTX-2 Community model (no AGPL/GPL)
    requires_flag = None  # vestigial (registry IS the menu; no flag gate)
    engine_version = "1"
    declared_isolation = _MC.ISOLATION_IN_PROCESS
    target_fps = 25
    #: BUG-LOCAL-117d boomerang: ltx_video LOOPS by default (forward+reverse,
    #: the 5/09-6/05 "more motion" look).
    _LOOP_VIA_REVERSE_DEFAULT = True

    def _loop_via_reverse(self) -> bool:
        """Whether render_clip renders the boomerang (half-render + forward/reverse
        mirror). ``OTR_LTX_LOOP_VIA_REVERSE``: truthy {on,1,true,yes}, false
        {off,0,false,no}, invalid -> LOUD warn + engine default
        (``_LOOP_VIA_REVERSE_DEFAULT``)."""
        raw = os.environ.get("OTR_LTX_LOOP_VIA_REVERSE")
        if raw is None:
            return bool(self._LOOP_VIA_REVERSE_DEFAULT)
        v = raw.strip().lower()
        if v in ("on", "1", "true", "yes"):
            return bool(self._LOOP_VIA_REVERSE_DEFAULT)
        if v in ("off", "0", "false", "no"):
            return False
        _LOG.warning("[eng_ltx_video] OTR_LTX_LOOP_VIA_REVERSE=%r invalid "
                     "(use on/off) -- using engine default %s", raw,
                     self._LOOP_VIA_REVERSE_DEFAULT)
        return bool(self._LOOP_VIA_REVERSE_DEFAULT)

    # ---- config resolution (env override -> folder_paths -> join) ----
    # The GGUF mini recipe's weight artifacts (defaults = the frozen mini's
    # filenames, workflows/ltx_bookend_mini_repro_gguf_mit.json); each resolves
    # via ComfyUI folder_paths so a box never needs a code edit. The distilled
    # LoRA keeps its own (name, path) tuple helper (_distilled_lora_file, below).
    def _unet_name(self):
        return os.environ.get("OTR_LTX_VIDEO_UNET",
                              "ltx-2.3-22b-dev-Q3_K_M.gguf")

    def _encoder_name(self):
        return os.environ.get("OTR_LTX_VIDEO_TEXT_ENCODER",
                              "gemma_3_12B_it_fp4_mixed.safetensors")

    def _encoder_device(self):
        """LTXAVTextEncoderLoader device for the Gemma-3 text encoder. Default
        'default' (GPU) = the frozen-mini recipe. The 2026-06-16 full-episode
        all-LTX soak MEASURED device='cpu' vs 'default' and found the LTX-phase
        NVML peak IDENTICAL (~15.85 GB both ways): ComfyUI's dynamic memory
        manager already evicts the encoder before the unet + VAE-decode peak, so
        the encoder device does NOT move the peak. The 2026-06-16 battle adopted
        Q3_K_M as the default unet (per-clip peak ~14804 MiB at the 14.5 GB cap,
        2.2x faster than Q4_K_S with no decode offload, objective quality
        comparable -- the operator's call); 'cpu' only makes the text encode run
        slowly ON the CPU for no VRAM gain, so 'default' stays the default; the
        env knob remains for boxes that want the encoder off-GPU. (M0 @512x288x97:
        Q4_K_S 15594 MB, Q3_K_M 13688 MB.)"""
        return os.environ.get("OTR_LTX_VIDEO_ENCODER_DEVICE", "default")

    def _projection_ckpt(self):
        return os.environ.get("OTR_LTX_VIDEO_PROJECTION_CKPT",
                              "ltx-2.3-22b-dev.safetensors")

    def _video_vae_name(self):
        return os.environ.get("OTR_LTX_VIDEO_VAE",
                              "ltx-2.3-22b-dev_video_vae.safetensors")

    def _weight_paths(self):
        """(label, full_path, floor_bytes) for each required weight artifact,
        RECIPE-AWARE (S5): common = GGUF unet, Gemma-3 text encoder, video VAE,
        projection ckpt. single_pass adds the 1.1 distilled LoRA; hq_two_stage
        adds the ORIGINAL 384 LoRA + the spatial latent upsampler (required
        ONLY there so single_pass stays usable on installs lacking it)."""
        paths = [
            ("transformer GGUF", _resolve("unet", self._unet_name()), _FLOOR_UNET),
            ("Gemma-3 text encoder",
             _resolve("text_encoders", self._encoder_name()), _FLOOR_ENCODER),
            ("video VAE", _resolve("vae", self._video_vae_name()), _FLOOR_VIDEO_VAE),
            ("projection ckpt",
             _resolve("checkpoints", self._projection_ckpt()), _FLOOR_PROJECTION_CKPT),
        ]
        if self._recipe() == RECIPE_HQ_TWO_STAGE:
            lora_name, lora_path = self._hq_lora_file()
            paths.append(("HQ distilled LoRA (384)",
                          lora_path or _resolve("loras", lora_name), _FLOOR_LORA))
            paths.append(("latent spatial upscaler",
                          _resolve("latent_upscale_models",
                                   _ltx_hq_upscale_model_name()),
                          _FLOOR_UPSCALER))
        else:
            lora_name, lora_path = self._distilled_lora_file()
            paths.append(("distilled LoRA",
                          lora_path or _resolve("loras", lora_name), _FLOOR_LORA))
        return paths

    # ---- usability (fail-closed BEFORE any forward; no heavy import) ----
    def assert_usable(self, host_caps, profile, request_template=None):
        """Ordered, fail-closed-before-GPU gate (PINNED reasons only); mirrors
        eng_ltx_av.assert_usable (the proven GGUF lane), V-12-safe (no heavy
        import -- runs at lock/validate time on the CPU box):
        1 BUG-070 SageAttention gate (int8-PV aborts LTX silently);
        2 the 5 GGUF weight artifacts present + above their sanity floor;
        3 node gate -- every required ComfyUI class resolves (lazy mapping read).
        All fail-closed NAMED EngineUnusable so the manager's fallback catches it."""
        # BUG-070 SageAttention contamination (int8-PV aborts LTX silently)
        _MC.assert_sage_not_patched(self.name, self.family)
        # 2b -- recipe resolution FIRST (S5, fail-loud BEFORE node/weight
        #       gating): a bad OTR_LTX_VIDEO_RECIPE or an unrecognized unet
        #       family RAISES here, at the gate, never mid-render (NO
        #       FALLBACKS). _node_candidates + _weight_paths consult the same
        #       resolver, so the graph cannot drift from the gate.
        recipe = self._recipe()
        # 3 -- weights present + above the sanity floor (realpath -> broken
        #      symlinks fail)
        for label, path, floor in self._weight_paths():
            real = os.path.realpath(path) if path else ""
            if not real or not os.path.exists(real):
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                    "%s %s not found at %r (set the matching OTR_LTX_VIDEO_* env; "
                    "install the LTX-2.3 GGUF unet + Gemma-3 encoder + LTX video "
                    "VAE + distilled LoRA + projection ckpt)"
                    % (self.name, label, path), kind="video")
            if os.path.getsize(real) < floor:
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                    "%s %s at %r is below the %d-byte floor (truncated/wrong "
                    "file?)" % (self.name, label, real, floor), kind="video")
        # 4 -- node gate: every required ComfyUI class must resolve (lazy
        #      read), on the ACTIVE recipe's candidate set (S5)
        from . import wrapper_bridge as _wb
        missing = []
        mapping = _wb.node_class_mappings()
        active_cands = (self._node_candidates_hq()
                        if recipe == RECIPE_HQ_TWO_STAGE
                        else self._node_candidates())
        for _logical, candidates in active_cands.items():
            try:
                _wb.resolve_node_class(candidates, mapping)
            except Exception:  # noqa: BLE001 - collect every missing class
                missing.append("/".join(candidates))
        if missing:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "%s missing required ComfyUI node class(es): %s (install/update "
                "ComfyUI-GGUF + ComfyUI-LTXVideo)" % (self.name, ", ".join(missing)),
                kind="video")
        return self.name

    #: Terminal node of the graph (its IMAGE output is encoded to the clip).
    _TERMINAL = "vaedecode"

    # ---- LTX-I2V (ticket Part B 2026-06-11; LK-1a: DEFAULT ON) ----
    # With the flag on (DEFAULT since LK-1a -- the 2026-06-11 look
    # restoration; the Part-B probe PASSED at 1472x832x169f) AND a request
    # init image present, render_clip builds the image-conditioned wrapper
    # graph (LTXVImgToVideo) instead of txt2vid. ENGINE_FAMILY stays
    # text_to_video (no registry/family change, no new widgets) -- the
    # branch lives entirely inside the adapter. The decode band is
    # UNTOUCHED: the same _ltx_frame_length floor/cap governs both paths
    # (do NOT touch the 169f floor). Set OTR_ENABLE_LTX_I2V=0 to restore
    # the text-only path (the murk the operator rejected).
    @staticmethod
    def _i2v_enabled() -> bool:
        return os.environ.get("OTR_ENABLE_LTX_I2V", "1") == "1"

    @staticmethod
    def _init_image_path(request) -> str:
        """The request's init image path (asset_refs still/init_image/image),
        '' when none. Same key set the cheap families read."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        assets = get("asset_refs") or {}
        if isinstance(assets, dict):
            for key in ("still", "init_image", "image"):
                v = assets.get(key)
                if isinstance(v, str) and v:
                    return v
                if isinstance(v, dict) and v.get("path"):
                    return v["path"]
        return ""

    def _use_i2v(self, request) -> bool:
        """True iff the env flag is on AND the request carries an init image
        that exists on disk. Flag OFF = the text path, byte-identical. Flag ON
        with a missing or stale init image REFUSES -- it does not degrade.

        A4 (2026-07-27): this used to log and return False, which put two
        contradictory policies over one state. ``render_driver`` already calls
        that state terminal for the ledger path ("NO FALLBACK to text-only
        rendering", render_driver.py:2146-2150), so whichever boundary the
        request reached first decided whether a missing still killed the
        episode or silently changed how it was rendered. The driver's check is
        also weaker than this one: it only asks whether the still INDEX holds a
        non-empty path, while this asks whether the file is on disk -- so a
        STALE path passed the driver and degraded here, and requests built
        through the older ledger-free ``build_request`` never met the driver's
        check at all. One policy now, enforced at the boundary that can see the
        file. The deliberate text-only render keeps its explicit opt-out.

        Mirrors the shipped sibling contract in ``cheap_families``
        (``_require_still``): a still-REQUIRED family refuses its floor rather
        than shipping something that merely looks finished."""
        if not self._i2v_enabled():
            return False
        p = self._init_image_path(request)
        if p and os.path.exists(p):
            return True
        raise RuntimeError(
            "%s has LTX-I2V enabled but the request carries no usable on-disk "
            "init image (asset_refs still/init_image=%r) -- refusing the "
            "text-only path (NO FALLBACKS). The image phase must mint this "
            "beat's still before the video render; set OTR_ENABLE_LTX_I2V=0 "
            "to render text-only deliberately." % (self.name, p))

    def _node_candidates_i2v(self):
        """The image-conditioned graph's node candidates (on the ACTIVE
        sampling set -- LK-1b). resolve_graph_classes fail-LOUD names any
        missing wrapper class (the probe's first gate).

        BUG-LOCAL-095 fix (2026-06-11): LTXVImgToVideoConditionOnly, NOT
        LTXVImgToVideo. The difference is fundamental:
        - LTXVImgToVideo encodes the image INTO the latent (slot 2) and
          conditions all frames. At strength 1.0 that is a freeze-frame;
          at 0.75 the 2B re-noises the still into red mush at 1472x832.
        - LTXVImgToVideoConditionOnly only embeds the image into the FIRST
          FRAME CONDITIONING (pos/neg, slots 0/1). The sampler still receives
          a pure-noise EmptyLTXVLatentVideo latent -- the model generates motion
          freely from frame 1 onward while being anchored to the still's look.
        Therefore: keep "latent" (EmptyLTXVLatentVideo) in the candidate set;
        the sampler's latent_image comes from "latent", not "img2vid"."""
        cands = dict(self._node_candidates_sampling())
        # Keep "latent" (EmptyLTXVLatentVideo) -- the installed ConditionOnly node
        # CONSUMES that pure-noise latent (writes the image into its first frame +
        # a noise_mask) and returns the conditioned latent for the sampler.
        cands["loadimage"] = ("LoadImage",)
        cands["img2vid"] = ("LTXVImgToVideoConditionOnly",)
        return cands

    def _build_graph_i2v(self, plan, length, width, height, image_name):
        """The declarative image-conditioned LTXV graph (BUG-LOCAL-095 motion
        contract; rewired 2026-06-12 for the installed wrapper API):

          EmptyLTXVLatentVideo("latent") + LoadImage -> LTXVImgToVideoConditionOnly
              (vae, image, latent, strength=1.0) -> [sampler].latent_image ->
              VAEDecode; conditioning from CLIPTextEncode -> LTXVConditioning("cond").

        The installed LTXVImgToVideoConditionOnly writes the encoded image into the
        FIRST latent frame(s) and attaches noise_mask = 1.0-strength there (1.0 on
        all later frames). At strength=1.0 the first frame is locked to the still
        and the rest denoise freely -> anchored start + real motion. This keeps the
        BUG-LOCAL-095 contract that defeated the LTXVImgToVideo freeze/red-mush;
        only the node's signature changed (it now takes + returns a LATENT instead
        of emitting conditioning, so positive/negative come from the text encoders).
        VERIFY-ON-GPU: motion is present (NOT a freeze-frame) + decode/crop 1472x832."""
        from . import wrapper_bridge as _wb
        W = _wb.Wire
        graph = self._build_graph(plan, length, width, height)
        # Keep "latent" (EmptyLTXVLatentVideo) -- ConditionOnly does NOT
        # produce a latent; the sampler needs pure-noise init for motion.
        graph["loadimage"] = {"class": "loadimage",
                              "inputs": {"image": image_name}}
        # LTXVImgToVideoConditionOnly (installed wrapper API, confirmed 2026-06-12):
        # inputs (vae, image, latent, strength[, bypass]); returns ONE LATENT.
        # Its generate() writes the encoded image into the FIRST latent frame(s)
        # and attaches noise_mask = 1.0-strength on those frames (1.0 elsewhere).
        # At strength=1.0 the first frame is LOCKED to the still (mask 0.0) while
        # every later frame is freely denoised (mask 1.0) -> anchored start + real
        # motion. This PRESERVES the BUG-LOCAL-095 motion contract (no freeze, no
        # red mush); only the node's signature changed. It no longer emits
        # conditioning -- positive/negative still flow from the text encoders via
        # the base "cond" (LTXVConditioning) node, left untouched here.
        # (Pre-2026-06-12 the node took positive/negative and output conditioning;
        # feeding the old kwargs now raises "unexpected keyword argument 'positive'".)
        # BUG-LOCAL-412 (operator 2026-06-15, 6/5 parity): the 6/5 LTX recipe used
        # I2V strength 0.75 at the 832x480 native canvas -- a SOFT anchor that
        # leaves room for coherent motion. The post-refactor default was bumped to
        # 1.0 to fight the mush at 1472x832, but that hard-locks frame 0 while later
        # frames drift; now that render_driver renders LTX at native 832x480
        # (OTR_LTX_RENDER_CANVAS) the 6/5 0.75 is correct again. Env-overridable.
        # GGUF splice: I2V strength 0.75 at native 832x480 (the mini's value).
        # DISTILLED mode HARDCODES it to the mini (#1 invariant -- env ignored);
        # ksampler/manual keeps the OTR_LTX_I2V_STRENGTH knob.
        # i2v conditioning strength. DEFAULT 0.75 = the frozen mini, so with the
        # env UNSET the distilled path still reproduces the mini EXACTLY (#1
        # invariant intact). 2026-06-16 operator framing fix (roundtable pass03):
        # OTR_LTX_I2V_STRENGTH is now HONORED in distilled too (was ksampler-only)
        # -- raising it (e.g. 0.85) makes the clip hug the well-framed still
        # harder = less subject/head drift, the panel's PRIMARY lever for keeping
        # the subject in frame. Opt-in only; the default is unchanged.
        try:
            cond_strength = float(os.environ.get("OTR_LTX_I2V_STRENGTH", "0.75"))
        except (TypeError, ValueError):
            cond_strength = 0.75
        cond_strength = max(0.0, min(1.0, cond_strength))
        graph["img2vid"] = {
            "class": "img2vid",
            "inputs": {"vae": W("videovae", 0), "image": W("loadimage", 0),
                       "latent": W("latent", 0), "strength": cond_strength}}
        # The image-anchored latent (carrying its first-frame noise_mask) IS the
        # sampler's latent_image now -- NOT the raw pure-noise EmptyLTXVLatentVideo.
        # Conditioning is unchanged: the base "cond" node still reads pos/neg from
        # the text encoders, so motion past frame 0 is driven normally.
        sampler_node = "sampleradv" if "sampleradv" in graph else "ksampler"
        graph[sampler_node]["inputs"]["latent_image"] = W("img2vid", 0)
        return graph

    # ---- in-process graph spec (the frozen GGUF mini recipe;
    # workflows/ltx_bookend_mini_repro_gguf_mit.json -- the VERIFIED-WORKING path)
    # Topology (distilled default):
    #   UnetLoaderGGUF -> LoraLoaderModelOnly(distilled @0.70) -> CFGGuider.model;
    #   LTXAVTextEncoderLoader(Gemma-3 + projection ckpt) -> CLIPTextEncode x2 ->
    #   LTXVConditioning; EmptyLTXVLatentVideo -> SamplerCustomAdvanced
    #   (KSamplerSelect euler_cfg_pp + ManualSigmas 8-step + RandomNoise +
    #   CFGGuider cfg=1.0) -> VAEDecodeTiled(512/64/4096/8, video VAE).
    def _node_candidates(self):
        """Ordered ComfyUI node-class candidates per BASE graph node (the GGUF
        mini recipe). The 5 loaders/decode live here so they propagate to the
        sampling + i2v candidate sets (which build on this) AND to the
        assert_usable node gate. resolve_graph_classes fail-LOUD names any
        missing wrapper class."""
        return {
            "unet": ("UnetLoaderGGUF",),
            "lora": ("LoraLoaderModelOnly",),
            "videovae": ("VAELoader",),
            "te": ("LTXAVTextEncoderLoader",),
            "pos": ("CLIPTextEncode",),
            "neg": ("CLIPTextEncode",),
            "latent": ("EmptyLTXVLatentVideo",),
            "cond": ("LTXVConditioning",),
            "ksampler": ("KSampler",),
            "vaedecode": ("VAEDecodeTiled",),
        }

    # ---- LK-1b: sampler mode + distilled LoRA resolution ----
    @staticmethod
    def _sampler_mode() -> str:
        """``distilled`` (DEFAULT -- the frozen GGUF mini recipe: 8-step
        LTX_DISTILLED_SIGMAS, ``euler_cfg_pp``, CFGGuider cfg=1.0, distilled LoRA
        @0.70 on the 22B GGUF unet -- the VERIFIED-WORKING path) or ``ksampler``
        (30-step euler manual A/B path, kept selectable via OTR_LTX_SAMPLER=
        ksampler). In distilled mode the recipe values are HARDCODED to the mini
        (env knobs ignored; see _build_graph) -- the #1 splice invariant. The env
        knobs (OTR_LTX_CFG / _SAMPLER_NAME / _DISTILLED_LORA_STRENGTH /
        _I2V_STRENGTH) apply to ksampler/manual only. Invalid falls back LOUD to
        distilled."""
        mode = os.environ.get("OTR_LTX_SAMPLER", "distilled").strip().lower()
        if mode not in ("distilled", "ksampler"):
            _LOG.warning("[eng_ltx_video] unknown OTR_LTX_SAMPLER=%r -- "
                         "using distilled default", mode)
            mode = "distilled"
        return mode

    @staticmethod
    def _sampler_name() -> str:
        """The KSampler/KSamplerSelect ``sampler_name``. DEFAULT ``euler_cfg_pp``
        (CFG++) -- the documented dynamic-motion sampler the 5/09+5/28+6/1+6/5
        good LTX bookends recorded (BUG-LOCAL-412 forensic); override via
        OTR_LTX_SAMPLER_NAME. The post-cleanbreak refactor had left it on plain
        ``euler``; restored 2026-06-14 per operator ("make LTX just as it was")."""
        return os.environ.get("OTR_LTX_SAMPLER_NAME", "euler_cfg_pp").strip()

    def _distilled_lora_file(self):
        """``(lora_name, abs_path)`` for the distilled LoRA; ``abs_path`` is
        '' when the file is not on disk. Searched under the ComfyUI-relative
        loras dir + the HF_HOME-derived shared-models loras dir (C:\\ComfyUI-Models
        on this box). Kept as a (name, path) tuple per the splice plan (the new
        4-artifact resolvers handle the rest)."""
        name = os.environ.get("OTR_LTX_DISTILLED_LORA",
                              _LTX_DISTILLED_LORA_DEFAULT)
        if not name:
            return "", ""
        cand_dirs = [os.path.join(_COMFY_ROOT, "models", "loras")]
        hf_home = os.environ.get("HF_HOME", "")
        if hf_home:
            cand_dirs.append(os.path.join(os.path.dirname(hf_home), "loras"))
        for d in cand_dirs:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return name, p
        return name, ""

    def _node_candidates_sampling(self):
        """The ACTIVE sampling candidates: the base set (which ALREADY includes
        the GGUF unet + distilled LoRA + video VAE + Gemma text-encoder loaders),
        with the KSampler node swapped for the distilled chain (KSamplerSelect +
        RandomNoise + CFGGuider + SamplerCustomAdvanced) in distilled mode. The
        SIGMAS source is the in-adapter ``_SigmasFromValues`` (injected after
        resolve -- not a registered node class). The distilled LoRA is ALWAYS
        present (the GGUF recipe wires it in BOTH sampler modes)."""
        cands = dict(self._node_candidates())
        if self._sampler_mode() == "distilled":
            del cands["ksampler"]
            cands["samplersel"] = ("KSamplerSelect",)
            cands["noise"] = ("RandomNoise",)
            cands["guider"] = ("CFGGuider",)
            cands["sampleradv"] = ("SamplerCustomAdvanced",)
        return cands

    # ---- S5 recipe resolution (mirrors eng_ltx_av._recipe/_detect_recipe) ----
    def _recipe(self):
        """Resolve the ACTIVE recipe. Reads env EVERY call (a soak can flip
        recipes per leg by swapping OTR_LTX_VIDEO_UNET / OTR_LTX_VIDEO_RECIPE).
        auto -> unet-family detection; explicit invalid value RAISES (NO
        FALLBACKS); hq_two_stage on a distilled unet RAISES (the HQ ladder
        assumes the DEV unet -- full distillation blunts the refine)."""
        sel = os.environ.get("OTR_LTX_VIDEO_RECIPE", RECIPE_AUTO).strip().lower()
        base = os.path.basename(self._unet_name()).lower()
        if sel != RECIPE_AUTO:
            if sel not in _VALID_RECIPES:
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                    "OTR_LTX_VIDEO_RECIPE=%r invalid -- use %s"
                    % (sel, _RECIPE_MENU), kind="video")
            if sel == RECIPE_HQ_TWO_STAGE and base.startswith(
                    "ltx-2.3-22b-distilled"):
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                    "OTR_LTX_VIDEO_RECIPE=hq_two_stage requires a DEV-family "
                    "unet; %r is distilled -- use single_pass or swap "
                    "OTR_LTX_VIDEO_UNET" % base, kind="video")
            return sel
        return self._detect_recipe(self._unet_name())

    def _detect_recipe(self, unet_name):
        """auto: derive the recipe from the unet basename family. An unknown
        family RAISES (the operator sets OTR_LTX_VIDEO_RECIPE) -- a name that
        matches NO known family must never silently get a mismatched ladder."""
        base = os.path.basename(unet_name).lower()
        if base.startswith("ltx-2.3-22b-distilled"):
            return RECIPE_SINGLE_PASS
        if base.startswith("ltx-2.3-22b-dev"):
            return RECIPE_HQ_TWO_STAGE
        raise EngineUnusable(
            self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
            "unrecognized LTX unet family %r -- set OTR_LTX_VIDEO_RECIPE=%s"
            % (base, _RECIPE_MENU), kind="video")

    def _hq_lora_file(self):
        """(lora_name, abs_path) for the HQ recipe's half-strength LoRA (the
        ORIGINAL non-1.1 384). Same search dirs as _distilled_lora_file."""
        name = os.environ.get("OTR_LTX_HQ_LORA", _LTX_HQ_LORA_DEFAULT)
        if not name:
            return "", ""
        cand_dirs = [os.path.join(_COMFY_ROOT, "models", "loras")]
        hf_home = os.environ.get("HF_HOME", "")
        if hf_home:
            cand_dirs.append(os.path.join(os.path.dirname(hf_home), "loras"))
        for d in cand_dirs:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return name, p
        return name, ""

    def _node_candidates_hq(self):
        """The silent two-stage HQ graph's node candidates: the loaders +
        conditioning from the base set, the guide-image chain, the Inplace
        pair, the latent upsampler, CropGuides, and BOTH sampler chains.
        NO audio classes (the whole point of the silent port)."""
        return {
            "unet": ("UnetLoaderGGUF",),
            "lora": ("LoraLoaderModelOnly",),
            "te": ("LTXAVTextEncoderLoader",),
            "pos": ("CLIPTextEncode",),
            "neg": ("CLIPTextEncode",),
            "cond": ("LTXVConditioning",),
            "videovae_dec": ("VAELoader",),
            "videovae_enc": ("VAELoader",),
            "loadimage": ("LoadImage",),
            "imagescale": ("ImageScale",),
            "resizelonger": ("ResizeImagesByLongerEdge",),
            "preprocess": ("LTXVPreprocess",),
            "emptylatent": ("EmptyLTXVLatentVideo",),
            "inplace_base": ("LTXVImgToVideoInplace",),
            "inplace_refine": ("LTXVImgToVideoInplace",),
            "upscale_loader": ("LatentUpscaleModelLoader",),
            "upscaler": ("LTXVLatentUpsampler",),
            "cropguides": ("LTXVCropGuides",),
            "ksel": ("KSamplerSelect",),
            "noise": ("RandomNoise",),
            "guider": ("CFGGuider",),
            "sampler": ("SamplerCustomAdvanced",),
            "ksel_refine": ("KSamplerSelect",),
            "noise_refine": ("RandomNoise",),
            "guider_refine": ("CFGGuider",),
            "sampler_refine": ("SamplerCustomAdvanced",),
            "decode": ("VAEDecodeTiled",),
        }

    def _build_graph_hq(self, plan, length, width, height, image_name):
        """The SILENT two-stage HQ graph (S5 port of eng_ltx_av
        _build_graph_ia2v, audio lane deleted):

        STAGE A -- MOTION (half canvas): guide still -> ImageScale(1920x1088)
        -> ResizeImagesByLongerEdge(1536) -> LTXVPreprocess(18); planted
        IN-PLACE (strength 0.7, SOFT anchor) into an EmptyLTXVLatentVideo at
        width/2 x height/2; denoised by euler_ancestral_cfg_pp over the
        front-loaded 8-step ladder. The latent is PURE VIDEO -- no AV concat,
        no separate, no zero-masked audio latent.

        STAGE B -- DETAIL (full canvas): LTXVLatentUpsampler x2 -> re-anchor
        the SAME preprocessed still at strength 1.0 -> 3-step euler_cfg_pp
        refine (0.85 -> 0) with conditioning routed through
        LTXVCropGuides(base latent) -> tiled decode. Silent by contract (V-1).

        The init still is REQUIRED (NO FALLBACKS -- render_clip raises before
        this builds when absent; defense at the build seam too). Base noise =
        the beat seed; refine noise = seed+1 (two independent deterministic
        streams, C7-style)."""
        from . import wrapper_bridge as _wb
        W = _wb.Wire
        if not image_name:
            raise _wb.GraphExecutionError(
                "%s (hq_two_stage) requires an init image for EVERY beat -- "
                "no image means no graph, never a silent t2v downgrade"
                % self.name)
        positive = plan.get("text_prompt") or "a cinematic scene"
        negative = plan.get("negative_prompt") or os.environ.get(
            "OTR_LTX_NEGATIVE", _LTX_DEFAULT_NEGATIVE)
        seed = int(plan.get("seed", 0) or 0)
        lora_name, _lora_path = self._hq_lora_file()
        base_w, base_h = int(width) // 2, int(height) // 2  # exact halving
        g = {
            "unet": {"class": "unet",
                     "inputs": {"unet_name": self._unet_name()}},
            "lora": {"class": "lora", "inputs": {
                "model": W("unet", 0), "lora_name": lora_name,
                "strength_model": _LTX_HQ_LORA_STRENGTH}},
            "te": {"class": "te", "inputs": {
                "text_encoder": self._encoder_name(),
                "ckpt_name": self._projection_ckpt(),
                "device": self._encoder_device()}},
            "pos": {"class": "pos",
                    "inputs": {"text": positive, "clip": W("te", 0)}},
            "neg": {"class": "neg",
                    "inputs": {"text": negative, "clip": W("te", 0)}},
            "cond": {"class": "cond", "inputs": {
                "positive": W("pos", 0), "negative": W("neg", 0),
                "frame_rate": float(self.target_fps)}},
            "videovae_dec": {"class": "videovae_dec",
                             "inputs": {"vae_name": self._video_vae_name()}},
            # encode-side VAE SEPARATE (BUG-414 split, ported): its last
            # consumer (inplace_refine) runs BEFORE the refine sampler, so
            # free_after_use reclaims it ahead of both activation peaks.
            "videovae_enc": {"class": "videovae_enc",
                             "inputs": {"vae_name": self._video_vae_name()}},
            # ---- guide-image conditioning chain (canonical texture prep) ----
            "loadimage": {"class": "loadimage", "inputs": {"image": image_name}},
            "imagescale": {"class": "imagescale", "inputs": {
                "image": W("loadimage", 0), "upscale_method": "lanczos",
                "width": _LTX_HQ_GUIDE_SCALE_W,
                "height": _LTX_HQ_GUIDE_SCALE_H,
                "crop": "center"}},
            "resizelonger": {"class": "resizelonger", "inputs": {
                "images": W("imagescale", 0),
                "longer_edge": _LTX_HQ_GUIDE_LONGER_EDGE}},
            "preprocess": {"class": "preprocess", "inputs": {
                "image": W("resizelonger", 0),
                "img_compression": _LTX_HQ_GUIDE_COMPRESSION}},
            # ---- STAGE A: motion at half canvas (pure video latent) ----
            "emptylatent": {"class": "emptylatent", "inputs": {
                "width": base_w, "height": base_h,
                "length": int(length), "batch_size": 1}},
            "inplace_base": {"class": "inplace_base", "inputs": {
                "vae": W("videovae_enc", 0), "image": W("preprocess", 0),
                "latent": W("emptylatent", 0),
                "strength": _LTX_HQ_I2V_BASE_STRENGTH, "bypass": False}},
            "guider": {"class": "guider", "inputs": {
                "model": W("lora", 0), "positive": W("cond", 0),
                "negative": W("cond", 1), "cfg": _LTX_DISTILLED_CFG}},
            "sigmas": {"class": "sigmas",
                       "inputs": {"values": list(LTX_DISTILLED_SIGMAS)}},
            "ksel": {"class": "ksel",
                     "inputs": {"sampler_name": _LTX_HQ_BASE_SAMPLER}},
            "noise": {"class": "noise", "inputs": {"noise_seed": seed}},
            "sampler": {"class": "sampler", "inputs": {
                "noise": W("noise", 0), "guider": W("guider", 0),
                "sampler": W("ksel", 0), "sigmas": W("sigmas", 0),
                "latent_image": W("inplace_base", 0)}},
            # ---- STAGE B: latent-upsample x2 + re-anchor + refine ----
            "upscale_loader": {"class": "upscale_loader", "inputs": {
                "model_name": _ltx_hq_upscale_model_name()}},
            "upscaler": {"class": "upscaler", "inputs": {
                "samples": W("sampler", 0),
                "upscale_model": W("upscale_loader", 0),
                "vae": W("videovae_enc", 0)}},
            "inplace_refine": {"class": "inplace_refine", "inputs": {
                "vae": W("videovae_enc", 0), "image": W("preprocess", 0),
                "latent": W("upscaler", 0),
                "strength": _LTX_HQ_I2V_REFINE_STRENGTH, "bypass": False}},
            "cropguides": {"class": "cropguides", "inputs": {
                "positive": W("cond", 0), "negative": W("cond", 1),
                "latent": W("sampler", 0)}},
            "guider_refine": {"class": "guider_refine", "inputs": {
                "model": W("lora", 0), "positive": W("cropguides", 0),
                "negative": W("cropguides", 1), "cfg": _LTX_DISTILLED_CFG}},
            "sigmas_refine": {"class": "sigmas_refine", "inputs": {
                "values": list(LTX_HQ_REFINE_SIGMAS)}},
            "ksel_refine": {"class": "ksel_refine", "inputs": {
                "sampler_name": _LTX_HQ_REFINE_SAMPLER}},
            "noise_refine": {"class": "noise_refine",
                             "inputs": {"noise_seed": seed + 1}},
            "sampler_refine": {"class": "sampler_refine", "inputs": {
                "noise": W("noise_refine", 0), "guider": W("guider_refine", 0),
                "sampler": W("ksel_refine", 0), "sigmas": W("sigmas_refine", 0),
                "latent_image": W("inplace_refine", 0)}},
            "decode": {"class": "decode", "inputs": {
                "samples": W("sampler_refine", 0), "vae": W("videovae_dec", 0),
                "tile_size": 512, "overlap": 64,
                "temporal_size": 4096, "temporal_overlap": 8}},
        }
        return g

    def _dims(self, request):
        """(width, height) from the request canvas with LTX landscape defaults."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        canvas = get("canvas") or {}
        c_get = canvas.get if isinstance(canvas, dict) else (
            lambda k, d=None: getattr(canvas, k, d))
        w = int(c_get("w", 0) or 0) or _LTX_DEFAULT_W
        h = int(c_get("h", 0) or 0) or _LTX_DEFAULT_H
        return w, h

    def _build_graph(self, plan, length, width, height):
        """The declarative LTXV graph -- the frozen GGUF mini recipe reproduced
        EXACTLY (workflows/ltx_bookend_mini_repro_gguf_mit.json, the
        VERIFIED-WORKING path; the #1 splice invariant):
          UnetLoaderGGUF -> LoraLoaderModelOnly(distilled @0.70) -> CFGGuider.model
          LTXAVTextEncoderLoader(Gemma-3 + projection ckpt) -> CLIPTextEncode x2 ->
          LTXVConditioning; EmptyLTXVLatentVideo; (distilled) KSamplerSelect
          euler_cfg_pp + ManualSigmas 8-step + RandomNoise + CFGGuider cfg=1.0 +
          SamplerCustomAdvanced -> VAEDecodeTiled(512/64/4096/8, video VAE)."""
        from . import wrapper_bridge as _wb
        W = _wb.Wire
        mode = self._sampler_mode()
        # DISTILLED is the GGUF mini production recipe -- its values are HARDCODED
        # to the mini (the #1 invariant; env knobs ignored in distilled mode and
        # apply to the ksampler/manual A/B path only).
        if mode == "distilled":
            cfg = _LTX_DISTILLED_CFG                # 1.0 (mini CFGGuider)
            sampler_name = "euler_cfg_pp"           # mini KSamplerSelect
            lora_strength = _LTX_DISTILLED_LORA_STRENGTH   # 0.70 (mini LoRA)
        else:
            try:
                cfg = float(os.environ.get("OTR_LTX_CFG", str(_LTX_KSAMPLER_CFG)))
            except (TypeError, ValueError):
                cfg = _LTX_KSAMPLER_CFG
            sampler_name = self._sampler_name()
            try:
                lora_strength = float(os.environ.get(
                    "OTR_LTX_DISTILLED_LORA_STRENGTH",
                    str(_LTX_DISTILLED_LORA_STRENGTH)))
            except (TypeError, ValueError):
                lora_strength = _LTX_DISTILLED_LORA_STRENGTH
        positive = plan.get("text_prompt") or "a cinematic scene"
        negative = plan.get("negative_prompt") or os.environ.get(
            "OTR_LTX_NEGATIVE", _LTX_DEFAULT_NEGATIVE)
        seed = int(plan.get("seed", 0))
        lora_name, _lora_path = self._distilled_lora_file()
        graph = {
            # GGUF unet -> distilled LoRA (ALWAYS wired; the mini's @0.70). The
            # LoRA node returns a PATCHED reference to the base unet -- both are
            # kept in render_clip's VRAM keep-set so free_after_use never dangles
            # the patcher.
            "unet": {"class": "unet",
                     "inputs": {"unet_name": self._unet_name()}},
            "lora": {"class": "lora",
                     "inputs": {"lora_name": lora_name,
                                "strength_model": lora_strength,
                                "model": W("unet", 0)}},
            # Gemma-3 text encoder (reads the projection ckpt) + the video VAE.
            "te": {"class": "te",
                   "inputs": {"text_encoder": self._encoder_name(),
                              "ckpt_name": self._projection_ckpt(),
                              "device": self._encoder_device()}},
            "videovae": {"class": "videovae",
                         "inputs": {"vae_name": self._video_vae_name()}},
            "pos": {"class": "pos",
                    "inputs": {"text": positive, "clip": W("te", 0)}},
            "neg": {"class": "neg",
                    "inputs": {"text": negative, "clip": W("te", 0)}},
            "latent": {"class": "latent",
                       "inputs": {"width": int(width), "height": int(height),
                                  "length": int(length), "batch_size": 1}},
            "cond": {"class": "cond",
                     "inputs": {"positive": W("pos", 0), "negative": W("neg", 0),
                                "frame_rate": float(self.target_fps)}},
        }
        # The LoRA-wrapped unet IS the model for the sampler/guider (NEVER the raw
        # unet -- the LoRA patch must be in the forward).
        model_wire = W("lora", 0)
        if mode == "distilled":
            # The frozen mini sampling chain: KSamplerSelect(euler_cfg_pp) +
            # ManualSigmas (the 9 LTX_DISTILLED_SIGMAS, via the in-adapter
            # _SigmasFromValues injector) + RandomNoise(seed) + CFGGuider(1.0) +
            # SamplerCustomAdvanced. 8 sampling steps (the sigma schedule IS the
            # step count; OTR_LTX_STEPS is a ksampler-mode knob only).
            graph["samplersel"] = {"class": "samplersel",
                                   "inputs": {"sampler_name": sampler_name}}
            graph["sigmas"] = {"class": "sigmas",
                               "inputs": {"values":
                                          list(LTX_DISTILLED_SIGMAS)}}
            graph["noise"] = {"class": "noise",
                              "inputs": {"noise_seed": seed}}
            graph["guider"] = {"class": "guider",
                               "inputs": {"model": model_wire,
                                          "positive": W("cond", 0),
                                          "negative": W("cond", 1),
                                          "cfg": cfg}}
            graph["sampleradv"] = {
                "class": "sampleradv",
                "inputs": {"noise": W("noise", 0), "guider": W("guider", 0),
                           "sampler": W("samplersel", 0),
                           "sigmas": W("sigmas", 0),
                           "latent_image": W("latent", 0)}}
            samples = W("sampleradv", 0)
        else:
            steps = int(os.environ.get("OTR_LTX_STEPS", "30"))
            graph["ksampler"] = {
                "class": "ksampler",
                "inputs": {"seed": seed, "steps": steps,
                           "cfg": cfg, "sampler_name": sampler_name,
                           "scheduler": "normal", "denoise": 1.0,
                           "model": model_wire,
                           "positive": W("cond", 0),
                           "negative": W("cond", 1),
                           "latent_image": W("latent", 0)}}
            samples = W("ksampler", 0)
        # VAEDecodeTiled (the mini's 512/64/4096/8) on the video VAE.
        graph["vaedecode"] = {"class": "vaedecode",
                              "inputs": {"samples": samples,
                                         "vae": W("videovae", 0),
                                         "tile_size": 512, "overlap": 64,
                                         "temporal_size": 4096,
                                         "temporal_overlap": 8}}
        return graph

    # ---- residency (resolve the installed wrapper nodes; weights load on call) -
    def load(self):
        """Resolve the installed ComfyUI node classes for the ACTIVE sampling set
        (the distilled GGUF chain by default), fail-closed NAMED (WrapperNodeMissing)
        if any is absent. The heavy weight load happens when the loader nodes
        execute in render_clip (ComfyUI's own model management); assert_usable
        already gated the 5 weight files + floors. The live VRAM<=14.5 GB peak +
        SageAttention-clean determinism are the GPU motion smoke (operator)."""
        from . import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(
            self._node_candidates_sampling())
        self._loaded = True

    def render_clip(self, request, prepared):
        """Drive ONE text->video clip via the in-process LTX wrapper graph, encode
        the decoded IMAGE batch to a SILENT bt709 clip (V-1), retain the MODEL
        patcher for V-4 teardown, and assert the mid-render NVML ceiling. Returns
        the raw ``{out_path, frame_count}`` canonicalize() normalises. Fail-closed
        NAMED if a wrapper node is missing."""
        from . import wrapper_bridge as _wb
        from ._tmp import otr_engine_tmp_mp4
        plan = self._build_render_request(request)            # pure, CPU-tested
        recipe = self._recipe()
        if recipe == RECIPE_HQ_TWO_STAGE:
            return self._render_clip_hq(request, prepared, plan)
        use_i2v = self._use_i2v(request)
        # LK-1b: resolve the ACTIVE candidate set (sampler mode + LoRA
        # wiring are env/ckpt-dependent). A load()-time cache (or a test's
        # injected fakes) wins on the non-i2v path; resolve_graph_classes
        # fail-LOUD names every missing wrapper class at once.
        if use_i2v:
            classes = _wb.resolve_graph_classes(self._node_candidates_i2v())
        else:
            classes = dict(getattr(self, "_classes", None)
                           or _wb.resolve_graph_classes(
                               self._node_candidates_sampling()))
        if self._sampler_mode() == "distilled":
            # The SIGMAS source is in-adapter (not a registered node class);
            # injected AFTER resolve so the resolver never sees it. A test
            # fake may pre-supply its own "sigmas" class.
            classes.setdefault("sigmas", _SigmasFromValues)
        width, height = self._dims(request)
        # LTX requires length == 8n+1 and dims multiple-of-32; the frame ask is
        # floored/capped/snapped by the pure helper (round 5 F1: a >cap ask is
        # rendered AT the cap and the composite hold-fills the beat window).
        # The decode floor governs BOTH paths (i2v included -- do not touch).
        length = _ltx_frame_length(plan["target_frame_count"], self.target_fps)
        width = max(32, (width // 32) * 32)
        height = max(32, (height // 32) * 32)
        # BUG-LOCAL-117d boomerang restore: when looping, render the HALF source
        # length here (after dims, before graph build) -- the in-tensor mirror
        # below doubles it back to >= the beat window. Leaves the global
        # _ltx_frame_length / 169 decode floor UNTOUCHED (this path has its own
        # proven native-canvas floor).
        loop_via_reverse = self._loop_via_reverse()
        if loop_via_reverse:
            length = _ltx_loop_source_length(plan["target_frame_count"],
                                             self.target_fps)
            _LOG.warning("[eng_ltx_video] loop_via_reverse ON: render src=%d -> "
                         "mirror %d frames (target %d) @ %dx%d", length,
                         2 * length - 1, plan["target_frame_count"], width, height)
        if use_i2v:
            init_path = self._init_image_path(request)
            image_name = _wb.stage_into_comfy_input(init_path)
            _LOG.warning(
                "[eng_ltx_video] LTX-I2V: conditioning on init image %s "
                "(%dx%d, length %d)", os.path.basename(init_path),
                width, height, length)
            graph = self._build_graph_i2v(plan, length, width, height,
                                          image_name)
        else:
            graph = self._build_graph(plan, length, width, height)
        # free_after_use (2026-06-09 capstone catch + GGUF splice): the Gemma-3
        # text encoder + the projection ckpt + the intermediates must NOT stay
        # co-resident with the GGUF UNET through the sampler (the 16 GB ceiling).
        # The bridge frees each intermediate (te/conds/latent) once its last
        # consumer ran. KEEP both "unet" and "lora": the LoRA node returns a
        # PATCHED reference to the base unet, so evicting "unet" under
        # free_after_use would DANGLE the patcher (the sampler would have no
        # weights). The terminal is kept for the IMAGE read-out.
        results = _wb.run_graph(graph, classes, free_after_use=True,
                                keep={"unet", "lora", self._TERMINAL})
        images = results[self._TERMINAL][0]                   # VAEDecodeTiled IMAGE batch
        bucket = prepared.setdefault("patchers", self._patchers) \
            if isinstance(prepared, dict) else self._patchers
        # The LoRA-wrapped unet is the resident MODEL patcher (V-4 teardown).
        model = results.get("lora", (None,))[0]
        if model is not None and callable(getattr(model, "detach", None)) \
                and id(model) not in {id(p) for p in bucket}:
            bucket.append(model)
        frames = _wb.images_to_uint8(images)
        if loop_via_reverse:
            if len(frames) >= 2:
                frames = _boomerang_frames(frames)
            else:
                _LOG.warning("[eng_ltx_video] loop_via_reverse: only %d frame(s) "
                             "decoded -- skipping mirror", len(frames))
        out_path = otr_engine_tmp_mp4("otr_ltx_")
        path, n = _wb.encode_frames_to_silent_mp4(frames, out_path, self.target_fps)
        return {"out_path": path, "frame_count": n,
                "ltx_loop_via_reverse": bool(loop_via_reverse)}

    def _render_clip_hq(self, request, prepared, plan):
        """S5: drive ONE silent two-stage HQ clip (hq_two_stage recipe). The
        init still is REQUIRED -- a missing/stale still raises LOUD (NO
        FALLBACK to the text path: the HQ ladder anchors BOTH stages on the
        still). Same length/dims/loop/encode/ceiling mechanics as the
        single-pass path."""
        from . import wrapper_bridge as _wb
        from ._tmp import otr_engine_tmp_mp4
        init_path = self._init_image_path(request)
        if not (init_path and os.path.exists(init_path)):
            raise _wb.GraphExecutionError(
                "%s (hq_two_stage) requires an on-disk init image for every "
                "beat; got %r -- fix the still mint upstream or select "
                "OTR_LTX_VIDEO_RECIPE=single_pass (NO silent t2v downgrade)"
                % (self.name, init_path))
        classes = _wb.resolve_graph_classes(self._node_candidates_hq())
        # BOTH sigma sources are in-adapter literals (not registered nodes);
        # injected AFTER resolve. A test fake may pre-supply its own.
        classes.setdefault("sigmas", _SigmasFromValues)
        classes.setdefault("sigmas_refine", _SigmasFromValues)
        width, height = self._dims(request)
        length = _ltx_frame_length(plan["target_frame_count"], self.target_fps)
        width = max(32, (width // 32) * 32)
        height = max(32, (height // 32) * 32)
        loop_via_reverse = self._loop_via_reverse()
        if loop_via_reverse:
            length = _ltx_loop_source_length(plan["target_frame_count"],
                                             self.target_fps)
            _LOG.warning("[eng_ltx_video] HQ loop_via_reverse ON: render "
                         "src=%d -> mirror %d frames (target %d) @ %dx%d",
                         length, 2 * length - 1, plan["target_frame_count"],
                         width, height)
        image_name = _wb.stage_into_comfy_input(init_path)
        _LOG.warning(
            "[eng_ltx_video] S5 HQ two-stage: base %dx%d -> upsample x2 -> "
            "refine %dx%d, length %d, init %s (silent; recipe=hq_two_stage)",
            width // 2, height // 2, width, height, length,
            os.path.basename(init_path))
        graph = self._build_graph_hq(plan, length, width, height, image_name)
        results = _wb.run_graph(graph, classes, free_after_use=True,
                                keep={"unet", "lora", "decode"})
        images = results["decode"][0]
        bucket = prepared.setdefault("patchers", self._patchers) \
            if isinstance(prepared, dict) else self._patchers
        model = results.get("lora", (None,))[0]
        if model is not None and callable(getattr(model, "detach", None)) \
                and id(model) not in {id(p) for p in bucket}:
            bucket.append(model)
        frames = _wb.images_to_uint8(images)
        if loop_via_reverse:
            if len(frames) >= 2:
                frames = _boomerang_frames(frames)
            else:
                _LOG.warning("[eng_ltx_video] HQ loop_via_reverse: only %d "
                             "frame(s) decoded -- skipping mirror", len(frames))
        out_path = otr_engine_tmp_mp4("otr_ltx_")
        path, n = _wb.encode_frames_to_silent_mp4(frames, out_path,
                                                  self.target_fps)
        return {"out_path": path, "frame_count": n,
                "ltx_recipe": RECIPE_HQ_TWO_STAGE,
                "ltx_loop_via_reverse": bool(loop_via_reverse)}

    def canonicalize(self, raw, request, profile):
        """Normalize a rendered clip into the ALWAYS-SILENT bt709 / yuv420p
        CanonicalClip contract (frame_count is the integer timing authority)."""
        return self._clip_from_raw(raw, request)

    # ---- pure helpers (CPU-testable; no wrapper, no heavy import) ----
    def _build_render_request(self, request):
        """Pure: the normalized inference request the LTX wrapper consumes, from
        a VideoRequest-shaped object OR a plain dict. Deterministic (seed flows
        straight through) -- the render-twice determinism contract (V-7)."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        timing = get("timing") or {}
        t_get = timing.get if isinstance(timing, dict) else (
            lambda k, d=None: getattr(timing, k, d))
        seeds = get("seed_bundle") or {}
        s_get = seeds.get if isinstance(seeds, dict) else (
            lambda k, d=None: getattr(seeds, k, d))
        return {
            "text_prompt": get("text_prompt") or "",
            "negative_prompt": get("negative_prompt") or "",
            "fps": int(self.target_fps),
            "target_frame_count": int(t_get("target_frame_count", 0) or 0),
            "seed": int(s_get("request_seed", 0) or 0),
        }

    def _clip_from_raw(self, raw, request):
        """Pure: shape a worker / wrapper result into the silent CanonicalClip
        dict (bt709 / yuv420p; frame_count is the integer timing authority)."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        raw = raw or {}
        return {
            "clip_id": get("shot_id") or get("request_id") or "ltx_video_clip",
            "type": "video", "path": raw.get("out_path", ""),
            "container": "mp4", "codec": "h264", "pixel_format": "yuv420p",
            "fps": int(self.target_fps),
            "frame_count": int(raw.get("frame_count", 0) or 0),
            "has_audio": False,            # V-1: only OTR_MasterAudioMux emits audio
            "color_primaries": "bt709", "transfer": "bt709", "matrix": "bt709",
            "engine_id": self.name, "family": self.family,
        }


__all__ = ["LtxVideoEngine"]
