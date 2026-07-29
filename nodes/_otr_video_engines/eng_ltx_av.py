"""LTX-2.3 AUDIO-INPUT (A2V) lane -- additive, in-process, default-OFF / dark.

A NEW, DARK, ADDITIVE engine pair that drives video from the per-beat slice of the
FROZEN master audio + a text prompt (+ a FLUX still for the talk lane). It is NOT
the golden prompt-only ``ltx_video`` engine and shares NO code or env with it --
the two lanes diverge on purpose (this lane snaps frames UP via
``av_dims.next_8n1``; ``eng_ltx_video`` snaps DOWN). ``eng_ltx_video.py`` is FROZEN
and is never imported or touched here.

ONE adapter over the shared core (M0-GROUNDED graph; Gemma-3 encoder offloaded to
CPU). Per-beat recipe / quant / canvas / measured VRAM peak are logged LOUD by
render_clip and threaded into the clip raw (S-B observability); the
fit-vs-quality recipe pick + the measured peaks live in the bakeoff manifest, NOT
a hardcoded number here (see docs/ the LTX-AV GGUF quant bakeoff + each run log):

* ``ltx_audio_in`` -- the ONE audio-in lane; roles (announcer_visual,
  music_visual, character_video); family ``audio_conditioned_video``; required
  text_prompt + audio_ref + init_image; I2V on the beat's WIDE scene still + the
  audio slice (music OR voice); the per-role default for the music/announcer
  bookends; NO fallback (fail LOUD). The old ltx_av_talk/ltx_av_music split was
  REMOVED 2026-06-26 -- the talk-vs-scene routing lives on the BEAT ROLE in
  render_driver (``_is_character_face_beat``), not on two engines.

V-1 absolute: the lane DISCARDS LTX's audio side entirely -- the graph terminates
at ``LTXVSeparateAVLatent -> video_latent -> VAEDecodeTiled`` (the audio_latent
branch + ``LTXVAudioVAEDecode`` are never wired), the clip is ALWAYS silent
(has_audio False), and only ``OTR_MasterAudioMux`` emits audio.
``test_audio_byte_identical`` stays green.

Cold-import clean (V-12): module scope imports only stdlib + the dep-free shared
helpers + the registry. torch / the LTX wrapper nodes are imported LAZILY inside
``load`` / ``render_clip`` (the GPU slice), never here. NVML is REQUIRED for this
lane (heaviest engine) -- assert_usable fails CLOSED when NVML is absent so the
ceiling guard can never silently no-op. UTF-8, no BOM, ASCII-only source.

Config (env; each resolves via ComfyUI folder_paths so a box never needs a code
edit): OTR_ENABLE_LTX_AV (opt-in flag); OTR_LTX_AV_UNET (GGUF unet in models/unet);
OTR_LTX_AV_TEXT_ENCODER (Gemma-3 in text_encoders); OTR_LTX_AV_PROJECTION_CKPT
(the LTX ckpt supplying the text-projection, in checkpoints); OTR_LTX_AV_VIDEO_VAE
+ OTR_LTX_AV_AUDIO_VAE (in vae). RESTART ComfyUI after any mid-render cancel
(a wedged PID holds the AS-3 lease ~120 s; reclaim only frees dead PIDs).
"""
from __future__ import annotations

import contextlib
import logging
import os

from .._otr_shared import av_dims as _AVD
from .._otr_shared.still_plan_helpers import StillPlanRow
from .frame_contract import CONTINUITY_SOFT_REFERENCE, FrameContract
from . import motion_common as _MC
from .registry import EngineUnusable, EngineUsabilityReason, register
from .wan_shared import ffprobe_clip_fields, validate_silent_clip_contract

_LOG = logging.getLogger("OTR.eng_ltx_av")


def _env_num(name, default, cast):
    """``cast(os.environ[name])`` that can never take the IMPORT down.

    THE DEFECT THIS REPLACES (2026-07-27, chunk 7b slice 1; both r2 kibitz
    seats reached it independently). These four constants parsed the
    environment at MODULE SCOPE with a bare ``int()`` / ``float()``, so a
    malformed value did not fail a render closed with a named message -- it
    raised ``ValueError`` while the module was still importing. This adapter
    then never registered, and ``frame_contract.frame_contract_for`` resolves
    an adapter it cannot reach to ``SINGLE_ONLY`` through its own swallowed
    ``except Exception`` (``frame_contract.py:243-247``). Net effect of one
    typo in one environment variable: the engine silently vanished and its
    lane quietly reverted to unbounded single-clip behaviour, with nothing in
    the log naming the variable.

    LOUD warning plus the declared default instead. Deliberately NOT a
    behaviour change for a WELL-FORMED value -- whether the environment may
    move a declared ceiling at all is the 7b fork, settled separately in
    ``docs/2026-07-27-multiclip-7b-fork-r3-coding-plan.md``. This slice fixes
    only the crash, and keeps these names module-level VALUES because
    ``tests/test_engine_contract_roster.py:184`` reads
    ``_LTX_AV_MAX_FRAMES`` as one.
    """
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return cast(default)
    try:
        return cast(str(raw).strip())
    except (TypeError, ValueError):
        _LOG.warning("[eng_ltx_av] invalid %s=%r -- using default %r",
                     name, raw, default)
        return cast(default)


# --- frame + canvas grounding (LTX-AV lane only; snap UP, never copy Lane A) ---
_LTX_AV_MIN_FRAMES = _AVD._LTX_MIN_FRAMES        # 9 (8n+1 floor)
_LTX_AV_MAX_FRAMES = _env_num("OTR_LTX_AV_MAX_FRAMES", 497, int)  # M0 initial
_LTX_AV_NATIVE_W = 832
_LTX_AV_NATIVE_H = 480
# Default sampler recipe (the M0-proven 8-step distilled-ish base pass; cfg 3.0
# matched the probe). All env-overridable; never shared with eng_ltx_video.
_LTX_AV_STEPS = _env_num("OTR_LTX_AV_STEPS", 8, int)
_LTX_AV_CFG = _env_num("OTR_LTX_AV_CFG", 3.0, float)
_LTX_AV_I2V_STRENGTH = _env_num("OTR_LTX_AV_I2V_STRENGTH", 1.0, float)
# ASCII-only negative (CLAUDE.md). One shared constant; cap 240 in the driver.
_LTX_DEFAULT_NEGATIVE = (
    "low quality, worst quality, blurry, jpeg artifacts, distorted, deformed, "
    "static, frozen pose, still image, watermark, text")

# Decode tiling (LTX-AV quality wire, 2026-06-27): the WHOLE-CLIP temporal decode
# (VAEDecodeTiled temporal_size 4096 / overlap 8) eliminates the inter-tile
# "flash" seam the bakeoff measured (seam p99 0.235 -> 0.0) at the SAME s/it and
# peak VRAM as the legacy tiled decode -- the operator's PROVISIONAL winner pick
# (operator eyeballs the final clip; the bakeoff-converged tiled 128/32 is the
# LOUD env fallback if a smoke trips the 14.5 GB ceiling). Read INSIDE
# _build_graph so a monkeypatch needs no module reload. Spatial tile 512 /
# overlap 64 stays fixed.
_LTX_AV_DECODE_TEMPORAL_SIZE_DEFAULT = 4096
_LTX_AV_DECODE_TEMPORAL_OVERLAP_DEFAULT = 8


def _decode_temporal_knobs():
    """Resolve ``(temporal_size, temporal_overlap)`` for VAEDecodeTiled from the
    env, FAIL-LOUD (NO clamp). Absent -> the whole-clip defaults (4096/8).
    Present but non-int -> raise; ``size <= 0`` -> raise; ``overlap < 0`` ->
    raise; ``overlap >= size`` -> raise. A NAMED ValueError fires BEFORE the graph
    is built so a bad box config fails loud instead of silently degrading."""
    def _read_int(name, default):
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            return int(default)
        try:
            return int(str(raw).strip())
        except (TypeError, ValueError):
            raise ValueError(
                "eng_ltx_av: %s=%r is not an integer (decode temporal knob; "
                "no clamp -- fix the env)" % (name, raw))
    size = _read_int("OTR_LTX_AV_DECODE_TEMPORAL_SIZE",
                     _LTX_AV_DECODE_TEMPORAL_SIZE_DEFAULT)
    overlap = _read_int("OTR_LTX_AV_DECODE_TEMPORAL_OVERLAP",
                        _LTX_AV_DECODE_TEMPORAL_OVERLAP_DEFAULT)
    if size <= 0:
        raise ValueError(
            "eng_ltx_av: OTR_LTX_AV_DECODE_TEMPORAL_SIZE=%d must be > 0 "
            "(no clamp)" % size)
    if overlap < 0:
        raise ValueError(
            "eng_ltx_av: OTR_LTX_AV_DECODE_TEMPORAL_OVERLAP=%d must be >= 0 "
            "(no clamp)" % overlap)
    if overlap >= size:
        raise ValueError(
            "eng_ltx_av: OTR_LTX_AV_DECODE_TEMPORAL_OVERLAP=%d must be < "
            "OTR_LTX_AV_DECODE_TEMPORAL_SIZE=%d (no clamp)" % (overlap, size))
    return size, overlap

# Weight sanity floors (GiB) -- catch a truncated / wrong download, NOT exact
# byte checks. Q3_K_M unet ~9.3 GiB; Gemma-3 fp4 ~8.2 GiB; video VAE ~1.35 GiB.
_GiB = 1024 ** 3
_FLOOR_UNET = 8 * _GiB
_FLOOR_ENCODER = 6 * _GiB
_FLOOR_VIDEO_VAE = 1 * _GiB
_FLOOR_AUDIO_VAE = int(0.2 * _GiB)
_FLOOR_LORA = 5 * _GiB
_FLOOR_PROJECTION_CKPT = 30 * _GiB

# --- VRAM reserve (2026-06-26): force a PARTIAL unet load so the 22B AV unet
# leaves room for the audio-conditioned activation peak instead of cramming the
# card full and spilling to system RAM (the 6.84-vs-223 s/it knife-edge on a 16GB
# box whose desktop apps already eat ~5GB). OTR_LTX_AV_RESERVE_VRAM_GB (default
# 4.0) is held free during run_graph via ComfyUI's real EXTRA_RESERVED_VRAM global
# (the same lever as --reserve-vram), restored in finally so a LOUD render failure
# never leaks it into the next engine. =0 disables. Works in BOTH the GUI and the
# headless path (a boot-only CLI arg cannot reach the Desktop app). 4.0 NOT 3.0:
# on a 16GB card 3.0 left usable (~10958MB) still > the 10537MB unet, so it
# FULL-loaded + the activation peak grazed the 435MB-free cliff (~72 s/it marginal
# spill, 2026-06-26 live). 4.0 pushes usable BELOW the unet -> a true PARTIAL load
# with ~4.4GB activation room -> steady, no spill. Raise it if a heavier desktop
# still spills.
_LTX_AV_RESERVE_VRAM_GB = float(os.environ.get("OTR_LTX_AV_RESERVE_VRAM_GB", "4.0"))


@contextlib.contextmanager
def _ltx_av_vram_reserve():
    """Hold OTR_LTX_AV_RESERVE_VRAM_GB free across the LTX-AV graph run so the
    unet loads partially. Pure no-op when the reserve is <=0, ComfyUI is absent,
    or the global is already higher. Exception-safe (restores in finally)."""
    gb = _LTX_AV_RESERVE_VRAM_GB
    if gb <= 0:
        yield
        return
    try:
        from comfy import model_management as _mm
    except Exception:  # noqa: BLE001
        yield
        return
    old = getattr(_mm, "EXTRA_RESERVED_VRAM", None)
    target = int(gb * 1024 * 1024 * 1024)
    bumped = False
    try:
        if old is not None and target > old:
            _mm.EXTRA_RESERVED_VRAM = target
            bumped = True
            _LOG.warning("[eng_ltx_av] reserving %.1f GB VRAM for the LTX-AV "
                         "render (partial unet load; was %.0f MB)",
                         gb, old / 1024 / 1024)
        yield
    finally:
        if bumped:
            _mm.EXTRA_RESERVED_VRAM = old

# --- recipe selector: the recipe FOLLOWS THE MODEL (2026-06-26 bakeoff) --------
# The A2V lane runs ONE of THREE recipes, chosen at graph-build, NEVER an in-render
# fallback (CLAUDE.md NO FALLBACKS -- fail LOUD):
#   sharp_lora        dev GGUF + distilled LoRA @0.70 + euler_cfg_pp + the fixed
#                     8-step LTX_DISTILLED_SIGMAS + cfg 1.0 + i2v strength 0.75;
#                     ModelSamplingLTXV + LTXVScheduler DROPPED (the LoRA carries
#                     the shift -- ModelSamplingLTXV would double-shift -> blur).
#                     The kept HERO/final path.
#   distilled_native  distilled-1.1 GGUF (distillation BAKED IN): the same
#                     distilled recipe as sharp_lora but with NO LoRA -- the unet
#                     feeds the guider directly. The new DEFAULT daily driver.
#   m0_base           ModelSamplingLTXV + LTXVScheduler + euler + cfg 3.0 +
#                     strength 1.0, no LoRA. Legacy base pass, OVERRIDE-ONLY.
# Selection: env OTR_LTX_AV_RECIPE (auto|ia2v_canonical|sharp_lora|
# distilled_native|m0_base), default "auto" -> derived from the OTR_LTX_AV_UNET
# basename family (dev -> ia2v_canonical since 2026-07-02; distilled-1.1 ->
# distilled_native). The single-pass recipes mirror eng_ltx_video's distilled
# chain; that module is FROZEN and never imported here, so the tiny helpers
# below are DUPLICATED on purpose (V-12 cold-import).
RECIPE_AUTO = "auto"
RECIPE_SHARP_LORA = "sharp_lora"
RECIPE_DISTILLED_NATIVE = "distilled_native"
RECIPE_M0_BASE = "m0_base"
#: ia2v_canonical (2026-07-02, operator GO after the isolation smoke): the
#: comfy.org "LTX-2.3: Image Audio to Video" LIP-SYNC recipe, transplanted
#: node-for-node (docs/2026-07-02-canonical-ia2v/). TWO-STAGE: motion at half
#: canvas, latent-upsample x2, audio re-concat, 3-step refine. The NEW dev
#: -family default -- the single-pass recipes froze the grille-mouth (i2v
#: strength 1.0 hard-pins the still; full distillation blunts audio coupling).
RECIPE_IA2V = "ia2v_canonical"
_VALID_RECIPES = (RECIPE_SHARP_LORA, RECIPE_DISTILLED_NATIVE, RECIPE_M0_BASE,
                  RECIPE_IA2V)
_RECIPE_MENU = "auto|ia2v_canonical|sharp_lora|distilled_native|m0_base"


#: Distilled sigma schedule (ComfyUI-Goofer, GPU-proven on the RTX 5080). 8 steps;
#: last sigma 0.0 = full denoise. Duplicated from eng_ltx_video (frozen), V-12.
LTX_DISTILLED_SIGMAS = (
    1.0, 0.99375, 0.9875, 0.98125, 0.975,
    0.909375, 0.725, 0.421875, 0.0,
)
_LTX_AV_SHARP_CFG = 1.0
_LTX_AV_SHARP_SAMPLER = "euler_cfg_pp"
_LTX_AV_SHARP_I2V_STRENGTH = 0.75
#: The distilled LoRA (the same artifact eng_ltx_video wires @0.70 on the 22B GGUF).
_LTX_AV_DISTILLED_LORA_DEFAULT = os.path.join(
    "ltxv", "ltx2", "ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
_LTX_AV_DISTILLED_LORA_STRENGTH = 0.7

# ---- ia2v_canonical constants (verbatim from the canonical graph; the live
# ---- 2026-07-02 isolation smoke on OUR still is the ground truth) ----
#: Refine ladder: 3 steps from 0.85 -- detail only, never re-decides motion.
IA2V_REFINE_SIGMAS = (0.85, 0.7250, 0.4219, 0.0)
#: The canonical uses the ORIGINAL (non-1.1) distilled-lora-384 at HALF
#: strength on the DEV unet: half-distilled keeps dev's audio coupling while
#: buying most of the distilled speed.
_IA2V_LORA_NAME = os.path.join(
    "ltxv", "ltx2", "ltx-2.3-22b-distilled-lora-384.safetensors")
_IA2V_LORA_STRENGTH = 0.5
_IA2V_I2V_BASE_STRENGTH = 0.7      # motion pass: the still is a soft anchor
_IA2V_I2V_REFINE_STRENGTH = 1.0    # detail pass: hard re-anchor
_IA2V_BASE_SAMPLER = "euler_ancestral_cfg_pp"   # ancestral keeps motion alive
#: Guide-image conditioning chain (canonical texture prep). CANVAS-INDEPENDENT
#: (lips-dont-talk kibitz, probes P7/P8 vs production): the canonical scales
#: the guide to a FIXED 1920x1088 then caps the longer edge at 1536 -- a
#: sharp, DOWNSCALED guide at any render canvas. The first transplant scaled
#: 1.5x the RENDER canvas instead, so a 512x288 production render built a
#: 768x432 guide and UPSCALED it to 1536: a soft, double-resampled mouth
#: prior (every probe articulated because the probe harness kept the fixed
#: canonical numbers -- the engine did not).
_IA2V_GUIDE_SCALE_W = 1920
_IA2V_GUIDE_SCALE_H = 1088
_IA2V_GUIDE_LONGER_EDGE = 1536
_IA2V_GUIDE_COMPRESSION = 18
#: Spatial latent upsampler x2 (folder key latent_upscale_models -- mapped in
#: scripts/_otr_headless_model_paths.yaml). Required ONLY by ia2v_canonical so
#: the other recipes stay usable on installs lacking it.
_FLOOR_UPSCALER = int(0.05 * _GiB)


def _ia2v_upscale_model_name():
    return os.environ.get("OTR_LTX_AV_UPSCALE_MODEL",
                          "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")


def _recipe_config(recipe):
    """Static per-recipe config consumed UNIFORMLY by _weight_paths /
    _node_candidates / _build_graph / render_clip -- never a binary 'sharp' bool,
    so the three states (and the three director roles) stay consistent.

    use_lora           -> require + wire LoraLoaderModelOnly (sharp_lora only).
    use_modelsampling  -> require + wire ModelSamplingLTXV + LTXVScheduler (m0 only).
    manual_sigmas      -> inject the in-adapter fixed LTX_DISTILLED_SIGMAS
                          (sharp_lora AND distilled_native); else LTXVScheduler."""
    if recipe == RECIPE_SHARP_LORA:
        return {"use_lora": True, "use_modelsampling": False, "manual_sigmas": True,
                "sampler": _LTX_AV_SHARP_SAMPLER, "cfg": _LTX_AV_SHARP_CFG,
                "i2v_strength": _LTX_AV_SHARP_I2V_STRENGTH, "two_stage": False,
                "lora_name": _LTX_AV_DISTILLED_LORA_DEFAULT,
                "lora_strength": _LTX_AV_DISTILLED_LORA_STRENGTH}
    if recipe == RECIPE_DISTILLED_NATIVE:
        return {"use_lora": False, "use_modelsampling": False, "manual_sigmas": True,
                "sampler": _LTX_AV_SHARP_SAMPLER, "cfg": _LTX_AV_SHARP_CFG,
                "i2v_strength": _LTX_AV_SHARP_I2V_STRENGTH, "two_stage": False}
    if recipe == RECIPE_M0_BASE:
        return {"use_lora": False, "use_modelsampling": True, "manual_sigmas": False,
                "sampler": "euler", "cfg": _LTX_AV_CFG,
                "i2v_strength": _LTX_AV_I2V_STRENGTH, "two_stage": False}
    if recipe == RECIPE_IA2V:
        # the canonical comfy.org IA2V lip-sync recipe (2026-07-02 transplant):
        # BASE (motion) = half canvas + Inplace i2v @0.7 + zero-masked audio
        # latent + euler_ancestral_cfg_pp over the front-loaded 8-step ladder;
        # REFINE (detail) = latent-upsample x2 + re-anchor @1.0 + audio
        # re-concat + 3-step euler_cfg_pp with LTXVCropGuides.
        return {"use_lora": True, "use_modelsampling": False, "manual_sigmas": True,
                "sampler": _IA2V_BASE_SAMPLER, "cfg": _LTX_AV_SHARP_CFG,
                "i2v_strength": _IA2V_I2V_BASE_STRENGTH, "two_stage": True,
                "lora_name": _IA2V_LORA_NAME,
                "lora_strength": _IA2V_LORA_STRENGTH,
                "refine_sampler": _LTX_AV_SHARP_SAMPLER,
                "refine_sigmas": IA2V_REFINE_SIGMAS,
                "refine_strength": _IA2V_I2V_REFINE_STRENGTH}
    raise ValueError("unknown LTX-AV recipe %r" % recipe)


class _SigmasFromValues:
    """In-adapter SIGMAS-from-literal-values node (sharp mode). Duplicated from
    eng_ltx_video (frozen) per V-12 so we never guess ManualSigmas' widget API.
    Injected into the resolved-classes map by render_clip; lazy torch import."""

    FUNCTION = "get"

    def get(self, values):
        import torch  # lazy -- only inside an actual GPU render
        return (torch.tensor([float(v) for v in values],
                             dtype=torch.float32),)


def _resolve(folder, name):
    """Resolve a model filename to a full path via ComfyUI folder_paths (honors
    extra_model_paths.yaml), with a best-effort join fallback for the headless /
    CPU existence check (no folder_paths registered)."""
    if not name:
        return ""
    try:
        import folder_paths  # type: ignore
        p = folder_paths.get_full_path(folder, name)
        if p:
            return p
    except Exception:  # noqa: BLE001 - headless/CPU
        pass
    here = os.path.abspath(__file__)
    comfy_models = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.dirname(here))))), "models", folder, name)
    return comfy_models


class _LtxAvBase(_MC.MotionEngineBase):
    """Shared LTX-AV core: assert_usable gate, graph spec, render lifecycle.

    Subclasses set the lane identity (name / family / roles / required_inputs /
    fallback_engine) and ``_is_talk``. The I2V-vs-t2v branch is INTERNAL to
    ``_build_graph`` so talk + music share one resident load path."""

    default_roles = ()                  # subclasses set their default roles
    commercial_clean = True             # Apache GGUF + LTX-2 Community + distilled LoRA
    requires_flag = None  # vestigial (registry IS the menu; no flag gate)
    engine_version = "1"
    declared_isolation = _MC.ISOLATION_IN_PROCESS
    target_fps = 25
    # LTX-AV renders WIDE (832x480 native / 1472x832 canvas). Declaring the aspect
    # makes OTR_VideoDirector mint a WIDE character still to match -- without it the
    # director defaulted to a 832x1216 PORTRAIT still that the wide render then
    # centre-cropped, lopping the subject's head off (operator catch 2026-06-17).
    render_aspect = "wide"
    _is_talk = False
    _TERMINAL = "decode"

    # ---- config resolution (env override -> folder_paths -> join) ----
    def _unet_name(self):
        return os.environ.get("OTR_LTX_AV_UNET", "ltx-2.3-22b-dev-Q3_K_M.gguf")

    def _quant_label(self):
        """The GGUF quant / precision token from the unet basename (e.g.
        ``Q3_K_M`` / ``Q2_K`` / ``fp8``) for the per-beat observability line + the
        S-E recipe stamp; ``""`` when the basename carries none. Pure."""
        import re
        m = re.search(r"(Q\d+(?:_[A-Za-z0-9]+)*|fp8|fp16|bf16)",
                      os.path.basename(str(self._unet_name())))
        return m.group(1) if m else ""

    def _encoder_name(self):
        return os.environ.get("OTR_LTX_AV_TEXT_ENCODER",
                              "gemma_3_12B_it_fp4_mixed.safetensors")

    def _projection_ckpt(self):
        return os.environ.get("OTR_LTX_AV_PROJECTION_CKPT",
                              "ltx-2.3-22b-dev.safetensors")

    def _video_vae_name(self):
        return os.environ.get("OTR_LTX_AV_VIDEO_VAE",
                              "ltx-2.3-22b-dev_video_vae.safetensors")

    def _audio_vae_name(self):
        return os.environ.get("OTR_LTX_AV_AUDIO_VAE",
                              "ltx-2.3-22b-dev_audio_vae.safetensors")

    def _distilled_lora_name(self, rcfg=None):
        """The distilled LoRA filename; env-overridable. PER-RECIPE default:
        sharp_lora keeps the 1.1 LoRA, ia2v_canonical uses the canonical
        original-384 (@0.5, set in the recipe config). Resolution + floor are
        handled by _resolve('loras', ...) like the other artifacts."""
        if rcfg is None:
            rcfg = _recipe_config(self._recipe())
        return os.environ.get(
            "OTR_LTX_AV_DISTILLED_LORA",
            rcfg.get("lora_name", _LTX_AV_DISTILLED_LORA_DEFAULT))

    def wants_talking_prompt(self):
        """True when the active recipe is the canonical two-stage IA2V
        lip-sync graph. The render driver consults THIS hook (never an env
        string) to swap the bookend/character motion prompts to the TALKING
        register -- probe P4 (2026-07-02, docs/2026-07-02-canonical-ia2v/)
        measured the scene register collapsing articulation 4.15 -> 1.18 and
        hallucinating on-video title text. RAISES on recipe misconfig (kibitz
        r2: no silent double-swallow) -- the driver's caller catches + logs
        LOUD, and assert_usable still hard-fails the render itself."""
        return bool(_recipe_config(self._recipe())["two_stage"])

    # ---- recipe resolution (fail-loud; the recipe FOLLOWS THE MODEL) ----
    def _recipe(self):
        """Resolve the active recipe. Read fresh every call (an operator flips
        daily<->hero per beat by swapping OTR_LTX_AV_UNET / OTR_LTX_AV_RECIPE).
        NO FALLBACKS: a retired flag, a bad override, or an unrecognized unet
        RAISE -- never guess, never warn-and-continue, never double-distill."""
        if os.environ.get("OTR_LTX_AV_SHARP") is not None:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                "OTR_LTX_AV_SHARP is retired -- use OTR_LTX_AV_RECIPE="
                + _RECIPE_MENU, kind="video")
        sel = os.environ.get("OTR_LTX_AV_RECIPE", RECIPE_AUTO).strip().lower()
        base = os.path.basename(
            str(self._unet_name() or "").replace("\\", "/")).lower()
        if sel != RECIPE_AUTO:
            if sel not in _VALID_RECIPES:
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                    "OTR_LTX_AV_RECIPE=%r invalid -- use %s"
                    % (sel, _RECIPE_MENU), kind="video")
            if sel == RECIPE_IA2V and base.startswith("ltx-2.3-22b-distilled"):
                # the ia2v recipe stacks distilled-lora-384@0.5 on the unet; on
                # an already-distilled unet that DOUBLE-DISTILLS -> mush. LOUD.
                raise EngineUnusable(
                    self.name, self.family,
                    EngineUsabilityReason.MALFORMED_CONFIG,
                    "OTR_LTX_AV_RECIPE=ia2v_canonical requires a DEV-family "
                    "unet (got %r) -- the recipe's distilled LoRA on a "
                    "distilled unet would double-distill" % base, kind="video")
            return sel
        return self._detect_recipe(self._unet_name())

    def _detect_recipe(self, unet_name):
        """auto: STRICT family match on the unet basename (lowercased). An
        unknown family RAISES (the operator sets OTR_LTX_AV_RECIPE) -- a name that
        trips both tokens or neither is ambiguous and never silently resolved.
        DEV family -> ia2v_canonical (operator GO 2026-07-02: the canonical
        lip-sync recipe is the dev default for ALL ltx_audio_in beats;
        sharp_lora stays explicitly selectable)."""
        base = os.path.basename(str(unet_name or "").replace("\\", "/")).lower()
        if base.startswith("ltx-2.3-22b-distilled-1.1-"):
            return RECIPE_DISTILLED_NATIVE
        if base.startswith("ltx-2.3-22b-dev-"):
            return RECIPE_IA2V
        raise EngineUnusable(
            self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
            "cannot auto-detect the LTX-AV recipe from unet %r -- set "
            "OTR_LTX_AV_RECIPE=%s" % (base, _RECIPE_MENU),
            kind="video")

    @staticmethod
    def _keep_set(terminal, rcfg):
        """Residency keep-set for run_graph: unet + terminal, plus ONLY the model
        head node that actually exists in this recipe's graph (distilled_native
        has neither lora nor modelsampling -- keeping a missing node is a crash)."""
        keep = {"unet", terminal}
        if rcfg["use_lora"]:
            keep.add("lora")
        elif rcfg["use_modelsampling"]:
            keep.add("modelsampling")
        return keep

    def _weight_paths(self):
        """(label, full_path, floor_bytes) for each required weight artifact. The
        projection ckpt is ALWAYS required (LTXAVTextEncoderLoader reads it); the
        distilled LoRA is required only in SHARP mode."""
        paths = [
            ("transformer GGUF", _resolve("unet", self._unet_name()), _FLOOR_UNET),
            ("Gemma-3 text encoder",
             _resolve("text_encoders", self._encoder_name()), _FLOOR_ENCODER),
            ("projection ckpt",
             _resolve("checkpoints", self._projection_ckpt()), _FLOOR_PROJECTION_CKPT),
            ("video VAE", _resolve("vae", self._video_vae_name()), _FLOOR_VIDEO_VAE),
            ("audio VAE", _resolve("vae", self._audio_vae_name()), _FLOOR_AUDIO_VAE),
        ]
        rcfg = _recipe_config(self._recipe())
        if rcfg["use_lora"]:
            paths.append(("distilled LoRA",
                          _resolve("loras", self._distilled_lora_name(rcfg)),
                          _FLOOR_LORA))
        if rcfg["two_stage"]:
            # required ONLY by ia2v_canonical -- other recipes stay usable on
            # installs lacking the spatial upscaler (Sub-plan-A contract rule).
            paths.append(("latent spatial upscaler",
                          _resolve("latent_upscale_models",
                                   _ia2v_upscale_model_name()),
                          _FLOOR_UPSCALER))
        return paths

    # ---- usability (fail-closed BEFORE any forward; no heavy import) ----
    def assert_usable(self, host_caps, profile, request_template=None):
        """Ordered, fail-closed-before-GPU gate (five PINNED reasons only):
        1 BUG-070 Sage gate; 2 NVML REQUIRED (this lane only -- fail
        closed so the ceiling guard never silently no-ops); 3 node gate (every
        required ComfyUI class resolves); 4 weight floors (realpath + size);
        5 av_dims on request_template.canvas (None tolerated)."""
        # BUG-070 SageAttention contamination (int8-PV aborts LTX silently)
        _MC.assert_sage_not_patched(self.name, self.family)
        # 3 -- NVML REQUIRED for the heaviest lane: the VRAM peak telemetry +
        #      the settle-wait need a live probe; fail closed rather than run an
        #      unbounded heavy forward blind on a host we cannot measure.
        from .._otr_shared import gpu_residency as _GR
        if not _GR.nvml_available():
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.INCOMPATIBLE_PROFILE,
                "%s requires NVML for VRAM telemetry; NVML is unavailable on "
                "this host (the LTX-AV lane fails closed rather than run an "
                "unbounded heavy forward blind)" % self.name, kind="video")
        # 3b -- recipe resolution FIRST (fail-loud BEFORE node/weight gating):
        #       a retired OTR_LTX_AV_SHARP, a bad OTR_LTX_AV_RECIPE, or an
        #       unrecognized unet family RAISES here, at the gate, never mid-render
        #       (CLAUDE.md NO FALLBACKS). _node_candidates + _weight_paths below
        #       consult the same resolver, so the graph cannot drift from the gate.
        self._recipe()
        # 4 -- node gate: every required ComfyUI class must resolve (lazy read)
        from . import wrapper_bridge as _wb
        missing = []
        mapping = _wb.node_class_mappings()
        for logical, candidates in self._node_candidates().items():
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
        # 5 -- weights present + above the sanity floor (realpath -> broken
        #      symlinks fail)
        for label, path, floor in self._weight_paths():
            real = os.path.realpath(path) if path else ""
            if not real or not os.path.exists(real):
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                    "%s %s not found at %r (set the matching OTR_LTX_AV_* env)"
                    % (self.name, label, path), kind="video")
            if os.path.getsize(real) < floor:
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                    "%s %s at %r is below the %d-byte floor (truncated/wrong "
                    "file?)" % (self.name, label, real, floor), kind="video")
        # 6 -- dims on the provided canvas (None tolerated); wrap any ValueError
        if request_template is not None:
            try:
                w, h = self._canvas_dims(request_template)
                if w and h:
                    _AVD.assert_ltx_dims(w, h, _LTX_AV_MIN_FRAMES)
            except EngineUnusable:
                raise
            except Exception as exc:  # noqa: BLE001 - no raw ValueError escapes
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                    "%s canvas dims invalid for LTX: %s" % (self.name, exc),
                    kind="video")
        return self.name

    # ---- graph spec (M0-grounded; classes resolve via wrapper_bridge) ----
    def _node_candidates(self):
        cands = {
            "unet": ("UnetLoaderGGUF",),
            "te": ("LTXAVTextEncoderLoader",),
            "pos": ("CLIPTextEncode",),
            "neg": ("CLIPTextEncode",),
            "cond": ("LTXVConditioning",),
            "videovae": ("VAELoader",),
            "audiovae": ("VAELoader",),
            "loadaudio": ("LoadAudio",),
            "audioenc": ("LTXVAudioVAEEncode",),
            "concat": ("LTXVConcatAVLatent",),
            "noise": ("RandomNoise",),
            "ksel": ("KSamplerSelect",),
            "guider": ("CFGGuider",),
            "sampler": ("SamplerCustomAdvanced",),
            "separate": ("LTXVSeparateAVLatent",),
            "decode": ("VAEDecodeTiled",),
            # i2v (scene still / face) AND t2v (empty latent) classes are BOTH
            # resolved; _build_graph wires whichever the request needs -- i2v when
            # an init image is present (always for the talk face; the scene still
            # for music/announcer beats), else the empty-latent t2v path.
            "loadimage": ("LoadImage",),
            "i2v": ("LTXVImgToVideo",),
            "emptylatent": ("EmptyLTXVLatentVideo",),
        }
        rcfg = _recipe_config(self._recipe())
        if rcfg["use_lora"]:
            # sharp_lora / ia2v_canonical: the LoRA-wrapped unet feeds the
            # guider; fixed sigmas come from the in-adapter injector.
            cands["lora"] = ("LoraLoaderModelOnly",)
        if rcfg["use_modelsampling"]:
            # m0_base: ModelSamplingLTXV + LTXVScheduler (no LoRA, no manual sigmas).
            cands["modelsampling"] = ("ModelSamplingLTXV",)
            cands["sched"] = ("LTXVScheduler",)
        if rcfg["two_stage"]:
            # ia2v_canonical (2026-07-02 transplant): the canonical two-stage
            # graph's extra classes -- required ONLY when the recipe is active.
            cands["inplace_base"] = ("LTXVImgToVideoInplace",)
            cands["inplace_refine"] = ("LTXVImgToVideoInplace",)
            cands["solidmask"] = ("SolidMask",)
            cands["noisemask"] = ("SetLatentNoiseMask",)
            cands["upscale_loader"] = ("LatentUpscaleModelLoader",)
            cands["upscaler"] = ("LTXVLatentUpsampler",)
            cands["cropguides"] = ("LTXVCropGuides",)
            cands["imagescale"] = ("ImageScale",)
            cands["resizelonger"] = ("ResizeImagesByLongerEdge",)
            cands["preprocess"] = ("LTXVPreprocess",)
            cands["ksel_refine"] = ("KSamplerSelect",)
            cands["noise_refine"] = ("RandomNoise",)
            cands["sampler_refine"] = ("SamplerCustomAdvanced",)
            cands["separate_base"] = ("LTXVSeparateAVLatent",)
            cands["concat_refine"] = ("LTXVConcatAVLatent",)
            cands["guider_refine"] = ("CFGGuider",)
        # distilled_native: adds NEITHER -- the distilled unet feeds the guider
        # directly + the in-adapter fixed sigmas (distillation baked into the GGUF).
        return cands

    def _build_graph(self, plan, length, width, height, audio_name, image_name):
        """The declarative LTX-AV A2V graph (wrapper_bridge.run_graph format).

        Common: GGUF unet -> ModelSamplingLTXV; LTXAVTextEncoderLoader (Gemma-3 on
        CPU) -> pos/neg CLIPTextEncode -> LTXVConditioning; VAELoader x2
        (video+audio); LoadAudio -> LTXVAudioVAEEncode. Talk branch: LoadImage ->
        LTXVImgToVideo (I2V conditioning) -> video latent. Music branch:
        EmptyLTXVLatentVideo. Both: LTXVConcatAVLatent(video,audio) ->
        SamplerCustomAdvanced(LTXVScheduler/euler/CFGGuider) -> LTXVSeparateAVLatent
        -> VAEDecodeTiled(video only; audio latent DROPPED, V-1)."""
        from . import wrapper_bridge as _wb
        W = _wb.Wire
        rcfg = _recipe_config(self._recipe())
        if rcfg["two_stage"]:
            return self._build_graph_ia2v(plan, length, width, height,
                                          audio_name, image_name)
        positive = plan.get("text_prompt") or "a vintage radio broadcast scene"
        negative = os.environ.get("OTR_LTX_AV_NEGATIVE", _LTX_DEFAULT_NEGATIVE)
        seed = int(plan.get("seed", 0) or 0)
        cfg = rcfg["cfg"]
        sampler_name = rcfg["sampler"]
        i2v_strength = rcfg["i2v_strength"]
        use_i2v = bool(image_name)
        g = {
            "unet": {"class": "unet", "inputs": {"unet_name": self._unet_name()}},
            "te": {"class": "te", "inputs": {
                "text_encoder": self._encoder_name(),
                "ckpt_name": self._projection_ckpt(),
                "device": os.environ.get("OTR_LTX_AV_ENCODER_DEVICE", "cpu")}},
            "pos": {"class": "pos", "inputs": {"text": positive, "clip": W("te", 0)}},
            "neg": {"class": "neg", "inputs": {"text": negative, "clip": W("te", 0)}},
            "cond": {"class": "cond", "inputs": {
                "positive": W("pos", 0), "negative": W("neg", 0),
                "frame_rate": float(self.target_fps)}},
            "videovae_dec": {"class": "videovae",
                             "inputs": {"vae_name": self._video_vae_name()}},
            "audiovae": {"class": "audiovae",
                         "inputs": {"vae_name": self._audio_vae_name()}},
            "loadaudio": {"class": "loadaudio", "inputs": {"audio": audio_name}},
            "audioenc": {"class": "audioenc", "inputs": {
                "audio": W("loadaudio", 0), "audio_vae": W("audiovae", 0)}},
            "noise": {"class": "noise", "inputs": {"noise_seed": seed}},
            "ksel": {"class": "ksel", "inputs": {"sampler_name": sampler_name}},
        }
        # Model head -> guider, per recipe:
        #   sharp_lora       : unet -> LoRA -> guider (the LoRA carries the shift;
        #                      NO ModelSamplingLTXV or it would double-shift -> blur).
        #   distilled_native : unet -> guider DIRECTLY (distillation baked into the
        #                      GGUF -- no LoRA, no ModelSamplingLTXV).
        #   m0_base          : unet -> ModelSamplingLTXV -> guider (no LoRA).
        if rcfg["use_lora"]:
            g["lora"] = {"class": "lora", "inputs": {
                "model": W("unet", 0),
                "lora_name": self._distilled_lora_name(),
                "strength_model": _LTX_AV_DISTILLED_LORA_STRENGTH}}
            model_wire = W("lora", 0)
        elif rcfg["use_modelsampling"]:
            g["modelsampling"] = {"class": "modelsampling", "inputs": {
                "model": W("unet", 0), "max_shift": 2.05, "base_shift": 0.95}}
            model_wire = W("modelsampling", 0)
        else:
            model_wire = W("unet", 0)
        # i2v when an init image is present (the talk face ALWAYS; the scene still
        # for music/announcer beats); else the empty-latent t2v path.
        if use_i2v:
            g["loadimage"] = {"class": "loadimage", "inputs": {"image": image_name}}
            # VideoVAE split (roundtable 2026-06-26, VRAM headroom): a SEPARATE
            # encode-side VAE node so run_graph's free_after_use drops its ~1.38 GB
            # right after i2v -- BEFORE the sampler activation peak -- instead of
            # pinning it through the whole denoise loop. The single shared
            # "videovae" used to feed BOTH i2v AND decode, so the last-consumer
            # free never fired until decode -> ~1.4 GB stolen from the sampler ->
            # sysmem spill (the 6.84-vs-223 s/it knife-edge). decode reloads the
            # VAE (cache-warm) after the peak. Same class, distinct graph node.
            g["videovae_enc"] = {"class": "videovae",
                                 "inputs": {"vae_name": self._video_vae_name()}}
            g["i2v"] = {"class": "i2v", "inputs": {
                "positive": W("cond", 0), "negative": W("cond", 1),
                "vae": W("videovae_enc", 0), "image": W("loadimage", 0),
                "width": int(width), "height": int(height), "length": int(length),
                "batch_size": 1, "strength": i2v_strength}}
            video_latent = W("i2v", 2)
            guider_pos, guider_neg = W("i2v", 0), W("i2v", 1)
        else:
            g["emptylatent"] = {"class": "emptylatent", "inputs": {
                "width": int(width), "height": int(height),
                "length": int(length), "batch_size": 1}}
            video_latent = W("emptylatent", 0)
            guider_pos, guider_neg = W("cond", 0), W("cond", 1)
        g["concat"] = {"class": "concat", "inputs": {
            "video_latent": video_latent, "audio_latent": W("audioenc", 0)}}
        g["guider"] = {"class": "guider", "inputs": {
            "model": model_wire, "positive": guider_pos,
            "negative": guider_neg, "cfg": cfg}}
        # manual_sigmas (sharp_lora + distilled_native): the fixed
        # LTX_DISTILLED_SIGMAS via the in-adapter injector ("sigmas" is added to the
        # resolved-class map by render_clip). m0_base: LTXVScheduler.
        if rcfg["manual_sigmas"]:
            g["sigmas"] = {"class": "sigmas",
                           "inputs": {"values": list(LTX_DISTILLED_SIGMAS)}}
            sigmas_wire = W("sigmas", 0)
        else:
            g["sched"] = {"class": "sched", "inputs": {
                "steps": _LTX_AV_STEPS, "max_shift": 2.05, "base_shift": 0.95,
                "stretch": True, "terminal": 0.1, "latent": W("concat", 0)}}
            sigmas_wire = W("sched", 0)
        g["sampler"] = {"class": "sampler", "inputs": {
            "noise": W("noise", 0), "guider": W("guider", 0),
            "sampler": W("ksel", 0), "sigmas": sigmas_wire,
            "latent_image": W("concat", 0)}}
        g["separate"] = {"class": "separate",
                         "inputs": {"av_latent": W("sampler", 0)}}
        # Decode temporal tiling: env-overridable per-render, FAIL-LOUD (no clamp).
        # Default 4096/8 = whole-clip (no inter-tile seam); env -> tiled fallback.
        decode_t_size, decode_t_overlap = _decode_temporal_knobs()
        g["decode"] = {"class": "decode", "inputs": {
            "samples": W("separate", 0), "vae": W("videovae_dec", 0),
            "tile_size": 512, "overlap": 64,
            "temporal_size": decode_t_size,
            "temporal_overlap": decode_t_overlap}}
        return g

    def _build_graph_ia2v(self, plan, length, width, height, audio_name,
                          image_name):
        """The canonical comfy.org LTX-2.3 IA2V TWO-STAGE lip-sync graph
        (2026-07-02 transplant; spec = docs/2026-07-02-canonical-ia2v/
        ia2v_flat_api_prompt.json, live-proven on OUR radio still).

        STAGE A -- MOTION (half canvas): guide still -> ImageScale(1.5x) ->
        ResizeImagesByLongerEdge(1536) -> LTXVPreprocess(18); planted IN-PLACE
        (LTXVImgToVideoInplace, strength 0.7 -- a SOFT anchor so the audio can
        actuate the mouth) into an EmptyLTXVLatentVideo at width/2 x height/2;
        the audio latent rides FROZEN under SetLatentNoiseMask(SolidMask(0));
        denoised by euler_ancestral_cfg_pp (ancestral = fresh noise each step,
        keeps motion alive) over the front-loaded 8-step ladder (5 of 8 steps
        in the top ~2.5%% of noise -- the motion-decision region).

        STAGE B -- DETAIL (full canvas): separate -> LTXVLatentUpsampler x2 on
        the VIDEO latent -> re-anchor the SAME preprocessed still at strength
        1.0 -> RE-CONCAT stage A's AUDIO latent -> 3-step euler_cfg_pp refine
        (0.85 -> 0; sharpness only, never re-decides motion) with conditioning
        routed through LTXVCropGuides(base latent) -> separate -> tiled decode
        (video only -- the audio latent is DROPPED; audio stays mux-LAST, V-1).

        i2v is REQUIRED (still-required engine; NO FALLBACKS -- render_clip
        raises before this builds when init_image is absent). Base noise =
        the beat seed; refine noise = seed+1 (two independent, deterministic
        streams -- the canonical used randomize/fixed, we keep C7-style
        determinism per beat)."""
        from . import wrapper_bridge as _wb
        W = _wb.Wire
        rcfg = _recipe_config(self._recipe())
        if not image_name:
            # defense at the build seam too (render_clip already gates): the
            # canonical graph plants the still at BOTH stages -- no image means
            # no graph, never a silent t2v downgrade. NO FALLBACKS.
            raise _wb.GraphExecutionError(
                "%s (ia2v_canonical) requires an init image for EVERY beat"
                % self.name)
        positive = plan.get("text_prompt") or "a vintage radio broadcast scene"
        negative = os.environ.get("OTR_LTX_AV_NEGATIVE", _LTX_DEFAULT_NEGATIVE)
        seed = int(plan.get("seed", 0) or 0)
        cfg = rcfg["cfg"]
        base_w, base_h = int(width) // 2, int(height) // 2  # exact canonical halving
        g = {
            "unet": {"class": "unet", "inputs": {"unet_name": self._unet_name()}},
            "lora": {"class": "lora", "inputs": {
                "model": W("unet", 0),
                "lora_name": self._distilled_lora_name(rcfg),
                "strength_model": rcfg["lora_strength"]}},
            "te": {"class": "te", "inputs": {
                "text_encoder": self._encoder_name(),
                "ckpt_name": self._projection_ckpt(),
                "device": os.environ.get("OTR_LTX_AV_ENCODER_DEVICE", "cpu")}},
            "pos": {"class": "pos", "inputs": {"text": positive, "clip": W("te", 0)}},
            "neg": {"class": "neg", "inputs": {"text": negative, "clip": W("te", 0)}},
            "cond": {"class": "cond", "inputs": {
                "positive": W("pos", 0), "negative": W("neg", 0),
                "frame_rate": float(self.target_fps)}},
            "videovae_dec": {"class": "videovae",
                             "inputs": {"vae_name": self._video_vae_name()}},
            # encode-side VAE stays a SEPARATE node (BUG-414 split): its last
            # consumer (inplace_refine) runs BEFORE the refine sampler, so
            # free_after_use reclaims it ahead of both activation peaks.
            "videovae_enc": {"class": "videovae",
                             "inputs": {"vae_name": self._video_vae_name()}},
            "audiovae": {"class": "audiovae",
                         "inputs": {"vae_name": self._audio_vae_name()}},
            # ---- guide-image conditioning chain (canonical texture prep) ----
            "loadimage": {"class": "loadimage", "inputs": {"image": image_name}},
            "imagescale": {"class": "imagescale", "inputs": {
                "image": W("loadimage", 0), "upscale_method": "lanczos",
                "width": _IA2V_GUIDE_SCALE_W,
                "height": _IA2V_GUIDE_SCALE_H,
                "crop": "center"}},
            "resizelonger": {"class": "resizelonger", "inputs": {
                "images": W("imagescale", 0),
                "longer_edge": _IA2V_GUIDE_LONGER_EDGE}},
            "preprocess": {"class": "preprocess", "inputs": {
                "image": W("resizelonger", 0),
                "img_compression": _IA2V_GUIDE_COMPRESSION}},
            # ---- audio latent, FROZEN under a zero mask ----
            "loadaudio": {"class": "loadaudio", "inputs": {"audio": audio_name}},
            "audioenc": {"class": "audioenc", "inputs": {
                "audio": W("loadaudio", 0), "audio_vae": W("audiovae", 0)}},
            "solidmask": {"class": "solidmask", "inputs": {
                "value": 0.0, "width": int(width), "height": int(height)}},
            "noisemask": {"class": "noisemask", "inputs": {
                "samples": W("audioenc", 0), "mask": W("solidmask", 0)}},
            # ---- STAGE A: motion at half canvas ----
            "emptylatent": {"class": "emptylatent", "inputs": {
                "width": base_w, "height": base_h,
                "length": int(length), "batch_size": 1}},
            "inplace_base": {"class": "inplace_base", "inputs": {
                "vae": W("videovae_enc", 0), "image": W("preprocess", 0),
                "latent": W("emptylatent", 0),
                "strength": rcfg["i2v_strength"], "bypass": False}},
            "concat": {"class": "concat", "inputs": {
                "video_latent": W("inplace_base", 0),
                "audio_latent": W("noisemask", 0)}},
            "guider": {"class": "guider", "inputs": {
                "model": W("lora", 0), "positive": W("cond", 0),
                "negative": W("cond", 1), "cfg": cfg}},
            "sigmas": {"class": "sigmas",
                       "inputs": {"values": list(LTX_DISTILLED_SIGMAS)}},
            "ksel": {"class": "ksel",
                     "inputs": {"sampler_name": rcfg["sampler"]}},
            "noise": {"class": "noise", "inputs": {"noise_seed": seed}},
            "sampler": {"class": "sampler", "inputs": {
                "noise": W("noise", 0), "guider": W("guider", 0),
                "sampler": W("ksel", 0), "sigmas": W("sigmas", 0),
                "latent_image": W("concat", 0)}},
            "separate_base": {"class": "separate_base",
                              "inputs": {"av_latent": W("sampler", 0)}},
            # ---- STAGE B: latent-upsample + re-anchor + audio re-concat ----
            "upscale_loader": {"class": "upscale_loader", "inputs": {
                "model_name": _ia2v_upscale_model_name()}},
            "upscaler": {"class": "upscaler", "inputs": {
                "samples": W("separate_base", 0),
                "upscale_model": W("upscale_loader", 0),
                "vae": W("videovae_enc", 0)}},
            "inplace_refine": {"class": "inplace_refine", "inputs": {
                "vae": W("videovae_enc", 0), "image": W("preprocess", 0),
                "latent": W("upscaler", 0),
                "strength": rcfg["refine_strength"], "bypass": False}},
            "concat_refine": {"class": "concat_refine", "inputs": {
                "video_latent": W("inplace_refine", 0),
                "audio_latent": W("separate_base", 1)}},
            "cropguides": {"class": "cropguides", "inputs": {
                "positive": W("cond", 0), "negative": W("cond", 1),
                "latent": W("separate_base", 0)}},
            "guider_refine": {"class": "guider_refine", "inputs": {
                "model": W("lora", 0), "positive": W("cropguides", 0),
                "negative": W("cropguides", 1), "cfg": cfg}},
            "sigmas_refine": {"class": "sigmas_refine", "inputs": {
                "values": list(rcfg["refine_sigmas"])}},
            "ksel_refine": {"class": "ksel_refine", "inputs": {
                "sampler_name": rcfg["refine_sampler"]}},
            "noise_refine": {"class": "noise_refine",
                             "inputs": {"noise_seed": seed + 1}},
            "sampler_refine": {"class": "sampler_refine", "inputs": {
                "noise": W("noise_refine", 0), "guider": W("guider_refine", 0),
                "sampler": W("ksel_refine", 0), "sigmas": W("sigmas_refine", 0),
                "latent_image": W("concat_refine", 0)}},
            "separate": {"class": "separate",
                         "inputs": {"av_latent": W("sampler_refine", 0)}},
        }
        decode_t_size, decode_t_overlap = _decode_temporal_knobs()
        g["decode"] = {"class": "decode", "inputs": {
            "samples": W("separate", 0), "vae": W("videovae_dec", 0),
            "tile_size": 512, "overlap": 64,
            "temporal_size": decode_t_size,
            "temporal_overlap": decode_t_overlap}}
        return g

    def _weight_receipt(self, path):
        """A BOUNDED, stable receipt for one weight file: (basename, size,
        mtime_ns). Mirrors ``eng_ltx_8gb._file_receipt`` deliberately -- NOT a
        content hash, because the identity is re-read before every segment and
        hashing a multi-gigabyte GGUF per segment would cost more than the
        render. Size + mtime catches a swapped or rebuilt weight.

        A path that does not resolve or cannot be stat'ed returns a NAMED
        absence rather than raising: `assert_usable` already owns "this weight
        is missing" and says it far better, and raising here would turn a
        missing-weight run into a confusing session error instead of the
        usability refusal it is.
        """
        if not path:
            return ("<unresolved>", -1, -1)
        try:
            st = os.stat(path)
        except OSError:
            return (os.path.basename(path), -1, -1)
        return (os.path.basename(path), int(st.st_size), int(st.st_mtime_ns))

    def session_identity(self):
        """What this adapter's HANDLES are -- engine, recipe, weights.

        ``BeatSession`` REFUSES a multi-segment beat from an engine that cannot
        answer this (``SessionIdentityUnavailable``, no fallback), because
        nothing could then prove segment N renders with the model segment 1
        loaded. Declared on the BASE so every LTX-AV lane inherits it at once;
        the sibling `ltx_video` lane reached a live render gate without one and
        refused there, 730 seconds into a leg, which is the most expensive
        possible place to learn it.

        Deliberately PRE-LOAD STABLE, on the same contract as
        ``eng_ltx_8gb.session_identity``: read once before the weights land,
        again after ``prepare()``, and before every segment, so it may only
        describe things that do not change across the load. The recipe is
        carried because it selects the whole sampling ladder AND which weights
        are required; every per-segment value -- prompt, seed, frame count,
        canvas, the conditioning audio -- is excluded.

        Not cached, by design: the whole job is to notice a weight that MOVED,
        so the receipts are re-stat'ed on every ask (a stat per weight, not a
        hash).
        """
        parts = [self.name, self._recipe()]
        for label, path, _floor in self._weight_paths():
            parts.append("%s=%r" % (label, self._weight_receipt(path)))
        return tuple(str(part) for part in parts)

    # ---- residency ----
    def load(self):
        """Resolve the installed ComfyUI node classes (fail-closed NAMED if
        absent). The heavy weight load happens when the loader nodes execute in
        render_clip (ComfyUI's own model management); the AS-3 lease brackets the
        real residency."""
        from . import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
        self._loaded = True

    def render_clip(self, request, prepared):
        """Drive ONE audio-conditioned clip via the in-process LTX-AV graph and
        encode the decoded IMAGE batch to a SILENT bt709 clip (V-1: the audio
        latent is dropped at LTXVSeparateAVLatent; only the mux adds audio)."""
        from . import wrapper_bridge as _wb
        from ._tmp import otr_engine_tmp_mp4
        recipe = self._recipe()
        rcfg = _recipe_config(recipe)
        # one LOUD line per beat so any render log self-documents the recipe that
        # actually ran (recipe FOLLOWS THE MODEL) -- no need to infer it from the
        # absence of a LoRA load.
        _LOG.info("[OTR video] ltx_audio_in recipe=%s unet=%s",
                  recipe, os.path.basename(str(self._unet_name())))
        plan = self._build_render_request(request)
        if not plan["audio_path"]:
            raise _wb.GraphExecutionError(
                "%s requires audio_ref (got %r)" % (self.name, plan["audio_path"]))
        if self._is_talk and not plan["init_image"]:
            raise _wb.GraphExecutionError(
                "%s (talk) requires init_image (got %r)"
                % (self.name, plan["init_image"]))
        if rcfg["two_stage"] and not plan["init_image"]:
            # ia2v_canonical is i2v-ONLY: the canonical graph plants the still
            # in-place at both stages. NO FALLBACKS -- a beat without a still
            # is a hard error, never a silent t2v downgrade.
            raise _wb.GraphExecutionError(
                "%s (ia2v_canonical) requires init_image for EVERY beat "
                "(got %r)" % (self.name, plan["init_image"]))
        classes = dict(getattr(self, "_classes", None)
                       or _wb.resolve_graph_classes(self._node_candidates()))
        if rcfg["manual_sigmas"]:
            # the SIGMAS source is in-adapter (not a registered node class); inject
            # AFTER resolve so the resolver never sees it (mirrors eng_ltx_video).
            # Fires for BOTH sharp_lora and distilled_native (both run the fixed
            # distilled sigmas) -- NOT just the LoRA path.
            classes.setdefault("sigmas", _SigmasFromValues)
        if rcfg["two_stage"]:
            # ia2v_canonical: the refine ladder is a SECOND in-adapter sigmas
            # source (0.85 -> 0, 3 steps).
            classes.setdefault("sigmas_refine", _SigmasFromValues)
        audio_name = _wb.stage_into_comfy_input(plan["audio_path"])
        # i2v on ANY init image (the talk face; the scene still for music/announcer)
        image_name = (_wb.stage_into_comfy_input(plan["init_image"])
                      if plan["init_image"] else "")
        width, height = self._render_dims(request)
        length = _AVD.next_8n1(plan["target_frame_count"] or self.target_fps)
        if length > _LTX_AV_MAX_FRAMES:
            length = _AVD.next_8n1(_LTX_AV_MAX_FRAMES)
            if (length - 1) % _AVD._LTX_TEMPORAL_BASE != 0:
                length = _LTX_AV_MAX_FRAMES
        _AVD.assert_ltx_dims(width, height, length)
        # S-B observability: ONE comprehensive per-beat line so any render log
        # self-documents EXACTLY what ran (recipe / quant / LoRA / canvas / frames
        # / audio source) -- the fit regression was invisible because the log only
        # carried recipe+unet. The measured peak is logged at assert time below.
        _audio_src = os.path.basename(str(plan["audio_path"] or ""))
        _LOG.info(
            "[OTR video] ltx_audio_in PLAN recipe=%s unet=%s quant=%s lora=%s "
            "canvas=%dx%d frames=%d audio=%s",
            recipe, os.path.basename(str(self._unet_name())), self._quant_label(),
            bool(rcfg["use_lora"]), width, height, length, _audio_src or "-")
        graph = self._build_graph(plan, length, width, height, audio_name, image_name)
        # free_after_use (the eng_ltx_video pattern): evict the Gemma encoder +
        # intermediates before the unet+VAE-decode peak so the GGUF unet (+ the LoRA
        # patch in sharp mode) never co-resides with the encoder (the 14.5 GB
        # ceiling). KEEP the unet + the model head (lora in sharp / modelsampling
        # in M0) + the terminal so the patcher is never dangled. distilled_native
        # has NO head node, so _keep_set keeps only {unet, terminal} -- keeping a
        # node that does not exist in the graph would crash run_graph.
        keep = self._keep_set(self._TERMINAL, rcfg)
        # PASS-PM: sample the REAL render-window VRAM peak (first heavy load
        # through VAEDecode), not just the post-cleanup instantaneous read -- the
        # whole-clip decode + the additively-resident encoder can peak mid-render.
        # Leak/cleanup-safe: results/images init to None so the finally never
        # dangles a patcher or encodes a half-built result.
        results = images = None
        probe = _MC.VramPeakProbe(interval_s=0.1).start()
        try:
            with _ltx_av_vram_reserve():
                results = _wb.run_graph(graph, classes, free_after_use=True,
                                        keep=keep)
            images = results[self._TERMINAL][0]
        finally:
            peak = probe.stop()
            # retain the patchers ONLY if the graph actually produced a result;
            # ALWAYS reclaim (LOUD; never unload_all) so the resident stack drops
            # under the ceiling before the assert + the next beat.
            if results is not None:
                self._retain_model_patchers(results, prepared)
            _wb.reclaim_idle_models(reason="%s post-decode" % self.name)
        # Telemetry: log the MEASURED render-window peak (VramPeakProbe). No
        # ceiling enforcement -- the operator's tier JSON owns the OOM budget.
        if peak:
            _LOG.info("[%s] render-window VRAM peak: %d MB", self.name, int(peak))
        if images is None:                       # success path always sets images
            raise _wb.GraphExecutionError(
                "%s: run_graph produced no terminal image" % self.name)
        frames = _wb.images_to_uint8(images)
        out_path = otr_engine_tmp_mp4("otr_ltx_av_")
        path, n = _wb.encode_frames_to_silent_mp4(frames, out_path, self.target_fps)
        # M7 (added 2026-07-27): see the twin comment in eng_humo. This adapter
        # and eng_humo were the two of six that returned a fully self-declared
        # dict, so nothing ffprobed what they actually wrote before the ledger
        # recorded it.
        validate_silent_clip_contract(ffprobe_clip_fields(path), self.target_fps)
        # S-B/E5: thread the recipe receipt into the raw so the episode report +
        # the ledger recipe-stamp self-document the fit (recipe / quant / LoRA /
        # canvas / audio source) alongside the measured peak.
        return {"out_path": path, "frame_count": n, "vram_peak_mb": peak,
                "recipe": recipe, "unet": os.path.basename(str(self._unet_name())),
                "quant": self._quant_label(), "use_lora": bool(rcfg["use_lora"]),
                "canvas": "%dx%d" % (width, height), "audio_source": _audio_src}

    def canonicalize(self, raw, request, profile):
        return self._clip_from_raw(raw, request)

    def _retain_model_patchers(self, results, prepared):
        """Best-effort V-4: keep the MODEL ModelPatchers the graph produced so
        teardown can detach(unpatch_all=True) them."""
        bucket = prepared.setdefault("patchers", self._patchers) \
            if isinstance(prepared, dict) else self._patchers
        seen = {id(p) for p in bucket}
        for nid in ("unet", "modelsampling", "lora"):
            out = results.get(nid)
            if not out:
                continue
            obj = out[0]
            if id(obj) not in seen and callable(getattr(obj, "detach", None)):
                bucket.append(obj)
                seen.add(id(obj))

    # ---- pure helpers (CPU-testable; no wrapper, no heavy import) ----
    @staticmethod
    def _ref_path(ref):
        """Pull a filesystem path out of an audio_ref / init_image that may be a
        bare string OR a mapping carrying a ``path`` key (the AudioRef shape)."""
        if not ref:
            return ""
        if isinstance(ref, str):
            return ref
        if isinstance(ref, dict):
            return ref.get("path") or ""
        return getattr(ref, "path", "") or ""

    def _init_image_ref(self, request):
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        assets = get("asset_refs") or {}
        if isinstance(assets, dict):
            return assets.get("init_image") or ""
        return ""

    def _canvas_dims(self, request):
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        canvas = get("canvas") or {}
        c_get = canvas.get if isinstance(canvas, dict) else (
            lambda k, d=None: getattr(canvas, k, d))
        return int(c_get("w", 0) or 0), int(c_get("h", 0) or 0)

    def _render_dims(self, request):
        """The render canvas: request.canvas.w/h when present, else the native
        1472x832. Snapped to LTX's 32-multiple via assert_ltx_dims downstream."""
        w, h = self._canvas_dims(request)
        if w and h:
            return w, h
        return _LTX_AV_NATIVE_W, _LTX_AV_NATIVE_H

    def _build_render_request(self, request):
        """Pure: the normalized inference request the LTX-AV graph consumes."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        timing = get("timing") or {}
        t_get = timing.get if isinstance(timing, dict) else (
            lambda k, d=None: getattr(timing, k, d))
        seeds = get("seed_bundle") or {}
        s_get = seeds.get if isinstance(seeds, dict) else (
            lambda k, d=None: getattr(seeds, k, d))
        return {
            "init_image": self._init_image_ref(request),
            "audio_path": self._ref_path(get("audio_ref")),
            "text_prompt": get("text_prompt") or "",
            "fps": int(self.target_fps),
            "target_frame_count": int(t_get("target_frame_count", 0) or 0),
            "seed": int(s_get("request_seed", 0) or 0),
        }

    def _clip_from_raw(self, raw, request):
        """Pure: shape a wrapper result into the silent CanonicalClip dict
        (bt709 / yuv420p; has_audio False -- only OTR_MasterAudioMux adds audio)."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        raw = raw or {}
        return {
            "clip_id": get("shot_id") or get("request_id") or "ltx_av_clip",
            "type": "video", "path": raw.get("out_path", ""),
            "container": "mp4", "codec": "h264", "pixel_format": "yuv420p",
            "fps": int(self.target_fps),
            "frame_count": int(raw.get("frame_count", 0) or 0),
            "has_audio": False,
            "color_primaries": "bt709", "transfer": "bt709", "matrix": "bt709",
            "engine_id": self.name, "family": self.family,
            # PASS-PM: the REAL render-window NVML peak (None off the GPU box),
            # threaded render_clip -> here -> render_shot -> the episode report.
            "vram_peak_mb": raw.get("vram_peak_mb"),
            # S-B/E5 recipe receipt (None for a raw that did not carry it).
            "recipe": raw.get("recipe"), "unet": raw.get("unet"),
            "quant": raw.get("quant"), "use_lora": raw.get("use_lora"),
            "render_canvas": raw.get("canvas"),
            "audio_source": raw.get("audio_source"),
        }


#: S1 (2026-07-25) per-model still plan for ltx_audio_in (spec section 3,
#: Shape A -- scene spine with an audio-conditioned LTX twist).
#: FILE-LOCAL, fully declared. Base scene spine (scene_open + scene_beat +
#: scene_character all WIDE + required=always) plus two extra rows the spec
#: calls out for this engine specifically: (1) a WIDE radio face per
#: bookend role, required=when_engine_talking (announcer/music bookends
#: get a scene-radio face when the engine's ``wants_talking_prompt``
#: hook fires); (2) a cast portrait on character beats,
#: required=when_engine_talking. Nothing reads the plan at S1.
_LTX_AUDIO_IN_STILL_PLAN = (
    StillPlanRow(kind="scene_open", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     "full-frame macro, centered subject"),
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_beat", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
                 framing_geometry=(
                     ("cinematic three-quarter framing, people shown with "
                      "full heads and clear headroom inside frame, faces "
                      "unobstructed, balanced composition")),
                 style_tail_policy="full"),
    StillPlanRow(kind="scene_character", cardinality="per_beat",
                 target_class="scene", aspect="wide", required="always",
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
    # Per spec section 3: '+ WIDE radio face per bookend role,
    # required=when_engine_talking'. A WIDE face still per bookend role
    # (announcer_visual / music_visual) fed into I2V when the engine talks.
    # S1b CORRECTION (kibitz r4, codex gpt-5.6-sol high; grounded against the
    # producer): this row is a PORTRAIT-class object, not a scene one. The
    # producer mints it at otr_meta_brief_image_prompt.py:1782-1790 with
    # kind="portrait" / source="ltx_radio_face", from
    # build_radio_host_prompt(meta, "wide", radio_host_style=
    # "ltx_radio_mouth") -- and that branch composes the WIDE portrait anchor
    # (_style_anchor_for_aspect("wide"), no talking flag), so the layer-2 text
    # is WIDE_PORTRAIT_GEOMETRY rather than the talking close-up bust. The
    # radio_host_style is a LITERAL at that call site, so there is no runtime
    # branch being frozen here.
    StillPlanRow(kind="portrait", cardinality="per_bookend_role",
                 target_class="portrait", aspect="wide",
                 required="when_engine_talking",
                 framing_geometry=(
                     ("in-character cinematic medium shot, head and "
                      "shoulders, face clearly visible, subject centred with "
                      "natural headroom above the head (never crop the top of "
                      "the head)")),
                 style_tail_policy="full"),
    # Per spec section 3: '+ cast portrait on character beats,
    # required=when_engine_talking'. The cast portrait feeds the I2V head on
    # character beats when the engine talks.
    StillPlanRow(kind="portrait", cardinality="per_recurring_subject",
                 target_class="portrait", aspect="inherit_engine",
                 required="when_engine_talking",
                 framing_geometry=(
                     ("in-character face-forward frontal close-up bust, "
                      "looking directly at the camera, the whole face and "
                      "mouth clearly visible and unobstructed, the face "
                      "brightly lit and filling much of the frame, subject "
                      "centred with natural headroom above the head (never "
                      "crop the top of the head)")),
                 style_tail_policy="full"),
)


@register
class LtxAudioInEngine(_LtxAvBase):
    """LTX AUDIO-IN -- the ONE audio-in LTX lane (operator 2026-06-26).

    ONE engine for every audio-reactive role: it does LTX I2V on WHATEVER still
    the pipeline mints -- a radio-bookend scene still, a character scene still, a
    face -- conditioned on the shot AUDIO, music OR voice. It does NOT care which;
    the render core does ``i2v on ANY init image``. The simplest model:
    ``ltx_video`` is the regular (no-audio) LTX lane, ``ltx_audio_in`` is the
    audio-in lane.

    The old talk/music split (``ltx_av_talk`` audio_driven_face / ``ltx_av_music``
    audio_conditioned_video) was REMOVED 2026-06-26: that split was never about the
    engine -- it encoded two ROLE routings (portrait+clean-audio+char-prompt for a
    talking head vs scene-still+ambient-audio+scene-prompt for a bookend). That
    routing now lives on the BEAT ROLE in ``render_driver`` (``_is_character_face_
    beat``), NOT on two engines. Do not reintroduce the split.

    Mechanics: ``_is_talk=True`` selects the I2V branch (condition on the still +
    the audio slice -- there is no separate LTX 'lip-sync' parameter). The render
    driver hands it the beat's WIDE scene still (scene_open radio bookend for the
    announcer/music BOOKENDS, scene_character for character beats), so init_image
    is never missing. ``default_roles`` makes it the per-role DEFAULT for the music
    + announcer bookends (the slot the deleted ltx_av_music held). Required:
    text_prompt + audio_ref + init_image. NO fallbacks -- fail LOUD. Same LTX-AV
    weights / VRAM ceiling as the lane."""

    name = "ltx_audio_in"
    family = "audio_conditioned_video"   # agnostic -- NOT audio_driven_face
    #: S1 per-model still plan (see ``_LTX_AUDIO_IN_STILL_PLAN`` above --
    #: scene spine + wide-radio-face-per-bookend + cast-portrait-per-
    #: recurring-subject, both extras required when the engine talks).
    still_plan = _LTX_AUDIO_IN_STILL_PLAN
    roles = ("announcer_visual", "music_visual", "character_video")
    # the per-role DEFAULT for the music + announcer bookends (inheriting the slot
    # the deleted ltx_av_music engine held). default_roles must be subset of roles.
    default_roles = ("music_visual", "announcer_visual")
    required_inputs = ("text_prompt", "audio_ref", "init_image")
    #: THE FRAME LADDER (chunk 7a, 2026-07-26). 8n+1: 9 .. 497 (8*61+1).
    #: The LITERAL 497, not ``_LTX_AV_MAX_FRAMES`` -- that constant is
    #: ``int(os.environ.get("OTR_LTX_AV_MAX_FRAMES", "497"))``, resolved at
    #: IMPORT, so binding the contract to it would let the environment rewrite
    #: a declaration the image phase already planned against. Env disagreement
    #: is a REFUSAL in 7b, never a silent re-plan.
    #: CONTINUITY: soft_reference, not strict. The i2v strength is
    #: recipe-dependent (0.7 ia2v motion pass / 0.75 sharp / 1.0 base), so
    #: frame 0 is a hard pin in some recipes and a soft anchor in others. A
    #: contract that is only sometimes true is a jump cut.
    frame_contract = FrameContract(
        min_frames=9,
        max_frames=497,
        quantum=8,
        native_fps=25,
        allow_tail_trim=True,
        continuity=CONTINUITY_SOFT_REFERENCE,
    )
    accepts_still = True                 # mint a still for EVERY shot (bookends incl.)
    fallback_engine = None               # NO FALLBACKS (547671d): fail LOUD
    _is_talk = True                      # I2V branch: condition on the still + audio


__all__ = ["LtxAudioInEngine"]
