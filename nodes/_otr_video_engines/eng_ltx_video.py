"""LTX-Video text->video motion adapter (A-S5 / CW-6) -- in-process, default-OFF.

LTX-Video is a fast text-driven video generator (it can also act as a base-clip
PROVIDER for a downstream consumer). Unlike the latentsync Path-B sidecar, LTX
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

Config (env): ``OTR_ENABLE_LTX_VIDEO`` opt-in flag; ``OTR_LTX_VIDEO_CKPT`` the
primary checkpoint path the load probe checks (verify-at-build; default under
``ComfyUI/models/checkpoints``).

LK-1 look restoration (2026-06-11): ``OTR_LTX_SAMPLER`` selects the sampling
chain -- ``distilled`` (DEFAULT: the legacy 8-step LTX_DISTILLED_SIGMAS +
euler + CFGGuider cfg=1.0 path that produced the 6/5 look) or ``ksampler``
(the round-5 30-step rollback). ``OTR_ENABLE_LTX_I2V`` defaults ON: LTX shots
condition on their ST-3 scene stills (``OTR_LTX_I2V_STRENGTH`` default 1.0,
probe-proven; 0.75 re-noises into mush at 1472x832). The 22B distilled LoRA
wires only on a 22B-tier checkpoint (``OTR_LTX_DISTILLED_LORA``/``_STRENGTH``).
"""
from __future__ import annotations

import logging
import os

from . import motion_common as _MC
from .registry import EngineUnusable, EngineUsabilityReason, register

_LOG = logging.getLogger("OTR.video.eng_ltx_video")

_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
_COMFY_ROOT = os.path.dirname(os.path.dirname(_REPO_ROOT))

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


_LTX_DEFAULT_W = 768
_LTX_DEFAULT_H = 512
_LTX_DEFAULT_NEGATIVE = (
    "low quality, worst quality, blurry, distorted, watermark, text, static")

# --------------------------------------------------------------------------- #
# LK-1b (2026-06-11 look restoration): the legacy distilled sampler config.
# Recovered from the deleted OTR_BatchLTXRender (d57535ac; final form
# 70d379b~1) -- the engine that rendered the 6/5 look the operator wants
# back. Evidence trail: the box's User env pins OTR_LTX_ENGINE=v0_9 and the
# legacy node's own default was v0_9 ("visually indistinguishable from
# v2_3 on OTR content", 4.4x faster), so the production look was the 2B
# v0.9 checkpoint + THIS schedule -- NOT the 22B RES4LYF path.
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

#: The legacy workflow's distilled LoRA (nodes 60/61 consolidated @0.7,
#: bfc761a). Its keys target the LTX 2.3 22B DiT, so it is wired ONLY when
#: the active checkpoint is a 22B file -- on the default 2B v0.9 ckpt a
#: LoraLoaderModelOnly apply would be a silent key-mismatch no-op.
_LTX_DISTILLED_LORA_DEFAULT = os.path.join(
    "ltxv", "ltx2", "ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
_LTX_DISTILLED_LORA_STRENGTH = 0.7


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


@register
class LtxVideoEngine(_MC.MotionEngineBase):
    """The ltx_video text->video adapter (in-process, default-OFF / dark)."""

    name = "ltx_video"
    family = "text_to_video"
    # Generative motion b-roll / background / music visuals -- the roles whose
    # only required input is a text prompt. ``announcer_visual`` GRANTED
    # 2026-06-10 (production restore): the operator's radio OPEN renders the
    # period radio-station b-roll on the announcer beats INSTEAD of a
    # lip-synced face -- the deliberate "like before" look (the role supplies
    # text_prompt; LTX ignores the beat's audio_ref by family). A talking-face
    # announcer remains available by selecting humo on the OTR_VideoDirector
    # announcer slot.
    roles = ("scene_broll", "background_abstract", "music_visual",
             "announcer_visual")
    # PRODUCTION DEFAULTS 2026-06-10: the shipped workflow routes the radio
    # OPEN (announcer) + the theme-music visual through ltx_video, so those
    # two roles are the engine's in-stack defaults (registry-usable with no
    # env). scene_broll / background_abstract stay flag-gated extras.
    default_roles = ("announcer_visual", "music_visual")
    fallback_engine = "still_kenburns"
    required_inputs = ("text_prompt",)
    commercial_clean = False            # license is profile data; verify-at-build
    requires_flag = "OTR_ENABLE_LTX_VIDEO"
    engine_version = "1"
    declared_isolation = _MC.ISOLATION_IN_PROCESS
    target_fps = 25

    # ---- config resolution (env override -> box default) ----
    def _ckpt_path(self):
        explicit = os.environ.get("OTR_LTX_VIDEO_CKPT")
        if explicit:
            return explicit
        # Build candidate dirs: the ComfyUI-relative default + the HF_HOME-derived
        # shared-models root (C:\ComfyUI-Models\huggingface -> C:\ComfyUI-Models).
        # HF_HOME is always set in the OTR launch bat; this is the reliable source
        # on this box.  Fall back to _COMFY_ROOT for other setups.
        candidate_dirs = [os.path.join(_COMFY_ROOT, "models", "checkpoints")]
        hf_home = os.environ.get("HF_HOME", "")
        if hf_home:
            candidate_dirs.append(
                os.path.join(os.path.dirname(hf_home), "checkpoints"))
        # Prefer the versioned on-disk name (v0.9+) before the bare default.
        for d in candidate_dirs:
            for name in ("ltx-video-2b-v0.9.safetensors", "ltx-video-2b.safetensors"):
                p = os.path.join(d, name)
                if os.path.exists(p):
                    return p
        # Nothing found; return HF_HOME-derived v0.9 path (or _COMFY_ROOT) so the
        # MISSING_MODEL error message names the expected file.
        fallback_dir = (os.path.join(os.path.dirname(hf_home), "checkpoints")
                        if hf_home else candidate_dirs[0])
        return os.path.join(fallback_dir, "ltx-video-2b-v0.9.safetensors")

    def _installed(self):
        """True iff the primary checkpoint AND T5 text encoder exist on disk
        (no import -- cheap, headless-safe). The full wrapper check is the GPU
        smoke. Both are required: the v0.9 checkpoint does not bundle T5."""
        if not os.path.exists(self._ckpt_path()):
            return False
        te_name = self._text_encoder_name()
        hf_home = os.environ.get("HF_HOME", "")
        if hf_home:
            te_path = os.path.join(
                os.path.dirname(hf_home), "text_encoders", te_name)
            if not os.path.exists(te_path):
                return False
        return True

    # ---- usability (fail-closed BEFORE any forward; no heavy import) ----
    def assert_usable(self, host_caps, profile, request_template=None):
        """Fail closed before any forward: the opt-in flag, then the BUG-070
        SageAttention gate, then checkpoint presence (verify-at-build). Imports
        nothing heavy -- runs at lock/validate time on the CPU box."""
        # PROMOTED DEFAULT-ON 2026-06-10 (production restore): the saved
        # production workflow routes the announcer/music radio open through
        # ltx_video, and a ComfyUI Desktop render must work from the file
        # alone (no env patching). LTX was GPU-proven on this stack
        # (49f/14.9s, capstone night). Set OTR_ENABLE_LTX_VIDEO=0 to opt OUT;
        # the Sage gate + checkpoint presence below still fail CLOSED on any
        # box that cannot actually render it (-> LOUD fallback chain).
        if os.getenv(self.requires_flag, "1") == "0":
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.GATED_BY_FLAG,
                "%s disabled by %s=0" % (self.name, self.requires_flag),
                kind="video")
        self._assert_stack_ready()
        return self.name

    def _assert_stack_ready(self):
        """The shared post-flag usability ladder (BUG-070 Sage gate, then
        checkpoint + T5 presence). Factored so the 0-E preset engines run the
        IDENTICAL stack checks after their own opt-in flag gate."""
        _MC.assert_sage_not_patched(self.name, self.family)   # BUG-070 (S5 gate)
        if not self._installed():
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "%s not installed: checkpoint=%s (OTR_LTX_VIDEO_CKPT) + "
                "T5 encoder=%s (OTR_LTX_T5_ENCODER) -- both required for v0.9; "
                "install and verify on the GPU box"
                % (self.name, self._ckpt_path(), self._text_encoder_name()),
                kind="video")

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
        '' when none. Same key set the cheap families + latentsync read."""
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
        that exists on disk. Flag on + missing/stale init = LOUD fallback to
        the text path (never silent); flag off = text path, byte-identical."""
        if not self._i2v_enabled():
            return False
        p = self._init_image_path(request)
        if p and os.path.exists(p):
            return True
        _LOG.warning(
            "[eng_ltx_video] LTX-I2V LOUD: i2v enabled (default since LK-1a) "
            "but the request carries no on-disk init image (%r) -- falling "
            "back to the text->video path", p)
        return False

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
        # Keep "latent" (EmptyLTXVLatentVideo) -- ConditionOnly output is
        # (CONDITIONING, CONDITIONING) only; it does not produce a latent.
        cands["loadimage"] = ("LoadImage",)
        cands["img2vid"] = ("LTXVImgToVideoConditionOnly",)
        return cands

    def _build_graph_i2v(self, plan, length, width, height, image_name):
        """The declarative image-conditioned LTXV graph (BUG-LOCAL-095 fix):

          LoadImage -> LTXVImgToVideoConditionOnly(positive, negative, vae,
              image, ...) -> LTXVConditioning -> [sampler](latent from
              EmptyLTXVLatentVideo) -> VAEDecode.

        LTXVImgToVideoConditionOnly embeds the init image into the FIRST FRAME
        CONDITIONING only (outputs pos+neg conditioning, NO latent). The sampler
        gets a pure-noise EmptyLTXVLatentVideo latent so it generates real motion.
        LTXVImgToVideo (the old node) with strength=1.0 froze every frame;
        at 0.75 it re-noised into mush. ConditionOnly is the DMM-proven path.
        VERIFY-ON-GPU: exact input names of the installed wrapper's
        LTXVImgToVideoConditionOnly and decode/crop at 1472x832."""
        from . import wrapper_bridge as _wb
        W = _wb.Wire
        graph = self._build_graph(plan, length, width, height)
        # Keep "latent" (EmptyLTXVLatentVideo) -- ConditionOnly does NOT
        # produce a latent; the sampler needs pure-noise init for motion.
        graph["loadimage"] = {"class": "loadimage",
                              "inputs": {"image": image_name}}
        # LTXVImgToVideoConditionOnly: takes (positive, negative, vae, image,
        # width, height, length, batch_size); outputs (CONDITIONING, CONDITIONING).
        # No "strength" widget -- the image is always embedded at full weight
        # into the first-frame conditioning. That is what gives motion.
        graph["img2vid"] = {
            "class": "img2vid",
            "inputs": {"positive": W("pos", 0), "negative": W("neg", 0),
                       "vae": W("checkpoint", 2), "image": W("loadimage", 0),
                       "width": int(width), "height": int(height),
                       "length": int(length), "batch_size": 1}}
        graph["cond"] = {
            "class": "cond",
            "inputs": {"positive": W("img2vid", 0), "negative": W("img2vid", 1),
                       "frame_rate": float(self.target_fps)}}
        # BUG-LOCAL-095: sampler latent_image comes from the pure-noise
        # EmptyLTXVLatentVideo ("latent"), NOT from the img2vid node (which
        # produces no latent). This is what enables motion past frame 0.
        sampler_node = "sampleradv" if "sampleradv" in graph else "ksampler"
        graph[sampler_node]["inputs"]["latent_image"] = W("latent", 0)
        return graph

    # ---- in-process graph spec (GPU-VERIFIED 2026-06-09 probe_f3) ----
    # Topology:
    #   CheckpointLoaderSimple (model slot-0, vae slot-2; clip slot-1 = None
    #   for the v0.9 checkpoint -- no bundled T5) +
    #   CLIPLoader(type=ltxv, t5xxl_fp16.safetensors) (clip slot-0) ->
    #   CLIPTextEncode x2 -> EmptyLTXVLatentVideo ->
    #   LTXVConditioning -> KSampler -> VAEDecode.
    # LTX 2.3 secondary: LTXVGemmaCLIPModelLoader for the encoder slot if
    # switching to the 22B gemma-based model (requires different ckpt path +
    # different graph topology for LTXVBaseSampler; left as operator-gated).
    def _node_candidates(self):
        """Ordered ComfyUI node-class candidates per graph node.
        GPU-verified against the box's LTX v0.9 + T5-XXL install (probe_f3)."""
        return {
            "checkpoint": ("CheckpointLoaderSimple",),
            # CLIPLoader loads T5-XXL from text_encoders/ with type=ltxv.
            # Secondary: LTXAVTextEncoderLoader (LTX-AV / Gemma path; requires
            # gemma encoder + ltxv ckpt_name, different tokenizer).
            "encoder": ("CLIPLoader",),
            "pos": ("CLIPTextEncode",),
            "neg": ("CLIPTextEncode",),
            "latent": ("EmptyLTXVLatentVideo",),
            "cond": ("LTXVConditioning",),
            "ksampler": ("KSampler",),
            "vaedecode": ("VAEDecode",),
        }

    # ---- LK-1b: sampler mode + distilled LoRA resolution ----
    @staticmethod
    def _sampler_mode() -> str:
        """``distilled`` (DEFAULT -- the legacy 8-step euler/CFG-1.0 path,
        the 6/5 look) or ``ksampler`` (the round-5 30-step path, kept as
        the rollback). Invalid values fall back LOUD to distilled."""
        mode = os.environ.get("OTR_LTX_SAMPLER", "distilled").strip().lower()
        if mode not in ("distilled", "ksampler"):
            _LOG.warning("[eng_ltx_video] unknown OTR_LTX_SAMPLER=%r -- "
                         "using the distilled default", mode)
            mode = "distilled"
        return mode

    def _distilled_lora_file(self):
        """``(lora_name, abs_path)`` for the distilled LoRA; ``abs_path`` is
        '' when the file is not on disk. Searched under the ComfyUI-relative
        loras dir + the HF_HOME-derived shared-models loras dir (the same
        two roots ``_ckpt_path`` walks)."""
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

    def _use_distilled_lora(self) -> bool:
        """True iff the active checkpoint is a 22B-tier file AND the
        distilled LoRA exists on disk (verified BEFORE wiring, per the
        LK-1 ticket). The default 2B v0.9 ckpt never wires it (the LoRA's
        keys target the 22B DiT -- applying it there is a silent no-op,
        which is worse than not wiring it). A 22B ckpt with the LoRA file
        missing renders WITHOUT it, LOUD -- never silent."""
        if "22b" not in self._ckpt_name().lower():
            return False
        name, path = self._distilled_lora_file()
        if path:
            return True
        _LOG.warning(
            "[eng_ltx_video] LK-1b LOUD: 22B checkpoint %s but the distilled "
            "LoRA %r is NOT on disk -- rendering WITHOUT it (un-distilled "
            "22B at distilled sigmas degrades; install the LoRA or set "
            "OTR_LTX_DISTILLED_LORA)", self._ckpt_name(), name)
        return False

    def _node_candidates_sampling(self):
        """The ACTIVE sampling candidates: the base set, with the KSampler
        node swapped for the legacy distilled chain (KSamplerSelect +
        RandomNoise + CFGGuider + SamplerCustomAdvanced) in distilled mode,
        plus LoraLoaderModelOnly when the distilled LoRA wires. The SIGMAS
        source is the in-adapter ``_SigmasFromValues`` (injected after
        resolve -- it is not a registered node class)."""
        cands = dict(self._node_candidates())
        if self._sampler_mode() == "distilled":
            del cands["ksampler"]
            cands["samplersel"] = ("KSamplerSelect",)
            cands["noise"] = ("RandomNoise",)
            cands["guider"] = ("CFGGuider",)
            cands["sampleradv"] = ("SamplerCustomAdvanced",)
        if self._use_distilled_lora():
            cands["lora"] = ("LoraLoaderModelOnly",)
        return cands

    def _ckpt_name(self):
        return os.environ.get("OTR_LTX_VIDEO_CKPT_NAME") or os.path.basename(
            self._ckpt_path())

    def _text_encoder_name(self):
        """T5-XXL file name for CLIPLoader (type=ltxv).  Env override -> auto-
        detect from the shared HF_HOME-derived text_encoders dir -> fallback.
        Verified GPU smoke: t5xxl_fp16.safetensors (probe_f3 2026-06-09)."""
        explicit = os.environ.get("OTR_LTX_T5_ENCODER")
        if explicit:
            return explicit
        hf_home = os.environ.get("HF_HOME", "")
        if hf_home:
            te_dir = os.path.join(os.path.dirname(hf_home), "text_encoders")
            for name in ("t5xxl_fp16.safetensors", "t5xxl_fp8_e4m3fn.safetensors",
                         "t5xxl_fp8.safetensors", "t5-xxl.safetensors"):
                if os.path.exists(os.path.join(te_dir, name)):
                    return name
        return "t5xxl_fp16.safetensors"   # default name expected on this box

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
        """The declarative LTXV graph (wrapper_bridge.run_graph format).
        GPU-verified topology 2026-06-09 (probe_f3, RTX 5080):
          CheckpointLoaderSimple -> CLIPLoader(ltxv/T5-XXL) ->
          CLIPTextEncode x2 -> EmptyLTXVLatentVideo ->
          LTXVConditioning -> KSampler -> VAEDecode."""
        from . import wrapper_bridge as _wb
        W = _wb.Wire
        mode = self._sampler_mode()
        cfg_default = (_LTX_DISTILLED_CFG if mode == "distilled"
                       else _LTX_KSAMPLER_CFG)
        try:
            cfg = float(os.environ.get("OTR_LTX_CFG", str(cfg_default)))
        except (TypeError, ValueError):
            cfg = cfg_default
        positive = plan.get("text_prompt") or "a cinematic scene"
        negative = plan.get("negative_prompt") or os.environ.get(
            "OTR_LTX_NEGATIVE", _LTX_DEFAULT_NEGATIVE)
        seed = int(plan.get("seed", 0))
        graph = {
            "checkpoint": {"class": "checkpoint",
                           "inputs": {"ckpt_name": self._ckpt_name()}},
            # encoder: CLIPLoader loads T5-XXL from text_encoders/ (type=ltxv).
            # The v0.9 checkpoint does NOT bundle T5; slot-1 from CheckpointLoaderSimple
            # returns None. CLIPLoader is the correct path for this box.
            "encoder": {"class": "encoder",
                        "inputs": {"clip_name": self._text_encoder_name(),
                                   "type": "ltxv"}},
            "pos": {"class": "pos",
                    "inputs": {"text": positive, "clip": W("encoder", 0)}},
            "neg": {"class": "neg",
                    "inputs": {"text": negative, "clip": W("encoder", 0)}},
            "latent": {"class": "latent",
                       "inputs": {"width": int(width), "height": int(height),
                                  "length": int(length), "batch_size": 1}},
            "cond": {"class": "cond",
                     "inputs": {"positive": W("pos", 0), "negative": W("neg", 0),
                                "frame_rate": float(self.target_fps)}},
        }
        # LK-1b: the distilled LoRA wires between the checkpoint and the
        # sampler's MODEL input -- ONLY on a 22B-tier ckpt with the file
        # verified on disk (_use_distilled_lora). The 2B default never
        # wires it.
        model_wire = W("checkpoint", 0)
        if self._use_distilled_lora():
            lora_name, lora_path = self._distilled_lora_file()
            try:
                lora_strength = float(os.environ.get(
                    "OTR_LTX_DISTILLED_LORA_STRENGTH",
                    str(_LTX_DISTILLED_LORA_STRENGTH)))
            except (TypeError, ValueError):
                lora_strength = _LTX_DISTILLED_LORA_STRENGTH
            _LOG.warning("[eng_ltx_video] LK-1b: distilled LoRA wired: %s "
                         "@ %.2f (%s)", lora_name, lora_strength, lora_path)
            graph["lora"] = {"class": "lora",
                             "inputs": {"lora_name": lora_name,
                                        "strength_model": lora_strength,
                                        "model": model_wire}}
            model_wire = W("lora", 0)
        if mode == "distilled":
            # The legacy OTR_BatchLTXRender sampling chain, verbatim:
            # KSamplerSelect("euler") + RandomNoise(seed) + CFGGuider(1.0) +
            # SamplerCustomAdvanced(LTX_DISTILLED_SIGMAS). 8 steps; the
            # sigma schedule IS the step count (OTR_LTX_STEPS is a
            # ksampler-mode knob only).
            graph["samplersel"] = {"class": "samplersel",
                                   "inputs": {"sampler_name": "euler"}}
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
                           "cfg": cfg, "sampler_name": "euler",
                           "scheduler": "normal", "denoise": 1.0,
                           "model": model_wire,
                           "positive": W("cond", 0),
                           "negative": W("cond", 1),
                           "latent_image": W("latent", 0)}}
            samples = W("ksampler", 0)
        graph["vaedecode"] = {"class": "vaedecode",
                              "inputs": {"samples": samples,
                                         "vae": W("checkpoint", 2)}}
        return graph

    # ---- residency (resolve the installed wrapper nodes; weights load on call) -
    def load(self):
        """Fail CLOSED until installed, then RESOLVE the installed LTX node classes
        (fail-closed NAMED if absent). Weight load happens when the loader nodes
        execute inside render_clip; the live VRAM<=14.5 GB peak + SageAttention-clean
        determinism are the CW-6 GPU smoke (operator)."""
        if not self._installed():
            raise RuntimeError(
                "ltx_video not installed: checkpoint missing at %s -- install the "
                "LTX wrapper + ckpt, set OTR_ENABLE_LTX_VIDEO=1, and run the CW-6 "
                "GPU smoke" % self._ckpt_path())
        from . import wrapper_bridge as _wb
        # LK-1b: resolve the ACTIVE sampling set (distilled chain by
        # default) so the cache matches what render_clip builds.
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
        import tempfile
        plan = self._build_render_request(request)            # pure, CPU-tested
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
        # free_after_use (2026-06-09 capstone catch): the fp16 T5 encoder
        # (~9.5 GB) must NOT stay co-resident with the LTX UNET through the
        # sampler -- the first live clip breached the machine-wide 14.5 GB
        # ceiling (15.5 GB incl. the desktop baseline). The bridge frees each
        # intermediate (encoder/conds/latent) once its last consumer ran;
        # "checkpoint" is kept for the V-4 patcher teardown, the terminal for
        # the IMAGE read-out.
        results = _wb.run_graph(graph, classes, free_after_use=True,
                                keep={"checkpoint", self._TERMINAL})
        images = results[self._TERMINAL][0]                   # VAEDecode IMAGE batch
        bucket = prepared.setdefault("patchers", self._patchers) \
            if isinstance(prepared, dict) else self._patchers
        model = results.get("checkpoint", (None,))[0]
        if model is not None and callable(getattr(model, "detach", None)) \
                and id(model) not in {id(p) for p in bucket}:
            bucket.append(model)
        frames = _wb.images_to_uint8(images)
        fd, out_path = tempfile.mkstemp(suffix=".mp4", prefix="otr_ltx_")
        os.close(fd)
        path, n = _wb.encode_frames_to_silent_mp4(frames, out_path, self.target_fps)
        if not os.environ.get("OTR_TEST_MODE"):
            _MC.assert_vram_within_ceiling(self.name + "-render")  # PASS-PM mid-render
        return {"out_path": path, "frame_count": n}

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


# --------------------------------------------------------------------------- #
# ltx_orbit -- the 0-E easy on-ramp "3D-feel" camera-orbit preset (2026-06-11)
# --------------------------------------------------------------------------- #
#: The v1 orbit preset (ONE preset, turntable-style; the optional camera-LoRA
#: is v1.1 and gated on naming the exact asset + license -- 0-E spec). Appended
#: to the shot's prompt; env-overridable, deterministic given the env.
_ORBIT_PROMPT_DEFAULT = (
    "camera slowly orbiting the subject in a smooth circular arc, steady "
    "orbital dolly move, strong motion parallax, subject centered and at a "
    "consistent scale, cinematic depth")
#: Orbit-specific negative additions (the base default negative still applies;
#: these suppress the locked-off look the preset exists to avoid).
_ORBIT_NEGATIVE_DEFAULT = (
    "static camera, locked-off shot, frozen frame, camera shake, jump cut")


@register
class LtxOrbitEngine(LtxVideoEngine):
    """``ltx_orbit`` -- a camera-orbit PROMPT PRESET over the existing LTX
    adapter (0-E ticket 1; zero new deps, zero new graph topology).

    HONEST LABEL: this is "3D-feel", NOT real 3D -- it renders the same LTX
    text->video forward with orbit-camera prompt language appended; there is no
    mesh, no depth map, no camera rig. The first REAL 3D object ships with
    ``mesh_stage``. Registered SELECTABLE-NOT-DEFAULT (empty ``default_roles``
    + its own opt-in flag) until operator look-QA passes; serves ALL THREE
    OTR_VideoDirector dropdown slots (announcer / music / other-beats incl.
    character_video -- honest-label motion, no lip-sync claim).
    """

    name = "ltx_orbit"
    honest_label = "3D-feel camera orbit (LTX prompt preset -- not real 3D)"
    # All three OTR_VideoDirector slots: announcer_video_model,
    # music_video_model, and other_beats_video_model (character_video /
    # scene_broll / background_abstract). required_inputs stays
    # ("text_prompt",), which every role supplies (role_compat AS-1), so the
    # engine fits all five roles.
    roles = ("announcer_visual", "music_visual", "character_video",
             "scene_broll", "background_abstract")
    #: NEVER a default until the operator's look-QA promotes it (0-E gate).
    default_roles = ()
    requires_flag = "OTR_ENABLE_LTX_ORBIT"
    engine_version = "1"

    def assert_usable(self, host_caps, profile, request_template=None):
        """Fail closed: ltx_orbit is OPT-IN (default OFF -- the 0-E lane lands
        selectable-not-default), then the identical LTX stack ladder (BUG-070
        Sage gate + checkpoint/T5 presence) via ``_assert_stack_ready``."""
        if os.getenv(self.requires_flag, "0") != "1":
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.GATED_BY_FLAG,
                "%s is opt-in (0-E easy on-ramp; selectable-not-default until "
                "operator look-QA); set %s=1 to enable it"
                % (self.name, self.requires_flag), kind="video")
        self._assert_stack_ready()
        return self.name

    def _build_render_request(self, request):
        """Pure: the base LTX plan with the orbit preset appended to the
        positive prompt and the orbit negatives appended to the EFFECTIVE
        negative (base default preserved when the request carries none).
        Deterministic given the env; the seed flows through untouched (V-7)."""
        plan = super()._build_render_request(request)
        orbit = (os.environ.get("OTR_LTX_ORBIT_PROMPT")
                 or _ORBIT_PROMPT_DEFAULT).strip()
        base = (plan.get("text_prompt") or "").strip()
        plan["text_prompt"] = (base + ", " + orbit) if base else orbit
        orbit_neg = (os.environ.get("OTR_LTX_ORBIT_NEGATIVE")
                     or _ORBIT_NEGATIVE_DEFAULT).strip()
        base_neg = (plan.get("negative_prompt") or "").strip() or \
            os.environ.get("OTR_LTX_NEGATIVE", _LTX_DEFAULT_NEGATIVE).strip()
        plan["negative_prompt"] = (base_neg + ", " + orbit_neg) if base_neg \
            else orbit_neg
        return plan


__all__ = ["LtxVideoEngine", "LtxOrbitEngine"]
