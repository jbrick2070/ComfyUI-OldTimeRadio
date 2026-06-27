"""LTX-2.3 AUDIO-INPUT (A2V) lane -- additive, in-process, default-OFF / dark.

A NEW, DARK, ADDITIVE engine pair that drives video from the per-beat slice of the
FROZEN master audio + a text prompt (+ a FLUX still for the talk lane). It is NOT
the golden prompt-only ``ltx_video`` engine and shares NO code or env with it --
the two lanes diverge on purpose (this lane snaps frames UP via
``av_dims.next_8n1``; ``eng_ltx_video`` snaps DOWN). ``eng_ltx_video.py`` is FROZEN
and is never imported or touched here.

ONE adapter over the shared core (M0-GROUNDED graph; GGUF Q3_K_M proven on the
RTX 5080 at 13688 MB peak <= the 14500 ceiling, Gemma-3 encoder offloaded to CPU):

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
from . import motion_common as _MC
from .registry import EngineUnusable, EngineUsabilityReason, register

_LOG = logging.getLogger("OTR.eng_ltx_av")

# --- frame + canvas grounding (LTX-AV lane only; snap UP, never copy Lane A) ---
_LTX_AV_MIN_FRAMES = _AVD._LTX_MIN_FRAMES        # 9 (8n+1 floor)
_LTX_AV_MAX_FRAMES = int(os.environ.get("OTR_LTX_AV_MAX_FRAMES", "497"))  # M0 initial
_LTX_AV_NATIVE_W = 832
_LTX_AV_NATIVE_H = 480
# Default sampler recipe (the M0-proven 8-step distilled-ish base pass; cfg 3.0
# matched the probe). All env-overridable; never shared with eng_ltx_video.
_LTX_AV_STEPS = int(os.environ.get("OTR_LTX_AV_STEPS", "8"))
_LTX_AV_CFG = float(os.environ.get("OTR_LTX_AV_CFG", "3.0"))
_LTX_AV_I2V_STRENGTH = float(os.environ.get("OTR_LTX_AV_I2V_STRENGTH", "1.0"))
# ASCII-only negative (CLAUDE.md). One shared constant; cap 240 in the driver.
_LTX_DEFAULT_NEGATIVE = (
    "low quality, worst quality, blurry, jpeg artifacts, distorted, deformed, "
    "static, frozen pose, still image, watermark, text")

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

# --- SHARP mode: the GPU-proven distilled chain on the A2V graph (2026-06-17) ---
# OTR_LTX_AV_SHARP (default ON) selects the distilled sharpness recipe -- the
# distilled LoRA @0.70 + euler_cfg_pp + the 8-step LTX_DISTILLED_SIGMAS + cfg 1.0,
# with ModelSamplingLTXV + LTXVScheduler DROPPED (the LoRA-wrapped unet feeds the
# guider directly; the fixed sigmas already carry the shift, so ModelSamplingLTXV
# would double-shift -> blur). This is a CONFIG mode chosen at graph-build, NEVER
# an in-render fallback. =0 restores the M0 base pass. The recipe mirrors
# eng_ltx_video's distilled chain; that module is FROZEN and never imported here,
# so the tiny helpers below are DUPLICATED on purpose (V-12 cold-import).


def _sharp_enabled():
    """SHARP recipe on (default) vs the M0 base pass (OTR_LTX_AV_SHARP=0)."""
    return os.environ.get("OTR_LTX_AV_SHARP", "1") != "0"


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
    requires_flag = "OTR_ENABLE_LTX_AV"
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

    def _distilled_lora_name(self):
        """The distilled LoRA filename (sharp mode); env-overridable. Resolution +
        floor are handled by _resolve('loras', ...) like the other artifacts."""
        return os.environ.get("OTR_LTX_AV_DISTILLED_LORA",
                              _LTX_AV_DISTILLED_LORA_DEFAULT)

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
        if _sharp_enabled():
            paths.append(("distilled LoRA",
                          _resolve("loras", self._distilled_lora_name()), _FLOOR_LORA))
        return paths

    # ---- usability (fail-closed BEFORE any forward; no heavy import) ----
    def assert_usable(self, host_caps, profile, request_template=None):
        """Ordered, fail-closed-before-GPU gate (six PINNED reasons only):
        1 flag; 2 BUG-070 Sage gate; 3 NVML REQUIRED (this lane only -- fail
        closed so the ceiling guard never silently no-ops); 4 node gate (every
        required ComfyUI class resolves); 5 weight floors (realpath + size);
        6 av_dims on request_template.canvas (None tolerated)."""
        # 1 -- opt-OUT flag (DEFAULT ON: ltx_av is the music/announcer default;
        #      set OTR_ENABLE_LTX_AV=0 to disable the audio-in lane)
        if os.getenv(self.requires_flag, "1") == "0":
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.GATED_BY_FLAG,
                "%s disabled by %s=0" % (self.name, self.requires_flag),
                kind="video")
        # 2 -- BUG-070 SageAttention contamination (int8-PV aborts LTX silently)
        _MC.assert_sage_not_patched(self.name, self.family)
        # 3 -- NVML REQUIRED for the heaviest lane (grounded fail-open risk:
        #      probe_used_mb()->0 makes the ceiling asserts no-op)
        from .._otr_shared import gpu_residency as _GR
        if not _GR.nvml_available():
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.INCOMPATIBLE_PROFILE,
                "%s requires NVML to enforce the %d MB ceiling; NVML is "
                "unavailable on this host (the LTX-AV lane fails closed rather "
                "than run an unbounded heavy forward)"
                % (self.name, _MC.dynamic_vram_ceiling_mb()), kind="video")
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
        if _sharp_enabled():
            # SHARP: the LoRA-wrapped unet feeds the guider; the fixed sigmas come
            # from the in-adapter _SigmasFromValues injector -- ModelSamplingLTXV
            # and LTXVScheduler are DROPPED (the LoRA + fixed shift replace them).
            cands["lora"] = ("LoraLoaderModelOnly",)
        else:
            # M0 base pass: ModelSamplingLTXV + LTXVScheduler (no LoRA).
            cands["modelsampling"] = ("ModelSamplingLTXV",)
            cands["sched"] = ("LTXVScheduler",)
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
        sharp = _sharp_enabled()
        positive = plan.get("text_prompt") or "a vintage radio broadcast scene"
        negative = os.environ.get("OTR_LTX_AV_NEGATIVE", _LTX_DEFAULT_NEGATIVE)
        seed = int(plan.get("seed", 0) or 0)
        cfg = _LTX_AV_SHARP_CFG if sharp else _LTX_AV_CFG
        sampler_name = _LTX_AV_SHARP_SAMPLER if sharp else "euler"
        i2v_strength = _LTX_AV_SHARP_I2V_STRENGTH if sharp else _LTX_AV_I2V_STRENGTH
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
        # SHARP: distilled LoRA-wrapped unet feeds the guider (NO ModelSamplingLTXV
        # -- ManualSigmas carries the shift, so it would double-shift). M0: the
        # ModelSamplingLTXV-shifted unet feeds the guider (no LoRA).
        if sharp:
            g["lora"] = {"class": "lora", "inputs": {
                "model": W("unet", 0),
                "lora_name": self._distilled_lora_name(),
                "strength_model": _LTX_AV_DISTILLED_LORA_STRENGTH}}
            model_wire = W("lora", 0)
        else:
            g["modelsampling"] = {"class": "modelsampling", "inputs": {
                "model": W("unet", 0), "max_shift": 2.05, "base_shift": 0.95}}
            model_wire = W("modelsampling", 0)
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
        # SHARP: the fixed LTX_DISTILLED_SIGMAS via the in-adapter injector ("sigmas"
        # is added to the resolved-class map by render_clip). M0: LTXVScheduler.
        if sharp:
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
        g["decode"] = {"class": "decode", "inputs": {
            "samples": W("separate", 0), "vae": W("videovae_dec", 0),
            "tile_size": 512, "overlap": 64,
            "temporal_size": 64, "temporal_overlap": 8}}
        return g

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
        sharp = _sharp_enabled()
        plan = self._build_render_request(request)
        if not plan["audio_path"]:
            raise _wb.GraphExecutionError(
                "%s requires audio_ref (got %r)" % (self.name, plan["audio_path"]))
        if self._is_talk and not plan["init_image"]:
            raise _wb.GraphExecutionError(
                "%s (talk) requires init_image (got %r)"
                % (self.name, plan["init_image"]))
        classes = dict(getattr(self, "_classes", None)
                       or _wb.resolve_graph_classes(self._node_candidates()))
        if sharp:
            # the SIGMAS source is in-adapter (not a registered node class);
            # inject AFTER resolve so the resolver never sees it (mirrors eng_ltx_video)
            classes.setdefault("sigmas", _SigmasFromValues)
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
        graph = self._build_graph(plan, length, width, height, audio_name, image_name)
        # free_after_use (the eng_ltx_video pattern): evict the Gemma encoder +
        # intermediates before the unet+VAE-decode peak so the GGUF unet (+ the LoRA
        # patch in sharp mode) never co-resides with the encoder (the 14.5 GB
        # ceiling). KEEP the unet + the model head (lora in sharp / modelsampling
        # in M0) + the terminal so the patcher is never dangled.
        keep = {"unet", self._TERMINAL, "lora" if sharp else "modelsampling"}
        with _ltx_av_vram_reserve():
            results = _wb.run_graph(graph, classes, free_after_use=True, keep=keep)
        images = results[self._TERMINAL][0]
        self._retain_model_patchers(results, prepared)
        frames = _wb.images_to_uint8(images)
        out_path = otr_engine_tmp_mp4("otr_ltx_av_")
        path, n = _wb.encode_frames_to_silent_mp4(frames, out_path, self.target_fps)
        # BUG-291 reclaim (LOUD; never unload_all): evict the umt5/Gemma encoder +
        # idle patchers so the resident stack drops under the ceiling before the
        # PASS-PM assert and the next beat starts drained.
        _wb.reclaim_idle_models(reason="%s post-decode" % self.name)
        if not os.environ.get("OTR_TEST_MODE"):
            _MC.assert_vram_within_ceiling("%s-render" % self.name)
        return {"out_path": path, "frame_count": n}

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
        }


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
    roles = ("announcer_visual", "music_visual", "character_video")
    # the per-role DEFAULT for the music + announcer bookends (inheriting the slot
    # the deleted ltx_av_music engine held). default_roles must be subset of roles.
    default_roles = ("music_visual", "announcer_visual")
    required_inputs = ("text_prompt", "audio_ref", "init_image")
    accepts_still = True                 # mint a still for EVERY shot (bookends incl.)
    fallback_engine = None               # NO FALLBACKS (547671d): fail LOUD
    _is_talk = True                      # I2V branch: condition on the still + audio


__all__ = ["LtxAudioInEngine"]
