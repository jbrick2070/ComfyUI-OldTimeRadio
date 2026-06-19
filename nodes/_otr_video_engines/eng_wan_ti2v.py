"""Wan 2.2 TI2V-5B image->video motion adapter (8GB tier) -- in-process, default-OFF.

The 5B TI2V sibling of ``wan_i2v``. It animates a still into motion on the 5B
Wan 2.2 model, which the distribution targets at the 8GB tier (GGUF Q5_K_M ~3.6 GB
UNET; Apache-2.0, commercial-clean). It runs on CORE ComfyUI Wan nodes -- NOT the
KJ wrapper (same BUG-070 / numpy-pin reasons as wan_i2v).

The 5B graph DIFFERS from the 14B I2V graph: the 5B uses ``Wan22ImageToVideoLatent``
(vae/width/height/length/batch_size + optional start_image -> LATENT) rather than
``WanImageToVideo`` (which bundles positive/negative/latent), so the KSampler takes
the CLIPTextEncode positive/negative DIRECTLY and the latent from
``Wan22ImageToVideoLatent``. The 5B core node class + the loader presence were
captured from a LIVE ``/object_info`` on the 5080 (2026-06-14, VERIFY-AT-BUILD,
GO_FORWARD 4A) before this engine was coded.

The 5B REQUIRES the Wan2.2 VAE (``wan2.2_vae.safetensors``) -- its latent
compression differs from the 2.1 VAE, so feeding the 2.1 VAE corrupts the decode.
``assert_usable`` fails CLOSED (M8) if the resolved VAE basename is empty or is the
2.1 VAE. The umt5 CLIP is shared with wan_i2v.

Registered DEFAULT-OFF / dark (empty ``default_roles`` + gated behind
``OTR_ENABLE_WAN_TI2V``); fails CLOSED until the operator enables it AND the GGUF +
the Wan2.2 VAE are on disk. The pure dims/aspect/materialize/canonicalize helpers +
the M7 silent-clip contract proof are SHARED with wan_i2v via :mod:`wan_shared`;
the loaders, node candidates and the 5B graph below are engine-specific. Import-time
is cold-import clean (V-12). UTF-8, no BOM, ASCII-only.

Config (env): ``OTR_ENABLE_WAN_TI2V`` opt-in flag; ``OTR_WAN_TI2V_CKPT`` the GGUF
(or safetensors) UNET path the load probe checks; ``OTR_WAN_TI2V_LOADER``
gguf|safetensors (else inferred from the unet extension); ``OTR_WAN_TI2V_UNET_NAME``
/ ``OTR_WAN_TI2V_CLIP_NAME`` / ``OTR_WAN_TI2V_VAE_NAME`` loader basenames;
``OTR_WAN_TI2V_CLIP_DIR`` / ``OTR_WAN_TI2V_VAE_DIR`` / ``OTR_WAN_TI2V_UNET_DIR`` dir
overrides; ``OTR_WAN_TI2V_SHIFT`` (ModelSamplingSD3 sigma shift, default 5.0 for the
5B) / ``OTR_WAN_TI2V_STEPS`` / ``OTR_WAN_TI2V_CFG`` / ``OTR_WAN_TI2V_SAMPLER`` /
``OTR_WAN_TI2V_SCHEDULER`` / ``OTR_WAN_TI2V_NEGATIVE``.
"""
from __future__ import annotations

import logging
import os

from . import motion_common as _MC
from . import wan_shared as _WS
from .._otr_shared.role_compat import ROLES
from .registry import EngineUnusable, EngineUsabilityReason, register
from .wan_shared import (
    _WAN_DEFAULT_NEGATIVE, ffprobe_clip_fields, validate_silent_clip_contract)

_LOG = logging.getLogger("OTR.video.wan_ti2v")

_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
_COMFY_ROOT = os.path.dirname(os.path.dirname(_REPO_ROOT))

#: The 2.1 VAE basename the 5B must NOT use (M8): its latent compression differs.
_WAN21_VAE_BASENAME = "wan_2.1_vae.safetensors"
#: The APPROVED Wan2.2 VAE basenames (floor fail-closed whitelist, 2026-06-18
#: roundtable): any other present-but-wrong VAE corrupts the 5B decode, so the
#: guard accepts only these rather than merely rejecting empty / the 2.1 VAE.
_WAN22_VAE_ALLOWED = frozenset({"wan2.2_vae.safetensors"})
#: Default 8GB-tier GGUF (Q5_K_M; Apache-2.0). Resolved via folder_paths /
#: OTR_WAN_TI2V_CKPT at runtime; this basename is the fallback the loader reads.
_TI2V_DEFAULT_UNET = "Wan2.2-TI2V-5B-Q5_K_M.gguf"
#: Floor default umt5 CLIP = the GGUF encoder (2026-06-18 roundtable). The official
#: fp8 (umt5_xxl_fp8_e4m3fn_scaled.safetensors) throws Float8_e4m3fn on Mac MPS
#: (ComfyUI #9255); GGUF dequantizes to fp16 -> the cross-platform 8GB path. Override
#: with OTR_WAN_TI2V_CLIP_NAME (+ OTR_WAN_TI2V_CLIP_LOADER) for an fp16 safetensors.
_TI2V_DEFAULT_CLIP = "umt5-xxl-encoder-Q5_K_M.gguf"
# Frame budgeting (2026-06-18 CLIP-FILL roundtable, supersedes the static 8GB
# floor): the old code hard-CLAMPED every clip to 17 frames (0.68s @ 25fps) so a
# ~280-frame beat froze after 0.68s (the composite held the last frame). The fix
# PREDICTS the VRAM-affordable length per beat via motion_common.
# compute_real_frame_budget (a zero-cost mem_get_info read + a cost model -- never
# react-to-OOM) bounded by the beat's audio-derived target, then the render
# ping-pong-extends that short render up to the full target so the beat is FILLED
# with motion. _TI2V_MIN_FRAMES is the motion floor (always >= this many 4n+1
# frames of real motion); _TI2V_MAX_FRAMES is the absolute hard cap;
# OTR_WAN_TI2V_MAX_FRAMES lets a tiny/8GB card pin a lower hard cap.
_TI2V_MIN_FRAMES = 17
_TI2V_DEFAULT_FRAMES = 17
_TI2V_MAX_FRAMES = 177
#: Reference render canvas used when _floor_length is called without explicit dims
#: (matches OTR_VIDEO_LANDSCAPE_CANVAS / the cost-model telemetry reference).
_TI2V_COST_REF_W = 1472
_TI2V_COST_REF_H = 832


@register
class WanTi2vEngine(_WS.WanInitImageMixin, _MC.MotionEngineBase):
    """The wan_ti2v 5B image->video adapter (8GB tier; in-process, sidecar_optional)."""

    name = "wan_ti2v"
    family = "image_to_video"
    #: Still-aspect identity (2026-06-17): Wan TI2V renders 16:9, so the director
    #: mints a WIDE init still (non-HuMo, non-mesh-portrait).
    render_aspect = "wide"
    # FLEXIBLE (operator 2026-06-18): eligible for EVERY role -- role_compat is the
    # real gate (it admits wan_ti2v only where the role supplies the required
    # init_image: announcer/music/character/scene_broll do; the pure background_
    # abstract floor does not, so it's correctly excluded there). Opening `roles`
    # lets the operator pick wan_ti2v for the announcer slot (it animates that beat's
    # selected still); the required_inputs check still prevents a truly broken pick.
    roles = ROLES
    default_roles = ()
    required_inputs = ("init_image",)
    commercial_clean = True             # GGUF + VAE are Apache-2.0 (see MODEL_MANIFEST)
    requires_flag = "OTR_ENABLE_WAN_TI2V"
    engine_version = "1"
    declared_isolation = _MC.ISOLATION_SIDECAR_OPTIONAL
    target_fps = 25
    #: Floor sampler/scheduler whitelist (2026-06-18 roundtable): only core, rock-
    #: solid CROSS-PLATFORM choices -- uni_pc/sa_solver/MoEKSampler are NVIDIA-leaning
    #: / custom / NaN-prone on MPS. assert_usable fails closed on anything else; the
    #: default below is IN the whitelist (so an unset env passes -- no self-reject).
    _PORTABLE_SAMPLERS = frozenset({"euler"})
    _PORTABLE_SCHEDULERS = frozenset({"simple", "beta", "normal"})
    _DEFAULT_SAMPLER = "euler"
    _DEFAULT_SCHEDULER = "simple"

    # ---- config resolution (env override -> box default) ----
    def _ckpt_path(self):
        return os.environ.get("OTR_WAN_TI2V_CKPT") or os.path.join(
            _COMFY_ROOT, "models", "diffusion_models", _TI2V_DEFAULT_UNET)

    def _installed(self):
        """The 5B UNET is present if the explicit path exists OR folder_paths
        resolves its basename (the GGUF lives in C:\\ComfyUI-Models via
        extra_model_paths, not under the comfy root's models/)."""
        if os.path.exists(self._ckpt_path()):
            return True
        return self._resolve_model_file(
            ("diffusion_models", "unet"), self._loader_names()["unet"],
            "OTR_WAN_TI2V_UNET_DIR") is not None

    def resolve_isolation(self):
        """``in_process`` by default; escalates to ``sidecar_required`` when
        SageAttention is resident (BUG-070). Pure."""
        return _MC.resolve_isolation(
            self.declared_isolation, _MC.sageattention_patched())

    # ---- usability (fail-closed BEFORE any forward; no heavy import) ----
    def _aux_loader_files(self):
        """(label, folder_paths categories, basename, dir-override env) for the
        REQUIRED aux graph loaders beyond the UNET ckpt: the umt5 CLIP and the
        Wan2.2 VAE (M6)."""
        names = self._loader_names()
        return (
            ("CLIP/umt5", ("text_encoders", "clip"), names["clip"],
             "OTR_WAN_TI2V_CLIP_DIR"),
            ("VAE", ("vae",), names["vae"], "OTR_WAN_TI2V_VAE_DIR"),
        )

    def assert_usable(self, host_caps, profile, request_template=None):
        """Fail closed before any forward: the opt-in flag, the UNET ckpt, the M8
        Wan2.2-VAE guard, then ALL aux graph loaders (umt5 CLIP + the 2.2 VAE)
        present on disk."""
        if os.getenv(self.requires_flag, "0") != "1":
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.GATED_BY_FLAG,
                "wan_ti2v is opt-in; set %s=1 (core Comfy Wan 5B nodes, no KJ "
                "wrapper)" % self.requires_flag, kind="video")
        # Fail CLOSED on a bad/non-portable render knob (sampler whitelist + env
        # range-checks) BEFORE any forward -- never a raw crash mid-render.
        self._resolve_render_config()
        if not self._installed():
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "wan_ti2v UNET not found at %s; fetch the Wan2.2 TI2V-5B GGUF and "
                "set OTR_WAN_TI2V_CKPT (or drop it in models/diffusion_models)"
                % self._ckpt_path(), kind="video")
        # M8: the 5B REQUIRES the Wan2.2 VAE -- an empty / 2.1 / any other
        # wrong-but-present VAE silently corrupts the decode. Fail CLOSED unless the
        # resolved basename is in the approved Wan2.2 whitelist (2026-06-18 roundtable
        # tightened this from "not empty / not 2.1" to an explicit allow-list).
        vae_base = os.path.basename(self._loader_names()["vae"] or "").lower()
        if vae_base not in _WAN22_VAE_ALLOWED:
            _why = ("the 2.1 VAE (latent compression differs)"
                    if vae_base == _WAN21_VAE_BASENAME
                    else "empty" if not vae_base else "not an approved Wan2.2 VAE")
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "wan_ti2v requires the Wan2.2 VAE; resolved basename %r is %s -- the "
                "5B decode needs one of %s (M8). Set OTR_WAN_TI2V_VAE_NAME"
                "=wan2.2_vae.safetensors"
                % (vae_base, _why, sorted(_WAN22_VAE_ALLOWED)), kind="video")
        missing = self._missing_loaders()
        if missing:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "wan_ti2v required loader file(s) absent: %s -- the 5B graph needs "
                "the umt5 CLIP and the Wan2.2 VAE on disk too, not just the UNET "
                "(offline invariant, no runtime fetch); fix the *_NAME envs or "
                "point OTR_WAN_TI2V_CLIP_DIR / OTR_WAN_TI2V_VAE_DIR at the "
                "installed basenames"
                % ", ".join("%s=%r" % (lbl, nm) for lbl, nm in missing),
                kind="video")
        return self.name

    #: Terminal node of the graph (its IMAGE output is encoded to the clip).
    _TERMINAL = "vaedecode"

    # ---- in-process graph spec (CORE Wan 2.2 TI2V-5B; verified vs /object_info) -
    def _loader_mode(self):
        """``gguf`` (``UnetLoaderGGUF``) or ``safetensors`` (``UNETLoader``).
        Explicit via ``OTR_WAN_TI2V_LOADER``; else inferred from the unet
        extension (the 8GB-tier default is the GGUF)."""
        mode = (os.environ.get("OTR_WAN_TI2V_LOADER") or "").strip().lower()
        if mode in ("gguf", "safetensors"):
            return mode
        return ("gguf" if self._loader_names()["unet"].lower().endswith(".gguf")
                else "safetensors")

    def _clip_loader_mode(self):
        """``gguf`` (``CLIPLoaderGGUF``) or ``safetensors`` (``CLIPLoader``) for the
        umt5 encoder. Explicit via ``OTR_WAN_TI2V_CLIP_LOADER``; else inferred from
        the clip extension (the floor default is the GGUF umt5 -- fp8 safetensors
        throws Float8_e4m3fn on Mac MPS, ComfyUI #9255, 2026-06-18 roundtable)."""
        mode = (os.environ.get("OTR_WAN_TI2V_CLIP_LOADER") or "").strip().lower()
        if mode in ("gguf", "safetensors"):
            return mode
        return ("gguf" if self._loader_names()["clip"].lower().endswith(".gguf")
                else "safetensors")

    def _tiled_vae(self):
        """Whether to decode through ``VAEDecodeTiled`` (the floor default ON: the
        video-VAE decode is a top VRAM-peak driver at 8GB). ``OTR_WAN_TI2V_TILED_VAE``
        falsey {0,false,no,off} forces the plain ``VAEDecode``."""
        return (os.environ.get("OTR_WAN_TI2V_TILED_VAE", "1").strip().lower()
                not in ("0", "false", "no", "off"))

    def _node_candidates(self):
        """Ordered ComfyUI node-class candidates per graph node (CORE Wan 2.2
        TI2V-5B + ComfyUI-GGUF, schema-verified vs the live /object_info 2026-06-18).
        The 5B latent node is ``Wan22ImageToVideoLatent`` (NOT ``WanImageToVideo``).
        The CLIP loader + the VAE-decode node switch per the floor knobs above."""
        unet_cls = (("UnetLoaderGGUF",) if self._loader_mode() == "gguf"
                    else ("UNETLoader",))
        clip_cls = (("CLIPLoaderGGUF",) if self._clip_loader_mode() == "gguf"
                    else ("CLIPLoader",))
        vaedecode_cls = (("VAEDecodeTiled",) if self._tiled_vae()
                         else ("VAEDecode",))
        return {
            "unet": unet_cls,
            "modelsampling": ("ModelSamplingSD3",),
            "clip": clip_cls,
            "pos": ("CLIPTextEncode",),
            "neg": ("CLIPTextEncode",),
            "vae": ("VAELoader",),
            "loadimage": ("LoadImage",),
            "latent": ("Wan22ImageToVideoLatent",),
            "ksampler": ("KSampler",),
            "vaedecode": vaedecode_cls,
        }

    def _loader_names(self):
        """Model FILENAMES the loader nodes consume (env-overridable). The VAE
        defaults to the Wan2.2 VAE (the 5B requirement, M8)."""
        return {
            "unet": os.environ.get("OTR_WAN_TI2V_UNET_NAME")
            or os.path.basename(self._ckpt_path()),
            "clip": os.environ.get(
                "OTR_WAN_TI2V_CLIP_NAME", _TI2V_DEFAULT_CLIP),
            "vae": os.environ.get("OTR_WAN_TI2V_VAE_NAME", "wan2.2_vae.safetensors"),
        }

    def _resolve_render_config(self):
        """Parse + RANGE-CHECK the render knobs ONCE (shared by assert_usable and
        _build_graph, 2026-06-18 roundtable). A bad env value fails CLOSED here with
        a named MALFORMED_CONFIG, never a raw int()/float() crash mid-render. The
        sampler/scheduler are validated against the cross-platform floor whitelist."""
        def _num(env, dflt, lo, hi, cast):
            raw = os.environ.get(env)
            if raw is None or raw == "":
                return cast(dflt)
            try:
                val = cast(raw)
            except (TypeError, ValueError):
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                    "%s=%r is not a valid number" % (env, raw), kind="video")
            if not (lo <= val <= hi):
                raise EngineUnusable(
                    self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                    "%s=%s out of range [%s, %s]" % (env, val, lo, hi), kind="video")
            return val

        sampler = (os.environ.get("OTR_WAN_TI2V_SAMPLER")
                   or self._DEFAULT_SAMPLER).strip()
        if sampler not in self._PORTABLE_SAMPLERS:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                "OTR_WAN_TI2V_SAMPLER=%r is not in the cross-platform floor whitelist "
                "%s -- wan_ti2v is the 8GB/Mac/AMD floor; uni_pc/sa_solver/MoEKSampler "
                "are not portable. Use a heavier engine for those."
                % (sampler, sorted(self._PORTABLE_SAMPLERS)), kind="video")
        scheduler = (os.environ.get("OTR_WAN_TI2V_SCHEDULER")
                     or self._DEFAULT_SCHEDULER).strip()
        if scheduler not in self._PORTABLE_SCHEDULERS:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MALFORMED_CONFIG,
                "OTR_WAN_TI2V_SCHEDULER=%r is not in the floor whitelist %s"
                % (scheduler, sorted(self._PORTABLE_SCHEDULERS)), kind="video")
        return {
            "steps": _num("OTR_WAN_TI2V_STEPS", 30, 1, 100, int),
            "cfg": _num("OTR_WAN_TI2V_CFG", 5.0, 0.0, 30.0, float),
            "shift": _num("OTR_WAN_TI2V_SHIFT", 5.0, 0.1, 20.0, float),
            "sampler": sampler,
            "scheduler": scheduler,
        }

    def _floor_length(self, target_frame_count, width=None, height=None):
        """The VRAM-PREDICTED render length (4n+1) for this beat (clip-fill fix).

        REPLACES the old hard 17-frame "8GB floor" that froze every wan clip to
        0.68s: ``motion_common.compute_real_frame_budget`` reads live free VRAM
        (zero-cost ``mem_get_info``) + a cost model to PREDICT how many of the
        beat's audio-derived ``target_frame_count`` frames fit under the ceiling --
        never react-to-OOM. The render then ping-pong-extends this (possibly short)
        render up to the full target so the beat is FILLED with motion. The motion
        floor (17) always wins; ``OTR_WAN_TI2V_MAX_FRAMES`` pins an absolute hard
        cap for a tiny/8GB card (default = the engine max, not 17)."""
        from . import wrapper_bridge as _wb
        target = int(target_frame_count or _TI2V_DEFAULT_FRAMES)
        try:
            hard_cap = int(os.environ.get(
                "OTR_WAN_TI2V_MAX_FRAMES", str(_TI2V_MAX_FRAMES)))
        except (TypeError, ValueError):
            hard_cap = _TI2V_MAX_FRAMES
        hard_cap = max(_TI2V_MIN_FRAMES, min(hard_cap, _TI2V_MAX_FRAMES))
        target = max(1, min(target, hard_cap))
        if width is None or height is None:
            width, height = _TI2V_COST_REF_W, _TI2V_COST_REF_H
        budget = _MC.compute_real_frame_budget(
            _MC.free_vram_mb(), target, int(width), int(height), self.name)
        return _wb.quantize_frames_4n1(
            budget, min_frames=_TI2V_MIN_FRAMES, max_frames=hard_cap)

    def _build_graph(self, request, image_name, plan, length, width, height):
        """The declarative Wan TI2V-5B graph (wrapper_bridge.run_graph format).
        ``Wan22ImageToVideoLatent`` builds the latent from the VAE + init image; the
        KSampler takes the CLIPTextEncode positive/negative DIRECTLY (unlike the
        14B's WanImageToVideo, which bundles them). ModelSamplingSD3 applies the 5B
        sigma shift (default 5.0)."""
        from . import wrapper_bridge as _wb
        W = _wb.Wire
        names = self._loader_names()
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        cfg_knobs = self._resolve_render_config()        # range-checked, fail-closed
        steps = cfg_knobs["steps"]
        cfg = cfg_knobs["cfg"]
        shift = cfg_knobs["shift"]
        sampler = cfg_knobs["sampler"]
        scheduler = cfg_knobs["scheduler"]
        positive = get("text_prompt") or "subtle natural motion"
        negative = os.environ.get("OTR_WAN_TI2V_NEGATIVE", _WAN_DEFAULT_NEGATIVE)
        if self._loader_mode() == "gguf":
            unet_inputs = {"unet_name": names["unet"]}
        else:
            unet_inputs = {"unet_name": names["unet"], "weight_dtype": "default"}
        # CLIPLoaderGGUF takes clip_name + type ONLY (no `device` arg, verified vs
        # /object_info 2026-06-18); the core CLIPLoader also takes device.
        if self._clip_loader_mode() == "gguf":
            clip_inputs = {"clip_name": names["clip"], "type": "wan"}
        else:
            clip_inputs = {"clip_name": names["clip"], "type": "wan",
                           "device": "default"}
        return {
            "unet": {"class": "unet", "inputs": unet_inputs},
            "modelsampling": {"class": "modelsampling",
                              "inputs": {"model": W("unet", 0), "shift": shift}},
            "clip": {"class": "clip", "inputs": clip_inputs},
            "pos": {"class": "pos",
                    "inputs": {"text": positive, "clip": W("clip", 0)}},
            "neg": {"class": "neg",
                    "inputs": {"text": negative, "clip": W("clip", 0)}},
            "vae": {"class": "vae", "inputs": {"vae_name": names["vae"]}},
            "loadimage": {"class": "loadimage", "inputs": {"image": image_name}},
            "latent": {"class": "latent",
                       "inputs": {"width": int(width), "height": int(height),
                                  "length": int(length), "batch_size": 1,
                                  "vae": W("vae", 0),
                                  "start_image": W("loadimage", 0)}},
            "ksampler": {"class": "ksampler",
                         "inputs": {"seed": int(plan.get("seed", 0)), "steps": steps,
                                    "cfg": cfg, "sampler_name": sampler,
                                    "scheduler": scheduler, "denoise": 1.0,
                                    "model": W("modelsampling", 0),
                                    "positive": W("pos", 0),
                                    "negative": W("neg", 0),
                                    "latent_image": W("latent", 0)}},
            "vaedecode": {"class": "vaedecode",
                          "inputs": self._vaedecode_inputs(W)},
        }

    def _vaedecode_inputs(self, W):
        """VAEDecode inputs; the tiled path adds the schema-verified tile/temporal
        knobs (8GB floor defaults; temporal_size = frames decoded at a time = the
        big video-VAE peak lever). All env-overridable."""
        base = {"samples": W("ksampler", 0), "vae": W("vae", 0)}
        if not self._tiled_vae():
            return base
        def _i(env, dflt):
            try:
                return int(os.environ.get(env, str(dflt)))
            except (TypeError, ValueError):
                return dflt
        base.update({
            "tile_size": _i("OTR_WAN_TI2V_VAE_TILE", 256),
            "overlap": _i("OTR_WAN_TI2V_VAE_OVERLAP", 64),
            "temporal_size": _i("OTR_WAN_TI2V_VAE_TEMPORAL", 16),
            "temporal_overlap": _i("OTR_WAN_TI2V_VAE_TEMPORAL_OVERLAP", 8),
        })
        return base

    # ---- residency ----
    def load(self):
        """Fail CLOSED until installed, then resolve the installed CORE Wan node
        classes. Weights load when the loader nodes execute in render_clip."""
        if not self._installed():
            raise RuntimeError(
                "wan_ti2v not installed: UNET missing at %s -- fetch the Wan2.2 "
                "TI2V-5B GGUF, set OTR_ENABLE_WAN_TI2V=1" % self._ckpt_path())
        from . import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
        self._loaded = True

    def render_clip(self, request, prepared):
        """Drive ONE image->video clip via the in-process Wan TI2V-5B graph: stage
        the init image, execute the graph, encode the decoded IMAGE batch to a
        SILENT bt709 clip (V-1), retain the MODEL patcher for V-4 teardown, and
        assert the mid-render NVML ceiling. M7 ffprobe-proves the silent-clip
        contract before the mux trusts it. Fail-closed NAMED on a missing node /
        init image."""
        from . import wrapper_bridge as _wb
        from ._tmp import otr_engine_tmp_mp4
        plan = self._build_render_request(request)            # pure, CPU-tested
        if not plan["init_image"]:
            raise _wb.GraphExecutionError(
                "wan_ti2v requires init_image (got %r)" % plan["init_image"])
        classes = getattr(self, "_classes", None) \
            or _wb.resolve_graph_classes(self._node_candidates())
        width, height = self._dims(request)
        image_name = self._materialize_init_image(
            request, plan["init_image"], width, height)
        # CLIP-FILL: PREDICT the VRAM-affordable render length for THIS canvas
        # (never react-to-OOM); the short render is ping-pong-extended to the beat
        # target below so the composite never freeze-fills the beat.
        length = self._floor_length(plan["target_frame_count"], width, height)
        graph = self._build_graph(request, image_name, plan, length, width, height)
        # free_after_use: the umt5 text-encode frees before the 5B UNET + the 2.2
        # VAE decode; "unet" kept for V-4 patcher teardown, "vae" for the decode,
        # the terminal for the IMAGE read-out. The NVML peak probe spans the whole
        # render window so the render-phase peak gates the ceiling.
        probe = _MC.VramPeakProbe(interval_s=0.1).start()
        try:
            results = _wb.run_graph(graph, classes, free_after_use=True,
                                    keep={"unet", "vae", self._TERMINAL})
            images = results[self._TERMINAL][0]               # VAEDecode IMAGE batch
        finally:
            render_peak = probe.stop()
        bucket = prepared.setdefault("patchers", self._patchers) \
            if isinstance(prepared, dict) else self._patchers
        model = results.get("unet", (None,))[0]
        if model is not None and callable(getattr(model, "detach", None)) \
                and id(model) not in {id(p) for p in bucket}:
            bucket.append(model)
        frames = _wb.images_to_uint8(images)
        # CLIP-FILL: ping-pong-extend the VRAM-bounded short render up to the
        # beat's audio-derived target so the composite fills the beat with motion
        # instead of holding the last frame (the 0.68s-then-freeze bug). A no-op
        # when the native render already meets the target (extend returns as-is).
        target_frames = int(plan.get("target_frame_count") or 0)
        n_native = len(frames)
        if target_frames > n_native:
            frames = _wb.extend_frames_to_target(frames, target_frames)
            _LOG.warning(
                "[OTR video] wan_ti2v CLIP-FILL: rendered %d frame(s) -> "
                "ping-pong extended to %d (beat target %d) @ %dx%d so the beat "
                "is FILLED with motion (no hold-last-frame freeze)",
                n_native, len(frames), target_frames, width, height)
        out_path = otr_engine_tmp_mp4("otr_wan_ti2v_")
        path, n = _wb.encode_frames_to_silent_mp4(frames, out_path, self.target_fps)
        # M7: PROVE the silent-clip color/stream contract on the emitted mp4.
        validate_silent_clip_contract(ffprobe_clip_fields(path), self.target_fps)
        if not os.environ.get("OTR_TEST_MODE"):
            post_mb = _MC.vram_used_mb() or 0
            _LOG.info("[OTR video] wan_ti2v VRAM render-phase peak %s MB / post %s "
                      "MB (ceiling %s MB)", render_peak, post_mb,
                      _MC.dynamic_vram_ceiling_mb())
            _MC.assert_peak_within_ceiling(render_peak, "wan_ti2v-render")
        return {"out_path": path, "frame_count": n}

    def canonicalize(self, raw, request, profile):
        return self._clip_from_raw(raw, request)


__all__ = ["WanTi2vEngine"]
