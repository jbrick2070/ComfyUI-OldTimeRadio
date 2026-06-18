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
from .registry import EngineUnusable, EngineUsabilityReason, register
from .wan_shared import (
    _WAN_DEFAULT_NEGATIVE, ffprobe_clip_fields, validate_silent_clip_contract)

_LOG = logging.getLogger("OTR.video.wan_ti2v")

_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
_COMFY_ROOT = os.path.dirname(os.path.dirname(_REPO_ROOT))

#: The 2.1 VAE basename the 5B must NOT use (M8): its latent compression differs.
_WAN21_VAE_BASENAME = "wan_2.1_vae.safetensors"
#: Default 8GB-tier GGUF (Q5_K_M; Apache-2.0). Resolved via folder_paths /
#: OTR_WAN_TI2V_CKPT at runtime; this basename is the fallback the loader reads.
_TI2V_DEFAULT_UNET = "Wan2.2-TI2V-5B-Q5_K_M.gguf"
_TI2V_MIN_FRAMES = 33
_TI2V_MAX_FRAMES = 177


@register
class WanTi2vEngine(_WS.WanInitImageMixin, _MC.MotionEngineBase):
    """The wan_ti2v 5B image->video adapter (8GB tier; in-process, sidecar_optional)."""

    name = "wan_ti2v"
    family = "image_to_video"
    #: Still-aspect identity (2026-06-17): Wan TI2V renders 16:9, so the director
    #: mints a WIDE init still (non-HuMo, non-mesh-portrait).
    render_aspect = "wide"
    # Same image-init roles as wan_i2v: scene b-roll, music visual, character.
    roles = ("scene_broll", "music_visual", "character_video")
    default_roles = ()
    required_inputs = ("init_image",)
    commercial_clean = True             # GGUF + VAE are Apache-2.0 (see MODEL_MANIFEST)
    requires_flag = "OTR_ENABLE_WAN_TI2V"
    engine_version = "1"
    declared_isolation = _MC.ISOLATION_SIDECAR_OPTIONAL
    target_fps = 25

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
        if not self._installed():
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "wan_ti2v UNET not found at %s; fetch the Wan2.2 TI2V-5B GGUF and "
                "set OTR_WAN_TI2V_CKPT (or drop it in models/diffusion_models)"
                % self._ckpt_path(), kind="video")
        # M8: the 5B REQUIRES the Wan2.2 VAE -- an empty name or the 2.1 VAE is a
        # silent decode-corruption trap. Guard the NAME (config) before the
        # file-presence check so a wrong-but-present 2.1 VAE fails LOUD here.
        vae_base = os.path.basename(self._loader_names()["vae"] or "").lower()
        if not vae_base or vae_base == _WAN21_VAE_BASENAME:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "wan_ti2v requires the Wan2.2 VAE; resolved VAE basename %r is "
                "empty or the 2.1 VAE (%s) -- the 5B latent compression differs, "
                "so the 2.1 VAE corrupts the decode (M8). Set OTR_WAN_TI2V_VAE_NAME"
                "=wan2.2_vae.safetensors" % (vae_base, _WAN21_VAE_BASENAME),
                kind="video")
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

    def _node_candidates(self):
        """Ordered ComfyUI node-class candidates per graph node (CORE Wan 2.2
        TI2V-5B, captured from the live /object_info 2026-06-14). The 5B latent
        node is ``Wan22ImageToVideoLatent`` (NOT ``WanImageToVideo``)."""
        unet_cls = (("UnetLoaderGGUF",) if self._loader_mode() == "gguf"
                    else ("UNETLoader",))
        return {
            "unet": unet_cls,
            "modelsampling": ("ModelSamplingSD3",),
            "clip": ("CLIPLoader",),
            "pos": ("CLIPTextEncode",),
            "neg": ("CLIPTextEncode",),
            "vae": ("VAELoader",),
            "loadimage": ("LoadImage",),
            "latent": ("Wan22ImageToVideoLatent",),
            "ksampler": ("KSampler",),
            "vaedecode": ("VAEDecode",),
        }

    def _loader_names(self):
        """Model FILENAMES the loader nodes consume (env-overridable). The VAE
        defaults to the Wan2.2 VAE (the 5B requirement, M8)."""
        return {
            "unet": os.environ.get("OTR_WAN_TI2V_UNET_NAME")
            or os.path.basename(self._ckpt_path()),
            "clip": os.environ.get(
                "OTR_WAN_TI2V_CLIP_NAME", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
            "vae": os.environ.get("OTR_WAN_TI2V_VAE_NAME", "wan2.2_vae.safetensors"),
        }

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
        steps = int(os.environ.get("OTR_WAN_TI2V_STEPS", "30"))
        cfg = float(os.environ.get("OTR_WAN_TI2V_CFG", "5.0"))
        shift = float(os.environ.get("OTR_WAN_TI2V_SHIFT", "5.0"))
        sampler = os.environ.get("OTR_WAN_TI2V_SAMPLER", "uni_pc")
        scheduler = os.environ.get("OTR_WAN_TI2V_SCHEDULER", "simple")
        positive = get("text_prompt") or "subtle natural motion"
        negative = os.environ.get("OTR_WAN_TI2V_NEGATIVE", _WAN_DEFAULT_NEGATIVE)
        if self._loader_mode() == "gguf":
            unet_inputs = {"unet_name": names["unet"]}
        else:
            unet_inputs = {"unet_name": names["unet"], "weight_dtype": "default"}
        return {
            "unet": {"class": "unet", "inputs": unet_inputs},
            "modelsampling": {"class": "modelsampling",
                              "inputs": {"model": W("unet", 0), "shift": shift}},
            "clip": {"class": "clip",
                     "inputs": {"clip_name": names["clip"], "type": "wan",
                                "device": "default"}},
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
                          "inputs": {"samples": W("ksampler", 0), "vae": W("vae", 0)}},
        }

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
        length = _wb.quantize_frames_4n1(
            plan["target_frame_count"] or self.target_fps,
            min_frames=_TI2V_MIN_FRAMES, max_frames=_TI2V_MAX_FRAMES)
        graph = self._build_graph(request, image_name, plan, length, width, height)
        # free_after_use: the umt5 text-encode frees before the 5B UNET + the 2.2
        # VAE decode; "unet" kept for V-4 patcher teardown, "vae" for the decode,
        # the terminal for the IMAGE read-out. The NVML peak probe spans the whole
        # render window so the render-phase peak gates the ceiling.
        probe = _MC.VramPeakProbe(interval_s=1.0).start()
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
