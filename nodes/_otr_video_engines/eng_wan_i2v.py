"""Wan 2.2 image->video motion adapter (A-S5 / CW-6) -- in-process, default-OFF.

Wan i2v animates a still init image into motion. It runs on CORE ComfyUI Wan
nodes (``UNETLoader`` / ``CLIPLoader`` type=wan / ``WanImageToVideo`` / ``KSampler``
/ ``VAEDecode`` + a ``ModelSamplingSD3`` sigma shift) -- NOT the KJ Wan wrapper.
The KJ wrapper's pin audit demands numpy<2 / transformers<=4.51.3, but this box is
numpy 2.4 / transformers 5.5, and KJNodes is what drags in SageAttention; core
Comfy supports the A14B natively, so the engine deliberately stays off that dep
path (GO_FORWARD section 1A, 2026-06-12).

Like ltx_video it runs IN-PROCESS in the main cu130 venv by default. Isolation is
``sidecar_optional``: ``resolve_isolation`` escalates to a cu128 sidecar ONLY when
SageAttention is ACTIVE (``comfy.model_management.sage_attention_enabled()``, the
real switch -- mere module residency from Comfy's import-time probe does not
count, BUG-070). The headless WAN lane boots without ``--use-sage-attention`` so
the in-process path is the norm. Registered DEFAULT-OFF / dark (empty
``default_roles`` + gated behind ``OTR_ENABLE_WAN_I2V``); fails CLOSED until the
operator enables it AND the checkpoint is on disk.

init_image aspect: the init is MATERIALIZED into the canvas (padded / cropped per
``aspect_policy`` with ONE uniform scale, default ``pad``) and the derived image
is staged, so ``WanImageToVideo``'s internal resize is a no-op and a portrait init
NEVER silently stretches into a landscape canvas (pre-mortem N9). Import-time is
cold-import clean (V-12). UTF-8, no BOM, ASCII-only.

The pure dims/aspect/materialize/canonicalize helpers + the M7 silent-clip
contract proof are SHARED with wan_ti2v via :mod:`wan_shared` (GO_FORWARD 4A:
"share only pure helpers; keep loaders + node candidates + graph SEPARATE"); the
loaders, node candidates and the 14B I2V graph below stay engine-specific.

Config (env): ``OTR_ENABLE_WAN_I2V`` opt-in flag; ``OTR_WAN_I2V_CKPT`` the primary
checkpoint path the load probe checks; ``OTR_WAN_I2V_LOADER`` safetensors|gguf
(else inferred from the unet extension); ``OTR_WAN_SHIFT`` (ModelSamplingSD3 sigma
shift, default 8.0 for the 14B); ``OTR_WAN_STEPS`` / ``OTR_WAN_CFG`` /
``OTR_WAN_SAMPLER`` / ``OTR_WAN_SCHEDULER`` / ``OTR_WAN_NEGATIVE``.
"""
from __future__ import annotations

import logging
import os

from . import motion_common as _MC
from . import wan_shared as _WS
from .registry import EngineUnusable, EngineUsabilityReason, register
# Re-export the shared M7 clip-contract helpers so render_clip below and the
# existing test imports (``from ...eng_wan_i2v import validate_silent_clip_contract``)
# keep resolving after the helpers moved to wan_shared.
from .wan_shared import (  # noqa: F401
    _WAN_DEFAULT_NEGATIVE, _parse_fps, ffprobe_clip_fields,
    validate_silent_clip_contract)

_LOG = logging.getLogger("OTR.video.wan_i2v")

_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
_COMFY_ROOT = os.path.dirname(os.path.dirname(_REPO_ROOT))

# A-ship in-process forward. Shared mechanics (resolution, declarative executor,
# silent bt709 encode, VRAM guard) are proven in wrapper_bridge; the GRAPH below is
# the CORE-NODE Wan 2.2 image->video topology, VERIFIED against the installed
# /object_info on the 5080 (2026-06-12, GO_FORWARD 1A): UNETLoader (or UnetLoaderGGUF)
# -> ModelSamplingSD3(shift) -> KSampler.model; CLIPLoader(type=wan) -> CLIPTextEncode
# x2; VAELoader + LoadImage -> WanImageToVideo[positive,negative,latent] -> KSampler
# (uni_pc/simple) -> VAEDecode. The bare /prompt smoke (scripts/otr_wan_smoke.py)
# rendered this graph end-to-end (832x480, 81f, real motion, no warp). Filenames /
# knobs are env-overridable; the init_image is materialized padded/cropped (never
# stretched, N9). NOTE: the low-noise 14B fp8 is a SINGLE expert -- the two-expert
# HIGH/LOW MoE handoff (Path B) is a future option, not wired here.
_WAN_MIN_FRAMES = 33
_WAN_MAX_FRAMES = 177


@register
class WanI2VEngine(_WS.WanInitImageMixin, _MC.MotionEngineBase):
    """The wan_i2v image->video adapter (in-process default; sidecar_optional)."""

    name = "wan_i2v"
    family = "image_to_video"
    #: Still-aspect identity (2026-06-17): Wan I2V renders 16:9, so the director
    #: mints a WIDE init still (non-HuMo, non-mesh-portrait).
    render_aspect = "wide"
    # Animate a still into motion -- the roles that can supply an init image
    # (music visual, character). (scene_broll/background_abstract removed
    # 2026-07-01, rip-sfx-broll.)
    roles = ("music_visual", "character_video")
    default_roles = ()
    required_inputs = ("init_image",)
    commercial_clean = False            # license is profile data; verify-at-build
    requires_flag = None  # vestigial (registry IS the menu; no flag gate)
    engine_version = "1"
    declared_isolation = _MC.ISOLATION_SIDECAR_OPTIONAL
    target_fps = 25

    # ---- config resolution (env override -> box default) ----
    def _ckpt_path(self):
        return os.environ.get("OTR_WAN_I2V_CKPT") or os.path.join(
            _COMFY_ROOT, "models", "checkpoints", "wan2.2-i2v.safetensors")

    def _installed(self):
        return os.path.exists(self._ckpt_path())

    def resolve_isolation(self):
        """Runtime isolation tier: ``in_process`` by default, escalating to
        ``sidecar_required`` when SageAttention is resident (BUG-070). Pure read
        of the declared tier + ``sys.modules``."""
        return _MC.resolve_isolation(
            self.declared_isolation, _MC.sageattention_patched())

    # ---- usability (fail-closed BEFORE any forward; no heavy import) ----
    def _aux_loader_files(self):
        """(label, folder_paths categories, basename, dir-override env) for the
        REQUIRED aux graph loaders beyond the UNET ckpt: the umt5 CLIP and the
        VAE. M6 (GO_FORWARD 4A): a forward dies mid-graph without these, so
        assert_usable proves all three loaders present, not just the ckpt."""
        names = self._loader_names()
        return (
            ("CLIP/umt5", ("text_encoders", "clip"), names["clip"],
             "OTR_WAN_I2V_CLIP_DIR"),
            ("VAE", ("vae",), names["vae"], "OTR_WAN_I2V_VAE_DIR"),
        )

    def assert_usable(self, host_caps, profile, request_template=None):
        """Fail closed before any forward: ALL three graph loaders
        (UNET + umt5 CLIP + VAE) present on disk. Wan TOLERATES Sage by
        escaping to a sidecar (``resolve_isolation``), so it does NOT hard-fail
        on Sage the way ltx_video does."""
        if not self._installed():
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "wan_i2v checkpoint not found at %s; fetch the Wan 2.2 ckpt and "
                "set OTR_WAN_I2V_CKPT (the core UNETLoader reads the basename)"
                % self._ckpt_path(), kind="video")
        missing = self._missing_loaders()
        if missing:
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "wan_i2v required loader file(s) absent: %s -- the core Wan graph "
                "needs the umt5 CLIP and the VAE on disk too, not just the UNET "
                "ckpt (offline invariant, no runtime fetch); fix the *_NAME envs "
                "or point OTR_WAN_I2V_CLIP_DIR / OTR_WAN_I2V_VAE_DIR at the "
                "installed basenames"
                % ", ".join("%s=%r" % (lbl, nm) for lbl, nm in missing),
                kind="video")
        return self.name

    #: Terminal node of the graph (its IMAGE output is encoded to the clip).
    _TERMINAL = "vaedecode"

    # ---- in-process graph spec (CORE Wan i2v; verified vs /object_info) ----
    def _loader_mode(self):
        """The UNET loader mode: ``safetensors`` (core ``UNETLoader``) or ``gguf``
        (``UnetLoaderGGUF``). Explicit via ``OTR_WAN_I2V_LOADER``; else inferred
        from the unet filename extension. Fail-closed: the resolver raises NAMED if
        the chosen loader class is not installed."""
        mode = (os.environ.get("OTR_WAN_I2V_LOADER") or "").strip().lower()
        if mode in ("gguf", "safetensors"):
            return mode
        return ("gguf" if self._loader_names()["unet"].lower().endswith(".gguf")
                else "safetensors")

    def _node_candidates(self):
        """Ordered ComfyUI node-class candidates per graph node (CORE Wan 2.2 i2v,
        verified against the installed /object_info 2026-06-12). The unet loader
        switches on ``_loader_mode`` so both the fp8 safetensors and a GGUF quant
        resolve fail-closed."""
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
            "wan": ("WanImageToVideo",),
            "ksampler": ("KSampler",),
            "vaedecode": ("VAEDecode",),
        }

    def _loader_names(self):
        """Model FILENAMES the loader nodes consume (env-overridable; defaults are
        placeholders the operator confirms against the installed Wan models)."""
        return {
            "unet": os.environ.get("OTR_WAN_I2V_UNET_NAME")
            or os.path.basename(self._ckpt_path()),
            "clip": os.environ.get(
                "OTR_WAN_I2V_CLIP_NAME", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
            "vae": os.environ.get("OTR_WAN_I2V_VAE_NAME", "wan_2.1_vae.safetensors"),
        }

    def _build_graph(self, request, image_name, plan, length, width, height):
        """The declarative Wan i2v graph (wrapper_bridge.run_graph format). CORE
        Wan 2.2 topology, verified against the installed /object_info. The unet
        loader emits ``weight_dtype`` only on the safetensors path; the GGUF path
        (UnetLoaderGGUF) takes ``unet_name`` alone. ModelSamplingSD3 applies the
        sigma shift (~8.0 for the 14B) the bare placeholder omitted."""
        from . import wrapper_bridge as _wb
        W = _wb.Wire
        names = self._loader_names()
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        steps = int(os.environ.get("OTR_WAN_STEPS", "20"))
        cfg = float(os.environ.get("OTR_WAN_CFG", "3.5"))
        shift = float(os.environ.get("OTR_WAN_SHIFT", "8.0"))
        sampler = os.environ.get("OTR_WAN_SAMPLER", "uni_pc")
        scheduler = os.environ.get("OTR_WAN_SCHEDULER", "simple")
        positive = get("text_prompt") or "subtle natural motion"
        negative = os.environ.get("OTR_WAN_NEGATIVE", _WAN_DEFAULT_NEGATIVE)
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
            "wan": {"class": "wan",
                    "inputs": {"width": int(width), "height": int(height),
                               "length": int(length), "batch_size": 1,
                               "positive": W("pos", 0), "negative": W("neg", 0),
                               "vae": W("vae", 0), "start_image": W("loadimage", 0)}},
            "ksampler": {"class": "ksampler",
                         "inputs": {"seed": int(plan.get("seed", 0)), "steps": steps,
                                    "cfg": cfg, "sampler_name": sampler,
                                    "scheduler": scheduler, "denoise": 1.0,
                                    "model": W("modelsampling", 0),
                                    "positive": W("wan", 0),
                                    "negative": W("wan", 1),
                                    "latent_image": W("wan", 2)}},
            "vaedecode": {"class": "vaedecode",
                          "inputs": {"samples": W("ksampler", 0), "vae": W("vae", 0)}},
        }

    # ---- residency (resolve the installed wrapper nodes; weights load on call) -
    def load(self):
        """Fail CLOSED until installed, then RESOLVE the installed CORE Wan node
        classes (fail-closed NAMED if absent). Weight load happens when the loader
        nodes execute inside render_clip; the live render-phase VRAM<=14.5 GB peak
        is asserted there via the NVML probe."""
        if not self._installed():
            raise RuntimeError(
                "wan_i2v not installed: checkpoint missing at %s -- fetch the Wan "
                "2.2 ckpt, set OTR_ENABLE_WAN_I2V=1 (core Comfy Wan nodes)"
                % self._ckpt_path())
        from . import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
        self._loaded = True

    def render_clip(self, request, prepared):
        """Drive ONE image->video clip via the in-process Wan wrapper graph: stage
        the init image into ComfyUI's input dir, execute the graph, encode the
        decoded IMAGE batch to a SILENT bt709 clip (V-1), retain the MODEL patcher
        for V-4 teardown, and assert the mid-render NVML ceiling. Returns the raw
        ``{out_path, frame_count}`` canonicalize() normalises. Fail-closed NAMED if
        a wrapper node or the init image is missing."""
        from . import wrapper_bridge as _wb
        from ._tmp import otr_engine_tmp_mp4
        plan = self._build_render_request(request)            # pure, CPU-tested
        if not plan["init_image"]:
            raise _wb.GraphExecutionError(
                "wan_i2v requires init_image (got %r)" % plan["init_image"])
        classes = getattr(self, "_classes", None) \
            or _wb.resolve_graph_classes(self._node_candidates())
        width, height = self._dims(request)
        # Materialize the init into the canvas (pad/crop, ONE uniform scale, N9) and
        # stage THAT, so WanImageToVideo's internal resize is a no-op and the
        # declared aspect_policy (default pad) is actually honoured.
        image_name = self._materialize_init_image(
            request, plan["init_image"], width, height)
        length = _wb.quantize_frames_4n1(
            plan["target_frame_count"] or self.target_fps,
            min_frames=_WAN_MIN_FRAMES, max_frames=_WAN_MAX_FRAMES)
        graph = self._build_graph(request, image_name, plan, length, width, height)
        # free_after_use (2026-06-09, the LTX capstone catch applied here
        # preemptively): umt5-fp8 + the 14B fp8 Wan UNET must not stay
        # co-resident through the sampler on a 16 GB card. "unet" is kept for
        # the V-4 patcher teardown, "vae" for the decode, the terminal for the
        # IMAGE read-out. The NVML peak probe spans the whole render window (first
        # heavy load through VAEDecode) so the render-phase peak (not a post-encode
        # instantaneous read) is what gates the 14.5 GB ceiling. (CS-4: if this
        # peak ever busts, split text-encode into a pre-pass that frees umt5 before
        # the 14B loads -- the bare /prompt smoke without free_after_use peaked at
        # the very edge, 14499 MB; the engine's free_after_use is the mitigation.)
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
        out_path = otr_engine_tmp_mp4("otr_wan_")
        path, n = _wb.encode_frames_to_silent_mp4(frames, out_path, self.target_fps)
        # M7: PROVE the silent-clip color/stream contract on the emitted mp4
        # before canonicalize self-declares it -- ffprobe is the source of truth,
        # the hardcoded dict is not. Fail-LOUD NAMED on any drift.
        validate_silent_clip_contract(ffprobe_clip_fields(path), self.target_fps)
        if not os.environ.get("OTR_TEST_MODE"):
            post_mb = _MC.vram_used_mb() or 0
            _LOG.info("[OTR video] wan_i2v VRAM render-phase peak %s MB / post %s "
                      "MB (ceiling %s MB)", render_peak, post_mb,
                      _MC.dynamic_vram_ceiling_mb())
            _MC.assert_peak_within_ceiling(render_peak, "wan_i2v-render")
        return {"out_path": path, "frame_count": n}

    def canonicalize(self, raw, request, profile):
        return self._clip_from_raw(raw, request)


__all__ = ["WanI2VEngine"]
