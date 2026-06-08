"""HuMo audio-driven-face motion adapter (A-S6 / CW-7) -- in-process, default-OFF.

HuMo is OTR's heaviest engine: an audio-conditioned image-to-video model that
animates a reference PORTRAIT (init_image) in sync with a speech AUDIO reference
into a talking-character clip. It is the ``audio_driven_face`` family -- the
talking-head path for the announcer + character roles. Like ltx_video / wan_i2v
it runs IN-PROCESS in the main ComfyUI cu130 venv (it loads
MODEL+CLIP+VAE+AUDIO_ENCODER internally via ``comfy.model_management``) and is the
SINGLE resident heavy engine while it holds the AS-3 lease. Native output is
480x832 @ 25 fps; a portrait init is fit to the canvas with ONE uniform scale,
never stretched, and the compositor pillarboxes the portrait clip (pre-mortem N9).

Registered DEFAULT-OFF / dark (empty ``default_roles`` + gated behind
``OTR_ENABLE_HUMO``) so it shows in the static per-role dropdown (V-6) but is
never a default and fails CLOSED until the operator enables it AND the HuMo
wrapper + checkpoints are installed and verified on the GPU box (the A-S6 smoke).
No model is "primary" -- HuMo is one peer adapter among the motion engines.

Fallback: a render-time failure degrades HuMo to its ``fallback_engine``
(``latentsync``), which degrades to the zero-VRAM ``still_kenburns`` radio floor;
the chain ``humo -> latentsync -> still_kenburns`` is acyclic and terminates (see
``nodes/_otr_shared/fallback.py``). The audio that drives HuMo is the FROZEN
master; HuMo emits an ALWAYS-SILENT clip (``has_audio`` False) -- only
``OTR_MasterAudioMux`` ever adds audio (invariant V-1).

Cold-import clean (V-12): module scope imports only stdlib + the dep-free shared
helpers + the registry. torch / the HuMo wrapper are imported LAZILY in ``load``
/ ``render_clip`` (the GPU-smoke render slice), never here. UTF-8, no BOM,
ASCII-only source.

Config (env): ``OTR_ENABLE_HUMO`` opt-in flag; ``OTR_HUMO_CKPT`` the primary
checkpoint path the load probe checks (verify-at-build; the full multi-handle
MODEL+CLIP+VAE+AUDIO_ENCODER load is confirmed on the GPU box).
"""
from __future__ import annotations

import os

from . import motion_common as _MC
from .registry import EngineUnusable, EngineUsabilityReason, register

_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
_COMFY_ROOT = os.path.dirname(os.path.dirname(_REPO_ROOT))

# A-ship in-process forward (the native graph PROVEN in scripts/render_humo_batch.py
# build_humo_prompt, verified on this RTX 5080 at 480x832 fp8). The graph TOPOLOGY +
# widget values are legacy-proven; the exact checkpoint / encoder FILENAMES are
# env-overridable (VERIFY-ON-GPU) so the operator confirms them against the
# installed models without editing code. Native portrait 480x832 @ 25 fps; the Wan
# 2.1 VAE 4n+1 length rule is enforced via wrapper_bridge.quantize_frames_4n1.
_HUMO_MIN_FRAMES = 33          # below this has hung this hardware (legacy floor)
_HUMO_MAX_FRAMES = 177         # last empirically verified ceiling at 480x832 fp8
_HUMO_NATIVE_W = 480
_HUMO_NATIVE_H = 832
# An ASCII negative (CLAUDE.md: ASCII-only source). HuMo's best negative is the
# ByteDance Chinese default; set OTR_HUMO_NEGATIVE to it on the box to match the
# legacy template exactly.
_HUMO_DEFAULT_NEGATIVE = (
    "low quality, worst quality, blurry, jpeg artifacts, distorted, deformed, "
    "extra fingers, bad hands, bad face, static, watermark, text")


@register
class HuMoEngine(_MC.MotionEngineBase):
    """The humo audio-driven-face adapter (in-process, default-OFF / dark)."""

    name = "humo"
    family = "audio_driven_face"
    # Talking-head roles only: HuMo needs BOTH a portrait (init_image) AND a
    # speech audio_ref, which only the announcer + character roles supply.
    # role_compat excludes the audio-less roles (music / scene / background)
    # fail-closed.
    roles = ("announcer_visual", "character_video")
    default_roles = ()
    required_inputs = ("audio_ref", "init_image")
    commercial_clean = False            # license is profile data; verify-at-build
    requires_flag = "OTR_ENABLE_HUMO"
    engine_version = "1"
    declared_isolation = _MC.ISOLATION_IN_PROCESS
    target_fps = 25
    #: Family-degradation next hop. A render-time failure falls here, then on to
    #: the radio floor: humo -> latentsync -> still_kenburns (see
    #: nodes/_otr_shared/fallback.py). One single-linked hop per engine.
    fallback_engine = "latentsync"

    # ---- config resolution (env override -> box default) ----
    def _ckpt_path(self):
        return os.environ.get("OTR_HUMO_CKPT") or os.path.join(
            _COMFY_ROOT, "models", "diffusion_models", "humo",
            "humo_1.7B.safetensors")

    def _installed(self):
        """True iff the primary checkpoint exists on disk (no import -- cheap,
        headless-safe). The full MODEL+CLIP+VAE+AUDIO_ENCODER multi-handle load
        (+ the low/high/gguf tier pick) is the GPU smoke."""
        return os.path.exists(self._ckpt_path())

    # ---- usability (fail-closed BEFORE any forward; no heavy import) ----
    def assert_usable(self, host_caps, profile, request_template=None):
        """Fail closed before any forward: the opt-in flag, then checkpoint
        presence (verify-at-build). Imports nothing heavy -- runs at
        lock/validate time on the CPU box. HuMo loads in-process; its
        SageAttention tolerance is a GPU-smoke verify item, NOT a hard CPU gate
        (unlike ltx_video's BUG-070 int8-PV abort)."""
        if os.getenv(self.requires_flag, "0") != "1":
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.GATED_BY_FLAG,
                "humo is opt-in; set %s=1 and install the HuMo wrapper + "
                "checkpoints" % self.requires_flag, kind="video")
        if not self._installed():
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "humo checkpoint not found at %s; install the HuMo wrapper + "
                "ckpt and verify on the GPU box (set OTR_HUMO_CKPT)"
                % self._ckpt_path(), kind="video")
        return self.name

    #: Terminal node of the graph (its IMAGE output is encoded to the clip).
    _TERMINAL = "vaedecode"

    # ---- in-process graph spec (proven topology; filenames VERIFY-ON-GPU) ----
    def _node_candidates(self):
        """Ordered ComfyUI node-class candidates per graph node (all core /
        comfy_extras classes used by the proven legacy HuMo graph)."""
        return {
            "unet": ("UNETLoader",),
            "lora": ("LoraLoaderModelOnly",),
            "modelsampling": ("ModelSamplingSD3",),
            "clip": ("CLIPLoader",),
            "pos": ("CLIPTextEncode",),
            "neg": ("CLIPTextEncode",),
            "vae": ("VAELoader",),
            "loadaudio": ("LoadAudio",),
            "audioenc_loader": ("AudioEncoderLoader",),
            "audioenc": ("AudioEncoderEncode",),
            "loadimage": ("LoadImage",),
            "humo": ("WanHuMoImageToVideo",),
            "ksampler": ("KSampler",),
            "vaedecode": ("VAEDecode",),
        }

    def _loader_names(self):
        """Model / encoder FILENAMES the loader nodes consume. Defaults are the
        legacy-proven names; each is env-overridable so the operator points at the
        installed files without editing code (VERIFY-ON-GPU)."""
        return {
            "unet": os.environ.get("OTR_HUMO_UNET_NAME")
            or os.path.basename(self._ckpt_path()),
            "lora": os.environ.get(
                "OTR_HUMO_LORA_NAME",
                "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"),
            "clip": os.environ.get(
                "OTR_HUMO_CLIP_NAME", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
            "vae": os.environ.get("OTR_HUMO_VAE_NAME", "wan_2.1_vae.safetensors"),
            "whisper": os.environ.get(
                "OTR_HUMO_AUDIO_ENCODER_NAME", "whisper_large_v3_fp16.safetensors"),
        }

    def _native_dims(self):
        """HuMo's native render size (portrait 480x832; env-overridable). The
        compositor pillarboxes the portrait into the 16:9 canvas (memory: keep
        HuMo, accept the pillarbox) -- N9 no-stretch is preserved upstream."""
        w = int(os.environ.get("OTR_HUMO_WIDTH", _HUMO_NATIVE_W))
        h = int(os.environ.get("OTR_HUMO_HEIGHT", _HUMO_NATIVE_H))
        return w, h

    def _build_graph(self, image_name, audio_name, plan, length, width, height):
        """The declarative HuMo graph (wrapper_bridge.run_graph format). Mirrors
        the proven build_humo_prompt wiring node-for-node:
        UNETLoader->LoRA->ModelSamplingSD3->KSampler (model); CLIPLoader->pos/neg;
        VAELoader; LoadAudio->AudioEncoderEncode(+AudioEncoderLoader);
        LoadImage->ref_image; WanHuMoImageToVideo->KSampler->VAEDecode."""
        from . import wrapper_bridge as _wb
        names = self._loader_names()
        steps = int(os.environ.get("OTR_HUMO_STEPS", "6"))
        cfg = float(os.environ.get("OTR_HUMO_CFG", "1.0"))
        positive = plan.get("text_prompt") or "a person speaking, subtle facial motion"
        negative = os.environ.get("OTR_HUMO_NEGATIVE", _HUMO_DEFAULT_NEGATIVE)
        W = _wb.Wire
        # The lightx2v distill LoRA is a 14B-shaped adapter: it is INCOMPATIBLE
        # with the 1.7B tier (shape-mismatch -> not merged + wasted VRAM). Make
        # it optional so the 1.7B tier runs LoRA-free (set OTR_HUMO_LORA_NAME to
        # none/skip and raise OTR_HUMO_STEPS, since the distill shortcut is gone).
        lora_name = names["lora"]
        skip_lora = (not lora_name) or str(lora_name).strip().lower() in (
            "none", "skip", "off")
        graph = {
            "unet": {"class": "unet",
                     "inputs": {"unet_name": names["unet"],
                                "weight_dtype": "default"}},
            "clip": {"class": "clip",
                     "inputs": {"clip_name": names["clip"], "type": "wan",
                                "device": "default"}},
            "pos": {"class": "pos",
                    "inputs": {"text": positive, "clip": W("clip", 0)}},
            "neg": {"class": "neg",
                    "inputs": {"text": negative, "clip": W("clip", 0)}},
            "vae": {"class": "vae", "inputs": {"vae_name": names["vae"]}},
            "loadaudio": {"class": "loadaudio", "inputs": {"audio": audio_name}},
            "audioenc_loader": {"class": "audioenc_loader",
                                "inputs": {"audio_encoder_name": names["whisper"]}},
            "audioenc": {"class": "audioenc",
                         "inputs": {"audio_encoder": W("audioenc_loader", 0),
                                    "audio": W("loadaudio", 0)}},
            "loadimage": {"class": "loadimage", "inputs": {"image": image_name}},
            "humo": {"class": "humo",
                     "inputs": {"width": int(width), "height": int(height),
                                "length": int(length), "batch_size": 1,
                                "positive": W("pos", 0), "negative": W("neg", 0),
                                "vae": W("vae", 0),
                                "audio_encoder_output": W("audioenc", 0),
                                "ref_image": W("loadimage", 0)}},
        }
        model_src = "unet"
        if not skip_lora:
            graph["lora"] = {"class": "lora",
                             "inputs": {"lora_name": lora_name,
                                        "strength_model": 1.0,
                                        "model": W("unet", 0)}}
            model_src = "lora"
        graph["modelsampling"] = {
            "class": "modelsampling",
            "inputs": {"shift": 8.0, "model": W(model_src, 0)}}
        graph["ksampler"] = {
            "class": "ksampler",
            "inputs": {"seed": int(plan.get("seed", 0)), "steps": steps,
                       "cfg": cfg, "sampler_name": "uni_pc",
                       "scheduler": "simple", "denoise": 1.0,
                       "model": W("modelsampling", 0),
                       "positive": W("humo", 0), "negative": W("humo", 1),
                       "latent_image": W("humo", 2)}}
        graph["vaedecode"] = {
            "class": "vaedecode",
            "inputs": {"samples": W("ksampler", 0), "vae": W("vae", 0)}}
        return graph

    def _retain_model_patchers(self, results, prepared):
        """Best-effort V-4: keep the MODEL ModelPatchers the graph produced so
        teardown can detach(unpatch_all=True) them (the lease + VRAM settle-wait in
        MotionEngineBase.teardown is the real residency guard)."""
        bucket = prepared.setdefault("patchers", self._patchers) \
            if isinstance(prepared, dict) else self._patchers
        seen = {id(p) for p in bucket}
        for nid in ("unet", "lora", "modelsampling"):
            out = results.get(nid)
            if not out:
                continue
            obj = out[0]
            if id(obj) not in seen and callable(getattr(obj, "detach", None)):
                bucket.append(obj)
                seen.add(id(obj))

    # ---- residency (resolve the installed wrapper nodes; weights load on call) -
    def load(self):
        """Fail CLOSED until installed, then RESOLVE the installed ComfyUI wrapper
        node classes (fail-closed NAMED if absent). The heavy weight load happens
        when the loader nodes execute inside render_clip (ComfyUI's own model
        management), so load() stays cheap and the AS-3 lease brackets the real
        residency. The live VRAM<=14.5 GB peak + render-twice pixels are the A-S6
        GPU smoke (operator)."""
        if not self._installed():
            raise RuntimeError(
                "humo not installed: checkpoint missing at %s -- install the HuMo "
                "wrapper + ckpt, set OTR_ENABLE_HUMO=1, and run the A-S6 GPU "
                "smoke" % self._ckpt_path())
        from . import wrapper_bridge as _wb
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
        self._loaded = True

    def render_clip(self, request, prepared):
        """Drive ONE audio-driven-face clip via the in-process HuMo wrapper graph.

        Stages the portrait + speech into ComfyUI's input dir (the proven legacy
        LoadImage / LoadAudio pattern), executes the proven node graph in
        dependency order, encodes the decoded IMAGE batch to a SILENT bt709 clip
        (V-1), retains the MODEL patchers for V-4 teardown, and asserts the
        mid-render NVML ceiling. Returns the raw ``{out_path, frame_count}`` that
        canonicalize() normalises. Fail-closed NAMED if a wrapper node is missing
        or an input is absent."""
        from . import wrapper_bridge as _wb
        import tempfile
        plan = self._build_render_request(request)            # pure, CPU-tested
        if not plan["audio_path"] or not plan["init_image"]:
            raise _wb.GraphExecutionError(
                "humo requires both audio_ref and init_image (got audio=%r "
                "init_image=%r)" % (plan["audio_path"], plan["init_image"]))
        classes = getattr(self, "_classes", None) \
            or _wb.resolve_graph_classes(self._node_candidates())
        audio_name = _wb.stage_into_comfy_input(plan["audio_path"])
        image_name = _wb.stage_into_comfy_input(plan["init_image"])
        length = _wb.quantize_frames_4n1(
            plan["target_frame_count"] or self.target_fps,
            min_frames=_HUMO_MIN_FRAMES, max_frames=_HUMO_MAX_FRAMES)
        width, height = self._native_dims()
        graph = self._build_graph(image_name, audio_name, plan, length, width, height)
        # Render FULLY RESIDENT -- the proven BUG-265 low_vram_default path: the
        # HuMo-1.7B stack (3.3 GB + umt5/whisper) stays resident with zero offload
        # on a 16 GB card. (No free_after_use: forcing inter-node model eviction
        # only fragmented the allocator into an OOM -- it is NOT what the working
        # OTR_BatchHumoRender does.)
        results = _wb.run_graph(graph, classes)
        images = results[self._TERMINAL][0]                   # VAEDecode IMAGE batch
        self._retain_model_patchers(results, prepared)
        frames = _wb.images_to_uint8(images)
        out_path = tempfile.mktemp(suffix=".mp4", prefix="otr_humo_")
        path, n = _wb.encode_frames_to_silent_mp4(frames, out_path, self.target_fps)
        # Restore the proven HuMo VRAM discipline the refactor dropped: the frames
        # are on disk, so evict the umt5 CLIP + whisper encoders (BUG-291 detach
        # reclaim, the same the legacy free_otr_pipeline_residue used as
        # inter-phase cleanup) so the resident stack drops back under the ceiling
        # before the PASS-PM assert -- and so the next soak beat starts drained
        # (no cross-beat accumulation). LOUD; no unload_all_models (V-4/V-5).
        _wb.reclaim_idle_models(reason="humo post-decode")
        _MC.assert_vram_within_ceiling("humo-render")         # PASS-PM mid-render
        return {"out_path": path, "frame_count": n}

    def canonicalize(self, raw, request, profile):
        """Normalize a rendered clip into the ALWAYS-SILENT bt709 / yuv420p
        CanonicalClip contract (frame_count is the integer timing authority)."""
        return self._clip_from_raw(raw, request)

    # ---- pure helpers (CPU-testable; no wrapper, no heavy import) ----
    @staticmethod
    def _ref_path(ref):
        """Pull a filesystem path out of an audio_ref / init_image that may be a
        bare string OR a mapping carrying a ``path`` key (the schema AudioRef
        shape). Returns "" when nothing path-like is present."""
        if not ref:
            return ""
        if isinstance(ref, str):
            return ref
        if isinstance(ref, dict):
            return ref.get("path") or ""
        return getattr(ref, "path", "") or ""

    def _init_image_ref(self, request):
        """The portrait init image path from ``asset_refs{init_image}`` (or "")."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        assets = get("asset_refs") or {}
        if isinstance(assets, dict):
            return assets.get("init_image") or ""
        return ""

    def _aspect_plan(self, request):
        """The pad / crop / fit transform mapping the portrait init into the
        canvas with ONE uniform scale (never a stretch, pre-mortem N9). Returns
        ``None`` when the canvas or init dims are absent (the GPU smoke probes the
        real init dims), but still validates the policy token fail-closed."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        canvas = get("canvas") or {}
        c_get = canvas.get if isinstance(canvas, dict) else (
            lambda k, d=None: getattr(canvas, k, d))
        dst_w = int(c_get("w", 0) or 0)
        dst_h = int(c_get("h", 0) or 0)
        policy = (c_get("aspect_policy", _MC.DEFAULT_ASPECT_POLICY)
                  or _MC.DEFAULT_ASPECT_POLICY)
        src_w = int(get("init_w", 0) or 0)
        src_h = int(get("init_h", 0) or 0)
        if min(dst_w, dst_h, src_w, src_h) <= 0:
            _MC.assert_aspect_policy(policy)     # validate the token even unsized
            return None
        return _MC.resolve_aspect_transform(src_w, src_h, dst_w, dst_h, policy)

    def _build_render_request(self, request):
        """Pure: the normalized inference request the HuMo wrapper consumes, from
        a VideoRequest-shaped object OR a plain dict. Deterministic (seed + audio
        + init + aspect flow straight through) -- the render-twice determinism
        contract (V-7). The audio_ref is the FROZEN master that DRIVES the face;
        the output clip stays silent (V-1)."""
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
            "aspect_plan": self._aspect_plan(request),
        }

    def _clip_from_raw(self, raw, request):
        """Pure: shape a worker / wrapper result into the silent CanonicalClip
        dict (bt709 / yuv420p; frame_count is the integer timing authority)."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        raw = raw or {}
        return {
            "clip_id": get("shot_id") or get("request_id") or "humo_clip",
            "type": "video", "path": raw.get("out_path", ""),
            "container": "mp4", "codec": "h264", "pixel_format": "yuv420p",
            "fps": int(self.target_fps),
            "frame_count": int(raw.get("frame_count", 0) or 0),
            "has_audio": False,            # V-1: only OTR_MasterAudioMux emits audio
            "color_primaries": "bt709", "transfer": "bt709", "matrix": "bt709",
            "engine_id": self.name, "family": self.family,
        }


__all__ = ["HuMoEngine"]
