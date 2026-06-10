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
"""
from __future__ import annotations

import os

from . import motion_common as _MC
from .registry import EngineUnusable, EngineUsabilityReason, register

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
_LTX_DEFAULT_W = 768
_LTX_DEFAULT_H = 512
_LTX_DEFAULT_NEGATIVE = (
    "low quality, worst quality, blurry, distorted, watermark, text, static")


@register
class LtxVideoEngine(_MC.MotionEngineBase):
    """The ltx_video text->video adapter (in-process, default-OFF / dark)."""

    name = "ltx_video"
    family = "text_to_video"
    # Generative motion b-roll / background / music visuals -- the roles whose
    # only required input is a text prompt. NOT a talking-head role (no lipsync).
    roles = ("scene_broll", "background_abstract", "music_visual")
    default_roles = ()
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
                "ltx_video disabled by %s=0" % self.requires_flag, kind="video")
        _MC.assert_sage_not_patched(self.name, self.family)   # BUG-070 (S5 gate)
        if not self._installed():
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "ltx_video not installed: checkpoint=%s (OTR_LTX_VIDEO_CKPT) + "
                "T5 encoder=%s (OTR_LTX_T5_ENCODER) -- both required for v0.9; "
                "install and verify on the GPU box"
                % (self._ckpt_path(), self._text_encoder_name()), kind="video")
        return self.name

    #: Terminal node of the graph (its IMAGE output is encoded to the clip).
    _TERMINAL = "vaedecode"

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
        steps = int(os.environ.get("OTR_LTX_STEPS", "30"))
        cfg = float(os.environ.get("OTR_LTX_CFG", "3.0"))
        positive = plan.get("text_prompt") or "a cinematic scene"
        negative = plan.get("negative_prompt") or os.environ.get(
            "OTR_LTX_NEGATIVE", _LTX_DEFAULT_NEGATIVE)
        return {
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
            "ksampler": {"class": "ksampler",
                         "inputs": {"seed": int(plan.get("seed", 0)), "steps": steps,
                                    "cfg": cfg, "sampler_name": "euler",
                                    "scheduler": "normal", "denoise": 1.0,
                                    "model": W("checkpoint", 0),
                                    "positive": W("cond", 0),
                                    "negative": W("cond", 1),
                                    "latent_image": W("latent", 0)}},
            "vaedecode": {"class": "vaedecode",
                          "inputs": {"samples": W("ksampler", 0),
                                     "vae": W("checkpoint", 2)}},
        }

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
        self._classes = _wb.resolve_graph_classes(self._node_candidates())
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
        classes = getattr(self, "_classes", None) \
            or _wb.resolve_graph_classes(self._node_candidates())
        width, height = self._dims(request)
        # LTX requires length == 8n+1 and dims multiple-of-32.
        length = max(_LTX_MIN_FRAMES,
                     int(plan["target_frame_count"] or self.target_fps))
        length = ((length - 1) // 8) * 8 + 1               # snap to 8n+1
        width = max(32, (width // 32) * 32)
        height = max(32, (height // 32) * 32)
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
            _MC.assert_vram_within_ceiling("ltx_video-render")  # PASS-PM mid-render
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


__all__ = ["LtxVideoEngine"]
