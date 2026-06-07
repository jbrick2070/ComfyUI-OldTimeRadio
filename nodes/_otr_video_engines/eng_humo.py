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

    # ---- residency (the heavy import + load is the A-S6 GPU-smoke slice) ----
    def load(self):
        """Fail CLOSED until installed; the real HuMo wrapper import + the
        MODEL+CLIP+VAE+AUDIO_ENCODER load (umt5_xxl CLIP, wan_2.1 VAE,
        whisper_large_v3 audio encoder + the HuMo diffusion model / LoRA tier) is
        the A-S6 GPU-smoke slice (lazy via comfy.model_management, never at module
        scope)."""
        if not self._installed():
            raise RuntimeError(
                "humo not installed: checkpoint missing at %s -- install the HuMo "
                "wrapper + ckpt, set OTR_ENABLE_HUMO=1, and run the A-S6 GPU "
                "smoke" % self._ckpt_path())
        # GPU-smoke slice: lazily import the installed HuMo wrapper, build the
        # MODEL+CLIP+VAE+AUDIO_ENCODER handles internally via comfy.model_
        # management, track patchers for the V-4 teardown, set _loaded.
        raise NotImplementedError(
            "humo in-process load is the A-S6 GPU-smoke render slice; confirm the "
            "installed HuMo wrapper INPUT_TYPES + the MODEL/CLIP/VAE/AUDIO_ENCODER "
            "handles and the VRAM<=14.5 GB peak on sm_120 before enabling")

    def render_clip(self, request, prepared):
        """Drive ONE audio-driven-face clip via the in-process HuMo wrapper. The
        pure request build is CPU-tested; the wrapper forward is the GPU-smoke
        slice."""
        plan = self._build_render_request(request)            # pure, CPU-tested
        raise NotImplementedError(
            "humo.render_clip is the A-S6 GPU-smoke slice (in-process HuMo wrapper "
            "node-class forward); built request keys: %s" % sorted(plan))

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
