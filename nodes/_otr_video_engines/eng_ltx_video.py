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


@register
class LtxVideoEngine(_MC.MotionEngineBase):
    """The ltx_video text->video adapter (in-process, default-OFF / dark)."""

    name = "ltx_video"
    family = "text_to_video"
    # Generative motion b-roll / background / music visuals -- the roles whose
    # only required input is a text prompt. NOT a talking-head role (no lipsync).
    roles = ("scene_broll", "background_abstract", "music_visual")
    default_roles = ()
    required_inputs = ("text_prompt",)
    commercial_clean = False            # license is profile data; verify-at-build
    requires_flag = "OTR_ENABLE_LTX_VIDEO"
    engine_version = "1"
    declared_isolation = _MC.ISOLATION_IN_PROCESS
    target_fps = 25

    # ---- config resolution (env override -> box default) ----
    def _ckpt_path(self):
        return os.environ.get("OTR_LTX_VIDEO_CKPT") or os.path.join(
            _COMFY_ROOT, "models", "checkpoints", "ltx-video-2b.safetensors")

    def _installed(self):
        """True iff the primary checkpoint exists on disk (no import -- cheap,
        headless-safe). The full wrapper INPUT_TYPES check is the GPU smoke."""
        return os.path.exists(self._ckpt_path())

    # ---- usability (fail-closed BEFORE any forward; no heavy import) ----
    def assert_usable(self, host_caps, profile, request_template=None):
        """Fail closed before any forward: the opt-in flag, then the BUG-070
        SageAttention gate, then checkpoint presence (verify-at-build). Imports
        nothing heavy -- runs at lock/validate time on the CPU box."""
        if os.getenv(self.requires_flag, "0") != "1":
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.GATED_BY_FLAG,
                "ltx_video is opt-in; set %s=1 and install the LTX wrapper + "
                "checkpoints" % self.requires_flag, kind="video")
        _MC.assert_sage_not_patched(self.name, self.family)   # BUG-070 (S5 gate)
        if not self._installed():
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "ltx_video checkpoint not found at %s; install the LTX wrapper + "
                "ckpt and verify on the GPU box (set OTR_LTX_VIDEO_CKPT)"
                % self._ckpt_path(), kind="video")
        return self.name

    # ---- residency (the heavy import + load is the CW-6 GPU-smoke slice) ----
    def load(self):
        """Fail CLOSED until installed; the real wrapper import + model load is
        the CW-6 GPU-smoke slice (lazy, never at module scope)."""
        if not self._installed():
            raise RuntimeError(
                "ltx_video not installed: checkpoint missing at %s -- install the "
                "LTX wrapper + ckpt, set OTR_ENABLE_LTX_VIDEO=1, and run the CW-6 "
                "GPU smoke" % self._ckpt_path())
        # GPU-smoke slice: lazily import the installed LTX wrapper, build the
        # model handles (MODEL+CLIP+VAE: ltx-video-2b + gemma encoder + LTX 2.3
        # distilled LoRA @0.7), track patchers for the V-4 teardown, set _loaded.
        raise NotImplementedError(
            "ltx_video in-process load is the CW-6 GPU-smoke render slice; "
            "confirm the installed LTX wrapper INPUT_TYPES and SageAttention-clean "
            "determinism on sm_120 before enabling")

    def render_clip(self, request, prepared):
        """Drive ONE text->video clip via the in-process LTX wrapper. The pure
        request build is CPU-tested; the wrapper forward is the GPU-smoke slice."""
        plan = self._build_render_request(request)            # pure, CPU-tested
        raise NotImplementedError(
            "ltx_video.render_clip is the CW-6 GPU-smoke slice (in-process LTX "
            "wrapper node-class forward); built request keys: %s" % sorted(plan))

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
