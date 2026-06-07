"""Wan 2.2 image->video motion adapter (A-S5 / CW-6) -- in-process, default-OFF.

Wan i2v animates a still init image into motion (a two-expert HIGH/LOW MoE + an
ordered LoRA stack + a ModelSamplingSD3 sigma split -- all adapter-internal
profile data, never request fields, V-11). Like ltx_video it runs IN-PROCESS in
the main cu130 venv by default; but Wan is a dependency-contamination suspect
(its wrapper / KJNodes can pull SageAttention), so its isolation is
``sidecar_optional``: ``resolve_isolation`` escalates it to a cu128 sidecar when
SageAttention is resident (BUG-070). Registered DEFAULT-OFF / dark (empty
``default_roles`` + gated behind ``OTR_ENABLE_WAN_I2V``); fails CLOSED until the
operator enables it AND the wrapper + checkpoints are installed and the KJNodes
pin audit (dep-pilot gate F) clears on the GPU box.

init_image aspect: a portrait init into a landscape canvas is padded / cropped
with ONE uniform scale (``motion_common.resolve_aspect_transform``) -- NEVER a
silent stretch (pre-mortem N9). Import-time is cold-import clean (V-12); the Wan
wrapper import + sampling is the CW-6 GPU-smoke slice. UTF-8, no BOM, ASCII-only.

Config (env): ``OTR_ENABLE_WAN_I2V`` opt-in flag; ``OTR_WAN_I2V_CKPT`` the
primary checkpoint path the load probe checks (verify-at-build).
"""
from __future__ import annotations

import os

from . import motion_common as _MC
from .registry import EngineUnusable, EngineUsabilityReason, register

_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
_COMFY_ROOT = os.path.dirname(os.path.dirname(_REPO_ROOT))


@register
class WanI2VEngine(_MC.MotionEngineBase):
    """The wan_i2v image->video adapter (in-process default; sidecar_optional)."""

    name = "wan_i2v"
    family = "image_to_video"
    # Animate a still into motion -- the roles that can supply an init image
    # (scene b-roll, music visual, character). background_abstract supplies only
    # text, so role_compat excludes it fail-closed.
    roles = ("scene_broll", "music_visual", "character_video")
    default_roles = ()
    required_inputs = ("init_image",)
    commercial_clean = False            # license is profile data; verify-at-build
    requires_flag = "OTR_ENABLE_WAN_I2V"
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
    def assert_usable(self, host_caps, profile, request_template=None):
        """Fail closed before any forward: the opt-in flag, then checkpoint
        presence. Wan TOLERATES Sage by escaping to a sidecar
        (``resolve_isolation``), so it does NOT hard-fail on Sage the way
        ltx_video does."""
        if os.getenv(self.requires_flag, "0") != "1":
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.GATED_BY_FLAG,
                "wan_i2v is opt-in; set %s=1 and install the Wan wrapper + "
                "checkpoints (clear the KJNodes pin audit)" % self.requires_flag,
                kind="video")
        if not self._installed():
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "wan_i2v checkpoint not found at %s; install the Wan wrapper + "
                "ckpt and verify on the GPU box (set OTR_WAN_I2V_CKPT)"
                % self._ckpt_path(), kind="video")
        return self.name

    # ---- residency (the heavy import + load is the CW-6 GPU-smoke slice) ----
    def load(self):
        if not self._installed():
            raise RuntimeError(
                "wan_i2v not installed: checkpoint missing at %s -- install the "
                "Wan wrapper + ckpt, set OTR_ENABLE_WAN_I2V=1, and run the CW-6 "
                "GPU smoke" % self._ckpt_path())
        # GPU-smoke slice: lazily import the installed Wan wrapper, build the
        # two-expert MoE + ordered LoRA stack + ModelSamplingSD3 sigma split,
        # track patchers for the V-4 teardown, set _loaded.
        raise NotImplementedError(
            "wan_i2v in-process load is the CW-6 GPU-smoke render slice; confirm "
            "the installed Wan wrapper expert-pair / LoRA order / sigma split and "
            "the KJNodes pin audit on sm_120 before enabling")

    def render_clip(self, request, prepared):
        plan = self._build_render_request(request)            # pure, CPU-tested
        raise NotImplementedError(
            "wan_i2v.render_clip is the CW-6 GPU-smoke slice (in-process Wan "
            "wrapper node-class forward); built request keys: %s" % sorted(plan))

    def canonicalize(self, raw, request, profile):
        return self._clip_from_raw(raw, request)

    # ---- pure helpers (CPU-testable; no wrapper, no heavy import) ----
    def _init_image_ref(self, request):
        """The init image path from ``asset_refs{init_image}`` (or "")."""
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        assets = get("asset_refs") or {}
        if isinstance(assets, dict):
            return assets.get("init_image") or ""
        return ""

    def _aspect_plan(self, request):
        """The pad / crop / fit transform mapping the init image into the canvas
        with ONE uniform scale (never a stretch, pre-mortem N9). Returns ``None``
        when the canvas or init dims are absent (the GPU smoke probes the real
        init dims), but still validates the policy token fail-closed."""
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
        """Pure: the normalized inference request the Wan wrapper consumes.
        Deterministic (seed + aspect plan flow straight through) -- the
        render-twice determinism contract (V-7)."""
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
            "fps": int(self.target_fps),
            "target_frame_count": int(t_get("target_frame_count", 0) or 0),
            "seed": int(s_get("request_seed", 0) or 0),
            "aspect_plan": self._aspect_plan(request),
        }

    def _clip_from_raw(self, raw, request):
        get = request.get if isinstance(request, dict) else (
            lambda k, d=None: getattr(request, k, d))
        raw = raw or {}
        return {
            "clip_id": get("shot_id") or get("request_id") or "wan_i2v_clip",
            "type": "video", "path": raw.get("out_path", ""),
            "container": "mp4", "codec": "h264", "pixel_format": "yuv420p",
            "fps": int(self.target_fps),
            "frame_count": int(raw.get("frame_count", 0) or 0),
            "has_audio": False,            # V-1: only OTR_MasterAudioMux emits audio
            "color_primaries": "bt709", "transfer": "bt709", "matrix": "bt709",
            "engine_id": self.name, "family": self.family,
        }


__all__ = ["WanI2VEngine"]
