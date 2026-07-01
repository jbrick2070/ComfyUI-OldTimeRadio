"""still_parallax -- 2.5D depth-parallax over existing stills (0-E ticket 2).

The second 0-E easy on-ramp engine: ONE depth inference per still via
**DepthAnythingV2-SMALL** (the PINNED checkpoint: the Small model is
Apache-2.0; the larger DA-V2 Base/Large checkpoints are CC-BY-NC and are
BANNED from this engine -- license gate, 0-E spec), then a pure-numpy
depth-weighted pixel-shift camera sway rendered through the platform's
silent bt709 pipeline. Extends the ``static_motion`` cheap family
(still_motion's family): same canvas/timing/still resolution helpers,
same canonical-clip shape, plus a real depth model.

CPU-DEGRADABLE: the model is ~25M params and runs on CPU when no CUDA
device exists (slower, identical output contract). UNREGISTERED 2026-06-30
(item 2 rip-out, "registry IS the menu"): no longer imported by
nodes/_otr_video_engines/__init__.py and carries no ``@register`` decorator,
so it is NOT selectable and direct imports do not re-register it. The class
stays on disk (it returns when re-enabled) and still serves the three
OTR_VideoDirector slots that can supply a still (announcer / music /
other-beats via character_video + scene_broll -- background_abstract only
supplies text, so role_compat excludes it fail-closed) via DIRECT
instantiation.

HONEST LABEL: "2.5D" -- a single still warped by its depth map. No mesh,
no camera rig; the first REAL 3D object ships with ``mesh_stage``.

Cold-import clean (V-12): module scope imports only stdlib + the dep-free
registry + the cheap-family base. torch / transformers / PIL / numpy are
imported LAZILY inside load / render_clip.

Config (env): ``OTR_ENABLE_STILL_PARALLAX`` opt-in flag;
``OTR_DEPTH_ANYTHING_V2_DIR`` the local model dir (fail-closed when absent
-- offline-first, never a runtime download); ``OTR_STILL_PARALLAX_AMP_PX``
sway amplitude in px (default 14).
"""
from __future__ import annotations

import logging
import os

from .cheap_families import _CheapFamilyBase
from .registry import EngineUnusable, EngineUsabilityReason

_LOG = logging.getLogger("OTR.video.eng_still_parallax")

#: The pinned HF repo id (transformers port of DepthAnythingV2-SMALL).
#: Apache-2.0. Recorded in the registry row's model_requirements; the
#: bigger DA-V2 checkpoints (Base/Large) are CC-BY-NC -- do NOT swap them
#: in here without a license record (0-E E-7 discipline).
_DA2_SMALL_REPO = "depth-anything/Depth-Anything-V2-Small-hf"
_DA2_CACHE_DIRNAME = "models--depth-anything--Depth-Anything-V2-Small-hf"


def resolve_model_src(path):
    """Resolve a directory for ``from_pretrained(local_files_only=True)``.

    A plain local model dir (``config.json`` beside the weights) passes
    through unchanged. An HF CACHE repo dir (``models--*``) resolves to its
    newest ``snapshots/<rev>`` that contains ``config.json`` --
    ``from_pretrained`` CANNOT take the cache repo dir itself (proven live
    2026-06-11, the A-lane fix: repo dir raises OSError, snapshot loads).
    Pure stdlib; never imports transformers."""
    snaps = os.path.join(path, "snapshots")
    if not os.path.isdir(snaps):
        return path
    best = None
    best_mtime = None
    for name in sorted(os.listdir(snaps)):
        cand = os.path.join(snaps, name)
        if not os.path.isfile(os.path.join(cand, "config.json")):
            continue
        mtime = os.path.getmtime(cand)
        if best is None or mtime > best_mtime:
            best, best_mtime = cand, mtime
    return best or path

_AMP_DEFAULT = 14.0


def _amp_px():
    """Sway amplitude in pixels (env-overridable, clamped >= 0; LOUD on a
    malformed value). Pure given the env."""
    raw = os.environ.get("OTR_STILL_PARALLAX_AMP_PX")
    if raw is None or not str(raw).strip():
        return float(_AMP_DEFAULT)
    try:
        val = float(str(raw).strip())
    except (TypeError, ValueError):
        _LOG.warning("[eng_still_parallax] invalid OTR_STILL_PARALLAX_AMP_PX"
                     "=%r -- using default %.1f", raw, _AMP_DEFAULT)
        return float(_AMP_DEFAULT)
    if val < 0.0:
        _LOG.warning("[eng_still_parallax] OTR_STILL_PARALLAX_AMP_PX=%.2f "
                     "below 0 -- clamping to 0", val)
        return 0.0
    return val


# --------------------------------------------------------------------------- #
# Pure helpers (CPU-tested; numpy imported lazily, no model required)
# --------------------------------------------------------------------------- #
def normalize_depth(depth):
    """Normalize a raw relative-depth array to float32 in [0, 1] (1 = near).
    A constant map normalizes to all zeros (flat scene -> no parallax)."""
    import numpy as np
    d = np.asarray(depth, dtype="float32")
    lo = float(d.min())
    hi = float(d.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(d, dtype="float32")
    return ((d - lo) / (hi - lo)).astype("float32")


def parallax_offsets(n, amp):
    """Per-frame (dx, dy) sway offsets: one full horizontal sine cycle plus a
    quarter-amplitude double-rate vertical bob (a gentle figure-8). Starts AND
    ends at (0, 0) so frame 0 equals the source still and the clip loops
    seamlessly. Deterministic -- pure trig of the frame index, NO RNG (the
    fixed-seed trap never enters; determinism is structural)."""
    import math
    n = max(1, int(n))
    span = float(max(1, n - 1))
    out = []
    for t in range(n):
        phase = t / span
        dx = float(amp) * math.sin(2.0 * math.pi * phase)
        dy = float(amp) * 0.25 * math.sin(4.0 * math.pi * phase)
        out.append((dx, dy))
    return out


def synth_parallax_frames(img, depth01, n, amp):
    """The 2.5D warp: gather each output pixel from a depth-weighted source
    offset (near pixels sway by +amp, far pixels by -amp; backward warp with
    edge clamp). ``img`` is uint8 (H,W,3); ``depth01`` float in [0,1] shaped
    (H,W). Returns uint8 (n,H,W,3). Pure numpy, deterministic."""
    import numpy as np
    img = np.ascontiguousarray(np.asarray(img, dtype="uint8"))
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError("expected (H,W,3) uint8 still, got %r" % (img.shape,))
    d = np.asarray(depth01, dtype="float32")
    if d.shape != img.shape[:2]:
        raise ValueError("depth shape %r does not match still %r"
                         % (d.shape, img.shape[:2]))
    h, w = d.shape
    off = (d - 0.5) * 2.0                      # [-1, 1]: far -> -1, near -> +1
    yy, xx = np.meshgrid(np.arange(h, dtype="float32"),
                         np.arange(w, dtype="float32"), indexing="ij")
    frames = np.empty((max(1, int(n)), h, w, 3), dtype="uint8")
    for i, (dx, dy) in enumerate(parallax_offsets(n, amp)):
        x_src = np.clip(np.rint(xx - dx * off), 0, w - 1).astype("int32")
        y_src = np.clip(np.rint(yy - dy * off), 0, h - 1).astype("int32")
        frames[i] = img[y_src, x_src]
    return frames


def fit_cover_box(src_w, src_h, dst_w, dst_h):
    """ONE uniform scale to COVER the canvas + the centered crop box (never a
    stretch -- the N9 invariant the cheap floor also honors). Returns
    (scaled_w, scaled_h, left, top). Pure."""
    src_w, src_h = max(1, int(src_w)), max(1, int(src_h))
    dst_w, dst_h = max(1, int(dst_w)), max(1, int(dst_h))
    scale = max(dst_w / src_w, dst_h / src_h)
    scaled_w = max(dst_w, int(round(src_w * scale)))
    scaled_h = max(dst_h, int(round(src_h * scale)))
    left = (scaled_w - dst_w) // 2
    top = (scaled_h - dst_h) // 2
    return scaled_w, scaled_h, left, top


class StillParallaxEngine(_CheapFamilyBase):
    """``still_parallax`` -- depth-parallax static_motion peer (0-E ticket 2)."""

    name = "still_parallax"
    honest_label = ("2.5D depth parallax (DepthAnythingV2-SMALL over a still "
                    "-- not real 3D)")
    family = "static_motion"
    roles = ("announcer_visual", "music_visual", "character_video",
             "scene_broll")
    #: NEVER a default until the operator's look-QA promotes it (0-E gate).
    default_roles = ()
    requires_flag = None  # vestigial (registry IS the menu; no flag gate)
    required_inputs = ("init_image",)   # a REAL still is the whole point
    commercial_clean = True             # Apache-2.0 SMALL ckpt (pinned above)
    engine_version = "1"
    uses_still = True
    #: Declared LOUD fallback (the chain would universal-floor here anyway;
    #: declaring it keeps the degradation explicit + ledger-stampable).
    fallback_engine = "still_motion"

    def __init__(self):
        self._model = None
        self._processor = None

    # ---- config resolution (env override -> HF-cache default) ----
    def _model_dir(self):
        explicit = os.environ.get("OTR_DEPTH_ANYTHING_V2_DIR")
        if explicit:
            return explicit
        hf_home = os.environ.get("HF_HOME", "")
        if hf_home:
            return os.path.join(hf_home, "hub", _DA2_CACHE_DIRNAME)
        return _DA2_CACHE_DIRNAME          # relative -> exists() False -> LOUD

    def _installed(self):
        """True iff a LOADABLE DA-V2-SMALL snapshot exists locally (cheap,
        stdlib-only: resolves the HF-cache snapshot and requires its
        config.json). Offline-first: a missing model FAILS CLOSED -- this
        engine never downloads at render time."""
        src = resolve_model_src(self._model_dir())
        return os.path.isdir(src) and os.path.isfile(
            os.path.join(src, "config.json"))

    # ---- usability (fail-closed BEFORE any forward; no heavy import) ----
    def assert_usable(self, host_caps, profile, request_template=None):
        """Fail closed on the pinned local model: it must exist on disk."""
        if not self._installed():
            raise EngineUnusable(
                self.name, self.family, EngineUsabilityReason.MISSING_MODEL,
                "%s model missing: DepthAnythingV2-SMALL not found at %s "
                "(set OTR_DEPTH_ANYTHING_V2_DIR, or pre-fetch '%s' into the "
                "HF cache; offline-first -- no runtime download)"
                % (self.name, self._model_dir(), _DA2_SMALL_REPO),
                kind="video")
        return self.name

    # ---- residency (lazy transformers; CPU-degradable) ----
    def load(self):
        """Load the pinned DA-V2-SMALL (local files only). CUDA when present,
        else CPU -- identical output contract either way."""
        if self._model is not None:
            return
        if not self._installed():
            raise RuntimeError(
                "still_parallax model missing at %s -- see assert_usable"
                % self._model_dir())
        from transformers import (AutoImageProcessor,
                                  AutoModelForDepthEstimation)
        import torch
        # HF-cache repo dirs resolve to their newest snapshot (A-lane fix).
        src = resolve_model_src(self._model_dir())
        self._processor = AutoImageProcessor.from_pretrained(
            src, local_files_only=True)
        model = AutoModelForDepthEstimation.from_pretrained(
            src, local_files_only=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._model = model.to(device).eval()
        _LOG.info("[eng_still_parallax] DA-V2-SMALL loaded on %s", device)

    def unload(self):
        self._model = None
        self._processor = None

    def _depth_map(self, pil_img):
        """ONE deterministic depth inference (eval mode, no sampling) ->
        float32 (H,W) in [0,1], 1 = near. Resized to the still's size."""
        import torch
        self.load()
        inputs = self._processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
        pred = out.predicted_depth
        if pred.ndim == 3:                  # (B,H,W) -> (B,1,H,W)
            pred = pred.unsqueeze(1)
        pred = torch.nn.functional.interpolate(
            pred, size=(pil_img.height, pil_img.width),
            mode="bicubic", align_corners=False)
        return normalize_depth(pred[0, 0].detach().cpu().numpy())

    # ---- render (the 0-E parallax sweep; LOUD on a missing still) ----
    def render_clip(self, request, prepared=None):
        """Depth-parallax sweep over the request's still -> a silent bt709
        clip. A MISSING still raises FileNotFoundError LOUDLY (classified
        DEPENDENCY_MISSING -> the chain falls back; this engine never renders
        a slate floor under the parallax stamp -- honest engines only)."""
        from . import wrapper_bridge as _wb       # lazy: cold-import clean
        import numpy as np
        from ._tmp import otr_engine_tmp_mp4
        from PIL import Image
        w, h, fps = self._canvas_dims(request)
        n = self._frame_count(request, fps)
        still = self._still_path(request)
        if not still or not os.path.exists(still):
            raise FileNotFoundError(
                "still_parallax requires an on-disk still (asset_refs still/"
                "init_image/image); got %r -- falling back is the chain's "
                "job, not this engine's" % (still,))
        img = Image.open(still).convert("RGB")
        scaled_w, scaled_h, left, top = fit_cover_box(
            img.width, img.height, w, h)
        img = img.resize((scaled_w, scaled_h), Image.LANCZOS).crop(
            (left, top, left + w, top + h))
        depth01 = self._depth_map(img)
        frames = synth_parallax_frames(
            np.asarray(img, dtype="uint8"), depth01, n, _amp_px())
        out_path = otr_engine_tmp_mp4("otr_parallax_")
        path, n_out = _wb.encode_frames_to_silent_mp4(frames, out_path, fps)
        return self._floor_clip(request, path, fps, n_out)


__all__ = ["StillParallaxEngine", "normalize_depth", "parallax_offsets",
           "synth_parallax_frames", "fit_cover_box"]
