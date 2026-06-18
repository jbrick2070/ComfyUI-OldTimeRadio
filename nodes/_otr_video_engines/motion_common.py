"""Shared in-process motion-engine helpers (A-S5 / CW-6: LTX + Wan).

The motion video engines -- ltx_video (text->video), wan_i2v (image->video), and
humo (A-S6) -- run IN-PROCESS in the main ComfyUI cu130 / torch-2.10 venv: they
call the installed ComfyUI wrapper node classes directly (no GraphBuilder),
unlike a Path-B cu128 subprocess sidecar. This module factors the pieces
those in-process motion adapters share, so each adapter file stays small and
every guard is tested once:

* MotionEngineBase -- the AS-3 single-heavy-engine GPU residency lease on
  ``prepare`` plus a V-4 patcher-detach ``teardown`` that NEVER calls
  ``unload_all_models``;
* the BUG-070 SageAttention contamination gate -- int8-PV SageAttention
  process-aborts LTX with NO traceback, so ``ltx_video`` fails CLOSED before its
  first forward (``assert_sage_not_patched``) and ``wan_i2v`` is routed to a
  sidecar when Sage is resident (``resolve_isolation``);
* ``init_image`` aspect handling that maps a source image into the canvas with a
  SINGLE uniform scale (``resolve_aspect_transform`` / ``assert_no_silent_stretch``)
  so a portrait init never silently stretches into a landscape canvas
  (pre-mortem N9).

Cold-import clean (V-12): module scope imports only the stdlib + the dep-free
shared GPU lease + the dep-free registry error types. torch / diffusers / the LTX
/ Wan wrappers are imported LAZILY inside each adapter's ``load`` / ``render_clip``
(the GPU-smoke render slice), never here. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import os
import sys

from .._otr_shared import gpu_residency as _GR
from .registry import EngineUnusable, EngineUsabilityReason

#: Machine-wide VRAM ceiling for the single resident heavy engine (A invariant).
#: This constant is the 16 GB-box FALLBACK; runtime consumers read
#: :func:`dynamic_vram_ceiling_mb` so a profile-stamped run (GATE B S1/S2:
#: OTR_WorkflowValidator exports ``OTR_VRAM_CEILING_MB`` every execution, and
#: headless launchers set it directly) tightens the budget WITHOUT a code edit.
VRAM_CEILING_MB = 14500


def dynamic_vram_ceiling_mb() -> int:
    """The ACTIVE machine-wide VRAM ceiling, read at DISPATCH time (GATE B S1:
    env > the 14500 fallback -- no graph introspection). Invalid env values
    fall back LOUD-less to the constant (the validator already warned)."""
    raw = (os.environ.get("OTR_VRAM_CEILING_MB") or "").strip()
    if raw:
        try:
            val = int(raw)
            if val > 0:
                return val
        except ValueError:
            pass
    return VRAM_CEILING_MB

#: Aspect policies an init image may be fit into the canvas with (mirrors
#: schemas.Canvas.aspect_policy). Each uses ONE uniform scale, so the aspect ratio
#: is preserved; the forbidden behavior -- an implicit non-uniform stretch -- is
#: never emitted.
ASPECT_POLICIES = ("pad", "crop", "fit")
DEFAULT_ASPECT_POLICY = "pad"

#: Engine isolation tiers (schemas dependency_manifest.isolation).
ISOLATION_IN_PROCESS = "in_process"
ISOLATION_SIDECAR_REQUIRED = "sidecar_required"
ISOLATION_SIDECAR_OPTIONAL = "sidecar_optional"


# --------------------------------------------------------------------------- #
# BUG-070 SageAttention contamination gate
# --------------------------------------------------------------------------- #
def sageattention_patched(modules=None, env=None):
    """True if SageAttention is ACTIVE / forced on (not merely installed).

    2026-06-09 (capstone soak catch): current ComfyUI core imports
    ``sageattention`` UNCONDITIONALLY at ``comfy.ldm.modules.attention``
    import -- an availability probe that leaves the module in ``sys.modules``
    on EVERY boot when the pip package is installed, regardless of
    ``--use-sage-attention``. Module residency therefore no longer implies
    activation. Inside a live ComfyUI process the REAL activation switch is
    ``comfy.model_management.sage_attention_enabled()`` -- consult it.

    Precedence: the explicit operator override ``OTR_SAGEATTENTION_PATCHED=1``
    (a wrapper that monkeypatched comfy attention invisibly) -> the live
    comfy activation switch -> the ``sys.modules`` heuristic (non-comfy
    contexts, e.g. CPU tests, which inject ``modules``/``env``). Pure +
    side-effect free; never imports sageattention itself.
    """
    environ = os.environ if env is None else env
    if environ.get("OTR_SAGEATTENTION_PATCHED", "0") == "1":
        return True
    if modules is None and env is None:
        try:
            from comfy import model_management as _mm
            return bool(_mm.sage_attention_enabled())
        except Exception:  # noqa: BLE001 -- not inside ComfyUI; heuristic below
            pass
    mods = sys.modules if modules is None else modules
    return "sageattention" in mods


def assert_sage_not_patched(engine_name, family, *, modules=None, env=None):
    """Fail CLOSED (BUG-070) if SageAttention is patched/resident.

    int8-PV SageAttention process-aborts LTX-Video with NO traceback, so a motion
    engine that cannot tolerate it must refuse to run BEFORE the first forward.
    Raises :class:`EngineUnusable` (INCOMPATIBLE_PROFILE) when patched; returns
    ``engine_name`` when clear.
    """
    if sageattention_patched(modules=modules, env=env):
        raise EngineUnusable(
            engine_name, family, EngineUsabilityReason.INCOMPATIBLE_PROFILE,
            "SageAttention is patched/resident; %s refuses to run in-process "
            "(BUG-070: int8-PV SageAttention process-aborts with no traceback). "
            "Disable SageAttention (e.g. KJNodes) or run this engine in a cu128 "
            "sidecar" % engine_name,
            kind="video")
    return engine_name


def resolve_isolation(declared_isolation, sage_patched):
    """Resolve an engine's runtime isolation tier (pure).

    ``sidecar_optional`` (wan_i2v) escalates to ``sidecar_required`` when
    SageAttention is resident -- running in-process next to a Sage-patched
    attention is toxic (BUG-070). ``sidecar_required`` stays required; everything
    else runs ``in_process``.
    """
    if declared_isolation == ISOLATION_SIDECAR_REQUIRED:
        return ISOLATION_SIDECAR_REQUIRED
    if declared_isolation == ISOLATION_SIDECAR_OPTIONAL and sage_patched:
        return ISOLATION_SIDECAR_REQUIRED
    return ISOLATION_IN_PROCESS


# --------------------------------------------------------------------------- #
# init_image aspect handling -- no silent stretch (pre-mortem N9)
# --------------------------------------------------------------------------- #
def assert_aspect_policy(policy):
    """Validate an aspect policy; an unknown policy (which could imply an
    implicit stretch) is rejected fail-closed."""
    if policy not in ASPECT_POLICIES:
        raise ValueError(
            "aspect_policy %r not in %r (an implicit stretch is forbidden)"
            % (policy, ASPECT_POLICIES))
    return policy


def _even(value):
    """Round to the nearest even int (model-stride / yuv420p mod-2 safe)."""
    n = int(round(value))
    return n - (n % 2)


def resolve_aspect_transform(src_w, src_h, dst_w, dst_h,
                             policy=DEFAULT_ASPECT_POLICY):
    """Map a source init image into the dst canvas with ONE uniform scale.

    ``pad`` / ``fit`` scale to FIT inside the canvas (letterbox / pillarbox bars);
    ``crop`` scales to COVER the canvas (center-crop the overflow). Either way a
    single scalar ``scale`` is applied to both axes, so the aspect ratio is
    preserved and the result is NEVER an implicit stretch. Returns a plan dict
    (even ``scaled_w`` / ``scaled_h``, ``pad_x`` / ``pad_y`` or ``crop_x`` /
    ``crop_y``, ``scale``, ``policy``). Raises on a non-positive dimension or an
    unknown policy.
    """
    assert_aspect_policy(policy)
    for label, value in (("src_w", src_w), ("src_h", src_h),
                         ("dst_w", dst_w), ("dst_h", dst_h)):
        if int(value) <= 0:
            raise ValueError("%s must be positive, got %r" % (label, value))
    sw, sh, dw, dh = int(src_w), int(src_h), int(dst_w), int(dst_h)
    if policy == "crop":
        scale = max(dw / sw, dh / sh)
    else:                                   # pad | fit -> fit inside the canvas
        scale = min(dw / sw, dh / sh)
    scaled_w, scaled_h = _even(sw * scale), _even(sh * scale)
    plan = {
        "policy": policy, "scale": scale,
        "src_w": sw, "src_h": sh, "dst_w": dw, "dst_h": dh,
        "scaled_w": scaled_w, "scaled_h": scaled_h,
        "pad_x": max(0, dw - scaled_w) // 2, "pad_y": max(0, dh - scaled_h) // 2,
        "crop_x": max(0, scaled_w - dw) // 2, "crop_y": max(0, scaled_h - dh) // 2,
    }
    assert_no_silent_stretch(plan)
    return plan


def assert_no_silent_stretch(plan, tol=0.02):
    """Guard: ``plan`` scaled both axes by the SAME factor (aspect preserved).

    Recovers the effective per-axis scale from the plan; if they differ by more
    than ``tol`` (a non-uniform / implicit stretch) it raises. Even-rounding
    introduces a sub-pixel delta, hence the small tolerance.
    """
    sx = plan["scaled_w"] / plan["src_w"]
    sy = plan["scaled_h"] / plan["src_h"]
    if abs(sx - sy) > tol * max(sx, sy):
        raise ValueError(
            "aspect plan stretches (sx=%.4f sy=%.4f); a uniform scale is "
            "required: %r" % (sx, sy, plan))
    return True


# --------------------------------------------------------------------------- #
# Mid-sampling NVML ceiling probe (PASS-PM: peak during render, not just
# pre/post -- an additively-resident LoRA delta can breach mid-sample)
# --------------------------------------------------------------------------- #
def vram_used_mb():
    """Machine-wide VRAM used (MB) via the shared NVML probe, or ``None`` when
    NVML is unavailable (e.g. the CPU box). Never ComfyUI ``get_free_memory()``
    (this-process view only -- it cannot see a sidecar's allocation)."""
    if not _GR.nvml_available():
        return None
    return _GR.probe_used_mb()


def assert_vram_within_ceiling(label="render", ceiling_mb=None):
    """Mid-sampling NVML guard: assert machine-wide used VRAM has not breached the
    single-heavy-engine ceiling DURING a render.

    The GPU-smoke render loop calls this from inside the sampling callback (not
    only at the pre/post boundary), because a LoRA delta is additively resident
    and can push the peak past the ceiling mid-sample. A no-op when NVML is
    unavailable (the CPU box -- the boundary settle-wait in ``teardown`` covers
    that path). Raises ``RuntimeError`` on a breach; returns the used MB (or
    ``None`` when NVML is absent).
    """
    if ceiling_mb is None:
        ceiling_mb = dynamic_vram_ceiling_mb()   # env-at-dispatch (GATE B S1)
    used = vram_used_mb()
    if used is None:
        return None
    if used > int(ceiling_mb):
        raise RuntimeError(
            "VRAM ceiling breached mid-%s: %d MB > %d MB (single resident heavy "
            "engine)" % (label, used, int(ceiling_mb)))
    return used


class VramPeakProbe:
    """Background NVML sampler: the PEAK machine-wide VRAM (MB) observed across a
    render window (first heavy model load through VAEDecode), not just the
    instantaneous pre/post boundary.

    A post-render single read (the pattern this replaces) fires AFTER the GPU work
    and misses the sampler / text-encode peak; this thread samples every
    ``interval_s`` for the duration of the window so an additively-resident encoder
    or LoRA delta that breaches mid-render is actually caught. A pure no-op (peak
    stays 0) when NVML is unavailable (the CPU box); ``threading`` is stdlib so the
    cold-import invariant (V-12) holds. Use ``start()`` before the render call and
    ``stop()`` after the decoded IMAGE is in hand."""

    def __init__(self, interval_s=1.0):
        self._interval = float(interval_s)
        self._stop = None
        self._thread = None
        self.peak_mb = 0

    def _loop(self):
        while not self._stop.is_set():
            used = vram_used_mb()
            if used is not None and used > self.peak_mb:
                self.peak_mb = used
            self._stop.wait(self._interval)

    def start(self):
        import threading
        if vram_used_mb() is None:           # NVML absent (CPU box) -> no-op probe
            return self
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.peak_mb


def assert_peak_within_ceiling(peak_mb, label="render", ceiling_mb=None):
    """Assert a MEASURED render-window peak (from :class:`VramPeakProbe`) has not
    breached the single-heavy-engine ceiling. A no-op when ``peak_mb`` is 0/falsy
    (NVML absent -> nothing measured). Raises ``RuntimeError`` on a breach."""
    if ceiling_mb is None:
        ceiling_mb = dynamic_vram_ceiling_mb()
    if peak_mb and int(peak_mb) > int(ceiling_mb):
        raise RuntimeError(
            "VRAM ceiling breached across %s window: %d MB > %d MB (single "
            "resident heavy engine)" % (label, int(peak_mb), int(ceiling_mb)))
    return peak_mb


# --------------------------------------------------------------------------- #
# In-process motion-engine base (AS-3 lease + V-4 teardown)
# --------------------------------------------------------------------------- #
class MotionEngineBase:
    """Shared lifecycle for an IN-PROCESS motion engine (LTX / Wan / HuMo).

    Subclasses set the registry-core metadata (``name`` / ``family`` / ``roles``
    / ...) and implement ``load`` / ``render_clip`` / ``canonicalize`` /
    ``assert_usable``. This base provides the AS-3 single-heavy-engine lease on
    ``prepare`` and the V-4 patcher-detach ``teardown`` (NEVER
    ``unload_all_models``), so every motion adapter serialises behind one lease
    and tears down without the global unload. ``__init__`` is cheap (no weights).
    """

    declared_isolation = ISOLATION_IN_PROCESS
    binds_seed = True

    #: Coverage architecture (2026-06-18): EVERY in-process motion lane accepts the
    #: role's SELECTED image (init still) by default -- the image dispatcher reads
    #: this ONE capability to decide whether to mint the still, so a new video engine
    #: gets the chosen image automatically with NO per-engine whitelist ("one and
    #: done"). Audio-only lanes (ltx_av_music) override to False; the pure procedural
    #: floors (visualizer / abstract) declare False too. ltx_video inherits True here,
    #: which is what lets a flux2/flux still drive a silent LTX i2v clip. Plain attr
    #: (cold-import clean). See docs/2026-06-18-coverage-arch-wiring/.
    accepts_still = True

    def __init__(self):
        self._loaded = False
        self._patchers = []

    # load / unload bracket residency; the heavy import + wrapper load is the
    # CW-6 GPU-smoke slice, added per engine (lazy, never at module scope).
    def load(self):  # pragma: no cover - overridden by each engine
        raise NotImplementedError

    def unload(self):
        self._patchers = []
        self._loaded = False

    def prepare(self, host_caps, profile, session_ctx):
        """Take the SHARED single-heavy-engine lease (AS-3) BEFORE loading
        weights, then load. FAIL CLOSED: a held lease or a failed load raises and
        the lease is never stranded."""
        lease = _GR.acquire(
            timeout_s=float(os.getenv("OTR_GPU_LEASE_TIMEOUT_S", "120")))
        try:
            self.load()
        except BaseException:
            _GR.release(lease)              # never strand the lease on a failure
            raise
        return {"engine_id": self.name, "lease": lease,
                "patchers": self._patchers}

    def teardown(self, prepared):
        """Detach every tracked patcher (V-4), drop residency, RELEASE the lease,
        then bounded-wait for machine-wide VRAM below the ceiling. Idempotent +
        never raises out of teardown. NEVER ``unload_all_models`` (V-4 / V-5)."""
        self._detach_patchers(prepared)
        self.unload()
        lease = (prepared or {}).get("lease")
        had_lease = lease is not None
        _GR.release(lease)
        if had_lease:
            _GR.wait_until_below_mb(dynamic_vram_ceiling_mb(),
                                    attempts=3, sleep_s=2.0)

    @staticmethod
    def _detach_patchers(prepared):
        """V-4: detach EVERY tracked patcher (Wan experts + each LoRA) with
        ``patcher.detach(unpatch_all=True)`` and clear strong refs. NEVER
        ``unload_all_models()``. Guarded + idempotent; a no-op on the CPU box
        where nothing was tracked."""
        for patcher in list((prepared or {}).get("patchers") or []):
            try:
                detach = getattr(patcher, "detach", None)
                if callable(detach):
                    detach(unpatch_all=True)
            except Exception:              # noqa: BLE001 - teardown must not raise
                pass
        if isinstance(prepared, dict):
            prepared["patchers"] = []


__all__ = [
    "VRAM_CEILING_MB", "dynamic_vram_ceiling_mb",
    "ASPECT_POLICIES", "DEFAULT_ASPECT_POLICY",
    "ISOLATION_IN_PROCESS", "ISOLATION_SIDECAR_REQUIRED",
    "ISOLATION_SIDECAR_OPTIONAL", "sageattention_patched",
    "assert_sage_not_patched", "resolve_isolation", "assert_aspect_policy",
    "resolve_aspect_transform", "assert_no_silent_stretch", "vram_used_mb",
    "assert_vram_within_ceiling", "VramPeakProbe", "assert_peak_within_ceiling",
    "MotionEngineBase",
]
