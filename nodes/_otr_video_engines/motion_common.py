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

import math
import os
import sys

from .._otr_shared import gpu_residency as _GR
from .registry import EngineUnusable, EngineUsabilityReason

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
# Mid-sampling NVML telemetry (PASS-PM: peak during render, not just pre/post --
# an additively-resident LoRA delta shows up mid-sample). TELEMETRY ONLY: the
# OOM budget is owned by the operator's tier JSON now, so there is no ceiling
# assert -- the peak is sampled + logged, never enforced.
# --------------------------------------------------------------------------- #
def vram_used_mb():
    """Machine-wide VRAM used (MB) via the shared NVML probe, or ``None`` when
    NVML is unavailable (e.g. the CPU box). Never ComfyUI ``get_free_memory()``
    (this-process view only -- it cannot see a sidecar's allocation)."""
    if not _GR.nvml_available():
        return None
    return _GR.probe_used_mb()


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


# --------------------------------------------------------------------------- #
# Dynamic-VRAM frame budget (2026-06-18 clip-fill roundtable: ChatGPT + Gemini +
# DeepSeek, Claude judged + grounded). PREDICT how many real frames an engine can
# render within the live VRAM budget from a ZERO-COST mem_get_info read + a cost
# model -- NEVER react-to-OOM (a CUDA OOM inside ComfyUI's long-lived process
# corrupts the caching allocator, so OOM is a bug to AVOID, never a control
# signal). Replaces the wan_ti2v hard 17-frame "8GB floor" that froze every clip
# to 0.68s. Pure given its inputs (free VRAM is read by the caller and passed in),
# so the math is CPU-testable without a GPU.
# --------------------------------------------------------------------------- #

#: Telemetry reference resolution the per-frame cost is measured at (wan_ti2v
#: render-phase peak 10277 MB @ 17 frames @ 1472x832 -> 7000 + 185*17 ~= 10145).
_FRAME_COST_REF_PIXELS = 1472 * 832

#: Per-engine VRAM cost model SEED: ``vram_mb ~= overhead_mb + per_frame_mb *
#: frames`` at :data:`_FRAME_COST_REF_PIXELS`. ``overhead`` is the resident model
#: + fixed buffers (held constant across resolutions -- conservative);
#: ``per_frame`` is the activation/decode cost that scales with pixel area. Refine
#: from observed peaks; a new engine without a row uses :data:`_DEFAULT_FRAME_COST`.
#: Globally env-overridable via OTR_VIDEO_COST_OVERHEAD_MB / OTR_VIDEO_COST_PER_FRAME_MB.
FRAME_COST_MODEL = {
    "wan_ti2v": (7000.0, 185.0),
}
#: Fallback cost row for an engine not in :data:`FRAME_COST_MODEL` (use the wan
#: 5B figure -- the conservative low-VRAM tier the budget mainly guards).
_DEFAULT_FRAME_COST = (7000.0, 185.0)

#: Per-engine MOTION floor (4n+1 minimum): the fewest frames of motion a beat may
#: carry. It is the LOWER BOUND handed to ``quantize_frames_4n1``; it is NOT an
#: override of the VRAM budget.
#:
#: CORRECTED 2026-08-01 (kibitz r4, verified against this file). This comment used
#: to claim "the floor WINS over the budget (if the floor itself OOMs, the
#: render-window NVML probe catches it LOUD)". BOTH HALVES WERE FALSE, and the
#: pair of them nearly bought a real defect:
#:   * The floor does NOT win. ``compute_real_frame_budget`` raises
#:     :class:`MotionBudgetError` whenever ``affordable < snapped``, and ``snapped``
#:     carries the floor -- so an unaffordable floor REFUSES the render rather than
#:     rendering short.
#:   * ``VramPeakProbe`` catches nothing. Its own contract above is explicit:
#:     "there is no ceiling assert -- the peak is sampled + logged, never enforced."
#: A plan written against the old wording proposed bypassing the refusal on the
#: strength of a probe that would not have caught the resulting CUDA OOM. Preflight
#: refusal is the ONLY guard on this path; keep it that way.
#:
#: NOTE ``snapped`` CAN fall below the floor: ``quantize_frames_4n1`` applies
#: ``max_frames`` AFTER the minimum, so a target under the floor stays under it.
#: LTX has its own decode floor; the generic default is 1.
FRAME_MOTION_FLOOR = {"wan_ti2v": 17}
_DEFAULT_MOTION_FLOOR = 1

#: Fraction of the usable VRAM the predictor may spend -- head-room for allocator
#: fragmentation so the prediction never has to react-to-OOM. Env OTR_VIDEO_BUDGET_MARGIN.
_BUDGET_MARGIN = 0.85


def free_vram_mb():
    """Machine-wide FREE VRAM (MB) via a ZERO-COST ``torch.cuda.mem_get_info``
    read, or ``None`` when torch/CUDA is unavailable (the CPU box / unit tests).

    This is the "probe" the dynamic frame budget uses: 0 bytes allocated, 0 GPU
    time, no render. NEVER a render-probe (try-then-OOM corrupts the allocator).
    Pure telemetry; never raises."""
    try:
        import torch  # type: ignore
        if not torch.cuda.is_available():
            return None
        free_b, _total_b = torch.cuda.mem_get_info()
        return float(free_b) / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001 -- no torch/CUDA -> caller trusts the target
        return None


def _cost_model_for(engine_name):
    """(overhead_mb, per_frame_mb) for ``engine_name`` with global env overrides."""
    overhead, per_frame = FRAME_COST_MODEL.get(engine_name, _DEFAULT_FRAME_COST)
    raw_o = (os.environ.get("OTR_VIDEO_COST_OVERHEAD_MB") or "").strip()
    raw_f = (os.environ.get("OTR_VIDEO_COST_PER_FRAME_MB") or "").strip()
    try:
        if raw_o:
            overhead = float(raw_o)
        if raw_f:
            per_frame = float(raw_f)
    except (TypeError, ValueError):
        pass
    return float(overhead), float(per_frame)


class MotionBudgetError(RuntimeError):
    """The STATIC frame budget cannot fit the live VRAM cost model.

    S4 platform-portability (2026-07-10): the render NEVER resizes itself --
    lower the frame_count widget, free VRAM, or pick a lighter engine."""


def has_measured_cost_row(engine_name) -> bool:
    """Is there a cost row MEASURED for this engine, or only the fallback?

    The distinction is the whole point. ``_DEFAULT_FRAME_COST`` makes an engine
    nobody has ever measured look exactly like an engine that passed a
    calibration, because the prediction runs either way and returns a number.
    That is how a guard becomes theatre: the check executes, reports a limit,
    and the limit describes a different engine.

    Callers that want a REAL guard ask this first and say plainly when the
    answer is no, rather than enforcing a borrowed number against hardware it
    was never measured on.
    """
    return str(engine_name or "") in FRAME_COST_MODEL


def assert_frame_affordable(free_vram_mb_value, frame_count, canvas_w,
                            canvas_h, engine_name):
    """Refuse an UNAFFORDABLE length. The guard the planned path never had.

    ``_planned_length`` deliberately does not consult the VRAM predictor -- a
    planned segment's length is arithmetic the rest of the beat was built
    around, not a preference. That was survivable while planned segments were
    rare; with ``wan_ti2v`` now coverage-planned it means the ONLY enforcing
    guard was bypassed on the path that does most of the rendering
    (``VramPeakProbe`` samples and never enforces).

    So: same cost model, same refusal, no re-snapping. The caller passes the
    length it already decided on and the SAME free-VRAM accounting
    ``_floor_length`` uses (live free + hoisted, so resident weights are not
    charged twice). Raises :class:`MotionBudgetError`; returns ``frame_count``.
    """
    compute_real_frame_budget(free_vram_mb_value, frame_count,
                              canvas_w, canvas_h, engine_name)
    return int(frame_count)


def compute_real_frame_budget(free_vram_mb_value, target_frame_count,
                              canvas_w, canvas_h, engine_name):
    """S4 platform-portability REWRITE (2026-07-10): the frame budget is the
    STATIC widget value -- geometry only (engine motion floor + 4n+1 snap),
    NEVER a VRAM-adaptive resize. The pre-S4 version silently shrank
    tight-VRAM clips toward the floor (output length varied with host state
    -- exactly the auto-adapt class this campaign kills; the clip-fill era
    is superseded by the per-tier frame_budget widget).

    The cost model (``vram ~= overhead + per_frame_at_res * frames``,
    ``budget = free * margin``) is KEPT as a fail-loud PREDICTION: if the
    static target cannot fit, this RAISES :class:`MotionBudgetError` before
    the engine burns a doomed forward. ``free_vram_mb_value`` None / <= 0
    (no NVML/torch -- the CPU box) skips the prediction (geometry only).
    Pure (no GPU read here -- the caller passes ``free_vram_mb()``);
    CPU-tested."""
    from . import wrapper_bridge as _wb
    target = max(1, int(target_frame_count or 1))
    floor = int(FRAME_MOTION_FLOOR.get(engine_name, _DEFAULT_MOTION_FLOOR))
    snapped = _wb.quantize_frames_4n1(target, min_frames=floor,
                                      max_frames=target)
    # No live VRAM reading -> geometry only (CPU box / cloud lanes).
    if free_vram_mb_value is None or float(free_vram_mb_value) <= 0:
        return snapped
    overhead, per_frame = _cost_model_for(engine_name)
    pixels = max(1, int(canvas_w) * int(canvas_h))
    per_frame_at_res = per_frame * (pixels / float(_FRAME_COST_REF_PIXELS))
    try:
        margin = float(os.environ.get("OTR_VIDEO_BUDGET_MARGIN", _BUDGET_MARGIN))
    except (TypeError, ValueError):
        margin = _BUDGET_MARGIN
    budget_mb = float(free_vram_mb_value) * margin

    # ---- THE ZERO-SLOPE HOLE (kibitz r3, found INDEPENDENTLY by both lanes;
    # control flow specified by r2 MUST-FIX 3). Until 2026-08-02 BOTH the
    # fixed-overhead admission and the frame admission lived under
    # `if per_frame_at_res > 0`, so a cost row whose per-frame measured ZERO
    # disabled the ONLY enforcing guard on this path entirely -- no refusal
    # ever fired, even when the resident model alone did not fit. It could not
    # fire while the shipped row was (7000, 185); it arms itself the moment a
    # MEASURED row lands, and the estimator fit clamps slope at >= 0 precisely
    # because the low end of the ladder is nearly flat. So this is a
    # PREREQUISITE of the cost-row commit, not a follow-up to it.
    #
    # 1. a malformed cost model is a configuration error, not a budget answer
    for label, value in (("overhead", overhead),
                         ("per_frame", per_frame_at_res),
                         ("margin", margin)):
        if not math.isfinite(value) or value < 0:
            raise MotionBudgetError(
                "engine %s: cost model %s is %r -- a cost model must be finite "
                "and non-negative. Check FRAME_COST_MODEL and the "
                "OTR_VIDEO_COST_* / OTR_VIDEO_BUDGET_MARGIN overrides."
                % (engine_name, label, value))
    # 2. the FIXED overhead must fit, whatever the slope. Unconditional.
    if budget_mb < overhead:
        raise MotionBudgetError(
            "engine %s: the model's fixed overhead alone (%.0f MB) exceeds the "
            "usable budget %.0f MB (free=%.0f MB, margin=%.2f). No frame count "
            "is affordable -- free VRAM or pick a lighter engine."
            % (engine_name, overhead, budget_mb,
               float(free_vram_mb_value), margin))
    # 3. zero slope is legal ONLY now that the overhead has been proven to fit:
    #    frames are free, so any length is affordable.
    # 4. a positive slope prices the frames.
    if per_frame_at_res > 0:
        affordable = int((budget_mb - overhead) / per_frame_at_res)
        if affordable < snapped:
            raise MotionBudgetError(
                "engine %s: static frame budget %d (snapped %d) exceeds the "
                "cost-model's affordable %d frames (free=%.0f MB, "
                "margin=%.2f). NO silent resize -- lower the frame_count "
                "widget, free VRAM, or pick a lighter engine."
                % (engine_name, target, snapped, max(0, affordable),
                   float(free_vram_mb_value), margin))
    return snapped


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
    invocable = True
    invocability_reason = ""

    #: Dynamic-VRAM frame budget, exposed on the base so every motion engine can
    #: PREDICT (never react-to-OOM) how many of a beat's frames fit the live VRAM
    #: budget, then loop/ping-pong-extend the short render to the full target.
    #: Reads free VRAM via :func:`free_vram_mb` (zero-cost mem_get_info). Static so
    #: the prediction math stays pure + CPU-testable.
    compute_real_frame_budget = staticmethod(compute_real_frame_budget)
    free_vram_mb = staticmethod(free_vram_mb)

    #: Coverage architecture (2026-06-18): EVERY in-process motion lane accepts the
    #: role's SELECTED image (init still) by default -- the image dispatcher reads
    #: this ONE capability to decide whether to mint the still, so a new video engine
    #: gets the chosen image automatically with NO per-engine whitelist ("one and
    #: done"). Audio-only lanes (ltx_av_music) override to False; the pure procedural
    #: floors (visualizer / abstract) declare False too. ltx_video inherits True here,
    #: which is what lets a flux2/flux still drive a silent LTX i2v clip. Plain attr
    #: (cold-import clean). See docs/2026-06-18-coverage-arch-wiring/.
    accepts_still = True

    #: The ledger-stamped v2 render policy for the CURRENT episode, captured on
    #: ``prepare`` (WAN 8GB launch contract, 2026-07-24). Class-level default so
    #: an engine used without ``prepare`` (unit fixtures, single-shot probes)
    #: reads an empty policy = unpinned.
    _active_profile: dict = {}

    def __init__(self):
        self._loaded = False
        self._patchers = []
        self._active_profile = {}

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
        the lease is never stranded.

        Also captures the episode's v2 render policy (WAN 8GB launch contract,
        2026-07-24) so an engine can read its tier's ``max_render_frames``
        ceiling at render time. Read-only; never mutated.

        ``session_ctx`` IS RETURNED IN ``prepared`` (2026-08-01). It used to be
        accepted and DROPPED, which is a regression the multi-clip work
        introduced and which took a live campaign to surface:

        * ``BeatSession`` passes it in (``beat_session.py:165-167``) because it
          is the ONLY channel that tells ``render_clip`` whether its output will
          be concatenated with other segments into one beat -- a coverage-planned
          segment's request is shaped exactly like a single-clip beat's.
        * Dropping it left ``prepared["session_ctx"]`` absent, so every consumer
          reading ``multi_clip`` saw nothing and took the single-clip branch.
        * ``eng_wan_ti2v`` and ``eng_humo`` each re-added it LOCALLY and worked;
          ``ltx_video`` / ``ltx_8gb`` / ``ltx_av`` / ``wan_i2v`` never did, so
          ``ltx_video``'s loop-fill enabled the boomerang on a beat whose length a
          coverage plan had already decided.
        * ``ltx_video`` last delivered episode clips 2026-07-06; the beat session
          landed 2026-07-25 (``4fa992e6`` / ``e90dedf1``). It worked before
          multi-clip existed, which is exactly why "it always worked" and "it
          fails now" are both true.

        THE BASE OWNS THIS, not each adapter. Two adapters carrying a private
        workaround for a shared-base gap is how the sibling adapters were left
        broken in the first place -- the same failure class as the ``eng_wan_i2v``
        VRAM-peak lesson. A single-clip render (``render_single``) never opens a
        ``BeatSession``, so it passes ``None`` and gets ``{}`` here: absent and
        empty must stay indistinguishable to callers."""
        self._active_profile = dict(profile or {})
        lease = _GR.acquire(
            timeout_s=float(os.getenv("OTR_GPU_LEASE_TIMEOUT_S", "120")))
        try:
            self.load()
        except BaseException:
            _GR.release(lease)              # never strand the lease on a failure
            raise
        return {"engine_id": self.name, "lease": lease,
                "patchers": self._patchers,
                "session_ctx": dict(session_ctx or {})}

    def profile_max_render_frames(self):
        """The tier's ABSOLUTE render-length ceiling in frames from the captured
        v2 policy, or 0 when unpinned (WAN 8GB launch contract, 2026-07-24).

        This is the profile-carried twin of a per-engine ``*_MAX_FRAMES`` env
        pin: a production episode leg is submitted to an already-booted server,
        so ``launch.env`` cannot reach it -- the ceiling has to travel
        profile -> director widget -> ledger -> here. Never raises; a malformed
        stamp reads as unpinned rather than failing a render.

        DELEGATES TO THE ONE NORMALIZER (2026-07-27, B3 post-code panel).
        This used to be a third hand-copied ``max(0, int(x or 0))``, alongside
        the ledger stamp and the planner. They agreed only because someone
        copied carefully; the whole point of B3's single helper is that the
        adapter-side cap and the planned ceiling can never read the same stamp
        differently.
        """
        try:
            from . import frame_contract as _fc  # type: ignore
        except ImportError:  # pragma: no cover -- flat test imports
            import frame_contract as _fc  # type: ignore
        return _fc.normalized_planning_ceiling(
            (self._active_profile or {}).get("max_render_frames"))

    def teardown(self, prepared):
        """Detach every tracked patcher (V-4), drop residency, RELEASE the lease,
        then bounded stability-wait for machine-wide VRAM to settle (no ceiling --
        the reclaim already happened; this just absorbs teardown latency).
        Idempotent + never raises out of teardown. NEVER ``unload_all_models``
        (V-4 / V-5).

        THE LEASE RELEASE SITS IN A ``finally`` (2026-07-26, chunk-5 QA panel,
        agy Gemini 3.6 Flash High; grounded and confirmed). ``_detach_patchers``
        is guarded per patcher, but ``unload()`` is OVERRIDDEN per engine, and
        an override that raises used to leave this method before the release --
        stranding the shared single-heavy-engine lease in global state. The NEXT
        episode then blocks on ``acquire`` for its full 120s timeout and fails
        for a reason that has nothing to do with it. Reclaim is best-effort;
        the lease is not."""
        lease = (prepared or {}).get("lease")
        had_lease = lease is not None
        try:
            self._detach_patchers(prepared)
            self.unload()
        finally:
            _GR.release(lease)
            if had_lease:
                _GR.wait_until_stable(attempts=3, sleep_s=2.0)

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
    "ASPECT_POLICIES", "DEFAULT_ASPECT_POLICY",
    "ISOLATION_IN_PROCESS", "ISOLATION_SIDECAR_REQUIRED",
    "ISOLATION_SIDECAR_OPTIONAL", "sageattention_patched",
    "assert_sage_not_patched", "resolve_isolation", "assert_aspect_policy",
    "resolve_aspect_transform", "assert_no_silent_stretch", "vram_used_mb",
    "VramPeakProbe",
    "FRAME_COST_MODEL", "FRAME_MOTION_FLOOR", "free_vram_mb",
    "assert_frame_affordable", "compute_real_frame_budget",
    "MotionEngineBase",
]
