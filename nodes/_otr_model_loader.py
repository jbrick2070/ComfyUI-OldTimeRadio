"""nodes/_otr_model_loader.py

Thin facade over story_orchestrator._load_llm / _unload_llm. Exists to give
the v2.0 path (_otr_outline, _otr_line_composer, OTR_LedgerScriptWriter) a
stable import surface that doesn't depend on the legacy orchestrator's
internal layout.

DOES NOT extract code from story_orchestrator.py during the in-flight FULL
acceptance soak. Re-exports only. When the legacy writer retires in v2.1,
the loader code can move here without changing any v2.0 import sites.

Public surface:
    load_llm(model_id, *, device='cuda', optimization_profile='Standard') -> dict
        Returns a cache_entry dict: {model, tokenizer, model_id, device,
        quantized, budget_profile, context_cap}. Wraps _load_llm's tuple
        return into the dict shape documented by _otr_outline.py.

    unload_llm() -> int
        Re-export of _unload_llm. Frees VRAM globally. Returns the new
        cache epoch (see _CACHE_EPOCH_LOCK); most callers ignore it.

    unload_llm_if_local_resident() -> bool
        Handoff helper: skips the full torch/CUDA teardown when the writer
        used only remote LLM providers and no local cache entry exists.

    Per-model context caps are resolved by
    _otr_model_catalog.resolve_context_cap (single source of truth); load_llm
    uses the caller-supplied context_cap when given and resolves through the
    catalog otherwise. The old local MODEL_CONTEXT_CAPS table is gone.

    make_generate_fn(cache_entry) -> GenerateFn
        Wraps a cache_entry into a chat-template-aware callable matching
        the GenerateFn contract used by _otr_outline.generate_outline and
        _otr_line_composer.compose_line:
            (messages, *, temperature, max_new_tokens) -> str

Status: Phase 2 of v2.0 sprint. Stdlib + lazy imports of torch and
story_orchestrator only.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

# Live token heartbeat. A LEAF module on purpose: _otr_constrained_generate
# imports FROM this file, so the streamer cannot live there without a cycle --
# which is exactly why this transport had no live view for so long.
#
# RELATIVE WITH A BARE FALLBACK, and this file specifically NEEDS both halves:
# it is imported as `nodes._otr_model_loader` on a server AND as a bare
# top-level `_otr_model_loader` by tests that put nodes/ on sys.path (see
# tests/test_writer_llm_unload.py:23). A plain relative import raises
# "no known parent package" under the second form -- which is how the first
# version of this line took three tests red.
try:
    from . import _otr_writer_heartbeat as _OTRHB  # type: ignore
except ImportError:  # loaded with nodes/ on sys.path
    import _otr_writer_heartbeat as _OTRHB  # type: ignore

try:
    from ._otr_generation_budget import (
        GenerationContextOverflowError,
        GenerationDegeneracyError,
        PromptContextOverflowError,
        fit_output_tokens,
    )
except ImportError:  # pragma: no cover - flat-module compatibility tests
    from _otr_generation_budget import (  # type: ignore
        GenerationContextOverflowError,
        GenerationDegeneracyError,
        PromptContextOverflowError,
        fit_output_tokens,
    )

log = logging.getLogger("OTR")


__all__ = [
    "load_llm",
    "unload_llm",
    "has_local_resident_llm",
    "unload_llm_if_local_resident",
    "invalidate_cache_no_gpu_teardown",
    "request_slot",
    "make_generate_fn",
    "make_polish_generate_fn",
    "ModelLoaderError",
    "GenerationDeadlineExceededError",
    # The deadline API is public because the GGUF backend consumes it from
    # another module -- it is a cross-module contract, not an internal.
    "set_generation_deadline",
    "get_generation_deadline",
    "generation_deadline_expired",
    "LLM_CACHE",
]


# ---------------------------------------------------------------------------
# B1c: shared slot-aware LLM cache (the modern facade).
#
# Records the currently-resident cache_entry so request_slot can detect
# slot transitions and decide between cache-reuse (same model) vs full
# unload + reload (different model). S30 B5 collapsed
# visual/llm_polish.py's local cache into this single source of truth
# so the 16 GB card never double-loads a model.
# ---------------------------------------------------------------------------


LLM_CACHE: dict[str, Any] = {
    "model_id": None,
    "slot": None,
    "cache_entry": None,
}

# PBUG-20260825-04 (orphan-lifecycle window): request_slot's own Step-9
# cache store (below, "cache (policy-keyed, S1)") writes UNCONDITIONALLY
# once load_llm() returns -- with no check for whether the CALLER is still
# wanted. story_orchestrator._run_with_timeout abandons a worker thread on
# a wall-clock timeout (generation is not cancellable mid-token) and
# invalidates LLM_CACHE via invalidate_cache_no_gpu_teardown() -- but if
# the abandoned worker's OWN load_llm() call then finishes successfully
# (weights materialize fine, the tripwire passes), its Step-9 write lands
# in the dict AFTER the invalidation, unguarded, and a completely
# unrelated LATER request_slot call can then take a cache HIT on that
# entry and start ITS OWN generate() on the SAME model object the orphan is
# still actively generating with in another thread. That is not VRAM
# contention -- it is two threads mutating one model's generation/KV-cache
# state concurrently.
#
# _CACHE_EPOCH closes that specific window without the larger
# orphan-occupancy registry (deferred to a dedicated session -- see
# docs/PROD_BUG_LOG.md PBUG-20260825-04). Every invalidation path bumps the
# epoch; request_slot snapshots it on entry and only performs the Step-9
# store if the epoch is UNCHANGED. An abandoned call's late write is then
# silently skipped -- its caller is gone, the entry dies with the orphan's
# own stack frame, exactly as it already does today for everything except
# this one unconditional dict write.
#
# OWNERSHIP, not just a counter: request_slot can invalidate the cache
# ITSELF, mid-call, as ordinary control flow -- a GGUF load-config change
# or a cross-model slot transition both tear down the old resident model
# before loading the replacement, in the SAME call whose Step-9 store
# follows a few lines later. The DECISION to self-unload is made from an
# unlocked read a few lines before the actual teardown call ("a different
# model is resident"); if an external invalidation races into that exact
# gap, the self-unload still runs, so its own resulting bump must NOT be
# blindly adopted as if it were legitimately this call's own. The claim is
# therefore CONDITIONAL, atomically: only bump and hand back a new epoch if
# the caller's snapshot was STILL current at the exact moment of the
# attempt; otherwise, no-op entirely (no GPU touch, no LLM_CACHE mutation,
# no epoch adoption) and let the caller keep its now-provably-stale
# snapshot, so its eventual store correctly fails instead of laundering
# someone else's invalidation into a legitimate-looking self-triggered one.
# See _self_unload / _detach_and_invalidate_locked's ``expected_epoch`` gate
# below -- request_slot must never call the raw, unconditional unload_llm()
# for ANY of its own self-triggered teardowns, success path or failure-path
# cleanup alike.
#
# Atomicity: the clear+bump and the check+publish must never be observably
# separate operations, or a concurrent invalidation can land between
# "epoch still matches" and "write the entry" and still let a stale entry
# through. Every mutation of _CACHE_EPOCH or the LLM_CACHE identity fields
# goes through the helpers below, all sharing one lock.
_CACHE_EPOCH_LOCK = threading.Lock()
_CACHE_EPOCH: int = 0


def _current_cache_epoch() -> int:
    with _CACHE_EPOCH_LOCK:
        return _CACHE_EPOCH


def _detach_and_invalidate_locked(
    expected_epoch: int | None = None,
) -> tuple[dict | None, int | None]:
    """Atomically capture the resident cache_entry, clear LLM_CACHE's
    identity fields, and bump the epoch -- all under one lock acquisition,
    so no concurrent publish/read can observe the clear and the bump as
    separate steps.

    With ``expected_epoch=None`` (the unconditional case: ``unload_llm``'s
    and ``invalidate_cache_no_gpu_teardown``'s ordinary external-caller
    behavior), always proceeds and returns ``(detached_entry, new_epoch)``.

    With ``expected_epoch`` given (request_slot's own self-triggered
    teardown, via ``_self_unload``), proceeds ONLY if the live epoch still
    equals it; otherwise this is a complete no-op and returns
    ``(None, None)`` -- the caller has been externally invalidated since it
    last checked and must not claim ownership of a bump it did not itself
    (legitimately) cause.
    """
    global _CACHE_EPOCH
    with _CACHE_EPOCH_LOCK:
        if expected_epoch is not None and _CACHE_EPOCH != expected_epoch:
            return None, None
        entry = LLM_CACHE.get("cache_entry")
        LLM_CACHE.clear()
        LLM_CACHE.update({"model_id": None, "slot": None, "cache_entry": None})
        _CACHE_EPOCH += 1
        return entry, _CACHE_EPOCH


def _try_cache_hit_locked(
    normalized: str, slot: str, *,
    gguf_key: str | None = None, policy_key: str | None = None,
) -> dict | None:
    """Atomically check a cache-hit predicate and capture+return the
    matching entry (also stamping ``slot``) under the same lock -- a
    separate check-then-return would let an invalidation land between the
    predicate passing and the return, producing either a ``None`` where
    request_slot's contract promises a dict, or a hit against an entry
    that no longer exists. Pass exactly one of ``gguf_key``/``policy_key``
    to select which identity dimension gates the hit. Returns ``None`` on
    any miss; the caller is responsible for its own (unlocked, advisory)
    miss-reason logging afterward.
    """
    with _CACHE_EPOCH_LOCK:
        if (
            LLM_CACHE.get("model_id") != normalized
            or LLM_CACHE.get("cache_entry") is None
        ):
            return None
        if gguf_key is not None and LLM_CACHE.get("gguf_load_key") != gguf_key:
            return None
        if policy_key is not None and LLM_CACHE.get("policy_key") != policy_key:
            return None
        LLM_CACHE["slot"] = slot
        return LLM_CACHE["cache_entry"]


def _publish_cache_entry_if_current(expected_epoch: int, fields: dict) -> bool:
    """Publish ``fields`` into LLM_CACHE iff the epoch is still
    ``expected_epoch``, atomically with the read -- the check and the
    write happen under the same lock a concurrent invalidation also uses,
    closing the gap a separate check-then-write would leave open. Returns
    whether the publish happened; the caller logs on False."""
    with _CACHE_EPOCH_LOCK:
        if _CACHE_EPOCH != expected_epoch:
            return False
        LLM_CACHE.update(fields)
        return True


# 2026-08-25 (orphan-lifecycle round, Fable's cold-take finding): the codebase
# already has a working per-token deadline check --
# story_orchestrator.GemmaHeartbeatStreamer.put() raises TimeoutError once a
# thread-local deadline has passed -- but it lives on a DIFFERENT, legacy
# streamer that make_generate_fn's shared TRANSFORMERS transport (below)
# never wired in. NewsCuration/NewsCurationDeep (the two _run_with_timeout-
# bound phases) go through make_generate_fn, so on the transformers lane an
# abandoned worker previously ran to its FULL max_new_tokens budget before
# its thread could unwind -- minutes, at slow token rates, of extra orphan
# lifetime with nothing checking whether anyone was still waiting.
#
# This is the shared-layer equivalent, owned here rather than imported from
# story_orchestrator (this file is the transport every caller already goes
# through; per this file's own docstring it is meant to absorb shared
# behavior as the legacy orchestrator's internals are phased out, not reach
# back into them). _run_with_timeout COMPUTES one absolute monotonic deadline
# BEFORE submitting its worker, the worker INSTALLS that same value and clears
# it in its own finally, and the parent waits only the remaining duration.
# (Corrected 2026-08-25: this comment used to say the deadline was set before
# submission, while the code computed it inside the worker -- so the worker's
# deadline outlived the parent's timeout by the scheduling delay. One shared
# value removes the skew and makes this sentence true.) Both
# user-facing transformers generate closures (make_generate_fn,
# make_polish_generate_fn) check it via _DeadlineStoppingCriteria, cutting
# an abandoned worker's remaining lifetime to ~1 token instead of the full
# budget. (load_llm's own internal CUDA-warmup model.generate() call --
# max_new_tokens=1, do_sample=False, already exempt from the sibling
# degeneracy guard for the same reason -- is not wired in: one warmup
# token adds negligible orphan lifetime, not worth a third criterion.)
#
# THE GGUF LANE IS NOW COVERED TOO (2026-08-25, closing the PBUG-20260825-04
# deferral). It could not use a StoppingCriteria: llama-cpp-python's
# create_chat_completion accepts no `stopping_criteria` and does not forward
# one (verified against installed 0.3.33), so the GGUF lane instead takes
# DEADLINE-CONDITIONAL STREAMING inside _otr_gguf_backend -- with no deadline
# registered it makes the exact same non-streaming call it always did, and
# with one it streams and stops between chunks. The backend call-time imports
# get_generation_deadline() from here; the state stays owned in this module.
#
# WHY IT STOPPED BEING THEORETICAL: six committed status="shipping" profiles
# (otr_g4_fastwan/_humo/_ltx_8gb/_ltx_audio_in/_ltx_video/_wan_ti2v) pin
# technical_model to unsloth/gemma-4-12b-it-GGUF, and profile status is
# validated but is NOT an application gate -- so real shipping runs reach this
# lane. The default unprofiled canonical run does not (its technical slot is
# the transformers gemma-4-12b row), which is why this looked latent at first.
#
# WHAT A DEADLINE STILL CANNOT INTERRUPT, on EITHER lane: prompt evaluation.
# The criterion/stream is only consulted per GENERATED token. It also cannot
# interrupt the model LOAD -- which is why the worker checks for an
# already-expired deadline BEFORE calling fn() at all (story_orchestrator),
# since request_slot runs inside the timed worker and a cold ~12 GB GGUF load
# is the realistic way to blow a 65 s budget.
#
# LATCH, don't raise mid-decode -- mirrors the established pattern in
# _otr_decode_guard.py's degeneracy criterion: raising from inside a
# StoppingCriteria skips Transformers' own generate()
# cleanup and arrives at the caller unclassified. So the criterion sets
# ``.hit`` and returns True, letting generate() return normally -- but both
# generate() call sites below then check ``.hit`` and RAISE
# GenerationDeadlineExceededError instead of returning the truncated text as
# a normal result. Without that check, a deadline hit that lands just before
# _run_with_timeout's own future.result(timeout=...) fires would let the
# truncated output race through as a silent SUCCESS -- no cache
# invalidation, no _LLMTimeoutWorkflowPause, an accepted-but-wrong artifact.
_GENERATION_DEADLINE = threading.local()


def set_generation_deadline(deadline: float | None) -> None:
    """Set (or clear, with ``None``) a deadline for THIS thread's
    subsequent generation calls.

    ``deadline`` is a ``time.monotonic()`` value, NOT ``time.time()``.
    Changed 2026-08-25: an epoch clock is not monotonic, so an NTP step or
    a DST-adjacent correction could move a live deadline backwards (expire
    instantly) or forwards (never expire). Every producer and consumer of
    this value reads ``time.monotonic()`` -- see get_generation_deadline().
    """
    _GENERATION_DEADLINE.value = deadline


def get_generation_deadline() -> float | None:
    """This thread's registered ``time.monotonic()`` deadline, or None.

    Public because the GGUF backend needs it and cannot inherit a
    transformers ``StoppingCriteria``. It call-time imports this getter
    rather than the thread-local itself, and the state deliberately stays
    OWNED HERE rather than moving to a leaf module: this repo supports both
    the ``nodes.x`` and bare ``x`` import forms, so a leaf holding the state
    could be instantiated twice, yielding two thread-locals and (if the
    exception moved too) two class identities for
    ``GenerationDeadlineExceededError``. One owner, one identity.
    """
    return getattr(_GENERATION_DEADLINE, "value", None)


def generation_deadline_expired() -> bool:
    """True iff a deadline is registered for this thread AND it has passed."""
    deadline = get_generation_deadline()
    return deadline is not None and time.monotonic() > deadline


class _DeadlineStoppingCriteria:
    """transformers ``StoppingCriteria``: stop once this thread's
    registered deadline (if any) has passed. See the module comment above
    ``set_generation_deadline`` for why this exists.

    Returns a scalar ``bool``, matching ``_otr_decode_guard``'s degeneracy
    criterion (the sibling this file installs alongside it at every local
    ``generate()`` call) rather than a batch-shaped tensor -- consistency
    with the one other criterion in this codebase's StoppingCriteriaList,
    not a new contract for just this one.
    """

    def __init__(self) -> None:
        self.hit = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.hit:
            return True
        # monotonic, not time.time() -- see set_generation_deadline().
        if generation_deadline_expired():
            self.hit = True
        return self.hit


_REMOTE_CACHE_PROVIDERS = frozenset({"openrouter", "comfy_credits", "google_api"})


# ---------------------------------------------------------------------------
# S30 B1b: MODEL_CONTEXT_CAPS static dict + DEFAULT_CONTEXT_CAP constant
# DELETED. Context-cap resolution now goes through
# nodes._otr_model_catalog.resolve_context_cap which returns a tiered
# ContextCapVerdict (PASS for curated overrides, WARN for parsed
# config.json, UNKNOWN for unresolved) and clamps everything against
# HARD_VRAM_CONTEXT_LIMIT.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Error class
# ---------------------------------------------------------------------------


class ModelLoaderError(RuntimeError):
    """Raised when load_llm or make_generate_fn cannot complete.

    Wraps lower-level exceptions from the legacy _load_llm path so
    callers get a stable exception type to catch.
    """


class GenerationDeadlineExceededError(ModelLoaderError):
    """A model.generate() call was cut short by set_generation_deadline().

    Raised AFTER generate() returns -- _DeadlineStoppingCriteria latches
    rather than raising mid-decode, same reason as the degeneracy guard.
    _run_with_timeout is the intended catcher: it routes this the same way
    it routes its own FuturesTimeout (cache invalidation + a workflow-pause
    raise), so a deadline hit can never surface as a truncated "successful"
    result to a caller that never asked for one. Not a capacity/rerollable
    error -- deliberately NOT a _CapacityError subclass -- because the
    retry ladder must not treat a deadline-abandoned generation as an
    ordinary "reroll and try again" case.
    """

    def __init__(
        self, message: str, *, generated_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.generated_tokens = generated_tokens


_GEMMA4_UNIFIED_MODEL_IDS = frozenset({"google/gemma-4-12b-it"})
_GEMMA4_UNIFIED_MIN_TRANSFORMERS = "5.10.4"


def _require_transformers_model_support(
    model_id: str,
    installed_version: str | None = None,
) -> None:
    """Fail early when a selected model needs a newer Transformers build."""
    normalized = str(model_id or "").split(" ", 1)[0].strip()
    if normalized not in _GEMMA4_UNIFIED_MODEL_IDS:
        return
    if installed_version is None:
        import transformers

        installed_version = transformers.__version__
    from packaging.version import InvalidVersion, Version

    try:
        supported = Version(str(installed_version)) >= Version(
            _GEMMA4_UNIFIED_MIN_TRANSFORMERS
        )
    except InvalidVersion:
        supported = False
    if not supported:
        raise ModelLoaderError(
            f"{normalized!r} uses the gemma4_unified architecture and requires "
            f"transformers>={_GEMMA4_UNIFIED_MIN_TRANSFORMERS}; this ComfyUI "
            f"environment has transformers=={installed_version}. Upgrade the "
            "shared ComfyUI venv before loading the model. The retired 5.5 "
            "text-tower remap is not a valid inference path."
        )


# ---------------------------------------------------------------------------
# S0 portability helpers (docs/2026-07-09-platform-portability-final.md)
# ---------------------------------------------------------------------------


def _plan_max_memory(
    model_id: str, total_vram: float, *, cuda_available: bool,
    quant_policy: str,
):
    """VRAM-budget plan for the transformers loader.

    Returns the ``max_memory`` dict for ``from_pretrained`` or ``None``.
    The integer key ``0`` names CUDA device 0, so on a CUDA-less host the
    only honest plan is ``None`` (plain CPU/MPS load). The pre-S0 code
    built the CUDA-keyed dict from model-id string tags alone, handing
    transformers a device map for hardware that does not exist on
    cpu/mps hosts (fresh-install breaker).

    2026-08-25: every budget below (3.2GiB for a 2B-tag, 6.8GiB for a
    9b/12b/e4b/4b-it tag, ``total_vram - 2.5`` above 12 GiB) was sized for a
    4-BIT (NF4) footprint -- the S1 comment above ``load_llm`` records that
    "its resolved value for every production id was NF4" back when these
    numbers were chosen, i.e. a tagged model_id was ALWAYS quantized at the
    time. ``quant_policy="none"`` pairing with a small tagged model (e.g.
    ``otr_4060_floor``'s ``google/gemma-4-E2B-it``) postdates this function
    and was never threaded back in -- so an UNQUANTIZED bf16 load (needing
    roughly 4x a 4-bit footprint) was still being capped at the 4-bit number,
    with the remainder silently CPU-offloaded by ``device_map="auto"``. Live
    symptom on an 8 GB RTX 4060 tonight: the 3-4B ``E2B`` model ran
    noticeably SLOWER than the 12B NF4 model on the same card -- exactly what
    partial CPU offload looks like, every forward pass paying PCIe
    round-trips for the CPU-resident layers.

    So this budget applies ONLY to an actually-quantized load. An
    unquantized (``quant_policy == "none"``) request returns ``None`` here,
    unconditionally, before any ``total_vram``/tag branching -- no cap, no
    artificial CPU-offload escape hatch. The model either fits fully on GPU
    (the normal case for a profile that chose a small model BECAUSE it chose
    no quantization) or ``model = model.to(device)`` (the existing
    ``quant_config is None and max_memory is None`` branch, already
    exercised today by every non-tagged model_id) raises a fast, honest CUDA
    OOM instead of a silent multi-minute-per-line render that looks hung.
    """
    if not cuda_available:
        return None
    if quant_policy not in ("bnb_nf4", "bnb_8bit"):
        return None
    sid = (model_id or "").lower()
    is_actually_2b = any(tag in sid for tag in ("2b-it", "2b_it")) or sid.endswith("2b")
    if total_vram >= 12.0:
        return {0: f"{total_vram - 2.5:.1f}GiB", "cpu": "32GiB"}
    if is_actually_2b:
        return {0: "3.2GiB", "cpu": "32GiB"}
    if any(tag in sid for tag in ("9b", "12b", "e4b", "4b-it")):
        return {0: "6.8GiB", "cpu": "32GiB"}
    return None


def _bug098_scan_linear4bit_devices(model) -> tuple[int, list[str]]:
    """Model-local BUG-LOCAL-098 check: for every ``bitsandbytes.Linear4bit``
    module in ``model``, is its actual weight tensor on a CUDA device?

    Returns ``(linear4bit_count, off_cuda_module_paths)``.
    ``linear4bit_count == -1`` means the scan itself raised (e.g. an exotic
    module tree that breaks ``named_modules()``) -- callers treat that as
    "could not verify", not as "verified fine", and fall through to the
    ``is_loaded_in_4bit`` flag as the remaining signal.

    2026-08-25 (PBUG-20260825-04): this REPLACES a process-global
    ``torch.cuda.memory_allocated()`` delta as the correctness predicate.
    That counter reflects the WHOLE process, not this call's own
    allocation -- a concurrent orphan worker (an abandoned
    NewsCuration/NewsCurationDeep timeout thread, left running because
    generation is not cancellable mid-token -- see
    ``story_orchestrator._run_with_timeout``) freeing its own tensors in
    the same wall-clock window can produce a negative or near-zero NET
    delta even when THIS load fully succeeded. Confirmed live on an 8 GB
    RTX 4060: ``linear4bit_count=592``, ``is_loaded_in_4bit=True``, and
    every actual weight tensor really was on CUDA -- yet the old delta
    check (``delta >= 0.0``) saw ``vram_delta=-0.00GiB`` and killed a
    working load. Asking the model directly is immune to whatever else the
    process's allocator is doing concurrently.
    """
    count = 0
    off_cuda: list[str] = []
    try:
        for mod_path, m in model.named_modules():
            cls_name = type(m).__name__
            mod_module = type(m).__module__ or ""
            if not (cls_name == "Linear4bit"
                    and mod_module.startswith("bitsandbytes")):
                continue
            count += 1
            w = getattr(m, "weight", None)
            wdev = getattr(w, "device", None)
            if wdev is None or wdev.type != "cuda":
                off_cuda.append(f"{mod_path}={wdev}")
    except Exception:  # noqa: BLE001
        return -1, []
    return count, off_cuda


def _apply_matmul_precision_policy() -> None:
    """TF32 OFF for byte-identical determinism (I-2 / C-1); Ampere+ (sm80+)
    gets 'high' matmul precision for LLM throughput. The capability probe is
    GUARDED: ``torch.cuda.get_device_capability()`` raises on a CUDA-less
    host (the S0 loader:257 crash), and the sm80 check only means anything
    on CUDA anyway. The canonical headless launcher additionally exports
    NVIDIA_TF32_OVERRIDE=0 before torch imports; see nodes/_otr_determinism.py."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.set_float32_matmul_precision('high')


# ---------------------------------------------------------------------------
# load_llm -- wraps tuple return into dict
# ---------------------------------------------------------------------------


def load_llm(
    model_id: str,
    *,
    device: str = "cuda",
    optimization_profile: str = "Standard",
    context_cap: int | None = None,
    policy: Any = None,
) -> dict[str, Any]:
    """Load an LLM and return a cache_entry dict.

    The always-load primitive: `request_slot` is the canonical entry
    point that handles cache-hit / cache-miss; `load_llm` builds and
    returns a fresh cache_entry every time it's called. Owns the
    bitsandbytes / NF4 / 8-bit / Standard / Obsidian profile body.

    Returns a cache_entry dict shaped for the v2.0 path:
        {
            "model":        <torch model>,
            "tokenizer":    <tokenizer>,
            "model_id":     <canonical model_id, no UI suffix>,
            "device":       <device string actually placed on>,
            "quantized":    <bool, True for NF4/8-bit profiles>,
            "context_cap":  <int, caller-provided override or resolved
                            via _otr_model_catalog.resolve_context_cap>,
        }

    Args:
        model_id: HF model identifier. UI suffixes like "[BETA]" or
                  "[8-bit]" are tolerated and stripped.
        device:   target device. Defaults "cuda".
        optimization_profile: one of "Standard", "Obsidian", "8-bit".
        context_cap: optional caller-provided context cap. `request_slot`
                  pre-resolves via `_otr_model_catalog.resolve_context_cap`
                  (tiered ContextCapVerdict) and forwards the resolved value
                  through. Defaults to None, in which case load_llm resolves
                  through the SAME catalog function -- the single source of
                  truth for every load path.

    Raises ModelLoaderError on any underlying failure (wraps the
    original exception via __cause__).
    """
    from .story_orchestrator import _runtime_log

    try:
        # `model_id_full` is the in-body name for the same value as
        # the caller-facing `model_id` argument.
        model_id_full = model_id

        # Strip [BETA] or [8-bit] labels used in the UI dropdown
        _stripped_model_id = model_id_full.split(" ")[0]

        # S1 platform-portability (2026-07-10): resolve the EXPLICIT runtime
        # policy (None = the nv50 16 GB baseline -- identical resolved
        # values to the deleted auto machinery below). policy.device wins
        # over the legacy `device` kwarg: one source of truth.
        from ._otr_shared.llm_policy import BASELINE_POLICY
        _policy = policy if policy is not None else BASELINE_POLICY
        device = _policy.device

        # S1: quantization is an EXPLICIT policy field. The legacy tag
        # predicate (Obsidian profile + "4-bit"/"9b"/"12b"/"nemo"/...
        # model-id substrings) is DELETED -- its resolved value for every
        # production id was NF4, which is exactly the policy default.
        requested_quantized = _policy.quant_policy in ("bnb_nf4", "bnb_8bit")

        # 2026-07-19: context-cap resolution is the catalog's SINGLE source of
        # truth -- _otr_model_catalog.resolve_context_cap (tiered
        # PASS/WARN/UNKNOWN; an authoritative soak-tested override for a
        # vram_fit_tier=="PASS" row, otherwise clamped to
        # HARD_VRAM_CONTEXT_LIMIT). request_slot already passes the resolved
        # value in via `context_cap`; a direct/legacy caller that reaches
        # load_llm WITHOUT it (e.g. the _LegacyTransformersBackendBase
        # delegate in _otr_model_runtime) now resolves through the SAME path
        # instead of a stale hardcoded table. This completes the S30 B1b
        # migration that already deleted the module-level MODEL_CONTEXT_CAPS:
        # a duplicated function-local table is exactly how Mistral-Nemo would
        # load at a stale 8192 on the no-cap path while request_slot loaded
        # 16384 (the BUG-LOCAL-101 lineage -- Mistral was 16384, dropped to
        # 8192 in S21.2 for audio co-residency; the 420/720w script pass needs
        # 16384 back and NF4 + a DynamicCache make it fit the 16 GB card).
        _resolved_id = str(model_id_full).split(" ", 1)[0].strip()
        if context_cap is not None:
            _cap = int(context_cap)
        else:
            from . import _otr_model_catalog as _otr_catalog
            _cap = int(_otr_catalog.resolve_context_cap(_resolved_id).value)

        log.info(f"Loading LLM model: {_stripped_model_id} (quantized={requested_quantized})")

        # Lazy import - only pay the cost when actually generating
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _require_transformers_model_support(
            _stripped_model_id, transformers.__version__,
        )

        # -- Zero-Prime VRAM Hardening (v1.4) --
        # Detect hardware and purge memory BEFORE loading even the
        # tokenizer to prevent the 15GB transient spike on 16GB cards.
        total_vram = 0
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # 2026-04-30: Sync BEFORE eviction. bnb-NF4 + Blackwell sm_120 +
        # CUDA 13 surfaces cudaErrorUnknown on the post-eviction call
        # when async kernel completions from the prior generation are
        # still in flight while empty_cache() touches their memory.
        # Triple-confirmed by 2026-04-30 round-robin. See
        # docs/2026-04-30-spine-cuda-crash/.
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as _sync_err:  # noqa: BLE001
                _runtime_log(
                    f"VRAM_RESET: pre-evict synchronize() failed "
                    f"({_sync_err}); proceeding anyway"
                )

        # Nuclear Power Wash (Global Eviction)
        try:
            import comfy.model_management
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            _runtime_log("[StoryOrchestrator] Zero-Prime: ComfyUI Models Evicted.")
        except Exception as evict_err:  # noqa: BLE001 -- eviction is best-effort
            # NAMED, never swallowed. A failed global eviction leaves ComfyUI's
            # models resident, and on a 16 GB card that is the usual cause of an
            # OOM several stages later -- which used to be undiagnosable because
            # a failed wash and a successful one produced identical logs.
            # Still non-fatal: the gc/empty_cache/ipc_collect below reclaim what
            # they can, and a reset attempt must not abort the episode.
            _runtime_log(
                f"[StoryOrchestrator] Zero-Prime: ComfyUI model eviction FAILED "
                f"({type(evict_err).__name__}: {evict_err}); models may remain "
                f"resident -- suspect this first if a later stage OOMs.")

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Post-Wash Analytics
        if torch.cuda.is_available():
            free_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / (1024**3)
            _runtime_log(f"[StoryOrchestrator] Zero-Prime VRAM State: {free_gb:.1f}GB Free. Capacity: {total_vram:.1f}GB")

        # -- VRAM Budgeting (Early Allocation) --
        # S0 portability: keyed on the live backend -- the CUDA-device-0 plan
        # is only built when CUDA exists (see _plan_max_memory).
        max_memory = _plan_max_memory(
            _stripped_model_id, total_vram,
            cuda_available=torch.cuda.is_available(),
            quant_policy=_policy.quant_policy)
        if max_memory is not None and total_vram >= 12.0:
            _runtime_log(f"[StoryOrchestrator] Sovereignty Buffer Active: {total_vram - 2.5:.1f}GB Budget")

        # TF32 off + Ampere+ matmul precision (guarded off-CUDA; S0 fix for
        # the unguarded get_device_capability crash on cpu/mps hosts).
        _apply_matmul_precision_policy()

        # -- VRAM Hardening v1.4: Strict Handoff --
        try:
            from ._otr_bark_lib import _unload_bark
            _unload_bark()
        except ImportError:
            pass
        except Exception as handoff_err:
            log.warning("[StoryOrchestrator] Bark handoff failed: %s", handoff_err)

        # BUG-LOCAL-109 (2026-05-05) defensive guard: refuse the
        # canonical "auto" sentinel; caller must resolve to a concrete
        # model_id first.
        _mid_lower = (str(_stripped_model_id) or "").strip().lower()
        if not _mid_lower or _mid_lower.startswith("auto"):
            raise RuntimeError(
                f"load_llm: refusing to load model_id={_stripped_model_id!r} -- "
                "the 'auto (use story model)' sentinel must be resolved "
                "by the caller before load_llm is reached. See BUG-LOCAL-109."
            )

        load_dtype = torch.bfloat16

        # S1: attention implementation is an EXPLICIT policy field. The FA2
        # auto-probe (distribution('flash-attn') + import) is DELETED -- on
        # the Blackwell sm_120 / Windows / torch 2.10 baseline it always
        # resolved to sdpa, which is the policy default. An FA2 wheel
        # appearing later is honoured by setting llm_attn_impl explicitly,
        # not by a probe. Still the single source of truth for
        # `attn_implementation` in common_kwargs, logged on every load.
        attn_impl = _policy.attn_impl
        _runtime_log(f"[StoryOrchestrator] Attention selector (policy): attn_implementation={attn_impl}")

        # 4-bit / 8-bit quantization -- EXPLICIT policy, no model-id tag
        # magic (S1; the "2bit"/"3bit" wing-ding upgrade + vram_safe_tags
        # predicate are deleted with it). bitsandbytes missing while the
        # policy requires it is a HARD FAIL: silently proceeding at
        # bfloat16 OOMs a 16 GB card at ~24 GiB -- the exact fallback
        # class BUG-LOCAL-098 exists to catch.
        quant_config = None
        needs_8bit = _policy.quant_policy == "bnb_8bit"
        needs_4bit = _policy.quant_policy == "bnb_nf4"

        if needs_8bit or needs_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as _bnb_err:
                raise ModelLoaderError(
                    f"llm.quant_policy={_policy.quant_policy!r} requires "
                    "bitsandbytes, which is not importable on this host. "
                    "Set llm.quant_policy='none' in the platform profile "
                    "(bnb lanes are OFF on ROCm/MPS/CPU tiers) or install "
                    "bitsandbytes. NO silent bf16 fallback."
                ) from _bnb_err
        if needs_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            log.info("[StoryOrchestrator] Enabling 8-bit quantization (policy)")
        elif needs_4bit:
            # BUG-LOCAL-098: instantiate BitsAndBytesConfig FRESH per
            # call. transformers mutates internal flags during
            # from_pretrained; a reused instance silently skips
            # quantization on the second call -> fp16 fallback -> OOM at
            # 24 GiB on 16 GiB GPU.
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            log.info("[StoryOrchestrator] Enabling 4-bit quantization (NF4, policy)")
            _runtime_log("[StoryOrchestrator] 4-bit NF4 active (policy)")

        from transformers import AutoTokenizer, AutoModelForCausalLM

        # BUG-LOCAL-085 fix: resolve HF_HOME from HKCU\Environment so
        # cache_dir is correct even when ComfyUI Desktop's process
        # didn't inherit User-scope env vars.
        try:
            from . import _otr_hf_env as _OTR_HF
            _hf_home_resolved = _OTR_HF.ensure_hf_home()
            _runtime_log(f"[StoryOrchestrator] HF_HOME resolved -> {_hf_home_resolved}")
        except Exception as _hf_err:
            _runtime_log(f"[StoryOrchestrator] HF_HOME helper unavailable ({_hf_err}); using os.environ fallback")
            _OTR_HF = None
            _hf_home_resolved = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

        cache_dir_path = os.path.join(_hf_home_resolved, "hub")

        # Try snapshot path first (preferred for sharded models on Windows).
        snapshot_path = None
        if _OTR_HF is not None:
            try:
                snapshot_path = _OTR_HF.resolve_snapshot_dir(_stripped_model_id, hf_home=_hf_home_resolved)
            except Exception as _snap_err:
                _runtime_log(f"[StoryOrchestrator] snapshot resolve failed ({_snap_err}); using model_id fallback")
        load_target = snapshot_path or _stripped_model_id
        if snapshot_path:
            _runtime_log(f"[StoryOrchestrator] Loading from canonical snapshot: {snapshot_path}")
        else:
            _runtime_log(f"[StoryOrchestrator] Snapshot not found in cache; falling back to model_id with cache_dir")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                load_target,
                local_files_only=True,
                trust_remote_code=False,
                cache_dir=cache_dir_path,
            )
            _runtime_log("LLM tokenizer loaded from cache (no HTTP checks)")
        except Exception as local_err:
            raise ModelLoaderError(
                f"Failed to load tokenizer for {_stripped_model_id!r} from "
                f"the local HF cache at {cache_dir_path!r}; OTR does not use "
                "a network fallback inside load_llm. Ensure the complete "
                "snapshot is under C:\\ComfyUI-Models."
            ) from local_err

        # Gemma 4 12B is cached as two HF revisions on the production box:
        # the complete weighted revision predates chat_template.jinja, while
        # refs/main points at a newer metadata-only revision. Keep model/config
        # coherent with the weighted snapshot and attach only the newer local
        # template when the weighted tokenizer has none. No file is downloaded
        # and no synthetic overlay/LoRA is required.
        if not getattr(tokenizer, "chat_template", None) and _OTR_HF is not None:
            try:
                _template_path = _OTR_HF.resolve_snapshot_file(
                    _stripped_model_id,
                    "chat_template.jinja",
                    hf_home=_hf_home_resolved,
                )
                if _template_path:
                    tokenizer.chat_template = Path(_template_path).read_text(
                        encoding="utf-8",
                    )
                    _runtime_log(
                        "[StoryOrchestrator] Attached chat_template.jinja "
                        f"from local cached metadata: {_template_path}"
                    )
            except Exception as _template_err:
                raise ModelLoaderError(
                    f"Failed to compose the local tokenizer metadata for "
                    f"{_stripped_model_id!r}: {_template_err}"
                ) from _template_err

        common_kwargs = dict(
            cache_dir=cache_dir_path,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            torch_dtype=load_dtype,
            # Consume the resolved attention selector (above). On the
            # Blackwell sm_120 / Windows / torch 2.10 stack `attn_impl`
            # is always "sdpa" -- FA2 has no prebuilt wheel -- but the
            # selector stays the single source of truth so an FA2 wheel
            # appearing later is honoured without a second edit here.
            attn_implementation=attn_impl,
        )

        if max_memory is not None:
            common_kwargs["max_memory"] = max_memory
            common_kwargs["device_map"] = "auto"

        if quant_config is not None:
            common_kwargs["quantization_config"] = quant_config
            # Flagship Sovereignty: force 100% GPU on 14.5+ GiB cards.
            if total_vram >= 14.5:
                common_kwargs["device_map"] = {"": 0}
                _runtime_log(
                    f"[StoryOrchestrator] Flagship Sovereignty: "
                    f"Forcing 100% GPU for {_stripped_model_id} "
                    f"(total_vram={total_vram:.2f} GiB)"
                )
            else:
                _runtime_log(
                    f"[StoryOrchestrator] device_map=auto path "
                    f"(total_vram={total_vram:.2f} GiB < 14.5 GiB)"
                )

        try:
            model_config = None
            try:
                from transformers import AutoConfig
                _cfg_kwargs = {
                    "trust_remote_code": False,
                    "local_files_only": True,
                    "cache_dir": cache_dir_path,
                }
                model_config = AutoConfig.from_pretrained(load_target, **_cfg_kwargs)
                if hasattr(model_config, "max_position_embeddings") and model_config.max_position_embeddings > _cap:
                    _runtime_log(f"[StoryOrchestrator] Hardening: Capping 128k context to {_cap} (Saves ~6GB VRAM)")
                    model_config.max_position_embeddings = _cap
            except Exception as _cfg_err:
                log.warning("[StoryOrchestrator] Config hardening failed: %s", _cfg_err)

            # BUG-LOCAL-098 tripwire setup: measure VRAM before load.
            _bug098_vram_before_gib = (
                torch.cuda.memory_allocated() / (1024 ** 3)
                if torch.cuda.is_available() else 0.0
            )

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    load_target,
                    local_files_only=True,
                    config=model_config,
                    **common_kwargs,
                )
            except ValueError as _dispatch_err:
                # Operator directive 2026-08-29: guards do not kill a render;
                # an OOM is the only killer. bnb-4bit's validate_environment
                # REFUSES a load whose device_map plans any CPU module unless
                # fp32 CPU offload is explicitly permitted -- so a 12B NF4 on
                # an 8 GB card died on a REFUSAL, not on memory. The 8-bit
                # branch above already passes llm_int8_enable_fp32_cpu_offload
                # (the flag covers 4-bit too, despite the int8 name); extend
                # the same permission here as a LOUD one-shot retry: GPU-fit
                # modules stay NF4 on CUDA, overflow runs fp32 on CPU --
                # slower per token, but it LOADS, and only a genuine CUDA OOM
                # can still end it. Anything but this exact refusal re-raises.
                if not (needs_4bit
                        and "dispatched on the cpu" in str(_dispatch_err).lower()):
                    raise
                log.warning(
                    "[StoryOrchestrator] %s exceeds the GPU budget for a full "
                    "NF4 load (%s); RETRYING with fp32 CPU offload -- "
                    "CPU-resident layers run at fp32, expect a slower writer. "
                    "Per operator directive the only remaining killer is a "
                    "real OOM.", _stripped_model_id, _dispatch_err)
                _runtime_log(
                    f"[StoryOrchestrator] NF4 CPU-offload retry active for "
                    f"{_stripped_model_id} (loud degradation, not a kill)")
                # BUG-LOCAL-098: fresh BitsAndBytesConfig for the retry --
                # transformers mutates the instance during from_pretrained.
                _offload_quant = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=True,
                )
                _retry_kwargs = dict(common_kwargs)
                _retry_kwargs["quantization_config"] = _offload_quant
                model = AutoModelForCausalLM.from_pretrained(
                    load_target,
                    local_files_only=True,
                    config=model_config,
                    **_retry_kwargs,
                )
            _runtime_log(
                f"LLM model loaded from "
                f"{'canonical snapshot' if snapshot_path else 'model_id with cache_dir'} "
                f"(no HTTP checks)"
            )

            # BUG-LOCAL-098 tripwire: fail loud if NF4 silently dropped to
            # fp16. See _bug098_scan_linear4bit_devices's docstring for why
            # this checks the model directly rather than a VRAM delta.
            if quant_config is not None and torch.cuda.is_available():
                _bug098_vram_after_gib = torch.cuda.memory_allocated() / (1024 ** 3)
                _bug098_delta_gib = (
                    _bug098_vram_after_gib - _bug098_vram_before_gib
                )
                _bug098_linear4bit_count, _bug098_off_cuda = (
                    _bug098_scan_linear4bit_devices(model)
                )
                _bug098_is_loaded_in_4bit = bool(
                    getattr(model, "is_loaded_in_4bit", False)
                )
                _bug098_module_signal = (
                    _bug098_linear4bit_count > 0
                    or _bug098_is_loaded_in_4bit
                )
                # Vacuously true when linear4bit_count <= 0 (nothing found
                # to check) -- _bug098_module_signal is what catches that
                # case; this signal only speaks to modules it actually saw.
                _bug098_materialized = not _bug098_off_cuda
                _runtime_log(
                    f"[BUG-098 tripwire] post-load: "
                    f"linear4bit_count={_bug098_linear4bit_count} "
                    f"is_loaded_in_4bit={_bug098_is_loaded_in_4bit} "
                    f"materialized_on_cuda={_bug098_materialized} "
                    f"vram_delta={_bug098_delta_gib:.2f}GiB (telemetry only)"
                )
                if not _bug098_module_signal or not _bug098_materialized:
                    try:
                        model.cpu()
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        del model
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        import gc as _bug098_gc
                        _bug098_gc.collect()
                        torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001
                        pass
                    raise RuntimeError(
                        f"BUG-LOCAL-098: NF4 quantized load did not "
                        f"materialize for {_stripped_model_id!r}. "
                        f"linear4bit_count={_bug098_linear4bit_count} "
                        f"is_loaded_in_4bit={_bug098_is_loaded_in_4bit} "
                        f"off_cuda_modules={_bug098_off_cuda[:5]!r}"
                        f"{'...' if len(_bug098_off_cuda) > 5 else ''} "
                        f"vram_delta={_bug098_delta_gib:.2f}GiB (telemetry). "
                        f"This is the bitsandbytes second-load silent "
                        f"fp16 fallback. Workaround: restart ComfyUI "
                        f"Desktop and re-queue. Tracked as BUG-LOCAL-098."
                    )
        except (OSError, ValueError) as local_err:
            # Name the REAL cause -- this wrap spent a night labeling a bnb
            # device-map refusal as a cache problem, which sent diagnosis to
            # the wrong place three times before anyone read the chained
            # traceback.
            raise ModelLoaderError(
                f"Failed to load LLM model {_stripped_model_id!r} (local HF "
                f"cache at {cache_dir_path!r}; OTR does not use a network "
                f"fallback inside load_llm). Underlying error: "
                f"{type(local_err).__name__}: {str(local_err)[:400]}"
            ) from local_err

        if quant_config is None and max_memory is None:
            model = model.to(device)
        model = model.eval()

        actual_quant = (quant_config is not None)
        _runtime_log(f"LLM loaded: {_stripped_model_id} (quantized={actual_quant}, budget={optimization_profile}) [v1.5]")

        # CUDA kernel warmup -- absorbs the 30-60s JIT compile cost
        # for SDPA + BitsAndBytes 4-bit on Blackwell.
        try:
            _warmup_start = time.time()
            _runtime_log("WARMUP: Starting 1-token CUDA kernel warmup...")
            _warmup_ids = tokenizer("Test.", return_tensors="pt")["input_ids"].to(model.device)
            with torch.no_grad():
                model.generate(
                    _warmup_ids,
                    max_new_tokens=1,
                    do_sample=False,
                )
            del _warmup_ids
            torch.cuda.empty_cache()
            _warmup_sec = time.time() - _warmup_start
            _runtime_log(f"WARMUP: CUDA kernels compiled in {_warmup_sec:.1f}s - generation will start instantly")
            log.info("[StoryOrchestrator] CUDA warmup complete (%.1fs) - first generate will not stall", _warmup_sec)
        except Exception as _warmup_err:
            log.warning("[StoryOrchestrator] CUDA warmup failed (non-fatal): %s", _warmup_err)
            _runtime_log(f"WARMUP: Failed (non-fatal): {_warmup_err}")

        return {
            "model":       model,
            "tokenizer":   tokenizer,
            "model_id":    _stripped_model_id,
            "device":      device,
            "quantized":   actual_quant,
            "context_cap": _cap,
        }
    except ModelLoaderError:
        raise
    except Exception as exc:  # noqa: BLE001
        # Sprint H Commit B1 layer 2 (2026-05-17): inner failure-path
        # cleanup. AutoModelForCausalLM.from_pretrained, the warmup
        # generate pass, the BUG-LOCAL-098 tripwire, and the post-load
        # .to(device) call all run AFTER `model` is bound to GPU
        # weights but BEFORE this function returns. If any of them
        # raise, the cache_entry never gets stored in LLM_CACHE, so
        # downstream `unload_llm()` (which reads LLM_CACHE) can't find
        # the orphan to drop. The retry in
        # `_otr_style_picker._run_inventor` then cache-misses and a
        # second copy gets loaded on top of the orphan -> "Currently
        # allocated 29.97 GiB" OOM seen in Sprint H iter 1 logs.
        #
        # Pair with the layer-1 wrapper in `request_slot()`: layer 1
        # catches load_llm raising as a whole; layer 2 catches in-body
        # failures so the orphan is dropped at first opportunity.
        # Belt-and-braces -- both layers are idempotent.
        try:
            _orphan = locals().get("model")
            if _orphan is not None and hasattr(_orphan, "to"):
                try:
                    _orphan.to("cpu")
                except Exception:  # noqa: BLE001
                    pass
            try:
                del _orphan
            except Exception:  # noqa: BLE001
                pass
            try:
                # Drop the local binding too so gc can reap.
                del model  # noqa: F821
            except Exception:  # noqa: BLE001
                pass
            import gc as _gc
            _gc.collect()
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                    try:
                        _torch.cuda.ipc_collect()
                    except Exception:  # noqa: BLE001
                        pass
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            # Cleanup must never mask the real load failure.
            pass
        raise ModelLoaderError(
            f"load_llm failed for model_id={model_id!r}: {exc}"
        ) from exc


def _teardown_gpu_for_entry(entry: dict | None) -> None:
    """Physical GPU teardown for a detached cache entry (or a no-op for
    ``None``). Shared by ``unload_llm`` (unconditional) and request_slot's
    ownership-checked ``_self_unload`` (conditional) -- both need the exact
    same teardown sequence once they have actually decided to run it.

    Canonical sequence (matches reference_chained_backend_teardown):
        1. model.to("cpu")           -- move weights off the GPU.
        2. del cache_entry           -- drop references so gc can reap.
        3. gc.collect()              -- purge Python-side refs.
        4. torch.cuda.empty_cache()  -- return free blocks to allocator.
        5. torch.cuda.ipc_collect()  -- release inter-process CUDA IPC
                                        handles. CRITICAL when LLM load
                                        follows a video-model run (FLUX/
                                        HuMo/LTX). Without ipc_collect,
                                        the next load_llm can OOM even
                                        when the byte budget fits.
        6. torch.cuda.synchronize()  -- let in-flight ops finish.

    Never raises -- a teardown failure should NOT propagate as a node
    error.
    """
    import gc

    if entry is not None:
        if entry.get("provider") == "gguf_native":
            try:
                from ._otr_gguf_backend import GGUFNativeBackend
                GGUFNativeBackend().unload(entry)
            except Exception as exc:  # noqa: BLE001
                log.debug("[OTR_ModelLoader] GGUF unload failed: %s", exc)
        model = entry.get("model")
        if (
            entry.get("provider") != "gguf_native"
            and model is not None
            and hasattr(model, "to")
        ):
            try:
                model.to("cpu")
            except Exception as exc:  # noqa: BLE001
                log.debug("[OTR_ModelLoader] model.to(cpu) failed: %s", exc)
    del entry
    gc.collect()

    try:
        import torch  # noqa: F401

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception as exc:  # noqa: BLE001
                log.debug("[OTR_ModelLoader] ipc_collect skipped: %s", exc)
            try:
                torch.cuda.synchronize()
            except Exception as exc:  # noqa: BLE001
                log.debug("[OTR_ModelLoader] synchronize skipped: %s", exc)
    except ImportError:
        pass


def unload_llm() -> int:
    """Full, UNCONDITIONAL VRAM teardown for cross-model slot transitions.

    Returns the new cache epoch (see _CACHE_EPOCH_LOCK); most external
    callers ignore it. This is the authoritative-invalidator shape used by
    every caller OUTSIDE request_slot's own control flow (cast_lock.py,
    _otr_bark_lib.py, _otr_model_runtime.py, _otr_vram_levers.py) -- always
    tears down whatever is currently resident and always bumps. NOT used
    by request_slot's own self-triggered transitions; see ``_self_unload``
    for why those need a conditional (ownership-checked) claim instead.

    Also tears down story_orchestrator's legacy LLM stack as a
    best-effort fallback. The orchestrator's `_LLM_CACHE` dict + its
    `_load_llm` body remain alive as the underlying implementation
    layer (this loader's `load_llm` still delegates back to them);
    the teardown ensures both surfaces are quiesced together.

    S30 B4b: the three production importers (batch_bark_generator,
    _otr_bark_lib, scene_sequencer) now import this `unload_llm`
    directly rather than the orchestrator's `_unload_llm` (the
    audit-miss BUG-LOCAL-226 fix). Story orchestrator's
    `_generate_with_llm` also routes through `request_slot("technical",
    ...)` to acquire its cache_entry; the RSS news path no longer
    holds a parallel reference to the legacy cache.
    """
    entry, new_epoch = _detach_and_invalidate_locked()
    _teardown_gpu_for_entry(entry)
    return new_epoch


def has_local_resident_llm() -> bool:
    """True when the singleton cache currently owns a local LLM resource.

    Remote OpenRouter / Comfy Credits requests deliberately do not populate
    ``LLM_CACHE``. If a provider-tagged remote entry is ever present, it still
    carries no weights and must not trigger the local torch/CUDA teardown path.

    This does NOT make orphan GPU occupancy visible -- it still reads
    "nothing resident" the instant a timeout invalidates the cache dict,
    even if the orphan worker is still actively on GPU. That is the
    deferred orphan-occupancy registry (PBUG-20260825-04), not this
    function's job. What this DOES fix (r4 kibitz, Cursor): the read is
    now inside ``_CACHE_EPOCH_LOCK`` -- ``_detach_and_invalidate_locked``'s
    clear() and update() are two separate statements even though both run
    under the lock, so an unlocked read here could still observe a
    momentarily-cleared dict mid-invalidation. Reading under the same lock
    closes that torn-read window without touching the larger deferred
    question.
    """
    with _CACHE_EPOCH_LOCK:
        entry = LLM_CACHE.get("cache_entry")
        if entry is None:
            return False
        if isinstance(entry, dict) and entry.get("provider") in _REMOTE_CACHE_PROVIDERS:
            return False
        return True


def unload_llm_if_local_resident() -> bool:
    """Unload only when a local LLM is actually resident.

    Returns True when ``unload_llm()`` was called. Handoff callers use this to
    keep all-cloud LLM runs from importing torch just to clear an empty local
    allocator, while preserving ``unload_llm()`` for real local teardown and
    load-failure orphan cleanup.
    """
    if not has_local_resident_llm():
        return False
    unload_llm()
    return True



def invalidate_cache_no_gpu_teardown() -> int:
    """Clear LLM_CACHE dict references WITHOUT touching the GPU.

    Returns the new cache epoch (see _CACHE_EPOCH_LOCK); most callers
    (this is the EXTERNAL-invalidation path, called by
    story_orchestrator._run_with_timeout on an orphaned worker, never by
    request_slot's own control flow) ignore it.

    Use case: timeout recovery when an orphan worker thread may
    still be executing CUDA kernels on the cached model. Calling
    `unload_llm()` here would race the active kernel: `model.to("cpu")`
    moves weights mid-write and `torch.cuda.empty_cache()` can
    deallocate memory the kernel is still reading from -- both trigger
    `cudaErrorIllegalAddress`.

    The orphan thread's stack frame holds the model reference and the
    generate loop continues to completion on its own references. Once
    the orphan exits naturally, GC + a subsequent clean `unload_llm`
    call (when the next `request_slot` loads a different model)
    handles cleanup safely.

    NOT a general-purpose helper -- only use in code paths where GPU
    teardown is unsafe (timeout recovery, signal handlers).

    S31 B4 (2026-05-14): fixes the TIMEOUT_RECOVERY CUDA-race
    regression introduced at S30 B4b. S30 B4b rewired
    `story_orchestrator._run_with_timeout` to call `unload_llm()`
    on timeout -- but the comment ("avoids cudaErrorIllegalAddress
    from orphan worker still on GPU") was structurally wrong: the
    new behavior actively CAUSES that error. Pre-B4b the path was
    dict-invalidation-only (safe). S31 B4 reverts to safe semantics
    via this helper. See BUG-LOCAL-228 for the regression log.
    """
    _entry, new_epoch = _detach_and_invalidate_locked()
    return new_epoch


def _assert_policy_admits_vram(
    model_id: str, ctx_verdict: Any, policy: Any,
) -> None:
    """THE policy-admission calculation: may THIS request load this model
    under the ceiling it arrived with?

    Called once per request, before BOTH cache reuse and loading, for every
    LOCAL lane. ``vram_ceiling_gb == 0`` disables the gate (cpu tier -- there
    is no VRAM to fit). Remote lanes never reach it: they return earlier and
    use zero local VRAM.

    Why it does not live inside either loader branch (A1, 2026-07-27):
    neither ``LLMRuntimePolicy.cache_key()`` nor ``GGUFLoadConfig.reuse_key()``
    carries the ceiling, and both are RIGHT not to -- admission does not shape
    the loaded artifact. But that means a resident model is reused on load
    IDENTITY alone, so a model admitted under a permissive ceiling used to
    satisfy a stricter-ceiling request by cache hit. Gating the load alone
    cannot close that: the question is asked of the REQUEST, not of the cache.
    This mirrors the lane backstop, which was already placed correctly -- the
    two admission fields now get the same treatment.

    ONE estimator serves every local lane. ``check_vram_fit`` already prices a
    ``gguf_native`` row from its pinned on-disk artifact plus that row's KV
    cache (``_otr_model_catalog._estimate_resident_gb``), so the GGUF lane
    never needed a second calculation -- it needed this one to RUN. The GGUF
    backend's own in-load preflight answers a different question ("does this
    box have free VRAM right now") and stays where it is; a physical-free
    probe cannot enforce a tier ceiling on a card that is larger than it.
    """
    from . import _otr_model_catalog as _otr_catalog
    from ._otr_model_inputs import VRAMFitFailedError

    if policy.vram_ceiling_gb <= 0:
        log.info(
            "[Selector] VRAM-fit gate disabled by policy "
            "(vram_ceiling_gb=0, cpu tier)"
        )
        return

    fit_verdict = _otr_catalog.check_vram_fit(
        model_id, ctx_verdict.value, ceiling_gb=policy.vram_ceiling_gb,
    )

    # FAIL escalates BEFORE any cache reuse, network or disk work. A 70B pick
    # on a 16 GB card must not trigger snapshot_download or a disk-space
    # pre-check pass; both waste minutes on a doomed-to-OOM load.
    if fit_verdict.tier == "FAIL":
        raise VRAMFitFailedError(
            f"VRAMFitFailedError: {model_id!r}: {fit_verdict.reason}. "
            f"ctx_cap={ctx_verdict.tier}@{ctx_verdict.value}",
            estimated_gb=fit_verdict.estimated_gb,
            ceiling_gb=fit_verdict.ceiling_gb,
        )

    # Combined caution log (everything below PASS/PASS).
    if not (fit_verdict.tier == "PASS" and ctx_verdict.tier == "PASS"):
        log.info(
            "[Selector] proceeding with caution: ctx_cap=%s@%d, "
            "vram_fit=%s@%.1f GB",
            ctx_verdict.tier,
            ctx_verdict.value,
            fit_verdict.tier,
            fit_verdict.estimated_gb,
        )


def _self_unload(my_epoch: int, *, slot: str) -> int:
    """request_slot's own "tear down the resident model before loading a
    different one" step, for all 4 self-triggered transition branches
    (GGUF load-config change, GGUF slot transition, transformers policy
    change, transformers slot transition).

    The DECISION to call this is made from an unlocked read a few lines
    earlier ("a different model is resident") -- if an external
    invalidation (a timeout handler, or an unrelated caller elsewhere)
    races into the gap between that read and this actual call, ``my_epoch``
    is already stale by the time this runs, even though the decision to
    self-unload was made before that. Atomically claims ownership of
    ``my_epoch`` before touching anything: if it still holds, tears the
    resident model down and returns the epoch THIS call's own bump
    produced (legitimate to adopt -- it was this call's own doing). If it
    no longer holds, this is a no-op (no GPU touch, no LLM_CACHE mutation)
    and returns ``my_epoch`` UNCHANGED -- the caller must not adopt a new
    epoch it did not itself cause, or it launders an external invalidation
    into a legitimate-looking self-triggered one and its eventual cache
    store wrongly succeeds.
    """
    entry, new_epoch = _detach_and_invalidate_locked(expected_epoch=my_epoch)
    if new_epoch is None:
        log.warning(
            "[Selector] slot=%s self-triggered teardown found its epoch "
            "snapshot already stale (an external invalidation raced ahead "
            "of it) -- not claiming a new epoch; this call's eventual "
            "cache store will correctly be skipped",
            slot,
        )
        return my_epoch
    _teardown_gpu_for_entry(entry)
    return new_epoch


def request_slot(
    slot: str, model_id: str, policy: Any = None, load_config: Any = None,
) -> dict[str, Any]:
    """Slot-aware entry point. Loads (or reuses cached) LLM, handling
    cache reuse vs full teardown automatically.

    ``load_config`` (GGUF row registry, 2026-07-16): the immutable per-slot
    GGUF load contract resolved by the writer's preflight. When present it is
    the resident-reuse identity (repo_id + resolved path + quant + n_ctx +
    n_batch + n_gpu_layers) and is threaded to the backend load -- no live-env
    rebuild. Ignored for non-GGUF rows.

    B1d order, as CORRECTED by A1 (2026-07-27) -- admission before REUSE,
    not merely before network/disk work:
      1. normalize model_id via catalog.validate_model_id (strips
         [NOT DOWNLOADED] suffix, structural rejection, admit-path check).
      2. Lane backstop, then the REMOTE dispatch (zero local VRAM; returns).
      3. resolve_context_cap(model_id) -> tiered ContextCapVerdict.
      4. _assert_policy_admits_vram -> check_vram_fit against the POLICY
         ceiling; FAIL raises VRAMFitFailedError. Steps 3-4 run for every
         LOCAL lane and run BEFORE every cache read below, because the
         ceiling is deliberately absent from both reuse keys -- so a
         cache hit would otherwise inherit the ceiling of whichever
         request loaded the model first. It also still fires before
         auto_download, so a 70B-on-16GB pick triggers no network pull.
      5. GGUF dispatch: reuse on load identity, else load. Returns.
      6. Transformers cache hit (same model_id + same policy cache_key)
         -> return entry; a mismatched cache_key is a teardown, never
         a silent reuse.
      7. auto_download_if_missing -- gated/disk-space pre-flight +
         snapshot_download. Local-cache short-circuit fires inside the
         catalog helper.
      8. _self_unload() (only if a different model was resident), then
         load_llm(model_id, context_cap=ctx_verdict.value) -- skips the
         second catalog walk by forwarding the resolved cap.
      9. Cache the entry under (slot, model_id).

    `slot` is "creative" or "technical" -- used for log lines + cache
    keying. The cache holds at most one resident model regardless of
    slot; same-slot reuse and cross-slot identity-reuse both return the
    cached entry without a full teardown.
    """
    from . import _otr_model_catalog as _otr_catalog
    from ._otr_shared.llm_policy import BASELINE_POLICY, lane_for_row

    if slot not in ("creative", "technical"):
        raise ModelLoaderError(
            f"request_slot: slot must be 'creative' or 'technical', got {slot!r}"
        )

    # S1 platform-portability (2026-07-10): resolve the explicit runtime
    # policy. None = the nv50 baseline (an API backstop for direct callers;
    # every production call-site threads a real policy).
    if policy is None:
        log.info("[Selector] slot=%s policy=None -> nv50 BASELINE", slot)
    _policy = policy if policy is not None else BASELINE_POLICY

    # Snapshot BEFORE any local work. If this call is abandoned (its owning
    # _run_with_timeout gives up and invalidates LLM_CACHE) but its own
    # load_llm()/backend.load() call later succeeds anyway, the epoch will
    # have moved on by the time it reaches its own cache store below -- see
    # the _CACHE_EPOCH docstring for why this write must then be skipped
    # rather than silently adopted by a later, unrelated caller.
    _my_cache_epoch = _current_cache_epoch()

    # Step 1: normalize.
    normalized = _otr_catalog.validate_model_id(model_id)

    # [OpenRouter S3] Remote branch (FC2 seam 1) -- the dispatch table is
    # otherwise dormant. A virtual catalog row carries
    # loader_backend="openrouter_http"; route it to the remote backend's
    # load(), which returns a provider-tagged cache_entry using ZERO local
    # VRAM. SKIP steps 3-8 (resolve_context_cap, check_vram_fit,
    # auto_download_if_missing, the resident-model teardown, load_llm) and
    # -- critically -- LEAVE any resident local model in LLM_CACHE
    # UNTOUCHED (C2 no-evict). Placed before the Step 2 cache-hit read so
    # a remote request never reads or mutates LLM_CACHE: the common config
    # (creative=remote, technical=local) must not evict + reload the local
    # model across slot transitions. Remote makes zero CUDA / snapshot /
    # download calls.
    # BUG-LOCAL-299: this remote-routing gate shipped recognizing ONLY
    # "openrouter_http". A Comfy Credits row (loader_backend="comfy_credits_http")
    # fell through to the LOCAL path below, so ComfyUI tried to HF-download the
    # virtual handle (e.g. "comfy:slot-a") -> HFValidationError, aborting the run
    # before the lane was ever exercised. Route BOTH remote loader_backends; the
    # _otr_model_runtime dispatch table already maps each key to its backend, so
    # a future remote lane only adds its key to this tuple.
    # Virtual catalog rows must be intercepted before the HF cache/download
    # path below. Remote rows are zero-VRAM and do not disturb a resident
    # local model. The GGUF row is different: it is in-process VRAM and must
    # participate in the singleton cache/teardown discipline.
    _REMOTE_DISPATCH_BACKENDS = ("openrouter_http", "comfy_credits_http", "google_api_http")
    _GGUF_DISPATCH_BACKENDS = ("gguf_native",)
    _virtual_row = _otr_catalog._by_repo_id().get(normalized)

    # S1 runtime lane backstop: the profile's lane_allowlist is enforced at
    # validate/emit time upstream; this is the last line of defense so a
    # hand-crafted workflow cannot smuggle a disallowed lane through. NO
    # FALLBACK: pick an admitted lane or change the platform profile.
    _lane = lane_for_row(_virtual_row)
    if not _policy.admits_lane(_lane):
        raise ModelLoaderError(
            f"request_slot: lane '{_lane}' (model {normalized!r}) is not "
            f"admitted by the profile lane_allowlist "
            f"{list(_policy.lane_allowlist)} -- NO FALLBACK."
        )

    if (
        _virtual_row is not None
        and getattr(_virtual_row, "loader_backend", None) in _REMOTE_DISPATCH_BACKENDS
    ):
        from ._otr_model_runtime import get_backend_for_row
        log.info(
            "[Selector] slot=%s remote-dispatched backend for %s (no local VRAM; "
            "resident local model left in place, no-evict)",
            slot, normalized,
        )
        return get_backend_for_row(_virtual_row).load(
            normalized, _virtual_row, policy=_policy,
        )

    # Steps 3-5, HOISTED (A1, 2026-07-27): the context cap and THE policy
    # admission calculation, once, before every local-lane cache read and
    # before every local-lane load. They used to sit below both cache-hit
    # returns and below the GGUF dispatch, so the ceiling could only ever
    # gate a fresh TRANSFORMERS load -- see _assert_policy_admits_vram for
    # why neither reuse key can carry the ceiling instead.
    ctx_verdict = _otr_catalog.resolve_context_cap(normalized)
    _assert_policy_admits_vram(normalized, ctx_verdict, _policy)

    if (
        _virtual_row is not None
        and getattr(_virtual_row, "loader_backend", None) in _GGUF_DISPATCH_BACKENDS
    ):
        from ._otr_model_runtime import get_backend_for_row
        # Resident-reuse identity for the in-process GGUF singleton. The
        # threaded load_config's reuse_key (repo_id + resolved path + quant +
        # n_ctx + n_batch + n_gpu_layers) is the artifact-shaping identity that
        # policy.cache_key() alone cannot see (it misses the resolved path /
        # n_batch / n_gpu_layers). Without a load_config (direct/legacy caller)
        # fall back to the raw policy key -- the pre-registry behavior.
        _gguf_key = (
            load_config.reuse_key() if load_config is not None
            else _policy.cache_key()
        )
        # A resident model only counts as a hit when it was loaded under the
        # SAME load identity. Silent stale reuse is the bug class this
        # campaign kills. The whole check-then-return is one atomic locked
        # read (_try_cache_hit_locked) -- a separate check-then-return would
        # let an invalidation land between the predicate passing and the
        # return, producing a hit against an entry that no longer exists.
        _hit = _try_cache_hit_locked(normalized, slot, gguf_key=_gguf_key)
        if _hit is not None:
            log.info("[Selector] slot=%s reuse cache for %s", slot, normalized)
            return _hit  # type: ignore[return-value]
        if (
            LLM_CACHE.get("model_id") == normalized
            and LLM_CACHE.get("cache_entry") is not None
        ):
            log.info(
                "[Selector] gguf load-config change for %s (%s -> %s): "
                "full teardown",
                normalized, LLM_CACHE.get("gguf_load_key"), _gguf_key,
            )
            _my_cache_epoch = _self_unload(_my_cache_epoch, slot=slot)
        if LLM_CACHE.get("model_id") not in (None, normalized):
            log.info(
                "[Selector] slot transition: %s -> %s (full teardown)",
                LLM_CACHE.get("model_id"),
                normalized,
            )
            _my_cache_epoch = _self_unload(_my_cache_epoch, slot=slot)
        # r4 kibitz finding (Cursor): the transformers load below is
        # wrapped in try/_self_unload so a load failure after partial GPU
        # allocation doesn't strand orphan VRAM for a cache-miss retry to
        # pile a second copy on top of (Sprint H iter 3); this GGUF load
        # was not, pre-existing this session. Same shape, same reasoning.
        try:
            cache_entry = get_backend_for_row(_virtual_row).load(
                normalized, _virtual_row, policy=_policy, load_config=load_config,
            )
        except Exception:
            log.warning(
                "[Selector] GGUF backend.load() raised for %s; running "
                "self-unload to drop any orphan VRAM before retry",
                normalized,
            )
            try:
                _self_unload(_my_cache_epoch, slot=slot)
            except Exception:  # noqa: BLE001
                log.exception("[Selector] self-unload also raised; continuing")
            raise
        _published = _publish_cache_entry_if_current(_my_cache_epoch, {
            "model_id": normalized,
            "slot": slot,
            "cache_entry": cache_entry,
            "policy_key": _policy.cache_key(),
            "gguf_load_key": _gguf_key,
        })
        if not _published:
            log.warning(
                "[Selector] slot=%s GGUF load for %s completed after this "
                "call was abandoned (cache epoch advanced) -- NOT adopting "
                "into LLM_CACHE; a later caller would otherwise take a "
                "cache hit on a model this orphaned call may still be "
                "using",
                slot, normalized,
            )
        return cache_entry

    # Capability-gate architecture support before cache/download work. A stale
    # ComfyUI venv must not spend time resolving a 23.9 GB model only to fail in
    # AutoConfig with an opaque `gemma4_unified` error.
    _require_transformers_model_support(normalized)

    # Step 2: cache hit on the same model id (regardless of slot) -- policy
    # keyed (S1): a mismatched policy_key is a MISS + teardown, never reuse.
    # Atomic locked check-and-return (_try_cache_hit_locked) -- see the GGUF
    # branch above for why a separate check-then-return is not safe here.
    _hit = _try_cache_hit_locked(normalized, slot, policy_key=_policy.cache_key())
    if _hit is not None:
        log.info("[Selector] slot=%s reuse cache for %s", slot, normalized)
        return _hit  # type: ignore[return-value]
    if LLM_CACHE.get("model_id") == normalized and LLM_CACHE.get("cache_entry") is not None:
        log.info(
            "[Selector] policy change for %s (%s -> %s): full teardown",
            normalized, LLM_CACHE.get("policy_key"), _policy.cache_key(),
        )
        _my_cache_epoch = _self_unload(_my_cache_epoch, slot=slot)

    # Steps 3-6 ran above, before the cache reads -- see the hoist comment.

    # Step 7: ensure on-disk + handle gating / disk-space pre-flight.
    # Local-cache short-circuit (B1d) fires inside this helper when the
    # snapshot is already on disk.
    # Resolve the canonical HF root BEFORE the catalog's local-cache probe.
    # ComfyUI Desktop can inherit a stale/malformed HF_HUB_CACHE; the helper
    # repairs it to <HF_HOME>/hub so the already-present C:\ComfyUI-Models
    # snapshot short-circuits without any network call.
    from pathlib import Path as _Path
    from . import _otr_hf_env as _otr_hf

    _resolved_hf_home = _otr_hf.ensure_hf_home()
    _otr_catalog.auto_download_if_missing(
        normalized,
        hub_root=_Path(_resolved_hf_home) / "hub",
    )

    # Step 8: if a different model is resident, unload it. Then load.
    if LLM_CACHE.get("model_id") not in (None, normalized):
        log.info(
            "[Selector] slot transition: %s -> %s (full teardown)",
            LLM_CACHE.get("model_id"),
            normalized,
        )
        _my_cache_epoch = _self_unload(_my_cache_epoch, slot=slot)

    # Sprint H iter 3 (2026-05-17): orphan-model guard. load_llm may
    # raise AFTER AutoModelForCausalLM.from_pretrained successfully
    # allocates the weights on GPU (e.g. BNB quantization, warmup pass,
    # tripwire). load_llm's OWN inner failure handler already cleans up
    # the local model reference IT was building (see its layer-2 comment);
    # this outer handler is about LLM_CACHE's bookkeeping -- so the retry
    # starts from a clean slate instead of cache-missing and loading a
    # SECOND copy on top of whatever is resident (the "Currently allocated
    # 29.97 GiB" OOM on the retry inside _otr_style_picker._run_inventor's
    # 3-attempt loop).
    #
    # r4 kibitz finding (Cursor): calling the UNCONDITIONAL unload_llm()
    # here (rather than the ownership-checked _self_unload every OTHER
    # self-triggered teardown in this function uses) reopens the exact
    # class of bug r3 just closed, on the failure path instead of the
    # success path. `load_llm()` for a slow/abandoned call can take long
    # enough that a completely different, legitimate caller publishes a
    # fresh resident model in the meantime; an unconditional unload_llm()
    # here would tear THAT model down -- possibly while it is actively in
    # use elsewhere -- and bump the epoch out from under its owner, purely
    # because THIS call's own (unrelated) load happened to fail. Routing
    # through _self_unload makes this a no-op unless _my_cache_epoch is
    # still current, i.e. unless this call still owns whatever is resident.
    try:
        cache_entry = load_llm(
            normalized, context_cap=ctx_verdict.value, policy=_policy,
        )
    except Exception:
        log.warning(
            "[Selector] load_llm raised for %s; running self-unload "
            "to drop any orphan VRAM before retry",
            normalized,
        )
        try:
            _self_unload(_my_cache_epoch, slot=slot)
        except Exception:  # noqa: BLE001
            log.exception("[Selector] self-unload also raised; continuing")
        raise

    # Step 9: cache (policy-keyed, S1). Epoch-guarded -- see the
    # _CACHE_EPOCH docstring: an abandoned call whose load_llm() completes
    # after its owning timeout gave up must NOT publish its cache_entry for
    # a later, unrelated caller to adopt and start a second concurrent
    # generate() on the same model object.
    _published = _publish_cache_entry_if_current(_my_cache_epoch, {
        "model_id": normalized,
        "slot": slot,
        "cache_entry": cache_entry,
        "policy_key": _policy.cache_key(),
    })
    if not _published:
        log.warning(
            "[Selector] slot=%s load for %s completed after this call was "
            "abandoned (cache epoch advanced) -- NOT adopting into "
            "LLM_CACHE; a later caller would otherwise take a cache hit on "
            "a model this orphaned call may still be using",
            slot, normalized,
        )
    return cache_entry


# ---------------------------------------------------------------------------
# make_generate_fn -- chat-template adapter
# ---------------------------------------------------------------------------


def _normalize_messages_for_cache_entry(
    cache_entry: dict[str, Any], messages: list[dict],
) -> list[dict]:
    """BUG-LOCAL-262: fold system messages into the first user turn
    for tokenizers whose chat template rejects the system role.

    Probes the tokenizer once and caches the verdict on the
    cache_entry under `_system_role_supported`, so the probe runs
    once per model residency rather than per generate call (both
    make_generate_fn and make_polish_generate_fn share the entry).
    """
    from . import _otr_loader_backends as _otr_loader_backends

    tokenizer = cache_entry["tokenizer"]
    supported = cache_entry.get("_system_role_supported")
    if supported is None:
        supported = _otr_loader_backends.tokenizer_supports_system_role(
            tokenizer,
        )
        cache_entry["_system_role_supported"] = supported
    if supported:
        return messages
    return _otr_loader_backends.normalize_messages_for_tokenizer(
        tokenizer, messages,
    )


def make_generate_fn(cache_entry: dict[str, Any]):
    """Wrap a cache_entry into the GenerateFn callable.

    Returns a callable matching:
        (messages, *, temperature, max_new_tokens) -> str

    where `messages` is a list[dict] in chat format
    ([{"role": "system", "content": ...}, {"role": "user", "content": ...}])
    and the return is the raw decoded string from the model with the
    prompt prefix removed.

    Generation params hardcoded for the v2.0 path:
        do_sample=True
        top_p=0.92

    Caller controls temperature and max_new_tokens per call.

    Raises ModelLoaderError if the cache_entry is missing required
    keys or if torch is not importable at first call time.
    """
    # [OpenRouter S3] Remote branch (FC2 seam 2). A provider-tagged
    # remote entry has no model/tokenizer; return the remote generate_fn
    # before the local-key check below. Uses zero local VRAM.
    if cache_entry.get("provider") == "openrouter":
        from ._otr_openrouter_backend import make_openrouter_generate_fn
        return make_openrouter_generate_fn(cache_entry)
    # BUG-LOCAL-299: Comfy Credits sibling -- same zero-VRAM remote seam.
    if cache_entry.get("provider") == "comfy_credits":
        from ._otr_comfy_backend import make_comfy_credits_generate_fn
        return make_comfy_credits_generate_fn(cache_entry)
    if cache_entry.get("provider") == "google_api":
        from ._otr_google_api.llm import make_google_api_generate_fn
        return make_google_api_generate_fn(cache_entry)
    # Native GGUF lane: in-process llama-cpp-python, no daemon or port.
    if cache_entry.get("provider") == "gguf_native":
        from ._otr_gguf_backend import make_gguf_generate_fn
        return make_gguf_generate_fn(cache_entry)
    required = {"model", "tokenizer"}
    missing = required - set(cache_entry)
    if missing:
        raise ModelLoaderError(
            f"cache_entry missing required keys: {sorted(missing)}"
        )

    model = cache_entry["model"]
    tokenizer = cache_entry["tokenizer"]

    def generate_fn(messages, *, temperature, max_new_tokens):
        # Lazy torch import. Raised as ModelLoaderError to match the
        # facade's exception contract.
        try:
            import torch
        except ImportError as exc:
            raise ModelLoaderError("torch not available") from exc

        require_full_output = bool(getattr(
            messages, "_otr_require_full_output_budget", False,
        ))
        reserve_remaining = bool(getattr(
            messages, "_otr_reserve_remaining_output_capacity", False,
        ))
        bounded_capacity = reserve_remaining and max_new_tokens is not None
        fail_on_output_limit = bool(getattr(
            messages, "_otr_fail_on_output_limit", False,
        ))
        messages = _normalize_messages_for_cache_entry(cache_entry, messages)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        context_cap = int(cache_entry.get("context_cap") or 8192)
        requested_tokens = context_cap if reserve_remaining else max_new_tokens
        try:
            effective_max_new_tokens = fit_output_tokens(
                requested_tokens,
                context_cap=context_cap,
                prompt_tokens=inputs["input_ids"].shape[1],
                label=f"local model {cache_entry.get('model_id', '<unknown>')}",
                require_full=require_full_output or bounded_capacity,
            )
        except GenerationContextOverflowError as exc:
            # THE TWO LOCAL TRANSPORTS MUST AGREE (2026-08-13). This used to
            # raise a bare ModelLoaderError with NO phase for the exact
            # condition OTR_LedgerScriptWriter raises as a phase-carrying
            # PromptContextOverflowError. The ladder reads the PHASE to decide
            # whether a failure is rerollable, so an identical runaway was
            # rerollable on one transport and terminal on the other, purely
            # from which one the pass happened to take. The phase is read off
            # the error rather than assumed here, so a pre-call refusal that IS
            # retryable cannot be mislabelled by this line.
            raise PromptContextOverflowError(
                str(exc), phase=exc.phase,
            ) from exc
        # THE LIVENESS GUARD (2026-08-13). This transport was unprotected when
        # the guard shipped, because the guard was installed per-WRAPPER in
        # OTR_LedgerScriptWriter instead of at every local generate().
        #
        # Today's live callers all pass small numeric caps (8-300 tokens), so
        # this is prophylaxis, not an emergency. It is worth doing anyway
        # because THIS transport honours reserve-remaining -- the comment
        # below concedes it "can legitimately be handed >14k output tokens" --
        # so the first caller that reserves the window here would reopen the
        # 22-minute runaway with nothing watching it. A guard installed only
        # where the runaway happened to occur is a guard waiting to be missed.
        from transformers import StoppingCriteriaList  # noqa: I001
        try:
            from ._otr_decode_guard import make_degeneracy_criterion
        except ImportError:  # pragma: no cover - flat/standalone import path
            from _otr_decode_guard import (  # type: ignore
                make_degeneracy_criterion,
            )
        _guard = make_degeneracy_criterion(inputs["input_ids"].shape[1])
        _deadline_guard = _DeadlineStoppingCriteria()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=0.92,
                max_new_tokens=effective_max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList(
                    [_guard, _deadline_guard]),
                # Read-only live heartbeat (2026-08-13). A reserve-remaining
                # pass can legitimately be handed >14k output tokens, and with
                # no streamer that is a silent twenty-minute wait whose only
                # signal arrives at the ceiling. None when disabled.
                streamer=_OTRHB.make_streamer(
                    tokenizer,
                    f"llm:{cache_entry.get('model_id', '<unknown>')}"),
            )
        # Strip prompt prefix from decoded output.
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = out[0][prompt_len:]
        if _deadline_guard.hit:
            # Checked FIRST: a deadline hit means this call's owning
            # _run_with_timeout has (or is about to have) given up on it --
            # raising here rather than returning text is what stops a
            # truncated result from racing through as a silent success.
            # See GenerationDeadlineExceededError's docstring.
            log.warning(
                "[OTR.llm] generation deadline exceeded after %s of a "
                "%s-token allowance -- raising instead of returning "
                "truncated text (PBUG-20260825-04)",
                len(generated_ids), effective_max_new_tokens,
            )
            raise GenerationDeadlineExceededError(
                "generation was cut short by its caller's wall-clock "
                "deadline; the caller has abandoned this call",
                generated_tokens=len(generated_ids),
            )
        if getattr(_guard, "hit", False):
            # Classified BEFORE the output-limit check below, for the same
            # reason the writer classifies degeneracy first: a halted decode
            # stops with its allowance unspent, so reporting it as capacity
            # exhaustion would send the next reader hunting a budget defect
            # that does not exist.
            telemetry = _guard.telemetry()
            log.error(
                "[OTR.llm] DECODE HALTED (%s): repeated a %s-token run "
                "verbatim %s times after %s generated tokens of a %s-token "
                "allowance. Telemetry: %s",
                _guard.reason, telemetry.get("cycle_tokens"),
                telemetry.get("required_repeats"), len(generated_ids),
                effective_max_new_tokens, telemetry,
            )
            raise GenerationDegeneracyError(
                "generation was halted by the liveness guard: the output "
                "repeated a run of tokens verbatim rather than progressing",
                halt_reason=_guard.reason,
                repetition=telemetry,
                raw_completion=tokenizer.decode(
                    generated_ids, skip_special_tokens=True,
                ),
                prompt_tokens=prompt_len,
                generated_tokens=len(generated_ids),
                effective_output_tokens=effective_max_new_tokens,
            )
        if fail_on_output_limit and len(generated_ids) >= effective_max_new_tokens:
            raise ModelLoaderError(
                "prose generation exhausted the full remaining provider/context "
                "capacity; the partial artifact is not eligible for reroll"
            )
        return tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

    return generate_fn


# ---------------------------------------------------------------------------
# Polish-specific generate fn (LFC sprint commit 3, section 6.4)
# ---------------------------------------------------------------------------


# Polish-specific sampling baked in per ADR section 6.4. None of these
# are configurable per-call -- the whole point of the dedicated polish
# fn is that the writer's closure-captured composer tuning
# (repetition_penalty, min_p, top_p tweaks) cannot leak in.
_POLISH_TOP_P: float = 0.9
_POLISH_DO_SAMPLE: bool = True


def make_polish_generate_fn(cache_entry: dict[str, Any]):
    """Build a polish-specific generate fn from `cache_entry`.

    LFC sprint commit 3, ADR section 6.4 (2026-05-11). Polish is a
    short, targeted rewrite -- conceptually closer to a constrained
    edit than the composer's long-form generation. The writer's main
    `make_generate_fn` (via the OTR_LedgerScriptWriter
    `_build_truncating_generate_fn` wrapper) bakes
    repetition_penalty / min_p / top_p into its closure tuned for
    composition. Those settings leak into polish via closure capture
    and produce awkward substitutions on short rewrites.

    The polish fn here is a SEPARATE closure off the same cache_entry
    with composer-independent sampling:

        temperature      -- caller-provided per call (defaults to 0.4
                            via _otr_line_composer.polish_line)
        top_p            -- 0.9 (slightly tighter than composer 0.92)
        do_sample        -- True
        min_p            -- not passed (transformers default 0)
        repetition_penalty -- not passed (transformers default 1.0)

    Returns a callable with the same signature as `make_generate_fn`:
        (messages, *, temperature, max_new_tokens) -> str
    """
    # [OpenRouter S3] Remote branch (FC2 seam 2). A provider-tagged
    # remote entry has no model/tokenizer; the remote generate_fn applies
    # the same sampling the caller passes (polish callers pass their own
    # temperature), so one closure covers both factories.
    if cache_entry.get("provider") == "openrouter":
        from ._otr_openrouter_backend import make_openrouter_generate_fn
        return make_openrouter_generate_fn(cache_entry)
    # BUG-LOCAL-299: Comfy Credits sibling -- same zero-VRAM remote seam.
    if cache_entry.get("provider") == "comfy_credits":
        from ._otr_comfy_backend import make_comfy_credits_generate_fn
        return make_comfy_credits_generate_fn(cache_entry)
    if cache_entry.get("provider") == "google_api":
        from ._otr_google_api.llm import make_google_api_generate_fn
        return make_google_api_generate_fn(cache_entry)
    # Native GGUF lane: in-process llama-cpp-python, no daemon or port.
    if cache_entry.get("provider") == "gguf_native":
        from ._otr_gguf_backend import make_gguf_generate_fn
        return make_gguf_generate_fn(cache_entry)
    required = {"model", "tokenizer"}
    missing = required - set(cache_entry)
    if missing:
        raise ModelLoaderError(
            f"cache_entry missing required keys: {sorted(missing)}"
        )

    model = cache_entry["model"]
    tokenizer = cache_entry["tokenizer"]

    def polish_generate_fn(messages, *, temperature, max_new_tokens):
        try:
            import torch
        except ImportError as exc:
            raise ModelLoaderError("torch not available") from exc

        require_full_output = bool(getattr(
            messages, "_otr_require_full_output_budget", False,
        ))
        reserve_remaining = bool(getattr(
            messages, "_otr_reserve_remaining_output_capacity", False,
        ))
        bounded_capacity = reserve_remaining and max_new_tokens is not None
        fail_on_output_limit = bool(getattr(
            messages, "_otr_fail_on_output_limit", False,
        ))
        messages = _normalize_messages_for_cache_entry(cache_entry, messages)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        context_cap = int(cache_entry.get("context_cap") or 8192)
        requested_tokens = context_cap if reserve_remaining else max_new_tokens
        try:
            effective_max_new_tokens = fit_output_tokens(
                requested_tokens,
                context_cap=context_cap,
                prompt_tokens=inputs["input_ids"].shape[1],
                label=f"local polish {cache_entry.get('model_id', '<unknown>')}",
                require_full=require_full_output or bounded_capacity,
            )
        except GenerationContextOverflowError as exc:
            # THE TWO LOCAL TRANSPORTS MUST AGREE (2026-08-13). This used to
            # raise a bare ModelLoaderError with NO phase for the exact
            # condition OTR_LedgerScriptWriter raises as a phase-carrying
            # PromptContextOverflowError. The ladder reads the PHASE to decide
            # whether a failure is rerollable, so an identical runaway was
            # rerollable on one transport and terminal on the other, purely
            # from which one the pass happened to take. The phase is read off
            # the error rather than assumed here, so a pre-call refusal that IS
            # retryable cannot be mislabelled by this line.
            raise PromptContextOverflowError(
                str(exc), phase=exc.phase,
            ) from exc
        from transformers import StoppingCriteriaList  # noqa: I001
        try:
            from ._otr_decode_guard import make_degeneracy_criterion
        except ImportError:  # pragma: no cover - flat/standalone import path
            from _otr_decode_guard import (  # type: ignore
                make_degeneracy_criterion,
            )
        _guard = make_degeneracy_criterion(inputs["input_ids"].shape[1])
        _deadline_guard = _DeadlineStoppingCriteria()

        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=_POLISH_DO_SAMPLE,
                temperature=temperature,
                top_p=_POLISH_TOP_P,
                max_new_tokens=effective_max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList(
                    [_guard, _deadline_guard]),
                streamer=_OTRHB.make_streamer(
                    tokenizer,
                    f"polish:{cache_entry.get('model_id', '<unknown>')}"),
            )
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = out[0][prompt_len:]
        if _deadline_guard.hit:
            log.warning(
                "[OTR.llm] polish generation deadline exceeded after %s of "
                "a %s-token allowance -- raising instead of returning "
                "truncated text (PBUG-20260825-04)",
                len(generated_ids), effective_max_new_tokens,
            )
            raise GenerationDeadlineExceededError(
                "polish generation was cut short by its caller's "
                "wall-clock deadline; the caller has abandoned this call",
                generated_tokens=len(generated_ids),
            )
        if getattr(_guard, "hit", False):
            telemetry = _guard.telemetry()
            log.error(
                "[OTR.llm] POLISH DECODE HALTED (%s): repeated a %s-token run "
                "verbatim %s times after %s generated tokens of a %s-token "
                "allowance. Telemetry: %s",
                _guard.reason, telemetry.get("cycle_tokens"),
                telemetry.get("required_repeats"), len(generated_ids),
                effective_max_new_tokens, telemetry,
            )
            raise GenerationDegeneracyError(
                "polish generation was halted by the liveness guard: the output "
                "repeated a run of tokens verbatim rather than progressing",
                halt_reason=_guard.reason,
                repetition=telemetry,
                raw_completion=tokenizer.decode(
                    generated_ids, skip_special_tokens=True,
                ),
                prompt_tokens=prompt_len,
                generated_tokens=len(generated_ids),
                effective_output_tokens=effective_max_new_tokens,
            )
        if fail_on_output_limit and len(generated_ids) >= effective_max_new_tokens:
            raise ModelLoaderError(
                "prose generation exhausted the full remaining provider/context "
                "capacity; the partial artifact is not eligible for reroll"
            )
        return tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

    return polish_generate_fn


# ---------------------------------------------------------------------------
# Self-test (run as `python nodes/_otr_model_loader.py`)
# ---------------------------------------------------------------------------
