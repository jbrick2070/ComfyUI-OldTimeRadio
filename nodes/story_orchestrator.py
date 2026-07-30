r"""
OTR Orchestrator - Script Writer + Ledger Writer for "SIGNAL LOST"
===================================================================

Two nodes:
  1. LLMScriptWriter - Fetches real daily science news via RSS, feeds
     it to an LLM to generate a full audio drama script. Contemporary
     sci-fi anthology format (Black Mirror / NPR Invisibilia / Arrival).
     News-as-spine: real headlines become the inciting incident,
     extrapolated to dramatic extremes. Includes a hard-science
     epilogue citing real sources (ArXiv, Nature, etc.).

  2. LPL writer + cast helpers - Generates the L3 ledger directly
     (cast, lines, meta.visual_plan, meta.style) via the
     LedgerScriptWriter path. Cast-lock invariants run at writer
     exit; downstream consumers read the ledger via
     FreezeCascade.script_json fanout.

The legacy LLMDirector class was removed in voice-path-cleanbreak
S2 (commit 249bc06). Voice and video paths share the L3 ledger as
the single source of truth; there is no Director-shape projection
anywhere in active code.

LLM runs via transformers (local GPU). Content safety filter
catches profanity/NSFW that slips past the prompt policy.

v1.0  2026-04-04  Jeffrey Brick
v2.0  2026-05-13  voice-path-cleanbreak S23.2 (docstring scrub)
"""

import json
import logging
import os
import random
from random import SystemRandom

# OS-backed RNG for the Lemmy easter-egg coin flip.
# We can't use the seeded module-level `random` because it's seeded per-episode
# from the fingerprint (for reproducible Gemma behavior), which would freeze the
# 11% roll into "always on" or "always off" for any given widget config.
# SystemRandom is unaffected by random.seed() and gives a true ~11% per run.
_LEMMY_RNG = SystemRandom()
_LEMMY_HISTORY = []  # Rolling window of recent Lemmy coin flips (True/False)
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

# Project State (v1.4 Theme C) - series bible for cross-episode consistency.
# Read-only during generation. See nodes/project_state.py for the write path.
from .project_state import ProjectState
# Per-phase VRAM telemetry (v1.4 Theme C). CUDA-absent safe.
from ._vram_log import vram_snapshot, vram_reset_peak, force_vram_offload

# Canonical OTR paths -- single source of truth for output locations.
# (director_raw_dump_dir was deleted in voice-path-cleanbreak S23.1
# with no live consumer remaining; no replacement import needed.)


def _flush_vram_keep_llm():
    """Lightweight VRAM flush: clears KV cache fragments and fragmentation
    but keeps the LLM model weights on GPU.

    Use between LLM phases within a single write_script() run where the same
    model will be called again immediately. Avoids the ~13s-per-reload penalty
    caused by force_vram_offload() evicting the model from VRAM.

    force_vram_offload() is still used at node BOUNDARIES where we need to
    hand off GPU to a different model (e.g., LLM - Bark TTS).
    """
    import gc
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        gc.collect()


# Lazy heavy imports (Section 8) - torch, numpy, transformers inside methods/classes only

log = logging.getLogger("OTR")

# BaseStreamer for custom heartbeat logic.
# Graceful stub allows importing this module in test environments without
# a GPU or transformers installed - ScriptParser and pure-logic tests work fine;
# actual LLM generation will raise ImportError at call time as expected.
try:
    from transformers.generation.streamers import BaseStreamer, TextStreamer
except ImportError:
    class BaseStreamer:  # type: ignore[no-redef]
        """Stub - transformers not installed in this environment."""
        def put(self, value): pass
        def end(self): pass
    class TextStreamer(BaseStreamer):  # type: ignore[no-redef]
        pass

def _runtime_log(msg):
    """Write a persistent heartbeat to otr_runtime.log for monitoring."""
    try:
        ts = datetime.now().strftime("%H:%M:%S")
        log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "otr_runtime.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except: pass

def _truncate_at_sentence_boundary(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, trying to back up to the nearest sentence boundary."""
    if len(text) <= max_chars:
        return text
    
    truncated = text[:max_chars]
    # Look for the last terminal punctuation
    match = re.search(r'([.!?])(?=\s|$)[^.!?]*$', truncated)
    if match:
        return truncated[:match.end()]
    
    # If no punctuation found, just back up to last space
    last_space = truncated.rfind(' ')
    if last_space > 0:
        return truncated[:last_space] + '...'
    return truncated + '...'

def _tail_at_sentence_boundary(text: str, target_chars: int) -> str:
    """Take the LAST target_chars of text, walking forward to the next sentence start."""
    if len(text) <= target_chars:
        return text
        
    tail = text[-target_chars:]
    # Find the FIRST terminal punctuation in the tail
    match = re.search(r'[.!?]\s+', tail)
    if match:
        return tail[match.end():]
        
    # If no punctuation, walk forward to first space
    first_space = tail.find(' ')
    if first_space > 0:
        return tail[first_space+1:]
    return tail

def _inject_scene_transitions(script_text: str) -> tuple:
    """Detect '=== SCENE ===' boundaries and inject transition SFX where lacking.
    Returns (modified_text, transition_count).
    """
    lines = script_text.split('\n')
    out_lines = []
    transition_count = 0
    idx = 0
    
    while idx < len(lines):
        line = lines[idx]
        out_lines.append(line)
        
        # Check if line is a scene boundary (but don't inject after Scene 1 which has opening music)
        if re.match(r'^===\s*SCENE\s+(?![1]\b)(.+?)\s*(?:===|\*\*\*)', line.strip(), re.IGNORECASE):
            # Look ahead at the next non-empty line
            lookahead = idx + 1
            while lookahead < len(lines):
                next_line = lines[lookahead].strip()
                if not next_line:
                    lookahead += 1
                    continue
                
                # If the next thing is just dialogue, inject a transition
                if next_line.startswith('[VOICE:'):
                    out_lines.append("")
                    out_lines.append("[SFX: Scene transition - low bass sweep or static crossfade]")
                    out_lines.append("(beat)")
                    transition_count += 1
                break
                
        idx += 1
        
    return "\n".join(out_lines), transition_count



# -----------------------------------------------------------------------------
# Phase 3c: WALL-CLOCK TIMEOUT WRAPPER
# Heavy LLM phases (Open-Close outlines, Critique, Revision) can hang if
# LLM stalls on a malformed prompt or GPU goes sideways. We run the
# call in a worker thread and bound it with a wall-clock budget. On timeout
# the thread is left to drain in the background (Gemma generation is not
# cancellable mid-token) but the caller gets control back via TimeoutError
# and the pipeline can fall back to its last known-good artifact.
# -----------------------------------------------------------------------------
class _LLMTimeout(Exception):
    """Raised when an LLM phase exceeds its wall-clock budget."""
    pass


class _LLMTimeoutWorkflowPause(_LLMTimeout):
    """S22.1 (IMP-26): raised when an LLM timeout occurs AND the next
    workflow stage cannot safely run with an orphan worker still on
    GPU. ComfyUI catches this at the node boundary and halts the
    queue cleanly.

    Subclasses ``_LLMTimeout`` so existing handlers that catch the
    base class still match -- they just see a more specific subtype
    now. New downstream consumers can branch on this class for
    graceful UI handling (e.g., a "rerun" prompt).

    The cache invalidation in ``_run_with_timeout`` (via
    `_otr_model_loader.invalidate_cache_no_gpu_teardown`) handles
    the LLM -> LLM case (next LLM phase forces a fresh load from
    disk). Raising this class handles the LLM -> visual case where the
    next stage is FLUX/LTX/HuMo and would race the orphan's still-
    running CUDA kernels. The only safe move is to halt the workflow
    so the orphan finishes naturally without anything else trying
    to touch the GPU.

    Assumption: ComfyUI's node-execution layer surfaces uncaught
    exceptions as queue halts. Stable since the 2025 unified-
    execution refactor; if a future ComfyUI version swallows the
    exception, this assumption needs revisiting.
    """
    pass


import threading
_TIMEOUT_CTX = threading.local()

def _run_with_timeout(fn, timeout_sec, phase_label="LLM"):
    """Run fn() in a worker thread with a wall-clock timeout.

    Returns fn's return value on success.
    Raises _LLMTimeout if the budget is exceeded.
    Re-raises any exception fn raised.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    vram_reset_peak(phase_label)

    def _worker():
        _TIMEOUT_CTX.deadline = time.time() + timeout_sec
        try:
            return fn()
        finally:
            if hasattr(_TIMEOUT_CTX, "deadline"):
                del _TIMEOUT_CTX.deadline

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"otr-{phase_label}")

    try:
        future = executor.submit(_worker)
        try:
            res = future.result(timeout=timeout_sec)
            vram_snapshot(phase_label)
            return res
        except FuturesTimeout:
            _runtime_log(f"TIMEOUT: {phase_label} exceeded {timeout_sec}s wall-clock budget")
            log.warning("[Timeout] %s phase exceeded %ds - abandoning and falling back",
                        phase_label, timeout_sec)
            vram_snapshot(f"{phase_label}_timeout")

            # BUG-LOCAL-111 + BUG-LOCAL-228 fix: timeout-recovery cache
            # invalidation.
            #
            # When FuturesTimeout fires, the worker thread is still running
            # an LLM forward pass on GPU. Python cannot safely terminate
            # threads, and `executor.shutdown(wait=False)` below does NOT
            # kill the worker -- it keeps churning until its forward pass
            # completes naturally (could be 30-60+ more seconds for a 16K
            # prompt). Result: the GPU has in-flight kernels the main
            # thread doesn't control. The cached model instance thinks
            # it's idle but the orphan is still mutating its tensors.
            # The NEXT phase that calls model.cpu() / any CUDA op
            # collides with the orphan's stale ops and Python aborts
            # with `cudaErrorIllegalAddress`.
            #
            # The recovery path invalidates the canonical
            # `_otr_model_loader.LLM_CACHE` dict references in-place
            # via `invalidate_cache_no_gpu_teardown` -- dict-only
            # invalidator, NO GPU calls. Orphan thread keeps its
            # model reference in the worker's stack frame and exits
            # naturally; the next `request_slot` forces a fresh load.
            try:
                # `invalidate_cache_no_gpu_teardown` (NOT `unload_llm`)
                # is the GPU-safe path here. `unload_llm` would call
                # `model.to("cpu")` + `torch.cuda.empty_cache()` while
                # the orphan worker thread is still executing CUDA
                # kernels on the cached model -- which is exactly the
                # `cudaErrorIllegalAddress` failure mode. The helper
                # invalidates the cache dict references WITHOUT
                # touching the GPU; the orphan thread holds the model
                # in its stack frame and exits naturally.
                # See BUG-LOCAL-228.
                from . import _otr_model_loader as _otr_loader_mod
                _otr_loader_mod.invalidate_cache_no_gpu_teardown()
                _runtime_log(
                    f"TIMEOUT_RECOVERY: LLM_CACHE invalidated (GPU "
                    f"untouched; orphan {phase_label} worker keeps "
                    f"its model reference until natural completion). "
                    f"Next request_slot forces a fresh load."
                )
            except Exception as _recovery_exc:  # noqa: BLE001
                log.warning(
                    "[Timeout] cache invalidation failed (next phase may "
                    "still crash on stale CUDA state): %s",
                    _recovery_exc,
                )

            # S22.1 (IMP-26): raise the workflow-pause subclass so
            # ComfyUI halts the queue before the next stage races
            # the orphan worker's still-running CUDA kernels. The
            # cache invalidation above handles LLM -> LLM; this
            # raise handles LLM -> visual (FLUX / LTX / HuMo).
            raise _LLMTimeoutWorkflowPause(
                f"{phase_label} exceeded {timeout_sec}s; orphan "
                f"worker still on GPU. Halting workflow to prevent "
                f"the next visual stage from racing the orphan's "
                f"CUDA kernels. Re-run the workflow; the cache "
                f"invalidation above guarantees the next attempt "
                f"loads fresh."
            )
    finally:
        # Don't wait for the orphaned worker - let it drain in the background.
        # Combined with the cache invalidation above, the orphan completes
        # in its own time WITHOUT the next phase trying to reuse its tensors.
        executor.shutdown(wait=False)


# -----------------------------------------------------------------------------
# CAST CONSOLIDATION HELPERS
# Used by the LPL writer's cast-lock path to collapse near-duplicate cast
# rows that arise from prefix-overlap (LLOYD vs LLOYD KAPOOR) or LLM typo
# divergence (STANLEY vs STANLEARY). BUG-LOCAL-068 anchor preserved.
# -----------------------------------------------------------------------------


def _norm_cast_key(s):
    """Normalise a character name for lookup: strip, upper, _ -> space."""
    return (s or "").strip().upper().replace("_", " ")


def _cast_names_should_merge(name_a, name_b, fuzzy_ratio=0.85):
    """Decide whether two normalised cast names refer to the same character.

    Returns (should_merge: bool, winner_name: str | None) where winner_name
    is the name that should survive the merge.

    Rules (in order of precedence):
      1. Exact equality after strip/upper/underscore-normalisation -> merge
      2. Token prefix overlap: shorter is a strict whitespace-aligned prefix
         of longer (e.g. "LLOYD" prefix of "LLOYD KAPOOR") -> merge,
         winner = longer (more information).
      3. SequenceMatcher.ratio >= fuzzy_ratio AND both have the same
         single-token shape (no spaces) -> merge, winner = longer
         (typo divergence: "STANLEY" + "STANLEARY" -> keep "STANLEY"
         only when a clear winner emerges; ties favour the longer string).

    Returns (False, None) when the names are distinct characters.
    Empty / one-char names are never merged.
    """
    if not name_a or not name_b:
        return (False, None)
    a = name_a.strip()
    b = name_b.strip()
    if len(a) <= 1 or len(b) <= 1:
        return (False, None)
    if a == b:
        return (True, a)
    # Token prefix overlap: shorter must end on a word boundary inside longer
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    if long_.startswith(short + " "):
        return (True, long_)
    # Pure typo divergence: only collapse single-token names. We do NOT
    # apply fuzzy ratio across multi-token names because "ROBERT FROST"
    # and "ROBERT FORD" would otherwise merge incorrectly.
    if " " in a or " " in b:
        return (False, None)
    import difflib
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    if ratio >= fuzzy_ratio:
        # Prefer the longer string. If tied, prefer the alphabetically
        # earlier so the choice is deterministic.
        if len(a) > len(b):
            return (True, a)
        if len(b) > len(a):
            return (True, b)
        return (True, min(a, b))
    return (False, None)


def _consolidate_similar_cast_rows(cast_rows):
    """Collapse near-duplicate cast rows in place.

    For each pair of rows whose names should merge per
    `_cast_names_should_merge`, fold the loser's data into the winner:
      - voice_preset: any non-None value wins (loser's wins if winner has
        None — common pattern: speaker row has dialogue but no voice;
        description row has voice but no dialogue).
      - description / gender: any non-empty value wins.
      - line_count, word_count: summed (defensive — usually only one row
        carries them, but if the parser tagged both we don't lose dialogue).
      - char_id, name: from the winner.

    Loser rows are removed. Order of survivors preserves the order of
    their first appearance. Returns a NEW list; does not mutate the
    input list, but row dicts inside the result are the original objects
    (mutated in place where merging happened).

    Idempotent: running twice yields the same result as running once.

    BUG-LOCAL-098: callers that need to rewrite downstream `char_id`
    references (e.g. `lines[i].char_id`) should use
    ``_consolidate_similar_cast_rows_with_aliases`` which returns
    ``(consolidated_cast, alias_map)`` where alias_map is
    ``{loser_char_id: winner_char_id}``. The plain function here is
    kept for back-compat with callers that don't track aliases.
    """
    consolidated, _ = _consolidate_similar_cast_rows_with_aliases(cast_rows)
    return consolidated


def _consolidate_similar_cast_rows_with_aliases(cast_rows):
    """BUG-LOCAL-098: like ``_consolidate_similar_cast_rows`` but also
    returns the ``{loser_char_id: winner_char_id}`` alias map so
    callers can rewrite line/sfx/music ``char_id`` references to the
    surviving cast entry. Without this, dedup leaves the cast clean
    but ``lines[i].char_id`` still points at the dropped char_id,
    causing Bark to fail voice resolution for those lines (observed
    on the Arcadia run: l001 + l033 referenced ``c02`` after the
    ANNOUCNER/ANNOUNCER typo merge dropped c02 from cast).
    """
    if not cast_rows or len(cast_rows) < 2:
        return list(cast_rows or []), {}

    keep = list(cast_rows)
    aliases: dict[str, str] = {}  # loser_char_id -> winner_char_id
    i = 0
    while i < len(keep):
        row_i = keep[i]
        name_i = _norm_cast_key(row_i.get("name"))
        j = i + 1
        while j < len(keep):
            row_j = keep[j]
            name_j = _norm_cast_key(row_j.get("name"))
            should, winner = _cast_names_should_merge(name_i, name_j)
            if should:
                # Identify winner row vs loser row
                if winner == name_i:
                    win_row, lose_row = row_i, row_j
                else:
                    win_row, lose_row = row_j, row_i
                # Fold loser into winner
                if not win_row.get("voice_preset") and lose_row.get("voice_preset"):
                    win_row["voice_preset"] = lose_row["voice_preset"]
                if not win_row.get("description") and lose_row.get("description"):
                    win_row["description"] = lose_row["description"]
                if not win_row.get("gender") and lose_row.get("gender"):
                    win_row["gender"] = lose_row["gender"]
                # Numeric fields summed defensively
                for fld in ("line_count", "word_count"):
                    a = win_row.get(fld) or 0
                    b = lose_row.get(fld) or 0
                    if a or b:
                        win_row[fld] = a + b
                # Make sure winner uses the canonical winner name
                win_row["name"] = winner
                # BUG-LOCAL-098: record loser->winner char_id alias so
                # the caller can rewrite lines[i].char_id and other
                # references that pointed at the loser. Chains of
                # aliases (e.g. STANLEY -> STANLEARY -> STANLERY are
                # all aliases of the final survivor) get flattened by
                # walking the alias map transitively below.
                lose_cid = (lose_row.get("char_id") or "").strip()
                win_cid = (win_row.get("char_id") or "").strip()
                if lose_cid and win_cid and lose_cid != win_cid:
                    aliases[lose_cid] = win_cid
                # Remove loser, restart inner walk so we re-check the
                # winner row against everyone else (chained typos like
                # STANLEY / STANLEARY / STANLERY all collapse safely).
                if lose_row is row_i:
                    keep.pop(i)
                    name_i = _norm_cast_key(row_i.get("name") if i < len(keep) else None)
                    # i now points at what was row_j; recheck from j=i+1
                    j = i + 1
                    if i < len(keep):
                        row_i = keep[i]
                        name_i = _norm_cast_key(row_i.get("name"))
                    else:
                        break
                else:
                    keep.pop(j)
                    # j stays the same -- now points at the next row
                continue
            j += 1
        i += 1

    # Flatten chained aliases: if A -> B and B -> C, rewrite A -> C.
    # Bounded loop in case of cycles (shouldn't happen but defensive).
    for _ in range(len(aliases) + 1):
        changed = False
        for k, v in list(aliases.items()):
            if v in aliases and aliases[v] != v:
                aliases[k] = aliases[v]
                changed = True
        if not changed:
            break

    return keep, aliases


# -----------------------------------------------------------------------------
# Phase 3d: BARK VOICE HEALTH CHECK
# Synthesize a 1-second test clip for each active English preset at startup.
# Any preset that returns silence or NaN gets removed from _VOICE_PROFILES
# for the rest of the session, so the writer can never re-assign a broken
# voice. Runs once per process, lazily on first ScriptWriter init so we
# don't pay the Bark load cost in environments that only import the module.
# -----------------------------------------------------------------------------
_BARK_HEALTH_CHECKED = False
_BARK_HEALTH_DISABLED = set()


def _bark_test_presets(presets_to_test):
    """Run a 1-second synthesis test on each given Bark preset.

    Pure helper: no global state mutation, no idempotency flag. Returns
    (passed: set, disabled: set, reason: str | None). Reason is non-None
    only on a "skip everything" outcome (Bark unimportable / probe failed).

    Used by both the legacy full-catalog `_bark_health_check()` and the
    new lazy cast-only `_bark_health_check_for_cast()`.
    """
    if not presets_to_test:
        return set(), set(), None
    presets_to_test = sorted(set(presets_to_test))

    try:
        import numpy as np
        from ._otr_bark_lib import _load_bark
        from ._otr_bark_lib import _generate_single_line
    except ImportError as e:
        log.info("[VoiceHealth] Bark not importable (%s) - skipping health check", e)
        _runtime_log(f"VOICE_HEALTH_SKIPPED: bark unavailable ({e})")
        return set(), set(), f"bark unavailable ({e})"

    # Smoke test on a single preset before the full sweep -- catches
    # "Bark itself is broken" early so we don't mark every preset failed.
    try:
        model, processor = _load_bark(device="cuda")
        _probe, _ = _generate_single_line("Test.", presets_to_test[0], model, processor, temperature=0.6)
    except Exception as e:
        log.warning("[VoiceHealth] Bark probe failed (%s) - leaving presets untested", e)
        _runtime_log(f"VOICE_HEALTH_SKIPPED: bark probe failed ({e})")
        return set(presets_to_test), set(), f"probe failed ({e})"

    test_text = "Testing one two three."
    passed, disabled = set(), set()
    for preset in presets_to_test:
        t0 = time.time()
        try:
            arr, _ = _generate_single_line(test_text, preset, model, processor, temperature=0.6)
            if arr.size == 0:
                raise ValueError("empty audio")
            if not np.isfinite(arr).all():
                raise ValueError("NaN/Inf in output")
            if float(np.max(np.abs(arr))) < 1e-4:
                raise ValueError("silent output")
            passed.add(preset)
            log.info("[VoiceHealth] %s OK (%.1fs)", preset, time.time() - t0)
            _runtime_log(f"VOICE_HEALTH_OK: {preset} ({time.time()-t0:.1f}s)")
        except Exception as e:
            disabled.add(preset)
            log.warning("[VoiceHealth] %s FAILED: %s", preset, e)
            _runtime_log(f"VOICE_HEALTH_DISABLED: {preset} - {e}")
    return passed, disabled, None


def _bark_health_check():
    """LEGACY: full-catalog Bark warmup + global pool mutation.

    Tests every active en_speaker_* preset and removes failed presets
    from `_VOICE_PROFILES`, `_ANNOUNCER_PRESETS`, and `_LEMMY_PROFILE`.
    Idempotent: runs only on the first call per process.

    NOTE: This function is kept exported as a manual catalog-validation
    tool and for any external caller. It is no longer invoked by the
    default orchestration path -- the lazy `_bark_health_check_for_cast()`
    runs after cast assignment instead. Historical context: see commit
    249bc06 (voice-path-cleanbreak S2).
    """
    global _BARK_HEALTH_CHECKED, _VOICE_PROFILES, _ANNOUNCER_PRESETS, _LEMMY_PROFILE
    if _BARK_HEALTH_CHECKED:
        return
    _BARK_HEALTH_CHECKED = True

    log.info("[VoiceHealth] Running 1-second Bark health check on full catalog...")
    _runtime_log("VOICE_HEALTH: Starting full-catalog Bark preset health check")
    presets_to_test = sorted({vp[0] for vp in _VOICE_PROFILES} |
                              {p for p, _ in _ANNOUNCER_PRESETS} |
                              {_LEMMY_PROFILE["voice_preset"]})
    _, disabled, reason = _bark_test_presets(presets_to_test)
    if reason:
        return
    if disabled:
        _BARK_HEALTH_DISABLED.update(disabled)
        _VOICE_PROFILES[:] = [vp for vp in _VOICE_PROFILES if vp[0] not in disabled]
        _ANNOUNCER_PRESETS[:] = [(p, n) for p, n in _ANNOUNCER_PRESETS if p not in disabled]
        if _LEMMY_PROFILE["voice_preset"] in disabled:
            survivors = [vp[0] for vp in _VOICE_PROFILES if vp[1] == "male"]
            if survivors:
                fallback = survivors[0]
                log.warning("[VoiceHealth] LEMMY preset disabled - falling back to %s", fallback)
                _runtime_log(f"VOICE_HEALTH_DISABLED: LEMMY preset replaced with {fallback}")
                _LEMMY_PROFILE["voice_preset"] = fallback
        _runtime_log(f"VOICE_HEALTH: {len(disabled)} preset(s) disabled, {len(_VOICE_PROFILES)} remain")
    else:
        _runtime_log(f"VOICE_HEALTH: All {len(presets_to_test)} presets passed")


def _bark_health_check_for_cast(cast_rows):
    """LAZY: validate only the Bark presets the cast actually uses.

    Trades a one-time ~120s full-catalog warmup at writer start for a
    ~5-25s targeted check after the writer locks the cast. On any
    preset failure, swaps the cast row's voice_preset for a known-good fallback
    of the same gender from `_VOICE_PROFILES` and records the swap.

    Mutates cast_rows IN PLACE; returns the list (same object) for
    chainability. Kokoro voices (`bm_*` / `am_*` / `bf_*` / `af_*`) are
    skipped because Kokoro has its own loader path.
    """
    if not cast_rows:
        return cast_rows or []

    # Extract distinct Bark presets the cast actually uses
    bark_voices_in_cast = set()
    for r in cast_rows:
        vp = r.get("voice_preset") or ""
        if isinstance(vp, str) and vp.startswith("v2/"):
            bark_voices_in_cast.add(vp)

    if not bark_voices_in_cast:
        _runtime_log("VOICE_HEALTH_LAZY: cast has no Bark presets (Kokoro-only?); skipped")
        return cast_rows

    log.info("[VoiceHealth] Lazy cast check on %d preset(s): %s",
             len(bark_voices_in_cast), sorted(bark_voices_in_cast))
    _runtime_log(f"VOICE_HEALTH_LAZY: testing {len(bark_voices_in_cast)} cast preset(s)")
    _, disabled, reason = _bark_test_presets(sorted(bark_voices_in_cast))
    if reason:
        # Bark not importable or probe failed -- leave cast alone, BatchBark
        # will surface the issue at generation time as it always has.
        return cast_rows
    if not disabled:
        _runtime_log(f"VOICE_HEALTH_LAZY: all {len(bark_voices_in_cast)} cast preset(s) passed")
        return cast_rows

    # Re-assign each disabled preset to a fallback voice from the same
    # gender pool, avoiding duplicates within the cast where possible.
    in_use = {r.get("voice_preset") for r in cast_rows if r.get("voice_preset")}
    log.warning("[VoiceHealth] %d cast preset(s) failed; remapping",
                len(disabled))
    for row in cast_rows:
        vp = row.get("voice_preset") or ""
        if vp not in disabled:
            continue
        gender = (row.get("gender") or "").lower()
        # Prefer same-gender survivors not already used by another cast row
        candidates = [pp for pp, gg in _VOICE_PROFILES
                      if pp not in disabled
                      and pp not in _BARK_HEALTH_DISABLED
                      and (not gender or gg == gender)
                      and pp not in in_use]
        # Fallback 1: same gender, even if already in use
        if not candidates and gender:
            candidates = [pp for pp, gg in _VOICE_PROFILES
                          if pp not in disabled
                          and pp not in _BARK_HEALTH_DISABLED
                          and gg == gender]
        # Fallback 2: any surviving voice
        if not candidates:
            candidates = [pp for pp, gg in _VOICE_PROFILES
                          if pp not in disabled and pp not in _BARK_HEALTH_DISABLED]
        if not candidates:
            log.error("[VoiceHealth] no fallback available for cast row %s; voice_preset stays %s",
                      row.get("name"), vp)
            _runtime_log(f"VOICE_HEALTH_LAZY: no fallback available for {row.get('name')}; preset {vp} retained (BatchBark will fail)")
            continue
        new_vp = candidates[0]
        log.warning("[VoiceHealth] %s remapped: %s -> %s (gender=%s)",
                    row.get("name") or "(unnamed)", vp, new_vp, gender or "any")
        _runtime_log(f"VOICE_HEALTH_LAZY: remap {row.get('name')} {vp} -> {new_vp}")
        row["voice_preset"] = new_vp
        in_use.discard(vp)
        in_use.add(new_vp)
    _BARK_HEALTH_DISABLED.update(disabled)
    return cast_rows

# -----------------------------------------------------------------------------
# LOG CLEANUP - compliant fixes handle most warnings at the source.
# These catch residual library noise from urllib3/httpx cache checks.
# -----------------------------------------------------------------------------
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# PROCEDURAL CHARACTER GENERATOR - name, age, gender, demeanor, accent, voice
# All traits derived deterministically from episode seed + character index.
# LEMMY stays LEMMY with fixed traits. ANNOUNCER stays ANNOUNCER.
#
# BARK TTS ACCENT RULES (per Suno documentation):
#   - Foreign preset + pure English text = English spoken with that accent
#   - en_speaker_* = neutral American/British English
#   - de_speaker_* = English with German accent
#   - fr_speaker_* = English with French accent
#   - es_speaker_* = English with Spanish accent  ... etc.
#   - ALL text is ALWAYS pure ASCII English (enforced by ASCII sanitizer
#     in batch_bark_generator.py) - this prevents language drift
#   - Temperature capped at 0.55 for international presets (0.5 first lines)
# -----------------------------------------------------------------------------

# Sci-fi character name pools - contemporary, neutral, tech-aligned
# Omni-Retro 5-Pillar Naming Pool - short, punchy, Bark-optimized (1-2 syllables, hard consonants)
# Pillars: 1950s Americana Noir, Afrofuturism, Neo-Tokyo Cyberpunk, Thai Density, Russian Dieselpunk
_FIRST_NAMES = [
    # 1950s Americana Noir
    "Vance", "Stone", "Margot", "Nora", "Sully", "Mac", "Hayes",
    "Cole", "Drake", "Quinn", "Reese", "Kane", "Carter", "Blake",
    # Afrofuturism
    "Malik", "Zuri", "Chidi", "Ayo", "Oya", "Kael", "Tariq", "Nia",
    # Neo-Tokyo Cyberpunk
    "Ren", "Akira", "Kenji", "Yuki", "Sora", "Jiro", "Rei", "Hiro",
    # Thai Density
    "Krit", "Mali", "Niran", "Sunan", "Dao", "Pim", "Som",
    # Russian Dieselpunk
    "Lev", "Anya", "Dmitri", "Sergei", "Volkov", "Mira", "Yuri",
    # Simpsons (sci-fi viable)
    "Nelson", "Martin", "Carl", "Lenny", "Montgomery", "Seymour", "Edna",
    "Ned", "Barney", "Moe", "Kent", "Rod", "Todd", "Jimbo", "Dolph", "Kearney",
    # Pulp adventure (generic first names)
    "Dale", "Tommy", "Pinky",
    # Public domain classics (published before 1931)
    "Alice", "Allan", "Ayesha", "Cavor", "Dracula", "Edward", "Griffin", "Gulliver",
    "Henry", "James", "John", "Karnacki", "Leviathan", "Mina", "Nemo", "Phileas",
    "Quasimodo", "Robinson", "Sherlock", "Smee", "Tarkon", "Victor", "Watson", "Wendy",
    # Peter O'Toole characters
    "Lawrence", "Reginald", "Anton", "Priam", "Maurice", "Alan",
    # Jim Carrey characters
    "Truman", "Fletcher", "Joel", "Stanley", "Walter", "Ace", "Lloyd", "Bruce",
    # Robin Williams characters
    "Mork", "Adrian", "Sean", "Andrew", "Parry", "Malcolm", "Daniel", "Chris",
    # The Office - generic character first names
    "Michael", "Pam", "Ryan", "Kevin", "Kelly", "Meredith",
    "Stanley", "Toby", "Darryl", "Erin", "Creed", "Oscar", "Phyllis",
    # Real actor first names
    "Steve", "Rainn", "Jenna", "Mindy", "Ellie", "Rashida", "Ed",
    # Classic fiction characters (generic)
    "Clarisse", "Doug", "Travis", "Charlie", "Will", "Faber",
    "Rick", "Palmer", "Glen", "Isidore", "Bob", "Donna", "Juliana",
    "Manfred", "Leo",
    # Richard Pryor characters
    "Gus", "Monty", "Duane", "Rufus", "Leroy", "Skip", "Grover",
    # Robin Williams (additional)
    "Peter", "Sailor", "Djinn",
]

_LAST_NAMES = [
    "Stone", "Shaw", "Cross", "Wells", "Steele", "Frost", "Pierce", "Vaughn",
    "Black", "Drake", "Hayes", "Kane", "Voss", "Cranston", "Kendall", "Reeves",
    "Volkov", "Sato", "Tanaka", "Okafor", "Diallo", "Sirikit", "Petrov",
    # Generic last names (scrubbed franchise-specific)
    "Burns", "Hibbert", "Flanders", "Houten", "Smithers",
    "Terwilliger", "Bouvier", "Simpson", "Gordon", "Ming",
    "Carruthers", "Corben",
    # The Office - character last names (generic ones only)
    "Scott", "Halpert", "Beesly", "Howard", "Bernard", "Malone",
    "Kapoor", "Palmer", "Hudson", "Martin", "Flenderson", "Philbin", "Vance",
    # Ray Bradbury (generic)
    "Beatty", "Spender", "Stendahl", "Eckels", "Halloway",
    # Misc classic (generic)
    "Steiner",
]

# Trait pools for procedural character profiles
_GENDERS = ["male", "female"]
_AGE_BRACKETS = ["20s", "30s", "40s", "50s", "60s"]
_DEMEANORS = [
    "calm", "intense", "warm", "sharp", "dry", "energetic",
    "measured", "wry", "stoic", "anxious", "confident", "weary",
]

# Accent pool - 100% English-native presets only.
# Foreign presets (de_speaker, fr_speaker, etc.) caused Bark hallucinations:
# the model generates foreign-language phonemes when given English text,
# producing gibberish instead of accented English. Until Bark's multilingual
# stability improves, all characters use en_speaker_* presets.
# See: v1.1 "Test Signal" critique - Lemmy (de_speaker_0) was unintelligible.
_ACCENTS = [
    ("neutral",  "en", 1.00),   # English-only - no foreign presets
]

# Per-character voice traits pool. Drawn at cast-roll time so each character
# carries a fixed (gender, age_band, tone, energy, vocab_register, signature)
# tuple across every scene. Without this, the LLM improvises traits inside
# each [VOICE:] tag and the same character drifts in voice/age/gender between
# scenes. Signature is a one-line speech tic the LLM keeps consistent.
# Keep this list balanced: ~50/50 gender split, varied age bands, varied
# energy levels, no two entries with identical (gender, age, tone) triples.
_VOICE_TRAITS = [
    ("male",   "30s", "wry",          "low",    "blue-collar",  "trims sentences mid-thought"),
    ("female", "40s", "clipped",      "high",   "technical",    "answers questions with questions"),
    ("male",   "50s", "weathered",    "low",    "plainspoken",  "drops articles when stressed"),
    ("female", "20s", "curious",      "medium", "academic",     "names tools by their model number"),
    ("male",   "60s", "gravelly",     "low",    "rural",        "ends sentences with 'son' or 'kid'"),
    ("female", "30s", "sardonic",     "medium", "urban",        "speaks in fragments under stress"),
    ("male",   "40s", "anxious",      "high",   "bureaucratic", "qualifies every statement"),
    ("female", "50s", "authoritative","medium", "command",      "issues orders, never asks"),
    ("male",   "20s", "earnest",      "high",   "vernacular",   "uses contractions even when formal"),
    ("female", "60s", "quiet",        "low",    "lyrical",      "speaks in metaphor when scared"),
]


# Voice presets mapped by gender + vocal quality + language code.
# English-native presets (en_speaker_*) have known vocal qualities.
# International presets (xx_speaker_*) are grouped by speaker index tendencies.
# Each entry: (preset, gender, quality_tags)
_VOICE_PROFILES = [
    # -- English native (neutral accent) --
    ("v2/en_speaker_0", "male",   "en", {"authoritative", "deep", "50s", "60s", "announcer", "commander"}),
    ("v2/en_speaker_1", "male",   "en", {"calm", "measured", "30s", "40s", "technical", "pilot"}),
    ("v2/en_speaker_3", "male",   "en", {"energetic", "sharp", "20s", "30s", "rebel", "technician"}),
    ("v2/en_speaker_5", "male",   "en", {"warm", "weary", "wry", "50s", "60s", "doctor", "scientist"}),
    ("v2/en_speaker_6", "male",   "en", {"intense", "dry", "stoic", "40s", "officer", "android"}),
    ("v2/en_speaker_8", "male",   "en", {"gravelly", "anxious", "confident", "40s", "50s", "engineer", "mechanic"}),
    # English native (female)
    ("v2/en_speaker_2", "female", "en", {"clipped", "precise", "30s", "40s", "officer", "neutral-british"}), # Sounds precise/British-adjacent
    ("v2/en_speaker_4", "female", "en", {"warm", "energetic", "wry", "30s", "40s", "pilot", "explorer"}),
    ("v2/en_speaker_9", "female", "en", {"authoritative", "confident", "intense", "50s", "60s", "commander", "senator"}),
    # FIX-3 (v1.2): en_speaker_7 reclassified to female to prevent CAST_GENDER_POOL_EXHAUSTED
    # on 3-female episodes (was causing VEX/ZARA to share en_speaker_9 and sound identical).
    # Bark labels en_speaker_7 as androgynous - in English it reads soft/lighter so we
    # use it as the "younger" female slot (20s, anxious/sharp/technician).
    ("v2/en_speaker_7", "female", "en", {"sharp", "anxious", "nervous", "20s", "30s", "technician", "hacker"}),
    # -- DISABLED: Foreign accent presets ------------------------------
    # These caused Bark hallucinations - the model generates foreign-language
    # phonemes when fed English text, producing gibberish. Kept as comments
    # for future reference if Bark's multilingual stability improves.
    # See v1.1 "Test Signal" critique: de_speaker_0 (Lemmy) was unintelligible,
    # fr_speaker lines also showed artifacts.
    #
    # German:  de_speaker_0/3/5 (male), de_speaker_2/7 (female)
    # Spanish: es_speaker_0/6/8 (male), es_speaker_4/9 (female)
    # French:  fr_speaker_1/5 (male), fr_speaker_2/4 (female)
    # Indian:  hi_speaker_0/5 (male), hi_speaker_4/9 (female)
    # Italian: it_speaker_0/6 (male), it_speaker_4/9 (female)
    # Japanese: ja_speaker_1/6 (male), ja_speaker_4 (female)
    # Korean:  ko_speaker_0 (male), ko_speaker_4 (female)
    # Russian: ru_speaker_0/3 (male), ru_speaker_4/9 (female)
    # Brazilian: pt_speaker_0 (male), pt_speaker_4 (female)
    # Polish:  pl_speaker_0 (male), pl_speaker_4 (female)
]

# ANNOUNCER voice pool - randomized per episode for gender balance (50/50 male/female)
# ANNOUNCER always uses neutral English (en_speaker_*) - no accent
_ANNOUNCER_PRESETS = [
    ("v2/en_speaker_0", "Male, authoritative, deep"),
    ("v2/en_speaker_1", "Male, measured, calm"),
    ("v2/en_speaker_4", "Female, warm, energetic"),
    ("v2/en_speaker_9", "Female, mature, authoritative"),
]

# LEMMY fixed profile - always gravelly/raspy male, English-native preset
_LEMMY_PROFILE = {
    "name": "LEMMY",
    "gender": "male",
    "age": "50s",
    "demeanor": "gravelly",
    "accent": "neutral",  # English-native preset; gravelly tone comes from en_speaker_8 vocal quality
    "voice_preset": "v2/en_speaker_8",  # English native - gravelly, confident, 40s-50s. Avoids Bark hallucination from de_speaker
    "notes": "Male, gravelly/raspy, 50s, gruff mechanic voice, iconic",
}


def _pick_accent(rng) -> tuple:
    """Weighted random accent selection. Returns (accent_label, lang_code).

    ~60% neutral English, ~40% spread across international accents.
    Uses cumulative distribution for deterministic weighted selection.
    """
    roll = rng.random()
    cumulative = 0.0
    for label, code, weight in _ACCENTS:
        cumulative += weight
        if roll < cumulative:
            return label, code
    # Fallback (rounding errors)
    return "neutral", "en"


def _generate_character_profile(character_idx: int, episode_seed: str = "",
                                gender_hint: str = None) -> dict:
    """Generate a full procedural character profile - deterministic per episode.

    Returns a dict with: name, gender, age, demeanor, accent, voice_preset, notes.
    All traits are seeded so reruns of the same episode produce identical casts.

    Voice preset selection:
      1. Pick gender, age, demeanor, accent procedurally
      2. Filter voice profiles by gender AND accent language code
      3. Score by trait overlap (age, demeanor)
      4. Best match wins (ties broken by RNG shuffle)

    Safety rails (downstream):
      - ASCII sanitizer strips non-ASCII from all text before Bark
      - Temperature capped at 0.55 for international presets
      - All dialogue is always written in pure English
    """
    rng = random.Random(f"{episode_seed}_char_{character_idx}")

    # Generate name
    first = rng.choice(_FIRST_NAMES)
    last = rng.choice(_LAST_NAMES)
    name = f"{first} {last}".upper()

    # Generate traits - honor gender_hint from script's [VOICE: NAME, gender, ...] tag
    # if provided. This is BUG-004 fix: previously the procedural cast picked random
    # genders, producing male voices on female characters and vice versa.
    if gender_hint and gender_hint.lower() in ("male", "female"):
        gender = gender_hint.lower()
    else:
        gender = rng.choice(_GENDERS)
    age = rng.choice(_AGE_BRACKETS)
    demeanor = rng.choice(_DEMEANORS)
    accent_label, lang_code = _pick_accent(rng)

    # Filter voice profiles by gender AND language code
    candidates = [vp for vp in _VOICE_PROFILES
                  if vp[1] == gender and vp[2] == lang_code]

    # If no match for this gender+accent combo, fall back to same-gender English
    if not candidates:
        candidates = [vp for vp in _VOICE_PROFILES
                      if vp[1] == gender and vp[2] == "en"]
        accent_label = "neutral"
        lang_code = "en"

    # Safety net - should never happen
    if not candidates:
        candidates = [vp for vp in _VOICE_PROFILES if vp[2] == "en"]

    # Score each candidate by how many tags overlap with character traits
    char_tags = {age, demeanor}
    scored = []
    for preset, _, _, tags in candidates:
        overlap = len(char_tags & tags)
        scored.append((overlap, preset, tags))

    # Sort by overlap (best match first), break ties with RNG shuffle
    rng.shuffle(scored)
    scored.sort(key=lambda x: x[0], reverse=True)
    best_preset = scored[0][1]

    # Build descriptive notes
    accent_note = f", {accent_label} accent" if accent_label != "neutral" else ""
    notes = f"{gender.capitalize()}, {demeanor}, {age}{accent_note}"

    return {
        "name": name,
        "gender": gender,
        "age": age,
        "demeanor": demeanor,
        "accent": accent_label,
        "voice_preset": best_preset,
        "notes": notes,
    }


def _generate_announcer_profile(episode_seed: str = "", gender_hint: str | None = None) -> dict:
    """Pick a Announcer voice from the balanced pool, seeded per episode.
    If gender_hint is provided (from script [VOICE: ANNOUNCER, gender, ...] tag),
    filter the pool to matching gender first; fall back to full pool if none match.
    ANNOUNCER always uses neutral English - no accent."""
    rng = random.Random(f"{episode_seed}_announcer")
    pool = _ANNOUNCER_PRESETS
    if gender_hint:
        gh = gender_hint.lower()
        # Use startswith to avoid "male" matching inside "female"
        filtered = [(p, n) for p, n in _ANNOUNCER_PRESETS
                    if n.lower().startswith(gh)]
        if filtered:
            pool = filtered
    preset, notes = rng.choice(pool)
    return {
        "name": "ANNOUNCER",
        "voice_preset": preset,
        "notes": notes,
    }


# -----------------------------------------------------------------------------
# NEWS FETCHER - pulls real science headlines to seed the story
# -----------------------------------------------------------------------------

SCIENCE_NEWS_FEEDS = [
    # -- Open-access: full article text fetchable, no paywall --
    "https://www.sciencedaily.com/rss/all.xml",           # Best: full articles, open
    "https://www.eurekalert.org/rss/technology_engineering.xml",  # Press releases, open
    "https://www.eurekalert.org/rss/space.xml",           # Press releases, open
    "https://www.eurekalert.org/rss/biology.xml",         # Press releases, open
    "https://www.eurekalert.org/rss/chemistry_physics.xml", # Press releases, open
    "https://www.eurekalert.org/rss/earth_environment.xml", # Press releases, open
    # -- Government / institutional (fully open) --
    "https://www.nasa.gov/rss/dyn/breaking_news.rss",     # NASA, open
    "https://www.nih.gov/news-events/news-releases.xml",  # NIH, open
    "https://www.nsf.gov/rss/rss_www_news.xml",           # NSF, open
    # -- UCLA Newsroom (open-access institutional research) --
    "https://newsroom.ucla.edu/cats/health_+_behavior.xml",      # Best: full content:encoded in RSS
    "https://newsroom.ucla.edu/cats/science_+_technology.xml",   # Open-access, URL scrape works
    "https://newsroom.ucla.edu/cats/environment_+_climate.xml",  # Open-access, URL scrape works
    # -- Open journalism (full text accessible) --
    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",  # BBC, open
    "https://feeds.arstechnica.com/arstechnica/science",  # Ars, open
    "https://theconversation.com/us/science/rss",         # The Conversation, open
    "https://cosmosmagazine.com/feed/",                   # Cosmos, open
    # -- MIT News (open-access university research feeds) --
    "https://news.mit.edu/topic/mitartificial-intelligence2-rss.xml",  # MIT AI, open
    "https://news.mit.edu/topic/mitmachine-learning-rss.xml",          # MIT machine learning, open
    "https://news.mit.edu/topic/mitcomputers-rss.xml",                 # MIT CSAIL / computer science, open
    "https://news.mit.edu/topic/mitrobotics-rss.xml",                  # MIT robotics, open
    "https://news.mit.edu/topic/mitquantum-computing-rss.xml",         # MIT quantum computing, open
    "https://news.mit.edu/rss/topic/neuroscience-neurology-and-cognitive-sciences",  # MIT brain + cognitive sci, open
    "https://news.mit.edu/topic/mitsynthetic-biology-rss.xml",         # MIT synthetic biology, open
    "https://news.mit.edu/topic/mitmedia-lab-0-rss.xml",               # MIT Media Lab, open
    # -- Carnegie Mellon (open-access university research feeds) --
    "https://www.ri.cmu.edu/feed/",                       # CMU Robotics Institute, open
    "https://www.cs.cmu.edu/news/feed.rss",               # CMU School of Computer Science, open
    "https://blog.ml.cmu.edu/feed/",                      # ML@CMU research blog (low-volume), open
    "https://hcii.cmu.edu/taxonomy/term/72/feed",         # CMU HCII human-computer interaction, open
    "https://www.sei.cmu.edu/news/feeds/latest/rss/",     # CMU SEI software engineering / AI / security, open
]


#: Set (once) to the ImportError text when the HTML body scraper cannot load.
#: Read by the v4 source-floor failure path so a missing package is never
#: misreported as an exhausted feed pool. Empty string = scraper is available.
_BODY_SCRAPER_UNAVAILABLE: str = ""


def body_scraper_unavailable() -> str:
    """The reason article-body scraping is off, or "" when it is working."""
    return _BODY_SCRAPER_UNAVAILABLE


def _fetch_full_article(url, timeout=20):
    """Fetch the full text of a science article from its URL.

    Fetches through the bounded seam (`_otr_feed_fetch`) and uses
    BeautifulSoup to strip HTML boilerplate and extract the article body.
    Returns all extracted text admitted by the secure 2 MiB fetch seam so the
    story engine gets methodology, findings, implications, and the article
    tail - not just the RSS teaser or an arbitrary local slice.

    Wave 5: the fetch is https-only, size-capped, redirect-capped and
    address-checked. The two failure classes are NOT the same thing and are
    handled differently on purpose. A `FeedFetchUnavailable` (paywall, bot
    block, 404, timeout) still returns "" so the caller degrades to the next
    candidate exactly as before. A `FeedFetchRefused` -- our own bound tripped,
    e.g. a redirect into the private network or a 40 MB body -- PROPAGATES.
    That is a defect in the URL or the configuration, and swallowing it is how
    this function spent an unknown number of episodes silently returning ""
    for a missing bs4 (see the scar below).

    The scraper tries a cascade of CSS selectors before falling back to
    the full document, so it handles sites that don't use semantic
    <article>/<main> tags (e.g. UCLA Newsroom, institutional press pages).
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        # A MISSING PACKAGE MUST NOT LOOK LIKE A PAYWALL.
        #
        # This used to `return ""`, silently. The caller then logged "scrape
        # blocked - RSS summary" and fell back to the RSS teaser, and the v4
        # source floor eventually failed the run with "No science RSS candidate
        # met the v4 source floor ... after inspecting 10 candidates" -- an error
        # that blames the FEEDS for a package that was never installed.
        #
        # Found 2026-07-14: bs4 was absent from the ComfyUI venv, so EVERY
        # science article body had been "" for as long as that was true. Every
        # science-sourced episode was written from a ~120-character headline
        # teaser instead of the methodology and findings this function exists to
        # fetch (a live probe after installing it returned 2,041 and 6,708
        # characters from the same feed, in 0.3s). The degradation was total,
        # permanent, and invisible.
        #
        # So say it once, LOUDLY, and record it so the floor's own failure
        # message can name the real cause instead of the feeds.
        global _BODY_SCRAPER_UNAVAILABLE
        if not _BODY_SCRAPER_UNAVAILABLE:
            _BODY_SCRAPER_UNAVAILABLE = str(exc)
            log.error(
                "[NewsFetcher] ARTICLE BODY SCRAPING IS DISABLED: %s. Every "
                "article will fall back to its RSS teaser (~100-300 chars), so "
                "the v4 source floor (>=400 chars) can only ever be met by the "
                "few feeds that publish full content:encoded. Stories will be "
                "written from headlines, not findings. Fix: "
                "pip install beautifulsoup4",
                exc,
            )
        return ""

    from ._otr_feed_fetch import (
        FeedFetchRefused, FeedFetchUnavailable, fetch_article,
    )

    try:
        document = fetch_article(url, deadline_s=timeout)
    except FeedFetchRefused:
        raise                      # our own bound -- never swallowed
    except FeedFetchUnavailable as exc:
        log.debug("[NewsFetcher] article body unavailable (%s): %s",
                  exc.reason, url)
        return ""

    try:
        soup = BeautifulSoup(document.text, "html.parser")

        # Strip boilerplate - nav, ads, footer, sidebar, scripts
        for tag in soup(["script", "style", "nav", "footer", "header",
                          "aside", "form", "noscript", "iframe"]):
            tag.decompose()

        # Cascade of content selectors - most specific to least.
        # Covers: semantic HTML5, WordPress/CMS class conventions,
        # institutional press release pages (UCLA, NIH, NSF, EurekaAlert).
        _SELECTORS = [
            "article",
            "main",
            '[class*="article-body"]',
            '[class*="article__body"]',
            '[class*="story-body"]',
            '[class*="entry-content"]',
            '[class*="post-content"]',
            '[class*="content-body"]',
            '[class*="wysiwyg"]',
            '[class*="rich-text"]',
            '[class*="body-copy"]',
            '[class*="release-body"]',      # EurekaAlert press releases
            '[class*="article-content"]',
            '[id*="article-body"]',
            '[id*="main-content"]',
            "div.content",
            "div.body",
        ]

        body = None
        for selector in _SELECTORS:
            body = soup.select_one(selector)
            if body:
                break
        if body is None:
            body = soup  # last resort - full stripped document

        # Extract paragraphs AND headings - h2/h3 carry section context
        # (methodology, implications, researcher quotes) that's often the
        # richest science content buried past the lede.
        content_tags = body.find_all(["p", "h2", "h3"])
        text = " ".join(tag.get_text(" ", strip=True) for tag in content_tags)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception:
        return ""


# 2026-04-29: News-history persistence path. Stores recently-used article
# URLs so the curator skips them on the next run.
#
# BUG-LOCAL-090 (2026-05-04 EVENING): moved from <repo>/config/news_history.json
# to <output>/otr/state/news_history.json. The repo is code; this is
# per-machine runtime state. Living under output/ aligns with where every
# other persistent OTR state lives (episodes/, obs/, etc.) and keeps the
# repo working tree clean. The legacy path is read-only -- on first run
# after migration the loader picks up legacy entries, the next save writes
# only to the new path, and from then on legacy is dead but harmless.
try:
    from . import _otr_paths as _OTR_PATHS  # type: ignore
    _NEWS_HISTORY_PATH = str(_OTR_PATHS.otr_state_dir() / "news_history.json")
except Exception:  # noqa: BLE001 -- defensive at import time
    _NEWS_HISTORY_PATH = os.path.join(
        os.path.expanduser("~"), ".otr_state", "news_history.json",
    )
_NEWS_HISTORY_LEGACY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "news_history.json",
)
_NEWS_HISTORY_MAX_ENTRIES = 200  # rolling window; oldest entries drop off
# 2026-05-04 (BUG-LOCAL-090): only block URLs used within this many days.
# Older entries are kept on disk for audit but no longer filter the pool,
# so a 5-day-old headline is fair game again. Without this, RSS feeds that
# rotate slowly (43-headline pool with 200-entry history) get filtered to
# zero and the fallback has to restore the unfiltered pool every run.
_NEWS_HISTORY_FILTER_DAYS = 5


def _read_news_history_file(path: str) -> list:
    """Read and JSON-parse the news_history file at ``path``. Returns
    the raw list (or empty list on any error). Used by both the new
    canonical path and the BUG-LOCAL-090 legacy migration fallback."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    except Exception:  # noqa: BLE001 -- best-effort
        return []


def _load_news_history() -> set[str]:
    """Return set of recently-used article URLs (within
    ``_NEWS_HISTORY_FILTER_DAYS`` days).

    Used to filter the candidate pool so back-to-back runs don't pick the
    same RSS feed top story. Entries older than the TTL window are kept
    on disk (for audit) but excluded from the active filter set so a
    headline can recycle into the pool after enough time has passed.

    BUG-LOCAL-090 migration: the canonical path is
    ``<output>/otr/state/news_history.json``. If the new path is missing
    or empty, fall back to the legacy ``<repo>/config/news_history.json``
    so a user's existing history carries forward on the first post-fix
    run. The next save writes only to the new path, after which legacy
    becomes stale-but-harmless.

    Failures return an empty set -- the dedup is best-effort, never
    blocks.
    """
    data = _read_news_history_file(_NEWS_HISTORY_PATH)
    if not data:
        # First-run fallback: pick up legacy entries from the
        # pre-BUG-090 path so the user keeps their dedup window.
        data = _read_news_history_file(_NEWS_HISTORY_LEGACY_PATH)

    cutoff = datetime.now() - timedelta(days=_NEWS_HISTORY_FILTER_DAYS)
    fresh: set[str] = set()
    for entry in data or []:
        url = (entry or {}).get("url")
        if not url:
            continue
        ts = (entry or {}).get("timestamp") or ""
        try:
            entry_dt = datetime.fromisoformat(ts) if ts else None
        except (TypeError, ValueError):
            entry_dt = None
        # Missing or unparseable timestamps -> treat as fresh (safer to
        # filter them once than to surface a same-day repeat).
        if entry_dt is None or entry_dt >= cutoff:
            fresh.add(url)
    return fresh


def _record_news_usage(url: str, headline: str) -> None:
    """Append (url, headline, timestamp) to news_history.json.

    Cap at _NEWS_HISTORY_MAX_ENTRIES rolling. Older entries drop off so the
    file never grows unbounded but recent picks are remembered.

    BUG-LOCAL-090 migration: writes go to the new canonical path
    (``<output>/otr/state/news_history.json``). On first save after
    migration, if the new path is empty/missing but legacy entries
    exist, the legacy list is loaded as the seed so the user's dedup
    window carries forward.
    """
    if not url:
        return
    try:
        # Read existing entries from new path; fall back to legacy if
        # new is empty/missing (one-time migration carry-forward).
        data = _read_news_history_file(_NEWS_HISTORY_PATH)
        if not data:
            data = _read_news_history_file(_NEWS_HISTORY_LEGACY_PATH)
        data.append({
            "url":          str(url),
            "headline":     str(headline)[:240],
            "timestamp":    datetime.now().isoformat(timespec="seconds"),
        })
        if len(data) > _NEWS_HISTORY_MAX_ENTRIES:
            data = data[-_NEWS_HISTORY_MAX_ENTRIES:]
        os.makedirs(os.path.dirname(_NEWS_HISTORY_PATH), exist_ok=True)
        with open(_NEWS_HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info("[NewsFetcher] Recorded usage: %s ... (%d total entries)",
                 headline[:60], len(data))
    except Exception as exc:  # noqa: BLE001 -- best-effort
        log.warning("[NewsFetcher] Failed to record news_history: %s", exc)


def _llm_rank_news_candidates(
    pool: list[dict],
    model_id: str = "mistralai/Mistral-Nemo-Instruct-2407",
    optimization_profile: str = "Standard",
    top_k: int = 5,
    load_config=None,
    policy=None,
) -> list[dict]:
    """Use the LLM to rank news headlines for genre-fit, return top_k.

    Cheap LLM call: short prompt (43 headlines x ~100 chars = ~5K chars),
    short response (just indices), temp=0.0 for deterministic ranking.
    Returns the top_k highest-ranked candidates ordered by LLM preference.

    On any failure (LLM unavailable, parse error, etc.) falls back to the
    original shuffled-order top_k. The downstream body-fetch loop still
    works -- LLM ranking is an enhancement, not a hard requirement.
    """
    if len(pool) <= top_k:
        return list(pool)
    try:
        # Trim to first 30 candidates to keep prompt bounded; pool is
        # already shuffled so no systematic bias.
        candidates = pool[:30]
        headline_list = "\n".join(
            f"{i + 1}. {(p.get('headline') or '').strip()[:160]}"
            for i, p in enumerate(candidates)
        )
        prompt = (
            f"You are picking news headlines for a radio drama episode. "
            f"From the numbered list below, choose the {top_k} "
            f"headlines with the strongest narrative potential -- prefer "
            f"specific events, mysteries, breakthroughs, or human stakes "
            f"over generic announcements or PR pieces.\n\n"
            f"Return ONLY the chosen indices, comma-separated, no other text. "
            f"Example: 3,7,12,18,22\n\n"
            f"Headlines:\n{headline_list}\n\n"
            f"Top {top_k} indices:"
        )
        # 2026-04-29: 65-second wall-clock budget for the curation LLM call
        # (Jeffrey requested: "give it 65 secs to do the search"). The
        # ranker is a short-output call (~64 tokens of indices) so under
        # normal conditions it returns in 5-15 sec. The 65s budget is a
        # ceiling, not a target -- it bounds the worst-case where prompt
        # processing on a cold cache or a 12B model takes longer than
        # expected. On timeout, _run_with_timeout raises _LLMTimeout and
        # the outer except block below falls back to shuffle order, so
        # the run continues without LLM ranking. The cache-invalidation
        # we added in BUG-LOCAL-111 (commit 27e54e9) also fires here so
        # the orphan worker doesn't poison the next phase's CUDA state.
        def _do_rank_call():
            # 2026-04-29 fix: transformers rejects temperature=0.0 with
            # "must be strictly positive". For greedy-deterministic
            # ranking, use a tiny positive (0.05) -- effectively argmax
            # but passes the validator. The ranking output is just a
            # comma-separated list of indices so any low-temp value
            # produces stable picks.
            #
            # LLM slot: technical -- structured short-output ranking
            # task. Caller composes the chat message; make_generate_fn
            # bakes top_p=0.92 (the canonical surface does not expose
            # per-call top_p override -- acceptable for a stochastic-
            # argmax ranker at temperature=0.05).
            from . import _otr_model_loader as _OTRML
            cache_entry = _OTRML.request_slot(
                "technical", model_id, policy=policy, load_config=load_config,
            )
            gen_fn = _OTRML.make_generate_fn(cache_entry)
            return gen_fn(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05,
                max_new_tokens=64,
            )
        response = _run_with_timeout(
            _do_rank_call,
            timeout_sec=65,
            phase_label="NewsCuration",
        )
        # Parse: extract integers, dedupe, cap at top_k
        seen = set()
        indices: list[int] = []
        for tok in re.split(r"[^\d]+", str(response or "")):
            if not tok:
                continue
            try:
                idx = int(tok) - 1  # 1-indexed in prompt -> 0-indexed
            except ValueError:
                continue
            if 0 <= idx < len(candidates) and idx not in seen:
                seen.add(idx)
                indices.append(idx)
            if len(indices) >= top_k:
                break
        if not indices:
            if load_config is not None:
                raise RuntimeError(
                    "[NewsFetcher] local GGUF news ranking returned no "
                    f"parseable indices (response={str(response)[:120]!r}); "
                    "refusing to silently fall back to shuffle order "
                    "(operator directive: no local-LM fallbacks)."
                )
            log.warning("[NewsFetcher] LLM ranking returned no parseable indices "
                        "(response=%r) - falling back to shuffle order",
                        str(response)[:120])
            return list(pool[:top_k])
        ranked = [candidates[i] for i in indices]
        log.info("[NewsFetcher] LLM-ranked top %d candidates:", len(ranked))
        for r in ranked:
            log.info("[NewsFetcher]   - %s", (r.get("headline") or "")[:80])
        return ranked
    except Exception as exc:  # noqa: BLE001 -- enhancement for non-GGUF lanes only
        if load_config is not None:
            log.error(
                "[NewsFetcher] local GGUF news ranking failed (%s); failing "
                "loud (operator directive: no local-LM fallbacks)", exc,
            )
            raise
        log.warning("[NewsFetcher] LLM ranking failed (%s) - falling back to "
                    "shuffle order", exc)
        return list(pool[:top_k])


def _llm_rerank_with_bodies(
    candidates_with_body: list[dict],
    model_id: str = "mistralai/Mistral-Nemo-Instruct-2407",
    optimization_profile: str = "Standard",
    load_config=None,
    policy=None,
) -> list[dict]:
    """Body-aware second-pass news rank ("Option B / 65s budget").

    Phase-1 ranking (`_llm_rank_news_candidates`) operates on headlines
    only -- ~160 chars each. This pass feeds the LLM the first ~800
    chars of each candidate's actual article body so the pick is based
    on narrative bones, not the catchy title. Returns the input list
    re-ordered (best first). On ANY failure (timeout, parse error, LLM
    unavailable) returns the input list unchanged so the caller's
    normal fallback walk still works. Body re-rank is an enhancement,
    never a blocker.

    Designed to fit inside the 65-second news-curation wall-clock
    budget alongside Phase 1: Phase 1 ~10-15s + parallel body-fetch
    ~5-10s + this re-rank ~25-40s ~= 50s total.
    """
    if len(candidates_with_body) <= 1:
        return list(candidates_with_body)
    try:
        blocks = []
        for i, c in enumerate(candidates_with_body):
            headline = (c.get("headline") or "").strip()[:160]
            body = (c.get("full_text") or c.get("summary") or "").strip()
            body_preview = _body_rerank_preview(body).replace("\n", " ")
            blocks.append(
                f"{i + 1}. HEADLINE: {headline}\n   ARTICLE: {body_preview}"
            )
        text = "\n\n".join(blocks)
        prompt = (
            f"You are picking ONE news story to seed a radio drama. "
            f"You have already shortlisted {len(candidates_with_body)} "
            f"candidates by headline. Now you can read each article body. "
            f"Choose the SINGLE story with the strongest narrative bones "
            f"for an audio drama: specific human stakes, "
            f"mystery, scientific breakthrough, or vivid scene potential. "
            f"Avoid press releases, funding announcements, and generic "
            f"'researchers find X' filler.\n\n"
            f"Return ONLY the chosen index, no other text. Example: 3\n\n"
            f"Candidates:\n{text}\n\n"
            f"Best index:"
        )

        def _do_rerank_call():
            # Same temperature=0.05 trick as headline rank: transformers
            # rejects 0.0; tiny positive value is effectively argmax.
            #
            # LLM slot: technical -- single-index body rerank.
            from . import _otr_model_loader as _OTRML
            cache_entry = _OTRML.request_slot(
                "technical", model_id, policy=policy, load_config=load_config,
            )
            gen_fn = _OTRML.make_generate_fn(cache_entry)
            return gen_fn(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.05,
                max_new_tokens=8,
            )

        response = _run_with_timeout(
            _do_rerank_call,
            timeout_sec=40,
            phase_label="NewsCurationDeep",
        )
        m = re.search(r"\d+", str(response or ""))
        if not m:
            if load_config is not None:
                raise RuntimeError(
                    "[NewsFetcher] local GGUF body re-rank returned no "
                    f"parseable index (response={str(response)[:120]!r}); "
                    "refusing to silently keep headline order "
                    "(operator directive: no local-LM fallbacks)."
                )
            log.warning(
                "[NewsFetcher] body re-rank returned no parseable index "
                "(response=%r) - keeping headline order",
                str(response)[:120],
            )
            return list(candidates_with_body)
        idx = int(m.group(0)) - 1
        if not (0 <= idx < len(candidates_with_body)):
            if load_config is not None:
                raise RuntimeError(
                    f"[NewsFetcher] local GGUF body re-rank index {idx + 1} out "
                    f"of range (have {len(candidates_with_body)}); refusing to "
                    "silently keep headline order (operator directive: no "
                    "local-LM fallbacks)."
                )
            log.warning(
                "[NewsFetcher] body re-rank index %d out of range "
                "(have %d) - keeping headline order",
                idx + 1, len(candidates_with_body),
            )
            return list(candidates_with_body)
        chosen = candidates_with_body[idx]
        rest = [c for i, c in enumerate(candidates_with_body) if i != idx]
        log.info(
            "[NewsFetcher] body re-rank chose #%d: %s",
            idx + 1, (chosen.get("headline") or "")[:80],
        )
        return [chosen] + rest
    except Exception as exc:  # noqa: BLE001 -- enhancement for non-GGUF lanes only
        if load_config is not None:
            log.error(
                "[NewsFetcher] local GGUF body re-rank failed (%s); failing "
                "loud (operator directive: no local-LM fallbacks)", exc,
            )
            raise
        log.warning(
            "[NewsFetcher] body re-rank failed (%s) - keeping headline order",
            exc,
        )
        return list(candidates_with_body)


_RSS_FRAGMENT_BLOCK_OR_BREAK_RE = re.compile(
    r"</?(?:address|article|aside|blockquote|br|dd|details|dialog|div|dl|dt|"
    r"fieldset|figcaption|figure|footer|form|h[1-6]|header|hgroup|hr|li|main|"
    r"""nav|ol|p|pre|section|table|tbody|td|tfoot|th|thead|tr|ul)(?=[\s/>])"""
    r"""(?:[^"'<>]|"[^"]*"|'[^']*')*>""",
    re.IGNORECASE,
)
_RSS_FRAGMENT_ANY_TAG_RE = re.compile(
    r"""<(?:[^"'<>]|"[^"]*"|'[^']*')*>"""
)


def _extract_rss_fragment_text(fragment: str) -> str:
    """Extract text from an inline RSS HTML fragment without fusing blocks.

    Only explicit block/break tags create a separator. Remaining inline tags
    are removed without one so ``H<sub>2</sub>O`` and
    ``anti-<em>microbial</em>`` retain their literal spelling. HTML entity
    spellings are deliberately left untouched for the downstream coordinate
    normalizer.
    """
    text = fragment or ""
    text = _RSS_FRAGMENT_BLOCK_OR_BREAK_RE.sub(" ", text)
    text = _RSS_FRAGMENT_ANY_TAG_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _select_rss_content(entry) -> tuple[str, int | None, int]:
    """Choose the longest usable RSS ``content`` alternative.

    Raw list positions are preserved for the source receipt. Malformed rows
    still count, but cannot displace a usable alternative. A non-list top-level
    value is not an alternatives collection and is treated as absent.
    """
    raw = entry.get("content", []) if hasattr(entry, "get") else []
    if not isinstance(raw, list):
        return "", None, 0
    best_text = ""
    best_index: int | None = None
    for index, row in enumerate(raw):
        if not hasattr(row, "get"):
            continue
        value = row.get("value")
        if not isinstance(value, str):
            continue
        try:
            extracted = _extract_rss_fragment_text(value)
        except Exception as exc:  # one malformed alternative cannot hide later rows
            log.debug(
                "[NewsFetcher] RSS content alternative %d was unusable: %s",
                index, exc,
            )
            continue
        if extracted and len(extracted) > len(best_text):
            best_text = extracted
            best_index = index
    return best_text, best_index, len(raw)


def _select_news_body(candidate: dict, fetched_body: str) -> tuple[str, str]:
    """Choose the most complete clean body with stable route tie-breaking."""
    rss = str(candidate.get("rss_full") or "")
    article = str(fetched_body or "")
    summary = str(candidate.get("summary") or "")
    # max() keeps the first member on ties: RSS, then linked article, then
    # summary. The route for a selected summary records whether a URL existed.
    choices = (
        (rss, "rss_full"),
        (article, "url_scrape"),
        (
            summary,
            "summary_fallback" if candidate.get("link") else "summary_only",
        ),
    )
    return max(choices, key=lambda item: len(item[0]))


def _body_rerank_preview(text: str, limit: int = 800) -> str:
    """Return a bounded head/middle/tail view without modifying source text."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("body preview limit must be a positive integer")
    body = str(text or "")
    if len(body) <= limit:
        return body
    separator = " ... "
    if limit <= 2 * len(separator) + 2:
        return body[:limit]
    available = limit - 2 * len(separator)
    head_chars = available // 3
    middle_chars = available // 3
    tail_chars = available - head_chars - middle_chars
    middle_start = max(0, (len(body) - middle_chars) // 2)
    return separator.join((
        body[:head_chars],
        body[middle_start:middle_start + middle_chars],
        body[-tail_chars:],
    ))


def _fetch_science_news(max_feeds=10,  # kept: max_feeds is API stability arg; current body iterates the full feed list. Wiring is a future feature, not a cleanbreak target
                         model_id=None, optimization_profile="Standard",
                         *, load_config=None, policy=None):
    """Fetch science stories from multiple RSS feeds in parallel.

    2026-04-29: now also (a) filters out previously-used URLs via
    config/news_history.json, (b) calls the LLM to rank remaining
    candidates by narrative fit, and (c) records the chosen article to
    history after selection.

    Style-engine consolidation (2026-07-05): this stage runs BEFORE the
    single style engine (which needs script_brief, not yet produced) --
    ranking is style-agnostic by design; the `style` parameter and its
    hardcoded "mission_control_procedural" fallback are removed entirely.

    Original fast-path behaviour (shuffle + first-with-enough-body) is
    preserved when model_id is None or LLM ranking fails -- the dedup
    still works regardless. Shipped behind model_id so legacy callers
    without it fall back to the simple path.

    Uses ThreadPoolExecutor to hit all feeds simultaneously, dramatically
    reducing the wait time when feeds are slow or unresponsive. Each feed
    has its own timeout.
    """
    try:
        import feedparser
    except ImportError:
        msg = (
            "-==================================================================-\n"
            "-  CRITICAL: feedparser is missing.                              -\n"
            "-  Run `pip install feedparser` to enable live science news.     -\n"
            "-  The OTR ScriptWriter REQUIRES real headlines - no fallback.   -\n"
            "-==================================================================-"
        )
        log.error(msg)
        raise ImportError(msg)

    from ._otr_feed_fetch import FeedFetchRefused, fetch_feed

    def _fetch_single_feed(feed_url):
        data = []
        try:
            # Wave 5: the bytes arrive through the bounded seam and feedparser
            # is handed a STRING. It never touches the network here.
            #
            # What this replaced: `feedparser.parse(feed_url)` -- which does its
            # own unbounded urllib fetch -- wrapped in a PROCESS-GLOBAL
            # socket.setdefaulttimeout(7). That global was set and restored by
            # every worker in a ~30-wide thread pool concurrently, so the
            # timeout any given feed actually ran under was whatever another
            # thread had most recently installed. It was never a per-feed
            # timeout; it only looked like one.
            feed = feedparser.parse(fetch_feed(feed_url).text)

            for entry in feed.entries[:6]:
                title = entry.get("title", "").strip()
                if not title:
                    continue


                rss_full, rss_content_index, rss_content_count = (
                    _select_rss_content(entry)
                )
                summary = entry.get("summary", "").strip()
                summary = re.sub(r'<[^>]+>', '', summary).strip()
                data.append({
                    "headline": title,
                    "summary": summary,
                    "rss_full": rss_full,
                    "source": feed.feed.get("title", feed_url.split("/")[2]),
                    "date": entry.get("published", str(datetime.now().date())),
                    "link": entry.get("link", ""),
                    "_rss_content_index": rss_content_index,
                    "_rss_content_count": rss_content_count,
                })
            return data
        except FeedFetchRefused:
            # A shipped or client-authored feed row that is http://, points at
            # a private address, or serves something that is not a feed is a
            # CONFIGURATION defect, not a flaky feed. Let it out of the pool
            # worker so `future.result()` re-raises it and the run fails loud.
            raise
        except Exception as e:
            log.debug("[NewsFetcher] Feed failed %s: %s", feed_url, e)
            return []

    pool = []
    feeds_hit = 0
    shuffled_feeds = SCIENCE_NEWS_FEEDS[:]
    random.shuffle(shuffled_feeds)

    log.info("[NewsFetcher] Starting parallel fetch from %d sources...", len(shuffled_feeds))
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=len(shuffled_feeds)) as executor:
        futures = {executor.submit(_fetch_single_feed, url): url for url in shuffled_feeds}
        for future in as_completed(futures):
            results = future.result()
            if results:
                pool.extend(results)
                feeds_hit += 1

    fetch_time = time.time() - start_time
    log.info("[NewsFetcher] Parallel fetch complete in %.2fs. Pool: %d headlines from %d feeds.",
             fetch_time, len(pool), feeds_hit)

    if not pool:
        log.error("[NewsFetcher] ALL feeds failed - check network connectivity")
        raise RuntimeError(
            "No science headlines could be fetched. Check your internet connection. "
            "The OTR ScriptWriter requires live RSS feeds to generate scripts."
        )

    # 2026-04-29: history-aware deduplication + LLM-curated ranking.
    # 1) drop any candidate whose URL is in news_history.json (back-to-
    #    back runs no longer pick the same Orion Flywheel article).
    # 2) shuffle the remaining pool to break feed-order bias.
    # 3) optionally call the LLM to rank top 5 by narrative fit for the
    #    requested style. This step adds ~10-30s of LLM time but
    #    the LLM is the same one NewsSummary will load anyway, so the
    #    NewsSummary phase that follows hits a cache HIT instead of
    #    paying the load cost twice.
    # 2026-04-29 BUG-LOCAL-112: history-wipe restoration. The previous
    # implementation had a comment admitting the reset was a "no-op" --
    # if every URL in the fresh fetch was already in news_history.json,
    # the filter emptied `pool` and the fall-through logged a warning
    # but never restored the pool. Result: body-fetch saw 0 candidates,
    # writer fell back to no-news, news-seeded plot was lost.
    # Real fix: stash the unfiltered pool before filtering, restore it
    # if the filter wipes everything. The history dedup still wins on
    # the typical day; the reset only fires when every fresh headline
    # is in history (which means the rolling cap is too small for the
    # user's run cadence and we'd rather repeat than starve).
    unfiltered_pool = list(pool)
    used_urls = _load_news_history()
    if used_urls:
        before = len(pool)
        pool = [
            p for p in pool
            if not (p.get("link") and p["link"] in used_urls)
        ]
        dropped = before - len(pool)
        if dropped:
            log.info(
                "[NewsFetcher] Filtered %d previously-used candidate(s) via "
                "news_history (%d remaining of %d)",
                dropped, len(pool), before,
            )
    if not pool:
        # All N candidates were already used. Restore the unfiltered
        # pool -- better to pick a recent repeat than to starve the
        # writer with zero news context.
        log.warning(
            "[NewsFetcher] All %d candidate(s) filtered out by history -- "
            "restoring unfiltered pool so the writer still gets a real "
            "article (history dedup will catch up as new headlines come "
            "in; consider raising the rolling cap if this happens often)",
            len(unfiltered_pool),
        )
        pool = list(unfiltered_pool)
        used_urls = set()

    random.shuffle(pool)

    # LLM rank: only if model_id provided and pool is non-trivial. This
    # spends one short LLM call up-front to pick narrative-fit
    # candidates. The LLM stays warm for NewsSummary which fires next.
    if model_id and len(pool) > 5:
        ranked = _llm_rank_news_candidates(
            pool,
            model_id=model_id,
            optimization_profile=optimization_profile,
            top_k=5,
            load_config=load_config,
            policy=policy,
        )
        # Put LLM-ranked picks at the front of the pool; everything
        # else stays as a fallback in case all 5 ranked picks have
        # thin bodies.
        ranked_links = {r.get("link") for r in ranked if r.get("link")}
        non_ranked = [p for p in pool if p.get("link") not in ranked_links]
        pool = ranked + non_ranked

    # 2026-04-29 Option B: parallel body-fetch + LLM body-aware re-rank.
    # Old behavior: serial walk-the-list, break at first candidate above
    # the content floor. Time spent: only as long as candidate-1 fetch
    # took. New behavior: body-fetch ALL top-N in parallel (network-
    # bound, fast), then ask the LLM to re-pick using actual article
    # text instead of just the headline. Total budget ~50s, comfortably
    # under the 65s news-curation ceiling. The LLM stays warm for
    # NewsSummary which fires next, so the re-rank's GPU time is not
    # wasted.
    #
    # Thin content (<400 chars) gives the writer too little to
    # extrapolate from -- the story ends up generic rather than
    # grounded in real science. Anything below the floor is excluded
    # from the re-rank pool.
    CONTENT_FLOOR = 400
    MAX_ATTEMPTS = 5

    def _resolve_body(candidate: dict) -> dict:
        """Body resolver for one candidate. Pure; thread-safe."""
        out = dict(candidate)
        fetched_body = (
            _fetch_full_article(out["link"], timeout=5)
            if out.get("link") else ""
        )
        out["full_text"], out["_body_source"] = _select_news_body(
            out, fetched_body,
        )
        log.info(
            "[NewsFetcher] [%s] selected %s body: %d chars",
            (out.get("headline") or "")[:50],
            out["_body_source"],
            len(out["full_text"]),
        )
        return out

    attempts = pool[:MAX_ATTEMPTS]
    log.info(
        "[NewsFetcher] Body-fetching top %d candidate(s) in parallel...",
        len(attempts),
    )
    body_start = time.time()
    # Cap workers at len(attempts) -- ThreadPoolExecutor errors on max_workers=0.
    with ThreadPoolExecutor(max_workers=max(1, len(attempts))) as ex:
        fetched = list(ex.map(_resolve_body, attempts))
    log.info(
        "[NewsFetcher] Body-fetch complete in %.2fs",
        time.time() - body_start,
    )

    rich = [
        candidate for candidate in fetched
        if len(candidate.get("full_text", "")) >= CONTENT_FLOOR
    ]

    if rich:
        log.info(
            "[NewsFetcher] %d/%d candidate(s) passed content floor "
            "(>=%d chars%s) -> body re-rank",
            len(rich), len(fetched), CONTENT_FLOOR, "",
        )
        if model_id and len(rich) > 1:
            rich = _llm_rerank_with_bodies(
                rich,
                model_id=model_id,
                optimization_profile=optimization_profile,
                load_config=load_config,
                policy=policy,
            )
        chosen = rich[0]
    else:
        # All candidates thin - take the richest available so the run
        # continues. Better a thin real story than a hard fail.
        chosen = max(
            fetched,
            key=lambda x: len(x.get("full_text", x.get("summary", ""))),
        )
        chosen.setdefault("full_text", chosen.get("summary", ""))
        log.warning(
            "[NewsFetcher] All %d candidate(s) were thin - using richest "
            "available (%d chars): %s",
            len(fetched), len(chosen["full_text"]),
            chosen.get("headline", "")[:60],
        )

    # Record selection so back-to-back runs don't repeat. Best-effort;
    # logged warning on failure, never blocks generation.
    try:
        _record_news_usage(
            url=chosen.get("link", ""),
            headline=chosen.get("headline", ""),
        )
    except Exception as _hist_exc:  # noqa: BLE001
        log.warning("[NewsFetcher] history record failed (non-fatal): %s",
                    _hist_exc)

    return [chosen]


# -----------------------------------------------------------------------------
# LLM INFERENCE WRAPPER
# -----------------------------------------------------------------------------

# ── Token Budget Ratios ──────────────────────────────────────────────────────
# target_words * ratio = max_new_tokens. Different content types tokenize at
# different rates. Radio drama is dialogue-dominant (~60% character lines),
# so structural overhead (VOICE tags, SFX, ENV, scene headers) is lower than
# a screenplay or narration-heavy format.
#
# Breakdown for dialogue-dominant OTR scripts:
#   tokenizer overhead:  ~1.3 tokens per English word
#   script markup:       ~1.2x (VOICE/SFX/ENV tags, scene headers, beats)
#   combined:            1.3 * 1.2 = 1.56 → round to 1.6
#
# Revision/rewrite passes include structural reorganization → higher overhead.
# Outlines and pitches are almost entirely non-dialogue description.
_TOKEN_RATIO_DIALOGUE = 1.6    # dialogue-dominant (OTR radio drama default)
_TOKEN_RATIO_MIXED = 2.0       # revision/rewrite passes, structural changes
_TOKEN_RATIO_OUTLINE = 2.2     # outlines, pitches, descriptions
_TOKEN_RATIO_ACT_CHUNK = 2.0   # per-act chunked generation (needs slack for act boundaries)
_TOKEN_RATIO_ACT_OBSIDIAN = 2.5  # Obsidian 4GB: wider slack for constrained KV cache


# ── Intelligent Dialogue Name Normalizer (BUG-023) ──────────────────────────
# LLMs at high temperature produce creative dialogue formatting that breaks
# the standard NAME: regex. This normalizer strips ALL variants down to
# canonical FIRSTNAME LASTNAME: format before any word-count or parsing runs.
#
# Handles:
#   **FIRST LAST**, concerned: text    → FIRST LAST: text
#   *FIRST LAST*(angry): text          → FIRST LAST: text
#   __FIRST LAST__: text               → FIRST LAST: text
#   FIRST_LAST: text                   → FIRST LAST: text
#   *FIRST_LAST*, whispering: text     → FIRST LAST: text
#   [FIRST LAST, traits] text          → FIRST LAST: text  (BUG-LOCAL-063)
#   [FIRST, mood] text                 → FIRST: text       (BUG-LOCAL-063)
#
# Standard NAME: lines pass through unchanged (regex only fires on decorated
# names — plain uppercase + colon is a no-op match that rewrites identically).
_RE_LLM_DIALOGUE_NAME = re.compile(
    r'^'
    r'[*_]{0,2}'                           # leading **, *, __, _
    r'([A-Z][A-Z0-9_ ]{0,25})'            # character name (may have underscores)
    r'[*_]{0,2}'                           # trailing **, *, __, _
    r'(?:\s*[,(]\s*[a-z][a-z ]*[)]?)?'    # optional emotion: ", concerned" or "(angry)"
    r':(?=\s)',                             # colon followed by whitespace (avoid SFX:rumble)
    re.MULTILINE
)

# BUG-LOCAL-063 (2026-04-24): The normalizer's promise --
# "All downstream consumers (word-count regex, FORMAT_NORM, PARSE) see clean
# text" -- broke when Mistral Nemo's preferred [NAME, mood] text shorthand
# landed here unrecognized. Run #4 produced a clean script with ~18 lines of
# `[MINDSY, Female, 30s, Urgent, Determined] Kane, I've crunched the numbers!`
# style dialogue, but WORD_ENFORCEMENT's dialogue-counting regex (which looks
# for `NAME:` form) saw zero and triggered the rescue pipeline, which then
# discarded itself, producing a near-empty final MP4. Pattern below rewrites
# bracket-shorthand to canonical `NAME:` form so every downstream counter
# sees the same clean text. Structural tokens (ENV/SFX/MUSIC/VOICE/ACT/SCENE/
# BEAT/PAUSE/TRANSITION/FADE/CUT/INT/EXT) pass through untouched.
_BRACKET_STRUCTURAL_TOKENS = frozenset({
    "ENV", "SFX", "MUSIC", "VOICE", "ACT", "SCENE", "BEAT", "PAUSE",
    "TRANSITION", "CONTINUED", "CONT", "END", "FADE", "CUT", "INT", "EXT",
    "TITLE", "NOTE", "TARGET", "STYLE", "NARRATOR", "SYSTEM_SENTINEL",
    "OPENING", "CLOSING", "INTERSTITIAL",
})
_RE_LLM_BRACKET_NAME_DIALOGUE = re.compile(
    r'^'
    r'\['                                   # opening bracket
    r'([A-Z][A-Z0-9_ ]{1,25})'             # character name (may have underscores/spaces)
    r'(?:,\s*[^\]]*?)?'                    # optional traits list (non-greedy)
    r'\]'                                   # closing bracket
    r'\s+(?=\S)',                           # at least one space, dialogue follows
    re.MULTILINE,
)


def _normalize_dialogue_names(text):
    """Intelligent LLM output normalizer — strips all creative formatting
    variants down to canonical NAME: format in one pass.

    Called once before WORD_EXTEND (Step 0) and once on extension LLM output.
    All downstream consumers (word-count regex, FORMAT_NORM, PARSE) see clean text.
    """
    def _clean_colon(m):
        name = m.group(1).strip().replace('_', ' ')
        # Collapse multiple spaces (from stripped underscores or padding)
        name = ' '.join(name.split())
        return f'{name}:'
    # Pass 1: bracket-shorthand '[NAME, mood] dialogue' -> 'NAME: dialogue'.
    # Skip structural bracketed tokens ([ENV:...], [SFX:...], [VOICE: NAME...],
    # [ACT TWO], [SCENE 3]). The [VOICE: NAME, ...] form is handled by Pass 2
    # after the bracket is stripped and the NAME: colon form remains.
    def _clean_bracket(m):
        raw_name = m.group(1).strip().replace('_', ' ')
        first_word = raw_name.split()[0] if raw_name else ''
        if first_word.upper() in _BRACKET_STRUCTURAL_TOKENS:
            return m.group(0)  # leave structural tags untouched
        name = ' '.join(raw_name.split())
        return f'{name}: '
    text = _RE_LLM_BRACKET_NAME_DIALOGUE.sub(_clean_bracket, text)
    # Pass 2: classic colon + bold/underscore decorated forms.
    return _RE_LLM_DIALOGUE_NAME.sub(_clean_colon, text)


# ── Scene inventory (diagnostic instrumentation) ────────────────
# Extracts the list of scene tokens from a script so the orchestrator
# can log scene counts at every pipeline checkpoint. A scene leak in
# any pass (WORD_EXTEND, ANNOUNCER, FORMAT_NORM, GRAMMARIAN, PARSE)
# shows up as a count drop in the soak log, localizing the bug.
#
# BUG-LOCAL-026 fix: restrict the scene-number capture to digits only.
# The previous pattern (\S+?) matched literals like "FINAL" that the
# creative LLM emits as a closing-scene marker ('=== SCENE FINAL ==='),
# inflating scene counts and fooling the FORMAT_NORM skip heuristic.
# Any '=== SCENE FINAL ===' is promoted to 'END' (terminator) below.
_RE_SCENE_MARKER = re.compile(
    r'===\s*SCENE\s+(\d+)(?:\s*:\s*[^=]*?)?\s*===',
    re.IGNORECASE
)
_RE_SCENE_TERMINATOR = re.compile(
    r'===\s*SCENE\s+FINAL\b[^=]*===',
    re.IGNORECASE
)


def _scene_inventory(text):
    """Return the ordered list of scene tokens found in the script.

    Recognizes canonical '=== SCENE N ===' markers (numeric only) and a
    trailing '=== SCENE FINAL ===' terminator. Returns tokens like
    ['1', '2', '3'] or ['1', '2', 'END']. Empty list means no scene
    markers present (valid for short scripts pre-FORMAT_NORM).
    """
    if not text:
        return []
    tokens = [m.group(1) for m in _RE_SCENE_MARKER.finditer(text)]
    if _RE_SCENE_TERMINATOR.search(text):
        tokens.append("END")
    return tokens


def _log_scene_checkpoint(stage, text):
    """Emit a SCENE_TRACK log line for the given pipeline stage."""
    tokens = _scene_inventory(text)
    _runtime_log(
        f"SCENE_TRACK: {stage} | count={len(tokens)} | tokens={tokens}"
    )
    return tokens


# ── Name cleanup (fuzzy match against canonical cast) ────────────
# BUG-020 fix: Under maximum chaos, LLMs hallucinate variant spellings
# (NEMEO_SIRIKIT instead of NEMO SIRIKIT). This pure-Python pass reads
# the canonical cast from config/episode_cast.txt and fuzzy-matches
# every CHARACTER: line against the roster. No LLM call, no VRAM cost.

def _name_similarity(a: str, b: str) -> float:
    """Simple character-level similarity ratio (0.0 to 1.0).
    Uses longest common subsequence length / max length.
    Good enough for catching NEMEO->NEMO without pulling in difflib."""
    a, b = a.upper(), b.upper()
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    # Levenshtein-style: count matching chars in order
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    matches = 0
    last_idx = -1
    for ch in shorter:
        for j in range(last_idx + 1, len(longer)):
            if longer[j] == ch:
                matches += 1
                last_idx = j
                break
    return matches / max(len(a), len(b))


def _cleanup_character_names(script_text: str, cast_config_path: str,
                             pre_rolled_cast: list) -> str:
    """Consistency-focused name cleanup for LLM-generated scripts.

    Two jobs:
      1. CONSISTENCY: If the same character appears as both "NEMO" (50 lines)
         and "NEMEO" (2 lines), collapse the rare variant into the dominant one.
      2. GARBLE DETECTION: If a name is very close to a canonical cast name
         (similarity > 0.75) but misspelled, fix it. This catches maximum-chaos
         hallucinations like NEMEO_SIRIKIT -> NEMO SIRIKIT.

    Does NOT force names to match the roster rigidly. "DR NEMO" or "CAPTAIN NEMO"
    are fine as long as the LLM uses them consistently.

    Pure Python - no LLM call, no VRAM cost, runs in milliseconds.

    Args:
        script_text: The full script after FORMAT_NORM
        cast_config_path: Path to config/episode_cast.txt
        pre_rolled_cast: Fallback list of canonical names

    Returns:
        Script text with corrected character names
    """
    # Build canonical name set from config file or fallback
    canonical_names = set()
    try:
        with open(cast_config_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                name_part = line.split("|")[0].strip().upper()
                if name_part:
                    canonical_names.add(name_part)
    except (FileNotFoundError, OSError):
        pass

    if not canonical_names and pre_rolled_cast:
        canonical_names = {n.upper() for n in pre_rolled_cast}

    # Always include fixed names that should never be "corrected"
    canonical_names.update({"LEMMY", "ANNOUNCER"})

    # Names that should never be touched (structural, not characters)
    _skip_names = frozenset({
        "SCENE", "ACT", "NOTE", "TARGET", "STYLE", "SFX",
        "ENV", "NARRATOR", "OPENING", "CLOSING", "MUSIC",
        "VOICE", "SOUND", "FADE", "CUT", "TRANSITION",
    })

    # Extract all unique names from the script WITH occurrence counts
    bare_names = re.findall(r'^([A-Z][A-Z0-9_ ]{1,30}):', script_text, re.MULTILINE)
    voice_names = re.findall(r'\[VOICE:\s*([A-Z][A-Z0-9_ ]+)[,\]]', script_text)
    all_occurrences = [n.strip().upper() for n in bare_names + voice_names]

    # Count occurrences per name
    name_counts = {}
    for n in all_occurrences:
        if n not in _skip_names:
            name_counts[n] = name_counts.get(n, 0) + 1

    if not name_counts:
        _runtime_log("NAME_CLEANUP: No character names found in script - skipping")
        return script_text

    # --- PASS 1: CONSISTENCY ---
    # Group similar names together. If one variant has 2 uses and another
    # has 40, the rare one is a typo. Collapse into the dominant variant.
    replacements = {}
    processed = set()

    # Sort by frequency (most common first) so dominant names anchor groups
    sorted_names = sorted(name_counts.items(), key=lambda x: -x[1])

    for dominant_name, dominant_count in sorted_names:
        if dominant_name in processed or dominant_name in _skip_names:
            continue
        processed.add(dominant_name)

        for rare_name, rare_count in sorted_names:
            if rare_name in processed or rare_name in _skip_names:
                continue
            if rare_name == dominant_name:
                continue

            sim = _name_similarity(rare_name, dominant_name)
            # High similarity + dominant name is much more frequent = typo
            if sim >= 0.75 and dominant_count >= rare_count * 3:
                replacements[rare_name] = dominant_name
                processed.add(rare_name)
                _runtime_log(
                    f"NAME_CLEANUP: CONSISTENCY '{rare_name}' ({rare_count}x) -> "
                    f"'{dominant_name}' ({dominant_count}x) [sim={sim:.2f}]"
                )

    # --- PASS 2: GARBLE DETECTION (against canonical roster) ---
    # Only for names not already handled in pass 1 and not in canonical set.
    for script_name, count in name_counts.items():
        if script_name in processed or script_name in canonical_names:
            continue
        if script_name in _skip_names:
            continue

        best_match = None
        best_score = 0.0
        for canon in canonical_names:
            if canon in _skip_names:
                continue
            score = _name_similarity(script_name, canon)
            if score > best_score:
                best_score = score
                best_match = canon

        # Only fix if very high similarity (0.75+) - this catches
        # NEMEO->NEMO but leaves DR NEMO alone (low similarity to NEMO)
        if best_match and best_score >= 0.75 and count <= 3:
            replacements[script_name] = best_match
            _runtime_log(
                f"NAME_CLEANUP: GARBLE '{script_name}' ({count}x) -> "
                f"'{best_match}' [sim={best_score:.2f}]"
            )

    if not replacements:
        _runtime_log(
            f"NAME_CLEANUP: {len(name_counts)} unique names, all consistent - "
            f"no fixes needed"
        )
        return script_text

    # Apply replacements (longest first to avoid partial matches)
    for bad_name, good_name in sorted(replacements.items(), key=lambda x: -len(x[0])):
        # Replace in bare format: BADNAME: -> GOODNAME:
        script_text = re.sub(
            rf'^{re.escape(bad_name)}:', f'{good_name}:',
            script_text, flags=re.MULTILINE
        )
        # Replace in VOICE tags: [VOICE: BADNAME, -> [VOICE: GOODNAME,
        script_text = script_text.replace(
            f'[VOICE: {bad_name},', f'[VOICE: {good_name},'
        )
        script_text = script_text.replace(
            f'[VOICE: {bad_name}]', f'[VOICE: {good_name}]'
        )

    _runtime_log(
        f"NAME_CLEANUP: Fixed {len(replacements)} name variant(s): "
        f"{', '.join(f'{k}->{v}' for k, v in replacements.items())}"
    )
    return script_text


# ── Dual-format dialogue extraction ─────────────────────────────
# Scripts may contain dialogue in EITHER format:
#   Bare:  NAME: dialogue text
#   VOICE: [VOICE: NAME, emotion] "dialogue text"
# The word-count regex and WORD_EXTEND character extraction must
# recognize BOTH to avoid false zero-dialogue detection (BUG-025).

_RE_BARE_DIALOGUE = re.compile(
    r'^([A-Z][A-Z0-9_ ]{1,25}):\s+(.+)$', re.MULTILINE
)
_RE_VOICE_TAG_DIALOGUE = re.compile(
    r'^\[VOICE:\s*([A-Z][A-Z0-9_ ]+)[,\]].*?\]\s*["\u201C]?(.+?)["\u201D]?\s*$',
    re.MULTILINE
)

_DIALOGUE_FALSE_POSITIVES = frozenset({
    "SCENE", "ACT", "NOTE", "TARGET", "STYLE", "SFX",
    "ENV", "NARRATOR", "OPENING", "CLOSING", "MUSIC",
    "ANNOUNCER",
    # BUG-LOCAL-037: BUG-LOCAL-035's TITLE: <name> first-line convention
    # was being parsed as a "TITLE" character speaking the title text.
    # Block it everywhere a NAME: <text> shape gets read as dialogue.
    "TITLE",
})

def _extract_all_dialogue(text):
    """Extract (name, dialogue) pairs from both bare and VOICE-tag formats.

    Returns a list of (character_name, dialogue_text) tuples with false
    positives (SFX, ENV, SCENE, ANNOUNCER, etc.) already filtered out.
    """
    bare = [
        (name.strip(), dialogue)
        for name, dialogue in _RE_BARE_DIALOGUE.findall(text)
        if name.strip() not in _DIALOGUE_FALSE_POSITIVES
    ]
    voice = [
        (name.strip(), dialogue)
        for name, dialogue in _RE_VOICE_TAG_DIALOGUE.findall(text)
        if name.strip() not in _DIALOGUE_FALSE_POSITIVES
    ]
    return bare + voice




# Register the LLM unloader with the VRAM Power Wash system so that
# force_vram_offload() at node entry points also evicts the LLM.
# `register_vram_cleanup`'s caller (`force_vram_offload` in
# `_vram_log.py`) already wraps each callback invocation in
# `try/except: pass`, so the callback contract is "no-arg callable"
# only -- no wrapper required.
from ._vram_log import register_vram_cleanup
from . import _otr_model_loader as _otr_loader_mod

register_vram_cleanup(_otr_loader_mod.unload_llm)


class GemmaHeartbeatStreamer(BaseStreamer):
    """Custom streamer that pulses heartbeats to _runtime_log for Canonical Tokens.

    Hooks into model.generate() at the token level. Every time Gemma completes
    a line that contains a recognizable script tag (=== SCENE, [VOICE:], [SFX:],
    [ENV:], (beat)), it writes a timestamped entry to otr_runtime.log immediately.

    Also tracks:
      - Scene count (how many === SCENE === tags so far)
      - Dialogue line count (how many [VOICE:] tags)
      - Unique character names seen
      - Token generation speed (tokens/sec, reported every 100 tokens)

    The OTR Monitor (otr_monitor.py) tails otr_runtime.log and folds these
    heartbeats into the live dashboard - so you can watch the script being
    written in real time without touching ComfyUI.
    """

    def __init__(self, tokenizer, skip_prompt=False, live_ledger=False, **decode_kwargs):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs

        # v1.5: Incremental decoding state (Resolves O(N^2) complexity)
        self.token_cache = []
        self.print_len = 0
        self.line_buffer = ""

        self.on_prompt_end = True
        self.print_streamer = TextStreamer(tokenizer, skip_prompt=skip_prompt, **decode_kwargs)

        # Counters for the dashboard
        self.scene_count = 0
        self.dialogue_count = 0
        self.sfx_count = 0
        self.characters_seen = set()

        # Token speed tracking
        self.total_tokens = 0
        self._start_time = time.time()
        self._last_speed_report = 0

        # L1.5 live-ledger streaming hook. Only the script-body call site
        # passes live_ledger=True; spine/critique/revision/grammarian
        # streamers leave the singleton untouched. The body streamer resets
        # the singleton at __init__ so a fresh episode starts clean, then
        # writes partial ledger snapshots every _LEDGER_THROTTLE_LINES new
        # dialogue lines. The post-parse end-hook overwrites with the
        # canonical parsed cast + lines on the SAME file.
        self.live_ledger = bool(live_ledger)
        self._streamed_lines = []   # list of (name, full_text)
        self._streamed_chars = {}   # name -> char_id
        self._last_ledger_save_at = 0
        self._LEDGER_THROTTLE_LINES = 3
        # NOTE 2026-04-29: removed `new_ledger()` here. write_script() now
        # owns ledger creation at workflow entry (much earlier than this
        # streamer construction), so calling new_ledger() here would WIPE
        # the early ledger that's been accumulating gen_params + git_commit
        # + meta state during NewsFetcher / model load / OpenClose Spine.
        # The streamer's _emit_partial_ledger / _record_streamed_line use
        # get_ledger() (which returns the singleton, creates one if missing)
        # so this branch is no longer needed for first-write semantics.
        # If write_script's early init fails for any reason, get_ledger()
        # still creates a placeholder on first read -- no functional regression.

    def put(self, value):
        """Processes a new batch of tokens incrementally."""
        # Check strict streaming timeout
        if hasattr(_TIMEOUT_CTX, "deadline") and time.time() > _TIMEOUT_CTX.deadline:
            raise TimeoutError("Streaming deadline exceeded - gracefully aborting generator")

        # Standard console output
        self.print_streamer.put(value)

        # -- Token-level processing --
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("GemmaHeartbeatStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.on_prompt_end:
            self.on_prompt_end = False
            return

        token_list = value.tolist()
        self.token_cache.extend(token_list)
        self.total_tokens += len(token_list)

        # v1.5: Incremental decoding logic (adapted from transformers.TextStreamer)
        # Instead of decoding EVERYTHING every token, we only decode the new slice.
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        
        # Determine the "new" text generated in this step
        if text.endswith("\n") or text.endswith("\r"):
            new_text = text[self.print_len:]
            self.line_buffer += new_text
            self._process_line(self.line_buffer.strip())
            self.line_buffer = ""
            self.print_len = len(text)
        elif text.endswith(" ") or text.endswith(".") or text.endswith("!") or text.endswith("?"):
            # Partial line update
            new_text = text[self.print_len:]
            self.line_buffer += new_text
            self.print_len = len(text)
        
        # Report speed every 25 tokens (v1.5 CLEAN: higher heartbeat frequency)
        if self.total_tokens - self._last_speed_report >= 25:
            elapsed = time.time() - self._start_time
            if elapsed > 0:
                tps = self.total_tokens / elapsed
                _runtime_log(
                    f"ScriptWriter: {self.total_tokens} tokens | "
                    f"{tps:.1f} tok/s | {self.scene_count} scenes | "
                    f"{self.dialogue_count} lines | "
                    f"{len(self.characters_seen)} chars"
                )
                self._last_speed_report = self.total_tokens

    def end(self):
        """Flush the remaining buffer and report final stats."""
        self.print_streamer.end()
        # v1.5: Flush any remaining incremental line_buffer content
        if self.token_cache:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            remaining = text[self.print_len:]
            self.line_buffer += remaining
        if self.line_buffer.strip():
            self._process_line(self.line_buffer.strip())
        self.token_cache = []
        self.line_buffer = ""
        self.print_len = 0

        elapsed = time.time() - self._start_time
        tps = self.total_tokens / elapsed if elapsed > 0 else 0
        _runtime_log(
            f"ScriptWriter DONE: {self.total_tokens} tokens in {elapsed:.1f}s "
            f"({tps:.1f} tok/s) | {self.scene_count} scenes | "
            f"{self.dialogue_count} dialogue lines | "
            f"Characters: {', '.join(sorted(self.characters_seen)) or 'none'}"
        )
        # L1.5: flush any tail lines that hadn't crossed the throttle
        # threshold so the viewer sees the final draft before critique
        # runs and overwrites.
        if self.live_ledger and self._streamed_lines:
            self._emit_partial_ledger()

    # -- L1.5 live-ledger helpers --------------------------------------
    def _record_streamed_line(self, name, full_text):
        """Track a fresh dialogue line and write a partial ledger snapshot.

        Only fires when self.live_ledger=True (script-body call site).
        Throttled to one save per _LEDGER_THROTTLE_LINES new lines so the
        on-disk file isn't rewritten on every token. Failures are
        swallowed -- the ledger never blocks streaming.
        """
        if not self.live_ledger:
            return
        try:
            self._streamed_lines.append((name, full_text or ""))
            if name not in self._streamed_chars:
                self._streamed_chars[name] = f"c{len(self._streamed_chars)+1:02d}"
            if (self.dialogue_count - self._last_ledger_save_at
                    >= self._LEDGER_THROTTLE_LINES):
                self._emit_partial_ledger()
                self._last_ledger_save_at = self.dialogue_count
        except Exception:
            pass

    def _emit_partial_ledger(self):
        """Save the running cast + lines accumulated during streaming."""
        try:
            from .production_ledger import get_ledger
            led = get_ledger()
            cast_rows = [
                {"char_id": cid, "name": name}
                for name, cid in self._streamed_chars.items()
            ]
            line_rows = []
            for i, (name, text) in enumerate(self._streamed_lines):
                line_rows.append({
                    "line_id":  f"l{i+1:03d}",
                    "char_id":  self._streamed_chars.get(name),
                    "text":     text,
                })
            led.set_cast(cast_rows)
            led.set_lines(line_rows)
            led.save()
        except Exception:
            pass

    def _process_line(self, line):
        """Detect Canonical Tokens, update counters, pulse the heartbeat."""
        if not line:
            return

        # -- Scene break: === SCENE X === -----------------------------
        if "===" in line:
            self.scene_count += 1
            _runtime_log(f"ScriptWriter: {line.strip()}")
            return

        # -- Voice tag: [VOICE: NAME, traits] dialogue ----------------
        if "[VOICE:" in line.upper():
            self.dialogue_count += 1
            try:
                # Case-insensitive tag extraction
                line_up = line.upper()
                start_idx = line_up.find("[VOICE:") + 7
                end_idx = line.find("]", start_idx)
                tag_content = line[start_idx:end_idx]

                name = tag_content.split(",", 1)[0].strip().upper()
                self.characters_seen.add(name)

                # Full dialogue (untruncated) for the ledger; trim only
                # the trailing punctuation/markdown decorations.
                full_dialogue = line[end_idx+1:].strip().strip('"*_“”')
                clean_dialogue = full_dialogue[:60]
                _runtime_log(f"ScriptWriter: [{self.dialogue_count}] {name}: {clean_dialogue}")
                self._record_streamed_line(name, full_dialogue)
            except (IndexError, ValueError):
                _runtime_log(f"ScriptWriter: Voice line #{self.dialogue_count}")
            return

        # -- SFX tag --------------------------------------------------
        if "[SFX:" in line:
            self.sfx_count += 1
            try:
                desc = line.split("[SFX:", 1)[1].split("]", 1)[0].strip()
                _runtime_log(f"ScriptWriter: SFX #{self.sfx_count}: {desc[:50]}")
            except (IndexError, ValueError):
                _runtime_log(f"ScriptWriter: SFX #{self.sfx_count}")
            return

        # -- ENV tag --------------------------------------------------
        if "[ENV:" in line:
            try:
                desc = line.split("[ENV:", 1)[1].split("]", 1)[0].strip()
                _runtime_log(f"ScriptWriter: ENV: {desc[:50]}")
            except (IndexError, ValueError):
                pass
            return

        # -- Bare "CHARACTER: dialogue" format (BUG-007 format) --------
        # The LLM often writes "DALE: I heard something" instead of
        # [VOICE: DALE, traits] tags. Detect NAME: at start of line.
        # BUG-023: normalize Markdown bold before matching
        import re
        line = _normalize_dialogue_names(line)
        bare_match = re.match(r'^([A-Z][A-Z0-9_ ]{1,25}):\s+(.+)', line)
        if bare_match:
            name = bare_match.group(1).strip()
            # Skip false positives like "SCENE:", "SFX:", "ENV:", "NOTE:"
            # BUG-LOCAL-037: TITLE added so the streaming heartbeat does not
            # mistake the writer-prompt's "TITLE: <...>" first line for a
            # character speaking the title text.
            if name not in ("SCENE", "SFX", "ENV", "NOTE", "NARRATOR",
                            "ACT", "OPENING", "CLOSING", "TARGET", "STYLE",
                            "TITLE"):
                self.dialogue_count += 1
                self.characters_seen.add(name)
                full_dialogue = bare_match.group(2).strip().strip('"*_“”')
                clean_dialogue = full_dialogue[:60]
                _runtime_log(f"ScriptWriter: [{self.dialogue_count}] {name}: {clean_dialogue}")
                self._record_streamed_line(name, full_dialogue)
                return

        # -- Beat pause -----------------------------------------------
        if "(beat)" in line.lower():
            return  # beats are too frequent to log individually


# -----------------------------------------------------------------------------
# v1.4 Theme B - Sentence-boundary truncation helpers
#
# Replace the old `acts[-1][:3000]` / `acts[-1][-500:]` hard slices with
# boundary-aware truncation so the chunked context never hands Phase B a
# half-sentence. Both helpers fall back to hard truncation if no sentence
# boundary is found within a reasonable scan window - telemetry, not magic.
# -----------------------------------------------------------------------------

# Sentence-ending punctuation recognized by the boundary walkers.
_SENTENCE_END_CHARS = ".!?"
# How far to walk looking for a boundary before giving up and hard-cutting.
_BOUNDARY_SCAN_WINDOW = 300


def _truncate_at_sentence_boundary(text, max_chars):
    """Truncate `text` at the last sentence boundary before `max_chars`.

    Walks backward from the cut point looking for sentence-ending punctuation
    (`.`, `!`, `?`) followed by whitespace or end-of-string, or a blank-line
    paragraph break. If nothing is found in the last `_BOUNDARY_SCAN_WINDOW`
    characters the function falls back to a hard cut so the caller never gets
    an oversized string.
    """
    if not text or len(text) <= max_chars:
        return text
    snippet = text[:max_chars]
    lower_bound = max(0, len(snippet) - _BOUNDARY_SCAN_WINDOW)
    for i in range(len(snippet) - 1, lower_bound, -1):
        ch = snippet[i]
        if ch in _SENTENCE_END_CHARS:
            next_ch = snippet[i + 1] if i + 1 < len(snippet) else ""
            if next_ch in ("", " ", "\n", "\r", "\t", '"', "'"):
                return snippet[: i + 1]
        if ch == "\n" and i + 1 < len(snippet) and snippet[i + 1] == "\n":
            return snippet[:i]
    return snippet


# This lets the next phase automatically inherit the exact same model memory
# space the Script Writer loaded without requiring the user to sync two disjointed dropdowns.
# -----------------------------------------------------------------------------
_CURRENT_LLM_MODEL = "mistralai/Mistral-Nemo-Instruct-2407"


# ============================================================================
# SHARED INFERENCE ENGINE
# Both nodes call this loader. It caches the model in VRAM and tracks the peak
# memory watermark for diagnostics.
# ============================================================================


def _tail_at_sentence_boundary(text, max_chars):
    """Return the trailing region of `text` starting at a sentence boundary.

    Used for the "last N chars for dialogue continuity" case. Walks forward
    from `len(text) - max_chars` looking for the start of a fresh sentence so
    the caller never receives a tail that begins mid-word. If no sentence
    boundary is found within the scan window, falls back to the next word
    boundary (space or newline) so the tail still never starts mid-word.
    """
    if not text or len(text) <= max_chars:
        return text
    start = len(text) - max_chars
    snippet = text[start:]
    scan = min(_BOUNDARY_SCAN_WINDOW, len(snippet) - 1)
    for i in range(scan):
        ch = snippet[i]
        if ch in _SENTENCE_END_CHARS:
            next_ch = snippet[i + 1] if i + 1 < len(snippet) else ""
            if next_ch in (" ", "\n", "\r", "\t"):
                return snippet[i + 2 :].lstrip()
        if ch == "\n" and i + 1 < len(snippet) and snippet[i + 1] == "\n":
            return snippet[i + 2 :].lstrip()
    # Word-boundary fallback: no sentence end found in the window, so at least
    # start from the next whitespace so the tail is not mid-word.
    for i in range(min(50, len(snippet))):
        if snippet[i] in (" ", "\n", "\r", "\t"):
            return snippet[i + 1 :].lstrip()
    return snippet


# -----------------------------------------------------------------------------
# v1.4 Theme B - Automatic scene transitions
#
# When Gemma writes back-to-back scenes without any handoff cue, the audio
# engine has nothing to work with and the result sounds like a hard cut. This
# helper detects adjacent `=== SCENE N ===` markers with no transition in
# between and injects a `[TRANSITION: brief pause]` placeholder. Downstream
# SceneSequencer and BatchBark treat transition cues as audio beats.
# -----------------------------------------------------------------------------

_SCENE_MARKER_RE = re.compile(r"===\s*SCENE\s+\S+\s*===", re.IGNORECASE)
_HANDOFF_CUE_RE = re.compile(
    r"\[TRANSITION:|\[FADE\b|\[SFX:[^\]]*transition",
    re.IGNORECASE,
)


def _inject_scene_transitions(script_text):
    """Inject `[TRANSITION: brief pause]` between scenes lacking a handoff cue.

    Walks adjacent scene markers in reverse so each insertion does not disturb
    the offsets of earlier matches. Returns a tuple of (new_text, injections).
    """
    if not script_text:
        return script_text, 0
    matches = list(_SCENE_MARKER_RE.finditer(script_text))
    if len(matches) < 2:
        return script_text, 0

    injections = 0
    for idx in range(len(matches) - 1, 0, -1):
        prev_end = matches[idx - 1].end()
        curr_start = matches[idx].start()
        gap = script_text[prev_end:curr_start]
        if _HANDOFF_CUE_RE.search(gap):
            continue
        script_text = (
            script_text[:curr_start]
            + "[TRANSITION: brief pause]\n\n"
            + script_text[curr_start:]
        )
        injections += 1
    return script_text, injections


# -----------------------------------------------------------------------------
# NODE 1: SCRIPT WRITER
# -----------------------------------------------------------------------------

# Path to the SIGNAL LOST canon file. Read at write_script time (not at
# module load) so editing the canon between runs takes effect on the
# next render without restarting ComfyUI.
_CANON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "OTR-CANON.md",
)


def _load_canon_for_writer(skip: bool = False, compact: bool = False) -> str:
    """Build a writer-prompt canon block from OTR-CANON.md.

    Pulls the writer-relevant sections (Tonal canon, Period rules,
    Recurring motifs, Used premises/twists/motifs) and wraps them in
    an XML-flavored block. The original consumer (SCAFFOLDING_PREAMBLE)
    was deleted in Sprint C C2b as orphan code; this function is itself
    orphan-pending-Sprint-G-audit. Returns the empty string on:
      - skip=True (legacy small-model collapse guard, kept for back-compat)
      - missing file (logs warning, never fatal)
      - parse error (logs warning, never fatal)

    `compact=True` returns a shrunk-down period+tonal anchor that
    still fits comfortably in a small-model's context (Gemma-4 E2B,
    ~2B effective). Drops the "Used premises / twists / motifs"
    sections (they exist for de-duplication, not period anchoring)
    and the prose-heavy "Tonal canon" / "Recurring motifs" sections.
    Keeps only "Period rules" -- the part that actually moves the
    needle on small-model anachronism leakage (BUG-LOCAL-114-class
    issues caught by the critic's A15-A19 rules).

    The returned block has NO `{placeholders}` so `.format()` on the
    full system prompt won't choke on canon content.
    """
    if skip:
        return ""
    try:
        if not os.path.exists(_CANON_PATH):
            log.warning("[Canon] OTR-CANON.md missing at %s - skipping", _CANON_PATH)
            return ""
        with open(_CANON_PATH, "r", encoding="utf-8") as _f:
            text = _f.read()
    except Exception as exc:  # noqa: BLE001
        log.warning("[Canon] read failed (%s) - skipping", exc)
        return ""
    # Pull only the sections the writer needs. Order matters (rules
    # before "do not repeat" list).
    # Compact mode keeps ONLY the section that gives small-model
    # writers a real period anchor (Gemma-4 E2B-class). Larger models
    # get the full canon including motifs + de-dup hints.
    if compact:
        keep_headers = ("## Period rules",)
    else:
        keep_headers = (
            "## Tonal canon",
            "## Period rules",
            "## Recurring motifs",
            "## Used premises (auto-updated)",
            "## Used twists (auto-updated)",
            "## Used motifs (auto-updated)",
        )
    parts: list[str] = []
    for header in keep_headers:
        idx = text.find(header)
        if idx < 0:
            continue
        # Find the next "## " heading (or end of file)
        next_idx = text.find("\n## ", idx + len(header))
        if next_idx < 0:
            section = text[idx:]
        else:
            section = text[idx:next_idx]
        # Strip trailing blank lines
        parts.append(section.rstrip())
    if not parts:
        return ""
    body = "\n\n".join(parts)
    # Escape literal braces so .format() doesn't try to interpolate
    # any "{...}" tokens that may appear in canon prose.
    body = body.replace("{", "{{").replace("}", "}}")
    block = (
        "<canon>\n"
        "The following is established SIGNAL LOST canon. Honour it. Do not\n"
        "contradict the tonal rules, period rules, or recurring motifs.\n"
        "Avoid repeating any premise, twist, or motif listed under\n"
        "\"Used ... (auto-updated)\" -- pick a fresh angle for this\n"
        "episode.\n\n"
        f"{body}\n"
        "</canon>\n\n"
    )
    return block


# ============================================================================
# 2026-05-15 Sprint C C2b: SCAFFOLDING_PREAMBLE + SCRIPT_SYSTEM_PROMPT
# constants deleted. Both were orphaned during the eec4718 LPL extraction
# sprint and had zero live consumers across nodes/, visual/, scripts/, and
# tests/ (verified via 8-search audit at C2b: getattr / glob-import /
# substring-content / alt-name / git log -S / writer-callsite / __all__ /
# tests). The current ledger pipeline composes prompts from per-phase
# system constants in _otr_outline, _otr_line_composer,
# _otr_ledger_reviewer, and _otr_period_prompts.
#
# No back-compat shim. Per the standing directive (no legacy back-compat),
# orphan constants from a deleted pipeline are deleted, not preserved.
# Future contributors who try `from .story_orchestrator import
# SCRIPT_SYSTEM_PROMPT` will get AttributeError -- intentional, so dead
# wirings fail loud.
#
# Broader orphan-constant sweep of this file (3000+ lines, gutted across
# multiple sprints) deferred to Sprint G per the no-scope-creep rule.
# Each candidate gets its own 8-search audit before deletion.
# ============================================================================



# ---------------------------------------------------------------------------
# 2026-05-10: LegacyLLMScriptWriter shim removed alongside
# nodes/_otr_legacy_writer.py. The v2.0 canonical writer is
# OTR_LedgerScriptWriter (LPL). Any caller that still does
# `from .story_orchestrator import LLMScriptWriter` will now raise
# AttributeError -- intentional, so stale wirings fail loudly instead of
# silently using a dead code path.
# ---------------------------------------------------------------------------
