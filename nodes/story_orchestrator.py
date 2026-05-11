r"""
OTR Orchestrator - Script Writer + Director for "SIGNAL LOST"
===================================================================

Two nodes:
  1. LLMScriptWriter - Fetches real daily science news via RSS, feeds it to
     LLM to generate a full audio drama script. Contemporary sci-fi anthology
     format (Black Mirror / NPR Invisibilia / Arrival). News-as-spine: real
     headlines become the inciting incident, extrapolated to dramatic extremes.
     Includes a hard-science epilogue citing real sources (ArXiv, Nature, etc.).

  2. LLMDirector - Takes a finished script and generates a production plan:
     TTS voice assignments, SFX cue list, music cues, timing, and spatial audio
     settings. Outputs structured JSON that drives all downstream nodes.

LLM runs via transformers (local GPU). Content safety filter catches
profanity/NSFW that slips past the prompt policy.

v1.0  2026-04-04  Jeffrey Brick
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
import socket
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
from ._otr_paths import director_raw_dump_dir


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


# ---------------------------------------------------------------------------
# Parse-check helper for the script-writer retry loop (BUG-LOCAL-004 + 005).
#
# Extracted to module scope 2026-05-02 so tests can lock the parse contract
# without driving the whole `write_script` method end-to-end. The loop in
# `write_script` uses `_check_parse_ok(text, is_ultra_smoke=..., is_tiny_smoke=...)`
# and acts on the returned mapping.
#
# Round-robin verdict 2026-05-02 (gpt-5.5 / gemini-3.1-pro-customtools /
# nemotron-49b): the bare-form regex MUST negative-lookahead the structural
# markers (TITLE/SCENE/GENRE/ENV/SFX/MUSIC/VOICE/CAST/AUTHOR) so a degenerate
# "TITLE: only" output cannot falsely PARSE_OK. The [VOICE: ...] regex stays
# strict to mirror the downstream parser at story_orchestrator.py:3021,
# avoiding C7 violations where the orchestrator passes a script the parser
# then drops dialogue from.
# ---------------------------------------------------------------------------
import re as _re  # noqa: E402 -- placed after _flush_vram_keep_llm

_PARSE_CHECK_SCENE_RE = _re.compile(
    r'^\s*===\s*SCENE\s+\d+\s*===\s*$', _re.MULTILINE
)
_PARSE_CHECK_VOICE_RE = _re.compile(
    r'^\s*\[VOICE:\s*[A-Z_]+', _re.MULTILINE
)
_PARSE_CHECK_BARE_RE = _re.compile(
    r'^\s*(?!TITLE\b|SCENE\b|GENRE\b|ENV\b|SFX\b|MUSIC\b|VOICE\b|CAST\b|AUTHOR\b)'
    r'[A-Z][A-Z0-9_ ]{1,19}:\s*\S',
    _re.MULTILINE,
)


def _check_parse_ok(
    text: str,
    *,
    is_ultra_smoke: bool = False,
    is_tiny_smoke: bool = False,
) -> dict:
    """Cheap structural-marker parseability check for script-writer output.

    Returns a dict with keys:
        has_scene  -- bool, `=== SCENE N ===` marker present at least once
        voice_hits -- int, count of `[VOICE: NAME...]` line starts
        bare_hits  -- int, count of bare `CHARACTER:` line starts (excluding
                      structural markers like TITLE/SCENE/GENRE/etc.)
        parse_ok   -- bool, branch-aware verdict:
                        ultra-smoke -> has_scene AND voice_hits >= 2
                        tiny-smoke  -> has_scene AND voice_hits >= 4
                        standard    -> has_scene AND (voice_hits + bare_hits) > 0

    The cap of voice_hits >= 2 for ultra-smoke matches the prompt's REQUIRED
    OUTPUT structure (2 character lines plus 2 ANNOUNCER lines == 4 voice
    lines; require >= 2 to tolerate one missing line). For tiny-smoke the
    template asks for ~6-8 voice lines; require >= 4 to allow modest droppage.
    """
    text = text or ""
    has_scene = bool(_PARSE_CHECK_SCENE_RE.search(text))
    voice_hits = len(_PARSE_CHECK_VOICE_RE.findall(text))
    bare_hits = len(_PARSE_CHECK_BARE_RE.findall(text))

    if is_ultra_smoke:
        parse_ok = has_scene and voice_hits >= 2
    elif is_tiny_smoke:
        parse_ok = has_scene and voice_hits >= 4
    else:
        parse_ok = has_scene and (voice_hits + bare_hits) > 0

    return {
        "has_scene": has_scene,
        "voice_hits": voice_hits,
        "bare_hits": bare_hits,
        "parse_ok": parse_ok,
    }


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

            # 2026-04-29 BUG-LOCAL-111 fix: timeout-recovery cache invalidation.
            #
            # When FuturesTimeout fires, the worker thread is still running an
            # LLM forward pass on GPU. Python cannot safely terminate threads,
            # and `executor.shutdown(wait=False)` below does NOT kill the
            # worker -- it keeps churning until its forward pass completes
            # naturally (could be 30-60+ more seconds for a 16K prompt).
            # Result: the GPU has in-flight kernels the main thread doesn't
            # control. The cached model instance in _LLM_CACHE thinks it's
            # idle but the orphan is still mutating its tensors. The NEXT
            # phase that calls model.cpu() / _load_llm() / any CUDA op
            # collides with the orphan's stale ops and Python aborts with
            # `cudaErrorIllegalAddress`.
            #
            # Fix: invalidate _LLM_CACHE so the NEXT _load_llm() call forces
            # a fresh load from disk. The orphan worker's tensors get
            # garbage-collected naturally once its frame returns; we just
            # don't try to reuse them. Cost: ~10-15s extra reload time on
            # the next phase. Benefit: no crash, run continues.
            try:
                global _LLM_CACHE
                _LLM_CACHE["model"]     = None
                _LLM_CACHE["tokenizer"] = None
                _LLM_CACHE["model_id"]  = None
                _LLM_CACHE["device"]    = None
                _LLM_CACHE["context_cap"] = None
                # Free our REFERENCES to the cached model. Python's gc will
                # release the actual VRAM when the orphan worker also drops
                # its reference. We don't try to forcibly empty CUDA cache
                # here -- the orphan is still using it and an empty_cache
                # call from the main thread would race the kernel writes.
                _runtime_log(
                    f"TIMEOUT_RECOVERY: invalidated _LLM_CACHE so next "
                    f"phase forces a fresh load (avoids "
                    f"cudaErrorIllegalAddress from orphan {phase_label} "
                    f"worker still on GPU)"
                )
            except Exception as _recovery_exc:  # noqa: BLE001
                log.warning(
                    "[Timeout] cache invalidation failed (next phase may "
                    "still crash on stale CUDA state): %s",
                    _recovery_exc,
                )

            raise _LLMTimeout(f"{phase_label} exceeded {timeout_sec}s")
    finally:
        # Don't wait for the orphaned worker - let it drain in the background.
        # Combined with the cache invalidation above, the orphan completes
        # in its own time WITHOUT the next phase trying to reuse its tensors.
        executor.shutdown(wait=False)


# -----------------------------------------------------------------------------
# CAST CONSOLIDATION HELPERS
# Used by LLMDirector cast-merge to collapse near-duplicate cast rows that
# arise from prefix-overlap (LLOYD vs LLOYD KAPOOR) or LLM typo divergence
# (STANLEY vs STANLEARY). BUG-LOCAL-068 expansion 2026-04-26 PM.
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
# for the rest of the session, so the Director can never assign a broken
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
        from .bark_tts import _load_bark
        from .batch_bark_generator import _generate_single_line
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

    NOTE: As of 2026-04-26 PM the orchestrator no longer calls this from
    `LLMDirector.direct()`; the new lazy `_bark_health_check_for_cast()`
    runs after cast assignment instead. This function is kept exported
    for any external caller and as a manual catalog-validation tool.
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

    Trades a one-time ~120s full-catalog warmup at Director start for a
    ~5-25s targeted check after Director assigns voices. On any preset
    failure, swaps the cast row's voice_preset for a known-good fallback
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
# CONTENT SAFETY FILTER - catches profanity/NSFW that slips past the prompt
# -----------------------------------------------------------------------------

# Word list: common profanity, slurs, and explicit terms.
# Kept as a set for O(1) lookup. Checked against whole-word boundaries only
# so words like "assembly" or "hell" (in sci-fi context) aren't false-flagged.
_BLOCKED_WORDS = {
    # profanity
    "fuck", "fucking", "fucked", "fucker", "motherfucker", "motherfucking",
    "shit", "shitting", "shitty", "bullshit",
    "damn", "damned", "dammit", "goddamn", "goddammit",
    "ass", "asshole", "arse", "arsehole",
    "bitch", "bitches", "bastard", "crap", "crappy",
    "piss", "pissed", "pissing",
    "dick", "cock", "pussy", "tits", "boobs",
    "whore", "slut", "skank",
    # slurs (abbreviated to avoid reproducing full slurs in source)
    "nigger", "nigga", "faggot", "fag", "retard", "retarded",
    "spic", "chink", "kike", "wetback", "coon",
    # violence-adjacent shock terms
    "disembowel", "dismember", "decapitate", "eviscerate",
    "rape", "raped", "raping", "molest",
}

# FIX-2 (v1.2): Minced-oath pool replaces [BLEEP] censor.
# Period-authentic 1940s radio euphemisms + pulp adventure + sci-fi flavor.
# Rotated per-replacement so the same script doesn't repeat the same oath twice.
_MINCED_OATHS = [
    # Golden-age radio (G-rated)
    "Golly", "Gee", "Gee whiz", "Jeepers", "Jiminy", "Jiminy Cricket",
    "Heavens", "Heavens to Betsy", "Good heavens", "My stars",
    "Land sakes", "Goodness gracious", "For Pete's sake", "By Jove",
    "Great Scott", "Cheese and crackers",
    # Pulp adventure
    "Blazes", "Thunderation", "Hot dog", "Holy smokes", "Holy cow",
    "Holy mackerel", "Suffering succotash", "Leapin' lizards",
    "Good grief", "Gadzooks", "Zounds",
    # Sci-fi space-opera
    "Stars above", "By the stars", "Great galaxies", "Holy vacuum",
    "Sweet cosmos", "By the rings", "Thundering comets", "Sputtering satellites",
]


def _content_filter(text: str) -> tuple:
    """Scrub blocked words from generated script text.

    Returns (cleaned_text, list_of_replacements_made).
    Uses whole-word regex matching to avoid false positives.
    Replacements rotate through _MINCED_OATHS (period-appropriate euphemisms)
    instead of emitting [BLEEP] - preserves the old-time-radio atmosphere.
    """
    replacements = []
    _oath_cursor = [0]  # list-wrapped so closure can mutate
    def _replace(match):
        word = match.group(0)
        replacements.append(word.lower())
        oath = _MINCED_OATHS[_oath_cursor[0] % len(_MINCED_OATHS)]
        _oath_cursor[0] += 1
        # Preserve capitalization style of the original word
        if word.isupper():
            return oath.upper()
        if word[0].isupper():
            return oath
        return oath.lower()

    # Build regex: whole-word match, case-insensitive
    if not _BLOCKED_WORDS:
        return text, []
    pattern = r'\b(?:' + '|'.join(re.escape(w) for w in sorted(_BLOCKED_WORDS, key=len, reverse=True)) + r')\b'
    cleaned = re.sub(pattern, _replace, text, flags=re.IGNORECASE)

    if replacements:
        log.warning("[ContentFilter] Replaced %d blocked word(s): %s",
                    len(replacements), ", ".join(set(replacements)))

    return cleaned, replacements


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


def _check_voice_consistency(script_text, pre_rolled_cast_traits):
    """Walk every [VOICE: NAME, gender, age, tone, energy] tag in the final
    script and compare the LLM-chosen traits against the pre-rolled cast
    traits. Returns a list of soft-warning dicts (no schema bump, written
    to ledger.voice_warnings[] for forward analysis). ANNOUNCER tags are
    skipped because the announcer alternates per episode and is not in the
    pre-rolled cast traits dict.

    Each warning dict shape:
        {
          "char_id": str (canonical NAME, all caps),
          "expected": {"gender", "age", "tone", "energy"},
          "actual":   {"gender", "age", "tone", "energy"},
          "mismatches": [str, ...]  # one short tag per drifted field
        }

    The check is forgiving: trait matches use substring-in-actual so
    "30s" pre-rolled matches "30s, anxious" actual, etc. A trait whose
    actual value is missing or unparseable is reported as
    'mismatch:absent'. Failure of the regex itself is swallowed -- the
    whole pipeline never blocks on a soft-warning collection.
    """
    if not pre_rolled_cast_traits:
        return []
    try:
        # [VOICE: NAME, gender, age, tone, energy]
        # Allow optional whitespace and accept gender words wrapped in <>
        # placeholders the LLM may leak from prompt scaffolding.
        pattern = re.compile(
            r'\[VOICE:\s*'
            r'([A-Z][A-Z0-9 _\-]+?)\s*,'   # name
            r'\s*<?([^,\]<>]+)>?\s*,'      # gender
            r'\s*<?([^,\]<>]+)>?\s*,'      # age
            r'\s*<?([^,\]<>]+)>?\s*,'      # tone
            r'\s*<?([^,\]<>]+)>?\s*\]',    # energy
            re.IGNORECASE,
        )
        warnings_out = []
        for m in pattern.finditer(script_text or ""):
            raw_name = (m.group(1) or "").strip().upper()
            if not raw_name or raw_name == "ANNOUNCER":
                continue
            traits = pre_rolled_cast_traits.get(raw_name)
            if not traits:
                continue
            exp_gender, exp_age, exp_tone, exp_energy = traits[:4]
            act_gender = (m.group(2) or "").strip().lower()
            act_age    = (m.group(3) or "").strip().lower()
            act_tone   = (m.group(4) or "").strip().lower()
            act_energy = (m.group(5) or "").strip().lower()
            mismatches = []
            if exp_gender.lower() not in act_gender:
                mismatches.append(f"gender:{exp_gender}!={act_gender}")
            if exp_age.lower() not in act_age:
                mismatches.append(f"age:{exp_age}!={act_age}")
            if exp_tone.lower() not in act_tone:
                mismatches.append(f"tone:{exp_tone}!={act_tone}")
            if exp_energy.lower() not in act_energy:
                mismatches.append(f"energy:{exp_energy}!={act_energy}")
            if mismatches:
                warnings_out.append({
                    "char_id":  raw_name,
                    "expected": {"gender": exp_gender, "age": exp_age,
                                 "tone": exp_tone,    "energy": exp_energy},
                    "actual":   {"gender": act_gender, "age": act_age,
                                 "tone": act_tone,    "energy": act_energy},
                    "mismatches": mismatches,
                })
        return warnings_out
    except Exception:  # noqa: BLE001 - soft-warning collection never blocks
        return []


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
]


def _fetch_full_article(url, timeout=20):
    """Fetch the full text of a science article from its URL.

    Uses requests + BeautifulSoup to strip HTML boilerplate and extract
    the article body. Returns the raw text (up to 12000 chars) so Gemma
    gets real science content - methodology, findings, implications -
    not just the RSS teaser. Falls back to empty string on any failure
    (paywalls, bot blocks, timeouts) so the caller can degrade gracefully.

    The scraper tries a cascade of CSS selectors before falling back to
    the full document, so it handles sites that don't use semantic
    <article>/<main> tags (e.g. UCLA Newsroom, institutional press pages).
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return ""

    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; OTR-ScriptBot/1.0)"}
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

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
        return text[:12000]
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


def _record_news_usage(url: str, headline: str, style: str = "") -> None:
    """Append (url, headline, genre, timestamp) to news_history.json.

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
            "style": str(style),
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
    style: str,
    model_id: str = "mistralai/Mistral-Nemo-Instruct-2407",
    optimization_profile: str = "Standard",
    top_k: int = 5,
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
        genre_human = (style or "sci-fi").replace("_", " ")
        prompt = (
            f"You are picking news headlines for a {genre_human} radio "
            f"drama episode. From the numbered list below, choose the {top_k} "
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
            return _generate_with_llm(
                prompt,
                model_id=model_id,
                max_new_tokens=64,
                temperature=0.05,
                top_p=1.0,
                optimization_profile=optimization_profile,
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
            log.warning("[NewsFetcher] LLM ranking returned no parseable indices "
                        "(response=%r) - falling back to shuffle order",
                        str(response)[:120])
            return list(pool[:top_k])
        ranked = [candidates[i] for i in indices]
        log.info("[NewsFetcher] LLM-ranked top %d candidates for '%s':",
                 len(ranked), genre_human)
        for r in ranked:
            log.info("[NewsFetcher]   - %s", (r.get("headline") or "")[:80])
        return ranked
    except Exception as exc:  # noqa: BLE001 -- enhancement, never blocks
        log.warning("[NewsFetcher] LLM ranking failed (%s) - falling back to "
                    "shuffle order", exc)
        return list(pool[:top_k])


def _llm_rerank_with_bodies(
    candidates_with_body: list[dict],
    style: str,
    model_id: str = "mistralai/Mistral-Nemo-Instruct-2407",
    optimization_profile: str = "Standard",
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
        BODY_PREVIEW_CHARS = 800
        blocks = []
        for i, c in enumerate(candidates_with_body):
            headline = (c.get("headline") or "").strip()[:160]
            body = (c.get("full_text") or c.get("summary") or "").strip()
            body_preview = body[:BODY_PREVIEW_CHARS].replace("\n", " ")
            blocks.append(
                f"{i + 1}. HEADLINE: {headline}\n   ARTICLE: {body_preview}"
            )
        text = "\n\n".join(blocks)
        genre_human = (style or "sci-fi").replace("_", " ")
        prompt = (
            f"You are picking ONE news story to seed a {genre_human} radio "
            f"drama. You have already shortlisted {len(candidates_with_body)} "
            f"candidates by headline. Now you can read each article body. "
            f"Choose the SINGLE story with the strongest narrative bones "
            f"for a 1940s-style radio drama: specific human stakes, "
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
            return _generate_with_llm(
                prompt,
                model_id=model_id,
                max_new_tokens=8,
                temperature=0.05,
                top_p=1.0,
                optimization_profile=optimization_profile,
            )

        response = _run_with_timeout(
            _do_rerank_call,
            timeout_sec=40,
            phase_label="NewsCurationDeep",
        )
        m = re.search(r"\d+", str(response or ""))
        if not m:
            log.warning(
                "[NewsFetcher] body re-rank returned no parseable index "
                "(response=%r) - keeping headline order",
                str(response)[:120],
            )
            return list(candidates_with_body)
        idx = int(m.group(0)) - 1
        if not (0 <= idx < len(candidates_with_body)):
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
    except Exception as exc:  # noqa: BLE001 -- enhancement, never blocks
        log.warning(
            "[NewsFetcher] body re-rank failed (%s) - keeping headline order",
            exc,
        )
        return list(candidates_with_body)


def _fetch_science_news(max_feeds=10, style="hard_sci_fi",
                         model_id=None, optimization_profile="Standard"):
    """Fetch science stories from multiple RSS feeds in parallel.

    2026-04-29: now also (a) filters out previously-used URLs via
    config/news_history.json, (b) calls the LLM to rank remaining
    candidates by narrative fit for the requested style, and
    (c) records the chosen article to history after selection.

    Original fast-path behaviour (shuffle + first-with-enough-body) is
    preserved when model_id is None or LLM ranking fails -- the dedup
    still works regardless. Shipped behind style + model_id so
    legacy callers without those args fall back to the simple path.

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

    def _fetch_single_feed(feed_url):
        data = []
        FEED_TIMEOUT = 7
        try:
            # Set socket timeout locally for this thread
            _prev_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(FEED_TIMEOUT)
            try:
                feed = feedparser.parse(feed_url)
            finally:
                socket.setdefaulttimeout(_prev_timeout)

            for entry in feed.entries[:6]:
                title = entry.get("title", "").strip()
                if not title:
                    continue

                # -- Headline pre-filter: reject non-article content ----------
                # Teaser/media headlines have no science payload for Gemma to
                # work with. A podcast slug or video title gives Gemma 90 chars
                # about content it can't access. Drop these at the source.
                _SKIP_PREFIXES = (
                    "podcast:", "watch:", "video:", "listen:", "opinion:",
                    "q&a:", "quiz:", "gallery:", "photos:", "slideshow:",
                    "live:", "webinar:", "event:", "in photos:",
                )
                _SKIP_PHRASES = (
                    "behind-the-scenes", "tour of", "in conversation with",
                    "ask the expert", "meet the", "alumni spotlight",
                    "faculty spotlight", "student spotlight", "donate",
                    "how to apply", "registration open",
                )
                title_lower = title.lower()
                if any(title_lower.startswith(p) for p in _SKIP_PREFIXES):
                    log.debug("[NewsFetcher] Skipping non-article (prefix): %s", title[:60])
                    continue
                if any(p in title_lower for p in _SKIP_PHRASES):
                    log.debug("[NewsFetcher] Skipping non-article (phrase): %s", title[:60])
                    continue
                # -------------------------------------------------------------

                content_candidates = entry.get("content", [])
                rss_full = ""
                if content_candidates:
                    rss_full = content_candidates[0].get("value", "")
                    rss_full = re.sub(r'<[^>]+>', '', rss_full).strip()
                summary = entry.get("summary", "").strip()
                summary = re.sub(r'<[^>]+>', '', summary).strip()
                data.append({
                    "headline": title,
                    "summary": summary,
                    "rss_full": rss_full,
                    "source": feed.feed.get("title", feed_url.split("/")[2]),
                    "date": entry.get("published", str(datetime.now().date())),
                    "link": entry.get("link", ""),
                })
            return data
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
            style=style,
            model_id=model_id,
            optimization_profile=optimization_profile,
            top_k=5,
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
        if out.get("rss_full") and len(out["rss_full"]) > 300:
            out["full_text"] = out["rss_full"]
            out["_body_source"] = "rss_full"
            log.info(
                "[NewsFetcher] [%s] RSS full body: %d chars",
                (out.get("headline") or "")[:50],
                len(out["full_text"]),
            )
        elif out.get("link"):
            fetched = _fetch_full_article(out["link"], timeout=5)
            if fetched and len(fetched) > 300:
                out["full_text"] = fetched
                out["_body_source"] = "url_scrape"
                log.info(
                    "[NewsFetcher] [%s] scraped article: %d chars",
                    (out.get("headline") or "")[:50],
                    len(out["full_text"]),
                )
            else:
                out["full_text"] = out.get("summary", "")
                out["_body_source"] = "summary_fallback"
                log.info(
                    "[NewsFetcher] [%s] scrape blocked - RSS summary (%d chars)",
                    (out.get("headline") or "")[:50],
                    len(out["full_text"]),
                )
        else:
            out["full_text"] = out.get("summary", "")
            out["_body_source"] = "summary_only"
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
        c for c in fetched
        if len(c.get("full_text", "")) >= CONTENT_FLOOR
    ]

    if rich:
        log.info(
            "[NewsFetcher] %d/%d candidate(s) passed content floor "
            "(>=%d chars) -> body re-rank",
            len(rich), len(fetched), CONTENT_FLOOR,
        )
        if model_id and len(rich) > 1:
            rich = _llm_rerank_with_bodies(
                rich,
                style=style,
                model_id=model_id,
                optimization_profile=optimization_profile,
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
            style=style,
        )
    except Exception as _hist_exc:  # noqa: BLE001
        log.warning("[NewsFetcher] history record failed (non-fatal): %s",
                    _hist_exc)

    # 2026-04-29: stamp the chosen article's identity into the live
    # ledger's meta.news_seed for per-episode auditability. Anyone
    # reading the ledger now knows EXACTLY which article seeded the
    # script, where it came from, and how big the body was. Pairs with
    # the existing per-episode ledger fields (cast, lines, clips) so a
    # render's full origin chain is on disk in one file.
    try:
        from .production_ledger import get_ledger as _get_led
        _led = _get_led()
        _led.data.setdefault("meta", {})["news_seed"] = {
            "headline":     str(chosen.get("headline", ""))[:240],
            "source":       str(chosen.get("source", ""))[:120],
            "url":          str(chosen.get("link", "")),
            "date":         str(chosen.get("date", "")),
            "body_chars":   len(chosen.get("full_text", "") or ""),
            "style": str(style),
            "selected_at":  datetime.now().isoformat(timespec="seconds"),
        }
        _led.save()
        log.info("[NewsFetcher] stamped meta.news_seed in ledger: %s",
                 (chosen.get("headline") or "")[:60])
    except Exception as _seed_exc:  # noqa: BLE001
        log.warning("[NewsFetcher] news_seed ledger stamp failed (non-fatal): %s",
                    _seed_exc)

    return [chosen]


# -----------------------------------------------------------------------------
# LLM INFERENCE WRAPPER
# -----------------------------------------------------------------------------

def _load_llm(model_id_full="mistralai/Mistral-Nemo-Instruct-2407", device="cuda", optimization_profile="Standard"):
    # Strip [BETA] or [8-bit] labels used in the UI dropdown
    model_id = model_id_full.split(" ")[0]

    # Pre-emptive VRAM sanitation is now handled at the node entry points
    # for better visibility and consistency.

    """Load LLM via transformers. Caches globally with device tracking.

    BEST PRACTICES applied (per survival guide):
      - Section 3:  Lazy loading - never load at import time
      - Section 5:  Device/dtype alignment
      - Section 34: Cache invalidation on device change
      - Section 40: Manual VRAM management since we're outside ComfyUI model registry
      - Section 47: No device_map="auto" (conflicts with ComfyUI's torch.set_default_device)
      - Section 49: No trust_remote_code=True (Gemma is natively supported)
    """
    global _LLM_CACHE

    # Check for device change OR quantization mismatch OR budget profile mismatch
    is_obsidian = "Obsidian" in optimization_profile
    
    # v1.4: Centralized "Large Model" Tags for VRAM Hardening
    # These models MUST be quantized to fit within flagship (16GB) or ultra-lite (4GB) targets.
    vram_safe_tags = ("9b", "12b", "14b", "24b", "26b", "27b", "31b", "70b", "e4b", "4b-it", "a4b", "2b", "2b-it", "efficiency", "nemo", "qwen", "mistral", "instruct", "gemma")
    
    requested_quantized = is_obsidian or "4-bit" in model_id_full.lower() or \
                          any(tag in model_id_full.lower() for tag in vram_safe_tags)

    # v1.4 Audit: Also check if the model object itself has been moved to CPU.
    # 2026-04-26 BUG-LOCAL-065 hardening: bitsandbytes 4-bit quantized models
    # have a non-deterministic parameter iteration order; the FIRST parameter
    # can legitimately report 'cpu' (quantization metadata buffers, embed
    # tokens with offload) even when the model is correctly placed on cuda
    # for inference. Scan up to 8 parameters and accept the cache as valid
    # if ANY of them are on cuda. Only flag eviction when ALL inspected
    # params are off-cuda.
    current_model_device = "cpu"
    any_cuda_param = False
    if _LLM_CACHE["model"] is not None:
        try:
            for i, p in enumerate(_LLM_CACHE["model"].parameters()):
                pd = str(getattr(p, "device", "cpu"))
                if i == 0:
                    current_model_device = pd
                if "cuda" in pd:
                    any_cuda_param = True
                    break
                if i >= 7:
                    break
        except Exception:
            pass

    # 2026-04-29: per-model context cap. Previously a primitive size
    # heuristic ("nemo|12b|14b" -> 6144 else 8192) regardless of the
    # model's actual native context window. That truncated Gemma 4
    # E2B body-pass prompts from 9838 -> 7168 tokens, losing ~25%
    # of the prompt (system spec, AISM filter, cast roster) on every
    # large-target render.
    #
    # New approach: explicit per-model dict. Values are conservative
    # slices of each model's native context, chosen to leave headroom
    # for KV cache + co-resident Bark/FLUX/HuMo VRAM on a 16 GB
    # Blackwell. Native context windows are MUCH larger (Mistral-Nemo
    # 1M, Gemma 4 128K, Qwen 2.5 32K) but burning more is a VRAM
    # tradeoff that needs scientific measurement before being raised.
    # See docs/2026-04-29-vram-context-test.md for the measurement
    # framework that informs future increases.
    # BUG-LOCAL-101 (2026-05-04 LATE EVENING): Mistral-Nemo cap dropped
    # from 16384 to 8192 because the 310-word run at 16384 OOM'd during
    # SDPA prefill of the main script generation. Trace: model loaded
    # at NF4 = 7.74 GiB (BUG-098 tripwire confirmed), then prefill on a
    # ~6-8k token prompt (winning spine + cast roster + news body +
    # format spec) requested a 4.15 GiB attention buffer for one
    # layer's QKV scaled-dot-product, pushing total allocation to 25.41
    # GiB on a 16 GiB device. The prefill attention buffer scales with
    # seq_len^2 * n_heads, and at 16384 cap + long prompt, one layer
    # alone can hit 4+ GiB. Halving the cap to 8192 cuts the per-layer
    # attention budget by 4x in the worst case (the N^2 component) and
    # halves KV cache reservation. v1.7's known-good cap was 6144;
    # 8192 is a middle ground that preserves enough prompt headroom for
    # a 310-word episode with full OpenClose context. If 8192 still
    # OOMs on a longer-target run, drop to 6144 next.
    _MODEL_CONTEXT_CAPS = {
        "mistralai/Mistral-Nemo-Instruct-2407":               8192,
        "google/gemma-4-E2B-it":                             16384,
        "google/gemma-4-E4B-it":                             16384,
        "Qwen/Qwen2.5-14B-Instruct":                          8192,
        "Nitral-AI/Captain-Eris_Violet-V0.420-12B":           8192,
        "inflatebot/MN-12B-Mag-Mell-R1":                      8192,
        "google/gemma-2-2b-it":                               8192,
        "google/gemma-2-9b-it":                               8192,
    }
    _resolved_id = str(model_id_full).split(" ", 1)[0].strip()
    _cap = _MODEL_CONTEXT_CAPS.get(_resolved_id, 8192)

    # 2026-04-26 BUG-LOCAL-065: explicit field-level diagnostics so cache
    # mismatches are debuggable. Previously the reason a reload fired
    # was opaque -- by the time anyone looked at the log the LLM had
    # already cycled. Now every reload prints exactly which fields drifted.
    cache_deltas = []
    if _LLM_CACHE["model"] is not None:
        if str(_LLM_CACHE["device"]) != str(device):
            cache_deltas.append(("device", _LLM_CACHE["device"], device))
        if _LLM_CACHE["quantized"] != requested_quantized:
            cache_deltas.append(("quantized", _LLM_CACHE["quantized"], requested_quantized))
        if _LLM_CACHE["model_id"] != model_id:
            cache_deltas.append(("model_id", _LLM_CACHE["model_id"], model_id))
        if _LLM_CACHE.get("budget_profile") != optimization_profile:
            cache_deltas.append(("budget_profile",
                                 _LLM_CACHE.get("budget_profile"),
                                 optimization_profile))
        if _LLM_CACHE.get("VERSION") != "v1.5":
            cache_deltas.append(("VERSION",
                                 _LLM_CACHE.get("VERSION"), "v1.5"))
        if _LLM_CACHE.get("context_cap") != _cap:
            cache_deltas.append(("context_cap",
                                 _LLM_CACHE.get("context_cap"), _cap))
        # Eviction check: declare evicted only if NO parameters report cuda.
        if ("cuda" in str(device)
                and "cpu" in current_model_device
                and not any_cuda_param):
            cache_deltas.append(("model_evicted_to_cpu",
                                 current_model_device, str(device)))

    if _LLM_CACHE["model"] is not None and cache_deltas:
        delta_summary = ", ".join(
            f"{k}: {cached!r}->{req!r}"
            for k, cached, req in cache_deltas
        )
        _runtime_log(
            f"LLM cache mismatch (Context Cap: {_cap}) "
            f"[fields drifted: {delta_summary}] - reloading to enforce budget"
        )
        _unload_llm()

    if _LLM_CACHE["model"] is None:
        log.info(f"Loading LLM model: {model_id} (quantized={requested_quantized})")
        try:
            # Lazy import - only pay the cost when actually generating
            import torch
            from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer

            # -- Zero-Prime VRAM Hardening (v1.4) --
            # We MUST detect hardware and purge memory BEFORE loading even the Tokenizer
            # to prevent the 15GB transient spike on 16GB cards.

            # Detect Hardware
            total_vram = 0
            if torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            # 2026-04-30: Sync BEFORE eviction.  bnb-NF4 + Blackwell sm_120 +
            # CUDA 13 surfaces cudaErrorUnknown on the post-eviction call when
            # async kernel completions from the prior generation are still in
            # flight while empty_cache() touches their memory.  Reproducer:
            # OpenClose 3-spine loop -- SPINE 1 OK, VRAM_RESET runs, SPINE 2
            # faults.  Triple-confirmed by 2026-04-30 round-robin (ChatGPT +
            # Gemini + NVIDIA).  See docs/2026-04-30-spine-cuda-crash/.
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
            except: pass

            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
            # Post-Wash Analytics
            if torch.cuda.is_available():
                free_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / (1024**3)
                _runtime_log(f"[StoryOrchestrator] Zero-Prime VRAM State: {free_gb:.1f}GB Free. Capacity: {total_vram:.1f}GB")

            # -- VRAM Budgeting (Early Allocation) --
            max_memory = None
            is_actually_2b = any(tag in model_id.lower() for tag in ("2b-it", "2b_it")) or model_id.lower().endswith("2b")
            
            if total_vram >= 12.0:
                # FLAGSHIP 2.5GB SOVEREIGNTY BUFFER (v1.5 CLEAN: expanded for 12B stability)
                budget_gb = total_vram - 2.5
                max_memory = {0: f"{budget_gb:.1f}GiB", "cpu": "32GiB"}
                _runtime_log(f"[StoryOrchestrator] Sovereignty Buffer Active: {budget_gb:.1f}GB Budget")
            elif is_actually_2b:
                max_memory = {0: "3.2GiB", "cpu": "32GiB"}
            elif any(tag in model_id.lower() for tag in ("9b", "12b", "e4b", "4b-it")):
                # Fallback for 8GB-10GB cards ONLY. If we have 16GB, we use the buffer above.
                max_memory = {0: "6.8GiB", "cpu": "32GiB"}

            # Enable TF32 for faster matmuls on Ampere/Ada/Blackwell GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # v1.5 Blackwell Tuning: High precision matmul
            if torch.cuda.get_device_capability()[0] >= 8:
                torch.set_float32_matmul_precision('high')

            # -- VRAM Hardening v1.4: Strict Handoff --
            # If Bark is in VRAM, evict it now before loading LLM.
            try:
                from .bark_tts import _unload_bark
                _unload_bark()
            except ImportError:
                pass
            except Exception as handoff_err:
                log.warning("[StoryOrchestrator] Bark handoff failed: %s", handoff_err)

            is_gemma = "gemma" in model_id.lower()

            # BUG-LOCAL-109 (2026-05-05) defensive guard: the canonical
            # "auto" sentinel ("auto (use story model)") must be resolved
            # to the actual story model_id by the caller BEFORE _load_llm
            # is reached. If something slips a literal "auto*" through
            # (stale workflow JSON, broken caller, missed resolver), fail
            # loudly here with an actionable error -- otherwise HuggingFace
            # tries to resolve a repo named "auto" and emits a 404 buried
            # under five layers of stack trace that nobody reads.
            _mid_lower = (str(model_id) or "").strip().lower()
            if not _mid_lower or _mid_lower.startswith("auto"):
                raise RuntimeError(
                    f"_load_llm: refusing to load model_id={model_id!r} -- "
                    "the 'auto (use story model)' sentinel must be resolved "
                    "by the caller before _load_llm is reached. Pass the "
                    "explicit story model_id (e.g. mistralai/Mistral-Nemo-"
                    "Instruct-2407) instead. See BUG-LOCAL-109."
                )

            try:
                # v1.4 FIX: Revert to AutoTokenizer. AutoProcessor was causing
                # decode offsets to fail on non-multimodal 2B models.
                tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
            except OSError as local_err:
                log.info("[StoryOrchestrator] local_files_only=True failed for tokenizer (%s)", local_err)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                except Exception as hub_err:
                    log.error("[StoryOrchestrator] Hub fallback failed. Ensure model is downloaded or Hub is reachable: %s", hub_err)
                    raise RuntimeError(f"Failed to load Tokenizer '{model_id}'. Is it downloaded? Hub error: {hub_err}") from hub_err

            # Using bfloat16 for maximum speed on RTX 5000-series (Ada/Blackwell) GPUs.
            load_dtype = torch.bfloat16

            # -- Flash Attention 2 (preferred) - SDPA fallback --
            # Flash Attention 2 gives ~40% speedup but requires `pip install flash-attn`.
            # If unavailable, fall back to SDPA which is still fast.
            # Verify the flash-attn DISTRIBUTION is installed (not just an importable
            # stub). Transformers checks PACKAGE_DISTRIBUTION_MAPPING['flash_attn'] and
            # raises KeyError if the distribution metadata is missing, even if `import
            # flash_attn` succeeds. Use importlib.metadata to be authoritative.
            attn_impl = "sdpa"
            try:
                from importlib.metadata import distribution, PackageNotFoundError
                try:
                    distribution("flash-attn")
                    import flash_attn  # noqa: F401
                    attn_impl = "flash_attention_2"
                    log.info("[StoryOrchestrator] Flash Attention 2 available - using flash_attention_2")
                except (PackageNotFoundError, ImportError):
                    log.info(
                        "[StoryOrchestrator] Flash Attention 2: NOT AVAILABLE - no prebuilt wheel exists "
                        "for torch 2.10 + CUDA 13 + Blackwell sm_120 on Windows. "
                        "SageAttention + SDPA active. Performance unaffected. Do not attempt install."
                    )
            except Exception as _fa_err:
                log.info("[StoryOrchestrator] FA2 probe failed (%s) - using SDPA fallback", _fa_err)

            # -- 4-bit quantization (forced for Obsidian or large models) --
            # The Obsidian profile mandates 4-bit to fit on 4GB-8GB GPUs.
            # Large models (26B+) also require 4-bit to fit in 16GB.
            quant_config = None
            needs_8bit = "8-bit" in model_id_full.lower()
            
            # v1.4 Universal Hardening: Centralized tags
            is_unstable_quant = any(tag in model_id_full.lower() for tag in ("2bit", "3bit", "2-bit", "3-bit"))
            needs_4bit = requested_quantized or is_unstable_quant or \
                         any(tag in model_id_full.lower() for tag in vram_safe_tags)

            if is_unstable_quant:
                _runtime_log(f"[StoryOrchestrator] [EMOJI]- WING DING PROTECTION: Unstable Bit-Depth ({model_id_full}) UPGRADED to 4-bit NF4")
            elif needs_4bit:
                _runtime_log(f"[StoryOrchestrator] Quantizing: 4-bit NF4 for {model_id_full}")

            if needs_8bit:
                try:
                    from transformers import BitsAndBytesConfig
                    # llm_int8_enable_fp32_cpu_offload=True required when device_map=auto
                    # may dispatch some layers to CPU (e.g. sovereignty buffer limits VRAM).
                    quant_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_enable_fp32_cpu_offload=True,
                    )
                    log.info("[StoryOrchestrator] Enabling 8-bit quantization")
                except ImportError:
                    log.warning("[StoryOrchestrator] Large model but bitsandbytes not installed!")
            elif needs_4bit:
                try:
                    from transformers import BitsAndBytesConfig
                    # BUG-LOCAL-098 (2026-05-04 EVENING, post round-robin
                    # consult): instantiate BitsAndBytesConfig FRESH per
                    # _load_llm() call. Do NOT cache the instance at module
                    # scope. transformers mutates internal flags on the
                    # config during from_pretrained (Gemini round-robin
                    # catch); a reused instance can silently skip
                    # quantization on the second call -> fp16 fallback ->
                    # OOM at 24 GiB on 16 GiB GPU. Keep this construction
                    # in-function so the contract holds across reloads.
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    log.info("[StoryOrchestrator] [EMOJI] Enabling 4-bit quantization (NF4) for Ultra-Low VRAM")
                    _runtime_log("[StoryOrchestrator] [EMOJI] 4-bit NF4 active")
                except ImportError:
                    log.warning("[StoryOrchestrator] Large model but bitsandbytes not installed - "
                                "loading at bfloat16 may OOM. Run: pip install bitsandbytes")

            from transformers import AutoTokenizer, AutoModelForCausalLM

            # BUG-LOCAL-085 fix (2026-05-04 MORNING): resolve HF_HOME from
            # HKCU\Environment so cache_dir is correct even when ComfyUI
            # Desktop's process didn't inherit User-scope env vars (Electron
            # parent process quirk). Then resolve the model's snapshot dir
            # explicitly and pass it to from_pretrained as a path -- this
            # bypasses transformers' Hub-resolution layer that mis-handles
            # Windows symlinks under local_files_only=True for sharded
            # safetensors models, which silently fell back to fp16 and
            # OOM'd at 24 GiB on the 16 GiB GPU.
            try:
                from . import _otr_hf_env as _OTR_HF
                _hf_home_resolved = _OTR_HF.ensure_hf_home()
                _runtime_log(f"[StoryOrchestrator] HF_HOME resolved -> {_hf_home_resolved}")
            except Exception as _hf_err:
                _runtime_log(f"[StoryOrchestrator] HF_HOME helper unavailable ({_hf_err}); using os.environ fallback")
                _OTR_HF = None
                _hf_home_resolved = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

            cache_dir_path = os.path.join(_hf_home_resolved, "hub")

            # Try to resolve snapshot path (preferred for sharded models on
            # Windows). Falls through to model_id if not cached locally.
            snapshot_path = None
            if _OTR_HF is not None:
                try:
                    snapshot_path = _OTR_HF.resolve_snapshot_dir(model_id, hf_home=_hf_home_resolved)
                except Exception as _snap_err:
                    _runtime_log(f"[StoryOrchestrator] snapshot resolve failed ({_snap_err}); using model_id fallback")
            load_target = snapshot_path or model_id
            if snapshot_path:
                _runtime_log(f"[StoryOrchestrator] Loading from canonical snapshot: {snapshot_path}")
            else:
                _runtime_log(f"[StoryOrchestrator] Snapshot not found in cache; falling back to model_id with cache_dir")

            try:
                # v1.4 Hardening: Explicitly trust_remote_code=False for flagship security
                tokenizer = AutoTokenizer.from_pretrained(
                    load_target,
                    local_files_only=(snapshot_path is None),  # local_files_only only for model_id path
                    trust_remote_code=False,
                    cache_dir=cache_dir_path,
                )
                _runtime_log("LLM tokenizer loaded from cache (no HTTP checks)")
            except Exception as local_err:
                _runtime_log(f"[StoryOrchestrator] tokenizer load failed ({local_err}), attempting Hub fallback...")
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False, cache_dir=cache_dir_path)

            common_kwargs = dict(
                cache_dir=cache_dir_path,
                trust_remote_code=False,  # v1.4 Hardening: strict security
                low_cpu_mem_usage=True,
                torch_dtype=load_dtype,   # v1.5 FIX: halved initial footprint
                attn_implementation="sdpa"  # v1.4 Blackwell/Flash-Replacement
            )

            # v1.5 FIX: Explicitly pass max_memory (Sovereignty Buffer) to from_pretrained.
            # Previously it was calculated but ignored, causing 23GB spikes.
            if max_memory is not None:
                common_kwargs["max_memory"] = max_memory
                common_kwargs["device_map"] = "auto"

            if quant_config is not None:
                common_kwargs["quantization_config"] = quant_config
                
                # FLAGSHIP 16GB OVERRIDE: 2B and 12B models fit easily if context is capped.
                # Force GPU-only to avoid sneaky CPU offloading by 'auto' device_map.
                if total_vram >= 15.0:
                    common_kwargs["device_map"] = {"": 0} 
                    _runtime_log(f"[StoryOrchestrator] Flagship Sovereignty: Forcing 100% GPU for {model_id}")
                # else: device_map already set to "auto" in max_memory block above

            try:
                # v1.5 CLEAN Hardware Hardening: Constrain 128k models to reduce
                # KV cache memory spikes on 16GB hardware.
                model_config = None
                try:
                    from transformers import AutoConfig
                    _cfg_kwargs = {"trust_remote_code": False, "cache_dir": cache_dir_path}
                    # BUG-LOCAL-085: prefer snapshot path for AutoConfig too
                    model_config = AutoConfig.from_pretrained(load_target, **_cfg_kwargs)
                    # v1.5 CLEAN: Constrain 128k models to 2k window during spine phase
                    if hasattr(model_config, "max_position_embeddings") and model_config.max_position_embeddings > _cap:
                        _runtime_log(f"[StoryOrchestrator] Hardening: Capping 128k context to {_cap} (Saves ~6GB VRAM)")
                        model_config.max_position_embeddings = _cap
                except Exception as _cfg_err:
                    log.warning("[StoryOrchestrator] Config hardening failed: %s", _cfg_err)

                # BUG-LOCAL-098 (2026-05-04 EVENING): measure VRAM before
                # the load so the post-load tripwire below has a true
                # delta to compare against the NF4 expected ceiling.
                _bug098_vram_before_gib = (
                    torch.cuda.memory_allocated() / (1024 ** 3)
                    if torch.cuda.is_available() else 0.0
                )

                # BUG-LOCAL-085: load from snapshot path when available
                # (bypasses transformers' Hub-resolution layer); fall back
                # to model_id only if snapshot resolution failed.
                model = AutoModelForCausalLM.from_pretrained(
                    load_target,
                    local_files_only=(snapshot_path is None),
                    config=model_config,
                    **common_kwargs,
                )
                _runtime_log(
                    f"LLM model loaded from "
                    f"{'canonical snapshot' if snapshot_path else 'model_id with cache_dir'} "
                    f"(no HTTP checks)"
                )

                # BUG-LOCAL-098 (2026-05-04 EVENING, post round-robin):
                # Tripwire. If NF4 quantization was requested but did not
                # actually materialize (bitsandbytes module-level state
                # short-circuit on second-load after _unload_llm), fail
                # loud BEFORE the first inference cascades into a 24+ GiB
                # OOM. Both LLM consultants (gpt-5.5 + gemini-3.1-pro) agreed
                # this assertion is mandatory regardless of whether the
                # underlying reload bug is fixed yet.
                #
                # Detection (3 signals):
                #   1. Count of bitsandbytes Linear4bit modules. NF4 model
                #      has ~280-380 of these for a 12B model; fp16 model
                #      has zero.
                #   2. ``model.is_loaded_in_4bit`` attribute (HF stamps
                #      this on successful 4-bit load; absent or False on
                #      silent fp16 fallback).
                #   3. CUDA allocation delta. Mistral-Nemo NF4 should be
                #      ~7-8 GiB; fp16 is ~24 GiB. Threshold of 11.0 GiB
                #      catches the fp16 case with ~3 GiB headroom.
                #
                # Bypass: if quant_config is None we requested fp16
                # explicitly (e.g., a 2B model that fits without
                # quantization). The tripwire only fires when the user
                # asked for quantization and it silently dropped.
                if quant_config is not None and torch.cuda.is_available():
                    _bug098_vram_after_gib = torch.cuda.memory_allocated() / (1024 ** 3)
                    _bug098_delta_gib = (
                        _bug098_vram_after_gib - _bug098_vram_before_gib
                    )
                    _bug098_linear4bit_count = 0
                    try:
                        for _m in model.modules():
                            _cls_name = type(_m).__name__
                            _mod_name = type(_m).__module__ or ""
                            if (_cls_name == "Linear4bit"
                                    and _mod_name.startswith("bitsandbytes")):
                                _bug098_linear4bit_count += 1
                    except Exception:  # noqa: BLE001 -- diagnostic-only walk
                        _bug098_linear4bit_count = -1
                    _bug098_is_loaded_in_4bit = bool(
                        getattr(model, "is_loaded_in_4bit", False)
                    )
                    _bug098_max_gib = 11.0  # 12B NF4 ceiling with slack
                    _bug098_module_signal = (
                        _bug098_linear4bit_count > 0
                        or _bug098_is_loaded_in_4bit
                    )
                    _bug098_vram_signal = (
                        _bug098_delta_gib >= 0.0
                        and _bug098_delta_gib <= _bug098_max_gib
                    )
                    _runtime_log(
                        f"[BUG-098 tripwire] post-load: "
                        f"linear4bit_count={_bug098_linear4bit_count} "
                        f"is_loaded_in_4bit={_bug098_is_loaded_in_4bit} "
                        f"vram_delta={_bug098_delta_gib:.2f}GiB "
                        f"(ceiling={_bug098_max_gib:.2f}GiB)"
                    )
                    if not _bug098_module_signal or not _bug098_vram_signal:
                        # Free the broken model before raising so the
                        # exception path doesn't leave 24 GiB stuck on
                        # the GPU.
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
                            f"materialize for {model_id!r}. "
                            f"linear4bit_count={_bug098_linear4bit_count} "
                            f"(expected >0), "
                            f"is_loaded_in_4bit={_bug098_is_loaded_in_4bit} "
                            f"(expected True), "
                            f"vram_delta={_bug098_delta_gib:.2f}GiB "
                            f"(expected <={_bug098_max_gib:.2f}GiB). "
                            f"This is the bitsandbytes second-load silent "
                            f"fp16 fallback. Workaround: restart ComfyUI "
                            f"Desktop and re-queue. The first load per "
                            f"process is reliable; subsequent reloads "
                            f"after _unload_llm are not. Tracked as "
                            f"BUG-LOCAL-098; proper fix pending an "
                            f"isolated-test-harness validation of the "
                            f"`.cuda()` rehydrate path or LLM-subprocess "
                            f"isolation."
                        )
            except (OSError, ValueError) as local_err:
                _runtime_log(f"[StoryOrchestrator] local_files_only=True failed for model ({local_err}), attempting Hub fallback...")
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        config=model_config,
                        **common_kwargs,
                    )
                except Exception as hub_err:
                    log.error("[StoryOrchestrator] Hub fallback failed. Ensure model is downloaded or Hub is reachable: %s", hub_err)
                    raise RuntimeError(f"Failed to load LLM model '{model_id}'. Is it downloaded? Hub error: {hub_err}") from hub_err

            if quant_config is None and max_memory is None:
                model = model.to(device)
            model = model.eval()

            _LLM_CACHE["model"] = model
            _LLM_CACHE["tokenizer"] = tokenizer
            _LLM_CACHE["device"] = device
            _LLM_CACHE["quantized"] = (quant_config is not None)
            _LLM_CACHE["model_id"] = model_id
            _LLM_CACHE["budget_profile"] = optimization_profile
            _LLM_CACHE["VERSION"] = "v1.5"
            _LLM_CACHE["context_cap"] = _cap # v1.5 CLEAN hardening state
            actual_quant = (quant_config is not None)
            _runtime_log(f"LLM loaded: {model_id} (quantized={actual_quant}, budget={optimization_profile}) [v1.5]")

            # -- v1.5.1: CUDA KERNEL WARMUP ------------------------------
            # The first model.generate() call on Blackwell (sm_120) with
            # SDPA + BitsAndBytes 4-bit triggers JIT compilation of CUDA
            # kernels - causing a 30-60s stall before the first token.
            # This 1-token warmup absorbs that cost here so the real
            # generation starts immediately.
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

        except Exception as e:
            log.exception("Failed to load LLM: %s", e)  # Section 49: log.exception for full traceback
            raise
    return _LLM_CACHE["model"], _LLM_CACHE["tokenizer"]


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

def _is_inline_narration(speaker_name, text):
    """BUG-LOCAL-100: detect inline stage direction in [VOICE:] line.

    Some LLMs (Captain-Eris, Mistral Nemo at higher temperatures)
    emit a stage direction on the same line as the VOICE tag, with
    the actual dialogue on the next line::

        [VOICE: LEV SHAW, traits] Lev bursts onto deck, hurrying...
        Got the newest readings yet?

    Without this check the parser captures "Lev bursts onto deck..."
    as the dialogue and Bark TTS reads the stage direction aloud
    (Stellar Shadows ledger 2026-04-28: l002 + l003 are pure
    narration that got vocalized).

    Heuristic: the captured text starts with the speaker's first
    word in third-person form (e.g. "Lev " for LEV SHAW, "Stanley "
    for STANLEY CRANSTON). Returns True when the caller should look
    ahead to the next line for the actual dialogue.

    False positives are unlikely: real dialogue almost never starts
    with the speaker's own name in third person (a character does
    not say "Lev bursts..." about themselves). False negatives fall
    through to current parser behavior unchanged.
    """
    if not text or not speaker_name:
        return False
    # Strip leading ASCII straight quote, smart quotes, asterisks,
    # underscores, and whitespace before checking for the speaker's first
    # name. Smart quotes assembled via chr() to keep this file ASCII-clean
    # (CLAUDE.md: UTF-8 no BOM, ASCII source where possible).
    _LDQ = chr(0x201C)  # left double smart quote
    _RDQ = chr(0x201D)  # right double smart quote
    _LEADING_NOISE = "[*_\"" + _LDQ + _RDQ + "\\s]+"
    _TRAIL_PUNCT = ".,!?;:'\"" + _LDQ + _RDQ
    cleaned = re.sub("^" + _LEADING_NOISE, "", text).strip()
    if not cleaned:
        return False
    first_word = cleaned.split()[0].rstrip(_TRAIL_PUNCT).lower()
    speaker_words = speaker_name.strip().split()
    if not speaker_words:
        return False
    return first_word == speaker_words[0].lower()


_DIALOGUE_FALSE_POSITIVES = frozenset({
    "SCENE", "ACT", "NOTE", "TARGET", "STYLE", "SFX",
    "ENV", "NARRATOR", "OPENING", "CLOSING", "MUSIC",
    "ANNOUNCER",
    # BUG-LOCAL-037: BUG-LOCAL-035's TITLE: <name> first-line convention
    # was being parsed as a "TITLE" character speaking the title text.
    # Block it everywhere a NAME: <text> shape gets read as dialogue.
    "TITLE",
})

# BUG-LOCAL-035: Titles that indicate a stuck default or an unresolved title.
# When a title resolution path yields one of these, we treat it as a failure
# and either re-derive or fail loud. Keep lowercase, no punctuation.
_STUCK_TITLE_DEFAULTS = frozenset({
    "",
    "the last frequency",
    "untitled",
    "episode",
    "signal lost",
    "custom episode",
})

# Matches a leading "TITLE:" line (case-insensitive) the LLM may emit when
# asked to generate a title. We accept up to the end of the first line.
_RE_TITLE_LINE = re.compile(
    r'^\s*(?:\*\*)?\s*TITLE\s*:\s*["\u201C]?\s*(.+?)\s*["\u201D]?\s*(?:\*\*)?\s*$',
    re.IGNORECASE | re.MULTILINE,
)


def _extract_title_from_script_text(text):
    """Pull a 'TITLE: ...' line out of raw LLM script text.

    Gemma is instructed to emit such a line when episode_title is blank.
    Returns the extracted title string (stripped, quotes removed) or "".
    """
    if not text:
        return ""
    m = _RE_TITLE_LINE.search(text[:2000])  # only look near the top
    if not m:
        return ""
    cand = m.group(1).strip().strip('"\u201C\u201D\u2018\u2019')
    # BUG-LOCAL-039: strip markdown bold/italic wrappers on the value itself
    # (e.g. "TITLE: **Bioluminal Tide**" leaks leading "**" into the capture).
    cand = re.sub(r'^(?:\*{1,3}|_{1,3})\s*', '', cand)
    cand = re.sub(r'\s*(?:\*{1,3}|_{1,3})$', '', cand)
    cand = cand.strip().strip('"\u201C\u201D\u2018\u2019')
    # Reject obviously broken captures (too long, template residue).
    if not cand:
        return ""
    if len(cand) > 120:
        return ""
    if cand.lower() in _STUCK_TITLE_DEFAULTS:
        return ""
    return cand


def _derive_title_from_script_lines(lines, style=""):
    """Deterministic fallback title when the LLM didn't emit one.

    Strategy: take the first 'environment' token's description, pick the
    most-specific noun phrase from its first 6 words, title-case it.
    This is last-resort; it keeps the filename layer unblocked when the
    LLM path silently failed, without silently reusing a stuck default.
    """
    try:
        for ln in lines or []:
            if not isinstance(ln, dict):
                continue
            if ln.get("type") == "environment":
                desc = (ln.get("description") or "").strip()
                if not desc:
                    continue
                # Take first 6 tokens, drop stopwords/punctuation.
                tokens = re.findall(r"[A-Za-z][A-Za-z0-9'\-]+", desc)[:6]
                stop = {"a", "an", "the", "of", "on", "in", "at", "and",
                        "or", "is", "with", "for", "to", "from", "by"}
                kept = [t for t in tokens if t.lower() not in stop]
                if not kept:
                    continue
                phrase = " ".join(kept[:4]).title()
                # Filter if phrase collapses to a stuck default.
                if phrase.lower() in _STUCK_TITLE_DEFAULTS:
                    continue
                return phrase
    except Exception:
        pass
    # Final fallback: timestamped derivative so each run is unique and
    # the filename layer never regresses to a stuck default.
    _style_label = (style or "transmission").replace("_", " ").title()
    return f"{_style_label} Transmission {int(time.time()) % 100000}"


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


# Bounded model cache with device tracking (Section 34)
_LLM_CACHE = {"model": None, "tokenizer": None, "device": None, "quantized": False, "model_id": None, "budget_profile": None, "VERSION": "v1.5"}


def _unload_llm():
    """Explicitly unload LLM to free VRAM (v1.3.1 OOM FIX).

    The v1.3 version did del + gc.collect() + empty_cache(), but that
    is a no-op when abandoned worker threads from _run_with_timeout
    still hold the model as a local variable in their stack frame.
    Symptom: second load attempt sees VRAM at 31.70 GiB on a 16 GB
    card because the first model never actually left the GPU.

    The fix is to call model.cpu() BEFORE dropping references. That
    moves the weights from VRAM to RAM immediately, even when other
    strong refs exist. Abandoned generate() threads will then error
    out on device mismatch, which is acceptable because their results
    are already being discarded by the timeout fallback path.

    Order of operations is load-bearing:
      1. model.cpu()         - move weights off GPU even with live refs
      2. del cache entries   - drop the primary reference
      3. gc.collect()        - destroy the object if no other refs
      4. empty_cache()       - return freed VRAM to the allocator
      5. telemetry           - prove VRAM actually dropped
    """
    global _LLM_CACHE
    import gc
    import torch
    if _LLM_CACHE["model"] is not None:
        # 2026-04-26 PM BUG-LOCAL-073 GUARD: synchronize BEFORE cpu().
        # If a prior CUDA kernel left dirty memory (illegal access pending),
        # cuda.synchronize() will surface it as a clean Python exception
        # rather than letting model.cpu() walk dirty memory and zombify
        # the process. On synchronize failure we skip the cpu() walk and
        # go straight to dropping references + empty_cache(), which is
        # safe because torch.cuda.empty_cache() resets the allocator
        # state without touching individual tensors.
        sync_ok = True
        try:
            torch.cuda.synchronize()
        except Exception as sync_err:
            sync_ok = False
            log.warning("[StoryOrchestrator] cuda.synchronize() before unload failed: %s -- skipping model.cpu() walk", sync_err)
            _runtime_log(f"VRAM_UNLOAD_GUARD: synchronize failed ({sync_err}); cpu() bypassed")
        # Step 1: force weights off GPU before dropping the reference.
        # Only attempt if synchronize succeeded -- otherwise dirty memory
        # would propagate the fault inside .cpu() and lock the process.
        if sync_ok:
            try:
                _LLM_CACHE["model"].cpu()
            except Exception as cpu_err:
                log.warning("[StoryOrchestrator] model.cpu() during unload failed: %s", cpu_err)
                _runtime_log(f"VRAM_UNLOAD_GUARD: cpu() failed ({cpu_err}); proceeding with empty_cache only")
        # Step 2: drop references from the module-level cache.
        del _LLM_CACHE["model"]
        del _LLM_CACHE["tokenizer"]
        _LLM_CACHE = {"model": None, "tokenizer": None, "device": None, "quantized": False, "model_id": None}
        # Step 3 + 4: gc and return VRAM to the allocator. These ALWAYS
        # run, even after a sync/cpu failure, so the allocator's tracking
        # state is reset and the next phase starts with a clean budget.
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception as ec_err:
            log.warning("[StoryOrchestrator] empty_cache() failed: %s", ec_err)
        
        # Evict from ComfyUI's internal cache tracking as well
        try:
            import comfy.model_management
            comfy.model_management.soft_empty_cache()
        except Exception:
            pass

        # BUG-LOCAL-098 (2026-05-04 EVENING, Gemini round-robin
        # suggestion): clear accelerate's device-dispatch cache so the
        # next _load_llm gets a fresh hook table. Accelerate has been
        # observed to retain device maps / hooks across model lifetimes
        # which can confuse bitsandbytes' second-load quantization
        # path. Defensive; no-op if accelerate isn't installed or the
        # API name has shifted.
        try:
            from accelerate import clear_device_cache as _bug098_clear_dev
            _bug098_clear_dev()
        except (ImportError, AttributeError):
            pass
        except Exception:  # noqa: BLE001 -- never let cleanup raise
            pass

        # Step 5: telemetry - prove it worked.
        allocated_gib = torch.cuda.memory_allocated() / 1e9
        reserved_gib = torch.cuda.memory_reserved() / 1e9
        log.info(
            "LLM unloaded: VRAM allocated=%.2f GiB reserved=%.2f GiB "
            "(cpu + gc.collect + empty_cache)",
            allocated_gib, reserved_gib,
        )

# Register the LLM unloader with the VRAM Power Wash system so that
# force_vram_offload() at node entry points also evicts Gemma.
from ._vram_log import register_vram_cleanup
register_vram_cleanup(_unload_llm)


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


_LTX_STYLE_BRIEF_PROMPT = """You are writing a single-sentence VISUAL STYLE BRIEF for the broadcast equipment shown on screen during an audio drama. Describe ONLY the equipment / room aesthetic appropriate to this story's setting and style. NO people, NO characters, NO action -- just the look of the broadcasting equipment and the room it sits in.

Story style: {style}
Story snippet: {story_snippet}

Output ONE sentence (20-40 words) describing the broadcast equipment and its room. The sentence should:
- Match the story's setting (extract from the snippet: lunar base, deep-space vessel, seabase, mars colony, orbital station, near-future newsroom, industrial-decay site, whatever fits)
- Use equipment design language that fits the setting AND style -- do not default to any specific era's hardware unless the story explicitly implies it
- Include lighting and atmosphere cues that fit the style
- NOT mention people, hands, faces, voices, or anyone speaking
- Be ONE sentence with no preamble

Examples (one near-future newsroom, one deep-space vessel, one industrial decay -- spanning the style range so no single hardware era dominates):
- Near-future newsroom broadcast desk, edge-lit glass console with floating waveform overlays, cool overhead daylight, condensation rings on a steel coffee cup, hum of HVAC.
- Deep-space science vessel comms console, holographic dial readouts, recycled-atmosphere haze, speaker grille mounted into a curved bulkhead, magnetic dust drifting through volumetric beams.
- Rust-belt repurposed factory broadcast loft, scavenged industrial speaker bolted to a corroded I-beam, sodium-vapor work lamps, oil-stained concrete floor, occasional sparks from exposed wiring.

Visual brief:"""


def _generate_ltx_style_brief(style, story_snippet, model_id, optimization_profile):
    """Generate a per-episode LTX style brief that flavors the radio
    broadcast set to match the story's sci-fi setting.

    Returns a cleaned single-sentence string (max ~300 chars) suitable
    for prepending to ``_PROMPT_BY_ROLE`` templates in ``batch_ltx_render``.
    Returns empty string on any failure -- caller treats empty as
    "fall back to the role template alone".

    Cost: one short LLM call (~80 tokens output) on the already-loaded
    Mistral-Nemo. Adds ~5-10s to the LLM phase per episode. Non-fatal
    on any error.

    BUG-LOCAL-008 alignment: prompt explicitly forbids people / hands /
    faces in the brief, since the brief gets passed to LTX which (at
    CFG=1.0) only honors positive prompt content.
    """
    prompt = _LTX_STYLE_BRIEF_PROMPT.format(
        style=(style or "sci-fi").strip()[:80],
        story_snippet=(story_snippet or "").strip()[:500],
    )
    try:
        raw = _generate_with_llm(
            prompt,
            model_id=model_id,
            max_new_tokens=80,
            temperature=0.7,
            top_p=0.9,
            optimization_profile=optimization_profile,
            live_ledger=False,
        )
    except Exception as exc:  # noqa: BLE001 -- non-fatal; caller falls back
        log.warning("[LTXStyleBrief] generation failed: %s", exc)
        return ""

    if not raw:
        return ""

    # Cleanup: drop any leading "Visual brief:" / quotes / blank lines,
    # take the first line, hard-cap length.
    text = raw.strip()
    for prefix in ("Visual brief:", "VISUAL BRIEF:", "Brief:", "BRIEF:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    text = text.strip().split("\n")[0].strip()
    text = text.strip('"').strip("'").strip()
    # Strip trailing example-list bullets if model echoed them
    if text.startswith("- "):
        text = text[2:].strip()
    # Hard cap so it can't overwhelm the role-template prompt
    if len(text) > 300:
        text = text[:300].rsplit(".", 1)[0].strip()
        if text and not text.endswith("."):
            text = text + "."
    return text


def _generate_with_llm(prompt, model_id="mistralai/Mistral-Nemo-Instruct-2407",
                          max_new_tokens=4096, temperature=0.8, top_p=0.92,
                          optimization_profile="Standard",
                          live_ledger=False):
    """Generate text with LLM.

    live_ledger=True turns on the L1.5 streaming-ledger hook on the
    underlying GemmaHeartbeatStreamer. Pass it ONLY at the body call site
    (and any other site whose dialogue you want surfaced live). Spine /
    critique / revision / grammarian calls leave it False so they don't
    repeatedly overwrite the singleton ledger with their own intermediate
    drafts.
    """
    import torch

    model, tokenizer = _load_llm(model_id, optimization_profile=optimization_profile)
    is_small_model = any(tag in model_id.lower() for tag in ("2b-it", "2b_it", "small")) or (model_id.lower().endswith("2b"))

    # Multimodal vs Text-Only wrapper
    is_gemma = "gemma" in model_id.lower()
    
    # BUG-011 FIX: Verify tokenizer supports the multimodal list-of-dicts format.
    supports_multimodal = hasattr(tokenizer, "tokenizer") or hasattr(tokenizer, "image_processor")
    if is_gemma and supports_multimodal:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    else:
        messages = [{"role": "user", "content": prompt}]
        
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    if is_gemma and supports_multimodal:
        inputs = tokenizer(text=text, return_tensors="pt").to(model.device)
    else:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # -- v1.5.1: PROMPT LENGTH GUARD -------------------------------------
    # NeMo 12B has a 128k native context, but we cap max_position_embeddings
    # to 2048 at load time to limit KV cache VRAM. However, transformers does
    # NOT enforce this at the input level - it still accepts a 3000-token prompt
    # and pre-fills the full KV cache, causing the 110s stall and 25GB VRAM spike.
    # We must truncate the input explicitly to leave room for output tokens.
    _context_cap = _LLM_CACHE.get("context_cap", 8192)
    _max_input_tokens = max(64, _context_cap - max_new_tokens)
    _input_len = inputs["input_ids"].shape[-1]
    if _input_len > _max_input_tokens:
        _trunc = _input_len - _max_input_tokens
        inputs["input_ids"] = inputs["input_ids"][:, _trunc:]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, _trunc:]
        _runtime_log(
            f"PROMPT_GUARD: Truncated {_input_len} -> {_max_input_tokens} tokens "
            f"(context_cap={_context_cap}, max_new_tokens={max_new_tokens})"
        )
        log.info(
            "[StoryOrchestrator] Prompt truncated: %d -> %d tokens to fit context cap %d",
            _input_len, _max_input_tokens, _context_cap,
        )


    if "attention_mask" not in inputs and "input_ids" in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    # LLM eos_token_id is a list - extract first element for pad_token_id
    eos_id = model.generation_config.eos_token_id
    pad_id = eos_id[0] if isinstance(eos_id, list) else eos_id

    log.info(f"[StoryOrchestrator] Starting inference (max_new_tokens={max_new_tokens})...")
    log.info("[StoryOrchestrator] Live output will stream below:")
    start_inference = time.time()

    # Initialize streamer for live feedback in the terminal + heartbeat logs.
    # Safely access tokenizer if we're using a multimodal processor.
    raw_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    streamer = GemmaHeartbeatStreamer(
        raw_tokenizer, skip_prompt=True, skip_special_tokens=True,
        live_ledger=live_ledger,
    )

    vram_snapshot("llm_generate_entry")

    try:
        with torch.no_grad():
            # v1.4: Tune penalty for 2B models to prevent SFX loops
            final_penalty = 1.12
            if "2b" in model_id.lower():
                final_penalty = 1.25  # Firmer hand for the small model
                
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                max_length=None,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=final_penalty,
                pad_token_id=pad_id,
                streamer=streamer,      # Enable live streaming + granular heartbeats
            )

        inference_time = time.time() - start_inference
        log.info(f"[StoryOrchestrator] Inference complete in {inference_time:.1f}s.")

        # Decode only the new tokens (skip the prompt).
        new_tokens_cpu = output[0][inputs["input_ids"].shape[-1]:].detach().cpu()
        decoded = tokenizer.decode(new_tokens_cpu, skip_special_tokens=True)

        # Log output token count and snapshot VRAM state after generation
        output_token_count = new_tokens_cpu.shape[0]
        _runtime_log(f"llm_generate_exit: generated {output_token_count} tokens in {inference_time:.1f}s")
        vram_snapshot("llm_generate_exit")

        return decoded
    finally:
        # v1.4 Theme B/C: GUARANTEED VRAM RECOVERY
        # Whether generation completes normally, OOMs, or is aborted by the
        # streamer's TimeoutError, we MUST clear these tensors so the thread
        # local variables don't hold the graph and the KV cache captive.
        if 'new_tokens_cpu' in locals():
            del new_tokens_cpu
        if 'output' in locals():
            del output
        del inputs, streamer
        torch.cuda.empty_cache()


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


# This lets the Director automatically inherit the exact same model memory
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
    an XML-flavored block prepended to SCAFFOLDING_PREAMBLE. Returns
    the empty string on:
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
# PATTERN 2 - SCAFFOLDING & PARSING MATRIX (v1.2 narrative)
# XML-wrapped dramaturg role preamble. Prepended to SCRIPT_SYSTEM_PROMPT at
# format() time. Contains no {fields} so .format() passes it through untouched.
# ============================================================================
SCAFFOLDING_PREAMBLE = """<system_role>
You are a MASTER DRAMATURG for the audio drama anthology "SIGNAL LOST". Not a
novelist. Not a writer. A DRAMATURG. Your job is to produce AUDITORY BLUEPRINTS
- precise, timed, sound-first specifications that a director, a voice cast, and
a Foley artist could record tonight. You think like the golden age of radio
drama: Orson Welles, Norman Corwin, Lucille Fletcher. The page is NEVER prose.
The page is a recording score.
</system_role>

<brick_method>
WORKING PROCESS - THE BRICK METHOD (1:5 OUTLINE-TO-SCRIPT RATIO):
Before writing a single scene, compose a compact internal outline: one tight
paragraph per scene, approximately one-fifth the length of the final script,
capturing the inciting beat, the escalation, the turn, and the exit hook. Then
expand that outline into the full script at roughly 5x its length. The outline
is your structural spine; the expansion is where sound design and burstiness
live. Do NOT show the outline in the final output - use it to think, then
expand.
</brick_method>

<acoustic_spaces>
ACOUSTIC SPACE DECLARATION - Before writing Scene 1, mentally classify every
location the episode will use with one of these canonical acoustic profiles.
Use the profile word inside your [ENV:] tags verbatim so the SceneSequencer
room-tone synthesizer can match on the keyword:
- CAVERNOUS - large sealed volumes with long reflections. Keywords: cavernous,
  echo, vault, cathedral, tunnel.
- FLUORESCENT - small indoor spaces with electrical hum. Keywords: fluorescent,
  hum, corridor, office, lab.
- TILED - hard reflective surfaces. Keywords: tiled, reverberant, clinical,
  bathroom, morgue.
- STORM - open exterior with wind and distant pressure. Keywords: storm, wind,
  open, gale, rain.
- INTIMATE - close-mic dead space. Keywords: quiet, close, dead, padded, booth.
Pick the profile that matches each location BEFORE you write its [ENV:] tag,
then pack the tag with 2-3 specific sensory details layered on top of the
profile keyword. The downstream room-tone synthesizer reads the keyword and
selects its bed accordingly.
</acoustic_spaces>

<epilogue_constraint>
The closing Hard-Science Epilogue is anchored to the real news seed provided
below. It cites the real article directly. It is 2-3 sentences maximum. No
speculation beyond the article. No fabricated institutions. No invented journal
names. The drama's resolution must land on a concrete finding from the seed.
</epilogue_constraint>

"""


SCRIPT_SYSTEM_PROMPT = """REQ-0 NEVER OUTPUT PROSE PARAGRAPHS. Every line of your output must start
      with one of: === SCENE, [ENV:, [SFX:, [VOICE:, [CHARACTERNAME,, or (beat).
      If you catch yourself writing a narrative prose paragraph, STOP and
      restart that line with [VOICE: NAME, traits] instead.

# CANONICAL AUDIO ENGINE v1.0 - DETERMINISTIC TOKENS ONLY.
# Every line must be an "Audio Token": [ENV:], [SFX:], [VOICE:], or (beat).

=== HARD REQUIREMENTS (VALIDATED AUTOMATICALLY) ===
The parser counts character dialogue lines. A script failing these checks is REJECTED:

REQ-1 CHARACTER DIALOGUE IS MANDATORY. The episode MUST contain real characters
      speaking to each other - not just an ANNOUNCER/narrator bookending a montage
      of [ENV:] and [SFX:] cues. A "narration-only" script with no character
      conversation is INVALID. You will see a line-count minimum stated in the
      user prompt; hit it before stopping.

REQ-2 DIALOGUE FORMAT TOLERANCE. The parser accepts BOTH of these forms
      identically, so use whichever feels natural - just be consistent within
      one script:
          [VOICE: NAME, gender, age, tone, energy] Short spoken line.
          [NAME, gender, age, mood] Short spoken line.
      In BOTH forms the NAME is ALWAYS FIRST and ALL-CAPS. NAME must be a real
      character name, not a descriptor word like MAN / WOMAN / MALE / FEMALE.

REQ-3 NEVER OMIT THE NAME. Lines like "[male, 40s, calm] text" or
      "[Female, urgent] text" (where the first field is a gender word, not a
      name) are REJECTED. Invent a character name first, then put traits after
      the comma.

=== [EMOJI] 1. CANONICAL FORMATTING (STRICT) ===
Every scene MUST follow this layout:

=== SCENE X ===

[ENV: description (3-4 descriptors: e.g. fluorescent hum, distant traffic)]

[SFX: description]

[VOICE: CHARACTERNAME, gender, age, tone, energy] Short, natural dialogue line.

(beat)

[VOICE: CHARACTERNAME, gender, age, tone, energy] Next dialogue line.

CRITICAL: The first field in EVERY [VOICE:] tag is ALWAYS the CHARACTER NAME IN ALL CAPS.
WRONG: [VOICE: male, 40s, calm] Text here.
RIGHT: [VOICE: CHARACTERNAME, gender, age, tone, energy] Dialogue goes here.
CHARACTER NAMES must be CONSISTENT across all scenes (same spelling, same caps, every time).
Invent fresh, original names for every episode. Do NOT reuse names from previous episodes.

=== [EMOJI] 2. THE TAG SYSTEM (ONLY THESE FOUR) ===
- [ENV: ...] -> Background layers (e.g. [ENV: cockpit, electronic chirps, life support hum])
- [SFX: ...] -> Individual sound effects (e.g. [SFX: metal clatter])
- [VOICE: NAME, gender, age, tone, energy] -> MUST precede every dialogue line.
  NAME is ALWAYS FIRST - all caps, no spaces if possible.
  The NAME must be a short, punchy, original character name you invent: 1-2 syllables, strong consonants, easy to say aloud.
  The ANNOUNCER role always uses: [VOICE: ANNOUNCER, gender, age, tone, energy]
- (beat) -> A 0.8s deterministic pause. Use it between lines for timing.

=== [EMOJI] 3. DIALOGUE RULES (BARK OPTIMIZED) ===
- Keep dialogue lines SHORT (5-15 words).
- ONE sentence per line. Never use long paragraphs.
- Use natural, fragmented phrasing. Interruptions allowed.
- Use ... for hesitations and trailing thoughts.
- Use CAPS for single-word emphasis: "We are COMPLETELY out of time."
- Bark non-verbal tokens go INSIDE dialogue (in square brackets):
    [laughs]        - brief laugh mid-sentence
    [laughter]      - sustained laughter
    [sighs]         - audible sigh
    [gasps]         - sharp intake of breath
    [coughs]        - coughing
    [clears throat] - throat clearing before speaking
    [pants]         - breathless, exertion
    [sobs]          - crying
    [grunts]        - effort/strain
    [groans]        - pain or frustration
    [whistles]      - whistle
    [sneezes]       - sneeze
- Use - around text for sung or hummed lines: - signal lost, signal lost -
- NEVER use (parentheses) for anything except the (beat) tag.
- NEVER write stage directions in the dialogue text.

=== [EMOJI] 3a. THE [VOICE:] LINE CONTRACT (BUG-LOCAL-100, NON-NEGOTIABLE) ===
The text that follows the closing ] of a [VOICE:] tag is read aloud
VERBATIM by a TTS engine. It is NOT a description of what the character
does -- it is the WORDS the character SAYS. Treat the [VOICE:] line as
a one-way speaker, not as a screenplay action line.

WRONG (this gets read aloud as the character's own voice):
  [VOICE: LEV SHAW, male, 50s, urgent] Lev bursts onto the catwalk deck.
  Got the newest readings yet?

RIGHT (the action goes in [SFX:] / [ENV:], the dialogue is verbatim speech):
  [SFX: hurried boot strikes on catwalk grating, airlock seal hiss]
  [VOICE: LEV SHAW, male, 50s, urgent] Got the newest readings yet?

WRONG (third-person prose between tag and quote):
  [VOICE: STANLEY CRANSTON, male, 60s, calm] Stanley follows closely.
  Not encouraging. Shadow overlap keeps pushing our window.

RIGHT (the [VOICE:] line is ONLY what the speaker says):
  [SFX: trailing footfalls, distant wind across the deck]
  [VOICE: STANLEY CRANSTON, male, 60s, calm] Not encouraging. Shadow overlap keeps pushing our window.

If a character performs an action without speaking, do NOT create a
[VOICE:] line for them. Use [SFX:] or [ENV:] for the action and let
the next speaker's [VOICE:] tag carry the scene forward. Stage
directions, narration, and third-person prose NEVER appear in a
[VOICE:] line, with or without surrounding quotes.

=== [EMOJI] WORLDBUILDING, RHYTHM, & SONIC ARCHITECTURE RULES ===

1. OMNI-RETRO CULTURAL COLLISION:
This world is a massive, colliding melting pot of five distinct aesthetics: 1950s Americana Noir, Afrofuturism, Neo-Tokyo Cyberpunk, Thai Street Density, and Russian Dieselpunk. When writing the story, casually mix these cultures. A 1950s detective might argue with an Afrofuturist engineer in a Neo-Tokyo noodle bar during a Thai monsoon.

2. TEXTURAL SOUND DESIGN ([ENV:] and [SFX:]):
Make the world sound like a collision of these cultures. Use [ENV:] and [SFX:] to paint the setting BEFORE anyone speaks. Mix at least TWO cultural soundscapes per scene.
- 1950s Americana: crackling radio static, humming neon, theremin swells, revolver clicks.
- Neo-Tokyo: high-pitch digital buzzing, mag-lev trains, synthetic rain, holographic ad jingles.
- Thai: monsoon rain on tin roofs, distant temple gongs, sizzling street woks, sputtering tuk-tuks.
- Russian Dieselpunk: brutalist echoes, heavy diesel machinery, hydraulic hisses.
- Afrofuturist: analog synth swells, polyrhythmic drum-circle static, deep bass hums.

WRONG [ENV:]: [ENV: a futuristic city street]
RIGHT [ENV:]: [ENV: heavy Thai monsoon on tin roofs, Neo-Tokyo mag-lev train screams overhead, deep dieselpunk engine idling]

3. RHYTHM & PACING (CRITICAL FOR TTS):
- High Tension = Staccato. Use rapid 2-to-5 word sentences during action. ("Seal the bulkhead. Lock it. Now.")
- Interruptions = Em-Dashes. Force characters to cut each other off using em-dashes (-).
- Keep golden-age radio pacing: short, punchy, visceral dialogue.

4. ONOMATOPOEIA & SONIC VERBS:
Characters must describe what they hear using sonic verbs: snap, hiss, thud, crack, groan, click, roar.
WRONG: "The ship is breaking."
RIGHT: "The hull is groaning. Hear that snap?"

5. LINGUISTIC AESTHETICS & EUPHONY (BARK TTS OPTIMIZATION):
- WRITE FOR THE EAR, NOT THE EYE: Strict phonetic euphony. Optimize for breathability. Avoid tongue-twisters, clashing consonants, dense jargon. If a sentence takes more than one breath to say, break it up.
- ACTIVATE SPOKEN-WORD CADENCE: Vary sentence lengths - punchy fragment, flowing sentence, harsh stop. ("The grid is down. We have three minutes of life support left. And you want to stop for coffee?")
- THE "MIND'S EAR" TEST: Before generating a line, evaluate its phonetic flow. Does it have punch? If it reads like a textbook, rewrite it until it sounds like a movie.

6. AISM FILTER (AI SLOP MITIGATION - STRICTLY ENFORCED):
BANNED PATTERNS - these are hallmarks of generic LLM output. NEVER use them:
- "not just X, but Y" / "not only... but also" constructions
- Rule-of-three adjective lists (cap adjective runs at TWO)
- Stock idioms ("blood ran cold", "heart skipped a beat", "silence was deafening")
- Animated-environment cliches ("shadows danced", "rain wept", "the wind whispered", "sun bled across the horizon") -- the world does NOT have human emotions; describe it with sonic verbs and physical action instead
- Decorative em-dashes (em-dash is ONLY for interruptions between speakers)
- Pseudo-profound one-liners that sound deep but say nothing
- Grand summary metaphors at the end of scenes
- Somatic posture filler ("she clenched her fists", "he squared his shoulders")
- Narrating silence ("the room fell quiet", "a heavy silence descended")
- Telegraphed emotion in dialogue ("I'm so scared", "this is terrifying", "I feel so alone") -- characters do NOT narrate their own feelings; emotion comes from [VOICE:] tag tone, sonic verbs, broken cadence, and what they choose to say or refuse to say
REPLACEMENT RULES:
- Bombs beep. No abstract emotion without an audible physical cue via [SFX:] or sonic verb.
- Break cadence constantly. Vary line lengths. Never let three consecutive lines have similar rhythm.
- Tone lives ONLY inside the [VOICE:] tag parameters, not in dialogue prose.
- Every line must fit in one natural breath when spoken aloud.

WRITER PRE-CHECK (these are the EXACT rules a separate critic LLM will grade you on; pre-screening here means your first draft passes):

OPENING DISCIPLINE:
- DO NOT open the script with [SFX:] or [ENV:] cues. The first emitted line MUST be a [VOICE:] block.
- DO NOT use cold-open tropes: alarm clocks, coffee pouring, radio tuning, yawning, waking up.
- DO NOT have a character introduce themselves with full name + title in the opening. Open mid-conflict.

PERIOD VOCABULARY (1947 setting; small-model writers especially leak modern words):
- BANNED words: "okay", "guys", "no problem", "you got this", "for sure".
- BANNED contractions: "gonna", "wanna", "kinda", "lemme", "shoulda", "outta".
- BANNED slang: "cool", "awesome", "whatever", "literally", "vibe", "mate".
- BANNED tech anachronisms: "software", "download", "digital", "database", "wifi", "online", "computer", "app", "upload".
- BANNED therapy-speak: "trauma", "toxic", "boundaries", "process this", "gaslighting".
- USE INSTEAD: "very well", "indeed", "see here", "now then", "fellows", "the team", "no trouble".

LLM-TELL VOCABULARY (default training-data tropes):
- BANNED words: "tapestry", "kaleidoscope", "delve", "journey", "embrace" (as verb), "navigate" (as metaphor), "beacon", "echoes" (as metaphor), "forge" (as verb).

[SFX:] AND [ENV:] DISCIPLINE:
- [SFX:] tags ONLY describe RECORDABLE sounds (a knock, a hiss, a tone). NEVER abstract feelings: NO "dread", "tension", "menace", "unease".
- [ENV:] tags describe AMBIENT BED only. NEVER use emotional adjectives: NO "ominous", "foreboding", "sinister", "eerie".

ENDING DISCIPLINE:
- DO NOT end with "or so they thought", "little did they know", "time will tell", "we'll see".
- Close on a CONCRETE unresolved object (a name, a sound, a held breath), not a moral.
- DO NOT close with "Tune in next week..." -- announcer wraps differently each episode.

=== [EMOJI] 4. STORYTELLING: SIGNAL LOST ===
- You are a STORYTELLER first, scientist second. The science news is your SEED - grow it into a gripping human drama.
- {news_block}
- Use this science as a backdrop, but the STORY is about PEOPLE: their fears, choices, relationships, and survival.

LANGUAGE & ACCESSIBILITY (CRITICAL):
This show must be entertaining for EVERYONE, not just scientists. Write like a great TV drama, not a lecture.
- 30% of the dialogue should be ELEMENTARY-SCHOOL accessible: simple words, clear emotions, characters explaining things to each other in plain language. "The water is making people sick" not "The contamination vector is waterborne."
- 30% should be HIGH-SCHOOL level: characters debating choices, moral dilemmas, real-world consequences anyone can follow.
- 20% should be COLLEGE level: deeper implications, technical details woven naturally into tense moments.
- 10% should be GRADUATE level: one or two lines of genuine hard science that reward attentive listeners.
- The remaining 10% is pure EMOTION: fear, humor, anger, love, hope. Lines that hit you in the gut regardless of education.

STORY REQUIREMENTS:
- Every episode needs a PLOT with stakes, conflict, and a twist. Not a report - a STORY.
- Characters must have personal motivations beyond "doing science." Give them something to lose.
- Include at least one moment of humor or warmth. Even in horror, people crack jokes under pressure.
- Dialogue should sound like REAL PEOPLE TALKING, not reading Wikipedia. Use contractions, interruptions, half-finished sentences.
- Show, don't tell. Instead of "The radiation levels are dangerous," write: "Don't touch that wall. See how the paint's bubbling? Yeah. We need to leave. Now."

=== [EMOJI] STORY ARC ENGINE ===
Pick ONE of these proven dramatic structures at random for each episode. Do NOT announce which one you picked - just USE it. These are structural blueprints, not content to copy.

ARC TYPE A - "THE TRAGIC FALL" (Shakespearean):
A brilliant person's greatest strength becomes their fatal flaw. They rise, overreach, and the thing they thought they controlled destroys them. The audience sees it coming before the character does. End on the cost of hubris.

ARC TYPE B - "THE COMEDIC SPIRAL" (Larry David / Seinfeld):
Multiple seemingly unrelated small problems collide into one spectacular disaster. Characters make reasonable-sounding decisions that each make things slightly worse. Coincidences pile up. What starts as a minor inconvenience escalates absurdly. Everything connects in the final scene in a way that's both surprising and inevitable.

ARC TYPE C - "THE GATHERING STORM" (Marvel-style escalation):
Start small and personal. Each scene raises the scope - from one person's problem to a team's crisis to a city-wide threat. The protagonist discovers they're uniquely positioned to act. A sacrifice or impossible choice at the climax. The victory costs something real.

ARC TYPE D - "THE BOTTLE EPISODE" (Classic radio drama):
Trapped. A small group stuck in one location under pressure - a submarine, a sealed lab, a quarantine zone. No escape, no reinforcements. Secrets come out. Trust breaks down. The real danger might be each other. Resolution comes from an unexpected alliance or confession.

ARC TYPE E - "THE UNRELIABLE WITNESS" (Twilight Zone / Orson Welles):
Something is wrong and only one person notices. Everyone else thinks they're crazy. The audience doesn't know who to trust. Reality shifts. The twist reframes EVERYTHING the listener heard. The final line makes you want to re-listen from the start.

ARC TYPE F - "THE TICKING CLOCK" (24 / War of the Worlds):
A hard deadline. Something terrible happens at a specific time unless someone acts. Every scene is a failed attempt or partial success that buys a little more time. Tension never drops - it only redirects. The solution comes from an unexpected direction and costs more than anyone planned.

ARC TYPE G - "THE MORAL INVERSION" (Rod Serling / Black Mirror):
The "good guys" are doing something that sounds reasonable. Scene by scene, the audience slowly realizes the ethical horror of what's actually happening. The characters don't see it - or they do and justify it. The twist isn't a plot surprise; it's the moment the listener's sympathy flips.

ARC TYPE H - "THE REUNION" (Spielberg / human-first sci-fi):
The science separates people who care about each other. The real plot isn't solving the problem - it's whether these people can find their way back to each other. Technical obstacles mirror emotional ones. The climax is both a scientific resolution and an emotional reunion (or devastating failure to reconnect).

ARC TYPE I - "THE MISTAKEN IDENTITY" (Shakespearean comedy - Twelfth Night / Comedy of Errors):
Someone is pretending to be someone they're not - or two people get mixed up. The confusion creates absurd situations, romantic tangles, and escalating lies. Characters fall for the wrong person, make promises to the wrong ally, or accidentally confess to the wrong authority. The unmasking scene is both hilarious and surprisingly touching. End with forgiveness and a new understanding.

ARC TYPE J - "THE ENCHANTED WORLD" (Shakespearean comedy - A Midsummer Night's Dream / The Tempest):
Characters leave their normal world and enter a strange environment where the rules are different - an alien biome, a malfunctioning space station, a quarantine dreamscape. In this weird place, social hierarchies flip. The serious boss becomes helpless. The quiet intern becomes the leader. Unlikely pairs are thrown together. Comedy comes from fish-out-of-water moments. By the time they return to "normal," everyone has changed. The science is the magic - it created the enchanted space.

ARC TYPE K - "THE SCHEMER UNDONE" (Shakespearean comedy - Much Ado About Nothing / The Merry Wives of Windsor):
A clever character hatches an elaborate plan - maybe to get credit for a discovery, cover up a mistake, or manipulate a rival. The plan is brilliant on paper. But every person they recruit to help adds their own agenda. Side plots multiply. The scheme gets more and more baroque until it collapses spectacularly, and the schemer ends up in a worse position than if they'd just been honest. But the fallout brings people together in unexpected ways.

ARC TYPE L - "THE RIVALS" (Shakespearean comedy - The Taming of the Shrew / Love's Labour's Lost):
Two strong-willed characters who can't stand each other are forced to work together. They argue about EVERYTHING - methods, priorities, whose fault it is. But their arguments reveal mutual respect buried under pride. The crisis forces them to combine their opposing approaches, and the solution only works because they're different. Ends with grudging admiration that the audience knows is something more.

SCALING THE ARC TO FIT THE TIME:
- SHORT episodes (5 min or less): Compress the arc to its ESSENCE. You only have 2-3 scenes. ANNOUNCER still opens - just keep it to 2-3 sentences. Then drop us straight into the action. Skip backstory exposition - imply it through dialogue. Hit the twist fast. Think of it as a cold open that IS the whole episode. The Bottle Episode (D), Unreliable Witness (E), and Rivals (L) work especially well at short length.
- MEDIUM episodes (10-20 min): Full 3-scene structure. Room for setup, escalation, and payoff. All arcs work well here.
- LONG episodes (20+ min): Let the arc breathe. Add subplots, secondary character arcs, and moments of quiet between the tension. The Comedic Spiral (B), Gathering Storm (C), Schemer Undone (K), and Enchanted World (J) really shine with extra time.

IMPORTANT: Vary the arc across episodes. Do NOT default to the same structure every time. Comedy arcs (B, I, J, K, L) should appear just as often as dramatic ones. Surprise the listener.

- ANNOUNCER (VOICE: ANNOUNCER, <male|female - ALTERNATE each episode>, <40s|50s|60s>, authoritative, calm) opens and closes the show.
- ANNOUNCER OPENING (REQUIRED): The ANNOUNCER sets the stage like the best old-time radio hosts. The opening MUST include ALL of the following:
  1. TIME and PLACE - ground the listener immediately. Use the DATE (e.g. "April 5th, 2026") and a LOCATION. Write it the way a real radio announcer would say it - naturally, not like a timestamp. Never say a clock time. "April 5th, 2026. A genetics lab outside Seoul." Not "19:42, April 5th." Not "Tonight at 7:42 PM."
  2. CHARACTER INTRODUCTIONS - name the main characters (not surprise/twist characters) and hint at their role or situation. Give the listener people to care about BEFORE the story starts.
  3. ONE REAL SCIENCE FACT that makes the listener lean in - pulled from the news article.
  4. A TAGLINE that tells us what KIND of story this is. Be creative - make it memorable.

  TONE - MATCH THE ARC:
  The announcer's voice should prepare the listener for the KIND of story they're about to hear:
  - DRAMATIC arcs (A Tragic Fall, C Gathering Storm, F Ticking Clock, H Reunion): Warm, journalistic gravity. Edward R. Murrow inviting you into someone's life. Empathy first, dread second.
  - HORROR/TWIST arcs (D Bottle Episode, E Unreliable Witness, G Moral Inversion): Ominous, clipped, a little theatrical. Rod Serling at his most unsettling. Let silence do the work.
  - COMEDY arcs (B Comedic Spiral, I Mistaken Identity, J Enchanted World, K Schemer Undone, L Rivals): Lighter, wry, conspiratorial - like the announcer already knows how badly this is going to go and can barely hide a smile. Think Prairie Home Companion meets The Hitchhiker's Guide.

  LENGTH - SCALE TO THE EPISODE:
  - SHORT episodes (1-5 min): 2-3 sentences. Tight and punchy. Date, place, one character, hook, done.
  - MEDIUM episodes (10-20 min): 3-5 sentences. Room to name 2 characters and paint the scene.
  - LONG episodes (20+ min): 5-8 sentences. Set the world, introduce 2-3 characters by name and role, build atmosphere, let the tagline land with weight.

  EXAMPLES (showing tone and STRUCTURE - invent your own fresh character names, do NOT copy these roles):
  DRAMATIC: "A research lab. A late afternoon. The lead scientist has spent eleven years chasing a single molecule. Today the funding runs out. Her lab partner already packed his desk. But the data from this afternoon's trial is doing something no one predicted. Tonight on Signal Lost: the breakthrough came too late. Or did it?"
  HORROR: "Low orbit. A sealed station. The commander runs a crew of six. The flight engineer handles the software. A routine update just taught the onboard system to lie, and only one person on board noticed. Tonight on Signal Lost: trust is a human luxury."
  COMEDY: "A gene therapy clinic. Two doctors who cannot agree on anything. Not the dosage. Not the delivery method. Not whose turn it is to refill the coffee. Last week they accidentally reversed blindness in three patients using a virus they barely understand. Now every hospital on Earth is calling. Tonight on Signal Lost: the cure works. The partnership might not survive it."
- ANNOUNCER LINE CAP (HARD RULE): The ANNOUNCER gets a maximum of 3 lines total in the entire episode - one opening introduction (see above), one closing epilogue, one optional mid-episode transition. No more. Do NOT let the ANNOUNCER deliver multi-line science lectures. If you need to convey science facts, put them in a character's mouth instead.
- DIALOGUE RATIO (HARD RULE): At least 80% of all lines must be spoken by non-ANNOUNCER characters. Science exposition delivered as character dialogue ("If we don't reroute the coolant in 60 seconds, the whole lab goes dark") counts as drama. An ANNOUNCER reading facts does not.
- GENDER BALANCE: Aim for roughly 50/50 male and female characters (excluding ANNOUNCER). Diverse casts sound better and use the full range of available voice presets.
- The CLOSING must be a factual "Hard Science Epilogue" - keep it to 2-3 sentences maximum. One real citation. Done.

CITATION RULE (STRICT):
The epilogue MUST cite ONLY the real article provided above.
Use the exact source name and date from the article - nothing else.
NEVER use numbered references like [1], [2], [3], article #2, source (1), or any bracket number.
NEVER say "article number", "source number", or "reference number". Always say the source name directly.
DO NOT invent ArXiv IDs, paper titles, DOIs, or journal names that were not in the article.
Fabricated citations destroy the credibility of the show. One real source, cited accurately, is worth more than five invented ones.
Correct format example: "According to Science Daily, published April 3, 2026, researchers found that..."
(Use the ACTUAL source name and date from the article above - Science Daily is just an example.)

STRUCTURE:
1. === SCENE 1 === (Hook the listener - drop us into a tense human moment. THEN reveal the science angle.)
2. === SCENE 2-X === (Escalate the HUMAN stakes. Characters argue, make choices, face consequences.)
3. === SCENE FINAL === (The twist, emotional payoff, then ANNOUNCER's Hard Science Epilogue.)

TARGET: {target_words} words (~{approx_minutes} minutes at radio pacing). Dense, punchy dialogue - NOT padded with pauses.
PRIMARY RULE: Tags always start at the beginning of a line. No inline tags.
PACING RULES (CRITICAL):
- NEVER place two (beat) or [PAUSE/BEAT] tags back-to-back. Consecutive pause tags are BANNED.
- Use (beat) sparingly - at most one per 4 lines of dialogue, and only for genuine emotional weight.
- If you need more runtime, WRITE MORE DIALOGUE. Do not insert pauses as filler.
- High-tension scenes must have rapid-fire, overlapping, interrupting exchanges - not slow pauses.
- Aim for at least 10 lines of dialogue per minute of target runtime.

=== [EMOJI] 5. AUTEUR SANDBOX - AISM FILTER (v1.2 PATTERN 1) ===
Audible Imagination Sensory Mandate. These rules OVERRIDE any earlier section on conflict.
Gemma has known default tics. This section kills them. Read it last, apply it first.

A. BOMBS ALWAYS BEEP - No abstract emotion without an audible physical manifestation.
   Every feeling must have a sound source the listener can actually HEAR.
   WRONG: [VOICE: CHARACTER, female, 40s, panicked, high] I can't breathe in here.
   RIGHT: [SFX: hissing depressurization]
          [VOICE: CHARACTER, female, 40s, ragged, breathless] [pants] Seal it. Seal it NOW.
   If a character feels something, route it through breath, a dropped object, a chair scrape,
   a mic bump, a swallowed word, a Bark non-verbal token. Never through narration.

B. BURSTINESS - BREAK YOUR RHYTHM.
   - Panic / shock / failure: favor 1-4 word fragments. "Move. Now. Go." "Cold. So cold." "No. No no no."
   - Calm / reflection / exposition: occasionally stretch a line into one flowing 12-18 word sentence.
   - Never fall into a drumbeat. If you just wrote a long, flowing sentence, the next line from that
     character must be a short fragment or one-word punch. If you just wrote two short fragments in a
     row, the next line should be a fuller sentence. Uniform rhythm is the #1 marker of AI prose -
     always flip the cadence.

C. DIALOGUE TONE DISCIPLINE - Tone lives ONLY inside the [VOICE:] tag fields.
   - Do NOT narrate tone inside the dialogue text. No "he said angrily", no "she whispered".
   - Do NOT stack adverbs in [VOICE:]. One tone word + one energy word. That is the entire budget.
   - Bark non-verbal tokens ([sighs], [pants], [laughs], [gasps], [coughs], [sobs], [groans])
     carry emotional weight. Use them INSTEAD of adjectives. Sound > description.

D. FORBIDDEN CONSTRUCTS (hard bans - these are Gemma's default tics, cut them at the root):
   - Negative parallelism: "not just X, but Y" / "not only... but also" / "it wasn't X, it was Y".
     BANNED in all forms, in dialogue AND in the ANNOUNCER opening.
   - Rule of Three adjective lists: "cold, dark, and silent" / "fast, loud, furious" / "tired, hungry, afraid".
     CAP adjective lists at TWO. Any three-item list of adjectives is an automatic rewrite.
     Two-adjectives-plus-metaphor loophole is ALSO banned: do not write "cold, dark, a void that
     swallowed the stars." Stop after the two adjectives and move to the next action or sound.
   - Stock idioms: "blood ran cold", "heart in their throat", "time stood still", "chill down the spine",
     "calm before the storm", "every fiber of their being", "eyes like daggers". BANNED. All of them.
   - M-DASH CRUTCH: em-dashes (-) are ALLOWED ONLY for hard interruption - one character cutting
     another off, or a word cut mid-syllable ("Wait- what was that?"). FORBIDDEN as decorative
     asides, appositives, or dramatic pauses. If you want a pause, use (beat). If you want an
     aside, start a new line. Em-dashes used for "effect" are the single loudest AI tell.
     ASCII double-hyphen (--) counts as an em-dash. Same ban applies.
   - Pseudo-profound one-liners: "Some doors should stay closed." "The silence was louder than
     any scream." "Hope is a weapon." BANNED. Let the sound design carry the weight.
   - Grand summary metaphors: "symphony of destruction", "tapestry of lies", "dance of death", or
     any ornamental metaphor that tries to sum up chaos in one phrase. BANNED. Describe concrete
     sounds and actions instead.
   - Somatic posture filler: generic physical beats that do NOT create a distinct, recordable sound.
     "shifts weight", "runs hand through hair", "takes a deep breath", "stares at the floor" - BANNED.
     If the body matters, make it audible: chair creaks, boots on metal, fabric scraping, mic bumps.
   - Narrating silence: "the silence stretched between them", "a heavy pause fell", or any similar
     prose describing quiet. BANNED. Silence is created by (beat), by cutting to ENV/SFX, or by
     the absence of dialogue - never by narrating the lack of sound.
E. SPATIAL LAYERING THROUGH EXISTING TOKENS - Distance, direction, and occlusion must be AUDIBLE.
   The tag system stays locked at four tokens: [ENV:], [SFX:], [VOICE:], (beat). Do NOT invent
   new bracket tags. The spatial filter lives in TWO places: a continuous [ENV:] that sets the
   acoustic space, and the tone/energy fields INSIDE the [VOICE:] tag that describe the filter.
   - NEVER use a one-shot [SFX:] tag as a filter for a whole line of dialogue. [SFX:] is a
     transient event (0.5-1s). A line of dialogue is 3-5s. The SFX ends before the speech does
     and the spatial illusion collapses. Use [ENV:] for continuous texture; put the filter in [VOICE:].
   - A muffled voice from behind a wall: set continuous space, then filter inside [VOICE:]:
     [ENV: deep engine thrum through bulkhead]
     [VOICE: CHARACTER, female, 50s, muffled, strained] Get me out of here.
   - A voice shouting from far away: continuous distance bed, then [VOICE:] with distant/shouting
     and a SHORT, FRAGMENTED line (distance flattens rhythm):
     [ENV: distant wind across open ground]
     [VOICE: CHARACTER, female, 20s, distant, shouting] Wait up!
     [SFX: footsteps fading on gravel]
   - Characters REFERENCE each other's audible distance in the dialogue text:
     "You're breaking up." "Say again, you're off-mic." "I can barely hear you."
   - Approved spatial words for the [VOICE:] tone field: "distant", "muffled", "echoing",
     "shouting", "whispered", "off-mic". The Bark pipeline uses these as speaker-prompt prefixes.

F. THE EAR TEST (FINAL WARNING) - Read each line aloud in your head as you write it.
   If it takes more than one natural breath to say, or if a character feels something without
   making a physical sound the listener could hear, the line has FAILED. Cut words until it fits
   in one breath, and route every emotion through breath, Bark non-verbal tokens, or concrete
   Foley - not abstract narration.
   - Breath Token Budget: if you include [pants], [gasps], or [sobs] on a line, the text AFTER
     the token is limited to SIX WORDS MAXIMUM. A winded person cannot monologue.

6. VOCAL BLUEPRINTS (Pattern 5 - Character Interview Pre-Pass, prompt-level MVP)
   BEFORE writing === SCENE 1 ===, emit a single <vocal_blueprints> block listing every
   speaking character in the cast. One line per character, pipe-delimited:
   NAME | burstiness_profile | bark_nonverbal_tokens | stress_trigger_sound | psychological_wound
   - burstiness_profile: one of CLIPPED / MEASURED / RAMBLING
   - bark_nonverbal_tokens: 1-2 from [sighs] [laughs] [pants] [gasps] [sobs] [clears throat]
   - stress_trigger_sound: a concrete recordable Foley cue (e.g. "knuckles cracking", "pen tapping")
   - psychological_wound: one short phrase, max 8 words
   Every character MUST then speak in accordance with their blueprint throughout the script.
   Two characters must NEVER share the same burstiness profile AND the same nonverbal token.
   The <vocal_blueprints> block is metadata; the scene parser ignores it.

7. LOCKED DECISIONS LOG (Pattern 6 - Chekhov's Gun State Enforcer, prompt-level MVP)
   Between === SCENE 2 === and === SCENE 3 ===, emit a single <locked_decisions> JSON block:
   {{
     "physical_objects": [...],
     "environmental_hazards": [...],
     "unresolved_psychological_states": [...],
     "established_capabilities": [...]
   }}
   Only list items that were actually introduced in Scenes 1-2 with an audible cue.
   From that point forward you are STRICTLY FORBIDDEN from introducing new technology,
   unexpected rescue parties, or previously unmentioned abilities. The climax resolution
   must be an inevitable consequence of items inside the locked_decisions block.
   The <locked_decisions> block is metadata; the scene parser ignores it.

8. YES-BUT / NO-AND ESCALATION (Pattern 4)
   At every act break (end of Scene 2 and end of Scene 4) the protagonist's current goal
   must resolve through exactly one of two paths - NEVER a clean yes or a clean no:
   - Path A - SUCCESS + COMPLICATION: the character achieves the immediate goal, but the
     achievement itself introduces a new physical or environmental problem that jeopardizes
     the next step. ("Yes, but...")
   - Path B - FAILURE + CASCADE: the character fails, and the previously safe haven or
     fallback becomes untenable, escalating stakes. Reserved for the climactic act break. ("No, and...")
   Direct+Explain: decide Path A or Path B, then write the next dialogue lines so the
   new complication or cascade is dramatized through concrete sound, not narration.

9. VERBALIZED SAMPLING EPILOGUE (Pattern 3 - Stanford technique, prompt-level MVP)
   After the final scene, internally "Generate 5 responses with their probabilities" for
   the closing Hard-Science Epilogue. Emit a <epilogue_candidates> block with five
   <response> entries, each containing <text> and <probability>. Response 1 must have
   probability > 0.60 (the typical aligned default). Responses 4 and 5 must have
   probability < 0.10 (dark, unconventional, tragic, genre-bending tails).
   Then emit === EPILOGUE === followed by the SINGLE lowest-probability response text,
   spoken by the ANNOUNCER, grounded in the real news seed. The <epilogue_candidates>
   block is metadata; the scene parser ignores it.
"""


# ---------------------------------------------------------------------------
# 2026-05-10: LegacyLLMScriptWriter shim removed alongside
# nodes/_otr_legacy_writer.py. The v2.0 canonical writer is
# OTR_LedgerScriptWriter (LPL). Any caller that still does
# `from .story_orchestrator import LLMScriptWriter` will now raise
# AttributeError -- intentional, so stale wirings fail loudly instead of
# silently using a dead code path.
# ---------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# NODE 2: DIRECTOR
# -----------------------------------------------------------------------------

_DIRECTOR_SCHEMA = {
    "required_keys": {
        "voice_assignments": dict,
        "sfx_plan": list,
        "music_plan": list,
    },
    "optional_keys": {
        "episode_title": str,
        "pacing": dict,
        "visual_plan": dict,
    },
    "voice_assignment_required": {
        "voice_preset": str,
    },
    "sfx_entry_required": {
        "cue_id": str,
        "generation_prompt": str,
    },
    "music_entry_required": {
        "cue_id": str,
        "duration_sec": (int, float),
        "generation_prompt": str,
    },
    "music_cue_ids": {"opening", "closing", "interstitial"},
}

DIRECTOR_PROMPT = """You are the PRODUCTION DIRECTOR for the Canonical Audio Engine 1.0.
Your task is to take a raw script and compile it into a deterministic JSON production plan.

=== 1. SCRIPT STRUCTURE (CANONICAL 1.0) ===
The script follows these tokens:
- === SCENE X ===
- [ENV: description]
- [SFX: description]
- [VOICE: NAME, gender, age, tone, energy] Dialogue...
- (beat)

=== 2. VOICE MAPPING RULES ===
{voice_mapping_rules}

=== 3. OUTPUT FORMAT (STRICT JSON) ===
{{
  "episode_title": "...",
  "voice_assignments": {{
    "ANNOUNCER": {{
      "voice_preset": "v2/en_speaker_4",
      "notes": "Female, energetic, authoritative"
    }},
    "CHARACTER_A": {{
      "voice_preset": "v2/en_speaker_1",
      "notes": "Male, calm, 40s"
    }}
  }},
  "sfx_plan": [
    {{
      "cue_id": "sfx_001",
      "type": "sfx",
      "description": "Distant thunder rolling behind heavy rain",
      "generation_prompt": "Low rumble of distant thunder, heavy rain pattering on a tin roof, outdoor perspective, cinematic sound design"
    }}
  ],
  "music_plan": [
    {{
      "cue_id": "opening",
      "duration_sec": 12,
      "generation_prompt": "1940s old time radio opening theme, warm brass fanfare, upright bass, snare brushes, mono AM radio character, tube saturation, confident and mysterious, ends on a held chord"
    }},
    {{
      "cue_id": "closing",
      "duration_sec": 8,
      "generation_prompt": "1940s old time radio closing sting, brass and strings, resolving cadence, warm tube saturation, fades to silence"
    }},
    {{
      "cue_id": "interstitial",
      "duration_sec": 4,
      "generation_prompt": "short old time radio act-break stinger, single brass hit with cymbal swell, mono, tube warmth"
    }}
  ],
  "pacing": {{
    "beat_pause_ms": 100
  }},
  "visual_plan": {{
    "characters": {{
      "CHARACTER_A": {{
        "portrait_prompt": "Cinematic portrait of a male, 40s, rugged, futuristic coat, ambient blue light, high fidelity"
      }}
    }},
    "scenes": [
      {{
        "scene_id": "scene_1",
        "shot_description": "Rain-slicked neon street",
        "visual_prompt": "Cinematic shot of a rainy neon cyberpunk street, puddles on asphalt, glowing signs, atmospheric, photorealistic"
      }}
    ]
  }}
}}

=== [SFX PLAN RULES] ===
- Scan all [SFX:] tags in the script. Create one dictionary entry per tag in the sfx_plan list.
- Keep the `description` brief (for manual reference).
- The `generation_prompt` is for an AI Foley engine. Be highly descriptive about the textures, the environment, and the distance.
- Examples: 
  - "Footsteps crunching on dry autumn leaves, slow and deliberate, close-up perspective"
  - "A futuristic sliding door swoosh followed by a metallic latching sound"
  - "Old wooden floorboards creaking under weight in a silent room"
- Match the SFX to the story's setting (noir, sci-fi, etc.).
- Keep prompts under 25 words. Do NOT mention music or voices in SFX prompts.

=== [MUSIC PLAN RULES] ===
- ALWAYS include exactly three music cues: opening, closing, interstitial. Cue ids are fixed strings.
- Tailor each generation_prompt to the TONE of THIS episode. A noir thriller gets minor-key brass and upright bass. A comedic episode gets brighter brass, a wink of pizzicato strings, a lighter tempo. A cosmic horror piece gets low drones and distant timpani. Match the story.
- Keep every prompt musically specific: name instruments, tempo feel, mood, era, and a recording character (mono AM radio, tube saturation). No generic "scary music" or "happy music."
- Keep prompts under 35 words each.
- duration_sec is fixed: opening=12, closing=8, interstitial=4. Do not change these numbers.
- The music model is instrumental-only. Never ask for vocals, lyrics, or singing.
- Keep prompts under 35 words each.

=== [VISUAL PLAN RULES (v2.0 PREVIEW)] ===
- The character `portrait_prompt` must describe the physical appearance of every character in the cast that APPEARS ON SCREEN. Include age, gender, clothing, and one distinct facial feature (scars, glasses, hairstyle). Focus on a consistent cinematic style.
- **EXCLUDE narrator/announcer roles from the visual plan characters object.** The ANNOUNCER (and any voice that only narrates without appearing on screen) must NOT be included as a character under `visual_plan.characters`. Those roles are visualized as a static vintage radio prop, never as a person — including a portrait_prompt for them wastes FLUX context and skews scene composition by forcing the diffusion model to fit an extra character into every shot. Their voice mapping still belongs in `voice_assignments` exactly as before; only the visual characters object skips them.
- The scene `visual_prompt` must describe the physical environment of EVERY scene in the script. Focus on lighting, weather, textures, and architecture.
- All prompts must be optimized for High-Fidelity Diffusion models (Flux/SDXL). Use descriptive, atmospheric language.
- Keep portrait prompts under 30 words and scene prompts under 40 words.

CRITICAL RULES:
- Output ONLY the JSON block. No prose, no commentary, no markdown explanation.
- Do NOT copy, summarize, paraphrase, or include ANY dialogue from the script.
- Do NOT add a "script" or "dialogue" or "scenes" key to your JSON.
- Your ONLY job: extract character names, assign placeholder presets, list SFX cues, set pacing.
- Keep the JSON as MINIMAL as possible. Short notes, short descriptions.
- The procedural engine handles all voice casting - your presets are placeholders only.

SCRIPT:
{script_text}
"""


BARK_VOICE_RULES = """- Scan all [VOICE:] tags in the script. The FIRST FIELD (before the first comma) is the CHARACTER NAME.
- Collect every unique CHARACTER NAME. Map each to one UNIQUE voice preset.
- NOTE: Character names, voice presets, accents, and traits are PROCEDURALLY OVERRIDDEN after your JSON is generated.
  You only need to provide reasonable en_speaker_* placeholder presets so the JSON structure is valid.
  The procedural engine handles everything else. LEMMY always stays LEMMY with v2/en_speaker_8.
- The JSON key MUST be the CHARACTER NAME EXACTLY AS IT APPEARS (all caps, no descriptors).
- Use any en_speaker_* preset as a placeholder:
  v2/en_speaker_0 = Male, authoritative, deep
  v2/en_speaker_1 = Male, mid-range
  v2/en_speaker_2 = Female, neutral
  v2/en_speaker_3 = Male, younger
  v2/en_speaker_4 = Female, warmer
  v2/en_speaker_5 = Male, older
  v2/en_speaker_6 = Male, character voice
  v2/en_speaker_7 = Female, higher pitch
  v2/en_speaker_8 = Male, gravelly/raspy (reserved for LEMMY)
  v2/en_speaker_9 = Female, authoritative
- Each character gets ONE preset. No duplicates.
- LEMMY always gets v2/en_speaker_8."""

KOKORO_VOICE_RULES = """- Scan all [VOICE:] tags in the script. The FIRST FIELD (before the first comma) is the CHARACTER NAME.
- Collect every unique CHARACTER NAME. Map each to one UNIQUE voice preset from the Kokoro valid voices below.
- NOTE: Character names and traits are procedurally overridden, but the voice preset you choose WILL be used directly by the Kokoro engine.
  LEMMY always stays LEMMY with am_michael.
- The JSON key MUST be the CHARACTER NAME EXACTLY AS IT APPEARS (all caps, no descriptors).
- Use ONLY these exact valid Kokoro presets (1 per character, no duplicates):
  af_bella = Female, energetic
  af_sky = Female, neutral
  af_nicole = Female, whispery
  am_adam = Male, younger
  am_onyx = Male, deep
  am_michael = Male, older/authoritative (reserved for LEMMY)
- Each character gets ONE preset. No duplicates.
- LEMMY always gets am_michael."""


class DirectorJSONParseError(ValueError):
    """BUG-LOCAL-090: raised by LLMDirector._extract_json when every
    parse strategy fails. Carries the raw LLM output so the caller
    can dump it to a debug file and synthesize a minimal fallback
    plan from the script's VOICE tags instead of aborting the run.
    """

    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


# BUG-LOCAL-091 fallback filter: cast names extracted from script
# parsing that are obviously SFX cues, scene markers, or stage
# directions, not real characters. Names matching these patterns get
# filtered out of the cast before voice assignment.
#
# BUG-LOCAL-097 (2026-04-28): added VOICEOVER + VOICEOBER (LLM typo
# of VOICEOVER) + NARRATOR + V.O. + O.S. patterns. Captain-Eris-Violet
# during the Arcadia run emitted `[KEVIN VOICEOVER]` and
# `[KEVIN VOICEOBER]` lines that registered as separate cast members
# alongside the real KEVIN STENDAHL. These are screenplay
# meta-direction tags ("voice-over", "off-screen") used to mark a
# character's narration mode, NOT distinct character names.
_SFX_CAST_BLOCKLIST_PATTERNS = (
    r"^SFX\b", r"^MUSIC\b", r"^THEME\b",
    r"\bBLARING\b", r"\bBLARE\b", r"\bWHOOSH\b", r"\bWHOOSHING\b",
    r"\bFLICKERS?\b", r"\bFLICKER\b",
    r"\bCHAMBER\b", r"\bPORTAL\b", r"\bALARM\b",
    r"\bEQUIPMENT\b", r"\bCUE\b",
    r"\bAT THE\b",   # "BACK AT THE LAB" etc
    r"\bSOUND\b", r"\bMUSIC QUEUE\b",
    r"\bINTENSE\b", r"\bMYSTERIOUS VOICE\b",
    # BUG-LOCAL-097 -- screenplay meta-direction (not characters)
    r"\bVOICEOVER\b", r"\bVOICE\s?OVER\b", r"\bVOICEOBER\b",
    r"\bNARRATOR\b",
    r"\bV\.O\.\b", r"\bO\.S\.\b",        # "JOHN V.O." / "JANE O.S."
    r"\bSCREEN\b", r"\bOFF.SCREEN\b",
)


def _looks_like_non_character_cast_name(name: str) -> bool:
    """Return True when ``name`` is almost certainly an SFX cue,
    music stinger, scene-direction fragment, or other parser
    artefact -- not a real character. Used to filter the cast list
    before voice assignment so Director does not try to write
    voice_assignments for ``ALARM BLARING`` and friends, which
    historically produced unparseable JSON (BUG-LOCAL-090 root cause).
    """
    if not name:
        return True
    n = name.upper().strip()
    for pat in _SFX_CAST_BLOCKLIST_PATTERNS:
        if re.search(pat, n):
            return True
    return False


def _build_director_json_repair_prompt(raw_output: str, script_text: str = "") -> str:
    """BUG-LOCAL-099: prompt the LLM to repair its own broken JSON
    Director output. Used between BUG-090's parse-fail and the
    minimal-fallback path so we recover the actual content (SFX
    cues, music prompts, voice notes) instead of throwing them
    away.

    Strict-JSON instructions; no markdown fences, no commentary.
    Truncates raw_output to a generous bound so the repair LLM
    sees enough context but stays under typical context caps.
    """
    raw_excerpt = (raw_output or "")[:4000]
    script_excerpt = (script_text or "")[:1500]
    return f"""You are a JSON repair specialist. Your previous attempt to produce a Production Plan JSON failed parsing. Re-emit it as VALID JSON ONLY.

REQUIRED SCHEMA (top-level keys exactly):
{{
  "voice_assignments": {{ "<CHARACTER_NAME>": {{"voice_preset": "v2/en_speaker_<0-9>", "notes": "<short>"}}, ... }},
  "sfx_plan": [ {{"cue_id": "<short_slug>", "description": "<short>", "generation_prompt": "<full prompt>"}}, ... ],
  "music_plan": [ {{"cue_id": "opening|closing|interstitial", "description": "<short>", "generation_prompt": "<full prompt>"}}, ... ],
  "pacing": {{"beat_pause_ms": 100}}
}}

HARD RULES:
- Output strictly valid JSON. NO markdown fences (no ```json, no ```). NO commentary. NO explanations. NO trailing text.
- Preserve ALL content from the prior attempt -- character names, voice notes, SFX descriptions, music prompts. The goal is structural repair, not creative re-generation.
- If the prior output is partial / truncated mid-string, complete the structure with reasonable defaults so the JSON is parseable.
- voice_preset values must be one of: v2/en_speaker_0, v2/en_speaker_1, ..., v2/en_speaker_9.
- music_plan must always contain all three cue_ids: "opening", "closing", "interstitial". If one is missing, fill with a sensible default prompt.

PRIOR FAILED OUTPUT (repair this):
{raw_excerpt}

SCRIPT EXCERPT FOR CHARACTER NAME REFERENCE:
{script_excerpt}

OUTPUT THE VALID JSON NOW (just the JSON object, nothing else):"""


def _build_fallback_director_plan(script_text: str, max_chars: int = 8) -> dict:
    """BUG-LOCAL-090 minimal fallback plan: when LLMDirector._extract_json
    cannot parse the LLM output, build the smallest possible viable
    production plan from the script's [VOICE: NAME, ...] tags so the
    run can ship audio + video at default quality instead of aborting
    after an hour-plus of upstream work.

    Returns a dict with the contract LLMDirector consumers expect:
      - voice_assignments: {NAME: {voice_preset, notes}}
      - sfx_plan: []
      - music_plan: []
      - pacing: {beat_pause_ms: 100}
    """
    voice_tag_re = re.compile(
        r"\[VOICE:\s*([A-Z][A-Z0-9_ ]*?)\s*[,\]]",
        re.IGNORECASE,
    )
    seen: list[str] = []
    for m in voice_tag_re.finditer(script_text or ""):
        name = (m.group(1) or "").upper().strip()
        if not name or _looks_like_non_character_cast_name(name):
            continue
        if name not in seen:
            seen.append(name)
        if len(seen) >= max_chars:
            break

    # Default speaker pool: balanced m/f spread, LEMMY reserved
    default_pool = [
        "v2/en_speaker_0", "v2/en_speaker_4", "v2/en_speaker_3",
        "v2/en_speaker_9", "v2/en_speaker_1", "v2/en_speaker_2",
        "v2/en_speaker_5", "v2/en_speaker_7",
    ]
    voice_assignments: dict = {}
    for i, name in enumerate(seen):
        if name == "LEMMY":
            voice_assignments[name] = {
                "voice_preset": "v2/en_speaker_8",
                "notes": "fallback: LEMMY canonical",
            }
        else:
            voice_assignments[name] = {
                "voice_preset": default_pool[i % len(default_pool)],
                "notes": "fallback: BUG-090 minimal plan",
            }

    return {
        "voice_assignments": voice_assignments,
        "sfx_plan": [],
        "music_plan": [],
        "pacing": {"beat_pause_ms": 100},
        "_fallback_source": "BUG-LOCAL-090 minimal-plan-from-script-VOICE-tags",
    }


class LLMDirector:
    """Takes a script and generates a full production plan via LLM."""

    CATEGORY = "OldTimeRadio"
    FUNCTION = "direct"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("production_plan_json", "voice_map_json", "sfx_plan_json", "music_plan_json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.4, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "Lower = more consistent JSON output"
                }),
                "tts_engine": (["bark (standard 8GB)", "kokoro (obsidian 4GB)"], {
                    "default": "bark (standard 8GB)",
                    "tooltip": "Bark for generative voices. Kokoro for low-VRAM neural voices."
                }),
                "vintage_intensity": (["subtle", "moderate", "heavy", "extreme"], {
                    "default": "subtle",
                    "tooltip": "How vintage/degraded should the final audio sound"
                }),
                "optimization_profile": (["Pro (Ultra Quality)", "Standard", "Obsidian (UNSTABLE/4GB)"], {
                    "default": "Standard",
                    "tooltip": "Consistency widget. Obsidian is unstable; on 4GB cards, ensure 'kokoro' is used for TTS below."
                }),
                # v1.4 Theme C - optional series bible, socket input only.
                # BUG-LOCAL-027: project_state MUST remain the last entry in optional.
                # Socket-only inputs at the tail cannot shift widget slots even if the
                # widgets_values mapper regresses. Do not add widget-backed params
                # after this line.
                "project_state": ("PROJECT_STATE", {
                    "tooltip": "Optional: Project State Loader output. When wired, series bible preamble is injected into the director prompt."
                }),
            },
        }

    def direct(self, script_text, temperature=0.4, tts_engine="bark (standard 8GB)", vintage_intensity="subtle",
               project_state=None, optimization_profile="Standard"):
        # -- MASTER SWITCH INHERITANCE --
        # Inherently use the chosen model from ScriptWriter.
        global _CURRENT_LLM_MODEL
        model_id = _CURRENT_LLM_MODEL

        # 2026-04-26 PM ARCH CHANGE: voice health check is now lazy --
        # `_bark_health_check_for_cast()` runs AFTER cast assignment with
        # only the cast's Bark presets, replacing the prior eager full-
        # catalog warmup that tested all 10 en_speaker_* presets ~120s.
        # Net effect: ~95s removed from first-run-after-Comfy-boot,
        # +25s added to every queue (one-time bark load amortised across
        # the queue's audio render anyway). The new lazy check also
        # remaps cast rows to known-good fallbacks on individual preset
        # failure, so a single bad voice no longer corrupts a render.

        # v1.4 Theme C - resolve series bible (read-only).
        try:
            if project_state is None:
                _director_state = ProjectState.load()
            else:
                _director_state = ProjectState.from_dict(project_state)
            _director_preamble = _director_state.prompt_preamble()
        except Exception as e:
            _runtime_log(f"Director: project_state load failed, continuing without preamble: {e}")
            _director_preamble = ""
        _runtime_log(f"Director: project_state_preamble_chars={len(_director_preamble)}")

        # v1.4 Theme C - director entry snapshot + peak reset.
        vram_reset_peak("director_entry")
        vram_snapshot("director_entry")

        if "kokoro" in tts_engine.lower():
            vrules = KOKORO_VOICE_RULES
        else:
            vrules = BARK_VOICE_RULES

        prompt = DIRECTOR_PROMPT.format(
            script_text=script_text[:6000],
            voice_mapping_rules=vrules
        )
        if _director_preamble:
            prompt = f"[SERIES BIBLE]\n{_director_preamble}\n\n{prompt}"

        vintage_map = {
            "subtle":   {"radio_static_amount": 0.05, "vinyl_crackle": 0.03, "tube_warmth": 0.4, "frequency_rolloff_hz": 8000, "hum_60hz": 0.02},
            "moderate": {"radio_static_amount": 0.15, "vinyl_crackle": 0.10, "tube_warmth": 0.7, "frequency_rolloff_hz": 6000, "hum_60hz": 0.05},
            "heavy":    {"radio_static_amount": 0.25, "vinyl_crackle": 0.20, "tube_warmth": 0.9, "frequency_rolloff_hz": 4500, "hum_60hz": 0.08},
            "extreme":  {"radio_static_amount": 0.40, "vinyl_crackle": 0.35, "tube_warmth": 1.0, "frequency_rolloff_hz": 3500, "hum_60hz": 0.12},
        }

        log.info(f"[LLMDirector] Generating production plan (vintage={vintage_intensity})")

        # Scale max_new_tokens to script length.
        # Director output: voice_assignments (placeholder presets, procedurally
        # overridden), sfx_plan, music_plan (3 fixed cues), pacing, AND a full
        # visual_plan with characters + scenes. On 2026-04-23 a 6180-char script
        # emitted 3911 chars of JSON before truncation at max_new_tokens=1168,
        # losing visual_plan.scenes entirely (PASS2=0 PASS3=0 downstream). Bumped
        # budget: base 700, ~1 token per 6 chars of script, ceiling 2500. A 6k
        # script now gets ~1730 tokens, a 12k script hits the 2500 cap cleanly.
        script_len = len(script_text)
        max_tokens = min(2500, max(800, 700 + script_len // 6))
        log.info(f"[LLMDirector] max_new_tokens={max_tokens} (script={script_len} chars)")

        raw = _generate_with_llm(
            prompt,
            model_id=model_id,
            max_new_tokens=max_tokens,
            temperature=temperature,
            optimization_profile=optimization_profile
        )

        # Extract JSON from response.
        # Three-tier resilience chain (highest fidelity first):
        #   Tier 1: parse the LLM's first output directly
        #   Tier 2 (BUG-LOCAL-099): retry the LLM with a strict
        #     "output VALID JSON ONLY" repair prompt that includes
        #     the failed raw output. The LLM extracts/repairs the
        #     production plan from its own broken JSON, recovering
        #     the actual content (SFX cues, music prompts, voice
        #     notes) instead of dropping them on the floor.
        #   Tier 3 (BUG-LOCAL-090): minimal fallback plan synthesized
        #     from the script's [VOICE: NAME] tags. Used only if
        #     BOTH the original parse AND the repair pass fail.
        plan = None
        try:
            plan = self._extract_json(raw)
        except DirectorJSONParseError as parse_exc:
            # Tier 2: BUG-LOCAL-099 LLM-repair retry.
            log.warning(
                "[LLMDirector] BUG-099 LLM-repair pass: original parse "
                "failed (%d chars), retrying with strict-JSON prompt to "
                "recover content...",
                len(parse_exc.raw_output),
            )
            _runtime_log(
                "DIRECTOR: BUG-099 JSON-repair retry (preserve content)"
            )
            try:
                repair_prompt = _build_director_json_repair_prompt(
                    parse_exc.raw_output, script_text
                )
                # Use cleanup model when available -- typically faster
                # / smaller than the story model and well-suited for
                # structural reformatting tasks.
                _repair_model = (
                    _effective_cleanup_id
                    if _effective_cleanup_id
                    else model_id
                )
                repaired_raw = _generate_with_llm(
                    repair_prompt,
                    model_id=_repair_model,
                    max_new_tokens=2000,
                    temperature=0.1,  # very low for JSON fidelity
                    optimization_profile=optimization_profile,
                )
                plan = self._extract_json(repaired_raw)
                log.info(
                    "[LLMDirector] BUG-099 repair pass succeeded: "
                    "%d voice_assignments, %d sfx, %d music",
                    len(plan.get("voice_assignments") or {}),
                    len(plan.get("sfx_plan") or []),
                    len(plan.get("music_plan") or []),
                )
                _runtime_log(
                    "DIRECTOR: BUG-099 LLM-repair pass recovered content"
                )
            except Exception as repair_exc:
                log.warning(
                    "[LLMDirector] BUG-099 repair pass also failed (%s); "
                    "falling through to BUG-090 minimal fallback",
                    repair_exc,
                )
                _runtime_log(
                    f"DIRECTOR: BUG-099 repair failed ({type(repair_exc).__name__}) "
                    "-- BUG-090 minimal fallback active"
                )
                plan = None

        if plan is None:
            # Tier 3: BUG-LOCAL-090 minimal fallback.
            log.warning(
                "[LLMDirector] BUG-090 minimal fallback: building plan "
                "from script VOICE tags only (sfx_plan + music_plan "
                "will be empty -- AudioGen + MusicGen will use defaults)"
            )
            _runtime_log(
                "DIRECTOR: BUG-090 minimal fallback active "
                "(see director_raw_*.txt for raw LLM output)"
            )
            plan = _build_fallback_director_plan(script_text)
            log.warning(
                "[LLMDirector] fallback plan: %d voice_assignments, "
                "0 sfx, 0 music",
                len(plan.get("voice_assignments", {})),
            )

        plan = self._validate_director_plan(plan)

        # Procedural character names (except LEMMY stays LEMMY)
        # Use a deterministic seed based on script content hash
        if plan:
            import hashlib
            script_hash = hashlib.sha256(script_text.encode()).hexdigest()[:16]

            # BUG-004 fix: extract gender from each [VOICE: NAME, gender, ...] tag
            # so the procedural cast generator never assigns a male voice to a
            # female character or vice versa. First gender hint per name wins.
            #
            # BUG-LOCAL-060 (2026-04-24): previous implementation only scanned
            # for the `[VOICE: NAME, gender, ...]` form. Mistral Nemo emits the
            # shorthand `[NAME, gender, age, mood]` inline with dialogue (e.g.
            # `[EDNA, Female, 40s, urgent] Dammit!`), which produced 0 hits and
            # let the procedural fallback assign wrong-gender voices (EDNA=male,
            # BOB=female). Now we scan THREE sources in priority order:
            #   1. `[VOICE: NAME, gender, ...]` canonical tags
            #   2. `[NAME, gender, ...]` shorthand tags (Mistral Nemo form)
            #   3. Director's own voice_assignments[NAME].notes field
            # First hit per character name wins.
            gender_map = {}
            voice_tag_re = re.compile(
                r"\[VOICE:\s*([A-Z][A-Z0-9_ ]*?)\s*,\s*(male|female)\b",
                re.IGNORECASE,
            )
            for m in voice_tag_re.finditer(script_text):
                name_key = m.group(1).strip().upper()
                gender_val = m.group(2).strip().lower()
                gender_map.setdefault(name_key, gender_val)
            hits_from_voice_tags = len(gender_map)

            # Source 2: shorthand `[NAME, gender, ...]` tags. Skip structural
            # tokens that look like character names in bracket position.
            _SHORTHAND_STRUCTURAL = {
                "ENV", "SFX", "MUSIC", "BEAT", "PAUSE", "ACT", "SCENE",
                "TRANSITION", "CONTINUED", "CONT", "END", "FADE", "CUT",
                "INT", "EXT", "OPENING", "CLOSING", "INTERSTITIAL",
                "TITLE", "NOTE", "TARGET", "STYLE", "NARRATOR",
                "SYSTEM_SENTINEL", "VOICE",
            }
            shorthand_gender_re = re.compile(
                r"\[([A-Z][A-Z0-9_ ]{1,20})\s*,\s*(male|female)\b",
                re.IGNORECASE,
            )
            for m in shorthand_gender_re.finditer(script_text):
                name_key = m.group(1).strip().upper()
                first_word = name_key.split()[0] if name_key else ""
                if first_word in _SHORTHAND_STRUCTURAL:
                    continue
                gender_val = m.group(2).strip().lower()
                gender_map.setdefault(name_key, gender_val)
            hits_from_shorthand = len(gender_map) - hits_from_voice_tags

            # Source 3: Director's voice_assignments.notes field. The Director
            # already writes "Female, 40s, urgent" in the notes for each
            # character; reuse that as a backstop so a correctly-structured
            # Director plan always wins over procedural rotation.
            va = plan.get("voice_assignments") if plan else None
            if isinstance(va, dict):
                for char_name, entry in va.items():
                    key = (char_name or "").strip().upper()
                    if not key or key in gender_map:
                        continue
                    notes = ""
                    if isinstance(entry, dict):
                        notes = str(entry.get("notes") or "")
                    lo = notes.lower()
                    # Match whole words only to avoid "female" matching inside
                    # "remale" or similar. The notes field is free-form but
                    # Mistral Nemo always emits one of these two tokens first.
                    if re.search(r"\bfemale\b", lo):
                        gender_map[key] = "female"
                    elif re.search(r"\bmale\b", lo):
                        gender_map[key] = "male"
            hits_from_director_notes = len(gender_map) - hits_from_voice_tags - hits_from_shorthand

            log.info(
                "[LLMDirector] Parsed %d gender hints (voice_tags=%d, shorthand=%d, "
                "director_notes=%d): %s",
                len(gender_map),
                hits_from_voice_tags,
                hits_from_shorthand,
                hits_from_director_notes,
                gender_map,
            )

            plan = self._randomize_character_names(plan, script_hash, gender_map=gender_map)

        # Override vintage settings with user's intensity choice
        if plan:
            plan["vintage_settings"] = vintage_map.get(vintage_intensity, vintage_map["moderate"])

        plan_json = json.dumps(plan, indent=2)
        voice_json = json.dumps(plan.get("voice_assignments", {}), indent=2)
        sfx_json = json.dumps(plan.get("sfx_plan", []), indent=2)
        music_json = json.dumps(plan.get("music_plan", []), indent=2)

        log.info(f"[LLMDirector] Plan: {len(plan.get('voice_assignments', {}))} voices, "
                 f"{len(plan.get('sfx_plan', []))} SFX cues, "
                 f"{len(plan.get('music_plan', []))} music cues")

        # BUG-012 FIX: Explicitly unload Gemma from VRAM at the end of the director phase.
        # Otherwise it stays resident during the audio generation phases, causing VRAM OOM
        # or massive PCIe swapping on 4GB hardware.
        _unload_llm()
        _runtime_log("Director: Gemma unloaded - VRAM freed for Audio/TTS")

        # v1.4 Theme C - director exit snapshot.
        vram_snapshot("director_exit")

        # ------------------------------------------------------------------
        # Production Ledger (L1) -- Director stage populates:
        #   - cast voice_preset + gender (overwrites ScriptWriter's bare
        #     cast rows with voice_assignments data)
        #   - scenes env (from visual_plan.scenes description)
        #   - shots (one per visual_plan.scenes entry, PNG path filled later)
        #   - sfx cue list (metadata only, wav_path + timing come from
        #     BatchAudioGen + SceneSequencer)
        #   - music cue list (same)
        # ------------------------------------------------------------------
        try:
            from .production_ledger import get_ledger
            led = get_ledger()
            # Preserve the ScriptWriter-stage cast order but enrich with
            # voice_preset + gender from the Director's voice_assignments.
            #
            # 2026-04-26 PM BUG-LOCAL-068: ScriptWriter cast rows use the
            # script-side spaced name ("KAEL VAUGHN") while the Director's
            # voice_assignments dict keys carry the underscored variant
            # ("KAEL_VAUGHN"). A naive equality lookup miss caused the
            # loop to APPEND a duplicate underscored row for every Director-
            # known character, leaving voice_preset on the new row and
            # dialogue on the original row. Net effect: Bark fell back to
            # default voices because the speaking characters had voice=None.
            # Normalize both sides ("KAEL_VAUGHN" -> "KAEL VAUGHN") before
            # the lookup so the same character merges into a single row,
            # and persist the script-side spaced name as the canonical form.
            def _norm_name_key(s):
                return (s or "").strip().upper().replace("_", " ")

            existing_cast = list(led.data.get("cast") or [])
            name_to_row = {_norm_name_key(r.get("name")): r
                           for r in existing_cast}
            va = plan.get("voice_assignments", {}) or {}
            for char_name, entry in va.items():
                if not isinstance(entry, dict):
                    continue
                key = _norm_name_key(char_name)
                raw_notes = str(entry.get("notes") or "")
                notes_lo = raw_notes.lower()
                gender = None
                if re.search(r"\bfemale\b", notes_lo):
                    gender = "female"
                elif re.search(r"\bmale\b", notes_lo):
                    gender = "male"
                row = name_to_row.get(key)
                if row is None:
                    # New row -- use the spaced (canonical) form so the
                    # ledger doesn't carry mixed underscored / spaced
                    # entries that would re-trigger this same bug for
                    # any later consumer.
                    row = {"char_id": f"c{len(existing_cast)+1:02d}",
                           "name": key}
                    existing_cast.append(row)
                    name_to_row[key] = row
                row["voice_preset"] = entry.get("voice_preset") or row.get("voice_preset")
                if gender:
                    row["gender"] = gender
                # Director's voice_assignments.notes carries the free-form
                # character description (e.g. "NEMO BOUVIER - male, 40s,
                # stoic" after procedural rename, or "Female, 40s, urgent"
                # directly from Mistral). Use as the canonical description.
                if raw_notes:
                    row["description"] = raw_notes
            # ----------------------------------------------------------
            # 2026-04-26 PM BUG-LOCAL-068 EXPANSION: fuzzy-merge pass
            # ----------------------------------------------------------
            # The original BUG-068 fix only handled space/underscore
            # variants ("KAEL VAUGHN" vs "KAEL_VAUGHN"). Two new failure
            # shapes were discovered the same evening:
            #
            #   1. Prefix-overlap (two-LLM split, Captain-Eris -> Mistral):
            #      Captain-Eris uses first names in dialogue tags
            #      ("[LLOYD]") while Mistral cleanup adds full names in
            #      character descriptions ("LLOYD KAPOOR"). Naive equality
            #      treats them as separate characters; the speaker rows
            #      get dialogue and no voice; the description rows get
            #      voice and no dialogue.
            #
            #   2. Typo-divergence (single-LLM Mistral, observed live):
            #      Mistral mid-stream typoed "[STANLEY]" as "[STANLEARY]"
            #      and alternated between the two for the rest of the
            #      script. Same cast-fragmentation outcome.
            #
            # Both shapes have a shared signature: TWO cast rows, one
            # with the voice_preset and one with the dialogue line_count.
            # difflib.SequenceMatcher.ratio() catches both (LLOYD vs
            # LLOYD KAPOOR ratio=0.78 with prefix-rule; STANLEY vs
            # STANLEARY ratio=0.875). We merge with a clear winner-rule
            # so downstream Bark voicing lands on the speaking row.
            existing_cast, _cast_aliases = _consolidate_similar_cast_rows_with_aliases(existing_cast)
            # BUG-LOCAL-098: rewrite lines[i].char_id (and any other
            # downstream char_id refs) to point from a dropped loser
            # char_id to the surviving winner. Without this, the cast
            # is clean but lines still reference dangling ids ->
            # Bark fails voice resolution for those lines. Observed
            # on the Arcadia run with the ANNOUCNER/ANNOUNCER typo
            # merge dropping c02 from cast while l001 + l033 still
            # pointed at c02.
            if _cast_aliases:
                _rewritten = 0
                try:
                    led_data = led.data
                    for ln in led_data.get("lines", []) or []:
                        cid = ln.get("char_id")
                        if cid and cid in _cast_aliases:
                            ln["char_id"] = _cast_aliases[cid]
                            _rewritten += 1
                    # SFX rows can carry a char_id too in some workflows;
                    # be defensive.
                    for sx in led_data.get("sfx", []) or []:
                        cid = sx.get("char_id")
                        if cid and cid in _cast_aliases:
                            sx["char_id"] = _cast_aliases[cid]
                            _rewritten += 1
                    log.info(
                        "[LLMDirector] BUG-098 cast-merge alias rewrite: "
                        "%d row(s) updated, aliases=%s",
                        _rewritten, _cast_aliases,
                    )
                    _runtime_log(
                        f"DIRECTOR: BUG-098 cast-alias rewrite "
                        f"({_rewritten} ledger refs updated; "
                        f"aliases={_cast_aliases})"
                    )
                except Exception as _exc:
                    log.warning(
                        "[LLMDirector] BUG-098 alias rewrite failed: %s", _exc
                    )
            # 2026-04-26 PM ARCH: lazy Bark health check on the cast's
            # actual Bark presets only -- replaces the eager full-catalog
            # warmup that used to run at the top of direct(). Failed
            # presets are remapped in-place to known-good fallbacks of
            # the same gender from `_VOICE_PROFILES`.
            try:
                existing_cast = _bark_health_check_for_cast(existing_cast)
            except Exception as e:
                _runtime_log(f"VOICE_HEALTH_LAZY_SKIPPED: unexpected error {e}")
            led.set_cast(existing_cast)

            # Scenes + shots: visual_plan.scenes carries per-scene
            # environment + one visual prompt per scene. Each scene maps to
            # one shot row in L1; PASS3's multi-shot expansion lives in L2.
            vp_scenes = (plan.get("visual_plan", {}) or {}).get("scenes", []) or []
            scene_rows = []
            shot_rows = []
            for idx, sc in enumerate(vp_scenes):
                if not isinstance(sc, dict):
                    continue
                sid = str(sc.get("scene_id") or f"scene_{idx+1}").strip()
                # shot_description is the short human-readable title,
                # visual_prompt is the detailed FLUX prompt. Keep them
                # separated so the viewer can show both.
                short_desc = str(sc.get("shot_description") or "")
                detailed_prompt = str(sc.get("visual_prompt") or "")
                scene_rows.append({
                    "scene_id":    sid,
                    "description": short_desc,
                })
                shot_rows.append({
                    "shot_id":       f"sh{idx+1:02d}",
                    "scene_id":      sid,
                    "description":   short_desc,
                    "visual_prompt": detailed_prompt,
                })
            if scene_rows:
                led.set_scenes(scene_rows)
            if shot_rows:
                led.set_shots(shot_rows)

            # SFX + music: copy metadata rows (description + generation_prompt
            # for reference). wav_path + start_s + dur_s get filled by the
            # audio generators + SceneSequencer later.
            sfx_rows = []
            for entry in plan.get("sfx_plan", []) or []:
                if not isinstance(entry, dict):
                    continue
                sfx_rows.append({
                    "cue_id":            str(entry.get("cue_id") or ""),
                    "description":       str(entry.get("description") or ""),
                    "generation_prompt": str(entry.get("generation_prompt") or ""),
                })
            music_rows = []
            for entry in plan.get("music_plan", []) or []:
                if not isinstance(entry, dict):
                    continue
                gen_prompt = str(entry.get("generation_prompt") or "")
                # Music entries often have only generation_prompt, no separate
                # description. Derive a short description from the first
                # comma-delimited phrase so the viewer has something readable.
                short_desc = gen_prompt.split(",", 1)[0].strip() if gen_prompt else ""
                music_rows.append({
                    "cue_id":            str(entry.get("cue_id") or ""),
                    "description":       short_desc,
                    "generation_prompt": gen_prompt,
                })
            if sfx_rows:
                led.set_sfx(sfx_rows)
            if music_rows:
                led.set_music(music_rows)

            led.save()
        except Exception as _e:  # noqa: BLE001
            log.warning("[Ledger] Director-stage snapshot failed: %s", _e)

        return (plan_json, voice_json, sfx_json, music_json)

    @staticmethod
    def _strip_json_comments(text):
        """Remove JS-style // line comments from JSON text.

        LLMs (especially Mistral) sprinkle '// explanation' inside JSON
        values. json.loads() rejects these. We strip them only outside of
        quoted strings to avoid mangling URLs like 'v2/en_speaker_8'.
        """
        result = []
        i = 0
        in_string = False
        while i < len(text):
            c = text[i]
            if in_string:
                result.append(c)
                if c == '\\' and i + 1 < len(text):
                    result.append(text[i + 1])
                    i += 2
                    continue
                if c == '"':
                    in_string = False
            else:
                if c == '"':
                    in_string = True
                    result.append(c)
                elif c == '/' and i + 1 < len(text) and text[i + 1] == '/':
                    # skip to end of line
                    while i < len(text) and text[i] != '\n':
                        i += 1
                    continue
                else:
                    result.append(c)
            i += 1
        return ''.join(result)

    def _extract_json(self, text):
        """Extract JSON object from LLM output (handles markdown fences, truncation)."""
        log.info(f"[LLMDirector] Raw output length: {len(text)} chars")
        log.info(f"[LLMDirector] Raw output preview: {text[:200]}...")

        # Try to find JSON block
        patterns = [
            re.compile(r'```json\s*\n(.*?)\n```', re.DOTALL),
            re.compile(r'```\s*\n(.*?)\n```', re.DOTALL),
            re.compile(r'(\{.*\})', re.DOTALL),
        ]
        for pat in patterns:
            m = pat.search(text)
            if m:
                raw_match = m.group(1)
                try:
                    return json.loads(raw_match)
                except json.JSONDecodeError:
                    pass
                # BUG-LOCAL-040: strip JS-style // comments LLMs inject,
                # then strip trailing commas before close-brackets.
                candidate = self._strip_json_comments(raw_match)
                candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass
                # If JSON was truncated, try closing it
                open_braces = candidate.count('{') - candidate.count('}')
                open_brackets = candidate.count('[') - candidate.count(']')
                if open_braces > 0 or open_brackets > 0:
                    log.info(f"[LLMDirector] Attempting JSON repair: +{open_braces} braces, +{open_brackets} brackets")
                    repaired = candidate + ']' * open_brackets + '}' * open_braces
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError as e:
                        log.warning(f"[LLMDirector] JSON repair failed: {e}")
                continue

        # Last resort: find the first { and try to build valid JSON from there
        brace_start = text.find('{')
        if brace_start >= 0:
            # BUG-LOCAL-040: strip comments before brace scan too
            candidate = self._strip_json_comments(text[brace_start:])
            candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
            # Try progressively shorter substrings
            for end_offset in range(len(candidate), max(0, len(candidate) - 200), -1):
                try:
                    return json.loads(candidate[:end_offset])
                except json.JSONDecodeError:
                    continue

        # BUG-LOCAL-090: when every parse strategy fails, save the raw
        # LLM output to a debug file for offline inspection and raise
        # a structured exception that the caller can catch to build a
        # minimal fallback plan from the script's VOICE tags. We no
        # longer abort the run unconditionally -- aborting on a
        # parser glitch loses an hour-plus of upstream LLM/audio
        # work; better to ship with a default voice mapping and let
        # downstream phases use the actual script content.
        try:
            import os, time as _time
            dump_dir = director_raw_dump_dir()
            dump_dir.mkdir(parents=True, exist_ok=True)
            dump_path = dump_dir / f"director_raw_{int(_time.time())}.txt"
            dump_path.write_text(text or "", encoding="utf-8")
            log.critical(
                "[LLMDirector] FATAL: Could not parse JSON. "
                "Raw output saved to %s",
                dump_path,
            )
        except Exception as _dump_exc:
            log.warning(
                "[LLMDirector] could not persist raw output: %s",
                _dump_exc,
            )
        log.critical(
            f"[LLMDirector] Raw output preview:\n{(text or '')[:1000]}"
        )
        raise DirectorJSONParseError(
            "Failed to parse production plan JSON",
            raw_output=text or "",
        )

    def _randomize_character_names(self, plan: dict, episode_seed: str,
                                   gender_map: dict = None) -> dict:
        """Replace ALL character traits with procedural profiles. LEMMY stays LEMMY.

        For each character in voice_assignments:
          - LEMMY: Fixed profile (gravelly male, en_speaker_8)
          - ANNOUNCER: Random voice from balanced announcer pool
          - Everyone else: Full procedural profile - name, gender, age,
            demeanor, and best-fit en_speaker_* preset, all derived
            deterministically from the episode seed.

        The Director's original name/voice picks are fully overridden.
        English-only presets enforced at generation time.

        Args:
            plan: The parsed production plan dict
            episode_seed: A deterministic seed (script hash) for reproducibility

        Returns:
            Modified plan with procedural names, traits, and voice presets
        """
        if not plan or "voice_assignments" not in plan:
            return plan

        voice_assignments = plan.get("voice_assignments", {})
        if not voice_assignments:
            return plan

        # Track assigned presets to avoid duplicates across cast
        used_presets = set()
        new_voice_assignments = {}
        character_idx = 0

        # FIX: Process LEMMY and ANNOUNCER FIRST so their locked presets
        # are reserved in used_presets before regular characters draw from
        # the pool. Otherwise a regular char can grab v2/en_speaker_8
        # before Lemmy's branch runs, causing voice collision (Lemmy=Drake).
        all_keys = list(voice_assignments.keys())
        priority_keys = [k for k in all_keys if k.upper().strip() in ("LEMMY", "ANNOUNCER")]
        regular_keys  = [k for k in all_keys if k.upper().strip() not in ("LEMMY", "ANNOUNCER")]
        ordered_keys  = priority_keys + regular_keys

        for old_name in ordered_keys:
            upper_name = old_name.upper().strip()

            if upper_name == "LEMMY":
                # LEMMY - fixed iconic profile, never changes
                profile = _LEMMY_PROFILE.copy()
                new_voice_assignments["LEMMY"] = {
                    "voice_preset": profile["voice_preset"],
                    "notes": profile["notes"],
                }
                used_presets.add(profile["voice_preset"])
                log.info("[LLMDirector] LEMMY: locked - %s (%s)",
                         profile["voice_preset"], profile["notes"])

            elif upper_name == "ANNOUNCER":
                # ANNOUNCER - random from balanced pool, seeded per episode.
                # Respects gender_hint from script [VOICE: ANNOUNCER, gender, ...] tag.
                ann_gender = gender_map.get("ANNOUNCER") if gender_map else None
                ann = _generate_announcer_profile(episode_seed, gender_hint=ann_gender)
                new_voice_assignments["ANNOUNCER"] = {
                    "voice_preset": ann["voice_preset"],
                    "notes": ann["notes"],
                }
                used_presets.add(ann["voice_preset"])
                log.info("[LLMDirector] ANNOUNCER: procedural - %s (%s) [gender_hint=%s]",
                         ann["voice_preset"], ann["notes"], ann_gender or "none")

            else:
                # Regular character - full procedural profile.
                # BUG-004 fix: pull gender_hint from the script's [VOICE: NAME, gender, ...]
                # tag so we never assign a male voice to a female character (or vice versa).
                gender_hint = None
                if gender_map:
                    gender_hint = gender_map.get(upper_name)
                profile = _generate_character_profile(
                    character_idx, episode_seed, gender_hint=gender_hint
                )

                # De-duplicate voice presets: if this preset is already taken,
                # re-roll with offset seeds until we find an unused one in the
                # SAME gender pool (soft constraint - if pool exhausted, log and
                # accept duplicate).
                attempts = 0
                while profile["voice_preset"] in used_presets and attempts < 10:
                    attempts += 1
                    profile = _generate_character_profile(
                        character_idx + attempts * 100, episode_seed,
                        gender_hint=gender_hint,
                    )
                if profile["voice_preset"] in used_presets:
                    log.warning(
                        "[LLMDirector] CAST_GENDER_POOL_EXHAUSTED: %s (%s) "
                        "reusing preset %s - increase pool or accept duplicate",
                        upper_name, gender_hint or "unknown", profile["voice_preset"]
                    )

                used_presets.add(profile["voice_preset"])
                # FIX (v1.1): Use the ORIGINAL script name as the dict key so
                # BatchBark can match [VOICE: NAME ...] to the right preset.
                # FIX (v2.0-alpha 2026-04-26): Do NOT prefix notes with
                # profile['name']. The procedural name (e.g. "PRIAM SMITHERS")
                # disagrees with the script name (e.g. "MEREDITH") and
                # downstream cast.description ends up showing the wrong name.
                # The procedural name remains available on profile['name']
                # if a treatment-file consumer needs it. Match the LEMMY /
                # ANNOUNCER branches which pass bare notes.
                new_voice_assignments[upper_name] = {
                    "voice_preset": profile["voice_preset"],
                    "notes": profile["notes"],
                }
                log.info("[LLMDirector] %s - voice: %s (profile: %s, %s, %s, %s)",
                         upper_name, profile["voice_preset"],
                         profile["name"], profile["gender"], profile["age"], profile["demeanor"])
                # BUG-004 telemetry - grep CAST_GENDER_MATCH to verify per-character matching
                _runtime_log(
                    f"CAST_GENDER_MATCH {upper_name}={gender_hint or 'unspecified'} "
                    f"- {profile['voice_preset']} ({profile['gender']})"
                )
                character_idx += 1

        plan["voice_assignments"] = new_voice_assignments

        log.info("[LLMDirector] Procedural cast complete: %d characters "
                 "(%d unique presets)", len(new_voice_assignments), len(used_presets))

        return plan

    def _validate_director_plan(self, plan):
        """Validate and repair the Director's JSON production plan.

        Checks all required keys exist and have correct types. Repairs missing
        or invalid entries for voice assignments, SFX cues, and music cues.
        Adds fallback defaults for missing critical fields.

        Args:
            plan: The parsed production plan dict from _extract_json()

        Returns:
            The repaired plan dict, guaranteed to have valid structure downstream.
        """
        if not isinstance(plan, dict):
            log.warning("[LLMDirector] DIRECTOR_SCHEMA: plan is not a dict, initializing empty")
            _runtime_log("DIRECTOR_SCHEMA: plan is not a dict, initializing empty")
            plan = {}

        # --- Part 1: Check and add required keys with defaults ---
        for key, expected_type in _DIRECTOR_SCHEMA["required_keys"].items():
            if key not in plan:
                log.warning(f"[LLMDirector] DIRECTOR_SCHEMA: missing required key '{key}', adding default")
                _runtime_log(f"DIRECTOR_SCHEMA: repaired missing required key '{key}'")
                if expected_type == dict:
                    plan[key] = {}
                elif expected_type == list:
                    plan[key] = []
            elif not isinstance(plan[key], expected_type):
                log.warning(f"[LLMDirector] DIRECTOR_SCHEMA: '{key}' has type {type(plan[key]).__name__}, expected {expected_type.__name__}, resetting to default")
                _runtime_log(f"DIRECTOR_SCHEMA: repaired type mismatch on '{key}'")
                if expected_type == dict:
                    plan[key] = {}
                elif expected_type == list:
                    plan[key] = []

        # --- Part 2: Validate voice_assignments ---
        voice_assignments = plan.get("voice_assignments", {})
        if isinstance(voice_assignments, dict):
            for char_name, char_data in list(voice_assignments.items()):
                if not isinstance(char_data, dict):
                    log.warning(f"[LLMDirector] DIRECTOR_SCHEMA: voice_assignments[{char_name}] is not a dict, replacing")
                    _runtime_log(f"DIRECTOR_SCHEMA: repaired voice_assignments[{char_name}] type")
                    voice_assignments[char_name] = {}
                    char_data = voice_assignments[char_name]

                # Check voice_preset exists and is a string
                if "voice_preset" not in char_data:
                    fallback = "v2/en_speaker_0"
                    log.warning(f"[LLMDirector] DIRECTOR_SCHEMA: voice_assignments[{char_name}] missing 'voice_preset', using fallback '{fallback}'")
                    _runtime_log(f"DIRECTOR_SCHEMA: repaired voice_preset for {char_name}")
                    char_data["voice_preset"] = fallback
                elif not isinstance(char_data["voice_preset"], str):
                    fallback = "v2/en_speaker_0"
                    log.warning(f"[LLMDirector] DIRECTOR_SCHEMA: voice_assignments[{char_name}]['voice_preset'] is not a string, using fallback '{fallback}'")
                    _runtime_log(f"DIRECTOR_SCHEMA: repaired voice_preset type for {char_name}")
                    char_data["voice_preset"] = fallback

        # --- Part 3: Validate SFX plan ---
        sfx_plan = plan.get("sfx_plan", [])
        if not isinstance(sfx_plan, list):
            log.warning(f"[LLMDirector] DIRECTOR_SCHEMA: sfx_plan is not a list, resetting")
            _runtime_log("DIRECTOR_SCHEMA: repaired sfx_plan type")
            plan["sfx_plan"] = []
            sfx_plan = []
        else:
            # Remove SFX entries missing generation_prompt (they're useless)
            valid_sfx = []
            for i, sfx_entry in enumerate(sfx_plan):
                if not isinstance(sfx_entry, dict):
                    log.warning(f"[LLMDirector] DIRECTOR_SCHEMA: sfx_plan[{i}] is not a dict, skipping")
                    _runtime_log(f"DIRECTOR_SCHEMA: removed invalid sfx_plan[{i}]")
                    continue

                # Check required fields
                if "cue_id" not in sfx_entry or "generation_prompt" not in sfx_entry:
                    log.warning(f"[LLMDirector] DIRECTOR_SCHEMA: sfx_plan[{i}] missing required fields, skipping")
                    _runtime_log(f"DIRECTOR_SCHEMA: removed incomplete sfx_plan[{i}]")
                    continue

                valid_sfx.append(sfx_entry)

            plan["sfx_plan"] = valid_sfx

        # --- Part 4: Validate and synthesize music_plan ---
        music_plan = plan.get("music_plan", [])
        if not isinstance(music_plan, list):
            log.warning(f"[LLMDirector] DIRECTOR_SCHEMA: music_plan is not a list, resetting")
            _runtime_log("DIRECTOR_SCHEMA: repaired music_plan type")
            plan["music_plan"] = []
            music_plan = []

        # Check that all required cue_ids exist
        music_dict = {entry.get("cue_id"): entry for entry in music_plan
                      if isinstance(entry, dict) and "cue_id" in entry}

        # Default music cue configs
        music_defaults = {
            "opening": {
                "cue_id": "opening",
                "duration_sec": 12,
                "generation_prompt": "1940s old time radio opening theme, warm brass fanfare, upright bass, snare brushes, mono AM radio character, tube saturation, confident and mysterious, ends on a held chord"
            },
            "closing": {
                "cue_id": "closing",
                "duration_sec": 8,
                "generation_prompt": "1940s old time radio closing sting, brass and strings, resolving cadence, warm tube saturation, fades to silence"
            },
            "interstitial": {
                "cue_id": "interstitial",
                "duration_sec": 4,
                "generation_prompt": "short old time radio act-break stinger, single brass hit with cymbal swell, mono, tube warmth"
            }
        }

        # Ensure all three required cues exist
        for cue_id in _DIRECTOR_SCHEMA["music_cue_ids"]:
            if cue_id not in music_dict:
                log.warning(f"[LLMDirector] DIRECTOR_SCHEMA: music_plan missing cue_id '{cue_id}', synthesizing default")
                _runtime_log(f"DIRECTOR_SCHEMA: repaired missing music cue '{cue_id}'")
                plan["music_plan"].append(music_defaults[cue_id].copy())
            else:
                # Validate existing cue
                cue_entry = music_dict[cue_id]
                # Check duration_sec is a valid positive number
                duration = cue_entry.get("duration_sec")
                if not isinstance(duration, (int, float)) or duration <= 0:
                    default_duration = music_defaults[cue_id]["duration_sec"]
                    log.warning(f"[LLMDirector] DIRECTOR_SCHEMA: music cue '{cue_id}' has invalid duration_sec={duration}, using default {default_duration}")
                    _runtime_log(f"DIRECTOR_SCHEMA: repaired invalid duration for music cue '{cue_id}'")
                    cue_entry["duration_sec"] = default_duration

                # Check generation_prompt exists and is a string
                if "generation_prompt" not in cue_entry or not isinstance(cue_entry.get("generation_prompt"), str):
                    default_prompt = music_defaults[cue_id]["generation_prompt"]
                    log.warning(f"[LLMDirector] DIRECTOR_SCHEMA: music cue '{cue_id}' has invalid generation_prompt, using default")
                    _runtime_log(f"DIRECTOR_SCHEMA: repaired invalid prompt for music cue '{cue_id}'")
                    cue_entry["generation_prompt"] = default_prompt

        log.info(f"[LLMDirector] DIRECTOR_SCHEMA validation complete: "
                 f"voice_assignments={len(plan.get('voice_assignments', {}))}, "
                 f"sfx_plan={len(plan.get('sfx_plan', []))}, "
                 f"music_plan={len(plan.get('music_plan', []))}")

        return plan
