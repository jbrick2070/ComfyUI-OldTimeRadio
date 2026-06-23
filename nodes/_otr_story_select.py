"""nodes/_otr_story_select.py -- Best-of-N structural story-refine selector.

Local-only by DEFAULT, opt-in remote, DETERMINISTIC best-of-N OUTLINE selector
(2026-06-23, 4-round roundtable-converged). NOT a QA-reroll gate: candidates are
FRESH-GENERATED outline structures and the keep-best gate is a PURE deterministic
scorer -- never "ask the same model to try again on the same beats".

This module hosts:
  * StoryScore / score_outline  -- the pure structural scorer (chunk 2).
  * select_best_outline + resolve_best_of_n  -- the cast_seed-keyed selector,
    flag parse, and provider gate (chunk 3); optional remote + cost guard
    (chunk 4).

The scorer runs on the RAW beat intents BEFORE any grounding: build_sq_data
MUTATES intent and substitutes the generic crisis nouns, which would zero out
ungrounded_crisis_density (the roundtable R3 catch). build_sq_data still runs
exactly ONCE downstream on the winning outline -- never here.

Dependency note: this imports only the stdlib-leaf _otr_story_quality_l12 public
helpers at module load (no torch, no _otr_outline cycle). torch is imported
LOCALLY inside select_best_outline (the writer forbids module-level torch).

UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, List

# Public L1/L2 helpers. _otr_story_quality_l12 is a stdlib-only leaf
# (hashlib/re/unicodedata) that never imports _otr_outline, so a module-level
# import here forms no cycle and pulls no heavy deps. Package import in
# production; flat import when loaded standalone / under test.
try:
    from ._otr_story_quality_l12 import (
        count_ungrounded_crisis,
        premise_noun_palette,
        premise_texts,
    )
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_story_quality_l12 import (  # type: ignore
        count_ungrounded_crisis,
        premise_noun_palette,
        premise_texts,
    )

log = logging.getLogger("OTR")

# Token rule MIRRORS _otr_story_quality_l12._TOKEN_RE so the scorer tokenizes
# beat intents identically to count_ungrounded_crisis / premise_noun_palette
# (that symbol is module-private there; re-declared here, not imported, so the
# scorer never depends on a private name).
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")

# Voiced == character + announcer, matching _otr_story_quality_l12._is_voiced
# -- the exact scope build_sq_data grounds over. Announcer bookends are voiced
# (Kokoro renders them). Kept local so the scorer never reaches a private name.
_VOICED_ROLES = ("character", "announcer")


def _is_voiced(role: str) -> bool:
    return role in _VOICED_ROLES


# ---------------------------------------------------------------------------
# Chunk 2 -- pure structural scorer
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StoryScore:
    """Pure, deterministic structural score for ONE candidate outline.

    Lower ``ungrounded_crisis_density`` is better; higher
    ``distinct_conflict_nouns`` and ``premise_grounding`` are better. Computed
    on the RAW beat intents BEFORE grounding. ``character_want_clarity`` and
    ``winner_grade`` were CUT from v0 (no wants data at this stage; the grade
    is unused by the comparator)."""

    ungrounded_crisis_density: float
    distinct_conflict_nouns: int
    premise_grounding: float


def score_outline(outline: Any, meta: Any, roster: Any) -> StoryScore:
    """Score a candidate ``outline`` structurally. PURE -- never mutates the
    outline / beats, never calls build_sq_data.

    Metrics (all over VOICED beats = character + announcer):
      * ``ungrounded_crisis_density`` = (sum of count_ungrounded_crisis(
        beat.intent, grounded)) / max(1, total voiced-intent tokens). The
        cross-episode "console standoff" sameness signal -- lower is better.
      * ``distinct_conflict_nouns`` = number of DISTINCT premise-grounded
        content tokens surfaced across the voiced beat intents -- higher better.
      * ``premise_grounding`` = fraction of voiced beats whose intent references
        at least one premise/roster noun -- higher better.

    ``grounded`` palette = premise_noun_palette(roster, premise,
    *premise_texts(meta)) -- identical to build_sq_data's grounding source.
    """
    premise = str(getattr(outline, "premise", "") or "")
    grounded = premise_noun_palette(roster, premise, *premise_texts(meta))

    voiced_intents: List[str] = [
        str(getattr(b, "intent", "") or "")
        for b in (getattr(outline, "beats", None) or [])
        if _is_voiced(str(getattr(b, "speaker_role", "") or ""))
    ]

    total_voiced_beats = len(voiced_intents)
    total_voiced_intent_words = 0
    ungrounded_total = 0
    distinct_grounded: set = set()
    referencing_beats = 0

    for intent in voiced_intents:
        toks = _TOKEN_RE.findall(intent)
        total_voiced_intent_words += len(toks)
        ungrounded_total += count_ungrounded_crisis(intent, grounded)
        beat_refs = False
        for tok in toks:
            low = tok.casefold()
            if low in grounded:
                distinct_grounded.add(low)
                beat_refs = True
        if beat_refs:
            referencing_beats += 1

    density = ungrounded_total / max(1, total_voiced_intent_words)
    grounding = referencing_beats / max(1, total_voiced_beats)
    return StoryScore(
        ungrounded_crisis_density=density,
        distinct_conflict_nouns=len(distinct_grounded),
        premise_grounding=grounding,
    )


# ---------------------------------------------------------------------------
# Chunk 3 -- flag parse + provider gate + the selector
# ---------------------------------------------------------------------------
# Local writers run free (no paid call), so a generous cap. The tighter remote
# cap (REMOTE_BEST_OF_N_MAX = 3) + the fail-closed cost guard live in chunk 4.
_LOCAL_BEST_OF_N_MAX = 6

# Deterministic, index-keyed STRUCTURAL-variation instructions (candidate i>=1).
# Each pushes the outline away from the "console standoff" sameness toward a
# different dramatic spine. Candidate 0 always uses "" (byte-identical prompt).
_DIVERSITY_HINTS = (
    "open on the personal stake of one character, not the institutional threat "
    "or the wider crisis",
    "make the central conflict interpersonal -- two people who want incompatible "
    "things -- rather than a race against a system or a countdown",
    "let the turning point be a character's private choice or admission, and keep "
    "the decisive moment on-stage between the characters",
    "ground every beat in the specific premise nouns (the actual contested thing) "
    "and avoid generic control-room hardware as the conflict",
    "structure it as an investigation or negotiation that escalates through what "
    "the characters learn about each other, not through external alarms",
)


def _diversity_hint_for(i: int) -> str:
    """Structural-variation instruction for candidate ``i`` (i >= 1).
    Deterministic by index; candidate 0 always uses "" (handled by the caller)."""
    return _DIVERSITY_HINTS[(i - 1) % len(_DIVERSITY_HINTS)]


def _parse_best_of_n_flag(raw: Any):
    """Parse OTR_STORY_BEST_OF_N -> (requested_n, effective_n, clamp_reason).

    blank / non-int / <=1 => disabled (1, 1, ...); LOUD warn on a non-int value.
    int >= 2 => requested_n = value, effective_n = min(value, _LOCAL_BEST_OF_N_MAX)."""
    s = str(raw if raw is not None else "").strip()
    if s == "":
        return 1, 1, ""
    try:
        val = int(s)
    except (TypeError, ValueError):
        log.warning(
            "[best_of_n] OTR_STORY_BEST_OF_N=%r is not an integer; best-of-N "
            "disabled (single outline path)", raw,
        )
        return 1, 1, "non_int_flag"
    if val <= 1:
        return 1, 1, ""
    requested_n = val
    effective_n = min(val, _LOCAL_BEST_OF_N_MAX)
    clamp_reason = "max_local_6" if val > _LOCAL_BEST_OF_N_MAX else ""
    return requested_n, effective_n, clamp_reason


def resolve_best_of_n(resolved: Any, *, allow_remote: bool = False):
    """Resolve the effective candidate count -> (requested_n, effective_n,
    clamp_reason). Reads OTR_STORY_BEST_OF_N, then applies the provider gate.

    Provider gate: a REMOTE creative writer (``openrouter:`` / ``comfy:``) clamps
    to N=1 by default (local-only); the operator opts in via chunk 4's
    allow_remote (which then applies the tighter remote cap + cost guard). When
    ``effective_n < 2`` the writer runs the existing single path -- no selector,
    no telemetry key (the byte-identical path)."""
    import os
    raw = os.environ.get("OTR_STORY_BEST_OF_N", "0")
    requested_n, effective_n, clamp_reason = _parse_best_of_n_flag(raw)
    if effective_n < 2:
        return requested_n, effective_n, clamp_reason

    model = str((resolved or {}).get("creative_writing_model", "") or "")
    is_remote = model.startswith(("openrouter:", "comfy:"))
    if is_remote and not allow_remote:
        log.warning(
            "[best_of_n] remote creative writer %r -> best-of-N clamped to N=1 "
            "(local-only by default; set OTR_STORY_BEST_OF_N_ALLOW_REMOTE=1 to "
            "opt in)", model,
        )
        return requested_n, 1, "remote_provider_local_only"
    # NOTE (chunk 4): the allow_remote=True branch (tighter cap + fail-closed
    # cost guard) is inserted here; chunk 3 leaves remote => N=1.
    return requested_n, effective_n, clamp_reason


@dataclass
class _Candidate:
    index: int
    outline: Any
    score: Any        # StoryScore on success, None on generation failure.
    ok: bool
    error_type: Any   # str on failure, None on success.


def _merge_best_of_n_telemetry(meta: Any, effective_n: int, winner_index: int,
                               candidates: List["_Candidate"]) -> None:
    """Merge the best_of_n telemetry block into meta.story_quality (never
    replace the dict -- consistent with the L5a setdefault/update rule). plain
    JSON primitives only. ``requested_n`` + ``clamp_reason`` are placeholders
    here; the writer stamps the real gate-derived values after the selector
    returns (it owns the flag parse + provider gate)."""
    if not isinstance(meta, dict):
        return
    scores = []
    for c in candidates:
        if c.ok:
            scores.append({
                "candidate_index": c.index,
                "ok": True,
                "ungrounded_crisis_density": c.score.ungrounded_crisis_density,
                "distinct_conflict_nouns": c.score.distinct_conflict_nouns,
                "premise_grounding": c.score.premise_grounding,
            })
        else:
            scores.append({
                "candidate_index": c.index,
                "ok": False,
                "error_type": c.error_type,
            })
    sq = meta.setdefault("story_quality", {})
    if isinstance(sq, dict):
        sq["best_of_n"] = {
            "requested_n": effective_n,   # writer overwrites with the true value
            "effective_n": effective_n,
            "winner_index": winner_index,
            "scores": scores,
            "clamp_reason": "",           # writer overwrites with the true value
        }


def select_best_outline(generate_outline_fn, outline_req, *, cast_seed, n, meta,
                        roster):
    """Generate ``n`` candidate outlines under cast_seed-keyed seeds + structural
    diversity_hints, score each with the PURE scorer, and return the best.

    Determinism: candidate ``i`` is seeded with sha256(f"{cast_seed}:outline:{i}").
    Candidate 0 uses diversity_hint="" (byte-identical PROMPT); i>=1 gets a
    structural-variation instruction. Each generation is wrapped in
    try/except OutlineFailedError -> LOUD + continue. Keep-best comparator:
    (ungrounded_crisis_density asc, distinct_conflict_nouns desc,
    premise_grounding desc, candidate index asc). Never-fail: if every candidate
    raised, run ONE deterministic fallback at the i=0 seed + hint=""; if THAT
    raises too, fail LOUD. build_sq_data is NOT called here (runs once
    downstream on the winner). Telemetry is merged into meta.story_quality."""
    import hashlib
    import random
    import dataclasses
    import torch  # local import; module-level torch is forbidden in the writer.
    try:
        from ._otr_outline import OutlineFailedError
    except ImportError:  # pragma: no cover - standalone / test load
        from _otr_outline import OutlineFailedError  # type: ignore

    def _seed_rngs(idx: int) -> None:
        h = hashlib.sha256(
            f"{cast_seed}:outline:{idx}".encode("utf-8")
        ).hexdigest()
        seed_int = int(h, 16)
        # Seed IMMEDIATELY before the call. Best-effort for local backends;
        # remote backends may ignore process seeds -- diversity_hint is the
        # primary diversity lever there (chunk 4).
        torch.manual_seed(seed_int % (2 ** 64))
        random.seed(seed_int % (2 ** 32))

    candidates: List[_Candidate] = []
    for i in range(n):
        hint = "" if i == 0 else _diversity_hint_for(i)
        req_i = dataclasses.replace(outline_req, diversity_hint=hint)
        _seed_rngs(i)
        try:
            outline_i = generate_outline_fn(req_i)
        except OutlineFailedError as exc:
            log.warning(
                "[best_of_n] candidate %d/%d FAILED to generate (%s); "
                "skipping", i, n, type(exc).__name__,
            )
            candidates.append(_Candidate(i, None, None, False,
                                         type(exc).__name__))
            continue
        score_i = score_outline(outline_i, meta, roster)
        candidates.append(_Candidate(i, outline_i, score_i, True, None))
        log.info(
            "[best_of_n] candidate %d/%d scored: density=%.4f distinct=%d "
            "grounding=%.3f", i, n, score_i.ungrounded_crisis_density,
            score_i.distinct_conflict_nouns, score_i.premise_grounding,
        )

    ok = [c for c in candidates if c.ok]
    if ok:
        winner = min(ok, key=lambda c: (
            c.score.ungrounded_crisis_density,
            -c.score.distinct_conflict_nouns,
            -c.score.premise_grounding,
            c.index,
        ))
        log.info(
            "[best_of_n] winner = candidate %d (%d of %d generated; "
            "density=%.4f distinct=%d grounding=%.3f)",
            winner.index, len(ok), n,
            winner.score.ungrounded_crisis_density,
            winner.score.distinct_conflict_nouns,
            winner.score.premise_grounding,
        )
        _merge_best_of_n_telemetry(meta, n, winner.index, candidates)
        return winner.outline

    # Never-fail: every candidate raised. ONE deterministic fallback at the i=0
    # seed + diversity_hint="" (NOT "normal"). If THAT raises too, fail LOUD.
    log.warning(
        "[best_of_n] all %d candidate(s) failed to generate; running ONE "
        "deterministic fallback (i=0 seed, no hint)", n,
    )
    _seed_rngs(0)
    req0 = dataclasses.replace(outline_req, diversity_hint="")
    outline0 = generate_outline_fn(req0)  # may raise OutlineFailedError -> LOUD
    _merge_best_of_n_telemetry(meta, n, 0, candidates)
    return outline0
