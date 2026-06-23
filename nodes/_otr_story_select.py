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
