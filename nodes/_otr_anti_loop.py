"""nodes/_otr_anti_loop.py -- deterministic mechanical anti-loop / dedupe.

Pure, no-LLM detection of two repair-worthy defects across the VOICED
lines of a ledger (character + announcer). This is the UNCONDITIONAL
repair-target source for the story-quality spine (story-quality Phase 1,
A4): because the story critic returns ``StoryCriticReport.clean()``
identically whether it succeeded, found nothing, or silently failed, a
deterministic floor is the only thing that guarantees real repair targets
exist. The spine recomposes CHARACTER targets through the editor's
recompose seam (one repair owner); the freeze cascade unions these
targets into the critic's ``reroll_targets`` (A3).

Detected defects:
  (a) near-duplicate spoken lines -- normalized (lowercase, collapsed
      whitespace, stripped punctuation), both >= 6 words, an exact
      normalized match OR ``difflib.SequenceMatcher`` ratio >= 0.90;
  (b) the "What if...?" loop -- the same normalized question-prefix
      (the words through the first "?") repeated across >= 3 voiced
      lines OR >= 2 CONSECUTIVE voiced lines (a single rhetorical
      question is fine).

The announcer open/close TAGLINES are EXEMPT: the first and last
``speaker_role == "announcer"`` lines are locked structural content and
are repaired -- if truncated -- by the dedicated announcer composer (A6),
never by a character reroll path (the critic excludes them by design).

Detection emits ``AntiLoopTarget`` records (voiced-view index = the same
index space the editor + recompose_fn use, a beat_id string, the speaker
role, the defect kind, and an actionable hint). Detection is pure; the
caller owns repair. ``repair_character_anti_loops`` is the spine-side
repair owner: it recomposes CHARACTER targets in place via a supplied
recompose_fn and never raises.

UTF-8 no BOM. 4-space indentation. Deterministic (C7-safe): pure string
operations + difflib, no RNG, no LLM, no clock.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, List, Optional

log = logging.getLogger("OTR_AntiLoop")

# Tunables (kept module-level so tests + the cascade share one source).
NEAR_DUP_MIN_WORDS = 6
NEAR_DUP_RATIO = 0.90
QUESTION_PREFIX_WORDS = 4
LOOP_MIN_TOTAL = 3        # same question-prefix across >= 3 voiced lines
LOOP_MIN_CONSECUTIVE = 2  # OR >= 2 consecutive voiced lines

_WS = re.compile(r"\s+")
# Strip everything that is not a word char or inter-word space for the
# normalized comparison form (drops punctuation, music glyphs, brackets).
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

_DUP_HINT = (
    "This line repeats an earlier line almost verbatim. Write a fresh, "
    "distinct line that advances the scene -- do not restate."
)
_LOOP_HINT = (
    "This line reuses the same rhetorical question pattern as nearby lines. "
    "Break the loop: make a concrete statement or take a different action."
)
_BOTH_HINT = (
    "This line both repeats an earlier line and reuses a looping question "
    "pattern. Write a fresh, concrete line that advances the scene."
)


@dataclass(frozen=True)
class AntiLoopTarget:
    """One deterministic repair target.

    voiced_index: index into the VOICED view (character + announcer, in
        order) -- the same space the editor + recompose_fn use.
    beat_id: the ledger line's beat_id (str), for the critic-union path.
    speaker_role: "character" or "announcer".
    kind: "near_duplicate", "question_loop", or "near_duplicate+question_loop".
    hint: an actionable recompose/reroll instruction.
    """

    voiced_index: int
    beat_id: str
    speaker_role: str
    kind: str
    hint: str


def _normalize(text: Any) -> str:
    """Lowercase, drop punctuation, collapse whitespace. Never raises."""
    try:
        s = str(text or "")
        s = _PUNCT.sub(" ", s)
        s = _WS.sub(" ", s).strip().lower()
        return s
    except Exception:  # noqa: BLE001
        return ""


def _word_count(normalized: str) -> int:
    return len(normalized.split()) if normalized else 0


def _question_prefix(text: Any) -> Optional[str]:
    """Normalized words through the first '?', capped at QUESTION_PREFIX_WORDS.

    Returns None when the line has no '?' (so a non-question line never
    joins a question loop) or the segment is empty.
    """
    try:
        s = str(text or "")
        qpos = s.find("?")
        if qpos == -1:
            return None
        norm = _normalize(s[:qpos])
        words = norm.split()
        if not words:
            return None
        return " ".join(words[:QUESTION_PREFIX_WORDS])
    except Exception:  # noqa: BLE001
        return None


def _announcer_exempt_indices(voiced: List[dict]) -> set:
    """The first and last announcer lines = the open/close taglines.

    Locked structural content; never a character-reroll target. Returns
    the set of voiced-view indices to skip.
    """
    ann = [i for i, ln in enumerate(voiced)
           if isinstance(ln, dict) and ln.get("speaker_role") == "announcer"]
    if not ann:
        return set()
    return {ann[0], ann[-1]}


def find_anti_loop_targets(voiced: List[dict]) -> List[AntiLoopTarget]:
    """Detect near-duplicate + question-loop defects over the voiced view.

    `voiced` is the editor's voiced view (character + announcer dicts, in
    order). Pure + deterministic; never raises. Returns one target per
    offending line, sorted by voiced index, deduplicated (a line caught by
    both rules gets a single combined target).
    """
    try:
        if not voiced:
            return []
        exempt = _announcer_exempt_indices(voiced)

        # Eligible = every voiced line except the announcer taglines.
        eligible = [
            i for i, ln in enumerate(voiced)
            if isinstance(ln, dict) and i not in exempt
        ]

        dup_idx: set = set()
        loop_idx: set = set()

        # (a) near-duplicate: keep the first occurrence, flag later twins.
        norms = {i: _normalize(voiced[i].get("text")) for i in eligible}
        long_enough = [i for i in eligible
                       if _word_count(norms[i]) >= NEAR_DUP_MIN_WORDS]
        for a_pos in range(len(long_enough)):
            i = long_enough[a_pos]
            if i in dup_idx:
                continue
            for b_pos in range(a_pos + 1, len(long_enough)):
                j = long_enough[b_pos]
                if j in dup_idx:
                    continue
                na, nb = norms[i], norms[j]
                if na == nb or SequenceMatcher(None, na, nb).ratio() >= NEAR_DUP_RATIO:
                    dup_idx.add(j)  # j is the later twin -> repair it

        # (b) question-loop: group eligible question lines by prefix.
        prefixes = {}
        for i in eligible:
            p = _question_prefix(voiced[i].get("text"))
            if p:
                prefixes.setdefault(p, []).append(i)
        for p, idxs in prefixes.items():
            if len(idxs) >= LOOP_MIN_TOTAL:
                for j in idxs[1:]:  # keep the first, flag the rest
                    loop_idx.add(j)
            # OR >= LOOP_MIN_CONSECUTIVE consecutive voiced lines.
            for k in range(1, len(idxs)):
                if idxs[k] - idxs[k - 1] == 1:
                    loop_idx.add(idxs[k])

        targets: List[AntiLoopTarget] = []
        for i in sorted(dup_idx | loop_idx):
            ln = voiced[i]
            in_dup = i in dup_idx
            in_loop = i in loop_idx
            if in_dup and in_loop:
                kind, hint = "near_duplicate+question_loop", _BOTH_HINT
            elif in_dup:
                kind, hint = "near_duplicate", _DUP_HINT
            else:
                kind, hint = "question_loop", _LOOP_HINT
            targets.append(AntiLoopTarget(
                voiced_index=i,
                beat_id=str(ln.get("beat_id") or ""),
                speaker_role=str(ln.get("speaker_role") or "character"),
                kind=kind,
                hint=hint,
            ))
        return targets
    except Exception as exc:  # noqa: BLE001 -- detection must never break a run
        log.warning("[OTR_AntiLoop] detection failed: %r", exc)
        return []


def repair_character_anti_loops(
    led: Any,
    voiced: List[dict],
    recompose_fn: Callable[[int, str, str], str],
) -> dict:
    """Spine-side repair owner: recompose CHARACTER anti-loop targets in place.

    For every CHARACTER target, call ``recompose_fn(voiced_index,
    original_text, hint)`` and write the result back onto the voiced line
    dict (which is a reference into led.data["lines"], so the ledger is
    mutated). Announcer targets are NOT touched here -- announcer defects
    are owned by the dedicated announcer composer (A6). recompose_fn is
    fail-soft (returns the original text on any error), so a beat is never
    corrupted. Never raises.

    Returns a status dict: {"status", "targets", "character_targets",
    "repaired", "kinds"}.
    """
    summary = {
        "status": "ok",
        "targets": 0,
        "character_targets": 0,
        "repaired": 0,
        "kinds": {},
    }
    try:
        targets = find_anti_loop_targets(voiced)
        summary["targets"] = len(targets)
        for t in targets:
            summary["kinds"][t.kind] = summary["kinds"].get(t.kind, 0) + 1
        for t in targets:
            if t.speaker_role != "character":
                continue
            summary["character_targets"] += 1
            i = t.voiced_index
            if not (0 <= i < len(voiced)):
                continue
            line = voiced[i]
            original = str(line.get("text") or "")
            new_text = recompose_fn(i, original, t.hint)
            if new_text and str(new_text).strip() and str(new_text) != original:
                line["text"] = str(new_text)
                summary["repaired"] += 1
        return summary
    except Exception as exc:  # noqa: BLE001 -- repair must never break a run
        log.warning("[OTR_AntiLoop] character repair failed: %r", exc)
        summary["status"] = f"ERROR:{type(exc).__name__}"
        return summary
