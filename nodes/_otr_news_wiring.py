"""nodes/_otr_news_wiring.py -- writer-side glue for news_interpreter
briefs.

Two helpers, both pure (no I/O, no LLM calls). They land in commit 4
of the news_interpreter sprint (ADR docs/news_interpreter_adr.md
section 5).

  override_announcer_close(line_rows, news_close_brief)
      Stamp the news_close_brief verbatim onto the LAST announcer
      line in the per-beat output. The line composer already wrote
      something there from the outline's beat.intent; we replace it
      with the news_interpreter's purpose-specific closing read so
      the announcer's final beat carries the actual journalistic
      content from the source article.

  post_assembly_keyterm_check(line_rows, key_terms, min_required=2)
      Walk every voiced line (speaker_role in {character, announcer}),
      concatenate their text, and verify each key_term landed via the
      same word-boundary regex used by news_interpreter.v1_validate.
      Returns (landed, missing) lists; the caller decides whether to
      log + proceed, warn, or hard-fail. Per ADR section 4.4.

Both helpers live in their own small module so tests can exercise
them without importing the heavy OTR_LedgerScriptWriter module
(which pulls in comfy.utils + the LLM loader).
"""
from __future__ import annotations

import re
from typing import Optional


# Voiced roles that count for the post-assembly key_terms check.
# music_* / sfx beats don't carry dialogue text -- excluded.
_VOICED_ROLES: frozenset[str] = frozenset({"character", "announcer"})


# Word-boundary regex pattern shared with news_interpreter.v1_validate.
# Built fresh per term so we can re.escape() correctly.
def _word_boundary_pattern(term: str) -> str:
    return (
        r"(?<![A-Za-z0-9])"
        + re.escape(term)
        + r"(?![A-Za-z0-9])"
    )


def override_announcer_close(
    line_rows: list[dict],
    news_close_brief: str,
) -> Optional[dict]:
    """Stamp news_close_brief onto the LAST announcer line.

    Returns the line_row that was mutated, or ``None`` if no announcer
    line was found in ``line_rows`` or ``news_close_brief`` is empty
    (whitespace-only counts as empty).

    ``line_rows`` is the writer's pre-set_lines() in-flight list. Each
    row carries a private ``_speaker_role`` key (set by the per-beat
    loop, stripped before set_lines). The override mutates the row's
    ``text`` field in place. ``char_count`` and ``word_count`` are
    recomputed at set_lines time, so we don't touch them here.

    Idempotent: calling twice with the same brief is a no-op.
    """
    brief = (news_close_brief or "").strip()
    if not brief:
        return None
    last_idx = -1
    for i, row in enumerate(line_rows):
        if row.get("_speaker_role") == "announcer":
            last_idx = i
    if last_idx < 0:
        return None
    line_rows[last_idx]["text"] = brief
    return line_rows[last_idx]


def post_assembly_keyterm_check(
    line_rows: list[dict],
    key_terms: tuple[str, ...] | list[str],
    *,
    min_required: int = 2,
) -> tuple[list[str], list[str]]:
    """Word-boundary check that each key_term landed in dialogue.

    Returns ``(landed, missing)`` -- two disjoint lists of terms.
    The caller decides policy (warn / hard-fail / repair pass).

    ``min_required`` is for the caller's policy decision; this
    function does not enforce it. Per ADR section 4.4, the
    canonical policy is:
      - zero terms landed -> hard fail (with a repair pass before)
      - some missing but ``len(landed) >= min_required`` -> warn
      - all landed -> pass clean

    Only voiced lines (speaker_role in {character, announcer}) count.
    music / sfx beats have no dialogue, so the term cannot land there.
    Match is case-insensitive (same as v1_validate in
    news_interpreter).
    """
    # Concat every voiced line's text. _speaker_role on line_rows is
    # the in-flight key the writer uses before set_lines() strips it;
    # post-write consumers (downstream nodes) see a public 'speaker_role'
    # instead. This helper accepts either via .get with fallback.
    parts: list[str] = []
    for r in line_rows:
        role = r.get("_speaker_role") or r.get("speaker_role")
        if role in _VOICED_ROLES:
            parts.append(r.get("text", "") or "")
    full_text = " ".join(parts)
    landed: list[str] = []
    missing: list[str] = []
    for term in key_terms:
        if not term:
            continue
        if re.search(_word_boundary_pattern(term), full_text, re.IGNORECASE):
            landed.append(term)
        else:
            missing.append(term)
    # min_required is a caller policy knob; documented in signature
    # so the parameter shows up in inspect.signature for any caller
    # that wants to surface the canonical default. We do not act on
    # it here.
    _ = min_required
    return landed, missing
