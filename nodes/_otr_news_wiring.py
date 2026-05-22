"""nodes/_otr_news_wiring.py -- writer-side glue for news_interpreter
briefs.

  post_assembly_keyterm_check(line_rows, key_terms, min_required=2)
      Walk every voiced line (speaker_role in {character, announcer}),
      concatenate their text, and verify each key_term landed via the
      same word-boundary regex used by news_interpreter.v1_validate.
      Returns (landed, missing) lists; the caller decides whether to
      log + proceed, warn, or hard-fail. Per ADR section 4.4.

This helper lives in its own small module so tests can exercise it
without importing the heavy OTR_LedgerScriptWriter module (which
pulls in comfy.utils + the LLM loader).

History: `override_announcer_close` also lived here -- it stamped
`news_close_brief` onto the last announcer line. It was retired
2026-05-22 (BUG-LOCAL-255): it matched a private `_speaker_role`
key absent from the ledger's `lines[]` rows, so the close was
silently never applied. The announcer closing line is now written
by `_otr_line_composer.compose_announcer_outro`, a dedicated
creative pass run post-loop in OTR_LedgerScriptWriter.
"""
from __future__ import annotations

import re


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
