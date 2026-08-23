"""nodes/_otr_news_wiring.py -- writer-side glue for news_interpreter
briefs.

  post_assembly_keyterm_check(line_rows, key_terms)
      Walk every voiced line (speaker_role in {character, announcer}),
      concatenate their text, and measure which optional key_terms landed.
      Returns (landed, missing) telemetry lists; neither list gates publish.

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
# music_* beats don't carry dialogue text -- excluded.
_VOICED_ROLES: frozenset[str] = frozenset({"character", "announcer"})



def post_assembly_keyterm_check(
    line_rows: list[dict],
    key_terms: tuple[str, ...] | list[str],
) -> tuple[list[str], list[str]]:
    """Word-boundary check that each key_term landed in dialogue.

    Returns ``(landed, missing)`` -- two disjoint telemetry lists.
    Missing terms never trigger repair, retry, or
    publication failure.

    Only voiced lines (speaker_role in {character, announcer}) count.
    music beats have no dialogue, so the term cannot land there.
    Match is case-insensitive.
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
        
        # Clean parenthetical plurals: e.g. pylon(s) -> pylon
        t = re.sub(r"\((s|es)\)$", "", term).strip()
        # Also derive singular form if term ends with s/es
        t_sing = re.sub(r"(s|es)$", "", t).strip() if t.lower().endswith(('s', 'es')) else t
        
        matched = False
        for cand in (t, t_sing):
            pattern = (
                r"(?<![A-Za-z0-9])"
                + re.escape(cand)
                + r"(?:s|es)?(?![A-Za-z0-9])"
            )
            if re.search(pattern, full_text, re.IGNORECASE):
                matched = True
                break
                
        if matched:
            landed.append(term)
        else:
            missing.append(term)
    return landed, missing
