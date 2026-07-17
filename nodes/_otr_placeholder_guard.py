"""Literal placeholder-token guard (v4 campaign, P1(vii)).

Scoreboard defect (unlogged; same class as PBUG-20260712-06 / -20260712-01): a
production ledger shipped with a bare placeholder token where a real value
belongs -- a character NAMED ``X``, a line that is just ``Y``, a ``TBD`` speaker.
These are production-plausible (they survive schema shape checks) yet are
obviously wrong on the microphone / in the credits.

This is a deterministic, token-BOUNDARY validator over NAMED exact field paths
(cast name / character description, line speaker, and a spoken line that is
WHOLLY a placeholder). It normalizes surrounding quotes + punctuation before the
comparison, is case-insensitive, and matches a WHOLE field value -- so a
legitimate word that merely CONTAINS the letters (``box``, ``xylophone``,
``Yara``) never trips, and ``X marks the spot`` (a real sentence) is not flagged
because the whole line is not a placeholder. No authored repair: a placeholder is
a generation bug the pack must fix, not prose to re-write.

Opt-in per bank via ``defaults.placeholder_guard`` (Bucket B, per-pack; default
False -> INERT for every current bank). Terminal =
``_otr_ledger_freeze._check_g13_placeholder_tokens`` (in ``run_gap_audit`` -> the
one path every family crosses); raises ``FreezeAssertionError`` at Phase 10.
Self-contained. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

from typing import Any, List

__all__ = [
    "PLACEHOLDER_TOKENS",
    "is_placeholder_value",
    "find_placeholder_fields",
]

# Bare tokens that are never a real name / line / speaker -- only scaffolding.
PLACEHOLDER_TOKENS = frozenset({
    "x", "y", "z",
    "xx", "yy", "zz", "xxx", "yyy", "zzz",
    "tbd", "tba", "tbc", "todo", "tk", "fixme",
    "placeholder", "lorem", "ipsum", "n/a", "na", "none", "null",
    "name", "character", "speaker",  # literal field-name leaks
})

# Surrounding punctuation / quotes stripped before the comparison (NOT the
# interior -- 'n/a' keeps its slash; leading/trailing only).
_STRIP = " \t\r\n\"'`.,!?;:()[]{}*_-–—‘’“”"


def is_placeholder_value(value: Any) -> bool:
    """True when *value*, stripped of surrounding quotes/punctuation and
    casefolded, is exactly a bare placeholder token. Whole-value only -- a
    sentence that merely contains 'x' is not a placeholder. Never raises."""
    try:
        norm = str(value or "").strip().strip(_STRIP).strip().casefold()
    except Exception:  # noqa: BLE001
        return False
    return bool(norm) and norm in PLACEHOLDER_TOKENS


def find_placeholder_fields(ledger_data: Any) -> "List[str]":
    """Return ``"path: value"`` for every named field carrying a bare
    placeholder token. Checks cast ``name`` / ``character_description``, each
    line's ``speaker``, and any spoken line whose whole ``text`` is a
    placeholder. Read-only; never raises."""
    hits: "List[str]" = []
    if not isinstance(ledger_data, dict):
        return hits
    for i, row in enumerate(ledger_data.get("cast") or []):
        if not isinstance(row, dict):
            continue
        for field in ("name", "character_description"):
            val = row.get(field)
            if val is not None and is_placeholder_value(val):
                hits.append(f"cast[{i}].{field}={val!r}")
    for line in ledger_data.get("lines") or []:
        if not isinstance(line, dict):
            continue
        lid = line.get("line_id") or "<no line_id>"
        spk = line.get("speaker")
        if spk is not None and is_placeholder_value(spk):
            hits.append(f"lines[{lid}].speaker={spk!r}")
        text = line.get("text")
        role = str(line.get("speaker_role") or "").strip().lower()
        # A WHOLE spoken line that is only a placeholder token (music rows carry
        # generation prompts, not spoken text -> out of scope).
        if (
            isinstance(text, str)
            and not role.startswith("music")
            and is_placeholder_value(text)
        ):
            hits.append(f"lines[{lid}].text={text!r}")
    return hits
