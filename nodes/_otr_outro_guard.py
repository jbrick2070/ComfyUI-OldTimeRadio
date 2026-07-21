"""Outro cast-completeness guard (v4 campaign, P1(v)).

WHY. A radio sign-off credits the cast by name. The outro is ONE authored
free-form announcer line (no structured roster field at a forced coordinate), so
a missing required name cannot be Python-appended without inventing prose. The
PBUG-20260710-02 fix already checks outro PRESENCE against the real final rows;
this adds COMPLETENESS: every character who actually appeared should be named.

WHAT. Opt-in per bank via the validated scalar default
``defaults.require_outro_cast_complete`` (bool, default False -> INERT for every
current bank; a v4 bank flips it True in Phase 2). Two parts:

  1. deterministic scan (pure) -- ``missing_final_cast_names(ledger_data)`` reads
     the REAL final rows: the character char_ids (announcer/music excluded, the
     same convention as production_ledger.cast_ids_from_ledger) that carry a
     non-skipped spoken line, mapped to their cast name. A name is PRESENT when
     its full form OR any significant token (alpha, len>=3, not a title) appears
     in the outro line, casefolded + word-bounded. Leniency is deliberate: a gate
     must never FALSE-fail a good episode (THE LAW), so a coincidental token match
     that spares a raise is the safe error direction.
  2. authored repair -- ``apply_outro_completeness_repair`` asks the creative slot
     to re-author the outro line naming the full cast; keeps the rewrite only when
     it is non-empty AND complete; one bounded retry; else leaves the original for
     the deterministic terminal. Python NEVER appends a name to prose.

THE LAW. The deterministic terminal is
``_otr_ledger_freeze._check_g12_outro_cast_complete`` (in ``run_gap_audit`` -> the
one path every family crosses); the authored repair only improves. Self-contained
(no ledger/rules import -> no circular import from the freeze module). UTF-8 no
BOM, SFW.
"""
from __future__ import annotations

import re
from typing import Any, Callable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - package and standalone import styles
    from ._otr_text_metrics import set_line_text_metrics
except ImportError:  # pragma: no cover
    from _otr_text_metrics import set_line_text_metrics  # type: ignore

__all__ = [
    "final_cast_names",
    "find_outro_text",
    "missing_final_cast_names",
    "apply_outro_completeness_repair",
]

# Sentinel char_ids that are NOT nameable characters (mirror
# production_ledger._NON_CHARACTER_CHAR_ID_SENTINELS without importing it).
_NON_CHARACTER_CHAR_IDS = frozenset({
    "announcer", "music", "music_open", "music_close", "music_inter",
})
_NAME_TITLES = frozenset({"dr", "mr", "mrs", "ms", "mx", "prof", "sir", "dame"})


def _character_id_to_name(ledger_data: dict) -> "dict[str, str]":
    """char_id -> display name for real CHARACTER rows only (announcer / music /
    name==ANNOUNCER excluded, mirroring cast_ids_from_ledger)."""
    out: "dict[str, str]" = {}
    for row in (ledger_data.get("cast") or []):
        if not isinstance(row, dict):
            continue
        cid = str(row.get("char_id") or "").strip()
        if not cid or cid.lower() in _NON_CHARACTER_CHAR_IDS:
            continue
        name = str(row.get("name") or "").strip()
        if not name or name.upper() == "ANNOUNCER":
            continue
        out[cid] = name
    return out


def final_cast_names(ledger_data: Any) -> "List[str]":
    """Names of the characters who actually appear in the final rows: a real
    character char_id carrying at least one NON-skipped spoken line. First-
    appearance order, de-duped. Never raises."""
    if not isinstance(ledger_data, dict):
        return []
    id_to_name = _character_id_to_name(ledger_data)
    if not id_to_name:
        return []
    seen: set = set()
    names: "List[str]" = []
    for line in (ledger_data.get("lines") or []):
        if not isinstance(line, dict):
            continue
        if line.get("skip") is True:
            continue
        if str(line.get("speaker_role") or "").strip().lower() != "character":
            continue
        cid = str(line.get("char_id") or "").strip()
        name = id_to_name.get(cid)
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _last_announcer_row(ledger_data: Any) -> "Optional[dict]":
    """The LAST announcer line BY POSITION (the sign-off beat), or None. The
    sign-off is positional: an empty last announcer row means the outro is
    missing (the existing presence check owns that) -- an earlier announcer line
    (the intro) must NOT be mistaken for the outro. Never raises."""
    if not isinstance(ledger_data, dict):
        return None
    found: "Optional[dict]" = None
    for line in (ledger_data.get("lines") or []):
        if not isinstance(line, dict):
            continue
        if str(line.get("speaker_role") or "").strip().lower() == "announcer":
            found = line
    return found


def find_outro_text(ledger_data: Any) -> "Optional[str]":
    """The sign-off text: the LAST announcer line's text, or None when it is
    absent/empty. Never raises."""
    row = _last_announcer_row(ledger_data)
    if row is None:
        return None
    text = row.get("text")
    return text if isinstance(text, str) and text.strip() else None


def _significant_tokens(name: str) -> "List[str]":
    toks = re.findall(r"[^\W\d_]+", name, flags=re.UNICODE)  # alpha runs
    return [t for t in toks if len(t) >= 3 and t.casefold() not in _NAME_TITLES]


def _name_present(name: str, outro_casefold: str) -> bool:
    """A name is present when its full form OR any significant token appears in
    the outro, word-bounded + casefolded. Lenient by design (never false-fail)."""
    full = name.strip().casefold()
    if full and re.search(rf"(?<!\w){re.escape(full)}(?!\w)", outro_casefold):
        return True
    for tok in _significant_tokens(name):
        if re.search(rf"(?<!\w){re.escape(tok.casefold())}(?!\w)", outro_casefold):
            return True
    return False


def missing_final_cast_names(ledger_data: Any) -> "List[str]":
    """Final-cast names NOT credited in the outro line. Empty when there is no
    outro line yet (presence is the existing epilogue check's job) or no final
    cast. Never raises."""
    required = final_cast_names(ledger_data)
    if not required:
        return []
    outro = find_outro_text(ledger_data)
    if not outro:
        return []
    low = outro.casefold()
    return [n for n in required if not _name_present(n, low)]


# ---------------------------------------------------------------------------
# Authored repair (creative slot; keep-if-complete)
# ---------------------------------------------------------------------------

def _append_flag(line: dict, flag: str) -> None:
    flags = line.get("compose_flags")
    if not isinstance(flags, list):
        flags = []
        line["compose_flags"] = flags
    flags.append(flag)


def _repair_messages(outro: str, required: "Sequence[str]") -> list:
    roster = ", ".join(required)
    return [
        {
            "role": "system",
            "content": (
                "You revise the announcer's closing line of a radio drama. Keep "
                "it a single natural sign-off in the same tone. It MUST credit "
                "every listed cast member by name. Do not add stage directions, "
                "quotation marks, or a speaker label. Return ONLY the revised "
                "line as plain prose."
            ),
        },
        {
            "role": "user",
            "content": f"Cast to credit by name: {roster}\nClosing line: {outro}",
        },
    ]


def apply_outro_completeness_repair(
    ledger_data: Any,
    creative_fn: "Optional[Callable[..., Any]]",
    *,
    max_attempts: int = 2,
) -> int:
    """Re-author the outro line so it credits the full final cast. Returns 1 when
    the outro was rewritten, else 0. Keeps a rewrite only when it is non-empty
    AND names every required cast member; otherwise leaves the original for the
    deterministic terminal. Python never appends a name to prose. Never raises."""
    if creative_fn is None or not isinstance(ledger_data, dict):
        return 0
    required = final_cast_names(ledger_data)
    if not required:
        return 0
    row = _last_announcer_row(ledger_data)
    if row is None:
        return 0
    if not missing_final_cast_names(ledger_data):
        return 0
    original = str(row.get("text") or "")
    budget = min(320, max(64, len(original.split()) * 3 + 48))
    for _ in range(max(1, max_attempts)):
        try:
            # LLM slot: creative -- the announcer sign-off is a narrative framing
            # pass; the deterministic G12 terminal is the gate.
            raw = creative_fn(
                _repair_messages(original, required),
                temperature=0.7,
                max_new_tokens=budget,
                stop=None,
            )
        except Exception:  # noqa: BLE001 -- a failed slot call is not a ship-stop
            return 0
        cand = str(raw or "").strip().strip('"').strip()
        if not cand:
            continue
        set_line_text_metrics(row, cand)
        if not missing_final_cast_names(ledger_data):
            _append_flag(row, "outro_cast_repair")
            return 1
    # exhausted: restore the original (Python invents nothing) for the terminal
    set_line_text_metrics(row, original)
    return 0
