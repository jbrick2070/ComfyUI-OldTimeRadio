"""
_otr_ledger_consumers.py -- shared read-side helpers for v2 ledger consumers
============================================================================

The new ``OTR_LedgerScriptWriter`` emits ``script_json`` as a serialized
production-ledger dict (``{"cast": [...], "lines": [...], "meta": {...}}``)
instead of the legacy parser-list shape (``[{"type": "dialogue",
"content": "[VOICE: NAME, traits] text"}, ...]``).

Every downstream audio/video consumer used to do
``json.loads(script_json)`` -> iterate as a list -> regex-parse
``[VOICE: NAME, traits]`` from a ``content`` field. That path crashes on
the new ledger dict. This module gives those consumers a single, audited
read surface so the parsing logic lives in one place rather than copied
into seven node files.

Distinct from ``_otr_ledger.py``:

  * ``_otr_ledger.py`` -- WRITE-side I/O: load_ledger_safe / save_ledger_safe
    against the on-disk ``*_ledger.json``, plus per-line / per-clip patches
    used for ledger write-back AFTER an audio or video node has rendered.
  * ``_otr_ledger_consumers.py`` (this module) -- READ-side parsing:
    parse the ``script_json`` STRING input that flows through the ComfyUI
    graph from OTR_LedgerScriptWriter, validate shape, surface structured
    fields (char_id / speaker_role / text / cast voice_preset).

Strict on shape, graceful on missing fields:

  * ``load_ledger`` raises ``ValueError`` on the legacy parser-list shape.
    A stale workflow that wires the legacy writer into a rewritten
    consumer fails LOUD at the consumer boundary, not silently with
    half-degraded audio mid-soak.
  * ``cast_lookup`` / ``speaker_name`` / ``voice_preset`` degrade
    gracefully (return ``{}`` / ``"UNKNOWN"`` / ``None``) on missing
    char_id so a stub ledger or a non-character line (announcer, sfx,
    music) doesn't blow up.
  * ``production_plan_or_empty`` returns ``{}`` for empty / None /
    invalid plan_json so consumers can demote production_plan_json
    from required to optional without a special-case branch.

UTF-8 no BOM. No GPU. No I/O. Safe to import anywhere.
"""
from __future__ import annotations

import json
from typing import Iterator, Optional, Set


def load_ledger(script_json: str) -> dict:
    """Parse ``script_json`` and confirm it's a ledger dict.

    Raises ``ValueError`` on the legacy parser-list shape so stale
    wirings fail loudly at the consumer boundary instead of silently
    producing half-degraded audio.
    """
    data = json.loads(script_json)
    if isinstance(data, list):
        raise ValueError(
            "legacy parser-list format not supported by ledger consumer; "
            "rewire to OTR_LedgerScriptWriter"
        )
    if not isinstance(data, dict):
        raise ValueError(
            f"expected ledger dict, got {type(data).__name__}"
        )
    return data


def iter_lines(
    ledger: dict,
    *,
    roles: Optional[Set[str]] = None,
    include_skipped: bool = False,
) -> Iterator[dict]:
    """Yield ``ledger['lines']``, optionally filtered by ``speaker_role``.

    ``roles`` is a set of allowed ``speaker_role`` values. A line with a
    missing or unknown ``speaker_role`` is skipped when a filter set is
    supplied. With ``roles=None`` (default) every line is yielded
    regardless of ``speaker_role`` -- used by the sequencer, which needs
    the full timeline.

    Post-Phase-3 review (Rec 5, 2026-05-11): also skips any line with
    ``line.get("skip") == True`` OR an empty ``text`` (the §7 skip-
    canonical mute pattern from Step 2.5 / Script Doctor skip edits).
    Both signals together are belt-and-suspenders: setting either
    alone is sufficient to mute; honoring both here protects against
    a consumer that handles only one.

    Pass ``include_skipped=True`` to disable the skip filter (used
    by forensic / audit code paths that want to see what got muted
    and why).
    """
    for line in ledger.get("lines") or []:
        if roles is not None and line.get("speaker_role") not in roles:
            continue
        if not include_skipped:
            # §7 skip-canonical mute (Rec 5, 2026-05-11). Honor
            # explicit skip=True (Step 2.5 phantom skip + doctor
            # skip action), AND empty `text` (belt-and-suspenders
            # signal stamped alongside skip=True). Either alone
            # mutes the line; both together are the defense-in-
            # depth pattern. Downstream TTS / clip-timing consumers
            # (Bark, Kokoro, SceneSequencer) all flow through this
            # one helper so the gate lives in one place.
            if line.get("skip"):
                continue
            text = line.get("text") or ""
            if not text.strip():
                continue
        yield line


def cast_lookup(ledger: dict, char_id: str) -> dict:
    """Return the cast entry whose ``char_id`` matches.

    The ledger's ``cast`` is a LIST of dicts (each carrying its own
    ``char_id``), not a dict keyed by char_id. Returns an empty dict on
    miss so callers can chain ``.get(...)`` without a KeyError.
    """
    if not char_id:
        return {}
    for entry in ledger.get("cast") or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("char_id") or "") == str(char_id):
            return entry
    return {}


def speaker_name(ledger: dict, line: dict) -> str:
    """Resolve a line's ``char_id`` to its cast ``name``.

    Returns ``"UNKNOWN"`` when the cast lookup misses or the line has no
    ``char_id`` (e.g. announcer / sfx / music lines whose ``char_id`` is
    a role tag, not a real cast member).
    """
    char_id = (line or {}).get("char_id") or ""
    name = cast_lookup(ledger, char_id).get("name")
    return str(name) if name else "UNKNOWN"


def voice_preset(ledger: dict, line: dict) -> Optional[str]:
    """Resolve a line's ``char_id`` to its cast ``voice_preset``.

    Returns ``None`` on miss so the caller can fall back to its own
    default (e.g. Bark's gender-aware hash, Kokoro's grab-bag pick).
    """
    char_id = (line or {}).get("char_id") or ""
    return cast_lookup(ledger, char_id).get("voice_preset")


def production_plan_or_empty(plan_json: str) -> dict:
    """Parse the optional Director ``production_plan_json``; return ``{}``
    for empty / None / invalid input.

    The Director is not part of the v2 ledger flow. Consumers that took
    ``production_plan_json`` as required get this graceful fallback so
    an unwired socket degrades cleanly.
    """
    if not plan_json:
        return {}
    try:
        parsed = json.loads(plan_json)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


__all__ = [
    "load_ledger",
    "iter_lines",
    "cast_lookup",
    "speaker_name",
    "voice_preset",
    "production_plan_or_empty",
]
