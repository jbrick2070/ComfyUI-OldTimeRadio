"""
_otr_ledger_consumers.py -- shared read-side helpers for v2 ledger consumers
============================================================================

The new ``OTR_LedgerScriptWriter`` emits ``script_json`` as a serialized
production-ledger dict (``{"cast": [...], "lines": [...], "meta": {...}}``)
instead of the legacy parser-list shape (``[{"type": "dialogue",
"content": "[VOICE: NAME, traits] text"}, ...]``).

A consumer that does ``json.loads(script_json)`` -> iterate as a list ->
regex-parse ``[VOICE: NAME, traits]`` from a ``content`` field crashes on
the ledger dict. This module gives consumers a single, audited
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
    char_id so a stub ledger or a non-character line (announcer,
    music) doesn't blow up.

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

    Skip-canonical mute (Rec 5 / Gap 5, 2026-05-11): skips any line
    with ``line.get("skip") == True``. This is the AUTHORITATIVE
    mute signal -- set by Step 2.5 phantom-skip fallback, by the
    Script Doctor's `skip` edit action, and (belt-and-suspenders)
    by the writer itself which clears `text=""` in lockstep with
    skip=True. We deliberately DO NOT filter on empty `text` alone:
    an empty-text line WITHOUT skip=True is a pending-compose row
    (the brief window between init_lines_from_outline and
    update_line_text) or a bug, and either way deserves to surface
    at the consumer rather than be silently dropped (Gap 5
    recommendation, post-Phase-3 final review).

    Pass ``include_skipped=True`` to disable the skip filter (used
    by forensic / audit code paths that want to see what got muted
    and why).
    """
    for line in ledger.get("lines") or []:
        if roles is not None and line.get("speaker_role") not in roles:
            continue
        if not include_skipped and line.get("skip"):
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
    # Role-tag alias (capstone soak catch 2026-06-09): announcer-pass lines can
    # carry the ROLE TAG ('announcer') instead of the cast row id ('c01') --
    # run2's intro line b001 did. Resolve the tag to the ANNOUNCER cast row so
    # the announcer engine reads ITS OWN voice fields instead of an empty dict.
    if str(char_id).strip().lower() == "announcer":
        for entry in ledger.get("cast") or []:
            if isinstance(entry, dict) and \
                    str(entry.get("name") or "").strip().upper() == "ANNOUNCER":
                return entry
    return {}


def speaker_name(ledger: dict, line: dict) -> str:
    """Resolve a line's ``char_id`` to its cast ``name``.

    Returns ``"UNKNOWN"`` when the cast lookup misses or the line has no
    ``char_id`` (e.g. announcer / music lines whose ``char_id`` is
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


def voice_assignments_from_cast(led: dict) -> dict:
    """Render-time derivation of the ``voice_assignments`` shape from the
    canonical ``led["cast"]``.

    Nothing is persisted at ``meta.voice_assignments``: the cast contract is
    the only source of truth for per-character voice data, and a stored
    derived view in ``meta`` invites drift between cast.voice_preset and
    meta.voice_assignments[name].voice_preset.

    Shape:
        {
          "<name>": { "voice_preset": "<v2/...>" },
          ...   # ANNOUNCER excluded regardless of engine -- this surface is
          ...   # character-only by design (voice_assignments has never
          ...   # covered the announcer row, whether it renders on Kokoro or,
          ...   # since 2026-08-24, Bark)
        }

    The ``notes`` field is intentionally absent: ``portrait_prompt`` (in the
    visual_plan) is the canonical character description.
    """
    out: dict = {}
    for c in led.get("cast") or []:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        if not name or name == "ANNOUNCER":
            continue
        out[name] = {
            "voice_preset": c.get("voice_preset") or "",
        }
    return out


__all__ = [
    "load_ledger",
    "iter_lines",
    "cast_lookup",
    "speaker_name",
    "voice_preset",
    "voice_assignments_from_cast",
]
