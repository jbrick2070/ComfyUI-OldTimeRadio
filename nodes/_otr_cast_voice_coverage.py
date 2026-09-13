"""Every cast member gets a VOICE, or the episode fails while repair is cheap.

Operator ruling (2026-08-02): "every cast member needs a voice -- it's a radio
drama, not a mime show. It needs to either have an LLM write its lines, or
entirely remove the character from the ledger."

THE DEFECT THIS CLOSES (PBUG-20260802-02, mechanism proven from the live
ledger `pending_20260802_024714`): gemma cast "The Relay" and wrote its five
lines as pure stage direction -- no sayable dialogue. Every existing check
passed, because every existing check asks a different question:

  * the scifi_news_pro assemble gate checks RAW text -- stage direction counts, pass;
  * `stamp_receipt` minted authorship proofs from those rows -- voiced then;
  * the writer-tail cleanup (`_otr_ledger_cleanup.py:255`) then stripped stage
    directions, found "nothing sayable", and -- CORRECTLY -- emptied the rows
    with `tts_skip_reason='empty_spoken_text_at_ledger_cleanup'`;
  * the freeze gate then refused a proof-coverage mismatch naming
    `shot_001_b2`, five stages after the real answer, which was "character
    c03 never got sayable dialogue".

So the predicate here is SAYABLE surface -- `clean_spoken_text(text)` -- not
raw text. A row this gate accepts is exactly a row the cleanup will never
empty, which is what makes proofs minted after it stable.

WHERE IT RUNS: the top of `_otr_content_authorship.stamp_receipt`, the one
call site both content-owned lanes share (`_otr_scifi_news_pro.py:2538`,
the retired codex lane), immediately BEFORE proofs are minted and BEFORE
the ledger is frozen -- the panels' converged placement (Sonnet wiring review
B6; codex CUT 1: enforce pre-assembly so no portrait, caption, credit, voice
or proof ever exists for a character that will not survive).

WHAT "REMOVE" MEANS: reroll the cast candidate upstream. NEVER post-ledger
surgery -- a half-removed char_id today gets a real randomly-seeded fallback
voice (`_otr_voice_node_common.py:109-127`) and the render PASSES, which is
the worst outcome of all. This module REFUSES, loudly and by name, and since
2026-09-13 it also owns the operator's second legal exit as a pure function:
:func:`remove_silent_cast_members` drops the silent member and every row that
referenced them BEFORE the receipt's proofs are minted and before any line is
voiced, captioned or credited -- the gate's own placement. The voice
assignment already on the cast row leaves with the row. The first exit (write
the lines) belongs to the producers; the news lane chose removal (PBUG-20260913-02).

Deliberately NO machine-versus-human heuristic (codex CUT 2): the schemas
carry free-text roles, "The Relay" CAN be voiced, and a name heuristic would
be one more ambiguous policy. Every cast member gets a voice. Full stop.

Pure: no I/O, no LLM, no GPU. `clean_spoken_text` is imported function-locally
exactly as `_otr_ledger_cleanup` does, so this module stays import-light.
"""
from __future__ import annotations

from typing import Any, Mapping, MutableMapping

try:
    from ._otr_ledger_scrub import row_is_verbatim as _row_is_verbatim
except ImportError:  # pragma: no cover -- flat test/standalone load
    from _otr_ledger_scrub import row_is_verbatim as _row_is_verbatim  # type: ignore

__all__ = ["CastVoiceCoverageError", "is_announcer_cast_row",
           "missing_cast_members", "remove_silent_cast_members",
           "require_voice_coverage"]

#: The announcer's LINES carry the sentinel char_id ``"announcer"`` while its
#: CAST row carries an ordinary roster id (production_ledger.py:114-139), so
#: every consumer needs a way to recognise the row itself.
ANNOUNCER_SENTINEL = "announcer"


def is_announcer_cast_row(row: "Mapping[str, Any]") -> bool:
    """ONE definition of "this cast row is the announcer", shared by every
    layer that has to answer the question.

    It exists because the two layers shipped on 2026-08-02 answered it
    DIFFERENTLY: this module matched on name-or-role, while
    ``_otr_ledger_freeze`` matched on char_id-or-name. A row with
    ``role="announcer"`` and any other name was therefore an announcer to the
    gate (skipped, credited via the sentinel) and a CHARACTER to the freeze --
    which then hard-refused it for owning no line under its own roster id. Two
    predicates for one concept is how the layers drift apart; there is now one.

    Matches on any of the three signals a real ledger carries: the sentinel in
    ``char_id``, the canonical display name, or an explicit announcer role.
    Role is what a localised roster should rely on.
    """
    char_id = str(row.get("char_id") or "").strip().lower()
    name = str(row.get("name") or "").strip().upper()
    role = str(row.get("role") or row.get("speaker_role") or "").strip().lower()
    return (char_id == ANNOUNCER_SENTINEL
            or name == "ANNOUNCER"
            or role == ANNOUNCER_SENTINEL)


class CastVoiceCoverageError(ValueError):
    """A cast member has no sayable line. Typed, so ladders can catch it
    WITHOUT parsing message text (codex MUST-FIX 6), and carrying the
    structured facts a retry prompt needs to name the silent character."""

    def __init__(self, *, owner_bank: str, missing: "list[dict[str, str]]",
                 cast_total: int, voiced_total: int) -> None:
        self.owner_bank = str(owner_bank)
        self.missing = list(missing)
        self.cast_total = int(cast_total)
        self.voiced_total = int(voiced_total)
        names = ", ".join(
            "%s (%s)" % (m.get("name") or "?", m.get("char_id") or "?")
            for m in self.missing)
        super().__init__(
            "cast voice coverage failed for bank %r: %d of %d cast member(s) "
            "have no SAYABLE line: %s. This is a radio drama -- every cast "
            "member gets a voice. Either write real dialogue for them "
            "(stage directions are stripped before TTS and do not count) or "
            "reroll the cast so they are never in the ledger at all. "
            "Refusing BEFORE authorship proofs are minted, because the "
            "writer-tail cleanup empties unsayable rows and a proof minted "
            "from state a later stage invalidates is a proof of nothing."
            % (self.owner_bank, len(self.missing), self.cast_total, names))


def _sayable(text: Any, *, keep_parentheticals: bool = False) -> bool:
    """True when the row's text survives the SAME stripper TTS uses.

    This is the whole point of the gate: raw non-empty text is what the old
    checks accepted, and raw non-empty text full of stage direction is exactly
    what the cleanup later empties. One predicate, shared with the cleanup
    (`_otr_ledger_cleanup._clean_spoken_text`), so the gate and the cleanup
    can never disagree about what counts as a voice.
    """
    raw = str(text or "")
    if not raw.strip():
        return False
    try:
        from ._otr_script_prep import clean_spoken_text
    except ImportError:  # pragma: no cover -- flat test/standalone load
        from _otr_script_prep import clean_spoken_text  # type: ignore
    return bool(clean_spoken_text(raw, keep_parentheticals=keep_parentheticals).strip())


def missing_cast_members(ledger_data: Mapping[str, Any]
                         ) -> "tuple[list[dict[str, str]], int, int]":
    """``(missing, cast_total, voiced_total)`` -- the cast rows that own no
    non-skipped SAYABLE line, plus the totals the refusal reports. Pure, and
    the ONE predicate both exits share, so the refusal and the removal can
    never disagree about who is silent.
    """
    cast_rows = [r for r in (ledger_data.get("cast") or [])
                 if isinstance(r, Mapping)]
    lines = [r for r in (ledger_data.get("lines") or [])
             if isinstance(r, Mapping)]

    voiced_ids: "set[str]" = set()
    announcer_voiced = False
    for row in lines:
        if bool(row.get("skip")):
            continue
        if not _sayable(row.get("text"),
                        keep_parentheticals=_row_is_verbatim(row)):
            continue
        cid = str(row.get("char_id") or "").strip()
        if cid.lower() == ANNOUNCER_SENTINEL:
            announcer_voiced = True
        elif cid:
            voiced_ids.add(cid)

    # ONLY ONE ROW MAY RIDE THE SENTINEL (QA 2026-08-02). The exemption is a
    # STRING test, so it matched every row whose name or role said "Announcer"
    # -- and all of them rode the single shared `announcer_voiced` boolean. A
    # cast that legitimately contains a second announcer-ish character (a
    # sentient PA system, say) shipped it SILENT through both layers with
    # freeze_verdict='frozen_clean' and zero warnings. The sentinel credits ONE
    # row; every other announcer-named row is an ordinary character and must
    # own a line of its own.
    announcer_rows = [r for r in cast_rows if is_announcer_cast_row(r)]
    credited = announcer_rows[0] if announcer_rows else None

    missing: "list[dict[str, str]]" = []
    for row in cast_rows:
        cid = str(row.get("char_id") or "").strip()
        if not cid:
            continue
        if row is credited:
            if not announcer_voiced:
                missing.append({"char_id": cid,
                                "name": str(row.get("name") or "ANNOUNCER")})
            continue
        if cid not in voiced_ids:
            missing.append({"char_id": cid,
                            "name": str(row.get("name") or "")})

    return missing, len(cast_rows), len(voiced_ids)


def require_voice_coverage(ledger_data: Mapping[str, Any], *,
                           owner_bank: str) -> None:
    """Raise :class:`CastVoiceCoverageError` unless every character cast row
    owns at least one non-skipped line with SAYABLE text.

    ANNOUNCER is matched by its line sentinel (`char_id == "announcer"`), not
    by its cast-row char_id, because announcer lines deliberately carry the
    sentinel rather than the roster id (`production_ledger.py:114-139`;
    codex MUST-FIX 4: never compare raw IDs for ANNOUNCER). Music sentinel
    rows are ignored -- they are cues, not voices.
    """
    missing, cast_total, voiced_total = missing_cast_members(ledger_data)
    if missing:
        raise CastVoiceCoverageError(
            owner_bank=owner_bank, missing=missing,
            cast_total=cast_total, voiced_total=voiced_total)


def remove_silent_cast_members(ledger_data: "MutableMapping[str, Any]", *,
                               owner_bank: str) -> dict:
    """Operator ruling 2026-08-02, second exit: "entirely remove the character
    from the ledger". Drops every cast row with no sayable line, its lines, the
    beats those lines owned, shots left with nothing, its proof-map entries, and
    recounts ``meta.cast_contract.num_characters_locked``. Must run before the
    receipt's proofs are minted and before any line is voiced, captioned or
    credited -- the same place the refusal runs -- so nothing downstream ever
    names the removed member; the voice assignment already on the cast row
    leaves with the row.

    The credited announcer is never removed: a silent announcer is a real gap
    this exit cannot close, so it refuses exactly as the gate does.

    Returns the receipt it also stamps at ``meta.cast_voice_coverage_removed``
    (empty ``removed`` means nothing happened). Pure: no I/O, no LLM.
    """
    missing, cast_total, voiced_total = missing_cast_members(ledger_data)
    receipt: dict = {
        "owner_bank": str(owner_bank), "ruling": "2026-08-02",
        "removed": [], "line_ids": [], "beat_ids": [], "shot_ids": [],
        "cast_before": cast_total, "cast_after": cast_total,
    }
    if not missing:
        return receipt
    cast_rows = [r for r in (ledger_data.get("cast") or []) if isinstance(r, Mapping)]
    by_id = {str(r.get("char_id") or "").strip(): r for r in cast_rows}
    for m in missing:
        row = by_id.get(str(m.get("char_id") or ""))
        if row is not None and is_announcer_cast_row(row):
            raise CastVoiceCoverageError(
                owner_bank=owner_bank, missing=missing,
                cast_total=cast_total, voiced_total=voiced_total)
    ids = {str(m.get("char_id") or "").strip() for m in missing}

    lines = list(ledger_data.get("lines") or [])
    gone_lines = [r for r in lines if isinstance(r, Mapping)
                  and str(r.get("char_id") or "").strip() in ids]
    gone_line_ids = {str(r.get("line_id") or "") for r in gone_lines}
    ledger_data["lines"] = [r for r in lines if not any(r is g for g in gone_lines)]

    kept_beats = []
    for b in list(ledger_data.get("beats") or []):
        if not isinstance(b, MutableMapping):
            kept_beats.append(b)
            continue
        if str(b.get("char_id") or "").strip() in ids:
            receipt["beat_ids"].append(str(b.get("beat_id") or ""))
            continue
        owned = list(b.get("line_ids") or [])
        left = [l for l in owned if str(l) not in gone_line_ids]
        if owned and not left:
            receipt["beat_ids"].append(str(b.get("beat_id") or ""))
            continue
        if left != owned:
            b["line_ids"] = left
        kept_beats.append(b)
    ledger_data["beats"] = kept_beats

    referenced = {str(b.get("shot_id") or "") for b in kept_beats if isinstance(b, Mapping)}
    referenced |= {str(r.get("shot_id") or "") for r in ledger_data["lines"] if isinstance(r, Mapping)}
    kept_shots = []
    for s in list(ledger_data.get("shots") or []):
        if isinstance(s, Mapping) and str(s.get("shot_id") or "") not in referenced:
            receipt["shot_ids"].append(str(s.get("shot_id") or ""))
            continue
        kept_shots.append(s)
    if "shots" in ledger_data:
        ledger_data["shots"] = kept_shots

    ledger_data["cast"] = [r for r in cast_rows
                           if str(r.get("char_id") or "").strip() not in ids]
    meta = ledger_data.setdefault("meta", {})
    # The news lane keeps its proof map as a LIST of entry payloads, each
    # naming its line_id (`_otr_scifi_news_pro._spoken_row`); a dict keyed by
    # line_id is accepted too. Either way the removed lines' entries go.
    lane_meta = meta.get("scifi_news_pro")
    proof_map = lane_meta.get("proof_map") if isinstance(lane_meta, MutableMapping) else None
    if isinstance(proof_map, MutableMapping):
        for lid in gone_line_ids:
            proof_map.pop(lid, None)
    elif isinstance(proof_map, list):
        lane_meta["proof_map"] = [
            e for e in proof_map
            if not (isinstance(e, Mapping) and str(e.get("line_id") or "") in gone_line_ids)]
    contract = meta.get("cast_contract")
    if isinstance(contract, MutableMapping) and "num_characters_locked" in contract:
        contract["num_characters_locked"] = sum(
            1 for r in ledger_data["cast"]
            if isinstance(r, Mapping) and not is_announcer_cast_row(r))

    receipt["removed"] = [dict(m) for m in missing]
    receipt["line_ids"] = sorted(gone_line_ids)
    receipt["cast_after"] = len(ledger_data["cast"])
    meta["cast_voice_coverage_removed"] = dict(receipt)
    return receipt
