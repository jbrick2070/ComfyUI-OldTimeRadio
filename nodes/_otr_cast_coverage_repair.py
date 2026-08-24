"""Repair a silent cast slot BEFORE the freeze gate has to refuse it.

THE DEFECT (PBUG-20260802-02, third manifestation, shakespeare/MARIA,
2026-08-24). A composition pass can allocate dialogue for some cast members
and not others under a tight beat budget -- the writer locks a cast (e.g.
Shakespeare's per-scene ``cast_hints``), but Stage 2's outline check
(``_otr_outline._phase_check``) only asks "did an invented speaker sneak in",
never the reverse ("did every locked member get a beat"). A locked member can
legitimately receive zero beats. The freeze cascade's universal backstop
(``_otr_ledger_freeze.cast_coverage_gaps``, wired into
``_check_per_cast_invariants``) always catches this -- correctly, and it must
keep refusing on failure -- but nothing before it ever tried to fix it.

THE STANDING PHILOSOPHY THIS RESPECTS (``_otr_cast_voice_coverage.py``):
gates only ever REFUSE, loudly and by name; repair is the PRODUCER's job,
never post-ledger surgery. This module runs INSIDE the writer's own tail
(``_otr_writer_tail.py``, after the clean-transaction window settles and
before the freeze cascade node ever runs), so by the time a gate might see
this ledger, the gap is either genuinely filled or genuinely absent -- never
patched around after the fact.

THE REPAIR ITSELF is ONE bounded ``compose_line`` call per gap character,
reusing the exact same seeded ``creative_fn`` slot every other line in the
episode already goes through -- no new sampling mechanism. On failure the row
is left EXACTLY as it already was: byte-identical to today's refuse-and-halt.
This can only ever improve the success case.

TWO MODES, both mechanical facts of the ledger shape
(``production_ledger.init_lines_from_outline`` stamps ``line_id == beat_id``
1:1 with every outline beat before composition runs):
  * MODE 1 -- a ``lines[]`` row exists for the character (composition ran,
    produced nothing sayable, writer-tail cleanup already flagged
    ``skip=True``). Retry that exact slot.
  * MODE 2 -- Stage 2 never allocated the character a beat at all, so no
    ``lines[]``/``beats[]`` row references them. Mint one new row, in the
    ledger only -- it is NEVER appended to the pydantic ``Outline.beats``, so
    no outline validator is retroactively re-run over state it already
    approved.

FIDELITY GRAFT, shakespeare-specific but not shakespeare-exclusive: when
``meta.source_meta.cast_hints_presence`` names a real attested speech for the
gap character (``_otr_shakespeare_sources.cast_presence_from_text``), it
rides into ``LineRequest.source_block`` so the repair is grounded in the
source's own words -- honoring the fidelity-lane contract ("the author's own
language is carried as written") instead of merely working around its
absence. Any bank without that data simply composes without a source_block,
exactly like every other line in that lane already does.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from . import _otr_ledger as _OTRL
from . import _otr_ledger_freeze as _OTRLF
from . import _otr_line_composer as _OTRLC

SCHEMA_VERSION = "cast_coverage_repair_v1"

#: Guidance text for a repaired slot. Not narrated to the listener -- it is
#: prompt context only, the same role LineRequest.intent always plays.
_REPAIR_INTENT = (
    "Speak now -- give this character one real, in-character line that "
    "establishes their presence in the scene."
)


def _next_beat_id(ledger_data: dict) -> str:
    """The next ``bNNN`` id, scanned from the settled ledger -- collision-safe
    against every existing lines[]/beats[] row. Format matches
    ``_otr_outline.py``'s own minting (``f"b{n:03d}"``); this pass runs after
    outline generation finished, so it cannot reuse that stage's in-memory
    counter and must re-derive the floor from what actually landed.
    """
    max_n = 0
    for key in ("lines", "beats"):
        for row in ledger_data.get(key) or []:
            if not isinstance(row, dict):
                continue
            bid = str(row.get("beat_id") or "")
            if bid.startswith("b") and bid[1:].isdigit():
                max_n = max(max_n, int(bid[1:]))
    return f"b{max_n + 1:03d}"


def _last_lines(ledger_data: dict, *, limit: int = 2) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for row in ledger_data.get("lines") or []:
        if not isinstance(row, dict) or row.get("skip"):
            continue
        text = str(row.get("text") or "")
        if not text:
            continue
        out.append((str(row.get("speaker") or ""), text))
    return out[-limit:]


def _source_block_for(name: str, presence: dict, scene_label: str) -> str:
    entry = presence.get(name) or presence.get(name.upper())
    if not isinstance(entry, dict):
        return ""
    speech = str(entry.get("first_speech") or "").strip()
    if not speech:
        return ""
    return (
        f"SOURCE (verbatim, {scene_label}, spoken by {name}):\n{speech}"
    )


def repair_zero_coverage_cast(
    led: Any,
    *,
    creative_fn: Callable,
    canon_header: str,
    style_descriptor: str,
    source_bank_id: str,
    meta: dict,
    creative_repo_id: Optional[str] = None,
    slot_scheduler: Any = None,
) -> dict:
    """Fill every zero-coverage cast slot found in ``led.data``, in place.

    Returns the ``meta.cast_coverage_repair`` receipt: ``schema_version``,
    ``attempted``/``repaired``/``failed`` char_id lists. Never raises --
    a ``compose_line`` failure is recorded in ``failed`` and the row is left
    untouched; the freeze gate downstream is the one thing allowed to refuse.
    """
    receipt: dict = {
        "schema_version": SCHEMA_VERSION,
        "attempted": [],
        "repaired": [],
        "failed": [],
    }
    data = led.data if hasattr(led, "data") else led
    gaps = _OTRLF.cast_coverage_gaps(data)
    if not gaps:
        return receipt

    cast_by_id = {
        row.get("char_id"): row
        for row in data.get("cast") or []
        if isinstance(row, dict)
    }
    source_meta = meta.get("source_meta") or {}
    presence = source_meta.get("cast_hints_presence") or {}
    scene_label = str(source_meta.get("scene_label") or "").strip()

    def _helper_ctx(name: str):
        if slot_scheduler is not None and hasattr(slot_scheduler, "helper_context"):
            return slot_scheduler.helper_context(name)
        import contextlib
        return contextlib.nullcontext()

    for char_id, name in gaps:
        receipt["attempted"].append(char_id)
        row = cast_by_id.get(char_id) or {}
        voice_card = str(
            row.get("traits") or row.get("character_description") or name
        )
        matching = [
            ln for ln in (data.get("lines") or [])
            if isinstance(ln, dict) and ln.get("char_id") == char_id
        ]

        if matching:
            # MODE 1 -- an existing slot produced nothing sayable.
            target = matching[0]
            line_id = str(target.get("line_id") or target.get("beat_id") or "")
            intent = str(target.get("beat_intent") or "") or _REPAIR_INTENT
            arc_phase = str(target.get("arc_phase") or "")
            mood = str(target.get("traits") or "")
        else:
            # MODE 2 -- Stage 2 never allocated this character a beat.
            # Ledger-only: never appended to the pydantic Outline.beats, so
            # no outline validator is re-run over state it already approved.
            beat_id = _next_beat_id(data)
            arc_phase = ""
            for ln in reversed(data.get("lines") or []):
                if isinstance(ln, dict) and not ln.get("skip"):
                    arc_phase = str(ln.get("arc_phase") or "")
                    break
            new_line = {
                "line_id": beat_id, "shot_id": None, "beat_id": beat_id,
                "speaker": name or None, "char_id": char_id, "text": "",
                "traits": None, "boundary": None, "char_count": 0,
                "word_count": 0, "bark_wav_path": None, "start_s": None,
                "dur_s": None, "speaker_role": "character",
                "arc_phase": arc_phase or None, "compose_flags": [],
                "beat_intent": _REPAIR_INTENT, "target_words": None,
                "dialogue_slot_id": None,
            }
            data.setdefault("lines", []).append(new_line)
            data.setdefault("beats", []).append({
                "beat_id": beat_id, "shot_id": None, "scene_id": None,
                "speaker": name or None, "char_id": char_id,
                "line_ids": [beat_id], "start_s": None, "dur_s": None,
            })
            line_id = beat_id
            intent = _REPAIR_INTENT
            mood = ""

        req = _OTRLC.LineRequest(
            speaker=name,
            intent=intent,
            mood=mood,
            canon_header=canon_header,
            last_lines=_last_lines(data),
            style_descriptor=style_descriptor,
            character_voice_card=voice_card,
            arc_phase=arc_phase,
            source_block=_source_block_for(name, presence, scene_label),
            speaker_role="character",
        )
        try:
            with _helper_ctx("cast_coverage_repair"):
                result = _OTRLC.compose_line(
                    creative_fn=creative_fn,
                    req=req,
                    creative_repo_id=creative_repo_id,
                    source_bank_id=source_bank_id,
                )
        except _OTRLC.LineCompositionFailedError:
            receipt["failed"].append(char_id)
            continue

        if not result.text.strip():
            receipt["failed"].append(char_id)
            continue

        _OTRL.patch_line_text(data, line_id, result.text)
        _OTRL.patch_line_fields(data, line_id, {
            "skip": False,
            "tts_skip_reason": None,
            "compose_flags": list(result.compose_flags) + ["cast_coverage_repair"],
        })
        receipt["repaired"].append(char_id)

    return receipt


__all__ = ["SCHEMA_VERSION", "repair_zero_coverage_cast"]
