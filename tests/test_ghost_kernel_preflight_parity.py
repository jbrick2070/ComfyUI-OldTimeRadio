"""Cast-time preflight and the render row hand the Ghost kernel the same inputs.

The preflight's temporary shot was `shot_id = beat_id`; the durable row is
`shot_<beat_id>`; the render driver looked the ordinal up by the literal
shot id, so the preflight always resolved at ordinal 0 while
`resolve_crux_kernel` cycles the PLACE by ordinal -- the same object composed
"in the archive" at preflight and "in the yard" on the row. And the beat-text
lookup matched `shot_id` or the line's SCRIPT `beat_id` (a different namespace)
but never `line_id`, which is the key ledger lines actually carry, so the
crux ranking could see no dialogue where the row sees the line.

Now the preflight stamps the prospective plan's ordinal on its temporary shot,
the driver keys the ordinal on the shot's canonical identity, and dialogue is
joined on `line_id` first. This test compares the RESOLVER INPUTS the two
paths produce; source-string assertions cannot establish that equivalence.
"""
from __future__ import annotations

import inspect

from nodes._otr_video_engines import render_driver as RD

LINES = [
    {"line_id": "l001", "text": "The pen sits on the ledger.",
     "beat_id": "l002", "speaker_role": "character"},     # a FOREIGN beat_id
    {"line_id": "l002", "text": "In the yard, a canister.",
     "beat_id": "sb_01", "speaker_role": "character"},
    {"line_id": "l003", "text": "", "speaker_role": "music"},
]
DURABLE = [
    {"shot_id": "shot_l001", "source_line_ids": ["l001"], "role": "character_visual"},
    {"shot_id": "shot_l002", "source_line_ids": ["l002"], "role": "character_visual"},
    {"shot_id": "shot_l003", "source_line_ids": [], "role": "music_visual"},
]


def _ledger():
    return {"lines": [dict(ln) for ln in LINES],
            "video": {"shots": [dict(s) for s in DURABLE]}}


def _preflight_shot(beat_id, ordinal):
    """The temporary shot exactly as `_assert_family_inputs_satisfiable_cast_time`
    builds it: bare beat_id, the ShotLock link, and now the planned ordinal."""
    return {"shot_id": beat_id, "source_line_ids": [beat_id],
            "role": "character_visual", "planned_ordinal": ordinal}


def test_the_preflight_and_the_row_resolve_the_same_ordinal():
    ledger = _ledger()
    for i, row in enumerate(DURABLE):
        temp = _preflight_shot(row["source_line_ids"][0] if row["source_line_ids"]
                               else "l003", i)
        assert RD._planned_ordinal_for_shot(ledger, temp) == i
        assert RD._planned_ordinal_for_shot(ledger, row) == i


def test_an_older_caller_without_the_stamp_still_matches_by_identity():
    """A temporary shot that carries the ShotLock link but no stamped ordinal
    resolves through the plan by identity -- never silently to 0."""
    ledger = _ledger()
    temp = {"shot_id": "l002", "source_line_ids": ["l002"]}
    assert RD._planned_ordinal_for_shot(ledger, temp) == 1
    # and a shot the plan does not list still falls back to 0, as documented
    assert RD._planned_ordinal_for_shot(ledger, {"shot_id": "shot_zzz",
                                                 "source_line_ids": ["zzz"]}) == 0
    assert RD._planned_ordinal_for_shot({}, temp) == 0


def test_the_preflight_and_the_row_see_the_same_dialogue():
    ledger = _ledger()
    temp = _preflight_shot("l002", 1)
    row = ledger["video"]["shots"][1]
    assert RD._beat_text_for_shot(ledger, temp) == RD._beat_text_for_shot(ledger, row)
    assert RD._beat_text_for_shot(ledger, row) == "In the yard, a canister."


def test_a_foreign_script_beat_id_never_leaks_another_line_into_the_text():
    """Line l001 carries the SCRIPT beat_id "l002" -- a different namespace that
    happens to spell a line id. The join is on line_id first, so the l002 shot
    must see only its own line."""
    ledger = _ledger()
    row = ledger["video"]["shots"][1]
    assert RD._beat_text_for_shot(ledger, row) == "In the yard, a canister."
    assert RD._beat_text_for_shot(ledger, ledger["video"]["shots"][0]) == \
        "The pen sits on the ledger."


def test_the_legacy_match_survives_for_a_ledger_without_line_ids():
    ledger = {"lines": [{"beat_id": "b7", "text": "legacy fixture line"}]}
    assert RD._beat_text_for_shot(ledger, {"shot_id": "shot_b7",
                                           "source_line_ids": ["b7"]}) == "legacy fixture line"


def test_the_preflight_call_site_actually_passes_the_ordinal():
    """The wiring, at its real site: the cast-time preflight is called with the
    beat's prospective ordinal, and the temporary shot carries it."""
    from nodes import otr_shot_lock as SL
    body = inspect.getsource(SL._assert_family_inputs_satisfiable_cast_time)
    assert 'shot["planned_ordinal"]' in body, "the preflight never stamps the ordinal"
    # The ONE caller, by name -- a whole-module grep would be satisfied by the
    # signature's own default (cursor, hardening pass 2026-09-12).
    caller = inspect.getsource(SL.build_execution_plan)
    assert "planned_ordinal=planned_ordinal" in caller, (
        "build_execution_plan no longer hands the preflight its ordinal")
