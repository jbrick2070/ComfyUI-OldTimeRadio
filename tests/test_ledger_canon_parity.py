"""tests/test_ledger_canon_parity.py -- story-ledger DRIFT chunk 2.

The deterministic cross-stage consistency guard
(nodes/_otr_ledger_consistency.py). Pins:

  1. the SOURCE-OF-TRUTH matrix covers every non-optional EpisodeCanon field +
     references the StoryContract source fields (a NEW required field with no
     parity row fails here -- the "every non-optional upstream field has a
     populated mapped equivalent" contract).
  2. a GOLDEN GOOD fixture -> 0 defects.
  3. the GOLDEN sound_palette REGRESSION (contract.sound_world set, canon
     sound_palette empty) -> exactly that defect (the class that shipped
     silently for weeks because no canon-parity test existed).
  4. other drift: missing canon field, outline<->canon divergence, an orphan
     line.beat_id, a locked cast slot with no ledger row, a malformed cast row.
  5. evaluate_consistency stamps meta.consistency_status + raises only in strict.

Pure-Python. No GPU, no LLM. Duck-typed fixtures (dict + the real dataclasses).
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_canon import EpisodeCanon  # noqa: E402
from nodes._otr_style_catalog import StoryContract  # noqa: E402
from nodes._otr_ledger_consistency import (  # noqa: E402
    SOURCE_OF_TRUTH_MATRIX,
    ConsistencyError,
    assert_ledger_consistency,
    evaluate_consistency,
)


# ---------------------------------------------------------------------------
# Golden fixtures
# ---------------------------------------------------------------------------
def _contract(sound_world="wind, static, a lonely carrier tone", slug="lost_signal"):
    return StoryContract(
        slug=slug, label="Lost Signal", sound_world=sound_world,
        story_engine="isolation", ending_mode="open", ending_tag="unresolved",
        ending_template="", grammar="",
    )


def _outline():
    return SimpleNamespace(
        title="The Last Transmission",
        premise="A lone operator hears a voice that should not exist.",
        setting="an orbital relay station",
        time_of_day="deep night",
        beats=[SimpleNamespace(beat_id=f"b{i:03d}") for i in range(1, 5)],
    )


def _canon(sound_palette=None):
    return EpisodeCanon(
        title="The Last Transmission",
        premise="A lone operator hears a voice that should not exist.",
        setting="an orbital relay station",
        time_of_day="deep night",
        sound_palette=(["wind", "static", "a lonely carrier tone"]
                       if sound_palette is None else sound_palette),
    )


def _ledger(slug="lost_signal", sound_world="wind, static, a lonely carrier tone"):
    return {
        "meta": {"story_contract": {"slug": slug, "sound_world": sound_world}},
        "cast": [
            {"char_id": "c01", "name": "RIVERA", "speaker_role": "character"},
            {"char_id": "c02", "name": "ANNOUNCER", "speaker_role": "announcer"},
        ],
        "lines": [
            {"line_id": "l001", "beat_id": "b001", "char_id": "c01"},
            {"line_id": "l002", "beat_id": "b002", "char_id": "c01"},
        ],
    }


def _castlock_slots():
    return [
        SimpleNamespace(char_id="c01", name="RIVERA", role="protagonist"),
        SimpleNamespace(char_id="c02", name="ANNOUNCER", role="announcer"),
    ]


# ---------------------------------------------------------------------------
# 1. Matrix reflects the models
# ---------------------------------------------------------------------------
def test_matrix_covers_every_required_canon_field():
    matrix_fields = {r.field for r in SOURCE_OF_TRUTH_MATRIX}
    for f in dataclasses.fields(EpisodeCanon):
        required = (
            f.default is dataclasses.MISSING
            and f.default_factory is dataclasses.MISSING
        )
        if required or f.name == "sound_palette":
            assert f.name in matrix_fields, (
                f"EpisodeCanon.{f.name} has no SOURCE_OF_TRUTH_MATRIX row -- "
                "a non-optional field with no parity rule can drift silently"
            )


def test_matrix_sources_reference_storycontract_fields():
    sources = " ".join(r.source for r in SOURCE_OF_TRUTH_MATRIX)
    # the StoryContract -> canon/ledger seams that the regression exposed
    assert "sound_world" in sources  # contract.sound_world -> canon.sound_palette
    assert "slug" in sources         # contract.slug -> meta.story_contract.slug
    # StoryContract really exposes those fields (guards a rename)
    contract_fields = {f.name for f in dataclasses.fields(StoryContract)}
    assert {"slug", "sound_world"} <= contract_fields


# ---------------------------------------------------------------------------
# 2. Golden GOOD -> clean
# ---------------------------------------------------------------------------
def test_golden_good_fixture_is_clean():
    defects = assert_ledger_consistency(
        contract=_contract(), outline=_outline(),
        castlock=_castlock_slots(), canon=_canon(), ledger=_ledger(),
    )
    assert defects == [], f"unexpected defects on the clean fixture: {defects}"


# ---------------------------------------------------------------------------
# 3. The GOLDEN sound_palette regression
# ---------------------------------------------------------------------------
def test_golden_sound_palette_regression_is_caught():
    # contract carries a rich sound_world, but the canon sound_palette is empty
    # (the exact drift that shipped silently before 2baba3a4).
    defects = assert_ledger_consistency(
        contract=_contract(sound_world="hull pings, sonar, held breath"),
        outline=_outline(), canon=_canon(sound_palette=[]), ledger=_ledger(),
    )
    fields = [d.field for d in defects]
    assert fields == ["sound_palette"], fields
    assert "sound_world" in defects[0].source
    assert "empty" in defects[0].detail


def test_sound_palette_ok_when_contract_has_no_sound_world():
    # no sound_world to propagate -> an empty palette is NOT a defect.
    defects = assert_ledger_consistency(
        contract=_contract(sound_world=""),
        outline=_outline(), canon=_canon(sound_palette=[]), ledger=_ledger(),
    )
    assert [d.field for d in defects] == []


# ---------------------------------------------------------------------------
# 4. Other drift classes
# ---------------------------------------------------------------------------
def test_missing_canon_premise_is_caught():
    canon = _canon()
    object.__setattr__(canon, "premise", "")  # EpisodeCanon is a plain dataclass
    defects = assert_ledger_consistency(
        contract=_contract(), outline=_outline(), canon=canon, ledger=_ledger(),
    )
    assert "premise" in [d.field for d in defects]


def test_outline_canon_divergence_is_caught():
    canon = _canon()
    object.__setattr__(canon, "setting", "a totally different place")
    defects = assert_ledger_consistency(
        outline=_outline(), canon=canon,
    )
    d = [x for x in defects if x.field == "setting"]
    assert d and "diverges" in d[0].detail


def test_orphan_line_beat_id_is_caught():
    led = _ledger()
    led["lines"].append({"line_id": "l003", "beat_id": "b999", "char_id": "c01"})
    defects = assert_ledger_consistency(outline=_outline(), ledger=led)
    bad = [d for d in defects if d.field == "beat_id"]
    assert bad and "b999" in bad[0].found


def test_locked_slot_without_ledger_row_is_caught():
    slots = _castlock_slots()
    slots.append(SimpleNamespace(char_id="c03", name="GHOST", role="character"))
    defects = assert_ledger_consistency(castlock=slots, ledger=_ledger())
    bad = [d for d in defects if d.field == "cast"]
    assert bad and any("c03" in d.expected for d in bad)


def test_malformed_cast_row_is_caught():
    led = _ledger()
    led["cast"].append({"char_id": "", "name": ""})  # missing both
    defects = assert_ledger_consistency(ledger=led)
    assert any(d.field == "cast" for d in defects)


# ---------------------------------------------------------------------------
# 5. The driver: stamps meta.consistency_status; raises only in strict
# ---------------------------------------------------------------------------
def test_evaluate_stamps_status_clean():
    led = _ledger()
    status = evaluate_consistency(
        contract=_contract(), outline=_outline(),
        castlock=_castlock_slots(), canon=_canon(), ledger=led,
    )
    assert status["clean"] is True and status["defect_count"] == 0
    assert led["meta"]["consistency_status"]["clean"] is True


def test_evaluate_nonstrict_warns_does_not_raise():
    led = _ledger()
    # a defect present, but non-strict -> stamp + return, NEVER raise (audio-safe)
    status = evaluate_consistency(
        contract=_contract(sound_world="x, y"), outline=_outline(),
        canon=_canon(sound_palette=[]), ledger=led, strict=False,
    )
    assert status["clean"] is False
    assert led["meta"]["consistency_status"]["defect_count"] >= 1


def test_evaluate_strict_raises_on_defect():
    led = _ledger()
    with pytest.raises(ConsistencyError):
        evaluate_consistency(
            contract=_contract(sound_world="x, y"), outline=_outline(),
            canon=_canon(sound_palette=[]), ledger=led, strict=True,
        )
    # status is still stamped before the raise (never silently pass)
    assert led["meta"]["consistency_status"]["clean"] is False


def test_assert_never_raises_on_garbage_input():
    # duck-typed robustness: nonsense args never raise.
    assert assert_ledger_consistency(123, "x", object(), None, []) is not None


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
