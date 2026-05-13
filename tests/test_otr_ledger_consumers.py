"""tests/test_otr_ledger_consumers.py -- helper API tests for `_otr_ledger_consumers.py`.

Covers the five public read-side helpers used by every L3 consumer:

    load_ledger, iter_lines, cast_lookup, speaker_name,
    voice_preset

(production_plan_or_empty was deleted in voice-path-cleanbreak
S23.6 along with its TestProductionPlanOrEmpty test class.)

These tests treat the helper module as the contract. If a future schema
bump (l4) changes ledger shape, the failures land here first instead of
manifesting deep inside one of the seven shipped consumers.

Hermetic: no GPU, no I/O, no ComfyUI imports. Uses
`tests/fixtures/ledger_stub.py` for the canonical stub ledger shape.
Companion to the seven per-consumer self-tests in `tests/test_*_ledger.py`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make `nodes/` importable without ComfyUI's package path. Mirrors the
# pattern used by the seven per-consumer self-tests.
_NODES_DIR = Path(__file__).resolve().parent.parent / "nodes"
if str(_NODES_DIR) not in sys.path:
    sys.path.insert(0, str(_NODES_DIR))

import _otr_ledger_consumers as _OTRLC  # noqa: E402

from tests.fixtures.ledger_stub import (  # noqa: E402
    make_stub_ledger,
    make_legacy_list,
)


# ---------------------------------------------------------------------------
# load_ledger
# ---------------------------------------------------------------------------

class TestLoadLedger:
    """`load_ledger` is the loud-fail boundary for legacy wirings.

    Strict on shape: dict in -> dict out. Anything else raises a named
    ValueError so the consumer halts at the consumer boundary instead
    of silently producing half-degraded audio mid-soak.
    """

    def test_dict_input_returns_dict(self):
        led_in = make_stub_ledger()
        out = _OTRLC.load_ledger(json.dumps(led_in))
        assert isinstance(out, dict)
        assert out["cast"][0]["char_id"] == "c01"
        assert len(out["lines"]) == 6  # default with_sfx + with_music = True

    def test_legacy_list_input_raises_value_error(self):
        legacy = make_legacy_list()
        with pytest.raises(ValueError) as exc_info:
            _OTRLC.load_ledger(json.dumps(legacy))
        # Message must point at the right wiring fix per ROADMAP
        # patterns doc Pattern 1 contract.
        assert "legacy parser-list" in str(exc_info.value)
        assert "OTR_LedgerScriptWriter" in str(exc_info.value)

    def test_non_dict_non_list_raises_value_error(self):
        # JSON string root, JSON int root, JSON null root -- all bad shape.
        for bad_payload in ('"plain string"', "42", "null", "true"):
            with pytest.raises(ValueError) as exc_info:
                _OTRLC.load_ledger(bad_payload)
            assert "expected ledger dict" in str(exc_info.value)

    def test_invalid_json_propagates(self):
        # `json.JSONDecodeError` IS a `ValueError` subclass, so the
        # raise contract is preserved.
        with pytest.raises(ValueError):
            _OTRLC.load_ledger("{ not json")

    def test_empty_dict_input_returns_empty_dict(self):
        # Strict on shape, NOT strict on completeness. An empty dict is
        # a valid (if degenerate) ledger; downstream `.get("lines")`
        # / `.get("cast")` calls handle the missing arrays.
        out = _OTRLC.load_ledger("{}")
        assert out == {}


# ---------------------------------------------------------------------------
# iter_lines
# ---------------------------------------------------------------------------

class TestIterLines:
    """`iter_lines` is the canonical role-filter walk over `ledger['lines']`.

    The role-filter judgment rule (ROADMAP patterns doc, Pattern 2):
    `roles=None` yields every line; a set widens / narrows the walk.
    File-level architectural comments outrank architect specs when
    they conflict; consumers MUST NOT widen role filters past their
    documented behavior without checking the in-file rationale.
    """

    def test_no_filter_yields_every_line(self):
        led = make_stub_ledger()
        out = list(_OTRLC.iter_lines(led))
        assert len(out) == 6
        # Original order is preserved -- iter_lines is NOT a sort.
        assert [ln["line_id"] for ln in out] == [
            "b001", "b002", "b003", "b004", "b005", "b006"
        ]

    def test_character_role_filter_yields_dialogue_only(self):
        led = make_stub_ledger()
        out = list(_OTRLC.iter_lines(led, roles={"character"}))
        assert len(out) == 2
        assert all(ln["speaker_role"] == "character" for ln in out)
        assert {ln["char_id"] for ln in out} == {"c01", "c02"}

    def test_announcer_role_filter_yields_announcer_only(self):
        led = make_stub_ledger()
        out = list(_OTRLC.iter_lines(led, roles={"announcer"}))
        assert len(out) == 1
        assert out[0]["line_id"] == "b002"

    def test_multi_role_filter_yields_union(self):
        led = make_stub_ledger()
        out = list(_OTRLC.iter_lines(led, roles={"character", "announcer"}))
        # 2 character + 1 announcer
        assert len(out) == 3
        assert {ln["speaker_role"] for ln in out} == {"character", "announcer"}

    def test_empty_role_filter_yields_nothing(self):
        led = make_stub_ledger()
        out = list(_OTRLC.iter_lines(led, roles=set()))
        # Empty set means no allowed roles -> empty iteration.
        assert out == []

    def test_unknown_role_skipped_when_filtered(self):
        led = make_stub_ledger()
        # Inject a line with no `speaker_role` key.
        led["lines"].append({
            "line_id":  "b099",
            "char_id":  "?",
            "text":     "stray",
        })
        out = list(_OTRLC.iter_lines(led, roles={"character"}))
        # Stray line skipped because its role doesn't match the filter.
        assert all(ln.get("line_id") != "b099" for ln in out)

    def test_unknown_role_yielded_when_no_filter(self):
        led = make_stub_ledger()
        led["lines"].append({"line_id": "b099", "text": "stray"})
        out = list(_OTRLC.iter_lines(led))
        # No filter -> stray included.
        assert any(ln.get("line_id") == "b099" for ln in out)

    def test_missing_lines_key_yields_empty(self):
        led = {"cast": []}
        assert list(_OTRLC.iter_lines(led)) == []

    def test_lines_value_none_yields_empty(self):
        led = {"lines": None, "cast": []}
        assert list(_OTRLC.iter_lines(led)) == []


# ---------------------------------------------------------------------------
# cast_lookup
# ---------------------------------------------------------------------------

class TestCastLookup:
    """`cast_lookup(led, char_id)` walks the LIST-shaped `ledger['cast']`
    and returns the entry whose `char_id` matches.

    The cast is a LIST of dicts (each carrying its own `char_id`),
    NOT a dict keyed by char_id -- this matches the on-disk shape that
    `production_ledger.py:set_cast` stamps.
    """

    def test_known_char_id_returns_entry(self):
        led = make_stub_ledger()
        entry = _OTRLC.cast_lookup(led, "c01")
        assert entry["name"] == "LEMMY"
        assert entry["voice_preset"] == "v2/en_speaker_3"

    def test_second_char_id_returns_correct_entry(self):
        # Verifies the walk doesn't short-circuit on the first entry.
        led = make_stub_ledger()
        entry = _OTRLC.cast_lookup(led, "c02")
        assert entry["name"] == "ASTRA"
        assert entry["voice_preset"] == "v2/en_speaker_9"

    def test_unknown_char_id_returns_empty_dict(self):
        led = make_stub_ledger()
        assert _OTRLC.cast_lookup(led, "c99") == {}

    def test_empty_char_id_returns_empty_dict(self):
        led = make_stub_ledger()
        assert _OTRLC.cast_lookup(led, "") == {}

    def test_none_char_id_returns_empty_dict(self):
        led = make_stub_ledger()
        # Defensive: announcer/sfx/music line `char_id` is a role tag,
        # NOT a real cast member; passing None must degrade cleanly.
        assert _OTRLC.cast_lookup(led, None) == {}

    def test_missing_cast_key_returns_empty_dict(self):
        led = {"lines": []}
        assert _OTRLC.cast_lookup(led, "c01") == {}

    def test_cast_value_none_returns_empty_dict(self):
        led = {"cast": None}
        assert _OTRLC.cast_lookup(led, "c01") == {}

    def test_cast_with_non_dict_entries_skips_safely(self):
        # A malformed cast row (None, list, etc.) must not crash the
        # walk; the helper is hostile-input safe.
        led = {
            "cast": [
                None,
                "not a dict",
                {"char_id": "c01", "name": "LEMMY"},
            ],
        }
        entry = _OTRLC.cast_lookup(led, "c01")
        assert entry["name"] == "LEMMY"

    def test_char_id_string_coercion(self):
        # If the cast stamps char_id as int by accident, the helper
        # still resolves a string lookup (str-coercion on both sides).
        led = {"cast": [{"char_id": 1, "name": "X"}]}
        assert _OTRLC.cast_lookup(led, "1")["name"] == "X"


# ---------------------------------------------------------------------------
# speaker_name
# ---------------------------------------------------------------------------

class TestSpeakerName:
    """`speaker_name(led, line)` resolves a line's `char_id` to its cast
    `name`. Returns ``"UNKNOWN"`` on miss so consumers' log lines stay
    greppable when an orphan line slips through.
    """

    def test_character_line_returns_cast_name(self):
        led = make_stub_ledger()
        line = next(
            ln for ln in led["lines"]
            if ln.get("char_id") == "c01"
        )
        assert _OTRLC.speaker_name(led, line) == "LEMMY"

    def test_announcer_line_returns_unknown(self):
        # announcer's `char_id` is the role tag "announcer", which has
        # no entry in cast[]. The helper degrades to UNKNOWN so the
        # consumer logs are clean.
        led = make_stub_ledger()
        line = next(
            ln for ln in led["lines"]
            if ln.get("speaker_role") == "announcer"
        )
        assert _OTRLC.speaker_name(led, line) == "UNKNOWN"

    def test_sfx_line_returns_unknown(self):
        led = make_stub_ledger()
        line = next(
            ln for ln in led["lines"]
            if ln.get("speaker_role") == "sfx"
        )
        assert _OTRLC.speaker_name(led, line) == "UNKNOWN"

    def test_missing_char_id_returns_unknown(self):
        led = make_stub_ledger()
        line = {"line_id": "b099", "text": "no char_id"}
        assert _OTRLC.speaker_name(led, line) == "UNKNOWN"

    def test_none_line_returns_unknown(self):
        led = make_stub_ledger()
        assert _OTRLC.speaker_name(led, None) == "UNKNOWN"

    def test_empty_dict_line_returns_unknown(self):
        led = make_stub_ledger()
        assert _OTRLC.speaker_name(led, {}) == "UNKNOWN"

    def test_cast_entry_with_no_name_returns_unknown(self):
        # Defensive: a cast entry that exists but lacks `name` shouldn't
        # raise; the helper returns UNKNOWN.
        led = {
            "cast": [{"char_id": "c01"}],   # no name
            "lines": [],
        }
        line = {"char_id": "c01", "text": "x"}
        assert _OTRLC.speaker_name(led, line) == "UNKNOWN"


# ---------------------------------------------------------------------------
# voice_preset
# ---------------------------------------------------------------------------

class TestVoicePreset:
    """`voice_preset(led, line)` resolves a line's `char_id` to its cast
    `voice_preset`. Returns ``None`` on miss so the consumer can fall back
    to its own gender-aware hash (Bark) or seeded grab-bag (Kokoro).
    """

    def test_known_char_id_returns_preset(self):
        led = make_stub_ledger()
        line = {"char_id": "c01", "text": "x"}
        assert _OTRLC.voice_preset(led, line) == "v2/en_speaker_3"

    def test_announcer_line_returns_none(self):
        # announcer's char_id is the role tag, not a cast member.
        led = make_stub_ledger()
        line = next(
            ln for ln in led["lines"]
            if ln.get("speaker_role") == "announcer"
        )
        assert _OTRLC.voice_preset(led, line) is None

    def test_unknown_char_id_returns_none(self):
        led = make_stub_ledger()
        line = {"char_id": "c99", "text": "x"}
        assert _OTRLC.voice_preset(led, line) is None

    def test_missing_char_id_returns_none(self):
        led = make_stub_ledger()
        line = {"line_id": "b099", "text": "x"}
        assert _OTRLC.voice_preset(led, line) is None

    def test_none_line_returns_none(self):
        led = make_stub_ledger()
        assert _OTRLC.voice_preset(led, None) is None

    def test_cast_entry_with_no_preset_returns_none(self):
        led = {
            "cast": [{"char_id": "c01", "name": "LEMMY"}],   # no voice_preset
            "lines": [],
        }
        line = {"char_id": "c01", "text": "x"}
        assert _OTRLC.voice_preset(led, line) is None


# production_plan_or_empty + TestProductionPlanOrEmpty were removed in
# voice-path-cleanbreak S23.6 along with the helper they tested.


# ---------------------------------------------------------------------------
# Cross-helper composition
# ---------------------------------------------------------------------------

class TestComposition:
    """Tests that validate the canonical Pattern 2 walk shape used by
    every shipped consumer. Mirrors the sequence from the patterns doc:

        led = load_ledger(script_json)
        for line in iter_lines(led, roles={"character"}):
            text     = (line.get("text") or "").strip()
            line_id  = line.get("line_id")
            name     = speaker_name(led, line)
            preset   = voice_preset(led, line)
    """

    def test_pattern_2_walk_resolves_every_field(self):
        led_in = make_stub_ledger()
        led = _OTRLC.load_ledger(json.dumps(led_in))
        rows = []
        for line in _OTRLC.iter_lines(led, roles={"character"}):
            rows.append({
                "line_id": line.get("line_id"),
                "text":    (line.get("text") or "").strip(),
                "name":    _OTRLC.speaker_name(led, line),
                "preset":  _OTRLC.voice_preset(led, line),
            })
        assert rows == [
            {
                "line_id": "b003",
                "text":    "Get the kit out, fast.",
                "name":    "LEMMY",
                "preset":  "v2/en_speaker_3",
            },
            {
                "line_id": "b004",
                "text":    "On it.",
                "name":    "ASTRA",
                "preset":  "v2/en_speaker_9",
            },
        ]

    def test_announcer_walk_skips_character_lines(self):
        led_in = make_stub_ledger()
        led = _OTRLC.load_ledger(json.dumps(led_in))
        rows = list(_OTRLC.iter_lines(led, roles={"announcer"}))
        assert len(rows) == 1
        assert rows[0]["text"] == "From the studio, tonight's broadcast."
        # announcer's char_id is the role tag, so cast lookup misses
        # gracefully -> name=UNKNOWN, preset=None. This is the doc'd
        # contract Bark+Kokoro rely on.
        assert _OTRLC.speaker_name(led, rows[0]) == "UNKNOWN"
        assert _OTRLC.voice_preset(led, rows[0]) is None

    def test_legacy_list_short_circuits_pattern_2(self):
        legacy_json = json.dumps(make_legacy_list())
        # Pattern 1 contract: `load_ledger` raises ValueError on legacy
        # list; the consumer is expected to either propagate (loud
        # posture: Bark/Kokoro/Sequencer/AudioGen/ProcSFX/Video) or
        # catch + log + passthrough (non-blocking posture: Critic).
        with pytest.raises(ValueError):
            _OTRLC.load_ledger(legacy_json)
