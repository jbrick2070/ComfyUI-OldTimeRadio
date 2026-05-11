"""tests/test_phase0_name_roster.py — Phase 0 name-roster gate.

Covers (per script-writing-architecture synthesis §3 Phase 0):

  * build_allowed_roster   — cast names UPPERCASED + ANNOUNCER + key_terms
  * detect_phantom_names   — three heuristics (ALL-CAPS, titled, Title-Case
                             bigram mid-sentence); roster + speaker skips
  * compose_line           — returns LineResult; flags phantoms via the
                             gate; never rerolls on a name violation
                             (empty / oversize reroll paths preserved)
  * aggregate_compose_flags — kind-count rollup for meta.compose_flag_summary

Pure-Python tests. No GPU. Mock generate_fn for every compose_line case
(no LLM allocation). Unit-scope only — soak / end-to-end is Jeffrey's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the package importable when pytest is invoked from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_line_composer import (  # noqa: E402
    LineRequest,
    LineResult,
    LineCompositionFailedError,
    aggregate_compose_flags,
    build_allowed_roster,
    compose_line,
    detect_phantom_names,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _req(speaker: str, roster=frozenset(), target_words: int = 12) -> LineRequest:
    return LineRequest(
        speaker=speaker,
        intent="reveal the signal",
        mood="tense",
        target_words=target_words,
        canon_header="TITLE: x\nSETTING: y\nTIME: z\nPREMISE: w",
        last_lines=[],
        allowed_roster=roster,
    )


def _const_generate_fn(text: str):
    """Mock generate_fn that always returns the same text."""
    def fn(messages, *, temperature, max_new_tokens):
        return text
    return fn


# ---------------------------------------------------------------------------
# build_allowed_roster
# ---------------------------------------------------------------------------


class TestBuildAllowedRoster:

    def test_includes_uppercase_cast_names(self):
        roster = build_allowed_roster(
            cast_rows=[{"name": "ALICE"}, {"name": "Bob"}, {"name": "carl"}],
        )
        assert "ALICE" in roster
        assert "BOB" in roster
        assert "CARL" in roster

    def test_always_includes_announcer(self):
        roster = build_allowed_roster()
        assert "ANNOUNCER" in roster

    def test_announcer_excludable_for_pure_test_contexts(self):
        roster = build_allowed_roster(include_announcer=False)
        assert "ANNOUNCER" not in roster

    def test_key_terms_merged_uppercased(self):
        roster = build_allowed_roster(
            cast_rows=[{"name": "ALICE"}],
            key_terms=("CERN", "Voyager", "jpl"),
        )
        assert "CERN" in roster
        assert "VOYAGER" in roster
        assert "JPL" in roster

    def test_empty_inputs_yield_announcer_only(self):
        roster = build_allowed_roster()
        assert roster == frozenset({"ANNOUNCER"})

    def test_returns_frozenset(self):
        roster = build_allowed_roster()
        assert isinstance(roster, frozenset)

    def test_accepts_tuple_rows(self):
        roster = build_allowed_roster(cast_rows=[("ALICE",), ("BOB",)])
        assert "ALICE" in roster
        assert "BOB" in roster

    def test_accepts_string_rows(self):
        roster = build_allowed_roster(cast_rows=["ALICE", "BOB"])
        assert "ALICE" in roster

    def test_strips_whitespace_from_terms(self):
        roster = build_allowed_roster(key_terms=("  CERN  ", " "))
        assert "CERN" in roster
        # Bare-whitespace term must not slip in as empty-string.
        assert "" not in roster


# ---------------------------------------------------------------------------
# detect_phantom_names
# ---------------------------------------------------------------------------


@pytest.fixture
def roster():
    return build_allowed_roster(
        cast_rows=[{"name": "ALICE"}, {"name": "BOB"}],
        key_terms=("CERN", "Voyager", "JPL"),
    )


class TestDetectPhantomNames:

    def test_empty_text_returns_empty(self, roster):
        assert detect_phantom_names("", "ALICE", roster) == []

    def test_known_cast_name_not_flagged(self, roster):
        assert detect_phantom_names(
            "ALICE waits in the hall.", "BOB", roster,
        ) == []

    def test_key_term_not_flagged(self, roster):
        assert detect_phantom_names(
            "The CERN team is ready.", "ALICE", roster,
        ) == []
        assert detect_phantom_names(
            "Voyager has been silent for hours.", "BOB", roster,
        ) == []

    def test_titled_phantom_flagged(self, roster):
        flagged = detect_phantom_names(
            "Dr. Patel can confirm the readings.", "ALICE", roster,
        )
        assert flagged == ["Dr. Patel"]

    def test_allcaps_phantom_flagged(self, roster):
        # CARLA is uppercase but not in the roster.
        flagged = detect_phantom_names(
            "CARLA knows what the signal means.", "ALICE", roster,
        )
        assert flagged == ["CARLA"]

    def test_title_case_bigram_mid_sentence_flagged(self, roster):
        # "John Bishop" is a Title-Case bigram in mid-sentence.
        flagged = detect_phantom_names(
            "I told you John Bishop would be the one to find it.",
            "ALICE",
            roster,
        )
        assert "John Bishop" in flagged

    def test_sentence_start_title_case_not_flagged(self, roster):
        # "The radio crackles." — "The radio" starts a sentence; not a name.
        # "The Council" alone (no bigram body) at sentence start: still
        # not flagged because we skip the first word.
        flagged = detect_phantom_names(
            "The radio crackles.", "ALICE", roster,
        )
        assert flagged == []

    def test_speakers_own_name_not_flagged(self, roster):
        # If the model leaked the speaker's own name into the body, the
        # strip pipeline normally removes the leading "ALICE:" but a
        # mention of ALICE in mid-sentence shouldn't auto-flag.
        flagged = detect_phantom_names(
            "ALICE will not back down.", "ALICE", roster,
        )
        assert flagged == []

    def test_common_allcaps_words_not_flagged(self, roster):
        # "OK", "TV", "AI", "USA" — too common to flag.
        for word in ("OK", "TV", "AI", "USA", "UK"):
            assert detect_phantom_names(
                f"It is {word} for now.", "ALICE", roster,
            ) == [], f"{word!r} should not be flagged"

    def test_deduplicates_repeated_phantom(self, roster):
        flagged = detect_phantom_names(
            "Dr. Patel insists. Dr. Patel waits for an answer.",
            "ALICE",
            roster,
        )
        assert flagged == ["Dr. Patel"]

    def test_multiple_phantoms_first_seen_order(self, roster):
        flagged = detect_phantom_names(
            "CARLA mentioned Dr. Patel during the briefing.",
            "ALICE",
            roster,
        )
        assert flagged == ["CARLA", "Dr. Patel"]

    def test_never_raises_on_garbage_input(self, roster):
        # Defensive — shouldn't raise on any of these.
        assert detect_phantom_names("\n\n\n", "ALICE", roster) == []
        assert detect_phantom_names("!!!", "ALICE", roster) == []


# ---------------------------------------------------------------------------
# compose_line + LineResult contract
# ---------------------------------------------------------------------------


class TestComposeLineLineResult:

    def test_clean_line_no_flags(self, roster):
        req = _req("ALICE", roster=roster)
        fn = _const_generate_fn("I found something I cannot explain.")
        res = compose_line(fn, req)
        assert isinstance(res, LineResult)
        assert res.text == "I found something I cannot explain."
        assert res.compose_flags == ()

    def test_phantom_titled_name_flagged(self, roster):
        req = _req("ALICE", roster=roster)
        fn = _const_generate_fn("Dr. Patel can confirm the readings.")
        res = compose_line(fn, req)
        assert res.text == "Dr. Patel can confirm the readings."
        assert res.compose_flags == ("phantom_name:Dr. Patel",)

    def test_phantom_does_not_trigger_reroll(self, roster):
        # Composer commits the line with the phantom intact — no
        # second LLM call. Per §6.A: cast is locked; no reroll path.
        call_count = {"n": 0}

        def fn(messages, *, temperature, max_new_tokens):
            call_count["n"] += 1
            return "CARLA insists the signal is real."

        req = _req("ALICE", roster=roster)
        res = compose_line(fn, req)
        assert call_count["n"] == 1, (
            f"name violation must NOT reroll (per §6.A); "
            f"got {call_count['n']} generate_fn calls"
        )
        assert res.compose_flags == ("phantom_name:CARLA",)

    def test_legit_cast_reference_no_flag(self, roster):
        # ALICE referring to BOB by name (both locked) — clean.
        req = _req("ALICE", roster=roster)
        fn = _const_generate_fn("BOB knows what the signal means.")
        res = compose_line(fn, req)
        assert res.compose_flags == ()

    def test_legit_news_term_no_flag(self, roster):
        # CERN is in roster via key_terms.
        req = _req("ALICE", roster=roster)
        fn = _const_generate_fn("The CERN report changed everything.")
        res = compose_line(fn, req)
        assert res.compose_flags == ()

    def test_empty_roster_gate_skipped(self):
        # Back-compat: a LineRequest without allowed_roster (frozenset())
        # short-circuits the gate entirely; LineResult.compose_flags is
        # always empty.
        req = _req("ALICE", roster=frozenset())
        fn = _const_generate_fn("Dr. Patel says hello.")
        res = compose_line(fn, req)
        assert res.compose_flags == ()

    def test_empty_response_still_rerolls(self, roster):
        # Phase 0 must NOT touch the existing empty-response reroll path.
        call_count = {"n": 0}

        def fn(messages, *, temperature, max_new_tokens):
            call_count["n"] += 1
            return "" if call_count["n"] == 1 else "OK now I see it."

        req = _req("ALICE", roster=roster)
        res = compose_line(fn, req)
        assert call_count["n"] == 2
        assert res.text == "OK now I see it."

    def test_oversize_response_still_rerolls(self, roster):
        # Phase 0 must NOT touch the oversize reroll path either.
        call_count = {"n": 0}

        def fn(messages, *, temperature, max_new_tokens):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return " ".join(["word"] * 200)
            return "Short reply."

        req = _req("ALICE", roster=roster, target_words=10)
        res = compose_line(fn, req)
        assert call_count["n"] == 2
        assert res.text == "Short reply."

    def test_exhaustion_still_raises(self, roster):
        req = _req("ALICE", roster=roster)
        fn = _const_generate_fn("")
        with pytest.raises(LineCompositionFailedError) as exc_info:
            compose_line(fn, req)
        assert exc_info.value.request.speaker == "ALICE"


# ---------------------------------------------------------------------------
# aggregate_compose_flags
# ---------------------------------------------------------------------------


class TestAggregateComposeFlags:

    def test_empty_ledger(self):
        assert aggregate_compose_flags({}) == {}

    def test_empty_lines(self):
        assert aggregate_compose_flags({"lines": []}) == {}

    def test_rolls_up_phantom_name_kind(self):
        ledger = {"lines": [
            {"line_id": "b001",
             "compose_flags": ["phantom_name:Dr. Patel"]},
            {"line_id": "b002",
             "compose_flags": ["phantom_name:CARLA",
                                "phantom_name:Dr. Patel"]},
            {"line_id": "b003", "compose_flags": []},
            {"line_id": "b004"},  # field absent
        ]}
        assert aggregate_compose_flags(ledger) == {"phantom_name": 3}

    def test_multiple_kinds(self):
        ledger = {"lines": [
            {"line_id": "b001",
             "compose_flags": ["phantom_name:CARLA", "some_other_kind:foo"]},
            {"line_id": "b002",
             "compose_flags": ["some_other_kind:bar"]},
        ]}
        assert aggregate_compose_flags(ledger) == {
            "phantom_name": 1,
            "some_other_kind": 2,
        }

    def test_never_raises_on_garbage(self):
        assert aggregate_compose_flags(None) == {}  # type: ignore[arg-type]
        assert aggregate_compose_flags({"lines": [None, "garbage"]}) == {}
        # Row with non-string flag is coerced via str() — the kind is
        # whatever falls out before the first ":".
        ledger = {"lines": [{"compose_flags": [42]}]}
        # str(42) = "42", split on ":" yields "42" — counts under "42".
        assert aggregate_compose_flags(ledger) == {"42": 1}
