"""v4 campaign P1(iii): bank-aware GENRE / spoken-text boundary guard.

Covers the three pieces + their safety contract:
  * the boundary matcher (casefolded, Unicode-aware, punctuation + plural) --
    "gun" never fires on "begun"; "spaceship" catches "spaceships";
  * the deterministic Phase-10 G10 terminal (opt-in via meta.genre_banned_phrases;
    INERT for every current bank; fires only on a surviving hit -- THE LAW);
  * the writer-boundary authored repair (creative slot; keep-if-clean; never
    raises; stamps a compose-flag breadcrumb).

Canary across execution families is by construction: G10 lives in
`run_gap_audit`, the one path every family crosses (codex phase_10 finalizer,
inline run_freeze_cascade, fable2 finalizer). Pure / CPU. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_genre_guard as GG  # noqa: E402
from nodes import _otr_ledger_freeze as LF  # noqa: E402
from nodes import _otr_story_routing as SR  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Boundary matcher
# ---------------------------------------------------------------------------

class TestBoundaryMatcher:
    def test_word_boundary_no_false_positive(self):
        assert GG.find_banned_phrase_hits("we had begun the work", ["gun"]) == []
        assert GG.find_banned_phrase_hits("a gunmetal sky", ["gun"]) == []
        assert GG.find_banned_phrase_hits("he fired the shotgun", ["gun"]) == []

    def test_matches_standalone_word_casefolded(self):
        assert GG.find_banned_phrase_hits("hand me the gun.", ["gun"]) == ["gun"]
        assert GG.find_banned_phrase_hits("A GUN!", ["gun"]) == ["gun"]
        assert GG.find_banned_phrase_hits("Gun", ["gun"]) == ["gun"]

    def test_simple_plural_single_token(self):
        assert GG.find_banned_phrase_hits(
            "two spaceships landed", ["spaceship"]) == ["spaceship"]
        assert GG.find_banned_phrase_hits("the guns", ["gun"]) == ["gun"]

    def test_punctuation_adjacent(self):
        for frag in ["(gun)", "gun,", "gun-metal grey", "gun! now", "'gun'"]:
            assert GG.find_banned_phrase_hits(frag, ["gun"]) == ["gun"], frag

    def test_multiword_phrase_flexible_whitespace(self):
        assert GG.find_banned_phrase_hits(
            "to Mission   Control now", ["mission control"]) == ["mission control"]
        assert GG.find_banned_phrase_hits(
            "missioncontrol", ["mission control"]) == []

    def test_short_token_not_substring(self):
        # RSS must not fire inside another word.
        assert GG.find_banned_phrase_hits("the grass is green", ["RSS"]) == []
        assert GG.find_banned_phrase_hits("via RSS feed", ["RSS"]) == ["RSS"]

    def test_unicode_word_boundary(self):
        # accented letters are word chars, so 'sume' inside 'resume' (accented)
        # is bounded and must NOT match.
        assert GG.find_banned_phrase_hits("résumé sent", ["sumé"]) == []
        assert GG.find_banned_phrase_hits(
            "a café, warm", ["café"]) == ["café"]

    def test_dedupes_and_preserves_order(self):
        hits = GG.find_banned_phrase_hits("gun and knife", ["knife", "gun", "gun"])
        assert hits == ["knife", "gun"]

    def test_empty_and_nonstring_safe(self):
        assert GG.find_banned_phrase_hits("", ["gun"]) == []
        assert GG.find_banned_phrase_hits(None, ["gun"]) == []
        assert GG.find_banned_phrase_hits("gun", []) == []
        assert GG.find_banned_phrase_hits("gun", None) == []


# ---------------------------------------------------------------------------
# 2. Spoken scope (music rows excluded)
# ---------------------------------------------------------------------------

class TestSpokenScope:
    def test_is_spoken_role(self):
        assert GG.is_spoken_role("announcer")
        assert GG.is_spoken_role("character")
        assert GG.is_spoken_role("")            # untyped -> conservative in-scope
        assert not GG.is_spoken_role("music_inter")
        assert not GG.is_spoken_role("music_open")
        assert not GG.is_spoken_role("MUSIC")

    def test_scan_skips_music_rows(self):
        led = {"lines": [
            {"line_id": "m1", "speaker_role": "music_inter", "text": "spaceship dawn"},
            {"line_id": "b1", "speaker_role": "character", "text": "clear skies"},
        ]}
        assert GG.scan_spoken_lines(led, ["spaceship"]) == []

    def test_scan_reports_spoken_hit(self):
        led = {"lines": [
            {"line_id": "b1", "speaker_role": "character", "text": "a spaceship!"},
        ]}
        assert GG.scan_spoken_lines(led, ["spaceship"]) == [("b1", ["spaceship"])]


# ---------------------------------------------------------------------------
# 3. Parser validation of the opt-in flag
# ---------------------------------------------------------------------------

class TestParserValidation:
    @staticmethod
    def _row(defaults):
        return {
            "source_bank_id": "t", "label": "t", "source_kind": "custom",
            "interpreter": "", "fetcher": "",
            "default_story_model": "m", "default_story_pipeline": "p",
            "defaults": defaults, "required_seams": [], "runnable": False,
            "guide_ref": "",
        }

    def test_non_bool_genre_guard_rejected(self):
        with pytest.raises(SR.RegistryValidationError):
            SR._parse_bank(self._row({"genre_guard_spoken": "yes"}), "test")

    def test_bool_genre_guard_accepted(self):
        b = SR._parse_bank(self._row({"genre_guard_spoken": True}), "test")
        assert b.defaults["genre_guard_spoken"] is True

    def test_absent_genre_guard_ok(self):
        b = SR._parse_bank(self._row({}), "test")
        assert "genre_guard_spoken" not in b.defaults


# ---------------------------------------------------------------------------
# 4. Every current runnable bank is INERT (no opt-in) -> suite-safe rollout
# ---------------------------------------------------------------------------

_CURRENT_BANKS = [
    "media_archive", "original_radio", "scifi_news_pro",
    "public_domain_story_v3", "shakespeare",
]


class TestCurrentBanksInert:
    @pytest.mark.parametrize("bank_id", _CURRENT_BANKS)
    def test_no_current_bank_opts_in(self, bank_id):
        bank = SR.require_runnable_bank(bank_id)
        assert not (bank.defaults or {}).get("genre_guard_spoken")


# ---------------------------------------------------------------------------
# 5. Deterministic Phase-10 G10 terminal
# ---------------------------------------------------------------------------

def _led(phrases, lines):
    """A ledger dict; when `phrases` is not None the bank has opted in."""
    meta = {"source_bank": "x", "schema_version": "l3-2026-05-14"}
    if phrases is not None:
        meta["genre_guard_spoken"] = True
        meta["genre_banned_phrases"] = list(phrases)
    return {"schema_version": "l3-2026-05-14", "meta": meta, "lines": lines}


def _g10_errors(ledger_data):
    errors: list = []
    LF._check_g10_genre_spoken_text(ledger_data, errors, [])
    return errors


class TestG10Terminal:
    def test_inert_when_not_opted_in(self):
        # A banned word on the microphone but NO opt-in -> G10 silent.
        led = _led(None, [
            {"line_id": "b1", "speaker_role": "character", "text": "a spaceship"},
        ])
        assert _g10_errors(led) == []

    def test_fires_on_surviving_hit(self):
        led = _led(["spaceship"], [
            {"line_id": "b1", "speaker_role": "character", "text": "a spaceship!"},
        ])
        errs = _g10_errors(led)
        assert errs and "G10" in errs[0] and "b1" in errs[0]

    def test_boundary_no_false_positive(self):
        led = _led(["gun"], [
            {"line_id": "b1", "speaker_role": "character", "text": "we had begun"},
        ])
        assert _g10_errors(led) == []

    def test_skips_music_rows(self):
        led = _led(["spaceship"], [
            {"line_id": "m1", "speaker_role": "music_inter", "text": "spaceship dawn"},
        ])
        assert _g10_errors(led) == []

    def test_announcer_rows_in_scope(self):
        led = _led(["spaceship"], [
            {"line_id": "b1", "speaker_role": "announcer", "text": "a spaceship tale"},
        ])
        assert _g10_errors(led) != []

    def test_run_gap_audit_includes_g10(self):
        # run_gap_audit runs G10 identically for either label; use "pre" so the
        # test file carries no literal retired-Phase-9 label shape (S33 sweep).
        # The Phase-10 RAISE path is proved separately below.
        led = _led(["spaceship"], [
            {"line_id": "b1", "speaker_role": "character", "text": "the spaceship"},
        ])
        report = LF.run_gap_audit(led, label="pre")
        assert any("G10" in e for e in report.errors)

    def test_run_gap_audit_no_g10_when_inert(self):
        led = _led(None, [
            {"line_id": "b1", "speaker_role": "character", "text": "the spaceship"},
        ])
        report = LF.run_gap_audit(led, label="pre")
        assert not any("G10" in e for e in report.errors)

    def test_phase_10_raises_with_g10(self):
        led = _led(["spaceship"], [
            {"line_id": "b1", "speaker_role": "character", "text": "the spaceship"},
        ])
        with pytest.raises(LF.FreezeAssertionError) as ei:
            LF.phase_10_gap_audit_post_and_freeze(led)
        assert any("G10" in e for e in ei.value.errors)


# ---------------------------------------------------------------------------
# 6. Writer-boundary authored repair
# ---------------------------------------------------------------------------

def _fn_returns(value):
    def _f(messages, *, temperature=None, max_new_tokens=None, stop=None):
        return value
    return _f


def _fn_raises():
    def _f(messages, *, temperature=None, max_new_tokens=None, stop=None):
        raise RuntimeError("slot boom")
    return _f


class TestAuthoredRepair:
    def test_reauthors_dirty_line_when_clean(self):
        led = {"lines": [
            {"line_id": "b1", "speaker_role": "character",
             "text": "Hand me the gun.", "word_count": 4},
        ]}
        n = GG.apply_genre_spoken_repair(
            led, ["gun"], _fn_returns("Hand me the tool."))
        assert n == 1
        row = led["lines"][0]
        assert row["text"] == "Hand me the tool."
        assert row["word_count"] == 4
        assert any(f.startswith("genre_repair:removed=gun")
                   for f in row.get("compose_flags", []))

    def test_keeps_original_when_rewrite_still_dirty(self):
        led = {"lines": [
            {"line_id": "b1", "speaker_role": "character", "text": "the gun"},
        ]}
        n = GG.apply_genre_spoken_repair(led, ["gun"], _fn_returns("still a gun"))
        assert n == 0
        assert led["lines"][0]["text"] == "the gun"
        assert "compose_flags" not in led["lines"][0]

    def test_never_raises_on_creative_fn_error(self):
        led = {"lines": [
            {"line_id": "b1", "speaker_role": "character", "text": "the gun"},
        ]}
        assert GG.apply_genre_spoken_repair(led, ["gun"], _fn_raises()) == 0
        assert led["lines"][0]["text"] == "the gun"

    def test_noop_when_disabled_or_clean(self):
        led = {"lines": [
            {"line_id": "b1", "speaker_role": "character", "text": "clear skies"},
        ]}
        assert GG.apply_genre_spoken_repair(led, ["gun"], _fn_returns("x")) == 0
        assert GG.apply_genre_spoken_repair(led, ["gun"], None) == 0
        assert GG.apply_genre_spoken_repair(led, [], _fn_returns("x")) == 0

    def test_skips_music_rows(self):
        led = {"lines": [
            {"line_id": "m1", "speaker_role": "music_inter", "text": "gun theme"},
        ]}
        assert GG.apply_genre_spoken_repair(
            led, ["gun"], _fn_returns("clean")) == 0
        assert led["lines"][0]["text"] == "gun theme"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
