"""v4 campaign P1(vi): header<->scene STRUCTURAL coherence guard.

Unique scene_ids + no line referencing an undeclared scene (music rows out of
scope). Semantic scene-vs-beat matching is deliberately NOT done (unlawful LLM
gate). Deterministic G15 terminal, opt-in, inert for every current bank.
Pure / CPU. UTF-8 no BOM, SFW.

THE JOIN IS beat_id -> beats[].scene_id, NOT lines[].scene_id (fixed
2026-08-28, kibitz r2 codex-reviewed). No writer has ever put scene_id on a
LINE row -- lines[] carries beat_id, beats[] carries scene_id, and that is
the only join the schema has. `find_scene_coherence_issues` now returns
`(issues, checked)`; `_check_g15_scene_coherence` takes a fourth `info` dict
and distinguishes VACUITY (armed, scenes declared, zero real linkages
examined -- FAILS) from a legitimate "no scenes at all" skip (unchanged,
still clean).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_scene_guard as SG  # noqa: E402
from nodes import _otr_ledger_freeze as LF  # noqa: E402
from nodes import _otr_story_routing as SR  # noqa: E402


def _scenes():
    return [{"scene_id": "s01", "env": "lab"}, {"scene_id": "s02", "env": "hall"}]


def _beats(*rows):
    """``rows`` are ``(beat_id, scene_id_or_None)`` pairs."""
    return [{"beat_id": bid, "scene_id": sid} for bid, sid in rows]


class TestSceneCoherence:
    def test_clean(self):
        led = {
            "scenes": _scenes(),
            "beats": _beats(("bt1", "s01"), ("bt2", "s02")),
            "lines": [
                {"line_id": "b1", "speaker_role": "character", "beat_id": "bt1"},
                {"line_id": "b2", "speaker_role": "character", "beat_id": "bt2"},
            ],
        }
        issues, checked = SG.find_scene_coherence_issues(led)
        assert issues == []
        assert checked == 2

    def test_dangling_beat_ref(self):
        """A line's beat declares a scene that does not exist in scenes[]."""
        led = {
            "scenes": _scenes(),
            "beats": _beats(("bt1", "s99")),
            "lines": [
                {"line_id": "b1", "speaker_role": "character", "beat_id": "bt1"},
            ],
        }
        issues, checked = SG.find_scene_coherence_issues(led)
        assert checked == 1
        assert any("unknown scene" in i and "s99" in i and "bt1" in i
                   for i in issues)

    def test_duplicate_scene_id(self):
        led = {"scenes": [{"scene_id": "s01"}, {"scene_id": "s01"}],
               "beats": [], "lines": []}
        issues, checked = SG.find_scene_coherence_issues(led)
        assert checked == 0
        assert any("duplicate scene_id" in i for i in issues)

    def test_duplicate_scene_id_survives_alongside_a_real_join(self):
        """The duplicate-scene_id check runs independently of the line<->beat
        join -- it must fire even when other lines resolve cleanly."""
        led = {
            "scenes": [{"scene_id": "s01"}, {"scene_id": "s01"},
                      {"scene_id": "s02"}],
            "beats": _beats(("bt1", "s02")),
            "lines": [
                {"line_id": "b1", "speaker_role": "character", "beat_id": "bt1"},
            ],
        }
        issues, checked = SG.find_scene_coherence_issues(led)
        assert checked == 1
        assert any("duplicate scene_id" in i for i in issues)

    def test_no_scenes_skips(self):
        led = {"scenes": [], "beats": _beats(("bt1", "s99")),
               "lines": [{"line_id": "b1", "speaker_role": "character",
                          "beat_id": "bt1"}]}
        assert SG.find_scene_coherence_issues(led) == ([], 0)

    def test_music_row_excluded(self):
        led = {
            "scenes": _scenes(),
            "beats": _beats(("bt1", "s99")),
            "lines": [{"line_id": "m1", "speaker_role": "music_inter",
                      "beat_id": "bt1"}],
        }
        issues, checked = SG.find_scene_coherence_issues(led)
        assert issues == []
        assert checked == 0  # the music row is excluded before the join runs

    def test_line_with_no_beat_id_is_not_counted(self):
        led = {"scenes": _scenes(), "beats": _beats(("bt1", "s01")),
               "lines": [{"line_id": "b1", "speaker_role": "character"}]}
        issues, checked = SG.find_scene_coherence_issues(led)
        assert issues == []
        assert checked == 0

    def test_beat_without_scene_id_is_ok_and_not_counted(self):
        """A beat that declares no scene is not an error -- scene declaration
        is per-beat optional."""
        led = {
            "scenes": _scenes(),
            "beats": _beats(("bt1", None)),
            "lines": [{"line_id": "b1", "speaker_role": "character",
                      "beat_id": "bt1"}],
        }
        issues, checked = SG.find_scene_coherence_issues(led)
        assert issues == []
        assert checked == 0

    def test_dangling_beat_id_not_counted(self):
        """The line's beat_id does not resolve to any beat at all."""
        led = {"scenes": _scenes(), "beats": _beats(("bt1", "s01")),
               "lines": [{"line_id": "b1", "speaker_role": "character",
                          "beat_id": "bt-does-not-exist"}]}
        issues, checked = SG.find_scene_coherence_issues(led)
        assert issues == []
        assert checked == 0

    def test_non_string_beat_id_never_raises(self):
        """A malformed beat_id (list/int/etc) must be skipped, not crash --
        the function's own contract is 'never raises'."""
        led = {
            "scenes": _scenes(),
            "beats": [{"beat_id": ["not", "hashable"], "scene_id": "s01"},
                     {"beat_id": 42, "scene_id": "s02"}],
            "lines": [{"line_id": "b1", "speaker_role": "character",
                      "beat_id": ["not", "hashable"]}],
        }
        issues, checked = SG.find_scene_coherence_issues(led)
        assert issues == []
        assert checked == 0

    def test_non_dict_safe(self):
        assert SG.find_scene_coherence_issues(None) == ([], 0)

    def test_the_old_line_scene_id_field_is_no_longer_read(self):
        """THE REGRESSION GUARD. This shapes a ledger exactly like every
        current production ledger (and every fixture this test file used to
        carry): scene_id written on the LINE, never on a beat. Under the OLD
        (buggy) reading this silently returned issues=[] and looked clean --
        it did on all 55 published ledgers. Under the fix it must be
        VACUOUS-ELIGIBLE (checked stays 0), not silently clean-and-correct;
        `_check_g15_scene_coherence` is what turns that into a hard failure
        when the gate is armed (see TestG15Terminal.test_vacuity_fires)."""
        led = {
            "scenes": _scenes(),
            "lines": [
                {"line_id": "b1", "speaker_role": "character",
                 "scene_id": "s01"},  # the dead field -- never read any more
            ],
        }
        issues, checked = SG.find_scene_coherence_issues(led)
        assert checked == 0
        assert issues == []  # not itself an "issue" -- the CALLER decides


# ---------------------------------------------------------------------------
# Parser + current banks inert
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

    def test_non_bool_rejected(self):
        with pytest.raises(SR.RegistryValidationError):
            SR._parse_bank(self._row({"scene_coherence_check": "yes"}), "t")

    def test_bool_accepted(self):
        b = SR._parse_bank(self._row({"scene_coherence_check": True}), "t")
        assert b.defaults["scene_coherence_check"] is True


_CURRENT_BANKS = [
    "media_archive", "original", "scifi_news_pro",
    "public_domain", "shakespeare",
]


class TestCurrentBanksInert:
    @pytest.mark.parametrize("bank_id", _CURRENT_BANKS)
    def test_no_current_bank_opts_in(self, bank_id):
        bank = SR.require_runnable_bank(bank_id)
        assert not (bank.defaults or {}).get("scene_coherence_check")


# ---------------------------------------------------------------------------
# G15 terminal
# ---------------------------------------------------------------------------

def _led(flag, *, beats=None, lines=None):
    meta = {"source_bank": "x"}
    if flag:
        meta["scene_coherence_check"] = True
    return {"schema_version": "l3-2026-05-14", "meta": meta,
            "scenes": [{"scene_id": "s01"}],
            "beats": beats if beats is not None else [],
            "lines": lines if lines is not None else []}


def _g15(ledger_data):
    errors: list = []
    info: dict = {}
    LF._check_g15_scene_coherence(ledger_data, errors, [], info)
    return errors, info


class TestG15Terminal:
    def test_inert_without_opt_in(self):
        errs, info = _g15(_led(False))
        assert errs == []
        assert info == {}  # nothing written when not required

    def test_vacuity_fires_when_armed_and_nothing_resolves(self):
        """THE DEFECT THIS WHOLE FIX EXISTS TO CATCH. Scenes are declared,
        the gate is armed, and no line resolves through beat_id to a
        scene-bearing beat -- exactly the shape every current ledger has,
        since no writer populates the join."""
        errs, info = _g15(_led(True))
        assert errs and "G15" in errs[0] and "ZERO" in errs[0]
        assert info["scene_coherence_required"] is True
        assert info["scene_coherence_checked"] == 0
        assert info["scene_coherence_verdict"] == "vacuous"

    def test_no_scenes_at_all_is_still_a_clean_skip(self):
        """The legitimate, pre-existing behaviour: a bank that declares no
        scene structure at all is not penalised, even when armed."""
        led = {"meta": {"scene_coherence_check": True}, "scenes": [],
               "beats": [], "lines": []}
        errs, info = _g15(led)
        assert errs == []
        assert info["scene_coherence_verdict"] == "clean"
        assert info["scene_coherence_checked"] == 0

    def test_clean_when_the_real_join_resolves(self):
        led = _led(True,
                    beats=[{"beat_id": "bt1", "scene_id": "s01"}],
                    lines=[{"line_id": "b1", "speaker_role": "character",
                            "beat_id": "bt1"}])
        errs, info = _g15(led)
        assert errs == []
        assert info["scene_coherence_verdict"] == "clean"
        assert info["scene_coherence_checked"] == 1

    def test_dangling_reference_fires_through_the_real_join(self):
        led = _led(True,
                    beats=[{"beat_id": "bt1", "scene_id": "s99"}],
                    lines=[{"line_id": "b1", "speaker_role": "character",
                            "beat_id": "bt1"}])
        errs, info = _g15(led)
        assert errs and "G15" in errs[0] and "s99" in errs[0]
        assert info["scene_coherence_verdict"] == "issues"
        assert info["scene_coherence_checked"] == 1

    def test_vacuity_does_not_suppress_an_issue_found_in_the_same_pass(self):
        """A duplicate scene_id must still be reported even when the join
        itself resolves nothing -- vacuity is a SEPARATE finding, not a
        replacement for one already made."""
        led = {"meta": {"scene_coherence_check": True},
               "scenes": [{"scene_id": "s01"}, {"scene_id": "s01"}],
               "beats": [], "lines": []}
        errs, info = _g15(led)
        assert len(errs) == 2  # the vacuity error AND the duplicate-id error
        assert any("ZERO" in e for e in errs)
        assert any("duplicate scene_id" in e for e in errs)
        assert info["scene_coherence_verdict"] == "vacuous"
        assert info["scene_coherence_issues"]

    def test_run_gap_audit_includes_g15(self):
        led = _led(True,
                    beats=[{"beat_id": "bt1", "scene_id": "s99"}],
                    lines=[{"line_id": "b1", "speaker_role": "character",
                            "beat_id": "bt1"}])
        report = LF.run_gap_audit(led, label="pre")
        assert any("G15" in e for e in report.errors)
        assert report.info["scene_coherence_verdict"] == "issues"

    def test_phase_10_raises(self):
        led = _led(True,
                    beats=[{"beat_id": "bt1", "scene_id": "s99"}],
                    lines=[{"line_id": "b1", "speaker_role": "character",
                            "beat_id": "bt1"}])
        with pytest.raises(LF.FreezeAssertionError) as ei:
            LF.phase_10_gap_audit_post_and_freeze(led)
        assert any("G15" in e for e in ei.value.errors)

    def test_run_gap_audit_is_still_read_only(self):
        """The gate must not mutate the ledger it inspects."""
        led = _led(True,
                    beats=[{"beat_id": "bt1", "scene_id": "s01"}],
                    lines=[{"line_id": "b1", "speaker_role": "character",
                            "beat_id": "bt1"}])
        import copy
        before = copy.deepcopy(led)
        LF.run_gap_audit(led, label="pre")
        assert led == before


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
