"""v4 campaign P1(vi): header<->scene STRUCTURAL coherence guard.

Unique scene_ids + no line referencing an undeclared scene (music rows out of
scope). Semantic scene-vs-beat matching is deliberately NOT done (unlawful LLM
gate). Deterministic G15 terminal, opt-in, inert for every current bank.
Pure / CPU. UTF-8 no BOM, SFW.
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


class TestSceneCoherence:
    def test_clean(self):
        led = {"scenes": _scenes(), "lines": [
            {"line_id": "b1", "speaker_role": "character", "scene_id": "s01"},
            {"line_id": "b2", "speaker_role": "character", "scene_id": "s02"},
        ]}
        assert SG.find_scene_coherence_issues(led) == []

    def test_dangling_line_ref(self):
        led = {"scenes": _scenes(), "lines": [
            {"line_id": "b1", "speaker_role": "character", "scene_id": "s99"},
        ]}
        issues = SG.find_scene_coherence_issues(led)
        assert any("unknown scene" in i and "s99" in i for i in issues)

    def test_duplicate_scene_id(self):
        led = {"scenes": [{"scene_id": "s01"}, {"scene_id": "s01"}], "lines": []}
        issues = SG.find_scene_coherence_issues(led)
        assert any("duplicate scene_id" in i for i in issues)

    def test_no_scenes_skips(self):
        led = {"scenes": [], "lines": [
            {"line_id": "b1", "speaker_role": "character", "scene_id": "s99"}]}
        assert SG.find_scene_coherence_issues(led) == []

    def test_music_row_excluded(self):
        led = {"scenes": _scenes(), "lines": [
            {"line_id": "m1", "speaker_role": "music_inter", "scene_id": "s99"}]}
        assert SG.find_scene_coherence_issues(led) == []

    def test_line_without_scene_id_ok(self):
        led = {"scenes": _scenes(), "lines": [
            {"line_id": "b1", "speaker_role": "character"}]}
        assert SG.find_scene_coherence_issues(led) == []

    def test_non_dict_safe(self):
        assert SG.find_scene_coherence_issues(None) == []


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
    "media_archive", "original_radio", "scifi_fable2", "scifi_codex",
    "media_archive_v3", "public_domain_story_v3", "shakespeare_v3",
    "scifi_fable2_v3", "scifi_codex_v3", "scifi_sonnet_v3",
]


class TestCurrentBanksInert:
    @pytest.mark.parametrize("bank_id", _CURRENT_BANKS)
    def test_no_current_bank_opts_in(self, bank_id):
        bank = SR.require_runnable_bank(bank_id)
        assert not (bank.defaults or {}).get("scene_coherence_check")


# ---------------------------------------------------------------------------
# G15 terminal
# ---------------------------------------------------------------------------

def _led(flag):
    meta = {"source_bank": "x"}
    if flag:
        meta["scene_coherence_check"] = True
    return {"schema_version": "l3-2026-05-14", "meta": meta,
            "scenes": [{"scene_id": "s01"}],
            "lines": [{"line_id": "b1", "speaker_role": "character",
                       "scene_id": "s99"}]}


def _g15_errors(ledger_data):
    errors: list = []
    LF._check_g15_scene_coherence(ledger_data, errors, [])
    return errors


class TestG15Terminal:
    def test_fires_when_opted_in(self):
        errs = _g15_errors(_led(True))
        assert errs and "G15" in errs[0]

    def test_inert_without_opt_in(self):
        assert _g15_errors(_led(False)) == []

    def test_clean_ok(self):
        led = {"meta": {"scene_coherence_check": True},
               "scenes": [{"scene_id": "s01"}],
               "lines": [{"line_id": "b1", "speaker_role": "character",
                          "scene_id": "s01"}]}
        assert _g15_errors(led) == []

    def test_run_gap_audit_includes_g15(self):
        report = LF.run_gap_audit(_led(True), label="pre")
        assert any("G15" in e for e in report.errors)

    def test_phase_10_raises(self):
        with pytest.raises(LF.FreezeAssertionError) as ei:
            LF.phase_10_gap_audit_post_and_freeze(_led(True))
        assert any("G15" in e for e in ei.value.errors)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
