"""T1 pitch room + greenlight (2026-06-23).

Pure / CPU. The LLM stages are exercised by monkeypatching structured_call so
no model is loaded -- the divergence seeds, validation, fallback, tie-break,
frozen-replace handoff, and the default-OFF byte-identical gate are all proven
without inference. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_outline as OUT  # noqa: E402
from nodes import _otr_pitch_room as PR  # noqa: E402
from nodes import _otr_structured_call as SC  # noqa: E402


def _req():
    from nodes import _otr_episode_budget as EB
    budget = EB.compute_episode_budget(
        target_words=400,
        act_count=EB.default_act_count(400),
        include_act_breaks=True,
        num_characters=2,
    )
    return OUT.OutlineRequest(
        news_seed="A district adopts a tutoring algorithm whose grading weights are disputed.",
        style="hard sci-fi procedural",
        target_words=400,
        character_cast=("MALI", "MANFRED"),
        budget=budget,
    )


def _fake_candidates(n=3):
    out = []
    for i in range(1, n + 1):
        out.append({
            "id": i,
            "logline": f"Pitch {i} logline about the disputed grading weights.",
            "protagonist": f"Protagonist {i}",
            "antagonist_or_pressure": "the district board",
            "genre_mode": "drama",
            "emotional_core": "trust under pressure",
            "theme_sentence": "Truth costs something.",
            "final_20_seconds": "Mali makes the call on-stage and lives with it.",
            "conflict_type": "a struggle to be believed",
            "setting_class": "a school district office",
            "surprise": 4,
            "human_want": 5,
            "stageability": 5,
            "console_standoff_risk": i,   # pitch 1 has the lowest risk
            "why_different": f"differs from the others in axis {i}",
        })
    return out


# ---------------------------------------------------------------------------
# Flag gate
# ---------------------------------------------------------------------------
class TestEnabled:
    def test_default_off(self):
        assert PR.pitch_room_enabled(env={}) is False

    def test_on_variants(self):
        for v in ("1", "true", "YES", "on"):
            assert PR.pitch_room_enabled(env={"OTR_ENABLE_PITCH_ROOM": v}) is True

    def test_off_variants(self):
        for v in ("0", "false", "no", "off", ""):
            assert PR.pitch_room_enabled(env={"OTR_ENABLE_PITCH_ROOM": v}) is False


# ---------------------------------------------------------------------------
# Divergence seeds
# ---------------------------------------------------------------------------
class TestSeeds:
    def test_three_distinct_genres_and_archetypes(self):
        seeds = PR.build_pitch_seeds(seed_context=42, domain="education")
        assert len(seeds) == 3
        assert [s["id"] for s in seeds] == [1, 2, 3]
        assert len({s["genre"] for s in seeds}) == 3
        assert len({s["archetype"] for s in seeds}) == 3

    def test_conflict_material_from_palette(self):
        seeds = PR.build_pitch_seeds(seed_context=7, domain="education")
        from nodes._otr_story_quality_l12 import conflict_palette
        pal = conflict_palette("education")
        for s in seeds:
            assert s["conflict_object"] in pal["conflict_objects"]
            assert s["conflict_type"] in pal["conflict_types"]

    def test_deterministic(self):
        a = PR.build_pitch_seeds(seed_context=99, domain="energy")
        b = PR.build_pitch_seeds(seed_context=99, domain="energy")
        assert a == b

    def test_distinct_pick_returns_distinct(self):
        picks = PR._distinct_pick(("a", "b", "c", "d"), base=123456, count=3)
        assert len(picks) == 3 and len(set(picks)) == 3


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
class TestPureHelpers:
    def test_winner_to_brief_uses_template(self):
        brief = PR.winner_to_brief(_fake_candidates()[0])
        assert "Protagonist:" in brief
        assert "Conflict:" in brief
        assert "Final 20s:" in brief

    def test_winner_to_brief_truncates(self):
        cand = _fake_candidates()[0]
        cand["logline"] = "word " * 500
        brief = PR.winner_to_brief(cand)
        assert len(brief.split(" ")) <= PR._BRIEF_MAX_WORDS

    def test_tie_break_lowest_risk_then_id(self):
        # pitch 1 has the lowest console_standoff_risk in the fixture.
        assert PR._tie_break(_fake_candidates()) == 1

    def test_tie_break_ties_break_on_id(self):
        cands = _fake_candidates()
        for c in cands:
            c["console_standoff_risk"] = 2   # all tied
        assert PR._tie_break(cands) == 1     # lowest id wins

    def test_valid_candidates_dedups_and_bounds(self):
        class _C:
            def __init__(self, i):
                self.id = i
            def model_dump(self):
                return {"id": self.id}
        class _Slate:
            candidates = [_C(1), _C(1), _C(2), _C(9), _C(3)]
        got = PR._valid_candidates(_Slate())
        assert [c["id"] for c in got] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Frontier greenlight resolution (fail-closed to local)
# ---------------------------------------------------------------------------
class TestFrontierResolve:
    def _local(self):
        return lambda *a, **k: "x"

    def test_default_local(self):
        fn, src = PR._resolve_greenlight_fn(self._local(), None, {})
        assert src == "local"

    def test_no_model_falls_to_local(self):
        fn, src = PR._resolve_greenlight_fn(
            self._local(), {"generate_fn": lambda *a, **k: "y"},
            {"OTR_ENABLE_FRONTIER_GREENLIGHT": "1"},
        )
        assert src == "local"

    def test_no_frontier_fn_falls_to_local(self):
        fn, src = PR._resolve_greenlight_fn(
            self._local(), None,
            {"OTR_ENABLE_FRONTIER_GREENLIGHT": "1", "OTR_GREENLIGHT_MODEL": "x:y"},
        )
        assert src == "local"

    def test_frontier_used_when_all_present(self):
        ffn = lambda *a, **k: "frontier"
        fn, src = PR._resolve_greenlight_fn(
            self._local(), {"generate_fn": ffn},
            {"OTR_ENABLE_FRONTIER_GREENLIGHT": "1", "OTR_GREENLIGHT_MODEL": "x:y"},
        )
        assert src == "frontier" and fn is ffn


# ---------------------------------------------------------------------------
# run_pitch_room -- structured_call monkeypatched
# ---------------------------------------------------------------------------
class _FakeCand:
    def __init__(self, d):
        self._d = d
        self.id = d["id"]
    def model_dump(self):
        return dict(self._d)


class _FakeSlate:
    def __init__(self, cands):
        self.candidates = [_FakeCand(c) for c in cands]


class _FakeDecision:
    def __init__(self, sid, ranking, rationale="picked"):
        self.selected_id = sid
        self.ranking = ranking
        self.rationale = rationale


def _patch_sc(monkeypatch, *, slate=None, decision=None, raise_on=None):
    def fake(*, prompt, schema, slot_fn, helper_name="", post_validator=None, **kw):
        if raise_on and helper_name == raise_on:
            raise RuntimeError("boom")
        if helper_name == "OTR_PitchRoom":
            return slate
        if helper_name == "OTR_PitchGreenlight":
            return decision
        raise AssertionError(f"unexpected helper {helper_name}")
    monkeypatch.setattr(SC, "structured_call", fake)


class TestRunPitchRoom:
    def test_greenlit_replaces_brief(self, monkeypatch):
        cands = _fake_candidates()
        _patch_sc(monkeypatch, slate=_FakeSlate(cands),
                  decision=_FakeDecision(2, [2, 1, 3]))
        meta = {}
        req = _req()
        new_req, pm = PR.run_pitch_room(
            req, generate_fn=lambda *a, **k: "x", seed_context=42, meta=meta, env={},
        )
        assert pm.status == "greenlit"
        assert pm.selected_id == 2
        assert new_req is not req                       # frozen replace -> new obj
        assert new_req.script_brief                     # brief replaced
        assert req.script_brief == ""                   # original untouched
        assert meta["story_quality"]["pitch"]["status"] == "greenlit"
        assert meta["story_quality"]["pitch"]["selected_id"] == 2

    def test_too_few_candidates_falls_back(self, monkeypatch):
        _patch_sc(monkeypatch, slate=_FakeSlate(_fake_candidates(2)),
                  decision=_FakeDecision(1, [1]))
        meta = {}
        req = _req()
        new_req, pm = PR.run_pitch_room(
            req, generate_fn=lambda *a, **k: "x", seed_context=42, meta=meta, env={},
        )
        assert pm.status == "failed_fallback"
        assert new_req is req                            # original kept
        assert meta["story_quality"]["pitch"]["status"] == "failed_fallback"

    def test_greenlight_failure_tie_breaks(self, monkeypatch):
        cands = _fake_candidates()
        _patch_sc(monkeypatch, slate=_FakeSlate(cands),
                  decision=None, raise_on="OTR_PitchGreenlight")
        meta = {}
        new_req, pm = PR.run_pitch_room(
            _req(), generate_fn=lambda *a, **k: "x", seed_context=42, meta=meta, env={},
        )
        assert pm.status == "greenlit"
        assert pm.selected_id == 1                       # tie-break: lowest risk
        assert pm.greenlight_source.endswith("tie_break")

    def test_generation_failure_falls_back(self, monkeypatch):
        _patch_sc(monkeypatch, slate=None, decision=None, raise_on="OTR_PitchRoom")
        meta = {}
        req = _req()
        new_req, pm = PR.run_pitch_room(
            req, generate_fn=lambda *a, **k: "x", seed_context=42, meta=meta, env={},
        )
        assert pm.status == "failed_fallback"
        assert new_req is req

    def test_brief_is_grounded_in_winner(self, monkeypatch):
        cands = _fake_candidates()
        _patch_sc(monkeypatch, slate=_FakeSlate(cands),
                  decision=_FakeDecision(3, [3, 2, 1]))
        new_req, pm = PR.run_pitch_room(
            _req(), generate_fn=lambda *a, **k: "x", seed_context=1, meta={}, env={},
        )
        assert "Protagonist 3" in new_req.script_brief
