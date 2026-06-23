"""Best-of-N structural story-refine selector (2026-06-23).

Local-only (default), opt-in remote, deterministic best-of-N OUTLINE selector.
This file grows one chunk at a time:

  Chunk 1 -- OutlineRequest.diversity_hint + _build_user_prompt render
             (flag-off / empty-hint => byte-identical prompt).
  Chunk 2 -- score_outline pure scorer + StoryScore (raw-intent metrics).
  Chunk 3 -- select_best_outline selector + flag parse + provider gate.
  Chunk 4 -- optional remote best-of-N + fail-closed cost guard.

Pure / CPU. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_outline as OUT  # noqa: E402
from nodes import _otr_story_select as SEL  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: a valid OutlineRequest (budget is required by the v2.0 contract).
# ---------------------------------------------------------------------------
def _req(diversity_hint=None):
    from nodes import _otr_episode_budget as EB
    budget = EB.compute_episode_budget(
        target_words=400,
        act_count=EB.default_act_count(400),
        include_act_breaks=True,
        num_characters=2,
    )
    kwargs = dict(
        news_seed="A deep-space signal is detected near a dying star.",
        style="hard sci-fi procedural",
        target_words=400,
        character_cast=("MALI", "MANFRED"),
        budget=budget,
    )
    if diversity_hint is not None:
        kwargs["diversity_hint"] = diversity_hint
    return OUT.OutlineRequest(**kwargs)


# ---------------------------------------------------------------------------
# Chunk 1 -- diversity_hint field + prompt render
# ---------------------------------------------------------------------------
class TestDiversityHint:
    def test_field_defaults_to_empty(self):
        assert _req().diversity_hint == ""

    def test_empty_hint_prompt_is_byte_identical_to_default(self):
        # Field defaulted vs explicitly "" must produce the SAME prompt, and
        # neither may carry the structural-variation overlay. This is the
        # byte-identical guarantee for candidate 0 / every non-selector call.
        defaulted = OUT._build_user_prompt(_req())
        explicit_empty = OUT._build_user_prompt(_req(""))
        assert defaulted == explicit_empty
        assert "Structural variation" not in defaulted

    def test_whitespace_only_hint_is_treated_as_empty(self):
        # The render strips; a whitespace-only hint must not perturb the prompt.
        assert OUT._build_user_prompt(_req("   ")) == OUT._build_user_prompt(_req(""))

    def test_nonempty_hint_is_rendered_verbatim(self):
        hint = "open on the personal stake, not the institutional threat"
        prompt = OUT._build_user_prompt(_req(hint))
        assert "Structural variation" in prompt
        assert hint in prompt

    def test_nonempty_hint_only_appends(self):
        # The hinted prompt is the empty prompt plus the appended overlay block:
        # every line of the empty prompt is still present, in order.
        empty = OUT._build_user_prompt(_req(""))
        hinted = OUT._build_user_prompt(_req("vary which stake opens the story"))
        assert empty != hinted
        assert hinted.startswith(empty.split("\nBuild a dramatic outline")[0])
        assert "vary which stake opens the story" in hinted

    def test_different_hints_produce_different_prompts(self):
        a = OUT._build_user_prompt(_req("open on the turn"))
        b = OUT._build_user_prompt(_req("open on the consequence"))
        assert a != b


# ---------------------------------------------------------------------------
# Chunk 2 -- score_outline pure scorer
# ---------------------------------------------------------------------------
from dataclasses import dataclass as _dc  # noqa: E402


@_dc
class _FakeBeat:
    speaker_role: str
    intent: str = ""


@_dc
class _FakeOutline:
    premise: str
    beats: list


_ROSTER = ("MALI", "MANFRED")


def _generic_outline():
    # A "console standoff": intents stuffed with GENERIC crisis nouns that are
    # NOT in the premise palette; nothing references the premise.
    return _FakeOutline(
        premise="Two engineers stand in a control room.",
        beats=[
            _FakeBeat("music_open", "theme plays"),
            _FakeBeat("character", "the reactor console overloads as the countdown begins"),
            _FakeBeat("character", "they grab the lever and the failsafe switch"),
            _FakeBeat("character", "the gauge climbs toward meltdown"),
            _FakeBeat("music_close", "theme fades"),
        ],
    )


def _grounded_outline():
    # Premise-anchored: intents reference premise/roster nouns; no generic
    # crisis nouns at all.
    return _FakeOutline(
        premise="A district adopts a tutoring algorithm whose grading weights are disputed.",
        beats=[
            _FakeBeat("music_open", "theme plays"),
            _FakeBeat("character", "Mali questions the tutoring algorithm grading weights"),
            _FakeBeat("character", "Manfred defends the district adoption decision"),
            _FakeBeat("character", "the disputed weights reshape a flagged transcript"),
            _FakeBeat("music_close", "theme fades"),
        ],
    )


class TestScoreOutline:
    def test_returns_storyscore(self):
        s = SEL.score_outline(_generic_outline(), {}, _ROSTER)
        assert isinstance(s, SEL.StoryScore)

    def test_deterministic(self):
        o = _grounded_outline()
        assert SEL.score_outline(o, {}, _ROSTER) == SEL.score_outline(o, {}, _ROSTER)

    def test_ungrounded_crisis_nonzero_on_raw_intents(self):
        # Verify-at-build #2: the metric MUST discriminate on raw intents.
        s = SEL.score_outline(_generic_outline(), {}, _ROSTER)
        assert s.ungrounded_crisis_density > 0.0

    def test_grounded_beats_generic_on_every_axis(self):
        g = SEL.score_outline(_generic_outline(), {}, _ROSTER)
        p = SEL.score_outline(_grounded_outline(), {}, _ROSTER)
        assert p.ungrounded_crisis_density < g.ungrounded_crisis_density
        assert p.premise_grounding > g.premise_grounding
        assert p.distinct_conflict_nouns > g.distinct_conflict_nouns

    def test_grounded_outline_density_zero_grounding_full(self):
        p = SEL.score_outline(_grounded_outline(), {}, _ROSTER)
        assert p.ungrounded_crisis_density == 0.0
        assert p.premise_grounding == pytest.approx(1.0)

    def test_announcer_counts_as_voiced(self):
        # An announcer beat with a generic crisis noun must contribute (proves
        # _is_voiced includes announcer, mirroring build_sq_data's scope).
        o = _FakeOutline(
            premise="A quiet town.",
            beats=[_FakeBeat("announcer", "a reactor meltdown looms")],
        )
        s = SEL.score_outline(o, {}, ())
        assert s.ungrounded_crisis_density > 0.0

    def test_no_voiced_beats_is_zero_not_crash(self):
        # Division-by-zero guards: only non-voiced beats => clean zeros.
        o = _FakeOutline(
            premise="A countdown reactor meltdown lever.",
            beats=[_FakeBeat("music_open", "x"), _FakeBeat("sfx", "y")],
        )
        s = SEL.score_outline(o, {}, ())
        assert s.ungrounded_crisis_density == 0.0
        assert s.premise_grounding == 0.0
        assert s.distinct_conflict_nouns == 0

    def test_pure_no_mutation_of_intents(self):
        o = _generic_outline()
        before = [b.intent for b in o.beats]
        SEL.score_outline(o, {}, _ROSTER)
        assert [b.intent for b in o.beats] == before

    def test_empty_outline_no_crash(self):
        s = SEL.score_outline(_FakeOutline(premise="", beats=[]), {}, ())
        assert s.ungrounded_crisis_density == 0.0
        assert s.distinct_conflict_nouns == 0
        assert s.premise_grounding == 0.0


# ---------------------------------------------------------------------------
# Chunk 3 -- flag parse + provider gate + selector
#
# The writer-level flag-OFF invariant (exactly one generate_outline call + no
# meta.story_quality.best_of_n key) is guaranteed structurally: run() calls the
# selector ONLY when resolve_best_of_n returns effective_n >= 2, and it is
# covered end-to-end by the existing writer suite (flag unset by default).
# TestResolveBestOfN below proves the gate returns effective_n < 2 for every
# disabled / remote-clamped case, so the byte-identical path is taken.
# ---------------------------------------------------------------------------
from nodes._otr_outline import OutlineFailedError  # noqa: E402


def _fail(req):
    raise OutlineFailedError(attempts=[("raw", "boom")], request=req)


class TestResolveBestOfN:
    LOCAL = {"creative_writing_model": "mistral-nemo"}
    REMOTE_OR = {"creative_writing_model": "openrouter:meta-llama/llama-3.1-70b"}
    REMOTE_CF = {"creative_writing_model": "comfy:some-model"}

    def test_unset_is_disabled(self, monkeypatch):
        monkeypatch.delenv("OTR_STORY_BEST_OF_N", raising=False)
        assert SEL.resolve_best_of_n(self.LOCAL) == (1, 1, "")

    def test_zero_and_one_disabled(self, monkeypatch):
        for v in ("0", "1"):
            monkeypatch.setenv("OTR_STORY_BEST_OF_N", v)
            assert SEL.resolve_best_of_n(self.LOCAL) == (1, 1, "")

    def test_blank_disabled(self, monkeypatch):
        monkeypatch.setenv("OTR_STORY_BEST_OF_N", "   ")
        assert SEL.resolve_best_of_n(self.LOCAL) == (1, 1, "")

    def test_non_int_disabled(self, monkeypatch):
        monkeypatch.setenv("OTR_STORY_BEST_OF_N", "abc")
        assert SEL.resolve_best_of_n(self.LOCAL) == (1, 1, "non_int_flag")

    def test_local_n3(self, monkeypatch):
        monkeypatch.setenv("OTR_STORY_BEST_OF_N", "3")
        assert SEL.resolve_best_of_n(self.LOCAL) == (3, 3, "")

    def test_local_clamped_to_6(self, monkeypatch):
        monkeypatch.setenv("OTR_STORY_BEST_OF_N", "9")
        assert SEL.resolve_best_of_n(self.LOCAL) == (9, 6, "max_local_6")

    def test_remote_openrouter_clamps_to_1(self, monkeypatch):
        monkeypatch.setenv("OTR_STORY_BEST_OF_N", "3")
        assert SEL.resolve_best_of_n(self.REMOTE_OR) == (3, 1, "remote_provider_local_only")

    def test_remote_comfy_clamps_to_1(self, monkeypatch):
        monkeypatch.setenv("OTR_STORY_BEST_OF_N", "3")
        assert SEL.resolve_best_of_n(self.REMOTE_CF) == (3, 1, "remote_provider_local_only")

    def test_remote_off_stays_off(self, monkeypatch):
        # Provider gate is consulted only when the flag is >= 2.
        monkeypatch.setenv("OTR_STORY_BEST_OF_N", "0")
        assert SEL.resolve_best_of_n(self.REMOTE_OR) == (1, 1, "")


class TestSelectBestOutline:
    def _base_req(self):
        return _req()  # real, valid OutlineRequest (diversity_hint defaults "")

    def test_picks_lowest_density_winner(self):
        # candidate 0 (hint="") -> generic (bad); i>=1 (hint set) -> grounded.
        def gen(req):
            return _generic_outline() if req.diversity_hint == "" else _grounded_outline()
        meta = {}
        out = SEL.select_best_outline(
            gen, self._base_req(), cast_seed=42, n=2, meta=meta, roster=_ROSTER,
        )
        assert out is not None
        bon = meta["story_quality"]["best_of_n"]
        assert bon["winner_index"] == 1
        assert bon["effective_n"] == 2
        assert len(bon["scores"]) == 2

    def test_candidate_0_uses_empty_hint(self):
        seen = []
        def gen(req):
            seen.append(req.diversity_hint)
            return _grounded_outline()
        SEL.select_best_outline(
            gen, self._base_req(), cast_seed=7, n=3, meta={}, roster=_ROSTER,
        )
        assert seen[0] == ""
        assert all(h != "" for h in seen[1:])  # i>=1 carry a structural hint

    def test_deterministic_across_runs(self):
        hints = []
        def gen(req):
            hints.append(req.diversity_hint)
            return _generic_outline() if req.diversity_hint == "" else _grounded_outline()
        m1, m2 = {}, {}
        SEL.select_best_outline(gen, self._base_req(), cast_seed=99, n=3, meta=m1, roster=_ROSTER)
        run1 = list(hints); hints.clear()
        SEL.select_best_outline(gen, self._base_req(), cast_seed=99, n=3, meta=m2, roster=_ROSTER)
        run2 = list(hints)
        assert run1 == run2
        assert (m1["story_quality"]["best_of_n"]["winner_index"]
                == m2["story_quality"]["best_of_n"]["winner_index"])

    def test_failed_candidate_recorded_and_skipped(self):
        def gen(req):
            if req.diversity_hint != "":
                _fail(req)
            return _grounded_outline()
        meta = {}
        SEL.select_best_outline(
            gen, self._base_req(), cast_seed=1, n=2, meta=meta, roster=_ROSTER,
        )
        bon = meta["story_quality"]["best_of_n"]
        assert bon["winner_index"] == 0
        by_idx = {s["candidate_index"]: s for s in bon["scores"]}
        assert by_idx[0]["ok"] is True
        assert by_idx[1]["ok"] is False
        assert by_idx[1]["error_type"] == "OutlineFailedError"

    def test_never_fail_deterministic_fallback(self):
        calls = {"n": 0}
        def gen(req):
            calls["n"] += 1
            if calls["n"] <= 3:          # the 3 candidates all fail
                _fail(req)
            return _grounded_outline()   # the deterministic fallback succeeds
        meta = {}
        out = SEL.select_best_outline(
            gen, self._base_req(), cast_seed=5, n=3, meta=meta, roster=_ROSTER,
        )
        assert out is not None
        bon = meta["story_quality"]["best_of_n"]
        assert bon["winner_index"] == 0
        assert all(s["ok"] is False for s in bon["scores"])
        assert calls["n"] == 4           # 3 candidates + 1 fallback

    def test_never_fail_loud_when_fallback_also_fails(self):
        def gen(req):
            _fail(req)
        with pytest.raises(OutlineFailedError):
            SEL.select_best_outline(
                gen, self._base_req(), cast_seed=5, n=2, meta={}, roster=_ROSTER,
            )

    def test_telemetry_score_fields(self):
        def gen(req):
            return _grounded_outline()
        meta = {}
        SEL.select_best_outline(
            gen, self._base_req(), cast_seed=3, n=2, meta=meta, roster=_ROSTER,
        )
        for s in meta["story_quality"]["best_of_n"]["scores"]:
            assert s["ok"] is True
            assert "ungrounded_crisis_density" in s
            assert "distinct_conflict_nouns" in s
            assert "premise_grounding" in s

    def test_telemetry_merges_not_replaces(self):
        meta = {"story_quality": {"l12_domain": "energy"}}
        def gen(req):
            return _grounded_outline()
        SEL.select_best_outline(
            gen, self._base_req(), cast_seed=3, n=2, meta=meta, roster=_ROSTER,
        )
        assert meta["story_quality"]["l12_domain"] == "energy"
        assert "best_of_n" in meta["story_quality"]


# ---------------------------------------------------------------------------
# Chunk 4 -- optional remote best-of-N + fail-closed cost guard
# ---------------------------------------------------------------------------
class TestRemoteCostGuard:
    def test_estimate_is_positive_and_scales(self):
        t1, u1 = SEL.estimate_remote_cost(1)
        t3, u3 = SEL.estimate_remote_cost(3)
        assert t3 == 3 * t1
        assert u3 == pytest.approx(3 * u1)
        assert u1 > 0.0

    def test_guard_ok_within_budget(self):
        allowed, reason, est = SEL.remote_cost_guard(3)
        assert allowed == 3
        assert reason == "remote_ok"
        assert est > 0.0

    def test_guard_clamps_on_per_run_token_breach(self):
        allowed, reason, _est = SEL.remote_cost_guard(
            3, per_outline_tokens=10_000_000,
        )
        assert allowed == 1
        assert reason == "cost_guard_per_run_budget"

    def test_guard_clamps_on_autonomy_ceiling(self):
        # Expensive price trips guard (b) without breaching the token guard (a).
        allowed, reason, est = SEL.remote_cost_guard(3, price_per_1k=10.0)
        assert allowed == 1
        assert reason == "cost_guard_autonomy_ceiling"
        assert est >= SEL._AUTONOMY_CEILING_USD

    def test_guard_token_checked_before_usd(self):
        _allowed, reason, _est = SEL.remote_cost_guard(
            3, per_outline_tokens=10_000_000, price_per_1k=10.0,
        )
        assert reason == "cost_guard_per_run_budget"


class TestResolveRemote:
    REMOTE = {"creative_writing_model": "openrouter:meta-llama/llama-3.1-70b"}
    LOCAL = {"creative_writing_model": "mistral-nemo"}

    def test_remote_default_off_clamps_to_1(self, monkeypatch):
        monkeypatch.setenv("OTR_STORY_BEST_OF_N", "3")
        monkeypatch.delenv("OTR_STORY_BEST_OF_N_ALLOW_REMOTE", raising=False)
        assert SEL.resolve_best_of_n(self.REMOTE) == (3, 1, "remote_provider_local_only")

    def test_remote_opt_in_runs(self, monkeypatch):
        monkeypatch.setenv("OTR_STORY_BEST_OF_N", "3")
        monkeypatch.setenv("OTR_STORY_BEST_OF_N_ALLOW_REMOTE", "1")
        assert SEL.resolve_best_of_n(self.REMOTE) == (3, 3, "remote_ok")

    def test_remote_opt_in_capped_to_3(self, monkeypatch):
        # requested 9 -> local-clamp 6 -> remote-cap 3.
        monkeypatch.setenv("OTR_STORY_BEST_OF_N", "9")
        monkeypatch.setenv("OTR_STORY_BEST_OF_N_ALLOW_REMOTE", "1")
        assert SEL.resolve_best_of_n(self.REMOTE) == (9, 3, "remote_max_3")

    def test_remote_opt_in_truthy_variants(self, monkeypatch):
        monkeypatch.setenv("OTR_STORY_BEST_OF_N", "2")
        for v in ("1", "true", "YES", "on"):
            monkeypatch.setenv("OTR_STORY_BEST_OF_N_ALLOW_REMOTE", v)
            req, eff, _reason = SEL.resolve_best_of_n(self.REMOTE)
            assert (req, eff) == (2, 2)

    def test_local_ignores_allow_remote_flag(self, monkeypatch):
        monkeypatch.setenv("OTR_STORY_BEST_OF_N", "3")
        monkeypatch.setenv("OTR_STORY_BEST_OF_N_ALLOW_REMOTE", "1")
        assert SEL.resolve_best_of_n(self.LOCAL) == (3, 3, "")

    def test_remote_cost_guard_fails_closed_in_resolve(self, monkeypatch):
        # Shrink the per-outline estimate's ceiling headroom so the guard clamps.
        monkeypatch.setenv("OTR_STORY_BEST_OF_N", "3")
        monkeypatch.setenv("OTR_STORY_BEST_OF_N_ALLOW_REMOTE", "1")
        monkeypatch.setattr(SEL, "_REMOTE_OUTLINE_TOKENS_EST", 10_000_000)
        req, eff, reason = SEL.resolve_best_of_n(self.REMOTE)
        assert eff == 1
        assert reason == "cost_guard_per_run_budget"


class TestCostProbeTelemetry:
    def test_per_candidate_cost_recorded(self):
        state = {"cum": 0.0}
        def gen(req):
            state["cum"] += 1.5          # each candidate "costs" $1.50
            return _grounded_outline()
        def probe():
            return state["cum"]
        meta = {}
        SEL.select_best_outline(
            gen, _req(), cast_seed=1, n=2, meta=meta, roster=_ROSTER,
            cost_probe=probe,
        )
        for s in meta["story_quality"]["best_of_n"]["scores"]:
            assert s["cost_usd"] == pytest.approx(1.5)

    def test_cost_usd_none_without_probe(self):
        def gen(req):
            return _grounded_outline()
        meta = {}
        SEL.select_best_outline(
            gen, _req(), cast_seed=1, n=2, meta=meta, roster=_ROSTER,
        )
        for s in meta["story_quality"]["best_of_n"]["scores"]:
            assert s["cost_usd"] is None


# ---------------------------------------------------------------------------
# v1 chunk 0 -- wire diversity_hint into the REAL Path C builders (fix the
# shipped v0 dead-code bug: the hint was rendered only in the test-only
# _build_user_prompt, so best-of-N candidates varied only by RNG seed).
# ---------------------------------------------------------------------------
class TestPathCDiversityHint:
    def _macro(self):
        return OUT._MacroShape(
            title="Chandra's Echo",
            premise="An observatory catches a signal someone wants buried.",
            setting="A mountaintop radio observatory",
            time_of_day="night",
        )

    # --- macro (Stage 1) ---
    def test_macro_empty_hint_no_render(self):
        assert "Structural variation" not in OUT._build_macro_user_prompt(_req(""))

    def test_macro_empty_is_byte_identical_to_default(self):
        assert OUT._build_macro_user_prompt(_req()) == OUT._build_macro_user_prompt(_req(""))

    def test_macro_renders_hint(self):
        hint = "open on the personal stake, not the institutional threat"
        p = OUT._build_macro_user_prompt(_req(hint))
        assert "Structural variation" in p
        assert hint in p

    # --- beat (Stage 3) ---
    def test_beat_empty_hint_no_render(self):
        p = OUT._build_beat_user_prompt(
            _req(""), self._macro(), "rising_action", "MANFRED", (1, 4),
        )
        assert "Structural variation" not in p

    def test_beat_empty_is_byte_identical_to_default(self):
        a = OUT._build_beat_user_prompt(
            _req(), self._macro(), "rising_action", "MANFRED", (1, 4),
        )
        b = OUT._build_beat_user_prompt(
            _req(""), self._macro(), "rising_action", "MANFRED", (1, 4),
        )
        assert a == b

    def test_beat_renders_hint(self):
        hint = "make the conflict interpersonal, not a countdown"
        p = OUT._build_beat_user_prompt(
            _req(hint), self._macro(), "rising_action", "MANFRED", (1, 4),
        )
        assert "Structural variation" in p
        assert hint in p

    def test_different_hints_diverge_the_macro_prompt(self):
        # The point of the fix: a different hint => a different production prompt
        # (=> genuinely different candidates, not RNG-only variation).
        a = OUT._build_macro_user_prompt(_req("open on the turn"))
        b = OUT._build_macro_user_prompt(_req("open on the consequence"))
        assert a != b
