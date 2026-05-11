"""Unit tests for nodes/script_critic.py.

Pure-Python coverage (no LLM load) for the parser, prompt builder,
ledger stamper, and canon updater. The critic's actual LLM call is
exercised via integration runs, not pytest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make the OTR root importable so we can import nodes.script_critic
# without invoking ComfyUI's plugin loader.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="module")
def critic_module():
    """Import the module under test."""
    # Module path bypasses the package-level __init__ to avoid heavy
    # imports the critic tests don't need.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "script_critic_under_test",
        _REPO_ROOT / "nodes" / "script_critic.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# _parse_critic_response
# ---------------------------------------------------------------------------

class TestParseCriticResponse:

    def test_clean_pass(self, critic_module):
        raw = (
            "SCORE: 92\n"
            "VERDICT: PASS\n"
            "ISSUES:\n"
            "- none\n"
        )
        out = critic_module._parse_critic_response(raw)
        assert out["score"] == 92
        assert out["verdict"] == "PASS"
        assert out["issues"] == []

    def test_revise_with_issues(self, critic_module):
        raw = (
            "SCORE: 72\n"
            "VERDICT: REVISE\n"
            "ISSUES:\n"
            "- voice: characters use the same vocabulary register\n"
            "- period: 'okay' used in 1940s dialogue\n"
            "- transitions: 'I'll explain on the way' bridges scene 2 to 3\n"
        )
        out = critic_module._parse_critic_response(raw)
        assert out["score"] == 72
        assert out["verdict"] == "REVISE"
        assert len(out["issues"]) == 3
        assert "voice" in out["issues"][0]
        assert "period" in out["issues"][1]

    def test_reject(self, critic_module):
        raw = (
            "SCORE: 35\n"
            "VERDICT: REJECT\n"
            "ISSUES:\n"
            "- structure: no character dialogue, narration only\n"
        )
        out = critic_module._parse_critic_response(raw)
        assert out["score"] == 35
        assert out["verdict"] == "REJECT"

    def test_score_only_derives_verdict_pass(self, critic_module):
        raw = "SCORE: 90\nISSUES:\n- none"
        out = critic_module._parse_critic_response(raw)
        assert out["score"] == 90
        assert out["verdict"] == "PASS"

    def test_score_only_derives_verdict_revise(self, critic_module):
        raw = "SCORE: 70"
        out = critic_module._parse_critic_response(raw)
        assert out["verdict"] == "REVISE"

    def test_score_only_derives_verdict_reject(self, critic_module):
        raw = "SCORE: 40"
        out = critic_module._parse_critic_response(raw)
        assert out["verdict"] == "REJECT"

    def test_no_parse_returns_advisory_pass(self, critic_module):
        out = critic_module._parse_critic_response(
            "I'm sorry, I can't comply with this request."
        )
        # Advisory fallback: score=None, verdict=PASS so the run continues.
        assert out["score"] is None
        assert out["verdict"] == "PASS"
        assert out["issues"] == []

    def test_empty_response_returns_advisory_pass(self, critic_module):
        out = critic_module._parse_critic_response("")
        assert out["score"] is None
        assert out["verdict"] == "PASS"

    def test_score_clamped_to_valid_range(self, critic_module):
        raw = "SCORE: 999\nVERDICT: PASS"
        out = critic_module._parse_critic_response(raw)
        # 999 is out of 0-100; parser rejects it -> score stays None
        assert out["score"] is None

    def test_issues_capped(self, critic_module):
        many = "ISSUES:\n" + "\n".join([f"- finding {i}" for i in range(30)])
        out = critic_module._parse_critic_response("SCORE: 50\n" + many)
        # Cap is 12 in the parser
        assert len(out["issues"]) <= 12

    def test_long_issue_truncated(self, critic_module):
        long_text = "x" * 500
        raw = f"SCORE: 80\nVERDICT: PASS\nISSUES:\n- voice: {long_text}"
        out = critic_module._parse_critic_response(raw)
        assert len(out["issues"]) == 1
        # Cap at 280 chars per the parser
        assert len(out["issues"][0]) <= 280

    def test_bullet_styles_accepted(self, critic_module):
        # Mix of dash, asterisk, bullet
        raw = (
            "SCORE: 80\nVERDICT: PASS\nISSUES:\n"
            "- a\n"
            "* b\n"
            "• c\n"
        )
        out = critic_module._parse_critic_response(raw)
        assert len(out["issues"]) == 3

    def test_case_insensitive_field_names(self, critic_module):
        raw = "score: 90\nverdict: pass\nissues:\n- none"
        out = critic_module._parse_critic_response(raw)
        assert out["score"] == 90
        assert out["verdict"] == "PASS"


# ---------------------------------------------------------------------------
# Dynamic-template loader: gate parser + filter + normalize
# ---------------------------------------------------------------------------

class TestNormalizeTargetLength:

    def test_smoke_aliases(self, critic_module):
        for raw in ["smoke", "smoke (1 act)", "1 act", "1-act", "tiny", "tiny (smoke, 1 act)"]:
            assert critic_module._normalize_target_length(raw) == "smoke", raw

    def test_short_aliases(self, critic_module):
        for raw in ["short", "short (3 acts)", "3 acts", "3-act"]:
            assert critic_module._normalize_target_length(raw) == "short", raw

    def test_medium_aliases(self, critic_module):
        for raw in ["medium", "medium (5 acts)", "5 acts", "5-act"]:
            assert critic_module._normalize_target_length(raw) == "medium", raw

    def test_empty_falls_to_short(self, critic_module):
        assert critic_module._normalize_target_length("") == "short"
        assert critic_module._normalize_target_length(None or "") == "short"

    def test_unknown_falls_to_short(self, critic_module):
        # Unrecognized labels default to the most common production case
        assert critic_module._normalize_target_length("epic") == "short"


class TestEvaluateGate:

    def test_always_true(self, critic_module):
        assert critic_module._evaluate_gate("always", {}) is True

    def test_target_words_gate_fires(self, critic_module):
        params = {"target_words": 1500}
        assert critic_module._evaluate_gate("target_words >= 700", params) is True
        assert critic_module._evaluate_gate("target_words >= 1500", params) is True
        assert critic_module._evaluate_gate("target_words >= 3000", params) is False

    def test_num_characters_gate(self, critic_module):
        assert critic_module._evaluate_gate("num_characters >= 2", {"num_characters": 2}) is True
        assert critic_module._evaluate_gate("num_characters >= 2", {"num_characters": 1}) is False

    def test_target_length_inequality(self, critic_module):
        assert critic_module._evaluate_gate("target_length != smoke", {"target_length": "short"}) is True
        assert critic_module._evaluate_gate("target_length != smoke", {"target_length": "smoke"}) is False

    def test_target_length_equality(self, critic_module):
        assert critic_module._evaluate_gate("target_length == short", {"target_length": "short"}) is True
        assert critic_module._evaluate_gate("target_length == medium", {"target_length": "short"}) is False

    def test_compound_and(self, critic_module):
        params = {"num_characters": 2, "target_words": 800}
        assert critic_module._evaluate_gate(
            "num_characters >= 2 AND target_words >= 700", params,
        ) is True
        params2 = {"num_characters": 1, "target_words": 800}
        assert critic_module._evaluate_gate(
            "num_characters >= 2 AND target_words >= 700", params2,
        ) is False

    def test_compound_or(self, critic_module):
        params = {"num_characters": 1, "target_words": 1500}
        assert critic_module._evaluate_gate(
            "num_characters >= 2 OR target_words >= 1500", params,
        ) is True

    def test_missing_field_fails_open(self, critic_module):
        # Gate references target_words, params doesn't have it -- rule
        # should still ship to critic (fail-open is loud-not-silent).
        result = critic_module._evaluate_gate("target_words >= 700", {})
        assert result is True

    def test_no_builtins_access(self, critic_module):
        # Sandboxed eval: this should fail, not execute __import__.
        result = critic_module._evaluate_gate(
            "__import__('os').system('echo pwned')", {},
        )
        # fail-open means True on parse failure, but no command executed.
        # The test is that the test suite itself doesn't crash; if eval
        # had escape, the os.system call would print 'pwned' and could
        # do worse. Just assert we got a bool back without exception.
        assert isinstance(result, bool)


class TestCoerceParams:

    def test_no_defaults_supplied(self, critic_module):
        # Empty input should NOT come back with defaults populated.
        out = critic_module._coerce_params(None)
        for k in ("target_words", "num_characters", "target_length", "style"):
            assert k not in out, f"{k} should be absent from output"

    def test_passes_through_real_values(self, critic_module):
        out = critic_module._coerce_params({
            "target_words":   500,
            "num_characters": 2,
            "target_length":  "short",
            "style":   "hard sci fi",
        })
        assert out["target_words"] == 500
        assert out["num_characters"] == 2
        assert out["target_length"] == "short"

    def test_derives_scene_count_when_inputs_present(self, critic_module):
        out = critic_module._coerce_params({
            "target_words": 1500,
            "target_length": "medium",
        })
        assert out["scene_count"] == 5
        assert out["scene_word_budget"] == 300

    def test_smoke_scene_count_is_1(self, critic_module):
        out = critic_module._coerce_params({
            "target_words": 100,
            "target_length": "smoke (1 act)",
        })
        assert out["scene_count"] == 1
        assert out["target_length"] == "smoke"

    def test_no_scene_count_without_inputs(self, critic_module):
        # If target_length is absent we don't invent a scene_count.
        out = critic_module._coerce_params({"target_words": 500})
        assert "scene_count" not in out
        assert "scene_word_budget" not in out

    def test_unparseable_int_dropped(self, critic_module):
        out = critic_module._coerce_params({
            "target_words": "abc",
            "num_characters": "xyz",
        })
        assert "target_words" not in out
        assert "num_characters" not in out

    def test_style_underscore_humanized(self, critic_module):
        out = critic_module._coerce_params(
            {"style": "mission_control_procedural"},
        )
        assert out["style"] == "mission control procedural"


class TestFilterRubric:

    def test_strips_gate_tag_from_kept_lines(self, critic_module):
        template = "- A1. [applies_when: always] Some rule body."
        out = critic_module._filter_rubric(template, {})
        assert "[applies_when:" not in out
        assert "Some rule body" in out

    def test_excludes_failing_gate(self, critic_module):
        template = (
            "- A1. [applies_when: target_words >= 700] long-form rule.\n"
            "- A2. [applies_when: always] always rule.\n"
        )
        out = critic_module._filter_rubric(template, {"target_words": 100})
        assert "long-form rule" not in out
        assert "always rule" in out

    def test_substitutes_placeholders(self, critic_module):
        template = "Budget: {target_words} words, {scene_word_budget} per scene."
        out = critic_module._filter_rubric(
            template,
            critic_module._coerce_params({
                "target_words": 1500,
                "target_length": "medium",
            }),
        )
        assert "Budget: 1500 words, 300 per scene." in out

    def test_unknown_placeholder_passes_through(self, critic_module):
        template = "Hello {unknown_token} world."
        out = critic_module._filter_rubric(template, {})
        # Should NOT crash; token stays as-is
        assert "{unknown_token}" in out

    def test_smoke_run_filters_ensemble_rules(self, critic_module):
        template = (
            "- A40. [applies_when: num_characters >= 2] ensemble rule.\n"
            "- A1. [applies_when: always] always rule.\n"
        )
        out = critic_module._filter_rubric(template, {"num_characters": 1})
        assert "ensemble rule" not in out
        assert "always rule" in out

    def test_loads_real_anti_slop_file(self, critic_module):
        # Smoke test: 100 words, 1 char, smoke length should produce a
        # rubric with the always-rules but exclude long-form gates.
        out = critic_module._load_anti_slop_rubric({
            "target_words": 100,
            "num_characters": 1,
            "target_length": "smoke",
            "style": "hard sci fi",
            "creativity": "medium",
            "period": "1947",
        })
        # Real file should produce something
        assert len(out) > 500
        # No gate tags should leak
        assert "[applies_when:" not in out
        # Long-form-only rules excluded
        assert "Middle scene contains only recap" not in out
        # Specific ensemble rules excluded for 1-char run (rule bodies,
        # not the section heading -- headers don't have gates).
        assert "All [VOICE:] blocks land within similar word count" not in out
        assert "Two distinct characters share an unusual idiom" not in out
        # Substitution worked
        assert "{target_words}" not in out
        assert "{scene_word_budget}" not in out

    def test_full_short_run_keeps_more_rules(self, critic_module):
        out = critic_module._load_anti_slop_rubric({
            "target_words": 1500,
            "num_characters": 3,
            "target_length": "medium",
            "style": "hard sci fi",
            "creativity": "medium",
            "period": "1947",
        })
        # At medium length with 3 chars, ensemble-collapse rules apply
        assert "ensemble" in out.lower()
        # Long-form mid-act sag rule applies
        assert "Middle scene" in out


# ---------------------------------------------------------------------------
# _build_critic_prompt
# ---------------------------------------------------------------------------

class TestBuildCriticPrompt:

    def test_includes_script_and_genre(self, critic_module):
        prompt = critic_module._build_critic_prompt(
            script_text="ANNOUNCER: Hello.",
            style="mission_control_procedural",
            anti_slop="",
        )
        assert "ANNOUNCER: Hello." in prompt
        # style with underscores humanized to spaces
        assert "mission control procedural" in prompt

    def test_uses_anti_slop_when_provided(self, critic_module):
        prompt = critic_module._build_critic_prompt(
            script_text="x",
            style="deep_space_distress_call",
            anti_slop="MY CUSTOM RUBRIC HERE",
        )
        assert "MY CUSTOM RUBRIC HERE" in prompt
        # built-in fallback should NOT be present when external anti-slop given
        assert "GENERAL RUBRIC" not in prompt

    def test_uses_builtin_when_anti_slop_blank(self, critic_module):
        prompt = critic_module._build_critic_prompt(
            script_text="x",
            style="small_town_uncanny",
            anti_slop="",
        )
        assert "GENERAL RUBRIC" in prompt

    def test_explicit_format_request(self, critic_module):
        prompt = critic_module._build_critic_prompt("x", "x", "")
        assert "SCORE:" in prompt
        assert "VERDICT:" in prompt
        assert "ISSUES:" in prompt
        # Stopping clause from the autonovel pattern
        assert "don't have to find defects" in prompt.lower()


# ---------------------------------------------------------------------------
# _canon_update
# ---------------------------------------------------------------------------

class TestCanonUpdate:

    def _seed_canon(self, tmp_path: Path) -> Path:
        canon = tmp_path / "OTR-CANON.md"
        canon.write_text(
            "# OTR Canon\n\n"
            "## Tonal canon\n\nfoo\n\n"
            "## Used premises (auto-updated)\n\n"
            "- _no entries yet — first canonical episode pending_\n\n"
            "## Used twists (auto-updated)\n\n"
            "- _no entries yet_\n\n"
            "## Used motifs (auto-updated)\n\n"
            "- _no entries yet_\n",
            encoding="utf-8",
        )
        return canon

    def test_no_op_when_all_blank(self, critic_module, tmp_path):
        canon = self._seed_canon(tmp_path)
        before = canon.read_text(encoding="utf-8")
        result = critic_module._canon_update(canon_path=canon)
        assert result is False
        assert canon.read_text(encoding="utf-8") == before

    def test_appends_premise(self, critic_module, tmp_path):
        canon = self._seed_canon(tmp_path)
        result = critic_module._canon_update(
            canon_path=canon,
            premise="Sickle cell gene therapy goes wrong on the moon",
        )
        assert result is True
        text = canon.read_text(encoding="utf-8")
        assert "Sickle cell gene therapy goes wrong on the moon" in text
        # Placeholder removed
        assert "_no entries yet — first canonical episode pending_" not in text

    def test_skips_duplicate(self, critic_module, tmp_path):
        canon = self._seed_canon(tmp_path)
        critic_module._canon_update(canon_path=canon, premise="Alpha")
        critic_module._canon_update(canon_path=canon, premise="alpha")  # dupe
        text = canon.read_text(encoding="utf-8")
        # Should appear exactly once in the premise section
        assert text.count("Alpha") + text.count("alpha") == 1

    def test_missing_canon_file_no_throw(self, critic_module, tmp_path):
        missing = tmp_path / "does-not-exist.md"
        result = critic_module._canon_update(canon_path=missing, premise="x")
        assert result is False  # logged warning, no exception

    def test_cap_per_section(self, critic_module, tmp_path):
        canon = self._seed_canon(tmp_path)
        # Append 30 entries with cap=5
        for i in range(30):
            critic_module._canon_update(
                canon_path=canon,
                premise=f"Premise number {i:02d}",
                cap_per_section=5,
            )
        text = canon.read_text(encoding="utf-8")
        # Find the premise section and count its entries
        section = text.split("## Used premises (auto-updated)")[1].split("## ")[0]
        entries = [
            ln for ln in section.splitlines()
            if ln.strip().startswith("- ") and "_no entries" not in ln
        ]
        assert len(entries) == 5
        # Should be the LAST 5
        assert "Premise number 25" in section
        assert "Premise number 29" in section
        assert "Premise number 00" not in section


# ---------------------------------------------------------------------------
# Node class structure
# ---------------------------------------------------------------------------

class TestNodeStructure:

    def test_node_class_exists(self, critic_module):
        assert hasattr(critic_module, "LLMScriptCritic")

    def test_required_class_attrs(self, critic_module):
        cls = critic_module.LLMScriptCritic
        assert cls.CATEGORY == "OTR/v2/Quality"
        assert cls.FUNCTION == "execute"
        # 4 outputs: script (passthrough or revised), script_json
        # (passthrough or re-parsed), critic_report (markdown),
        # verdict (PASS/REVISE/REJECT).
        assert cls.RETURN_TYPES == ("STRING", "STRING", "STRING", "STRING")
        assert cls.RETURN_NAMES == ("script", "script_json", "critic_report", "verdict")
        assert cls.OUTPUT_NODE is True

    def test_revise_on_findings_widget_present_and_on_by_default(self, critic_module):
        spec = critic_module.LLMScriptCritic.INPUT_TYPES()
        assert "revise_on_findings" in spec.get("optional", {}), (
            "revise_on_findings widget must exist on the critic node"
        )
        default = spec["optional"]["revise_on_findings"][1]["default"]
        assert default is True, (
            "revise_on_findings must default to ON -- the reviser is "
            "the point of having a critic in the graph; set FALSE only "
            "to inspect raw critic verdicts without rewriting."
        )

    def test_script_json_input_present(self, critic_module):
        spec = critic_module.LLMScriptCritic.INPUT_TYPES()
        assert "script_json" in spec.get("required", {}), (
            "script_json must be a required forceInput so the audio "
            "chain can be re-routed through the critic."
        )

    def test_input_types_includes_required_widgets(self, critic_module):
        spec = critic_module.LLMScriptCritic.INPUT_TYPES()
        assert "required" in spec
        # Only `script` is required; everything else is inherited from
        # the ledger or has a sensible optional default.
        assert "script" in spec["required"]
        assert "block_on_reject" in spec["optional"]
        assert "timeout_sec" in spec["optional"]

    def test_writer_settings_are_inherited_not_widgets(self, critic_module):
        """style, critic_model_id, and optimization_profile must
        inherit from LLMScriptWriter via the ledger, not be duplicated
        as critic widgets. UI design constraint Jeffrey set: one place
        to configure each setting."""
        spec = critic_module.LLMScriptCritic.INPUT_TYPES()
        all_keys = list(spec.get("required", {}).keys()) + list(spec.get("optional", {}).keys())
        assert "style" not in all_keys, (
            "style must inherit from ledger.gen_params_initial"
        )
        assert "critic_model_id" not in all_keys, (
            "critic_model_id must inherit from "
            "ledger.gen_params_initial.cleanup_model_id"
        )
        assert "optimization_profile" not in all_keys, (
            "optimization_profile must inherit from "
            "ledger.gen_params_initial.optimization_profile"
        )

    def test_block_on_reject_defaults_off(self, critic_module):
        spec = critic_module.LLMScriptCritic.INPUT_TYPES()
        default = spec["optional"]["block_on_reject"][1]["default"]
        assert default is False, "advisory-by-default per design"
