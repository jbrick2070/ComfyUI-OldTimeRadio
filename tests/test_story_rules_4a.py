"""tests/test_story_rules_4a.py

Multi-modal story schema STAGE 4 CHUNK 4A -- story-content rule packs
(STAGE4_SUBPLAN v4 FINAL; kibitz r1-r4 + 3-lens Sonnet fan-out).

Pins:
  1. EXTRACTION FIDELITY: nodes/story_rules/science_news.json ==
     the Python fixture constants (pattern sources, replacement pairs,
     banned phrases) -- generated from them, pinned forever after.
  2. Loader: lazy, fail-loud matrix (dup keys, control chars = the JSON
     `\\b` backspace trap, length cap, bad regex, bad replacement
     template, unknown/missing keys, stem/coordinate, unregistered stem,
     missing science), _clear_caches, missing-pack semantics.
  3. FLAG PARITY: rules=science == rules=None (fixture) on trip fixtures,
     for every wrapper (reason strings, not booleans).
  4. FAIL-LOUD THREADING: compose_line/compose_announcer_outro on a bank
     without a rules pack raise UnknownStoryRulesError (no swallow).
  5. BUG-LOCAL-417: compose_line_draft keeps _SYSTEM_PROMPT object
     identity on science+repo-None; a NON-science bank routes through the
     resolver even with repo None.
  6. AST guards: (A) no production reads of the vocabulary constants
     outside their defining modules; (B) production wrapper callers pass
     rules=; both with synthetic self-tests.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from nodes import _otr_line_hygiene as hyg
from nodes import _otr_story_rules as sr
from nodes import _otr_line_composer as lc
from nodes._otr_stage3_validators import DEFAULT_BANNED_PHRASES

_REPO = Path(__file__).resolve().parent.parent
_NODES = _REPO / "nodes"
_PACK = _NODES / "story_rules" / "science_news.json"

_CONST_SETS = {
    "cliche_patterns": hyg._CLICHE_RES,
    "stage_business_patterns": hyg._STAGE_BUSINESS_RES,
    "on_the_nose_patterns": hyg._ON_THE_NOSE_RES,
    "banned_thesis_patterns": hyg._BANNED_THESIS_RES,
    "personal_cost_patterns": hyg._PERSONAL_COST_BOILERPLATE_RES,
}

_VOCAB_CONSTS = ("_CLICHE_RES", "_STAGE_BUSINESS_RES", "_ON_THE_NOSE_RES",
                 "_CLICHE_REPLACEMENTS", "_BANNED_THESIS_RES",
                 "_PERSONAL_COST_BOILERPLATE_RES", "DEFAULT_BANNED_PHRASES")


@pytest.fixture(autouse=True)
def _fresh():
    sr._clear_caches()
    yield
    sr._clear_caches()


def _mutated_pack(tmp_path, monkeypatch, mutate):
    raw = json.loads(_PACK.read_text(encoding="utf-8"))
    mutate(raw)
    scratch = tmp_path / "story_rules"
    scratch.mkdir()
    (scratch / "science_news.json").write_text(
        json.dumps(raw), encoding="utf-8")
    monkeypatch.setattr(sr, "_STORY_RULES_ROOT", scratch)
    sr._clear_caches()


# ---------------------------------------------------------------------------
# 1. Extraction fidelity
# ---------------------------------------------------------------------------
class TestExtractionFidelity:
    def test_pattern_sources_match_constants(self):
        raw = json.loads(_PACK.read_text(encoding="utf-8"))
        for field, consts in _CONST_SETS.items():
            assert raw[field] == [rx.pattern for rx in consts], field

    def test_replacements_and_phrases_match(self):
        raw = json.loads(_PACK.read_text(encoding="utf-8"))
        assert raw["cliche_replacements"] == [
            [rx.pattern, repl] for rx, repl in hyg._CLICHE_REPLACEMENTS]
        assert raw["banned_phrases"] == list(DEFAULT_BANNED_PHRASES)

    def test_loaded_pack_compiles_equivalently(self):
        rules = sr.resolve_story_rules("science_news")
        assert [rx.pattern for rx in rules.cliche] == [
            rx.pattern for rx in hyg._CLICHE_RES]
        assert all(rx.flags == hyg._CLICHE_RES[0].flags
                   for rx in rules.cliche)
        assert [(rx.pattern, r) for rx, r in rules.cliche_replacements] == [
            (rx.pattern, r) for rx, r in hyg._CLICHE_REPLACEMENTS]


# ---------------------------------------------------------------------------
# 2. Loader contract
# ---------------------------------------------------------------------------
class TestLoader:
    def test_lazy_and_caches(self, monkeypatch):
        import importlib
        import sys
        calls = []
        real = Path.read_text

        def _spy(self, *a, **k):
            if "story_rules" in str(self):
                calls.append(str(self))
            return real(self, *a, **k)

        monkeypatch.setattr(Path, "read_text", _spy)
        original = sys.modules.get("nodes._otr_story_rules")
        try:
            sys.modules.pop("nodes._otr_story_rules", None)
            mod = importlib.import_module("nodes._otr_story_rules")
            assert calls == [], f"import-time I/O: {calls}"
            mod._clear_caches()
            mod.resolve_story_rules("science_news")
            assert calls
        finally:
            if original is not None:
                sys.modules["nodes._otr_story_rules"] = original

    def test_missing_pack_semantics(self):
        # custom_source_bank is registered + runnable:false + has NO rules
        # pack: legal at sweep time, hard error on explicit resolve.
        ids = sr.list_rules_ids()
        assert "science_news" in ids
        assert "media_archive" in ids
        assert "public_domain_story" in ids
        assert "custom_source_bank" not in ids
        with pytest.raises(sr.UnknownStoryRulesError):
            sr.resolve_story_rules("custom_source_bank")

    def test_public_domain_rules_load(self):
        rules = sr.resolve_story_rules("public_domain_story")
        assert rules.rules_id == "public_domain_story"
        assert "smoking" in rules.banned_phrases

    def test_get_story_rules_meta_paths(self):
        assert sr.get_story_rules({}).rules_id == "science_news"
        assert sr.get_story_rules(None).rules_id == "science_news"
        assert sr.get_story_rules(
            {"source_bank": "science_news"}).rules_id == "science_news"

    @pytest.mark.parametrize("mutate,fragment", [
        (lambda d: d.update(extra=1), "unknown key"),
        (lambda d: d.pop("banned_phrases"), "missing required"),
        (lambda d: d.update(schema_version="v9"), "schema_version"),
        (lambda d: d.update(rules_id="other"), "does not match"),
        (lambda d: d["cliche_patterns"].append("(unclosed"), "invalid regex"),
        (lambda d: d["cliche_patterns"].append("x" * 201), "exceeds"),
        (lambda d: d["cliche_patterns"].append("bad\bboundary"),
         "control character"),
        (lambda d: d["cliche_replacements"].append([r"\bfoo\b", r"\9"]),
         "invalid replacement template"),
        (lambda d: d["cliche_replacements"].append(["only-one"]),
         "two-string"),
    ])
    def test_fail_loud_matrix(self, tmp_path, monkeypatch, mutate, fragment):
        _mutated_pack(tmp_path, monkeypatch, mutate)
        with pytest.raises(sr.StoryRulesValidationError) as ei:
            sr.list_rules_ids()
        assert fragment in str(ei.value)

    def test_duplicate_json_key_rejected(self, tmp_path, monkeypatch):
        raw_text = _PACK.read_text(encoding="utf-8")
        dup = raw_text.rstrip().rstrip("}") + ', "label": "dup twice"}'
        scratch = tmp_path / "story_rules"
        scratch.mkdir()
        (scratch / "science_news.json").write_text(dup, encoding="utf-8")
        monkeypatch.setattr(sr, "_STORY_RULES_ROOT", scratch)
        sr._clear_caches()
        with pytest.raises(sr.StoryRulesValidationError) as ei:
            sr.list_rules_ids()
        assert "duplicate JSON key" in str(ei.value)

    def test_unregistered_stem_and_missing_science(self, tmp_path,
                                                   monkeypatch):
        raw = _PACK.read_text(encoding="utf-8")
        scratch = tmp_path / "story_rules"
        scratch.mkdir()
        (scratch / "science_news.json").write_text(raw, encoding="utf-8")
        (scratch / "not_a_bank.json").write_text(raw, encoding="utf-8")
        monkeypatch.setattr(sr, "_STORY_RULES_ROOT", scratch)
        sr._clear_caches()
        with pytest.raises(sr.StoryRulesValidationError) as ei:
            sr.list_rules_ids()
        assert "not a registered source_bank_id" in str(ei.value)
        scratch2 = tmp_path / "story_rules2"
        scratch2.mkdir()
        monkeypatch.setattr(sr, "_STORY_RULES_ROOT", scratch2)
        sr._clear_caches()
        with pytest.raises(sr.StoryRulesValidationError) as ei2:
            sr.list_rules_ids()
        assert "science_news" in str(ei2.value)


# ---------------------------------------------------------------------------
# 3. Flag parity (rules=science == fixture defaults; reason strings)
# ---------------------------------------------------------------------------
_TRIP = {
    "cliche": "Listen, you're playing with fire, and we must act.",
    "stage_business": "I'll go check the readings and report back.",
    "on_the_nose": "I'm so scared, and this is dangerous.",
    "thesis": "Tonight's revelation is the cost of speed.",
    "personal_cost": "Think about the trust they will lose either way.",
    "clean": "The reactor hums a half-step flat tonight.",
}


class TestFlagParity:
    def test_all_wrappers_parity(self):
        rules = sr.resolve_story_rules("science_news")
        for text in _TRIP.values():
            assert hyg.flag_cliche(text) == hyg.flag_cliche(
                text, rules=rules)
            assert hyg.flag_stage_business(text) == hyg.flag_stage_business(
                text, rules=rules)
            assert hyg.flag_on_the_nose(text) == hyg.flag_on_the_nose(
                text, rules=rules)
            assert hyg.flag_thesis_close(text) == hyg.flag_thesis_close(
                text, rules=rules)
            assert (hyg.flag_personal_cost_boilerplate(text)
                    == hyg.flag_personal_cost_boilerplate(text, rules=rules))
            assert hyg.find_cliche_phrase(text) == hyg.find_cliche_phrase(
                text, rules=rules)
            assert hyg.repair_cliche_span(text) == hyg.repair_cliche_span(
                text, rules=rules)

    def test_every_json_cliche_pattern_has_live_repair(self):
        # r4 S2: the science pack twin of the every-pattern-has-a-repair
        # coverage lock, run against the LOADED pack.
        rules = sr.resolve_story_rules("science_news")
        probes = (
            "You're playing with fire.", "This changes everything.",
            "We're not leaving anything to chance.",
            "Leaving nothing to chance.", "There's no turning back.",
            "Against all odds, we made it.", "It hangs in the balance.",
            "Over my dead body.", "Not on my watch.",
            "Some things are best left buried.", "We are running out of time.",
            "Act before it's too late.", "Safety first.",
            "The plan goes up in smoke.",
        )
        for probe in probes:
            hit, _ = hyg.flag_cliche(probe, rules=rules)
            assert hit, probe
            repaired = hyg.repair_cliche_span(probe, rules=rules)
            assert not hyg.flag_cliche(repaired, rules=rules)[0], probe


# ---------------------------------------------------------------------------
# 4. Fail-loud threading
# ---------------------------------------------------------------------------
class TestFailLoudThreading:
    def test_compose_line_raises_on_missing_rules_pack(self):
        req = lc.LineRequest(
            speaker="MARGOT", intent="steady", mood="calm",
            target_words=8, canon_header="", last_lines=[])
        with pytest.raises(sr.UnknownStoryRulesError):
            lc.compose_line(
                creative_fn=lambda *a, **k: "A quiet line.",
                req=req, source_bank_id="custom_source_bank")

    def test_outro_raises_on_missing_rules_pack(self):
        with pytest.raises(sr.UnknownStoryRulesError):
            lc.compose_announcer_outro(
                creative_fn=lambda *a, **k: "Good night.",
                script_brief="brief", news_close_brief="close",
                intro_text="intro", source_bank_id="custom_source_bank")


# ---------------------------------------------------------------------------
# 5. BUG-LOCAL-417 draft routing
# ---------------------------------------------------------------------------
class TestBug417:
    def test_science_repo_none_keeps_object_identity(self, monkeypatch):
        captured = {}
        real = lc._build_user_prompt

        def _spy_build(req):
            return real(req)

        monkeypatch.setattr(lc, "_build_user_prompt", _spy_build)

        def _fn(prompt=None, **k):
            captured["system"] = prompt[0]["content"]
            return "A quiet line about the machine."

        req = lc.LineRequest(
            speaker="MARGOT", intent="steady", mood="calm",
            target_words=8, canon_header="", last_lines=[])
        lc.compose_line_draft(creative_fn=_fn, req=req,
                              source_bank_id="science_news")
        assert captured["system"] is lc._SYSTEM_PROMPT

    def test_non_science_repo_none_routes_resolver(self, monkeypatch):
        from nodes import _otr_creative_prompt_router as router
        seen = []
        monkeypatch.setattr(
            router, "resolve_creative_system_prompt",
            lambda repo_id, phase, source_bank_id="science_news":
                seen.append((repo_id, source_bank_id)) or "PACK PROMPT")

        def _fn(prompt=None, **k):
            return "A quiet line about the archive."

        req = lc.LineRequest(
            speaker="MARGOT", intent="steady", mood="calm",
            target_words=8, canon_header="", last_lines=[])
        lc.compose_line_draft(creative_fn=_fn, req=req,
                              source_bank_id="media_archive")
        assert seen == [(None, "media_archive")]


# ---------------------------------------------------------------------------
# 6. AST guards (+ synthetic self-tests)
# ---------------------------------------------------------------------------
_DEFINING = {"_otr_line_hygiene.py", "_otr_stage3_validators.py"}
_WRAPPERS = {"flag_cliche", "flag_stage_business", "flag_on_the_nose",
             "flag_thesis_close", "flag_personal_cost_boilerplate",
             "find_cliche_phrase", "repair_cliche_span"}


def _scan_constant_reads(source: str, relname: str):
    offenders = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _VOCAB_CONSTS:
            offenders.append(f"{relname}:{node.lineno}:{node.id}")
        elif isinstance(node, ast.ImportFrom):
            # loop var `imp` NOT `alias` (B7 marker; CW-6 gotcha).
            for imp in node.names:
                if imp.name in _VOCAB_CONSTS:
                    offenders.append(
                        f"{relname}:{node.lineno}:import {imp.name}")
    return offenders


class TestAstGuards:
    def test_no_production_reads_of_vocab_constants(self):
        offenders = []
        for path in _NODES.rglob("*.py"):
            rel = path.relative_to(_NODES).as_posix()
            if Path(rel).name in _DEFINING:
                continue
            offenders += _scan_constant_reads(
                path.read_text(encoding="utf-8"), rel)
        assert not offenders, offenders

    def test_guard_a_self_test_fires(self):
        synthetic = "from nodes._otr_line_hygiene import _CLICHE_RES\n"
        assert _scan_constant_reads(synthetic, "synthetic.py")

    def test_production_wrapper_callers_pass_rules(self):
        # Guard B: in nodes/, every call to a wrapper passes rules= --
        # EXCEPT inside the two defining modules (fixture-default lane).
        offenders = []
        for path in _NODES.rglob("*.py"):
            rel = path.relative_to(_NODES).as_posix()
            if Path(rel).name in _DEFINING:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for call in ast.walk(tree):
                if not isinstance(call, ast.Call):
                    continue
                name = getattr(call.func, "id",
                               getattr(call.func, "attr", ""))
                if name in _WRAPPERS:
                    if "rules" not in {k.arg for k in call.keywords}:
                        offenders.append(f"{rel}:{call.lineno}:{name}")
        assert not offenders, (
            f"production wrapper calls missing rules= : {offenders}")

    def test_guard_b_self_test_fires(self):
        tree = ast.parse("x = flag_cliche('text')\n")
        hits = [c for c in ast.walk(tree)
                if isinstance(c, ast.Call)
                and getattr(c.func, "id", "") in _WRAPPERS
                and "rules" not in {k.arg for k in c.keywords}]
        assert hits
