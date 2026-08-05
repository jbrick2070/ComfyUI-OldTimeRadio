"""Shakespeare source interpreter tests."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nodes._otr_shakespeare_sources import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    ShakespeareBriefs,
    ShakespeareInterpreterError,
    build_shakespeare_briefs,
)
from nodes._otr_source_payload import validate_interpreter_result

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "nodes" / "_otr_shakespeare_sources.py"


def _sample_payload():
    return {
        "headline": "Macbeth, Act 1, Scene 3",
        "summary": "Three witches greet Macbeth with titles that unsettle him.",
        "full_text": (
            "FIRST WITCH: All hail, Macbeth! BANQUO: Good sir, why do you start?"
        ),
        "source": "Folger Shakespeare",
        "date": "c. 1606 | CC BY-NC 3.0",
        "link": "https://www.folger.edu/explore/shakespeares-works/macbeth/read/",
        "seed_text": "Macbeth, Act 1, Scene 3\nSpeakers: FIRST WITCH, BANQUO",
    }


def _good_briefs(**over):
    fields = {
        "casting_brief": (
            "Macbeth hears the new titles with controlled dread; Banquo is "
            "alert, skeptical, and careful with the witches' promises."
        ),
        "script_brief": (
            "Adapt the heath encounter as compressed radio pressure: the "
            "witches name Macbeth's present and possible future, Banquo tests "
            "the promise, and Macbeth cannot stop reaching toward the title."
        ),
        "news_close_brief": (
            "This episode adapts Macbeth, Act 1, Scene 3 from Folger "
            "Shakespeare, a CC BY-NC scene source used here for noncommercial "
            "study and adaptation."
        ),
        "key_terms": ["Macbeth", "Banquo", "witches", "prophecy"],
    }
    fields.update(over)
    return ShakespeareBriefs(**fields)


def test_briefs_pass_validate_interpreter_result():
    brief = _good_briefs()
    dump = validate_interpreter_result(brief, origin="shakespeare")
    assert dump["casting_brief"] == brief.casting_brief
    assert dump["script_brief"] == brief.script_brief
    assert dump["key_terms"] == brief.key_terms
    assert brief.prompt_version == PROMPT_VERSION
    assert brief.schema_version == SCHEMA_VERSION


def test_wrapper_translates_only_shakespeare_interpreter_error():
    from nodes import _otr_source_payload as osp
    from nodes import _otr_shakespeare_sources as shx

    def _raise_shx_error(*, technical_fn, payload, model_id):
        raise shx.ShakespeareInterpreterError(attempts=2, reason="test-boom")

    with patch(
        "nodes._otr_shakespeare_sources.build_shakespeare_briefs",
        side_effect=_raise_shx_error,
    ):
        bank = SimpleNamespace(
            source_bank_id="shakespeare",
            fetcher="shakespeare_folger",
            interpreter="shakespeare_interpreter",
        )
        interp_fn = osp.resolve_interpreter(bank)
        with pytest.raises(osp.SourceInterpretError, match="test-boom") as ei:
            interp_fn(
                bank=bank,
                payload=_sample_payload(),
                technical_fn=lambda *a, **kw: "{}",
                model_id="test",
            )
        assert isinstance(ei.value.__cause__, shx.ShakespeareInterpreterError)


def test_unexpected_shakespeare_interpreter_errors_propagate_hard():
    def _blow_up(msgs, *, temperature, max_new_tokens):
        raise RuntimeError("coding bug")

    with pytest.raises(RuntimeError, match="coding bug"):
        build_shakespeare_briefs(
            technical_fn=_blow_up,
            payload=_sample_payload(),
            model_id="test",
            max_attempts=1,
        )


def test_build_shakespeare_briefs_rejects_max_attempts_lt_one():
    with pytest.raises(ValueError, match="max_attempts"):
        build_shakespeare_briefs(
            technical_fn=lambda *a, **kw: "{}",
            payload=_sample_payload(),
            max_attempts=0,
        )


def test_shakespeare_briefs_preserve_optional_terms_without_caps():
    assert _good_briefs(key_terms=[]).key_terms == []
    terms = [f"term{i}" for i in range(20)] + ["x" * 200]
    brief = _good_briefs(key_terms=terms)
    assert brief.key_terms == terms


def test_shakespeare_prompt_keeps_rights_terms_and_no_content_guardrail(monkeypatch):
    captured = {}

    def _fake_structured_call(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        return _good_briefs()

    monkeypatch.setattr(
        "nodes._otr_shakespeare_sources.structured_call",
        _fake_structured_call,
    )
    brief = build_shakespeare_briefs(
        technical_fn=lambda *a, **kw: "{}",
        payload=_sample_payload(),
        model_id="test-model",
    )
    prompt_text = "\n".join(m["content"] for m in captured["prompt"])
    # RIGHTS terms stay -- they are a licensing fact about the source.
    assert "CC BY-NC" in prompt_text
    assert "Folger Shakespeare" in prompt_text
    # SAFETY terms are GONE (operator directive 2026-08-05). Telling the model
    # to avoid "guns/knives/weapons" while handing it MACBETH is a fidelity
    # defect: "Is this a dagger which I see before me" is the play. This
    # assertion is inverted on purpose so the clause cannot creep back in.
    for term in ("no profanity", "guns/knives/weapons", "sexual/nudity"):
        assert term not in prompt_text, (
            "a content guardrail returned to the shakespeare prompt: %r" % term)
    assert brief.model_id == "test-model"
    assert brief.attempts == 0
    assert brief.source_hash


def test_shakespeare_module_import_posture():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", "") or ""
            names = [a.name for a in getattr(node, "names", [])]
            combined = module + " " + " ".join(names)
            assert "OTR_LedgerScriptWriter" not in combined
            assert "news_interpreter" not in combined
            assert "_otr_story_routing" not in combined
