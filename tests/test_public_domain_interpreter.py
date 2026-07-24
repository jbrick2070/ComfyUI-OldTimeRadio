"""Public-domain source interpreter tests."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nodes._otr_public_domain_sources import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    PublicDomainBriefs,
    PublicDomainInterpreterError,
    build_public_domain_briefs,
)
from nodes._otr_source_payload import validate_interpreter_result

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "nodes" / "_otr_public_domain_sources.py"


def _sample_payload():
    return {
        "headline": "The Time Machine - The impossible arrival",
        "summary": "A shaken traveler returns with a strange machine.",
        "full_text": (
            "The traveler returns to the room, exhausted, and asks the witnesses "
            "to judge the machine by what it has cost him."
        ),
        "source": "Project Gutenberg",
        "date": "1895",
        "link": "https://www.gutenberg.org/ebooks/35",
        "seed_text": "The Time Machine by H. G. Wells\nUnit: The impossible arrival",
    }


def _good_briefs(**over):
    fields = {
        "casting_brief": (
            "The traveler is shaken but precise; two skeptical witnesses press "
            "him for proof without becoming villains."
        ),
        "script_brief": (
            "Adapt the arrival scene as a chamber drama: the traveler enters, "
            "the machine silently confirms his claim, and disbelief turns into "
            "fearful attention while the original ending remains intact."
        ),
        "news_close_brief": (
            "This episode adapts H. G. Wells's public-domain novel The Time "
            "Machine, preserving the return, the witnesses, and the unsettling "
            "proof of the journey."
        ),
        "key_terms": ["The Time Machine", "traveler", "witnesses"],
    }
    fields.update(over)
    return PublicDomainBriefs(**fields)


def test_briefs_pass_validate_interpreter_result():
    brief = _good_briefs()
    dump = validate_interpreter_result(brief, origin="public_domain")
    assert dump["casting_brief"] == brief.casting_brief
    assert dump["script_brief"] == brief.script_brief
    assert dump["key_terms"] == brief.key_terms
    assert brief.prompt_version == PROMPT_VERSION
    assert brief.schema_version == SCHEMA_VERSION


def test_wrapper_translates_only_public_domain_interpreter_error():
    from nodes import _otr_source_payload as osp
    from nodes import _otr_public_domain_sources as pds

    def _raise_pd_error(*, technical_fn, payload, model_id):
        raise pds.PublicDomainInterpreterError(attempts=2, reason="test-boom")

    with patch(
        "nodes._otr_public_domain_sources.build_public_domain_briefs",
        side_effect=_raise_pd_error,
    ):
        bank = SimpleNamespace(
            source_bank_id="public_domain_story",
            fetcher="public_domain_source",
            interpreter="public_domain_interpreter",
        )
        interp_fn = osp.resolve_interpreter(bank)
        with pytest.raises(osp.SourceInterpretError, match="test-boom") as ei:
            interp_fn(
                bank=bank,
                payload=_sample_payload(),
                technical_fn=lambda *a, **kw: "{}",
                model_id="test",
            )
        assert isinstance(ei.value.__cause__, pds.PublicDomainInterpreterError)


def test_unexpected_public_domain_interpreter_errors_propagate_hard():
    def _blow_up(msgs, *, temperature, max_new_tokens):
        raise RuntimeError("coding bug")

    with pytest.raises(RuntimeError, match="coding bug"):
        build_public_domain_briefs(
            technical_fn=_blow_up,
            payload=_sample_payload(),
            model_id="test",
            max_attempts=1,
        )


def test_build_public_domain_briefs_rejects_max_attempts_lt_one():
    with pytest.raises(ValueError, match="max_attempts"):
        build_public_domain_briefs(
            technical_fn=lambda *a, **kw: "{}",
            payload=_sample_payload(),
            max_attempts=0,
        )


def test_public_domain_briefs_preserve_optional_terms_without_caps():
    assert _good_briefs(key_terms=[]).key_terms == []
    terms = [f"term{i}" for i in range(20)] + ["x" * 200]
    brief = _good_briefs(key_terms=terms)
    assert brief.key_terms == terms


def test_public_domain_prompt_carries_narrow_safety_guidance(monkeypatch):
    captured = {}

    def _fake_structured_call(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        return _good_briefs()

    monkeypatch.setattr(
        "nodes._otr_public_domain_sources.structured_call",
        _fake_structured_call,
    )
    brief = build_public_domain_briefs(
        technical_fn=lambda *a, **kw: "{}",
        payload=_sample_payload(),
        model_id="test-model",
    )
    prompt_text = "\n".join(m["content"] for m in captured["prompt"])
    for term in ("no profanity", "guns/knives/weapons", "sexual/nudity"):
        assert term in prompt_text
    assert brief.model_id == "test-model"
    assert brief.attempts == 0
    assert brief.source_hash


def test_prompt_asks_for_named_and_role_designated_cast_in_order(monkeypatch):
    """Runtime cast extraction must pull the source's OWN people -- proper names
    when the text gives them AND role-designations when it does not (the Time
    Traveller), in appearance order, never a generic placeholder."""
    captured = {}

    def _fake_structured_call(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        return _good_briefs(character_names=["the Time Traveller", "Filby"])

    monkeypatch.setattr(
        "nodes._otr_public_domain_sources.structured_call",
        _fake_structured_call,
    )
    build_public_domain_briefs(
        technical_fn=lambda *a, **kw: "{}",
        payload=_sample_payload(),
        model_id="test-model",
    )
    prompt_text = "\n".join(m["content"] for m in captured["prompt"]).lower()
    assert "role-designation" in prompt_text          # the Time Traveller case
    assert "order of appearance" in prompt_text
    assert "never invent" in prompt_text
    assert "placeholder" in prompt_text                # bans 'Witness 1'


def test_role_designated_cast_survives_the_field_cleaner():
    """A source that names no one (Wells) still yields a faithful cast: the
    text's own role-designations, kept verbatim."""
    brief = _good_briefs(character_names=["the Time Traveller", "the Medical Man", "Filby"])
    assert brief.character_names == ["the Time Traveller", "the Medical Man", "Filby"]


def test_empty_cast_is_rejected_and_retried_to_failure():
    """The cast is a REQUIRED generation input, not best-effort. A source brain
    that returns valid briefs but omits the cast must be retried, then fail loud
    -- never fall through to a generic placeholder cast."""
    import json

    body = json.dumps({
        "casting_brief": _good_briefs().casting_brief,
        "script_brief": _good_briefs().script_brief,
        "news_close_brief": _good_briefs().news_close_brief,
        "key_terms": ["The Time Machine", "traveler", "witnesses"],
        # character_names deliberately omitted -> validator must reject
    })
    with pytest.raises(PublicDomainInterpreterError):
        build_public_domain_briefs(
            technical_fn=lambda *a, **kw: body,
            payload=_sample_payload(),
            model_id="test",
            max_attempts=2,
        )


def test_a_cast_bearing_brief_passes_the_ladder():
    """The same source brain, now returning the story's real cast, passes."""
    import json

    body = json.dumps({
        "casting_brief": _good_briefs().casting_brief,
        "script_brief": _good_briefs().script_brief,
        "news_close_brief": _good_briefs().news_close_brief,
        "key_terms": ["The Time Machine", "traveler", "witnesses"],
        "character_names": ["the Time Traveller", "the Medical Man", "Filby"],
    })
    brief = build_public_domain_briefs(
        technical_fn=lambda *a, **kw: body,
        payload=_sample_payload(),
        model_id="test",
        max_attempts=2,
    )
    assert brief.character_names[0] == "the Time Traveller"


def test_public_domain_module_import_posture():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", "") or ""
            names = [a.name for a in getattr(node, "names", [])]
            combined = module + " " + " ".join(names)
            assert "OTR_LedgerScriptWriter" not in combined
            assert "news_interpreter" not in combined
            assert "_otr_story_routing" not in combined
