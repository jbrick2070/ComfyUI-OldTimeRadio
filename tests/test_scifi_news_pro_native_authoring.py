"""SciFi News Pro authors the NEW story in the episode language.

The news source stays as published: the dossier extraction and the voice
casting are internal receipts. Every pass whose output is spoken or shown --
pitch, treatment (the title is born there), spoken cast labels, the closing
news read and the whole-play script -- receives the row-owned native
authoring instruction ahead of its system prompt. Empty leaves every prompt
byte-identical, so English and Off never move.

CPU only, no model, no network. UTF-8 no BOM.
"""
from __future__ import annotations

import inspect

import pytest

from nodes import _otr_scifi_news_pro as PRO

RULE = "Escribe todo el diálogo en español."


class _Dump:
    def model_dump(self, *_a, **_k):
        return {}


class _Aliases:
    characters = ()


def _dossier():
    return PRO.DossierLLM(
        facts_to_keep=["one fact"],
        named_entities=PRO.NamedEntities(people=["Someone Real"]),
    )


def _treatment():
    return PRO.Treatment(
        title="T", dramatic_question="Q", setting="S",
        cast_shapes=[PRO.CastShape(name="Ada", role="r", want="w",
                                   pressure="p", register="clipped")],
        turn="turn", priced_ending={"choice": "c", "cost_paid": "p"},
        news_thread="thread", news_close_read="",
    )


@pytest.fixture
def seams(monkeypatch):
    """Stub the pack seams and the call so only the assembled system is observed."""
    captured = {}
    monkeypatch.setattr(PRO, "_seam", lambda pack, name: "SYS:" + name)

    def call(**kwargs):
        captured["system"] = kwargs["prompt"][0]["content"]
        return _Aliases()

    monkeypatch.setattr(PRO, "structured_call", call)

    def ladder(creative_fn, **kwargs):
        captured["system"] = kwargs["system"]
        return "", None, {}

    monkeypatch.setattr(PRO, "_run_markup_ladder", ladder)
    return captured


def _fn(messages, **_k):
    raise AssertionError("the stubbed call never reaches a slot")


PASSES = [
    ("scifi_news_pro_pitch_system",
     lambda rule: PRO._pass_pitch(
         _fn, None, _Dump(), [{"name": "c", "shape": "s"}],
         {"name": "n", "note": "x"}, n_max=2, language_instruction=rule)),
    ("scifi_news_pro_treatment_system",
     lambda rule: PRO._pass_treatment(
         _fn, None, _Dump(), _Dump(), {"name": "n", "note": "x"}, n_max=2,
         provenance={}, digest="", decision=None, language_instruction=rule)),
    ("scifi_news_pro_news_read_system",
     lambda rule: PRO._pass_news_read(
         _fn, None, _dossier(), {}, "digest", ["Ada"], language_instruction=rule)),
    ("scifi_news_pro_script_system",
     lambda rule: PRO._pass_script(
         _fn, None, _treatment(), "digest", PRO._build_envelope(1), ["Ada"],
         language_instruction=rule)),
]


@pytest.mark.parametrize("seam_name, run", PASSES, ids=[p[0] for p in PASSES])
def test_a_spoken_pass_leads_its_system_prompt_with_the_rule(seams, seam_name, run):
    run(RULE)
    assert seams["system"] == RULE + "\n\n" + "SYS:" + seam_name


@pytest.mark.parametrize("seam_name, run", PASSES, ids=[p[0] for p in PASSES])
def test_an_empty_rule_leaves_a_spoken_pass_byte_identical(seams, seam_name, run):
    run("")
    assert seams["system"] == "SYS:" + seam_name


def test_spoken_cast_labels_receive_the_rule(seams):
    PRO._pass_cast_aliases(_fn, None, _treatment(), language_instruction=RULE)
    assert seams["system"] == RULE + "\n\n" + PRO._CAST_ALIAS_SYSTEM
    PRO._pass_cast_aliases(_fn, None, _treatment(), language_instruction="")
    assert seams["system"] == PRO._CAST_ALIAS_SYSTEM


@pytest.mark.parametrize("fn", [
    PRO._pass_pitch, PRO._pass_treatment, PRO._pass_news_read,
    PRO._pass_cast_aliases, PRO._pass_script,
])
def test_every_spoken_pass_defaults_the_rule_to_empty(fn):
    param = inspect.signature(fn).parameters["language_instruction"]
    assert param.default == ""


@pytest.mark.parametrize("fn", [PRO._pass_dossier, PRO._pass_casting])
def test_internal_receipt_passes_take_no_rule(fn):
    assert "language_instruction" not in inspect.signature(fn).parameters


def test_the_runner_reads_the_stamp_once_and_hands_it_to_the_five_spoken_passes():
    src = inspect.getsource(PRO.run_scifi_news_pro_episode)
    assert "language_instruction = _EPLANG.native_authoring_instruction(meta)" in src
    assert src.count("language_instruction=language_instruction") == 5
