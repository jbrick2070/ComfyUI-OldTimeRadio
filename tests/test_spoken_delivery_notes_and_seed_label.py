"""Two things a listener heard on air on 2026-10-16 (RTX 4060 8 GB box),
reproduced from the exact lines the writers produced that night.

1. Kokoro read parenthesised delivery notes aloud. The ledger keeps the line
   as written on purpose -- captions show the note and the stills read it
   (operator ruling 2026-08-05, `_otr_captions.py`) -- and every voice engine
   drops it when it speaks. Bark, Chatterbox and IndexTTS2 did; Kokoro's
   `prepare_text` was the identity, and on 8 GB boxes Kokoro voices the
   characters.
2. The scifi_news_pro close credited a typed premise to its provenance label:
   "...sank thirty years ago, according to User Seed." The custom-premise
   article carried "User Seed" in its OUTLET field, and the close pass is told
   to name who reported the find.

CPU only, no model, no network. UTF-8 no BOM.
"""
from __future__ import annotations

import json

import pytest

from nodes import _otr_media_archive_interpreter as MEDIA
from nodes import _otr_public_domain_sources as PUBLIC
from nodes import _otr_scifi_news_pro as PRO
from nodes import _otr_shakespeare_sources as BARD
from nodes import _otr_source_payload as OSP
from nodes import news_interpreter as SCIENCE
from nodes.OTR_LedgerScriptWriter import _resolve_inputs
from nodes._otr_audio_engines import eng_bark, eng_chatterbox, eng_indextts2, eng_kokoro
from nodes._otr_script_prep import keep_spoken_parentheticals

#: Verbatim from the shipped ledgers and server logs of that night.
LEAKED = [
    ("(distorted, urgent static) ...the pressure drops. the ice is coming.",
     "...the pressure drops. the ice is coming."),
    ("(sharp, questioning whisper) Keeper, that is just interference. "
     "The barometer is stable.",
     "Keeper, that is just interference. The barometer is stable."),
    ("(louder, more urgent) ...heavy spray hitting the beam... the reef mouth "
     "is obscured... you must alter course...",
     "...heavy spray hitting the beam... the reef mouth is obscured... you "
     "must alter course..."),
    ("(measured, deeply concerned tone) Daniel, the data shows it.",
     "Daniel, the data shows it."),
    ("(Sound of a heavy metal clasp being forced open) Then we ride this "
     "tide, Voss.",
     "Then we ride this tide, Voss."),
]


@pytest.mark.parametrize("line, spoken", LEAKED, ids=[s[:24] for s, _ in LEAKED])
def test_kokoro_never_reads_a_delivery_note(line, spoken):
    assert eng_kokoro.KokoroEngine().prepare_text(line) == spoken


@pytest.mark.parametrize("engine", [
    eng_kokoro.KokoroEngine, eng_chatterbox.ChatterboxEngine,
    eng_indextts2.IndexTTS2Engine, eng_bark.BarkEngine,
], ids=lambda cls: cls.__name__)
def test_no_local_engine_speaks_the_note(engine):
    """One delivery rule for every local voice: the note is never spoken."""
    prepared = engine().prepare_text(LEAKED[0][0])
    assert "distorted" not in prepared and "(" not in prepared
    assert "the pressure drops" in prepared


def test_a_line_without_markup_reaches_kokoro_unchanged():
    """The announcer's ordinary lines: nothing to clean, nothing moves."""
    line = "Tonight, from the lonely spire: The Thirty Year Echo."
    assert eng_kokoro.KokoroEngine().prepare_text(line) == line


def test_a_verbatim_parenthetical_is_still_spoken_on_kokoro():
    """Folger's "(God shield us!)" is Bottom's speech, not a direction. The
    voice node drops the glyphs on a verbatim row BEFORE any engine prep, so
    Kokoro's cleaner finds no parenthetical to remove."""
    line = ("to bring in (God shield us!) a lion among ladies is a most "
            "dreadful thing")
    delivered = keep_spoken_parentheticals(line).strip()
    assert "God shield us!" in eng_kokoro.KokoroEngine().prepare_text(delivered)


PREMISE = (
    "A 1950s lighthouse keeper on a storm-cut island picks up radio calls "
    "from a cargo ship that sank thirty years ago, and the calls describe "
    "tonight's weather."
)


@pytest.fixture
def typed_premise(monkeypatch):
    monkeypatch.delenv("OTR_SOURCE_SNAPSHOT_MANIFEST", raising=False)
    return _resolve_inputs(custom_premise=PREMISE)


def test_a_typed_premise_has_no_outlet(typed_premise):
    assert typed_premise["seed_source"] == "custom_premise"
    assert typed_premise["news_article"]["source"] == ""
    assert "User Seed" not in json.dumps(typed_premise["news_article"])


def _close_prompt(monkeypatch, article):
    """The user prompt the close pass sends, built the way the lane builds it."""
    captured = {}

    def call(**kwargs):
        captured["user"] = kwargs["prompt"][1]["content"]
        return PRO.NewsCloseRead(news_close_read="placeholder")

    monkeypatch.setattr(PRO, "structured_call", call)
    monkeypatch.setattr(PRO, "_seam", lambda pack, name: "SYS")
    provenance = {key: str(article.get(key) or "")
                  for key in ("headline", "source", "date", "link")}
    dossier = PRO.DossierLLM(
        facts_to_keep=["a cargo ship sank thirty years ago"],
        named_entities=PRO.NamedEntities())
    PRO._pass_news_read(
        lambda *_a, **_k: "", None, dossier, provenance,
        PRO._build_source_preview(article), ["Captain Thorne"])
    return captured["user"]


def test_the_close_is_never_handed_a_label_as_its_source(monkeypatch, typed_premise):
    user = _close_prompt(monkeypatch, typed_premise["news_article"])
    assert "User Seed" not in user
    assert OSP.NO_NAMED_SOURCE_NOTE in user


def test_a_real_outlet_is_still_offered_to_the_close(monkeypatch):
    article = {
        "headline": "Wreck found", "summary": "", "full_text": "A wreck was found.",
        "source": "ScienceDaily", "date": "2026-10-16",
        "link": "https://www.sciencedaily.com/releases/2026/10/x.htm",
        "seed_text": "Wreck found",
    }
    user = _close_prompt(monkeypatch, article)
    assert "ScienceDaily" in user
    assert OSP.NO_NAMED_SOURCE_NOTE not in user


# Every bank with a source contract hands a typed premise to its own
# interpreter, and every one of them is asked for a closing source note --
# "who found or reported it", "a Folger/noncommercial source note", "a source
# attribution note". The operator: "there are like 5 different source banks
# you need to scrub for the same defect fix."

def _prompt_text(messages):
    return "\n".join(m["content"] for m in messages) if isinstance(messages, list) else messages


def _science(article):
    return SCIENCE._build_user_prompt(
        headline=article["headline"], outlet=article["source"],
        pub_date=article["date"], cleaned_body=article["full_text"])


BANK_PROMPTS = [
    ("media_archive", lambda a: _prompt_text(MEDIA._build_prompt(a))),
    ("public_domain", lambda a: _prompt_text(PUBLIC._build_interpreter_prompt(a))),
    ("shakespeare", lambda a: _prompt_text(BARD._build_interpreter_prompt(a))),
    ("science_news", _science),
]


@pytest.mark.parametrize("bank, build", BANK_PROMPTS, ids=[b for b, _ in BANK_PROMPTS])
def test_every_bank_tells_its_interpreter_a_typed_premise_has_no_one_to_credit(
        bank, build, typed_premise):
    prompt = build(dict(typed_premise["news_article"]))
    assert OSP.NO_NAMED_SOURCE_NOTE in prompt
    assert "User Seed" not in prompt


@pytest.mark.parametrize("bank, build", BANK_PROMPTS, ids=[b for b, _ in BANK_PROMPTS])
def test_a_fetched_source_with_an_outlet_is_left_alone(bank, build):
    article = {
        "headline": "Reel found", "summary": "A lost reel.", "full_text": "A lost reel was found.",
        "source": "Library of Congress", "date": "2026-10-16", "link": "", "seed_text": "Reel found",
    }
    assert OSP.NO_NAMED_SOURCE_NOTE not in build(article)


def test_the_news_used_payload_names_no_outlet_for_a_typed_premise():
    from nodes import _otr_writer_tail as TAIL

    class _Outline:
        title = "The Thirty Year Echo"
        premise = PREMISE

    entry = json.loads(TAIL._build_news_payload(_Outline(), PREMISE, "custom_premise"))[0]
    assert entry["source"] == ""
