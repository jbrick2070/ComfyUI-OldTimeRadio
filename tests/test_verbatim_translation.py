"""The verbatim passage is translated on a non-English row -- texts only.

Operator ruling 2026-09-18. Speakers and the cut are the plan's; the model
returns texts in order; validation is structural; failure is loud; English
never makes the call. CPU only, no model, no network. UTF-8 no BOM.
"""
from __future__ import annotations

import inspect
import json

import pytest

from nodes import _otr_episode_languages as EL
from nodes import _otr_passage_selector as PS
from nodes import _otr_provenance as PROV
from nodes import _otr_story_input as SI
from nodes import _otr_verbatim_lane as VL
from nodes import _otr_verbatim_translation as VT

RULE = "Escribe el episodio en español."


def _entries(*texts, speakers=None):
    speakers = speakers or ["FIRST WITCH", "SECOND WITCH", "THIRD WITCH"]
    return tuple(
        PS.BeatPlanEntry(speaker=speakers[i % len(speakers)], text=t,
                         speech_index=i, chunk_ordinal=0, chunk_count=1)
        for i, t in enumerate(texts)
    )


class _Slot:
    """Answers each call from a queue; records every prompt it saw."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        reply = self.replies.pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)


def test_batches_stay_under_the_source_word_budget_and_never_empty():
    entries = _entries("one two three", "four five", "six seven eight nine",
                       "ten")
    assert VT.batch_entries(entries, max_source_words=5) == [[0, 1], [2, 3]]
    assert VT.batch_entries(entries, max_source_words=1) == [[0], [1], [2], [3]]
    assert VT.batch_entries(entries, max_source_words=100) == [[0, 1, 2, 3]]
    assert VT.batch_entries((), max_source_words=5) == []


def test_texts_are_replaced_in_order_and_everything_else_is_copied():
    entries = _entries("When shall we three meet again?",
                       "When the hurlyburly's done.",
                       "That will be ere the set of sun.")
    slot = _Slot([{"texts": ["¿Cuándo volveremos a vernos?",
                             "Cuando acabe el tumulto.",
                             "Será antes de ponerse el sol."]}])
    out, receipt = VT.translate_entries(
        entries, language_instruction=RULE, creative_fn=slot,
        max_source_words=100)
    assert [e.text for e in out] == [
        "¿Cuándo volveremos a vernos?", "Cuando acabe el tumulto.",
        "Será antes de ponerse el sol."]
    assert [e.speaker for e in out] == [e.speaker for e in entries]
    assert [(e.speech_index, e.chunk_ordinal, e.chunk_count) for e in out] == \
        [(e.speech_index, e.chunk_ordinal, e.chunk_count) for e in entries]
    assert receipt["entries"] == 3 and receipt["batches"] == 1
    assert receipt["source_sha256"] != receipt["text_sha256"]
    messages, kwargs = slot.calls[0]
    assert messages[0]["content"].startswith(RULE + "\n\n")
    assert "1. FIRST WITCH: When shall we three meet again?" in messages[1]["content"]
    assert "3. THIRD WITCH: That will be ere the set of sun." in messages[1]["content"]


def test_a_long_passage_is_translated_in_consecutive_batches():
    entries = _entries("a b c d e", "f g h i j", "k l m n o")
    slot = _Slot([{"texts": ["uno", "dos"]}, {"texts": ["tres"]}])
    out, receipt = VT.translate_entries(
        entries, language_instruction=RULE, creative_fn=slot,
        max_source_words=10)
    assert [e.text for e in out] == ["uno", "dos", "tres"]
    assert receipt["batches"] == 2 and len(slot.calls) == 2
    assert "2 lines" in slot.calls[0][0][1]["content"]
    assert "1 lines" in slot.calls[1][0][1]["content"]


def test_output_budget_is_explicit_and_sized_to_the_batch():
    entries = _entries("a b c d e f g h i j", "k")
    slot = _Slot([{"texts": ["x", "y"]}])
    VT.translate_entries(entries, language_instruction=RULE, creative_fn=slot)
    kwargs = slot.calls[0][1]
    assert kwargs.get("max_new_tokens") == max(
        VT._OUTPUT_TOKENS_FLOOR, 11 * VT._OUTPUT_TOKENS_PER_SOURCE_WORD)


def test_a_wrong_count_is_retried_once_then_fails_loud():
    entries = _entries("one", "two")
    slot = _Slot([{"texts": ["uno"]}, {"texts": ["uno", "dos", "tres"]}])
    with pytest.raises(RuntimeError) as caught:
        VT.translate_entries(entries, language_instruction=RULE, creative_fn=slot)
    assert "could not be translated" in str(caught.value)
    assert len(slot.calls) == 2


def test_an_empty_line_is_a_structural_failure():
    entries = _entries("one", "two")
    slot = _Slot([{"texts": ["uno", "   "]}, {"texts": ["uno", "dos"]}])
    out, receipt = VT.translate_entries(
        entries, language_instruction=RULE, creative_fn=slot)
    assert [e.text for e in out] == ["uno", "dos"]
    assert len(slot.calls) == 2


def test_no_instruction_is_a_caller_bug_not_a_silent_pass_through():
    with pytest.raises(ValueError):
        VT.translate_entries(_entries("one"), language_instruction="",
                             creative_fn=_Slot([]))


def test_translate_plan_keeps_the_receipt_and_rerenders_the_passage():
    entries = _entries("Fair is foul.", "Hover through the fog.")
    plan = VL.VerbatimPlan(entries=entries, speakers=("FIRST WITCH", "SECOND WITCH"),
                           passage_text="FIRST WITCH\nFair is foul.\n\nSECOND WITCH\nHover through the fog.",
                           seed="1|x", receipt={"status": "planned"})
    slot = _Slot([{"texts": ["Lo bello es feo.", "Flotemos entre la niebla."]}])
    out, receipt = VT.translate_plan(
        plan, language_instruction=RULE, creative_fn=slot, iso="es",
        model_id="test-creative")
    assert out.speakers == plan.speakers and out.seed == plan.seed
    assert out.receipt is plan.receipt
    assert out.passage_text == (
        "FIRST WITCH\nLo bello es feo.\n\nSECOND WITCH\nFlotemos entre la niebla.")
    assert receipt["iso"] == "es" and receipt["model_id"] == "test-creative"
    assert receipt["receipt_version"] == VT.RECEIPT_VERSION


# --------------------------------------------------------------------------- #
# the writer: translated before the outline reads verbatim_texts, never on English
# --------------------------------------------------------------------------- #


def test_the_writer_translates_under_the_empty_instruction_guard_before_the_outline():
    from nodes import OTR_LedgerScriptWriter as W
    src = inspect.getsource(W.OTR_LedgerScriptWriter.run)
    call = src.index("_OTRVT.translate_plan(")
    guard = src.rindex("if _vt_instruction:", 0, call)
    assert src.rindex("_EPLANG.native_authoring_instruction(meta)", 0, guard) > 0
    assert call < src.index("verbatim_texts=")
    assert call > src.index('_verbatim_plan = resolved.get("verbatim_plan")')
    assert 'meta.setdefault("verbatim_passage", {})["translation"]' in src


def test_english_and_off_carry_no_instruction_so_no_translation_runs():
    assert EL.native_authoring_instruction({"episode_language": "en"}) == ""
    assert EL.native_authoring_instruction({}) == ""
    assert EL.native_authoring_instruction({"episode_language": "es"}).strip()


# --------------------------------------------------------------------------- #
# the spoken credit sentences follow the row
# --------------------------------------------------------------------------- #


class _Identity:
    def __init__(self, work_title, author):
        self.work_title = work_title
        self.author = author


def test_the_licensed_coda_speaks_the_row_and_english_is_byte_identical():
    prov = {"status": "licensed_noncommercial"}
    who = _Identity("Macbeth", "William Shakespeare")
    english = PROV.spoken_coda_line(prov, who)
    assert english == "Tonight's scene was drawn from William Shakespeare's Macbeth."
    assert PROV.spoken_coda_line(prov, who, episode_meta={"episode_language": "en"}) == english
    assert PROV.spoken_coda_line(prov, who, episode_meta={}) == english
    spanish = PROV.spoken_coda_line(prov, who, episode_meta={"episode_language": "es"})
    assert spanish == "La escena de esta noche fue tomada de Macbeth, de William Shakespeare."


@pytest.mark.parametrize("status, key", [
    ("public_domain_us", "coda_public_domain_us"),
    ("cc0", "coda_cc0"),
    ("research_only", "coda_research_only"),
    ("synthetic", "coda_synthetic"),
])
def test_the_generic_coda_follows_the_row(status, key):
    row = EL.row_by_label("French")
    assert PROV.spoken_coda_line({"status": status}, episode_meta={"episode_language": "fr"}) \
        == row.spoken[key]
    assert PROV.spoken_coda_line({"status": status}) == PROV._CODA_BY_STATUS[status]


def test_the_named_coda_follows_the_row():
    who = _Identity("Gertrude the Governess", "Stephen Leacock")
    out = PROV.spoken_coda_line({"status": "public_domain_us"}, who,
                                episode_meta={"episode_language": "ja"})
    assert out == "今夜の物語は、Stephen Leacockの『Gertrude the Governess』を脚色したものです。"


def test_an_unreadable_row_degrades_the_coda_to_english():
    prov = {"status": "public_domain_us"}
    assert PROV.spoken_coda_line(prov, episode_meta={"episode_language": "tlh"}) \
        == PROV._CODA_BY_STATUS["public_domain_us"]


def test_the_attribution_sentence_follows_the_row_and_english_is_byte_identical():
    assert SI.attribution_sentence("Ada Byron") == "Tonight's story is by Ada Byron."
    assert SI.attribution_sentence("Ada Byron", episode_meta={"episode_language": "en"}) \
        == "Tonight's story is by Ada Byron."
    assert SI.attribution_sentence("") == SI.ANONYMOUS_ATTRIBUTION
    assert SI.attribution_sentence("Ada Byron", episode_meta={"episode_language": "es"}) \
        == "La historia de esta noche es de Ada Byron."
    assert SI.attribution_sentence("", episode_meta={"episode_language": "es"}) \
        == "La historia de esta noche viene de uno de nuestros oyentes."
    assert SI.attribution_sentence("X", episode_meta={"episode_language": "tlh"}) \
        == "Tonight's story is by X."
    receipt = SI.attribution_receipt("Ada Byron", episode_meta={"episode_language": "es"})
    assert receipt["sentence"] == "La historia de esta noche es de Ada Byron."


def test_every_row_formats_every_credit_template_without_error():
    who = _Identity("Work", "Author")
    for row in EL.load_registry()[0]:
        meta = {"episode_language": row.iso}
        for status in ("public_domain_us", "cc0", "research_only", "synthetic"):
            assert PROV.spoken_coda_line({"status": status}, episode_meta=meta).strip()
        for status in ("public_domain_us", "cc0", "research_only", "licensed_noncommercial"):
            line = PROV.spoken_coda_line({"status": status}, who, episode_meta=meta)
            assert "Work" in line and "Author" in line, (row.iso, status)
        assert "Ada" in SI.attribution_sentence("Ada", episode_meta=meta)
        assert SI.attribution_sentence("", episode_meta=meta).strip()


# --------------------------------------------------------------------------- #
# the model echoes the speaker label back (live proof 2026-09-18)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("returned, speaker, want", [
    # The live defect: the label was TRANSLATED, so it matches no plan string.
    ("ANTÍFON DE EFESO: Ve, vete. Traedme un pico de hierro.",
     "ANTIPHOLUS OF EPHESUS", "Ve, vete. Traedme un pico de hierro."),
    ("BALTHASAR: Ten paciencia, señor.", "BALTHASAR", "Ten paciencia, señor."),
    ("Balthasar: Ten paciencia.", "BALTHASAR", "Ten paciencia."),
    ("船長：老大！", "BOATSWAIN", "老大！"),
    # Label only -> empty, so the validator retries. Returning the raw label
    # here is what shipped "ANA:" as a spoken row (post-QA, 2026-09-18).
    ("ANA: ", "ANA", ""),
])
def test_an_echoed_label_is_stripped(returned, speaker, want):
    assert VT.strip_echoed_label(returned, speaker) == want


@pytest.mark.parametrize("text", [
    "Escucha: no hay nadie aquí.",            # ordinary prose with a colon
    "Te lo digo así: vete.",
    "No hay ningún dos puntos aquí.",
    "Mi señor, atended: la hora llega.",
])
def test_ordinary_prose_with_a_colon_is_untouched(text):
    assert VT.strip_echoed_label(text, "ANA") == text


def test_the_strip_runs_inside_the_translation_and_reaches_the_entries():
    entries = _entries("Go, get thee gone.", "Have patience, sir.")
    slot = _Slot([{"texts": ["ANTÍFON: Ve, vete.", "BALTASAR: Ten paciencia."]}])
    out, _ = VT.translate_entries(entries, language_instruction=RULE,
                                  creative_fn=slot, max_source_words=100)
    assert [e.text for e in out] == ["Ve, vete.", "Ten paciencia."]


def test_the_prompt_forbids_the_label_in_words_too():
    assert "never repeat the speaker's name" in VT._SYSTEM


def test_a_label_only_reply_is_empty_so_the_validator_retries():
    """It used to ship "ANA:" as the whole spoken row: the structural check
    ran on the RAW reply, found it non-empty, and the strip happened after."""
    assert VT.strip_echoed_label("ANA: ", "ANA") == ""
    entries = _entries("Speak now.")
    slot = _Slot([{"texts": ["ANA: "]}, {"texts": ["Habla ahora."]}])
    out, _ = VT.translate_entries(entries, language_instruction=RULE,
                                  creative_fn=slot)
    assert [e.text for e in out] == ["Habla ahora."]
    assert len(slot.calls) == 2, "the label-only reply must be retried"


def test_a_salutation_the_source_itself_carries_is_never_stripped():
    """Only a prefix the MODEL added goes; if the English line opens with a
    label-shaped salutation, the translation is entitled to one too."""
    assert VT.strip_echoed_label("MI SEÑOR: no lo olvides", "ANA",
                                 "MY LORD: forget it not") == \
        "MI SEÑOR: no lo olvides"
    assert VT.strip_echoed_label("ANA: no lo olvides", "ANA",
                                 "forget it not") == "no lo olvides"
