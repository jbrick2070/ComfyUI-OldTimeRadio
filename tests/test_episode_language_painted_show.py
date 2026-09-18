"""The painted show speaks the episode's language: spoken chrome, captions, credits.

Covers the downstream half of the multilingual one-switch (build-contract items
3 and 8): the Python-authored announcer strings in `_otr_line_composer`, the
reserved announcer label burned into SDH by `_otr_captions`, and the audience
chrome printed by `otr_credits_roll`.

THE ENGLISH REGRESSION GATE is asserted directly here: every surface below must
render character-for-character what it rendered before this feature, both for an
unstamped ledger (Off / legacy) and for an explicitly English one.
"""
from __future__ import annotations

import pytest

from nodes import _otr_captions as CAP
from nodes import _otr_episode_languages as el
from nodes import _otr_line_composer as LC
from nodes import otr_credits_roll as CR

NON_ENGLISH = ["Spanish", "Portuguese", "Italian", "French",
               "Hindi", "Japanese", "Mandarin"]


def _meta(label=None):
    meta = {"source_bank": "original"}
    if label is not None:
        meta.update(el.resolve_ledger(label))
    return meta


UNSTAMPED = [None, {}, _meta(), _meta("English")]


class _Brief:
    """The three fields `fallback_safe_open` reads, by direct attribute."""

    def __init__(self, setting="a cold pier", time_of_day="after midnight",
                 work_title=""):
        self.setting = setting
        self.time_of_day = time_of_day
        self.work_title = work_title


# --------------------------------------------------------------------------- #
# English is byte-identical -- the regression gate
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("meta", UNSTAMPED)
def test_the_english_sign_on_is_unchanged(meta):
    assert LC.fallback_announcer_intro("", episode_meta=meta) == (
        "Good evening. This is SIGNAL LOST.")
    assert LC.fallback_announcer_intro("a signal returns", episode_meta=meta) == (
        "Good evening. This is SIGNAL LOST. Tonight: a signal returns")


@pytest.mark.parametrize("meta", UNSTAMPED)
def test_the_english_sign_off_is_unchanged(meta):
    assert LC.fallback_announcer_outro("", episode_meta=meta) == (
        "This has been SIGNAL LOST. Good night.")
    assert LC.fallback_announcer_outro("from the archive", episode_meta=meta) == (
        "This has been SIGNAL LOST. from the archive Good night.")


@pytest.mark.parametrize("meta", UNSTAMPED)
def test_the_english_safe_open_is_unchanged(meta):
    assert LC.fallback_safe_open(_Brief(), episode_meta=meta) == (
        "Good evening. This is SIGNAL LOST. "
        "We open on after midnight, a cold pier.")
    assert LC.fallback_safe_open(
        _Brief(work_title="Macbeth"), episode_meta=meta) == (
        "Good evening. This is SIGNAL LOST. Tonight, a scene from Macbeth. "
        "We open on after midnight, a cold pier.")
    assert LC.fallback_safe_open(
        _Brief(setting="", time_of_day=""), episode_meta=meta) == (
        "Good evening. This is SIGNAL LOST.")


@pytest.mark.parametrize("meta", UNSTAMPED)
def test_the_english_work_frame_sentence_matches_the_public_template(meta):
    assert LC.work_frame_sentence("Macbeth", episode_meta=meta) == (
        LC.WORK_FRAME_SENTENCE.format(frame="Macbeth"))


def test_a_legacy_caller_passing_no_meta_at_all_still_speaks_english():
    """Every pre-existing call site and self-test goes through this path."""
    assert LC.fallback_announcer_intro("") == "Good evening. This is SIGNAL LOST."
    assert LC.fallback_announcer_outro("") == "This has been SIGNAL LOST. Good night."
    assert LC.fallback_safe_open(_Brief(setting="", time_of_day="")) == (
        "Good evening. This is SIGNAL LOST.")
    assert LC.splice_work_frame("Tonight, nonsense. Then the rain.", "Macbeth") == (
        "Tonight, a scene from Macbeth. Then the rain.")


# --------------------------------------------------------------------------- #
# every non-English row speaks its own words
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("label", NON_ENGLISH)
def test_the_sign_on_uses_the_rows_own_greeting_and_station_id(label):
    row = el.row_by_label(label)
    line = LC.fallback_announcer_intro("", episode_meta=_meta(label))
    assert line.startswith(row.spoken["sign_on_greeting"])
    assert row.spoken["station_id_open"] in line
    assert "Good evening" not in line


@pytest.mark.parametrize("label", NON_ENGLISH)
def test_the_sign_off_uses_the_rows_own_words(label):
    row = el.row_by_label(label)
    line = LC.fallback_announcer_outro("", episode_meta=_meta(label))
    assert row.spoken["station_id_close"] in line
    assert row.spoken["sign_off_greeting"] in line
    assert "Good night" not in line


@pytest.mark.parametrize("label", NON_ENGLISH)
def test_the_safe_open_uses_the_rows_own_open_on_and_work_prefix(label):
    row = el.row_by_label(label)
    line = LC.fallback_safe_open(_Brief(work_title="Macbeth"),
                                 episode_meta=_meta(label))
    assert row.spoken["open_on_prefix"] in line
    assert row.spoken["work_line_prefix"] in line
    assert row.spoken["tonight_label"] in line
    # The WORK's own title is never rendered -- only the phrasing around it.
    assert "Macbeth" in line
    assert "We open on" not in line and "a scene from" not in line


@pytest.mark.parametrize("label", ["English"] + NON_ENGLISH)
def test_signal_lost_survives_on_every_row(label):
    """The call sign is not a phrase to be rendered. Grammar bends around it."""
    meta = _meta(label)
    for line in (LC.fallback_announcer_intro("", episode_meta=meta),
                 LC.fallback_announcer_outro("", episode_meta=meta),
                 LC.fallback_safe_open(_Brief(), episode_meta=meta)):
        assert "SIGNAL LOST" in line, (label, line)


def test_spanish_reads_as_the_fable_seed_authored_it():
    meta = _meta("Spanish")
    assert LC.fallback_announcer_intro("", episode_meta=meta) == (
        "Buenas noches. Esta es SIGNAL LOST.")
    assert LC.fallback_announcer_outro("", episode_meta=meta) == (
        "Esto ha sido SIGNAL LOST. Buenas noches.")
    assert LC.work_frame_sentence("Macbeth", episode_meta=meta) == (
        "Esta noche, una escena de Macbeth.")


def test_the_work_frame_splice_keeps_the_native_sentence():
    spliced = LC.splice_work_frame(
        "Tonight, nonsense. Y luego la lluvia.", "Macbeth",
        episode_meta=_meta("Spanish"))
    assert spliced == "Esta noche, una escena de Macbeth. Y luego la lluvia."


def test_an_unreadable_row_degrades_to_english_instead_of_silencing_the_open():
    """A spoken line is never the thing that fails. A label is not worth an
    episode."""
    assert LC.spoken_chrome({"episode_language": "tlh"})["sign_on_greeting"] == (
        "Good evening")


# --------------------------------------------------------------------------- #
# the announcer prompt asks for the line in the episode's language
# --------------------------------------------------------------------------- #


def test_the_english_announcer_system_prompt_is_untouched():
    for meta in UNSTAMPED:
        assert LC._announcer_system("SEAM TEXT", meta) == "SEAM TEXT"


@pytest.mark.parametrize("label", NON_ENGLISH)
def test_the_native_instruction_leads_the_announcer_seam(label):
    row = el.row_by_label(label)
    system = LC._announcer_system("SEAM TEXT", _meta(label))
    assert system.startswith(row.authoring["writer_instruction"])
    assert system.endswith("SEAM TEXT")


# --------------------------------------------------------------------------- #
# captions -- the reserved announcer label
# --------------------------------------------------------------------------- #


def test_the_english_sdh_still_burns_announcer():
    for label in (None, "English"):
        ledger = {"meta": _meta(label)}
        assert CAP._reserved_announcer_label(ledger, "ANNOUNCER") == "ANNOUNCER"


@pytest.mark.parametrize("label,expected", [
    ("Spanish", "LOCUTOR"),
    ("Portuguese", "LOCUTOR"),
    ("Italian", "ANNUNCIATORE"),
    ("French", "ANNONCEUR"),
    ("Hindi", "उद्घोषक"),
    ("Japanese", "アナウンサー"),
    ("Mandarin", "播音员"),
])
def test_the_reserved_announcer_label_is_native_per_row(label, expected):
    ledger = {"meta": _meta(label)}
    assert CAP._reserved_announcer_label(ledger, "ANNOUNCER") == expected


def test_an_unreadable_ledger_burns_the_label_it_was_given():
    """Captions are a deliverable; a missing label never ships no subtitles."""
    ledger = {"meta": {"episode_language": "tlh"}}
    assert CAP._reserved_announcer_label(ledger, "ANNOUNCER") == "ANNOUNCER"
    assert CAP._reserved_announcer_label(None, "ANNOUNCER") == "ANNOUNCER"


def test_the_caption_painter_actually_substitutes_the_label():
    """A test that calls the helper proves the helper, never the wiring."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "nodes"
           / "_otr_captions.py").read_text(encoding="utf-8")
    assert 'nm = _reserved_announcer_label(ledger, nm)' in src
    # The structural key is NOT renamed -- it stays a display substitution.
    assert 'if nm.upper() == "ANNOUNCER":' in src


@pytest.mark.parametrize("label", ["English"] + NON_ENGLISH)
def test_the_reserved_announcer_name_is_never_a_gendered_role(label):
    """Station role, not a person (Fable: LOCUTOR, never LOCUTORA)."""
    name = el.row_by_label(label).spoken["reserved_announcer_name"]
    assert name == name.strip() and name
    assert "LOCUTORA" not in name


# --------------------------------------------------------------------------- #
# credits -- audience chrome native, machine receipts English
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("meta", UNSTAMPED)
def test_the_english_credits_chrome_is_todays_headers(meta):
    chrome = CR._credits_chrome(meta)
    assert chrome["models_header"] == "MODELS"
    assert chrome["production_ledger_header"] == "[ PRODUCTION LEDGER ]"
    assert chrome["cast_voices_header"] == "CAST & VOICES"
    assert chrome["story_spine_header"] == "[ STORY SPINE ]"
    assert chrome["premise_label"] == "Premise:"
    assert chrome["subject_label"] == "Subject:"
    assert chrome["system_header"] == "[ SYSTEM ]"
    assert chrome["classified_transcript_header"] == "[ CLASSIFIED TRANSCRIPT ]"


@pytest.mark.parametrize("label", NON_ENGLISH)
def test_the_credits_chrome_is_native_per_row(label):
    row = el.row_by_label(label)
    assert CR._credits_chrome(_meta(label)) == dict(row.credits)


def test_an_unreadable_row_prints_english_headers_rather_than_no_card():
    """`_require` still fails loud for a missing RECEIPT. A missing header is a
    missing translation, which is not worth withholding the episode."""
    chrome = CR._credits_chrome({"episode_language": "tlh"})
    assert chrome["models_header"] == "MODELS"


def test_the_abridged_mark_keeps_the_rows_own_ledger_header():
    english = {"chrome": CR._credits_chrome(_meta("English"))}
    assert CR._abridged_header(english) == "[ PRODUCTION LEDGER -- ABRIDGED ]"
    spanish = {"chrome": CR._credits_chrome(_meta("Spanish"))}
    assert CR._abridged_header(spanish) == "[ LIBRO DE PRODUCCIÓN -- ABRIDGED ]"
    # A layout with no chrome at all (a legacy dict, or a self-test's) keeps
    # the constant it always used.
    assert CR._abridged_header({}) == CR._ABRIDGED_HEADER


def test_the_credits_layout_carries_the_chrome_and_the_language_header():
    """The drawers receive a LAYOUT, not a ledger -- a second reader would be a
    second answer waiting to disagree."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "nodes"
           / "otr_credits_roll.py").read_text(encoding="utf-8")
    assert 'chrome = _credits_chrome(meta)' in src
    assert '"chrome": dict(chrome),' in src
    assert '"language_header": str(meta.get("language_header") or ""),' in src
    # No audience header is left as a literal in the layout builder.
    for gone in ('"header": "MODELS"', '"header": "[ STORY SPINE ]"',
                 '"header": "[ SYSTEM ]"', '("Premise:",', '("Subject:",'):
        assert gone not in src, gone


@pytest.mark.parametrize("label", ["English"] + NON_ENGLISH)
def test_machine_serials_are_never_row_values(label):
    """VRAM, CUDA, SEED, REV, model ids: the LABEL may be native, the VALUE is a
    serial number, and a translated serial is a wrong serial."""
    row = el.row_by_label(label)
    for serial in ("VRAM", "CUDA", "SEED", "REV", "CPU", "RAM", "GPU"):
        assert serial not in row.credits, serial
        assert serial not in row.spoken, serial
