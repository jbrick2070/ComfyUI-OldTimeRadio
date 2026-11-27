"""Cast names in the episode's own script, on captions and credits only.

Operator 2026-11-27: "ROBIN:" and "MALVOLIO:" opened every caption turn of a
Japanese, Mandarin or Hindi episode. One fail-soft call after the cast is
final writes `cast[].native_display_name`; captions and credits paint it; the
identity `name` never changes. Reference spellings from agy (Gemini).
CPU only, no model. UTF-8 no BOM.
"""
from __future__ import annotations

import inspect
import json

import pytest

from nodes import _otr_cast_display_names as CDN


class _Slot:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        reply = self.replies.pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)


def _ledger(iso, names=("VIOLA", "MALVOLIO")):
    cast = [{"char_id": "c01", "name": "ANNOUNCER"}]
    cast += [{"char_id": "c%02d" % (i + 2), "name": n} for i, n in enumerate(names)]
    return {"meta": {"episode_language": iso}, "cast": cast}


@pytest.mark.parametrize("iso, expected", [
    ("en", False), ("es", False), ("pt", False), ("it", False), ("fr", False),
    ("hi", True), ("ja", True), ("zh", True),
])
def test_only_non_latin_script_episodes_ask(iso, expected):
    assert CDN.needs_native_names({"episode_language": iso}) is expected


@pytest.mark.parametrize("meta", [None, {}, {"episode_language": ""}, "x",
                                  {"episode_language": "tlh"}])
def test_off_legacy_and_unreadable_never_ask(meta):
    assert CDN.needs_native_names(meta) is False


def test_english_makes_no_call_and_changes_no_byte():
    led = _ledger("en")
    before = json.dumps(led, sort_keys=True)
    slot = _Slot([])
    assert CDN.populate(led, technical_fn=slot) == {}
    assert slot.calls == [] and json.dumps(led, sort_keys=True) == before


@pytest.mark.parametrize("iso, viola, malvolio", [
    ("ja", "ヴァイオラ", "マルヴォーリオ"),
    ("zh", "薇奥拉", "马伏里奥"),
    ("hi", "वायोला", "माल्वोलियो"),
])
def test_accepted_names_land_on_the_cast_row_and_the_identity_name_stays(iso, viola, malvolio):
    led = _ledger(iso)
    slot = _Slot([{"names": [{"n": 1, "display_name": viola},
                             {"n": 2, "display_name": malvolio}]}])
    receipt = CDN.populate(led, technical_fn=slot)
    rows = {r["name"]: r for r in led["cast"]}
    assert rows["VIOLA"]["native_display_name"] == viola
    assert rows["MALVOLIO"]["native_display_name"] == malvolio
    assert "native_display_name" not in rows["ANNOUNCER"]
    assert receipt["accepted"] == 2 and receipt["asked"] == 2
    assert led["meta"]["cast_display_names"] is receipt
    assert len(slot.calls) == 1
    assert "ANNOUNCER" not in slot.calls[0][0][1]["content"]


def test_each_bad_name_keeps_its_latin_spelling_alone():
    led = _ledger("zh", names=("VIOLA", "MALVOLIO", "KENT", "LEAR"))
    slot = _Slot([{"names": [
        {"n": 1, "display_name": "VIOLA"},          # echoed
        {"n": 2, "display_name": "马伏里奥"},     # good
        {"n": 3, "display_name": "Kent the loyal"},  # wrong script
        # LEAR missing
    ]}])
    receipt = CDN.populate(led, technical_fn=slot)
    rows = {r["name"]: r for r in led["cast"]}
    assert rows["MALVOLIO"]["native_display_name"] == "马伏里奥"
    for latin in ("VIOLA", "KENT", "LEAR"):
        assert "native_display_name" not in rows[latin]
    assert receipt["rejected"] == {"VIOLA": "the Latin name echoed",
                                   "KENT": "not in the episode's script",
                                   "LEAR": "missing from the reply"}


def test_two_speakers_who_would_read_alike_both_keep_latin():
    led = _ledger("ja", names=("ANTIPHOLUS OF EPHESUS", "ANTIPHOLUS OF SYRACUSE"))
    slot = _Slot([{"names": [
        {"n": 1, "display_name": "アンティフォラス"},
        {"n": 2, "display_name": "アンティフォラス"}]}])
    CDN.populate(led, technical_fn=slot)
    assert all("native_display_name" not in r for r in led["cast"])


def test_a_failed_call_keeps_every_latin_name_and_never_raises():
    led = _ledger("hi")
    slot = _Slot(["not json", "still not json"])
    receipt = CDN.populate(led, technical_fn=slot)
    assert receipt["error"]
    assert all("native_display_name" not in r for r in led["cast"])


def test_a_ledger_that_already_has_names_is_not_asked_again():
    led = _ledger("ja")
    led["cast"][1]["native_display_name"] = "ヴァイオラ"
    slot = _Slot([])
    assert CDN.populate(led, technical_fn=slot) == {}
    assert slot.calls == []


@pytest.mark.parametrize("display, reason", [
    ("", "empty"),
    ("薇奥拉：", "punctuation a name does not carry"),
    ('["薇奥拉"', "punctuation a name does not carry"),
    ("薇" * 40, "too long"),
    ("viola", "the Latin name echoed"),
    ("Viola", "the Latin name echoed"),
])
def test_judge_names_the_reason(display, reason):
    assert CDN.judge_display_name("VIOLA", display) == reason


@pytest.mark.parametrize("display", ["第一女巫", "第一の魔女", "पहली डायन",
                                     "エフェソスのアンティフォラス", "レジナルド・シリキット"])
def test_role_words_and_compound_names_are_accepted(display):
    assert CDN.judge_display_name("FIRST WITCH", display) is None


def test_display_name_prefers_native_then_identity_then_legacy():
    assert CDN.display_name({"name": "VIOLA", "native_display_name": "薇奥拉"}) == "薇奥拉"
    assert CDN.display_name({"name": "VIOLA"}) == "VIOLA"
    assert CDN.display_name({"name": "", "display_name": "Viola"}) == "Viola"
    assert CDN.display_name(None) == ""


def test_set_cast_keeps_the_field_and_adds_nothing_when_absent():
    from nodes import production_ledger as PL
    led = PL.Ledger.__new__(PL.Ledger)
    led.data = {"cast": []}
    PL.Ledger.set_cast(led, [{"char_id": "c02", "name": "VIOLA",
                              "native_display_name": "薇奥拉"},
                             {"char_id": "c03", "name": "KENT"}])
    rows = {r["name"]: r for r in led.data["cast"]}
    assert rows["VIOLA"]["native_display_name"] == "薇奥拉"
    assert "native_display_name" not in rows["KENT"]


def test_captions_paint_the_native_name_and_the_identity_name_is_untouched():
    from nodes import _otr_captions as C
    src = inspect.getsource(C)
    assert "nm = _OTRCDN.display_name(cast_row)" in src


def test_the_writer_tail_asks_once_after_reconcile_and_before_coverage_repair():
    from nodes import _otr_writer_tail as WT
    src = inspect.getsource(WT)
    call = src.index("_OTRCDN.populate(led.data, technical_fn=technical_generate_fn)")
    assert src.index("_clean_window.reconcile()") < call
    assert call < src.index("_OTRCCR.repair_zero_coverage_cast(")
    assert 'helper_context("cast_display_names")' in src


def test_credits_cast_and_transcript_read_the_native_name():
    from nodes import otr_credits_roll as CR
    src = inspect.getsource(CR)
    assert 'name = "%s (%s)" % (native, name)' in src
    assert 'spk = str(cast_row.get("native_display_name") or spk)' in src


# Sonnet QA on 003c3b84f: "any non-Latin letter" let a Latin gloss, Cyrillic,
# Hangul and a caption line break through.
@pytest.mark.parametrize("display, iso", [
    ("Viola (ヴァイオラ)", "ja"),
    ("ヴァイオラ VIOLA", "ja"),
    ("Violа", "ja"),                 # Cyrillic homoglyph
    ("비올라", "ja"),         # Hangul on a Japanese episode
    ("ヴァ" + chr(92) + "Nイ", "ja"),      # an ASS line break
    ("ヴァイオラ", "zh"),                 # kana on a Mandarin episode
    ("薇奥拉", "hi"),
])
def test_a_name_outside_the_episodes_script_is_refused(display, iso):
    assert CDN.judge_display_name("VIOLA", display, iso) is not None


@pytest.mark.parametrize("display, iso", [
    ("ヴァイオラ", "ja"), ("リア・王", "ja"), ("マルヴォーリオー", "ja"),
    ("第一の魔女", "ja"), ("薇奥拉", "zh"), ("第一女巫", "zh"),
    ("वायोला", "hi"), ("पहली डायन", "hi"),
])
def test_a_name_in_the_episodes_script_is_accepted(display, iso):
    assert CDN.judge_display_name("VIOLA", display, iso) is None


def test_populate_judges_against_the_episodes_own_script():
    led = _ledger("zh")
    slot = _Slot([{"names": [{"n": 1, "display_name": "ヴァイオラ"},
                             {"n": 2, "display_name": "马伏里奥"}]}])
    CDN.populate(led, technical_fn=slot)
    rows = {r["name"]: r for r in led["cast"]}
    assert "native_display_name" not in rows["VIOLA"]
    assert rows["MALVOLIO"]["native_display_name"] == "马伏里奥"


# Sonnet QA on 2718c6d55: ordinary surnames were refused, junk was accepted.
@pytest.mark.parametrize("display, iso", [
    ("佐々木", "ja"), ("﨑山", "ja"), ("𠮷田", "ja"), ("ﾘｱ", "ja"),
    ("兵士2", "ja"), ("क़ासिम", "hi"), ("किरण‍", "hi"),
])
def test_ordinary_names_with_marks_and_variants_are_accepted(display, iso):
    assert CDN.judge_display_name("X", display, iso) is None


@pytest.mark.parametrize("display, iso", [
    ("ア☺", "ja"), ("ア́", "ja"), ("राम।", "hi"),
])
def test_symbols_dandas_and_stray_accents_are_refused(display, iso):
    assert CDN.judge_display_name("X", display, iso) is not None


def test_the_reply_has_room_for_a_fenced_pretty_printed_answer():
    """4060, 2026-11-27: three names, 96 tokens, the reply cut off mid-name on
    both attempts. The budget now leaves room for the fence and indentation."""
    assert CDN._output_budget(3) >= 300
    assert CDN._output_budget(1) >= 192
    led = _ledger("zh", names=("TRUMAN MING", "MIRA BERNARD", "YURI KENDALL"))
    slot = _Slot([{"names": [{"n": 1, "display_name": "杜鲁门·明"},
                             {"n": 2, "display_name": "米拉·伯纳德"},
                             {"n": 3, "display_name": "尤里·肯德尔"}]}])
    CDN.populate(led, technical_fn=slot)
    assert slot.calls[0][1]["max_new_tokens"] == CDN._output_budget(3)


def test_rows_are_matched_by_number_whatever_order_or_key_they_carry():
    """Mac, 2026-11-27: Qwen 3.5 4B translated the "name" key; Sonnet QA on
    5b6dcedb0: pairing by position mislabels a reordered reply. The number
    the name was asked with is the only key."""
    led = _ledger("ja", names=("ORLANDO", "CORIN", "TOUCHSTONE"))
    slot = _Slot([{"names": [{"n": 2, "display_name": "コリン"},
                             {"n": 3, "display_name": "タッチストーン"},
                             {"n": 1, "display_name": "オーランドー"}]}])
    CDN.populate(led, technical_fn=slot)
    rows = {r["name"]: r.get("native_display_name") for r in led["cast"]}
    assert rows == {"ANNOUNCER": None, "ORLANDO": "オーランドー",
                    "CORIN": "コリン", "TOUCHSTONE": "タッチストーン"}
    assert "1. ORLANDO" in slot.calls[0][0][1]["content"]


def test_an_invented_row_and_a_repeated_number_match_nothing_extra():
    led = _ledger("ja", names=("ORLANDO", "CORIN"))
    slot = _Slot([{"names": [{"n": 0, "display_name": "アナウンサー"},
                             {"n": 9, "display_name": "タッチストーン"},
                             {"n": 1, "display_name": "オーランドー"},
                             {"n": 1, "display_name": "コリン"}]}])
    receipt = CDN.populate(led, technical_fn=slot)
    rows = {r["name"]: r.get("native_display_name") for r in led["cast"]}
    assert rows["ORLANDO"] == "オーランドー" and rows["CORIN"] is None
    assert receipt["rejected"] == {"CORIN": "missing from the reply"}