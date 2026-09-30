"""The pro lane's closing factual read now proves its own attribution.

`_pass_news_read` shipped with NO `post_validator` at all, while its codex
twin (P6) has verified and cleaned its coda since it was built. So on this
lane the one line whose entire job is to tell a listener where the fact
stopped and the fiction started was never checked for naming a source, nor
for smuggling an invented character into a factual report.

Both findings here are PROVENANCE. Length, register, sentence count and craft
are not inspected -- an audit may never fail a story for those (THE LAW,
2026-07-22), and nothing below does.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_scifi_news_pro as F2  # noqa: E402


def _dossier(*, people=(), places=(), things=(), numbers=()):
    return F2.DossierLLM(
        facts_to_keep=["The detector logged a neutrino burst."],
        allowed_numbers=list(numbers),
        named_entities=F2.NamedEntities(
            people=list(people), places=list(places), things=list(things),
        ),
        dramatizable_vectors=[],
    )


def _read(text: str) -> F2.NewsCloseRead:
    return F2.NewsCloseRead(news_close_read=text)


def test_a_read_that_names_an_indexed_entity_passes():
    check = F2._make_news_read_validator(
        _dossier(things=["Double Chooz"]), ["MARA VELL"],
    )
    assert check(_read(
        "The Double Chooz detector really did record the burst."
    )) is None


def test_a_read_that_names_an_allowed_number_passes():
    check = F2._make_news_read_validator(
        _dossier(numbers=["17.2"]), ["MARA VELL"],
    )
    assert check(_read("Researchers measured 17.2 over the run.")) is None


def test_a_read_that_names_no_source_at_all_is_reported():
    check = F2._make_news_read_validator(
        _dossier(places=["Chooz"], numbers=["17.2"]), [],
    )
    finding = check(_read("Scientists continue to study the phenomenon."))
    assert finding is not None
    assert "never names the real source" in finding


def test_an_invented_character_in_a_factual_read_is_reported():
    check = F2._make_news_read_validator(
        _dossier(things=["Double Chooz"]), ["MARA VELL", "TOBIAS"],
    )
    finding = check(_read(
        "The Double Chooz detector logged the burst, as MARA VELL reported."
    ))
    assert finding is not None
    assert "names invented characters" in finding
    assert "MARA VELL" in finding
    assert "TOBIAS" not in finding, "only the names actually spoken are named"


def test_both_findings_arrive_together_so_one_retry_can_fix_both():
    check = F2._make_news_read_validator(
        _dossier(things=["Double Chooz"]), ["MARA VELL"],
    )
    finding = check(_read("MARA VELL said the work continues."))
    assert finding is not None
    assert "never names the real source" in finding
    assert "names invented characters" in finding


# --------------------------------------------------------------------------- #
# The source's names as the episode's own script writes them (PBUG-20260929-06)
# --------------------------------------------------------------------------- #
def test_a_hindi_close_naming_the_source_in_devanagari_passes():
    """The 2026-09-29 soak, twice: "एमआईटी न्यूज के अनुसार, एमआईटी के इंजीनियरों
    ने..." names MIT News and MIT and was refused, because the check knew only
    the Latin spellings. With the spellings this language writes, it passes."""
    dossier = _dossier(people=["Markus Buehler"], places=["MIT"])
    close = _read("एमआईटी न्यूज के अनुसार, एमआईटी के इंजीनियरों ने एक एआई मॉडल विकसित किया है।")
    without = F2._make_news_read_validator(
        dossier, [], provenance={"link": "https://news.mit.edu/2026/x"}, language_iso="hi")
    assert without(close) is not None
    with_names = F2._make_news_read_validator(
        dossier, [], provenance={"link": "https://news.mit.edu/2026/x"}, language_iso="hi",
        native_names={"MIT": ["एमआईटी"], "Markus Buehler": ["मार्कस बुहलर"]})
    assert with_names(close) is None


@pytest.mark.parametrize("close,names_it", [
    ("एमआईटी के इंजीनियरों ने कहा।", True),      # the term, then a space
    ("(एमआईटी) ने कहा।", True),                  # inside punctuation
    ("एमआईटीयन ने कहा।", False),                 # a longer Devanagari word
    ("प्रीएमआईटी ने कहा।", False),               # a prefix joins it
])
def test_a_devanagari_spelling_matches_as_a_whole_word(close, names_it):
    """Composer QA on c3dbaa6c: Devanagari is a spaced script, so a spelling
    must match as a whole word -- vowel signs are marks, not letters, and
    must neither break a term nor let it match inside a longer word."""
    check = F2._make_news_read_validator(
        _dossier(places=["MIT"]), [], provenance={"link": "https://news.mit.edu/2026/x"},
        language_iso="hi", native_names={"MIT": ["एमआईटी"]})
    assert (check(_read(close)) is None) is names_it


def test_a_real_person_written_in_katakana_is_not_refused_as_fiction():
    """2026-09-29: the cast borrowed the real Jonathan Bow as ジョナサン・ボウ
    and the factual close naming him was refused as invention, because the
    exemption held only the Latin spelling."""
    dossier = _dossier(people=["Jonathan Bow"], things=["BBC"])
    check = F2._make_news_read_validator(
        dossier, ["ジョナサン・ボウ"], provenance={"link": "https://www.bbc.co.uk/news/x"},
        language_iso="ja", native_names={"Jonathan Bow": ["ジョナサン・ボウ"]})
    assert check(_read("BBCニュースによると、ジョナサン・ボウ氏が化石を発見した。")) is None


def test_a_native_spelling_never_widens_the_source_gate_for_english():
    """English closes are held to the dossier's own names; a native spelling
    is only ever an extra way to match, never a substitute for the gate."""
    check = F2._make_news_read_validator(
        _dossier(places=["MIT"]), [], native_names={"MIT": ["एमआईटी"]})
    assert check(_read("Scientists continue to study the phenomenon.")) is not None


@pytest.mark.parametrize("value,ok", [
    ("एमआईटी", True),
    ("ジョナサン・ボウ", True),
    ("ボウ", True),
    ("マサチューセッツ工科大学", True),
    ("MIT", False),                 # Latin: the dossier's own anchor already
    ("研究者", False),               # three Han characters: an ordinary word
    ("土星", False),                 # two Han characters
    ("", False),
    ("エ", False),                   # one letter
    ("エ" * 61, False),
])
def test_native_name_ok(value, ok):
    assert F2._native_name_ok(value) is ok


def test_native_names_pass_keeps_only_asked_names_and_sound_spellings(monkeypatch):
    """Python judges the reply: a name nobody asked for, a Latin spelling and
    a three-character Han word are dropped; at most three spellings survive."""
    replies = F2.NativeNames(rows=[
        F2.NativeNameRow(name="MIT", spellings=["एमआईटी", "MIT", "एम.आई.टी.", "एमआइटी", "एमआईटी", "एम आई टी"]),
        F2.NativeNameRow(name="mit", spellings=["एमआईटी"]),
        F2.NativeNameRow(name="Stranger", spellings=["अजनबी"]),
        F2.NativeNameRow(name="Jonathan Bow", spellings=["研究者", ""]),
    ])
    monkeypatch.setattr(F2, "structured_call", lambda **kw: replies)
    out = F2._pass_native_names(
        lambda *a, **k: "", _dossier(people=["Jonathan Bow"], places=["MIT"]),
        {"link": "https://news.mit.edu/2026/x"}, language_label="Hindi")
    # The outlet label "mit" and the place "MIT" are one name.
    assert out == {"MIT": ["एमआईटी", "एम.आई.टी.", "एमआइटी"]}


def test_native_names_pass_degrades_to_nothing_on_failure(monkeypatch):
    def boom(**kw):
        raise RuntimeError("no model")
    monkeypatch.setattr(F2, "structured_call", boom)
    assert F2._pass_native_names(
        lambda *a, **k: "", _dossier(places=["MIT"]), {}, language_label="Hindi") == {}


def test_native_names_pass_asks_nothing_without_names():
    calls = []
    def never(**kw):
        calls.append(kw)
    assert F2._pass_native_names(never, _dossier(numbers=["17.2"]), {}, language_label="Hindi") == {}
    assert calls == []


def test_the_runner_asks_for_native_spellings_before_the_close_and_hands_them_over():
    """A helper nothing calls is this repo's most repeated defect: the call
    must sit in run_scifi_news_pro_episode, before the news read, and the news
    read must receive its result."""
    import inspect
    src = inspect.getsource(F2.run_scifi_news_pro_episode)
    ask = src.index("_pass_native_names(")
    close = src.index("_pass_news_read(")
    assert ask < close
    assert "native_names=native_names" in src[close:close + 600]
    assert "_NATIVE_SCRIPT_ISOS" in src[:ask]


def test_an_empty_dossier_does_not_accuse():
    """A close is only asked to name a source when a source was indexed."""
    check = F2._make_news_read_validator(_dossier(), [])
    assert check(_read("Scientists continue to study the phenomenon.")) is None


def test_matching_is_word_boundary_not_substring():
    """"MIT" must not be found inside "transmitted" -- the twin's rule."""
    check = F2._make_news_read_validator(_dossier(things=["MIT"]), [])
    finding = check(_read("The signal was transmitted overnight."))
    assert finding is not None, "a substring match would have passed this"


@pytest.mark.parametrize("close", [
    "MITの研究者によると、新しいロボットが完成した。",
    "据MIT的研究人员称，新机器人已经完成。",
    "ＭＩＴの研究者によると、新しいロボットが完成した。",
])
def test_a_latin_name_flush_against_japanese_or_chinese_names_the_source(close):
    """PBUG-20260929-06. Japanese and Chinese write no spaces, and `\\b`
    counts Han and kana as word characters, so "MITの" never matched "MIT"
    and a close that named its source was refused for naming nothing."""
    check = F2._make_news_read_validator(_dossier(places=["MIT"]), [])
    assert check(_read(close)) is None


@pytest.mark.parametrize("close", [
    "2026年に発表された研究だ。",
    "该研究于2026年发表。",
    "यह शोध २०२६ में प्रकाशित हुआ।",
    "２０２６年に発表された。",
])
def test_a_number_in_the_closes_own_digits_or_against_han_names_the_source(close):
    check = F2._make_news_read_validator(_dossier(numbers=["2026"]), [])
    assert check(_read(close)) is None


def test_a_longer_latin_word_before_a_particle_is_still_not_the_anchor():
    """The boundary moved for Han and kana only: "Adamの" is not "Ada"."""
    check = F2._make_news_read_validator(_dossier(people=["Ada"]), [])
    assert check(_read("Adamの研究によると、成果が出た。")) is not None


def test_an_anchor_ending_in_a_full_stop_matches_before_a_space():
    """`\\b` after a final "." needed a letter to follow, so "U.S." never
    matched "the U.S. agency"."""
    check = F2._make_news_read_validator(_dossier(places=["U.S."]), [])
    assert check(_read("The U.S. agency confirmed the burst.")) is None


def test_a_short_entity_name_is_not_an_anchor():
    check = F2._make_news_read_validator(_dossier(people=["Xi"]), [])
    assert check(_read("Nothing here names anyone.")) is None


@pytest.mark.parametrize("cast_name", ["", "  ", "Al"])
def test_blank_and_two_letter_cast_names_never_accuse(cast_name):
    check = F2._make_news_read_validator(
        _dossier(things=["Double Chooz"]), [cast_name],
    )
    assert check(_read("The Double Chooz detector logged it.")) is None


def test_the_pass_is_actually_wired_to_the_validator():
    """The finding that started this: the call site passed no validator."""
    import inspect

    source = inspect.getsource(F2._pass_news_read)
    assert "post_validator=_make_news_read_validator(" in source


def test_cast_names_never_reach_the_model_only_the_validator():
    """PBUG-20260824-01 Class B, THE FIX. `_pass_news_read` used to build a
    "FICTIONAL CAST NAMES (never use these ...)" block into the model's own
    prompt -- the exact tokens it must not emit, mirroring the distractor
    `_script_user_prompt` already excludes `news_close_read` for (2026-08-18).
    `cast_names` must still reach the validator, which checks it
    independently of whatever the prompt contains; without that half this
    test would pass whether or not the validator was also broken."""
    import inspect

    source = " ".join(inspect.getsource(F2._pass_news_read).split())
    assert "FICTIONAL CAST NAMES" not in source
    assert ("post_validator=_make_news_read_validator( dossier, cast_names, "
            "provenance=provenance, language_iso=language_iso, "
            "native_names=native_names)") in source


# --------------------------------------------------------------------------- #
# A close in another language names its source by the outlet or by digits
# (PBUG-20260929-06; the closes below are the live ones, 2026-09-29).
# --------------------------------------------------------------------------- #

_SCIENCEDAILY = {"headline": "The ice blasting from Saturn's moon Enceladus",
                 "source": "Latest Science News -- ScienceDaily",
                 "date": "Tue, 29 Sep 2026 05:58:00 EDT",
                 "link": "https://www.sciencedaily.com/releases/2026/09/260929053528.htm"}
_BBC = {"headline": "Fossil found on Welsh beach ends 'confusion'", "source": "BBC News",
        "date": "Mon, 28 Sep 2026 16:04:59 GMT",
        "link": "https://www.bbc.co.uk/news/articles/ckz7zd7nxj73o?at_medium=RSS"}


@pytest.mark.parametrize("iso, provenance, dossier, close", [
    ("zh", _SCIENCEDAILY, dict(people=["Frank Postberg"], places=["Saturn", "Enceladus"]),
     "根据 ScienceDaily 报道，研究人员发现土星卫星恩塞拉多斯的海洋水滴会自然分离并浓缩化学品。"),
    ("ja", _BBC, dict(people=["Jonathan Bow"], places=["Wales"]),
     "BBCニュースは、ウェールズで発見された2億年前の魚の顎の化石が混乱を解消したと報道しました。"),
])
def test_a_close_in_another_language_that_names_the_outlet_names_the_source(
        iso, provenance, dossier, close):
    check = F2._make_news_read_validator(
        _dossier(**dossier), [], provenance=provenance, language_iso=iso)
    assert check(_read(close)) is None


def test_an_english_close_is_still_held_to_the_dossiers_own_names():
    """The outlet counts only where the article's names were translated away."""
    check = F2._make_news_read_validator(
        _dossier(places=["Saturn", "Enceladus"]), [], provenance=_SCIENCEDAILY, language_iso="en")
    finding = check(_read("According to ScienceDaily, researchers reported a discovery."))
    assert finding is not None and "never names the real source" in finding


def test_an_empty_dossier_is_still_pardoned_in_another_language():
    """An outlet anchor must not turn the empty-dossier pardon into a demand."""
    check = F2._make_news_read_validator(_dossier(), [], provenance=_BBC, language_iso="ja")
    assert check(_read("研究は続いている。")) is None


def test_a_number_phrases_digits_name_the_source_in_another_language():
    check = F2._make_news_read_validator(
        _dossier(places=["Wales"], numbers=["200 years", "1,200 samples"]), [],
        provenance=_BBC, language_iso="ja")
    assert check(_read("化石が200年にわたる混乱を解消した。")) is None
    assert check(_read("1200個の試料が調べられた。")) is None


def test_the_items_own_year_is_never_the_anchor():
    """A close could say the year without naming anything from the item."""
    check = F2._make_news_read_validator(
        _dossier(places=["Enceladus"], numbers=["29 Sep 2026", "10"]), [],
        provenance=_SCIENCEDAILY, language_iso="zh")
    finding = check(_read("该研究于2026年发表。"))
    assert finding is not None and "never names the real source" in finding


@pytest.mark.parametrize("link, label", [
    ("https://www.bbc.co.uk/news/articles/x", "bbc"),
    ("https://news.mit.edu/2026/living-transistors-0817", "mit"),
    ("https://www.sciencedaily.com/releases/2026/09/x.htm", "sciencedaily"),
    ("https://blog.ml.cmu.edu/2026/x/", "cmu"),
    ("https://www.nasa.gov/news-release/x", "nasa"),
    ("https://news.example.org/x", "example"),
    ("https://foo.news.edu/x", ""),
    ("https://www.news.org/x", ""),
    ("http://192.168.100.200/x", ""),
    ("https://www.example:8443/x", ""),
    ("", ""),
    ("not a link", ""),
])
def test_the_outlet_is_the_label_in_front_of_the_public_suffix(link, label):
    assert F2._outlet_label(link) == label


def test_the_episode_language_reaches_the_close_check():
    """The helper proves the rule; this proves the lane hands it the language."""
    import inspect

    source = " ".join(inspect.getsource(F2.run_scifi_news_pro_episode).split())
    # The whole call. The iso is read once into a local (2026-09-30) that the
    # native-spellings pass and the close both use; the cameo roll above them
    # reads the same stamp.
    assert "language_iso = _EPLANG.iso_from_meta(meta)" in source
    assert ("read = _pass_news_read( fn, pack, dossier, provenance, source_preview, "
            "cast_names, language_instruction=language_instruction, "
            "language_iso=language_iso, native_names=native_names, )") in source
