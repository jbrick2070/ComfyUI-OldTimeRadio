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


def test_an_empty_dossier_does_not_accuse():
    """A close is only asked to name a source when a source was indexed."""
    check = F2._make_news_read_validator(_dossier(), [])
    assert check(_read("Scientists continue to study the phenomenon.")) is None


def test_matching_is_word_boundary_not_substring():
    """"MIT" must not be found inside "transmitted" -- the twin's rule."""
    check = F2._make_news_read_validator(_dossier(things=["MIT"]), [])
    finding = check(_read("The signal was transmitted overnight."))
    assert finding is not None, "a substring match would have passed this"


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
