"""A name the SOURCE uses is never an "invented character".

PBUG-20260829-20. The scifi_news_pro closing read is a FACTUAL report, and the
validator rejects it for naming fictional cast members. But this lane builds its
characters FROM the article's entities, so a real person named in the source
routinely lands in `cast_names` as well -- and the factual close, correctly
naming him, was rejected as invention.

Observed live: an MIT News item about Pat Pataranutaporn. The writer's own
entity pass had extracted him under PEOPLE next to PLACES - MIT. The validator
failed the read twice and killed the episode at 5.7 minutes, with an error that
refutes itself:

    "the closing read is a FACTUAL report and it names invented characters
     (Pataranutaporn). Report only what the source says, using the source's
     own names"

No retry could rescue it -- a lower-temperature repair produced the same
correct text and was rejected identically, because the answer was never wrong.
"""
from __future__ import annotations

import types

from nodes._otr_scifi_news_pro import _make_news_read_validator


def _dossier(people=(), places=(), things=(), numbers=()):
    ents = types.SimpleNamespace(people=list(people), places=list(places),
                                 things=list(things))
    return types.SimpleNamespace(named_entities=ents, allowed_numbers=list(numbers))


def _read(text):
    return types.SimpleNamespace(news_close_read=text)


def test_a_real_person_from_the_source_may_be_named():
    """The exact failure: the source names him AND the cast borrowed him."""
    dossier = _dossier(people=["Pat Pataranutaporn"], places=["MIT"])
    check = _make_news_read_validator(dossier, ["Pataranutaporn", "Dex Mercer"])
    verdict = check(_read("At MIT, Pat Pataranutaporn studies neural transparency."))
    assert verdict is None, (
        "a source-attested name was still rejected as invention: %s" % verdict)


def test_a_genuinely_invented_character_is_still_caught():
    """The real check must survive: a name the source never mentions."""
    dossier = _dossier(people=["Pat Pataranutaporn"], places=["MIT"])
    check = _make_news_read_validator(dossier, ["Dex Mercer"])
    verdict = check(_read("At MIT, Dex Mercer uncovered the neural transparency plot."))
    assert verdict and "invented" in verdict, (
        "a fictional character in a factual report went unflagged: %r" % verdict)


def test_the_source_attribution_check_is_untouched():
    """A close naming nothing from the dossier is still refused."""
    dossier = _dossier(people=["Pat Pataranutaporn"], places=["MIT"])
    check = _make_news_read_validator(dossier, [])
    verdict = check(_read("That is all we know for now. Goodnight."))
    assert verdict and "never names the real source" in verdict


def test_matching_is_case_insensitive():
    dossier = _dossier(people=["Pat Pataranutaporn"], places=["MIT"])
    check = _make_news_read_validator(dossier, ["PATARANUTAPORN"])
    assert check(_read("MIT researcher Pataranutaporn published the study.")) is None


def test_an_empty_dossier_does_not_accuse():
    """Nothing indexed means nothing can be proven either way."""
    check = _make_news_read_validator(_dossier(), ["Dex Mercer"])
    assert check(_read("A quiet night on the wire.")) is None
