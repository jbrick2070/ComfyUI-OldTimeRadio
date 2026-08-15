"""The coda must name what it adapted -- and must not name the licensor.

Three live episodes on 2026-08-15 closed without naming their source:
`ghost_of_elsinore` said "We've lost our signal." and named nobody;
`midnights_ticktock` said "a work in the public domain" without naming
"Gertrude the Governess" or Stephen Leacock. The mechanism was one function
reading one field: `spoken_coda_line` keyed on licence STATUS alone and took
no per-work input at all.

These tests pin the identity extraction (including the rule the whole review
panel got wrong) and the two directions of the coda: unchanged without an
identity, richer with one.
"""

import pytest

from nodes import _otr_provenance as PROV
from nodes._otr_source_identity import (
    SOURCE_IDENTITY_VERSION,
    SourceIdentity,
    identity_from_meta,
)


# --------------------------------------------------------------------------
# identity extraction
# --------------------------------------------------------------------------

def _pd_meta(**over):
    meta = {
        "source_bank": "public_domain",
        "source_meta": {
            "source_ref": "gertrude_governess:main",
            "title": "Nonsense Novels",
            "author": "Stephen Leacock",
            "unit_label": "Gertrude the Governess",
            "source_url": "https://www.gutenberg.org/cache/epub/4682/pg4682.txt",
        },
    }
    meta["source_meta"].update(over)
    return meta


def test_public_domain_identity_reads_title_and_author():
    ident = identity_from_meta(_pd_meta())
    assert ident.source_kind == "public_domain"
    assert ident.work_title == "Nonsense Novels"
    assert ident.author == "Stephen Leacock"
    assert ident.names_a_work is True
    assert ident.is_degraded is False
    assert ident.provenance["work_title"] == "meta.source_meta.title"


def test_the_unit_label_is_never_the_work_title():
    """The rule the corpus settled, against four reviewers' advice.

    60 of 65 units carry a `label` that differs from `title`, but `label` is a
    SCENE DESCRIPTOR -- dracula's is "Harker at the castle". Announcing it as
    the work would name a passage as if it were a book on ~57 of 65 units.
    """
    ident = identity_from_meta(_pd_meta(
        title="Dracula",
        author="Bram Stoker",
        unit_label="Harker at the castle",
    ))
    assert ident.work_title == "Dracula"
    assert "Harker" not in ident.work_title
    assert "Harker" not in (ident.post_headline or "")


def test_shakespeare_identity_supplies_the_author_the_lane_has_no_field_for():
    ident = identity_from_meta({
        "source_bank": "shakespeare",
        "source_meta": {"play_title": "Macbeth", "play_code": "Mac"},
    })
    assert ident.work_title == "Macbeth"
    assert ident.author == "William Shakespeare"
    assert ident.is_degraded is False


def test_media_archive_identity_names_the_post_not_only_the_feed():
    ident = identity_from_meta({
        "source_bank": "media_archive",
        "source_meta": {
            "source_label": "Now See Hear!",
            "post_headline": "Film loans for the month",
        },
    })
    assert ident.post_headline == "Film loans for the month"
    assert ident.work_title == "Now See Hear!"
    assert ident.is_degraded is False


def test_media_archive_without_a_stamped_headline_is_degraded():
    """The prerequisite defect: the post headline was never stamped durably."""
    ident = identity_from_meta({
        "source_bank": "media_archive",
        "source_meta": {"source_label": "Now See Hear!"},
    })
    assert ident.post_headline == ""
    assert ident.is_degraded is True


@pytest.mark.parametrize("meta", [None, "", 42, [], {}, {"source_bank": "original"}])
def test_unreadable_or_invention_lane_meta_never_raises(meta):
    ident = identity_from_meta(meta)
    assert isinstance(ident, SourceIdentity)
    assert ident.names_a_work is False
    assert ident.is_degraded is True


def test_receipt_carries_field_level_provenance_and_a_version():
    receipt = identity_from_meta(_pd_meta()).as_receipt()
    assert receipt["version"] == SOURCE_IDENTITY_VERSION
    assert receipt["degraded"] is False
    assert receipt["field_provenance"]["author"] == "meta.source_meta.author"


# --------------------------------------------------------------------------
# the coda itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize("status", sorted(PROV._CODA_BY_STATUS))
def test_without_an_identity_the_coda_is_byte_identical_to_before(status):
    """Backward compatibility is the whole reason identity is optional."""
    assert PROV.spoken_coda_line({"status": status}) == PROV._CODA_BY_STATUS[status]


def test_public_domain_coda_names_the_work_and_the_author():
    line = PROV.spoken_coda_line(
        {"status": "public_domain_us"},
        identity_from_meta(_pd_meta(title="The Time Machine", author="H. G. Wells")),
    )
    assert "The Time Machine" in line
    assert "H. G. Wells" in line


def test_cc0_keeps_the_phrase_its_existing_test_pins():
    line = PROV.spoken_coda_line(
        {"status": "cc0"}, identity_from_meta(_pd_meta()))
    assert "dedicated to the public domain" in line
    assert "Nonsense Novels" in line


def test_a_licensed_source_now_names_the_author_and_the_play():
    """The 2026-08-15 supersession. Shakespeare's lane is licensed_noncommercial."""
    line = PROV.spoken_coda_line(
        {"status": "licensed_noncommercial", "source_label": "Folger Shakespeare",
         "license_label": "CC BY-NC 3.0"},
        identity_from_meta({
            "source_bank": "shakespeare",
            "source_meta": {"play_title": "Macbeth"},
        }),
    )
    assert "Macbeth" in line
    assert "Shakespeare" in line
    # The licensor and the licence stay print-only. This is the half of the
    # 2026-08-05 ruling that still stands.
    assert "Folger" not in line
    assert "CC BY" not in line


def test_a_licensed_source_with_no_identity_still_says_nothing():
    assert PROV.spoken_coda_line({"status": "licensed_noncommercial"}) == ""


def test_a_half_known_identity_degrades_to_the_generic_sentence():
    """An author without a title is vaguer than the sentence it would replace."""
    partial = identity_from_meta(_pd_meta(title=""))
    line = PROV.spoken_coda_line({"status": "public_domain_us"}, partial)
    assert line == PROV._CODA_BY_STATUS["public_domain_us"]


def test_the_coda_never_raises_on_a_junk_identity():
    for junk in (None, "", 0, object()):
        PROV.spoken_coda_line({"status": "public_domain_us"}, junk)
