"""PBUG-20260815-06, durable-headline half -- an episode must be able to NAME
the post it adapted.

THE SELECTION HALF SHIPPED 2026-08-19 (`3be1c1e1`): the lane no longer retells
the newest feed post forever. This is the OTHER half, and the PBUG called it
its own prerequisite: recording a URL into the shared dedup history is not the
same as the episode's record. Until the headline is stamped in durable meta,
a finished ledger cannot say what it was about.

THE CONSUMER WAS BUILT FIRST AND THE PRODUCER WAS NEVER WIRED.
`_otr_source_identity.identity_from_meta` reads
`source_meta["post_headline"]` on the media_archive lane, and
`SourceIdentity.is_degraded` is True for that lane exactly when the headline is
missing. `_rss_source_fetch_result` built `source_meta` as kind / source_ref /
source_url / source_label / source_date and never the headline -- so EVERY
media_archive episode carried a degraded identity, silently, for as long as
that module has existed. Measured before the fix:

    identity_from_meta(<a real media_archive meta>).is_degraded  ->  True

These tests pin the producer, the consumer, and the join between them, because
the defect lived precisely in the gap where each side looked fine alone.
"""

from __future__ import annotations

import pytest

from nodes import _otr_source_payload as SP
from nodes import _otr_source_identity as SID


def _payload(**over):
    """A valid seven-key source payload; the contract is an EXACT key set."""
    base = {
        "headline": "A lost reel surfaces in a Kansas basement",
        "summary": "A print thought destroyed turns up intact.",
        "full_text": "A print thought destroyed turns up intact. " * 4,
        "source": "Now See Hear!",
        "date": "2026-08-19",
        "link": "https://blogs.loc.gov/now-see-hear/lost-reel",
        "seed_text": "A lost reel surfaces in a Kansas basement.",
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------
# The producer
# --------------------------------------------------------------------------

def test_media_archive_stamps_the_selected_post_headline():
    result = SP._rss_source_fetch_result(
        _payload(), fetcher_kind="media_archive_rss",
    )
    assert result.source_meta["post_headline"] == (
        "A lost reel surfaces in a Kansas basement"
    )


def test_science_lane_stamps_it_too_rather_than_branching_on_the_lane():
    """One shape for both RSS lanes.

    "the selected post's headline" means the same thing on both, so the shared
    helper does not grow a per-lane branch. Science does not consume it yet;
    stamping a true field costs nothing and keeps the lanes aligned.
    """
    result = SP._rss_source_fetch_result(
        _payload(), fetcher_kind="science_rss",
    )
    assert result.source_meta["post_headline"] == (
        "A lost reel surfaces in a Kansas basement"
    )


def test_headline_is_stripped_and_a_missing_one_degrades_quietly():
    """A feed with no usable title must not raise -- a feed lane may never
    fail a render over a metadata field."""
    padded = SP._rss_source_fetch_result(
        _payload(headline="   spaced out   "), fetcher_kind="media_archive_rss",
    )
    assert padded.source_meta["post_headline"] == "spaced out"

    blank = SP._rss_source_fetch_result(
        _payload(headline=""), fetcher_kind="media_archive_rss",
    )
    assert blank.source_meta["post_headline"] == ""


def test_the_seven_key_payload_contract_is_untouched():
    """`post_headline` rides in source_meta, never in the payload -- an
    unknown payload key is a hard contract error by design."""
    result = SP._rss_source_fetch_result(
        _payload(), fetcher_kind="media_archive_rss",
    )
    assert set(result.payload) == set(SP.SOURCE_PAYLOAD_KEYS)
    assert "post_headline" not in result.payload

    with pytest.raises(SP.SourcePayloadContractError):
        SP.validate_source_payload(
            dict(_payload(), post_headline="nope"), origin="test",
        )


# --------------------------------------------------------------------------
# The join -- producer output feeds consumer input
# --------------------------------------------------------------------------

def test_a_stamped_media_archive_episode_is_no_longer_a_degraded_identity():
    """The whole point. Before this fix `is_degraded` was True on every
    media_archive episode."""
    result = SP._rss_source_fetch_result(
        _payload(), fetcher_kind="media_archive_rss",
    )
    meta = {"source_bank": "media_archive",
            "source_meta": dict(result.source_meta)}
    identity = SID.identity_from_meta(meta)

    assert identity.source_kind == "media_archive"
    assert identity.post_headline == (
        "A lost reel surfaces in a Kansas basement"
    )
    assert not identity.is_degraded
    assert identity.provenance["post_headline"] == (
        "meta.source_meta.post_headline"
    )


def test_headline_and_publication_stay_distinct_fields():
    """ONE FIELD TWO MEANINGS is this repo's dominant defect shape.

    On media_archive `work_title` is the PUBLICATION ("Now See Hear!") and the
    headline is the POST. They must never collapse into each other, and the
    publication must stay out of ADAPTATION_SOURCE_KINDS so nothing announces
    a magazine as a play.
    """
    result = SP._rss_source_fetch_result(
        _payload(), fetcher_kind="media_archive_rss",
    )
    identity = SID.identity_from_meta(
        {"source_bank": "media_archive",
         "source_meta": dict(result.source_meta)}
    )
    assert identity.work_title == "Now See Hear!"
    assert identity.post_headline != identity.work_title
    assert identity.source_kind not in SID.ADAPTATION_SOURCE_KINDS


def test_missing_headline_still_reports_itself_degraded():
    """The degrade must stay VISIBLE -- a silent empty is how this defect hid
    in the first place."""
    result = SP._rss_source_fetch_result(
        _payload(headline=""), fetcher_kind="media_archive_rss",
    )
    identity = SID.identity_from_meta(
        {"source_bank": "media_archive",
         "source_meta": dict(result.source_meta)}
    )
    assert identity.post_headline == ""
    assert identity.is_degraded


# --------------------------------------------------------------------------
# The durable carry -- helper output must actually survive into ledger meta
# --------------------------------------------------------------------------
#
# Added after the codex QA lane pointed out that everything above stops at the
# helper. The producer being right is not the same as the value ARRIVING: the
# writer copies `source_meta` wholesale into `meta["source_meta"]`, but it
# also POPS `_news_seed_receipt` out of that same dict as transient. A field
# that shared that fate would look perfect in every test above and still be
# absent from every ledger.

def test_normalize_fetch_result_carries_the_headline_through():
    """`normalize_fetch_result` is the real step between the fetcher and the
    writer -- it copies the sidecar the writer then stamps."""
    result = SP._rss_source_fetch_result(
        _payload(), fetcher_kind="media_archive_rss",
    )
    _payload_out, source_meta, _rights = SP.normalize_fetch_result(
        result, origin="durable carry test",
    )
    assert source_meta["post_headline"] == (
        "A lost reel surfaces in a Kansas basement"
    )


def test_the_headline_survives_the_transient_receipt_pop():
    """The writer does `source_meta.pop("_news_seed_receipt", {})` before
    stamping. `post_headline` must NOT be swept out with it.

    This mirrors `OTR_LedgerScriptWriter._resolve_inputs`, which pops the
    receipt as transient and then stamps the remainder into durable
    `meta["source_meta"]`.
    """
    result = SP._rss_source_fetch_result(
        _payload(),
        fetcher_kind="science_rss",
        news_seed_receipt={"headline": "h", "selected_at": "2026-08-19"},
    )
    _p, source_meta, _r = SP.normalize_fetch_result(
        result, origin="durable carry test",
    )
    assert "_news_seed_receipt" in source_meta          # transient, present here

    # ...the writer's own two lines, verbatim in shape:
    source_meta = dict(source_meta or {})
    receipt = source_meta.pop("_news_seed_receipt", {})
    durable_meta = {"source_meta": dict(source_meta)}

    assert receipt                                       # popped, as intended
    assert "_news_seed_receipt" not in durable_meta["source_meta"]
    assert durable_meta["source_meta"]["post_headline"] == (
        "A lost reel surfaces in a Kansas basement"
    ), "the headline was swept out with the transient receipt"


def test_an_episode_can_name_what_it_adapted_from_durable_meta_alone():
    """The PBUG's actual requirement, end to end from the producer.

    Nothing here is hand-built: the meta is what the fetcher produced, carried
    through normalize, stripped of the transient receipt, and read back by the
    identity authority -- which is how a frozen ledger will be read later.
    """
    result = SP._rss_source_fetch_result(
        _payload(), fetcher_kind="media_archive_rss",
    )
    _p, source_meta, _r = SP.normalize_fetch_result(
        result, origin="durable carry test",
    )
    source_meta = dict(source_meta)
    source_meta.pop("_news_seed_receipt", None)

    identity = SID.identity_from_meta(
        {"source_bank": "media_archive", "source_meta": source_meta}
    )
    assert identity.post_headline == (
        "A lost reel surfaces in a Kansas basement"
    )
    assert not identity.is_degraded
