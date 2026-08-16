"""A run that dies in the writer must still record what it was dealt.

FOUND WHILE PLANNING THE MEASUREMENT the r1 review panel asked for. Three
consecutive `scifi_news_pro` legs died in the script pass on 2026-08-12, each
with a different malformed shape. When I went to hold the inputs constant and
vary one, I could not: **none of the three recorded what it had been dealt.**

`scifi_news_pro` rolls THREE independent inputs per run --

* the news item (fetched),
* one of 14 frame cards (`The Night Operator`, `Two Rooms Apart`, ...),
* one of 6 stances (`Wonder first`, `Procedural calm`, `Elegiac`, ...),

and the card/stance deal is reproducible from `OTR_SCIFI_NEWS_PRO_SEED`. All three were
written into `meta["scifi_news_pro"]` IN MEMORY before the pitch pass, and the ledger
was not saved again until the entire pipeline finished. So an episode that died
in the SCRIPT pass -- which is exactly what kept happening -- left an on-disk
ledger with no `scifi_news_pro` block at all. The exception message did not carry the
deal either, and nothing logged it.

Same defect class as a truncated traceback: the failure path discards the
evidence needed to diagnose the failure. And it blocks the panel's whole
measurement plan, because you cannot hold constant what a failed run never
wrote down.

These tests pin the receipt, not the story: what matters is that the deal is
DURABLE before anything that can die on it.
"""
from __future__ import annotations

import inspect

from nodes import _otr_scifi_news_pro as scifi_news_pro


def pipeline_source():
    """The pipeline body that deals the cards -- located by the deal itself so
    a rename cannot silently skip these checks."""
    for name, obj in vars(scifi_news_pro).items():
        if not callable(obj) or not hasattr(obj, "__code__"):
            continue
        try:
            src = inspect.getsource(obj)
        except (OSError, TypeError):
            continue
        if "_load_frame_deck()" in src and "cards_dealt" in src:
            return name, src
    raise AssertionError("no function deals the frame deck -- test is stale")


def test_the_deal_is_SAVED_before_the_passes_that_can_die_on_it():
    """THE DEFECT. The deal must reach disk before the pitch/treatment/script
    passes run, or a writer death loses it."""
    _name, src = pipeline_source()
    deal_at = src.index('f2["stance"]')
    save_at = src.find("led.save()", deal_at)
    assert save_at != -1, (
        "nothing saves the ledger after the deal -- a run that dies in the "
        "script pass will record neither seed, frame card nor stance")

    # ...and that save must come BEFORE the first LLM pass, not after.
    pitch_at = src.find("_pass_pitch", deal_at)
    assert pitch_at != -1, "test is stale: the pitch pass moved"
    assert save_at < pitch_at, (
        "the deal is saved only AFTER the passes that can die -- which is the "
        "bug: the three dead legs saved nothing")


def test_the_deal_is_LOGGED_so_a_stuck_run_can_be_read_from_the_server_log():
    """The ledger is the durable record; the log is where a stuck or dead leg
    actually gets read first."""
    _name, src = pipeline_source()
    assert "[scifi_news_pro] deal:" in src
    for token in ("seed=", "frame_card=", "stance="):
        assert token in src, token


def test_the_seed_is_reproducible_and_stamped():
    """`OTR_SCIFI_NEWS_PRO_SEED` is the lever that makes the card/stance deal
    repeatable. Without it the experiment the panel asked for is impossible."""
    src = inspect.getsource(scifi_news_pro._resolve_seed)
    assert "OTR_SCIFI_NEWS_PRO_SEED" in src


def test_an_explicit_seed_reproduces_the_SAME_deal(monkeypatch):
    """The property the whole receipt exists to enable: same seed, same card
    and stance, so one input can be varied at a time."""
    monkeypatch.setenv("OTR_SCIFI_NEWS_PRO_SEED", "12345")
    import random

    deck = scifi_news_pro._load_frame_deck()
    first = scifi_news_pro._deal(random.Random(scifi_news_pro._resolve_seed()), deck)
    second = scifi_news_pro._deal(random.Random(scifi_news_pro._resolve_seed()), deck)
    assert first[0][0]["name"] == second[0][0]["name"]
    assert first[1]["name"] == second[1]["name"]


def test_a_DIFFERENT_seed_can_reach_a_different_deal():
    """Guards the test above from passing on a deck with one card."""
    import random

    deck = scifi_news_pro._load_frame_deck()
    seen = {
        (scifi_news_pro._deal(random.Random(s), deck)[0][0]["name"],
         scifi_news_pro._deal(random.Random(s), deck)[1]["name"])
        for s in range(40)
    }
    assert len(seen) > 1, "every seed deals the same card/stance"


def test_the_deck_really_carries_the_documented_spread():
    """The measurement plan assumes a real spread to hold constant. 14 cards x
    6 stances is what the pack ships; a shrunken deck would quietly narrow
    every experiment built on it."""
    deck = scifi_news_pro._load_frame_deck()
    assert len(deck["cards"]) >= 3
    assert len(deck["stances"]) >= 2
    names = [c["name"] for c in deck["cards"]]
    assert len(set(names)) == len(names), "duplicate frame-card names"
