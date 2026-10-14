"""The gallery cards: one per shipped graph, lettered with what it is.

Operator, 2026-10-14: each graph's square gallery image should say in big
letters what is unique about it -- "NVIDIA 8 GB", "MAC ... LTX VIDEO" -- with
the Mac and AMD ones easy to call out, and each hardware class on its own
background. scripts/gallery_thumbs.py derives the words from the matrix;
build_variants writes and checks them.
"""
from __future__ import annotations

import io
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import build_variants as bv  # noqa: E402
import gallery_thumbs as gt  # noqa: E402
from nodes._otr_shared.capability_profiles import (  # noqa: E402
    load_profile, shipping_ids,
)


def _all_cards():
    cards = {"otr_canonical": gt.CANONICAL_CARD, "otr_app": gt.APP_CARD}
    for pid in shipping_ids():
        cards[pid] = gt.card_for_profile(load_profile(pid))
    return cards


def test_every_shipped_graph_has_a_card_that_says_something_different():
    cards = _all_cards()
    seen = {}
    for pid, card in cards.items():
        assert card not in seen, "%s and %s carry the same card" % (pid, seen[card])
        seen[card] = pid


def test_the_mac_and_amd_cards_lead_with_their_hardware():
    for pid in shipping_ids():
        profile = load_profile(pid)
        card = gt.card_for_profile(profile)
        vendor = profile.get("gpu_vendor")
        if vendor == "apple":
            assert card.brand == "MAC" and card.palette == "apple", pid
        elif vendor == "amd":
            assert card.brand == "AMD" and card.palette == "amd", pid
        elif vendor == "nvidia":
            assert card.brand == "NVIDIA" and card.tier.endswith(("GB", "GB+")), pid


def test_each_hardware_class_has_its_own_background():
    fields = {gt.PALETTES[name][:2] for name in gt.PALETTES}
    assert len(fields) == len(gt.PALETTES)
    used = {card.palette for card in _all_cards().values()}
    assert used <= set(gt.PALETTES)


def test_an_nvidia_row_with_no_size_in_its_id_says_so():
    profile = dict(load_profile("otr_8gb_video"), id="otr_nvidia_video")
    with pytest.raises(ValueError, match="VRAM size in the id"):
        gt.card_for_profile(profile)


def test_a_palette_change_makes_every_card_stale(monkeypatch):
    master = b"art"
    card = gt.CANONICAL_CARD
    before = gt.comment_for(card, master)
    palettes = dict(gt.PALETTES)
    palettes["any"] = ((0, 0, 0),) + palettes["any"][1:]
    monkeypatch.setattr(gt, "PALETTES", palettes)
    assert gt.comment_for(card, master) != before


def test_a_lane_with_no_words_is_refused_by_name():
    profile = dict(load_profile("otr_8gb_video"))
    profile["role_overrides"] = dict(profile["role_overrides"],
                                     character_visual="some_new_engine")
    with pytest.raises(ValueError, match="some_new_engine"):
        gt.card_for_profile(profile)


def test_a_drawn_card_carries_its_words_in_the_jpeg():
    pytest.importorskip("PIL")
    from PIL import Image

    master = gt.MASTER_ART.read_bytes()
    card = gt.card_for_profile(load_profile("otr_mac16_video"))
    comment = gt.comment_for(card, master)
    data = gt.render(card, master, comment)
    assert gt.jpeg_comment(data) == comment
    assert gt.is_whole_jpeg(data)
    assert not gt.is_whole_jpeg(data[:500])     # a truncated card is caught
    image = Image.open(io.BytesIO(data))
    assert image.format == "JPEG" and image.size == (gt.CARD_SIZE, gt.CARD_SIZE)
    assert "MAC | 16 GB | LTX VIDEO" in comment


def test_the_committed_cards_are_current():
    assert bv._thumbnail_failures() == []
