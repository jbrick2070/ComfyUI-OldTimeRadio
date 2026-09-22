# -*- coding: utf-8 -*-
"""A measured NF4 resident must beat the halving projection in the badge.

THE PROJECTION IS A RULE OF THUMB WITH A KNOWN BIAS. ``fit_tags_for`` prices
a quantized NVIDIA load as ``download_gb / 2``. The download is bf16 -- two
bytes a parameter -- so halving it models ONE byte a parameter, which is an
8-bit load. NF4 is four bits. The projection therefore overstates a quantized
row by close to 2x, which is invisible while every row is small enough to fit
anyway and is not invisible at 27B.

THE HARM IS RECORDED, not hypothetical. ``vram_badge_for``'s own docstring
carries PBUG-20260829-17: a badge that read "(7.9 GB)" against an 8 GB card
sent an owner past the smallest, cheapest writer in the list, and "the cost of
that was a user walking away". A withheld ``nv24-nf4`` tag is the same defect
wearing the opposite sign -- the row simply claims no machine at all, and the
one owner it was added for reads that as "not for me".

This file pins the override and, more importantly, pins that it is
LOAD-BEARING: the second test asserts the projection alone would still get it
wrong, so nobody can delete the table on the grounds that the tags look fine.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_model_catalog as catalog  # noqa: E402

BIG_CARD_WRITER = "Qwen/Qwen3.8-27B"


def _nv24_budget() -> float:
    return dict(catalog._fit_budgets())["nv24"]


def test_the_measured_row_claims_the_card_it_was_measured_on():
    """17.7 GiB measured against a 22.0 GB budget is a fit, and must say so."""
    tags = catalog.fit_tags_for(BIG_CARD_WRITER)
    assert "nv24-nf4" in tags, (
        f"{BIG_CARD_WRITER} measured 17.8 GiB peak under NF4 against an "
        f"nv24 budget of {_nv24_budget()} GB and still claims no machine "
        f"class: {tags!r}. A row that fits nothing is a row nobody picks.")
    # It must NOT claim the small cards. This is the honesty half.
    for withheld in ("mac16", "mac16-tight", "nv8", "nv8-nf4", "nv16",
                     "nv16-nf4"):
        assert withheld not in tags, (
            f"{BIG_CARD_WRITER} claims {withheld!r}; a 51.75 GiB download "
            "must never advertise a small card")


def test_the_override_is_load_bearing_not_decorative():
    """Delete the table and the tag goes away -- so it cannot be 'simplified'.

    Without this, a future reader sees correct tags, assumes the projection
    produced them, and removes the table as redundant.
    """
    row = next(r for r in catalog.CURATED_LLM_MODELS
               if r.repo_id == BIG_CARD_WRITER)
    projected = row.approx_safetensors_gb / 2.0
    budget = _nv24_budget()
    assert projected > budget, (
        f"the projection ({projected:.1f} GB) now fits the nv24 budget "
        f"({budget} GB) on its own, so this test proves nothing. Either the "
        "download size or the budget moved -- re-derive both before trusting "
        "the tag.")
    measured = catalog.MEASURED_NVIDIA_NF4_GB[BIG_CARD_WRITER]
    assert measured <= budget, (
        f"the measured figure {measured} GB no longer fits {budget} GB; the "
        "tag is now wrong and the row is advertising a card it cannot use")


def test_an_unmeasured_row_still_uses_the_projection():
    """The table is an override for measured rows, never a new requirement.

    Every other row must keep the exact tags it had before the table existed,
    which is what makes this safe to add to shared code.
    """
    assert set(catalog.MEASURED_NVIDIA_NF4_GB) == {BIG_CARD_WRITER}, (
        "a row was measured into the table; extend this pin deliberately "
        "rather than loosening it")
    # gemma-4-12b-it is the nearest neighbour: same loader backend, same
    # baked NF4, unmeasured, and it earns its tags from the projection.
    assert catalog.fit_tags_for("google/gemma-4-12b-it") == (
        "nv16-nf4", "nv24-nf4")
    assert catalog.fit_tags_for("Qwen/Qwen3.5-4B") == (
        "mac16-tight", "nv8-nf4", "nv16", "nv24")
