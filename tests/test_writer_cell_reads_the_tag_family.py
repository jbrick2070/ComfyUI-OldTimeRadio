# -*- coding: utf-8 -*-
"""A writer cell must read the tag FAMILY, and must not deny a card that fits.

``fit_tags_for`` emits QUALIFIED tags -- ``nv16-nf4`` where only the quantized
load fits, ``mac16-tight`` where it fits with no margin. ``_cell`` used to test
``key in tags``, exact membership, which matches ``nv16`` and never
``nv16-nf4``.

WHY IT SURVIVED SO LONG, and why that is the interesting part: a curated memory
RECEIPT is consulted before this branch, and both rows carrying only ``-nf4``
tags for nv16 had one, so both printed **proven** and the bug was invisible.
The first row to reach this branch with a quantized-only tag printed **no**
under "16 GB+ NVIDIA" for a writer measured at 17.8 GB on a 33.7 GB card.

The second half is about the COLUMN rather than the tag. "16 GB+ NVIDIA" is one
bucket whose own blurb names the 5080 and the 3090 -- 16 GB and 24 GB. A row
that fits the top of that bucket and not the bottom is not a "no" to everyone
reading the column, and the generator's own note says why that matters: "a table
that implies [your hardware cannot do this] sends people to buy a machine they
already own."
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "otr_dropdown_matrix_probe",
    REPO_ROOT / "scripts" / "otr_dropdown_matrix.py")
matrix = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = matrix
_spec.loader.exec_module(matrix)

from nodes import _otr_model_catalog as catalog  # noqa: E402


def _writer_cell(tags, key):
    """A writer row that has reached the derived branch: no receipt."""
    row = {"availability": {key: "ok"}, "remote": False,
           "fit_tags": tuple(tags), "memory": {}}
    return matrix._cell(row, key)


def test_a_quantized_tag_counts_as_a_fit():
    """``nv16-nf4`` means it fits a 16 GB card. The cell must say so."""
    assert matrix._writer_tag_verdict(("nv16-nf4",), "nv16") == "fits"
    assert _writer_cell(("nv16-nf4", "nv24-nf4"), "nv16") == "fits"
    assert matrix._writer_tag_verdict(("nv8-nf4",), "nv8") == "fits"


def test_a_tight_tag_still_reads_tight_and_is_not_lost():
    assert matrix._writer_tag_verdict(("mac16-tight",), "mac16") == "**tight**"
    assert _writer_cell(("mac16-tight", "nv8-nf4"), "mac16") == "**tight**"


def test_a_genuine_miss_is_still_a_no():
    """The honesty half -- this must not turn every cell green."""
    assert matrix._writer_tag_verdict(("nv24-nf4",), "nv8") == ""
    assert _writer_cell(("nv24-nf4",), "nv8") == "**no**"
    assert _writer_cell(("nv24-nf4",), "mac16") == "**no**"


def test_a_key_is_never_matched_by_a_different_class():
    """``nv8`` must not be satisfied by ``nv16``, in either direction."""
    for tags, key in ((("nv16",), "nv8"), (("nv8",), "nv16"),
                      (("nv24-nf4",), "nv16"), (("mac16",), "nv16")):
        assert matrix._writer_tag_verdict(tags, key) == "", (tags, key)


def test_the_16gb_plus_column_does_not_deny_a_24gb_card():
    """The bucket spans the 5080 and the 3090; say which end."""
    assert _writer_cell(("nv24-nf4",), "nv16") == "**24 GB+**"
    assert _writer_cell(("nv24",), "nv16") == "**24 GB+**"
    # and a row that fits neither is still a flat no
    assert _writer_cell((), "nv16") == "**no**"


def test_every_curated_writer_row_renders_a_defensible_nv16_cell():
    """Grounded against the LIVE catalog, not a fixture.

    Any row whose tags say a 16 GB or 24 GB card can run it must never print
    a bare no in that column.
    """
    for row in catalog.CURATED_LLM_MODELS:
        if getattr(row, "provider", "local") != "local":
            continue
        tags = catalog.fit_tags_for(row.repo_id)
        if not tags:
            continue
        cell = _writer_cell(tags, "nv16")
        fits_16 = matrix._writer_tag_verdict(tags, "nv16")
        fits_24 = matrix._writer_tag_verdict(tags, "nv24")
        if fits_16 or fits_24:
            assert cell != "**no**", (
                f"{row.repo_id} tags {tags} say a card in the 16 GB+ column "
                f"can run it, but the cell prints {cell!r}")


def test_the_legend_defines_every_word_the_writer_table_prints():
    """A table may not print a mark the legend leaves undefined."""
    legend = matrix._LEGEND
    doc = (REPO_ROOT / "docs" / "DROPDOWN_MATRIX.md").read_text(encoding="utf-8")
    assert "**24 GB+**" in legend, (
        "the cell can print '24 GB+' but the legend never says what it means")
    if "**24 GB+**" in doc:
        assert "24 GB+" in legend
