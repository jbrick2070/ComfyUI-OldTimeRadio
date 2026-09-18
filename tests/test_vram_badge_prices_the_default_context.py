"""Writer GGUF rows are gone. The 2507 badge must not come back as a fake size.

PBUG-20260829-17 used to pin that a GGUF badge priced default n_ctx, not
the row maximum. `GGUF_ROWS` is now empty, so that id is not a picker
row and must not grow a leftover number.
"""
from __future__ import annotations

from nodes._otr_model_catalog import vram_badge_for

QWEN_2507 = "unsloth/Qwen3-4B-Instruct-2507-GGUF"


def test_retired_writer_gguf_has_no_badge():
    assert vram_badge_for(QWEN_2507) == ""


def test_transformers_rows_are_unchanged():
    """Only a live GGUF row would carry a KV term."""
    for repo in ("google/gemma-4-E2B-it", "google/gemma-2-2b-it"):
        badge = vram_badge_for(repo)
        assert badge and "ctx" not in badge, (
            "%s badge %r gained a context suffix it has no use for"
            % (repo, badge))
        assert "download" in badge, repo


def test_a_badge_never_raises_on_an_unknown_row():
    """A picker must render even for a row nothing can estimate."""
    assert vram_badge_for("definitely/not-a-real-model-xyz") == ""
