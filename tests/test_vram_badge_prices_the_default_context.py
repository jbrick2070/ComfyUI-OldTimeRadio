"""What the writer picker's size badge says, for rows it can and cannot price.

PBUG-20260829-17 is why no surviving row carries a context term: every row is a
safetensors download whose badge does not move with the context asked for.
"""
from __future__ import annotations

from nodes import _otr_model_catalog as catalog
from nodes._otr_model_catalog import vram_badge_for


def test_transformers_rows_carry_no_context_term():
    for repo in ("google/gemma-4-E2B-it", "google/gemma-2-2b-it"):
        badge = vram_badge_for(repo)
        assert badge and "ctx" not in badge, (
            "%s badge %r gained a context suffix it has no use for"
            % (repo, badge))
        assert "download" in badge, repo


def test_a_badge_never_raises_on_an_unknown_row():
    """A picker must render even for a row nothing can estimate."""
    assert vram_badge_for("definitely/not-a-real-model-xyz") == ""


def test_an_uncurated_row_with_an_estimate_gets_its_resident_size(monkeypatch):
    """No catalog download size, but a resident estimate: the badge states it.

    This path referenced an undefined name until 2026-09-25, so a cached
    uncurated writer raised NameError inside the dropdown builder instead of
    getting a badge.
    """
    monkeypatch.setattr(catalog, "_estimate_resident_gb", lambda _repo: 5.0)
    monkeypatch.setattr(catalog, "fit_tags_for", lambda _repo: ())
    assert vram_badge_for("someone/an-uncurated-local-model") == " (5.0 GB)"
