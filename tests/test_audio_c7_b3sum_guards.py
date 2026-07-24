"""Positive audio prompt-routing and baseline model coverage."""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_music_prompt_routes_arbitrary_brief_terms_through_protocol():
    source = (REPO_ROOT / "nodes" / "_otr_music_prompt.py").read_text(
        encoding="utf-8"
    )
    assert "_read_brief_field" in source
    assert "def compose_music_prompt" in source


def test_writer_default_mistral_nemo_pinned():
    from nodes import _otr_model_catalog

    assert _otr_model_catalog.DEFAULT_LLM == (
        "mistralai/Mistral-Nemo-Instruct-2407"
    )
