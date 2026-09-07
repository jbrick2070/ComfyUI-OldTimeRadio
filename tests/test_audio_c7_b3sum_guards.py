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


def test_writer_default_is_pinned_and_runs_on_the_smallest_target():
    """The writer widget default is PINNED, and to a row that actually loads.

    Was pinned to mistralai/Mistral-Nemo-Instruct-2407 until 2026-09-06. That
    row is not broken -- it is the soak-tested C7 baseline and it runs happily
    on a 16 GB card at its 12.0 GB resident size. It was the wrong DEFAULT,
    because a default is what someone gets having changed nothing, and on the
    project's smallest supported target it is a 24 GB download that then does
    not fit at all.

    Qwen/Qwen3.5-4B is pinned instead: measured on a physical 8 GB RTX 4060 on
    2026-09-06 at 2.99 GiB resident and 14.47 tok/s, ungated, Apache-2.0, and it
    carried a complete one-act episode to obs_publish on that card.

    THIS IS A WIDGET DEFAULT, NOT A FALLBACK. Nothing substitutes this row for a
    model the user selected: a selected model that cannot load raises, by
    operator directive. A ComfyUI COMBO must hold some value, and an unpinned
    one silently resolves to whichever row sits at index 0 -- which is why this
    assertion exists at all. Changing the pin is a deliberate act; deleting it
    is not.

    C7 audio byte-identity is unaffected: the clamp and prompt-routing tests
    pass Mistral-Nemo as an explicit literal and never read DEFAULT_LLM.
    """
    from nodes import _otr_model_catalog

    assert _otr_model_catalog.DEFAULT_LLM == "Qwen/Qwen3.5-4B"
