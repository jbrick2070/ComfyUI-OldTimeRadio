# -*- coding: utf-8 -*-
"""A row's preflight `required_models` never names the music engine's weights.

WHY (2026-09-26, TEST_WAVE B5-B7 on the 4060 at 1585d38a). All three
`otr_8gb_ltx25_*` rows were refused by the runner's preflight in two seconds:
"requires model file(s) the running server cannot see:
stable_audio_3_small_music_base.safetensors". They were the only rows that
listed Stable Audio 3's files -- added 2026-09-23 (ac612930), before the
queue-time asset planner fetched them. Since then the planner asks the engine
which checkpoint it will use, and the engine takes the base file OR the
post-trained one already on disk (`_CKPT_PREFERENCE`). A box holding only the
post-trained file therefore got "use that one, download nothing", and the
preflight then refused the base file it had been told to require. Pinning one
name contradicts an engine that accepts several.

The music engine's files are the planner's, never a row's: `required_models`
names the video lane's weights, as every other row always has.

Headless. No ComfyUI server, no model, no GPU.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from nodes._otr_audio_engines import eng_stable_audio_3 as sa3

REPO = pathlib.Path(__file__).resolve().parents[1]
ROWS = json.loads((REPO / "config" / "workflow_matrix.json").read_text(encoding="utf-8"))["rows"]

#: Stable Audio 3's checkpoints (any of which the engine accepts) and its text
#: encoder: the planner resolves these, a row never pins one.
MUSIC_ENGINE_FILES = set(sa3._CKPT_PREFERENCE) | {"t5gemma_b_b_ul2.safetensors"}


@pytest.mark.parametrize("row", ROWS, ids=lambda r: r["id"])
def test_no_row_pins_a_music_engine_weight(row):
    required = set((row.get("preflight") or {}).get("required_models") or [])
    assert not (required & MUSIC_ENGINE_FILES), sorted(required & MUSIC_ENGINE_FILES)


def test_the_engine_still_accepts_more_than_one_checkpoint():
    """The reason pinning one name is wrong: if this ever collapses to a single
    file, revisit the rule above rather than deleting this test."""
    assert len(sa3._CKPT_PREFERENCE) > 1
