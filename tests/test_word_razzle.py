"""word_razzle tombstone -- retired 2026-09-17.

Operator: "WORD_RAZZLE / CLOUD GETS RIPPED". The Pixverse cloud adapter is
gone. The id stays in RETIRED_ENGINE_IDS so a saved graph or force-map fails
as RetiredEngineError, never as a silent remap and never as a generic
unregistered miss. Local razzle_ltx_8gb is the remaining kinetic lane.
"""
from __future__ import annotations

import pytest

from nodes import otr_video_director as vd
from nodes._otr_shared.public_engines import (
    RETIRED_ENGINE_IDS,
    RetiredEngineError,
    check_retired_engine,
)
from nodes._otr_video_engines import registry as vreg


def test_word_razzle_is_retired_and_unregistered():
    assert "word_razzle" in RETIRED_ENGINE_IDS
    assert not vreg.is_registered("word_razzle")
    assert "word_razzle" not in vreg.CAPABILITIES
    with pytest.raises(RetiredEngineError, match="word_razzle"):
        check_retired_engine("word_razzle")


def test_word_razzle_is_absent_from_the_video_combo():
    parsed = {vd._engine_id_from_pick(c) for c in vd._video_model_combo()}
    assert "word_razzle" not in parsed


def test_word_razzle_has_no_pixverse_partner_row():
    from nodes._otr_shared.cloud_media_invoke import partner_rows
    assert "cloud_pixverse_i2v" not in partner_rows()
