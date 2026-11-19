"""Stable Audio 3's operator pins are read when used, not at import (audit
secondary). OTR_SA3_CKPT / _TEXT_ENCODER / _CLIP_TYPE were snapshotted when
the pack loaded, so an A/B pin in a running server did nothing until a
restart. UTF-8, no BOM, SFW.
"""
from __future__ import annotations

from nodes._otr_audio_engines import eng_stable_audio_3 as sa3


def test_a_checkpoint_pin_set_after_import_is_honoured(monkeypatch):
    monkeypatch.setenv("OTR_SA3_CKPT", "my_sa3_base.safetensors")
    assert sa3.StableAudio3Engine.resolve_ckpt() == ("my_sa3_base.safetensors", True)


def test_the_text_encoder_and_clip_type_follow_the_env(monkeypatch):
    monkeypatch.setenv("OTR_SA3_TEXT_ENCODER", "other_t5.safetensors")
    monkeypatch.setenv("OTR_SA3_CLIP_TYPE", "stable_audio_alt")
    assert sa3._text_encoder() == "other_t5.safetensors"
    assert sa3._clip_type() == "stable_audio_alt"
    monkeypatch.delenv("OTR_SA3_TEXT_ENCODER")
    assert sa3._text_encoder() == sa3._TENC_DEFAULT
