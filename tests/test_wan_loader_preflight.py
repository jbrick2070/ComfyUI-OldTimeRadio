"""GO_FORWARD section 4A M6 unit tests -- Wan assert_usable loader preflight.

assert_usable previously preflighted ONLY the UNET checkpoint, so a forward
could die mid-graph when the umt5 CLIP or the VAE was absent. M6 makes
assert_usable prove all three graph loaders present on disk (offline invariant:
no runtime fetch) before any forward.

Pure-Python: the dir-override envs (OTR_WAN_I2V_CLIP_DIR / _VAE_DIR) short
-circuit folder_paths / the real models tree, so these are deterministic with
no ComfyUI runtime and no real weights.
"""

from __future__ import annotations

import pytest

from nodes._otr_video_engines.eng_wan_i2v import WanI2VEngine
from nodes._otr_video_engines.registry import (
    EngineUnusable, EngineUsabilityReason,
)


def _stage_full_install(tmp_path, monkeypatch, *, clip=True, vae=True):
    """Point every Wan loader at temp dirs and (optionally) create each file.
    Returns nothing; sets the envs so assert_usable resolves deterministically."""
    ckpt = tmp_path / "wan2.2-i2v.safetensors"
    ckpt.write_bytes(b"unet-placeholder")
    monkeypatch.setenv("OTR_ENABLE_WAN_I2V", "1")
    monkeypatch.setenv("OTR_WAN_I2V_CKPT", str(ckpt))

    clip_dir = tmp_path / "text_encoders"
    clip_dir.mkdir()
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    monkeypatch.setenv("OTR_WAN_I2V_CLIP_DIR", str(clip_dir))
    monkeypatch.setenv("OTR_WAN_I2V_VAE_DIR", str(vae_dir))

    eng = WanI2VEngine()
    names = eng._loader_names()
    if clip:
        (clip_dir / names["clip"]).write_bytes(b"clip-placeholder")
    if vae:
        (vae_dir / names["vae"]).write_bytes(b"vae-placeholder")
    return eng


def test_assert_usable_passes_when_all_three_loaders_present(tmp_path, monkeypatch):
    eng = _stage_full_install(tmp_path, monkeypatch)
    assert eng.assert_usable(host_caps={}, profile={}) == "wan_i2v"


def test_assert_usable_fails_closed_when_clip_absent(tmp_path, monkeypatch):
    eng = _stage_full_install(tmp_path, monkeypatch, clip=False)
    with pytest.raises(EngineUnusable) as exc:
        eng.assert_usable(host_caps={}, profile={})
    assert exc.value.reason is EngineUsabilityReason.MISSING_MODEL
    assert "CLIP/umt5" in str(exc.value)


def test_assert_usable_fails_closed_when_vae_absent(tmp_path, monkeypatch):
    eng = _stage_full_install(tmp_path, monkeypatch, vae=False)
    with pytest.raises(EngineUnusable) as exc:
        eng.assert_usable(host_caps={}, profile={})
    assert exc.value.reason is EngineUsabilityReason.MISSING_MODEL
    assert "VAE" in str(exc.value)


def test_assert_usable_no_flag_gate(tmp_path, monkeypatch):
    # No flag gate (registry IS the menu): a full install is usable even with the
    # (vestigial) OTR_ENABLE_WAN_I2V unset.
    eng = _stage_full_install(tmp_path, monkeypatch)
    monkeypatch.delenv("OTR_ENABLE_WAN_I2V", raising=False)
    assert eng.assert_usable(host_caps={}, profile={}) == "wan_i2v"


def test_assert_usable_ckpt_check_precedes_aux_check(tmp_path, monkeypatch):
    # Flag on, but the UNET ckpt is missing -> MISSING_MODEL names the ckpt,
    # not the aux loaders.
    eng = _stage_full_install(tmp_path, monkeypatch)
    monkeypatch.setenv("OTR_WAN_I2V_CKPT", str(tmp_path / "does-not-exist.safetensors"))
    with pytest.raises(EngineUnusable) as exc:
        eng.assert_usable(host_caps={}, profile={})
    assert exc.value.reason is EngineUsabilityReason.MISSING_MODEL
    assert "checkpoint not found" in str(exc.value)


def test_missing_loaders_lists_both_when_both_absent(tmp_path, monkeypatch):
    eng = _stage_full_install(tmp_path, monkeypatch, clip=False, vae=False)
    missing = {label for label, _name in eng._missing_loaders()}
    assert missing == {"CLIP/umt5", "VAE"}


def test_resolve_model_file_none_for_bogus_name(tmp_path, monkeypatch):
    # No dir override -> folder_paths / standard layout; a clearly-bogus name
    # resolves nowhere -> None (the offline fail-closed path).
    monkeypatch.delenv("OTR_WAN_I2V_VAE_DIR", raising=False)
    eng = WanI2VEngine()
    assert eng._resolve_model_file(
        ("vae",), "otr-definitely-absent-xyz-9f2a.safetensors",
        "OTR_WAN_I2V_VAE_DIR") is None
