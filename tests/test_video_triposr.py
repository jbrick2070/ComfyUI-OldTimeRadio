"""triposr -- the lower-tier MIT 3D mesher (dark scaffold, 2026-06-18).

Registered DEFAULT-OFF / dark (V-6: appears in the full registry but never a
default; fails closed until the operator installs a TripoSR node + MIT weights
and GPU-validates). Kept OUT of the tested-only dropdown until then.
"""
from __future__ import annotations

import pytest

from nodes._otr_video_engines import eng_triposr  # noqa: F401  (register adapter)
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import schemas


def test_triposr_registered_in_full_dropdown():
    # V-6: present in the full static registry...
    assert "triposr" in vreg.all_engine_names()
    eng = vreg.get_engine("triposr")
    assert eng.commercial_clean is True          # MIT
    assert eng.default_roles == ()               # never a default (dark)
    assert eng.requires_flag == "OTR_ENABLE_TRIPOSR"
    assert eng.required_inputs == ("init_image",)
    assert eng.fallback_engine == "still_parallax"


def test_triposr_family_is_static_not_character_3d():
    eng = vreg.get_engine("triposr")
    # STATIC mesher: turntable motion only, NEVER lip-sync.
    assert eng.family == "image_to_video"
    assert eng.family in schemas.FAMILIES
    assert "audio_ref" not in eng.required_inputs


def test_triposr_capabilities_row():
    row = vreg.CAPABILITIES["triposr"]
    assert row["vram_class"] == "medium"          # the 8 GB tier
    assert row["required_toolchain"] is None      # no cu128 toolkit
    assert row["requires_sidecar"] is False
    assert row["model_requirements"] == ["triposr"]


def test_triposr_hidden_until_validated():
    # The tested-only dropdown gate hides it until GPU validation lands.
    assert "triposr" not in vreg.VALIDATED_ENGINES
    assert "triposr" not in vreg.validated_engine_names()


def test_triposr_fails_closed_when_flag_off(monkeypatch):
    monkeypatch.delenv("OTR_ENABLE_TRIPOSR", raising=False)
    eng = vreg.get_engine("triposr")
    with pytest.raises(vreg.EngineUnusable) as ei:
        eng.assert_usable(host_caps=None, profile=None)
    assert ei.value.reason == vreg.EngineUsabilityReason.GATED_BY_FLAG


def test_triposr_flag_on_then_missing_weights(monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_TRIPOSR", "1")
    monkeypatch.delenv("OTR_TRIPOSR_WEIGHTS_DIR", raising=False)
    eng = vreg.get_engine("triposr")
    with pytest.raises(vreg.EngineUnusable) as ei:
        eng.assert_usable(host_caps=None, profile=None)
    assert ei.value.reason == vreg.EngineUsabilityReason.MISSING_MODEL


def test_triposr_load_and_render_are_dark(monkeypatch):
    eng = vreg.get_engine("triposr")
    with pytest.raises(RuntimeError):
        eng.load()
    with pytest.raises(NotImplementedError):
        eng.render_clip(request=None, prepared=None)
