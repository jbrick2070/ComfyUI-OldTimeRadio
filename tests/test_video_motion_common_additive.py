"""Additive CPU coverage for the shared in-process motion-engine helpers (A-S5).

New, self-contained tests for nodes/_otr_video_engines/motion_common.py pure
surface, complementing tests/test_video_motion*.py: the BUG-070 SageAttention
contamination gate (module/env detection + fail-closed), isolation-tier
resolution, the init_image aspect transform (ONE uniform scale, never an implicit
stretch; even dims), the no-silent-stretch guard, NVML-safe VRAM probes, and the
MotionEngineBase load/unload/detach contract (V-4: never unload_all_models). No
GPU required (NVML-absent paths are handled). No production edits. UTF-8, no BOM,
ASCII-only, SFW.
"""
from __future__ import annotations

import pytest

from nodes._otr_video_engines import motion_common as mc
from nodes._otr_shared import engine_registry_base as erb


# --------------------------------------------------------------------------- #
# BUG-070 SageAttention gate
# --------------------------------------------------------------------------- #
def test_sageattention_patched_detects_module_and_env_override():
    assert mc.sageattention_patched(modules={"sageattention": object()}) is True
    assert mc.sageattention_patched(
        modules={}, env={"OTR_SAGEATTENTION_PATCHED": "1"}) is True
    assert mc.sageattention_patched(modules={}, env={}) is False


def test_assert_sage_not_patched_raises_when_patched_else_returns():
    with pytest.raises(erb.EngineUnusable) as exc:
        mc.assert_sage_not_patched(
            "ltx_video", "text_to_video", modules={"sageattention": 1})
    assert exc.value.reason is erb.EngineUsabilityReason.INCOMPATIBLE_PROFILE
    assert mc.assert_sage_not_patched(
        "ltx_video", "text_to_video", modules={}, env={}) == "ltx_video"


def test_resolve_isolation_escalates_optional_when_sage_resident():
    assert mc.resolve_isolation(mc.ISOLATION_SIDECAR_REQUIRED, False) == (
        mc.ISOLATION_SIDECAR_REQUIRED)
    assert mc.resolve_isolation(mc.ISOLATION_SIDECAR_OPTIONAL, True) == (
        mc.ISOLATION_SIDECAR_REQUIRED)
    assert mc.resolve_isolation(mc.ISOLATION_SIDECAR_OPTIONAL, False) == (
        mc.ISOLATION_IN_PROCESS)
    assert mc.resolve_isolation(mc.ISOLATION_IN_PROCESS, True) == (
        mc.ISOLATION_IN_PROCESS)
    assert mc.resolve_isolation("anything_else", False) == mc.ISOLATION_IN_PROCESS


# --------------------------------------------------------------------------- #
# init_image aspect handling -- no silent stretch
# --------------------------------------------------------------------------- #
def test_assert_aspect_policy_accepts_known_rejects_unknown():
    for p in mc.ASPECT_POLICIES:
        assert mc.assert_aspect_policy(p) == p
    with pytest.raises(ValueError):
        mc.assert_aspect_policy("stretch")


def test_resolve_aspect_transform_pad_fits_inside_uniform_scale():
    plan = mc.resolve_aspect_transform(480, 832, 1920, 1080, "pad")
    assert plan["policy"] == "pad"
    assert plan["scaled_w"] % 2 == 0 and plan["scaled_h"] % 2 == 0
    # fits INSIDE the canvas (letterbox/pillarbox), never larger
    assert plan["scaled_w"] <= 1920 and plan["scaled_h"] <= 1080
    # ONE uniform scale -> per-axis scales match
    sx = plan["scaled_w"] / plan["src_w"]
    sy = plan["scaled_h"] / plan["src_h"]
    assert abs(sx - sy) < 0.02 * max(sx, sy)


def test_resolve_aspect_transform_crop_covers_uniform_scale():
    plan = mc.resolve_aspect_transform(480, 832, 1920, 1080, "crop")
    assert plan["policy"] == "crop"
    assert plan["scaled_w"] % 2 == 0 and plan["scaled_h"] % 2 == 0
    # covers the canvas (>= on both axes)
    assert plan["scaled_w"] >= 1920 and plan["scaled_h"] >= 1080


def test_resolve_aspect_transform_rejects_nonpositive_and_unknown_policy():
    with pytest.raises(ValueError):
        mc.resolve_aspect_transform(0, 832, 1920, 1080, "pad")
    with pytest.raises(ValueError):
        mc.resolve_aspect_transform(480, 832, 1920, 1080, "stretch")


def test_assert_no_silent_stretch_rejects_nonuniform_plan():
    uniform = {"src_w": 100, "scaled_w": 200, "src_h": 50, "scaled_h": 100}
    assert mc.assert_no_silent_stretch(uniform) is True
    stretched = {"src_w": 100, "scaled_w": 200, "src_h": 100, "scaled_h": 110}
    with pytest.raises(ValueError):
        mc.assert_no_silent_stretch(stretched)


# --------------------------------------------------------------------------- #
# NVML-safe VRAM probes (no GPU required)
# --------------------------------------------------------------------------- #
def test_vram_helpers_are_safe_without_nvml():
    v = mc.vram_used_mb()
    assert v is None or (isinstance(v, int) and v >= 0)


# --------------------------------------------------------------------------- #
# MotionEngineBase lifecycle (V-4: never unload_all_models)
# --------------------------------------------------------------------------- #
def test_motion_base_defaults_and_load_not_implemented():
    base = mc.MotionEngineBase()
    assert mc.MotionEngineBase.declared_isolation == mc.ISOLATION_IN_PROCESS
    assert mc.MotionEngineBase.binds_seed is True
    with pytest.raises(NotImplementedError):
        base.load()


def test_motion_base_unload_resets_state():
    base = mc.MotionEngineBase()
    base._loaded = True
    base._patchers = [object()]
    base.unload()
    assert base._loaded is False
    assert base._patchers == []


def test_motion_base_detach_patchers_is_robust_and_clears():
    calls = []

    class _P:
        def detach(self, unpatch_all=False):
            calls.append(unpatch_all)

    class _Bad:
        def detach(self, unpatch_all=False):
            raise RuntimeError("detach blew up")

    prepared = {"patchers": [_P(), _Bad()]}
    mc.MotionEngineBase._detach_patchers(prepared)
    assert calls == [True]                      # _P detached with unpatch_all=True
    assert prepared["patchers"] == []           # cleared even though _Bad raised
    # None is a safe no-op
    mc.MotionEngineBase._detach_patchers(None)
