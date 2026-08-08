"""Queue item 8 (2026-08-08): both application boundaries call
cross_validate_profile.

Codex r4 MF-3: `apply_profile` in `nodes/_otr_workflow_apply.py` is called
directly by BOTH `scripts/build_variants.py:build_variant` AND
`scripts/otr_api.py:apply_profile_to_workflow`. If cross_validate is only
installed at build_variants, live api runs (canonical + otr_upscale_ship)
bypass validation and a broken upscale_stage.engine slips through to render.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_build_variants_calls_cross_validate():
    """`scripts/build_variants.py:build_variant()` must import and call
    `cross_validate_profile` before `apply_profile`."""
    src = (REPO / "scripts" / "build_variants.py").read_text(encoding="utf-8")
    assert "cross_validate_profile" in src, (
        "build_variants.py does not reference cross_validate_profile; "
        "Codex r4 MF-3 boundary gap regressed")
    # Order: the call must appear BEFORE the actual `applied = apply_profile(...)`
    # invocation. Use `applied = apply_profile(canonical` to skip a docstring
    # mention of apply_profile that predates the real call.
    xv_idx = src.find("cross_validate_profile(profile")
    ap_idx = src.find("applied = apply_profile(canonical")
    assert xv_idx != -1, "no cross_validate_profile(profile, ...) call found"
    assert ap_idx != -1, "no `applied = apply_profile(canonical, ...)` call found"
    assert xv_idx < ap_idx, (
        "cross_validate_profile must be called BEFORE apply_profile; if the "
        "profile is invalid, we don't want any canonical mutation attempted")


def test_otr_api_calls_cross_validate():
    """`scripts/otr_api.py:apply_profile_to_workflow` must import and call
    `cross_validate_profile` before delegating to `apply_profile`."""
    src = (REPO / "scripts" / "otr_api.py").read_text(encoding="utf-8")
    assert "cross_validate_profile" in src, (
        "otr_api.py does not reference cross_validate_profile; "
        "Codex r4 MF-3 boundary gap regressed")
    # The call must appear inside apply_profile_to_workflow, BEFORE the
    # return apply_profile(...) at line ~826.
    fn_start = src.find("def apply_profile_to_workflow(")
    assert fn_start != -1
    # Read a generous window; the fn is long (~30 lines).
    fn_body = src[fn_start:fn_start + 3000]
    assert "cross_validate_profile" in fn_body, (
        "apply_profile_to_workflow does not call cross_validate_profile "
        "in its body; live API runs bypass upscale-engine validation")
    xv_idx = fn_body.find("cross_validate_profile(")
    ap_idx = fn_body.find("return apply_profile(")
    assert xv_idx != -1, "no cross_validate_profile(...) call in fn body"
    assert ap_idx != -1, "no `return apply_profile(...)` in fn body"
    assert xv_idx < ap_idx, (
        "cross_validate_profile must precede the return apply_profile call")


def test_widget_mapping_recognizes_upscale_registry():
    """The widget-mapping validator's `_REGISTRY_NAMES` tuple must include
    "upscale" so an `upscale_stage.engine` entry with `registry: "upscale"`
    validates cleanly."""
    from nodes._otr_shared.capability_profiles import _REGISTRY_NAMES
    assert "upscale" in _REGISTRY_NAMES, (
        f"_REGISTRY_NAMES missing 'upscale': {_REGISTRY_NAMES!r}. The mapping "
        f"validator would reject the new upscale_stage.engine entry.")
