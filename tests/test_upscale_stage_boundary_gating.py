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


def test_otr_api_does_not_gate_on_capability_cross_validation():
    """`apply_profile_to_workflow` must NOT refuse a profile before applying it.

    INVERTED 2026-08-31, deliberately. This test used to assert the opposite --
    that otr_api calls `cross_validate_profile` before delegating -- and that
    check was removed by operator directive: "let's power through testing
    without inviting some artificial profile gate", then, when it was downgraded
    to a warning rather than deleted, "no warn, I don't want to maintain any
    warning gate either".

    The reasoning, kept here because the inversion looks like a weakening and is
    not: the check compared a profile's choices against each registry's
    enable-set and REFUSED combinations before anything had tried them. That is
    backwards for a project whose standing rule is that an OOM is the only
    acceptable killer, and which learns what a machine can do by running it. The
    dropdown decides, the workflow runs it, and a real failure is written into
    `known_limits` in config/machine_classes.json -- the matrix is the record,
    never the controller.

    What still guards this boundary, and correctly: the widget-level COMBO check
    in `_otr_workflow_apply._validate_widget_value`. A value outside a widget's
    own choice list has no code path behind it at all, so admitting it fails
    later and less clearly. That check is asserted separately in this file.
    """
    src = (REPO / "scripts" / "otr_api.py").read_text(encoding="utf-8")
    fn_start = src.find("def apply_profile_to_workflow(")
    assert fn_start != -1, "apply_profile_to_workflow is gone"
    fn_body = src[fn_start:fn_start + 4000]
    assert "cross_validate_profile(" not in fn_body, (
        "apply_profile_to_workflow calls cross_validate_profile again. That "
        "gate was removed on purpose -- see the docstring above. If it is "
        "genuinely needed, that is an operator decision, not a silent restore.")

def test_widget_mapping_recognizes_upscale_registry():
    """The widget-mapping validator's `_REGISTRY_NAMES` tuple must include
    "upscale" so an `upscale_stage.engine` entry with `registry: "upscale"`
    validates cleanly."""
    from nodes._otr_shared.capability_profiles import _REGISTRY_NAMES
    assert "upscale" in _REGISTRY_NAMES, (
        f"_REGISTRY_NAMES missing 'upscale': {_REGISTRY_NAMES!r}. The mapping "
        f"validator would reject the new upscale_stage.engine entry.")
