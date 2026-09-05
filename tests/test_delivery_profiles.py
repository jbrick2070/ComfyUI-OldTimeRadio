"""E.3 -- the v2.0 delivery-profile contract is neutral-only + strict identity.

Plan E.3 ships exactly ONE delivery profile in v2.0 -- ``neutral`` -- and it is
a strict IDENTITY: no text projection, no tags, no param transform on top of the
engine defaults, so the default audio path is unchanged. Populated profiles are
deferred to v2.1, and ``delivery_profile_version`` is a RE-BASELINE TRIGGER.

These tests lock that contract: if a future change populates a profile or bumps
a version, it fails here and forces the coordinated re-baseline the plan
requires (rather than silently perturbing the audio cache key).
"""
from __future__ import annotations

import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "nodes" / "_otr_delivery_profiles.py"


def test_only_neutral_ships_in_v2():
    from nodes._otr_delivery_profiles import available_delivery_profiles

    # Re-baseline guard: adding a profile in v2.0 must be a deliberate, tracked
    # change (v2.1) -- not a silent edit. This pins the v2.0 set.
    assert available_delivery_profiles() == ["neutral"]


def test_versions_are_pinned_rebaseline_triggers():
    import nodes._otr_delivery_profiles as dp

    # Bumping either is a re-baseline trigger (plan): pin them so the bump is
    # never accidental.
    assert dp.DELIVERY_PROFILE_VERSION == "1"
    assert dp.DELIVERY_PROJECTION_VERSION == "1"


def test_neutral_text_projection_is_identity():
    from nodes._otr_delivery_profiles import get_delivery_profile

    neutral = get_delivery_profile("neutral")
    for text in ("Hello.", "  spaced  ", "[VOICE] tagged", "", "a\nb"):
        assert neutral.project_text(text) == text
    # None folds to the empty string (never a literal "None" in the prompt).
    assert neutral.project_text(None) == ""


def test_neutral_param_overlay_is_identity_copy():
    from nodes._otr_delivery_profiles import get_delivery_profile

    neutral = get_delivery_profile("neutral")
    params = {"temperature": 0.7, "cfg": 0.5}
    out = neutral.overlay_params(params)
    assert out == params, "neutral must not change engine params"
    assert out is not params, "overlay must return a copy, not alias the input"
    assert neutral.overlay_params(None) == {}


def test_unknown_profile_fails_closed():
    from nodes._otr_delivery_profiles import (
        UnknownDeliveryProfile, get_delivery_profile,
    )

    with pytest.raises(UnknownDeliveryProfile):
        get_delivery_profile("dramatic")


def test_delivery_identity_is_in_the_request_cache_key():
    from nodes._otr_resolved_request import IN_KEY_FIELDS, build_resolved_request

    # The id + version stay in the IN_KEY as identity even while only neutral
    # ships (dropping them would force a future re-baseline -- I-6 / E.3).
    assert "delivery_profile_id" in IN_KEY_FIELDS
    assert "delivery_profile_version" in IN_KEY_FIELDS

    req = build_resolved_request(
        role="char_voice", engine_name="bark", engine_profile_id="p",
        engine_impl_version="1", char_id="c1", line_id="l1", prepared_text="hi",
    )
    assert req.delivery_profile_id == "neutral"
    assert req.delivery_profile_version == "1"

    # The version participates in stable_line_seed: a different version must
    # change the derived seed (so a re-baseline is forced, not silent).
    other = build_resolved_request(
        role="char_voice", engine_name="bark", engine_profile_id="p",
        engine_impl_version="1", char_id="c1", line_id="l1", prepared_text="hi",
        delivery_profile_version="2",
    )
    assert other.stable_line_seed != req.stable_line_seed


def test_source_is_ascii_no_em_dash():
    src = SRC.read_text(encoding="utf-8")
    assert "—" not in src, "em-dash forbidden in OTR python source (CLAUDE.md)"
    src.encode("ascii")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
