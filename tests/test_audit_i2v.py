"""word_razzle Phase 0 -- the --audit-i2v classifier (PURE, no live core).

The audit walks the live comfy_api_nodes catalog for a promptable, non-V3,
image-to-video row (the animated word card's cloud engine). The LIVE walk needs
the comfy core (and corrupts pytest teardown -- run it as a subprocess, exactly
like the pin --check drift test), so here we test the PURE classifier + verdict
logic against synthetic schemas. This is the "test invokes it" guard the plan
requires.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[1]
           / "scripts" / "otr_pin_partner_nodes.py")


def _load():
    spec = importlib.util.spec_from_file_location("otr_pin_partner_nodes", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:                          # comfy core unresolvable on this box
        pytest.skip("comfy core unresolvable (rc4) -- audit classifier untestable",
                    allow_module_level=True)
    return mod


aud = _load()


def _inp(required=None, optional=None, hidden=None):
    return {"required": required or {}, "optional": optional or {},
            "hidden": hidden or {}}


# --------------------------------------------------------------------------- #
# _is_v3_token
# --------------------------------------------------------------------------- #
def test_is_v3_token():
    assert aud._is_v3_token("DYNAMICCOMBO_V3")
    assert aud._is_v3_token("DynamicCombo_v3")
    assert not aud._is_v3_token("COMBO")
    assert not aud._is_v3_token("IMAGE")
    assert not aud._is_v3_token("STRING")


# --------------------------------------------------------------------------- #
# _classify_video_row -- pass / fail per filter
# --------------------------------------------------------------------------- #
def test_promptable_i2v_row_passes():
    rec = aud._classify_video_row(
        "SomePixVerseI2V",
        _inp(required={"image": "IMAGE", "prompt": "STRING", "seed": "INT"}),
        ["VIDEO"])
    assert rec["pass"] is True
    assert rec["prompt_kwarg"] == "prompt"
    assert rec["image_input"] == "image"
    assert rec["seed_supported"] is True
    assert rec["contains_dynamic_v3"] is False


def test_required_audio_non_avatar_fails():
    rec = aud._classify_video_row(
        "SomeTalkingNode",
        _inp(required={"image": "IMAGE", "prompt": "STRING", "audio": "AUDIO"}),
        ["VIDEO"])
    assert rec["pass"] is False
    assert rec["required_audio"] == "audio"
    assert rec["filters"]["no_required_audio"] is False
    assert any("requires AUDIO" in r for r in rec["reasons"])


def test_avatar_is_audio_exempt():
    rec = aud._classify_video_row(
        "KlingAvatarNode",
        _inp(required={"image": "IMAGE", "audio": "AUDIO"},
             optional={"prompt": "STRING"}),
        ["VIDEO"])
    assert rec["filters"]["no_required_audio"] is True     # exempt, not auto-fail
    assert rec["audio_available_exempt"] is True
    assert rec["pass"] is True                             # promptable + image + exempt


def test_v3_dynamic_row_blocked():
    rec = aud._classify_video_row(
        "ByteDance2ReferenceNode",
        _inp(required={"first_frame": "IMAGE", "prompt": "STRING",
                       "model": "DYNAMICCOMBO_V3"}),
        ["VIDEO"])
    assert rec["contains_dynamic_v3"] is True
    assert rec["pass"] is False
    assert rec["image_input"] == "first_frame"             # name-hint resolves it
    assert any("dynamic V3" in r for r in rec["reasons"])


def test_no_prompt_fails():
    rec = aud._classify_video_row(
        "PlainI2V", _inp(required={"image": "IMAGE"}), ["VIDEO"])
    assert rec["pass"] is False
    assert rec["prompt_kwarg"] is None


def test_no_image_fails():
    rec = aud._classify_video_row(
        "PlainT2V", _inp(required={"prompt": "STRING"}), ["VIDEO"])
    assert rec["pass"] is False
    assert rec["image_input"] is None


def test_missing_seed_is_flag_not_hard_fail():
    rec = aud._classify_video_row(
        "NoSeedI2V", _inp(required={"image": "IMAGE", "prompt": "STRING"}),
        ["VIDEO"])
    assert rec["seed_supported"] is False
    assert rec["pass"] is True                             # seed absence != fail
    assert any("no seed" in r for r in rec["reasons"])


def test_prompt_name_variant_resolves():
    rec = aud._classify_video_row(
        "V", _inp(required={"reference_image": "IMAGE", "positive_prompt": "STRING"}),
        ["VIDEO"])
    assert rec["prompt_kwarg"] == "positive_prompt"
    assert rec["image_input"] == "reference_image"


# --------------------------------------------------------------------------- #
# _audit_verdict -- the decision tree
# --------------------------------------------------------------------------- #
def test_verdict_candidate_found():
    assert aud._audit_verdict([{"pass": True}]) == "CANDIDATE_FOUND"


def test_verdict_blocked_on_v3():
    assert aud._audit_verdict(
        [{"pass": False, "contains_dynamic_v3": True}]) == "BLOCKED_ON_V3_EXPANSION"


def test_verdict_blocked_no_usable_row():
    assert aud._audit_verdict(
        [{"pass": False, "contains_dynamic_v3": False}]) == "BLOCKED_NO_USABLE_ROW"
    assert aud._audit_verdict([]) == "BLOCKED_NO_USABLE_ROW"
