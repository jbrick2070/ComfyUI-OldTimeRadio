"""S4b/S4c (2026-07-02): face-forward portrait mint for the ia2v talking
lane + radio-face default-on for its bookends.

Proof8 root cause: S4 routed character inits to the PORTRAIT, but the
brief-styled portrait mint produced dark profile/wide compositions (mouth
not visible) -- the lip-sync coupling had nothing to drive. S4b threads a
per-role ``talking`` map through the director policy chain and mints those
portraits face-forward + warm, skipping the era/grade tails (the exact
ltx_radio_mouth split that fixed the radio). S4c retires the
OTR_LTX_RADIO_FACE A/B into default-on for talking bookends (operator
eyeball: "should be talking lips?").
"""
import json

import pytest

import nodes.otr_meta_brief_image_prompt as mbp
from nodes.otr_meta_brief_image_prompt import (
    STYLE_ANCHOR_TALKING,
    _talking_roles_from_policy,
    compose_image_prompt_fallback,
    derive_image_prompts,
)

_META = {"script_brief": "a desert observatory race against time",
         "setting": "a remote desert observatory"}
_CAST = [{"char_id": "c01", "name": "LEAD ASTRONOMER",
          "character_description": "40s, Lead Astronomer. Face: oval, "
                                   "hooded eyes, aquiline nose."}]


# --------------------------------------------------------------------------- #
# policy chain
# --------------------------------------------------------------------------- #


def test_talking_roles_policy_parse():
    pol = json.dumps({"talking": {"character_video": True,
                                  "music_visual": False}})
    got = _talking_roles_from_policy(pol)
    assert got == {"character_video": True, "music_visual": False}


def test_talking_roles_policy_malformed_is_empty():
    assert _talking_roles_from_policy("") == {}
    assert _talking_roles_from_policy("{not json") == {}
    assert _talking_roles_from_policy(json.dumps({"talking": "yes"})) == {}


def test_video_director_role_talking_map():
    from nodes.otr_video_director import OTRVideoDirector
    resolved = {
        "announcer_video_model": {"engine_id": "ltx_audio_in"},
        "music_video_model": {"engine_id": "ltx_audio_in"},
        "character_video_model": {"engine_id": "ltx_audio_in"},
    }
    talk = OTRVideoDirector._role_talking(resolved)
    assert set(talk) == {"announcer_visual", "music_visual",
                         "character_video"}
    # default env (dev unet) -> ia2v register -> talking engine
    assert talk["character_video"] is True


def test_video_director_role_talking_false_on_single_pass(monkeypatch):
    from nodes.otr_video_director import OTRVideoDirector
    monkeypatch.setenv(
        "OTR_LTX_AV_UNET",
        r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")
    resolved = {
        "announcer_video_model": {"engine_id": "ltx_audio_in"},
        "music_video_model": {"engine_id": "ltx_audio_in"},
        "character_video_model": {"engine_id": "ltx_audio_in"},
    }
    talk = OTRVideoDirector._role_talking(resolved)
    assert talk["character_video"] is False


# --------------------------------------------------------------------------- #
# S4b: the talking portrait style
# --------------------------------------------------------------------------- #


def _portrait_obj(payload, cid="c01"):
    return next(o for o in payload["objects"]
                if o.get("kind") == "portrait" and o.get("char_id") == cid)


def test_talking_portrait_is_face_forward_and_warm():
    payload, _w = derive_image_prompts(
        _CAST, _META, llm_fn=None,
        talking_roles={"character_video": True})
    p = _portrait_obj(payload)["prompt"]
    assert "face-forward frontal close-up" in p
    assert "mouth clearly visible" in p
    assert "warm dramatic lighting" in p


def test_talking_portrait_skips_era_and_grade_tails():
    from nodes._otr_story_brief_helpers import IMAGE_GRADE_TAIL
    payload, _w = derive_image_prompts(
        _CAST, _META, llm_fn=None,
        talking_roles={"character_video": True})
    p = _portrait_obj(payload)["prompt"]
    assert IMAGE_GRADE_TAIL not in p
    assert "muted color grade" not in p


def test_non_talking_portrait_unchanged():
    from nodes._otr_story_brief_helpers import IMAGE_GRADE_TAIL
    payload, _w = derive_image_prompts(_CAST, _META, llm_fn=None)
    p = _portrait_obj(payload)["prompt"]
    assert "face-forward frontal close-up" not in p
    assert IMAGE_GRADE_TAIL in p           # legacy finish intact


def test_talking_fallback_uses_talking_anchor():
    p = compose_image_prompt_fallback(_META, _CAST[0], "wide", talking=True)
    assert p.endswith(STYLE_ANCHOR_TALKING)


def test_llm_instruction_carries_talking_framing():
    req = mbp._build_char_prompt_request(_CAST[0], _META, "observatory",
                                         "wide", talking=True)
    assert "DIRECTLY at the camera" in req
    assert "never a profile" in req
    assert STYLE_ANCHOR_TALKING in req


# --------------------------------------------------------------------------- #
# S4c: radio-face stills mint by default for talking bookends
# --------------------------------------------------------------------------- #


def _radio_face_ids(payload):
    return {o["object_id"] for o in payload["objects"]
            if str(o.get("object_id", "")).endswith("_radio_face_169")}


def test_radio_face_mints_for_talking_bookends_without_env(monkeypatch):
    monkeypatch.delenv("OTR_LTX_RADIO_FACE", raising=False)
    payload, _w = derive_image_prompts(
        _CAST, _META, llm_fn=None,
        talking_roles={"announcer_visual": True, "music_visual": True})
    ids = _radio_face_ids(payload)
    assert "still_announcer_visual_radio_face_169" in ids
    assert "still_music_visual_radio_face_169" in ids


def test_radio_face_absent_without_talking_or_env(monkeypatch):
    monkeypatch.delenv("OTR_LTX_RADIO_FACE", raising=False)
    payload, _w = derive_image_prompts(_CAST, _META, llm_fn=None)
    assert _radio_face_ids(payload) == set()
