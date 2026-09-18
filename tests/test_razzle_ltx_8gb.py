"""razzle_ltx_8gb -- local LTX 0.9.8 sibling of word_razzle.

Offline. No Comfy graph load. Pins registration, the shared prompt owner,
the inherited substrate, and the provision/shortcode wiring.
"""
from __future__ import annotations

from nodes import otr_video_director as vd
from nodes._otr_shared import shortcodes as SC
from nodes._otr_video_engines import eng_ltx_8gb as lx
from nodes._otr_video_engines import eng_razzle_ltx_8gb as rlx
from nodes._otr_video_engines import frame_contract as fc
from nodes._otr_video_engines import razzle_prompt as rp
from nodes._otr_video_engines import registry as vreg


def test_registered_and_capabilities_twin_ltx_8gb():
    assert vreg.is_registered("razzle_ltx_8gb")
    assert vreg.CAPABILITIES["razzle_ltx_8gb"] == vreg.CAPABILITIES["ltx_8gb"]
    assert set(vreg.CAPABILITIES) == set(vreg.all_engine_names())


def test_never_a_default_role():
    assert tuple(rlx.RazzleLtx8gbEngine.default_roles) == ()
    for role in ("announcer_visual", "music_visual", "character_visual"):
        assert vreg.default_engine_for_role(role) != "razzle_ltx_8gb"


def test_in_video_combo():
    parsed = {vd._engine_id_from_pick(c) for c in vd._video_model_combo()}
    assert "razzle_ltx_8gb" in parsed
    assert vd.exact_menu_option_for("razzle_ltx_8gb")


def test_planning_cap_and_shortcode():
    assert "razzle_ltx_8gb" in fc.PLANNING_CAP_ENGINES
    assert SC.code_for("video_lane", "razzle_ltx_8gb") == "rlx8"


def test_positive_leads_with_raised_motion():
    eng = rlx.RazzleLtx8gbEngine()
    prompt = eng._compose_positive({"text_prompt": "a glowing radio on a bench"})
    lowered = prompt.lower()
    assert "full, decisive action" in lowered
    assert "purposeful camera" in lowered
    assert "a glowing radio on a bench" in prompt
    assert rp.damping_hits(prompt) == ()


def test_negative_merges_ltx_recipe_and_razzle_extras():
    eng = rlx.RazzleLtx8gbEngine()
    parent = lx.Ltx8gbEngine()
    got = eng._negative_prompt()
    base = parent._negative_prompt()
    assert "low quality" in got
    assert "static hold" in got
    assert "frozen frame" in got
    assert "no motion" in got
    assert base in got or all(tok in got for tok in ("low quality", "static"))


def test_env_motion_override(monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_RAZZLE_MOTION_PROMPT", "custom motion clause")
    eng = rlx.RazzleLtx8gbEngine()
    prompt = eng._compose_positive({"text_prompt": "beat text"})
    assert prompt.startswith("custom motion clause")
    assert "beat text" in prompt


def test_env_neg_replaces_extras_not_appended(monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_RAZZLE_NEG", "only this extra")
    eng = rlx.RazzleLtx8gbEngine()
    got = eng._negative_prompt()
    assert "only this extra" in got
    assert "low quality" in got
    assert "static hold" not in got


def test_cloud_neg_literal_matches_shared_extra():
    from nodes._otr_video_engines import eng_cloud_video as ecv
    assert ecv._RAZZLE_NEG_DEFAULT == rp.NEG_EXTRA


def test_inherits_ltx_frame_contract_and_still_plan():
    child = rlx.RazzleLtx8gbEngine()
    parent = lx.Ltx8gbEngine()
    assert child.frame_contract == parent.frame_contract
    assert child.still_plan == parent.still_plan
