# -*- coding: utf-8 -*-
"""GUARD tests for the 2026-07-01 rip-sfx-broll cleanbreak.

Pins the post-rip role model so the dead subsystem cannot silently creep back:
no "sfx" speaker-role, exactly three video roles, LOUD rejection of old sfx
ledgers at every converted fallback site, the KEPT widgets still present, and
the canonical workflow pinned to a registered character engine.

Contract: kibitz-runs/2026-07-01-rip-sfx-broll/r2/final.md (+ r3/r4 folds).
"""
from __future__ import annotations

import json
import os

import pytest

from nodes import _otr_speaker_role as SR
from nodes._otr_shared import role_compat as RC
from nodes._otr_shared import role_slots as RS
from nodes._otr_shared import slot_matrix as SM

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WF = os.path.join(_REPO, "workflows", "otr_scifi_16gb_full.json")


# ---------------------------------------------------------------------------
# (a) speaker-role model: 5 roles, no sfx
# ---------------------------------------------------------------------------

def test_valid_speaker_roles_has_no_sfx_and_exactly_five():
    assert "sfx" not in SR.VALID_SPEAKER_ROLES
    assert SR.VALID_SPEAKER_ROLES == (
        "character", "announcer", "music_open", "music_close", "music_inter",
    )
    assert not hasattr(SR, "SPEAKER_ROLE_SFX")


# ---------------------------------------------------------------------------
# (b) video-role model: exactly the 3 roles, matching input map
# ---------------------------------------------------------------------------

def test_role_enum_is_exactly_the_three_roles():
    assert {r.value for r in RC.Role} == {
        "announcer_visual", "music_visual", "character_video",
    }
    assert set(RC.ROLE_AVAILABLE_INPUTS) == {r.value for r in RC.Role}
    assert set(RC.ROLES) == {r.value for r in RC.Role}


def test_slot_maps_have_no_dead_roles_or_slots():
    assert set(RS.ROLE_TO_VIDEO_SLOT) == {
        "announcer_visual", "music_visual", "character_video",
    }
    assert "scene_broll_video_model" not in RS.VIDEO_SLOT_ROLES
    assert "background_abstract_video_model" not in RS.VIDEO_SLOT_ROLES
    # NO legacy other_beats video slot (2026-07-03 consolidation)
    assert "other_beats_video_model" not in RS.VIDEO_SLOT_ROLES
    assert set(SM.ROLE_TO_PROFILE_KEY) == set(SM.ALL_ROLES)
    assert "scene_broll" not in SM.ALL_ROLES
    assert "background_abstract" not in SM.ALL_ROLES


# ---------------------------------------------------------------------------
# (c) old sfx ledgers / unknown roles fail LOUD at every converted site
# ---------------------------------------------------------------------------

def test_resolve_speaker_role_raises_on_sfx_and_unknown():
    with pytest.raises(ValueError):
        SR.resolve_speaker_role({"line_id": "l01", "speaker_role": "sfx"})
    with pytest.raises(ValueError):
        SR.resolve_speaker_role({"line_id": "l01", "speaker_role": "narrator"})
    with pytest.raises(ValueError):
        SR.resolve_speaker_role({"line_id": "l01"})          # missing field
    with pytest.raises(ValueError):
        SR.resolve_speaker_role(None)                        # non-mapping
    # the happy path still normalizes
    assert SR.resolve_speaker_role(
        {"speaker_role": " Character "}) == "character"


def test_stamp_default_role_raises_instead_of_backfilling():
    with pytest.raises(ValueError):
        SR.stamp_default_role({"line_id": "l01"})            # missing
    with pytest.raises(ValueError):
        SR.stamp_default_role({"speaker_role": "sfx"})       # dead role
    with pytest.raises(TypeError):
        SR.stamp_default_role(["not", "a", "dict"])          # wrong type
    row = {"speaker_role": "ANNOUNCER"}
    assert SR.stamp_default_role(row)["speaker_role"] == "announcer"


def test_slot_for_role_and_engine_id_raise_on_dead_roles():
    for dead in ("scene_broll", "background_abstract", "sfx", "bogus"):
        with pytest.raises(ValueError):
            RS.slot_for_role(dead)
        with pytest.raises(ValueError):
            RS.engine_id_for_role({}, dead)
    # NO FALLBACK (2026-07-03): the legacy other_beats migration lane is gone --
    # an empty character slot resolves to "".
    assert RS.engine_id_for_role(
        {"other_beats_video_model": "still_flat"}, "character_video") == ""


def test_shot_lock_video_role_raises_on_sfx():
    from nodes.otr_shot_lock import _video_role_for_line, SPEAKER_TO_VIDEO_ROLE
    assert "sfx" not in SPEAKER_TO_VIDEO_ROLE
    with pytest.raises(ValueError):
        _video_role_for_line({"line_id": "l02", "speaker_role": "sfx"})
    assert _video_role_for_line(
        {"speaker_role": "character"}) == "character_video"


def test_scene_still_targets_raise_on_sfx_row():
    from nodes.otr_meta_brief_image_prompt import derive_scene_still_targets
    lines = [
        {"line_id": "l00", "speaker_role": "announcer", "text": "hi"},
        {"line_id": "x00", "speaker_role": "sfx", "text": "[static]"},
    ]
    with pytest.raises(ValueError):
        derive_scene_still_targets(lines, fps=25)


def test_ledger_freeze_rejects_sfx_row_as_error():
    from nodes import _otr_ledger_freeze as LF
    assert "sfx" not in LF.ALLOWED_SPEAKER_ROLES
    assert not hasattr(LF, "SFX_DUR_MIN_S")
    led = {
        "schema_version": LF.EXPECTED_SCHEMA_VERSION,
        "cast": [], "lines": [
            {"line_id": "l00", "beat_id": "l00", "speaker_role": "sfx",
             "char_id": "sfx", "text": "boom"},
        ],
        "beats": [], "scenes": [], "shots": [], "music": [], "clips": [],
        "meta": {},
    }
    report = LF.run_gap_audit(led, label="guard")
    assert any("speaker_role" in e and "sfx" in e for e in report.errors), \
        report.errors


# ---------------------------------------------------------------------------
# (d) KEPT surfaces: other_beats image slot + registry eligibility + wf pin
# ---------------------------------------------------------------------------

def test_video_director_keeps_other_beats_image_and_drops_dead_widgets():
    from nodes.otr_video_director import OTRVideoDirector
    it = OTRVideoDirector.INPUT_TYPES()
    req, opt = it["required"], it.get("optional", {})
    assert "other_beats_image_model" in req                  # decision-locked KEEP (image side)
    # 2026-07-03: other_beats_video retired; character is a first-class video slot
    assert "other_beats_video_model" not in req and "other_beats_video_model" not in opt
    assert "character_video_model" in req
    for dead in ("other_beats_clip_mode", "other_beats_n",
                 "scene_broll_video_model", "background_abstract_video_model"):
        assert dead not in req and dead not in opt, dead


def test_registry_offers_engines_for_all_three_roles():
    from nodes._otr_video_engines import registry as vreg
    for role in ("announcer_visual", "music_visual", "character_video"):
        eligible = vreg.engines_for_role(role)
        assert eligible, f"no eligible engines for {role}"


def test_workflow_json_node87_matches_live_widget_model():
    with open(_WF, "r", encoding="utf-8") as f:
        d = json.load(f)
    n87 = next(n for n in d["nodes"] if n["id"] == 87)
    n3 = next(n for n in d["nodes"] if n["id"] == 3)
    # 2026-07-03 clean-UI: allow_auto_fallback (15->14), episode_duration_target
    # (14->13), then the other_beats_video consolidation (13->12; character
    # promoted to video slot 3 = widgets index 2).
    assert len(n87["widgets_values"]) == 12, n87["widgets_values"]
    names87 = {i.get("name") for i in n87["inputs"]}
    assert not ({"other_beats_clip_mode", "other_beats_n",
                 "scene_broll_video_model",
                 "background_abstract_video_model"} & names87)
    assert "allow_auto_fallback" not in names87
    assert "episode_duration_target" not in names87
    assert "other_beats_video_model" not in names87
    # character_video_model is now video slot 3 (widgets index 2); the pinned
    # engine must be a registered engine (no inherit sentinel any more).
    pin = n87["widgets_values"][2]
    from nodes._otr_video_engines import registry as vreg
    assert vreg.is_registered(pin), pin
    # node 3: sfx overlay gone, script_json link landed on slot 2
    names3 = [i.get("name") for i in n3["inputs"]]
    assert "sfx_audio_clips" not in names3
    assert "sfx_offset_ms" not in names3
    assert len(n3["widgets_values"]) == 6
    lk2 = next(l for l in d["links"] if l[0] == 2)
    assert lk2[3] == 3 and lk2[4] == names3.index("script_json")
