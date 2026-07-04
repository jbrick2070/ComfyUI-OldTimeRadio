"""Gap-audit F1/F3 regression tests (2026-06-10): the brief prompt finisher
and its three consumer sites. Roundtable-hardened plan:
docs/2026-06-10-brief-downstream-gaps/roundtable/pass01_plan.md.

Pins: era-tail v2 precedence + default fallback; LTX char budget with the
no-on-screen-text clause preserved; finishing ordered guards -> finish ->
hash at BOTH the ShotLock M4 site and the image-prompt site; empty prompts
never invent a subject.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_story_brief_helpers as sbh  # noqa: E402
from nodes import otr_meta_brief_image_prompt as mbp  # noqa: E402


_OK_META = {
    "story_brief_status": "ok",
    "story_brief": ("A lonely lighthouse crew battles a storm that speaks. "
                    "The sea glows green at midnight."),
    "story_brief_terms": {
        "setting": ["a storm-wracked lighthouse"],
        "lighting": ["lantern glow"],
        "atmosphere": ["uneasy"],
    },
}


def test_era_tail_default_when_brief_absent():
    assert sbh.get_era_tail({}) == sbh.ERA_TAIL_DEFAULT
    assert sbh.get_era_tail({"story_brief_status": "failed"}) \
        == sbh.ERA_TAIL_DEFAULT


def test_era_tail_uses_v1_lighting_when_ok():
    tail = sbh.get_era_tail(_OK_META)
    assert "lantern glow" in tail and "uneasy" in tail


def test_finish_appends_era_and_style_tails():
    out = sbh.finish_visual_prompt(_OK_META, "a keeper at the rail")
    assert out.startswith("a keeper at the rail")
    assert "lantern glow" in out
    assert sbh.STYLE_TAIL_DEFAULT.split(",")[1].strip() in out  # 35mm film look


def test_finish_empty_prompt_stays_empty():
    assert sbh.finish_visual_prompt(_OK_META, "") == ""
    assert sbh.finish_visual_prompt(_OK_META, "   ") == ""


def test_finish_max_chars_preserves_no_text_clause():
    long_core = ("a sweeping cinematic establishing shot of the lighthouse "
                 "rocks under towering storm clouds with rain lashing the "
                 "lantern room, " + sbh.NO_TEXT_CLAUSE)
    out = sbh.finish_visual_prompt(_OK_META, long_core, max_chars=240,
                                   style_tail=False)
    assert len(out) <= 240 + 2
    assert out.endswith(sbh.NO_TEXT_CLAUSE)
    assert " " in out and not out.rstrip().endswith(",")


def test_finish_style_tail_toggle():
    out = sbh.finish_visual_prompt(_OK_META, "a keeper", style_tail=False)
    assert "35mm" not in out


def test_image_prompt_hash_matches_finished_prompt():
    """The stamped hash must be computed AFTER finishing (DS#1/GPT#6)."""
    cast = [{"char_id": "c1", "name": "KEEPER",
             "character_description": "60s, weathered face, oilskin coat"}]
    out, _w = mbp.derive_image_prompts(cast, _OK_META, llm_fn=None)
    entry = mbp.objects_by_id(out)["c1"]
    assert "lantern glow" in entry["prompt"]          # finished
    assert entry["prompt_hash"] == mbp._content_hash(entry["prompt"])


def test_image_person_guard_then_finish_no_retrigger():
    """NO-FALLBACK rip (2026-07-03): a no-person LLM prompt now FAILS LOUD at the
    person guard (a faceless init starves the audio-driven-face engine) -- no
    silent template swap. The llm_fn=None template lane still finishes cleanly."""
    cast = [{"char_id": "c1", "name": "KEEPER",
             "character_description": "60s, weathered face, oilskin coat"}]
    # Passes the CONSISTENCY gate (shares "lighthouse" with the setting) but
    # carries no person-evidence -> the person guard now RAISES.
    with pytest.raises(RuntimeError, match="no-fallback rip"):
        mbp.derive_image_prompts(
            cast, _OK_META,
            llm_fn=lambda _p: "an empty lighthouse lantern room at dusk")
    # The template LANE (llm_fn=None) leads with the appearance + finishes; the
    # tails never re-trigger the person guard.
    out, _warns = mbp.derive_image_prompts(cast, _OK_META, llm_fn=None)
    entry = mbp.objects_by_id(out)["c1"]
    assert entry["source"].startswith("template")
    assert "weathered face" in entry["prompt"]        # template led
    assert "lantern glow" in entry["prompt"]          # finished after guard


def test_shotlock_m4_prompt_hash_matches_finished_prompt():
    from nodes import otr_shot_lock as sl
    ledger = {
        "meta": _OK_META,
        "cast": [{"char_id": "c1", "name": "KEEPER",
                  "character_description": "60s, weathered face"}],
        "lines": [{"line_id": "b001", "char_id": "c1",
                   "speaker_role": "char_voice", "text": "The storm speaks.",
                   "dur_s": 4.0}],
    }
    beats = sl.extract_beats(ledger)
    creative, _w = sl.derive_creative_directives(beats, _OK_META, ledger)
    cre = creative["b001"]
    assert "lantern glow" in cre["text_prompt"]       # finished
    assert cre["prompt_hash"] == sl._content_hash(cre["text_prompt"])


def test_scene_open_motion_and_budget(monkeypatch):
    """An OPEN (announcer/music) role renders the 6/5 MOTION template within
    budget (the i2v still carries the look); a NON-open text role still composes
    from the SHORT brief within budget; the operator env override is verbatim."""
    from nodes._otr_video_engines import render_driver as rd
    monkeypatch.delenv("OTR_LTX_RADIO_PROMPT", raising=False)
    ledger = {
        "meta": _OK_META,
        "lines": [{"line_id": "b001", "char_id": "announcer",
                   "speaker_role": "announcer", "text": "Tonight...",
                   "start_s": 0.0, "dur_s": 5.0}],
    }
    shot = {"shot_id": "shot_b001", "source_line_ids": ["b001"],
            "role": "announcer_visual", "engine_id": "ltx_video",
            "group_id": "grp_announcer_visual", "target_frame_count": 50,
            "creative": {}}
    req = rd.build_request_from_shot(shot, ledger)
    p = req["text_prompt"]
    assert p.startswith("Continuous shot, same console")   # the motion template
    assert "Tuning dial needle sweeps" in p                # motion verb present
    assert len(p) <= 240
    # NON-open text-engine role still composes from the brief within budget,
    # ending on the camera/no-text clauses.
    broll = dict(shot, shot_id="shot_b002", role="scene_broll",
                 group_id="grp_scene_broll")
    p3 = rd.build_request_from_shot(broll, ledger)["text_prompt"]
    assert p3.endswith("no on-screen text"), p3
    assert len(p3) <= 242
    # env override: verbatim, unfinished
    monkeypatch.setenv("OTR_LTX_RADIO_PROMPT", "OPERATOR SAYS EXACTLY THIS")
    req2 = rd.build_request_from_shot(shot, ledger)
    assert req2["text_prompt"] == "OPERATOR SAYS EXACTLY THIS"


# --------------------------------------------------------------------------- #
# LTX-I2V ticket Part A (2026-06-11): scene prompts ride the STILL era tail
# --------------------------------------------------------------------------- #
def test_finish_era_profile_default_is_full_byte_identical():
    full = sbh.finish_visual_prompt(_OK_META, "a keeper at the rail")
    explicit = sbh.finish_visual_prompt(_OK_META, "a keeper at the rail",
                                        era_profile="full")
    assert full == explicit


def test_finish_era_profile_still_uses_trimmed_tail():
    out = sbh.finish_visual_prompt(_OK_META, "a keeper at the rail",
                                   style_tail=False, era_profile="still")
    assert out.startswith("a keeper at the rail")
    tail = sbh.get_era_tail(_OK_META, profile="still")
    assert out.endswith(tail)


def test_ltx_scene_prompt_uses_still_profile_tail(monkeypatch):
    """The driver's LTX scene composer must share the stills' palette diet:
    the composed prompt's era tail is the STILL profile, not the full one."""
    from nodes._otr_video_engines import render_driver as rd
    monkeypatch.delenv("OTR_LTX_RADIO_PROMPT", raising=False)
    # A palette-heavy brief where full vs still tails visibly diverge.
    meta = {
        "story_brief_status": "ok",
        "story_brief": "A dim relay station on the Martian flats at dusk.",
        "story_brief_terms": {
            "setting": ["a dim relay station"],
            "lighting": ["red dust haze", "low tungsten"],
            "atmosphere": ["oppressive"],
        },
        "visual_palette": ["rust red", "burnt orange", "deep crimson"],
        "atmosphere_line": "rust-red dust hangs over the flats",
    }
    still_tail = sbh.get_era_tail(meta, profile="still")
    ledger = {"meta": meta, "lines": []}
    shot = {"shot_id": "shot_b009", "source_line_ids": [],
            "role": "scene_broll", "engine_id": "ltx_video",
            "group_id": "grp_scene_broll", "target_frame_count": 50,
            "creative": {}}
    req = rd.build_request_from_shot(shot, ledger)
    p = req["text_prompt"]
    # The still tail leads the tail section (the 240-char cap may trim its
    # end, so assert the still tail's HEAD made it in right after the clauses
    # and the full top-3 palette did NOT ride along whole).
    assert still_tail.split(",")[0] in p
    full_tail = sbh.get_era_tail(meta, profile="full")
    if full_tail != still_tail:
        assert full_tail not in p


# --------------------------------------------------------------------------- #
# HuMo-seam ticket Part C (2026-06-11): M4 -> HuMo prompt + the gear scrub
# --------------------------------------------------------------------------- #
def _face_shot(role, creative=None, shot_id="shot_b003"):
    return {"shot_id": shot_id, "source_line_ids": ["b003"], "role": role,
            "engine_id": "humo", "group_id": f"grp_{role}",
            "target_frame_count": 50, "creative": creative or {}}


_FACE_LEDGER = {"meta": _OK_META, "lines": [
    {"line_id": "b003", "char_id": "c1", "speaker_role": "char_voice",
     "text": "The storm speaks.", "start_s": 1.0, "dur_s": 3.0}]}


def test_humo_character_beat_consumes_m4_prompt_with_gear_scrub():
    from nodes._otr_video_engines import render_driver as rd
    creative = {"text_prompt": ("KEEPER, 60s weathered face, speaking into a "
                                "studio microphone, lantern glow"),
                "source": "llm"}
    req = rd.build_request_from_shot(
        _face_shot("character_video", creative), _FACE_LEDGER)
    p = req["text_prompt"]
    assert req["observability"]["prompt_source"] == "m4"
    assert "microphone" not in p.lower() and "studio" not in p.lower()
    assert "weathered face" in p          # the creative content survived
    assert "no microphone" not in p.lower()   # NEVER negations


def test_humo_announcer_beat_keeps_radio_styling(monkeypatch):
    # NOTE 2026-06-30: _enforce_radio_is_host redirects this fixture's
    # engine_id="humo" pick to "ltx_audio_in" before any prompt logic runs (an
    # announcer_visual beat can no longer stay on a HuMo-family engine -- "the
    # radio is the host"). The M4-verbatim-prompt handling below is IDENTICAL
    # on either engine for this exempt (announcer, gear-not-scrubbed) case, so
    # the assertions are unchanged; only the reason they hold has shifted.
    # NOTE 2026-07-02 (lips-dont-talk): under ia2v_canonical the TALKING
    # register now OUTRANKS M4 on announcer bookends -- this M4-verbatim
    # contract belongs to the SINGLE-PASS recipes, so pin one (the ia2v
    # behavior is locked in test_ltx_av_ia2v_canonical.py).
    from nodes._otr_video_engines import render_driver as rd
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "distilled_native")
    monkeypatch.setenv(
        "OTR_LTX_AV_UNET",
        r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")
    creative = {"text_prompt": ("vintage radio announcer at a chrome "
                                "microphone, ON AIR sign"), "source": "llm"}
    shot = _face_shot("announcer_visual", creative)
    req = rd.build_request_from_shot(shot, _FACE_LEDGER)
    assert shot["engine_id"] == "ltx_audio_in"           # redirected off HuMo
    assert "microphone" in req["text_prompt"].lower()   # exempt BY DESIGN
    assert req["observability"]["prompt_source"] == "m4"


def test_humo_character_beat_missing_creative_falls_back_loud_and_gear_free():
    from nodes._otr_video_engines import render_driver as rd
    req = rd.build_request_from_shot(
        _face_shot("character_video", {}), _FACE_LEDGER)
    p = req["text_prompt"]
    assert req["observability"]["prompt_source"] == "default_scrubbed"
    assert p == rd._CHAR_FACE_FALLBACK_PROMPT
    assert not rd._GEAR_WORDS_RD.search(p)
    assert "no " not in p.lower()                      # no negations


def test_announcer_beat_missing_creative_redirects_off_humo_to_radio_console(
        monkeypatch):
    """2026-06-30 HuMo-improve plan (reversing Route-A 2026-06-28, "the radio
    is the host"): render_driver._enforce_radio_is_host now redirects an
    announcer_visual pick of a HuMo-family engine to ltx_audio_in BEFORE any
    prompt logic runs (this fixture's ``_face_shot`` still requests
    engine_id="humo" -- the point is that the driver no longer honors it for
    this role). A beat with no M4 creative prompt therefore no longer falls to
    the old hardcoded "1940s radio studio" HuMo default: it flows through the
    pre-existing role-driven LTX motion-prompt path (_ltx_motion_role_key),
    producing an animated RADIO-CONSOLE prompt -- never a human face.

    2026-07-02 (lips-dont-talk): the console-motion contract belongs to the
    SINGLE-PASS recipes; under ia2v_canonical this announcer bookend swaps to
    the TALKING register (locked in test_ltx_av_ia2v_canonical.py)."""
    from nodes._otr_video_engines import render_driver as rd
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "distilled_native")
    monkeypatch.setenv(
        "OTR_LTX_AV_UNET",
        r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")
    shot = _face_shot("announcer_visual", {})
    req = rd.build_request_from_shot(shot, _FACE_LEDGER)
    assert shot["engine_id"] == "ltx_audio_in"              # redirected off HuMo
    assert req["observability"]["prompt_source"] == "motion_role"
    assert "console" in req["text_prompt"].lower()


def test_gear_scrub_regex_lockstep_with_image_prompt_module():
    """The driver's local mirror must stay in LOCKSTEP with the image-prompt
    module's _GEAR_WORDS (the shared-scrub contract of the ticket)."""
    from nodes._otr_video_engines import render_driver as rd
    assert rd._GEAR_WORDS_RD.pattern == mbp._GEAR_WORDS.pattern
    sample = "a person speaking, studio microphone, on-air sign, lantern glow"
    assert rd._scrub_gear(sample) == mbp._scrub_gear_words(sample)


def test_scene_broll_on_ltx_no_longer_generic(monkeypatch):
    """GPT#8: a no-creative scene_broll shot on ltx_video gets the brief
    core, not the generic '1940s radio studio' default."""
    from nodes._otr_video_engines import render_driver as rd
    monkeypatch.delenv("OTR_LTX_RADIO_PROMPT", raising=False)
    ledger = {"meta": _OK_META, "lines": []}
    shot = {"shot_id": "shot_b009", "source_line_ids": [],
            "role": "scene_broll", "engine_id": "ltx_video",
            "group_id": "grp_scene_broll", "target_frame_count": 50,
            "creative": {}}
    req = rd.build_request_from_shot(shot, ledger)
    assert "1940s radio studio" not in req["text_prompt"]
    assert "lighthouse" in req["text_prompt"]
    # b-roll is NOT an open: no vintage-radio set-dressing clause
    assert "vintage radio set" not in req["text_prompt"]
