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
    """A no-person LLM prompt falls back to the template, THEN gets finished;
    the tails never re-trigger the guard."""
    cast = [{"char_id": "c1", "name": "KEEPER",
             "character_description": "60s, weathered face, oilskin coat"}]
    # The stub passes the CONSISTENCY gate (shares "lighthouse" with the
    # setting) but carries no person-evidence -> only the person guard fires.
    out, warns = mbp.derive_image_prompts(
        cast, _OK_META,
        llm_fn=lambda _p: "an empty lighthouse lantern room at dusk")
    entry = mbp.objects_by_id(out)["c1"]
    assert entry["source"] == "template_person_guard"
    assert "weathered face" in entry["prompt"]        # template led
    assert "lantern glow" in entry["prompt"]          # finished after guard
    assert any("depicts no PERSON" in w for w in warns)


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


def test_scene_open_brief_core_and_budget(monkeypatch):
    """render_driver composes the scene prompt from the SHORT brief within
    the LTX budget; the operator env override is verbatim."""
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
    assert "lighthouse" in p                    # the brief core made it in
    assert p.endswith(rd_no_text := "no on-screen text"), p
    assert len(p) <= 242
    # env override: verbatim, unfinished
    monkeypatch.setenv("OTR_LTX_RADIO_PROMPT", "OPERATOR SAYS EXACTLY THIS")
    req2 = rd.build_request_from_shot(shot, ledger)
    assert req2["text_prompt"] == "OPERATOR SAYS EXACTLY THIS"


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
