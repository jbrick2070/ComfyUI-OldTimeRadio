"""tests/test_visual_styles_b.py

Visual-style TOTAL COVERAGE chunk B (STAGE3_TOTAL_COVERAGE_SUBPLAN v5 FINAL)
-- the non-default packs AUTHOR the A1/A2-consumed field set in their own
voice. The r4 header requirement: every authored field CHANGED from the A1
sci-fi defaults. Per-surface forced-meta delta tests + per-surface
NEGATIVE-VOCAB smokes (r1 codex OPT: a styled surface must not carry the
pack's own forbidden terms after composition).

The chunk-C still_word fields stay at the sci-fi defaults (pinned in
test_visual_styles_a1.TestDormantDefaults).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes import _otr_story_brief_helpers as helpers
from nodes import _otr_visual_styles as vs
from nodes import otr_meta_brief_image_prompt as imgp
from nodes._otr_video_engines import render_driver as rd

_REPO = Path(__file__).resolve().parent.parent
_STYLES_DIR = _REPO / "nodes" / "visual_styles"

_NON_DEFAULT_IDS = ("anime", "archival_documentary", "cartoon",
                    "paper_origami", "storybook_engraving")

#: The chunk-B AUTHORED field set (A1/A2-consumed surfaces). still_word_*
#: fields are chunk-C and stay at the sci-fi defaults.
_AUTHORED_STR_FIELDS = (
    "portrait_look", "portrait_look_talking", "portrait_instruction_look",
    "scene_instruction_look", "announcer_subject_face",
    "announcer_subject_ltx_mouth", "announcer_subject_object",
    "radio_object_look", "plate_look", "non_character_emblem_fallback")
_AUTHORED_DICT_FIELDS = ("open_subjects", "motion_registers")

#: A brief whose text carries none of any pack's forbidden vocabulary, so a
#: banned term in a composed output can only come from a missed re-route.
_META_BRIEF = {
    "style": "a quiet village mystery",
    "story_brief_terms": {
        "setting": ["a rain-washed village square", "an old stone well"],
        "lighting": ["soft lamplight"],
        "atmosphere": ["gentle unease"],
    },
    "episode_title": "The Well Remembers",
}
_CHAR = {"char_id": "c01", "name": "MARGOT",
         "appearance": "a wiry engineer in a patched overcoat"}
_LINE = {"beat_id": "b002", "beat_intent": "she listens at the well",
         "traits": "tense", "text": "The stones are humming."}


def _meta_for(style_id: str) -> dict:
    m = dict(_META_BRIEF)
    m["visual_style"] = style_id
    return m


def _raw(style_id: str) -> dict:
    return json.loads(
        (_STYLES_DIR / f"{style_id}.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _fresh_registry():
    vs._clear_caches()
    yield
    vs._clear_caches()


@pytest.fixture()
def _single_pass_recipe(monkeypatch):
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "distilled_native")
    monkeypatch.setenv(
        "OTR_LTX_AV_UNET",
        r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")
    monkeypatch.delenv("OTR_LTX_OPEN_MOTION_KEY", raising=False)
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)


# ---------------------------------------------------------------------------
# 1. Raw-field deltas: every authored field CHANGED from the sci-fi defaults
# ---------------------------------------------------------------------------
class TestAuthoredFieldDeltas:
    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    def test_every_authored_str_field_changed(self, style_id):
        sci = _raw("sci_fi_radio")
        raw = _raw(style_id)
        for name in _AUTHORED_STR_FIELDS:
            assert raw[name] != sci[name], (
                f"{style_id}.{name} still carries the sci-fi default -- "
                f"chunk B must author every field in the pack's voice")

    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    def test_every_authored_dict_value_changed(self, style_id):
        sci = _raw("sci_fi_radio")
        raw = _raw(style_id)
        for name in _AUTHORED_DICT_FIELDS:
            for key, sci_val in sci[name].items():
                assert raw[name][key] != sci_val, (
                    f"{style_id}.{name}[{key!r}] still carries the sci-fi "
                    f"default value")

    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    def test_scene_instruction_look_now_authored(self, style_id):
        # sci_fi ships "" (exempt); the styled packs author a real look.
        assert vs.resolve_visual_style(style_id).scene_instruction_look


# ---------------------------------------------------------------------------
# 2. Forced-meta surface deltas (every A1/A2-routed surface changes)
# ---------------------------------------------------------------------------
class TestSurfaceDeltas:
    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    def test_portrait_anchors_delta(self, style_id):
        s = vs.resolve_visual_style(style_id)
        for talking in (False, True):
            for aspect in ("portrait", "wide"):
                styled = imgp._style_anchor_for_aspect(aspect,
                                                       talking=talking,
                                                       style=s)
                default = imgp._style_anchor_for_aspect(aspect,
                                                        talking=talking)
                assert styled != default
                look = (s.portrait_look_talking if talking
                        else s.portrait_look)
                assert styled.endswith(look)

    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    @pytest.mark.parametrize("arm", ["console_face", "ltx_radio_mouth",
                                     "radio_object"])
    def test_radio_host_arms_delta(self, style_id, arm):
        d = imgp.build_radio_host_prompt(dict(_META_BRIEF), "portrait",
                                         radio_host_style=arm)
        s = imgp.build_radio_host_prompt(_meta_for(style_id), "portrait",
                                         radio_host_style=arm)
        assert d != s

    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    def test_open_subjects_delta(self, style_id):
        s = vs.resolve_visual_style(style_id)
        for role, syn in (("music_visual", True),
                          ("announcer_visual", False),
                          ("music_visual", False)):
            styled = helpers.get_open_subject(role, syn, _META_BRIEF,
                                              style=s)
            default = helpers.get_open_subject(role, syn, _META_BRIEF)
            assert styled != default
            # the brief-driven FORM survives inside the styled template
            assert helpers.radio_form_from_meta(_META_BRIEF) in styled

    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    def test_llm_instruction_delta(self, style_id):
        s = vs.resolve_visual_style(style_id)
        styled = imgp._build_char_prompt_request(_CHAR, _META_BRIEF,
                                                 "village square",
                                                 "portrait", style=s)
        default = imgp._build_char_prompt_request(_CHAR, _META_BRIEF,
                                                  "village square",
                                                  "portrait")
        assert styled != default
        assert s.portrait_instruction_look in styled
        req = imgp._build_char_scene_request(_CHAR, _META_BRIEF,
                                             "village square", _LINE,
                                             style=s)
        assert f"style_look: {s.scene_instruction_look}\n" in req
        assert "style_look:" not in imgp._build_char_scene_request(
            _CHAR, _META_BRIEF, "village square", _LINE)

    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    def test_plate_and_emblem_delta(self, style_id):
        s = vs.resolve_visual_style(style_id)
        styled = imgp._compose_background_plate_prompt(
            _meta_for(style_id), "village square")
        default = imgp._compose_background_plate_prompt(
            dict(_META_BRIEF), "village square")
        assert styled != default
        assert s.plate_look in styled
        subj, src = imgp._mesh_fodder_subject_and_source(
            _meta_for(style_id), None, {"beat_intent": "the well hums"},
            "village square", "character_video", style=s)
        assert subj == s.non_character_emblem_fallback.format(
            base="the well hums")
        assert src == "non_character_emblem_fallback"

    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    def test_motion_registers_delta_through_real_builder(
            self, _single_pass_recipe, style_id):
        def _shot():
            return {"shot_id": "shot_b000_music_open", "beat_id": "b000",
                    "engine_id": "ltx_audio_in", "role": "music_visual",
                    "family": "audio_conditioned_video",
                    "target_frame_count": 25, "source_line_ids": [],
                    "char_id": "", "creative": {}}

        def _led(meta):
            return {"meta": meta,
                    "video": {"video_revision": 1, "shots": []},
                    "lines": [{"line_id": "b000", "char_id": "",
                               "start_s": 0.0, "dur_s": 2.0}],
                    "images": {"images": []}}

        styled = rd.build_request_from_shot(_shot(),
                                            _led(_meta_for(style_id)))
        default = rd.build_request_from_shot(_shot(),
                                             _led(dict(_META_BRIEF)))
        assert styled["text_prompt"] != default["text_prompt"]
        s = vs.resolve_visual_style(style_id)
        assert styled["text_prompt"].startswith(
            s.motion_registers["music_open"][:60])
        assert styled["observability"]["visual_style"] == style_id


# ---------------------------------------------------------------------------
# 3. Negative-vocab smokes (r1 codex OPT): no pack's own forbidden term
#    survives into any styled composed surface.
# ---------------------------------------------------------------------------
class TestNegativeVocabSmokes:
    def _surfaces(self, style_id):
        s = vs.resolve_visual_style(style_id)
        meta = _meta_for(style_id)
        outs = []
        for arm in ("console_face", "ltx_radio_mouth", "radio_object"):
            outs.append(imgp.build_radio_host_prompt(
                meta, "portrait", radio_host_style=arm))
        for talking in (False, True):
            outs.append(imgp.compose_image_prompt_fallback(
                meta, _CHAR, "portrait", talking=talking))
        outs.append(imgp._compose_background_plate_prompt(
            meta, "village square"))
        outs.append(imgp._compose_mesh_fodder_prompt(
            meta, None, {}, "village square", "music_visual"))
        for role, syn in (("music_visual", True),
                          ("announcer_visual", False)):
            outs.append(helpers.get_open_subject(role, syn, meta, style=s))
        outs.extend(s.motion_registers.values())
        outs.append(imgp._build_char_prompt_request(
            _CHAR, meta, "village square", "portrait", style=s))
        outs.append(imgp._build_char_scene_request(
            _CHAR, meta, "village square", _LINE, style=s))
        outs.append(helpers.compose_still_prompt(
            meta, kind="scene_beat", role="announcer_visual"))
        return outs

    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    def test_no_own_forbidden_term_survives(self, style_id):
        s = vs.resolve_visual_style(style_id)
        offenders = []
        for i, text in enumerate(self._surfaces(style_id)):
            low = text.lower()
            for term in s.forbidden_terms:
                if term.lower() in low:
                    offenders.append(f"surface[{i}] carries {term!r}: "
                                     f"{text[:90]}...")
        assert not offenders, "\n".join(offenders)
