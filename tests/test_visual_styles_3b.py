"""tests/test_visual_styles_3b.py

Multi-modal story schema STAGE 3 CHUNK 3B -- the non-default visual-style
packs, ADDRESSABLE but
DORMANT (no production episode can select them until the 3C widget stamps
meta["visual_style"]; addressability is proven HERE by forced meta only).

Pins (STAGE3_SUBPLAN v5 section 3):
  1. Registry sweep covers all packs; per-pack the v1 EXACT schema holds
     (the loader enforces; these tests pin the authored VALUES + flags).
  2. Forced-meta spot tests: each non-default pack CHANGES the finished
     prompt at every TAIL seam; allow_radio_tails=false drops the broadcast
     tail; empty-string tails append NOTHING (no dangling ", ").
  3. sci_fi_radio stays byte-identical (the default lane is untouched --
     re-pinned here against a forced default meta).
  4. The archival adaptation STRIPPED the lab's v2.0 subject/motion/
     ledger_directives fields (v1 exact-key law) -- pinned structurally.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes import _otr_story_brief_helpers as helpers
from nodes import _otr_visual_styles as vs
from nodes import otr_meta_brief_image_prompt as imgp

_REPO = Path(__file__).resolve().parent.parent
_STYLES_DIR = _REPO / "nodes" / "visual_styles"

_ALL_IDS = ("anime", "archival_documentary", "cartoon", "paper_origami",
            "recur_frac", "sci_fi_radio", "shakespeare_stage_realism",
            "storybook_engraving", "video_art")
_NON_DEFAULT_IDS = tuple(i for i in _ALL_IDS if i != "sci_fi_radio")

# Chunk A1 (2026-07-05): the schema is v2 -- the v1 tail keys PLUS the 11
# look/subject str fields + 4 dict fields (r4 authoritative inventory).
_V2_KEYS = {"style_id", "label", "positive_tail", "image_grade_tail",
            "broadcast_tail", "allow_radio_tails",
            "era_tail", "schema_version",
            "portrait_look", "portrait_look_talking",
            "portrait_instruction_look", "scene_instruction_look",
            "announcer_subject_face", "announcer_subject_ltx_mouth",
            "announcer_subject_object", "radio_object_look", "plate_look",
            "non_character_emblem_fallback", "still_word_title_mood_style",
            "open_subjects", "motion_registers", "still_word_typography",
            "still_word_backdrop",
            # ONE STYLE AUTHORITY (2026-08-17): the pack's own NEGATIVE. Schema-
            # OPTIONAL so frozen embedded packs still validate, but every
            # SHIPPED pack carries it -- pinned here and asserted non-empty in
            # tests/test_visual_style_negative.py.
            "negative_tail"}

_META_BRIEF = {
    "story_brief_terms": {
        "setting": ["a mars listening post", "dust-caked consoles"],
        "lighting": ["harsh sodium light", "deep shadow"],
    },
    "episode_title": "Signal in the Dust",
}


def _meta_for(style_id: str) -> dict:
    m = dict(_META_BRIEF)
    m["visual_style"] = style_id
    return m


@pytest.fixture(autouse=True)
def _fresh_registry():
    vs._clear_caches()
    yield
    vs._clear_caches()


# ---------------------------------------------------------------------------
# 1. Registry + authored values
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_all_registered_in_order(self):
        assert vs.list_style_ids() == _ALL_IDS

    @pytest.mark.parametrize("style_id", _ALL_IDS)
    def test_v2_exact_keys_on_disk(self, style_id):
        raw = json.loads(
            (_STYLES_DIR / f"{style_id}.json").read_text(encoding="utf-8"))
        assert set(raw) == _V2_KEYS, (
            f"{style_id}: v2 schema is exact; the lab's subject/motion/"
            f"ledger_directives fields are NOT schema fields "
            f"(STAGE3_TOTAL_COVERAGE_SUBPLAN section 1a)")
        assert raw["schema_version"] == "v2"

    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    def test_non_default_flags(self, style_id):
        style = vs.resolve_visual_style(style_id)
        assert style.allow_radio_tails is False

    def test_archival_kept_its_three_tails(self):
        s = vs.resolve_visual_style("archival_documentary")
        assert s.positive_tail and s.image_grade_tail and s.broadcast_tail
        assert s.era_tail == "grounded archival documentary aesthetic"

    @pytest.mark.parametrize("style_id", ("anime", "cartoon", "paper_origami"))
    def test_flat_styles_have_empty_grade_broadcast_era(self, style_id):
        s = vs.resolve_visual_style(style_id)
        assert s.image_grade_tail == ""
        assert s.broadcast_tail == ""
        assert s.era_tail == ""


# ---------------------------------------------------------------------------
# 2. Forced-meta tail deltas at every routed seam
# ---------------------------------------------------------------------------
class TestForcedMetaDeltas:
    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    def test_finish_visual_prompt_delta(self, style_id):
        base = "a glowing tube radio on a bench"
        default_out = helpers.finish_visual_prompt(_META_BRIEF, base)
        styled_out = helpers.finish_visual_prompt(_meta_for(style_id), base)
        style = vs.resolve_visual_style(style_id)
        assert styled_out != default_out
        assert styled_out.endswith(style.positive_tail)
        assert helpers.STYLE_TAIL_DEFAULT not in styled_out

    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    def test_compose_still_prompt_scene_beat_delta(self, style_id):
        default_out = helpers.compose_still_prompt(
            _META_BRIEF, kind="scene_beat", role="announcer_visual")
        styled_out = helpers.compose_still_prompt(
            _meta_for(style_id), kind="scene_beat", role="announcer_visual")
        style = vs.resolve_visual_style(style_id)
        assert styled_out != default_out
        assert style.positive_tail in styled_out
        # allow_radio_tails=false on every non-default pack: the broadcast
        # tail (default OR pack) must be absent.
        assert helpers.RADIO_BROADCAST_TAIL not in styled_out
        if style.broadcast_tail:
            assert style.broadcast_tail not in styled_out

    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    def test_portrait_finish_delta(self, style_id):
        cast = [{
            "char_id": "c01", "name": "MARGOT", "gender": "female",
            "appearance": "a wiry engineer in a patched flight suit",
        }]
        default_payload, _w1 = imgp.derive_image_prompts(
            cast, dict(_META_BRIEF), llm_fn=None)
        styled_payload, _w2 = imgp.derive_image_prompts(
            cast, _meta_for(style_id), llm_fn=None)

        def _portrait(payload):
            for obj in payload.get("objects") or []:
                if obj.get("object_id") == "c01":
                    return obj["prompt"]
            raise AssertionError("c01 portrait missing from payload")

        style = vs.resolve_visual_style(style_id)
        d, s = _portrait(default_payload), _portrait(styled_payload)
        assert d != s
        assert helpers.IMAGE_GRADE_TAIL in d
        assert helpers.IMAGE_GRADE_TAIL not in s
        if style.image_grade_tail:
            assert style.image_grade_tail in s

    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    def test_radio_host_delta(self, style_id):
        d = imgp.build_radio_host_prompt(dict(_META_BRIEF), "portrait",
                                         radio_host_style="radio_object")
        s = imgp.build_radio_host_prompt(_meta_for(style_id), "portrait",
                                         radio_host_style="radio_object")
        assert d != s
        assert helpers.IMAGE_GRADE_TAIL not in s

    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    def test_background_plate_and_mesh_delta(self, style_id):
        d_plate = imgp._compose_background_plate_prompt(
            dict(_META_BRIEF), "mars post")
        s_plate = imgp._compose_background_plate_prompt(
            _meta_for(style_id), "mars post")
        assert d_plate != s_plate
        d_mesh = imgp._compose_mesh_fodder_prompt(
            dict(_META_BRIEF), None, {}, "mars post", "music_visual")
        s_mesh = imgp._compose_mesh_fodder_prompt(
            _meta_for(style_id), None, {}, "mars post", "music_visual")
        # The mesh era tail only differs when the era text differs; with a
        # populated brief the brief text wins for both -- force the empty
        # brief to expose the pack era_tail delta.
        assert d_plate != s_plate
        d_mesh_empty = imgp._compose_mesh_fodder_prompt(
            {}, None, {}, "", "music_visual")
        s_mesh_empty = imgp._compose_mesh_fodder_prompt(
            {"visual_style": style_id}, None, {}, "", "music_visual")
        style = vs.resolve_visual_style(style_id)
        assert style.positive_tail.split(",", 1)[0] in s_mesh
        assert style.positive_tail.split(",", 1)[0] in s_mesh_empty
        if style.era_tail != helpers.ERA_TAIL_DEFAULT:
            assert d_mesh_empty != s_mesh_empty
        assert imgp.MESH_FODDER_POS_SCAFFOLD in d_mesh and \
            imgp.MESH_FODDER_POS_SCAFFOLD in s_mesh

    @pytest.mark.parametrize("style_id", _NON_DEFAULT_IDS)
    def test_still_word_delta(self, style_id):
        line = {"text": "We hear it in the static."}
        d = imgp.compose_still_word_prompt(
            {**_META_BRIEF}, "character_video", line)
        s = imgp.compose_still_word_prompt(
            _meta_for(style_id), "character_video", line)
        style = vs.resolve_visual_style(style_id)
        if style.image_grade_tail != helpers.IMAGE_GRADE_TAIL:
            assert d != s
        if not style.image_grade_tail:
            assert helpers.IMAGE_GRADE_TAIL not in s


# ---------------------------------------------------------------------------
# 3. Empty-tail hygiene + default-lane byte identity
# ---------------------------------------------------------------------------
class TestHygiene:
    @pytest.mark.parametrize("style_id", ("anime", "cartoon", "paper_origami"))
    def test_empty_tails_no_dangling_separators(self, style_id):
        out = helpers.compose_still_prompt(
            _meta_for(style_id), kind="scene_beat", role="announcer_visual")
        assert ", ," not in out
        assert not out.endswith(",")
        assert ",  " not in out
        fin = helpers.finish_visual_prompt({"visual_style": style_id},
                                           "a subject")
        assert ", ," not in fin and not fin.endswith(",")

    def test_empty_era_tail_with_empty_brief_vanishes(self):
        # anime: era_tail="" + empty brief -> get_era_tail returns "" and the
        # join drops it entirely (r4 pin: empty-string replacement is EXACT).
        style = vs.resolve_visual_style("anime")
        assert helpers.get_era_tail({}, profile="still", style=style) == ""
        fin = helpers.finish_visual_prompt(
            {"visual_style": "anime"}, "a subject")
        assert fin == "a subject, " + style.positive_tail

    def test_default_lane_still_byte_identical(self):
        forced = helpers.compose_still_prompt(
            _meta_for("sci_fi_radio"), kind="scene_beat",
            role="announcer_visual")
        implicit = helpers.compose_still_prompt(
            dict(_META_BRIEF), kind="scene_beat", role="announcer_visual")
        assert forced == implicit
        assert helpers.STYLE_TAIL_DEFAULT in implicit
        assert helpers.RADIO_BROADCAST_TAIL in implicit
