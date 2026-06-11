"""Look-QA round 5 (2026-06-10): the acceptance-eyeball fixes, CPU-only.

Covers F1 (LTX frame cap), F2 (per-beat brief prompts + prompt observability +
diversity status), F3 (talking-head person anchor, guard-before-anchor), F4
(writer self-vocative attribution repair -- the b004 shape), F5 (char_id join
hardening + manifest positioned-mode fallback). Plan + panel record:
docs/2026-06-10-look-qa-round5/.
"""
import hashlib
import logging
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "nodes"))

from nodes._otr_video_engines import eng_ltx_video as _ltx          # noqa: E402
from nodes._otr_video_engines import render_driver as _rd           # noqa: E402
from nodes import otr_shot_lock as _sl                               # noqa: E402


# --------------------------------------------------------------------------- #
# F1 -- the LTX frame cap (pure helper)
# --------------------------------------------------------------------------- #

class TestLtxFrameCap:
    def test_over_cap_ask_caps_to_default(self, monkeypatch):
        monkeypatch.delenv("OTR_LTX_MAX_FRAMES", raising=False)
        # the 2026-06-10 mud open: a 238-frame ask for the 9.5s gap. Default
        # cap = 169, the live-DECODE-PROVEN length on this wrapper stack (121
        # trips the wrapper VAEDecode tensor mismatch -- ticking_lab catch).
        assert _ltx._ltx_frame_length(238, 24) == 169
        assert _ltx._LTX_MAX_FRAMES_DEFAULT == 169

    def test_under_cap_ask_snaps_8n1(self, monkeypatch):
        monkeypatch.delenv("OTR_LTX_MAX_FRAMES", raising=False)
        assert _ltx._ltx_frame_length(50, 24) == 49      # 8n+1 snap below cap

    def test_cap_env_override_respected(self, monkeypatch):
        monkeypatch.setenv("OTR_LTX_MAX_FRAMES", "57")
        assert _ltx._ltx_frame_length(238, 24) == 57     # 57 = 8*7+1

    def test_non_8n1_cap_snaps_down_below_cap(self, monkeypatch):
        monkeypatch.setenv("OTR_LTX_MAX_FRAMES", "120")
        out = _ltx._ltx_frame_length(238, 24)
        assert out == 113 and out <= 120 and (out - 1) % 8 == 0

    def test_invalid_env_falls_to_default(self, monkeypatch):
        monkeypatch.setenv("OTR_LTX_MAX_FRAMES", "not-a-number")
        assert _ltx._ltx_frame_length(238, 24) == 169

    def test_below_floor_env_clamps(self, monkeypatch):
        monkeypatch.setenv("OTR_LTX_MAX_FRAMES", "2")
        out = _ltx._ltx_frame_length(238, 24)
        assert out == _ltx._LTX_MIN_FRAMES

    def test_zero_ask_uses_fallback(self, monkeypatch):
        monkeypatch.delenv("OTR_LTX_MAX_FRAMES", raising=False)
        assert _ltx._ltx_frame_length(0, 24) == 17       # 24 -> snap 17


# --------------------------------------------------------------------------- #
# F2 -- per-beat scene prompts + observability
# --------------------------------------------------------------------------- #

def _scene_ledger(lines_extra=None):
    """A minimal planned ledger with 3 text-engine shots (synthetic open +
    announcer + outro) on distinct beats."""
    lines = [
        {"line_id": "b001", "char_id": "announcer",
         "text": "In the heart of the city.", "start_s": 9.5, "dur_s": 7.0,
         "beat_intent": "revelation", "arc_phase": "setup"},
        {"line_id": "b005", "char_id": "announcer",
         "text": "Tune in next week.", "start_s": 28.0, "dur_s": 5.7,
         "beat_intent": "resolution", "arc_phase": "resolution"},
    ]
    for ln in (lines_extra or []):
        lines.append(ln)
    return {
        "meta": {"story_brief": "An innovator unveils a machine.",
                 "story_brief_terms": {"setting": ["foundry", "night city"],
                                       "lighting": ["warm"],
                                       "atmosphere": ["wonder"]}},
        "cast": [{"char_id": "c01", "name": "ANNOUNCER"}],
        "lines": lines,
        "images": {"images": []},
    }


def _shot(shot_id, role, engine="ltx_video", source_line_ids=None, **kw):
    base = {"shot_id": shot_id, "role": role, "engine_id": engine,
            "group_id": f"grp_{role}",
            "source_line_ids": (source_line_ids
                                if source_line_ids is not None
                                else [shot_id.replace("shot_", "")]),
            "target_frame_count": 49, "cache_keys": {}}
    base.update(kw)
    return base


class TestPerBeatScenePrompts:
    def test_three_ltx_prompts_differ(self, monkeypatch):
        monkeypatch.delenv("OTR_LTX_RADIO_PROMPT", raising=False)
        led = _scene_ledger()
        shots = [
            _shot("shot_b000_music_open", "music_visual",
                  source_line_ids=[], start_s=0.0, dur_s=9.5),
            _shot("shot_b001", "announcer_visual"),
            _shot("shot_b005", "announcer_visual"),
        ]
        prompts = [_rd.build_request_from_shot(s, led)["text_prompt"]
                   for s in shots]
        assert len(set(prompts)) == 3, prompts
        assert all(len(p) <= 240 for p in prompts)

    def test_synthetic_open_detected_by_structure_not_role(self, monkeypatch):
        monkeypatch.delenv("OTR_LTX_RADIO_PROMPT", raising=False)
        led = _scene_ledger()
        # role says announcer_visual, but EMPTY source_line_ids = synthetic
        s = _shot("shot_b000_music_open", "announcer_visual",
                  source_line_ids=[], start_s=0.0, dur_s=9.5)
        p = _rd.build_request_from_shot(s, led)["text_prompt"]
        assert "opening establishing shot" in p

    def test_bright_tokens_survive_the_finisher(self, monkeypatch):
        monkeypatch.delenv("OTR_LTX_RADIO_PROMPT", raising=False)
        led = _scene_ledger()
        p = _rd.build_request_from_shot(
            _shot("shot_b001", "announcer_visual"), led)["text_prompt"]
        assert "glowing warmly" in p and "dials" in p

    def test_beat_clauses_intent_and_arc(self):
        line = {"beat_intent": "revelation", "arc_phase": "rising"}
        out = _rd._beat_clauses(line, "shot_x")
        assert out == ["a moment of revelation", "rising stakes"]

    def test_beat_clauses_unmapped_intent_loose_fallback(self):
        out = _rd._beat_clauses({"beat_intent": "brooding"}, "shot_x")
        assert out == ["a beat of brooding"]

    def test_beat_clauses_free_text_intent_bounded(self):
        # live catch: the writer emits SENTENCES as beat_intent
        out = _rd._beat_clauses(
            {"beat_intent": "Open the episode and orient the listener "
                            "to the stakes."}, "shot_x")
        assert out == ["a beat of open the episode and orient the"]

    def test_beat_clauses_absent_fields_skip(self):
        assert _rd._beat_clauses({}, "shot_x") == []
        assert _rd._beat_clauses(None, "shot_x") == []

    def test_env_override_stamped_as_env_source(self, monkeypatch):
        monkeypatch.setenv("OTR_LTX_RADIO_PROMPT", "the operator prompt")
        led = _scene_ledger()
        req = _rd.build_request_from_shot(
            _shot("shot_b001", "announcer_visual"), led)
        assert req["text_prompt"] == "the operator prompt"
        assert req["_prompt_source"] == "env"

    def test_brief_prompt_stamps_meta(self, monkeypatch):
        monkeypatch.delenv("OTR_LTX_RADIO_PROMPT", raising=False)
        led = _scene_ledger()
        req = _rd.build_request_from_shot(
            _shot("shot_b001", "announcer_visual"), led)
        assert req["_prompt_source"] == "brief+beat"
        assert len(req["_prompt_sha8"]) == 8
        assert req["_prompt_chars"] == len(req["text_prompt"])

    def test_m4_creative_prompt_stamped_m4(self, monkeypatch):
        monkeypatch.delenv("OTR_LTX_RADIO_PROMPT", raising=False)
        led = _scene_ledger()
        s = _shot("shot_b001", "announcer_visual",
                  creative={"text_prompt": "a writer prompt", "source": "llm"})
        req = _rd.build_request_from_shot(s, led)
        assert req["text_prompt"] == "a writer prompt"
        assert req["_prompt_source"] == "m4"
        assert req["_prompt_subsource"] == "llm"


class TestPromptDiversityStatus:
    def test_all_equal_fails(self):
        trace = [{"prompt_source": "brief+beat", "prompt_sha8": "aaaa1111"},
                 {"prompt_source": "brief+beat", "prompt_sha8": "aaaa1111"}]
        st = _rd.ltx_prompt_diversity_status(trace)
        assert st["n"] == 2 and st["distinct"] == 1 and not st["ok"]

    def test_distinct_passes(self):
        trace = [{"prompt_source": "brief+beat", "prompt_sha8": "aaaa1111"},
                 {"prompt_source": "brief+beat", "prompt_sha8": "bbbb2222"}]
        assert _rd.ltx_prompt_diversity_status(trace)["ok"]

    def test_single_prompt_vacuously_ok(self):
        trace = [{"prompt_source": "brief+beat", "prompt_sha8": "aaaa1111"}]
        st = _rd.ltx_prompt_diversity_status(trace)
        assert st["n"] == 1 and st["ok"]

    def test_env_source_exempt(self):
        trace = [{"prompt_source": "env", "prompt_sha8": "aaaa1111"},
                 {"prompt_source": "env", "prompt_sha8": "aaaa1111"}]
        st = _rd.ltx_prompt_diversity_status(trace)
        assert st["n"] == 0 and st["ok"]

    def test_empty_and_none_ok(self):
        assert _rd.ltx_prompt_diversity_status([])["ok"]
        assert _rd.ltx_prompt_diversity_status(None)["ok"]


# --------------------------------------------------------------------------- #
# F3 -- talking-head person anchor (guard BEFORE anchor)
# --------------------------------------------------------------------------- #

class TestPersonAnchor:
    def test_person_anchor_ok_requires_face_token_in_head(self):
        app = "a stocky engineer in a grey work shirt"
        good = "the stocky engineer, face visible, gripping the rail"
        bad = "fingers dancing across the console, sparks flying"
        assert _sl._person_anchor_ok(good, app)
        assert not _sl._person_anchor_ok(bad, app)

    def test_subject_anchor_leads_with_face_tokens(self):
        a = _sl._subject_anchor("a stocky engineer in a grey shirt")
        assert a.startswith("face visible, speaking to camera")
        assert "stocky engineer" in a

    def test_subject_anchor_truncates_long_appearance(self):
        a = _sl._subject_anchor("x" * 400)
        assert len(a) < 200

    def test_object_only_llm_prompt_falls_to_anchored_template(self):
        beats = [{"beat_id": "b1", "role": _sl.Role.CHARACTER_VIDEO.value,
                  "char_id": "c02", "text": "It's alive.",
                  "samples": None, "sample_rate": None, "dur_s": 2.0}]
        led = {"cast": [{"char_id": "c02", "name": "HAYES VANCE",
                         "portrait_prompt": "a stocky engineer in grey"}],
               "lines": []}
        meta = {"story_brief_terms": {"setting": ["foundry"]}}
        # the b002 shape: an LLM prompt describing PROPS without the person
        llm = (lambda prompt:
               '[{"beat_id": "b1", "expression": "", "motion": "", '
               '"camera": "", "text_prompt": "fingers dancing across the '
               'console, sparks flying over the foundry floor"}]')
        creative, warnings = _sl.derive_creative_directives(
            beats, meta, led, llm_fn=llm)
        tp = creative["b1"]["text_prompt"]
        assert tp.startswith("face visible, speaking to camera")
        assert creative["b1"]["source"] == "template_consistency"
        assert any("person anchor" in w for w in warnings)

    def test_person_grounded_llm_prompt_kept_and_anchored(self):
        beats = [{"beat_id": "b1", "role": _sl.Role.CHARACTER_VIDEO.value,
                  "char_id": "c02", "text": "It's alive.",
                  "samples": None, "sample_rate": None, "dur_s": 2.0}]
        led = {"cast": [{"char_id": "c02", "name": "HAYES VANCE",
                         "portrait_prompt": "a stocky engineer in grey"}],
               "lines": []}
        meta = {"story_brief_terms": {"setting": ["foundry"]}}
        llm = (lambda prompt:
               '[{"beat_id": "b1", "expression": "", "motion": "", '
               '"camera": "", "text_prompt": "the stocky engineer face '
               'forward speaking, foundry behind"}]')
        creative, _ = _sl.derive_creative_directives(
            beats, meta, led, llm_fn=llm)
        tp = creative["b1"]["text_prompt"]
        assert tp.startswith("face visible, speaking to camera")
        assert "stocky engineer face forward speaking" in tp
        assert creative["b1"]["source"] == "llm"

    def test_template_path_gets_anchor_without_warning_spam(self):
        beats = [{"beat_id": "b1", "role": _sl.Role.CHARACTER_VIDEO.value,
                  "char_id": "c02", "text": "It's alive.",
                  "samples": None, "sample_rate": None, "dur_s": 2.0}]
        led = {"cast": [{"char_id": "c02", "name": "HAYES VANCE",
                         "portrait_prompt": "a stocky engineer in grey"}],
               "lines": []}
        meta = {"story_brief_terms": {"setting": ["foundry"]}}
        creative, warnings = _sl.derive_creative_directives(
            beats, meta, led, llm_fn=None)
        tp = creative["b1"]["text_prompt"]
        assert tp.startswith("face visible, speaking to camera")
        assert creative["b1"]["source"] == "template"
        assert not any("person anchor" in w for w in warnings)

    def test_prompt_hash_matches_final_anchored_prompt(self):
        beats = [{"beat_id": "b1", "role": _sl.Role.CHARACTER_VIDEO.value,
                  "char_id": "c02", "text": "It's alive.",
                  "samples": None, "sample_rate": None, "dur_s": 2.0}]
        led = {"cast": [{"char_id": "c02", "name": "HAYES VANCE",
                         "portrait_prompt": "a stocky engineer in grey"}],
               "lines": []}
        creative, _ = _sl.derive_creative_directives(beats, {}, led,
                                                     llm_fn=None)
        row = creative["b1"]
        assert row["prompt_hash"] == _sl._content_hash(row["text_prompt"])


# --------------------------------------------------------------------------- #
# F4 -- ShotLock backstop warning (the writer-side repair is exercised via the
#       writer integration; here the frozen-side detector is pinned)
# --------------------------------------------------------------------------- #

class TestSelfVocativeBackstop:
    def test_extract_beats_warns_on_own_name_vocative(self, caplog):
        led = {
            "cast": [{"char_id": "c03", "name": "GULLIVER REEVES"}],
            "lines": [{"line_id": "b004", "char_id": "c03",
                       "text": "Gulliver, it's not just a machine."}],
        }
        with caplog.at_level(logging.WARNING):
            _sl.extract_beats(led)
        assert any("OWN speaker's name" in r.message for r in caplog.records)

    def test_no_warning_for_other_speaker_vocative(self, caplog):
        led = {
            "cast": [{"char_id": "c02", "name": "HAYES VANCE"},
                     {"char_id": "c03", "name": "GULLIVER REEVES"}],
            "lines": [{"line_id": "b004", "char_id": "c02",
                       "text": "Gulliver, it's not just a machine."}],
        }
        with caplog.at_level(logging.WARNING):
            _sl.extract_beats(led)
        assert not any("OWN speaker's name" in r.message
                       for r in caplog.records)


# --------------------------------------------------------------------------- #
# F5 -- join hardening + manifest positioned-mode fallback
# --------------------------------------------------------------------------- #

class TestJoinHardening:
    def test_extract_beats_normalizes_announcer_char_id(self):
        led = {
            "cast": [{"char_id": "c01", "name": "ANNOUNCER"}],
            "lines": [{"line_id": "b001", "char_id": "announcer",
                       "text": "Tonight, a tale.", "speaker_role": "announcer"}],
        }
        beats = _sl.extract_beats(led)
        assert beats[0]["char_id"] == "c01"

    def test_unknown_char_id_passes_through(self):
        led = {"cast": [{"char_id": "c01", "name": "ANNOUNCER"}],
               "lines": [{"line_id": "b001", "char_id": "c09",
                          "text": "Hello there."}]}
        assert _sl.extract_beats(led)[0]["char_id"] == "c09"

    def test_shot_rows_carry_char_id(self):
        beats = [{"beat_id": "b002", "role": _sl.Role.CHARACTER_VIDEO.value,
                  "char_id": "c02", "text": "x", "samples": None,
                  "sample_rate": None, "dur_s": 1.0}]
        budget = {"per_beat": {"b002": 25}, "clip_mode": "unique_per_beat"}
        groups, shots = _sl.build_execution_plan(beats, budget, {}, {})
        assert shots[0]["char_id"] == "c02"

    def test_driver_prefers_shot_char_id(self):
        led = {"meta": {}, "cast": [],
               "lines": [{"line_id": "b001", "char_id": "announcer",
                          "start_s": 1.0, "dur_s": 2.0}],
               "images": {"images": [
                   {"object_id": "c01", "path": "/tmp/c01.png"}]}}
        s = _shot("shot_b001", "character_video", engine="humo",
                  char_id="c01")
        req = _rd.build_request_from_shot(s, led)
        assert req["asset_refs"]["init_image"] == "/tmp/c01.png"

    def test_missing_portrait_warns_for_talking_head(self, caplog):
        led = {"meta": {}, "cast": [], "lines": [],
               "images": {"images": []}}
        s = _shot("shot_b002", "character_video", engine="humo",
                  char_id="c02")
        with caplog.at_level(logging.WARNING):
            _rd.build_request_from_shot(s, led)
        assert any("NO portrait-index entry" in r.message
                   for r in caplog.records)

    def test_no_portrait_warning_for_text_engine(self, caplog):
        led = _scene_ledger()
        s = _shot("shot_b001", "announcer_visual")
        with caplog.at_level(logging.WARNING):
            _rd.build_request_from_shot(s, led)
        assert not any("NO portrait-index entry" in r.message
                       for r in caplog.records)


class TestManifestPositionedFallback:
    def _result(self):
        led = {
            "video": {"video_revision": 1, "fps": 25,
                      "canonical_canvas": {"w": 1472, "h": 832},
                      "shots": [
                          {"shot_id": "shot_b000_music_open",
                           "source_line_ids": [], "start_s": 0.0,
                           "dur_s": 9.5, "engine_id": "ltx_video",
                           "target_frame_count": 238, "char_id": ""},
                          {"shot_id": "shot_b001",
                           "source_line_ids": ["b001"],
                           "engine_id": "ltx_video",
                           "target_frame_count": 175, "char_id": "c01"},
                      ]},
            "lines": [{"line_id": "b001", "char_id": "announcer",
                       "start_s": 9.5, "dur_s": 7.0}],
            "images": {"images": [{"object_id": "c01",
                                   "path": "/tmp/c01.png"}]},
        }
        return {"ledger": led, "clips": {}, "trace": []}

    def test_synthetic_row_start_s_falls_back_to_shot(self):
        m = _rd.build_clip_manifest(self._result(), episode_id="ep")
        rows = {r["beat_id"]: r for r in m["clips"]}
        assert rows["b000_music_open"]["start_s"] == 0.0
        assert rows["b001"]["start_s"] == 9.5

    def test_beat_id_uses_shared_rule(self):
        m = _rd.build_clip_manifest(self._result(), episode_id="ep")
        ids = [r["beat_id"] for r in m["clips"]]
        assert "b000_music_open" in ids and "shot_b000_music_open" not in ids

    def test_rows_carry_char_id_and_init_image(self):
        m = _rd.build_clip_manifest(self._result(), episode_id="ep")
        rows = {r["beat_id"]: r for r in m["clips"]}
        assert rows["b001"]["char_id"] == "c01"
        assert rows["b001"]["init_image"] == "/tmp/c01.png"

    def test_positioned_mode_restored_with_synthetic_row(self):
        from nodes.otr_silent_composite import plan_timeline_segments
        m = _rd.build_clip_manifest(self._result(), episode_id="ep")
        for r in m["clips"]:
            assert r["start_s"] is not None
