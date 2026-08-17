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


@pytest.fixture(autouse=True)
def _prompt_only_lanes_disable_i2v(monkeypatch):
    """This module tests prompt composition without minted scene stills.

    Real LTX dispatch is default-on I2V and fails closed when its still is
    missing; prompt-only tests select the documented text-only opt-out.
    """
    monkeypatch.setenv("OTR_ENABLE_LTX_I2V", "0")


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

    def test_short_ask_raised_to_decode_floor(self, monkeypatch):
        monkeypatch.delenv("OTR_LTX_MAX_FRAMES", raising=False)
        monkeypatch.setenv("OTR_LTX_MIN_DECODE_FRAMES", "169")
        # The r5b live catch: b004's 137f ask hit the wrapper VAEDecode tensor
        # mismatch, so an ask below the floor is raised into the tiled band.
        #
        # THE FLOOR IS NOW SET EXPLICITLY, because the SHIPPED default stopped
        # being 169 (lane 9, 2026-08-11: the band is open at the declared
        # 1024x576). The mechanism this test owns -- "an ask below the ACTIVE
        # floor comes back AT the floor" -- is unchanged and still worth
        # pinning, so the floor it needs is now stated rather than inherited.
        assert _ltx._ltx_frame_length(137, 24) == 169
        assert _ltx._ltx_frame_length(50, 24) == 169

    def test_at_the_SHIPPED_floor_a_short_ask_renders_its_own_length(
            self, monkeypatch):
        """The other side of the same mechanism, and the lane 9 behaviour.

        With the decode band open at the declared canvas the floor is the
        bottom of the ladder, so a short ask is NOT raised -- it renders itself,
        snapped to 8n+1. This is what stopped a 2 s beat rendering 169 frames
        and discarding 119 of them.
        """
        monkeypatch.delenv("OTR_LTX_MAX_FRAMES", raising=False)
        monkeypatch.delenv("OTR_LTX_MIN_DECODE_FRAMES", raising=False)
        assert _ltx._LTX_DECODE_FLOOR_DEFAULT == _ltx._LTX_MIN_FRAMES
        assert _ltx._ltx_frame_length(137, 24) == 137     # already 8n+1
        assert _ltx._ltx_frame_length(50, 24) == 49       # snapped DOWN to 8n+1
        assert _ltx._ltx_frame_length(9, 24) == 9         # the ladder's bottom

    def test_decode_floor_env_override(self, monkeypatch):
        monkeypatch.delenv("OTR_LTX_MAX_FRAMES", raising=False)
        monkeypatch.setenv("OTR_LTX_MIN_DECODE_FRAMES", "49")
        assert _ltx._ltx_frame_length(50, 24) == 49      # 8n+1 snap, low floor

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

    def test_zero_ask_uses_fallback_then_floor(self, monkeypatch):
        monkeypatch.delenv("OTR_LTX_MAX_FRAMES", raising=False)
        # A zero ask falls back to the supplied fallback, and the ACTIVE floor
        # then applies to that. With the floor high the fallback rises to it;
        # at the shipped floor it simply snaps to 8n+1. Both directions asserted
        # so the test proves the fallback is CONSULTED rather than proving one
        # floor's arithmetic (lane 9 moved the shipped floor to 9).
        monkeypatch.setenv("OTR_LTX_MIN_DECODE_FRAMES", "169")
        assert _ltx._ltx_frame_length(0, 24) == 169
        monkeypatch.delenv("OTR_LTX_MIN_DECODE_FRAMES", raising=False)
        assert _ltx._ltx_frame_length(0, 24) == 17     # fallback 24 -> 8n+1

    def test_EVERY_planner_legal_length_round_trips_unchanged(self, monkeypatch):
        """THE risk the floor move introduced, pinned.

        `_ltx_frame_length` snaps with ((n-1)//8)*8+1 -- it rounds DOWN. While
        the contract was min==max==169 the planner had exactly ONE legal length,
        so that snap was never exercised against a planner-produced value.
        `min_frames=9, quantum=8` opens the whole ladder.

        If any length the planner considers legal came back SHORTER, the render
        would disagree with the plan it was partitioned against, and
        `render_beat_coverage` refuses that mismatch -- the same class of live
        failure ("rendered N frame(s) but its plan asked for M") that forced the
        169 declaration in the first place. So this asserts the round trip over
        every legal length AND over every segment the REAL partition planner
        emits across a wide span of beats, rather than trusting the arithmetic.
        """
        monkeypatch.delenv("OTR_LTX_MAX_FRAMES", raising=False)
        monkeypatch.delenv("OTR_LTX_MIN_DECODE_FRAMES", raising=False)
        from nodes._otr_video_engines import coverage_plan as cp
        from nodes._otr_video_engines.eng_ltx_video import LtxVideoEngine

        contract = LtxVideoEngine.frame_contract
        legal = contract.legal_lengths()
        assert legal, "an enumerable ladder is the premise of this test"
        for n in legal:
            assert _ltx._ltx_frame_length(n, 25) == n, (
                "legal length %d does not round-trip" % n)

        segments = 0
        for beat in range(1, 600):
            try:
                plan = cp.partition_beat(beat, contract)
            except cp.CoveragePlanError:
                continue
            for seg in plan.segments:
                segments += 1
                asked = int(seg.render_frames)
                assert _ltx._ltx_frame_length(asked, 25) == asked, (
                    "beat %d: the planner asked for %d and the adapter would "
                    "render %d -- render_beat_coverage refuses that gap"
                    % (beat, asked, _ltx._ltx_frame_length(asked, 25)))
        # Guard against the check going hollow: a run that produced no segments
        # would report "clean" while asserting nothing at all.
        assert segments > 1000, "expected a real sweep, got %d segments" % segments

    def test_cap_below_floor_wins(self, monkeypatch):
        monkeypatch.setenv("OTR_LTX_MAX_FRAMES", "57")
        # The floor is SET here rather than inherited from the default. This
        # test's invariant is "an operator cap BELOW the decode floor wins", and
        # that requires a floor above the cap to mean anything at all. It rode
        # the old 169 default until lane 9 moved it to 9, at which point the cap
        # was no longer below the floor and the test was asserting nothing --
        # it went red for the right reason. State the floor, keep the invariant.
        monkeypatch.setenv("OTR_LTX_MIN_DECODE_FRAMES", "169")
        # effective floor = min(169, 57) = 57, so a 30f ask rises only to 57,
        # never past the operator's ceiling
        assert _ltx._ltx_frame_length(30, 24) == 57
        assert _ltx._ltx_frame_length(238, 24) == 57



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
    def test_role_motion_templates_motion_centric(self, monkeypatch):
        # 6/5 BUG-LOCAL-112 restoration: different ROLES get different motion
        # templates; same-role beats SHARE the template (visual variety comes
        # from the per-beat i2v still, not the text prompt). All motion-centric.
        monkeypatch.delenv("OTR_LTX_RADIO_PROMPT", raising=False)
        led = _scene_ledger()
        p_music = _rd.build_request_from_shot(
            _shot("shot_b000_music_open", "music_visual",
                  source_line_ids=[], start_s=0.0, dur_s=9.5), led)["text_prompt"]
        p_ann1 = _rd.build_request_from_shot(
            _shot("shot_b001", "announcer_visual"), led)["text_prompt"]
        p_ann2 = _rd.build_request_from_shot(
            _shot("shot_b005", "announcer_visual"), led)["text_prompt"]
        for p in (p_music, p_ann1, p_ann2):
            assert p.startswith("Continuous shot, same console")
            assert len(p) <= 240
        assert p_music != p_ann1            # music_open vs announcer template
        assert p_ann1 == p_ann2             # same role -> same template

    def test_synthetic_open_detected_by_structure_not_role(self, monkeypatch):
        monkeypatch.delenv("OTR_LTX_RADIO_PROMPT", raising=False)
        led = _scene_ledger()
        # role says announcer_visual, but EMPTY source_line_ids + the
        # b000_music_open suffix = synthetic music open -> a MUSIC motion
        # template wins over the role (structure is definitive). 2026-06-15: the
        # DEFAULT open is now the DYNAMIC music_open (the 6/12 smear was the
        # 1472x832 blur, fixed at 832x480 -> renders sharp + moving).
        monkeypatch.delenv("OTR_LTX_OPEN_MOTION_KEY", raising=False)
        s = _shot("shot_b000_music_open", "announcer_visual",
                  source_line_ids=[], start_s=0.0, dur_s=9.5)
        p = _rd.build_request_from_shot(s, led)["text_prompt"]
        # 2026-08-17 subject-first rewrite: the music_open discriminator moved
        # from "vibrates aggressively" to "races across the frequencies". The
        # old marker was one of FOUR phrases the Seedance softener rewrote on
        # this very register -- it shipped as "vibrates subtly", i.e. the
        # energy inverted on the episode's most energetic beat. The assertion
        # is unchanged in intent: structure (music_open) beats role (announcer).
        assert "races across frequencies" in p            # dynamic music_open template
        assert "Tuning dial needle sweeps" not in p       # NOT the announcer (structure won)
        # music_inter remains the calm rollback via env:
        monkeypatch.setenv("OTR_LTX_OPEN_MOTION_KEY", "music_inter")
        p2 = _rd.build_request_from_shot(s, led)["text_prompt"]
        assert "Oscilloscope dances to the rhythm" in p2  # calm rollback honored

    def test_open_leads_with_motion_not_subject(self, monkeypatch):
        # 6/5 BUG-LOCAL-112: the i2v still carries the LOOK; the video prompt is
        # MOTION-ONLY -> it leads with the motion frame, NOT the set subject
        # (the pre-112 dilution that caused flat pans).
        monkeypatch.delenv("OTR_LTX_RADIO_PROMPT", raising=False)
        led = _scene_ledger()
        p = _rd.build_request_from_shot(
            _shot("shot_b001", "announcer_visual"), led)["text_prompt"]
        assert p.startswith("Continuous shot, same console")
        assert "Tuning dial needle sweeps" in p            # motion verb present
        assert not p.startswith("a 1940s radio station")   # NOT the set subject
        assert "An innovator unveils a machine" not in p   # logline still banned

    def test_motion_template_passed_through_not_finished(self, monkeypatch):
        # motion template is verbatim (+ optional atmosphere fragment); it is
        # NOT run through finish_visual_prompt, so its motion verbs survive.
        monkeypatch.delenv("OTR_LTX_RADIO_PROMPT", raising=False)
        led = _scene_ledger()
        p = _rd.build_request_from_shot(
            _shot("shot_b001", "announcer_visual"), led)["text_prompt"]
        assert "Vacuum tubes pulse" in p and "Brass speaker grille trembles" in p

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
        assert req["observability"]["prompt_source"] == "env"

    def test_motion_prompt_stamps_meta(self, monkeypatch):
        monkeypatch.delenv("OTR_LTX_RADIO_PROMPT", raising=False)
        led = _scene_ledger()
        req = _rd.build_request_from_shot(
            _shot("shot_b001", "announcer_visual"), led)
        # OPEN roles now stamp motion_role (the brief+beat path is for non-open
        # text-engine roles like retired_role_a).
        assert req["observability"]["prompt_source"] == "motion_role"
        assert len(req["observability"]["prompt_sha8"]) == 8
        assert req["observability"]["prompt_chars"] == len(req["text_prompt"])

    def test_m4_creative_prompt_stamped_m4(self, monkeypatch):
        monkeypatch.delenv("OTR_LTX_RADIO_PROMPT", raising=False)
        led = _scene_ledger()
        s = _shot("shot_b001", "announcer_visual",
                  creative={"text_prompt": "a writer prompt", "source": "llm"})
        req = _rd.build_request_from_shot(s, led)
        assert req["text_prompt"] == "a writer prompt"
        assert req["observability"]["prompt_source"] == "m4"
        assert req["observability"]["prompt_subsource"] == "llm"


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
    # (person-anchor DETECTOR removed 2026-07-04 -- test_person_anchor_ok_* deleted;
    # the _subject_anchor prompt COMPOSITION below is what remains and is tested.)

    def test_subject_anchor_leads_with_face_tokens(self):
        a = _sl._subject_anchor("a stocky engineer in a grey shirt")
        assert a.startswith("face visible, speaking to camera")
        assert "stocky engineer" in a

    def test_subject_anchor_truncates_long_appearance(self):
        a = _sl._subject_anchor("x" * 400)
        assert len(a) < 200

    def test_object_only_llm_prompt_is_preserved_and_subject_anchored(self):
        """Authored visual vocabulary is not replaced by Python token overlap."""
        beats = [{"beat_id": "b1", "role": _sl.Role.CHARACTER_VIDEO.value,
                  "char_id": "c02", "text": "It's alive.",
                  "samples": None, "sample_rate": None, "dur_s": 2.0}]
        led = {"cast": [{"char_id": "c02", "name": "HAYES VANCE",
                         "portrait_prompt": "a stocky engineer in grey"}],
               "lines": []}
        meta = {"story_brief_terms": {"setting": ["foundry"]}}
        llm = (lambda prompt:
               '[{"beat_id": "b1", "expression": "", "motion": "", '
               '"camera": "", "text_prompt": "fingers dancing across the '
               'console, sparks flying over the foundry floor"}]')
        creative, warnings = _sl.derive_creative_directives(
            beats, meta, led, llm_fn=llm)
        row = creative["b1"]
        assert row["source"] == "llm"
        assert row["text_prompt"].startswith(
            "face visible, speaking to camera, a stocky engineer in grey")
        assert "fingers dancing across the console" in row["text_prompt"]
        assert "foundry floor" in row["text_prompt"]
        assert not warnings

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
                       "speaker_role": "character",
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
                       "speaker_role": "character",
                       "text": "Gulliver, it's not just a machine."}],
        }
        with caplog.at_level(logging.WARNING):
            _sl.extract_beats(led)
        assert not any("OWN speaker's name" in r.message
                       for r in caplog.records)


# --------------------------------------------------------------------------- #
# Portrait prompt guidance (wording belongs in the LLM request, not a Python
# vocabulary classifier)
# --------------------------------------------------------------------------- #

class TestPortraitPromptGuidance:
    def test_style_anchor_positive_only_three_quarter(self):
        from nodes import otr_meta_brief_image_prompt as mb
        low = mb.STYLE_ANCHOR.lower()
        assert "three-quarter" in low
        assert "no microphone" not in low and "not a recording" not in low

    def test_instruction_guides_world_grounding_and_upper_body(self):
        from nodes import otr_meta_brief_image_prompt as mb
        req = mb._build_char_prompt_request(
            {"char_id": "c02", "portrait_prompt": "an engineer"}, {}, "lab")
        assert "Do not mention radios" in req
        assert "head and upper body" in req


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
                          "speaker_role": "character",
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
