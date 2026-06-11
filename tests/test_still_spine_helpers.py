"""Still-spine ST-1: shared open-subject + 5-layer still-prompt composer.

Locks the W1 helper seam (STILL_SPINE_SPRINT_PLAN ST-1):

  * get_open_subject -- the driver's round-5 concrete radio-set wording,
    moved verbatim (3 branches: synthetic / announcer_visual / other);
  * PARITY -- the render driver's LTX open prompt and the scene still
    prompt for the same role lead with the SAME subject string;
  * get_era_tail(profile="still") -- atmosphere line + palette top-2 +
    lighting top-2, word-boundary-capped ~120 chars; the default "full"
    profile is byte-identical to the pre-still-spine behavior;
  * compose_still_prompt -- legacy 5-layer order (subject / setting top-2 /
    framing / trimmed era tail / style tail); NO_TEXT_CLAUSE on scene
    kinds ONLY; portraits lead with the character's own description.

Pure CPU tests; no I/O, no GPU, no ComfyUI imports.
"""

from __future__ import annotations

import pytest

from nodes import _otr_story_brief_helpers as helpers


def _meta_ok() -> dict:
    return {
        "story_brief": "a dim relay station on the Martian flats at dusk",
        "story_brief_status": "ok",
        "story_brief_terms": {
            "setting":    ["relay station", "martian flats", "dust airlock"],
            "lighting":   ["red dusk glow", "panel backlight", "hard rim"],
            "atmosphere": ["static", "tense", "ominous"],
        },
        "atmosphere_line": "thin red dust hangs in the dusk air",
        "visual_palette": ["rust red", "burnt amber", "steel grey"],
    }


# ---------------------------------------------------------------------------
# 1. get_open_subject -- the moved driver wording
# ---------------------------------------------------------------------------


class TestGetOpenSubject:
    def test_synthetic_open_wording(self):
        s = helpers.get_open_subject("music_visual", True)
        assert s.startswith("a vintage radio set warming up on a wooden table")
        assert "glowing dials and tubes" in s and "tungsten" in s

    def test_announcer_wording(self):
        s = helpers.get_open_subject("announcer_visual", False)
        assert s.startswith("a 1940s radio station studio")
        assert "lit dials and tubes" in s

    def test_other_open_wording(self):
        s = helpers.get_open_subject("music_visual", False)
        assert s.startswith("a vintage radio set glowing warmly")

    def test_synthetic_wins_over_role(self):
        assert helpers.get_open_subject("announcer_visual", True) == \
            helpers.get_open_subject("music_visual", True)

    def test_pure_and_total(self):
        # never raises, never empty, on any junk input
        for role in ("", None, "scene_broll", 7):
            for syn in (True, False):
                assert helpers.get_open_subject(role, syn)


class TestDriverParity:
    """The ONE-source-of-truth contract: the driver imports and calls
    get_open_subject; the old inline wording is GONE from the driver."""

    def test_driver_calls_helper(self):
        import inspect
        from nodes._otr_video_engines import render_driver as rd
        src = inspect.getsource(rd)
        assert "get_open_subject" in src
        # the moved literal must not survive inline in the driver
        assert "warming up on a wooden" not in src

    @pytest.mark.parametrize("kind,role,synthetic", [
        ("scene_open", "music_visual", True),
        ("scene_beat", "announcer_visual", False),
        ("scene_beat", "music_visual", False),
    ])
    def test_still_prompt_leads_with_driver_subject(self, kind, role, synthetic):
        subject = helpers.get_open_subject(role, synthetic)
        still = helpers.compose_still_prompt(
            _meta_ok(), kind=kind, role=role, beat_id="b000")
        assert still.startswith(subject)


# ---------------------------------------------------------------------------
# 2. era-tail profiles
# ---------------------------------------------------------------------------


class TestEraTailProfiles:
    def test_full_profile_unchanged_default(self):
        meta = _meta_ok()
        assert helpers.get_era_tail(meta) == helpers.get_era_tail(meta, "full")

    def test_still_profile_content(self):
        tail = helpers.get_era_tail(_meta_ok(), profile="still")
        assert "thin red dust hangs in the dusk air" in tail
        assert "rust red" in tail and "burnt amber" in tail
        assert "steel grey" not in tail            # palette top-2 only
        assert "red dusk glow" in tail and "panel backlight" in tail
        assert "hard rim" not in tail              # lighting top-2 only

    def test_still_profile_cap(self):
        meta = _meta_ok()
        meta["atmosphere_line"] = "x" * 60 + " " + "y" * 80
        tail = helpers.get_era_tail(meta, profile="still")
        assert len(tail) <= 120
        assert not tail.endswith(",")

    def test_still_profile_never_empty(self):
        assert helpers.get_era_tail({}, profile="still") == \
            helpers.ERA_TAIL_DEFAULT


# ---------------------------------------------------------------------------
# 3. compose_still_prompt -- the 5-layer order
# ---------------------------------------------------------------------------


class TestComposeStillPrompt:
    def test_scene_layer_order(self):
        meta = _meta_ok()
        p = helpers.compose_still_prompt(
            meta, kind="scene_open", role="music_visual", beat_id="b000")
        subject = helpers.get_open_subject("music_visual", True)
        i_subj = p.find(subject)
        i_set = p.find("relay station, martian flats")
        i_frame = p.find(helpers.STILL_FRAMING_OPEN)
        i_tail = p.find("thin red dust")
        i_style = p.find(helpers.STYLE_TAIL_DEFAULT)
        assert -1 not in (i_subj, i_set, i_frame, i_tail, i_style)
        assert i_subj < i_set < i_frame < i_tail < i_style

    def test_scene_carries_no_text_clause(self):
        p = helpers.compose_still_prompt(
            _meta_ok(), kind="scene_open", role="music_visual")
        assert p.endswith(helpers.NO_TEXT_CLAUSE)

    def test_portrait_no_no_text_clause(self):
        p = helpers.compose_still_prompt(
            _meta_ok(), kind="portrait",
            char_entry={"portrait_prompt": "a weathered radio engineer"})
        assert not p.endswith(helpers.NO_TEXT_CLAUSE)

    def test_portrait_leads_with_character(self):
        p = helpers.compose_still_prompt(
            _meta_ok(), kind="portrait",
            char_entry={"portrait_prompt": "a weathered engineer, kind eyes"})
        assert p.startswith("a weathered engineer, kind eyes")
        assert helpers.STILL_FRAMING_PORTRAIT in p

    def test_portrait_key_chain(self):
        for key in ("portrait_prompt", "appearance", "character_description"):
            p = helpers.compose_still_prompt(
                _meta_ok(), kind="portrait", char_entry={key: "marker-xyz"})
            assert p.startswith("marker-xyz"), key

    def test_never_empty_on_bare_meta(self):
        for kind in ("scene_open", "scene_beat", "portrait"):
            p = helpers.compose_still_prompt({}, kind=kind, role="")
            assert p and helpers.STYLE_TAIL_DEFAULT in p

    def test_deterministic(self):
        a = helpers.compose_still_prompt(
            _meta_ok(), kind="scene_open", role="music_visual")
        b = helpers.compose_still_prompt(
            _meta_ok(), kind="scene_open", role="music_visual")
        assert a == b


# ---------------------------------------------------------------------------
# 4. ST-2: the versioned object payload (emission; W1 seam)
# ---------------------------------------------------------------------------


def _lines_timed():
    return [
        {"line_id": "b001", "speaker_role": "announcer",
         "char_id": "announcer", "start_s": 8.4, "dur_s": 4.0},
        {"line_id": "b002", "speaker_role": "char_voice",
         "char_id": "c01", "start_s": 12.4, "dur_s": 6.0},
        {"line_id": "b005", "speaker_role": "music_close",
         "char_id": "music_close", "start_s": 18.4, "dur_s": 5.0},
    ]


class TestSceneStillObjects:
    def test_b000_present_without_shotlock(self):
        """The W1 QA assert: the open object exists from PURE helper
        derivation on the lines -- no ShotLock, no video.shots anywhere."""
        from nodes import otr_meta_brief_image_prompt as mbp
        payload, _w = mbp.derive_image_prompts(
            [], _meta_ok(), llm_fn=None, lines=_lines_timed())
        objs = mbp.objects_by_id(payload)
        assert "still_b000_music_open" in objs
        opn = objs["still_b000_music_open"]
        assert opn["kind"] == "scene_open"
        assert opn["beat_id"] == "b000_music_open"
        assert opn["role"] == "music_visual"
        assert opn["source"] == "scene_timed"

    def test_open_announcer_outro_targets_only(self):
        from nodes import otr_meta_brief_image_prompt as mbp
        targets, _w = mbp.derive_scene_still_targets(_lines_timed())
        assert [t["beat_id"] for t in targets] == [
            "b000_music_open", "b001", "b005"]
        assert [t["kind"] for t in targets] == [
            "scene_open", "scene_beat", "scene_beat"]
        assert targets[1]["role"] == "announcer_visual"
        assert targets[2]["role"] == "music_visual"

    def test_pretiming_open_emitted_loud(self):
        """Pre-audio ledger (no start_s anywhere) still emits the open
        target -- LOUD -- so production never loses the 6/5 open still."""
        from nodes import otr_meta_brief_image_prompt as mbp
        lines = [{k: v for k, v in ln.items() if k != "start_s"}
                 for ln in _lines_timed()]
        targets, warns = mbp.derive_scene_still_targets(lines)
        opens = [t for t in targets if t["kind"] == "scene_open"]
        assert opens and opens[0]["source"] == "scene_pretiming"
        assert any("OPTIMISTICALLY" in w for w in warns)

    def test_no_open_when_no_head_gap(self):
        from nodes import otr_meta_brief_image_prompt as mbp
        lines = _lines_timed()
        lines[0]["start_s"] = 0.5            # real timing, no head gap
        targets, warns = mbp.derive_scene_still_targets(lines)
        assert not [t for t in targets if t["kind"] == "scene_open"]
        assert not warns

    def test_guards_branch_by_kind(self):
        """Scene stills SKIP the person guard + gear scrub (they are radio
        sets, not people) and carry the no-text clause; portraits keep
        their guards. The derivation paths split BEFORE the guards."""
        from nodes import otr_meta_brief_image_prompt as mbp
        cast = [{"char_id": "c01", "name": "HAYES",
                 "portrait_prompt": "a stocky engineer near a radio console"}]
        payload, warns = mbp.derive_image_prompts(
            cast, _meta_ok(), llm_fn=None, lines=_lines_timed())
        objs = mbp.objects_by_id(payload)
        opn = objs["still_b000_music_open"]
        assert "radio" in opn["prompt"].lower()          # gear NOT scrubbed
        assert opn["prompt"].endswith(helpers.NO_TEXT_CLAUSE)
        port = objs["c01"]
        assert port["kind"] == "portrait"
        assert "radio" not in port["prompt"].lower()     # gear scrubbed
        assert not port["prompt"].endswith(helpers.NO_TEXT_CLAUSE)
        assert not any("still_b000" in w for w in warns)

    def test_scene_dims_landscape_div32(self, monkeypatch):
        from nodes import otr_meta_brief_image_prompt as mbp
        monkeypatch.setenv("OTR_VIDEO_LANDSCAPE_CANVAS", "1475x839")
        w, h = mbp._landscape_still_dims()
        assert (w, h) == (1472, 832)         # snapped DOWN to /32
        monkeypatch.delenv("OTR_VIDEO_LANDSCAPE_CANVAS")
        payload, _w = mbp.derive_image_prompts(
            [], _meta_ok(), llm_fn=None, lines=_lines_timed())
        opn = mbp.objects_by_id(payload)["still_b000_music_open"]
        assert opn["w"] % 32 == 0 and opn["h"] % 32 == 0
        assert opn["w"] > opn["h"]           # landscape

    def test_scene_prompt_is_shared_composer(self):
        """W1 parity at the OBJECT level: the emitted open prompt IS the
        shared composer's output (subject parity with the driver locked)."""
        from nodes import otr_meta_brief_image_prompt as mbp
        payload, _w = mbp.derive_image_prompts(
            [], _meta_ok(), llm_fn=None, lines=_lines_timed())
        opn = mbp.objects_by_id(payload)["still_b000_music_open"]
        expect = helpers.compose_still_prompt(
            _meta_ok(), kind="scene_open", role="music_visual",
            beat_id="b000_music_open")
        assert opn["prompt"] == expect
        assert opn["prompt_hash"] == mbp._content_hash(expect)

    def test_payload_versioned_and_portraits_migrated(self):
        from nodes import otr_meta_brief_image_prompt as mbp
        cast = [{"char_id": "c01", "portrait_prompt": "a stocky engineer"}]
        payload, _w = mbp.derive_image_prompts(
            cast, _meta_ok(), llm_fn=None, lines=_lines_timed())
        assert payload["version"] == 1
        kinds = {o["object_id"]: o["kind"] for o in payload["objects"]}
        assert kinds["c01"] == "portrait"    # object_id == char_id
        for o in payload["objects"]:
            for k in ("object_id", "kind", "role", "w", "h",
                      "prompt", "prompt_hash", "source"):
                assert k in o, (o.get("object_id"), k)

    def test_determinism_same_ledger_same_hashes(self):
        from nodes import otr_meta_brief_image_prompt as mbp
        a, _ = mbp.derive_image_prompts([], _meta_ok(), llm_fn=None,
                                        lines=_lines_timed())
        b, _ = mbp.derive_image_prompts([], _meta_ok(), llm_fn=None,
                                        lines=_lines_timed())
        assert a == b
