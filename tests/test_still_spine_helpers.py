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
    """The STILL prompt carries the LOOK subject (get_open_subject, in the image
    helper). The driver's OPEN VIDEO prompt is MOTION-centric (6/5
    BUG-LOCAL-112) -- it no longer inlines the radio-set subject; that wording
    lives only in the still helper, never inline in the driver."""

    def test_driver_open_video_prompt_is_motion_not_subject(self):
        import inspect
        from nodes._otr_video_engines import render_driver as rd
        src = inspect.getsource(rd)
        # the OPEN video prompt is built from the motion templates ...
        assert "_LTX_MOTION_PROMPT_BY_ROLE" in src
        # ... and the moved set-subject literal never survives inline here
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

    def test_scene_carries_bug411_restored_tails(self):
        """BUG-411: scene stills (the radio bookend) carry the restored 6/5
        cinematic grade + broadcast-distress tails, AFTER the shared style tail
        and BEFORE the no-text clause (layer order + no-text contract intact)."""
        p = helpers.compose_still_prompt(
            _meta_ok(), kind="scene_open", role="music_visual")
        assert helpers.IMAGE_GRADE_TAIL in p
        assert helpers.RADIO_BROADCAST_TAIL in p
        assert p.find(helpers.STYLE_TAIL_DEFAULT) < p.find(helpers.IMAGE_GRADE_TAIL)
        assert p.find(helpers.IMAGE_GRADE_TAIL) < p.find(helpers.RADIO_BROADCAST_TAIL)
        assert p.endswith(helpers.NO_TEXT_CLAUSE)

    def test_portrait_has_no_radio_broadcast_tail(self):
        """BUG-411 stays SCOPED: portraits do NOT gain the radio broadcast tail
        (a person, not a radio set) -- the restore touches scene stills only."""
        p = helpers.compose_still_prompt(
            _meta_ok(), kind="portrait",
            char_entry={"portrait_prompt": "a weathered engineer"})
        assert helpers.RADIO_BROADCAST_TAIL not in p
        assert helpers.IMAGE_GRADE_TAIL not in p


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

    def test_every_beat_gets_a_scene_still_target(self):
        # operator 2026-06-18: EVERY beat carries its OWN scene still regardless of
        # role/model -- the character/dialogue beat b002 (char_voice) used to be
        # SKIPPED (announcer/music-only), which is what made silent LTX hit the
        # LTX-I2V MISSING-STILL degrade. Now every beat gets a scene_beat target;
        # the image dispatcher's accepts_still gate is the one place that decides
        # whether the still is actually minted.
        from nodes import otr_meta_brief_image_prompt as mbp
        targets, _w = mbp.derive_scene_still_targets(_lines_timed())
        assert [t["beat_id"] for t in targets] == [
            "b000_music_open", "b001", "b002", "b005"]
        assert [t["kind"] for t in targets] == [
            "scene_open", "scene_beat", "scene_beat", "scene_beat"]
        assert targets[1]["role"] == "announcer_visual"
        assert targets[2]["role"] == "character_video"   # dialogue beat now covered
        assert targets[3]["role"] == "music_visual"

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

    def test_portrait_carries_cinematic_grade_bug411(self):
        """BUG-411 consistency (operator 2026-06-14): portraits now carry the
        same cinematic GRADE tail as the scene stills, so a flux_still beat in
        place of HuMo shows the 6/5 graded look. Radio tail stays OFF portraits
        (a person is not a radio set); the gear scrub still removes radio words."""
        from nodes import otr_meta_brief_image_prompt as mbp
        cast = [{"char_id": "c01",
                 "portrait_prompt": "a weathered engineer, kind eyes"}]
        payload, _w = mbp.derive_image_prompts(cast, _meta_ok(), llm_fn=None)
        port = mbp.objects_by_id(payload)["c01"]
        assert helpers.IMAGE_GRADE_TAIL in port["prompt"]
        assert helpers.RADIO_BROADCAST_TAIL not in port["prompt"]
        # idempotent: the grade tail appears exactly once
        assert port["prompt"].count(helpers.IMAGE_GRADE_TAIL) == 1

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


# ---------------------------------------------------------------------------
# 5. ST-3: dispatcher -- slots, seeds, cache key, episode dir (W2/W3 seams)
# ---------------------------------------------------------------------------


class TestDispatcherStillSpine:
    def test_slot_resolution_per_role(self):
        from nodes import otr_image_gen_dispatcher as disp
        policy = {"image_models": {
            "announcer_image_model": {"engine_id": "eng_a"},
            "music_image_model": {"engine_id": "eng_m"},
            "other_beats_image_model": {"engine_id": "eng_o"},
        }}
        assert disp.resolve_engine_for_role(policy, "announcer_visual") == \
            ("eng_a", "announcer_image_model", False)
        assert disp.resolve_engine_for_role(policy, "music_visual") == \
            ("eng_m", "music_image_model", False)
        assert disp.resolve_engine_for_role(policy, "character_video") == \
            ("eng_o", "other_beats_image_model", False)
        assert disp.resolve_engine_for_role(policy, "scene_broll") == \
            ("eng_o", "other_beats_image_model", False)

    def test_slot_fallback_is_flagged(self):
        from nodes import otr_image_gen_dispatcher as disp
        policy = {"image_models": {
            "other_beats_image_model": {"engine_id": "eng_o"}}}
        eid, slot, fell = disp.resolve_engine_for_role(policy, "music_visual")
        assert (eid, slot, fell) == ("eng_o", "music_image_model", True)

    def test_seed_request_hash_per_object(self):
        from nodes import otr_image_gen_dispatcher as disp
        cfg = {"mode": "request_hash", "request_seed": 7}
        s1 = disp.resolve_object_seed(cfg, "c1", "ph1")
        s2 = disp.resolve_object_seed(cfg, "still_b000", "ph2")
        assert s1 != s2                              # per-object
        assert s1 == disp.resolve_object_seed(cfg, "c1", "ph1")  # stable
        fixed = {"mode": "fixed", "request_seed": 7}
        assert disp.resolve_object_seed(fixed, "anything", "ph") == 7

    def test_bookend_scene_open_fixed_seed(self, monkeypatch):
        """BUG-411: the radio bookend (kind=scene_open) pins to the
        deterministic 6/5 seed 4242 (env-overridable), independent of the
        request-hash; every other kind keeps its per-object hashed seed."""
        from nodes import otr_image_gen_dispatcher as disp
        cfg = {"mode": "request_hash", "request_seed": 99}
        assert disp.resolve_object_seed(
            cfg, "still_b000_music_open", "phX", kind="scene_open") == 4242
        # other kinds untouched (no kind, or a non-open kind) -> hashed seed
        assert disp.resolve_object_seed(
            cfg, "still_b000_music_open", "phX", kind="scene_beat") != 4242
        monkeypatch.setenv("OTR_RADIO_BOOKEND_SEED", "777")
        assert disp.resolve_object_seed(
            cfg, "anything", "ph", kind="scene_open") == 777

    def test_cache_key_gains_kind_w_h(self):
        from nodes import otr_image_gen_dispatcher as disp
        base = ("r", "o", "ph", 1, "e", "1")
        assert disp.request_cache_key(*base, kind="portrait", w=832, h=1216) \
            != disp.request_cache_key(*base, kind="scene_open", w=832, h=1216)
        assert disp.request_cache_key(*base, kind="scene_open", w=1472, h=832) \
            != disp.request_cache_key(*base, kind="scene_open", w=832, h=832)

    def test_scene_still_dispatch_end_to_end(self, tmp_path):
        """A scene object renders on the MUSIC slot, lands in the episode
        stills dir + the manifest, carries w/h to the engine call, and never
        touches cast."""
        import numpy as np
        from nodes import otr_image_gen_dispatcher as disp
        from nodes._otr_image_engines import registry as ireg
        saved = dict(ireg._IMAGE_REGISTRY._registry)
        try:
            import types
            ireg._IMAGE_REGISTRY._registry.clear()
            ireg._IMAGE_REGISTRY._registry["flux_gen1"] = types.SimpleNamespace(
                name="flux_gen1", roles=("music_visual", "character_video"),
                default_roles=("music_visual",), commercial_clean=True,
                requires_flag=None, required_inputs=("text_prompt",),
                engine_version="1")
            ledger = {"episode_id": "ep_scene", "cast": []}
            policy = {"image_models": {
                "music_image_model": {"engine_id": "flux_gen1"},
                "other_beats_image_model": {"engine_id": "flux_gen1"}},
                "seed": {"mode": "request_hash", "request_seed": 3}}
            payload = {"version": 1, "objects": [{
                "object_id": "still_b000_music_open", "kind": "scene_open",
                "role": "music_visual", "beat_id": "b000_music_open",
                "w": 1472, "h": 832,
                "prompt": "a vintage radio set warming up, no on-screen text",
                "prompt_hash": "ph-open", "source": "scene_timed"}]}
            seen = {}

            def gen_fn(req):
                seen.update(req)
                return np.full((8, 8, 3), 50, dtype=np.uint8)

            led, done, _r, warns = disp.dispatch_images(
                ledger, policy, payload, gen_fn=gen_fn,
                output_dir=str(tmp_path), lockdir=tmp_path / "l.lockdir",
            )
            assert warns == []
            assert seen["engine_id"] == "flux_gen1"
            assert seen["width"] == 1472 and seen["height"] == 832
            assert seen["kind"] == "scene_open"
            row = led["images"]["images"][0]
            assert row["kind"] == "scene_open"
            assert row["beat_id"] == "b000_music_open"
            assert "otr/episodes/ep_scene/stills/" in \
                row["path"].replace("\\", "/")
            import os, json as _json
            assert os.path.exists(row["path"])
            man = _json.loads(open(
                os.path.join(os.path.dirname(row["path"]),
                             "stills_manifest.json"),
                encoding="utf-8").read())
            assert man["stills"][0]["beat_id"] == "b000_music_open"
            assert led["cast"] == []        # cast never grown by a scene still
        finally:
            ireg._IMAGE_REGISTRY._registry.clear()
            ireg._IMAGE_REGISTRY._registry.update(saved)

    def test_st4_still_index_and_family_init(self, tmp_path, monkeypatch):
        """ST-4 / W6: _still_index keys scene_* rows by beat_id (newest row
        wins); image_to_video + static_motion shots take init from the scene
        still; audio_driven_face keeps the portrait; a missing still falls
        back LOUD to the pre-spine init; _init_source/_init_image stamped."""
        from nodes._otr_video_engines import cheap_families  # noqa: F401
        from nodes._otr_video_engines import render_driver as rd
        still = tmp_path / "still_b007_abc.png"
        still.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 80)
        portrait = tmp_path / "c01_portrait.png"
        portrait.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 80)
        ledger = {
            "video": {"video_revision": 1, "shots": []},
            "lines": [{"line_id": "b007", "char_id": "c01",
                       "start_s": 1.0, "dur_s": 2.0}],
            "images": {"images": [
                {"object_id": "c01", "kind": "portrait", "char_id": "c01",
                 "path": str(portrait)},
                {"object_id": "still_b007", "kind": "scene_beat",
                 "beat_id": "b007", "path": "stale.png"},
                {"object_id": "still_b007", "kind": "scene_beat",
                 "beat_id": "b007", "path": str(still)},   # newest wins
            ]},
        }
        assert rd._still_index(ledger) == {"b007": str(still)}

        def shot(engine, family):
            return {"shot_id": "shot_b007", "beat_id": "b007",
                    "engine_id": engine, "family": family,
                    "target_frame_count": 25, "source_line_ids": ["b007"],
                    "char_id": "c01", "creative": {}}

        # static_motion + image_to_video -> scene still
        for eng, fam in (("still_kenburns", "static_motion"),
                         ("wan_i2v", "image_to_video")):
            req = rd.build_request_from_shot(shot(eng, fam), ledger)
            assert req["observability"]["init_source"] == "scene_still", eng
            assert req["observability"]["init_image"] == still.name
            assert req["asset_refs"]["init_image"] == str(still)
        # audio_driven_face -> portrait (unchanged)
        req = rd.build_request_from_shot(
            shot("humo", "audio_driven_face"), ledger)
        assert req["observability"]["init_source"] == "portrait"
        assert req["asset_refs"]["init_image"] == str(portrait)
        # text engine (ltx_video): LK-1a flipped i2v DEFAULT ON -- the scene
        # still wins when present; OTR_ENABLE_LTX_I2V=0 restores the
        # pre-spine portrait-by-char init.
        monkeypatch.delenv("OTR_ENABLE_LTX_I2V", raising=False)
        req = rd.build_request_from_shot(
            shot("ltx_video", "text_to_video"), ledger)
        assert req["observability"]["init_source"] == "scene_still"
        monkeypatch.setenv("OTR_ENABLE_LTX_I2V", "0")
        req = rd.build_request_from_shot(
            shot("ltx_video", "text_to_video"), ledger)
        assert req["observability"]["init_source"] == "portrait"
        monkeypatch.delenv("OTR_ENABLE_LTX_I2V", raising=False)

        # missing scene still -> LOUD fallback to the pre-spine init
        ledger2 = {k: (dict(v) if isinstance(v, dict) else v)
                   for k, v in ledger.items()}
        ledger2["images"] = {"images": [
            {"object_id": "c01", "kind": "portrait", "char_id": "c01",
             "path": str(portrait)}]}
        req = rd.build_request_from_shot(
            shot("wan_i2v", "image_to_video"), ledger2)
        assert req["observability"]["init_source"] == "portrait"   # fell back, stamped truthfully
        assert req["asset_refs"]["init_image"] == str(portrait)

    def test_st5_kenburns_reads_driver_built_request(self, tmp_path):
        """ST-5/W6 pin: the request the DRIVER builds for a static_motion
        shot is readable by still_kenburns' own _still_path (the ST-0 probe,
        locked as a test) -- the 6/5 conditioned look ships through this
        seam with zero new GPU risk."""
        from nodes._otr_video_engines import render_driver as rd
        from nodes._otr_video_engines.cheap_families import StillKenBurnsFamily
        still = tmp_path / "still_b001_xyz.png"
        still.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 80)
        ledger = {
            "video": {"video_revision": 1, "shots": []},
            "lines": [{"line_id": "b001", "start_s": 0.0, "dur_s": 2.0}],
            "images": {"images": [
                {"object_id": "still_b001", "kind": "scene_open",
                 "beat_id": "b001", "path": str(still)}]},
        }
        shot = {"shot_id": "shot_b001", "engine_id": "still_kenburns",
                "family": "static_motion", "target_frame_count": 25,
                "source_line_ids": ["b001"], "char_id": "", "creative": {}}
        req = rd.build_request_from_shot(shot, ledger)
        eng = StillKenBurnsFamily()
        assert eng.uses_still is True
        assert eng._still_path(req) == str(still)

    def test_st5_wan_i2v_reads_driver_built_request(self, tmp_path):
        """ST-5/W6 pin: the same driver-built request feeds wan_i2v's
        _init_image_ref (init = the beat's scene still under the
        OTR_ENABLE_WAN_I2V opt-in; the adapter itself stays flag-gated)."""
        from nodes._otr_video_engines import render_driver as rd
        from nodes._otr_video_engines.eng_wan_i2v import WanI2VEngine
        still = tmp_path / "still_b002_xyz.png"
        still.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 80)
        ledger = {
            "video": {"video_revision": 1, "shots": []},
            "lines": [{"line_id": "b002", "start_s": 0.0, "dur_s": 2.0}],
            "images": {"images": [
                {"object_id": "still_b002", "kind": "scene_beat",
                 "beat_id": "b002", "path": str(still)}]},
        }
        shot = {"shot_id": "shot_b002", "engine_id": "wan_i2v",
                "family": "image_to_video", "target_frame_count": 25,
                "source_line_ids": ["b002"], "char_id": "", "creative": {}}
        req = rd.build_request_from_shot(shot, ledger)
        eng = WanI2VEngine() if isinstance(WanI2VEngine, type) else WanI2VEngine
        assert eng._init_image_ref(req) == str(still)
        assert req["observability"]["init_source"] == "scene_still"

    def test_st4_manifest_rows_gain_init_source(self):
        from nodes._otr_video_engines import render_driver as rd
        result = {
            "ledger": {
                "video": {"video_revision": 1, "fps": 25,
                          "canonical_canvas": {"w": 1472, "h": 832},
                          "shots": [{"shot_id": "shot_b001",
                                     "source_line_ids": ["b001"],
                                     "engine_id": "wan_i2v",
                                     "target_frame_count": 25,
                                     "char_id": "c01", "start_s": 0.0}]},
                "lines": [{"line_id": "b001", "start_s": 0.0}],
                "images": {"images": []},
            },
            "clips": {"shot_b001": {"path": "", "engine_id": "wan_i2v",
                                    "frame_count": 0}},
            "trace": [{"shot_id": "shot_b001", "attempts": 1,
                       "final_engine": "wan_i2v",
                       "init_source": "scene_still",
                       "init_image": "still_b001_abc.png"}],
        }
        man = rd.build_clip_manifest(result, episode_id="ep")
        row = man["clips"][0]
        assert row["init_source"] == "scene_still"
        assert row["init_image_used"] == "still_b001_abc.png"

    def test_st7_ordering_gate_node_contracts(self):
        """ST-7 ordering: the render node declares the image_done forceInput
        gate (W4) and the dispatcher declares episode_id (DS-3) -- the json
        wires are pinned in test_workflow_live_passes_validator."""
        from nodes.otr_video_render_batch import OTRVideoRenderBatch
        from nodes.otr_image_gen_dispatcher import OTRImageGenDispatcher
        rb_opt = OTRVideoRenderBatch.INPUT_TYPES()["optional"]
        assert "image_done" in rb_opt
        assert rb_opt["image_done"][1].get("forceInput") is True
        di_opt = OTRImageGenDispatcher.INPUT_TYPES()["optional"]
        assert "episode_id" in di_opt
        assert di_opt["episode_id"][1].get("forceInput") is True
        # ShotLock's additive episode_id output (slot 4)
        from nodes.otr_shot_lock import OTRShotLock
        assert OTRShotLock.RETURN_NAMES[4] == "episode_id"
        assert OTRShotLock.RETURN_TYPES[4] == "STRING"

    def test_unkeyed_episode_is_loud(self, tmp_path):
        import numpy as np
        from nodes import otr_image_gen_dispatcher as disp
        from nodes._otr_image_engines import registry as ireg
        saved = dict(ireg._IMAGE_REGISTRY._registry)
        try:
            import types
            ireg._IMAGE_REGISTRY._registry.clear()
            ireg._IMAGE_REGISTRY._registry["flux_gen1"] = types.SimpleNamespace(
                name="flux_gen1", roles=("character_video",),
                default_roles=(), commercial_clean=True, requires_flag=None,
                required_inputs=("text_prompt",), engine_version="1")
            ledger = {"cast": [{"char_id": "c1", "name": "X"}]}
            policy = {"image_models": {
                "other_beats_image_model": {"engine_id": "flux_gen1"}},
                "seed": {"request_seed": 0}}
            payload = {"version": 1, "objects": [{
                "object_id": "c1", "kind": "portrait",
                "role": "character_video", "char_id": "c1",
                "w": 832, "h": 1216, "prompt": "a person, face visible",
                "prompt_hash": "ph", "source": "template"}]}
            _led, _d, _r, warns = disp.dispatch_images(
                ledger, policy, payload,
                gen_fn=lambda _q: np.full((4, 4, 3), 9, dtype=np.uint8),
                output_dir=str(tmp_path), lockdir=tmp_path / "l.lockdir")
            assert any("episode_id missing" in w for w in warns)
        finally:
            ireg._IMAGE_REGISTRY._registry.clear()
            ireg._IMAGE_REGISTRY._registry.update(saved)
