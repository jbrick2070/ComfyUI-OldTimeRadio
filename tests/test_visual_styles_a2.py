"""tests/test_visual_styles_a2.py

Visual-style TOTAL COVERAGE chunk A2 (STAGE3_TOTAL_COVERAGE_SUBPLAN v5 FINAL)
-- video/mesh lane: pack-owned MOTION REGISTERS substituted in
build_request_from_shot (r1 AG M4: render_clip has no meta), the
radio-object / background-plate LOOK splits, the pack-owned no-character
EMBLEM template, and the ADDITIVE provenance keys (r2 codex M5 + r4 codex
M3) on both the image prompt objects and the render-request observability.

Pins:
  1. MOTION: sci_fi prompts byte-identical through the REAL builder; the
     pack VALUE is what renders (a leaktest pack changes the prompt); the
     silent `or ...["announcer"]` fallback is RETIRED (a stub style with a
     missing console key raises KeyError, never remaps); the
     OTR_LTX_OPEN_MOTION_KEY retarget works off the STATIC key set.
  2. NO-LEAK (r3): the probe-locked _IA2V_TALKING_PROMPT_ANNOUNCER is a
     VERBATIM Python constant -- pack motion_registers["announcer"] never
     reaches a talking announcer bookend, even from a hostile pack.
  3. LOOK SPLITS: object anchors / plate scaffold compose geometry + pack
     look byte-identically; the emblem template formats {base} at the real
     fallback; music_visual keeps radio_form_from_meta.
  4. PROVENANCE: visual_style + prompt_field_source stamped ADDITIVELY on
     the 4 scene-family image objects + the motion-lane request
     observability; prompt text + sha stamps unchanged; the trace-copy
     allowlist carries the new keys.
  5. AST guards: zero production Loads of the retired motion fixture; the
     A2 look fixtures read only in their designated lanes.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from nodes import _otr_story_brief_helpers as helpers
from nodes import _otr_visual_styles as vs
from nodes import otr_meta_brief_image_prompt as imgp
from nodes._otr_video_engines import render_driver as rd

_REPO = Path(__file__).resolve().parent.parent
_NODES = _REPO / "nodes"
_STYLES_DIR = _NODES / "visual_styles"

_META_BRIEF = {
    "style": "a tense sci-fi thriller aboard a space station",
    "story_brief_terms": {
        "setting": ["a mars listening post", "dust-caked consoles"],
        "lighting": ["harsh sodium light", "deep shadow"],
        "atmosphere": ["claustrophobic dread"],
    },
    "episode_title": "Signal in the Dust",
}
_CHAR = {"char_id": "c01", "name": "MARGOT",
         "appearance": "a wiry engineer in a patched flight suit"}


@pytest.fixture(autouse=True)
def _fresh_registry():
    vs._clear_caches()
    yield
    vs._clear_caches()


@pytest.fixture()
def _single_pass_recipe(monkeypatch):
    # single-pass ltx recipe: no talking register, no radio-face requirement
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "distilled_native")
    monkeypatch.setenv(
        "OTR_LTX_AV_UNET",
        r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")
    monkeypatch.delenv("OTR_LTX_OPEN_MOTION_KEY", raising=False)
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)


def _led(meta):
    return {"meta": meta,
            "video": {"video_revision": 1, "shots": []},
            "lines": [{"line_id": "b000", "char_id": "",
                       "start_s": 0.0, "dur_s": 2.0}],
            "images": {"images": []}}


def _shot(sid="shot_b000_music_open", role="music_visual", sids=None):
    return {"shot_id": sid, "beat_id": "b000", "engine_id": "ltx_audio_in",
            "role": role, "family": "audio_conditioned_video",
            "target_frame_count": 25, "source_line_ids": list(sids or []),
            "char_id": "", "creative": {}}


def _expected_motion(key, meta=_META_BRIEF):
    # fixture-composed expectation: register + one atmosphere fragment
    base = rd._LTX_MOTION_PROMPT_BY_ROLE[key]
    atmo = ", ".join([str(t).strip() for t in
                      (meta.get("story_brief_terms", {})
                       .get("atmosphere") or [])if str(t).strip()][:1])
    if atmo and len(base) + len(atmo) + 7 <= rd._LTX_MOTION_PROMPT_MAX:
        return f"{base} {atmo} mood."
    return base


# ---------------------------------------------------------------------------
# 1. Motion registers through the REAL builder
# ---------------------------------------------------------------------------
class TestMotionRegisters:
    @pytest.mark.parametrize("sid,role,sids,key", [
        ("shot_b000_music_open", "music_visual", [], "music_open"),
        ("shot_b004", "announcer_visual", ["b000"], "announcer"),
        ("shot_b005_close", "music_visual", ["b000"], "music_close"),
        ("shot_b006", "music_visual", ["b000"], "music_inter"),
    ])
    def test_sci_fi_byte_identical(self, _single_pass_recipe, sid, role,
                                   sids, key):
        req = rd.build_request_from_shot(_shot(sid, role, sids),
                                         _led(dict(_META_BRIEF)))
        assert req["text_prompt"] == _expected_motion(key)
        assert (req["observability"]["prompt_field_source"]
                == "motion_registers:%s" % key)
        assert req["observability"]["visual_style"] == "sci_fi_radio"

    def test_pack_value_is_what_renders(self, _single_pass_recipe,
                                        tmp_path, monkeypatch):
        # a leaktest pack with a DIFFERENT music_open register changes the
        # rendered prompt -- proof the pack VALUE (not the fixture) renders.
        raw = json.loads((_STYLES_DIR / "sci_fi_radio.json")
                         .read_text(encoding="utf-8"))
        scratch = tmp_path / "visual_styles"
        scratch.mkdir()
        (scratch / "sci_fi_radio.json").write_text(
            json.dumps(raw), encoding="utf-8")
        leak = dict(raw)
        leak["style_id"] = "leaktest"
        leak["label"] = "Leak Test"
        leak["motion_registers"] = dict(raw["motion_registers"])
        leak["motion_registers"]["music_open"] = (
            "Paper radio unfolds. Creases flex with the beat.")
        (scratch / "leaktest.json").write_text(json.dumps(leak),
                                               encoding="utf-8")
        monkeypatch.setattr(vs, "_VISUAL_STYLES_ROOT", scratch)
        vs._clear_caches()
        meta = dict(_META_BRIEF)
        meta["visual_style"] = "leaktest"
        req = rd.build_request_from_shot(_shot(), _led(meta))
        assert "Paper radio unfolds." in req["text_prompt"]
        assert req["observability"]["visual_style"] == "leaktest"

    def test_missing_console_key_raises_never_remaps(self, _single_pass_recipe,
                                                     monkeypatch):
        # r2 codex S3: the silent `or ...["announcer"]` fallback is retired.
        class _Stub:
            style_id = "stub"
            motion_registers = {}
        monkeypatch.setattr(vs, "get_visual_style", lambda meta: _Stub())
        with pytest.raises(KeyError):
            rd.build_request_from_shot(_shot(), _led(dict(_META_BRIEF)))

    def test_env_retarget_uses_static_key_set(self, _single_pass_recipe,
                                              monkeypatch):
        monkeypatch.setenv("OTR_LTX_OPEN_MOTION_KEY", "music_inter")
        req = rd.build_request_from_shot(_shot(), _led(dict(_META_BRIEF)))
        assert req["text_prompt"] == _expected_motion("music_inter")
        monkeypatch.setenv("OTR_LTX_OPEN_MOTION_KEY", "no_such_key")
        req2 = rd.build_request_from_shot(_shot(), _led(dict(_META_BRIEF)))
        assert req2["text_prompt"] == _expected_motion("music_inter")

    def test_unknown_style_fails_the_shot_loud(self, _single_pass_recipe):
        with pytest.raises(vs.UnknownVisualStyleError):
            rd.build_request_from_shot(
                _shot(), _led({"visual_style": "no_such_style"}))


# ---------------------------------------------------------------------------
# 2. NO-LEAK: the probe-locked talking prompt outranks every pack
# ---------------------------------------------------------------------------
class TestTalkingNoLeak:
    def test_pack_announcer_motion_never_leaks_into_talking(
            self, _single_pass_recipe, tmp_path, monkeypatch):
        raw = json.loads((_STYLES_DIR / "sci_fi_radio.json")
                         .read_text(encoding="utf-8"))
        scratch = tmp_path / "visual_styles"
        scratch.mkdir()
        hostile = dict(raw)
        hostile["motion_registers"] = dict(raw["motion_registers"])
        hostile["motion_registers"]["announcer"] = (
            "HOSTILE PACK MOTION that must never reach the lip-sync lane.")
        (scratch / "sci_fi_radio.json").write_text(json.dumps(hostile),
                                                   encoding="utf-8")
        monkeypatch.setattr(vs, "_VISUAL_STYLES_ROOT", scratch)
        vs._clear_caches()
        # force the talking register active (engine-owned probe stubbed)
        monkeypatch.setattr(rd, "_ia2v_talking_register_active",
                            lambda eng: True)
        shot = _shot(sid="shot_b004", role="announcer_visual", sids=["b000"])
        led = _led(dict(_META_BRIEF))
        # the talking lane REQUIRES the wide radio-face still in the ledger
        face = tmp_path / "radio_face.png"
        face.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 80)
        led["images"]["images"].append(
            {"object_id": "still_announcer_visual_radio_face_169",
             "kind": "portrait", "path": str(face)})
        req = rd.build_request_from_shot(shot, led)
        assert req["text_prompt"] == rd._IA2V_TALKING_PROMPT_ANNOUNCER
        assert "HOSTILE" not in req["text_prompt"]
        assert (req["observability"]["prompt_field_source"]
                == "python:ia2v_talking")
        # music bookends KEEP the (hostile-pack) console register -- P2
        req2 = rd.build_request_from_shot(_shot(), _led(dict(_META_BRIEF)))
        assert req2["text_prompt"] != rd._IA2V_TALKING_PROMPT_ANNOUNCER

    def test_talking_constants_stay_python(self):
        # P8: verbatim constants; packs have no field for them.
        assert not hasattr(vs.resolve_visual_style("sci_fi_radio"),
                           "ia2v_talking_prompt")
        assert "talking to the viewer" in rd._IA2V_TALKING_PROMPT_ANNOUNCER


# ---------------------------------------------------------------------------
# 3. Look splits + emblem (byte-identity)
# ---------------------------------------------------------------------------
class TestLookSplits:
    def test_object_anchor_fixtures_compose(self):
        assert imgp._RADIO_OBJECT_ANCHOR == "%s, %s" % (
            imgp.RADIO_OBJECT_GEOMETRY, imgp.RADIO_OBJECT_LOOK_DEFAULT)
        assert imgp._RADIO_OBJECT_ANCHOR_WIDE == "%s, %s" % (
            imgp.RADIO_OBJECT_GEOMETRY_WIDE, imgp.RADIO_OBJECT_LOOK_DEFAULT)
        assert imgp.BACKGROUND_PLATE_POS_SCAFFOLD == "%s, %s" % (
            imgp.BACKGROUND_PLATE_GEOMETRY, imgp.PLATE_LOOK_DEFAULT)

    def test_pack_fixture_byte_identity(self):
        s = vs.resolve_visual_style("sci_fi_radio")
        assert s.radio_object_look == imgp.RADIO_OBJECT_LOOK_DEFAULT
        assert s.plate_look == imgp.PLATE_LOOK_DEFAULT
        assert (s.non_character_emblem_fallback
                == imgp.NON_CHARACTER_EMBLEM_FALLBACK_DEFAULT)

    def test_radio_object_host_byte_identical(self):
        s = vs.resolve_visual_style("sci_fi_radio")
        for aspect in ("portrait", "wide"):
            styled = imgp.build_radio_host_prompt(
                _META_BRIEF, aspect, radio_host_style="radio_object",
                vstyle=s)
            entry = imgp.build_radio_host_prompt(
                _META_BRIEF, aspect, radio_host_style="radio_object")
            assert styled == entry
            geo = (imgp.RADIO_OBJECT_GEOMETRY_WIDE if aspect == "wide"
                   else imgp.RADIO_OBJECT_GEOMETRY)
            assert ("%s, %s" % (geo, s.radio_object_look)) in styled

    def test_plate_byte_identical(self):
        s = vs.resolve_visual_style("sci_fi_radio")
        styled = imgp._compose_background_plate_prompt(
            _META_BRIEF, "mars post", style=s)
        entry = imgp._compose_background_plate_prompt(_META_BRIEF,
                                                      "mars post")
        assert styled == entry
        assert imgp.BACKGROUND_PLATE_POS_SCAFFOLD in styled

    def test_emblem_template_formats_base(self):
        s = vs.resolve_visual_style("sci_fi_radio")
        subj, src = imgp._mesh_fodder_subject_and_source(
            _META_BRIEF, None, {"beat_intent": "the door seals shut"},
            "mars post", "character_video", style=s)
        assert subj == ("a single emblematic object representing "
                        "the door seals shut")
        assert src == "non_character_emblem_fallback"
        # legacy fixture lane byte-identical
        assert imgp._mesh_fodder_subject(
            _META_BRIEF, None, {"beat_intent": "the door seals shut"},
            "mars post", "character_video") == subj

    def test_music_and_announcer_keep_radio_form_never_emblem(self):
        s = vs.resolve_visual_style("sci_fi_radio")
        for role in ("music_visual", "announcer_visual"):
            subj, src = imgp._mesh_fodder_subject_and_source(
                _META_BRIEF, None, {}, "mars post", role, style=s)
            assert subj == imgp.radio_form_from_meta(_META_BRIEF)
            assert src == "brief:radio_form"

    def test_cast_appearance_arm(self):
        s = vs.resolve_visual_style("sci_fi_radio")
        subj, src = imgp._mesh_fodder_subject_and_source(
            _META_BRIEF, _CHAR, {}, "mars post", "character_video", style=s)
        assert subj == _CHAR["appearance"]
        assert src == "cast:appearance"

    def test_geometry_has_no_look_vocabulary(self):
        for name in ("RADIO_OBJECT_GEOMETRY", "RADIO_OBJECT_GEOMETRY_WIDE",
                     "BACKGROUND_PLATE_GEOMETRY"):
            geo = getattr(imgp, name)
            assert "dramatic film lighting" not in geo
            assert "period-accurate" not in geo


# ---------------------------------------------------------------------------
# 4. Provenance on the derived image objects
# ---------------------------------------------------------------------------
class TestImageObjectProvenance:
    def _payload(self, **kw):
        lines = [
            {"speaker_role": "announcer", "char_id": "announcer",
             "beat_id": "b001", "text": "Good evening."},
            {"speaker_role": "character", "char_id": "c01",
             "beat_id": "b002", "text": "It is coming from the dust."},
        ]
        payload, _w = imgp.derive_image_prompts(
            [_CHAR], dict(_META_BRIEF), llm_fn=None, lines=lines, **kw)
        return {o["object_id"]: o for o in payload["objects"]}

    def test_scene_still_objects_stamped(self):
        objs = self._payload()
        scene = [o for o in objs.values()
                 if o["kind"] in ("scene_open", "scene_beat",
                                  "scene_character")]
        assert scene
        for o in scene:
            assert o["visual_style"] == "sci_fi_radio"
            if o["kind"] == "scene_character":
                assert o["prompt_field_source"] == "cast:appearance"
            else:
                key = helpers.open_subject_key(
                    o["role"], o["kind"] == "scene_open")
                assert (o["prompt_field_source"]
                        == "open_subjects:%s" % key)
            assert o["prompt_hash"] == imgp._content_hash(o["prompt"])

    def test_mesh_and_plate_objects_stamped(self):
        objs = self._payload(mesh_fodder_roles={"announcer_visual"})
        mesh = [o for o in objs.values() if o["kind"] == "mesh_fodder"]
        plate = [o for o in objs.values()
                 if o["kind"] == "scene_background_plate"]
        assert mesh and plate
        for o in mesh:
            assert o["visual_style"] == "sci_fi_radio"
            assert o["prompt_field_source"] == "brief:radio_form"
        for o in plate:
            assert o["visual_style"] == "sci_fi_radio"
            assert o["prompt_field_source"] == "plate_look"

    def test_still_word_objects_stamped(self):
        # Chunk C: still_word maps are pack-owned now, so the word-card
        # provenance stamps the PACK field (still_word_typography:<genre>),
        # not the retired "python:still_word_word:<genre>" literal.
        objs = self._payload(still_word_roles={"character_video"})
        cards = [o for o in objs.values() if o["source"] == "still_word"]
        assert cards
        for o in cards:
            assert o["visual_style"] == "sci_fi_radio"
            assert o["prompt_field_source"].startswith(
                "still_word_typography:")


# ---------------------------------------------------------------------------
# 5. AST guards + trace allowlist
# ---------------------------------------------------------------------------
_RD = _NODES / "_otr_video_engines" / "render_driver.py"
_IMGP = _NODES / "otr_meta_brief_image_prompt.py"


class TestAstGuards:
    def test_no_production_loads_of_the_motion_fixture(self):
        offenders = []
        for path in _NODES.rglob("*.py"):
            rel = path.relative_to(_NODES).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Load)
                        and node.id == "_LTX_MOTION_PROMPT_BY_ROLE"):
                    offenders.append(f"{rel}:{node.lineno}")
        assert not offenders, (
            f"production loads of the retired motion fixture (route through "
            f"the pack motion_registers): {offenders}")

    def test_a2_look_fixtures_read_only_in_designated_lanes(self):
        tree = ast.parse(_IMGP.read_text(encoding="utf-8"))
        names = {"RADIO_OBJECT_LOOK_DEFAULT", "PLATE_LOOK_DEFAULT",
                 "NON_CHARACTER_EMBLEM_FALLBACK_DEFAULT"}
        allowed = set()
        for node in tree.body:  # module-level fixture compositions
            if isinstance(node, ast.Assign):
                for n in ast.walk(node):
                    if isinstance(n, ast.Name) and n.id in names:
                        allowed.add(n.lineno)
        for node in ast.walk(tree):  # the one legacy lane
            if (isinstance(node, ast.FunctionDef)
                    and node.name == "_mesh_fodder_subject_and_source"):
                for n in ast.walk(node):
                    if isinstance(n, ast.Name) and n.id in names:
                        allowed.add(n.lineno)
        offenders = [
            f"{n.lineno}:{n.id}" for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            and n.id in names and n.lineno not in allowed]
        assert not offenders

    def test_trace_allowlist_carries_the_provenance_keys(self):
        src = _RD.read_text(encoding="utf-8")
        assert '"visual_style", "prompt_field_source"' in src, (
            "the run_episode trace-copy allowlist must carry the chunk-A2 "
            "provenance keys (r2 codex M5)")

    def test_builder_resolves_style_outside_any_swallow(self):
        # the get_visual_style call in build_request_from_shot must not sit
        # inside a broad try (ImportError-only shims allowed).
        tree = ast.parse(_RD.read_text(encoding="utf-8"))
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "build_request_from_shot")
        for node in ast.walk(fn):
            if not isinstance(node, ast.Try):
                continue
            broad = any(
                h.type is None or (isinstance(h.type, ast.Name)
                                   and h.type.id in ("Exception",
                                                     "BaseException"))
                for h in node.handlers)
            if not broad:
                continue
            for call in ast.walk(node):
                if (isinstance(call, ast.Call)
                        and getattr(call.func, "id",
                                    getattr(call.func, "attr", ""))
                        == "get_visual_style"):
                    raise AssertionError(
                        "get_visual_style wrapped in a broad except in "
                        "build_request_from_shot (no-fallback law)")
