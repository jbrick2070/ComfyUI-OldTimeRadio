"""tests/test_visual_styles_a1.py

Visual-style TOTAL COVERAGE chunk A1 (STAGE3_TOTAL_COVERAGE_SUBPLAN v5 FINAL,
kibitz r1-r4 converged 2026-07-05) -- schema v2 loader + sci_fi_radio
byte-identical IMAGE-LANE re-routes (portrait-look trio + announcer subjects +
open subjects + LLM instruction look).

Pins:
  1. Loader v2 matrix: exact field set on every shipped pack; dict exact
     keys; {form}/{base} template lints; mouth-vocab lint; 240-char motion
     budget; new-field non-empty rule (scene_instruction_look exempt);
     forbidden-terms lint over ALL new leaves; a v1 pack fails load LOUD
     naming the path + "upgrade to v2".
  2. EXTRACTION FIXTURES: sci_fi_radio.json's v2 fields == the Python
     fixture constants byte-for-byte (imgp looks/subjects, helpers open
     subjects, render_driver motion registers, imgp still_word maps).
  3. SEAM BYTE-IDENTITY (the A1 build gate, r2 codex CUT: seam-level string
     equality, NOT full-episode): every re-routed composer's OUTPUT under a
     default meta equals the constants-built pre-change expectation --
     portrait anchors x3, radio-host x3 dispatch arms, open subjects x3,
     the LLM instruction texts, the deterministic portrait fallback.
  4. GEOMETRY guards: *_GEOMETRY constants carry no pack look vocabulary.
  5. AST guards: no production reads of the extracted fixture constants
     outside their designated legacy lanes; every production
     _style_anchor_for_aspect / get_open_subject call passes style=.
  6. Dormant-field pin: the 4 non-default packs carry the sci-fi default
     values for every NEW field (behavior identical to the tails-only v1
     delta until chunk B authors them).
  7. FAIL-LOUD: an unknown meta["visual_style"] raises at the
     derive_image_prompts ENTRY and through each re-routed builder.
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
_PACK = _STYLES_DIR / "sci_fi_radio.json"

_ALL_IDS = ("anime", "archival_documentary", "cartoon", "paper_origami",
            "sci_fi_radio", "storybook_engraving")
_NON_DEFAULT_IDS = tuple(i for i in _ALL_IDS if i != "sci_fi_radio")

_NEW_STR_FIELDS = (
    "portrait_look", "portrait_look_talking", "portrait_instruction_look",
    "scene_instruction_look", "announcer_subject_face",
    "announcer_subject_ltx_mouth", "announcer_subject_object",
    "radio_object_look", "plate_look", "non_character_emblem_fallback",
    "still_word_title_mood_style")
_NEW_DICT_FIELDS = ("open_subjects", "motion_registers",
                    "still_word_typography", "still_word_backdrop")

_META_BRIEF = {
    "style": "a tense sci-fi thriller aboard a space station",
    "story_brief_terms": {
        "setting": ["a mars listening post", "dust-caked consoles"],
        "lighting": ["harsh sodium light", "deep shadow"],
    },
    "episode_title": "Signal in the Dust",
}
_CHAR = {"char_id": "c01", "name": "MARGOT", "gender": "female",
         "appearance": "a wiry engineer in a patched flight suit"}
_LINE = {"beat_id": "b002", "beat_intent": "she hears the signal",
         "traits": "tense", "text": "It is coming from the dust."}
_BAD_META = {"visual_style": "no_such_style"}


@pytest.fixture(autouse=True)
def _fresh_registry():
    vs._clear_caches()
    yield
    vs._clear_caches()


def _raw(style_id: str) -> dict:
    return json.loads(
        (_STYLES_DIR / f"{style_id}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Loader v2 matrix
# ---------------------------------------------------------------------------
class TestLoaderV2:
    @pytest.mark.parametrize("style_id", _ALL_IDS)
    def test_every_shipped_pack_loads_v2(self, style_id):
        style = vs.resolve_visual_style(style_id)
        assert style.schema_version == "v2"
        for name in _NEW_STR_FIELDS:
            assert isinstance(getattr(style, name), str)
        for name in _NEW_DICT_FIELDS:
            mapping = getattr(style, name)
            assert all(isinstance(v, str) and v for v in mapping.values())

    def test_dict_fields_are_immutable(self):
        style = vs.resolve_visual_style("sci_fi_radio")
        for name in _NEW_DICT_FIELDS:
            with pytest.raises(TypeError):
                getattr(style, name)["announcer"] = "x"

    def test_v1_pack_fails_loud_with_upgrade_message(self, tmp_path,
                                                     monkeypatch):
        raw = _raw("sci_fi_radio")
        for name in _NEW_STR_FIELDS + _NEW_DICT_FIELDS:
            raw.pop(name)
        raw["schema_version"] = "v1"
        scratch = tmp_path / "visual_styles"
        scratch.mkdir()
        (scratch / "sci_fi_radio.json").write_text(json.dumps(raw),
                                                   encoding="utf-8")
        monkeypatch.setattr(vs, "_VISUAL_STYLES_ROOT", scratch)
        vs._clear_caches()
        with pytest.raises(vs.VisualStyleValidationError) as ei:
            vs.list_style_ids()
        msg = str(ei.value)
        assert "upgrade to v2" in msg
        assert "sci_fi_radio.json" in msg          # names the path

    @pytest.mark.parametrize("mutate,exc_fragment", [
        (lambda d: d.update(portrait_look=""), "must be non-empty"),
        (lambda d: d.update(portrait_look=12), "must be str"),
        (lambda d: d.update(
            announcer_subject_ltx_mouth="a mouth radio, no placeholder"),
         "exactly once"),
        (lambda d: d.update(
            announcer_subject_ltx_mouth="{form} and a stray {other} brace "
                                        "with big lips and a mouth"),
         "exactly once"),
        (lambda d: d.update(
            announcer_subject_ltx_mouth="{form} a silent dial face radio"),
         "mouth-prominence"),
        (lambda d: d.update(
            non_character_emblem_fallback="an object with no placeholder"),
         "exactly once"),
        (lambda d: d["open_subjects"].pop("announcer"), "EXACTLY"),
        (lambda d: d["open_subjects"].update(extra="{form} x"), "EXACTLY"),
        (lambda d: d["open_subjects"].update(
            announcer="no form placeholder here"), "exactly once"),
        (lambda d: d["motion_registers"].pop("music_open"), "EXACTLY"),
        (lambda d: d["motion_registers"].update(
            announcer="Pans. " * 60), "motion budget"),
        (lambda d: d["still_word_typography"].pop("sci-fi"), "EXACTLY"),
        (lambda d: d["still_word_backdrop"].update(default=""),
         "non-empty string"),
        (lambda d: d.update(scene_instruction_look=7), "must be str"),
        (lambda d: d.update(forbidden_terms=["needle-fan"]),
         "appears in its own"),
    ])
    def test_fail_loud_matrix_v2(self, tmp_path, monkeypatch, mutate,
                                 exc_fragment):
        raw = _raw("sci_fi_radio")
        mutate(raw)
        scratch = tmp_path / "visual_styles"
        scratch.mkdir()
        (scratch / "sci_fi_radio.json").write_text(json.dumps(raw),
                                                   encoding="utf-8")
        monkeypatch.setattr(vs, "_VISUAL_STYLES_ROOT", scratch)
        vs._clear_caches()
        with pytest.raises(vs.VisualStyleValidationError) as ei:
            vs.list_style_ids()
        assert exc_fragment in str(ei.value)

    def test_scene_instruction_look_empty_is_legal(self):
        # r4 AG M1: the ONE exempted field; sci_fi ships "".
        assert vs.resolve_visual_style(
            "sci_fi_radio").scene_instruction_look == ""


# ---------------------------------------------------------------------------
# 2. Extraction fixtures -- pack values == Python fixture constants
# ---------------------------------------------------------------------------
class TestExtractionFixtures:
    def test_portrait_looks(self):
        s = vs.resolve_visual_style("sci_fi_radio")
        assert s.portrait_look == imgp.PORTRAIT_LOOK_DEFAULT
        assert s.portrait_look_talking == imgp.TALKING_PORTRAIT_LOOK_DEFAULT
        assert (s.portrait_instruction_look
                == imgp.PORTRAIT_INSTRUCTION_LOOK_DEFAULT)

    def test_announcer_subjects(self):
        s = vs.resolve_visual_style("sci_fi_radio")
        assert s.announcer_subject_face == imgp._RADIO_CONSOLE_FACE
        assert (s.announcer_subject_ltx_mouth
                == imgp._RADIO_CONSOLE_MOUTH.replace("%s", "{form}"))
        assert s.announcer_subject_object == imgp._RADIO_OBJECT_SUBJECT

    def test_open_subjects(self):
        s = vs.resolve_visual_style("sci_fi_radio")
        assert s.open_subjects["synthetic"] == \
            helpers.OPEN_SUBJECT_SYNTHETIC_DEFAULT
        assert s.open_subjects["announcer"] == \
            helpers.OPEN_SUBJECT_ANNOUNCER_DEFAULT
        assert s.open_subjects["default"] == \
            helpers.OPEN_SUBJECT_DEFAULT_DEFAULT

    def test_motion_registers_match_render_driver(self):
        # A2 consumes these; A1 pins the extraction so the values can never
        # drift from the live console-motion register before the re-route.
        s = vs.resolve_visual_style("sci_fi_radio")
        assert dict(s.motion_registers) == rd._LTX_MOTION_PROMPT_BY_ROLE

    def test_still_word_maps_match(self):
        # Chunk C consumes these; A1 pins the extraction.
        s = vs.resolve_visual_style("sci_fi_radio")
        assert dict(s.still_word_typography) == imgp._STILL_WORD_TYPOGRAPHY
        assert dict(s.still_word_backdrop) == imgp._STILL_WORD_BACKDROP
        assert (s.still_word_title_mood_style
                == imgp._STILL_WORD_TITLE_MOOD_STYLE)

    def test_a2_look_fields_match_their_source_literals(self):
        # radio_object_look / plate_look / emblem are consumed by chunk A2;
        # the pack values equal the literals they replaced (fixtures).
        s = vs.resolve_visual_style("sci_fi_radio")
        assert s.radio_object_look in imgp._RADIO_OBJECT_ANCHOR
        assert s.radio_object_look in imgp._RADIO_OBJECT_ANCHOR_WIDE
        assert s.plate_look in imgp.BACKGROUND_PLATE_POS_SCAFFOLD
        assert (s.non_character_emblem_fallback.replace("{base}", "%s")
                == "a single emblematic object representing %s")

    def test_legacy_anchor_fixtures_compose_from_geometry_plus_look(self):
        assert imgp.STYLE_ANCHOR == "%s, %s" % (
            imgp.PORTRAIT_GEOMETRY, imgp.PORTRAIT_LOOK_DEFAULT)
        assert imgp.STYLE_ANCHOR_WIDE == "%s, %s" % (
            imgp.WIDE_PORTRAIT_GEOMETRY, imgp.PORTRAIT_LOOK_DEFAULT)
        assert imgp.STYLE_ANCHOR_TALKING == "%s, %s" % (
            imgp.TALKING_PORTRAIT_GEOMETRY,
            imgp.TALKING_PORTRAIT_LOOK_DEFAULT)


# ---------------------------------------------------------------------------
# 3. Seam byte-identity (default meta -> pre-change output)
# ---------------------------------------------------------------------------
class TestSeamByteIdentity:
    def test_style_anchor_seams(self):
        s = vs.resolve_visual_style("sci_fi_radio")
        for talking in (False, True):
            for aspect in ("portrait", "wide"):
                styled = imgp._style_anchor_for_aspect(
                    aspect, talking=talking, style=s)
                legacy = imgp._style_anchor_for_aspect(aspect,
                                                       talking=talking)
                assert styled == legacy
        assert imgp._style_anchor_for_aspect("portrait") == imgp.STYLE_ANCHOR
        assert imgp._style_anchor_for_aspect("wide") == imgp.STYLE_ANCHOR_WIDE
        assert imgp._style_anchor_for_aspect(
            "portrait", talking=True) == imgp.STYLE_ANCHOR_TALKING

    def test_radio_host_three_arms_byte_identical(self):
        # Reconstruct each arm's pre-change prompt from the FIXTURE constants
        # (the 3A pattern) and require equality with the pack-routed output.
        form = imgp.radio_form_from_meta(_META_BRIEF)
        overt = imgp._radio_face_overtness(_META_BRIEF)
        s = vs.resolve_visual_style("sci_fi_radio")
        for aspect, obj_anchor in (("portrait", imgp._RADIO_OBJECT_ANCHOR),
                                   ("wide", imgp._RADIO_OBJECT_ANCHOR_WIDE)):
            got = imgp.build_radio_host_prompt(
                _META_BRIEF, aspect, radio_host_style="radio_object")
            core = ", ".join(
                ["%s, %s" % (form, imgp._RADIO_OBJECT_SUBJECT), obj_anchor])
            assert got.startswith(core)

            got = imgp.build_radio_host_prompt(
                _META_BRIEF, aspect, radio_host_style="console_face")
            core = ", ".join([
                "%s, %s, %s" % (form, imgp._RADIO_CONSOLE_FACE, overt),
                imgp._style_anchor_for_aspect(aspect)])
            assert got.startswith(core)

            got = imgp.build_radio_host_prompt(
                _META_BRIEF, aspect, radio_host_style="ltx_radio_mouth")
            expected = "%s, warm dramatic lighting" % ", ".join([
                "%s, %s" % (imgp._RADIO_CONSOLE_MOUTH % form, overt),
                imgp._style_anchor_for_aspect(aspect)])
            assert got == expected
        # vstyle threading == entry resolve (no drift between the lanes)
        assert imgp.build_radio_host_prompt(
            _META_BRIEF, "portrait", radio_host_style="console_face",
            vstyle=s) == imgp.build_radio_host_prompt(
            _META_BRIEF, "portrait", radio_host_style="console_face")

    def test_open_subjects_byte_identical(self):
        s = vs.resolve_visual_style("sci_fi_radio")
        form = helpers.radio_form_from_meta(_META_BRIEF)
        cases = (
            ("music_visual", True,
             "%s warming up on a table, glowing dials and tubes, "
             "warm filament glow" % form),
            ("announcer_visual", False,
             "%s in a broadcast booth, glowing warmly, lit dials and tubes"
             % form),
            ("music_visual", False,
             "%s glowing warmly, vacuum tubes and dials" % form),
        )
        for role, syn, expected in cases:
            assert helpers.get_open_subject(role, syn, _META_BRIEF,
                                            style=s) == expected
            assert helpers.get_open_subject(role, syn,
                                            _META_BRIEF) == expected

    def test_llm_instruction_texts_byte_identical(self):
        s = vs.resolve_visual_style("sci_fi_radio")
        for talking in (False, True):
            for aspect in ("portrait", "wide"):
                styled = imgp._build_char_prompt_request(
                    _CHAR, _META_BRIEF, "mars post", aspect,
                    talking=talking, style=s)
                entry = imgp._build_char_prompt_request(
                    _CHAR, _META_BRIEF, "mars post", aspect, talking=talking)
                assert styled == entry
                assert "photographic and period-consistent" in styled
        req = imgp._build_char_scene_request(_CHAR, _META_BRIEF, "mars post",
                                             _LINE, style=s)
        assert req == imgp._build_char_scene_request(
            _CHAR, _META_BRIEF, "mars post", _LINE)
        # sci_fi ships scene_instruction_look="" -> NO style_look line
        assert "style_look:" not in req

    def test_scene_instruction_look_appends_only_when_non_empty(self):
        s = vs.resolve_visual_style("sci_fi_radio")
        loaded = {f.name: getattr(s, f.name)
                  for f in type(s).__dataclass_fields__.values()}
        loaded["scene_instruction_look"] = "bold ink-wash look"
        styled = vs.VisualStyle(**loaded)
        req = imgp._build_char_scene_request(_CHAR, _META_BRIEF, "mars post",
                                             _LINE, style=styled)
        assert "style_look: bold ink-wash look\n" in req

    def test_portrait_fallback_byte_identical(self):
        s = vs.resolve_visual_style("sci_fi_radio")
        for talking in (False, True):
            styled = imgp.compose_image_prompt_fallback(
                _META_BRIEF, _CHAR, "portrait", talking=talking, style=s)
            entry = imgp.compose_image_prompt_fallback(
                _META_BRIEF, _CHAR, "portrait", talking=talking)
            assert styled == entry


# ---------------------------------------------------------------------------
# 4. Geometry guards -- no pack look vocabulary in the geometry constants
# ---------------------------------------------------------------------------
class TestGeometryGuards:
    _LOOK_VOCAB = ("period-accurate", "dramatic film lighting",
                   "warm dramatic lighting", "costume")

    @pytest.mark.parametrize("name", ["PORTRAIT_GEOMETRY",
                                      "WIDE_PORTRAIT_GEOMETRY",
                                      "TALKING_PORTRAIT_GEOMETRY"])
    def test_geometry_has_no_look_vocabulary(self, name):
        geo = getattr(imgp, name)
        for term in self._LOOK_VOCAB:
            assert term not in geo, (
                f"{name} carries pack look vocabulary {term!r} -- the "
                f"geometry-vs-look split (only LOOK moves to packs)")


# ---------------------------------------------------------------------------
# 5. AST guards
# ---------------------------------------------------------------------------
_IMGP = _NODES / "otr_meta_brief_image_prompt.py"
_HELPERS = _NODES / "_otr_story_brief_helpers.py"

_ANCHOR_NAMES = ("STYLE_ANCHOR", "STYLE_ANCHOR_WIDE", "STYLE_ANCHOR_TALKING")
_SUBJECT_FIXTURES = ("_RADIO_CONSOLE_FACE", "_RADIO_CONSOLE_MOUTH",
                     "_RADIO_OBJECT_SUBJECT")
_LOOK_DEFAULTS = ("PORTRAIT_LOOK_DEFAULT", "TALKING_PORTRAIT_LOOK_DEFAULT",
                  "PORTRAIT_INSTRUCTION_LOOK_DEFAULT")
_OPEN_DEFAULTS = ("OPEN_SUBJECT_SYNTHETIC_DEFAULT",
                  "OPEN_SUBJECT_ANNOUNCER_DEFAULT",
                  "OPEN_SUBJECT_DEFAULT_DEFAULT")


def _function_def(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


def _loads_in(node: ast.AST, names) -> list:
    return [n.id for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            and n.id in names]


class TestAstGuards:
    def test_no_production_loads_of_legacy_anchor_constants(self):
        # The composed STYLE_ANCHOR* fixtures are Store-only in production
        # (tests import them freely). Any production Load = a missed
        # re-route.
        offenders = []
        for path in _NODES.rglob("*.py"):
            rel = path.relative_to(_NODES).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Name)
                        and isinstance(node.ctx, ast.Load)
                        and node.id in _ANCHOR_NAMES):
                    offenders.append(f"{rel}:{node.lineno}:{node.id}")
        assert not offenders, (
            f"production loads of the legacy anchors (route through "
            f"_style_anchor_for_aspect(style=)): {offenders}")

    def test_radio_host_builder_reads_pack_not_fixtures(self):
        tree = ast.parse(_IMGP.read_text(encoding="utf-8"))
        fn = _function_def(tree, "build_radio_host_prompt")
        hits = _loads_in(fn, _SUBJECT_FIXTURES)
        assert not hits, (
            f"build_radio_host_prompt reads subject fixtures {hits} -- the "
            f"three dispatch arms must read the pack")

    def test_look_defaults_read_only_in_the_designated_lanes(self):
        tree = ast.parse(_IMGP.read_text(encoding="utf-8"))
        anchor_fn = _function_def(tree, "_style_anchor_for_aspect")
        allowed = set()
        for node in ast.walk(anchor_fn):
            if isinstance(node, ast.Name) and node.id in _LOOK_DEFAULTS:
                allowed.add(node.lineno)
        # module-level anchor fixture definitions are Store targets with a
        # Load of the look default on the SAME assignment -- collect them.
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = {t.id for t in node.targets
                           if isinstance(t, ast.Name)}
                if targets & set(_ANCHOR_NAMES):
                    for n in ast.walk(node):
                        if (isinstance(n, ast.Name)
                                and n.id in _LOOK_DEFAULTS):
                            allowed.add(n.lineno)
        offenders = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id in _LOOK_DEFAULTS
                    and node.lineno not in allowed):
                offenders.append(f"{node.lineno}:{node.id}")
        assert not offenders, (
            f"look-default fixtures read outside _style_anchor_for_aspect's "
            f"legacy lane / the anchor fixture definitions: {offenders}")

    def test_open_subject_defaults_read_only_in_get_open_subject(self):
        tree = ast.parse(_HELPERS.read_text(encoding="utf-8"))
        fn = _function_def(tree, "get_open_subject")
        allowed = {n.lineno for n in ast.walk(fn)
                   if isinstance(n, ast.Name) and n.id in _OPEN_DEFAULTS}
        offenders = []
        for node in ast.walk(tree):
            if (isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id in _OPEN_DEFAULTS
                    and node.lineno not in allowed):
                offenders.append(f"{node.lineno}:{node.id}")
        assert not offenders

    def test_production_anchor_callers_pass_style(self):
        # Every _style_anchor_for_aspect call in the image module passes
        # style= (the no-style lane is the tests-only legacy fixture).
        tree = ast.parse(_IMGP.read_text(encoding="utf-8"))
        offenders = []
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            name = getattr(call.func, "id",
                           getattr(call.func, "attr", ""))
            if name != "_style_anchor_for_aspect":
                continue
            if "style" not in {k.arg for k in call.keywords}:
                offenders.append(str(call.lineno))
        assert not offenders, (
            f"_style_anchor_for_aspect callers missing style=: {offenders}")

    def test_production_get_open_subject_callers_pass_style(self):
        offenders = []
        for path in _NODES.rglob("*.py"):
            rel = path.relative_to(_NODES).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for call in ast.walk(tree):
                if not isinstance(call, ast.Call):
                    continue
                name = getattr(call.func, "id",
                               getattr(call.func, "attr", ""))
                if name != "get_open_subject":
                    continue
                if "style" not in {k.arg for k in call.keywords}:
                    offenders.append(f"{rel}:{call.lineno}")
        assert not offenders, (
            f"production get_open_subject callers missing style=: "
            f"{offenders}")

    def test_char_request_builders_read_pack_instruction_look(self):
        tree = ast.parse(_IMGP.read_text(encoding="utf-8"))
        fn = _function_def(tree, "_build_char_prompt_request")
        attrs = {n.attr for n in ast.walk(fn)
                 if isinstance(n, ast.Attribute)}
        assert "portrait_instruction_look" in attrs
        fn2 = _function_def(tree, "_build_char_scene_request")
        attrs2 = {n.attr for n in ast.walk(fn2)
                  if isinstance(n, ast.Attribute)}
        assert "scene_instruction_look" in attrs2


# ---------------------------------------------------------------------------
# 6. Dormant-field pin -- RETIRED. Chunk B authored the A1/A2-consumed fields
# (delta tests in test_visual_styles_b.py); chunk C authored + wired the
# still_word fields (delta + byte-identity tests in test_visual_styles_c.py).
# Every v2 field is now consumed in the pack's voice -- nothing stays dormant.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7. Fail-loud through the re-routed entries
# ---------------------------------------------------------------------------
class TestFailLoud:
    def test_derive_image_prompts_raises_at_entry(self):
        with pytest.raises(vs.UnknownVisualStyleError):
            imgp.derive_image_prompts([_CHAR], dict(_BAD_META), llm_fn=None)

    def test_fallback_raises(self):
        with pytest.raises(vs.UnknownVisualStyleError):
            imgp.compose_image_prompt_fallback(dict(_BAD_META), _CHAR)

    def test_char_request_builders_raise(self):
        with pytest.raises(vs.UnknownVisualStyleError):
            imgp._build_char_prompt_request(_CHAR, dict(_BAD_META), "x")
        with pytest.raises(vs.UnknownVisualStyleError):
            imgp._build_char_scene_request(_CHAR, dict(_BAD_META), "x", _LINE)

    def test_missing_open_subject_key_raises_not_falls_back(self):
        # exact-key indexing on the pack map: a (hypothetically) broken
        # style object raises KeyError, never a silent default subject.
        class _Broken:
            open_subjects = {}
        with pytest.raises(KeyError):
            helpers.get_open_subject("announcer_visual", False, _META_BRIEF,
                                     style=_Broken())
