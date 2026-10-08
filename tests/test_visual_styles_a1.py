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
     fixture constants byte-for-byte (helpers open subjects; the motion
     registers are pinned byte for byte in tests/test_visual_styles_a2.py).
     The image-prompt module keeps no fixture: it reads the pack alone.
  3. SEAM BYTE-IDENTITY (the A1 build gate, r2 codex CUT: seam-level string
     equality, NOT full-episode): every re-routed composer's OUTPUT under a
     default meta equals its pack-composed expectation -- radio-host x3
     dispatch arms, open subjects x3, the LLM instruction texts, the
     deterministic portrait fallback.
  4. GEOMETRY guards: *_GEOMETRY constants carry no pack look vocabulary.
  5. AST guards: no production reads of the open-subject defaults outside
     get_open_subject's legacy lane; every production get_open_subject call
     passes style=.
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

_REPO = Path(__file__).resolve().parent.parent
_NODES = _REPO / "nodes"
_STYLES_DIR = _NODES / "visual_styles"
_PACK = _STYLES_DIR / "sci_fi_radio.json"

_ALL_IDS = ("anime", "archival_documentary", "cartoon", "paper_origami",
            "recur_frac", "sci_fi_radio", "shakespeare_stage_realism",
            "storybook_engraving", "video_art")
_NON_DEFAULT_IDS = tuple(i for i in _ALL_IDS if i != "sci_fi_radio")

_NEW_STR_FIELDS = (
    "portrait_look", "portrait_instruction_look",
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
    def test_open_subjects(self):
        s = vs.resolve_visual_style("sci_fi_radio")
        assert s.open_subjects["synthetic"] == \
            helpers.OPEN_SUBJECT_SYNTHETIC_DEFAULT
        assert s.open_subjects["announcer"] == \
            helpers.OPEN_SUBJECT_ANNOUNCER_DEFAULT
        assert s.open_subjects["default"] == \
            helpers.OPEN_SUBJECT_DEFAULT_DEFAULT


# ---------------------------------------------------------------------------
# 3. Seam byte-identity (default meta -> pre-change output)
# ---------------------------------------------------------------------------
class TestSeamByteIdentity:
    def test_radio_host_three_arms_byte_identical(self):
        # Reconstruct each arm's prompt from the PACK fields (the 3A pattern)
        # and require equality with the routed output.
        form = imgp.radio_form_from_meta(_META_BRIEF)
        overt = imgp._radio_face_overtness(_META_BRIEF)
        s = vs.resolve_visual_style("sci_fi_radio")
        for aspect, obj_geometry in (("portrait", imgp.RADIO_OBJECT_GEOMETRY),
                                     ("wide", imgp.RADIO_OBJECT_GEOMETRY_WIDE)):
            got = imgp.build_radio_host_prompt(
                _META_BRIEF, aspect, radio_host_style="radio_object")
            core = ", ".join(
                ["%s, %s" % (form, s.announcer_subject_object),
                 "%s, %s" % (obj_geometry, s.radio_object_look)])
            assert got.startswith(core)

            got = imgp.build_radio_host_prompt(
                _META_BRIEF, aspect, radio_host_style="console_face")
            core = ", ".join([
                "%s, %s, %s" % (form, s.announcer_subject_face, overt),
                imgp._style_anchor_for_aspect(aspect, style=s)])
            assert got.startswith(core)

            got = imgp.build_radio_host_prompt(
                _META_BRIEF, aspect, radio_host_style="ltx_radio_mouth")
            expected = "%s, warm dramatic lighting" % ", ".join([
                "%s, %s" % (s.announcer_subject_ltx_mouth.format(form=form),
                            overt),
                imgp._style_anchor_for_aspect(aspect, style=s)])
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
        for aspect in ("portrait", "wide"):
            styled = imgp._build_char_prompt_request(
                _CHAR, _META_BRIEF, "mars post", aspect, style=s)
            entry = imgp._build_char_prompt_request(
                _CHAR, _META_BRIEF, "mars post", aspect)
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
        styled = imgp.compose_image_prompt_fallback(
            _META_BRIEF, _CHAR, "portrait", style=s)
        entry = imgp.compose_image_prompt_fallback(
            _META_BRIEF, _CHAR, "portrait")
        assert styled == entry


# ---------------------------------------------------------------------------
# 4. Geometry guards -- no pack look vocabulary in the geometry constants
# ---------------------------------------------------------------------------
class TestGeometryGuards:
    _LOOK_VOCAB = ("period-accurate", "dramatic film lighting",
                   "warm dramatic lighting", "costume")

    @pytest.mark.parametrize("name", ["PORTRAIT_GEOMETRY",
                                      "WIDE_PORTRAIT_GEOMETRY"])
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

_OPEN_DEFAULTS = ("OPEN_SUBJECT_SYNTHETIC_DEFAULT",
                  "OPEN_SUBJECT_ANNOUNCER_DEFAULT",
                  "OPEN_SUBJECT_DEFAULT_DEFAULT")


def _function_def(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name} not found")


class TestAstGuards:
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
