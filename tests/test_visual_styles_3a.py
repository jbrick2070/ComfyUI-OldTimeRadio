"""tests/test_visual_styles_3a.py

Multi-modal story schema STAGE 3 CHUNK 3A -- visual-style pack loader +
sci_fi_radio byte-identical chokepoint routing (STAGE3_SUBPLAN v5 FINAL,
kibitz r1-r4 + Sonnet 3-lens fan-out).

Pins:
  1. EXTRACTION FIXTURE: nodes/visual_styles/sci_fi_radio.json tails ==
     the _otr_story_brief_helpers constants, byte-for-byte.
  2. Loader: lazy (zero import-time I/O), fail-loud matrix, sweep,
     schema_version pin, style_id regex + path coordinate, load-time
     forbidden-terms lint, _clear_caches, deterministic list_style_ids
     with sci_fi_radio always a valid choice.
  3. BYTE-IDENTITY MATRIX: default meta -> composed prompts identical to
     constants-built expectations across era profiles + still kinds.
  4. FAIL-LOUD: meta["visual_style"]=unknown raises UnknownVisualStyleError
     through every routed composer entry (no swallow).
  5. AST GUARDS: (a) no production module reads the 4 tail constants
     outside their definitions in _otr_story_brief_helpers.py; (b) no
     visual-prompt seam wraps a style call in a bare except-Exception
     (ImportError-only shims allowed); (c) production get_era_tail callers
     pass style=.
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
_PACK = _NODES / "visual_styles" / "sci_fi_radio.json"

_TAIL_NAMES = ("ERA_TAIL_DEFAULT", "STYLE_TAIL_DEFAULT",
               "IMAGE_GRADE_TAIL", "RADIO_BROADCAST_TAIL")


@pytest.fixture(autouse=True)
def _fresh_registry():
    vs._clear_caches()
    yield
    vs._clear_caches()


# ---------------------------------------------------------------------------
# 1. Extraction fixture -- byte identity JSON <-> constants
# ---------------------------------------------------------------------------
class TestExtractionFixture:
    def test_pack_tails_byte_identical_to_constants(self):
        raw = json.loads(_PACK.read_text(encoding="utf-8"))
        assert raw["positive_tail"] == helpers.STYLE_TAIL_DEFAULT
        assert raw["image_grade_tail"] == helpers.IMAGE_GRADE_TAIL
        assert raw["broadcast_tail"] == helpers.RADIO_BROADCAST_TAIL
        assert raw["era_tail"] == helpers.ERA_TAIL_DEFAULT
        assert raw["allow_radio_tails"] is True
        assert raw["forbidden_terms"] == []

    def test_loaded_default_matches_constants(self):
        style = vs.resolve_visual_style("sci_fi_radio")
        assert style.positive_tail == helpers.STYLE_TAIL_DEFAULT
        assert style.image_grade_tail == helpers.IMAGE_GRADE_TAIL
        assert style.broadcast_tail == helpers.RADIO_BROADCAST_TAIL
        assert style.era_tail == helpers.ERA_TAIL_DEFAULT


# ---------------------------------------------------------------------------
# 2. Loader contract
# ---------------------------------------------------------------------------
class TestLoader:
    def test_lazy_no_import_time_io(self, monkeypatch):
        # Re-import the module fresh with a read_text sentinel: import must
        # not touch the disk (mirror of the Stage-2 lazy pin). The ORIGINAL
        # module object is restored afterwards -- a lingering fresh copy
        # would split the exception-class identity for every later test
        # (helpers resolves the module lazily by name).
        import importlib
        import sys
        calls = []
        real = Path.read_text

        def _spy(self, *a, **k):
            if "visual_styles" in str(self):
                calls.append(str(self))
            return real(self, *a, **k)

        monkeypatch.setattr(Path, "read_text", _spy)
        original = sys.modules.get("nodes._otr_visual_styles")
        try:
            sys.modules.pop("nodes._otr_visual_styles", None)
            mod = importlib.import_module("nodes._otr_visual_styles")
            assert calls == [], f"import-time I/O detected: {calls}"
            # First resolve triggers the load.
            mod._clear_caches()
            mod.resolve_visual_style("sci_fi_radio")
            assert calls, "expected the first resolve to read the packs"
        finally:
            if original is not None:
                sys.modules["nodes._otr_visual_styles"] = original

    def test_default_resolution_paths(self):
        assert vs.get_visual_style({}).style_id == "sci_fi_radio"
        assert vs.get_visual_style(None).style_id == "sci_fi_radio"
        assert vs.get_visual_style(
            {"visual_style": ""}).style_id == "sci_fi_radio"
        assert vs.get_visual_style(
            {"visual_style": "sci_fi_radio"}).style_id == "sci_fi_radio"

    def test_unknown_id_hard_error(self):
        with pytest.raises(vs.UnknownVisualStyleError):
            vs.resolve_visual_style("no_such_style")
        with pytest.raises(vs.UnknownVisualStyleError):
            vs.get_visual_style({"visual_style": "no_such_style"})

    def test_list_ids_deterministic_and_default_valid(self):
        ids = vs.list_style_ids()
        assert ids == vs.list_style_ids()
        assert "sci_fi_radio" in ids
        assert list(ids) == sorted(ids), (
            "filename-order == sorted order for flat lowercase ids")

    def test_clear_caches_reloads(self):
        first = vs.resolve_visual_style("sci_fi_radio")
        vs._clear_caches()
        second = vs.resolve_visual_style("sci_fi_radio")
        assert first == second
        assert first is not second

    @pytest.mark.parametrize("mutate,exc_fragment", [
        (lambda d: d.update(style_id="wrong_name"), "does not match"),
        (lambda d: d.update(schema_version="v99"), "schema_version"),
        (lambda d: d.update(extra_key=1), "unknown key"),
        (lambda d: d.pop("era_tail"), "missing required"),
        (lambda d: d.update(style_id="Bad-Id!"), "must match"),
        (lambda d: d.update(allow_radio_tails="yes"), "must be a bool"),
        (lambda d: d.update(forbidden_terms=["film grain"]),
         "appears in its own"),
    ])
    def test_fail_loud_matrix(self, tmp_path, monkeypatch, mutate,
                              exc_fragment):
        # Build a scratch styles dir holding ONLY the mutated pack at the
        # canonical sci_fi_radio.json path -- each mutation trips its own
        # validation rule (the path-coordinate rule fires only for the
        # renamed-style_id case, by design).
        raw = json.loads(_PACK.read_text(encoding="utf-8"))
        scratch = tmp_path / "visual_styles"
        scratch.mkdir()
        bad = dict(raw)
        mutate(bad)
        (scratch / "sci_fi_radio.json").write_text(
            json.dumps(bad), encoding="utf-8")
        monkeypatch.setattr(vs, "_VISUAL_STYLES_ROOT", scratch)
        vs._clear_caches()
        with pytest.raises(vs.VisualStyleValidationError) as ei:
            vs.list_style_ids()
        assert exc_fragment in str(ei.value)

    def test_malformed_json_and_stray_files(self, tmp_path, monkeypatch):
        scratch = tmp_path / "visual_styles"
        scratch.mkdir()
        (scratch / "sci_fi_radio.json").write_text("{not json",
                                                   encoding="utf-8")
        monkeypatch.setattr(vs, "_VISUAL_STYLES_ROOT", scratch)
        vs._clear_caches()
        with pytest.raises(vs.VisualStyleValidationError):
            vs.list_style_ids()
        (scratch / "sci_fi_radio.json").write_text(
            _PACK.read_text(encoding="utf-8"), encoding="utf-8")
        (scratch / "notes.txt").write_text("x", encoding="utf-8")
        vs._clear_caches()
        with pytest.raises(vs.VisualStyleValidationError):
            vs.list_style_ids()

    def test_missing_default_pack_hard_error(self, tmp_path, monkeypatch):
        raw = json.loads(_PACK.read_text(encoding="utf-8"))
        raw["style_id"] = "other_style"
        scratch = tmp_path / "visual_styles"
        scratch.mkdir()
        (scratch / "other_style.json").write_text(json.dumps(raw),
                                                  encoding="utf-8")
        monkeypatch.setattr(vs, "_VISUAL_STYLES_ROOT", scratch)
        vs._clear_caches()
        with pytest.raises(vs.VisualStyleValidationError) as ei:
            vs.list_style_ids()
        assert "default style" in str(ei.value)


# ---------------------------------------------------------------------------
# 3. Byte-identity matrix (default meta -> unchanged output)
# ---------------------------------------------------------------------------
_META_EMPTY: dict = {}
_META_BRIEF = {
    "story_brief_terms": {
        "setting": ["a mars listening post", "dust-caked consoles"],
        "lighting": ["harsh sodium light", "deep shadow"],
    },
    "episode_title": "Signal in the Dust",
}


class TestByteIdentityMatrix:
    @pytest.mark.parametrize("profile", ["full", "still", "portrait"])
    @pytest.mark.parametrize("meta", [_META_EMPTY, _META_BRIEF])
    def test_get_era_tail_matrix(self, profile, meta):
        legacy = helpers.get_era_tail(meta, profile=profile)
        styled = helpers.get_era_tail(
            meta, profile=profile,
            style=vs.resolve_visual_style("sci_fi_radio"))
        via_default = helpers.get_era_tail(meta, profile=profile,
                                           style=None)
        assert styled == legacy
        assert via_default == legacy

    @pytest.mark.parametrize("kind,role", [
        ("scene_open", "music_visual"),
        ("scene_beat", "announcer_visual"),
        ("scene_character", "character_video"),
        ("portrait", "character_video"),
    ])
    @pytest.mark.parametrize("meta", [_META_EMPTY, _META_BRIEF])
    def test_compose_still_prompt_matrix(self, kind, role, meta):
        ce = {"appearance": "a wiry engineer in a patched flight suit"}
        out = helpers.compose_still_prompt(
            meta, kind=kind, role=role, char_entry=ce)
        # Reconstruct the expectation from the CONSTANTS (the fixture).
        expected = helpers.compose_still_prompt(
            meta, kind=kind, role=role, char_entry=ce,
            style=vs.resolve_visual_style("sci_fi_radio"))
        assert out == expected
        assert helpers.STYLE_TAIL_DEFAULT in out
        if kind not in ("portrait",):
            assert helpers.IMAGE_GRADE_TAIL in out
        if kind in ("scene_open", "scene_beat"):
            assert helpers.RADIO_BROADCAST_TAIL in out
        if kind == "scene_character":
            assert helpers.RADIO_BROADCAST_TAIL not in out

    @pytest.mark.parametrize("style_tail", [True, False])
    @pytest.mark.parametrize("era_profile", ["full", "still", "portrait"])
    def test_finish_visual_prompt_matrix(self, style_tail, era_profile):
        out = helpers.finish_visual_prompt(
            _META_BRIEF, "a glowing tube radio on a bench",
            style_tail=style_tail, era_profile=era_profile)
        expected_tail_present = helpers.STYLE_TAIL_DEFAULT in out
        assert expected_tail_present == style_tail
        assert out.startswith("a glowing tube radio on a bench")

    def test_finish_visual_prompt_empty_prompt_short_circuits(self):
        # Empty prompt returns '' BEFORE style resolution -- no raise even
        # with an unknown style (documented short-circuit).
        assert helpers.finish_visual_prompt(
            {"visual_style": "no_such_style"}, "") == ""

    def test_radio_host_and_plate_and_mesh_unchanged(self):
        host = imgp.build_radio_host_prompt(_META_BRIEF, "portrait",
                                            radio_host_style="radio_object")
        assert helpers.IMAGE_GRADE_TAIL in host
        plate = imgp._compose_background_plate_prompt(_META_BRIEF, "mars post")
        assert helpers.IMAGE_GRADE_TAIL in plate
        assert plate.endswith(helpers.NO_TEXT_CLAUSE)
        mesh = imgp._compose_mesh_fodder_prompt(
            _META_BRIEF, None, {}, "mars post", "music_visual")
        assert imgp.MESH_FODDER_POS_SCAFFOLD in mesh


# ---------------------------------------------------------------------------
# 4. Fail-loud through the composer entries (no swallow)
# ---------------------------------------------------------------------------
_BAD_META = {"visual_style": "no_such_style"}


class TestFailLoudThroughComposers:
    def test_compose_still_prompt_raises(self):
        with pytest.raises(vs.UnknownVisualStyleError):
            helpers.compose_still_prompt(_BAD_META, kind="scene_beat",
                                         role="announcer_visual")

    def test_finish_visual_prompt_raises(self):
        with pytest.raises(vs.UnknownVisualStyleError):
            helpers.finish_visual_prompt(_BAD_META, "a subject")

    def test_radio_host_raises(self):
        with pytest.raises(vs.UnknownVisualStyleError):
            imgp.build_radio_host_prompt(_BAD_META, "portrait",
                                         radio_host_style="radio_object")

    def test_background_plate_raises(self):
        with pytest.raises(vs.UnknownVisualStyleError):
            imgp._compose_background_plate_prompt(_BAD_META, "a setting")

    def test_mesh_fodder_raises(self):
        with pytest.raises(vs.UnknownVisualStyleError):
            imgp._compose_mesh_fodder_prompt(
                _BAD_META, None, {}, "a setting", "music_visual")

    def test_still_word_raises(self):
        with pytest.raises(vs.UnknownVisualStyleError):
            imgp.compose_still_word_prompt(
                {**_BAD_META, "episode_title": "T"},
                "character_video", {"text": "We hear it in the static."})


# ---------------------------------------------------------------------------
# 5. AST guards
# ---------------------------------------------------------------------------
_PROMPT_MODULES = [
    _NODES / "otr_meta_brief_image_prompt.py",
    _NODES / "_otr_story_brief_helpers.py",
    _NODES / "_otr_video_engines" / "render_driver.py",
    _NODES / "_otr_video_engines" / "motion_common.py",
]

_STYLE_CALL_NAMES = {"finish_visual_prompt", "get_era_tail",
                     "get_visual_style", "compose_still_prompt",
                     "_resolve_style", "resolve_visual_style"}


def _production_py_files():
    for p in _NODES.rglob("*.py"):
        yield p


class TestAstGuards:
    def test_no_production_reads_of_tail_constants(self):
        offenders = []
        for path in _production_py_files():
            rel = path.relative_to(_NODES).as_posix()
            if rel == "_otr_story_brief_helpers.py":
                continue  # definitions + the legacy no-style lane
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in _TAIL_NAMES:
                    offenders.append(f"{rel}:{node.lineno}:{node.id}")
                elif isinstance(node, ast.ImportFrom):
                    # loop var named `imp` NOT `alias` -- the B7 forbidden
                    # sweep flags `alias` as a runtime marker (CW-6 gotcha).
                    for imp in node.names:
                        if imp.name in _TAIL_NAMES:
                            offenders.append(
                                f"{rel}:{node.lineno}:import {imp.name}")
        assert not offenders, (
            "production reads of the tail constants (must route through the "
            f"visual-style pack): {offenders}"
        )

    def test_no_bare_except_around_style_calls(self):
        offenders = []
        for path in _PROMPT_MODULES:
            rel = path.name
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Try):
                    continue
                broad = any(
                    h.type is None
                    or (isinstance(h.type, ast.Name)
                        and h.type.id in ("Exception", "BaseException"))
                    for h in node.handlers
                )
                if not broad:
                    continue
                for call in ast.walk(node):
                    if isinstance(call, ast.Call):
                        name = getattr(call.func, "id",
                                       getattr(call.func, "attr", ""))
                        if name in _STYLE_CALL_NAMES:
                            offenders.append(
                                f"{rel}:{node.lineno} wraps {name}() in a "
                                f"broad except")
        assert not offenders, (
            "style calls must never be swallowed (no-fallback law): "
            f"{offenders}"
        )

    def test_production_get_era_tail_callers_pass_style(self):
        # Every production get_era_tail call outside the helpers module
        # itself must pass style= (the legacy no-style lane is tests-only).
        offenders = []
        for path in _production_py_files():
            rel = path.relative_to(_NODES).as_posix()
            if rel in ("_otr_story_brief_helpers.py",):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for call in ast.walk(tree):
                if not isinstance(call, ast.Call):
                    continue
                name = getattr(call.func, "id",
                               getattr(call.func, "attr", ""))
                if name != "get_era_tail":
                    continue
                kwargs = {k.arg for k in call.keywords}
                if "style" not in kwargs:
                    offenders.append(f"{rel}:{call.lineno}")
        assert not offenders, (
            f"production get_era_tail callers missing style=: {offenders}"
        )
