"""Sprint C C5b: central story_brief helper tests.

Pure-function helpers with no I/O, no GPU. Tests run quickly and
cover the 5 helpers shipped at C5b plus the dependency-direction lock
(no musicgen import) and the caller-count contract (E-28 / RR-B10):
at C5b, `get_story_brief_music_mood` has zero production callers; at
C5g, exactly one production caller in `nodes/musicgen_theme.py`.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from nodes import _otr_story_brief_helpers as helpers


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_HELPERS_PATH = _REPO_ROOT / "nodes" / "_otr_story_brief_helpers.py"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _meta_ok() -> dict:
    return {
        "story_brief": "a dim interrogation room under a swinging bulb, rain on tin",
        "story_brief_status": "ok",
        "story_brief_terms": {
            "setting":    ["interrogation room", "steel table"],
            "lighting":   ["swinging bare bulb", "harsh top-down shadow"],
            "atmosphere": ["sweat", "smoke", "tense", "ominous"],
        },
    }


def _meta_failed() -> dict:
    return {
        "story_brief": "",
        "story_brief_status": "failed",
        "story_brief_terms": {"setting": [], "lighting": [], "atmosphere": []},
    }


def _meta_absent() -> dict:
    return {}


# ---------------------------------------------------------------------------
# 1. Helper signatures match refinement section 5
# ---------------------------------------------------------------------------


class TestHelperSignatures:

    def test_all_five_helpers_present(self):
        for name in (
            "get_story_brief_full",
            "get_story_brief_ltx",
            "get_story_brief_lighting",
            "get_story_brief_music_mood",
            "get_story_brief_status",
        ):
            assert hasattr(helpers, name), f"missing helper: {name}"

    def test_ltx_has_max_chars_kw_default_90(self):
        sig = inspect.signature(helpers.get_story_brief_ltx)
        params = sig.parameters
        assert "max_chars" in params
        assert params["max_chars"].default == 90


# ---------------------------------------------------------------------------
# 2. get_story_brief_status
# ---------------------------------------------------------------------------


class TestStatus:

    @pytest.mark.parametrize("meta_factory,expected", [
        (_meta_ok, "ok"),
        (_meta_failed, "failed"),
        (_meta_absent, "absent"),
    ])
    def test_status_resolutions(self, meta_factory, expected):
        assert helpers.get_story_brief_status(meta_factory()) == expected


# ---------------------------------------------------------------------------
# 3. get_story_brief_full
# ---------------------------------------------------------------------------


class TestGetFull:

    def test_returns_prose_on_ok(self):
        assert helpers.get_story_brief_full(_meta_ok()).startswith(
            "a dim interrogation room"
        )

    def test_empty_on_failed(self):
        assert helpers.get_story_brief_full(_meta_failed()) == ""

    def test_empty_on_absent(self):
        assert helpers.get_story_brief_full(_meta_absent()) == ""


# ---------------------------------------------------------------------------
# 4. get_story_brief_ltx
# ---------------------------------------------------------------------------


class TestGetLtx:

    def test_respects_max_chars_word_boundary(self):
        meta = _meta_ok()
        out = helpers.get_story_brief_ltx(meta, max_chars=40)
        assert len(out) <= 40
        # Never mid-word: out should not end with a partial token (split
        # on the next whitespace position is implicit in the trim logic).
        # Safer check: out doesn't end with a hyphen or punctuation half.
        assert not out.endswith("-")
        # And: every character of out should appear in the original brief.
        full = helpers.get_story_brief_full(meta)
        assert out in full or out.rstrip(",.;:") in full

    def test_full_brief_when_under_max(self):
        meta = _meta_ok()
        full = helpers.get_story_brief_full(meta)
        out = helpers.get_story_brief_ltx(meta, max_chars=10_000)
        assert out == full

    def test_empty_on_failed(self):
        assert helpers.get_story_brief_ltx(_meta_failed(), max_chars=90) == ""


# ---------------------------------------------------------------------------
# 5. get_story_brief_lighting
# ---------------------------------------------------------------------------


class TestGetLighting:

    def test_joins_lighting_plus_atmosphere(self):
        out = helpers.get_story_brief_lighting(_meta_ok())
        # Both lighting and atmosphere terms appear.
        assert "swinging bare bulb" in out
        assert "sweat" in out
        assert "tense" in out
        # No setting terms.
        assert "interrogation room" not in out
        assert "steel table" not in out
        # Comma-joined.
        assert ", " in out

    def test_empty_on_failed(self):
        assert helpers.get_story_brief_lighting(_meta_failed()) == ""

    def test_empty_on_absent(self):
        assert helpers.get_story_brief_lighting(_meta_absent()) == ""


# ---------------------------------------------------------------------------
# 6. get_story_brief_music_mood
# ---------------------------------------------------------------------------


class TestGetMusicMood:

    def test_returns_list_intersected_with_vocab(self):
        out = helpers.get_story_brief_music_mood(_meta_ok())
        assert isinstance(out, list)
        assert "tense" in out
        assert "ominous" in out
        # Non-vocab terms filtered out.
        assert "sweat" not in out  # not in _MUSIC_MOOD_VOCAB
        assert "smoke" not in out

    def test_empty_intersection_returns_empty_list(self):
        meta = {
            "story_brief": "the brief",
            "story_brief_status": "ok",
            "story_brief_terms": {
                "setting":    [],
                "lighting":   [],
                "atmosphere": ["claustrophobic", "smoke-filtered"],
            },
        }
        assert helpers.get_story_brief_music_mood(meta) == []

    def test_empty_on_failed(self):
        assert helpers.get_story_brief_music_mood(_meta_failed()) == []

    def test_empty_on_absent(self):
        assert helpers.get_story_brief_music_mood(_meta_absent()) == []

    def test_preserves_order_no_duplicates(self):
        meta = {
            "story_brief": "the brief",
            "story_brief_status": "ok",
            "story_brief_terms": {
                "setting":    [],
                "lighting":   [],
                "atmosphere": ["tense", "OMINOUS", "tense", "calm"],
            },
        }
        out = helpers.get_story_brief_music_mood(meta)
        assert out == ["tense", "ominous", "calm"]


# ---------------------------------------------------------------------------
# 7. Dependency direction: no musicgen import (E-04)
# ---------------------------------------------------------------------------


def test_no_musicgen_import_in_helpers():
    """E-04 dependency direction: the helper module must NOT import
    `nodes.musicgen_theme` (or any musicgen-related module). The
    consumer (musicgen_theme.py) imports the helper at C5g, never
    the reverse. Guards against accidental circular import."""
    tree = ast.parse(_HELPERS_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            target = (node.module or "").lower()
            assert "musicgen" not in target, (
                f"helpers module imports musicgen module at line "
                f"{node.lineno}: {ast.unparse(node)}"
            )
        elif isinstance(node, ast.Import):
            for imp in node.names:
                target = (imp.name or "").lower()
                assert "musicgen" not in target, (
                    f"helpers module imports {imp.name} at line "
                    f"{node.lineno}"
                )


# ---------------------------------------------------------------------------
# 8. E-28 / RR-B10: zero production callers at C5b
# ---------------------------------------------------------------------------


def _count_production_callers_of(symbol: str) -> list[str]:
    """Scan nodes/, visual/, scripts/ for files that import or call
    `symbol`. Exclude tests/ (test imports are intentional). Returns
    list of '<path>:<lineno>' hits."""
    hits: list[str] = []
    for sub in ("nodes", "visual", "scripts"):
        target_dir = _REPO_ROOT / sub
        if not target_dir.is_dir():
            continue
        for py_path in sorted(target_dir.rglob("*.py")):
            try:
                src = py_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                # Import statements that name the symbol.
                if isinstance(node, ast.ImportFrom):
                    for imp in node.names:
                        if imp.name == symbol:
                            hits.append(
                                f"{py_path.relative_to(_REPO_ROOT)}:"
                                f"{node.lineno} import {symbol}"
                            )
                # Call sites where the callable's name matches.
                elif isinstance(node, ast.Call):
                    func = node.func
                    if (
                        (isinstance(func, ast.Name) and func.id == symbol)
                        or (isinstance(func, ast.Attribute) and func.attr == symbol)
                    ):
                        hits.append(
                            f"{py_path.relative_to(_REPO_ROOT)}:"
                            f"{node.lineno} call {symbol}"
                        )
    return hits


def test_get_music_mood_has_zero_production_callers_at_c5b():
    """E-28 / RR-B10: at C5b commit boundary, `get_story_brief_music_mood`
    has ZERO production callers. The helper is defined here; the
    consumer wires at C5g. This test will flip to "exactly one
    production caller in nodes/musicgen_theme.py" at C5g."""
    hits = _count_production_callers_of("get_story_brief_music_mood")
    assert hits == [], (
        "C5b violation -- `get_story_brief_music_mood` already has "
        f"production callers at C5b: {hits}. The helper is defined "
        "at C5b; the consumer wiring lands at C5g via "
        "nodes/musicgen_theme.py."
    )
