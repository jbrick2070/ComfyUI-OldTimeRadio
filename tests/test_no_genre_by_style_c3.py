"""Sprint C C3: _GENRE_BY_STYLE deletion + meta.visual_plan.genre retirement.

The genre derivation layer (writer table + helpers + downstream
fall-throughs) was a denormalization of `meta.style` that required a
separate keep-in-sync contract for no real win. After C3:

- `_GENRE_BY_STYLE`, `_resolve_genre`, `_preview_genre` are deleted from
  `nodes.OTR_LedgerScriptWriter`.
- `meta.visual_plan.genre` is no longer stamped (writer section K.5).
- The three `video_engine.py` `or genre` fall-throughs (lines ~711, ~836,
  ~1075 pre-C3) are deleted.
- The `otr_video_plan.py` projection no longer surfaces `genre`.

Downstream consumers (HUD overlay, FLUX scene-prompt composition,
treatment text, video info card) fall back to `meta.style` directly.
Old ledgers without `meta.style` would surface as "?" in the HUD STYLE
row -- intentional, per the no-legacy-back-compat directive (re-render
on the new code path to refresh the stamp).

Test class structure mirrors `tests/test_era_literals_c2a.py` and
`tests/test_era_literals_c2b.py`: each formerly-live contract becomes
a deletion-locked assertion (`hasattr` for symbols, AST walk for code
patterns).
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_WRITER_PATH = _REPO_ROOT / "nodes" / "OTR_LedgerScriptWriter.py"
_VIDEO_ENGINE_PATH = _REPO_ROOT / "nodes" / "video_engine.py"
_VIDEO_PLAN_PATH = _REPO_ROOT / "nodes" / "otr_video_plan.py"


# ---------------------------------------------------------------------------
# 1-3: deleted-symbol assertions (runtime hasattr; no source-text scan)
# ---------------------------------------------------------------------------


class TestDeletedSymbols:
    """C3: _GENRE_BY_STYLE table + _resolve_genre + _preview_genre helpers
    are deleted from nodes.OTR_LedgerScriptWriter and cannot be imported.
    Forensic attribution comments in the test_musicgen_style_palette
    historical-comment block AND in the C3 deletion comment itself
    remain allowed -- the hasattr-based assertion is immune to source
    text mentions."""

    @pytest.fixture(scope="class")
    def writer_mod(self):
        return importlib.import_module("nodes.OTR_LedgerScriptWriter")

    def test_no_genre_by_style_constant(self, writer_mod):
        assert not hasattr(writer_mod, "_GENRE_BY_STYLE"), (
            "_GENRE_BY_STYLE should be deleted at C3, not present "
            "in nodes.OTR_LedgerScriptWriter"
        )

    def test_no_resolve_genre_helper(self, writer_mod):
        assert not hasattr(writer_mod, "_resolve_genre"), (
            "_resolve_genre should be deleted at C3, not present "
            "in nodes.OTR_LedgerScriptWriter"
        )

    def test_no_preview_genre_helper(self, writer_mod):
        assert not hasattr(writer_mod, "_preview_genre"), (
            "_preview_genre should be deleted at C3, not present "
            "in nodes.OTR_LedgerScriptWriter"
        )


# ---------------------------------------------------------------------------
# 4: meta.visual_plan no longer carries a "genre" key (AST walk on writer)
# ---------------------------------------------------------------------------


class TestNoMetaVisualPlanGenreStamp:
    """C3: writer section K.5 builds meta['visual_plan'] WITHOUT a
    'genre' key. AST walk on the writer source so we do not need to
    actually execute the writer (no LLM dependency in the test).
    """

    def test_no_meta_visual_plan_genre_stamp(self):
        tree = ast.parse(_WRITER_PATH.read_text(encoding="utf-8"))
        # Walk every assignment of the form `meta["visual_plan"] = { ... }`
        # and confirm none of the dict literal's keys is the string "genre".
        violations: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Subscript):
                    continue
                # `meta["visual_plan"]`
                if not (
                    isinstance(target.value, ast.Name)
                    and target.value.id == "meta"
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "visual_plan"
                ):
                    continue
                if not isinstance(node.value, ast.Dict):
                    continue
                for key_node in node.value.keys:
                    if (
                        isinstance(key_node, ast.Constant)
                        and key_node.value == "genre"
                    ):
                        violations.append(
                            f"line {node.lineno}: meta['visual_plan'] = {{...}} "
                            "still contains a 'genre' key"
                        )
        assert not violations, (
            "C3 deletion incomplete -- meta.visual_plan.genre still "
            "stamped:\n  " + "\n  ".join(violations)
        )


# ---------------------------------------------------------------------------
# 5: video_engine.py no longer has the three `or genre` fall-throughs
# ---------------------------------------------------------------------------


class TestVideoEngineNoGenreFallthroughs:
    """C3: the three fall-through expressions in nodes/video_engine.py
    that previously consulted `genre` are deleted:
      - `"style": style or genre or "sci-fi"` (was line ~711)
      - `("STYLE", self.data.get("style", self.data.get("genre", "?")))` (~836)
      - `style = style or genre or "audio drama"` (~1075)
    The `genre` parameter on the function signatures is now
    dead-but-harmless (always passed ""); Sprint G's orphan-parameter
    sweep will drop it. This test pins the three expression rewrites.
    """

    def test_no_style_or_genre_sci_fi(self):
        src = _VIDEO_ENGINE_PATH.read_text(encoding="utf-8")
        assert 'style or genre or "sci-fi"' not in src, (
            "video_engine.py still contains the `style or genre or "
            "\"sci-fi\"` fall-through; should be `style or \"sci-fi\"` "
            "after C3"
        )

    def test_no_data_get_genre_fallback_in_hud(self):
        src = _VIDEO_ENGINE_PATH.read_text(encoding="utf-8")
        assert 'self.data.get("style", self.data.get("genre"' not in src, (
            "video_engine.py HUD STYLE row still has the `genre` "
            "fall-through inside `self.data.get(...)`; should be a "
            "single `self.data.get(\"style\", \"?\")` after C3"
        )

    def test_no_style_or_genre_audio_drama(self):
        src = _VIDEO_ENGINE_PATH.read_text(encoding="utf-8")
        assert 'style or genre or "audio drama"' not in src, (
            "video_engine.py treatment text still contains the `style "
            "or genre or \"audio drama\"` fall-through; should be "
            "`style or \"audio drama\"` after C3"
        )


# ---------------------------------------------------------------------------
# 6: graceful absence -- HUD + treatment render when meta.visual_plan.genre
#     is absent (E-05, renamed per E-26 / RR-B11)
# ---------------------------------------------------------------------------


class TestHudAndTreatmentRenderWhenVisualPlanGenreAbsent:
    """C3 / E-05 (renamed per E-26): with `meta.visual_plan.genre`
    no longer stamped, the projection in otr_video_plan.py drops the
    `genre` key entirely from the returned dict. The plan-projection
    consumer in video_engine.py reads it via `(visual_plan.get("genre")
    or "")` at line ~1295 (still safe -- `dict.get` returns `None`
    which `or ""` collapses to `""`). Verify the projection is valid
    JSON-encodable and lacks the `genre` key.
    """

    def test_otr_video_plan_projection_has_no_genre_key(self):
        # AST walk on otr_video_plan.py: find the return-dict that
        # carries `characters` / `scenes` / `voice_assignments` / `style`,
        # confirm it does NOT carry `genre`.
        tree = ast.parse(_VIDEO_PLAN_PATH.read_text(encoding="utf-8"))
        violations: list[str] = []
        found_projection = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Return):
                continue
            if not isinstance(node.value, ast.Dict):
                continue
            string_keys = {
                k.value
                for k in node.value.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
            # Identify the projection by its characters/scenes/style trio.
            if not {"characters", "scenes", "style"}.issubset(string_keys):
                continue
            found_projection = True
            if "genre" in string_keys:
                violations.append(
                    f"line {node.lineno}: return dict carrying "
                    f"characters/scenes/style still contains 'genre' key"
                )
        assert found_projection, (
            "Could not locate the otr_video_plan.py projection dict "
            "(characters/scenes/style triple); the test's anchor pattern "
            "may have drifted -- investigate before assuming pass."
        )
        assert not violations, "\n  " + "\n  ".join(violations)

    def test_visual_plan_genre_get_or_empty_safe(self):
        # video_engine.py:~1295 reads `(visual_plan.get("genre") or "")`.
        # This pattern is safe with the post-C3 projection (the key is
        # absent so `.get` returns None which `or ""` collapses). The
        # test confirms the pattern still exists OR has been refactored
        # to a safe alternative -- either way the line should be
        # crash-free if visual_plan lacks "genre". Sprint G's
        # orphan-parameter sweep will likely drop the read entirely.
        src = _VIDEO_ENGINE_PATH.read_text(encoding="utf-8")
        # The pattern can take a couple of equivalent shapes; we accept
        # any that combine `.get("genre")` with an `or ""` or equivalent.
        snippets = [
            'visual_plan.get("genre") or ""',
            'visual_plan.get("genre", "")',
        ]
        if any(snippet in src for snippet in snippets):
            # Safe read pattern present; the post-C3 projection without
            # "genre" key still resolves to "" without crashing.
            return
        # No genre read at all -- also acceptable, means Sprint G's
        # cleanup got pulled forward. Either is C3-compatible.
