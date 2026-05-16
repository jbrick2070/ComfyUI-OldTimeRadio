"""Sprint C C5g: MusicGen reads meta.story_brief via mood-vocab helper.

C5g scope was revised at execution time per operator directive 2026-05-15
(Option 2): pytest-only structural wiring ships in this commit; the
audio C7 baseline reset b3sum captures (E-12, E-16) are deferred to
Sprint A. The three runtime-gated tests below (`TestRuntimeOnly`) ship
with `@pytest.mark.skipif(not OTR_REGRESSION_RUNTIME)` so Sprint A's
empirical-verification pass can unlock them once the fixture files
are captured on the GPU machine.

Active pytest coverage (no GPU):
  1. helper imports at module level
  2. status helper imported (E-07 observability)
  3. mood prefix prepended on status=ok with vocab terms
  4. prompt unchanged on status=failed
  5. prompt unchanged on status=absent (legacy ledger)
  6. prompt unchanged on empty-intersection (atmosphere outside vocab)
  7. E-06 meta-threading canary -- in-vocab term reaches prompt,
     out-of-vocab unique-token is filtered out (helper-correct)
  8. E-07 log carries story_brief_status
  9. no new LLM call added (MusicGen stays an audio model call site)
 10. E-28 exactly-one production caller of get_story_brief_music_mood
 11. E-28 strict: the one caller is `nodes/musicgen_theme.py`

Runtime-gated (Sprint A unlocks):
 A1. test_audio_c7_new_baseline_captured -- fixture file present + non-empty
 A2. test_c5g_audio_matches_legacy_when_brief_absent -- E-16 isolation
 A3. test_audio_c7_byte_identical_c5g_new_baseline -- ongoing canary
"""

from __future__ import annotations

import ast
import logging
import os
import sys
import types
from pathlib import Path

import pytest


# musicgen_theme imports torch / numpy / comfy modules at module load.
# Pure unit tests of the new helper need none of that; stub the heavy
# deps so the import succeeds in the sandbox.
for _stub_name in ("folder_paths",):
    if _stub_name not in sys.modules:
        _stub = types.ModuleType(_stub_name)
        _stub.get_output_directory = lambda: str(Path.cwd())  # noqa: E731
        sys.modules[_stub_name] = _stub

from nodes import musicgen_theme as mgt


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_MGT_PATH = _REPO_ROOT / "nodes" / "musicgen_theme.py"


_CANARY = "zebra_lantern_atmosphere_731"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _meta_ok(*, atmosphere: list[str] | None = None) -> dict:
    return {
        "story_brief": "the brief prose",
        "story_brief_status": "ok",
        "story_brief_terms": {
            "setting": [], "lighting": [],
            "atmosphere": list(atmosphere or ["tense", "ominous"]),
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
# 1-2. Helper imports at module level
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mgt_ast() -> ast.AST:
    return ast.parse(_MGT_PATH.read_text(encoding="utf-8"))


def test_musicgen_imports_music_mood_helper(mgt_ast):
    """Helper imported at module level, not hot-path."""
    found = False
    for node in mgt_ast.body:
        if isinstance(node, ast.ImportFrom) and node.module == "_otr_story_brief_helpers":
            for imp in node.names:
                if imp.name == "get_story_brief_music_mood":
                    found = True
    assert found, (
        "nodes/musicgen_theme.py does not import get_story_brief_music_mood "
        "at module level"
    )


def test_musicgen_imports_status_helper(mgt_ast):
    found = False
    for node in mgt_ast.body:
        if isinstance(node, ast.ImportFrom) and node.module == "_otr_story_brief_helpers":
            for imp in node.names:
                if imp.name == "get_story_brief_status":
                    found = True
    assert found, (
        "nodes/musicgen_theme.py does not import get_story_brief_status "
        "at module level"
    )


# ---------------------------------------------------------------------------
# 3-6. Mood prefix behavior
# ---------------------------------------------------------------------------


_LEGACY_PROMPT = "ambient orchestral pad, slow strings, low cellos"


class TestMoodPrefix:

    def test_prepends_mood_when_status_ok(self):
        meta = _meta_ok(atmosphere=["tense", "ominous"])
        out, status, terms = mgt._apply_story_brief_mood_prefix(_LEGACY_PROMPT, meta)
        assert out.startswith("tense, ominous, "), (
            f"prompt does not start with expected mood prefix; got: {out[:80]}"
        )
        assert _LEGACY_PROMPT in out
        assert status == "ok"
        assert terms == ["tense", "ominous"]

    def test_unchanged_when_status_failed(self):
        out, status, terms = mgt._apply_story_brief_mood_prefix(
            _LEGACY_PROMPT, _meta_failed(),
        )
        assert out == _LEGACY_PROMPT, (
            f"prompt changed despite status=failed; got: {out[:80]}"
        )
        assert status == "failed"
        assert terms == []

    def test_unchanged_when_status_absent(self):
        out, status, _terms = mgt._apply_story_brief_mood_prefix(
            _LEGACY_PROMPT, _meta_absent(),
        )
        assert out == _LEGACY_PROMPT
        assert status == "absent"

    def test_unchanged_when_mood_intersection_empty(self):
        """atmosphere terms are out-of-vocab -> legacy prompt."""
        meta = _meta_ok(atmosphere=["claustrophobic", "smoke-filtered"])
        out, status, terms = mgt._apply_story_brief_mood_prefix(_LEGACY_PROMPT, meta)
        assert out == _LEGACY_PROMPT
        assert status == "ok"
        assert terms == []


# ---------------------------------------------------------------------------
# 7. E-06 meta-threading canary
# ---------------------------------------------------------------------------


def test_meta_canary_in_vocab_term_reaches_prompt_out_of_vocab_filtered():
    """E-06 with a twist: an in-vocab atmosphere term ("tense") AND
    the unique canary token both ride in `atmosphere`. The helper
    should filter the canary OUT (it's not in `_MUSIC_MOOD_VOCAB`)
    and let "tense" through. Proves the helper is wired AND filtering
    correctly. A regression that bypasses the helper would leak the
    canary into the prompt."""
    meta = _meta_ok(atmosphere=["tense", _CANARY])
    out, _status, terms = mgt._apply_story_brief_mood_prefix(_LEGACY_PROMPT, meta)
    assert "tense" in out
    assert _CANARY not in out, (
        f"E-06 canary {_CANARY!r} leaked into MusicGen prompt; helper "
        "filtering not engaged or wiring bypassed"
    )
    assert terms == ["tense"]


# ---------------------------------------------------------------------------
# 8. E-07 log surfaces story_brief_status
# ---------------------------------------------------------------------------


def test_musicgen_logs_story_brief_status_source():
    """The MusicGenTheme.execute log line includes story_brief_status."""
    src = _MGT_PATH.read_text(encoding="utf-8")
    assert "story_brief_status=" in src, (
        "musicgen_theme.py does not surface story_brief_status in any "
        "log statement; E-07 observability missing"
    )


# ---------------------------------------------------------------------------
# 9. No new LLM call added at this site
# ---------------------------------------------------------------------------


def test_musicgen_no_llm_call_added(mgt_ast):
    """C5g must NOT add an LLM call (make_generate_fn, request_slot,
    or similar) inside musicgen_theme.py. MusicGen stays an audio
    model call site only."""
    forbidden = {"make_generate_fn", "request_slot"}
    for node in ast.walk(mgt_ast):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name in forbidden:
            pytest.fail(
                f"forbidden LLM call `{name}` at line {node.lineno} in "
                "nodes/musicgen_theme.py; C5g must not turn MusicGen "
                "into an LLM call site"
            )


# ---------------------------------------------------------------------------
# 10-11. E-28 production-caller contract
# ---------------------------------------------------------------------------


def _production_callers_of(symbol: str) -> list[str]:
    """Return relative-path strings of files under nodes/, visual/,
    scripts/ that either import or call `symbol`. Excludes tests/."""
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
                if isinstance(node, ast.ImportFrom):
                    for imp in node.names:
                        if imp.name == symbol:
                            hits.append(str(py_path.relative_to(_REPO_ROOT)))
                            break
                elif isinstance(node, ast.Call):
                    func = node.func
                    if (
                        (isinstance(func, ast.Name) and func.id == symbol)
                        or (isinstance(func, ast.Attribute) and func.attr == symbol)
                    ):
                        hits.append(str(py_path.relative_to(_REPO_ROOT)))
    # Dedupe while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def test_get_music_mood_has_exactly_one_production_caller_at_c5g():
    """E-28 / RR-B10: at C5g commit boundary, exactly ONE production
    file references get_story_brief_music_mood. C5b's caller-count
    test will fail post-C5g (it asserted zero callers); after this
    sprint closes, that test gets either renamed or deleted as part
    of Sprint A cleanup."""
    callers = _production_callers_of("get_story_brief_music_mood")
    # C5b's test_get_music_mood_has_zero_production_callers_at_c5b will
    # fail at C5g + later sprints; this test asserts the post-C5g
    # contract: exactly one production consumer.
    assert len(callers) == 1, (
        f"C5g wiring contract violated -- expected 1 production caller "
        f"of get_story_brief_music_mood, got {len(callers)}: {callers}"
    )


def test_get_music_mood_caller_is_musicgen_theme():
    """E-28 strict form: the one production caller is
    `nodes/musicgen_theme.py` -- not some other consumer that
    accidentally pulled in the helper."""
    callers = _production_callers_of("get_story_brief_music_mood")
    assert callers == ["nodes\\musicgen_theme.py"] or callers == ["nodes/musicgen_theme.py"], (
        f"C5g wiring contract violated -- expected sole caller to be "
        f"nodes/musicgen_theme.py, got: {callers}"
    )


# ---------------------------------------------------------------------------
# Runtime-gated (Sprint A unlocks these)
# ---------------------------------------------------------------------------


_RUNTIME_REASON = (
    "Set OTR_REGRESSION_RUNTIME=1 to run the C5g audio C7 baseline "
    "reset tests. Requires ComfyUI + GPU + the two fixture files "
    "(captured during Sprint A): "
    "tests/fixtures/audio_c7_baseline_pre_c5g.wav.b3sum and "
    "tests/fixtures/audio_c7_baseline.wav.b3sum."
)


class TestRuntimeOnly:
    """Tests gated on OTR_REGRESSION_RUNTIME. Sprint A unlocks these
    after capturing the pre-C5g forensic b3sum + the new canonical
    baseline b3sum on the GPU machine."""

    @pytest.mark.skipif(
        not os.environ.get("OTR_REGRESSION_RUNTIME"),
        reason=_RUNTIME_REASON,
    )
    def test_audio_c7_new_baseline_captured(self):
        """E-12: tests/fixtures/audio_c7_baseline.wav.b3sum exists, is
        non-empty, and differs from the pre-C5g forensic b3sum
        captured at tests/fixtures/audio_c7_baseline_pre_c5g.wav.b3sum.
        Sprint A captures both fixture files."""
        new_bs = _REPO_ROOT / "tests" / "fixtures" / "audio_c7_baseline.wav.b3sum"
        pre_bs = _REPO_ROOT / "tests" / "fixtures" / "audio_c7_baseline_pre_c5g.wav.b3sum"
        assert new_bs.is_file(), (
            f"new canonical baseline b3sum not present at {new_bs}; "
            "Sprint A capture not yet run"
        )
        assert pre_bs.is_file(), (
            f"pre-C5g forensic b3sum not present at {pre_bs}; "
            "Sprint A capture not yet run"
        )
        new_text = new_bs.read_text(encoding="utf-8").strip()
        pre_text = pre_bs.read_text(encoding="utf-8").strip()
        assert new_text, "new baseline b3sum is empty"
        assert pre_text, "pre-C5g forensic b3sum is empty"
        assert new_text != pre_text, (
            "new and pre-C5g b3sums are identical -- the C5g code change "
            "did not actually shift the audio output; smuggled-no-op "
            "regression class fired"
        )

    @pytest.mark.skipif(
        not os.environ.get("OTR_REGRESSION_RUNTIME"),
        reason=_RUNTIME_REASON,
    )
    def test_c5g_audio_matches_legacy_when_brief_absent(self):
        """E-16 / RR-A2 isolation test. Force meta.story_brief_status
        to 'absent'; run the full C5g pipeline. Assert byte-identical
        match to tests/fixtures/audio_c7_baseline_pre_c5g.wav.b3sum.
        Proves the audio shift between pre-C5g and new-baseline is
        caused EXCLUSIVELY by the mood-prefix code path, not by an
        unintended regression elsewhere in the C5g diff.

        Sprint A scope -- placeholder until the empirical-verification
        harness lands."""
        pytest.skip(
            "Sprint A empirical-verification harness scope; placeholder "
            "until the runtime test driver is built."
        )

    @pytest.mark.skipif(
        not os.environ.get("OTR_REGRESSION_RUNTIME"),
        reason=_RUNTIME_REASON,
    )
    def test_audio_c7_byte_identical_c5g_new_baseline(self):
        """Ongoing canary post-C5g: every commit boundary must
        re-confirm byte-identical match to the new canonical baseline.
        Sprint A unlocks this when the fixture is captured."""
        pytest.skip(
            "Sprint A empirical-verification harness scope; placeholder."
        )
