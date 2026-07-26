"""docs/ENGINE_MATRIX.md may not drift from the live engine registry.

Multi-clip coverage chunk 7a (2026-07-26). The matrix answers the operator's
standing question -- aspect, resolution, seconds per clip, prompt contract,
still requirements, per model -- and its whole value is that it is TRUE. A
requirements document that has quietly gone stale is worse than no document:
it reads authoritative while describing an engine that no longer exists, and
the next reader stops looking.

So the doc is generated, never hand-edited, and this test is the gate. Add an
engine, change a ladder, retire a lane -- the suite says so, by name, in the
same commit.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "ENGINE_MATRIX.md"
TOOL = ROOT / "tools" / "engine_matrix.py"


def _tool():
    """Import tools/engine_matrix.py by path -- tools/ is not a package."""
    spec = importlib.util.spec_from_file_location("_otr_engine_matrix", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_generator_exists_and_the_doc_was_generated():
    assert TOOL.exists(), "tools/engine_matrix.py is the only writer of the doc"
    assert DOC.exists(), (
        "docs/ENGINE_MATRIX.md is missing. Run: python tools/engine_matrix.py")


def test_the_doc_matches_the_live_registry():
    """The drift gate itself."""
    fresh = _tool().render()
    current = DOC.read_text(encoding="utf-8")
    assert current == fresh, (
        "docs/ENGINE_MATRIX.md no longer matches the live engine registry. "
        "Regenerate it in this commit: python tools/engine_matrix.py")


def test_the_doc_is_marked_generated_so_nobody_hand_edits_it():
    head = DOC.read_text(encoding="utf-8")[:800]
    assert "GENERATED FILE" in head
    assert "tools/engine_matrix.py" in head


def test_the_doc_is_ascii_and_has_no_BOM():
    raw = DOC.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "BOM in a generated doc"
    try:
        raw.decode("ascii")
    except UnicodeDecodeError as exc:
        pytest.fail("docs/ENGINE_MATRIX.md is not ASCII: %s" % exc)


def test_every_registered_engine_has_a_row():
    """The count in the doc is not decoration -- it is the completeness claim."""
    import nodes._otr_video_engines  # noqa: F401
    from nodes._otr_video_engines import registry as vreg

    text = DOC.read_text(encoding="utf-8")
    names = sorted(vreg.all_engine_names())
    missing = [n for n in names if ("| %s |" % n) not in text]
    assert not missing, "engines absent from the matrix: %s" % ", ".join(missing)
    assert "registered engine names: **%d**" % len(names) in text


def test_the_google_rows_report_the_CANVAS_rate_not_the_providers():
    """The conversion most likely to be silently wrong, pinned in the doc too.

    Veo generates at 24 fps and is COUNTED at the canvas's 25, because
    ``canonicalize()`` resamples before anything reads a frame count. This test
    itself asserted the opposite until the chunk-7a QA panel corrected the
    declaration -- it was named ``..._report_24_fps_not_the_canvas_25`` and
    demanded ``menu: 96, 144, 192``. It agreed with the bug because it was
    written from the same wrong premise, one file over. Pin the published
    seconds too: those are what a reader actually checks against Veo's docs,
    and 4/6/8 is right whichever rate you count in -- it is the FRAMES that
    move.
    """
    text = DOC.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("| google_veo_video |"):
            assert "menu: 100, 150, 200" in line
            assert "menu: 4, 6, 8 s" in line
            assert "| 25 |" in line
            assert "96" not in line, "the provider-rate frame counts came back"
            return
    pytest.fail("no google_veo_video row in the matrix")
