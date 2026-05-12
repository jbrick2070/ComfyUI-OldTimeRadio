"""tests/test_lfc_w4_writer_polish_fn.py

LFC sprint Commit 12.2 -- W4 fix. OTR_LedgerScriptWriter builds a
dedicated polish_generate_fn off cache_entry and threads it into
every compose_line call, so the inline polish path (when
enable_polish_pass=True) does NOT inherit the writer's composer-
tuned closure (min_p / repetition_penalty / top_p).
"""
from __future__ import annotations

import re as _re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REPO_ROOT = Path(__file__).resolve().parents[1]
WRITER_SRC = (REPO_ROOT / "nodes" / "OTR_LedgerScriptWriter.py").read_text(
    encoding="utf-8"
)


def _flat(text: str) -> str:
    """Collapse implicit-concat string boundaries for stable lookup.

    `"foo " "bar"` -> `"foobar"` after passing through this.
    """
    out = _re.sub(r"\s+", " ", text)
    return out.replace('" "', "")


_FLAT = _flat(WRITER_SRC)


class TestW4WriterBuildsDedicatedPolishFn:
    def test_make_polish_generate_fn_called_in_writer(self):
        assert "_OTRML.make_polish_generate_fn(" in _FLAT, (
            "W4 fix: writer must build a dedicated polish_generate_fn "
            "off cache_entry via _OTRML.make_polish_generate_fn"
        )

    def test_polish_fallback_logged_at_warning(self):
        # Locate the fallback log site -- the writer wraps the
        # factory call in try/except and falls back to None on
        # failure. The log must be at WARNING level so a real
        # loader regression surfaces in the boot log.
        marker = "make_polish_generate_fn unavailable"
        idx = _FLAT.find(marker)
        assert idx != -1, "W4 fallback log message not found"
        context = _FLAT[max(0, idx - 400): idx]
        warn_pos = context.rfind("log.warning")
        debug_pos = context.rfind("log.debug")
        assert warn_pos != -1, (
            f"expected log.warning before {marker!r}; got: {context!r}"
        )
        assert warn_pos > debug_pos, (
            "log.debug appears more recent than log.warning in the "
            "fallback context; W4 fix regressed"
        )

    def test_polish_generate_fn_threaded_to_compose_line(self):
        # Every compose_line call in the writer must pass the
        # polish_generate_fn kwarg.
        compose_call_count = _FLAT.count("_OTRLC.compose_line(")
        threaded_count = _FLAT.count(
            "polish_generate_fn=polish_generate_fn"
        )
        assert compose_call_count >= 1, "no compose_line calls found"
        assert threaded_count == compose_call_count, (
            f"W4 fix: every compose_line call must pass "
            f"polish_generate_fn (found {compose_call_count} "
            f"compose_line calls, {threaded_count} threaded)"
        )

    def test_polish_fallback_to_none_preserves_back_compat(self):
        # When the factory raises, the writer must set
        # polish_generate_fn = None so compose_line falls back to
        # generate_fn (pre-W4 behaviour). This preserves
        # functionality on older builds where
        # make_polish_generate_fn isn't yet exported.
        assert "polish_generate_fn = None" in _FLAT, (
            "W4 fix: fallback path must set polish_generate_fn=None "
            "so compose_line's back-compat path engages"
        )


class TestW4ComposeLineSurfaceUnchanged:
    def test_compose_line_keyword_back_compat(self):
        # compose_line's polish_generate_fn kwarg defaults to None,
        # so any caller (including legacy tests) that doesn't pass
        # the kwarg still works.
        from nodes import _otr_line_composer as _OTRLC
        import inspect
        sig = inspect.signature(_OTRLC.compose_line)
        assert "polish_generate_fn" in sig.parameters
        param = sig.parameters["polish_generate_fn"]
        assert param.default is None, (
            "compose_line.polish_generate_fn must default None "
            "to preserve legacy call sites"
        )


class TestW4DocumentationConsistency:
    def test_writer_inline_comment_references_w4(self):
        # The W4 fix is a small but meaningful change; the comment
        # at the call site should mention the closure-leak rationale
        # so a future reader understands why TWO generate fns
        # exist.
        assert "W4 fix" in WRITER_SRC, (
            "W4 fix: writer should carry an inline comment "
            "explaining why the polish_generate_fn is separate"
        )
        assert "closure" in WRITER_SRC.lower(), (
            "W4 fix: writer comment should reference the "
            "closure-leak rationale (composer tuning leaking "
            "into polish)"
        )
