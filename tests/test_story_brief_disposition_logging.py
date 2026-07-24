"""Sprint E E3 / R-3: uniform story_brief disposition logging guard.

Asserts the new `log_story_brief_disposition` helper exists and is
importable, and that the function emits a uniform single log line of
the documented shape. AST-walking the 5 consumer files for
import-and-call would couple this test to the exact consumer source
shapes, which the next round of consumer edits (E10 humo logs, etc)
will rearrange; we keep the guard at the helper level and verify each
consumer separately in its own targeted test.
"""
from __future__ import annotations

import logging
from io import StringIO

import pytest

from nodes._otr_story_brief_helpers import log_story_brief_disposition


CONSUMER_IDS = ("flux_env", "flux_portrait", "ltx", "humo", "musicgen")


def _capture_log() -> tuple[logging.Logger, StringIO]:
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("test_story_brief_disposition")
    logger.setLevel(logging.INFO)
    # Reset handlers each invocation so tests don't leak across runs.
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.addHandler(handler)
    logger.propagate = False
    return logger, buf


@pytest.mark.parametrize("consumer_id", CONSUMER_IDS)
def test_disposition_logs_ok_path(consumer_id: str):
    meta = {
        "story_brief_status": "ok",
        "story_brief": "smoke-tinged corridor lit by failing sodium lamps",
        "story_brief_terms": {
            "setting":    ["corridor", "lobby"],
            "lighting":   ["sodium", "failing", "amber"],
            "atmosphere": ["tense", "ominous"],
        },
    }
    logger, buf = _capture_log()
    status = log_story_brief_disposition(meta, consumer_id, logger)
    assert status == "ok"
    line = buf.getvalue().strip()
    assert f"[story_brief:{consumer_id}]" in line
    assert "status=ok" in line
    # "smoke-tinged corridor lit by failing sodium lamps" -> 49 chars
    assert "brief_chars=49" in line
    assert "setting=2/lighting=3/atmosphere=2" in line


@pytest.mark.parametrize("consumer_id", CONSUMER_IDS)
def test_disposition_logs_absent_path(consumer_id: str):
    """A pre-Sprint-C ledger has no story_brief_status key. The
    disposition resolves to 'absent' and the terms summary is all
    zeros."""
    meta = {}  # legacy ledger
    logger, buf = _capture_log()
    status = log_story_brief_disposition(meta, consumer_id, logger)
    assert status == "absent"
    line = buf.getvalue().strip()
    assert f"[story_brief:{consumer_id}]" in line
    assert "status=absent" in line
    assert "brief_chars=0" in line
    assert "setting=0/lighting=0/atmosphere=0" in line


@pytest.mark.parametrize("consumer_id", CONSUMER_IDS)
def test_disposition_logs_failed_path(consumer_id: str):
    meta = {
        "story_brief_status": "failed",
        # On failure, story_brief and terms may still be partially
        # populated from the failed-repair sentinel, but the disposition
        # log treats it as zero-info.
        "story_brief": "leaked junk",
        "story_brief_terms": {"atmosphere": ["urgent"]},
    }
    logger, buf = _capture_log()
    status = log_story_brief_disposition(meta, consumer_id, logger)
    assert status == "failed"
    line = buf.getvalue().strip()
    assert f"[story_brief:{consumer_id}]" in line
    assert "status=failed" in line
    assert "brief_chars=0" in line


def test_disposition_returns_status_string():
    """The helper returns the resolved status so callers can branch
    on it (e.g. MusicGen needs status='ok' to apply mood prefix)."""
    ok_meta = {
        "story_brief_status": "ok",
        "story_brief": "x",
        "story_brief_terms": {},
    }
    failed_meta = {"story_brief_status": "failed"}
    absent_meta = {}
    logger, _buf = _capture_log()
    assert log_story_brief_disposition(ok_meta, "musicgen", logger) == "ok"
    assert log_story_brief_disposition(failed_meta, "musicgen", logger) == "failed"
    assert log_story_brief_disposition(absent_meta, "musicgen", logger) == "absent"


def test_helper_module_no_consumer_imports():
    """Dependency direction guard. _otr_story_brief_helpers must NOT
    import any consumer module -- direction is consumer -> helper.
    Cross-imports trip circular-load at ComfyUI registration time."""
    import nodes._otr_story_brief_helpers as h
    src = open(h.__file__, encoding="utf-8").read()
    forbidden = (
        "import nodes.batch_flux_render",
        "import nodes.musicgen_theme",
        "import nodes.batch_humo_render",
        "import nodes.batch_ltx_render",
        "import visual.batch_flux_render",
        "import visual.batch_flux_portrait_render",
        "from nodes.musicgen_theme",
        "from nodes.batch_humo_render",
    )
    for needle in forbidden:
        assert needle not in src, (
            f"helper module must not import consumer; found {needle!r}"
        )
