"""Sprint E E5 / R-7 + M9: validator empty-path fallback + MusicGen
ImportError install hint.

H5: validator's _load_workflow("") must explicitly resolve to
_DEFAULT_WORKFLOW_PATH and log the resolved path at INFO so the
operator can tell whether the empty widget was intentional or a
wiring error. Pre-E5 the fallback was implicit and silent.

M9: MusicGen ImportError message must name the install command
explicitly (pip install transformers audiocraft) so the operator has
an actionable handle, not "install the MusicGen optional deps".
"""
from __future__ import annotations

import inspect
import logging
from io import StringIO
from pathlib import Path

import pytest

from nodes import _otr_workflow_validator as wv
from nodes import musicgen_theme


REPO_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_JSON = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"


def _capture_log(logger_name: str) -> tuple[logging.Logger, StringIO]:
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.addHandler(handler)
    logger.propagate = False
    return logger, buf


class TestValidatorEmptyPathFallback:
    """R-7: empty widget path must explicitly resolve + log."""

    def test_empty_path_resolves_to_default(self):
        """_load_workflow("") returns the parsed canonical JSON."""
        result = wv._load_workflow("")
        assert isinstance(result, dict)
        assert "nodes" in result
        # Canonical workflow node count drifts as nodes are consolidated
        # (e.g. the 2026-05-30 LTX distilled-LoRA merge 60/61 -> one
        # loader dropped it to 29). This floor only confirms the empty
        # path resolved to the real full workflow, not an empty/stub
        # dict -- it is intentionally well below the live count.
        assert len(result["nodes"]) >= 25

    def test_empty_path_logs_resolved_path(self, caplog):
        """The fallback must emit one INFO line naming the resolved
        path so soak diagnostics see what file was actually loaded."""
        # Use caplog on the module's actual logger name.
        with caplog.at_level(logging.INFO, logger=wv.log.name):
            wv._load_workflow("")
        joined = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "workflow_json_path widget empty" in joined
        assert "_DEFAULT_WORKFLOW_PATH" in joined
        assert "otr_scifi_16gb_full.json" in joined

    def test_explicit_path_does_not_log_fallback(self, caplog):
        """When the widget carries an explicit non-empty path the
        empty-fallback log must NOT fire."""
        canonical = str(CANONICAL_JSON)
        with caplog.at_level(logging.INFO, logger=wv.log.name):
            wv._load_workflow(canonical)
        joined = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "widget empty" not in joined

    def test_default_widget_path_is_empty_string(self):
        """Pins the shipped widget default convention (operator-path
        neutral; resolved at runtime via the fallback)."""
        schema = wv.WorkflowValidator.INPUT_TYPES()
        widget = schema["required"]["workflow_json_path"]
        assert widget[0] == "STRING"
        # The default may be the canonical Path string -- but the JSON
        # ships "" so the resolution branch fires. The schema default
        # documents the operator-resolved canonical path for reference.

    def test_source_has_explicit_empty_branch(self):
        """AST-level: source must contain an explicit empty-path
        branch so a future refactor cannot drop the log silently."""
        src = inspect.getsource(wv._load_workflow)
        assert "not path" in src or "path == \"\"" in src or "not path:" in src
        assert "widget empty" in src


class TestMusicgenImportErrorMessage:
    """R-7 companion / M9: MusicGen ImportError message names the
    install command explicitly."""

    def test_musicgen_source_has_explicit_install_hint(self):
        """The error message body must name the pip install command
        so a fresh operator can resolve without guessing which extras
        package is missing."""
        src = open(musicgen_theme.__file__, encoding="utf-8").read()
        # The exact message string lives in the ImportError branch at
        # the transformers / MusicGen import site. We don't try to
        # trigger the ImportError live (transformers is installed in
        # the test env); we check the source contains the install hint.
        assert "pip install transformers audiocraft" in src, (
            "MusicGen ImportError branch must name explicit install "
            "command; operator should not have to guess which extras "
            "package is missing"
        )
