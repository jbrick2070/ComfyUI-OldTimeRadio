"""Sprint D D0c -- SA-101 silent clamp visibility for repair pass.

The C5a1 repair pass clamps the LLM sampling temperature via
`min(reflection_temperature + 0.15, 0.55)`. Pre-D0c there was no log
between the clamp computation and the LLM call, so Sprint A
inspectors could not tell a 0.55-ceilinged retry from a 0.35 retry.

D0c adds a single additive `log.info` line. Two assertions:

  test_repair_pass_emits_clamp_log_when_clamp_fires
      Force the repair pass and confirm exactly one log.info call
      carries the substring '[OTR_StoryBrief] repair pass clamped'
      plus the resolved repair_temperature formatted to 3 decimals.

  test_repair_pass_clamp_log_does_not_break_no_change_logs_rule_for_other_strings
      AST-walk the module, collect every `log.<level>` call's first
      string argument, and pin a snapshot. Pre-existing log strings
      stay byte-stable; only the new D0c line appears.
"""
from __future__ import annotations

import ast
import logging
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_story_brief as story_brief_mod  # noqa: E402


CLAMP_LOG_SUBSTRING = "[OTR_StoryBrief] repair pass clamped"


def test_repair_pass_emits_clamp_log_when_clamp_fires(
    caplog: pytest.LogCaptureFixture,
) -> None:
    captured: list[str] = []

    def fake_technical_fn(messages, *, temperature, max_new_tokens):
        captured.append(f"temp={temperature:.3f}")
        return '{"setting": [], "lighting": [], "atmosphere": []}'

    caplog.set_level(logging.INFO, logger="OTR")
    out = story_brief_mod._repair_pass(
        failed_output="{ broken json",
        rejection_reasons=["json_parse_failed", "missing_setting_key"],
        technical_fn=fake_technical_fn,
        base_user_message="placeholder base prompt",
        reflection_temperature=0.30,
    )

    assert captured, "fake technical_fn was never invoked by _repair_pass"

    clamp_records = [
        r for r in caplog.records
        if CLAMP_LOG_SUBSTRING in r.getMessage()
    ]
    assert len(clamp_records) == 1, (
        f"expected exactly one clamp log line; got {len(clamp_records)}: "
        f"{[r.getMessage() for r in clamp_records]}"
    )

    msg = clamp_records[0].getMessage()
    # base=0.30 + bump=0.15 = 0.45 (under 0.55 ceiling)
    assert "base=0.300" in msg, msg
    assert "bump=0.150" in msg, msg
    assert "ceiling=0.550" in msg, msg
    assert "repair_temperature=0.450" in msg, msg
    assert "json_parse_failed" in msg, msg
    assert "missing_setting_key" in msg, msg
    assert out  # repair pass returned the fake technical_fn output


def test_repair_pass_log_clamps_to_ceiling_at_high_base_temperature(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """At base=0.50, the clamp pins at 0.55 (not 0.65)."""
    def fake_technical_fn(messages, *, temperature, max_new_tokens):
        return '{"setting": [], "lighting": [], "atmosphere": []}'

    caplog.set_level(logging.INFO, logger="OTR")
    story_brief_mod._repair_pass(
        failed_output="{",
        rejection_reasons=["validation_failed"],
        technical_fn=fake_technical_fn,
        base_user_message="x",
        reflection_temperature=0.50,
    )

    clamp_records = [
        r for r in caplog.records
        if CLAMP_LOG_SUBSTRING in r.getMessage()
    ]
    assert len(clamp_records) == 1
    msg = clamp_records[0].getMessage()
    assert "base=0.500" in msg, msg
    assert "repair_temperature=0.550" in msg, (
        f"expected clamp at 0.55 ceiling for base 0.50; got: {msg}"
    )


def test_repair_pass_clamp_log_does_not_break_no_change_logs_rule_for_other_strings() -> None:
    """Snapshot every `log.<level>` literal-string call in the module.

    The D0c new line carries the CLAMP_LOG_SUBSTRING. Every OTHER
    log call's first-arg literal string is pinned against a snapshot
    captured at D0c authorship time. If a future commit modifies any
    pre-existing log string this test fails loud.
    """
    src = Path(story_brief_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    log_strings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "log"
            and func.attr in {"debug", "info", "warning", "error", "critical"}
        ):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            log_strings.append(first.value)
        elif isinstance(first, ast.JoinedStr):
            log_strings.append(ast.unparse(first))

    # Bucket: the one new D0c clamp line vs everything else.
    new_d0c_lines = [s for s in log_strings if CLAMP_LOG_SUBSTRING in s]
    pre_existing_lines = [s for s in log_strings if CLAMP_LOG_SUBSTRING not in s]

    assert len(new_d0c_lines) == 1, (
        f"expected exactly one D0c clamp log string; got {len(new_d0c_lines)}"
    )

    # Pre-existing log strings must each include the [OTR_StoryBrief]
    # prefix (module convention) -- this is the no-change-logs invariant
    # for this file. If a future commit adds a log line without the
    # prefix or modifies an existing prefixed line, this catches it.
    for s in pre_existing_lines:
        assert "[OTR_StoryBrief]" in s, (
            f"pre-existing log string missing [OTR_StoryBrief] prefix "
            f"(D0c boundary violation): {s!r}"
        )

    assert pre_existing_lines, (
        "expected at least one pre-existing log string in "
        "_otr_story_brief.py (snapshot floor)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
