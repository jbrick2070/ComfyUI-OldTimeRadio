"""tests/test_script_critic_ledger.py -- self-test for the ledger-pure
script_critic rewrite.

Mocks the heavy LLM calls (story_orchestrator._generate_with_llm,
_run_with_timeout, LLMScriptWriter._parse_script) and the on-disk
ledger save (in_flight_ledger_path returns None) so each case runs
CPU-only in well under a second.

Covers four paths:
  - PASS  -- ledger pass-through with verdict + meta stamps
  - REVISE -- ledger.lines[].text mutation + char_count/word_count
              recompute via _recompute_text_metrics
  - legacy parser-list input -- loud-fall-through to PASS passthrough
  - empty script -- skipped path, meta.critic_skipped_reason stamped
"""
from __future__ import annotations

import json
import re
from unittest.mock import MagicMock, patch

import pytest

from tests.fixtures.ledger_stub import make_legacy_list, make_stub_ledger


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------

def _fake_run_with_timeout(fn, **kwargs):
    """Replace story_orchestrator._run_with_timeout: just call the inner fn."""
    return fn()


def _fake_pass_response(*_args, **_kwargs):
    return "SCORE: 95\nVERDICT: PASS\nISSUES:\n- none\n"


def _build_revise_llm():
    """Two-call mock: first call returns the critic's REVISE verdict,
    second call returns the revised script text."""
    rev_text = (
        "[VOICE: ANNOUNCER, warm baritone] A new dawn breaks tonight.\n\n"
        "[VOICE: LEMMY, gruff] Take it slow, kid.\n\n"
        "[VOICE: ASTRA, calm] Acknowledged.\n"
    )
    state = {"calls": 0}

    def _impl(*_args, **_kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            return (
                "SCORE: 60\nVERDICT: REVISE\n"
                "ISSUES:\n- exposition: too long\n- pacing: drag\n"
            )
        return rev_text

    return _impl, rev_text


@pytest.fixture
def silent_disk_save():
    """Make _persist_and_emit a no-op on the disk side: in_flight_ledger_path
    returns None so save_ledger_safe is skipped, keeping tests hermetic."""
    with patch("nodes._otr_ledger.in_flight_ledger_path", return_value=None), \
         patch("nodes._otr_ledger.save_ledger_safe", return_value=True):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_critic_pass_path_emits_ledger_dict(silent_disk_save):
    """PASS verdict -> ledger out, meta.critic_verdict + script_gates stamped."""
    from nodes.script_critic import LLMScriptCritic

    led_in = make_stub_ledger()
    script_json = json.dumps(led_in)
    script_text = (
        "[VOICE: ANNOUNCER, warm baritone] Tonight's broadcast.\n\n"
        "[VOICE: LEMMY, gruff] Get the kit out, fast.\n"
    )

    with patch(
        "nodes.story_orchestrator._generate_with_llm",
        side_effect=_fake_pass_response,
    ), patch(
        "nodes.story_orchestrator._run_with_timeout",
        side_effect=_fake_run_with_timeout,
    ):
        out_script, out_json, out_report, out_verdict = LLMScriptCritic().execute(
            script=script_text,
            script_json=script_json,
            revise_on_findings=False,
            block_on_reject=False,
            timeout_sec=10,
        )

    assert out_verdict == "PASS"
    assert out_script == script_text

    out_led = json.loads(out_json)
    assert isinstance(out_led, dict), "wire output must be a ledger dict"
    assert "lines" in out_led and "cast" in out_led, "ledger shape preserved"
    assert out_led["meta"]["critic_verdict"] == "PASS"
    assert out_led["meta"]["critic_score"] == 95
    assert out_led["meta"]["critic_report"], "meta.critic_report stamped"

    gates = out_led.get("script_gates") or []
    assert len(gates) == 1, "exactly one gate appended on a single critic run"
    assert gates[0]["verdict"] == "PASS"
    assert gates[0]["score"] == 95
    assert gates[0]["kind"] == "script_critic"


def test_critic_revise_path_mutates_ledger_text(silent_disk_save):
    """REVISE verdict + revise_on_findings=True -> ledger.lines[].text
    mutated by parallel-index against parsed dialogue, char_count and
    word_count recomputed (not stale from the stub)."""
    from nodes.script_critic import LLMScriptCritic

    led_in = make_stub_ledger()
    script_json = json.dumps(led_in)
    script_text = (
        "[VOICE: ANNOUNCER, warm baritone] From the studio, tonight's broadcast.\n\n"
        "[VOICE: LEMMY, gruff] Get the kit out, fast.\n\n"
        "[VOICE: ASTRA, calm] On it.\n"
    )

    fake_llm, _rev_text = _build_revise_llm()

    fake_parsed_dialogue = [
        {"type": "dialogue", "character_name": "ANNOUNCER",
         "line": "A new dawn breaks tonight.", "voice_traits": "warm baritone"},
        {"type": "dialogue", "character_name": "LEMMY",
         "line": "Take it slow, kid.", "voice_traits": "gruff"},
        {"type": "dialogue", "character_name": "ASTRA",
         "line": "Acknowledged.", "voice_traits": "calm"},
    ]
    fake_writer_class = MagicMock()
    fake_writer_class.return_value._parse_script = MagicMock(
        return_value=fake_parsed_dialogue,
    )

    with patch(
        "nodes.story_orchestrator._generate_with_llm",
        side_effect=fake_llm,
    ), patch(
        "nodes.story_orchestrator._run_with_timeout",
        side_effect=_fake_run_with_timeout,
    ), patch(
        "nodes.story_orchestrator.LLMScriptWriter",
        new=fake_writer_class,
    ):
        out_script, out_json, _out_report, out_verdict = LLMScriptCritic().execute(
            script=script_text,
            script_json=script_json,
            revise_on_findings=True,
            block_on_reject=False,
            timeout_sec=10,
        )

    assert out_verdict == "REVISE"
    assert "Take it slow" in out_script, "revised script text emitted on script socket"

    out_led = json.loads(out_json)
    assert isinstance(out_led, dict)
    assert out_led["meta"]["critic_verdict"] == "REVISE"

    # Verify all 3 voiced lines (1 announcer + 2 character) got their text
    # mutated to the revised dialogue, in the order the ledger stored them.
    voiced = [
        ln for ln in out_led["lines"]
        if ln.get("speaker_role") in ("character", "announcer")
    ]
    assert len(voiced) == 3, f"expected 3 voiced lines, got {len(voiced)}"
    assert [ln["text"] for ln in voiced] == [
        "A new dawn breaks tonight.",
        "Take it slow, kid.",
        "Acknowledged.",
    ]

    # char_count + word_count must be recomputed via the regex helper so
    # they stay consistent with the writer's initial-stamp formula.
    word_re = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")
    for ln in voiced:
        text = ln["text"]
        assert ln["char_count"] == len(text), (
            f"char_count drift on {text!r}: got {ln['char_count']}"
        )
        assert ln["word_count"] == len(word_re.findall(text)), (
            f"word_count drift on {text!r}: got {ln['word_count']}"
        )

    # Non-voiced lines (sfx / music) must NOT have been touched.
    non_voiced = [
        ln for ln in out_led["lines"]
        if ln.get("speaker_role") not in ("character", "announcer")
    ]
    assert all(
        ln["text"] in ("Theme up.", "metal door slam", "Theme out.")
        for ln in non_voiced
    ), "non-voiced lines should be untouched"

    # script_revisions[] gets exactly one append on a single revise pass.
    revisions = out_led.get("script_revisions") or []
    assert len(revisions) == 1
    assert revisions[0]["verdict_before"] == "REVISE"
    assert revisions[0]["issues_addressed"] == 2

    # Gate entry stamped (the critic runs gate stamp BEFORE the revision pass).
    gates = out_led.get("script_gates") or []
    assert len(gates) == 1
    assert gates[0]["verdict"] == "REVISE"


def test_critic_legacy_list_input_passes_through(silent_disk_save):
    """Legacy parser-list input -> load_ledger raises ValueError ->
    critic falls through to PASS passthrough; report flags the wiring."""
    from nodes.script_critic import LLMScriptCritic

    legacy_json = json.dumps(make_legacy_list())
    script_text = "[VOICE: LEMMY, gruff] Hello there."

    out_script, out_json, out_report, out_verdict = LLMScriptCritic().execute(
        script=script_text,
        script_json=legacy_json,
        revise_on_findings=False,
        block_on_reject=False,
        timeout_sec=10,
    )

    assert out_verdict == "PASS"
    assert out_script == script_text
    # Wire output is the legacy passthrough -- no merged-led to emit.
    assert json.loads(out_json) == make_legacy_list()
    # Markdown report names the wiring problem so the audit shows it.
    assert "legacy" in out_report.lower() or "rewire" in out_report.lower()


def test_critic_empty_script_skips_with_pass_meta(silent_disk_save):
    """Empty script -> SKIPPED, but meta.critic_verdict + skipped_reason
    still stamp on the wire ledger."""
    from nodes.script_critic import LLMScriptCritic

    led_in = make_stub_ledger()
    script_json = json.dumps(led_in)

    out_script, out_json, _out_report, out_verdict = LLMScriptCritic().execute(
        script="",
        script_json=script_json,
        revise_on_findings=False,
        block_on_reject=False,
        timeout_sec=10,
    )

    assert out_verdict == "PASS"
    assert out_script == ""

    out_led = json.loads(out_json)
    assert out_led["meta"]["critic_verdict"] == "PASS"
    assert out_led["meta"]["critic_skipped_reason"] == "empty_script"

    gates = out_led.get("script_gates") or []
    assert len(gates) == 1
    assert gates[0]["skipped_reason"] == "empty_script"
