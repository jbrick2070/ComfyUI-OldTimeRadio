"""Story-Quality R2 Final QA -- the lever metrics in story_quality_scan.

r2_lever_metrics counts the shipped craft-lever flags + structural signals over a
frozen ledger (read-only). Pure / CPU. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

_spec = importlib.util.spec_from_file_location(
    "story_quality_scan", str(_REPO / "scripts" / "story_quality_scan.py"))
_scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scan)


def _ledger(lines, meta=None, cast=None):
    return {"lines": lines, "meta": meta or {}, "cast": cast or []}


def test_counts_weak_flags():
    led = _ledger([
        {"speaker_role": "character", "text": "You're playing with fire, Skip."},
        {"speaker_role": "character", "text": "I'm scared, this is dangerous."},
        {"speaker_role": "character", "text": "clenches jaw You mean it?"},
        {"speaker_role": "announcer", "text": "And so the lesson is haste makes waste."},
    ])
    m = _scan.r2_lever_metrics(led)
    assert m["cliche_lines"] == 1
    assert m["on_the_nose_lines"] == 1
    assert m["leading_stage_dir_lines"] == 1
    assert m["thesis_close"] is True


def test_clean_episode_all_zero():
    led = _ledger([
        {"speaker_role": "character", "text": "The decay curve gives us ten months."},
        {"speaker_role": "announcer", "text": "The pencil lies still on the empty console."},
    ])
    m = _scan.r2_lever_metrics(led)
    assert m["cliche_lines"] == 0 and m["on_the_nose_lines"] == 0
    assert m["leading_stage_dir_lines"] == 0 and m["thesis_close"] is False


def test_structural_signals():
    led = _ledger(
        [{"speaker_role": "character", "text": "Hello."}],
        meta={
            "specificity_anchors": ["three robotic arms", "Swift"],
            "central_object": "the pencil",
            "dramatic_state": {
                "character_a_wants": "CHRIS: honor the established commitment, whatever the cost",
                "character_b_wants": "SKIP: force a compromise that protects the status quo",
            },
        },
        cast=[
            {"name": "CHRIS", "speech_signature": "clipped and terse"},
            {"name": "SKIP", "speech_signature": "warm and rambling"},
            {"name": "ANNOUNCER", "speech_signature": "narrator"},
        ],
    )
    m = _scan.r2_lever_metrics(led)
    assert m["has_specificity_anchors"] is True and m["n_specificity_anchors"] == 2
    assert m["has_central_object"] is True
    assert m["wants_default"] is True            # boilerplate wants detected
    assert m["voice_distinct_ratio"] == 1.0      # two distinct char registers


def test_scan_ledger_merges_lever_metrics(tmp_path):
    import json
    p = tmp_path / "ep_ledger.json"
    p.write_text(json.dumps(_ledger(
        [{"speaker_role": "character", "text": "You're playing with fire."}],
        meta={"central_object": "the pencil"})), encoding="utf-8")
    row = _scan.scan_ledger(str(p), None)
    assert row["cliche_lines"] == 1
    assert row["has_central_object"] is True


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
