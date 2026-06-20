"""Tests for the per-run resolved-model ledger (historical accuracy in the
episode credits sheet).

A `~latest` OpenRouter alias resolves to a CONCRETE model server-side; the
backend records that (from the response body's `model` field) + the call's
cost so the treatment can report exactly what served the run.
"""
from __future__ import annotations

import json

from nodes import _otr_openrouter_backend as orb


def _reset():
    orb.reset_run_budget()


def test_record_and_snapshot_accumulates():
    _reset()
    body = {"model": "anthropic/claude-opus-4-8",
            "usage": {"cost": 0.0120}, "choices": []}
    orb._record_resolved("~anthropic/claude-opus-latest", body)
    orb._record_resolved("~anthropic/claude-opus-latest", body)
    snap = orb.resolved_models_snapshot()
    assert "~anthropic/claude-opus-latest" in snap
    entry = snap["~anthropic/claude-opus-latest"]
    assert entry["resolved"] == "anthropic/claude-opus-4-8"
    assert entry["calls"] == 2
    assert abs(entry["cost_usd"] - 0.024) < 1e-6


def test_reset_clears_resolved_ledger():
    _reset()
    orb._record_resolved("~openai/gpt-latest",
                         {"model": "openai/gpt-5.5", "usage": {}})
    assert orb.resolved_models_snapshot()
    orb.reset_run_budget()
    assert orb.resolved_models_snapshot() == {}


def test_record_is_best_effort_on_garbage():
    _reset()
    # Missing model / non-numeric cost must not raise.
    orb._record_resolved("~x/y", {"usage": {"cost": "n/a"}})
    orb._record_resolved("~x/y", {})
    snap = orb.resolved_models_snapshot()
    assert snap["~x/y"]["calls"] == 2
    assert snap["~x/y"]["cost_usd"] == 0.0


def test_resolved_models_surface_in_treatment(tmp_path):
    """The treatment WRITER/LLM CONFIG block reports the concrete resolved
    model + cost from the backend snapshot."""
    from tests.fixtures.ledger_stub import make_stub_ledger
    from nodes.video_engine import _write_story_treatment

    _reset()
    orb._record_resolved(
        "~anthropic/claude-opus-latest",
        {"model": "anthropic/claude-opus-4-8", "usage": {"cost": 0.05}})

    led = make_stub_ledger()
    led["meta"]["gen_params_initial"] = {
        "creative_writing_model": "openrouter:~anthropic/claude-opus-latest",
        "technical_model": "openrouter:~anthropic/claude-opus-latest",
    }
    out_path = tmp_path / "ResolvedEp.mp4"
    out_path.touch()
    _write_story_treatment(
        out_path=str(out_path), episode_title="Resolved Ep", led=led,
        voice_assignments={}, style="", genre="",
        news_used=json.dumps([{"headline": "h"}]),
        duration=10.0, W=1280, H=720, fps=24, size_mb=1.0,
    )
    content = list(tmp_path.glob("*.txt"))[0].read_text(encoding="utf-8")
    assert "Resolved (OpenRouter)" in content
    assert "anthropic/claude-opus-4-8" in content
    assert "Total OpenRouter cost" in content
    _reset()
