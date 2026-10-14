"""OTR_VideoRenderBatch grades every real episode it renders.

The grader's own tests hand it documents. These drive the node's real
``_render_episode`` with a stand-in render driver and read the report it
returns -- the dict written as ``node_episode_report.json`` -- so they fail
if the call is ever unwired. UTF-8, no BOM.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from nodes import otr_video_render_batch as vrb  # noqa: E402

FROZEN = {"announcer_visual": "ltx25_audio_in_16gb", "character_video": "humo"}


def _ledger():
    return {"episode_id": "ep_graded", "video": {
        "roles_effective": dict(FROZEN),
        "shots": [
            {"shot_id": "shot_b0", "role": "announcer_visual",
             "engine_id": "ltx25_audio_in_16gb", "target_frame_count": 50},
            {"shot_id": "shot_b1", "role": "character_video",
             "engine_id": "humo", "target_frame_count": 50},
        ]}}


def _row(shot_id, engine_id):
    return {"shot_id": shot_id, "engine_id": engine_id, "exists": True,
            "frame_count": 50, "extension_mode": "none",
            "native_frame_count": 50, "delivered_native_frame_count": 50}


class _Driver:
    """Just enough of render_driver for ``_render_episode`` to reach the end."""

    def __init__(self, rows):
        self.rows = rows

    def resolve_final_shot_engines(self, ledger):
        return ledger

    def validate_and_repair_still_spine(self, ledger):
        return None

    def run_real_episode(self, ledger, master_audio_path=""):
        return {"ledger": ledger, "clips": {}, "trace": [],
                "vram_peak_mb": None, "receipts": []}

    def resolve_episode_id_for_clip_persistence(self, episode_id, **_kw):
        return episode_id

    def persist_episode_clips(self, ep, episode_id):
        return None

    def build_clip_manifest(self, ep, episode_id=""):
        hist = {}
        for row in self.rows:
            hist[row["engine_id"]] = hist.get(row["engine_id"], 0) + 1
        return {"clips": list(self.rows), "n_beats": len(self.rows),
                "clip_count": len(self.rows), "engine_histogram": hist,
                "video_revision": 1, "floor_clip_count": 0}

    def ltx_prompt_diversity_status(self, trace):
        return {"ok": True}


@pytest.fixture(autouse=True)
def _no_ledger_writes(monkeypatch):
    monkeypatch.setattr(vrb, "_stamp_floored_video_shots", lambda ep: None)
    monkeypatch.setattr(vrb, "_stamp_render_engines_meta",
                        lambda manifest, vram: None)
    monkeypatch.setattr(vrb, "_stamp_render_trace",
                        lambda receipts, render_run_id="": 0)


def _report(rows):
    import json
    report, _manifest, name = vrb.OTRVideoRenderBatch._render_episode(
        _Driver(rows), json.dumps(_ledger()))
    assert name == "node_episode_report.json"
    return report


def test_an_honest_episode_reports_zero_findings():
    report = _report([_row("shot_b0", "ltx25_audio_in_16gb"),
                      _row("shot_b1", "humo")])
    assert report["acceptance_finding_count"] == 0


def test_a_swapped_route_is_counted_and_named_in_the_log(caplog):
    with caplog.at_level(logging.WARNING, logger="OTR"):
        report = _report([_row("shot_b0", "humo"),
                          _row("shot_b1", "ltx25_audio_in_16gb")])
    assert report["acceptance_finding_count"] == 2
    assert report["ok"] is True, "grading records; it never fails the render"
    text = caplog.text
    assert "acceptance: 2 finding(s)" in text
    assert "delivered_engine" in text and "shot_b0" in text


def test_a_grader_that_cannot_run_leaves_the_render_alone(monkeypatch):
    from nodes._otr_video_engines import acceptance as acc

    def _boom(ledger, manifest):
        raise RuntimeError("grader fault")

    monkeypatch.setattr(acc, "grade_episode", _boom)
    report = _report([_row("shot_b0", "ltx25_audio_in_16gb"),
                      _row("shot_b1", "humo")])
    assert report["acceptance_finding_count"] is None
    assert report["ok"] is True
