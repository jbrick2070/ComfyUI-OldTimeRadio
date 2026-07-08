"""Regression tests for the OTR scope-render profiler.

These tests keep the GPU-acceleration planning harness tied to the real
canonical workflow and the production CPU/PIL draw helpers it is meant to time.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = REPO_ROOT / "scripts" / "profile_scope_render.py"


def _load_profile_module():
    spec = importlib.util.spec_from_file_location("otr_profile_scope_render", PROFILE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_canonical_scope_contract_is_live():
    profiler = _load_profile_module()
    contract = profiler.load_canonical_scope_contract()
    assert contract["scene_node_id"] == 94
    assert contract["blend_node_id"] == 93
    assert contract["scope_link"] == 273
    assert contract["scope_slot"] == 9
    assert contract["scene_widgets"][1:4] == [1920, 1080, "ffmpeg"]
    assert contract["blend_audio_bars"] == "bottom"


def test_scene_scope_profile_draw_only_is_nonblank_and_timed():
    profiler = _load_profile_module()
    result = profiler.profile_scene_scopes(
        frames=8,
        width=160,
        height=90,
        sink="none",
        landscape_bars="bottom",
    )
    assert result["target"] == "scene_scopes"
    assert result["fps"] == 25
    assert result["frames"] == 8
    assert result["sink"]["mode"] == "none"
    assert result["sink"]["frames"] == 8
    assert result["timings_seconds"]["audio_analysis"] >= 0.0
    assert result["timings_seconds"]["frame_draw"] > 0.0
    assert result["timings_seconds"]["pipe_write"] == 0.0
    assert result["derived"]["checksum_green"] > 0


def test_post_bars_profile_draw_only_is_nonblank_and_timed():
    profiler = _load_profile_module()
    result = profiler.profile_post_bars(frames=8, width=160, height=90, sink="none")
    assert result["target"] == "post_bars"
    assert result["fps"] == 25
    assert result["frames"] == 8
    assert result["sink"]["mode"] == "none"
    assert result["sink"]["frames"] == 8
    assert result["timings_seconds"]["audio_analysis"] >= 0.0
    assert result["timings_seconds"]["frame_draw"] > 0.0
    assert result["derived"]["checksum_green"] > 0


def test_profile_cli_outputs_json(capsys):
    profiler = _load_profile_module()
    rc = profiler.main([
        "--target", "post_bars",
        "--frames", "3",
        "--resolution", "96x54",
        "--sink", "none",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"target": "post_bars"' in out
    assert '"resolution": [' in out


def test_bad_resolution_fails_loud():
    profiler = _load_profile_module()
    with pytest.raises(ValueError):
        profiler.parse_resolution("wide")

