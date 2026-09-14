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


def test_canonical_scope_contract_reports_both_nodes_off_the_canvas():
    """Both scope nodes left the canonical, and the profiler says so.

    This asserted a live node-93 -> node-94 wiring until 2026-09-13. 8171e994
    removed both nodes: node 93 shipped with bypass=True, which copies input to
    output, so the pass drew nothing and still cost a render stage.

    Pinning the ABSENCE rather than deleting the test keeps the decision
    visible -- if either node comes back to the canonical, this goes red on
    purpose and whoever rewired it has to say why. The node classes themselves
    still ship and are still profiled by the tests below; it is only the
    canonical wiring that is gone.
    """
    profiler = _load_profile_module()
    contract = profiler.load_canonical_scope_contract()
    assert contract["blend_on_canonical"] is False, (
        "OTR_PostUpscaleProcgenBlend is back on the canonical; 8171e994 took "
        "it off because it shipped bypassed. Say why it returned."
    )
    assert contract["scene_on_canonical"] is False, (
        "OTR_SceneAwareScopes is back on the canonical; it fed the bypassed "
        "blend. Say why it returned."
    )
    assert contract["blend_node_ids"] == []
    assert contract["scene_node_ids"] == []


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

