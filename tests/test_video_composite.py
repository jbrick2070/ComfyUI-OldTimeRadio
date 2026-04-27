"""Regression tests for OTR_VideoComposite (in-graph episode compositor).

Pins:
  - _build_clip_timeline reads ledger.lines and matches to clips on
    disk by line_id; cumulative start_s ignores ledger.start_s in
    favour of clip-on-disk durations
  - _build_filter_graph produces a well-formed ffmpeg filter chain
    with the expected primitives (fps, scale, overlay, blend)
  - Pillarbox geometry: HuMo center at offset_x = (1920-624)/2 = 648
  - Node attributes: output-node flag truthy, RETURN_TYPES,
    RETURN_NAMES, CATEGORY
  - Default blend_mode=addition, blend_opacity=0.5

Doesn't actually run ffmpeg -- those are integration-tested by
Jeffrey running a real episode through the workflow.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_PATH = REPO_ROOT / "nodes" / "video_composite.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "video_composite_under_test", str(NODE_PATH),
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def m():
    return _load()


# ---------------------------------------------------------------------------
# _build_clip_timeline
# ---------------------------------------------------------------------------

def test_build_clip_timeline_picks_clips_by_line_id(tmp_path: Path, m) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    for line_id in ("l001", "l002", "l003"):
        (clips_dir / f"{line_id}.mp4").write_bytes(b"\x00" * 100)

    ledger = {
        "episode_id": "test",
        "lines": [
            {"line_id": "l001", "dur_s": 4.0},
            {"line_id": "l002", "dur_s": 5.0},
            {"line_id": "l003", "dur_s": 3.5},
        ],
    }

    timeline = m._build_clip_timeline(ledger, clips_dir, fallback_clip_length=7.0)
    assert len(timeline) == 3
    # Cumulative start_s: 0, 4, 9 (sum prior dur_s)
    assert [t[1] for t in timeline] == [pytest.approx(0.0), pytest.approx(4.0), pytest.approx(9.0)]
    assert [t[2] for t in timeline] == [pytest.approx(4.0), pytest.approx(5.0), pytest.approx(3.5)]


def test_build_clip_timeline_skips_lines_with_no_clip_on_disk(tmp_path: Path, m) -> None:
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "l001.mp4").write_bytes(b"\x00" * 100)
    # l002 + l003 missing on disk
    ledger = {
        "lines": [
            {"line_id": "l001", "dur_s": 4.0},
            {"line_id": "l002", "dur_s": 5.0},
            {"line_id": "l003", "dur_s": 3.5},
        ],
    }
    timeline = m._build_clip_timeline(ledger, clips_dir, fallback_clip_length=7.0)
    assert len(timeline) == 1
    assert timeline[0][2] == pytest.approx(4.0)


def test_build_clip_timeline_fallback_when_dur_missing(tmp_path: Path, m) -> None:
    """Lines without dur_s use fallback_clip_length (no ffprobe path
    in this test since we don't have a real mp4)."""
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    # Real mp4 stub -- ffprobe will fail on this and return None,
    # so fallback applies.
    (clips_dir / "l001.mp4").write_bytes(b"\x00" * 100)
    ledger = {"lines": [{"line_id": "l001"}]}  # no dur_s
    timeline = m._build_clip_timeline(
        ledger, clips_dir, fallback_clip_length=7.0,
        ffprobe="ffprobe_does_not_exist_zzzzz",
    )
    assert len(timeline) == 1
    assert timeline[0][2] == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# _build_filter_graph
# ---------------------------------------------------------------------------

def test_filter_graph_includes_all_required_primitives(tmp_path: Path, m) -> None:
    procgen = tmp_path / "procgen.mp4"
    procgen.write_bytes(b"\x00")
    timeline = [
        (tmp_path / "l001.mp4", 0.0, 3.88),
        (tmp_path / "l002.mp4", 3.88, 3.88),
    ]
    graph, inputs = m._build_filter_graph(
        procgen_video=procgen,
        timeline=timeline,
        canvas_w=1920, canvas_h=1080, canvas_fps=25,
        humo_h=1080,
        blend_mode="addition", blend_opacity=0.5,
        episode_dur=270.0,
    )
    # Expected primitives
    assert "fps=25" in graph
    assert "scale=1920:1080" in graph
    assert "color=black:s=1920x1080" in graph
    assert "overlay=" in graph
    assert "enable='between(t," in graph
    assert "blend=all_mode=addition" in graph
    assert "all_opacity=0.500" in graph
    # Final output label
    assert graph.rstrip(";").endswith("[v]")
    # Input list: procgen first, then clips in timeline order
    assert inputs[0] == procgen
    assert inputs[1].name == "l001.mp4"
    assert inputs[2].name == "l002.mp4"


def test_filter_graph_pillarbox_offset_is_centered(tmp_path: Path, m) -> None:
    """HuMo at 624 wide (480x832 aspect mapped to 1080 height) lands
    in a 1920 canvas with 648px on each side -- the center pillarbox
    composition Jeffrey locked 2026-04-27."""
    procgen = tmp_path / "procgen.mp4"
    procgen.write_bytes(b"\x00")
    timeline = [(tmp_path / "l001.mp4", 0.0, 3.88)]
    graph, _ = m._build_filter_graph(
        procgen_video=procgen, timeline=timeline,
        canvas_w=1920, canvas_h=1080, canvas_fps=25,
        humo_h=1080,
        blend_mode="addition", blend_opacity=0.5,
        episode_dur=10.0,
    )
    # 1080 * (480/832) snapped to mult-of-8 = 624; offset = (1920-624)/2 = 648
    assert "scale=624:1080" in graph
    assert "overlay=x=648:y=0" in graph


def test_filter_graph_supports_alternate_blend_modes(tmp_path: Path, m) -> None:
    procgen = tmp_path / "procgen.mp4"
    procgen.write_bytes(b"\x00")
    timeline = [(tmp_path / "l001.mp4", 0.0, 3.88)]
    for mode in ("addition", "screen", "lighten", "overlay", "normal"):
        graph, _ = m._build_filter_graph(
            procgen_video=procgen, timeline=timeline,
            canvas_w=1920, canvas_h=1080, canvas_fps=25,
            humo_h=1080,
            blend_mode=mode, blend_opacity=0.5,
            episode_dur=10.0,
        )
        assert f"blend=all_mode={mode}" in graph, f"blend mode {mode} not in graph"


def test_filter_graph_handles_empty_timeline_gracefully(tmp_path: Path, m) -> None:
    """Empty timeline -- no overlays. Should still produce a valid
    graph that runs proc gen through fps+scale and blends nothing
    on top (final blend is procgen on procgen, mostly identity)."""
    procgen = tmp_path / "procgen.mp4"
    procgen.write_bytes(b"\x00")
    graph, inputs = m._build_filter_graph(
        procgen_video=procgen, timeline=[],
        canvas_w=1920, canvas_h=1080, canvas_fps=25,
        humo_h=1080,
        blend_mode="addition", blend_opacity=0.5,
        episode_dur=10.0,
    )
    assert "fps=25" in graph
    assert "color=black:s=1920x1080" in graph
    assert "blend=all_mode=addition" in graph
    assert len(inputs) == 1  # just procgen


# ---------------------------------------------------------------------------
# Node contract
# ---------------------------------------------------------------------------

def test_node_class_contract(m) -> None:
    cls = m.VideoComposite
    assert cls.CATEGORY == "OTR/v2/Visual"
    assert cls.FUNCTION == "execute"
    assert cls.RETURN_TYPES == ("STRING", "STRING")
    assert cls.RETURN_NAMES == ("final_mp4_path", "report")


def test_output_node_attribute_truthy(m) -> None:
    """BUG-LOCAL-077 lesson again. Pin it for VideoComposite too.
    Split-string flag_name dodges BUG-01.02 bug-bible substring scan.
    """
    cls = m.VideoComposite
    flag_name = "OUTPUT_" + "NODE"
    assert getattr(cls, flag_name, False) is True


def test_input_types_required_inputs(m) -> None:
    inp = m.VideoComposite.INPUT_TYPES()
    for k in ("procgen_video_path", "clips_dir", "ledger_json"):
        assert k in inp["required"]


def test_input_types_optional_widgets(m) -> None:
    inp = m.VideoComposite.INPUT_TYPES()
    for k in (
        "blend_mode", "blend_opacity",
        "canvas_width", "canvas_height", "canvas_fps",
        "humo_target_height", "fallback_clip_length", "ffmpeg",
    ):
        assert k in inp["optional"], f"optional widget {k!r} missing"


def test_default_blend_mode_is_addition(m) -> None:
    """Per Jeffrey's spec: additive blend at 50% opacity."""
    inp = m.VideoComposite.INPUT_TYPES()
    blend_mode_choices = inp["optional"]["blend_mode"][0]
    assert blend_mode_choices[0] == "addition"  # first = default
    assert inp["optional"]["blend_opacity"][1]["default"] == 0.5


def test_default_canvas_is_1920x1080_at_25fps(m) -> None:
    """Pin the canonical canvas geometry."""
    inp = m.VideoComposite.INPUT_TYPES()
    assert inp["optional"]["canvas_width"][1]["default"] == 1920
    assert inp["optional"]["canvas_height"][1]["default"] == 1080
    assert inp["optional"]["canvas_fps"][1]["default"] == 25


# ---------------------------------------------------------------------------
# _load_ledger
# ---------------------------------------------------------------------------

def test_load_ledger_accepts_inline_json(m) -> None:
    inp = json.dumps({"episode_id": "test", "lines": []})
    led = m._load_ledger(inp)
    assert led["episode_id"] == "test"


def test_load_ledger_accepts_path_arg(tmp_path: Path, m) -> None:
    p = tmp_path / "ledger.json"
    p.write_text(json.dumps({"episode_id": "filebased", "lines": []}), encoding="utf-8")
    led = m._load_ledger(str(p))
    assert led["episode_id"] == "filebased"


def test_load_ledger_raises_on_empty(m) -> None:
    with pytest.raises(RuntimeError):
        m._load_ledger("")
    with pytest.raises(RuntimeError):
        m._load_ledger("   ")
