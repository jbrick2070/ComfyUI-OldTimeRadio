"""proven_frame_count reuses a probe its caller just ran on the same clip.

Five render paths ran ``ffprobe_clip_fields(out_path)`` for the stream
contract and then ``proven_frame_count``, which ran the identical ffprobe again
for ``nb_frames``. The caller now hands its fields in. These pin that the
proof itself is unchanged: omitted fields still probe, an empty mapping means
"probed, no count" and decodes, zero is a count, and a mismatch still refuses.
"""
from pathlib import Path

import pytest

from nodes._otr_video_engines import wan_shared as WS
from nodes._otr_video_engines import wrapper_bridge as WB

ROOT = Path(__file__).resolve().parents[1]


def _forbid(name):
    def _raise(*_a, **_k):
        raise AssertionError(name + " must not run")
    return _raise


def test_preprobed_fields_skip_the_second_probe(monkeypatch):
    monkeypatch.setattr(WS, "ffprobe_clip_fields", _forbid("a second ffprobe"))
    monkeypatch.setattr(WS, "ffprobe_counted_frames", _forbid("a decode"))
    assert WB.proven_frame_count("clip.mp4", 48, preprobed_fields={"nb_frames": 48}) == 48


def test_omitted_fields_still_probe_with_the_callers_ffprobe(monkeypatch):
    calls = []

    def fields(path, *, ffprobe="ffprobe"):
        calls.append((path, ffprobe))
        return {"nb_frames": 12}

    monkeypatch.setattr(WS, "ffprobe_clip_fields", fields)
    assert WB.proven_frame_count("c.mp4", 12, ffprobe="custom-ffprobe") == 12
    assert calls == [("c.mp4", "custom-ffprobe")]


def test_an_empty_mapping_is_supplied_not_omitted(monkeypatch):
    monkeypatch.setattr(WS, "ffprobe_clip_fields", _forbid("a second ffprobe"))
    monkeypatch.setattr(WS, "ffprobe_counted_frames", lambda path, *, ffprobe="ffprobe": 7)
    assert WB.proven_frame_count("c.mp4", 7, preprobed_fields={}) == 7


def test_zero_is_a_count_not_a_missing_count(monkeypatch):
    monkeypatch.setattr(WS, "ffprobe_counted_frames", _forbid("a decode"))
    assert WB.proven_frame_count("c.mp4", 0, preprobed_fields={"nb_frames": 0}) == 0


def test_a_mismatch_still_refuses():
    with pytest.raises(WB.GraphExecutionError):
        WB.proven_frame_count("c.mp4", 10, preprobed_fields={"nb_frames": 9})


@pytest.mark.parametrize("rel", [
    "nodes/_otr_video_engines/eng_visualizer.py",
    "nodes/_otr_video_engines/eng_viz_rainbow.py",
    "nodes/_otr_video_engines/eng_viz_camera.py",
    "nodes/_otr_video_engines/eng_viz_mandala.py",
    "nodes/_otr_video_engines/cheap_families.py",
])
def test_each_render_path_hands_its_probe_in(rel):
    src = (ROOT / rel).read_text(encoding="utf-8")
    assert "validate_silent_clip_contract(ffprobe_clip_fields(out_path), fps)" not in src
    assert "preprobed_fields=fields" in src
