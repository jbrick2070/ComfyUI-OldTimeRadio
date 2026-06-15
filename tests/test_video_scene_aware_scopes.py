"""§4D regression: OTR_SceneAwareScopes -- the v2 scene-aware reactive scopes.

Covers the build verify-list: per-beat eligibility (portrait->gutters,
landscape->suppress, head/inter gap->centre, credits tail->suppress), GREEN-ONLY
(no amber/red leak), deterministic frame rendering, empty-manifest fail-early,
the silent -an encode, and the blend's 3-input gbrp double-blend filtergraph.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OTR_TEST_MODE", "1")

from nodes import otr_scene_aware_scopes as sc  # noqa: E402


def _manifest():
    # head gap [0,50) / portrait clip [50,100) / inter gap [100,150) /
    # landscape clip [150,200) / tail credits [200,300)
    return {
        "episode_id": "ep_test", "fps": 25, "total_target_frames": 300,
        "clips": [
            {"order": 0, "shot_id": "s1", "engine_id": "humo",
             "path": "/fake/portrait.mp4", "target_frame_count": 50,
             "start_s": 2.0, "exists": True},
            {"order": 1, "shot_id": "s2", "engine_id": "ltx",
             "path": "/fake/land.mp4", "target_frame_count": 50,
             "start_s": 6.0, "exists": True},
        ],
    }


@pytest.fixture
def fake_aspect(monkeypatch):
    """portrait.mp4 -> portrait(True); land.mp4 -> landscape(False)."""
    def _fake(path, ffprobe, cache):
        if not path:
            return None
        return "portrait" in path
    monkeypatch.setattr(sc, "_probe_is_portrait", _fake)


def test_eligibility_by_source_and_aspect(fake_aspect):
    plan, total = sc.plan_scope_frames(_manifest(), 1920, 1080)
    assert total == 300

    def mode(fi):
        m = plan[fi]
        return m[0] if m else "suppress"

    assert mode(10) == "centre"     # head gap -> keep alive
    assert mode(60) == "gutters"    # portrait clip -> gutters
    assert mode(120) == "centre"    # inter gap -> keep alive
    assert mode(160) == "suppress"  # landscape clip -> suppress
    assert mode(250) == "suppress"  # tail credits -> suppress


def test_unprobeable_clip_suppresses(monkeypatch):
    monkeypatch.setattr(sc, "_probe_is_portrait", lambda p, f, c: None)
    plan, total = sc.plan_scope_frames(_manifest(), 1920, 1080)
    assert plan[60] is None  # un-probeable clip -> suppress (not a crash)


def test_empty_manifest_fails_early():
    node = sc.SceneAwareScopes()
    import json
    with pytest.raises(ValueError):
        node.render_scopes(json.dumps({"clips": []}), audio=None, out_w=320, out_h=240)


def _draw_frame(W, H, signal, key="ep", fi=60):
    img = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    g, lcx, rcx, cy, r = sc._gutter_geom(W, H)
    freqs = [np.clip(np.random.default_rng(i).random(32).astype(np.float32), 0, 1)
             for i in range(fi + 1)]
    waves = [(np.random.default_rng(i + 7).random(200).astype(np.float32) * 2 - 1)
             for i in range(fi + 1)]
    env = {"fi": fi, "fps": 25, "key": key, "vol": 0.6,
           "signal": signal, "loss": 1 - signal, "trig": signal}
    sc.draw_fft_scope(d, lcx, cy, r, freqs[-6:], env)
    sc.draw_scope(d, rcx, cy, r, waves[-6:], env)
    return np.asarray(img)


def test_green_only_no_amber_or_red():
    # GREEN-ONLY: the red channel is never lit (amber/red are forbidden). The
    # CRT_GREEN phosphor carries a small blue tint (B=65) that the blend's
    # colorchannelmixer zeroes -- so only R==0 is asserted here.
    arr = _draw_frame(1280, 720, signal=0.6)
    assert arr[:, :, 0].max() == 0, "red channel lit -> amber/red leak"
    assert arr[:, :, 1].max() > 0, "green channel must carry the scope"


def test_amp_clamped_inside_gutter():
    # r + amp must never cross into the centre band (amp <= r*0.35).
    g, lcx, rcx, cy, r = sc._gutter_geom(1920, 1080)
    amp = r * 0.35
    assert lcx + r + amp <= g, "left scope crosses the centre band"
    assert rcx - r - amp >= 1920 - g, "right scope crosses the centre band"


def test_frame_determinism():
    a = _draw_frame(1280, 720, signal=0.05, fi=40)  # idle path uses seeded rng
    b = _draw_frame(1280, 720, signal=0.05, fi=40)
    assert np.array_equal(a, b), "idle scope frame is not deterministic"


def test_blend_three_input_filtergraph():
    from nodes.otr_post_upscale_procgen_blend import _build_blend_cmd
    cmd = _build_blend_cmd(
        Path("src.mp4"), Path("pgn.mp4"), Path("out.mp4"),
        "screen", 1.0, "ffmpeg", green_only_overlay=True,
        source_dims=(1920, 1080), scopes_mp4=Path("scopes.mp4"))
    assert cmd.count("-i") == 3
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "[2:v]" in fc
    assert "[main][pgn]blend=all_mode=screen" in fc
    assert "blend=all_mode=lighten" in fc
    assert fc.strip().endswith("[v]")


def test_blend_two_input_unchanged_when_no_scopes():
    from nodes.otr_post_upscale_procgen_blend import _build_blend_cmd
    cmd = _build_blend_cmd(
        Path("src.mp4"), Path("pgn.mp4"), Path("out.mp4"),
        "screen", 1.0, "ffmpeg", green_only_overlay=True, source_dims=(1920, 1080))
    assert cmd.count("-i") == 2
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "[2:v]" not in fc


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_silent_encode_no_audio_stream(tmp_path, monkeypatch):
    import json
    import subprocess
    monkeypatch.setattr(sc, "_probe_is_portrait", lambda p, f, c: True)
    m = {"episode_id": "enc", "fps": 25, "total_target_frames": 40,
         "clips": [{"order": 0, "shot_id": "s1", "engine_id": "humo",
                    "path": "/fake/portrait.mp4", "target_frame_count": 40,
                    "start_s": 0.0, "exists": True}]}
    node = sc.SceneAwareScopes()
    out = node.render_scopes(json.dumps(m), audio=None, out_w=320, out_h=180)
    path = out["result"][0]
    assert os.path.isfile(path)
    pr = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    assert pr.stdout.strip() == "", "scopes_only.mp4 must be silent (-an)"
    os.remove(path)


def test_scopes_span_master_audio_no_blend_clamp_bug406(monkeypatch):
    """BUG-LOCAL-406: the scopes track must span the full MASTER-audio length, not
    the beats-only total_target_frames. The downstream §4D 3-input blend uses
    shortest=1, so a scopes input shorter than the master clamps the whole blended
    video below the master -> the mux clone-holds the last frame over the closing
    theme (the FREEZE + missing HUD/scopes treatment). The node has the master
    `audio` input, so it pads the plan tail to the master length."""
    import json
    import torch
    monkeypatch.setattr(sc, "_probe_is_portrait", lambda p, f, c: True)
    captured = {}

    def _fake_encode(gen, total, out, w, h, fps, ffmpeg):
        captured["total"] = total
        captured["gen_n"] = sum(1 for _ in gen)   # drain: no IndexError on the tail

    monkeypatch.setattr(sc, "_encode_silent_mp4", _fake_encode)

    # beats budget = 50 frames (2 s @25); master audio = 5 s -> 125 frames.
    m = {"episode_id": "ext", "fps": 25, "total_target_frames": 50,
         "clips": [{"order": 0, "shot_id": "s1", "engine_id": "humo",
                    "path": "/fake/portrait.mp4", "target_frame_count": 50,
                    "start_s": 0.0, "exists": True}]}
    sr = 24000
    wf = torch.zeros(1, 1, int(sr * 5.0), dtype=torch.float32)   # 5 s master mix
    sc.SceneAwareScopes().render_scopes(
        json.dumps(m), audio={"waveform": wf, "sample_rate": sr},
        out_w=160, out_h=90)
    assert captured["total"] == 125          # extended from beats(50) to master(125)
    assert captured["gen_n"] == 125          # frames render across the full span
