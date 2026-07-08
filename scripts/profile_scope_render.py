"""Profile OTR scope and bottom-bar rendering against the canonical workflow.

The profiler is deliberately production-shaped: it imports the real scene-aware
scope helpers, the real post-blend bar geometry, and the shared green bar drawer.
It does not touch Comfy widgets or mutate the workflow.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
NODES_DIR = REPO_ROOT / "nodes"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(NODES_DIR) not in sys.path:
    sys.path.insert(0, str(NODES_DIR))

WORKFLOW_PATH = REPO_ROOT / "workflows" / "otr_canonical.json"
FPS_HARD_LOCK = 25

from nodes import otr_post_upscale_procgen_blend as post_blend  # noqa: E402
from nodes import otr_scene_aware_scopes as scene_scopes  # noqa: E402
from nodes._otr_shared.encode_sink import RawVideoSink  # noqa: E402
from nodes._otr_shared.scope_draw import analyze_audio_np, freq_bars_green  # noqa: E402


def load_canonical_scope_contract(workflow_path: Path = WORKFLOW_PATH) -> dict[str, Any]:
    """Return and validate the live node-93/node-94 scope contract."""
    wf = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
    nodes = {int(n["id"]): n for n in wf.get("nodes", [])}
    links = {int(l[0]): l for l in wf.get("links", [])}
    n93 = nodes.get(93)
    n94 = nodes.get(94)
    if not n93 or n93.get("type") != "OTR_PostUpscaleProcgenBlend":
        raise RuntimeError("profile_scope_render: node 93 is not OTR_PostUpscaleProcgenBlend")
    if not n94 or n94.get("type") != "OTR_SceneAwareScopes":
        raise RuntimeError("profile_scope_render: node 94 is not OTR_SceneAwareScopes")

    n93_inputs = [i.get("name") for i in n93.get("inputs", [])]
    n94_outputs = n94.get("outputs", [])
    if "scopes_mp4_path" not in n93_inputs:
        raise RuntimeError("profile_scope_render: node 93 lacks scopes_mp4_path input")
    scope_slot = n93_inputs.index("scopes_mp4_path")
    scope_link = n93["inputs"][scope_slot].get("link")
    if scope_link is None:
        raise RuntimeError("profile_scope_render: node 93 scopes_mp4_path is unwired")
    link = links.get(int(scope_link))
    if not link or link[1:5] != [94, 0, 93, scope_slot]:
        raise RuntimeError(
            "profile_scope_render: expected node 94 output 0 wired to "
            f"node 93 scopes slot {scope_slot}; got {link!r}"
        )
    if int(scope_link) not in (n94_outputs[0].get("links") or []):
        raise RuntimeError("profile_scope_render: node 94 output does not list the scope link")

    n93_widgets = list(n93.get("widgets_values") or [])
    n94_widgets = list(n94.get("widgets_values") or [])
    if len(n93_widgets) != 11 or n93_widgets[10] not in {"bottom", "off"}:
        raise RuntimeError("profile_scope_render: node 93 audio_bars widget contract drifted")
    if len(n94_widgets) != 5 or n94_widgets[1:4] != [1920, 1080, "ffmpeg"]:
        raise RuntimeError("profile_scope_render: node 94 widget contract drifted")
    if n94_widgets[4] not in {"off", "bottom"}:
        raise RuntimeError("profile_scope_render: node 94 landscape_bars widget drifted")

    return {
        "workflow": str(Path(workflow_path)),
        "scope_link": int(scope_link),
        "scope_slot": int(scope_slot),
        "scene_node_id": 94,
        "blend_node_id": 93,
        "scene_widgets": n94_widgets,
        "blend_audio_bars": n93_widgets[10],
    }


def parse_resolution(value: str) -> tuple[int, int]:
    left, sep, right = str(value).lower().partition("x")
    if not sep:
        raise ValueError(f"resolution must look like WIDTHxHEIGHT, got {value!r}")
    w, h = int(left), int(right)
    if w <= 0 or h <= 0:
        raise ValueError(f"resolution must be positive, got {value!r}")
    return w, h


def synthetic_audio(sample_rate: int, seconds: float) -> np.ndarray:
    """Deterministic mono float32 audio with tone, pulse, and quiet sections."""
    n = max(1, int(float(sample_rate) * float(seconds)))
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    base = 0.18 * np.sin(2.0 * np.pi * 110.0 * t)
    harmonic = 0.08 * np.sin(2.0 * np.pi * 440.0 * t)
    pulse = ((np.sin(2.0 * np.pi * 3.0 * t) > 0.72).astype(np.float32) * 0.22)
    noise = np.random.default_rng(42).normal(0.0, 0.015, size=n).astype(np.float32)
    audio = base + harmonic + pulse + noise
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def _profile_output_path(target: str, sink: str) -> Path | None:
    if sink != "mp4":
        return None
    root = Path(tempfile.gettempdir()) / "otr_scope_profile"
    root.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    return root / f"{target}_{stamp}.mp4"


def profile_scene_scopes(
    *,
    frames: int,
    width: int,
    height: int,
    sink: str = "none",
    ffmpeg: str = "ffmpeg",
    landscape_bars: str = "bottom",
) -> dict[str, Any]:
    """Profile the node-94 draw stack with synthetic audio."""
    contract = load_canonical_scope_contract()
    total = max(1, int(frames))
    fps = FPS_HARD_LOCK
    audio = synthetic_audio(22050, max(total / fps, 0.5))

    t0 = time.perf_counter()
    volume, freqs, waves = scene_scopes._analyze_audio_np(audio, 22050, total, fps)
    signal, trig, loss = scene_scopes._dual_ema(volume)
    analysis_seconds = time.perf_counter() - t0

    gutter = scene_scopes._gutter_geom(width, height)
    centre = scene_scopes._centre_geom(width, height)
    bars = scene_scopes._bars_geom(width, height)
    modes: list[Any] = [
        ("gutters", gutter[1], gutter[2], gutter[3], gutter[4]),
        ("centre", centre[0], centre[1], centre[2]),
    ]
    if str(landscape_bars).lower() == "bottom":
        modes.append(("bars", bars[0], bars[1], bars[2], bars[3]))
    modes.append(None)

    draw_seconds = 0.0
    checksum = 0
    output_path = _profile_output_path("scene_scopes", sink)
    with RawVideoSink(
        mode=sink,
        width=width,
        height=height,
        fps=fps,
        ffmpeg=ffmpeg,
        output_path=output_path,
    ) as raw_sink:
        for fi in range(total):
            t1 = time.perf_counter()
            img = Image.new("RGB", (width, height), scene_scopes.CRT_BLACK)
            mode = modes[fi % len(modes)]
            if mode is not None:
                draw = ImageDraw.Draw(img)
                lo = max(0, fi - scene_scopes._TRAIL_N + 1)
                env = {
                    "fi": fi,
                    "fps": fps,
                    "key": "profile_scene_scopes",
                    "vol": float(volume[fi]),
                    "signal": float(signal[fi]),
                    "loss": float(loss[fi]),
                    "trig": float(trig[fi]),
                }
                if mode[0] == "gutters":
                    _, lcx, rcx, cy, r = mode
                    scene_scopes.draw_fft_scope(draw, lcx, cy, r, freqs[lo:fi + 1], env)
                    scene_scopes.draw_scope(draw, rcx, cy, r, waves[lo:fi + 1], env)
                elif mode[0] == "bars":
                    _, bx, by, bw, bh = mode
                    freq_bars_green(draw, freqs[fi], bx, by, bw, bh)
                else:
                    _, cx, cy, r = mode
                    scene_scopes.draw_fft_scope(draw, cx, cy, r, freqs[lo:fi + 1], env)
                    scene_scopes.draw_scope(draw, cx, cy, int(r * 0.6), waves[lo:fi + 1], env)
            frame = np.asarray(img, dtype=np.uint8)
            checksum = (checksum + int(frame[:, :, 1].sum())) % 1_000_000_007
            draw_seconds += time.perf_counter() - t1
            raw_sink.write(frame)
        sink_stats = raw_sink.stats

    return _result(
        target="scene_scopes",
        frames=total,
        width=width,
        height=height,
        contract=contract,
        timings={
            "audio_analysis": analysis_seconds,
            "frame_draw": draw_seconds,
            "pipe_write": sink_stats.pipe_seconds,
            "encode_wait": sink_stats.encode_seconds,
        },
        sink_stats=sink_stats.__dict__,
        checksum=checksum,
    )


def profile_post_bars(
    *,
    frames: int,
    width: int,
    height: int,
    sink: str = "none",
    ffmpeg: str = "ffmpeg",
) -> dict[str, Any]:
    """Profile the node-93 always-on audio-bars draw stack."""
    contract = load_canonical_scope_contract()
    total = max(1, int(frames))
    fps = FPS_HARD_LOCK
    audio = synthetic_audio(22050, max(total / fps, 0.5))

    t0 = time.perf_counter()
    _volume, freqs, _waves = analyze_audio_np(audio, 22050, total, fps)
    analysis_seconds = time.perf_counter() - t0
    mx, by, bw, bh = post_blend._bars_strip_geom(width, height)

    draw_seconds = 0.0
    checksum = 0
    output_path = _profile_output_path("post_bars", sink)
    with RawVideoSink(
        mode=sink,
        width=width,
        height=height,
        fps=fps,
        ffmpeg=ffmpeg,
        output_path=output_path,
    ) as raw_sink:
        for fi in range(total):
            t1 = time.perf_counter()
            img = Image.new("RGB", (width, height), (0, 0, 0))
            freq_bars_green(ImageDraw.Draw(img), freqs[fi], mx, by, bw, bh)
            frame = np.asarray(img, dtype=np.uint8)
            checksum = (checksum + int(frame[:, :, 1].sum())) % 1_000_000_007
            draw_seconds += time.perf_counter() - t1
            raw_sink.write(frame)
        sink_stats = raw_sink.stats

    return _result(
        target="post_bars",
        frames=total,
        width=width,
        height=height,
        contract=contract,
        timings={
            "audio_analysis": analysis_seconds,
            "frame_draw": draw_seconds,
            "pipe_write": sink_stats.pipe_seconds,
            "encode_wait": sink_stats.encode_seconds,
        },
        sink_stats=sink_stats.__dict__,
        checksum=checksum,
    )


def _result(
    *,
    target: str,
    frames: int,
    width: int,
    height: int,
    contract: dict[str, Any],
    timings: dict[str, float],
    sink_stats: dict[str, Any],
    checksum: int,
) -> dict[str, Any]:
    total_seconds = sum(float(v) for v in timings.values())
    return {
        "target": target,
        "frames": int(frames),
        "resolution": [int(width), int(height)],
        "fps": FPS_HARD_LOCK,
        "workflow_contract": contract,
        "timings_seconds": {k: round(float(v), 6) for k, v in timings.items()},
        "derived": {
            "measured_total_seconds": round(total_seconds, 6),
            "draw_fps": round(frames / timings["frame_draw"], 3)
            if timings["frame_draw"] > 0 else None,
            "checksum_green": int(checksum),
        },
        "sink": sink_stats,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["scene_scopes", "post_bars"], default="scene_scopes")
    parser.add_argument("--frames", type=int, default=75)
    parser.add_argument("--resolution", default="320x180")
    parser.add_argument("--sink", choices=["none", "null", "mp4"], default="none")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--landscape-bars", choices=["off", "bottom"], default="bottom")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)

    width, height = parse_resolution(args.resolution)
    if args.target == "scene_scopes":
        result = profile_scene_scopes(
            frames=args.frames,
            width=width,
            height=height,
            sink=args.sink,
            ffmpeg=args.ffmpeg,
            landscape_bars=args.landscape_bars,
        )
    else:
        result = profile_post_bars(
            frames=args.frames,
            width=width,
            height=height,
            sink=args.sink,
            ffmpeg=args.ffmpeg,
        )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

