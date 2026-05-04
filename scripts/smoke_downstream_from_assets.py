"""
smoke_downstream_from_assets.py
================================

REPRO HARNESS for the BUG-LOCAL-031 RTXUpscale-emits-black bug.

Instead of running the full ~90 min OTR pipeline (Story -> Director -> Bark
-> Kokoro -> MusicGen -> AudioGen -> SceneSequencer -> AudioEnhance ->
EpisodeAssembler -> SignalLostVideo -> FLUX -> HuMo -> LTX -> Composite ->
RTXUpscale -> PostUpscaleProcgenBlend), this script REUSES pre-rendered
assets from a previous completed run and ONLY exercises the suspect tail:

    [composite output mp4] -> RTXUpscale -> PostUpscaleProcgenBlend

That tail runs in roughly 60-120 seconds, so we can iterate on the
RTXUpscale fix without re-paying the 90-minute upstream tax each cycle.

Usage:
    python scripts/smoke_downstream_from_assets.py
    python scripts/smoke_downstream_from_assets.py --episode-id <slug>
    python scripts/smoke_downstream_from_assets.py --skip-upscale-bypass
    python scripts/smoke_downstream_from_assets.py --inspect-only

Default: pick the most recent episode under output/otr/episodes/.

By default the script writes outputs to a sibling smoke dir
``<ep_dir>/_smoke_<timestamp>/`` so we never clobber the original run's
deliverables. Pass ``--inplace`` to write to the canonical obs/ dir
instead (only do this when you want to overwrite a known-bad obs output
with a corrected one).

Each stage produces:
  * The output mp4
  * An ffprobe report (dims, duration, bitrate, frame count)
  * A sample frame extracted at ~5s mark (PNG + size in KB)

The bitrate + frame-size pair is the diagnostic signal: if a stage's
output bitrate collapses to <200 kbps and frame size <100 KB, that
stage emitted black frames.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Add the parent dir so we can import nodes/* without ComfyUI runtime.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Stub ComfyUI's `folder_paths` module so nodes that import it at module
# load time succeed under headless CLI. We only need the bare attributes
# OTR's path helpers actually call. Set this BEFORE importing any node.
import types as _types  # noqa: E402
if "folder_paths" not in sys.modules:
    _fp = _types.ModuleType("folder_paths")
    # ComfyUI's models_dir is the only attribute OTR reaches for in the
    # paths used by RTXUpscale / PostUpscaleProcgenBlend. Anchor at
    # <ComfyUI>/models so the shape matches the real runtime.
    _fp.models_dir = str(ROOT.parent.parent / "models")
    _fp.get_output_directory = lambda: str(ROOT.parent.parent / "output")
    _fp.get_input_directory = lambda: str(ROOT.parent.parent / "input")
    _fp.get_temp_directory = lambda: str(ROOT.parent.parent / "temp")
    sys.modules["folder_paths"] = _fp

OUTPUT_OTR = ROOT.parent.parent / "output" / "otr"
EPISODES_DIR = OUTPUT_OTR / "episodes"


# ---------------------------------------------------------------------------
# Output report accumulator
# ---------------------------------------------------------------------------

class StageReport:
    """Per-stage timing + ffprobe + black-frame signal."""

    def __init__(self):
        self.lines: list[str] = []

    def add(self, msg: str):
        print(msg)
        self.lines.append(msg)

    def section(self, title: str):
        sep = "=" * len(title)
        print()
        print(sep)
        print(title)
        print(sep)
        self.lines.append("")
        self.lines.append(sep)
        self.lines.append(title)
        self.lines.append(sep)


# ---------------------------------------------------------------------------
# Episode + asset discovery
# ---------------------------------------------------------------------------

def find_episode_dir(episode_id: str | None) -> Path | None:
    if not EPISODES_DIR.exists():
        return None
    if episode_id:
        target = EPISODES_DIR / episode_id
        return target if target.exists() else None
    candidates = [p for p in EPISODES_DIR.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def gather_assets(ep_dir: Path) -> dict | None:
    """Locate the pre-rendered assets we need for the downstream chain.

    Returns a dict on success, prints what's missing on failure.
    """
    audio_dir = ep_dir / "audio"
    composited_dir = ep_dir / "composited"
    videos_dir = ep_dir / "videos"

    # Composite mp4 (existing -- input to RTXUpscale OR overwritten by re-run)
    composite_candidates = list(composited_dir.glob(f"{ep_dir.name}*.mp4"))
    composite_mp4 = composite_candidates[0] if composite_candidates else None

    # Procgen mp4 (input to PostUpscaleProcgenBlend AND VideoComposite re-run)
    procgen_candidates = list(audio_dir.glob(f"{ep_dir.name}*.mp4"))
    procgen_mp4 = procgen_candidates[0] if procgen_candidates else None

    # Per-line video clips (input to VideoComposite re-run)
    per_line_clips = sorted(videos_dir.glob("l*.mp4")) if videos_dir.exists() else []

    # Ledger (input to VideoComposite re-run)
    ledger_candidates = list(audio_dir.glob(f"{ep_dir.name}*_ledger.json"))
    ledger_path = ledger_candidates[0] if ledger_candidates else None

    missing = []
    if composite_mp4 is None or not composite_mp4.exists():
        missing.append(
            f"composited/{ep_dir.name}*.mp4 (the VideoComposite output)"
        )
    if procgen_mp4 is None or not procgen_mp4.exists():
        missing.append(
            f"audio/{ep_dir.name}*.mp4 (the SignalLostVideo procgen output)"
        )
    if missing:
        print("MISSING required pre-rendered assets:")
        for m in missing:
            print(f"  - {m}")
        return None

    return {
        "ep_dir": ep_dir,
        "composite_mp4": composite_mp4,
        "procgen_mp4": procgen_mp4,
        "videos_dir": videos_dir,
        "per_line_clips": per_line_clips,
        "ledger_path": ledger_path,
    }


# ---------------------------------------------------------------------------
# ffprobe + sample-frame extraction
# ---------------------------------------------------------------------------

def ffprobe_summary(path: Path) -> dict:
    """Return dict with width, height, duration_s, bitrate_kbps, nb_frames."""
    if not shutil.which("ffprobe"):
        return {"error": "ffprobe not on PATH"}
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,duration,bit_rate,nb_frames",
            "-of", "default=nw=1",
            str(path),
        ], stderr=subprocess.PIPE).decode("utf-8")
    except Exception as exc:
        return {"error": f"ffprobe failed: {exc}"}
    d: dict = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    try:
        return {
            "width": int(d.get("width", 0)),
            "height": int(d.get("height", 0)),
            "duration_s": float(d.get("duration", 0.0)),
            "bitrate_kbps": int(d.get("bit_rate", 0)) // 1000,
            "nb_frames": int(d.get("nb_frames", 0)),
        }
    except Exception:
        return {"error": f"ffprobe parse failed; raw={d}"}


def extract_sample_frame(
    path: Path,
    out_png: Path,
    seek_seconds: float = 5.0,
) -> int | None:
    """Extract one frame at seek_seconds. Returns size in bytes or None."""
    if not shutil.which("ffmpeg"):
        return None
    try:
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(seek_seconds),
            "-i", str(path),
            "-frames:v", "1",
            str(out_png),
        ], check=True)
    except Exception:
        return None
    return out_png.stat().st_size if out_png.exists() else None


def report_stage(
    label: str,
    mp4: Path,
    smoke_dir: Path,
    r: StageReport,
):
    """Print ffprobe + sample-frame report for one mp4 in the chain."""
    if not mp4.exists():
        r.add(f"[{label}] MISSING: {mp4}")
        return
    info = ffprobe_summary(mp4)
    if "error" in info:
        r.add(f"[{label}] ffprobe error: {info['error']}")
        return
    size_mb = mp4.stat().st_size / (1024 * 1024)
    r.add(
        f"[{label}] {mp4.name}\n"
        f"   size = {size_mb:.2f} MB | dims = {info['width']}x{info['height']} "
        f"| duration = {info['duration_s']:.2f}s | "
        f"bitrate = {info['bitrate_kbps']} kbps | nb_frames = {info['nb_frames']}"
    )
    sample_png = smoke_dir / f"frame_{label}.png"
    sample_bytes = extract_sample_frame(mp4, sample_png, seek_seconds=5.0)
    if sample_bytes is not None:
        kb = sample_bytes / 1024.0
        verdict = ""
        if kb < 100:
            verdict = "  <-- SUSPECT: keyframe < 100 KB suggests near-solid color"
        r.add(f"   sample frame @ 5s = {kb:.1f} KB{verdict}")
        r.add(f"   sample saved: {sample_png}")


# ---------------------------------------------------------------------------
# Stage runners (call the actual nodes via direct Python import)
# ---------------------------------------------------------------------------

def run_video_composite(
    procgen_mp4: Path,
    videos_dir: Path,
    ledger_path: Path,
    out_mp4: Path,
    r: StageReport,
) -> bool:
    """Invoke OTR_VideoComposite.execute on pre-rendered per-line clips.

    Exercises the BUG-LOCAL-031 Track 1 per-clip duration matching
    fix in `_layered_per_clip_silent`. Output goes to the canonical
    composited/<ep>.mp4 path (clobbered by each smoke run).
    """
    r.section("STAGE 0 -- VideoComposite (re-runs the BUG-031 sync fix)")
    try:
        from nodes.video_composite import VideoComposite  # type: ignore
    except Exception as exc:
        r.add(f"FAIL: cannot import VideoComposite: {exc}")
        return False
    node = VideoComposite()
    try:
        # Defaults match the canonical workflow JSON Node 52 widgets.
        result_path, report = node.execute(
            procgen_video_path=str(procgen_mp4),
            clips_dir=str(videos_dir),
            ledger_json=str(ledger_path),
            blend_mode="lighten",
            blend_opacity=0.0,
            canvas_width=1472,
            canvas_height=832,
            canvas_fps=25,
            humo_target_height=832,
            humo_pillar_width=512,
            fallback_clip_length=7.0,
            ffmpeg="ffmpeg",
            cleanup_clips_after_assembly=False,
            audio_source="master_mix_per_clip_mux",
            strict_c7=True,
        )
    except Exception as exc:
        r.add(f"FAIL: VideoComposite.execute raised: {exc}")
        return False
    if not result_path:
        r.add(f"FAIL: VideoComposite returned empty path. report:\n{report}")
        return False
    src = Path(result_path)
    if not src.exists():
        r.add(f"FAIL: VideoComposite returned non-existent path: {src}")
        return False
    # Move into smoke dir so we don't clobber the canonical composite,
    # then ALSO leave a copy at canonical for downstream nodes to use.
    import shutil as _sh
    if src.resolve() != out_mp4.resolve():
        _sh.copy2(src, out_mp4)
    r.add(f"OK: composite at {out_mp4}")
    r.add(f"--- node report (truncated) ---\n{report[:1200]}")
    return True


def run_rtx_upscale(
    composite_mp4: Path,
    out_mp4: Path,
    bypass: bool,
    r: StageReport,
    diagnostic_dump_dir: Path | None = None,
) -> bool:
    """Invoke OTR_RTXUpscale.execute on a pre-rendered composite mp4.

    Writes output to ``out_mp4``. Returns True on success.
    When ``diagnostic_dump_dir`` is set, RTXUpscale will dump per-chunk
    PNGs + a chunk_stats.txt file there for BUG-LOCAL-031 forensics.
    """
    r.section(f"STAGE 1 -- RTXUpscale (bypass={bypass})")
    try:
        from nodes.rtx_upscale import RTXUpscale  # type: ignore
    except Exception as exc:
        r.add(f"FAIL: cannot import RTXUpscale: {exc}")
        return False

    node = RTXUpscale()
    # The node writes to otr_obs_dir() / src.name by convention. We then
    # MOVE it to our smoke dir to avoid clobbering the canonical obs/.
    try:
        kwargs = dict(
            source_mp4_path=str(composite_mp4),
            bypass=bool(bypass),
        )
        if diagnostic_dump_dir is not None:
            kwargs["diagnostic_dump_dir"] = str(diagnostic_dump_dir)
            r.add(f"   diagnostic_dump_dir = {diagnostic_dump_dir}")
        result_path, report = node.execute(**kwargs)
    except Exception as exc:
        r.add(f"FAIL: RTXUpscale.execute raised: {exc}")
        return False

    if not result_path:
        r.add(f"FAIL: RTXUpscale returned empty path. report:\n{report}")
        return False

    src = Path(result_path)
    if not src.exists():
        r.add(f"FAIL: RTXUpscale returned non-existent path: {src}")
        return False

    # Move into smoke dir so we don't clobber the canonical obs/ output.
    if src.resolve() != out_mp4.resolve():
        shutil.move(str(src), str(out_mp4))

    r.add(f"OK: wrote {out_mp4}")
    r.add(f"--- node report (truncated) ---\n{report[:600]}")
    return True


def run_post_blend(
    upscaled_mp4: Path,
    procgen_mp4: Path,
    out_mp4: Path,
    r: StageReport,
) -> bool:
    """Invoke OTR_PostUpscaleProcgenBlend.blend on the upscaled mp4 +
    pre-rendered procgen mp4. Writes to out_mp4."""
    r.section("STAGE 2 -- PostUpscaleProcgenBlend")
    try:
        from nodes.otr_post_upscale_procgen_blend import (  # type: ignore
            PostUpscaleProcgenBlend,
        )
    except Exception as exc:
        r.add(f"FAIL: cannot import PostUpscaleProcgenBlend: {exc}")
        return False

    node = PostUpscaleProcgenBlend()
    try:
        result_path, report = node.blend(
            source_mp4_path=str(upscaled_mp4),
            procgen_mp4_path=str(procgen_mp4),
        )
    except Exception as exc:
        r.add(f"FAIL: PostUpscaleProcgenBlend.blend raised: {exc}")
        return False

    if not result_path:
        r.add(f"FAIL: PostUpscaleProcgenBlend returned empty path. report:\n{report}")
        return False

    src = Path(result_path)
    if not src.exists():
        r.add(f"FAIL: PostUpscaleProcgenBlend returned non-existent path: {src}")
        return False

    # Move into smoke dir.
    if src.resolve() != out_mp4.resolve():
        shutil.move(str(src), str(out_mp4))

    r.add(f"OK: wrote {out_mp4}")
    r.add(f"--- node report (truncated) ---\n{report[:600]}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="OTR downstream-only smoke harness")
    p.add_argument("--episode-id", default=None,
                   help="Episode slug (default: most recent)")
    p.add_argument("--upscale-bypass", action="store_true",
                   help="Skip RTX VSR; just copy composite to obs/ via the "
                        "node's bypass mode. Use to isolate whether the "
                        "RTX VSR step itself or the surrounding ffmpeg "
                        "pipeline is the black-output culprit.")
    p.add_argument("--inspect-only", action="store_true",
                   help="Skip RTXUpscale + PostUpscaleProcgenBlend invocations; "
                        "only ffprobe + sample-frame report on the existing "
                        "input assets so we can see the input shape without "
                        "running anything.")
    p.add_argument("--diagnostic-dump", action="store_true",
                   help="Pass diagnostic_dump_dir into RTXUpscale so it dumps "
                        "per-chunk PNGs (input_uint8, post_nvvfx_float, "
                        "post_clamp_byte) + chunk_stats.txt for BUG-LOCAL-031 "
                        "forensics. Dumps land in the smoke dir under "
                        "rtx_diag/. Use to localize where in the chunk "
                        "pipeline the visual content dies.")
    args = p.parse_args()

    print("== smoke_downstream_from_assets.py ==")
    print()

    ep_dir = find_episode_dir(args.episode_id)
    if ep_dir is None:
        print(f"No episode dir found under {EPISODES_DIR}")
        return 1
    print(f"Target episode: {ep_dir.name}")

    assets = gather_assets(ep_dir)
    if assets is None:
        return 1

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    smoke_dir = ep_dir / f"_smoke_{ts}"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    print(f"Smoke output dir: {smoke_dir}")

    r = StageReport()
    r.section("INPUT ASSETS (pre-rendered)")
    report_stage("input_composite", assets["composite_mp4"], smoke_dir, r)
    report_stage("input_procgen", assets["procgen_mp4"], smoke_dir, r)

    if args.inspect_only:
        r.section("INSPECT-ONLY mode -- skipping node invocations")
        (smoke_dir / "report.txt").write_text("\n".join(r.lines), encoding="utf-8")
        return 0

    # Stage 1: RTXUpscale
    upscaled_mp4 = smoke_dir / f"smoke_rtx_{assets['composite_mp4'].name}"
    rtx_diag_dir = (smoke_dir / "rtx_diag") if args.diagnostic_dump else None
    ok = run_rtx_upscale(
        composite_mp4=assets["composite_mp4"],
        out_mp4=upscaled_mp4,
        bypass=args.upscale_bypass,
        r=r,
        diagnostic_dump_dir=rtx_diag_dir,
    )
    if not ok:
        (smoke_dir / "report.txt").write_text("\n".join(r.lines), encoding="utf-8")
        return 1
    report_stage("rtx_upscale_out", upscaled_mp4, smoke_dir, r)

    # Stage 2: PostUpscaleProcgenBlend
    blended_mp4 = smoke_dir / (
        upscaled_mp4.stem + "_procgen_blended.mp4"
    )
    ok = run_post_blend(
        upscaled_mp4=upscaled_mp4,
        procgen_mp4=assets["procgen_mp4"],
        out_mp4=blended_mp4,
        r=r,
    )
    if not ok:
        (smoke_dir / "report.txt").write_text("\n".join(r.lines), encoding="utf-8")
        return 1
    report_stage("post_blend_out", blended_mp4, smoke_dir, r)

    r.section("DIAGNOSTIC SUMMARY")
    r.add(
        "If 'rtx_upscale_out' bitrate collapsed (e.g. < 200 kbps) and the\n"
        "sample frame is < 100 KB, RTXUpscale is the bug site.\n"
        "If 'post_blend_out' duration > input_composite duration, the blend\n"
        "filter is over-running on the longer procgen input."
    )

    (smoke_dir / "report.txt").write_text("\n".join(r.lines), encoding="utf-8")
    print(f"\nFull report saved: {smoke_dir / 'report.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
