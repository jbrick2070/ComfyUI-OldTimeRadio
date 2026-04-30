r"""qa_waveforms.py -- per-line waveform PNGs for OTR audio QA.

Renders one matplotlib waveform PNG per dialogue line in
ledger.lines[], plus an episode-summary plot showing the full audio
with vertical markers at every line + scene boundary. Useful for
catching:

  - Bark artifacts (clipping, dead air, sudden silence)
  - Pre-pad violations (BUG-102: dialogue starts in the prepad zone)
  - AudioEnhance over-ducking
  - Mistimed scene transitions in SceneSequencer

Output:
    <ComfyUI/output>/otr/qa_waveforms/<episode_id>/
        manifest.json
        episode.png                              full-episode waveform
        line_001_NAME_t00.50_d2.30.png           per-line slice
        line_002_NAME_t02.80_d3.10.png
        ...

Each per-line PNG shows the raw waveform with markers for prepad
silence (left), the dialogue region, and any tail silence (right).
The x-axis is seconds since line start.

Usage:
    C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe ^
        scripts\qa_waveforms.py --ledger PATH\TO\<episode_id>_ledger.json

    # or pull the most recent ledger:
    python scripts\qa_waveforms.py --latest

Requires: matplotlib, numpy, scipy.io.wavfile (or soundfile fallback).
All three are already pinned by other OTR nodes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [qa_waveforms] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("qa_waveforms")

REPO_ROOT = Path(__file__).resolve().parent.parent
COMFY_OUT = REPO_ROOT.parent.parent / "output"
LEDGER_DIR = COMFY_OUT / "otr" / "audio"


def _find_latest_ledger() -> Optional[Path]:
    if not LEDGER_DIR.exists():
        return None
    cands = list(LEDGER_DIR.glob("*_ledger.json"))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _extract_audio_from_video(video_path: Path) -> Optional[Path]:
    """Extract audio track from an .mp4/.mov to a sibling .wav.

    Used when ledger.final_audio_path is null but final_video_path
    points at a finished episode (the typical SignalLostVideo case).
    Returns the wav path on success, None on failure.
    """
    import shutil
    import subprocess
    import tempfile
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    out_path = Path(tempfile.gettempdir()) / f"qa_audio_{video_path.stem}.wav"
    cmd = [
        ffmpeg, "-y",
        "-i", str(video_path),
        "-vn",                    # drop video
        "-ac", "1",               # mono
        "-ar", "48000",           # 48 kHz
        "-c:a", "pcm_s16le",      # 16-bit PCM
        str(out_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
        )
        if result.returncode != 0:
            log.warning(
                "ffmpeg audio-extract failed (rc=%d): %s",
                result.returncode,
                (result.stderr or b"").decode(errors="replace")[:200],
            )
            return None
        if out_path.exists():
            log.info("extracted audio from video: %s", out_path)
            return out_path
    except Exception as exc:  # noqa: BLE001
        log.warning("ffmpeg audio-extract error: %s", exc)
    return None


def _load_audio(audio_path: Path) -> Tuple[any, int]:
    """Load WAV file, return (samples_mono_float32, sample_rate).

    Tries soundfile first (handles 32-bit float, multi-channel cleanly),
    falls back to scipy.io.wavfile, then to ffmpeg-extract-and-retry.
    """
    import numpy as np

    # Try soundfile first
    try:
        import soundfile as sf  # type: ignore
        data, sr = sf.read(str(audio_path), always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        return data.astype(np.float32), int(sr)
    except Exception as sf_exc:  # noqa: BLE001
        log.debug("soundfile load failed (%s) - trying scipy", sf_exc)

    try:
        from scipy.io import wavfile  # type: ignore
        sr, data = wavfile.read(str(audio_path))
        if data.ndim == 2:
            data = data.mean(axis=1)
        # Normalize to float32 in [-1, 1]
        if data.dtype.kind in ("i", "u"):
            max_val = float(np.iinfo(data.dtype).max)
            data = data.astype(np.float32) / max_val
        else:
            data = data.astype(np.float32)
        return data, int(sr)
    except Exception as sp_exc:
        raise RuntimeError(
            f"Could not load audio from {audio_path}: "
            f"soundfile + scipy both failed ({sp_exc})"
        )


def _decimate_for_plot(samples, max_pts: int = 4000):
    """Downsample a long waveform for plotting without losing peaks."""
    import numpy as np
    n = len(samples)
    if n <= max_pts:
        return np.arange(n), samples
    bucket = n // max_pts
    # Reshape and take min/max per bucket so peaks survive
    trimmed = samples[: bucket * max_pts]
    reshaped = trimmed.reshape(max_pts, bucket)
    mins = reshaped.min(axis=1)
    maxs = reshaped.max(axis=1)
    # Interleave mins and maxs so the plot still looks like a waveform
    idx = np.arange(max_pts) * bucket
    return idx, np.where(np.abs(maxs) > np.abs(mins), maxs, mins)


def _render_line_plot(
    samples,
    sr: int,
    line_idx: int,
    name: str,
    text: str,
    out_path: Path,
    prepad_s: float = 0.0,
) -> bool:
    """Render one per-line waveform PNG with prepad/dialogue markers."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        log.error("matplotlib/numpy unavailable: %s", exc)
        return False

    duration_s = len(samples) / sr if sr else 0.0
    fig, ax = plt.subplots(figsize=(10, 2.5), dpi=110)
    idx, decim = _decimate_for_plot(samples)
    t = idx / float(sr) if sr else idx
    ax.plot(t, decim, linewidth=0.6, color="#1e6091")

    # Mark prepad region if non-zero
    if prepad_s > 0:
        ax.axvspan(0, prepad_s, color="#fbe8a6", alpha=0.7, label=f"prepad {prepad_s:.2f}s")

    # Auto-detect tail silence (last 100ms with abs() < 1% of peak)
    if len(decim) > 0 and np.max(np.abs(decim)) > 0:
        thresh = 0.01 * float(np.max(np.abs(decim)))
        tail_window = max(int(0.1 * sr), 1)
        # Walk backwards
        end = len(samples)
        i = end - 1
        while i > 0 and abs(samples[i]) < thresh:
            i -= 1
        tail_start_s = (i + 1) / float(sr) if sr else duration_s
        if duration_s - tail_start_s > 0.05:
            ax.axvspan(tail_start_s, duration_s, color="#d6e4f0", alpha=0.7,
                       label=f"tail silence {duration_s - tail_start_s:.2f}s")

    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_xlim(0, max(duration_s, 0.1))
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    title = f"line {line_idx:03d}  [{name}]  {duration_s:.2f}s"
    if text:
        # Truncate long dialogue for the title
        display_text = text if len(text) <= 80 else text[:77] + "..."
        title += f"\n{display_text}"
    ax.set_title(title, fontsize=9, loc="left")
    # Only show legend if any artist actually got a label (prepad /
    # tail-silence regions). Otherwise matplotlib emits a noisy
    # UserWarning about "no artists with labels".
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=110)
        return True
    finally:
        plt.close(fig)


def _render_episode_plot(
    samples,
    sr: int,
    lines: list[dict],
    scenes: list[dict],
    out_path: Path,
) -> bool:
    """Render the full-episode waveform with line + scene markers."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        log.error("matplotlib/numpy unavailable: %s", exc)
        return False

    duration_s = len(samples) / sr if sr else 0.0
    fig, ax = plt.subplots(figsize=(16, 4), dpi=110)
    idx, decim = _decimate_for_plot(samples, max_pts=8000)
    t = idx / float(sr) if sr else idx
    ax.plot(t, decim, linewidth=0.5, color="#1e6091")

    # Vertical markers per line start
    for ln in lines:
        st = float(ln.get("start_s", 0.0))
        ax.axvline(st, color="#bb6622", linewidth=0.5, alpha=0.6)

    # Scene boundaries (dashed, more prominent)
    for sc in scenes:
        st = float(sc.get("start_s", 0.0))
        if st > 0:
            ax.axvline(st, color="#22aa44", linewidth=1.0, linestyle="--", alpha=0.7)

    ax.axhline(0, color="black", linewidth=0.3)
    ax.set_xlim(0, max(duration_s, 0.1))
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("amplitude")
    ax.set_title(
        f"episode waveform   {duration_s:.1f}s   "
        f"({len(lines)} lines, {len(scenes)} scenes)",
        fontsize=10, loc="left",
    )
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    try:
        fig.savefig(out_path, dpi=110)
        return True
    finally:
        plt.close(fig)


def _slug(text: str) -> str:
    """Filesystem-safe slug for inclusion in filenames."""
    out = "".join(c if c.isalnum() else "_" for c in (text or ""))
    return (out or "X")[:24]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ledger", type=Path)
    g.add_argument("--latest", action="store_true")
    ap.add_argument(
        "--audio", type=Path,
        help="Override audio path (default = ledger.final_audio_path)",
    )
    args = ap.parse_args()

    ledger_path = args.ledger if args.ledger else _find_latest_ledger()
    if not ledger_path or not ledger_path.exists():
        log.error("ledger not found: %s", ledger_path)
        return 2
    log.info("ledger: %s", ledger_path)

    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("failed to parse ledger: %s", exc)
        return 2

    episode_id = ledger.get("episode_id") or ledger_path.stem.replace("_ledger", "")

    audio_path: Optional[Path] = args.audio
    if audio_path is None:
        fa = ledger.get("final_audio_path")
        if fa:
            audio_path = Path(fa)
    if audio_path is None or not audio_path.exists():
        # Fallback: extract audio from final_video_path. Standard for
        # episodes that ran SignalLostVideo -- the wav gets baked
        # into the mp4 and the original is no longer kept.
        fv = ledger.get("final_video_path")
        if fv and Path(fv).exists():
            log.info(
                "final_audio_path missing/null - extracting from video: %s",
                fv,
            )
            audio_path = _extract_audio_from_video(Path(fv))
        if audio_path is None or not audio_path.exists():
            log.error(
                "no audio source found. final_audio_path=%r, "
                "final_video_path=%r",
                ledger.get("final_audio_path"),
                ledger.get("final_video_path"),
            )
            return 2
    log.info("audio: %s", audio_path)

    try:
        samples, sr = _load_audio(audio_path)
    except Exception as exc:
        log.error("audio load failed: %s", exc)
        return 2
    log.info("audio loaded: %d samples @ %d Hz (%.1fs)",
             len(samples), sr, len(samples) / sr if sr else 0.0)

    lines = ledger.get("lines") or []
    scenes = ledger.get("scenes") or []

    out_root = COMFY_OUT / "otr" / "qa_waveforms" / episode_id
    out_root.mkdir(parents=True, exist_ok=True)

    # Episode-level summary plot
    ep_path = out_root / "episode.png"
    log.info("rendering episode summary -> %s", ep_path.name)
    _render_episode_plot(samples, sr, lines, scenes, ep_path)

    # Per-line slices
    manifest: list[dict] = []
    success = 0
    for i, ln in enumerate(lines, start=1):
        st = float(ln.get("start_s", 0.0))
        dur = float(ln.get("dur_s", 0.0))
        prepad = float(ln.get("prepad_s", 0.0) or 0.0)
        name = ln.get("voice") or ln.get("character") or "X"
        text = ln.get("text") or ""
        if dur <= 0:
            continue
        i0 = max(0, int(st * sr))
        i1 = min(len(samples), int((st + dur) * sr))
        if i1 <= i0:
            continue
        slug = _slug(name)
        fname = f"line_{i:03d}_{slug}_t{st:06.2f}_d{dur:05.2f}.png"
        fpath = out_root / fname
        ok = _render_line_plot(
            samples[i0:i1], sr, i, name, text, fpath, prepad_s=prepad,
        )
        if ok:
            success += 1
        manifest.append({
            "index":     i,
            "filename":  fname,
            "voice":     name,
            "start_s":   round(st, 3),
            "dur_s":     round(dur, 3),
            "prepad_s":  round(prepad, 3),
            "exists":    ok,
        })

    (out_root / "manifest.json").write_text(
        json.dumps(
            {
                "episode_id":   episode_id,
                "audio_path":   str(audio_path),
                "ledger_path":  str(ledger_path),
                "sample_rate":  sr,
                "duration_s":   round(len(samples) / sr if sr else 0.0, 3),
                "line_count":   len(lines),
                "scene_count":  len(scenes),
                "lines":        manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log.info(
        "wrote %d/%d line plots + episode summary to %s",
        success, len(lines), out_root,
    )
    return 0 if success > 0 or len(lines) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
