#!/usr/bin/env python
"""Measure the audio-to-lip offset of a rendered talking-head clip.

WHY THIS EXISTS. `BUG_BIBLE.yaml:2343` records that HuMo audio leads the lips by
100-200 ms with the face static for the first 3-6 frames. The prescribed remedy
-- pre-roll the audio and trim an equal number of frames -- **cannot be built
until the error is classified**, because pre-roll plus an equal trim is
algebraically a NO-OP when the lag is constant rather than onset-only. This tool
produces the classification:

    EARLY-ONLY   lag is large in the first window and ~0 later
                 -> the conditioning starts late; pre-roll IS the fix.
    CONSTANT     lag is the same in early, middle and late windows
                 -> advance the 25 Hz conditioning features instead; a pre-roll
                    plus an equal trim would cancel and change nothing.
    GROWING      lag increases across the clip
                 -> a rate or timestamp bug (feature Hz vs fps), not a pad.

HOW IT MEASURES. Two one-dimensional signals are built at exactly the video
frame rate and cross-correlated:

  * VIDEO -- mouth articulation energy. The face box is found once (OpenCV
    frontal Haar, median over sampled frames), the mouth ROI is its lower band,
    and the per-frame signal is the mean absolute difference between consecutive
    frames inside that ROI. Speech articulation is the dominant motion there.
  * AUDIO -- `librosa` onset strength over the same window, with the hop set to
    `sample_rate / fps` so one envelope value lands on one video frame.

SIGN CONVENTION, stated once and asserted by the self-test:

    lag_ms > 0  means the VIDEO LAGS THE AUDIO -- i.e. audio leads the lips,
                which is the direction BUG_BIBLE:2343 describes.

`--self-test` proves the sign and the accuracy against synthetic ground truth
before any real number is believed. Run it first; a measurement from an
uncalibrated instrument is an opinion.

Usage:

    # prove the instrument
    python scripts/otr_measure_av_offset.py --self-test

    # measure one clip against its window of the episode master
    python scripts/otr_measure_av_offset.py \
        --clip   otr/episodes/<ep>/clips/shot_l001_character_video_humo.mp4 \
        --audio  otr/episodes/<ep>/audio/<master>.wav \
        --audio-start 9.5 --audio-dur 7.91075

    # every HuMo clip in an episode, paired with the ledger's line timing
    python scripts/otr_measure_av_offset.py --episode otr/episodes/<ep> --engine humo
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

#: Lag search range. 1000 ms each way is far wider than the 100-200 ms the bug
#: describes, so a mis-classification shows up as a peak near the edge rather
#: than being silently clipped into range.
MAX_LAG_MS = 1000.0

#: A cross-correlation peak below this is not a measurement. Real articulation
#: against real speech onsets correlates well; a static face or a music bed does
#: not, and reporting its argmax as "the offset" would be noise with a decimal
#: point on it.
MIN_PEAK_CORR = 0.15


# --------------------------------------------------------------------------- #
# decode
# --------------------------------------------------------------------------- #
def probe_video(path):
    """(fps, width, height, nb_frames) for the first video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=r_frame_rate,width,height,nb_frames", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    num, _, den = st["r_frame_rate"].partition("/")
    fps = float(num) / float(den or 1)
    nb = st.get("nb_frames")
    return fps, int(st["width"]), int(st["height"]), (int(nb) if nb else 0)


def decode_gray(path, width, height):
    """Whole clip as a uint8 [N,H,W] grayscale array."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True)
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = buf.size // (width * height)
    if n == 0:
        raise SystemExit("decoded zero frames from %s" % path)
    return buf[:n * width * height].reshape(n, height, width)


# --------------------------------------------------------------------------- #
# the two signals
# --------------------------------------------------------------------------- #
def find_mouth_roi(frames):
    """(y0, y1, x0, x1) for the mouth band.

    The face box is the median OpenCV frontal-Haar detection over sampled
    frames; the mouth band is its lower 45%, horizontally centred at 70%. When
    no face is found -- a stylised or non-frontal render -- fall back to the
    lower-middle of the frame, which is where a framed talking head's mouth sits
    anyway. The fallback is REPORTED, because a measurement taken on the wrong
    rectangle should never be mistaken for one taken on a face.
    """
    import cv2

    n, h, w = frames.shape
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    boxes = []
    for idx in np.linspace(0, n - 1, min(n, 24), dtype=int):
        found = cascade.detectMultiScale(frames[idx], scaleFactor=1.15,
                                         minNeighbors=5,
                                         minSize=(max(24, w // 12), max(24, h // 12)))
        for (x, y, bw, bh) in found:
            boxes.append((x, y, bw, bh))

    if len(boxes) >= 3:
        arr = np.array(boxes, dtype=float)
        x, y, bw, bh = np.median(arr, axis=0)
        y0 = int(y + 0.55 * bh)
        y1 = int(min(h, y + 1.05 * bh))
        x0 = int(x + 0.15 * bw)
        x1 = int(min(w, x + 0.85 * bw))
        if y1 - y0 >= 8 and x1 - x0 >= 8:
            return (y0, y1, x0, x1), "face(%d detections)" % len(boxes)

    return (int(h * 0.55), int(h * 0.90), int(w * 0.25), int(w * 0.75)), "FALLBACK-no-face"


def video_motion_signal(frames, roi):
    """Per-frame mouth articulation energy, length N (first element 0)."""
    y0, y1, x0, x1 = roi
    band = frames[:, y0:y1, x0:x1].astype(np.float32)
    diff = np.abs(np.diff(band, axis=0)).mean(axis=(1, 2))
    return np.concatenate([[0.0], diff])


def audio_onset_signal(wav_path, fps, start_s=None, dur_s=None, sr=16000):
    """Onset-strength envelope sampled at exactly `fps` values per second."""
    import librosa

    y, _ = librosa.load(str(wav_path), sr=sr, mono=True,
                        offset=float(start_s or 0.0),
                        duration=(float(dur_s) if dur_s else None))
    if y.size == 0:
        raise SystemExit("empty audio window from %s" % wav_path)
    hop = int(round(sr / float(fps)))
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    return env.astype(np.float32)


# --------------------------------------------------------------------------- #
# correlation
# --------------------------------------------------------------------------- #
def _z(sig):
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig - sig.mean()
    norm = np.linalg.norm(sig)
    return sig / norm if norm > 1e-12 else sig


def measure_lag(video_sig, audio_sig, fps, max_lag_ms=MAX_LAG_MS):
    """Lag in ms at peak correlation, positive when VIDEO LAGS AUDIO.

    Both signals are trimmed to a common length and z-normalised, so the peak
    value is a correlation coefficient in [-1, 1] and comparable between clips.
    The peak is refined by parabolic interpolation, which recovers sub-frame
    resolution -- necessary because one frame is 40 ms and the effect under test
    is 100-200 ms.
    """
    n = min(len(video_sig), len(audio_sig))
    if n < 8:
        return None
    vid, aud = _z(video_sig[:n]), _z(audio_sig[:n])

    max_lag = int(round(max_lag_ms / 1000.0 * fps))
    max_lag = max(1, min(max_lag, n - 2))

    corr = np.correlate(vid, aud, mode="full")
    centre = len(vid) - 1
    lags = np.arange(-max_lag, max_lag + 1)
    window = corr[centre - max_lag: centre + max_lag + 1]
    if window.size != lags.size:
        return None

    k = int(np.argmax(window))
    peak = float(window[k])
    best = float(lags[k])

    if 0 < k < len(window) - 1:                       # parabolic refinement
        a, b, c = window[k - 1], window[k], window[k + 1]
        denom = (a - 2.0 * b + c)
        if abs(denom) > 1e-12:
            best += float(np.clip(0.5 * (a - c) / denom, -1.0, 1.0))

    return {"lag_ms": best / fps * 1000.0, "lag_frames": best, "corr": peak}


def windowed_lags(video_sig, audio_sig, fps, parts=3):
    """Lag measured independently over EARLY / MIDDLE / LATE thirds."""
    names = ["EARLY", "MIDDLE", "LATE"] if parts == 3 else \
            ["W%d" % i for i in range(parts)]
    n = min(len(video_sig), len(audio_sig))
    edges = np.linspace(0, n, parts + 1, dtype=int)
    out = []
    for i in range(parts):
        lo, hi = edges[i], edges[i + 1]
        got = measure_lag(video_sig[lo:hi], audio_sig[lo:hi], fps)
        out.append({"window": names[i], "from_s": lo / fps, "to_s": hi / fps,
                    **(got or {"lag_ms": None, "corr": 0.0})})
    return out


def classify(windows, whole):
    """EARLY-ONLY / CONSTANT / GROWING / INCONCLUSIVE from the window lags."""
    usable = [w for w in windows if w.get("lag_ms") is not None
              and w.get("corr", 0) >= MIN_PEAK_CORR]
    if len(usable) < 2:
        return "INCONCLUSIVE", "fewer than two windows cleared corr >= %.2f" % MIN_PEAK_CORR

    lags = [w["lag_ms"] for w in usable]
    first, last = lags[0], lags[-1]
    spread = max(lags) - min(lags)
    later = lags[1:]
    later_mean = sum(later) / len(later)

    if spread <= 40.0:
        return "CONSTANT", ("all windows within %.0f ms (%.0f..%.0f); a pre-roll "
                            "plus an equal trim would cancel -- advance the 25 Hz "
                            "conditioning features instead" % (spread, min(lags), max(lags)))
    if abs(first) >= 60.0 and abs(later_mean) <= 40.0 and abs(first) > abs(later_mean) * 2:
        return "EARLY-ONLY", ("first window %.0f ms vs %.0f ms later; the "
                              "conditioning starts late -- pre-roll IS the fix"
                              % (first, later_mean))
    if abs(last) > abs(first) + 60.0 and (last - first) * (1 if last > 0 else -1) > 0:
        return "GROWING", ("%.0f ms -> %.0f ms across the clip; this is a rate or "
                           "timestamp bug (feature Hz vs fps), not a pad"
                           % (first, last))
    return "MIXED", ("windows %s -- spread %.0f ms with no clean early/constant/"
                     "growing shape" % (["%.0f" % v for v in lags], spread))


# --------------------------------------------------------------------------- #
# self-test: prove sign and accuracy on synthetic ground truth
# --------------------------------------------------------------------------- #
def self_test():
    """Build signals with a KNOWN injected offset and assert the sign.

    A pulse train is the audio; the video signal is the same train delayed by a
    known number of frames. Delaying the video must report a POSITIVE lag -- the
    'audio leads the lips' direction. This is the calibration the plan demands
    before any real number is trusted.
    """
    fps = 25.0
    n = 500
    rng = np.random.default_rng(7)

    audio = np.zeros(n, dtype=np.float64)
    onsets = rng.choice(np.arange(10, n - 10), size=60, replace=False)
    audio[onsets] = 1.0
    audio = np.convolve(audio, np.array([0.2, 0.6, 1.0, 0.6, 0.2]), mode="same")
    audio += rng.normal(0, 0.02, n)

    print("self-test -- sign convention: lag_ms > 0 means VIDEO LAGS AUDIO\n")
    failures = []
    for truth_frames in (0, 1, 3, 5, -3):
        video = np.roll(audio, truth_frames)
        if truth_frames > 0:
            video[:truth_frames] = 0.0
        elif truth_frames < 0:
            video[truth_frames:] = 0.0
        video = video + rng.normal(0, 0.02, n)

        got = measure_lag(video, audio, fps)
        truth_ms = truth_frames / fps * 1000.0
        err = got["lag_ms"] - truth_ms
        ok = abs(err) <= 12.0 and got["corr"] >= 0.5
        status = "OK " if ok else "FAIL"
        print("  %s injected %+6.1f ms -> measured %+7.1f ms  (err %+5.1f, corr %.3f)"
              % (status, truth_ms, got["lag_ms"], err, got["corr"]))
        if not ok:
            failures.append(truth_ms)

    if failures:
        print("\nSELF-TEST FAILED for %s -- do not trust any measurement." % failures)
        return 1
    print("\nSELF-TEST PASSED: sign is correct and error stays under half a frame.")
    return 0


# --------------------------------------------------------------------------- #
# one clip / one episode
# --------------------------------------------------------------------------- #
def measure_clip(clip, wav, audio_start, audio_dur, shift_ms=0.0, label=""):
    fps, w, h, _ = probe_video(clip)
    frames = decode_gray(clip, w, h)
    roi, roi_how = find_mouth_roi(frames)
    vid = video_motion_signal(frames, roi)

    start = float(audio_start or 0.0) + shift_ms / 1000.0
    aud = audio_onset_signal(wav, fps, start, audio_dur)

    whole = measure_lag(vid, aud, fps)
    wins = windowed_lags(vid, aud, fps)
    verdict, why = classify(wins, whole)

    return {
        "label": label or pathlib.Path(clip).name,
        "clip": str(clip), "fps": fps, "frames": int(frames.shape[0]),
        "canvas": "%dx%d" % (w, h), "roi": roi, "roi_source": roi_how,
        "audio_start_s": start, "audio_dur_s": audio_dur,
        "shift_ms": shift_ms,
        "whole": whole, "windows": wins,
        "verdict": verdict, "why": why,
    }


def report(res):
    print("\n=== %s ===" % res["label"])
    print("  %s @ %.3g fps, %d frames | mouth ROI %s from %s"
          % (res["canvas"], res["fps"], res["frames"], res["roi"], res["roi_source"]))
    if res["shift_ms"]:
        print("  CONTROL: audio deliberately shifted %+.0f ms" % res["shift_ms"])
    whole = res["whole"]
    if whole:
        print("  WHOLE CLIP   lag %+7.1f ms (%+.2f frames)  corr %.3f"
              % (whole["lag_ms"], whole["lag_frames"], whole["corr"]))
    for w in res["windows"]:
        if w.get("lag_ms") is None:
            print("  %-6s       (no measurement)" % w["window"])
            continue
        flag = "" if w["corr"] >= MIN_PEAK_CORR else "   <-- weak, ignored"
        print("  %-6s %4.1f-%4.1fs  lag %+7.1f ms  corr %.3f%s"
              % (w["window"], w["from_s"], w["to_s"], w["lag_ms"], w["corr"], flag))
    print("  VERDICT: %s -- %s" % (res["verdict"], res["why"]))


def episode_clips(episode, engine):
    """[(clip_path, line_start_s, line_dur_s)] for one engine, from the ledger."""
    episode = pathlib.Path(episode)
    ledgers = list(episode.glob("audio/*_ledger.json")) + list(episode.glob("*_ledger.json"))
    if not ledgers:
        raise SystemExit("no ledger under %s" % episode)
    blob = json.loads(ledgers[0].read_text(encoding="utf-8"))

    timing = {ln.get("line_id"): (ln.get("start_s"), ln.get("dur_s"))
              for ln in blob.get("lines", []) if ln.get("line_id")}

    masters = sorted(episode.glob("audio/*master.wav"))
    if not masters:
        raise SystemExit("no master wav under %s/audio" % episode)

    out = []
    for clip in sorted((episode / "clips").glob("*.mp4")):
        if engine not in clip.stem:
            continue
        line_id = clip.stem.split("_")[1] if "_" in clip.stem else None
        start, dur = timing.get(line_id, (None, None))
        if start is None:
            print("  (skipping %s -- no ledger timing for line %r)" % (clip.name, line_id))
            continue
        out.append((clip, float(start), float(dur), masters[0]))
    return out


def static_onset_frames(frames, roi=None):
    """How many leading frames are STATIC before motion begins.

    BUG-07.13's ``cause`` field asserts the model "hold[s] the reference portrait
    static for ~3-6 output frames before motion onset". That is a claim about
    VIDEO ALONE -- no audio, no correlation, no SyncNet -- and it is therefore
    the cheapest thing in M1 that can discriminate anything. Either the leading
    freeze is there and this counts it exactly, or it is not and the Bible's own
    cause field is wrong.

    The threshold is derived from the CLIP'S OWN noise floor rather than a fixed
    constant: a dark scene and a bright one have very different inter-frame
    jitter at CRF 18, and a constant would silently mean different things on
    each. The floor is the 10th percentile of consecutive-frame differences,
    which is a quiet moment in this clip; motion is anything several times that.

    Returns (static_leading_frames, threshold, diffs).
    """
    band = frames if roi is None else frames[:, roi[0]:roi[1], roi[2]:roi[3]]
    band = band.astype(np.float32)
    diffs = np.abs(np.diff(band, axis=0)).mean(axis=(1, 2))
    if diffs.size < 4:
        return 0, 0.0, diffs

    floor = float(np.percentile(diffs, 10))
    ceiling = float(np.percentile(diffs, 90))
    # A clip with no motion at all has floor ~= ceiling; guard so the threshold
    # cannot collapse onto the noise and report the whole clip as "static".
    threshold = max(floor * 3.0, floor + 0.25 * max(1e-6, ceiling - floor))

    n_static = 0
    for d in diffs:
        if d >= threshold:
            break
        n_static += 1
    return n_static, threshold, diffs


def run_static_onset(episodes_root, engine="humo", limit=None):
    """Step 1 of M1: the leading-static-frame distribution over every episode.

    Reports per clip and then the distribution, because the research report's
    standing discipline is the median across lines and seeds -- never one
    attractive clip.
    """
    root = pathlib.Path(episodes_root)
    eps = sorted((d for d in root.iterdir() if d.is_dir()),
                 key=lambda d: d.stat().st_mtime, reverse=True)
    rows = []
    scanned = 0
    for ep in eps:
        clips = sorted((ep / "clips").glob("*%s*.mp4" % engine)) \
            if (ep / "clips").is_dir() else []
        if not clips:
            continue
        scanned += 1
        for clip in clips:
            try:
                fps, w, h, _ = probe_video(clip)
                frames = decode_gray(clip, w, h)
            except Exception as exc:
                rows.append({"clip": clip.name, "episode": ep.name,
                             "error": str(exc)[:60]})
                continue
            roi, roi_how = find_mouth_roi(frames)
            n_static, thr, diffs = static_onset_frames(frames, roi)
            rows.append({
                "episode": ep.name, "clip": clip.name,
                "frames": int(frames.shape[0]), "fps": fps,
                "static_frames": n_static,
                "static_ms": n_static / fps * 1000.0,
                "threshold": thr, "roi_source": roi_how,
            })
        if limit and scanned >= limit:
            break

    good = [r for r in rows if "error" not in r]
    print("episodes with %s clips scanned: %d | clips measured: %d"
          % (engine, scanned, len(good)))
    print("\n%-46s %6s %7s %8s  %s" % ("CLIP", "FRAMES", "STATIC", "MS", "ROI"))
    for r in good:
        print("%-46s %6d %7d %8.0f  %s" % (
            r["clip"][:46], r["frames"], r["static_frames"], r["static_ms"],
            r["roi_source"]))

    if good:
        vals = np.array([r["static_frames"] for r in good], dtype=float)
        ms = np.array([r["static_ms"] for r in good], dtype=float)
        print("\n--- DISTRIBUTION over %d clips ---" % len(vals))
        print("  static leading frames: min %d  median %.1f  mean %.2f  max %d"
              % (vals.min(), np.median(vals), vals.mean(), vals.max()))
        print("  as milliseconds      : min %.0f  median %.0f  mean %.0f  max %.0f"
              % (ms.min(), np.median(ms), ms.mean(), ms.max()))
        in_band = int(((vals >= 3) & (vals <= 6)).sum())
        print("  BUG-07.13 claims 3-6 static frames: %d/%d clips (%.0f%%) fall in "
              "that band" % (in_band, len(vals), 100.0 * in_band / len(vals)))
        zero = int((vals == 0).sum())
        print("  clips with NO leading freeze at all: %d (%.0f%%)"
              % (zero, 100.0 * zero / len(vals)))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true",
                    help="prove the sign convention on synthetic ground truth and exit")
    ap.add_argument("--clip")
    ap.add_argument("--audio")
    ap.add_argument("--audio-start", type=float, default=0.0)
    ap.add_argument("--audio-dur", type=float, default=None)
    ap.add_argument("--episode", help="measure every clip of --engine in this episode")
    ap.add_argument("--engine", default="humo")
    ap.add_argument("--control-shift-ms", type=float, default=None,
                    help="also measure with the audio shifted by this much, to "
                         "confirm the reading moves by the injected amount")
    ap.add_argument("--json-out")
    ap.add_argument("--static-onset", metavar="EPISODES_ROOT",
                    help="M1 step 1: count the leading STATIC frames of every "
                         "clip of --engine under this episodes root. Pure video, "
                         "no audio, no GPU -- tests BUG-07.13's cause field.")
    ap.add_argument("--limit", type=int, default=None,
                    help="with --static-onset, stop after this many episodes")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.static_onset:
        rows = run_static_onset(args.static_onset, args.engine, args.limit)
        if args.json_out:
            pathlib.Path(args.json_out).write_text(
                json.dumps(rows, indent=2, default=str), encoding="utf-8")
            print("\nwrote %s" % args.json_out)
        return 0

    results = []
    if args.episode:
        pairs = episode_clips(args.episode, args.engine)
        if not pairs:
            raise SystemExit("no %s clips found in %s" % (args.engine, args.episode))
        print("measuring %d %s clip(s) from %s"
              % (len(pairs), args.engine, pathlib.Path(args.episode).name))
        for clip, start, dur, wav in pairs:
            res = measure_clip(clip, wav, start, dur)
            report(res)
            results.append(res)
            if args.control_shift_ms:
                ctl = measure_clip(clip, wav, start, dur,
                                   shift_ms=args.control_shift_ms,
                                   label=clip.name + " [SHIFTED CONTROL]")
                report(ctl)
                results.append(ctl)
    elif args.clip and args.audio:
        res = measure_clip(args.clip, args.audio, args.audio_start, args.audio_dur)
        report(res)
        results.append(res)
        if args.control_shift_ms:
            ctl = measure_clip(args.clip, args.audio, args.audio_start, args.audio_dur,
                               shift_ms=args.control_shift_ms,
                               label=pathlib.Path(args.clip).name + " [SHIFTED CONTROL]")
            report(ctl)
            results.append(ctl)
    else:
        ap.error("need --self-test, or --clip with --audio, or --episode")

    if len(results) > 1:
        print("\n=========== SUMMARY ===========")
        for r in results:
            whole = r["whole"] or {}
            print("%-52s %-12s whole %+7.1f ms corr %.2f"
                  % (r["label"][:52], r["verdict"],
                     whole.get("lag_ms", 0.0), whole.get("corr", 0.0)))

    if args.json_out:
        pathlib.Path(args.json_out).write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8")
        print("\nwrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
