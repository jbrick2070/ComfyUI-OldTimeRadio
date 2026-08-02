#!/usr/bin/env python
"""45-word end-to-end campaign across every local video engine.

Replaces tmp/_g4_campaign.sh, which had three defects that cost a whole night of
GPU time on 2026-08-01:

1. ONE SHARED LOG. Every leg appended to tmp/_g4_campaign.txt and each new
   campaign truncated it, so two concurrent runs interleaved into an unreadable
   file and destroyed each other's record. Here every leg owns its own log.

2. NO RESET BETWEEN LEGS. A client-side ``RESULT TIMEOUT`` does NOT cancel the
   server job -- the render keeps running and the next leg queues BEHIND the
   ghost. That is how one stalled leg turned into a pile-up. Here every leg
   clears the queue and waits for VRAM to fall back to baseline before the next
   one starts.

3. NOTHING STOPPED A SECOND CAMPAIGN. Two runs fought over 16 GB and both
   thrashed. A lock file now makes a double launch fail loudly.

Success is the ASSET, never the exit code: each leg is verified by ffprobe on
the published file and by audio-vs-video coverage, per the standing rule that a
render is only done when the file is on disk.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]
PY = r"C:/Users/jeffr/Documents/ComfyUI/.venv/Scripts/python.exe"
SERVER = "http://127.0.0.1:8000"
OBS = pathlib.Path(r"C:/Users/jeffr/Documents/ComfyUI/output/otr/obs")
EPISODES = pathlib.Path(r"C:/Users/jeffr/Documents/ComfyUI/output/otr/episodes")
LOCK = REPO / "tmp" / "_w45_campaign.lock"

# Engine -> profile, ordered CHEAPEST-AND-LEAST-PROVEN FIRST.
#
# The first ordering ran wan_ti2v first, which is backwards: it is the slowest
# engine on the box (~12 min/clip, 13-15 clips per leg) AND it already has a
# shipped episode, so it spent the first three hours re-proving something known
# while five untested engines waited. Information per hour is what matters in an
# unattended run -- an engine that is going to fail should fail in the first
# hour, not the twentieth. fastwan_8gb sits second-to-last as a re-confirmation
# of the engine that already shipped, and the slow incumbent goes last.
LEGS = [
    ("ltx_8gb", "otr_g4_ltx_8gb"),
    ("ltx_video", "otr_g4_ltx_video"),
    ("ltx_audio_in", "otr_g4_ltx_audio_in"),
    ("humo", "otr_g4_humo"),
    ("fastwan_8gb", "otr_g4_fastwan"),
    ("wan_ti2v", "otr_g4_wan_ti2v"),
]

# MEASURED on this box, 2026-08-01, rather than estimated from the fastwan leg:
# a 45-word episode is ~66.7 s of audio = 1666 frames at 25 fps, and wan_ti2v
# renders at most 177 frames (7.08 s) per segment, so a leg is 13-15 clips. Two
# consecutive clips landed 12 minutes apart (22:25:32 -> 22:37:29), on top of
# ~45 min of writer + audio. That is a ~3.5-4 h leg for the incumbent.
#
# 10800 s (3 h) was still too small and would have killed the first leg about
# four clips from the end, after burning the whole three hours -- the expensive
# failure, because a client timeout does not cancel the server job. Sized to the
# measurement with headroom instead of to a round number.
LEG_TIMEOUT_S = 21600
VRAM_BASELINE_MB = 2600
VRAM_SETTLE_S = 240


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _say(msg: str) -> None:
    print("[w45 %s] %s" % (_now(), msg), flush=True)


def _server_json(path: str):
    with urllib.request.urlopen(SERVER + path, timeout=15) as fh:
        return json.load(fh)


def _server_post(path: str, payload: dict) -> None:
    req = urllib.request.Request(
        SERVER + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=15).read()


def _vram_used_mb() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()
    return int(out[0]) if out and out[0].strip().isdigit() else -1


def reset_between_legs() -> None:
    """Clear the queue and wait for the GPU to actually let go.

    Without this a stranded job from a timed-out leg keeps rendering and the
    next leg silently queues behind it.
    """
    try:
        _server_post("/queue", {"clear": True})
        _server_post("/interrupt", {})
    except Exception as exc:                       # server may be mid-restart
        _say("queue clear failed (continuing): %s" % exc)

    deadline = time.time() + VRAM_SETTLE_S
    while time.time() < deadline:
        time.sleep(10)
        try:
            queue = _server_json("/queue")
        except Exception:
            continue
        busy = len(queue.get("queue_running", [])) + len(queue.get("queue_pending", []))
        used = _vram_used_mb()
        if busy == 0 and 0 <= used <= VRAM_BASELINE_MB:
            _say("reset clean: queue empty, vram %d MB" % used)
            return
    _say("reset TIMED OUT: queue busy or vram still %d MB -- continuing anyway"
         % _vram_used_mb())


def _ffprobe(path: pathlib.Path, entries: str, stream: str | None = None) -> str:
    cmd = ["ffprobe", "-v", "error"]
    if stream:
        cmd += ["-select_streams", stream]
    cmd += ["-show_entries", entries, "-of", "csv=p=0", str(path)]
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def _duration(path: pathlib.Path) -> float:
    raw = _ffprobe(path, "format=duration")
    try:
        return float(raw.split(",")[0])
    except ValueError:
        return 0.0


def delivered_engines(episode: pathlib.Path) -> list[str]:
    """Which engine ACTUALLY rendered each clip, per the ledger.

    This is the check that separates "six engines work" from "six legs produced
    a file". A dark engine does not necessarily fail the run -- OTR can fall
    closed and hand the beat to still_parallax, which publishes a perfectly
    valid episode that never touched the engine under test (the 2026-06-12
    mesh_stage catch: "PASS but engine NOT in the trace"). The ledger records
    the truth per clip in ``delivered_engine``.
    """
    ledgers = list(episode.glob("*_ledger.json")) + list(episode.glob("audio/*_ledger.json"))
    if not ledgers:
        return []
    try:
        blob = json.loads(ledgers[0].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []

    seen: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            got = node.get("delivered_engine")
            if isinstance(got, str) and got:
                seen.append(got)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(blob)
    return seen


def verify_asset(started_at: float) -> dict:
    """A leg passes only if it PUBLISHED, and only if video covers audio."""
    published = [p for p in OBS.glob("*.mp4") if p.stat().st_mtime >= started_at]
    if not published:
        return {"ok": False, "why": "no new file in otr/obs"}
    asset = max(published, key=lambda p: p.stat().st_mtime)

    dims = _ffprobe(asset, "stream=width,height,nb_frames", stream="v:0")
    seconds = _duration(asset)
    audio = _ffprobe(asset, "stream=codec_name", stream="a:0")
    if seconds <= 0:
        return {"ok": False, "why": "unreadable/zero-length asset", "asset": asset.name}
    if not audio:
        return {"ok": False, "why": "no audio stream", "asset": asset.name}

    # Coverage: every second of audio must have real video behind it.
    coverage = "not measured"
    engines: list[str] = []
    episode_dirs = [d for d in EPISODES.iterdir()
                    if d.is_dir() and d.stat().st_mtime >= started_at]
    if episode_dirs:
        ep = max(episode_dirs, key=lambda d: d.stat().st_mtime)
        engines = delivered_engines(ep)
        masters = list(ep.glob("audio/*master.wav"))
        clips = sorted(ep.glob("clips/*.mp4"))
        if masters and clips:
            a = _duration(masters[0])
            v = sum(_duration(c) for c in clips)
            coverage = "audio %.2fs video %.2fs clips %d %s" % (
                a, v, len(clips), "COVERS" if v >= a else "SHORT")
            if v < a:
                return {"ok": False, "why": "video short of audio",
                        "asset": asset.name, "coverage": coverage}

    return {"ok": True, "asset": asset.name, "dims": dims,
            "seconds": round(seconds, 2), "audio": audio, "coverage": coverage,
            "delivered": sorted(set(engines))}


def run_leg(engine: str, profile: str, words: int) -> dict:
    log_path = REPO / "tmp" / ("_w45_%s.log" % engine)
    _say("LEG %s (profile %s) -> %s" % (engine, profile, log_path.name))
    started_at = time.time()

    cmd = [PY, str(REPO / "scripts" / "otr_canonical_api_run.py"),
           "--profile", profile,
           "--words", str(words),
           "--source-bank", "roll (any eligible bank)",
           "--visual-style", "roll (any style)",
           "--timeout", str(LEG_TIMEOUT_S)]
    env = dict(os.environ, PYTHONUTF8="1")

    with open(log_path, "w", encoding="utf-8") as log:
        log.write("### %s  profile=%s  start=%s\n" % (engine, profile, _now()))
        log.flush()
        try:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  cwd=str(REPO), env=env,
                                  timeout=LEG_TIMEOUT_S + 600).returncode
        except subprocess.TimeoutExpired:
            code = -1
            log.write("\n### runner timeout after %ds\n" % (LEG_TIMEOUT_S + 600))

    elapsed = time.time() - started_at
    verdict = verify_asset(started_at)

    # The engine under test must be the engine that actually rendered. A
    # published file proves the pipeline ran; only the ledger proves WHICH
    # engine ran, and a silent fall-back to still_parallax would otherwise be
    # scored as a pass for an engine that never loaded.
    if verdict.get("ok"):
        delivered = verdict.get("delivered") or []
        if not delivered:
            verdict.update(ok=False, why="ledger records no delivered_engine")
        elif delivered != [engine]:
            verdict.update(
                ok=False,
                why="wrong engine delivered: expected %s, ledger says %s"
                    % (engine, ", ".join(delivered)))

    verdict.update(engine=engine, profile=profile, exit=code,
                   minutes=round(elapsed / 60.0, 1), log=log_path.name)

    _say("LEG %s -> %s (exit=%s, %.1f min) %s" % (
        engine, "PASS" if verdict["ok"] else "FAIL",
        code, elapsed / 60.0, verdict.get("why", verdict.get("asset", ""))))
    return verdict


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--words", type=int, default=45)
    ap.add_argument("--only", default=None,
                    help="comma-separated engine names to run instead of all")
    args = ap.parse_args(argv)

    if LOCK.exists():
        print("REFUSING: %s exists -- a campaign is already running.\n"
              "Two campaigns on one 16 GB GPU thrash and corrupt each other's\n"
              "results. Stop the other run, then delete the lock." % LOCK,
              file=sys.stderr)
        return 2
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text("pid=%d start=%s\n" % (os.getpid(), _now()), encoding="utf-8")

    legs = LEGS
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        legs = [leg for leg in LEGS if leg[0] in wanted]

    results = []
    try:
        for engine, profile in legs:
            reset_between_legs()
            results.append(run_leg(engine, profile, args.words))
            (REPO / "tmp" / "_w45_results.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8")
    finally:
        LOCK.unlink(missing_ok=True)

    print("\n=========== 45-WORD CAMPAIGN ===========")
    for r in results:
        print("%-14s %-5s exit=%-4s %5s min  %s" % (
            r["engine"], "PASS" if r["ok"] else "FAIL", r["exit"],
            r["minutes"], r.get("why", r.get("coverage", ""))))
    passed = sum(1 for r in results if r["ok"])
    print("%d/%d engines passed" % (passed, len(results)))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
