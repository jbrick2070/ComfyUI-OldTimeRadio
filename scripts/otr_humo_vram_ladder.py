#!/usr/bin/env python
"""M2 -- the paired HuMo VRAM ladder, both orientations, cold and warm.

THE QUESTION. `eng_humo.py:106` caps the 14B tiers at 97 frames and its own
comment says that number is "a QUALITY-supported target, not a measured memory
ceiling". No HuMo VRAM peak has ever been measured on this box: the lane was the
only heavy one without a `VramPeakProbe` until it was added 2026-08-02, and
every HuMo `per_clip` ledger row before that date is null. So the cap is a
reasoned bound with no data under it.

THE PREDICTION TO FALSIFY. The two orientations match at every rung. `humo`
renders 480x832 and `humo_14B_169` renders 832x480 -- the SAME checkpoint, equal
pixels, equal token grid. If they differ, the cause is tiling or kernel
selection, not the model.

WHY TWO NUMBERS PER CELL, NOT ONE. `VramPeakProbe` reports MACHINE-WIDE used
VRAM via NVML, so it includes the desktop and anything else on the box. That is
the right quantity for "did we come near the card's limit" and the WRONG one for
"how much does this render cost". Recording only the machine-wide peak is how
`docs/2026-08-02-MEASUREMENT-ltx-av-vram-vs-frames.md` got its "phantom 15.9 GB".
Every cell here records the pre-render used VRAM and derives:

    incremental_mb = peak_used_mb - pre_used_mb

COLD MEANS COLD. A cold cell is measured after a SERVER RESTART, so the weight
load is inside the probe window. Without the restart only the very first cell of
a session is cold and every later "first render" is silently warm. The warm cell
is the immediately following repeat against the same resident server.

Usage:
    python scripts/otr_humo_vram_ladder.py --portrait P.png --audio A.wav
    python scripts/otr_humo_vram_ladder.py --rungs 49,97 --engines humo --dry-run
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import re
import subprocess
import time
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]
PY = r"C:/Users/jeffr/Documents/ComfyUI/.venv/Scripts/python.exe"
LAUNCH = REPO / "scripts" / "_otr_soak_server_launch.cmd"
SERVER_LOG = REPO / "tmp" / "_m2_humo_server.log"
SERVER = "http://127.0.0.1:8000"

#: The 14.5 GB real-world target from CLAUDE.md, in MB.
CEILING_MB = 14848

PEAK_RE = re.compile(
    r"\[(?P<engine>[\w.]+)\] render-window VRAM peak:\s*(?P<mb>\d+)\s*MB"
    r"\s*\(length=(?P<length>\d+)\s+canvas=(?P<canvas>\d+x\d+)\)")


def _now():
    return _dt.datetime.now().strftime("%H:%M:%S")


def say(msg):
    print("[m2 %s] %s" % (_now(), msg), flush=True)


def vram_used_mb():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip().splitlines()
    return int(out[0]) if out and out[0].strip().isdigit() else -1


def server_listening():
    try:
        urllib.request.urlopen(SERVER + "/queue", timeout=5).read()
        return True
    except Exception:
        return False


def kill_servers():
    """Selective kill by CommandLine -- never a blanket python kill.

    A blanket kill also takes out the MCP extension pythons and severs the
    tools driving this run (CLAUDE.md section 4).
    """
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -match 'ComfyUI\\\\ComfyUI\\\\main\\.py' } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                   capture_output=True, text=True)
    for _ in range(30):
        if not server_listening():
            return True
        time.sleep(2)
    return False


def boot_server(timeout_s=120):
    ps = (
        "Start-Process -FilePath '%s' -ArgumentList @('%s','HUMO') "
        "-WindowStyle Hidden" % (LAUNCH, SERVER_LOG)
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                   capture_output=True, text=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(3)
        if server_listening():
            return True
    return False


def log_size():
    try:
        return SERVER_LOG.stat().st_size
    except OSError:
        return 0


def peaks_since(offset):
    """Every VramPeakProbe line written after byte ``offset``.

    Opened in BINARY and decoded afterwards. Text mode was the first version and
    it silently lost the warm cell: ``stat().st_size`` is a BYTE count, but a
    text-mode handle on Windows translates CRLF to LF, so seeking to a byte
    offset lands somewhere else in the stream and the probe line after it was
    never seen. The render had run -- 164 seconds of it -- and the harness
    reported the cell as failed with no peak, which is a far more expensive lie
    than a crash.
    """
    try:
        with open(SERVER_LOG, "rb") as fh:
            fh.seek(offset)
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return [m.groupdict() for m in PEAK_RE.finditer(tail)]


def run_cell(engine, frames, portrait, audio, timeout_s, buster=0):
    """One render. ``buster`` must be UNIQUE per submitted cell.

    Without it the warm repeat submits a byte-identical graph, ComfyUI returns
    the CACHED history for that prompt in about five seconds -- carrying the
    cold run's elapsed_s with it -- and no new probe line is written. The result
    looks like a fast second render and is actually a copy of the first. It
    lands in ``oom_index``, which the node documents as inert in mode=single.
    """
    pre = vram_used_mb()
    mark = log_size()
    started = time.time()

    cmd = [PY, str(REPO / "scripts" / "_otr_single_engine_smoke.py"),
           "--engine", engine, "--frames", str(frames),
           "--portrait", str(portrait), "--audio", str(audio),
           "--cache-buster", str(buster),
           "--timeout", str(timeout_s)]
    env = dict(os.environ, PYTHONUTF8="1")
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          cwd=str(REPO), env=env, timeout=timeout_s + 300)
    elapsed = time.time() - started

    found = peaks_since(mark)
    mine = [p for p in found if int(p["length"]) == int(frames)]
    peak = int(mine[-1]["mb"]) if mine else (int(found[-1]["mb"]) if found else None)
    canvas = (mine[-1]["canvas"] if mine else (found[-1]["canvas"] if found else None))

    ok = proc.returncode == 0 and peak is not None
    row = {
        "engine": engine, "frames": frames, "canvas": canvas,
        "pre_used_mb": pre, "peak_used_mb": peak,
        "incremental_mb": (peak - pre) if (peak is not None and pre >= 0) else None,
        "elapsed_s": round(elapsed, 1), "ok": ok,
        "exit": proc.returncode,
    }
    if not ok:
        row["stderr_tail"] = (proc.stdout or "")[-400:]
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rungs", default="49,65,81,97")
    ap.add_argument("--engines", default="humo,humo_14B_169",
                    help="paired orientations: humo is 480x832, humo_14B_169 is "
                         "832x480, same 14B checkpoint")
    ap.add_argument("--portrait", required=False, default="")
    ap.add_argument("--audio", required=False, default="")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default=str(REPO / "tmp" / "_m2_humo_ladder.json"))
    ap.add_argument("--no-restart", action="store_true",
                    help="skip the per-cell server restart (cold becomes warm; "
                         "only for a quick re-check, never for the record)")
    ap.add_argument("--continue-past-breach", action="store_true",
                    help="keep climbing after a rung breaches the ceiling. The "
                         "plan says stop at the first breach, which assumes the "
                         "breach arrives partway UP the ladder. When the LOWEST "
                         "rung already breaches, stopping there measures almost "
                         "nothing -- and the open question is whether peak tracks "
                         "frame count at all, which three other engines say it "
                         "does not. A render that fails still fails the cell; "
                         "this only declines to stop the sweep.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    rungs = [int(r) for r in args.rungs.split(",") if r.strip()]
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    cells = [(e, r) for r in rungs for e in engines]

    say("ladder: %d cells (%s x %s), cold+warm each"
        % (len(cells), ",".join(engines), ",".join(str(r) for r in rungs)))
    if args.dry_run:
        for e, r in cells:
            print("  would run %-14s %3d frames (cold, then warm)" % (e, r))
        return 0

    if not args.portrait or not args.audio:
        ap.error("--portrait and --audio are required for a real run")

    rows = []
    stop_after_rung = None
    cell_seq = 0
    for engine, frames in cells:
        # STOP AT A RUNG BOUNDARY, NEVER MID-PAIR. The plan says to stop at the
        # first rung breaching the ceiling cold, but the whole point of this
        # ladder is the ORIENTATION COMPARISON, and halting the instant the
        # first orientation breaches leaves its partner unmeasured -- which
        # destroys the only comparison the milestone exists to make. So the
        # breach is remembered and enforced once the rung's pair is complete.
        if stop_after_rung is not None and frames != stop_after_rung:
            say("STOPPING after the %d-frame pair: it breached the %d MB "
                "ceiling cold" % (stop_after_rung, CEILING_MB))
            break

        if not args.no_restart:
            say("restarting server for a TRUE COLD %s @ %d" % (engine, frames))
            kill_servers()
            if not boot_server():
                say("SERVER DID NOT COME UP -- aborting")
                break

        for phase in ("cold", "warm"):
            say("%s %-14s %3d frames ..." % (phase.upper(), engine, frames))
            cell_seq += 1
            row = run_cell(engine, frames, args.portrait, args.audio,
                           args.timeout, buster=cell_seq)
            row["phase"] = phase
            row["cache_buster"] = cell_seq
            # A "render" that returned faster than any real one did not run.
            if row["ok"] and row["elapsed_s"] < 20:
                row.update(ok=False,
                           why="returned in %.1fs -- ComfyUI served a CACHED "
                               "prompt, this cell did not render"
                               % row["elapsed_s"])
            rows.append(row)
            pathlib.Path(args.out).write_text(json.dumps(rows, indent=2),
                                              encoding="utf-8")
            if not row["ok"]:
                say("  FAILED (exit=%s) -- %s"
                    % (row["exit"], row.get("stderr_tail", "")[:200]))
                continue
            say("  peak %s MB | pre %s MB | incremental %s MB | %s | %.0fs"
                % (row["peak_used_mb"], row["pre_used_mb"],
                   row["incremental_mb"], row["canvas"], row["elapsed_s"]))
            if phase == "cold" and (row["peak_used_mb"] or 0) > CEILING_MB:
                say("  BREACH: cold peak %s MB is over the %d MB ceiling%s"
                    % (row["peak_used_mb"], CEILING_MB,
                       "" if args.continue_past_breach
                       else " (finishing this rung's pair before stopping)"))
                if not args.continue_past_breach:
                    stop_after_rung = frames

    print("\n=========== M2 HUMO VRAM LADDER ===========")
    print("%-14s %6s %5s %-10s %9s %9s %12s" % (
        "ENGINE", "FRAMES", "PHASE", "CANVAS", "PRE_MB", "PEAK_MB", "INCREMENTAL"))
    for r in rows:
        print("%-14s %6d %5s %-10s %9s %9s %12s%s" % (
            r["engine"], r["frames"], r["phase"], r["canvas"] or "?",
            r["pre_used_mb"], r["peak_used_mb"], r["incremental_mb"],
            "  <-- OVER CEILING" if (r["peak_used_mb"] or 0) > CEILING_MB else ""))

    # The paired comparison this milestone exists to make.
    print("\n--- ORIENTATION PAIRING (the prediction to falsify) ---")
    for frames in rungs:
        pair = {r["engine"]: r for r in rows
                if r["frames"] == frames and r["phase"] == "cold" and r["ok"]}
        if len(pair) == 2:
            a, b = engines[0], engines[1]
            da = pair[a]["peak_used_mb"]
            db = pair[b]["peak_used_mb"]
            print("  %3d frames: %s %s MB vs %s %s MB -> delta %+d MB %s"
                  % (frames, a, da, b, db, db - da,
                     "(MATCH)" if abs(db - da) < 300 else "(DIVERGENT)"))
        else:
            print("  %3d frames: incomplete pair, no comparison" % frames)

    print("\nwrote %s" % args.out)
    print("MEASUREMENT ONLY -- this does not qualify any tier. Any cap change "
          "must be re-proved through workflows/otr_canonical.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
