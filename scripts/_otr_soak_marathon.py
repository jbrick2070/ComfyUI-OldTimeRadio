"""_otr_soak_marathon.py -- N-hour unattended MULTI-MODEL soak (operator ask
2026-06-09: "the architecture has dropdowns -- truly soak the workflow by
testing all sorts of options: multi-LLM, multi-image, multi-video,
multi-audio").

A PLAYLIST of episodes rotates the real dropdown surface -- video engine
(humo production / forced all-LTX / procgen floor), char-voice engine
(indextts2 / bark / dia), music engine
(stable_audio_3 / musicgen), writer LLM (mistral-nemo / local_gemma4_12b
external OpenAI-compatible server) and word count -- one FRESH headless server per episode (per-leg env
via the launcher's _marathon_extra_env.cmd hook), every episode through the
scripts/_otr_soak_capstone hard gates:

  completion + a PLAYABLE AAC mp4 in the operator's otr/obs + byte-identical
  archival PCM audio + no stray writes + V-3 render-phase VRAM <= 14.5 GB.

The engine histogram is asserted STRICTLY on production-humo legs and logged
(informational) on experiment legs. A failure NEVER stops the marathon: full
traceback to marathon.log + a results.jsonl row, then the next episode.
chatterbox (dep-pin danger) and cloud lanes (credit spend) are excluded from
the unattended rotation by design.

Usage:  python scripts/_otr_soak_marathon.py --hours 5
Results: scripts/_otr_soak_capstone_results/marathon_<stamp>/
UTF-8, no BOM, ASCII-only source. The frozen master audio is read-only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import requests  # noqa: E402

import _otr_soak_capstone as soak  # noqa: E402  (the gate engine)

LAUNCHER = os.path.join(_HERE, "_otr_soak_server_launch.cmd")
EXTRA_ENV = os.path.join(_HERE, "_otr_soak_capstone_results",
                         "_marathon_extra_env.cmd")
COMFY = "http://127.0.0.1:8000"

# Node ids in the canonical workflow (run_combo_matrix.py probe 2026-06-05).
N_WRITER, N_CASTLOCK, N_CHARVOICE, N_MUSIC = 1, 80, 81, 83
GEMMA = "local_gemma4_12b"
MISTRAL = "mistralai/Mistral-Nemo-Instruct-2407"

#: The dropdown-rotation playlist. lane: launcher arg2 ('' = humo production,
#: 'LTX' = Sage-free LTX boot, 'HUMO=0' = floor). env: per-leg server env via
#: the launcher hook. patches: (node, widget, value) on top of the workflow.
#: expect_engine: strict histogram engine or '' (informational).
PLAYLIST = [
    dict(leg="ep01_all_ltx_120w", lane="LTX",
         env={"OTR_FORCE_ENGINE_MAP": "*=ltx_video"},
         words=120, chars=2,
         patches=[(N_CHARVOICE, "engine", "indextts2"),
                  (N_MUSIC, "engine", "stable_audio_3")],
         expect_engine=""),
    dict(leg="ep03_humo_bark_musicgen_60w", lane="",
         env={}, words=60, chars=3,
         patches=[(N_CHARVOICE, "engine", "bark"),
                  (N_MUSIC, "engine", "musicgen"),
                  (N_WRITER, "creative_writing_model", MISTRAL),
                  (N_WRITER, "technical_model", MISTRAL)],
         expect_engine="humo"),
    dict(leg="ep04_humo_gemma_80w", lane="",
         env={}, words=80, chars=2,
         patches=[(N_CHARVOICE, "engine", "indextts2"),
                  (N_MUSIC, "engine", "stable_audio_3"),
                  (N_WRITER, "creative_writing_model", GEMMA),
                  (N_WRITER, "technical_model", GEMMA)],
         expect_engine="humo"),
    # dia swapped out 2026-06-10: its Path-B sidecar venv is NOT installed on
    # this box (named fail-closed documented in marathon_20260610_022442).
    dict(leg="ep05_humo_idx_musicgen_100w", lane="",
         env={}, words=100, chars=3,
         patches=[(N_CHARVOICE, "engine", "indextts2"),
                  (N_MUSIC, "engine", "musicgen"),
                  (N_WRITER, "creative_writing_model", MISTRAL),
                  (N_WRITER, "technical_model", GEMMA)],
         expect_engine="humo"),
    # Cloud writer lanes (operator-authorized 2026-06-09 night: "use an
    # openrouter claude or a comfy credit gpt -- mix it up to be sure they
    # all work"; spend = cents/episode, well under the $20 autonomy line).
    # The slot MODEL resolves to the live catalog first-choice inside
    # run_leg; a missing key/auth fails CLOSED with a named error -- which is
    # itself the lane evidence.
    dict(leg="ep05b_openrouter_claude_60w", lane="",
         env={"OTR_ENABLE_OPENROUTER": "1"},
         words=60, chars=2,
         patches=[(N_CHARVOICE, "engine", "indextts2"),
                  (N_MUSIC, "engine", "stable_audio_3"),
                  (N_WRITER, "creative_writing_model", "openrouter:slot-a"),
                  (N_WRITER, "technical_model", "openrouter:slot-a")],
         expect_engine="humo"),
    dict(leg="ep05c_comfy_credits_40w", lane="",
         env={"OTR_ENABLE_COMFY_CREDITS": "1"},
         words=40, chars=2,
         patches=[(N_CHARVOICE, "engine", "indextts2"),
                  (N_MUSIC, "engine", "stable_audio_3"),
                  (N_WRITER, "creative_writing_model", "comfy:slot-a"),
                  (N_WRITER, "technical_model", "comfy:slot-a")],
         expect_engine="humo"),
    dict(leg="ep06_floor_30w", lane="FLOOR",
         env={}, words=30, chars=2,
         patches=[(N_CHARVOICE, "engine", "indextts2"),
                  (N_MUSIC, "engine", "stable_audio_3")],
         expect_engine="", expect_floor=True),
    dict(leg="ep07_humo_160w_4ch", lane="",
         env={}, words=160, chars=4,
         patches=[(N_CHARVOICE, "engine", "indextts2"),
                  (N_MUSIC, "engine", "stable_audio_3"),
                  (N_WRITER, "creative_writing_model", MISTRAL),
                  (N_WRITER, "technical_model", MISTRAL)],
         expect_engine="humo"),
]


def log_line(fh, msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def kill_server():
    """Kill the :8000 server and WAIT until the port actually closes.

    2026-06-10 marathon catch: without the wait, launch_server's first poll
    can hit the DYING previous server, return 'up' instantly, and the leg
    submits against the STALE env (r06's openrouter slot rejected
    value_not_in_list because the old no-flag schema answered)."""
    for _ in range(10):
        subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-NetTCPConnection -LocalPort 8000 -State Listen "
             "-ErrorAction SilentlyContinue).OwningProcess | Select-Object "
             "-Unique | %{ Stop-Process -Id $_ -Force "
             "-ErrorAction SilentlyContinue }"],
            capture_output=True, check=False)
        time.sleep(4)
        try:
            requests.get(COMFY + "/system_stats", timeout=3)
        except Exception:  # noqa: BLE001 -- port closed = done
            return
    raise RuntimeError("port 8000 still answering after kill attempts")


def write_extra_env(env: dict):
    if not env:
        try:
            os.remove(EXTRA_ENV)
        except FileNotFoundError:
            pass
        return
    lines = ["@echo off"] + ["set %s=%s" % (k, v) for k, v in env.items()]
    with open(EXTRA_ENV, "w", encoding="ascii", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")


def launch_server(log_path: str, lane: str, timeout_s: int = 240) -> bool:
    """Spawn the launcher DIRECTLY (no `start` hop -- `cmd /c start` from a
    detached parent never spawned the child on this box). CREATE_NO_WINDOW
    (0x08000000) keeps it headless; the launcher redirects the server's own
    stdout into ``log_path``."""
    cmd = ["cmd", "/c", LAUNCHER, log_path]
    if lane:
        cmd.append(lane)
    subprocess.Popen(cmd, shell=False, creationflags=0x08000000)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            requests.get(COMFY + "/system_stats", timeout=5)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(5)
    return False


def run_marathon(hours: float, max_eps: int, skip: int = 0) -> int:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(soak.RESULTS_DIR, "marathon_" + stamp)
    os.makedirs(out_dir, exist_ok=True)
    soak.RESULTS_DIR = out_dir
    deadline = time.time() + hours * 3600
    fh = open(os.path.join(out_dir, "marathon.log"), "a", encoding="utf-8")
    jl = open(os.path.join(out_dir, "results.jsonl"), "a", encoding="utf-8")
    log_line(fh, "MARATHON start: %.1fh budget, %d-leg playlist (then repeat),"
             " results -> %s" % (hours, len(PLAYLIST), out_dir))
    results = []
    i = int(skip)                          # resume mid-playlist after a fix
    n_run = 0
    while time.time() < deadline and n_run < max_eps:
        spec = dict(PLAYLIST[i % len(PLAYLIST)])
        i += 1
        n_run += 1
        leg = "r%02d_%s" % (i, spec["leg"])
        row = {"leg": leg, "lane": spec.get("lane", ""),
               "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        server_log = os.path.join(out_dir, "server_%s.log" % leg)
        try:
            kill_server()
            write_extra_env(spec.get("env") or {})
            # The DRIVER process also needs the engine-map env? No -- the MAP
            # acts inside the SERVER. The driver needs only the obs/output
            # pins, which the parent env already carries.
            log_line(fh, "%s: launching server (lane=%r env=%r)..."
                     % (leg, spec.get("lane", ""), spec.get("env") or {}))
            if not launch_server(server_log, spec.get("lane", "")):
                raise RuntimeError("server not up in 240s (see %s)" % server_log)
            soak.SMOKE_WORDS = 0           # words flow via explicit patches
            patches = list(spec.get("patches") or [])
            if spec.get("words"):
                patches = ([(N_WRITER, "target_words", int(spec["words"])),
                            (N_WRITER, "num_characters", int(spec.get("chars", 2))),
                            (N_WRITER, "act_count", "auto")] + patches)
            log_line(fh, "%s: running the episode (%d widget patches)"
                     % (leg, len(patches)))
            rc = soak.run_leg(leg, expect_floor=bool(spec.get("expect_floor")),
                              expect_engine=spec.get("expect_engine", ""),
                              extra_patches=patches)
            row["pass"] = (rc == 0)
            if rc != 0:
                row["error"] = ("run_leg rc=%d -- a hard gate failed (see "
                                "marathon.log)" % rc)
            log_line(fh, "%s: %s" % (leg, "PASS" if rc == 0 else "FAIL"))
        except KeyboardInterrupt:
            row["pass"] = False
            row["error"] = "KeyboardInterrupt"
            results.append(row)
            jl.write(json.dumps(row) + "\n")
            jl.flush()
            log_line(fh, "interrupted -- stopping")
            break
        except Exception as exc:  # noqa: BLE001 -- log + continue, never stop
            row["pass"] = False
            row["error"] = "%s: %s" % (type(exc).__name__, exc)
            log_line(fh, "%s: ERROR %s" % (leg, row["error"]))
            fh.write(traceback.format_exc() + "\n")
            fh.flush()
        results.append(row)
        jl.write(json.dumps(row) + "\n")
        jl.flush()
        log_line(fh, "%s done; %.0f min of budget left"
                 % (leg, max(0.0, deadline - time.time()) / 60))
    write_extra_env({})                    # always clean the env hook
    kill_server()
    n_pass = sum(1 for r in results if r.get("pass"))
    log_line(fh, "MARATHON DONE: %d episodes, %d PASS, %d FAIL"
             % (len(results), n_pass, len(results) - n_pass))
    for r in results:
        if not r.get("pass"):
            log_line(fh, "  FAILED %s: %s" % (r["leg"], r.get("error", "?")))
    fh.close()
    jl.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=5.0)
    ap.add_argument("--max-eps", type=int, default=99)
    ap.add_argument("--skip", type=int, default=0,
                    help="start at playlist index N (resume after a fix)")
    args = ap.parse_args(argv)
    return run_marathon(args.hours, args.max_eps, skip=args.skip)


if __name__ == "__main__":
    raise SystemExit(main())
