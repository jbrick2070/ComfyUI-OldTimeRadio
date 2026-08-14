#!/usr/bin/env python
"""Unattended overnight driver for the 45-word campaign.

The operator went to bed with one campaign in flight and one engine already
root-caused-and-fixed but UNPROVEN, because the fix lands in code the running
server loaded hours ago. This script is what turns "a campaign is running" into
"the box keeps working until every engine is proven or has failed twice".

WHAT IT DOES, in order:

1. WAIT for the in-flight campaign to finish. It never interrupts a live leg --
   a leg is 15 min to 3.5 h of GPU time and killing one wastes all of it.
2. RESTART the server. This is the whole reason the script exists: a production
   leg is submitted to an already-booted server, so every code fix made after
   the boot is invisible to it. ltx_video's contract fix is in exactly that
   state.
3. RE-RUN every engine that is not yet a PASS -- the ones that failed, and any
   the first campaign never reached.
4. REPEAT until either every engine passes or an engine has failed twice.
   Twice is the stopping rule on purpose: a second identical failure means the
   fix in between did not work, and a third unattended attempt would just burn
   the night re-proving the same wrong model of the problem. That is the
   operator's own two-strikes rule applied to a machine that cannot think
   between attempts.

WHAT IT DELIBERATELY DOES NOT DO: diagnose or edit code. It cannot. Every leg
result is written to disk so the diagnosis can happen when someone is awake, and
each round's outcome is a notification that wakes the assistant.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
PY = r"C:/Users/jeffr/Documents/ComfyUI/.venv/Scripts/python.exe"
CAMPAIGN = REPO / "scripts" / "otr_w45_campaign.py"
BOOT = REPO / "scripts" / "_otr_w45_boot.ps1"
LOCK = REPO / "tmp" / "_w45_campaign.lock"
RESULTS = REPO / "tmp" / "_w45_results.json"
LEDGER = REPO / "tmp" / "_w45_overnight.json"

def _all_engines() -> list:
    """The engines the CAMPAIGN will actually run -- asked, never hardcoded.

    This list used to be a hardcoded six: ltx_8gb, ltx_video, ltx_audio_in,
    humo, fastwan_8gb, wan_ti2v. The campaign was corrected on 2026-08-02 to
    cover all NINETEEN registered local engines, and this file was not, so the
    unattended overnight driver went on trying to prove six of nineteen and
    reporting completion -- the same "coverage that hides its denominator"
    defect, one file downstream.

    Deriving it from ``otr_w45_campaign.build_legs()`` means the two can no
    longer disagree. If the campaign refuses to build a roster (a local engine
    with no profile), that refusal surfaces here too rather than being papered
    over with a stale list.
    """
    scripts_dir = str(REPO / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import otr_w45_campaign as _campaign
    return [engine for engine, _profile in _campaign.build_legs()]

MAX_ROUNDS = 4
MAX_FAILURES_PER_ENGINE = 2


def say(msg: str) -> None:
    print("[overnight %s] %s"
          % (_dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)


def wait_for_campaign_to_finish(max_wait_s: int = 6 * 3600) -> None:
    """Block until no campaign owns the lock. Never kills a live leg."""
    if not LOCK.exists():
        say("no campaign in flight")
        return
    say("a campaign holds the lock -- waiting it out (never interrupt a live leg)")
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        time.sleep(60)
        if not LOCK.exists():
            say("in-flight campaign finished")
            return
    say("WARNING: waited %d h and the lock is still held; taking it over"
        % (max_wait_s // 3600))
    LOCK.unlink(missing_ok=True)


def passed_engines() -> set:
    """Engines with a verified PASS -- asset, coverage AND delivered_engine."""
    out = set()
    for path in (RESULTS, LEDGER):
        if not path.exists():
            continue
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict) and row.get("ok") and row.get("engine"):
                out.add(str(row["engine"]))
    return out


def restart_server(attempts: int = 3) -> bool:
    """Boot a fresh server so code fixed since the last boot is actually live.

    RETRIES, because a single failure here is not evidence the box is broken.
    On 2026-08-02 this gave up after one try at 02:39 and parked the whole night
    with two engines unproven -- and the identical boot script, run by hand five
    minutes later, brought the server up first time. The failure was transient:
    the boot ran seconds after the previous server was killed at the end of a
    leg, and the launcher never wrote a byte to its log, so the .cmd died before
    even redirecting rather than the server failing to start.

    A boot costs ~20 s. A false "this needs a human" costs the rest of the
    night. Retry with a widening settle, and only then stop.
    """
    for attempt in range(1, attempts + 1):
        say("restarting the server so post-boot code fixes become live "
            "(attempt %d/%d)" % (attempt, attempts))
        try:
            proc = subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(BOOT)],
                capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            say("server boot TIMED OUT on attempt %d" % attempt)
            proc = None
        if proc is not None and "SERVER UP" in (proc.stdout or ""):
            return True
        for line in (proc.stdout or "").strip().splitlines()[-3:] if proc else []:
            say("  boot: %s" % line)
        if attempt < attempts:
            settle = 30 * attempt
            say("boot attempt %d failed -- settling %ds before retrying "
                "(a just-killed server can still be releasing the port)"
                % (attempt, settle))
            time.sleep(settle)
    say("server did NOT come up after %d attempts -- stopping, this needs a "
        "human" % attempts)
    return False


def run_campaign(engines: list) -> list:
    """One campaign pass over `engines`. Returns the per-leg verdicts."""
    LOCK.unlink(missing_ok=True)
    say("running legs: %s" % ", ".join(engines))
    env = dict(os.environ, PYTHONUTF8="1")
    subprocess.run(
        [PY, str(CAMPAIGN), "--acts", "1", "--only", ",".join(engines)],
        cwd=str(REPO), env=env)
    if not RESULTS.exists():
        return []
    try:
        return json.loads(RESULTS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def main() -> int:
    wait_for_campaign_to_finish()

    history: list = []
    if LEDGER.exists():
        try:
            history = json.loads(LEDGER.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            history = []
    # Fold in whatever the in-flight campaign already proved.
    if RESULTS.exists():
        try:
            history.extend(json.loads(RESULTS.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass

    failures: dict = {}
    for row in history:
        if isinstance(row, dict) and row.get("engine") and not row.get("ok"):
            failures[row["engine"]] = failures.get(row["engine"], 0) + 1

    all_engines = _all_engines()
    say("roster from the campaign: %d local engine(s)" % len(all_engines))

    for round_no in range(1, MAX_ROUNDS + 1):
        done = {r["engine"] for r in history
                if isinstance(r, dict) and r.get("ok") and r.get("engine")}
        done |= passed_engines()
        todo = [e for e in all_engines
                if e not in done
                and failures.get(e, 0) < MAX_FAILURES_PER_ENGINE]

        if not todo:
            say("nothing left to run")
            break

        say("ROUND %d -- proven: %s | remaining: %s"
            % (round_no, sorted(done) or "none", todo))

        if not restart_server():
            break

        results = run_campaign(todo)
        history.extend(results)
        LEDGER.write_text(json.dumps(history, indent=2), encoding="utf-8")

        for row in results:
            if isinstance(row, dict) and row.get("engine") and not row.get("ok"):
                failures[row["engine"]] = failures.get(row["engine"], 0) + 1

        stuck = [e for e, n in failures.items()
                 if n >= MAX_FAILURES_PER_ENGINE]
        if stuck:
            say("TWO STRIKES, stopping these: %s -- the fix between attempts "
                "did not work, so a third unattended swing would burn the night "
                "on the same wrong model of the problem" % ", ".join(sorted(stuck)))

    # ---- the morning report ------------------------------------------------
    done = {r["engine"] for r in history
            if isinstance(r, dict) and r.get("ok") and r.get("engine")}
    print("\n=========== OVERNIGHT SUMMARY ===========")
    for engine in all_engines:
        rows = [r for r in history
                if isinstance(r, dict) and r.get("engine") == engine]
        if not rows:
            print("%-14s NOT RUN" % engine)
            continue
        last = rows[-1]
        state = "PASS" if last.get("ok") else "FAIL"
        detail = last.get("coverage") if last.get("ok") else last.get("why")
        print("%-14s %-4s (%s attempt(s), %s min)  %s"
              % (engine, state, len(rows), last.get("minutes"), detail))
    proven = {e for e in done if e in set(all_engines)}
    print("\n%d/%d local engines proven" % (len(proven), len(all_engines)))
    not_run = [e for e in all_engines if e not in proven]
    if not_run:
        print("NOT PROVEN (%d): %s" % (len(not_run), ", ".join(not_run)))
    LEDGER.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return 0 if len(proven) == len(all_engines) else 1


if __name__ == "__main__":
    raise SystemExit(main())
