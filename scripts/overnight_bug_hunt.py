"""scripts/overnight_bug_hunt.py

Sprint H bug-hunt: TWO-PROCESS supervisor. Long-lived loop driver
that NEVER touches the GPU directly. Spawns scripts/worker_iter.py
per iter via subprocess.Popen, waits with a 20-min outer timeout,
reads the worker's result file, appends one row to the master
JSONL, then checks stop conditions and either advances or halts.

Rewritten in place at 2026-05-17 per synthesis section 3.6. The
old single-process loop (pre-rewrite) launched ComfyUI itself,
queue_smoke'd, polled history, and called kill_all_python.bat
between iters. That design had three failure modes the rewrite
closes:

  1. Pre-rewrite supervisor + ComfyUI shared one process tree; a
     ComfyUI crash could take the loop down. Post-rewrite,
     ComfyUI is a grandchild of the supervisor (worker -> ComfyUI),
     so a tree-kill of the worker drops ComfyUI without touching
     the supervisor.
  2. kill_all_python.bat from the loop killed THIS supervisor too
     (Jeffrey's "kill it all" rule was fine in attended mode, fatal
     for an autonomous loop). Post-rewrite, the supervisor invokes
     scripts/sweep_python_excluding.bat <SUPERVISOR_PID> between
     iters; its own PID is preserved.
  3. ComfyUI launched via PowerShell Start-Process -> cmd /c chain
     was unreachable by taskkill /F /PID /T. Post-rewrite, the
     worker spawns ComfyUI as its DIRECT subprocess.Popen child
     (no .bat, no PowerShell wrapper), so the kill tree is clean.

Output:
  logs/overnight_results.jsonl   -- one JSON object per iter,
                                    supervisor-written only.
                                    Worker never writes here.
  logs/overnight_supervisor.log  -- this script's own progress
                                    trace (per-iter spawn/wait
                                    events, stop-condition decisions).
  logs/worker_iter_<N>.json      -- worker-written, supervisor-read.
  logs/comfy_session_iter_<N>.log -- worker-tee'd, ComfyUI console.
  logs/sweep_betweeniter.log     -- between-iter sweep events.
  logs/teardown.log              -- ffmpeg_global_kill audit lines.

Invocation:
  scripts/sweep_and_launch.bat --iters 12 --inter-iter-sec 10

  -- the bat does a blanket pre-launch sweep, then spawns this
  supervisor. Direct python invocation also works but you lose
  the prelaunch-sweep audit log line.

Stop conditions (synthesis section 7):
  - 3 consecutive cleans -> SUCCESS (return 0)
  - 2 consecutive same-class fails -> halt for manual triage
  - 2 consecutive timeouts -> halt (likely systemic hang)
  - 2 consecutive worker_crash -> halt (supervisor can't trust the
        harness; same-class branch already covers this but spelled
        out for clarity)
  - C7 audio bytes diverge from tests/test_audio_byte_identical.py
        baseline -> halt, rollback. NOT autonomously checked here
        (test doesn't exist at this revision; section 7's C7 gate
        depends on a test that's in tests/test_workflow_canonical_baseline.py
        instead). Operator runs the canonical-baseline pytest
        manually after every loop close.

Retry policy (synthesis section 1.10): one OOM retry max. If an
iter classifies as llm_oom OR video_oom, the NEXT iter is the
retry. Two OOMs back-to-back trigger the same-class halt.

KeyboardInterrupt: handler taskkills the current worker tree
before re-raising so the operator can Ctrl+C cleanly during
attended validation (synthesis section 3.7).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

REPO = Path(r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio")
COMFY_DIR = Path(r"C:\Users\jeffr\Documents\ComfyUI")
LOGS_DIR = COMFY_DIR / "logs"
RESULTS_JSONL = LOGS_DIR / "overnight_results.jsonl"
SUPERVISOR_LOG = LOGS_DIR / "overnight_supervisor.log"

WORKER_PY = REPO / "scripts" / "worker_iter.py"
SWEEP_EXCLUDE_BAT = REPO / "scripts" / "sweep_python_excluding.bat"
PY = Path(r"C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe")

# Synthesis section 2 step 5: 20-min outer wait on each worker.
WORKER_WAIT_S = 1200
# Port probe starts here.
PORT_BASE = 8000
PORT_PROBE_MAX = 32

# Failure classes (synthesis section 2 fixed dictionary).
FAILURE_CLASSES = {
    "graph_widget", "missing_model", "llm_oom", "video_oom",
    "ffmpeg_composite", "comfyui_startup", "port_occupied",
    "orphan_detected", "vram_contaminated", "worker_crash",
    "crash_process", "timeout", "unknown",
}


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _log(msg: str) -> None:
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    SUPERVISOR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with SUPERVISOR_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _pick_free_port(start: int = PORT_BASE, attempts: int = PORT_PROBE_MAX) -> int:
    """Probe loopback starting at `start`; first unbound port wins."""
    for offset in range(attempts):
        port = start + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(
        f"could not find free port in [{start}, {start + attempts})"
    )


def _wmi_cmdline_for(pid: int) -> str | None:
    """Look up a PID's CommandLine via WMI. Returns None on miss."""
    try:
        proc = subprocess.run(
            [
                "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-Command",
                (
                    f"$p = Get-CimInstance Win32_Process -Filter "
                    f"'ProcessId={int(pid)}'; "
                    "if ($p -and $p.CommandLine) { Write-Output $p.CommandLine } "
                    "else { Write-Output '' }"
                ),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        out = (proc.stdout or "").strip()
        return out or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _run_between_iter_sweep(keep_pids: list[int]) -> None:
    """Synthesis section 2 supervisor step 2 + Jeffrey 2026-05-17 round-
    robin §B. The keep-list is a variadic list of PIDs to exclude from
    the sweep -- under the uv launcher-stub model both the stub PID
    (os.getppid() in most cases) and the real cpython PID (os.getpid())
    are running python.exe, and BOTH have to survive.
    """
    if not SWEEP_EXCLUDE_BAT.is_file():
        _log(f"WARN sweep bat missing: {SWEEP_EXCLUDE_BAT}")
        return
    if not keep_pids:
        _log("WARN sweep called with empty keep_pids list; skipping")
        return
    args = ["cmd", "/c", str(SWEEP_EXCLUDE_BAT)] + [str(int(p)) for p in keep_pids]
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        out = (proc.stdout or "").strip()
        if out:
            _log("sweep_python_excluding output: " + out.splitlines()[-1][:200])
    except subprocess.TimeoutExpired:
        _log("WARN sweep_python_excluding TIMED OUT (60s)")
    except FileNotFoundError as exc:
        _log(f"WARN sweep_python_excluding FileNotFoundError: {exc}")


def _kill_tree(pid: int) -> None:
    try:
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid), "/T"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _spawn_worker(iter_idx: int, port: int) -> subprocess.Popen:
    """Spawn worker_iter.py as a direct child of the supervisor."""
    args = [
        str(PY),
        str(WORKER_PY),
        str(iter_idx),
        "--port", str(port),
    ]
    # No shell, no detached flag -- we want a direct child so
    # taskkill /T from the supervisor can drop its tree if the
    # worker hangs past WORKER_WAIT_S.
    return subprocess.Popen(
        args,
        cwd=str(REPO),
        shell=False,
    )


def _read_worker_result(iter_idx: int) -> dict[str, Any] | None:
    path = LOGS_DIR / f"worker_iter_{iter_idx:03d}.json"
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _append_master(rec: dict[str, Any]) -> None:
    RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _stop_decision(history: deque[dict[str, Any]]) -> str | None:
    """Return a stop reason or None. Synthesis section 7."""
    if len(history) >= 3:
        last3 = list(history)[-3:]
        if all(h.get("failure_class") == "success" for h in last3):
            return "SUCCESS: 3 consecutive cleans"
    if len(history) >= 2:
        last2 = list(history)[-2:]
        c0 = last2[0].get("failure_class")
        c1 = last2[1].get("failure_class")
        if c0 and c0 == c1 and c0 != "success":
            return f"halt: 2 consecutive {c0} failures"
        if c0 == "timeout" and c1 == "timeout":
            return "halt: 2 consecutive timeouts"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sprint H two-process bug-hunt supervisor."
    )
    parser.add_argument("--iters", type=int, default=12,
                        help="Max iters to run (default 12).")
    parser.add_argument("--inter-iter-sec", type=int, default=10,
                        help="Sleep between iters AFTER sweep (default 10s).")
    parser.add_argument("--worker-wait-s", type=int, default=WORKER_WAIT_S,
                        help="Outer wait per worker (default 1200s = 20min).")
    parser.add_argument("--until-iso", default=None,
                        help="Stop time as ISO8601 local. Overrides --iters "
                             "if reached first.")
    parser.add_argument("--no-stop-conditions", action="store_true",
                        help="Disable section-7 early-halt. Use only for "
                             "diagnostic full-N walks.")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # B1 fix (Jeffrey 2026-05-17 round-robin): capture BOTH the real-
    # cpython PID (os.getpid()) AND the launcher-stub PID
    # (os.getppid() under the uv stub model). Both are running
    # `python.exe` to WMI; both must survive the between-iter sweep
    # or the supervisor gets killed by its own cleanup.
    #
    # Sanity canary: if the parent's WMI cmdline doesn't reference
    # this script, the uv-stub assumption may have changed (e.g.
    # uv updated to a non-stub launch model, or the bat now wraps
    # in cmd /c). Log a WARN and proceed -- the keep-list pass is
    # still correct as a safety net, the warning surfaces a model
    # drift for the next round-robin to consider.
    real_pid = os.getpid()
    parent_pid = os.getppid()
    parent_cmd = _wmi_cmdline_for(parent_pid)
    if parent_cmd and "overnight_bug_hunt.py" not in parent_cmd:
        _log(
            f"WARN parent_pid={parent_pid} cmdline does not match "
            f"supervisor; got {parent_cmd!r}. uv-stub assumption may "
            f"have changed; passing both PIDs to sweep keep-list as "
            f"safety net"
        )
    keep_pids: list[int] = [int(real_pid), int(parent_pid)]
    # Dedup (in case the parent IS this python).
    keep_pids = sorted({p for p in keep_pids if p > 0})

    _log("=== overnight bug-hunt SUPERVISOR START ===")
    _log(f"real_pid={real_pid} parent_pid={parent_pid} "
         f"keep_pids={keep_pids} iters={args.iters} "
         f"inter_iter_sec={args.inter_iter_sec} "
         f"worker_wait_s={args.worker_wait_s} "
         f"until_iso={args.until_iso}")

    until_dt = None
    if args.until_iso:
        try:
            until_dt = dt.datetime.fromisoformat(args.until_iso)
        except ValueError:
            _log(f"WARN bad --until-iso: {args.until_iso!r} ignored")

    # Rolling window of the last few iter results for stop-condition
    # decisions. Keeping the last 4 is enough to cover both 3-clean
    # SUCCESS and 2-same-class halt rules.
    history: deque[dict[str, Any]] = deque(maxlen=4)
    current_worker: subprocess.Popen | None = None

    def _on_sigint(signum, frame):  # noqa: ARG001
        # Tree-kill any in-flight worker so the operator's Ctrl+C
        # doesn't leave a half-running ComfyUI behind.
        _log("KeyboardInterrupt: tearing down current worker tree")
        nonlocal_proc = current_worker
        if nonlocal_proc is not None and nonlocal_proc.poll() is None:
            _kill_tree(int(nonlocal_proc.pid))
        # Re-raise as KeyboardInterrupt to exit the loop cleanly.
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except (ValueError, OSError):
        # Some embedded contexts disallow signal binding; the loop
        # still works without the handler, just less polite on Ctrl+C.
        pass

    try:
        for i in range(1, args.iters + 1):
            if until_dt and dt.datetime.now() >= until_dt:
                _log(f"reached --until-iso {args.until_iso}; "
                     f"stopping at iter {i - 1}")
                break

            _log(f"--- ITER {i} starting ---")

            # Step 1: between-iter sweep (skipped on iter 1 because
            # sweep_and_launch.bat already ran a pre-launch sweep).
            if i > 1:
                _run_between_iter_sweep(keep_pids)

            # Step 2: pick a free port.
            try:
                port = _pick_free_port()
            except RuntimeError as exc:
                _log(f"ITER {i}: port pick failed ({exc}); halting")
                rec = {
                    "iter": i,
                    "port": None,
                    "status": "halt",
                    "failure_class": "port_occupied",
                    "exception_message": str(exc),
                    "start_ts": _now(),
                    "end_ts": _now(),
                }
                _append_master(rec)
                history.append(rec)
                return 1
            _log(f"ITER {i}: port={port}")

            # Step 3: spawn worker.
            t0 = time.time()
            try:
                current_worker = _spawn_worker(i, port)
            except FileNotFoundError as exc:
                _log(f"ITER {i}: worker spawn FileNotFoundError: {exc}")
                return 2
            _log(f"ITER {i}: worker PID={current_worker.pid}")

            # Step 4: wait with timeout.
            try:
                rc = current_worker.wait(timeout=args.worker_wait_s)
                _log(f"ITER {i}: worker exited rc={rc} "
                     f"wall={time.time() - t0:.1f}s")
            except subprocess.TimeoutExpired:
                _log(f"ITER {i}: worker timed out after "
                     f"{args.worker_wait_s}s; kill_tree")
                _kill_tree(int(current_worker.pid))
                time.sleep(3)

            # Step 5: read worker result; missing -> worker_crash.
            res = _read_worker_result(i)
            if res is None:
                _log(f"ITER {i}: no worker result file; "
                     f"classifying as worker_crash")
                res = {
                    "iter": i,
                    "port": port,
                    "comfyui_pid": None,
                    "prompt_id": None,
                    "status": "worker_crash",
                    "failure_class": "worker_crash",
                    "exception_message": (
                        f"no logs/worker_iter_{i:03d}.json found "
                        f"after worker exit / kill"
                    ),
                    "exception_type": None,
                    "executed_count": 0,
                    "outputs_count": 0,
                    "peak_vram_gb": None,
                    "wall_time_s": round(time.time() - t0, 2),
                    "start_ts": _now(),
                    "end_ts": _now(),
                    "comfyui_log_basename": f"comfy_session_iter_{i:03d}.log",
                    "sweep_log_basename": None,
                }
            # Supervisor stamps its own wall-time for the worker;
            # worker's own wall_time_s is per-iter, supervisor's
            # `supervisor_wall_s` includes spawn + read overhead.
            res["supervisor_wall_s"] = round(time.time() - t0, 2)
            _append_master(res)
            history.append(res)

            _log(f"ITER {i} END status={res.get('status')} "
                 f"class={res.get('failure_class')} "
                 f"peak_VRAM={res.get('peak_vram_gb')}")

            # Step 6: stop conditions.
            if not args.no_stop_conditions:
                stop = _stop_decision(history)
                if stop is not None:
                    _log(f"STOP_DECISION: {stop}")
                    return 0 if stop.startswith("SUCCESS") else 3

            current_worker = None

            if i < args.iters:
                _log(f"sleeping {args.inter_iter_sec}s before next iter")
                time.sleep(args.inter_iter_sec)

    except KeyboardInterrupt:
        _log("=== overnight bug-hunt SUPERVISOR Ctrl+C ===")
        return 130

    _log("=== overnight bug-hunt SUPERVISOR DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
