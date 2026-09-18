#!/usr/bin/env python
"""Seconds-long working-gate pytest. Not the ~10 minute full suite.

Operator 2026-09-17: a 10-minute full suite is the CHUNK gate on a mature
row. Iteration uses this file so a window is not paying 10 minutes per edit.

From the pack root, Windows venv:

    $env:PYTHONUTF8=1
    C:\\Users\\jeffr\\Documents\\ComfyUI\\.venv\\Scripts\\python.exe scripts\\otr_working_gate.py

Append extra nodeids for the row you are in:

    ... python.exe scripts\\otr_working_gate.py tests/test_cast_lock.py::test_auto_registry_stamps_voice_refs

List the default set:

    ... python.exe scripts\\otr_working_gate.py --list

Chunk gate, once the row is green (~10 min):

    $env:PYTHONUTF8=1
    C:\\Users\\jeffr\\Documents\\ComfyUI\\.venv\\Scripts\\python.exe -m pytest -q -p no:cacheprovider tests

DEFAULT_NODEIDS holds closed-row pins still worth re-running plus any
OPEN row whose verify is already green. Drop a path in the same commit
that closes its row. Do not add a still-red verify (that is the row).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

# Closed-row pins plus the runner itself. ASCII only.
DEFAULT_NODEIDS: tuple[str, ...] = (
    "tests/test_brief_reader.py::test_the_music_prompt_normalises_its_setting_terms",
    "tests/test_my_story_runner.py::test_spoken_line_accepts_the_truncated_tex_key_from_the_runpod_act",
    "tests/test_working_gate_script.py",
)


def _python() -> str:
    env = os.environ.get("OTR_PYTHON", "").strip()
    if env:
        return env
    win = r"C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
    if os.path.isfile(win):
        return win
    return sys.executable


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "nodeids",
        nargs="*",
        help="extra pytest nodeids appended to DEFAULT_NODEIDS",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="print the default nodeids and exit",
    )
    args = ap.parse_args(argv)
    nodeids = list(DEFAULT_NODEIDS) + list(args.nodeids)
    if args.list:
        sys.stdout.write("\n".join(nodeids) + "\n")
        return 0
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    cmd = [
        _python(),
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--tb=short",
        *nodeids,
    ]
    return subprocess.call(cmd, cwd=_REPO, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
