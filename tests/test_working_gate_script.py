"""The working-gate runner stays pointed at files that exist."""
from __future__ import annotations

import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "otr_working_gate.py"


def test_working_gate_script_lists_existing_paths():
    env = {**dict(**__import__("os").environ), "PYTHONUTF8": "1"}
    out = subprocess.check_output(
        [sys.executable, str(SCRIPT), "--list"],
        cwd=str(REPO),
        env=env,
        text=True,
    )
    rows = [ln.strip() for ln in out.splitlines() if ln.strip()]
    assert rows, "working gate listed nothing"
    for nodeid in rows:
        rel = nodeid.split("::", 1)[0]
        assert (REPO / rel).is_file(), nodeid
