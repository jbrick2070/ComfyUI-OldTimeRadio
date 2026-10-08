"""scripts/kill_otr_zombies.ps1 selects ONLY orphaned OTR sidecars.

The script used to kill any python with more than 10 s of CPU, and every
python at all when nothing listened on :8000 -- which reached the Desktop
ComfyUI on :8188 and unrelated applications. It now requires a positive OTR
marker AND a provably dead parent. These tests drive its -InventoryPath test
mode with a mocked process table; that mode never terminates anything.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "kill_otr_zombies.ps1"

PACK = r"D:\custom_nodes\ComfyUI-OldTimeRadio"
PY = r'"C:\ComfyUI\.venv\Scripts\python.exe"'


def _proc(pid, ppid, name, cmd, created):
    return {"ProcessId": pid, "ParentProcessId": ppid, "Name": name,
            "CommandLine": cmd, "CreationDate": created}


INVENTORY = [
    _proc(4, 0, "System", "", "2026-10-08T00:00:00"),
    _proc(50, 4, "Comfy Desktop.exe", r'"C:\Program Files\Comfy Desktop\Comfy Desktop.exe"', "2026-10-08T07:59:00"),
    # Desktop ComfyUI on :8188 -- never a target.
    _proc(100, 50, "python.exe", PY + r' "C:\ComfyUI\main.py" --port 8188', "2026-10-08T08:00:00"),
    # An orphaned headless ComfyUI (parent gone) is still ComfyUI -- never a target.
    _proc(211, 990, "python.exe", PY + r' D:\ComfyUI\main.py --port 8000', "2026-10-08T08:30:00"),
    # Orphaned Chatterbox worker: parent PID no longer exists -> TARGET.
    _proc(201, 999, "python.exe", PY + " " + PACK + r"\scripts\_otr_chatterbox_worker.py --serve", "2026-10-08T09:00:00"),
    # IndexTTS2 worker whose ComfyUI parent is alive and older -> live, not a target.
    _proc(202, 100, "python.exe", PY + " " + PACK + r"\scripts\_otr_indextts2_worker.py", "2026-10-08T09:05:00"),
    # Worker whose parent PID was recycled by a YOUNGER process -> TARGET.
    _proc(300, 4, "notepad.exe", r"C:\Windows\notepad.exe", "2026-10-08T10:00:00"),
    _proc(203, 300, "python.exe", PY + " " + PACK + r"\scripts\_otr_chatterbox_worker.py", "2026-10-08T09:10:00"),
    # Unrelated orphaned python, and an unrelated busy python -> never targets.
    _proc(204, 998, "python.exe", PY + r" C:\tools\other_app.py", "2026-10-08T09:00:00"),
    _proc(206, 50, "pythonw.exe", PY + r" C:\tools\heavy_cpu_job.py", "2026-10-08T08:10:00"),
    # A Claude / MCP helper is protected even if it matches a worker name.
    _proc(205, 997, "python.exe", PY + r" C:\Users\u\desktop-commander\_otr_fake_worker.py", "2026-10-08T09:00:00"),
    # Ambiguous identities are skipped: no creation time, no recorded parent.
    _proc(207, 995, "python.exe", PY + " " + PACK + r"\scripts\_otr_chatterbox_worker.py", None),
    _proc(208, 0, "python.exe", PY + " " + PACK + r"\scripts\_otr_indextts2_worker.py", "2026-10-08T09:00:00"),
    # ffmpeg: OTR path + dead parent -> TARGET; no OTR marker -> never;
    # OTR path but a live older parent -> not a target.
    _proc(301, 996, "ffmpeg.exe", r"ffmpeg -y -i C:\Temp\otr_assemble_ab12\seg_001.mp4 -c copy out.mp4", "2026-10-08T09:20:00"),
    _proc(302, 994, "ffmpeg.exe", r"ffmpeg -i C:\Videos\holiday.mp4 out.mp4", "2026-10-08T09:20:00"),
    _proc(303, 100, "ffmpeg.exe", r"ffmpeg -i D:\ComfyUI\output\otr\episodes\ep1\a.wav b.wav", "2026-10-08T09:30:00"),
    # The worker must be the SCRIPT python runs, not a mention: a linter run on
    # a worker file is not a worker (Codex review of 91389427).
    _proc(209, 991, "python.exe", PY + r" C:\tools\pylint.py " + PACK + r"\scripts\_otr_chatterbox_worker.py",
          "2026-10-08T09:00:00"),
    # A real worker launched with -u from a quoted path with spaces -> TARGET.
    _proc(210, 989, "python.exe", PY + r' -u "C:\Program Files\OTR Pack\scripts\_otr_indextts2_worker.py" --model-dir m',
          "2026-10-08T09:00:00"),
    # Only no-value flags, -X/-W with a value, and `--` may sit between the
    # interpreter and the worker; a quoted bare worker name (cwd launch) counts.
    _proc(212, 986, "python.exe", PY + " -X utf8 " + PACK + r"\scripts\_otr_chatterbox_worker.py",
          "2026-10-08T09:00:00"),
    _proc(213, 985, "python.exe", PY + " -- " + PACK + r"\scripts\_otr_chatterbox_worker.py",
          "2026-10-08T09:00:00"),
    _proc(214, 984, "python.exe", PY + ' "_otr_indextts2_worker.py" --device cpu', "2026-10-08T09:00:00"),
    # -m runs a MODULE named that, not the worker script -> never a target.
    _proc(215, 983, "python.exe", PY + " -m _otr_chatterbox_worker.py", "2026-10-08T09:00:00"),
    # Path markers start at a segment boundary: neither of these is OTR's.
    _proc(305, 988, "ffmpeg.exe", r"ffmpeg -i C:\Videos\not_otr_cbx_report.wav out.mp4", "2026-10-08T09:20:00"),
    _proc(306, 987, "ffmpeg.exe", r"ffmpeg -i D:\backup\ComfyUI-OldTimeRadio-old\a.wav b.wav", "2026-10-08T09:20:00"),
    # imageio-ffmpeg's versioned build, which nodes/_otr_shared/ffmpeg.py falls
    # back to: same rules as ffmpeg.exe -> TARGET when orphaned with an OTR path.
    _proc(304, 992, "ffmpeg-win-x86_64-v7.1.exe",
          r"ffmpeg-win-x86_64-v7.1.exe -i D:\ComfyUI\output\otr\episodes\ep2\b.wav c.wav", "2026-10-08T09:40:00"),
    # Any ffmpeg* build (proc.py's own prefix rule), e.g. a pinned OTR_FFMPEG;
    # ffprobe is never an ffmpeg target, even on an OTR path.
    _proc(307, 982, "ffmpeg7.exe", r"ffmpeg7 -i D:\ComfyUI\output\otr\obs\ep3.mp4 -c copy x.mp4",
          "2026-10-08T09:40:00"),
    _proc(308, 981, "ffprobe.exe", r"ffprobe D:\ComfyUI\output\otr\obs\ep3.mp4", "2026-10-08T09:40:00"),
]


def _powershell():
    if os.name != "nt":
        pytest.skip("PowerShell 5.1 script")
    exe = shutil.which("powershell.exe") or shutil.which("powershell")
    if not exe:
        pytest.skip("PowerShell is unavailable")
    return exe


def _select(tmp_path, inventory):
    inv = tmp_path / "inventory.json"
    inv.write_text(json.dumps(inventory), encoding="utf-8")
    result = subprocess.run(
        [_powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(SCRIPT), "-InventoryPath", str(inv)],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    lines = [ln for ln in result.stdout.splitlines() if ln.startswith("SELECTED_JSON: ")]
    assert len(lines) == 1, result.stdout + result.stderr
    rows = json.loads(lines[0][len("SELECTED_JSON: "):])
    if isinstance(rows, dict):
        rows = [rows]
    return {row["PID"]: row["Kind"] for row in rows}


def test_selects_only_orphaned_otr_sidecars(tmp_path):
    assert _select(tmp_path, INVENTORY) == {
        201: "otr-worker",
        203: "otr-worker",
        210: "otr-worker",
        212: "otr-worker",
        213: "otr-worker",
        214: "otr-worker",
        301: "otr-ffmpeg",
        304: "otr-ffmpeg",
        307: "otr-ffmpeg",
    }


def test_dst_fall_back_does_not_make_a_live_parent_look_younger(tmp_path):
    # 2026-11-01, US Pacific fall-back: the parent started 01:50 PDT (08:50Z)
    # and its child 01:20 PST (09:20Z). Compared as local wall-clock the
    # parent looks 30 minutes YOUNGER -- a "recycled PID" -- and the live
    # worker would be killed. Compared in UTC the parent is older: parented.
    # (The pre-fix script only reverses this pair on a US-Pacific clock; on
    # other zones the test still pins the correct answer.)
    inventory = [
        _proc(4, 0, "System", "", "2026-11-01T00:00:00Z"),
        _proc(100, 4, "python.exe", PY + r' "C:\ComfyUI\main.py"', "2026-11-01T01:50:00-07:00"),
        _proc(201, 100, "python.exe", PY + " " + PACK + r"\scripts\_otr_chatterbox_worker.py",
              "2026-11-01T01:20:00-08:00"),
    ]
    assert _select(tmp_path, inventory) == {}


def test_an_empty_inventory_path_is_an_error_not_live_mode():
    result = subprocess.run(
        [_powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(SCRIPT), "-InventoryPath", ""],
        capture_output=True, text=True, timeout=60)
    combined = result.stdout + result.stderr
    assert result.returncode == 2, combined
    assert "SELECTED_JSON" not in combined
    assert "Orphaned OTR sidecars" not in combined
    assert "No orphaned OTR sidecars" not in combined


def test_empty_inventory_selects_nothing(tmp_path):
    assert _select(tmp_path, []) == {}


def test_no_comfy_listener_does_not_widen_selection(tmp_path):
    # Nothing listens anywhere and every python is busy or orphaned, but
    # none carries an OTR marker: nothing may be selected.
    inventory = [
        _proc(4, 0, "System", "", "2026-10-08T00:00:00"),
        _proc(401, 993, "python.exe", PY + r" C:\tools\a.py", "2026-10-08T09:00:00"),
        _proc(402, 4, "python.exe", PY + r" C:\tools\b.py", "2026-10-08T09:00:00"),
    ]
    assert _select(tmp_path, inventory) == {}


def test_old_heuristics_are_gone():
    src = SCRIPT.read_text(encoding="utf-8")
    assert "-gt 10" not in src, "CPU-time targeting must not return"
    assert "LocalPort 8000" not in src, "a missing :8000 listener must not widen selection"
    # Test mode must exit before any Stop-Process call can run.
    test_mode = src.index("if ($PSBoundParameters.ContainsKey('InventoryPath')) {")
    assert src.index("exit 0", test_mode) < src.index("Stop-Process")
