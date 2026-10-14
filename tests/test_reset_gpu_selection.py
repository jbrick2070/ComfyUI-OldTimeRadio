"""scripts/otr_reset_gpu.ps1 never selects the tools that drive the session.

CLAUDE.md section 4 bans a blanket ``Stop-Process -Name python``: it also kills
the Claude MCP extension pythons and severs the session's own tools mid-run. The
script's two lists ARE that safety property -- a process is a target only when
its command line matches a kill marker AND carries no never-kill marker. No
PowerShell is needed to check it: read both lists out of the script and apply
them with PowerShell's own semantics (``-match`` is case-insensitive, and the
never-kill list is matched as an escaped literal).
"""
import re
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "otr_reset_gpu.ps1"
TEXT = SCRIPT.read_text(encoding="utf-8")


def _list(name):
    m = re.search(r"\$%s\s*=\s*@\((.*?)\)" % name, TEXT, re.S)
    assert m, "the script no longer declares $%s" % name
    return re.findall(r"'([^']+)'", m.group(1))


NEVER = _list("NeverKill")
KILL = _list("KillMarkers")


def selected(cmdline):
    low = cmdline.lower()
    if any(n.lower() in low for n in NEVER):
        return False
    return any(re.search(m, cmdline, re.I) for m in KILL)


def test_a_comfyui_server_on_either_box_is_a_target():
    assert selected(r'"C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe" '
                    r'C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\main.py --port 8000 --cuda-malloc')
    assert selected(r'C:\OTR-CleanRoom\python_embeded\python.exe -s ComfyUI\main.py '
                    r'--windows-standalone-build --port 8000')


def test_the_headless_harness_is_a_target():
    assert selected(r'python.exe C:\repo\scripts\otr_soak_run.py --legs 3')
    assert selected(r'python.exe main.py --extra-model-paths-config '
                    r'C:\repo\scripts\_otr_headless_model_paths.yaml')


def test_the_claude_tooling_is_never_a_target_even_when_it_names_comfyui():
    for cmd in (
        r'python.exe "C:\Users\u\AppData\Roaming\Claude\Claude Extensions\ant.dir.desktop-commander\server.py"',
        r'python.exe C:\tools\windows-mcp\server.py --watch ComfyUI\main.py',
        r'python.exe -m claude-code.bridge ComfyUI\main.py',
        r'python.exe C:\x\ModelContextProtocol\run.py ComfyUI\main.py',
    ):
        assert not selected(cmd), cmd


def test_an_unrelated_python_is_left_alone():
    assert not selected(r'python.exe C:\tools\other_app.py')
    assert not selected(r'python.exe -m http.server 8000')


def test_the_port_holder_is_checked_against_the_never_kill_list_too():
    """The second sweep kills whatever listens on the port; it must apply the same
    protection, or a protected process squatting on :8000 would be terminated."""
    assert TEXT.count("$NeverKill") >= 3, (
        "the never-kill list must guard the process sweep AND the port-holder kill")
