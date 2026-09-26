"""prestartup_script.py must never import torch (2026-09-26).

ComfyUI prints "Torch already imported, torch should never be imported before
this point" when a prestartup script loads torch, and ours did on every boot:
its Mac attention fix imported torch to ask whether MPS exists. The fix asks
the platform instead. Each case runs the real script in a FRESH interpreter,
because the test session itself has long since imported torch.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "prestartup_script.py"

_ENV = dict(os.environ, PYTHONUTF8="1", OTR_SKIP_KOKORO_PREFETCH="1",
            HF_HUB_OFFLINE="1", OTR_TEST_MODE="1")


def _run(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, encoding="utf-8", env=_ENV, cwd=str(REPO),
                         timeout=180)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


def test_prestartup_leaves_torch_unimported_on_this_platform():
    code = ("import runpy, sys\n"
            "runpy.run_path(%r, run_name='prestartup_script')\n"
            "print('torch' in sys.modules)\n" % str(SCRIPT))
    assert _run(code) == "False"


def test_on_macos_the_attention_flag_is_set_without_torch():
    """The Mac branch, exercised on any box: a stand-in `comfy.cli_args` holds
    the args object ComfyUI reads, and `sys.platform` reads darwin."""
    code = ("import runpy, sys, types\n"
            "sys.platform = 'darwin'\n"
            "args = types.SimpleNamespace(use_pytorch_cross_attention=False,\n"
            "    use_split_cross_attention=False, use_quad_cross_attention=False)\n"
            "comfy = types.ModuleType('comfy'); comfy.__path__ = []\n"
            "cli = types.ModuleType('comfy.cli_args'); cli.args = args\n"
            "sys.modules['comfy'] = comfy; sys.modules['comfy.cli_args'] = cli\n"
            "runpy.run_path(%r, run_name='prestartup_script')\n"
            "print(args.use_pytorch_cross_attention, 'torch' in sys.modules)\n"
            % str(SCRIPT))
    assert _run(code) == "True False"
