# -*- coding: utf-8 -*-
"""No VRAM arithmetic may refuse a load. An OOM is the only authority.

Operator directive, restated 2026-09-21: "I don't want any VRAM guards."
It is the same rule as 2026-08-29's "prove me wrong with an OOM but don't put
an artificial gate", and the reason is measured rather than stylistic: every
one of these estimates has been WRONG in the direction that refuses working
configurations --

  * the GGUF estimate priced the whole file on the GPU and ignored
    `n_gpu_layers`, so a 12B at a partial offload was refused;
  * the fit gate priced a Q4_K_M as a Q8_0 (PBUG-20260829-08) and a 4096
    request at the row's 8192 (PBUG-20260829-20), refusing a load the card
    had already performed with all 48 layers resident;
  * the probe refusal killed the episode when the READING failed, which is
    not evidence about whether the model fits.

This test is a tripwire: it asserts the VRAM path contains no raise, so a
future "small safety check" cannot quietly reintroduce a gate.
"""
import ast
import pathlib

PACK = pathlib.Path(__file__).resolve().parents[1]
GGUF = PACK / "nodes" / "_otr_gguf_backend.py"
LOADER = PACK / "nodes" / "_otr_model_loader.py"


def _raises_inside_vram_block(path: pathlib.Path) -> list:
    """Every `raise` whose enclosing lines mention a VRAM reading."""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines()
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue
        lo = max(0, node.lineno - 14)
        window = "\n".join(lines[lo:node.lineno]).lower()
        if ("mem_get_info" in window or "free_gb" in window
                or "vram preflight" in window or "vram-fit" in window
                or "estimated_needed_gb" in window):
            found.append((path.name, node.lineno,
                          lines[node.lineno - 1].strip()[:80]))
    return found


def test_the_gguf_vram_preflight_never_raises():
    offenders = _raises_inside_vram_block(GGUF)
    assert offenders == [], (
        "a VRAM reading must never refuse a load -- an OOM is the only "
        "authority. Offending raises: %r" % (offenders,))


def test_the_fit_gate_never_raises():
    offenders = _raises_inside_vram_block(LOADER)
    assert offenders == [], (
        "the VRAM-fit estimate is a recommendation, not a capability "
        "refusal. Offending raises: %r" % (offenders,))


def test_a_probe_failure_is_survivable_in_source():
    """The probe's except branch must log and continue, not raise."""
    src = GGUF.read_text(encoding="utf-8")
    i = src.index("mem_get_info")
    window = src[i:i + 1400]
    assert "PROCEEDING ANYWAY" in window, (
        "the probe failure path no longer says it proceeds; a guard may have "
        "been reintroduced")
    assert "free_gb is not None" in src, (
        "the estimate comparison must tolerate an unreadable probe rather "
        "than depending on it")
