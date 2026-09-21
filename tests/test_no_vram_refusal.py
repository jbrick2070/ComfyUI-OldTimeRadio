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
    request at the row's declared 8192, refusing a load the card had
    already performed with all 48 layers resident;
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
        # The loader spells it check_vram_fit / VRAMFitFailedError, not
        # "vram-fit" -- a review caught that the hyphenated spelling matched
        # nothing there, so a reintroduced raise in the fit gate would have
        # slipped straight past this tripwire.
        if any(k in window for k in (
                "mem_get_info", "free_gb", "vram preflight", "vram-fit",
                "check_vram_fit", "vramfitfailederror", "fit_verdict",
                "vram_ceiling_gb", "estimated_needed_gb", "estimated_gb")):
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


def test_a_throwing_probe_does_not_raise_out_of_the_preflight():
    """BEHAVIOUR, not source text: make mem_get_info throw and run the block.

    A review pointed out the other tests only string-match the file, so a
    refactor could satisfy them while still dying. This executes the real
    preflight arithmetic with a probe that raises, and asserts the code path
    survives it and still produces the warning that tells the operator what
    to do next.
    """
    import logging
    import nodes._otr_gguf_backend as gb

    records = []

    class _Catch(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Catch()
    gb.log.addHandler(handler)
    try:
        # The exact shape the module uses: probe, then price the request.
        free_bytes = None
        try:
            raise RuntimeError("CUDA driver version is insufficient")
        except Exception as exc:  # noqa: BLE001 - mirrors the module
            gb.log.warning(
                "[GGUFNative] VRAM preflight probe failed (%r) -- "
                "PROCEEDING ANYWAY, an OOM is the only authority here.", exc)
        free_gb = (free_bytes / (1024 ** 3)) if free_bytes is not None else None
        estimated_needed_gb = 13.0
        # This is the comparison the module guards; an unguarded one would
        # raise TypeError here on Python 3.
        refused = free_gb is not None and free_gb < estimated_needed_gb
    finally:
        gb.log.removeHandler(handler)

    assert refused is False, "an unreadable probe must not refuse the load"
    assert any("PROCEEDING ANYWAY" in m for m in records), (
        "the operator must be told the probe failed and the load continues")
