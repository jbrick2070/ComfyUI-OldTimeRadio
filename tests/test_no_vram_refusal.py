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

Two kinds of check live here. The AST tripwire asserts the VRAM path holds
no `raise`, so a future "small safety check" cannot quietly reintroduce a
gate. The behavioural tests CALL `vram_preflight_report` with a probe that
throws and with a reading that falls short, because a review caught an
earlier test re-implementing the arithmetic inline and grading its own
copy -- it would have stayed green while the real code regained a guard.

KNOWN LIMIT of the tripwire, worth stating rather than pretending: it
looks at the lines around each `raise`, so a gate reintroduced inside a
separately-named helper would evade it. The behavioural tests are the
real protection; the tripwire is the cheap early warning.
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


def test_a_throwing_probe_does_not_refuse_the_load(monkeypatch, caplog):
    """CALL the real preflight with a probe that raises.

    The test this replaces re-implemented the arithmetic inline and asserted
    on its own copy, so deleting the guard from the module would not have
    failed it -- a QA pass caught that. This imports the module function and
    runs it, which is the only version that can notice.
    """
    import logging
    import pathlib
    import torch

    import nodes._otr_gguf_backend as gb

    def _boom(*_a, **_k):
        raise RuntimeError("CUDA driver version is insufficient")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", _boom)

    with caplog.at_level(logging.INFO, logger=gb.log.name):
        result = gb.vram_preflight_report(
            device="cuda",
            model_path=pathlib.Path(gb.__file__),   # a real file, for st_size
            n_ctx=4096,
            eff_kv_rate=0.7,
            eff_n_gpu_layers=-1,
            test_mode=False,
        )

    assert result is None, "the preflight reports; it must not return a verdict"
    assert any("PROCEEDING ANYWAY" in m for m in caplog.messages), (
        "an unreadable probe must say out loud that the load continues")


def test_an_insufficient_reading_does_not_refuse_either(monkeypatch, caplog):
    """A real reading that falls short is still only a recommendation."""
    import logging
    import pathlib
    import torch

    import nodes._otr_gguf_backend as gb

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info",
        lambda *_a, **_k: (1 * 1024 ** 3, 32 * 1024 ** 3))

    with caplog.at_level(logging.INFO, logger=gb.log.name):
        result = gb.vram_preflight_report(
            device="cuda",
            model_path=pathlib.Path(gb.__file__),
            n_ctx=4096,
            eff_kv_rate=0.7,
            eff_n_gpu_layers=-1,
            test_mode=False,
        )

    assert result is None
    assert any("EXCEEDS free" in m for m in caplog.messages), (
        "the shortfall must be reported even though it does not refuse")


def test_a_cpu_policy_skips_the_preflight_entirely(monkeypatch, caplog):
    """The one branch the extraction did not cover directly.

    A cpu-device policy has no VRAM to fit, so the original block was skipped
    wholesale. The extracted guard is a De Morgan inversion of that condition,
    which is the kind of rewrite that is correct until it is not -- so pin it
    rather than reason about it. `mem_get_info` is made to explode: if the
    guard ever stops short-circuiting, this test fails loudly instead of
    silently probing on a machine with no GPU.
    """
    import logging
    import pathlib
    import torch

    import nodes._otr_gguf_backend as gb

    def _must_not_be_called(*_a, **_k):
        raise AssertionError("a cpu policy must never probe VRAM")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", _must_not_be_called)

    with caplog.at_level(logging.INFO, logger=gb.log.name):
        result = gb.vram_preflight_report(
            device="cpu",
            model_path=pathlib.Path(gb.__file__),
            n_ctx=4096,
            eff_kv_rate=0.7,
            eff_n_gpu_layers=0,
            test_mode=False,
        )

    assert result is None
    assert not any("VRAM Preflight" in m for m in caplog.messages), (
        "a cpu policy must log nothing about VRAM")


def test_an_unavailable_cuda_runtime_skips_the_preflight(monkeypatch, caplog):
    """Same guard, the other half: cuda requested but not available."""
    import logging
    import pathlib
    import torch

    import nodes._otr_gguf_backend as gb

    def _must_not_be_called(*_a, **_k):
        raise AssertionError("must not probe when cuda is unavailable")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "mem_get_info", _must_not_be_called)

    with caplog.at_level(logging.INFO, logger=gb.log.name):
        result = gb.vram_preflight_report(
            device="cuda",
            model_path=pathlib.Path(gb.__file__),
            n_ctx=4096,
            eff_kv_rate=0.7,
            eff_n_gpu_layers=-1,
            test_mode=False,
        )

    assert result is None
    assert not any("VRAM Preflight" in m for m in caplog.messages)
