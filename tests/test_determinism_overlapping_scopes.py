"""Overlapping ``deterministic_inference`` scopes restore the process state.

Audit B9 (2026-11-19). The cloud fan-out enters the scope from up to four
worker threads. Each entry used to save and restore its own snapshot of
process-global flags, so two scopes exiting out of order restored each
other's STRICT state: deterministic algorithms, cuDNN-deterministic and the
math-only SDPA pin stayed on for the rest of the server's life. On the M6 Mac
a switch inside the entry also left NaN memory filling on, after which MPS
baddbmm(beta=0) returned 1024/1024 NaN. The interleavings are driven
deterministically here; torch's flags are process-global, so this is exactly
what two worker threads do when the scheduler switches at that point.
UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import threading

import pytest

torch = pytest.importorskip("torch")

from nodes import _otr_determinism as det  # noqa: E402


def _state():
    return {
        "det": torch.are_deterministic_algorithms_enabled(),
        "fill": bool(torch.utils.deterministic.fill_uninitialized_memory),
        "cudnn_det": torch.backends.cudnn.deterministic,
        "bench": torch.backends.cudnn.benchmark,
        "mm_tf32": torch.backends.cuda.matmul.allow_tf32,
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "mem_eff": torch.backends.cuda.mem_efficient_sdp_enabled(),
    }


@pytest.fixture(autouse=True)
def _known_process_state():
    before = _state()
    yield
    assert getattr(det, "_SCOPE", {"depth": 0})["depth"] == 0
    torch.use_deterministic_algorithms(before["det"])
    torch.utils.deterministic.fill_uninitialized_memory = before["fill"]
    torch.backends.cudnn.deterministic = before["cudnn_det"]
    torch.backends.cudnn.benchmark = before["bench"]


def test_first_in_first_out_restores_everything():
    before = _state()
    a = det.deterministic_inference(1, warn_only=True)
    b = det.deterministic_inference(2, warn_only=True)
    a.__enter__()
    b.__enter__()
    a.__exit__(None, None, None)
    assert torch.are_deterministic_algorithms_enabled(), (
        "the scope still open must keep the strict state")
    b.__exit__(None, None, None)
    assert _state() == before


def test_a_switch_inside_the_entry_restores_everything(monkeypatch):
    """B enters between A's use_deterministic_algorithms(True) and the rest of
    A's setup -- the window that left NaN filling on, on the Mac."""
    before = _state()
    real = torch.use_deterministic_algorithms
    held = {}

    def switching(mode, **kw):
        real(mode, **kw)
        if mode and "b" not in held:
            # Can only happen without the scope lock; with it, B blocks until
            # A's setup is done. Entering on another thread proves that.
            held["b"] = det.deterministic_inference(2, warn_only=True)
            held["t"] = threading.Thread(target=held["b"].__enter__)
            held["t"].start()

    monkeypatch.setattr(torch, "use_deterministic_algorithms", switching)
    a = det.deterministic_inference(1, warn_only=True)
    a.__enter__()
    monkeypatch.setattr(torch, "use_deterministic_algorithms", real)
    held["t"].join(timeout=10)
    a.__exit__(None, None, None)
    held["b"].__exit__(None, None, None)
    assert _state() == before


def test_four_threads_overlapping_restore_everything():
    before = _state()
    gate = threading.Barrier(4)
    errors = []

    def worker(seed):
        try:
            with det.deterministic_inference(seed, warn_only=True):
                gate.wait(timeout=10)
        except BaseException as exc:  # noqa: BLE001 -- surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20)
    assert not errors
    assert _state() == before


def test_a_single_scope_still_applies_and_restores():
    before = _state()
    with det.deterministic_inference(7, warn_only=True):
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cuda.matmul.allow_tf32 is False
    assert _state() == before


def test_the_same_seed_gives_the_same_draw():
    with det.deterministic_inference(11):
        first = torch.rand(4)
    with det.deterministic_inference(11):
        second = torch.rand(4)
    assert torch.equal(first, second)


def test_a_seed_torch_refuses_leaves_no_scope_open():
    """Sonnet QA 2026-11-19: the depth was counted before seeding, so a seed
    torch rejects (outside 64 bits) raised with the scope still counted and
    the strict state stayed on for the rest of the process."""
    before = _state()
    for _ in range(2):
        with pytest.raises((ValueError, RuntimeError, OverflowError)):
            with det.deterministic_inference(2 ** 70):
                pass
    assert det._SCOPE["depth"] == 0
    assert _state() == before
