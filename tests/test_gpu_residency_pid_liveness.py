"""How the GPU-residency lease decides a lock holder is still alive.

``_pid_alive`` is the whole safety of the lease: say ALIVE about a dead pid and
a crashed run wedges the platform until a timeout; say DEAD about a live one and
two heavy engines load weights at once, which is the invariant the lease exists
to keep. So it fails SAFE -- only a definitively-absent pid returns False.

WHAT CHANGED (scan collapse, batch a, 2026-09-04): a ctypes ``OpenProcess``
probe used to answer this on Windows when psutil was missing. It was the pack's
one process-inspection site, and psutil is line 20 of ComfyUI core's own
``requirements.txt``, so the branch only ever ran on a box that was already
broken. It is gone; Windows without psutil now warns ONCE and returns True.

THE TRADE IS TESTED HERE, not just described: the warning has to name the
consequence (reclamation is off), it has to fire exactly once however many times
the function is polled, and ``os.kill`` must never be reached on Windows.

CPU-only: no GPU, no lease directory, no spawn.
"""
from __future__ import annotations

import logging
import sys

import pytest

from nodes._otr_shared import gpu_residency as gr


@pytest.fixture
def no_psutil(monkeypatch):
    """A box where importing psutil fails, with the once-per-process latch
    reset so the assertions below are not order-dependent."""
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(gr, "_WARNED_NO_PSUTIL", False)


def _never_called(*_a, **_k):
    raise AssertionError("os.kill was reached; it is POSIX-only now")


def test_a_nonpositive_pid_is_dead_without_asking_anything():
    assert gr._pid_alive(0) is False
    assert gr._pid_alive(-1) is False


def test_psutil_answers_when_it_is_there(monkeypatch):
    """The normal path on every platform, and the one the stale-lease
    reclamation in test_video_platform_aseam.py actually travels."""
    monkeypatch.setattr(gr.os, "kill", _never_called)
    assert gr._pid_alive(gr.os.getpid()) is True
    # implausibly high, so it cannot exist on any box
    assert gr._pid_alive(2_000_000_000) is False


def test_windows_without_psutil_says_ALIVE_and_never_calls_os_kill(
        monkeypatch, no_psutil):
    """os.name is patched too. Without that this test passes vacuously on a
    POSIX runner -- which is exactly where it would be least noticed."""
    monkeypatch.setattr(gr.os, "name", "nt")
    monkeypatch.setattr(gr.os, "kill", _never_called)
    assert gr._pid_alive(2_000_000_000) is True


def test_the_warning_fires_ONCE_however_often_liveness_is_polled(
        monkeypatch, no_psutil, caplog):
    """acquire() polls _pid_alive every 0.25 s for up to 120 s. An unlatched
    warning here would emit four times a second for two minutes and bury the
    rest of the run's log -- so 'once per process' is the contract, not a
    nicety."""
    monkeypatch.setattr(gr.os, "name", "nt")
    monkeypatch.setattr(gr.os, "kill", _never_called)
    with caplog.at_level(logging.WARNING, logger=gr.log.name):
        for _ in range(50):
            assert gr._pid_alive(2_000_000_000) is True
    warnings = [r for r in caplog.records if r.name == gr.log.name
                and r.levelno >= logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]


def test_the_warning_names_the_CONSEQUENCE_not_just_the_cause(
        monkeypatch, no_psutil, caplog):
    """"psutil is missing" tells an operator nothing they can act on. What they
    need to know is that a crashed run's lease will now time out instead of
    being reclaimed, and that installing psutil restores it."""
    monkeypatch.setattr(gr.os, "name", "nt")
    monkeypatch.setattr(gr.os, "kill", _never_called)
    with caplog.at_level(logging.WARNING, logger=gr.log.name):
        gr._pid_alive(2_000_000_000)
    text = " ".join(r.getMessage() for r in caplog.records
                    if r.name == gr.log.name)
    assert "psutil" in text
    assert "reclamation" in text
    assert "time out" in text


def test_no_process_inspection_call_survives_in_this_module():
    """The registry-scan half of the change, pinned so it cannot come back by
    accident. ``OpenProcess`` through ctypes was the pack's single
    `process_inspection` finding, and the fix is only worth anything while the
    call stays gone.

    AST, NOT A GREP -- and the first draft of this test proved why by going red
    on the COMMENT four lines above the deleted code, which explains what used
    to be there. A gate that cannot tell a call from prose about a call is a
    gate that gets deleted the first time it cries wolf."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(gr))
    calls = [n.func.attr for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert "OpenProcess" not in calls
    assert "WinDLL" not in calls
    imports = {a.name.split(".")[0] for n in ast.walk(tree)
               if isinstance(n, ast.Import) for a in n.names}
    assert "ctypes" not in imports
