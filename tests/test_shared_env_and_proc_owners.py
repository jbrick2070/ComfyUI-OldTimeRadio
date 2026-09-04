"""The two owners of the registry collapse, tested as contracts.

``nodes/_otr_shared/env.py`` is a SPELLING: whatever ``os.environ`` would have
returned, returned unchanged, read live on every call. The three guards
(``tests/test_env_single_owner.py`` and friends) prove nothing else spells it;
these prove the spelling itself is faithful, because a hundred-file migration to
an owner that quietly changes a value is worse than no migration at all.

``nodes/_otr_shared/proc.py`` is a BOUNDARY: the same spawn, plus a named
refusal for anything outside the executables this pack measurably runs. The
allowlist cases below are fed from the argv receipt
(``docs/2026-09-04-registry-findings-collapse/argv0_receipt.txt``), in both the
Windows and the POSIX shape, because ``argv[0]`` is an absolute path at most of
the 35 sites.

CPU-only: no GPU, no model load, and the one real spawn is the interpreter
already running the suite.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from nodes._otr_shared import env as otr_env
from nodes._otr_shared import proc as otr_proc


# --------------------------------------------------------------------------- #
# env: a spelling, and a LIVE one
# --------------------------------------------------------------------------- #
def test_get_returns_exactly_what_os_environ_would(monkeypatch):
    monkeypatch.setenv("OTR_PROBE_KNOB", "  spaced value  ")
    assert otr_env.get("OTR_PROBE_KNOB") == os.environ.get("OTR_PROBE_KNOB")
    assert otr_env.get("OTR_PROBE_KNOB") == "  spaced value  "


def test_an_unset_knob_returns_the_caller_default_untouched(monkeypatch):
    """The caller keeps its own default AND its own type. A live site passes an
    int (`int(otr_env.get("OTR_RADIO_BOOKEND_SEED", 4242))`), so narrowing this
    to `str | None` would describe a contract the pack does not keep."""
    monkeypatch.delenv("OTR_PROBE_KNOB", raising=False)
    assert otr_env.get("OTR_PROBE_KNOB") is None
    assert otr_env.get("OTR_PROBE_KNOB", "0") == "0"
    assert otr_env.get("OTR_PROBE_KNOB", 4242) == 4242


def test_reads_are_live_and_never_cached(monkeypatch):
    """conftest pops names at import, hundreds of tests monkeypatch, and both
    launchers pin at boot. One cached read breaks all three silently."""
    monkeypatch.delenv("OTR_PROBE_KNOB", raising=False)
    assert otr_env.get("OTR_PROBE_KNOB") is None
    monkeypatch.setenv("OTR_PROBE_KNOB", "1")
    assert otr_env.get("OTR_PROBE_KNOB") == "1"
    monkeypatch.setenv("OTR_PROBE_KNOB", "2")
    assert otr_env.get("OTR_PROBE_KNOB") == "2"
    monkeypatch.delenv("OTR_PROBE_KNOB")
    assert otr_env.get("OTR_PROBE_KNOB") is None


def test_pin_reaches_the_real_environment(monkeypatch):
    monkeypatch.delenv("OTR_PROBE_KNOB", raising=False)
    otr_env.pin("OTR_PROBE_KNOB", "9")
    try:
        assert os.environ["OTR_PROBE_KNOB"] == "9"
    finally:
        os.environ.pop("OTR_PROBE_KNOB", None)


@pytest.mark.parametrize("bad", [None, 9, 9.5, b"9", ["9"]])
def test_pin_refuses_a_non_string_and_NAMES_the_knob(monkeypatch, bad):
    """`None` is the likely one, and it is the dangerous one: a site meaning
    "unset this" that silently pinned nothing would be indistinguishable from a
    pin that worked."""
    monkeypatch.delenv("OTR_PROBE_KNOB", raising=False)
    with pytest.raises(TypeError) as exc:
        otr_env.pin("OTR_PROBE_KNOB", bad)
    assert "OTR_PROBE_KNOB" in str(exc.value)
    assert "OTR_PROBE_KNOB" not in os.environ


def test_setdefault_only_pins_when_the_operator_has_not(monkeypatch):
    """The try/finally is not decoration. `monkeypatch.delenv(raising=False)`
    on an ALREADY-ABSENT name records nothing to restore, so the setdefault
    below would leak OTR_PROBE_KNOB into the real process environment for the
    rest of the session -- and a test that pins a knob for every test after it
    is the exact failure this file is meant to catch elsewhere."""
    monkeypatch.delenv("OTR_PROBE_KNOB", raising=False)
    try:
        assert otr_env.setdefault("OTR_PROBE_KNOB", "a") == "a"
        assert otr_env.setdefault("OTR_PROBE_KNOB", "b") == "a"
        assert os.environ["OTR_PROBE_KNOB"] == "a"
    finally:
        os.environ.pop("OTR_PROBE_KNOB", None)


def test_setdefault_refuses_a_non_string(monkeypatch):
    monkeypatch.delenv("OTR_PROBE_KNOB", raising=False)
    with pytest.raises(TypeError):
        otr_env.setdefault("OTR_PROBE_KNOB", 1)


def test_unpin_returns_what_it_held_and_never_raises_for_an_unset_name(monkeypatch):
    monkeypatch.setenv("OTR_PROBE_KNOB", "held")
    assert otr_env.unpin("OTR_PROBE_KNOB") == "held"
    assert "OTR_PROBE_KNOB" not in os.environ
    assert otr_env.unpin("OTR_PROBE_KNOB") is None      # the second call is safe


def test_snapshot_is_a_copy_a_consumer_cannot_write_through(monkeypatch):
    """Both consumers hand it to something else -- the Blender spawn's sanitizer
    and the routing freeze -- and one of them copies again."""
    monkeypatch.setenv("OTR_PROBE_KNOB", "1")
    snap = otr_env.snapshot()
    assert snap["OTR_PROBE_KNOB"] == "1"
    assert snap is not os.environ
    snap["OTR_PROBE_KNOB"] = "mutated"
    snap["OTR_PROBE_ONLY_IN_THE_COPY"] = "x"
    assert os.environ["OTR_PROBE_KNOB"] == "1"
    assert "OTR_PROBE_ONLY_IN_THE_COPY" not in os.environ


# --------------------------------------------------------------------------- #
# proc: the same spawn, plus a named boundary
# --------------------------------------------------------------------------- #
#: Every argv[0] basename in the receipt. Bare and POSIX-shaped, so these hold
#: on every box.
_ALLOWED_ARGV0 = [
    "ffmpeg", "ffmpeg.exe", "FFmpeg.EXE",           # case and .exe normalize
    "/usr/bin/ffmpeg",
    "ffprobe", "/usr/bin/ffprobe",
    "git", "git.exe",
    "nvidia-smi", "nvidia-smi.exe",
    "blender", "/opt/blender/blender",
    # the sidecar venv interpreters -- the basename varies by box and by venv
    "python", "python.exe", "pythonw.exe", "python3", "python3.10",
    "/opt/venvs/indextts2/bin/python3.10",
]

#: The Windows shapes, which only NORMALIZE on Windows: `os.path.basename` is
#: the platform's, and a backslash is a legal filename character on POSIX. Kept
#: separate rather than made platform-independent on purpose -- a normalizer
#: that always split on backslash would let `/tmp/evil\ffmpeg` through on Linux,
#: and an allowlist that fails OPEN is worse than one that is platform-shaped.
_ALLOWED_ARGV0_WINDOWS = [
    r"C:\ffmpeg\bin\ffmpeg.exe",
    r"C:\ffmpeg\bin\ffprobe.exe",
    r"C:\Program Files\Git\cmd\git.exe",
    r"C:\Program Files\Blender Foundation\blender.exe",
    r"C:\Users\x\.venvs\indextts2\Scripts\python.exe",
]

_REFUSED_ARGV0 = [
    "curl", "powershell.exe", "cmd.exe", "bash", "sh", "wget", "/bin/sh",
    "ffmpeg-7.1",           # a versioned basename is NOT admitted; see below
    "blender-4.2",
]

_ON_WINDOWS = os.name == "nt"


@pytest.mark.parametrize("argv0", _ALLOWED_ARGV0)
def test_the_allowlist_admits_every_executable_the_receipt_found(argv0):
    # _check, not run(): the boundary is what is under test, and spawning
    # ffmpeg to prove it would make a CPU-only test need a binary.
    otr_proc._check([argv0, "-version"])            # must not raise


@pytest.mark.skipif(not _ON_WINDOWS, reason="basename is the platform's")
@pytest.mark.parametrize("argv0", _ALLOWED_ARGV0_WINDOWS)
def test_the_allowlist_admits_a_windows_absolute_path(argv0):
    otr_proc._check([argv0, "-version"])


@pytest.mark.parametrize("argv0", _REFUSED_ARGV0)
def test_an_unlisted_executable_is_refused_BEFORE_it_is_spawned(argv0):
    """Including the versioned basenames. They are not in the receipt, so they
    are not admitted -- and if one ever appears the error names it at the spawn,
    which is the whole point of having a boundary rather than a shim."""
    with pytest.raises(otr_proc.ExecutableNotAllowed) as exc:
        otr_proc._check([argv0, "-version"])
    assert "ALLOWED_EXECUTABLES" in str(exc.value)


@pytest.mark.skipif(not _ON_WINDOWS, reason="basename is the platform's")
def test_a_windows_absolute_path_to_an_unlisted_binary_is_still_refused():
    with pytest.raises(otr_proc.ExecutableNotAllowed):
        otr_proc._check([r"C:\Windows\System32\cmd.exe", "/c", "dir"])


def test_a_string_argv_is_refused_because_it_is_a_shell_parse():
    with pytest.raises(otr_proc.ExecutableNotAllowed) as exc:
        otr_proc.run("ffmpeg -version")
    assert "list" in str(exc.value)


def test_an_empty_argv_is_refused_by_NAME_not_by_IndexError():
    """The allowlist reads argv[0]; without this the empty case would surface as
    an IndexError from inside the owner."""
    with pytest.raises(otr_proc.ExecutableNotAllowed) as exc:
        otr_proc.run([])
    assert "empty" in str(exc.value)


@pytest.mark.parametrize("spawn", ["run", "popen"])
def test_shell_true_is_refused_on_both_entry_points(spawn):
    with pytest.raises(otr_proc.ExecutableNotAllowed) as exc:
        getattr(otr_proc, spawn)([sys.executable, "-c", "pass"], shell=True)
    assert "shell=True" in str(exc.value)


def test_run_returns_the_real_CompletedProcess_and_forwards_its_keywords():
    done = otr_proc.run([sys.executable, "-c", "print('otr')"],
                        stdout=otr_proc.PIPE, stderr=otr_proc.DEVNULL,
                        timeout=60, check=True)
    assert isinstance(done, subprocess.CompletedProcess)
    assert done.returncode == 0
    assert done.stdout.decode().strip() == "otr"


def test_popen_returns_the_real_Popen():
    child = otr_proc.popen([sys.executable, "-c", "pass"],
                           stdout=otr_proc.DEVNULL, stderr=otr_proc.DEVNULL)
    try:
        assert isinstance(child, subprocess.Popen)
    finally:
        child.wait(timeout=60)


def test_the_owner_wraps_no_exception():
    """Callers already catch subprocess errors by name. If the owner wrapped
    them, every existing `except` clause would go quietly dead."""
    with pytest.raises(subprocess.CalledProcessError):
        otr_proc.run([sys.executable, "-c", "raise SystemExit(3)"],
                     stdout=otr_proc.DEVNULL, stderr=otr_proc.DEVNULL,
                     check=True)


def test_the_re_exports_are_IDENTITY_aliases():
    """Tests construct `subprocess.CompletedProcess` and raise
    `subprocess.TimeoutExpired`; an alias that is merely equal would make an
    `except otr_proc.TimeoutExpired` clause silently stop matching."""
    for name in ("PIPE", "DEVNULL", "STDOUT", "CompletedProcess", "Popen",
                 "CalledProcessError", "TimeoutExpired"):
        assert getattr(otr_proc, name) is getattr(subprocess, name), name


def test_every_allowlist_entry_carries_its_reason():
    for name, reason in otr_proc.ALLOWED_EXECUTABLES.items():
        assert name == name.lower(), name
        assert reason.strip(), name
