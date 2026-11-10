"""Self-tests for the host system-spec collector used by the episode
treatment / credits sheet (nodes/_otr_sys_specs.py).

The collector is BEST-EFFORT and must NEVER raise -- every probe degrades to
"(unknown)" rather than failing the render. These tests assert the contract:
a flat str->str dict with all expected keys, present even when optional deps
(psutil / torch) are missing.
"""
from __future__ import annotations

import builtins

from nodes._otr_sys_specs import collect_system_specs

_EXPECTED_KEYS = {
    "os", "cpu", "cpu_cores", "ram", "ram_peak",
    "gpu", "vram", "cuda", "python", "torch",
}


def test_collect_never_reads_the_machine_name(monkeypatch):
    """The credits burn these specs into a video made to be shared, and a
    computer's network name often carries its owner's name (2026-11-10,
    registry information-disclosure on 2.3.51). Probed with a sentinel so the
    test cannot pass by luck on a box whose name happens to be unusual."""
    import platform
    import socket
    monkeypatch.setattr(platform, "node", lambda: "SENTINEL-MACHINE-NAME")
    monkeypatch.setattr(socket, "gethostname", lambda: "SENTINEL-MACHINE-NAME")
    specs = collect_system_specs()
    assert "hostname" not in specs
    assert not any("SENTINEL-MACHINE-NAME" in v for v in specs.values())


def test_collect_returns_all_keys_as_strings():
    specs = collect_system_specs()
    assert isinstance(specs, dict)
    assert _EXPECTED_KEYS.issubset(specs.keys())
    for k, v in specs.items():
        assert isinstance(v, str), f"{k} -> {type(v)} (must be str)"
    # Python version is always probeable in-process.
    assert specs["python"] not in ("", "(unknown)")


def test_collect_never_raises_without_optional_deps(monkeypatch):
    """Force psutil + torch imports to fail; the collector must still return a
    full dict with the unprobeable fields degraded to '(unknown)'."""
    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name in ("psutil", "torch"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)
    specs = collect_system_specs()
    assert _EXPECTED_KEYS.issubset(specs.keys())
    # torch-dependent fields degrade cleanly (nvidia-smi may or may not exist).
    assert specs["torch"] == "(unknown)"
