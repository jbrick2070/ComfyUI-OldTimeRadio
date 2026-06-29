"""Wave 3 / F -- the audio dependency-pilot harness (scripts/otr_audio_dep_pilot).

Headless structure tests. The opt-in engine libraries are absent in the pytest
sandbox, so every probe must report ``lib_absent`` WITHOUT crashing and WITHOUT
importing a banned dependency. The harness is the operator's GPU-box tool; here
we lock its pure helpers, its absent-lib behavior, and its no-drift contract
with the real adapter registry.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PILOT_SRC = REPO_ROOT / "scripts" / "otr_audio_dep_pilot.py"


def _load_pilot():
    spec = importlib.util.spec_from_file_location("otr_audio_dep_pilot", PILOT_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PILOT = _load_pilot()


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_forward_accepts_generator_tristate():
    def with_gen(a, generator=None):
        return a

    def without_gen(a, b):
        return a

    def with_kwargs(a, **kw):
        return a

    assert PILOT.forward_accepts_generator(with_gen) is True
    assert PILOT.forward_accepts_generator(without_gen) is False
    assert PILOT.forward_accepts_generator(with_kwargs) is True
    # Non-introspectable callable -> None (the bit_exact-disqualified state).
    assert PILOT.forward_accepts_generator(123) is None


def test_snapshot_violations_flags_torch_change_and_banned_deps():
    clean = {"torch": "2.10.0", "xformers": False, "flash_attn": False}
    assert PILOT.snapshot_violations(clean, clean) == []

    swapped = {"torch": "2.11.0", "xformers": False, "flash_attn": False}
    viol = PILOT.snapshot_violations(clean, swapped)
    assert any("torch version changed" in v for v in viol)

    pulled = {"torch": "2.10.0", "xformers": True, "flash_attn": False}
    viol = PILOT.snapshot_violations(clean, pulled)
    assert any("xformers" in v for v in viol)


def test_dep_snapshot_shape():
    snap = PILOT.dep_snapshot()
    assert set(snap) == {"torch", "xformers", "flash_attn"}
    # torch is resident in the test env; the banned-dep probes are booleans.
    assert snap["torch"] is not None
    assert isinstance(snap["xformers"], bool)
    assert isinstance(snap["flash_attn"], bool)


# --------------------------------------------------------------------------- #
# Probe behavior (headless: libraries absent)
# --------------------------------------------------------------------------- #
def test_probe_one_absent_lib_is_clean_and_imports_no_banned_dep():
    for name in PILOT.PROBE_ENGINES:
        v = PILOT.probe_one(name, do_import=True)
        assert v["engine"] == name
        # Library absent in the sandbox -> a clean, non-crashing verdict.
        assert v["status"] in {"lib_absent", "import_error"}, v
        assert v["adapter_registered"] is True, "adapter must be registered"
        assert "TODO-for-F" in v["assumed_call"]
        assert v["supports_external_generator_ready"] is False
    # The probes must not have dragged a banned dep into this process.
    assert "xformers" not in sys.modules
    assert "flash_attn" not in sys.modules


def test_probe_one_unknown_engine():
    v = PILOT.probe_one("not_a_real_engine")
    assert v["status"] == "unknown_engine"


def test_probe_one_no_import_reports_structure_only():
    v = PILOT.probe_one("chatterbox", do_import=False)
    assert v["status"] == "not_imported"
    assert v["adapter_registered"] is True
    # The adapter forward does not yet thread a generator (that is the F TODO).
    assert v["adapter_forward_binds_generator"] in (False, None)


def test_run_pilot_in_process_headless_not_ready():
    report = PILOT.run_pilot(isolated=False)
    assert report["engine_count"] == len(PILOT.PROBE_ENGINES)
    assert report["ready_count"] == 0  # libs absent -> nothing structurally ready
    assert report["all_structural_preconditions_met"] is False
    assert {v["engine"] for v in report["engines"]} == set(PILOT.PROBE_ENGINES)


# --------------------------------------------------------------------------- #
# No-drift contract with the real adapter registry
# --------------------------------------------------------------------------- #
def test_probe_engines_match_registry_adapters():
    from nodes import _otr_audio_engines as AE

    for name, spec in PILOT.PROBE_ENGINES.items():
        adapter = AE.get_engine(name)  # raises if the name is unregistered
        assert type(adapter).__name__ == spec["adapter_class"]
        assert hasattr(adapter, spec["forward"]), (
            f"{name} adapter is missing its forward {spec['forward']!r}"
        )
        # NOTE: the probe manifest no longer carries a "flag" key (C5 -- the
        # registry IS the menu; flags do not gate). module/class/forward is the
        # no-drift contract.


def test_pilot_covers_the_dep_needing_engines():
    from nodes import _otr_audio_engines as AE

    # The dep pilot's probe set is the CURATED list of NON-NATIVE voice/music
    # engines that need Blackwell dependency validation (an external sidecar
    # library OR an out-of-process venv). It is metadata, NOT derived from a flag
    # gate (the registry IS the menu; C5). No-drift contract: every probe entry is
    # a REGISTERED, NON-NATIVE engine, and the byte-identical in-graph defaults
    # (bark / kokoro / musicgen) + the ComfyUI-native engine (stable_audio_3) are
    # EXCLUDED -- they have no external lib to probe.
    probe = set(PILOT.PROBE_ENGINES)
    for name in probe:
        adapter = AE.get_engine(name)                # raises if unregistered
        assert not getattr(adapter, "native", False), name
    for excluded in ("bark", "kokoro", "musicgen", "stable_audio_3"):
        assert excluded not in probe, excluded


def test_source_is_ascii_no_em_dash():
    src = PILOT_SRC.read_text(encoding="utf-8")
    assert "—" not in src, "em-dash forbidden in OTR python source (CLAUDE.md)"
    src.encode("ascii")  # ASCII-only source


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
