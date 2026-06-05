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
    for name in PILOT.OPT_IN_ENGINES:
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
    assert report["engine_count"] == len(PILOT.OPT_IN_ENGINES)
    assert report["ready_count"] == 0  # libs absent -> nothing structurally ready
    assert report["all_structural_preconditions_met"] is False
    assert {v["engine"] for v in report["engines"]} == set(PILOT.OPT_IN_ENGINES)


# --------------------------------------------------------------------------- #
# No-drift contract with the real adapter registry
# --------------------------------------------------------------------------- #
def test_opt_in_engines_match_registry_adapters():
    from nodes import _otr_audio_engines as AE

    for name, spec in PILOT.OPT_IN_ENGINES.items():
        adapter = AE.get_engine(name)  # raises if the name is unregistered
        assert type(adapter).__name__ == spec["adapter_class"]
        assert hasattr(adapter, spec["forward"]), (
            f"{name} adapter is missing its forward {spec['forward']!r}"
        )
        assert adapter.requires_flag == spec["flag"]


def test_pilot_covers_exactly_the_optin_engines():
    from nodes import _otr_audio_engines as AE

    # Every flag-gated (non-default) engine with an EXTERNAL dependency library is
    # covered, and the harness invents none. Legacy default engines
    # (bark/kokoro/musicgen) are byte-identical defaults and excluded; ComfyUI-
    # NATIVE engines (e.g. stable_audio_3) drive ComfyUI's own nodes and have no
    # external lib to probe, so they are excluded from the dep pilot too.
    gated = set()
    for role in ("char_voice", "announcer_voice", "music"):
        for eng in AE.engines_for_role(role):
            adapter = AE.get_engine(eng)
            if getattr(adapter, "requires_flag", None) and not getattr(adapter, "native", False):
                gated.add(eng)
    assert set(PILOT.OPT_IN_ENGINES) == gated


def test_source_is_ascii_no_em_dash():
    src = PILOT_SRC.read_text(encoding="utf-8")
    assert "—" not in src, "em-dash forbidden in OTR python source (CLAUDE.md)"
    src.encode("ascii")  # ASCII-only source


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
