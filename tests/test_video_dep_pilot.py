"""A-S4 -- the video dependency-pilot harness (scripts/otr_video_dep_pilot).

Headless structure tests. The opt-in video engine libraries are absent in the
pytest sandbox, so every probe must report ``lib_absent`` WITHOUT crashing and
WITHOUT importing a banned dependency. Locks the pure helpers, the absent-lib
behavior, and the no-drift contract with the real video registry.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PILOT_SRC = REPO_ROOT / "scripts" / "otr_video_dep_pilot.py"


def _load_pilot():
    spec = importlib.util.spec_from_file_location("otr_video_dep_pilot", PILOT_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PILOT = _load_pilot()


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_snapshot_violations_flags_torch_change_and_banned_deps():
    clean = {"torch": "2.10.0", "xformers": False, "flash_attn": False,
             "sageattention": False}
    assert PILOT.snapshot_violations(clean, clean) == []

    swapped = dict(clean, torch="2.8.0")
    assert any("torch version changed" in v
               for v in PILOT.snapshot_violations(clean, swapped))

    pulled = dict(clean, sageattention=True)
    assert any("sageattention" in v
               for v in PILOT.snapshot_violations(clean, pulled))


def test_dep_snapshot_shape():
    snap = PILOT.dep_snapshot()
    assert set(snap) == {"torch", "xformers", "flash_attn", "sageattention"}
    assert snap["torch"] is not None
    for dep in ("xformers", "flash_attn", "sageattention"):
        assert isinstance(snap[dep], bool)


# --------------------------------------------------------------------------- #
# Probe behavior (headless: libraries absent)
# --------------------------------------------------------------------------- #
def test_probe_one_absent_lib_is_clean_and_imports_no_banned_dep():
    for name in PILOT.PROBE_ENGINES:
        v = PILOT.probe_one(name, do_import=True)
        assert v["engine"] == name
        if PILOT.PROBE_ENGINES[name].get("lib_in_stack"):
            # In-stack rows (spine deps, e.g. soundfile for viz_green; the
            # still_parallax row was REMOVED 2026-06-30, item 2 rip-out) report
            # the truthful lib_in_stack verdict; the real gate is the local
            # model snapshot / ffmpeg via assert_usable.
            assert v["status"] == "lib_in_stack", v
        else:
            # Library absent in the sandbox -> a clean, non-crashing verdict.
            assert v["status"] in {"lib_absent", "import_error"}, v
        assert v["adapter_registered"] is True, "adapter must be registered"
        assert "TODO-for-GPU-smoke" in v["assumed_call"]
        assert v["import_clean_ready"] is False
    # The probes must not have dragged a banned dep into this process.
    for dep in ("xformers", "flash_attn", "sageattention"):
        assert dep not in sys.modules


def test_probe_one_unknown_engine():
    assert PILOT.probe_one("not_a_real_engine")["status"] == "unknown_engine"


def test_probe_one_no_import_reports_structure_only():
    v = PILOT.probe_one("humo", do_import=False)
    assert v["status"] == "not_imported"
    assert v["adapter_registered"] is True
    assert v["forward_present"] is True


def test_run_pilot_in_process_headless_not_ready():
    report = PILOT.run_pilot(isolated=False)
    assert report["engine_count"] == len(PILOT.PROBE_ENGINES)
    assert report["ready_count"] == 0          # libs absent -> nothing import-clean-ready
    assert report["all_imports_clean"] is False
    assert {v["engine"] for v in report["engines"]} == set(PILOT.PROBE_ENGINES)


# --------------------------------------------------------------------------- #
# No-drift contract with the real video registry
# --------------------------------------------------------------------------- #
def test_opt_in_engines_match_registry_adapters():
    from nodes._otr_video_engines.registry import get_engine

    for name, spec in PILOT.PROBE_ENGINES.items():
        adapter = get_engine(name)             # raises if the name is unregistered
        assert type(adapter).__name__ == spec["adapter_class"]
        assert hasattr(adapter, spec["forward"]), (
            f"{name} adapter is missing its forward {spec['forward']!r}"
        )
        # NOTE: the adapter registry no longer carries a flag gate (registry IS
        # the menu). The probe manifest's vestigial "flag" key is retired in the
        # later harness pass; it is NO LONGER asserted against adapter.requires_flag.


def test_pilot_covers_registered_dep_needing_engines():
    """The pilot's probe set is the CURATED list of video engines that need an
    external library / sidecar dep-verified -- it is metadata, NOT derived from a
    flag gate (the registry no longer carries one; registry IS the menu). No-drift
    contract: every probe entry is a registered engine, and the cheap radio-floor
    families (no external lib) are excluded."""
    from nodes._otr_video_engines import registry as vreg

    names = set(vreg.all_engine_names())
    assert set(PILOT.PROBE_ENGINES) <= names          # every probe entry is registered
    assert "still_motion" not in PILOT.PROBE_ENGINES  # cheap floor: no lib to probe
    assert "humo" in PILOT.PROBE_ENGINES                # a real dep-needing engine


def test_new_source_is_ascii_no_em_dash():
    # The worker + install scripts (scripts/_*) are gitignored by repo
    # convention (same as the chatterbox / dia sidecars) so they live only in
    # the working tree; check them WHEN PRESENT but never fail a clean checkout.
    for src_path in (
        PILOT_SRC,
        REPO_ROOT / "nodes" / "_otr_shared" / "sidecar.py",
    ):
        if not src_path.exists():
            continue
        src = src_path.read_text(encoding="utf-8")
        assert "—" not in src, f"em-dash forbidden in {src_path.name} (CLAUDE.md)"
        src.encode("ascii")                    # ASCII-only source


# --------------------------------------------------------------------------- #
# Custom-node startup-contamination scan (KJNodes gate F + torch.hub ban)
# --------------------------------------------------------------------------- #
def test_scan_source_detects_module_scope_contaminant():
    src = "import os\nimport sageattention\nfrom xformers import ops\n"
    f = PILOT.scan_source_for_contaminants(src)
    assert f["module_scope_imports"] == ["sageattention", "xformers"]
    assert f["torch_hub_load"] is False and f["parse_error"] is False


def test_scan_source_ignores_function_scope_import():
    # A lazy import inside a function is NOT a startup contaminant.
    src = "def f():\n    import sageattention\n    return sageattention\n"
    f = PILOT.scan_source_for_contaminants(src)
    assert f["module_scope_imports"] == []


def test_scan_source_detects_torch_hub_load():
    f = PILOT.scan_source_for_contaminants("import torch\nm = torch.hub.load('a/b', 'c')\n")
    assert f["torch_hub_load"] is True


def test_scan_source_parse_error_recorded_not_raised():
    f = PILOT.scan_source_for_contaminants("def (:\n")
    assert f["parse_error"] is True and f["module_scope_imports"] == []


def test_scan_custom_nodes_flags_contaminated_node(tmp_path):
    (tmp_path / "CleanNode").mkdir()
    (tmp_path / "CleanNode" / "__init__.py").write_text("import os\n", encoding="utf-8")
    (tmp_path / "BadNode").mkdir()
    (tmp_path / "BadNode" / "__init__.py").write_text(
        "import sageattention\n", encoding="utf-8")
    report = PILOT.scan_custom_nodes(str(tmp_path))
    assert report["scanned_dir_present"] is True
    assert report["startup_audit_clean"] is False
    assert {n["node"] for n in report["flagged_nodes"]} == {"BadNode"}


def test_scan_custom_nodes_headless_missing_dir_is_clean(tmp_path):
    report = PILOT.scan_custom_nodes(str(tmp_path / "does_not_exist"))
    assert report["scanned_dir_present"] is False
    assert report["startup_audit_clean"] is True and report["flagged_nodes"] == []


def test_run_pilot_report_includes_custom_node_scan():
    report = PILOT.run_pilot(isolated=False)
    assert "custom_node_scan" in report
    scan = report["custom_node_scan"]
    assert set(scan) >= {"custom_nodes_dir", "scanned_dir_present",
                         "flagged_nodes", "startup_audit_clean",
                         "startup_contaminants"}
    assert report["startup_audit_clean"] in (True, False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
