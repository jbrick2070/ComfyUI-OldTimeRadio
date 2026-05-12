"""tests/test_lfc_wiring_smoke_script.py

LFC commit 12.7 -- pytest wrapper around scripts/lfc_wiring_smoke.py.
Runs the smoke script's main() and asserts exit code 0 so the
wiring-invariant gate runs in the same `pytest` invocation as the
rest of the LFC suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Make scripts/ importable so we can call main() in-process.
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_lfc_wiring_smoke_passes():
    """scripts/lfc_wiring_smoke.py main([]) returns 0."""
    import importlib
    smoke = importlib.import_module("lfc_wiring_smoke")
    rc = smoke.main([])
    assert rc == 0, (
        "scripts/lfc_wiring_smoke.py failed at HEAD. Run "
        "`python scripts/lfc_wiring_smoke.py` for the full failure "
        "details."
    )


def test_lfc_wiring_smoke_strict_passes():
    """--strict mode also passes -- no warnings in the baseline."""
    import importlib
    smoke = importlib.import_module("lfc_wiring_smoke")
    rc = smoke.main(["--strict"])
    assert rc == 0


def test_smoke_module_surface():
    """Sanity check the smoke script exposes the expected public
    constants -- they're effectively a second source of truth for
    the cascade widget contract, so a future drift between this
    file and OTR_LedgerFreezeCascade.INPUT_TYPES surfaces here."""
    import importlib
    smoke = importlib.import_module("lfc_wiring_smoke")
    assert len(smoke.EXPECTED_WIDGET_VALUES) == 10
    assert smoke.EXPECTED_RETURN_NAMES == (
        "script_text", "script_json", "news_used",
        "estimated_minutes", "freeze_verdict",
    )
    assert smoke.CASCADE_NODE_ID == 62
    assert "OTR_LedgerScriptReviewer" in smoke.LEGACY_TOKENS
    assert "OTR_Gemma4Director" in smoke.LEGACY_TOKENS
    # D4 extension (commit 12.17): extended legacy class set.
    assert "OTR_LLMScriptWriter" in smoke.LEGACY_TOKENS
    assert "OTR_Gemma4ScriptWriter" in smoke.LEGACY_TOKENS
    assert "_RENAME_ALIASES" in smoke.LEGACY_TOKENS
    # Workflow-JSON class_type scan list exists + is non-empty.
    assert len(smoke.WORKFLOW_LEGACY_CLASS_TYPES) >= 4
    assert "OTR_LedgerScriptReviewer" in smoke.WORKFLOW_LEGACY_CLASS_TYPES


def test_workflow_legacy_class_scan_catches_planted_node(tmp_path,
                                                          monkeypatch):
    """D4 extension (commit 12.17): plant a synthetic workflow file
    with a legacy class_type and confirm the scanner surfaces it."""
    import importlib
    smoke = importlib.import_module("lfc_wiring_smoke")

    # Point ROOT at a tmpdir containing a planted workflow file.
    fake_root = tmp_path
    wf_dir = fake_root / "workflows"
    wf_dir.mkdir()
    planted_ui = {
        "nodes": [
            {"id": 1, "type": "OTR_LedgerScriptReviewer"},
            {"id": 2, "type": "OTR_LedgerScriptWriter"},  # current name
        ],
        "links": [],
    }
    (wf_dir / "planted_ui.json").write_text(
        __import__("json").dumps(planted_ui), encoding="utf-8",
    )
    planted_api = {
        "1": {"class_type": "OTR_LLMScriptWriter", "inputs": {}},
    }
    (wf_dir / "planted_api.json").write_text(
        __import__("json").dumps(planted_api), encoding="utf-8",
    )

    monkeypatch.setattr(smoke, "ROOT", fake_root)
    hits = smoke._scan_workflow_class_types()
    hit_classes = {ctype for (_rel, _nid, _f, ctype) in hits}
    assert "OTR_LedgerScriptReviewer" in hit_classes
    assert "OTR_LLMScriptWriter" in hit_classes
    # Current class name must NOT be flagged.
    assert "OTR_LedgerScriptWriter" not in hit_classes
