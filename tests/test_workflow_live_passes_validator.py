"""S16.6: the production workflow JSON must pass the full validator
in default mode. Strict-unknown-types mode is exercised at
production-loader time (S14.2.1) where every OTR class is
registered; the bare test env has known import-skip cases for
heavy optional deps (HuMo / LTX / Upscale) that legitimately can't
import.

This test is the cumulative gate for S16.1 (widget-name scrub),
S16.2 (extended check 5), S16.3 (positional widget-drift), S16.4
(FluxPortrait.ledger_json wired), and S16.5 (link-tuple + dup-dedup).
Any of those regressing will fire here.
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from nodes._workflow_validation import (  # noqa: E402
    validate_workflow_contract,
)


CANONICAL_WORKFLOW = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"


def _otr_mappings_via_existing_test_helper() -> dict:
    """Reuse the AST-walk + importlib helper from
    test_workflow_contract_validation.py so we don't duplicate the
    parsing logic. Failing imports are silently skipped -- that's
    fine for default mode; strict mode is exercised elsewhere.
    """
    import test_workflow_contract_validation as twv
    return twv._otr_node_class_mappings()


def test_production_workflow_passes_default_validation():
    wf = json.loads(CANONICAL_WORKFLOW.read_text(encoding="utf-8"))
    mappings = _otr_mappings_via_existing_test_helper()
    # Default mode: strict_unknown_types=False so test-env class-
    # skip doesn't fire. The S14.2.1 production loader path enables
    # strict mode where every class IS registered.
    validate_workflow_contract(wf, mappings)
