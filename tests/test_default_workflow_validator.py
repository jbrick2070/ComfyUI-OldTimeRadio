"""Sprint D D3 -- default-workflow creative-binding hard-fail validator.

Two assertions:

  test_default_workflow_validator_passes_on_shipped_default
      The actual default-shipped workflow JSON
      (`workflows/otr_scifi_16gb_full.json`) MUST pass the
      check_default_workflow_creative_binding gate. Both writer
      slots bind to Mistral-Nemo (mit_equivalent, modern) so the
      validator returns zero violations.

  test_default_workflow_validator_hard_fails_on_non_mit_equivalent_creative_binding
      Inject a fake workflow that binds a synthetic research_lane +
      otr_1940s_v1 row to the creative slot, plus a fake catalog
      module carrying that row. The validator returns a violation
      that names the offending repo_id and both the disallowed
      license_audit_status and the disallowed prompt_profile.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_model_catalog as catalog  # noqa: E402
from tools.audit_workflow_schema import (  # noqa: E402
    check_default_workflow_creative_binding,
)


WORKFLOW_PATH = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"


def test_default_workflow_validator_passes_on_shipped_default() -> None:
    """The shipped default workflow MUST pass. Both writer slots
    bind to Mistral-Nemo which is mit_equivalent + modern.
    """
    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    violations = check_default_workflow_creative_binding(workflow, catalog)
    assert violations == [], (
        "default workflow JSON violates creative-binding gate at "
        "D3 landing:\n  " + "\n  ".join(violations)
    )


def test_default_workflow_validator_hard_fails_on_non_mit_equivalent_creative_binding() -> None:
    """Inject a fake workflow JSON that binds a synthetic
    research_lane + otr_1940s_v1 row to a writer slot, plus a fake
    catalog module carrying only that row. The validator returns at
    least one violation for the license_audit_status AND one for the
    prompt_profile.

    The synthetic row stands in for the talkie-lm/talkie-1930-13b-it
    row removed 2026-05-22: with no curated non-mit_equivalent period
    row to point at, the validator's two-contract logic is exercised
    via the catalog_module injection seam the function already
    exposes for unit tests.
    """
    synthetic_repo_id = "period-lm/synthetic-period-13b"
    synthetic_row = catalog.CuratedModel(
        repo_id=synthetic_repo_id,
        requires_auth=True,
        loader_backend="transformers_gptq_int4",
        vram_fit_tier="UNKNOWN",
        approx_safetensors_gb=7.5,
        prompt_profile="otr_1940s_v1",
        license="non_commercial",
        license_audit_status="research_lane",
    )
    fake_catalog = types.SimpleNamespace(
        CURATED_LLM_MODELS=(synthetic_row,),
    )
    fake_workflow = {
        "nodes": [
            {
                "id": 999,
                "type": "OTR_LedgerScriptWriter",
                "widgets_values": [
                    "",                  # episode_title
                    350,                 # target_words
                    2,                   # num_characters
                    42,                  # seed
                    synthetic_repo_id,   # creative -> synthetic period row
                    synthetic_repo_id,   # technical (same row, also flagged)
                ],
            },
        ],
    }
    violations = check_default_workflow_creative_binding(
        fake_workflow, fake_catalog,
    )
    assert violations, (
        "fake workflow binding a research_lane otr_1940s_v1 row to a "
        "writer slot produced ZERO violations; validator failed to "
        "catch the binding"
    )
    # The synthetic binding should trip BOTH the license_audit_status
    # check AND the prompt_profile check.
    license_violation = any(
        "license_audit_status" in v and synthetic_repo_id in v
        for v in violations
    )
    profile_violation = any(
        "prompt_profile" in v and "otr_1940s_v1" in v
        for v in violations
    )
    assert license_violation, (
        f"validator did not flag the row's license_audit_status "
        f"violation. Got:\n  " + "\n  ".join(violations)
    )
    assert profile_violation, (
        f"validator did not flag the row's prompt_profile "
        f"violation. Got:\n  " + "\n  ".join(violations)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
