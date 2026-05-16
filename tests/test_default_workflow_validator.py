"""Sprint D D3 -- default-workflow creative-binding hard-fail validator.

Two assertions:

  test_default_workflow_validator_passes_on_shipped_default
      The actual default-shipped workflow JSON
      (`workflows/otr_scifi_16gb_full.json`) MUST pass the
      check_default_workflow_creative_binding gate. Both writer
      slots bind to Mistral-Nemo (mit_equivalent, modern) so the
      validator returns zero violations.

  test_default_workflow_validator_hard_fails_on_non_mit_equivalent_creative_binding
      Inject a fake workflow that binds talkie (research_lane) to
      the creative slot. The validator returns a violation that
      names the offending repo_id and the disallowed
      license_audit_status.
"""
from __future__ import annotations

import json
import sys
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
    """Inject a fake workflow JSON that binds talkie (research_lane,
    otr_1940s_v1) to a writer slot. The validator returns at least
    one violation for the license_audit_status AND one for the
    prompt_profile.
    """
    fake_workflow = {
        "nodes": [
            {
                "id": 999,
                "type": "OTR_LedgerScriptWriter",
                "widgets_values": [
                    "",                                  # episode_title
                    350,                                 # target_words
                    2,                                   # num_characters
                    42,                                  # seed
                    "talkie-lm/talkie-1930-13b-it",      # creative -> talkie
                    "mistralai/Mistral-Nemo-Instruct-2407",  # technical -> nemo
                ],
            },
        ],
    }
    violations = check_default_workflow_creative_binding(fake_workflow, catalog)
    assert violations, (
        "fake workflow binding talkie to writer slot produced ZERO "
        "violations; validator failed to catch the research_lane "
        "binding"
    )
    # The talkie binding should trip BOTH the license_audit_status
    # check AND the prompt_profile check.
    license_violation = any(
        "license_audit_status" in v and "talkie" in v.lower()
        for v in violations
    )
    profile_violation = any(
        "prompt_profile" in v and "otr_1940s_v1" in v
        for v in violations
    )
    assert license_violation, (
        f"validator did not flag talkie's license_audit_status "
        f"violation. Got:\n  " + "\n  ".join(violations)
    )
    assert profile_violation, (
        f"validator did not flag talkie's prompt_profile "
        f"violation. Got:\n  " + "\n  ".join(violations)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
