"""Sprint D D1a -- CuratedModel schema extension + workflow guardrail.

Four assertions per v3 plan:

  test_curated_model_has_extended_fields
      The dataclass carries the 6 new D1a fields.

  test_existing_rows_default_to_prompt_profile_modern
      Every pre-D1a row backfills with prompt_profile = "modern".

  test_no_curated_row_uses_otr_1940s_v1_profile
      No curated row uses the period profile (the talkie row was
      removed 2026-05-22; the period-routing surface stays parked).

  test_default_workflow_only_binds_mit_equivalent_rows_to_creative_slot
      The shipped default workflow JSON's writer-node widgets_values
      MUST NOT bind any catalog row with license_audit_status != "mit_equivalent"
      to either writer slot. Pinned at D1a; D3 lands a runtime
      OTR_WorkflowValidator hard-fail with the same shape.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_model_catalog as catalog  # noqa: E402

WORKFLOW_PATH = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"

D1A_NEW_FIELDS = (
    "prompt_profile",
    "chat_template_kind",
    "stop_tokens",
    "context_window",
    "license",
    "license_audit_status",
)


def test_curated_model_has_extended_fields() -> None:
    field_names = {f.name for f in dataclasses.fields(catalog.CuratedModel)}
    missing = [n for n in D1A_NEW_FIELDS if n not in field_names]
    assert not missing, (
        f"CuratedModel missing D1a fields: {missing}. "
        f"Present fields: {sorted(field_names)}"
    )


def test_existing_rows_default_to_prompt_profile_modern() -> None:
    """Every row whose repo_id was in the pre-D1a catalog MUST carry
    `prompt_profile = "modern"`. Catches accidental flips of a modern
    model into the period-routing slot.
    """
    # 2026-05-23: catalog pruned -- the two WARN-tier 12B community
    # rows (Captain-Eris_Violet-V0.420-12B, MN-12B-Mag-Mell-R1) were
    # removed. These four are the survivors.
    PRE_D1A_REPO_IDS = {
        "mistralai/Mistral-Nemo-Instruct-2407",
        "google/gemma-4-E2B-it",
        "google/gemma-4-E4B-it",
        "Qwen/Qwen2.5-14B-Instruct",
    }
    rows_by_id = {m.repo_id: m for m in catalog.CURATED_LLM_MODELS}
    failures: list[str] = []
    for repo_id in PRE_D1A_REPO_IDS:
        row = rows_by_id.get(repo_id)
        if row is None:
            failures.append(f"pre-D1a row missing from catalog: {repo_id!r}")
            continue
        if row.prompt_profile != "modern":
            failures.append(
                f"{repo_id!r} has prompt_profile = {row.prompt_profile!r}; "
                f"expected 'modern'"
            )
    assert not failures, "\n  " + "\n  ".join(failures)


def test_no_curated_row_uses_otr_1940s_v1_profile() -> None:
    """No curated row carries prompt_profile = "otr_1940s_v1" at
    present. The broken talkie-lm/talkie-1930-13b-it row (the only
    period row ever curated) was removed 2026-05-22. The period-
    routing surface stays parked; this test pins the empty state so
    a future period model is added deliberately, not silently.
    """
    period_rows = [
        m.repo_id
        for m in catalog.CURATED_LLM_MODELS
        if m.prompt_profile == "otr_1940s_v1"
    ]
    assert period_rows == [], (
        f"unexpected curated otr_1940s_v1 period row(s): {period_rows}. "
        f"If a period model was added deliberately, update this test "
        f"to assert its expected schema."
    )


def _writer_creative_and_technical_slot_values(
    workflow: dict,
) -> list[tuple[int, str]]:
    """Walk the default workflow JSON's OTR_LedgerScriptWriter nodes
    and return [(node_id, value)] pairs for every widget value that
    looks like a curated repo_id."""
    curated_ids = {m.repo_id for m in catalog.CURATED_LLM_MODELS}
    out: list[tuple[int, str]] = []
    for n in workflow["nodes"]:
        if n.get("type") != "OTR_LedgerScriptWriter":
            continue
        for v in n.get("widgets_values") or []:
            if isinstance(v, str) and v in curated_ids:
                out.append((n["id"], v))
    return out


def test_default_workflow_only_binds_mit_equivalent_rows_to_creative_slot() -> None:
    """The default-shipped workflow JSON's writer node widgets MUST NOT
    bind any catalog row with license_audit_status != "mit_equivalent"
    to either slot. Pinned at D1a; D3 lands an OTR_WorkflowValidator
    runtime hard-fail with the same contract.
    """
    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    rows_by_id = {m.repo_id: m for m in catalog.CURATED_LLM_MODELS}
    bindings = _writer_creative_and_technical_slot_values(workflow)
    assert bindings, (
        "default workflow has no OTR_LedgerScriptWriter widget value "
        "matching a curated repo_id; cannot verify the guardrail "
        "applies to anything"
    )
    failures: list[str] = []
    for node_id, repo_id in bindings:
        row = rows_by_id[repo_id]
        if row.license_audit_status != "mit_equivalent":
            failures.append(
                f"writer node id={node_id} binds {repo_id!r} "
                f"(license_audit_status={row.license_audit_status!r}); "
                f"expected 'mit_equivalent' for any default-shipped "
                f"workflow binding"
            )
    assert not failures, "\n  " + "\n  ".join(failures)


def test_d1a_loader_backend_literal_includes_gptq_int4() -> None:
    """The dataclass Literal must accept the new GPTQ adapter key.
    Catches accidental removal of transformers_gptq_int4 from the
    catalog's accepted backend set.
    """
    src = (REPO_ROOT / "nodes" / "_otr_model_catalog.py").read_text(
        encoding="utf-8",
    )
    assert '"transformers_gptq_int4"' in src, (
        "transformers_gptq_int4 missing from CuratedModel.loader_backend "
        "Literal in nodes/_otr_model_catalog.py"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
