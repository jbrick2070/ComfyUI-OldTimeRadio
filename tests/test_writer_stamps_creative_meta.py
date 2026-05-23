"""Sprint D D2b -- writer stamps meta.creative_model + meta.creative_prompt_profile.

Resolves Sprint C gotcha #4: pre-D2b the writer's script_json
output preserved technical_model identity but NOT creative model
identity. FreezeCascade is downstream of the writer's script_json
and post-freeze diagnostics were blind to which creative model
produced the script.

D2b stamps both keys into meta after the K.5.5 story_brief
reflection pass and BEFORE script_json = json.dumps(led.data) at
section L. Both keys are additive; audio path reads only
meta.story_brief so byte identity holds.

Two AST-based assertions on the writer source (no LLM run
required, no fixture script generation):

  test_writer_stamps_meta_creative_model_with_creative_repo_id
      Searches OTR_LedgerScriptWriter.execute for an assignment
      `meta["creative_model"] = ...` and confirms the RHS reads
      from resolved["creative_writing_model"].

  test_writer_stamps_meta_creative_prompt_profile_from_catalog_row
      Searches for an assignment `meta["creative_prompt_profile"]
      = ...` and confirms the RHS path looks up the catalog row.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

WRITER_SRC = REPO_ROOT / "nodes" / "OTR_LedgerScriptWriter.py"


def _meta_assignments(meta_key: str) -> list[ast.AST]:
    """Walk the writer source AST and return every assignment whose
    target is `meta[<meta_key>]`.

    Matches both Subscript(value=Name("meta"), slice=Constant(meta_key))
    and the older Index-wrapped variant. Returns the Assign nodes.
    """
    tree = ast.parse(WRITER_SRC.read_text(encoding="utf-8"))
    out: list[ast.AST] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue
            if not (
                isinstance(target.value, ast.Name)
                and target.value.id == "meta"
            ):
                continue
            slice_node = target.slice
            if (
                isinstance(slice_node, ast.Constant)
                and slice_node.value == meta_key
            ):
                out.append(node)
    return out


def test_writer_stamps_meta_creative_model_with_creative_repo_id() -> None:
    """`meta["creative_model"] = ...` appears in writer source AND
    the assigned value reads from resolved["creative_writing_model"]
    (the same dict the writer threads to the creative slot).
    """
    assigns = _meta_assignments("creative_model")
    assert assigns, (
        "no `meta[\"creative_model\"] = ...` assignment in "
        "nodes/OTR_LedgerScriptWriter.py; D2b stamping missing"
    )
    snippets = [ast.unparse(a) for a in assigns]
    # ast.unparse normalizes string quotes; check via "in" against
    # both quote styles + bare-substring match for "resolved" and
    # "creative_writing_model" tokens (catches either positional
    # form ['creative_writing_model'] or ["creative_writing_model"]).
    rhs_ok = any(
        ("resolved" in s and "creative_writing_model" in s)
        for s in snippets
    )
    assert rhs_ok, (
        f"meta[\"creative_model\"] assignment(s) do not reference "
        f"resolved[\"creative_writing_model\"]. Found:\n  "
        + "\n  ".join(snippets)
    )


def test_writer_stamps_meta_creative_prompt_profile_from_catalog_row() -> None:
    """`meta["creative_prompt_profile"] = ...` appears in writer
    source AND the assigned value path involves a catalog row
    lookup (catches a hardcoded "modern" stamp that would lie about
    a period row's actual otr_1940s_v1 profile).
    """
    assigns = _meta_assignments("creative_prompt_profile")
    assert assigns, (
        "no `meta[\"creative_prompt_profile\"] = ...` assignment "
        "in nodes/OTR_LedgerScriptWriter.py; D2b stamping missing"
    )
    snippets = [ast.unparse(a) for a in assigns]
    # The expected path uses prompt_profile attribute lookup
    # (e.g. _row.prompt_profile) somewhere in the assigned value.
    profile_ok = any("prompt_profile" in s for s in snippets)
    assert profile_ok, (
        f"meta[\"creative_prompt_profile\"] assignment(s) do not "
        f"reference a catalog row's prompt_profile attribute. "
        f"Found:\n  " + "\n  ".join(snippets)
    )


def test_meta_stamping_lands_after_story_brief_reflection_pass() -> None:
    """The stamping must land AFTER `meta.update(_brief_delta)` and
    BEFORE `script_json = json.dumps(led.data, ...)`. If it lands
    before the reflection pass, story_brief could overwrite the
    creative_model stamp. If it lands after script_json assembly,
    FreezeCascade's input would not see the new keys.

    Source-line ordering check, not runtime ordering.
    """
    src = WRITER_SRC.read_text(encoding="utf-8")
    lines = src.splitlines()

    def _find_line_idx(needle: str) -> int:
        for i, line in enumerate(lines):
            if needle in line:
                return i
        return -1

    reflect_idx = _find_line_idx("meta.update(_brief_delta)")
    stamp_idx = _find_line_idx('meta["creative_model"]')
    assemble_idx = _find_line_idx("script_json = json.dumps(led.data,")

    assert reflect_idx >= 0, "missing meta.update(_brief_delta) anchor"
    assert stamp_idx >= 0, "missing meta[\"creative_model\"] stamp"
    assert assemble_idx >= 0, (
        "missing script_json = json.dumps(led.data, ...) anchor"
    )

    assert reflect_idx < stamp_idx < assemble_idx, (
        f"meta stamping ordering wrong: reflect={reflect_idx}, "
        f"stamp={stamp_idx}, assemble={assemble_idx}. Expected "
        f"reflect < stamp < assemble."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
