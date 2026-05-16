"""Sprint D D0d -- workflow JSON wiring invariants.

Five structural assertions on workflows/otr_scifi_16gb_full.json that
pin the D0d rewires against regression:

  test_no_ledger_json_input_sources_from_signal_lost_video_path_output
      No HuMo / VideoComposite / LTX ledger_json input may source from
      SignalLostVideo.video_path (a video file path string, not a frozen
      ledger). Pinned against the Sprint C miswire class.

  test_all_ledger_json_inputs_source_from_freeze_cascade_script_json
      Every node with a ledger_json input must source from
      FreezeCascade.script_json. Catches missed rewires + future
      regressions.

  test_audio_gate_does_not_source_from_signal_lost_video_path
      VideoPlan.audio_gate must NOT source from SignalLostVideo.video_path
      (which forced FLUX planning to wait on 1080p procgen).

  test_humo_portraits_dir_is_linked_to_portrait_render_output
      HuMo.portraits_dir must be linked to BatchFluxPortraitRender's
      portraits_dir output (face reference is content, not gate).

  test_otr_workflow_validator_class_has_output_node_true
      OTR_WorkflowValidator must carry OUTPUT_NODE=True so ComfyUI
      executes it. Pin against accidental removal.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

WORKFLOW_PATH = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"
VALIDATOR_SRC = REPO_ROOT / "nodes" / "_otr_workflow_validator.py"
PORTRAIT_SRC = REPO_ROOT / "visual" / "batch_flux_portrait_render.py"


def _ast_extract_class_attr_tuple(
    src_path: Path, class_name: str, attr_name: str,
) -> tuple[str, ...] | None:
    """Return the tuple-of-strings value for a class-level attribute.

    AST-parse the source file (no execution), find the named class,
    locate the attr assignment, and return its tuple-of-strings value.
    Avoids importing the module (which would pull in ComfyUI runtime
    dependencies like `folder_paths` that don't resolve under pytest).
    """
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == attr_name:
                    if isinstance(stmt.value, ast.Tuple):
                        out: list[str] = []
                        for elt in stmt.value.elts:
                            if (
                                isinstance(elt, ast.Constant)
                                and isinstance(elt.value, str)
                            ):
                                out.append(elt.value)
                        return tuple(out)
    return None


def _ast_extract_class_attr_bool(
    src_path: Path, class_name: str, attr_name: str,
) -> bool | None:
    """Return the boolean value for a class-level attribute. AST-only."""
    tree = ast.parse(src_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == attr_name:
                    if (
                        isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, bool)
                    ):
                        return stmt.value.value
    return None


@pytest.fixture(scope="module")
def workflow() -> dict:
    return json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def nodes_by_id(workflow: dict) -> dict[int, dict]:
    return {n["id"]: n for n in workflow["nodes"]}


@pytest.fixture(scope="module")
def links_by_id(workflow: dict) -> dict[int, list]:
    return {lk[0]: lk for lk in workflow["links"]}


def _find_input_link_id(node: dict, input_name: str) -> int | None:
    for inp in node.get("inputs") or []:
        if inp.get("name") == input_name:
            return inp.get("link")
    raise AssertionError(
        f"node id={node['id']} type={node.get('type')} has no input "
        f"named {input_name!r}"
    )


def _resolve_link_source(
    link_id: int, links_by_id: dict, nodes_by_id: dict,
) -> tuple[str, str]:
    """Return (source_node_type, source_output_name) for a given link id."""
    lk = links_by_id[link_id]
    src_node_id, src_slot = lk[1], lk[2]
    src_node = nodes_by_id[src_node_id]
    src_outputs = src_node.get("outputs") or []
    src_out_name = (
        src_outputs[src_slot].get("name", "?")
        if src_slot < len(src_outputs) else "?"
    )
    return src_node.get("type", "?"), src_out_name


def test_no_ledger_json_input_sources_from_signal_lost_video_path_output(
    workflow: dict, nodes_by_id: dict, links_by_id: dict,
) -> None:
    """The Sprint C miswire: HuMo / VideoComposite / LTX ledger_json
    were sourced from SignalLostVideo.video_path. D0d rewires them
    away. This test pins the rewire and catches the regression class.
    """
    violations: list[str] = []
    for n in workflow["nodes"]:
        for inp in n.get("inputs") or []:
            if inp.get("name") != "ledger_json":
                continue
            link_id = inp.get("link")
            if link_id is None:
                continue
            src_type, src_out = _resolve_link_source(
                link_id, links_by_id, nodes_by_id,
            )
            if src_type == "OTR_SignalLostVideo" and src_out == "video_path":
                violations.append(
                    f"node id={n['id']} type={n.get('type')} "
                    f"ledger_json link#{link_id} sources from "
                    f"{src_type}.{src_out} (D0d miswire regression)"
                )
    assert not violations, (
        "ledger_json inputs miswired to SignalLostVideo.video_path:\n  "
        + "\n  ".join(violations)
    )


def test_all_ledger_json_inputs_source_from_freeze_cascade_script_json(
    workflow: dict, nodes_by_id: dict, links_by_id: dict,
) -> None:
    """Every node with a ledger_json input must source from
    FreezeCascade.script_json. Catches missed rewires.
    """
    failures: list[str] = []
    seen_ledger_json_inputs = 0
    for n in workflow["nodes"]:
        for inp in n.get("inputs") or []:
            if inp.get("name") != "ledger_json":
                continue
            seen_ledger_json_inputs += 1
            link_id = inp.get("link")
            if link_id is None:
                failures.append(
                    f"node id={n['id']} type={n.get('type')} "
                    f"ledger_json is unlinked"
                )
                continue
            src_type, src_out = _resolve_link_source(
                link_id, links_by_id, nodes_by_id,
            )
            if not (
                src_type == "OTR_LedgerFreezeCascade"
                and src_out == "script_json"
            ):
                failures.append(
                    f"node id={n['id']} type={n.get('type')} "
                    f"ledger_json link#{link_id} sources from "
                    f"{src_type}.{src_out}; expected "
                    f"OTR_LedgerFreezeCascade.script_json"
                )
    assert seen_ledger_json_inputs >= 4, (
        f"expected >=4 ledger_json inputs in default workflow; "
        f"found {seen_ledger_json_inputs}"
    )
    assert not failures, (
        "ledger_json wiring violations:\n  " + "\n  ".join(failures)
    )


def test_audio_gate_does_not_source_from_signal_lost_video_path(
    workflow: dict, nodes_by_id: dict, links_by_id: dict,
) -> None:
    """VideoPlan.audio_gate must NOT source from SignalLostVideo.video_path.
    Pre-D0d wiring forced FLUX planning to wait on 1080p procgen video
    completion. D0d sources from FreezeCascade.script_json (cheap and
    semantically correct as a completion signal).
    """
    violations: list[str] = []
    for n in workflow["nodes"]:
        for inp in n.get("inputs") or []:
            if inp.get("name") != "audio_gate":
                continue
            link_id = inp.get("link")
            if link_id is None:
                continue
            src_type, src_out = _resolve_link_source(
                link_id, links_by_id, nodes_by_id,
            )
            if src_type == "OTR_SignalLostVideo" and src_out == "video_path":
                violations.append(
                    f"node id={n['id']} type={n.get('type')} "
                    f"audio_gate link#{link_id} sources from "
                    f"{src_type}.{src_out} (D0d miswire regression)"
                )
    assert not violations, (
        "audio_gate inputs miswired to SignalLostVideo.video_path:\n  "
        + "\n  ".join(violations)
    )


def test_humo_portraits_dir_is_linked_to_portrait_render_output(
    workflow: dict, nodes_by_id: dict, links_by_id: dict,
) -> None:
    """HuMo.portraits_dir must be linked to BatchFluxPortraitRender's
    portraits_dir output (per G1 operator decision: portraits are
    content, not gate).
    """
    humo_nodes = [
        n for n in workflow["nodes"]
        if n.get("type") == "OTR_BatchHumoRender"
    ]
    assert humo_nodes, "default workflow has no OTR_BatchHumoRender node"

    for humo in humo_nodes:
        link_id = _find_input_link_id(humo, "portraits_dir")
        assert link_id is not None, (
            f"HuMo node id={humo['id']} portraits_dir is unlinked; "
            f"D0d requires wiring to BatchFluxPortraitRender.portraits_dir"
        )
        src_type, src_out = _resolve_link_source(
            link_id, links_by_id, nodes_by_id,
        )
        assert (
            src_type == "OTR_BatchFluxPortraitRender"
            and src_out == "portraits_dir"
        ), (
            f"HuMo node id={humo['id']} portraits_dir link#{link_id} "
            f"sources from {src_type}.{src_out}; expected "
            f"OTR_BatchFluxPortraitRender.portraits_dir"
        )


def test_otr_workflow_validator_class_has_output_node_true() -> None:
    """OTR_WorkflowValidator must carry OUTPUT_NODE=True so ComfyUI
    executes the node even without a downstream consumer of its
    validation_report output.

    The source-side class is named `WorkflowValidator`; it is
    registered as `OTR_WorkflowValidator` via NODE_CLASS_MAPPINGS in
    the same file. AST-extract the class-level OUTPUT_NODE assignment
    to avoid pulling in ComfyUI runtime imports under pytest.
    """
    value = _ast_extract_class_attr_bool(
        VALIDATOR_SRC, "WorkflowValidator", "OUTPUT_NODE",
    )
    assert value is True, (
        f"WorkflowValidator.OUTPUT_NODE is {value!r}; expected True so "
        f"ComfyUI executes the validator at workflow run time. Source: "
        f"{VALIDATOR_SRC.relative_to(REPO_ROOT)}"
    )

    # Cross-check: the OTR_-prefixed registration name is what the
    # workflow JSON binds against. NODE_CLASS_MAPPINGS must expose
    # WorkflowValidator under the OTR_WorkflowValidator key.
    src = VALIDATOR_SRC.read_text(encoding="utf-8")
    assert "OTR_WorkflowValidator" in src and "WorkflowValidator" in src, (
        "NODE_CLASS_MAPPINGS registration missing the OTR_WorkflowValidator "
        "alias for WorkflowValidator class"
    )


def test_batch_flux_portrait_render_emits_portraits_dir_output() -> None:
    """The D0d wiring partner: BatchFluxPortraitRender must return a
    third STRING value named `portraits_dir`. Without this output the
    HuMo.portraits_dir link cannot land in the JSON.

    AST-extract the class-level RETURN_TYPES + RETURN_NAMES to avoid
    pulling in `folder_paths` (ComfyUI runtime module) at import time.
    """
    return_types = _ast_extract_class_attr_tuple(
        PORTRAIT_SRC, "BatchFluxPortraitRender", "RETURN_TYPES",
    )
    return_names = _ast_extract_class_attr_tuple(
        PORTRAIT_SRC, "BatchFluxPortraitRender", "RETURN_NAMES",
    )
    assert return_types == ("IMAGE", "STRING", "STRING"), (
        f"BatchFluxPortraitRender.RETURN_TYPES is {return_types!r}; "
        f"expected ('IMAGE', 'STRING', 'STRING'). Source: "
        f"{PORTRAIT_SRC.relative_to(REPO_ROOT)}"
    )
    assert return_names == ("portrait_batch", "report", "portraits_dir"), (
        f"BatchFluxPortraitRender.RETURN_NAMES is {return_names!r}; "
        f"expected ('portrait_batch', 'report', 'portraits_dir'). Source: "
        f"{PORTRAIT_SRC.relative_to(REPO_ROOT)}"
    )


def test_workflow_link_records_match_node_input_link_ids(
    workflow: dict, nodes_by_id: dict, links_by_id: dict,
) -> None:
    """Sanity: every input.link referenced by a node must exist in
    workflow['links'], and every link in workflow['links'] must have
    consistent src/dst node ids.
    """
    for n in workflow["nodes"]:
        for inp in n.get("inputs") or []:
            link_id = inp.get("link")
            if link_id is None:
                continue
            assert link_id in links_by_id, (
                f"node id={n['id']} input {inp.get('name')!r} references "
                f"link#{link_id} which is not in workflow['links']"
            )
            lk = links_by_id[link_id]
            assert lk[3] == n["id"], (
                f"link#{link_id} dest_node {lk[3]} != node id={n['id']}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
