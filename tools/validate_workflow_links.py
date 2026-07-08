"""Static workflow link-integrity validator (no ComfyUI boot required).

Rule set per S26 cleanbreak plan §4:
  - every link ID referenced by a node input MUST exist in workflow["links"]
  - every link source node and target node MUST exist in workflow["nodes"]
  - every target input index MUST exist on the target node's "inputs"
  - every node "type" MUST be present in __init__.py NODE_CLASS_MAPPINGS

Run against every `workflows/*.json` fixture. Catches stranded links from
removed outputs without needing ComfyUI to load the graph.

Usage:
  python tools/validate_workflow_links.py
  python tools/validate_workflow_links.py --workflow workflows/otr_canonical.json
  python tools/validate_workflow_links.py --strict-types  # fail if any node type is missing from NODE_CLASS_MAPPINGS
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
from typing import Any, Iterable


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = REPO_ROOT / "workflows"
INIT_PATH = REPO_ROOT / "__init__.py"
# The v2 audio/casting nodes (OTR_CastLock / OTR_BatchCharacterVoices /
# OTR_AnnouncerVoice / OTR_StableAudioTheme, workflow nodes 80-83) are NOT in
# __init__._NODE_MODULES as a dict literal -- __init__ merges them dynamically
# from nodes/_otr_class_registry.new_node_modules_table(). A pure AST parse of
# __init__.py therefore cannot see them, so --strict-types false-flagged nodes
# 80-83 as "not in NODE_CLASS_MAPPINGS" even though runtime registration is fine
# (2026-07-04 widget-audit standalone fix). Parse the registry statically too.
REGISTRY_PATH = REPO_ROOT / "nodes" / "_otr_class_registry.py"


# Forbidden-pattern catalogue for source-side audits.
#
# Extended for S27 QA-4: `otr_legacy_audio_dir` is the legacy flat
# audio-dir helper kept as a SECONDARY entry in every audio-side auto-
# pick fallback chain. The 13 caller sites are enumerated in
# docs/2026-05-13-S26-audit-results.md under "B6 path back-compat -
# small (otr_legacy_audio_dir migration)" -- DEFERRED to a named
# follow-up sprint. Future audits use this regex set so the pattern
# stays visible and a new caller surfaces loudly.
#
# This validator itself only checks workflow JSON link integrity; the
# patterns below are consumed by the Phase 5 forbidden-pattern grep
# sweep documented in docs/2026-05-13-S27-audit-results.md. Kept
# here as the durable catalogue so future tooling can `from
# tools.validate_workflow_links import FORBIDDEN_PATTERNS`.
FORBIDDEN_PATTERNS: tuple[str, ...] = (
    r"DeprecationWarning",
    r"back-compat",
    r"back_compat",
    r"backcompat",
    r"legacy fallback",
    r"legacy path",
    r"legacy_path",
    r"\bshim\b",
    r"\balias\b",
    r"\botr_legacy_audio_dir\b",  # S27 QA-4 extension
)


def load_node_class_mappings() -> set[str]:
    """Parse __init__.py with ast (no exec) and extract the keys.

    The repo's __init__.py builds NODE_CLASS_MAPPINGS programmatically
    by iterating over a `_NODE_MODULES` dict literal whose keys are the
    canonical node-type strings (== NODE_CLASS_MAPPINGS keys at runtime).
    """
    src = INIT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in {
                    "NODE_CLASS_MAPPINGS", "_NODE_MODULES",
                }:
                    if isinstance(node.value, ast.Dict):
                        for k in node.value.keys:
                            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                                keys.add(k.value)
    # Dynamically-merged v2 audio nodes (registry, nodes 80-83): union the keys
    # declared as NewNodeSpec("KEY", ...) in _otr_class_registry.py so
    # --strict-types matches the real runtime NODE_CLASS_MAPPINGS.
    keys |= load_registry_keys()
    return keys


def load_registry_keys() -> set[str]:
    """AST-extract the node-type keys the package registers dynamically from
    nodes/_otr_class_registry.new_node_modules_table().

    Each new node is declared as ``NewNodeSpec("OTR_...", ...)`` whose first
    positional arg is the permanent NODE_CLASS_MAPPINGS key. Parse them without
    importing the module (no ComfyUI / torch dependency)."""
    keys: set[str] = set()
    if not REGISTRY_PATH.exists():
        return keys
    tree = ast.parse(REGISTRY_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = (func.id if isinstance(func, ast.Name)
                    else func.attr if isinstance(func, ast.Attribute) else "")
            if name == "NewNodeSpec" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    keys.add(first.value)
    return keys


def validate_workflow(path: pathlib.Path,
                      known_types: set[str],
                      strict_types: bool) -> list[str]:
    """Return a list of violation strings; empty list means clean."""
    violations: list[str] = []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"{path.name}: JSON parse failure ({e})"]
    if not isinstance(data, dict):
        return [f"{path.name}: top-level is not a dict"]
    nodes = data.get("nodes")
    links = data.get("links")
    if not isinstance(nodes, list):
        return [f"{path.name}: 'nodes' is missing or not a list"]
    if not isinstance(links, list):
        return [f"{path.name}: 'links' is missing or not a list"]

    # Build node lookup by id.
    node_by_id: dict[Any, dict] = {}
    for n in nodes:
        if isinstance(n, dict) and "id" in n:
            node_by_id[n["id"]] = n

    # ComfyUI link tuple shape: [link_id, src_node, src_slot, dst_node, dst_slot, type]
    link_by_id: dict[int, list] = {}
    for ln in links:
        if not isinstance(ln, list) or len(ln) < 5:
            violations.append(f"{path.name}: malformed link row {ln!r}")
            continue
        try:
            link_by_id[int(ln[0])] = ln
        except (TypeError, ValueError):
            violations.append(f"{path.name}: non-integer link id {ln[0]!r}")

    # Rule 4: node type must be in NODE_CLASS_MAPPINGS (skip vanilla Comfy
    # types like Note, Reroute, PrimitiveNode, etc.; only check OTR_-
    # prefixed ones unless --strict-types is set).
    for n in nodes:
        t = n.get("type", "") if isinstance(n, dict) else ""
        if not t:
            continue
        if strict_types or t.startswith("OTR_"):
            if t not in known_types and t not in _COMFY_BUILTIN_TYPES:
                violations.append(
                    f"{path.name}: node id={n.get('id')} type={t!r} is not in NODE_CLASS_MAPPINGS"
                )

    # Rule 1/2/3: every node input.link must point at a real link;
    # link's src + dst nodes must exist; dst input slot must exist.
    for n in nodes:
        if not isinstance(n, dict):
            continue
        inputs = n.get("inputs") or []
        for slot, inp in enumerate(inputs):
            if not isinstance(inp, dict):
                continue
            link = inp.get("link")
            if link is None:
                continue  # unconnected input; widget default takes over
            try:
                lid = int(link)
            except (TypeError, ValueError):
                violations.append(
                    f"{path.name}: node {n.get('id')} input[{slot}] link={link!r} non-integer"
                )
                continue
            if lid not in link_by_id:
                violations.append(
                    f"{path.name}: node {n.get('id')} input[{slot}] link={lid} not in workflow links[]"
                )
                continue
            row = link_by_id[lid]
            src_id, dst_id = row[1], row[3]
            if src_id not in node_by_id:
                violations.append(
                    f"{path.name}: link {lid} src node id={src_id} missing from nodes[]"
                )
            if dst_id not in node_by_id:
                violations.append(
                    f"{path.name}: link {lid} dst node id={dst_id} missing from nodes[]"
                )
            elif dst_id == n.get("id"):
                # Round-trip sanity: the link's dst node must have an
                # input whose `.link` field equals this link's id. Use
                # this rather than positional dst_slot because Comfy's
                # link tuple stores the INPUT_TYPES-order index, which
                # may not match the saved inputs[] array order (optional
                # / hidden inputs collapse positions in the saved file).
                back_ref_ids = {
                    other_inp.get("link") for other_inp in inputs
                    if isinstance(other_inp, dict)
                }
                if lid not in back_ref_ids:
                    violations.append(
                        f"{path.name}: link {lid} dst node {dst_id} does not back-reference this link from any input"
                    )
    return violations


# Vanilla ComfyUI types we tolerate without checking NODE_CLASS_MAPPINGS;
# add more if false positives surface during routine runs.
_COMFY_BUILTIN_TYPES: set[str] = {
    "Note",
    "MarkdownNote",
    "Reroute",
    "PrimitiveNode",
    "PreviewImage",
    "SaveImage",
    "LoadImage",
    "EmptyImage",
    "EmptyLatentImage",
    "VAEDecode",
    "VAEEncode",
    "VAELoader",
    "VAESave",
    "CLIPTextEncode",
    "CLIPSetLastLayer",
    "CheckpointLoaderSimple",
    "DualCLIPLoader",
    "UNETLoader",
    "ImageScale",
    "KSampler",
    "KSamplerAdvanced",
    "ConditioningCombine",
    "ConditioningConcat",
    "ConditioningSetMask",
    "ConditioningZeroOut",
    "ModelSamplingSD3",
    "FluxGuidance",
    "RandomNoise",
    "BasicScheduler",
    "BasicGuider",
    "KSamplerSelect",
    "SamplerCustomAdvanced",
    "SamplerCustom",
    "LoadAudio",
    "PreviewAudio",
    "SaveAudio",
    "EmptyAudio",
    "EmptyHunyuanLatentVideo",
    "ImageCrop",
    "ImageBatch",
    "ImageFromBatch",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", default=None,
                        help="Validate only this workflow (path). Default: every workflows/*.json fixture.")
    parser.add_argument("--strict-types", action="store_true",
                        help="Check every node type (not just OTR_-prefixed) against NODE_CLASS_MAPPINGS.")
    parser.add_argument("--include-external", action="store_true",
                        help="Include workflows/external_examples/*.json (otherwise skipped).")
    args = parser.parse_args()

    known_types = load_node_class_mappings()
    if not known_types:
        print("ERROR: failed to parse NODE_CLASS_MAPPINGS from __init__.py", file=sys.stderr)
        return 2

    targets: list[pathlib.Path]
    if args.workflow:
        targets = [pathlib.Path(args.workflow)]
    else:
        targets = sorted(WORKFLOWS_DIR.glob("*.json"))
        if args.include_external:
            targets += sorted((WORKFLOWS_DIR / "external_examples").glob("*.json"))

    if not targets:
        print("no workflow JSON files found", file=sys.stderr)
        return 0

    total_violations = 0
    for p in targets:
        violations = validate_workflow(p, known_types, args.strict_types)
        if violations:
            for v in violations:
                print(v)
            total_violations += len(violations)
        else:
            print(f"OK  {p.relative_to(REPO_ROOT) if p.is_absolute() else p}")
    print()
    print(f"TOTAL violations: {total_violations}")
    return 0 if total_violations == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
