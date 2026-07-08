"""Build and optionally submit the canonical OTR workflow through ComfyUI API.

This is the small, boring headless entrypoint agents should use when the
operator asks for an API render/smoke:

* Always load workflows/otr_canonical.json from this repo.
* Apply engine/dropdown capability profiles through the single profile applier.
* Patch only creative/story widgets directly, through otr_api.patch_creative.
* Convert with scripts/otr_api.py and fail loud on schema/widget drift.

It deliberately has no --workflow argument. If a run needs a different graph,
that is a workflow change, not a headless smoke.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
CANONICAL_WORKFLOW = REPO_ROOT / "workflows" / "otr_canonical.json"
DEFAULT_PROMPT_DUMP = HERE / "_otr_canonical_api_prompt.json"

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from otr_api import (  # noqa: E402
    COMFYUI_URL,
    apply_profile_to_workflow,
    fetch_schemas,
    load_workflow,
    patch_creative,
    poll_history,
    submit_prompt,
    workflow_to_api_prompt,
)


def _parse_value(raw: str) -> Any:
    """Parse CLI values as JSON when possible; otherwise keep a string."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _node_id_for(workflow: dict, node_ref: str) -> int:
    """Resolve a node id or unique node type in the canonical workflow."""
    if node_ref.isdigit():
        nid = int(node_ref)
        if any(n.get("id") == nid for n in workflow.get("nodes", [])):
            return nid
        raise SystemExit(f"node id {nid} is not in the canonical workflow")

    matches = [
        int(n["id"])
        for n in workflow.get("nodes", [])
        if n.get("type") == node_ref
    ]
    if not matches:
        raise SystemExit(
            f"node type {node_ref!r} is not in the canonical workflow"
        )
    if len(matches) > 1:
        raise SystemExit(
            f"node type {node_ref!r} appears {len(matches)} times; use id.widget"
        )
    return matches[0]


def _apply_set(workflow: dict, schemas: dict, patch: str) -> str:
    """Apply NODE_OR_ID.widget=value through the creative whitelist."""
    if "=" not in patch:
        raise SystemExit(f"--set must look like NODE.widget=value, got {patch!r}")
    left, raw_value = patch.split("=", 1)
    if "." not in left:
        raise SystemExit(f"--set must name NODE.widget, got {patch!r}")
    node_ref, widget = left.rsplit(".", 1)
    nid = _node_id_for(workflow, node_ref)
    value = _parse_value(raw_value)
    patch_creative(workflow, nid, widget, value, schemas)
    return f"{node_ref}.{widget}={value!r}"


def _apply_writer_shortcuts(workflow: dict, schemas: dict, args) -> list[str]:
    writer_id = _node_id_for(workflow, "OTR_LedgerScriptWriter")
    shortcuts: list[tuple[str, Any]] = []
    if args.words is not None:
        shortcuts.append(("target_words", int(args.words)))
    if args.title is not None:
        shortcuts.append(("episode_title", args.title))
    if args.premise is not None:
        shortcuts.append(("custom_premise", args.premise))
    if args.source_bank is not None:
        shortcuts.append(("source_bank", args.source_bank))
    if args.visual_style is not None:
        shortcuts.append(("visual_style", args.visual_style))
    if args.creative_model is not None:
        shortcuts.append(("creative_writing_model", args.creative_model))
    if args.technical_model is not None:
        shortcuts.append(("technical_model", args.technical_model))
    if args.num_characters is not None:
        shortcuts.append(("num_characters", int(args.num_characters)))
    if args.act_count is not None:
        shortcuts.append(("act_count", str(args.act_count)))

    applied = []
    for widget, value in shortcuts:
        patch_creative(workflow, writer_id, widget, value, schemas)
        applied.append(f"OTR_LedgerScriptWriter.{widget}={value!r}")
    return applied


def _schemas(offline: bool) -> dict:
    if not offline:
        return fetch_schemas()
    from nodes import _otr_workflow_apply as workflow_apply

    return workflow_apply.build_offline_schemas()


def build_api_prompt(args) -> tuple[dict, list[str]]:
    canonical = CANONICAL_WORKFLOW.resolve()
    expected = (
        REPO_ROOT / "workflows" / "otr_canonical.json"
    ).resolve()
    if canonical != expected:
        raise SystemExit(f"canonical workflow path mismatch: {canonical}")
    if not canonical.is_file():
        raise SystemExit(f"canonical workflow missing: {canonical}")

    print(f"[canonical-api] workflow={canonical}", flush=True)
    schemas = _schemas(args.offline_schemas)
    workflow = load_workflow(str(canonical))

    applied: list[str] = []
    if args.profile and args.profile.lower() != "none":
        workflow = apply_profile_to_workflow(workflow, args.profile, schemas)
        applied.append(f"profile={args.profile}")

    applied.extend(_apply_writer_shortcuts(workflow, schemas, args))
    for patch in args.set:
        applied.append(_apply_set(workflow, schemas, patch))

    prompt = workflow_to_api_prompt(workflow, schemas)
    return prompt, applied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the canonical OTR workflow through the ComfyUI API."
    )
    parser.add_argument("--profile", default="none",
                        help="explicit capability profile id, e.g. none or cloud_all")
    parser.add_argument("--words", type=int, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--premise", default=None)
    parser.add_argument("--source-bank", default=None)
    parser.add_argument("--visual-style", default=None)
    parser.add_argument("--creative-model", default=None)
    parser.add_argument("--technical-model", default=None)
    parser.add_argument("--num-characters", type=int, default=None)
    parser.add_argument("--act-count", default=None)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NODE_OR_ID.widget=value",
        help="direct creative patch; managed engine widgets are refused",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="build and dump the API prompt without POST /prompt")
    parser.add_argument("--offline-schemas", action="store_true",
                        help="use offline node schemas; for unit tests only")
    parser.add_argument("--dump-prompt", default=str(DEFAULT_PROMPT_DUMP))
    parser.add_argument("--timeout", type=int, default=5400)
    parser.add_argument("--poll-s", type=int, default=5)
    args = parser.parse_args(argv)

    prompt, applied = build_api_prompt(args)
    dump_path = pathlib.Path(args.dump_prompt)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    dump_path.write_text(json.dumps(prompt, indent=2), encoding="utf-8")

    print(f"[canonical-api] comfy_url={COMFYUI_URL}", flush=True)
    print(f"[canonical-api] prompt_nodes={len(prompt)}", flush=True)
    print(f"[canonical-api] prompt_dump={dump_path}", flush=True)
    if applied:
        print("[canonical-api] applied:", flush=True)
        for item in applied:
            print(f"  {item}", flush=True)

    if args.dry_run:
        print("[canonical-api] DRY_RUN complete; prompt not submitted", flush=True)
        return 0

    prompt_id = submit_prompt(prompt)
    print(f"[canonical-api] QUEUED prompt_id={prompt_id}", flush=True)
    status, err = poll_history(
        prompt_id, timeout_s=args.timeout, poll_s=args.poll_s
    )
    print(f"[canonical-api] RESULT {status} prompt_id={prompt_id}", flush=True)
    if status != "SUCCESS":
        if err:
            print(f"[canonical-api] ERROR {err}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
