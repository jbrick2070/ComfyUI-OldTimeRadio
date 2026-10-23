<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# Workflow Validator (opt-in, S14.2)

An optional first node that checks the workflow against what this machine can run before anything heavy loads, and names what would fail.

## What it does

Opt-in execution-time workflow contract validator. Place as the first node in a workflow to catch contract drift at queue time. Reads the workflow JSON from disk and runs the same validate_workflow_contract check that runs in CI.

## Inputs

The shipped workflows set these; most people never change them.

**`workflow_json_path`** -- text; starts as a path on this machine.

Path of the workflow JSON this validator audits (default: the canonical graph). Template/expert field -- generated variants stamp it; a missing file fails the validation report, not the render.

**`validate_anyway`** -- on or off; starts as on.

True: run the structural audit and RAISE on a contract or widget-vector violation, halting the queue before any model loads. False: skip the audit body and report 'skipped' -- the diagnostic bypass for a deliberately drifted graph.

This flag governs the STRUCTURAL AUDIT only. Creator-input admission (My Story) always runs, in both settings, because it is what stops a blank or misdirected submission from costing an asset download.

**`strict_unknown_types`** -- on or off; starts as on.

True: an unregistered node type in the audited JSON is a reported violation. False: unknown types are listed as informational only. Keep True on the canonical graph.

### Optional

**`profile_id`** -- text; starts as empty.

Generation stamp: which hardware profile generated this variant. Written by build_variants, empty on the canonical master. Do not hand-edit.

**`master_hash`** -- text; starts as empty.

Generation stamp: semantic hash of this variant as emitted. A mismatch against the workflow file means this graph is a stale copy (re-open it from Browse Templates) or the file was edited (reinstall the pack); never hand-fix.

**`generated_by`** -- text; starts as empty.

Generation stamp: the tool/version that emitted this variant. Empty on the canonical master; informational in the validation report.

**`credential`** -- a connection from another node (`STRING`).

Wire the '0 - Comfy Credential' node here. It carries no key -- only the order.

## Outputs

- `validation_report` (`STRING`)

This is an output node: the workflow runs to reach it.

## Read more

- [PREFLIGHT.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/PREFLIGHT.md)
- [MACHINES.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/MACHINES.md)

Class `OTR_WorkflowValidator`, category `OldTimeRadio/diagnostics`.
