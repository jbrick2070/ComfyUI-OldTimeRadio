#!/usr/bin/env python3
"""scripts/lfc_wiring_smoke.py — LFC clean-break invariants smoke check.

Single-shot script that runs the wiring + class-surface invariants
the LFC clean-break protocol pins. Designed to run in CI (`python
scripts/lfc_wiring_smoke.py` from repo root) so any future
regression that violates these invariants fails loudly.

Checks
  1. Workflow JSON: cascade node id 62 widgets_values matches
     INPUT_TYPES defaults exactly (C5).
  2. Workflow JSON: cascade node widgets_values length matches the
     INPUT_TYPES count.
  3. Python: OTR_LedgerFreezeCascade.RETURN_NAMES matches the
     workflow JSON's outputs[].name order (C6).
  4. Repo source: no legacy class names (OTR_LedgerScriptReviewer,
     OTR_Gemma4Director, OTR_LLMScriptWriter, OTR_Gemma4ScriptWriter,
     _RENAME_ALIASES) appear in source files outside the
     allow-list (clean-break step 7, D4 extension 12.17).
  5. Workflow JSON: news_used link chain integrity --
     writer[2] -> cascade[2] -> SignalLostVideo[2].
     Pre-fix bypass link (writer -> SignalLostVideo direct) absent.
  6. Workflow JSON: last_link_id >= max link id present.
  7. Workflow JSONs: no node carries a legacy class identifier
     in its `type` / `class_type` field (D4 extension 12.17).
     Catches the case where a saved-but-stale workflow file still
     references a renamed class.

Exit codes
  0   all checks passed.
  1   one or more checks failed; details printed to stderr.

Usage:
  python scripts/lfc_wiring_smoke.py
  python scripts/lfc_wiring_smoke.py --strict   # warnings = failures
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF_PATH = ROOT / "workflows" / "otr_scifi_16gb_full.json"

# Make `nodes` importable when running the script directly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASCADE_NODE_ID = 62
WRITER_NODE_ID = 1
SLV_NODE_ID = 12

EXPECTED_WIDGET_VALUES = [
    "mistralai/Mistral-Nemo-Instruct-2407",
    False,    # enable_phase_3_polish
    False,    # polish_announcer_beats
    False,    # enable_phase_4_scene_coherence
    False,    # enable_phase_4_5_smart_suggestion
    False,    # enable_phase_5_voice_drift
    False,    # enable_phase_6_episode_arc
    True,     # enable_phase_7_audio_readiness
    True,     # enable_phase_8_video_readiness
    14.0,     # vram_ceiling_gb
]

EXPECTED_RETURN_NAMES = (
    "script_text", "script_json", "news_used",
    "estimated_minutes", "freeze_verdict",
)

LEGACY_TOKENS = (
    "OTR_LedgerScriptReviewer",
    "OTR_Gemma4Director",
    # D4 extension (commit 12.17): broader legacy class name set.
    "OTR_LLMScriptWriter",       # superseded by OTR_LedgerScriptWriter
    "OTR_Gemma4ScriptWriter",    # defensive; not currently in tree
    "_RENAME_ALIASES",            # legacy rename-alias dict removed in 12.3
)

# Workflow-JSON-only scan: catch saved-but-stale workflows whose
# nodes still carry a renamed class id in the `type` / `class_type`
# field. Source-code scan above misses these because the strings live
# inside workflow JSON, not Python.
WORKFLOW_LEGACY_CLASS_TYPES = (
    "OTR_LedgerScriptReviewer",
    "OTR_Gemma4Director",
    "OTR_LLMScriptWriter",
    "OTR_Gemma4ScriptWriter",
    "OTR_LegacyScriptWriter",
)

SCAN_DIRS = ("nodes", "tests", "otr_v2", "visual", "scripts")
WORKFLOW_DIRS = ("workflows",)

LEGACY_ALLOW_FILES = {
    "tests/test_legacy_contract_retired.py",
    "tests/test_lfc_freeze_cascade_orchestrator.py",
    "tests/test_lfc_wiring_smoke_script.py",   # pytest wrapper
    "scripts/lfc_wiring_smoke.py",             # this file itself
}


def _load_workflow() -> dict:
    return json.loads(WF_PATH.read_text(encoding="utf-8"))


def _node(wf: dict, node_id: int) -> dict:
    return next(n for n in wf["nodes"] if n.get("id") == node_id)


def _scan_legacy_token(token: str) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for top in SCAN_DIRS:
        dir_path = ROOT / top
        if not dir_path.exists():
            continue
        for path in dir_path.rglob("*.py"):
            rel = path.relative_to(ROOT).as_posix()
            if rel in LEGACY_ALLOW_FILES:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            if token not in text:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if token not in line:
                    continue
                hash_pos = line.find("#")
                token_pos = line.find(token)
                if hash_pos != -1 and hash_pos < token_pos:
                    continue
                stripped = line.lstrip()
                if stripped.startswith(('"""', "'''")):
                    continue
                if stripped.startswith('"') or stripped.startswith("'"):
                    continue
                hits.append((rel, lineno, line.rstrip()))
    return hits


def check_widget_defaults(wf: dict, errors: list, warnings: list) -> None:
    node = _node(wf, CASCADE_NODE_ID)
    actual = node.get("widgets_values") or []
    if actual != EXPECTED_WIDGET_VALUES:
        errors.append(
            f"C5: cascade widgets_values mismatch.\n"
            f"  expected: {EXPECTED_WIDGET_VALUES}\n"
            f"  actual:   {actual}"
        )


def check_widget_count(wf: dict, errors: list, warnings: list) -> None:
    node = _node(wf, CASCADE_NODE_ID)
    actual_len = len(node.get("widgets_values") or [])
    if actual_len != len(EXPECTED_WIDGET_VALUES):
        errors.append(
            f"C5: cascade widgets_values length {actual_len} != "
            f"expected {len(EXPECTED_WIDGET_VALUES)}"
        )


def check_return_names(wf: dict, errors: list, warnings: list) -> None:
    try:
        from nodes.OTR_LedgerFreezeCascade import OTR_LedgerFreezeCascade
    except Exception as exc:
        errors.append(
            f"C6: failed to import OTR_LedgerFreezeCascade ({exc})"
        )
        return
    py_names = OTR_LedgerFreezeCascade.RETURN_NAMES
    if py_names != EXPECTED_RETURN_NAMES:
        errors.append(
            f"C6: Python RETURN_NAMES mismatch.\n"
            f"  expected: {EXPECTED_RETURN_NAMES}\n"
            f"  actual:   {py_names}"
        )
        return
    node = _node(wf, CASCADE_NODE_ID)
    json_names = tuple(
        o.get("name", "") for o in (node.get("outputs") or [])
    )
    if json_names != EXPECTED_RETURN_NAMES:
        errors.append(
            f"C6: workflow JSON cascade outputs slot-name order "
            f"mismatch.\n"
            f"  expected: {EXPECTED_RETURN_NAMES}\n"
            f"  actual:   {json_names}"
        )


def check_legacy_tokens_absent(errors: list, warnings: list) -> None:
    for token in LEGACY_TOKENS:
        hits = _scan_legacy_token(token)
        if hits:
            lines = "\n".join(
                f"  {rel}:{lineno}: {text}"
                for rel, lineno, text in hits
            )
            errors.append(
                f"clean-break: legacy token `{token}` found in source.\n"
                f"  hits:\n{lines}"
            )


def check_news_used_chain(wf: dict, errors: list, warnings: list) -> None:
    writer = _node(wf, WRITER_NODE_ID)
    cascade = _node(wf, CASCADE_NODE_ID)
    slv = _node(wf, SLV_NODE_ID)

    w_out = next(
        (o for o in writer.get("outputs", []) or []
         if o.get("name") == "news_used"),
        None,
    )
    if w_out is None:
        errors.append("news_used: writer has no news_used output")
        return
    w_links = w_out.get("links") or []
    if 108 not in w_links:
        errors.append(
            f"news_used: writer.news_used.links should include 108, "
            f"got {w_links}"
        )
    if 18 in w_links:
        errors.append(
            "news_used: writer.news_used still has the pre-fix "
            "bypass link 18 -- W2 fix regressed"
        )

    c_out = next(
        (o for o in cascade.get("outputs", []) or []
         if o.get("name") == "news_used"),
        None,
    )
    if c_out is None:
        errors.append("news_used: cascade has no news_used output")
    elif 110 not in (c_out.get("links") or []):
        errors.append(
            f"news_used: cascade.news_used.links should include 110, "
            f"got {c_out.get('links')}"
        )

    s_in = next(
        (i for i in slv.get("inputs", []) or []
         if i.get("name") == "news_used"),
        None,
    )
    if s_in is None:
        errors.append("news_used: SignalLostVideo has no news_used input")
    elif s_in.get("link") != 110:
        errors.append(
            f"news_used: SignalLostVideo.news_used.input.link should "
            f"be 110, got {s_in.get('link')}"
        )

    link_ids = {l[0] for l in (wf.get("links") or [])}
    for required in (108, 110):
        if required not in link_ids:
            errors.append(
                f"news_used: link object {required} missing from "
                f"workflow.links"
            )
    if 18 in link_ids:
        errors.append(
            "news_used: legacy link object 18 still in workflow.links"
        )


def _scan_workflow_class_types() -> list[tuple[str, int, str, str]]:
    """Walk every workflow JSON under WORKFLOW_DIRS, surface any node
    whose `type` or `class_type` field carries a legacy class id.

    Returns list of (rel_path, node_id, field_name, value) tuples.
    """
    hits: list[tuple[str, int, str, str]] = []
    for top in WORKFLOW_DIRS:
        dir_path = ROOT / top
        if not dir_path.exists():
            continue
        for path in dir_path.rglob("*.json"):
            rel = path.relative_to(ROOT).as_posix()
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                # Skip malformed JSON; other guardrail tests catch parse errors.
                continue
            # UI-format workflow: doc["nodes"] is a list of dicts with "type".
            for node in doc.get("nodes") or []:
                if not isinstance(node, dict):
                    continue
                node_id = node.get("id", -1)
                ctype = node.get("type")
                if isinstance(ctype, str) and ctype in WORKFLOW_LEGACY_CLASS_TYPES:
                    hits.append((rel, node_id, "type", ctype))
            # API-format prompt dict: each value is {class_type, inputs}.
            for k, v in doc.items():
                if not isinstance(v, dict):
                    continue
                ctype = v.get("class_type")
                if isinstance(ctype, str) and ctype in WORKFLOW_LEGACY_CLASS_TYPES:
                    try:
                        nid = int(k)
                    except (TypeError, ValueError):
                        nid = -1
                    hits.append((rel, nid, "class_type", ctype))
    return hits


def check_workflow_no_legacy_class_types(
    errors: list, warnings: list,
) -> None:
    hits = _scan_workflow_class_types()
    if hits:
        lines = "\n".join(
            f"  {rel} node {nid}: {field}={ctype!r}"
            for rel, nid, field, ctype in hits
        )
        errors.append(
            "clean-break: workflow JSON carries legacy class id in "
            "node.type / class_type field.\n"
            f"  hits:\n{lines}"
        )


def check_last_link_id(wf: dict, errors: list, warnings: list) -> None:
    last_link = wf.get("last_link_id", 0)
    max_present = max(
        (l[0] for l in (wf.get("links") or [])),
        default=0,
    )
    if last_link < max_present:
        errors.append(
            f"last_link_id={last_link} < max link id present "
            f"({max_present})"
        )


def main(argv: list[str]) -> int:
    strict = "--strict" in argv
    wf = _load_workflow()
    errors: list[str] = []
    warnings: list[str] = []

    check_widget_defaults(wf, errors, warnings)
    check_widget_count(wf, errors, warnings)
    check_return_names(wf, errors, warnings)
    check_legacy_tokens_absent(errors, warnings)
    check_news_used_chain(wf, errors, warnings)
    check_last_link_id(wf, errors, warnings)
    check_workflow_no_legacy_class_types(errors, warnings)

    if warnings:
        for w in warnings:
            print(f"WARN: {w}", file=sys.stderr)

    if errors:
        for e in errors:
            print(f"FAIL: {e}", file=sys.stderr)
        return 1

    if strict and warnings:
        print(
            f"FAIL: {len(warnings)} warning(s) in --strict mode",
            file=sys.stderr,
        )
        return 1

    print(
        f"[lfc_wiring_smoke] OK -- 7 checks passed "
        f"({len(warnings)} warning(s))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
