"""scripts/wire_reviewer_into_workflow.py

One-shot script — wires OTR_LedgerScriptReviewer between
OTR_LedgerScriptWriter and OTR_LLMScriptCritic in
workflows/otr_scifi_16gb_full.json.

Idempotent: if the reviewer is already wired (a node with
type=OTR_LedgerScriptReviewer exists), exits without changing
anything.

Atomic write: writes to .tmp, then os.replace. Preserves
existing widgets_values order, link arrays, etc.

Status: Phase 3 wiring (2026-05-11).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = ROOT / "workflows" / "otr_scifi_16gb_full.json"


def main() -> int:
    if not WORKFLOW_PATH.exists():
        print(f"ERROR: {WORKFLOW_PATH} does not exist", file=sys.stderr)
        return 1

    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        wf = json.load(f)

    # Idempotency guard.
    existing = next(
        (n for n in wf["nodes"]
         if n.get("type") == "OTR_LedgerScriptReviewer"),
        None,
    )
    if existing is not None:
        print(f"OTR_LedgerScriptReviewer already wired (node id={existing['id']}); "
              f"no-op.")
        return 0

    # ----- Locate writer + critic ---------------------------------------
    writer = next(
        (n for n in wf["nodes"]
         if n.get("type") == "OTR_LedgerScriptWriter"),
        None,
    )
    critic = next(
        (n for n in wf["nodes"]
         if n.get("type") == "OTR_LLMScriptCritic"),
        None,
    )
    if writer is None:
        print("ERROR: no OTR_LedgerScriptWriter node found", file=sys.stderr)
        return 1
    if critic is None:
        print("ERROR: no OTR_LLMScriptCritic node found "
              "(expected midstream between writer + director)",
              file=sys.stderr)
        return 1

    writer_id = writer["id"]
    critic_id = critic["id"]
    new_node_id = wf["last_node_id"] + 1
    next_link_id = wf["last_link_id"] + 1

    # ----- Compute writer position to place reviewer to the right ------
    wpos = writer.get("pos", [0, 0])
    cpos = critic.get("pos", [0, 0])
    # Position reviewer between writer and critic on the canvas.
    new_x = (wpos[0] + cpos[0]) // 2
    new_y = wpos[1]

    # ----- Locate links 84 (script_text) and 85 (script_json) ----------
    # Writer outputs:
    #   slot 0 (script_text) -> [84]
    #   slot 1 (script_json) -> [85]
    #   slot 2 (news_used)   -> [18]   <-- left alone, downstream
    #   slot 3 (estimated_minutes) -> null

    links = wf["links"]
    # Find link with id=84 (writer.script_text -> critic.script).
    link_by_id = {ln[0]: ln for ln in links}
    if 84 not in link_by_id or 85 not in link_by_id:
        print("ERROR: expected links 84 + 85 between writer and critic; "
              "wiring assumption broken.", file=sys.stderr)
        return 1

    # ----- Build the new reviewer node ----------------------------------
    # 5 inputs (script_text, script_json, news_used, estimated_minutes,
    # model_id) and 5 outputs (the writer's 4-tuple + reviewer_verdict).
    # Inputs that are widget-bound (model_id) use the "widget" hint;
    # the rest are link-bound from the writer.
    #
    # Link plan:
    #   link 106 : writer.script_text       -> reviewer.script_text
    #   link 107 : writer.script_json       -> reviewer.script_json
    #   link 108 : writer.news_used         -> reviewer.news_used
    #   link 109 : writer.estimated_minutes -> reviewer.estimated_minutes
    #   link 84 (REUSED): reviewer.script_text -> critic.script
    #   link 85 (REUSED): reviewer.script_json -> critic.script_json
    #
    # Reusing links 84 + 85 keeps critic's existing input link refs
    # valid; we just swap the source endpoint from writer to reviewer.

    L_RW_TEXT = next_link_id     ; next_link_id += 1  # 106
    L_RW_JSON = next_link_id     ; next_link_id += 1  # 107
    L_RW_NEWS = next_link_id     ; next_link_id += 1  # 108
    L_RW_EST  = next_link_id     ; next_link_id += 1  # 109

    reviewer_node = {
        "id": new_node_id,
        "type": "OTR_LedgerScriptReviewer",
        "pos": [int(new_x), int(new_y)],
        "size": [380, 320],
        "title": "1b. Script Reviewer (3-pass cast-gated)",
        "properties": {
            "Node name for S&R": "OTR_LedgerScriptReviewer",
        },
        # widgets_values order MUST match the order optional widgets
        # are declared in INPUT_TYPES (alphabetical within ComfyUI's
        # internal ordering rules, but ComfyUI saves them in the
        # declared order). Defined here in INPUT_TYPES declaration
        # order:
        #   required: script_text   (widget-bound STRING)
        #   optional: script_json, news_used, estimated_minutes,
        #             model_id
        "widgets_values": [
            "",  # script_text (widget-bound when no link; left empty)
            "",  # script_json (widget-bound when no link)
            "",  # news_used
            0,   # estimated_minutes
            "mistralai/Mistral-Nemo-Instruct-2407",  # model_id
        ],
        "inputs": [
            {"name": "script_text", "type": "STRING",
             "link": L_RW_TEXT,
             "widget": {"name": "script_text"}},
            {"name": "script_json", "type": "STRING",
             "link": L_RW_JSON,
             "widget": {"name": "script_json"}},
            {"name": "news_used", "type": "STRING",
             "link": L_RW_NEWS,
             "widget": {"name": "news_used"}},
            {"name": "estimated_minutes", "type": "INT",
             "link": L_RW_EST,
             "widget": {"name": "estimated_minutes"}},
        ],
        "outputs": [
            {"name": "script_text", "type": "STRING",
             "links": [84], "slot_index": 0},
            {"name": "script_json", "type": "STRING",
             "links": [85], "slot_index": 1},
            {"name": "news_used", "type": "STRING",
             "links": [], "slot_index": 2},
            {"name": "estimated_minutes", "type": "INT",
             "links": [], "slot_index": 3},
            {"name": "reviewer_verdict", "type": "STRING",
             "links": [], "slot_index": 4},
        ],
        "flags": {},
        "order": writer.get("order", 0) + 1,
        "mode": 0,
        "color": "#2B3C5A",
        "bgcolor": "#1C273C",
    }

    # ----- Add the reviewer node ----------------------------------------
    wf["nodes"].append(reviewer_node)

    # ----- Rewrite the writer's output `links` lists --------------------
    # Writer's slot 0 (script_text): was [84], now [106]
    # Writer's slot 1 (script_json): was [85], now [107]
    # Writer's slot 2 (news_used):   [18, 108] (preserve existing)
    # Writer's slot 3 (estimated_minutes): was null, now [109]
    for out in writer.get("outputs", []):
        name = out.get("name")
        if name == "script_text":
            out["links"] = [L_RW_TEXT]
        elif name == "script_json":
            out["links"] = [L_RW_JSON]
        elif name == "news_used":
            existing_links = list(out.get("links") or [])
            if L_RW_NEWS not in existing_links:
                existing_links.append(L_RW_NEWS)
            out["links"] = existing_links
        elif name == "estimated_minutes":
            out["links"] = [L_RW_EST]

    # ----- Rewrite links table -----------------------------------------
    # Each link is [link_id, src_node, src_slot, dst_node, dst_slot, type]
    # Reassign links 84 + 85 to point FROM reviewer (new_node_id), not
    # from writer.
    for ln in links:
        if ln[0] == 84:
            # was: [84, writer_id, 0, critic_id, 0, "STRING"]
            ln[1] = new_node_id   # src_node = reviewer
            ln[2] = 0             # src_slot = reviewer.script_text
        elif ln[0] == 85:
            ln[1] = new_node_id   # src_node = reviewer
            ln[2] = 1             # src_slot = reviewer.script_json

    # Add the 4 new links (writer -> reviewer).
    links.append([L_RW_TEXT, writer_id, 0, new_node_id, 0, "STRING"])
    links.append([L_RW_JSON, writer_id, 1, new_node_id, 1, "STRING"])
    links.append([L_RW_NEWS, writer_id, 2, new_node_id, 2, "STRING"])
    links.append([L_RW_EST,  writer_id, 3, new_node_id, 3, "INT"])

    # ----- Bookkeeping -------------------------------------------------
    wf["last_node_id"] = new_node_id
    wf["last_link_id"] = L_RW_EST  # 109

    # ----- Atomic write ------------------------------------------------
    tmp_path = str(WORKFLOW_PATH) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(wf, f, indent=2, ensure_ascii=False)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp_path, WORKFLOW_PATH)

    print(f"OK: reviewer wired as node id={new_node_id}")
    print(f"    new links: {L_RW_TEXT}, {L_RW_JSON}, {L_RW_NEWS}, {L_RW_EST}")
    print(f"    reassigned: link 84 src -> reviewer.slot_0, "
          f"link 85 src -> reviewer.slot_1")
    print(f"    last_node_id: {wf['last_node_id']}, "
          f"last_link_id: {wf['last_link_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
