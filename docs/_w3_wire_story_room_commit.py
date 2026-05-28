"""Wave 3 bridge: wire OTR_StoryRoomCommit between writer + freeze cascade.

Inserts a new node (OTR_StoryRoomCommit) into the writer ->
freeze cascade backbone. Ships dormant (commit=False default)
so the legacy pipeline is byte-identical.

Topology change:
  Before:  Writer.script_json (link 107) -> FreezeCascade.script_json
  After:   Writer.script_json (link 107) -> Commit.script_json
           Commit.script_json (NEW link) -> FreezeCascade.script_json
           Extract.story_room_extraction (NEW link) -> Commit.story_room_extraction

Idempotent: re-running with OTR_StoryRoomCommit already present
exits without mutating.

Run from the repo root via the project venv:
    C:\\Users\\jeffr\\Documents\\ComfyUI\\.venv\\Scripts\\python.exe \
        docs\\_w3_wire_story_room_commit.py
"""
from __future__ import annotations

import json
import sys

WORKFLOW_PATH = "workflows/otr_scifi_16gb_full.json"

WRITER_NODE_ID = 1
WRITER_SCRIPT_JSON_OUT_SLOT = 1


def main() -> int:
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data["nodes"]
    links = data["links"]

    if any(n.get("type") == "OTR_StoryRoomCommit" for n in nodes):
        existing = next(
            n for n in nodes if n.get("type") == "OTR_StoryRoomCommit"
        )
        print(
            f"[w3] OTR_StoryRoomCommit already present "
            f"(id={existing['id']}); no-op."
        )
        return 0

    extract = next(
        (n for n in nodes if n.get("type") == "OTR_StoryRoomExtract"),
        None,
    )
    if extract is None:
        print(
            "[w3] FATAL: OTR_StoryRoomExtract node not found. Wave 2 "
            "Agent G must land before Wave 3 commit can wire."
        )
        return 1
    extract_node_id = extract["id"]
    extract_out_slot = 0  # story_room_extraction

    # Find the existing writer -> freeze cascade link on script_json.
    # Link format: [link_id, src_node, src_slot, dst_node, dst_slot, type]
    writer_to_cascade_link = None
    for link in links:
        lid, src, src_slot, dst, dst_slot, ltype = link
        if (
            src == WRITER_NODE_ID
            and src_slot == WRITER_SCRIPT_JSON_OUT_SLOT
            and ltype == "STRING"
        ):
            writer_to_cascade_link = link
            break
    if writer_to_cascade_link is None:
        print(
            "[w3] FATAL: could not find writer.script_json -> "
            "freeze cascade link. Workflow topology may have drifted."
        )
        return 1
    cascade_node_id = writer_to_cascade_link[3]
    cascade_dst_slot = writer_to_cascade_link[4]
    original_link_id = writer_to_cascade_link[0]
    print(
        f"[w3] found writer.script_json -> cascade link "
        f"id={original_link_id} dst=({cascade_node_id},{cascade_dst_slot})"
    )

    next_node_id = max(n["id"] for n in nodes) + 1
    next_link_id = max(l[0] for l in links) + 1
    link_commit_to_cascade = next_link_id
    link_extract_to_commit = next_link_id + 1

    # Redirect link 107 (writer.script_json) -> Commit.script_json
    # (slot 0).
    writer_to_cascade_link[3] = next_node_id
    writer_to_cascade_link[4] = 0

    # New Commit node entry.
    commit_node = {
        "id": next_node_id,
        "type": "OTR_StoryRoomCommit",
        "pos": [
            500,
            900,
        ],
        "size": [
            440,
            200,
        ],
        "title": "Story Room Commit (Wave 3 bridge, DORMANT)",
        "flags": {},
        "order": 102,
        "mode": 0,
        "inputs": [
            {
                "name": "script_json",
                "type": "STRING",
                "link": original_link_id,
            },
            {
                "name": "story_room_extraction",
                "type": "STRING",
                "link": link_extract_to_commit,
            },
            {
                "name": "commit",
                "type": "BOOLEAN",
                "link": None,
            },
        ],
        "outputs": [
            {
                "name": "script_json",
                "type": "STRING",
                "links": [link_commit_to_cascade],
                "slot_index": 0,
            },
        ],
        "properties": {
            "Node name for S&R": "OTR_StoryRoomCommit",
            "cnr_id": "OldTimeRadio",
        },
        "widgets_values": [
            "",       # script_json (multiline str widget)
            "",       # story_room_extraction (multiline str widget)
            False,    # commit (BOOLEAN -- defaults dormant)
        ],
        "color": "#2B3C5A",
        "bgcolor": "#1C273C",
    }
    nodes.append(commit_node)

    # New link: Commit.script_json -> FreezeCascade (where the
    # original writer link used to go).
    links.append([
        link_commit_to_cascade,
        next_node_id,
        0,
        cascade_node_id,
        cascade_dst_slot,
        "STRING",
    ])

    # New link: Extract.story_room_extraction -> Commit.story_room_extraction.
    links.append([
        link_extract_to_commit,
        extract_node_id,
        extract_out_slot,
        next_node_id,
        1,
        "STRING",
    ])
    extract_out = extract["outputs"][extract_out_slot]
    if extract_out.get("links") is None:
        extract_out["links"] = []
    extract_out["links"].append(link_extract_to_commit)

    if "last_node_id" in data:
        data["last_node_id"] = max(data["last_node_id"], next_node_id)
    if "last_link_id" in data:
        data["last_link_id"] = max(
            data["last_link_id"], link_extract_to_commit,
        )

    with open(WORKFLOW_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(
        f"[w3] wired OTR_StoryRoomCommit as node {next_node_id}; "
        f"new links {link_commit_to_cascade}/{link_extract_to_commit}; "
        f"redirected writer.script_json link {original_link_id} from "
        f"({cascade_node_id},{cascade_dst_slot}) -> ({next_node_id},0)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
