"""Wave 2 Agent F: workflow JSON wire-up for OTR_StoryRoom.

Adds a new node entry (id 75, type OTR_StoryRoom) downstream of the
Director (node 74) and wires:
  - Director.director_brief (out slot 0) -> Story Room.director_brief (in slot 0)
  - Writer.creative_writing_model (node 1 out slot 4) -> Story Room.creative_writing_model
  - Writer.technical_model (node 1 out slot 5) -> Story Room.technical_model
Story Room's output socket stays disconnected -- Wave 2 Agent G's
OTR_StoryRoomExtract will wire it in the next commit.

The widget vector ships use_story_room=False so PD1 byte-identity
holds for the legacy audio path. Wave 3 operator A/B flips it on.

Run from the repo root via the project venv:
    C:\\Users\\jeffr\\Documents\\ComfyUI\\.venv\\Scripts\\python.exe \
        docs\\_w2f_wire_story_room.py

Idempotent: re-running with the OTR_StoryRoom node already present
exits without mutating the file.
"""
from __future__ import annotations

import json
import os
import sys

WORKFLOW_PATH = "workflows/otr_scifi_16gb_full.json"

WRITER_NODE_ID = 1
WRITER_CREATIVE_OUT_SLOT = 4
WRITER_TECHNICAL_OUT_SLOT = 5
DIRECTOR_NODE_ID = 74
DIRECTOR_OUT_SLOT = 0


def main() -> int:
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data["nodes"]
    links = data["links"]

    # Idempotency guard: if OTR_StoryRoom already lives in the workflow,
    # bail without mutating. Lets the script be re-run safely from CI.
    existing = [n for n in nodes if n.get("type") == "OTR_StoryRoom"]
    if existing:
        print(
            f"[w2f] OTR_StoryRoom already present (id={existing[0]['id']}); "
            "no-op."
        )
        return 0

    next_node_id = max(n["id"] for n in nodes) + 1
    next_link_id = max(l[0] for l in links) + 1

    link_brief_to_room = next_link_id
    link_creative_to_room = next_link_id + 1
    link_technical_to_room = next_link_id + 2

    # New StoryRoom node entry. Placement is downstream of the Director
    # (pos ~ Director y + 260) so the visual flow reads left-to-right
    # in the editor without overlapping the Editor node.
    new_node = {
        "id": next_node_id,
        "type": "OTR_StoryRoom",
        "pos": [
            500,
            520,
        ],
        "size": [
            440,
            260,
        ],
        "title": "Story Room (Sprint 10B Wave 2 Agent F, DORMANT)",
        "flags": {},
        "order": 100,
        "mode": 0,
        "inputs": [
            {
                "name": "director_brief",
                "type": "STRING",
                "link": link_brief_to_room,
            },
            {
                "name": "news_seed",
                "type": "STRING",
                "link": None,
            },
            {
                "name": "cast_names",
                "type": "STRING",
                "link": None,
            },
            {
                "name": "creative_writing_model",
                "type": "STRING",
                "link": link_creative_to_room,
                "shape": 7,
            },
            {
                "name": "technical_model",
                "type": "STRING",
                "link": link_technical_to_room,
                "shape": 7,
            },
        ],
        "outputs": [
            {
                "name": "story_room_transcript",
                "type": "STRING",
                # Disconnected today; Wave 2 Agent G wires the Extract
                # node to this output socket next commit.
                "links": None,
                "slot_index": 0,
            },
        ],
        "properties": {
            "Node name for S&R": "OTR_StoryRoom",
            "cnr_id": "OldTimeRadio",
        },
        # Widget vector mirrors the INPUT_TYPES `required` widget
        # order: director_brief (multiline str), news_seed (multiline
        # str), cast_names (str), use_story_room (bool), max_writer_turns
        # (int), max_editor_cycles (int), max_total_turns (int). The
        # three STRING widgets get empty defaults; the live values flow
        # in over the input sockets above.
        "widgets_values": [
            "",
            "",
            "",
            False,
            4,
            3,
            18,
        ],
        "color": "#2B3C5A",
        "bgcolor": "#1C273C",
    }
    nodes.append(new_node)

    # Wire Director -> Story Room.director_brief.
    # Link format: [link_id, source_node, source_slot, target_node,
    #               target_slot, type].
    links.append([
        link_brief_to_room,
        DIRECTOR_NODE_ID,
        DIRECTOR_OUT_SLOT,
        next_node_id,
        0,
        "STRING",
    ])
    # Update the Director output socket's `links` field so the editor
    # renders the wire. Director output `links` was null pre-wire.
    director = next(n for n in nodes if n["id"] == DIRECTOR_NODE_ID)
    director_out = director["outputs"][DIRECTOR_OUT_SLOT]
    if director_out.get("links") is None:
        director_out["links"] = []
    director_out["links"].append(link_brief_to_room)

    # Wire writer.creative_writing_model -> Story Room.
    links.append([
        link_creative_to_room,
        WRITER_NODE_ID,
        WRITER_CREATIVE_OUT_SLOT,
        next_node_id,
        3,
        "STRING",
    ])
    writer = next(n for n in nodes if n["id"] == WRITER_NODE_ID)
    creative_out = writer["outputs"][WRITER_CREATIVE_OUT_SLOT]
    if creative_out.get("links") is None:
        creative_out["links"] = []
    creative_out["links"].append(link_creative_to_room)

    # Wire writer.technical_model -> Story Room.
    links.append([
        link_technical_to_room,
        WRITER_NODE_ID,
        WRITER_TECHNICAL_OUT_SLOT,
        next_node_id,
        4,
        "STRING",
    ])
    technical_out = writer["outputs"][WRITER_TECHNICAL_OUT_SLOT]
    if technical_out.get("links") is None:
        technical_out["links"] = []
    technical_out["links"].append(link_technical_to_room)

    # Bump the workflow's "last_node_id" / "last_link_id" if present.
    if "last_node_id" in data:
        data["last_node_id"] = max(data["last_node_id"], next_node_id)
    if "last_link_id" in data:
        data["last_link_id"] = max(data["last_link_id"], link_technical_to_room)

    with open(WORKFLOW_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(
        f"[w2f] wired OTR_StoryRoom as node {next_node_id}; "
        f"new links {link_brief_to_room}/{link_creative_to_room}/"
        f"{link_technical_to_room}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
