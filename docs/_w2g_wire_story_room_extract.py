"""Wave 2 Agent G: workflow JSON wire-up for OTR_StoryRoomExtract.

Adds a new node entry (type OTR_StoryRoomExtract) downstream of the
Story Room (Wave 2 Agent F) and wires:
  - StoryRoom.story_room_transcript (node 75 out 0) ->
    Extract.story_room_transcript (in 0)
  - Writer.technical_model (node 1 out 5) -> Extract.technical_model

The Extract output ships disconnected -- Wave 3's bridge wires the
downstream consumers (announcer / continuity / music+SFX / bark) when
the operator A/B confirms use_story_room flips on by default.

Run from the repo root via the project venv:
    C:\\Users\\jeffr\\Documents\\ComfyUI\\.venv\\Scripts\\python.exe \
        docs\\_w2g_wire_story_room_extract.py

Idempotent: re-running with the OTR_StoryRoomExtract node already
present exits without mutating the file.
"""
from __future__ import annotations

import json
import sys

WORKFLOW_PATH = "workflows/otr_scifi_16gb_full.json"

WRITER_NODE_ID = 1
WRITER_TECHNICAL_OUT_SLOT = 5


def main() -> int:
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data["nodes"]
    links = data["links"]

    existing = [n for n in nodes if n.get("type") == "OTR_StoryRoomExtract"]
    if existing:
        print(
            f"[w2g] OTR_StoryRoomExtract already present "
            f"(id={existing[0]['id']}); no-op."
        )
        return 0

    story_room = next(
        (n for n in nodes if n.get("type") == "OTR_StoryRoom"), None,
    )
    if story_room is None:
        print(
            "[w2g] FATAL: OTR_StoryRoom node not found. Wave 2 Agent F "
            "must land before Wave 2 Agent G can wire."
        )
        return 1
    story_room_node_id = story_room["id"]
    story_room_out_slot = 0

    next_node_id = max(n["id"] for n in nodes) + 1
    next_link_id = max(l[0] for l in links) + 1

    link_transcript = next_link_id
    link_technical = next_link_id + 1

    new_node = {
        "id": next_node_id,
        "type": "OTR_StoryRoomExtract",
        "pos": [
            1100,
            520,
        ],
        "size": [
            440,
            220,
        ],
        "title": "Story Room Extract (Sprint 10B Wave 2 Agent G, DORMANT)",
        "flags": {},
        "order": 101,
        "mode": 0,
        "inputs": [
            {
                "name": "story_room_transcript",
                "type": "STRING",
                "link": link_transcript,
            },
            {
                "name": "cast_names",
                "type": "STRING",
                "link": None,
            },
            {
                "name": "news_seed",
                "type": "STRING",
                "link": None,
            },
            {
                "name": "technical_model",
                "type": "STRING",
                "link": link_technical,
                "shape": 7,
            },
        ],
        "outputs": [
            {
                "name": "story_room_extraction",
                "type": "STRING",
                # Disconnected today; Wave 3 wires the bridge into the
                # downstream consumers when use_story_room flips on.
                "links": None,
                "slot_index": 0,
            },
        ],
        "properties": {
            "Node name for S&R": "OTR_StoryRoomExtract",
            "cnr_id": "OldTimeRadio",
        },
        # Widget vector mirrors INPUT_TYPES required widget order.
        # `story_room_transcript` is the multiline STRING widget;
        # `cast_names` + `news_seed` are optional/forceInput so they
        # do not have widget slots.
        "widgets_values": [
            "",
        ],
        "color": "#2B3C5A",
        "bgcolor": "#1C273C",
    }
    nodes.append(new_node)

    # Wire StoryRoom -> Extract.story_room_transcript.
    links.append([
        link_transcript,
        story_room_node_id,
        story_room_out_slot,
        next_node_id,
        0,
        "STRING",
    ])
    sr_out = story_room["outputs"][story_room_out_slot]
    if sr_out.get("links") is None:
        sr_out["links"] = []
    sr_out["links"].append(link_transcript)

    # Wire Writer.technical_model -> Extract.technical_model.
    links.append([
        link_technical,
        WRITER_NODE_ID,
        WRITER_TECHNICAL_OUT_SLOT,
        next_node_id,
        3,
        "STRING",
    ])
    writer = next(n for n in nodes if n["id"] == WRITER_NODE_ID)
    technical_out = writer["outputs"][WRITER_TECHNICAL_OUT_SLOT]
    if technical_out.get("links") is None:
        technical_out["links"] = []
    technical_out["links"].append(link_technical)

    if "last_node_id" in data:
        data["last_node_id"] = max(data["last_node_id"], next_node_id)
    if "last_link_id" in data:
        data["last_link_id"] = max(data["last_link_id"], link_technical)

    with open(WORKFLOW_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(
        f"[w2g] wired OTR_StoryRoomExtract as node {next_node_id}; "
        f"new links {link_transcript}/{link_technical}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
