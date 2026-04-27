"""One-shot patcher: add OTR_PostAudioVideoPipeline node to the FULL
workflow JSON, wired to OTR_SignalLostVideo's video_path output.

Run once to migrate the FULL workflow. Idempotent: re-running with the
node already present is a no-op.

Adds:
    - new node id = last_node_id + 1 (44)
    - new link id = last_link_id + 1 (71)
    - link wires SignalLostVideo (id=12) output[0] (STRING video_path)
      -> PostAudioVideoPipeline.audio_or_ledger_path

Per BUG-LOCAL-069 the new node carries flags={}, order=max+1, mode=0
to satisfy ComfyUI's zod schema validator.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"


def _build_post_audio_node(node_id: int, order: int, link_id: int) -> dict:
    """Canonical ComfyUI node shape for OTR_PostAudioVideoPipeline.

    The audio_or_ledger_path widget is converted to a STRING input so
    SignalLostVideo's video_path can wire in. The remaining widgets
    stay as widget values.
    """
    return {
        "id": node_id,
        "type": "OTR_PostAudioVideoPipeline",
        "pos": [600, 1700],
        "size": [460, 240],
        "flags": {},
        "order": order,
        "mode": 0,
        "title": "9. Post-Audio Video Pipeline (HuMo + Concat)",
        "inputs": [
            {
                "name": "audio_or_ledger_path",
                "type": "STRING",
                "link": link_id,
                "widget": {"name": "audio_or_ledger_path"},
            },
        ],
        "outputs": [
            {
                "name": "status",
                "type": "STRING",
                "links": None,
                "shape": 3,
                "slot_index": 0,
            },
            {
                "name": "log_path",
                "type": "STRING",
                "links": None,
                "shape": 3,
                "slot_index": 1,
            },
        ],
        "properties": {"Node name for S&R": "OTR_PostAudioVideoPipeline"},
        # widgets_values order must match INPUT_TYPES declaration:
        #   audio_or_ledger_path, clip_length, max_clips,
        #   portrait_map, skip_pass1, mode
        "widgets_values": ["", 7.0, 0, "", True, "fire"],
        "color": "#1E3A5F",
        "bgcolor": "#13253F",
    }


def main() -> int:
    if not WORKFLOW.exists():
        print(f"FATAL: workflow not found: {WORKFLOW}", file=sys.stderr)
        return 1

    with open(WORKFLOW, "r", encoding="utf-8") as f:
        wf = json.load(f)

    nodes = wf.get("nodes") or []
    links = wf.get("links") or []

    # Idempotency check
    for n in nodes:
        if n.get("type") == "OTR_PostAudioVideoPipeline":
            print(
                f"OTR_PostAudioVideoPipeline already present (id={n.get('id')}); "
                f"no-op."
            )
            return 0

    # Find SignalLostVideo (id=12) for the source link
    src_node = None
    for n in nodes:
        if n.get("type") == "OTR_SignalLostVideo":
            src_node = n
            break
    if src_node is None:
        print("FATAL: OTR_SignalLostVideo not found in workflow", file=sys.stderr)
        return 2

    src_id = int(src_node["id"])
    src_slot = 0  # video_path is output[0]

    # Confirm the source output slot is what we expect
    src_outputs = src_node.get("outputs") or []
    if not src_outputs or (src_outputs[0].get("name") != "video_path"):
        print(
            f"FATAL: SignalLostVideo output[0] is not video_path "
            f"(got {src_outputs[0].get('name') if src_outputs else 'none'!r})",
            file=sys.stderr,
        )
        return 3

    # Allocate new ids
    new_node_id = int(wf.get("last_node_id", max(n["id"] for n in nodes))) + 1
    new_link_id = int(wf.get("last_link_id", max((L[0] for L in links), default=0))) + 1
    new_order = max((int(n.get("order", 0)) for n in nodes), default=0) + 1

    # Build node
    new_node = _build_post_audio_node(new_node_id, new_order, new_link_id)

    # Build link: [link_id, src_id, src_slot, dst_id, dst_slot, type]
    new_link = [new_link_id, src_id, src_slot, new_node_id, 0, "STRING"]

    # Append the new link to the source node's output[0].links list
    src_outputs[0].setdefault("links", [])
    if src_outputs[0]["links"] is None:
        src_outputs[0]["links"] = []
    if new_link_id not in src_outputs[0]["links"]:
        src_outputs[0]["links"].append(new_link_id)

    # Append new node + new link
    nodes.append(new_node)
    links.append(new_link)
    wf["nodes"] = nodes
    wf["links"] = links
    wf["last_node_id"] = new_node_id
    wf["last_link_id"] = new_link_id

    # Per BUG-LOCAL-069: every node must carry flags / order / mode.
    # Validate the inserted node before writing.
    required = {"flags", "order", "mode"}
    missing = [f for f in required if f not in new_node]
    if missing:
        print(f"FATAL: new node missing required fields: {missing}", file=sys.stderr)
        return 4

    # Atomic write
    tmp = WORKFLOW.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(wf, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(WORKFLOW)

    print(
        f"OK: added OTR_PostAudioVideoPipeline node id={new_node_id} "
        f"order={new_order}, link id={new_link_id} from "
        f"SignalLostVideo(id={src_id}) output 0 -> input 0. "
        f"last_node_id={new_node_id}, last_link_id={new_link_id}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
