"""
_fix_workflow_link_misroute.py  --  BUG-LOCAL-124 surgical patch
================================================================

The workflow JSON had Node 12 (OTR_SignalLostVideo) .video_path output
wired into TWO inputs that expect a JSON ledger manifest, not a video
file path:

    Link 79: Node 12.0 -> Node 51.5 (BatchHumoRender.ledger_json)
    Link 82: Node 12.0 -> Node 52.2 (VideoComposite.ledger_json)

ComfyUI's STRING type doesn't differentiate file paths from JSON
strings, so the misroute slipped past type-checking. Downstream nodes
called json.loads() on a string that was actually an .mp4 path,
fell through to "no manifest" defaults: HuMo skipped scene cuts, no
radio still resolution, fallback chains kicked in incorrectly.

Fix:
1. Remove links 79 and 82.
2. Wire Node 3 (OTR_SceneSequencer) .scene_manifest_json (output slot 2)
   to both Node 51.ledger_json and Node 52.ledger_json.
3. Update Node 12.video_path output's link list to drop 79 and 82.
4. Update last_link_id.

Idempotent: if links 79/82 already removed, script no-ops with a clear
message.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"
LINKS_TO_REMOVE = {79, 82}


def main() -> int:
    data = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    nodes_by_id = {n["id"]: n for n in data["nodes"]}

    link_table = {l[0]: l for l in data["links"]}
    if not LINKS_TO_REMOVE.intersection(link_table.keys()):
        print(
            "[no-op] Neither link 79 nor link 82 is present -- assuming "
            "the patch has already been applied. Aborting cleanly."
        )
        return 0

    # Pre-flight: confirm misrouted links match expected shape.
    expected_shapes = {
        79: (12, 0, 51, 5, "STRING"),
        82: (12, 0, 52, 2, "STRING"),
    }
    for lid, expected in expected_shapes.items():
        if lid not in link_table:
            continue
        got = tuple(link_table[lid][1:6])
        if got != expected:
            print(
                f"[ABORT] Link {lid} shape mismatch. Expected "
                f"{expected}, got {got}."
            )
            return 1

    # Confirm Node 3 has scene_manifest_json on slot 2.
    n3 = nodes_by_id.get(3)
    if n3 is None or n3.get("type") != "OTR_SceneSequencer":
        print(f"[ABORT] Node 3 is missing or wrong type")
        return 1
    n3_outputs = n3.get("outputs", [])
    if (
        len(n3_outputs) <= 2
        or n3_outputs[2].get("name") != "scene_manifest_json"
    ):
        print("[ABORT] Node 3 output slot 2 is not scene_manifest_json")
        return 1

    # Confirm Node 51 input 5 is ledger_json.
    n51 = nodes_by_id.get(51)
    if n51 is None or n51.get("type") != "OTR_BatchHumoRender":
        print("[ABORT] Node 51 missing or wrong type")
        return 1
    n51_inputs = n51.get("inputs", [])
    if (
        len(n51_inputs) <= 5
        or n51_inputs[5].get("name") != "ledger_json"
    ):
        print("[ABORT] Node 51 input slot 5 is not ledger_json")
        return 1

    # Confirm Node 52 input 2 is ledger_json.
    n52 = nodes_by_id.get(52)
    if n52 is None or n52.get("type") != "OTR_VideoComposite":
        print("[ABORT] Node 52 missing or wrong type")
        return 1
    n52_inputs = n52.get("inputs", [])
    if (
        len(n52_inputs) <= 2
        or n52_inputs[2].get("name") != "ledger_json"
    ):
        print("[ABORT] Node 52 input slot 2 is not ledger_json")
        return 1

    print("[patch] preconditions OK -- applying link surgery")

    # Step 1: drop links 79 and 82 from the global links list.
    data["links"] = [l for l in data["links"] if l[0] not in LINKS_TO_REMOVE]
    print(f"[patch] removed links {sorted(LINKS_TO_REMOVE)} from data.links")

    # Step 2: clear the input link refs.
    n51_inputs[5]["link"] = None
    n52_inputs[2]["link"] = None
    print("[patch] cleared Node 51 input 5 and Node 52 input 2 link refs")

    # Step 3: drop links 79 and 82 from Node 12.video_path output.
    n12 = nodes_by_id.get(12)
    if n12 is None:
        print("[ABORT] Node 12 missing")
        return 1
    n12_out0 = n12.get("outputs", [])[0]
    if n12_out0.get("name") != "video_path":
        print("[ABORT] Node 12 output 0 is not video_path")
        return 1
    old_links = list(n12_out0.get("links") or [])
    n12_out0["links"] = [l for l in old_links if l not in LINKS_TO_REMOVE]
    print(
        f"[patch] Node 12.video_path links: {old_links} -> "
        f"{n12_out0['links']}"
    )

    # Step 4: add new links.
    next_link_id = max((l[0] for l in data["links"]), default=0) + 1
    new_a = next_link_id
    new_b = next_link_id + 1
    data["links"].append([new_a, 3, 2, 51, 5, "STRING"])
    data["links"].append([new_b, 3, 2, 52, 2, "STRING"])
    print(f"[patch] added link {new_a}: Node 3.2 -> Node 51.5 (ledger_json)")
    print(f"[patch] added link {new_b}: Node 3.2 -> Node 52.2 (ledger_json)")

    # Step 5: update Node 3 output slot 2 links list.
    n3_out2 = n3_outputs[2]
    if n3_out2.get("links") is None:
        n3_out2["links"] = []
    n3_out2["links"].extend([new_a, new_b])
    print(f"[patch] Node 3.scene_manifest_json.links: {n3_out2['links']}")

    # Step 6: update Node 51 input 5 and Node 52 input 2.
    n51_inputs[5]["link"] = new_a
    n52_inputs[2]["link"] = new_b
    print("[patch] wired Node 51 input 5 and Node 52 input 2 to new links")

    # Step 7: update last_link_id.
    data["last_link_id"] = max(data.get("last_link_id", 0), new_b)
    print(f"[patch] last_link_id updated to {data['last_link_id']}")

    WORKFLOW.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] wrote patched workflow to {WORKFLOW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
