"""BUG-LOCAL-086 workflow patcher.

Adds:
  1. New `flux_done_gate` IMAGE input slot to OTR_BatchHumoRender (node 51).
  2. New link 83: UnloadAll (node 24).image (slot 0) -> BatchHumoRender (node 51).flux_done_gate (slot 7).
  3. Bumps last_link_id 82 -> 83.
  4. Appends link 83 to UnloadAll.outputs[0].links list.

Idempotent -- detects an existing flux_done_gate input and bails clean.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

WF = Path(r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\workflows\otr_scifi_16gb_full.json")


def main() -> int:
    wf = json.loads(WF.read_text(encoding="utf-8"))
    nodes_by_id = {n["id"]: n for n in wf["nodes"]}

    humo = nodes_by_id.get(51)
    if humo is None:
        print("FATAL: node 51 (OTR_BatchHumoRender) not found")
        return 1
    unload = nodes_by_id.get(24)
    if unload is None:
        print("FATAL: node 24 (OTR_UnloadAll) not found")
        return 2

    # Idempotency check
    for inp in humo.get("inputs", []):
        if inp.get("name") == "flux_done_gate":
            print("idempotent: flux_done_gate input already present on node 51 -- no-op")
            return 0

    # 1. New link id
    new_link_id = wf["last_link_id"] + 1
    print(f"new link id: {new_link_id}")

    # 2. Append new input to humo.inputs
    new_input = {
        "name": "flux_done_gate",
        "type": "IMAGE",
        "link": new_link_id,
        "shape": 7,
    }
    humo["inputs"].append(new_input)
    new_slot = len(humo["inputs"]) - 1
    print(f"added input slot {new_slot} to node 51 (BatchHumoRender)")

    # 3. Append new link
    new_link = [new_link_id, 24, 0, 51, new_slot, "IMAGE"]
    wf["links"].append(new_link)
    print(f"added link: {new_link}")

    # 4. Append link id to UnloadAll.outputs[0].links
    ua_out = unload["outputs"][0]
    if ua_out.get("links") is None:
        ua_out["links"] = []
    ua_out["links"].append(new_link_id)
    print(f"unloadall.outputs[0].links now: {ua_out['links']}")

    # 5. Bump last_link_id
    wf["last_link_id"] = new_link_id

    # Write back
    WF.write_text(json.dumps(wf, indent=2), encoding="utf-8")
    print(f"OK -- wrote {WF}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
