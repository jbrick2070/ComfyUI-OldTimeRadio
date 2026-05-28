"""Wave 3 wire fix-up: update the cascade's script_json input link.

When the original wire script redirected link 107 to target the
Commit node, it forgot to update the cascade's script_json input
record from link 107 -> link 225 (the new Commit -> Cascade link).
The two workflow consistency tests catch this drift.
"""
from __future__ import annotations

import json
import sys

WORKFLOW_PATH = "workflows/otr_scifi_16gb_full.json"


def main() -> int:
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    commit = next(
        (n for n in data["nodes"] if n.get("type") == "OTR_StoryRoomCommit"),
        None,
    )
    if commit is None:
        print("[w3-fixup] FATAL: OTR_StoryRoomCommit missing.")
        return 1
    commit_out_links = commit["outputs"][0].get("links") or []
    if not commit_out_links:
        print("[w3-fixup] FATAL: Commit output has no links.")
        return 1
    new_link_id = commit_out_links[0]

    # Find the cascade by walking the links table for the new link's
    # destination.
    for link in data["links"]:
        lid, src, src_slot, dst, dst_slot, ltype = link
        if lid == new_link_id:
            cascade_id = dst
            dst_slot_used = dst_slot
            break
    else:
        print("[w3-fixup] FATAL: new commit link not found in links.")
        return 1

    cascade = next(n for n in data["nodes"] if n["id"] == cascade_id)
    target_input = cascade["inputs"][dst_slot_used]
    old_link = target_input.get("link")
    target_input["link"] = new_link_id
    print(
        f"[w3-fixup] cascade {cascade_id}.{target_input['name']} "
        f"input link {old_link} -> {new_link_id}"
    )

    with open(WORKFLOW_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
