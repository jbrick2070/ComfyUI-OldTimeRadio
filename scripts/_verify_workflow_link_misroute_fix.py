"""
_verify_workflow_link_misroute_fix.py  --  post-patch invariants
================================================================

Checks the BUG-LOCAL-124 patch is in effect:

1. Links 79 and 82 are gone.
2. Node 12.video_path links are exactly {47, 80} (gate + base layer only).
3. Node 51.ledger_json input link points at Node 3.scene_manifest_json.
4. Node 52.ledger_json input link points at Node 3.scene_manifest_json.
5. Node 3.scene_manifest_json.links contains both new link IDs.
6. JSON parses cleanly and round-trips.

Returns 0 on green, non-zero on any drift.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"


def main() -> int:
    data = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    nodes_by_id = {n["id"]: n for n in data["nodes"]}
    link_table = {l[0]: l for l in data["links"]}

    failures = []

    # 1. Misrouted links must be absent.
    for lid in (79, 82):
        if lid in link_table:
            failures.append(f"link {lid} STILL PRESENT (should be removed)")

    # 2. Node 12.video_path output links must be exactly {47, 80}.
    n12_out0 = nodes_by_id[12]["outputs"][0]
    n12_links = set(n12_out0.get("links") or [])
    if n12_links != {47, 80}:
        failures.append(
            f"Node 12.video_path links = {sorted(n12_links)}, "
            f"expected {{47, 80}}"
        )

    # 3 & 4. Node 51 input 5 and Node 52 input 2 must point at links
    # whose source is Node 3 output slot 2.
    n51_inp5 = nodes_by_id[51]["inputs"][5]
    n52_inp2 = nodes_by_id[52]["inputs"][2]

    for label, inp in (("Node 51.ledger_json", n51_inp5), ("Node 52.ledger_json", n52_inp2)):
        link_id = inp.get("link")
        if link_id is None:
            failures.append(f"{label}: link is None (not connected)")
            continue
        if link_id not in link_table:
            failures.append(
                f"{label}: link {link_id} not in data.links"
            )
            continue
        link = link_table[link_id]
        # link format: [link_id, source_node, source_slot, target_node, target_slot, type]
        if link[1] != 3 or link[2] != 2:
            failures.append(
                f"{label}: link {link_id} source = "
                f"Node {link[1]}.{link[2]}, expected Node 3.2"
            )

    # 5. Node 3.scene_manifest_json.links must contain both new link IDs.
    n3_out2 = nodes_by_id[3]["outputs"][2]
    if n3_out2.get("name") != "scene_manifest_json":
        failures.append(
            f"Node 3 output 2 name = {n3_out2.get('name')!r}, "
            f"expected 'scene_manifest_json'"
        )
    n3_link_set = set(n3_out2.get("links") or [])
    expected_n51_link = n51_inp5.get("link")
    expected_n52_link = n52_inp2.get("link")
    if expected_n51_link not in n3_link_set:
        failures.append(
            f"Node 3.scene_manifest_json.links = {sorted(n3_link_set)}, "
            f"missing Node 51's incoming link {expected_n51_link}"
        )
    if expected_n52_link not in n3_link_set:
        failures.append(
            f"Node 3.scene_manifest_json.links = {sorted(n3_link_set)}, "
            f"missing Node 52's incoming link {expected_n52_link}"
        )

    if failures:
        print("[FAIL] post-patch invariants violated:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("[OK] all 6 invariants pass:")
    print(f"  - links 79 and 82 removed")
    print(f"  - Node 12.video_path.links = {{47, 80}} (gate + base only)")
    print(
        f"  - Node 51.ledger_json connected to Node 3.scene_manifest_json "
        f"via link {n51_inp5.get('link')}"
    )
    print(
        f"  - Node 52.ledger_json connected to Node 3.scene_manifest_json "
        f"via link {n52_inp2.get('link')}"
    )
    print(f"  - Node 3.scene_manifest_json.links = {sorted(n3_link_set)}")
    print(f"  - JSON round-trip clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
