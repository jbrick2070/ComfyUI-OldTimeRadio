"""
_verify_bug124_reverted.py  --  confirm the BUG-124 patch is REVERTED
======================================================================

The QA report's recommended fix turned out to be wrong:
``scene_manifest_json`` is a stub output that's never populated, so
wiring BatchHumoRender to it produced a permanently-empty "[]" string
that crashed _load_ledger_with_path with "ledger path not found: []".

The OLD wiring (Node 12.video_path -> Node 51.ledger_json) was
actually a working pattern: BatchHumoRender's .mp4-stem fallback
chain explicitly handles that case.

This verifier confirms the ORIGINAL link state has been restored:
- Link 79: Node 12.0 -> Node 51.5 (ledger_json) -- PRESENT
- Link 82: Node 12.0 -> Node 52.2 (ledger_json) -- PRESENT
- Links 86, 87 (the broken patch): ABSENT
- Node 12.video_path.links: {47, 79, 80, 82}
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

    # Original misroute links MUST be present (they're working code).
    for lid, expected in (
        (79, (12, 0, 51, 5)),
        (82, (12, 0, 52, 2)),
    ):
        if lid not in link_table:
            failures.append(f"link {lid} missing (revert incomplete)")
            continue
        got = tuple(link_table[lid][1:5])
        if got != expected:
            failures.append(
                f"link {lid} shape = {got}, expected {expected}"
            )

    # Broken patch links MUST be absent.
    for lid in (86, 87):
        if lid in link_table:
            failures.append(
                f"link {lid} still present (broken patch not fully reverted)"
            )

    # Node 12.video_path.links must be exactly {47, 79, 80, 82}.
    n12_links = set(nodes_by_id[12]["outputs"][0].get("links") or [])
    if n12_links != {47, 79, 80, 82}:
        failures.append(
            f"Node 12.video_path.links = {sorted(n12_links)}, "
            f"expected {{47, 79, 80, 82}}"
        )

    if failures:
        print("[FAIL] revert incomplete:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("[OK] BUG-124 patch fully reverted -- workflow restored to working state:")
    print("  - link 79: Node 12.video_path -> Node 51.ledger_json (RESTORED)")
    print("  - link 82: Node 12.video_path -> Node 52.ledger_json (RESTORED)")
    print("  - links 86, 87 (broken patch): REMOVED")
    print("  - Node 12.video_path.links = {47, 79, 80, 82}")
    print(
        "  Note: BatchHumoRender's .mp4-stem fallback handles the mp4 "
        "path correctly via _load_ledger_with_path lines 1694-1756."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
