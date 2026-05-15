"""S31 B6 Fix 3 -- workflow JSON top-level `links[]` rows agree with
per-node `inputs[]` array indexes.

ComfyUI workflow JSONs hold connection data in TWO places:
1. Top-level `links[]` table: list of [link_id, src_node, src_slot,
   dst_node, dst_slot, type].
2. Per-node `inputs[]` arrays: each input slot has its own `link`
   field pointing back at a `link_id`.

The two tables should agree on which slot index each link targets
on the destination node. Pre-S31 B6 Fix 3, four rows in
`workflows/otr_scifi_16gb_full.json` had `dst_slot` off-by-one
relative to the actual position of that input in the target node's
`inputs[]` array:

  [14, 11, 0, 3, 2, "AUDIO"]   -> should be dst_slot=1
  [20, 13, 0, 3, 3, "AUDIO"]   -> should be dst_slot=2
  [25, 15, 0, 3, 4, "AUDIO"]   -> should be dst_slot=3
  [105, 14, 1, 12, 4, "AUDIO"] -> should be dst_slot=3

ComfyUI runtime probably routed correctly via the per-node arrays
(internally consistent there), but the top-level table was wrong --
including out-of-range slot indices that any new validator would
trip on.

This test enforces the agreement for every link row in every shipped
workflow JSON. Run on a freshly post-fix workflow set: zero
violations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


PACK_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_DIR = PACK_ROOT / "workflows"


def _workflow_paths() -> list[Path]:
    return sorted(WORKFLOWS_DIR.glob("*.json"))


@pytest.mark.parametrize("wf_path", _workflow_paths(),
                         ids=lambda p: p.name)
def test_workflow_link_rows_match_target_input_indexes(wf_path):
    """For every node input with `link=L`: find the top-level link row
    where `row[0] == L`, then assert `row[3] == target_node_id` AND
    `row[4] == actual_input_idx` AND `row[4] < len(target.inputs)`.

    Catches the off-by-one shape that Fix 3 corrected (and any
    future drift between the two link tables on any workflow).
    """
    wf = json.loads(wf_path.read_text(encoding="utf-8"))
    links_by_id = {row[0]: row for row in wf.get("links", [])}
    nodes_by_id = {n["id"]: n for n in wf.get("nodes", [])}

    violations: list[str] = []
    for node in wf.get("nodes", []):
        for input_idx, inp in enumerate(node.get("inputs", []) or []):
            link_id = inp.get("link")
            if link_id is None:
                continue
            if link_id not in links_by_id:
                violations.append(
                    f"node {node['id']} input[{input_idx}] "
                    f"({inp.get('name')}) references link={link_id} "
                    f"that has no top-level row"
                )
                continue
            row = links_by_id[link_id]
            # Row schema: [id, src_node, src_slot, dst_node, dst_slot, type]
            if len(row) < 5:
                violations.append(
                    f"link row {row} is malformed (<5 fields)"
                )
                continue
            row_dst_node = row[3]
            row_dst_slot = row[4]
            if row_dst_node != node["id"]:
                violations.append(
                    f"link {link_id}: top-level dst_node={row_dst_node} "
                    f"but per-node array says it targets node "
                    f"{node['id']} input[{input_idx}]"
                )
                continue
            if row_dst_slot != input_idx:
                violations.append(
                    f"link {link_id}: top-level dst_slot={row_dst_slot} "
                    f"but actual target slot is input[{input_idx}] "
                    f"(node {node['id']}, input name "
                    f"{inp.get('name')!r}). Off-by-{row_dst_slot - input_idx}."
                )
            target_node = nodes_by_id.get(row_dst_node)
            if target_node is not None:
                n_inputs = len(target_node.get("inputs", []) or [])
                if row_dst_slot >= n_inputs:
                    violations.append(
                        f"link {link_id}: dst_slot={row_dst_slot} is "
                        f"OUT OF RANGE for node {row_dst_node} "
                        f"(only {n_inputs} input slots)"
                    )

    assert not violations, (
        f"{wf_path.name} has link-row violations:\n  "
        + "\n  ".join(violations)
    )
