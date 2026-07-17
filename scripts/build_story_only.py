"""Generate workflows/otr_story_only.json from the CURRENT canonical.

The story-only graph is the writer path of the canonical workflow with the
media tail removed -- used for fast STORY-QUALITY scoring (bake-offs, variant
comparisons) where only the transcript/ledger matters, not the rendered video.

It keeps exactly three nodes:
  63  OTR_WorkflowValidator   (gate; validate_anyway=False for a diagnostic load)
   1  OTR_LedgerScriptWriter  (writes the story ledger + episode_canon)
  62  OTR_LedgerFreezeCascade (finalises/freezes the ledger; OUTPUT_NODE=True so
                               it terminates the stripped graph)

and only the six links that wire them (63->1 gate_in; 1->62 x5). Every other
node and link is dropped, and dangling link references inside the kept nodes are
pruned. Regenerate this whenever the canonical writer/freeze wiring changes:

    python scripts/build_story_only.py

This is a DERIVED scoring artifact, not a competing production workflow: real
episode renders still load workflows/otr_canonical.json.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CANON = REPO / "workflows" / "otr_canonical.json"
DST = REPO / "workflows" / "otr_story_only.json"

KEEP_NODES = {63, 1, 62}
KEEP_LINKS = {279, 106, 230, 108, 109, 115}  # 63->1 gate_in ; 1->62 (5 outputs)


def build() -> dict:
    canon = json.loads(CANON.read_text(encoding="utf-8"))
    out = copy.deepcopy(canon)
    out["nodes"] = [n for n in canon["nodes"] if n["id"] in KEEP_NODES]
    out["links"] = [l for l in canon["links"] if l[0] in KEEP_LINKS]

    missing_n = KEEP_NODES - {n["id"] for n in out["nodes"]}
    missing_l = KEEP_LINKS - {l[0] for l in out["links"]}
    if missing_n or missing_l:
        raise SystemExit(
            f"canonical no longer has the story path (missing nodes {missing_n}, "
            f"links {missing_l}); re-derive KEEP_NODES/KEEP_LINKS from the current "
            "writer->freeze wiring."
        )

    for n in out["nodes"]:
        for inp in n.get("inputs", []) or []:
            if inp.get("link") is not None and inp["link"] not in KEEP_LINKS:
                inp["link"] = None
        for outp in n.get("outputs", []) or []:
            if outp.get("links"):
                outp["links"] = [lid for lid in outp["links"] if lid in KEEP_LINKS]
        if n["type"] == "OTR_WorkflowValidator":
            wv = n.get("widgets_values")
            if isinstance(wv, list) and len(wv) >= 2:
                wv[1] = False  # validate_anyway (positional index 1)
            elif isinstance(wv, dict):
                wv["validate_anyway"] = False

    out["last_node_id"] = max(n["id"] for n in out["nodes"])
    out["last_link_id"] = max((l[0] for l in out["links"]), default=0)

    ids = {n["id"] for n in out["nodes"]}
    for l in out["links"]:
        assert l[1] in ids and l[3] in ids, f"link {l} references a dropped node"
    return out


def main() -> int:
    out = build()
    DST.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {DST} (nodes={len(out['nodes'])} links={len(out['links'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
