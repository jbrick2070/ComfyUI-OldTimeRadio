"""One-shot patcher: remove the in-graph HuMo demo branch (nodes
26-41) and the orphaned Sage HuMo patch (node 43) from the FULL
workflow JSON.

Why:
    The OTR_PostAudioVideoPipeline node (id=44, BUG-LOCAL-076) handles
    real per-line HuMo batching against the actual episode audio via a
    subprocess. The in-graph HuMo demo at nodes 26-41 was a one-shot
    smoke that read a static humo_test.wav and saved to a static
    filename_prefix, overwriting the previous demo on every queue.
    With the post-audio pipeline shipped, the in-graph demo is
    redundant for production AND wastes ~4 minutes per queue rendering
    a clip nobody uses.

What this removes:
    - Nodes 26-41 (the entire in-graph HuMo branch:
      ImageFromBatch -> UNETLoader -> LoraLoaderModelOnly ->
      ModelSamplingSD3 -> CLIPLoader -> CLIPTextEncode (pos+neg) ->
      VAELoader -> LoadAudio -> AudioEncoderLoader -> AudioEncoderEncode
      -> WanHuMoImageToVideo -> KSampler -> VAEDecode -> CreateVideo ->
      SaveVideo).
    - Node 43 (PathchSageAttentionKJ HuMo, DISABLED per BUG-070) which
      becomes orphaned once nodes 26-41 are gone (its input from id=29
      and its output to id=38 both reference deleted nodes).
    - All links that reference any of the removed node IDs as either
      source or destination.

What this keeps:
    - The audio path (story writer / director / scene sequencer /
      bark / kokoro / musicgen / audiogen / episode assembler / audio
      enhance / signal lost video).
    - The FLUX environment stills path (FLUX checkpoint loader /
      BatchFluxRender / UnloadAll / SaveImage). FLUX -> ImageFromBatch
      link (68) is severed; FLUX's other outputs (to UnloadAll and
      SaveImage) remain intact.
    - Node 42 (PathchSageAttentionKJ FLUX, DISABLED per BUG-070) is
      independent of the HuMo branch -- stays.
    - OTR_VideoPlan / OTR_ShotDurationCalculator (the read-only
      Director/script adapters used downstream by FLUX).
    - OTR_PostAudioVideoPipeline (id=44) -- the new node that handles
      real HuMo batching post-audio.

Idempotent: re-running with the nodes already removed is a no-op.
last_node_id / last_link_id are NOT decremented (they're high water
marks, not the live count) so re-running this patcher does not
collide with any future node insertions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"

# IDs to remove. 26-41 is the in-graph HuMo demo; 43 is the orphaned
# Sage HuMo patch.
REMOVE_NODE_IDS = set(range(26, 42)) | {43}


def main() -> int:
    if not WORKFLOW.exists():
        print(f"FATAL: workflow not found: {WORKFLOW}", file=sys.stderr)
        return 1

    with open(WORKFLOW, "r", encoding="utf-8") as f:
        wf = json.load(f)

    nodes = wf.get("nodes") or []
    links = wf.get("links") or []

    # Idempotency check
    present = [n["id"] for n in nodes if n["id"] in REMOVE_NODE_IDS]
    if not present:
        print("In-graph HuMo demo already removed; no-op.")
        return 0

    # 1. Remove nodes
    nodes_kept = [n for n in nodes if n["id"] not in REMOVE_NODE_IDS]
    n_removed = len(nodes) - len(nodes_kept)

    # 2. Remove links that touch any removed node
    #    Link format: [link_id, src_node_id, src_slot, dst_node_id, dst_slot, type]
    links_kept = [
        L for L in links
        if (L[1] not in REMOVE_NODE_IDS and L[3] not in REMOVE_NODE_IDS)
    ]
    removed_link_ids = {L[0] for L in links if L not in links_kept}
    n_links_removed = len(links) - len(links_kept)

    # 3. Scrub references to removed link IDs from any kept node's
    #    outputs[].links list. This handles the case where FLUX (id=23)
    #    output[0] still listed link 68 (which is now gone).
    for n in nodes_kept:
        for out in n.get("outputs") or []:
            ls = out.get("links")
            if ls is None:
                continue
            cleaned = [lid for lid in ls if lid not in removed_link_ids]
            if cleaned != ls:
                out["links"] = cleaned if cleaned else None

    # 4. Update workflow
    wf["nodes"] = nodes_kept
    wf["links"] = links_kept
    # last_node_id / last_link_id are high water marks, NOT live
    # counts. Do not decrement them; future inserts pick id+1 from the
    # high water mark and never collide with the removed range.

    # 5. Per BUG-LOCAL-069: every kept node must still carry flags /
    #    order / mode. Validate.
    required = {"flags", "order", "mode"}
    missing = [(n["id"], n.get("type"), f) for n in nodes_kept
               for f in required if f not in n]
    if missing:
        print(f"FATAL: kept nodes missing required fields: {missing}",
              file=sys.stderr)
        return 4

    # 6. Atomic write
    tmp = WORKFLOW.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(wf, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(WORKFLOW)

    print(
        f"OK: removed {n_removed} nodes (ids "
        f"{sorted(REMOVE_NODE_IDS & {n['id'] for n in nodes})}) and "
        f"{n_links_removed} links. {len(nodes_kept)} nodes / "
        f"{len(links_kept)} links remain. "
        f"last_node_id={wf.get('last_node_id')}, "
        f"last_link_id={wf.get('last_link_id')} unchanged."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
