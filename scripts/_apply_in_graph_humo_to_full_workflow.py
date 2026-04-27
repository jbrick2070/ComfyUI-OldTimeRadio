"""One-shot patcher: wire the in-graph batch HuMo + composite chain
into otr_scifi_16gb_full.json.

After this runs, the FULL workflow contains:
    audio path (unchanged)
    FLUX environment stills (unchanged)
    NEW HuMo loader chain:
        UNETLoader (HuMo 14B fp8 e4m3fn scaled, Kijai)
        LoraLoaderModelOnly (lightx2v 4-step distill)
        ModelSamplingSD3 (shift=8)
        CLIPLoader (umt5_xxl)
        VAELoader (wan_2.1)
        AudioEncoderLoader (whisper_large_v3_fp16)
    OTR_BatchHumoRender
        model     <- ModelSamplingSD3
        clip      <- CLIPLoader
        vae       <- VAELoader
        audio_encoder <- AudioEncoderLoader
        audio     <- EpisodeAssembler.episode_audio
        ledger_json   <- SignalLostVideo.video_path  (suffix-swap inside node)
        portraits_dir <- "" (auto-resolve to ComfyUI output root)
    OTR_UnloadAll  (between HuMo and Composite for VRAM hygiene)
    OTR_VideoComposite
        procgen_video_path <- SignalLostVideo.video_path
        clips_dir          <- BatchHumoRender.clips_dir
        ledger_json        <- SignalLostVideo.video_path  (suffix-swap inside node)

OTR_PostAudioVideoPipeline (id=44, the subprocess trigger) is REMOVED
since the batch is now in-graph.

Per BUG-LOCAL-069: every new node carries flags={}, order=N+1, mode=0.
Idempotent: re-running with the new chain already present is a no-op.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"


def _next_id(wf: dict) -> int:
    nodes = wf.get("nodes") or []
    return int(wf.get("last_node_id", max((n["id"] for n in nodes), default=0))) + 1


def _next_link_id(wf: dict) -> int:
    links = wf.get("links") or []
    return int(wf.get("last_link_id", max((L[0] for L in links), default=0))) + 1


def _add_link(wf: dict, src_id: int, src_slot: int, dst_id: int, dst_slot: int, link_type: str) -> int:
    """Append a link entry, register it on the source node's outputs[].links list."""
    new_id = _next_link_id(wf)
    wf["links"].append([new_id, src_id, src_slot, dst_id, dst_slot, link_type])
    wf["last_link_id"] = new_id

    # Register on src outputs[].links
    src_node = next((n for n in wf["nodes"] if n["id"] == src_id), None)
    if src_node is not None:
        outs = src_node.get("outputs") or []
        if src_slot < len(outs):
            ls = outs[src_slot].get("links")
            if ls is None:
                outs[src_slot]["links"] = [new_id]
            else:
                ls.append(new_id)
    return new_id


def _add_node(
    wf: dict, *, ntype: str, title: str, pos: list[int], size: list[int],
    inputs: list[dict] | None = None,
    outputs: list[dict] | None = None,
    widgets_values: list | None = None,
    properties: dict | None = None,
    color: str = "#1E3A5F", bgcolor: str = "#13253F",
) -> dict:
    nid = _next_id(wf)
    order = max((int(n.get("order", 0)) for n in wf["nodes"]), default=0) + 1
    node = {
        "id": nid,
        "type": ntype,
        "pos": pos,
        "size": size,
        "flags": {},
        "order": order,
        "mode": 0,
        "title": title,
        "inputs": inputs or [],
        "outputs": outputs or [],
        "properties": properties or {"Node name for S&R": ntype},
        "widgets_values": widgets_values or [],
        "color": color,
        "bgcolor": bgcolor,
    }
    wf["nodes"].append(node)
    wf["last_node_id"] = nid
    return node


def _find_node(wf: dict, ntype: str) -> dict | None:
    for n in wf["nodes"]:
        if n.get("type") == ntype:
            return n
    return None


def main() -> int:
    if not WORKFLOW.exists():
        print(f"FATAL: workflow not found: {WORKFLOW}", file=sys.stderr)
        return 1

    with open(WORKFLOW, "r", encoding="utf-8") as f:
        wf = json.load(f)

    # ----- Idempotency: bail if BatchHumoRender already present -----
    if _find_node(wf, "OTR_BatchHumoRender") is not None:
        print("OTR_BatchHumoRender already present; no-op.")
        return 0

    # ----- Locate upstream nodes we need to wire from -----
    sig_video = _find_node(wf, "OTR_SignalLostVideo")
    if sig_video is None:
        print("FATAL: OTR_SignalLostVideo not found", file=sys.stderr)
        return 2
    ep_assembler = _find_node(wf, "OTR_EpisodeAssembler")
    if ep_assembler is None:
        print("FATAL: OTR_EpisodeAssembler not found", file=sys.stderr)
        return 2

    # ----- Remove old PostAudioVideoPipeline (node 44) -----
    old_post = _find_node(wf, "OTR_PostAudioVideoPipeline")
    if old_post is not None:
        old_id = int(old_post["id"])
        # Remove the node
        wf["nodes"] = [n for n in wf["nodes"] if n["id"] != old_id]
        # Remove links touching it
        kept_links = [L for L in wf["links"] if L[1] != old_id and L[3] != old_id]
        removed_link_ids = {L[0] for L in wf["links"] if L not in kept_links}
        wf["links"] = kept_links
        # Scrub references from kept nodes' outputs[].links
        for n in wf["nodes"]:
            for out in n.get("outputs") or []:
                ls = out.get("links")
                if ls is None:
                    continue
                cleaned = [lid for lid in ls if lid not in removed_link_ids]
                out["links"] = cleaned if cleaned else None
        print(f"  removed OTR_PostAudioVideoPipeline (id={old_id})")

    # ----- Add HuMo loader chain -----
    # Layout: column at x=2200 (right of FLUX path), descending y
    x_col = 2200
    y = 200

    # 1. UNETLoader (HuMo 14B fp8)
    unet = _add_node(wf,
        ntype="UNETLoader",
        title="Load HuMo 14B fp8 e4m3fn scaled (Kijai)",
        pos=[x_col, y], size=[420, 100],
        outputs=[{"name": "MODEL", "type": "MODEL", "links": None, "shape": 3, "slot_index": 0}],
        widgets_values=["Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors", "default"],
    )
    y += 140

    # 2. LoraLoaderModelOnly (lightx2v)
    lora = _add_node(wf,
        ntype="LoraLoaderModelOnly",
        title="lightx2v 4-step distill",
        pos=[x_col, y], size=[420, 80],
        inputs=[{"name": "model", "type": "MODEL", "link": None}],
        outputs=[{"name": "MODEL", "type": "MODEL", "links": None, "shape": 3, "slot_index": 0}],
        widgets_values=["lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", 1.0],
    )
    y += 120
    lora["inputs"][0]["link"] = _add_link(wf, unet["id"], 0, lora["id"], 0, "MODEL")

    # 3. ModelSamplingSD3 (shift=8)
    sampling = _add_node(wf,
        ntype="ModelSamplingSD3",
        title="ModelSamplingSD3 shift=8",
        pos=[x_col, y], size=[420, 60],
        inputs=[{"name": "model", "type": "MODEL", "link": None}],
        outputs=[{"name": "MODEL", "type": "MODEL", "links": None, "shape": 3, "slot_index": 0}],
        widgets_values=[8.0],
    )
    y += 100
    sampling["inputs"][0]["link"] = _add_link(wf, lora["id"], 0, sampling["id"], 0, "MODEL")

    # 4. CLIPLoader (umt5_xxl)
    clip_loader = _add_node(wf,
        ntype="CLIPLoader",
        title="Load umt5_xxl text encoder",
        pos=[x_col, y], size=[420, 80],
        outputs=[{"name": "CLIP", "type": "CLIP", "links": None, "shape": 3, "slot_index": 0}],
        widgets_values=["umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default"],
    )
    y += 120

    # 5. VAELoader (wan_2.1)
    vae_loader = _add_node(wf,
        ntype="VAELoader",
        title="Load wan_2.1 VAE",
        pos=[x_col, y], size=[420, 60],
        outputs=[{"name": "VAE", "type": "VAE", "links": None, "shape": 3, "slot_index": 0}],
        widgets_values=["wan_2.1_vae.safetensors"],
    )
    y += 100

    # 6. AudioEncoderLoader (whisper_large_v3_fp16)
    audio_loader = _add_node(wf,
        ntype="AudioEncoderLoader",
        title="Load Whisper Large v3",
        pos=[x_col, y], size=[420, 60],
        outputs=[{"name": "AUDIO_ENCODER", "type": "AUDIO_ENCODER", "links": None, "shape": 3, "slot_index": 0}],
        widgets_values=["whisper_large_v3_fp16.safetensors"],
    )
    y += 100

    # ----- OTR_BatchHumoRender -----
    humo = _add_node(wf,
        ntype="OTR_BatchHumoRender",
        title="Batch HuMo Render (in-graph batch)",
        pos=[x_col + 460, 200], size=[480, 460],
        inputs=[
            {"name": "model",         "type": "MODEL",         "link": None},
            {"name": "clip",          "type": "CLIP",          "link": None},
            {"name": "vae",           "type": "VAE",           "link": None},
            {"name": "audio_encoder", "type": "AUDIO_ENCODER", "link": None},
            {"name": "audio",         "type": "AUDIO",         "link": None},
            {"name": "ledger_json",   "type": "STRING",        "link": None, "widget": {"name": "ledger_json"}},
            {"name": "portraits_dir", "type": "STRING",        "link": None, "widget": {"name": "portraits_dir"}},
        ],
        outputs=[
            {"name": "clips_dir",  "type": "STRING", "links": None, "shape": 3, "slot_index": 0},
            {"name": "clip_count", "type": "INT",    "links": None, "shape": 3, "slot_index": 1},
            {"name": "report",     "type": "STRING", "links": None, "shape": 3, "slot_index": 2},
        ],
        # widgets_values: ComfyUI's runtime DOES NOT include
        # placeholders for widgets that have been converted to inputs
        # (widget: key + non-None link in inputs[]). Earlier patcher
        # versions wrote placeholders here and the values shifted by
        # the placeholder count, producing "cfg got 'uni_pc'" / "scheduler
        # got '480'" validation errors. Below: only ledger_json is
        # link-converted (so its placeholder is omitted); portraits_dir
        # is a real widget with link=None (placeholder stays as ""),
        # so the auto-resolve branch fires.
        widgets_values=[
            "",   # portraits_dir (real widget; "" -> node auto-resolve)
            7.0,  # clip_length
            0,    # max_clips (0 = unlimited)
            7,    # seed
            6,    # steps
            1.0,  # cfg
            "uni_pc", "simple",
            480, 832,
        ],
    )
    humo["inputs"][0]["link"] = _add_link(wf, sampling["id"],     0, humo["id"], 0, "MODEL")
    humo["inputs"][1]["link"] = _add_link(wf, clip_loader["id"],  0, humo["id"], 1, "CLIP")
    humo["inputs"][2]["link"] = _add_link(wf, vae_loader["id"],   0, humo["id"], 2, "VAE")
    humo["inputs"][3]["link"] = _add_link(wf, audio_loader["id"], 0, humo["id"], 3, "AUDIO_ENCODER")
    humo["inputs"][4]["link"] = _add_link(wf, ep_assembler["id"], 0, humo["id"], 4, "AUDIO")
    # ledger_json + portraits_dir wired from SignalLostVideo.video_path
    # via .mp4 -> _ledger.json suffix-swap inside the node
    humo["inputs"][5]["link"] = _add_link(wf, sig_video["id"], 0, humo["id"], 5, "STRING")
    # portraits_dir kept as widget "" -> node auto-resolves to ComfyUI
    # output root, which makes _find_portrait globs hit FLUX env stills.

    # ----- VRAM hygiene note (no UnloadAll node added here) -----
    # OTR_UnloadAll's existing class takes IMAGE input; an explicit
    # node between HuMo and Composite would need a STRING-passthrough
    # variant. Skipped because VideoComposite is ffmpeg-only (no GPU)
    # so it doesn't contend for VRAM with HuMo. The FLUX -> HuMo
    # handoff IS VRAM-critical, and the existing OTR_UnloadAll node
    # already sits between OTR_BatchFluxRender and SaveImage --
    # comfy.model_management.unload_all_models + soft_empty_cache
    # both fire there before HuMo loads. If a future composite step
    # needs VRAM, add a STRING-passthrough UnloadAll variant here.

    # ----- OTR_VideoComposite -----
    composite = _add_node(wf,
        ntype="OTR_VideoComposite",
        title="Video Composite (1080p pillarbox + additive proc gen)",
        pos=[x_col + 960, 200], size=[480, 360],
        inputs=[
            {"name": "procgen_video_path", "type": "STRING", "link": None, "widget": {"name": "procgen_video_path"}},
            {"name": "clips_dir",          "type": "STRING", "link": None, "widget": {"name": "clips_dir"}},
            {"name": "ledger_json",        "type": "STRING", "link": None, "widget": {"name": "ledger_json"}},
        ],
        outputs=[
            {"name": "final_mp4_path", "type": "STRING", "links": None, "shape": 3, "slot_index": 0},
            {"name": "report",         "type": "STRING", "links": None, "shape": 3, "slot_index": 1},
        ],
        # widgets_values: omit placeholders for inputs that are
        # widget-converted-to-link. All three (procgen_video_path,
        # clips_dir, ledger_json) have link references, so their
        # placeholders are NOT included. Only the 8 real widgets
        # (blend_mode through ffmpeg) get values.
        widgets_values=[
            "addition", 0.5,
            1920, 1080, 25,
            1080, 7.0,
            "ffmpeg",
        ],
    )
    composite["inputs"][0]["link"] = _add_link(wf, sig_video["id"], 0, composite["id"], 0, "STRING")
    composite["inputs"][1]["link"] = _add_link(wf, humo["id"],      0, composite["id"], 1, "STRING")
    composite["inputs"][2]["link"] = _add_link(wf, sig_video["id"], 0, composite["id"], 2, "STRING")

    # ----- Validate every node still carries flags/order/mode (BUG-069) -----
    required = {"flags", "order", "mode"}
    missing = [(n["id"], n.get("type"), k) for n in wf["nodes"]
               for k in required if k not in n]
    if missing:
        print(f"FATAL: nodes missing required schema fields: {missing}", file=sys.stderr)
        return 4

    # ----- Atomic write -----
    tmp = WORKFLOW.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(wf, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp.replace(WORKFLOW)

    print(f"OK: in-graph HuMo + Composite chain wired into {WORKFLOW.name}")
    print(f"  nodes: {len(wf['nodes'])}, links: {len(wf['links'])}")
    print(f"  last_node_id: {wf['last_node_id']}, last_link_id: {wf['last_link_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
