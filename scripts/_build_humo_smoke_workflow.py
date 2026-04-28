"""Build a minimal HuMo-only smoke workflow JSON.

Skips the audio path + FLUX entirely. Loads:
  - existing audio episode mp4 via LoadAudio
  - HuMo loader chain (UNETLoader + Lora + ModelSamplingSD3 +
    CLIPLoader + VAELoader + AudioEncoderLoader)
  - OTR_BatchHumoRender (with hardcoded ledger path string)
  - OTR_VideoComposite

Usage:
  python scripts/_build_humo_smoke_workflow.py

Writes workflows/otr_humo_smoke.json. Default fixture is the
Resonance Chamber run (signal_lost_resonance_chamber_20260427_164718).
Edit FIXTURE_LEDGER_MP4 below to point at any other episode mp4.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "workflows" / "otr_humo_smoke.json"

# Fixture episode -- BatchHumoRender + VideoComposite both auto-swap
# the .mp4 suffix to _ledger.json internally.
FIXTURE_LEDGER_MP4 = (
    r"C:\Users\jeffr\Documents\ComfyUI\output\otr\audio"
    r"\signal_lost_resonance_chamber_20260427_164718.mp4"
)


def _node(node_id, ntype, pos, size, inputs=None, outputs=None,
          widgets_values=None, title=None, color="#1E3A5F", bgcolor="#13253F"):
    """Canonical ComfyUI node shape (BUG-LOCAL-069 schema-compliant)."""
    return {
        "id": node_id,
        "type": ntype,
        "pos": pos,
        "size": size,
        "flags": {},
        "order": node_id,  # use id as initial order; topology resolves at runtime
        "mode": 0,
        "title": title or ntype,
        "inputs": inputs or [],
        "outputs": outputs or [],
        "properties": {"Node name for S&R": ntype},
        "widgets_values": widgets_values or [],
        "color": color,
        "bgcolor": bgcolor,
    }


def main() -> int:
    nodes = []
    links = []
    next_link_id = [1]

    def add_link(src_id, src_slot, dst_id, dst_slot, link_type):
        lid = next_link_id[0]
        next_link_id[0] += 1
        links.append([lid, src_id, src_slot, dst_id, dst_slot, link_type])
        # Register on src node's output
        for n in nodes:
            if n["id"] == src_id:
                outs = n.get("outputs") or []
                if src_slot < len(outs):
                    if outs[src_slot].get("links") is None:
                        outs[src_slot]["links"] = []
                    outs[src_slot]["links"].append(lid)
        return lid

    # 1. LoadAudio -- reads the Resonance Chamber mp4 from input/.
    # NOTE: ComfyUI's LoadAudio expects the file in input/ dir. For
    # our smoke we'll use the absolute path approach -- the audio
    # input on BatchHumoRender accepts the AUDIO type directly.
    # Easier: convert via ffmpeg? Actually LoadAudio supports paths.
    nodes.append(_node(
        node_id=1, ntype="LoadAudio",
        title="Load episode audio (Resonance Chamber)",
        pos=[100, 100], size=[400, 100],
        outputs=[{"name": "AUDIO", "type": "AUDIO", "links": None,
                  "shape": 3, "slot_index": 0}],
        widgets_values=[FIXTURE_LEDGER_MP4],  # full path; LoadAudio handles abs paths in modern Comfy
    ))

    # 2. UNETLoader (HuMo)
    nodes.append(_node(
        node_id=2, ntype="UNETLoader",
        title="Load HuMo 14B fp8 e4m3fn scaled (Kijai)",
        pos=[100, 240], size=[420, 100],
        outputs=[{"name": "MODEL", "type": "MODEL", "links": None,
                  "shape": 3, "slot_index": 0}],
        widgets_values=[
            "Wan2_1-HuMo-14B_fp8_e4m3fn_scaled_KJ.safetensors",
            "default",
        ],
    ))

    # 3. LoraLoaderModelOnly (lightx2v)
    nodes.append(_node(
        node_id=3, ntype="LoraLoaderModelOnly",
        title="lightx2v 4-step distill",
        pos=[100, 380], size=[420, 80],
        inputs=[{"name": "model", "type": "MODEL", "link": None}],
        outputs=[{"name": "MODEL", "type": "MODEL", "links": None,
                  "shape": 3, "slot_index": 0}],
        widgets_values=[
            "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            1.0,
        ],
    ))
    add_link(2, 0, 3, 0, "MODEL")

    # 4. ModelSamplingSD3 (shift=8)
    nodes.append(_node(
        node_id=4, ntype="ModelSamplingSD3",
        title="ModelSamplingSD3 shift=8",
        pos=[100, 500], size=[420, 60],
        inputs=[{"name": "model", "type": "MODEL", "link": None}],
        outputs=[{"name": "MODEL", "type": "MODEL", "links": None,
                  "shape": 3, "slot_index": 0}],
        widgets_values=[8.0],
    ))
    add_link(3, 0, 4, 0, "MODEL")

    # 5. CLIPLoader (umt5_xxl)
    nodes.append(_node(
        node_id=5, ntype="CLIPLoader",
        title="Load umt5_xxl text encoder",
        pos=[100, 600], size=[420, 80],
        outputs=[{"name": "CLIP", "type": "CLIP", "links": None,
                  "shape": 3, "slot_index": 0}],
        widgets_values=[
            "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default",
        ],
    ))

    # 6. VAELoader (wan_2.1)
    nodes.append(_node(
        node_id=6, ntype="VAELoader",
        title="Load wan_2.1 VAE",
        pos=[100, 720], size=[420, 60],
        outputs=[{"name": "VAE", "type": "VAE", "links": None,
                  "shape": 3, "slot_index": 0}],
        widgets_values=["wan_2.1_vae.safetensors"],
    ))

    # 7. AudioEncoderLoader (whisper)
    nodes.append(_node(
        node_id=7, ntype="AudioEncoderLoader",
        title="Load Whisper Large v3",
        pos=[100, 820], size=[420, 60],
        outputs=[{"name": "AUDIO_ENCODER", "type": "AUDIO_ENCODER",
                  "links": None, "shape": 3, "slot_index": 0}],
        widgets_values=["whisper_large_v3_fp16.safetensors"],
    ))

    # 8. OTR_BatchHumoRender
    # widgets_values canonical (per BUG-LOCAL-079):
    #   [0]  ledger_json placeholder (link wins)  -- here we use a
    #        widget value with the .mp4 path; node auto-suffix-swaps
    #        to _ledger.json.  No link-wired input for ledger_json
    #        in this smoke (simpler topology), so the widget value
    #        is the source of truth.
    #   [1]  portraits_dir placeholder ("" -> auto-resolve)
    #   [2]  clip_length 7.0
    #   [3]  max_clips 4 (smoke -- 4 clips, not 30)
    #   [4]  seed 7
    #   [5]  control_after_generate "randomize"
    #   [6]  steps 6
    #   [7]  cfg 1.0
    #   [8]  sampler_name "uni_pc"
    #   [9]  scheduler "simple"
    #   [10] width 480
    #   [11] height 832
    nodes.append(_node(
        node_id=8, ntype="OTR_BatchHumoRender",
        title="Batch HuMo Render (smoke -- 4 clips)",
        pos=[600, 240], size=[480, 460],
        inputs=[
            {"name": "model", "type": "MODEL", "link": None},
            {"name": "clip", "type": "CLIP", "link": None},
            {"name": "vae", "type": "VAE", "link": None},
            {"name": "audio_encoder", "type": "AUDIO_ENCODER", "link": None},
            {"name": "audio", "type": "AUDIO", "link": None},
            # ledger_json + portraits_dir intentionally NOT inputs --
            # we use widget values for the smoke (path string +
            # auto-resolve "")
        ],
        outputs=[
            {"name": "clips_dir", "type": "STRING", "links": None,
             "shape": 3, "slot_index": 0},
            {"name": "clip_count", "type": "INT", "links": None,
             "shape": 3, "slot_index": 1},
            {"name": "report", "type": "STRING", "links": None,
             "shape": 3, "slot_index": 2},
        ],
        widgets_values=[
            FIXTURE_LEDGER_MP4,  # ledger_json (mp4 path -> swap-suffix internally)
            "",                  # portraits_dir (auto-resolve)
            7.0,                 # clip_length
            4,                   # max_clips (smoke: 4 only)
            7,                   # seed
            "randomize",         # control_after_generate
            6,                   # steps
            1.0,                 # cfg
            "uni_pc",            # sampler_name
            "simple",            # scheduler
            480,                 # width
            832,                 # height
        ],
    ))
    add_link(4, 0, 8, 0, "MODEL")
    add_link(5, 0, 8, 1, "CLIP")
    add_link(6, 0, 8, 2, "VAE")
    add_link(7, 0, 8, 3, "AUDIO_ENCODER")
    add_link(1, 0, 8, 4, "AUDIO")

    # 9. OTR_VideoComposite
    nodes.append(_node(
        node_id=9, ntype="OTR_VideoComposite",
        title="Video Composite (1080p pillarbox + additive)",
        pos=[1100, 240], size=[480, 360],
        inputs=[
            {"name": "procgen_video_path", "type": "STRING", "link": None,
             "widget": {"name": "procgen_video_path"}},
            {"name": "clips_dir", "type": "STRING", "link": None,
             "widget": {"name": "clips_dir"}},
            # ledger_json kept as widget value (the mp4 path again)
        ],
        outputs=[
            {"name": "final_mp4_path", "type": "STRING", "links": None,
             "shape": 3, "slot_index": 0},
            {"name": "report", "type": "STRING", "links": None,
             "shape": 3, "slot_index": 1},
        ],
        widgets_values=[
            "",                  # procgen_video_path placeholder (link wins)
            "",                  # clips_dir placeholder (link wins)
            FIXTURE_LEDGER_MP4,  # ledger_json widget value
            "addition",
            0.5,
            1920, 1080, 25,
            1080, 7.0,
            "ffmpeg",
        ],
    ))
    # procgen_video_path <- the same mp4 path as fixture (use widget;
    # but VideoComposite expects a STRING input with link -- so we
    # need a tiny "PrimitiveNode" or "String" emitter to feed it).
    # Simpler for smoke: just hardcode via widget value, drop the
    # link, use widget="" which falls through to CONSTANT widget.
    # ACTUALLY easier: set link directly from BatchHumoRender's
    # clips_dir and use widget for procgen_video_path.
    add_link(8, 0, 9, 1, "STRING")  # clips_dir from BatchHumoRender

    # For procgen_video_path we'd want a String constant. ComfyUI
    # has a PrimitiveNode but it's in some packs. For the smoke,
    # leave procgen_video_path as widget value (no link). User
    # double-clicks the input to convert to widget if needed; or
    # we set widget="" and the widget value at slot 0 is used.
    # The widget value index 0 in widgets_values above is "" -- the
    # actual value the user sets in the widget. To make it work,
    # we keep procgen_video_path as a real widget (no link, no
    # widget: key in inputs). Drop the converted-to-input form.
    nodes[-1]["inputs"] = [
        # clips_dir wired from BatchHumoRender:
        {"name": "clips_dir", "type": "STRING", "link": None,
         "widget": {"name": "clips_dir"}},
    ]
    # Re-add the link from BatchHumoRender (link id 6 above pointed
    # at inputs[1] -- now there's only one input slot, so dst_slot=0)
    # Drop the prior add_link and re-add:
    # Easier: rewrite the entry's input link manually.
    nodes[-1]["inputs"][0]["link"] = links[-1][0]
    # Update the link's dst_slot from 1 to 0:
    links[-1][4] = 0

    # Reorder widgets_values for VideoComposite now that only
    # clips_dir is converted-to-input. The widget order in
    # INPUT_TYPES (after dropping link-converted ones from
    # placeholder list):
    #   [0] procgen_video_path (real widget)
    #   [1] -- clips_dir is converted-to-input, placeholder stays
    #   [2] ledger_json (real widget)
    #   [3] blend_mode
    #   ...
    # Per BUG-079: keep placeholders for converted-input slots.
    nodes[-1]["widgets_values"] = [
        FIXTURE_LEDGER_MP4,  # [0] procgen_video_path (real widget)
        "",                  # [1] clips_dir placeholder (link wins)
        FIXTURE_LEDGER_MP4,  # [2] ledger_json (real widget)
        "addition",
        0.5,
        1920, 1080, 25,
        1080, 7.0,
        "ffmpeg",
    ]

    wf = {
        "last_node_id": 9,
        "last_link_id": next_link_id[0] - 1,
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }

    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(wf, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"OK: wrote {OUT}")
    print(f"  fixture: {FIXTURE_LEDGER_MP4}")
    print(f"  nodes: {len(nodes)}, links: {len(links)}")
    print(f"  max_clips: 4 (smoke; bump in widget for fuller test)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
