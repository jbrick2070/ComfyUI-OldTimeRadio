"""build_ltx_motion_workflow.py -- emit a ComfyUI workflow JSON that
batches multiple radio_bookend stills through one LTX motion render
each, all sharing the same loader / prompt / sigma chain so the
expensive setup (model load, T5 encode) happens ONCE.

Usage:
    python scripts\\build_ltx_motion_workflow.py
    python scripts\\build_ltx_motion_workflow.py --role announcer --stills 4
    python scripts\\build_ltx_motion_workflow.py --prompt "Custom prompt" --stills 6

Output:
    workflows\\ltx_motion_batch.json -- load this file in ComfyUI's UI
    (drag-drop or "Load" button), inspect / rearrange nodes if you
    want, then click "Queue Prompt". Each radio_bookend.png produces
    one animated WEBP (auto-plays inline in ComfyUI's node UI) plus
    one MP4.

Design (BUG-LOCAL-112 verification harness):
    Shared (loaded once -- expensive):
        LowVRAMCheckpointLoader, CLIPLoader (T5),
        CLIPTextEncode positive (motion prompt), CLIPTextEncode negative,
        LTXVConditioning, KSamplerSelect, ManualSigmas,
        EmptyLTXVLatentVideo, CFGGuider

    Per still (cheap -- N parallel chains):
        LoadImage, LTXVImgToVideoConditionOnly, RandomNoise,
        SamplerCustomAdvanced, VAEDecode, SaveAnimatedWEBP

License: MIT (matches OTR repo). Stock ComfyUI core nodes only;
no GPL nodes (no VHS).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# BUG-LOCAL-112 motion prompts (mirror nodes/batch_ltx_render._PROMPT_BY_ROLE).
PROMPTS = {
    "announcer": (
        "Continuous shot, same console throughout. Tuning dial needle "
        "sweeps rhythmically. Vacuum tubes pulse. Brass speaker grille "
        "trembles. Dust motes drift. Slow handheld dolly forward."
    ),
    "music_open": (
        "Continuous shot, same console throughout. Dial whip-pans across "
        "frequencies. Tube filaments ignite from cold to white-hot. "
        "Speaker grille vibrates aggressively. Dynamic dolly push forward."
    ),
    "music_close": (
        "Continuous shot, same console throughout. Dial settles. Tube "
        "filaments cool from white through deep amber. Smoke trails "
        "from cooling tubes. Slow dolly pull back."
    ),
    "music_inter": (
        "Continuous shot, same console throughout. Dial steady, glowing. "
        "Oscilloscope dances to the rhythm. VU meters bounce. Tubes "
        "pulse with the bass. Slow orbit around the speaker."
    ),
    "sfx": (
        "Continuous shot, same console throughout. Snap zoom on the "
        "dial as needle spikes hard. Tubes surge with electric arcs. "
        "Speaker grille rattles violently. Quick whip-pan to the dial."
    ),
}

# Render constants (mirror nodes/batch_ltx_render).
LTX_FPS = 25
LTX_WIDTH = 832
LTX_HEIGHT = 480
LTX_CFG = 1.0
LTX_I2V_STRENGTH = 0.75
LTX_LENGTH = 121  # native LTX-Video v0.9 training length, 4.84s @ 25fps
LTX_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
LTX_CHECKPOINT = "ltx-video-2b-v0.9.safetensors"
T5_CLIP = "t5xxl_fp16.safetensors"


def find_recent_radio_bookends(n: int = 4) -> list[str]:
    """Find the N most recent radio_bookend_*.png files across episodes.
    Returns just the basenames (LoadImage looks in input/).
    """
    eps_root = Path(
        os.path.expanduser("~"), "Documents", "ComfyUI",
        "output", "otr", "episodes",
    )
    if not eps_root.exists():
        return []
    cands = list(eps_root.glob("**/stills/radio_bookend_*.png"))
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in cands[:n]]


def copy_to_input(still_path: str) -> str:
    """Copy a radio_bookend.png into ComfyUI's input/ dir so LoadImage
    can find it. Returns just the basename for the workflow widget."""
    import shutil
    src = Path(still_path)
    if not src.exists():
        raise FileNotFoundError(f"still missing: {src}")
    input_dir = Path(
        os.path.expanduser("~"), "Documents", "ComfyUI", "input",
    )
    input_dir.mkdir(parents=True, exist_ok=True)
    dst = input_dir / src.name
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    return dst.name


# ---------------------------------------------------------------------------
# ComfyUI workflow JSON helpers
# ---------------------------------------------------------------------------

class WorkflowBuilder:
    """Tiny helper that builds a ComfyUI workflow JSON in the format the
    UI's "Load" button accepts. Positions nodes in a left-to-right column
    layout so the graph is readable when Jeffrey opens it."""

    def __init__(self):
        self.nodes: list[dict] = []
        self.links: list[list] = []
        self._next_node_id = 1
        self._next_link_id = 1

    def _new_node_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def _new_link_id(self) -> int:
        lid = self._next_link_id
        self._next_link_id += 1
        return lid

    def add_node(self, node_type: str, *, pos, size=None, title=None,
                 inputs=None, outputs=None, widgets_values=None,
                 properties=None) -> dict:
        node = {
            "id": self._new_node_id(),
            "type": node_type,
            "pos": list(pos),
            "size": list(size or [320, 100]),
            "flags": {},
            "order": len(self.nodes),
            "mode": 0,
            "inputs": inputs or [],
            "outputs": outputs or [],
            "properties": properties or {"Node name for S&R": node_type},
            "widgets_values": widgets_values or [],
        }
        if title:
            node["title"] = title
        self.nodes.append(node)
        return node

    def link(self, *, from_node, from_slot, to_node, to_slot, link_type) -> int:
        lid = self._new_link_id()
        # Append to source node's output links list
        for out in from_node["outputs"]:
            if out.get("slot_index") == from_slot or (
                out.get("name") and out is from_node["outputs"][from_slot]
            ):
                out.setdefault("links", []).append(lid)
                break
        # Set the destination input's link
        dest_input = to_node["inputs"][to_slot]
        dest_input["link"] = lid
        # Add to top-level links
        self.links.append([
            lid, from_node["id"], from_slot, to_node["id"], to_slot, link_type,
        ])
        return lid

    def to_json(self) -> dict:
        return {
            # ComfyUI's frontend Zod schema requires a valid UUID at "id".
            # A slug like "ltx-motion-batch-2026-05-06" gets rejected at
            # workflow-load time with "Invalid uuid at id" (BUG-LOCAL-114).
            # Pinned UUIDv4 so the JSON is diff-friendly across rebuilds.
            "id": "1e7ec912-1b40-4c0d-9a5b-6c0d5e7a9b3e",
            "revision": 1,
            "last_node_id": self._next_node_id - 1,
            "last_link_id": self._next_link_id - 1,
            "nodes": self.nodes,
            "links": self.links,
            "groups": [],
            "config": {},
            "extra": {"ds": {"scale": 0.6, "offset": [0, 0]}},
            "version": 0.4,
        }


def _make_input(name: str, type_: str) -> dict:
    return {"name": name, "type": type_, "link": None}


def _make_output(name: str, type_: str, slot_index: int) -> dict:
    return {"name": name, "type": type_, "links": [], "slot_index": slot_index}


def build(role: str, prompt_override: str | None, n_stills: int) -> dict:
    """Build a minimal LTX render workflow:
        radio still --> LTX --> preview video

    n_stills = 1 (default) gives the simplest possible chain: one
    LoadImage, one render, one SaveAnimatedWEBP preview. n_stills > 1
    repeats the per-still chain so multiple radio_bookends share the
    expensive loaders while each gets its own preview.
    """
    prompt = prompt_override or PROMPTS[role]
    stills = find_recent_radio_bookends(n_stills)
    if not stills:
        raise SystemExit(
            "No radio_bookend_*.png found under output/otr/episodes/. "
            "Run a full episode first or pass --stills-list explicitly."
        )

    # Copy stills into input/ so LoadImage can find them
    basenames = [copy_to_input(s) for s in stills]
    rows = [(b, LTX_I2V_STRENGTH) for b in basenames]

    wb = WorkflowBuilder()

    # ============ SHARED CHAIN (column layout, x=0..1400) ============
    Y0 = 100
    DY = 250

    # --- Column A: loaders ---
    n_ckpt = wb.add_node(
        "LowVRAMCheckpointLoader",
        pos=(40, Y0),
        size=(320, 110),
        title="LTX 2B Loader",
        inputs=[_make_input("dependencies", "*")],
        outputs=[
            _make_output("MODEL", "MODEL", 0),
            _make_output("CLIP", "CLIP", 1),
            _make_output("VAE", "VAE", 2),
        ],
        widgets_values=[LTX_CHECKPOINT],
    )
    n_clip = wb.add_node(
        "CLIPLoader",
        pos=(40, Y0 + DY),
        size=(320, 110),
        title="T5-XXL Text Encoder",
        outputs=[_make_output("CLIP", "CLIP", 0)],
        widgets_values=[T5_CLIP, "ltxv", "default"],
    )

    # --- Column B: prompts ---
    n_pos_prompt = wb.add_node(
        "CLIPTextEncode",
        pos=(420, Y0),
        size=(420, 140),
        title="POSITIVE prompt (BUG-112 motion-centric)",
        inputs=[_make_input("clip", "CLIP")],
        outputs=[_make_output("CONDITIONING", "CONDITIONING", 0)],
        widgets_values=[prompt],
    )
    n_neg_prompt = wb.add_node(
        "CLIPTextEncode",
        pos=(420, Y0 + 200),
        size=(420, 100),
        title="NEGATIVE prompt (inert at CFG=1.0)",
        inputs=[_make_input("clip", "CLIP")],
        outputs=[_make_output("CONDITIONING", "CONDITIONING", 0)],
        widgets_values=[""],
    )

    # --- Column C: LTXV conditioning + sampler/sigma + empty latent ---
    n_ltxv_cond = wb.add_node(
        "LTXVConditioning",
        pos=(900, Y0),
        size=(280, 90),
        title="LTXV Conditioning (frame_rate=25)",
        inputs=[
            _make_input("positive", "CONDITIONING"),
            _make_input("negative", "CONDITIONING"),
        ],
        outputs=[
            _make_output("positive", "CONDITIONING", 0),
            _make_output("negative", "CONDITIONING", 1),
        ],
        widgets_values=[LTX_FPS],
    )
    n_sampler = wb.add_node(
        "KSamplerSelect",
        pos=(900, Y0 + 130),
        size=(280, 60),
        title="Sampler (euler)",
        outputs=[_make_output("SAMPLER", "SAMPLER", 0)],
        widgets_values=["euler"],
    )
    n_sigmas = wb.add_node(
        "ManualSigmas",
        pos=(900, Y0 + 220),
        size=(280, 80),
        title="Sigmas (distilled 8-step)",
        outputs=[_make_output("SIGMAS", "SIGMAS", 0)],
        widgets_values=[LTX_SIGMAS],
    )
    n_empty_latent = wb.add_node(
        "EmptyLTXVLatentVideo",
        pos=(900, Y0 + 320),
        size=(280, 110),
        title=f"Empty Latent ({LTX_WIDTH}x{LTX_HEIGHT}, {LTX_LENGTH} frames)",
        outputs=[_make_output("LATENT", "LATENT", 0)],
        widgets_values=[LTX_WIDTH, LTX_HEIGHT, LTX_LENGTH, 1],
    )
    n_guider = wb.add_node(
        "CFGGuider",
        pos=(1240, Y0),
        size=(280, 90),
        title=f"CFG Guider (cfg={LTX_CFG})",
        inputs=[
            _make_input("model", "MODEL"),
            _make_input("positive", "CONDITIONING"),
            _make_input("negative", "CONDITIONING"),
        ],
        outputs=[_make_output("GUIDER", "GUIDER", 0)],
        widgets_values=[LTX_CFG],
    )

    # Wire shared chain
    wb.link(from_node=n_clip, from_slot=0, to_node=n_pos_prompt, to_slot=0, link_type="CLIP")
    wb.link(from_node=n_clip, from_slot=0, to_node=n_neg_prompt, to_slot=0, link_type="CLIP")
    wb.link(from_node=n_pos_prompt, from_slot=0, to_node=n_ltxv_cond, to_slot=0, link_type="CONDITIONING")
    wb.link(from_node=n_neg_prompt, from_slot=0, to_node=n_ltxv_cond, to_slot=1, link_type="CONDITIONING")
    wb.link(from_node=n_ckpt, from_slot=0, to_node=n_guider, to_slot=0, link_type="MODEL")
    wb.link(from_node=n_ltxv_cond, from_slot=0, to_node=n_guider, to_slot=1, link_type="CONDITIONING")
    wb.link(from_node=n_ltxv_cond, from_slot=1, to_node=n_guider, to_slot=2, link_type="CONDITIONING")

    # ============ PER-STILL CHAINS (one row per still, x>=1600) ============
    PER_X0 = 1600
    PER_DX = 360
    PER_DY = 320

    for i, (basename, _row_strength) in enumerate(rows):
        row_y = Y0 + i * PER_DY
        title_suffix = basename.replace("radio_bookend_signal_lost_", "").replace(".png", "")
        if len(title_suffix) > 32:
            title_suffix = title_suffix[:30] + "..."

        # LoadImage
        n_img = wb.add_node(
            "LoadImage",
            pos=(PER_X0, row_y),
            size=(320, 280),
            title=f"Still {i+1}: {title_suffix}",
            outputs=[
                _make_output("IMAGE", "IMAGE", 0),
                _make_output("MASK", "MASK", 1),
            ],
            widgets_values=[basename, "image"],
        )

        # LTXVImgToVideoConditionOnly
        n_i2v = wb.add_node(
            "LTXVImgToVideoConditionOnly",
            pos=(PER_X0 + PER_DX, row_y),
            size=(280, 110),
            title=f"i2v init (strength={_row_strength})",
            inputs=[
                _make_input("vae", "VAE"),
                _make_input("image", "IMAGE"),
                _make_input("latent", "LATENT"),
            ],
            outputs=[_make_output("LATENT", "LATENT", 0)],
            widgets_values=[_row_strength],
        )

        # RandomNoise (different seed per still so we don't get identical clips)
        n_noise = wb.add_node(
            "RandomNoise",
            pos=(PER_X0 + PER_DX, row_y + 130),
            size=(280, 80),
            title=f"Noise seed={42 + i * 137}",
            outputs=[_make_output("NOISE", "NOISE", 0)],
            widgets_values=[42 + i * 137, "fixed"],
        )

        # SamplerCustomAdvanced
        n_samp = wb.add_node(
            "SamplerCustomAdvanced",
            pos=(PER_X0 + PER_DX * 2, row_y),
            size=(280, 200),
            title=f"Sample (still {i+1})",
            inputs=[
                _make_input("noise", "NOISE"),
                _make_input("guider", "GUIDER"),
                _make_input("sampler", "SAMPLER"),
                _make_input("sigmas", "SIGMAS"),
                _make_input("latent_image", "LATENT"),
            ],
            outputs=[
                _make_output("output", "LATENT", 0),
                _make_output("denoised_output", "LATENT", 1),
            ],
            widgets_values=[],
        )

        # VAEDecode
        n_dec = wb.add_node(
            "VAEDecode",
            pos=(PER_X0 + PER_DX * 3, row_y),
            size=(220, 70),
            title=f"VAE Decode (still {i+1})",
            inputs=[
                _make_input("samples", "LATENT"),
                _make_input("vae", "VAE"),
            ],
            outputs=[_make_output("IMAGE", "IMAGE", 0)],
            widgets_values=[],
        )

        # SaveAnimatedWEBP -- auto-plays inline in ComfyUI's node UI.
        # Stock core node, MIT-safe.
        n_save = wb.add_node(
            "SaveAnimatedWEBP",
            pos=(PER_X0 + PER_DX * 4, row_y),
            size=(320, 240),
            title=f"PREVIEW (still {i+1})",
            inputs=[_make_input("images", "IMAGE")],
            outputs=[],
            widgets_values=[
                f"ltx_motion_smoke_{i+1}",
                LTX_FPS,
                False,         # lossless
                85,            # quality
                "default",     # method
            ],
        )

        # Wire per-still chain
        wb.link(from_node=n_ckpt, from_slot=2, to_node=n_i2v, to_slot=0, link_type="VAE")
        wb.link(from_node=n_img, from_slot=0, to_node=n_i2v, to_slot=1, link_type="IMAGE")
        wb.link(from_node=n_empty_latent, from_slot=0, to_node=n_i2v, to_slot=2, link_type="LATENT")

        wb.link(from_node=n_noise, from_slot=0, to_node=n_samp, to_slot=0, link_type="NOISE")
        wb.link(from_node=n_guider, from_slot=0, to_node=n_samp, to_slot=1, link_type="GUIDER")
        wb.link(from_node=n_sampler, from_slot=0, to_node=n_samp, to_slot=2, link_type="SAMPLER")
        wb.link(from_node=n_sigmas, from_slot=0, to_node=n_samp, to_slot=3, link_type="SIGMAS")
        wb.link(from_node=n_i2v, from_slot=0, to_node=n_samp, to_slot=4, link_type="LATENT")

        wb.link(from_node=n_samp, from_slot=0, to_node=n_dec, to_slot=0, link_type="LATENT")
        wb.link(from_node=n_ckpt, from_slot=2, to_node=n_dec, to_slot=1, link_type="VAE")
        wb.link(from_node=n_dec, from_slot=0, to_node=n_save, to_slot=0, link_type="IMAGE")

    # ============ NOTE NODE (instructions) ============
    note = wb.add_node(
        "Note",
        pos=(40, Y0 + DY * 2 + 40),
        size=(820, 320),
        title="README -- BUG-LOCAL-112 motion smoke",
        widgets_values=[
            "BUG-LOCAL-112 verification harness.\n"
            "\n"
            f"Prompt ({len(prompt)} chars, motion-centric):\n"
            f"  {prompt}\n"
            "\n"
            "Each radio_bookend.png on the right renders one LTX clip with\n"
            "the same prompt + same loader chain but a different noise seed.\n"
            "Output is one animated WEBP per still that plays inline in the\n"
            "node UI (top-right). Effective static = no motion. Real motion =\n"
            "visible animation across the full ~5 seconds.\n"
            "\n"
            "Knobs locked at production values:\n"
            f"  cfg={LTX_CFG}  i2v_strength={LTX_I2V_STRENGTH}\n"
            f"  width={LTX_WIDTH}  height={LTX_HEIGHT}  length={LTX_LENGTH} frames\n"
            f"  fps={LTX_FPS}  sampler=euler  sigmas=8-step distilled\n"
            "\n"
            "If you want quantitative motion measurement on the saved mp4s,\n"
            "they land at output/ next to the WEBPs. Run:\n"
            "  python scripts\\ltx_motion_mad.py --mp4 <path>\n"
            "  -- sustained 15-30 MAD = real motion; <5 = static.\n",
        ],
    )

    return wb.to_json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--role", choices=list(PROMPTS.keys()), default="announcer")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Override the role-derived prompt with a custom string")
    parser.add_argument("--stills", type=int, default=1,
                        help="Number of most-recent radio_bookend stills to batch "
                             "(default 1 = simplest possible: one still -> LTX -> "
                             "preview)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path (default: workflows/ltx_motion_batch.json)")
    args = parser.parse_args()

    workflow = build(args.role, args.prompt, args.stills)

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "workflows",
        "ltx_motion_batch.json",
    )
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(workflow, f, indent=2)
    print(f"wrote {out_path}", flush=True)
    print(f"  {len(workflow['nodes'])} nodes, {len(workflow['links'])} links",
          flush=True)
    print(f"  prompt ({len(args.prompt or PROMPTS[args.role])} chars): "
          f"{(args.prompt or PROMPTS[args.role])[:90]}...", flush=True)
    print(flush=True)
    print("Open in ComfyUI: drag-drop this JSON onto the canvas, or", flush=True)
    print("click \"Load\" and select it. Then click \"Queue Prompt\".", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
