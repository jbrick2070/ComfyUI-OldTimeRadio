"""build_ltx_avjoint_smoke.py -- minimal AV-joint LTX i2v smoke that
mirrors the proven DMM (`comfyui-data-media-machine`) pipeline.

DISCOVERY (BUG-LOCAL-116, 2026-05-06):
LTX 2B v0.9 is trained on AUDIO-VIDEO JOINT latents. Sampling on a
video-only latent (what our prior smoke + production pipeline does)
puts the model out of distribution -- the failure mode is FREEZE.
DMM's batch_video.py samples on `LTXVConcatAVLatent(video, audio)` and
splits the denoised output back via `LTXVSeparateAVLatent`. Same base
model, same sigmas, same sampler -- but the latent shape is correct.

This builder emits the simplest possible workflow that matches the
DMM pattern:

    LowVRAMCheckpointLoader  ->  model, video_vae
    CLIPLoader (T5-XXL)      ->  clip
    LTXVAudioVAELoader       ->  audio_vae      <-- NEW vs prior smoke
    LoadImage                ->  radio still
    CLIPTextEncode pos/neg   ->  conditionings
    LTXVConditioning         ->  cond_pos, cond_neg
    EmptyLTXVLatentVideo     ->  empty video latent
    LTXVImgToVideoConditionOnly(strength=0.75)  ->  video latent w/ image
    LTXVEmptyLatentAudio     ->  empty audio latent     <-- NEW
    LTXVConcatAVLatent       ->  AV-joint latent        <-- NEW
    RandomNoise + KSamplerSelect + ManualSigmas + CFGGuider
    SamplerCustomAdvanced(latent_image=AV-joint)        <-- key change
    LTXVSeparateAVLatent     ->  video, audio (we discard audio)  <-- NEW
    VAEDecode (video VAE)    ->  IMAGE
    SaveAnimatedWEBP         ->  inline preview

License: stock ComfyUI core + LTXVideo custom node pack (both MIT-safe).
No LoRA in this minimal version (LightX2V LoRA is a quality lift, not
a motion-presence requirement). No two-stage upscaler. No Gemma encoder.

Usage:
    python scripts\\build_ltx_avjoint_smoke.py
    python scripts\\build_ltx_avjoint_smoke.py --prompt "Custom motion"
    python scripts\\build_ltx_avjoint_smoke.py --still C:\\path\\to\\still.png
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

DEFAULT_PROMPT = (
    "Continuous video shot, same room throughout. Slow handheld camera "
    "dolly forward through the console. Console lights flicker rhythmically. "
    "Atmospheric haze drifts. Subtle camera shake."
)

LTX_FPS = 25
LTX_WIDTH = 832
LTX_HEIGHT = 480
LTX_CFG = 1.0
LTX_I2V_STRENGTH = 0.75
LTX_LENGTH = 121
LTX_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
LTX_CHECKPOINT = "ltx-video-2b-v0.9.safetensors"
T5_CLIP = "t5xxl_fp16.safetensors"


def find_default_still() -> str:
    eps_root = Path(
        os.path.expanduser("~"), "Documents", "ComfyUI",
        "output", "otr", "episodes",
    )
    if not eps_root.exists():
        return ""
    cands = list(eps_root.glob("**/stills/radio_bookend_*.png"))
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cands[0]) if cands else ""


def copy_to_input(still_path: str) -> str:
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


def _make_input(name, type_):
    return {"name": name, "type": type_, "link": None}


def _make_output(name, type_, slot):
    return {"name": name, "type": type_, "links": [], "slot_index": slot}


class WB:
    def __init__(self):
        self.nodes = []
        self.links = []
        self._nid = 1
        self._lid = 1

    def add(self, type_, *, pos, size=None, title=None,
            inputs=None, outputs=None, widgets_values=None):
        n = {
            "id": self._nid,
            "type": type_,
            "pos": list(pos),
            "size": list(size or [320, 100]),
            "flags": {},
            "order": len(self.nodes),
            "mode": 0,
            "inputs": inputs or [],
            "outputs": outputs or [],
            "properties": {"Node name for S&R": type_},
            "widgets_values": widgets_values or [],
        }
        if title:
            n["title"] = title
        self._nid += 1
        self.nodes.append(n)
        return n

    def link(self, src, src_slot, dst, dst_slot, type_):
        lid = self._lid
        self._lid += 1
        for o in src["outputs"]:
            if o.get("slot_index") == src_slot:
                o.setdefault("links", []).append(lid)
                break
        dst["inputs"][dst_slot]["link"] = lid
        self.links.append([lid, src["id"], src_slot, dst["id"], dst_slot, type_])


def build(still_basename: str, prompt: str) -> dict:
    wb = WB()
    Y = 100
    DY = 280

    # ---- Loaders (column 1) ----
    n_ckpt = wb.add(
        "LowVRAMCheckpointLoader",
        pos=(40, Y), size=(320, 110), title="LTX 2B Loader",
        inputs=[_make_input("dependencies", "*")],
        outputs=[
            _make_output("MODEL", "MODEL", 0),
            _make_output("CLIP", "CLIP", 1),
            _make_output("VAE", "VAE", 2),
        ],
        widgets_values=[LTX_CHECKPOINT],
    )
    n_clip = wb.add(
        "CLIPLoader",
        pos=(40, Y + 200), size=(320, 110), title="T5-XXL Text Encoder",
        outputs=[_make_output("CLIP", "CLIP", 0)],
        widgets_values=[T5_CLIP, "ltxv", "default"],
    )
    n_avae = wb.add(
        "LTXVAudioVAELoader",
        pos=(40, Y + 400), size=(320, 80),
        title="Audio VAE (BUG-116 - AV-joint pipeline)",
        outputs=[_make_output("VAE", "VAE", 0)],
        widgets_values=[LTX_CHECKPOINT],
    )
    n_img = wb.add(
        "LoadImage",
        pos=(40, Y + 540), size=(320, 280),
        title="radio_bookend.png",
        outputs=[
            _make_output("IMAGE", "IMAGE", 0),
            _make_output("MASK", "MASK", 1),
        ],
        widgets_values=[still_basename, "image"],
    )

    # ---- Prompts (column 2) ----
    n_pos = wb.add(
        "CLIPTextEncode",
        pos=(420, Y), size=(420, 160), title="POSITIVE prompt",
        inputs=[_make_input("clip", "CLIP")],
        outputs=[_make_output("CONDITIONING", "CONDITIONING", 0)],
        widgets_values=[prompt],
    )
    n_neg = wb.add(
        "CLIPTextEncode",
        pos=(420, Y + 220), size=(420, 100),
        title="NEGATIVE prompt (inert at CFG=1.0)",
        inputs=[_make_input("clip", "CLIP")],
        outputs=[_make_output("CONDITIONING", "CONDITIONING", 0)],
        widgets_values=[""],
    )

    # ---- Conditioning + sampler/sigma/empty latents (column 3) ----
    n_cond = wb.add(
        "LTXVConditioning",
        pos=(900, Y), size=(280, 90), title="LTXV Conditioning",
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
    n_samp = wb.add(
        "KSamplerSelect",
        pos=(900, Y + 130), size=(280, 60), title="Sampler (euler)",
        outputs=[_make_output("SAMPLER", "SAMPLER", 0)],
        widgets_values=["euler"],
    )
    n_sigmas = wb.add(
        "ManualSigmas",
        pos=(900, Y + 220), size=(280, 80), title="Sigmas (distilled 8-step)",
        outputs=[_make_output("SIGMAS", "SIGMAS", 0)],
        widgets_values=[LTX_SIGMAS],
    )
    n_empty_v = wb.add(
        "EmptyLTXVLatentVideo",
        pos=(900, Y + 320), size=(280, 110),
        title=f"Empty Video Latent ({LTX_WIDTH}x{LTX_HEIGHT}, {LTX_LENGTH}f)",
        outputs=[_make_output("LATENT", "LATENT", 0)],
        widgets_values=[LTX_WIDTH, LTX_HEIGHT, LTX_LENGTH, 1],
    )
    n_empty_a = wb.add(
        "LTXVEmptyLatentAudio",
        pos=(900, Y + 460), size=(280, 110),
        title="Empty Audio Latent (BUG-116)",
        inputs=[_make_input("audio_vae", "VAE")],
        outputs=[_make_output("LATENT", "LATENT", 0)],
        widgets_values=[LTX_LENGTH, LTX_FPS, 1],
    )

    # ---- i2v init + AV concat (column 4) ----
    n_i2v = wb.add(
        "LTXVImgToVideoConditionOnly",
        pos=(1240, Y), size=(280, 110),
        title=f"i2v init (strength={LTX_I2V_STRENGTH})",
        inputs=[
            _make_input("vae", "VAE"),
            _make_input("image", "IMAGE"),
            _make_input("latent", "LATENT"),
        ],
        outputs=[_make_output("LATENT", "LATENT", 0)],
        widgets_values=[LTX_I2V_STRENGTH],
    )
    n_concat = wb.add(
        "LTXVConcatAVLatent",
        pos=(1240, Y + 160), size=(280, 90),
        title="Concat AV Latent (BUG-116 -- the missing piece)",
        inputs=[
            _make_input("video_latent", "LATENT"),
            _make_input("audio_latent", "LATENT"),
        ],
        outputs=[_make_output("LATENT", "LATENT", 0)],
        widgets_values=[],
    )
    n_guider = wb.add(
        "CFGGuider",
        pos=(1240, Y + 280), size=(280, 90),
        title=f"CFG Guider (cfg={LTX_CFG})",
        inputs=[
            _make_input("model", "MODEL"),
            _make_input("positive", "CONDITIONING"),
            _make_input("negative", "CONDITIONING"),
        ],
        outputs=[_make_output("GUIDER", "GUIDER", 0)],
        widgets_values=[LTX_CFG],
    )
    n_noise = wb.add(
        "RandomNoise",
        pos=(1240, Y + 400), size=(280, 80), title="Noise seed=42",
        outputs=[_make_output("NOISE", "NOISE", 0)],
        widgets_values=[42, "fixed"],
    )

    # ---- Sampling + split + decode (column 5) ----
    n_sam = wb.add(
        "SamplerCustomAdvanced",
        pos=(1580, Y), size=(280, 200), title="Sample (AV-joint)",
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
    n_split = wb.add(
        "LTXVSeparateAVLatent",
        pos=(1580, Y + 230), size=(280, 90),
        title="Separate AV (BUG-116)",
        inputs=[_make_input("av_latent", "LATENT")],
        outputs=[
            _make_output("video_latent", "LATENT", 0),
            _make_output("audio_latent", "LATENT", 1),
        ],
        widgets_values=[],
    )
    n_dec = wb.add(
        "VAEDecode",
        pos=(1900, Y), size=(220, 70), title="VAE Decode (video)",
        inputs=[
            _make_input("samples", "LATENT"),
            _make_input("vae", "VAE"),
        ],
        outputs=[_make_output("IMAGE", "IMAGE", 0)],
        widgets_values=[],
    )
    n_save = wb.add(
        "SaveAnimatedWEBP",
        pos=(2160, Y), size=(360, 280), title="PREVIEW (auto-plays)",
        inputs=[_make_input("images", "IMAGE")],
        outputs=[],
        widgets_values=[
            "ltx_avjoint_smoke", LTX_FPS, False, 85, "default",
        ],
    )

    # ---- Note ----
    wb.add(
        "Note",
        pos=(40, Y + 860), size=(820, 240),
        title="README -- BUG-LOCAL-116 AV-joint smoke",
        widgets_values=[
            "BUG-LOCAL-116 hypothesis: LTX 2B v0.9 is trained on AV-joint\n"
            "latents. Sampling on a video-only latent (what our prior smoke\n"
            "and production pipeline did) puts the model OOD -> freeze.\n"
            "\n"
            "DMM's batch_video.py samples on LTXVConcatAVLatent(video, audio)\n"
            "and splits via LTXVSeparateAVLatent. Same base model, same\n"
            "sigmas, same sampler -- but the latent SHAPE matches what the\n"
            "model expects. That's the missing piece.\n"
            "\n"
            "If this workflow produces motion: hypothesis confirmed, fix\n"
            "is to update OTR's batch_ltx_render.py to use the AV-joint\n"
            "pipeline. If still static: the bug is not the latent shape;\n"
            "next escalation is the LightX2V LoRA + euler_ancestral_cfg_pp.\n",
        ],
    )

    # ---- Wire ----
    wb.link(n_clip, 0, n_pos, 0, "CLIP")
    wb.link(n_clip, 0, n_neg, 0, "CLIP")
    wb.link(n_pos, 0, n_cond, 0, "CONDITIONING")
    wb.link(n_neg, 0, n_cond, 1, "CONDITIONING")
    wb.link(n_avae, 0, n_empty_a, 0, "VAE")
    wb.link(n_ckpt, 2, n_i2v, 0, "VAE")
    wb.link(n_img, 0, n_i2v, 1, "IMAGE")
    wb.link(n_empty_v, 0, n_i2v, 2, "LATENT")
    wb.link(n_i2v, 0, n_concat, 0, "LATENT")
    wb.link(n_empty_a, 0, n_concat, 1, "LATENT")
    wb.link(n_ckpt, 0, n_guider, 0, "MODEL")
    wb.link(n_cond, 0, n_guider, 1, "CONDITIONING")
    wb.link(n_cond, 1, n_guider, 2, "CONDITIONING")
    wb.link(n_noise, 0, n_sam, 0, "NOISE")
    wb.link(n_guider, 0, n_sam, 1, "GUIDER")
    wb.link(n_samp, 0, n_sam, 2, "SAMPLER")
    wb.link(n_sigmas, 0, n_sam, 3, "SIGMAS")
    wb.link(n_concat, 0, n_sam, 4, "LATENT")
    wb.link(n_sam, 0, n_split, 0, "LATENT")
    wb.link(n_split, 0, n_dec, 0, "LATENT")
    wb.link(n_ckpt, 2, n_dec, 1, "VAE")
    wb.link(n_dec, 0, n_save, 0, "IMAGE")

    return {
        "id": "1f9e8d72-2c51-4b3a-8f7d-9e5a6b1c2d3e",
        "revision": 1,
        "last_node_id": wb._nid - 1,
        "last_link_id": wb._lid - 1,
        "nodes": wb.nodes,
        "links": wb.links,
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.7, "offset": [0, 0]}},
        "version": 0.4,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--still", type=str, default=None,
                        help="Path to radio_bookend.png (default: most recent)")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    still = args.still or find_default_still()
    if not still:
        raise SystemExit("No radio_bookend_*.png found; pass --still PATH")
    basename = copy_to_input(still)

    workflow = build(basename, args.prompt)

    out = args.out or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "workflows", "ltx_avjoint_smoke.json",
    )
    Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(workflow, f, indent=2)
    print(f"wrote {out}")
    print(f"  {len(workflow['nodes'])} nodes, {len(workflow['links'])} links")
    print(f"  still: {basename}")
    print(f"  prompt ({len(args.prompt)} chars)")
    print()
    print("Open in ComfyUI: drag-drop or Load -> Queue Prompt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
