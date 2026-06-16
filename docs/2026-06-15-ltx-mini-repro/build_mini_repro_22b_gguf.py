"""Build workflows/ltx_bookend_mini_repro_22b_gguf.json -- the LIGHTEST bookend
repro: LTX 2.3 22B at Q4 GGUF (~13GB) instead of the 46GB checkpoint, so it fits
16GB without thrash. Same v0_9/Goofer sampling + sci-fi prompt + 105+boomerang.

  UnetLoaderGGUF(ltx-2.3-22b-dev-Q4_K_S.gguf) -> MODEL
  LoraLoaderModelOnly(22B distilled LoRA @ 0.70) -> MODEL
  VAELoader(ltx-2.3-22b-dev_video_vae)            -> VAE   (GGUF has no VAE)
  LTXAVTextEncoderLoader(gemma_3_12B_it_fp4_mixed + ltx-2.3-22b-dev) -> CLIP
  CLIPTextEncode(scifi)/(neg) -> LTXVConditioning(25)
  LoadImage + EmptyLTXVLatentVideo(832x480x105) -> LTXVImgToVideoConditionOnly(0.75)
  KSamplerSelect(euler_cfg_pp)+ManualSigmas+RandomNoise+CFGGuider(1.0)
    -> SamplerCustomAdvanced -> VAEDecodeTiled(512/64/4096/8)
    -> VHS_VideoCombine(pingpong, h264-mp4, 25fps)

Peak VRAM ~13GB (Q4 unet) -- the Gemma encode runs first then offloads. UTF-8 no BOM.
"""
import json, os
OUT = (r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
       r"\workflows\ltx_bookend_mini_repro_22b_gguf.json")
GGUF = "ltx-2.3-22b-dev-Q4_K_S.gguf"
LORA = r"ltxv\ltx2\ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
GEMMA = "gemma_3_12B_it_fp4_mixed.safetensors"
DEVCKPT = "ltx-2.3-22b-dev.safetensors"   # encoder connectors (CLIP-only extract)
VVAE = "ltx-2.3-22b-dev_video_vae.safetensors"
STILL = "radio_bookend_mimicry_b001.png"
SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
MOTION = "Continuous shot, a dramatic scene from a sci-fi radio drama"
NEG = "low quality, worst quality, blurry, distorted, watermark, text, static"

N = {}
def node(i, t, pos, size, widgets, ins, outs):
    N[i] = dict(type=t, pos=pos, size=size, widgets=widgets, ins=ins, outs=outs)

node(1, "UnetLoaderGGUF", [80, 60], [330, 80], [GGUF], [], [("MODEL", "MODEL")])
node(2, "LoraLoaderModelOnly", [80, 200], [330, 110], [LORA, 0.70],
     [("model", "MODEL")], [("MODEL", "MODEL")])
node(3, "VAELoader", [80, 360], [330, 80], [VVAE], [], [("VAE", "VAE")])
node(4, "LTXAVTextEncoderLoader", [80, 490], [330, 110], [GEMMA, DEVCKPT, "default"],
     [], [("CLIP", "CLIP")])
node(5, "CLIPTextEncode", [470, 60], [380, 120], [MOTION],
     [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")])
node(6, "CLIPTextEncode", [470, 220], [380, 130], [NEG],
     [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")])
node(7, "LoadImage", [80, 640], [330, 314], [STILL, "image"],
     [], [("IMAGE", "IMAGE"), ("MASK", "MASK")])
node(8, "EmptyLTXVLatentVideo", [470, 400], [330, 130], [832, 480, 105, 1],
     [], [("LATENT", "LATENT")])
node(9, "LTXVImgToVideoConditionOnly", [900, 120], [340, 120], [0.75, False],
     [("vae", "VAE"), ("image", "IMAGE"), ("latent", "LATENT")], [("latent", "LATENT")])
node(10, "LTXVConditioning", [900, 320], [320, 100], [25.0],
     [("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
     [("positive", "CONDITIONING"), ("negative", "CONDITIONING")])
node(11, "KSamplerSelect", [900, 470], [300, 60], ["euler_cfg_pp"],
     [], [("SAMPLER", "SAMPLER")])
node(12, "ManualSigmas", [900, 580], [340, 80], [SIGMAS], [], [("SIGMAS", "SIGMAS")])
node(13, "RandomNoise", [900, 710], [300, 82], [1, "fixed"], [], [("NOISE", "NOISE")])
node(14, "CFGGuider", [1300, 330], [300, 98], [1.0],
     [("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
     [("GUIDER", "GUIDER")])
node(15, "SamplerCustomAdvanced", [1660, 300], [320, 150], [],
     [("noise", "NOISE"), ("guider", "GUIDER"), ("sampler", "SAMPLER"),
      ("sigmas", "SIGMAS"), ("latent_image", "LATENT")],
     [("output", "LATENT"), ("denoised_output", "LATENT")])
node(16, "VAEDecodeTiled", [2040, 300], [320, 130], [512, 64, 4096, 8],
     [("samples", "LATENT"), ("vae", "VAE")], [("IMAGE", "IMAGE")])
node(17, "VHS_VideoCombine", [2420, 280], [400, 320],
     [25.0, 0, "otr_ltx_22b_gguf/repro_b001", "video/h264-mp4", True, True],
     [("images", "IMAGE")], [("Filenames", "VHS_FILENAMES")])

WIRES = [
    (1, 0, 2, 0), (4, 0, 5, 0), (4, 0, 6, 0),
    (3, 0, 9, 0), (7, 0, 9, 1), (8, 0, 9, 2),
    (5, 0, 10, 0), (6, 0, 10, 1),
    (2, 0, 14, 0), (10, 0, 14, 1), (10, 1, 14, 2),
    (13, 0, 15, 0), (14, 0, 15, 1), (11, 0, 15, 2), (12, 0, 15, 3), (9, 0, 15, 4),
    (15, 0, 16, 0), (3, 0, 16, 1), (16, 0, 17, 0),
]

links = []
out_links = {(i, s): [] for i in N for s in range(len(N[i]["outs"]))}
in_link = {}
for lid, (sn, ss, dn, ds) in enumerate(WIRES, start=1):
    links.append([lid, sn, ss, dn, ds, N[sn]["outs"][ss][1]])
    out_links[(sn, ss)].append(lid)
    in_link[(dn, ds)] = lid

nodes_json = []
for i in sorted(N):
    n = N[i]
    inputs = [{"name": nm, "type": ty, "link": in_link.get((i, s))}
              for s, (nm, ty) in enumerate(n["ins"])]
    outputs = [{"name": nm, "type": ty, "slot_index": s, "links": out_links[(i, s)]}
               for s, (nm, ty) in enumerate(n["outs"])]
    nodes_json.append({"id": i, "type": n["type"], "pos": n["pos"], "size": n["size"],
                       "flags": {}, "order": i - 1, "mode": 0, "inputs": inputs,
                       "outputs": outputs, "properties": {"Node name for S&R": n["type"]},
                       "widgets_values": n["widgets"]})
graph = {"last_node_id": max(N), "last_link_id": len(links), "nodes": nodes_json,
         "links": links, "groups": [], "config": {}, "extra": {}, "version": 0.4}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    json.dump(graph, f, indent=2, ensure_ascii=False)
    f.write("\n")
print("WROTE", OUT, "| nodes", len(nodes_json), "| links", len(links))
