"""Build workflows/ltx_bookend_mini_repro_22b.json -- the REAL b001 recipe as a
standalone litegraph the operator loads + runs MANUALLY in ComfyUI.

The faithful stack (what otr_scifi_16gb_full__ORIGINAL_b001_c9af198.json used):
  LowVRAMCheckpointLoader(ltx-2.3-22b-dev) -> MODEL,VAE
  LoraLoaderModelOnly(ltx-2.3-22b-distilled-lora-384-1.1 @ 0.70) -> MODEL
  LTXAVTextEncoderLoader(gemma_3_12B_it_fp4_mixed + ltx-2.3-22b-dev) -> CLIP   [the Gemma encoder]
  CLIPTextEncode(motion)/(neg) -> LTXVConditioning(25fps)
  LoadImage(radio bookend still) + EmptyLTXVLatentVideo(832x480x105)
    -> LTXVImgToVideoConditionOnly(strength 0.75)
  v0_9 sampling: KSamplerSelect(euler_cfg_pp)+ManualSigmas(8-step distilled)+RandomNoise
    +CFGGuider(cfg 1.0) -> SamplerCustomAdvanced -> VAEDecodeTiled(512/64/4096/8)
    -> VHS_VideoCombine(pingpong boomerang, h264-mp4, 25fps)

Server ACCEPTED this exact node set (smoke POST, 0 node_errors). 105 render frames
-> ~209 after pingpong = the b001 length. NO Flux/HuMo/upscale/audio/compositor.
The branch-gate / gate_signal machinery is a VRAM-defer trick, not needed standalone.
UTF-8, no BOM.
"""
import json, os

OUT = (r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
       r"\workflows\ltx_bookend_mini_repro_22b.json")

CKPT  = "ltx-2.3-22b-dev.safetensors"
LORA  = r"ltxv\ltx2\ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
GEMMA = "gemma_3_12B_it_fp4_mixed.safetensors"
STILL = "radio_bookend_mimicry_b001.png"
SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
# Operator-chosen prompt (2026-06-15): sets the sci-fi radio-drama MOOD without
# naming any object to animate, so the model never invents a dial/tubes/radio
# that isn't in the still. (Typo "drmataic" -> "dramatic".)
MOTION = "Continuous shot, a dramatic scene from a sci-fi radio drama"
NEG = "low quality, worst quality, blurry, distorted, watermark, text, static"

N = {}
def node(i, t, pos, size, widgets, ins, outs):
    N[i] = dict(type=t, pos=pos, size=size, widgets=widgets, ins=ins, outs=outs)

node(1, "LowVRAMCheckpointLoader", [80, 60], [330, 110], [CKPT],
     [], [("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")])
node(2, "LoraLoaderModelOnly", [80, 240], [330, 110], [LORA, 0.70],
     [("model", "MODEL")], [("MODEL", "MODEL")])
node(3, "LTXAVTextEncoderLoader", [80, 410], [330, 110], [GEMMA, CKPT, "default"],
     [], [("CLIP", "CLIP")])
node(4, "CLIPTextEncode", [470, 60], [380, 150], [MOTION],
     [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")])
node(5, "CLIPTextEncode", [470, 250], [380, 130], [NEG],
     [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")])
node(6, "LoadImage", [80, 580], [330, 314], [STILL, "image"],
     [], [("IMAGE", "IMAGE"), ("MASK", "MASK")])
node(7, "EmptyLTXVLatentVideo", [470, 430], [330, 130], [832, 480, 105, 1],
     [], [("LATENT", "LATENT")])
node(8, "LTXVImgToVideoConditionOnly", [900, 120], [340, 120], [0.75, False],
     [("vae", "VAE"), ("image", "IMAGE"), ("latent", "LATENT")], [("latent", "LATENT")])
node(9, "LTXVConditioning", [900, 320], [320, 100], [25.0],
     [("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
     [("positive", "CONDITIONING"), ("negative", "CONDITIONING")])
node(10, "KSamplerSelect", [900, 470], [300, 60], ["euler_cfg_pp"],
     [], [("SAMPLER", "SAMPLER")])
node(11, "ManualSigmas", [900, 580], [340, 80], [SIGMAS],
     [], [("SIGMAS", "SIGMAS")])
node(12, "RandomNoise", [900, 710], [300, 82], [1, "fixed"],
     [], [("NOISE", "NOISE")])
node(13, "CFGGuider", [1300, 330], [300, 98], [1.0],
     [("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
     [("GUIDER", "GUIDER")])
node(14, "SamplerCustomAdvanced", [1660, 300], [320, 150], [],
     [("noise", "NOISE"), ("guider", "GUIDER"), ("sampler", "SAMPLER"),
      ("sigmas", "SIGMAS"), ("latent_image", "LATENT")],
     [("output", "LATENT"), ("denoised_output", "LATENT")])
node(15, "VAEDecodeTiled", [2040, 300], [320, 130], [512, 64, 4096, 8],
     [("samples", "LATENT"), ("vae", "VAE")], [("IMAGE", "IMAGE")])
node(16, "VHS_VideoCombine", [2420, 280], [400, 320],
     [25.0, 0, "otr_ltx_22b/repro_b001", "video/h264-mp4", True, True],
     [("images", "IMAGE")], [("Filenames", "VHS_FILENAMES")])

WIRES = [
    (3, 0, 4, 0), (3, 0, 5, 0),
    (1, 0, 2, 0),                      # LowVRAM MODEL -> LoRA
    (1, 2, 8, 0), (6, 0, 8, 1), (7, 0, 8, 2),   # i2v vae/image/latent
    (4, 0, 9, 0), (5, 0, 9, 1),        # cond -> LTXVConditioning
    (2, 0, 13, 0), (9, 0, 13, 1), (9, 1, 13, 2),  # guider model/pos/neg
    (12, 0, 14, 0), (13, 0, 14, 1), (10, 0, 14, 2), (11, 0, 14, 3), (8, 0, 14, 4),
    (14, 0, 15, 0), (1, 2, 15, 1),     # decode samples/vae
    (15, 0, 16, 0),                    # -> VHS
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
