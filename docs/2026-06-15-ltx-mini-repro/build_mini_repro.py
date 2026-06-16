"""Build workflows/ltx_bookend_mini_repro.json -- a STANDALONE litegraph UI
workflow that reproduces the GOOD 6/6 LTX radio bookend MOTION in isolation.

Topology (mirrors the ledger-proven v0_9 engine path of b001/b005, commit c9af198):
  LoadImage(radio bookend still) ----------------------------+
  CheckpointLoaderSimple(ltx-video-2b-v0.9) -> MODEL,VAE     |
  CLIPLoader(t5xxl_fp16, ltxv) -> CLIP                       |
  CLIPTextEncode(motion) / CLIPTextEncode(neg)               v
  EmptyLTXVLatentVideo(832x480x209) -> LTXVImgToVideoConditionOnly(strength 0.75)
  LTXVConditioning(frame_rate 25) -> CFGGuider(cfg 1.0)
  KSamplerSelect(euler_cfg_pp) + ManualSigmas(8-step distilled) + RandomNoise
     -> SamplerCustomAdvanced -> VAEDecodeTiled(512/64/4096/8)
     -> VHS_VideoCombine(pingpong boomerang, h264-mp4, 25fps)

NO Flux, NO HuMo, NO upscale, NO audio, NO compositor.

CRITICAL (the 2026-06-15 mush fix): b001.mp4 is 209 frames, but the engine renders
HALF -- 105 frames (8*13+1) -- then ffmpeg-boomerangs (2*105-1 = 209) to the full
length (batch_ltx_render.py BUG-LOCAL-117d, chunk_render_dur_s = chunk_dur_s/2).
Rendering 209 pure-forward frames lets the i2v anchor decay over the long sequence
and BLOOMS into mush. So this graph renders 105 and boomerangs in-node via
VHS_VideoCombine pingpong=True. Verified sharp on the live 5080, 2026-06-15.

All recipe constants are the ledger-proven 6/6 values. Slot types/orders verified
live against :8000 /object_info (ComfyUI 0.24.1). UTF-8, no BOM.
"""
import json, os

OUT = (r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
       r"\workflows\ltx_bookend_mini_repro.json")

# ---- ledger-proven 6/6 recipe (b001) ---------------------------------------
STILL      = "radio_bookend_mimicry_b001.png"   # staged in ComfyUI\input
LTX_CKPT   = "ltx-video-2b-v0.9.safetensors"
T5         = "t5xxl_fp16.safetensors"
# LENGTH is the HALF render length (105 = 8*13+1). VHS pingpong doubles it to the
# 209-frame b001 final. For b005 (241 final) use 121 (8*15+1). NEVER render the
# full 209 in one forward pass -- it MUSHES (the 2026-06-15 bug).
WIDTH, HEIGHT, LENGTH = 832, 480, 105
STRENGTH   = 0.75
SAMPLER    = "euler_cfg_pp"
SIGMAS     = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
CFG        = 1.0
FRAME_RATE = 25.0
SEED       = 1                                  # base seed (fixed); adjustable
# VAEDecodeTiled (NOTE: temporal_size default is 64 -- must override to 4096)
TILE, OVERLAP, TSIZE, TOVERLAP = 512, 64, 4096, 8
# announcer motion template (BUG-LOCAL-112, motion-only, <=188 chars)
MOTION = ("Continuous shot, same console throughout. Tuning dial needle sweeps "
          "rhythmically. Vacuum tubes pulse. Brass speaker grille trembles. "
          "Dust motes drift. Slow handheld dolly forward.")
NEG    = "low quality, worst quality, blurry, distorted, watermark, text, static"

# ---- node specs: id -> (type, pos, size, widgets, [in slots], [out slots]) --
# in slot  = (name, type); out slot = (name, type)
N = {}
def node(i, t, pos, size, widgets, ins, outs):
    N[i] = dict(type=t, pos=pos, size=size, widgets=widgets, ins=ins, outs=outs)

node(1, "LoadImage", [80, 80], [320, 314], [STILL, "image"],
     [], [("IMAGE", "IMAGE"), ("MASK", "MASK")])
node(2, "CheckpointLoaderSimple", [80, 440], [320, 100], [LTX_CKPT],
     [], [("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")])
node(3, "CLIPLoader", [80, 600], [320, 106], [T5, "ltxv", "default"],
     [], [("CLIP", "CLIP")])
node(4, "CLIPTextEncode", [460, 80], [380, 160], [MOTION],
     [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")])
node(5, "CLIPTextEncode", [460, 300], [380, 140], [NEG],
     [("clip", "CLIP")], [("CONDITIONING", "CONDITIONING")])
node(6, "EmptyLTXVLatentVideo", [460, 500], [320, 130], [WIDTH, HEIGHT, LENGTH, 1],
     [], [("LATENT", "LATENT")])
node(7, "LTXVImgToVideoConditionOnly", [900, 120], [340, 120], [STRENGTH, False],
     [("vae", "VAE"), ("image", "IMAGE"), ("latent", "LATENT")], [("latent", "LATENT")])
node(8, "LTXVConditioning", [900, 320], [320, 100], [FRAME_RATE],
     [("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
     [("positive", "CONDITIONING"), ("negative", "CONDITIONING")])
node(9, "KSamplerSelect", [900, 470], [300, 60], [SAMPLER],
     [], [("SAMPLER", "SAMPLER")])
node(10, "ManualSigmas", [900, 580], [340, 80], [SIGMAS],
     [], [("SIGMAS", "SIGMAS")])
node(11, "RandomNoise", [900, 710], [300, 82], [SEED, "fixed"],
     [], [("NOISE", "NOISE")])
node(12, "CFGGuider", [1300, 320], [300, 98], [CFG],
     [("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
     [("GUIDER", "GUIDER")])
node(13, "SamplerCustomAdvanced", [1660, 300], [320, 150], [],
     [("noise", "NOISE"), ("guider", "GUIDER"), ("sampler", "SAMPLER"),
      ("sigmas", "SIGMAS"), ("latent_image", "LATENT")],
     [("output", "LATENT"), ("denoised_output", "LATENT")])
node(14, "VAEDecodeTiled", [2040, 300], [320, 130], [TILE, OVERLAP, TSIZE, TOVERLAP],
     [("samples", "LATENT"), ("vae", "VAE")], [("IMAGE", "IMAGE")])
# VHS_VideoCombine with pingpong=True = the in-graph boomerang (forward+reverse),
# turning the sharp 105-frame render into the ~209-frame b001 loop in one Queue.
# widgets order (required, non-link): frame_rate, loop_count, filename_prefix,
# format, pingpong, save_output.
node(15, "VHS_VideoCombine", [2420, 280], [400, 320],
     [FRAME_RATE, 0, "otr_ltx_bookend_mini/repro_b001", "video/h264-mp4", True, True],
     [("images", "IMAGE")], [("Filenames", "VHS_FILENAMES")])

# ---- links: (src_node, src_slot, dst_node, dst_slot) -----------------------
WIRES = [
    (3, 0, 4, 0),    # CLIP -> pos.clip
    (3, 0, 5, 0),    # CLIP -> neg.clip
    (2, 2, 7, 0),    # VAE  -> i2v.vae
    (1, 0, 7, 1),    # IMAGE-> i2v.image
    (6, 0, 7, 2),    # LATENT-> i2v.latent
    (4, 0, 8, 0),    # pos  -> ltxcond.positive
    (5, 0, 8, 1),    # neg  -> ltxcond.negative
    (2, 0, 12, 0),   # MODEL-> guider.model
    (8, 0, 12, 1),   # ltxcond.pos -> guider.positive
    (8, 1, 12, 2),   # ltxcond.neg -> guider.negative
    (11, 0, 13, 0),  # NOISE -> sca.noise
    (12, 0, 13, 1),  # GUIDER-> sca.guider
    (9, 0, 13, 2),   # SAMPLER-> sca.sampler
    (10, 0, 13, 3),  # SIGMAS-> sca.sigmas
    (7, 0, 13, 4),   # i2v.latent -> sca.latent_image
    (13, 0, 14, 0),  # sca.output -> vaedt.samples
    (2, 2, 14, 1),   # VAE  -> vaedt.vae
    (14, 0, 15, 0),  # IMAGE-> savewebm.images
]

# ---- assemble litegraph ----------------------------------------------------
links = []
out_links = {(i, s): [] for i in N for s in range(len(N[i]["outs"]))}
in_link = {}
for lid, (sn, ss, dn, ds) in enumerate(WIRES, start=1):
    ltype = N[sn]["outs"][ss][1]
    links.append([lid, sn, ss, dn, ds, ltype])
    out_links[(sn, ss)].append(lid)
    in_link[(dn, ds)] = lid

nodes_json = []
for i in sorted(N):
    n = N[i]
    inputs = []
    for s, (nm, ty) in enumerate(n["ins"]):
        inputs.append({"name": nm, "type": ty, "link": in_link.get((i, s))})
    outputs = []
    for s, (nm, ty) in enumerate(n["outs"]):
        outputs.append({"name": nm, "type": ty, "slot_index": s,
                        "links": out_links[(i, s)]})
    nodes_json.append({
        "id": i, "type": n["type"], "pos": n["pos"], "size": n["size"],
        "flags": {}, "order": i - 1, "mode": 0,
        "inputs": inputs, "outputs": outputs,
        "properties": {"Node name for S&R": n["type"]},
        "widgets_values": n["widgets"],
    })

graph = {
    "last_node_id": max(N),
    "last_link_id": len(links),
    "nodes": nodes_json,
    "links": links,
    "groups": [],
    "config": {},
    "extra": {},
    "version": 0.4,
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    json.dump(graph, f, indent=2, ensure_ascii=False)
    f.write("\n")
print("WROTE", OUT)
print("nodes:", len(nodes_json), "links:", len(links))
