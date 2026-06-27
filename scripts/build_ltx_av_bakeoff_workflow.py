"""
build_ltx_av_bakeoff_workflow.py
================================

Emit the STANDALONE, ISOLATED LTX-AV quant-bakeoff workflow JSON
(``workflows/ltx_av_bakeoff_gguf.json``).

This is NOT the production pipeline -- it is a minimal LTX-AV A2V render graph
whose ONLY swappable artifact is the GGUF transformer (the ``UnetLoaderGGUF``
node). Every other artifact (Gemma encoder, projection ckpt, video/audio VAE),
the fixed still + the fixed driving-line audio, and the sampler recipe are held
constant so a per-quant speed (s/it) + peak-VRAM + side-by-side quality compare
is apples-to-apples.

Recipe == the production SHARP path from
``nodes/_otr_video_engines/eng_ltx_av.py`` (2026-06-17):
  distilled LoRA @0.70 + euler_cfg_pp + the 8-step LTX_DISTILLED_SIGMAS (via the
  RES4LYF "Sigmas From Text" node, the JSON-expressible stand-in for the engine's
  in-adapter _SigmasFromValues) + cfg 1.0 + LTXVImgToVideo strength 0.75, with
  ModelSamplingLTXV + LTXVScheduler DROPPED (the LoRA + fixed sigmas carry the
  shift; ModelSamplingLTXV would double-shift -> blur).

The runner (scripts/run_ltx_av_bakeoff.py) mutates ONLY:
  * UnetLoaderGGUF.unet_name   -> the quant under test
  * SaveVideo.filename_prefix  -> otr/episodes/_bakeoff/<quant>
  * the distilled runs DROP the SHARP LoRA (the distilled-1.1 GGUF already bakes
    the distillation in): the LoraLoaderModelOnly node is removed and CFGGuider
    is repointed straight at the UnetLoaderGGUF MODEL.

Fixed pre-baked inputs (real beat b002 of episode
signal_lost_keystrokes_of_denial_20260617, the same pair the talk mini uses):
  LoadImage = c02_466a19906ccb.png   (HAYES VANCE portrait, WIDE-matched)
  LoadAudio = c02_b002_line.wav      (his real beat-b002 line)

UTF-8, no BOM. ASCII-only content.
"""

import json
import os

# --- constants (production-faithful; eng_ltx_av.py) -------------------------
UNET_DEFAULT = "ltx-2.3-22b-dev-Q3_K_M.gguf"
ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
PROJECTION_CKPT = "ltx-2.3-22b-dev.safetensors"
VIDEO_VAE = "ltx-2.3-22b-dev_video_vae.safetensors"
AUDIO_VAE = "ltx-2.3-22b-dev_audio_vae.safetensors"
# loras-folder-relative; backslash matches the Windows folder_paths listing.
SHARP_LORA = os.path.join("ltxv", "ltx2",
                          "ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
SHARP_LORA_STRENGTH = 0.7

# The 8-step distilled sigma schedule (eng_ltx_av.LTX_DISTILLED_SIGMAS).
DISTILLED_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975,
                    0.909375, 0.725, 0.421875, 0.0)
SIGMAS_TEXT = ", ".join(repr(s) for s in DISTILLED_SIGMAS)

CFG = 1.0
SAMPLER = "euler_cfg_pp"
I2V_STRENGTH = 0.75
FPS = 25.0
CANVAS_W, CANVAS_H, LENGTH = 832, 480, 105

IMAGE_NAME = "c02_466a19906ccb.png"
AUDIO_NAME = "c02_b002_line.wav"
NEGATIVE = ("low quality, worst quality, blurry, jpeg artifacts, distorted, "
            "deformed, static, frozen pose, still image, watermark, text")
POSITIVE = ("Medium close-up of HAYES VANCE, a 50s lead researcher -- oval "
            "face, heavy brows, aquiline nose, thin lips, strong jawline, a "
            "small scar on the left cheek; shoulders hunched, speaking with "
            "tense, intrigued unease. Behind him a bustling neon-lit metropolis "
            "glows through rain-streaked glass at midnight. Cinematic 35mm "
            "film, moody volumetric lighting, sharp focus, talking head, "
            "subtle natural head motion.")

NOTE = (
    "LTX-AV GGUF QUANT BAKEOFF -- standalone, isolated A2V render graph.\n\n"
    "SWAPPABLE: only the UnetLoaderGGUF transformer changes between runs.\n"
    "  dev          = ltx-2.3-22b-dev-Q3_K_M.gguf            (SHARP LoRA ON)\n"
    "  distilled-1.1 = distilled-1.1/...-Q2_K|Q3_K_M|Q4_K_M  (SHARP LoRA OFF)\n\n"
    "The runner (scripts/run_ltx_av_bakeoff.py) drops the LoraLoaderModelOnly\n"
    "node for the distilled runs (the distilled GGUF bakes the distillation in)\n"
    "and repoints CFGGuider.model straight at UnetLoaderGGUF.\n\n"
    "Recipe == production SHARP (eng_ltx_av.py): euler_cfg_pp + 8-step distilled\n"
    "sigmas (Sigmas From Text) + cfg 1.0 + i2v strength 0.75; ModelSamplingLTXV\n"
    "and LTXVScheduler are DROPPED. Canvas 832x480x105, Gemma on CPU.\n\n"
    "Per-quant clip -> otr/episodes/_bakeoff/<quant>.mp4 (muxed with the driving\n"
    "line audio so you can hear the lip-sync). s/it + peak VRAM logged by the\n"
    "runner to otr/episodes/_bakeoff/bakeoff_results.json + .md.")


# --- tiny litegraph builder -------------------------------------------------
class Graph:
    def __init__(self):
        self.nodes = []
        self.links = []  # [link_id, src_node, src_slot, dst_node, dst_slot, type]
        self._nid = 0
        self._lid = 0
        self._x = 60
        self._col = 0

    def add(self, ntype, widgets=None, inputs=None, outputs=None, pos=None):
        """inputs:  list of (name, type)
           outputs: list of (name, type)   (links filled in by connect())"""
        self._nid += 1
        nid = self._nid
        if pos is None:
            col = self._col
            pos = [60 + (col % 6) * 360, 60 + (col // 6) * 260]
            self._col += 1
        node = {
            "id": nid,
            "type": ntype,
            "pos": pos,
            "size": [330, 100],
            "flags": {},
            "order": nid - 1,
            "mode": 0,
            "inputs": [{"name": n, "type": t, "link": None}
                       for (n, t) in (inputs or [])],
            "outputs": [{"name": n, "type": t, "slot_index": i, "links": []}
                        for i, (n, t) in enumerate(outputs or [])],
            "properties": {"Node name for S&R": ntype},
            "widgets_values": list(widgets) if widgets else [],
        }
        self.nodes.append(node)
        return nid

    def _node(self, nid):
        for n in self.nodes:
            if n["id"] == nid:
                return n
        raise KeyError(nid)

    def connect(self, src_nid, src_slot, dst_nid, dst_slot):
        src = self._node(src_nid)
        dst = self._node(dst_nid)
        ltype = src["outputs"][src_slot]["type"]
        self._lid += 1
        lid = self._lid
        self.links.append([lid, src_nid, src_slot, dst_nid, dst_slot, ltype])
        src["outputs"][src_slot]["links"].append(lid)
        dst["inputs"][dst_slot]["link"] = lid
        return lid

    def dump(self):
        return {
            "last_node_id": self._nid,
            "last_link_id": self._lid,
            "nodes": self.nodes,
            "links": self.links,
            "groups": [],
            "config": {},
            "extra": {},
            "version": 0.4,
        }


def build():
    g = Graph()

    unet = g.add("UnetLoaderGGUF", widgets=[UNET_DEFAULT],
                 outputs=[("MODEL", "MODEL")])
    lora = g.add("LoraLoaderModelOnly",
                 widgets=[SHARP_LORA, SHARP_LORA_STRENGTH],
                 inputs=[("model", "MODEL")], outputs=[("MODEL", "MODEL")])
    te = g.add("LTXAVTextEncoderLoader",
               widgets=[ENCODER, PROJECTION_CKPT, "cpu"],
               outputs=[("CLIP", "CLIP")])
    pos = g.add("CLIPTextEncode", widgets=[POSITIVE],
                inputs=[("clip", "CLIP")], outputs=[("CONDITIONING", "CONDITIONING")])
    neg = g.add("CLIPTextEncode", widgets=[NEGATIVE],
                inputs=[("clip", "CLIP")], outputs=[("CONDITIONING", "CONDITIONING")])
    cond = g.add("LTXVConditioning", widgets=[FPS],
                 inputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING")],
                 outputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING")])
    vae_dec = g.add("VAELoader", widgets=[VIDEO_VAE], outputs=[("VAE", "VAE")])
    vae_enc = g.add("VAELoader", widgets=[VIDEO_VAE], outputs=[("VAE", "VAE")])
    avae = g.add("VAELoader", widgets=[AUDIO_VAE], outputs=[("VAE", "VAE")])
    loadaudio = g.add("LoadAudio", widgets=[AUDIO_NAME], outputs=[("AUDIO", "AUDIO")])
    audioenc = g.add("LTXVAudioVAEEncode",
                     inputs=[("audio", "AUDIO"), ("audio_vae", "VAE")],
                     outputs=[("LATENT", "LATENT")])
    loadimg = g.add("LoadImage", widgets=[IMAGE_NAME, "image"],
                    outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")])
    i2v = g.add("LTXVImgToVideo",
                widgets=[CANVAS_W, CANVAS_H, LENGTH, 1, I2V_STRENGTH],
                inputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                        ("vae", "VAE"), ("image", "IMAGE")],
                outputs=[("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                         ("latent", "LATENT")])
    concat = g.add("LTXVConcatAVLatent",
                   inputs=[("video_latent", "LATENT"), ("audio_latent", "LATENT")],
                   outputs=[("LATENT", "LATENT")])
    noise = g.add("RandomNoise", widgets=[0, "fixed"], outputs=[("NOISE", "NOISE")])
    ksel = g.add("KSamplerSelect", widgets=[SAMPLER], outputs=[("SAMPLER", "SAMPLER")])
    guider = g.add("CFGGuider", widgets=[CFG],
                   inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                           ("negative", "CONDITIONING")],
                   outputs=[("GUIDER", "GUIDER")])
    sigmas = g.add("Sigmas From Text", widgets=[SIGMAS_TEXT],
                   outputs=[("SIGMAS", "SIGMAS")])
    sampler = g.add("SamplerCustomAdvanced",
                    inputs=[("noise", "NOISE"), ("guider", "GUIDER"),
                            ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
                            ("latent_image", "LATENT")],
                    outputs=[("output", "LATENT"), ("denoised_output", "LATENT")])
    sep = g.add("LTXVSeparateAVLatent",
                inputs=[("av_latent", "LATENT")],
                outputs=[("video_latent", "LATENT"), ("audio_latent", "LATENT")])
    decode = g.add("VAEDecodeTiled", widgets=[512, 64, 64, 8],
                   inputs=[("samples", "LATENT"), ("vae", "VAE")],
                   outputs=[("IMAGE", "IMAGE")])
    createvid = g.add("CreateVideo", widgets=[FPS],
                      inputs=[("images", "IMAGE"), ("audio", "AUDIO")],
                      outputs=[("VIDEO", "VIDEO")])
    savevid = g.add("SaveVideo",
                    widgets=["otr/episodes/_bakeoff/clip", "mp4", "h264"],
                    inputs=[("video", "VIDEO")])
    g.add("Note", widgets=[NOTE], pos=[60, 1300])

    # --- wiring (SHARP: lora-wrapped unet -> guider; ManualSigmas carries shift)
    g.connect(unet, 0, lora, 0)            # unet -> lora.model
    g.connect(lora, 0, guider, 0)          # lora -> guider.model
    g.connect(te, 0, pos, 0)
    g.connect(te, 0, neg, 0)
    g.connect(pos, 0, cond, 0)
    g.connect(neg, 0, cond, 1)
    # i2v conditioning carries pos/neg through to the guider (production path)
    g.connect(cond, 0, i2v, 0)
    g.connect(cond, 1, i2v, 1)
    g.connect(vae_enc, 0, i2v, 2)          # encode-side video VAE (freed early)
    g.connect(loadimg, 0, i2v, 3)
    g.connect(i2v, 0, guider, 1)           # i2v.positive -> guider.positive
    g.connect(i2v, 1, guider, 2)           # i2v.negative -> guider.negative
    g.connect(loadaudio, 0, audioenc, 0)
    g.connect(avae, 0, audioenc, 1)
    g.connect(i2v, 2, concat, 0)           # video latent
    g.connect(audioenc, 0, concat, 1)      # audio latent
    g.connect(noise, 0, sampler, 0)
    g.connect(guider, 0, sampler, 1)
    g.connect(ksel, 0, sampler, 2)
    g.connect(sigmas, 0, sampler, 3)
    g.connect(concat, 0, sampler, 4)
    g.connect(sampler, 0, sep, 0)
    g.connect(sep, 0, decode, 0)           # video latent only (audio dropped, V-1)
    g.connect(vae_dec, 0, decode, 1)       # decode-side video VAE
    g.connect(decode, 0, createvid, 0)
    g.connect(loadaudio, 0, createvid, 1)  # mux driving line audio (audible clip)
    g.connect(createvid, 0, savevid, 0)

    return g.dump()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "workflows", "ltx_av_bakeoff_gguf.json")
    out = os.path.normpath(out)
    data = build()
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print("wrote %s (%d nodes, %d links)"
          % (out, data["last_node_id"], data["last_link_id"]))


if __name__ == "__main__":
    main()
