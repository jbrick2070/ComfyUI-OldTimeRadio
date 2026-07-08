"""
build_ltx_av_q_bakeoff_workflow.py
==================================

Emit the STANDALONE, ISOLATED LTX-AV *QUALITY* bakeoff workflow JSON
(``scripts/otr_ltx_av_q_bakeoff_distilled_native.json``).

This is the retained isolated builder for the 2026-06-27 LTX-AV quality bakeoff
(PLAN v5). The differences from the retired quant-bakeoff builder are deliberate
and load-bearing:

  1. RECIPE = ``distilled_native`` (NOT sharp_lora). The graph drops the SHARP
     LoRA entirely: UnetLoaderGGUF feeds CFGGuider.model DIRECTLY. NO
     LoraLoaderModelOnly, NO ModelSamplingLTXV, NO LTXVScheduler. This MATCHES
     the production ``distilled_native`` path exactly (eng_ltx_av.py
     _recipe_config: use_lora=False, use_modelsampling=False, manual_sigmas=True,
     euler_cfg_pp, cfg 1.0, i2v strength 0.75) -- the soak ran the distilled-1.1
     Q3_K_M unet with the DEV video/audio VAE + DEV projection + Gemma-3 encoder
     (the engine's env defaults; the recipe swaps ONLY the LoRA, never the VAEs).

  2. CANVAS = 512x288, LENGTH = 153 (>=153f, 8n+1). This is the NATIVE render the
     production render_driver.py:1116 clamp produces today; the bakeoff measures
     it directly (the softness root is the tiny native render, then the
     ~8.3x-area composite upscale -- both isolated here).

  3. OUTPUT = SILENT, via SaveImage (a lossless PNG frame batch) -- NOT the legacy
     CreateVideo/SaveVideo audio-mux. The runner reads the saved frames and
     encodes them with the PRODUCTION encoder
     ``wrapper_bridge.encode_frames_to_silent_mp4`` (libx264 crf18, bt709, CPU,
     NO audio track), so a winner wires byte-for-byte and the measured clip is
     the real production artifact -- not a differently-encoded, audio-muxed one.

This builder emits the L0 BASELINE graph (512x288 / temporal 64-8 / spatial
512-64 / native distilled sigmas / i2v 0.75). The runner
(scripts/run_ltx_av_q_bakeoff.py) mutates ONE lever per leg ON THE CONVERTED API
PROMPT (canvas / temporal+spatial tile / sigmas-text / i2v_strength) and writes a
FAIL-LOUD resolved-API-prompt manifest before every render.

Fixed pre-baked inputs (real beat b002 of episode
signal_lost_keystrokes_of_denial_20260617 -- the same pair the quant bakeoff used):
  LoadImage = c02_466a19906ccb.png   (HAYES VANCE portrait, WIDE-matched)
  LoadAudio = c02_b002_line.wav      (his real beat-b002 driving line)

Lives under scripts/ (NOT workflows/) ON PURPOSE: tests/test_workflow_json_guardrails.py
globs workflows/*.json top-level and audits widget-count vs live INPUT_TYPES; this
is an isolated bakeoff harness input, not a production workflow.

UTF-8, no BOM. ASCII-only content.
"""

import json
import os

# --- constants (production distilled_native; eng_ltx_av.py) ------------------
# The daily-driver distilled unet (the 2026-06-26 bakeoff winner). The runner can
# override per leg, but the baseline JSON pins it so the manifest reads a real value.
UNET_DEFAULT = os.path.join("distilled-1.1",
                            "ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")
# DEV companions -- the production distilled_native path reuses these (the recipe
# swaps ONLY the LoRA; OTR_LTX_AV_VIDEO_VAE/AUDIO_VAE/PROJECTION are NOT set by the
# 30-word smoke / the soak, so the engine env defaults -- dev -- are what ran).
ENCODER = "gemma_3_12B_it_fp4_mixed.safetensors"
PROJECTION_CKPT = "ltx-2.3-22b-dev.safetensors"
VIDEO_VAE = "ltx-2.3-22b-dev_video_vae.safetensors"
AUDIO_VAE = "ltx-2.3-22b-dev_audio_vae.safetensors"

# The 8-step distilled sigma schedule (eng_ltx_av.LTX_DISTILLED_SIGMAS).
DISTILLED_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975,
                    0.909375, 0.725, 0.421875, 0.0)
SIGMAS_TEXT = ", ".join(repr(s) for s in DISTILLED_SIGMAS)

CFG = 1.0
SAMPLER = "euler_cfg_pp"
I2V_STRENGTH = 0.75
FPS = 25.0
# L0 baseline canvas + the production decode tiling (eng_ltx_av.py:556-559).
CANVAS_W, CANVAS_H, LENGTH = 512, 288, 153
DECODE_TILE, DECODE_OVERLAP = 512, 64
DECODE_TEMPORAL_SIZE, DECODE_TEMPORAL_OVERLAP = 64, 8
SEED = 0

IMAGE_NAME = "c02_466a19906ccb.png"
AUDIO_NAME = "c02_b002_line.wav"
# the runner writes the SILENT clip; SaveImage drops a lossless PNG frame batch the
# runner reads + encodes via the production encode_frames_to_silent_mp4.
FRAMES_PREFIX = "otr/episodes/_bakeoff_ltxq/_frames/L0/frame"
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
    "LTX-AV QUALITY BAKEOFF -- standalone, isolated distilled_native A2V graph.\n\n"
    "RECIPE = distilled_native (matches production): NO LoRA, NO ModelSamplingLTXV,\n"
    "NO LTXVScheduler. UnetLoaderGGUF -> CFGGuider.model directly; fixed 8-step\n"
    "distilled sigmas (Sigmas From Text) + euler_cfg_pp + cfg 1.0 + i2v 0.75.\n"
    "distilled-1.1 Q3_K_M unet + DEV video/audio VAE + DEV projection + Gemma-3.\n\n"
    "L0 BASELINE: 512x288x153, decode tile 512/64, temporal 64/8.\n"
    "The runner (scripts/run_ltx_av_q_bakeoff.py) varies ONE lever per leg on the\n"
    "converted API prompt (canvas / temporal+spatial tile / sigmas-text / i2v),\n"
    "writes a FAIL-LOUD resolved-API-prompt manifest, then renders.\n\n"
    "OUTPUT = SILENT: SaveImage drops a lossless PNG frame batch; the runner encodes\n"
    "it via the production wrapper_bridge.encode_frames_to_silent_mp4 (libx264 crf18,\n"
    "bt709, CPU, NO audio) -> otr/episodes/_bakeoff_ltxq/<leg>.mp4 for side-by-side QA.")


# --- tiny litegraph builder (same shape as the quant bakeoff) ----------------
class Graph:
    def __init__(self):
        self.nodes = []
        self.links = []  # [link_id, src_node, src_slot, dst_node, dst_slot, type]
        self._nid = 0
        self._lid = 0
        self._col = 0

    def add(self, ntype, widgets=None, inputs=None, outputs=None, pos=None):
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
        ltype = src["outputs"][src_slot]["type"]
        self._lid += 1
        lid = self._lid
        self.links.append([lid, src_nid, src_slot, dst_nid, dst_slot, ltype])
        src["outputs"][src_slot]["links"].append(lid)
        self._node(dst_nid)["inputs"][dst_slot]["link"] = lid
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

    # distilled_native: NO LoraLoaderModelOnly -- the unet feeds the guider direct.
    unet = g.add("UnetLoaderGGUF", widgets=[UNET_DEFAULT],
                 outputs=[("MODEL", "MODEL")])
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
    noise = g.add("RandomNoise", widgets=[SEED, "fixed"], outputs=[("NOISE", "NOISE")])
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
    decode = g.add("VAEDecodeTiled",
                   widgets=[DECODE_TILE, DECODE_OVERLAP,
                            DECODE_TEMPORAL_SIZE, DECODE_TEMPORAL_OVERLAP],
                   inputs=[("samples", "LATENT"), ("vae", "VAE")],
                   outputs=[("IMAGE", "IMAGE")])
    # SILENT terminal: a lossless PNG frame batch (NO CreateVideo/SaveVideo
    # audio-mux). The runner reads these frames + encodes via the production
    # encode_frames_to_silent_mp4.
    saveimg = g.add("SaveImage", widgets=[FRAMES_PREFIX],
                    inputs=[("images", "IMAGE")])
    g.add("Note", widgets=[NOTE], pos=[60, 1300])

    # --- wiring (distilled_native: unet -> guider direct; NO LoRA) -----------
    g.connect(unet, 0, guider, 0)          # unet -> guider.model (no LoRA wrap)
    g.connect(te, 0, pos, 0)
    g.connect(te, 0, neg, 0)
    g.connect(pos, 0, cond, 0)
    g.connect(neg, 0, cond, 1)
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
    g.connect(decode, 0, saveimg, 0)       # SILENT: frames -> SaveImage

    return g.dump()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.normpath(os.path.join(
        here, "otr_ltx_av_q_bakeoff_distilled_native.json"))
    data = build()
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    print("wrote %s (%d nodes, %d links)"
          % (out, data["last_node_id"], data["last_link_id"]))


if __name__ == "__main__":
    main()
