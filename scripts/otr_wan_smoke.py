"""otr_wan_smoke -- BARE-BONES isolated Wan 2.2 i2v motion smoke (2026-06-12).

Clone of otr_ltx_motion_smoke for the Wan video-engine eyeball (GO_FORWARD 1A).
One radio-console still in -> one short SILENT clip out (SaveWEBM so you can
EYEBALL MOTION), NO audio / Flux / HuMo / episode pipeline. Bare /prompt graph
on CORE Comfy Wan nodes (NOT the KJ wrapper), verified against /object_info:

  UNETLoader(wan2.2 14B fp8) -> ModelSamplingSD3(shift) -> KSampler.model
  CLIPLoader(umt5, type=wan) -> CLIPTextEncode x2
  VAELoader(wan_2.1_vae) + LoadImage(still)
  -> WanImageToVideo(w,h,length,batch) [pos,neg,latent]
  -> KSampler(uni_pc/simple) -> VAEDecode -> SaveWEBM

Determinism (V-7): FIXED motion-prompt STRING + FIXED seed, NO LLM in the loop --
the only variable is --seed. The seed is stamped into the clip filename.

Model names are passed as BASENAMES under the models root (NOT the absolute
OTR_WAN_I2V_CKPT path); we fail BEFORE /prompt if a name is not visible to Comfy.

A background NVML poller samples nvidia-smi every 2s and reports the whole-run
peak so the Path-A-vs-GGUF decision (does fp8 bust 14.5GB?) is data-driven.
Run AFTER the WAN-lane server is up on :8000.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000"
UNET = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
CLIP = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE = "wan_2.1_vae.safetensors"
STILL = "otr_ltx_smoke_still.png"   # in the comfy input dir (same still as LTX eval)
# motion-only b-roll prompt (radio console), <=188 chars, mirrors the LTX eval.
MOTION_PROMPT = ("Continuous shot, same console throughout. Tuning dial needle "
                 "sweeps rhythmically. Vacuum tubes pulse. Brass speaker grille "
                 "trembles. Dust motes drift. Slow handheld dolly forward.")
NEG = "low quality, worst quality, blurry, distorted, watermark, text, static"


def _get_json(path, timeout=30):
    with urllib.request.urlopen(URL + path, timeout=timeout) as r:
        return json.load(r)


def assert_model_visible(node, field, fname):
    info = _get_json("/object_info/%s" % node)[node]
    inp = info.get("input", {}) or {}
    enum = None
    for kind in ("required", "optional"):
        spec = (inp.get(kind, {}) or {}).get(field)
        if spec and isinstance(spec[0], list):
            enum = spec[0]
    if enum is None or fname not in enum:
        raise SystemExit(
            "FAIL preflight: %s.%s does NOT see %r (visible: %s)"
            % (node, field, fname, (enum or [])[:8]))
    print("[smoke] visible: %s.%s -> %s" % (node, field, fname))


def build_workflow(a) -> dict:
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": a.unet, "weight_dtype": a.weight_dtype}},
        "1m": {"class_type": "ModelSamplingSD3",
               "inputs": {"model": ["1", 0], "shift": a.shift}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": a.clip, "type": "wan", "device": "default"}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": a.prompt, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": a.negative, "clip": ["2", 0]}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": a.vae}},
        "6": {"class_type": "LoadImage", "inputs": {"image": a.still}},
        "7": {"class_type": "WanImageToVideo",
              "inputs": {"width": a.width, "height": a.height,
                         "length": a.length, "batch_size": 1,
                         "positive": ["3", 0], "negative": ["4", 0],
                         "vae": ["5", 0], "start_image": ["6", 0]}},
        "8": {"class_type": "KSampler",
              "inputs": {"seed": a.seed, "steps": a.steps, "cfg": a.cfg,
                         "sampler_name": a.sampler, "scheduler": a.scheduler,
                         "denoise": 1.0, "model": ["1m", 0],
                         "positive": ["7", 0], "negative": ["7", 1],
                         "latent_image": ["7", 2]}},
        "9": {"class_type": "VAEDecode",
              "inputs": {"samples": ["8", 0], "vae": ["5", 0]}},
        "10": {"class_type": "SaveWEBM",
               "inputs": {"images": ["9", 0],
                          "filename_prefix": "otr_wanmotion/%s_seed%d" % (a.tag, a.seed),
                          "codec": "vp9", "fps": float(a.fps), "crf": 18.0}},
    }


class VramPoller(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._stop = threading.Event()
        self.peak = 0

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                mb = int(out.stdout.strip().splitlines()[0])
                self.peak = max(self.peak, mb)
            except Exception:
                pass
            self._stop.wait(2.0)

    def stop(self):
        self._stop.set()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--unet", default=UNET)
    p.add_argument("--clip", default=CLIP)
    p.add_argument("--vae", default=VAE)
    p.add_argument("--still", default=STILL)
    p.add_argument("--weight_dtype", default="default")
    p.add_argument("--shift", type=float, default=8.0)   # 14B sigma shift
    p.add_argument("--sampler", default="uni_pc")
    p.add_argument("--scheduler", default="simple")
    p.add_argument("--length", type=int, default=81)     # 4n+1
    p.add_argument("--cfg", type=float, default=3.5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=16)        # Wan 14B native cadence
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default=MOTION_PROMPT)
    p.add_argument("--negative", default=NEG)
    p.add_argument("--tag", default="i2v14b")
    a = p.parse_args()

    try:
        urllib.request.urlopen(URL + "/system_stats", timeout=5)
    except Exception as e:
        print("FAIL: server not reachable: %s" % e)
        return 1

    # preflight: fail BEFORE /prompt if a model name is not visible to Comfy
    assert_model_visible("UNETLoader", "unet_name", a.unet)
    assert_model_visible("CLIPLoader", "clip_name", a.clip)
    assert_model_visible("VAELoader", "vae_name", a.vae)

    wf = build_workflow(a)
    print("[smoke] tag=%s seed=%d sampler=%s shift=%.1f steps=%d cfg=%.1f "
          "len=%d %dx%d fps=%d prompt_len=%d"
          % (a.tag, a.seed, a.sampler, a.shift, a.steps, a.cfg, a.length,
             a.width, a.height, a.fps, len(a.prompt)), flush=True)
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        URL + "/prompt", data=json.dumps({"prompt": wf}).encode(),
        headers={"Content-Type": "application/json"}), timeout=20))
    if "error" in r or r.get("node_errors"):
        print("FAIL: rejected: %s"
              % json.dumps(r.get("node_errors", r.get("error")), indent=2)[:1500])
        return 1
    pid = r.get("prompt_id")
    print("[smoke] queued prompt_id=%s" % pid, flush=True)
    poller = VramPoller()
    poller.start()
    t0 = time.time()
    deadline = t0 + 1200
    rc = 1
    while time.time() < deadline:
        try:
            entry = _get_json("/history/%s" % pid, timeout=10).get(pid)
        except Exception:
            entry = None
        if entry:
            st = entry.get("status", {})
            ss = st.get("status_str", "")
            errs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
            if errs:
                print("\nFAIL: execution_error: %s" % json.dumps(errs, indent=2)[:1800])
                break
            if ss == "error":
                print("\nFAIL: status=error: %s" % st.get("messages"))
                break
            if ss == "success" or entry.get("outputs"):
                files = []
                for _nid, o in entry.get("outputs", {}).items():
                    for key in ("images", "gifs", "video", "files"):
                        for f in (o.get(key) or []):
                            if isinstance(f, dict):
                                files.append("%s/%s" % (f.get("subfolder", ""),
                                                        f.get("filename", "")))
                print("\nPASS: rendered in %.0fs -> %s" % (time.time() - t0, files),
                      flush=True)
                rc = 0
                break
        time.sleep(3)
        sys.stdout.write("."); sys.stdout.flush()
    else:
        print("\nFAIL: timed out")
    poller.stop()
    time.sleep(0.1)
    print("[smoke] WHOLE-RUN NVML PEAK: %d MB (telemetry only; the OOM budget "
          "is owned by the external per-hardware tier JSON)" % poller.peak)
    return rc


if __name__ == "__main__":
    sys.exit(main())
