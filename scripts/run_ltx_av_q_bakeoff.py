"""
run_ltx_av_q_bakeoff.py
=======================

Headless LTX-AV *QUALITY* bakeoff (PLAN v5, 2026-06-27). Isolated, distilled_native
A2V sweep that fixes the soft/512x288 + temporal-seam "flash" + init-hold stutter on
the production ltx_audio_in clips. Renders ONE fixed still+audio through a graph that
MATCHES production distilled_native, varying ONE lever per leg, and writes each SILENT
clip + per-leg metrics to ``output/otr/episodes/_bakeoff_ltxq/<leg>.mp4`` for operator
+ AI side-by-side QA.

Scope (CLAUDE.md): this harness touches ONLY this script + the isolated workflow
``scripts/otr_ltx_av_q_bakeoff_distilled_native.json``. It submits a RAW core-node API
graph over HTTP, so it does NOT import or alter eng_ltx_av.py / eng_humo.py /
eng_wan_ti2v.py -- the engines stay sacred. The ONE production module it touches is the
COLD-IMPORT-CLEAN ``wrapper_bridge`` (stdlib-only at module scope, V-12), imported by
FILE PATH to reuse the EXACT production silent encoder (``encode_frames_to_silent_mp4``)
so a winning leg wires byte-for-byte.

Boot-per-leg (NOT the legacy single-session loop): each leg does its own
reset_box() + boot + one /prompt + teardown. Two independent reasons (the 2026-06-26
convergence catch): (1) ``--reserve-vram`` is startup-only; (2) ComfyUI deliberately
does NOT force-unload between prompts (wrapper_bridge _soft_free, BUG-265 anti-frag), so
a prior leg's resident unet/VAE contaminates the next peak-VRAM measurement. A fresh boot
is the only clean per-leg peak.

THE LOAD-BEARING RAIL -- the per-leg FAIL-LOUD resolved-API-PROMPT manifest: before any
render, the runner asserts the CONVERTED+MUTATED API prompt actually carries the
distilled_native recipe at the leg's lever values (canvas / length / sigmas-text /
temporal+spatial-tile / i2v / unet=distilled / LoRA+ModelSamplingLTXV+LTXVScheduler
ABSENT / cfg=1.0 / euler_cfg_pp / seed / fps + prompt+asset SHA + boot reserve +
baseline VRAM + audio-ABSENT). Any mismatch ABORTS the leg -- the #1 risk is silently
measuring the WRONG graph (the legacy quant builder is SHARP/LoRA @832x480x105). WATCH
THE FIRST MANIFEST (--dry-validate prints them all, no GPU) before the L0 render.

Stages (PLAN v5):
  L0     baseline 512x288 / temporal 64-8 / native sigmas / i2v 0.75 (the reference).
  Stage0 FREE composite-scaler A/B on L0's clip (bilinear vs lanczos+unsharp). No GPU.
  Stage1 DECODE temporal sweep {64-16, 64-32, 128-16, 128-32, +4096-8 whole-clip LAST}.
  Stage2 SAMPLER 2x2 {i2v 0.75/0.62} x {native/re-spaced sigmas} on the best decode.
  Stage3 CANVAS {640x384, 704x384} on the stage1+2 winner; reserve 4 -> 5-6 on spill.

Per-leg gates (objective): ffprobe matches the manifest; peak VRAM recorded as telemetry
only (no ceiling gate -- the OOM budget is owned by the external per-hardware tier JSON);
s/it <= 30 AND <= 2.5x the 512 baseline (LOAD-BEARING abort -- prevents the 223 s/it sysmem
spiral); scene-cuts gt(scene,0.20)=0; freezedetect(-55dB:d=0.12) 0 freezes >=0.12s OR
>=75% reduction vs L0; temporal seam (dynamic indices) p99-luma jump <2.5 AND <=1.5x the
local median, else rank by >=75% seam-reduction-vs-L0; sharpness (Stage 0 + Stage 3 only)
Laplacian variance >=15% over the L0-conformed-with-the-Stage-0-winning-scaler.

  python run_ltx_av_q_bakeoff.py --dry-validate   # boot once, print every manifest, NO render
  python run_ltx_av_q_bakeoff.py                  # full sweep (GPU; serialize behind any other render)

STOP after the sweep + report; the operator picks the winner before any production wiring.
UTF-8, no BOM. ASCII-only.
"""

import argparse
import copy
import glob
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(REPO, "tests")
for _p in (REPO, TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _run_baseline as RB  # noqa: E402  (proven workflow->API + submit/poll)

# --- import the PRODUCTION silent encoder by FILE PATH (V-12 cold-import clean;
#     bypasses the nodes package __init__ / ComfyUI / torch) so a winning leg
#     wires byte-for-byte to production. ---------------------------------------
_WB_PATH = os.path.join(REPO, "nodes", "_otr_video_engines", "wrapper_bridge.py")
_wb_spec = importlib.util.spec_from_file_location("otr_wb_ltxq_bakeoff", _WB_PATH)
_WB = importlib.util.module_from_spec(_wb_spec)
_wb_spec.loader.exec_module(_WB)

COMFY_ROOT = r"C:\Users\jeffr\Documents\ComfyUI"
VENV_PY = r"C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
MODELS_UNET = r"C:\ComfyUI-Models\unet"
SRC_INPUT = r"C:\Users\jeffr\Documents\ComfyUI\input"
SERVER_INPUT = r"C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\input"
FIXED_ASSETS = ["c02_466a19906ccb.png", "c02_b002_line.wav"]
STILL_NAME, AUDIO_NAME = FIXED_ASSETS[0], FIXED_ASSETS[1]
OUTPUT_BASE = os.environ.get("COMFYUI_OUTPUT", os.path.join(COMFY_ROOT, "output"))
BAKEOFF_DIR = os.path.join(OUTPUT_BASE, "otr", "episodes", "_bakeoff_ltxq")
FRAMES_ROOT = os.path.join(BAKEOFF_DIR, "_frames")
WORKFLOW = os.path.join(REPO, "scripts", "otr_ltx_av_q_bakeoff_distilled_native.json")
BASE_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8000")
PORT = 8000

FPS = 25.0
STEPS = 8                       # 8-step distilled schedule (wall/steps s/it fallback)
SIT_ABS_CAP = 30.0              # s/it absolute ceiling (LOAD-BEARING abort)
SIT_REL_CAP = 2.5               # s/it <= 2.5x the 512 baseline
RESERVE_GB = 4.0                # production reserve (eng_ltx_av.py:90); step up on spill
COMPOSITE_W, COMPOSITE_H = 1472, 832   # the production composite canvas

# the production distilled sigma schedule (eng_ltx_av.LTX_DISTILLED_SIGMAS) ----
NATIVE_SIGMAS = (1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0)
# RE-SPACED 8-step gamble (Gemini sigma-boiling hypothesis): native bunches 5/8 steps
# above 0.97 then drops 0.42 -> 0.0 in one step; this spreads the denoise into smoothly
# increasing steps (no near-1.0 bunch, no terminal cliff). Native stays the safe default.
RESPACED_SIGMAS = (1.0, 0.95, 0.88, 0.79, 0.68, 0.55, 0.40, 0.22, 0.0)


def sigmas_text(tpl):
    return ", ".join(repr(s) for s in tpl)


def log(msg):
    print("[ltxq] %s" % msg, flush=True)


# --------------------------------------------------------------------------- #
# leg spec
# --------------------------------------------------------------------------- #
def make_leg(label, stage, *, w=512, h=288, length=153,
             tile=512, overlap=64, t_size=64, t_overlap=8,
             i2v=0.75, sigmas="native", reserve=RESERVE_GB, sharpness=False):
    """One bakeoff leg. ``sharpness`` legs (Stage 0/3) also conform + Laplacian."""
    return {"label": label, "stage": stage, "w": w, "h": h, "length": length,
            "tile": tile, "overlap": overlap, "t_size": t_size, "t_overlap": t_overlap,
            "i2v": i2v, "sigmas": sigmas, "reserve": reserve, "sharpness": sharpness}


def spatial_tile_for(width):
    """Whole-frame spatial decode (no spatial seams; we hunt TEMPORAL seams). Tile =
    the canvas width rounded to the LTX /32 spatial base; overlap stays 64."""
    return max(512, int(round(width / 32.0)) * 32)


# --------------------------------------------------------------------------- #
# box reset (CLAUDE.md S4: selective CIM kill, NEVER a blanket python kill)
# --------------------------------------------------------------------------- #
def reset_box():
    ps = r"""
$ErrorActionPreference='SilentlyContinue'
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match 'ComfyUI\\main\.py' -or
                 ($_.CommandLine -match 'main\.py' -and $_.CommandLine -match '--port\s+8000') } |
  ForEach-Object { Write-Output ("kill server pid " + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force }
Get-NetTCPConnection -LocalPort 8000 -State Listen |
  ForEach-Object { Write-Output ("kill port-owner pid " + $_.OwningProcess); Stop-Process -Id $_.OwningProcess -Force }
Start-Sleep -Seconds 2
$used = (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
         Where-Object { $_.CommandLine -match 'main\.py' }).Count
Write-Output ("residual comfy server procs: " + $used)
"""
    r = subprocess.run(["powershell.exe", "-NoProfile", "-Command", ps],
                       capture_output=True, text=True, timeout=60)
    for ln in (r.stdout or "").splitlines():
        if ln.strip():
            log("reset: " + ln.strip())


# --------------------------------------------------------------------------- #
# server lifecycle (boot PER LEG -- reserve is startup-only)
# --------------------------------------------------------------------------- #
SOAK_LAUNCH_CMD = os.path.join(REPO, "scripts", "_otr_soak_server_launch.cmd")


def boot_server(server_log_path, reserve_gb=RESERVE_GB):
    """Boot headless ComfyUI via the proven launcher (CLAUDE.md S5) with the FLOOR
    profile (heavy OTR engines OFF; the core + GGUF + LTX node classes still
    register). ``reserve_gb`` rides OTR_HEADLESS_RESERVE_VRAM_GB -> --reserve-vram
    <gb> (reserve that many GB free, matching the production 4 GB)."""
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("OTR_TEST_MODE", None)
    if reserve_gb:
        env["OTR_HEADLESS_RESERVE_VRAM_GB"] = str(reserve_gb)
    else:
        env.pop("OTR_HEADLESS_RESERVE_VRAM_GB", None)
    log("booting ComfyUI :%d (reserve %s GB) -> %s"
        % (PORT, reserve_gb, server_log_path))
    proc = subprocess.Popen([SOAK_LAUNCH_CMD, server_log_path, "FLOOR"],
                            cwd=os.path.dirname(SOAK_LAUNCH_CMD), env=env)
    return proc


def wait_ready(timeout_s=180):
    import requests
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            r = requests.get(BASE_URL + "/object_info", timeout=10)
            if r.status_code == 200:
                log("server ready in %.0fs" % (time.time() - start))
                return r.json()
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError("ComfyUI did not become ready within %ds" % timeout_s)


# --------------------------------------------------------------------------- #
# NVML peak-VRAM sampler (live peak readable mid-render; recorded telemetry)
# --------------------------------------------------------------------------- #
class VramPeak:
    def __init__(self, interval=0.25):
        self.interval = interval
        self.peak_mb = 0
        self.baseline_mb = None
        self._stop = threading.Event()
        self._t = None

    def _run(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            first = True
            while not self._stop.is_set():
                used = pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 * 1024)
                if first:
                    self.baseline_mb = round(used, 1)
                    first = False
                if used > self.peak_mb:
                    self.peak_mb = used
                time.sleep(self.interval)
            pynvml.nvmlShutdown()
        except Exception as e:  # noqa: BLE001
            log("NVML sampler error: %s" % e)
            self.peak_mb = -1

    def __enter__(self):
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        time.sleep(0.4)  # let the baseline sample land
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=3)


def nvml_used_mb():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mb = pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 * 1024)
        pynvml.nvmlShutdown()
        return round(mb, 1)
    except Exception:  # noqa: BLE001
        return -1.0


def parse_sit(log_slice):
    sit = None
    for m in re.finditer(r"(\d+\.?\d*)\s*s/it", log_slice):
        sit = float(m.group(1))            # last s/it wins (the sampler bar)
    if sit is None:
        for m in re.finditer(r"(\d+\.?\d*)\s*it/s", log_slice):
            its = float(m.group(1))
            sit = (1.0 / its) if its else None
    pexec = None
    m = re.search(r"Prompt executed in ([0-9:.]+)", log_slice)
    if m:
        pexec = m.group(1)
    return sit, pexec


# --------------------------------------------------------------------------- #
# prompt load + mutate per leg
# --------------------------------------------------------------------------- #
def _find(prompt, class_type):
    for nid, nd in prompt.items():
        if nd.get("class_type") == class_type:
            return nid
    return None


def _find_all(prompt, class_type):
    return [nid for nid, nd in prompt.items() if nd.get("class_type") == class_type]


def load_base_prompt(schemas):
    with open(WORKFLOW, encoding="utf-8") as f:
        wf = json.load(f)
    base = RB._workflow_to_api_prompt(wf, schemas)
    pruned = [nid for nid, nd in base.items() if nd.get("class_type") not in schemas]
    for nid in pruned:
        del base[nid]
    if pruned:
        log("pruned non-executable nodes: %s" % ", ".join(pruned))
    return base


def mutate_for_leg(base, leg, frames_prefix):
    """Apply the leg's ONE-lever delta to the converted API prompt. Every value the
    manifest asserts is set HERE by name (LiteGraph positional drift already resolved
    by the workflow->API conversion)."""
    p = copy.deepcopy(base)
    i2v_nid = _find(p, "LTXVImgToVideo")
    p[i2v_nid]["inputs"]["width"] = int(leg["w"])
    p[i2v_nid]["inputs"]["height"] = int(leg["h"])
    p[i2v_nid]["inputs"]["length"] = int(leg["length"])
    p[i2v_nid]["inputs"]["strength"] = float(leg["i2v"])
    dec_nid = _find(p, "VAEDecodeTiled")
    p[dec_nid]["inputs"]["tile_size"] = int(leg["tile"])
    p[dec_nid]["inputs"]["overlap"] = int(leg["overlap"])
    p[dec_nid]["inputs"]["temporal_size"] = int(leg["t_size"])
    p[dec_nid]["inputs"]["temporal_overlap"] = int(leg["t_overlap"])
    sig_nid = _find(p, "Sigmas From Text")
    tpl = RESPACED_SIGMAS if leg["sigmas"] == "respaced" else NATIVE_SIGMAS
    p[sig_nid]["inputs"]["text"] = sigmas_text(tpl)
    save_nid = _find(p, "SaveImage")
    p[save_nid]["inputs"]["filename_prefix"] = frames_prefix
    return p


# --------------------------------------------------------------------------- #
# the FAIL-LOUD resolved-API-prompt manifest (the load-bearing rail)
# --------------------------------------------------------------------------- #
def _sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _sha256_file(path):
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_manifest(prompt, leg, reserve_gb, baseline_vram_mb, conform_vf=None):
    """Extract the RESOLVED values from the converted+mutated API prompt and the run
    context; return (manifest dict, checks list). Each check = (name, expected, actual,
    ok). assert_manifest() raises on any ok=False."""
    i2v = prompt[_find(prompt, "LTXVImgToVideo")]["inputs"]
    dec = prompt[_find(prompt, "VAEDecodeTiled")]["inputs"]
    sig = prompt[_find(prompt, "Sigmas From Text")]["inputs"]
    guider = prompt[_find(prompt, "CFGGuider")]["inputs"]
    ksel = prompt[_find(prompt, "KSamplerSelect")]["inputs"]
    noise = prompt[_find(prompt, "RandomNoise")]["inputs"]
    cond = prompt[_find(prompt, "LTXVConditioning")]["inputs"]
    unet = prompt[_find(prompt, "UnetLoaderGGUF")]["inputs"]
    te = prompt[_find(prompt, "LTXAVTextEncoderLoader")]["inputs"]
    vaes = [prompt[n]["inputs"].get("vae_name") for n in _find_all(prompt, "VAELoader")]
    pos_txt = neg_txt = ""
    for nid in _find_all(prompt, "CLIPTextEncode"):
        t = prompt[nid]["inputs"].get("text", "")
        if "low quality" in t.lower():
            neg_txt = t
        else:
            pos_txt = t
    exp_sig = sigmas_text(RESPACED_SIGMAS if leg["sigmas"] == "respaced"
                          else NATIVE_SIGMAS)
    unet_base = os.path.basename(str(unet.get("unet_name", "")).replace("\\", "/"))

    absent = {cls: (_find(prompt, cls) is None) for cls in
              ("LoraLoaderModelOnly", "ModelSamplingLTXV", "LTXVScheduler",
               "CreateVideo", "SaveVideo")}
    manifest = {
        "leg": leg["label"], "stage": leg["stage"],
        "canvas": [i2v.get("width"), i2v.get("height")], "length": i2v.get("length"),
        "i2v_strength": i2v.get("strength"),
        "decode_spatial_tile": [dec.get("tile_size"), dec.get("overlap")],
        "decode_temporal": [dec.get("temporal_size"), dec.get("temporal_overlap")],
        "sigmas_text": sig.get("text"), "sigmas_kind": leg["sigmas"],
        "cfg": guider.get("cfg"), "sampler": ksel.get("sampler_name"),
        "seed": noise.get("noise_seed"), "fps": cond.get("frame_rate"),
        "unet": unet.get("unet_name"), "video_audio_vae": vaes,
        "text_encoder": next((v for v in te.values()
                              if isinstance(v, str) and v.endswith(".safetensors")), None),
        "absent_nodes": absent,
        "terminal": "SaveImage" if _find(prompt, "SaveImage") else None,
        "encoder": "wrapper_bridge.encode_frames_to_silent_mp4 (libx264 crf18 bt709, "
                   "CPU, SILENT)", "audio_absent": True,
        "reserve_vram_gb": reserve_gb, "baseline_vram_mb": baseline_vram_mb,
        "prompt_sha": [_sha256_text(pos_txt), _sha256_text(neg_txt)],
        "still_sha": _sha256_file(os.path.join(SRC_INPUT, STILL_NAME)),
        "audio_sha": _sha256_file(os.path.join(SRC_INPUT, AUDIO_NAME)),
        "conform_vf": conform_vf,
        "conform_vf_sha": _sha256_text(conform_vf) if conform_vf else None,
    }
    sp_tile = spatial_tile_for(leg["w"])
    checks = [
        ("canvas", [leg["w"], leg["h"]], manifest["canvas"]),
        ("length", leg["length"], manifest["length"]),
        ("i2v_strength", leg["i2v"], manifest["i2v_strength"]),
        ("spatial_tile", [sp_tile, 64], manifest["decode_spatial_tile"]),
        ("temporal", [leg["t_size"], leg["t_overlap"]], manifest["decode_temporal"]),
        ("sigmas_text", exp_sig, manifest["sigmas_text"]),
        ("cfg", 1.0, manifest["cfg"]),
        ("sampler", "euler_cfg_pp", manifest["sampler"]),
        ("seed", 0, manifest["seed"]),
        ("fps", FPS, manifest["fps"]),
        ("unet_distilled", True, unet_base.startswith("ltx-2.3-22b-distilled-1.1-")),
        ("LoRA_absent", True, absent["LoraLoaderModelOnly"]),
        ("ModelSamplingLTXV_absent", True, absent["ModelSamplingLTXV"]),
        ("LTXVScheduler_absent", True, absent["LTXVScheduler"]),
        ("CreateVideo_absent", True, absent["CreateVideo"]),
        ("SaveVideo_absent", True, absent["SaveVideo"]),
        ("terminal_SaveImage", "SaveImage", manifest["terminal"]),
    ]
    checks = [(n, e, a, (e == a)) for (n, e, a) in checks]
    return manifest, checks


def assert_manifest(manifest, checks):
    bad = [(n, e, a) for (n, e, a, ok) in checks if not ok]
    log("MANIFEST %s | canvas=%s len=%s temporal=%s spatial=%s i2v=%s sigmas=%s "
        "cfg=%s sampler=%s seed=%s unet=%s reserve=%sGB baseVRAM=%sMB"
        % (manifest["leg"], manifest["canvas"], manifest["length"],
           manifest["decode_temporal"], manifest["decode_spatial_tile"],
           manifest["i2v_strength"], manifest["sigmas_kind"], manifest["cfg"],
           manifest["sampler"], manifest["seed"],
           os.path.basename(str(manifest["unet"])), manifest["reserve_vram_gb"],
           manifest["baseline_vram_mb"]))
    for (n, e, a, ok) in checks:
        log("  [%s] %-24s expect=%r actual=%r" % ("PASS" if ok else "FAIL", n, e, a))
    if bad:
        raise AssertionError(
            "MANIFEST FAIL on leg %s -- measuring the WRONG graph; ABORT: %s"
            % (manifest["leg"], "; ".join("%s exp=%r got=%r" % b for b in bad)))


# --------------------------------------------------------------------------- #
# frames -> silent clip (PRODUCTION encoder) + frame metrics
# --------------------------------------------------------------------------- #
def read_saved_frames(leg_label):
    import numpy as np
    from PIL import Image
    pat = os.path.join(FRAMES_ROOT, leg_label, "frame_*.png")
    paths = sorted(glob.glob(pat))
    if not paths:
        raise FileNotFoundError("no SaveImage frames at %s" % pat)
    arr = np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
                    for p in paths], axis=0)
    return arr  # (N, H, W, 3) uint8


def encode_silent(frames, out_path):
    if os.path.exists(out_path):
        os.remove(out_path)
    return _WB.encode_frames_to_silent_mp4(frames, out_path, FPS)


def luma_seam_metric(frames, t_size, t_overlap, length):
    """Per-frame mean luma -> the jump at the DYNAMIC tiled-decode seam indices.
    Returns p99 of the seam jumps + the max local-median ratio. Whole-clip decode
    (t_size >= length) has NO seam."""
    import numpy as np
    f = frames.astype(np.float64)
    luma = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    mean_luma = luma.mean(axis=(1, 2))
    delta = np.abs(np.diff(mean_luma))           # |Y[i]-Y[i-1]|, index i-1 == frame i
    n = frames.shape[0]
    stride = int(t_size) - int(t_overlap)
    if int(t_size) >= int(length) or stride <= 0:
        return {"no_seam": True, "seam_indices": [], "seam_jumps": [],
                "p99": 0.0, "max_local_ratio": 0.0}
    seams = [k * stride for k in range(1, n) if 0 < k * stride < n]
    if not seams:
        return {"no_seam": True, "seam_indices": [], "seam_jumps": [],
                "p99": 0.0, "max_local_ratio": 0.0}
    jumps, ratios = [], []
    for s in seams:
        j = float(delta[s - 1])                  # delta index s-1 == jump into frame s
        jumps.append(round(j, 4))
        lo, hi = max(0, s - 9), min(len(delta), s + 8)
        local = np.concatenate([delta[lo:s - 1], delta[s:hi]]) if hi > s else delta[lo:s - 1]
        med = float(np.median(local)) if local.size else 0.0
        ratios.append((j / med) if med > 1e-6 else (999.0 if j > 0.05 else 0.0))
    return {"no_seam": False, "seam_indices": seams, "seam_jumps": jumps,
            "p99": round(float(np.percentile(jumps, 99)), 4),
            "max_local_ratio": round(float(max(ratios)), 3)}


def laplacian_variance(frames, sample=12):
    """Mean 4-neighbour Laplacian variance over evenly sampled frames (sharpness)."""
    import numpy as np
    n = frames.shape[0]
    idx = np.linspace(0, n - 1, min(sample, n)).astype(int)
    vs = []
    for i in idx:
        f = frames[i].astype(np.float64)
        g = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
        lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
               - 4 * g[1:-1, 1:-1])
        vs.append(float(lap.var()))
    return round(float(np.mean(vs)), 2)


# --------------------------------------------------------------------------- #
# ffmpeg / ffprobe metrics on the encoded clip
# --------------------------------------------------------------------------- #
def ffprobe_streams(clip):
    cmd = ["ffprobe", "-v", "error", "-show_entries",
           "stream=codec_type,width,height,nb_read_frames,avg_frame_rate",
           "-count_frames", "-of", "json", clip]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
        d = json.loads(out or "{}")
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
    has_audio = any(s.get("codec_type") == "audio" for s in d.get("streams", []))
    return {"width": v.get("width"), "height": v.get("height"),
            "nb_frames": int(v.get("nb_read_frames") or 0),
            "avg_frame_rate": v.get("avg_frame_rate"), "has_audio": has_audio}


def scene_cuts(clip, thresh=0.20):
    # the select filter's `scene` is an expression var, NOT frame metadata, so
    # metadata=print sees nothing -> count the selected frames via showinfo (one
    # `n:<i>` log line per frame that passes gt(scene,thresh)).
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", clip,
           "-vf", "select='gt(scene,%.2f)',showinfo" % thresh,
           "-an", "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return len(re.findall(r"Parsed_showinfo_\d+.*?\bn:\s*\d+", r.stderr or ""))
    except Exception:  # noqa: BLE001
        return -1


def freeze_segments(clip, noise_db=-55, dur=0.12):
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", clip,
           "-vf", "freezedetect=n=%ddB:d=%.2f" % (noise_db, dur),
           "-map", "0:v", "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        durs = [float(x) for x in re.findall(
            r"freeze_duration:\s*([0-9.]+)", r.stderr or "")]
        return [round(x, 3) for x in durs if x >= dur]
    except Exception:  # noqa: BLE001
        return []


# --------------------------------------------------------------------------- #
# composite conform (Stage 0 scaler A/B + Stage 3 sharpness conform). No GPU.
# --------------------------------------------------------------------------- #
def conform_vf_string(scaler, w=COMPOSITE_W, h=COMPOSITE_H, fps=int(FPS)):
    """The production-style composite scale (otr_silent_composite _seg_vf) at the
    1472x832 canvas. scaler='bilinear' = the current default; 'lanczos_unsharp' adds
    flags=lanczos + a mild luma unsharp."""
    if scaler == "lanczos_unsharp":
        scale = ("scale=%d:%d:force_original_aspect_ratio=decrease:flags=lanczos,"
                 "unsharp=5:5:0.8:5:5:0.0," % (w, h))
    else:
        scale = "scale=%d:%d:force_original_aspect_ratio=decrease," % (w, h)
    return (scale + "pad=%d:%d:(ow-iw)/2:(oh-ih)/2:color=black,fps=%d" % (w, h, fps))


_BT709 = ["-vsync", "cfr", "-pix_fmt", "yuv420p", "-color_primaries", "bt709",
          "-color_trc", "bt709", "-colorspace", "bt709", "-c:v", "libx264",
          "-crf", "18", "-preset", "fast"]


def conform_clip(src, out, scaler):
    vf = conform_vf_string(scaler)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-an",
           "-vf", vf] + _BT709 + [out]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if p.returncode != 0 or not os.path.exists(out):
        raise RuntimeError("conform failed (%s): %s" % (scaler, p.stderr[-300:]))
    return vf


def frames_from_clip(clip, sample=12):
    """Decode evenly-spaced frames from a clip into a (k,H,W,3) uint8 array (for the
    conformed-clip sharpness measure)."""
    import numpy as np
    from PIL import Image
    tmp = os.path.join(FRAMES_ROOT, "_conform_tmp")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    # sample every Nth frame via select; -vsync 0 keeps only selected
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", clip, "-vf",
           "select='not(mod(n\\,10))'", "-vsync", "0",
           os.path.join(tmp, "f_%04d.png")]
    subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    paths = sorted(glob.glob(os.path.join(tmp, "f_*.png")))[:sample]
    if not paths:
        return None
    return np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
                     for p in paths], axis=0)


# --------------------------------------------------------------------------- #
# poll with the LOAD-BEARING aborts (s/it ceiling + wall cap); VRAM is
# telemetry-only (the OOM budget is owned by the external per-hardware tier
# JSON, not this harness).
# --------------------------------------------------------------------------- #
def poll_with_abort(pid, server_log, slog_offset, vp, baseline_sit, baseline_wall):
    """Poll /history to completion, aborting (reset_box -> kill) on: parsed s/it
    > max(30, 2.5*baseline); or wall > the per-leg hard cap. Returns (status,
    reason, sit, pexec). Peak VRAM (``vp.peak_mb``) is sampled throughout and
    recorded in the result -- never compared to a ceiling here."""
    import requests
    # the gate is s/it <= 30 AND <= 2.5x baseline -> abort if EITHER is exceeded
    rel_cap = (SIT_REL_CAP * baseline_sit) if baseline_sit else None
    wall_cap = max(360.0, 2.5 * baseline_wall) if baseline_wall else 600.0
    t0 = time.time()
    last_sit = None
    while True:
        wall = time.time() - t0
        with open(server_log, "r", encoding="utf-8", errors="replace") as f:
            f.seek(slog_offset)
            sl = f.read()
        sit, pexec = parse_sit(sl)
        if sit is not None:
            last_sit = sit
            if sit > SIT_ABS_CAP or (rel_cap is not None and sit > rel_cap):
                reset_box()
                cap = SIT_ABS_CAP if sit > SIT_ABS_CAP else rel_cap
                return ("abort", "s/it %.1f > cap %.1f (the 223 spiral guard)"
                        % (sit, cap), sit, None)
        # completion check
        try:
            r = requests.get(BASE_URL + "/history/" + pid, timeout=10)
            if r.status_code == 200 and r.json().get(pid):
                return ("ok", None, last_sit, pexec)
        except Exception:  # noqa: BLE001
            pass
        if wall > wall_cap:
            reset_box()
            return ("abort", "wall %.0fs > cap %.0fs" % (wall, wall_cap), last_sit, None)
        time.sleep(2.0)


# --------------------------------------------------------------------------- #
# one render leg
# --------------------------------------------------------------------------- #
def run_render_leg(leg, baseline_sit, baseline_wall):
    """reset -> boot(reserve) -> manifest(FAIL-LOUD) -> submit -> poll/abort -> read
    frames -> silent encode -> metrics -> gates -> teardown."""
    label = leg["label"]
    server_log = os.path.join(BAKEOFF_DIR, "comfy_server_%s.log" % label)
    frames_prefix = "otr/episodes/_bakeoff_ltxq/_frames/%s/frame" % label
    frames_dir = os.path.join(FRAMES_ROOT, label)
    out_clip = os.path.join(BAKEOFF_DIR, label + ".mp4")
    leg["tile"] = spatial_tile_for(leg["w"])         # spatial tile scales to width
    result = {"label": label, "stage": leg["stage"], "reserve_gb": leg["reserve"],
              "status": "error"}
    # pre-delete stale frames/clip (unique-run discipline)
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir, ignore_errors=True)
    if os.path.exists(out_clip):
        os.remove(out_clip)

    proc = None
    try:
        reset_box()
        proc = boot_server(server_log, reserve_gb=leg["reserve"])
        schemas = wait_ready()
        base = load_base_prompt(schemas)
        prompt = mutate_for_leg(base, leg, frames_prefix)

        with VramPeak() as vp:
            baseline_vram = vp.baseline_mb if vp.baseline_mb is not None else nvml_used_mb()
            manifest, checks = build_manifest(prompt, leg, leg["reserve"], baseline_vram)
            with open(os.path.join(BAKEOFF_DIR, "manifest_%s.json" % label),
                      "w", encoding="utf-8") as f:
                json.dump({"manifest": manifest,
                           "checks": [{"name": n, "expect": e, "actual": a, "ok": ok}
                                      for (n, e, a, ok) in checks]}, f, indent=2)
            assert_manifest(manifest, checks)        # FAIL-LOUD -> raises, no render
            result["manifest"] = manifest

            with open(server_log, "r", encoding="utf-8", errors="replace") as f:
                f.seek(0, os.SEEK_END)
                slog_offset = f.tell()
            t0 = time.time()
            pid = RB._submit_prompt(prompt, BASE_URL)
            status, reason, sit, pexec = poll_with_abort(
                pid, server_log, slog_offset, vp, baseline_sit, baseline_wall)
            wall = time.time() - t0

        result["peak_vram_mb"] = round(vp.peak_mb, 1)
        result["baseline_vram_mb"] = baseline_vram
        result["wall_s"] = round(wall, 1)
        if sit is None:
            with open(server_log, "r", encoding="utf-8", errors="replace") as f:
                f.seek(slog_offset)
                sit, pexec = parse_sit(f.read())
        result["s_per_it"] = sit if sit is not None else round(wall / STEPS, 3)
        result["s_per_it_source"] = "log" if sit is not None else "wall/steps"
        result["prompt_executed_in"] = pexec

        if status == "abort":
            result["status"] = "abort"
            result["error"] = reason
            log("LEG %s ABORT: %s" % (label, reason))
            return result

        # silent encode via the PRODUCTION encoder
        frames = read_saved_frames(label)
        result["frame_count"] = int(frames.shape[0])
        encode_silent(frames, out_clip)
        result["clip"] = out_clip
        result["clip_bytes"] = os.path.getsize(out_clip)

        # metrics
        result["ffprobe"] = ffprobe_streams(out_clip)
        result["scene_cuts"] = scene_cuts(out_clip)
        result["freezes"] = freeze_segments(out_clip)
        result["seam"] = luma_seam_metric(frames, leg["t_size"], leg["t_overlap"],
                                          leg["length"])
        if leg.get("sharpness"):
            result["sharpness_native"] = laplacian_variance(frames)

        # gates (peak_vram_mb is recorded telemetry only -- no ceiling gate)
        peak = result["peak_vram_mb"]
        sitv = result["s_per_it"]
        gates = {
            "sit_abs_ok": (sitv <= SIT_ABS_CAP),
            "sit_rel_ok": (baseline_sit is None or sitv <= SIT_REL_CAP * baseline_sit),
            "scene_cuts_ok": (result["scene_cuts"] == 0),
            "ffprobe_canvas_ok": (result["ffprobe"].get("width") == leg["w"]
                                  and result["ffprobe"].get("height") == leg["h"]),
            "audio_absent_ok": (result["ffprobe"].get("has_audio") is False),
        }
        if not result["seam"].get("no_seam"):
            gates["seam_ok"] = (result["seam"]["p99"] < 2.5
                                and result["seam"]["max_local_ratio"] <= 1.5)
        result["gates"] = gates
        result["status"] = "ok"
        log("LEG %s OK | %ss/it | peakVRAM %.0fMB | scene=%s freezes=%s seam_p99=%s | %s"
            % (label, sitv, peak, result["scene_cuts"], len(result["freezes"]),
               result["seam"].get("p99"), out_clip))
    except AssertionError as e:
        result["status"] = "manifest_fail"
        result["error"] = str(e)
        log("LEG %s MANIFEST_FAIL: %s" % (label, e))
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e)
        log("LEG %s EXCEPTION: %r" % (label, e))
    finally:
        try:
            reset_box()
        except Exception as e:  # noqa: BLE001
            log("teardown reset error: %r" % e)
    return result


# --------------------------------------------------------------------------- #
# Stage 0 -- FREE composite-scaler A/B on the L0 clip (no GPU)
# --------------------------------------------------------------------------- #
def run_stage0(l0_clip):
    log("STAGE 0 (free): composite-scaler A/B on %s" % os.path.basename(l0_clip))
    out = {"stage": 0, "input": l0_clip, "scalers": {}}
    for scaler in ("bilinear", "lanczos_unsharp"):
        clip = os.path.join(BAKEOFF_DIR, "S0_%s.mp4" % scaler)
        if os.path.exists(clip):
            os.remove(clip)
        try:
            vf = conform_clip(l0_clip, clip, scaler)
            fr = frames_from_clip(clip)
            sharp = laplacian_variance(fr) if fr is not None else None
            out["scalers"][scaler] = {
                "clip": clip, "vf": vf, "vf_sha": _sha256_text(vf),
                "sharpness": sharp, "bytes": os.path.getsize(clip)}
            log("  S0 %-16s sharpness=%s -> %s" % (scaler, sharp, os.path.basename(clip)))
        except Exception as e:  # noqa: BLE001
            out["scalers"][scaler] = {"error": repr(e)}
            log("  S0 %s FAIL: %r" % (scaler, e))
    a = out["scalers"].get("bilinear", {}).get("sharpness") or 0
    b = out["scalers"].get("lanczos_unsharp", {}).get("sharpness") or 0
    winner = "lanczos_unsharp" if b >= a else "bilinear"
    out["winner"] = winner
    out["sharpness_gain_pct"] = round(((b - a) / a * 100.0), 1) if a else None
    log("  STAGE 0 winner = %s (gain %s%% over bilinear)"
        % (winner, out["sharpness_gain_pct"]))
    return out


# --------------------------------------------------------------------------- #
# winner picks (objective carry-forward; operator picks the FINAL look from clips)
# --------------------------------------------------------------------------- #
def _passes(r):
    g = r.get("gates", {})
    return (r.get("status") == "ok" and g.get("sit_abs_ok")
            and g.get("sit_rel_ok"))


def pick_decode_winner(l0, stage1):
    """Lowest temporal-seam among passing legs; whole-clip (no_seam) wins outright.
    Ties -> lower s/it. Falls back to L0's 64/8 if nothing passes."""
    cands = [r for r in ([l0] + stage1) if _passes(r)]
    if not cands:
        return l0
    def key(r):
        s = r.get("seam", {})
        return (0 if s.get("no_seam") else 1, s.get("p99", 9e9), r.get("s_per_it", 9e9))
    return sorted(cands, key=key)[0]


def pick_sampler_winner(stage2):
    """Fewest freezes, then lowest seam, then s/it."""
    cands = [r for r in stage2 if _passes(r)]
    if not cands:
        return None
    def key(r):
        return (len(r.get("freezes", [])), r.get("seam", {}).get("p99", 9e9),
                r.get("s_per_it", 9e9))
    return sorted(cands, key=key)[0]


# --------------------------------------------------------------------------- #
# results writers
# --------------------------------------------------------------------------- #
def write_results(results, extra=None):
    payload = {"legs": results, "extra": extra or {}}
    with open(os.path.join(BAKEOFF_DIR, "ltxq_bakeoff_results.json"),
              "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    lines = [
        "# LTX-AV quality bakeoff (distilled_native, PLAN v5)",
        "",
        "Fixed still `%s` + audio `%s`, distilled-1.1 Q3_K_M + DEV VAEs, "
        "euler_cfg_pp, cfg 1.0, seed 0, SILENT (encode_frames_to_silent_mp4)."
        % (STILL_NAME, AUDIO_NAME),
        "",
        "| leg | stage | canvas | temporal | i2v | sigmas | s/it | peakVRAM MB | "
        "scene | freezes | seam p99 | status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m = r.get("manifest", {})
        seam = r.get("seam", {})
        seam_s = "no-seam" if seam.get("no_seam") else seam.get("p99", "-")
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r.get("label", "?"), r.get("stage", "?"),
            "x".join(str(c) for c in m.get("canvas", [])) or "-",
            "-".join(str(c) for c in m.get("decode_temporal", [])) or "-",
            m.get("i2v_strength", "-"), m.get("sigmas_kind", "-"),
            r.get("s_per_it", "-"), r.get("peak_vram_mb", "-"),
            r.get("scene_cuts", "-"), len(r.get("freezes", [])) if r.get("freezes") is not None else "-",
            seam_s, r.get("status", "?") + ("" if r.get("status") == "ok"
                                            else " (%s)" % str(r.get("error", ""))[:40])))
    if extra and extra.get("stage0"):
        s0 = extra["stage0"]
        lines += ["", "## Stage 0 composite scaler A/B (free)",
                  "winner = **%s** (sharpness gain %s%% over bilinear)"
                  % (s0.get("winner"), s0.get("sharpness_gain_pct"))]
    if extra and extra.get("picks"):
        lines += ["", "## Carry-forward picks (objective; operator picks the FINAL look)"]
        for k, v in extra["picks"].items():
            lines.append("- %s -> **%s**" % (k, v))
    with open(os.path.join(BAKEOFF_DIR, "ltxq_bakeoff_results.md"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# dry-validate: boot once, build + ASSERT every manifest, render NOTHING
# --------------------------------------------------------------------------- #
def dry_validate():
    server_log = os.path.join(BAKEOFF_DIR, "comfy_server_dryvalidate.log")
    legs = representative_legs()
    ok = True
    try:
        reset_box()
        boot_server(server_log, reserve_gb=RESERVE_GB)
        schemas = wait_ready()
        base = load_base_prompt(schemas)
        with open(os.path.join(BAKEOFF_DIR, "debug_base_prompt.json"),
                  "w", encoding="utf-8") as f:
            json.dump(base, f, indent=2)
        for leg in legs:
            leg["tile"] = spatial_tile_for(leg["w"])
            prompt = mutate_for_leg(base, leg, "otr/episodes/_bakeoff_ltxq/_frames/%s/frame"
                                    % leg["label"])
            manifest, checks = build_manifest(prompt, leg, leg["reserve"], -1.0)
            with open(os.path.join(BAKEOFF_DIR, "manifest_%s.json" % leg["label"]),
                      "w", encoding="utf-8") as f:
                json.dump({"manifest": manifest,
                           "checks": [{"name": n, "expect": e, "actual": a, "ok": k}
                                      for (n, e, a, k) in checks]}, f, indent=2)
            try:
                assert_manifest(manifest, checks)
            except AssertionError as e:
                ok = False
                log("DRY manifest FAIL: %s" % e)
        log("DRY-VALIDATE %s" % ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    finally:
        try:
            reset_box()
        except Exception as e:  # noqa: BLE001
            log("teardown reset error: %r" % e)


def representative_legs():
    """A static superset that exercises every lever the manifest asserts (for the
    no-GPU preflight): L0 + one per stage incl. a re-spaced + a canvas leg."""
    return [
        make_leg("L0", 1, sharpness=True),
        make_leg("L1e_whole", 1, t_size=4096, t_overlap=8),
        make_leg("L2d_i62_respaced", 2, i2v=0.62, sigmas="respaced"),
        make_leg("L3b_704x384", 3, w=704, h=384, sharpness=True),
    ]


# --------------------------------------------------------------------------- #
# full sweep
# --------------------------------------------------------------------------- #
def full_sweep(only_stage=None):
    results = []
    extra = {"picks": {}}
    # --- L0 baseline (reference for every relative gate) ---
    l0 = run_render_leg(make_leg("L0", 1, sharpness=True), None, None)
    results.append(l0)
    write_results(results, extra)
    if l0.get("status") != "ok":
        log("FATAL: L0 baseline did not render (%s) -- no baseline; STOP."
            % l0.get("error"))
        return results, extra
    baseline_sit = l0["s_per_it"]
    baseline_wall = l0["wall_s"]
    log("BASELINE s/it=%s wall=%ss -> abort caps: s/it>%.1f or >%.1f"
        % (baseline_sit, baseline_wall, SIT_ABS_CAP, SIT_REL_CAP * baseline_sit))

    # --- Stage 0 (free) on the L0 clip ---
    extra["stage0"] = run_stage0(l0["clip"])
    write_results(results, extra)

    # --- Stage 1 decode sweep (4096/8 LAST) ---
    stage1 = []
    if only_stage in (None, 1):
        s1_legs = [
            make_leg("L1a_t64o16", 1, t_size=64, t_overlap=16),
            make_leg("L1b_t64o32", 1, t_size=64, t_overlap=32),
            make_leg("L1c_t128o16", 1, t_size=128, t_overlap=16),
            make_leg("L1d_t128o32", 1, t_size=128, t_overlap=32),
            make_leg("L1e_whole_4096o8", 1, t_size=4096, t_overlap=8),  # LAST
        ]
        for leg in s1_legs:
            r = run_render_leg(leg, baseline_sit, baseline_wall)
            stage1.append(r)
            results.append(r)
            write_results(results, extra)
    dec_win = pick_decode_winner(l0, stage1)
    extra["picks"]["decode"] = "%s (temporal %s)" % (
        dec_win["label"], dec_win.get("manifest", {}).get("decode_temporal"))
    log("DECODE winner -> %s" % extra["picks"]["decode"])

    # --- Stage 2 sampler 2x2 on the best decode ---
    stage2 = []
    if only_stage in (None, 2):
        # carry the winning decode tiling onto the 2x2
        dwt = dec_win.get("manifest", {}).get("decode_temporal", [64, 8])
        for i2v in (0.75, 0.62):
            for sg in ("native", "respaced"):
                lbl = "L2_i%02d_%s" % (int(i2v * 100), sg)
                leg = make_leg(lbl, 2, t_size=dwt[0], t_overlap=dwt[1], i2v=i2v, sigmas=sg)
                r = run_render_leg(leg, baseline_sit, baseline_wall)
                stage2.append(r)
                results.append(r)
                write_results(results, extra)
    smp_win = pick_sampler_winner(stage2) or dec_win
    extra["picks"]["sampler"] = "%s (i2v=%s sigmas=%s)" % (
        smp_win["label"], smp_win.get("manifest", {}).get("i2v_strength"),
        smp_win.get("manifest", {}).get("sigmas_kind"))
    log("SAMPLER winner -> %s" % extra["picks"]["sampler"])

    # --- Stage 3 canvas on the stage1+2 winner ---
    if only_stage in (None, 3):
        wm = smp_win.get("manifest", {})
        wdt = wm.get("decode_temporal", [64, 8])
        wi2v = wm.get("i2v_strength", 0.75)
        wsg = wm.get("sigmas_kind", "native")
        for (cw, ch) in ((640, 384), (704, 384)):
            leg = make_leg("L3_%dx%d" % (cw, ch), 3, w=cw, h=ch,
                           t_size=wdt[0], t_overlap=wdt[1], i2v=wi2v, sigmas=wsg,
                           sharpness=True)
            r = run_render_leg(leg, baseline_sit, baseline_wall)
            results.append(r)
            write_results(results, extra)
            # reserve step-up retry on a VRAM/spill/s-it abort (5 then 6 GB)
            if r.get("status") == "abort":
                for bump in (5.0, 6.0):
                    log("STAGE3 %s abort (%s) -> retry at reserve %sGB"
                        % (leg["label"], r.get("error"), bump))
                    leg2 = dict(leg)
                    leg2["reserve"] = bump
                    leg2["label"] = "%s_r%d" % (leg["label"], int(bump))
                    r = run_render_leg(leg2, baseline_sit, baseline_wall)
                    results.append(r)
                    write_results(results, extra)
                    if r.get("status") == "ok":
                        break
    write_results(results, extra)
    return results, extra


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def stage_assets():
    os.makedirs(SERVER_INPUT, exist_ok=True)
    for name in FIXED_ASSETS:
        src = os.path.join(SRC_INPUT, name)
        dst = os.path.join(SERVER_INPUT, name)
        if not os.path.exists(src):
            raise FileNotFoundError("fixed asset missing: %s" % src)
        shutil.copy2(src, dst)
        log("staged asset -> %s" % dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-validate", action="store_true",
                    help="boot once, build + ASSERT every manifest, render NOTHING")
    ap.add_argument("--only-stage", type=int, default=None,
                    help="run only stage N (1|2|3); default = full sweep")
    args = ap.parse_args()

    os.makedirs(BAKEOFF_DIR, exist_ok=True)
    os.makedirs(FRAMES_ROOT, exist_ok=True)
    stage_assets()
    log("bakeoff dir: %s" % BAKEOFF_DIR)
    log("workflow:    %s" % WORKFLOW)

    if args.dry_validate:
        return dry_validate()
    results, extra = full_sweep(only_stage=args.only_stage)
    ok = sum(1 for r in results if r.get("status") == "ok")
    log("DONE. %d legs, %d ok. Results -> %s"
        % (len(results), ok, os.path.join(BAKEOFF_DIR, "ltxq_bakeoff_results.md")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
