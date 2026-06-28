"""
run_humo_bakeoff.py
===================

Headless HuMo STANDALONE VRAM/quality bakeoff (roundtables/2026-06-27-humo-optim/
final.md). DIAGNOSTIC ONLY -- renders clips + metrics for the operator eyeball and
changes NOTHING in production. Cloned from scripts/run_ltx_av_q_bakeoff.py.

It answers ONE gate: can the 5/21 HuMo-14B (humo_14B_169: fp8 + lightx2v distill
LoRA + ModelSamplingSD3 shift 8 @ 6 steps / cfg 1.0) fit <= 13.5 GB with real
headroom when the umt5 CLIP + whisper encoders are evicted (the sibling
OTR_BakeoffReclaim node, two-stage) BEFORE the heavy 14B sampler -- vs the
shipping humo_1.7B_169.

LEGS (fixed still c02_466a19906ccb.png + audio c02_b002_line.wav + seed 0):
  i_14B_single            humo_14B_169 single-graph 6-step distill (= the 5/21 path).
  ii_14B_twostage         humo_14B_169 + OTR_BakeoffReclaim on the latent edge.
  iii_1p7B_control        humo_1.7B_169 (current ship; 16:9 to match the candidate).
  iv_sentinel_14B_twostage  fire a REAL ltx_audio_in render first (LTX-AV uses Gemma
                          + LTX VAEs, NOT Whisper), THEN leg (ii) in the SAME resident
                          session -> the production-true cross-engine peak.

Boot-per-leg WITHOUT FLOOR (FLOOR clears OTR_ENABLE_HUMO -> HuMoEngine.assert_usable
fails closed); the launcher's default branch sets OTR_ENABLE_HUMO=1. reset_box does a
SELECTIVE CIM kill (by command line, excl. THIS pid) -- the Comfy server + port 8000
owners + the cmd.exe soak shell + any stale run_humo_bakeoff.py -- NEVER a blanket
python kill (that would sever the MCP pythons). Reset before EVERY leg.

VRAM: external NVML render-window peak. DECISION GATE = peak <= 13500 MB; the 14500
MB box ceiling is the OOM-abort guard, REPORTED separately (not the gate).
OUTPUT: SaveImage PNG batch -> production wrapper_bridge.encode_frames_to_silent_mp4
(SILENT); ffprobe-verified (no audio stream) -> otr/episodes/_bakeoff_humo/<leg>.mp4.

  python run_humo_bakeoff.py --dry-validate   # boot once no-FLOOR, assert manifests, NO render
  python run_humo_bakeoff.py                  # full 4-leg bakeoff (GPU)

STOP after the sweep + report; the operator eyeballs the clips before any production
wiring. UTF-8, no BOM. ASCII-only. SFW.
"""

import argparse
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
SCRIPTS = os.path.join(REPO, "scripts")
TESTS = os.path.join(REPO, "tests")
for _p in (REPO, SCRIPTS, TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _run_baseline as RB                      # noqa: E402 (workflow->API + submit)
import build_humo_bakeoff_workflow as BUILD     # noqa: E402 (the per-leg /prompt builder)

# Production silent encoder by FILE PATH (V-12 cold-import clean: no nodes pkg /
# ComfyUI / torch) so a winning leg wires byte-for-byte to production.
_WB_PATH = os.path.join(REPO, "nodes", "_otr_video_engines", "wrapper_bridge.py")
_wb_spec = importlib.util.spec_from_file_location("otr_wb_humo_bakeoff", _WB_PATH)
_WB = importlib.util.module_from_spec(_wb_spec)
_wb_spec.loader.exec_module(_WB)

COMFY_ROOT = r"C:\Users\jeffr\Documents\ComfyUI"
VENV_PY = r"C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
SRC_INPUT = r"C:\Users\jeffr\Documents\ComfyUI\input"
SERVER_INPUT = r"C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\input"
FIXED_ASSETS = BUILD.FIXED_ASSETS
STILL_NAME, AUDIO_NAME = BUILD.STILL_NAME, BUILD.AUDIO_NAME
OUTPUT_BASE = os.environ.get("COMFYUI_OUTPUT", os.path.join(COMFY_ROOT, "output"))
BAKEOFF_DIR = os.path.join(OUTPUT_BASE, "otr", "episodes", "_bakeoff_humo")
FRAMES_ROOT = os.path.join(BAKEOFF_DIR, "_frames")
BASE_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8000")
PORT = 8000
SOAK_LAUNCH_CMD = os.path.join(REPO, "scripts", "_otr_soak_server_launch.cmd")
# The sentinel reuses the proven isolated LTX-AV distilled_native workflow (real
# ltx_audio_in path: Gemma TE + LTX VAEs, NOT Whisper).
LTX_SENTINEL_WF = os.path.join(REPO, "scripts",
                               "otr_ltx_av_q_bakeoff_distilled_native.json")

FPS = 25.0
VRAM_GATE_MB = 13500          # the DECISION gate (fit with REAL headroom -> promote)
BOX_CEILING_MB = 14500        # single-resident reference ceiling, REPORTED (not a gate/abort)
# OOM-abort = the TRUE physical guard. nvidia-smi "used" for a --cuda-malloc process
# includes the allocator's reserved/cached pool, so a render can read ~15-15.9 GB and
# still complete within the 16 GB card; aborting at 14.5 only kills renders that would
# finish (and yields no clip). We abort just under physical VRAM so renders COMPLETE
# and produce clips for the eyeball, then REPORT the peak vs 13500/14500.
OOM_ABORT_MB = int(os.environ.get("OTR_BAKEOFF_OOM_ABORT_MB", "16000"))
SIT_ABS_CAP = 90.0            # s/it spiral guard (HuMo 14B is slower than LTX)
POLL_S = 2.0


def log(msg):
    print("[humo] %s" % msg, flush=True)


# --------------------------------------------------------------------------- #
# box reset (CLAUDE.md S4: SELECTIVE CIM kill, NEVER a blanket python kill)
# --------------------------------------------------------------------------- #
def _ps_json(cmd, timeout=60):
    """Run a PowerShell snippet and parse its ConvertTo-Json output -> a list (a
    single object is wrapped). Returns [] on any error / empty output."""
    try:
        r = subprocess.run(["powershell.exe", "-NoProfile", "-Command", cmd],
                           capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "").strip()
        if not out:
            return []
        data = json.loads(out)
        return data if isinstance(data, list) else [data]
    except Exception as e:  # noqa: BLE001
        log("reset: ps-json error: %r" % e)
        return []


def reset_box():
    """SELECTIVE reset (CLAUDE.md S4): kill the ComfyUI server + port-8000 owner +
    the cmd.exe soak shell + any STALE run_humo_bakeoff.py -- NEVER a blanket python
    kill (that severs the MCP pythons). The current-PID exclusion is done in PYTHON
    (a reliable int compare), then each target is killed by explicit PID -- the
    PowerShell ``-ne $cur`` scope guard proved unreliable and self-killed the run."""
    cur = os.getpid()
    procs = _ps_json("Get-CimInstance Win32_Process -Filter \"Name='python.exe' or "
                     "Name='cmd.exe'\" | Select-Object ProcessId,ParentProcessId,Name,"
                     "CommandLine | ConvertTo-Json -Compress")
    # Build pid->ppid and EXCLUDE cur + its whole ancestor chain. The .venv
    # python.exe is a uv shim that re-execs the real interpreter, so cur (the real
    # interp) AND its parent shim share this exact command line -- killing the
    # parent shim takes the run down with it (the self-kill we hit).
    ppid = {}
    for p in procs:
        try:
            ppid[int(p.get("ProcessId") or 0)] = int(p.get("ParentProcessId") or 0)
        except Exception:  # noqa: BLE001
            continue
    protect = {cur}
    walk, seen = cur, set()
    while walk in ppid and walk not in seen:
        seen.add(walk)
        walk = ppid[walk]
        if walk:
            protect.add(walk)
    targets = set()
    for p in procs:
        try:
            pid = int(p.get("ProcessId") or 0)
        except Exception:  # noqa: BLE001
            continue
        if pid <= 0 or pid in protect:
            continue
        cl = p.get("CommandLine") or ""
        name = (p.get("Name") or "").lower()
        if name == "python.exe" and (
                re.search(r"ComfyUI[\\/]+main\.py", cl)
                or ("main.py" in cl and re.search(r"--port\s+8000", cl))
                or "run_humo_bakeoff.py" in cl):
            targets.add(pid)
        elif name == "cmd.exe" and "_otr_soak_server_launch" in cl:
            targets.add(pid)
    for o in _ps_json("Get-NetTCPConnection -LocalPort 8000 -State Listen | "
                      "Select-Object OwningProcess | ConvertTo-Json -Compress"):
        try:
            pid = int(o.get("OwningProcess") or 0)
        except Exception:  # noqa: BLE001
            pid = 0
        if pid and pid not in protect:
            targets.add(pid)
    for pid in sorted(targets):
        subprocess.run(["powershell.exe", "-NoProfile", "-Command",
                        "Stop-Process -Id %d -Force -ErrorAction SilentlyContinue" % pid],
                       capture_output=True, text=True, timeout=20)
        log("reset: killed pid %d" % pid)
    time.sleep(2)
    rep = _ps_json("@{listen=((Get-NetTCPConnection -LocalPort 8000 -State Listen|"
                   "Measure-Object).Count);used=((nvidia-smi --query-gpu=memory.used "
                   "--format=csv,noheader,nounits) -join ',')} | ConvertTo-Json -Compress")
    if rep:
        log("reset: port8000 listeners=%s | nvidia-smi used MB=%s"
            % (rep[0].get("listen"), rep[0].get("used")))


def boot_server(server_log_path):
    """Boot headless ComfyUI via the proven launcher WITHOUT a profile arg, so the
    launcher's default branch sets OTR_ENABLE_HUMO=1 (FLOOR/LTX/WAN would clear it).
    No --reserve-vram (production HuMo runs fully resident)."""
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("OTR_TEST_MODE", None)
    env.pop("OTR_HEADLESS_RESERVE_VRAM_GB", None)   # HuMo: no startup reserve
    alloc = os.environ.get("OTR_BAKEOFF_ALLOC_CONF")
    if alloc:
        env["PYTORCH_CUDA_ALLOC_CONF"] = alloc      # allocator A/B (Step A)
    else:
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    log("booting ComfyUI :%d (no-FLOOR, OTR_ENABLE_HUMO=1, ALLOC_CONF=%s) -> %s"
        % (PORT, alloc or "<default>", server_log_path))
    return subprocess.Popen([SOAK_LAUNCH_CMD, server_log_path],
                            cwd=os.path.dirname(SOAK_LAUNCH_CMD), env=env)


def wait_ready(timeout_s=240):
    import requests
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            r = requests.get(BASE_URL + "/object_info", timeout=10)
            if r.status_code == 200:
                log("server ready in %.0fs" % (time.time() - start))
                return r.json()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    raise TimeoutError("ComfyUI did not become ready within %ds" % timeout_s)


# --------------------------------------------------------------------------- #
# NVML peak-VRAM sampler
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
        time.sleep(0.4)
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=3)


def parse_sit(log_slice):
    sit = None
    for m in re.finditer(r"(\d+\.?\d*)\s*s/it", log_slice):
        sit = float(m.group(1))
    if sit is None:
        for m in re.finditer(r"(\d+\.?\d*)\s*it/s", log_slice):
            its = float(m.group(1))
            sit = (1.0 / its) if its else None
    pexec = None
    m = re.search(r"Prompt executed in ([0-9:.]+)", log_slice)
    if m:
        pexec = m.group(1)
    return sit, pexec


def grep_log(server_log, offset, pattern):
    try:
        with open(server_log, "r", encoding="utf-8", errors="replace") as f:
            f.seek(offset)
            return re.findall(pattern, f.read())
    except Exception:  # noqa: BLE001
        return []


def log_tail_offset(server_log):
    try:
        with open(server_log, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, os.SEEK_END)
            return f.tell()
    except Exception:  # noqa: BLE001
        return 0


# --------------------------------------------------------------------------- #
# checkpoint-exists assert (the server's loader enums == what it can actually load)
# --------------------------------------------------------------------------- #
def _enum_options(schemas, cls, param):
    """The available options for a loader's combo param, across BOTH /object_info
    schema shapes on this box: the old core format ``[[file1, file2, ...], {meta}]``
    (list at [0]) AND the newer V3 COMBO format ``["COMBO", {"options": [...]}]``
    (options under [1]["options"]) -- AudioEncoderLoader uses the latter."""
    try:
        node = schemas[cls]["input"]
        for sec in ("required", "optional"):
            if param in node.get(sec, {}):
                ti = node[sec][param]
                if isinstance(ti, list) and ti:
                    if isinstance(ti[0], list):
                        return ti[0]                       # old: list at [0]
                    if len(ti) > 1 and isinstance(ti[1], dict):
                        opts = ti[1].get("options")         # V3 COMBO: [1]["options"]
                        if isinstance(opts, list):
                            return opts
    except Exception:  # noqa: BLE001
        pass
    return []


def assert_checkpoints(schemas, meta):
    """Fail LOUD if any of this leg's checkpoint files is NOT among the server's
    loader enums (== not installed). Never silently skip a leg into a false DONE."""
    want = [(meta.get("loader_class", "UNETLoader"),
             meta.get("loader_param", "unet_name"), meta["unet"]),
            ("CLIPLoader", "clip_name", meta["clip"]),
            ("VAELoader", "vae_name", meta["vae"]),
            ("AudioEncoderLoader", "audio_encoder_name", meta["whisper"])]
    if meta.get("lora"):
        want.append(("LoraLoaderModelOnly", "lora_name", meta["lora"]))
    missing = []
    for cls, param, fname in want:
        opts = _enum_options(schemas, cls, param)
        if fname not in opts:
            missing.append("%s.%s=%s" % (cls, param, fname))
    if missing:
        raise AssertionError(
            "CHECKPOINT MISSING on leg %s (not in server loader enums): %s"
            % (meta["label"], "; ".join(missing)))
    log("checkpoints OK for %s (unet/clip/vae/whisper%s present)"
        % (meta["label"], "/lora" if meta.get("lora") else ""))


# --------------------------------------------------------------------------- #
# the FAIL-LOUD resolved-prompt manifest (extract from the built /prompt, assert)
# --------------------------------------------------------------------------- #
def _find(prompt, cls):
    for nid, nd in prompt.items():
        if nd.get("class_type") == cls:
            return nid
    return None


def _sha256_file(path):
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def build_manifest(prompt, meta):
    """Extract the RESOLVED recipe from the built /prompt and assert it matches the
    intended (meta). The #1 risk is measuring the WRONG graph -- any mismatch aborts."""
    lc = meta.get("loader_class", "UNETLoader")
    lp = meta.get("loader_param", "unet_name")
    unet = prompt[_find(prompt, lc)]["inputs"]
    msss = prompt[_find(prompt, "ModelSamplingSD3")]["inputs"]
    ks = prompt[_find(prompt, "KSampler")]["inputs"]
    humo = prompt[_find(prompt, "WanHuMoImageToVideo")]["inputs"]
    lora_nid = _find(prompt, "LoraLoaderModelOnly")
    reclaim_nid = _find(prompt, BUILD.RECLAIM_CLASS)
    length = humo.get("length")
    manifest = {
        "leg": meta["label"], "engine_id": meta["engine_id"],
        "two_stage": meta["two_stage"], "sentinel": meta["sentinel"],
        "unet": unet.get(lp), "loader_class": lc,
        "lora": (prompt[lora_nid]["inputs"].get("lora_name") if lora_nid else None),
        "shift": msss.get("shift"), "steps": ks.get("steps"), "cfg": ks.get("cfg"),
        "seed": ks.get("seed"),
        "dims": [humo.get("width"), humo.get("height")], "length": length,
        "terminal": "SaveImage" if _find(prompt, "SaveImage") else None,
        "reclaim_present": reclaim_nid is not None,
        "still_sha": _sha256_file(os.path.join(SRC_INPUT, STILL_NAME)),
        "audio_sha": _sha256_file(os.path.join(SRC_INPUT, AUDIO_NAME)),
        "encoder": "wrapper_bridge.encode_frames_to_silent_mp4 (libx264 crf18 "
                   "bt709, SILENT)", "audio_absent": True,
        "vram_decision_gate_mb": VRAM_GATE_MB, "box_ceiling_mb": BOX_CEILING_MB,
        "alloc_conf": os.environ.get("OTR_BAKEOFF_ALLOC_CONF") or "<default>",
    }
    checks = [
        ("unet", meta["unet"], manifest["unet"]),
        ("lora", meta["lora"], manifest["lora"]),
        ("shift_8", 8.0, float(manifest["shift"]) if manifest["shift"] is not None else None),
        ("steps", meta["steps"], manifest["steps"]),
        ("cfg", meta["cfg"], manifest["cfg"]),
        ("seed", meta["seed"], manifest["seed"]),
        ("dims", [meta["width"], meta["height"]], manifest["dims"]),
        ("length", meta["length"], manifest["length"]),
        ("frame_4n1", 0, ((int(length) - 1) % 4) if length is not None else -1),
        ("terminal_SaveImage", "SaveImage", manifest["terminal"]),
        ("reclaim_present", meta["two_stage"], manifest["reclaim_present"]),
    ]
    checks = [(n, e, a, (e == a)) for (n, e, a) in checks]
    return manifest, checks


def assert_manifest(manifest, checks):
    log("MANIFEST %s | engine=%s unet=%s lora=%s shift=%s steps=%s cfg=%s seed=%s "
        "dims=%s len=%s two_stage=%s"
        % (manifest["leg"], manifest["engine_id"],
           os.path.basename(str(manifest["unet"])), bool(manifest["lora"]),
           manifest["shift"], manifest["steps"], manifest["cfg"], manifest["seed"],
           manifest["dims"], manifest["length"], manifest["two_stage"]))
    for (n, e, a, ok) in checks:
        log("  [%s] %-20s expect=%r actual=%r" % ("PASS" if ok else "FAIL", n, e, a))
    bad = [(n, e, a) for (n, e, a, ok) in checks if not ok]
    if bad:
        raise AssertionError(
            "MANIFEST FAIL on %s -- measuring the WRONG graph; ABORT: %s"
            % (manifest["leg"], "; ".join("%s exp=%r got=%r" % b for b in bad)))


# --------------------------------------------------------------------------- #
# frames -> silent clip + metrics
# --------------------------------------------------------------------------- #
def read_saved_frames(label):
    import numpy as np
    from PIL import Image
    pat = os.path.join(FRAMES_ROOT, label, "frame*.png")
    paths = sorted(glob.glob(pat))
    if not paths:
        raise FileNotFoundError("no SaveImage frames at %s" % pat)
    arr = np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
                    for p in paths], axis=0)
    return arr


def blue_cast_delta(frames, still_path):
    """Mean R/G/B of rendered frames vs the source still (pure PIL+numpy, no face
    libs) -- the de-blue concern. Reports B-R for both (the engine de-blue metric)."""
    import numpy as np
    from PIL import Image
    fm = frames.reshape(-1, 3).mean(axis=0)
    out = {"frame_mean_rgb": [round(float(x), 2) for x in fm],
           "frame_b_minus_r": round(float(fm[2] - fm[0]), 2)}
    try:
        s = np.asarray(Image.open(still_path).convert("RGB"), dtype=np.float64)
        sm = s.reshape(-1, 3).mean(axis=0)
        out["still_mean_rgb"] = [round(float(x), 2) for x in sm]
        out["still_b_minus_r"] = round(float(sm[2] - sm[0]), 2)
        out["delta_b_minus_r"] = round(out["frame_b_minus_r"] - out["still_b_minus_r"], 2)
    except Exception as e:  # noqa: BLE001
        out["still_error"] = repr(e)
    return out


def face_metrics_softgated(frames):
    """Optional face/mouth metrics -- SOFT-GATED: skipped (never failing) if the
    libs are absent (they are NOT in requirements.txt)."""
    out = {"status": "skipped", "reason": "no face libs"}
    try:
        import cv2  # noqa: F401
    except Exception:  # noqa: BLE001
        return out
    try:
        import cv2
        import numpy as np
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        det = cv2.CascadeClassifier(cascade_path)
        if det.empty():
            return {"status": "skipped", "reason": "no haarcascade"}
        n = frames.shape[0]
        idx = np.linspace(0, n - 1, min(12, n)).astype(int)
        hits = 0
        for i in idx:
            gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            faces = det.detectMultiScale(gray, 1.1, 4)
            if len(faces):
                hits += 1
        return {"status": "ok", "engine": "opencv-haar",
                "frames_sampled": int(len(idx)), "frames_with_face": int(hits),
                "face_detect_rate": round(hits / max(1, len(idx)), 3)}
    except Exception as e:  # noqa: BLE001
        return {"status": "skipped", "reason": repr(e)}


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


# --------------------------------------------------------------------------- #
# poll a submitted prompt to completion, aborting on the box ceiling / s-it spiral
# --------------------------------------------------------------------------- #
def poll_with_abort(pid, server_log, slog_offset, vp, wall_cap=1800.0):
    import requests
    t0 = time.time()
    last_sit = None
    while True:
        wall = time.time() - t0
        if vp.peak_mb and vp.peak_mb > OOM_ABORT_MB:
            reset_box()
            return ("abort", "VRAM %.0fMB > OOM guard %d (near physical)"
                    % (vp.peak_mb, OOM_ABORT_MB), last_sit, None)
        with open(server_log, "r", encoding="utf-8", errors="replace") as f:
            f.seek(slog_offset)
            sl = f.read()
        sit, pexec = parse_sit(sl)
        if sit is not None:
            last_sit = sit
            if sit > SIT_ABS_CAP:
                reset_box()
                return ("abort", "s/it %.1f > cap %.1f (sysmem spiral guard)"
                        % (sit, SIT_ABS_CAP), sit, None)
        if "ERROR" in sl and "Prompt execution failed" in sl:
            return ("error", "server reported Prompt execution failed", last_sit, None)
        try:
            r = requests.get(BASE_URL + "/history/" + pid, timeout=10)
            if r.status_code == 200 and r.json().get(pid):
                return ("ok", None, last_sit, pexec)
        except Exception:  # noqa: BLE001
            pass
        if wall > wall_cap:
            reset_box()
            return ("abort", "wall %.0fs > cap %.0fs" % (wall, wall_cap), last_sit, None)
        time.sleep(POLL_S)


# --------------------------------------------------------------------------- #
# the LTX-AV sentinel render (real ltx_audio_in: Gemma TE + LTX VAEs, NOT Whisper)
# --------------------------------------------------------------------------- #
def run_ltx_sentinel(schemas, server_log):
    if not os.path.exists(LTX_SENTINEL_WF):
        raise FileNotFoundError("LTX sentinel workflow missing: %s" % LTX_SENTINEL_WF)
    with open(LTX_SENTINEL_WF, encoding="utf-8") as f:
        wf = json.load(f)
    prompt = RB._workflow_to_api_prompt(wf, schemas)
    pruned = [nid for nid, nd in prompt.items() if nd.get("class_type") not in schemas]
    for nid in pruned:
        del prompt[nid]
    # assert this really is the LTX-AV path (Gemma TE + LTX, NOT whisper)
    classes = {nd.get("class_type") for nd in prompt.values()}
    if "LTXAVTextEncoderLoader" not in classes:
        raise AssertionError("sentinel workflow is NOT ltx_audio_in (no "
                             "LTXAVTextEncoderLoader); classes=%s" % sorted(classes))
    if "AudioEncoderLoader" in classes:
        raise AssertionError("sentinel LTX workflow unexpectedly uses Whisper "
                             "(AudioEncoderLoader) -- not the LTX-AV residency we want")
    log("sentinel: submitting real LTX-AV render (%d nodes, Gemma TE + LTX VAEs)"
        % len(prompt))
    off = log_tail_offset(server_log)
    pid = RB._submit_prompt(prompt, BASE_URL)
    with VramPeak() as vp:
        status, reason, sit, pexec = poll_with_abort(pid, server_log, off, vp,
                                                     wall_cap=1200.0)
    log("sentinel LTX render: status=%s peakVRAM=%.0fMB %s"
        % (status, vp.peak_mb, "" if status == "ok" else "(%s)" % reason))
    if status != "ok":
        raise RuntimeError("sentinel LTX render did not complete: %s" % reason)
    return {"ltx_peak_mb": round(vp.peak_mb, 1), "ltx_sit": sit, "ltx_pexec": pexec}


# --------------------------------------------------------------------------- #
# one HuMo render leg
# --------------------------------------------------------------------------- #
def run_leg(leg):
    label = leg["label"]
    server_log = os.path.join(BAKEOFF_DIR, "comfy_server_%s.log" % label)
    frames_dir = os.path.join(FRAMES_ROOT, label)
    out_clip = os.path.join(BAKEOFF_DIR, label + ".mp4")
    result = {"label": label, "status": "error"}
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir, ignore_errors=True)
    if os.path.exists(out_clip):
        os.remove(out_clip)

    proc = None
    try:
        reset_box()
        proc = boot_server(server_log)
        schemas = wait_ready()
        prompt, meta = BUILD.build_leg_prompt(leg, schemas=schemas)
        result["meta"] = meta
        assert_checkpoints(schemas, meta)
        manifest, checks = build_manifest(prompt, meta)
        with open(os.path.join(BAKEOFF_DIR, "manifest_%s.json" % label),
                  "w", encoding="utf-8") as f:
            json.dump({"manifest": manifest,
                       "checks": [{"name": n, "expect": e, "actual": a, "ok": ok}
                                  for (n, e, a, ok) in checks]}, f, indent=2)
        assert_manifest(manifest, checks)
        result["manifest"] = manifest

        # Sentinel: fire the real LTX-AV render FIRST (same resident session).
        if meta["sentinel"]:
            result["sentinel"] = run_ltx_sentinel(schemas, server_log)

        slog_offset = log_tail_offset(server_log)
        t0 = time.time()
        with VramPeak() as vp:
            baseline = vp.baseline_mb
            pid = RB._submit_prompt(prompt, BASE_URL)
            status, reason, sit, pexec = poll_with_abort(pid, server_log, slog_offset, vp)
        wall = time.time() - t0
        result["peak_vram_mb"] = round(vp.peak_mb, 1)
        result["baseline_vram_mb"] = baseline
        result["wall_s"] = round(wall, 1)
        result["s_per_it"] = sit
        result["prompt_executed_in"] = pexec

        # two-stage / sentinel: assert the encoder-evict actually RAN
        if meta["two_stage"]:
            markers = grep_log(server_log, slog_offset,
                               r"\[OTR_BakeoffReclaim\] ENCODER-EVICT marker=(\w+)")
            result["reclaim_markers"] = markers
            if not markers:
                result["status"] = "abort"
                result["error"] = "two-stage leg ran but OTR_BakeoffReclaim never logged"
                log("LEG %s ABORT: reclaim node did not run" % label)
                return result

        # honest in-process VRAM meter (Step A): the probe logs TRUE peak since the
        # pre-sampler reset. nvidia-smi "used" counts the cached reserve pool; the gap
        # between max_allocated (true demand) and reserved/NVML is the fit question.
        pm = grep_log(server_log, slog_offset,
                      r"\[OTR_BakeoffVramProbe\] marker=\w+ max_allocated_mb=([\d.]+) "
                      r"max_reserved_mb=([\d.]+)")
        if pm:
            a, rv = pm[-1]
            result["true_vram_mb"] = {"max_allocated": float(a), "max_reserved": float(rv)}

        if status != "ok":
            result["status"] = status
            result["error"] = reason
            log("LEG %s %s: %s" % (label, status.upper(), reason))
            return result

        frames = read_saved_frames(label)
        result["frame_count"] = int(frames.shape[0])
        if os.path.exists(out_clip):
            os.remove(out_clip)
        _WB.encode_frames_to_silent_mp4(frames, out_clip, FPS)
        result["clip"] = out_clip
        result["clip_bytes"] = os.path.getsize(out_clip)
        result["asset_exists"] = os.path.exists(out_clip)

        result["ffprobe"] = ffprobe_streams(out_clip)
        result["blue_cast"] = blue_cast_delta(frames, os.path.join(SRC_INPUT, STILL_NAME))
        result["face_metrics"] = face_metrics_softgated(frames)

        peak = result["peak_vram_mb"]
        ta = (result.get("true_vram_mb") or {}).get("max_allocated")
        exp_len = (result.get("manifest") or {}).get("length")
        result["gates"] = {
            "nvml_peak_mb": peak,
            "true_alloc_mb": ta,
            "true_alloc_le_13500": (ta is not None and ta <= VRAM_GATE_MB),
            "nvml_le_14500": peak <= BOX_CEILING_MB,
            "promotable": (ta is not None and ta <= VRAM_GATE_MB
                           and peak <= BOX_CEILING_MB),
            "audio_absent": result["ffprobe"].get("has_audio") is False,
            "frame_count_ok": (exp_len is not None
                               and result["frame_count"] == exp_len),
        }
        result["status"] = "ok"
        log("LEG %s OK | true-alloc %sMB (<=%d:%s) | NVML %.0fMB (<=%d:%s) | "
            "promotable=%s | %ss/it | %d/%s frames | B-R %s vs still %s | %s"
            % (label, ("%.0f" % ta) if ta is not None else "n/a", VRAM_GATE_MB,
               result["gates"]["true_alloc_le_13500"], peak, BOX_CEILING_MB,
               result["gates"]["nvml_le_14500"], result["gates"]["promotable"],
               result["s_per_it"], result["frame_count"], exp_len,
               result["blue_cast"].get("frame_b_minus_r"),
               result["blue_cast"].get("still_b_minus_r"), out_clip))
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
# results writers
# --------------------------------------------------------------------------- #
def write_results(results):
    with open(os.path.join(BAKEOFF_DIR, "humo_bakeoff_results.json"),
              "w", encoding="utf-8") as f:
        json.dump({"legs": results}, f, indent=2)
    lines = [
        "# HuMo standalone bakeoff (final.md, 2026-06-27)",
        "",
        "Fixed still `%s` + audio `%s`, seed 0, SILENT "
        "(encode_frames_to_silent_mp4). DECISION gate = peak VRAM <= %d MB; box "
        "ceiling %d MB reported separately. DIAGNOSTIC -- production untouched."
        % (STILL_NAME, AUDIO_NAME, VRAM_GATE_MB, BOX_CEILING_MB),
        "",
        "| leg | engine | two-stage | alloc-conf | true-alloc MB | NVML MB | "
        "promotable | s/it | frames | B-R frame | B-R still | audio? | status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        m = r.get("manifest", {})
        g = r.get("gates", {})
        tv = r.get("true_vram_mb", {})
        bc = r.get("blue_cast", {})
        fp = r.get("ffprobe", {})
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r.get("label", "?"), m.get("engine_id", "-"), m.get("two_stage", "-"),
                m.get("alloc_conf", "-"),
                ("%.0f" % tv["max_allocated"]) if tv.get("max_allocated") is not None
                else "-",
                r.get("peak_vram_mb", "-"), g.get("promotable", "-"),
                r.get("s_per_it", "-"), r.get("frame_count", "-"),
                bc.get("frame_b_minus_r", "-"), bc.get("still_b_minus_r", "-"),
                ("yes" if fp.get("has_audio") else "no") if fp else "-",
                r.get("status", "?") + ("" if r.get("status") == "ok"
                                        else " (%s)" % str(r.get("error", ""))[:48])))
    lines += ["", "Clips: `otr/episodes/_bakeoff_humo/<leg>.mp4` -- operator eyeball "
              "decides before any promotion (two-stage split / VramPeakProbe / "
              "profile flip are DEFERRED)."]
    with open(os.path.join(BAKEOFF_DIR, "humo_bakeoff_results.md"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# dry-validate: boot once no-FLOOR, assert every leg manifest + registration, NO render
# --------------------------------------------------------------------------- #
def dry_validate():
    server_log = os.path.join(BAKEOFF_DIR, "comfy_server_dryvalidate.log")
    ok = True
    proc = None
    try:
        reset_box()
        proc = boot_server(server_log)
        schemas = wait_ready()
        if BUILD.RECLAIM_CLASS not in schemas:
            ok = False
            log("DRY FAIL: %s not registered (is custom_nodes/otr_bakeoff_helper "
                "installed + server restarted?)" % BUILD.RECLAIM_CLASS)
        if "WanHuMoImageToVideo" not in schemas:
            ok = False
            log("DRY FAIL: WanHuMoImageToVideo not registered")
        for _cls in (BUILD.RESET_CLASS, BUILD.PROBE_CLASS):
            if _cls not in schemas:
                ok = False
                log("DRY FAIL: %s not registered (restart ComfyUI to load the "
                    "sibling pack's new nodes)" % _cls)
        for leg in BUILD.LEGS:
            prompt, meta = BUILD.build_leg_prompt(leg, schemas=schemas)
            checks = BUILD.validate_prompt(prompt, meta, schemas=schemas)
            try:
                assert_checkpoints(schemas, meta)
            except AssertionError as e:
                ok = False
                log("DRY checkpoint FAIL: %s" % e)
            manifest, mchecks = build_manifest(prompt, meta)
            with open(os.path.join(BAKEOFF_DIR, "manifest_%s.json" % leg["label"]),
                      "w", encoding="utf-8") as f:
                json.dump({"manifest": manifest,
                           "checks": [{"name": n, "expect": e, "actual": a, "ok": k}
                                      for (n, e, a, k) in mchecks]}, f, indent=2)
            bad = [(n, d) for (n, k, d) in checks if not k]
            try:
                assert_manifest(manifest, mchecks)
            except AssertionError as e:
                ok = False
                log("DRY manifest FAIL: %s" % e)
            if bad:
                ok = False
                for (n, d) in bad:
                    log("DRY structural FAIL %s: %s %s" % (leg["label"], n, d))
            log("DRY leg %s built + asserted" % leg["label"])
        log("DRY-VALIDATE %s" % ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    finally:
        try:
            reset_box()
        except Exception as e:  # noqa: BLE001
            log("teardown reset error: %r" % e)


# --------------------------------------------------------------------------- #
def stage_assets():
    os.makedirs(SERVER_INPUT, exist_ok=True)
    for name in FIXED_ASSETS:
        src = os.path.join(SRC_INPUT, name)
        if not os.path.exists(src):
            raise FileNotFoundError("fixed asset missing: %s" % src)
        shutil.copy2(src, os.path.join(SERVER_INPUT, name))
        log("staged asset -> %s" % os.path.join(SERVER_INPUT, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-validate", action="store_true",
                    help="boot once no-FLOOR, assert every leg manifest, render NOTHING")
    ap.add_argument("--only", default=None,
                    help="run only the leg whose label contains this substring")
    ap.add_argument("--alloc-conf", default=None,
                    help="set PYTORCH_CUDA_ALLOC_CONF for the boot (Step A A/B), e.g. "
                         "'expandable_segments:True'")
    ap.add_argument("--reset-selftest", action="store_true",
                    help="call reset_box() once and confirm THIS process survives")
    args = ap.parse_args()

    if args.reset_selftest:
        log("reset-selftest: pid=%d calling reset_box()..." % os.getpid())
        reset_box()
        log("reset-selftest: SURVIVED (reset_box did not kill self)")
        return 0

    if args.alloc_conf is not None:
        os.environ["OTR_BAKEOFF_ALLOC_CONF"] = args.alloc_conf
    os.makedirs(BAKEOFF_DIR, exist_ok=True)
    os.makedirs(FRAMES_ROOT, exist_ok=True)
    stage_assets()
    log("bakeoff dir: %s" % BAKEOFF_DIR)

    if args.dry_validate:
        return dry_validate()

    legs = BUILD.LEGS
    if args.only:
        legs = [l for l in legs if args.only in l["label"]]
    results = []
    for leg in legs:
        r = run_leg(leg)
        results.append(r)
        write_results(results)
        log("--- leg %s done: %s ---" % (leg["label"], r.get("status")))
    ok = sum(1 for r in results if r.get("status") == "ok")
    log("DONE. %d legs, %d ok. Results -> %s"
        % (len(results), ok, os.path.join(BAKEOFF_DIR, "humo_bakeoff_results.md")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
