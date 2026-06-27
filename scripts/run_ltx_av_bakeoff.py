"""
run_ltx_av_bakeoff.py
=====================

Headless LTX-AV GGUF quant bakeoff. Renders ONE fixed still+audio through each
quant under test, logging per-quant s/it + peak VRAM, and writing each clip to
``output/otr/episodes/_bakeoff/<quant>.mp4`` for a side-by-side speed + quality
compare.

Quants (CLAUDE.md / 2026-06-26 plan):
  dev          ltx-2.3-22b-dev-Q3_K_M.gguf              SHARP LoRA ON  (baseline)
  distilled    distilled-1.1/...-Q2_K.gguf              SHARP LoRA OFF
  distilled    distilled-1.1/...-Q3_K_M.gguf            SHARP LoRA OFF
  distilled    distilled-1.1/...-Q4_K_M.gguf            SHARP LoRA OFF

The distilled-1.1 GGUF bakes the distillation in, so the SHARP distilled LoRA is
DROPPED for those runs (the LoraLoaderModelOnly node is removed and CFGGuider is
repointed straight at UnetLoaderGGUF). Everything else (encoder, VAEs, still,
audio, sigmas, sampler, cfg, canvas, seed) is held constant.

This script OWNS the whole run: it resets the box (selective CIM kill -- never a
blanket python kill, per CLAUDE.md S4), boots a FRESH headless ComfyUI on :8000,
runs every leg, then tears down the server it spawned. Launch it DETACHED and
poll its progress log (the ~60s MCP ceiling).

UTF-8, no BOM. ASCII-only.
"""

import argparse
import copy
import glob
import json
import os
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

COMFY_ROOT = r"C:\Users\jeffr\Documents\ComfyUI"
VENV_PY = r"C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
MODELS_UNET = r"C:\ComfyUI-Models\unet"
# the headless server reads LoadImage/LoadAudio from the INSTALL-root input dir
# (folder_paths.get_input_directory()), NOT Documents\ComfyUI\input where the
# episode assets live -- stage the fixed still+audio across before the legs.
SRC_INPUT = r"C:\Users\jeffr\Documents\ComfyUI\input"
SERVER_INPUT = r"C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\input"
FIXED_ASSETS = ["c02_466a19906ccb.png", "c02_b002_line.wav"]
OUTPUT_BASE = os.environ.get("COMFYUI_OUTPUT", os.path.join(COMFY_ROOT, "output"))
BAKEOFF_DIR = os.path.join(OUTPUT_BASE, "otr", "episodes", "_bakeoff")
WORKFLOW = os.path.join(REPO, "workflows", "ltx_av_bakeoff_gguf.json")
BASE_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8000")
PORT = 8000
STEPS = 8  # the 8-step distilled schedule (for the exec/steps s/it fallback)

# (label, unet filename relative to the unet/ models folder, sharp_lora_on)
QUANTS = [
    ("dev_Q3_K_M",          "ltx-2.3-22b-dev-Q3_K_M.gguf",                       True),
    ("distilled11_Q2_K",    r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q2_K.gguf", False),
    ("distilled11_Q3_K_M",  r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf", False),
    ("distilled11_Q4_K_M",  r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf", False),
]


def log(msg):
    print("[bakeoff] %s" % msg, flush=True)


def stage_assets():
    """Copy the fixed still + audio into the server's input dir (LoadImage /
    LoadAudio resolve filenames there). Fail loud if a source is missing."""
    import shutil
    os.makedirs(SERVER_INPUT, exist_ok=True)
    for name in FIXED_ASSETS:
        src = os.path.join(SRC_INPUT, name)
        dst = os.path.join(SERVER_INPUT, name)
        if not os.path.exists(src):
            raise FileNotFoundError("fixed asset missing: %s" % src)
        shutil.copy2(src, dst)
        log("staged asset -> %s" % dst)


# --------------------------------------------------------------------------
# box reset (CLAUDE.md S4: selective CIM kill, never a blanket python kill)
# --------------------------------------------------------------------------
def reset_box():
    ps = r"""
$ErrorActionPreference='SilentlyContinue'
# kill ONLY a ComfyUI server (cmdline names main.py) -- never the MCP pythons
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match 'ComfyUI\\main\.py' -or
                 ($_.CommandLine -match 'main\.py' -and $_.CommandLine -match '--port\s+8000') } |
  ForEach-Object { Write-Output ("kill server pid " + $_.ProcessId); Stop-Process -Id $_.ProcessId -Force }
# free port 8000
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


# --------------------------------------------------------------------------
# server lifecycle
# --------------------------------------------------------------------------
SOAK_LAUNCH_CMD = os.path.join(REPO, "scripts", "_otr_soak_server_launch.cmd")


def boot_server(server_log_path):
    """Boot the headless server via the PROVEN launcher (CLAUDE.md S5): it
    pins the real Desktop-v2 main.py, --cuda-malloc, --user-directory, the
    MANDATORY --extra-model-paths-config yaml (without which the LTX wrapper
    nodes fall to the floor), UTF-8 stdio, and the unified output dir. We pass
    the FLOOR arg so the heavy HuMo engine stays OFF (the raw LTX-AV node
    classes still register; engine enable-flags don't gate node registration).
    The .cmd redirects its own stdout to <server_log_path> (%1)."""
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)  # MUST run on the GPU
    env.pop("OTR_TEST_MODE", None)
    log("booting ComfyUI :%d via %s -> %s"
        % (PORT, os.path.basename(SOAK_LAUNCH_CMD), server_log_path))
    proc = subprocess.Popen([SOAK_LAUNCH_CMD, server_log_path, "FLOOR"],
                            cwd=os.path.dirname(SOAK_LAUNCH_CMD), env=env)
    return proc, None


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


# --------------------------------------------------------------------------
# NVML peak-VRAM sampler
# --------------------------------------------------------------------------
class VramPeak:
    def __init__(self, interval=0.25):
        self.interval = interval
        self.peak_mb = 0
        self._stop = threading.Event()
        self._t = None

    def _run(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            while not self._stop.is_set():
                used = pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 * 1024)
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
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=3)


# --------------------------------------------------------------------------
# s/it parsing from the server-log slice for this leg
# --------------------------------------------------------------------------
def parse_sit(log_slice):
    import re
    sit = None
    for m in re.finditer(r"(\d+\.?\d*)\s*s/it", log_slice):
        sit = float(m.group(1))            # last s/it wins (the sampler bar)
    if sit is None:
        for m in re.finditer(r"(\d+\.?\d*)\s*it/s", log_slice):
            its = float(m.group(1))
            sit = (1.0 / its) if its else None
    prompt_exec = None
    m = re.search(r"Prompt executed in ([0-9:.]+)", log_slice)
    if m:
        prompt_exec = m.group(1)
    return sit, prompt_exec


# --------------------------------------------------------------------------
# prompt mutation per quant
# --------------------------------------------------------------------------
def _find(prompt, class_type):
    for nid, nd in prompt.items():
        if nd.get("class_type") == class_type:
            return nid
    return None


def mutate(base_prompt, unet_enum, sharp_lora_on, save_prefix):
    p = copy.deepcopy(base_prompt)
    unet_nid = _find(p, "UnetLoaderGGUF")
    p[unet_nid]["inputs"]["unet_name"] = unet_enum
    save_nid = _find(p, "SaveVideo")
    p[save_nid]["inputs"]["filename_prefix"] = save_prefix
    lora_nid = _find(p, "LoraLoaderModelOnly")
    guider_nid = _find(p, "CFGGuider")
    if not sharp_lora_on and lora_nid is not None:
        # drop the SHARP LoRA: repoint guider.model straight at the unet
        p[guider_nid]["inputs"]["model"] = [str(unet_nid), 0]
        del p[lora_nid]
    return p


def resolve_unet_enum(schemas, fname):
    """Match a quant filename to the live UnetLoaderGGUF dropdown string
    (handles the subfolder path separator ComfyUI actually lists)."""
    info = schemas.get("UnetLoaderGGUF", {})
    opts = (info.get("input", {}).get("required", {}).get("unet_name")
            or [None])[0]
    if not isinstance(opts, list):
        return None
    base = os.path.basename(fname)
    # exact, then basename, then separator-insensitive
    for o in opts:
        if o == fname:
            return o
    for o in opts:
        if os.path.basename(o.replace("\\", "/")) == base:
            return o
    norm = fname.replace("\\", "/").lower()
    for o in opts:
        if o.replace("\\", "/").lower() == norm:
            return o
    return None


# --------------------------------------------------------------------------
# one leg
# --------------------------------------------------------------------------
def run_leg(label, base_prompt, unet_enum, sharp_lora_on, server_log_path):
    import requests
    prefix = "otr/episodes/_bakeoff/%s" % label
    prompt = mutate(base_prompt, unet_enum, sharp_lora_on, prefix)
    log("LEG %s | unet=%s | sharp_lora=%s" % (label, unet_enum, sharp_lora_on))

    with open(server_log_path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        slog_offset = f.tell()

    t0 = time.time()
    result = {"label": label, "unet": unet_enum, "sharp_lora": sharp_lora_on,
              "status": "error"}
    try:
        with VramPeak() as vp:
            pid = RB._submit_prompt(prompt, BASE_URL)
            RB._poll_for_completion(pid, BASE_URL, timeout_s=1800.0)
        wall = time.time() - t0
        result["peak_vram_mb"] = round(vp.peak_mb, 1)
        result["wall_s"] = round(wall, 1)

        with open(server_log_path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(slog_offset)
            sl = f.read()
        sit, pexec = parse_sit(sl)
        result["s_per_it"] = sit if sit is not None else round(wall / STEPS, 3)
        result["s_per_it_source"] = "log" if sit is not None else "wall/steps"
        result["prompt_executed_in"] = pexec

        # locate + normalize the produced clip to <label>.mp4
        produced = sorted(glob.glob(os.path.join(BAKEOFF_DIR, label + "*.mp4")))
        if produced:
            final = os.path.join(BAKEOFF_DIR, label + ".mp4")
            src = produced[-1]
            if os.path.abspath(src) != os.path.abspath(final):
                if os.path.exists(final):
                    os.remove(final)
                os.replace(src, final)
            result["clip"] = final
            result["clip_bytes"] = os.path.getsize(final)
            result["status"] = "ok"
            log("LEG %s OK | %.3fs/it | peak %.0f MB | %s"
                % (label, result["s_per_it"], result["peak_vram_mb"], final))
        else:
            result["error"] = "no mp4 produced under %s" % BAKEOFF_DIR
            log("LEG %s FAIL: %s" % (label, result["error"]))
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e)
        log("LEG %s EXCEPTION: %r" % (label, e))
    return result


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-boot", action="store_true",
                    help="assume a server is already up on :8000 (skip reset+boot)")
    ap.add_argument("--keep-server", action="store_true",
                    help="do not tear down the server at the end")
    ap.add_argument("--dry-validate", action="store_true",
                    help="boot, convert, resolve quant enums, write the debug "
                         "prompt and exit -- render NOTHING (cheap pre-flight)")
    args = ap.parse_args()

    os.makedirs(BAKEOFF_DIR, exist_ok=True)
    stage_assets()
    server_log = os.path.join(BAKEOFF_DIR, "comfy_server.log")
    progress_log = os.path.join(BAKEOFF_DIR, "bakeoff_progress.log")
    # progress log is this script's own stdout when launched detached
    log("bakeoff dir: %s" % BAKEOFF_DIR)
    log("workflow:    %s" % WORKFLOW)

    # which quants actually exist on disk
    legs = []
    for label, fname, sharp in QUANTS:
        full = os.path.join(MODELS_UNET, fname)
        if os.path.exists(full):
            legs.append((label, fname, sharp))
        else:
            log("SKIP %s: missing on disk -> %s" % (label, full))
    if not legs:
        log("FATAL: no quant files found under %s" % MODELS_UNET)
        return 2

    proc = logf = None
    try:
        if not args.no_boot:
            reset_box()
            proc, logf = boot_server(server_log)
        schemas = wait_ready()

        with open(WORKFLOW, encoding="utf-8") as f:
            wf = json.load(f)
        base_prompt = RB._workflow_to_api_prompt(wf, schemas)
        # drop non-executable / frontend-only nodes (e.g. Note) -- they are not
        # in /object_info and ComfyUI rejects the whole prompt if they appear.
        pruned = [nid for nid, nd in base_prompt.items()
                  if nd.get("class_type") not in schemas]
        for nid in pruned:
            del base_prompt[nid]
        if pruned:
            log("pruned non-executable nodes: %s" % ", ".join(pruned))
        log("converted workflow -> %d-node API prompt" % len(base_prompt))

        # dump the converted base prompt for debugging
        with open(os.path.join(BAKEOFF_DIR, "debug_base_prompt.json"),
                  "w", encoding="utf-8") as f:
            json.dump(base_prompt, f, indent=2)

        if args.dry_validate:
            log("DRY-VALIDATE: resolving quant enums (no render)")
            ok = True
            for label, fname, sharp in legs:
                ue = resolve_unet_enum(schemas, fname)
                log("  %-22s sharp=%-5s -> %s" % (label, sharp, ue))
                if ue is None:
                    ok = False
            for cls in ("UnetLoaderGGUF", "LoraLoaderModelOnly", "CFGGuider",
                        "SaveVideo", "Sigmas From Text", "LTXAVTextEncoderLoader"):
                present = cls in schemas
                log("  node-class %-24s present=%s" % (cls, present))
                ok = ok and present
            log("DRY-VALIDATE %s" % ("PASS" if ok else "FAIL"))
            return 0 if ok else 1

        results = []
        for label, fname, sharp in legs:
            unet_enum = resolve_unet_enum(schemas, fname)
            if unet_enum is None:
                log("SKIP %s: not in live UnetLoaderGGUF dropdown" % label)
                results.append({"label": label, "status": "error",
                                "error": "unet not in dropdown enum"})
                continue
            results.append(run_leg(label, base_prompt, unet_enum, sharp, server_log))
            write_results(results)  # incremental
        write_results(results)
        log("DONE. %d legs." % len(results))
        return 0
    finally:
        if proc is not None and not args.keep_server:
            # the .cmd spawns main.py as a grandchild -- terminate() only kills
            # the cmd shell, so tear the server down via the selective CIM reset.
            log("tearing down server (selective reset)")
            try:
                reset_box()
            except Exception as e:  # noqa: BLE001
                log("teardown reset error: %r" % e)


def write_results(results):
    rj = os.path.join(BAKEOFF_DIR, "bakeoff_results.json")
    with open(rj, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    # markdown table
    lines = [
        "# LTX-AV GGUF quant bakeoff",
        "",
        "Fixed still `c02_466a19906ccb.png` + audio `c02_b002_line.wav`, "
        "832x480x105, 8-step distilled schedule, euler_cfg_pp, cfg 1.0, seed 0.",
        "Distilled-1.1 runs DROP the SHARP LoRA; dev keeps it.",
        "",
        "| quant | sharp LoRA | s/it | peak VRAM (MB) | wall (s) | clip | status |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            r.get("label", "?"),
            "on" if r.get("sharp_lora") else "off",
            r.get("s_per_it", "-"),
            r.get("peak_vram_mb", "-"),
            r.get("wall_s", "-"),
            os.path.basename(r["clip"]) if r.get("clip") else "-",
            r.get("status", "?") + ("" if r.get("status") == "ok"
                                    else " (%s)" % r.get("error", "")),
        ))
    with open(os.path.join(BAKEOFF_DIR, "bakeoff_results.md"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
