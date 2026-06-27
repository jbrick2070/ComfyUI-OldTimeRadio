"""
run_wan_ti2v_bakeoff.py
=======================

ISOLATED Wan 2.2 TI2V-5B GGUF low-VRAM bakeoff (image-to-video ONLY). Renders ONE
fixed still through each quant under test, under each VRAM-clamp tier, logging
per-leg s/it + peak VRAM + peak SYSTEM RAM, and writing each clip to
``output/otr/episodes/_bakeoff_wan/<clamp>__<quant>.mp4`` for a side-by-side
speed + quality + survivability compare on a low-VRAM (6-8 GB) target.

Why a SEPARATE runner from run_ltx_av_bakeoff.py (do NOT just clone it):
  The LTX runner boots the server ONCE and loops every leg in that single
  session. That is WRONG for this bakeoff for two independent reasons the
  2026-06-27 convergence roundtable nailed:
    1. ``--reserve-vram`` (the VRAM clamp) is a startup-only global -- it cannot
       be mutated live over HTTP, so each clamp tier needs its own boot.
    2. ComfyUI deliberately does NOT force-unload resident models between prompts
       (wrapper_bridge._soft_free, the BUG-265 anti-fragmentation discipline), so
       the previous quant's UNET/TE stay resident and contaminate the next leg's
       peak-VRAM measurement. A fresh boot is the only clean per-quant peak.
  So a "leg" = (quant x clamp tier), and EVERY leg does its own
  reset_box() + boot_server() + one /prompt + teardown.

Scope (CLAUDE.md): this harness touches ONLY this script + the isolated workflow
workflows/otr_wan_ti2v_bakeoff_gguf.json. It submits a RAW core-node API graph
over HTTP, so it does NOT import or alter eng_wan_ti2v.py / eng_humo.py /
eng_ltx_av.py -- the HuMo + LTX + Wan engines stay sacred. The graph mirrors
eng_wan_ti2v._build_graph exactly; swapping a quant changes node 1 unet_name ONLY
(NO node substitution -- if a quant ever needs more, that trips the FAILURE
CRITERION: document and STOP before wiring).

The VRAM clamp rides the launcher's new optional OTR_HEADLESS_RESERVE_VRAM_GB env
(default unset = no clamp): the runner sets it per clamp tier, the launcher turns
it into --reserve-vram <GB>.

This script OWNS the whole run: per leg it resets the box (selective CIM kill --
never a blanket python kill, CLAUDE.md S4), boots a FRESH headless ComfyUI on
:8000, renders, then tears down the server it spawned. Launch it DETACHED and
poll its progress log (the ~60s MCP ceiling).

  python run_wan_ti2v_bakeoff.py --dry-validate     # cheap no-GPU preflight per quant
  python run_wan_ti2v_bakeoff.py                    # full sweep (GPU; serialize behind any other render)

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
# the headless server reads LoadImage from the INSTALL-root input dir
# (folder_paths.get_input_directory()), NOT Documents\ComfyUI\input -- stage the
# fixed still across before the legs.
SRC_INPUT = r"C:\Users\jeffr\Documents\ComfyUI\input"
SERVER_INPUT = r"C:\Users\jeffr\ComfyUI-Installs\ComfyUI\ComfyUI\input"
FIXED_ASSETS = ["c02_466a19906ccb.png"]  # i2v: still only, no audio
OUTPUT_BASE = os.environ.get("COMFYUI_OUTPUT", os.path.join(COMFY_ROOT, "output"))
BAKEOFF_DIR = os.path.join(OUTPUT_BASE, "otr", "episodes", "_bakeoff_wan")
# API-format harness graph lives next to this runner (NOT under workflows/): that
# dir's guardrail/zod tests require canonical LiteGraph shape, and this is an
# intentionally API-format isolated bakeoff input.
WORKFLOW = os.path.join(REPO, "scripts", "otr_wan_ti2v_bakeoff_gguf.json")
BASE_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8000")
PORT = 8000
STEPS = 30  # KSampler steps (for the wall/steps s/it fallback)

# (label, unet filename relative to the unet/ models folder). The 5B is a true
# one-variable swap -- no LoRA, no sharp flag (unlike LTX). QuantStack naming.
# Q5_K_M = the current engine baseline (launcher line 76); Q3_K_M + Q2_K are the
# low-VRAM battleground. Missing-on-disk legs are SKIPPED (Step 3 download gates).
QUANTS = [
    ("baseline_Q5_K_M", "Wan2.2-TI2V-5B-Q5_K_M.gguf"),
    ("Q3_K_M",          "Wan2.2-TI2V-5B-Q3_K_M.gguf"),
    ("Q2_K",            "Wan2.2-TI2V-5B-Q2_K.gguf"),
]

# (label, simulated usable VRAM GB or None for the full card). reserve-vram is
# computed per tier as max(0, total_vram_gb - target_gb) at runtime.
CLAMP_TIERS = [
    ("vram_full", None),
    ("vram_8gb",  8),
    ("vram_6gb",  6),
]

# the core node classes the graph needs -- the dry-validate currency gate
REQUIRED_CLASSES = (
    "UnetLoaderGGUF", "CLIPLoaderGGUF", "VAELoader", "ModelSamplingSD3",
    "CLIPTextEncode", "LoadImage", "Wan22ImageToVideoLatent", "KSampler",
    "VAEDecodeTiled", "CreateVideo", "SaveVideo",
)


def log(msg):
    print("[wan-bakeoff] %s" % msg, flush=True)


def total_vram_gb():
    """Total board VRAM (GB) via NVML; fallback 16 (the 5080 laptop)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        gb = pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024 ** 3)
        pynvml.nvmlShutdown()
        return gb
    except Exception:  # noqa: BLE001
        return 16.0


def reserve_for_target(target_gb, total_gb):
    """GB to --reserve-vram so usable VRAM ~= target_gb on a total_gb card."""
    if target_gb is None:
        return None
    return round(max(0.0, total_gb - float(target_gb)), 1)


def stage_assets():
    """Copy the fixed still into the server's input dir (LoadImage resolves
    filenames there). Fail loud if a source is missing."""
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
# server lifecycle (boot PER LEG -- the clamp is startup-only)
# --------------------------------------------------------------------------
SOAK_LAUNCH_CMD = os.path.join(REPO, "scripts", "_otr_soak_server_launch.cmd")


def boot_server(server_log_path, reserve_gb=None):
    """Boot the headless server via the PROVEN launcher (CLAUDE.md S5) with the
    FLOOR profile: heavy OTR engines OFF, but the core Comfy + ComfyUI-GGUF node
    classes (UnetLoaderGGUF/CLIPLoaderGGUF/Wan22ImageToVideoLatent/...) still
    register -- engine enable-flags don't gate node registration. ``reserve_gb``
    rides the launcher's optional OTR_HEADLESS_RESERVE_VRAM_GB -> --reserve-vram
    to simulate a low-VRAM card. The .cmd redirects its own stdout to %1."""
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)  # MUST run on the GPU
    env.pop("OTR_TEST_MODE", None)
    if reserve_gb:
        env["OTR_HEADLESS_RESERVE_VRAM_GB"] = str(reserve_gb)
        log("VRAM clamp: --reserve-vram %s" % reserve_gb)
    else:
        env.pop("OTR_HEADLESS_RESERVE_VRAM_GB", None)
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
# peak VRAM + peak SYSTEM RAM sampler (one thread, both metrics)
# --------------------------------------------------------------------------
class PeakSampler:
    def __init__(self, interval=0.25):
        self.interval = interval
        self.peak_vram_mb = 0
        self.peak_sysram_mb = 0
        self._stop = threading.Event()
        self._t = None

    def _run(self):
        nvml = h = psutil = None
        try:
            import pynvml as nvml
            nvml.nvmlInit()
            h = nvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:  # noqa: BLE001
            log("NVML unavailable: %s" % e)
            self.peak_vram_mb = -1
        try:
            import psutil
        except Exception as e:  # noqa: BLE001
            log("psutil unavailable: %s" % e)
            self.peak_sysram_mb = -1
        while not self._stop.is_set():
            if h is not None:
                try:
                    used = nvml.nvmlDeviceGetMemoryInfo(h).used / (1024 * 1024)
                    if used > self.peak_vram_mb:
                        self.peak_vram_mb = used
                except Exception:  # noqa: BLE001
                    pass
            if psutil is not None:
                try:
                    used = psutil.virtual_memory().used / (1024 * 1024)
                    if used > self.peak_sysram_mb:
                        self.peak_sysram_mb = used
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(self.interval)
        if h is not None:
            try:
                nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass

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
    # a clear sysmem-spill signature in the comfy log (advisory; the peak-sysram
    # metric is the quantitative measure)
    spilled = bool(re.search(r"(?i)(sysmem|system memory|fall(?:ing|s)? back|"
                             r"offload|swap)", log_slice)) or None
    return sit, prompt_exec, spilled


# --------------------------------------------------------------------------
# prompt load + mutation per quant
# --------------------------------------------------------------------------
def _find(prompt, class_type):
    for nid, nd in prompt.items():
        if nd.get("class_type") == class_type:
            return nid
    return None


def load_base_prompt(schemas):
    """Load the isolated workflow. It is authored in API-prompt format (named
    inputs) to avoid LiteGraph positional-widget drift; if a LiteGraph (UI) export
    is ever dropped in instead (top-level ``nodes`` list), convert it via the
    proven RB path. Non-executable nodes (e.g. Note) are pruned either way."""
    with open(WORKFLOW, encoding="utf-8") as f:
        wf = json.load(f)
    if isinstance(wf, dict) and isinstance(wf.get("nodes"), list):
        base = RB._workflow_to_api_prompt(wf, schemas)
    else:
        base = {k: v for k, v in wf.items()
                if not k.startswith("_") and isinstance(v, dict)
                and "class_type" in v}
    pruned = [nid for nid, nd in base.items()
              if nd.get("class_type") not in schemas]
    for nid in pruned:
        del base[nid]
    if pruned:
        log("pruned non-executable nodes: %s" % ", ".join(pruned))
    return base


def mutate(base_prompt, unet_enum, save_prefix):
    p = copy.deepcopy(base_prompt)
    unet_nid = _find(p, "UnetLoaderGGUF")
    p[unet_nid]["inputs"]["unet_name"] = unet_enum
    save_nid = _find(p, "SaveVideo")
    p[save_nid]["inputs"]["filename_prefix"] = save_prefix
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
# one leg = (quant x clamp tier): own reset + boot + render + teardown
# --------------------------------------------------------------------------
def run_leg(quant_label, fname, clamp_label, reserve_gb, total_gb):
    leg_label = "%s__%s" % (clamp_label, quant_label)
    server_log = os.path.join(BAKEOFF_DIR, "comfy_server_%s.log" % leg_label)
    result = {"leg": leg_label, "quant": quant_label, "unet_file": fname,
              "clamp": clamp_label, "reserve_vram_gb": reserve_gb,
              "total_vram_gb": round(total_gb, 1), "status": "error"}
    proc = None
    try:
        reset_box()
        proc, _ = boot_server(server_log, reserve_gb=reserve_gb)
        schemas = wait_ready()
        unet_enum = resolve_unet_enum(schemas, fname)
        if unet_enum is None:
            result["error"] = "unet not in live UnetLoaderGGUF dropdown"
            log("LEG %s SKIP: %s" % (leg_label, result["error"]))
            result["status"] = "skip"
            return result
        base = load_base_prompt(schemas)
        prefix = "otr/episodes/_bakeoff_wan/%s" % leg_label
        prompt = mutate(base, unet_enum, prefix)
        log("LEG %s | unet=%s | reserve=%s" % (leg_label, unet_enum, reserve_gb))

        with open(server_log, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, os.SEEK_END)
            slog_offset = f.tell()

        t0 = time.time()
        import requests  # noqa: F401  (RB submit uses it; ensure present)
        with PeakSampler() as ps:
            pid = RB._submit_prompt(prompt, BASE_URL)
            RB._poll_for_completion(pid, BASE_URL, timeout_s=2400.0)
        wall = time.time() - t0
        result["peak_vram_mb"] = round(ps.peak_vram_mb, 1)
        result["peak_sysram_mb"] = round(ps.peak_sysram_mb, 1)
        result["wall_s"] = round(wall, 1)

        with open(server_log, "r", encoding="utf-8", errors="replace") as f:
            f.seek(slog_offset)
            sl = f.read()
        sit, pexec, spilled = parse_sit(sl)
        result["s_per_it"] = sit if sit is not None else round(wall / STEPS, 3)
        result["s_per_it_source"] = "log" if sit is not None else "wall/steps"
        result["prompt_executed_in"] = pexec
        result["sysmem_spill_log_hint"] = spilled

        produced = sorted(glob.glob(os.path.join(BAKEOFF_DIR, leg_label + "*.mp4")))
        if produced:
            final = os.path.join(BAKEOFF_DIR, leg_label + ".mp4")
            src = produced[-1]
            if os.path.abspath(src) != os.path.abspath(final):
                if os.path.exists(final):
                    os.remove(final)
                os.replace(src, final)
            result["clip"] = final
            result["clip_bytes"] = os.path.getsize(final)
            result["status"] = "ok"
            log("LEG %s OK | %ss/it | peakVRAM %.0f MB | peakSysRAM %.0f MB | %s"
                % (leg_label, result["s_per_it"], result["peak_vram_mb"],
                   result["peak_sysram_mb"], final))
        else:
            result["error"] = "no mp4 produced under %s" % BAKEOFF_DIR
            log("LEG %s FAIL: %s" % (leg_label, result["error"]))
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e)
        log("LEG %s EXCEPTION: %r" % (leg_label, e))
    finally:
        # tear down THIS leg's server so the next clamp tier boots clean (the
        # whole reason for boot-per-leg: no residency / no stale reserve carryover)
        try:
            reset_box()
        except Exception as e:  # noqa: BLE001
            log("teardown reset error: %r" % e)
    return result


def write_results(results):
    rj = os.path.join(BAKEOFF_DIR, "wan_bakeoff_results.json")
    with open(rj, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    lines = [
        "# Wan 2.2 TI2V-5B GGUF low-VRAM bakeoff",
        "",
        "Fixed still `%s`, 832x480x49, euler/simple, 30 steps, cfg 5.0, shift 5.0, "
        "seed 0. One-variable unet swap (no LoRA). Clamp = simulated usable VRAM "
        "via --reserve-vram on a %.0fGB card." % (FIXED_ASSETS[0],
                                                  results[0].get("total_vram_gb", 16)
                                                  if results else 16),
        "",
        "| leg | quant | clamp | reserve GB | s/it | peak VRAM (MB) | peak SysRAM (MB) "
        "| wall (s) | clip bytes | status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r.get("leg", "?"), r.get("quant", "?"), r.get("clamp", "?"),
            r.get("reserve_vram_gb", "-"), r.get("s_per_it", "-"),
            r.get("peak_vram_mb", "-"), r.get("peak_sysram_mb", "-"),
            r.get("wall_s", "-"), r.get("clip_bytes", "-"),
            r.get("status", "?") + ("" if r.get("status") == "ok"
                                    else " (%s)" % r.get("error", "")),
        ))
    with open(os.path.join(BAKEOFF_DIR, "wan_bakeoff_results.md"),
              "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# dry-validate: boot ONCE (no clamp), confirm node-class currency + quant enums,
# render NOTHING. The cheap no-GPU-cost preflight (CLAUDE.md: confirm before GPU).
# --------------------------------------------------------------------------
def dry_validate():
    server_log = os.path.join(BAKEOFF_DIR, "comfy_server_dryvalidate.log")
    proc = None
    ok = True
    try:
        reset_box()
        proc, _ = boot_server(server_log, reserve_gb=None)
        schemas = wait_ready()
        for cls in REQUIRED_CLASSES:
            present = cls in schemas
            log("  node-class %-26s present=%s" % (cls, present))
            ok = ok and present
        base = load_base_prompt(schemas)
        with open(os.path.join(BAKEOFF_DIR, "debug_base_prompt.json"),
                  "w", encoding="utf-8") as f:
            json.dump(base, f, indent=2)
        for label, fname in QUANTS:
            ue = resolve_unet_enum(schemas, fname)
            log("  quant %-18s -> %s" % (label, ue))
            if ue is None:
                log("    (not on disk yet -- Step 3 download gates this leg)")
        log("DRY-VALIDATE node-currency %s" % ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    finally:
        try:
            reset_box()
        except Exception as e:  # noqa: BLE001
            log("teardown reset error: %r" % e)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-validate", action="store_true",
                    help="boot once, confirm node-class currency + quant enums, "
                         "render NOTHING (cheap pre-flight)")
    ap.add_argument("--clamps", default="",
                    help="comma list of clamp tiers to run (default all: %s)"
                         % ",".join(c[0] for c in CLAMP_TIERS))
    ap.add_argument("--quants", default="",
                    help="comma list of quant labels to run (default all)")
    args = ap.parse_args()

    os.makedirs(BAKEOFF_DIR, exist_ok=True)
    stage_assets()
    log("bakeoff dir: %s" % BAKEOFF_DIR)
    log("workflow:    %s" % WORKFLOW)

    if args.dry_validate:
        return dry_validate()

    total_gb = total_vram_gb()
    log("total board VRAM: %.1f GB" % total_gb)

    clamp_sel = set(s.strip() for s in args.clamps.split(",") if s.strip())
    quant_sel = set(s.strip() for s in args.quants.split(",") if s.strip())
    clamps = [c for c in CLAMP_TIERS if not clamp_sel or c[0] in clamp_sel]
    quants = [q for q in QUANTS if not quant_sel or q[0] in quant_sel]

    results = []
    for clamp_label, target_gb in clamps:
        reserve_gb = reserve_for_target(target_gb, total_gb)
        for quant_label, fname in quants:
            results.append(run_leg(quant_label, fname, clamp_label,
                                   reserve_gb, total_gb))
            write_results(results)  # incremental
    write_results(results)
    log("DONE. %d legs." % len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
