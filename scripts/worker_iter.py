"""scripts/worker_iter.py -- one-iter bug-hunt worker.

Sprint H two-process supervisor design (synthesis 2026-05-17
section 2). The supervisor spawns this worker once per iter via
subprocess.Popen and waits with a 20-min timeout. The worker owns
the entire iter:

  1. Install exit hook that always writes logs/worker_iter_<n>.json
     (even on uncaught crash). The supervisor reads that file to
     classify the iter; no file means worker_crash.
  2. Pre-flight port check (synthesis section 2 step 2).
  3. Launch ComfyUI as the worker's DIRECT child via
     subprocess.Popen with the inline env+args copied from
     scripts/start_comfy_h0_baseline.bat -- NOT through that .bat
     and NOT through any PowerShell wrapper. A wrapped chain
     breaks `taskkill /F /PID <worker> /T` tree walking
     (synthesis section 1.1 + 6.2).
  4. Poll /system_stats with 60s timeout + verify PID owns port.
  5. OTR_WorkflowValidator live pass (optional, skipped if the
     validator subgraph isn't present in the target workflow).
  6. Pre-flight VRAM check via nvidia-smi.
  7. POST workflows/otr_scifi_16gb_full.json -- the single
     canonical workflow file. Jeffrey 2026-05-17 round-robin:
     "One workflow file. No bughunt sibling." The harness is now
     reconciled to the single source of truth; iters will be
     slower per the Mistral-Nemo writer default, which overnight
     tolerates.
  8. Poll /history with 15-min timeout.
  9. Classify outcome, overwrite result file.
 10. Unconditional teardown (synthesis section 2 step 10):
     taskkill /F /PID <comfyui> /T then ffmpeg-global escalation
     if any phantom remains.

Usage (spawned by supervisor only; no manual invocation):
  python scripts/worker_iter.py <iter_idx> --port <chosen_port>

Result file shape:
  {
    "iter": <int>,
    "port": <int>,
    "comfyui_pid": <int|null>,
    "prompt_id": <str|null>,
    "status": "SUCCESS|FAIL|TIMEOUT|...",
    "failure_class": <one of the synthesis section 2 classes>,
    "exception_message": <str|null>,
    "exception_type": <str|null>,
    "executed_count": <int>,
    "outputs_count": <int>,
    "peak_vram_gb": <float|null>,
    "wall_time_s": <float>,
    "start_ts": <iso>,
    "end_ts": <iso>,
    "comfyui_log_basename": <str>,
    "sweep_log_basename": <str|null>,
  }

Failure classes (fixed dictionary, synthesis section 2 +
2026-05-18 retest #8/#9 follow-ups):
  graph_widget, missing_model, writer_budget, writer_outline,
  llm_oom, video_oom, ffmpeg_composite, comfyui_startup,
  port_occupied, orphan_detected, vram_contaminated,
  crash_process, worker_crash, timeout, unknown.
"""
from __future__ import annotations

import argparse
import atexit
import datetime as dt
import json
import os
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

REPO = Path(r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio")
COMFY_DIR = Path(r"C:\Users\jeffr\Documents\ComfyUI")
LOGS_DIR = COMFY_DIR / "logs"

# Jeffrey 2026-05-17 overnight directive: reconcile harness to a
# single workflow source of truth. The bughunt sibling has been
# deleted; the loop POSTs the canonical _full.json. Iters run
# slower per the Mistral-Nemo writer default -- overnight tolerates
# this. C7 byte-identity is owned by the audio pipeline, not the
# writer-side workflow file.
WORKFLOW_PATH = REPO / "workflows" / "otr_scifi_16gb_full.json"

PY = Path(r"C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe")
COMFY_MAIN = Path(
    r"C:\Users\jeffr\AppData\Local\Programs\ComfyUI\resources\ComfyUI\main.py"
)

# Synthesis section 2 step 4: 60s startup timeout.
STARTUP_TIMEOUT_S = 60
# Sprint H 3.7 closure run (Jeffrey 2026-05-18): exec budget
# raised from 900s (15 min, calibrated for the writer phase
# alone) to 12600s (3.5 hr) for the full v2.0-alpha pipeline.
# Wall-time decomposition from memory project_humo_procgen_
# verified_2026-05-05 (commit ab4c8cf, sirens_print 36 min for
# 2 character lines + 1 announcer = ~12-18 min/clip):
#     audio branch       ~10 min  (writer + outline + MusicGen
#                                  + Kokoro + Bark + Assembler)
#     FLUX env + portraits  ~5 min   (radio bookend + PASS1=3)
#     LTX text encoder      ~2 min
#     LTX env motion         ~5 min   (per shot, PASS3 dependent)
#     HuMo Phase A/B/main  ~145 min  (12 character lines
#                                     x ~12 min each)
#     ffmpeg composite      ~3 min
#     total                ~170 min  (call it 3 hr)
# 3.5 hr gives ~30 min headroom for the dynamic offloader
# pageflush + ComfyUI's executor inter-node coordination.
EXEC_TIMEOUT_S = 12600
# Pre-flight VRAM thresholds per synthesis section 2 step 6.
VRAM_IDLE_TARGET_GB = 2.0
VRAM_WARN_GB = 4.0


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _http_get_json(url: str, timeout: int = 5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _vram_used_gb_via_smi() -> float | None:
    """Read nvidia-smi for VRAM used in GB. None on failure.

    Out-of-process check used at the pre-flight VRAM gate BEFORE
    ComfyUI's /system_stats is reachable. After /system_stats is
    up, we prefer that endpoint (it's the same number but cheaper).
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return None
        raw = (proc.stdout or "").strip().splitlines()
        if not raw:
            return None
        mb = int(raw[0].strip())
        return round(mb / 1024.0, 2)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def _vram_used_gb_via_comfy(url: str) -> float | None:
    j = _http_get_json(url + "/system_stats", timeout=3)
    if not j:
        return None
    try:
        d = j.get("devices", [{}])[0]
        total = d.get("vram_total") or 0
        free = d.get("vram_free") or 0
        return round((total - free) / (1024 ** 3), 2)
    except (KeyError, IndexError, TypeError):
        return None


def _port_is_free(port: int) -> bool:
    """Synthesis section 2 step 2: pre-flight port check (A2 fix).

    Pre-fix: netstat-based check + PID-equality readiness check
    failed under the uv launcher-stub model -- the stub holds the
    Popen-returned PID, the real cpython holds the TCP socket,
    netstat reports the real cpython, the equality check returned
    False against a healthy ComfyUI. False-positive `comfyui_startup`
    rejection on every iter.

    Post-fix (Jeffrey 2026-05-17 round-robin §A2):
        socket.bind(('127.0.0.1', port)) and immediately release.
        Bind succeeds -> port is free.
        Bind raises OSError -> port is held by something else
        (orphan ComfyUI, stale supervisor, another service).

    No netstat, no PID lookup, no stub vs real-cpython ambiguity.
    Combined with the post-launch shape check
    (`_comfyui_responding_with_real_shape`), stale orphans surface
    as `port_occupied` rather than masquerading as healthy ComfyUI.
    """
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


# Required JSON-shape keys for a healthy ComfyUI /system_stats body.
# Picked from the live response in iter-1 evidence so we know they're
# present; broad enough that a stale orphan returning {} or a stub
# HTTP server returning {"status": "ok"} cannot impersonate.
_SYSTEM_STATS_REQUIRED_PATHS = (
    ("system",),
    ("system", "comfyui_version"),
    ("system", "pytorch_version"),
    ("devices",),
)


def _comfyui_responding_with_real_shape(url: str) -> bool:
    """Synthesis section 2 step 4: post-launch readiness gate
    (A1 fix). Replaces the strict PID-owns-port check.

    Pre-fix: confirm that comfy_proc.pid == netstat owner of the
    port. Broke under the uv launcher-stub model (see _port_is_free
    docstring).

    Post-fix (Jeffrey 2026-05-17 round-robin §A1): trust that
    `/system_stats` returns a body shaped like real ComfyUI, not a
    spoofer. Required shape:
        body is a dict
        body.system is a dict containing comfyui_version + pytorch_version
        body.devices is a non-empty list
        body.devices[0] is a dict containing vram_total
    """
    j = _http_get_json(url + "/system_stats", timeout=3)
    if not isinstance(j, dict):
        return False
    # Top-level shape: a real ComfyUI returns "system" dict and a
    # non-empty "devices" list. A stub HTTP server / spoofer almost
    # never assembles both.
    sys_block = j.get("system")
    if not isinstance(sys_block, dict):
        return False
    if not isinstance(sys_block.get("comfyui_version"), str):
        return False
    if not isinstance(sys_block.get("pytorch_version"), str):
        return False
    devices = j.get("devices")
    if not isinstance(devices, list) or not devices:
        return False
    d0 = devices[0]
    if not isinstance(d0, dict):
        return False
    # vram_total is an int (bytes). 0 / None is suspect; ComfyUI on
    # the 5080 always reports the full board.
    vt = d0.get("vram_total")
    if not isinstance(vt, int) or vt <= 0:
        return False
    return True


def _detect_crash_subclass(comfy_log_text: str) -> str | None:
    """Identify a crash subclass from the ComfyUI log tail.

    Sprint H §3.7 Commit 1 (Jeffrey 2026-05-17 round-robin): when
    failure_class is 'crash_process', the subclass is metadata that
    helps operators triage without re-reading the log. The class
    itself stays 'crash_process'; subclass is None or one of:
        - 'access_violation': Windows fatal access violation /
          segfault triggered by torch storage / unmarshal / native ext
        - 'safetensors_load': load_torch_file or safetensors header
          failures (often pair with access_violation when shape
          mismatches make torch.storage read invalid memory)
        - 'cuda': cudaError / CUDA kernel launch / device-side assert
        - None: no recognizable subclass marker found in tail
    """
    if not comfy_log_text:
        return None
    text = comfy_log_text.lower()
    if "windows fatal exception: access violation" in text:
        return "access_violation"
    if "cudaerror" in text or "cuda error" in text or "device-side assert" in text:
        return "cuda"
    if (
        "load_torch_file" in text
        or "safetensors" in text
        and ("header" in text or "could not load" in text or "load_clip" in text)
    ):
        return "safetensors_load"
    return None


def _classify_failure(
    *,
    comfy_log_text: str,
    exception_message: str | None,
    exception_type: str | None,
    status: str,
    executed_count: int = 0,
) -> str:
    """Map a (log tail + exception + execution counters) tuple to a
    synthesis section 2 failure class. Pure dispatch on string content;
    no I/O here.

    Sprint H §3.7 hardening (Jeffrey 2026-05-17): ComfyUI's /history
    returns `status_str='success'` (lowercase) when the prompt was
    accepted by the queue but had validation errors that produced
    zero executed nodes -- the prompt got "executed" only insofar as
    ComfyUI processed the rejection. Map that case to `graph_widget`
    instead of `unknown`. Plain executed prompts retain `success`.

    Sprint H §3.7 Commit 1: 'CRASH_PROCESS' status (synthesized by
    `_poll_until_done` on ComfyUI process death) maps to
    'crash_process' class. Subclass detection runs separately via
    `_detect_crash_subclass` and lands in the result file as
    metadata; failure_class stays 'crash_process'.
    """
    if status == "CRASH_PROCESS":
        return "crash_process"
    status_lower = (status or "").lower()
    if status_lower == "success":
        if executed_count == 0:
            # ComfyUI accepted the POST, rejected all outputs during
            # validation. submit_prompt's node_errors guard should have
            # caught this at POST time; if it didn't (e.g. a future
            # ComfyUI release omits node_errors from the response),
            # this branch is the second-line classifier.
            return "graph_widget"
        return "success"

    msg_l = ((exception_message or "") + "\n" + (exception_type or "")).lower()
    log_l = (comfy_log_text or "").lower()

    # graph_widget: widget validation surfaced by validator or
    # ComfyUI's own /prompt validation, BEFORE execution starts.
    graph_widget_markers = (
        "prompt outputs validation",
        "widget_name",
        "invalid input",
        "return type mismatch",
        "object_info",
    )
    if any(m in msg_l for m in graph_widget_markers):
        return "graph_widget"

    # missing_model: model not on disk / path resolution / HF_HOME
    # / registry key.
    missing_markers = (
        "no such file or directory",
        "[not downloaded]",
        "huggingfacehub",
        "snapshot_download",
        "errno 2",
    )
    if any(m in msg_l for m in missing_markers):
        return "missing_model"

    # writer_budget: EpisodeBudget preflight validator rejected the
    # widget combo BEFORE any LLM call. Surfaced 2026-05-18 §3.7
    # retest #9 -- when smoke target_words rose 30 -> 300, the
    # smoke act_count=1 fell below the new default_act_count(300)=3.
    # The defect is in the harness's smoke override profile, NOT a
    # topology, VRAM, or writer issue. Module-level marker catches
    # any future sibling exception types from _otr_episode_budget.
    budget_markers = (
        "invalidepisodebudgeterror",
        "_otr_episode_budget",
        "below default ",
        "exceeds max ",
        "below minimum of 30",
    )
    if any(m in msg_l for m in budget_markers):
        return "writer_budget"

    # writer_outline: outline phase produced an outline that fails
    # validation. Surfaced 2026-05-18 §3.7 retest #8 -- Mistral-Nemo
    # on tight smoke budgets generated beats with target_words=0
    # (pydantic >=3 floor) or arc_phase mismatches against the
    # configured arc_phases budget. The defect is in the outline
    # generator's compliance with the schema, NOT a topology or
    # VRAM issue. Routes here so the supervisor's same-class halt
    # rule reports the real defect instead of treating it as
    # `unknown`.
    outline_markers = (
        "outlinefailederror",
        "outlinebudgetviolation",
        "outline generation failed after",
        "_otr_outline",
    )
    if any(m in msg_l for m in outline_markers):
        return "writer_outline"

    # OOM split into LLM phase vs video phase by which node-class
    # appears in the log tail above the OOM line.
    if "outofmemoryerror" in msg_l or "cuda out of memory" in log_l:
        # Video-side classes that come AFTER the writer in the
        # graph order.
        video_class_markers = (
            "otr_batchhumorender",
            "otr_batchltxrender",
            "otr_batchfluxrender",
            "fluxsamplercustom",
            "ksampler",
            "vaedecode",
        )
        if any(m in log_l for m in video_class_markers):
            return "video_oom"
        return "llm_oom"

    # ffmpeg composite / audio mux failure.
    ffmpeg_markers = (
        "ffmpeg",
        "audio mux",
        "concat",
        "moov atom not found",
    )
    if any(m in msg_l for m in ffmpeg_markers) or "ffmpeg" in log_l[-2000:]:
        # Only call it ffmpeg_composite if the actual exception is
        # ffmpeg-pointed; bare "ffmpeg" hits in the log are noise.
        if any(m in msg_l for m in ffmpeg_markers):
            return "ffmpeg_composite"

    if status == "TIMEOUT":
        return "timeout"

    return "unknown"


def _kill_tree(pid: int) -> None:
    """taskkill /F /PID <pid> /T. Best-effort; missing pid is fine."""
    try:
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid), "/T"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


def _phantom_ffmpeg_present() -> bool:
    try:
        proc = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq ffmpeg.exe"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return "ffmpeg.exe" in (proc.stdout or "")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _ffmpeg_global_kill() -> None:
    """Last-resort ffmpeg sweep + audit log line.

    Synthesis section 2 step 10: if the worker's process tree
    kill leaves a phantom ffmpeg.exe alive, escalate to
    `taskkill /F /IM ffmpeg.exe` AND log to logs\\teardown.log
    so the unrelated-ffmpeg edge case is auditable.
    """
    teardown_log = LOGS_DIR / "teardown.log"
    teardown_log.parent.mkdir(parents=True, exist_ok=True)
    with teardown_log.open("a", encoding="utf-8") as f:
        f.write(
            f"[{_now()}] ffmpeg_global_kill: phantom ffmpeg.exe "
            "survived tree kill; escalating to /IM\n"
        )
    try:
        subprocess.run(
            ["taskkill", "/F", "/IM", "ffmpeg.exe"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


class WorkerResult:
    """Mutable result-record holder. On atexit, dumps to JSON."""

    def __init__(self, iter_idx: int, port: int):
        self.iter_idx = iter_idx
        self.port = port
        self.path = LOGS_DIR / f"worker_iter_{iter_idx:03d}.json"
        self.data: dict = {
            "iter": iter_idx,
            "port": port,
            "comfyui_pid": None,
            "prompt_id": None,
            "status": "worker_crash",
            "failure_class": "worker_crash",
            "crash_subclass": None,
            "exception_message": None,
            "exception_type": None,
            "executed_count": 0,
            "outputs_count": 0,
            "peak_vram_gb": None,
            "wall_time_s": None,
            "start_ts": _now(),
            "end_ts": None,
            "comfyui_log_basename": f"comfy_session_iter_{iter_idx:03d}.log",
            "sweep_log_basename": None,
        }
        self.t0 = time.time()

    def update(self, **kw) -> None:
        self.data.update(kw)

    def write(self) -> None:
        if self.data.get("end_ts") is None:
            self.data["end_ts"] = _now()
        if self.data.get("wall_time_s") is None:
            self.data["wall_time_s"] = round(time.time() - self.t0, 2)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic-ish: write to temp, replace.
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)


def _launch_comfyui(
    iter_idx: int,
    port: int,
) -> tuple[subprocess.Popen | None, Path]:
    """Spawn ComfyUI as a direct child of this worker.

    Inline env+args copied from scripts/start_comfy_h0_baseline.bat.
    NOT invoked through that bat; a .bat in the launch chain breaks
    `taskkill /F /PID <worker> /T` tree walking per synthesis
    section 1.1 + 6.2.
    """
    log_path = LOGS_DIR / f"comfy_session_iter_{iter_idx:03d}.log"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TORCH_SDPA_BACKEND"] = "math"
    env["HF_HOME"] = r"C:\ComfyUI-Models\huggingface"

    args = [
        str(PY),
        str(COMFY_MAIN),
        "--port", str(port),
        "--highvram",
        "--force-fp16",
        "--cuda-malloc",
        "--user-directory", str(COMFY_DIR),
    ]

    log_f = log_path.open("w", encoding="utf-8", errors="replace")
    try:
        proc = subprocess.Popen(
            args,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(COMFY_DIR),
            # Important: NOT shell=True; that would insert a cmd.exe
            # wrapper into the tree and break taskkill /T.
            shell=False,
        )
    except FileNotFoundError as exc:
        log_f.close()
        raise RuntimeError(f"ComfyUI launch FileNotFoundError: {exc}") from exc
    return proc, log_path


def _post_workflow(url: str) -> str | None:
    """Patch + POST the canonical _full.json workflow. Returns
    prompt_id on SUCCESS, None on schema / submit error.

    Smoke patches: 300 words, 2 characters, 1 act. _full.json
    carries the canonical writer model on widgets[5]/[6] -- no
    model patch is needed; the patch path is just the smoke
    profile.

    target_words history: was 30 (mirrored scripts/queue_smoke.py)
    until 2026-05-18 §3.7 retest #8. With target_words=30 the
    outline budget allocator drove per-beat target_words to 0 on
    some beats (schema requires >=3), and Mistral-Nemo on a tight
    total budget routinely produced arc_phase mismatches against
    a single-phase ['scene'] budget. Raising to 300 gives the
    outline ~30-50 words per beat, well inside Mistral-Nemo's
    reliable zone, and still inside the §3.7 wall-time budget.
    Production-size scripts (target_words=350 default on the
    node 1 widget) were never affected -- this was a smoke-config
    artifact, not a writer defect.
    """
    # Lazy import so a /prompt URL failure surfaces from the right
    # frame rather than at module import time.
    from otr_api import (  # noqa: E402
        fetch_schemas,
        load_workflow,
        patch_widget_by_name,
        submit_prompt,
        workflow_to_api_prompt,
    )

    # otr_api reads COMFYUI_URL from env. Set it to the worker's
    # chosen port for this call.
    os.environ["COMFYUI_URL"] = url

    # Smoke profile overrides aligned to the EpisodeBudget validator
    # constraint set (see nodes/_otr_episode_budget.py
    # compute_episode_budget). For target_words=300:
    #     constraint                              smoke value  ok?
    #     target_words >= 30                      300          yes
    #     1 <= act_count <= 7                     3            yes
    #     num_characters >= 1                     2            yes
    #     act_count >= default_act_count(tw)      3 (==dflt)   yes
    #         default_act_count(300) = 3
    #     act_count <= max_act_count(tw)          3 (<=6)      yes
    #         max_act_count(300) = 6
    # act_count history: was 1 until 2026-05-18 §3.7 retest #10.
    # When target_words rose from 30 -> 300 (retest #9), the
    # EpisodeBudget validator's act_count >= default rule started
    # firing because default_act_count(300) = 3 but smoke shipped 1.
    # Raised to 3 (the conservative default for tw=300).
    #
    # include_act_breaks=False history: was the workflow default
    # (True) until 2026-05-18 §3.7 retest #11. With include_act_breaks
    # =True the budget records `music_inter_count = act_count - 1`
    # (= 2 at act_count=3). The outline LLM then emits music-interlude
    # beats inline (beat_id='m001', 'm002') which violate the outline
    # schema's beat_id pattern `^b\d{3}$` plus lack required intent /
    # target_words / mood fields. Setting include_act_breaks=False
    # zeros music_inter_count so the outline never asks for music
    # beats. Independent of writer model choice (was hit on both
    # Mistral-Nemo and the Gemma-4 destination -- belt-and-braces).
    schemas = fetch_schemas()
    wf = load_workflow(str(WORKFLOW_PATH))
    patch_widget_by_name(wf, 1, "target_words", 300, schemas)
    patch_widget_by_name(wf, 1, "num_characters", 2, schemas)
    patch_widget_by_name(wf, 1, "act_count", 3, schemas)
    patch_widget_by_name(wf, 1, "include_act_breaks", False, schemas)
    api = workflow_to_api_prompt(wf, schemas)
    return submit_prompt(api)


def _poll_until_done(
    url: str,
    prompt_id: str,
    deadline_s: float,
    *,
    on_vram: callable | None = None,
    comfy_proc: subprocess.Popen | None = None,
) -> tuple[str, str | None, str | None, int, int]:
    """Poll /history until status_str != PENDING, ComfyUI process death,
    or deadline hit.

    Returns
        (status, exception_message, exception_type,
         executed_count, outputs_count).

    Status values produced by this poll loop:
        - <status_str from /history>: normal resolution path
        - 'CRASH_PROCESS': ComfyUI died before /history resolved.
          comfy_proc.poll() is checked each tick; a non-None returncode
          BEFORE the prompt resolves means the process is gone (fatal
          access violation, OOMKill, segfault, etc.). The exit code
          surfaces as exception_type 'comfyui_exit_<rc>' so the
          classifier can log it without further log-scraping.
        - 'TIMEOUT': /history poll deadline elapsed with ComfyUI still
          alive but the prompt unresolved.

    Sprint H §3.7 Commit 1 (Jeffrey 2026-05-17 round-robin): pre-fix
    the worker spun the full EXEC_TIMEOUT_S deadline on every fatal
    ComfyUI crash because /history can't resolve a prompt for a dead
    server. The crash class (safetensors load access violation,
    cudaError, OOMKill) needs to be classified within seconds, not
    fifteen minutes. Track comfy_proc aliveness each tick to bound
    the worst-case worker_wall for crash classes.
    """
    while time.time() < deadline_s:
        # Crash detection BEFORE /history poll. A dead ComfyUI cannot
        # produce a /history entry; spinning the deadline is wasteful.
        if comfy_proc is not None:
            rc = comfy_proc.poll()
            if rc is not None:
                return (
                    "CRASH_PROCESS",
                    f"ComfyUI process exited (rc={rc}) before /history "
                    "resolved the prompt",
                    f"comfyui_exit_{rc}",
                    0,
                    0,
                )
        h = _http_get_json(f"{url}/history/{prompt_id}", timeout=5)
        if on_vram is not None:
            v = _vram_used_gb_via_comfy(url)
            if v is not None:
                on_vram(v)
        if h and prompt_id in h:
            entry = h[prompt_id]
            status_block = entry.get("status", {}) or {}
            status_str = status_block.get("status_str", "unknown")
            outputs_count = len(entry.get("outputs") or {})
            exc_msg = None
            exc_type = None
            executed_count = 0
            for m in status_block.get("messages", []) or []:
                if not isinstance(m, list) or len(m) < 2:
                    continue
                tag, payload = m[0], m[1]
                if tag == "execution_error":
                    exc_msg = (payload.get("exception_message") or "")[:1500]
                    exc_type = payload.get("exception_type")
                    executed_count = len(payload.get("executed") or [])
            return status_str, exc_msg, exc_type, executed_count, outputs_count
        time.sleep(5)
    return "TIMEOUT", "exceeded poll deadline", None, 0, 0


def _tail_file(path: Path, n_lines: int = 200) -> str:
    if not path.is_file():
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n_lines:])
    except OSError:
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sprint H bug-hunt worker (one iter, then exits)."
    )
    parser.add_argument("iter", type=int)
    parser.add_argument("--port", type=int, required=True,
                        help="TCP port for ComfyUI HTTP server.")
    args = parser.parse_args()

    iter_idx = int(args.iter)
    port = int(args.port)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    result = WorkerResult(iter_idx=iter_idx, port=port)
    # Synthesis section 2 step 1: always-write-a-result-file
    # exit hook. Register BEFORE anything else can raise.
    atexit.register(result.write)

    comfy_proc: subprocess.Popen | None = None
    url = f"http://127.0.0.1:{port}"

    try:
        # Step 2: port preflight via socket.bind (A2 fix).
        # Pre-fix used netstat; broke under uv stub model.
        if not _port_is_free(port):
            result.update(
                status="port_occupied",
                failure_class="port_occupied",
                exception_message=(
                    f"port {port} bind failed at preflight (orphan ComfyUI "
                    "or another service is holding the port)"
                ),
            )
            return 2

        # Step 3: launch ComfyUI as direct child.
        comfy_proc, log_path = _launch_comfyui(iter_idx, port)
        result.update(comfyui_pid=int(comfy_proc.pid))

        # Step 4: poll /system_stats and confirm body shape matches
        # real ComfyUI (A1 fix). Pre-fix gated on PID-owns-port via
        # netstat; broke under uv stub model. Post-fix: shape-check
        # the JSON response so a stub HTTP server / spoofer can't
        # masquerade as ComfyUI.
        startup_deadline = time.time() + STARTUP_TIMEOUT_S
        ready = False
        while time.time() < startup_deadline:
            if _comfyui_responding_with_real_shape(url):
                ready = True
                break
            # ComfyUI died before becoming ready -> classify and bail.
            if comfy_proc.poll() is not None:
                break
            time.sleep(2)
        if not ready:
            result.update(
                status="comfyui_startup",
                failure_class="comfyui_startup",
                exception_message=(
                    f"/system_stats did not return a real-ComfyUI-shaped "
                    f"body within {STARTUP_TIMEOUT_S}s "
                    f"(comfy exit={comfy_proc.poll()})"
                ),
            )
            return 3

        # Step 5: validator preflight is graph-conditional. Skipped
        # at this revision; the worker leans on ComfyUI's own
        # /prompt validation pass to surface graph_widget class.
        # When OTR_WorkflowValidator wiring is finalized in the
        # canonical _full.json workflow, this is where the validator-only
        # subgraph submit would go (synthesis section 2 step 5).

        # Step 6: pre-flight VRAM gate.
        vram_now = _vram_used_gb_via_comfy(url) or _vram_used_gb_via_smi()
        if vram_now is not None and vram_now > VRAM_WARN_GB:
            # Filtered re-kill is the supervisor's job (synthesis
            # section 2 step 6). The worker just refuses to run
            # against a contaminated card.
            result.update(
                status="vram_contaminated",
                failure_class="vram_contaminated",
                peak_vram_gb=vram_now,
                exception_message=(
                    f"pre-flight VRAM {vram_now:.2f} GB exceeds "
                    f"{VRAM_WARN_GB:.2f} GB ceiling"
                ),
            )
            return 4

        # Step 7: POST the canonical _full.json workflow.
        try:
            prompt_id = _post_workflow(url)
        except Exception as exc:  # noqa: BLE001
            result.update(
                status="FAIL",
                failure_class="graph_widget",
                exception_message=str(exc)[:1500],
                exception_type=type(exc).__name__,
            )
            return 5
        if not prompt_id:
            result.update(
                status="FAIL",
                failure_class="graph_widget",
                exception_message="/prompt submit returned no prompt_id",
            )
            return 5
        result.update(prompt_id=prompt_id)

        # Step 8 + 9: poll, classify, write result.
        peak = vram_now or 0.0

        def _peak(v: float) -> None:
            nonlocal peak
            if v > peak:
                peak = v

        deadline = time.time() + EXEC_TIMEOUT_S
        status, exc_msg, exc_type, exec_count, out_count = _poll_until_done(
            url, prompt_id, deadline,
            on_vram=_peak,
            comfy_proc=comfy_proc,
        )
        # Sprint H §3.7 Commit 1: on process-death classification,
        # tail more log lines (250 vs 200) so the operator has enough
        # context to spot the fatal frame + preceding load_torch_file
        # / load_clip / quantization warnings without re-reading the
        # raw log.
        tail_lines = 250 if status == "CRASH_PROCESS" else 200
        log_tail = _tail_file(log_path, n_lines=tail_lines)
        failure_class = _classify_failure(
            comfy_log_text=log_tail,
            exception_message=exc_msg,
            exception_type=exc_type,
            status=status,
            executed_count=exec_count,
        )
        # Subclass metadata only fires when failure_class is
        # crash_process. Stays None otherwise so existing rows are
        # unchanged.
        crash_subclass = None
        if failure_class == "crash_process":
            crash_subclass = _detect_crash_subclass(log_tail)
        # Diagnostic: surface raw history shape so future drift of the
        # status/executed/outputs triple is immediately diagnosable from
        # the result file alone -- no log-tail forensics required.
        result.update(
            status=status,
            failure_class=failure_class,
            crash_subclass=crash_subclass,
            exception_message=exc_msg,
            exception_type=exc_type,
            executed_count=exec_count,
            outputs_count=out_count,
            history_status_raw=status,
            history_executed_count=exec_count,
            history_outputs_count=out_count,
            peak_vram_gb=round(peak, 2) if peak else None,
        )
        return 0

    except Exception as exc:  # noqa: BLE001
        # Worker-thread crash. The atexit hook still fires and
        # writes a result file; we only update the data first.
        tb = traceback.format_exc(limit=4)
        result.update(
            status="worker_crash",
            failure_class="worker_crash",
            exception_message=(str(exc) + "\n" + tb)[:1500],
            exception_type=type(exc).__name__,
        )
        return 99

    finally:
        # Step 10: unconditional teardown.
        if comfy_proc is not None and comfy_proc.poll() is None:
            _kill_tree(int(comfy_proc.pid))
            time.sleep(2)
        if _phantom_ffmpeg_present():
            _ffmpeg_global_kill()


if __name__ == "__main__":
    sys.exit(main())
