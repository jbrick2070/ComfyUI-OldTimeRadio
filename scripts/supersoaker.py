"""
SUPERSOAKER -- Recursive phase verifier
========================================
Walks through P0 -> P1 -> P2 feature gates one episode at a time.
After each phase:
  * If all required probes fire and no strategic gates trip, advance.
  * Otherwise, STOP and write a handoff note for Jeffrey. No autofix.
    Design decisions stay with Jeffrey.

Each phase runs ONE episode against ComfyUI (http://localhost:8000) with
a config chosen to exercise the features that shipped in that phase.
Phase results are appended to logs/supersoaker_log.md.

Usage:
    python scripts/supersoaker.py                     # P0 -> P1 -> P2
    python scripts/supersoaker.py --only P0           # run a single phase
    python scripts/supersoaker.py --start-at P1       # skip P0
    python scripts/supersoaker.py --no-advance        # run one, then stop

Strategic gates (always halt + hand off to Jeffrey):
    * VRAM peak > 14.5 GB
    * PARSE_FATAL in runtime log
    * DRIFT_DETECTED (widget drift)
    * Treatment written with zero dialogue lines
    * Episode FAIL from ComfyUI /history endpoint
    * Any probe marked `required` did not fire

API plumbing (prompt POST, schema conversion, history polling, etc.)
is reused from soak_operator so the soak contract stays one source of truth.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
import uuid

import requests

# ---------------------------------------------------------------------------
# Reuse proven API plumbing from soak_operator
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from soak_operator import (  # noqa: E402
    COMFYUI,
    WORKFLOW_PATH,
    RUNTIME_LOG,
    POLL_S,
    WV_GENRE,
    WV_TARGET_WORDS,
    WV_TARGET_LENGTH,
    WV_STYLE,
    WV_CREATIVITY,
    WV_OPT_PROFILE,
    _workflow_to_api_prompt,
    count_treatments,
    check_treatment,
)

# Additional widget index not exported from soak_operator
WV_SELF_CRITIQUE = 7

TIMEOUT_S = 1800
VRAM_CEILING_GB = 14.5

OTR_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
LOG_DIR = os.path.join(OTR_ROOT, "logs")
SUPERSOAKER_LOG = os.path.join(LOG_DIR, "supersoaker_log.md")
HANDOFF_PATH = os.path.join(OTR_ROOT, "JEFFREY_HANDOFF.md")

# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------
# Each phase defines:
#   config   - widgets_values overrides for node 1
#   probes   - list of (id, name, regex, required) tuples scanned against
#              the NEW runtime log tail produced by this phase's episode
#   notes    - what we're verifying (human-readable, prints in report)

PHASE_P0 = {
    "id": "P0",
    "title": "P0 -- self-critique guard + director JSON schema",
    "config": {
        "genre": "hard_sci_fi",
        "words": 700,
        "length": "medium (5 acts)",
        "style": "tense claustrophobic",
        "creativity": "balanced",
        "profile": "Standard",
        "self_critique": True,  # REQUIRED to exercise P0 #1
    },
    "probes": [
        ("P0_1", "Self-critique dialogue guard",
         r"CRITIQUE_(COLLAPSE|PASS|GUARD|APPLIED|REVISE_DROPPED)", True),
        ("P0_2", "Director schema validator",
         r"DIRECTOR_SCHEMA", True),
    ],
    "notes": (
        "Verifies: character-line-count post-critique check and Director JSON "
        "schema validate/repair fire on a standard episode."
    ),
}

PHASE_P1 = {
    "id": "P1",
    "title": "P1 -- length-sorted Bark + VRAM sentinel + chaos",
    "config": {
        "genre": "cosmic_horror",
        "words": 1050,
        "length": "medium (5 acts)",
        "style": "chaotic black-mirror",
        "creativity": "maximum chaos",  # P1 #9
        "profile": "Standard",
        "self_critique": False,
    },
    "probes": [
        ("P1_5", "Bark batch ran (length sort implicit)",
         r"BatchBark(Generator)?", True),
        ("P1_7", "VRAM sentinel on bark batch",
         r"VRAM_SENTINEL_(ENTRY|EXIT|OFFLOAD).*bark_batch", True),
        ("P1_9", "Maximum chaos creativity accepted",
         r"(creativity=maximum chaos|creativity='maximum chaos')", False),
    ],
    "notes": (
        "Verifies: Bark batch path runs, sentinel fires on entry/exit, and "
        "the maximum-chaos creativity tier is accepted by the scriptwriter."
    ),
}

PHASE_P2 = {
    "id": "P2",
    "title": "P2 -- per-LLM-call VRAM snapshots",
    "config": {
        "genre": "cyberpunk",
        "words": 700,
        "length": "short (3 acts)",
        "style": "hard-sci-fi procedural",
        "creativity": "balanced",
        "profile": "Standard",
        "self_critique": False,
    },
    "probes": [
        ("P2_12", "VRAM snapshot around LLM calls",
         r"llm_generate_(entry|exit)", True),
    ],
    "notes": (
        "Verifies: per-LLM-call VRAM snapshots bracket every _generate_with_llm "
        "call, enabling post-hoc VRAM accounting."
    ),
}

PHASES = [PHASE_P0, PHASE_P1, PHASE_P2]
PHASES_BY_ID = {p["id"]: p for p in PHASES}

# ---------------------------------------------------------------------------
# Strategic gates: patterns that always halt the sequencer
# ---------------------------------------------------------------------------
STRATEGIC_GATES = [
    ("PARSE_FATAL",
     re.compile(r"PARSE_FATAL", re.IGNORECASE)),
    ("DRIFT_DETECTED",
     re.compile(r"DRIFT_DETECTED", re.IGNORECASE)),
    ("FFMPEG_HANG",
     re.compile(r"ffmpeg.*(hang|timeout|killed)", re.IGNORECASE)),
    ("SUBPROCESS_TIMEOUT",
     re.compile(r"subprocess.*timeout", re.IGNORECASE)),
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def print_banner(msg: str) -> None:
    bar = "=" * 72
    print(bar, flush=True)
    print(f"  {msg}", flush=True)
    print(bar, flush=True)


def print_section(msg: str) -> None:
    print("", flush=True)
    print(f"  {msg}", flush=True)
    print(f"  {'-' * 60}", flush=True)


def ensure_dirs() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def append_log(text: str) -> None:
    try:
        with open(SUPERSOAKER_LOG, "a", encoding="utf-8") as f:
            f.write(text)
    except OSError as e:
        print(f"  WARNING: could not write supersoaker log: {e}", flush=True)


def write_handoff(reason: str, phase: dict, detail: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    body = (
        f"# SUPERSOAKER HANDOFF\n\n"
        f"**Stopped:** {stamp}\n"
        f"**Phase:** {phase['id']} -- {phase['title']}\n"
        f"**Reason:** {reason}\n\n"
        f"## Detail\n\n{detail}\n\n"
        f"## Next steps (your call, not mine)\n\n"
        f"- Inspect the new tail of `{RUNTIME_LOG}` for the phase just run.\n"
        f"- Check the treatment file written (if any) under ComfyUI output.\n"
        f"- Decide whether to:\n"
        f"  1. Fix the feature and re-run `python scripts/supersoaker.py "
        f"--only {phase['id']}`\n"
        f"  2. Log a bug in `BUG_LOG.md` and promote to Bug Bible if "
        f"applicable.\n"
        f"  3. Revise the probe itself if the feature is working but the "
        f"regex is wrong.\n\n"
        f"Design strategy stays with you. I will not autofix.\n"
    )
    with open(HANDOFF_PATH, "w", encoding="utf-8") as f:
        f.write(body)
    print(f"  Handoff written: {HANDOFF_PATH}", flush=True)


# ---------------------------------------------------------------------------
# Log tail probe
# ---------------------------------------------------------------------------
class RuntimeLogProbe:
    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        self.start_offset = self._current_size()

    def _current_size(self) -> int:
        try:
            return os.path.getsize(self.log_path)
        except OSError:
            return 0

    def read_new_lines(self) -> list[str]:
        try:
            with open(self.log_path, "r", encoding="utf-8",
                      errors="replace") as f:
                f.seek(self.start_offset)
                return f.readlines()
        except OSError:
            return []


def latest_runtime_line() -> str:
    try:
        with open(RUNTIME_LOG, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, 2)
            end = f.tell()
            f.seek(max(0, end - 4000))
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            return lines[-1] if lines else ""
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Workflow + submit
# ---------------------------------------------------------------------------
def patch_workflow(workflow: dict, cfg: dict) -> None:
    for node in workflow.get("nodes", []):
        if node.get("id") == 1 and node.get("type") == "OTR_Gemma4ScriptWriter":
            wv = node["widgets_values"]
            wv[WV_GENRE] = cfg["genre"]
            wv[WV_TARGET_WORDS] = cfg["words"]
            wv[WV_TARGET_LENGTH] = cfg["length"]
            wv[WV_STYLE] = cfg["style"]
            wv[WV_CREATIVITY] = cfg["creativity"]
            wv[WV_OPT_PROFILE] = cfg["profile"]
            wv[WV_SELF_CRITIQUE] = cfg["self_critique"]
            return
    raise RuntimeError(
        "Workflow patch failed: no node id=1 of type OTR_Gemma4ScriptWriter"
    )


def wait_for_clear_queue(max_wait_s: int = 600) -> None:
    start = time.time()
    while time.time() - start < max_wait_s:
        try:
            q = requests.get(f"{COMFYUI}/queue", timeout=10).json()
            running = len(q.get("queue_running", []))
            pending = len(q.get("queue_pending", []))
            if running == 0 and pending == 0:
                return
            print(f"  Queue busy (running={running}, pending={pending}) "
                  f"-- waiting {POLL_S}s...", flush=True)
            time.sleep(POLL_S)
        except Exception:
            return


def submit(api_prompt: dict) -> str:
    client_id = str(uuid.uuid4())
    resp = requests.post(
        f"{COMFYUI}/prompt",
        json={"prompt": api_prompt, "client_id": client_id},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"POST /prompt returned HTTP {resp.status_code}: {resp.text[:300]}"
        )
    prompt_id = resp.json().get("prompt_id")
    if not prompt_id:
        raise RuntimeError(
            f"Submit response missing prompt_id: {resp.text[:300]}"
        )
    return prompt_id


def poll_until_done(prompt_id: str, tail_every_s: int = 20) -> tuple[str, str]:
    start = time.time()
    next_tail = start + tail_every_s
    result = "TIMEOUT"
    error_msg = ""

    while time.time() - start < TIMEOUT_S:
        if time.time() >= next_tail:
            last = latest_runtime_line()
            if last:
                print(f"  [{int(time.time() - start)}s] {last[:120]}",
                      flush=True)
            next_tail = time.time() + tail_every_s

        try:
            hist = requests.get(
                f"{COMFYUI}/history/{prompt_id}", timeout=10
            ).json()
            if prompt_id in hist:
                status = hist[prompt_id].get("status", {})
                if status.get("completed", False):
                    result = "SUCCESS"
                    break
                if status.get("status_str") == "error":
                    result = "FAIL"
                    error_msg = str(
                        status.get("messages", "Workflow execution error")
                    )[:500]
                    break
        except Exception:
            pass
        time.sleep(POLL_S)

    return result, error_msg


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------
def evaluate_probes(new_log_lines: list[str], phase: dict) -> list[dict]:
    text = "\n".join(new_log_lines)
    results = []
    for probe_id, name, pattern, required in phase["probes"]:
        regex = re.compile(pattern, re.IGNORECASE)
        hits = len(regex.findall(text))
        if required:
            passed = hits > 0
            verdict = "PASS" if passed else "FAIL"
        else:
            passed = True
            verdict = "OK" if hits > 0 else "NOT SEEN"
        results.append({
            "id": probe_id,
            "name": name,
            "hits": hits,
            "required": required,
            "passed": passed,
            "verdict": verdict,
        })
    return results


def evaluate_strategic_gates(new_log_lines: list[str]) -> list[str]:
    text = "\n".join(new_log_lines)
    trips = []
    for name, pattern in STRATEGIC_GATES:
        if pattern.search(text):
            trips.append(name)
    return trips


def extract_vram_peak(new_log_lines: list[str]) -> float | None:
    peak = None
    for line in new_log_lines:
        m = re.search(r"VRAM_SNAPSHOT .*? peak_gb=([\d.]+)", line)
        if m:
            try:
                v = float(m.group(1))
                if peak is None or v > peak:
                    peak = v
            except ValueError:
                continue
    return peak


def extract_dialogue_count(new_log_lines: list[str]) -> int | None:
    for line in reversed(new_log_lines):
        m = re.search(r"ScriptWriter: CAST_MAP .*? \| (\d+) lines", line)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------
def run_phase(phase: dict, workflow_path: str) -> dict:
    """Run a single phase. Returns a report dict with keys:
        result, probes, gates_tripped, vram_peak, dialogue, treatment,
        duration, error
    """
    print_banner(f"SUPERSOAKER PHASE {phase['id']}: {phase['title']}")
    print(f"  Config: {phase['config']}", flush=True)
    print(f"  Notes : {phase['notes']}", flush=True)
    print("", flush=True)

    # Load + patch workflow (fresh copy each phase)
    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)
    workflow = copy.deepcopy(workflow)
    patch_workflow(workflow, phase["config"])

    # Mark log offset BEFORE submitting
    probe = RuntimeLogProbe(RUNTIME_LOG)

    # Convert web -> API format
    print("  Fetching node schemas...", flush=True)
    schemas = requests.get(f"{COMFYUI}/object_info", timeout=30).json()
    api_prompt = _workflow_to_api_prompt(workflow, schemas)

    wait_for_clear_queue()
    treatments_before = count_treatments()
    print("  Submitting prompt...", flush=True)

    start_time = time.time()
    result, error_msg = "UNKNOWN", ""
    try:
        prompt_id = submit(api_prompt)
        print(f"  Submitted. prompt_id={prompt_id}", flush=True)
        print(f"  Polling (max {TIMEOUT_S}s)...", flush=True)
        result, error_msg = poll_until_done(prompt_id)
    except Exception as e:
        result = "FAIL"
        error_msg = str(e)[:500]
        print(f"  EXCEPTION: {error_msg}", flush=True)

    duration = int(time.time() - start_time)
    new_lines = probe.read_new_lines()
    has_treatment, _ = check_treatment(treatments_before)

    probe_results = evaluate_probes(new_lines, phase)
    gates = evaluate_strategic_gates(new_lines)
    vram_peak = extract_vram_peak(new_lines)
    dialogue = extract_dialogue_count(new_lines)

    # Strategic gate: VRAM ceiling
    if vram_peak is not None and vram_peak > VRAM_CEILING_GB:
        gates.append(f"VRAM_CEILING_EXCEEDED({vram_peak:.1f}GB)")
    # Strategic gate: treatment written but zero dialogue
    if has_treatment and dialogue == 0:
        gates.append("ZERO_DIALOGUE")

    return {
        "phase_id": phase["id"],
        "result": result,
        "error": error_msg,
        "duration_s": duration,
        "treatment": has_treatment,
        "dialogue": dialogue,
        "vram_peak_gb": vram_peak,
        "probes": probe_results,
        "gates_tripped": gates,
        "new_log_line_count": len(new_lines),
    }


def print_phase_report(report: dict) -> None:
    print_section(f"PHASE {report['phase_id']} RESULT")
    print(f"  Episode result:  {report['result']}", flush=True)
    print(f"  Duration:        {report['duration_s']}s", flush=True)
    print(f"  Treatment:       {'YES' if report['treatment'] else 'NO'}",
          flush=True)
    print(f"  Dialogue lines:  {report['dialogue']}", flush=True)
    print(f"  VRAM peak:       "
          f"{report['vram_peak_gb'] if report['vram_peak_gb'] is not None else 'unknown'} GB",
          flush=True)
    print(f"  New log lines:   {report['new_log_line_count']}", flush=True)
    if report["error"]:
        print(f"  Error:           {report['error'][:200]}", flush=True)

    print("", flush=True)
    print("  Feature probes:", flush=True)
    for r in report["probes"]:
        mark = "[PASS]" if r["verdict"] == "PASS" else \
               "[FAIL]" if r["verdict"] == "FAIL" else \
               "[OK]  " if r["verdict"] == "OK" else "[----]"
        req = "(required)" if r["required"] else "(info)"
        print(f"    {mark} {r['id']:<6} {r['name']:<40} "
              f"hits={r['hits']:<3} {req}", flush=True)

    if report["gates_tripped"]:
        print("", flush=True)
        print("  STRATEGIC GATES TRIPPED:", flush=True)
        for g in report["gates_tripped"]:
            print(f"    !! {g}", flush=True)


def append_phase_to_log(report: dict, phase: dict) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    probe_lines = "\n".join(
        f"  - [{r['verdict']}] {r['id']} {r['name']} hits={r['hits']}"
        for r in report["probes"]
    )
    gates = (", ".join(report["gates_tripped"])
             if report["gates_tripped"] else "none")
    entry = (
        f"\n### SUPERSOAKER {phase['id']} -- {stamp}\n"
        f"- **Result:** {report['result']}\n"
        f"- **Duration:** {report['duration_s']}s\n"
        f"- **Treatment:** {'YES' if report['treatment'] else 'NO'}\n"
        f"- **Dialogue:** {report['dialogue']}\n"
        f"- **VRAM peak:** {report['vram_peak_gb']} GB\n"
        f"- **Gates tripped:** {gates}\n"
        f"- **Probes:**\n{probe_lines}\n"
    )
    if report["error"]:
        entry += f"- **Error:** {report['error'][:300]}\n"
    append_log(entry)


def phase_decision(report: dict, phase: dict) -> tuple[bool, str]:
    """Returns (advance?, stop_reason_if_not)."""
    if report["result"] != "SUCCESS":
        return False, f"episode did not complete ({report['result']})"
    if report["gates_tripped"]:
        return False, (f"strategic gate tripped: "
                       f"{', '.join(report['gates_tripped'])}")
    failed_required = [r for r in report["probes"]
                       if r["required"] and not r["passed"]]
    if failed_required:
        names = ", ".join(f"{r['id']} ({r['name']})" for r in failed_required)
        return False, f"required probe(s) did not fire: {names}"
    return True, ""


# ---------------------------------------------------------------------------
# Sequencer
# ---------------------------------------------------------------------------
def preflight_comfyui() -> bool:
    try:
        resp = requests.get(f"{COMFYUI}/system_stats", timeout=5)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"  ERROR: ComfyUI not reachable at {COMFYUI} -- {e}", flush=True)
        print(f"  Start ComfyUI Desktop first (http://localhost:8000).",
              flush=True)
        return False


def sequence(phases: list[dict], workflow_path: str, auto_advance: bool) -> int:
    ensure_dirs()

    if not preflight_comfyui():
        return 2

    for i, phase in enumerate(phases):
        report = run_phase(phase, workflow_path)
        print_phase_report(report)
        append_phase_to_log(report, phase)

        advance, reason = phase_decision(report, phase)
        if advance:
            print("", flush=True)
            print(f"  >> Phase {phase['id']} PASSED.", flush=True)
            if not auto_advance:
                print(f"  >> --no-advance set. Stopping after this phase.",
                      flush=True)
                return 0
            if i < len(phases) - 1:
                print(f"  >> Advancing to phase "
                      f"{phases[i + 1]['id']}...", flush=True)
                time.sleep(5)  # breathing room
        else:
            print("", flush=True)
            print(f"  >> Phase {phase['id']} FAILED: {reason}", flush=True)
            write_handoff(
                reason=reason,
                phase=phase,
                detail=(
                    f"Episode result: {report['result']}\n"
                    f"Dialogue lines: {report['dialogue']}\n"
                    f"VRAM peak: {report['vram_peak_gb']} GB\n"
                    f"Gates: {report['gates_tripped']}\n"
                    f"Probes: {[(r['id'], r['verdict'], r['hits']) for r in report['probes']]}\n"
                    f"Error: {report['error'][:400]}\n"
                ),
            )
            print(f"  >> Handing off to Jeffrey. Fix, then re-run with: "
                  f"--only {phase['id']}", flush=True)
            return 1

    print_banner("SUPERSOAKER COMPLETE -- all phases PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Supersoaker: recursive P0 -> P1 -> P2 feature verifier"
    )
    parser.add_argument("--only", choices=list(PHASES_BY_ID.keys()),
                        help="Run a single phase only")
    parser.add_argument("--start-at", choices=list(PHASES_BY_ID.keys()),
                        help="Start at this phase and continue forward")
    parser.add_argument("--no-advance", action="store_true",
                        help="After a passing phase, stop instead of "
                             "advancing to the next")
    parser.add_argument("--workflow", default=WORKFLOW_PATH,
                        help="Workflow JSON path")
    args = parser.parse_args()

    if args.only:
        phases = [PHASES_BY_ID[args.only]]
    elif args.start_at:
        start_idx = next(i for i, p in enumerate(PHASES)
                         if p["id"] == args.start_at)
        phases = PHASES[start_idx:]
    else:
        phases = PHASES

    auto_advance = not args.no_advance and len(phases) > 1

    print_banner(f"SUPERSOAKER start {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Phases:     {[p['id'] for p in phases]}", flush=True)
    print(f"  Auto-advance: {auto_advance}", flush=True)
    print(f"  Workflow:   {args.workflow}", flush=True)
    print(f"  ComfyUI:    {COMFYUI}", flush=True)
    print(f"  Log:        {SUPERSOAKER_LOG}", flush=True)
    print("", flush=True)

    return sequence(phases, args.workflow, auto_advance)


if __name__ == "__main__":
    sys.exit(main())
