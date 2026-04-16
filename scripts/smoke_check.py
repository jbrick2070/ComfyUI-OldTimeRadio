"""
smoke_check.py
==============
Lightweight smoke / sanity checker for the OTR soak pipeline.
Called by the Cowork agent each cycle. Produces a structured report
the agent can read and act on.

Usage:
    python scripts/smoke_check.py
    python scripts/smoke_check.py --tail 20    # check last N soak runs only

AntiGravity: do not modify this file. The Cowork agent reads its output format.
"""

import os
import re
import sys
import json
import argparse
import datetime
from pathlib import Path

BASE_DIR    = Path(r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio")
OUTPUT_DIR  = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\old_time_radio")
SOAK_LOG    = BASE_DIR / "logs" / "soak_log.md"
AGENT_LOG   = BASE_DIR / "logs" / "agent_log.md"
OVERRIDES   = BASE_DIR / "scripts" / "watcher_overrides.json"
SCRIPTS_DIR = BASE_DIR / "scripts"

VRAM_CEILING   = 14.5
SHORT_DUR_SECS = 30
MIN_FILE_MB    = 5
STUCK_THRESHOLD = 5   # same title this many times = TITLE_STUCK

# ─────────────────────────────────────────────────────────────────────
# 1. SOURCE FILE HEALTH
# ─────────────────────────────────────────────────────────────────────
def check_source_files():
    """Verify all expected scripts exist and are non-empty."""
    expected = [
        "soak_operator.py",
        "treatment_scanner.py",
        "yoga_watchdog.py",
        "smoke_check.py",
    ]
    issues = []
    for fname in expected:
        path = SCRIPTS_DIR / fname
        if not path.exists():
            issues.append(f"MISSING_SCRIPT: {fname} not found")
        elif path.stat().st_size < 500:
            issues.append(f"TINY_SCRIPT: {fname} is suspiciously small ({path.stat().st_size} bytes)")
    return issues

# ─────────────────────────────────────────────────────────────────────
# 2. SOAK LOG ANALYSIS
# ─────────────────────────────────────────────────────────────────────
def parse_soak_log(tail=None):
    """
    Parse soak_log.md for run summaries and flag counts.
    Returns: list of run dicts, flag frequency dict.
    """
    if not SOAK_LOG.exists():
        return [], {"SOAK_LOG_MISSING": 1}

    text = SOAK_LOG.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if tail:
        # Rough heuristic: each run is ~15 lines; take last tail*20 lines
        lines = lines[-(tail * 20):]

    runs = []
    current_run = None
    for line in lines:
        # Detect run header e.g. "## Run 284"
        m = re.match(r"^#{1,3}\s+Run\s+(\d+)", line)
        if m:
            if current_run:
                runs.append(current_run)
            current_run = {
                "run_num": int(m.group(1)),
                "title": None,
                "flags": [],
                "duration_s": 0,
                "size_mb": 0,
                "vram_gb": 0,
            }
            continue
        if current_run is None:
            continue

        # Title line
        tm = re.search(r'Title\s*:\s*"([^"]+)"', line)
        if tm:
            current_run["title"] = tm.group(1)

        # Flag lines  >> FLAG_TYPE: ...
        fm = re.match(r"\s*>>\s+([A-Z_]+):", line)
        if fm:
            current_run["flags"].append(fm.group(1))

        # Duration
        dm = re.search(r"Duration\s*:\s*([\d.]+)\s*s", line)
        if dm:
            current_run["duration_s"] = float(dm.group(1))

        # File size
        sm = re.search(r"Size\s*:\s*([\d.]+)\s*MB", line)
        if sm:
            current_run["size_mb"] = float(sm.group(1))

        # VRAM
        vm = re.search(r"PEAKED AT ([\d.]+)GB", line)
        if vm:
            current_run["vram_gb"] = float(vm.group(1))

    if current_run:
        runs.append(current_run)

    # Tally flags across all parsed runs
    flag_counts = {}
    for run in runs:
        for flag in run["flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1

    return runs, flag_counts

# ─────────────────────────────────────────────────────────────────────
# 3. TITLE STUCK DETECTION
# ─────────────────────────────────────────────────────────────────────
def check_title_stuck(runs):
    """Return (is_stuck, stuck_title, consecutive_count)."""
    if not runs:
        return False, None, 0

    recent = [r for r in runs if r["title"]]
    if len(recent) < STUCK_THRESHOLD:
        return False, None, 0

    last_title = recent[-1]["title"]
    consecutive = 0
    for run in reversed(recent):
        if run["title"] == last_title:
            consecutive += 1
        else:
            break

    is_stuck = consecutive >= STUCK_THRESHOLD
    return is_stuck, last_title if is_stuck else None, consecutive

# ─────────────────────────────────────────────────────────────────────
# 4. WATCHER OVERRIDES SANITY
# ─────────────────────────────────────────────────────────────────────
def check_overrides():
    """Verify overrides JSON is valid and flags if soak is paused."""
    issues = []
    if not OVERRIDES.exists():
        return ["OVERRIDES_MISSING: watcher_overrides.json not found"]
    try:
        data = json.loads(OVERRIDES.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"OVERRIDES_CORRUPT: {e}"]

    if data.get("pause_soak") is True:
        issues.append("SOAK_PAUSED: pause_soak is true in overrides — soak is halted")

    return issues

# ─────────────────────────────────────────────────────────────────────
# 5. TREATMENT FILE SPOT CHECK
# ─────────────────────────────────────────────────────────────────────
def spot_check_treatments(n=5):
    """Quick check on the N most recent treatment files."""
    issues = []
    if not OUTPUT_DIR.exists():
        return ["OUTPUT_DIR_MISSING: output directory not found"]

    files = sorted(
        [f for f in OUTPUT_DIR.iterdir() if f.name.endswith("_treatment.txt")],
        key=lambda f: f.stat().st_mtime
    )[-n:]

    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        fname = path.name

        if "__NEEDS_LLM_CLOSING__" in text:
            issues.append(f"NEEDS_CLOSING: {fname}")
        if len(re.findall(r"\[SFX\]", text)) == 0:
            issues.append(f"NO_SFX: {fname}")
        m = re.search(r"Duration\s*:\s*([\d.]+)\s*s", text)
        if m and float(m.group(1)) < SHORT_DUR_SECS:
            issues.append(f"SHORT_DURATION: {fname} ({m.group(1)}s)")
        m = re.search(r"Size\s*:\s*([\d.]+)\s*MB", text)
        if m and float(m.group(1)) < MIN_FILE_MB:
            issues.append(f"TINY_FILE: {fname} ({m.group(1)}MB)")

    return issues

# ─────────────────────────────────────────────────────────────────────
# 6. AGENT LOG CONTINUITY
# ─────────────────────────────────────────────────────────────────────
def check_agent_log():
    """Warn if agent_log.md exists but hasn't been written to recently."""
    issues = []
    if AGENT_LOG.exists():
        mtime = AGENT_LOG.stat().st_mtime
        age_min = (datetime.datetime.now().timestamp() - mtime) / 60
        if age_min > 30:
            issues.append(f"AGENT_STALE: agent_log.md last updated {age_min:.0f} min ago — agent may have stopped")
    return issues

# ─────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────
def print_report(args):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*65}")
    print(f"  OTR SMOKE CHECK -- {now}")
    print(f"{'='*65}\n")

    all_issues = []

    # Source files
    src_issues = check_source_files()
    all_issues.extend(src_issues)

    # Soak log
    runs, flag_counts = parse_soak_log(tail=args.tail)
    is_stuck, stuck_title, stuck_count = check_title_stuck(runs)

    # Overrides
    ov_issues = check_overrides()
    all_issues.extend(ov_issues)

    # Treatment spot check
    tx_issues = spot_check_treatments()
    all_issues.extend(tx_issues)

    # Agent log continuity
    ag_issues = check_agent_log()
    all_issues.extend(ag_issues)

    # ── TITLE STUCK ────────────────────────────────────────────
    print("  TITLE ANALYSIS")
    print(f"  Runs parsed: {len(runs)}")
    if is_stuck:
        print(f"  !! TITLE_STUCK: '{stuck_title}' repeated {stuck_count}x consecutive")
        all_issues.append(f"TITLE_STUCK: '{stuck_title}' x{stuck_count}")
    elif runs:
        recent_titles = [r["title"] for r in runs[-5:] if r["title"]]
        print(f"  Recent titles: {recent_titles}")
        print(f"  Status: OK (no stuck title)")
    print()

    # ── FLAG FREQUENCY ─────────────────────────────────────────
    print("  FLAG FREQUENCY (recent soak runs)")
    if flag_counts:
        for tag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            bar = "#" * min(count, 20)
            print(f"    {tag:30s}  {count:3d}  {bar}")
    else:
        print("    (none)")
    print()

    # ── TREATMENT SPOT CHECK ───────────────────────────────────
    print("  TREATMENT SPOT CHECK (last 5 files)")
    if tx_issues:
        for i in tx_issues:
            print(f"    !! {i}")
    else:
        print("    OK")
    print()

    # ── SOURCE FILES ───────────────────────────────────────────
    print("  SOURCE FILE HEALTH")
    if src_issues:
        for i in src_issues:
            print(f"    !! {i}")
    else:
        print("    OK")
    print()

    # ── MISC ───────────────────────────────────────────────────
    if ov_issues or ag_issues:
        print("  OTHER ISSUES")
        for i in ov_issues + ag_issues:
            print(f"    !! {i}")
        print()

    # ── SUMMARY ────────────────────────────────────────────────
    print(f"{'='*65}")
    if all_issues:
        print(f"  RESULT: {len(all_issues)} issue(s) found")
        print()
        print("  PRIORITIZED ACTION LIST:")
        # Sort: TITLE_STUCK first, then others
        def priority(s):
            order = ["TITLE_STUCK", "SOAK_PAUSED", "SHORT_DURATION",
                     "ALL_SAME_GENDER", "SINGLE_LINE", "ZERO_DIALOGUE",
                     "EMPTY_SCRIPT", "NEEDS_CLOSING", "NO_SFX"]
            for i, tag in enumerate(order):
                if tag in s:
                    return i
            return 99
        for issue in sorted(all_issues, key=priority):
            print(f"    -> {issue}")
    else:
        print("  RESULT: CLEAN -- no issues found")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────
# CHATGPT CONSULTATION (optional, requires OPENAI_API_KEY env var)
# ─────────────────────────────────────────────────────────────────────
# Maps flag types to the source file + line-range most likely to contain
# the root cause.  The agent sends this context to ChatGPT for review.
_FLAG_CODE_CONTEXT = {
    "TITLE_STUCK":      ("nodes/story_orchestrator.py", "_extract_title_from_script_text"),
    "SHORT_DURATION":   ("nodes/story_orchestrator.py", "_compute_dialogue_lines"),
    "ALL_SAME_GENDER":  ("nodes/story_orchestrator.py", "_generate_character_profile"),
    "SINGLE_LINE_CHAR": ("nodes/story_orchestrator.py", "_critique_and_revise"),
    "ZERO_DIALOGUE":    ("nodes/story_orchestrator.py", "_parse_script"),
    "EMPTY_SCRIPT":     ("nodes/story_orchestrator.py", "_generate_with_llm"),
    "EMPTY_CAST":       ("nodes/story_orchestrator.py", "_randomize_character_names"),
    "TITLE_STUCK":      ("nodes/story_orchestrator.py", "_extract_title_from_script_text"),
}


def _extract_function_block(filepath, function_name, max_lines=60):
    """Extract a function body from a Python file for ChatGPT context."""
    try:
        full_path = BASE_DIR / filepath
        if not full_path.exists():
            return f"(File {filepath} not found)"
        text = full_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        # Find the def line
        start = None
        for i, line in enumerate(lines):
            if f"def {function_name}" in line:
                start = i
                break
        if start is None:
            return f"(Function {function_name} not found in {filepath})"
        # Capture up to max_lines or until next def at same indent
        indent = len(lines[start]) - len(lines[start].lstrip())
        end = min(start + max_lines, len(lines))
        for i in range(start + 1, end):
            if i < len(lines) and lines[i].strip():
                line_indent = len(lines[i]) - len(lines[i].lstrip())
                if line_indent <= indent and lines[i].strip().startswith("def "):
                    end = i
                    break
        return "\n".join(lines[start:end])
    except Exception as e:
        return f"(Error reading {filepath}: {e})"


def consult_chatgpt(top_flag):
    """Ask ChatGPT for a second opinion on the top-priority flag."""
    try:
        from second_opinion import ask_chatgpt
    except ImportError:
        # Try relative import path
        sys.path.insert(0, str(SCRIPTS_DIR))
        try:
            from second_opinion import ask_chatgpt
        except ImportError:
            return "CONSULT_SKIP: second_opinion.py not found"

    context = _FLAG_CODE_CONTEXT.get(top_flag)
    code_block = ""
    if context:
        filepath, func_name = context
        code_block = _extract_function_block(filepath, func_name)

    question = (
        f"The OTR soak pipeline is repeatedly flagging {top_flag}. "
        f"What is the most likely root cause and minimal fix?"
    )

    response = ask_chatgpt(top_flag, code_block, question)
    return response


def print_report(args):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*65}")
    print(f"  OTR SMOKE CHECK -- {now}")
    print(f"{'='*65}\n")

    all_issues = []

    # Source files
    src_issues = check_source_files()
    all_issues.extend(src_issues)

    # Soak log
    runs, flag_counts = parse_soak_log(tail=args.tail)
    is_stuck, stuck_title, stuck_count = check_title_stuck(runs)

    # Overrides
    ov_issues = check_overrides()
    all_issues.extend(ov_issues)

    # Treatment spot check
    tx_issues = spot_check_treatments()
    all_issues.extend(tx_issues)

    # Agent log continuity
    ag_issues = check_agent_log()
    all_issues.extend(ag_issues)

    # ── TITLE STUCK ────────────────────────────────────────────
    print("  TITLE ANALYSIS")
    print(f"  Runs parsed: {len(runs)}")
    if is_stuck:
        print(f"  !! TITLE_STUCK: '{stuck_title}' repeated {stuck_count}x consecutive")
        all_issues.append(f"TITLE_STUCK: '{stuck_title}' x{stuck_count}")
    elif runs:
        recent_titles = [r["title"] for r in runs[-5:] if r["title"]]
        print(f"  Recent titles: {recent_titles}")
        print(f"  Status: OK (no stuck title)")
    print()

    # ── FLAG FREQUENCY ─────────────────────────────────────────
    print("  FLAG FREQUENCY (recent soak runs)")
    if flag_counts:
        for tag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            bar = "#" * min(count, 20)
            print(f"    {tag:30s}  {count:3d}  {bar}")
    else:
        print("    (none)")
    print()

    # ── TREATMENT SPOT CHECK ───────────────────────────────────
    print("  TREATMENT SPOT CHECK (last 5 files)")
    if tx_issues:
        for i in tx_issues:
            print(f"    !! {i}")
    else:
        print("    OK")
    print()

    # ── SOURCE FILES ───────────────────────────────────────────
    print("  SOURCE FILE HEALTH")
    if src_issues:
        for i in src_issues:
            print(f"    !! {i}")
    else:
        print("    OK")
    print()

    # ── MISC ───────────────────────────────────────────────────
    if ov_issues or ag_issues:
        print("  OTHER ISSUES")
        for i in ov_issues + ag_issues:
            print(f"    !! {i}")
        print()

    # ── CHATGPT SECOND OPINION ─────────────────────────────────
    if getattr(args, "consult", False) and all_issues:
        # Sort to find top priority flag
        priority_order = ["TITLE_STUCK", "SOAK_PAUSED", "SHORT_DURATION",
                          "ALL_SAME_GENDER", "SINGLE_LINE", "ZERO_DIALOGUE",
                          "EMPTY_SCRIPT", "NEEDS_CLOSING", "NO_SFX"]
        top_flag = None
        for pf in priority_order:
            for issue in all_issues:
                if pf in issue:
                    top_flag = pf
                    break
            if top_flag:
                break
        if not top_flag and all_issues:
            top_flag = all_issues[0].split(":")[0]

        if top_flag:
            print("  CHATGPT SECOND OPINION")
            print(f"  Consulting on: {top_flag}")
            print(f"  {'-'*55}")
            response = consult_chatgpt(top_flag)
            # Indent response for readability
            for line in response.splitlines():
                print(f"    {line}")
            print()

    # ── SUMMARY ────────────────────────────────────────────────
    print(f"{'='*65}")
    if all_issues:
        print(f"  RESULT: {len(all_issues)} issue(s) found")
        print()
        print("  PRIORITIZED ACTION LIST:")
        # Sort: TITLE_STUCK first, then others
        def priority(s):
            order = ["TITLE_STUCK", "SOAK_PAUSED", "SHORT_DURATION",
                     "ALL_SAME_GENDER", "SINGLE_LINE", "ZERO_DIALOGUE",
                     "EMPTY_SCRIPT", "NEEDS_CLOSING", "NO_SFX"]
            for i, tag in enumerate(order):
                if tag in s:
                    return i
            return 99
        for issue in sorted(all_issues, key=priority):
            print(f"    -> {issue}")
    else:
        print("  RESULT: CLEAN -- no issues found")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OTR pipeline smoke/sanity checker")
    parser.add_argument("--tail", type=int, default=None,
                        help="Check only last N soak runs (default: all)")
    parser.add_argument("--consult", action="store_true",
                        help="Ask ChatGPT for second opinion on top-priority flag "
                             "(requires OPENAI_API_KEY env var)")
    args = parser.parse_args()
    print_report(args)
