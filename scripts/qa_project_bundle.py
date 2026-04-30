"""
qa_project_bundle.py  --  Build a single QA prompt bundle for round-robin
==========================================================================

Purpose
-------
Assemble the current state of OTR -- ROADMAP, BUG_LOG tail, node inventory,
workflow JSON shape, recent commits -- into one large markdown prompt
suitable for piping into _consult_round_robin.py.  Internal QA only,
NOT shipped with episodes.

Usage
-----
Print bundle to stdout:
    python scripts/qa_project_bundle.py

Pipe straight into round-robin:
    python scripts/qa_project_bundle.py > qa_prompt.md
    python scripts/_consult_round_robin.py --question qa_prompt.md ^
        --topic project-qa-2026-04-30

Token budget
------------
Aim for <= ~24k chars total so it fits comfortably in every model's
context window without truncation, including NVIDIA's free-tier ceiling.
Big files (workflow JSON, full BUG_LOG) get summarized, not pasted.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROADMAP = ROOT / "ROADMAP.md"
BUG_LOG = ROOT / "docs" / "BUG_LOG.md"
WORKFLOW_FULL = ROOT / "workflows" / "otr_scifi_16gb_full.json"
WORKFLOW_TEST = ROOT / "workflows" / "otr_scifi_16gb_TEST.json"
INIT_PY = ROOT / "__init__.py"


def _safe_read(path: Path, fallback: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"({fallback or path.name}: {type(e).__name__}: {e})"


def _tail_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def _head_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    return "\n".join(lines[:n])


def _git_log(n: int = 20) -> str:
    try:
        out = subprocess.check_output(
            ["git", "log", "--oneline", f"-{n}"],
            cwd=str(ROOT),
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        return out.decode("utf-8", errors="replace").strip()
    except Exception as e:
        return f"(git log failed: {type(e).__name__}: {e})"


def _git_status() -> str:
    try:
        out = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=str(ROOT),
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        return out.decode("utf-8", errors="replace").strip() or "(clean)"
    except Exception as e:
        return f"(git status failed: {type(e).__name__}: {e})"


def _git_branch() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        return out.decode("utf-8", errors="replace").strip()
    except Exception as e:
        return f"(unknown: {e})"


def _node_inventory() -> str:
    """Walk nodes/ + visual/ + otr_v2/ + subprocess_runners/ for *.py and
    list NODE_CLASS_MAPPINGS-style class names plus file sizes."""
    sections = []
    for sub in ("nodes", "visual", "otr_v2", "subprocess_runners"):
        d = ROOT / sub
        if not d.is_dir():
            continue
        rows = []
        for p in sorted(d.rglob("*.py")):
            rel = p.relative_to(ROOT).as_posix()
            try:
                size_kb = p.stat().st_size // 1024
                # Find class definitions (best-effort, no AST cost).
                class_lines = [
                    line.strip()
                    for line in p.read_text(encoding="utf-8", errors="replace").splitlines()
                    if line.startswith("class ")
                ]
                class_summary = ", ".join(
                    cl.split("class ", 1)[1].split("(", 1)[0].split(":", 1)[0].strip()
                    for cl in class_lines
                ) or "-"
                rows.append(f"- `{rel}` ({size_kb} KB): {class_summary}")
            except Exception as e:
                rows.append(f"- `{rel}`: (read failed: {e})")
        if rows:
            sections.append(f"### {sub}/\n\n" + "\n".join(rows))
    return "\n\n".join(sections) or "(no node inventory found)"


def _workflow_summary(path: Path) -> str:
    if not path.exists():
        return f"(missing: {path.name})"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return f"({path.name} parse failed: {e})"
    nodes = data.get("nodes", [])
    rows = []
    for n in nodes:
        nid = n.get("id", "?")
        ntype = n.get("type", "?")
        title = n.get("title") or ""
        widget_count = len(n.get("widgets_values", []) or [])
        rows.append(
            f"  - id={nid:>3} type={ntype} widgets={widget_count}"
            + (f" title={title!r}" if title else "")
        )
    rows.sort()
    links_count = len(data.get("links", []))
    return (
        f"**{path.name}**: {len(nodes)} nodes, {links_count} links\n\n"
        + "\n".join(rows[:80])
        + ("\n  ... (truncated)" if len(rows) > 80 else "")
    )


def _init_classes() -> str:
    """List NODE_CLASS_MAPPINGS keys from top-level __init__.py."""
    if not INIT_PY.exists():
        return "(no __init__.py)"
    text = _safe_read(INIT_PY)
    in_block = False
    keys = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("NODE_CLASS_MAPPINGS"):
            in_block = True
            continue
        if in_block:
            if s.startswith("}"):
                break
            if s.startswith('"') and ':' in s:
                key = s.split(':', 1)[0].strip().strip('",')
                keys.append(key)
    return f"NODE_CLASS_MAPPINGS keys ({len(keys)}):\n" + "\n".join(f"- {k}" for k in keys)


HEADER = """# OTR Project + ROADMAP QA -- Round-Robin Consultation

You are doing an **internal QA pass** on the ComfyUI-OldTimeRadio project,
v2.0-alpha branch.  This is NOT for shipping.  Your job is to:

1. Spot priority misalignment in ROADMAP vs current open bugs.
2. Flag any node, workflow, or design choice that looks brittle, redundant,
   or out of sync with the rest of the stack.
3. Sanity-check the last few commits -- did the recent fixes actually close
   the bugs they claim to close?
4. Recommend the **next 3-5 things to do before the next episode test run**,
   in priority order, with one-sentence justifications.

Constraints to respect (do not relitigate):

- Single RTX 5080 Laptop, 16 GB VRAM, Blackwell sm_120, Windows.
- 100% local for the shipped pipeline.  No cloud weights.
- VRAM ceiling 14.5 GB.  No weight streaming, no Flash Attention chasing,
  no quantization heroics.
- Audio output must remain byte-identical to v1.5 baseline (rule C7).
- Default models: writer = Mistral-Nemo 12B (LOCAL); critic + reviser = a
  different family for two-immune-system gate.
- Default audio path: HuMo native audio + master_mix slices, ffmpeg concat.
- "Do not chase VRAM/DRAM dragons" -- VRAM work is alarm plumbing only.

Be candid.  Flag uncertainty.  Cite specific file paths and line numbers
when you can.  If a recent commit message claims a fix that the current
code does not actually contain, say so.

---
"""


def main() -> int:
    parts = [HEADER]

    parts.append(f"## Branch + git status\n\nBranch: `{_git_branch()}`\n")
    parts.append("### git status (short)\n\n```\n" + _git_status() + "\n```\n")
    parts.append("### Last 20 commits\n\n```\n" + _git_log(20) + "\n```\n")

    parts.append("## ROADMAP.md (full)\n\n```markdown\n" + _safe_read(ROADMAP) + "\n```\n")

    bug_log_text = _safe_read(BUG_LOG)
    parts.append(
        "## BUG_LOG.md -- last ~400 lines\n\n```markdown\n"
        + _tail_lines(bug_log_text, 400)
        + "\n```\n"
    )

    parts.append("## Node inventory\n\n" + _node_inventory() + "\n")
    parts.append("## __init__.py registrations\n\n" + _init_classes() + "\n")

    parts.append("## Workflow JSON shapes\n\n" + _workflow_summary(WORKFLOW_FULL) + "\n")
    parts.append(_workflow_summary(WORKFLOW_TEST) + "\n")

    bundle = "\n".join(parts)

    # Token-budget guard.  Soft cap; print a warning to stderr if exceeded.
    max_chars = 60_000
    if len(bundle) > max_chars:
        print(
            f"[qa-bundle] WARNING: bundle is {len(bundle)} chars "
            f"(soft cap {max_chars}) -- may exceed model context",
            file=sys.stderr,
        )

    sys.stdout.write(bundle)
    sys.stdout.flush()
    print(
        f"[qa-bundle] OK -- {len(bundle)} chars written to stdout",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
