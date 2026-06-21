"""Stage-direction leak scan -- non-mutating precision gate.

Scans frozen episode ledgers and reports, per CHARACTER line, whether the bare
stage-direction DETECTOR fires (reroll candidate) and what the destructive FLOOR
would strip (`scrub_leading_stage_direction`). Mutates NOTHING -- this is the
precision gate for docs/2026-06-22-stage-direction-leak/: review the
`would_mutate=true` rows; the floor (BARE_STAGE_FLOOR_ACTIVE) is only enabled
once they show ~zero false positives.

Usage:
  python scripts/stage_direction_scan.py [ledger_dir] [--mutating-only]

ledger_dir defaults to the OTR episodes tree. Output is JSONL (one record per
flagged character line) to stdout + a summary to stderr. Pure offline; no server,
no torch.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_line_hygiene import (  # noqa: E402
    detect_leading_stage_business,
    scrub_leading_stage_direction,
)

_DEFAULT_DIR = r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes"


def scan_line(source_id, line):
    role = str(line.get("speaker_role") or "")
    if role != "character":
        return None
    raw = str(line.get("text") or "")
    if not raw:
        return None
    detect_hit, _hint = detect_leading_stage_business(raw)
    proposed = scrub_leading_stage_direction(raw)
    would_mutate = proposed != raw
    if not (detect_hit or would_mutate):
        return None
    reason = (
        "floor_strip" if would_mutate
        else "reroll_only (floor declined: long lead / guard)"
    )
    return {
        "source_id": source_id,
        "line_id": line.get("line_id"),
        "speaker_role": role,
        "raw_text": raw,
        "detect_hit": bool(detect_hit),
        "propose_hit": bool(would_mutate),
        "proposed_text": proposed if would_mutate else None,
        "guard_reasons": reason,
        "would_mutate": bool(would_mutate),
    }


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    mutating_only = "--mutating-only" in argv
    base = Path(args[0]) if args else Path(_DEFAULT_DIR)
    if not base.exists():
        print(f"ledger dir not found: {base}", file=sys.stderr)
        return 1

    ledgers = sorted(base.rglob("*_ledger.json"))
    n_files = n_lines = n_detect = n_mutate = 0
    for lp in ledgers:
        try:
            data = json.loads(lp.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"# skip {lp.name}: {type(exc).__name__}", file=sys.stderr)
            continue
        n_files += 1
        source_id = lp.parent.parent.name if lp.parent.name == "audio" else lp.stem
        for line in (data.get("lines") or []):
            rec = scan_line(source_id, line)
            if rec is None:
                continue
            n_lines += 1
            n_detect += int(rec["detect_hit"])
            n_mutate += int(rec["would_mutate"])
            if mutating_only and not rec["would_mutate"]:
                continue
            print(json.dumps(rec, ensure_ascii=False))

    print(
        f"\n[scan] ledgers={n_files} flagged_lines={n_lines} "
        f"detect_hits={n_detect} would_mutate={n_mutate}",
        file=sys.stderr,
    )
    print(
        "[scan] PRECISION GATE: review every would_mutate row above; enable "
        "BARE_STAGE_FLOOR_ACTIVE only if ~zero are false positives.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
