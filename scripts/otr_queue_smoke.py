"""otr_queue_smoke.py -- the standing 30-word live-smoke harness.

Replaces the per-session throwaway drivers: ONE submit + babysit loop over
the canonical workflow (or any generated variant) via scripts/otr_api.py's
name-based patching. Expects the headless server already listening on
COMFYUI_URL (default http://127.0.0.1:8000); reset + boot per CLAUDE.md
section 4/5 happen OUTSIDE this script.

Usage:
    python scripts/otr_queue_smoke.py                       # canonical, 30w
    python scripts/otr_queue_smoke.py --words 30 --source-bank original
    python scripts/otr_queue_smoke.py --workflow workflows/variants/otr_cpu_floor.json

Exit 0 only on history SUCCESS. The obs/episode asset check (Test-Path)
stays the caller's job -- the file check, not VRAM, distinguishes a
finished-but-resident server from a real miss.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import otr_api  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workflow",
                    default=str(REPO / "workflows" / "otr_canonical.json"))
    ap.add_argument("--words", type=int, default=30)
    ap.add_argument("--source-bank", default="",
                    help="optional source_bank override (blank = saved value)")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    wf = otr_api.load_workflow(args.workflow)
    schemas = otr_api.fetch_schemas()
    wf = otr_api.normalize_stamp_widgets_for_live_schema(wf, schemas)
    otr_api.patch_creative(wf, 1, "target_words", args.words, schemas)
    if args.source_bank:
        otr_api.patch_creative(wf, 1, "source_bank", args.source_bank,
                               schemas)
    prompt = otr_api.workflow_to_api_prompt(wf, schemas)
    pid = otr_api.submit_prompt(prompt)
    print(f"[smoke] submitted {pid} ({args.words}w, "
          f"{Path(args.workflow).name})", flush=True)

    t0 = time.time()

    def tick(elapsed, _status):
        print(f"[smoke] t={int(elapsed)}s", flush=True)

    status, err = otr_api.poll_history(pid, timeout_s=args.timeout,
                                       poll_s=15, on_tick=tick)
    print(f"[smoke] RESULT {status} after {int(time.time() - t0)}s "
          f"{(err or '')[:400]}", flush=True)
    return 0 if status == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
