"""
scripts/lhm_poller.py  --  Day 13 LHM telemetry sampler (CLI wrapper)
======================================================================

Thin CLI around :mod:`otr_v2.visual.lhm_monitor` for use with Windows
Task Scheduler or a manual overnight run:

    python scripts/lhm_poller.py --interval 60 --duration 1200 \\
        --out logs/lhm_2026-04-17_overnight.ndjson

Writes one NDJSON line per poll plus a final summary JSON alongside
the log file (``<stem>.summary.json``) with peak/mean/min for every
tracked metric and the three ROADMAP Day 13 ceiling-breach flags
(VRAM, RAM, GPU temperature).

Pure stdlib; runs in the main ComfyUI venv without pulling torch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on sys.path when this script is executed
# directly (e.g. `python scripts/lhm_poller.py`).
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from otr_v2.visual.lhm_monitor import (  # noqa: E402  -- path bootstrap
    poll_loop,
    summarize,
    summarize_ndjson,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LibreHardwareMonitor poller for OTR Day 13 overnight dry runs."
    )
    p.add_argument(
        "--out", required=True,
        help="NDJSON output path (one JSON sample per line)",
    )
    p.add_argument(
        "--interval", type=float, default=60.0,
        help="Seconds between polls (default 60 s).",
    )
    p.add_argument(
        "--duration", type=float, default=1200.0,
        help="Total run duration in seconds (default 1200 s = 20 min).",
    )
    p.add_argument(
        "--max-samples", type=int, default=None,
        help="Optional hard cap on sample count.",
    )
    p.add_argument(
        "--summary", default=None,
        help="Explicit summary JSON path (default: <out>.summary.json).",
    )
    p.add_argument(
        "--summarise-only", action="store_true",
        help="Skip polling; just summarise an existing NDJSON log.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_path = Path(args.out)
    summary_path = Path(args.summary) if args.summary else (
        out_path.with_suffix(out_path.suffix + ".summary.json")
    )

    if args.summarise_only:
        summary = summarize_ndjson(out_path)
    else:
        samples = poll_loop(
            out_path,
            interval_s=args.interval,
            duration_s=args.duration,
            max_samples=args.max_samples,
        )
        summary = summarize(samples)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary.to_dict(), indent=2),
        encoding="utf-8",
    )

    # Print a one-line status for overnight-shell visibility.
    d = summary.to_dict()
    print(
        f"[lhm_poller] n={d['n_samples']} "
        f"unreach={d['n_unreachable']} "
        f"vram_peak={d['gpu_vram_used_gb'].get('peak')} "
        f"ram_peak={d['ram_used_gb'].get('peak')} "
        f"gpu_temp_peak={d['gpu_temp_c'].get('peak')} "
        f"breach_vram={d['vram_ceiling_breached']} "
        f"breach_ram={d['ram_ceiling_breached']} "
        f"breach_gpu_temp={d['gpu_temp_ceiling_breached']}"
    )

    # Exit non-zero if any Day 13 ceiling was breached so the overnight
    # task scheduler flags the run as failed without further scripting.
    any_breach = (
        d["vram_ceiling_breached"]
        or d["ram_ceiling_breached"]
        or d["gpu_temp_ceiling_breached"]
    )
    return 2 if any_breach else 0


if __name__ == "__main__":
    sys.exit(main())
