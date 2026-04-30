r"""qa_episode.py -- one-shot QA artifact generator.

Runs both qa_frames.py and qa_waveforms.py against the most recent (or
specified) ledger and prints a summary path so the next QA pass knows
where to look. Tolerates partial failure -- if frames render but
waveforms fail, you still get the frames.

Usage:
    C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe ^
        scripts\qa_episode.py --latest

    python scripts\qa_episode.py --ledger PATH\TO\<episode_id>_ledger.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def _run(script: Path, *extra: str) -> int:
    cmd = [PYTHON, str(script), *extra]
    print(f"\n>>> {' '.join(cmd)}\n")
    try:
        return subprocess.run(cmd, check=False).returncode
    except KeyboardInterrupt:
        return 130


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ledger", type=Path)
    g.add_argument("--latest", action="store_true")
    ap.add_argument("--frames-only", action="store_true",
                    help="Skip qa_waveforms (faster, no matplotlib needed)")
    ap.add_argument("--waveforms-only", action="store_true",
                    help="Skip qa_frames (no ffmpeg needed)")
    args = ap.parse_args()

    ledger_args = (
        ["--ledger", str(args.ledger)] if args.ledger else ["--latest"]
    )

    rc_frames = 0
    rc_waves = 0

    if not args.waveforms_only:
        rc_frames = _run(REPO_ROOT / "scripts" / "qa_frames.py", *ledger_args)
    if not args.frames_only:
        rc_waves = _run(REPO_ROOT / "scripts" / "qa_waveforms.py", *ledger_args)

    print("\n--- QA SUMMARY ---")
    print(f"qa_frames    rc={rc_frames}")
    print(f"qa_waveforms rc={rc_waves}")
    if rc_frames == 0 and rc_waves == 0:
        print("OK")
        return 0
    return 1 if (rc_frames != 0 and rc_waves != 0) else 0


if __name__ == "__main__":
    sys.exit(main())
