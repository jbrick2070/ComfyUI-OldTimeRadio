"""otr_tree_doctor -- OH-4 migration + attic (operator-gated).

Brings an existing ``<output>/otr/`` tree into the output-tree contract
(docs/2026-06-11-output-tree-consolidation/OUTPUT_TREE_CONTRACT.md):

  otr/episodes/<id>/...   the asset of record
  otr/episodes/_shared/   {cache,tmp,state} system tiers
  otr/obs/                final deliverables, flat

Plan (per top-level entry found):

  stills/        LIVE cache -> MOVE into episodes/_shared/cache
                 (content-addressed collision policy: same name + same
                 size = drop the duplicate source; same name + different
                 size = QUARANTINE to the attic, never overwrite)
  state/         LIVE state -> MOVE into episodes/_shared/state (same
                 collision policy)
  tmp/           stale scratch -> ATTIC (the new tmp tier is
                 episodes/_shared/tmp; old scratch is debris)
  audio/ videos/ portraits/ script_gates/ blend_test/ _legacy_stills/
  _legacy_audio/ _legacy_portraits/ qa_waveforms/ qa_frames/ aship/
  aship_test/ _lane1/ and ANY other non-contract entry
                 dead debris -> ATTIC

The ATTIC is ``<output>/otr_attic_<timestamp>/`` -- OUTSIDE otr/, nothing
is deleted. The operator deletes the attic manually when satisfied
(operator law: the janitor's _shared/tmp sweep is the ONLY auto-delete).

SAFETY:
  * DRY-RUN by default -- prints the move table and exits. ``--live``
    performs the moves.
  * QUIESCENCE preflight: refuses to run live while a render is active
    (any running ffmpeg, or a ComfyUI server answering on :8000/:8188,
    or _shared/tmp & otr/tmp touched within --quiesce-s seconds).
    ``--force`` overrides (operator's own risk).

Usage (ComfyUI venv python, repo root):
  python scripts/otr_tree_doctor.py            # dry-run table
  python scripts/otr_tree_doctor.py --live     # operator said yes

UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)

from nodes import _otr_paths as P  # noqa: E402

#: Contract top level. _shared lives INSIDE episodes/.
KEEP = {"episodes", "obs"}
#: Live tiers that MIGRATE into _shared rather than attic.
MIGRATE_CACHE = {"stills"}
MIGRATE_STATE = {"state"}


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return "%.1f %s" % (n, unit) if unit != "B" else "%d B" % n
        n /= 1024.0
    return "%d B" % n


def tree_size(path: str) -> tuple[int, int]:
    files = total = 0
    for dirpath, _dirs, names in os.walk(path):
        for f in names:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
                files += 1
            except OSError:
                continue
    return files, total


def quiescent(quiesce_s: float) -> list[str]:
    """Empty list = quiet box; else the reasons it is NOT safe."""
    reasons = []
    # 1. any running ffmpeg = an active render/mux
    try:
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq ffmpeg.exe", "/FO", "CSV"],
            capture_output=True, text=True, timeout=15).stdout
        if "ffmpeg.exe" in out:
            reasons.append("ffmpeg.exe is running (active render/mux)")
    except Exception:  # noqa: BLE001 -- preflight best-effort
        pass
    # 2. a live ComfyUI server
    for port in (8000, 8188, 8199):
        try:
            import urllib.request
            urllib.request.urlopen(
                "http://127.0.0.1:%d/system_stats" % port, timeout=2)
            reasons.append("ComfyUI server answering on :%d" % port)
        except Exception:  # noqa: BLE001
            pass
    # 3. scratch tiers recently touched
    for scratch in (str(P.otr_shared_tmp_dir()),
                    str(P.comfy_output_dir() / "otr" / "tmp")):
        if not os.path.isdir(scratch):
            continue
        newest = 0.0
        for dirpath, _d, names in os.walk(scratch):
            for f in names:
                try:
                    newest = max(newest, os.path.getmtime(
                        os.path.join(dirpath, f)))
                except OSError:
                    continue
        if newest and (time.time() - newest) < quiesce_s:
            reasons.append("%s touched %.0fs ago (< quiesce %ds)"
                           % (scratch, time.time() - newest, quiesce_s))
    return reasons


def plan_moves(otr_root: str, attic: str):
    """(table_rows, actions). An action is (kind, src, dst):
    kind in {migrate-cache, migrate-state, attic}."""
    rows, actions = [], []
    cache_dst = str(P.otr_shared_cache_dir())
    state_dst = str(P.otr_state_dir())
    for name in sorted(os.listdir(otr_root)):
        src = os.path.join(otr_root, name)
        if name in KEEP:
            rows.append((name, "-", "-", "KEEP (contract)"))
            continue
        files, size = (tree_size(src) if os.path.isdir(src)
                       else (1, os.path.getsize(src)))
        if name in MIGRATE_CACHE and os.path.isdir(src):
            rows.append((name, str(files), human(size),
                         "MIGRATE -> episodes/_shared/cache"))
            actions.append(("migrate", src, cache_dst))
        elif name in MIGRATE_STATE and os.path.isdir(src):
            rows.append((name, str(files), human(size),
                         "MIGRATE -> episodes/_shared/state"))
            actions.append(("migrate", src, state_dst))
        else:
            rows.append((name, str(files), human(size),
                         "ATTIC -> %s" % os.path.basename(attic)))
            actions.append(("attic", src, os.path.join(attic, name)))
    return rows, actions


def do_migrate(src_dir: str, dst_dir: str, attic: str, live: bool) -> str:
    """Flat-merge src into dst with the content-addressed collision policy:
    same name + same size -> drop the source duplicate; same name +
    different size -> QUARANTINE the source into the attic. Returns a
    summary string."""
    os.makedirs(dst_dir, exist_ok=True)
    moved = dropped = quarantined = 0
    qdir = os.path.join(attic, "_quarantine", os.path.basename(src_dir))
    for entry in sorted(os.listdir(src_dir)):
        s = os.path.join(src_dir, entry)
        d = os.path.join(dst_dir, entry)
        if os.path.exists(d):
            same = (os.path.isfile(s) and os.path.isfile(d)
                    and os.path.getsize(s) == os.path.getsize(d))
            if same:
                dropped += 1
                if live:
                    os.remove(s)
            else:
                quarantined += 1
                if live:
                    os.makedirs(qdir, exist_ok=True)
                    shutil.move(s, os.path.join(qdir, entry))
        else:
            moved += 1
            if live:
                shutil.move(s, d)
    if live:
        try:
            os.rmdir(src_dir)          # only removes if now empty
        except OSError:
            pass
    return ("%d moved, %d duplicate-dropped, %d quarantined"
            % (moved, dropped, quarantined))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="perform the moves (default: dry-run table only)")
    ap.add_argument("--force", action="store_true",
                    help="skip the quiescence preflight (operator risk)")
    ap.add_argument("--quiesce-s", type=float, default=600.0)
    args = ap.parse_args()

    out_root = str(P.comfy_output_dir())
    otr_root = os.path.join(out_root, "otr")
    if not os.path.isdir(otr_root):
        print("no otr/ tree under %s -- nothing to do" % out_root)
        return 0
    attic = os.path.join(out_root,
                         "otr_attic_%s" % time.strftime("%Y%m%d_%H%M%S"))

    rows, actions = plan_moves(otr_root, attic)
    w = max(len(r[0]) for r in rows) + 2
    print("\nOH-4 MIGRATION PLAN  (output root: %s)" % out_root)
    print("attic: %s  (OUTSIDE otr/; nothing is deleted)\n" % attic)
    print("%-*s %10s %12s  %s" % (w, "entry", "files", "size", "action"))
    print("-" * (w + 50))
    for name, files, size, action in rows:
        print("%-*s %10s %12s  %s" % (w, name, files, size, action))
    n_moves = len(actions)
    if not args.live:
        print("\nDRY-RUN ONLY (%d entries would move). The operator gates "
              "the live run: python scripts/otr_tree_doctor.py --live"
              % n_moves)
        return 0

    if not args.force:
        reasons = quiescent(args.quiesce_s)
        if reasons:
            print("\nQUIESCENCE PREFLIGHT FAILED -- refusing the live run:")
            for r in reasons:
                print("  - %s" % r)
            print("(stop renders/servers, wait for tmp to go idle, or "
                  "--force at your own risk)")
            return 2

    os.makedirs(attic, exist_ok=True)
    print("\nLIVE RUN:")
    for kind, src, dst in actions:
        if kind == "migrate":
            summary = do_migrate(src, dst, attic, live=True)
            print("  MIGRATED %-18s -> %s (%s)"
                  % (os.path.basename(src), dst, summary))
        else:
            shutil.move(src, dst)
            print("  ATTIC    %-18s -> %s" % (os.path.basename(src), dst))
    # Post-check: the contract top level.
    left = sorted(os.listdir(otr_root))
    bad = [e for e in left if e not in KEEP]
    print("\notr/ top level now: %r" % left)
    if bad:
        print("STILL NON-CONTRACT: %r (investigate)" % bad)
        return 1
    print("CONTRACT HOLDS: otr/ top level == episodes + obs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
