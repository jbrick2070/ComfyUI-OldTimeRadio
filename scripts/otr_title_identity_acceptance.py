"""Acceptance for the title/identity family -- read the receipt, not the video.

THE BAR, in the operator's words: *a real story title on the card, and the
announcer naming the work it actually performed.* This turns that into a
deterministic read over the ledgers a bank-gate run produced, so a window does
not have to watch four episodes to know whether the fixes held.

WHAT IT CHECKS, per episode:

  1. TITLE IS NOT A HARNESS LABEL (PBUG-20260817-05). `title_source == "user"`
     on a headless run is a contradiction on its face -- nobody typed anything.
  2. THE WORK FRAME FIRED (PBUG-20260817-04). On an ADAPTATION lane
     (`ADAPTATION_SOURCE_KINDS`), `meta.announcer_work_frame` must be stamped
     and the intro row must carry `announcer_work_frame_rendered`. **This is
     the check that catches the fix failing SAFELY**: the splice is wrapped so
     it can never kill a render, which also means a scope error makes it a
     silent no-op with green tests. The stamp is the difference between
     working and merely not crashing.
  3. THE ANNOUNCER NAMES THE REAL WORK. The intro must contain the ledger's
     own `work_title`, and must NOT be a third string.
  4. THE SPAN IS PROTECTED. The intro row must carry
     `protected_fact_component`, or the clean stage may rewrite it
     (PBUG-20260815-01 deleted coda attributions on 9 of 14 voiced rows).
  5. NON-ADAPTATION LANES ARE UNTOUCHED. media_archive holds a PUBLICATION in
     `work_title`; a frame there would announce a work that was never
     performed, which is worse than the defect being fixed.

REPORTS. It never fails a render -- everything here has already happened.

Usage:
    python scripts/otr_title_identity_acceptance.py --since-minutes 240
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nodes._otr_paths import (  # noqa: E402  (the one OTR path authority)
    is_reserved_episode_entry,
    otr_episodes_root,
)
from nodes._otr_source_identity import (  # noqa: E402
    ADAPTATION_SOURCE_KINDS,
    identity_from_meta,
)

WORK_FRAME_FLAG = "announcer_work_frame_rendered"
PROTECTED_FLAG = "protected_fact_component"


def _intro_row(led: dict):
    for row in led.get("lines") or []:
        if isinstance(row, dict) and str(
                row.get("speaker") or "").upper() == "ANNOUNCER":
            return row
    return None


def audit_ledger(path: pathlib.Path) -> dict:
    try:
        led = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"episode": path.parent.parent.name,
                "checks": [("readable", False, f"{type(exc).__name__}")]}
    meta = led.get("meta") if isinstance(led, dict) else None
    if not isinstance(meta, dict):
        return {"episode": path.parent.parent.name,
                "checks": [("has_meta", False, "no meta mapping")]}

    identity = identity_from_meta(meta)
    is_adaptation = identity.source_kind in ADAPTATION_SOURCE_KINDS
    work = str(identity.work_title or "") if is_adaptation else ""
    row = _intro_row(led)
    intro = str((row or {}).get("text") or "")
    flags = [str(f) for f in ((row or {}).get("compose_flags") or [])]
    title_source = str(meta.get("title_source") or "")
    checks = []

    checks.append((
        "title is not a harness label",
        title_source != "user",
        f"title_source={title_source!r} title={meta.get('episode_title')!r}",
    ))

    if is_adaptation:
        checks.append((
            "work frame stamped (fix actually fired)",
            bool(meta.get("announcer_work_frame")),
            f"announcer_work_frame={meta.get('announcer_work_frame')!r}",
        ))
        checks.append((
            "intro row flagged rendered",
            WORK_FRAME_FLAG in flags,
            f"flags={flags}",
        ))
        checks.append((
            "announcer names the real work",
            bool(work) and work.lower() in intro.lower(),
            f"work={work!r} intro={intro[:110]!r}",
        ))
        checks.append((
            "spliced span is protected from the clean stage",
            PROTECTED_FLAG in flags,
            f"flags={flags}",
        ))
    else:
        checks.append((
            "non-adaptation lane got NO work frame",
            not meta.get("announcer_work_frame"),
            f"source_kind={identity.source_kind!r} "
            f"raw_work_title={identity.work_title!r}",
        ))

    return {"episode": path.parent.parent.name,
            "bank": str(meta.get("source_bank")),
            "kind": identity.source_kind,
            "title": meta.get("episode_title"),
            "checks": checks}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since-minutes", type=int, default=240)
    args = ap.parse_args(argv)

    cutoff = (datetime.datetime.now()
              - datetime.timedelta(minutes=args.since_minutes)).timestamp()
    root = otr_episodes_root()
    found = []
    for entry in sorted(root.iterdir()) if root.is_dir() else []:
        if not entry.is_dir() or is_reserved_episode_entry(entry.name):
            continue
        audio = entry / "audio"
        if not audio.is_dir():
            continue
        for led in audio.glob("*_ledger.json"):
            if led.stat().st_mtime >= cutoff:
                found.append(led)

    if not found:
        print(f"[acceptance] no ledgers in the last {args.since_minutes} min")
        return 0

    total_fail = 0
    for path in sorted(found, key=lambda p: p.stat().st_mtime):
        row = audit_ledger(path)
        fails = [c for c in row["checks"] if not c[1]]
        total_fail += len(fails)
        mark = "PASS" if not fails else "FAIL"
        print(f"\n[{mark}] {row['episode'][:60]}")
        print(f"       bank={row.get('bank')} kind={row.get('kind')!r} "
              f"title={row.get('title')!r}")
        for name, ok, detail in row["checks"]:
            print(f"       {'ok ' if ok else 'BAD'} {name}: {detail[:120]}")

    print(f"\n[acceptance] {len(found)} episode(s), {total_fail} failed check(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
