"""
fix_cast_descriptions.py -- one-shot cleanup for ledgers where the cast row's
character_description (renamed from `description` 2026-05-10) got polluted with
a separate period-name prefix from the LLM director.

Pattern detected:
    "PRIAM SMITHERS - Female, wry, 40s"   ->   "Female, wry, 40s"
    "MAURICE WELLS - Male, warm, 50s"     ->   "Male, warm, 50s"
    "NORA SATO - Male, intense, 50s"      ->   "Male, intense, 50s"

Leaves alone descriptions that don't match the "<ALLCAPS NAME> - <rest>" pattern,
and entries where character_description is null.

Reads either character_description (current) or the legacy description key
for back-compat with ledgers written before the 2026-05-10 rename. Always
writes character_description.

Usage:
    python scripts/fix_cast_descriptions.py <ledger_path> [--in-place | --out <path>]

Default behavior writes <ledger_path>.fixed.json next to the input. With --in-place
it overwrites the input after writing a .bak alongside.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# Match leading "ALLCAPS [SPACE OR ALLCAPS]+ - <rest>" but only when there are
# at least two uppercase tokens (so we don't strip "Male - tall and intense").
_PERIOD_NAME_PREFIX = re.compile(r"^[A-Z][A-Z]+(?:\s+[A-Z][A-Z]+)+\s*-\s*(.+)$")


def clean_description(desc: str | None) -> str | None:
    if not isinstance(desc, str) or not desc.strip():
        return desc
    m = _PERIOD_NAME_PREFIX.match(desc.strip())
    if m:
        return m.group(1).strip()
    return desc


def fix_ledger(in_path: Path, out_path: Path) -> tuple[int, list[tuple[str, str, str]]]:
    """Clean cast.character_description in place.

    Reads either character_description (post-2026-05-10 rename) or the legacy
    description key for back-compat. Always writes character_description.
    """
    data = json.loads(in_path.read_text(encoding="utf-8"))
    cast = data.get("cast", []) or []
    changes: list[tuple[str, str, str]] = []
    for row in cast:
        name = row.get("name", "?")
        old = row.get("character_description")
        if old is None:
            old = row.get("description")
        new = clean_description(old)
        if new != old:
            row["character_description"] = new
            # If the row only had the legacy key, drop it now that the new
            # one carries the cleaned value.
            if "description" in row and "character_description" not in row:
                row.pop("description", None)
            changes.append((name, str(old), str(new)))
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(cast), changes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger", type=Path, help="Path to ledger.json")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--in-place", action="store_true", help="Overwrite input (creates .bak)")
    grp.add_argument("--out", type=Path, help="Explicit output path")
    args = ap.parse_args()

    if not args.ledger.exists():
        print(f"FATAL: ledger not found: {args.ledger}", file=sys.stderr)
        return 2

    if args.in_place:
        bak = args.ledger.with_suffix(args.ledger.suffix + ".bak")
        shutil.copy2(args.ledger, bak)
        out_path = args.ledger
        print(f"[bak] {bak}")
    elif args.out:
        out_path = args.out
    else:
        out_path = args.ledger.with_suffix(".fixed.json")

    cast_n, changes = fix_ledger(args.ledger, out_path)
    print(f"[scan] cast rows: {cast_n}")
    if not changes:
        print("[ok] no period-name prefixes found; nothing to change")
    else:
        for name, old, new in changes:
            print(f"  {name:12s}  {old!r}")
            print(f"  {' ':12s}  -> {new!r}")
        print(f"[fixed] {len(changes)} description(s) cleaned")
    print(f"[wrote] {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
