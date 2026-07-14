"""Per-leg structural metrics for the 420/720 bakeoff verdict.

The banks DO NOT WRITE THE SAME SHAPE OF STORY AT LENGTH, and the operator has
ruled that target words are not chaseable -- do not normalize, RECORD. At 720
words original_codex56sol writes a 40-beat conversation and scifi_sonnet writes
five speeches. Those are not the same show. Any ranking that ignores this is
scoring "who writes a 144-word monologue" against "who writes 40 short
exchanges" and calling the difference a bank effect.

So for every leg we record: prompt_id, title, asset, beats, voiced lines,
words/line, and meta.word_budget.actual_ratio (advisory, WARN-only -- a bank
that undershot to 0.71 is not competing at the requested length).

Run ONLY when the GPU is idle. Reading episode files mid-render can collide with
the per-beat ledger saves (WinError 5).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(r"C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio")
EPISODES = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes")
OBS = Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\obs")

LEG_RE = re.compile(
    r"END\s+(?P<bank>\S+)\s+(?P<words>\d+)w\s+::\s+EXIT=(?P<exit>\d+)\s+::"
    r".*?RESULT (?P<result>\w+).*?::\s+prompt=(?P<prompt>[0-9a-f-]*)\s+::\s+(?P<secs>\d+)s"
)


def read_legs() -> list[dict]:
    legs: list[dict] = []
    for summary in sorted((REPO / "tmp").glob("bake*_*w_summary.txt")):
        for line in summary.read_text(encoding="utf-8", errors="replace").splitlines():
            m = LEG_RE.search(line)
            if not m:
                continue
            legs.append({
                "sweep": summary.stem,
                "bank": m.group("bank"),
                "words": int(m.group("words")),
                "result": m.group("result"),
                "prompt_id": m.group("prompt"),
                "secs": int(m.group("secs")),
            })
    # A re-legged bank supersedes its earlier failure at the same length.
    latest: dict[tuple[str, int], dict] = {}
    for leg in legs:
        latest[(leg["bank"], leg["words"])] = leg
    return list(latest.values())


def ledger_index() -> list[dict]:
    """Every episode ledger on disk, with the fields the verdict needs."""
    rows: list[dict] = []
    if not EPISODES.is_dir():
        return rows
    for ep_dir in EPISODES.iterdir():
        if not ep_dir.is_dir():
            continue
        for lg in (ep_dir / "audio").glob("*_ledger.json"):
            if lg.name.startswith("pending_"):
                continue
            try:
                data = json.loads(lg.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:  # noqa: BLE001
                print(f"  !! unreadable ledger {lg}: {exc}", file=sys.stderr)
                continue
            rows.append({
                "episode_id": ep_dir.name,
                "ledger": lg,
                "mtime": lg.stat().st_mtime,
                "data": data,
            })
    rows.sort(key=lambda r: r["mtime"])
    return rows


def measure(data: dict) -> dict:
    meta = data.get("meta") or {}
    wb = meta.get("word_budget") or {}
    lines = [r for r in (data.get("lines") or []) if isinstance(r, dict)]

    voiced = [
        r for r in lines
        if (r.get("speaker_role") or "") == "character" and not r.get("skip")
    ]
    voiced_words = sum(len(str(r.get("text") or "").split()) for r in voiced)

    # Beats: prefer an explicit beat list, else the distinct beat ids on the lines.
    beats = data.get("beats")
    if isinstance(beats, list) and beats:
        beat_count = len(beats)
    else:
        ids = {r.get("beat_id") for r in lines if r.get("beat_id")}
        beat_count = len(ids)

    actual_words = wb.get("actual_voiced_words")
    if not isinstance(actual_words, int):
        actual_words = voiced_words

    return {
        "title": (meta.get("title") or data.get("title") or "").strip(),
        "source_bank": meta.get("source_bank") or "",
        "target_words": wb.get("target_words"),
        "actual_voiced_words": actual_words,
        "actual_ratio": wb.get("actual_ratio"),
        "actual_drift": wb.get("actual_drift"),
        "beats": beat_count,
        "voiced_lines": len(voiced),
        "total_lines": len(lines),
        "words_per_line": round(actual_words / len(voiced), 1) if voiced else 0.0,
    }


def find_asset(episode_id: str) -> tuple[str, int]:
    if not OBS.is_dir():
        return ("", 0)
    best = None
    for f in OBS.iterdir():
        if f.is_file() and episode_id in f.name:
            if best is None or f.stat().st_size > best.stat().st_size:
                best = f
    if best is None:
        return ("", 0)
    return (best.name, best.stat().st_size)


def main() -> None:
    legs = read_legs()
    ledgers = ledger_index()

    print(f"legs found: {len(legs)}   ledgers on disk: {len(ledgers)}\n")

    # Match each leg to its ledger by source_bank + target_words, newest first.
    rows = []
    used: set[Path] = set()
    for leg in sorted(legs, key=lambda l: (l["words"], l["bank"])):
        match = None
        for lr in reversed(ledgers):
            if lr["ledger"] in used:
                continue
            m = measure(lr["data"])
            if m["source_bank"] != leg["bank"]:
                continue
            if m["target_words"] not in (leg["words"], None):
                continue
            match = (lr, m)
            break
        if match is None:
            rows.append({**leg, "note": "NO LEDGER MATCHED"})
            continue
        lr, m = match
        used.add(lr["ledger"])
        asset, size = find_asset(lr["episode_id"])
        rows.append({**leg, **m, "episode_id": lr["episode_id"],
                     "asset": asset, "asset_mb": round(size / 1_048_576, 1)})

    hdr = (f"{'bank':<22} {'w':>4} {'res':<8} {'beats':>5} {'voiced':>6} "
           f"{'w/line':>6} {'ratio':>6} {'MB':>6}  title")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("note"):
            print(f"{r['bank']:<22} {r['words']:>4} {r['result']:<8} "
                  f"{'--':>5} {'--':>6} {'--':>6} {'--':>6} {'--':>6}  {r['note']}")
            continue
        print(f"{r['bank']:<22} {r['words']:>4} {r['result']:<8} "
              f"{r['beats']:>5} {r['voiced_lines']:>6} {r['words_per_line']:>6} "
              f"{str(r['actual_ratio']):>6} {r['asset_mb']:>6}  {r['title'][:44]}")

    out = REPO / "tmp" / "bakeoff_metrics.json"
    out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
