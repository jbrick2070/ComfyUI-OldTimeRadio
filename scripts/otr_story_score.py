"""Quick story-quality FINDER over frozen ledgers (operator request 2026-08-16).

WHAT THIS IS. A read-only ranking so the operator can find the good episodes
in a ~1,900-ledger corpus. It scores STRUCTURE the way a listener meets it:
the announcer frame, the music frame, whether every cast member actually
speaks, whether one voice hogs the drama, how much of the episode is drama
rather than announcer, and (on the news lanes) whether the coda that names
the source is present.

WHAT THIS IS NOT (THE LAW, and the 2026-08-04 quality freeze). A gate. The
score is telemetry: it may never reject, reroll, retire, replace or block an
episode, and nothing in the render path imports this file. Length is
REPORTED but carries zero weight -- word count may never judge an episode.
This does not reopen quality chasing: it rewrites nothing and ranks what
already shipped.

TWO LEDGER ERAS, MEASURED (QA pass 2026-08-16, all 1,692 ledgers): the
modern shape carries ``char_id`` on 100% of character lines; the only other
shape is the ancient May-era one with NO ``speaker_role`` at all, which
scores 0 ("no voiced lines") before any join runs. There is no
speaker-name-only era in the live corpus -- the name fallback in the join
below is belt-and-braces for future shapes, exercised by no real ledger
today. (The operator's detector warning stands regardless: cast joins never
name-match a line when a char_id exists.)

The ``--judge`` pass (one local-LLM "keep listening?" call per SHORTLISTED
episode -- judge the candidates, not the corpus) is deliberately a separate
follow-up: the GPU belongs to renders when this v1 runs. See the plan's
queue item; this file reserves the flag and refuses politely.

Usage:
  python scripts/otr_story_score.py                    # default corpus root
  python scripts/otr_story_score.py --root <episodes> --out <dir> --top 50
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field

# Repo root on sys.path at IMPORT time, not main()-gated: score_ledger()
# imports the bank-variant helper lazily, and a pytest fixture or sibling
# script that imports this module without running main() must not trip a
# ModuleNotFoundError (QA finding 2, reproduced).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DEFAULT_ROOT = r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes"
TS_RE = re.compile(r"_(\d{8})_(\d{6})$")

#: Banks whose close is a NEWS coda that names the source; only these are
#: scored on coda presence. The fidelity/invention lanes close differently
#: and are not penalized for lacking a news coda.
NEWS_COD_A_BANKS = {"scifi_news", "scifi_news_pro", "science_news",
                    "media_archive"}

#: Weights, summing to 100. Structure and usage only -- length is
#: deliberately unweighted (THE LAW).
W_FRAME = 20        # announcer opens AND closes the episode
W_MUSIC = 10        # opening and closing music both present
W_ALL_SPEAK = 25    # every non-announcer cast row has a voiced line
W_BALANCE = 20      # the drama is not a monologue
W_DIALOGUE = 15     # character share of spoken words
W_CODA = 10         # news lanes: the coda exists (else redistributed)


def _norm_name(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


@dataclass
class EpisodeScore:
    path: str
    episode: str
    bank: str
    title: str
    date: str
    era: str                 # char_id | speaker_name (the line-join key used)
    n_cast: int = 0
    n_speaking: int = 0
    spoken_words: int = 0
    character_words: int = 0
    minutes: float = 0.0
    frame_ok: bool = False
    music_ok: bool = False
    all_speak: bool = False
    balance: float = 0.0     # 0..1, 1 = perfectly shared drama
    dialogue_share: float = 0.0
    coda_applicable: bool = False
    coda_ok: bool = False
    notes: list = field(default_factory=list)
    score: float = 0.0


def _voiced(line: dict) -> bool:
    return (str(line.get("speaker_role") or "") in ("announcer", "character")
            and str(line.get("text") or "").strip() != ""
            and not line.get("skip"))


def score_ledger(path: str) -> EpisodeScore | None:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or "cast" not in data:
        return None
    meta = data.get("meta") or {}
    lines = [l for l in (data.get("lines") or []) if isinstance(l, dict)]
    cast = [r for r in (data.get("cast") or []) if isinstance(r, dict)]

    ep_dir = os.path.basename(os.path.dirname(path))
    if ep_dir == "audio":
        ep_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
    m = TS_RE.search(ep_dir)

    row = EpisodeScore(
        path=path,
        episode=ep_dir,
        bank=str(meta.get("source_bank") or "<pre-bank>"),
        title=str(meta.get("episode_title") or ""),
        date=(f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}"
              if m else ""),
        era="char_id" if any("char_id" in l for l in lines)
        else "speaker_name",
    )

    voiced = [l for l in lines if _voiced(l)]
    if not voiced:
        row.notes.append("no voiced lines")
        return row

    # --- the announcer frame -----------------------------------------
    first_role = str(voiced[0].get("speaker_role") or "")
    last_role = str(voiced[-1].get("speaker_role") or "")
    row.frame_ok = first_role == "announcer" and last_role == "announcer"

    # --- the music frame ---------------------------------------------
    roles = {str(l.get("speaker_role") or "") for l in lines}
    music_meta = data.get("music") or []
    row.music_ok = (
        ("music_open" in roles and "music_close" in roles)
        or (isinstance(music_meta, list) and len(music_meta) >= 2)
    )

    # --- cast usage ----------------------------------------------------
    story_cast = [
        r for r in cast
        if _norm_name(r.get("name")) != "announcer"
        and str(r.get("char_id") or "").casefold() != "announcer"
    ]
    row.n_cast = len(story_cast)
    words_by_key: dict[str, int] = {}
    for l in voiced:
        if str(l.get("speaker_role")) != "character":
            continue
        wc = int(l.get("word_count") or 0)
        if wc <= 0:
            # A zero-word bucket would inflate len(words_by_key) and skew
            # the balance denominator (QA finding 4; one such line exists
            # corpus-wide). A character's SPEAKING status is decided by
            # words, so a zero-word line contributes nothing either way.
            continue
        key = (str(l.get("char_id"))
               if "char_id" in l else _norm_name(l.get("speaker")))
        words_by_key[key] = words_by_key.get(key, 0) + wc
    speaking = 0
    for r in story_cast:
        keys = {str(r.get("char_id")), _norm_name(r.get("name"))}
        if any(words_by_key.get(k, 0) > 0 for k in keys):
            speaking += 1
    row.n_speaking = speaking
    row.all_speak = bool(story_cast) and speaking == len(story_cast)
    if story_cast and not row.all_speak:
        row.notes.append(f"silent cast rows: {len(story_cast) - speaking}")

    # --- balance: is the drama shared? --------------------------------
    char_words = sum(words_by_key.values())
    if char_words > 0 and len(words_by_key) >= 2:
        top_share = max(words_by_key.values()) / char_words
        even = 1.0 / len(words_by_key)
        # 1.0 when perfectly even; 0.0 when one voice owns everything.
        row.balance = max(0.0, (1.0 - top_share) / (1.0 - even))
    elif len(words_by_key) == 1:
        row.balance = 0.0
        row.notes.append("single-voice drama")

    # --- words / dialogue share / runtime ------------------------------
    row.spoken_words = sum(int(l.get("word_count") or 0) for l in voiced)
    row.character_words = char_words
    row.dialogue_share = (char_words / row.spoken_words
                          if row.spoken_words else 0.0)
    try:
        ends = [float(l.get("start_s") or 0) + float(l.get("dur_s") or 0)
                for l in lines]
        row.minutes = round(max(ends) / 60.0, 1) if ends else 0.0
    except (TypeError, ValueError):
        row.minutes = 0.0

    # --- the coda, where one is owed -----------------------------------
    from nodes._otr_bank_variants import base_source_bank_id
    row.coda_applicable = (
        base_source_bank_id(row.bank) in NEWS_COD_A_BANKS
    )
    if row.coda_applicable:
        flag_hit = any(
            any("coda" in str(f) for f in (l.get("compose_flags") or []))
            for l in lines
        )
        # QA finding 1, corpus-proven: 92 of 339 news-lane ledgers carry a
        # genuine closing news read whose frozen line lost the diagnostic
        # flag. The coda on these lanes is BY CONSTRUCTION the closing
        # announcer beat (PBUG-20260814-02: Python appends it, last thing
        # the listener hears), so when no flag survived, a substantive
        # closing announcer read counts. 15 words separates a real
        # source-naming read from a bare "we've lost our signal" sign-off.
        structural_hit = (
            str(voiced[-1].get("speaker_role") or "") == "announcer"
            and int(voiced[-1].get("word_count") or 0) >= 15
        )
        row.coda_ok = flag_hit or structural_hit

    # --- composite ------------------------------------------------------
    score = 0.0
    score += W_FRAME * (1.0 if row.frame_ok else 0.0)
    score += W_MUSIC * (1.0 if row.music_ok else 0.0)
    score += W_ALL_SPEAK * (1.0 if row.all_speak else 0.0)
    score += W_BALANCE * row.balance
    score += W_DIALOGUE * min(1.0, row.dialogue_share / 0.75)
    if row.coda_applicable:
        score += W_CODA * (1.0 if row.coda_ok else 0.0)
    else:
        # Redistribute the coda weight so non-news banks score on the
        # same 0-100 scale instead of topping out at 90.
        score *= 100.0 / (100.0 - W_CODA)
    row.score = round(min(100.0, score), 1)
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=None,
                    help="output dir (default <root>/_shared/story_scores)")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--judge", action="store_true",
                    help="reserved for the local-LLM pass; not in v1")
    args = ap.parse_args(argv)
    if args.judge:
        print("--judge is the follow-up pass (local 12B over the shortlist, "
              "GPU-idle only). v1 is the deterministic finder.")
        return 2

    paths = glob.glob(os.path.join(args.root, "**", "*_ledger.json"),
                      recursive=True)
    rows: list[EpisodeScore] = []
    skipped = 0
    for path in sorted(paths):
        row = score_ledger(path)
        if row is None:
            skipped += 1
        else:
            rows.append(row)
    rows.sort(key=lambda r: (-r.score, r.episode))

    # `state` is one of the three sanctioned `_shared` subtrees (the output
    # tree contract's system tier: cache / tmp / state) -- a durable report
    # belongs there, not in an invented fourth sibling (QA finding 5).
    out_dir = args.out or os.path.join(
        args.root, "_shared", "state", "story_scores")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "story_scores.csv")
    fields = ["score", "episode", "bank", "date", "title", "minutes",
              "spoken_words", "n_cast", "n_speaking", "frame_ok", "music_ok",
              "all_speak", "balance", "dialogue_share", "coda_applicable",
              "coda_ok", "era", "notes", "path"]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for r in rows:
            writer.writerow([
                r.score, r.episode, r.bank, r.date, r.title, r.minutes,
                r.spoken_words, r.n_cast, r.n_speaking, r.frame_ok,
                r.music_ok, r.all_speak, round(r.balance, 3),
                round(r.dialogue_share, 3), r.coda_applicable, r.coda_ok,
                r.era, "; ".join(r.notes), r.path,
            ])

    top_path = os.path.join(out_dir, "story_scores_top.md")
    with open(top_path, "w", encoding="utf-8") as fh:
        fh.write("# Story finder -- top %d of %d scored episodes\n\n"
                 % (min(args.top, len(rows)), len(rows)))
        fh.write("Deterministic structure score (0-100); telemetry only, "
                 "never a gate. Length reported, unweighted.\n\n")
        fh.write("| score | episode | bank | date | min | words | cast |\n")
        fh.write("|---:|---|---|---|---:|---:|---:|\n")
        for r in rows[:args.top]:
            fh.write("| %.1f | %s | %s | %s | %.1f | %d | %d |\n" % (
                r.score, r.episode, r.bank, r.date, r.minutes,
                r.spoken_words, r.n_cast))

    eras = {}
    for r in rows:
        eras[r.era] = eras.get(r.era, 0) + 1
    print(f"scored {len(rows)} ledgers ({skipped} skipped); eras: {eras}")
    print(f"csv:  {csv_path}")
    print(f"top:  {top_path}")
    if rows:
        best = rows[0]
        print(f"best: {best.score:.1f}  {best.episode}  ({best.bank})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
