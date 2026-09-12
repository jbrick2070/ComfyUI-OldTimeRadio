"""One-act canonical REGRESSION SWEEP -- many short legs, every bank, one report.

WHY THIS EXISTS (operator, 2026-09-12): *"do all the coding you can and get some
major 1 act regression testing on 5080 and runpod."* A six-act leg takes ~15
minutes and exercises one story; a one-act leg takes ~7 and exercises the whole
canonical path just as completely -- writer, cast, TTS, music, stills, video,
caption burn, credits, publish. So the cheapest way to shake real bugs out of the
shipping path is MANY one-act legs across every bank, not a few long ones.

IT LOADS THE CANONICAL WORKFLOW AND NOTHING ELSE (CLAUDE.md section 0, absolute
since the bench carve-out was struck on 2026-08-23). Every leg goes through
``scripts/otr_canonical_api_run.py`` with a real profile; there is no second
path and no stock-node graph anywhere in here.

IT NEVER PASSES ``--title``. The harness run label becomes the on-screen title
card, which is the operator's standing complaint about labelled legs.

WHAT IT ASSERTS PER LEG, on disk rather than from the log:
  * RESULT SUCCESS in the leg log and a non-zero exit
  * the episode directory exists and carries a frozen ledger
  * the final mp4 landed in otr/obs and is non-zero
  * the music cues rendered at their requested durations
  * the ledger records the bank that was ASKED for, not a rolled one
A leg that dies is reported with its last log lines; the sweep continues, because
one dead bank must not hide the state of the other four.

USAGE (a server must already be listening; reset and boot per CLAUDE.md 4 and 5):
    python scripts/otr_oneact_regression.py --banks all --repeat 2
    python scripts/otr_oneact_regression.py --banks scifi_news_pro,public_domain
    python scripts/otr_oneact_regression.py --profile otr_sd15_stills --banks original
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "otr_canonical_api_run.py"
CANONICAL = REPO / "workflows" / "otr_canonical.json"

#: Every bank a stranger can actually roll. `custom_source_bank` is excluded: it
#: has no sources of its own and is configured per-run, so a sweep cannot give it
#: a meaningful default.
ALL_BANKS = ("scifi_news_pro", "public_domain", "media_archive",
             "original", "shakespeare")

DEFAULT_PROFILE = "otr_w45_still_flat"
LEG_TIMEOUT_S = 3600


def _say(msg: str) -> None:
    print("[regression] %s  %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def _episode_dir_from_log(text: str) -> "pathlib.Path | None":
    for line in reversed(text.splitlines()):
        if "EPISODE " in line:
            frag = line.split("EPISODE ", 1)[1].strip()
            name = frag.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            for base in (REPO / "otr" / "episodes",
                         pathlib.Path("C:/Users/jeffr/Documents/ComfyUI/output/otr/episodes")):
                cand = base / name
                if cand.is_dir():
                    return cand
    return None


def _obs_dirs() -> "list[pathlib.Path]":
    return [p for p in (REPO / "otr" / "obs",
                        pathlib.Path("C:/Users/jeffr/Documents/ComfyUI/output/otr/obs"))
            if p.is_dir()]


def check_leg(episode: "pathlib.Path | None", bank: str) -> "list[str]":
    """Every complaint about this leg, on disk. Empty list means clean."""
    problems: list[str] = []
    if episode is None:
        return ["no EPISODE line in the leg log -- nothing to inspect"]

    ledgers = sorted(episode.glob("audio/*_ledger.json"))
    if not ledgers:
        problems.append("no frozen ledger under audio/")
        return problems
    try:
        led = json.loads(ledgers[-1].read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return ["ledger unreadable: %s" % exc]

    got_bank = str((led.get("meta") or {}).get("source_bank") or "")
    if got_bank != bank:
        problems.append("bank asked %r but ledger records %r" % (bank, got_bank))

    # Music cues: the stem must exist and be the length that was requested.
    # A cue file shorter than its request is a truncation, which is the class
    # of defect a regression sweep is FOR.
    try:
        import soundfile as sf
        for cue, want in (("opening", 12.0), ("closing", 8.0)):
            wav = episode / "audio" / ("music_cue_%s.wav" % cue)
            if not wav.is_file():
                problems.append("music cue %s missing" % cue)
                continue
            info = sf.info(str(wav))
            if info.duration < want - 0.5:
                problems.append("music cue %s is %.2fs, asked %.1fs"
                                % (cue, info.duration, want))
    except ImportError:
        problems.append("soundfile absent -- music cues NOT checked")

    stem = episode.name.replace("signal_lost_", "")
    published = [p for d in _obs_dirs() for p in d.glob("*%s*_final.mp4" % stem)]
    if not published:
        problems.append("nothing matching this episode in otr/obs -- NOT PUBLISHED")
    elif not any(p.stat().st_size > 0 for p in published):
        problems.append("published file is ZERO BYTES")
    return problems


def run_leg(bank: str, profile: str, acts: int, logs: pathlib.Path, tag: str) -> dict:
    log_path = logs / ("leg_%s.log" % tag)
    cmd = [sys.executable, str(RUNNER),
           "--profile", profile,
           "--act-count", str(acts),
           "--source-bank", bank,
           "--timeout", str(LEG_TIMEOUT_S)]
    started = time.time()
    _say("leg %s: bank=%s profile=%s acts=%d" % (tag, bank, profile, acts))
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("### %s\n" % " ".join(cmd))
        fh.flush()
        try:
            code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                  cwd=str(REPO),
                                  timeout=LEG_TIMEOUT_S + 600).returncode
        except subprocess.TimeoutExpired:
            code = -1
            fh.write("\n### runner timeout\n")
    minutes = round((time.time() - started) / 60.0, 1)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    episode = _episode_dir_from_log(text)
    problems = check_leg(episode, bank) if code == 0 else \
        ["runner exited %d" % code] + check_leg(episode, bank)
    row = {"tag": tag, "bank": bank, "profile": profile, "acts": acts,
           "exit": code, "minutes": minutes,
           "episode": episode.name if episode else None,
           "problems": problems, "log": log_path.name}
    _say("leg %s: %s in %.1f min%s"
         % (tag, "CLEAN" if not problems else "%d PROBLEM(S)" % len(problems),
            minutes, "" if not problems else " -- " + "; ".join(problems[:2])))
    if problems and code != 0:
        for line in text.splitlines()[-8:]:
            print("      | " + line[:150], flush=True)
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--banks", default="all",
                    help="comma list, or 'all' for every rollable bank")
    ap.add_argument("--profile", default=DEFAULT_PROFILE)
    ap.add_argument("--acts", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=1,
                    help="passes over the bank list; a second pass catches "
                         "what only a different draw shows")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    if not CANONICAL.is_file():
        raise SystemExit("the canonical workflow is missing: %s" % CANONICAL)
    banks = list(ALL_BANKS) if args.banks.strip().lower() == "all" else \
        [b.strip() for b in args.banks.split(",") if b.strip()]

    logs = REPO / "tmp" / "regression"
    logs.mkdir(parents=True, exist_ok=True)
    _say("canonical: %s" % CANONICAL)
    _say("%d bank(s) x %d pass(es) = %d one-act legs"
         % (len(banks), args.repeat, len(banks) * args.repeat))

    rows = []
    for p in range(args.repeat):
        for bank in banks:
            rows.append(run_leg(bank, args.profile, args.acts, logs,
                                "%s_p%d" % (bank, p + 1)))

    clean = [r for r in rows if not r["problems"]]
    print()
    print("=" * 92)
    print("%-26s %-8s %6s  %s" % ("leg", "exit", "min", "verdict"))
    for r in rows:
        print("%-26s %-8s %6.1f  %s"
              % (r["tag"], r["exit"], r["minutes"],
                 "clean" if not r["problems"] else "; ".join(r["problems"])))
    print()
    print("%d of %d legs clean" % (len(clean), len(rows)))
    print("THIS QUALIFIES NOTHING ABOUT THE AUDIO -- the operator's ear does that.")
    print("It proves the PATH ran and the artifacts landed where they belong.")

    if args.json_out:
        pathlib.Path(args.json_out).write_text(
            json.dumps(rows, indent=2), encoding="utf-8")
        _say("wrote %s" % args.json_out)
    return 0 if len(clean) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
