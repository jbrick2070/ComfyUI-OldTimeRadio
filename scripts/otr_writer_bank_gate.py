"""Prove a WRITER fix against every story bank, not just the one that failed.

OPERATOR DIRECTIVE 2026-08-12: *"if we get a story writer fail on one media bank
and fix it, let's test it against all the other story banks."*

WHY THIS EXISTS. Both live writer failures in the 45-word every-visual-path
sweep (PBUG-20260812-02, PBUG-20260812-03) surfaced on ONE leg each, and the
campaign rolls `--source-bank "roll (any eligible bank)"` per leg -- so which
bank a defect appears on is luck. A fix proven on the bank that happened to fail
proves nothing about the other five, and the writer path is shared while each
bank brings its own prompt pack, cast conventions and source material. This gate
pins the bank instead of rolling it, and runs every one.

WHAT A LEG COSTS. The full canonical graph on the cheapest visual profile, so
each leg proves the whole path end to end (writer -> audio -> video -> publish),
not just that the writer returned. That is deliberately stronger than a
writer-only harness: a writer fix that produces a script no downstream stage can
use is not a fix.

WHAT IT REPORTS, AND WHY THE DISTINCTION MATTERS. A leg is classified as a
WRITER failure, a DOWNSTREAM failure, or a pass. Both defects this gate was
built for died in `OTR_LedgerScriptWriter` before any video work, and were
initially read as video-lane problems because the leg verdict was only
"no new file in otr/obs". Naming the stage that died is the difference between
one look and an hour.

USAGE (a server must already be listening on :8000 -- reset and boot per
CLAUDE.md sections 4 and 5 first):

    python scripts/otr_writer_bank_gate.py
    python scripts/otr_writer_bank_gate.py --banks shakespeare,original
    python scripts/otr_writer_bank_gate.py --profile otr_w45_still_word
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
PY = r"C:/Users/jeffr/Documents/ComfyUI/.venv/Scripts/python.exe"
LOCK = REPO / "tmp" / "_writer_bank_gate.lock"
RESULTS = REPO / "tmp" / "_writer_bank_gate_results.json"

#: Every bank the writer node offers, minus the two that are not a bank.
#:
#: `roll (any eligible bank)` is the SENTINEL this gate exists to replace, and
#: `custom_source_bank` needs operator-supplied source material that is not in
#: the repo -- a leg for it would fail on missing input and say nothing about
#: the writer. Both are excluded BY NAME rather than by a slice, so a newly
#: added bank joins this gate automatically instead of being silently skipped.
NOT_A_BANK = ("roll (any eligible bank)", "custom_source_bank")

#: Cheapest visual path, and the leg that carried PBUG-20260812-02 live.
DEFAULT_PROFILE = "otr_w45_still_flat"

LEG_TIMEOUT_S = 5400

#: Log markers that identify a failure as the WRITER's. Taken from the two live
#: failures verbatim, not invented -- see PROD_BUG_LOG PBUG-20260812-02/03.
WRITER_MARKERS = (
    "OTR_LedgerScriptWriter",
    "markup ladder exhausted",
    "pass 'script' failed",
    "Fable2ScriptError",
)


def banks_from_the_node() -> list:
    """The bank list read from the WRITER'S OWN `INPUT_TYPES`.

    Derived rather than hardcoded for the same reason the campaign derives its
    engine roster: a hand-kept copy silently omits whatever was added last, and
    a gate that quietly skips a bank while reporting "all banks passed" is worse
    than no gate. Raises if the widget cannot be read -- refusing beats running
    a subset under a total-coverage headline.
    """
    for path in (str(REPO), str(REPO.parent)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter as Writer

    spec = Writer.INPUT_TYPES()
    for section in ("required", "optional"):
        entry = spec.get(section, {}).get("source_bank")
        if entry:
            choices = entry[0] if isinstance(entry, tuple) else entry
            found = [b for b in choices if b not in NOT_A_BANK]
            if found:
                return found
    raise SystemExit(
        "REFUSING: could not read source_bank choices from "
        "OTR_LedgerScriptWriter.INPUT_TYPES(). A bank gate that cannot see the "
        "bank list cannot claim to have covered them.")


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _say(msg: str) -> None:
    print("[bank-gate] %s  %s" % (_now(), msg), flush=True)


def classify_failure(log_text: str) -> str:
    """WRITER vs DOWNSTREAM, from the leg log."""
    for marker in WRITER_MARKERS:
        if marker in log_text:
            return "WRITER"
    return "DOWNSTREAM"


def writer_defect_detail(log_text: str) -> str:
    """The defect rows the ladder reported, when it reported any."""
    marker = "last defects:"
    index = log_text.find(marker)
    if index < 0:
        return ""
    tail = log_text[index + len(marker):index + len(marker) + 400]
    rows = [line.strip() for line in tail.splitlines() if line.strip().startswith("-")]
    return " | ".join(rows[:4])


def run_bank(bank: str, profile: str, words: int) -> dict:
    slug = "".join(ch if ch.isalnum() else "_" for ch in bank)
    log_path = REPO / "tmp" / ("_bankgate_%s.log" % slug)
    _say("BANK %s (profile %s) -> %s" % (bank, profile, log_path.name))
    started = time.time()

    cmd = [PY, str(REPO / "scripts" / "otr_canonical_api_run.py"),
           "--profile", profile,
           "--words", str(words),
           "--source-bank", bank,
           "--visual-style", "roll (any style)",
           "--timeout", str(LEG_TIMEOUT_S)]
    env = dict(os.environ, PYTHONUTF8="1")

    with open(log_path, "w", encoding="utf-8") as log:
        log.write("### bank=%s profile=%s start=%s\n" % (bank, profile, _now()))
        log.flush()
        try:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  cwd=str(REPO), env=env,
                                  timeout=LEG_TIMEOUT_S + 600).returncode
        except subprocess.TimeoutExpired:
            code = -1
            log.write("\n### runner timeout\n")

    minutes = round((time.time() - started) / 60.0, 1)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    result = {"bank": bank, "profile": profile, "exit": code,
              "minutes": minutes, "ok": code == 0, "log": log_path.name}
    if code != 0:
        result["stage"] = classify_failure(text)
        detail = writer_defect_detail(text)
        if detail:
            result["defects"] = detail
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--words", type=int, default=45)
    ap.add_argument("--profile", default=DEFAULT_PROFILE)
    ap.add_argument("--banks", default=None,
                    help="comma-separated bank ids instead of every bank")
    args = ap.parse_args(argv)

    every = banks_from_the_node()
    banks = every
    if args.banks:
        wanted = {b.strip() for b in args.banks.split(",") if b.strip()}
        unknown = wanted - set(every)
        if unknown:
            raise SystemExit("REFUSING: not registered banks: %s"
                             % ", ".join(sorted(unknown)))
        banks = [b for b in every if b in wanted]

    if LOCK.exists():
        print("REFUSING: %s exists -- a gate is already running." % LOCK,
              file=sys.stderr)
        return 2
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text("pid=%d start=%s\n" % (os.getpid(), _now()), encoding="utf-8")

    results = []
    try:
        for bank in banks:
            results.append(run_bank(bank, args.profile, args.words))
            RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    finally:
        LOCK.unlink(missing_ok=True)

    print("\n=========== WRITER BANK GATE (%d words) ===========" % args.words)
    for r in results:
        print("%-18s %-4s exit=%-4s %6s min  %s %s" % (
            r["bank"], "PASS" if r["ok"] else "FAIL", r["exit"], r["minutes"],
            r.get("stage", ""), r.get("defects", "")))

    passed = sum(1 for r in results if r["ok"])
    writer_fails = [r for r in results if not r["ok"] and r.get("stage") == "WRITER"]
    print("%d/%d banks passed" % (passed, len(results)))
    print("banks registered: %d | run: %d" % (len(every), len(results)))

    # SAY WHAT WAS NOT COVERED. A gate reporting "6/6 passed" over a subset
    # reads as total coverage -- the same defect the campaign's own summary was
    # corrected for.
    skipped = [b for b in every if b not in {r["bank"] for r in results}]
    if skipped:
        print("NOT RUN (%d): %s" % (len(skipped), ", ".join(skipped)))
    if writer_fails:
        print("WRITER FAILURES on: %s"
              % ", ".join(r["bank"] for r in writer_fails))
        return 1
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
