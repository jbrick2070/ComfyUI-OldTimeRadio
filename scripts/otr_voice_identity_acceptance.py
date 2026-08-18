"""Acceptance for the voice-identity fix -- read the receipt, not the waveform.

THE BAR, in the operator's words: *"Nag 1 sounded good, Nag beat 2 was another
voice."* A character must not change voice between his own lines. This turns
that into a deterministic read over what a live leg actually produced, so a
window does not have to listen to four episodes to know whether the fix held --
and so the 2x2 arms can be compared on evidence rather than recollection.

WHAT IT CHECKS, per episode, from the render log's P-OBS lines:

  1. ONE SEED PER CHARACTER. Every line a given char_id speaks must carry the
     SAME `seed=`. This is the defect itself. On a `policy=line_v1` arm the
     seeds SHOULD differ -- that arm is the control, and this instrument
     reports the split rather than failing it.
  2. THE POLICY IS THE ONE THE ARM ASKED FOR. `policy=char_v1` on the fixed
     arms, `policy=line_v1` on the control arms. An arm that silently ran the
     other policy proves nothing, and a green log is exactly how that hides.
  3. THE EFFECTIVE EMOTION MASS IS AT OR UNDER THE CEILING. `emo_mass=` must
     be <= 0.4 on every char_voice line. The `(capped)` marker says the ceiling
     actually fired rather than merely being available.
  4. THE REFERENCE IS STABLE PER CHARACTER. `seed_ref=` must not change between
     a character's own lines -- a moving reference would move the seed
     legitimately and mask a regression.
  5. THE ALPHA IS THE ARM'S ALPHA. `alpha=` must match what the boot exported,
     which is how an arm proves it ran under the environment it claims.

IT NEVER FAILS A RENDER. Everything here has already happened; this reads
receipts. Exit code is a REPORT for the driver, never a gate on an episode.

Usage:
    python scripts/otr_voice_identity_acceptance.py --log tmp/_arm_a.log
    python scripts/otr_voice_identity_acceptance.py --since-minutes 240
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EFFECTIVE_EMOTION_MASS_CAP = 0.4

# One P-OBS line, as `_render_per_line` writes it.
_POBS = re.compile(
    r"line=(?P<line>\S+)\s+char=(?P<char>\S+)\s*->\s*"
    r"voice_ref_id=(?P<ref_id>\S+)\s+"
    r"ref=(?P<ref>\S+)\s+"
    r"engine=(?P<engine>\S+)\s+"
    r"alpha=(?P<alpha>\S+)\s+"
    r"delivery=(?P<delivery>\S+)\s+"
    r"emo_mass=(?P<mass>[0-9.]+|n/a)(?P<capped>\(capped\))?\s+"
    r"seed=(?P<seed>\d+)\s+"
    r"policy=(?P<policy>\S+)\s+"
    r"seed_ref=(?P<seed_ref>\S+)"
)


def parse_pobs(text: str) -> list:
    """Every per-line voice receipt in one render log."""
    rows = []
    for match in _POBS.finditer(text):
        row = match.groupdict()
        row["capped"] = bool(row["capped"])
        # chatterbox and dia carry the character SEED but have no emotion vector
        # to cap, so they report `n/a`. Parsing them as rows anyway is the point:
        # both opted into character seeding, and a seed-split regression on
        # those engines was invisible while this pattern required a number.
        row["mass"] = None if row["mass"] == "n/a" else float(row["mass"])
        row["seed"] = int(row["seed"])
        rows.append(row)
    return rows


def audit_rows(rows: list) -> dict:
    """The five checks, as counts and named offenders."""
    by_char = collections.defaultdict(list)
    for row in rows:
        if row["char"] != "-":
            by_char[row["char"]].append(row)

    split_characters = {}
    moving_reference = {}
    for char, char_rows in sorted(by_char.items()):
        seeds = {r["seed"] for r in char_rows}
        if len(seeds) > 1:
            split_characters[char] = {
                "lines": len(char_rows),
                "distinct_seeds": len(seeds),
                "seeds": sorted(seeds),
            }
        refs = {r["seed_ref"] for r in char_rows}
        if len(refs) > 1:
            moving_reference[char] = sorted(refs)

    over_cap = [
        {"line": r["line"], "char": r["char"], "mass": r["mass"]}
        for r in rows
        if r["mass"] is not None and r["mass"] > EFFECTIVE_EMOTION_MASS_CAP
    ]
    policies = collections.Counter(r["policy"] for r in rows)
    alphas = collections.Counter(r["alpha"] for r in rows)

    return {
        "voiced_lines": len(rows),
        "characters": len(by_char),
        "policies": dict(policies),
        "alphas": dict(alphas),
        "capped_lines": sum(1 for r in rows if r["capped"]),
        "max_effective_mass": max(
            (r["mass"] for r in rows if r["mass"] is not None), default=0.0),
        "rows_without_emotion": sum(1 for r in rows if r["mass"] is None),
        "characters_with_split_seeds": split_characters,
        "characters_with_moving_reference": moving_reference,
        "lines_over_the_cap": over_cap,
    }


def verdict(report: dict, expect_policy: str = "", expect_alpha: str = "") -> list:
    """Human-readable findings. An empty list is a clean arm."""
    findings = []
    if not report["voiced_lines"]:
        findings.append("NO VOICE RECEIPTS FOUND -- this log has no char_voice "
                        "P-OBS lines, so it proves nothing about the fix")
        return findings

    if report["lines_over_the_cap"]:
        findings.append(
            "EMOTION MASS OVER THE CEILING on %d line(s): %s"
            % (len(report["lines_over_the_cap"]),
               report["lines_over_the_cap"][:4]))

    if expect_policy:
        wrong = {p: n for p, n in report["policies"].items() if p != expect_policy}
        if wrong:
            findings.append(
                "WRONG SEED POLICY: expected every line on %r, also saw %s -- "
                "this arm did not run the environment it claims"
                % (expect_policy, wrong))

    if expect_alpha:
        wrong = {a: n for a, n in report["alphas"].items() if a != expect_alpha}
        if wrong:
            findings.append(
                "WRONG ALPHA: expected %r, also saw %s" % (expect_alpha, wrong))

    if report["characters_with_moving_reference"]:
        findings.append(
            "REFERENCE MOVED MID-EPISODE for %s -- a character's seed may have "
            "moved for a legitimate reason, which masks a regression"
            % sorted(report["characters_with_moving_reference"]))

    split = report["characters_with_split_seeds"]
    if expect_policy == "char_v1" and split:
        findings.append(
            "THE DEFECT IS STILL PRESENT: %d character(s) drew more than one "
            "engine seed across their own lines -- %s"
            % (len(split), sorted(split)))
    if expect_policy == "line_v1" and not split:
        findings.append(
            "CONTROL ARM DID NOT REPRODUCE: no character drew split seeds on a "
            "line_v1 arm, so the comparison has no contrast. Check that the "
            "arm really booted with the character seed disabled.")
    return findings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", action="append", default=[],
                    help="render/leg log to read (repeatable)")
    ap.add_argument("--expect-policy", default="",
                    help="char_v1 or line_v1 -- what this arm booted with")
    ap.add_argument("--expect-alpha", default="",
                    help="the alpha this arm exported, e.g. 0.4")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    if not args.log:
        ap.error("pass at least one --log")

    overall = []
    for name in args.log:
        path = pathlib.Path(name)
        if not path.is_absolute():
            path = REPO / path
        if not path.exists():
            print("MISSING LOG: %s" % path)
            overall.append({"log": str(path), "findings": ["log not found"]})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        report = audit_rows(parse_pobs(text))
        findings = verdict(report, args.expect_policy, args.expect_alpha)
        report["log"] = path.name
        report["findings"] = findings
        overall.append(report)

        if not args.json:
            print("=" * 72)
            print("ARM %s" % path.name)
            print("  voiced lines      : %d across %d characters"
                  % (report["voiced_lines"], report["characters"]))
            print("  seed policies     : %s" % report["policies"])
            print("  alphas            : %s" % report["alphas"])
            print("  max emotion mass  : %s (ceiling %s), %d line(s) capped"
                  % (report["max_effective_mass"], EFFECTIVE_EMOTION_MASS_CAP,
                     report["capped_lines"]))
            print("  split-seed chars  : %s"
                  % (sorted(report["characters_with_split_seeds"]) or "none"))
            if findings:
                for finding in findings:
                    print("  ! %s" % finding)
            else:
                print("  CLEAN")

    if args.json:
        print(json.dumps({"generated": datetime.datetime.now(
            datetime.timezone.utc).isoformat(), "arms": overall}, indent=2))

    # A REPORT, not a gate. Non-zero tells the DRIVER something needs reading;
    # it never reaches a render.
    return 1 if any(a.get("findings") for a in overall) else 0


if __name__ == "__main__":
    raise SystemExit(main())
