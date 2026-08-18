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

#: The role that owns the fix. Everything else is a different lane with a
#: different contract, and mixing them is how a checker cries wolf.
CHARACTER_LANE = "char_voice"

# One P-OBS line, as `_render_per_line` writes it. The ROLE prefix is captured
# because the two lanes answer to different contracts.
_POBS = re.compile(
    r"(?P<role>[a-z_]+):\s+"
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
    """The checks, as counts and named offenders -- PER LANE."""
    character_rows = [r for r in rows if r["role"] == CHARACTER_LANE]
    other_rows = [r for r in rows if r["role"] != CHARACTER_LANE]

    by_char = collections.defaultdict(list)
    for row in character_rows:
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
        for r in character_rows
        if r["mass"] is not None and r["mass"] > EFFECTIVE_EMOTION_MASS_CAP
    ]
    policies = collections.Counter(r["policy"] for r in character_rows)
    alphas = collections.Counter(r["alpha"] for r in character_rows)

    # THE INVERSE CHECK [QA-6]. The announcer is not a cloned character and has
    # no identity to hold steady across beats, so his profile was deliberately
    # left on the legacy seed. A row on the OTHER lane that turns up on
    # `char_v1` means the opt-in leaked past its scope.
    leaked = [
        {"role": r["role"], "line": r["line"], "policy": r["policy"]}
        for r in other_rows if r["policy"] != "line_v1"
    ]

    return {
        "voiced_lines": len(rows),
        "character_lane_lines": len(character_rows),
        "other_lane_lines": len(other_rows),
        "characters": len(by_char),
        "policies": dict(policies),
        "alphas": dict(alphas),
        "capped_lines": sum(1 for r in character_rows if r["capped"]),
        "max_effective_mass": max(
            (r["mass"] for r in character_rows if r["mass"] is not None),
            default=0.0),
        "rows_without_emotion": sum(
            1 for r in character_rows if r["mass"] is None),
        "characters_with_split_seeds": split_characters,
        "characters_with_moving_reference": moving_reference,
        "lines_over_the_cap": over_cap,
        "other_lane_rows_on_the_character_policy": leaked,
    }


def verdict(report: dict, expect_policy: str = "", expect_alpha: str = "",
            expect_mass_cap: float = EFFECTIVE_EMOTION_MASS_CAP) -> list:
    """Human-readable findings. An empty list is a clean arm."""
    findings = []
    if not report["character_lane_lines"]:
        findings.append("NO CHARACTER-LANE RECEIPTS FOUND -- this log has no "
                        "char_voice P-OBS lines, so it proves nothing about the "
                        "fix (%d line(s) on other lanes)"
                        % report["other_lane_lines"])
        return findings

    if report["other_lane_rows_on_the_character_policy"]:
        findings.append(
            "SCOPE LEAK [QA-6]: a non-character lane rendered on the character "
            "seed policy -- %s"
            % report["other_lane_rows_on_the_character_policy"][:4])

    # THE ARM DECLARES ITS OWN CEILING. A control arm booted with
    # OTR_INDEXTTS2_EMO_MASS_CAP=8 is SUPPOSED to exceed 0.4 -- reporting that
    # as a failure would read as "half the arms failed" when half the arms are
    # controls doing exactly what they were booted to do.
    over = [row for row in report["lines_over_the_cap"]
            if row["mass"] > expect_mass_cap]
    if over:
        findings.append(
            "EMOTION MASS OVER THIS ARM'S CEILING (%s) on %d line(s): %s"
            % (expect_mass_cap, len(over), over[:4]))

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
    ap.add_argument("--expect-mass-cap", type=float,
                    default=EFFECTIVE_EMOTION_MASS_CAP,
                    help="the emotion ceiling this arm booted with; pass 8 for "
                         "a control arm that deliberately runs without one")
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
        findings = verdict(report, args.expect_policy, args.expect_alpha,
                           args.expect_mass_cap)
        report["log"] = path.name
        report["findings"] = findings
        overall.append(report)

        if not args.json:
            print("=" * 72)
            print("ARM %s" % path.name)
            print("  character lines   : %d across %d characters "
                  "(+%d on other lanes, expected legacy)"
                  % (report["character_lane_lines"], report["characters"],
                     report["other_lane_lines"]))
            print("  seed policies     : %s" % report["policies"])
            print("  alphas            : %s" % report["alphas"])
            print("  max emotion mass  : %s (this arm's ceiling %s), "
                  "%d line(s) capped"
                  % (report["max_effective_mass"], args.expect_mass_cap,
                     report["capped_lines"]))
            if report["max_effective_mass"] > 1.0:
                print("  NOTE: mass above 1.0 makes the vendor's "
                      "(1 - sum) residual NEGATIVE -- the speaker's own "
                      "emotional embedding is subtracted, not merely replaced.")
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
