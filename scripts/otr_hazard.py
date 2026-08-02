#!/usr/bin/env python
"""BIG RED LIGHT: this tree is mid-surgery and knowingly broken.

Operator ruling 2026-08-02: "AI coders are too hesitant about temporarily
breaking software -- they need to learn it's OK while you are building the
fixes. Maybe you need a big red hazard light (code is broken now), but it's OK
in my coding UI."

The hesitation is real and it has a cost: an assistant that treats a red suite
as something to avoid will sequence work around staying green instead of around
being correct -- shipping half-fixes, or leaving a rule violated because
enforcing it breaks twelve tests at once. Breaking the tree on purpose, briefly,
is often the honest path.

What makes that safe is not avoiding breakage; it is never LOSING TRACK of it.
So: raise the flag, do the surgery in whatever order is actually correct, and
the flag is what guarantees nobody -- operator or assistant, now or three
sessions later -- mistakes a knowingly-broken tree for a working one.

    python scripts/otr_hazard.py raise "ripping the wan_ti2v mirror" \\
        --expect "wan_ti2v refuses until coverage lands"
    python scripts/otr_hazard.py status
    python scripts/otr_hazard.py clear

The flag lives at tmp/_OTR_BROKEN.json, is printed by `status` in red, and
`clear` REFUSES while the suite is still red -- lowering the flag is a claim
about the tree, so it has to be earned rather than asserted.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
FLAG = REPO / "tmp" / "_OTR_BROKEN.json"
PY = r"C:/Users/jeffr/Documents/ComfyUI/.venv/Scripts/python.exe"

_RED = "\033[91m"
_YEL = "\033[93m"
_GRN = "\033[92m"
_OFF = "\033[0m"


def _now() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _banner(text: str, colour: str = _RED) -> None:
    bar = "#" * 72
    print("%s%s\n%s\n%s%s" % (colour, bar, text, bar, _OFF), flush=True)


def _suite_green() -> "tuple[bool, str]":
    """Run the suite quietly and report (green, last-line)."""
    proc = subprocess.run(
        [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/"],
        cwd=str(REPO), capture_output=True, text=True)
    tail = [ln for ln in (proc.stdout or "").strip().splitlines() if ln.strip()]
    return proc.returncode == 0, (tail[-1] if tail else "(no output)")


def cmd_raise(args) -> int:
    payload = {
        "raised_at": _now(),
        "why": args.why,
        "expected_breakage": args.expect or "(not stated)",
        "raised_by": "assistant",
    }
    FLAG.parent.mkdir(parents=True, exist_ok=True)
    FLAG.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _banner("  CODE IS BROKEN ON PURPOSE -- surgery in progress\n"
            "  why      : %s\n"
            "  expected : %s\n"
            "  raised   : %s"
            % (payload["why"], payload["expected_breakage"],
               payload["raised_at"]))
    return 0


def cmd_status(args) -> int:
    if not FLAG.exists():
        print("%sno hazard flag -- the tree is not knowingly broken%s"
              % (_GRN, _OFF))
        return 0
    data = json.loads(FLAG.read_text(encoding="utf-8"))
    _banner("  CODE IS BROKEN ON PURPOSE -- surgery in progress\n"
            "  why      : %s\n"
            "  expected : %s\n"
            "  since    : %s"
            % (data.get("why"), data.get("expected_breakage"),
               data.get("raised_at")))
    return 1


def cmd_clear(args) -> int:
    if not FLAG.exists():
        print("%sno hazard flag to clear%s" % (_GRN, _OFF))
        return 0
    if args.force:
        FLAG.unlink()
        print("%sflag cleared with --force (suite NOT verified)%s"
              % (_YEL, _OFF))
        return 0
    print("verifying the tree before lowering the flag...", flush=True)
    green, tail = _suite_green()
    if not green:
        _banner("  FLAG STAYS UP -- the suite is still red\n"
                "  %s\n"
                "  Lowering the flag is a CLAIM about the tree. Earn it, or\n"
                "  pass --force if you are deliberately handing over broken."
                % tail)
        return 1
    FLAG.unlink()
    print("%s%s%s" % (_GRN, tail, _OFF))
    print("%shazard cleared -- the tree is green again%s" % (_GRN, _OFF))
    return 0


def cmd_runnable(args) -> int:
    """The whole thing in one character. "Can I run this right now?" -- Y/N.

    The banner is for a human mid-surgery; this is for everything else -- a
    script, a cron, a glance. Exit code carries it too (0 = Y), so it drops
    straight into a shell guard:

        python scripts/otr_hazard.py runnable && python scripts/otr_w45_campaign.py
    """
    if not FLAG.exists():
        print("Y")
        return 0
    data = json.loads(FLAG.read_text(encoding="utf-8"))
    print("N  (%s)" % data.get("why", "surgery in progress"))
    return 1


def cmd_coming(args) -> int:
    """The Coming Attractions board: what is behind the barrier, and what
    opens when it comes down. A hazard light says STOP; this says WHY and
    WHAT'S NEXT, which is the part a reader actually needs."""
    if not FLAG.exists():
        print("%sopen for business -- nothing behind a barrier%s" % (_GRN, _OFF))
        return 0
    data = json.loads(FLAG.read_text(encoding="utf-8"))
    print("%s+----------------------------------------------------------+" % _YEL)
    print("|                    COMING ATTRACTIONS                     |")
    print("+----------------------------------------------------------+%s" % _OFF)
    print("  under construction : %s" % data.get("why"))
    print("  expect while closed: %s" % data.get("expected_breakage"))
    print("  closed since       : %s" % data.get("raised_at"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_raise = sub.add_parser("raise", help="declare the tree knowingly broken")
    p_raise.add_argument("why", help="what surgery is in progress")
    p_raise.add_argument("--expect", default="",
                         help="what is expected to be red while it lasts")
    p_raise.set_defaults(func=cmd_raise)

    p_stat = sub.add_parser("status", help="is the tree knowingly broken?")
    p_stat.set_defaults(func=cmd_status)

    p_run = sub.add_parser("runnable", help="Y/N -- can I run this right now?")
    p_run.set_defaults(func=cmd_runnable)

    p_coming = sub.add_parser("coming",
                              help="Coming Attractions: what is behind the barrier")
    p_coming.set_defaults(func=cmd_coming)

    p_clear = sub.add_parser("clear", help="lower the flag (runs the suite)")
    p_clear.add_argument("--force", action="store_true",
                         help="lower it WITHOUT a green suite")
    p_clear.set_defaults(func=cmd_clear)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
