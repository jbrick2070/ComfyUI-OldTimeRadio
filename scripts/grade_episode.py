"""THE DURABLE ACCEPTANCE SCRIPT (WIRE-W5, r4/A6: "Name a durable repository
script").

Grades a rendered episode against the route its own ledger FROZE. Reads two
documents and nothing else -- no ComfyUI, no registry, no environment -- so it
runs anywhere the files do, including on a different box from the render.

    python scripts/grade_episode.py --ledger <ledger.json> --manifest <manifest.json>

Exit codes:
    0  the episode delivered the route it froze
    1  at least one per-shot acceptance finding (printed, one per line)
    2  a document could not be read

A grader nobody can run is the same failure mode as an unowned ruling, which is
why this file exists next to the module rather than only inside the test suite.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nodes._otr_video_engines import acceptance  # noqa: E402


def _load(path, what):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        print("cannot read the %s at %s: %s" % (what, path, exc),
              file=sys.stderr)
        raise SystemExit(2)


def _unwrap_ledger(doc):
    """The LEDGER, whether the file holds it directly or wraps it.

    ``OTR_VideoRenderBatch`` writes its retained ledger as
    ``{"ledger": {...}, "master_audio_path": "..."}``, and this script used to
    hand that WRAPPER straight to the grader -- which looks for ``video.shots``
    at the ROOT. The wrapper has no ``video`` key, so every shot vanished and
    the run printed "ACCEPTED: 0 shot(s)" and exited 0.

    That is the failure this script exists to refuse: it reported success on an
    episode it never graded. It already knew unreadable is not clean; empty had
    to learn the same lesson.
    """
    if isinstance(doc, dict) and "video" not in doc \
            and isinstance(doc.get("ledger"), dict):
        return doc["ledger"]
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Grade a rendered episode against its own frozen route.")
    parser.add_argument("--ledger", required=True,
                        help="the episode ledger JSON (carries video.shots and "
                             "video.roles_effective)")
    parser.add_argument("--manifest", required=True,
                        help="the clip manifest JSON (carries the DELIVERED "
                             "engine_id per shot)")
    parser.add_argument("--json", action="store_true",
                        help="emit the findings as JSON instead of lines")
    args = parser.parse_args(argv)

    ledger = _unwrap_ledger(_load(args.ledger, "ledger"))
    manifest = _load(args.manifest, "manifest")

    # PARSEABLE IS NOT THE SAME AS READABLE. ``json.load`` happily returns a
    # list, a string or a number for a file whose root is not an object, and
    # every reader below assumes a mapping -- so a document of the wrong SHAPE
    # crashed with an AttributeError instead of exiting 2, which is the exact
    # verdict this script promises for a document it cannot read.
    for doc, what, path in ((ledger, "ledger", args.ledger),
                            (manifest, "manifest", args.manifest)):
        if not isinstance(doc, dict):
            print("the %s at %s is %s, not a JSON object -- there is nothing "
                  "here to grade" % (what, path, type(doc).__name__),
                  file=sys.stderr)
            raise SystemExit(2)

    # A ZERO-SHOT LEDGER IS NOT A CLEAN EPISODE. Every rule below is per-shot,
    # so an empty shot list makes all of them vacuously true and the script
    # would exit 0 having judged nothing. "Could not grade" belongs with the
    # other document failures at exit 2 -- the same distinction
    # ``audit_voice_gender_consistency.py`` draws when its scan cannot finish.
    shots = (ledger.get("video") or {}).get("shots") or ()
    if not shots:
        print("the ledger at %s carries no video.shots, so there is nothing to "
              "grade -- this is NOT an accepted episode" % (args.ledger,),
              file=sys.stderr)
        raise SystemExit(2)

    findings = acceptance.grade_episode(ledger, manifest)

    if args.json:
        print(json.dumps(findings, indent=2, sort_keys=True))
    elif findings:
        print(acceptance.format_findings(findings))
    else:
        print("ACCEPTED: %d shot(s) delivered the route this episode froze."
              % len(shots))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
