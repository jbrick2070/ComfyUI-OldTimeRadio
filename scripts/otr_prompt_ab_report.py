"""Compare two renders of the SAME episode plan, prompt by prompt.

    python scripts/otr_prompt_ab_report.py <source episode> <replay episode>

Both arguments are an episode directory, an episode id, or a ledger path. The
report reads each side's ``meta.render_trace`` -- the durable per-clip receipt --
and prints, per shot, the seed both sides used and the text each side actually
sent.

WHY THIS EXISTS SEPARATELY FROM ``otr_verify_replay.py --ab``. That tool compares
two REPLAYS of one frozen bundle and requires both to carry the replay stamps. A
prompt-version experiment does not need two renders: the episode that was already
published IS the first arm, rendered from the same plan with the same seeds by
the previous composer. Only the second arm has to run. So this reads a published
episode beside a replay of it, which the verifier's A/A and A/B rules both refuse
by construction, and it makes exactly the two claims that matter:

    the SEEDS must be identical   -- or the pictures differ for a second reason
                                     and the comparison proves nothing;
    the PROMPTS must differ       -- or nothing was tested.

The seed is derived from ``render_request_hash``, which mixes the brief, the
cast, the beat and the character and has never included the prompt. That is what
makes the first claim true by construction rather than by luck, and it is worth
asserting anyway: if it ever stops being true, every A/B taken on this lane
becomes uninterpretable at once.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
EPISODE_ROOTS = (
    pathlib.Path(r"C:\Users\jeffr\Documents\ComfyUI\output\otr\episodes"),
    REPO_ROOT / "otr" / "episodes",
)


def load_ledger(arg: str) -> tuple:
    """(ledger, label) from an episode dir, an episode id, or a ledger path."""
    p = pathlib.Path(arg)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8")), p.parent.parent.name
    candidates = [p] if p.is_dir() else []
    if not candidates:
        for root in EPISODE_ROOTS:
            if (root / arg).is_dir():
                candidates.append(root / arg)
    if not candidates:
        raise SystemExit("no episode found for %r" % (arg,))
    ep = candidates[0]
    hits = sorted(glob.glob(os.path.join(str(ep), "audio", "*_ledger.json")))
    if not hits:
        raise SystemExit("no ledger under %s" % (ep / "audio"))
    return json.loads(pathlib.Path(hits[-1]).read_text(encoding="utf-8")), ep.name


def trace(led: dict) -> list:
    rows = ((led.get("meta") or {}).get("render_trace")) or []
    return [r for r in rows if isinstance(r, dict)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="the episode already published (arm A)")
    ap.add_argument("replay", help="the replay under the new composer (arm B)")
    ap.add_argument("--full", action="store_true",
                    help="print whole prompts instead of trimming to the width")
    args = ap.parse_args(argv)

    a_led, a_name = load_ledger(args.source)
    b_led, b_name = load_ledger(args.replay)
    a, b = trace(a_led), trace(b_led)

    print("A  %s   (%d trace row(s))" % (a_name, len(a)))
    print("B  %s   (%d trace row(s))" % (b_name, len(b)))
    print()

    ok = True
    if not a or not b:
        print("FAIL: one side carries no render trace, so there is nothing to "
              "compare. A ledger rendered before the durable receipt landed "
              "cannot be an arm.")
        return 1
    if len(a) != len(b):
        # NOT ZIPPED. Comparing a prefix would print per-shot verdicts that are
        # true of the prefix and false of the run.
        print("FAIL: the arms rendered a different number of clips (%d vs %d), "
              "so they are not the same plan and no per-shot line below would "
              "mean anything." % (len(a), len(b)))
        return 1

    by_b = {(r.get("shot_id"), r.get("segment_index")): r for r in b}
    same_seed = diff_prompt = 0
    for ra in a:
        rb = by_b.get((ra.get("shot_id"), ra.get("segment_index")))
        if rb is None:
            print("FAIL: %s is missing from arm B" % ra.get("shot_id"))
            ok = False
            continue
        seed_ok = ra.get("seed") == rb.get("seed")
        text_a = str(ra.get("text_prompt") or "")
        text_b = str(rb.get("text_prompt") or "")
        prompt_moved = text_a != text_b
        same_seed += 1 if seed_ok else 0
        diff_prompt += 1 if prompt_moved else 0
        ok &= seed_ok and prompt_moved
        width = 10000 if args.full else 150
        print("%-26s seed %-12s %s   prompt %s"
              % (ra.get("shot_id"), ra.get("seed"),
                 "SAME" if seed_ok else "MOVED -> %s" % rb.get("seed"),
                 "differs" if prompt_moved else "IDENTICAL (nothing tested)"))
        print("    A  %s" % text_a[:width])
        print("    B  %s" % text_b[:width])
        print()

    print("seeds identical on %d of %d shots" % (same_seed, len(a)))
    print("prompts differ on   %d of %d shots" % (diff_prompt, len(a)))
    print()
    print("VERDICT:", "PASS -- same seeds, different prompts" if ok
          else "FAIL -- see the lines above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
