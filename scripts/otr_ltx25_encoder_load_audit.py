"""Count LTX 2.5 text-encoder disk loads against shot renders in a server log.

THE ACCEPTANCE INSTRUMENT for the episode-scoped encoder cache. Before the
cache, a live canonical leg read the 8.86 GiB Gemma-4 12B Q5 GGUF encoder once
per shot -- 13 loads for 13 renders, a 1:1 ratio counted in
``tmp/ltx25_g8_server.log`` on 2026-08-20. After it, an episode should read the
encoder ONCE and hit the cache for every later shot.

It counts the LOADER'S OWN log line, not the adapter's claim about itself. The
GGUF text-encoder read prints a distinctive qtype histogram, and the CLIP-side
placement line prints once per constructed CLIP, so either is evidence the file
was reopened -- whereas a "cache HIT" line the adapter writes about itself
proves only that the adapter believes it.

Usage::

    python scripts/otr_ltx25_encoder_load_audit.py <server.log> [--expect-episodes N]

Exit 0 when the ratio is at or below one load per episode, 1 otherwise, so it
can gate a leg rather than merely describe one.
"""

from __future__ import annotations

import argparse
import re
import sys

#: The text encoder's GGUF qtype histogram. Distinct from the DiT's, which
#: starts ``F32 (2603)`` -- matching the wrong one would count the transformer.
TEXT_ENCODER_READ = re.compile(
    r"gguf qtypes: F16 \(2\), I16 \(5\), Q5_K \(245\), BF16 \(346\), Q6_K \(88\)")
#: One per CLIP actually constructed by the CPU-pinned loader.
ENCODER_PINNED = re.compile(r"text encoder pinned to CPU")
#: One per shot render the lane starts.
SHOT_RENDER = re.compile(r"ltx25_video PLAN")
#: The adapter's own view, used only to CROSS-CHECK the loader evidence.
CACHE_HIT = re.compile(r"encoder cache HIT")
CACHE_MISS = re.compile(r"encoder cache MISS")
SCOPE_OPEN = re.compile(r"encoder cache scope OPEN")
SCOPE_CLOSED = re.compile(r"encoder cache scope CLOSED")
#: THE SILENT-DEGRADATION SIGNAL, and the reason this pattern is in the audit
#: rather than only in the log. ``render_clip``'s ``finally`` runs
#: ``reclaim_idle_models`` on EVERY shot, which detaches every resident patcher
#: -- the cached CLIP included. ``_cached_clip_is_live`` catches a patcher left
#: unusable by that and degrades to a full reload, which is SAFE but silent:
#: the render is correct, the wall clock is exactly the pre-cache behaviour, and
#: nothing fails. So the audit names it explicitly instead of letting a
#: never-hitting cache read as a working one.
PLACEMENT_DROP = re.compile(r"dropping the cached text encoder")


def audit(path):
    counts = dict.fromkeys(
        ("reads", "pinned", "renders", "hits", "misses", "opens", "closes",
         "drops"), 0)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if TEXT_ENCODER_READ.search(line):
                counts["reads"] += 1
            if ENCODER_PINNED.search(line):
                counts["pinned"] += 1
            if SHOT_RENDER.search(line):
                counts["renders"] += 1
            if CACHE_HIT.search(line):
                counts["hits"] += 1
            if CACHE_MISS.search(line):
                counts["misses"] += 1
            if SCOPE_OPEN.search(line):
                counts["opens"] += 1
            if SCOPE_CLOSED.search(line):
                counts["closes"] += 1
            if PLACEMENT_DROP.search(line):
                counts["drops"] += 1
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--expect-episodes", type=int, default=1,
                    help="how many episodes this log covers (default 1)")
    args = ap.parse_args(argv)

    c = audit(args.log)
    print("shot renders started      : %d" % c["renders"])
    print("text-encoder DISK reads   : %d" % c["reads"])
    print("encoder CLIPs constructed : %d" % c["pinned"])
    print("adapter says HIT / MISS   : %d / %d" % (c["hits"], c["misses"]))
    print("scopes OPENED / CLOSED    : %d / %d" % (c["opens"], c["closes"]))
    print("cached encoder DROPPED    : %d" % c["drops"])

    if not c["renders"]:
        print("\nNO SHOT RENDERS IN THIS LOG -- nothing to audit.")
        return 1

    print("\nratio: %.2f encoder reads per shot render"
          % (c["reads"] / float(c["renders"])))

    ok = True
    # POSITIVE EVIDENCE FIRST -- THIS GATE USED TO FAIL OPEN AND THE r3 PANEL
    # CAUGHT IT. The only check was ``reads > expected``, so a log with zero
    # matched lines PASSED: regex drift against a renamed loader line, a
    # truncated log, or the wrong file entirely all read as a clean run. An
    # acceptance gate that cannot tell "proved good" from "saw nothing" is
    # worse than no gate, because it is quoted as a receipt.
    if c["reads"] == 0:
        print("FAIL: zero text-encoder reads matched in a log with %d render(s)"
              " -- the pattern did not match anything, so this proves NOTHING."
              " Check the loader line format before trusting this file."
              % c["renders"])
        ok = False
    if c["opens"] == 0:
        print("FAIL: no encoder cache scope was ever opened -- the driver never "
              "reached the adapter's hooks, so the cache was not exercised")
        ok = False
    # ONE SHOT PROVES NO REUSE. Reuse is the whole claim, so the log has to
    # contain at least one render that could have reused and did.
    if c["renders"] < 2:
        print("FAIL: %d render(s) in this log -- a single shot cannot "
              "demonstrate reuse" % c["renders"])
        ok = False
    elif c["hits"] == 0:
        print("FAIL: %d render(s) and not one cache HIT -- the cache is "
              "loading every shot" % c["renders"])
        ok = False
    # The loader's two independent signals must agree; if they diverge, one of
    # the patterns has drifted and neither count can be trusted.
    if c["reads"] != c["pinned"]:
        print("FAIL: %d GGUF read(s) but %d CLIP(s) constructed -- these must "
              "match; a pattern has drifted" % (c["reads"], c["pinned"]))
        ok = False
    # THE REAL GATE. One read per episode is the target; the pre-cache baseline
    # was one per SHOT.
    if c["reads"] > args.expect_episodes:
        print("FAIL: %d disk reads for %d episode(s) -- the cache is not "
              "holding across beats" % (c["reads"], args.expect_episodes))
        ok = False
    # A SCOPE THAT OPENS AND NEVER CLOSES IS THE LEAK THIS DESIGN EXISTS TO
    # PREVENT, and it is invisible in the ratio.
    if c["opens"] != c["closes"]:
        print("FAIL: %d scope(s) opened but %d closed -- an episode leaked its "
              "8.86 GiB encoder" % (c["opens"], c["closes"]))
        ok = False
    # A cache that is dropped every shot renders CORRECTLY and buys NOTHING.
    # It must not read as a pass.
    if c["drops"] and c["drops"] >= max(1, c["renders"] - args.expect_episodes):
        print("FAIL: the cached encoder was dropped %d time(s) across %d "
              "render(s) -- the cache is silently degrading to a full reload "
              "every shot" % (c["drops"], c["renders"]))
        ok = False
    if ok:
        print("PASS: %d disk read(s) across %d shot render(s)"
              % (c["reads"], c["renders"]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
