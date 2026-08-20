"""Prove the LTX 2.5 episode-scoped text-encoder cache actually held, per scope.

THE ACCEPTANCE INSTRUMENT. Before the cache, a live canonical leg read the
8.86 GiB Gemma-4 12B Q5 GGUF encoder once per shot -- **31 loads for 31 renders**
on 2026-08-20, ratio 1.00. After it, ONE cache scope should read the encoder
ONCE and hit for every later shot in that scope.

IT COUNTS THE LOADER'S OWN LOG LINE, never the adapter's claim about itself. The
GGUF text-encoder read prints a distinctive qtype histogram and the CLIP-side
placement line prints once per constructed CLIP; a "cache HIT" line the adapter
writes about itself proves only that the adapter believes it.

**IT CORRELATES PER SCOPE, and that is the whole design.** An earlier version
summed every matching line into one counter set, which let a good scope subsidise
a bad one: a scope that reloaded twice passed because a second, EMPTY scope had
raised the global allowance. The same aggregation failed the opposite way -- a
perfect episode failed if the same server log also contained the legitimate
unscoped `render_single` diagnostic, whose load counted against the episode's
budget. Both were found by the r4 review lane.

So: events are attributed to the scope interval they occur in, unscoped events
are REPORTED BUT NEVER GATED, and an empty scope grants nobody an allowance.

Usage::

    python scripts/otr_ltx25_encoder_load_audit.py <server.log>

Exit 0 only on positive evidence of reuse; 1 otherwise, so it can gate a leg
rather than merely describe one.
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
#: THE SILENT-DEGRADATION SIGNAL. ``render_clip``'s ``finally`` runs
#: ``reclaim_idle_models`` on EVERY shot, which detaches every resident patcher
#: -- the cached CLIP included. ``_cached_clip_is_live`` catches a patcher left
#: unusable and degrades to a full reload, which is SAFE but silent: the render
#: is correct and the wall clock is exactly the pre-cache behaviour. Named here
#: so a never-hitting cache cannot read as a working one.
PLACEMENT_DROP = re.compile(r"dropping the cached text encoder")

_FIELDS = ("reads", "pinned", "renders", "hits", "misses", "drops")


def _blank(**extra):
    d = dict.fromkeys(_FIELDS, 0)
    d.update(extra)
    return d


def parse_scopes(path):
    """Split a server log into cache-scope intervals plus the unscoped rest.

    Returns ``(scopes, unscoped)``. Each scope carries its own counters and a
    ``closed`` flag; ``unscoped`` collects everything outside any interval,
    which is the legitimate ``render_single`` diagnostic path and is reported
    rather than gated.
    """
    scopes, unscoped, cur = [], _blank(), None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if SCOPE_OPEN.search(line):
                if cur is not None:          # an open with no matching close
                    scopes.append(cur)
                cur = _blank(closed=False)
                continue
            if SCOPE_CLOSED.search(line):
                if cur is not None:
                    cur["closed"] = True
                    scopes.append(cur)
                    cur = None
                continue
            bucket = cur if cur is not None else unscoped
            if TEXT_ENCODER_READ.search(line):
                bucket["reads"] += 1
            if ENCODER_PINNED.search(line):
                bucket["pinned"] += 1
            if SHOT_RENDER.search(line):
                bucket["renders"] += 1
            if CACHE_HIT.search(line):
                bucket["hits"] += 1
            if CACHE_MISS.search(line):
                bucket["misses"] += 1
            if PLACEMENT_DROP.search(line):
                bucket["drops"] += 1
    if cur is not None:
        scopes.append(cur)
    return scopes, unscoped


def audit(path):
    """Aggregate totals across the whole log. Descriptive only.

    KEPT FOR REPORTING, NOT FOR GATING -- summing across scopes is exactly the
    bug this instrument was fixed for. :func:`main` gates on
    :func:`parse_scopes`.
    """
    scopes, unscoped = parse_scopes(path)
    total = _blank()
    for part in list(scopes) + [unscoped]:
        for f in _FIELDS:
            total[f] += part[f]
    total["opens"] = len(scopes)
    total["closes"] = sum(1 for s in scopes if s.get("closed"))
    return total


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--expect-episodes", type=int, default=1,
                    help=("Minimum number of cache scopes that must appear "
                          "(default 1). Each scope is then gated on its OWN "
                          "events; scopes never share an allowance."))
    args = ap.parse_args(argv)

    scopes, unscoped = parse_scopes(args.log)
    print("cache scopes found        : %d" % len(scopes))
    for i, s in enumerate(scopes):
        print("  scope %d: renders=%d reads=%d pinned=%d hits=%d misses=%d "
              "drops=%d closed=%s" % (i, s["renders"], s["reads"], s["pinned"],
                                      s["hits"], s["misses"], s["drops"],
                                      s.get("closed")))
    print("unscoped (diagnostic)     : renders=%d reads=%d"
          % (unscoped["renders"], unscoped["reads"]))

    ok = True

    # POSITIVE EVIDENCE FIRST. This gate used to FAIL OPEN: its only check was
    # ``reads > expected``, so a log with zero matched lines passed -- regex
    # drift, a truncated log, or the wrong file entirely all read as clean and
    # would have been quoted as a receipt. A gate that cannot tell "proved
    # good" from "saw nothing" is worse than no gate.
    if len(scopes) < max(1, args.expect_episodes):
        print("\nFAIL: %d cache scope(s) found, expected at least %d -- the "
              "driver never reached the adapter's hooks, so nothing was "
              "exercised" % (len(scopes), max(1, args.expect_episodes)))
        return 1

    proved_reuse = False
    for i, s in enumerate(scopes):
        if not s.get("closed"):
            # THE ONE CHECK THAT CANNOT DISTINGUISH "leaked" FROM "still
            # running", because a log has no end-of-file marker. On a COMPLETED
            # leg this is the 8.86 GiB leak and it is invisible in any ratio;
            # on a leg still rendering it is simply the current scope. Say both,
            # so nobody reads a mid-flight audit as a failure -- this instrument
            # is for finished legs.
            print("FAIL: scope %d opened and never closed. On a FINISHED leg "
                  "that is the 8.86 GiB leak. If this leg is still rendering, "
                  "this is the live scope and the audit is premature -- re-run "
                  "it once the episode publishes." % i)
            ok = False
        if s["renders"] == 0:
            # Reachable: the driver opens the scope before building the
            # request, so a request-building failure closes an empty one.
            # Harmless in itself, but it must never grant an allowance.
            print("NOTE: scope %d rendered nothing (it grants no allowance)" % i)
            continue
        if s["reads"] > 1:
            print("FAIL: scope %d read the encoder %d times -- one scope, one "
                  "load" % (i, s["reads"]))
            ok = False
        if s["reads"] != s["pinned"]:
            print("FAIL: scope %d has %d GGUF read(s) but %d CLIP(s) "
                  "constructed -- a pattern has drifted and neither count can "
                  "be trusted" % (i, s["reads"], s["pinned"]))
            ok = False
        if s["reads"] == 0:
            print("FAIL: scope %d rendered %d shot(s) with no encoder read at "
                  "all -- the pattern matched nothing, so this proves NOTHING"
                  % (i, s["renders"]))
            ok = False
        if s["renders"] >= 2:
            if s["hits"] == 0:
                print("FAIL: scope %d ran %d renders and not one cache HIT -- "
                      "it is loading every shot" % (i, s["renders"]))
                ok = False
            else:
                proved_reuse = True
        if s["drops"] and s["drops"] >= s["renders"] - 1:
            print("FAIL: scope %d dropped its cached encoder %d time(s) across "
                  "%d render(s) -- correct renders, zero benefit, and silent"
                  % (i, s["drops"], s["renders"]))
            ok = False

    # ONE SHOT PROVES NO REUSE, and reuse is the entire claim.
    if not proved_reuse:
        print("FAIL: no scope rendered 2+ shots with a cache HIT -- nothing "
              "here demonstrates reuse")
        ok = False

    if ok:
        print("\nPASS: every scope loaded the encoder at most once and reused "
              "it thereafter")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
