# My Story treatment prompt clarification

The supplied incident recovered on its existing lower-temperature retry. The
operator subsequently supplied the logs, and the exact focused excerpt is
archived here. It corrects the initial report: repetition already starts inside
the open logline; the captured failure does not show a transition into
dramatic_question. This change clarifies the two field roles that the prompt
left asymmetric. It does not establish that overlap caused the loop or that
the new wording eliminates stochastic repetition.

Observed in the original run: Qwen/Qwen3.8-27B NF4; attempt 1 at 0.850;
69-token cycle; halt at 288 tokens after 61.7 s; retry at 0.500 completes
849 tokens with EOS 248046. The writer completes The Extra Bowl with 49
ledger lines / 945 words. The cleaned full log contains no final publication
or Prompt executed marker. No post-change live model run is claimed.

Task list:
- [x] Read project lessons, relevant production history and Bible 12.100.
- [x] Trace shipped pack resolution, treatment schema and retry/repair delivery.
- [x] Clarify the existing pack instructions; keep the worked example and schema.
- [x] Scoped regressions: 342 passed (18.81 s).
- [x] Bug Bible: 24 passed, 36 skipped, 3 xfailed (2.27 s).
- [x] Base/retry/typed-repair delivery: real caller, synthetic halts, unchanged
  accepted artifact; temperatures [0.85, 0.5] and [0.85, 0.5, 0.1].
- [x] Pack and canonical JSON round-trip; only treatment instruction changed.
- [x] Full suite completed in 789.437 s, exit 2, 37 failing nodeids.
  All 37 were rerun twice on fixed HEAD 19958a92: with the original pack
  from 0566e30e, then with d11c31c3's pack restored byte-for-byte. Each
  run had the same 32 failures; five import-isolation failures from the
  full suite passed in both focused reruns. No introduced failing nodeid.
  Exact sets and JUnit totals are in failure_comparison.json.
- [x] Code pushed as d11c31c3; HEAD matched origin/main immediately afterward.
- [x] Composer 2.5 followed by Sonnet (CLI alias, high): no implementation
  blocker. These were two finished-diff reviews, not a four-round arc.
- [x] Named-file receipt commit prepared; pack restored to its verified hash,
  evidence copy hash matches the supplied log, JSON parses and diff check passes.

Runtime scope: every My Story treatment using the shipped pack on Windows,
Mac or pod. The existing graph resolves this pack dynamically; no new node or
widget is introduced and no canonical-workflow edit is required. Sampling,
schema constraints and the liveness guard remain unchanged.

Review judgment: Composer's baseline-tense ambiguity and incomplete check
description are fixed in driver_anchor.md. Its suggested substring snapshot
was declined: existing runtime/repair checks plus the delivery probe cover
transport without freezing prose wording in a new test. Sonnet's missing
same-field precedent is confirmed: PBUG-20260910-01 pairlock_03/04 documents
logline loops and declines an unproven title-only cure. That precedent is now
cited alongside PBUG-20260910-05 and Bible 12.100. The newly supplied artifact
supports an addendum to the existing PBUG, not a new bug ID or Bible rule.
The raw reviews retain their pre-log-delivery knowledge state.

Full-suite limitation: unrelated shared-path edits were committed as 19958a92
while the full suite was running. That run is not a pristine same-commit
baseline or a green full-suite qualification. The subsequent 37-test
original/candidate comparison held all other code fixed at 19958a92 and
compared failure identities, not just counts. The five full-suite-only
failures remain unclassified ordering/environment findings; no unrelated
test expectation or production code was changed to hide them.

Evidence SHA-256:
- verbatim_cycle_excerpt.log (exact supplied 126-line copy):
  ecfcb8e95041a5e19c643a0ea4a1808d7c99b8a31970437a0da5727ea5dd5984
- C:/Users/jeffr/Documents/otr_server_pod_full.log (read only, not copied):
  a30c2e5fa9c55eeefe913f6f75d946e14df371ed86dab9f684745224d545025d
- workflows/otr_canonical.json (unchanged, local CRLF bytes):
  2ab6bcdcf6c920d80ef72a6a9fbf9559ec4e6a17db92ef3b04636006c2f44fef

This is prompt hardening, not live model qualification. No new render was
started; the operator's running episode and its monitor were not altered.
