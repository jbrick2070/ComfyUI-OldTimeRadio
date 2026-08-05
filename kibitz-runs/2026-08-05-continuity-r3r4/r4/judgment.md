# r4 JUDGMENT -- convergence + the discarded-objections audit

**Judge:** Claude (Opus 5), driver, sole judge. **Date:** 2026-08-05.
**Panel:** Codex `gpt-5.6-sol` (high) + Antigravity `Gemini 3.6 Flash (High)`.

## The audit: 19 rejections, 16 SOUND, 3 PARTIAL, 0 WRONG

Nobody had checked these before. Both panelists audited all nineteen with
`file:line` evidence and **converged on every verdict.** No discarded objection
turned out to be a wrongly-buried must-fix. The design survives its own rejections.

The three PARTIALs are documentation defects, not design defects:

- **D01 -- the cited rng tally is wrong.** The plan claims the allocator consumes
  "0 calls at count 0 and 1, 2 at 2 and 3, 4 at 4 and 5". I measured
  **0, 0, 3, 3, 9, 11** (getrandbits, over the single `rng.shuffle` at
  `nodes/_otr_casting.py:615`). Antigravity independently measured
  **0,0,3,3,9,11** -- identical to mine. Codex called the tally "unsupported and
  already contradicted". **Three independent measurements agree.** The
  *conclusion* is untouched and in fact better supported: count=0 and count=1
  leave the stream identical, every count >= 2 diverges, so the reviewers'
  `count=len(unpinned)` fix would indeed re-break parity. **Action: correct the
  sentence in the plan record; keep override-in-place. No design change.**
- **D02 -- the corpus census is stale.** The decisive facts stand (the vendor
  parses the roster at `scripts/otr_fetch_public_domain.py:324` before rebinding
  `body` to the sliced scene at `:325`; no Folger whole-play body is on disk), but
  the "64 prose bodies + exactly 1 sidecar" count no longer matches the tree.
  **Action: drop the numeric claim, keep the rejection.**
- **D16 -- ImageScale could not be verified from the review sandbox.** Codex could
  not reach live `/object_info`. This becomes a **verify-at-build** item for
  chunk F, not a blocker now.

**D13 -- the one that mattered.** All three original reviews unanimously demanded
editing canonical node 87, and the harden overruled all three. Both r4 panelists
independently rate the rejection **SOUND**, with the mapping confirmed at
`config/profiles/widget_mapping.json:23-30` and the runner loading canonical
before applying the profile at `scripts/otr_canonical_api_run.py:132-164`. The
canonical workflow is not edited. The unanimous panel was wrong and the harden was
right.

## NEW must-fixes r4 found (beyond r3's M1-M7)

**M8 -- the 42/42 corpus test is unsatisfiable.** (Codex.) Track 1 Step 3 says the
supplement covers 12 names, then asserts every one of the 42 shipped cast_hints
resolves to male|female -- while *also* leaving ARIEL and PUCK to the operator with
no gender and no evidence. No valid implementation can satisfy that test.
**Verified by my own corpus measurement:** exactly 12 hints are unresolved by the
sidecar roster, and ARIEL/PUCK are two of them, so **40/42 is the ceiling**.
**Fix: ship 10 supplement entries and assert 40/42, with ARIEL and PUCK asserted
explicitly unknown.** This would have failed the build on first run.

**M9 -- the Track 2 live command does not select Shakespeare.** (Codex.)
`--profile otr_w45_still_flat` changes engine roles only; it does not set
`source_bank`, `source_ref` or the word budget, and canonical node 1 still reads
`scifi_news`. The leg as written would prove face continuity on the wrong lane.
**Fix: pass the source bank / source_ref / words explicitly through the runner's
own controls, and preflight that the chosen source yields one character on >= 3
beats before spending the A/B.**

**M10 -- `OTRImageGenDispatcher` has no `IS_CHANGED` while gaining env-flag
dependencies.** (Codex.) `OTR_PORTRAIT_IDENTITY_SEED` / `OTR_PORTRAIT_REFERENCE`
change behaviour that ComfyUI's cache cannot see. Combined with r3's M7 (env vars
cannot reach a resident process), each A/B arm needs a **full reset + fresh boot**,
with source snapshot, prompts, seeds and portrait bytes held equal and asserted
equal before the arms are compared.

## Accepted SHOULD-FIXes

- **Replace the stale headline.** "94 ledgers / 188 rows / 23%" becomes the
  grounded **88 / 176 / 25%**, or simply "44 confirmed contradictions" with no
  volatile denominator. Codex raised it; it matches my own measurement.
- **Keep `_build_zimage_graph` pure.** Stage the reference file in `render_image`
  and pass the staged basename in params, rather than mutating the filesystem
  inside a graph constructor. Accepted -- it keeps the graph snapshot tests honest.
- **Remove cut steps from the builder-facing plan.** Executable descriptions of
  steps declared cut create two incompatible readings. Accepted; the cut list is
  recorded once, in this judgment.

## DISCARDED

- Codex's overall VERDICT "no". Its blocking items are M8-M10 plus r3's M1-M7,
  all of which are now folded into the build. With those folded the queue is
  buildable, which is what Antigravity concluded ("yes-with-fixes") on the same
  evidence. A verdict is not a finding.
- Codex CUT 2 (move the operator-recording ingestion out of this build). It is
  **already out** -- Track 3 Step 3 is not in the critic's ship order and this
  session never queued it. Recorded, not actioned.
- Antigravity SHOULD-FIX 2 (log male-collision rates during the suite). The
  collision it worries about is unreachable at the shipped `num_characters=2`:
  exhaustion needs more than 17 male speakers in one episode. Telemetry for a
  condition that cannot occur is noise. Not actioned; the reasoning is recorded
  in the r3 judgment's Q1 answer.

## VERDICT

**Converged.** Nineteen rejections audited, none wrong. Ten must-fixes total
(M1-M10) across r3 and r4, all folded. Build proceeds in the critic's ship order.
