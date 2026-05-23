# Claude's Pre-Round-Robin Analysis -- Outline cast-drift crash

**Date:** 2026-05-23
**Written BEFORE seeing ChatGPT / Gemini.** This is the baseline to compare
their feedback against. Companion to
`2026-05-23-outline-cast-drift-problem-statement.md`.

---

## Decided recommendation (committed)

The bug is really **two independent problems** and each needs its own fix:

1. The creative LLM emits a beat speaker not in the locked cast.
2. When it does, the pipeline **hard-crashes** a 112-second run.

Fixing #1 perfectly would not excuse #2, and fixing #2 would not excuse #1.
So the answer is layered, in two tiers.

### Tier 1 -- do this now, regardless of the round-robin

Low risk, no architecture change, all inside `nodes/_otr_outline.py`. This
tier alone fully kills the crash and most of the adherence problem.

- **1a. Fuzzy snap inside `_phase_check`.** Before failing a beat for an
  off-cast speaker, try to match it to a cast name by normalized
  edit-distance (uppercase, strip punctuation/whitespace, Levenshtein <= ~2).
  If every off-cast speaker snaps, accept the beat with the snapped names.
  `LEMMEY` -> `LEMMY` then succeeds on attempt 1. Only genuine
  hallucinations (`CAPTAIN`) fall through.
- **1b. Deterministic fallback -- never crash on a cast-membership miss.**
  When the retry budget exhausts on a cast-membership failure, do NOT
  `raise OutlineFailedError`. Instead build the phase skeleton
  deterministically: assign the `phase_beat_count` beats to speakers by
  cycling `sorted(locked_cast)` (seeded). A 1-character cast -> every beat
  is `LEMMY`; multi-character -> round-robin. Zero LLM output needed, fully
  deterministic (safe for the seed/repro contract). The run completes with
  a valid -- if plainer -- outline.
- **1d. Fix the retry temperature schedule.** It currently goes
  0.70 -> 0.80 -> 0.30. Raising temperature on a constraint-adherence retry
  is backwards. Make it monotonically decrease, e.g. 0.70 -> 0.45 -> 0.25.
- **1f. Singleton-aware prompt.** When the locked cast has one character,
  the Stage 2 prompt should say it flatly: "Every beat's `speaker` must be
  exactly `LEMMY`." Trivial, removes ambiguity for the common 1-char case.

### Tier 2 -- the real prevention (recommended, but hold loosely)

- **2a. GBNF-constrain the Stage 2 `speaker` field to the locked-cast
  enum**, built dynamically from the locked cast each episode. The model
  physically cannot emit an off-cast name. This is the gold-standard
  prevention.
- **2e. Route Stage 2 (speaker assignment) as a `technical` pass**, not a
  `creative` one. It is structured name-to-beat assignment, not prose; the
  technical slot already owns the GBNF machinery and wants a steady low
  temperature, not "maximum chaos." Stage 1 (macro) and Stage 3 (beat
  intent/mood) stay `creative` -- those are genuinely creative.

## Why this split

- **#2 (the crash) is the urgent one.** A recoverable writer miss must never
  vaporize a 112 s run. Cast-membership is *always* deterministically
  recoverable -- the valid set is known -- so a crash there is never
  justified. Tier 1b is non-negotiable and I would ship it even if nothing
  else changed.
- **Tier 1 is enough to make the pipeline robust.** 1a converts the typo
  class into a pass; 1b guarantees completion; 1d/1f improve the odds the
  LLM just gets it right. After Tier 1, the only residual is *quality* --
  for multi-character episodes the deterministic fallback assigns speakers
  mechanically and loses the LLM's dramatic who-speaks-when judgment.
- **Tier 2 buys back that quality.** GBNF keeps the LLM making the dramatic
  choice while making an invalid choice impossible, so the Tier 1b fallback
  almost never has to fire. Its value scales with multi-character episodes
  (the normal case), which is why it is still recommended -- but it is a
  bigger change (two-model routing, the audit table, wiring tests), so it
  is Tier 2, not Tier 1.

## If forced to ship exactly one thing

Tier 1b -- the deterministic, no-crash fallback. It is the single change
that turns a total-loss crash into a completed run.

## Confidence levels (honest)

- **High confidence:** the two-problems framing; that the crash must be
  removed via deterministic repair (1b); that the retry temp schedule is
  backwards (1d); that fuzzy snap (1a) is correct and cheap.
- **Medium confidence:** that Stage 2 should be reclassified to a
  `technical` pass (2e). It is *mostly* mechanical, but "vary speakers for
  dramatic rhythm" is a thread of real creative judgment; an external view
  may argue it should stay creative-but-constrained.
- **Lower confidence -- I want the round-robin to pressure-test these:**
  - Whether GBNF (2a) is worth the implementation cost vs. relying on
    Tier 1 alone. A defensible position is "Tier 1 is sufficient; skip the
    GBNF complexity."
  - Whether GBNF constrained decoding is clean to apply on this stack
    (llama.cpp/transformers path, NF4-quantized, dynamic per-episode
    grammar) without quality or VRAM surprises.
  - Whether `gemma-4-E4B-it` is actually a better Stage-2 model than a
    constrained `Mistral-Nemo`, or whether model choice is a red herring
    once decoding is constrained.
  - Whether "maximum chaos" should simply be capped for the outline stage
    regardless -- a separate, blunt mitigation I did not put in either tier.

## What I expect the externals might say differently

- ChatGPT may favor "just make Tier 1 solid and skip GBNF" (simplicity).
- Gemini may push the model-choice angle harder (Mistral-Nemo unsuited to
  constrained structured output) or argue for capping outline temperature.
- Either may raise that the `_REPAIR_PROMPT_TEMPLATE` repair attempt should
  include the explicit allowed-cast list, not just the prior error -- a
  cheap improvement I under-weighted.

If they converge on "Tier 1 only, no GBNF," I would likely accept that for
now and file GBNF as a later quality improvement -- Tier 1 genuinely does
make the pipeline correct-and-robust; Tier 2 is a polish on multi-character
quality.

---

## Concrete implementation sketch (Tier 1)

In `nodes/_otr_outline.py`:

1. `_phase_check` -- before returning the "not in locked cast" error, run a
   `_snap_speaker(name, locked_cast)` helper (uppercase + de-punctuate +
   Levenshtein <= 2). If all beats snap, mutate `parsed` to the snapped
   names and return `None` (pass).
2. New `_deterministic_phase_skeleton(phase_name, phase_beat_count,
   locked_cast, seed)` -> `_PhaseSkeleton` -- cycles `sorted(locked_cast)`
   across the beat positions.
3. In `generate_outline`, the phase loop: if `_run_call_with_retry` returns
   `skeleton is None`, call `_deterministic_phase_skeleton(...)` and log a
   `WARNING` ("outline phase fell back to deterministic speaker
   assignment") instead of `raise OutlineFailedError`. Keep `raise` only
   for non-recoverable failures (e.g. Stage 1 macro total failure).
4. `_run_call_with_retry` -- temperature for fresh attempt `i` =
   `max(0.2, base_temperature - 0.25 * i)`.
5. `_build_phase_user_prompt` / `_PHASE_SYSTEM_PROMPT` -- when
   `len(locked_cast) == 1`, add the explicit single-speaker line.

Regression: add `tests/` coverage for `_snap_speaker` (typo snaps,
hallucination does not), `_deterministic_phase_skeleton` (beat count,
speakers all in cast, seed-stable), and a `generate_outline` test that
feeds an all-hallucinated Stage 2 and asserts it completes (no
`OutlineFailedError`) with a valid skeleton.
