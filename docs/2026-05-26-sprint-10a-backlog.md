# Sprint 10A — Backlog Tracker

**Date:** 2026-05-26
**Status:** Live. Updated as items land or new items surface.
**Parent plan:** `docs/story-generator-final-plan.md`

Step-numbered work landed in commits, plus emergent items found during
shadow-pass soak. Use this file to keep track of items the sprint
plan did not originally name.

---

## Step status

| Step | Status | Commit(s) |
|------|--------|-----------|
| 1 — Rubric | DONE | `c5cb09c` |
| 2 — Slot reconciliation | DONE | `2d8b599` |
| 3-A — Stage 1 schema | DONE | `cde7e4f` |
| 3-B — Constrained-decode call site | DONE | `e46bd2f` |
| 3-C — Shadow pass wired into writer | DONE | `fcbeadf` |
| **3-D — Stage 1 prompt: length bound** | PENDING | scheduled after step 4 |
| 4 — Cast audit | IN PROGRESS | — |
| 5 — Stage 3 validators | PENDING | — |
| 5+ conditional — numeric-range LogitsProcessor | PARKED | depends on 3-D soak |
| 6 — Stage 2 multi-turn roleplay | PENDING | — |
| 7 — Whole-episode critic | PENDING | — |
| 8 — Operator E2E gate | OPERATOR | — |
| 10A-LAB — A1 vs A2 listen test | OPTIONAL | — |

---

## Step 3-D — Stage 1 prompt tightening for length_target_words bound

**Surfaced by:** Shadow-pass soak episode `pending_20260526_114111`
(2026-05-26 11:41). Both Stage 1 attempts hit
`beats.N.length_target_words = 0` violating the schema's `ge=5` bound.

**Root cause:** lm-format-enforcer's `JsonSchemaParser` enforces
structure + type + Literal + regex at the token-sampling layer but
does NOT enforce numeric `minimum`/`maximum`. The LLM emitted JSON-
shape-valid output with `length_target_words: 0`; pydantic
post-validator caught it. Failure status was `parse_failed` for both
attempts, the whole pass exhausted, pipeline continued unaffected
(PD1 held).

**Fix shape:**
  * Add explicit bullet to the Stage 1 system prompt in
    `nodes/_otr_stage1_call.py::_STAGE1_SYSTEM_PROMPT`:
    `length_target_words MUST be an integer between 5 and 200
    inclusive. Never emit 0 or negative.`
  * Keep retry temperature at 0.35 (per the 2B principle).
  * Schema unchanged (`Stage1Beat.length_target_words: int = Field(...,
    ge=5, le=200)` stays as the post-validator backstop).

**Why prompt-layer first, not LogitsProcessor:**
Per operator directive 2026-05-26: do not speculatively build sampler-
layer enforcement when prompt-layer might be sufficient. Prompt
tightening is a 10-line change; LogitsProcessor is significant new
code. Test the cheaper fix first.

**Acceptance:**
  * Land in a small commit between step 4 and step 5.
  * One soak run after with shadow-pass ON; confirm
    `attempts[0].status == 'valid_first_attempt'` (or at minimum,
    no `parse_failed` traced back to `length_target_words` bound)
    before counting more 19/20 data points.
  * If soak shows the prompt fix did NOT clear the failure mode
    (i.e. the LLM still emits zero or out-of-range values despite
    the explicit instruction), promote the conditional step-5
    numeric-range LogitsProcessor work.

**Test:** Adding a `tests/test_stage1_prompt_length_bound.py`
source-level test that asserts the bullet is present in the system
prompt.

---

## Step 5+ conditional — Numeric-range LogitsProcessor

**Parked.** Only build if step 3-D's prompt tightening fails to get
shadow-pass above ~17/20 first-attempt valid plans in soak.

**Why parked, not killed:** Token-level numeric range enforcement is
the correct architectural fix; lm-format-enforcer's gap is a known
limitation. But prompt-layer instruction is the cheaper test and may
be sufficient on Mistral-Nemo at the temperatures we use. Don't
build the heavier mechanism speculatively.

**Scope (if needed):**
  * Custom `LogitsProcessor` in `nodes/_otr_constrained_generate.py`
    or a sibling module.
  * Chains with lm-format-enforcer's `prefix_allowed_tokens_fn`.
  * Tracks the in-flight numeric field via parser-state introspection
    OR a simpler regex-on-decode-context approach.
  * Rejects digit tokens that would produce an out-of-range value
    given the schema's `minimum` / `maximum`.

---

## Step 4 fixtures — named regressions from real runs

**Operator directive 2026-05-26:** real-world bugs from real soak
runs are worth more than synthetic test data. Capture each one as a
named regression fixture in step 4's cast-audit test file.

Current list:
  * **ROBINSON VOSS → female** — episode `pending_20260526_114111`
    (2026-05-26 11:41). The LLM cast 'ROBINSON VOSS' as gender=female
    with voice=v2/en_speaker_7 (timbre=sharp, role=foil). Robinson is
    conventionally a male-coded surname/first-name combo. Step 4 cast
    audit's deterministic name -> gender lookup must catch this and
    either repair or flag.
  * Prior session: Cole=female, Mira=male (run 2 from previous handoff)
    — also Reginald=female (run 1) — same family. Capture as fixtures.

---

## Open questions

  * Will Mistral-Nemo respect the step 3-D prompt bullet, or does
    the issue happen because the LLM doesn't understand that
    `length_target_words: 0` violates the explicit range it was
    told? Empirical via soak.
  * Cast-audit repair vs regenerate: when the audit catches a
    mismatch, does step 4 repair the plan in place (swap the wrong
    gender + matching voice) or regenerate the whole plan from
    Stage 1? Lean toward repair as the lower-cost option; spec
    in step 4.

---

**End of backlog.**
