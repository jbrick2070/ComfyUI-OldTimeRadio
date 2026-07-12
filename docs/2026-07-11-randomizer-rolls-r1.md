# Randomizer Rolls -- R1 (hardened via kibitz r1: codex gpt-5.6-sol@ultra + antigravity gemini-3.5-pro)

**Date:** 2026-07-11 (r1 pass complete; judgment log at
`kibitz-runs/2026-07-11-randomizer-rolls/r1/final.md`)
**Stage:** R1 hardened. R2 coding plan deferred until the repo coding window
is free. R1-only by operator directive.
**Operator ask:** a source bank randomizer and a visual pack randomizer.

---

## 1. The precedent (this pattern already ships)

The LEMMY cameo widget IS the house randomizer pattern: choices constant at
`nodes/OTR_LedgerScriptWriter.py:248-251`, widget definition at ~:2603-2624 --
a combo with a `roll (...)` entry, an OS-entropy draw at run time, and a
deterministic operator override.

**Clarification (r1):** LEMMY defaults to its roll; the source-bank widget
does NOT adopt that default. `science_news` stays the writer and canonical
default -- it is the byte-identical regression baseline. **The roll is
strictly opt-in.**

---

## 2. Design A -- source bank roll

**Widget.** The sentinel is PREPENDED in INPUT_TYPES (it is a UI command, not
a registry row):

    roll (any eligible bank)

No new widget: `widgets_values` positional law untouched. Caching is a
non-issue: the writer's `IS_CHANGED` returns `time()` (:3023-3028, always
re-executes), so a repeat queue with identical widgets still re-rolls.

**Eligibility preflight (r1 fold -- runnable != compatible).** The pool is
NOT "any runnable bank." Before the draw, filter to banks that:

1. are `runnable: true` and pass registry validation;
2. are compatible with the CURRENT request -- e.g. fable2's target_words gate
   (writer :3340-3356) and the all-custom-lanes refine rejection (:3357-3363):
   with refine active, only the two inline legacy pipelines are eligible;
3. carry rights compatible with unattended use: the v1 auto-roll pool
   EXCLUDES noncommercial-restricted banks (shakespeare, CC BY-NC). Direct
   manual selection is unaffected.

Filtering BEFORE the draw is legal; a failed bank AFTER the draw still fails
loud -- never silently re-roll.

**Resolution seam (r1 fold).** The writer's `run()` gates
`require_runnable_bank(source_bank)` before anything else, and refine
re-enters `run()`. Therefore: intercept the sentinel and resolve it to a
concrete bank id BEFORE `require_runnable_bank`; stamp the receipt; on refine
re-entry, detect the carried receipt and reuse the selected id -- one
submission = one roll, asserted by test.

**RNG (two-stage; SystemRandom cannot be seeded).**

- `OTR_BANK_SEED` set -> `random.Random(int(seed))` over the eligible list;
- absent -> `random.SystemRandom()`.
- Receipt: `meta["bank_roll"] = {requested, selected, seed, seed_source,
  eligible_order}` -- mirrors the existing cast-seed receipt pattern (build
  refs writer ~:1084-1104, ~:4023-4028). `resolved["source_bank"]` carries the
  selected id; HUD/credits come from the selected bank's defaults.

**Test surface (deterministic only -- no statistical trials).**
(a) sentinel present and FIRST in INPUT_TYPES choices -- update
`tests/test_source_bank_widget_2c.py:79` (asserts choices ==
list_bank_ids()) and the workflow guardrail exact-choice tests;
(b) seeded draw deterministic; (c) injected-RNG test proving the full
eligible tuple reaches an unbiased `choice`; (d) receipt stamped + selected
id is a real registry row; (e) refine re-entry reuses the receipt;
(f) empty eligible pool raises naming the filters that emptied it;
(g) CLAUDE.md section 0: OTR_WorkflowValidator + JSON round-trip + widget
audit in the same change.

---

## 3. Design B -- visual pack roll (UNBLOCKED by r1)

The 2026-07-05 "four disconnected surfaces" blocker is STALE: that machinery
is retired. `nodes/_otr_visual_styles.py` is now a single validated pack
registry -- packs live in `nodes/visual_styles/<style_id>.json`,
`list_style_ids()` (:362) enumerates, `resolve_visual_style` (:367) fails
closed on unknown ids, and the loader fails loud on a missing/invalid pack
directory (:324-351).

**Definition (r1 fold):** a "visual pack" is one of those prompt-look JSON
packs. It is NOT the per-role image/video engine selections owned by
OTR_VideoDirector.

Design B is therefore the SAME pattern as Design A, its own change, nothing
shared but the idiom:

- sentinel `roll (any style)` prepended on the visual_style combo;
- two-stage RNG with **`OTR_VISUAL_STYLE_SEED`** -- a NEW env var.
  `OTR_STYLE_SEED` is already taken: it steers narrative arc-shape
  (:4632-4647). Reusing it would let a visual control mutate story structure.
- roll once per episode, receipt in meta, resolved to a concrete style id
  before the existing style gate;
- still_word lettering law holds: the episode's rolled style is LOCKED for
  that episode; backdrop variation stays inside it.
- no Design-B placeholder code or widget values ship inside Design A's
  change.

---

## 4. Non-goals (both designs)

- No Python-authored story text; the roll selects, never writes.
- No new LLM slots, no third model.
- No word-count coupling (the eligibility preflight READS the request's
  target_words to test lane compatibility; it never trims, pads, or gates
  content).
- No per-bank/per-style weighting levers in v1.
- No auto-roll-eligibility capability flag (parked operator option; the
  `runnable` flag stays the curation surface).

## 5. Sequencing

1. Design A -> R2 coding plan via /kibitz when the coding window frees.
2. Design B -> R2 as its own change, same panel, after Design A lands.
