# Randomizer Rolls -- R1 problem statement + high-level design

**Date:** 2026-07-11
**Stage:** R1 (arc / design). R2 coding plan deferred until the repo window is free
(Codex currently driving 56SOL P0-to-green) and, for the visual roll, until the
style-surface decision lands.
**Operator ask:** "at some point we have a source bank randomizer function and
visual pack randomizer function."

---

## 1. The precedent (this pattern already ships)

The LEMMY cameo widget IS the house randomizer pattern
(`nodes/OTR_LedgerScriptWriter.py:248-251`):

- a combo with explicit choices: `roll (~11% chance)` / `always include` / `never include`
- the roll choice maps to an OS-entropy draw at run time
- the operator can override deterministically
- default = the roll

Both new randomizers should be instances of this pattern, not new machinery.

---

## 2. Design A -- source bank roll (small, unblocked)

**Widget.** The `source_bank` combo (values supplied by `list_bank_ids` at
INPUT_TYPES, writer ~line 128) gains ONE new allowed value:

    roll (any runnable bank)

No new widget is added, so `widgets_values` positional law (BUG-LOCAL-097) is
untouched -- the canonical workflow JSON stores whichever value the operator
selected, exactly as today.

**Resolution.** At run start, before ANY pass executes:

1. Candidate set = registry banks with `runnable: true` that pass registry
   validation (today: 10 banks; `custom_source_bank` excluded by its own flag).
2. Draw via `random.SystemRandom()` -- OS entropy, per the true-randomization
   law. Repro override: `OTR_BANK_SEED` env (mirrors `OTR_CAST_SEED` /
   `OTR_STYLE_SEED`; document alongside them).
3. Uniform weights in v1. Per-bank weights are a later, separate decision --
   do not build the weighting lever until someone asks for it (dead-lever law).

**Truthful receipts (the actual work).** The rolled bank must be stamped as the
REAL routing fact, not the widget string:

- `resolved["source_bank"]` = the rolled bank id (everything downstream already
  reads this -- 15+ call sites, no consumer changes).
- `meta` gains a small receipt: `{"bank_roll": {"requested": "roll", "selected":
  "<bank_id>", "seed_env": bool}}` -- consistent with the truthful-routing-
  receipts direction (`bfb8d81b`, `33574199`).
- HUD origin label / credits line come from the SELECTED bank's defaults,
  never from the roll sentinel.
- The roll happens once per queue submission and is frozen into the ledger
  meta; a re-run of the same episode re-rolls unless `OTR_BANK_SEED` pins it.

**Fail-closed edges.**

- Candidate set empty (all banks unrunnable) -> raise, name the registry state.
- Rolled bank fails story-rules resolution -> raise naming the bank; NEVER
  silently re-roll (a silent re-roll is a silent fallback).

**Test surface.** (a) sentinel present in INPUT_TYPES values; (b) seeded draw
deterministic under `OTR_BANK_SEED`; (c) unseeded draws hit >1 bank over N
trials; (d) receipt stamped + resolved id is a real registry row; (e) empty
candidate set raises.

**Size estimate:** one focused change in the writer + tests. Same-day.

---

## 3. Design B -- visual pack roll (BLOCKED, sequence it honestly)

**Blocker:** the style dropdown is four disconnected surfaces (2026-07-05
analysis: UI list, LLM prompt vocabulary, MusicGen terms, 100-entry catalog;
options A-D in that doc). A roll built today randomizes ONE surface and the
other three do not follow -- structured drift, worse than no randomizer.

**R1 position:**

1. First: operator picks a unification option from the 2026-07-05 doc
   (A-D). That decision is the real work.
2. Then: the roll is the same trivial pattern as Design A -- a
   `roll (any style)` combo entry, SystemRandom over the UNIFIED catalog,
   `OTR_STYLE_SEED` as the existing repro override (already read at writer
   ~line 4639), receipt stamped in meta.
3. Constraint carried forward: still_word lettering is per-episode LOCKED
   (operator 2026-07-04) -- the roll fires once per episode and freezes;
   backdrop variation stays inside the episode's locked style.

**Do not** build Design B's widget before the unification lands, even though
it is easy -- easy-and-wrong is how the four surfaces happened.

---

## 4. Non-goals (both designs)

- No Python-authored story text; the roll selects, it never writes.
- No new LLM slots, no third model.
- No word-count coupling.
- No per-bank/per-style weighting levers in v1.

## 5. Sequencing

1. Design A: R2 coding plan via /kibitz when the repo window frees (arc
   routing: mechanical rounds go to the local panel).
2. Design B: parked behind the style-surface decision; revisit the 2026-07-05
   doc with the operator, then fold the roll into that implementation.
