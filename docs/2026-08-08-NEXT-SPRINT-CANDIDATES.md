# NEXT-SPRINT CANDIDATES -- post queue item 8 (2026-08-08)

**Written 2026-08-08 at HEAD `3ebadbf1`** (item 8 shipped ~15 min ago). Operator
asked mid-session: "ask Fable how to move forward, or ask Codex via
`/kibitz-plugin:kibitz` plan for the next logical plan so we can get closer to
the lean-mean plan." This doc is the input the panel picks from (or improves).

**Constraint from the operator's standing rules:**

* Story quality is DONE (08-04). No writer/prose work.
* Two-strikes-then-panel (07-14). Every coding item gets a full kibitz arc
  first (08-04). This doc is the r1 input for the picked sprint.
* Immediate push after every green commit (07-10). One-atomic-commit
  scope preferred.
* No touching operator-dirty paths.
* Lean-mean is the STRATEGIC goal (`docs/2026-07-10-lean-mean-rip-final.md`);
  the operator's actual language is "closer to the lean-mean plan."

## State of the field (verified live at HEAD 3ebadbf1)

Blocked-on-operator items (do NOT pick):

* Item 1 chunk B: OpenRouter live leg -- needs `OPENROUTER_API_KEY` in launcher env.
* Items 3-7: operator eyeball / operator ruling / operator content.
* Item 9 chunk 3: Macbeth probe needs `COMFY_API_KEY` + Gemini keys in launcher.

Unblocked candidates (rank + why):

### CANDIDATE A -- Lean-mean re-ground + W0 (the STRATEGIC pick)

**Scope:** the operator has pinned `docs/2026-07-10-lean-mean-rip-final.md`'s
`W0 -> W1 -> ... -> W8` execution order as the shippability campaign. D-2
(RTXUpscale rip) was the codicil discharged in item 8. W0 is now unblocked.
The 07-15 drift-check header lists 5 stale areas that must be re-grounded
BEFORE any deletion, per the operator's r2-first pin: "The arc is the window's
first job, not a formality: if r2 says the kill list is wrong, the window's
output is a NEW r2, not a rip."

**What this sprint delivers:** an r2 arc against current HEAD for the lean-mean
plan (start with W0 -- the deletion-policy standing-law formalization + the
`ENGINE_MATRIX.md` generator precondition, per section 3's dependency edges),
producing a re-grounded r3-ready kill list. NO deletion; the sprint's output
is the arc itself.

**Panel:** per lean-mean's own rule, Claude codes/judges, kibitz = codex + agy;
Fable gets the SINGLE final gate on the lean-mean epoch commit only (CLAUDE.md
section 9 reality exception, not r1). That's a per-sprint override of the
08-06 Fable-r1 rule for THIS specific plan.

**Cost:** roughly one r2 → r3 → r4 arc + a re-grep + a re-survey. No render.

### CANDIDATE B -- Item-8 follow-up chip cluster (the DIRECT continuation)

**Scope:** four small chips owed from item 8's Fable final gate + Sonnet QA:

1. `SpandrelEsrgan._model_sha256` -- pin the actual Real-ESRGAN x2plus SHA
   printed by `scripts/ensure_upscale_models.py` (the user ran that
   PowerShell just now, so the SHA is either printed or about to be).
2. `IS_CHANGED` model-file block: swap the hardcoded engine name +
   filename for registry/engine metadata so engine #2 (whenever it lands)
   picks up its own fingerprint.
3. Stale RTXUpscale prose sweep: `nodes/video_engine.py:2086` tooltip +
   docstrings in `_otr_paths.py`, `_otr_memory.py`,
   `otr_post_upscale_procgen_blend.py` body comments.
4. Operator ruling on `meta.perfect_run_spacesaver` -- kept as no-op
   sentinel per Antigravity r2 MF-6, but the writer still stamps the
   flag with zero readers. Keep-or-retire is an operator call.

**What this sprint delivers:** ~50 LOC of cleanup across 4 files + one
tiny operator-pinned commit that closes item 8's residuals. Suite stays
9351/111/1 or better.

**Cost:** short. Fits inside one r2 → r3 → r4 arc if kibitz is per operator
directive; a case can be made for a scoped r3-only tail since the chips
are individually small.

### CANDIDATE C -- Slug-lists cleanup (the STILL-OPEN-SMALL #6 item)

**Scope:** widen the pattern chunk A shipped for OpenRouter
(`OPENROUTER_VERIFIED_ON_BY_ID` + a curation test) to every OTHER in-tree
slug list per the operator's 2026-08-07 explicit ask: Comfy LLM
(`_otr_comfy_backend.COMFY_LLM_MODELS`), Google API text models, audio
engines (ElevenLabs / Google TTS / Lyria), image engines (Nano / Seedream
/ Krea / Luma Photon / Google Image). Also fix the stale premise comment
at `_otr_comfy_backend.py:84-91` ("Reasoning models ... DELIBERATELY
EXCLUDED" -- 2026-08-07 signal-check found essentially every frontier
candidate now advertises reasoning including two slugs already in the
list).

**What this sprint delivers:** per-registry `<REG>_VERIFIED_ON_BY_ID`
frozenset + guard test file per registry (7 lists across 5-6 files),
same shape and same RED-on-mutation proof as the shipped OpenRouter
guard. Suite grows by ~10-15 tests.

**Cost:** medium. r2 → r3 → r4 arc, then implementation.

### CANDIDATE D -- Cloud-audio-cache SF#1 (yesterday's follow-up chip owed)

**Scope:** wrap the whole per-line voice loop plus `pack_audio_batch`
in one `try/finally` so a mid-loop `generate_voice` raise still persists
the completed lines' ledger stamps. Add `test_cache_on_multi_line_partial_
crash_stamps_completed_lines_via_finally` per the r4 spec.

**Cost:** short. Single-file change + one test.

## Recommendation

**CANDIDATE A (lean-mean re-ground + W0 arc)** is the operator's actual
"closer to the lean-mean plan" language taken literally, and every other
candidate is a chip-scale item that can pile up as follow-ups OR run
between lean-mean chunks. But: the operator's 08-06 Fable-r1 rule and the
07-24 pin that lean-mean itself runs Fable "single final gate only, not
r1" DISAGREE for this specific plan. r1 asks Fable + Codex + Antigravity
to break tie: does lean-mean's r2-first rule bind here (Fable = final
gate only), OR does the 08-06 standing rule override for THIS plan's r1?
r1 rules; driver disposes.

If lean-mean isn't picked, **CANDIDATE B** (item-8 chips) is the
smallest-scope forward step and pairs naturally with the
`ensure_upscale_models.py` run the operator is doing RIGHT NOW.

**Not recommending:** CANDIDATE C is legitimate but the operator's
"closer to lean-mean" phrasing points AWAY from it -- it is a wider
version of the OpenRouter chunk A that shipped last week, not a
lean-mean step.

## r1 questions for the panel

1. Which candidate best matches "closer to the lean-mean plan"?
2. Is lean-mean's Fable-final-gate-only rule intended to override the
   general 08-06 Fable-r1 rule for THIS plan's r1?
3. Any candidate you would ADD?
4. If A, does W0 have a single-commit shape, or does it need the
   ENGINE_MATRIX generator precondition addressed first?
5. If B, should the operator's pending PowerShell run for
   `ensure_upscale_models.py` produce the SHA and pin it before the
   arc opens, or is the SHA pin part of the arc itself?
