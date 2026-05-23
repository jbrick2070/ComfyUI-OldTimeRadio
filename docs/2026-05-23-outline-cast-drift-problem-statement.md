# Problem Statement -- Outline cast-drift hard crash (OTR writer)

**Date:** 2026-05-23
**Purpose:** Round-robin consultation seed (ChatGPT + Gemini -> Claude synthesis).
**Status:** root cause understood; solution undecided -- this document is for the round-robin.

---

## 1. One-line summary

The OTR writer's outline stage hard-crashes the entire ComfyUI run when the
creative LLM assigns a beat speaker that is not in the locked cast -- a typo'd
name (`LEMMEY` for `LEMMY`) or an invented one (`CAPTAIN`) -- and the 3-attempt
retry loop exhausts. We need the best fix.

## 2. What OTR is (context for an external reader)

OTR (`ComfyUI-OldTimeRadio`) is a local, offline ComfyUI custom-node pipeline
that generates a complete short sci-fi audio drama (and video) from a daily
science-news headline. The first stage is the **writer**
(`OTR_LedgerScriptWriter`), which builds the script in phases: news interpret
-> style pick -> **cast lock** -> **outline** -> line composition -> announcer
passes. The outline stage is the one that fails here.

Platform: Windows, single RTX 5080 (16 GB VRAM), Python 3.12, torch 2.10 /
CUDA 13, ComfyUI 0.22.2. Hard constraints: 100% local, open-source,
offline-first -- no cloud APIs, no paid services. LLMs load 4-bit (NF4)
quantized under a 14.5 GB VRAM ceiling. The project already uses GBNF
grammars for structured ("technical") LLM passes.

## 3. The failure (exact)

Episode `pending_20260523_160946`, run 2026-05-23 16:08. Console:

```
[OTR_LedgerScriptWriter] cast locked: 2 rows (announcer + 1 characters, lemmy_hit=True)
[OTR_LedgerScriptWriter] phase 2A budget: act_count=1, arc_phases=['scene'], per_phase_beats=[3]
[OTR_Outline.macro] success on attempt 1/3
[OTR_Outline.phase[scene]] attempt 1/3: fresh (temp=0.70)
[OTR_Outline.phase[scene]] attempt 1 failed: phase 'scene' beat speaker 'LEMMEY' is not in locked cast ['LEMMY']
[OTR_Outline.phase[scene]] attempt 2/3: fresh (temp=0.80)
[OTR_Outline.phase[scene]] attempt 2 failed: phase 'scene' beat speaker 'LEMMEY' is not in locked cast ['LEMMY']
[OTR_Outline.phase[scene]] attempt 3/3: repair call (temp=0.30)
[OTR_Outline.phase[scene]] attempt 3 failed: phase 'scene' beat speaker 'CAPTAIN' is not in locked cast ['LEMMY']
!!! Exception during processing !!! Outline generation failed after 3 attempts.
Last error: phase 'scene' beat speaker 'CAPTAIN' is not in locked cast ['LEMMY']
  ... OTR_LedgerScriptWriter.py line 2045 -> _otr_outline.py line 1493
OutlineFailedError: Outline generation failed after 3 attempts.
Prompt executed in 111.58 seconds
```

The `OutlineFailedError` is uncaught -- it propagates out of the node's
`run()` and ComfyUI aborts the whole prompt. No audio, no video, no usable
ledger. The run is a total loss at ~112 s in.

## 4. Run configuration (likely a contributing factor)

- `creative_model = mistralai/Mistral-Nemo-Instruct-2407`  (changed this run; prior runs used `google/gemma-4-E4B-it`)
- `technical_model = google/gemma-4-E4B-it`
- `creativity = "maximum chaos"`  -> writer-level sampling temp 0.95, top_p 0.99
- `num_characters = 1`  -> locked cast is a single character: `['LEMMY']` (plus the non-character ANNOUNCER)
- Outline Stage 2 base temperature is 0.70 (separate knob from the writer's 0.95).

So the trigger combination is: **Mistral-Nemo creative model + "maximum chaos"
+ a one-character cast.** A high-temperature model planning a "scene" naturally
wants multiple speakers; with only `LEMMY` allowed it both misspells `LEMMY`
and invents `CAPTAIN`.

## 5. Code path (`nodes/_otr_outline.py`)

The outline is a 3-stage LLM pipeline:
1. **Stage 1 macro** -- title/premise/setting. Succeeded.
2. **Stage 2 phase** -- assign a `speaker` (ALL-CAPS name) to each beat of a
   phase. **This is where it failed.**
3. **Stage 3 beat** -- flesh out each beat (intent/mood/word target).

**Stage 2 prompt** (`_PHASE_SYSTEM_PROMPT` + `_build_phase_user_prompt`)
already instructs the model:
> "Use ONLY names from the Cast. Never invent."
and includes a Cast block rendered by `_format_cast_block` that says
"use exactly these names in character-role beats" and lists the cast.

**Stage 2 validator** (`_phase_check`, an `extra_check` callback in
`generate_outline`):
```python
for b in parsed.beats:
    if b.speaker not in locked_cast_set:
        return f"phase {phase_name!r} beat speaker {b.speaker!r} is not in locked cast ..."
```
Exact set-membership. `LEMMEY` and `CAPTAIN` both fail it.

**Retry loop** (`_run_call_with_retry`, `max_attempts = 3`):
- attempt 1: fresh, temp = `base_temperature` (0.70)
- attempt 2: fresh, temp = `base_temperature + 0.1*attempt_idx` (**0.80 -- higher**)
- attempt 3: "repair call", temp 0.30, user prompt = `_REPAIR_PROMPT_TEMPLATE`
  carrying the previous raw response + the validation error string.
- All 3 fail -> returns `(None, attempts)` -> `generate_outline` does
  `raise OutlineFailedError(...)` (`_otr_outline.py:1493`).

The schema-level `speaker` field only enforces `min_length=1, max_length=40` +
ALL-CAPS; it does NOT constrain the value to the cast. The cast constraint is
only the post-hoc `_phase_check`.

## 6. Analysis

Two distinct drift types appeared in one run:
- **`LEMMEY`** -- a near-miss TYPO of a real cast name (edit distance 1).
  A normalized / fuzzy match could recover this.
- **`CAPTAIN`** -- a fully HALLUCINATED character with no relation to the
  cast. Fuzzy matching cannot recover this.

Design observations worth weighing:
- The retry **raises** temperature on attempt 2 (0.70 -> 0.80). For a
  constraint-adherence failure, more randomness is the wrong direction.
- The "repair" attempt (temp 0.30, error fed back) still hallucinated
  `CAPTAIN` -- so even the repair path did not hold with this model.
- There is **no graceful fallback**: with exactly one character locked, an
  off-cast speaker is unambiguous and could simply be snapped to `LEMMY`.
  Instead the run hard-crashes. A writer outline miss should arguably never
  take down a 112-second run.
- The cast constraint is enforced **after** generation (validate-and-retry),
  not **during** it. The project already has GBNF-grammar infrastructure for
  structured "technical" passes; Stage 2 (pure speaker-to-beat assignment) is
  arguably a structured task, yet it currently runs as a "creative"-slot call
  on the creative model with no grammar.
- Prior related defect: **BUG-LOCAL-233** -- "vocative drift" on the
  Mistral-Nemo writer (the model drifting on character names in dialogue).
  Mistral-Nemo has a known tendency to drift on names in this pipeline.

## 7. Constraints any solution must respect

- 100% local / offline / open-source. No cloud, no API, no paid services.
- VRAM ceiling 14.5 GB; LLMs are NF4-quantized; model swaps between the
  creative and technical slots already cost full teardown/reload time.
- **Audio is king** -- the audio path output must stay byte-identical to its
  baseline; any fix must not perturb it.
- Every LLM call is tagged `creative` or `technical` and routed to the
  matching model slot (project Prime Directive #6).
- Determinism matters: the pipeline is seed-driven for reproducibility (C7
  byte-identity contract downstream).

## 8. Open questions for the round-robin

1. What is the best primary fix -- constrained decoding (GBNF grammar /
   logit mask restricting `speaker` to the locked-cast enum), fuzzy
   speaker-name recovery, a graceful deterministic fallback, or a
   combination?
2. Should outline **Stage 2 (speaker assignment)** be reclassified from a
   `creative` pass to a `technical` pass -- structured JSON, gemma technical
   model, GBNF-constrained -- given it is name-assignment, not prose?
3. Should the pipeline ever hard-crash on an outline miss at all? What is the
   right graceful degradation (snap-to-cast, deterministic round-robin
   assignment, skip-the-run-cleanly)?
4. Is the retry temperature schedule backwards? Should retries DECREASE
   temperature for constraint failures?
5. Is "maximum chaos" + `num_characters=1` + Mistral-Nemo simply an unsound
   combination? Should "maximum chaos" be capped for the outline stage, or
   single-character casts be special-cased?
6. Anything model-specific: is Mistral-Nemo-Instruct-2407 a poor fit for
   constrained structured assignment vs gemma-4-E4B-it?

## 9. Candidate solution directions (to evaluate, not predetermined)

- **A. Constrained decoding** -- GBNF grammar (or logit-bias mask) so the
  `speaker` field can only emit a name from the locked-cast enum. Makes the
  failure physically impossible. Project already has GBNF infra.
- **B. Fuzzy speaker recovery** -- normalize + edit-distance match each
  emitted speaker to the cast before validating; snap near-misses
  (`LEMMEY` -> `LEMMY`). Does not help pure hallucinations.
- **C. Graceful fallback, no hard crash** -- after retries exhaust (or
  immediately, for a singleton cast), deterministically assign beat speakers
  from the locked cast instead of raising `OutlineFailedError`.
- **D. Fix the retry temp schedule** -- decrease temperature on each retry
  (e.g. 0.70 -> 0.40 -> 0.20) instead of increasing it.
- **E. Reclassify Stage 2 as a technical pass** -- route speaker assignment
  to the technical model + GBNF, off the creative slot.
- **F. Stronger / singleton-aware prompt** -- when the cast has one
  character, state explicitly: "Every beat's speaker must be exactly LEMMY."

A likely synthesis is A or E (make it impossible) + C (never crash) + D
(free win). The round-robin should confirm or correct this.

---

## Appendix -- key source locations

- `nodes/_otr_outline.py`
  - `generate_outline()` -- ~line 1345; raises `OutlineFailedError` at ~1493.
  - `_run_call_with_retry()` -- ~line 1120; retry + temp schedule.
  - `_phase_check()` cast-membership validator -- ~line 1463.
  - `_PHASE_SYSTEM_PROMPT` -- ~line 964; `_build_phase_user_prompt` ~1015;
    `_format_cast_block` ~510; `_REPAIR_PROMPT_TEMPLATE` ~551.
- `nodes/OTR_LedgerScriptWriter.py` -- calls `generate_outline` at ~line 2045.
- Prior related: BUG-LOCAL-233 (vocative drift, Mistral-Nemo writer) in
  `BUG_LOG.md`.
