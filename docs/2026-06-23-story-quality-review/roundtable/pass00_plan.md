# Story-Quality LIFT (post-R3) -- Improvement Plan v0 (seed for the roundtable)

**Goal:** make the OTR "Signal Lost" episodes genuinely BETTER STORIES on a WEAK LOCAL writer, WITHOUT adding
another QA/critic flag-and-reroll gate (the operator's explicit constraint; the R3 spine proved such gates
inert and the panel unanimously agrees a weak model just regenerates the same scene).

**Evidence base:** the 18-episode R3 flag-ON soak (`../STORY_REVIEW.md`) + the 4-model story critique
(`passA_STORY_CRITIQUE_SYNTHESIS.md`, GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro + Grok-4.3, all 4 converged).

---

## The core finding that reframes everything (grounded against the code)

OTR has ALREADY BUILT soft-prompt versions of nearly every fix the panel proposed, and the weak models IGNORE
them. Confirmed in the real source:

- `_otr_outline._build_beat_user_prompt` (nodes/_otr_outline.py:1166) already instructs: *"The intent MUST be an
  ACTION UNDER PRESSURE ... RAISE THE STAKE ... escalate, never tread water ... KEEP STANCE CONSISTENT ... a
  reversal ... ONLY as a deliberate turn this beat earns."* These are the exact behaviours the soak FAILED at.
- The beat-intent verb menu in that prompt is itself threat-biased: *"reveal, refuse, demand, bargain, accuse,
  conceal, choose, threaten"* -- it actively steers the weak model toward the standoff.
- `_otr_line_composer._build_user_prompt` (nodes/_otr_line_composer.py:1065) already ships full CAST voice
  cards, a DRAMATIC FRAME (Objective/Obstacle/Turn/Subtext/Tension), CONTINUITY CONSTRAINTS, `speech_signature`,
  and the L2 objective-deflection. The output is still interchangeable threat-noise.
- Arc phases already exist as a DETERMINISTIC skeleton (`EpisodeBudget.arc_phases` setup/complication/...,
  per-phase word + beat budgets, `arc_phase` monotonic validators) -- but a phase is a WORD/BEAT BUDGET, not a
  DRAMATIC FUNCTION. Nothing forces a reversal beat, a personal-stake beat, or an on-stage climax.
- `allowed_things` / `allowed_people` (brief-derived entities) already exist in the composer prompt -- a
  premise-specific conflict palette could be seeded from them with no new extraction.

**Conclusion:** re-instructing the model (more "escalate / be distinct / show don't tell") will add nothing --
it is already there and ignored. The only moves that will work are DETERMINISTIC and UPSTREAM (in the Python
that builds the beat skeleton + the prompts), or model-agnostic SELECTION (best-of-N), not model self-policing.

---

## Hard constraints (every fix must satisfy)

1. NO new flag-and-reroll critic/QA gate. (Operator + unanimous panel.) Deterministic regex sanitizers are OK
   (they repair, they do not ask the model to retry).
2. Works on a weak/small LOCAL model (mistral-nemo / gemma-12b class) -- prefer DETERMINISTIC negative
   constraints + required structural slots over positive style nudges.
3. Content-only: ledger schema `l3-2026-05-14` FIXED (new values ride free-form `meta` / `compose_flags`); ZERO
   `workflows/otr_scifi_16gb_full.json` change unless a node/widget is genuinely added (then same-commit).
4. Audio spine FROZEN: `test_audio_byte_identical` stays green; anything that changes generated dialogue is
   gated behind a flag (default-off) or is a deliberate, operator-gated golden re-baseline.
5. Deterministic / seed-keyed; 100% local/offline; UTF-8 no BOM; SFW.

---

## The lever set (ranked; UPSTREAM + deterministic first)

### L1 -- Crisis-noun denylist + brief-derived premise palette in the beat planner (TOP)
The single most-converged panel lever, and it attacks the root (sameness). Two parts, both deterministic:
- (a) **Denylist** the generic-crisis vocabulary in `_build_beat_user_prompt` AND as a post-outline check:
  override, purge, lever, console, lockdown, core, vent, scrubber, countdown, manual control, switch, drive,
  keycard. Cap such beats at <=2/episode; on exceed, regenerate THAT beat's intent once with the denylist
  reinforced (this is a beat-INTENT regeneration at outline time, not a line critic-reroll).
- (b) **Palette:** derive an allowed conflict-object/action list from the news brief (reuse `allowed_things` /
  the meta brief) and inject it into the beat prompt so the planner dramatizes the ACTUAL premise (classroom:
  lesson plan / parent board / demo; legal: injunction / leaked memo / testimony; astronomy: observation time /
  peer review / instrument).
- Also: de-bias the beat-intent verb menu (drop the threat-heavy default list or make it premise-conditioned).
Why weak-model-robust: small models obey explicit NEGATIVE token constraints even when they ignore positive
style advice.

### L2 -- Phase = dramatic FUNCTION, with required non-standoff beats (UPSTREAM, deterministic skeleton)
Upgrade the existing deterministic phase skeleton so a phase enforces a FUNCTION, not just a budget:
- Add two required beat SLOTS to the skeleton: a **personal-stake / relationship beat** (before any
  irreversible action) and an **on-stage climax/decision beat** (the decisive action DRAMATIZED with a sensory
  consequence -- not narrated by the announcer outro).
- Make the final phase's last voiced beat the climax by construction; make the announcer outro reference that
  on-stage choice (template: "Because X chose Y, [news outcome], but [cost] remains").
Why weak-model-robust: a required slot in the Python skeleton is enforced regardless of model compliance.

### L3 -- Action/dialogue split at the composer (kills narrated-action + meta-leak deterministically)
Have the composer emit `{internal_action, spoken_dialogue}` (or an equivalent delimited form) and send ONLY
`spoken_dialogue` to the ledger/TTS. This deterministically removes "Jettisoning module, bracing for impact."
and the EP18 director-note leak WITHOUT a reroll. Replaces the operator's candidate (a) and subsumes L7.
Byte-identity: composer output shape is audio-affecting -> behind a flag / deliberate re-baseline.

### L4 -- Deterministic transcript sanitizer (hygiene, regex, NOT an LLM gate)
Hard-repair/strip prompt-leak tokens ("voice should", "tone", lowercase "announcer:" inside a character line)
and unbalanced quotes at freeze. Pure regex; no model judgment. (Verify the real-ledger encoding too -- a
mojibake instance in the panel packet was a packet-build artifact, not confirmed in the ledger.)

### L5 -- Writer default = gemma-12b + fix the critic "too_many_edits" abort that hides it (free win)
The soak's best 3 episodes (the only "strong" arcs) were all LOCAL gemma-4-12b; frontier grok underperformed
it. Make gemma-12b the default creative writer. AND fix the critic harness bug where the richest gemma outputs
hit `freeze=too_many_edits -> arc="?"` and are never graded (it penalizes the best writer). This is the correct
re-scope of the operator's candidate (c): "pick the best writer we already have," not "buy a frontier API."

### L6 -- (optional, secondary) best-of-N line selection with a flatness scorer (operator candidate b)
Model-agnostic: sample N candidate lines for the highest-tension beats, score flatness deterministically (reuse
the critic's flat-dimension definitions as a pure scorer), keep the least-flat. NOTE: best-of-N lifts
LINE-level flatness but does NOT fix the structural sameness (if all N are the same standoff, selection cannot
help) -- so it is a polish lever AFTER L1/L2, not the primary fix, and it costs N generations. Decide after
L1/L2 land.

---

## What we will NOT do (unanimous panel + operator)
- No new flag-and-reroll critic gate (weak model regenerates the same scene).
- No more soft style nudges in the prompts (already present, already ignored).
- No "just hit 883 words" -- length is a SYMPTOM of the structural collapse; fixing L1/L2 lengthens naturally.
  Over-length incoherence is worse (panel-confirmed: a 430w episode was still overstuffed standoff).
- No "swap to a frontier writer" as the primary fix (planner is the constraint; grok < local gemma-12b here).

## Build order (proposed)
L5 (free, immediate) -> L1 (top structural) -> L2 (phase function) -> L3 (composer split) -> L4 (sanitizer) ->
then evaluate L6. Each chunk: full suite + Bug Bible green, no-drift JSON assert, flag-gated where audio-affecting,
commit + push. Re-soak a small matrix (gemma-12b + 1 other, 1 tier) after L1+L2 to measure sameness reduction.

## Open code-verify items (for the wiring round)
1. Where the beat-intent verb menu + denylist inject in `_build_beat_user_prompt`; how to add a post-outline
   crisis-noun cap that regenerates a beat INTENT (not a line) without a critic loop.
2. The arc_phase skeleton builder (where `EpisodeBudget.arc_phases` + per-phase beats are assembled) -- where to
   add the required personal-stake + climax slots.
3. Composer output contract + the freeze/scrub path -- feasibility + blast radius of the action/dialogue split.
4. The critic `too_many_edits` abort path (why gemma's long outputs go ungraded).
5. The writer-default selection point (make gemma-12b default) + `meta.story_quality` telemetry under-count
   (EP16 had objective_literal_retry stamps but l1_rerolls=0).
