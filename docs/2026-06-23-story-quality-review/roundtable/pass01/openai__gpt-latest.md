<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan has a plausible diagnosis but its proposed “no critic/reroll gate” constraint is contradicted by L1 and L6, and several core levers assume unverified infrastructure or evidence that is explicitly outside the provided grounding.

MUST-FIX BEFORE BUILD:

1. [L1 / Hard constraints #1] The plan says “NO new flag-and-reroll critic/QA gate,” but L1(a) proposes “on exceed, regenerate THAT beat’s intent once with the denylist reinforced.” That is still a generation retry loop triggered by a post-outline quality check, just renamed “beat-INTENT regeneration.” Concrete fix: replace regeneration with a deterministic repair strategy: either reject crisis nouns during intent construction before generation, hard-cap/crush offending tokens after outline generation, or deterministically substitute premise-palette terms. If retry is retained, revise the constraint to allow bounded outline-level retries and define why this is acceptable.

2. [L6 / Hard constraints #1 / What we will NOT do] L6 proposes “best-of-N line selection with a flatness scorer.” This is model-agnostic selection, but it is still a multi-generation quality-selection gate and conflicts with the anti-reroll arc of the document. The distinction between “critic gate” and “pure scorer” is not enough because the operational shape is still generate several, score, keep one. Concrete fix: move L6 out of the main improvement plan into a separately approved experiment, or explicitly amend the constraint to permit deterministic best-of-N selection while forbidding LLM-judged rerolls.

3. [The core finding / L1] The plan’s main narrative says soft prompts are already ignored, so only deterministic upstream constraints will work. But L1(b) primarily says to “inject” a brief-derived palette into the beat prompt, and L1 also says to “de-bias the beat-intent verb menu.” Those are prompt-shaping changes unless the planner is structurally forced to select from the palette. Concrete fix: make the palette a required slot in the beat skeleton/intent object, e.g. every beat intent must carry `conflict_object` selected deterministically from allowed brief entities, and prompt text may only verbalize that slot. Do not sell prompt injection alone as deterministic enforcement.

4. [L2] “Phase = dramatic FUNCTION” is underspecified at the concept level. The plan names two required slots, but does not define how they map to episode length, number of beats, speakers, existing phase boundaries, or what happens when the budget is too small. A “personal-stake beat” and “on-stage climax” can easily become labels without enforcement. Concrete fix: define a minimal phase contract before build: required slots, allowed phase positions, required fields, validation criteria, fallback behavior, and how the final voiced beat is selected. Example: `beat_role in {setup, pressure, personal_stake, irreversible_choice, consequence}` with exactly one `personal_stake` before first irreversible slot and exactly one final `climax_decision` in last voiced beat.

5. [L3 / Hard constraints #3-4] The action/dialogue split is a large contract change disguised as a simple hygiene fix. It changes composer output shape, freeze/scrub behavior, ledger/TTS input assumptions, and possibly existing tests. The plan says ledger schema is fixed and audio spine frozen, but L3 introduces a new structured composer output and only vaguely says “behind a flag / deliberate re-baseline.” Concrete fix: define the compatibility layer: where `internal_action` lives if ledger schema cannot change, whether it is discarded or stored in `meta`, how legacy composer output is accepted, and what exact flag gates the new path.

6. [L5] “Writer default = gemma-12b” is treated as a “free win,” but the plan’s own thesis says the planner/structure is the binding constraint, not the writer. It also rests on soak observations from files not provided here. Concrete fix: separate default-model selection from structural story repair. Make L5 an experiment or operator config change after verifying the missing evidence and default-selection path. Do not put it first in build order unless the source and eval data prove it has no blast radius. [ASSUMPTION] This depends on verifying the soak and config code.

7. [Evidence base / The core finding] Several high-level claims are presented as grounded against code and external review files, but the only supplied material is the plan itself. Claims about `_otr_outline._build_beat_user_prompt`, `_otr_line_composer._build_user_prompt`, `EpisodeBudget.arc_phases`, soak results, and model rankings cannot be verified from the provided grounding. Concrete fix: either include the cited source excerpts and eval tables in the spec or downgrade these to “verify” items. As written, the build plan overstates its evidentiary certainty.

SHOULD-FIX:

1. [L1] The crisis-noun denylist risks fighting the genre/premise rather than sameness. Terms like “console,” “manual control,” “drive,” “switch,” and “core” may be legitimate in science-fiction OTR. The plan gives no distinction between lazy generic usage and premise-valid usage. Concrete fix: make the cap contextual: count only ungrounded crisis nouns not present in the brief-derived palette or allowed entities, and allow explicit per-premise exemptions.

2. [L1] “Cap such beats at <=2/episode” is arbitrary and may fail on very short or long episodes. Concrete fix: express the cap as a ratio or beat-count-dependent rule, e.g. `max(1, floor(total_beats * 0.2))`, and define whether repeated noun variants count separately.

3. [L1 / Open code-verify #1] The plan assumes individual beat-intent regeneration is possible without destabilizing the outline. If neighboring beats depend on the regenerated intent, replacing one intent may break causal continuity. Concrete fix: if any regeneration/repair remains, define local invariants: preserve speaker, phase, beat role, causal predecessor/successor hooks, and premise object.

4. [L2] The “announcer outro reference that on-stage choice” template risks making episodes formulaic: “Because X chose Y...” every time. Concrete fix: define it as a semantic requirement rather than a fixed template, or provide a small deterministic template family keyed by seed.

5. [L3] “Kills narrated-action + meta-leak deterministically” overclaims. Splitting `{internal_action, spoken_dialogue}` only works if the model obeys the delimiter/schema. Weak local models may put action or director notes inside `spoken_dialogue`. Concrete fix: pair L3 with deterministic validation/scrubbing of `spoken_dialogue`; do not claim the split alone eliminates the leak.

6. [L4] Transcript sanitizer scope is too narrow for the stated leaks. It mentions “voice should,” “tone,” lowercase “announcer:” and unbalanced quotes, but not broader stage-direction/action leakage. Concrete fix: define a leakage taxonomy and a conservative sanitizer policy with examples of what is stripped versus preserved as legitimate dialogue.

7. [Build order] The proposed order starts with L5 even though L1/L2 are identified as the top structural fixes. This weakens the plan’s story: if structure is the root cause, default writer should not lead. Concrete fix: reorder to L1 → L2 → evaluation; then L5 if evidence supports default change; then L3/L4 hygiene.

8. [Hard constraints #3] “new values ride free-form `meta` / `compose_flags`” assumes all downstream consumers tolerate added metadata and unknown flags. Concrete fix: add a compatibility requirement: unknown `meta`/`compose_flags` entries must be ignored by freeze, TTS, tests, and workflow serialization. Verify this before adding fields. [ASSUMPTION]

9. [Open code-verify #5] `meta.story_quality` telemetry under-count is mentioned only in the verify list but not integrated into the plan. If measurement is unreliable, the re-soak cannot prove improvement. Concrete fix: add telemetry correctness as a prerequisite for the L1+L2 evaluation, or define a manual/eval-independent measure of sameness reduction.

OPTIONAL / NICE-TO-HAVE:

- [L1] Add seed-keyed rotation of premise-palette objects so episodes do not all pick the first/most salient brief entity.
- [L2] Add a small set of deterministic beat-role archetypes beyond the two required slots, but only after personal-stake and climax enforcement work.
- [Evaluation] Define success criteria for the “small matrix” re-soak before implementation: e.g. maximum generic-crisis noun density, required-role presence, and human-rated structural sameness.

CUT THESE (scope / over-engineering):

1. [L6] Cut best-of-N line selection from v0. It is costly, contradicts the anti-gate message, and does not address the stated root cause of structural sameness. Safe to defer until L1/L2 prove a new structural baseline.

2. [L5] Cut “frontier vs local” model ranking discussion from the build plan. The primary fix is deterministic skeleton/palette work; model preference is configuration/evaluation policy and should not be coupled to architecture changes.

3. [L3] Cut the full `{internal_action, spoken_dialogue}` contract from the first build slice unless a source review proves the composer/freeze path can absorb it cleanly. A narrower deterministic sanitizer can address prompt leaks first with less blast radius.

4. [L4] Cut mojibake discussion from v0. The plan itself says the mojibake instance was not confirmed in the ledger; it does not serve the story-quality goal until verified.

5. [Build order] Cut “commit + push” process detail from the architectural improvement plan. It is workflow hygiene, not part of the story-quality design, and distracts from unresolved contract decisions.