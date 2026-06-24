<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan is directionally bounded, but Candidate 1/2 cannot be implemented from the described interfaces: it invents pitch/greenlight data flow, assumes outline request compatibility, and proposes new escalation scopes not supported by the shown pure router.

MUST-FIX BEFORE BUILD:
1. [Candidate 1 / Handoff] `script_brief` compatibility is explicitly “verify-at-build,” but the plan depends on it. Grounding shows `_otr_story_select.select_best_outline(generate_outline_fn, outline_req, ...)` mutates an existing dataclass via `dataclasses.replace(outline_req, diversity_hint=hint)`, and `score_outline` reads `outline.premise` plus beat fields. No `script_brief` shape or `_otr_outline` request schema is shown. Concrete fix: before coding pitch room, inspect `_otr_outline.OutlineRequest` and document exact field(s) to populate from `GreenlightDecision.script_brief`; if no field exists, add a typed field with default/backward-compatible handling and tests that pass both old brief and greenlight brief through outline generation.

2. [Candidate 1 / PitchCandidate[] schema] The schema is underspecified for implementation. It lists fields but not IDs, types, validation, serialization, prompt/parse mechanism, or how candidates are referenced by `GreenlightDecision.selected_id`. `selected_id` has no corresponding `id` in `PitchCandidate`. Concrete fix: define Pydantic/dataclass models, e.g. `PitchCandidate(id: int, logline: str, ... conflict_type: str, ...)` and `GreenlightDecision(selected_id: int, ...)`; include max lengths, required/optional fields, and JSON parse/repair behavior using the repo’s `structured_call` pattern if that is the intended API.

3. [Candidate 1 / Force divergence] “existing palette in `_otr_story_quality_l12.py`” is not substantiated by grounding. The shown imports from `_otr_story_quality_l12` are `count_ungrounded_crisis`, `premise_noun_palette`, and `premise_texts`; no conflict-type palette API is shown. Concrete fix: verify the actual exported symbol for the conflict palette. If none exists, define a local constant in the new pitch module instead of importing a private/nonexistent symbol. Do not depend on private names unless you add a public accessor.

4. [Candidate 1 / frontier-backed greenlight] The plan says greenlight defaults to “FRONTIER lane” but gives no callable/provider interface. Grounding only shows `grade_story(..., generate_fn)` and `structured_call(... slot_fn=generate_fn ...)`; `resolve_best_of_n` identifies remote model strings via `creative_writing_model.startswith(("openrouter:", "comfy:"))`. Concrete fix: specify the exact function signature for the greenlight LLM call, how to resolve/select the frontier `generate_fn`, env/widget knobs, max tokens, cost guard, and fallback behavior when the frontier call fails or returns unparseable JSON.

5. [Candidate 0 / GATE] “Run ~10 pitch-room sets on LOCAL model -> compose -> `grade_story`” cannot be coded as written because the pitch room does not exist yet and `grade_story` requires `generate_fn` plus composed text/premise, not a workflow entrypoint. Concrete fix: either move Candidate 0 after a minimal pitch-room implementation, or define a temporary local-pitch function and exact compose function to call. Also define whether `grade_story.score_0_100 >= 75` uses `error_type is None`; otherwise grader failures return score 0 and may falsely stop the campaign.

6. [Candidate 0 / grade threshold] The plan says `>=75 (B+)`, but grounding’s `grade_story` prompt says `B+~=80, B~=75`. This mismatch will produce wrong operator decisions. Concrete fix: either set the gate to `>=80` for B+, or rename the gate to B/75 consistently.

7. [Candidate 2 / Tier routing] The plan proposes “route to divergent RE-OUTLINE” and “route back to PITCH ROOM,” but the shown escalation API only returns `EscalationScope.NONE|EPISODE|BEAT|LINE`; `EPISODE` stamps `needs_full_rerun` via `_otr_freeze_cascade.py` and `_otr_reroll_escalation.py` is pure/no I/O. There is no scope for `REOUTLINE` or `PITCH`. Concrete fix: add new enum values such as `REOUTLINE` and `PITCH` or add a second decision layer outside `decide_escalation_scope`; then update `_otr_freeze_cascade.py` wiring and tests to consume those scopes. Do not overload `EPISODE` unless the cascade can distinguish same-premise reoutline vs new-premise pitch.

8. [Candidate 2 / axis mapping] The proposed Tier 1/2 predicates do not match grounded axes. Grounding structural axes are `premise_clarity`, `continuity`, `resolution`, `emotional_arc`; there is no shown `flat middle`, `off-stage climax`, `weak resolution`, or `console-standoff collapse` axis. The plan mentions `arc_verdict` / `flat_lines`, but grounding only shows `verdict`, `failing_axes`, and `regeneration_hint`. Concrete fix: either map existing `failing_axes` exactly (`resolution` -> Tier 1, `emotional_arc` -> Tier 1, `premise_clarity` -> Tier 2, etc.) or update the critic schema to emit the new machine-readable fields and modify `_coerce_*` helpers accordingly.

9. [Candidate 2 / re-run outline best-of-N with failing axis penalty] Grounding `score_outline` signature is `score_outline(outline, meta, roster) -> StoryScore` with fixed comparator in `select_best_outline`; it accepts no failing-axis penalty or critique hint. Concrete fix: add an optional scoring context/penalty parameter and thread it through `select_best_outline`, or implement a separate `select_reoutline_candidate(..., penalty_axes=...)`. Include tests proving the original selector remains byte-identical when no penalty is passed.

10. [Candidate 2 / failed_premise_fingerprints] The plan requires excluding prior fingerprints but defines no data model or fingerprint algorithm. Concrete fix: define fingerprint fields, normalization, uniqueness, storage location in `meta.story_quality` or pitch state, and matching logic. Example: fingerprint normalized tuple `(conflict_type, setting_class, antagonist_or_pressure_class, final_20_seconds_summary_hash)`. Without this, “exclude” is not implementable.

11. [Candidate 3 / use_exchange] “config-only change to canonical workflow default/link” is not grounded. No `use_exchange` config key, workflow file, or test harness API is shown. Concrete fix: verify the actual widget/config name and canonical workflow serialization. Add a tiny PR that only changes that default after running the named harness. [ASSUMPTION] If the repo uses generated Comfy node JSON, changing Python defaults may not update existing workflows.

12. [Candidate 4 / every beat must TURN] `score_outline` currently sees only beat `intent` and `speaker_role` in grounding; no beat schema includes power/status/knowledge/emotion delta fields. Concrete fix: either implement a text heuristic over `intent` only and call it heuristic, or extend outline beat schema to include explicit `turn_type`/`state_change` fields and update outline generation prompts/parsers/tests.

SHOULD-FIX:
1. [Candidate 1 / REQUIRE rejecting >=1] This can fail when all three candidates are genuinely usable or when parsing returns only two valid candidates. Concrete fix: require at least 3 valid candidates before greenlight; if fewer, regenerate/repair the pitch set. Make “reject >=1 for sameness” conditional on candidate count and validation success, and define fallback if all are too similar.

2. [Candidate 1 / ordinal ranking] Tie handling is undefined. Concrete fix: define deterministic tie-breaker, e.g. lower `console_standoff_risk`, then candidate id.

3. [Candidate 1 / quote evidence per axis] Evidence quotes can exceed prompt/token budgets and may be empty if generated fields are terse. Concrete fix: bound evidence strings and validate nonempty evidence per scored axis.

4. [Candidate 0 / sample size] “~10 pitch-room sets” is not reproducible. Concrete fix: specify exact count, seeds, temperature, model id, and logging fields; otherwise GATE results are not comparable run to run.

5. [Candidate 0 / best grade] `grade_story` is itself LLM-based and may be noisy. The Deferred section mentions diagnosing grader noise, but the GATE uses a single grade as a stop/go decision. Concrete fix: grade the best few composed stories twice or record `error_type`, model id, seed, and raw weakness; do not stop solely on one unreplicated low score.

6. [Candidate 1 / no change to outline schema] This conflicts with the desire to feed richer brief/theme/ending fields unless `_otr_outline` already accepts them. Concrete fix: either flatten the greenlit brief into the existing premise/brief text field, or formally extend schema. Do not pass an object where existing code expects a string. [ASSUMPTION] Existing outline prompt likely expects string-like premise/brief.

7. [Candidate 2 / cap divergent re-plans separately] No cap names, defaults, env/widget source, or telemetry are defined. Concrete fix: add e.g. `OTR_STORY_REPLAN_MAX=2`, `OTR_STORY_REPITCH_MAX=1`, stamp counters in meta, and make exhaustion behavior deterministic.

8. [Candidate 2 / keep-best on exhaustion] Existing keep-best policy is referenced but not shown for these new tiers. Concrete fix: identify where prior candidates are stored and how to compare them. If no persisted composed candidate pool exists across full reruns, add one or change the behavior to “ship best current candidate.”

9. [Candidate 4 / outline critic before generation] Running an LLM outline critic before composition may add latency and another failure mode. Concrete fix: start with pure deterministic penalties inside `score_outline`; only add an LLM critic if deterministic checks prove insufficient.

10. [Candidate 4 / climax ON-MIC] This is not computable from current `score_outline` unless climax beat identity and staging are represented. Concrete fix: add a `beat.role == "climax"` or infer from final voiced beat position, then check `speaker_role in ("character","announcer")` and intent contains decisive action; otherwise defer.

OPTIONAL / NICE-TO-HAVE:
- Add telemetry blocks for `pitch_room`, `greenlight`, `replan_tier`, and `gate_probe` under `meta.story_quality` with JSON primitives only, matching `_merge_best_of_n_telemetry`.
- Add golden fixtures for “console standoff,” “off-stage climax,” and “good on-mic ending” to prevent subjective regressions.
- Log frontier greenlight cost separately from drafting cost.

CUT THESE (over-engineering):
1. [Candidate 4] Separate outline-level staging critic as an LLM node. Safe to cut for MVP because `score_outline` already exists as a deterministic pre-generation selector; add small deterministic penalties first.
2. [Candidate 2] Full Tier 2 automatic route back to pitch room in the first build. Safe to cut initially because it requires new state/cascade semantics; implement Tier 1 reoutline first, then add repitch once pitch-room state and fingerprints exist.
3. [Candidate 1] “Quote evidence per axis” in greenlight output. Safe to cut from first implementation because numeric/ordinal rubric plus short rationale is enough for routing; evidence quotes increase parse fragility and token use.