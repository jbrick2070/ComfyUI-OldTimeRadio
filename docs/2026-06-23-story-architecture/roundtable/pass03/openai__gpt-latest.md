<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. New PREM​ISE scope / pitch state / penalty threading change multiple existing contracts and are not sequenced tightly enough; current plan will break escalation routing and outline selection unless interfaces are defined first.

MUST-FIX BEFORE BUILD:
1. [Candidate 2 / _otr_reroll_escalation.py] `EscalationScope.PREMISE` is proposed, but the grounded enum only has `NONE`, `EPISODE`, `BEAT`, `LINE`. Adding a new enum value is not sufficient: every consumer that switches on `EscalationDecision.scope` must be updated before any critic can emit PREM​ISE. Concrete fix: implement PREM​ISE end-to-end in this order: add enum value; update `decide_escalation_scope`; update `_otr_freeze_cascade` routing; update meta serialization/deserialization expectations for `meta.reroll_escalation.scope`; add an exhaustion path that converts exhausted PREM​ISE to keep-best rather than falling into legacy LINE. Verify: all scope comparisons in `_otr_freeze_cascade.py`.

2. [Candidate 2 / Candidate 1] PREM​ISE re-entry depends on pitch metadata that Candidate 1 does not yet guarantee exists. Candidate 2 says fingerprint is stored in meta as `(conflict_type, setting_class, antagonist_class, hash(final_20_seconds))`, but Candidate 1 `PitchCandidate` schema has `antagonist_or_pressure`, not `antagonist_class`. That is a hard contract mismatch. Concrete fix: either add `antagonist_class` to `PitchCandidate` and require it in structured parsing, or change the fingerprint tuple to use the existing `antagonist_or_pressure` field after normalization. Do not implement Tier 2 until the selected pitch and fingerprint are stamped in meta by the initial pitch-room path.

3. [Candidate 2 / Candidate 1] PREM​ISE re-pitch exclusion cannot work as specified because `GreenlightDecision.failed_premise_fingerprints` is an output field, while Candidate 2 needs failed fingerprints as an input constraint to the pitch/greenlight call. Concrete fix: define a pitch-room input object or function signature containing `showrunner_note` and `excluded_fingerprints`; have the greenlight prompt receive those exclusions; reserve `failed_premise_fingerprints` in `GreenlightDecision` for newly identified failures only, or remove it from the output schema.

4. [Candidate 2 / _otr_reroll_escalation.py] Mapping `premise_clarity -> PREMISE` conflicts with current grounded behavior: `STRUCTURAL_AXES` includes `"premise_clarity"`, and `decide_escalation_scope` currently routes any structural hit to EPISODE. Concrete fix: change routing order so `premise_clarity` is intercepted before `structural_hits`, or split `STRUCTURAL_AXES` into premise vs episode structural axes. Add regression cases for `failing_axes=["premise_clarity"]` returning PREM​ISE and `["resolution"]` returning EPISODE.

5. [Candidate 2 / _otr_reroll_escalation.py] “console-standoff fingerprint -> PREM​ISE” has no grounded input path. `decide_escalation_scope` only reads `verdict`, `failing_axes`, `regeneration_hint`, and optional `story_critic_targets`; no fingerprint field is read. Concrete fix: either encode this as a named failing axis emitted by the critic, e.g. `console_standoff`, and add it to the PREM​ISE axis set, or extend `decide_escalation_scope` with an explicit optional `premise_fingerprint_flags` parameter and update `_otr_freeze_cascade` to pass it.

6. [Candidate 2 / Candidate 4] Both propose adding a penalty into outline selection, but the grounded fact says `score_outline(outline, meta, roster) -> StoryScore` is pure and takes no penalty, while `select_best_outline` currently steers via `dataclasses.replace(outline_req, diversity_hint=hint)`. The plan does not define where the penalty is represented without breaking scorer purity. Concrete fix: introduce an immutable `OutlineScoringContext` or explicit optional `penalty: Optional[...] = None` parameter to `score_outline` and `select_best_outline`, defaulting to `None`; ensure empty/None path is byte-identical. Do not pass penalties through mutable `meta` if “pure/frozen” is required.

7. [Candidate 4 / Candidate 2] Sequencing collision: Candidate 4 says staging enforcement is POST-outline/PRE-composition and feeds a numeric penalty into `score_outline`, but scoring/selecting outlines normally must occur before the final outline is chosen. If the penalty is computed after outline selection, it cannot influence `select_best_outline`. Concrete fix: run staging checks inside the outline candidate evaluation loop, before `select_best_outline` returns the winner; or rename it as a post-selection validator that can trigger re-outline. Pick one wiring path.

8. [Candidate 1 / Verify-at-build] The plan depends on importing the real `_otr_story_quality_l12` palette but leaves the exact symbol name/publicness as “verify-at-build.” That is a build blocker for Candidate 1 because pitch seeding requires the palette. Concrete fix: before implementing pitch-room, resolve the actual palette symbol and wrap it behind a local adapter, e.g. `load_conflict_palette() -> Mapping[str, ...]`, with a fallback or explicit startup error if unavailable.

9. [Candidate 1] Frontier greenlight says “reuse existing OpenRouter slot + cost guard” and `OTR_GREENLIGHT_MODEL`, but does not define config propagation through the workflow/writer layer. If the greenlight lives in workflow JSON, env vars and cost guard must be available there; if inside `OTR_LedgerScriptWriter`, they must be threaded through writer config. Concrete fix: decide location first, then add one config contract: `OTR_GREENLIGHT_MODEL`, enable/disable flag, timeout, max retries, and cost budget source. Fail closed to local on missing/invalid model before making any OpenRouter call.

10. [Candidate 1 / Candidate 0] Candidate 1 says frontier greenlight is gated by Candidate 0, but Candidate 0 is a temporary local probe, not a runtime feature flag or persisted capability check. There is no durable signal for Candidate 1 to read. Concrete fix: Candidate 0 must emit an operator decision/config value, e.g. `OTR_ENABLE_FRONTIER_GREENLIGHT=true`, or Candidate 1 must remain local unless that explicit flag is set.

11. [Candidate 0] The local-ceiling probe says “compose ONE scene” then `grade_story`, but `grade_story` may expect a full story/episode object rather than one scene. [ASSUMPTION] Concrete fix: verify `grade_story` input contract. If it requires full episode shape, add a scene-grade adapter or use the same minimal ledger/script object shape produced by normal composition, not an ad hoc scene string.

12. [Candidate 3 / Verify-at-build] “flip `use_exchange` config-only after GPU N=3” is under-specified for wiring. The plan references “canonical workflow JSON config-only change” but does not identify the exact config key, layer, or whether writer runtime overrides it. Concrete fix: verify the workflow JSON field name and precedence chain; add an N=3 harness that runs the canonical workflow with only that config diff and asserts the effective writer config has `use_exchange=True`.

SHOULD-FIX:
1. [Candidate 1] `GreenlightDecision.ranking:list[int]` and `selected_id:int` need validation against the generated candidates. Otherwise a parsed but semantically invalid frontier response can select a nonexistent pitch. Concrete fix: after structured parse, require `selected_id in candidate_ids`, `ranking` is a permutation/subset containing all valid IDs, and at least 3 valid candidates exist; otherwise fail closed to local or regenerate.

2. [Candidate 1] Deterministic tie-break is specified but not wired to parsed scores. The schema has `taste_rationale` but no per-axis numeric scores or explicit `console_standoff_risk` value. Concrete fix: add bounded enum/int fields for rubric scores and `console_standoff_risk`, or perform ranking entirely in local code using parsed rubric fields. Do not rely on prose rationale for tie-breaks.

3. [Candidate 1] `brief_for_outline` duplicates the separately mapped winning `PitchCandidate` fields. This can diverge: greenlight may select candidate 2 but write a brief inconsistent with candidate 2. Concrete fix: either generate `script_brief` locally from the selected `PitchCandidate`, or validate that `brief_for_outline` references the selected candidate and bounded fields only.

4. [Candidate 1 / OutlineRequest] `OutlineRequest.script_brief` takes precedence over raw `news_seed`; the plan adds a generated brief but does not say whether raw `news_seed` remains available for traceability or prompt context. Concrete fix: stamp both original seed and selected pitch metadata in `meta`, while passing only the concise pitch brief into `script_brief`.

5. [Candidate 2] Separate caps `OTR_STORY_REPITCH_MAX` and `OTR_STORY_REPLAN_MAX` require state counters that survive full reruns. If counters live only in local function state, a full rerun can reset them and loop. Concrete fix: store counters in episode/job meta passed through reruns, and increment before dispatching the rerun.

6. [Candidate 2] “on exhaustion, keep-best” needs a defined comparison source. If the failed current episode is the only candidate, “best” is undefined. Concrete fix: persist grade/critic score and artifact reference for each attempt; choose highest score or last composed artifact by deterministic fallback.

7. [Candidate 4] “final voiced beat is character/announcer with decisive intent” depends on role labels and `beat.intent`. The plan does not define handling for missing/empty `intent`. Concrete fix: missing `beat.intent` should receive the staging penalty or trigger re-outline, not pass silently.

8. [Candidate 4] The existing `BEAT_ROLE` role string is named as `"irreversible_choice-on-stage-as-the-last-beat"`, but the plan should not hard-code the literal in multiple modules. Concrete fix: expose/import a single constant or adapter from `_otr_story_quality_l12`; if not public, copy it once into a local constant with a test against the source sequence. Verify exact symbol access.

9. [Candidate 1] External OpenRouter call lacks retry/backoff/timeout specification. “Fail closed to local if disabled/unparseable” does not cover transient 429/5xx or timeout. Concrete fix: use the existing structured_call ladder retry policy if available; otherwise set bounded retries, exponential backoff, and parse-failure fallback to local.

10. [Candidate 0] “grade the best few TWICE” needs deterministic logging of both raw grader results and aggregation rule. Concrete fix: define pass if max of two, mean of two, or both >=75; otherwise the probe result is not reproducible.

OPTIONAL / NICE-TO-HAVE:
- [Candidate 1] Add a compact prompt-token budget test for `script_brief` length once the actual outline macro prompt tolerance is verified.
- [Candidate 2] Include attempted scopes and exhaustion reason in `meta.reroll_escalation` audit trail, not only the final decision.
- [Candidate 0] Store model id, temperature, seeds, and grader duplicate outputs in a machine-readable JSON artifact.

CUT THESE (over-engineering):
1. [Candidate 2] Cut Tier 2 PREM​ISE from the first build. The plan itself says ship Tier 1 first; PREM​ISE requires new enum routing, pitch metadata, fingerprint storage, exclusion input, and separate caps. Safe to cut because current grounded EPISODE full rerun already handles all structural axes behind `enable_critic_escalation`.

2. [Candidate 4] Cut the LLM outline-critic for staging until deterministic penalties are proven insufficient. Safe to cut because existing `BEAT_ROLE` plus `beat.intent` heuristic can be wired into outline selection without adding another external call, parse schema, or retry path.

3. [Candidate 1] Cut `failed_premise_fingerprints` from `GreenlightDecision` output in the initial build. Safe to cut because exclusion is only needed for Tier 2 PREM​ISE re-pitch, which should not ship until pitch state exists.