<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: yes-with-fixes — core sequence is buildable, but C0/C1/C2 still have build-blocking ambiguities around durable flag ownership, pitch-room schema/function contract, critic signal wiring, and temporary PREMISE routing semantics.

MUST-FIX BEFORE BUILD:
1. [Candidate 0 / Candidate 1] Durable flag ambiguity: C0 “emits a DURABLE operator flag `OTR_ENABLE_FRONTIER_GREENLIGHT`” but C1 treats it like runtime config gating `OTR_GREENLIGHT_MODEL`. It is not specified where this flag lives, who writes it, how C1 reads it, or precedence versus environment/config. Two builders could implement env var, workflow config, metadata, or file state, with incompatible behavior.
   Concrete fix: define the storage/precedence contract in one sentence, e.g. “C0 writes `OTR_ENABLE_FRONTIER_GREENLIGHT` into the canonical workflow config/env consumed by `OTR_LedgerScriptWriter.run()`; explicit operator env/config override wins; C1 reads only this resolved boolean via `<existing config accessor>`.” If exact accessor is unknown, add verify: identify config accessor and precedence before implementation.

2. [Candidate 1] Pitch-room call contract is under-specified: “called inside `OTR_LedgerScriptWriter.run()` AFTER news briefs, BEFORE `generate_outline` … injected generate_fn” does not define the function signature, inputs, outputs, or failure behavior of `_otr_pitch_room.py`. This is build-blocking because C1 also needs raw script brief, divergence seeds, palette adapter, local/frontier greenlight, metadata stamping, and frozen `OutlineRequest` replacement.
   Concrete fix: add a minimal explicit contract:
   `run_pitch_room(outline_req: OutlineRequest, *, generate_fn, local_model, frontier_model_config, seed_context, meta) -> tuple[OutlineRequest, PitchMeta]`
   or equivalent actual types after verify. Include: on pitch generation failure/unparseable/less than 3 valid candidates, either retry N then fall back to original `outline_req` or fail the run. Current text only says regenerate for invalid candidates, with no cap or terminal behavior.

3. [Candidate 1] Candidate regeneration has no cap: “>= 3 valid candidates (else regenerate)” can loop forever if schema parsing or validation keeps failing.
   Concrete fix: set a hard cap, e.g. “maximum 2 regeneration attempts; on exhaustion fall back to original script_brief and stamp `meta.story_quality.pitch.status='failed_fallback'`,” or “fail closed with explicit exception.” Pick one.

4. [Candidate 2 Tier 1] Critic signal source is ambiguous. The plan says “feed the 5B critic's `failing_axes`/`regeneration_hint` into the refine loop's `prior_critique`,” but does not identify where the 5B critic is invoked in the current workflow, whether it runs before every refine pass, only after `grade_story`, or how its output is merged with `grade_story.biggest_weakness`.
   Concrete fix: specify the exact data flow:
   - invoke critic after composed episode and before `_refine_loop` decides next pass;
   - construct `prior_critique = regeneration_hint + serialized failing_axes`;
   - if critic unavailable/unparseable, fall back to `grade_story.biggest_weakness`;
   - keep-best scoring remains based on existing grade unless explicitly changed.
   If critic invocation location/API is not known, add verify: locate critic call/output object and add adapter.

5. [Candidate 2 Tier 2 / temporary routing] Contradiction in PREMISE handling before Tier 2 ships. The plan says add `EscalationScope.PREMISE` and route `premise_hits -> PREMISE`, but also says “Until Tier 2 ships, route PREMISE -> EPISODE (temporary) so no consumer crashes.” It is unclear whether `decide_escalation_scope([premise_clarity])` should return PREMISE now, or whether downstream should remap PREMISE to EPISODE. The listed regression case expects `[premise_clarity]->PREMISE`, which conflicts with “route PREMISE -> EPISODE.”
   Concrete fix: split semantic decision from execution fallback:
   - `decide_escalation_scope` returns `PREMISE` for premise axes now.
   - `_otr_freeze_cascade` / execution layer maps `PREMISE` to EPISODE until Tier 2 is enabled, with explicit meta note `premise_degraded_to_episode=true`.
   - regression tests assert both: decision returns PREMISE; execution fallback does not crash and performs EPISODE reroll when Tier 2 disabled.

6. [Candidate 2 Tier 2] Counters in meta are specified but not keyed. “Caps `OTR_STORY_REPITCH_MAX`(1) / `OTR_STORY_REPLAN_MAX`(2) as counters STORED IN meta” does not say exact meta paths or increment points. Since refine reruns replace/recompose objects, counters can reset or double-count.
   Concrete fix: define exact paths and lifecycle, e.g. `meta.story_quality.repitch_count`, `meta.story_quality.replan_count`; increment immediately before attempting pitch-room repitch / re-outline pass; preserve via dataclass/meta merge across reruns; exhaustion means no further attempt and keep-best.

7. [Candidate 4] `score_outline` / `select_best_outline` API change is still risky because the penalty type/range/composition is not defined. “optional `penalty=None` kwarg” does not state whether penalty is numeric, callable, object, additive/subtractive, normalized, or how it affects tie-breaking. Different implementations will produce incompatible selection behavior.
   Concrete fix: define: `penalty: float | None`, interpreted as points subtracted from the existing outline score after all current scoring, default `None` byte-identical; clamp to `[0, X]` or leave unclamped explicitly; tie-break remains existing order after adjusted score. Also define where staging penalty value is stored for audit.

8. [Candidate 4] Deterministic staging rule relies on undefined outline shape. “climax = final voiced beat is character/announcer with a decisive `intent`” assumes fields such as voiced beat, speaker type, beat role, and intent exist and are consistently named. This is still marked like implementation guidance, not a verifiable contract.
   Concrete fix: add verify/build step before coding: inspect outline beat schema and map exact fields for `BEAT_ROLE`, speaker/voiced status, and `intent`. If any field is absent, implement the staging penalty only against existing fields and document fallback behavior.

SHOULD-FIX:
1. [Build order] The build order says “C2 Tier 2 (re-pitch)” after C4, but Candidate 2 labels Tier 2 “later sprint.” This is a planning regression: the locked build order includes a deferred item.
   Concrete fix: change build order to: `C0 -> operator ceiling decision -> C1 -> C2 Tier 1 -> C4 -> C3 parallel`, and list `C2 Tier 2` under deferred/later sprint only. If any Tier 2 scaffolding is in this sprint, call it “enum/fallback scaffolding,” not re-pitch.

2. [Candidate 0] “Grade the best few TWICE” is ambiguous. “Few” is not reproducible and affects cost/pass rate.
   Concrete fix: specify N, e.g. “grade top 3 by first-pass score twice” or “grade all candidates whose first-pass score >= 75 twice.”

3. [Candidate 0] “compose SHORT full episodes (reduced word budget)” lacks a concrete budget. This affects comparability of `grade_story` to production episodes.
   Concrete fix: specify exact token/word budget or config key, and log it with model id/seeds/temp.

4. [Candidate 1] `PitchCandidate.id:int` validity is underspecified. The plan validates selected_id and ranking, but not uniqueness or id range.
   Concrete fix: require unique ids, preferably exactly `[0,1,2]` or `[1,2,3]`, and ranking must be a permutation of those ids.

5. [Candidate 1] Tie-break only mentions “numeric console_standoff_risk then id” but not direction.
   Concrete fix: state “lower `console_standoff_risk` wins; then lower id wins” unless the intended direction differs.

6. [Candidate 1] “Length-bound the brief (verify macro-prompt tolerance)” is not actionable enough for implementation.
   Concrete fix: set an initial hard cap, e.g. “script_brief <= existing max brief length or <= N chars after verify; if exceeded, summarize/truncate structured fields in fixed order.” If no existing max exists, verify it and choose one before build.

7. [Candidate 2] New critic axis `console_standoff` is introduced in C2 but C1 also has `console_standoff_risk`. Relationship is not defined. One is a critic failing axis, the other is a pitch numeric feature; names are close enough to cause accidental reuse.
   Concrete fix: explicitly state that `console_standoff` is a critic failing axis enum, while `console_standoff_risk` is a pitch candidate numeric ranking field; do not serialize one as the other.

8. [Candidate 3] “zero slot drift” is not defined.
   Concrete fix: define measurable assertion: exact same slot count/order/ids before and after enabling `use_exchange`, or specify the workflow artifact to compare.

9. [Candidate 3] `use_exchange=True` field-name/precedence is still a verify item but is not included in Residual verify-at-build.
   Concrete fix: add it to VERIFY-AT-BUILD.

10. [Deferred] “refine-loop hardening … revisit after the levers land” is okay as deferred, but the plan now modifies the refine loop. Add one smoke regression around keep-best monotonicity across the new critique-source swap to avoid worsening the known issue.
   Concrete fix: test that enabling critic critique does not replace the kept best with a lower graded final unless existing behavior already does; if existing behavior is non-monotonic, at least log best-vs-final.

OPTIONAL / NICE-TO-HAVE:
- [Candidate 1] Include pitch-room prompt/version fingerprint in `meta.story_quality.pitch` for reproducibility, not only full pitch/fingerprint.
- [Candidate 4] Log separate staging penalty components: missing final decisive intent, off-stage climax, beat-turn heuristic. Useful for tuning without adding LLM critic.
- [Candidate 0] Persist failed local-ceiling probe artifacts for operator review when no local episode clears 75.

CUT THESE:
1. [Candidate 1] “local genre + protagonist-archetype axis” can be cut for v1 if palette integration plus three divergent seeds already yields enough pitch diversity. Safe because core goal is greenlit divergent pitches; extra axes are tuning, not infrastructure.
2. [Candidate 2 Tier 2] Full `PitchRequest(showrunner_note=..., excluded_fingerprints=...)` implementation should be cut from this build if Tier 2 is truly later sprint. Keep only enum/fallback scaffolding and meta fields needed not to crash.
3. [Candidate 4] “beat-turn heuristic” can be cut if the BEAT_ROLE final-beat deterministic rule is implementable. Safe because the stated v1 priority is staging enforcement inside selection; the irreversible-choice final beat is the highest-signal rule.
4. [Candidate 0] Outline-only score pre-filter can be cut unless cost is already proven high. Safe because the plan explicitly says outline-only is not the ceiling signal; it adds another scoring path without determining pass/fail.

VERIFY-AT-BUILD checklist:
1. [Candidate 1 / Residual verify] Verify exact palette symbol name/publicness in `_otr_story_quality_l12`; specifically locate the real `..._PALETTE` symbol and implement `load_conflict_palette()` against it.
2. [Candidate 4 / Residual verify] Verify exact `BEAT_ROLE` symbol name/publicness in `_otr_story_quality_l12`; import it rather than copying literals.
3. [Candidate 2 / Residual verify] Verify `_refine_loop` re-invokes outline + compose each pass and that swapping/augmenting critique source from `grade_story.biggest_weakness` to critic axes/hint does not break keep-best behavior.
4. [Candidate 2 / Residual verify] Verify `decide_escalation_scope` caller set and every `_otr_freeze_cascade` scope switch has explicit behavior for `EscalationScope.PREMISE`.
5. [Candidate 1 / Residual verify] Verify macro-prompt length tolerance for richer `script_brief`; set a hard cap before implementation.
6. [Candidate 3] Verify exact JSON field name for `use_exchange` and config precedence; confirm writer runtime does not override it.
7. [Candidate 3] Verify N=3 harness effective config reports `use_exchange=True`, VRAM <= 14.5 GB, and defined “slot drift” assertion passes.
8. [Candidate 1] Verify `OutlineRequest` is actually frozen and `dataclasses.replace(outline_req, script_brief=brief)` is valid for the real type.
9. [Candidate 1] Verify `structured_call` Pydantic schema path supports the proposed nested/list validation and unparseable fallback.
10. [Candidate 2] Verify critic output field names are exactly `failing_axes` and `regeneration_hint`; if not, add adapter rather than changing refine-loop internals directly.
11. [Candidate 4] Verify all callers of `score_outline` and `select_best_outline` before adding `penalty=None`; regression must prove byte-identical output when `penalty is None`.
12. [Candidate 4] Verify outline beat schema contains enough fields to implement “final voiced beat,” speaker/announcer/character classification, `intent`, and irreversible-choice role. If not, reduce deterministic staging rule to fields that actually exist.