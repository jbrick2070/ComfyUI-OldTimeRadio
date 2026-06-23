<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: yes-with-fixes — core plan is close, but pass-count semantics, refine/best-of-N collision behavior, and grade field/grade-failure semantics are still ambiguous enough to produce incompatible builds.

MUST-FIX BEFORE BUILD:
1. [The revision loop / Pass 0] Defect: field name regression: text says `if grade.score >= bar`, but [Grader] defines `StoryGrade.score_0_100` and explicitly maps schema `.score` to `.score_0_100`. Concrete fix: change all loop/comparator/telemetry references to use `grade.score_0_100`; reserve `.score` only for the parsed structured schema before conversion.

2. [Flags + gate + collision] Defect: `requested_passes`, `effective_passes`, `max_passes`, and `OTR_STORY_REFINE_PASSES` semantics are under-specified. A builder could interpret passes as “revision passes after baseline” or “total candidates including pass 0”; this changes cap behavior and loop count. Concrete fix: state one invariant, e.g. “`effective_passes` is total candidate count including mandatory pass 0; disabled means `effective_passes=1`; enabled minimum is 2; loop runs `i in range(effective_passes)` with pass 0 baseline and passes 1..effective_passes-1 revisions; clamp to `REFINE_MAX_PASSES=5` total including pass 0.” If instead env means revisions-after-baseline, say that explicitly and rename telemetry to avoid ambiguity.

3. [Flags + gate + collision] Defect: collision behavior conflicts with disabled behavior. Text says “Provider remote OR effective_passes<2 => DISABLED single path” but also says “resolve refine FIRST; if effective_passes>=2, BYPASS resolve_best_of_n/select_best_outline.” This leaves unclear whether existing best-of-N still runs when refine is Off but `OTR_STORY_BEST_OF_N` is set. Concrete fix: add explicit branch:
   - if refine disabled/off/remote-clamped: run existing pre-v1 story path exactly, including existing best-of-N resolution if configured;
   - if refine enabled with `effective_passes>=2`: bypass best-of-N entirely and assert no `best_of_n` telemetry key.
   This preserves existing v0 behavior and only suppresses best-of-N during active refine.

4. [Grader / The revision loop / Telemetry] Defect: grade parse failure semantics are contradictory/ambiguous. [Grader] says parse failure returns `StoryGrade(score=0, biggest_weakness="grader_unparseable")` plus `grade_error_type`; [Telemetry] says per pass has `score|grade_error_type`; [RefineCandidate] has `ok` and `error_type`; [The revision loop] says next pass hint is empty on grade-fail. A builder could mark the pass failed, exclude it from keep-best, or keep it with score 0. Concrete fix: define:
   - `ok=True` means generation + compose succeeded and candidate is shippable.
   - grader parse failure does not make candidate unshippable; candidate remains `ok=True`, `grade.score_0_100=0`, `grade_error_type="grader_unparseable"`, `normalized_hint=""`.
   - `error_type` is only generation/compose failure.
   - comparator includes all `ok=True` candidates even if grade failed.
   - telemetry emits both `score` and `grade_error_type` when applicable, not `score|grade_error_type`, or explicitly chooses one schema.

5. [Telemetry] Defect: `grade_delta=None at pass0 else score - previous_ok_score` is ambiguous after a grade parse failure or compose failure. “previous_ok_score” could mean previous candidate with compose ok, previous candidate with grader ok, or previous loop index. Concrete fix: define `previous_scored_pass` as the most recent prior candidate with `ok=True` and `grade_error_type is None`; if absent, `grade_delta=None`. Or simpler: `grade_delta = current_score - pass0_score` for all later passes, using 0 for grader parse failure. Pick one.

6. [Cancellation (R3)] Defect: cancellation is still under-specified as “VERIFY the exact API, e.g. model_management interrupt hook.” This is build-blocking because placement and exception behavior affect artifact safety. Concrete fix: add a required verify step in chunk 4: identify the existing interrupt function, call it before starting every nonzero pass and immediately before each grade call, and confirm it raises/returns in the same way current long-running nodes expect. State whether cancellation aborts loudly without committing any candidate except already-existing pre-run artifacts.

SHOULD-FIX:
1. [Operator decisions / CUT / Build order] Defect: plateau early-stop is both “OPT-IN default OFF” and listed under CUT as “plateau early-stop.” Concrete fix: remove plateau from [Operator decisions] entirely for v1, or explicitly say “not implemented in v1; future opt-in only.” Current text invites a builder to add dead config surface.

2. [Flags + gate + collision] Defect: `resolve_refine_passes` provider detection is not grounded to a concrete resolved key. Grounding for best-of-N uses `resolved["creative_writing_model"]` and remote prefixes `openrouter:`/`comfy:`. Concrete fix: state `provider` is derived from `resolved.get("creative_writing_model", "")` using exactly the same prefix check unless build verification finds the story writer uses a different key.

3. [Pass isolation + winner commit] Defect: “canon may be by-ref if read-only (verify; CUT deep-copying read-only canon)” is acceptable as verify, but the mutation consequence is severe. Concrete fix: in chunk 3, instrument/test that `_build_and_compose` with a losing candidate leaves canon byte/equality-identical; if not, deep-copy canon despite the cut.

4. [Revision overlay wiring] Defect: overlay activation says macro block emits “when `prior_macro`+`prior_critique` non-empty,” but pass i>=1 may have an empty normalized hint after grade-fail. Then the macro receives no prior macro either, violating “ALWAYS a REVISION seeded by prior story + graded weakness; spine may change but is not blank.” Concrete fix: decide behavior for empty critique:
   - either emit macro revise block when `prior_macro` is non-empty, with weakness omitted/“No usable critique; improve structure while preserving prior spine”;
   - or explicitly allow grade-fail passes to be fresh-ish seeded only by existing normal request.
   The former better matches the locked operator decision.

5. [Grader] Defect: `extract_spoken_text_for_grade(ledger)` says voiced row = “character/announcer; exclude music/sfx,” but exact row field names are unverified. Concrete fix is already in Verify-at-build, but add a test requirement in chunk 2 using a real/minimal ledger row fixture once field names are confirmed.

6. [Build order 4 / JSON widget] Defect: widget default has two conflicting operator notes: [Operator decisions] says delivery can leave default Off or set node default B; [Build order 4] says add widget default Off; [Build order 5] says default-Off. Concrete fix: lock it to default `Off` for this build and remove the “or set node default to B” parenthetical.

OPTIONAL / NICE-TO-HAVE:
- [Telemetry] Include `target_reached: bool` in `refine_loop` to avoid deriving it from `stop_reason`.
- [Logging] Log winning comparator tuple once at commit for easier soak analysis.
- [Tests] Add one targeted test that refine enabled produces no `best_of_n` telemetry even if `OTR_STORY_BEST_OF_N` is set.

CUT THESE:
1. [Operator decisions] Cut “Plateau early-stop is OPT-IN (default OFF).” It is already in CUT and not needed for target-grade/cap behavior.
2. [Operator decisions] Cut “OPERATOR DECISION at delivery: leave default Off, or set the node default to B...” The plan elsewhere locks default-Off; keeping this creates release ambiguity.
3. [Telemetry] Cut `provider` from refine telemetry if it only duplicates `resolved["creative_writing_model"]` classification and is not used in soak. Keep `clamp_reason`/`override_source`; those are operationally useful. Safe if logs/config already identify provider. [ASSUMPTION] provider is available elsewhere in run metadata.
4. [RefineCandidate] Cut `title` from the candidate if the winning commit always runs existing post-compose title regeneration and no intermediate artifact writes use candidate title. If pass-local title is needed for `prior_macro` or compose, keep it. [ASSUMPTION] existing post-compose title regen fully determines shipped title.

VERIFY-AT-BUILD checklist:
1. [Verify-at-build] Confirm `cast_seed` is in scope in `OTR_LedgerScriptWriter.run()` and in the new `_build_and_compose`/`refine_story` call path.
2. [The revision loop] Confirm all three reseed points actually affect their targets: Path C generation, line composition, and grader call.
3. [Code-grounded facts / Grader] Confirm `_seed_rngs` implementation uses `int(sha256.hexdigest(),16)`, `torch.manual_seed(seed % 2**64)`, and `random.seed(seed % 2**32)`, or centralize to the existing helper.
4. [Verify-at-build] Confirm exact exception classes: outline/generation failure, compose failure, config/model-init failures. Ensure only expected generation/compose failures become failed candidates; config/model-init remains LOUD.
5. [Grader] Confirm `structured_call` accepts exactly `base_temperature` and `structural_retry_temperature`, not `temperature`, and that `make_dispatching_repair_factory()` is importable at the grader location.
6. [Grader] Confirm real ledger row shape and fields for `extract_spoken_text_for_grade`; verify character and announcer rows are included and music/sfx rows excluded.
7. [Outline isolation] Confirm `outline.model_copy(deep=True)` exists for the returned outline type and isolates beat intent mutation by `build_sq_data`.
8. [Pass isolation + winner commit] Confirm `build_sq_data` mutates `beat.intent` in place and that `prior_macro` is built from `raw_outline` before mutation.
9. [Pass isolation + winner commit] Confirm `meta` is a dict and `score_outline(..., meta_local, ...)` sees a deep copy of incoming meta preserving news/brief/style context.
10. [Pass isolation + winner commit] Confirm canon is read-only during compose; if not, deep-copy canon for each candidate.
11. [Pass isolation + winner commit] Confirm existing post-compose title regeneration around ~L4356, canon write, FreezeCascade reroll-context meta threading around ~L3440-3606, and ledger save can be delayed and run once on the winner.
12. [Cancellation] Confirm the exact ComfyUI interrupt API and required behavior; add calls between passes and before grade.
13. [Flags + gate + collision] Confirm provider gate uses the same remote prefixes as grounded best-of-N: only `openrouter:` and `comfy:` are remote.
14. [Flags + gate + collision] Confirm absent widget in old JSON is treated as Off and does not break chunk 1-3 tests.
15. [Build order 4] Confirm widget append to `widgets_values` at the end of `otr_scifi_16gb_full.json`, UTF-8 no BOM, and JSON validates.
16. [Invariants] Confirm flag-off run has byte-identical output to baseline, no grade calls, no refine telemetry key, and existing best-of-N behavior remains unchanged when refine is disabled.