# v1 Iterative Story-REVISION Loop -- build plan (pass03, post-R3 wiring hardening)

Operator: a local loop that recursively REVISES the existing story (never from-scratch; spine only when
needed) until it reaches a target grade, then ships. Distinct from shipped v0 best-of-N.

## Operator decisions (locked)
- ALWAYS a REVISION seeded by the prior story + its graded weakness; spine may change but is not blank.
- Target grade = NODE DROPDOWN `refine_target_grade` (Off/C+/B/B+/A). Default-OFF preserves the
  byte-identical invariant (every other OTR quality flag is default-OFF + the audio golden is frozen); B
  is the RECOMMENDED target when enabled. (OPERATOR DECISION at delivery: leave default Off, or set the
  node default to B to make it always-on -- recommend Off for a research feature.) Widget goes IN
  otr_scifi_16gb_full.json (CLAUDE.md S0).
- Recurse until the grade reaches the target OR the hard cap (`REFINE_MAX_PASSES=5`); then ship keep-best.
  Plateau early-stop is OPT-IN (default OFF).

## Code-grounded facts (verified R2+R3 -- do not re-litigate)
- Production `generate_outline` = Path C: `_build_macro_user_prompt` (Stage 1, title/premise/setting = SPINE),
  `_build_phase_user_prompt` (Stage 2, SPEAKER ASSIGNMENT ONLY), `_build_beat_user_prompt` (Stage 3,
  intent/mood = LOCAL). `_build_user_prompt` is back-compat/test-only => shipped v0 `diversity_hint` is
  DEAD (chunk 0 fixes it). Wire revision steering into MACRO + BEAT only -- NEVER the phase prompt
  (injecting a structural hint into a speaker-assignment schema confuses it; Gemini+GPT CONFIRMED).
- `score_outline(outline, meta, roster)` calls `premise_texts(meta)` for the grounding palette =>
  per-pass `meta` MUST be a DEEP-COPY of the INCOMING meta (preserve news/brief/style context), NOT a
  fresh empty dict (Gemini+DeepSeek CONFIRMED). `build_sq_data` mutates `beat.intent` in place.
- `structured_call(prompt, schema, slot_fn, base_temperature, structural_retry_temperature,
  repair_prompt_factory, max_new_tokens, max_attempts, helper_name)` -- there is NO `temperature` kwarg
  (GPT CONFIRMED in generate_outline). `OutlineRequest` is FROZEN (no direct field assign in
  __post_init__). `meta` is a dict. `_seed_rngs` = `int(sha256.hexdigest(),16)`,
  `torch.manual_seed(%2**64)`, `random.seed(%2**32)`. Provider gate: ONLY `openrouter:`/`comfy:` are remote.

## The revision loop (inside OTR_LedgerScriptWriter.run(), default-OFF, local-only)
`RefineCandidate(pass_index, raw_outline, structural_score, ledger, canon, meta_local, title,
composed_text, grade, ok, error_type)`. (meta_delta CUT -- each pass carries a full deep-copied meta;
winner commits its meta + the refine_loop telemetry.)

- **Pass 0 (MANDATORY baseline = today's path):** `generate_outline` (Path C, empty overlays =>
  byte-identical) -> compose -> grade. If pass 0 compose FAILS, fail exactly as today (LOUD) -- the
  "all passes failed" fallback branch is CUT (pass 0 success guarantees a shippable candidate, so
  never-fail is automatic). If `grade.score >= bar`: ship pass 0.
- **Pass i>=1 (REVISE):** feed `prior_macro` (the prior winner's Title/Premise/Setting + numbered raw beat
  intents, capped) + normalized `prior_critique` into the MACRO + BEAT prompts -> `generate_outline`
  (called DIRECTLY, never `select_best_outline`) -> compose -> grade. Keep-best. Repeat to bar/cap.
- **Three seed points per pass (determinism -- R3 catch):** `_seed_rngs(f"{cast_seed}:refine:{i}")` before
  generation, `_seed_rngs(f"{cast_seed}:refine:{i}:compose")` before line composition (the composer may
  sample), `_seed_rngs(f"{cast_seed}:refine:{i}:grade")` before grading.

### Outline isolation (exact -- R3 catch)
Immediately after `generate_outline` returns: `raw_outline = outline.model_copy(deep=True)`;
`structural_score = score_outline(raw_outline, meta_local, roster)`; `working_outline =
outline.model_copy(deep=True)` for `build_sq_data` + compose. `prior_macro` for the NEXT pass is built
from THIS pass's `raw_outline` (pre-mutation intents), captured before build_sq_data runs.

### Grader (NEW, standalone, read-only -- real interface)
`grade_story(composed_text, premise, *, generate_fn) -> StoryGrade`. `StoryGrade =
@dataclass(frozen=True){score_0_100:int, biggest_weakness:str}`; map parsed schema `.score` ->
`.score_0_100`.
- `structured_call(prompt=[system+user], schema=_StoryGradeSchema{score:int 0..100,
  biggest_weakness:str Field(max_length=200)}, slot_fn=generate_fn, base_temperature=0.0,
  structural_retry_temperature=0.0, repair_prompt_factory=make_dispatching_repair_factory(),
  max_new_tokens=128, max_attempts=2, helper_name="OTR_StoryGrade")`. (temp 0 = determinism.)
- Rubric asks for the single biggest STRUCTURAL/dramatic weakness (arc / stakes / premise grounding /
  character want) -- NOT a line edit (line fixes are the downstream freeze-cascade doctor's job; an
  outline reviser can only act on structure).
- `composed_text = extract_spoken_text_for_grade(ledger)`: "SPEAKER: line" per voiced row
  (character/announcer; exclude music/sfx) so the grader can judge character consistency; cap ~4000
  chars (head+tail). Empty => floor grade.
- Parse failure => `StoryGrade(score=0, biggest_weakness="grader_unparseable")` + a separate
  `grade_error_type`; on grade-fail the next pass's `normalized_hint=""` (NEVER feed an infra failure as a
  revision steer).

### Revision overlay wiring (Path C)
- `OutlineRequest` gains `prior_critique: str = ""` and `prior_macro: str = ""` as FINAL fields (after
  diversity_hint, defaulted). NO __post_init__ mutation (frozen) -- normalization happens in
  `critique_to_hint` before `dataclasses.replace`. Add a test that `select_best_outline`'s
  `dataclasses.replace(req, diversity_hint=...)` still works after the new fields.
- `_build_macro_user_prompt`: when `prior_macro`+`prior_critique` non-empty, emit a REVISE block ("Current
  premise/arc: <prior_macro>. Biggest weakness: <prior_critique>. Revise to fix it, keep what works;
  change premise/arc only if the weakness is structural."). Empty => byte-identical.
- `_build_beat_user_prompt`: when `prior_critique` non-empty, one-line "Address this weakness:
  <prior_critique>". Empty => byte-identical. (Phase prompt UNTOUCHED -- speaker-routing revision is OUT
  OF SCOPE for v1.)
- `prior_macro` format: `"Title: {t}\nPremise: {p}\nSetting: {s}\nBeats: " + "; ".join(numbered raw
  voiced intents)`, capped (Stage 1 stays lean).
- `critique_to_hint(biggest_weakness) -> str`: one line; strip control chars/backticks/JSON fences;
  collapse whitespace; reject prompt-injection by NARROW regex `ignore (all )?(previous|system|developer)
  (instructions|messages|prompt)` (not a blanket "ignore previous" reject); <=200 chars at a word
  boundary. Empty/malformed => "".

### Pass isolation + winner commit (sequencing -- R3 catch)
`_build_and_compose(...)` takes PASS-LOCAL ledger/canon/meta/title containers and RETURNS a
`RefineCandidate`; it NEVER assigns `self.ledger`/`self.canon`/`self.title`/`meta.story_quality` or writes
artifacts. Deep-copy the mutated inputs (outline via model_copy(deep=True), meta via deepcopy of incoming);
canon may be by-ref if read-only (verify; CUT deep-copying read-only canon). Sequencing: loop builds
candidates -> pick winner -> COMMIT ONCE post-loop (the existing post-compose title regen ~L4356 + canon
write + FreezeCascade reroll-context meta threading ~L3440-3606 + ledger save run here, on the WINNER
only). Test: a losing pass cannot change live title/ledger/meta.

### Keep-best comparator (exact)
`winner = max(cands, key=lambda c: (c.grade.score_0_100, -c.structural_score.ungrounded_crisis_density,
c.structural_score.distinct_conflict_nouns, c.structural_score.premise_grounding, -c.pass_index))`.

### Flags + gate + collision
`resolve_refine_passes(resolved, *, env, widget_target) -> RefineConfig{requested_passes, effective_passes,
max_passes, bar, target_grade, provider, clamp_reason, override_source}`. Bar from widget map
{C+:68,B:75,B+:80,A:90}; env `OTR_STORY_REFINE_BAR`/`OTR_STORY_REFINE_PASSES` OVERRIDE the widget
(headless) -- record `override_source`. ABSENT widget (old JSON) => treated as Off (disabled) so chunks 1-3
tests run without it. Provider remote OR effective_passes<2 => DISABLED single path, NO grading, NO
refine_loop key. COLLISION: resolve refine FIRST; if `effective_passes>=2`, BYPASS
`resolve_best_of_n`/`select_best_outline` for the story path entirely (LOUD) -- assert no best_of_n key.

### Cancellation (R3)
Call the existing ComfyUI interrupt check between passes + before grade so a long refine is cancelable
(VERIFY the exact API, e.g. model_management interrupt hook).

### Telemetry (winner-commit only)
`meta.story_quality.refine_loop = {requested_passes, effective_passes, max_passes, bar, target_grade,
override_source, stopped_early, stop_reason, winner_pass, winner_grade, provider, clamp_reason,
passes:[{pass_index, ok, score|grade_error_type, normalized_hint, structural_score, grade_delta}]}`.
grade_delta=None at pass0 else score - previous_ok_score. (elapsed_s -> debug log, not telemetry.)

## Invariants
- Audio frozen + byte-identical when OFF: assert the current single path runs, all THREE Path C builders
  are byte-identical with empty overlays, and NO refine_loop key.
- Pass isolation (losers commit nothing); build_sq_data per candidate in isolated state; freeze + audio
  ONCE on the winner. Local-only; deterministic (3 seed points + grader temp 0); HARD CAP + keep-best;
  LOUD on real errors; widget change IN the JSON same commit (S0, append widgets_values at END +
  re-validate); UTF-8 no BOM; SFW.

## Build order (each chunk: full suite + Bug Bible green + JSON validated when touched, commit+push)
0. Wire `diversity_hint` into `_build_macro_user_prompt` + `_build_beat_user_prompt` (fix v0 dead code) +
   test best-of-N candidates now differ. Byte-identical when empty.
1. `OutlineRequest.prior_critique` + `prior_macro` (final fields) + MACRO+BEAT REVISE overlays +
   `critique_to_hint` + byte-identical (3 prompts) tests + select_best_outline replace() compat test.
2. `grade_story` + `StoryGrade` + `extract_spoken_text_for_grade` (real structured_call interface, temp 0,
   max_new_tokens 128, cap, floor fallback) + tests.
3. `_build_and_compose` pass-local helper (deep-copy outline+meta, 3 seed points, isolated RefineCandidate,
   commits nothing) + byte-identical single-pass test (this chunk, not deferred to 4).
4. `refine_story` loop + `resolve_refine_passes` (flag/widget/bar/provider gate + best_of_n bypass) +
   keep-best + winner-commit sequencing + never-fail (pass-0 mandatory) + cancellation hook + merged
   telemetry, wired into run(). Add `refine_target_grade` widget to the node + the JSON (default Off).
   Flag-off call-count + no-key + byte-identical tests.
5. VALIDATION soak (operator GPU): set B; measure grade LIFT (grade_delta) + v0 sameness vs baseline;
   operator written go/no-go. Default-OFF + unadvertised until it passes.

## CUT (R1-R3 consensus)
- Remote opt-in; nested best-of-N; loser-ledger retention; downstream-critic reuse; MAX_SECONDS (cap
  suffices); plateau early-stop (opt-in default OFF); meta_delta object; elapsed_s telemetry; "every pass
  failed" fallback (pass-0-mandatory replaces it); deep-copying read-only canon; phase-prompt critique.

## Verify-at-build
- cast_seed in scope; per-pass reseed actually re-rolls Path C generation + compose + grader; the exact
  exception classes (compose-fail = OutlineFailedError/known composer ValueError; config/model-init =
  RuntimeError/OSError => LOUD); ledger row->spoken-text shape; model_copy(deep) isolates intent mutation;
  canon is read-only during compose; the ComfyUI cancellation/interrupt API.
