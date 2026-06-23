# v1 Iterative Story-REVISION Loop -- FINAL build-ready plan (R4 converged)

4-round live roundtable: GPT-5.5 + Gemini-3.1-pro + DeepSeek-v4-pro, Claude code-grounded judge + panelist.
Spend: R1 $0.0931 + R2 $0.1790 + R3 $0.1777 + R4 $0.0704 = ~$0.5202. (R4 DeepSeek errored on reasoning-token
exhaustion; GPT + Gemini both returned yes-with-fixes = converged.) Artifacts: pass00..pass04 +
pass0N_judgment.md + per-model reviews + OPERATOR_INPUTS.md.

Goal (operator): a LOCAL loop that recursively REVISES the existing story until it reaches a target grade,
then ships. NOT write-from-scratch (a rewrite seeded by the prior story + its graded weakness; spine only
when needed). DISTINCT from shipped v0 best-of-N (v0 = best of N independent drafts; v1 = iteratively
revise ONE evolving draft).

## What the roundtable changed (the value)
The panel, grounded against the real code, caught a SHIPPED v0 BUG and several would-break-at-the-keyboard
issues: (1) **v0 `diversity_hint` is DEAD** -- rendered only in `_build_user_prompt`, which is
back-compat/test-only; production is Path C (`_build_macro/phase/beat_user_prompt`) -- so best-of-N
candidates vary only by RNG seed (this is why the smoke tied all 3). (2) per-pass `meta` must be a
DEEP-COPY of the incoming meta, not fresh (`score_outline -> premise_texts(meta)` reads news/brief context).
(3) `structured_call` has no `temperature` kwarg. (4) wire revision steering into MACRO + BEAT only, never
the speaker-only phase prompt. (5) three seed points per pass (generate / compose / grade). The REVISE
architecture HELD across all 4 rounds.

## Operator decisions (LOCKED)
- ALWAYS a REVISION seeded by the prior story + its graded weakness; spine may change but is never blank.
- Target grade = NODE DROPDOWN `refine_target_grade` (Off / C+ / B / B+ / A), **default Off** (preserves the
  byte-identical default-OFF invariant + the frozen audio golden; every other OTR quality flag is default-OFF).
  **B is the recommended pick when the operator enables it.** Widget goes IN otr_scifi_16gb_full.json (S0).
- Recurse until the grade reaches the target OR the hard cap; then ship keep-best. Plateau early-stop is
  NOT in v1 (future opt-in only).

## Code-grounded facts (verified R1-R4 -- do not re-litigate)
- Path C = `_build_macro_user_prompt` (Stage 1: title/premise/setting = SPINE), `_build_phase_user_prompt`
  (Stage 2: SPEAKER ASSIGN ONLY -- never inject critique here), `_build_beat_user_prompt` (Stage 3:
  intent/mood = LOCAL). `_build_user_prompt` is back-compat/test-only.
- `score_outline(outline, meta, roster)` -> `premise_texts(meta)` needs the incoming meta context.
  `build_sq_data` mutates `beat.intent` in place. `meta` is a dict. `OutlineRequest` is FROZEN.
- `structured_call(prompt, schema, slot_fn, base_temperature, structural_retry_temperature,
  repair_prompt_factory, max_new_tokens, max_attempts, helper_name)` -- NO `temperature` kwarg.
- `_seed_rngs` = `int(sha256.hexdigest(),16)`, `torch.manual_seed(%2**64)`, `random.seed(%2**32)`.
- Provider gate: ONLY `openrouter:`/`comfy:` (on `resolved["creative_writing_model"]`) are remote.

## The revision loop (inside OTR_LedgerScriptWriter.run(), default-OFF, local-only)
`RefineCandidate(pass_index, raw_outline, structural_score, ledger, canon, meta_local, composed_text,
grade, ok, error_type, grade_error_type)`. (CUT meta_delta, elapsed_s, title -- title is regenerated on
the winner by the existing post-compose path; prior_macro uses the OUTLINE title.)

**Pass semantics (LOCKED):** `effective_passes` = TOTAL candidate count INCLUDING the mandatory pass 0.
Disabled => `effective_passes=1` (single path). Enabled minimum = 2. Loop `for i in range(effective_passes)`:
pass 0 = baseline, passes 1..N-1 = revisions. Clamp to `REFINE_MAX_PASSES = 5` TOTAL (incl. pass 0).

- **Pass 0 (mandatory baseline = today's path):** `generate_outline` (Path C, empty overlays =>
  byte-identical) -> compose -> grade. If pass 0 COMPOSE fails, fail exactly as today (LOUD); the
  "all-passes-failed fallback" branch is CUT (pass 0 success guarantees a shippable candidate => never-fail
  is automatic). If `grade.score_0_100 >= bar`: ship pass 0, stop.
- **Pass i>=1 (REVISE):** feed `prior_macro` (prior winner's Title/Premise/Setting + numbered raw beat
  intents, capped) + normalized `prior_critique` into the MACRO + BEAT prompts -> `generate_outline`
  (DIRECTLY, never `select_best_outline`) -> compose -> grade. Keep-best. Repeat to bar/cap.
- **Three seed points per pass:** `_seed_rngs(f"{cast_seed}:refine:{i}")` before generation,
  `:refine:{i}:compose` before line composition, `:refine:{i}:grade` before grading.
- **Progress visibility:** one LOUD line per pass `[refine] pass i/N grade=NN target=BB`.

### Outline isolation (exact)
After `generate_outline`: `raw_outline = outline.model_copy(deep=True)`; `structural_score =
score_outline(raw_outline, meta_local, roster)`; `working_outline = outline.model_copy(deep=True)` for
build_sq_data + compose. Build the NEXT pass's `prior_macro` from THIS `raw_outline` (pre-mutation intents).

### Grader (NEW, standalone, read-only)
`grade_story(composed_text, premise, *, generate_fn) -> StoryGrade`; `StoryGrade =
@dataclass(frozen=True){score_0_100:int, biggest_weakness:str}` (map parsed schema `.score` ->
`.score_0_100`; use `.score_0_100` everywhere in loop/comparator/telemetry).
- `structured_call(prompt=[system,user], schema=_StoryGradeSchema{score:int 0..100, biggest_weakness:str
  Field(max_length=200)}, slot_fn=generate_fn, base_temperature=0.0, structural_retry_temperature=0.0,
  repair_prompt_factory=make_dispatching_repair_factory(), max_new_tokens=128, max_attempts=2,
  helper_name="OTR_StoryGrade")`.
- Rubric: the single biggest STRUCTURAL/dramatic weakness (arc / stakes / premise grounding / character
  want) -- NOT a line edit.
- `composed_text = extract_spoken_text_for_grade(ledger)`: "SPEAKER: line" per voiced row
  (character/announcer; exclude music/sfx). Cap at 4000 chars = `first 2000 + "\n...\n" + last 2000`.
  Empty => floor grade.
- Parse failure => candidate stays `ok=True`, `grade.score_0_100=0`, `grade_error_type="grader_unparseable"`,
  next pass `normalized_hint=""`. (`ok` = generate+compose succeeded/shippable; `error_type` is ONLY
  generate/compose failure; the comparator includes ALL `ok=True` candidates even if the grade failed.)

### Revision overlay wiring (Path C)
- `OutlineRequest` gains `prior_critique: str = ""` and `prior_macro: str = ""` as FINAL fields (after
  `diversity_hint`). NO __post_init__ mutation (frozen) -- normalize in `critique_to_hint` before
  `dataclasses.replace`. Test: `select_best_outline`'s `dataclasses.replace(req, diversity_hint=...)` still
  works after the new fields.
- `_build_macro_user_prompt`: when `prior_macro` non-empty, emit a REVISE block. With a critique:
  "Current premise/arc: <prior_macro>. Biggest weakness: <prior_critique>. Revise to fix it, keep what
  works; change premise/arc only if the weakness is structural." With NO usable critique (grade-fail):
  "Current premise/arc: <prior_macro>. Improve the structure while preserving the prior spine." (=> a
  grade-fail pass STILL revises, honoring "always a revision"; never blank.) Empty prior_macro =>
  byte-identical.
- `_build_beat_user_prompt`: when `prior_critique` non-empty, one line "Address this weakness:
  <prior_critique>". Empty => byte-identical. (Phase prompt UNTOUCHED.)
- `prior_macro` = `"Title: {t}\nPremise: {p}\nSetting: {s}\nBeats: " + "; ".join(numbered raw voiced
  intents)`, capped.
- `critique_to_hint(biggest_weakness) -> str`: single line (strip newlines + control chars + backticks +
  JSON fences; collapse whitespace); reject prompt-injection by NARROW regex `ignore (all
  )?(previous|system|developer) (instructions|messages|prompt)`; <=200 chars at a word boundary.
  Empty/malformed => "".

### Pass isolation + winner commit
`_build_and_compose(...)` takes PASS-LOCAL ledger/canon/meta containers, RETURNS a `RefineCandidate`, and
NEVER assigns `self.ledger`/`self.canon`/`self.title`/`meta.story_quality` or writes artifacts. Deep-copy
the mutated inputs (outline via model_copy(deep=True); meta via `copy.deepcopy(incoming_meta)`). canon: test
that a losing candidate leaves canon equal; deep-copy ONLY if the composer mutates it. Sequencing: loop ->
pick winner -> COMMIT ONCE post-loop (existing post-compose title regen ~L4356 + canon write + FreezeCascade
reroll-context meta threading ~L3440-3606 + ledger save run here on the WINNER). Test: a losing pass cannot
change live title/ledger/meta.

### Keep-best comparator (exact)
`winner = max(cands, key=lambda c: (c.grade.score_0_100, -c.structural_score.ungrounded_crisis_density,
c.structural_score.distinct_conflict_nouns, c.structural_score.premise_grounding, -c.pass_index))`.

### Flags + gate + collision (explicit branches)
`resolve_refine_passes(resolved, *, env, widget_target) -> RefineConfig{requested_passes, effective_passes,
max_passes, bar, target_grade, provider, clamp_reason, override_source}`. Bar from widget map
{C+:68,B:75,B+:80,A:90}; env `OTR_STORY_REFINE_BAR`/`OTR_STORY_REFINE_PASSES` OVERRIDE the widget (record
`override_source`). ABSENT widget (old JSON) => Off. provider from `resolved.get("creative_writing_model","")`
with the v0 prefix check.
- **refine DISABLED / Off / remote-clamped:** run the existing pre-v1 story path EXACTLY -- including the
  existing best-of-N resolution if `OTR_STORY_BEST_OF_N` is configured (v0 behavior PRESERVED).
- **refine ENABLED (effective_passes>=2):** BYPASS `resolve_best_of_n`/`select_best_outline` for the story
  path entirely (LOUD); assert NO `best_of_n` telemetry key.

### Cancellation (required chunk-4 step)
Identify the existing ComfyUI interrupt API (verify: `model_management.throw_exception_if_processing_
interrupted` / `interrupt_current_processing` or equivalent); call it before each nonzero pass and before
each grade; cancellation aborts LOUD without committing any candidate.

### Telemetry (winner-commit only; never replace `story_quality`)
`meta.story_quality.refine_loop = {requested_passes, effective_passes, max_passes, bar, target_grade,
override_source, target_reached:bool, stopped_early, stop_reason ("bar_reached" | "cap_reached_below_bar"),
winner_pass, winner_grade, provider, clamp_reason, passes:[{pass_index, ok, score_0_100, grade_error_type,
normalized_hint, structural_score, grade_delta}]}`. `grade_delta` = None at pass 0, else `score_0_100 -
pass0_score_0_100` (grader-failed pass uses 0). `stop_reason="cap_reached_below_bar"` => a LOUD warn so the
operator sees the chosen grade is unreachable for this model (dial it down).

## Invariants
- Audio frozen + byte-identical when OFF: assert the current single path runs, ALL THREE Path C builders are
  byte-identical across the 4 overlay combos (both empty / only diversity_hint / only prior_critique / both),
  NO grade calls, NO refine_loop key, and existing best-of-N behavior is unchanged when refine is disabled.
- Pass isolation (losers commit nothing); build_sq_data per candidate in isolated state; freeze + audio ONCE
  on the winner. Local-only; deterministic (3 seed points + grader temp 0); HARD CAP + keep-best; LOUD on
  real errors; widget IN the JSON same commit (S0, append at END + re-validate); UTF-8 no BOM; SFW.

## HONEST FLOOR (state at delivery; do not oversell)
The grader reuses the SAME weak local writer that wrote the story -- a weak model grading itself is lenient/
noisy. If the loop does not raise the grade across passes, v1 degrades to "keep-best of N revisions" (still
ships the best it found, never improves). v1 ships default-OFF and is NOT advertised as raising quality
until the validation soak shows a real per-pass grade lift.

## Build order (each chunk: full suite + Bug Bible green + JSON validated when touched, commit+push to v2.0-alpha)
0. Wire `diversity_hint` into `_build_macro_user_prompt` + `_build_beat_user_prompt` (fix the shipped v0
   dead-code bug) + a test proving best-of-N candidates now differ. Byte-identical when empty.
1. `OutlineRequest.prior_critique` + `prior_macro` (final fields) + MACRO+BEAT REVISE overlays (incl. the
   no-critique variant) + `critique_to_hint` + 4-combo byte-identical (3 prompts) tests + replace() compat.
2. (BLOCK on reading `production_ledger` row shape first) `extract_spoken_text_for_grade` + `grade_story` +
   `StoryGrade` (real structured_call interface, temp 0, max_new_tokens 128, 2000+2000 cap, floor fallback)
   + tests on a real/minimal ledger fixture.
3. `_build_and_compose` pass-local helper (deep-copy outline+meta, 3 seed points, isolated RefineCandidate,
   commits nothing) + byte-identical single-pass test (in THIS chunk).
4. `refine_story` loop + `resolve_refine_passes` (flag/widget/bar/provider gate + best_of_n bypass) +
   keep-best + winner-commit sequencing + never-fail (pass-0 mandatory) + cancellation hook + merged
   telemetry, wired into run(). Add the `refine_target_grade` widget to the node + the JSON (default Off).
   Flag-off call-count + no-key + byte-identical + "no best_of_n key when refine on" tests.
5. VALIDATION soak (operator GPU): pick B; measure grade LIFT (grade_delta) + v0 sameness vs baseline;
   operator written go/no-go. Default-OFF + unadvertised until it passes.

## Verify-at-build checklist (confirm each before/at the touching chunk)
1. `cast_seed` in scope in run() + the refine call path.
2. All three reseed points actually affect Path C generation, line composition, AND the grader.
3. `_seed_rngs` uses the v0 hashing (or reuse the helper).
4. Exact exception classes: generation/compose failure (OutlineFailedError / known composer ValueError) =>
   failed candidate; config/model-init (RuntimeError/OSError) => LOUD, never swallowed.
5. `structured_call` takes `base_temperature` (not `temperature`); `make_dispatching_repair_factory` is
   importable at the grader location.
6. Real ledger row shape for `extract_spoken_text_for_grade` (character+announcer in, music/sfx out).
7. `outline.model_copy(deep=True)` exists for the Outline type + isolates beat-intent mutation.
8. `prior_macro` built from raw_outline BEFORE build_sq_data.
9. per-pass `meta_local` is a deepcopy of incoming meta preserving news/brief/style.
10. canon is read-only during compose (else deep-copy it per candidate).
11. The post-compose title regen / canon write / FreezeCascade meta threading / ledger save can be delayed
    to a single winner-commit.
12. The exact ComfyUI interrupt API + behavior.
13. Provider gate prefixes match v0 (only openrouter:/comfy:).
14. Absent widget in old JSON => Off; chunks 1-3 tests run without the widget.
15. Widget appended to widgets_values at END, UTF-8 no BOM, JSON validates (OTR_WorkflowValidator).
16. Flag-off run is byte-identical, no grade calls, no refine key, and v0 best-of-N unchanged.
