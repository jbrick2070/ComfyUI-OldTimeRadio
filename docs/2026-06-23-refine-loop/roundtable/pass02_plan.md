# v1 Iterative Story-REVISION Loop -- build plan (pass02, post-R2 coding hardening)

Operator: a local loop that recursively REVISES the story until it reaches a target grade, then stops.
NOT best-of-N (that is shipped v0). NOT write-from-scratch -- a REWRITE of the existing story (spine
included when needed), seeded by the prior story + its graded weakness.

## Operator decisions (locked)
- ALWAYS a REVISION of the existing story, never blank-slate regeneration; the spine may change but is
  seeded by the prior story ("it's a rewrite not write from scratch ... at least it has some ideas to
  start with").
- Target grade = a NODE DROPDOWN widget: Off / C+ / B / B+ / A. DEFAULT = B. (Off = disabled = today's
  single path.) Widget goes IN `workflows/otr_scifi_16gb_full.json` (CLAUDE.md S0).
- Recurse until the grade reaches the target OR the hard cap; then ship keep-best. Plateau early-stop is
  OPT-IN (default OFF) so by default it keeps trying until the bar/cap.

## Code-grounded facts (verified this round -- do not re-litigate)
- **CRITICAL (R2 catch, CONFIRMED):** production `generate_outline` is Path C -- three structured calls
  using `_build_macro_user_prompt` (L1118, Stage 1 = title/premise/setting = the SPINE),
  `_build_phase_user_prompt` (L1142, speaker assign), `_build_beat_user_prompt` (L1191, Stage 3 = per-beat
  intent/mood = LOCAL). `_build_user_prompt` (L538) is BACK-COMPAT/TEST-ONLY (banner L1023; only callers
  are tests L2311/2329/2390). **=> shipped v0 `diversity_hint` (rendered only in `_build_user_prompt`,
  L604-609) is UNWIRED/DEAD in production -- best-of-N candidates differ only by RNG seed (this is why the
  smoke tied all 3).** v1 revision steering MUST wire into the Path C builders, and v0's diversity_hint
  should be wired there too (build chunk 0).
- The writer composes every line in-process (`_OTRLC.compose_line` ~L3761/3842/3919/4051); `build_sq_data`
  MUTATES `beat.intent` in place (~L654-659); `meta` is a dict (best_of_n telemetry landed as JSON);
  `OutlineRequest` is a frozen dataclass with required `budget`; `run_story_brief_reflection` (~L4530) is
  the in-writer local-LLM-call precedent for the grader.
- v0 `_otr_story_select.py`: `score_outline` (PURE, RAW intents, BEFORE build_sq_data), `_seed_rngs`
  (`int(sha256.hexdigest(),16)`, `torch.manual_seed(%2**64)`, `random.seed(%2**32)`), provider gate
  (ONLY `openrouter:`/`comfy:` are remote; everything else is local -- match this EXACTLY, no "unknown
  non-local" predicate exists).

## The revision loop (lives inside OTR_LedgerScriptWriter.run(), default-OFF, local-only)
Each PASS produces an ISOLATED `RefineCandidate`; only the WINNER commits to live writer state.

`RefineCandidate(pass_index, outline, raw_outline, structural_score, ledger, canon, meta_delta, title,
composed_text, grade, ok, error_type)`.

- **Pass 0 (baseline):** today's path -- `generate_outline` (Path C, no revision overlay => byte-identical)
  -> compose all lines -> grade. If `grade.score >= bar`: ship pass 0, done.
- **Pass i>=1 (revise):** feed the PRIOR winner's macro shape (title/premise/setting/beat intents) + the
  normalized critique into the Path C builders as a REVISION overlay -> `generate_outline` produces an
  improved outline (spine revised when the weakness is structural; beats tightened when local -- the LLM
  decides, seeded by the prior) -> compose -> grade. Keep-best by grade. Repeat until bar/cap.
- Seed per pass: `_seed_rngs(f"{cast_seed}:refine:{i}")` (exact v0 impl, `:refine:` namespace), set
  immediately before generation. Call `generate_outline` DIRECTLY (NOT `select_best_outline`, which would
  overwrite the seed with its own `:outline:0` and is best-of-N); wrap in `try/except OutlineFailedError`.

### Grader (NEW, standalone, read-only)
`grade_story(composed_text:str, premise:str, *, generate_fn, temperature=0.0) -> StoryGrade` where
`StoryGrade = @dataclass(frozen=True){score_0_100:int (clamped 0..100), biggest_weakness:str (trimmed)}`.
- Route through the existing `structured_call` ladder (same one `generate_outline` uses) with a small
  pydantic schema `{"score": int 0..100, "biggest_weakness": str Field(max_length=200)}` -> robust JSON
  parse + bounded retry for free. **temperature=0.0** (DETERMINISM: the grade drives early-stop +
  keep-best; a noisy grade makes the loop diverge). Seed before the call.
- `composed_text` = the SPOKEN dialogue lines pulled from the ledger rows (dialogue speaker_role in
  {character, announcer}; exclude music/sfx); define a max-char cap (summarize/truncate if over local
  context). Empty text => deterministic floor grade.
- Parse failure (ladder exhausts) => deterministic fallback `StoryGrade(score=0,
  biggest_weakness="grader_unparseable")` so keep-best still orders. A `grade_error_type` is recorded
  separately so the soak does not read a grader hiccup as a quality regression.

### Revision overlay wiring (Path C -- the R3 crux, drafted here)
- Add `prior_critique: str = ""` AND `prior_macro: str = ""` (the prior title/premise/setting digest) as
  the FINAL fields of the frozen `OutlineRequest` (after `diversity_hint`; defaulted; via
  `dataclasses.replace`). __post_init__ trims; no other validation.
- `_build_macro_user_prompt`: when `prior_macro`+`prior_critique` non-empty, emit a REVISE block ("Current
  premise/arc: <prior_macro>. Its biggest weakness: <prior_critique>. Revise to fix it, keeping what
  works; change the premise/arc only if the weakness is structural."). Empty => byte-identical.
- `_build_beat_user_prompt`: when `prior_critique` non-empty, emit a one-line "Address this weakness:
  <prior_critique>" steer. Empty => byte-identical.
- `critique_to_hint(biggest_weakness) -> str`: one line; strip control chars / backticks / JSON fences;
  collapse whitespace; reject/replace prompt-injection ("ignore previous/system/developer"); <=200 chars
  at a word/punctuation boundary (NOT a hard mid-word slice). Empty/malformed => "" (preserves
  byte-identical).
- **Build chunk 0 (v0 fix):** wire v0's `diversity_hint` into the SAME Path C builders (it is currently
  dead) with a test proving best-of-N candidates now actually differ. This unblocks both v0 and v1.

### Pass isolation (hard invariant)
Each pass deep-copies ALL mutable inputs before mutation: `outline.model_copy(deep=True)` for the pydantic
Outline (pydantic v2; `copy.deepcopy` only for non-pydantic), a FRESH `meta` dict, fresh ledger/canon
containers. `score_outline` runs on `raw_outline` (the pre-`build_sq_data` outline). NO pass merges into
the live `meta.story_quality` or writes artifacts/title/ledger; per-pass results live in a scalar
`refine_telemetry` object; only the WINNER commits (ledger/canon/title/meta + the merged telemetry). A
test asserts pass i+1 sees un-mutated intents and loser mutations never appear in the winner.

### Keep-best comparator (exact)
`winner = max(candidates, key=lambda c: (c.grade.score_0_100, -c.structural_score.ungrounded_crisis_density,
c.structural_score.distinct_conflict_nouns, c.structural_score.premise_grounding, -c.pass_index))`.
Never ship a revision graded worse than an earlier pass.

### Flags + gate
- Widget `refine_target_grade` (Off/C+/B/B+/A) -> bar via map {C+:68, B:75, B+:80, A:90}; default B. Env
  `OTR_STORY_REFINE_BAR` (int, clamp 0..100) + `OTR_STORY_REFINE_PASSES` (cap, clamp `REFINE_MAX_PASSES=5`)
  override for headless. Off/passes<2 => DISABLED (single path, byte-identical, no telemetry key).
- `resolve_refine_passes(resolved, *, env) -> RefineConfig{requested_passes, effective_passes, max_passes,
  bar, provider, clamp_reason}`. Provider gate matches v0 EXACTLY (only `openrouter:`/`comfy:` => clamp to
  1, LOUD). FLAG COLLISION: if `effective_passes>=2`, force best-of-N `effective_n=1` for the in-loop
  outline call (mutually exclusive; LOUD).

### Never-fail + failure taxonomy
compose-ok + grade-fail => RETAIN candidate at the floor grade (never discard a shippable story).
compose-fail (`OutlineFailedError` / known compose `ValueError`) => LOUD log + skip pass (keep-best
stands). A missing-model / config-init error => fail LOUD immediately (no pointless rerun). If EVERY pass
failed to compose => ONE deterministic fallback = the current byte-identical single path at
`:refine:0`; if THAT raises => LOUD. (Verify-at-build: enumerate the exact exception classes from the
writer/backend.)

### Telemetry (merged on winner-commit only; never replace `story_quality`)
`meta.story_quality.refine_loop = {requested_passes, effective_passes, max_passes, bar, target_grade,
stopped_early, stop_reason, winner_pass, winner_grade, provider, clamp_reason, passes:[{pass_index, ok,
score|grade_error_type, normalized_hint, structural_score, grade_delta, elapsed_s}]}`. `grade_delta` = None
at pass 0, else `score - previous_ok_score`. `normalized_hint` (not raw critique) for soak + leak safety.

## Invariants
- Audio frozen + byte-identical when OFF: assert the current single outline+compose path runs, the THREE
  Path C prompt builders are byte-identical with empty overlays, and NO `refine_loop` key exists.
- Pass isolation; build_sq_data runs once PER COMPOSED CANDIDATE in isolated state; freeze + audio run
  ONCE on the committed winner; losers write nothing.
- Local-only (remote => 1 pass); deterministic (cast_seed-keyed + grader temp 0); HARD CAP + keep-best;
  LOUD on real errors; the dropdown widget change goes IN the workflow JSON (S0) in the SAME commit as the
  code; UTF-8 no BOM; SFW.

## Build order (each chunk: full suite + Bug Bible green + JSON validated when touched, commit+push)
0. WIRE `diversity_hint` into the Path C builders (fix the shipped v0 dead-code bug) + a test proving
   best-of-N candidates now differ. Byte-identical when empty.
1. `OutlineRequest.prior_critique` + `prior_macro` (final fields) + Path C REVISE overlays (macro + beat),
   byte-identical when empty + `critique_to_hint` sanitizer + tests.
2. `grade_story` + `StoryGrade` (structured_call, temp 0, extraction, clamp, fallback) + tests.
3. `_build_and_compose(...)` pass-local helper returning an ISOLATED `RefineCandidate` (deep-copy all
   inputs, commit nothing); single-pass path byte-identical (asserted).
4. `refine_story(...)` loop + `resolve_refine_passes` (flag/widget/bar/provider gate + best_of_n mutual
   exclude) + keep-best + revision overlay threading + never-fail + merged telemetry, wired into run()
   (loop ONLY when effective_passes>=2). Add the `refine_target_grade` widget to the node + the JSON.
   Flag-off call-count + no-key + byte-identical (3 Path C prompts + ledger) tests.
5. VALIDATION soak (operator GPU): pick B in the dropdown; measure grade LIFT (per-pass grade_delta) +
   v0 sameness metrics vs baseline; operator written go/no-go. Default-OFF + unadvertised until it passes.

## CUT (R1+R2 consensus)
- Remote opt-in; nested best-of-N inside a pass; loser-ledger retention; downstream-critic reuse;
  `OTR_STORY_REFINE_MAX_SECONDS` (hard pass cap suffices for v1 -- re-add only if the soak shows long
  passes); plateau "no-improve" early-stop (opt-in, default OFF per operator).

## Verify-at-build
- cast_seed in scope at the loop; per-pass reseed actually re-rolls Path C generation + the grader;
  exact exception classes for the failure taxonomy; `model_copy(deep=True)` fully isolates intent
  mutation; the ledger row->spoken-text extraction for the grader; `meta` is a dict at the loop.
