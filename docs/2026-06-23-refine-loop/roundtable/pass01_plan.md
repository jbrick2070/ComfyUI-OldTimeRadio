# v1 Holistic Story-Refine Loop -- build plan (pass01, post-R1 arc hardening)

Operator ask: a local story-refine loop that "keeps improving the story and never stops" (the
"10th-grade B+? how do we make it better?" idea). This is the DEFERRED v1 of the best-of-N work.
v0 (shipped 4dc631a..4593bc5) is the one-shot structural OUTLINE selector and does NOT iterate.

## Code-grounded facts (do not re-litigate)
- `OTR_LedgerScriptWriter.run()` builds the WHOLE script in-process: outline (v0 best-of-N at ~L2709)
  -> `build_sq_data` F2 (~L2748) -> `episode_canon` + ledger skeleton (~L2871) -> composes EVERY line
  in-process via `_OTRLC.compose_line` (~L3761/3842/3919/4051) -> post-compose title regen -> save
  ledger. So the v1 loop is an IN-FUNCTION wrap in ONE node, like v0.
- `build_sq_data` MUTATES `beat.intent` in place (`setattr(b,"intent",new_intent)`, ~L654-659) and
  substitutes the generic crisis nouns -> looping it on shared beat objects CORRUPTS later passes.
- v0 `OutlineRequest.diversity_hint` is an INDEX-KEYED structural instruction (`_DIVERSITY_HINTS`),
  rendered by `_build_user_prompt` only when non-empty. v1 critique text must NOT reuse this field.
- The story CRITIC + line reroll + doctor edits live DOWNSTREAM in `OTR_LedgerFreezeCascade` (a
  separate node). `run_story_brief_reflection` (~L4530) is an existing in-writer local-LLM reflection
  call -- the precedent for an in-writer grader.
- v0 `nodes/_otr_story_select.py`: `score_outline` (PURE, RAW intents, computed BEFORE build_sq_data),
  `select_best_outline`, `resolve_best_of_n` (flag + provider gate). Audio spine FROZEN.

## What "never stops" means
Never HARD-FAIL on weak quality -- NOT an unbounded loop. The loop is bounded by a HARD CAP (and an
optional wall-clock budget); after the cap it ships the best version found (keep-best). The
quality-floor abort is removed; genuine errors still fail LOUD.

## HONEST FLOOR (state plainly; do not oversell)
The improvement mechanism is critique-informed regeneration. If a weak local model cannot turn its own
critique into a structurally better outline (the failure the prior panel named), v1 degrades GRACEFULLY
to "keep-best best-of-N at the COMPOSED level" -- it still ships the best of N composed stories, it just
does not truly *improve*. v1 must NOT be advertised as raising quality until the validation soak shows a
real grade lift. (Governance: this reverses pass04_plan_FINAL's "build v1 only after v0 validates" --
operator-directed; v1 ships behind a default-OFF flag, soak is post-build.)

## Lead design (R1-hardened)
Wrap `[build outline -> compose all lines -> grade]` in a bounded refine loop INSIDE
`OTR_LedgerScriptWriter.run()`, default-OFF, local-only, with strict PASS ISOLATION.

**Flags.**
- `OTR_STORY_REFINE_PASSES`: unset/0/1 => DISABLED (one pass == today, byte-identical); int >= 2 => up
  to N passes, clamped to `REFINE_MAX_PASSES = 5`.
- `OTR_STORY_REFINE_BAR`: target grade 0-100, default 80 (B+); parse/clamp (blank/non-int => 80).
- (optional) `OTR_STORY_REFINE_MAX_SECONDS`: wall-clock budget; 0/unset => off.
- Provider gate (reuse v0's exact classification): if `creative_writing_model` starts with
  `openrouter:`/`comfy:` => clamp to 1 pass, LOUD; fail CLOSED (clamp) for any unknown non-local handle.
  No remote in v1 (a pass is N full composes = N x many calls).

**PASS ISOLATION (hard invariant -- the R1 catch).** Each pass builds into an ISOLATED candidate:
deep-copy the outline BEFORE `build_sq_data` (which mutates intents); use pass-local meta / canon /
ledger snapshots. NO pass writes final artifacts or mutates winner-visible writer state except its own
candidate-local telemetry. Only the WINNER's outline/ledger/canon/title/meta is committed to the live
writer; losers are discarded (scalar telemetry only, no loser-ledger retention).

**The loop (pass i in range(effective_passes)):**
1. seed: `h = sha256(f"{cast_seed}:refine:{i}")`; seed torch+random immediately before generation
   (mirrors v0 `_seed_rngs`). cast_seed is the writer's existing cast RNG seed (verify in scope).
2. Build EXACTLY ONE outline (best-of-N NOT nested -- `effective_n=1` inside the loop). i==0:
   `prior_critique=""` (today's path). i>=1: `prior_critique =` the normalized weakness from pass i-1.
3. `score_outline(outline, meta, roster)` on the RAW intents (BEFORE build_sq_data) -> structural score.
4. On a deep-copied outline: `build_sq_data` + canon + compose ALL lines (the existing in-run path,
   factored into a pass-local helper `_build_and_compose(...)`).
5. GRADE: a NEW standalone read-only `grade_story(composed_lines, premise, meta) -> StoryGrade{
   score_0_100:int, biggest_weakness:str}` -- a lean local rubric prompt modeled on
   `run_story_brief_reflection`. NOT the downstream `_otr_story_critic` (separate node; reroll
   side-effects; circular-import risk). Robust parse: non-numeric/degenerate => deterministic low score.
6. Critique normalization: `critique_to_hint(biggest_weakness) -> bounded structural hint` (prefix
   "IMPROVEMENT HINT: ", trim <= 200 chars, no extra formatting) -> set as the next pass's
   `prior_critique` (a NEW `OutlineRequest` field, rendered by `_build_user_prompt` only when non-empty;
   empty => byte-identical; v0's `diversity_hint` stays untouched).
7. Early-stop: stop if `score >= REFINE_BAR`, OR if the grade has not improved for 2 consecutive passes,
   OR if the wall-clock budget is exceeded. Else keep-best and continue.
8. Keep-best comparator (deterministic): `(grade.score desc, structural_score: ungrounded_crisis_density
   asc / -distinct_conflict_nouns / -premise_grounding, pass_index asc)`.
9. After the loop: commit the keep-best candidate to the live writer; downstream freeze cascade
   (critic + reroll + doctor) + audio run EXACTLY ONCE on the winner -- v1 does not touch that path.

**Never-fail (R1-hardened).** compose-ok + grade-fail => RETAIN the candidate with a deterministic
low/unknown grade (a shippable story is never discarded for a grader hiccup). compose-fail => LOUD log +
skip that pass (keep-best stands). If EVERY pass failed to compose => ONE deterministic fallback
(`sha256(f"{cast_seed}:refine:0")` seed, `prior_critique=""`); if THAT raises => fail LOUD. A
missing-model / config error class fails LOUD immediately (no pointless rerun).

**Determinism.** cast_seed-keyed per-pass seeds; pure comparator; early-stop is data-dependent but
deterministic for fixed (seed, model, flag). VERIFY-AT-BUILD (carried from v0): that per-pass reseeding
actually re-rolls/repeats generation + grader output.

**Telemetry (merged; never replace `story_quality`).** `meta.story_quality.refine_loop = {requested_passes,
effective_passes, max_passes, bar, stopped_early, stop_reason, winner_pass, winner_grade, provider,
passes:[{pass_index, ok, score|error_type, biggest_weakness, normalized_hint, structural_score,
grade_delta}]}`. The `grade_delta` per pass is what the soak uses to answer "did it actually improve?".

## Invariants (may not break)
- Audio frozen + byte-identical when OFF: assert exactly the current single outline+compose path runs and
  NO `meta.story_quality.refine_loop` key exists when disabled.
- Pass isolation: losers never mutate winner-visible state or write artifacts.
- `build_sq_data` may run once PER COMPOSED CANDIDATE inside isolated pass state; the freeze cascade and
  audio run ONCE only, on the committed winner. (Corrected from pass00's "build_sq_data once total".)
- Local-only (remote => 1 pass, fail-closed); deterministic (cast_seed-keyed); HARD CAP + keep-best
  (never unbounded); LOUD on real errors; zero `otr_scifi_16gb_full.json` change (internal to the writer;
  if passes/bar become widgets they go IN the JSON same change); UTF-8 no BOM; SFW.

## Build order (code-first; each chunk: full suite + Bug Bible green + no-JSON-drift, commit+push)
1. Refactor `[build outline -> score_outline -> build_sq_data(deep-copy) -> canon -> compose-all-lines]`
   into a pass-local helper `_build_and_compose(...)` returning an ISOLATED candidate (composed ledger +
   structural score), committing nothing. Pure refactor: single-call path byte-identical (asserted).
2. `OutlineRequest.prior_critique: str = ""` + `_build_user_prompt` render only when non-empty
   (byte-identical when ""), + `critique_to_hint` normalizer. Tests.
3. `grade_story(...) -> StoryGrade` lean local rubric (read-only) + robust parse/fallback + tests
   (deterministic for fixed seed/model; non-degenerate score on a real composed story).
4. `refine_story(...)` loop + `resolve_refine_passes` (flag + bar + provider gate) + keep-best +
   critique-normalized prior_critique + early-stop (bar / no-improve-2 / wall-clock) + never-fail +
   pass isolation + merged telemetry, wired into run() (loop ONLY when effective_passes>=2; else the
   byte-identical single path). Flag-off call-count + no-key + byte-identical tests.
5. VALIDATION soak (operator GPU): `OTR_STORY_REFINE_PASSES=3` vs baseline; measure grade LIFT (per-pass
   grade_delta) + the v0 sameness metrics; operator written go/no-go. v1 stays default-OFF + unadvertised
   until this passes.

## CUT (R1 consensus)
- Remote opt-in / cost guard for v1 (a refine pass is N full composes; local-only, period).
- Nested v0 best-of-N inside a refine pass (O(N*M) blowup; one outline per pass).
- Exhaustive per-loser ledger retention (scalar telemetry + winner artifact only).
- Reusing the downstream `_otr_story_critic` in the loop (build the lean rubric instead).

## Verify-at-build (carried)
- cast_seed is in scope at the loop and is the writer's cast RNG seed.
- per-pass torch/random reseed actually changes generation + grader output (v0 unverified item).
- all full-compose creative providers use only `openrouter:`/`comfy:` prefixes (else fail closed).
- `_build_and_compose` deep-copy fully isolates beat-intent mutation (no shared-object leak across passes).
