# v1 Holistic Story-Refine Loop -- build plan (pass00, pre-roundtable draft)

Operator ask (2026-06-23): a story-refine loop that "keeps running until it improves the
story and never stops" -- the original "is this a good story, 10th-grade B+? how do we make
it better?" idea. This is the **v1** half DEFERRED from the best-of-N work; v0 (shipped
4dc631a..4593bc5) is the one-shot structural OUTLINE selector and does NOT iterate.

## What is already true (code-grounded; do not re-litigate)
- `OTR_LedgerScriptWriter.run()` builds the WHOLE script in-process: outline (v0 best-of-N
  selector wired at ~L2709) -> `build_sq_data` F2 (~L2748) -> `episode_canon` + ledger skeleton
  (~L2871) -> **composes every dialogue line in-process** via `_OTRLC.compose_line`
  (~L3761/3842/3919/4051) -> post-compose title regen -> save ledger. So a loop around
  `[outline -> compose-all-lines -> grade]` is an IN-FUNCTION wrap in ONE node, exactly like v0.
- v0 lives in `nodes/_otr_story_select.py`: `score_outline(outline, meta, roster) -> StoryScore`
  (PURE, raw intents, no build_sq_data), `select_best_outline(...)` (cast_seed-keyed sha256 seeds
  + structural `diversity_hint` for i>=1, keep-best, deterministic never-fail), `resolve_best_of_n`
  (flag parse + provider gate), and the chunk-4 remote cost guard. `OutlineRequest.diversity_hint`
  renders only when non-empty (byte-identical when "").
- The story CRITIC + line-level reroll + doctor edits run DOWNSTREAM in `OTR_LedgerFreezeCascade`
  (a separate node); the writer threads reroll-context onto `meta` (~L3440-3606) so the cascade can
  reconstruct a `LineRequest`. There is ALSO `run_story_brief_reflection` (~L4530) -- an existing
  post-compose local-LLM reflection call. So a "grade the composed story" hook already has precedent
  in-node.
- Audio spine is FROZEN (`test_audio_byte_identical`); the canonical workflow is
  `workflows/otr_scifi_16gb_full.json`.

## The v1 idea (what "never stops" really means)
"Never stops" = never HARD-FAIL on weak quality. It is NOT an unbounded loop. The loop is bounded
by a HARD CAP; after the cap it ships the best version it found (keep-best). Quality-floor abort is
removed; genuine errors (crash / missing model / all passes failed) still fail LOUD.

The improvement mechanism is NOT blind reroll (the prior roundtable already rejected "the weak model
rephrases the same standoff"). It is CRITIQUE-INFORMED regeneration: the grader returns the single
biggest weakness, and that text is fed into the NEXT pass's `diversity_hint` (the existing v0 prompt
overlay) so the next outline is steered, not re-rolled blind.

## Lead design (for the panel to harden)
Wrap the writer's `[build outline -> compose all lines]` block in a bounded refine loop, INSIDE
`OTR_LedgerScriptWriter.run()`, default-OFF.

**Flag.** `OTR_STORY_REFINE_PASSES`: unset/0/1 => DISABLED (one pass == today, byte-identical);
int >= 2 => up to N passes, clamped to a hard cap (`REFINE_MAX_PASSES = 5`). Read once at top of run().
Local-only: if `resolved["creative_writing_model"].startswith(("openrouter:","comfy:"))` => clamp to
1 pass, LOUD (a refine pass is a FULL compose = many paid calls; no remote in v1 -- revisit later with
a cost guard like v0 chunk 4).

**The loop (pass i in range(effective_passes)):**
1. cast_seed-keyed seed: `h = sha256(f"{cast_seed}:refine:{i}")`; seed torch+random immediately before
   generation (mirrors v0 `_seed_rngs`).
2. Build the outline. i==0: `diversity_hint=""` (today's path). i>=1: `diversity_hint =` the prior
   pass's grader critique (the "make it better" steer). The outline build MAY itself be the v0
   best-of-N selector (compose only the structural winner) -- see Open Question A.
3. `build_sq_data` + canon + compose ALL lines (the existing in-run code, factored into a helper so
   the loop can call it per pass).
4. GRADE the composed story: a LOCAL holistic rubric call `grade_story(composed_lines, premise, meta)
   -> StoryGrade{score_0_100:int, biggest_weakness:str}`. Reuse the `_otr_story_critic` machinery if it
   yields a numeric score; else a new lean rubric prompt (see Open Question B).
5. Early-stop: if `score >= REFINE_BAR` (default B+ ~= 80/100), STOP (good enough) -- record the pass.
   Else keep-best and continue.
6. Keep-best comparator (deterministic): `(grade.score desc, structural StoryScore tie-break:
   ungrounded_crisis_density asc / -distinct / -grounding, pass_index asc)`.
7. After the cap: ship the keep-best composed ledger. Downstream (freeze cascade critic+reroll+doctor,
   audio) runs EXACTLY ONCE on the winner -- v1 does not change the freeze/audio path.

**Never-fail.** If a pass raises during compose/grade -> LOUD log + skip that pass (keep-best stands).
If EVERY pass raised -> ONE deterministic fallback pass (i=0 seed, hint="") then LOUD-fail if that
raises too (mirrors v0's never-fail).

**Determinism.** cast_seed-keyed per-pass seeds; pure comparator; early-stop is data-dependent but
deterministic for a fixed (seed, model). Telemetry records the actual pass count.

**Telemetry (merged, never replace `story_quality`).** `meta.setdefault("story_quality",{})["refine_loop"]
= {requested_passes, effective_passes, max_passes, bar, stopped_early, winner_pass, winner_grade,
passes:[{pass_index, ok, score|error_type, biggest_weakness, structural_score}], provider}`.

**Invariants (may not break).** Audio frozen + byte-identical when OFF (assert: exactly ONE
outline+compose path, no `refine_loop` key when disabled); local-only by default; deterministic;
LOUD on real errors; keep-best + HARD CAP (never an unbounded loop); zero `otr_scifi_16gb_full.json`
change (internal to the writer; if passes/bar become widgets they go IN the JSON same change); UTF-8
no BOM; SFW; build_sq_data + freeze + audio still run ONCE on the winner.

## Open questions for the roundtable (where I want convergence)
- **A. Compose-cost vs best-of-N reuse.** Each pass is a FULL compose (minutes of local LLM). Should a
  pass first run v0 best-of-N to pick the structural-best outline and compose only THAT (cheaper), or
  compose one outline per pass? Does the v0 structural pre-filter meaningfully cut wasted composes?
- **B. The grader.** Reuse `_otr_story_critic` (already produces an arc verdict) for the numeric grade,
  or a new lean B+ rubric LLM call? Risk: a weak local model grading its own output is unreliable
  (lenient/noisy). Does the grade discriminate enough to drive keep-best? (The v0 30-word smoke showed
  the STRUCTURAL scorer tied all candidates -- the holistic grade must do better.)
- **C. Does critique-informed regeneration actually improve a weak local writer,** or does it rephrase
  the same beats (the failure mode the prior panel named)? If it cannot, v1 collapses to "best-of-N at
  the composed level" (keep-best, no real improvement) -- still useful, but be honest about it.
- **D. Overlap with the freeze-cascade reroll.** The cascade already does LINE-level grade->reroll->doctor
  edits. v1 is OUTLINE-level regenerate. Do they compose well, or does running the full critic per pass
  double the cost? Should v1's per-pass grade be a LIGHT rubric and leave the heavy critic to the single
  downstream freeze?
- **E. Runtime ceiling.** With cap 5 and ~minutes/compose, worst case is ~5x writer time. Acceptable for a
  default-OFF opt-in? Is early-stop-on-bar enough, or do we also need a wall-clock budget?
- **F. Validation gate.** The converged v0 plan gated v1 behind a measurement soak proving the rubric
  discriminates. Operator has asked to build now -- so v1 ships behind a default-OFF flag and the soak
  becomes post-build validation (same pattern as v0).

## Build order (code-first, each chunk: full suite + Bug Bible green + no-JSON-drift, commit+push)
1. Factor the writer's `[outline -> build_sq_data -> canon -> compose-all-lines]` into a callable helper
   `_build_and_compose(...)` returning the composed ledger + the StoryScore (pure refactor, byte-identical,
   asserted).
2. `grade_story(...) -> StoryGrade` (local rubric; reuse critic if viable) + tests (deterministic for a
   fixed seed/model; non-degenerate score on a real composed story).
3. `refine_story(...)` loop + `resolve_refine_passes` flag/provider gate + keep-best + critique-informed
   diversity_hint + never-fail fallback + merged telemetry, wired into run() (loop ONLY when
   effective_passes>=2; else the byte-identical single path). Flag-off call-count + no-key + byte-identical
   tests.
4. (Optional, later) remote opt-in + cost guard (like v0 chunk 4) -- DEFERRED unless the panel wants it now.
5. VALIDATION soak (operator GPU): `OTR_STORY_REFINE_PASSES=3` vs baseline; measure grade lift + the
   v0 sameness metrics; operator written go/no-go.
