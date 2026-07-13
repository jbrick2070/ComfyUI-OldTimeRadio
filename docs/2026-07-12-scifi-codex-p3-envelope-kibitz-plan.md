# Sci-Fi Codex P3 bounded-output repair plan

## A. Live failure to repair

The canonical 120-word `scifi_codex` reverify failed on 2026-07-12, prompt
`f26b727b-42c8-40d6-b3ee-001d7a869cf9`, before any image, media, or OBS work.
The selected input was the CMU RSS story “Train Robots With Internet Videos.”

P0 did not repeat PBUG-21: its first source span was non-literal and the
existing deterministic metadata repair corrected only the offset/quote metadata
before P1. P3 then failed on all three structured-call rungs:

1. base: `prompt_tokens=2615`, `generated_tokens=2800`,
   `max_new_tokens=2800`, incomplete `RadioScoreV4` JSON;
2. structural retry: the same 2,800-token cap and incomplete JSON;
3. typed repair: `prompt_tokens=4317`, the same 2,800-token cap and incomplete
   JSON.

The runner reported `RESULT FAIL`, `P3 failed after 3 attempt(s)`, and
`Prompt executed in 00:15:50`. No ledger, episode asset, OBS final, image, or
still output was accepted. This is a live P3 continuation of the BUG-11.50
capacity class, not a speculative image/still defect.

## B. Confirmed current code facts

1. `nodes/_otr_scifi_codex.py:_radio_score_output_token_budget()` returns 2,800
   tokens for the 120-word, 12-beat score. It deliberately preserves an
   8,192-token input window for PBUG-20’s repair prompt.
2. The P3 and `P3_rewrite` calls use that budget and `prompt_must_fit=True`.
   The live P3 repair prompt fit (4,317 input tokens), so it was not left
   truncated and PBUG-20’s lost-contract root cause did not recur.
3. `RadioScoreV4` and its nested `ScenePlanV4`, `ShotPlanV4`, `BeatPlanV4`, and
   `MusicCueV4` expose unbounded material strings and several unbounded lists;
   `AdvisoryWordPlanV4.per_beat` is also `list[dict[str, Any]]`. Therefore the
   current schema has no finite serialized output ceiling; a 2,800-token
   reservation is not defensible merely because it fits the repair input
   context.
4. The shared structured-call path may auto-clamp `Field(max_length=...)`
   strings. A P3 surface fix must not silently cut authored prose to make an
   invalid artifact fit: it needs an explicit fail-to-typed-repair / fail-closed
   behavior for an over-limit authored value.
5. Current P3 repair is already compact-tagged rather than a copyable JSON
   request envelope. It correctly locks the advisory graph and rejects wrapper
   roots, but asks the model for a complete replacement score.
6. The canonical workflow has no P3 widget/wiring change in scope. The
   all-visualizer effective-consumer gate is separate and must remain intact:
   it suppresses image authoring only when all actual consumers are visualizers;
   a known still-image consumer must still receive the required image. Do not
   “fix” the pending still-image surprise by changing the gate preemptively.

## C. Objective

Make a full `RadioScoreV4` response and its repair path fit the actual local
8,192-token context without silently truncating prompt or output. Preserve
creative ownership of title, premise, setting, scene/shot prose, beat intent,
and music descriptions/prompts. Keep unknown or semantically ambiguous scores
fail-closed. Do not solve this by raising `max_new_tokens` alone.

## D. Constraints and non-goals

1. Apply the five-representations rule: P3 base seam, schema, fixture/tests,
   parser/validator, and repair seam must agree.
2. Bound the model-facing P3 surface before calculating its reservation:
   counts, nested counts, all material serialized string lengths, and the
   remaining graph width. The authoritative advisory graph is already fixed
   before P3 and is a safe input to the budget.
3. Keep the exact requested root and forbidden-envelope rules. Do not recreate
   PBUG-20 by duplicating a request envelope or allowing prompt truncation.
4. Retain only existing deterministic repairs that change objectively derived
   metadata, and run them at every accepted-object boundary. Do not invent or
   rewrite story content in Python.
5. Consider a bounded typed patch only if the failure is a localized,
   independently owned semantic omission in an otherwise accepted score. A
   general incomplete full score remains a full-artifact capacity problem.
6. Preserve public API behavior and the canonical workflow. If no JSON wiring
   changes are needed, prove the canonical workflow has no delta and still
   passes validator/round-trip/link/widget audit.
7. Do not conflate this work with the still-image behavior. The next qualified
   run must retain the all-visualizer no-image proof and leave a genuine
   still-consumer path available for its later live test.

## E. Hardened lane evidence to evaluate and transfer selectively

1. Shared Sci-Fi P0 (`nodes/_otr_scifi_p0_contract.py`, used by Codex/Gemini/
   Sonnet) is the direct capacity model: finite rows, finite spans, finite
   strings, a compact model-visible contract, output/surface receipts, and a
   compact tagged repair context. It must be adapted to P3’s graph, not copied
   mechanically.
2. Sci-Fi Gemini P3 (`nodes/_otr_scifi_gemini.py:outline_output_token_budget`)
   already scales a bounded outline reservation by word steer and beat count,
   with `prompt_must_fit=True`. Audit whether its `OutlineV4` surface is truly
   finite and whether any of its count/string bounds apply to the Codex score.
3. The original hardened radio lane (`nodes/_otr_original_codex56sol.py`)
   demonstrates narrow accepted-object projections and a bounded
   `ScoreIntentPatch` for a localized semantic repair. Transfer only its
   ownership/merged-validation principles; do not copy its different score
   schema or whole-artifact transport.
4. Fable2’s hardened markup ladder is a useful separate contrast: when a
   response is truncated, it regenerates from bounded base context rather than
   asking a model to copy an incomplete raw artifact. Audit whether that
   recovery principle applies after P3 obtains a finite schema; do not import
   Fable2’s different lane/transport unchanged.

## F. Candidate implementation sequence (not yet approved as final design)

1. Inventory every serialized P3 field and classify it as locked graph
   metadata, bounded authored prose, or non-authoritative derived metadata.
2. Add explicit finite schema limits and matching base/repair prompt rules for
   all material P3 arrays and strings. The bounds must still permit a useful
   120-word radio score, and must reject/retry over-limit authored values rather
   than silently clamping their prose.
3. Replace the current budget calculation with one derived from the bounded
   P3 output shape and the locked beat count; journal a durable surface/budget
   receipt. Retain pre-generation prompt-fit failure.
4. Add focused tests that reject every over-limit list/string shape, prove
   P3/P3_rewrite reservations rise with actual graph width, prove a
   representative complete bounded score fits below the reservation, prove the
   compact repair prompt fits, and prove typed-repair responses cross the same
   accepted-object boundary.
5. Run focused tests, full Windows regression suite, Bug Bible regression,
   pycompile/UTF-8 checks, and canonical workflow validator/round-trip/link/
   widget audit. Commit and push the green chunk.
6. Selectively reset ComfyUI and rerun the same canonical 120-word Codex bank.
   Require P3 clearance plus full ledger, episode asset, `obs_publish OK`, OBS
   final file, and all-visualizer zero-image objects. Only then resume Fable2,
   Gemini, and Sonnet sequentially. Preserve the still-consumer scenario for
   its separate live surprise test.

## G. Decisions Kibitz must pressure-test

1. Is a bounded full score sufficient, or must P3 adopt a smaller initial
   score-plus-later-authoring split? Reject plans that move P3-owned creative
   decisions into deterministic Python.
2. Which explicit per-field and per-collection bounds are necessary and
   sufficient for a 120-word, 12-beat score without starving creative quality?
3. Can the repair context be made meaningfully smaller without dropping the
   locked graph or incorrectly asking the model to reconstruct mechanical
   metadata?
4. What cross-lane parity work is justified now? P3 schemas differ across
   Codex, Gemini, Sonnet, Fable2, and the original lane; do not blindly fan out
   a Codex-only contract.
5. What exact live proof distinguishes success from merely reaching a resident
   ComfyUI server?

## H. Final convergence and selected implementation

Kibitz completed all four rounds. The selected design is a directly bounded
canonical `RadioScoreV4`, not a new draft/compiler representation: static local
limits make the existing artifact finite while preserving P4 onward, ledger
assembly, and the canonical graph validator.

- maximum surface: 3 scenes, 2 shots and 4 beats per scene, 12 beats, 2 line
  IDs per beat, 24 line IDs, 3 music cues, 2 fact IDs per beat;
- maximum prose: title 64, premise 144, setting 80, scene env 56, scene/shot
  descriptions 72, visual/cue-generation prompts 120, beat speaker 40, intent
  64, arc 28, and cue description 80 characters;
- exact P3/P3-rewrite reservation: 2,900 output tokens and 5,292 input tokens
  in the local 8,192-token context;
- exact live tokenizer regression of max-width P3 and P3-rewrite base/repair
  envelopes: 2,621 score tokens, a 3,465-token P3 base prompt, and every
  measured envelope under the 5,292-token input reservation;
- P1/P2/P3/P3-rewrite/P4 authored fields reject an over-limit string into their
  typed repair instead of silently clamping it; legacy callers retain the
  shared default clamp behavior;
- rewrite base carries the accepted score plus review only, avoiding duplicate
  advisory/graph input; rewrite repair keeps its failed current score,
  rejection, derived locked graph/advisory, and review but no longer duplicates
  the accepted pre-rewrite score.

The RSS audit confirmed that the all-visualizer no-image consumer gate is
global, has no source-bank branch, and already covers every RSS bank. Its lane
schemas differ, so Codex P3 bounds are not copied into Gemini, Sonnet, Fable2,
Media Archive, or original lanes without a new live artifact.
