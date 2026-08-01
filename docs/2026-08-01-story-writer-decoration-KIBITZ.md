# Story-writer transport decoration -- kibitz r1 record

Durable record. Panel: codex `gpt-5.6-sol` (high), antigravity. Driver/judge: Claude.
Working artifacts (gitignored): `kibitz-runs/2026-08-01-story-writer-ledger/r1/`.
Input: `docs/2026-08-01-story-writer-ledger-robustness-r1.md`.

Panel: codex `gpt-5.6-sol` (high), antigravity. Driver and sole judge: Claude.
Every verdict below was checked against the real files on this box.

| # | Claim | Seat | Verdict |
|---|---|---|---|
| 1 | Decorated labels fall through to `_RE_SPEAKER` and become UNKNOWN_SPEAKER | both | **CONFIRMED** `_otr_fable2_markup.py:37-42,45-47`; reproduced in a unit fixture |
| 2 | There are TWO normalizers and they must not diverge -- `_normalize_line` decides classification, `normalize_fable2_markup_text` builds normalized_source + sha256 + proof | codex 3 | **CONFIRMED and ADOPTED.** The single strongest structural point of the round. Fixing only the classifier would have made the accepted script differ from the artifact hashed beside it |
| 3 | "Strip the structural prefix" is too narrow -- live output includes `**END.**`, decorated cast labels, single-star forms | codex 4 | **CONFIRMED** -- `**END.**` appears verbatim in `tmp/_fastwan_e2e.log`. My original scope would have missed it |
| 4 | Do NOT narrow `_RE_SPEAKER`; it is the closed-roster enforcement path | codex 5 | **CONFIRMED and ADOPTED.** Canonicalize BEFORE the catch-all instead; a genuinely off-roster name still raises UNKNOWN_SPEAKER (pinned by test) |
| 5 | The ladder does NOT retry with the same instruction -- it appends the defect list and a stage-direction repair note | codex 7 | **CONFIRMED; MY DOC WAS WRONG** `_otr_scifi_fable2.py:1721-1738` |
| 6 | The "fails after an hour of stills+audio" premise is false -- the writer failed at node 1 with nothing downstream executed | codex 6 | **CONFIRMED; MY DOC WAS WRONG.** Writer legs died in 1.5-11 min. The 82-minute leg was `wan_ti2v` failing at VIDEO. I conflated two different failures |
| 7 | The repair ladder is STATELESS -- no assistant turn carries the rejected draft, so "keep the same wording" is addressed to a model that cannot see it | agy S1 | **CONFIRMED** `_otr_scifi_fable2.py:1674-1692`. The best find of the round, and neither codex nor I had it |
| 8 | Fix: inject the previous draft as an assistant turn | agy S1 | **REFUTED AS PROPOSED -- implemented, then REVERTED.** `ProviderCapacityMessages` sets `_otr_prompt_must_fit` + `_otr_reserve_remaining_output_capacity`, and `prompt_no_room` refuses BEFORE the call when the prompt leaves less room than the artifact needs (`_otr_generation_budget.py:13-35`). At the `n_ctx=2048` this campaign runs, injecting a whole episode draft risks converting a RECOVERABLE formatting defect into a DETERMINISTIC capacity refusal. The problem is real; this fix is not free and neither panel priced it |
| 9 | VRAM estimator ignores the caller's ctx and has no quant input | both | **CONFIRMED** `_otr_model_catalog.py:1476-1525`. Measured: identical 14.60 GB at ctx 2048/4096/8192; `approx_safetensors_gb` 11.8 == the Q8_0 file exactly, while the tier loads Q4_K_M at 6.63 GB |
| 10 | Cut the estimator defect into its own plan -- no coupling to markup | codex CUT 1 | **ACCEPTED.** Recorded separately; not fixed here |
| 11 | Cut a general Markdown sanitizer -- only an enumerated grammar is safe | codex CUT 4 | **ACCEPTED.** Implemented as a closed grammar over `**`, `__`, `*`, `_` with balance required |
| 12 | Cut the runtime preflight generation | codex CUT 2 | **ACCEPTED** -- follows from row 6; a stochastic probe cannot predict a later stochastic response |
| 13 | Ladder should retire a candidate and request a fresh one rather than raising after 4 rungs | codex 2 | **DEFERRED, not rejected.** It is a real gap against `PRODUCTION_SPRINT_LESSONS` §§35-36, but it is a liveness-policy change to the writer's control flow and does not belong in the same commit as a parser fix |
| 14 | Markdown is not the only defect in the failed population (22 BAD_LINE_SHAPE, stage directions) | codex 8 | **CONFIRMED.** The fix is therefore NOT claimed to recover all three failed legs -- only the decoration class |
| 15 | Top-level torch/numpy imports in audio_enhance / scene_sequencer violate import isolation | agy S2 | **OUT OF SCOPE** -- unrelated lane, unverified here, not touched |

## What shipped

ONE change: a closed-grammar transport canonicalizer shared by both normalizers.

* Removes a BALANCED `**` / `__` / `*` / `_` wrapper from a structural label in
  three recognized shapes (`<M>LABEL:<M> payload`, `<M>LABEL<M>: payload`,
  `<M>TOKEN<M>`), and reports every removal.
* Leaves unbalanced, mixed, payload-internal and out-of-grammar decoration
  untouched, so they remain loud defects.
* Roster-independent, so closed-roster enforcement is unweakened.
* Used by `_normalize_line` AND `normalize_fable2_markup_text`, so the parsed
  script and the hashed proof artifact cannot diverge.

23 new tests. Full suite green.

## What did NOT ship, and why

* **The stateless-ladder fix (row 8).** Real problem, unpriced cost. Needs a
  capacity budget check before the draft can be injected -- otherwise it trades
  a recoverable defect for a deterministic one at low `n_ctx`.
* **Candidate liveness (row 13).** Real gap, separate change.
* **The VRAM estimator (rows 9, 10).** Real, decoupled, recorded separately.

## Field note

The live evidence changed underneath this round: `gemma-4-12b` (the canonical
model) produces **zero** markup defects, so the decoration was Mistral-shaped.
That does not make the fix unnecessary -- the parser should not be one model's
formatting habit away from losing an episode -- but it does mean the fix is
DEFENSIVE, not the thing standing between this pipeline and a finished run.
