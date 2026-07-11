# AGY R5 -- audit my fixes, then find the next kill (paste this whole file into agy)

REVIEWER ONLY. Do not edit source, do not git add/commit/push. Write to
`agy_review5.md` in the repo root and stop. Read the real files.
Label every claim CONFIRMED (you opened it) or [ASSUMPTION].

REPO: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
BRANCH: v2.0-alpha  HEAD: 74be961f

## Scorecard on your last review (be honest with yourself, I was)

Your R4 announcer finding was the best call anyone has made today -- and it was
half wrong, which is exactly why I want you looking at my fixes.

- ACCEPTED: Gemini's `CastV4.name` is free-form LLM text, and CastLock's Gate 1
  invariants skip the announcer row by the EXACT string "ANNOUNCER". A model-chosen
  "Narrator" would make them judge the announcer's kokoro preset (`bm_george`) as an
  invalid Bark voice and raise `CastingFailedError` in the media tail. Fixed at
  `74be961f` (validate at P3 + normalize in the metadata repair).
- WRONG: you said it hits Sonnet too. It does not. Sonnet's `CastLockV4.name` is
  `Literal["ANNOUNCER","ORUM","THESSALY","VESH"]` AND the cast dict is hardcoded --
  the announcer literally cannot be misnamed there. You asserted a shared defect
  from a shared symptom without checking the second lane. Do not do that again.
- ACCEPTED: `SceneCritiqueV4.line_fact_ids` and `AuditVerdictV4.defects` /
  `flagged_line_refs` / `invented_fact_flags` were required with no default, so a
  CLEAN critique and a CLEAR audit could only validate by finding fault. Defaults
  added.

## JOB 1 -- audit what I just shipped (adversarially)

Read these and try to break them:
- `nodes/_otr_scifi_gemini.py`: `validate_outline_cast_labels`,
  `normalize_outline_graph_metadata`, `repair_outline_metadata`, and the OutlineV4
  branch inside `typed_repair_factory`.
- `nodes/cast_lock.py`: `_assign_bark_voices` (the content-owned VERIFY-not-REPLAY
  branch).
- `nodes/OTR_LedgerScriptWriter.py`: the content-owned block in `_run_writer_tail`
  (episode_seed + delivery stamp).

Specifically:
1. `normalize_outline_graph_metadata` mutates the parsed dict: it forces every
   nested shot/beat `scene_id` to the parent scene's id, renumbers `order`
   globally across scenes (1..N, NOT per-scene), and rewrites `speaker` to
   "ANNOUNCER" for announcer beats. Is the GLOBAL beat numbering right, or does
   anything downstream (`_assemble`, `_GeminiTailFinalizer`, the music-cue anchors,
   `advisory_word_bands`) expect per-scene ordering? Check, do not guess.
2. Does forcing `beat.speaker = "ANNOUNCER"` collide with anything that matches
   speaker names against cast rows or dialogue text?
3. Is there any path where the P3 typed repair returns my hand-built
   "fill in the missing visual_prompts" prompt but the model's reply then fails for a
   DIFFERENT reason, and we have now spent our last attempt? Count the attempts in
   `_otr_structured_call.structured_call` and tell me exactly how many chances P3 has.
4. Anything I got wrong.

## JOB 2 -- the next kill on the Gemini path (P4 -> publish)

Gemini is running now and will die somewhere. Predict it. Walk P4 (per-scene draft),
P5 (critique), P6 (rewrite), `_assemble`, `_GeminiTailFinalizer`, the shared writer
tail, CastLock, freeze, media, credits, `obs_publish`.

For each: what breaks, is it MECHANICAL (derivable -> Python may repair it) or
CREATIVE (authored -> only the model may write it), and the fix sketch. That split is
the law here: Python judges, the LLM writes. A "fix" that has Python invent story
content is an automatic reject, and I will bounce it.

## JOB 3 -- Sonnet, cold

Sonnet has never completed a run and will be next. Same sweep, same split. Pay
attention to its per-line ladder (P2a/P2b per line index -- a literalist and a
speculator per line), the warden loop (P4/P5), and attestation (P6). What is
unsatisfiable, what is under-filled, and what does the shared tail expect that
Sonnet never stamps?

## Output (agy_review5.md)

JOB 1 AUDIT: what is wrong with my fixes, with file:line. Say "these hold" if they do.
JOB 2 GEMINI: ranked kill list.
JOB 3 SONNET: ranked kill list.
Each item: <file:line> -- <what breaks> -- <MECHANICAL|CREATIVE> -- <fix sketch>.
Five things you are sure of beat twenty guesses.
