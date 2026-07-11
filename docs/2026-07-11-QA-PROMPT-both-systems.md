# OTR QA SWEEP -- paste this whole file into agy AND into codex

Two agents are getting this. Answer independently; do not assume the other is right.
You are a REVIEWER: read anything, but do NOT edit source, do NOT git add/commit/push.
Write your findings to `qa_<yourname>.md` in the repo root (e.g. `qa_agy.md`,
`qa_codex.md`) and stop.

REPO: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
BRANCH: v2.0-alpha (pull first -- HEAD moves every few minutes)
Label every claim CONFIRMED (you opened the file / ran the number) or [ASSUMPTION].
Five things you are sure of beat twenty guesses. Retract anything you got wrong before.

## What this system is

An old-time-radio episode generator, built as a ComfyUI custom-node pack. A local
Mistral-Nemo-12B writes an episode through a ladder of strict typed passes (Pydantic v2,
`extra="forbid"`, `strict=True`). Three sci-fi source banks are in play -- `scifi_codex`,
`scifi_gemini`, `scifi_sonnet` -- each with its own pass ladder, plus a shared writer
tail, freeze cascade, cast lock, and media/render path.

State of the build:
- **Codex**: publishes a 30-word episode end to end. Verified asset in `otr\obs\`.
- **Gemini**: now clears P0-P6 (several passes rescued by deterministic repairs with NO
  LLM call). Currently dying in the scene critique/rewrite loop.
- **Sonnet**: has never completed a run.
Next milestone: all three publish at 30 words, then a 720-word bake-off across the
media packs.

## THE LAW (a "fix" that breaks this is an automatic reject)

**Python judges. The LLM writes.**
- Python may NEVER author, rewrite, trim, pad, or template story text.
- Deterministic repair is allowed ONLY for MECHANICAL metadata already implied by an
  accepted upstream artifact: ids, ordering, enums, a fixed role label, forbidden extra
  keys, a parent reference. If it is ambiguous -> FAIL CLOSED. Never guess.
- Anything AUTHORED -- dialogue, premise, a shot's visual_prompt, a beat's intent -- is
  the model's job. Python may reject it, never invent it.
- The requested word count is a creative SCALE REQUEST and a post-hoc statistic. It
  never causes a trim, a pad, a cull, or a rewrite.

## The defect classes we have already been killed by (pattern-match these FORWARD)

Every one of these was a live kill that cost a 15-minute render. This is the shape of
what you are hunting.

1. **A contract JSON cannot satisfy.** `pitches: tuple[PitchV4, PitchV4, PitchV4]` in a
   strict model fed from `json.loads`. JSON has no tuple; strict mode will not coerce a
   list. The field was unsatisfiable by construction -- the pass could NEVER pass.
2. **Required fields the model reliably omits**, where the value was mechanically
   derivable all along (a shot nested in scene s001 IS in s001; a beat's `order` is its
   position).
3. **Forbidden extra keys.** The model garnishes artifacts; `extra="forbid"` throws away
   an entire scene -- dialogue and all -- over one unrequested key.
4. **Legacy values copied from the model's own failed artifact** through typed repair
   (a `.v1` schema literal, a `beat_end` enum that no longer exists).
5. **SILENT PROMPT TRUNCATION.** `context_cap` is 8192; the generate_fn LEFT-truncates.
   A flat `max_new_tokens=3600` left only 4592 for input, and the repair prompt was 5408
   -- so the system/schema prefix was sliced off EVERY repair call. The model was not
   ignoring instructions; it never received them. This one cost four consecutive rolls
   while we all audited the wrong thing. `PROMPT_GUARD: Truncated 5408 -> 4592`.
6. **Producer-boundary gaps in the shared writer tail.** Content-owned lanes bypass
   legacy producers: no `text_for_tts` stamp (voice gate), no seed receipt (credits) --
   and then a "fix" that stamped `cast_contract.cast_seed` made CastLock try to REPLAY a
   cast the lane had rolled itself (`num_characters must be 1-6, got 0`).
7. **UNSATISFIABLE SEAM CONTRACTS -- read this twice.** The last two kills were in
   PROMPT TEXT, not Python:
   - the critique seam ordered the critic to "Ensure the total word count of the lines
     EQUALS the scene's target word limit." At 30 words over 6 beats (~5 words a beat)
     exact equality is unreachable, so the critic dutifully failed the scene, the bounded
     rewrite missed too, and the run died. **The model was obeying us.**
   - the same seam demanded "every audible scientific fact is correctly and traceably
     integrated" -- judged PER SCENE. Scene 2 was failed for not containing Fact F01,
     which belongs to Scene 1. An episode-level property enforced at scene level, which
     no scene can satisfy.

## YOUR JOB

### A. QA THE SEAMS AS CONTRACTS (highest value -- this is where the bugs now live)
Read every prompt seam in `nodes/story_packs/scifi_*/**.json` and every schema
instruction the lanes build. For each, ask:
- Is any rule UNSATISFIABLE, or satisfiable only by luck? (exact word counts, exact
  quotas, "must equal", "meet it exactly")
- Does any rule enforce an EPISODE-level property at a SCENE or LINE level (or a
  scene-level property at a line level)? That is the F01 bug, generalized.
- Does any rule ask the model for something Python already knows (a parent id, an
  index, an enum)? That is wasted budget and a guaranteed failure mode.
- Does any rule CONTRADICT another seam, or contradict the strict schema it feeds?
- Does any rule demand exactness where the system is explicitly advisory?
List every offender: `<pack>::<seam>` -- the exact sentence -- why it is unsatisfiable
or misscoped -- the corrected sentence you would ship.

### B. THE TRUNCATION ARITHMETIC (class 5, systemic)
`context_cap` = 8192. `max_input_tokens = 8192 - max_new_tokens`. For EVERY structured
call in all three lanes (including the typed-REPAIR prompt, which is always the fat one
-- it carries the failed artifact AND the validation error AND the original request):
does it fit at 30 words? At 720 words? Which passes still use a flat literal reservation
instead of one scaled to the artifact's real cost? Which should set
`prompt_must_fit=True` so a miss FAILS LOUD instead of lying to us?
Ground it: grep `C:\Users\jeffr\Documents\ComfyUI\comfyui_8000.log` for every
`PROMPT_GUARD` line. Evidence beats estimation.

### C. THE NEXT KILL
Gemini is in the critique/rewrite loop; Sonnet has never run. Predict what kills each
one next, ranked. For each: `<file:line>` -- what breaks -- which class above --
MECHANICAL or CREATIVE -- fix sketch.
Then the shared tail (`OTR_LedgerScriptWriter._run_writer_tail`, freeze cascade, cast
lock, credits, media): what ELSE does a content-owned lane silently bypass? That
boundary has now killed us three separate times.

### D. WHAT IS DANGEROUS ABOUT THE FIXES ALREADY IN
Try to break them: the deterministic repairs (`repair_outline_metadata`,
`repair_forbidden_extra_keys`, `repair_script_artifact_metadata`), the CastLock
content-owned VERIFY-not-REPLAY branch, and the `episode_seed` receipt. Where could a
deterministic repair silently destroy authored work, or accept something it should have
failed closed on?

## Output (`qa_<yourname>.md`)

A. SEAM CONTRACT DEFECTS: table of offenders + corrected sentences.
B. TRUNCATION TABLE: pass, reservation, input budget, base/repair prompt size, 30w and
   720w verdict, prompt_must_fit yes/no, ranked fix list.
C. NEXT KILLS: ranked, Gemini then Sonnet then shared tail.
D. ATTACKS ON THE EXISTING FIXES.
CONFIDENCE on every line.
