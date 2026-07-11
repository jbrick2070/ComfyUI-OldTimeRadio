# AGY MASTER PROMPT -- OTR sci-fi bake-off (paste this whole file into agy)

You are reviewing a REAL repo. Read the actual files before you claim anything.

REPO: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
BRANCH: v2.0-alpha
BASELINE COMMIT: e679b754 (suite 7581 passed / 31 skipped / 1 xfailed; Bug Bible 17 passed)

You are a REVIEWER ONLY. Do not edit, create, or delete any file in the repo.
Write your review to a file named agy_review.md in the repo root. Nothing else.

## What this system is

An OTR (old-time-radio) episode generator built as a ComfyUI custom-node pack. A
local Mistral-Nemo-12B writes an episode through a ladder of strict typed passes
(Pydantic v2, "strict" = extra keys forbidden). The sci-fi banks -- scifi_codex,
scifi_gemini, scifi_sonnet, scifi_fable2 -- are "content-owned" lanes: the lane
runner builds its own cast, score, and script, and seals the canonical line text
against a proof map, instead of running the legacy writer's content passes.

The Codex pass ladder (nodes/_otr_scifi_codex.py, run_scifi_codex_episode):
  P0 fact index (source-grounded evidence)  -> FactIndexV4
  P1 dramatic question                      -> DramaticQuestionV4
  P2 cast plan                              -> CastPlanV4
  P3 radio score (scenes/shots/beats/lines) -> RadioScoreV4      <- the accepted GRAPH
  P4 structure review                       -> StructureReviewV4
  P5 whole script (writes dialogue)         -> ScriptArtifactV4
  P6 listener review                        -> ListenerReviewV4
  P7 whole-script retake (polish)           -> ScriptArtifactV4
  P8 final audit                            -> FinalAuditV4
  P9 whole-script retake (if audit=rewrite) -> ScriptArtifactV4

Every pass runs a 3-attempt ladder: base call -> typed repair (or structural
retry) -> final attempt. A deterministic "metadata-only repair" can short-circuit
the LLM repair call when the defect is mechanical (ids, enums, schema literal) --
it must NEVER touch dialogue, premise, beats, or story content, and must fail
closed (return None) when the mapping is missing or ambiguous.

## Hard project laws (do not propose anything that violates these)

- Fix bugs at the ROOT CAUSE. No shims, no band-aids, no post-hoc moves.
- NEVER let Python rewrite story text. Python judges; the LLM writes.
- A deterministic repair may only normalize MECHANICAL metadata derived from an
  already-accepted upstream artifact. Ambiguous -> fail closed, never guess.
- The real workflow is workflows/otr_canonical.json. Any node/wiring/widget
  change must land IN that file in the SAME change as the code.
- Rendered assets go straight to otr\episodes\<ep>\, final to otr\obs\.
- Never the word "dummy" -- use "placeholder" or "stub". SFW. UTF-8, no BOM.
- Platform: Windows, RTX 5080 laptop, 16 GB VRAM, local/offline by default.

## Where we are right now (all live-verified today, 2026-07-11)

FIXED + PROVEN LIVE (commit e679b754):
- PBUG-20260711-13: P5 typed repair kept legacy metadata (schema_version .v1,
  boundary "beat_end"). Root fix = repair_script_artifact_metadata(): derives the
  v4 schema literal, drops forbidden extras, maps each line's shot_id, and derives
  boundary from the ACCEPTED SCORE GRAPH's line/shot/beat order. Live roll 12
  reproduced boundary="beat_end" and the deterministic repair fixed it with NO LLM
  repair call.
- PBUG-20260711-14: content-owned lanes never stamped text_for_tts (voice gate).
- PBUG-20260711-15: content-owned lanes reached CreditsRoll with no seed receipt.

JUST FIXED, IN THE WORKING TREE, NOT YET COMMITTED -- THIS IS WHAT YOU REVIEW:

### Defect A (was a hard blocker: the run died after 14 minutes of generation)
Live: `ValueError: num_characters must be 1-6, got 0`
  cast_lock.py:189 lock -> _assign_bark_voices
  cast_lock.py:353 -> _otr_casting.replay_voice_assignment
  _otr_casting.py:1211 -> assemble_pre_locked_rows  <- raises

Root cause: cast_lock._assign_bark_voices treats meta.cast_contract.cast_seed as
"the writer's seeded cast picker produced this cast -- REPLAY it". Content-owned
lanes build their own cast rows and stamp their own voice presets in the lane
runner (_otr_scifi_codex._assemble_ledger: announcer -> kokoro/bm_george;
c01/c02/c03 -> bark v2/en_speaker_6 / _3 / _0). The picker never ran, so the
contract has no num_characters_request -> int(None or 0) -> 0 -> ValueError.
The PBUG-15 credits fix had stamped cast_contract.cast_seed as a "seed receipt",
which CLOSED the `cast_seed is None` escape hatch these lanes relied on.

Fix as implemented:
1. nodes/OTR_LedgerScriptWriter.py (shared writer tail, content-owned branch):
   stamp meta["episode_seed"] ONLY -- otr_credits_roll.py:279-284 already accepts
   `(meta.cast_contract or {}).get("cast_seed", meta.get("episode_seed"))`, so
   episode_seed satisfies the credits receipt WITHOUT falsely claiming a
   replayable writer cast. Stop writing cast_contract.cast_seed / cast_seed_source
   / cast_contract_version for a lane that owns its cast.
2. nodes/cast_lock.py::_assign_bark_voices: if
   _otr_text_delivery.delivery_mode_for_meta(meta) == CONTENT_OWNED, preserve the
   lane's voice_preset values and SKIP the replay -- but still run the Gate 1
   invariants (_assert_unique_bark_voices, _assert_voice_preset_invariant) so a
   content-owned lane can never ship duplicate or non-"v2/" bark voices.
   (Same family discriminator the freeze cascade and voice lane already use.)
Tests added in tests/test_cast_lock.py + updated tests/test_fable2_tail_context.py.
Focused run: 142 passed (includes tests/test_cast_voice_replay_parity.py, which
pins that the LEGACY replay path is byte-identical and unchanged).

### Defect B (latent at 30w, expected to be fatal at 720w)
Live P7: `OUTPUT_CAP: prompt_tokens=4543 generated_tokens=2800 max_new_tokens=2800`
then `no decodable top-level JSON object found`, and the raw head shows the model
emitting `{ "artifact_inputs": { "accepted_line_count": 13, ... ` -- it echoed the
REQUEST ENVELOPE instead of the artifact root, ran out of output budget, and
produced truncated JSON. The structural retry happened to recover; that is luck.

Two compounding faults, fixed as:
1. _SCRIPT_ARTIFACT_ROOT_INSTRUCTION now forbids echoing the envelope keys
   (pass_id, artifact_inputs, result_json_schema) and requires the response to
   begin at `{"schema_version": "scifi_codex.script_artifact.v4", ...}`.
2. _script_output_token_budget(requested_words) scaled ONLY on the word steer:
   `min(5400, max(2800, words*4.5 + 1200))`. But a ScriptArtifactV4's size is
   driven by the ACCEPTED LINE COUNT (strict per-line metadata) as much as by
   dialogue words. It is now
   `_script_output_token_budget(requested_words, accepted_line_count)` =
   `min(5400, max(2800, words*4.5 + 130*lines + 600))`, computed AFTER the score
   is final (P3 / P3_rewrite) and passed to P5/P7/P9 as `script_token_budget`.

## THE QUESTION I MOST NEED YOU ON (the 720-word bake-off)

The local generate_fn (OTR_LedgerScriptWriter._build_truncating_generate_fn) does:
    context_cap = int(cache_entry.get("context_cap") or 8192)
    max_input_tokens = max(64, context_cap - int(max_new_tokens))
    if input_len > max_input_tokens: <LEFT-TRUNCATE the prompt>   # silent
The local transformers path sets no context_cap, so it is 8192 -- an arbitrary
default, not a model limit (Mistral-Nemo supports 128k).

At 30 words / 13 lines the P7 prompt was already 4543 tokens and the output
reservation is 2800 (4543 + 2800 = 7343, just under 8192). At 720 words the P7
prompt carries the full previous script (720 words of dialogue + per-line
metadata) + the line graph + the review, and the output must re-emit that whole
script again. Both grow together. My arithmetic says 720w cannot fit in 8192, and
because P5/P7/P9 do NOT set prompt_must_fit=True, the failure mode is a SILENT
LEFT-TRUNCATION that eats the system/schema prefix -- which is precisely the
already-logged PBUG-20260711-12 failure class.

Tell me, grounded in the actual code:
1. Confirm or refute the 8192 ceiling arithmetic for a 720-word Codex run.
   Estimate the real P7 prompt + output token cost at 720 words.
2. Which root fix is right, and why:
   (a) Raise context_cap for the local transformers path (derive from the model
       config / tokenizer, e.g. max_position_embeddings, with a VRAM-aware
       ceiling). What is the KV-cache cost at 16k / 32k for Mistral-Nemo-12B on a
       16 GB RTX 5080, and does it fit next to the rest of the OTR pipeline?
   (b) Make P7/P9 a LINE-LEVEL PATCH pass (return only revised lines, not the
       whole artifact) so output stays flat as word count grows. What breaks?
       (The deterministic repair, _validate_script_graph, and _assemble_ledger all
       assume a whole ScriptArtifactV4.)
   (c) Something better I have not considered.
3. Should P5/P7/P9 set prompt_must_fit=True so an over-budget prompt FAILS LOUD
   instead of silently left-truncating? What is the blast radius?
4. Anything in Defect A or Defect B above that is wrong, incomplete, or that
   breaks an existing test or invariant.

## Output format (write to agy_review.md)

VERDICT: <ship-as-is | yes-with-fixes | no>
MUST-FIX BEFORE BUILD: numbered; each with file:line and a concrete patch sketch.
SHOULD-FIX: numbered.
720W CAPACITY VERDICT: your arithmetic, then your recommended option (a/b/c) with
  the reasoning and the specific files/functions to change.
CUT THESE (over-engineering): numbered.

Label every claim CONFIRMED (you read the file), or [ASSUMPTION] if you did not.
Do not invent line numbers. If you cannot open a file, say so.
