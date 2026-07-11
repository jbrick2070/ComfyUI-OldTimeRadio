# AGY -- look ahead on the GEMINI + SONNET lanes (paste this whole file into agy)

REVIEWER ONLY. Do not edit source, do not git add/commit/push. Write your review to
`agy_review4.md` in the repo root and stop. Read the real files. Label every claim
CONFIRMED (you opened it) or [ASSUMPTION].

REPO: C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
BRANCH: v2.0-alpha  HEAD: a5e44d2c (plus an uncommitted Gemini P3 repair, below)

## The job

Codex now publishes a 30-word episode end to end. Gemini and Sonnet do not. Every
single roll so far has died on exactly ONE new defect, we fix it, and the next roll
dies on the NEXT one. I am tired of discovering these one 15-minute render at a
time. **Read ahead and tell me what kills the Gemini and Sonnet lanes next.**

## The defect classes we have already hit (pattern-match these forward)

Each of these was a live kill. They are the shape of what you are hunting.

1. **Contract JSON cannot satisfy.** `PitchSlateV4.pitches` was
   `tuple[PitchV4, PitchV4, PitchV4]` in a `strict=True` model fed from
   `json.loads`. JSON has no tuple; strict mode will not coerce a list. The field
   was unsatisfiable by construction -- the pass could never pass. (Gemini P1.)
   Guard now exists: `tests/test_scifi_lane_schema_parity.py`.
2. **Nested graph fields the model omits.** Gemini P3 `OutlineV4`: every nested
   shot/beat missing its parent `scene_id`, every beat missing `order`, shots
   missing `visual_prompt` -- 18, then 22 validation errors. Same class as Codex
   PBUG-06/08/09.
   Fix now in the tree (uncommitted): `normalize_outline_graph_metadata` derives the
   MECHANICAL fields (a shot nested in s001 IS in s001; order is position) and hands
   the model back a graph-normalized artifact whose only remaining job is the
   CREATIVE field (`visual_prompt`). Python never invents authored content.
3. **Legacy/enum metadata copied from the model's own failed artifact.**
   (Codex P5: `schema_version` .v1, `boundary` "beat_end".) Fixed by a deterministic
   metadata-only repair driven off the accepted upstream graph.
4. **Silent prompt truncation.** `context_cap` 8192; the generate_fn LEFT-TRUNCATES,
   eating the system/schema prefix. Codex P0 sets `prompt_must_fit=True`; I just
   added it to Gemini P0 and Sonnet P0.
5. **Output cap truncation / envelope echo.** Codex P7 emitted
   `{"artifact_inputs": ...}` (the INPUT envelope) and hit
   `generated_tokens == max_new_tokens` -> truncated JSON.
6. **Producer-boundary gaps in the SHARED writer tail.** Content-owned lanes bypass
   legacy producers: they never stamped `text_for_tts` (voice gate), never stamped a
   seed receipt (credits), and then a "fix" that stamped `cast_contract.cast_seed`
   made CastLock try to REPLAY a cast the lane rolled itself
   (`num_characters must be 1-6, got 0`). Gemini and Sonnet run the SAME shared
   tail, so they inherit whatever is still wrong there.

## What I want (agy_review4.md)

### A. GEMINI -- what kills the rest of the 30w run?
Trace `run_scifi_gemini_episode` (nodes/_otr_scifi_gemini.py) from P3 to publish:
P4 per-scene drafts, P5 critique, P6 rewrite, then `_assemble`, the shared writer
tail, CastLock, freeze, media, credits, obs_publish.
For EACH remaining pass and stage, tell me what breaks and why:
- Which strict models have required fields the local model reliably omits (the
  class-2 defect)? Which of those fields are MECHANICAL (derivable from context and
  therefore repairable in Python) vs CREATIVE (the model must author them)? That
  split is the whole design -- get it right.
- `_assemble` builds the ledger and `_GeminiTailFinalizer` pins expected text. What
  does it assume that P4/P6 may not deliver?
- Does Gemini's cast/voice path hit the same CastLock issue Codex just did? It builds
  its own cast in `_assemble` (voice_map: announcer -> kokoro/bm_george, c01..c03 ->
  bark v2/en_speaker_6/_3/_0). Does it satisfy the Gate 1 invariants
  (`_assert_unique_bark_voices`, `_assert_voice_preset_invariant`)?
- Gemini has NO deterministic repair for anything except P0 spans and (now) the P3
  outline graph. Where else does it need one?

### B. SONNET -- the same sweep, and it is stranger
`run_scifi_sonnet_episode` (nodes/_otr_scifi_sonnet.py) drafts PER LINE (P2a/P2b per
line index, a literalist and a speculator), then audits (P3), then a warden loop
(P4/P5), then attestation (P6). It has never run to completion.
- Which of its strict models are unsatisfiable or reliably under-filled?
- The per-line ladder means N model calls scale with line count. At 30 words that is
  small; at 720 it is not. What is the call count and the wall-clock cost at 720w,
  and does anything cap or budget it?
- `_assemble` + `_SonnetTailFinalizer`: same producer-boundary questions as Gemini.

### C. THE SHARED TAIL -- what else is content-owned-blind?
`OTR_LedgerScriptWriter._run_writer_tail` is where three separate producer gaps have
now bitten us (delivery stamp, seed receipt, cast replay). Audit it as a whole:
what OTHER legacy producer does a content-owned lane silently bypass? Enumerate every
stage in that tail that reads something a legacy lane's writer passes stamped and a
content-owned lane never sets. This is the highest-value question in this document --
it is the one that has killed us three times.

### D. RANK IT
Ordered list, most likely to kill the next roll first:
  <file:line> -- <what breaks> -- <which of the 6 classes above it is> --
  <MECHANICAL or CREATIVE> -- <fix sketch>

Do not pad. I would rather have five things you are sure of than twenty guesses.
