# pass01 judgment -- ltx_av_music FamilyInputGap (Claude = grounded judge)

## Panel
Grok-4.3 (detailed), Gemini-3.1-pro (short, one critical catch), GPT-5.5 (empty -- no usable
content). ~$0.20.

## CONFIRMED (grounded against render_driver.py + schemas.py)
- **Root cause:** `ltx_av_music` (family `audio_conditioned_video`) has
  `FAMILY_REQUIRED_INPUTS = ('text_prompt','audio_ref')`; a music beat with no per-line/shot timing
  gets no master-audio slice -> `audio_ref` absent -> `_assert_family_inputs_satisfiable`
  (render_driver.py:1278) raises `FamilyInputGap`. Confirmed: only ltx_av_music crashed (4x).
- **The existing shot-timing fallback is DEAD (Gemini, verified):** `ShotRow` is `_Forbid`
  (extra=forbid) with NO `start_s`/`dur_s` fields -- it CANNOT carry them, so
  `shot.get('start_s')`/`shot.get('dur_s')` are ALWAYS None. The "fall back to the shot row's
  timing" branch never actually supplied a window. `ShotRow` DOES carry `target_frame_count` (the
  beat's audio-derived length).
- **Whole-master slice is CATASTROPHIC (Gemini):** feeding the full ~episode-length master WAV to
  condition a ~2s music clip would blow up the audio encoder. The slice MUST be BOUNDED.

## ACCEPTED -> the build
1. **Render-side guard in `build_request_from_shot`** (Grok; CUT the ShotLock-upstream + routing
   options -- the driver owns the master slice and the no-fallbacks rule).
2. **FAMILY-based gate, not an engine-id whitelist (Grok):** trigger when
   `'audio_ref' in FAMILY_REQUIRED_INPUTS[engine_family]` -- covers ltx_av_music and any future
   audio_conditioned engine, never a hard-coded list.
3. **BOUNDED synthesized window (Gemini):** when the LINE lacks `start_s`/`dur_s`, set
   `dur = target_frame_count / fps` (the beat's real length; default ~4s if absent) and
   `start = cumulative sum of preceding shots' target_frame_count/fps` (the beat's timeline
   position), clamped >= 0. Slice THAT bounded span from the master. Never the whole master.
4. `audio_ref` STAYS required for `audio_conditioned_video` (do NOT make it optional) -- the builder
   GUARANTEES it (no-fallbacks). 
5. **Actionable log (Grok):** log the beat_id + the synthesized window when the guard fires;
   the FamilyInputGap message already names the engine/family.
6. **Unit test (Grok+Gemini):** a music shot with ZERO line timing + a master path -> `audio_ref`
   is populated (a bounded slice), and `_assert_family_inputs_satisfiable` passes.

## CUT
- Option 2 (stamp timings upstream in ShotLock) -- cross-module dependency not needed to close the
  crash; the render-side guard owns the master slice.
- Option 3 (capability-aware routing) -- would change the router + violates "ltx_av_music must work
  on music beats".
- Making `audio_ref` optional for the family -- weakens the contract; the builder guarantees it.

## VERIFY-AT-BUILD
- `ledger['video']['shots']` is ordered and each row carries `target_frame_count` (verified on
  ShotRow) -> the cumulative-start sum is computable in `build_request_from_shot`.
- `_slice_master_audio` accepts (path, start_s, dur_s, master_hash) -- mirror the existing call.

## Convergence
The two substantive reviews (Grok + Gemini) agree on the render-side bounded-slice family-gated fix
and independently surfaced the two grounding facts (dead shot-timing fallback; whole-master is
catastrophic). No contradictions. CONVERGED -- build it.
