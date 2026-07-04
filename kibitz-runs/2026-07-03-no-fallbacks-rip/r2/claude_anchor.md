# Claude anchor review — no-fallbacks rip PLAN (r2: implementability)

Grounded against the real Windows files (the 3 inventory audits + direct reads of
`_otr_voice_node_common.py`, `cast_lock.py`, `_otr_engine_profiles.py`,
`stable_audio_theme.py`, `eng_stable_audio_3.py`). CONFIRMED / MISREAD / UNVERIFIABLE
labels are mine.

## VERDICT: SOUND, ship-with-conditions. The rip is implementable, but the
byte-identical HAPPY PATH must be provably untouched and the pinning tests must be
retired IN THE SAME COMMIT as each rip. Two sequencing hazards below are MUST-FIX.

## MUST-FIX

1. **[CONFIRMED] The happy-path dispatch must stay byte-identical — the rip removes
   only the FALLBACK branches.** In `_otr_voice_node_common.py` the cloning-engine
   render calls `adapter.generate_voice(prepared, voice_ref, delivery_vector,
   engine_seed)` (line ~558-561) after the ref is resolved. The rip removes the
   `_bark_fb` branch (~514-552) and the best-effort `_resolve_clone_ref_path`
   (76-135), NOT that call. Guard: a golden render of indextts2 WITH a valid ref
   must remain bit-identical. Any proposed change that touches the post-resolution
   forward is REJECTED.

2. **[CONFIRMED] `_resolve_character_voices_fail_soft` is called UNCONDITIONALLY at
   `cast_lock.py:187`.** Ripping it to fail-hard means CastLock itself now raises on
   an orphan/unvoiced character line. R2 hazard: the fail-loud raise must be a NAMED
   error (EngineUnusable / a new VoiceCastingError), surfaced at CAST time with the
   line_id/char_id — never a bare `raise` mid-loop. And the writer/admission gate
   must be the thing that guarantees a valid cast, or every episode with a writer
   casting gap now hard-fails. Confirm the writer's cast contract already forbids
   orphans before ripping the repair, else the rip moves the failure earlier without
   a producer-side fix.

3. **[CONFIRMED] Pinning tests retire IN THE SAME COMMIT.** These assert the
   fallback behavior and WILL fail on the rip: `test_voice_casting_gender_agnostic.py`
   (gender-agnostic any-ref fallback), `test_batch_character_voices.py` (golden +
   dropdown), `test_tts_engine_sidecars.py` (missing_ref_fallback metadata),
   `test_bark_freeze_halt_bypass.py` (likely). The KNOWN-FAIL-GUARD will flag any as
   a regression. Each rip commit MUST convert these from "asserts fallback" to
   "asserts fail-loud raise" — not delete them, INVERT them.

## SHOULD-FIX

4. **[CONFIRMED] R1a-alone-green is not free.** The plan splits R1a (bark net) and
   R1b (cast_lock). But cast_lock's fail_soft repair currently FEEDS the voice nodes:
   an orphan cast that repair fixes today would, after R1a-only, reach the voice node
   with no ref and raise — so a test exercising that combined path breaks at R1a, not
   R1b. Fix: either (a) rip cast_lock repair + bark net in ONE commit (R1a==R1b), or
   (b) prove via the test list that no R1a-scoped test depends on cast_lock repair.
   Recommend collapsing R1a+R1b into one audio-voice commit — the two are entangled.

5. **[CONFIRMED] `missing_ref_fallback` metadata lives on indextts2/chatterbox/dia
   adapters + `base.py`.** Rip it from ALL of them + the two reader helpers
   (`_engine_requires_voice_ref` / `_engine_missing_ref_fallback`) in the same change,
   or a stale reader dereferences a removed attr. The elevenlabs/sonilo cloud
   adapters already set `missing_ref_fallback=None` — leave them.

6. **[UNVERIFIABLE→verify-at-build] Stage-direction silence rip.** The plan fails
   loud on empty-prepared-text beats. Before ripping, grep the goldens/fixtures for a
   stage-direction-only line; if any golden feeds one expecting the 0.30s silence
   clip, it breaks. Verify no fixture depends on it, then rip. (Operator: writer must
   never emit these.)

## INVARIANTS TO GUARD (reject any "fix" that breaks one)
- Local byte-identical defaults (indextts2+ref, kokoro+voice, SA3 music) unchanged.
- INPUT_TYPES / `build_engine_combo` / `load_resolver` C-5 safety KEPT (widget list
  must never crash the pack import).
- Transient network retry ladders (openrouter/ollama) KEPT — same-model retry ≠ swap.
- Every rip = a NAMED loud raise (EngineUnusable/ValueError/RenderError), never bare.
- Cloud already excluded from rank_chain — do not re-add.

## JUDGMENT SO FAR
Accepted: rip #1-#5 audio-voice; rip stage-direction silence (pending #6 grep).
Sequencing: collapse R1a+R1b (MUST-FIX #4). Keep: INPUT_TYPES safety, retries,
local rank-chain, observability best-effort. Verify-at-build: goldens for empty beats.
