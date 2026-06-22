# Voice-casting architecture -- robustness + solid libraries + LLM-best casting

OTR is an offline ComfyUI text->radio-drama pipeline. This roundtable hardens the
VOICE-CASTING architecture end to end, per operator direction (2026-06-22):
  (A) a line with NO spoken content must never crash/silence-garble TTS;
  (B) every approved voice model must have a SOLID library of voices;
  (C) the LLM should make the BEST casting call -- the right GENDER + VOICE for each
      character -- instead of a blind deterministic pick.
Run R1 (architecture) -> R2 (coding) -> R3 (wiring) -> R4 (convergence) as needed.
Claude is a code-grounded panelist + the sole judge. 100% local, deterministic,
fail-soft (audio is king), no workflow-JSON churn unless wiring truly needs it.

## Grounded facts (verified vs the real source this session)
### Robustness (A) -- SHIPPED baseline (pushed; pressure-test, don't re-derive)
- `_otr_voice_node_common._render_per_line`: when a line's `prepared` spoken text is
  empty (stage-direction-only, e.g. "(pauses, then flips the switch)"), emit a 0.30s
  SILENCE and SKIP the worker -- engine-agnostic net (IndexTTS2 otherwise torch.cat's
  over zero chunks and crashes the whole render).
- `_otr_story_spine` Stage 3.7 + `_otr_line_hygiene.is_stage_direction_only`: the
  writer now RECOMPOSES a stage-direction-only character line into real dialogue
  (the scrub kept it; clean_spoken_text empties it). Fail-soft -> silence net.
- `cast_lock._resolve_character_voices_fail_soft` (STEP 3): never abort on a missing
  voice; a character row with no preset gets a deterministic engine-agnostic identity
  (`v2/en_speaker_*` is the universal id every adapter maps), a mis-stamped announcer
  line re-routes to the announcer, a true orphan inherits a voiced character.

### Libraries (B) -- the bank as it stands (config/voice_reference_bank.json, 137 refs)
- Per engine: indextts2 36, chatterbox 37, dia 36, kokoro 28. bark is NOT in the bank
  -- it uses 10 `v2/en_speaker_0..9` presets (+ the STEP-3 identity namespace).
- Gender balance (cloners): ~22 female vs ~14-15 male each; kokoro 15F/13M. So every
  engine has a usable library but is MALE-LIGHT; age_band/timbre tags exist per entry.
- VoiceBankEntry = {voice_ref_id, engine, gender, timbre[], age_band}; identity is
  voice_ref_id (never the character name, I-9).

### Casting intelligence (C) -- TODAY it is DETERMINISTIC, NOT LLM
- `_otr_voice_bank.assign_voice_for_slot`: a SCORING ladder gender(100)/timbre(40)/
  role(20)/age(10), stable-sorted, keyed on `stable_cast_seed(episode_seed, char_id,
  gender, timbre, role, age_band)`; walks g+t+r+age -> g+t+r -> g+t -> g; raises unless
  allow_voice_reuse. Announcer pinned per engine via `announcer_voice_ref`.
- `cast_lock._assign_bark_voices`: replays the writer's deterministic bark picker by
  `meta.cast_contract.cast_seed` (byte-identical) and stamps `voice_preset` per char.
- The LLM picks the CAST (names + gender + description) at the writer; it does NOT
  pick the VOICE. Character gender comes from the writer's cast contract; the bank
  matches on it. So a character's voice is "first seeded match by gender/timbre",
  NOT "the voice that best fits this character's described age/persona".

## The architecture questions for the panel
1. **Casting intelligence (C):** should voice selection move from blind deterministic
   scoring toward an LLM-INFORMED pick -- the LLM choosing the best-fitting
   gender+voice from the selected engine's library given the character description
   (age, persona, register) -- while preserving DETERMINISM (seed-keyed),
   reproducibility, $0 when possible, and the engine-agnostic fallback? Pure-LLM vs
   pure-deterministic vs HYBRID (LLM proposes from the library, deterministic
   validates + fails closed to the seeded scorer)? Where does it run -- at the writer
   cast contract, or at CastLock, and is it one extra LLM call or folded into an
   existing one?
2. **Library solidity (B):** is 137 refs / ~14-22 per gender per engine ENOUGH for an
   anthology that wants distinct casts per episode? Define a coverage bar (min voices
   per (engine x gender x age_band)) + a deterministic no-collision policy across a
   3-5 char cast; address the male-light imbalance; keep bark's preset namespace
   coherent with the cloner banks.
3. **Robustness (A):** is the two-layer net (spine recompose + per-line silence)
   sufficient + correctly ordered, or should the empty-content check also gate the
   CRITIC / freeze so a stage-direction-only line is a named mechanical defect, not
   just silently recomposed? Any engine besides IndexTTS2 that needs its own guard?
4. **Engine-agnostic identity:** voice_preset is currently the universal id that each
   adapter maps from -- is that the right contract for an LLM-chosen voice, or should
   the LLM choose a `voice_ref_id` directly and the adapter resolve per engine?

## Invariants the answer MUST respect
Deterministic + seed-keyed (C7 reproducibility). 100% local / offline. Fail-soft
(audio is king -- never abort a render). Ledger {cast,lines,meta} wire format frozen
(new fields ride free-form meta). speaker_role is the ONLY role source. Identity is
voice_ref_id / voice_preset, never the character name (I-9). Regression suite + Bug
Bible green per chunk. Prefer no workflow-JSON change; if wiring truly needs a node/
widget, it goes IN otr_scifi_16gb_full.json in the same change (and stays default-ON).
