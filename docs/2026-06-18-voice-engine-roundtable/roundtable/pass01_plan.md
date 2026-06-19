# OTR Cast Voice-Engine Selection -- HARDENED (pass 1, judge-grounded)

## Decision: the final 1-2
- **PRIMARY (commercial-clean default): `chatterbox`** -- MIT, ~2-3 GB, zero-shot
  clone of any ref WAV, consumes the per-line delivery vector. GROUNDED: it is
  ALREADY fully wired -- 37 `engine="chatterbox"` bank refs in
  `config/voice_reference_bank.json` + a `char_chatterbox_v1` profile
  (`commercial_clean=true`, `allowed_voice_banks:[default]`). Nothing to register.
- **QUALITY OPTION (non-commercial / local-only): `indextts2`** -- best clone +
  delivery vector + emo_vector, but `commercial_clean=False` (bilibili license).
  Stays selectable; emits the EXISTING non-blocking gate warning when chosen so a
  commercial render never silently ships a non-commercial voice.
- **`kokoro`: ANNOUNCER-ONLY (char_voice promotion CUT this round -- see Rejected).**
- **`bark`: retained ONLY as the wired `missing_ref_fallback`** (PD1 "episode
  always renders"; `eng_indextts2.missing_ref_fallback="bark"`). Not a cast pick.
- **`Qwen3-TTS`: deferred** (clone path has no emotion knob; 7 GB isolated standup
  + Blackwell torch risk; no marginal value over the two cloners above).

These two give: unbounded voice variety (clone any vz_* ref incl. the operator's
own cast), per-character distinct+deterministic assignment (the bank's
`assign_voice_for_slot`, no-reuse), per-line emotion (delivery vector), and a
commercial-clean default with a quality escape hatch.

## The ONE real build task: an engine SELECTOR
The bank + profile wiring already exists; the only gap is that with
`voice_bank=default`, `cast_lock._resolve_char_engine` returns the FIRST entry of
`legacy_first_engines("char_voice") = (indextts2, chatterbox, dia, bark)` whose
profile allows that bank -> indextts2 always wins. To make chatterbox the
commercial-clean default WITHOUT changing personal-use behavior:
- Add a commercial-clean voice bank id (e.g. `default_clean`) to the bank/profile
  config; list it ONLY in `char_chatterbox_v1.allowed_voice_banks`; exclude it
  from indextts2's profile. The operator selects it via the EXISTING
  `voice_bank` dropdown on OTR_CastLock -- **no new widget, no new assigner**.
- (Alternative, rejected as default-changing: reorder `_LEGACY_FIRST_ENGINES`.)

## Verify-at-build (UNVERIFIABLE from this pass)
1. `chatterbox` (and `dia`) adapters actually consume `delivery_vector` and
   declare a valid `missing_ref_fallback` -- re-smoke both on the 5080.
2. The v2 audio chain resamples the per-engine rates to one timeline rate
   (kokoro 24000, indextts2 22050, bark 24000) -- confirm `EpisodeAssembler`
   resamples; else mixed-engine episodes drift.
3. `eng_kokoro.load()` calls `KPipeline(repo_id="hexgrad/Kokoro-82M")` -- confirm
   the base model is cached locally so a cold cache cannot network mid-render
   (begin_episode already preflights the voice `.pt`, not the base model).

## Plan corrections folded (CONFIRMED inaccuracies in pass00)
- Voice-ref attribute is NOT uniform: `indextts2` declares
  `voice_ref_kind="wav_path"` (a ref WAV path), while `kokoro`/`bark` declare
  `voice_ref_field` (`voice_ref_id`/`voice_preset`). Any uniform dispatcher read
  must handle both.
- `kokoro` is NOT CPU-capable as wired (`load()` hardcodes `device="cuda"`); drop
  the "<1 GB CPU floor" framing unless the device is parametrized.
- `bark` DOES give deterministic unique per-character presets via
  `_assign_bark_voices`; its real limit is a small pool + no delivery vector
  (corrected from "no per-character variety").

## Rejected (with rationale)
- **kokoro-as-char_voice -- CUT.** Two grounded blockers: (1)
  `eng_kokoro.prepare_text` is identity (no bracket/asterisk strip) -> characters
  would speak `[stage directions]` / `*asterisks*` aloud, unlike bark/indextts2
  which clean char text; (2) the missing-`.pt` guard in `generate_voice` swaps an
  unresolved ref to `self._episode_voice` (the single ANNOUNCER pool pick) ->
  every unresolved character collapses onto ONE announcer voice, breaking the
  per-character uniqueness + role-separation invariants. Plus no char pool. It is
  the largest net-new fragile change for marginal value -- chatterbox already
  covers the commercial-clean low-VRAM cloning tier. Keep kokoro announcer-only.
- **"Promoting chatterbox is a no-op needing vz_* re-registration under
  engine=chatterbox" (panel MUST-FIX) -- REJECTED as grounded-false.** Chatterbox
  already has 37 bank refs + a `default`-bank char profile. Only the selector
  above is missing.

## Invariants preserved (a fix that breaks one is rejected)
Single resident heavy <= 14.5 GB; 100% local/offline; determinism (seed-keyed,
per-character unique); per-line interface unchanged; dep isolation (V-12); frozen
audio spine + `test_audio_byte_identical` green; bark fallback stays wired;
UTF-8/no BOM; SFW.
