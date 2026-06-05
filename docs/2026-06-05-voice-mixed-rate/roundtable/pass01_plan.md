# Hardened fix: resample bark-fallback clips to the primary engine rate

Status: converged in 1 pass (4-model panel unanimous on direction; grounding
resolved the open items). This is the implementation-ready spec.

## Decision

**Direction A**, scoped to the bark-fallback branch only. Reject B, C, D.
`pack_audio_batch`'s mixed-rate `raise` stays as a tripwire for *unintended*
mixed-rate bugs; we stop *intended* fallback mixing from ever reaching it.

## Resampler: `scipy.signal.resample_poly` (NOT torchaudio) -- grounded correction

All four panel models recommended `torchaudio.functional.resample`. I verified it
IS present (torchaudio 2.10.0+cu130) and deterministic on this venv, so it would
work. **But the codebase already has a canonical resampler and deliberately
avoids torchaudio:** `scene_sequencer.py:109 _resample_audio` uses
`scipy.signal.resample_poly`, and comment **I-11** states the prior GPU torchaudio
fast path was removed for post-engine determinism. Match that precedent -- one
deterministic CPU resampler in the project, not two.

## The change (one site)

`_otr_voice_node_common.py`, the bark-fallback branch (~lines 401-407). After
`audio = _bark_fb.generate_voice(...)` and before `clips.append(audio)`:

- if `audio["sample_rate"] != sr`: resample the waveform from its rate (24000) to
  the primary `sr` (22050, already computed at line 313 and passed to
  `pack_audio_batch` at line 423), set `audio["sample_rate"] = sr`.
- **Downsample bark 24000 -> 22050.** Never upsample the indextts2 clips (that
  would touch the primary engine's bytes and break C7 bit-exact).

Recommended structure: add a shared `resample_audio(audio_dict, target_sr)` to
`_otr_audio_utils.py` that canonicalizes (`canonical_audio`), resamples the last
dim per channel with `resample_poly` (lazy `from scipy.signal import resample_poly`
inside the function), and returns a `{"waveform":[B,C,T],"sample_rate":target_sr}`
dict. Call it from the fallback branch. (Optional later: `scene_sequencer` can
dedupe onto the same helper -- not required for this fix.)

## Companion hardening (GPT-5.5 caught this; grounded CONFIRMED)

`_bark_fb.unload()` at `:418-422` runs *after* the loop, not in a `finally`. A
raise inside the loop (now including the resample) skips it and leaves Bark
resident. Wrap the per-line loop so `_bark_fb.unload()` executes in a `finally`,
mirroring `generate()`'s `_teardown(adapter)`.

## Invariants preserved (verified against the code)

- **Byte-identical:** change is inside the `engine in _OTR_CLONE_ENGINES and not
  voice_ref` branch only; all-index-with-refs and pure-bark paths are untouched.
- **C7 determinism:** `resample_poly` is deterministic CPU (the I-11 choice);
  index clips are never resampled.
- **Audio is king:** still always renders; the fallback just now packs cleanly.
- **AUDIO contract / mono_safe / `empty_audio_batch(sr)`:** unchanged.
- **C-5 / 16 GB:** lazy import, CPU only, no model, no VRAM.

## Already-handled -- do NOT "fix" (grounded MISREAD of two panel claims)

- **Downstream cross-batch rates** (Gemini + GPT worried kokoro-announcer 24000 vs
  index-char 22050 hit SceneSequencer mixed): SceneSequencer **already**
  standardizes every batch to 48000 via `_resample_audio` (`scene_sequencer.py:592,
  686`). The 2026-06-05 09:33 smoke assembled kokoro + bark + stable_audio to a
  final mp4 -- proof it normalizes. Not a second bug.
- **Grok "zero-line after fallback emits wrong rate":** the zero-line case
  short-circuits at `:306-307` *before* the loop, where `sr` is already the
  primary rate. No code change; cover with a test.

## Regression tests (new)

1. all-index-with-refs -> no resample, `sample_rate==22050`, byte-identical.
2. mixed (>=1 ref, >=1 fallback) -> no `mixed sample rates` raise,
   `sample_rate==22050`, `B==#in-role lines`.
3. all-fallback (indextts2 selected, zero refs) -> packs at 22050, not 24000.
4. single fallback line (catches "only resample when both rates appear" bugs).
5. `mono_safe` -> `C==1`.
6. zero in-role lines -> `empty_audio_batch` at the primary profile rate.
7. determinism: same seed, mixed case, byte-identical across two runs.

## Out of scope (panel CUT; I concur)

B (default resample in the packer -- weakens the tripwire), C (project-wide
canonical rate -- touches every adapter, re-baselines C7), D (whole-episode bark
demotion / auto-reuse -- changes casting intent and, per DeepSeek, would not even
remove the fallback at 6 chars / 4 refs). A non-default `target_sr` kwarg on
`pack_audio_batch` is a possible *later* nicety, not part of this fix.

## Separate items this run surfaced (not this fix)

- **Matrix C2/C6 act_count floor:** 55w/3 acts and 30w/7 acts violate the writer's
  ~50-words-per-act minimum (it raises). Fix the *matrix* (use `act_count=auto`
  or 1 for short episodes), not OTR.
- **gemma-4-12b `normalize_length`** structured call fails 3x (pydantic
  `projected_word_total` missing) -> advisory skip. Separate gemma-robustness item.
