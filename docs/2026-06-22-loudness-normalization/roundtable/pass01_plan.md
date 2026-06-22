# R1 CONVERGED -- per-segment loudness normalization (approach locked)

Panel (live, ~$0.08): `openai/gpt-5.5`, `google/gemini-3.1-pro`, `deepseek/deepseek-v4-pro`,
`x-ai/grok-4.3` + Claude grounded panelist. All four returned "no -- too high-level"; crucially they
named the **same** unresolved decisions, so folding them resolves every must-fix. Approach = CONVERGED.

## Decisions (frozen as the R2 coding input)

1. **Metric = flat RMS per dialogue clip, NOT LUFS (v1).** Grounded: `_normalize_clip(clip_np,
   target_peak=0.85)` has no `sample_rate` arg; BS.1770/LUFS needs SR + 400 ms gating -> short lines
   ("No.") crash, and `pyloudnorm` pulls `scipy`. RMS is zero-dep, SR-agnostic, short-clip-safe,
   deterministic. (Unanimous: GPT CUT#1, Gemini CUT#1, DeepSeek MUST#5, Claude.) Optional later:
   A-weighted RMS if a listening pass wants more perceptual accuracy.
2. **Separate, dialogue-ONLY path -- do NOT edit `_normalize_clip` in place.** Grounded: it is also
   called for SFX at `scene_sequencer.py:726` (and 0.85 sfx mix at :869); changing it would corrupt
   non-dialogue. Add a new `_loudness_normalize_clip` (or a `mode`/`kind` arg); leave the SFX caller on
   peak@0.85. R2 must enumerate ALL `_normalize_clip` call sites. (GPT#8, Gemini SHOULD#2, DeepSeek#4,
   Grok#2, Claude.)
3. **Gain = peak-safe, clamped:** `g = clamp(target_rms / clip_rms, max_cut, max_boost)` then
   `g = min(g, peak_ceiling / clip_peak)`. Gain only, no compression (preserves intra-clip dynamics).
   The peak-safety cap is mandatory -- loudness gain alone can push peaks > 1.0 and hard-clip
   downstream / in the crossfade before the master. (GPT#5, Gemini#2, DeepSeek#6, Claude.)
4. **Guards:** keep `peak < 1e-6` silence guard; ADD a noise-floor gate (~ -50 dBFS RMS ~= 0.0032
   linear) -> unity gain on room tone; unity gain for non-finite / empty / too-short (< min active
   samples). Always return `float32`. (Gemini#1, GPT#6, DeepSeek#3, Grok#1.)
5. **Target + master interaction (the double-gain trap):** set `target_rms` from TODAY'S mean dialogue
   RMS so program loudness is ~unchanged, AND in loudnorm mode default `OTR_MASTER_MAKEUP_DB -> 0.0`
   so `_master_loudness` becomes a transparent sample-peak ceiling. Grounded: the master
   peak-normalizes the whole episode then tanh-limits; with per-clip-leveled clips of differing peaks
   the +4 dB tanh re-squashes high-crest clips and UNDOES the leveling. (Gemini#3, GPT#4, DeepSeek#1,
   Grok#3 -- strong convergence.) Master ceiling stays as the peak-safety net.
6. **Mode flag + operator-gated re-baseline:** gate behind `OTR_SEGMENT_LOUDNORM = peak (default) |
   rms`. Default = byte-identical preserved (`test_audio_byte_identical` stays GREEN). Flipping to
   `rms` is the deliberate, operator-gated golden re-baseline (headless render regenerates the fixture;
   eyeball the diff; validate loudness across 2-3 episodes, not one). (GPT#2/#9, DeepSeek SHOULD#2,
   Claude.)
7. **Robustness + deterministic tests (CPU, no CUDA -- invariant I-11):** silence unchanged, room-tone
   unchanged, quiet boosted-to-clamp, loud attenuated, peak ceiling honored post-gain, short-clip
   fallback, repeat-run byte-identical. Fix docstrings: it is SAMPLE-peak (not true-peak), and describe
   the real algorithm. (GPT#3/SHOULD#1-2.)

## Proposed numeric defaults (confirm/calibrate in R2)

- `target_rms`: calibrate from the current mix (placeholder -20 dBFS RMS pending a measured pass)
- `max_boost` +9 dB / `max_cut` -12 dB ; noise gate -50 dBFS ; `peak_ceiling` 0.95
- `OTR_MASTER_MAKEUP_DB` in `rms` mode -> 0.0

## Out of scope / deferred (panel-agreed cuts)

LUFS/BS.1770/`pyloudnorm`; scene-level aggregation (per-clip is what evens shot-to-shot);
true-peak/oversampled limiting (sample-peak ceiling suffices for local radio); SFX/music/theme
normalization (dialogue only in v1).

## Verify-at-build (carried to R2/R3)

Enumerate every `_normalize_clip` call site; confirm dialogue clips are mono at the seam; measure the
real `target_rms`; confirm `scipy` presence is irrelevant (RMS uses numpy only).
