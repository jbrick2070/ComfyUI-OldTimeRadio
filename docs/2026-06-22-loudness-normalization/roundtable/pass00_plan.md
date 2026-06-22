# Per-segment LUFS/RMS voice-level normalization -- problem statement + proposed approach

Round 1 input (high-level arc / approach). Goal: even out PERCEIVED loudness shot-to-shot so one
spoken line is not noticeably quieter/louder than the next across an episode. The operator's "low /
thin" Bark perception is a LOUDNESS problem, not a peak problem.

## 1. Current behavior (code-grounded -- verify before asserting)

All in `nodes/scene_sequencer.py`:

- **`_normalize_clip(clip_np, target_peak=0.85)` (line 93)** -- per-clip **PEAK** normalization. Its
  own docstring states the trade-off plainly: *"Uses peak normalization (not RMS) to preserve
  dynamics within each clip while matching overall loudness across clips."* It scales each dialogue
  clip so its single loudest sample hits 0.85; silence (<1e-6) is left untouched.
- **`_master_loudness(waveform, ceiling_dbfs=-1.0, makeup_db=...)` (line 109)** -- EPISODE-level
  master applied ONCE after crossfade in `assemble()` (line 1122): normalize to ceiling -> tanh
  soft-knee makeup limiter (`OTR_MASTER_MAKEUP_DB`, default 4.0 dB, clamped 0..12) -> re-trim true
  peak to -1.0 dBFS. Deterministic (no RNG). Added by the operator 2026-06-06.
- **`assemble()` (line 1031)** -- opening theme + main scene mix + closing theme, equal-power (sqrt)
  crossfades, then the single `_master_loudness` pass.

**Why it reads low/thin:** peak normalization matches the loudest *sample*, not perceived loudness.
A punchy delivery and a soft, breathy line can both peak at 0.85 yet differ by many dB in
RMS/LUFS. The episode master then lifts the whole mix uniformly, so the relative imbalance between
clips survives.

## 2. Proposed approach (to harden)

Replace/augment the per-segment peak target with a **perceived-loudness target** (integrated LUFS or
RMS) per dialogue clip:

- Measure each clip's loudness, compute a gain to hit a target, apply gain only (do NOT compress --
  preserve intra-clip dynamics).
- **Max-gain clamp** (+/- N dB) so a near-silent clip is not slammed up to the target.
- **Noise-floor gate** -- clips below a threshold (pure room tone / silence) are left alone (today's
  `peak < 1e-6` guard is the seed of this).
- Keep `_master_loudness` as the final peak-SAFE ceiling; **reduce/retune the +4 dB master makeup**
  so per-segment loudness and master makeup do not stack into double-gain.
- Deterministic, CPU-only (matches invariant I-11: post-engine DSP never touches CUDA).

## 3. HARD constraints / cautions (frozen audio spine)

- This is the **FROZEN audio spine** (`scene_sequencer.py` EpisodeAssembler + scene build), NOT
  upstream TTS. Changing `_normalize_clip` changes output bytes.
- It **breaks `test_audio_byte_identical`** -> a **deliberate, operator-gated golden re-baseline**
  (headless/GPU render to regenerate the fixture), not a regression.
- It hits **every** voice engine (indextts2 / bark / kokoro / dia / chatterbox), not just Bark.
- 100% local / offline; deterministic; UTF-8 no BOM; SFW. No heavy new dependency without a verdict.

## 4. Open design questions for the panel

1. **LUFS (ITU-R BS.1770-4 integrated) vs simple RMS** -- LUFS is perceptually accurate (K-weighted,
   gated) but needs a lib (e.g. `pyloudnorm`) or a hand-rolled K-filter; RMS is zero-dep but cruder.
   Is RMS "good enough" for spoken-word, or is the K-weighting worth a dependency? (Verify what is
   already importable in the venv before assuming.)
2. **Target level** -- broadcast -23 LUFS vs a louder spoken-word/streaming target (~ -16 LUFS)? How
   does the per-segment target interact with the existing `_master_loudness` makeup + -1 dBFS
   ceiling to avoid double-gain?
3. **Max-gain clamp + noise-floor threshold** values that prevent pumping quiet room tone up into
   hiss, while still rescuing a genuinely quiet take.
4. **Measurement window for short clips** -- one-word lines / very short segments give unstable
   integrated-loudness reads; momentary/short-term window or a min-duration fallback to peak?
5. **Where to apply** -- per dialogue clip in scene sequencing, or a unified per-segment pass? Should
   announcer / SFX / music beds be excluded or use a different target?

## 5. Invariants the hardened plan MUST NOT break

- `test_audio_byte_identical` stays the GATE; the re-baseline is intentional and operator-gated.
- Determinism (no RNG; same input -> same output).
- No upstream-TTS change; the fix lives in the assembler/scene DSP only.
- No heavy new dependency without an explicit license/size verdict.
- `_master_loudness` remains the final peak-safety ceiling (no hard clipping).
