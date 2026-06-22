# R1 judgment log (Claude as judge)

Panel: gpt-5.5, gemini-3.1-pro, deepseek-v4-pro, grok-4.3 (live) + Claude grounded review.
Verdict: CONVERGED on approach. All four panel reviews flagged the same gaps; none were code misreads.

## ACCEPTED (grounded CONFIRMED -> folded into pass01_plan)

- RMS over LUFS for v1 (no sample_rate at the seam; short-clip gating crash; dep weight). CONFIRMED vs
  the real `_normalize_clip` signature.
- Dialogue-only via a SEPARATE function; do NOT mutate `_normalize_clip` (shared by the SFX caller at
  :726). CONFIRMED vs grep (call sites 726/869).
- Peak-safety cap after the loudness gain (`min(g, ceiling/peak)`). CONFIRMED (gain-only can exceed 1.0).
- Master double-gain: `OTR_MASTER_MAKEUP_DB -> 0.0` in loudnorm mode. CONFIRMED vs `_master_loudness`
  (peak-norm-then-tanh squashes high-crest clips).
- Noise-floor gate above the 1e-6 silence guard (~ -50 dBFS). CONFIRMED (1e-6 is digital silence only).
- Mode flag (`OTR_SEGMENT_LOUDNORM=peak|rms`, default peak) + operator-gated golden re-baseline.
- Robustness (NaN/Inf/empty/float32) + deterministic CPU tests; docstring fix (sample-peak, real algo).

## REJECTED / DEFERRED (with reason)

- LUFS / BS.1770 / `pyloudnorm` now -- over-engineering for spoken-word + dep/SR cost (panel CUTs).
- Scene-level loudness aggregation (DeepSeek optional) -- per-clip is what fixes shot-to-shot; defer.
- True-peak/oversampled limiting (GPT#3) -- accept as a WORDING fix only; sample-peak ceiling is enough.
- Broadcast -23 LUFS target -- N/A under RMS; target is an RMS dBFS level calibrated to the current mix.

## OPEN (verify-at-build, carried to R2/R3)

- Real `target_rms` value (measure the current dialogue mix).
- Full `_normalize_clip` call-site enumeration; mono-vs-stereo at the seam.
- Final numeric defaults (boost/cut clamp, gate, peak_ceiling) pending a measured/listening pass.

## No new material expected from a re-fan of R1

The four families agreed; remaining work is the CODING plan (R2), not more approach debate.
