# R2 judgment log (Claude as judge)

Panel: gpt-5.5, gemini-3.1-pro, deepseek-v4-pro, grok-4.3 (live, ~$0.12) + Claude grounding.
All four well-grounded (correctly read line 131's `os.environ.get(...,"4.0")`, the call sites, mono
squeeze, the docstring). No hallucinations to discard. Verdict: CONVERGED into a hardened spec.

## ACCEPTED -- MUST-FIX (CONFIRMED vs real code)

- **Unset detection**: `os.environ.get("OTR_MASTER_MAKEUP_DB","4.0")` can't tell unset from set
  (line 131). Use empty-string-safe parse + mode-aware default (rms->0.0, peak->4.0); explicit value
  always wins. (GPT#3, Gemini#2, DeepSeek#2, Grok#1 -- unanimous.)
- **log10(0) trap**: rms can underflow to 0 in float32 even past `peak<1e-6` -> `-inf`. Early-return on
  `rms<=0`; compute RMS in float64; wrap in `np.errstate`. (Gemini#1, DeepSeek S#1, Grok#3, GPT S#3.)
- **Env parse safety + caching**: empty string -> `float("")` ValueError; 6 reads/clip is wasteful and
  risks a mid-render crash. Parse+validate+clamp ONCE (lazy module cache); sign-enforce
  (boost>=0, cut<=0, ceiling>0). (Gemini#2/S#1, GPT#4, DeepSeek#1, Grok#2.)
- **Calibrate `target_rms_dbfs`**: -20 dBFS is a placeholder; Bark sits ~-16..-18 dBFS so -20 may cut
  hard. GATE: do NOT enable `rms` in prod until measured from a real dialogue mix; LOUD one-time warn if
  rms mode runs on the uncalibrated default. (Gemini opt, DeepSeek S#3, Grok#4.)
- **Linear peak units**: document all peak values + `OTR_SEGMENT_PEAK_CEILING` are LINEAR (0..1), not
  dBFS. (DeepSeek#3.)
- **Return annotation**: `-> np.ndarray`, not `-> np.float32`. (GPT#2.)

## ACCEPTED -- SHOULD-FIX

- Active-sample RMS (mask samples above a low floor; fallback whole-clip) so leading/trailing TTS
  padding doesn't over-boost speech. (GPT S#1.) `_trim_trailing_silence` (:568) already trims the tail.
- Peak-safety may attenuate beyond `max_cut_db` -- documented as intentional safety + tested. (GPT S#2.)
- Centralize `_segment_loudnorm_mode()` (DRY across `_level_dialogue_clip` + `_master_loudness`). (GPT S#4.)
- SFX-path regression test: :726 still `_normalize_clip(...,0.85)`. (GPT S#5.)
- `_master_loudness` docstring update; `_rms` empty guard; gate uses `<=`. (GPT S#6, DeepSeek S#1/S#2.)
- Tests isolate env with monkeypatch so rms never leaks into the byte-identical baseline. (GPT S#7.)
- Determinism test must hold WITH the env overrides set to clamped non-defaults. (Grok S#2.)

## REJECTED / DOWNGRADED (with reason)

- `_normalize_clip` docstring "fix" -> trivial/no-op (it already says "peak normalization (not RMS)");
  add the word "sample" only. (DeepSeek#4, Grok CUT#1.)
- "assert no CUDA" + "repeat byte-identical for pure helpers" -> brittle/low-value; instead just keep
  the new code numpy-only and cover peak-mode parity + golden. (GPT CUT#1/#2.)
- "validate 2-3 episodes" as a build gate -> keep as operator/manual release acceptance, not a unit
  test. (GPT CUT#3.)
- Grok "default raises until env supplied" -> REJECT (breaks the dark default); use a warn instead.
- Grok "combined `OTR_SEGMENT_LOUDNORM=rms:...` string" -> REJECT (less discoverable, custom parse);
  keep 5 explicit env vars but CACHE them (addresses the per-read overhead that motivated it).
- DeepSeek master mechanism option (c) (caller passes makeup) -> use the inline env-read in
  `_master_loudness` via the shared helper instead (one place, no split logic). Functionally equivalent.

## OPEN (verify-at-build, carried to R3)

- Confirm a module-level `import os` is reachable for the new helpers (grounding only showed a local
  `import os` inside `_master_loudness`). (GPT#5.)
- Measure the real `target_rms_dbfs`; reconfirm the 4 `_normalize_clip` callers are exhaustive.
