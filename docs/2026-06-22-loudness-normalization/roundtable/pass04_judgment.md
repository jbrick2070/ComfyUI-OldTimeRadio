# R4 judgment log -- CONVERGENCE (Claude as judge)

Panel: gpt-5.5, gemini-3.1-pro, deepseek-v4-pro, grok-4.3 (live, ~$0.19) + Claude grounding.
R4 found ONE genuine must-fix (a consolidation typo) + precision refinements; the ARCHITECTURE held firm
across R2->R3->R4. Folded -> CONVERGED. Total campaign spend ~$0.52.

## ACCEPTED -- MUST-FIX (unanimous, real)

- **dB-vs-linear gain bug (CRITICAL).** The pass04 consolidation wrote `g=clamp(target-rms_dbfs,...);
  return clip*g` -- multiplying the clip by a dB value (a -6 dB cut would multiply by -6.0: phase
  inversion + huge boost). All four caught it. FIX (restore the pass02 form): `gain_db = clamp(...);
  gain_lin = 10**(gain_db/20); gain_lin = min(gain_lin, peak_ceiling/peak); return (x*gain_lin).astype(f32)`.
- **Remove the `_segment_params` cache.** Caching at first call breaks `monkeypatch.setenv` test isolation
  and buys ~nothing (env-get is a dict lookup, negligible vs TTS/resample). Read env per call. (GPT CUT#1,
  DeepSeek CUT#2, Gemini#3, Grok#3.)
- **Wire the byte-compare skip-guard as a REAL edit** to `tests/test_audio_byte_identical.py`: skip when
  `os.environ.get("OTR_SEGMENT_LOUDNORM","peak").strip().lower()` not in {"","peak"}. The spec claimed it
  skips but listed no edit. (GPT#7, Gemini#5, DeepSeek [A].)
- **`_master_loudness` mode-aware default INLINE** (it reads os.environ directly, doesn't call
  `_segment_params`): explicit/non-empty `OTR_MASTER_MAKEUP_DB` wins (parse-safe); else 0.0 if rms else
  4.0. (GPT#5, Gemini#4, DeepSeek#3, Grok#1.)
- **`_rms` finite-check BEFORE squaring** (NaN/Inf -> mean NaN, not 0.0): `if not np.all(isfinite): return
  0.0`. All early-return paths return float32 (no silent float64 downstream). gate mask in LINEAR
  (`gate_amp=10**(gate/20); active=abs(x)>=gate_amp`). (GPT#2/S#3, DeepSeek#2/S#2.)

## ACCEPTED -- SHOULD-FIX

- Peak-safety legitimately attenuates BELOW max_cut to honor the ceiling -- it is applied AFTER the
  [max_cut,max_boost] dB clamp and is a SAFETY, not a loudness target. Tests assert `final peak <=
  ceiling` (not gain-in-range after peak-safety). REJECT Grok#2's "re-clamp after peak term" (it would
  defeat peak safety). (GPT#3, Grok#2 reframed.)
- Calibration script shares the SAME gate/RMS helper as production so `measured+4` is apples-to-apples.
  (GPT S#5.) Unknown mode -> warn once. (GPT S#1.) Preflight log (mode/target/makeup) lands in Chunk 1,
  not just the operator step. (GPT S#6.)
- Active-mask fallback is for the GATE-CHECK value only; if all samples <= gate the clip is room tone ->
  unchanged (no room-tone amplification). (DeepSeek S#3.)

## JUDGE'S DESIGN NOTE (Gemini#2 -- recorded, NOT a v1 code change)

Gemini flags that `_master_loudness` ALWAYS peak-normalizes the episode to the ceiling (line 128), so in
rms+makeup=0 it still applies one GLOBAL gain. That global gain is UNIFORM -> it preserves the
shot-to-shot (within-episode) leveling the feature targets (the operator's goal). Cross-EPISODE absolute
loudness matching is a larger, separate goal (would need an integrated-LUFS master target) and a deeper
frozen-spine change -- OUT of v1 scope. The `target=measured+4` is a STARTING POINT; the operator listen
test is the real validator (DeepSeek [A] agrees). Recorded as a verify-at-listen item, not a v1 mandate.

## REJECTED / DOWNGRADED

- Grok#2 re-clamp after peak-safety -> defeats clipping protection. Rejected.
- "one runtime peak run before landing" as a BUILD gate (GPT CUT#2) -> DOWNGRADE to release checklist; the
  in-suite peak-parity unit test is the build gate.
- Keep the tiny `_segment_loudnorm_mode()` helper (Grok CUT#2 wanted it inlined) -- it's used in 3 sites
  now (router, master, test skip-guard); GPT S#1 wanted exactly this consistency. Minor; kept.
- `_parse_env_float` generic sign-enforce (DeepSeek/Grok CUT) -> SIMPLIFY to inline try/except+clamp per
  var (4 floats); drop the generic indirection.

## CONVERGENCE

Approach (R1), coding (R2), wiring (R3) were not re-challenged. R4's items are codegen precision + one
typo. A 5th pass would only confirm the corrected `10**(dB/20)` formula -> STOP per the convergence rule.
The final build-ready spec is FINAL_PLAN.md.
