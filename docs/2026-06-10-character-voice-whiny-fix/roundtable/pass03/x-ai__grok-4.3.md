<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

**Blind spots (full pipeline)**

- Psychoacoustic thinness in drama mix: even RMS-matched clips can read “whiny” if they lack 2.5–5 kHz presence energy relative to the Kokoro announcer + bed. The plan’s spectral-centroid + low-mid ratio (P2a, P0 report) will catch gross tilt but not a narrow presence dip or formant-balance shift that zero-shot cloning often produces on polite or low-effort refs. (G18 already shows the four quiet originals are the exact refs most likely to produce this.)
- Prompt-content leakage inside IndexTTS2 (UNSURE): if the donor segment itself contains rising or questioning intonation, the cloner may copy that contour even when the delivery vector is omitted. P0 cell 3 only tests punctuation on the target line, not on the ref itself.
- Per-line vector + fresh seed interaction (G17): even with a fixed-character seed, a line that triggers “afraid + surprised” followed by one that triggers only “sad” can produce audible stance wobble that the ear interprets as pleading. The current plan’s seed-mode cell (P0-8) does not also log the per-line vector mass, so the two effects are confounded.
- 44.1 kHz → 22.05 kHz resampling quality inside the worker (G14). If the internal resampler has poor stop-band rejection, the cloned output loses exactly the high-frequency energy that gives vocal “body.” P0 cell 7 only tests one ref; the effect is ref-dependent.
- Mix-bed interaction with dynamic vs. compressed sources (H2c). Kokoro is dense; IndexTTS2 output is not. No measurement of short-term loudness or crest-factor difference against the actual bed is planned before P0a.

**Craft (voice-director / post-engineer moves that are automatable)**

1. Reference pre-EQ shelf: before any new ref is written into the bank, apply a gentle +2–3 dB low-shelf at 250–300 Hz (Q=0.7) and a presence dip at 3.5 kHz if the ref’s own spectral slope > –4.5 dB/oct. Store the processed version under a new sha; the audit tool (P2a) already computes slope, so this is a one-line addition.
2. Output-side “weight” pass (versioned, default-off): after `generate_voice` returns the clip, run a 3-band compressor with a 120 Hz crossover, 2:1 ratio, 6 ms attack, targeted only at the low band. This is exactly what a dialogue editor does on a thin ADR track and is fully deterministic + CPU-only.
3. Reference ranking by “effort” proxy already partially present in P2a (RMS, crest, voiced fraction). Add one more metric: ratio of energy in 120–350 Hz (male chest) vs. 180–450 Hz on voiced frames only. The four quiet originals will rank at the bottom; the composite “must-listen” score then naturally de-prioritizes them without manual listening.

**Critique of plan v3**

- P-OBS first is the single most valuable change (roundtable convergent). Everything else is gated on knowing which ref + alpha + vector actually hit the worker.
- P0 cell ordering (punctuation/ref before alpha sweep) is correct; the roundtable explicitly warned that a punctuation change alone could reproduce the whine.
- P2d loudness-normalizing the four quiet originals (G18) is the highest-ROI single action once the audit exists; the trimmer fix is correctly sequenced before any future download.
- Risk: changing vector derivation to PREPARED text (P1-1) discards parenthetical affect (G13). The plan acknowledges the tradeoff; cell 9 only measures it, which is the right scope.
- Cheaper dominant lever that could ship before the full table rewrite: `OTR_INDEXTTS2_EMO_ALPHA` default (P1-2). If P0 shows 0.45–0.65 wins on the real offending lines, this single env var plus the audit ranking may be sufficient for v3.1.
- P3b resolve-once cache must consume the identical tier-filtered bank that CastLock uses (P2c); otherwise the render-time fallback path (G16) leaks rejected refs. The plan states this, but the implementation order (P3 after P2c) must be enforced.
- Not worth doing in this round: dual-render rise-select (appendix) – 2× GPU cost for a marginal prosody fix when the seed-mode + alpha levers are still untested.

**Wildcards (≤30 min each, speculation flagged)**

1. (Speculation) Low-frequency modulation injection: take one thin male ref, mix in a 5 Hz sine at –32 dB on the amplitude envelope for 800 ms, re-normalize, and run it through P0 cell 4. Tests whether the cloner copies implied chest resonance.
2. (Speculation) Formant-aware low-shelf on cloned output only (H2c lane): a 180 Hz +3 dB shelf with formant tracking (simple LPC order 12) applied post-infer. 30-line batch, compare spectral tilt and operator “weight” score.
3. Measure correlation between terminal “?” count and final-500 ms F0 rise on the exact P0 matrix outputs. If r > 0.6, the punctuation lever (P1-3) becomes the dominant cheap fix; this is a 10-line pandas one-liner on the already-generated WAVs.