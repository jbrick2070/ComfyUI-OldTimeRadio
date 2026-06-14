# Wan vs LTX opener eyeball -- findings (2026-06-14)

Operator-driven look-QA of the Wan i2v motion vs the 5/30-6/5 LTX opener look. All clips
render the SAME still (`otr_ltx_smoke_still.png`: vintage radio console framed against a
stormy-sea window). Bare-graph smokes (`otr_wan_smoke.py`, `otr_ltx_motion_smoke.py`) on the
live :8000 WAN-lane server -- no production code touched.

## Clips rendered

| tag | engine | settings | render | result |
|-----|--------|----------|--------|--------|
| i2v14b_seed42 (existing) | Wan 2.2 i2v 14B | cfg3.5 / 81f / shift8 | -- | DRIFT |
| i2v14b_lockedcalm | Wan 2.2 i2v 14B | cfg2.0 / 49f / locked-tripod prompt | 423s | DRIFT |
| i2v14b_anchoredmin | Wan 2.2 i2v 14B | cfg1.5 / 33f / shift5 | 248s | COLLAPSE (incoherent) |
| ltx_ksampler_i2v | LTX 2B v0.9 | ksampler 30-step cfg3.0 / 97f / 768x448 | 30s | HOLDS |
| ltx_goofer_i2v | LTX 2B v0.9 | 8-step distilled / 97f / 768x448 | 9s | HOLDS |
| ltx_ksampler_hires | LTX 2B v0.9 | ksampler 30-step / 97f / 1216x704 | 69s | HOLDS |

## Finding

**Wan 2.2 i2v (14B) is the wrong tool for held-still openers.** It reproduces the input still
for ~1 frame, then the single-image conditioning decays and the sampler re-interprets the scene
into its own subject (a generic tube close-up / glowing tube). This is NOT tunable with easy
input knobs:
- Lowering CFG 3.5 -> 2.0 + a "locked tripod, same console, no camera move, no cuts" prompt:
  STILL drifts to a different subject.
- Lowering CFG -> 1.5: the render COLLAPSES into incoherent abstraction (too little guidance).
The only real mitigation is the two-expert HIGH/LOW MoE handoff (Path B / GO_FORWARD 4A S3),
which is real engineering, not a setting.

**LTX (2B v0.9) holds the composition.** All three LTX modes keep the exact console + candles +
stormy-sea framing through the whole clip and add only subtle life (candle flicker, water,
gentle drift). The ksampler 30-step path == the production default (BUG-LOCAL-113b). The hires
1216x704 render also holds -> LTX can go past 480p and stay coherent (addresses the "low-res
vs my 5/30-6/5 openers" note).

## Recommendation

1. **Wan i2v 14B -> back-burner for the music/announcer OPENER role.** Keep it selectable (it
   renders technically + passes acceptance), but it is not a default opener engine. Revisit only
   with Path B (two-expert handoff) if camera-motion b-roll is wanted later.
2. **LTX stays the opener engine.** It does exactly what the operator wants (hold the still,
   add motion).
3. **Promote LTX-REGR (GO_FORWARD section 5) to the active thread.** The remaining question is
   MOTION AMOUNT: the operator recalls the 5/30-6/5 LTX openers as more dynamic; these hold well
   but may read "too static." Next probe = sampler/strength/sigma/frame-cap sweep on LTX to
   restore the 5/30-6/5 motion dynamism while keeping the held composition. (LTX i2v --strength
   is the prime lever: 1.0 = max freedom; lower preserves the still harder. cfg + step count +
   the distilled-vs-ksampler schedule are secondary.)

## Artifacts
- Wan clips: `C:\Users\jeffr\Documents\ComfyUI\output\otr_wanmotion\`
- LTX clips: `C:\Users\jeffr\Documents\ComfyUI\output\otr_ltxmotion\`
- Frame grabs + montage: `docs/2026-06-14-wan-ti2v/eyeball_frames/`
- Formal wan `--acceptance --only wan` (40w) running in parallel: `scripts/coverage_sweep_summary.json`
