# FIX PLAN: talking prompts for ia2v_canonical (lips-dont-talk, probe-proven)

## Root cause (empirical, canonical harness, one variable per probe)

| probe | delta vs WORKING canonical | mouth motion | r vs onsets | verdict |
|-------|----------------------------|--------------|-------------|---------|
| ref   | none (working reference)   | 4.15 | 0.12 | TALKS |
| P1    | render canvas 512x288 (production scale) | 3.73 | 0.28 | TALKS -- resolution EXONERATED |
| P2    | music-only audio | 0.59 | -0.03 | DEAD -- LTX cannot lip-sync to music |
| P3    | 241 frames @ 25fps, 9.6s audio | 2.37 | 0.17 | talks softer -- co-factor only |
| P4    | production-style SCENE prompt (no talking language) | 1.18 | 0.07 | DEAD + hallucinated title text |
| P5    | wide character SCENE still (small face) | (pending) | | |

**The killer is the TEXT PROMPT REGISTER.** The ia2v recipe animates what the
prompt narrates; audio conditioning modulates it. Production prompts steer
motion AWAY from the mouth:
- Bookends: `render_driver._LTX_MOTION_PROMPT_BY_ROLE` (:546-561) = "dial
  needle sweeps, tubes pulse, grille TREMBLES, dolly forward" -- zero
  mouth/speech tokens (by design, for the OLD ambient recipe).
- Characters: M4 prompts are ~900-char identity/scene walls; "speaking to
  camera" drowns; "cinematic/35mm" tokens even hallucinate on-video text (P4
  frame evidence).
- Music beats: NOT a prompt problem -- P2 proves music audio cannot drive
  lips at all. Music bookends KEEP the console-motion register by design.

## The fix (driver-side; engine untouched; no workflow-JSON change)

S1 -- Engine capability hook. `LtxAudioInEngine.wants_talking_prompt()` ->
  `_recipe_config(self._recipe())["two_stage"]`. The driver consults the
  ENGINE (never an env string) so the register follows the recipe.

S2 -- Bookend talking register. In render_driver, when the FINAL routed
  engine is ltx_audio_in AND wants_talking_prompt(): the ANNOUNCER bookend
  motion prompt becomes the canonical talking register --
  "The radio is talking: its big rubbery lips open and close naturally in
  sync with the announcer's voice, its dial eyes glance subtly as it speaks.
  Static camera, the radio stays in place."
  music_open/music_close/music_inter keep the EXISTING console-motion
  prompts verbatim (P2). Non-ia2v recipes + all other engines byte-unchanged.

S3 -- Character talking register (same gate): replace the 900-char M4 wall
  with a COMPACT talking-forward prompt for ltx_audio_in char-face beats:
  "The <=120-char appearance fragment> is talking to the camera, lips moving
  naturally in sync with the speech, subtle head and hand gestures. Static
  camera." Hard cap `_LTX_MOTION_PROMPT_MAX` (240) with the appearance
  fragment trimmed first (the proven BUG-LOCAL-112 budget discipline). M4
  stays untouched for HuMo and every other engine.

S4 -- Character still routing (ONLY if P5 shows wide stills kill
  articulation): for ltx_audio_in char-face beats under the talking recipe,
  prefer the beat character's minted PORTRAIT (face-forward) over the scene
  still as init_image; fail LOUD if neither exists (no fallbacks).

S5 -- Tests: announcer-bookend prompt carries lips/sync tokens under ia2v;
  music prompts byte-unchanged; char prompt compact + talking-forward +
  capped; distilled_native/sharp_lora prompts byte-unchanged; S4 routing
  test if built.

S6 -- Retest live: 30-word all-ltx proof episode. Acceptance: per-beat RAW
  clip (episodes/<ep>/clips/*.mp4) mouth-motion >= 2.0 on SPEECH beats
  (>= ~half the canonical 4.15 reference; the frozen band measured 0.6-1.2)
  + frame triplets for the operator's sound-on eyeball; music beats exempt
  (expected console-motion). Then commit+push (suite + Bug Bible + B7 green
  first).

Invariants: NO FALLBACKS; audio byte-identical (mux-LAST untouched); no
node/widget change; 14.5GB ceiling untouched (prompt-only + init selection).

## r2 OUTCOME (implemented; review the REAL wiring in the repo)

BUILT (uncommitted working tree, suite 5941/0 on the combined tree):
- S1: `LtxAudioInEngine.wants_talking_prompt()` RAISES on misconfig (no
  silent double-swallow); driver helper `_ia2v_talking_register_active`
  catches ONCE and logs LOUD; decision memoized per shot in
  `build_request_from_shot` (`_talking_register`).
- S2: announcer bookend swap in the motion branch (`_talking_swap`), talking
  register kept PURE (no atmosphere append). ANNOUNCER M4-OUTRANK: an
  announcer bookend carrying an M4 prompt clears it under the register so
  the motion branch swap fires (kibitz r2 codex must-fix #1).
- S3: char-face compact prompt: first sentence else last comma-clause under
  120 chars + `_IA2V_TALKING_CLAUSE_CHARACTER`; bounded 236<=240 by
  construction (dead trimmer removed per antigravity r2).
- S5: 8 register tests in tests/test_ltx_av_ia2v_canonical.py (+2 legacy
  announcer tests deliberately pinned to distilled_native).
- S4: NOT BUILT (out of scope until the live retest shows character beats
  still under-articulating). If it ever lands: WIDE-portrait-only with a
  LOUD RenderError on a portrait-aspect still (antigravity r2 must-fix).
- P5 postscript: the probe accidentally used a face-forward still (not a
  wide scene shot) -- face-fraction hypothesis remains open; judged by the
  live retest per-beat clips.
- Acceptance for the live retest: raw per-beat clips scored by
  scripts/otr_talking_radio_probe_eval.py; SPEECH beats pass at
  mouth_motion >= 2.0; a 1.2-2.0 score = one re-roll then operator call;
  music beats exempt (console-motion by design).
