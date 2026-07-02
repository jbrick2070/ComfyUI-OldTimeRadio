# lips-dont-talk -- FINAL (kibitz r4 convergence)

Arc: r1 (problem framing) -> r2 (fix plan) -> r3 (wiring) -> r4 (convergence).
Panel: codex + antigravity + claude-code (agy quota-dead after r2; r4 ran
claude+codex per the --only fallback). Driver anchors + grounding each round.
Empirical backbone: probes P1-P8 on the WORKING canonical harness
(docs/2026-07-02-canonical-ia2v/probes.log), one variable per run.

## The locked findings (probe-proven)

| # | finding | evidence |
|---|---------|----------|
| 1 | The ia2v recipe animates what the prompt NARRATES; the production scene/console register froze the lips | P4: 4.15 -> 1.18 + hallucinated title text |
| 2 | Even a PARAPHRASED talking register scores HALF the canonical wording | P8: 1.72 vs P7 3.32, identical params |
| 3 | LTX cannot lip-sync to music; music bookends keep the console register BY DESIGN | P2: 0.59 |
| 4 | Canvas, fps, frames, length are all EXONERATED (alone and combined) | P1 3.73, P3 2.37, P6 3.05, P7 3.32 |
| 5 | The engine's guide prep scaled 1.5x the RENDER canvas -> soft upscaled mouth prior at 512x288 (probes passed because the harness kept the canonical FIXED 1920x1088) | graph diff vs ia2v_flat_api_prompt.json |
| 6 | 512x288 -> deliverable upscale was the operator's "really low quality"; 720 is NOT /32 (engine gate catches it LOUD) | side_by_side mp4; proof5b RenderError |

## Shipped (c427dbd5 + the /32 follow-up)

- TALKING register: `_IA2V_TALKING_PROMPT_ANNOUNCER` = the canonical text
  VERBATIM (code comment forbids "improving" it); char beats = compact
  identity fragment + `_IA2V_TALKING_CLAUSE_CHARACTER` (canonical token
  pattern), <=235 chars; M4 OUTRANKED on announcer bookends under ia2v ONLY;
  seam-gap no-M4 fallback talks; motion-clause + atmosphere appends guarded
  off the talking register; music register untouched.
- Engine hook `wants_talking_prompt()` raises-loud; driver catches ONCE
  (logged) + memoizes per shot.
- Guide chain canvas-INDEPENDENT: fixed 1920x1088 -> longer-edge 1536 ->
  compression 18 (canonical verbatim).
- ia2v AV canvas default 512x288 -> **1280x704** (/32-safe canonical-native;
  proof5b caught 720/32 LOUD; single-pass recipes keep 512x288).
- Tests: ia2v register/canvas/guide locks in test_ltx_av_ia2v_canonical.py;
  legacy canvas-clamp + announcer-prompt tests pinned to distilled_native.
  Suite green + Bug Bible green each chunk.

## r4 judgment log

- codex MF1 (acceptance harness mismatch): ACCEPTED as doc-fix -- acceptance
  = per-beat windows on the obs final via the slice log + raw-clip motion
  via the evaluator's _mouth_motion; the r/delta mode is the probe-pair
  mode. Recorded here; script generalization = follow-up, not a blocker.
- codex MF2 (scope wording stale re canvas): ACCEPTED -- this doc supersedes;
  the canvas change IS in scope (operator-directed) with the live NVML check
  in the proof run.
- codex MF3 / claude SF1 (plan prose vs code constants): ACCEPTED -- the
  constants in render_driver.py are the truth; prose updated here (finding 2).
- codex MF4 (workflow JSON defaults): REJECTED as must-fix -- node-87
  dropdowns are OPERATOR-owned picks; the talking lane engages when
  ltx_audio_in is selected (or forced by a harness). No JSON change belongs
  to this fix.
- claude SF2 (test count bookkeeping): ACCEPTED -- 9 register tests
  (5 ia2v-gated + 4 single-pass-pinned) + canvas/guide locks.
- codex SF1 / claude VAB3 (stale suite counts): ACCEPTED -- final green =
  suite_final2.out 5991/0 + the post-704 targeted 36/36 (+ full suite in the
  closing chunk).
- claude OPT1 (comment arithmetic 236 vs 235): ACCEPTED, folded here.
- S4 (portrait init for char beats): stays DEFERRED with the build rule --
  if character SPEECH beats miss the bar on the final proof, record
  init_source + face fraction, run a portrait-vs-wide A/B, THEN build S4
  wide-portrait-only with a LOUD aspect guard.
- LTXVExtendSampler chunking: demoted from "next lever" (P6 killed the
  length hypothesis) to back-burner quality option for very long beats.

## Verify-at-build (the live proof run)

Real workflow JSON loaded by the smoke; histogram all-ltx; raw clips under
episodes/<ep>/clips/; SPEECH beats scored (bar: motion >= 2.0, 1.2-2.0 = one
re-roll then operator call; music exempt); NVML peak <= 14.5GB at 1280x704;
suite + Bug Bible + push, HEAD==origin. Results appended to HANDOFF_LOG.
