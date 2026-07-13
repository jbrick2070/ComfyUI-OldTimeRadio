# Is the codex56 fair-play gate calibrated, or is the retake masking a P3 defect?

**Date:** 2026-07-13
**Base:** `1182c2bf` + the pending coordinate-gate deletion
**Status:** open question for review. Not a coding plan. No code has been written for it.

## The observation

I added a bounded truth-map retake at P4 (a corroborated fair-play block returns the
truth map to P3, which authored it, then re-audits once and fails closed). Before it, a
blocking finding raised `"fair-play audit rejected the truth map"` with no repair route
in existence.

It works. It has now fired on **both** live canonical 42-word runs since it landed:

- prompt `d5f66b1a`: P4 corroborated a block -> `P3_rerun` ran.
- prompt `07725d30`: P4 corroborated a block on its FIRST attempt -> `P3_rerun` ran.

Both runs then died inside my own coordinate validator (PBUG-20260713-13, now fixed by
deleting that gate) -- so I have not yet seen a `P4_rerun` verdict on a retaken map.

## The worry

A repair route that fires on 100% of runs is not a repair route. It is a second authoring
pass wearing a repair costume. Three readings, and I cannot yet distinguish them:

1. **P3 is under-specified.** The truth-map seam does not state some fair-play invariant
   the audit checks (clue-before-reveal order, audible sufficiency), so P3 reliably
   authors a map that P4 reliably rejects. The retake is papering over a missing rule at
   the authoring seam. If so, the fix belongs in the P3 seam, and the retake should
   become rare.
2. **P4 is over-strict / miscalibrated.** The fair-play seam invites `blocking=true` for
   ordinary imperfections, so a healthy truth map gets blocked. If so, the fix is to
   narrow what qualifies as blocking, and the retake should become rare.
3. **The retake is correctly load-bearing.** Fair play is genuinely hard to satisfy in one
   pass at 42 words, and a second pass is the honest cost. If so, leave it -- but say so
   deliberately, and expect ~2x the P3 token spend on every episode.

## What a reviewer should ground

- `_otr_original_codex56sol.py`: the P3 seam (`codex56_audible_truth_map`), the P4 seam
  (`codex56_fair_play_audit`), `_validate_truth_map`, `_corroborated_fair_blocks`, and
  the deterministic invariants P3's post_validator ALREADY proves.
- **Key question:** which fair-play properties does P4 check that P3's seam never states,
  and that `_validate_truth_map` does not already enforce deterministically? Anything in
  that gap is a rule the author was never given but is graded on -- the same class as
  PBUG-20260713-11.
- Is any fair-play property already proven in Python (making a model verdict on it
  redundant), and is any purely a matter of taste (making it non-blocking by the seam's
  own rule)?
- What is the honest cost of the retake at 120 and 720 words?

## Constraints on any proposal

- Python must never author story (Lesson 3). Clue placement, causality, and prose stay
  with the model.
- The deterministic validators are the authority; a model verdict may not overrule one.
- Fail-closed must survive: an unrepairable defect still ends the episode.
- No shims. Fix at the owning seam/validator boundary.
