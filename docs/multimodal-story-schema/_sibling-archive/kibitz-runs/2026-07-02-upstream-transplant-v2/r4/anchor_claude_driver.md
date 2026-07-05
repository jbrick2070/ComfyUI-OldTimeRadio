# r4 Anchor Review (Claude driver, pre-fan-out)

Target: r3/final.md + the landed tree (post-r3-fix state, 46 tests green).
Focus: convergence / residual defects.

VERDICT: converged pending panel residuals. Driver residual sweep:

- Docs vs code: R1 architecture doc still contains superseded language
  (model_plan naming, "any declared pipeline" runner wording, per-file hash
  breadth) - each superseded item is marked in-place or in round finals;
  the finals chain is the authority. Acceptable; no code impact.
- GO_FORWARD chunk T1 order now matches PATCH_PLAN atomicity (callee-first).
- Gate list completeness for T1: production-side additions still needed
  tomorrow (writer self-test count update, whitelist parity test, science
  byte-pin pre/post, style-slug non-consumption test for non-science) - all
  named in PATCH_PLAN; none implementable tonight without touching
  production. Correctly deferred.
- Lab residuals: none known; drift pins, mirrors, matrix, runner, e2e
  handoff all tested on the ComfyUI venv.

UNVERIFIABLE: live RSS import behavior (story_orchestrator not mirrored) -
carried as verify-at-transplant, cannot be resolved lab-side by design.
