# r4 anchor review (Claude driver) -- convergence pass on BUILD_PLAN v2

VERDICT: CONVERGED / build-ready. Every r3 must-fix is folded and was re-grounded against
the real files (r3/judgment.md). No new must-fix found on the driver's re-read.

Residual verify-at-build items (honest, non-blocking, all gated by the plan's own tests):
1. Suite-fallout breadth across the ~20 video test files (fixtures pin 5-role shapes) --
   resolved by the pytest loop, guarded against fallback re-introduction by the plan's
   "never re-add fallbacks" rule + the new guard test.
2. Whether OTR_WorkflowValidator's link audit checks dst_slot-vs-input-order natively --
   moot: the plan's post-edit audit checks slot semantics explicitly either way.
3. scripts/otr_video_soak.py + run_otr_30word_smoke.py role enumerations -- covered by the
   grep gate in the build sequence (rename fallout surfaces at import/collection time).

Convergence check against the r2 contract: every contract bullet maps to a concrete
file+line change in the v2 plan; the only intentional deviations are ADDITIVE (node-3 JSON
+ link-2 dst_slot surgery, sequencer loud dispatch, dead sfx writeback fields, soak-script
producers, test_fixture_dur_s_audit.py) -- each grounded and consistent with the
contract's "KEEP: nothing sfx" + NO-FALLBACKS locks. Stop at convergence; no r5.
