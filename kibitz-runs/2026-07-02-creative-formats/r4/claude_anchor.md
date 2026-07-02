# r4 Claude anchor (convergence / residual defects)

VERDICT: yes-with-fixes (small). Three rounds tightened arc, contracts,
and wiring without architectural reversal since r1's FORMAT=ENGINE
decision. Residuals are bookkeeping:

MUST-FIX:
1. [1c] The FormatContext sub-model needs a version field named in the
   plan (`format_ctx_version: int`) with the assert_usable version
   handshake from my r3 anchor -- it survived into "fails CLOSED when
   format_ctx is absent" but the VERSION mismatch case dropped out.
   Absent and stale are different failures; name both.
2. [3/4 acceptance] The golden-30s samples need a HOME and a gate
   label: tests/goldens/formats/ under OTR_RUN_CLOUD_SMOKE=1 (they
   spend Kling credits) -- stated in r3 anchor, not yet in the plan.
3. [7] The plan's sequencing section still says F1 needs "S1 + kling
   row only" in sec 3's header while sec 2 prerequisite 3 says
   "Cloud S3 VIDEO lane (kling rows live)". Harmonize to the narrower
   TRUE gate everywhere: F1 starts when S1 + the kling_lipsync
   ADAPTER exist (not full S3 matrix acceptance).

SHOULD-FIX:
1. Board manifest sha stamped into the production ledger (write-once,
   read-only after image phase) -- from r3 anchor, confirm it landed.
2. One line stating the whole plan's teardown safety: format engines
   are additive; no existing engine, schema default, or workflow JSON
   value changes until the fmt sprints, so the local baseline stays
   byte-identical throughout.

CONVERGENCE CALL: if the panel's r4 finds only items of this class,
declare converged and ship the plan to GO_FORWARD as queued work.
