# R4 Claude anchor review (convergence / residual defects)

VERDICT: yes-with-fixes (small). The architecture has been stable since
R2 and R3 only tightened wiring; remaining defects are bookkeeping-level.

MUST-FIX BEFORE BUILD:
1. [pass03 sec 2] Session teardown says "assembler done signal or prompt
   completion" -- for ABORTED runs neither may fire. Fix: table entries
   carry created_at; a sweep at next session-create evicts entries older
   than N hours and logs LEAKED_SESSION with any unreleased reservations.
2. [pass03 sec 8 S1] The registration test ("every shipped row id
   registered when flag on") needs its inverse: flag OFF -> resolver
   rejects every cloud row with GATED_BY_FLAG, and the offline suite
   runs BOTH states in one process (env toggling between tests is a
   classic leak; use explicit reload or subprocess isolation).
3. [pass03 sec 4] "Excluded from obs_publish" -- name the mechanism
   (path allowlist in obs_publish vs sweep-ignore file), else it lands
   as a comment nobody wired.

SHOULD-FIX:
1. [sec 7] The operator-facing summary table (row, tier, surface,
   approx_cost, license status) should be generated from the registry +
   yaml into docs at S0 end -- keeps the plan's tables from drifting
   from shipped truth.
2. [sec 8] Name the S4 no-GPU acceptance host explicitly (the
   CUDA_VISIBLE_DEVICES='' pattern already used by conftest) so the
   acceptance is reproducible.

OPTIONAL: none. CUT: none.

CONVERGENCE CALL: if the panel surfaces no new MUST-FIX class defects
beyond bookkeeping, declare converged after this round.
