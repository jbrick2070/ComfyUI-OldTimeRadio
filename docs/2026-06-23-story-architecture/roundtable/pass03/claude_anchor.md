<!-- Claude grounded anchor review -- R3 (wiring / integration / sequencing). Written before panel output. -->

VERDICT: yes-with-fixes. The candidate set is wireable, but three integration seams are wrong or unstated: where the pitch room executes (it is an in-conductor stage, not a graph node), which loop the two-tier escalation plugs into (the writer refine loop, NOT the terminal freeze-cascade exit), and the fact that the escalation it reuses ships DARK (enable_critic_escalation default OFF).

MUST-FIX BEFORE BUILD:

1. [C1 placement] The pitch room + greenlight must run as IN-CONDUCTOR stages inside `OTR_LedgerScriptWriter.run()`, between `news_interpreter` (which already sits at D.2->D.3, control-plane, in-conductor) and `generate_outline` -- NOT as new graph nodes. The brief/cast/outline data all flow inside the one conductor node; a new graph node would force a large workflow-JSON rewire and break that in-node flow. Fix: implement `_otr_pitch_room.py` as an injected-generate_fn helper (the `news_interpreter` pattern) called in run() after the briefs, writing the winner into `OutlineRequest.script_brief`.

2. [C2 loop target] "Re-outline / re-pitch" cannot hang off the freeze-cascade EPISODE branch: that branch stamps `needs_full_rerun` which is TERMINAL (skips Phase 7/8/10, halts Bark) -- it ends the pass, it does not re-plan-and-recompose. The existing re-run-and-keep mechanism is `OTR_LedgerScriptWriter._refine_loop` (v1, keep-best over up to 5 passes). Fix: wire Candidate 2 so a structural verdict drives the NEXT `_refine_loop` iteration with a re-plan (Tier 1 new outline / Tier 2 new pitch) as that iteration's input; the freeze cascade reports the verdict, the refine loop acts on it. Do not try to re-plan from inside the terminal cascade exit.

3. [C2 dark flag] The escalation Candidate 2 builds on (`decide_escalation_scope`) is gated by `enable_critic_escalation`, default OFF -- so today the structural verdict changes nothing. Fix: the Candidate 2 sprint must (a) turn enable_critic_escalation ON in the canonical workflow, (b) add `EscalationScope.PREMISE` + the freeze_cascade routing, and (c) validate them together on a live run; sequence this AFTER C0/C1 exist so there is a pitch room for PREMISE to escalate to.

SHOULD-FIX:

1. [C1/C2 config propagation] `OTR_GREENLIGHT_MODEL` / `OTR_STORY_REPITCH_MAX` / `OTR_STORY_REPLAN_MAX` must thread the same way the freeze cascade already gets model ids -- from the writer's broadcast sockets / env, not hard-coded. Fix: read them in the conductor and pass into the pitch/greenlight helper's generate_fn resolution + into `decide_escalation_scope`'s caps; fail-closed to local when OTR_GREENLIGHT_MODEL is unset.

2. [sequencing] C3 (`use_exchange`, compose-surface, downstream) and C1/C2 (premise/outline, upstream) are independent; flipping both at once makes a grade change unattributable. Fix: validate them on SEPARATE runs (one variable at a time) -- C3's N=3 VRAM/slot-drift run apart from C1's grade-delta run.

3. [sequencing] Lock the order: C0 gate -> (operator ceiling decision) -> C1 -> C2 Tier 1 -> C2 Tier 2 -> C4; C3 runs in parallel any time (config-only). C0 must not import C1's node (it uses the temp generate_pitches) -- confirm no back-dependency.

[ASSUMPTION] `_refine_loop` re-invokes the outline+compose path each pass (not just re-grades the same text); verify the loop body actually re-plans so Candidate 2 can ride it. If it only re-composes existing beats, Tier 1 re-outline needs a deeper hook.
