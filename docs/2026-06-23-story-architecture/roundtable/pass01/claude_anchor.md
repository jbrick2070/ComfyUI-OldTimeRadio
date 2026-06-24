<!-- Claude grounded anchor review -- R1 (arc/creative). Written before reading panel output. -->
<!-- Grounded against the real repo via Desktop Commander 2026-06-23. -->

VERDICT: yes-with-fixes. The inventory is now grounded and the root-cause diagnosis (premise/structural sameness) is correct and triangulated, but the plan stops at "~10 levers, pick 3-4" without (a) selection criteria, (b) a divergence-FORCING mechanism, or (c) a fix for the SILENT mis-attribution failure mode in B4. Those three gaps are what would make the build wander.

MUST-FIX BEFORE BUILD:

1. [S2A / S6] The deliverable is "rank to the best 3-4 candidates" but the plan gives no SELECTION CRITERIA, so the ranking is unfalsifiable. Fix: score every lever on (leverage-on-root-cause x grade-evidence) / (cost x risk), and commit to a recommended 3-4 with that scoring shown. Root cause is premise sameness, so any lever that does not change WHAT story gets told is by definition lower-leverage.

2. [S2 move 1 / S2A A1] Pitch-room divergence assumes the local model CAN generate genuinely different premises -- but the triangulated root cause is that gemma/nemo collapse every premise into a console standoff. "Pitch 3 episodes" from the same model likely yields 3 flavors of the same standoff. Fix: FORCE divergence structurally -- seed each pitch with a different conflict-type drawn from the existing conflict palette / beat_role sequence in `_otr_story_quality_l12.py`, a different protagonist archetype, and a different setting class. Divergence must be injected, not requested.

3. [S2A B4] The plan names the prose->ledger parser as "make-or-break" but mis-states the failure MODE. The danger is not a crash (fail-loud is easy); it is a SILENT mis-attribution -- a line assigned to the wrong existing speaker passes the cast audit (the name is real) but renders in the wrong voice, and the no-fallbacks pipeline never sees an error. Fix: make attribution deterministic, not inferred -- require the draft in a lightly-structured speaker-prefixed form (NAME: line), or re-derive beats from the prose and DIFF against the outline before stamping; treat any unmatched/ambiguous line as a loud halt, not a guess.

4. [S2 move 1 / greenlight] The "showrunner taste pass" is the SAME local model that already grades; if it had reliable taste, `grade_story` / the 5B critic would already be lifting quality above B. Fix: state what powers the taste-select -- frontier lane, a structural proxy (e.g. divergence distance between candidates + arc-verdict), or human-in-the-loop -- do not assume local taste. This collapses into the S5 local-ceiling decision, which must be resolved BEFORE A1 is built, not after.

SHOULD-FIX:

1. [S2A axes] Axis A and Axis B are presented as parallel but have a sequencing dependency: flipping B2/B4 without fixing A yields better-executed sameness (risky parser spent, grade barely moves); A without B yields divergent premises still rendered as flat ping-pong. Fix: sequence A1 FIRST (cheap, attacks the root cause), measure the grade delta, THEN commit to the B4 parser only if A1 alone does not clear the bar.

2. [S0 / S5] Non-monotonic refine (42->72->65) is named but not diagnosed. Grader-noise (same text scored differently) and composer-regression (revision genuinely worsens text) need opposite fixes. Fix: instrument grade variance on IDENTICAL text first to separate the two, before "hardening the refine loop."

3. [S2 move 2] "Close the critic->planner loop" is the highest-value REUSE in the plan (the critic's arc_verdict already exists) but is under-specified. Fix: name the concrete wire -- `_otr_reroll_escalation.decide_escalation_scope` EPISODE branch currently regenerates the same plan; re-point it at the A1 pitch-room with the failing axis as the divergence seed.

CUT / DEFER (to hold the campaign to 3-4 candidates):

1. [S2A] A3 (write-long-then-compress) overlaps B4 + best-of-N; defer.
2. [S2 move 4] Listener-taste critic upgrade is lower-leverage than the critic->planner loop; defer behind it.
3. [S2A others] Distinct-character-voices and stage-climax-on-mic are real but are beat/line polish the existing `compose_exchange` + beat_role can partly absorb; fold into B-ladder, do not run as standalone sprints.

[ASSUMPTION] I have read the module docstrings + the freeze-cascade call sites + INPUT_TYPES, not every line of `_otr_reroll.py` / `_otr_outline.py` internals; the escalation-branch rewire (SHOULD-FIX 3) is verify-at-build.
