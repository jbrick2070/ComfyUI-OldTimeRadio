# R3 ANCHOR (Claude, code-grounded) -- wiring / integration / sequencing

VERDICT: yes-with-fixes. Good news first: the fixes are almost entirely INTERNAL node
code -- NO workflow-JSON rewiring, no new nodes/widgets (low blast radius, honors the
"unwired code is dead" rule because there is nothing to wire). Two corrections to the
coding plan from the trace.

MUST-FIX (wiring/integration):
1. [FIX 5 -> trivial] role_mismatch is one line: `_otr_ledger_reviewer.py:500`
   `role = row.get("speaker_role") or row.get("tts_model") or ""`. Drop the
   `or row.get("tts_model")` fallback so an engine name can never be read as a role.
   THEN the upstream guarantee: trace the line-row builder and ensure every row gets an
   explicit `speaker_role` (the empty `speaker_role` is why the fallback ever fired).
   The schema split (cue_type vs speaker_role) is a SHOULD-FIX layered on top; the
   one-liner + speaker_role-always-set stops the violations today.
2. [FIX 3 RE-POINT -- the trace changes this] Do NOT add a SceneArcContext to compose.
   `LineRequest` already carries dramatic_question, beat_objective, beat_obstacle,
   beat_turn, beat_subtext, beat_tension(1..5), next_turn, outline_spine. The composer
   is NOT arc-blind. The arc unevenness lives UPSTREAM in BEAT-PLANNING (whatever sets
   beat_objective/turn/tension + the outline). Re-point FIX 3: audit the beat planner
   -- does beat_tension actually escalate across the arc? are objective/turn strong per
   beat? Fix the PLAN, and the per-line compose (already context-rich) renders it. [R4
   verify: read the beat/outline generator.]
3. [FIX 1] Thread `scope_line_ids: set[str] | None` through `run_story_critic` (def +
   both call sites): `_otr_freeze_cascade.py:754` passes None (whole-episode initial),
   `_otr_reroll.py:621` passes the patched target set. Inside the critic, when scope is
   set, evaluate only those line_ids + continuity neighbors. Add the reroll
   monotonic-decrease bail. Pure code; no graph change.
4. [FIX 2] Voice postcondition in OTR_CastLock (node 80) code: after replay, every
   character/announcer row -> non-None voice_preset (deterministic fallback or NAMED
   raise), independent of cast_seed. Pure code; no graph change.
5. No node/widget/JSON change required by any fix -> run OTR_WorkflowValidator only to
   PROVE no drift, not to add wiring.

SHOULD-FIX:
- The cue_type vs speaker_role schema split + invariant matrix + migration (GPT R2) --
  layer after the one-liner; needs a migration for existing `role` rows.

SEQUENCING (smallest blast radius first):
A. role_mismatch one-liner + speaker_role-always-set (FIX 5 core) -- trivial, test.
B. voice fail-closed (FIX 2) -- trivial, test.
C. critic scope + reroll monotonic bail (FIX 1) -- contained, test the convergence.
D. beat-planning arc audit (re-pointed FIX 3) + flat rubric (FIX 4) -- the quality
   levers; re-soak the minimal matrix to measure.
Run regression suite + Bug Bible after each per CLAUDE.md; commit+push per green chunk.

[R4 verify] the beat/outline planner (does tension escalate?); where speaker_role is
set on a line row; that no graph node consumes a field these fixes rename.
