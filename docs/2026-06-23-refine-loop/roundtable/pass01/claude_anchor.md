ROUND 1 -- CLAUDE ANCHOR (code-grounded; written before the fan-out)

VERDICT: yes-with-fixes. The architecture is sound and grounded -- the loop wraps
`[outline -> compose-all-lines]` INSIDE `OTR_LedgerScriptWriter.run()` (CONFIRMED: compose_line
runs in-process at ~L3761/3842/3919/4051; v0 selector already wraps the outline at ~L2709), so v1
is an in-function wrap like v0, not a multi-node DAG loop. BUT the plan rests on two UNPROVEN premises
the prior roundtable already flagged, and they must be de-risked or honestly bounded.

MUST-FIX BEFORE BUILD:
1. [The idea / Open-Q C] The core claim "critique-informed regeneration improves a weak local writer"
   is unproven and the prior panel named the exact failure mode (weak model rephrases the same beats).
   Fix: make the improvement mechanism TESTABLE (per-pass grade delta in telemetry) and state the
   HONEST FLOOR explicitly -- if critique-steering does not raise the grade across passes, v1 is
   "keep-best best-of-N at the composed level" (ships the best of N, never truly improves). Do not sell
   it as more.
2. [Open-Q B / grader] The grader is the linchpin and underspecified. A weak local model grading its
   own prose is lenient + noisy -> keep-best degenerates toward random (echoes the v0 30-word TIE where
   all candidates scored identically). Fix: pin WHICH model grades, a CONCRETE rubric (not "B+"), and
   anchor the grade on objective signals -- reuse v0 `score_outline` structural metrics + a small,
   targeted rubric -- rather than a free-form holistic 0-100 a weak model cannot produce reliably.
3. [Open-Q E / runtime] The real cost is TIME, not dollars: each pass is a FULL compose (a 320-word
   compose is minutes); cap 5 => 15-40 min/episode. Fix: bound with a pass cap AND a wall-clock budget,
   and make early-stop-on-bar the common path. "Never stops" must not mean "unusably slow."
4. [scope] The plan implicitly composes every candidate. That is the expensive half. Fix: the cheap v0
   structural score already ranks outlines PRE-compose; v1 should pick a strong outline first (v0) and
   compose+grade only that per pass -- not compose-N-blind. Make the compose budget explicit.

SHOULD-FIX:
5. [Open-Q D] Overlap with the downstream freeze-cascade critic (already grades + rerolls at LINE
   level) is a real integration risk. Decide now: v1's per-pass grade is a LIGHT rubric; the heavy
   critic stays the SINGLE downstream freeze pass. Avoids doubling cost.
6. [determinism] Early-stop makes pass-count data-dependent. Assert SAME (seed, model, flag) => SAME
   pass-count + winner; add a test.
7. [telemetry] Add the per-pass grade DELTA so the validation soak can answer the one decisive
   question: did the loop actually raise quality, or just pick the luckiest of N?

CUT THESE:
8. Build step 4 (remote opt-in). CUT from v1. A refine pass is N full composes = N x (many paid
   calls); a remote refine loop is a cost-runaway even with a guard. Local-only, period. v0 already
   owns the remote pattern if it is ever wanted.

[ASSUMPTION] The grader runs as an in-writer local LLM call (precedent: run_story_brief_reflection
~L4530). Verify a creative/technical slot fn is in scope at the grade point.
