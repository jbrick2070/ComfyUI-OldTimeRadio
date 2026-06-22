# R4 ANCHOR (Claude, code-grounded) -- convergence / residual defects

VERDICT: yes-with-minor. After three rounds the plan is sequenced, code-grounded, and
internal-only (no JSON rewire). It survived two real corrections in R3 (the false-halt
invariant; the build ordering). I see ONE residual structural gap + a few verify items,
none blocking the FIRST steps.

RESIDUAL (the one real gap):
1. STEP 6 (beat-planning arc audit) is still a "go read the planner then decide," not a
   defined change. It is also the BIGGEST quality lever (the "uneven arc" 50/55). Risk:
   STEPs 1-5 fix the mechanical defects (role/voice/reroll-convergence/flat-rubric) and
   may well lift clean-freeze rate, but if the underlying BEATS are weak, the critic can
   still rate the arc uneven. Fix: make "read the beat/outline planner + assess whether
   beat_tension escalates" the FIRST build task in STEP 6, with its own go/no-go, before
   committing to a beat-planner change. Do not let STEP 6 stay a vague aspiration.

NON-BLOCKING / VERIFY-AT-BUILD (fine to resolve during build):
2. STEP 1 "speaker_role set on every row" -- the exact line-row construction site that
   currently leaves it empty is unconfirmed; find it when implementing STEP 1.
3. STEP 3 cast_seed canonical key -- confirm one read path.
4. STEP 2 migration -- only needed if on-disk ledgers are replayed; confirm.

CONVERGENCE CALL: converged. The mechanical fixes (STEPs 1-5) are build-ready now with
tests; STEP 6 needs its grounding read as its first sub-task but does not block STEPs
1-5. No new must-fix beyond residual #1. Recommend: build STEPs 1-4 (small, contained,
high-confidence), re-soak to measure, THEN ground+decide STEP 6 with real data on
whether the arc is still uneven after the mechanical fixes. Do not re-loop the panel --
nothing new would surface.

[invariants guarded] no workflow-JSON change; fail-closed not silent; speaker_role is
the ONLY role source; reroll preserves approved lines; regression suite + Bug Bible per
green chunk.
