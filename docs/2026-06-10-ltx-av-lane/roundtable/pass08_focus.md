# PASS 08 REVIEW FOCUS: FINISHING / CONVERGENCE SWEEP

You are one panelist on the FINAL pass. The plan below is the product of
seven grounded passes (architecture, I/O, prompts, wiring, testing,
hardware, pre-mortem). Your ONLY job: find anything BUILD-BLOCKING that
survived -- a contradiction between locked sections, a missing
coder-actionable detail that would stall a coder window mid-ticket, a
gate that cannot be evaluated as written, or a sequencing error in the
milestones.

Explicitly NOT wanted: relitigating locked decisions, style opinions,
nice-to-haves, new features, more detectors, more tests beyond gaps in
the stated matrix. If a finding is not build-blocking, put it in ONE
line under SHOULD-CONSIDER at most.

Checklist to sweep:
1. CONTRADICTIONS: do any two sections disagree (names, env vars, file
   paths, thresholds, chain orders, role lists, tuple contents)?
2. CODER-STALLERS: for each touch-list item, could a competent coder
   start it from this document alone (plus the named grounding files)
   without a question back to the operator? Name the first question
   they would be forced to ask, if any.
3. GATE EVALUABILITY: is every PASS/FAIL bar measurable as written
   (units, tools, thresholds)? Any gate that depends on an undefined
   artifact?
4. MILESTONE ORDER: M0 before M1 is intentional (probe gates the lane;
   M1 is CPU-safe and COULD start in parallel) -- is the dependency
   structure stated correctly? Any milestone consuming an output that
   does not exist yet at its start?
5. TICKET BOUNDARIES: propose the cleanest cut of coder-window tickets
   (2-4 tickets) over M1-M4 with explicit done-criteria each -- or
   endorse a cut if one is implicit. One ticket must never span a
   suite-red intermediate state.
6. ANYTHING ELSE that would make you say "no" to "build-ready as-is?"

Output format: VERDICT line ("CONVERGED -- build-ready" or "NO --
build-blocking items remain"); then numbered BUILD-BLOCKING items (file/
section + the exact fix), then SHOULD-CONSIDER one-liners, then the
ticket-cut proposal. Terse. Cite grounding or mark VERIFY-AT-BUILD.
