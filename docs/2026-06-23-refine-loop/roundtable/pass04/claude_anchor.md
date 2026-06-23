ROUND 4 -- CLAUDE ANCHOR (convergence / residual defects; code-grounded)

VERDICT: yes-with-fixes (nearly converged). pass03 is internally consistent, matches the operator's
"revise-not-from-scratch, dropdown, until-grade" goal, and is lean enough to hand to a builder after a few
SMALL residual pins. The architecture has HELD since R1 (revise loop inside the writer node); R2/R3 were
pins, not rethinks.

MUST-FIX BEFORE BUILD (residual, small):
1. [Telemetry / runtime visibility] Each refine pass = (1 + act_count + num_voiced_beats) outline calls +
   every line compose + 1 grade call; at cap 5 that is ~5x an already-minutes-long compose. MAX_SECONDS was
   correctly cut, but the operator needs PROGRESS visibility on a long unattended run. Fix: a LOUD one-line
   `[refine] pass i/N grade=NN (target=BB)` log per pass (debug-level call-count estimate too). Not
   telemetry bloat -- a log line.
2. [Flags+gate / stop_reason] When the target is unreachable for the local model (e.g. A=90, model tops at
   ~78), EVERY run hits the cap then ships keep-best -- the slow path the operator was warned about. Fix:
   make `stop_reason="cap_reached_below_bar"` explicit in telemetry + a one-line LOUD warn so the soak /
   operator SEES that the chosen grade is unreachable and dials it down. (Reinforces "default B".)
3. [Build order 0 + 1 / overlay composition] chunk 0 wires `diversity_hint` and chunk 1 wires
   `prior_critique` into the SAME two builders (macro + beat). Confirm they compose cleanly when BOTH are
   non-empty. Fix: the byte-identical/render test must cover all FOUR combinations per builder (both empty
   => byte-identical; only hint; only critique; both) so a later refine pass (critique set, hint empty)
   and a best-of-N pass (hint set, critique empty) both render correctly.

SHOULD-FIX:
4. [Grader / build order 2] `extract_spoken_text_for_grade` ledger-row shape is still UNVERIFIABLE and the
   grader is on the critical path. Fix: chunk 2 must BLOCK on first reading `production_ledger` to confirm
   the row schema (where speaker_role/text live) before coding the extractor.
5. [Grader / honest floor] The grader reuses the SAME weak creative `generate_fn` that wrote the story --
   a weak model grading itself is the lenience risk flagged in R1. v1 accepts this; the soak measures it.
   Keep the honest-floor caveat (don't advertise quality lift until the soak proves grade lift) prominent
   at delivery.

CUT (final sweep): nothing material -- R1-R3 already cut remote, nested best-of-N, MAX_SECONDS, meta_delta,
elapsed_s telemetry, the all-failed fallback, phase-prompt critique, read-only-canon deep-copy.

VERIFY-AT-BUILD (all present in pass03's list -- confirm a concrete step each): exact exception classes;
ledger row -> spoken text; per-pass reseed re-rolls generate+compose+grade; model_copy(deep) isolates
intent mutation; canon read-only during compose; ComfyUI interrupt API; cast_seed in scope.

HOLISTIC: consistent end-to-end; matches the goal; buildable today in the 6-chunk order (0-5). Converged
pending the 3 small residuals above.
