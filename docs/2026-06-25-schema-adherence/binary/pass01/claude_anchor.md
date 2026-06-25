CLAUDE ANCHOR -- binary-decision addendum (R1 arc/coherence). Grounded vs nodes/_otr_line_hygiene.py (split_stage_business L7 :544, is_stage_direction_only) + _otr_repair_prompts.py + the converged pass04 plan.

VERDICT: yes-with-fixes. The lever is SOUND + genuinely complementary to pass04 (tolerance handles the complex schemas you must accept; binary decomposition AVOIDS the schema for the highest-risk classification) and it is the most model-agnostic primitive available. But the addendum lists 4 applications when only ONE is a clean, grounded win, and the byte-identity gate is treated as a footnote when it is the load-bearing design decision.

MUST-FIX BEFORE BUILD:
1. [Applications -- prune to the proven win] The 4 candidates are not equal.
   - #1 dialogue vs stage-direction is a CLEAR win: it slots EXACTLY where
     `split_stage_business` already ABSTAINS (returns `(text,"","")` on the non-
     balanced-quote / undelimited classes, :549/:560/:575/:582). The deterministic
     code already punts those lines to the strip chain, so a binary escalation adds
     accuracy with ZERO byte-identity risk on the lines regex already handles.
   - #3 speaker membership OVERLAPS the existing Levenshtein `cast_membership_repair`
     (deterministic, no LLM) -- likely redundant. CUT.
   - #2 edit/no-op and #4 normalize_length per-boundary split are plausible but touch
     the freeze/segmentation seam + change attempt accounting -- HIGHER risk.
   FIX: scope v1 to #1 ONLY (the punted dialogue/stage-direction class); mark #2/#4
   "evaluate after #1 proves out"; CUT #3.

2. [Byte-identity gate = the spine, not a footnote] A binary LLM call where there
   was none changes output for the local default -> breaks byte-identity UNLESS it
   fires ONLY when the deterministic classifier ABSTAINS. Make this the load-bearing
   rule: `binary_decide` is invoked ONLY on `split_stage_business`'s "not confident"
   `(text,"","")` return; every line the regex is confident about gets NO binary call
   -> byte-identical. "Binary lane OFF or undecidable -> exactly today's strip-chain
   behavior" is a HARD test.

3. [Do NOT build a parallel mechanism -- reuse the pass04 core] `binary_decide`
   must be a THIN WRAPPER over a 1-field schema (`Literal["A","B"]` or a `bool`)
   run through the SAME `parse_validate_tolerant` + fail-loud path pass04 just
   converged on -- NOT a new call/parse/repair ladder. Output contract = a bare
   single decisive token; parse = first-decisive-token (ultra-tolerant; there is
   barely any schema to violate); undecidable -> None -> deterministic fallback.

SHOULD-FIX:
1. [Determinism + measurement] Seed-keyed; same offline conformance-harness +
   telemetry discipline as pass04 -- fixtures of the PUNTED dialogue/stage-direction
   class with expected binary outcomes (mocked slot_fn) + counters (regex-abstain ->
   binary-fired -> fallback-fired). No GPU.
2. [Fallback == existing behavior] On None / call failure, the fallback MUST be
   today's exact behavior (leave the line to the strip chain). Make "binary lane off
   == byte-identical to today" a regression test, mirroring pass04's strict-first.

CUT:
1. [App #3 speaker membership] redundant with the deterministic Levenshtein resolver.
2. [The "always-binary" variant] breaks byte-identity + a call per line; escalation-
   only is the only viable form.

[ASSUMPTION] not yet re-read whether the Script Doctor edits (#2) and normalize_length
(#4) have a clean "abstain" seam like split_stage_business -- verify before promoting
either past #1.
