<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: yes-with-fixes — L5a advisory critic insertion point and verify steps for the three residual items remain underspecified; everything else has converged.

MUST-FIX BEFORE BUILD:
1. [build order step 1] Advisory grade-only `run_story_critic` placement is stated only as "BEFORE the terminal stop" — add the exact call site (`_otr_ledger_reviewer.py` after `edits_applied == -1` check but before the `original_snapshot` rollback at :2030) so the pre-rollback ledger is used.
2. [build order] The three residual verify-at-build items have no corresponding assert or test step — append one-line concrete checks to step 2: `assert type(beat).__name__ == "Beat"`, `assert "model_dump" not in outline_to_ledger_path`, and `assert allowed_people and allowed_things at writer call site`.
3. [L1 crisis-noun repair] "new SQ meta summary fields" is listed as a mutable target but the fields live under `beat.meta` — change the allowlist wording to `beat.intent + beat.meta["conflict_object"|"conflict_type"|"sensory_consequence"|"state_change"]` (or equivalent dotted access) so the deterministic substitutor has an unambiguous target.

SHOULD-FIX:
1. [Data model] `choice_summary` is already CUT yet still appears in the optional field list — delete the mention to prevent accidental scaffolding.
2. [Acceptance metric] "store raw numerator+denominator" lacks a concrete ledger key name — specify `meta["story_quality"]["ungrounded_crisis"] = {"matches": N, "total": M}`.

OPTIONAL / NICE-TO-HAVE:
- Add a one-line no-op guard in the L1 substitutor so it silently skips when flag is OFF (defensive, not required for build).

CUT THESE:
None — all remaining items are either required wiring or already deferred.

VERIFY-AT-BUILD checklist:
- Beat is exactly the Pydantic class shown in GROUNDING (no OutlineBeat alias).
- `model_dump()` path from outline to frozen ledger never emits excluded fields.
- `allowed_people`/`allowed_things` are populated at the writer call site before L1a render.
- Advisory critic report is present on terminal verdicts after the L5a edit.
- No-drift JSON assert passes with L12 flag both ON and OFF.