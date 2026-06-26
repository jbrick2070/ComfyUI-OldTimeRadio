<!-- Claude R1 anchor (grounded). Story-ledger integrity. -->

VERDICT: yes-with-fixes. The framing is right and the scope-narrowing (skip the
settled binary/leak/tolerance work) is correct. The two highest-leverage,
durable fixes are both DETERMINISTIC: (1) close the critic FAIL-OPEN, and (2) a
deterministic cross-stage consistency assertion set (the sound_palette-class
guard). The governing principle the doc states is the key insight and must drive
everything: an accuracy GUARD must not itself be an LLM that can fail-open --
otherwise we are stacking unreliable checks on unreliable output.

MUST-FIX (the durable core):
1. [Q-A / critic fail-open] CONFIRMED in `_otr_story_critic.py`: failure returns an
   all-empty report with `arc_verdict="strong"` (the safe-fallback at ~line 36/580),
   and absence-of-findings is treated as correctness. This is the worst class --
   a BROKEN accuracy check reads as a PERFECT story. Fix: a critic FAILURE (or a
   structurally-empty/timed-out result) must map to a NON-passing sentinel
   (e.g. `arc_verdict="unverified"`), distinct from a genuine "strong", so
   downstream never treats unrun == clean. Do NOT make this block render by default
   (that would gate ship on a flaky LLM) -- make it OBSERVABLE + restampable, like
   the loud fallback pattern used elsewhere. Cheap, deterministic, high-value.
2. [Q-B / cross-stage consistency] The sound_palette bug is the proof case: a field
   DERIVABLE from `StoryContract.sound_world` was silently absent from
   `episode_canon` for ~100 styles with nothing asserting it. Fix: a single
   deterministic, offline, CI-runnable `assert_ledger_consistency(contract, outline,
   castlock, canon, ledger)` that checks the canon/ledger faithfully reflect their
   upstream sources -- every contract-derived canon field is populated; canon
   title/premise/setting trace to the outline; ledger cast == CastLock; style ==
   contract.slug. This is the guard that catches the NEXT silent drop before ship.
   It must be PURE (no LLM) so it cannot fail-open.

SHOULD-FIX:
1. [Q-D / widget + schema drift] `OTR_WorkflowValidator` already exists for the
   positional-`widgets_values` drift (BUG-LOCAL-097). The fix is to make it a
   STANDING CI assertion over the canonical `otr_scifi_16gb_full.json` (widget-count
   vs live INPUT_TYPES, every wired input-name in INPUT_TYPES, link integrity), not
   an ad-hoc step. (verify: whether a test already invokes it in the suite.)
2. [Q-D / schema version] Add a tiny ledger-schema-version compat assertion: an
   old `l3-2026-05-14` ledger read by newer code must FAIL LOUD on a missing
   required field, never silently default to a wrong value.
3. [Q-C / freeze verdicts] Audit which freeze warn-classes ship under
   `frozen_with_warns` -- a continuity break the critic flagged should not be a mere
   warn. And guard that `frozen_with_doctor_edits` edits cannot push a line away
   from its outline beat (a doctor edit is itself a drift vector).

CUT / TRAP:
1. Do NOT build an LLM-based "whole-story accuracy validator" as the guard -- it
   reproduces the exact critic fail-open. The LLM critic stays a SIGNAL; the GUARDS
   are deterministic. The whole value is the deterministic consistency layer.
2. Do NOT widen the schema or add canon fields speculatively -- the guard asserts
   what ALREADY should hold, it does not invent new ledger content.

[ASSUMPTION] `OTR_WorkflowValidator` is not currently a CI-gated test (verify in
the suite). [ASSUMPTION] no existing pure-Python pass asserts canon-vs-contract
consistency (the sound_palette bug strongly implies none did).
