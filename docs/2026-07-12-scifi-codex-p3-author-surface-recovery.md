# Sci-Fi Codex P3 authored-surface recovery plan

## A. Decision to harden

The 120-word canonical `scifi_codex` qualification must reach a finalized
ledger and OBS asset under `workflows/otr_canonical.json`. No failed receipt,
pending ledger, or resident server counts as success.

This plan addresses only P3's repeat failure after the compact
`RadioScoreDraftV4` transport replacement. It does not change the canonical
workflow, source-bank selection, visualizer policy, or downstream P4-P9 work.

## B. Live evidence

1. `fab1bbbe-cfc1-484b-8f5b-61dfc296de6e` reached P3 after P0's new literal
   span repair. P3 emitted numeric `arc_phase` values and descriptive cue IDs.
   The compact P3 contract named fields and length caps but omitted their nested
   literal/type semantics.
2. `c91a7da6` added the exact `arc_phase` and cue-ID rules to the compact base
   and repair prompt. Focused tests, full Windows suite, Bug Bible, and
   workflow gates passed; the commit is pushed and equals origin.
3. Fresh prompt `4b19f3ed-bd28-4f84-9b81-5fcddfb89dc0` proved those literals
   were fixed: P3's base and typed repair contained no numeric phase or
   invented cue ID. Both nevertheless returned a complete draft with only three
   source-valid, model-authored strings over their strict caps: one scene
   description and two music-cue descriptions. The typed repair reduced one
   failing field but repeated the other three. The generic ladder correctly
   stopped after two calls because schema failures skip the useless structural
   reroll.

The live pattern is therefore a complete, semantically coherent draft with a
small set of overlong author-owned leaves, not a parser, graph, context-window,
or deterministic-mechanics defect.

## C. Current code-grounded constraints

- `nodes/_otr_scifi_codex.py` calls P3 through `_call_radio_score_draft`, using
  `RadioScoreDraftV4`, `clamp_overlong_strings=False`, and
  `include_result_json_schema=False` to keep measured base/restart/semantic-
  repair/rewrite envelopes below the 8,192-token local Gemma context window.
- `RadioScoreDraftV4` owns title/premise/setting, scene/shot prose, beat intent
  and arc, fact choice, cue choice, and local anchors. Python compiles only IDs,
  parents, ordering, speakers, advisory centers, and canonical cue placement.
- `_otr_structured_call.structured_call` uses base -> structural retry only for
  JSON decode failures -> typed repair. A schema-valid-but-content-invalid full
  rewrite is not a safe answer for a few author-owned string leaves.
- Generic string clamping is prohibited at this authored boundary. Truncating
  scene or music prose in Python would silently alter the model's creative
  decision and violate the P3 ownership boundary.
- `docs/PRODUCTION_SPRINT_LESSONS.md` sections 3, 4, 20, 21, and 22 require a
  bounded model-authored patch for localized semantic omissions/repairs, exact
  one-for-one targets, preserved valid literals, and full merged-artifact
  validation. The patch must not become a broad retake.

## D. Candidate root fix to evaluate

Introduce a P3-only **typed authored-text patch** only when all of the
following are true:

1. the raw response is one complete JSON object;
2. its only strict-schema defects are `string_too_long` at declared
   `RadioScoreDraftV4` authored string leaves;
3. every target path is whitelisted and maps one-to-one to the raw draft;
4. no graph, enum, reference, count, or non-string defect is present.

Python derives the exact target paths and maximum lengths from Pydantic's
validation errors. The creative model returns a small strict patch containing
only `{path, replacement_text}` rows, one per target. Python rejects missing,
duplicate, unknown, out-of-cap, or non-string targets; merges only accepted
replacement strings into the original raw object; then re-runs strict
`RadioScoreDraftV4` validation, the P3 compiler, rewrite-structure lock, and
the final `RadioScoreV4` graph validator.

The patch prompt must carry the exact original strings, all currently valid
immutable facts in the target context, each cap, and the rule that no other
creative or structural field may change. It must not resend or accept a full
replacement draft. A parse failure, an extra invalid field, ambiguity, or a
patch failure remains fail-closed through the existing bounded model ladder.

## E. Rejected alternatives

- Do not enable generic string clamping or use a Python `[:max]` slice: that
  changes model-authored prose without an authoring decision.
- Do not make another full-draft retry merely because the second answer made
  partial progress: that enlarges the failure surface and is the retry shim the
  production rules forbid.
- Do not map or normalize cue IDs/arc phases in Python: they are model-owned
  creative decisions. The previous live issue was correctly solved by making
  exact literals visible to the model.
- Do not increase the output/context cap: the draft is complete, and all
  measured envelopes already fit the safe 8,192-token window.

## F. Required implementation and verification

1. Define a narrow patch schema and pure target/merge helpers in
   `nodes/_otr_scifi_codex.py`; no workflow changes.
2. Cover valid single/multiple patches, missing/duplicate/unknown paths,
   over-cap replacements, non-string defects, graph-invalid merged outputs,
   base and typed-repair boundary behavior, and rewrite structure preservation.
3. Re-run the focused P3 tests, full Windows suite, Bug Bible, and the canonical
   validator/link/widget/round-trip audit; commit/push `v2.0-alpha` and verify
   `HEAD == origin`.
4. Selectively reset, verify port 8000/VRAM baseline, UTF-8 boot, and rerun the
   same canonical 120-word Codex bank. Require `RESULT SUCCESS`, fresh prompt
   ID/runtime/call repair receipt, actual finalized ledger, zero all-visualizer
   image objects, and existing OBS final asset.

## G. Driver anchor, before external review

VERDICT: **yes-with-fixes**. The localized authored-text patch is the only
currently supported path that preserves P3 ownership while avoiding a full
draft reauthoring loop. It is not yet build-ready until reviewers confirm its
Pydantic error filtering, patch schema, attempt-budget integration, and
rewritten-score invariants.

MUST-FIX BEFORE BUILD:

1. [D] Confirm a small patch can be introduced at the `structured_call` repair
   boundary without spending a hidden fourth model call or bypassing the existing
   journal/status receipt.
2. [D/F] Pin the exact path grammar and maximum-length authority so a patch
   cannot reach IDs, cast/fact choice, cue placement, or an unowned nested leaf.
3. [D/F] Validate the merged raw object through strict draft schema, compiler,
   rewrite signature, and final score graph on every acceptance path.
4. [F] Preserve the current 8,192-token proof for all P3/P3-rewrite envelopes;
   measure the patch prompt with the actual Gemma tokenizer before live use.

SHOULD-FIX:

1. [F] Make bounded patch attempt receipts distinguish base-draft rejection,
   patch request, patch acceptance, and merged graph acceptance without storing
   raw prompts or rejected prose in the durable ledger.

CUT THESE: generic clamping, a broad full-draft reroll, and context-cap changes.
