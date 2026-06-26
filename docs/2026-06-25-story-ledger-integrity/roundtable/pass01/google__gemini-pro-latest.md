<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The architecture correctly isolates the LLM critic from the deterministic structural gates, but it fundamentally fails its primary mandate (WHOLE-STORY accuracy) by treating an LLM crash as a "strong" story, and it wastes LLM attention on dead-end telemetry.

MUST-FIX BEFORE BUILD:
1. **[Section A / `_otr_story_critic.py`] The Critic Fails Open on Accuracy.**
   *Defect:* `StoryCriticReport.clean()` (lines 189-197) returns `arc_verdict="strong"` and empty issue lists when the LLM ladder exhausts or raises an exception (lines 445-455). Absence of a report is treated as positive proof of a masterpiece. This silently passes broken narratives and defeats the entire purpose of the critic.
   *Fix:* Add `"unverified"` to `ArcVerdict`. `clean()` must return `arc_verdict="unverified"`. The freeze cascade must map `"unverified"` to a warning state (`frozen_with_warns`) or block it, never `frozen_clean`.
2. **[Section C / `_otr_freeze_cascade.py`] Contradictory State in A3 Mechanical Floor.**
   *Defect:* The A3 mechanical floor (lines 567-590) unconditionally appends deterministic anti-loop targets to `story_critic_report.reroll_targets`. If the critic failed open, you now have a report asserting `arc_verdict="strong"` that simultaneously demands rerolls for looping dialogue.
   *Fix:* If the A3 floor adds targets to a report where `arc_verdict` is `"strong"` or `"unverified"`, it must deterministically downgrade `arc_verdict` to `"uneven"`.
3. **[Section B / Cross-stage consistency] Missing Deterministic Upstream Guard.**
   *Defect:* The `sound_palette` bug occurred because nothing asserts that the ledger actually ingested the upstream contract. Relying on the LLM to notice missing fields fails.
   *Fix:* Implement an offline, CI-runnable schema parity test (`test_ledger_canon_parity.py`). It must dynamically reflect the `StoryContract` and `CastLock` Pydantic models and assert that every non-optional field has a mapped, populated equivalent in the frozen `ledger.meta` and `episode_canon` schemas.

SHOULD-FIX:
1. **[Section 3 / `_otr_freeze_cascade.py`] Doctor Edits Introduce Unchecked Drift.**
   *Defect:* The Script Doctor (Phase 1+2) runs *before* the story critic. The critic evaluates the *post-doctor* lines against themselves, not against the original outline. If the doctor rewrites a line away from the outline's intent, the critic will judge the new line as "strong" because it lacks the upstream context.
   *Fix:* Pass the original `beat_intent` from the outline into the critic's `_render_critic_user_prompt` so the critic can explicitly flag if the current `actual_words` fail to deliver the original `beat_intent`.
2. **[Section D / Drift guards] Positional Widget Drift is Unguarded.**
   *Defect:* BUG-LOCAL-097 (positional `widgets_values` appending) will silently corrupt the workflow if a new widget is inserted mid-list.
   *Fix:* Add an offline CI test that parses the target ComfyUI workflow JSON, extracts `widgets_values`, and zips them against the live `INPUT_TYPES` keys of the corresponding node classes. Fail the build if the indices misalign.
3. **[Section A / `_otr_story_critic.py`] Hallucinated `char_id`s Bypass Validation.**
   *Defect:* `_make_critic_post_validator` explicitly ignores `char_id` hallucinations in `voice_drift` (lines 352-390). While claimed to be "cosmetic", passing hallucinated cast IDs into the freeze cascade pollutes telemetry and risks breaking downstream cast-dependent render routing.
   *Fix:* Add a simple `valid_char_ids` set check to the post-validator for `voice_drift`, just as you do for `line_ids`.

CUT THESE (scope / over-engineering):
1. **`StanceIssue` (lines 150-166 in `_otr_story_critic.py`).**
   *Why it is safe to cut:* The docstring explicitly states it is "TELEMETRY ONLY", "DELIBERATELY NOT wired to a RerollTarget", and a "silent dead-end repair path". Forcing the LLM to evaluate and populate a complex 7-field object for a dead-end observation dilutes its attention budget for the actual reroll targets. Delete the model and remove it from the prompt.

[ASSUMPTION] I am assuming `StoryContract` and `CastLock` exist as typed schemas upstream that can be imported into a CI test suite, and that the ComfyUI workflow JSON (`otr_scifi_16gb_full.json`) is committed to the repository to allow offline widget validation.