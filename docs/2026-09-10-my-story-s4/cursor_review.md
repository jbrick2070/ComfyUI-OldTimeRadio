VERDICT: yes-with-fixes -- canonical wiring, admission, runner, freeze policy and mux contract match D1 and would not brick a canonical first graph run; one surviving spec fork would let a later pass reintroduce a hard length gate D1 still writes as a post-validator.

MUST-FIX BEFORE BUILD:
1. [D1 §3 P2/P3 + r4 input L37-39 + nodes/_otr_my_story.py ActScript/StoryFrame] Build-blocking ambiguity that R1-R3 left open and r4 then split further. D1 still specifies `lines: 6..40` and intro/outro `1..3` / `1..2` as schema/post-validator bounds. Code enforces only nonempty (`Field(min_length=1)`, no max) in `nodes/_otr_my_story.py`. The r4 anchor says those counts are prompt guidance. Two builders, two incompatible lanes: one retries a 5-line act to death, one accepts it. Smallest lock: amend D1 P2/P3 in one sentence to match the code (nonempty spoken text + cast/topology gates only; numeric counts stay in `nodes/story_packs/my_story/my_story.json` as prompt text). Do not add pydantic `max_length`.

SHOULD-FIX:
1. [D1 §3 P0 + nodes/_otr_my_story.py `_make_interpret_validator`] Capacity refusal RAISES `MyStoryCastError` from a `post_validator` that `_otr_structured_call.validate_tolerant_data` documents as `-> str | None`. Today `structured_call` re-raises generic `Exception` on attempt 1 (`nodes/_otr_structured_call.py` ~975), so `tests/test_my_story_runner.py::test_an_impossible_cast_fails_before_any_creative_call` sees one technical call. A later ladder change that wraps all post-validator exceptions into `PostValidationError` would burn three LLM calls and fail with a planned-mismatch string instead of "voices in stock". Smallest fix: if `expected > voice_capacity`, return that English string (or raise after `structured_call` returns), never raise from inside the validator.
2. [D1 §2.3] Promised test: style roll + writer's digest equals validator's. What exists: validator persists `visual_style_requested == "roll (any style)"` (`tests/test_my_story_validator.py`); writer persists the same sentinel after `resolve_style_selection` rebinds (`tests/test_my_story_runner.py::test_real_writer_routes_user_fields_to_a_clean_ledger_and_shared_tail`). They never share one `prompt_id`/`digest`. Add one test: queued admit, then `OTR_LedgerScriptWriter.run(...)` with the same literals including the style sentinel, assert one draft file and equal digests.
3. [D1 §7.2 + nodes/otr_master_audio_mux.py `_delivery_intent`] `script_json=""` (INPUT_TYPES default, forceInput) is a hard fail for every bank. Canonical is wired (`workflows/otr_canonical.json` link `[291, 1, 1, 85, 10, "STRING"]`). An in-memory graph that has not reloaded that file will pass `""` and fail sci-fi/original too. Either document "mux script_json must be wired after this change" or treat default `""` as absent (`None`) and keep the error for non-JSON / non-object payloads. [ASSUMPTION] ComfyUI prompt conversion emits the default `""` for an unconnected new forceInput.
4. [D1 §3 P5 + nodes/_otr_my_story.py `_assemble`] Incremental path only `led.set_lines` + unchecked `led.save()` after preamble/acts; `set_scenes/set_shots/set_beats/set_music` and `_require_ledger_save` wait until the end. A crash after an incremental save leaves a partial on-disk ledger. Use `_require_ledger_save` on those incremental writes (still no need to publish incomplete hierarchies if freeze only sees the final return).
5. [D1 header L3-4] Still says nothing has been built. After this lock, one-line status update so the next window does not rebuild from a proposal.

OPTIONAL / NICE-TO-HAVE:
- Count the 135 claimed new cases (`r4 input` L35) rather than asserting the number.
- `banks.json` my_story extras `source_develop_verb` / `source_grounding_label` are not in D1 §1.1; harmless if the HUD already reads them.
- `GO_FORWARD_PLAN.md` still calls sprint 4 "NEXT"; stale vs the working tree.

CUT THESE:
1. Restoring D1's 6..40 / 1..3 as hard rejects -- operator no-length-chasing; code and pack already treat them as guidance.
2. My Story + `replay_from` support -- D1 §2.2 refuses it; `docs/GO_FORWARD_PLAN.md` scope cut is no replay-system work; `docs/MY_STORY_GUIDE.md` already says clear `replay_from`.
3. Sprint 5 App `extra.linearData`, mux UI envelope, live 1/3/6-act obs proof -- D1 §11; r4 input L8-9 already defers live publication.
4. A second draft writer, a third admission function, or a My Story-specific freeze policy table -- freeze already resolves `content_owned_readonly` from the missing `line_composer_system` seam (`nodes/_otr_freeze_cascade.py` + `tests/test_freeze_policy_readonly.py`).
5. Re-deriving the bank from original/sci-fi pass graphs -- D1 §9 settled; r4 input L4-6 forbids redesign.

VERIFY-AT-BUILD checklist:
- [r4 input L35] `pytest -q -p no:cacheprovider tests/test_my_story_*.py` and count cases; do not ship the "135" claim unverified. [ASSUMPTION] this review did not execute tests.
- [r4 input L35] full suite comparison vs the recorded baseline (plan: same 54 failures, zero new); Bug Bible relative path from the survival-guide repo.
- D1 §7.3 four wiring checks: `tests/test_widget_value_alignment.py`, `tests/test_canonical_widget_input_parity.py`, `tests/test_workflow_link_target_indexes.py`, `python scripts/build_variants.py --check` (soft-skip on uncommitted variants is expected). Grounded: node 1 `widgets_values` length 37; writer `script_json` links `[230, 291]`; mux inputs[10] `script_json` link 291; `last_link_id` 291.
- D1 §1.4: `my_story` absent from `eligible_bank_ids()` (`tests/test_my_story_registry.py` exists).
- D1 §7.1/§2.3: `rights_not_stamped` is informational (`nodes/_otr_publication_eligibility.py` `BLOCKING_REASONS` is only `rights_research_only`); freeze Phase 10 still stamps eligibility on the readonly path (`phase_10_gap_audit_post_and_freeze` remains in the readonly executed-tail). Live mux required-publish is sprint 5, not this lock.
- D1 §3 P4: announcer `char_id` c01 in cast, line sentinel `"announcer"` -- matches sci-fi (`nodes/_otr_scifi_news_pro.py` ~4338); canned freeze `_readonly_structural_validation` is asserted. Not a defect; confirm on first live freeze, not by changing ids.
- D1 §2.3 digest-after-style-roll: confirm the SHOULD-FIX test or explicitly accept the two split tests as the substitute.
- Earlier UNVERIFIABLE (App Mode frontend 1.49.6, `extra.linearData`) stays sprint 5; do not treat this r4 as Gate 1 / live-obs PASS.
