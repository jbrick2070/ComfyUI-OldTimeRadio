VERDICT: yes-with-fixes. A1 code matches the flexible-cast / binding-acts contract except one fix-introduced repair-instruction contradiction that can reproduce the live one-act lighthouse failure.

MUST-FIX BEFORE BUILD:
1. [GO_FORWARD A1.1 + D1 §3 P1 + nodes/_otr_my_story.py:_pass_treatment repair factory ~491-495] Defect: the typed treatment repair says both "Reorganize the treatment into exactly N acts" and "Do not change the selected acts." The second clause can be read as "keep the act list you already wrote," which is the exact 2-for-1 failure in GO_FORWARD "Grounded state" (prompt e11e5d86, `_make_treatment_validator` rejected count, no freeze). tests/test_my_story_runner.py::test_full_treatment_repair_preserves_material_and_matches_variable_controls cannot catch this: CountRepair returns the correct N on the repair rung regardless of the instruction. Concrete fix: drop "Do not change the selected acts." Replace with one sentence: keep people, events, relationships, and ending; change only act topology to the selected N.

SHOULD-FIX:
1. [D1 §2.1 vs D1 §5 / GO_FORWARD Operator contract + A1.1] Defect: §2.1 still says `num_characters` (1) and `act_count` (6) "keep their meaning." After the flexible-cast amendment that is false for characters (request/guidance vs binding). A later edit can re-bind count. Concrete fix: one clause in §2.1: act_count stays binding; `num_characters` is requested guidance, announcer excluded, actual follows the story.
2. [GO_FORWARD "Next action" ~96-97] Defect: cites `2026-09-10-my-story-next/` and `2026-09-10-my-story-a1/operator_amendment.md`. Neither path exists in this tree (glob zero). The live contract is already in GO_FORWARD Sprint A. Concrete fix: delete those two pointers or point at the files that actually hold the amendment (GO_FORWARD A1 + D1 §5).
3. [GO_FORWARD A1.5 + D1 §5 vs nodes/_otr_my_story.py:900-931 vs 1087-1092] Defect: `meta.my_story.counts.requested_characters` uses clamped `resolved["num_characters"]`; `cast_contract.num_characters_request` uses `source_meta["requested_num_characters"]` (StoryRequest, pre-clamp in nodes/_otr_writer_inputs.py:256-257). Widget 1..10 usually matches; an out-of-range API value splits the receipt. Concrete fix: both fields read the same raw StoryRequest integer; keep the clamp only as the prompt "requested" hint.
4. [driver_anchor "six combos" + tests/test_my_story_runner.py] Defect: requested!=actual is proven only through `_run()` (runner + readonly freeze helper). `test_real_writer_routes_user_fields_to_a_clean_ledger_and_shared_tail` uses num_characters=2 matching Ada/Tom and stubs `_run_writer_tail`. Dispatched path in OTR_LedgerScriptWriter.py:3642-3682 skips the inline lock_cast count gate at 4248, so this is likely safe, but first production is writer+tail+freeze 62. Concrete fix: one writer.run combo with requested!=actual that does not stub the tail, then freeze `_readonly_structural_validation`.
5. [D1 §3 P0 interpret user prompt ~429-433] Defect: still injects "distinct voices available: N" as if interpret should size the cast to stock. P4 already fails loud on exhaustion (nodes/_otr_my_story.py:652-658). Concrete fix: keep capacity as a receipt/log; remove it from the interpret user prompt so interpret cannot "plan down" the listener's people.

OPTIONAL / NICE-TO-HAVE:
- Exclusive named drop stays a `fidelity_discrepancies` receipt (D1 P1, GO_FORWARD A1.2). That matches "no new gates." Do not add an exclusive hard gate in this chunk.
- Writer `num_characters` tooltip already says request-not-cap (OTR_LedgerScriptWriter.py:2071-2077). Align D1 §2.1 with that, do not change the widget.

CUT THESE:
1. [nodes/_otr_my_story.py:494] "Do not change the selected acts." Safe: selected N is already in the first repair sentence and in `_make_treatment_validator`. This is the same edit as MUST-FIX 1.
2. [interpret user prompt voice_capacity line] Safe: pool exhaustion remains a P4 Python error; interpret planning is already non-rejecting (D1 P0).
3. [input.md / this QA] Any A2 HF-capacity, Sprint B credit, C layout, or A3 other-bank work. Safe: input.md and GO_FORWARD A2/A3/B/C already park them.

VERIFY-AT-BUILD checklist:
1. input.md "Focused196passed" / "same54 failures as S4" -- confirm focused count and that full/Bible deltas equal the known S4 baseline; do not claim all-green. [ASSUMPTION] 196/54 not re-run in this review.
2. Canonical 23 nodes / 63 links / 37 writer widgets / last_link_id 291 / writer script_json links [230, 291]. Grounded: last_link_id 291 and 37 widgets_values on node 1 in workflows/otr_canonical.json. verify: node count 23 and links-array length 63 (not counted here). Confirm this A1 diff still does not touch that file.
3. Live component: exact lighthouse one-act input, then 3 and 6, freeze node 62, requested!=actual recorded. GO_FORWARD A1 tests. No fixture PASS.
4. News label: writer output news_used source == banks.json `source_material_label` "Listener story idea" on a my_story_fields run; never "RSS Auto-Fetch". Unit exists (tests/test_scifi_news_pro_tail_context.py:506-524); prove on the live ledger/wire.
5. Cleanup title: canon file + script_json meta.episode_title + news headline stay identical after ledger_cleanup fills a blank outline title. Unit exists; prove on a live empty-title My Story.
6. No My Story RSS/snapshot fetch: fetcher "" in nodes/story_packs/banks.json:171; `_user_fields` skips `load_snapshot_for_bank` in nodes/_otr_writer_inputs.py:330-334. verify: live log has no RSS_FETCH / snapshot REPLAY on my_story.
7. Python 3.10 `add_note` absence: guarded (nodes/_otr_my_story.py:1140-1143) and log.error still fires. This box is 3.12. verify: if any runner is 3.10, primary provider/cancel error still propagates when history save fails.
8. Voice stock: actual distinct presets; exhaustion names the character (test_actual_voice_allocation_exhaustion_names_the_character). verify: a live cast larger than remaining open_voice_pool fails at P4, not at freeze.
9. HEAD vs origin after the push; input.md named diff-vs-703a355. verify: git diff --stat against that commit contains only the A1 files (runner/pack/tests/docs/tail), not workflows/otr_canonical.json.

[ASSUMPTION] Working tree matches the snapshot (modified: nodes/_otr_my_story.py, nodes/_otr_writer_tail.py, pack, tests, MY_STORY_GUIDE, GO_FORWARD_*, D1). This review did not execute git diff vs 703a355 or the suite.
