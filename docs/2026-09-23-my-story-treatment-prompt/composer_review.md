VERDICT: yes-with-fixes — Scoped work is a single `my_story_treatment_system` string edit with no workflow/code changes; wiring matches `nodes/_otr_my_story.py` and `nodes/story_packs/pipelines.json`, but `input.md` still contradicts itself (HOLD vs MUST-FIX none) and its CONFIRMED block is stale against the current pack file.

MUST-FIX BEFORE BUILD:
1. [input.md / VERDICT L3 vs MUST-FIX L31] Top line says "HOLD for a narrow prompt clarification" while the body specifies the full Change text and declares "MUST-FIX: none." Pick one build gate: either the clarification is already in [Change] (then VERDICT should be yes/yes-with-fixes, not HOLD) or list the missing clarification as a numbered MUST-FIX with exact wording to add.
2. [CONFIRMED L6-8] Claims "only dramatic_question has explicit field instructions" and that logline lacks field rules. Current shipped pack already defines logline, dramatic_question, and "After completing each field value, close its JSON string..." in `nodes/story_packs/my_story/my_story.json` (prompt_stages.my_story_treatment_system). Refresh CONFIRMED to post-change facts or label that block "pre-diff baseline" so R4 QA does not refute a already-landed tree.
3. [GO_FORWARD_PLAN §2 L232-235 + Change L25-29] "Verify prompt delivery and regressions" has no checkable procedure in `input.md`. Add one concrete step, e.g. assert `RT.resolve_story_pack("my_story").prompt_stages["my_story_treatment_system"]` contains the three new clauses (declarative logline, yes/no dramatic_question only in its field, close string / do not repeat), then run scoped pytest: `tests/test_my_story_registry.py`, `tests/test_my_story_runner.py`, `tests/test_story_pack_stage1.py::test_the_packs_are_still_valid_json_with_prompt_stages`.

SHOULD-FIX:
1. [Change L28-29 / GO_FORWARD §2] Unlike `scifi_news_pro`, there is no `tests/test_*_prompt_snapshots.py` for `my_story_treatment_system` (grep tests: only scifi snapshots). A one-key substring snapshot would prevent silent prompt drift on the next pack edit.
2. [UNVERIFIABLE L17-23] Correctly rejects loop causality and new PBUG; add an explicit non-goal in [Review scope]: improved P1 token economics or repetition rate is not a merge gate (only schema-valid treatment + existing ladder behavior).
3. [input.md meta] Document is a driver anchor + change receipt, not a sectioned build plan; add stable section IDs (CONFIRMED, UNVERIFIABLE, Change, Review scope) in headings so R4/R judge citations stay unambiguous.

OPTIONAL / NICE-TO-HAVE:
- Cross-link [UNVERIFIABLE] to `docs/PROD_BUG_LOG.md` PBUG-20260910-05 / Bible 12.100 with one line on what this prompt change does *not* claim to fix (binder/grammar termination), to stop builders reopening sampling or decode guards.

CUT THESE:
1. [input.md L3] "loop causality is unproven" in the one-line VERDICT — already fully covered in [UNVERIFIABLE L17-23]; duplicating it in VERDICT reads as a build block.
2. [Review scope L38-39] Second "do not mint PBUG" sentence — merge with [UNVERIFIABLE L22-23].

VERIFY-AT-BUILD checklist:
1. [UNVERIFIABLE L17-20] Qwen3.8-27B 69-token clause / 288 halt / 0.500 recovery — not reproducible from repo logs; live My Story P1 on operator profile; inspect `meta["my_story"]` treatment artifact: `logline` declarative, `dramatic_question` ends with `?`, no duplicated question clause inside `logline` string.
2. [CONFIRMED L5-7] `_pass_treatment` uses `_seam(pack, "my_story_treatment_system")` at `nodes/_otr_my_story.py` ~597; pack loaded via `resolve_story_pack` in `nodes/_otr_story_routing.py` ~793-806 — confirm runtime system message matches on-disk pack after deploy/restart.
3. [CONFIRMED L13-15] `_full_artifact_repair` at `nodes/_otr_my_story.py` ~475-510 copies `original_prompt` (includes system) into repair messages — run or unit-test an interrupted P1 path with `error.raw_completion` set; repair prompt must still lead with original system seam.
4. [Change L28-29] No new validator/sampling/retry — confirm `StoryTreatment` in `nodes/_otr_my_story.py` ~240-248 unchanged and `_make_treatment_validator` ~558-577 still only cast/act rules.
5. verify: git status of `nodes/story_packs/my_story/my_story.json` vs intended pushed diff (shell blocked in this review pass).
6. verify: first production run publishes through existing path (`obs_publish OK`) with no workflow edit to `workflows/otr_canonical.json` per [CONFIRMED L10-11].

Holistic: End-to-end consistency with [GO_FORWARD_PLAN §2] goal (clarify logline vs dramatic_question in pack only) holds; [ASSUMPTION] working tree already contains the proposed prompt text, so the remaining gap is receipt/document convergence and explicit verify steps, not design. Lean enough to hand to a builder once MUST-FIX 1-3 are patched in `input.md` / plan row.
