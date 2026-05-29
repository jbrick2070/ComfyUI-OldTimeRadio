# OTR Lean-Down Audit -- 2026-05-29

Deep-dive (3 parallel subagents) to find unused / unhelpful code so we ship a leaner,
meaner workflow. **Inventory + teardown spec -- nothing removed yet.** This is the
execution spec for v11 Card 2 (see GO_FORWARD_PLAN_v11). Active dialogue path = the
in-writer `use_exchange` grouped exchange; everything below is measured against that.

## MUST NOT REMOVE (load-bearing for the active path -- verified)
- `nodes/_otr_compose_exchange.py` -- the chosen dialogue engine (writer imports at :3568).
- `nodes/_otr_craft_floor.py` -- Tier-A gate; imported by the LIVE writer at :3572 (the
  use_exchange prepass), independent of the Story Room. Survives the Story Room removal.
- `nodes/_otr_editor_constraints.py` -- imported by the live writer at :3011; its only real
  dep is `_otr_beat_validators` (the director_brief/editor_pass refs are comments only).
- Shared libs: `_otr_stage1_plan`, `_otr_constrained_generate`, `_otr_legacy_to_stage1_adapter`,
  `_otr_whole_episode_critic`, `news_interpreter`, `production_ledger`, `story_orchestrator`.
- `workflows/otr_scifi_16gb_full.json`, `workflows/GO_FORWARD_PLAN_v11...md` (tracked),
  `requirements*.txt`, `config/episode_cast.txt`, `tests/fixtures/*`.

## Removal protocol (every code/JSON tier)
1. One concern per commit. 2. Bug Bible + core + audio + the affected suites green after each.
3. Re-save the workflow JSON in the SAME commit (it stores writer flags positionally; removing
   a widget shifts indices). 4. Keep the forbidden-pattern sweep + guardrail tests green
   (`test_legacy_audit_clean.py`, `test_workflow_json_guardrails.py`). 5. Jeffrey restarts
   ComfyUI to live-validate the rewired graph before the tier is "done".

---

## TIER 0 -- zero-risk disk cleanup (already .gitignored; pure disk, no git impact)
Safe to delete anytime; does not touch git. ~87 root `_*.txt` / `pytest_*.txt` / `test_out.txt`
/ `QA_*.txt`, root `*.log` (otr_runtime.log, w1.log, _b290_full.log), all `__pycache__/`
(14 dirs, ~829 `.pyc`), `io/` (13 MB sidecar run dirs), `outputs/` scratch, `scripts/_*.bat`,
`scripts/*.log`. Confidence: HIGH. (Mostly cosmetic -- they never reach git.)

## TIER 1 -- git-status cleanup (untracked, not ignored -- removing clears `git status` noise)
- `docs/sprint_drafts/build1..build4/` -- the 3 build .py drafts were PROMOTED into `nodes/`
  + `tests/` this session (confirmed: craft_floor, slot_drama_contract, compose_exchange all
  live + tracked). Drafts are redundant. (WIRING_SPEC.md / FINDINGS.md are design history --
  keep if wanted; tiny.) Confidence: HIGH.
- `workflows/GO_FORWARD_PLAN_v7/v8/v9_*.md` -- explicitly "superseded by v10" (v10 header).
  Untracked. Confidence: HIGH. `v10` -- untracked, historical as-built record; keep for
  provenance or drop (MED).
- Loose untracked problem-statement docs in `docs/` (2026-05-25.., 2026-05-26..__00_question,
  2026-05-28-better-story-problem-statement). MED -- keep `story-generator-final-plan.md` if
  it's still the live spec.
- Optional: add the above patterns to `.gitignore` instead of deleting, to keep the history.

---

## TIER 2 -- dormant code removal (the real lean-down; do in THIS order)
All four features default OFF and are OFF in the production workflow. None is the chosen path.

### 2.1 Multiturn dialogue (`use_multiturn_dialogue`) -- FIRST CUT, lowest risk
- Superseded by `use_exchange` (compose_exchange's own docstring says it mirrors this older
  Wave-0 path). Self-contained.
- Delete: `nodes/_otr_wave0_multiturn.py`, `nodes/_otr_stage2_call.py`, `nodes/_otr_stage2_prompt.py`
  (nothing else imports them). Strip the `_OTRW0MT` dispatch block in OTR_LedgerScriptWriter
  (~:2044 / :2133) + the `use_multiturn_dialogue` widget + kwarg (8 sites) + the resolved-dict key.
- Tests: delete `test_wave0_multiturn_dispatch.py`, `test_stage2_multiturn.py`; edit
  `test_stage1_shadow_pass_integration.py`, `test_otr_api_companions.py`,
  `test_workflow_json_guardrails.py`. Re-save workflow JSON (widget index shift).
- Confidence: HIGH / risk LOW.

### 2.2 Story Room cluster -- the headline removal (clean; one graph rewire)
Dormant (`commit=False`); Commit is a pure pass-through, so removal is byte-identical (PD1).
- Delete node wrappers + impl: `OTR_StoryRoom.py`+`_otr_story_room.py`,
  `OTR_StoryRoomExtract.py`+`_otr_story_room_extract.py`, `OTR_StoryRoomCommit.py`;
  feeders (Story-Room-only): `OTR_DirectorBrief.py`+`_otr_director_brief.py`,
  `OTR_EditorPass.py`+`_otr_editor_pass.py`, `_otr_writers_room_resolver.py`.
- `__init__.py` `_NODE_MODULES`: remove the 5 entries (StoryRoom @317, Extract @328,
  Commit @338, DirectorBrief @290, EditorPass @308).
- **Workflow JSON graph rewire (the only surgery):** delete nodes id 73 (EditorPass),
  74 (DirectorBrief), 75 (StoryRoom), 76 (Extract), 77 (Commit) and links 107, 218, 219,
  220, 221, 222, 223, 224, 225, 226; then add ONE link: writer (id 1) slot 1 `script_json`
  -> FreezeCascade (id 62) slot 1 `script_json`. (Today the chain is writer --107--> Commit
  --225--> FreezeCascade; collapse to one direct link.)
- Also detach the dormant island nodes id 78 (`OTR_BeatSelector`) + 79 (`OTR_Stage1FanOut`)
  -- see 2.4 (remove with fan-out).
- Tests: delete `test_otr_story_room*.py`, `test_storyroom_commit_failloud_c.py`,
  `test_otr_director_brief.py`, `test_otr_editor_pass.py`, `test_writers_room_resolver.py`,
  `test_bug_local_291/292/293*.py`, `test_dialogue_slot_id.py`, `test_workflow_director_freedom.py`.
  EDIT (do not delete): `test_legacy_audit_clean.py` (EXCLUDED_PATHS pins go stale),
  `test_constraint_editor_*`/`test_constraint_repair_*`/`test_writer_constraint_repair_splice.py`
  (drop the `_otr_director_brief`/`_otr_editor_pass` fixture imports -- the constraint-editor
  feature itself SURVIVES in the writer). Fix `scripts/smoke_fanout_constraint.py`.
- KEEP: `_otr_craft_floor.py`, `_otr_editor_constraints.py` (writer-owned; see MUST-NOT-REMOVE).
- Confidence: HIGH / risk LOW (only the one link rewire is delicate -> restart-validate).

### 2.3 Polish pass (`enable_polish_pass`) -- in-file surgery, MED risk
Redundant with the live Script Doctor + the now-ON Stage-3 validators (freeze cascade says so).
NOT a standalone module -- it lives inside the LIVE `_otr_line_composer.py`:
remove `needs_polish`/`polish_line`/`is_polish_refusal` + the `_POLISH_*` constants + the
`enable_polish_pass` branch in `compose_line` (~:2061) + the `__all__` export; remove
`make_polish_generate_fn` from `_otr_model_loader.py`; drop the flag in `_otr_reroll.py:573`
and the writer's widget + 3 call sites. Tests: delete `test_polish_speaker_prompts_locked.py`,
`test_lfc_w4_writer_polish_fn.py`, `test_lfc_polish_fixes.py`; edit others that pass the flag.
Confidence: HIGH (dormant) / risk MED (cuts inside a hot-path live module -> heavy regression).
(Already disabled in the workflow JSON this session.)

### 2.4 Stage-1 shadow pass + fan-out (`enable_stage1_shadow_pass` + `use_stage1_fanout`) -- remove together
Both are diagnostics for a Sprint-4.6 swap that never landed; fan-out hard-requires shadow ON.
- Strip the shadow block (~writer :2469-2720) and the fan-out block (~:2526-2620) + both
  widgets/kwargs. Delete nodes `OTR_Stage1FanOut.py`, `OTR_BeatSelector.py` + helpers
  `_otr_stage1_fanout.py`, `_otr_beat_selector.py`; and (freed once both are gone)
  `_otr_stage1_call.py`, `_otr_stage1_cast_audit.py`. Verify `_otr_name_gender` has no other
  caller before removing. Remove the 2 `_NODE_MODULES` entries (@348/@357) + the 2 JSON
  island nodes (id 78/79). ~10 test files reference these -> delete/edit. Re-save JSON.
- Confidence: HIGH (dormant) / risk MED (registry + JSON + many tests).

## TIER 3 -- registered-but-absent-from-workflow nodes (verify against OTHER workflows first)
Not in `otr_scifi_16gb_full.json`. Some may be used by alt workflows -> confirm before delete.
- HIGH/LOW: `OTR_BisectStringSource` (self-labeled TEMPORARY, delete-at-BUG-231-close).
- MED: `OTR_VisualBridge/VisualPoll/VisualRenderer/VisualPromptCoercion/VisualExtractFluxPrompt`
  (visual sidecar -- may belong to a different workflow), `OTR_CheckpointLoaderGated`
  (superseded by DeferredCheckpointLoader), `OTR_VideoConcat` (superseded by VideoComposite),
  `OTR_BatchProceduralSFX`, `OTR_ProjectStateLoader`, `OTR_VRAMGuardian`, `OTR_VRAMContextTest`,
  `OTR_SaveCopy`. Each: confirm no other workflow/doc references it, then drop registry +
  module + tests.

## TIER 4 -- tracked cruft (deliberate commits; Jeffrey's call)
- `docs/s28_diff_tmp.txt` (tracked temp), the 34 `docs/2026-05-13-S26..S29-*.txt` baseline dumps
  (superseded sprint audit history). Removing = a real commit.
- Workflow JSONs (tracked): `humo_smoke_*.json` (4), `_bisect_flux_*.json` (5 one-off VRAM
  debug rigs), `external_examples/*` (3 third-party refs). Keep only if still referenced for
  HuMo/Flux testing; otherwise remove deliberately.

## Sequencing
T0/T1 anytime (disk + git-status). Then T2 in order: multiturn -> Story Room -> polish ->
shadow+fanout, each its own commit + regression + JSON re-save + your restart. T3/T4 last,
after confirming no alt-workflow dependence. Net effect: ~12-16 node modules + ~6 dormant
helper modules + ~20 tests removed, one workflow graph simplified to writer -> freeze direct.
