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

---

# v2 EXECUTION PLAN (QA-revised + code-verified, 2026-05-29)

A reviewer QA'd the v1 plan against the JSON. Most points were right; two were
speculative (reviewer hadn't seen the node code) and I verified them. This v2 section
SUPERSEDES the v1 "Removal protocol" + "Sequencing" above.

## QA reconciliation (verified against the code, not just the JSON)
- **Validator (QA risk #5) -- mostly UNFOUNDED, verified.** `OTR_WorkflowValidator`
  (node 63) -> `nodes/_workflow_validation.py` checks NO node counts / required types /
  topology. Only Check 6 (link table) bites: no orphan links, no dup link ids,
  `last_link_id == max(link_id)`, no reserved ids {111,112}. So keep the link table
  consistent and NO validator edit is needed. Do NOT add the removed types to
  `DELETED_NODE_TYPES` (that deny-list is only for tombstoning types other workflows
  might still reference).
- **Story Room "dormant" wording (QA #2) -- CORRECTED.** The nodes DO execute at runtime
  (live logs showed DirectorBrief -> StoryRoom -> Extract running every episode);
  `commit=False` only skips Commit's WRITE. So the ledger/audio is byte-identical (Commit
  pass-through, confirmed in code), but removal also reclaims ~2 min/run and kills the
  Extract under-produce crash surface. "Out of the output," not "inert."
- **Widget-shift (QA #1) -- real but bounded.** The 4 removable writer widgets are all
  OPTIONAL (indices 15/17/18/22); the validator's widget-drift check walks only the 3
  REQUIRED widgets, so it won't catch a bad optional array -- but ComfyUI still maps
  `widgets_values` positionally at load. So widget removal MUST migrate the array
  (remove highest index first: 22 -> 18 -> 17 -> 15) in lockstep with the INPUT_TYPES /
  run() / _resolve_inputs edits. `test_workflow_canonical_baseline.py` only checks
  widgets[0..4], so it stays green.
- **Link bookkeeping (QA #3/#4) -- adopted exactly.** Deleting a link must scrub it from
  the global `links` array AND every surviving node's `outputs[].links` / `inputs[].link`,
  and a new link needs a fresh id + `last_link_id` bump. Do this PROGRAMMATICALLY (a
  migration script), not by hand.
- **Tier 0 (QA #4 traps) -- re-labelled** "zero GIT risk, NOT zero operational risk":
  preserve the most recent `io/` + `outputs/` run evidence for regression compare; only
  blanket-delete the old `_*.txt`/`__pycache__` scratch.
- **Polish (QA #3) -- re-framed** as hot-path refactor of the live `_otr_line_composer`,
  not cleanup. Highest regression exposure of the four; do it on its own with full audio
  regression.
- **Tier 3 (QA #5) -- hard gate:** scan EVERY workflow JSON + test fixture for a node type
  before deleting it (esp. visual/*). No delete on single-workflow absence alone.

## Verified surgery facts (from otr_scifi_16gb_full.json)
- `last_link_id=229`, `last_node_id=79`. Writer (1) `script_json`=output slot 1, links=[107].
- Story Room links = {107,218,219,220,221,222,223,224,225,226} (confirmed). Island links =
  {227,228,229} (nodes 79->78 only; node 79 inputs all null).
- Rewire = new link **[230, 1, 1, 62, 1, "STRING"]** (writer script_json out1 ->
  FreezeCascade in slot 1, replacing the old 225 from Commit).
- Node-array deltas: writer out[1].links [107]->[230]; out[4].links [221]->[] ; out[5].links
  [115,218,219,222,224]->[115]. FreezeCascade in[1].link 225->230. `last_link_id`->230.
  Result: 81->69 links, 38->31 nodes, max link id 230 == last_link_id (validator-clean).
- FLAG (not a blocker): after removal the writer's `creative_writing_model` broadcast
  output (slot 4) has zero consumers (Story Room was its only one); the writer still uses
  the creative model internally. Decide whether to leave the dangling output or prune it.
- MINOR: the writer's `if __name__=="__main__"` self-test has a stale `assert n_optional==15`
  (already 20 today; not in the pytest gate). If we drop 4 optional widgets, update it.

## SAFE SEQUENCE (do in this order; each its own commit + your restart to validate)
1. **Commit 1 -- JSON-ONLY graph migration** (no code touched): programmatically remove
   nodes 73,74,75,76,77,78,79; remove links {107,218..226,227,228,229}; add link 230
   (writer->FreezeCascade); scrub node arrays; set last_link_id=230. Validate
   programmatically: JSON parses, no orphan links, no dup ids, last_link_id==max, 31 nodes.
   YOU restart ComfyUI + confirm node 1 (writer) loads and the graph runs writer->freeze.
   The node classes stay registered (harmless, just unused) until step 3.
2. **Commit 2 -- multiturn removal** (lowest-risk code): delete the 3 Wave-0 modules + the
   dispatch block + the widget (array-migrate index 18) + tests. Regression.
3. **Commit 3 -- Story Room code removal:** now that the JSON no longer references them,
   delete the 6 Story-Room module/wrapper files + the 5 `_NODE_MODULES` entries + tests
   (delete the story-room tests; EDIT the constraint-editor tests to drop director/editor
   fixture imports -- the constraint-editor feature survives in the writer). Regression.
4. **Commit 4 -- shadow + fan-out removal** (together): strip both writer blocks + 2 nodes +
   helper modules + the 2 `_NODE_MODULES` entries; widgets array-migrate indices 22 then 17.
   Regression.
5. **Commit 5 -- polish removal** (hot-path; do last + carefully): excise the polish path
   from `_otr_line_composer` + `_otr_model_loader` + `_otr_reroll` + the widget (index 15).
   Full audio byte-identity regression.
6. **Commit 6 -- widget cleanup confirm + stale `__main__` assert fix + Two-Model dangling-
   output decision.** Final JSON re-save; full regression; your restart.
7. T1 (untracked git-status cruft) + T3/T4 after a repo-wide reference scan.

---

# v3 EXECUTION PLAN (round-2 QA + re-verified, 2026-05-29) -- THE LIVE PLAN

Round-2 reviewer QA, reconciled against the actual JSON + node code. This SUPERSEDES the
v2 sequencing. Two reviewer finds were real (Node 42 dead, the output-slot trap); one was
correctly cautious and verification CONFIRMED it must stay (Node 21).

## Already DONE -- verified on the real file (via Desktop Commander, not the sandbox)
Commit 1 (987742b) migrated the graph. Current state CONFIRMED: 31 nodes, 69 links,
last_link_id=230==max, 0 duplicate ids, 0 orphan links; Story Room nodes 73-79 absent;
writer.script_json -> link 230 -> FreezeCascade(62).script_json. So there is NOTHING left
to do for the Story Room *graph*; the v1/v2 "delete nodes 73-79 + add link 230" steps are
a NO-OP now. Any future JSON script MUST be idempotent (remove_if_present) and MUST run
through Desktop Commander -- the Linux/bash mount serves a CORRUPTED copy of this JSON
(trailing NUL/space padding; known stale-VM-mount issue). Edit/validate the real file only.

## Verified corrections to the round-2 QA
- **Node 42 `PathchSageAttentionKJ` -- DEAD, delete (NEW, verified).** Title "Patch Sage
  Attention (FLUX) -- DISABLED, BUG-LOCAL-070", widget="disabled" => pure MODEL
  pass-through; it's a KJNodes EXTERNAL node (removing it also drops that dep; it is not
  OTR_-prefixed so the validator never required it). Exact edits (link ids verified on the
  real file): delete link 203 `[203,71,0,42,0,"MODEL"]`; mutate link 69
  `[69,42,0,23,0,"MODEL"]` -> `[69,71,0,23,0,"MODEL"]`; node 71 outputs[0].links
  `[203,204]` -> `[69,204]`; node 23 inputs[0].link stays 69; delete the node-42 object;
  last_link_id stays 230 (max unchanged). Validator-safe.
- **Node 21 `OTR_FixedShotDurationStub` -- KEEP, do NOT delete (verified REQUIRED).** Despite
  "Stub" in the name, `otr_shot_duration_calculator.py` does a real transform
  (`expand_plan_with_durations`: clips-per-shot, global frame renumber + FLF boundary
  sharing, token regen) that `OTR_BatchFluxRender` consumes via link 41. Bypassing feeds
  FLUX an un-expanded plan = visual regression, and `tests/test_fixed_shot_duration_stub_rename.py`
  hard-pins node 21's type + links 40/41. ACTION: keep. OPTIONAL later: rename away from
  "Stub" (e.g. OTR_ShotDurationExpander) -- but that's a PD3 lockstep change (class +
  __init__ + JSON type + S&R name + that test), not a deletion. Not dead weight.
- **Prune writer `creative_writing_model` output -- exact, with the slot trap.** Verified:
  writer outputs slot 4=creative_writing_model (links []), slot 5=technical_model
  (links [115]); the ONLY link off slot 5 is 115=`[115,1,5,62,4,"STRING"]`, ZERO off slot 4.
  Edits: RETURN_TYPES drop one "STRING" (6->5); RETURN_NAMES drop creative_writing_model;
  run() return tuple drop resolved["creative_writing_model"]; JSON remove the slot-4 output
  object, renumber technical_model slot_index 5->4; **mutate link 115 src_slot 5->4 ->
  `[115,1,4,62,4,"STRING"]`** (THE trap); last_link_id unaffected. MUST update in lockstep:
  the writer `__main__` self-test (asserts the 6-tuple), and
  `tests/test_workflow_json_guardrails.py` TestWriterB2aSurface::test_writer_output_slot_indexes_stable
  + TestCascadeB3Surface::test_cascade_technical_socket_wired_in_canonical_json (assert
  outputs[4/5] names + src_slot==5 + len>=6). The widget-surface tests stay green (output
  prune != widget change).
- **Widget removal -- assert-guarded, highest-index-first.** For the 4 optional flags at
  widgets_values [15,17,18,22] (all currently False), the migration script MUST
  `assert len(widgets)==23 and widgets[15] is False and widgets[17] is False and
  widgets[18] is False and widgets[22] is False`, then pop indices in order 22,18,17,15 --
  in the SAME commit as the matching INPUT_TYPES / run() / _resolve_inputs edits. Never pop
  blind.

## THE TWO CRASH POINTS (everything else is manageable)
1. **Writer widget index drift** (removing optional widgets from INPUT_TYPES without
   migrating widgets_values, or wrong index/order).
2. **Output slot-index drift** (removing the creative_writing_model output without
   decrementing technical_model AND fixing link 115 src_slot 5->4).
After EVERY JSON change: re-assert no orphan links, no dup ids, last_link_id==max.

## v3 commit order (one concern each; regression + your restart per commit)
- **A. Graph validation (already satisfied):** no-op -- the migration landed; just confirm
  the invariants above before starting B. Done.
- **B. Delete Node 42** (disabled Sage patch) + bridge link 69 node71->node23; regression
  the FLUX render path.
- **C. Story Room CODE removal:** the graph no longer references it, so delete the 6
  module/wrapper files + the 5 `_NODE_MODULES` entries + Story-Room tests; EDIT the
  constraint-editor tests to drop director/editor fixture imports; keep `_otr_craft_floor`
  + `_otr_editor_constraints` (writer-owned).
- **D. Multiturn removal:** 3 Wave-0 modules + dispatch block + widget (pop index 18) +
  kwargs + resolved-dict key + tests.
- **E. Shadow + fan-out removal (together):** writer blocks + OTR_Stage1FanOut/OTR_BeatSelector
  + helpers + 2 `_NODE_MODULES` entries + widgets (pop 22 then 17) + tests; verify
  `_otr_name_gender` has no other caller.
- **F. Polish removal (last, riskiest -- hot-path):** excise from `_otr_line_composer` +
  `_otr_model_loader` + `_otr_reroll` + widget (pop 15) + polish tests; FULL audio
  byte-identity regression.
- **G. Prune writer creative_writing_model output:** the exact edits above (incl. link 115
  src_slot 5->4) + update the 3 pinning tests + the writer self-test.
- **Node 21:** KEEP. Separately, optionally rename (PD3 lockstep) -- not a deletion.
- **T1/T3/T4 cruft:** after a repo-wide reference scan (every workflow JSON + test fixture).

Net (beyond Commit 1): -1 more graph node (42), ~12-16 modules + the writer output socket
removed, widgets_values 23->19, one fewer external dep (KJNodes Sage patch).
