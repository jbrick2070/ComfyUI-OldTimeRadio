# OTR Lean-Down -- Go-Forward Plan

Goal: strip the pipeline to one clean path -- writer (use_exchange) -> freeze cascade ->
audio -> video -- removing dead/dormant machinery that does not serve the story. Audio is
king; nothing load-bearing is touched. Every call below is grounded in a verification pass
over the real files (link table, writer INPUT_TYPES via ast, importer graph, all workflow
JSONs, VRAM-primitive definition sites). The five big traps: master-link desync, writer
widget transposition, the slot 4/5 race, frontend-cache poisoning, and deleting VRAM-survival
logic by accident. Each has an explicit guard below.

## Current state (verified)
otr_scifi_16gb_full.json: 31 nodes, 69 links, last_link_id 230 (== max, no orphan/dup),
writer.script_json -> link 230 -> FreezeCascade. use_exchange ON; multiturn / shadow / fan-out
/ polish all OFF. All 10 workflow JSONs scanned: zero references to any node type slated for
deletion -- tombstoning deleted types is therefore safe. Remaining work: code removal + two
graph cuts + a conservative model-loader audit.

## Safety gate -- run before AND after every graph edit
The workflow JSON is TWO synchronized routing systems: each node's local `inputs[].link` /
`outputs[].links`, and the master `links` array. If either side drifts, the file can load
with ghost routes or silently wrong wiring. All edits go through Desktop Commander on the real
file; the bash/VM mount serves a corrupted copy. Run the link-table validator on every edit:
- JSON parses
- every node-local link id exists in the master `links` array
- every master link points to a real source AND target node (no orphans)
- no duplicate link ids
- last_link_id == max(link id)
- no reserved link ids (111, 112)
- no stale output-link ids left on surviving nodes

The worst failure is silent corruption, not a crash. The two methods below -- name-keyed
widget regen and full link mutation -- make that class of failure impossible, not just rare.

## Operator discipline -- backend + frontend cache (every writer-surface change)
ComfyUI caches node definitions in the running backend AND in the browser. Any change to the
writer's INPUT_TYPES, RETURN_TYPES, or widgets (steps 2, 3, 4, 6, 7) must follow this order or
the frontend will force a stale node definition over the new JSON and corrupt widgets_values
on save:
1. Edit Python first.
2. Stop ComfyUI.
3. Clear Python __pycache__.
4. Restart the ComfyUI backend.
5. Hard-refresh the browser (Ctrl+F5); clear ComfyUI local storage for the tab if needed.
6. THEN load the mutated workflow JSON.
7. THEN save.
8. THEN run the link-table validator.
Slot 4/5 race (step 7 specifically): never open the mutated JSON while the backend still
believes output slot 4 is creative_writing_model. Restart before loading.

## Keep -- load-bearing, never remove
- nodes/_otr_compose_exchange.py (dialogue engine), _otr_craft_floor.py (Tier-A gate),
  _otr_slot_drama_contract.py, _otr_editor_constraints.py + _otr_beat_validators.py.
  NOTE: _otr_editor_constraints.py is LIVE writer-owned constraint logic -- it is NOT Story
  Room code. Do not delete it when removing the Story Room cluster (adjacent, easy to confuse).
- Shared libs: _otr_stage1_plan (the LIVE outline generator -- see warning below),
  _otr_constrained_generate, _otr_legacy_to_stage1_adapter, _otr_whole_episode_critic,
  news_interpreter, production_ledger, story_orchestrator, _otr_model_loader, _vram_log,
  _otr_line_composer (polish carved out, file stays), _otr_reroll.
- VRAM operator tools: nodes/vram_guardian.py (OTR_VRAMGuardian) + nodes/vram_context_test.py
  (OTR_VRAMContextTest). Verified NOT story machinery and not in any workflow, BUT they are
  purpose-built manual VRAM-flush / context probes for the 16GB FLUX->HuMo handoff, cost one
  registry line each, and OTR_VRAMGuardian is pinned by tests/test_core.py. Keep them -- the
  real VRAM primitives (_flush_vram_keep_llm, request_slot, force_vram_offload) live in
  story_orchestrator / _otr_model_loader / _vram_log, so these nodes are a cheap safety net,
  not dead weight.
- Node 21 OTR_FixedShotDurationStub / otr_shot_duration_calculator.py -- REQUIRED (real
  per-shot frame expansion BatchFluxRender consumes; test-pinned). Do NOT delete. Optional
  later: rename off "Stub" in lockstep (class + __init__ + JSON type + S&R name + its test).
- Node 63 OTR_WorkflowValidator + _workflow_validation.py -- KEEP the module. Verified: it
  introspects each node's INPUT_TYPES() dynamically and NEVER inspects node outputs, so the
  creative_writing_model prune cannot false-fail it, and it pins no writer output count. Its
  widget-drift check catches a TRUNCATED or None widget slot but NOT a same-length
  transposition -- so it is not a safety net for widget removal; name-keyed regen is. The node
  is graph-detached; optional later: move checks to CI, drop the detached node.
- use_exchange writer widget (widgets_values[19]) -- the LIVE feature. KEEP.

## WARNING -- two near-identical names, opposite fates
- _otr_stage1_plan  (alias _OTRS1P) -- LIVE outline path. KEEP.
- _otr_stage1_call  (alias _OTRS1)  -- shadow-pass only (writer import + all call sites sit
  inside `if resolved.get("enable_stage1_shadow_pass")`, ~line 2487-2750). DELETE with the
  shadow cluster. Deleting _otr_stage1_plan by mistake breaks the live build.

## Complete deletion inventory (verified)

MULTITURN -- Wave-0 dialogue, superseded by use_exchange -- DELETE
- nodes/_otr_wave0_multiturn.py, _otr_stage2_call.py, _otr_stage2_prompt.py
- tests/test_wave0_multiturn_dispatch.py, test_stage2_multiturn.py
- writer: dispatch block + use_multiturn_dialogue widget (wv[18]) + kwarg + resolved key

STORY ROOM -- dormant writers-room, replaced by use_exchange -- DELETE
- nodes/OTR_StoryRoom.py, _otr_story_room.py, OTR_StoryRoomExtract.py,
  _otr_story_room_extract.py, OTR_StoryRoomCommit.py, OTR_DirectorBrief.py,
  _otr_director_brief.py, OTR_EditorPass.py, _otr_editor_pass.py,
  _otr_writers_room_resolver.py  (+ their 5 _NODE_MODULES entries)
- dedicated tests: test_otr_story_room, test_otr_story_room_extract, test_otr_director_brief,
  test_otr_editor_pass, test_writers_room_resolver, test_bug_local_293_extract_token_budget,
  test_bug_local_291_editor_token_budget
- EDIT, do not delete: test_constraint_editor_live_swap.py and
  test_writer_constraint_repair_splice.py import BOTH the cluster AND the live constraint
  editor -- drop only the Story-Room/Director/Editor fixtures, keep the live-editor coverage.
  test_dialogue_slot_id.py: confirm it is not asserting live slot-id behavior before cutting.
- scripts/smoke_fanout_constraint.py -- delete (dormant-feature smoke).
- Tombstone in DELETED_NODE_TYPES: OTR_StoryRoom, OTR_StoryRoomExtract, OTR_StoryRoomCommit,
  OTR_DirectorBrief, OTR_EditorPass (verified zero JSON refs -> safe).

SHADOW + FAN-OUT (+ cast audit + name-gender) -- diagnostics for a swap that never landed.
All reachable ONLY through enable_stage1_shadow_pass (verified by call-site line numbers) -- DELETE
- nodes/_otr_stage1_fanout.py, _otr_beat_selector.py, _otr_stage1_call.py,
  _otr_stage1_cast_audit.py, _otr_name_gender.py  (name_gender's only non-test caller is
  cast_audit, which is shadow-gated -> the whole sub-tree goes together)
- nodes/OTR_Stage1FanOut.py, OTR_BeatSelector.py  (+ their _NODE_MODULES entries)
- tests: test_stage1_fanout, test_stage1_fanout_and_select, test_beat_selector,
  test_beat_selector_dict_plan_a, test_stage1_call, test_stage1_prompt_bounds,
  test_stage1_shadow_pass_integration, test_stage1_cast_audit
- writer: shadow-pass block (~2467-2750) incl. the fan-out sub-block + the Build-3 local-alias
  import note; widgets enable_stage1_shadow_pass (wv[17]) + use_stage1_fanout (wv[22]) +
  kwargs + resolved keys. Leave _otr_stage1_plan untouched.
- Tombstone in DELETED_NODE_TYPES: OTR_Stage1FanOut, OTR_BeatSelector (zero JSON refs -> safe)

GRAPH-ONLY (no repo module)
- Node 42 PathchSageAttentionKJ (external KJNodes node, DISABLED) -- delete from JSON + bridge.

IN-FILE, NOT A MODULE (files stay; logic carved out)
- Polish: _otr_line_composer.py + _otr_model_loader.py + _otr_reroll.py; writer widget
  enable_polish_pass (wv[15]).

WRITER OUTPUT (graph + surface)
- creative_writing_model output (slot 4) -- verified ZERO consumers; safe to prune. Link 115
  is the ONLY slot-5 link, so the 5->4 renumber is complete.

REGISTERED-BUT-UNUSED -- scan-gated, LAST. The registry list is noisy (it still names
already-tombstoned types and nodes used only by other workflows: humo_smoke_*.json,
_bisect_*.json). Do NOT bulk-delete. For each candidate, prove zero references across ALL 10
workflow JSONs + every test + every import + every smoke script before deleting, then
tombstone one at a time. Candidates: OTR_BisectStringSource; OTR_Visual* sidecar;
OTR_CheckpointLoaderGated; OTR_VideoConcat; OTR_BatchProceduralSFX; OTR_ProjectStateLoader;
OTR_SaveCopy; _otr_lfc_context. (OTR_VRAMGuardian / OTR_VRAMContextTest are NOT here -- kept.)

## Widget removal method (the safe one) -- name-keyed regeneration
The writer's widgets_values is positional: required(3) then optional(20), declared order.
Verified 0-based map of the relevant slots:
  [15]=enable_polish_pass  [16]=lemmy_cameo  [17]=enable_stage1_shadow_pass
  [18]=use_multiturn_dialogue  [19]=use_exchange(KEEP)
  [20]=enable_production_stage3_validators  [21]=news_briefs_required  [22]=use_stage1_fanout
The widgets to remove are INTERLEAVED with keepers, so tail-lopping / blind index-popping is
wrong. For each commit that removes a widget:
1. Build `old_map = dict(zip(OLD_ORDERED_NAMES, widgets_values))` from current INPUT_TYPES order.
2. Edit INPUT_TYPES / run() / _resolve_inputs to drop the widget.
3. Regenerate `widgets_values = [old_map[name] for name in NEW_ORDERED_NAMES]`.
4. Assert: removed names absent from NEW order; every surviving name present in old_map;
   len(new) == new INPUT_TYPES count.
5. Follow the backend+frontend cache discipline above, then confirm node 1 loads and the run
   reaches audio.
Name-based, so it cannot transpose a value into the wrong field -- the failure the validator
does NOT catch.

## Execution order (one concern per commit; regression + cache-safe reload each)
1. Node 42 -- JSON only (mutate master AND node-local): delete master link
   [203,71,0,42,0,"MODEL"]; change link 69 [69,42,0,23,0,"MODEL"] -> [69,71,0,23,0,"MODEL"];
   node 71 outputs[0].links [203,204] -> [69,204]; node 23 input link stays 69; delete node
   42; last_link_id stays 230. Validate. Trap: deleting node-local 203 but leaving it in the
   master array.
2. Multiturn -- delete files + tests; writer dispatch block + widget (name-keyed regen).
3. Story Room -- delete the cluster files + _NODE_MODULES entries + dedicated tests; EDIT the
   two shared constraint-editor tests; delete the smoke script; tombstone the 5 types. Do NOT
   delete _otr_editor_constraints.py.
4. Shadow + fan-out -- delete the 5 modules + 2 node classes + 8 tests; writer shadow block +
   2 widgets (name-keyed regen); tombstone the 2 types. Leave _otr_stage1_plan untouched.
5. Model-loader audit -- conservative. Remove a loader path ONLY if it has zero remaining
   callers after steps 2-4: polish-only generate fn (make_polish_generate_fn), shadow-only
   cache warming, fan-out-only prep, multiturn/stage2-only paths, orphaned slot warmups,
   orphaned VRAM reservation. Do NOT delete multimodal / Gemini / API-model staging unless you
   prove it is not used by _otr_craft_floor, news_interpreter, the Tier-A gates, the live
   writer path, or any API-model path. The loader is shared hot-path code -- re-run a FULL
   live episode, not just pytest; a VRAM regression will not show in unit tests.
6. Polish (LAST; hot-path) -- remove needs_polish / polish_line / is_polish_refusal /
   _POLISH_* / the enable_polish_pass branch / make_polish_generate_fn / the reroll polish
   flag / widget (name-keyed regen) / polish tests. Treat as the riskiest refactor: you are
   editing live line-composer files, not deleting dormant modules. FULL audio byte-identity
   regression.
7. Prune writer creative_writing_model output -- RETURN_TYPES/NAMES drop it; run() tuple drop
   it; JSON delete output slot 4, renumber technical_model 5->4, change link 115
   [115,1,5,62,4,"STRING"] -> [115,1,4,62,4,"STRING"]; update the 3 writer-surface guardrail
   tests + the writer self-test. Use the slot 4/5 restart ordering above before loading JSON.
8. Cruft (scan-gated) -- only after the per-candidate zero-reference proof; tombstone each as
   it goes. Then untracked superseded plan docs + gitignored scratch/__pycache__.

Per commit: Bug Bible + core + audio + affected suites green; cache-safe ComfyUI restart to
confirm node 1 loads and the run reaches audio; re-run the link-table validator after any JSON
edit.
