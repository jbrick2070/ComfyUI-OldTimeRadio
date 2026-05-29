# OTR Lean-Down -- Go-Forward Plan

Goal: strip the pipeline to one clean path -- writer (use_exchange) -> freeze cascade ->
audio -> video -- removing dead/dormant machinery that does not serve the story. Story
quality is paramount; nothing load-bearing is touched.

## Current state
otr_scifi_16gb_full.json is graph-lean: 31 nodes, 69 links, last_link_id 230 (== max, no
orphan/dup links), writer.script_json -> link 230 -> FreezeCascade. use_exchange ON;
multiturn / shadow-pass / fan-out / polish all OFF. Remaining work: code removal + two small
graph cuts + a model-loader audit.

## JSON surgery rules (read before any graph edit)
The workflow JSON is TWO synchronized systems: the master `links` array at the bottom, and
each node's local `inputs[].link` / `outputs[].links`. EVERY link change must mutate BOTH --
node-local edits alone leave the file corrupt even when the visual graph looks right. All
edits go through Desktop Commander on the real file; the bash/VM mount serves a corrupted
copy. After every JSON edit run the link-table validation (keep it as a reusable script):
- JSON parses
- every node-local link id exists in the master `links` array
- every master link has a real source node and a real target node (no orphans)
- no duplicate link ids
- last_link_id == max(link id)
- no reserved link ids (111, 112)
- no stale output-link ids left on surviving nodes

## Keep -- load-bearing, never remove
- nodes/_otr_compose_exchange.py (dialogue engine), _otr_craft_floor.py (Tier-A gate),
  _otr_slot_drama_contract.py (contracts), _otr_editor_constraints.py + _otr_beat_validators.py.
- Shared libs: _otr_stage1_plan, _otr_constrained_generate, _otr_legacy_to_stage1_adapter,
  _otr_whole_episode_critic, news_interpreter, production_ledger, story_orchestrator,
  _otr_model_loader, _otr_line_composer (polish carved out, file stays), _otr_reroll.
- Node 21 OTR_FixedShotDurationStub / otr_shot_duration_calculator.py -- REQUIRED (real
  per-shot frame expansion BatchFluxRender consumes; test-pinned). Optional later: rename
  off "Stub" (lockstep: class + __init__ + JSON type + S&R name + its test). Do NOT delete.
- Node 63 OTR_WorkflowValidator + _workflow_validation.py -- KEEP the module. The node is
  graph-detached (no inputs/consumers); optional later: move its link-table checks to CI and
  drop the node from the production graph. Low priority -- it is a useful pre-run net.

## Complete deletion inventory (modules suspect for deletion)

MULTITURN -- Wave-0 dialogue, superseded by use_exchange -- DELETE
- nodes/_otr_wave0_multiturn.py
- nodes/_otr_stage2_call.py
- nodes/_otr_stage2_prompt.py

STORY ROOM -- dormant writers-room, replaced by use_exchange -- DELETE
- nodes/OTR_StoryRoom.py
- nodes/_otr_story_room.py
- nodes/OTR_StoryRoomExtract.py
- nodes/_otr_story_room_extract.py
- nodes/OTR_StoryRoomCommit.py
- nodes/OTR_DirectorBrief.py
- nodes/_otr_director_brief.py
- nodes/OTR_EditorPass.py
- nodes/_otr_editor_pass.py
- nodes/_otr_writers_room_resolver.py

STAGE-1 SHADOW + FAN-OUT -- diagnostics for a swap that never landed -- DELETE
- nodes/OTR_Stage1FanOut.py
- nodes/OTR_BeatSelector.py
- nodes/_otr_stage1_fanout.py
- nodes/_otr_beat_selector.py
- nodes/_otr_stage1_call.py
- nodes/_otr_stage1_cast_audit.py
- nodes/_otr_name_gender.py            [VERIFY-FIRST: confirm no other caller]

GRAPH-ONLY (no repo module)
- Node 42 PathchSageAttentionKJ (external KJNodes node, DISABLED) -- delete from JSON + bridge.

IN-FILE, NOT A MODULE (files stay; logic carved out)
- Polish: _otr_line_composer.py + _otr_model_loader.py + _otr_reroll.py

REGISTERED-BUT-UNUSED NODES -- [VERIFY-FIRST: scan every workflow JSON + test fixture + import + doc before delete]
- nodes/OTR_BisectStringSource.py        (temporary, BUG-231)
- nodes/OTR_VisualBridge.py, OTR_VisualPoll.py, OTR_VisualRenderer.py,
  OTR_VisualPromptCoercion.py, OTR_VisualExtractFluxPrompt.py   (visual sidecar; may be another workflow)
- nodes/OTR_CheckpointLoaderGated.py     (superseded by DeferredCheckpointLoader)
- nodes/OTR_VideoConcat.py               (superseded by VideoComposite)
- nodes/OTR_BatchProceduralSFX.py        (check _otr_sfx_lib coupling)
- nodes/OTR_ProjectStateLoader.py, OTR_VRAMGuardian.py, OTR_VRAMContextTest.py, OTR_SaveCopy.py
- nodes/_otr_lfc_context.py              (orphan candidate -- only a test references it)

## Execution order (one concern per commit; regression + restart each)
1. Node 42 -- JSON only (mutate master links AND node-local): delete master link
   [203,71,0,42,0,"MODEL"]; mutate master link 69 [69,42,0,23,0,"MODEL"] ->
   [69,71,0,23,0,"MODEL"]; node 71 outputs[0].links [203,204] -> [69,204]; node 23 input
   link stays 69; delete node 42; last_link_id stays 230. Run link-table validation.
2. Story Room code -- delete the 10 files + the 5 _NODE_MODULES entries + Story-Room tests;
   edit constraint-editor tests to drop DirectorBrief/EditorPass fixture imports only.
3. Multiturn -- delete the 3 files + writer dispatch block + widget [18] + kwargs +
   resolved-dict key + tests.
4. Shadow + fan-out -- delete the 7 files (verify _otr_name_gender) + 2 _NODE_MODULES
   entries + writer shadow/fan-out blocks + widgets ([22] then [17]) + tests.
5. Model-loader audit -- audit _otr_model_loader.py for prep tied ONLY to deleted features:
   make_polish_generate_fn, any shadow / fan-out / multiturn / diagnostic-only cache warming
   or VRAM reservation, unused multimodal/Gemini staging, large diagnostic-only model
   families, orphaned slot warmups. Remove orphaned loader paths -- dead VRAM prep silently
   burns VRAM. (Audit, then remove only what is confirmed orphaned.)
6. Polish (LAST; hot-path surgery) -- remove needs_polish / polish_line / is_polish_refusal
   / _POLISH_* constants / the enable_polish_pass branch / make_polish_generate_fn / reroll
   polish flag / writer widget [15] / polish tests. FULL audio byte-identity regression.
7. Prune writer creative_writing_model output -- RETURN_TYPES/NAMES drop it; run() return
   tuple drop it; JSON delete output slot 4, renumber technical_model slot 5->4, mutate
   master link 115 [115,1,5,62,4,"STRING"] -> [115,1,4,62,4,"STRING"]; update the 3
   writer-surface guardrail tests + the writer self-test.
8. Cruft (scan-gated) -- registered-but-unused nodes (scan every workflow JSON + fixture +
   import + doc first); untracked superseded plan docs; gitignored scratch + __pycache__.

## The two crash points
1. WRITER WIDGET-INDEX DRIFT (removing optional widgets). Two safe methods:
   - Phased (safest): replace each removed widget with an inert placeholder widget in
     Python -> load ComfyUI -> confirm node 1 loads -> save a clean workflow -> remove the
     placeholders + migrate the JSON array in the same commit.
   - Scripted: pop widgets_values highest index first (22 -> 18 -> 17 -> 15) in the SAME
     commit as the INPUT_TYPES / run() / _resolve_inputs edits, asserting FIRST
     len==23 and widgets_values[15]/[17]/[18]/[22] all False. No assert, no pop.
   Index map: [15]=enable_polish_pass [17]=enable_stage1_shadow_pass
   [18]=use_multiturn_dialogue [22]=use_stage1_fanout.
2. OUTPUT SLOT-INDEX DRIFT (pruning creative_writing_model) -- must decrement
   technical_model slot 5->4 AND mutate master link 115 src_slot 5->4. If 115 still points
   to source slot 5, the graph lies.

Per commit: Bug Bible + core + audio + affected suites green; restart ComfyUI to confirm
node 1 loads and the run reaches audio; re-run the link-table validation after any JSON edit.
