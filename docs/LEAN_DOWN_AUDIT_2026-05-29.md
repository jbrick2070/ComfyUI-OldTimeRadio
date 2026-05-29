# OTR Lean-Down -- Go-Forward Plan

Goal: strip the pipeline to one clean path -- writer (use_exchange) -> freeze cascade ->
audio -> video -- and remove the dead/dormant machinery that does not help the story.
Story quality is paramount; nothing load-bearing is touched.

## Current state
otr_scifi_16gb_full.json is already graph-lean: 31 nodes, 69 links, last_link_id 230
(== max, no orphan/dup links), writer.script_json -> link 230 -> FreezeCascade.
use_exchange ON; multiturn / shadow-pass / fan-out / polish all OFF. Remaining work is
code removal plus two small graph cuts. ALL JSON edits go through Desktop Commander on the
real file -- the bash/VM mount serves a corrupted copy.

## Keep -- load-bearing, never remove
- nodes/_otr_compose_exchange.py (dialogue engine), _otr_craft_floor.py (Tier-A gate),
  _otr_slot_drama_contract.py (contracts), _otr_editor_constraints.py + _otr_beat_validators.py.
- Shared libs: _otr_stage1_plan, _otr_constrained_generate, _otr_legacy_to_stage1_adapter,
  _otr_whole_episode_critic, news_interpreter, production_ledger, story_orchestrator,
  _otr_model_loader, _otr_line_composer (polish carved out, file stays), _otr_reroll.
- Node 21 OTR_FixedShotDurationStub / otr_shot_duration_calculator.py -- REQUIRED (real
  per-shot frame expansion that BatchFluxRender consumes; test-pinned). Optional later:
  rename off "Stub" (lockstep: class + __init__ + JSON type + S&R name + its test).

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

POLISH -- in-file, not a standalone module (files stay; logic carved out)
- _otr_line_composer.py, _otr_model_loader.py, _otr_reroll.py

REGISTERED-BUT-UNUSED NODES -- [VERIFY-FIRST: scan every workflow JSON + test fixture before delete]
- nodes/OTR_BisectStringSource.py        (temporary, BUG-231)
- nodes/OTR_VisualBridge.py, OTR_VisualPoll.py, OTR_VisualRenderer.py,
  OTR_VisualPromptCoercion.py, OTR_VisualExtractFluxPrompt.py   (visual sidecar; may be another workflow)
- nodes/OTR_CheckpointLoaderGated.py     (superseded by DeferredCheckpointLoader)
- nodes/OTR_VideoConcat.py               (superseded by VideoComposite)
- nodes/OTR_BatchProceduralSFX.py        (check _otr_sfx_lib coupling)
- nodes/OTR_ProjectStateLoader.py, OTR_VRAMGuardian.py, OTR_VRAMContextTest.py, OTR_SaveCopy.py
- nodes/_otr_lfc_context.py              (orphan candidate -- only a test references it)

## Execution order (one concern per commit; regression + restart each)
1. Node 42 -- JSON only: delete link 203; mutate link 69 -> [69,71,0,23,0,"MODEL"];
   node 71 outputs[0].links [203,204] -> [69,204]; delete node 42; last_link_id stays 230.
2. Story Room code -- delete the 10 files above + the 5 _NODE_MODULES entries + Story-Room
   tests; edit constraint-editor tests to drop director/editor fixture imports.
3. Multiturn -- delete the 3 files + writer dispatch block + widget (widgets_values [18]) +
   kwargs + resolved-dict key + tests.
4. Shadow + fan-out -- delete the 7 files (verify _otr_name_gender) + 2 _NODE_MODULES entries
   + writer shadow/fan-out blocks + widgets (pop [22] then [17]) + tests.
5. Polish -- carve out of _otr_line_composer + _otr_model_loader + _otr_reroll + widget [15]
   + polish tests; FULL audio byte-identity regression.
6. Prune writer creative_writing_model output -- RETURN_TYPES/NAMES drop it; run() return
   tuple drop it; JSON remove output slot 4, renumber technical_model 5->4, mutate link 115
   src_slot 5->4; update the 3 writer-surface guardrail tests + the writer self-test.
7. Cruft (after a repo-wide reference scan) -- the registered-but-unused nodes; untracked
   superseded plan docs; gitignored scratch + __pycache__.

## Protocol + the two crash points
- One concern per commit. After each: Bug Bible + core + audio + affected suites green;
  restart ComfyUI to confirm node 1 loads and the run reaches audio.
- All JSON edits via Desktop Commander (real file). After every JSON change re-assert:
  no orphan links, no duplicate ids, last_link_id == max(link id).
- CRASH POINT 1 -- writer widget-index drift: remove optional widgets only in the SAME
  commit as the matching INPUT_TYPES / run() / _resolve_inputs edits, popping widgets_values
  highest index first (22 -> 18 -> 17 -> 15), asserting length 23 + each popped value False first.
- CRASH POINT 2 -- output slot-index drift: pruning the creative_writing_model output must
  decrement technical_model slot 5->4 AND change link 115 src_slot 5->4.
- Writer widget index map (widgets_values, len 23): [15]=enable_polish_pass,
  [17]=enable_stage1_shadow_pass, [18]=use_multiturn_dialogue, [22]=use_stage1_fanout.
