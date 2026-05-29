# OTR Lean-Down -- Go-Forward Plan

Goal: strip the pipeline to one clean path -- writer (use_exchange) -> freeze cascade ->
audio -> video -- removing dead/dormant machinery that does not serve the story. Audio is
king; nothing load-bearing is touched. Every call below is grounded in a verification pass
over the real files (link table, writer INPUT_TYPES via ast, importer graph, loader/registry
scan, VRAM-primitive definition sites, all workflow JSONs). The big traps: master-link
desync, writer widget transposition, the slot 4/5 race, frontend-cache poisoning, deleting a
shared symbol that only LOOKS dormant, and deleting VRAM-survival logic. Each has a guard.

## Progress
- Sprint 1 / step 4 (Node 42 cut) DONE -- commit 7b67503, pushed to v2.0-alpha. Graph now
  30 nodes / 68 links, last_link_id 230. Link-table validation (7 checks) + tests/test_core.py
  (59 passed) green. NEXT: step 5 (Multiturn).
- Bug Bible regression was NOT runnable in the Cowork sandbox (survival-guide repo absent at
  C:\Users\jeffr\Documents\ComfyUI\comfyui-custom-node-survival-guide). On the full machine,
  run it every commit per CLAUDE.md alongside core + audio.

## Current state (verified)
otr_scifi_16gb_full.json: 30 nodes, 68 links, last_link_id 230 (== max, no orphan/dup),
writer.script_json -> link 230 -> FreezeCascade. use_exchange ON; multiturn / shadow / fan-out
/ polish all OFF. All 10 workflow JSONs scanned: zero references to any node type slated for
deletion -- tombstoning deleted types is safe.

## Preconditions (verified before any deletion)
- Pre-deletion reference scan DONE: story_orchestrator.py, _vram_log.py, and the model loader
  carry NO references to the multiturn / Story Room / shadow / fan-out features. The loader's
  only flagged symbol is make_polish_generate_fn -- which is KEPT (see Polish below).
- Registry is EXPLICIT, not dynamic: __init__.py defines `_NODE_MODULES = {...}` and runs
  `for ...: importlib.import_module(module_path)` (no glob / pkgutil / walk auto-discovery).
  Consequence: a deleted module's `_NODE_MODULES` entry MUST be removed in the SAME commit as
  the file, or the import loop throws ImportError on startup. No hidden auto-registration.
- Node 63 _workflow_validation.py READ in full: it does NOT enforce required Story Room /
  shadow / fan-out types, a writer output count, an old node/link count, or any topology
  shape. It checks per-node INPUT_TYPES contract + a deleted-type deny-list + link-table
  integrity only. Safe to keep; the prune cannot false-fail it. Tombstoned types are actively
  guarded by its deny-list (check 4), which is why tombstoning is the correct end-state.

## Safety gate -- run before AND after every graph edit
The workflow JSON is TWO synchronized routing systems: each node's local `inputs[].link` /
`outputs[].links`, and the master `links` array. If either drifts, the file loads with ghost
routes or silently wrong wiring. All edits via Desktop Commander on the real file (the VM
mount serves a corrupted copy). Link-table validator, every edit:
- JSON parses
- every node-local link id exists in the master `links` array
- every master link points to a real source AND target node (no orphans)
- no duplicate link ids
- last_link_id == max(link id)
- no reserved link ids (111, 112)
- no stale output-link ids left on surviving nodes
(The Sprint-1 apply script implements all 7 inline; reuse that pattern as the JSON gate.)

## Operator discipline -- backend + frontend cache (every writer-surface change)
ComfyUI caches node definitions in the running backend AND the browser. Any change to the
writer's INPUT_TYPES, RETURN_TYPES, or widgets (steps 5, 6, 7, 9, 10) must follow this order or
the frontend forces a stale node definition over the new JSON and corrupts widgets_values on
save:
1. Edit Python first. 2. Stop ComfyUI. 3. Clear Python __pycache__. 4. Restart the backend.
5. Hard-refresh the browser (Ctrl+F5); clear ComfyUI local storage for the tab if needed.
6. THEN load the mutated JSON. 7. THEN save. 8. THEN run the link-table validator.
Slot 4/5 race (step 10): never open the mutated JSON while the backend still believes output
slot 4 is creative_writing_model. Restart before loading.

## Keep -- load-bearing, never remove
- nodes/_otr_compose_exchange.py, _otr_craft_floor.py, _otr_slot_drama_contract.py,
  _otr_editor_constraints.py (LIVE writer constraint logic -- NOT Story Room code; do not
  delete it with the cluster), _otr_beat_validators.py.
- Shared libs: _otr_stage1_plan (LIVE outline -- see warning), _otr_constrained_generate,
  _otr_legacy_to_stage1_adapter, _otr_whole_episode_critic, news_interpreter,
  production_ledger, story_orchestrator, _otr_model_loader, _vram_log, _otr_line_composer,
  _otr_reroll.
- make_polish_generate_fn (in _otr_model_loader.py) -- KEEP. Despite the name it is NOT
  polish-only: OTR_LedgerFreezeCascade (live core node, line 348) and the writer base path
  (line 571, outside the polish gate) and _otr_line_composer all call it; test-pinned by
  test_lfc_w4_writer_polish_fn ("required v2.0"). It is a shared conservative-sampling
  factory. Polish-feature removal must NOT touch it.
- VRAM operator tools: nodes/vram_guardian.py (OTR_VRAMGuardian) + nodes/vram_context_test.py
  (OTR_VRAMContextTest). Not in any workflow and not on the live path, BUT they are purpose-
  built manual VRAM-flush / context probes for the 16GB FLUX->HuMo handoff, cost one registry
  line each, and OTR_VRAMGuardian is pinned by tests/test_core.py. Keep -- a cheap safety net.
- Node 21 OTR_FixedShotDurationStub -- REQUIRED (real frame expansion; test-pinned). Optional
  later: rename off "Stub" in lockstep. Node 63 validator module -- KEEP (see Preconditions).
- use_exchange writer widget (widgets_values[19]) -- the LIVE feature. KEEP.

## WARNING -- two near-identical names, opposite fates
- _otr_stage1_plan (_OTRS1P) -- LIVE outline path. KEEP.
- _otr_stage1_call (_OTRS1) -- shadow-pass only (all call sites inside
  `if resolved.get("enable_stage1_shadow_pass")`, ~2487-2750). DELETE with the shadow cluster.

## Deletion inventory (verified)
MULTITURN -- DELETE: nodes/_otr_wave0_multiturn.py, _otr_stage2_call.py, _otr_stage2_prompt.py;
tests test_wave0_multiturn_dispatch, test_stage2_multiturn; writer dispatch block +
use_multiturn_dialogue widget (wv[18]) + kwarg + resolved key.

STORY ROOM -- DELETE: OTR_StoryRoom, _otr_story_room, OTR_StoryRoomExtract,
_otr_story_room_extract, OTR_StoryRoomCommit, OTR_DirectorBrief, _otr_director_brief,
OTR_EditorPass, _otr_editor_pass, _otr_writers_room_resolver (+ their 5 _NODE_MODULES entries,
same commit); dedicated tests (test_otr_story_room, _extract, test_otr_director_brief,
test_otr_editor_pass, test_writers_room_resolver, test_bug_local_293, test_bug_local_291);
EDIT (not delete) test_constraint_editor_live_swap + test_writer_constraint_repair_splice
(drop only cluster fixtures, keep live-editor coverage); confirm test_dialogue_slot_id is not
asserting live behavior; delete scripts/smoke_fanout_constraint.py; tombstone OTR_StoryRoom,
OTR_StoryRoomExtract, OTR_StoryRoomCommit, OTR_DirectorBrief, OTR_EditorPass.

SHADOW + FAN-OUT (+ cast audit + name-gender), all shadow-gated -- DELETE: _otr_stage1_fanout,
_otr_beat_selector, _otr_stage1_call, _otr_stage1_cast_audit, _otr_name_gender, OTR_Stage1FanOut,
OTR_BeatSelector (+ _NODE_MODULES entries, same commit); tests test_stage1_fanout,
test_stage1_fanout_and_select, test_beat_selector, test_beat_selector_dict_plan_a,
test_stage1_call, test_stage1_prompt_bounds, test_stage1_shadow_pass_integration,
test_stage1_cast_audit; writer shadow block (~2467-2750) + widgets enable_stage1_shadow_pass
(wv[17]) + use_stage1_fanout (wv[22]) + kwargs/resolved keys; leave _otr_stage1_plan untouched;
tombstone OTR_Stage1FanOut, OTR_BeatSelector.

GRAPH-ONLY: Node 42 PathchSageAttentionKJ (sage_attention='disabled' passthrough, BUG-LOCAL-070)
-- DONE (commit 7b67503).

POLISH (in-file; per-symbol audit, NOT bulk): remove ONLY the enable_polish_pass-exclusive
surface (the enable_polish_pass branch + widget wv[15] + reroll polish flag + polish tests).
Audit each of needs_polish / polish_line / is_polish_refusal / _POLISH_* for live reuse before
removing -- make_polish_generate_fn already proved shared, so do not assume the rest are
polish-only. KEEP make_polish_generate_fn. Lowest leanness payoff, highest entanglement risk:
if a symbol is shared, leave it; widget-only removal is an acceptable outcome here.

WRITER OUTPUT: creative_writing_model output (slot 4) -- verified ZERO consumers; link 115 is
the ONLY slot-5 link, so the 5->4 renumber is complete.

REGISTERED-BUT-UNUSED -- scan-gated, LAST. Noisy list (names tombstoned types + other-workflow
nodes). Per candidate, prove zero refs across all 10 JSONs + tests + imports + smoke scripts,
then tombstone one at a time: OTR_BisectStringSource; OTR_Visual* sidecar;
OTR_CheckpointLoaderGated; OTR_VideoConcat; OTR_BatchProceduralSFX; OTR_ProjectStateLoader;
OTR_SaveCopy; _otr_lfc_context. (VRAM guardians are NOT here -- kept.)

## Widget removal method -- name-keyed regen WITH a value-assertion gate
widgets_values is positional: required(3) then optional(20). Verified slots:
  [15]=enable_polish_pass [16]=lemmy_cameo [17]=enable_stage1_shadow_pass
  [18]=use_multiturn_dialogue [19]=use_exchange(KEEP) [20]=enable_production_stage3_validators
  [21]=news_briefs_required [22]=use_stage1_fanout
Name-keyed regen alone trusts that the Python name order matches the JSON save-time schema.
Prove that first, every widget commit:
1. Assert the sentinel values at their known indices BEFORE migrating:
   wv[15]==False, wv[17]==False, wv[18]==False, wv[19]==True, wv[22]==False, plus the adjacent
   keepers (wv[16], wv[20], wv[21]) match expected. No clean assertion -> STOP, schema drifted.
2. old_map = dict(zip(OLD_ORDERED_NAMES, widgets_values)); edit INPUT_TYPES/run()/_resolve.
3. widgets_values = [old_map[name] for name in NEW_ORDERED_NAMES].
4. Assert removed names absent; surviving names all present; len matches.
5. Write the before/after widget map to a migration artifact (auditable).
6. Cache-safe restart, confirm node 1 loads and the run reaches audio.

## Execution order (one concern per commit; regression + cache-safe reload each)
1. Add/run the link-table validator. (DONE -- Sprint-1 apply script.)
2. Pre-deletion reference scan. (DONE; re-run if the tree changed.)
3. Node 63 topology verify. (DONE.)
4. Node 42 -- JSON-only link-203 bridge. (DONE -- commit 7b67503.)
5. Multiturn -- delete files + _NODE_MODULES entry + tests; writer dispatch block + widget
   (value-asserted name-keyed regen).
6. Story Room -- delete cluster files + 5 _NODE_MODULES entries + dedicated tests; EDIT the two
   shared constraint-editor tests; delete the smoke script; tombstone 5 types. Do NOT delete
   _otr_editor_constraints.py.
7. Shadow + fan-out -- delete 5 modules + 2 node classes + entries + 8 tests; writer shadow
   block + 2 widgets (value-asserted regen); tombstone 2 types. Leave _otr_stage1_plan.
8. Post-deletion loader cleanup -- conservative. The pre-scan found the loader clean of these
   features, so expect little/none. Remove a loader path ONLY with zero remaining callers; do
   NOT touch make_polish_generate_fn; do NOT remove multimodal/Gemini/API staging unless
   proven unused by _otr_craft_floor, news_interpreter, the Tier-A gates, the live writer, or
   any API path. Re-run a FULL live episode, not just pytest.
9. Polish (LAST; hot-path) -- per-symbol audit per the inventory; KEEP make_polish_generate_fn;
   widget via value-asserted regen. FULL audio byte-identity regression.
10. Prune creative_writing_model output -- RETURN_TYPES/NAMES + run() tuple; JSON delete output
    slot 4, technical_model 5->4, link 115 [115,1,5,62,4,"STRING"] -> [115,1,4,62,4,"STRING"];
    update 3 writer-surface tests + the writer self-test. Use the slot 4/5 restart ordering.
11. VRAM guardians -- KEEP (see above).
12. Cruft (scan-gated) -- per-candidate zero-reference proof, then tombstone each.

Per commit: core + audio + affected suites green (+ Bug Bible on the full machine); cache-safe
ComfyUI restart to confirm node 1 loads and the run reaches audio; re-run the link-table
validator after any JSON edit.

## Phase 2 -- Headless API testing (after sprints 5-12 land)
Goal: drive a full episode through ComfyUI's HTTP API with nobody watching the desktop, and
iterate to a clean render. This closes the "AI cannot see the ComfyUI Desktop console" gap:
launch ComfyUI ourselves so its stdout/stderr is captured in a Desktop Commander process the
AI can read.
1. Launch headless under Desktop Commander (its log becomes AI-readable):
   cd C:\Users\jeffr\Documents\ComfyUI && .venv\Scripts\python.exe main.py --port 8000
   Keep the PID; read_process_output tails PARSE_FATAL / tracebacks / VRAM spikes / FFmpeg.
   (Windows MCP can launch/focus the Desktop app instead if a GUI run is preferred.)
2. Export an API-format copy of the lean workflow. The UI JSON (nodes/links) is NOT what
   /prompt accepts -- /prompt wants the API-prompt dict keyed by node id with class_type +
   inputs. Use ComfyUI "Save (API Format)" or convert programmatically; keep it beside
   otr_scifi_16gb_full.json.
3. Queue: POST http://127.0.0.1:8000/prompt with {"prompt": <api_graph>, "client_id": <uuid>};
   capture prompt_id. A 400 here is a node/input contract error -- the body names the node.
4. Watch: poll http://127.0.0.1:8000/history/<prompt_id> until it appears (done) AND tail the
   launched process stdout. Optional: ws://127.0.0.1:8000/ws?clientId=<uuid> for progress +
   execution_error events.
5. On error: read the traceback from our own process log, fix autonomously (code / JSON /
   widget), re-run the relevant pytest, re-queue. Iterate.
6. Success gate (a working workflow): /history shows completed; the audio output file exists
   and is non-empty; no PARSE_FATAL; dialogue line count > 0; VRAM peak <= 14.5 GB; FFmpeg
   returned 0; audio byte-identity vs the pre-lean baseline still holds.
7. Record the proven API JSON + prompt_id + runtime in BUG_LOG.md.
Tooling note: Desktop Commander does the headless launch + log tail + git; Windows MCP can
drive the Desktop GUI (start, Queue, screenshot) if a desktop run is wanted. Neither needs
input from Jeffrey.

## Hard blockers before execution
- No widget migration without the value/type assertion gate.
- No module deletion without removing its _NODE_MODULES entry in the same commit.
- No JSON edit without master-links + node-local validation, before and after.
- No writer-surface edit without backend restart + browser cache wipe.
- No "polish" symbol removed until proven enable_polish_pass-exclusive (make_polish_generate_fn
  is shared -- KEEP).
- No VRAM-utility deletion: proven unused by code, not just absent from JSON -> still KEEP.
