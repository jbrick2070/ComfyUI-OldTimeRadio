# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-28 (QA-READY)

All sprints shipped. HEAD == origin/v2.0-alpha at `b2a9ac6`. Ready
for Jeffrey's synthetic QA review pass.

## Full commit chain (28 commits)

| Sprint | Commit | Title |
|---|---|---|
| 0 | e3e4793 | docs: baseline + refined Sprint 1-5 plan |
| 1 | 2302f1e | dialogue_slot_id keystone + narrow Extract path |
| 2 | c1ea43e | DramaticState + structural beat-sheet validators |
| 3 | e59f32b | arc-aware line composer prompt (Path A) |
| 4 | 79c8014 | best-of-N beat-sheet selector |
| 5 | ed833d8 | editor downgrade to constraint checker |
| docs | eb8fdcd | handoff refresh after Sprint 1-5 cascade |
| 2.1 | a0a175d | stamp DramaticState on meta after Stage 1 keystone |
| 3.1 | e6a433f | thread DRAMATIC FRAME fields into LineRequest from writer |
| 4.1 | ca0f860 | OTR_BeatSelector ComfyUI node + dict/object polymorphism |
| 5.1 | da4079d | constraint-editor diagnostic stamp on meta |
| docs | 42af9c6 | handoff refresh after wire-up cascade |
| 2.2 | 99fa77f | writer halts on news-brief exhaustion |
| 4.2 | d693bb2 | insert OTR_BeatSelector into canonical workflow JSON |
| 4.3 | 7f7464a | Stage 1 fan-out helper + 3 diversity-knob prompts |
| 5.2 | 54394ac | derive_repair_prompt verdict -> Writer revision block |
| docs | 8eb61d1 | handoff refresh after full cascade |
| 5.3 | bf86ab8 | LLM-side constraint editor (schema + prompt + run) |
| 5.4 | 4c6b0b1 | flip use_constraint_editor live in run_story_room + node |
| 5.5 | 74eec12 | splice derive_repair_prompt into Writer revision turn |
| 4.4 | 2d9557a | fan_out_and_select_stage1_plan composition helper |
| 4.5 | 1044734 | wire diagnostic Stage 1 fan-out into OTR_LedgerScriptWriter |
| docs | 6c78ba7 | handoff refresh after 5.3-5.5 + 4.4-4.5 |
| 4.6 | ac2af53 | fan-out winner becomes shadow plan (dedup LLM calls) |
| 4.8 | 3719b08 | OTR_Stage1FanOut ComfyUI node wrapper |
| 4.7 | df4e38d | workflow JSON wires OTR_Stage1FanOut -> OTR_BeatSelector |
| 5.6 | ab910f1 | soak audit template doc (5 episodes x 4 runs) |
| 2.3 | b2a9ac6 | end-to-end smoke script for fan-out + constraint editor |

## Final test counts (HEAD: b2a9ac6)

| Gate                 | Baseline | Now       |
|----------------------|----------|-----------|
| pytest tests/        | 3597     | **3763**  |
| Bug Bible regression | 23/1/2x  | 23/1/2x   |
| Forbidden sweep      | runtime 0| runtime 0 |
| Workflow JSON parse  | ok       | ok (38 nodes; last_node_id 79) |
| Module imports       | ok       | ok        |
| Smoke script         | (n/a)    | exit 0 (5/5 wires) |

**166 new pytest cases** added across the full cascade. Audio
byte-identity preserved on the legacy compose path.

## Ready for QA — what the synthetic review should examine

The full pipeline is shipped behind operator flags. Every new
capability defaults OFF (except Sprint 2.2 `news_briefs_required`
which defaults True per Jeffrey's directive). The QA review's
job: catch wiring drift, schema-name drift, prompt-text issues,
and back-compat hazards before any live test.

### Files to review per sprint cluster

**Sprint 1 (dialogue_slot_id keystone)**
- `nodes/_otr_outline.py` — Beat.dialogue_slot_id field +
  stamp_dialogue_slot_ids helper, called inside _assemble_outline.
- `nodes/_otr_stage1_plan.py` — Stage1Beat.dialogue_slot_id +
  stamp_dialogue_slot_ids called from parse_and_validate_plan.
- `nodes/production_ledger.py` — init_lines_from_outline +
  set_lines preserve dialogue_slot_id on every line row.
- `nodes/_otr_story_room_extract.py` — narrow DialogueOnlySchema +
  extract_dialogue_only function.
- `nodes/OTR_StoryRoomExtract.py` — opt-in narrow path; reads
  voice_slot_ids from the in-flight ledger.
- `nodes/OTR_StoryRoomCommit.py` — _commit_dialogue joins by
  dialogue_slot_id; raises StoryRoomCommitError on mismatch.

**Sprint 2 + 2.1 + 2.2 + 2.3 (DramaticState + validators + halt)**
- `nodes/_otr_dramatic_state.py` — DramaticState pydantic +
  derive_dramatic_state_from_meta helper.
- `nodes/_otr_beat_validators.py` — validate_beat_sheet + 4
  defect kinds + _attr dict/object polymorphism.
- `nodes/OTR_LedgerScriptWriter.py` — stamps meta.dramatic_state
  + halts on NewsInterpreterError when required=True.
- `scripts/smoke_fanout_constraint.py` — end-to-end smoke
  (Sprint 2.3).

**Sprint 3 + 3.1 (arc-aware line composer)**
- `nodes/_otr_line_composer.py` — LineRequest +7 Sprint 3 fields;
  _build_user_prompt renders DRAMATIC FRAME block + no-restate
  constraint.
- `nodes/OTR_LedgerScriptWriter.py` — _build_line_request threads
  dramatic_question + next_turn into LineRequest.

**Sprint 4 + 4.1 through 4.8 (best-of-N + fan-out)**
- `nodes/_otr_beat_selector.py` — 5-axis scorer +
  select_winning_beat_sheet + NoValidBeatSheetError +
  BeatSelectorAudit.to_dict.
- `nodes/_otr_stage1_fanout.py` — fan_out_stage1_plans + 3
  diversity-knob prompts + fan_out_and_select_stage1_plan
  composition helper.
- `nodes/OTR_BeatSelector.py` — ComfyUI wrapper (Sprint 4.1).
- `nodes/OTR_Stage1FanOut.py` — ComfyUI wrapper (Sprint 4.8).
- `nodes/OTR_LedgerScriptWriter.py` — use_stage1_fanout widget +
  diagnostic stamp on meta.stage1_fanout +
  used_as_shadow_plan dedup field (Sprint 4.6).
- `workflows/otr_scifi_16gb_full.json` — OTR_Stage1FanOut at
  node id 79 wired to OTR_BeatSelector at node id 78 (Sprint
  4.2 + 4.7).

**Sprint 5 + 5.1 through 5.6 (constraint editor)**
- `nodes/_otr_editor_constraints.py` — EditorConstraint codes +
  EditorConstraintVerdict dataclass +
  EDITOR_CONSTRAINTS_SYSTEM_PROMPT (no taste verbs) +
  check_constraints (Python) + check_constraints_from_ledger +
  derive_repair_prompt + EditorConstraintVerdictSchema (LLM
  surface) + build_constraint_editor_prompt +
  run_constraint_editor.
- `nodes/_otr_story_room.py` — _call_constraint_editor adapter +
  use_constraint_editor kwarg routes the editor cycle +
  build_writer_user_prompt splices derive_repair_prompt block.
- `nodes/OTR_StoryRoom.py` — use_constraint_editor widget +
  binds editor_generate_fn to EditorConstraintVerdictSchema.
- `nodes/OTR_LedgerScriptWriter.py` — Sprint 5.1 diagnostic stamp
  via check_constraints_from_ledger on meta.editor_constraints.

### QA pass invariants to spot-check

- **PD1 (audio):** no audio-path change anywhere; the legacy
  compose path stays byte-identical when all new flags OFF.
- **PD3 (workflow JSON):** every node-side surface change has a
  matching workflow JSON edit (Sprint 4.2 + 4.7 cover the new
  nodes; existing slot-position pin tests catch widget drift).
- **PD5 ("dummy" ban):** no occurrence of the word "dummy" in
  code, comments, fixtures, or commit messages.
- **PD6 (no model_id widget on consumers):** the forbidden-
  pattern sweep enforces. New nodes:
  - OTR_BeatSelector: pure Python, no LLM slot — no model_id.
  - OTR_Stage1FanOut: technical_model is forceInput only.
- **Adapter polymorphism (Sprint 4.1):** every site that reads
  `dramatic_state` from a parsed plan uses `_attr(obj, name)`
  instead of `getattr` so dict-from-JSON works.

## Smoke command for the QA run

```cmd
cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
git pull origin v2.0-alpha
git rev-parse HEAD
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest tests/ -q --no-header
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest "C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py" -q --no-header
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe docs\_s28_forbidden_sweep.py
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe scripts\smoke_fanout_constraint.py
```

Expected output:
- HEAD: b2a9ac6 or later
- pytest: 3763 passed / 21 skipped / 0 failed
- Bug Bible: 23 passed / 1 skipped / 2 xfailed
- Forbidden sweep: HITS=410 / forensic=410 / runtime=0
- Smoke: 5/5 wires intact; exit 0

## Soak audit template

When Jeffrey is ready to test, follow `docs/2026-05-28-soak-audit-
template.md`. Capture the four audit fields × 5 episodes × 4 flag
combinations. Bring the filled-in template back so the next
session can identify what stayed in spec and what needs a fix
sprint.

---
## Resume instructions
Open a fresh window with the OTR-OldTimeRadio folder selected,
attach `session_handoff.md`, and say:

"QA review the cascade. Read every file in the 'Files to review
per sprint cluster' section. Run the smoke command. Report any
schema-name drift, wiring drift, prompt issues, or back-compat
hazards. Group findings by severity (must-fix / should-fix / nit)
so I can decide which fix sprints to run."
