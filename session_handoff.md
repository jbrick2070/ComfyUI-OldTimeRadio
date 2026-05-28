# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-28 (full cascade + follow-ups)

All five build-plan sprints, all four wire-up sprints, all four
follow-up sprints, AND the four extension sprints (5.3 / 5.4 / 5.5
+ 4.4 / 4.5) shipped end-to-end on v2.0-alpha. HEAD ==
origin/v2.0-alpha at `1044734`. Ready for the operator's 5-episode
live soak.

## Commits shipped (21 total)

| Sprint | Commit | Title |
|---|---|---|
| 0 | e3e4793 | docs: baseline + refined Sprint 1-5 plan |
| 1 | 2302f1e | dialogue_slot_id keystone + narrow Extract path |
| 2 | c1ea43e | DramaticState + structural beat-sheet validators |
| 3 | e59f32b | arc-aware line composer prompt (Path A) |
| 4 | 79c8014 | best-of-N beat-sheet selector |
| 5 | ed833d8 | editor downgrade to constraint checker |
| docs | eb8fdcd | session_handoff refresh after Sprint 1-5 cascade |
| 2.1 | a0a175d | stamp DramaticState on meta after Stage 1 keystone |
| 3.1 | e6a433f | thread DRAMATIC FRAME fields into LineRequest from writer |
| 4.1 | ca0f860 | OTR_BeatSelector ComfyUI node + dict/object polymorphism |
| 5.1 | da4079d | constraint-editor diagnostic stamp on meta |
| docs | 42af9c6 | session_handoff refresh after wire-up cascade |
| 2.2 | 99fa77f | writer halts on news-brief exhaustion (Jeffrey directive) |
| 4.2 | d693bb2 | insert OTR_BeatSelector into canonical workflow JSON |
| 4.3 | 7f7464a | Stage 1 fan-out helper + three diversity-knob prompts |
| 5.2 | 54394ac | derive_repair_prompt -- verdict -> Writer revision block |
| docs | 8eb61d1 | session_handoff refresh after full cascade |
| 5.3 | bf86ab8 | LLM-side constraint editor surface (schema + prompt + run) |
| 5.4 | 4c6b0b1 | flip use_constraint_editor live in run_story_room + node |
| 5.5 | 74eec12 | splice derive_repair_prompt into Writer revision turn |
| 4.4 | 2d9557a | fan_out_and_select_stage1_plan composition helper |
| 4.5 | 1044734 | wire diagnostic Stage 1 fan-out into OTR_LedgerScriptWriter |

HEAD: 1044734 on v2.0-alpha (origin matches).

## Test counts

| Gate                 | Baseline | Now       |
|----------------------|----------|-----------|
| pytest tests/        | 3597     | **3751**  |
| Bug Bible regression | 23/1/2x  | 23/1/2x   |
| Forbidden sweep      | runtime 0| runtime 0 |
| Workflow JSON parse  | ok       | ok        |

154 new pytest cases added across the cascade. Audio byte-identity
preserved on the legacy compose path.

## Operator opt-in matrix

Every new capability ships behind a default-False flag (except
Sprint 2.2 which defaults True per Jeffrey's directive). All can
be flipped independently from the canonical workflow:

| Widget on OTR_LedgerScriptWriter | Default | Effect when ON |
|---|---|---|
| `news_briefs_required` (2.2) | **True** | NewsInterpreterError halts the writer (red graph) |
| `enable_stage1_shadow_pass` | False | Parallel Stage 1 audit (existing) |
| `use_stage1_fanout` (4.5) | False | Parallel 3-candidate fan-out audit; requires shadow pass ON |
| `use_multiturn_dialogue` | False | Wave 0 multi-turn dialogue dispatch |
| `enable_production_stage3_validators` | False | Stage 3 validators in legacy composer |

| Widget on OTR_StoryRoom | Default | Effect when ON |
|---|---|---|
| `use_story_room` | False | Story Room writers' room loop runs |
| `use_constraint_editor` (5.4) | False | Constraint editor + 1-cycle cap + per-code Writer repair block |

| Widget on OTR_StoryRoomCommit | Default | Effect when ON |
|---|---|---|
| `commit` (1.0) | False | Story Room dialogue commits to ledger (red graph on slot mismatch) |

## What episode generation stamps now

Per ledger, depending on flags:

- `meta.story_room_commit` (Sprint 1; when commit=True): proof
  block with commit_mode / draft_rows / voice_slots /
  rows_committed / rows_skipped / fallback_to_legacy /
  committed_slot_ids.
- `meta.dramatic_state` (Sprint 2.1): the four required
  DramaticState fields keyed off news_interpreter brief + cast.
- `meta.editor_constraints` (Sprint 5.1): pass_decision +
  failing_constraints + repair_note + cycle (diagnostic).
- `meta.news_briefs_halt_reason` (Sprint 2.2, on halt): the
  exception summary if news_briefs_required halted the run.
- `meta.stage1_fanout` (Sprint 4.5; when use_stage1_fanout +
  shadow pass ON): {ok, winner_knob, per_knob_ok,
  selector_audit, error} from the 3-candidate diversity-knob
  fan-out pass.

## The structural-quality toolchain end-to-end

**Ceiling (Sprint 4 line):**
`fan_out_and_select_stage1_plan` runs 3 candidate Stage 1 calls
with moral-dilemma / bureaucratic-absurd / intimate-personal-cost
system-prompt prefixes, validates each via Sprint 2 structural
validators, scores on five visible-structure axes
(clear_opposed_desires / costly_choice_present /
each_beat_changes_situation / ending_changed / no_alarm_pattern),
picks the highest-total eligible candidate (lowest-index tie
break). Diagnostic pass lives on
`meta.stage1_fanout` until the operator validates the knobs.

**Floor (Sprint 5 line):**
OTR_StoryRoom's `use_constraint_editor=True` rebinds the editor
LLM to `EditorConstraintVerdictSchema` (Literal[5 codes]), forces
`max_editor_cycles=1`, and the Writer revision turn renders
`derive_repair_prompt`'s per-code repair instructions instead of
taste-rubric notes. The OFF path is byte-identical to pre-Sprint-5
taste editor.

## Operator next steps

1. **5-episode soak** with `use_story_room=true` + `commit=true`
   (Sprint 1) + `news_briefs_required=true` (Sprint 2.2, default).
2. For each ledger, capture:
   - `meta.story_room_commit.rows_skipped == 0` AND
     `fallback_to_legacy == false`
   - `meta.dramatic_state` populated with all four fields
   - `meta.editor_constraints.failing_constraints` (diagnostic)
3. **Constraint editor A/B:** rerun two of the five episodes with
   `use_constraint_editor=true`; compare:
   - Editor wall-clock (taste editor 60-90 sec per cycle vs
     constraint editor <30 sec; cap 1 cycle vs 3)
   - Listen test: does the constraint editor's per-code Writer
     repair (Sprint 5.5) produce a tighter revision than the
     taste editor's overall_note?
4. **Diversity-knob A/B:** rerun two episodes with
   `enable_stage1_shadow_pass=true` + `use_stage1_fanout=true`.
   Inspect `meta.stage1_fanout.per_knob_ok` -- did all three knobs
   produce a valid Stage1Plan? Inspect
   `meta.stage1_fanout.selector_audit.scores_per_candidate`
   side-by-side -- do the three knobs produce structurally
   different candidates, or do they all look the same?
5. **Human A/B listen test** between legacy + new beat-engine +
   constraint-editor path. Ship the new path as default only when
   it wins.

## Remaining queued sprints

- **Sprint 4.6 -- fan-out swap into outline path.** Once the
  operator validates the diversity knobs produce structurally
  different candidates (Sprint 4.5 stamps), swap the WINNER of
  the fan-out into the live outline path (replaces
  `generate_outline` or feeds its winner). Touches
  OTR_LedgerScriptWriter's outline call surface. Operator-soak-
  driven.
- **Sprint 4.7 -- OTR_BeatSelector node fully wired.** Once the
  fan-out drives the outline, the detached OTR_BeatSelector
  workflow node (Sprint 4.2) becomes the canonical drag-and-drop
  for the fan-out path.

The pure-Python + LLM-side surfaces for these are all shipped;
each follow-up sprint becomes a thin integration job once the
operator's soak data informs the right defaults.

---
## Resume instructions
Open a fresh window with the OTR-OldTimeRadio folder selected,
attach `session_handoff.md`, and say:

"Read the handoff. Confirm pytest baseline 3751 + Bug Bible green
+ workflow JSON parses. Then guide the operator's 5-episode soak +
the Sprint 5.4 constraint-editor A/B + the Sprint 4.5 diversity-
knob A/B. Report the four audit fields (story_room_commit /
dramatic_state / editor_constraints / stage1_fanout) per episode."
