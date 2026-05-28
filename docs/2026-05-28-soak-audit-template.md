# Soak Audit Template — Sprint 1-5 + wire-ups + extensions

Use this template during the 5-episode soak to capture every audit
field stamped by the v2.0-alpha cascade. Three runs per episode (one
per opt-in flag set) gives a complete picture: legacy baseline,
constraint editor, fan-out, and both combined.

**HEAD covered by this template:** `df4e38d` (Sprint 4.7 wiring).

---

## Setup — one-time

Pull `v2.0-alpha`, run baseline regression:

```cmd
cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
git pull origin v2.0-alpha
git rev-parse HEAD
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest tests/ -q --no-header
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe -m pytest "C:/Users/jeffr/Documents/ComfyUI/comfyui-custom-node-survival-guide/tests/bug_bible_regression.py" -q --no-header
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe docs\_s28_forbidden_sweep.py
```

Confirm:
- HEAD == `df4e38d` or later
- pytest: 3763 passed (or higher) / 21 skipped / 0 failed
- Bug Bible: 23 passed / 1 skipped / 2 xfailed
- Forbidden sweep: `runtime: 0`

---

## Flag matrix — what to flip per episode

Five widgets on **OTR_LedgerScriptWriter**:

| Widget | Default | Role |
|---|---|---|
| `news_briefs_required` | **True** | Halts run if news briefs fail (Sprint 2.2) |
| `enable_stage1_shadow_pass` | False | Runs parallel Stage 1 audit |
| `use_stage1_fanout` | False | 3-candidate diversity-knob fan-out (Sprint 4.5 / 4.6) |
| `use_multiturn_dialogue` | False | Wave 0 Stage 2 multi-turn |
| `enable_production_stage3_validators` | False | Stage 3 validators in compose |

Two widgets on **OTR_StoryRoom**:

| Widget | Default | Role |
|---|---|---|
| `use_story_room` | False | Story Room loop runs (Wave 2) |
| `use_constraint_editor` | False | Constraint editor + 1-cycle cap (Sprint 5.4 / 5.5) |

One widget on **OTR_StoryRoomCommit**:

| Widget | Default | Role |
|---|---|---|
| `commit` | False | Story Room dialogue commits to ledger (Sprint 1) |

---

## The four runs per episode

For each of 5 episodes, run **four times** with these flag sets:

### Run A — Baseline (Sprint 1 only)
- `use_story_room=true` + `commit=true` + `news_briefs_required=true`
- All other Sprint 4/5 flags OFF
- Measures Sprint 1 keystone in isolation.

### Run B — Constraint editor on
- All of A, plus `use_constraint_editor=true`
- Measures the Sprint 5.4 swap + Sprint 5.5 Writer revision
  splice in isolation.

### Run C — Fan-out on
- All of A, plus `enable_stage1_shadow_pass=true` +
  `use_stage1_fanout=true`
- Measures the Sprint 4.5 / 4.6 fan-out in isolation.

### Run D — Both
- All of A + B + C
- Measures the full Sprint 4 + 5 cascade.

---

## Capture template — copy this block per episode per run

```
========================================
Episode: <slug>
Run: A | B | C | D
ledger path: <output\pending_*\ledger.json>
========================================

----- Sprint 1 commit audit -----
meta.story_room_commit.commit_mode:        <dialogue_slot_order | (absent)>
meta.story_room_commit.draft_rows:         <N>
meta.story_room_commit.voice_slots:        <N>
meta.story_room_commit.rows_committed:     <N>
meta.story_room_commit.rows_skipped:       <0 expected on PASS>
meta.story_room_commit.fallback_to_legacy: <false expected on PASS>
meta.story_room_commit.committed_slot_ids: <d001, d002, ...>
Extract per-attempt wall-clock: <secs; pass = <60s on the narrow path>

----- Sprint 2.1 DramaticState -----
meta.dramatic_state.dramatic_question:    <text | (absent)>
meta.dramatic_state.character_a_wants:    <text>
meta.dramatic_state.character_b_wants:    <text>
meta.dramatic_state.costly_choice_beat:   <d-id>
meta.dramatic_state.ending_change:        <text>

----- Sprint 2.2 news-brief halt -----
meta.news_briefs_halt_reason:             <(absent on success) | <error>>
news_interpreter attempts logged:         <N>

----- Sprint 4.5 / 4.6 fan-out -----
meta.stage1_fanout.ok:                    <true | false | (absent)>
meta.stage1_fanout.winner_knob:           <moral_dilemma | bureaucratic_absurd | intimate_personal_cost | null>
meta.stage1_fanout.used_as_shadow_plan:   <true on Sprint 4.6 dedup>
meta.stage1_fanout.per_knob_ok:           <list of {knob, ok, error}>
meta.stage1_fanout.selector_audit.winner_index:    <0..2>
meta.stage1_fanout.selector_audit.scores_per_candidate[winner].total: <0..5>
meta.stage1_fanout.error:                 <empty on success>
Stage 1 LLM call count (from console):    <3 on success; 4 on fail-through>

----- Sprint 5.1 / 5.4 / 5.5 constraint editor -----
meta.editor_constraints.pass_decision:    <true | false>
meta.editor_constraints.failing_constraints: <list of 5-code strings>
meta.editor_constraints.repair_note:      <text>
meta.editor_constraints.cycle:            <0 | 1>
Editor wall-clock per cycle:              <secs; constraint editor target <30s>
Writer revision turns observed:           <0 | 1; constraint editor cap = 1>

----- Audio integrity (PD1) -----
Final audio byte-identical to baseline (Run A)? <yes | no>
If no: hash diff vs Run A baseline:       <sha256 head>

----- Operator listen note -----
1-3 sentences, free-form. Did the new path produce a stronger
episode? What broke? What surprised?

```

---

## Aggregate report — fill at end of soak

| Metric | Sprint 1 baseline (Run A x5) | Constraint editor (Run B x2) | Fan-out (Run C x2) | Both (Run D x2) |
|---|---|---|---|---|
| Total runs | 5 | 2 | 2 | 2 |
| `rows_skipped > 0` count | 0 expected | 0 expected | 0 expected | 0 expected |
| `fallback_to_legacy=true` count | 0 expected | 0 expected | 0 expected | 0 expected |
| `dramatic_state` populated | 5 expected | 2 expected | 2 expected | 2 expected |
| `editor_constraints.pass_decision=true` | varies | report | varies | report |
| `stage1_fanout.ok=true` | absent | absent | 2 expected | 2 expected |
| `stage1_fanout.used_as_shadow_plan=true` (Sprint 4.6) | absent | absent | 2 expected | 2 expected |
| Avg Extract per-attempt wall-clock | <60s target | <60s | <60s | <60s |
| Avg editor wall-clock per cycle | ~60-90s baseline | <30s target | ~60-90s | <30s target |
| `news_briefs_halt_reason` count (red graph) | 0 unless news fails | 0 | 0 | 0 |

---

## Diversity-knob audit (Run C + D only)

For each episode where the fan-out was on, examine
`meta.stage1_fanout.per_knob_ok` and capture:

- How many knobs produced a valid plan (out of 3)?
- Were the three plans structurally DIFFERENT?
  - Different cast?
  - Different beat counts?
  - Different `dramatic_state.dramatic_question`?
  - Different `dramatic_state.costly_choice_beat` slot?
- Does the operator's reading agree with the selector's pick
  (winner_index)?
- Did any one knob consistently fail / produce malformed JSON?

A passing soak: knobs MORAL_DILEMMA, BUREAUCRATIC_ABSURD, and
INTIMATE_PERSONAL_COST each produce structurally distinct candidate
plans on at least 2 of the 2 episodes. If one knob always loses,
that's a prompt-tuning sprint candidate.

---

## Constraint editor wall-clock A/B (Run A vs B)

For the same episode run under A then B, capture:

- Editor cycle count: A typically 1-3, B always 1.
- Total editor wall-clock: A 60-300s, B target <60s.
- Writer revision turn count: A 0-3, B 0-1.
- Episode word count delta vs A: <±10% acceptable; >10% indicates
  the constraint editor cut quality somehow.
- Operator listen test: which sounds tighter? Which sounds more
  formulaic?

---

## Failure modes to watch for

- **`StoryRoomCommitError` red graph (Sprint 1):** slot count or
  order mismatch. Operator regenerates Stage 1 before retrying.
- **`news_briefs_halt_reason` non-empty (Sprint 2.2):** news pull
  failed. Operator re-queues; RSS pulls fresh on each queue.
- **`stage1_fanout.ok=false` with `error` field (Sprint 4.5/6):**
  fan-out itself failed; legacy single-call shadow pass ran
  through. Inspect `error` for per-knob failure mode (LLM raise vs
  parse failure vs all-validate-fail).
- **`editor_constraints.failing_constraints` containing the same
  code on every soak episode (Sprint 5.1):** the constraint is
  flagging legitimate operator content; either the rule is wrong
  or the Writer is producing a structural pattern that needs the
  Sprint 4 selector to filter out.
- **Audio byte-identity broken on Run A:** Sprint 1's PD1 contract
  is violated. Open a Bug Bible candidate immediately.

---

## Resume after soak

Bring the filled-in template back. The next sprint cascade (the
synthetic QA review) reads from this doc to identify which
behaviors stayed in spec and which need a fix sprint.
