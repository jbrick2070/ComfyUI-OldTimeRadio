# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-28 (Sprint 1-5 + wire-ups)

All five build-plan sprints AND all four wire-up sprints shipped
end-to-end on v2.0-alpha. HEAD == origin/v2.0-alpha at `da4079d`.
Episode generation now stamps DramaticState + editor-constraint
verdict + Sprint 1 dialogue_slot_id audit on every ledger. Ready
for the operator's 5-episode live soak.

## Commits shipped

| Sprint | Commit  | Title                                                      |
|--------|---------|------------------------------------------------------------|
| 0      | e3e4793 | docs: baseline + refined Sprint 1-5 plan                   |
| 1      | 2302f1e | dialogue_slot_id keystone + narrow Extract path            |
| 2      | c1ea43e | DramaticState + structural beat-sheet validators           |
| 3      | e59f32b | arc-aware line composer prompt (Path A)                    |
| 4      | 79c8014 | best-of-N beat-sheet selector                              |
| 5      | ed833d8 | editor downgrade to constraint checker                     |
| docs   | eb8fdcd | session_handoff refresh after Sprint 1-5 cascade           |
| 2.1    | a0a175d | stamp DramaticState on meta after Stage 1 keystone         |
| 3.1    | e6a433f | thread DRAMATIC FRAME fields into LineRequest from writer  |
| 4.1    | ca0f860 | OTR_BeatSelector ComfyUI node + dict/object polymorphism   |
| 5.1    | da4079d | constraint-editor diagnostic stamp on meta                 |

HEAD: da4079d on v2.0-alpha (origin matches).

## Test counts

| Gate                 | Baseline | After wire-ups |
|----------------------|----------|----------------|
| pytest tests/        | 3597     | **3685**       |
| Bug Bible regression | 23/1/2x  | 23/1/2x        |
| Forbidden sweep      | runtime 0| runtime 0      |
| Workflow JSON parse  | ok       | ok             |
| Module imports       | ok       | ok             |

88 new pytest cases added across the cascade (65 in Sprints 1-5,
23 in the four wire-ups: 10 + 0 + 8 + 5).

## What the wire-ups did

**Sprint 2.1 (a0a175d) -- DramaticState on every ledger.**
OTR_LedgerScriptWriter now calls `derive_dramatic_state_from_meta`
immediately after `init_lines_from_outline` and stamps
`meta["dramatic_state"]` from news_interpreter brief + locked cast
+ Sprint 1 voice slot ids. Helper picks the second-to-last voiced
slot as the default costly_choice_beat. Wrapped in try/except so
any pydantic edge case logs + skips rather than breaking the
writer.

**Sprint 3.1 (e6a433f) -- DRAMATIC FRAME plumbed into the per-beat
prompt.** The per-beat LineRequest builder in
OTR_LedgerScriptWriter pulls `dramatic_question` from
`meta["dramatic_state"]` and `next_turn` from the next voiced
outline beat's intent. Both fields default empty in LineRequest so
legacy episodes (no DramaticState stamped) produce byte-identical
pre-Sprint-3 prompts; the KV-cache static prefix invariant from
Sprint 3 holds.

**Sprint 4.1 (ca0f860) -- OTR_BeatSelector node registered.** New
ComfyUI node `OTR_BeatSelector` accepts up to 3 candidate
Stage1Plan JSON inputs and emits the winning plan + a selector
audit JSON. Pure Python -- no LLM call, no model_id widget (PD6).
Also fixed a polymorphism bug across the Sprint 2/4/5 modules:
`Stage1Plan.dramatic_state` is `Optional[Any]` so a JSON round-
trip carries it as a dict, not a DramaticState; added `_attr(obj,
name, default)` to read both shapes uniformly. Workflow JSON
insertion deferred (operator wires the new node into the
canonical workflow manually).

**Sprint 5.1 (da4079d) -- diagnostic editor_constraints stamp.**
OTR_LedgerScriptWriter now stamps `meta["editor_constraints"]`
with a Sprint 5 check_constraints verdict reconstructed from the
in-flight ledger (cast + lines + dramatic_state). The live taste-
rubric editor cycle is unchanged this sprint -- the stamp lets
operators see the constraint signal in every soak ledger so the
follow-up sprint that swaps in the constraint editor at
max_editor_cycles=1 has a forensic baseline.

## Operator next steps

1. **5-episode live soak** with `use_story_room=true` +
   `commit=true` on the canonical workflow. For each ledger
   inspect:
   - `meta.story_room_commit.commit_mode == "dialogue_slot_order"`
     and `rows_skipped == 0` and `fallback_to_legacy == false`
     (Sprint 1 contract).
   - `meta.dramatic_state` is populated with the four required
     DramaticState fields (Sprint 2.1).
   - `meta.editor_constraints.pass_decision` and
     `failing_constraints` -- diagnostic only this sprint, but the
     signal tells the next sprint what the constraint editor would
     have done if it were live (Sprint 5.1).
   - Extract per-attempt time logged in console should drop from
     5+ min to 30-60 sec on the narrow path.

2. **OTR_BeatSelector workflow wiring (optional).** The node is
   registered; drag it into the saved workflow JSON between the
   Stage 1 fan-out and the downstream consumer to opt into
   best-of-N. Without the fan-out (still pending its own sprint)
   the node accepts a single candidate and ships it after
   structural validation.

3. **Human A/B listen test.** Plan's "Done = all five green + one
   human listen test" gate. Legacy Path A vs new beat-engine +
   best-of-N path. Ship the new path as default only when it wins
   the listen test.

## Open follow-up sprints (queued, not blocking soak)

- **Stage 1 fan-out**: 3 LLM calls at the Stage 1 surface with
  diversity-knob system prompts (moral-dilemma / bureaucratic-
  absurd / intimate personal-cost) wired through OTR_BeatSelector.
  Touches OTR_LedgerScriptWriter outline call.
- **Constraint editor live swap**: replace the taste-rubric
  EditorVerdict in `run_story_room` with EditorConstraintVerdict
  + drop max_editor_cycles 3 -> 1 + one targeted repair turn.
  Touches the Story Room control flow; benefits from operator
  soak feedback on the diagnostic stamps first.
- **Writer halt on news-brief exhaustion** (Jeffrey 2026-05-27):
  Sprint 2 carryover. When `build_news_briefs` exhausts retries,
  the writer must halt or re-roll news rather than continuing with
  meta["news"] = None.
- **OTR_BeatSelector workflow JSON insertion (4.2)**: edit
  `workflows/otr_scifi_16gb_full.json` to add node id 78 wired
  between Stage 1 and the downstream consumer. Operator-driven
  this sprint; the node is already registered + importable.

## Tech stack & constraints

Unchanged. Branch: v2.0-alpha only. Git push: Desktop Commander
cmd shell only. Bug Bible regression after every code change.
Audio byte-identity at every gate. Never use the word "dummy".
Stage1Beat.dialogue_slot_id stamping predicate: speaker != "MUSIC"
(ANNOUNCER + cast names are voiced). _otr_outline.Beat stamping
predicate: speaker_role in {"character", "announcer"}. All Sprint
2 typed-state fields on Stage1Beat are Optional this sprint;
producers wire them as part of follow-up sprints.

---
## Resume instructions
Open a fresh window with the OTR-OldTimeRadio folder selected, attach
this file (`session_handoff.md`), and say:

"Read this handoff. Confirm pytest baseline 3685. Then guide the
5-episode soak per docs/2026-05-27-otr-quality-baseline.md and the
operator-checklist above. Report back the three numbers
(rows_skipped, dramatic_state populated, editor_constraints
pass/fail) per episode."
