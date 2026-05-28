# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-28 (full cascade)

All five build-plan sprints, all four wire-up sprints, AND all four
follow-up sprints shipped end-to-end on v2.0-alpha. HEAD ==
origin/v2.0-alpha at `54394ac`. Ready for the operator's 5-episode
live soak.

## Commits shipped (16 total)

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
| docs   | 42af9c6 | session_handoff refresh after wire-up cascade              |
| 2.2    | 99fa77f | writer halts on news-brief exhaustion (Jeffrey directive)  |
| 4.2    | d693bb2 | insert OTR_BeatSelector into canonical workflow JSON       |
| 4.3    | 7f7464a | Stage 1 fan-out helper + three diversity-knob prompts      |
| 5.2    | 54394ac | derive_repair_prompt -- verdict -> Writer revision block   |

HEAD: 54394ac on v2.0-alpha (origin matches).

## Test counts

| Gate                 | Baseline | Now       |
|----------------------|----------|-----------|
| pytest tests/        | 3597     | **3710**  |
| Bug Bible regression | 23/1/2x  | 23/1/2x   |
| Forbidden sweep      | runtime 0| runtime 0 |
| Workflow JSON parse  | ok       | ok        |

113 new pytest cases added across the cascade. Audio byte-identity
preserved (no live-flow change to legacy compose path).

## What each follow-up sprint did

**Sprint 2.2 (99fa77f) -- writer halts on news-brief exhaustion.**
Per Jeffrey 2026-05-27 directive ("the whole workflow needs to
stop and re-roll news until it works"). The
`news_briefs_required` BOOLEAN widget defaults True; on
`NewsInterpreterError` the writer stamps
`meta["news_briefs_halt_reason"]` + saves + re-raises (red graph).
Operator re-queues; news_interpreter pulls fresh from RSS each
queue, so the re-queue IS the re-roll. Set False for the legacy
graceful-degrade back-compat surface.

**Sprint 4.2 (d693bb2) -- OTR_BeatSelector placed in workflow JSON.**
Detached node at id 78 in `workflows/otr_scifi_16gb_full.json`.
Inputs link=None, outputs links=[] so live behavior is unchanged
until the operator wires it manually (once Sprint 4.3 fan-out
runs in production). 6 invariant tests pin the node's presence +
detachment.

**Sprint 4.3 (7f7464a) -- Stage 1 fan-out helper + diversity knobs.**
`nodes/_otr_stage1_fanout.py` ships `fan_out_stage1_plans(...)` +
three canonical diversity-knob prompt prefixes
(MORAL_DILEMMA / BUREAUCRATIC_ABSURD / INTIMATE_PERSONAL_COST).
Sequential by design (16 GB VRAM ceiling). Per-knob generate/
parse failures stay isolated; other knobs still run. The
integration into OTR_LedgerScriptWriter's outline call is the
next follow-up; the helper ships fully tested standalone (9
cases).

**Sprint 5.2 (54394ac) -- derive_repair_prompt bridge.** Pure-
Python helper that turns an EditorConstraintVerdict into a single
REVISE block for one targeted Writer revision turn
(MAX_REPAIR_TURNS=1). Names each failing constraint with a
concrete repair header; appends the verdict's repair_note. The
final run_story_room loop swap (Sprint 5.3) becomes a thin
integration sprint with this helper + Sprint 5.1's
`check_constraints_from_ledger` already in place.

## What episode generation does now

Every Story Room run produces a ledger carrying:
  * `meta.story_room_commit` (Sprint 1; only when commit=True):
    commit_mode + draft_rows + voice_slots + rows_committed +
    rows_skipped + fallback_to_legacy + committed_slot_ids.
  * `meta.dramatic_state` (Sprint 2.1): the four required
    DramaticState fields keyed off news_interpreter brief + cast.
  * `meta.editor_constraints` (Sprint 5.1): pass_decision +
    failing_constraints + repair_note + cycle -- diagnostic only
    this sprint.
  * `meta.news_briefs_halt_reason` (Sprint 2.2; only on halt):
    type + message of the NewsInterpreterError that triggered
    the red graph.

And the OTR_BeatSelector ComfyUI node is registered + placed in
the canonical workflow JSON, ready to wire up.

## Operator next steps

1. **5-episode live soak** with `use_story_room=true` +
   `commit=true` (Sprint 1) + `news_briefs_required=true`
   (Sprint 2.2, the default) on the canonical workflow.
2. For each ledger, capture:
   - `meta.story_room_commit.rows_skipped == 0` AND
     `fallback_to_legacy == false` (Sprint 1 contract).
   - `meta.dramatic_state` populated with all four fields
     (Sprint 2.1).
   - `meta.editor_constraints.failing_constraints` -- diagnostic;
     tells you what the constraint editor would have flagged.
3. **Extract per-attempt time** should drop from 5+ min to
   30-60 sec (Sprint 1 narrow path).
4. **OTR_BeatSelector wire-in (optional)**: drag connections from
   Stage 1 outputs into the BeatSelector's candidate inputs and
   the winning_plan_json output to the downstream consumer once
   the Sprint 4.3 fan-out lands in production.
5. **Human A/B listen test**: legacy Path A vs the new beat-engine
   + best-of-N path. Ship the new path as default only when it
   wins the listen test.

## Remaining queued sprints (all behind operator soak)

- **Sprint 4.4 -- Stage 1 fan-out integration in writer.** Call
  fan_out_stage1_plans from OTR_LedgerScriptWriter's outline
  path; feed candidates to OTR_BeatSelector; use winner as the
  outline. Opt-in flag for back-compat.
- **Sprint 5.3 -- constraint editor live swap in run_story_room.**
  Replace the taste editor LLM call with check_constraints_from
  _ledger + derive_repair_prompt splice. Drop max_editor_cycles
  3 -> 1. Opt-in flag for back-compat.
- **Sprint 4.5 -- workflow JSON full wiring.** Connect the
  detached OTR_BeatSelector to Stage 1 fan-out outputs and the
  downstream Writer once Sprint 4.4 lands.

Each is a small integration sprint with pure-Python helpers
already in place + the operator's soak data informing the
behavior of the live path.

## Tech stack & constraints

Unchanged. Branch: v2.0-alpha only. Git push: Desktop Commander
cmd shell only. Bug Bible regression after every code change.
Audio byte-identity at every gate. Never use the word "dummy".
All new modules are pure Python (pydantic + stdlib) at import
time -- no torch / no GPU dependency until the LLM call sites
that the writer plugs in. The constraint-editor + selector +
fan-out helpers all duck-type their inputs (dict OR object) so
JSON round-trips through the ledger work transparently.

---
## Resume instructions
Open a fresh window with the OTR-OldTimeRadio folder selected, attach
this file (`session_handoff.md`), and say:

"Read the handoff. Confirm pytest baseline 3710 + workflow JSON
parses + Bug Bible green. Then guide the 5-episode soak per
docs/2026-05-27-otr-quality-baseline.md and report the
meta.story_room_commit / meta.dramatic_state / meta.editor_
constraints / meta.news_briefs_halt_reason fields per episode."
