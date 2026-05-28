# Session Handoff -- ComfyUI-OldTimeRadio -- 2026-05-28 (sprint cascade)

The overnight Sprint 1-5 build plan landed. All five sprints
shipped end-to-end (review -> code -> wire -> regress -> commit ->
push). HEAD == origin/v2.0-alpha. Operator next step: 5-episode
live soak per docs/2026-05-27-otr-quality-baseline.md.

## Commits shipped

| Sprint | Commit  | Title                                                      |
|--------|---------|------------------------------------------------------------|
| 0      | e3e4793 | docs: baseline + refined Sprint 1-5 plan                   |
| 1      | 2302f1e | dialogue_slot_id keystone + narrow Extract path            |
| 2      | c1ea43e | DramaticState + structural beat-sheet validators           |
| 3      | e59f32b | arc-aware line composer prompt (Path A)                    |
| 4      | 79c8014 | best-of-N beat-sheet selector                              |
| 5      | ed833d8 | editor downgrade to constraint checker                     |

HEAD: ed833d8 on v2.0-alpha (origin matches).

## Test counts

| Gate                  | Baseline | After Sprint 5 |
|-----------------------|----------|----------------|
| pytest tests/         | 3597     | **3662**       |
| Bug Bible regression  | 23 + 2x  | 23 + 2x        |
| Forbidden-pattern     | runtime 0| runtime 0      |
| Workflow JSON parse   | ok       | ok             |
| No 0-byte files       | ok       | ok             |
| No BOM in new modules | n/a      | ok             |

65 new pytest cases added across the five sprints.

## What landed (one paragraph per sprint)

**Sprint 1 -- dialogue_slot_id keystone (2302f1e).** Voiced beats
get a separate `d001..dNNN` id stamped post-parse on both
`_otr_outline.Beat` and `_otr_stage1_plan.Stage1Beat`, mirrored
onto ledger lines via `production_ledger.init_lines_from_outline`
+ `set_lines`. New narrow `extract_dialogue_only` path emits
~500-800 tokens (target 30-60 sec wall-clock vs the pre-Sprint-1
5+ min) by reusing Stage 1 plan from the in-flight ledger and only
asking the LLM for dialogue rows. `OTR_StoryRoomCommit` joins by
`dialogue_slot_id` and raises `StoryRoomCommitError` (red graph)
on slot-count or slot-order mismatch, missing slot ids, empty
text, or pre-Sprint-1 ledger shape -- no silent fallback to
legacy compose. 22 unit tests + 3 updated existing tests.

**Sprint 2 -- DramaticState + validators (c1ea43e).** New
`DramaticState` pydantic carrying `dramatic_question`,
`character_a_wants`, `character_b_wants`, `costly_choice_beat`
(d-slot id), `ending_change` -- replaces the 350-char
`script_brief` postage-stamp as the episode's reproducibility
anchor. Stage1Beat gains nine Optional fields (objective /
obstacle / turn / tactics_used / state_before / state_after /
subtext / tension / next_turn) so beats can name visible
structure. New `validate_beat_sheet(plan)` returns a list of
defect strings keyed by `DefectKind` constants (DEAD_BEAT /
NO_COSTLY_CHOICE / UNRESOLVED_COSTLY_CHOICE / UNCHANGED_ENDING).
13 unit tests.

**Sprint 3 -- arc-aware line composer (e59f32b).** Path A only.
`LineRequest` gains seven Sprint 3 fields, all empty-defaulting.
`_build_user_prompt` renders a `DRAMATIC FRAME` block in the
per-beat tail directly ABOVE the LAST SPOKEN rolling window so
the next_turn the beat must reveal sits as the magnetic pole
closest to the generation slot. Output constraint appended above
"Speak now.": "Write 1 spoken line. Do not summarize the
objective. Do not explain the turn. Perform the objective
indirectly. The situation must be different after this line."
KV-cache static prefix bytes preserved (pinned by test).
7 unit tests.

**Sprint 4 -- best-of-N selector (79c8014).** Pure-Python
mechanical selector scores N candidate plans on 5 visible-
structure axes (clear_opposed_desires / costly_choice_present /
each_beat_changes_situation / ending_changed_from_beginning /
no_alarm_countdown_rescue). Validates each candidate via Sprint
2's `validate_beat_sheet` first; only validated candidates
eligible. Picks highest-total winner with deterministic
lowest-index tie break. `NoValidBeatSheetError` raised when all
candidates fail -- audit attached to exception. 11 unit tests.

**Sprint 5 -- editor downgrade (ed833d8).** Replaces the taste-
based editor rubric with five concrete structural constraints:
`WRONG_SPEAKER`, `PHANTOM_CHARACTER`, `MISSING_COSTLY_CHOICE`,
`NO_FINAL_THIRD_TURN`, `FORMAT_FAILURE`. Pure-Python
`check_constraints(plan, ...)` returns a typed
`EditorConstraintVerdict`. New `EDITOR_CONSTRAINTS_SYSTEM_PROMPT`
strips "make it better / improve pacing / more drama / rewrite"
taste verbs. `DEFAULT_MAX_EDITOR_CONSTRAINT_CYCLES = 1` (down
from 3). 12 unit tests.

## Operator next steps

1. **5-episode live soak** with `use_story_room=true` +
   `commit=true` on the canonical workflow. For each ledger paste
   the three numbers (rows_skipped, fallback_to_legacy count,
   editor-rubber-stamp rate) into
   `docs/2026-05-27-otr-quality-baseline.md`. Sprint 1 contract:
   every episode must show
   `meta.story_room_commit.commit_mode == "dialogue_slot_order"`,
   `rows_skipped == 0`, `fallback_to_legacy == false`. Extract
   per-attempt time should drop from 5+ min to 30-60 sec.

2. If 5/5 episodes clean, run the **human A/B listen test**
   (legacy Path A vs new beat-engine + best-of-N path) per the
   plan's "Done = all five green + one human listen-test" gate.
   The structural validators are the automated regression floor;
   the listen test is the only true quality gate.

3. **Wire-up sprints (deferred, queued)** -- each is a thin
   sprint that takes the surface this cascade landed and plugs
   it into the live LLM flow:
   - **Sprint 2.1**: news_interpreter writes DramaticState into
     the ledger meta; Director brief reads it; Writer prompt
     mentions it. Plus the writer-halt-on-news-brief-exhaustion
     rule per Jeffrey 2026-05-27.
   - **Sprint 3.1**: thread Stage1Beat Sprint 2 fields into
     `LineRequest.dramatic_question / beat_* / next_turn` from
     OTR_LedgerScriptWriter's per-beat loop.
   - **Sprint 4.1**: Stage 1 fan-out (3 LLM calls with the
     diversity-knob system prompts) + new `OTR_BeatSelector`
     ComfyUI wrapper node + workflow JSON insertion.
   - **Sprint 5.1**: replace taste-based EditorVerdict in
     `run_story_room` with check_constraints; drop
     max_editor_cycles 3 -> 1.

## What I did NOT touch overnight

- Live LLM call sites in OTR_LedgerScriptWriter / Wave 0 plan
  builder / run_story_room. The wire-up sprints above are the
  right scope for those.
- `workflows/otr_scifi_16gb_full.json` -- no socket renames; all
  Sprint 1-5 surface additions are Python-side (optional fields,
  new modules). The Sprint 4.1 wire-up adds node id 78 (OTR_BeatSelector)
  to the workflow JSON.
- `_otr_critic_rubric.py` and the pre-Sprint-5 taste-based
  EditorVerdict path. Both stay alive as the back-compat surface
  while Sprint 5.1 swaps in the constraint checker.

## Tech stack & constraints

Unchanged from 2026-05-27 handoff. Branch: **v2.0-alpha ONLY**.
Git push: Desktop Commander cmd shell only. Bug Bible regression
after every code change. Audio byte-identity at every gate.
Never use the word "dummy". Stage1Beat.dialogue_slot_id stamping
predicate: `speaker != "MUSIC"` (ANNOUNCER + cast names are
voiced). _otr_outline.Beat stamping predicate: `speaker_role in
{"character", "announcer"}`. Both surfaces stamped automatically
at parse-time (parse_and_validate_plan) / assembly-time
(_assemble_outline).

---
## Resume instructions
Open a fresh window with the OTR-OldTimeRadio folder selected, attach
this file (`session_handoff.md`), and say:

"Read this handoff. Run the pytest baseline (must be 3662). Then
guide the operator through the 5-episode soak per
docs/2026-05-27-otr-quality-baseline.md. If 5/5 clean, queue the
Sprint 2.1 / 3.1 / 4.1 / 5.1 wire-up sprints."
