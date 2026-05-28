# Sprint 0 Baseline -- 2026-05-27

Captured during the autonomous overnight build run. Per Sprint 0
contract: measurement only, no code.

## Scope of this baseline

The mounted workspace at the time of capture exposed the repo working
tree but NO live `output/pending_*` or `output/episodes/*.json` runs.
The most recent inspectable ledger artifact in the repo is
`docs/2026-05-19-first-stable-run-ledger.json` -- pre-Wave-3, predates
the StoryRoom / Extract / Commit nodes (`meta.story_room_commit`
returns `None`).

Sprint 1's live-soak verification (5 episodes with
`use_story_room=true` + `commit=true`) is the operator's gate -- it
captures the rows_committed / rows_skipped numbers against the new
slot-id code path. The pre-Sprint-1 baseline numbers from the existing
runs reside on the operator's machine (paths
`pending_20260527_222846`, `pending_20260527_223452`, plus older
runs) but are not present in this session's mount.

## What I CAN observe from the repo

- The latest committed ledger schema (`docs/2026-05-19-first-stable-run-ledger.json`):
  - Top-level keys include `beats`, `lines`, `cast`, `meta`.
  - `lines[*]` carries `line_id`, `beat_id`, `char_id`, `text`,
    `speaker_role`, `arc_phase`, plus audio + timing fields.
  - `meta` has NO `story_room_commit`, NO `story_room_extract`, NO
    `story_room_editor_verdicts` -- this run predates Wave 3.
  - Line count: 5 (single-shot test ledger).
- The schema design above is the surface Sprint 1 surgery targets:
  the new `dialogue_slot_id` column will mirror onto `lines[*]` so the
  StoryRoomCommit join can resolve voiced rows in O(1) by slot id
  rather than guessing from raw `beat_id` order.

## What the operator must capture on the morning soak

For each of the 5 Sprint 1 soak episodes, paste these three numbers
into this file (or a successor `docs/2026-05-28-otr-quality-baseline.md`):

1. **Skipped row count** (the bug being fixed):
   `meta.story_room_commit.rows_skipped` summed across the 5 episodes.
   Pre-Sprint-1 expectation: > 0. Post-Sprint-1 expectation: exactly 0.
2. **Commit fallback rate**:
   Count of episodes where
   `meta.story_room_commit.fallback_to_legacy == true`. Pre-Sprint-1
   expectation: > 0. Post-Sprint-1 expectation: exactly 0 (Sprint 1
   removes the silent legacy fallback).
3. **Editor rubber-stamp rate**:
   Count of episodes where
   `meta.story_room_editor_verdicts[0].pass_decision == true` AND
   `len(meta.story_room_editor_verdicts) == 1`. The current Editor is
   a 3-cycle taste editor; Sprint 5 caps it at 1 cycle and strips
   taste. Pre-Sprint-5 rubber-stamp rate is the comparison baseline for
   when Sprint 5 lands.

## Decision: Sprint 1 priority

Per the round-robin consensus in
`docs/OTR_story_quality_build_plan.md`, Sprint 1 (dialogue_slot_id +
extract scope reduction) is the highest-priority sprint regardless of
the rows_skipped count, because:

- The handoff documents an observed 5+ min Extract attempt
  (`pending_20260527_223452`) under the full `StoryRoomExtractionSchema`,
  which Sprint 1's narrow `DialogueOnlySchema` drops to 30-60 sec.
- The Wave 3 commit currently joins by raw `beat_id`, which the audit
  confirmed has zero defensive checks against count or order
  mismatch -- silently dropping rows is mechanically possible today.

Sprint 1 lands the keystone whether or not the pre-Sprint baseline
shows skipped rows. The post-Sprint metric is rows_skipped == 0 across
the 5-episode soak, captured by the operator and stamped here on
2026-05-28.

## Operator action on Sprint 1 commit + soak

1. Pull `v2.0-alpha` after the Sprint 1 commit lands.
2. Run 5 episodes in ComfyUI Desktop with `use_story_room=true` +
   `commit=true` on the canonical workflow.
3. For each ledger, paste the three numbers above into this file.
4. If any episode shows `rows_skipped > 0` or
   `fallback_to_legacy == true`: revert Sprint 1 and open a Bug Bible
   candidate. Do NOT proceed to Sprint 2 until 5/5 clean.
