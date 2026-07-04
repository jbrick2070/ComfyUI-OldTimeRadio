# r4 FINAL -- rip-sfx-broll build plan CONVERGED (2026-07-01)

Panel r4: driver anchor + codex (gpt-5.5) + claude-code (sonnet, re-run after a
harness-kill; it reviewed the repo MID-BUILD so its two "must-fixes" were
already-resolved states, see judgment). agy dropped (credit timeout, operator
directive). CONVERGENCE REACHED -- no new must-fix stands. Build proceeds from
docs/2026-07-01-rip-sfx-broll/BUILD_PLAN.md (v2) plus the r4 folds below.

## r4 folds (all grounded)
1. codex MF1: NO engine declares character_video in default_roles (grounded:
   eng_humo default_roles=(), only ltx_av declares music_visual+announcer_visual).
   Guard test (g) REPLACED: assert engines_for_role(role) NON-EMPTY for all 3
   roles + the canonical workflow node 87 pins character_video_model to a
   REGISTERED engine (the workflow pin is the character default, not the
   registry). No registry-default promotion (that would be new policy).
2. codex SF1: rename profile_keys_for_all_five -> profile_keys_for_all_roles
   (done in the slot_matrix rewrite); grep for callers.
3. codex SF2: tests/fixtures/ledger_stub.py (with_sfx param + sfx rows) +
   tests/test_full_workflow_v2_audio_wiring.py (sfx_audio_clips pins) added to
   the explicit test-rewrite list.
4. claude SF1: guard test imports default-engine helpers from
   nodes/_otr_video_engines/registry.py.
5. claude SF2: otr_shot_lock._DEFAULT_VIDEO_ROLE deletion + the
   otr_meta_brief_image_prompt lazy-import drop land in the SAME change (no
   interim broken import).
6. claude SF3: explicit pre-commit grep of scripts/ + tests/ for FIVE_ROLES /
   build_all_five / scene_broll / background_abstract / sfx leftovers.
7. claude verify-b CONFIRMED: node 87 post-rip widget count = 12 required + 3
   optional non-forceInput (gate_in forceInput=0) = 15 == the planned array.

## Resolved conflicts
- claude CUT2 said scripts/_otr_patch_pool_default.py does not exist; the
  driver's direct directory listing AND codex r3 (with content line numbers)
  both show it. DRIVER RULING: file exists; DELETE stands. (The agent's grep
  likely ran in a wrong cwd.)
- claude MF1/MF2 reviewed mid-build state: the speaker_role rewrite already
  removed SPEAKER_ROLE_SFX + its frozenset member + docstrings atomically; the
  "pre-applied outline edits" are this build's own edits, not stale plan lines.

Stop at convergence. No r5. Execute + green-gate + one atomic commit.
