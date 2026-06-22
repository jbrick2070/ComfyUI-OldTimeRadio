# R3 GROUNDING -- wiring trace (code-grounded)

## role_mismatch source = ONE LINE
`nodes/_otr_ledger_reviewer.py:500` (in `_render_cast_contract_for_audit`):
```python
role = row.get("speaker_role") or row.get("tts_model") or ""
```
When `speaker_role` is empty it falls back to `tts_model` (an ENGINE name: kokoro/bark).
The auditor then emits that engine name as `violation.expected`; the repair
(`apply_deterministic_cast_repairs` ~L1003) checks it against
`_ALLOWED_SPEAKER_ROLES = {character, announcer, music_open, music_close,
music_inter, sfx}`, it is not in the set, repair fails -> the logged
`role_mismatch ... expected='kokoro'`. FIX = drop the `or row.get("tts_model")`
fallback; require an explicit `speaker_role` per row (fill it at the upstream writer).

## compose ALREADY has arc context (FIX 3 re-point)
`nodes/_otr_line_composer.py` `LineRequest` (~L571-725) already carries: `arc_phase`,
`dramatic_question`, `beat_objective`, `beat_obstacle`, `beat_turn`, `beat_subtext`,
`beat_tension(1..5)`, `next_turn`, `last_lines`, `outline_spine`, `current_beat_block`.
So the composer is NOT arc-blind -- a new SceneArcContext would duplicate these. The
arc lever is therefore the BEAT-PLANNING that SETS beat_objective/turn/tension (the
outline / slot_drama_contracts), not the per-line compose. Missing only: scene-level
(not per-speaker) continuity constraints.

## critic call sites (FIX 1 thread point)
- `nodes/_otr_freeze_cascade.py:754` `run_story_critic(...)` -- initial WHOLE-EPISODE pass.
- `nodes/_otr_reroll.py:621` `run_story_critic(generate_fn, ledger_data, cast_rows)` --
  the reroll-loop pass. Add `scope_line_ids: set[str]` to `run_story_critic`; the
  reroll call passes the patched target set; freeze-cascade call passes None.

## workflow node surface (most fixes are INTERNAL code, NOT JSON rewiring)
`workflows/otr_scifi_16gb_full.json`: writer id=1 OTR_LedgerScriptWriter; cast id=80
OTR_CastLock; char voices id=81 OTR_BatchCharacterVoices; announcer id=82
OTR_AnnouncerVoice. The critic/reroll/reviewer run INSIDE the writer pipeline, not as
separate graph nodes -> FIX 1/4/5 are code-internal; FIX 2 is in OTR_CastLock (node 80).
No node/wiring/widget add is required by these fixes (confirm no widget drift).
