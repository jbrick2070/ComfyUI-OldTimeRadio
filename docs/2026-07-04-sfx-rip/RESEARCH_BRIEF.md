# RESEARCH BRIEF -- safely remove the remaining SFX subsystem (screenplay [SFX:] markup + director sfx_plan)

Operator directive (2026-07-04): "we should NOT be generating any SFX ledger or stuff." The retired
`sfx` SPEAKER ROLE is already gone (only isolated guards remained; those were neutralized). What
REMAINS and must be researched for a SAFE removal:

## The remaining SFX surfaces (to classify + plan removal)
1. **Screenplay `[SFX:]` markup** -- the story-writer LLM emits `[SFX: low bass sweep]`-style
   sound-effect stage directions, and LIVE code counts / strips / filters them:
   - `nodes/story_orchestrator.py` (sfx_count, `[SFX:` detection ~L2739-2746; SFX in structural-token
     lists / cast-blocklist ~L2059/2223/2390; transition-SFX injection ~L220-245).
   - `nodes/scene_sequencer.py:455` + `nodes/_otr_bark_lib.py:318` -- `re.sub(r'\[(?:ENV|SFX|MUSIC):...')`
     strips SFX from TTS text.
   - `nodes/_otr_casting.py` `_SFX_CAST_BLOCKLIST_PATTERNS` (~L1911-1996) -- rejects "SFX ..." as a fake
     cast name (BUG-LOCAL-090).
   - Tests: `test_core.py::test_strips_sfx_tag`, `test_cast_contract_rejects_structural_tokens.py::
     test_legacy_sfx_cue_artefacts_still_caught`, `test_constraint_repair_prompt.py`,
     `test_editor_constraints.py` + fixture scripts/treatments carrying `[SFX: ...]`.
2. **Director `sfx_plan` cue generation** -- fixtures (`sample_director_lemmy.json`,
   `reference_episode/director_satellites_collide.json`) carry `"sfx_plan": [...]` with `type:"sfx"`
   cues. Find what PRODUCES sfx_plan (is it live director output?) and what CONSUMES it.
3. **`sfx_<ep_id>` audio filenames** -- `test_filename_pattern_audit.py` references an `sfx_*.wav`
   reconstruction pattern. Is any code still writing sfx audio files?
4. **Coupled sfx SPEAKER-ROLE rejection messages** (LIVE, name 'sfx' in fixed text):
   `scene_sequencer.py:797`, `otr_meta_brief_image_prompt.py:644`, `otr_shot_lock.py:81`,
   `_otr_ledger_freeze` ALLOWED_SPEAKER_ROLES + the guard tests in `test_rip_sfx_broll_guard.py`
   (resolve/stamp/shot_lock/scene_still/ledger_freeze) that feed `speaker_role="sfx"` and assert a raise.
5. **Already-deleted-code comments** + the `rip-sfx-broll` migration label (historical; low priority).

## What the research must deliver
- A COMPLETE, grounded, ignore-blind inventory of every `sfx`/`[SFX:]`/`sfx_plan`/`SFX` occurrence in
  nodes/**, tests/**, config/**, workflows/**, classified: LIVE-FEATURE (removing changes behavior) vs
  DEAD/GUARD/COMMENT vs LABEL (`rip-sfx-broll`).
- The DEPENDENCY graph for the `[SFX:]` markup + `sfx_plan`: who writes them, who reads them, what
  happens to an episode if they stop being emitted (does anything require them / crash without them?).
- A SAFE, staged removal plan: what to delete vs neutralize, in what order, which tests to convert to
  positive/neutral, and how to prove no behavior regression (the LLM may still EMIT `[SFX:]`; if the
  stripper is removed, does that leak into TTS? -> the stripping may need to STAY even if generation is
  killed). Flag anything that must NOT be removed.
- The single biggest risk / gotcha, and how to verify green (full suite + Bug Bible + a smoke).

Do NOT edit any files. This is research only.
