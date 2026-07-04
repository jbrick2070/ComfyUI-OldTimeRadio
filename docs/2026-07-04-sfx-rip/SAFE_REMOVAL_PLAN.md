# SFX safe-removal plan (converged: Sonnet fanout + kibitz codex + agy)

## Headline finding (all 3 agree, grounded)
The SFX subsystem is ALREADY ~95% removed (rip-sfx-broll 2026-07-01 killed the sfx speaker role, the
`sfx[]` ledger schema, the sfx_cue field, `set_sfx`/`apply_sfx_timings`, the procedural SFX node, and
AudioGen-SFX). What remains:
- **NO live SFX generator** in the production writer path. The canonical workflow uses
  `OTR_LedgerScriptWriter` (NOT `story_orchestrator.py`, which is legacy/orphan RSS+unit-test code);
  `OTR_LedgerScriptWriter` explicitly notes the old `[SFX:]` token emission died with `sfx_cue`.
- **`sfx_plan` is DEAD** -- zero producers/consumers; only forbidden-socket lists + 2 orphan fixtures.
- The ONLY thing still actively instructing the LLM to write SFX = **`nodes/_otr_period_prompts.py:60`**
  (`OTR_PERIOD_SYSTEM_PROMPT` says `... [SFX: distant thunder]`), routed via
  `_otr_creative_prompt_router.py` for `prompt_profile=="otr_1940s_v1"` -- **DORMANT** (a test pins that
  no curated model row uses that profile today).

## THE RIP (minimal, safe)
1. **Kill the one active instruction:** remove `[SFX: distant thunder]` from `OTR_PERIOD_SYSTEM_PROMPT`
   (`_otr_period_prompts.py:~60`). This is the only line in the repo telling an LLM to emit SFX.
2. **Add a pinning guard:** a test asserting `"[SFX:" not in OTR_PERIOD_SYSTEM_PROMPT` (+ ideally audit
   the modern writer prompt + creative router so NO enabled prompt teaches `[SFX:]`).
3. **Housekeeping (optional, zero behavior):** delete the dead shadowed `_inject_scene_transitions`
   (`story_orchestrator.py:219-252`, never runs -- a duplicate-name landmine); delete the 2 orphan
   `sfx_plan` fixtures (`sample_director_lemmy.json`, `reference_episode/director_satellites_collide.json`)
   + their README paragraph (no test references them).

## MUST NOT REMOVE (all 3 flag this -- these are defenses, not features)
- The TTS strippers `scene_sequencer.py:455` + `_otr_bark_lib.py:318` (shared `[ENV|SFX|MUSIC:]` regex;
  ENV/MUSIC are live; strips a hallucinated `[SFX:]` so Bark never speaks it aloud).
- The cast-name SFX blocklist (`_otr_casting.py`) + editor FORMAT_FAILURE needle (`_otr_editor_constraints.py`)
  -- LLM-hallucination guards added after BUG-LOCAL-090/097.
- The `speaker_role="sfx"` rejection sites (`_otr_speaker_role.py`, `scene_sequencer.py`,
  `otr_meta_brief_image_prompt.py`, `otr_shot_lock.py`) -- protect against stale on-disk ledgers.
- The tombstone guards (`sfx_plan_json`/`sfx_audio_clips`/`sfx_offset_ms` forbidden sockets,
  `sfx_<ep_id>` filename-audit) -- prevent reintroduction.
- The `rip-sfx-broll` label + already-removed-code comments -- non-functional; CUT from scope (all 3).

## BIGGEST RISK
Conflating "no live generator" with "safe to delete the strippers/guards." The subsystem looks dead by
data-flow, but the LLM can hallucinate `[SFX:]` unprompted at any time -- the strippers/guards are the
only thing between that and a spoken-aloud `[SFX: ...]` in production audio. Kill the one instruction;
keep every defense.

## PROOF OF NO REGRESSION
Full suite + Bug Bible + `OTR_WorkflowValidator` (no workflow JSON change expected -- sfx sockets are
already forbidden). Optional "golden leakage" test: an injected `[SFX: door slam]` never reaches TTS
text or captions.

## RECOMMENDATION
Safe to proceed. It is a ~1-line prompt edit + a new guard test (+ optional dead-code/fixture cleanup),
NOT a broad "delete anything with sfx in the name" rip.
