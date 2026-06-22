# GROUNDING PACK R3 -- wiring facts (verified against real code)

## W1. needs_full_rerun: NO cross-run channel (DEFECT 2 make-or-break)
`needs_full_rerun` is a terminal OUTPUT STRING + a `meta.freeze_verdict` stamp on the cascade node
(`OTR_LedgerFreezeCascade.py:513-521`); NO node re-invokes the writer on it. The writer
(OTR_LedgerScriptWriter, node 1) is UPSTREAM of the cascade and contains zero matches for
needs_full_rerun/regeneration_hint/freeze_verdict -- it cannot read the verdict in the DAG. On any
rerun the writer calls `_PL.new_ledger(episode_id=None)` (`OTR_LedgerScriptWriter.py:2260`) which
overwrites the in-process `_CURRENT` ledger with a BLANK one (empty meta). `meta["regeneration_hint"]`
IS written (`_otr_freeze_cascade.py:900-901`) but is READ BY NOBODY (dead-end forensic stamp). On disk
the rerun gets a NEW `pending_<new-ts>` dir; the old ledger JSON is never reloaded. The A2 path means a
`needs_full_rerun` from the reroll loop does NOT halt -- it ships the episode (repair-then-ship); only a
Phase-10 gap-audit rejection hard-fails. **CONCLUSION: with workflow JSON frozen there is NO existing
channel to carry a coherence hint from a failed run into the regenerated episode. A "rerun" is a manual
operator re-queue. DEFECT 2 auto-repair via needs_full_rerun is NOT buildable.**

## W2. Prompt builders
- Per-LINE text prompt: `_otr_line_composer.py::_build_user_prompt(req)` (1050-1338). The "spoken words
  only" instruction ALREADY EXISTS at 1307-1315 ("Write 1 spoken line ... Speak in the first person;
  never narrate your own actions in the third person and never say your own name") -- and the corpus
  still leaks, so it is necessary-but-insufficient (strengthen here; Tiers 2/3 are load-bearing).
- Per-BEAT intent prompt: `_otr_outline.py::_build_beat_user_prompt` (1166-1236), "write the intent
  (NOT dialogue) ... action under pressure". This is where a DEFECT-2 antagonist-stance-consistency
  generation lever goes (JSON-free, no cross-run state).

## W3. speaker_role write points (complete)
Ledger-row writers: `production_ledger.py:795` (init_lines_from_outline; role derived from outline,
char_id derived FROM role) + `:885/:899` (set_lines). Mutators: `_otr_ledger_reviewer.py:1063`
(role_mismatch repair -> the b011 culprit), `cast_lock.py:473` (re-stamps a line to announcer WHEN
char_id IS the announcer -- LEGITIMATE; coercion must not fight it), `_otr_speaker_role.py:218`
(stamp_default_role backfill). Downstream/other builders: `scene_sequencer.py:925/937/1043/1059/1608`,
`_otr_radio_editor.py:1979`, `_otr_legacy_to_stage1_adapter.py:642`, `_otr_story_spine.py:168`; outline
Beat objects `_otr_outline.py:1499/1517/1536/1553`. A PRE-FREEZE consistency sweep over the whole ledger
is the catch-all (avoids instrumenting all builders).

## W4. FailedDimension
`typing.Literal["knowledge","pressure","relationship","decision","obstacle","tension","unspecified"]`
(`_otr_story_critic.py:174-182`); optional field on RerollTarget (default "unspecified"). The ONLY
consumer is `_otr_reroll.py:591-596` (free-form `[dim]` prefix via getattr). Adding "stance" is SAFE at
runtime; must ALSO update the critic system-prompt prose (`_otr_story_critic.py:310-329`) so the model
emits it + the tests. No match/case or exhaustive enumeration anywhere.

## W5. No per-line meta dict
Line rows are plain dicts with a FIXED field set: line_id, shot_id, beat_id, char_id, text, traits,
boundary, char_count, word_count, bark_wav_path, start_s, dur_s, speaker_role, arc_phase,
**compose_flags (list)**, beat_intent, target_words, dialogue_slot_id. There is NO per-line `meta` dict.
The free-form per-line channel is `compose_flags` (list of "kind:detail" strings). Episode-level (top)
`meta` holds story_critic_report and line_dramatic_frame (a dict keyed BY line_id). So per-line audit
breadcrumbs ride `compose_flags`; aggregate audit rides episode-level `meta`.
