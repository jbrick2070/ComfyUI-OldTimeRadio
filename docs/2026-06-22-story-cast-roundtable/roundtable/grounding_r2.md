# R2 GROUNDING -- what the real code ALREADY does (so the panel codes the real gaps)
Code-grounded reads of the production source. The R1 plan assumed two fixes that
are ALREADY implemented; the coding plan must target the REAL gaps below.

## ALREADY IMPLEMENTED (do not re-propose):
- **Prose IS decoupled from metadata.** `_otr_line_composer.py` `compose_line_draft()`
  (~L1689) generates ONLY the spoken dialogue; the system prompt (~L966) says "Only
  the words the character speaks out loud." arc_phase / trait / beat_intent /
  target_words arrive pre-built on the `LineRequest` (~L615-725) from the orchestrator
  -- the LLM never fills JSON metadata. So "constraint overload / decouple" is moot.
- **The critic already emits a concrete per-target instruction.** `_otr_story_critic.py`
  `RerollTarget` (~L165-175): fields `{line_id, hint}`; comment "hint is CONCRETE,
  ACTIONABLE feedback -- this string becomes the reroll instruction handed verbatim to
  the line composer" (~L169). So "the reroll is blind, add correction_instruction" is moot.
- **The reroll already does targeted patches of stable line_ids.** `_otr_reroll.py`
  (~L526-603): `while report.reroll_targets and cycles_run < MAX_REROLL_CYCLES`, then
  `for target in report.reroll_targets:` -> `_find_line_row(target.line_id)` ->
  `compose_line()` in isolation -> `led.update_line_text(line_id, new_text)`. Approved
  lines preserved. So "make the reroll targeted" is moot.

## THE REAL GAPS (code the coding plan against THESE):
1. **Per-line isolation generation.** `compose_line()` composes ONE beat from "this
   Beat + EpisodeCanon header + last N ledger lines" (`_otr_line_composer.py` L5-6) --
   no scene-arc view. Both the original AND each re-composed line are written blind to
   the scene's escalation trajectory. THIS is what makes lines "flat," and why a
   re-roll of a flat line produces another flat line.
2. **Whack-a-mole critic scope.** `_otr_story_critic.py` (~L410-415) re-scores the
   WHOLE character-dialogue ledger every call (filters to character rows ~L291-304, not
   the patched subset). So fixing 3 targets re-exposes 3 others -> cycle1=3, cycle2=3,
   never converges. The reroll loop has no monotonic-decrease requirement.
3. **"Flat" is pure LLM judgment.** `FlatLine.reason` (~L155-162) is a free string; no
   operational/code test. Composer and critic do not share a definition of the target.
4. **voice_preset=None is reachable.** `cast_lock.py`: if `cast_seed is None` (~L272)
   it returns early, "voice_preset preserved (no replay)"; and even with a seed, only
   `if cid in voices: row['voice_preset']=voices[cid]` (~L285-291) -- an unmatched
   char_id keeps None. No fail-closed assertion.
5. **role_mismatch source UNVERIFIED in cast_lock.** The audit fires (`OTR_LedgerReviewer`
   role_mismatch, engine name in role field) but cast_lock only READS
   `speaker_role or role` (~L45); the WRITE that puts an engine name into a role field
   is upstream -- an R3 wiring trace target.
