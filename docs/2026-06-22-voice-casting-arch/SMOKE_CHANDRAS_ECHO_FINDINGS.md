# Live smoke -- "Chandra's Echo" (2026-06-22) -- story-quality findings

Run: canonical `workflows/otr_scifi_16gb_full.json`, headless :8000, FLOOR lane
(all-visualizer), **local mistral-nemo** writer, 320 words, 3 characters,
`act_count=auto`, **OTR_BYPASS_FREEZE_HALT=1** (smoke stopgap so the critic did
not gate -- it flagged `arc_verdict=uneven` and shipped via repair-then-ship).
Episode: `output/otr/episodes/signal_lost_chandras_echo_20260622_141546/`.

## VC chunks 2-4 -- VERIFIED LIVE (the smoke's primary purpose)
- **Chunk 3 `meta.cast_voice_slots`** stamped with real timbre/age/role:
  c02 other/warm/adult/lead, c03 female/sharp/adult/foil, c04 male/deep/adult/support.
- **Chunk 4 hybrid LLM voice-fit fired + ACCEPTED:** c03 proposed+accepted
  `vz_donor_glenn` (12 cands), c04 proposed+accepted `vz_donor_marshal_indian`
  (12 cands); c02 (gender `other`) fell closed `no_cards` (bank has no `other`
  voices -> gender-agnostic ref `vz_donor_selfie`). indextts2 rendered all lines,
  no crash; per-character voices consistent + deterministic.
- Audio master saved (135.5s); no regression observed.

## STORY GRADE: C+ (~6/10) -- floor for this config, not the ceiling
News IS the crux (Chandra supernova-remnant signal = the dramatic object; opposed
wants real: Mali wants to amplify/disclose, Manfred wants to control). The
story-engine is clearly working. Drags:

### TOP FIX -- stage directions LEAK into spoken text (indextts2 speaks them)
The pre-freeze scrub fixed 10 lines but trailing, post-closing-quote directions
survived in the FROZEN text. These are spoken aloud:
- b005: `"... The world deserves to hear this." adjusts dials on the console`
- b010: `"... no bearing here." clutches her wedding ring tightly`
- b012: `"... is purely theoretical." taps his cane impatiently`
- b015: `... "tightens her scarf, a nervous gesture" I do hope ...` (mid-line)
- b017: `Sherlock, stop this at once! overrides systems, fingers dancing on the console I won't ...`
Pattern: a stage direction AFTER a closing quote / between quoted spans, with no
delimiter the current scrub catches. The earlier stage-direction-leak sprint
(`8c40182..6ce724d`) handled bare undelimited leads; this trailing/embedded
variant slipped. **Extend the detector + the freeze floor + the compose-line
reroll to this pattern, with a corpus case per line above.**

### Secondary craft issues
- **Incoherent antagonist arc:** Manfred swings supportive (b003) -> dismissive
  (b008) -> betrays Mali by leaking her research to the press (b011/b014) ->
  defends her life's work (b017). Motivation does not track. Candidate: the critic
  / reroll should catch a character whose stance reverses without a turn.
- **b011 role mis-stamp:** a MANFRED character line is tagged `speaker_role=announcer`.
- **Abrupt escalation:** observatory two-hander jumps to "presenting to the UN"
  (b015-b016) with no setup.

### Retest lever
Re-run WITHOUT the bypass (the STORY+CAST FIX STEPs 1-6 shipped today were meant to
make the freeze-halt gate trustworthy) to see the critic actually gate + reroll;
and/or a frontier writer (opus/gpt) for the quality ceiling.
