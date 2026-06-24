# OTR Story+Render Pipeline -- ASSUMPTION ATTACK BRIEF

This is NOT a plan to harden. It is the set of ASSUMPTIONS the OTR story+render
pipeline rests on. Your job: ATTACK them. "Looks good / build-ready" is a FAILED
response. Every claim must cite file:line in the grounding files.

## What the system does (one paragraph)
A news brief -> `OTR_LedgerScriptWriter.run()` builds an Outline (`_otr_outline.
generate_outline`: macro -> per-phase beats -> `_assemble_outline` stamps
announcer bookends + per-beat target_words + arc_phase). A deterministic
"story-grammar" layer (`_otr_story_quality_l12.py` + `_otr_style_catalog.py`)
picks a radio-drama STYLE per episode and assigns each voiced CHARACTER beat a
dramatic ROLE (setup/pressure/personal_stake/<climax>/consequence), with the
climax forced to the LAST voiced beat; the climax's TYPE is the style's
ending-taxonomy class. `_otr_line_composer` renders a per-beat prompt; the local
LLM (mistral-nemo or gemma-12b) writes each line. A critic/reroll/freeze cascade
edits, then audio (indextts2) + video (`_otr_video_engines/eng_ltx_av.py` LTX-AV
bookends + a CRT visualizer for body beats) render to an OBS final. Canonical
graph: `workflows/otr_scifi_16gb_full.json`.

## PROVEN-LIVE failures this session (your attack should explain/extend these)
- **gemma collapses into the exact console standoff the system targets**, in the
  BODY beats, despite style=`numbers_station_spy_thriller` + ending=`reversal` +
  L12 crisis-noun grounding ON: "press this red lever and let the atmosphere
  out", "blowing the fuel cells", "permanent system lockout", "bypass his lock
  in three seconds ... vacuum". The grammar shaped the CLIMAX + the announcer
  CLOSE but did NOT govern the body.
- **Style diversity dies at the writer.** style=`memory_erasure_clinic_session`
  produced a NASA lunar-mission story (the sound_world/story_engine were
  ignored); the writer anchors to the news premise. Only the ending_tag survives.
- mistral writes coherent, non-standoff stories from the SAME machinery; gemma
  does not. The engine is writer-dependent.

## The seven hunts (run every round, file:line evidence required)
1. **Forced-default** -- values/roles/choices forced identically on every output
   (episode/beat/render). Should each vary? What is lost by hardcoding it?
2. **False-distinction / merge** -- two named components/modes that are one thing
   differing only by input. Propose the collapse.
3. **Defended-invariant audit** -- every assumption a comment defends as a virtue
   is a SUSPECT. Argue it constrains the output.
4. **Upstream-varies / downstream-collapses** -- trace each diversity lever
   (premise, style, seed) to the final output; find where the variety dies.
5. **Dead / cargo-cult** -- flags/fields/branches/config never load-bearing
   (constant, never read, overridden downstream, or emitted-but-ignored).
6. **"Delete it" test** -- for the 5 most load-bearing constants/defaults, argue
   FOR removing each. Nothing-changes = dead; everything-changes = an
   undertested single point of failure.
7. **Single-prior trap** -- where the weak local model's most-probable training
   default silently wins over our intent, and where we rely on instruction-
   following it demonstrably ignores.

## Grounding files (attack these, not the symptom files)
- `nodes/_otr_story_quality_l12.py` -- beat-role spine, DOMAIN_PALETTE,
  GENERIC_CRISIS_NOUNS, assign_beat_roles (climax-last), build_sq_data
  (mutates beat.intent ONLY), fallback_content (_PERSONAL_COST has only
  "general").
- `nodes/_otr_outline.py` -- `_assemble_outline` (hardcoded announcer
  open/close intents, target_words, moods), generate_outline.
- `nodes/OTR_LedgerScriptWriter.py` -- F2 block (select_style -> climax_role ->
  build_sq_data -> ending-template injection), the writer model default.
- `nodes/_otr_line_composer.py` -- DRAMATIC FRAME render (beat_role /
  conflict_object / conflict_type / ending_template), the per-beat prompt.
- `nodes/_otr_style_catalog.py` -- STYLE_CATALOG (100 styles w/ sound_world /
  story_engine / ending_mode), select_style, ENDING_TEMPLATES, ending_tag map.
- `nodes/_otr_video_engines/eng_ltx_av.py` -- forced cfg/steps/negative/SHARP.
- `workflows/otr_scifi_16gb_full.json` -- canonical graph + saved widget values.

## Deliverable
A ranked "ASSUMPTIONS WE SHOULD KILL" list with the concrete fix for each, then
the 3 HIGHEST-LEVERAGE structural changes. No "build-ready" sign-off.
