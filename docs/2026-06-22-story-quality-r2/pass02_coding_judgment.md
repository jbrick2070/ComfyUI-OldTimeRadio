# R2 pass02 (CODING) judgment -- the seam + schema corrections (Claude = judge)

Panel GPT-5.5 + Gemini-3.1-pro + Grok-4.3, grounded vs _otr_outline / _otr_line_hygiene /
_otr_dramatic_state. ~$0.20. Strong convergence on FIVE build-blocking corrections -> the draft is
NOT build-ready until these land. Fold them, then the sprint is real.

## CONFIRMED corrections (all grounded)
1. **Beat intent / action-verb goes in OUTLINE Stage 3, NOT the line composer (Gemini+GPT, decisive).**
   The line composer CONSUMES `Beat.intent`; intents are WRITTEN in `_otr_outline.generate_outline`
   Stage 3 via `_build_beat_user_prompt` + the `_BeatFleshout` structured_call. So the
   "action-verb-under-pressure" + escalation (C4) constraints belong in `_build_beat_user_prompt`
   / a Stage-3 post_validator -- not the line prompt.
2. **No new Pydantic fields -> use `meta`, not DramaticState/cast schema (GPT, decisive).**
   `DramaticState` carries ONLY dramatic_question / character_a_wants / character_b_wants /
   costly_choice_beat / ending_change -- adding `central_object` (C2) or `specificity_anchors` (C1)
   as Pydantic fields IS a schema change (banned). Store them in the FREE-FORM ledger `meta` under
   agreed keys (e.g. meta["central_object"], meta["specificity_anchors"]). C3 speech_signature: only
   "promote F5" if the field already exists on the locked cast row -- VERIFY; else use meta.
3. **"Source-derived / non-default wants" needs a real classifier (GPT).** No provenance flag exists;
   `derive_dramatic_state_from_meta` emits `_DEFAULT_A_WANTS`/`_DEFAULT_B_WANTS` with cast names. Add
   a deterministic helper that detects default-templated wants (match against the _DEFAULT templates)
   -> only inject wants into prompts when NON-default.
4. **Hygiene = FLAGS ONLY; a separate bounded reroll wrapper (all 3).** `_otr_line_hygiene.py` is
   pure deterministic detection/scrub (no LLM). The cliche/stage-business/generic/on-the-nose gates
   RETURN (flag, reason); a SHARED wrapper `(beat, current_text, reject_reason, context) -> reroll`
   calls the VERIFIED line-composer recompose path. Hygiene never calls the LLM.
5. **Locate the real seams before coding (all 3, verify-at-build).** Unlocated today:
   (a) where `line.text` is materialized for a `music_inter` beat (S1 suppression point --
       `production_ledger.init_lines_from_outline` / composer / caption burn) + a role helper
       `is_spoken_role(role)`;
   (b) the line-composer function + its existing retry/recompose seam (S3 + the reroll wrapper);
   (c) the dedicated ANNOUNCER-close composer fn + signature (S2 reroll);
   (d) the locked CAST ROW model (does F5 speech_signature live there? -> C3).

## ACCEPTED details
- S2 banned-thesis scan = concrete CASE-INSENSITIVE regexes w/ straight+curly apostrophes
  ("Tonight['’]s revelation", "the lesson is", "reminding us", "proving \\w+ right",
  "\\w+ is now shared", "this shows") as a SHARED module constant -- NOT wildcard strings.
- C1 generic-line gate: scope = `speaker_role=="character"` on non-opener/closer/music beats;
  anchor match case-folded vs meta["specificity_anchors"]; EXCLUDE cast names + sentence-initial
  capitalization from proper-noun credit.
- C2 must derive BEFORE S2 (the close references the central object) -> ordering guard.

## CUT (panel)
- The arbitrary "3-5 rerolls/episode" cap (the existing 3-attempt retry budget already bounds cost).
- The "soft same-voice flag" (the hard per-line signature constraint suffices).

## VERDICT
Direction CONVERGED + correct; the draft needs the 5 corrections + the 4 seams LOCATED. Next: a
seam-location (wiring) pass against the REAL OTR_LedgerScriptWriter + production_ledger + cast model,
then the sprint plan is build-ready. (The creative intent is unchanged -- only the HOW is corrected.)
