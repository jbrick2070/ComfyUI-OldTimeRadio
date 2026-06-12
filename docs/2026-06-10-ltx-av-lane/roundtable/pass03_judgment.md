# pass03 (prompts) judgment -- Claude, judge + panelist

## ACCEPTED (grounded)

- ADAPTER-THIN PROMPTS (4/4): adapters read ONLY request.text_prompt /
  request.negative_prompt (eng_ltx_video/eng_humo precedent); NO import of
  _otr_story_brief_helpers in eng_ltx_av.py; locked by an AST test
  (test_get_music_mood_no_musicgen_import pattern).
- THE PROMPT GATE IS THE REAL DELTA (Claude MF1; GPT/Gemini/DeepSeek found
  the same starvation from the ShotLock side): render_driver.py:418's
  engine tuple ("ltx_video","wan_i2v") gates the WHOLE no-creative compose
  (override + brief scene). CHARACTER_BEARING_ROLES = {character_video}
  (otr_shot_lock.py) so announcer/music beats have NO M4 creative; without
  joining the gate, ltx_av shots fall to the generic ":193 radio studio"
  default -- the exact regression the gap audit killed. DELTAS:
  - ltx_av_music JOINS the existing tuple at :418 unchanged (override +
    "vintage radio set" clause + brief compose, verbatim reuse).
  - ltx_av_talk gets a SIBLING additive branch (same precedence shape,
    minus the radio override): if engine_id == "ltx_av_talk" and no
    creative prompt -> TALK fallback compose (below).
- REJECTED the ShotLock route (Gemini MF1/MF2: add ANNOUNCER_VISUAL to
  CHARACTER_BEARING_ROLES / new fallback loop in derive_creative_directives)
  -- that mutates SHIPPED behavior for existing engines (every ltx_video
  announcer beat would start getting M4 LLM prompts). The driver branch is
  engine-scoped and fires only for the new lane.
- OTR_LTX_RADIO_PROMPT ASYMMETRY (Gemini SF6 + Claude MF3 over GPT MF4 /
  DeepSeek SC1): the override is ENGINE-GATED in grounding (:418-427 inside
  the tuple), not role-scoped as GPT assumed. ltx_av_music honors it
  (scene engine for the open -- operator contract preserved); ltx_av_talk
  does NOT (verbatim room prose fights portrait-I2V conditioning). NO new
  env var (GPT cut + Claude agree; Gemini's OTR_LTX_AV_PROMPT sibling and
  DeepSeek's OTR_LTX_AV_NEGATIVE rejected -- knob sprawl).
- TALK FALLBACK TEMPLATE (merged Claude MF2 + Gemini SF4 + GPT MF3):
  subject default "a 1940s radio announcer" (announcer beats are the main
  consumers -- character beats DO get M4 creative via CHARACTER_BEARING_
  ROLES; announcer has no cast row) + "head and shoulders at a period
  microphone" (microphone as SETTING NOUN, no speech verbs -- Gemini's
  double-driving concern), finished via finish_visual_prompt (240 cap,
  style_tail=False) + "no on-screen text". FORBIDDEN: quoted dialogue,
  beat text, stage directions, vocative character names, caption text
  (DeepSeek's optional beat_text REJECTED). One M0 P1 cell tests +/- a
  speech verb ("speaking") before any verb lands.
- NEGATIVE PROMPT (GPT MF5 + Gemini SF5 over DeepSeek MF2): reuse
  _LTX_DEFAULT_NEGATIVE VERBATIM for both adapters, one shared constant in
  the core; "text" stays (captions burn later; frames should avoid text
  anyway); NO blind extension -- "frozen face/still image" terms only if
  M0 P1 shows audio-inert renders (recorded as a conditional, exact
  strings pre-agreed: ", frozen pose, still image" music-only).
- MUSIC PROMPT (Gemini CUT2 + Claude SC6 over DeepSeek MF5): verbatim
  reuse of today's open compose; NO rhythm/motion vocabulary in v1 (the
  audio conditioning carries it; text rhythm vocab risks literal strobing
  / over-constraint). ONE M0 P1 cell tests a single motion-energy clause;
  if it clearly wins, the clause lands as a constant in the DRIVER's music
  branch, not the adapter.
- 240-CHAR CAP KEPT (4/4) for consistency + hash stability; one optional
  M0 cell may probe 240-vs-longer; no lane-specific cap.
- PERSON GUARD stays upstream-only (4/4; _depicts_person in
  otr_meta_brief_image_prompt.py guards the PORTRAIT prompt chain).

## REJECTED / MISREADS

- GPT MF4 "OTR_LTX_RADIO_PROMPT is role-scoped, honor for ltx_av_talk":
  MISREAD -- grounding shows it engine-tuple-gated and scene-flavored;
  honoring it for a portrait-I2V engine contradicts the conditioning.
- DeepSeek MF2 negative-prompt extension incl. "silence" (a nonsense
  negative for a video model) + new env: REJECTED.
- DeepSeek MF4 template with optional beat_text + "speaking": REJECTED
  (dialogue/self-vocative risk; speech-verb question goes to M0).
- Gemini MF1/MF2 ShotLock edits: REJECTED (shipped-behavior mutation).

## VERIFY-AT-BUILD (carried)

- M4 creative content for character beats feeding ltx_av_talk remains
  suitable (it may embed b["text"] fragments per ShotLock's deterministic
  template -- confirm the shipped pre-freeze scrubs cover what reaches
  request.text_prompt; if a character creative prompt proves dialogue-y in
  M0, the talk fallback template is the fix, NOT a ShotLock edit).
- Driver talk-branch source for announcer subject when a cast-aliased
  announcer DOES have a description (use it when present, default
  otherwise).
