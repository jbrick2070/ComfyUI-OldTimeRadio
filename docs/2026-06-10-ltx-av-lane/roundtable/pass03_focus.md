# PASS 03 REVIEW FOCUS: PROMPTS

You are one panelist in an adversarial review of the plan below. THIS pass
is the PROMPTS pass. Architecture and I/O contracts are LOCKED (pass01/02)
-- do not relitigate; one-line flags only for fatal contradictions.

The repo just shipped a roundtable-hardened brief-to-downstream prompt
restoration (finish_visual_prompt + get_era_tail in
_otr_story_brief_helpers; brief-grounded LTX scene opens; in-character FLUX
portraits via character_description; person guard; no-text clause; 240-char
cap; OTR_LTX_RADIO_PROMPT verbatim lane; stage-direction and self-vocative
scrubs). The new lane must JOIN that pipeline, not fork it.

Pressure-test exactly these against the grounding:

1. PROMPT SOURCES per role: for ltx_av_talk (announcer/character) and
   ltx_av_music, where does text_prompt come from TODAY for the equivalent
   beats (brief-grounded scene opens for ltx_video; portrait/M4 appearance
   chains for character imagery)? Specify the exact helper calls /
   request fields the new adapters should rely on so prompts arrive
   ALREADY finished (finish_visual_prompt'd) -- the adapter should NOT
   compose prompts itself. True or false in the grounding?
2. TALKING-HEAD PROMPT CONTENT: the audio drives the lips; what should the
   text_prompt say (framing, period styling, "speaking" verbs?) and what
   must it NOT say (stage directions, caption text, character names that
   trigger self-vocative issues, anything the no-text clause exists to
   prevent)? Propose a 1-2 sentence TEMPLATE SHAPE (not literal prose) and
   the negative-prompt baseline, citing the existing defaults
   (_LTX_DEFAULT_NEGATIVE in eng_ltx_video.py).
3. MUSIC PROMPT CONTENT: for "visuals breathe with the track", does the
   prompt need motion/rhythm vocabulary, or is the audio conditioning
   expected to carry it (the model "hears")? Should the music prompt path
   reuse get_story_brief_ltx scene composition verbatim (240-char cap,
   no-text clause), and what (small) additive tail is justified?
4. PROMPT BUDGET: LTX-2.3 uses a large text encoder; is the 240-char brief
   cap appropriate for it, harmful, or irrelevant? Recommend whether the
   cap stays (consistency) or the lane gets its own cap, and why.
5. NEGATIVE PROMPT: eng_ltx_video ships a default negative; should the AV
   lane reuse it verbatim, extend it (e.g. "static, frozen, still image"
   pressure against audio-ignoring renders?), and does anything in it
   conflict with talking-head content ("text" exclusion vs captions are
   burned later -- fine)?
6. PERSON-GUARD INTERPLAY: portraits feeding ltx_av_talk pass the person
   guard upstream; does ANY prompt content here need to repeat that
   protection, or is repeating it scope creep?
7. OTR_LTX_RADIO_PROMPT: the verbatim operator override exists for the
   radio open. Should the AV lane honor the same env on announcer beats,
   a sibling env (OTR_LTX_AV_PROMPT?), or none? Pick one and defend.

Rules: cite grounding or VERIFY-AT-BUILD; LOCKED items stay locked; the
adapter must not import or duplicate brief logic (V-12-adjacent: prompt
composition stays upstream). Output: numbered MUST-FIX (file + what),
SHOULD-CONSIDER, OPEN-QUESTIONS. Terse.
