# pass03 prompts -- Claude panelist review (written before reading the panel)

MUST-FIX

1. (render_driver.py:418) The scene-prompt compose gate is the SAME
   engine-tuple `("ltx_video", "wan_i2v")` as the canvas gate at :387 --
   pass02 only added the new names to the CANVAS tuple. Without joining the
   PROMPT gate too, a no-creative ltx_av shot falls through to the generic
   "a 1940s radio studio" default (:193) -- the exact regression the
   2026-06-10 gap audit just killed. Both tuples must gain the new names
   (with the talk nuance in 2/3 below).
2. (prompt content, talk) The :420-456 fallback content is SCENE language
   ("a vintage radio set glowing in the scene", "slow cinematic camera
   drift") -- wrong for an I2V talking head; it fights the portrait
   conditioning. ltx_av_talk's no-creative fallback needs a TALK clause
   set: subject from character_description (the in-character portrait
   chain's source), "head and shoulders, speaking into a period
   microphone", period/light from the brief via finish_visual_prompt
   (240 cap, style_tail=False), "no on-screen text", natural-motion
   clause. NO dialogue text, NO character name vocatives (self-vocative
   scrub precedent), NO stage directions. M4 creative prompts, when
   present, take precedence unchanged.
3. (OTR_LTX_RADIO_PROMPT) ltx_av_music HONORS the existing override on
   open beats (it is a scene engine for the open -- verbatim radio-set
   prose is exactly what the operator expects). ltx_av_talk does NOT
   consume it: a verbatim scene override contradicts portrait-I2V
   conditioning; silently rendering a radio set while lip-syncing a
   portrait would be a confusing half-honored override. No new env var.
   Document the asymmetry in the adapter docstring + plan.
4. (no prompt composition in the adapter) Prompts arrive FINISHED in
   request.text_prompt (M4 creative > override > brief-composed, all
   upstream in the driver). The shared core must not import
   _otr_story_brief_helpers (consumer->helper direction stays in the
   DRIVER; the adapter consumes the request only). Encode as a test:
   eng_ltx_av has no brief-helper import (AST check like
   test_get_music_mood_no_musicgen_import).

SHOULD-CONSIDER

5. Negative prompt: reuse _LTX_DEFAULT_NEGATIVE verbatim ("low quality,
   worst quality, blurry, distorted, watermark, text, static") for both
   adapters; optionally extend with "frozen pose, motionless" for the
   music adapter ONLY (pressure against audio-inert renders); keep one
   shared constant in the core, not per-adapter drift.
6. Music prompt: reuse today's open compose VERBATIM (brief core +
   "a vintage radio set glowing" on opens + camera-drift clause + no-text).
   The "breathes with the track" effect is the AUDIO CONDITIONING's job;
   adding rhythm vocabulary ("pulsing", "beat-synced") risks literal
   strobing artifacts. Record a single M0 P1 experiment cell comparing
   with/without a motion-energy clause before any vocabulary lands.
7. 240-char cap: KEEP for consistency (ledger comparability, the gap-audit
   contract). The bigger LTX-2.3 encoder tolerating longer prompts is not
   evidence longer prompts render better -- M0 may probe 240 vs 480 once,
   else defer.
8. Person guard stays upstream-only; repeating face checks in prompt text
   is scope creep. The talk fallback's "head and shoulders" phrasing plus
   the existing portrait-side guard suffice.

OPEN-QUESTIONS

9. Does character_description reliably exist for ANNOUNCER beats (the
   announcer may have no cast row)? If absent, the talk fallback needs a
   period-safe default subject ("a 1940s radio announcer") -- confirm the
   portrait chain's announcer alias behavior in pass04 grounding.
10. Should the talk fallback mention emotion (the delivery vector exists
    audio-side)? Likely no for v1 (the audio carries it); flag for the
    whiny-voice/audio lane to revisit later, not this sprint.
