<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The adapter prompt extraction is trivial, but upstream `OTR_ShotLock` currently starves the new roles of prompts, which violates the "adapters do not compose prompts" rule.

MUST-FIX BEFORE BUILD:
1. [otr_shot_lock.py / CHARACTER_BEARING_ROLES] Defect: `announcer_visual` and `music_visual` are excluded from `CHARACTER_BEARING_ROLES`. They currently bypass `derive_creative_directives` entirely, meaning their `shots[].creative.text_prompt` is empty in the ledger. If the new adapters must not compose prompts themselves, they will receive blanks. Fix: Add `Role.ANNOUNCER_VISUAL.value` to `CHARACTER_BEARING_ROLES` so it gets the M4 LLM derivation. 
2. [otr_shot_lock.py / derive_creative_directives] Defect: `music_visual` beats do not need M4 LLM derivation, but they still need a brief-grounded prompt stamped into the ledger. Fix: Add a non-LLM fallback loop in `derive_creative_directives` for `music_visual` beats that assigns `text_prompt = finish_visual_prompt(meta, get_story_brief_ltx(meta, max_chars=90), max_chars=240)`.
3. [eng_ltx_av.py / _build_render_request] Defect: The plan does not specify the exact extraction field. Fix: Both new adapters must extract `request.get("text_prompt")` directly (mirroring `eng_humo.py` and `eng_ltx_video.py`). True in grounding: existing engines rely on this field and do not call brief helpers themselves.

SHOULD-FIX:
4. [Talking-Head Prompt Content] Defect: Using verbs like "speaking" or "talking" (like HuMo's fallback) in an audio-conditioned I2V model risks double-driving the lips or conflicting with the audio track. Fix: The template shape for `ltx_av_talk` should be strictly `"{appearance}, {setting}"` (plus the era tail added by `finish_visual_prompt`). It must NOT contain stage directions, character names (triggers self-vocative issues), or speech verbs.
5. [Negative Prompt] Defect: Missing baseline definition. Fix: Reuse `_LTX_DEFAULT_NEGATIVE` verbatim (`"low quality, worst quality, blurry, distorted, watermark, text, static"`). The "text" exclusion is safe because captions are burned post-render, and "static" provides good pressure against audio-ignoring frozen renders.
6. [OTR_LTX_RADIO_PROMPT] Defect: Reusing `OTR_LTX_RADIO_PROMPT` for `ltx_av_talk` announcer beats will break the render. That override is designed for `ltx_video`'s b-roll radio open (a room/microphone). Applying a room description to an `init_image` of a tightly-cropped face will confuse the I2V model. Fix: Honor NONE. If an operator override is required for the AV lane, use a sibling env (`OTR_LTX_AV_PROMPT`).

OPTIONAL / NICE-TO-HAVE:
- [Prompt Budget] Keep the `finish_visual_prompt` 240-char cap for consistency. While T5-XXL has a massive context window, the brief fragment is just flavor; uncapping it risks diluting the core visual subject (appearance/setting) with narrative prose.

CUT THESE (over-engineering):
1. [Person-Guard Interplay] Do not repeat the person guard in the `ltx_av_talk` text prompt. The `init_image` already passed `_depicts_person` upstream in `otr_meta_brief_image_prompt.py`. The I2V model will heavily condition on the face in the image; repeating it in text is redundant scope creep.
2. [Music Prompt Content] Do not add motion/rhythm vocabulary ("dancing", "pulsing") to the `ltx_av_music` prompt. The audio conditioning natively drives the motion; forcing it in text over-constrains the semantic content.