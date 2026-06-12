<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: yes-with-fixes. Prompt ownership/source rules are not explicit enough; adapter must consume already-finished request.text_prompt and must not fork brief/radio prompt logic.

MUST-FIX BEFORE BUILD:
1. [Open questions/pass03 PROMPTS + ARCHITECTURE “ONE new file eng_ltx_av.py”] nodes/_otr_video_engines/eng_ltx_av.py: make prompt handling adapter-thin. The AV adapters must read only request.text_prompt / request.negative_prompt and must not import/call get_story_brief_ltx, get_era_tail, finish_visual_prompt, or OTR_LTX_RADIO_PROMPT. Grounding: eng_ltx_video._build_render_request consumes request.text_prompt; HuMo._build_render_request consumes request.text_prompt; finish_visual_prompt is already the upstream finisher. Concrete fix: implement _build_render_request like eng_ltx_video/HuMo for prompt fields; default negative to the existing baseline string from eng_ltx_video._LTX_DEFAULT_NEGATIVE: “low quality, worst quality, blurry, distorted, watermark, text, static”.

2. [Open questions/pass03 PROMPTS] nodes/_otr_video_engines/render_driver.py [VERIFY-AT-BUILD]: confirm/extend the existing ltx_video scene-open prompt path to ltx_av_music and ltx_av_talk announcer_visual. Grounding shows ShotLock only derives creative for CHARACTER_BEARING_ROLES = {character_video}; announcer_visual and music_visual do not get ShotLock creative text_prompt. The helper doc names render_driver’s scene composer as the live “ltx_scene_open” consumer. Concrete fix: if render_driver’s current brief/OTR_LTX_RADIO_PROMPT path is engine-id-gated to “ltx_video”, widen it to the new AV engine IDs/roles upstream so request.text_prompt arrives already finished via existing brief composer + finish_visual_prompt(max_chars≈240) + no-text clause. Do not compose inside eng_ltx_av.py.

3. [Open questions/pass03 PROMPTS + otr_shot_lock.py grounding] nodes/otr_shot_lock.py or render_driver.py [VERIFY-AT-BUILD]: ltx_av_talk character_video must not receive raw dialogue/stage-direction/self-vocative text as the visual prompt. Grounding ShotLock fallback currently includes b["text"] in _deterministic_template and LLM composition; the user says scrubs shipped, but they are not visible in the excerpt. Concrete fix: verify the shipped scrub runs before AV request.text_prompt; if not, add upstream sanitization or an AV-specific upstream text_prompt source derived from character appearance/framing, not quoted dialogue. Template shape: “head-and-shoulders in-character portrait of the established appearance, visible face/mouth, period-accurate costume and setting, cinematic 35mm lighting; subtle natural speech/lip motion driven by the supplied audio.” Must not include quoted line text, captions, stage directions, character display names used vocatively, or “on-screen text”.

4. [Open questions/pass03 PROMPTS + OTR_LTX_RADIO_PROMPT] nodes/_otr_video_engines/render_driver.py [VERIFY-AT-BUILD]: honor the existing OTR_LTX_RADIO_PROMPT lane for announcer_visual upstream, not in the adapter, and do not add OTR_LTX_AV_PROMPT. Grounding only shows comments referencing render-driver role-scoped LTX radio-open behavior; the implementation is not excerpted. Concrete fix: same env override should populate request.text_prompt for announcer_visual regardless of whether selected engine is ltx_video or ltx_av_talk; eng_ltx_av.py remains oblivious.

5. [Open questions/pass03 PROMPTS + NEGATIVE PROMPT] nodes/_otr_video_engines/eng_ltx_av.py: define a negative-prompt policy before coding. Concrete fix: default to the eng_ltx_video baseline verbatim unless request.negative_prompt is present. Do not remove “text”: captions are burned later, and generated frames should still avoid text. Do not initially extend with extra “frozen/still image” terms; “static” is already present. Add extension only if M0/P1 shows audio-ignoring inert renders.

SHOULD-CONSIDER:
1. [PROMPT SOURCES] Current grounded source map:
   - character_video visual text_prompt: OTR_ShotLock.derive_creative_directives -> finish_visual_prompt(meta, text_prompt). True in grounding.
   - character/announcer portrait init_image prompts: OTR_MetaBriefImagePromptGen.derive_image_prompts -> person guard -> finish_visual_prompt. True in grounding, but this is the init_image prompt chain, not the AV adapter’s text_prompt.
   - announcer_visual text_prompt: not produced by ShotLock in grounding; must come from render_driver scene/radio-open path. VERIFY-AT-BUILD.
   - music_visual text_prompt: not produced by ShotLock creative in grounding; must come from render_driver scene composer. VERIFY-AT-BUILD.

2. [MUSIC PROMPT CONTENT] For ltx_av_music, reuse the existing brief-grounded LTX scene composition path: get_story_brief_ltx fragment, finish_visual_prompt, max ~240 chars, NO_TEXT_CLAUSE preserved. Small additive tail is enough: “subtle audio-reactive motion / camera breathes with the soundtrack.” Do not stuff beat/rhythm vocabulary; the audio conditioning should carry timing [ASSUMPTION pending M0 node behavior].

3. [PROMPT BUDGET] Keep the 220-240 char finished-prompt budget for consistency and cache/hash stability. LTX-2.3’s larger encoder may tolerate more text [ASSUMPTION], but expanding only the AV lane creates prompt drift with no grounded benefit yet.

4. [PERSON-GUARD INTERPLAY] Do not repeat the portrait person guard in eng_ltx_av.py. Grounding shows OTR_MetaBriefImagePromptGen already enforces _depicts_person before finishing portrait prompts. Repeating it in the video adapter would be scope creep and could reject valid finished visual prompts.

5. [TALKING-HEAD NEGATIVE] If P1 shows frozen mouths despite audio conditioning, append after the baseline, not before it: “frozen face, still image, unmoving mouth”. Do not ship that extension blindly; it may over-constrain portrait stability.

OPEN-QUESTIONS:
1. nodes/_otr_video_engines/render_driver.py: exact current code path that builds ltx_video text_prompt, including get_story_brief_ltx / finish_visual_prompt / NO_TEXT_CLAUSE / 240-char cap, is not excerpted. VERIFY-AT-BUILD.

2. nodes/_otr_video_engines/render_driver.py: exact implementation of OTR_LTX_RADIO_PROMPT is not excerpted. VERIFY-AT-BUILD that it is role-scoped and can feed ltx_av_talk announcer_visual via request.text_prompt.

3. Stage-direction and self-vocative scrub location is not present in the provided ShotLock excerpt. VERIFY-AT-BUILD that the scrub is upstream of all AV request.text_prompt construction.

4. LTX-AV node behavior with long prompts/audio conditioning is M0-only. Do not change prompt cap or negative prompt until probe evidence exists.

CUT THESE:
1. OTR_LTX_AV_PROMPT sibling env. It forks the restored radio/brief pipeline and creates two operator override semantics for the same announcer role.

2. Any eng_ltx_av.py import of _otr_story_brief_helpers for prompt composition. The adapter has enough request fields; importing brief logic violates the locked “join, not fork” rule.

3. Adapter-level person-guard validation. Portrait generation already enforces it before init_image exists; video prompt validation should stay upstream.