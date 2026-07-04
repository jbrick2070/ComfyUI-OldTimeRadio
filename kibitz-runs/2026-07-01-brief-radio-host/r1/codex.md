VERDICT: no. The plan’s core story says “HuMo-hosted brief-driven radio,” but the current architecture explicitly forbids HuMo on announcer/music and the saved workflow would not exercise the proposed host path.

MUST-FIX BEFORE BUILD:
1. [3.3/3.4/5] The plan reintroduces HuMo for announcer/music while the codebase’s current invariant says those roles must never dispatch HuMo. `C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_speaker_role.py:85` defines announcer/music as never-HuMo, `nodes/_otr_video_engines/render_driver.py:821` redirects announcer/music HuMo-family picks to `ltx_audio_in`, and `nodes/_otr_video_engines/eng_humo.py:96` says HuMo serves only `character_video`. Concrete fix: explicitly declare this plan as a replacement of the 2026-05-01/2026-06-30 invariant, add an opt-in toggle with clear semantics, and require a visual acceptance gate proving “radio body remains visible, not generic human host” before making it default.

2. [2/3.4/5] The real workflow will not show the proposed radio host. In `C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/workflows/otr_scifi_16gb_full.json`, node 87 `OTR_VideoDirector` has `announcer_video_model` and `music_video_model` saved as `viz_green`; the plan also says `viz_* -> no still at all`. Code-only changes would be dormant in the source-of-truth workflow. Concrete fix: either update the workflow JSON to the intended host engine path in the same build, or narrow the goal to an opt-in experiment and name the exact operator control/env override that activates it.

3. [3.1/6] “Reuse `get_era_tail` / `finish_visual_prompt` instead of a form map” does not satisfy “derive radio FORM.” `C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_story_brief_helpers.py:259` builds an atmosphere/palette/lighting tail, and `:458` only appends that tail to an existing subject. It does not produce nouns like “bakelite radio,” “field transceiver,” or “space-station comms console.” Concrete fix: add a small explicit `radio_form_from_meta(meta)` that reads `meta.style`, setting, atmosphere, and palette, returns a form noun phrase, and then uses `get_era_tail` only as finishing texture.

4. [3.2/6] The plan misidentifies the “radio-grounding gate.” The regex at `C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/otr_meta_brief_image_prompt.py:655` is `_GEAR_WORDS`, a scrubber for character portraits, not a positive radio/comms validator. The actual synthetic-announcer consistency gate at `:763-775` checks word overlap against appearance, not “reads as radio.” Concrete fix: add a real positive gate for radio-host prompts that accepts radio/comms/transceiver/console language and rejects pure face-only prompts.

5. [3.3/3.4] The plan does not define the schema/routing for a HuMo face-radio still, especially for music beats with no character id. Current HuMo init selection reads `_portrait_index(ledger).get(char_id)` at `C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/render_driver.py:921`; mesh identity is separate and only covers mesh fodder via `MESH_RADIO_HOST_SUBJECT_ID` in `nodes/otr_meta_brief_image_prompt.py:913-918`. Concrete fix: define a distinct `radio_host_portrait` object/id, how it is minted once per episode, and how announcer/music HuMo requests resolve it without contaminating mesh’s faceless radio object.

6. [3.1/5] “Hard-negative baby” is not supported by the current still dispatch path. `nodes/otr_meta_brief_image_prompt.py:188-190` says there is no per-object negative channel for normal stills, and `nodes/otr_image_gen_dispatcher.py:564-574` builds the generation request without passing `negative_prompt`; only mesh fodder adds one at `nodes/otr_meta_brief_image_prompt.py:927`. Concrete fix: make “adult, mature face” a required positive prompt token now, or extend the image-object schema and dispatcher to carry `negative_prompt` for portrait/radio-host objects.

SHOULD-FIX:
1. [3.4] “`ltx_audio_in` deferred” conflicts with current architecture: `ltx_audio_in` is the default role engine for `music_visual` and `announcer_visual` in `C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_video_engines/eng_ltx_av.py:856`, and render_driver redirects forbidden HuMo picks to it at `nodes/_otr_video_engines/render_driver.py:855`. Decide whether this pass changes LTX bookends, leaves them unchanged, or only changes HuMo opt-in.

2. [6] Face identity across bookends is listed as an open question, but it is central to the stated “radio hosts / sings in-world” arc. Concrete fix: promote it to an invariant: one episode-level radio-host face seed/id reused for open/inter/close, separate from mesh `radio_host`.

3. [1/7] The reference renders are cited as proof, but only PNG files are listed in `C:/Users/jeffr/Documents/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/docs/2026-07-01-overnight/`; [ASSUMPTION] I do not see a prompt/seed/brief manifest in the plan. Concrete fix: attach the exact prompt, seed, model, and source brief excerpt for v3 so the target is reproducible.

4. [5] The invariants do not include a regression for “generic human host instead of radio-host,” despite that being the exact historical failure cited in `nodes/_otr_video_engines/render_driver.py:781-788`. Concrete fix: add a visual/textual prompt assertion and a manual QA checklist item before acceptance.

OPTIONAL / NICE-TO-HAVE:
- Add a short glossary distinguishing “radio object,” “radio-host face still,” “scene still,” and “mesh fodder” so the plan stops overloading “radio-host still.”
- Record the operator toggle name in the plan before implementation.

CUT THESE (scope / over-engineering):
1. [6] Cut the LLM-call option for RADIO_FORM in this pass. The repo already has deterministic meta/style fields, and the immediate defect is hardcoded 1940s form; an LLM form resolver adds nondeterminism before the routing problem is solved.

2. [3.2] Cut “replace every fixed anchor” as a blanket goal. It is too broad and risks mixing HuMo face prompts, mesh fodder prompts, and LTX scene prompts. Replace only the specific surfaces selected by the final routing decision.

3. [3.3] Cut any default-on HuMo bookend promotion until the opt-in experiment proves the v3 “face embedded in radio body” survives actual HuMo animation. The safe early build is toggleable, with the existing animated-radio path preserved.