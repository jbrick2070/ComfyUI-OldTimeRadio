# ENGINE MATRIX -- the per-model requirements record

<!-- GENERATED FILE. Do not edit by hand.
     Regenerate:  python tools/engine_matrix.py
     Drift gate:  python tools/engine_matrix.py --check  (also a suite test)
-->

Every number here is read from the LIVE engine registry, so it cannot drift
from the adapters without the suite noticing. Written for multi-clip coverage
chunk 7a (2026-07-26), when every registered engine gained a declared
`FrameContract` and the per-engine opt-in was removed.

## How to read the clip window

`clip frames` is what ONE render call may legally produce. `step N` means the
ladder is arithmetic -- `min + k*N` -- so lengths off that grid have no legal
render and the planner renders the next length up and trims. `menu:` means the
provider serves a fixed set of lengths and nothing between them.

`clip seconds` is that window divided by `fps`. Where `fps` reads `canvas`, the
engine renders at whatever rate the canvas asks for and the seconds column is
meaningless rather than merely unknown -- it is marked `unbounded`.

**Google runs at 24 fps against a 25 fps canvas.** Veo's published menu is 4/6/8
SECONDS, which is 96/144/192 frames. The contract counts frames.

## What is NOT here, and why

* **The prompt text.** It is composed per episode by the story pass and varies
  per beat, so it is not a per-model requirement. What is recorded is the
  prompt CONTRACT: whether the lane takes text, and which conditioner rewrites
  it before it is sent.
* **A resolution number for the local lanes.** They negotiate size per render
  from the canvas and the profile (`_aspect_plan` / `_aspect_policy`). Printing
  a number here would be printing one the code never promised.
* **The rate the cloud providers actually DELIVER at.** No adapter declares it
  and nothing in the tree reads it back; the cloud rows convert seconds at the
  canvas's 25 fps because that is what `_CloudVideoBase._duration_seconds`
  itself assumes. This is a real open gap, not an omission.


## The matrix

| engine | side | family | aspect | resolution | clip frames | clip seconds | fps | continuity | tail trim |
|---|---|---|---|---|---|---|---|---|---|
| cloud_kling_avatar | provider | audio_driven_face | wide | provider default (none sent) | 50-7500 | 2-300 s | 25 | soft_reference | yes |
| cloud_seedance_2 | provider | audio_conditioned_video | wide | env OTR_CLOUD_SEEDANCE_RESOLUTION, default 720p | 100-375 step 25 | 4-15 s | 25 | soft_reference | yes |
| cloud_vidu_q2_pro_fast_720p | provider | image_to_video | wide | 720p (fixed) | 25-250 step 25 | 1-10 s | 25 | soft_reference | yes |
| cloud_vidu_q2_pro_fast_720p_sfx | provider | image_to_video | wide | 720p (fixed) | 25-250 step 25 | 1-10 s | 25 | soft_reference | yes |
| cloud_wan_i2v | provider | image_to_video | wide | env OTR_CLOUD_WAN_RESOLUTION, default 720P | 50-375 step 25 | 2-15 s | 25 | soft_reference | yes |
| cloud_wan_i2v_audio | provider | audio_conditioned_video | wide | env OTR_CLOUD_WAN_RESOLUTION, default 720P | 50-375 step 25 | 2-15 s | 25 | soft_reference | yes |
| google_omni_video | provider | text_to_video | wide | 720p (fixed) | 75-250 | 3-10 s | 25 | none | yes |
| google_veo_video | provider | text_to_video | wide | env OTR_GOOGLE_VEO_RESOLUTION, default 720p | menu: 100, 150, 200 | menu: 4, 6, 8 s | 25 | soft_reference | yes |
| google_vid_sfx_omni | provider | text_to_video | wide | 720p (fixed) | 75-250 | 3-10 s | 25 | none | yes |
| google_vid_sfx_veo_fast | provider | text_to_video | wide | env OTR_GOOGLE_VEO_RESOLUTION, default 720p | menu: 100, 150, 200 | menu: 4, 6, 8 s | 25 | soft_reference | yes |
| google_vid_sfx_veo_lite | provider | text_to_video | wide | env OTR_GOOGLE_VEO_RESOLUTION, default 720p | menu: 100, 150, 200 | menu: 4, 6, 8 s | 25 | soft_reference | yes |
| google_vid_sfx_veo_pro | provider | text_to_video | wide | env OTR_GOOGLE_VEO_RESOLUTION, default 720p | menu: 100, 150, 200 | menu: 4, 6, 8 s | 25 | soft_reference | yes |
| humo | local | audio_driven_face | portrait | canvas-negotiated (_aspect_plan) | 33-177 step 4 | 1.32-7.08 s | 25 | soft_reference | yes |
| humo_1.7B | local | audio_driven_face | portrait | canvas-negotiated (_aspect_plan) | 33-177 step 4 | 1.32-7.08 s | 25 | soft_reference | yes |
| humo_1.7B_169 | local | audio_driven_face | wide | canvas-negotiated (_aspect_plan) | 33-177 step 4 | 1.32-7.08 s | 25 | soft_reference | yes |
| humo_14B_169 | local | audio_driven_face | wide | canvas-negotiated (_aspect_plan) | 33-49 step 4 | 1.32-1.96 s | 25 | soft_reference | yes |
| ltx_8gb | local | image_to_video | wide | canvas-negotiated (_aspect_plan) | 9-161 step 8 | 0.36-6.44 s | 25 | strict_first_frame | yes |
| ltx_audio_in | local | audio_conditioned_video | wide | canvas | 9-497 step 8 | 0.36-19.88 s | 25 | soft_reference | yes |
| ltx_video | local | text_to_video | wide | canvas | 9-169 step 8 | 0.36-6.76 s | 25 | strict_first_frame | yes |
| mesh_stage | local | image_to_video | wide | canvas | 1.. (no ceiling) | unbounded | canvas | none | yes |
| still_flat | local | static_image_gen | wide | canvas | 1.. (no ceiling) | unbounded | canvas | none | yes |
| still_motion | local | static_motion | wide | canvas | 1.. (no ceiling) | unbounded | canvas | none | yes |
| still_pan | local | static_image_gen | wide | canvas | 1.. (no ceiling) | unbounded | canvas | none | yes |
| still_word | local | static_image_gen | wide | canvas | 1.. (no ceiling) | unbounded | canvas | none | yes |
| viz_camera | local | abstract | wide | canvas | 1.. (no ceiling) | unbounded | 25 | none | yes |
| viz_green | local | abstract | wide | canvas | 1.. (no ceiling) | unbounded | 25 | none | yes |
| viz_mxc_cpu | local | abstract | wide | canvas | 1.. (no ceiling) | unbounded | 25 | none | yes |
| viz_mxc_mandala | local | abstract | wide | canvas | 1.. (no ceiling) | unbounded | 25 | none | yes |
| wan_i2v | local | image_to_video | wide | canvas-negotiated (_aspect_plan) | 33-177 step 4 | 1.32-7.08 s | 25 | strict_first_frame | yes |
| wan_ti2v | local | image_to_video | wide | canvas-negotiated (_aspect_plan) | 17-177 step 4 | 0.68-7.08 s | 25 | strict_first_frame | yes |
| word_razzle | provider | image_to_video | wide | env OTR_CLOUD_PIXVERSE_QUALITY, default 1080p | menu: 125, 200 | menu: 5, 8 s | 25 | soft_reference | yes |

## Inputs and prompt contract

| engine | required inputs | prompt contract |
|---|---|---|
| cloud_kling_avatar | init_image, audio_ref | text_prompt OPTIONAL (sent when present) |
| cloud_seedance_2 | init_image, audio_ref, text_prompt | text_prompt REQUIRED |
| cloud_vidu_q2_pro_fast_720p | init_image, text_prompt | text_prompt REQUIRED |
| cloud_vidu_q2_pro_fast_720p_sfx | init_image, text_prompt | text_prompt REQUIRED |
| cloud_wan_i2v | init_image, text_prompt | text_prompt REQUIRED |
| cloud_wan_i2v_audio | init_image, audio_ref, text_prompt | text_prompt REQUIRED |
| google_omni_video | text_prompt | text_prompt REQUIRED |
| google_veo_video | text_prompt | text_prompt REQUIRED |
| google_vid_sfx_omni | text_prompt | text_prompt REQUIRED |
| google_vid_sfx_veo_fast | text_prompt | text_prompt REQUIRED |
| google_vid_sfx_veo_lite | text_prompt | text_prompt REQUIRED |
| google_vid_sfx_veo_pro | text_prompt | text_prompt REQUIRED |
| humo | audio_ref, init_image | text_prompt OPTIONAL (sent when present) |
| humo_1.7B | audio_ref, init_image | text_prompt OPTIONAL (sent when present) |
| humo_1.7B_169 | audio_ref, init_image | text_prompt OPTIONAL (sent when present) |
| humo_14B_169 | audio_ref, init_image | text_prompt OPTIONAL (sent when present) |
| ltx_8gb | init_image | text_prompt OPTIONAL (sent when present) |
| ltx_audio_in | text_prompt, audio_ref, init_image | text_prompt REQUIRED |
| ltx_video | text_prompt | text_prompt REQUIRED |
| mesh_stage | init_image | no text input |
| still_flat | text_prompt | text_prompt REQUIRED |
| still_motion | text_prompt | text_prompt REQUIRED |
| still_pan | text_prompt | text_prompt REQUIRED |
| still_word | text_prompt | text_prompt REQUIRED |
| viz_camera | - | no text input |
| viz_green | audio_ref | no text input |
| viz_mxc_cpu | - | no text input |
| viz_mxc_mandala | - | no text input |
| wan_i2v | init_image | text_prompt OPTIONAL (sent when present) |
| wan_ti2v | init_image | text_prompt OPTIONAL (sent when present) |
| word_razzle | init_image, text_prompt | text_prompt REQUIRED |

## Still requirements

Read as `kind/aspect/when-required`, straight off each adapter's
own `still_plan`. `inherit_engine` means the still is minted at
the engine's own `aspect` column above.

| engine | stills |
|---|---|
| cloud_kling_avatar | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/always |
| cloud_seedance_2 | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| cloud_vidu_q2_pro_fast_720p | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| cloud_vidu_q2_pro_fast_720p_sfx | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| cloud_wan_i2v | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| cloud_wan_i2v_audio | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| google_omni_video | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| google_veo_video | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| google_vid_sfx_omni | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| google_vid_sfx_veo_fast | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| google_vid_sfx_veo_lite | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| google_vid_sfx_veo_pro | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| humo | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/always |
| humo_1.7B | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/always |
| humo_1.7B_169 | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/always |
| humo_14B_169 | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/always |
| ltx_8gb | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| ltx_audio_in | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never; portrait/wide/when_engine_talking; portrait/inherit_engine/when_engine_talking |
| ltx_video | scene_open/wide/when_ltx_i2v_enabled; scene_beat/wide/when_ltx_i2v_enabled; scene_character/wide/when_ltx_i2v_enabled; portrait/inherit_engine/never |
| mesh_stage | mesh_fodder/wide/always; scene_background_plate/wide/always; portrait/inherit_engine/never |
| still_flat | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| still_motion | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| still_pan | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| still_word | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| viz_camera | none |
| viz_green | none |
| viz_mxc_cpu | none |
| viz_mxc_mandala | none |
| wan_i2v | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| wan_ti2v | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| word_razzle | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |

## Counts

* registered engine names: **31**
* provider-side: **13**
* local: **18**
* can chain (strict_first_frame): **4**
