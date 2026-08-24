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

**Two lanes run at 24 fps against a 25 fps canvas, and every contract in this
page counts CANVAS frames.** Veo's published menu is 4/6/8 SECONDS, which is
**100/150/200** frames at 25 -- not 96/144/192, which is the same menu counted
at Veo's own 24 and is a length nothing downstream can produce. (This paragraph
said 96/144/192 until 2026-08-12, contradicting the `google_veo_video` row in
the table above it.) The two lanes reach 25 by different routes:
`google_veo_video` by the provider-side duration-preserving resample in
`cloud_media_canonical`, and `minimax_h3_video` by its own nearest-source-frame
index map applied to the decoded batch before the encoder -- the local encoder
can LABEL a frame rate but never resample one, so a 24 fps render shipped
without that map would play ~4% short.

## What is NOT here, and why

* **The prompt text.** It is composed per episode by the story pass and varies
  per beat, so it is not a per-model requirement. What is recorded is the
  prompt CONTRACT: whether the lane takes text, and which conditioner rewrites
  it before it is sent.
* ~~**A resolution number for the local lanes.**~~ **RETRACTED 2026-08-02.**
  This omission was itself the defect. Refusing to print a local resolution
  "because the code never promised one" is exactly how `wan_i2v` came to sit on
  the shared 1472x832 landscape default with no opinion of its own, and how a
  profile asking for 832x480 could fail to reach the render. The number IS
  resolvable -- it is just resolved in three different places -- so the
  **effective canvas** column now walks the same precedence the driver walks and
  names which authority won.
* **The rate the cloud providers actually DELIVER at.** No adapter declares it
  and nothing in the tree reads it back; the cloud rows convert seconds at the
  canvas's 25 fps because that is what `_CloudVideoBase._duration_seconds`
  itself assumes. This is a real open gap, not an omission.


## The matrix

| engine | side | family | aspect | resolution | clip frames | clip seconds | fps | continuity | tail trim |
|---|---|---|---|---|---|---|---|---|---|
| animatediff15_v3_haunted_video | local | text_to_video | wide | canvas | 1.. (no ceiling) | unbounded | 25 | none | yes |
| cloud_kling_avatar | provider | audio_driven_face | wide | provider default (none sent) | 50-7500 | 2-300 s | 25 | soft_reference | yes |
| cloud_seedance_2 | provider | audio_conditioned_video | wide | env OTR_CLOUD_SEEDANCE_RESOLUTION, default 720p | 100-375 step 25 | 4-15 s | 25 | soft_reference | yes |
| cloud_vidu_q2_pro_fast_720p | provider | image_to_video | wide | 720p (fixed) | 25-250 step 25 | 1-10 s | 25 | soft_reference | yes |
| cloud_wan_i2v | provider | image_to_video | wide | env OTR_CLOUD_WAN_RESOLUTION, default 720P | 50-375 step 25 | 2-15 s | 25 | soft_reference | yes |
| cloud_wan_i2v_audio | provider | audio_conditioned_video | wide | env OTR_CLOUD_WAN_RESOLUTION, default 720P | 50-375 step 25 | 2-15 s | 25 | soft_reference | yes |
| fastwan_8gb | local | image_to_video | wide | canvas-negotiated (_aspect_plan) | 17-177 step 4 | 0.68-7.08 s | 25 | strict_first_frame | yes |
| google_omni_video | provider | text_to_video | wide | 720p (fixed) | 75-250 | 3-10 s | 25 | none | yes |
| google_veo_video | provider | text_to_video | wide | env OTR_GOOGLE_VEO_RESOLUTION, default 720p | menu: 100, 150, 200 | menu: 4, 6, 8 s | 25 | soft_reference | yes |
| humo | local | audio_driven_face | portrait | canvas-negotiated (_aspect_plan) | 33-97 step 4 | 1.32-3.88 s | 25 | soft_reference | yes |
| humo_1.7B | local | audio_driven_face | portrait | canvas-negotiated (_aspect_plan) | 33-177 step 4 | 1.32-7.08 s | 25 | soft_reference | yes |
| humo_1.7B_169 | local | audio_driven_face | wide | canvas-negotiated (_aspect_plan) | 33-177 step 4 | 1.32-7.08 s | 25 | soft_reference | yes |
| humo_14B_169 | local | audio_driven_face | wide | canvas-negotiated (_aspect_plan) | 33-97 step 4 | 1.32-3.88 s | 25 | soft_reference | yes |
| ltx25_video | local | image_to_video | wide | canvas | menu: 97 | menu: 3.88 s | 25 | strict_first_frame | yes |
| ltx_8gb | local | image_to_video | wide | canvas-negotiated (_aspect_plan) | 9-161 step 8 | 0.36-6.44 s | 25 | strict_first_frame | yes |
| ltx_audio_in | local | audio_conditioned_video | wide | canvas | 9-497 step 8 | 0.36-19.88 s | 25 | soft_reference | yes |
| ltx_video | local | text_to_video | wide | canvas | 9-169 step 8 | 0.36-6.76 s | 25 | strict_first_frame | yes |
| mesh_stage | local | image_to_video | wide | canvas | 1.. (no ceiling) | unbounded | canvas | none | yes |
| minimax_h3_audio_in | local | audio_conditioned_video | wide | canvas-negotiated (_aspect_plan) | menu: 129, 146, 164, 182, 200, 217, 235, 253, 270, 288, 306, 323, 341, 359, 377 | menu: 5.16, 5.84, 6.56, 7.28, 8, 8.68, 9.40, 10.12, 10.80, 11.52, 12.24, 12.92, 13.64, 14.36, 15.08 s | 25 | soft_reference | yes |
| minimax_h3_video | local | image_to_video | wide | canvas-negotiated (_aspect_plan) | menu: 129, 146, 164, 182, 200, 217, 235, 253, 270, 288, 306, 323, 341, 359, 377 | menu: 5.16, 5.84, 6.56, 7.28, 8, 8.68, 9.40, 10.12, 10.80, 11.52, 12.24, 12.92, 13.64, 14.36, 15.08 s | 25 | strict_first_frame | yes |
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
| animatediff15_v3_haunted_video | text_prompt | text_prompt REQUIRED |
| cloud_kling_avatar | init_image, audio_ref | text_prompt OPTIONAL (sent when present) |
| cloud_seedance_2 | init_image, audio_ref, text_prompt | text_prompt REQUIRED |
| cloud_vidu_q2_pro_fast_720p | init_image, text_prompt | text_prompt REQUIRED |
| cloud_wan_i2v | init_image, text_prompt | text_prompt REQUIRED |
| cloud_wan_i2v_audio | init_image, audio_ref, text_prompt | text_prompt REQUIRED |
| fastwan_8gb | init_image | text_prompt OPTIONAL (sent when present) |
| google_omni_video | text_prompt | text_prompt REQUIRED |
| google_veo_video | text_prompt | text_prompt REQUIRED |
| humo | audio_ref, init_image | text_prompt OPTIONAL (sent when present) |
| humo_1.7B | audio_ref, init_image | text_prompt OPTIONAL (sent when present) |
| humo_1.7B_169 | audio_ref, init_image | text_prompt OPTIONAL (sent when present) |
| humo_14B_169 | audio_ref, init_image | text_prompt OPTIONAL (sent when present) |
| ltx25_video | text_prompt, init_image | text_prompt REQUIRED |
| ltx_8gb | init_image | text_prompt OPTIONAL (sent when present) |
| ltx_audio_in | text_prompt, audio_ref, init_image | text_prompt REQUIRED |
| ltx_video | text_prompt | text_prompt REQUIRED |
| mesh_stage | init_image | no text input |
| minimax_h3_audio_in | audio_ref, init_image | text_prompt OPTIONAL (sent when present) |
| minimax_h3_video | init_image | text_prompt OPTIONAL (sent when present) |
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
| animatediff15_v3_haunted_video | none |
| cloud_kling_avatar | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/always |
| cloud_seedance_2 | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| cloud_vidu_q2_pro_fast_720p | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| cloud_wan_i2v | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| cloud_wan_i2v_audio | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| fastwan_8gb | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| google_omni_video | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| google_veo_video | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| humo | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/always |
| humo_1.7B | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/always |
| humo_1.7B_169 | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/always |
| humo_14B_169 | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/always |
| ltx25_video | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always |
| ltx_8gb | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| ltx_audio_in | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never; portrait/wide/when_engine_talking; portrait/inherit_engine/when_engine_talking |
| ltx_video | scene_open/wide/when_ltx_i2v_enabled; scene_beat/wide/when_ltx_i2v_enabled; scene_character/wide/when_ltx_i2v_enabled; portrait/inherit_engine/never |
| mesh_stage | mesh_fodder/wide/always; scene_background_plate/wide/always; portrait/inherit_engine/never |
| minimax_h3_audio_in | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
| minimax_h3_video | scene_open/wide/always; scene_beat/wide/always; scene_character/wide/always; portrait/inherit_engine/never |
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

## Effective render canvas

The canvas each engine ACTUALLY renders at, and which authority
decided it. `SHARED LANDSCAPE DEFAULT` means the engine expressed
no preference and inherited the composite canvas -- the dead
channel that cost `wan_8gb` a 268-minute leg. `engine _native_dims`
means the adapter sizes itself and IGNORES the request canvas.

| engine | effective canvas | decided by |
|---|---|---|
| animatediff15_v3_haunted_video | 512x288 | declared |
| cloud_kling_avatar | n/a -- renders remotely | see `resolution` column |
| cloud_seedance_2 | n/a -- renders remotely | see `resolution` column |
| cloud_vidu_q2_pro_fast_720p | n/a -- renders remotely | see `resolution` column |
| cloud_wan_i2v | n/a -- renders remotely | see `resolution` column |
| cloud_wan_i2v_audio | n/a -- renders remotely | see `resolution` column |
| fastwan_8gb | 832x480 | declared |
| google_omni_video | n/a -- renders remotely | see `resolution` column |
| google_veo_video | n/a -- renders remotely | see `resolution` column |
| humo | 480x832 | engine _native_dims |
| humo_1.7B | 480x832 | engine _native_dims |
| humo_1.7B_169 | 832x480 | engine _native_dims |
| humo_14B_169 | 832x480 | engine _native_dims |
| ltx25_video | 832x480 | declared |
| ltx_8gb | 512x288 | declared |
| ltx_audio_in | 1024x576 | declared |
| ltx_video | 1024x576 | declared |
| mesh_stage | 1472x832 | declared |
| minimax_h3_audio_in | 864x480 | declared |
| minimax_h3_video | 864x480 | declared |
| still_flat | 1472x832 | shared landscape (by design for this family) |
| still_motion | 1472x832 | shared landscape (by design for this family) |
| still_pan | 1472x832 | shared landscape (by design for this family) |
| still_word | 1472x832 | shared landscape (by design for this family) |
| viz_camera | 1472x832 | shared landscape (by design for this family) |
| viz_green | 1472x832 | shared landscape (by design for this family) |
| viz_mxc_cpu | 1472x832 | shared landscape (by design for this family) |
| viz_mxc_mandala | 1472x832 | shared landscape (by design for this family) |
| wan_i2v | 832x480 | declared |
| wan_ti2v | 832x480 | declared |
| word_razzle | 1472x832 | SHARED LANDSCAPE DEFAULT (unclaimed) |

## Multi-clip behaviour at a 442-frame beat

A 442-frame beat is 17.68 s at 25 fps -- long enough that every
bounded engine must split it. `render` is what the GPU or the
provider is asked to produce; `visible` is what survives to the
cut. They differ when a ladder has no legal length at the target
and the tail is trimmed -- so `render` is what VRAM must afford,
and `visible` is what the audio needs. `re-mints` counts stills
generated fresh per cut: a chained successor begins on its
predecessor's real terminal frame and owns no still, so only a
JUMP plan on a still-consuming lane ever re-mints.

| engine | join | segments (render frames) | render | visible | re-mints |
|---|---|---|---|---|---|
| animatediff15_v3_haunted_video | single | 1: 442 | 442 | 442 | 0 |
| cloud_kling_avatar | single | 1: 442 | 442 | 442 | 0 |
| cloud_seedance_2 | jump | 2: 350, 100 | 450 | 442 | 1 |
| cloud_vidu_q2_pro_fast_720p | jump | 2: 250, 200 | 450 | 442 | 1 |
| cloud_wan_i2v | jump | 2: 375, 75 | 450 | 442 | 1 |
| cloud_wan_i2v_audio | jump | 2: 375, 75 | 450 | 442 | 1 |
| fastwan_8gb | chain | 3: 177, 177, 93 | 447 | 442 | 0 |
| google_omni_video | jump | 2: 250, 192 | 442 | 442 | 1 |
| google_veo_video | jump | 3: 200, 150, 100 | 450 | 442 | 2 |
| humo | jump | 5: 97, 97, 97, 97, 57 | 445 | 442 | 0 |
| humo_1.7B | jump | 3: 177, 177, 89 | 443 | 442 | 0 |
| humo_1.7B_169 | jump | 3: 177, 177, 89 | 443 | 442 | 0 |
| humo_14B_169 | jump | 5: 97, 97, 97, 97, 57 | 445 | 442 | 0 |
| ltx25_video | chain | 5: 97, 97, 97, 97, 97 | 485 | 442 | 0 |
| ltx_8gb | chain | 3: 161, 161, 129 | 451 | 442 | 0 |
| ltx_audio_in | single | 1: 449 | 449 | 442 | 0 |
| ltx_video | chain | 3: 169, 169, 113 | 451 | 442 | 0 |
| mesh_stage | single | 1: 442 | 442 | 442 | 0 |
| minimax_h3_audio_in | jump | 2: 323, 129 | 452 | 442 | 1 |
| minimax_h3_video | chain | 2: 323, 129 | 452 | 442 | 0 |
| still_flat | single | 1: 442 | 442 | 442 | 0 |
| still_motion | single | 1: 442 | 442 | 442 | 0 |
| still_pan | single | 1: 442 | 442 | 442 | 0 |
| still_word | single | 1: 442 | 442 | 442 | 0 |
| viz_camera | single | 1: 442 | 442 | 442 | 0 |
| viz_green | single | 1: 442 | 442 | 442 | 0 |
| viz_mxc_cpu | single | 1: 442 | 442 | 442 | 0 |
| viz_mxc_mandala | single | 1: 442 | 442 | 442 | 0 |
| wan_i2v | chain | 3: 177, 177, 93 | 447 | 442 | 0 |
| wan_ti2v | chain | 3: 177, 177, 93 | 447 | 442 | 0 |
| word_razzle | jump | 3: 200, 125, 125 | 450 | 442 | 2 |

## Frame caps and the evidence behind them

`evidence` lists every `docs/` receipt the adapter's own source
cites. **MISSING** means the adapter cites a document that is not
in this repo -- a safety number nobody can check. This column
exists because the HuMo 49-frame ceiling cited
`docs/2026-06-27-humo-bakeoff`, which has never been in the tree,
and it read exactly like a measured number until someone looked.

| engine | cap | set by | evidence |
|---|---|---|---|
| animatediff15_v3_haunted_video | - | contract max | none cited |
| cloud_kling_avatar | - | contract max | none cited |
| cloud_seedance_2 | - | contract max | none cited |
| cloud_vidu_q2_pro_fast_720p | - | contract max | none cited |
| cloud_wan_i2v | - | contract max | none cited |
| cloud_wan_i2v_audio | - | contract max | none cited |
| fastwan_8gb | - | contract max | docs/2026-07-31-arm-c-fastwan-BUILD-SPEC.md |
| google_omni_video | - | contract max | none cited |
| google_veo_video | - | contract max | none cited |
| humo | 97 | safe_render_frames | docs/evidence |
| humo_1.7B | - | contract max | docs/evidence |
| humo_1.7B_169 | - | contract max | docs/evidence |
| humo_14B_169 | 97 | safe_render_frames | docs/evidence |
| ltx25_video | - | contract max | none cited |
| ltx_8gb | - | contract max | docs/2026-07-20-OTR-video-tiers |
| ltx_audio_in | - | contract max | **MISSING: docs/2026-07-02-canonical-ia2v** |
| ltx_video | - | contract max | docs/evidence |
| mesh_stage | - | contract max | **MISSING: docs/2026-06-11-comfy-native-3d-options** |
| minimax_h3_audio_in | - | contract max | docs/H3_LICENSE_ATTESTATION.md, docs/evidence |
| minimax_h3_video | - | contract max | docs/H3_LICENSE_ATTESTATION.md, docs/evidence |
| still_flat | - | contract max | none cited |
| still_motion | - | contract max | none cited |
| still_pan | - | contract max | none cited |
| still_word | - | contract max | none cited |
| viz_camera | - | contract max | none cited |
| viz_green | - | contract max | **MISSING: docs/2026-06-18-coverage-arch-wiring** |
| viz_mxc_cpu | - | contract max | none cited |
| viz_mxc_mandala | - | contract max | **MISSING: docs/2026-06-30-viz-rainbow** |
| wan_i2v | - | contract max | docs/2026-07-25-still-plans-locked-build-spec.md, docs/evidence |
| wan_ti2v | - | contract max | none cited |
| word_razzle | - | contract max | none cited |

## Counts

* registered engine names: **31**
* provider-side: **8**
* local: **23**
* can chain (strict_first_frame): **7**
