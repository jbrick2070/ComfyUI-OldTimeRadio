<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# VideoDirector (per-role model select)

Chooses the video lane for each role, the announcer, the characters and the music beats, and the still each lane starts from. The randomize switches draw one local lane for the whole episode instead.

## What it does

Configures visual model selections and rendering policies across announcer, music, and character roles. Sets per-role video models, source still-image models, canvas dimensions, frame rates, and render length ceilings before emitting the video policy. Change video or still model dropdowns for any role, adjust canvas resolution, or cap per-clip render frames.

## Inputs

On the shipped workflows these are the widgets a person changes; the ones on the app form are named as the form names them.

**Announcer - video** (`announcer_video_model`) -- a choice from the dropdown.

Video model for ANNOUNCER beats (role A).

**Music - video** (`music_video_model`) -- a choice from the dropdown.

Video model for MUSIC beats (role B).

**Characters - video** (`character_video_model`) -- a choice from the dropdown.

Video model for CHARACTER beats (face + audio, role C). Uses the saved dropdown value; profiles only retarget this when explicitly applied.

**Announcer - still** (`announcer_image_model`) -- a choice from the dropdown.

The still the announcer's video is built from. Nothing downloads while this role's video lane is procedural; z_image_turbo is about 19 GB the first time a lane actually needs it.

**Music - still** (`music_image_model`) -- a choice from the dropdown.

The still music beats are built from. Nothing downloads while this role's video lane is procedural; z_image_turbo is about 19 GB the first time a lane actually needs it.

**Characters - still** (`character_image_model`) -- a choice from the dropdown.

The still character beats are built from -- the shots that carry a face. Nothing downloads while this role's video lane is procedural; z_image_turbo is about 19 GB the first time a lane actually needs it.

**`fps`** -- a whole number from 1 to 60; starts as `25`.

Frame rate the video plan budgets beats at. Profile/platform-owned; downstream stages read it from the plan, so change it via profile, not per-episode.

**`canvas_w`** -- a whole number from 16 to 7680; starts as `832`.

Video plan canvas width (px). Profile/platform-owned per hardware tier; engines may declare their own native canvas that supersedes this at render.

**`canvas_h`** -- a whole number from 16 to 4320; starts as `480`.

Video plan canvas height (px). Profile/platform-owned -- see canvas_w.

### Optional

**`custom_models_json`** -- text, as many lines as you like; starts as `{}`.

When a role is set to '+ Add Custom Model', map the role key to a custom engine id here -- video AND image roles, e.g. {"character_video_model": "my_engine", "music_image_model": "my_image_engine"}. This is the one declaration the queue check and the render both read.

**`gate_in`** -- a connection from another node (`STRING`).

Optional ordering signal (opaque STRING).

**`device_policy`** -- a choice from the dropdown; starts as `default`.

Recorded on the episode receipt only. It does not choose the device -- each engine resolves that itself at render time.

**`dtype_policy`** -- a choice from the dropdown; starts as `fp8_ok`.

Recorded on the episode receipt only. No video engine currently reads it to choose a precision.

**`max_render_frames`** -- a whole number from 0 to 240; starts as `0`.

Absolute per-clip RENDER-length ceiling in frames for local video engines (0 = unpinned = the engine's own maximum). Beat LENGTH is never changed by it -- this caps what an engine renders, never what the episode plays. TWO LANES CONSUME IT DIFFERENTLY. ltx_8gb takes it as a COVERAGE-PLANNING cap: the beat is partitioned into real clips of at most this many frames and chained, so the ceiling decides how many clips cover the beat. The 8GB Wan tier takes it as an adapter-side NATIVE cap: it pins 17, renders 17 real frames, and ping-pong-extends them to the beat's audio length rather than requesting the whole 177-frame span and dying in the VRAM cost model. A value at or above an engine's own maximum has no effect on that engine.

**Randomize video (non-cloud)** (`roll_video_lanes`) -- on or off; starts as off.

Yes: one local video lane, drawn at random, renders the whole episode, in place of all three video pickers. Cloud lanes are never drawn, nor a lane this machine cannot run. A lane it draws downloads its models the first time, as if picked by hand. Suggested for 16 GB cards and up.

**Randomize stills (non-cloud)** (`roll_still_models`) -- on or off; starts as off.

Yes: one local image model, drawn at random, makes every still in the episode, in place of all three still pickers. Cloud models are never drawn. Nothing is drawn when no video lane in the episode uses a still. Suggested for 16 GB cards and up.

**Video lanes to roll** (`video_roll_pool`) -- text; starts as empty.

Read only when Randomize video is on. Tick the lanes to roll among (or type them, separated by commas); none rolls among every local lane that can run on this machine, one is simply that lane. A ticked lane this machine cannot run is left out. Recorded at meta.video_lane_roll.

**Still models to roll** (`still_roll_pool`) -- text; starts as empty.

Read only when Randomize stills is on. Tick the image models to roll among (or type them, separated by commas); none rolls among every local model that can run on this machine, one is simply that model. Recorded at meta.still_model_roll.

## Outputs

- `video_policy_json` (`STRING`)

## Read more

- [VIDEO_MODELS.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/VIDEO_MODELS.md)
- [MACHINES.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/MACHINES.md)

Class `OTR_VideoDirector`, category `OldTimeRadio/v2/video`.
