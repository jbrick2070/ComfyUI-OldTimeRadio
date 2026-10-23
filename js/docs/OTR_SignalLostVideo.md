<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# Signal Lost Video

Draws the procedural radio-dial video straight from the episode audio, with no AI model. It is the floor every other lane falls back to, and a lane of its own on small machines.

## What it does

Renders the procedural CRT radio-floor video directly from episode audio, without AI generation: an audio-reactive grid, CRT scanlines and vignette, and the episode title cards, on CPU at broadcast resolution. It sets the episode's video length and is the floor the composite falls back to under gaps. Adjust the frame rate or the output resolution.

## Inputs

The shipped workflows set these; most people never change them.

**`audio`** -- a connection from another node (`AUDIO`).

Master episode audio: sets the video length and drives the CRT picture's brightness and grain.

**`script_json`** -- text, as many lines as you like; starts as `[]`.

Parsed script JSON (pipeline compat)

**`news_used`** -- text, as many lines as you like; starts as `[]`.

News JSON (pipeline compat)

### Optional

**`fps`** -- a whole number from 12 to 60; starts as `25`.

Frames per second. Keep it equal to the rate on Caption Burn and Mux and Publish -- they are timing declarations, not converters.

**`resolution`** -- a choice from the dropdown.

Procgen output resolution. 1920x1080 = delivery res (BUG-030 Phase B default). 832x480 was the prior default (rendered cheap, then upscaled with everything else; legacy mode). 1280x720 / 854x480 / 3840x2160 retained for one-off needs.

## Outputs

- `video_path` (`STRING`)
- `title_card_plan_json` (`STRING`)

This is an output node: the workflow runs to reach it.

## Read more

- [VIDEO_MODELS.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/VIDEO_MODELS.md)

Class `OTR_SignalLostVideo`, category `OldTimeRadio`.
