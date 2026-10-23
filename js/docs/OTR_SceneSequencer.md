<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# Scene Sequencer

Lays the spoken lines end to end in script order, with the pauses between them, into one dialogue track.

## What it does

Sequences voiced dialogue clips from characters and the announcer end-to-end against the script ledger. Resamples clips to 48 kHz, manages inter-line pauses and pacing, and records exact line timing back into the ledger. Adjust start and end line ranges, or shift dialogue timing with dialogue offset.

## Inputs

The shipped workflows set these; most people never change them.

**`script_json`** -- text, as many lines as you like; starts as `[]`.

The script, as JSON, from the Story Writer.

### Optional

**`tts_audio_clips`** -- a connection from another node (`AUDIO`).

Character dialogue, already voiced, matched to the script in order. Without it those lines come out as silence. Announcer lines do not belong here -- they arrive on announcer_audio_clips.

**`announcer_audio_clips`** -- a connection from another node (`AUDIO`).

Announcer lines, already voiced, used in script order. They ride their own bus so the bookends stay separate from the cast.

**`start_line`** -- a whole number from 0 to 9999; starts as `0`.

First line to render (for chunked processing)

**`end_line`** -- a whole number from 1 to 9999; starts as `999`.

Last line to render

**`dialogue_offset_ms`** -- a number from -500.0 to 500.0; starts as `0.0`.

Shift all dialogue clips on the timeline (ms). Positive = delay, negative = advance.

## Outputs

- `scene_audio` (`AUDIO`)
- `render_log` (`STRING`)

## Read more

- [ARCHITECTURE.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/ARCHITECTURE.md)

Class `OTR_SceneSequencer`, category `OldTimeRadio`.
