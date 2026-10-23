<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# CreditsRoll (late unified viewer credits, silent tail)

Appends the credit roll, naming the models and voices that made the episode, to the tail of the silent video.

## What it does

Renders and appends a late retro sci-fi console credits sequence to the tail of the video before final audio muxing. Displays episode metadata, AI model receipts, ledger statistics, cast assignments, and the full scrolling transcript over a held final video frame. Outputs the extended silent video along with the declared credits duration so downstream mux timing guards remain aligned.

## Inputs

The shipped workflows set these; most people never change them.

**`video_path`** -- a connection from another node (`STRING`).

The finished video, with captions already burned in. Comes from Caption Burn.

**`clip_manifest_json`** -- a connection from another node (`STRING`).

Node 92 slot 1 (third fan-out consumer).

## Outputs

- `video_with_credits_path` (`STRING`)
- `declared_credits_tail_s` (`FLOAT`)
- `report` (`STRING`)

## Read more

- [RUN.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/RUN.md)

Class `OTR_CreditsRoll`, category `OldTimeRadio/v2/video`.
