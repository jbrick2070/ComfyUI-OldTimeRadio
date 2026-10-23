<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# Shot Lock (video plan authority)

Turns the frozen audio timeline into a shot plan: which beat is filmed by which lane, for how many frames, from which still. It is the video authority every render node reads.

## What it does

Locks the video plan and shot budget against the frozen audio timeline, acting as the video authority. Runs creative prompt derivation for character-bearing shots, partitions the episode into execution groups, and stamps the video plan into the ledger. Wire upstream script, audio-done gate, and video policy inputs to coordinate video planning.

## Inputs

The shipped workflows set these; most people never change them.

**`script_json`** -- a connection from another node (`STRING`).

Frozen ledger JSON (OTR_LedgerFreezeCascade out1 script_json). ShotLock stamps a video section into it.

**`audio_done`** -- a connection from another node (`STRING`).

Audio-done gate (OTR_EpisodeAssembler out3). Wiring it orders ShotLock AFTER audio timing freezes so the clip budget is bound against the real timeline. Opaque STRING.

**`video_policy_json`** -- a connection from another node (`STRING`).

Per-role selection policy from OTR_VideoDirector.

### Optional

**`gate_in`** -- a connection from another node (`STRING`).

Optional ordering signal (opaque STRING).

**`direction_json`** -- text, as many lines as you like; starts as empty.

Written by Radio Play Studio's shot editor. Leave empty.

## Outputs

- `patched_ledger_json` (`STRING`)
- `video_revision` (`INT`)
- `shot_report` (`STRING`)
- `done` (`STRING`)
- `episode_id` (`STRING`)

## Read more

- [VIDEO_MODELS.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/VIDEO_MODELS.md)

Class `OTR_ShotLock`, category `OldTimeRadio/v2/video`.
