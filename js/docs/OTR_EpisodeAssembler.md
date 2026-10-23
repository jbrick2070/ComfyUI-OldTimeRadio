<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# Episode Assembler

Puts the dialogue track and the theme cues together into the episode's master audio, the sound the video is cut to.

## What it does

Assembles sequenced scene dialogue with opening and closing theme music into the final episode audio track. Normalizes broadcast delivery loudness, writes the master WAV file, and emits the topological audio-done signal that unblocks video generation. Adjust opening and closing theme durations, crossfade timing, or connect custom music cue audio and manifest inputs.

## Inputs

The shipped workflows set these; most people never change them.

**`scene_audio`** -- a connection from another node (`AUDIO`).

Sequenced dialogue audio stream from SceneSequencer.

### Optional

**`opening_theme_audio`** -- a connection from another node (`AUDIO`).

Opening theme audio stream (used when no music cue manifest is wired).

**`closing_theme_audio`** -- a connection from another node (`AUDIO`).

Closing theme audio stream (used when no music cue manifest is wired).

**`opening_duration_sec`** -- a number from 0.0 to 60.0; starts as `10.0`.

Max duration of opening theme

**`closing_duration_sec`** -- a number from 0.0 to 60.0; starts as `8.0`.

Max duration of closing theme

**`crossfade_ms`** -- a whole number from 0 to 3000; starts as `500`.

Crossfade between theme and content

**`music_cue_audio`** -- a connection from another node (`AUDIO`).

Padded AUDIO batch of every music cue from OTR_StableAudioTheme.

**`music_cue_manifest_json`** -- text, as many lines as you like; starts as empty.

Cue manifest JSON from OTR_StableAudioTheme (maps placement -> batch row).

**`video_policy_json`** -- text, as many lines as you like; starts as empty.

Video policy JSON from OTR_VideoDirector. Read for ONE question: is any role on the LTX 2.5 foley lane? If so this node writes an UN-LEVELLED provisional master WAV and OTR_MasterAudioMux does the single delivery loudness pass after mixing the bed in. Empty -> today's behaviour, unchanged.

**`replay_descriptor`** -- text, as many lines as you like; starts as empty.

v2_ledger_json from OTR_LedgerFreezeCascade. On a canonical replay the frozen master WAV is copied and verified against the bundle's SHA-256; otherwise ignored.

## Outputs

- `episode_audio` (`AUDIO`)
- `output_path` (`STRING`)
- `episode_info` (`STRING`)
- `audio_done` (`STRING`)

This is an output node: the workflow runs to reach it.

## Read more

- [ARCHITECTURE.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/ARCHITECTURE.md)

Class `OTR_EpisodeAssembler`, category `OldTimeRadio`.
