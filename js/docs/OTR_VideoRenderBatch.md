<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# Video Render Batch (A-S7.5 in-process render)

Films every planned shot with the chosen video lanes, one clip per beat, inside the ComfyUI process.

## What it does

Executes the in-process video rendering loop across planned shots using registered video models. Verifies that all required still images exist on disk before rendering, tracks VRAM usage, and emits a frame-accurate clip manifest for composite assembly. Connect the planned ledger, frozen master audio path, and the image-done gate to control rendering.

## Inputs

The shipped workflows set these; most people never change them.

### Optional

**`patched_ledger_json`** -- a connection from another node (`STRING`).

The OTR_ShotLock-planned (and image-genned) ledger JSON. run_real_episode renders one REAL per-beat clip per shot; the emitted clip manifest feeds OTR_SilentComposite.

**`master_audio_path`** -- a connection from another node (`STRING`).

Path to the FROZEN master mix (MP4 or WAV). Beats whose ledger line has no per-line *_wav_path get audio_ref filled by slicing [start_s, start_s+dur_s] from this file (read-only ffmpeg; master is NEVER mutated). Wire from OTR_EpisodeAssembler.output_path (the frozen master WAV). Leave unset to degrade LOUD on missing per-line wavs.

**`image_done`** -- a connection from another node (`STRING`).

Image-done gate (mirrors audio_done; ST-0.2 still-spine). Wire from OTR_ImageGenDispatcher.image_done so the video render cannot start before every episode still exists on disk (W4). Opaque STRING; the token is never parsed.

## Outputs

- `render_report_json` (`STRING`)
- `clip_manifest_json` (`STRING`)

This is an output node: the workflow runs to reach it.

## Read more

- [VIDEO_MODELS.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/VIDEO_MODELS.md)
- [MACHINES.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/MACHINES.md)

Class `OTR_VideoRenderBatch`, category `OldTimeRadio/v2/video`.
