<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# MasterAudioMux (terminal mux-LAST, -c:a copy)

The last node: it lays the master audio under the finished video and publishes the episode to your output folder. The finished .mp4 is its result.

## What it does

The terminal publication node that muxes frozen master audio onto the silent video composite using lossless stream copying. Verifies audio-video duration agreement, asserts bit-for-bit audio integrity, and copies the completed episode to the OBS delivery directory when eligible. Connect the silent video and master audio paths, or adjust declared frame rate and output file paths.

## Inputs

The shipped workflows set these; most people never change them.

**`silent_video_path`** -- text; starts as empty.

Always-silent composite (OTR_SilentComposite). Audio is added HERE only.

**`master_audio_path`** -- text; starts as empty.

Frozen master mix from OTR_EpisodeAssembler.output_path (the master WAV). Copied with -c:a copy.

### Optional

**`audio_done`** -- text; starts as empty.

Audio-done gate (EpisodeAssembler out3). Orders the mux AFTER the audio freezes. Opaque.

**`declared_credits_tail_s`** -- a number; starts as `0.0`.

OTR_CreditsRoll's declared silent-credits tail duration. Makes the tail guard credits-aware (v <= a + declared + tol); 0 -> the OTR_MAX_CREDITS_TAIL_S env ceiling.

**`clip_manifest_json`** -- text; starts as empty.

Retired connector kept for topology compatibility (rip-sfx 2026-08-06). The manifest is hashed by IS_CHANGED but has no effect on the output.

**`fps`** -- a whole number from 1 to 120; starts as `25`.

Frame rate declared to the mux for timing math. Must match the silent video's real rate (the composite's manifest fps); the mux never resamples frames.

**`output_path`** -- text; starts as empty.

Final mp4 path. Empty -> <output>/otr/episodes/<stem>_final.mp4.

**`video_policy_json`** -- text; starts as empty.

Video policy JSON from OTR_VideoDirector. Read for ONE question: is any role on the LTX 2.5 foley lane? If so this node mixes the foley bed under the provisional master and performs the single delivery loudness pass.

**`foley_receipts_json`** -- text; starts as empty.

Clip manifest carrying the per-beat foley receipts (foley_path / foley_sha256 / start_s). The bed the mix is built from. Ignored unless video_policy_json names a foley route.

**`script_json`** -- text; starts as empty.

Writer ledger JSON (OTR_LedgerScriptWriter script_json). Read for ONE question: must this episode reach otr/obs to count as delivered? A story the listener wrote must, and a miss fails the run; every other bank is published too, whatever its rights receipt says.

## Outputs

- `final_video_path` (`STRING`)
- `report` (`STRING`)

This is an output node: the workflow runs to reach it.

## Read more

- [RUN.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/RUN.md)

Class `OTR_MasterAudioMux`, category `OldTimeRadio/v2/video`.
