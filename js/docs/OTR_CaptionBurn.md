<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# CaptionBurn (SDH open captions on the silent video; default-OFF)

Burns open captions and the title card onto the silent video, when it is switched on.

## What it does

Burns open SDH subtitles and the episode title card directly onto the silent composited video stream. Operates strictly on video frames so audio remains lossless and synchronized downstream. Toggle caption burning on or off, select caption styling between standard SDH and retro CRT, or adjust vertical margins and frame rate.

## Inputs

On the shipped workflows these are the widgets a person changes; the ones on the app form are named as the form names them.

**`video_path`** -- a connection from another node (`STRING`).

Silent video from OTR_SilentComposite. Captions burn here; audio is added later by MasterAudioMux.

### Optional

**`burn_captions`** -- on or off; starts as on.

Burn open captions into the video. On by default, matching the shipped graph. Needs an ffmpeg with the ass filter and libx264 -- without them the episode still finishes, just without captions.

**`caption_style`** -- a choice from the dropdown; starts as `sdh_standard`.

Visual style of the burned SDH captions. The OTR_CAPTION_STYLE env var may override the STYLE only -- no env var can turn burning on; only the burn_captions widget (or a supplied title-card plan, which forces the hero-title burn alone) does that.

**`fps`** -- a whole number from 1 to 120; starts as `25`.

Frame rate used for caption timing math. Must match the incoming silent video's real rate or captions drift against speech.

**`ledger_path`** -- text; starts as empty.

Optional explicit timed-ledger path. Empty -> resolved from the video stem (otr_audio_dir / in-flight ledger).

**`output_path`** -- text; starts as empty.

Captioned mp4 path. Empty (the shipped default) -> <stem>_captioned.mp4 beside the input video. With burn OFF and no title plan the input passes through untouched.

**`gate_in`** -- a connection from another node (`STRING`).

Optional ordering signal (opaque STRING).

**`title_card_plan_json`** -- a connection from another node (`STRING`).

Hero title-card plan from OTR_SignalLostVideo. When present the title burns REGARDLESS of burn_captions and any failure REFUSES instead of passing through -- the title is the only copy left once the procgen draw is suppressed.

## Outputs

- `video_path` (`STRING`)
- `report` (`STRING`)

## Read more

- [RUN.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/RUN.md)
- [MULTILINGUAL.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/MULTILINGUAL.md)

Class `OTR_CaptionBurn`, category `OldTimeRadio/v2/video`.
