<!-- Written by scripts/build_node_docs.py from config/node_docs.json and the node's own definition. Edit those, then run the script. -->
# SilentComposite (render -> one always-silent video)

Cuts the rendered clips and the procedural base into one silent video at a fixed frame rate, and runs the optional upscaler on each clip.

## What it does

Assembles rendered video clips and the procedural CRT base into one standardized, silent, CFR video stream tagged for BT.709 color. Strips existing audio to preserve the audio-is-king pipeline contract and performs optional per-clip AI upscaling. Adjust canvas dimensions, fallback frame rate, or the upscale model engine and device.

## Inputs

On the shipped workflows these are the widgets a person changes; the ones on the app form are named as the form names them.

**`base_video_path`** -- a connection from another node (`STRING`).

Base video to normalize (M1: the OTR_SignalLostVideo radio-floor mp4). Any audio is STRIPPED here (V-1).

### Optional

**`canvas_w`** -- a whole number from 16 to 7680; starts as `1920`.

Silent-composite canvas width (px). Profile/platform-owned: variants pin it per hardware tier; both dimensions must be even (yuv420p) or the encode refuses by name.

**`canvas_h`** -- a whole number from 16 to 4320; starts as `1080`.

Silent-composite canvas height (px). Profile/platform-owned, even values only -- see canvas_w.

**`fps`** -- a whole number from 1 to 120; starts as `25`.

Fallback frame rate for the silent timeline. When the clip manifest carries its own fps (the audio-derived budget), the manifest wins and this value is ignored.

**`output_path`** -- text; starts as empty.

Silent composite path. Empty -> <output>/otr/episodes/<stem>_silent.mp4.

**`gate_in`** -- a connection from another node (`STRING`).

Optional ordering signal (opaque STRING).

**`clip_manifest_json`** -- a connection from another node (`STRING`).

OTR_VideoRenderBatch clip manifest. When it carries per-beat clips the composite ASSEMBLES a frame-accurate CFR timeline (each beat conformed to its audio-derived frame count; gaps filled from base_video_path). Empty -> the single-base floor path.

**Upscaler** (`upscale_engine`) -- a choice from the dropdown.

Post-render upscale/enhance stage. Runs INSIDE the composite per-clip (sharpen=True branch only); floor / directory / black paths are unchanged. off = today's lanczos+unsharp, byte-identical. spandrel_esrgan = Real-ESRGAN x2 (any device, slow). rtx_vsr = NVIDIA RTX Video Super Resolution (RTX card; you install NVIDIA's nvidia-vfx package into ComfyUI's Python first, ~540 MB; ~100x faster than ESRGAN; falls back to off with a warning where it cannot run).

**`upscale_device`** -- text; starts as `cpu`.

cpu | cuda | cuda:N. Composite itself runs on CPU (ffmpeg); this widget selects the device the model loads on. Rejected invalid tokens fail loud. rtx_vsr runs on the NVIDIA GPU whatever this says (cpu means cuda:0 for it).

## Outputs

- `silent_video_path` (`STRING`)
- `report` (`STRING`)

## Read more

- [UPSCALERS.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/UPSCALERS.md)
- [VIDEO_MODELS.md](https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/main/apple/VIDEO_MODELS.md)

Class `OTR_SilentComposite`, category `OldTimeRadio/v2/video`.
