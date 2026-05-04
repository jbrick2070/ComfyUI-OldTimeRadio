# Question -- 2026-05-03

# Round-robin question — HuMo + LTX visual content disappears between composite and final OBS output

## Pipeline (current shipped state, commit `fbbf749`)

ComfyUI workflow `otr_scifi_16gb_full.json`. Five stages relevant here:

1. **`OTR_BatchHumoRender`** writes per-line character clips to `videos/l002.mp4` ... `l006.mp4` at native **480x832 portrait, H.264, 25 fps**, ~7s each.
2. **`OTR_BatchLTXRender`** writes per-line non-character clips (announcer + LTX broadcast units) to `videos/l001.mp4` and `videos/l007.mp4` at native **832x480 landscape, H.264, 25 fps**, ~7s each.
3. **`OTR_VideoComposite._render_master_mix_per_clip_mux_mode`** loops the 7 per-line clips through `_layered_per_clip_silent` (simple-pillarbox: scale-fit-then-pad-with-black to 1472x832), concat-demuxes them into `silent_combined.mp4`, then muxes that against the procgen master mix audio with `-c:a copy` -> writes `composited/<ep>.mp4`.
4. **`OTR_RTXUpscale`** consumes the 1472x832 composite, runs RTX VSR upscale to 1920x1080 via chunked ffmpeg piping (raw RGB24 in/out, returns a STRING path), saves to `obs/<ep>.mp4`. Audio mux is `-c:a copy`.
5. **`OTR_PostUpscaleProcgenBlend`** takes the RTXUpscale output + the 1920x1080 procgen mp4 from `OTR_SignalLostVideo`, builds an ffmpeg `-filter_complex blend=all_mode=lighten:all_opacity=0.5` chain, audio `-c:a copy` from source, writes `obs/<ep>_procgen_blended.mp4`. Recently DROPPED `-shortest` flag per a previous round-robin's C7 hardening recommendation.

## What the user observed (live soak run 2026-05-03 17:39 -> 18:58)

The OBS folder contains TWO outputs (note: only ONE was expected -- the post-blend one):

* `obs/<ep>.mp4` -- 1.63 MB, "audio + all black video" per the user
* `obs/<ep>_procgen_blended.mp4` -- 14.48 MB, "procgen scanlines visible, NO HuMo/LTX content visible underneath"

## Per-stage ffprobe (the smoking gun)

| File | Dims | Duration | Bitrate | Frames |
|---|---|---|---|---|
| `videos/l001.mp4` (LTX) | 832x480 | 7.72s | 708 kbps | 193 |
| `videos/l002.mp4` (HuMo) | 480x832 | 6.88s | 1824 kbps | 172 |
| `videos/l003.mp4` (HuMo) | 480x832 | 6.88s | 1474 kbps | 172 |
| `videos/l004.mp4` (HuMo) | 480x832 | 6.88s | 1057 kbps | 172 |
| `videos/l005.mp4` (HuMo) | 480x832 | 6.88s | 1878 kbps | 172 |
| `videos/l006.mp4` (HuMo) | 480x832 | 6.88s | 1631 kbps | 172 |
| `videos/l007.mp4` (LTX) | 832x480 | 7.72s | 686 kbps | 193 |
| `composited/_per_clip_mux_segments/silent_combined.mp4` | 1472x832 | 50.36s | 1544 kbps | 1259 |
| `composited/<ep>.mp4` (composite output, after audio mux) | 1472x832 | 50.36s | 1544 kbps | 1259 |
| **`obs/<ep>.mp4` (RTXUpscale output)** | **1920x1080** | **50.36s** | **96 kbps** | **1259** |
| `obs/<ep>_procgen_blended.mp4` (post-blend) | 1920x1080 | **113.92s** | 988 kbps | 2784 |

## Diagnosis hypothesis (what we believe is happening)

**Bug 1 -- RTXUpscale destroyed visual content.** Bitrate collapse from 1544 kbps to 96 kbps on identical dimensions and frame counts means frames went to near-solid-black (H.264 compresses solid color to ~nothing). Source was good, output is empty. The chunked ffmpeg pipeline preserved frame count and timing but the AI VSR model emitted black frames OR the encoder was misconfigured.

**Bug 2 -- PostUpscaleProcgenBlend duration overrun.** OBS post-blend is 113.92s vs source 50.36s. The procgen mp4 is ~94s + pad. Without `-shortest` (we dropped it earlier per a Gemini round-robin recommendation), the blend filter outputs the LONGER input duration. Audio is correctly 50s via `-c:a copy`. Result: 50s of audio over 113s of video. First 50s is procgen-over-black-source (procgen wins via lighten); next 63s is procgen + silence.

## What we want from this round-robin

1. **Localize Bug 1 (RTXUpscale black output).** Source `nodes/rtx_upscale.py` chunked ffmpeg pipeline. Raw RGB24 in via decode_proc, raw RGB24 out via encode_proc. The model is RTX VSR Quality=ULTRA. Common failure modes for "AI upscaler emits black": (a) input dim not divisible by some factor expected by the AI model, (b) input frame range expected to be 0-1 float but receiving 0-255 uint8 (or vice versa), (c) encoder libx264 receiving ALL black frames despite source being valid (color space mismatch?), (d) the AI VSR step itself OOM'ing silently and falling back to write zeros, (e) something in CUDA stream sync between decode_proc and encode_proc dropping frames. Which is most likely? What ffprobe / ffmpeg debug flags would prove which one?

2. **Resolve Bug 2 (duration overrun) without re-introducing the C7 risk that drove us to drop `-shortest`.** The previous round-robin (Gemini) said `-shortest` would silently truncate audio if procgen ended first, violating C7 byte-identity. But we now have the OPPOSITE problem: procgen is the LONGER input, so we need to clamp video to source duration. Options: (a) explicit `[v]trim=duration={src_dur}`, (b) `-t {src_dur}` flag, (c) framesync `eof_action=endall` on the blend filter. Which is C7-safe AND solves the overrun? How do we determine `src_dur` reliably (probe the source mp4 with ffprobe before building the cmd)?

3. **Confirm or reject our diagnosis hypothesis.** Are the bitrate-collapse and frame-count-preserved signals enough to localize Bug 1 to RTXUpscale? Or could the composite output be defective in a way that ffprobe doesn't catch (e.g. all valid color but at black-equivalent luma)?

## Constraints
* Local-only, no cloud
* C7 audio byte-identity MUST hold end-to-end
* RTX 5080 16 GB VRAM (Blackwell sm_120)
* Windows, Python 3.12, torch 2.10.0, CUDA 13.0
* We have full disk access to all the 11 files above for any diagnostic probing you want to suggest
* RTXUpscale source is `nodes/rtx_upscale.py` -- chunked ffmpeg subprocess.Popen pipeline, RETURN_TYPES is STRING (path, not IMAGE tensor)

Be concrete: name ffmpeg flags, name suspect log lines to grep for, name the order in which to probe each hypothesis. If you don't know a number, say so -- better to flag uncertainty than invent one.
