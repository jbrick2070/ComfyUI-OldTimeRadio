# ITEM F receipt: `otr_g4_wan_ti2v` is PROVEN

Carried in `GO_FORWARD_PLAN.md` as *"only `otr_w45_wan_ti2v` is proven;
`otr_g4_wan_ti2v` and `otr_upscale_ship` remain unexercised."* One of the two is
now exercised, end to end, through the real canonical workflow.

## The published artifact

    C:\Users\jeffr\Documents\ComfyUI\output\otr\obs\
      signal_lost_the_weeping_valve_20260823_001152_silent_procgen_blended_captioned_with_credits_final.mp4

`otr/obs` went 114 -> 115 files. Timestamp 2026-08-23 02:16. ffprobe:
**h264, 1920x1080, 25 fps, 4165 frames, 166.6 s (2:47), 104,031,627 bytes**,
plus an AAC stream at 6530 frames. Real episode, not a stub.

## The deltas were actually exercised, each verified separately

`otr_g4_wan_ti2v` differs from the already-proven `otr_w45_wan_ti2v` in exactly
three things. All three are confirmed, and none of them is taken from a log line
that merely *claims* success:

| delta | proof |
|---|---|
| writer = `unsloth/gemma-4-12b-it-GGUF` (Q4_K_M, 8192 ctx) | `meta.creative_model` AND `meta.technical_model` in the episode ledger; the server also logged `[Selector] slot=technical reuse cache for unsloth/gemma-4-12b-it-GGUF` |
| `video.max_render_frames = 81` | present in the applied API graph on `OTR_VideoDirector` |
| `wan_ti2v` on all three visual roles | **17 clip files, each naming the engine in its own filename** |

The clip filenames are the strongest evidence here, because the engine id is
baked into the name by the writer of the file:

    shot_b001_announcer_visual_wan_ti2v.mp4
    shot_b002..b015_character_video_wan_ti2v.mp4
    shot_b006 / b011_music_visual_wan_ti2v.mp4
    shot_b016_announcer_visual_wan_ti2v.mp4
    shot_music_closing_001_music_visual_wan_ti2v.mp4

One sampled clip: 832x480 (the profile's canvas), 198 frames, 7.92 s. Measured
render VRAM peak `12015 MB / post 6805 MB`, under the 14.5 GiB gate.

**`_silent_procgen_blended` in the final filename is NOT a substitution.** It is
`OTR_PostUpscaleProcgenBlend` (canonical node 93), a pipeline stage that blends
over the rendered video. A reader who takes that suffix as "it fell back to
procedural" will reach the wrong conclusion -- the clips above are the truth.

## Two defects this leg exposed, both already fixed and pushed

1. **The preflight gate could never pass nine profiles** (`b11a4269`). It exact-
   matched `preflight.required_models` against `/object_info`, which lists
   FILENAMES only, while nine profiles declare logical ids. `otr_upscale_ship`
   was refused with "the running server cannot see: real-esrgan-x2plus" while
   `RealESRGAN_x2plus.pth` was visible the whole time. **This is almost
   certainly why that profile was carried as "unexercised".**
2. **The runner called this very render dead while it was at 98% GPU**
   (`cebe7c75`). `--timeout` defaults to 5400 s; this episode needed longer, so
   the runner printed `RESULT TIMEOUT` and exited 1 at t=5396 s while the clip
   count climbed 21 -> 33 -> 37. The render finished fine 40 minutes later and
   published itself. Every wan episode run through the canonical runner has been
   reporting a false failure. The runner now asks the queue before letting a
   reader conclude anything.

## Honest notes

* The leg was run with the DEFAULT `--timeout`, which is why finding 2 surfaced.
  A long lane should use `--timeout 0` (the documented operator mode).
* Timing: submitted 00:00:48, published 02:16 -- roughly **2 h 15 m wall clock**
  for a 2:47 episode on this box. That is the real cost of the wan lane here and
  is worth knowing before anyone schedules a batch of them.
* `otr_upscale_ship`, F's other half, is unblocked by fix 1 above but is a
  SEPARATE leg and is not covered by this receipt.
