# OTR_HuMoConcat / OTR_VideoEpisodeAssembler — design sketch

**Date:** 2026-04-24 (sketched while download finishes; needs review before
implementation; not yet built)

**Purpose:** stitch N short HuMo clips (each 3.88s — 60s) into a single
full-episode MP4 with the master audio track muxed in. This is the step
that turns "many clips" into "one episode."

## Why this exists

HuMo can't produce a 7-min video in one call (Whisper context limits,
boundary error compounding, wall-clock risk). The practical pipeline:

```
N short HuMo clips  +  master_audio.wav  +  ledger timing data
                          ↓ ffmpeg
                  one final_episode.mp4
```

The ffmpeg step is straightforward but has design choices. This doc
locks them in before we build the node.

## Two flavors of "stitch"

| Mode | What | When to use |
|---|---|---|
| **Concat** | Sequential clips, no base track. Each clip plays full-frame, then the next begins. Audio is muxed from the master WAV. | Each clip already covers its timeline window contiguously (per-shot or per-scene plan). Cleanest visually. |
| **Overlay** | Clips composite OVER a base track (the existing `SignalLostVideo` POC: waveform + treatment HUD). Each clip appears at its `start_s`/`dur_s` window; base shows when no clip plays. | Per-line plan with gaps. Or hybrid (HuMo close-ups during dialogue, base track during music/SFX-only beats). Option C aesthetic. |

We probably want BOTH modes selectable via a node widget.

## Inputs (proposed node API)

```
OTR_HuMoConcat (or OTR_VideoEpisodeAssembler)

INPUTS:
    clips_json          : STRING -- JSON list of {clip_path, start_s, dur_s, [shot_id]}
                                   in timeline order. Comes from the ledger
                                   or hand-built.
    audio_path          : STRING -- path to master episode WAV (the full
                                   AudioEnhance + EpisodeAssembler output)
    base_video_path     : STRING -- (optional) path to base track MP4
                                   (signal_lost_*.mp4). Required for
                                   overlay mode, ignored for concat.
    mode                : DROPDOWN -- "concat" | "overlay" | "auto"
    output_dir          : STRING -- defaults to output/old_time_radio/
    output_filename     : STRING -- defaults to derived from episode_id
    overlay_position    : DROPDOWN -- "center" | "top-right" | "bottom-right"
                                     | "fullscreen" -- only used in overlay mode
    overlay_size_pct    : INT -- 100 = full-screen replacement,
                                 30 = picture-in-picture corner
                                 (only used in overlay mode)
    blend_mode          : DROPDOWN -- "none" | "soft_mask" | "screen"
                                     | "monitor_frame" -- pixel-blending
                                     style for overlay mode

OUTPUTS:
    final_video_path    : STRING -- path to the assembled MP4
    debug_summary       : STRING -- ffmpeg argv + timing report
```

## Concat mode — the simple case

Build an ffmpeg `concat` filter graph from the clips, then mux audio:

```bash
ffmpeg -y \
  -i clip001.mp4 \
  -i clip002.mp4 \
  -i clip003.mp4 \
  ...
  -i master_audio.wav \
  -filter_complex "[0:v][1:v][2:v]...[Nv:v]concat=n=N:v=1:a=0[v]" \
  -map "[v]" -map "<N+1>:a" \
  -c:v libx264 -c:a aac -shortest \
  output/old_time_radio/signal_lost_<title>_humo.mp4
```

Concat mode assumes:
- Clips are time-contiguous (no gaps)
- Total clip duration matches master audio duration
- Frame rates match (all 25 fps)

If clip durations sum to less than audio length: pad with the last frame
held (`tpad=stop_mode=clone`). If more: trim to audio length (`-shortest`).

## Overlay mode — Option C composite

Each clip overlays the base track at its `start_s` for `dur_s`. Base
shows when no clip is playing.

```bash
ffmpeg -y \
  -i base_track.mp4 \
  -i clip001.mp4 -i clip002.mp4 ... -i clipN.mp4 \
  -i master_audio.wav \
  -filter_complex "
    [0:v][1:v]overlay=x=W-w-20:y=20:enable='between(t,start1,start1+dur1)'[v1];
    [v1][2:v]overlay=x=W-w-20:y=20:enable='between(t,start2,start2+dur2)'[v2];
    ...
    [vN-1][Nv:v]overlay=x=W-w-20:y=20:enable='between(t,startN,startN+durN)'[vF]
  " \
  -map "[vF]" -map "<N+2>:a" \
  -c:v libx264 -c:a aac \
  output/old_time_radio/signal_lost_<title>_humo_overlay.mp4
```

The `enable='between(t,A,B)'` clause makes each clip only appear during
its timeline window. Outside that window, the base track shows through.

Overlay positions for the picture-in-picture portrait:
- `center`: `x=(W-w)/2:y=(H-h)/2`
- `top-right`: `x=W-w-20:y=20`
- `bottom-right`: `x=W-w-20:y=H-h-20`
- `fullscreen`: `x=0:y=0` (with scale=W:H upstream — replaces base entirely)

Blend modes:
- `none`: hard rectangle paste
- `soft_mask`: feathered alpha mask via `geq` filter (8px feather edge)
- `screen`: blend mode multiply or screen for ghostly overlay
- `monitor_frame`: composite portrait inside a CRT/video-phone bezel PNG
  (Option C's diegetic monitor look — bezel asset has to exist as a
  static image; we'd hand-make or FLUX-render it once)

## Where the timing data comes from

Two sources, dispatchable:

1. **Ledger-driven (production):** read `signal_lost_*_ledger.json`,
   pull `clips[]` (the future L2 table), iterate in `start_s` order. The
   node consumes the same ledger our viewer already reads.

2. **JSON-driven (manual):** caller passes a `clips_json` string with
   the explicit list. Useful for testing and one-off compositions.

Internally same code path — just different input source.

## Ledger schema additions (already in L1 schema)

The L1 ledger already has the fields needed:

```json
"clips": [
  {"clip_id": "cl001", "shot_id": "sh01", "line_id": "l001",
   "humo_clip_path": ".../humo/cl001.mp4",
   "audio_slice_start_s": 12.4,
   "audio_slice_dur_s": 3.88}
]
```

The concat node reads `humo_clip_path`, `audio_slice_start_s`,
`audio_slice_dur_s`. Plus `final_audio_path` from the ledger for the
master audio mux source.

## Output naming + ledger update

After successful concat:
- Output path: `output/old_time_radio/signal_lost_<title>_humo.mp4`
  (sibling to the existing `signal_lost_<title>_<ts>.mp4` POC base)
- Ledger update: `data["final_video_path"]` set to the new MP4
- Treatment file: copy + rename the existing `_treatment.txt` so it
  matches

## Failure modes + handling

| Issue | Detection | Mitigation |
|---|---|---|
| Clip fps mismatch | ffprobe each clip | reject; force 25 fps everywhere |
| Audio missing | path doesn't exist | fail fast, do not produce silent video |
| Total clips > audio length | sum durations | trim last clip; warn |
| Total clips < audio length | sum durations | pad with frozen last frame OR transparent if overlay |
| Clip codec mismatch | ffprobe each | re-encode all to h264 yuv420p first |
| Overlay PiP doesn't fit screen | pre-flight check | scale clip to fit |

## What I'd build first

L0 of this node: **concat mode only, ledger-driven, no overlay/blend**.
~80 lines of Python wrapping ffmpeg. Hard-coded `output/old_time_radio/`
output. No fancy options.

Then iterate:
- L1: add overlay mode (~30 more lines)
- L2: add blend modes (~20 lines per mode)
- L3: add diegetic monitor frame (needs the bezel PNG asset first)

## Open question — bedtime decision

Per Jeffrey's "no talking heads" preference, the actual production path
might be **Wan 2.2 + SVI Pro for atmospheric clips, no HuMo at all in
production**. In that case the same concat node handles Wan+SVI clips
identically — the file format is MP4, the ledger schema is the same.
This node is **video-model-agnostic** as long as the clip MP4s exist
and the timing data is in the ledger. So building it serves either path.

## Implementation order recommendation

1. **First**: complete the HuMo POC smoke test (one clip, prove model fits)
2. **Second**: if HuMo POC succeeds, build OTR_HuMoConcat L0 (concat
   mode only, ~80 lines) and assemble a 3-clip test episode
3. **Third**: if results are good, expand to per-scene production
4. **Parallel**: try Wan 2.2 + SVI Pro on the same clip-and-concat
   architecture; compare quality side-by-side
