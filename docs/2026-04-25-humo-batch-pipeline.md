# HuMo full-episode batch pipeline (Pattern B)

**Date:** 2026-04-25
**Status:** scaffolded, not yet run end-to-end on a full episode

## What this is

Two scripts that together turn a finished OTR audio episode + ledger into
an MP4 with HuMo character animation covering every shot:

```
scripts/render_humo_batch.py     — render N HuMo clips, one per line
scripts/concat_humo_episode.py   — stitch them back into one MP4
```

No new ComfyUI custom nodes. Both are pure-stdlib Python plus `ffmpeg`
on PATH, talking to a running ComfyUI server at `localhost:8000` via the
HTTP `/prompt` API.

## Why Pattern B (sequential single-prompt submissions)

We measured today: the HuMo cold-load (UNETLoader humo_17B + LoraLoader
lightx2v + CLIPLoader umt5 + VAELoader wan_2.1 + AudioEncoderLoader
whisper) costs ~50 seconds. Per-clip sampling at length=97 is ~4:14
(KSampler 6 steps × 42s/step). VAE decode + save adds ~12s.

If we submit 100 clips as 100 separate prompts back-to-back, ComfyUI's
model cache keeps the loaders warm between prompts. The 50s cold-load
amortizes to ~0.5s per clip. Each subsequent prompt pays only sampling
+ decode + per-clip image/audio swap.

Wall-clock estimate at length=97 (the only stable shape on RTX 5080
16 GB — see `2026-04-24-humo-poc-recipe.md`):

```
Cold load (one-time):              ~50s
Per-clip:
  Whisper re-encode (new audio):   ~3s
  KSampler 6 steps:                4:14
  VAE decode + create + save:      ~12s
  Subtotal per clip:               ~4:30

100 clips × 4:30 = 7 hours 30 minutes per 7-minute episode.
```

That's an overnight render. Slow but feasible for full per-shot
coverage. For sparse cutaways (5-10 clips per episode at scene
boundaries), the same pipeline runs in 30-60 minutes.

## End-to-end usage

Step 1 — render the audio episode the existing way (writes ledger):

```
# In ComfyUI: open workflows/otr_scifi_16gb_full.json, queue.
# Wait for the master WAV + per-line ledger to land.
```

Step 2 — render HuMo clips for every line in the ledger:

```powershell
cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio

C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe `
  scripts\render_humo_batch.py `
  --ledger output\old_time_radio\signal_lost_<title>_ledger.json `
  --out-dir output\old_time_radio\<title>_humo `
  --scope all
```

For the older astrotech ledger (no per-line timing), use `--auto-slice`
to evenly distribute clips across the master MP4's audio:

```powershell
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe `
  scripts\render_humo_batch.py `
  --ledger output\old_time_radio\signal_lost_astrotech_research_facility_humming_20260424_165327_ledger.json `
  --out-dir output\old_time_radio\astrotech_humo `
  --auto-slice 3.88 `
  --max-clips 5
```

For a small dramatic-beats subset (sparse cutaways):

```powershell
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe `
  scripts\render_humo_batch.py `
  --ledger ... `
  --out-dir ... `
  --scope first-per-scene
```

Step 3 — stitch the clips into a final episode:

```powershell
# Mode A: concat (clips fill the timeline, master audio replaces clip audio)
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe `
  scripts\concat_humo_episode.py `
  --mode concat `
  --clips-dir output\old_time_radio\<title>_humo `
  --master-wav output\old_time_radio\<title>.wav `
  --out output\old_time_radio\<title>_humo_full.mp4

# Mode B: overlay (clips composite onto a base video at ledger timing)
C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe `
  scripts\concat_humo_episode.py `
  --mode overlay `
  --clips-dir output\old_time_radio\<title>_humo `
  --ledger output\old_time_radio\<title>_ledger.json `
  --base-video output\old_time_radio\<title>.mp4 `
  --out output\old_time_radio\<title>_humo_overlay.mp4
```

## What the orchestrator does, line by line

For each line in the ledger (filtered by `--scope`):

1. **Pick a portrait.** First tries `cast[].portrait_path` from the
   ledger; falls back to indexing into `output/otr_humo_pass1_portrait_*.png`
   by cast position; final fallback is the first available portrait.

2. **Slice audio.** `ffmpeg -ss <line.start_s> -t 3.88 -ac 1 -ar 16000`
   from the master WAV (or from the master MP4's audio track if no
   separate WAV exists).

3. **Stage the inputs.** Both portrait and audio slice are copied/written
   into ComfyUI's `input/` directory with per-line filenames so we don't
   collide with other workflows.

4. **Build a HuMo prompt.** The full HuMo subgraph from
   `workflows/otr_videoplan_TEST_humo.json` (nodes 7-22), expressed as
   ComfyUI's API format with the per-line filenames + a deterministic
   seed.

5. **POST to `/prompt`.** Submit, get a `prompt_id`.

6. **Poll `/history/<id>`** every 2 seconds until `status.completed`.

7. **Move the resulting MP4** from ComfyUI's
   `output/old_time_radio/humo_episode/<line_id>_NNNNN_.mp4` to the
   `--out-dir` named simply `<line_id>.mp4`.

8. **Repeat.** ComfyUI's model cache stays warm because every loader
   node has identical widget values across calls.

## Limitations / known gaps

- **Per-line timing missing in the astrotech ledger.** The L1 ledger
  schema includes `lines[].start_s` / `dur_s`, but the FULL pipeline
  doesn't yet populate them. Until SceneSequencer is wired through to
  the ledger (Task #20), use `--auto-slice 3.88` to distribute clips
  evenly.

- **Variable-length lines truncated to 3.88s.** HuMo native is 97
  frames @ 25 fps. A line longer than 3.88s gets cut at 3.88s; a
  shorter line gets silence padding at the end. Production-grade per-
  line lip-sync would chunk long lines into multiple HuMo windows and
  stitch them — deferred until L1 (sparse cutaways) ships.

- **No quality scoring.** The orchestrator picks portraits in cast
  order. If the wrong character ends up rendered, fix is to populate
  `cast[].portrait_path` in the ledger.

- **Speaker labels can be empty in older ledgers.** Falls back to
  the first available HuMo PASS1 portrait.

- **Concat mode replaces audio entirely with the master WAV.** This
  works for full-coverage at length=97 (clips × 3.88s ≈ episode
  duration), but produces noticeable lip-sync drift if clip count ×
  3.88s diverges from master duration.

## Decision log — what we measured to get here

| Setup | Step time | Total | Notes |
|---|---|---|---|
| fp8 + normalvram | 42.6s | 6:26 | Yesterday, baseline |
| fp8 + normalvram (native, no FLUX) | 42.5s | 4:31 | This morning |
| Q5_K_M GGUF + normalvram | 42.1s | 6:38 | GGUF didn't speed up |
| Q5_K_M + smart-memory-off | 42.1s | 6:16 | -22s, marginal |
| Q5_K_M + lowvram | 77.4s | (cancelled) | lowvram = slower per-step |
| Q5_K_M + length=65 + normalvram | (HUNG) | — | OS swapped to disk during torch.compile |
| Q5_K_M + length=65 + lowvram | (HUNG) | — | Same RAM-spike crash |

**Verdict:** length=97 + normalvram + Q5_K_M GGUF is the only stable
fast configuration. Per-step is a hard 42s on this hardware. Pattern B
amortizes the load cost across N clips so the per-clip wall clock
asymptotes to ~4:30. That's the floor.

## When this batch pipeline becomes obsolete

- **FA3 / Blackwell flash-attention 3 lands.** Could drop step time
  to ~12s. Per-clip would drop to ~1:30. 100 clips in 2.5 hours.
- **Per-line ledger timing populated.** Then we drop `--auto-slice`
  and respect actual line boundaries. Less drift, better sync.
- **Director-flagged "headline" lines.** Auto-pick which lines deserve
  HuMo treatment vs which can use FLUX-still-with-pan-zoom.

Until then: Pattern B, length=97, ~6 hours of overnight render per
full-coverage episode.
