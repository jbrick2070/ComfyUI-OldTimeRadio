# LTX-AV quality bakeoff -- panel QA prompt (paste to the candidate models)

Paste everything between the lines below into ChatGPT / Gemini / DeepSeek (etc.). It is
self-contained. The clips named in the table are in
`otr\episodes\_bakeoff_ltxq\<leg>.mp4` for side-by-side eyeballing (the panel reasons from
the metrics; Jeffrey eyeballs the clips).

---------------------------------------------------------------------------------------

You are on a technical review panel. I ran an isolated A/B/C bakeoff to fix three defects
on the audio-driven "ltx_audio_in" clips in my local AI radio-drama pipeline (ComfyUI, one
RTX 5080 laptop, 16 GB VRAM, hard ceiling 14.5 GB for the single resident heavy engine):

  1. SOFTNESS -- the engine renders a tiny 512x288 native clip, then a composite step
     upscales it ~8.3x in area to a 1472x832 canvas (bilinear) -> mush.
  2. TEMPORAL "FLASH" -- a luminance discontinuity every ~56 frames, root-caused to the
     VAE temporal-tiling seams (temporal_size 64 / overlap 8 = a 56-frame stride; the jumps
     land exactly at the chunk boundaries). NOT hard scene cuts (0 scene cuts measured).
  3. INIT-HOLD STUTTER -- a brief hold at the start/end tied to the i2v init-image
     strength, not ping-pong.

Setup (held CONSTANT across every leg so it is apples-to-apples): one fixed still + one
fixed driving-line audio + fixed seed 0; recipe = distilled_native (distilled-1.1 Q3_K_M
GGUF unet, DEV video/audio VAE, Gemma-3 encoder; NO LoRA, NO ModelSamplingLTXV, NO
LTXVScheduler; fixed 8-step distilled sigmas; euler_cfg_pp; cfg 1.0); output is SILENT,
encoded with the exact production encoder (libx264 crf18 bt709). I vary ONE lever per leg.

Metrics per leg: s/it (lower = faster); peak VRAM in MB (HARD ceiling 14500); scene-cuts
(want 0); freezes = count of frozen segments >= 0.12 s (want 0); seam p99 = the 99th-pct
luminance jump at the tiling-seam frames on a 0-255 scale (lower = less flash; "no-seam" =
whole-clip decode, no tiling boundary at all).

| leg | canvas | decode temporal | i2v | sigmas | s/it | peakVRAM | scene | freezes | seam p99 | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| L0 (baseline) | 512x288 | 64 / 8 | 0.75 | native | 6.04 | 14337 | 0 | 0 | 0.2353 (jump=2.07x local median = the flash) | current production |
| L1a | 512x288 | 64 / 16 | 0.75 | native | 6.19 | 14076 | 0 | 0 | 0.2085 | more overlap |
| L1b | 512x288 | 64 / 32 | 0.75 | native | 6.67 | 14309 | 0 | 0 | 0.1235 | |
| L1c | 512x288 | 128 / 16 | 0.75 | native | 5.37 | 14379 | 0 | 0 | 0.1180 | |
| L1d | 512x288 | 128 / 32 | 0.75 | native | 5.56 | 14272 | 0 | 0 | 0.0321 (jump=0.57x median) | best TILED |
| L1e | 512x288 | 4096 / 8 (whole-clip) | 0.75 | native | 5.37 | 14338 | 0 | 0 | no-seam | seam ELIMINATED, same cost as L0 |
| L2_i75_native | 512x288 | whole-clip | 0.75 | native | 5.40 | 14473 | 0 | 0 | no-seam | |
| L2_i75_respaced | 512x288 | whole-clip | 0.75 | re-spaced | 6.18 | 14473 | 0 | 0 | no-seam | re-spaced sigma gamble |
| L2_i62_native | 512x288 | whole-clip | 0.62 | native | 6.16 | 14462 | 0 | 0 | no-seam | lower i2v = less init-hold? |
| L2_i62_respaced | 512x288 | whole-clip | 0.62 | re-spaced | 5.40 | 14415 | 0 | 0 | no-seam | |
| L3_640 | 640x384 | whole-clip | 0.75 | native | 8.98 | 14476 | 0 | 5 | no-seam | bigger render; freezes APPEARED |
| L3_704 @reserve4 | 704x384 | whole-clip | 0.75 | native | 8.74 | 14534 | -- | -- | -- | ABORTED: over the 14.5 GB ceiling |
| L3_704 @reserve5 | 704x384 | whole-clip | 0.75 | native | 8.75 | 13355 | 0 | 0 | no-seam | fits only with a 5 GB VRAM reserve |

Two more results:
  - FREE composite-scaler A/B at the common 1472x832 canvas (no GPU): lanczos+unsharp is
    14.7% sharper than the current bilinear (Laplacian variance 27.17 vs 23.69). This is the
    only apples-to-apples sharpness number; the per-leg native-resolution sharpness is NOT
    comparable across canvases (a smaller frame scores higher per-pixel), so do NOT read
    "512 is sharpest" from native Laplacian -- judge canvas sharpness by eye on the clips.
  - The seam metric "ratio" (jump / local median): L0 = 2.07 (a real visible seam), L1d =
    0.57 (seam below the noise floor), whole-clip = no seam.

What I want from you: recommend the SINGLE best combination for a good balance of
PERFORMANCE (s/it + VRAM headroom on a 16 GB card whose desktop apps already eat ~3-5 GB)
and QUALITY (kill the flash + the softness + the stutter), and rank your top 2-3 with one
line of rationale each. Specifically weigh in on:

  (a) DECODE: whole-clip (4096/8, seam=0, same speed/VRAM as baseline) vs the best tiled
      (L1d 128/32, seam ratio 0.57, slightly cheaper VRAM). Is whole-clip the right default
      given it sits ~14.3-14.5 GB (160-200 MB under the ceiling) on this knife-edge card,
      or is the tiny extra headroom of a tiled decode worth a faint residual seam?
  (b) CANVAS: stay 512x288 (fast, most headroom) or bump to 640x384 (1.6x slower, at the
      VRAM edge, and freezes=5 appeared) or 704x384 (needs a 5 GB reserve, 1.6x slower)?
      Is a bigger native render worth it if the output is upscaled to 1472x832 either way,
      and the cheap lanczos+unsharp scaler already recovers ~15% sharpness for free?
  (c) i2v STRENGTH 0.75 vs 0.62 and native vs re-spaced sigmas: the objective metrics do
      NOT separate these (seam=0, freezes=0 for all). Given distilled models are calibrated
      to their native sigma schedule, how much weight should the re-spaced gamble get, and
      is dropping i2v to 0.62 a sound lever for the init-hold stutter or a coherence risk?

State your assumptions. I will make the final call after eyeballing the clips.

---------------------------------------------------------------------------------------

## My (Claude's) read, for reference -- the safe balanced ship, pending your eyeball

- DECODE = whole-clip 4096/8. It ELIMINATES the seam (the #1 complaint) at the SAME s/it
  (5.37) and VRAM (14338) as the baseline, and it matches the shipped sister engine
  `eng_ltx_video` decode, so a win wires byte-for-byte. The tiled L1d (128/32) is the
  fallback if the whole-clip's ~160 MB headroom ever feels too tight under desktop load.
- SCALER = lanczos+unsharp. +14.7% sharpness at ZERO GPU cost -- the cheapest softness win.
- CANVAS = stay 512x288 for the default ship (fast, most headroom, 0 freezes). 640/704 are
  optional "hero" tiers but cost ~1.6x speed, ride the VRAM edge, and 640 showed freezes.
- i2v / sigmas = keep 0.75 + native as the safe default; treat i2v 0.62 and re-spaced sigmas
  as eyeball-only experiments for the stutter (objective metrics can't separate them).

Recommended balanced winner: **512x288 + whole-clip decode (4096/8) + lanczos+unsharp
scaler + i2v 0.75 + native sigmas** -- seam gone, ~15% sharper, 0 freezes, baseline speed.
