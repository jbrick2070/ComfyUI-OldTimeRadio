# HuMo standalone bakeoff -- RESULTS (2026-06-27)

Diagnostic build per `roundtables/2026-06-27-humo-optim/final.md` (r1->r4 kibitz-converged).
**Production was NOT touched** (eng_humo.py / workflows/otr_scifi_16gb_full.json / the OTR
pack `__init__.py` are byte-identical to HEAD). This run RENDERS clips + metrics for the
operator eyeball and changes nothing. All promotion is DEFERRED and operator-gated.

## What was built (new files only)
- `custom_nodes/otr_bakeoff_helper/__init__.py` -- a SIBLING node package (outside this
  repo, removable) registering `OTR_BakeoffReclaim`: a LATENT passthrough that, mid-graph,
  evicts ONLY the umt5 CLIP + whisper audio-encoder (keeps the UNET/LoRA/VAE), with an
  always-dirty `IS_CHANGED` and a unique per-call marker. Unit test: 5/5 pass.
- `scripts/build_humo_bakeoff_workflow.py` -- reuses `HuMoEngine._build_graph` READ-ONLY and
  translates the run_graph spec into a ComfyUI `/prompt` JSON (SaveImage terminal; splices
  OTR_BakeoffReclaim on the WanHuMoImageToVideo(slot2)->KSampler latent edge for two-stage
  legs). `--dry-validate` PASS.
- `scripts/run_humo_bakeoff.py` -- boot-per-leg headless runner (no-FLOOR so OTR_ENABLE_HUMO=1),
  selective ancestor-safe CIM reset before every leg, external NVML peak, fail-loud manifest +
  checkpoint-exists (handles the V3 COMBO `/object_info` enum), SaveImage->production silent
  encoder, ffprobe verify, blue-cast (PIL+numpy), soft-gated face metrics.

## Run conditions
Fixed still `c02_466a19906ccb.png` + audio `c02_b002_line.wav`, seed 0, 49 frames @ 25fps
(832x480 wide), SILENT clips via `wrapper_bridge.encode_frames_to_silent_mp4`. RTX 5080 16GB.
Boot-per-leg; reset before each. VRAM = external nvidia-smi render-window peak.

## Metrics

| leg | engine | two-stage | peak VRAM MB | <=13500 gate | s/it | B-R frame | B-R still | clip |
|---|---|---|---|---|---|---|---|---|
| i_14B_single | humo_14B_169 | no | **15996** | FAIL | 18.7 | 9.94 | 26.88 | i_14B_single.mp4 |
| ii_14B_twostage | humo_14B_169 | yes (reclaim) | **15779** | FAIL | 18.7 | 9.96 | 26.88 | ii_14B_twostage.mp4 |
| iii_1p7B_control | humo_1.7B_169 | no | **15089** | FAIL | 5.1 | 21.86 | 26.88 | iii_1p7B_control.mp4 |
| iv_sentinel_14B_twostage | humo_14B_169 | yes + LTX-AV resident | **15974** | FAIL | 18.7 | 9.96 | 26.88 | iv_sentinel_14B_twostage.mp4 |

(Sentinel: a real `ltx_audio_in` render -- Gemma TE + LTX VAEs, NOT Whisper -- ran first in
the same resident session and peaked 14944 MB; the two-stage HuMo then peaked 15974 MB on top.)
All four clips exist under `output/otr/episodes/_bakeoff_humo/` (181-315 KB each). The
`OTR_BakeoffReclaim` node fired on every two-stage leg: `resident=3 evicted=2 (text=1 audio=1)
kept vae=1 sampler_survived=True`.

## Verdict (the operator decision gate)
**The 14B does NOT fit <= 13.5 GB with real headroom** -- on this 16 GB card every HuMo leg
rides ~15-16 GB:
- The two-stage encoder eviction shaved only **~217 MB** off the single-graph 14B
  (15996 -> 15779). The fp8 14B UNET weights + Wan video activations dominate the peak; evicting
  the ~5 GB umt5+whisper block does not translate into a 5 GB peak drop (ComfyUI already offloads
  the TE under pressure, and the allocator cache / sampler activation refills the freed space --
  the documented BUG-265 behavior).
- Even the **shipping 1.7B control rides 15089 MB** at these settings, so the high peak is
  driven mostly by the umt5_xxl TE + whisper + 49-frame video latents, not the UNET size alone.
- Cross-engine (sentinel) does not change the picture: 15974 MB.

Per the final.md / GO_FORWARD JOB 3 rule ("fit 14B safely with REAL headroom -> promote; else
keep 1.7B"): **KEEP humo_1.7B** -- the 14B has no VRAM headroom even with the two-stage lever.

## Quality signal (for the eyeball)
Blue-cast (mean B-R of frames vs the source still's +26.88):
- 14B legs: **B-R ~9.9** (well colour-balanced, red recovered).
- 1.7B control: **B-R ~21.9** (closer to the still -> noticeably bluer / less corrected).
So the 14B clips look more colour-correct, but at an unsafe VRAM peak. The operator may prefer to
**harden the 1.7B colour (cfg / de-blue)** rather than promote the 14B. Eyeball the four clips.

## Caveat on the VRAM numbers
nvidia-smi "used" for a `--cuda-malloc` process includes the allocator's reserved/cached pool, so
the absolute peaks slightly over-state the true requirement. The **relative ordering**
(1.7B 15089 < 14B-two-stage 15779 < 14B-single 15996 ~ sentinel 15974) is the reliable signal,
and NONE reach the 13.5 GB safe-fit gate.

## Deferred (operator-gated, NOT done here)
The production two-stage split + a VramPeakProbe in eng_humo + the `config/profiles/16gb_full.json`
1.7B->14B flip are all DEFERRED. Nothing was promoted. Re-run any time:
`python scripts/run_humo_bakeoff.py` (or `--dry-validate`). The sibling helper lives at
`custom_nodes/otr_bakeoff_helper/` (outside this repo).
