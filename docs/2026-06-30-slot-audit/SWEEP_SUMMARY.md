# ALL-ENGINES x ALL-SLOTS SWEEP -- RESULTS (2026-06-30, live on the prod JSON)

Every video/still engine driven in-slot on `workflows/otr_scifi_16gb_full.json` via the all-5-role
capability profile (`slot_matrix.build_all_five_role_profile`), images = z_image_turbo (nvfp4), kokoro
voice. C0-C5 slot-audit fixes SHIPPED first; this soak proves them live. Each obs final was presented to
the operator for eyeball.

## VERDICTS

| # | engine | slots run | result | operator verdict |
|---|--------|-----------|--------|------------------|
| 1 | **still_flat** | all 5 | SUCCESS ~8m | KEEP -- "great still" |
| 2 | **still_pan** | all 5 | SUCCESS ~9m | KEEP -- "ok" |
| 3 | **still_motion** | all 5 | SUCCESS ~9m | KEEP -- "ok" |
| 4 | **visualizer** | ann/mus/char | SUCCESS ~8m | KEEP -- "no issues" (RENAME -> `viz_green`) |
| 5 | **humo_1.7B** | character | SUCCESS ~20m | KEEP -- portrait "great"; clip-mush + movement fixes queued |
| 6 | **still_parallax** | ann/mus/char | SUCCESS ~9m | **CUT 100%** -- "kinda weird but sucks" (retire queued) |
| 7 | **mesh_stage** | ann/mus/char | SUCCESS ~16m | KEEP -- "one must-have"; needs radio-opening + more headroom |
| 8 | **humo_14B_169** | ann/mus/char | SUCCESS ~19m | VIABLE with freed headroom + no multitasking; opening radio "great", some shots static |
| 9 | **ltx_video** | all 5 | SUCCESS ~17m | KEEP -- real LTX motion in every slot |
| 10 | **ltx_audio_in** | ann/mus/char | SUCCESS ~25m | KEEP -- fits VRAM; slow (full LTX-AV reload per beat) |
| 11 | **wan_i2v** | ann/mus/char | **OOM / FAIL** | NOT viable on 16 GB (14B + z_image > VRAM) + drifts off the still -> back-burner |
| 12 | **wan_ti2v** | ann/mus/char | SUCCESS ~24m | KEEP -- the 8 GB tier fits where the 14B i2v OOM'd |

(humo_1.7B_169 was also proven earlier in the music-still episode; the single-beat 14B music test rendered
in ~1.7m and the opening radio shot was operator-praised.)

## KEY FINDINGS
- **The slot-audit fix works live:** every capability-eligible engine renders real content in every slot
  it fits; stills mint (no black floor); visualizer/viz_mxc correctly mint NO still; the character slot
  delivers real video (humo/ltx/wan) instead of a static still. `delivered_engine` confirmed in the logs
  (e.g. `inter-beat reclaim still_flat->humo_1.7B_169`).
- **z_image_turbo (nvfp4) is the fast image tier** (~10s/still, 8 steps) vs flux (~30s + reload) -- wire it
  as the default image engine for fast iteration.
- **14B HuMo is viable** for single or few beats WITH freed VRAM headroom and no CPU-contending work
  (kibitz/suite); it spills but ComfyUI's dynamic offload handles it (~17s/step). Not for many-beat episodes.
- **wan_i2v OOMs** on 16 GB in the all-slots config (no fallback -> LOUD episode fail); back-burnered.
- **CPU/ffmpeg engines are the accessible floor:** stills + visualizer + the new viz_mxc_cpu need no GPU and
  render fastest (viz_mxc mints no image at all).

## FOLLOW-UPS (all in GO_FORWARD_PLAN.md)
- RETIRE still_parallax (100% rip-out). RENAME visualizer -> viz_green.
- HuMo improvements (portrait quality @ 1.7B/same-VRAM, clip-underrun mush fix, radio-host bookends, dropdown
  labels, HuMo-isolation smoke) -- kibitz post-soak.
- mesh_stage: opening = 3D radio + more headroom (MIN-ACCEPT) + optional r1 kibitz.
- viz_mxc_cpu SHIPPED this session (rainbow visualizer, green + pushed); viz_mxc_gpu deferred.
