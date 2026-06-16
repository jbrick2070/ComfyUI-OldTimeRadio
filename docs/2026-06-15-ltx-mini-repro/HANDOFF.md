# HANDOFF — Standalone MINI LTX bookend repro workflow (operator-requested 2026-06-15)

**Goal (operator, verbatim intent):** a tiny, standalone ComfyUI workflow that reproduces the GOOD
6/6 LTX radio bookends — **just the FLUX portion (mint the radio-bookend still) + the EXACT 6/6 LTX
animate portion + one easy output node so the operator can SEE the clip.** NOTHING else: **NO HuMo,
NO upscale, NO audio, NO full episode pipeline, NO compositor.** This is an A/B harness to isolate the
LTX look, not production.

The operator is NOT chasing BUG-413 (the episode-level "LTX falls to the procgen floor" wiring bug) in
this window. This window only rebuilds the minimal LTX render so the good 6/6 motion can be reproduced
and eyeballed in isolation.

---

## The reference clips to reproduce (the GOOD ones)
- `output\otr\episodes\signal_lost_mimicry_in_the_void_20260606_013833\videos\b001.mp4` (open bookend)
- `output\otr\episodes\signal_lost_mimicry_in_the_void_20260606_013833\videos\b005.mp4` (close bookend)

**Ledger-proven recipe for BOTH (commit `c9af198`, 6/6):**
| field | value |
|---|---|
| `ltx_engine` | `v0_9` (LTX-2B v0.9: `ltx-video-2b-v0.9.safetensors`) |
| `ltx_engine_label` | `v0_9_euler_cfg_pp` |
| `ltx_sampler_name` | **`euler_cfg_pp`** (CFG++) |
| steps / cfg | **8-step distilled / cfg 1.0** (CFGGuider) |
| `ltx_loop_via_reverse` | **True** (ffmpeg boomerang ping-pong) |
| `ltx_length` | **209** (b001) / **233** (b005) frames — audio-derived, `8n+1` |
| dims | **832 x 480** (LTX-2B native — the whole point; do NOT render at 1472x832, it mushes) |
| i2v strength | **0.75** (LTXVImgToVideoConditionOnly soft anchor) |
| `ref_source` | `ltx-radio-bookend` (animates the FLUX radio-bookend still) |
| `source_kind` | `ltx` |
| VAE decode | `VAEDecodeTiled` (tile 512 / overlap 64 / temporal 4096 / temporal_overlap 8) |

The exact FLUX radio-bookend still those clips animated is ON DISK and reusable directly:
`output\otr\episodes\signal_lost_mimicry_in_the_void_20260606_013833\stills\radio_bookend_signal_lost_mimicry_in_the_void_20260606_013833.png`
(meta `radio_bookend_prompt_source = "dynamic (story_brief_status=ok)"`).

---

## Where to pull the EXACT workflow from git (commit `c9af198`)
Both nodes were DELETED later (the LTX node in the cleanbreak `70d379b`), so pull from `c9af198`:
- **LTX recipe (code):** `git show 70d379b^:nodes/batch_ltx_render.py` (or `c9af198:nodes/batch_ltx_render.py`)
  — holds `LTX_V0_9_SAMPLER_NAME_DEFAULT="euler_cfg_pp"`, `LTX_WIDTH=832`/`LTX_HEIGHT=480`,
  `LTX_I2V_STRENGTH=0.75`, `LTX_CFG=1.0`, the 8-step distilled SIGMAS, `LTX_LOOP_VIA_REVERSE_DEFAULT="on"`,
  the VAEDecodeTiled params, and the `_build` graph topology.
- **FLUX bookend (code):** `git show e4cb3ac:visual/batch_flux_render.py` (`OTR_BatchFluxRender`) — the
  radio bookend path: FluxGuidance **3.5**, `_RADIO_FALLBACK_PROMPT` / `_RADIO_PROMPT_SUFFIX`
  (broadcast-distress), `radio_bookend_seed=4242`, flux1-dev-fp8 / steps 20 / cfg 1.0 / euler+simple.
- **Saved widget values (JSON):** `git show c9af198:workflows/otr_scifi_16gb_full.json` — node `55`
  `OTR_BatchLTXRender` (seed/method/cap) + node `23` `OTR_BatchFluxRender` (the bookend widgets: guidance
  3.5, style_suffix, radio_bookend_seed 4242).

---

## The fastest path — a tool that is already ~90% this workflow
`scripts/otr_ltx_motion_smoke.py` ALREADY IS the LTX mini-graph: ONE still in -> one short clip out via
`SaveWEBM`, NO audio/Flux/HuMo/episode. Its `--mode goofer` is the 8-step distilled
`KSamplerSelect + ManualSigmas + RandomNoise + CFGGuider(cfg 1.0) + SamplerCustomAdvanced` chain that
MIRRORS the real engine path, and its `GOOFER_SIGMAS` are the 6/6 8-step distilled schedule.

**Exact invocation to reproduce b001 (open) on the real bookend still** (server up on :8000, LTX lane
`OTR_ENABLE_LTX_VIDEO=1`; first copy the radio_bookend png into the comfy `input\` dir as the `--still`):
```
python scripts\otr_ltx_motion_smoke.py --mode goofer --sampler euler_cfg_pp ^
  --strength 0.75 --width 832 --height 480 --length 209 ^
  --still radio_bookend_mimicry.png --tag repro_b001
```
(I already proved this harness renders a clean 832x480 LTX clip in ~18s on the live box, 2026-06-15.)

**This window's actual deliverable:** turn that into a SELF-CONTAINED workflow the operator can open and
watch — either (A) a small **ComfyUI graph JSON** `workflows/ltx_bookend_mini_repro.json`
(FLUX-mint-bookend -> LTXV i2v animate -> SaveWEBM), or (B) extend `otr_ltx_motion_smoke.py` with a
`--flux-bookend` front-stage that mints the still first (flux1-dev-fp8 + FluxGuidance 3.5 + the radio
prompt) so it's one click, no episode. Operator preference: simplest thing that lets them SEE the video.
Add the **boomerang** as an optional ffmpeg post-step (or skip for v1; it only loops the clip).

---

## Acceptance
A `SaveWEBM` clip at **832x480**, `euler_cfg_pp` / 8-step distilled / cfg 1.0 / i2v 0.75 / length ~209,
animating the radio-bookend still, that **matches the MOTION of `b001.mp4` / `b005.mp4`** (dial sweep,
tube pulse, slow push-in — sharp, not mushy). No HuMo, no upscale, no audio, no compositor in the graph.

## Hard rules (CLAUDE.md)
Single resident heavy <=14.5GB; 100% local; UTF-8 no BOM; SFW; commit+push per green chunk on
`v2.0-alpha` (the mini workflow JSON + any script change are additive — do NOT touch the production
`otr_scifi_16gb_full.json`, the LTX-audio lane's `render_driver.py`, or Dakota's `wrapper_bridge.py`).
This is a standalone repro harness, fully non-colliding with the active lanes.
