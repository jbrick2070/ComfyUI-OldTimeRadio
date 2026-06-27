# LTX-AV recipe selector -- anchor plan (R0)

## Problem
`eng_ltx_av.py` (the `ltx_audio_in` A2V engine that serves ALL THREE
`OTR_VideoDirector` slots: announcer_video_model + music_video_model +
other_beats_video_model) currently has a BINARY recipe switch
`_sharp_enabled()` (`OTR_LTX_AV_SHARP`, default ON):

- ON  -> SHARP: distilled LoRA @0.70 + euler_cfg_pp + 8-step LTX_DISTILLED_SIGMAS
        + cfg 1.0 + i2v strength 0.75; ModelSamplingLTXV + LTXVScheduler DROPPED.
        Built for the **dev** Q3_K_M GGUF (the LoRA adds the distillation).
- OFF -> M0 base: ModelSamplingLTXV + LTXVScheduler + euler + cfg 3.0 + strength 1.0,
        no LoRA.

The 2026-06-26 quant bakeoff (3-way panel: Claude + Gemini + Codex converged)
picked **distilled-1.1 Q3_K_M, no LoRA** as the daily driver (dev-comparable
faces, 9.44 s/it, peak 15148 MB) and kept **dev Q3_K_M + SHARP LoRA** as the
hero/final path. The distilled-1.1 GGUF BAKES the distillation in, so it must run
the distilled recipe WITHOUT the separate LoRA -- a THIRD config the binary flag
cannot express. Pointing OTR_LTX_AV_UNET at the distilled file with SHARP still ON
would stack the LoRA on an already-distilled model (double-distill -> worse).

## Goal (operator anchor -- do not relitigate)
1. **Recipe FOLLOWS THE MODEL.** Operator flips daily<->hero by swapping
   `OTR_LTX_AV_UNET` only (distilled -> distilled_native; dev -> sharp_lora).
2. Keep BOTH recipes; add a third. Tri/quad-state selector replaces the binary flag.
3. Explicit override `OTR_LTX_AV_RECIPE` = `auto | sharp_lora | distilled_native | m0_base`.
4. **FAIL LOUD** on an ambiguous unet name with no override -- never silently
   double-distill or guess.
5. A2V (`ltx_audio_in`) ONLY. `eng_ltx_video` stays FROZEN, never imported.
6. Must work UNIFORMLY across all three director roles (announcer/music/other_beats).

## Recipe matrix
| aspect            | sharp_lora (dev)     | distilled_native (distilled-1.1) | m0_base        |
|-------------------|----------------------|----------------------------------|----------------|
| LoRA @0.70        | YES                  | NO                               | NO             |
| ModelSamplingLTXV | NO                   | NO                               | YES (2.05/0.95)|
| sigmas            | LTX_DISTILLED_SIGMAS | LTX_DISTILLED_SIGMAS             | LTXVScheduler  |
| sampler           | euler_cfg_pp         | euler_cfg_pp                     | euler          |
| cfg               | 1.0                  | 1.0                              | 3.0            |
| i2v strength      | 0.75                 | 0.75                             | 1.0            |
| model head -> guider | LoRA-wrapped unet | unet directly                   | ModelSampling  |
| required weights  | +distilled LoRA      | (none extra)                     | (none extra)   |

`distilled_native` == `sharp_lora` MINUS the LoRA node + LoRA weight requirement
(model head = unet directly). This is EXACTLY what the bakeoff distilled legs ran.

## Proposed expression (the thing to harden)
A single resolver `self._recipe()` consulted by all four call sites
(`_weight_paths`, `_node_candidates`, `_build_graph`, `render_clip`):

```
OTR_LTX_AV_RECIPE (default "auto"):
  auto             -> detect from unet name
  sharp_lora       -> force dev+LoRA recipe
  distilled_native -> force distilled-no-LoRA recipe
  m0_base          -> force the legacy base pass
  <anything else>  -> raise (fail loud: bad override)

auto detection from OTR_LTX_AV_UNET basename (lowercased):
  contains "distilled" and not "dev"  -> distilled_native
  contains "dev"       and not "distilled" -> sharp_lora
  otherwise (ambiguous / neither)     -> RAISE EngineUnusable(MALFORMED_CONFIG)
```

Resolution is fail-closed in `assert_usable` (so an ambiguous unet is caught at
the gate, before any GPU forward), and the same resolver feeds the graph build.

## Open questions for the panel (1-2 rounds)
1. **Auto-detect vs explicit flag default.** Is substring detection
   ("distilled"/"dev" on the basename) robust enough as the DEFAULT, or should
   default be `auto` that requires the name to clearly match a known family map,
   else fail loud? Risk: a future filename like `ltx-2.3-22b-dev-distilled-...`
   trips both tokens -> our rule sends it to ambiguous (RAISE). Good or too brittle?
2. **Ambiguous handling.** Fail loud (raise) vs fall back to a safe explicit
   default with a LOUD warning. Operator says fail loud -- confirm no role needs a
   silent default (announcer/music t2v vs character i2v).
3. **m0_base reachability.** Override-only (never auto) -- correct? It is a legacy
   escape hatch, never auto-selected.
4. **Back-compat of `OTR_LTX_AV_SHARP`.** Nothing in the repo/scripts sets it
   (verified). Cleanbreak-delete it, or honor `OTR_LTX_AV_SHARP=0 -> m0_base` as a
   thin legacy shim? (CLAUDE.md cleanbreak framing leans delete.)
5. **Where the LoRA-vs-no-LoRA + ModelSampling-vs-not decisions live** so the
   three roles (i2v character + t2v music/announcer) stay uniform -- any role-
   specific interaction with the recipe (e.g. t2v has no LTXVImgToVideo strength)?

## Wiring note (CLAUDE.md S0)
The LTX-AV unet + recipe are env/code-driven (`OTR_LTX_AV_UNET` /
`OTR_LTX_AV_RECIPE`). `otr_scifi_16gb_full.json` contains NO UnetLoaderGGUF / LTX-AV
nodes -- the engine runs in-process via OTR_VideoRenderBatch + wrapper_bridge, not
as litegraph nodes. So there is **no unet/recipe widget to change**; this is a
pure engine-code change. The canonical JSON will be round-trip + validator checked
to confirm it is unaffected.

## Tests + smoke (R-build)
- distilled unet + auto -> distilled_native (no LoRA in graph, no LoRA weight req).
- dev unet + auto -> sharp_lora (LoRA present).
- OTR_LTX_AV_RECIPE override wins over the unet-derived default.
- ambiguous unet + auto -> raises (fail loud).
- bad OTR_LTX_AV_RECIPE value -> raises.
- GPU smoke: distilled_native renders clean in announcer + music + other_beats slots.
