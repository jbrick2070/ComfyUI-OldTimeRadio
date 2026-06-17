# Judgment -- pass01 (Claude grounded the panel vs eng_ltx_av / eng_ltx_video / registry / render_driver / role_compat)

Panel (GPT-5.5 / Gemini-3.1-pro / Grok-4.3 / DeepSeek-v4, ~$0.30) converged 4/4. All MUST-FIX
below are CONFIRMED against the real code -> they ARE the build steps.

## CONFIRMED build steps (fold into C1/C2)
**C1 -- sharp recipe in eng_ltx_av (mirror eng_ltx_video, do NOT import it -- it's frozen):**
- Add `_SigmasFromValues` + `LTX_DISTILLED_SIGMAS` DUPLICATED into eng_ltx_av (lazy torch).
- Add `_distilled_lora_file()` + `_LTX_DISTILLED_LORA_*` (name @0.70) + `_FLOOR_LORA` (5 GiB) + `_FLOOR_PROJECTION_CKPT` (30 GiB).
- `_weight_paths()`: ADD distilled LoRA + projection ckpt (currently both missing -> late failures).
- Sharp mode in `_build_graph` (BOTH branches): wire `unet -> LoraLoaderModelOnly@0.70 -> CFGGuider.model`; DROP `modelsampling`; replace `sched`(LTXVScheduler) with the `sigmas` injector; `KSamplerSelect=euler_cfg_pp`; `cfg=1.0`; i2v strength 0.75. M0 recipe stays selectable via `OTR_LTX_AV_SHARP` (default ON; a CONFIG mode, NOT a fallback).
- `_node_candidates`: sharp set adds lora/samplersel/noise/guider/sampleradv, drops modelsampling/sched/ksel (mirror `_node_candidates_sampling`).
- `_retain_model_patchers`: retain `"lora"` (not modelsampling) in sharp mode (V-4 teardown / no VRAM leak).
- `render_clip`: `run_graph(..., free_after_use=True, keep={"unet","lora",_TERMINAL})` -- WITHOUT this the encoder co-resides and trips the 14.5 assert (likely also drops the engine-path peak BELOW the /prompt-smoke 15.3 GB).
- `commercial_clean=True` (license-clean, like eng_ltx_video); `_LTX_AV_NATIVE_W/H` -> 832/480 (no 1472x832 engine fallback).
- CPU graph-shape test: sharp graph has lora/sigmas/euler_cfg_pp/cfg1.0, NO modelsampling/sched; M0 graph keeps them.

**C2 -- defaults + JSON (ATOMIC, one commit -- GPT#9):**
- `LtxAvMusicEngine.roles += "announcer_visual"`; `default_roles=("music_visual","announcer_visual")`.
- eng_ltx_video `default_roles` -> `()` (un-claim; else default_engine_for_role still returns ltx_video).
- Flag opt-OUT: `assert_usable` step 1 -> raise only if `OTR_ENABLE_LTX_AV=="0"` (default ON), update docstrings.
- render_driver: null `SYNTH_FALLBACKS[ltx_av_*]` + adapter `fallback_engine=None` (no-fallbacks); `OTR_LTX_AV_RENDER_CANVAS` default 512x288 -> **832x480**.
- registry CAPABILITIES: add LoRA + projection-ckpt to both ltx_av rows' model_requirements.
- Workflow JSON `otr_scifi_16gb_full.json`: OTR_VideoDirector announcer/music dropdowns -> `ltx_av_music`; re-validate (OTR_WorkflowValidator + round-trip + link/widget audit). Land WITH the registry change.

**Claude's added design decision (not in the panel; from comparing the smokes to the engine):**
- The engine's music lane is t2v (EmptyLTXVLatentVideo, no still); your loved smokes were **i2v on a
  scene still + audio**. role_compat confirms announcer+music supply `init_image`. So the
  music/announcer ltx_av default must **i2v on the beat scene still when present** (mirror
  eng_ltx_video `_use_i2v` default ON), falling to empty-latent t2v only if no still. Implement in C1.

## Verify-at-build (quick reads during coding)
- `engine_registry_base.default_engine_for_role` resolution order (after un-claim) + default_roles<=roles validation.
- confirm SYNTH_FALLBACKS/fallback_engine are not READ by any live degrade path post-547671d (then nulling is cleanup).
- announcer beats always carry audio_ref in the plan builder (no-fallbacks -> else LOUD fail).

## CUTS (panel consensus)
- S1 label rename ("LTX (Audio In)"): label-only, keep IDs (314 refs), do LAST / non-blocking.
- No alias map / full-ID rename. No per-chunk push of S2 vs S3/S4 -- defaults+JSON push ATOMICALLY.
- M0 toggle: keep (cheap, default sharp) -- minor disagreement with DeepSeek's "delete M0"; harmless.
