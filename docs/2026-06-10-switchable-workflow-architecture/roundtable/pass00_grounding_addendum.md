# Judge's Grounding Addendum (verified against the repo 2026-06-10)

Facts below were verified line-by-line against the real codebase by the judge BEFORE this pass.
Treat them as GROUND TRUTH. Do not re-litigate them; build on them. Critique the PLAN, not these facts.

## Verified mechanisms that ALREADY exist (do not propose re-inventing them)
- `OTR_VideoDirector` (nodes/otr_video_director.py:79-130) + `OTR_ImageDirector`: per-role engine dropdowns
  (announcer / music / other-beats roles; character_video, scene_broll, background_abstract). Role-compat
  filtering happens at execute time via nodes/_otr_shared/role_compat.py -- the COMBO list is never mutated.
- Engine registries with fallback chains for audio AND video AND image:
  nodes/_otr_video_engines/registry.py, nodes/_otr_image_engines/registry.py, nodes/_otr_audio_engines/registry.py
  (shared base: nodes/_otr_shared/engine_registry_base.py). Fail-closed, LOUD.
- ~17 `OTR_ENABLE_*` env flags already gate engines (HUMO, LATENTSYNC, LTX_VIDEO, WAN_I2V, FLUX2_KLEIN, CHROMA,
  QWEN_IMAGE, ZIMAGE, LUMINA, SD35, HIDREAM, CHATTERBOX, DIA, STABLE_AUDIO, STABLE_AUDIO_3, INDEXTTS2, OPENROUTER, ...).
- `OTR_FORCE_ENGINE_MAP` EXISTS (nodes/_otr_video_engines/render_driver.py:607-653): `role=engine` /
  `*=engine` grammar, parse errors are LOUDLY ignored, fallback chains stay intact. Used by the marathon
  soak runner (scripts/_otr_soak_marathon.py).
- Sage attention is gated by consulting `comfy.model_management.sage_attention_enabled()`
  (nodes/_otr_video_engines/motion_common.py:63-77) -- i.e. an env/launch-level capability gate already exists.
- VRAM: `VRAM_CEILING_MB = 14500` in nodes/_otr_video_engines/wrapper_bridge.py:37; `reclaim_idle_models()`
  + `free_after_use` lifecycle in the same module. Single-heavy-engine residency is an enforced invariant.
- Workflow contract: nodes/_otr_workflow_validator.py enforces widgets_values length == serialized slot count
  (incl. hidden control_after_generate companions). Tests: test_otr_workflow_validator.py,
  test_workflow_live_passes_validator.py, test_workflow_validator_widget_vector.py.
- Model paths: nodes/_otr_paths.py honors HF_HOME (line ~490) and extra_model_paths.yaml. Downloaders exist
  piecemeal (scripts/download_models.sh, download_ltx_2_3.ps1, download_video_stack_weights.ps1,
  hf_download_driver.py, _otr_dl_indextts2_refs.py) -- no unified manifest/installer yet.
- ffmpeg: all composition/caption/upscale paths use libx264; master mux otr_master_audio_mux uses -c:v copy
  -c:a copy; the ONLY h264_nvenc is video_engine.py:396-447 behind _check_nvenc() with libx264 fallback.
- MPS: zero `torch.backends.mps` references in the codebase. Device checks are torch.cuda.is_available() +
  sys.platform=='win32' (~35 files). There is NO centralized device-routing module today.

## The drift bug, mechanically (sharper than the plan states)
There is NO literal second graph. The headless path (scripts/otr_api.py + scripts/queue_smoke.py +
the soak runners) loads THE SAME workflows/otr_scifi_16gb_full.json (30 KB, the only production workflow;
everything else in workflows/ is fixtures/external examples), converts it to an API prompt via live
/object_info schemas, and patches widgets BY NAME at submit time. The drift is therefore:
**saved widget defaults inside the production JSON** vs **the patch-set hard-coded in each headless script**.
The "second source of truth" is the per-script patch list. Any profile design must be consumed by BOTH the
UI load path and the headless submit path, or the drift bug survives.

## What does NOT exist yet (the green field this plan must define)
- No profiles.json / hardware-tier concept anywhere.
- No build/export/generator step producing per-tier workflow snapshots from a master.
- No onboarding wizard / auto-detector.
- No MPS routing; no centralized device-choice module.

## Platform invariants the plan may NOT break
- Mux-LAST with frozen audio; output mp4 audio must stay byte-identical to the master mix.
- Single heavy engine resident at a time, <=14.5 GB ceiling on the 16 GB tier.
- Determinism: seed-keyed reproducibility per profile; creative RNGs draw OS entropy unless
  OTR_CAST_SEED/OTR_STYLE_SEED env overrides are set.
- Fail-closed + LOUD on missing engines/toolchains (log swap + ledger restamp; never silent).
- widgets_values converter/validator contract must pass in every shipped artifact.
