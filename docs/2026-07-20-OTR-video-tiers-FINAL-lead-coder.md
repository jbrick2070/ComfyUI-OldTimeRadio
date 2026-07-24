# OTR Video Tiers — FINAL Lead-Coder Build Directive

**Date:** 2026-07-20  
**Status:** FINAL — implement these names and behaviors now  
**Scope:** Add two normal selectable 8GB video engines, expose the two already-working 16GB LTX-2.3 engines under final user-facing names, preserve the active 16GB internal latent upscaler, and avoid all VRAM gating/profile scaffolding.

---

## 1. Operator decisions — locked

1. **Do not label any engine “experimental” or “candidate.”**
2. **Do not add VRAM gates, GPU eligibility checks, hardware whitelists, automatic blocking, auto-downgrades, or warning scaffolding.**
3. **Do not add a new profile system or feature-flag system.** The registry is the menu; users may select any installed engine.
4. **Normal asset validation remains required.** A missing checkpoint, encoder, VAE, connector, or node should produce a precise missing-asset error. This is not a VRAM gate.
5. **Use the final public engine names now.** Do not plan a later “experimental → supported” rename.
6. **Preserve legacy engine IDs as aliases for saved-workflow compatibility.** Do not rip old IDs merely because the new display names exist.
7. **Testing determines the final saved workflow JSONs and documentation, not runtime permission.** The application must not decide whether a user is allowed to try an engine.
8. **All unneeded OTR models will be unloaded before video generation begins.** Whole-path VRAM testing must still include every component active during the selected video workflow.
9. **HuMo remains untouched.**
10. **UTF-8 without BOM, SFW, and commit/push each green chunk to `v2.0-alpha`.**

---

## 2. Final engine map

| Tier | Final selectable name | Implementation | Audio behavior | Upscaling |
| --- | --- | --- | --- | --- |
| 8GB | `ltx_8gb` | LTX-Video 0.9.8 distilled 2B | Silent T2V/I2V; mux OTR master audio afterward | Single-pass; none |
| 8GB | `wan_8gb` | Wan 2.2 TI2V-5B Q5_K_M; reuse existing `wan_ti2v` implementation | Silent T2V/I2V; mux OTR master audio afterward | Single-pass; none |
| 16GB A | `ltx23_16gb_audio_in` | Existing `ltx_audio_in` / `eng_ltx_av` IA2V route | Existing OTR audio conditions the video; master audio is preserved/muxed | Leave existing recipe unchanged |
| 16GB B | `ltx23_16gb_video` | Existing `ltx_video` / `eng_ltx_video` two-stage HQ route | No audio-in; silent output; mux OTR master audio afterward | Existing internal x2 latent upscaler remains active |

### Final user-facing labels

Use these labels from the first implementation:

- **LTX 0.9.8 2B — 8GB**
- **Wan 2.2 TI2V 5B — 8GB**
- **LTX 2.3 — 16GB Audio In**
- **LTX 2.3 — 16GB Video**

Do not append “experimental,” “candidate,” “beta,” or similar status wording.

---

## 3. `ltx_8gb` implementation

### Model and recipe

- Use `ltxv-2b-0.9.8-distilled.safetensors`.
- Do **not** use the original `ltx-video-2b-v0.9.safetensors` as the 8GB path.
- Do **not** describe this engine as LTX-2.3. It is the official 0.9.8 2B family.
- Use the exact 0.9.8 distilled workflow requirements: compatible T5 text encoder, VAE, scheduler/sampling settings, and guidance behavior.
- Treat 0.9.8 as its own adapter/recipe. Do not pretend it is only a model-filename swap inside the LTX-2.3 graph.

### Runtime behavior

- Support T2V and I2V where the installed nodes permit both.
- OTR’s normal use is silent I2V conditioned by the beat’s selected still.
- Use batch size 1.
- Prompt encoding may happen first; move conditioning to CPU and release/offload the text encoder before diffusion where the workflow supports it.
- Decode video, return a silent asset, and mux the OTR master audio on CPU.
- No internal latent, spatial, temporal, or post-generation upscaler in this 8GB workflow.

### Registration

- Add a normal `CAPABILITIES` row.
- `requires_flag=None` for the final `ltx_8gb` row.
- No VRAM query, device-memory threshold, compatibility gate, or automatic fallback.
- Keep only ordinary asset/node preflight checks.

---

## 4. `wan_8gb` implementation

### Model and reuse

- Use the existing `Wan2.2-TI2V-5B-Q5_K_M.gguf` asset.
- Reuse the proven `eng_wan_ti2v` adapter, orchestration, output handling, beat-fill logic, and cleanup helpers rather than forking a second implementation.
- Register `wan_8gb` as the final normal user-facing name.
- Preserve `wan_ti2v` as a legacy alias for saved-workflow compatibility.

### Runtime behavior

- Support T2V and I2V according to the existing implementation.
- Silent video output; mux OTR master audio afterward.
- Use batch size 1.
- Encode the prompt, release/offload UMT5 when possible, encode an optional source image, run Wan diffusion, release the transformer, then decode the VAE.
- No latent, spatial, temporal, or post-generation upscaler in this 8GB workflow.

### Registration

- The final `wan_8gb` row must be directly selectable without a new or inherited enable flag.
- `requires_flag=None` for `wan_8gb`.
- The old `wan_ti2v` alias may remain for compatibility, but selecting `wan_8gb` must not depend on `OTR_ENABLE_WAN_TI2V`.
- Keep ordinary asset/node preflight only.
- No VRAM gate, automatic fallback, GPU-brand gate, or backend whitelist.

---

## 5. 16GB A — `ltx23_16gb_audio_in`

This is a final display alias for the existing `ltx_audio_in` engine. Do not rebuild the engine.

### Existing behavior to preserve

- Existing engine: `ltx_audio_in` / `eng_ltx_av`.
- Shared transformer: `ltx-2.3-22b-dev-Q3_K_M.gguf`.
- Input: beat image/still plus the existing per-beat OTR audio.
- The audio conditions the video generation.
- The supplied OTR audio remains the authoritative master audio.
- The audio-VAE decode branch stays unwired if that is the current proven graph.
- `OTR_MasterAudioMux` performs the final audio mux.

### Required change

- Add the final user-facing name `ltx23_16gb_audio_in`.
- Preserve `ltx_audio_in` as a permanent legacy alias.
- Do not add a new gate, profile, hardware check, or separate engine fork.

---

## 6. 16GB B — `ltx23_16gb_video`

This is a final display alias for the existing `ltx_video` two-stage HQ engine. Do not rebuild the engine.

### Existing behavior to preserve

- Existing engine: `ltx_video` / `eng_ltx_video`.
- Shared transformer: `ltx-2.3-22b-dev-Q3_K_M.gguf`.
- Silent T2V/I2V route with no audio-in lane.
- Stage 1 creates motion at the lower latent resolution.
- The existing internal x2 latent upscaler runs.
- Stage 2 refines with the existing distilled LoRA.
- The final asset is silent; OTR master audio is muxed afterward.

### Internal upscaler — keep active

Do not remove, bypass, or defer the existing internal recipe upscaler:

- `LTXVLatentUpsampler`
- `LatentUpscaleModelLoader`
- `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`

It uses already-installed ComfyUI-LTXVideo nodes and remains part of the working 16GB HQ recipe.

### Required change

- Add the final user-facing name `ltx23_16gb_video`.
- Preserve `ltx_video` as a permanent legacy alias.
- Do not add a new gate, profile, hardware check, or separate engine fork.

---

## 7. Upscaling scope

### In scope now

- Keep the current 16GB LTX-2.3 two-stage internal x2 latent upscaler exactly as it works today.
- Keep both 8GB engines single-pass with no upscaler.

### Deferred to a separate project

The future model-agnostic upscaler bank is separate from this build. Its intended architecture is:

- a fourth registry namespace beside video, image, and audio engines,
- the same `engine_registry_base` pattern,
- self-registering upscaler adapters,
- per-adapter `CAPABILITIES`,
- registry is the menu,
- no enable gates,
- current LTX internal x2 path represented as the default bank entry when that project is built.

Do not add ad-hoc upscaler nodes or partial registry scaffolding in this video-tier sprint.

---

## 8. Whole-path VRAM accounting — testing only, never a runtime gate

No code should block selection based on these measurements. The operator will test each final workflow until stable and then save the final JSON variants.

### Pre-video cleanup contract

Before any of the four video routes begins:

1. Finish the OTR LLM work.
2. Finish TTS.
3. Finish source-image generation.
4. Unload the OTR LLM.
5. Unload TTS models.
6. Unload image diffusion models.
7. Unload prior CLIP/T5/Gemma encoders, VAEs, and video models.
8. Invoke the existing ComfyUI unload/free-memory path.
9. Confirm the GPU has returned to its normal ComfyUI-plus-desktop baseline before video loading begins.

Anything not actually unloaded must be counted in the following stage.

### Correct peak calculation

Do not use checkpoint size alone, and do not add every asset on disk together. Measure the maximum concurrent stage:

```text
total_device_usage_at_stage =
    desktop_driver_runtime_baseline
    + every model, tensor, cache, workspace, and latent resident in that stage

whole_path_peak = max(
    total_device_usage_during_prompt_encoding,
    total_device_usage_during_source_image_encoding,
    total_device_usage_during_source_audio_encoding_if_used,
    total_device_usage_during_stage_1_diffusion,
    total_device_usage_during_latent_upscale_if_used,
    total_device_usage_during_stage_2_refinement_if_used,
    total_device_usage_during_video_vae_decode,
    total_device_usage_during_audio_vae_decode_if_used
)
```

At each stage account for:

- model weights currently resident,
- GGUF dequantization/runtime buffers,
- LoRA patches active in that stage,
- text tokens and text-encoder activations,
- stored conditioning tensors,
- source image/audio tensors,
- video and audio latents,
- resolution and frame count,
- attention and sampler workspaces,
- VAE tile input/output buffers,
- preview buffers if enabled,
- framework caching/reserved memory,
- desktop/display/driver use,
- any model that failed to unload.

### Tokens, frames, and steps

- Prompt token length affects text-encoder activations and conditioning size and must be represented in the real test prompt.
- Resolution and frame count are major drivers of latent, attention, sampler, and decode memory.
- Sampling step count usually changes render time more than the instantaneous peak, but test the exact final step schedule because caches or graph behavior may change stage memory.
- Audio-in must include source-audio encoding and the frozen/conditioned audio latent during every stage where it remains resident.
- The no-audio-in 16GB graph must be measured as actually implemented; do not count an audio lane that the engine has deleted, and do not omit one that remains active.

### Measurement method

- Use external total-device reporting such as NVML for NVIDIA and the closest backend-equivalent tool elsewhere.
- Do not add a VRAM eligibility subsystem merely to collect the measurements.
- Existing logging may be used; otherwise keep measurement outside the runtime engine.
- Test after the real preceding OTR stages, not only from an isolated clean ComfyUI launch.
- Run one cold job and at least three consecutive warm jobs.
- Record total-device peak VRAM, system RAM, committed/pagefile use, render time, decode time, output validity, and memory after cleanup.
- Testing results belong in documentation and the chosen preset JSONs; they do not become code gates.

---

## 9. Clip sizing and beat coverage

Do not hard-code “8GB equals 3–5 seconds.” Determine each final JSON’s resolution, legal frame count, FPS, and step schedule through testing.

Use a practical progression while tuning:

| Pass | Resolution | Frames | Goal |
| --- | ---: | ---: | --- |
| A | 512x288 | 49 | Establish a valid complete render |
| B | 640x384 | 49 | Improve practical quality |
| C | 640x384 | 81 | Test longer coverage |

Adjust to each model’s legal dimensions and frame-count rules.

Short generated clips may loop or boomerang, but rendered/looped coverage must span the full beat window. Do not freeze-hold a single short clip for the unused remainder.

---

## 10. Workflow JSON strategy

### During implementation

- Keep `workflows/otr_canonical.json` valid and additive.
- Add all four final selectable names to the `OTR_VideoDirector` menu.
- Preserve existing engine IDs and saved-workflow compatibility.
- Do not add a profile framework, automatic GPU chooser, or conditional menu.
- The current default may remain unchanged while tuning; do not spend work on automatic default promotion.

### After the operator finishes testing

Save separate, ordinary workflow JSON files with the desired engine already selected. These are presets, not gated profiles:

- `workflows/otr_8gb_ltx.json`
- `workflows/otr_8gb_wan.json`
- `workflows/otr_16gb_ltx_audio_in.json`
- `workflows/otr_16gb_ltx_video.json`

Each JSON should contain the final tested resolution, frame count, FPS, step schedule, decoder settings, and selected engine for that route. The engine names are already final, so no later rename pass is needed.

---

## 11. Required code changes

1. Add the exact LTX 0.9.8 distilled checkpoint and dependency entries to the asset manifest.
2. Build `eng_ltx_8gb` using the correct 0.9.8 distilled graph, not the LTX-2.3 graph.
3. Register `ltx_8gb` as a normal row with `requires_flag=None` and ordinary asset preflight.
4. Register `wan_8gb` as a normal un-gated alias of the existing Wan adapter with `requires_flag=None`.
5. Preserve `wan_ti2v`, `ltx_video`, and `ltx_audio_in` as legacy aliases.
6. Add `ltx23_16gb_audio_in` and `ltx23_16gb_video` as final user-facing aliases only; do not fork or rebuild their engines.
7. Keep the existing 16GB x2 latent-upscale stage active.
8. Keep both 8GB routes single-pass/no-upscale.
9. Add all four final rows to `otr_canonical.json` in the same green change as the registry entries.
10. Reuse existing output publication, beat-fill, audio mux, and model cleanup paths.
11. Do not add GPU-memory detection, VRAM thresholds, compatibility flags, auto-fallback logic, profile objects, or warning scaffolding.
12. Do not alter HuMo.

---

## 12. Validation and commit order

1. Verify exact local/downloaded assets and node dependencies for LTX 0.9.8.
2. Run a standalone LTX 0.9.8 functional smoke on the RTX 5080.
3. Implement and register `ltx_8gb`.
4. Register `wan_8gb` against the existing adapter without a runtime flag.
5. Wire both 8GB rows into the canonical JSON.
6. Add the two final 16GB display aliases without changing engine code.
7. Verify the existing 16GB internal x2 upscaler is still connected and used.
8. Run `OTR_WorkflowValidator`, JSON round-trip, link audit, widget audit, AST parse, no-BOM check, regression suite, and Bug Bible.
9. Run the complete OTR leg for each route after actual LLM/TTS/image stages and confirmed cleanup.
10. Tune until each chosen preset works reliably on the operator’s target hardware.
11. Save the four final workflow JSON presets with their final parameters.
12. Commit and push each green chunk; verify local `HEAD` equals `origin/v2.0-alpha`.

---

## 13. Explicit non-goals

Do **not** implement any of the following in this sprint:

- “experimental” or “candidate” labels,
- later support-status renaming,
- VRAM eligibility gates,
- auto-detection that hides or disables menu rows,
- automatic model downgrade or fallback,
- new feature flags for these four final rows,
- GPU vendor/architecture whitelists,
- a new low-VRAM profile system,
- a new upscaler-bank scaffold,
- new 8GB upscaling,
- FP8/NVFP4/Q8 optimization forks,
- HuMo changes,
- deletion of legacy engine aliases.

---

## 14. Definition of done

The build is complete when:

- all four final names appear as normal selectable rows,
- neither 8GB row is gated by a feature flag or VRAM check,
- missing assets produce clear normal preflight errors,
- legacy IDs still load existing workflows,
- `ltx_8gb` uses the correct 0.9.8 distilled recipe,
- `wan_8gb` reuses the existing Wan implementation,
- the 16GB engines remain behaviorally unchanged,
- the 16GB no-audio-in route still uses its internal x2 latent upscaler,
- the 8GB routes remain single-pass/no-upscale,
- the complete OTR route unloads prior GPU-heavy stages before video,
- the operator can tune and run every row without code blocking the attempt,
- separate final JSON presets are saved after testing,
- all validation checks pass,
- each green chunk is committed and pushed to `v2.0-alpha`.
