<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: build-ready as-is? no. The plan still has executable contradictions and grounded code gaps around aspect handling, VRAM measurement, loader selection, and Phase-1/Phase-2 sequencing.

MUST-FIX BEFORE BUILD:
1. [Temperature / seeds / determinism + I/O contract; eng_wan_i2v.py] The plan says supplying `init_w`/`init_h` makes the aspect policy apply, but grounded code only computes `_aspect_plan()` and stores it in `plan`; `render_clip()` / `_build_graph()` never uses `plan["aspect_plan"]` before staging `LoadImage`. A non-landscape still can still be handed raw to `WanImageToVideo`. Concrete fix: before `_wb.stage_into_comfy_input()`, materialize a padded/cropped landscape image according to `_aspect_plan()` and stage that derived file; or explicitly verify the installed `WanImageToVideo` node performs cover/pad without stretch and remove the false engine-level guarantee.

2. [ComfyUI quirks & likely bugs; wrapper_bridge.py] The plan relies on `free_after_use=True` to free umt5 before the sampler, but grounded `wrapper_bridge._topo_order()` executes all initially-ready nodes alphabetically. For the current graph, ready nodes include `clip`, `loadimage`, `unet`, and `vae`, so the 14B UNET can be loaded before `pos`/`neg` consume and free `clip`. This can create exactly the TE+UNET co-residency the plan is trying to avoid. Concrete fix: make text encoding a separate pre-sampler phase, or change the graph executor to support explicit sequencing/priorities and schedule `CLIPLoader` + both `CLIPTextEncode`s + clip free before `UNETLoader`.

3. [Tasks 1; Hard rules; wrapper_bridge.py] The required “render-phase peak NVML” is not provided by the grounded engine. `render_clip()` only calls `_MC.assert_vram_within_ceiling("wan_i2v-render")` after `encode_frames_to_silent_mp4()`, which can miss the actual sampler peak. Concrete fix: add NVML polling around the render window, at minimum from first heavy model load through `VAEDecode`, and report both whole-run and render-phase peak separately. Do not treat the post-render instantaneous assert as the 14.5GB gate.

4. [ComfyUI quirks & likely bugs; Tasks 1/2; eng_wan_i2v.py] GGUF fallback and TI2V-5B cannot work through the existing engine graph as written: `_node_candidates()` only resolves `"UNETLoader"`, and `_build_graph()` always emits `"unet_name"` / `"weight_dtype"` for that loader. The plan says use `UnetLoaderGGUF` for the 14B fallback and TI2V GGUF, but does not specify the loader switch. Concrete fix: add an explicit loader mode/config path for GGUF, resolve `UnetLoaderGGUF`, emit its installed input names after TASK 0 signature verification, and cover both safetensors and GGUF branches in a fail-closed smoke.

5. [Tasks 2; Temperature / seeds / determinism + I/O contract] The 8GB “SECOND selectable Wan engine” is underspecified for build. There is only grounded `WanI2VEngine`; no TI2V engine contract is given: engine id, env flags, model/VAE env names, registry registration, required inputs, node candidates, canonical output contract, and profile-selection hook are missing. Concrete fix: define a separate engine, e.g. `wan_ti2v`, with its own flag/model/VAE checks, node graph, loader mode, canonicalize output, and dispatcher/profile selection tests before asking a coder to wire it.

6. [Tasks 1/2 vs Temperature / seeds / determinism + I/O contract] The document still mixes Phase 1 and Phase 2 assertions. Task 2 says “same asserts as task 1 (engine-in-trace...)”, while the later corrected section says Phase 1 bare `/prompt` has no trace and Phase 2 engine leg is where `final_engine` is asserted. Concrete fix: rewrite Tasks 1 and 2 into explicit substeps: Phase 1 bare `/prompt` for each model with no dispatcher/trace/audio assertions; Phase 2 real engine leg for each model with `final_engine`, silent mp4, NVML, and audio-spine assertions.

7. [Tasks 1; Grounded facts; eng_wan_i2v.py] Phase-1 `/prompt` smoke cannot rely on `OTR_WAN_I2V_CKPT` alone. Grounded `_build_graph()` passes the basename to `UNETLoader`; Comfy loader nodes normally consume model names relative to registered model folders, not arbitrary absolute paths. Concrete fix: in `scripts/otr_wan_smoke.py`, derive and pass the exact loader model name under `diffusion_models`, verify `_otr_headless_model_paths.yaml` exposes `C:\ComfyUI-Models\diffusion_models`, and fail before `/prompt` if the model name is not visible to Comfy.

8. [TASK 0; ComfyUI quirks & likely bugs; eng_wan_i2v.py] SageAttention isolation remains unresolved. The plan says choose disable-sage or confirm sidecar, but later says “staying in-process is safe.” Grounded `WanI2VEngine.resolve_isolation()` can escalate via `_MC.resolve_isolation(..., _MC.sageattention_patched())`. Concrete fix: make the build choice explicit before smoke. For the fast path, disable/uninstall/prevent import of SageAttention and assert `resolve_isolation()` returns in-process; otherwise provision and test the cu128 sidecar before any Wan render.

SHOULD-FIX:
1. [ComfyUI quirks & likely bugs; eng_wan_i2v.py] Update stale strings/docstrings that still say “install the Wan wrapper + KJNodes pin audit.” The plan now requires core Comfy Wan nodes, not KJ. Leaving these messages will send the coder/operator toward the dependency path the same section forbids.

2. [ComfyUI quirks & likely bugs] `ModelSamplingSD3` is called mandatory, but the plan does not specify how its installed signature is translated into graph inputs or where sigma shift values live. Concrete fix: after TASK 0, record exact input names and engine defaults for 14B and 5B; add it to `_node_candidates()` and `_build_graph()`.

3. [DECISION GATE Path B] The two-expert upgrade sequence does not state how the high-noise model is released before the low-noise model under a 14.5GB cap. Concrete fix: if Path B is attempted, design it as two sampler phases with explicit high-expert eviction before low-expert load, and measure peak across the handoff.

4. [Tasks 3/4] The eyeball gate output path is under `docs/2026-06-12-ltx23-motion/wan_clips/`, an LTX-named directory. If this is intentional, fine; otherwise use a Wan-specific path to avoid confusing artifacts and tracker references.

5. [Temperature / seeds / determinism + I/O contract] “Compare a frame hash + log any drift” is not a stable criterion if sub-pixel drift is allowed. Concrete fix: specify exact comparison: same seed/prompt/model metadata must match, output frame count/fps/dimensions must match, and perceptual/hash drift is logged with tolerance rather than pass/fail.

6. [Hard rules; Phase 2] [ASSUMPTION] If Phase 1 runs a resident Comfy server and Phase 2 invokes `render_clip()` in a different Python process, the resident server can invalidate VRAM measurements or double-load models. Verify whether `otr_run_leg.ps1` executes inside the same headless server process. If not, require killing/resetting the Phase-1 server before Phase 2.

OPTIONAL / NICE-TO-HAVE:
- Add the seed, model id, loader type, VAE name, node class names, steps/cfg/scheduler, and git commit to each generated clip sidecar JSON.
- Add a tiny preflight command that prints resolved Comfy node signatures and model visibility before any render.
- Store sha256/license records in a machine-readable manifest, not just tracker prose.

CUT THESE (over-engineering):
1. [The two engines; Tasks] Cut the optional Wan camera LoRA from the smoke scope. It is explicitly not on disk and not required to prove two selectable Wan engines.

2. [DECISION GATE Path B] Do not implement or fetch the high-noise 14B expert until Path A produces a clip and Jeffrey rejects the motion. The document already says this; keep it out of the first build chunk.

3. [Tasks 1/3] Make MAD optional-only or remove it from the first smoke. The plan says the gate is visual and MAD already misled the LTX evaluation; it should not block the Wan bring-up.

4. [TASK 0] If the chosen smoke route is “disable Sage and run core nodes in-process,” cut sidecar provisioning from the first build chunk. It is a separate compatibility task unless the in-process isolation assert fails.