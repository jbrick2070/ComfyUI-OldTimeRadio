<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan identifies the right blockers, but it is not build-ready because several required changes are underspecified against the actual graph code, and current code paths still pass Mac/8GB-broken defaults.

MUST-FIX BEFORE BUILD:
1. [P2 -- CLIP portability] Current code still defaults to the Mac-broken fp8 CLIP: `_loader_names()["clip"]` defaults to `umt5_xxl_fp8_e4m3fn_scaled.safetensors`, and `_node_candidates()["clip"]` is only `("CLIPLoader",)`. That contradicts P2’s “MOVE OFF fp8” requirement. Concrete fix: change the floor default CLIP basename to the chosen GGUF or fp16 umt5 file; add loader selection for CLIP, e.g. infer `.gguf` and use `CLIPLoaderGGUF`, otherwise `CLIPLoader`; update `_aux_loader_files()` categories if GGUF CLIP is stored elsewhere. verify: `CLIPLoaderGGUF` node input names and output shape from `/object_info`.

2. [P1 -- tiled VAE decode] “Add `VAEDecodeTiled` as a `vaedecode` node candidate” is not enough. Current `_build_graph()` always sends only `{"samples": ..., "vae": ...}` to the `vaedecode` alias. If `VAEDecodeTiled` has required tile/overlap inputs, the graph will fail at runtime. Concrete fix: add an explicit tiled decode mode/env, resolve the actual selected class, and build the correct input dict for `VAEDecodeTiled`; fail closed for the 8GB floor if tiled decode is required but the node is absent. verify: `VAEDecodeTiled` required inputs from ComfyUI `/object_info`.

3. [P1 -- frame default] The current render path still misuses FPS as frame count fallback: `plan["target_frame_count"] or self.target_fps` in `render_clip()`. Lowering `_TI2V_MIN_FRAMES` alone will not force the 8GB floor to 17 frames; an omitted frame count currently becomes 25 and quantizes as a valid 4n+1 count. Concrete fix: add `_TI2V_DEFAULT_FRAMES = 17`, change the fallback to `plan["target_frame_count"] or _TI2V_DEFAULT_FRAMES`, and change `_TI2V_MIN_FRAMES` from 33 to 17.

4. [P1 -- request override can still OOM floor] The plan lowers the floor/default frame count but does not prevent upstream `target_frame_count` from requesting 33+ frames. Current code honors `plan["target_frame_count"]` before the fallback and caps only at `_TI2V_MAX_FRAMES = 177`, so a floor-tier request can still hit the known 13GB-ish path. Concrete fix: for the 8GB floor, clamp effective length to `_TI2V_DEFAULT_FRAMES` unless an explicit higher-tier override/env is set and memory capability allows it.

5. [P2 -- sampler portability] Current code still defaults to `sampler = os.environ.get("OTR_WAN_TI2V_SAMPLER", "uni_pc")`, while P2 says the floor must be core `euler` + `simple` and fail-closed on nonportable values. Concrete fix: change the default sampler to `euler`, keep scheduler default `simple`, and validate `OTR_WAN_TI2V_SAMPLER`/`OTR_WAN_TI2V_SCHEDULER` in `assert_usable()` against an explicit whitelist before any render.

6. [P2/P3 -- env validation gap] `steps`, `cfg`, and `shift` are parsed inside `_build_graph()` with raw `int()`/`float()`. Bad env values will crash during render, not fail closed in `assert_usable()`. Concrete fix: parse and range-check `OTR_WAN_TI2V_STEPS`, `OTR_WAN_TI2V_CFG`, `OTR_WAN_TI2V_SHIFT`, sampler, and scheduler in one config resolver used by both `assert_usable()` and `_build_graph()`.

7. [P2 -- VAE guard incomplete] Current `assert_usable()` only rejects an empty VAE basename or exact `wan_2.1_vae.safetensors`. Any other wrong-but-present VAE basename passes, despite the header saying Wan 2.2 VAE is required. Concrete fix: fail closed unless the resolved VAE basename is the approved Wan 2.2 VAE name/list, e.g. `wan2.2_vae.safetensors`, with a deliberately named unsafe override only if needed.

8. [P1 -- low-VRAM offload] “Document/require ComfyUI `--lowvram` / sequential offload” is not an implementation. This engine runs in-process and the shown code does not set or verify ComfyUI low-VRAM mode. Concrete fix: define the exact runtime mechanism used by this launcher/process to enable low-VRAM/sequential offload, expose it in config, and have `assert_usable()` or startup logging fail/warn when the 8GB floor is enabled without it. [ASSUMPTION] This depends on how `wrapper_bridge` initializes ComfyUI.

9. [P1 -- peak probe] Current code uses `_MC.VramPeakProbe(interval_s=1.0)`, not the required `0.1`. Also the render comment explicitly calls it an “NVML peak probe”, which may not measure Mac MPS or AMD correctly. Concrete fix: change the interval to `0.1` for the floor measurement path and define backend-specific measurement for MPS/ROCm, or explicitly require external measurement for those platforms before claiming floor-fit. verify: `_MC.VramPeakProbe` supports non-NVIDIA backends.

10. [P3 -- OOM behavior] The plan says “pre-flight estimate or catch+retry” but does not specify the retry policy or cleanup ordering. Current `render_clip()` does no OOM catch/retry; it renders then asserts peak after completion. Concrete fix: implement one deterministic path: preflight reject/clamp before graph execution, or catch CUDA/ROCm/MPS OOM, unload/free Comfy objects/cache, retry once at the floor length/settings, and then fail with a named `EngineUnusable`/graph error if still OOM. Do not rely on post-render `assert_peak_within_ceiling()` as the first guard.

11. [P2 -- GGUF CLIP dependency sequencing] P2 moves CLIP to `CLIPLoaderGGUF`, but the plan only says it is “same dep family as `UnetLoaderGGUF`”. Current node resolution happens in `load()` via `_wb.resolve_graph_classes(self._node_candidates())`; if `CLIPLoaderGGUF` is missing, this will fail at load time, not as a clear usability failure. Concrete fix: include GGUF CLIP node availability in `assert_usable()` or make `load()` raise a named, actionable error that tells the operator to install ComfyUI-GGUF.

12. [P1/P2 -- explicit model path hidden dependency] Current `_installed()` accepts `OTR_WAN_TI2V_CKPT` if `os.path.exists(self._ckpt_path())`, but `_build_graph()` passes only `names["unet"]` to the Comfy loader, normally the basename. If the explicit ckpt path is outside ComfyUI model search paths, assert/load can pass while the loader cannot find the model. Concrete fix: either require the path’s directory to be registered with Comfy/folder_paths, pass an absolute path only if the node supports it, or make `_installed()` use the same resolution mechanism the graph loader will use. verify: whether `UnetLoaderGGUF`/`UNETLoader` accepts absolute paths in `unet_name`.

SHOULD-FIX:
1. [P1 -- acceptance threshold] “target measured peak < 8GB” is too loose for real 8GB cards, especially display GPUs and unified-memory Macs. Concrete fix: target the engine’s actual usable ceiling, e.g. `dynamic_vram_ceiling_mb()` minus safety margin, not nominal 8192 MB.

2. [A/B] The A/B “current baseline” is not fully defined after P2 because “current” currently includes fp8 CLIP in code, while the intended post-P2 floor does not. Concrete fix: define A and E as complete env/config bundles, including CLIP loader/name, sampler, scheduler, shift, frame count, tiled decode flag, lowvram flag, and resolution.

3. [P2 -- shift] “test 3.0 vs 5.0; pick the steadier” is not build-spec language. Concrete fix: either set the floor default now, or define the exact acceptance check and who records the decision before merge.

4. [P3 -- CFG/distill guard] The CFG/steps guard is phrased around a future distill/LoRA path that is explicitly out of scope. If retained, define the actual config signal that means “distill is configured”; otherwise there is nothing current code can validate.

5. [P1 -- tiled decode fallback semantics] The plan says tiled decode is “env-gated, default-on for the floor tier” but does not say what happens when the node is absent. Concrete fix: for floor tier, fail closed if absent; for non-floor, allow fallback to `VAEDecode`.

6. [P2 -- Mac/AMD verification] The plan says “STILL VERIFY-AT-BUILD” for GGUF on MPS/ROCm/DirectX. Make this a release gate with exact smoke command, model set, and expected failure classification, not an informal note.

7. [P1 -- encoder freeing claim] The plan says “the engine already frees the encoder before decode.” Current code uses `free_after_use=True` and `keep={"unet", "vae", self._TERMINAL}`; CLIP is not kept. That likely intends to free CLIP, but it depends on `wrapper_bridge.run_graph` behavior. verify: `run_graph` actually releases CLIP loader/model after text encoding and before VAE decode.

OPTIONAL / NICE-TO-HAVE:
1. Add a single resolved-config dump at render start: effective UNET/CLIP/VAE names, loaders, length, size, sampler, scheduler, shift, steps, cfg, tiled decode on/off, lowvram on/off.
2. Add a tiny “floor smoke” test mode that builds the graph and validates node/input schemas without loading full weights.
3. Add separate model manifest entries for the GGUF CLIP and fp16 fallback, including license and expected file size/checksum.

CUT THESE (over-engineering):
1. [P3 -- LoRA license guard] Cut from the floor build. The plan explicitly cuts Lightning/distill LoRA out of scope, and the shown engine has no LoRA path. Add the license guard when a LoRA path actually exists.

2. [P3 -- CFG/distill coupling guard] Cut or defer unless a distill/LoRA config is introduced in this build. Without a current distill signal, this becomes speculative validation code.

3. [A/B -- expected-OOM baseline run] Cut the mandatory 33-frame baseline on real 8GB hardware. It is expected to OOM and does not prove the floor works. Run the hardened E config for fit; run the 33-frame baseline only on larger hardware or as a controlled negative test.

4. [P2 -- shift bake-off] Cut from the portability/VRAM floor gate. Shift 3.0 vs 5.0 is a quality/stability tuning choice, not the blocker for 8GB fit or Mac/AMD portability.