<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The plan still leaves required node schemas and floor limits ambiguous, and has contradictions that can produce a build that passes default review but fails at load/render time.

MUST-FIX BEFORE BUILD:
1. [Build checklist 1 / Verify-at-build] `CLIPLoaderGGUF` is required by the new default, but its input schema is explicitly deferred to “Verify-at-build”. That is not build-ready: `_build_graph` cannot safely emit the `clip` node until the accepted inputs are known. Concrete fix: before implementation, record the actual `/object_info` schema for `CLIPLoaderGGUF` and specify the exact input dict for both modes. Keep the current `CLIPLoader` inputs only for safetensors if verified; use a separate GGUF input dict if `type` / `device` are not accepted.

2. [Build checklist 2 / Verify-at-build] `VAEDecodeTiled` is made default/fail-closed, but its required inputs are also deferred to “Verify-at-build”. The current graph only has `samples` and `vae`; the plan says add `tile_size/overlap` but does not define exact field names, defaults, or whether additional required video/temporal fields exist. Concrete fix: capture `/object_info` for `VAEDecodeTiled` first, then specify the exact `vaedecode` inputs and output index. Do not make it the default until the graph is schema-correct.

3. [Build checklist 4 / The method] The method says sampler/scheduler floor is `euler` / `simple`, but the checklist only whitelists sampler. `OTR_WAN_TI2V_SCHEDULER` would still accept any raw string unless the resolver rejects it. Concrete fix: add `_PORTABLE_SCHEDULERS = frozenset({"simple"})`; validate scheduler in the shared resolver and in `assert_usable`; use the same default `"simple"` in both `assert_usable` and `_build_graph`.

4. [Build checklist 5] “parse+range-checks steps/cfg/shift/sampler/scheduler” is underspecified. No ranges are given, so implementation and tests will guess. Concrete fix: define exact accepted ranges/defaults in the spec, e.g. steps default `30` with min/max, cfg default `5.0` with min/max, shift default `5.0` with min/max, sampler set `{"euler"}`, scheduler set `{"simple"}`. Invalid env values must raise `EngineUnusable` from `assert_usable`, not crash via `int()` / `float()` in `_build_graph`.

5. [Build checklist 3 / Acceptance] The floor frame clamp is not concrete. Current code has `_TI2V_MAX_FRAMES = 177`; the checklist says upstream `target_frame_count` cannot push `33+` “without an explicit higher-tier override” but does not define the cap or override mechanism. Concrete fix: add an explicit floor max, e.g. `_TI2V_DEFAULT_FRAMES = 17` and `_TI2V_FLOOR_MAX_FRAMES = 17`; clamp default floor renders to 17 regardless of upstream `target_frame_count`; add a named opt-in env for higher tiers if required, with validation that the value remains `4n+1` or is passed through `quantize_frames_4n1` intentionally.

6. [Build checklist 1] `CLIPLoaderGGUF` availability in `assert_usable` must be conditional on CLIP loader mode. As written, “Add `CLIPLoaderGGUF` availability to `assert_usable`” can make the safetensors fallback unusable on installs without ComfyUI-GGUF. Concrete fix: if `_clip_loader_mode() == "gguf"`, require `CLIPLoaderGGUF`; if `"safetensors"`, require `CLIPLoader`. Emit the “install ComfyUI-GGUF” error only for GGUF mode.

7. [Build checklist 1] The same extension availability problem already exists for the default UNET path: `_node_candidates()` defaults to `UnetLoaderGGUF`, but `assert_usable()` currently only checks files. If ComfyUI-GGUF is missing, usability can pass and `load()` fails later. Concrete fix: make `assert_usable()` validate required node classes for the resolved modes, at least `UnetLoaderGGUF` for GGUF UNET and `CLIPLoaderGGUF` for GGUF CLIP. [ASSUMPTION] This depends on `wrapper_bridge` exposing a safe/light way to resolve or inspect node classes; if not, add one or explicitly keep this as a load-time failure.

8. [The method / Build checklist 1] The safetensors CLIP fallback selection is brittle. The checklist says `OTR_WAN_TI2V_CLIP_LOADER` defaults to `gguf`; if an operator only sets `OTR_WAN_TI2V_CLIP_NAME=...safetensors`, the graph will still choose `CLIPLoaderGGUF` unless `_clip_loader_mode()` infers from the basename. Concrete fix: implement `_clip_loader_mode()` as env override if valid, otherwise infer from `OTR_WAN_TI2V_CLIP_NAME` extension, defaulting to GGUF only when no safetensors basename is selected.

9. [Acceptance vs Build checklist 4] “keep every knob env-overridable so bigger cards tune up” conflicts with fail-closed sampler whitelist `{"euler"}`. Concrete fix: either remove that acceptance sentence for sampler/scheduler, or define an explicit higher-tier override env that bypasses the portable whitelist. Do not leave both requirements active.

SHOULD-FIX:
1. [Build checklist 2] Tile size/overlap should be env-configurable if the goal is system-agnostic 8GB/MPS/AMD. Hard-coding “~256 / overlap” gives no way to recover from backend-specific tiling failures. Concrete fix: add `OTR_WAN_TI2V_VAE_TILE_SIZE` and `OTR_WAN_TI2V_VAE_OVERLAP`, with validated integer ranges and defaults.

2. [Build checklist 3] Changing `_TI2V_MIN_FRAMES` from 33 to 17 is not sufficient by itself because current `render_clip` uses `plan["target_frame_count"] or self.target_fps`; `target_fps` is 25. The checklist includes the default change, but tests must prove unset target resolves to 17, not 25. Concrete fix: add a unit test around `render_clip` length selection or a factored frame resolver.

3. [Build checklist 6] The VAE guard says “approved Wan2.2 name” but should specify exact comparison behavior. Concrete fix: define the approved lowercase basename set, currently apparently `{"wan2.2_vae.safetensors"}`, and reject everything else with a named message.

4. [Build checklist 7] “measurement path” is ambiguous. Current code probes every render at `interval_s=1.0`. Concrete fix: state whether production renders should always use `0.1`, or only the FIT/acceptance test should. If only FIT, do not change the render default; make the probe interval env/test-configurable.

5. [The method / Offload] “operator runs the ComfyUI server with `--lowvram`” is misleading for this in-process engine. Concrete fix: document that the process hosting the in-process ComfyUI execution must be started/configured with low-VRAM behavior; if this is a sidecar/server-specific instruction, separate it from the in-process path. [ASSUMPTION] Exact startup mechanism depends on code not shown.

6. [Build checklist 1 / 5] Invalid loader mode envs are not covered. Current `_loader_mode()` silently falls back if `OTR_WAN_TI2V_LOADER` is invalid. Concrete fix: add loader modes to the shared config resolver or explicitly fail closed on invalid `OTR_WAN_TI2V_LOADER` / `OTR_WAN_TI2V_CLIP_LOADER`.

7. [Build checklist + “tests + doc”] The plan says tests are allowed but does not list required test cases. Concrete fix: add tests for GGUF CLIP default, safetensors CLIP fallback, tiled/untiled VAE node selection, invalid sampler/scheduler rejection, invalid numeric env rejection, VAE basename rejection, and default frame count 17.

OPTIONAL / NICE-TO-HAVE:
- Add an assertion/log line showing resolved UNET loader mode, CLIP loader mode, VAE decode mode, frames, sampler, scheduler, steps/cfg/shift before graph execution.
- Add a doc table of all new env vars and defaults.

CUT THESE (over-engineering):
1. [Build checklist 7] Changing the production probe interval to `0.1` unconditionally is heavier than needed for normal renders. Safe cut: keep production at `1.0` and use `0.1` only in the FIT/acceptance measurement path, unless the ceiling assertion is explicitly intended to be high-resolution in production.

2. [Acceptance] “VERIFY-AT-BUILD on real Mac + AMD before claiming cross-platform floor” is valid release gating, but it should not block the code build once schema/unit issues are fixed. Safe cut from build checklist: keep it as release validation, not as an implementation task.