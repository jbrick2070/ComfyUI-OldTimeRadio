<!-- requested_model: ~anthropic/claude-sonnet-latest | resolved_model: anthropic/claude-4.6-sonnet-20260217 -->

VERDICT: yes-with-fixes. Four of the eight checklist items cannot be fully implemented as written because required input schemas and a key mechanism are undefined; the fp8 CLIP default is the active bug and remains un-named in the constant the checklist creates.

---

## MUST-FIX BEFORE BUILD

**1. [Build checklist, item 1 + _build_graph] `CLIPLoaderGGUF` input dict is unknown and the fix cannot be written without it.**
The current `_build_graph` passes `{"clip_name": ..., "type": "wan", "device": "default"}` to `CLIPLoader`. `CLIPLoaderGGUF` (ComfyUI-GGUF pattern) almost certainly does NOT accept `"type"` or `"device"` — passing them will raise a graph-validation or key error at runtime. The plan says "reconcile the clip inputs dict per loader" but defers this to "Verify-at-build." That is backwards: the conditional branch in `_build_graph` (`if self._clip_loader_mode() == "gguf": clip_inputs = {...} else: clip_inputs = {...}`) cannot be written until the actual `CLIPLoaderGGUF` input schema is known from `/object_info`. **Fix:** Pull `/object_info` for `CLIPLoaderGGUF` before writing a line of code for item 1; write the branch with the confirmed input set, just as the plan already does correctly for the UNET `unet_inputs` branch.

**2. [Build checklist, item 2 + _build_graph] `VAEDecodeTiled` input names and concrete values are unspecified.**
The method section says "tile ~256 / overlap" with no overlap value; the checklist says "add tile_size/overlap to the vaedecode inputs." Neither the parameter names (`tile_size`? `tile_width`/`tile_height`? `overlap`? `overlap_factor`?) nor their allowed ranges appear in the grounding. An incorrect name silently falls through if the bridge ignores unknown inputs, or crashes if it validates them. The output-index claim ("IMAGE output index 0") is also asserted without confirmation. **Fix:** Capture `VAEDecodeTiled` from `/object_info` before writing item 2; write the concrete dict with verified names and a named constant pair (`_TI2V_TILE_SIZE = 256`, `_TI2V_TILE_OVERLAP = <confirmed>`).

**3. [Build checklist, item 3] "Clamp the floor tier so an upstream `target_frame_count` can't push 33+" has no implementation.**
After the fix, `quantize_frames_4n1(upstream_value, min_frames=17, max_frames=177)` will faithfully produce 81 or 121 frames if the upstream request says so — the existing `max_frames=177` constant is unchanged and unchanged code is not protection. "An explicit higher-tier override" is never defined: no env var, no cap constant, no registry flag. This gap defeats the stated 8GB VRAM floor goal. **Fix:** Define `_TI2V_FLOOR_MAX_FRAMES = 17` (or a named env `OTR_WAN_TI2V_MAX_FRAMES` defaulting to 17), clamp `length` to it in `render_clip` before passing to `_build_graph`, and document how a caller signals they want more frames (e.g., they set the env explicitly to a higher number or use a different tier).

**4. [Build checklist, item 1 + _loader_names] The active fp8 bug is the default CLIP filename; the checklist never names the replacement constant.**
`_loader_names` defaults CLIP to `"umt5_xxl_fp8_e4m3fn_scaled.safetensors"` — the exact fp8 file that throws `Float8_e4m3fn TypeError` on MPS (ComfyUI #9255), which is the whole reason item 1 exists. The method section names the target as `umt5-xxl-encoder-Q5_K_M.gguf`, but the checklist only says "default basename -> the GGUF umt5" without naming the constant or the file. Without an explicit `_TI2V_DEFAULT_CLIP = "umt5-xxl-encoder-Q5_K_M.gguf"` constant mirroring `_TI2V_DEFAULT_UNET`, a developer editing `_loader_names` has no canonical source of truth. **Fix:** Add `_TI2V_DEFAULT_CLIP = "umt5-xxl-encoder-Q5_K_M.gguf"` at module level (beside `_TI2V_DEFAULT_UNET`); change the `_loader_names` default for CLIP from the fp8 string to `_TI2V_DEFAULT_CLIP`.

---

## SHOULD-FIX

**5. [Build checklist, item 4] Scheduler is not whitelisted; only the sampler is.**
The method section specifies `euler`/`simple` as the cross-platform pair. Item 4 adds `_PORTABLE_SAMPLERS` and validates `OTR_WAN_TI2V_SAMPLER`, but `OTR_WAN_TI2V_SCHEDULER` is unchecked: `scheduler = os.environ.get("OTR_WAN_TI2V_SCHEDULER", "simple")` passes through to `_build_graph` unconstrained. A caller setting `OTR_WAN_TI2V_SCHEDULER=karras` will silently get untested cross-platform behavior. **Fix:** Either add a `_PORTABLE_SCHEDULERS = frozenset({"simple"})` with the same validation pattern, or explicitly document that the scheduler is intentionally not restricted.

**6. [Build checklist, item 5] Config resolver error-handling path is not specified — specifically which exception type and where.**
Item 5 says "fail-closed on a missing node / invalid values," but a raw `ValueError` from `int("abc")` inside `_build_graph` would propagate as an uncaught exception, not an `EngineUnusable`, bypassing the standard error-reporting path. The config resolver should catch `ValueError` and raise `EngineUnusable` with a concrete message. If the resolver is called from `assert_usable`, this happens at usability-check time (correct); if only from `_build_graph`, it surfaces mid-render (incorrect). **Fix:** Specify that the resolver raises `EngineUnusable` and is called from `assert_usable`, not just `_build_graph`.

**7. [Build checklist, item 1] `CLIPLoaderGGUF` availability check in `assert_usable` — mechanism is unspecified.**
The plan says "Add `CLIPLoaderGGUF` availability to `assert_usable` (named 'install ComfyUI-GGUF' error)" but gives no implementation path. `wrapper_bridge.resolve_graph_classes` already does the availability check at `load()` time, which is too late. Verify whether `wrapper_bridge` exposes a `node_class_available(name)` predicate; if not, the check must go through ComfyUI's `NODE_CLASS_MAPPINGS`. **Fix:** Confirm the availability-check API (verify: `wrapper_bridge.node_class_available` or equivalent) and specify its call site in `assert_usable` gated on `_clip_loader_mode() == "gguf"`.

**8. [Build checklist, item 1 + _aux_loader_files] `_aux_loader_files` folder-paths categories for the GGUF CLIP are unverified.**
`_aux_loader_files` uses `("text_encoders", "clip")` as `folder_paths` lookup categories for the CLIP. After the default changes to a `.gguf` file, `_resolve_model_file` must find it in those same directories. If ComfyUI-GGUF registers or resolves CLIP GGUFs under a different directory key (e.g., a custom `clip_gguf` category), `_missing_loaders()` will incorrectly report the file as missing and `assert_usable` will fail even when the file is present. **Fix:** Verify from the ComfyUI-GGUF source which `folder_paths` category `CLIPLoaderGGUF` uses for resolution; update the tuple in `_aux_loader_files` if it differs from `("text_encoders", "clip")`.

---

## OPTIONAL / NICE-TO-HAVE

- Rename `OTR_WAN_TI2V_LOADER` → `OTR_WAN_TI2V_UNET_LOADER` to match parallel `OTR_WAN_TI2V_CLIP_LOADER`; currently asymmetric. Breaking change, so document in migration notes if done.
- The M8 VAE guard (item 6) currently checks "not empty AND not 2.1-VAE." The plan wants the stricter positive check ("IS the 2.2 VAE name"). Given the docstring already says this, the positive-check form is both more expressive and closes unknown-name-bypass silently. Low risk to add.
- `_TI2V_TILE_SIZE` and `_TI2V_TILE_OVERLAP` as named module-level constants (rather than magic numbers in `_build_graph`) would make them env-overridable later.

---

## CUT THESE

**1. [Build checklist, item 7] VramPeakProbe interval 1.0→0.1 — safe to defer.**
The acceptance criterion is "measured peak < ceiling." Whether the probe fires every 1.0s or 0.1s does not change correctness: if a transient spike is below 0.1s duration, neither interval catches it. A 10× finer interval adds 10× the probe-thread wake-ups during decode (which shares the GIL window). It is not wrong, but it solves no stated failure mode and can be done post-floor without gating the build. Cut for now; revisit if the acceptance test misses a known spike.