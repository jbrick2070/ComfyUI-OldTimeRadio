<!-- requested_model: ~anthropic/claude-sonnet-latest | resolved_model: anthropic/claude-4.6-sonnet-20260217 -->

VERDICT: no. The plan describes fixes that the grounding code does not implement; several implementation paths are underspecified or self-contradictory, and the stated acceptance criterion (measured peak < 8 GB) cannot be met with the mechanics the plan provides for the `vaedecode` and CLIP changes.

---

## MUST-FIX BEFORE BUILD

**1. [P2 / `_node_candidates` + `_loader_names`] CLIP loader class not switched — Mac fix is incomplete as described.**
The plan says "move off fp8 → `CLIPLoaderGGUF`." The grounding code has `"clip": ("CLIPLoader",)` in `_node_candidates` and `"umt5_xxl_fp8_e4m3fn_scaled.safetensors"` as the default in `_loader_names`. The plan never says to add a `_clip_loader_mode()` conditional analogous to the existing `_loader_mode()` for the UNET. Changing only the filename while leaving `CLIPLoader` in `_node_candidates` will fail at graph execution: `CLIPLoader` cannot load a `.gguf` file. The CLIP loader node class and its inputs (`clip_name`, `type`, `device`) differ from `CLIPLoaderGGUF`'s inputs. **Fix:** add a `_clip_loader_mode()` method gated on the CLIP filename extension; make `_node_candidates["clip"]` conditional; reconcile the `_build_graph` clip-node inputs dict for both paths.

**2. [P1 / `_node_candidates` + `_build_graph`] `VAEDecodeTiled` input schema not addressed.**
The plan says "Add `VAEDecodeTiled` as a `vaedecode` node candidate." The current `_build_graph` emits `{"samples": W("ksampler", 0), "vae": W("vae", 0)}` for that node. `VAEDecodeTiled` in ComfyUI takes additional parameters (`tile_size`, `overlap`). Verify: ComfyUI `/object_info` for `VAEDecodeTiled` — if those parameters are required (not defaulted by the server), the graph will error on execution. The plan does not name the tile_size or overlap values to use, nor the env vars that would control them. **Fix:** specify `tile_size` (e.g., 256 for 8 GB) and `overlap` in the plan and add them to the `vaedecode` inputs dict.

**3. [P1 / `render_clip` + module-level constants] `_TI2V_MIN_FRAMES` not changed; `target_fps` misuse not fixed.**
Grounding code: `_TI2V_MIN_FRAMES = 33`, fallback is `self.target_fps` (= 25). The call is `quantize_frames_4n1(25, min_frames=33, max_frames=177)`. Behavior of `quantize_frames_4n1` when the input is below `min_frames` is unspecified here ([ASSUMPTION] it clamps to 33, defeating the 17-frame target). The plan says add `_TI2V_DEFAULT_FRAMES` and lower the minimum to 17, but neither is done. On the 8 GB tier the fallback path produces 33 frames and OOMs — the exact failure the plan sets out to fix. **Fix:** set `_TI2V_MIN_FRAMES = 17`, add `_TI2V_DEFAULT_FRAMES = 17`, replace `or self.target_fps` with `or _TI2V_DEFAULT_FRAMES` in `render_clip`.

**4. [P2 / `_build_graph` + `assert_usable`] Default sampler is still `uni_pc`; whitelist not implemented.**
Plan states: "core `euler` + `simple` (rock-solid cross-platform). Add an `OTR_WAN_TI2V_SAMPLER` whitelist validated in `assert_usable`." Grounding: `sampler = os.environ.get("OTR_WAN_TI2V_SAMPLER", "uni_pc")`. `assert_usable` contains no sampler validation. A user who overrides the env var with `sa_solver` or `MoEKSampler` gets silent non-portable behavior; the plan explicitly says this must fail closed. **Fix:** Change default to `"euler"`, add a `_PORTABLE_SAMPLERS` frozenset, and validate `OTR_WAN_TI2V_SAMPLER` against it in `assert_usable` before the model-file checks.

**5. [P1 / `render_clip`] VRAM probe interval is 1.0 s in code; plan requires 0.1 s for the acceptance criterion.**
The plan explicitly says "Re-measure peak at `VramPeakProbe(interval_s=0.1)`" because the author already flagged that a 0.7 s sample may understate the true peak. The grounding code has `VramPeakProbe(interval_s=1.0)`. At 1.0 s the probe is less likely to catch short-duration decode peaks than the already-suspect 0.7 s sample. The stated acceptance criterion — "measured peak < 8 GB before claiming floor-fit" — cannot be trusted at 1.0 s. **Fix:** change to `interval_s=0.1` in `render_clip`.

---

## SHOULD-FIX

**1. [P1 / plan] `VAEDecodeTiled` env-gate is unnamed.**
Plan says "env-gated, default-on for the floor tier." No env var name is given, no fallback behavior is defined, and no location in the code is specified (should it affect `_node_candidates`? `assert_usable`?). Without this the implementer must guess. **Fix:** name the var (e.g., `OTR_WAN_TI2V_TILED_VAE`, default `"1"`), specify that it selects `VAEDecodeTiled` vs `VAEDecode` in `_node_candidates`, and list the tile_size default for 8 GB.

**2. [P3 / `assert_usable`] CFG/steps coupling guard is unimplement