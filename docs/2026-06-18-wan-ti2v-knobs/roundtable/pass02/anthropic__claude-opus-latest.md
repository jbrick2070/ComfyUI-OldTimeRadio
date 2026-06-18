<!-- requested_model: ~anthropic/claude-opus-latest | resolved_model: anthropic/claude-4.8-opus-20260528 -->

VERDICT: yes-with-fixes -- the VRAM direction is sound, but three wiring changes (tiled VAE, GGUF CLIP, sampler whitelist) are described as one-line swaps when the grounded graph code requires node-class + input-schema changes, and one will self-reject the engine.

MUST-FIX BEFORE BUILD:
1. [P2 Sampler] The proposed whitelist (drop `uni_pc`) collides with the engine's own default. In `_build_graph` the grounded default is `sampler = os.environ.get("OTR_WAN_TI2V_SAMPLER", "uni_pc")`. If `assert_usable` whitelists and rejects `uni_pc` while the unset default is `uni_pc`, the floor fails CLOSED with no env set. Fix: change the `_build_graph` default to `"euler"` AND have `assert_usable` read the same default value, so an unset env passes the whitelist.

2. [P1 Tiled VAE] "Add `VAEDecodeTiled` as a `vaedecode` candidate (env-gated)" does not work as written. (a) `resolve_graph_classes` picks the first *available* candidate; `VAEDecodeTiled` is a core node and always available, so candidate ordering cannot env-gate it -- you need a conditional candidate list keyed on the gating env. (b) `_build_graph` wires `vaedecode` inputs as only `{samples, vae}`; `VAEDecodeTiled` takes additional widget inputs (tile_size/overlap/temporal). Fix: build the candidate list conditionally on a new env (e.g. `OTR_WAN_TI2V_TILED_VAE`) and add the tile inputs in `_build_graph` when tiled is selected. verify: whether `wrapper_bridge.run_graph` auto-fills node default widget values for inputs you omit -- if it does not, an omitted `tile_size` will hard-error.

3. [P2 CLIP] "MOVE OFF fp8 -> GGUF umt5" is more than the `_loader_names()["clip"]` basename change implied. The grounded `clip` node candidate is `("CLIPLoader",)` with inputs `{clip_name, type:"wan", device:"default"}`. A GGUF umt5 needs `CLIPLoaderGGUF`, a different node with a different input schema. Fix: add `CLIPLoaderGGUF` to the `clip` candidate tuple and supply its correct inputs, OR take the fp16 safetensors path (stays on `CLIPLoader`). verify: `CLIPLoaderGGUF` input schema (does it accept the `type="wan"`/`device` args?).

4. [P1 Frames] Lowering `_TI2V_MIN_FRAMES` 33->17 alone will NOT yield a 17-frame floor. The grounded fallback is `plan["target_frame_count"] or self.target_fps` with `target_fps=25`; 25 >= 17 so it stays 25, not 17. The plan's own `_TI2V_DEFAULT_FRAMES` addition is the actual fix and must land in the same change -- list it as required, not optional.

SHOULD-FIX:
1. [P1 probe] The grounded probe is `VramPeakProbe(interval_s=1.0)`, not 0.1 -- the "0.7s sample" framing is inconsistent with the code. Make the interval change explicit (1.0 -> 0.1) and confirm the added sampling overhead is acceptable for a long render.
2. [P3 OOM] `assert_peak_within_ceiling` runs AFTER `run_graph` completes (non-test path). The plan's complaint about "render-then-assert" is correct; a pre-flight frame/size estimate *before* `run_graph` is the cheaper guard. A real CUDA/MPS OOM will still throw inside `run_graph` before the assert ever runs, so the retry path also needs to wrap the `run_graph` call, not the assert.
3. [P1 offload] `--lowvram`/sequential offload is a ComfyUI *server launch* flag; this engine is `in_process` and cannot set it per-render. "Require" it can only mean operator documentation -- state that, don't imply the engine controls it.
4. [General] The plan repeatedly says "floor tier" / "bigger cards can override up," but the grounded code is a single `WanTi2vEngine` class with no tier concept. Define the tier-selection mechanism (env? host_caps in `assert_usable`?) or the "env-gated default-on for the floor tier" language has nothing to bind to.

OPTIONAL / NICE-TO-HAVE:
- shift 3.0 vs 5.0 A/B is cheap; keep but it won't move VRAM, so don't let it gate the floor decision.

CUT THESE (over-engineering):
1. [P3 CFG/steps distill guard] The Lightning/distill LoRA is explicitly CUT from the floor; a guard for "if a distill is ever configured" protects a path that cannot exist in this tier. Safe to cut -- re-add it with the distill tier.
2. [P3 License guard for LoRA] No LoRA path exists in the floor graph (grounded `_node_candidates` has no LoRA node). Guarding a non-existent input is dead code. Safe to cut until a LoRA is actually added.
3. [P3] Doing BOTH pre-flight estimate AND catch+retry-at-lower-length is redundant for a "low floor" goal -- pick the pre-flight estimate (predictable, no wasted partial render). Safe to cut the retry loop.

[ASSUMPTION] `VAEDecodeTiled` is a core ComfyUI node and emits IMAGE at output index 0 like `VAEDecode` (so `results[self._TERMINAL][0]` still holds) -- verify against `/object_info`.
[ASSUMPTION] The "832x480 / 13.1 GB" baseline numbers are not in the grounding excerpt; `_dims(request)` is not shown -- verify the floor actually resolves to 480p.