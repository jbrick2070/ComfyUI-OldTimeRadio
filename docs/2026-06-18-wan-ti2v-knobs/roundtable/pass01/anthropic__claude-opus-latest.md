<!-- requested_model: ~anthropic/claude-opus-latest | resolved_model: anthropic/claude-4.8-opus-20260528 -->

VERDICT: no -- the stated 8GB floor is contradicted by the engine's own grounding (NVML peak 13.1 GB at 480p), and the headline lever (4-step LoRA) does not address VRAM, plus the LoRA candidates need graph changes that the spec treats as env-only.

MUST-FIX BEFORE BUILD:

1. [HARD CONSTRAINTS #1 + Current recipe "GPU smoke"] The grounding says the CURRENT 480p/33-frame render hits **NVML peak 13.1 GB**. `render_clip` calls `_MC.assert_peak_within_ceiling(render_peak, ...)` against `dynamic_vram_ceiling_mb()`. On an actual 8GB card that ceiling assert will fail-closed -- the "floor" does not run on the floor. Step count is a compute/latency lever, not a VRAM lever; going 30->4 steps (Candidate B/C) leaves the peak essentially unchanged because the peak is driven by model residency + the video VAE decode, not by iteration count. FIX: make the primary tuning target VRAM peak, not steps -- add a tiled VAE decode path (`VAEDecodeTiled` as a candidate for the `vaedecode` node) and/or cap frames, and re-measure peak on a memory-constrained config before any quality A/B. Until peak < ~8GB is demonstrated, no candidate satisfies constraint #1.

2. [Candidate recipes B/C + HARD CONSTRAINT #5] B (4-step Lightning LoRA) and C (6-step distill) are described as if they ship "behind an `OTR_WAN_TI2V_*` env." But `_node_candidates()` and `_build_graph()` contain **no LoRA loader node** -- there is no `LoraLoader`/`LoraLoaderModelOnly` between `unet`/`modelsampling` and `ksampler`. Adding a LoRA is a graph change, not a knob. Also `assert_usable` checks only UNET/CLIP/VAE; a set-but-missing LoRA file would fail at runtime, not fail-closed (violates #5's fail-closed intent). FIX: (a) add the LoRA node to `_node_candidates` + wire it in `_build_graph` gated on a LoRA-path env; (b) extend `_aux_loader_files`/`_missing_loaders` to fail-closed when the LoRA env is set but the file is absent.

3. [Candidate recipes B + Q1/Q3] The 4-step distill is not a single knob: it couples steps (4), cfg (~1.0), sampler (euler/lcm), AND the LoRA. Default cfg is 5.0 and default sampler is `uni_pc`; flipping only one env yields a broken render. FIX: ship B/C as a named recipe bundle (one env selects a coordinated {lora,steps,cfg,sampler,shift} set), not four independent envs. [ASSUMPTION] cfg≈1.0 for the distill is from the spec text, not grounding.

4. [HARD CONSTRAINT #2 / Q2 / Q4] Portability of the default path is unverified and the spec leaves it as an open question rather than resolving it before build. Three concrete unknowns block the cross-platform claim: (a) `UnetLoaderGGUF` is a custom node (ComfyUI-GGUF) -- verify: GGUF dequant works on MPS/ROCm/DirectML; (b) the default CLIP `umt5_xxl_fp8_e4m3fn_scaled.safetensors` is fp8 -- verify: fp8 compute/upcast on MPS/AMD; (c) `uni_pc` default sampler -- core, but verify on MPS. The code already has a `safetensors`/`UNETLoader` fallback (`_loader_mode`, `weight_dtype`), so a portable path exists. FIX: pick the portable default explicitly (likely fp16/safetensors UNET + fp16 CLIP + `euler` or `lcm`) and resolve these "verify:" items before locking the default; do not leave constraint #2 as an open question into build.

SHOULD-FIX:

1. [HARD CONSTRAINT #3 / Q3] LightX2V Lightning LoRA license is asserted-must-be Apache/MIT but not verified in grounding. verify: the HF repo's LICENSE for `Wan2.2-Lightning` / `Wan2.2-Distill-Loras`. If non-commercial, B/C are dead on arrival -- this gates the whole roundtable, resolve first.

2. [Q4 / Sources] `MoEKSampler` is referenced as possibly beating KSampler, but `_node_candidates` only offers `KSampler`. Using it is another custom-node graph change. For a "solid floor," explicitly exclude MoEKSampler and `sa_solver` (portability unverified) and standardize on core `euler`/`lcm`.

3. [Current recipe "Defaults"] "length 25 (min 33)" is misleading: `target_fps = 25` is the per-clip fps and the *fallback* frame count; `quantize_frames_4n1(..., min_frames=33)` means the actual minimum rendered length is 33 frames, never 25. State the real floor (33 frames) so the A/B measures the right thing.

4. [Q5] No stated fail-closed behavior for the OOM/ceiling case -- `assert_peak_within_ceiling` raises after a full render is computed (wasted work), not before. Consider a pre-flight estimate or a smaller default frame count for the 8GB tier so the floor degrades predictably rather than rendering then asserting.

OPTIONAL / NICE-TO-HAVE:
- Document the GGUF+LoRA interaction: applying a LoRA onto a quantized GGUF UNET sometimes needs a GGUF-aware LoRA loader; note which loader the bundle uses.
- At cfg 1.0 the negative `CLIPTextEncode` is computed but unused by KSampler -- harmless, but you can drop the neg wire in the distill bundle to save a little.

CUT THESE (over-engineering):
1. Candidate C (6-step distill) -- redundant with B for a "solid floor" decision. If the 4-step bundle is reliable it wins on speed; if it isn't, 6-step shares the same portability/license/graph-change risks. Pick A vs B (vs E control) and drop C to halve the A/B surface. Safe to cut because it adds no distinct reliability property the panel is ranking on.
2. Candidate E's `sa_solver` option -- portability unverified on MPS/AMD; fold E down to the single safe variant (`euler/beta`, shift 3.0) so the control stays cross-platform.